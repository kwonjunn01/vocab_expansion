"""In-place expansion of a byte-level BPE tokenizer.

New merges are learned by continuing BPE training on top of the *existing* tokenization of a
target-language corpus, and are appended after the original merges. As a result:

* every original token keeps its id,
* every new token is a concatenation of existing tokens (its embedding is initialised from them),
* any text is first tokenized exactly as before, and the new merges are then applied on top.

The last point holds because the original merges are exhausted on every word before a new merge
can fire, and a new merge only creates a token that no original merge refers to.

Two families of BPE are supported:

* byte-level BPE (GPT-2, Llama 3, Qwen, ...): every character is already covered by byte tokens.
* SentencePiece-style BPE with byte fallback (Llama 2, Mistral, ...): characters missing from the
  vocab are encoded as <0xXX> byte tokens. Frequent missing characters are first added as tokens
  of their own (their embedding is the mean of their byte tokens), and merges are then learned
  over characters. Byte tokens are never merged: a merged "<0xEB><0x8B>" would not decode.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

BYTE_TOKEN = re.compile(r"<0x[0-9A-Fa-f]{2}>")
METASPACE = "▁"


def _is_byte_token(token: str) -> bool:
    return BYTE_TOKEN.fullmatch(token) is not None


def _bytes_to_chars(pieces: list[str]) -> list[str]:
    """Replace runs of byte-fallback tokens by the characters they encode."""
    out, buffer = [], bytearray()
    for piece in pieces:
        if _is_byte_token(piece):
            buffer.append(int(piece[3:5], 16))
            continue
        if buffer:
            out.extend(buffer.decode("utf-8", errors="replace"))
            buffer.clear()
        out.append(piece)
    if buffer:
        out.extend(buffer.decode("utf-8", errors="replace"))
    return out


def _chars_to_bytes(pieces: list[str], vocab: dict) -> list[str]:
    """Encode characters that are not in the vocab back to byte-fallback tokens."""
    out = []
    for piece in pieces:
        if piece in vocab:
            out.append(piece)
        else:
            out.extend(f"<0x{b:02X}>" for b in piece.encode("utf-8"))
    return out


@dataclass
class ExpansionResult:
    tokenizer_json: dict
    new_tokens: list[str]  # in id order
    new_merges: list[tuple[str, str]]
    reused_tokens: list[str] = field(default_factory=list)  # merge results that were already in the vocab
    new_chars: list[str] = field(default_factory=list)  # byte-fallback only: characters added as tokens
    segmentations: dict[str, list[str]] | None = None  # final segmentation of each training word


def count_pretokens(backend: Tokenizer, texts: Iterable[str]) -> Counter:
    """Count the pre-tokens ("words") of a corpus, as the tokenizer sees them before BPE.

    SentencePiece-style tokenizers often have no pre-tokenizer and mark spaces with "▁": their text
    is also split before each "▁", so that words are counted, and no new token spans two words.
    """
    normalizer, pre_tokenizer = backend.normalizer, backend.pre_tokenizer
    counts: Counter = Counter()
    for text in texts:
        if normalizer is not None:
            text = normalizer.normalize_str(text)
        pieces = [text] if pre_tokenizer is None else [p for p, _ in pre_tokenizer.pre_tokenize_str(text)]
        for piece in pieces:
            counts.update(w for w in re.split(f"(?={METASPACE})", piece) if w)
    return counts


class _MergeLearner:
    """Incremental BPE pair counting with a lazy max-heap."""

    def __init__(self, words: list[list[str]], freqs: list[int]):
        self.words, self.freqs = words, freqs
        self.pair_counts: Counter = Counter()
        self.where: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                self.pair_counts[pair] += freqs[i]
                self.where[pair].add(i)
        # Ties are broken by the pair itself, so training is deterministic
        self.heap = [(-count, pair) for pair, count in self.pair_counts.items()]
        heapq.heapify(self.heap)

    def pop_best(self, banned: set) -> tuple[tuple[str, str] | None, int]:
        while self.heap:
            neg_count, pair = heapq.heappop(self.heap)
            if pair not in banned and self.pair_counts.get(pair, 0) == -neg_count:
                return pair, -neg_count
        return None, 0

    def merge(self, pair: tuple[str, str]) -> None:
        a, b = pair
        merged = a + b
        changed = set()
        # `where` may hold stale indices of words that no longer contain the pair: recounting them is a no-op
        for i in self.where.pop(pair, ()):
            word, freq = self.words[i], self.freqs[i]
            for p in zip(word, word[1:]):
                self.pair_counts[p] -= freq
                changed.add(p)
            out, j = [], 0
            while j < len(word):
                if j + 1 < len(word) and word[j] == a and word[j + 1] == b:
                    out.append(merged)
                    j += 2
                else:
                    out.append(word[j])
                    j += 1
            self.words[i] = out
            for p in zip(out, out[1:]):
                self.pair_counts[p] += freq
                self.where[p].add(i)
                changed.add(p)
        for p in changed:
            count = self.pair_counts.get(p, 0)
            if count > 0:
                heapq.heappush(self.heap, (-count, p))
            else:
                self.pair_counts.pop(p, None)


def expand_bpe(
    tokenizer_json: dict,
    pretoken_counts: Counter,
    num_new_tokens: int,
    min_frequency: int = 2,
    max_token_length: int | None = None,
    return_segmentations: bool = False,
) -> ExpansionResult:
    """Learn `num_new_tokens` new merges on top of a BPE tokenizer, given the pre-token counts of a corpus."""
    tokenizer_json = json.loads(json.dumps(tokenizer_json))  # deep copy
    model = tokenizer_json["model"]
    if model["type"] != "BPE":
        raise ValueError(f"Only BPE tokenizers can be expanded, got {model['type']}")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise NotImplementedError("BPE with a continuing_subword_prefix or end_of_word_suffix is not supported")

    original = Tokenizer.from_str(json.dumps(tokenizer_json)).model
    vocab: dict[str, int] = model["vocab"]
    added_tokens = tokenizer_json.get("added_tokens", [])
    special_contents = {t["content"] for t in added_tokens}
    next_id = max([*vocab.values(), *(t["id"] for t in added_tokens)]) + 1

    byte_fallback = bool(model.get("byte_fallback"))

    # Segment every word with the original tokenizer: new merges are learned on top of it
    segmented = []
    for word, freq in pretoken_counts.items():
        pieces = [t.value for t in original.tokenize(word)]
        segmented.append((word, freq, _bytes_to_chars(pieces) if byte_fallback else pieces))

    new_tokens, new_merges, reused, banned = [], [], [], set()
    new_chars = []
    if byte_fallback:
        # Frequent characters missing from the vocab become tokens; the others stay as byte tokens
        char_counts: Counter = Counter()
        for _, freq, pieces in segmented:
            for piece in pieces:
                if piece not in vocab:
                    char_counts[piece] += freq
        for char, count in char_counts.most_common():
            if count < min_frequency or len(new_chars) == num_new_tokens:
                break
            if char in special_contents:
                continue
            vocab[char] = next_id
            next_id += 1
            new_chars.append(char)
        new_tokens.extend(new_chars)
        segmented = [(word, freq, _chars_to_bytes(pieces, vocab)) for word, freq, pieces in segmented]
        logger.info("Added %d characters missing from the vocab (%d left as byte tokens)",
                    len(new_chars), len(char_counts) - len(new_chars))

    words, freqs, keys = [], [], []
    for word, freq, pieces in segmented:
        if len(pieces) > 1 or return_segmentations:
            words.append(pieces)
            freqs.append(freq)
            keys.append(word)
    logger.info("Learning merges on %d unique words", len(words))

    learner = _MergeLearner(words, freqs)
    while len(new_tokens) < num_new_tokens:
        pair, count = learner.pop_best(banned)
        if pair is None or count < min_frequency:
            logger.warning("Stopped after %d new tokens: no pair occurs %d+ times", len(new_tokens), min_frequency)
            break
        merged = pair[0] + pair[1]
        if (
            merged in special_contents
            or (max_token_length and len(merged) > max_token_length)
            or (byte_fallback and (_is_byte_token(pair[0]) or _is_byte_token(pair[1])))
        ):
            banned.add(pair)
            continue
        learner.merge(pair)
        new_merges.append(pair)
        if merged in vocab:
            # The token existed but was unreachable through the original merges: the new merge reuses its id
            reused.append(merged)
        else:
            vocab[merged] = next_id
            next_id += 1
            new_tokens.append(merged)

    merges = model["merges"]
    as_strings = bool(merges) and isinstance(merges[0], str)
    merges.extend(f"{a} {b}" if as_strings else [a, b] for a, b in new_merges)

    return ExpansionResult(
        tokenizer_json=tokenizer_json,
        new_tokens=new_tokens,
        new_merges=new_merges,
        reused_tokens=reused,
        new_chars=new_chars,
        segmentations=dict(zip(keys, learner.words)) if return_segmentations else None,
    )


def expand_tokenizer(
    tokenizer,
    texts: Iterable[str],
    num_new_tokens: int,
    output_dir: str,
    min_frequency: int = 2,
    max_token_length: int | None = None,
):
    """Expand a Hugging Face fast tokenizer and save it, with its config and chat template, to `output_dir`."""
    from transformers import AutoTokenizer

    backend = tokenizer.backend_tokenizer
    counts = count_pretokens(backend, texts)
    logger.info("Counted %d unique pre-tokens", len(counts))
    result = expand_bpe(json.loads(backend.to_str()), counts, num_new_tokens, min_frequency, max_token_length)
    logger.info("Added %d new tokens (%d merges reuse existing tokens)", len(result.new_tokens), len(result.reused_tokens))

    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(result.tokenizer_json, f, ensure_ascii=False)
    new_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    with open(os.path.join(output_dir, "expansion.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_new_tokens": len(result.new_tokens),
                "new_token_ids": [new_tokenizer.convert_tokens_to_ids(t) for t in result.new_tokens],
                "num_reused_tokens": len(result.reused_tokens),
                "num_new_chars": len(result.new_chars),
            },
            f,
        )
    return new_tokenizer, result


def verify_expansion(old_tokenizer, new_tokenizer, texts: Iterable[str]) -> dict:
    """Check that every original token keeps its id and that text round-trips; return compression stats."""
    old_vocab, new_vocab = old_tokenizer.get_vocab(), new_tokenizer.get_vocab()
    moved = [t for t, i in old_vocab.items() if new_vocab.get(t) != i]
    if moved:
        raise ValueError(f"{len(moved)} original tokens changed id, e.g. {moved[:5]}")

    n_bytes = old_tokens = new_tokens = 0
    for text in texts:
        old_ids = old_tokenizer.encode(text, add_special_tokens=False)
        new_ids = new_tokenizer.encode(text, add_special_tokens=False)
        if new_tokenizer.decode(new_ids) != old_tokenizer.decode(old_ids):
            raise ValueError(f"Text does not round-trip through the expanded tokenizer: {text[:100]!r}")
        n_bytes += len(text.encode("utf-8"))
        old_tokens += len(old_ids)
        new_tokens += len(new_ids)
    return {
        "bytes": n_bytes,
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "old_bytes_per_token": n_bytes / max(old_tokens, 1),
        "new_bytes_per_token": n_bytes / max(new_tokens, 1),
        "compression_gain": old_tokens / max(new_tokens, 1),
    }
