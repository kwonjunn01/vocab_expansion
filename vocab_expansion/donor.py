"""Restore tokens from a donor vocabulary, by content.

Typical case: the model's tokenizer was pruned from a larger one (e.g. Gemma 4's), and a new
language should get back the tokens the donor had for it. Tokens are matched by their string,
never by the donor's ids: restored tokens get new ids after the current vocabulary.

* Which tokens: the donor tokenizes the target-language corpus, and the most frequent donor
  tokens missing from the current vocabulary are restored first.
* How they become reachable: the donor merges that build them (and the intermediate tokens
  they need) are added. When the current merges are a subsequence of the donor's (the usual
  result of pruning), every merge is put back at its donor position, so the restored language is
  split exactly as the donor splits it. Otherwise they are appended after the current merges.
* Embeddings: when the donor's weights are given, a restored token copies its donor row,
  which was trained, instead of the mean of its sub-tokens.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch

from vocab_expansion.tokenizer_expansion import _is_byte_token

logger = logging.getLogger(__name__)


@dataclass
class RestoreResult:
    tokenizer_json: dict
    restored_tokens: list[str]  # in id order, intermediate tokens included
    new_merges: list[tuple[str, str]]
    unreachable: int  # frequent donor tokens no donor merge can build


def _pairs(merges: list) -> list[tuple[str, str]]:
    return [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges]


def check_compatible(tokenizer_json: dict, donor_json: dict) -> None:
    model, donor = tokenizer_json["model"], donor_json["model"]
    if model["type"] != "BPE" or donor["type"] != "BPE":
        raise ValueError("Both the tokenizer and the donor must be BPE tokenizers")
    if bool(model.get("byte_fallback")) != bool(donor.get("byte_fallback")):
        raise ValueError("The tokenizer and the donor disagree on byte_fallback")
    for part in ("normalizer", "pre_tokenizer"):
        if tokenizer_json.get(part) != donor_json.get(part):
            logger.warning("The tokenizer and the donor have different %ss: restored tokens may not match", part)


def count_donor_tokens(donor_tokenizer, texts: Iterable[str], batch_size: int = 1_000) -> Counter:
    """How often each donor token occurs when the donor tokenizes `texts`."""
    backend = donor_tokenizer.backend_tokenizer
    counts: Counter = Counter()
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            for encoding in backend.encode_batch(batch, add_special_tokens=False):
                counts.update(encoding.tokens)
            batch = []
    for encoding in backend.encode_batch(batch, add_special_tokens=False):
        counts.update(encoding.tokens)
    return counts


def restore_donor_tokens(
    tokenizer_json: dict,
    donor_json: dict,
    donor_counts: Counter,
    max_tokens: int,
    min_frequency: int = 2,
) -> RestoreResult:
    """Add up to `max_tokens` donor tokens (most frequent first) with the donor merges that build them."""
    check_compatible(tokenizer_json, donor_json)
    tokenizer_json = json.loads(json.dumps(tokenizer_json))  # deep copy
    model = tokenizer_json["model"]
    vocab: dict[str, int] = model["vocab"]
    added_tokens = tokenizer_json.get("added_tokens", [])
    special = {t["content"] for t in added_tokens} | {t["content"] for t in donor_json.get("added_tokens", [])}
    next_id = max([*vocab.values(), *(t["id"] for t in added_tokens)]) + 1

    donor_merges = _pairs(donor_json["model"]["merges"])
    producer: dict[str, tuple[str, str]] = {}
    for a, b in donor_merges:
        producer.setdefault(a + b, (a, b))  # the highest-priority merge that builds a token

    planned: list[str] = []
    planned_set: set[str] = set()

    def build(token: str, out: list[str]) -> bool:
        """Append to `out` the tokens to add so that `token` can be built. False if it cannot."""
        if token in vocab or token in planned_set or token in out:
            return True
        if token in special or _is_byte_token(token):
            return False
        if token in producer:
            a, b = producer[token]
            if not (build(a, out) and build(b, out)):
                return False
        elif len(token) != 1:  # only single characters are base symbols
            return False
        out.append(token)
        return True

    unreachable = 0
    for token, count in donor_counts.most_common():
        if count < min_frequency or len(planned) >= max_tokens:
            break
        if token in vocab or token in planned_set:
            continue
        out: list[str] = []
        if not build(token, out):
            unreachable += 1
            continue
        if len(planned) + len(out) <= max_tokens:
            planned.extend(out)
            planned_set.update(out)

    for token in planned:
        vocab[token] = next_id
        next_id += 1

    # Every donor merge that involves a restored token and whose parts now exist
    current = _pairs(model["merges"])
    existing = set(current)
    new_merges = [
        (a, b)
        for a, b in donor_merges
        if (a + b in planned_set or a in planned_set or b in planned_set)
        and a in vocab and b in vocab and a + b in vocab and (a, b) not in existing
    ]
    donor_rank: dict[tuple[str, str], int] = {}
    for rank, pair in enumerate(donor_merges):
        donor_rank.setdefault(pair, rank)
    ranks = [donor_rank.get(pair) for pair in current]
    if all(r is not None for r in ranks) and all(x < y for x, y in zip(ranks, ranks[1:])):
        ordered = sorted(current + new_merges, key=donor_rank.__getitem__)
    else:
        logger.warning("The current merges are not in the donor's order: restored merges are appended after them")
        ordered = current + new_merges
    as_strings = bool(model["merges"]) and isinstance(model["merges"][0], str)
    model["merges"] = [f"{a} {b}" if as_strings else [a, b] for a, b in ordered]

    logger.info("Restored %d donor tokens with %d merges (%d frequent donor tokens unreachable)",
                len(planned), len(new_merges), unreachable)
    return RestoreResult(tokenizer_json, planned, new_merges, unreachable)


def donor_agreement(tokenizer, donor_tokenizer, texts: Iterable[str]) -> float:
    """Share of the donor's tokens that `tokenizer` also produces, at the same place in the text.

    1.0 means the texts are split exactly as the donor splits them.
    """
    matched = total = 0
    for text in texts:
        ours = set(tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).offsets)
        theirs = donor_tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).offsets
        matched += sum(span in ours for span in theirs)
        total += len(theirs)
    return matched / max(total, 1)


def _weight_map(path: str) -> tuple[dict[str, str], callable]:
    """Map tensor names to safetensors files for a local directory or a Hugging Face model."""
    if os.path.isdir(path):
        local = lambda name: os.path.join(path, name)  # noqa: E731
        has_index = os.path.exists(local("model.safetensors.index.json"))
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        local = lambda name: hf_hub_download(path, name)  # noqa: E731
        try:
            local("model.safetensors.index.json")
            has_index = True
        except EntryNotFoundError:
            has_index = False
    if has_index:
        with open(local("model.safetensors.index.json")) as f:
            return json.load(f)["weight_map"], local
    from safetensors import safe_open

    with safe_open(local("model.safetensors"), "pt") as f:
        return {key: "model.safetensors" for key in f.keys()}, local


def load_donor_embeddings(path: str, embedding_key: str | None = None, output_key: str | None = None) -> dict:
    """Read only the donor's input (and untied output) embedding matrices from its safetensors files."""
    from safetensors import safe_open

    weight_map, local = _weight_map(path)
    if embedding_key is None:
        candidates = [k for k in weight_map if k.endswith("embed_tokens.weight")]
        if not candidates:
            raise ValueError(f"No embed_tokens.weight in {path}: set donor.embedding_key")
        # Multimodal checkpoints (e.g. Gemma) keep the text embeddings under language_model
        candidates.sort(key=lambda k: ("language_model" not in k, len(k)))
        embedding_key = candidates[0]
    if output_key is None:
        output_key = next((k for k in weight_map if k.endswith("lm_head.weight")), None)

    def read(key):
        with safe_open(local(weight_map[key]), "pt") as f:
            return f.get_tensor(key)

    logger.info("Donor embeddings: %s%s", embedding_key, f", output: {output_key}" if output_key else " (tied)")
    return {"input": read(embedding_key), "output": read(output_key) if output_key else None}


def donor_rows(donor_tokenizer, embeddings: dict) -> dict:
    """What `expand_embeddings` needs to copy donor rows: the donor vocab and its matrices."""
    return {"vocab": donor_tokenizer.get_vocab(), **embeddings}


def copy_rows(weight: torch.Tensor, rows: dict[int, int], source: torch.Tensor) -> None:
    """weight[new_id] = source[donor_id] for every pair in `rows`."""
    if not rows:
        return
    ids = torch.tensor(list(rows))
    weight[ids] = source[torch.tensor(list(rows.values()))].to(weight.device, weight.dtype)
