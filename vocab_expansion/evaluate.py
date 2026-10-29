"""Tokenizer-independent evaluation.

Perplexity per token cannot compare models with different tokenizers: the expanded tokenizer
makes fewer, longer tokens. Bits per byte (total negative log-likelihood of a text, divided by
its length in UTF-8 bytes) can.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def bits_per_byte(model, tokenizer, texts: list[str], max_length: int = 2048) -> float:
    """Bits per UTF-8 byte over `texts`. Long texts are scored in windows that overlap by one token."""
    model.eval()
    prefix = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    total_nll, total_bytes = 0.0, 0
    for text in texts:
        ids = prefix + tokenizer.encode(text, add_special_tokens=False)
        # Every token but the first is predicted; with a BOS prefix, that covers the whole text
        for start in range(0, max(len(ids) - 1, 1), max_length - 1):
            window = torch.tensor([ids[start : start + max_length]], device=model.device)
            if window.shape[1] < 2:
                continue
            logits = model(window).logits[0, :-1].float()
            total_nll += torch.nn.functional.cross_entropy(logits, window[0, 1:], reduction="sum").item()
        total_bytes += len(text.encode("utf-8"))
    return total_nll / math.log(2) / total_bytes


def fertility(tokenizer, texts: list[str]) -> dict:
    """Tokens per word and bytes per token. Words are split on whitespace."""
    n_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return {"tokens_per_word": n_tokens / max(n_words, 1), "bytes_per_token": n_bytes / max(n_tokens, 1)}


def evaluate(models: dict[str, tuple], eval_sets: dict[str, list[str]], max_length: int = 2048) -> dict:
    """`models` maps a name to (model, tokenizer). Returns {set: {model: metrics}}."""
    return {
        set_name: {
            name: {"bits_per_byte": bits_per_byte(model, tok, texts, max_length), **fertility(tok, texts)}
            for name, (model, tok) in models.items()
        }
        for set_name, texts in eval_sets.items()
    }
