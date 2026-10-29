"""Resize a model to an expanded tokenizer and initialise the new embedding rows.

Each new token is a concatenation of original tokens, so its input (and, if untied, output)
embedding is the mean of the rows of those tokens. Original rows are left untouched, so the
logits of original tokens are unchanged for text tokenized with the original tokenizer.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def new_token_decompositions(old_tokenizer, new_tokenizer) -> dict[int, list[int]]:
    """Map each new token id to the ids of the original tokens it is made of."""
    old_vocab = old_tokenizer.get_vocab()
    old_model = old_tokenizer.backend_tokenizer.model
    return {
        idx: [old_vocab[t.value] for t in old_model.tokenize(token)]
        for token, idx in new_tokenizer.get_vocab().items()
        if token not in old_vocab
    }


def _set_rows(weight: torch.Tensor, decompositions: dict[int, list[int]], padding_ids: list[int], n_old: int):
    for idx, parts in decompositions.items():
        weight[idx] = weight[parts].float().mean(0).to(weight.dtype)
    if padding_ids:
        weight[padding_ids] = weight[:n_old].float().mean(0).to(weight.dtype)


@torch.no_grad()
def expand_embeddings(model, old_tokenizer, new_tokenizer, pad_to_multiple_of: int | None = 64) -> dict:
    """Resize `model` to `new_tokenizer` and initialise the new rows. Returns metadata for training."""
    decompositions = new_token_decompositions(old_tokenizer, new_tokenizer)
    n_old = max(old_tokenizer.get_vocab().values()) + 1
    n_new = max(new_tokenizer.get_vocab().values()) + 1

    old_rows = model.get_input_embeddings().weight.shape[0]
    rows = n_new if not pad_to_multiple_of else math.ceil(n_new / pad_to_multiple_of) * pad_to_multiple_of
    rows = max(rows, old_rows)
    if rows > old_rows:
        model.resize_token_embeddings(rows, mean_resizing=False)
    # Rows created by the resize that no token uses: give them a harmless value instead of a random one
    padding_ids = list(range(max(old_rows, n_new), rows))

    input_embeddings = model.get_input_embeddings().weight
    _set_rows(input_embeddings, decompositions, padding_ids, n_old)

    output = model.get_output_embeddings()
    tied = output is None or output.weight.data_ptr() == input_embeddings.data_ptr()
    if not tied:
        _set_rows(output.weight, decompositions, padding_ids, n_old)
        if getattr(output, "bias", None) is not None:
            for idx, parts in decompositions.items():
                output.bias[idx] = output.bias[parts].mean()
            if padding_ids:
                output.bias[padding_ids] = output.bias[:n_old].mean()

    logger.info(
        "Embeddings: %d -> %d rows, %d new tokens, %d padding rows, tied=%s",
        old_rows, rows, len(decompositions), len(padding_ids), tied,
    )
    return {
        "old_vocab_size": n_old,
        "new_vocab_size": n_new,
        "embedding_rows": rows,
        "new_token_ids": sorted(decompositions),
        "padding_ids": padding_ids,
        "tied_embeddings": tied,
    }


@torch.no_grad()
def original_logits(model, input_ids: torch.Tensor, n_old: int) -> torch.Tensor:
    """Logits over the original vocabulary, to check that expansion did not change them."""
    model.eval()
    return model(input_ids.to(model.device)).logits[..., :n_old].float().cpu()
