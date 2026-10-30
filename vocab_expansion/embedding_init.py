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


def _check_no_other_vocab_parameters(model, rows: int) -> None:
    """Only the input and output embeddings are resized: fail if another matrix is indexed by token id."""
    embeddings = {model.get_input_embeddings().weight.data_ptr()}
    if model.get_output_embeddings() is not None:
        embeddings.add(model.get_output_embeddings().weight.data_ptr())
    others = [n for n, p in model.named_parameters() if p.dim() and p.shape[0] == rows and p.data_ptr() not in embeddings]
    if others:
        raise ValueError(
            f"These parameters also have one row per token and would not be resized: {others}. "
            "Models with per-layer token embeddings are not supported yet."
        )


@torch.no_grad()
def expand_embeddings(model, old_tokenizer, new_tokenizer, pad_to_multiple_of: int | None = 64, donor: dict | None = None) -> dict:
    """Resize `model` to `new_tokenizer` and initialise the new rows. Returns metadata for training.

    `donor` ({"vocab", "input", "output"}, see `vocab_expansion.donor.donor_rows`): new tokens found in the
    donor vocabulary copy their donor rows instead of the mean of their sub-tokens.
    """
    decompositions = new_token_decompositions(old_tokenizer, new_tokenizer)
    n_old = max(old_tokenizer.get_vocab().values()) + 1
    n_new = max(new_tokenizer.get_vocab().values()) + 1

    old_rows = model.get_input_embeddings().weight.shape[0]
    _check_no_other_vocab_parameters(model, old_rows)
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

    from_donor = {}
    if donor is not None:
        from vocab_expansion.donor import copy_rows

        if donor["input"].shape[1] != input_embeddings.shape[1]:
            raise ValueError(f"Donor embeddings have width {donor['input'].shape[1]}, the model {input_embeddings.shape[1]}")
        tokens = new_tokenizer.convert_ids_to_tokens(list(decompositions))
        from_donor = {idx: donor["vocab"][t] for idx, t in zip(decompositions, tokens) if t in donor["vocab"]}
        copy_rows(input_embeddings, from_donor, donor["input"])

    if not tied:
        _set_rows(output.weight, decompositions, padding_ids, n_old)
        if donor is not None:
            # A donor with tied embeddings has no separate output matrix: its input rows are its output rows
            copy_rows(output.weight, from_donor, donor["output"] if donor["output"] is not None else donor["input"])
        if getattr(output, "bias", None) is not None:
            for idx, parts in decompositions.items():
                output.bias[idx] = output.bias[parts].mean()
            if padding_ids:
                output.bias[padding_ids] = output.bias[:n_old].mean()

    logger.info(
        "Embeddings: %d -> %d rows, %d new tokens (%d copied from the donor), %d padding rows, tied=%s",
        old_rows, rows, len(decompositions), len(from_donor), len(padding_ids), tied,
    )
    return {
        "old_vocab_size": n_old,
        "new_vocab_size": n_new,
        "embedding_rows": rows,
        "new_token_ids": sorted(decompositions),
        "donor_initialized_ids": sorted(from_donor),
        "padding_ids": padding_ids,
        "tied_embeddings": tied,
    }


@torch.no_grad()
def original_logits(model, input_ids: torch.Tensor, n_old: int) -> torch.Tensor:
    """Logits over the original vocabulary, to check that expansion did not change them."""
    model.eval()
    return model(input_ids.to(model.device)).logits[..., :n_old].float().cpu()
