import pytest
import torch

from conftest import build_model, korean_texts
from vocab_expansion.embedding_init import expand_embeddings, new_token_decompositions, original_logits
from vocab_expansion.tokenizer_expansion import expand_tokenizer


@pytest.mark.parametrize("tied", [True, False])
def test_new_rows_are_mean_of_parts_and_original_logits_unchanged(english_tokenizer, tmp_path, tied):
    torch.manual_seed(0)
    new_tok, _ = expand_tokenizer(english_tokenizer, korean_texts(300), 40, str(tmp_path / "tok"))
    model = build_model(english_tokenizer, tied)
    reference = model.get_input_embeddings().weight.detach().clone()
    reference_out = model.get_output_embeddings().weight.detach().clone()

    probe = english_tokenizer("The quick brown fox", return_tensors="pt").input_ids
    n_old = len(english_tokenizer)
    before = original_logits(model, probe, n_old)
    meta = expand_embeddings(model, english_tokenizer, new_tok, pad_to_multiple_of=64)
    after = original_logits(model, probe, n_old)

    assert torch.allclose(before, after, atol=1e-5)
    assert meta["embedding_rows"] % 64 == 0 and meta["embedding_rows"] >= len(new_tok)
    assert meta["tied_embeddings"] == tied
    assert model.config.vocab_size == meta["embedding_rows"]

    emb = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight
    assert torch.equal(emb[:n_old], reference[:n_old])
    for idx, parts in new_token_decompositions(english_tokenizer, new_tok).items():
        assert len(parts) >= 2
        torch.testing.assert_close(emb[idx], reference[parts].mean(0))
        torch.testing.assert_close(out[idx], reference_out[parts].mean(0))


def test_new_tokens_can_reuse_spare_rows(english_tokenizer, tmp_path):
    """A model may have more embedding rows than tokens: new tokens then take over the spare rows."""
    new_tok, _ = expand_tokenizer(english_tokenizer, korean_texts(300), 20, str(tmp_path / "tok"))
    rows = len(english_tokenizer) + 100
    model = build_model(english_tokenizer, tied=True, rows=rows)
    meta = expand_embeddings(model, english_tokenizer, new_tok, pad_to_multiple_of=64)
    assert meta["embedding_rows"] == rows  # no resize needed
    assert meta["padding_ids"] == []
    emb = model.get_input_embeddings().weight
    for idx, parts in new_token_decompositions(english_tokenizer, new_tok).items():
        torch.testing.assert_close(emb[idx], emb[parts].mean(0))
