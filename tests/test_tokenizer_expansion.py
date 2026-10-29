import json

import pytest
from transformers import AutoTokenizer

from conftest import ENGLISH, korean_texts
from vocab_expansion.tokenizer_expansion import count_pretokens, expand_bpe, expand_tokenizer, verify_expansion


def test_original_ids_are_kept_and_new_ids_are_appended(any_tokenizer, tmp_path):
    new, result = expand_tokenizer(any_tokenizer, korean_texts(300), 50, str(tmp_path / "tok"))
    old_vocab, new_vocab = any_tokenizer.get_vocab(), new.get_vocab()
    assert all(new_vocab[t] == i for t, i in old_vocab.items())
    assert len(result.new_tokens) == 50
    new_ids = sorted(new_vocab[t] for t in result.new_tokens)
    assert new_ids == list(range(max(old_vocab.values()) + 1, max(old_vocab.values()) + 51))


def test_inference_matches_training_segmentation(any_tokenizer):
    """The saved merges must reproduce, at inference, the segmentation learned during training."""
    backend = any_tokenizer.backend_tokenizer
    counts = count_pretokens(backend, korean_texts(300))
    result = expand_bpe(json.loads(backend.to_str()), counts, 80, return_segmentations=True)
    from tokenizers import Tokenizer

    expanded = Tokenizer.from_str(json.dumps(result.tokenizer_json)).model
    for word, pieces in result.segmentations.items():
        assert [t.value for t in expanded.tokenize(word)] == pieces


def test_compression_and_round_trip(any_tokenizer, tmp_path):
    new, _ = expand_tokenizer(any_tokenizer, korean_texts(300), 60, str(tmp_path / "tok"))
    korean = verify_expansion(any_tokenizer, new, korean_texts(50, seed=1))
    english = verify_expansion(any_tokenizer, new, ENGLISH)
    assert korean["compression_gain"] > 2
    assert english["compression_gain"] == pytest.approx(1.0)  # Korean merges don't touch English


def test_saved_tokenizer_reloads(any_tokenizer, tmp_path):
    expand_tokenizer(any_tokenizer, korean_texts(300), 30, str(tmp_path / "tok"))
    reloaded = AutoTokenizer.from_pretrained(tmp_path / "tok")
    assert reloaded.eos_token == any_tokenizer.eos_token
    assert len(reloaded) == len(any_tokenizer) + 30
    with open(tmp_path / "tok" / "expansion.json") as f:
        assert json.load(f)["num_new_tokens"] == 30


def test_max_token_length_and_min_frequency(any_tokenizer):
    backend = any_tokenizer.backend_tokenizer
    counts = count_pretokens(backend, korean_texts(300))
    result = expand_bpe(json.loads(backend.to_str()), counts, 200, max_token_length=4)
    assert all(len(t) <= 4 for t in result.new_tokens)
    result = expand_bpe(json.loads(backend.to_str()), counts, 10_000, min_frequency=1_000_000)
    assert result.new_tokens == []


def test_byte_fallback_adds_characters_and_never_merges_byte_tokens(tmp_path):
    from conftest import build_byte_fallback_tokenizer

    old = build_byte_fallback_tokenizer(ENGLISH * 20)
    assert "<0xED>" in old.tokenize("학습")  # Korean is byte-encoded before expansion
    new, result = expand_tokenizer(old, korean_texts(300), 60, str(tmp_path / "tok"))
    assert "습" in result.new_chars
    assert not any("<0x" in t for t in result.new_tokens)
    assert not any("<0x" in t for t in new.tokenize(" ".join(korean_texts(20, seed=1))))
