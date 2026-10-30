"""Restoring a language's tokens from the vocabulary a tokenizer was pruned from."""

import json
import sys

import pytest
import torch
import yaml
from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

import pipeline
from conftest import ENGLISH, build_byte_fallback_tokenizer, build_model, build_tokenizer, korean_texts
from vocab_expansion.donor import donor_agreement, donor_rows, load_donor_embeddings
from vocab_expansion.embedding_init import expand_embeddings
from vocab_expansion.tokenizer_expansion import _is_byte_token, expand_tokenizer, verify_expansion

BILINGUAL = ENGLISH * 20 + korean_texts(300)


def prune(tokenizer, keep_texts):
    """Keep the tokens `keep_texts` need (with the tokens their merges go through), renumbered from 0."""
    tj = json.loads(tokenizer.backend_tokenizer.to_str())
    model = tj["model"]
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]]
    producer = {}
    for a, b in merges:
        producer.setdefault(a + b, (a, b))

    keep = {t for t in model["vocab"] if _is_byte_token(t)} | set(pre_tokenizers.ByteLevel.alphabet())
    stack = [t for text in keep_texts for t in tokenizer.tokenize(text)]
    while stack:
        token = stack.pop()
        if token not in keep:
            keep.add(token)
            stack.extend(producer.get(token, ()))
    kept = sorted((t for t in model["vocab"] if t in keep), key=model["vocab"].get)
    specials = [t["content"] for t in tj["added_tokens"]]
    vocab = {t: i for i, t in enumerate(specials + [t for t in kept if t not in specials])}
    model["vocab"] = vocab
    model["merges"] = [[a, b] for a, b in merges if a in vocab and b in vocab and a + b in vocab]
    for added in tj["added_tokens"]:
        added["id"] = vocab[added["content"]]
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tj)),
        **{k: getattr(tokenizer, k) for k in ("bos_token", "eos_token", "pad_token", "unk_token") if getattr(tokenizer, k)},
    )


@pytest.fixture(params=["byte_level", "byte_fallback"])
def donor_and_pruned(request):
    build = build_tokenizer if request.param == "byte_level" else build_byte_fallback_tokenizer
    donor = build(BILINGUAL, vocab_size=900)
    pruned = prune(donor, ENGLISH)
    assert len(pruned) < len(donor)
    return donor, pruned


def test_restored_tokens_split_korean_like_the_donor(donor_and_pruned, tmp_path):
    donor, pruned = donor_and_pruned
    budget = len(donor) - len(pruned)  # room for everything that was pruned
    new, result = expand_tokenizer(pruned, korean_texts(300), budget, str(tmp_path / "tok"), donor=donor, learn_remaining=False)

    assert result.restored_tokens and result.new_tokens == result.restored_tokens
    donor_vocab = donor.get_vocab()
    assert all(t in donor_vocab for t in result.restored_tokens)  # matched by content, not by id
    ko = korean_texts(50, seed=1)
    assert verify_expansion(pruned, new, ko)["compression_gain"] > 2
    assert verify_expansion(pruned, new, ENGLISH)["compression_gain"] == pytest.approx(1.0)
    assert donor_agreement(new, donor, ko) == 1.0


def test_budget_is_shared_with_learned_merges(donor_and_pruned, tmp_path):
    donor, pruned = donor_and_pruned
    new, result = expand_tokenizer(
        pruned, korean_texts(300), 30, str(tmp_path / "tok"), donor=donor, donor_max_tokens=10, min_frequency=2
    )
    assert 0 < len(result.restored_tokens) <= 10
    assert len(result.new_tokens) == 30  # the rest is learned
    assert len(new) == len(pruned) + 30
    with open(tmp_path / "tok" / "expansion.json") as f:
        info = json.load(f)
    assert info["num_restored_from_donor"] + info["num_learned"] == 30


@pytest.mark.parametrize("tied", [True, False])
def test_restored_tokens_copy_donor_rows(donor_and_pruned, tmp_path, tied):
    donor, pruned = donor_and_pruned
    torch.manual_seed(0)
    donor_model = build_model(donor, tied)
    donor_model.save_pretrained(tmp_path / "donor_model")
    # The pruned model keeps the donor's rows of the tokens it kept
    model = build_model(pruned, tied)
    donor_ids = [donor.convert_tokens_to_ids(t) for t in pruned.convert_ids_to_tokens(list(range(len(pruned))))]
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(donor_model.get_input_embeddings().weight[donor_ids])
        model.get_output_embeddings().weight.copy_(donor_model.get_output_embeddings().weight[donor_ids])

    new, result = expand_tokenizer(pruned, korean_texts(300), 40, str(tmp_path / "tok"), donor=donor)
    embeddings = load_donor_embeddings(str(tmp_path / "donor_model"))
    assert (embeddings["output"] is None) == tied
    meta = expand_embeddings(model, pruned, new, 64, donor_rows(donor, embeddings))

    # Every restored token, and any learned token the donor also has, starts from its donor row
    assert set(new.convert_tokens_to_ids(result.restored_tokens)) <= set(meta["donor_initialized_ids"])
    for token in result.restored_tokens:
        idx, donor_idx = new.convert_tokens_to_ids(token), donor.convert_tokens_to_ids(token)
        torch.testing.assert_close(model.get_input_embeddings().weight[idx], donor_model.get_input_embeddings().weight[donor_idx])
        torch.testing.assert_close(model.get_output_embeddings().weight[idx], donor_model.get_output_embeddings().weight[donor_idx])


def test_other_vocab_sized_parameters_are_refused(english_tokenizer, tmp_path):
    new, _ = expand_tokenizer(english_tokenizer, korean_texts(300), 20, str(tmp_path / "tok"))
    model = build_model(english_tokenizer, tied=True)
    model.model.per_layer_embeddings = torch.nn.Parameter(torch.zeros(len(english_tokenizer), 8))
    with pytest.raises(ValueError, match="per_layer_embeddings"):
        expand_embeddings(model, english_tokenizer, new)


def test_pipeline_with_a_donor(tmp_path):
    donor = build_byte_fallback_tokenizer(BILINGUAL, vocab_size=900)
    pruned = prune(donor, ENGLISH)
    donor_model = build_model(donor, tied=True)
    donor_model.save_pretrained(tmp_path / "donor")
    donor.save_pretrained(tmp_path / "donor")
    build_model(pruned, tied=True).save_pretrained(tmp_path / "base")
    pruned.save_pretrained(tmp_path / "base")
    with open(tmp_path / "ko.jsonl", "w") as f:
        for i in range(200):
            f.write(json.dumps({"text": " ".join(korean_texts(5, seed=i))}, ensure_ascii=False) + "\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "base_model": str(tmp_path / "base"),
        "output_dir": str(tmp_path / "run"),
        "launcher": sys.executable,
        "num_new_tokens": len(donor) - len(pruned),  # room for every pruned token
        "donor": {"tokenizer": str(tmp_path / "donor"), "model": str(tmp_path / "donor")},
        "learn_remaining": False,
        "languages": [{"name": "ko", "data": str(tmp_path / "ko.jsonl")}],
        "eval_docs": 5,
    }))
    pipeline.main(["--config", str(config_path), "--stages", "tokenizer", "init"])

    run = tmp_path / "run"
    report = json.loads((run / "tokenizer" / "verification.json").read_text())
    assert report["ko"]["donor_agreement"] == 1.0
    meta = json.loads((run / "model_init" / "expansion_meta.json").read_text())
    assert meta["donor_initialized_ids"]
    model = AutoModelForCausalLM.from_pretrained(run / "model_init")
    new = PreTrainedTokenizerFast.from_pretrained(run / "tokenizer")
    idx = meta["donor_initialized_ids"][0]
    donor_idx = donor.convert_tokens_to_ids(new.convert_ids_to_tokens(idx))
    torch.testing.assert_close(model.get_input_embeddings().weight[idx], donor_model.get_input_embeddings().weight[donor_idx])
