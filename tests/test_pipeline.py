"""End-to-end run of every stage on a tiny model, on CPU."""

import json
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM

import pipeline
from conftest import ENGLISH, build_model, korean_texts

TRAIN_ARGS = {
    "max_steps": 3,
    "per_device_train_batch_size": 2,
    "learning_rate": 1.0e-2,
    "logging_steps": 1,
    "save_strategy": "no",
    "eval_strategy": "steps",
    "eval_steps": 3,
    "report_to": "none",
    "use_cpu": True,
}


def write_jsonl(path, texts):
    with open(path, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def test_full_pipeline(english_tokenizer, tmp_path):
    base = tmp_path / "base"
    build_model(english_tokenizer, tied=False).save_pretrained(base)
    english_tokenizer.save_pretrained(base)
    write_jsonl(tmp_path / "ko.jsonl", [" ".join(korean_texts(5, seed=i)) for i in range(200)])
    write_jsonl(tmp_path / "en.jsonl", [" ".join(ENGLISH)] * 100)
    ko = {"path": "json", "data_files": str(tmp_path / "ko.jsonl"), "split": "train"}
    en = {"path": "json", "data_files": str(tmp_path / "en.jsonl"), "split": "train"}

    config = {
        "base_model": str(base),
        "output_dir": str(tmp_path / "run"),
        "launcher": sys.executable,
        "tokenizer": {"num_new_tokens": 40, "pad_to_multiple_of": 64, "corpus": [{"dataset": ko, "max_docs": 100}]},
        "data": {
            "seq_len": 64,
            "train_tokens": 64 * 40,
            "sources": [
                {"name": "ko", "dataset": ko, "eval_docs": 10, "weight": 0.75},
                {"name": "en", "dataset": en, "eval_docs": 5, "weight": 0.25},
            ],
        },
        "stage1": {"args": dict(TRAIN_ARGS, weight_decay=0.1)},
        "stage2": {"args": TRAIN_ARGS},
        "eval": {"max_length": 64, "sets": [{"name": "ko", "dataset": ko, "max_docs": 5}, {"name": "en", "dataset": en, "max_docs": 5}]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    pipeline.main(["--config", str(config_path)])

    run = tmp_path / "run"
    meta = json.loads((run / "model_init" / "expansion_meta.json").read_text())
    new_ids = meta["new_token_ids"]
    n_old = meta["old_vocab_size"]

    init = AutoModelForCausalLM.from_pretrained(run / "model_init")
    stage1 = AutoModelForCausalLM.from_pretrained(run / "stage1")
    stage2 = AutoModelForCausalLM.from_pretrained(run / "stage2")
    for get in ("get_input_embeddings", "get_output_embeddings"):
        w0, w1 = getattr(init, get)().weight, getattr(stage1, get)().weight
        assert torch.equal(w0[:n_old], w1[:n_old]), "stage 1 changed original rows"
        assert not torch.equal(w0[new_ids], w1[new_ids]), "stage 1 did not train new rows"
    body = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(init.state_dict()[body], stage1.state_dict()[body]), "stage 1 changed the body"
    assert not torch.equal(stage1.state_dict()[body], stage2.state_dict()[body]), "stage 2 did not train the body"

    results = json.loads((run / "eval" / "results.json").read_text())
    assert set(results["ko"]) == {"original", "model_init", "stage1", "stage2"}
    assert results["ko"]["stage2"]["tokens_per_word"] < results["ko"]["original"]["tokens_per_word"]

    # Finished stages are skipped on a second run
    pipeline.main(["--config", str(config_path)])
