import json
import sys

import pytest
import yaml

import pipeline
from conftest import ENGLISH, build_model, korean_texts
from vocab_expansion.config import dataset_kwargs, num_new_tokens, resolve_config


def test_dataset_kwargs_from_locations(tmp_path):
    (tmp_path / "ko").mkdir()
    (tmp_path / "ko" / "a.jsonl").write_text("{}")
    assert dataset_kwargs({"data": str(tmp_path / "ko")})["path"] == "json"
    (tmp_path / "p.parquet").write_text("")
    (tmp_path / "books.txt").write_text("")
    pattern = str(tmp_path / "*.parquet")
    assert dataset_kwargs({"data": pattern}) == {"path": "parquet", "data_files": pattern, "split": "train", "streaming": True}
    assert dataset_kwargs({"data": str(tmp_path / "books.txt")})["path"] == "text"
    hub = dataset_kwargs({"data": "HuggingFaceFW/fineweb-2", "subset": "kor_Hang"})
    assert hub == {"path": "HuggingFaceFW/fineweb-2", "split": "train", "streaming": True, "name": "kor_Hang"}
    raw = {"path": "json", "data_files": "x.jsonl", "split": "train"}
    assert dataset_kwargs({"data": raw}) is raw


def test_short_form_mixture_and_overrides():
    config = resolve_config({
        "base_model": "m",
        "output_dir": "o",
        "target_vocab_size": 1000,
        "languages": [{"name": "ko", "data": "a/*.jsonl"}, {"name": "ja", "data": "b/*.jsonl", "weight": 3}],
        "replay": [{"name": "en", "data": "c/*.jsonl"}],
        "replay_ratio": 0.2,
        "stage2": {"args": {"learning_rate": 1.0e-5}},
    })
    weights = {s["name"]: s["weight"] for s in config["data"]["sources"]}
    assert weights == pytest.approx({"ko": 0.2, "ja": 0.6, "en": 0.2})
    assert [c["dataset"]["data_files"] for c in config["tokenizer"]["corpus"]] == ["a/*.jsonl", "b/*.jsonl"]  # no replay
    assert [s["name"] for s in config["eval"]["sets"]] == ["ko", "ja", "en"]
    assert config["tokenizer"]["target_vocab_size"] == 1000
    assert config["stage2"]["args"]["learning_rate"] == 1.0e-5
    assert config["stage2"]["args"]["num_train_epochs"] == 1  # defaults are kept


def test_size_is_required_and_must_grow():
    with pytest.raises(ValueError, match="target_vocab_size"):
        resolve_config({"base_model": "m", "output_dir": "o", "languages": [{"name": "ko", "data": "a.jsonl"}]})
    assert num_new_tokens({"target_vocab_size": 1200}, 1000) == 200
    assert num_new_tokens({"num_new_tokens": 50}, 1000) == 50
    with pytest.raises(ValueError, match="not larger"):
        num_new_tokens({"target_vocab_size": 900}, 1000)


def test_short_form_pipeline_reaches_target_size(english_tokenizer, tmp_path):
    base = tmp_path / "base"
    build_model(english_tokenizer, tied=True).save_pretrained(base)
    english_tokenizer.save_pretrained(base)
    (tmp_path / "ko").mkdir()
    with open(tmp_path / "ko" / "part0.jsonl", "w") as f:
        for i in range(200):
            f.write(json.dumps({"text": " ".join(korean_texts(5, seed=i))}, ensure_ascii=False) + "\n")
    with open(tmp_path / "en.txt", "w") as f:
        f.write("\n".join(ENGLISH * 30))

    target = len(english_tokenizer) + 40
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "base_model": str(base),
        "output_dir": str(tmp_path / "run"),
        "launcher": sys.executable,
        "target_vocab_size": target,
        "languages": [{"name": "ko", "data": str(tmp_path / "ko")}],
        "replay": [{"name": "en", "data": str(tmp_path / "en.txt")}],
        "eval_docs": 5,
        "train_tokens": 64 * 20,
        "data": {"seq_len": 64, "num_proc": None},
    }))
    pipeline.main(["--config", str(config_path), "--stages", "tokenizer", "init", "data"])

    meta = json.loads((tmp_path / "run" / "model_init" / "expansion_meta.json").read_text())
    assert meta["new_vocab_size"] == target
    assert meta["embedding_rows"] % 64 == 0
    report = json.loads((tmp_path / "run" / "tokenizer" / "verification.json").read_text())
    assert report["ko"]["compression_gain"] > 1.5
    assert report["en"]["compression_gain"] == pytest.approx(1.0)


def test_missing_local_data_fails_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_kwargs({"name": "ko", "data": str(tmp_path / "nope")})
    with pytest.raises(FileNotFoundError):
        dataset_kwargs({"name": "ko", "data": str(tmp_path / "*.jsonl")})
