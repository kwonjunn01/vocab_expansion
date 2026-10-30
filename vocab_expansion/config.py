"""Configuration: a short form (languages + data locations + target vocab size) expanded into the full form.

Short form:

    base_model: LiquidAI/LFM3-1.7B
    output_dir: runs/ko
    target_vocab_size: 96000          # or num_new_tokens: 30000
    languages:                        # languages to add
      - name: ko
        data: /data/ko/*.jsonl        # file, glob, directory, or Hugging Face dataset
    replay:                           # original-language data, against forgetting
      - name: en
        data: HuggingFaceFW/fineweb-edu
        subset: sample-10BT
    replay_ratio: 0.25
    train_tokens: 2_000_000_000
    donor:                            # optional: restore the language's tokens from a larger vocabulary
      tokenizer: google/gemma-4-...   # the vocabulary the model's was pruned from
      model: google/gemma-4-...       # optional: copy the donor's embedding rows of restored tokens

Any section of the full form (tokenizer, data, stage1, stage2, eval, ...) can still be given and
overrides what is generated.
"""

from __future__ import annotations

import copy
import glob
import os

import yaml

EXTENSIONS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".csv": "csv", ".txt": "text", ".arrow": "arrow"}

DEFAULTS = {
    "trust_remote_code": True,
    "tokenizer": {"min_frequency": 5, "max_token_length": 32, "pad_to_multiple_of": 64},
    "init": {"dtype": "float32", "max_logit_change": 1.0e-2},
    "data": {"seq_len": 2048, "num_proc": 8},
    "stage1": {
        "dtype": "float32",
        "train_new_rows_only": True,
        "args": {
            "max_steps": 2000,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.0e-3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.02,
            "weight_decay": 0.0,
            "bf16": True,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "save_strategy": "steps",
            "save_steps": 1000,
            "save_total_limit": 1,
            "report_to": "none",
        },
    },
    "stage2": {
        "dtype": "float32",
        "args": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.01,
            "weight_decay": 0.1,
            "bf16": True,
            "gradient_checkpointing": True,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 1000,
            "save_strategy": "steps",
            "save_steps": 2000,
            "save_total_limit": 2,
            "report_to": "none",
        },
    },
    "eval": {"max_length": 2048},
}


def deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` into a copy of `base`. Nested dicts are merged, everything else is replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def dataset_kwargs(entry: dict) -> dict:
    """`datasets.load_dataset` keyword arguments for a short-form language or replay entry."""
    data = entry["data"]
    if isinstance(data, dict):  # already load_dataset kwargs
        return data
    data = os.path.expanduser(str(data))
    streaming = entry.get("streaming", True)
    if data.startswith(("/", ".")) and not glob.glob(data, recursive=True):
        raise FileNotFoundError(f"No data found at {data} (for {entry.get('name')})")
    if os.path.isdir(data):
        for ext, builder in EXTENSIONS.items():
            if glob.glob(os.path.join(data, "**", f"*{ext}"), recursive=True):
                return {"path": builder, "data_files": os.path.join(data, "**", f"*{ext}"), "split": "train", "streaming": streaming}
        raise ValueError(f"No {', '.join(EXTENSIONS)} files in {data}")
    ext = os.path.splitext(data.replace("*", "x"))[1].lower()  # "/x/*.jsonl" -> ".jsonl"
    if ext in EXTENSIONS:
        return {"path": EXTENSIONS[ext], "data_files": data, "split": "train", "streaming": streaming}
    if os.path.exists(data):
        raise ValueError(f"Unsupported file type: {data} (supported: {', '.join(EXTENSIONS)})")
    # A Hugging Face dataset name
    kwargs = {"path": data, "split": entry.get("split", "train"), "streaming": streaming}
    if entry.get("subset"):
        kwargs["name"] = entry["subset"]
    return kwargs


def _from_short_form(raw: dict) -> dict:
    languages, replay = raw["languages"], raw.get("replay", [])
    for entry in languages + replay:
        if "name" not in entry or "data" not in entry:
            raise ValueError(f"Every language and replay entry needs a name and data: {entry}")
    replay_ratio = raw.get("replay_ratio", 0.25 if replay else 0.0)
    eval_docs = raw.get("eval_docs", 200)
    tokenizer_docs = raw.get("tokenizer_docs_per_language", 200_000)

    def sources(entries, share):
        total = sum(e.get("weight", 1.0) for e in entries)
        return [
            {
                "name": e["name"],
                "dataset": dataset_kwargs(e),
                "text_field": e.get("text_field", "text"),
                "max_docs": e.get("max_docs"),
                "eval_docs": eval_docs,
                "weight": share * e.get("weight", 1.0) / total,
            }
            for e in entries
        ]

    data_sources = sources(languages, 1.0 - replay_ratio) + (sources(replay, replay_ratio) if replay else [])
    return {
        "tokenizer": {
            # The same number of documents per language, so that every language gets its share of merges.
            # Held-out eval documents are skipped.
            "corpus": [
                {"dataset": s["dataset"], "text_field": s["text_field"], "max_docs": tokenizer_docs, "skip_docs": eval_docs}
                for s in data_sources[: len(languages)]
            ],
        },
        "data": {"train_tokens": raw.get("train_tokens"), "sources": data_sources},
        # The first documents of each source are the held-out ones
        "eval": {
            "sets": [
                {"name": s["name"], "dataset": s["dataset"], "text_field": s["text_field"], "max_docs": min(100, eval_docs)}
                for s in data_sources
            ]
        },
    }


def resolve_config(raw: dict) -> dict:
    """Defaults, then what the short form generates, then every explicit section of `raw`."""
    config = copy.deepcopy(DEFAULTS)
    if "languages" in raw:
        config = deep_merge(config, _from_short_form(raw))
    size = {k: raw[k] for k in ("target_vocab_size", "num_new_tokens", "donor", "learn_remaining") if k in raw}
    config = deep_merge(config, {"tokenizer": size})
    config = deep_merge(config, {k: v for k, v in raw.items() if k not in SHORT_FORM_KEYS})
    tokenizer = config["tokenizer"]
    if not tokenizer.get("target_vocab_size") and not tokenizer.get("num_new_tokens"):
        raise ValueError("Set target_vocab_size (final vocabulary size) or num_new_tokens")
    for key in ("base_model", "output_dir"):
        if key not in config:
            raise ValueError(f"Missing {key}")
    return config


SHORT_FORM_KEYS = {
    "languages", "replay", "replay_ratio", "eval_docs", "tokenizer_docs_per_language",
    "target_vocab_size", "num_new_tokens", "train_tokens", "donor", "learn_remaining",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return resolve_config(yaml.safe_load(f))


def num_new_tokens(tokenizer_config: dict, old_vocab_size: int) -> int:
    """New tokens to add: from num_new_tokens, or from the target size of the final vocabulary."""
    if tokenizer_config.get("num_new_tokens"):
        return int(tokenizer_config["num_new_tokens"])
    target = int(tokenizer_config["target_vocab_size"])
    if target <= old_vocab_size:
        raise ValueError(f"target_vocab_size {target} is not larger than the current vocabulary ({old_vocab_size})")
    return target - old_vocab_size
