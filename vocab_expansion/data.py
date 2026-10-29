"""Corpus loading, tokenization and packing into fixed-length rows."""

from __future__ import annotations

import logging
import math
from itertools import islice
from typing import Iterator

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


def iter_texts(dataset: dict, text_field: str = "text", max_docs: int | None = None, skip_docs: int = 0) -> Iterator[str]:
    """Yield texts from a source. `dataset` holds the keyword arguments of `datasets.load_dataset`."""
    rows = load_dataset(**dataset)
    for row in islice(rows, skip_docs, None if max_docs is None else skip_docs + max_docs):
        text = row[text_field]
        if text:
            yield text


def _docs(dataset: dict, text_field: str, max_docs: int | None, skip_docs: int):
    for text in iter_texts(dataset, text_field, max_docs, skip_docs):
        yield {"text": text}


def tokenize_and_pack(texts: Dataset, tokenizer, seq_len: int, num_proc: int | None = None) -> Dataset:
    """Tokenize documents, end each one with EOS, concatenate them and cut the stream into rows of `seq_len` tokens."""
    eos = [tokenizer.eos_token_id]

    def tokenize(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": [x + eos for x in ids]}

    def pack(batch):
        stream = [t for ids in batch["input_ids"] for t in ids]
        n = len(stream) // seq_len * seq_len  # the remainder of each batch is dropped
        return {"input_ids": [stream[i : i + seq_len] for i in range(0, n, seq_len)]}

    tokenized = texts.map(tokenize, batched=True, remove_columns=texts.column_names, num_proc=num_proc)
    return tokenized.map(pack, batched=True, batch_size=1_000, num_proc=num_proc)


def build_source(source: dict, tokenizer, seq_len: int, eval_docs: int, num_proc: int | None) -> tuple[Dataset, Dataset | None]:
    """Packed train and eval rows of one source. The first `eval_docs` documents are held out for evaluation."""
    kwargs = dict(dataset=source["dataset"], text_field=source.get("text_field", "text"))
    train = Dataset.from_generator(_docs, gen_kwargs=dict(kwargs, max_docs=source.get("max_docs"), skip_docs=eval_docs))
    train = tokenize_and_pack(train, tokenizer, seq_len, num_proc)
    held_out = None
    if eval_docs:
        held_out = Dataset.from_generator(_docs, gen_kwargs=dict(kwargs, max_docs=eval_docs, skip_docs=0))
        held_out = tokenize_and_pack(held_out, tokenizer, seq_len, num_proc)
    return train, held_out


def mix(datasets: dict[str, Dataset], weights: dict[str, float], total_rows: int | None, seed: int = 0) -> Dataset:
    """Mix sources by token share. Without `total_rows`, every row of every source is used once and weights are ignored."""
    if total_rows is None:
        return concatenate_datasets(list(datasets.values())).shuffle(seed=seed)
    total_weight = sum(weights.values())
    parts = []
    for name, ds in datasets.items():
        wanted = round(total_rows * weights[name] / total_weight)
        if len(ds) == 0 or wanted == 0:
            continue
        repeats = math.ceil(wanted / len(ds))
        if repeats > 1:
            logger.warning("Source %s has %d rows for %d wanted: repeating it %d times", name, len(ds), wanted, repeats)
        full = concatenate_datasets([ds.shuffle(seed=seed + r) for r in range(repeats)])
        parts.append(full.select(range(wanted)))
        logger.info("Source %s: %d rows (%.1f%%)", name, wanted, 100 * wanted / total_rows)
    return concatenate_datasets(parts).shuffle(seed=seed)


def prepare_data(config: dict, tokenizer, output_dir: str) -> None:
    """Build the packed training mixture and the held-out eval rows, and save them under `output_dir`."""
    data = config["data"]
    seq_len, num_proc = data["seq_len"], data.get("num_proc")
    train_sets, eval_sets, weights = {}, [], {}
    for source in data["sources"]:
        train, held_out = build_source(source, tokenizer, seq_len, source.get("eval_docs", 0), num_proc)
        logger.info("Source %s: %d train rows (%.1fM tokens)", source["name"], len(train), len(train) * seq_len / 1e6)
        train_sets[source["name"]] = train
        weights[source["name"]] = source.get("weight", 1.0)
        if held_out is not None:
            eval_sets.append(held_out)

    total_tokens = data.get("train_tokens")
    total_rows = None if total_tokens is None else int(total_tokens) // seq_len
    mix(train_sets, weights, total_rows, data.get("seed", 0)).save_to_disk(f"{output_dir}/train")
    if eval_sets:
        concatenate_datasets(eval_sets).save_to_disk(f"{output_dir}/eval")
