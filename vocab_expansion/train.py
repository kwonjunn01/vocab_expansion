"""Two-stage adaptation to the expanded tokenizer.

Stage 1 trains the embeddings only (by default only the new rows), with the rest of the model frozen.
Stage 2 is full-model continued pre-training on the same packed mixture.

Launch with `accelerate launch -m vocab_expansion.train --config cfg.yaml --stage 1` (or torchrun).
Stage 1 trains few parameters and fits DDP; use FSDP or DeepSpeed for stage 2 if needed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from vocab_expansion.config import load_config

logger = logging.getLogger(__name__)

STAGE_INPUTS = {1: "model_init", 2: "stage1"}


def collate(rows):
    input_ids = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def train_rows_only(weight: torch.nn.Parameter, rows: list[int]) -> None:
    """Zero the gradient of every row of `weight` except `rows`."""
    mask = torch.zeros(weight.shape[0], 1, dtype=weight.dtype, device=weight.device)
    mask[rows] = 1
    weight.register_hook(lambda grad: grad * mask.to(grad.device, grad.dtype))


def setup_stage1(model, meta: dict, new_rows_only: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    weights = [model.get_input_embeddings().weight]
    if not meta["tied_embeddings"]:
        weights.append(model.get_output_embeddings().weight)
    for weight in weights:
        weight.requires_grad_(True)
        if new_rows_only:
            train_rows_only(weight, meta["new_token_ids"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    out = config["output_dir"]
    stage = config[f"stage{args.stage}"]
    model_dir = os.path.join(out, STAGE_INPUTS[args.stage])
    with open(os.path.join(out, "model_init", "expansion_meta.json")) as f:
        meta = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=config.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=getattr(torch, stage.get("dtype", "float32")),
        trust_remote_code=config.get("trust_remote_code", False),
    )

    training_args = dict(stage["args"])
    if "warmup_ratio" in training_args and "warmup_ratio" not in inspect.signature(TrainingArguments).parameters:
        # Transformers 5 dropped warmup_ratio: a float warmup_steps is a fraction of the total steps
        training_args["warmup_steps"] = float(training_args.pop("warmup_ratio"))
    if args.stage == 1:
        new_rows_only = stage.get("train_new_rows_only", True)
        setup_stage1(model, meta, new_rows_only)
        if new_rows_only and training_args.get("weight_decay", 0.0):
            # Weight decay would still shrink the frozen original rows
            logger.warning("Setting weight_decay to 0: stage 1 must not modify the original embedding rows")
            training_args["weight_decay"] = 0.0
        if new_rows_only:
            logger.info("Stage 1: training %d new embedding rows only", len(meta["new_token_ids"]))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Stage %d: %.1fM trainable parameters", args.stage, n_trainable / 1e6)

    eval_path = os.path.join(out, "data", "eval")
    eval_dataset = load_from_disk(eval_path) if os.path.exists(eval_path) else None
    if eval_dataset is None:
        training_args["eval_strategy"] = "no"

    output_dir = os.path.join(out, f"stage{args.stage}")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=os.path.join(output_dir, "checkpoints"), **training_args),
        train_dataset=load_from_disk(os.path.join(out, "data", "train")),
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=stage.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    metrics = trainer.evaluate() if eval_dataset is not None else {}  # every rank must take part
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
