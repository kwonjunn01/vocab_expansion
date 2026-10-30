"""Vocabulary expansion pipeline: one config, one command.

    python pipeline.py --config configs/korean.yaml                   # every stage
    python pipeline.py --config configs/korean.yaml --print-config    # show the full config
    python pipeline.py --config configs/korean.yaml --stages stage2   # one stage
    python pipeline.py --config configs/korean.yaml --from stage1     # resume

Stages: tokenizer -> init -> data -> stage1 -> stage2 -> evaluate.
A finished stage writes a _DONE marker and is skipped next time, unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys

import torch
import yaml

from vocab_expansion.config import load_config, num_new_tokens
from vocab_expansion.data import iter_texts, prepare_data
from vocab_expansion.donor import donor_agreement, donor_rows, load_donor_embeddings
from vocab_expansion.embedding_init import expand_embeddings, original_logits
from vocab_expansion.evaluate import evaluate
from vocab_expansion.tokenizer_expansion import expand_tokenizer, verify_expansion

logger = logging.getLogger("pipeline")

STAGES = ["tokenizer", "init", "data", "stage1", "stage2", "evaluate"]
OUTPUTS = {"tokenizer": "tokenizer", "init": "model_init", "data": "data", "stage1": "stage1", "stage2": "stage2", "evaluate": "eval"}


def load_tokenizer(path, config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=config.get("trust_remote_code", False))


def load_model(path, config, dtype="float32"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, dtype), trust_remote_code=config.get("trust_remote_code", False)
    )


def corpus_texts(sources: list[dict]):
    for source in sources:
        yield from iter_texts(
            source["dataset"], source.get("text_field", "text"), source.get("max_docs"), source.get("skip_docs", 0)
        )


def eval_texts(config) -> dict[str, list[str]]:
    sets = config.get("eval", {}).get("sets", [])
    return {s["name"]: list(iter_texts(s["dataset"], s.get("text_field", "text"), s.get("max_docs", 100))) for s in sets}


def run_tokenizer(config, out_dir):
    cfg = config["tokenizer"]
    old = load_tokenizer(config["base_model"], config)
    old_size = max(old.get_vocab().values()) + 1
    n_new = num_new_tokens(cfg, old_size)
    logger.info("Vocabulary: %d -> %d tokens (%d new)", old_size, old_size + n_new, n_new)
    donor_cfg = cfg.get("donor") or {}
    donor = load_tokenizer(donor_cfg["tokenizer"], config) if donor_cfg.get("tokenizer") else None
    new, _ = expand_tokenizer(
        old,
        lambda: corpus_texts(cfg["corpus"]),
        num_new_tokens=n_new,
        output_dir=out_dir,
        min_frequency=cfg.get("min_frequency", 2),
        max_token_length=cfg.get("max_token_length"),
        donor=donor,
        donor_max_tokens=donor_cfg.get("max_tokens"),
        learn_remaining=cfg.get("learn_remaining", True),
    )
    report = {}
    for name, texts in eval_texts(config).items():
        report[name] = verify_expansion(old, new, texts)
        if donor is not None:
            # 1.0: the expanded tokenizer splits these texts exactly like the donor
            report[name]["donor_agreement"] = donor_agreement(new, donor, texts)
    with open(os.path.join(out_dir, "verification.json"), "w") as f:
        json.dump(report, f, indent=2)
    for name, stats in report.items():
        logger.info("%s: %.2fx fewer tokens (%.2f -> %.2f bytes/token)", name, stats["compression_gain"],
                    stats["old_bytes_per_token"], stats["new_bytes_per_token"])


def run_init(config, out_dir):
    old_tok = load_tokenizer(config["base_model"], config)
    new_tok = load_tokenizer(os.path.join(config["output_dir"], "tokenizer"), config)
    model = load_model(config["base_model"], config, config.get("init", {}).get("dtype", "float32"))
    donor_cfg = config["tokenizer"].get("donor") or {}
    donor = None
    if donor_cfg.get("model"):
        embeddings = load_donor_embeddings(donor_cfg["model"], donor_cfg.get("embedding_key"), donor_cfg.get("output_key"))
        donor = donor_rows(load_tokenizer(donor_cfg.get("tokenizer", donor_cfg["model"]), config), embeddings)

    # The original logits of text tokenized with the original tokenizer must not change
    probe = old_tok("Vocabulary expansion must not change what the model already knows.", return_tensors="pt").input_ids
    n_old = max(old_tok.get_vocab().values()) + 1
    before = original_logits(model, probe, n_old)
    meta = expand_embeddings(model, old_tok, new_tok, config["tokenizer"].get("pad_to_multiple_of", 64), donor)
    diff = (original_logits(model, probe, n_old) - before).abs().max().item()
    logger.info("Max change of original logits after expansion: %.2e", diff)
    if diff > config.get("init", {}).get("max_logit_change", 1e-2):
        raise RuntimeError(f"Expansion changed the original logits by {diff}")

    model.save_pretrained(out_dir)
    new_tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "expansion_meta.json"), "w") as f:
        json.dump(meta, f)


def run_data(config, out_dir):
    prepare_data(config, load_tokenizer(os.path.join(config["output_dir"], "tokenizer"), config), out_dir)


def run_training(config, stage, config_path):
    launcher = shlex.split(config.get("launcher", f"{sys.executable}"))
    cmd = launcher + ["-m", "vocab_expansion.train", "--config", config_path, "--stage", stage[-1]]
    logger.info("Running: %s", shlex.join(cmd))
    repo = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [repo, os.environ.get("PYTHONPATH")])))
    subprocess.run(cmd, check=True, env=env)


def run_evaluate(config, out_dir):
    texts = eval_texts(config)
    if not texts:
        logger.warning("No eval sets in the config")
        return
    dtype = config.get("eval", {}).get("dtype", "bfloat16" if torch.cuda.is_available() else "float32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = {"original": config["base_model"]}
    for stage in ["model_init", "stage1", "stage2"]:
        path = os.path.join(config["output_dir"], stage)
        if os.path.exists(os.path.join(path, "config.json")):
            candidates[stage] = path
    results = {}
    for name, path in candidates.items():
        # One model in memory at a time
        model = load_model(path, config, dtype).to(device)
        scores = evaluate({name: (model, load_tokenizer(path, config))}, texts, config.get("eval", {}).get("max_length", 2048))
        for set_name, by_model in scores.items():
            results.setdefault(set_name, {}).update(by_model)
        del model
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for set_name, by_model in results.items():
        for name, m in by_model.items():
            logger.info("[%s] %-10s bits/byte %.4f  tokens/word %.2f", set_name, name, m["bits_per_byte"], m["tokens_per_word"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="stages to run (default: all)")
    parser.add_argument("--from", dest="start", choices=STAGES, help="run this stage and every later one")
    parser.add_argument("--force", action="store_true", help="rerun stages that are already done")
    parser.add_argument("--print-config", action="store_true", help="print the full config and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = load_config(args.config)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return
    stages = args.stages or STAGES[STAGES.index(args.start) if args.start else 0 :]

    for stage in STAGES:
        if stage not in stages:
            continue
        out_dir = os.path.join(config["output_dir"], OUTPUTS[stage])
        done = os.path.join(out_dir, "_DONE")
        if os.path.exists(done) and not args.force:
            logger.info("Skipping %s: already done (%s)", stage, out_dir)
            continue
        logger.info("=== %s ===", stage)
        os.makedirs(out_dir, exist_ok=True)
        if stage in ("stage1", "stage2"):
            run_training(config, stage, args.config)
        else:
            {"tokenizer": run_tokenizer, "init": run_init, "data": run_data, "evaluate": run_evaluate}[stage](config, out_dir)
        open(done, "w").close()


if __name__ == "__main__":
    main()
