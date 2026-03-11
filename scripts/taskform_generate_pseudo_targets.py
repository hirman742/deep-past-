#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generation_utils import (
    apply_task_prefix,
    build_bad_words_ids,
    build_generate_kwargs,
    normalize_task_prefix,
    resolve_generation_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--id-col", default="oare_id")
    ap.add_argument("--source-col", default="transliteration")
    ap.add_argument("--predict-batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=-1)
    ap.add_argument("--length-penalty", type=float, default=-1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=-1)
    ap.add_argument("--min-new-tokens", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=-1)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml")
    cfg = _load_yaml(cfg_path)

    checkpoint_dir = _resolve_path(args.checkpoint_dir, REPO_ROOT / "runs" / "missing_checkpoint")
    input_csv = _resolve_path(args.input_csv, REPO_ROOT / "data" / "interim" / "missing.csv")
    output_csv = _resolve_path(args.output_csv, REPO_ROOT / "reports" / "missing.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.get("model", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)

    model_name = str(model_cfg.get("name", "google/byt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 640))
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))

    if args.num_beams > 0:
        generation_settings["num_beams"] = int(args.num_beams)
    if args.length_penalty > 0:
        generation_settings["length_penalty"] = float(args.length_penalty)
    if args.no_repeat_ngram_size >= 0:
        generation_settings["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    if args.min_new_tokens >= 0:
        generation_settings["min_new_tokens"] = int(args.min_new_tokens)
    if args.max_new_tokens >= 0:
        generation_settings["max_new_tokens"] = int(args.max_new_tokens)

    frame = pd.read_csv(input_csv)
    if int(args.limit) > 0:
        frame = frame.head(int(args.limit)).reset_index(drop=True)
    if args.source_col not in frame.columns:
        raise KeyError(f"Missing source column {args.source_col!r} in {input_csv}")
    if args.id_col not in frame.columns:
        frame[args.id_col] = [f"pseudo_{idx:06d}" for idx in range(len(frame))]

    sources_raw = frame[args.source_col].fillna("").astype(str).tolist()
    sources = [apply_task_prefix(text, task_prefix) for text in sources_raw]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )
    generate_kwargs = build_generate_kwargs(
        num_beams=int(generation_settings["num_beams"]),
        length_penalty=float(generation_settings["length_penalty"]),
        max_new_tokens=int(generation_settings["max_new_tokens"]),
        min_new_tokens=int(generation_settings["min_new_tokens"]),
        no_repeat_ngram_size=int(generation_settings["no_repeat_ngram_size"]),
        bad_words_ids=bad_words_ids,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, int(args.predict_batch_size))):
            tokenized = tokenizer(
                batch_sources,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            generated = model.generate(**tokenized, **generate_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([text.strip() for text in decoded])

    out = frame.copy()
    out["source"] = sources
    out["pseudo_target"] = predictions
    out["pseudo_target_word_len"] = out["pseudo_target"].fillna("").astype(str).map(lambda text: len(text.split()))
    out.to_csv(output_csv, index=False)

    print(f"OK: wrote {output_csv}")
    print(
        "INFO: pseudo generation "
        f"rows={len(out)} "
        f"beams={generation_settings['num_beams']} "
        f"lp={generation_settings['length_penalty']} "
        f"max_new={generation_settings['max_new_tokens']}"
    )


if __name__ == "__main__":
    main()
