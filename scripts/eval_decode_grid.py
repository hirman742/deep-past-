from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generation_utils import build_bad_words_ids, build_generate_kwargs, resolve_generation_settings


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _generate_texts(
    *,
    model,
    tokenizer,
    sources: list[str],
    max_source_length: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    bad_words_ids: list[list[int]] | None,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    out: list[str] = []
    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, batch_size)):
            tokenized = tokenizer(
                batch_sources,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            generated = model.generate(
                **tokenized,
                **build_generate_kwargs(
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                ),
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            out.extend([x.strip() for x in decoded])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--beams", default="4,5,8")
    ap.add_argument("--length-penalties", default="0.8,1.0,1.2,1.4")
    ap.add_argument("--no-repeat-ngram-sizes", default="0")
    ap.add_argument("--predict-batch-size", type=int, default=32)
    ap.add_argument("--max-val-samples", type=int, default=0)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_path = processed_dir / "train_proc.csv"
    folds_path = processed_dir / "folds.csv"
    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    run_dir = run_root.parent / f"{run_root.name}_fold{args.fold}"
    checkpoint_dir = run_dir / "best_model"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")

    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    merged = train_df.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    val_df = merged[merged["fold"] == args.fold].reset_index(drop=True)
    if args.max_val_samples > 0:
        val_df = val_df.iloc[: args.max_val_samples].copy()
    if val_df.empty:
        raise ValueError(f"Fold {args.fold} has no validation rows")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))
    default_max_new_tokens = int(generation_settings["max_new_tokens"])
    default_min_new_tokens = int(generation_settings["min_new_tokens"])
    beams = _parse_ints(args.beams)
    length_penalties = _parse_floats(args.length_penalties)
    ngram_sizes = _parse_ints(args.no_repeat_ngram_sizes)
    if not beams or not length_penalties or not ngram_sizes:
        raise ValueError("decode grid lists cannot be empty")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sources = val_df["source"].fillna("").astype(str).tolist()
    references = val_df["target"].fillna("").astype(str).tolist()
    rows: list[dict[str, Any]] = []

    for num_beams, length_penalty, no_repeat_ngram_size in product(beams, length_penalties, ngram_sizes):
        print(
            "RUN:",
            f"beams={num_beams}",
            f"lp={length_penalty}",
            f"no_repeat_ngram_size={no_repeat_ngram_size}",
        )
        predictions = _generate_texts(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            max_source_length=max_source_length,
            max_new_tokens=default_max_new_tokens,
            min_new_tokens=default_min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            batch_size=args.predict_batch_size,
            device=device,
        )
        bleu = sacrebleu.corpus_bleu(predictions, [references]).score
        chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score
        bleu_01 = float(bleu) / 100.0
        chrfpp_01 = float(chrfpp) / 100.0
        geom = math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))
        geom_01 = math.sqrt(max(bleu_01, 0.0) * max(chrfpp_01, 0.0))
        rows.append(
            {
                "fold": int(args.fold),
                "num_beams": int(num_beams),
                "length_penalty": float(length_penalty),
                "no_repeat_ngram_size": int(no_repeat_ngram_size),
                "eval_bleu": float(bleu),
                "eval_chrfpp": float(chrfpp),
                "eval_geom": float(geom),
                "eval_bleu_01": float(bleu_01),
                "eval_chrfpp_01": float(chrfpp_01),
                "eval_geom_01": float(geom_01),
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["eval_geom", "eval_bleu"], ascending=False).reset_index(drop=True)
    csv_path = run_dir / "decode_grid_metrics.csv"
    json_path = run_dir / "decode_grid_best.json"
    result_df.to_csv(csv_path, index=False)

    best = result_df.iloc[0].to_dict()
    json_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {csv_path}")
    print(f"OK: wrote {json_path}")
    print(
        "INFO: best geom/bleu/chrfpp="
        f"{best['eval_geom']:.4f}/"
        f"{best['eval_bleu']:.4f}/"
        f"{best['eval_chrfpp']:.4f}"
    )


if __name__ == "__main__":
    main()
