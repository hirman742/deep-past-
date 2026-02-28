from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generation_utils import build_bad_words_ids, build_generate_kwargs, resolve_generation_settings
from metrics_utils import build_metric_signatures, compute_translation_metrics


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
    return [items[i : i + size] for i in range(0, len(items), size)]


def _char_len(text: str) -> int:
    return len(text or "")


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--predict-batch-size", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--shuffle-source", action="store_true")
    ap.add_argument("--num-beams", type=int, default=-1)
    ap.add_argument("--length-penalty", type=float, default=-1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=-1)
    ap.add_argument("--min-new-tokens", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=-1)
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

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
    checkpoint_dir = _resolve_path(args.checkpoint_dir, run_dir / "best_model")

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    suffix_parts: list[str] = []
    if args.tag:
        suffix_parts.append(str(args.tag).strip().replace(" ", "_"))
    if args.shuffle_source:
        suffix_parts.append("srcshuffle")
    suffix = f"_{'_'.join(x for x in suffix_parts if x)}" if suffix_parts else ""

    predictions_out = diag_dir / f"val_predictions_diagnostic{suffix}.csv"
    samples_out = diag_dir / f"val_samples_50{suffix}.csv"
    summary_out = diag_dir / f"val_diagnostic_summary{suffix}.json"

    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    merged = train_df.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    val_df = merged[merged["fold"] == args.fold].reset_index(drop=True)
    if val_df.empty:
        raise ValueError(f"Fold {args.fold} has empty validation split")
    if args.max_rows > 0:
        val_df = val_df.head(int(args.max_rows)).reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Validation subset is empty after --max-rows filtering")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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

    sources = val_df["source"].fillna("").astype(str).tolist()
    references = val_df["target"].fillna("").astype(str).tolist()
    if args.shuffle_source:
        shuffled_sources = sources.copy()
        random.Random(args.seed).shuffle(shuffled_sources)
        sources = shuffled_sources
    predictions: list[str] = []

    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, args.predict_batch_size)):
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
                **generate_kwargs,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([x.strip() for x in decoded])

    if len(predictions) != len(references):
        raise RuntimeError("Prediction/reference length mismatch")

    src_char_lens = [_char_len(x) for x in sources]
    ref_char_lens = [_char_len(x) for x in references]
    pred_char_lens = [_char_len(x) for x in predictions]
    pred_tok_lens = [len(tokenizer.encode(x, add_special_tokens=True)) for x in predictions]
    ref_tok_lens = [len(tokenizer.encode(x, add_special_tokens=True)) for x in references]

    metrics = compute_translation_metrics(predictions=predictions, references=references)

    empty_pred = [idx for idx, text in enumerate(predictions) if not text.strip()]
    copy_source = [idx for idx, (src, pred) in enumerate(zip(sources, predictions)) if src.strip() == pred.strip()]
    top_preds = Counter(predictions).most_common(10)
    normalized_bad_regex = str(generation_settings["bad_tokens_regex"] or "").strip()
    if normalized_bad_regex:
        bad_token_pattern = re.compile(normalized_bad_regex)
        bad_token_mask = [bool(bad_token_pattern.search(x)) for x in predictions]
    else:
        bad_token_mask = [False for _ in predictions]
    exact_extra_id0 = [idx for idx, text in enumerate(predictions) if text.strip() == "<extra_id_0>"]
    shorter_half = [
        idx
        for idx, (p, r) in enumerate(zip(pred_char_lens, ref_char_lens))
        if r > 0 and _safe_div(float(p), float(r)) < 0.5
    ]

    pred_df = pd.DataFrame(
        {
            "oare_id": val_df["oare_id"].astype(str).tolist(),
            "source": sources,
            "reference": references,
            "prediction": predictions,
            "src_char_len": src_char_lens,
            "ref_char_len": ref_char_lens,
            "pred_char_len": pred_char_lens,
            "pred_ref_char_ratio": [
                _safe_div(float(p), float(r)) for p, r in zip(pred_char_lens, ref_char_lens)
            ],
            "pred_tok_len": pred_tok_lens,
            "ref_tok_len": ref_tok_lens,
            "pred_ref_tok_ratio": [_safe_div(float(p), float(r)) for p, r in zip(pred_tok_lens, ref_tok_lens)],
            "is_empty_pred": [idx in set(empty_pred) for idx in range(len(predictions))],
            "is_copy_source": [idx in set(copy_source) for idx in range(len(predictions))],
            "has_bad_token_regex": bad_token_mask,
            "is_exact_extra_id_0": [idx in set(exact_extra_id0) for idx in range(len(predictions))],
        }
    )
    pred_df.to_csv(predictions_out, index=False)

    sample_n = min(max(1, args.sample_size), len(pred_df))
    sample_df = pred_df.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)
    sample_df.to_csv(samples_out, index=False)

    summary = {
        "config_path": str(cfg_path),
        "checkpoint_dir": str(checkpoint_dir),
        "fold": int(args.fold),
        "decode": {
            "num_beams": int(generation_settings["num_beams"]),
            "length_penalty": float(generation_settings["length_penalty"]),
            "no_repeat_ngram_size": int(generation_settings["no_repeat_ngram_size"]),
            "max_new_tokens": int(generation_settings["max_new_tokens"]),
            "min_new_tokens": int(generation_settings["min_new_tokens"]),
            "suppress_extra_ids": bool(generation_settings["suppress_extra_ids"]),
            "bad_tokens_regex": str(generation_settings["bad_tokens_regex"]),
            "suppressed_bad_word_ids_count": int(len(bad_words_ids or [])),
        },
        "diagnostic_mode": {
            "shuffle_source": bool(args.shuffle_source),
            "max_rows": int(args.max_rows),
            "tag": str(args.tag),
        },
        "metrics": {
            "bleu": float(metrics["bleu"]),
            "chrfpp": float(metrics["chrfpp"]),
            "geom": float(metrics["geom"]),
            "bleu_01": float(metrics["bleu_01"]),
            "chrfpp_01": float(metrics["chrfpp_01"]),
            "geom_01": float(metrics["geom_01"]),
            "scales": {
                "bleu": "0-100",
                "chrfpp": "0-100",
                "geom": "sqrt(bleu * chrfpp)",
            },
            "signatures": build_metric_signatures(),
        },
        "output_health": {
            "num_rows": int(len(predictions)),
            "empty_prediction_ratio_pct": 100.0 * _safe_div(float(len(empty_pred)), float(len(predictions))),
            "copy_source_ratio_pct": 100.0 * _safe_div(float(len(copy_source)), float(len(predictions))),
            "pred_shorter_than_half_ref_ratio_pct": 100.0 * _safe_div(float(len(shorter_half)), float(len(predictions))),
            "unique_prediction_ratio_pct": 100.0
            * _safe_div(float(len(set(predictions))), float(len(predictions))),
            "has_bad_token_regex_ratio_pct": 100.0
            * _safe_div(float(sum(1 for x in bad_token_mask if x)), float(len(predictions))),
            "exact_extra_id_0_ratio_pct": 100.0 * _safe_div(float(len(exact_extra_id0)), float(len(predictions))),
            "top_repeated_predictions": [{"text": t, "count": int(c)} for t, c in top_preds],
        },
        "length_stats": {
            "pred_char_mean": float(pd.Series(pred_char_lens).mean()),
            "ref_char_mean": float(pd.Series(ref_char_lens).mean()),
            "pred_char_p95": float(pd.Series(pred_char_lens).quantile(0.95)),
            "pred_char_p99": float(pd.Series(pred_char_lens).quantile(0.99)),
            "ref_char_p95": float(pd.Series(ref_char_lens).quantile(0.95)),
            "ref_char_p99": float(pd.Series(ref_char_lens).quantile(0.99)),
            "pred_tok_mean": float(pd.Series(pred_tok_lens).mean()),
            "ref_tok_mean": float(pd.Series(ref_tok_lens).mean()),
            "pred_tok_p95": float(pd.Series(pred_tok_lens).quantile(0.95)),
            "pred_tok_p99": float(pd.Series(pred_tok_lens).quantile(0.99)),
            "ref_tok_p95": float(pd.Series(ref_tok_lens).quantile(0.95)),
            "ref_tok_p99": float(pd.Series(ref_tok_lens).quantile(0.99)),
        },
        "artifacts": {
            "predictions_csv": str(predictions_out),
            "samples_csv": str(samples_out),
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {predictions_out}")
    print(f"OK: wrote {samples_out}")
    print(f"OK: wrote {summary_out}")
    print(
        "INFO: geom/bleu/chrfpp="
        f"{summary['metrics']['geom']:.4f}/"
        f"{summary['metrics']['bleu']:.4f}/"
        f"{summary['metrics']['chrfpp']:.4f}, "
        f"empty={summary['output_health']['empty_prediction_ratio_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
