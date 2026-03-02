from __future__ import annotations

import argparse
import json
import time
from itertools import product
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


def _norm_lp(value: float) -> float:
    return round(float(value), 6)


def _combo_key(
    *,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    max_new_tokens: int,
) -> tuple[int, float, int, int, int]:
    return (
        int(num_beams),
        _norm_lp(length_penalty),
        int(no_repeat_ngram_size),
        int(min_new_tokens),
        int(max_new_tokens),
    )


def _score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("eval_geom", 0.0)),
        float(row.get("eval_bleu", 0.0)),
        float(row.get("eval_chrfpp", 0.0)),
    )


def _concat_chunks(values: list[str]) -> str:
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    return "\n".join(cleaned).strip()


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _filter_original_rows(frame: pd.DataFrame, *, enabled: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame, {
            "aggregate_original_only": bool(enabled),
            "rows_before": 0,
            "rows_after": 0,
            "filtered_rows": 0,
            "applied": False,
            "reason": "empty_frame",
        }
    if not bool(enabled):
        return frame, {
            "aggregate_original_only": False,
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "filtered_rows": 0,
            "applied": False,
            "reason": "disabled",
        }
    markers_present = "is_short_aligned" in frame.columns or "chunk_mode" in frame.columns
    if not markers_present:
        return frame, {
            "aggregate_original_only": True,
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "filtered_rows": 0,
            "applied": False,
            "reason": "no_short_alignment_markers",
        }
    short_mask = pd.Series(False, index=frame.index)
    if "is_short_aligned" in frame.columns:
        short_mask = short_mask | _as_bool_series(frame["is_short_aligned"])
    if "chunk_mode" in frame.columns:
        chunk_mode = frame["chunk_mode"].fillna("").astype(str).str.strip().str.lower()
        short_mask = short_mask | chunk_mode.str.startswith("short_aligned")
    filtered = frame.loc[~short_mask].copy()
    return filtered, {
        "aggregate_original_only": True,
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "filtered_rows": int(short_mask.sum()),
        "applied": True,
    }


def _aggregate_parent_texts(
    *,
    val_df: pd.DataFrame,
    predictions: list[str],
    references: list[str],
) -> tuple[list[str], list[str]] | None:
    if "parent_oare_id" in val_df.columns:
        parent_col = "parent_oare_id"
    elif "parent_id" in val_df.columns:
        parent_col = "parent_id"
    else:
        return None
    work = val_df.copy()
    work["prediction"] = predictions
    work["reference"] = references
    sort_cols = [parent_col]
    if "chunk_index" in work.columns:
        sort_cols.append("chunk_index")
    work = work.sort_values(sort_cols).reset_index(drop=True)
    grouped = work.groupby(parent_col, as_index=False).agg(
        prediction=("prediction", lambda s: _concat_chunks(s.tolist())),
        reference=("reference", lambda s: _concat_chunks(s.tolist())),
    )
    return (
        grouped["prediction"].fillna("").astype(str).tolist(),
        grouped["reference"].fillna("").astype(str).tolist(),
    )


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
    ap.add_argument("--min-new-tokens-list", default="")
    ap.add_argument("--max-new-tokens-list", default="")
    ap.add_argument("--predict-batch-size", type=int, default=32)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--aggregate-by-parent", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--aggregate-original-only", dest="aggregate_original_only", action="store_true")
    ap.add_argument("--no-aggregate-original-only", dest="aggregate_original_only", action="store_false")
    ap.set_defaults(aggregate_original_only=True)
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
    val_df_for_agg, agg_filter_stats = _filter_original_rows(
        val_df,
        enabled=bool(args.aggregate_original_only),
    )

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))
    default_max_new_tokens = int(generation_settings["max_new_tokens"])
    default_min_new_tokens = int(generation_settings["min_new_tokens"])
    if args.min_new_tokens_list.strip():
        min_new_tokens_values = _parse_ints(args.min_new_tokens_list)
    else:
        min_new_tokens_values = [default_min_new_tokens]
    if args.max_new_tokens_list.strip():
        max_new_tokens_values = _parse_ints(args.max_new_tokens_list)
    else:
        max_new_tokens_values = [default_max_new_tokens]
    beams = _parse_ints(args.beams)
    length_penalties = _parse_floats(args.length_penalties)
    ngram_sizes = _parse_ints(args.no_repeat_ngram_sizes)
    if not beams or not length_penalties or not ngram_sizes or not min_new_tokens_values or not max_new_tokens_values:
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
    csv_path = run_dir / "decode_grid_metrics.csv"
    json_path = run_dir / "decode_grid_best.json"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        if not existing_df.empty:
            rows = existing_df.to_dict(orient="records")

    done_keys = {
        _combo_key(
            num_beams=int(row["num_beams"]),
            length_penalty=float(row["length_penalty"]),
            no_repeat_ngram_size=int(row["no_repeat_ngram_size"]),
            min_new_tokens=int(row.get("min_new_tokens", default_min_new_tokens)),
            max_new_tokens=int(row.get("max_new_tokens", default_max_new_tokens)),
        )
        for row in rows
    }

    combos = list(product(beams, length_penalties, ngram_sizes, min_new_tokens_values, max_new_tokens_values))
    total_runs = len(combos)
    start_time = time.time()
    completed_runs = len(done_keys)

    metric_signatures = build_metric_signatures()

    for run_idx, (num_beams, length_penalty, no_repeat_ngram_size, min_new_tokens, max_new_tokens) in enumerate(combos, start=1):
        key = _combo_key(
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
        )
        if key in done_keys:
            print(
                "SKIP:",
                f"[{run_idx}/{total_runs}]",
                f"beams={num_beams}",
                f"lp={length_penalty}",
                f"no_repeat_ngram_size={no_repeat_ngram_size}",
                f"min_new_tokens={min_new_tokens}",
                f"max_new_tokens={max_new_tokens}",
            )
            continue

        print(
            "RUN:",
            f"[{run_idx}/{total_runs}]",
            f"beams={num_beams}",
            f"lp={length_penalty}",
            f"no_repeat_ngram_size={no_repeat_ngram_size}",
            f"min_new_tokens={min_new_tokens}",
            f"max_new_tokens={max_new_tokens}",
        )
        predictions = _generate_texts(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            batch_size=args.predict_batch_size,
            device=device,
        )
        metric_predictions = predictions
        metric_references = references
        aggregate_mode = str(args.aggregate_by_parent).strip().lower()
        do_aggregate = aggregate_mode == "on" or (
            aggregate_mode == "auto" and ("parent_oare_id" in val_df.columns or "parent_id" in val_df.columns)
        )
        if do_aggregate and not val_df_for_agg.empty:
            aggregated = _aggregate_parent_texts(
                val_df=val_df_for_agg,
                predictions=predictions if val_df_for_agg is val_df else [predictions[i] for i in val_df_for_agg.index.tolist()],
                references=references if val_df_for_agg is val_df else [references[i] for i in val_df_for_agg.index.tolist()],
            )
            if aggregated is not None:
                metric_predictions, metric_references = aggregated
        metrics = compute_translation_metrics(predictions=metric_predictions, references=metric_references)
        rows.append(
            {
                "fold": int(args.fold),
                "num_beams": int(num_beams),
                "length_penalty": float(length_penalty),
                "no_repeat_ngram_size": int(no_repeat_ngram_size),
                "min_new_tokens": int(min_new_tokens),
                "max_new_tokens": int(max_new_tokens),
                "eval_bleu": float(metrics["bleu"]),
                "eval_chrfpp": float(metrics["chrfpp"]),
                "eval_geom": float(metrics["geom"]),
                "eval_bleu_01": float(metrics["bleu_01"]),
                "eval_chrfpp_01": float(metrics["chrfpp_01"]),
                "eval_geom_01": float(metrics["geom_01"]),
                "eval_rows": int(len(metric_predictions)),
                "metric_level": "parent_reconstructed" if (do_aggregate and metric_predictions is not predictions) else "chunk_or_sample",
                "aggregate_original_only": bool(args.aggregate_original_only),
                "aggregate_filtered_rows": int(agg_filter_stats.get("filtered_rows", 0)),
            }
        )
        current_row = rows[-1]
        done_keys.add(key)
        completed_runs += 1
        best_so_far = max(rows, key=_score_key)
        best_payload = dict(best_so_far)
        best_payload["metric_signatures"] = metric_signatures
        best_payload["completed_runs"] = int(completed_runs)
        best_payload["total_planned_runs"] = int(total_runs)
        best_payload["progress_pct"] = 100.0 * float(completed_runs) / float(max(1, total_runs))
        elapsed = time.time() - start_time
        best_payload["elapsed_seconds"] = float(elapsed)
        json_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        pd.DataFrame([current_row]).to_csv(
            csv_path,
            mode="a",
            header=not csv_path.exists(),
            index=False,
        )
        print(
            "OK:",
            f"completed={completed_runs}/{total_runs}",
            f"geom={current_row['eval_geom']:.4f}",
            f"best_geom={best_so_far['eval_geom']:.4f}",
        )

    result_df = (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=["num_beams", "length_penalty", "no_repeat_ngram_size", "min_new_tokens", "max_new_tokens"],
            keep="last",
        )
        .sort_values(["eval_geom", "eval_bleu"], ascending=False)
        .reset_index(drop=True)
    )
    result_df.to_csv(csv_path, index=False)

    best = result_df.iloc[0].to_dict()
    best["metric_signatures"] = metric_signatures
    best["completed_runs"] = int(len(result_df))
    best["total_planned_runs"] = int(total_runs)
    best["progress_pct"] = 100.0 * float(len(result_df)) / float(max(1, total_runs))
    best["elapsed_seconds"] = float(time.time() - start_time)
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
