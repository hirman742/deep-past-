from __future__ import annotations

import argparse
import json
import math
import re
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
DIGIT_PATTERN = re.compile(r"\d+")


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


def _chunk(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _concat_chunks(values: list[str]) -> str:
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    return "\n".join(cleaned).strip()


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


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
    frame: pd.DataFrame,
    predictions: list[str],
    references: list[str],
) -> tuple[list[str], list[str], int] | None:
    if "parent_oare_id" in frame.columns:
        parent_col = "parent_oare_id"
    elif "parent_id" in frame.columns:
        parent_col = "parent_id"
    else:
        return None

    work = frame.copy()
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
        int(len(grouped)),
    )


def _extract_digit_set(text: str) -> set[str]:
    return set(DIGIT_PATTERN.findall(text or ""))


def _digit_match_score(source: str, prediction: str) -> float:
    src_digits = _extract_digit_set(source)
    if not src_digits:
        return 1.0
    pred_digits = _extract_digit_set(prediction)
    return _safe_div(float(len(src_digits & pred_digits)), float(len(src_digits)))


def _repeat_penalty(text: str, *, n: int = 3) -> float:
    tokens = [tok for tok in (text or "").split(" ") if tok]
    if len(tokens) <= n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    repeated = len(ngrams) - len(set(ngrams))
    return _safe_div(float(repeated), float(len(ngrams)))


def _length_outlier_penalty(
    *,
    prediction_tok_len: int,
    source_tok_len: int,
    ratio_lo: float,
    ratio_hi: float,
) -> float:
    ratio = _safe_div(float(prediction_tok_len), float(max(1, source_tok_len)))
    if ratio_lo <= ratio <= ratio_hi:
        return 0.0
    band = max(1e-6, ratio_hi - ratio_lo)
    if ratio < ratio_lo:
        return float((ratio_lo - ratio) / band)
    return float((ratio - ratio_hi) / band)


def _normalize_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [0.5 for _ in values]
    return [float((x - min_v) / (max_v - min_v)) for x in values]


def _prepare_rerank_ratio_band(frame: pd.DataFrame, tokenizer) -> tuple[float, float]:
    src = frame["source"].fillna("").astype(str).tolist()
    tgt = frame["target"].fillna("").astype(str).tolist()
    src_lens = [len(tokenizer.encode(x, add_special_tokens=True)) for x in src]
    tgt_lens = [len(tokenizer.encode(x, add_special_tokens=True)) for x in tgt]
    ratios = [
        _safe_div(float(t), float(max(1, s)))
        for s, t in zip(src_lens, tgt_lens)
        if int(s) > 0 and int(t) > 0
    ]
    if not ratios:
        return 0.25, 2.5
    series = pd.Series(ratios)
    lo = float(series.quantile(0.05))
    hi = float(series.quantile(0.95))
    if hi <= lo:
        lo, hi = 0.25, 2.5
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/byt5_small_lora_chunked_stage1_r8_qv.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--tag", default="nbest_rerank")
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--predict-batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--num-return-sequences", type=int, default=4)
    ap.add_argument("--length-penalty", type=float, default=1.2)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--weight-model", type=float, default=1.0)
    ap.add_argument("--weight-digit", type=float, default=0.8)
    ap.add_argument("--weight-repeat", type=float, default=0.6)
    ap.add_argument("--weight-length", type=float, default=0.5)
    ap.add_argument("--aggregate-original-only", dest="aggregate_original_only", action="store_true")
    ap.add_argument("--no-aggregate-original-only", dest="aggregate_original_only", action="store_false")
    ap.set_defaults(aggregate_original_only=True)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "byt5_small_lora_chunked_stage1_r8_qv.yaml")
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
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_dir}")

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(args.tag or "nbest_rerank").strip().replace(" ", "_")
    summary_out = diag_dir / f"nbest_rerank_summary_{suffix}.json"
    row_out = diag_dir / f"nbest_rerank_rowwise_{suffix}.csv"
    cand_out = diag_dir / f"nbest_rerank_candidates_{suffix}.csv"
    recon_out = diag_dir / f"nbest_rerank_reconstructed_{suffix}.csv"

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

    model_name = str(model_cfg.get("name", "google/byt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 384))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_split = merged[merged["fold"] != args.fold].reset_index(drop=True)
    ratio_lo, ratio_hi = _prepare_rerank_ratio_band(train_split, tokenizer)

    generation_settings["num_beams"] = int(args.num_beams)
    generation_settings["length_penalty"] = float(args.length_penalty)
    generation_settings["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    generation_settings["min_new_tokens"] = int(args.min_new_tokens)
    generation_settings["max_new_tokens"] = int(args.max_new_tokens)
    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )

    generate_kwargs = build_generate_kwargs(
        num_beams=int(max(args.num_beams, args.num_return_sequences)),
        length_penalty=float(args.length_penalty),
        max_new_tokens=int(args.max_new_tokens),
        min_new_tokens=int(args.min_new_tokens),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        bad_words_ids=bad_words_ids,
    )
    generate_kwargs["num_return_sequences"] = int(args.num_return_sequences)
    generate_kwargs["return_dict_in_generate"] = True
    generate_kwargs["output_scores"] = True

    sources = val_df["source"].fillna("").astype(str).tolist()
    references = val_df["target"].fillna("").astype(str).tolist()
    source_tok_lens = [len(tokenizer.encode(x, add_special_tokens=True)) for x in sources]

    selected_predictions: list[str] = []
    top1_predictions: list[str] = []
    row_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    cursor = 0
    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, int(args.predict_batch_size))):
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
            decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            seq_scores = generated.sequences_scores
            if seq_scores is not None:
                seq_scores_list = [float(x) for x in seq_scores.detach().cpu().tolist()]
            else:
                seq_scores_list = [0.0 for _ in decoded]

            bsz = len(batch_sources)
            nbest = int(args.num_return_sequences)
            for i in range(bsz):
                row_idx = cursor + i
                src_text = sources[row_idx]
                src_tok = source_tok_lens[row_idx]
                ref_text = references[row_idx]
                candidate_block: list[dict[str, Any]] = []
                for k in range(nbest):
                    idx = i * nbest + k
                    pred_text = decoded[idx].strip()
                    pred_tok = len(tokenizer.encode(pred_text, add_special_tokens=True))
                    model_score = float(seq_scores_list[idx])
                    digit_score = _digit_match_score(src_text, pred_text)
                    repeat_penalty = _repeat_penalty(pred_text, n=3)
                    length_penalty = _length_outlier_penalty(
                        prediction_tok_len=pred_tok,
                        source_tok_len=src_tok,
                        ratio_lo=ratio_lo,
                        ratio_hi=ratio_hi,
                    )
                    candidate_block.append(
                        {
                            "row_index": int(row_idx),
                            "candidate_rank": int(k),
                            "prediction": pred_text,
                            "model_score": model_score,
                            "digit_match": float(digit_score),
                            "repeat_penalty": float(repeat_penalty),
                            "length_outlier_penalty": float(length_penalty),
                            "pred_tok_len": int(pred_tok),
                        }
                    )

                model_norms = _normalize_minmax([x["model_score"] for x in candidate_block])
                for idx_cand, cand in enumerate(candidate_block):
                    cand["model_score_norm"] = float(model_norms[idx_cand])
                    cand["rerank_score"] = float(
                        float(args.weight_model) * cand["model_score_norm"]
                        + float(args.weight_digit) * cand["digit_match"]
                        - float(args.weight_repeat) * cand["repeat_penalty"]
                        - float(args.weight_length) * cand["length_outlier_penalty"]
                    )

                top1 = candidate_block[0]
                best = max(candidate_block, key=lambda x: float(x["rerank_score"]))
                top1_predictions.append(top1["prediction"])
                selected_predictions.append(best["prediction"])

                row_records.append(
                    {
                        "row_index": int(row_idx),
                        "oare_id": str(val_df.iloc[row_idx]["oare_id"]),
                        "parent_oare_id": str(val_df.iloc[row_idx].get("parent_oare_id", "")),
                        "chunk_index": int(val_df.iloc[row_idx].get("chunk_index", 0)),
                        "source": src_text,
                        "reference": ref_text,
                        "top1_prediction": top1["prediction"],
                        "rerank_prediction": best["prediction"],
                        "top1_model_score": float(top1["model_score"]),
                        "rerank_model_score": float(best["model_score"]),
                        "rerank_score": float(best["rerank_score"]),
                        "source_tok_len": int(src_tok),
                    }
                )
                candidate_records.extend(candidate_block)
            cursor += bsz

    if len(selected_predictions) != len(references):
        raise RuntimeError("Prediction/reference size mismatch in rerank output")

    top1_metrics = compute_translation_metrics(predictions=top1_predictions, references=references)
    rerank_metrics = compute_translation_metrics(predictions=selected_predictions, references=references)

    row_df = pd.DataFrame(row_records)
    row_df.to_csv(row_out, index=False)
    pd.DataFrame(candidate_records).to_csv(cand_out, index=False)

    aggregate_input = val_df.copy()
    aggregate_input["prediction"] = selected_predictions
    aggregate_input["reference"] = references
    aggregate_input["source"] = sources
    agg_df, agg_stats = _filter_original_rows(aggregate_input, enabled=bool(args.aggregate_original_only))
    reconstructed = _aggregate_parent_texts(
        frame=agg_df,
        predictions=agg_df["prediction"].fillna("").astype(str).tolist(),
        references=agg_df["reference"].fillna("").astype(str).tolist(),
    )
    reconstructed_summary: dict[str, Any] | None = None
    if reconstructed is not None:
        rec_predictions, rec_references, rec_rows = reconstructed
        rec_metrics = compute_translation_metrics(predictions=rec_predictions, references=rec_references)
        if "parent_oare_id" in agg_df.columns:
            parent_col = "parent_oare_id"
        elif "parent_id" in agg_df.columns:
            parent_col = "parent_id"
        else:
            parent_col = "oare_id"
        rec_df = (
            agg_df.sort_values([parent_col, "chunk_index"] if "chunk_index" in agg_df.columns else [parent_col])
            .groupby(parent_col, as_index=False)
            .agg(
                prediction=("prediction", lambda s: _concat_chunks(s.tolist())),
                reference=("reference", lambda s: _concat_chunks(s.tolist())),
            )
            .rename(columns={parent_col: "oare_id"})
        )
        rec_df.to_csv(recon_out, index=False)
        reconstructed_summary = {
            "num_rows": int(rec_rows),
            "metrics": {k: float(v) for k, v in rec_metrics.items()},
            "filter": agg_stats,
            "artifact": str(recon_out),
        }

    summary = {
        "config_path": str(cfg_path),
        "checkpoint_dir": str(checkpoint_dir),
        "fold": int(args.fold),
        "rows": int(len(val_df)),
        "decode": {
            "num_beams": int(args.num_beams),
            "num_return_sequences": int(args.num_return_sequences),
            "length_penalty": float(args.length_penalty),
            "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
            "min_new_tokens": int(args.min_new_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "suppress_extra_ids": bool(generation_settings["suppress_extra_ids"]),
            "bad_tokens_regex": str(generation_settings["bad_tokens_regex"]),
            "suppressed_bad_word_ids_count": int(len(bad_words_ids or [])),
        },
        "rerank": {
            "weights": {
                "model": float(args.weight_model),
                "digit": float(args.weight_digit),
                "repeat": float(args.weight_repeat),
                "length": float(args.weight_length),
            },
            "ratio_band": {"lo": float(ratio_lo), "hi": float(ratio_hi)},
            "aggregate_original_only": bool(args.aggregate_original_only),
        },
        "metrics": {
            "top1": {k: float(v) for k, v in top1_metrics.items()},
            "rerank": {k: float(v) for k, v in rerank_metrics.items()},
            "delta_rerank_minus_top1": {
                k: float(rerank_metrics[k] - top1_metrics[k]) for k in top1_metrics.keys()
            },
            "signatures": build_metric_signatures(),
        },
        "artifacts": {
            "rowwise_csv": str(row_out),
            "candidates_csv": str(cand_out),
        },
    }
    if reconstructed_summary is not None:
        summary["reconstructed"] = reconstructed_summary

    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {row_out}")
    print(f"OK: wrote {cand_out}")
    if reconstructed_summary is not None:
        print(f"OK: wrote {recon_out}")
    print(f"OK: wrote {summary_out}")
    print(
        "INFO: top1 geom/bleu/chrfpp="
        f"{top1_metrics['geom']:.4f}/{top1_metrics['bleu']:.4f}/{top1_metrics['chrfpp']:.4f}"
    )
    print(
        "INFO: rerank geom/bleu/chrfpp="
        f"{rerank_metrics['geom']:.4f}/{rerank_metrics['bleu']:.4f}/{rerank_metrics['chrfpp']:.4f}"
    )
    if reconstructed_summary is not None and "metrics" in reconstructed_summary:
        metrics = reconstructed_summary["metrics"]
        print(
            "INFO: reconstructed rerank geom/bleu/chrfpp="
            f"{metrics['geom']:.4f}/{metrics['bleu']:.4f}/{metrics['chrfpp']:.4f}"
        )


if __name__ == "__main__":
    main()
