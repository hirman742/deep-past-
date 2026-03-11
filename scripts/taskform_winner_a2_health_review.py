from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from taskform_phase12_common import (
    build_health,
    clamp_consecutive_repeated_spans,
    collapse_formula_loops,
    compute_translation_metrics,
    formula_count,
    internal_repeat_score,
    repair_gap_markers,
    resolve_path,
    word_count,
    write_json,
    write_text,
)
from taskform_winner_a2_retrieval_eval import _compare_output_health


REPO_ROOT = Path(__file__).resolve().parents[1]
HINT_RE = re.compile(r"Retrieved English hint:\s*(.*)", flags=re.DOTALL)
BAD_TOKEN_RE = re.compile(r"<extra_id_0>|<unk>", flags=re.IGNORECASE)


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    needed = {"source", "reference", "prediction"}
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    if "parent_oare_id" not in frame.columns:
        raise KeyError(f"Missing parent_oare_id in {path}")
    if "chunk_index" not in frame.columns:
        raise KeyError(f"Missing chunk_index in {path}")
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].fillna("").astype(str)
    return out


def _extract_hint(source: str) -> str:
    match = HINT_RE.search(source or "")
    return match.group(1).strip() if match else ""


def _aggregate_parent_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    ordered = pred_df.sort_values(["parent_oare_id", "chunk_index"]).reset_index(drop=True)
    grouped = (
        ordered.groupby("parent_oare_id", as_index=False)
        .agg(
            source=("source", lambda s: "\n".join(item.strip() for item in s.astype(str) if item.strip())),
            reference=("reference", lambda s: "\n".join(item.strip() for item in s.astype(str) if item.strip())),
            prediction=("prediction", lambda s: "\n".join(item.strip() for item in s.astype(str) if item.strip())),
        )
        .rename(columns={"parent_oare_id": "id"})
    )
    return grouped


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _filter_original_chunks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    markers_present = "is_short_aligned" in frame.columns or "chunk_mode" in frame.columns
    if not markers_present:
        return frame.copy()
    short_mask = pd.Series(False, index=frame.index)
    if "is_short_aligned" in frame.columns:
        short_mask = short_mask | _as_bool_series(frame["is_short_aligned"])
    if "chunk_mode" in frame.columns:
        chunk_mode = frame["chunk_mode"].fillna("").astype(str).str.strip().str.lower()
        short_mask = short_mask | chunk_mode.str.startswith("short_aligned")
    return frame.loc[~short_mask].copy().reset_index(drop=True)


def _diagnostic_health(frame: pd.DataFrame) -> dict[str, Any]:
    predictions = frame["prediction"].fillna("").astype(str).tolist()
    references = frame["reference"].fillna("").astype(str).tolist()
    sources = frame["source"].fillna("").astype(str).tolist()
    pred_counts = Counter(predictions)
    empty = 0
    copy_source = 0
    shorter_half = 0
    bad_token = 0
    extra_id_0 = 0
    for src, ref, pred in zip(sources, references, predictions):
        if not pred.strip():
            empty += 1
        if pred.strip() == src.strip():
            copy_source += 1
        if len(pred) < max(1.0, len(ref) / 2.0):
            shorter_half += 1
        if BAD_TOKEN_RE.search(pred):
            bad_token += 1
        if pred.strip() == "<extra_id_0>":
            extra_id_0 += 1
    total = max(1, len(predictions))
    repeated = [
        {"text": text, "count": count}
        for text, count in pred_counts.most_common(10)
        if text.strip() and count > 1
    ]
    return {
        "num_rows": int(len(predictions)),
        "empty_prediction_ratio_pct": round(100.0 * float(empty) / float(total), 4),
        "copy_source_ratio_pct": round(100.0 * float(copy_source) / float(total), 4),
        "pred_shorter_than_half_ref_ratio_pct": round(100.0 * float(shorter_half) / float(total), 4),
        "unique_prediction_ratio_pct": round(100.0 * float(pd.Series(predictions).nunique()) / float(total), 4),
        "has_bad_token_regex_ratio_pct": round(100.0 * float(bad_token) / float(total), 4),
        "exact_extra_id_0_ratio_pct": round(100.0 * float(extra_id_0) / float(total), 4),
        "top_repeated_predictions": repeated,
    }


def _evaluate_variant(frame: pd.DataFrame) -> dict[str, Any]:
    predictions = frame["prediction"].fillna("").astype(str).tolist()
    references = frame["reference"].fillna("").astype(str).tolist()
    chunk_metrics = compute_translation_metrics(predictions=predictions, references=references)
    chunk_health = _diagnostic_health(frame)

    reconstructed = _aggregate_parent_df(_filter_original_chunks(frame))
    rec_predictions = reconstructed["prediction"].fillna("").astype(str).tolist()
    rec_references = reconstructed["reference"].fillna("").astype(str).tolist()
    rec_metrics = compute_translation_metrics(predictions=rec_predictions, references=rec_references)
    rec_health = _diagnostic_health(reconstructed)
    rec_aux_health = build_health(rec_predictions, rec_references)
    rec_health["aux_repeat_prediction_ratio_pct"] = float(rec_aux_health["repeat_prediction_ratio_pct"])
    rec_health["aux_internal_repeat_trigram_ratio_pct"] = float(rec_aux_health["internal_repeat_trigram_ratio_pct"])
    rec_health["aux_pred_word_mean"] = float(rec_aux_health["pred_word_mean"])
    rec_health["aux_ref_word_mean"] = float(rec_aux_health["ref_word_mean"])
    return {
        "chunk": {
            "metrics": {
                "bleu": float(chunk_metrics["bleu"]),
                "chrfpp": float(chunk_metrics["chrfpp"]),
                "geom": float(chunk_metrics["geom"]),
            },
            "output_health": chunk_health,
        },
        "reconstructed": {
            "metrics": {
                "bleu": float(rec_metrics["bleu"]),
                "chrfpp": float(rec_metrics["chrfpp"]),
                "geom": float(rec_metrics["geom"]),
            },
            "output_health": rec_health,
        },
        "reconstructed_df": reconstructed,
    }


def _repeat_trim_long_only(prediction: str, reference: str, source: str) -> str:
    if internal_repeat_score(prediction) <= 0:
        return prediction
    if word_count(prediction) < (0.8 * word_count(reference)):
        return prediction
    patched = repair_gap_markers(prediction)
    patched = clamp_consecutive_repeated_spans(patched, max_span=12, max_occurrences=2)
    patched = collapse_formula_loops(patched, max_repeats=2)
    return patched


def _loop_to_hint(prediction: str, reference: str, source: str) -> str:
    if internal_repeat_score(prediction) <= 0:
        return prediction
    hint = _extract_hint(source)
    return hint if hint else prediction


def _apply_variant(frame: pd.DataFrame, fn: Callable[[str, str, str], str]) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = [
        fn(prediction, reference, source)
        for prediction, reference, source in zip(
            frame["prediction"].fillna("").astype(str).tolist(),
            frame["reference"].fillna("").astype(str).tolist(),
            frame["source"].fillna("").astype(str).tolist(),
        )
    ]
    return out


def _variant_changed_rows(raw_frame: pd.DataFrame, patched_frame: pd.DataFrame) -> pd.DataFrame:
    raw_pred = raw_frame["prediction"].fillna("").astype(str)
    patched_pred = patched_frame["prediction"].fillna("").astype(str)
    out = raw_frame.loc[raw_pred.ne(patched_pred)].copy()
    out = out.rename(columns={"prediction": "prediction_raw"})
    out["prediction_patched"] = patched_frame.loc[out.index, "prediction"].fillna("").astype(str)
    return out.reset_index(drop=True)


def _score_for_selection(anchor_eval: dict[str, Any], full_eval: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(anchor_eval["reconstructed"]["metrics"]["geom"]),
        float(full_eval["reconstructed"]["metrics"]["geom"]),
        -float(full_eval["chunk"]["output_health"]["pred_shorter_than_half_ref_ratio_pct"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor64-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_diagnostic_taskform_winner_a2_r1_wlite_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--fullval-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_diagnostic_taskform_winner_a2_r1_wlite_fullval_20260310.csv",
    )
    ap.add_argument(
        "--incumbent-fullval-summary",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json",
    )
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_health_review_20260310")
    args = ap.parse_args()

    anchor_path = resolve_path(args.anchor64_csv, REPO_ROOT / "runs")
    fullval_path = resolve_path(args.fullval_csv, REPO_ROOT / "runs")
    incumbent_summary_path = resolve_path(args.incumbent_fullval_summary, REPO_ROOT / "runs")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    anchor_raw = _load_frame(anchor_path)
    fullval_raw = _load_frame(fullval_path)
    incumbent_summary = json.loads(incumbent_summary_path.read_text(encoding="utf-8"))

    variant_fns: list[tuple[str, Callable[[str, str, str], str], str]] = [
        ("raw", lambda prediction, reference, source: prediction, "baseline winner output"),
        (
            "repeat_trim_long_only",
            _repeat_trim_long_only,
            "trim repeated spans only on long loop-heavy rows",
        ),
        (
            "loop_to_hint",
            _loop_to_hint,
            "fallback to retrieved English hint only on rows with loop score",
        ),
    ]

    anchor_evals: dict[str, dict[str, Any]] = {}
    fullval_evals: dict[str, dict[str, Any]] = {}
    anchor_frames: dict[str, pd.DataFrame] = {}
    fullval_frames: dict[str, pd.DataFrame] = {}
    changed_row_paths: dict[str, str] = {}

    for label, fn, _ in variant_fns:
        anchor_variant = _apply_variant(anchor_raw, fn)
        fullval_variant = _apply_variant(fullval_raw, fn)
        anchor_frames[label] = anchor_variant
        fullval_frames[label] = fullval_variant
        anchor_evals[label] = _evaluate_variant(anchor_variant)
        fullval_evals[label] = _evaluate_variant(fullval_variant)
        if label != "raw":
            changed = _variant_changed_rows(fullval_raw, fullval_variant)
            changed_path = report_dir / f"{label}_fullval_changed_rows.csv"
            changed.to_csv(changed_path, index=False)
            changed_row_paths[label] = str(changed_path)

    raw_label = "raw"
    recommended_label = max(
        (label for label in anchor_evals if label != raw_label),
        key=lambda label: _score_for_selection(anchor_evals[label], fullval_evals[label]),
    )

    candidate_rows: list[dict[str, Any]] = []
    for label, _, note in variant_fns:
        anchor_eval = anchor_evals[label]
        full_eval = fullval_evals[label]
        compare_vs_i0 = _compare_output_health(
            {"output_health": full_eval["chunk"]["output_health"]},
            {"output_health": incumbent_summary.get("output_health", {}) or {}},
        )
        candidate_rows.append(
            {
                "label": label,
                "note": note,
                "anchor64_reconstructed_geom": round(float(anchor_eval["reconstructed"]["metrics"]["geom"]), 4),
                "anchor64_chunk_geom": round(float(anchor_eval["chunk"]["metrics"]["geom"]), 4),
                "fullval_reconstructed_geom": round(float(full_eval["reconstructed"]["metrics"]["geom"]), 4),
                "fullval_chunk_geom": round(float(full_eval["chunk"]["metrics"]["geom"]), 4),
                "fullval_unique_prediction_ratio_pct": round(
                    float(full_eval["chunk"]["output_health"]["unique_prediction_ratio_pct"]), 4
                ),
                "fullval_short_ratio_pct": round(
                    float(full_eval["chunk"]["output_health"]["pred_shorter_than_half_ref_ratio_pct"]), 4
                ),
                "fullval_health_no_regression_vs_i0": bool(compare_vs_i0["no_regression"]),
            }
        )

    ranking_df = pd.DataFrame(candidate_rows).sort_values(
        ["anchor64_reconstructed_geom", "fullval_reconstructed_geom"],
        ascending=False,
    )
    ranking_path = report_dir / "variant_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    recommended_anchor_chunk_path = report_dir / f"{recommended_label}_anchor64_chunk.csv"
    recommended_fullval_chunk_path = report_dir / f"{recommended_label}_fullval_chunk.csv"
    recommended_anchor_recon_path = report_dir / f"{recommended_label}_anchor64_reconstructed.csv"
    recommended_fullval_recon_path = report_dir / f"{recommended_label}_fullval_reconstructed.csv"

    anchor_frames[recommended_label].to_csv(recommended_anchor_chunk_path, index=False)
    fullval_frames[recommended_label].to_csv(recommended_fullval_chunk_path, index=False)
    anchor_evals[recommended_label]["reconstructed_df"].to_csv(recommended_anchor_recon_path, index=False)
    fullval_evals[recommended_label]["reconstructed_df"].to_csv(recommended_fullval_recon_path, index=False)

    summary = {
        "status": "review_ready_with_a3_candidate",
        "recommended_variant": recommended_label,
        "variants": candidate_rows,
        "anchor64": {
            label: {
                "chunk": payload["chunk"],
                "reconstructed": {
                    "metrics": payload["reconstructed"]["metrics"],
                    "output_health": payload["reconstructed"]["output_health"],
                },
            }
            for label, payload in anchor_evals.items()
        },
        "fullval": {
            label: {
                "chunk": payload["chunk"],
                "reconstructed": {
                    "metrics": payload["reconstructed"]["metrics"],
                    "output_health": payload["reconstructed"]["output_health"],
                },
            }
            for label, payload in fullval_evals.items()
        },
        "fullval_health_vs_i0": {
            label: _compare_output_health(
                {"output_health": fullval_evals[label]["chunk"]["output_health"]},
                {"output_health": incumbent_summary.get("output_health", {}) or {}},
            )
            for label in fullval_evals
        },
        "artifacts": {
            "variant_ranking_csv": str(ranking_path),
            "recommended_anchor64_chunk_csv": str(recommended_anchor_chunk_path),
            "recommended_fullval_chunk_csv": str(recommended_fullval_chunk_path),
            "recommended_anchor64_reconstructed_csv": str(recommended_anchor_recon_path),
            "recommended_fullval_reconstructed_csv": str(recommended_fullval_recon_path),
            "changed_row_csvs": changed_row_paths,
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A2 F Review / Health Repair",
        "",
        f"- recommended variant: `{recommended_label}`",
        f"- raw anchor64 reconstructed geom: `{anchor_evals['raw']['reconstructed']['metrics']['geom']:.4f}`",
        f"- recommended anchor64 reconstructed geom: `{anchor_evals[recommended_label]['reconstructed']['metrics']['geom']:.4f}`",
        f"- raw full-val reconstructed geom: `{fullval_evals['raw']['reconstructed']['metrics']['geom']:.4f}`",
        f"- recommended full-val reconstructed geom: `{fullval_evals[recommended_label]['reconstructed']['metrics']['geom']:.4f}`",
        f"- raw full-val unique ratio: `{fullval_evals['raw']['chunk']['output_health']['unique_prediction_ratio_pct']:.4f}`",
        f"- recommended full-val unique ratio: `{fullval_evals[recommended_label]['chunk']['output_health']['unique_prediction_ratio_pct']:.4f}`",
        f"- raw full-val short ratio: `{fullval_evals['raw']['chunk']['output_health']['pred_shorter_than_half_ref_ratio_pct']:.4f}`",
        f"- recommended full-val short ratio: `{fullval_evals[recommended_label]['chunk']['output_health']['pred_shorter_than_half_ref_ratio_pct']:.4f}`",
        f"- clears auto health gate vs incumbent: `{bool(summary['fullval_health_vs_i0'][recommended_label]['no_regression'])}`",
        "- note: current repair is recommended as an A3 candidate and manual-review aid, not as an automatic replacement for raw W-lite",
    ]
    write_text(report_dir / "manual_review.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
