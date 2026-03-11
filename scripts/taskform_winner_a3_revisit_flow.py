#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu

from taskform_phase12_common import (
    evaluate_predictions,
    formula_count,
    internal_repeat_score,
    resolve_path,
    safe_text,
    tokenize_words,
    write_json,
    write_text,
)
from taskform_winner_a2_retrieval_eval import _load_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "oare_id"
    if "reference" not in frame.columns or "prediction" not in frame.columns:
        raise KeyError(f"Missing reference/prediction in {path}")
    out = frame[[id_col, "source", "reference", "prediction"]].copy()
    out = out.rename(columns={id_col: "id"})
    out["id"] = out["id"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    out["reference"] = out["reference"].fillna("").astype(str)
    out["prediction"] = out["prediction"].fillna("").astype(str)
    return out


def _sentence_geom(prediction: str, reference: str) -> float:
    bleu = sacrebleu.metrics.BLEU().sentence_score(prediction, [reference]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).sentence_score(prediction, [reference]).score
    return math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))


def _bottom_quartile_mask(values: list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    threshold = float(series.quantile(0.25)) if len(series) else 0.0
    return series.le(threshold)


def _pair_error_overlap(left_scores: list[float], right_scores: list[float]) -> dict[str, float]:
    left_mask = _bottom_quartile_mask(left_scores)
    right_mask = _bottom_quartile_mask(right_scores)
    overlap = left_mask & right_mask
    union = left_mask | right_mask
    return {
        "bottom25_overlap_rows": int(overlap.sum()),
        "bottom25_union_rows": int(union.sum()),
        "bottom25_overlap_ratio_pct": round(100.0 * float(overlap.mean() if len(overlap) else 0.0), 4),
        "bottom25_jaccard": round(float(overlap.sum()) / float(max(1, union.sum())), 4),
    }


def _heuristic_pick(source: str, left: str, right: str) -> tuple[str, str]:
    def score(text: str) -> tuple[float, float, float]:
        words = len(tokenize_words(text))
        src_words = max(1, len(tokenize_words(source)))
        short_penalty = max(0.0, (0.12 * float(src_words)) - float(words))
        return (
            -4.0 * float(internal_repeat_score(text)),
            -2.0 * float(formula_count(text)),
            -float(short_penalty),
        )

    left_score = score(left)
    right_score = score(right)
    if left_score > right_score:
        return left, "baseline"
    if right_score > left_score:
        return right, "probe"
    if len(left) >= len(right):
        return left, "baseline"
    return right, "probe"


def _pairwise_oracle(baseline_label: str, baseline_df: pd.DataFrame, probe_label: str, probe_df: pd.DataFrame) -> dict[str, Any]:
    merge_mode = "id_source_reference"
    merged = baseline_df.merge(probe_df, on=["id", "source", "reference"], suffixes=("_baseline", "_probe"), how="inner")
    if merged.empty:
        merge_mode = "id_reference"
        baseline_fallback = baseline_df.drop_duplicates(subset=["id", "reference"], keep="first")
        probe_fallback = probe_df.drop(columns=["source"], errors="ignore").drop_duplicates(
            subset=["id", "reference"], keep="first"
        )
        merged = baseline_fallback.merge(
            probe_fallback,
            on=["id", "reference"],
            suffixes=("_baseline", "_probe"),
            how="inner",
        )
        if "source" not in merged.columns and "source_baseline" in merged.columns:
            merged["source"] = merged["source_baseline"].fillna("").astype(str)
    if merged.empty:
        merge_mode = "source_reference"
        baseline_fallback = baseline_df.drop(columns=["id"], errors="ignore").drop_duplicates(
            subset=["source", "reference"], keep="first"
        )
        probe_fallback = probe_df.drop(columns=["id"], errors="ignore").drop_duplicates(
            subset=["source", "reference"], keep="first"
        )
        merged = baseline_fallback.merge(
            probe_fallback,
            on=["source", "reference"],
            suffixes=("_baseline", "_probe"),
            how="inner",
        )
    if merged.empty:
        raise ValueError(f"No overlapping rows between {baseline_label} and {probe_label}")

    baseline_summary = evaluate_predictions(
        predictions=merged["prediction_baseline"].tolist(),
        references=merged["reference"].tolist(),
        tag=f"{baseline_label}_baseline",
        subset_name="diversity_probe",
    )
    probe_summary = evaluate_predictions(
        predictions=merged["prediction_probe"].tolist(),
        references=merged["reference"].tolist(),
        tag=f"{probe_label}_probe",
        subset_name="diversity_probe",
    )

    oracle_predictions: list[str] = []
    heuristic_predictions: list[str] = []
    heuristic_choice_counter: Counter[str] = Counter()
    disagreement = 0
    baseline_win = 0
    probe_win = 0
    for row in merged.to_dict(orient="records"):
        baseline_pred = safe_text(row["prediction_baseline"])
        probe_pred = safe_text(row["prediction_probe"])
        reference = safe_text(row["reference"])
        source = safe_text(row["source"])
        if baseline_pred != probe_pred:
            disagreement += 1
        baseline_geom = _sentence_geom(baseline_pred, reference)
        probe_geom = _sentence_geom(probe_pred, reference)
        if baseline_geom >= probe_geom:
            oracle_predictions.append(baseline_pred)
            baseline_win += 1
        else:
            oracle_predictions.append(probe_pred)
            probe_win += 1
        heuristic_pred, heuristic_choice = _heuristic_pick(source, baseline_pred, probe_pred)
        heuristic_predictions.append(heuristic_pred)
        heuristic_choice_counter[heuristic_choice] += 1

    oracle_summary = evaluate_predictions(
        predictions=oracle_predictions,
        references=merged["reference"].tolist(),
        tag=f"{baseline_label}_vs_{probe_label}_oracle",
        subset_name="diversity_probe",
        note="oracle per-row upper bound, not deployable",
    )
    heuristic_summary = evaluate_predictions(
        predictions=heuristic_predictions,
        references=merged["reference"].tolist(),
        tag=f"{baseline_label}_vs_{probe_label}_heuristic",
        subset_name="diversity_probe",
        note="selector prefers lower repeat/formula pressure",
    )
    disagreement_ratio = 100.0 * float(disagreement) / float(max(1, len(merged)))
    oracle_delta = float(oracle_summary["eval_geom"]) - max(float(baseline_summary["eval_geom"]), float(probe_summary["eval_geom"]))
    heuristic_delta = float(heuristic_summary["eval_geom"]) - max(float(baseline_summary["eval_geom"]), float(probe_summary["eval_geom"]))
    return {
        "rows": int(len(merged)),
        "baseline_label": baseline_label,
        "probe_label": probe_label,
        "baseline": baseline_summary,
        "probe": probe_summary,
        "oracle_upper_bound": oracle_summary,
        "heuristic_selector": heuristic_summary,
        "disagreement_ratio_pct": round(disagreement_ratio, 4),
        "baseline_oracle_wins": int(baseline_win),
        "probe_oracle_wins": int(probe_win),
        "heuristic_choice_counts": dict(heuristic_choice_counter),
        "oracle_delta_geom_vs_best_single": round(oracle_delta, 4),
        "heuristic_delta_geom_vs_best_single": round(heuristic_delta, 4),
        "merge_mode": merge_mode,
    }


def _reconstructed_csv_from_probe(candidate: dict[str, Any]) -> Path:
    diag_payload = _load_json(Path(str(candidate["diagnose_summary_path"])))
    reconstructed_csv = Path(
        str((((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", "")))
    )
    if not reconstructed_csv.exists():
        raise FileNotFoundError(f"Missing reconstructed csv: {reconstructed_csv}")
    return reconstructed_csv


def _merge_candidate_frames(frames: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, str]:
    merged_by_id_ref: pd.DataFrame | None = None
    for label, frame in frames:
        subset = frame[["id", "source", "reference", "prediction"]].copy()
        renamed = subset.rename(
            columns={
                "source": f"source_{label}",
                "prediction": f"prediction_{label}",
            }
        )
        if merged_by_id_ref is None:
            merged_by_id_ref = renamed
        else:
            merged_by_id_ref = merged_by_id_ref.merge(renamed, on=["id", "reference"], how="inner")
    if merged_by_id_ref is not None and not merged_by_id_ref.empty:
        return merged_by_id_ref, "id_reference"

    merged_by_text: pd.DataFrame | None = None
    id_cols: list[str] = []
    for label, frame in frames:
        subset = frame[["id", "source", "reference", "prediction"]].drop_duplicates(
            subset=["source", "reference"], keep="first"
        )
        renamed = subset.rename(
            columns={
                "id": f"id_{label}",
                "prediction": f"prediction_{label}",
            }
        )
        id_cols.append(f"id_{label}")
        if merged_by_text is None:
            merged_by_text = renamed
        else:
            merged_by_text = merged_by_text.merge(renamed, on=["source", "reference"], how="inner")
    if merged_by_text is None or merged_by_text.empty:
        raise ValueError("No overlapping ids or source/reference pairs across A3 revisit candidate files")

    merged_by_text["id"] = (
        merged_by_text[id_cols].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
        if id_cols
        else pd.Series([f"row::{idx}" for idx in range(len(merged_by_text))], dtype=str)
    )
    return merged_by_text, "source_reference"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_RAW_LONGTRAIN_20260311_fold0/diagnostics/val_predictions_reconstructed_winner_retrieval_raw_wlite_longtrain_anchor64_20260311.csv",
    )
    ap.add_argument("--baseline-label", default="retrieval_raw_longtrain")
    ap.add_argument("--combo-summary", default="reports/taskform_winner_combo_retrieval_replay25_probe_20260311/summary.json")
    ap.add_argument("--replay-summary", default="reports/taskform_winner_replay_band_probe_20260311/summary.json")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a3_revisit_20260311")
    args = ap.parse_args()

    baseline_csv = resolve_path(
        args.baseline_csv,
        REPO_ROOT / "runs" / "TASKFORM_WINNER_A2_RETRIEVAL_RAW_LONGTRAIN_20260311_fold0" / "diagnostics" / "val_predictions_reconstructed_winner_retrieval_raw_wlite_longtrain_anchor64_20260311.csv",
    )
    combo_summary_path = resolve_path(
        args.combo_summary,
        REPO_ROOT / "reports" / "taskform_winner_combo_retrieval_replay25_probe_20260311" / "summary.json",
    )
    replay_summary_path = resolve_path(
        args.replay_summary,
        REPO_ROOT / "reports" / "taskform_winner_replay_band_probe_20260311" / "summary.json",
    )
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a3_revisit_20260311")
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline_label = safe_text(args.baseline_label) or "baseline"
    baseline_df = _load_frame(baseline_csv)
    combo_summary = _load_json(combo_summary_path)
    replay_summary = _load_json(replay_summary_path)

    candidate_specs: list[tuple[str, Path]] = [(baseline_label, baseline_csv)]

    combo_status = safe_text(combo_summary.get("status", ""))
    if combo_status != "reject_stop":
        combo_candidate = (combo_summary.get("candidates", {}) or {}).get("combo", {}) or {}
        if combo_candidate:
            candidate_specs.append(("combo_replay25", _reconstructed_csv_from_probe(combo_candidate)))

    replay_best_label = safe_text(replay_summary.get("best_label", ""))
    replay_status = safe_text(replay_summary.get("status", ""))
    if replay_status != "reject_stop" and replay_best_label and replay_best_label != "ctrl":
        replay_candidate = (replay_summary.get("candidates", {}) or {}).get(replay_best_label, {}) or {}
        if replay_candidate:
            candidate_specs.append((replay_best_label, _reconstructed_csv_from_probe(replay_candidate)))

    if len(candidate_specs) < 2:
        summary = {
            "status": "skipped_no_probe_candidates",
            "reason": "No eligible post-probe candidates available for A3 revisit",
            "candidate_specs": [{"label": label, "csv": str(path)} for label, path in candidate_specs],
        }
        write_json(report_dir / "summary.json", summary)
        write_text(
            report_dir / "gate_report.md",
            "# Winner A3 Revisit\n\n- status: `skipped_no_probe_candidates`\n- reason: no eligible probe winners\n",
        )
        return

    frames = [(label, _load_frame(path)) for label, path in candidate_specs]
    merged, merge_mode = _merge_candidate_frames(frames)

    labels = [label for label, _ in candidate_specs]
    if "source" not in merged.columns:
        source_cols = [f"source_{label}" for label in labels]
        merged["source"] = merged[source_cols].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
    else:
        merged["source"] = merged["source"].fillna("").astype(str)
    if "reference" not in merged.columns:
        ref_cols = [f"reference_{label}" for label in labels]
        merged["reference"] = merged[ref_cols].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
    else:
        merged["reference"] = merged["reference"].fillna("").astype(str)

    row_records: list[dict[str, Any]] = []
    sentence_scores: dict[str, list[float]] = {label: [] for label in labels}
    unique_count_total = 0.0
    all_same_rows = 0
    all_unique_rows = 0

    for record in merged.to_dict(orient="records"):
        ref = safe_text(record["reference"])
        row_out: dict[str, Any] = {"id": safe_text(record["id"]), "reference": ref}
        row_predictions = []
        for label in labels:
            pred = safe_text(record[f"prediction_{label}"])
            score = _sentence_geom(pred, ref)
            sentence_scores[label].append(score)
            row_out[f"prediction_{label}"] = pred
            row_out[f"sentence_geom_{label}"] = round(score, 4)
            row_predictions.append(pred)
        unique_count = len(set(row_predictions))
        unique_count_total += float(unique_count)
        if unique_count == 1:
            all_same_rows += 1
        if unique_count == len(labels):
            all_unique_rows += 1
        row_out["unique_candidate_count"] = int(unique_count)
        row_records.append(row_out)

    pairwise: dict[str, Any] = {}
    for left, right in itertools.combinations(labels, 2):
        left_preds = merged[f"prediction_{left}"].fillna("").astype(str).tolist()
        right_preds = merged[f"prediction_{right}"].fillna("").astype(str).tolist()
        exact_mask = pd.Series(left_preds).eq(pd.Series(right_preds))
        pairwise[f"{left}__vs__{right}"] = {
            "rows": int(len(merged)),
            "exact_overlap_rows": int(exact_mask.sum()),
            "exact_overlap_ratio_pct": round(100.0 * float(exact_mask.mean() if len(exact_mask) else 0.0), 4),
            **_pair_error_overlap(sentence_scores[left], sentence_scores[right]),
        }

    candidate_metrics = {}
    for label in labels:
        candidate_metrics[label] = {
            "rows": int(len(merged)),
            "unique_prediction_ratio_pct": round(
                100.0 * float(pd.Series(merged[f"prediction_{label}"]).nunique()) / float(max(1, len(merged))),
                4,
            ),
            "sentence_geom_mean": round(float(sum(sentence_scores[label])) / float(max(1, len(sentence_scores[label]))), 4),
        }

    oracle_pairwise: dict[str, Any] = {}
    best_oracle_key = ""
    best_oracle_delta = -1e9
    for label in labels:
        if label == baseline_label:
            continue
        probe_df = _load_frame(dict(candidate_specs)[label])
        oracle_pairwise[label] = _pairwise_oracle(baseline_label, baseline_df, label, probe_df)
        delta = float(oracle_pairwise[label]["oracle_delta_geom_vs_best_single"])
        if delta > best_oracle_delta:
            best_oracle_delta = delta
            best_oracle_key = label

    summary = {
        "status": "completed_revisit_audit",
        "rows": int(len(merged)),
        "merge_mode": merge_mode,
        "baseline_label": baseline_label,
        "candidates": candidate_metrics,
        "pool": {
            "candidate_count": int(len(labels)),
            "mean_unique_candidates_per_row": round(unique_count_total / float(max(1, len(merged))), 4),
            "unique_candidate_ratio_pct": round(100.0 * unique_count_total / float(max(1, len(merged) * len(labels))), 4),
            "rows_all_same_ratio_pct": round(100.0 * float(all_same_rows) / float(max(1, len(merged))), 4),
            "rows_all_unique_ratio_pct": round(100.0 * float(all_unique_rows) / float(max(1, len(merged))), 4),
        },
        "pairwise": pairwise,
        "oracle_pairwise": oracle_pairwise,
        "best_oracle_label": best_oracle_key,
        "best_oracle_delta_geom_vs_best_single": round(best_oracle_delta, 4),
        "artifacts": {
            "rowwise_csv": str(report_dir / "rowwise.csv"),
            "summary_json": str(report_dir / "summary.json"),
            "gate_report_md": str(report_dir / "gate_report.md"),
        },
    }
    pd.DataFrame(row_records).to_csv(report_dir / "rowwise.csv", index=False)
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# Winner A3 Revisit",
        "",
        f"- status: `{summary['status']}`",
        f"- rows: `{summary['rows']}`",
        f"- pool unique candidate ratio: `{summary['pool']['unique_candidate_ratio_pct']}`",
        f"- rows all same: `{summary['pool']['rows_all_same_ratio_pct']}`",
        f"- rows all unique: `{summary['pool']['rows_all_unique_ratio_pct']}`",
        f"- best oracle label: `{summary['best_oracle_label']}`",
        f"- best oracle delta vs best single: `{summary['best_oracle_delta_geom_vs_best_single']}`",
        "",
        "## Candidate uniqueness",
        "",
    ]
    for label in labels:
        metrics = candidate_metrics[label]
        lines.extend(
            [
                f"- `{label}` unique prediction ratio: `{metrics['unique_prediction_ratio_pct']}`",
                f"- `{label}` sentence geom mean: `{metrics['sentence_geom_mean']}`",
            ]
        )
    lines.extend(["", "## Pairwise overlap", ""])
    for key, metrics in pairwise.items():
        lines.extend(
            [
                f"- `{key}` exact overlap: `{metrics['exact_overlap_ratio_pct']}`",
                f"- `{key}` bottom25 overlap / jaccard: `{metrics['bottom25_overlap_ratio_pct']}` / `{metrics['bottom25_jaccard']}`",
            ]
        )
    lines.extend(["", "## Oracle Pairwise", ""])
    for label, metrics in oracle_pairwise.items():
        lines.extend(
            [
                f"- `{baseline_label} vs {label}` disagreement ratio: `{metrics['disagreement_ratio_pct']}`",
                f"- `{baseline_label} vs {label}` oracle delta vs best single: `{metrics['oracle_delta_geom_vs_best_single']}`",
                f"- `{baseline_label} vs {label}` heuristic delta vs best single: `{metrics['heuristic_delta_geom_vs_best_single']}`",
            ]
        )
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
