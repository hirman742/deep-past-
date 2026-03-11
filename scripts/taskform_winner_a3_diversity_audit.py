from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu

from taskform_phase12_common import evaluate_predictions, resolve_path, write_json, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _load_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "oare_id"
    needed = {id_col, "source", "reference", "prediction"}
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    out = frame[[id_col, "source", "reference", "prediction"]].copy()
    out = out.rename(
        columns={
            id_col: "id",
            "source": f"source_{label}",
            "reference": f"reference_{label}",
            "prediction": f"prediction_{label}",
        }
    )
    out["id"] = out["id"].fillna("").astype(str)
    for col in out.columns:
        if col != "id":
            out[col] = out[col].fillna("").astype(str)
    return out


def _sentence_geom(prediction: str, reference: str) -> float:
    bleu = sacrebleu.metrics.BLEU().sentence_score(prediction, [reference]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).sentence_score(prediction, [reference]).score
    return math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))


def _merge_frames(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for _, frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="id", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping ids across candidate files")

    source_cols = [f"source_{label}" for label, _ in frames]
    ref_cols = [f"reference_{label}" for label, _ in frames]
    merged["source"] = merged[source_cols].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
    merged["reference"] = merged[ref_cols].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
    return merged


def _pair_similarity(left_preds: list[str], right_preds: list[str]) -> dict[str, float]:
    left_to_right = evaluate_predictions(
        predictions=left_preds,
        references=right_preds,
        tag="left_to_right",
        subset_name="pair_similarity",
    )
    right_to_left = evaluate_predictions(
        predictions=right_preds,
        references=left_preds,
        tag="right_to_left",
        subset_name="pair_similarity",
    )
    return {
        "self_bleu_mean": round(
            0.5 * (float(left_to_right["eval_bleu"]) + float(right_to_left["eval_bleu"])),
            4,
        ),
        "self_chrfpp_mean": round(
            0.5 * (float(left_to_right["eval_chrfpp"]) + float(right_to_left["eval_chrfpp"])),
            4,
        ),
        "self_geom_mean": round(
            0.5 * (float(left_to_right["eval_geom"]) + float(right_to_left["eval_geom"])),
            4,
        ),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--incumbent-csv",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_reconstructed_taskform_a2_a1_incumbent_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--r1-smoke-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_20260310_fold0/diagnostics/val_predictions_reconstructed_taskform_winner_a2_r1_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--w-lite-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_reconstructed_taskform_winner_a2_r1_wlite_anchor64_20260310.csv",
    )
    ap.add_argument("--report-dir", default="reports/taskform_winner_a3_diversity_20260310")
    args = ap.parse_args()

    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a3_diversity_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_specs = [
        ("incumbent", resolve_path(args.incumbent_csv, REPO_ROOT / "runs")),
        ("retrieval_smoke", resolve_path(args.r1_smoke_csv, REPO_ROOT / "runs")),
        ("retrieval_wlite", resolve_path(args.w_lite_csv, REPO_ROOT / "runs")),
    ]
    frames = [(label, _load_predictions(path, label)) for label, path in candidate_specs]
    merged = _merge_frames(frames)

    labels = [label for label, _ in candidate_specs]
    row_records: list[dict[str, Any]] = []
    sentence_scores: dict[str, list[float]] = {label: [] for label in labels}
    unique_count_total = 0.0
    all_same_rows = 0
    all_unique_rows = 0

    for record in merged.to_dict(orient="records"):
        ref = _safe_text(record["reference"])
        row_out: dict[str, Any] = {
            "id": _safe_text(record["id"]),
            "reference": ref,
        }
        row_predictions = []
        for label in labels:
            pred = _safe_text(record[f"prediction_{label}"])
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
            **_pair_similarity(left_preds, right_preds),
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
            "sentence_geom_mean": round(
                float(sum(sentence_scores[label])) / float(max(1, len(sentence_scores[label]))),
                4,
            ),
        }

    diversity_summary = {
        "status": "ready",
        "rows": int(len(merged)),
        "candidates": candidate_metrics,
        "pool": {
            "candidate_count": int(len(labels)),
            "mean_unique_candidates_per_row": round(unique_count_total / float(max(1, len(merged))), 4),
            "unique_candidate_ratio_pct": round(
                100.0 * unique_count_total / float(max(1, len(merged) * len(labels))),
                4,
            ),
            "rows_all_same_ratio_pct": round(100.0 * float(all_same_rows) / float(max(1, len(merged))), 4),
            "rows_all_unique_ratio_pct": round(100.0 * float(all_unique_rows) / float(max(1, len(merged))), 4),
        },
        "pairwise": pairwise,
        "artifacts": {
            "rowwise_csv": str(report_dir / "rowwise.csv"),
            "summary_json": str(report_dir / "summary.json"),
            "report_md": str(report_dir / "report.md"),
        },
    }

    pd.DataFrame(row_records).to_csv(report_dir / "rowwise.csv", index=False)
    write_json(report_dir / "summary.json", diversity_summary)

    lines = [
        "# A3 Diversity Audit",
        "",
        f"- rows: `{diversity_summary['rows']}`",
        f"- pool unique candidate ratio: `{diversity_summary['pool']['unique_candidate_ratio_pct']}`",
        f"- rows all same: `{diversity_summary['pool']['rows_all_same_ratio_pct']}`",
        f"- rows all unique: `{diversity_summary['pool']['rows_all_unique_ratio_pct']}`",
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
                f"- `{key}` self-BLEU / self-chrF++ / self-geom: `{metrics['self_bleu_mean']}` / `{metrics['self_chrfpp_mean']}` / `{metrics['self_geom_mean']}`",
                f"- `{key}` bottom25 overlap / jaccard: `{metrics['bottom25_overlap_ratio_pct']}` / `{metrics['bottom25_jaccard']}`",
            ]
        )
    write_text(report_dir / "report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
