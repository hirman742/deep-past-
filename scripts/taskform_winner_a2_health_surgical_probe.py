from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from taskform_phase12_common import (
    clamp_consecutive_repeated_spans,
    collapse_formula_loops,
    compute_translation_metrics,
    internal_repeat_score,
    repair_gap_markers,
    resolve_path,
    word_count,
    write_json,
    write_text,
)
from taskform_winner_a2_health_review import (
    _aggregate_parent_df,
    _compare_output_health,
    _diagnostic_health,
    _filter_original_chunks,
    _load_frame,
)
from taskform_winner_a2_retrieval_eval import _compute_hard_subset_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repeat_trim_long_only(prediction: str, reference: str) -> str:
    if internal_repeat_score(prediction) <= 0:
        return prediction
    if word_count(prediction) < (0.8 * word_count(reference)):
        return prediction
    patched = repair_gap_markers(prediction)
    patched = clamp_consecutive_repeated_spans(patched, max_span=12, max_occurrences=2)
    patched = collapse_formula_loops(patched, max_repeats=2)
    return patched


def _fallback_variant(
    *,
    raw_df: pd.DataFrame,
    incumbent_df: pd.DataFrame,
    repeat_ge: int,
    extra_repeat_ge: int,
    extra_max_len: int,
    apply_trim_after_fallback: bool,
) -> pd.DataFrame:
    merged = raw_df.merge(
        incumbent_df[["oare_id", "prediction"]].rename(columns={"prediction": "prediction_incumbent"}),
        on="oare_id",
        how="left",
    )
    raw_predictions = merged["prediction"].fillna("").astype(str).tolist()
    incumbent_predictions = merged["prediction_incumbent"].fillna("").astype(str).tolist()
    references = merged["reference"].fillna("").astype(str).tolist()
    counts = Counter(raw_predictions)

    patched: list[str] = []
    for raw_pred, incumbent_pred, reference in zip(raw_predictions, incumbent_predictions, references):
        use_fallback = counts[raw_pred] >= int(repeat_ge) or (
            counts[raw_pred] >= int(extra_repeat_ge) and len(raw_pred) <= int(extra_max_len)
        )
        chosen = incumbent_pred if use_fallback and incumbent_pred else raw_pred
        if apply_trim_after_fallback:
            chosen = _repeat_trim_long_only(chosen, reference)
        patched.append(chosen)

    out = merged.drop(columns=["prediction_incumbent"]).copy()
    out["prediction"] = patched
    return out


def _evaluate_bundle(
    *,
    frame: pd.DataFrame,
    hard_ids_csv: Path,
    out_reconstructed_csv: Path,
) -> dict[str, Any]:
    predictions = frame["prediction"].fillna("").astype(str).tolist()
    references = frame["reference"].fillna("").astype(str).tolist()
    chunk_metrics = compute_translation_metrics(predictions=predictions, references=references)
    chunk_health = _diagnostic_health(frame)

    reconstructed = _aggregate_parent_df(_filter_original_chunks(frame))
    reconstructed.to_csv(out_reconstructed_csv, index=False)
    rec_predictions = reconstructed["prediction"].fillna("").astype(str).tolist()
    rec_references = reconstructed["reference"].fillna("").astype(str).tolist()
    rec_metrics = compute_translation_metrics(predictions=rec_predictions, references=rec_references)
    rec_health = _diagnostic_health(reconstructed)
    hard_subset = _compute_hard_subset_metrics(out_reconstructed_csv, hard_ids_csv)

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
        "hard_subset": hard_subset,
    }


def _score_row(row: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        1 if bool(row["fullval_health_no_regression_vs_i0"]) else 0,
        float(row["fullval_reconstructed_geom"]),
        float(row["hard_geom"]),
        float(row["anchor64_reconstructed_geom"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor64-wlite-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_diagnostic_taskform_winner_a2_r1_wlite_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--fullval-wlite-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_diagnostic_taskform_winner_a2_r1_wlite_fullval_20260310.csv",
    )
    ap.add_argument(
        "--anchor64-incumbent-csv",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_diagnostic_taskform_a2_a1_incumbent_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--fullval-incumbent-csv",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv",
    )
    ap.add_argument(
        "--incumbent-fullval-summary",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json",
    )
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_health_surgical_20260310")
    args = ap.parse_args()

    anchor_wlite = _load_frame(resolve_path(args.anchor64_wlite_csv, REPO_ROOT / "runs"))
    fullval_wlite = _load_frame(resolve_path(args.fullval_wlite_csv, REPO_ROOT / "runs"))
    anchor_inc = _load_frame(resolve_path(args.anchor64_incumbent_csv, REPO_ROOT / "runs"))
    fullval_inc = _load_frame(resolve_path(args.fullval_incumbent_csv, REPO_ROOT / "runs"))
    incumbent_fullval_summary = json.loads(
        resolve_path(args.incumbent_fullval_summary, REPO_ROOT / "runs").read_text(encoding="utf-8")
    )
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = [
        ("raw", 99, 99, 0, False, "baseline raw W-lite"),
        ("fallback_80", 5, 4, 80, False, "fallback to incumbent on repeat>=5 or repeat>=4 with len<=80"),
        ("fallback_140", 5, 4, 140, False, "fallback to incumbent on repeat>=5 or repeat>=4 with len<=140"),
        ("fallback_180", 5, 4, 180, False, "fallback to incumbent on repeat>=5 or repeat>=4 with len<=180"),
        (
            "fallback_180_plus_trim",
            5,
            4,
            180,
            True,
            "same as fallback_180, then trim long internal repeats",
        ),
    ]

    rows: list[dict[str, Any]] = []
    variant_payloads: dict[str, Any] = {}

    for label, repeat_ge, extra_repeat_ge, extra_max_len, apply_trim, note in variant_specs:
        if label == "raw":
            anchor_variant = anchor_wlite.copy()
            fullval_variant = fullval_wlite.copy()
        else:
            anchor_variant = _fallback_variant(
                raw_df=anchor_wlite,
                incumbent_df=anchor_inc,
                repeat_ge=repeat_ge,
                extra_repeat_ge=extra_repeat_ge,
                extra_max_len=extra_max_len,
                apply_trim_after_fallback=apply_trim,
            )
            fullval_variant = _fallback_variant(
                raw_df=fullval_wlite,
                incumbent_df=fullval_inc,
                repeat_ge=repeat_ge,
                extra_repeat_ge=extra_repeat_ge,
                extra_max_len=extra_max_len,
                apply_trim_after_fallback=apply_trim,
            )

        anchor_recon_csv = report_dir / f"{label}_anchor64_reconstructed.csv"
        fullval_recon_csv = report_dir / f"{label}_fullval_reconstructed.csv"
        anchor_chunk_csv = report_dir / f"{label}_anchor64_chunk.csv"
        fullval_chunk_csv = report_dir / f"{label}_fullval_chunk.csv"
        anchor_variant.to_csv(anchor_chunk_csv, index=False)
        fullval_variant.to_csv(fullval_chunk_csv, index=False)

        anchor_eval = _evaluate_bundle(
            frame=anchor_variant,
            hard_ids_csv=hard_ids_csv,
            out_reconstructed_csv=anchor_recon_csv,
        )
        fullval_eval = _evaluate_bundle(
            frame=fullval_variant,
            hard_ids_csv=hard_ids_csv,
            out_reconstructed_csv=fullval_recon_csv,
        )
        fullval_health_vs_i0 = _compare_output_health(
            {"output_health": fullval_eval["chunk"]["output_health"]},
            {"output_health": incumbent_fullval_summary.get("output_health", {}) or {}},
        )

        row = {
            "label": label,
            "note": note,
            "anchor64_reconstructed_geom": round(float(anchor_eval["reconstructed"]["metrics"]["geom"]), 4),
            "anchor64_chunk_geom": round(float(anchor_eval["chunk"]["metrics"]["geom"]), 4),
            "fullval_reconstructed_geom": round(float(fullval_eval["reconstructed"]["metrics"]["geom"]), 4),
            "fullval_chunk_geom": round(float(fullval_eval["chunk"]["metrics"]["geom"]), 4),
            "hard_geom": round(float(fullval_eval["hard_subset"]["eval_geom"]), 4),
            "fullval_unique_prediction_ratio_pct": round(
                float(fullval_eval["chunk"]["output_health"]["unique_prediction_ratio_pct"]),
                4,
            ),
            "fullval_short_ratio_pct": round(
                float(fullval_eval["chunk"]["output_health"]["pred_shorter_than_half_ref_ratio_pct"]),
                4,
            ),
            "fullval_health_no_regression_vs_i0": bool(fullval_health_vs_i0["no_regression"]),
        }
        rows.append(row)
        variant_payloads[label] = {
            "anchor64": anchor_eval,
            "fullval": fullval_eval,
            "fullval_health_vs_i0": fullval_health_vs_i0,
            "artifacts": {
                "anchor64_chunk_csv": str(anchor_chunk_csv),
                "anchor64_reconstructed_csv": str(anchor_recon_csv),
                "fullval_chunk_csv": str(fullval_chunk_csv),
                "fullval_reconstructed_csv": str(fullval_recon_csv),
            },
        }

    ranking = pd.DataFrame(rows).sort_values(
        by=[
            "fullval_health_no_regression_vs_i0",
            "fullval_reconstructed_geom",
            "hard_geom",
            "anchor64_reconstructed_geom",
        ],
        ascending=[False, False, False, False],
    )
    ranking_path = report_dir / "variant_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    recommended_label = str(ranking.iloc[0]["label"])

    summary = {
        "status": "ready_for_promote_compare" if bool(variant_payloads[recommended_label]["fullval_health_vs_i0"]["no_regression"]) else "review_stop",
        "recommended_variant": recommended_label,
        "variants": rows,
        "variant_payloads": variant_payloads,
        "artifacts": {
            "variant_ranking_csv": str(ranking_path),
            "recommended_fullval_chunk_csv": variant_payloads[recommended_label]["artifacts"]["fullval_chunk_csv"],
            "recommended_fullval_reconstructed_csv": variant_payloads[recommended_label]["artifacts"]["fullval_reconstructed_csv"],
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A2 Health Surgical Probe",
        "",
        f"- recommended variant: `{recommended_label}`",
        f"- status: `{summary['status']}`",
        f"- raw full-val reconstructed geom: `{variant_payloads['raw']['fullval']['reconstructed']['metrics']['geom']:.4f}`",
        f"- recommended full-val reconstructed geom: `{variant_payloads[recommended_label]['fullval']['reconstructed']['metrics']['geom']:.4f}`",
        f"- raw full-val unique ratio: `{variant_payloads['raw']['fullval']['chunk']['output_health']['unique_prediction_ratio_pct']:.4f}`",
        f"- recommended full-val unique ratio: `{variant_payloads[recommended_label]['fullval']['chunk']['output_health']['unique_prediction_ratio_pct']:.4f}`",
        f"- raw full-val short ratio: `{variant_payloads['raw']['fullval']['chunk']['output_health']['pred_shorter_than_half_ref_ratio_pct']:.4f}`",
        f"- recommended full-val short ratio: `{variant_payloads[recommended_label]['fullval']['chunk']['output_health']['pred_shorter_than_half_ref_ratio_pct']:.4f}`",
        f"- clears health gate vs incumbent: `{bool(variant_payloads[recommended_label]['fullval_health_vs_i0']['no_regression'])}`",
        f"- hard subset geom: `{variant_payloads[recommended_label]['fullval']['hard_subset']['eval_geom']:.4f}`",
    ]
    write_text(report_dir / "manual_review.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
