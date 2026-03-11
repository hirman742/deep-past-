from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import compute_translation_metrics, resolve_path, safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import REPO_ROOT
from taskform_winner_a2_retrieval_wlite_eval import _load_json


def _metric_triplet(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "geom": float(payload.get(f"{prefix}_reconstructed_geom", payload.get(f"{prefix}_geom", 0.0)) or 0.0),
        "chunk_geom": float(payload.get(f"{prefix}_chunk_geom", 0.0) or 0.0),
    }


def _summarize_repeat_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby(["raw_prediction", "raw_count"], dropna=False)
        .agg(
            changed_rows=("oare_id", "count"),
            changed_parents=("parent_id", "nunique"),
        )
        .reset_index()
        .sort_values(["raw_count", "changed_rows", "raw_prediction"], ascending=[False, False, True])
    )
    rows: list[dict[str, Any]] = []
    for record in grouped.head(10).to_dict(orient="records"):
        rows.append(
            {
                "raw_prediction": safe_text(str(record["raw_prediction"]))[:240],
                "raw_count": int(record["raw_count"]),
                "changed_rows": int(record["changed_rows"]),
                "changed_parents": int(record["changed_parents"]),
            }
        )
    return rows


def _changed_subset_metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    metrics = compute_translation_metrics(
        predictions=frame[column].fillna("").astype(str).tolist(),
        references=frame["reference"].fillna("").astype(str).tolist(),
    )
    return {
        "bleu": round(float(metrics["bleu"]), 4),
        "chrfpp": round(float(metrics["chrfpp"]), 4),
        "geom": round(float(metrics["geom"]), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wlite-summary",
        default="reports/taskform_winner_a2_retrieval_wlite_eval_20260310/summary.json",
    )
    ap.add_argument(
        "--health-surgical-summary",
        default="reports/taskform_winner_a2_health_surgical_20260310/summary.json",
    )
    ap.add_argument(
        "--support-summary",
        default="reports/taskform_winner_a2_support_20260310/summary.json",
    )
    ap.add_argument(
        "--hard-ids-csv",
        default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv",
    )
    ap.add_argument(
        "--report-dir",
        default="reports/taskform_winner_a2_promote_compare_20260310",
    )
    args = ap.parse_args()

    wlite_summary = _load_json(resolve_path(args.wlite_summary, REPO_ROOT / "reports"))
    surgical_summary = _load_json(resolve_path(args.health_surgical_summary, REPO_ROOT / "reports"))
    support_summary = _load_json(resolve_path(args.support_summary, REPO_ROOT / "reports"))
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    recommended_label = str(surgical_summary.get("recommended_variant", "fallback_180"))
    recommended_payload = ((surgical_summary.get("variant_payloads", {}) or {}).get(recommended_label, {}) or {})
    raw_payload = ((surgical_summary.get("variant_payloads", {}) or {}).get("raw", {}) or {})

    incumbent_fullval = (wlite_summary.get("incumbent_fullval", {}) or {})
    incumbent_anchor = (wlite_summary.get("incumbent_anchor64", {}) or {})
    raw_wlite_fullval = (wlite_summary.get("wlite_fullval", {}) or {})
    raw_wlite_anchor = (wlite_summary.get("wlite_anchor64", {}) or {})

    recommended_anchor = (recommended_payload.get("anchor64", {}) or {})
    recommended_fullval = (recommended_payload.get("fullval", {}) or {})
    recommended_health_vs_i0 = (recommended_payload.get("fullval_health_vs_i0", {}) or {})
    raw_health_vs_i0 = (raw_payload.get("fullval_health_vs_i0", {}) or {})

    raw_chunk_csv = Path((raw_payload.get("artifacts", {}) or {}).get("fullval_chunk_csv", ""))
    recommended_chunk_csv = Path((recommended_payload.get("artifacts", {}) or {}).get("fullval_chunk_csv", ""))
    incumbent_chunk_csv = REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "diagnostics" / "val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv"

    raw_df = pd.read_csv(raw_chunk_csv)
    recommended_df = pd.read_csv(recommended_chunk_csv)
    incumbent_df = pd.read_csv(incumbent_chunk_csv)
    hard_df = pd.read_csv(hard_ids_csv)
    hard_parent_ids = set(hard_df["oare_id"].fillna("").astype(str)) | set(hard_df["parent_oare_id"].fillna("").astype(str))

    merged = (
        raw_df[
            [
                "oare_id",
                "parent_oare_id",
                "chunk_index",
                "chunk_mode",
                "is_short_aligned",
                "reference",
                "prediction",
            ]
        ]
        .rename(columns={"prediction": "raw_prediction"})
        .merge(
            recommended_df[["oare_id", "prediction"]].rename(columns={"prediction": "recommended_prediction"}),
            on="oare_id",
            how="inner",
        )
        .merge(
            incumbent_df[["oare_id", "prediction"]].rename(columns={"prediction": "incumbent_prediction"}),
            on="oare_id",
            how="inner",
        )
    )
    raw_counts = Counter(raw_df["prediction"].fillna("").astype(str))
    changed = merged.loc[merged["raw_prediction"] != merged["recommended_prediction"]].copy()
    changed["parent_id"] = changed["parent_oare_id"].fillna(changed["oare_id"]).astype(str)
    changed["raw_count"] = changed["raw_prediction"].map(raw_counts).fillna(0).astype(int)
    changed["is_hard_parent"] = changed["parent_id"].isin(hard_parent_ids)

    changed_rows_csv = report_dir / "changed_rows.csv"
    changed.to_csv(changed_rows_csv, index=False)

    repeat_groups = _summarize_repeat_groups(changed)
    write_json(report_dir / "repeat_group_summary.json", repeat_groups)

    changed_subset_metrics = {
        "raw": _changed_subset_metrics(changed, "raw_prediction"),
        recommended_label: _changed_subset_metrics(changed, "recommended_prediction"),
        "incumbent": _changed_subset_metrics(changed, "incumbent_prediction"),
    }

    changed_stats = {
        "changed_rows": int(len(changed)),
        "changed_rows_ratio_pct": round(100.0 * float(len(changed) / len(raw_df)), 4) if len(raw_df) else 0.0,
        "changed_parent_rows": int(changed["parent_id"].nunique()),
        "changed_original_rows": int((changed["chunk_mode"].fillna("none") == "none").sum()),
        "changed_ratio_rows": int((changed["chunk_mode"].fillna("") == "ratio").sum()),
        "changed_short_aligned_rows": int((changed["chunk_mode"].fillna("") == "short_aligned_gale_church").sum()),
        "changed_hard_rows": int(changed["is_hard_parent"].sum()),
        "changed_hard_parents": int(changed.loc[changed["is_hard_parent"], "parent_id"].nunique()),
        "raw_repeat_histogram": {
            str(int(k)): int(v)
            for k, v in sorted(changed["raw_count"].value_counts().to_dict().items())
        },
    }

    recommended_anchor_geom = float(((recommended_anchor.get("reconstructed", {}) or {}).get("metrics", {}) or {}).get("geom", 0.0))
    recommended_fullval_geom = float(((recommended_fullval.get("reconstructed", {}) or {}).get("metrics", {}) or {}).get("geom", 0.0))
    recommended_hard_geom = float(
        ((recommended_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0)
    )

    raw_anchor_geom = float(((raw_payload.get("anchor64", {}) or {}).get("reconstructed", {}) or {}).get("metrics", {}).get("geom", 0.0))
    raw_fullval_geom = float(((raw_payload.get("fullval", {}) or {}).get("reconstructed", {}) or {}).get("metrics", {}).get("geom", 0.0))
    raw_hard_geom = float(((raw_payload.get("fullval", {}) or {}).get("hard_subset", {}) or {}).get("eval_geom", 0.0))

    incumbent_anchor_geom = float(incumbent_anchor.get("eval_geom", 0.0) or 0.0)
    incumbent_fullval_geom = float(incumbent_fullval.get("eval_geom", 0.0) or 0.0)
    incumbent_hard_geom = float(((incumbent_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))

    summary = {
        "status": "manual_promote_recommended"
        if bool(recommended_health_vs_i0.get("no_regression")) and recommended_fullval_geom > incumbent_fullval_geom
        else "review_stop",
        "selected_candidate": recommended_label,
        "candidate_type": "posthoc_incumbent_fallback_on_repeated_generic_chunks",
        "note": "health-safe candidate for promote compare, not a retrained checkpoint",
        "scoreboard": {
            "incumbent": {
                "anchor64_reconstructed_geom": round(incumbent_anchor_geom, 4),
                "fullval_reconstructed_geom": round(incumbent_fullval_geom, 4),
                "hard_geom": round(incumbent_hard_geom, 4),
            },
            "retrieval_wlite_raw": {
                "anchor64_reconstructed_geom": round(raw_anchor_geom, 4),
                "fullval_reconstructed_geom": round(raw_fullval_geom, 4),
                "hard_geom": round(raw_hard_geom, 4),
            },
            recommended_label: {
                "anchor64_reconstructed_geom": round(recommended_anchor_geom, 4),
                "fullval_reconstructed_geom": round(recommended_fullval_geom, 4),
                "hard_geom": round(recommended_hard_geom, 4),
            },
        },
        "deltas": {
            "recommended_vs_incumbent": {
                "anchor64_reconstructed_geom": round(recommended_anchor_geom - incumbent_anchor_geom, 4),
                "fullval_reconstructed_geom": round(recommended_fullval_geom - incumbent_fullval_geom, 4),
                "hard_geom": round(recommended_hard_geom - incumbent_hard_geom, 4),
            },
            "recommended_vs_raw_wlite": {
                "anchor64_reconstructed_geom": round(recommended_anchor_geom - raw_anchor_geom, 4),
                "fullval_reconstructed_geom": round(recommended_fullval_geom - raw_fullval_geom, 4),
                "hard_geom": round(recommended_hard_geom - raw_hard_geom, 4),
            },
        },
        "health": {
            "raw_vs_incumbent": raw_health_vs_i0,
            "recommended_vs_incumbent": recommended_health_vs_i0,
        },
        "changed_rows": changed_stats,
        "changed_subset_metrics": changed_subset_metrics,
        "repeat_group_summary": repeat_groups,
        "support_bundle": {
            "status": str(support_summary.get("status", "")),
            "summary_path": str(resolve_path(args.support_summary, REPO_ROOT / "reports")),
            "cache_hit_stats_json": str(support_summary.get("cache_hit_stats_json", "")),
            "neighbor_quality_audit_json": str(support_summary.get("neighbor_quality_audit_json", "")),
            "latency_report_json": str(support_summary.get("latency_report_json", "")),
            "memory_usage_report_json": str(support_summary.get("memory_usage_report_json", "")),
            "official_like_template_json": str(support_summary.get("official_like_template_json", "")),
        },
        "artifacts": {
            "recommended_fullval_chunk_csv": str(recommended_chunk_csv),
            "recommended_fullval_reconstructed_csv": str(
                ((recommended_payload.get("artifacts", {}) or {}).get("fullval_reconstructed_csv", ""))
            ),
            "raw_fullval_chunk_csv": str(raw_chunk_csv),
            "raw_fullval_reconstructed_csv": str(((raw_payload.get("artifacts", {}) or {}).get("fullval_reconstructed_csv", ""))),
            "changed_rows_csv": str(changed_rows_csv),
            "repeat_group_summary_json": str(report_dir / "repeat_group_summary.json"),
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A2 Promote Compare",
        "",
        f"- status: `{summary['status']}`",
        f"- selected candidate: `{recommended_label}`",
        "- candidate type: post-hoc incumbent fallback on repeated generic chunk outputs",
        "- note: this is a compare candidate, not a new trained checkpoint",
        "",
        "## Scoreboard",
        "",
        f"- incumbent full-val / hard / anchor64: `{incumbent_fullval_geom:.4f} / {incumbent_hard_geom:.4f} / {incumbent_anchor_geom:.4f}`",
        f"- raw W-lite full-val / hard / anchor64: `{raw_fullval_geom:.4f} / {raw_hard_geom:.4f} / {raw_anchor_geom:.4f}`",
        f"- {recommended_label} full-val / hard / anchor64: `{recommended_fullval_geom:.4f} / {recommended_hard_geom:.4f} / {recommended_anchor_geom:.4f}`",
        f"- {recommended_label} vs incumbent full-val delta: `{recommended_fullval_geom - incumbent_fullval_geom:+.4f}`",
        f"- {recommended_label} vs raw W-lite full-val delta: `{recommended_fullval_geom - raw_fullval_geom:+.4f}`",
        "",
        "## Health",
        "",
        f"- raw full-val health no_regression vs incumbent: `{bool(raw_health_vs_i0.get('no_regression'))}`",
        f"- {recommended_label} full-val health no_regression vs incumbent: `{bool(recommended_health_vs_i0.get('no_regression'))}`",
        f"- raw unique delta vs incumbent: `{float(raw_health_vs_i0.get('unique_prediction_ratio_pct', 0.0)):+.4f}`",
        f"- {recommended_label} unique delta vs incumbent: `{float(recommended_health_vs_i0.get('unique_prediction_ratio_pct', 0.0)):+.4f}`",
        f"- raw short delta vs incumbent: `{float(raw_health_vs_i0.get('pred_shorter_than_half_ref_ratio_pct', 0.0)):+.4f}`",
        f"- {recommended_label} short delta vs incumbent: `{float(recommended_health_vs_i0.get('pred_shorter_than_half_ref_ratio_pct', 0.0)):+.4f}`",
        "",
        "## Changed Rows",
        "",
        f"- changed chunk rows: `{changed_stats['changed_rows']}` / `{len(raw_df)}` (`{changed_stats['changed_rows_ratio_pct']:.4f}%`)",
        f"- changed parent rows: `{changed_stats['changed_parent_rows']}`",
        f"- changed original rows: `{changed_stats['changed_original_rows']}`",
        f"- changed ratio rows: `{changed_stats['changed_ratio_rows']}`",
        f"- changed short-aligned rows: `{changed_stats['changed_short_aligned_rows']}`",
        f"- changed hard rows / parents: `{changed_stats['changed_hard_rows']}` / `{changed_stats['changed_hard_parents']}`",
        "",
        "## Local Cost On Changed Rows",
        "",
        f"- raw changed-subset geom: `{changed_subset_metrics['raw']['geom']:.4f}`",
        f"- {recommended_label} changed-subset geom: `{changed_subset_metrics[recommended_label]['geom']:.4f}`",
        f"- incumbent changed-subset geom: `{changed_subset_metrics['incumbent']['geom']:.4f}`",
        "- interpretation: health is recovered by replacing a small set of high-frequency generic chunk outputs, but those changed rows are locally weaker than raw W-lite.",
        "",
        "## Repeat Groups",
        "",
    ]
    for record in repeat_groups:
        lines.append(
            f"- raw_count=`{record['raw_count']}` changed_rows=`{record['changed_rows']}` changed_parents=`{record['changed_parents']}` text=`{record['raw_prediction']}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- use `{recommended_label}` as the `A2_F_review` promote candidate",
            "- keep raw W-lite as score ceiling reference, but do not promote it directly while health gate remains red",
            "- do not treat this as a new trained model; it is a selective fallback post-process candidate",
        ]
    )
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")

    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
