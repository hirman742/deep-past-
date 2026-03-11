from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from taskform_phase12_common import resolve_path, write_json, write_text
from taskform_winner_a2_retrieval_eval import _load_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path


def _as_float(value: Any) -> float:
    return float(value or 0.0)


def _report_lines(summary: dict[str, Any]) -> list[str]:
    score = summary["scoreboard"]
    deltas = summary["deltas"]
    health = summary["health"]
    official = summary["official_like"]
    support = summary["support_bundle"]
    changed = summary["changed_rows"]
    artifacts = summary["artifacts"]
    lines = [
        "# A2 Freeze Bundle",
        "",
        f"- status: `{summary['status']}`",
        f"- selected candidate: `{summary['selected_candidate']}`",
        f"- candidate type: `{summary['candidate_type']}`",
        f"- freeze date: `{summary['freeze_date_utc']}`",
        "",
        "## Decision",
        "",
        "- promote compare candidate is `fallback_180`",
        "- raw retrieval W-lite remains score ceiling reference only",
        "- this is a post-hoc health-safe candidate, not a new retrained checkpoint",
        "",
        "## Scoreboard",
        "",
        f"- incumbent full-val / hard / anchor64: `{score['incumbent']['fullval_reconstructed_geom']:.4f} / {score['incumbent']['hard_geom']:.4f} / {score['incumbent']['anchor64_reconstructed_geom']:.4f}`",
        f"- raw W-lite full-val / hard / anchor64: `{score['retrieval_wlite_raw']['fullval_reconstructed_geom']:.4f} / {score['retrieval_wlite_raw']['hard_geom']:.4f} / {score['retrieval_wlite_raw']['anchor64_reconstructed_geom']:.4f}`",
        f"- frozen candidate full-val / hard / anchor64: `{score['fallback_180']['fullval_reconstructed_geom']:.4f} / {score['fallback_180']['hard_geom']:.4f} / {score['fallback_180']['anchor64_reconstructed_geom']:.4f}`",
        f"- frozen vs incumbent full-val delta: `{deltas['recommended_vs_incumbent']['fullval_reconstructed_geom']:+.4f}`",
        f"- frozen vs raw W-lite full-val delta: `{deltas['recommended_vs_raw_wlite']['fullval_reconstructed_geom']:+.4f}`",
        "",
        "## Health",
        "",
        f"- no_regression vs incumbent: `{health['recommended_vs_incumbent']['no_regression']}`",
        f"- unique delta vs incumbent: `{health['recommended_vs_incumbent']['unique_prediction_ratio_pct']:+.4f}`",
        f"- short delta vs incumbent: `{health['recommended_vs_incumbent']['pred_shorter_than_half_ref_ratio_pct']:+.4f}`",
        f"- empty delta vs incumbent: `{health['recommended_vs_incumbent']['empty_prediction_ratio_pct']:+.4f}`",
        "",
        "## Official-like",
        "",
        f"- status: `{official['status']}`",
        f"- bridge probe status: `{official['probe_status']}`",
        f"- note: {official['note']}",
        f"- recommendation: {official['recommendation']}",
        "",
        "## Support",
        "",
        f"- support bundle status: `{support['status']}`",
        f"- cache hit stats: `{support['cache_hit_stats_json']}`",
        f"- neighbor quality audit: `{support['neighbor_quality_audit_json']}`",
        f"- latency report: `{support['latency_report_json']}`",
        f"- memory report: `{support['memory_usage_report_json']}`",
        "",
        "## Changed Rows",
        "",
        f"- changed chunk rows: `{changed['changed_rows']}` / `1225` (`{changed['changed_rows_ratio_pct']:.4f}%`)",
        f"- changed parents: `{changed['changed_parent_rows']}`",
        f"- changed original rows: `{changed['changed_original_rows']}`",
        f"- changed hard rows / parents: `{changed['changed_hard_rows']}` / `{changed['changed_hard_parents']}`",
        "",
        "## Frozen Artifacts",
        "",
        f"- full-val chunk csv: `{artifacts['recommended_fullval_chunk_csv']}`",
        f"- full-val reconstructed csv: `{artifacts['recommended_fullval_reconstructed_csv']}`",
        f"- raw full-val chunk csv: `{artifacts['raw_fullval_chunk_csv']}`",
        f"- raw full-val reconstructed csv: `{artifacts['raw_fullval_reconstructed_csv']}`",
        f"- changed rows csv: `{artifacts['changed_rows_csv']}`",
        f"- repeat summary json: `{artifacts['repeat_group_summary_json']}`",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--promote-summary",
        default="reports/taskform_winner_a2_promote_compare_20260310/summary.json",
    )
    ap.add_argument(
        "--support-summary",
        default="reports/taskform_winner_a2_support_20260310/summary.json",
    )
    ap.add_argument(
        "--report-dir",
        default="reports/taskform_winner_a2_freeze_20260310",
    )
    args = ap.parse_args()

    promote_summary_path = resolve_path(args.promote_summary, REPO_ROOT / "reports")
    support_summary_path = resolve_path(args.support_summary, REPO_ROOT / "reports")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    promote = _load_json(_must_exist(promote_summary_path))
    support = _load_json(_must_exist(support_summary_path))

    official_like_path = Path(
        (promote.get("support_bundle", {}) or {}).get("official_like_template_json", "")
    )
    official_like = _load_json(_must_exist(official_like_path)) if official_like_path.as_posix() else {}
    probe = (official_like.get("probe", {}) or {})

    artifacts = promote.get("artifacts", {}) or {}
    for key in (
        "recommended_fullval_chunk_csv",
        "recommended_fullval_reconstructed_csv",
        "raw_fullval_chunk_csv",
        "raw_fullval_reconstructed_csv",
        "changed_rows_csv",
        "repeat_group_summary_json",
    ):
        _must_exist(Path(artifacts[key]))

    summary = {
        "status": "candidate_frozen_manual_promote_recommended",
        "selected_candidate": promote.get("selected_candidate", "fallback_180"),
        "candidate_type": promote.get(
            "candidate_type", "posthoc_incumbent_fallback_on_repeated_generic_chunks"
        ),
        "freeze_date_utc": "2026-03-10",
        "scoreboard": promote.get("scoreboard", {}),
        "deltas": promote.get("deltas", {}),
        "health": promote.get("health", {}),
        "changed_rows": promote.get("changed_rows", {}),
        "official_like": {
            "status": official_like.get("status", "template_ready"),
            "note": official_like.get("note", "official-like remains local proxy until bridge lands"),
            "probe_status": probe.get("status", "unknown"),
            "recommendation": probe.get("recommendation", ""),
            "probe_json": str(official_like_path.parent / "official_metric_probe" / "official_metric_probe.json"),
        },
        "support_bundle": {
            "status": support.get("status", "unknown"),
            "cache_hit_stats_json": support.get("cache_hit_stats_json", ""),
            "neighbor_quality_audit_json": support.get("neighbor_quality_audit_json", ""),
            "latency_report_json": support.get("latency_report_json", ""),
            "memory_usage_report_json": support.get("memory_usage_report_json", ""),
            "official_like_template_json": support.get("official_like_template_json", ""),
        },
        "artifacts": artifacts,
        "frozen_candidate": {
            "fullval_reconstructed_geom": _as_float(
                ((promote.get("scoreboard", {}) or {}).get("fallback_180", {}) or {}).get(
                    "fullval_reconstructed_geom"
                )
            ),
            "hard_geom": _as_float(
                ((promote.get("scoreboard", {}) or {}).get("fallback_180", {}) or {}).get("hard_geom")
            ),
            "anchor64_reconstructed_geom": _as_float(
                ((promote.get("scoreboard", {}) or {}).get("fallback_180", {}) or {}).get(
                    "anchor64_reconstructed_geom"
                )
            ),
        },
        "upstream": {
            "promote_summary_json": str(promote_summary_path),
            "support_summary_json": str(support_summary_path),
        },
    }

    write_json(report_dir / "manifest.json", summary)
    write_json(report_dir / "summary.json", summary)
    write_text(report_dir / "gate_report.md", "\n".join(_report_lines(summary)) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
