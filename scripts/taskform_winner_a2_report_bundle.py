from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from taskform_phase12_common import resolve_path, write_json, write_text
from taskform_winner_a2_retrieval_eval import _load_json
from taskform_winner_a2_retrieval_wlite_eval import (
    REPO_ROOT,
    _build_cache_hit_stats,
    _build_neighbor_quality_audit,
)


def _sum_file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _write_report(path: Path, lines: list[str]) -> None:
    write_text(path, "\n".join(lines) + "\n")


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wlite-config",
        default="reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml",
    )
    ap.add_argument("--retrieval-report-dir", default="reports/taskform_winner_a2_retrieval_20260310")
    ap.add_argument("--smoke-summary", default="reports/taskform_winner_a2_retrieval_eval_20260310/summary.json")
    ap.add_argument("--rk-summary", default="reports/taskform_winner_a2_retrieval_probe_20260310/summary.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_support_20260310")
    args = ap.parse_args()

    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_support_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)
    wlite_cfg_path = resolve_path(args.wlite_config, REPO_ROOT / "reports")
    retrieval_report_dir = resolve_path(args.retrieval_report_dir, REPO_ROOT / "reports")
    smoke_summary_path = resolve_path(args.smoke_summary, REPO_ROOT / "reports")
    rk_summary_path = resolve_path(args.rk_summary, REPO_ROOT / "reports")

    cache_hit_stats = _build_cache_hit_stats(wlite_cfg_path, int(args.fold))
    write_json(report_dir / "retrieval_cache_hit_stats.json", cache_hit_stats)

    neighbor_quality_summary, neighbor_quality_md = _build_neighbor_quality_audit(retrieval_report_dir)
    write_json(report_dir / "nearest_neighbor_quality_audit.json", neighbor_quality_summary)
    write_text(report_dir / "nearest_neighbor_quality_audit.md", neighbor_quality_md)

    smoke_summary = _load_json(smoke_summary_path)
    rk_summary = _load_json(rk_summary_path)

    incumbent_anchor = smoke_summary.get("incumbent_anchor64", {}) or {}
    retrieval_anchor = (smoke_summary.get("retrieval_top1", {}) or {}).get("anchor64", {}) or {}
    wlite_anchor_path = (
        REPO_ROOT
        / "runs"
        / "TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0"
        / "diagnostics"
        / "decode_grid_best_taskform_winner_a2_r1_wlite_anchor64_20260310.json"
    )
    wlite_fullval_path = (
        REPO_ROOT
        / "runs"
        / "TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0"
        / "diagnostics"
        / "decode_grid_best_taskform_winner_a2_r1_wlite_fullval_20260310.json"
    )
    incumbent_fullval_path = (
        REPO_ROOT
        / "runs"
        / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0"
        / "diagnostics"
        / "decode_grid_best_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json"
    )
    wlite_anchor = _load_json(wlite_anchor_path) if wlite_anchor_path.exists() else {}
    wlite_fullval = _load_json(wlite_fullval_path) if wlite_fullval_path.exists() else {}
    incumbent_fullval = _load_json(incumbent_fullval_path) if incumbent_fullval_path.exists() else {}

    latency_report = {
        "status": "partial_ready" if wlite_fullval else "anchor_only",
        "anchor64": {
            "incumbent_elapsed_seconds": float(incumbent_anchor.get("elapsed_seconds", 0.0) or 0.0),
            "retrieval_smoke_elapsed_seconds": float(retrieval_anchor.get("elapsed_seconds", 0.0) or 0.0),
            "retrieval_wlite_elapsed_seconds": float(wlite_anchor.get("elapsed_seconds", 0.0) or 0.0),
        },
        "fullval": {
            "incumbent_elapsed_seconds": float(incumbent_fullval.get("elapsed_seconds", 0.0) or 0.0),
            "retrieval_wlite_elapsed_seconds": float(wlite_fullval.get("elapsed_seconds", 0.0) or 0.0),
        },
    }
    anchor_inc = float(latency_report["anchor64"]["incumbent_elapsed_seconds"] or 0.0)
    anchor_wlite = float(latency_report["anchor64"]["retrieval_wlite_elapsed_seconds"] or 0.0)
    full_inc = float(latency_report["fullval"]["incumbent_elapsed_seconds"] or 0.0)
    full_wlite = float(latency_report["fullval"]["retrieval_wlite_elapsed_seconds"] or 0.0)
    latency_report["ratios"] = {
        "anchor64_wlite_vs_incumbent": round(anchor_wlite / anchor_inc, 4) if anchor_inc > 0 else None,
        "fullval_wlite_vs_incumbent": round(full_wlite / full_inc, 4) if full_inc > 0 else None,
    }
    write_json(report_dir / "latency_report.json", latency_report)
    _write_report(
        report_dir / "latency_report.md",
        [
            "# A2 Latency Report",
            "",
            f"- anchor64 incumbent seconds: `{anchor_inc:.4f}`",
            f"- anchor64 retrieval W-lite seconds: `{anchor_wlite:.4f}`",
            f"- anchor64 ratio: `{latency_report['ratios']['anchor64_wlite_vs_incumbent']}`",
            f"- full-val incumbent seconds: `{full_inc:.4f}`",
            f"- full-val retrieval W-lite seconds: `{full_wlite:.4f}`",
            f"- full-val ratio: `{latency_report['ratios']['fullval_wlite_vs_incumbent']}`",
        ],
    )

    checkpoint_dir = (
        REPO_ROOT / "runs" / "TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0" / "best_model"
    )
    processed_dir = REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0"
    memory_report = {
        "status": "estimate_ready",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_size_bytes": _sum_file_bytes(checkpoint_dir),
        "processed_dir": str(processed_dir),
        "processed_size_bytes": _sum_file_bytes(processed_dir),
        "rk_probe_datastore": ((rk_summary.get("rk", {}) or {}).get("datastore", {}) or {}),
        "rk_probe_artifacts": ((rk_summary.get("rk", {}) or {}).get("artifacts", {}) or {}),
    }
    write_json(report_dir / "memory_usage_report.json", memory_report)
    _write_report(
        report_dir / "memory_usage_report.md",
        [
            "# A2 Memory Report",
            "",
            f"- checkpoint size bytes: `{memory_report['checkpoint_size_bytes']}`",
            f"- processed dir size bytes: `{memory_report['processed_size_bytes']}`",
            f"- RK datastore total target tokens: `{memory_report['rk_probe_datastore'].get('total_target_tokens')}`",
            f"- RK hidden size: `{memory_report['rk_probe_datastore'].get('hidden_size')}`",
            f"- RK estimated fp16 key store GiB: `{memory_report['rk_probe_datastore'].get('estimated_fp16_key_store_gib')}`",
        ],
    )

    official_probe_dir = report_dir / "official_metric_probe"
    official_probe_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "taskform_official_metric_probe.py"),
            "--out-dir",
            str(official_probe_dir),
        ]
    )
    official_probe = _load_json(official_probe_dir / "official_metric_probe.json")
    official_like_template = {
        "status": "template_ready",
        "note": "official-like remains local proxy until bridge lands",
        "probe": official_probe,
    }
    write_json(report_dir / "official_like_template.json", official_like_template)
    _write_report(
        report_dir / "official_like_template.md",
        [
            "# Official-like Template",
            "",
            f"- status: `{official_like_template['status']}`",
            "- note: official-like remains local proxy until bridge lands",
            f"- probe status: `{official_probe.get('status')}`",
            f"- recommendation: {official_probe.get('recommendation', '')}",
        ],
    )

    summary = {
        "status": "ready_partial_bundle",
        "cache_hit_stats_json": str(report_dir / "retrieval_cache_hit_stats.json"),
        "neighbor_quality_audit_json": str(report_dir / "nearest_neighbor_quality_audit.json"),
        "latency_report_json": str(report_dir / "latency_report.json"),
        "memory_usage_report_json": str(report_dir / "memory_usage_report.json"),
        "official_like_template_json": str(report_dir / "official_like_template.json"),
    }
    write_json(report_dir / "summary.json", summary)
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
