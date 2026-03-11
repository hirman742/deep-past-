#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from taskform_a2_a1_flow import REPO_ROOT, SCRIPTS_DIR, _load_json, _resolve_path, _run
from taskform_phase12_common import safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import _compare_output_health


def _decode_json_path(checkpoint_dir: Path, tag: str) -> Path:
    run_dir = checkpoint_dir.parent
    return run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"


def _diag_json_path(checkpoint_dir: Path, tag: str) -> Path:
    run_dir = checkpoint_dir.parent
    return run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"


def _run_decode_and_diagnose(
    *,
    python_exec: str,
    cfg_path: Path,
    checkpoint_dir: Path,
    fold: int,
    tag: str,
    max_rows: int,
    rk_enabled: bool,
    rk_bias_strength: float,
    rk_max_bias_steps: int,
    rk_report_dir: Path,
) -> dict[str, Any]:
    decode_json = _decode_json_path(checkpoint_dir, tag)
    diag_json = _diag_json_path(checkpoint_dir, tag)
    reused_decode = decode_json.exists()
    reused_diag = diag_json.exists()

    if not reused_decode:
        cmd = [
            python_exec,
            str(SCRIPTS_DIR / "eval_decode_grid.py"),
            "--config",
            str(cfg_path),
            "--fold",
            str(int(fold)),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--tag",
            tag,
            "--beams",
            "4",
            "--length-penalties",
            "0.7",
            "--no-repeat-ngram-sizes",
            "0",
            "--min-new-tokens-list",
            "0",
            "--max-new-tokens-list",
            "384",
            "--predict-batch-size",
            "16",
            "--max-val-samples",
            str(int(max_rows)),
        ]
        if rk_enabled:
            cmd.extend(
                [
                    "--rk-enabled",
                    "--rk-k",
                    "8",
                    "--rk-raw-pool-k",
                    "48",
                    "--rk-bias-strength",
                    str(float(rk_bias_strength)),
                    "--rk-max-bias-steps",
                    str(int(rk_max_bias_steps)),
                    "--rk-report-dir",
                    str(rk_report_dir),
                ]
            )
        _run(cmd)

    if not reused_diag:
        cmd = [
            python_exec,
            str(SCRIPTS_DIR / "diagnose_val_outputs.py"),
            "--config",
            str(cfg_path),
            "--fold",
            str(int(fold)),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--tag",
            tag,
            "--max-rows",
            str(int(max_rows)),
            "--predict-batch-size",
            "16",
            "--sample-size",
            str(min(50, int(max_rows))),
            "--num-beams",
            "4",
            "--length-penalty",
            "0.7",
            "--no-repeat-ngram-size",
            "0",
            "--min-new-tokens",
            "0",
            "--max-new-tokens",
            "384",
        ]
        if rk_enabled:
            cmd.extend(
                [
                    "--rk-enabled",
                    "--rk-k",
                    "8",
                    "--rk-raw-pool-k",
                    "48",
                    "--rk-bias-strength",
                    str(float(rk_bias_strength)),
                    "--rk-max-bias-steps",
                    str(int(rk_max_bias_steps)),
                    "--rk-report-dir",
                    str(rk_report_dir),
                ]
            )
        _run(cmd)

    decode_payload = _load_json(decode_json)
    diag_payload = _load_json(diag_json)
    return {
        "tag": tag,
        "decode_json": str(decode_json),
        "diagnose_json": str(diag_json),
        "reused_decode": bool(reused_decode),
        "reused_diagnose": bool(reused_diag),
        "decode": decode_payload,
        "diagnose": diag_payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="reports/taskform_winner_a2_retrieval_raw_longtrain_20260311/generated_configs/winner_retrieval_raw_wlite_longtrain.yaml",
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_RAW_LONGTRAIN_20260311_fold0/best_model",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--rk-bias-strength", type=float, default=0.5)
    ap.add_argument("--rk-max-bias-steps", type=int, default=32)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--report-dir", default="reports/taskform_winner_rk_truehook_revisit_smoke_20260311")
    args = ap.parse_args()

    python_exec = str(REPO_ROOT / ".venv-deeppast" / "bin" / "python")
    cfg_path = _resolve_path(
        args.base_config,
        REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_raw_longtrain_20260311" / "generated_configs" / "winner_retrieval_raw_wlite_longtrain.yaml",
    )
    checkpoint_dir = _resolve_path(
        args.checkpoint_dir,
        REPO_ROOT / "runs" / "TASKFORM_WINNER_A2_RETRIEVAL_RAW_LONGTRAIN_20260311_fold0" / "best_model",
    )
    report_dir = _resolve_path(
        args.report_dir,
        REPO_ROOT / "reports" / "taskform_winner_rk_truehook_revisit_smoke_20260311",
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    rk_report_root = report_dir / "rk_hook_artifacts"
    rk_report_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    baseline_tag = f"winner_rk_truehook_revisit_baseline_smoke{int(args.max_rows)}_{safe_text(args.run_suffix)}"
    hook_tag = f"winner_rk_truehook_revisit_smoke{int(args.max_rows)}_a05s32_{safe_text(args.run_suffix)}"

    baseline = _run_decode_and_diagnose(
        python_exec=python_exec,
        cfg_path=cfg_path,
        checkpoint_dir=checkpoint_dir,
        fold=int(args.fold),
        tag=baseline_tag,
        max_rows=int(args.max_rows),
        rk_enabled=False,
        rk_bias_strength=0.0,
        rk_max_bias_steps=0,
        rk_report_dir=rk_report_root / "baseline",
    )
    hook = _run_decode_and_diagnose(
        python_exec=python_exec,
        cfg_path=cfg_path,
        checkpoint_dir=checkpoint_dir,
        fold=int(args.fold),
        tag=hook_tag,
        max_rows=int(args.max_rows),
        rk_enabled=True,
        rk_bias_strength=float(args.rk_bias_strength),
        rk_max_bias_steps=int(args.rk_max_bias_steps),
        rk_report_dir=rk_report_root / "weak_truehook",
    )

    baseline_decode_geom = float((baseline.get("decode", {}) or {}).get("eval_geom", 0.0) or 0.0)
    hook_decode_geom = float((hook.get("decode", {}) or {}).get("eval_geom", 0.0) or 0.0)
    baseline_diag = baseline.get("diagnose", {}) or {}
    hook_diag = hook.get("diagnose", {}) or {}
    baseline_recon_metrics = ((baseline_diag.get("reconstructed", {}) or {}).get("metrics", {}) or {})
    hook_recon_metrics = ((hook_diag.get("reconstructed", {}) or {}).get("metrics", {}) or {})
    baseline_recon_geom = float(baseline_recon_metrics.get("geom", 0.0) or 0.0)
    hook_recon_geom = float(hook_recon_metrics.get("geom", 0.0) or 0.0)
    baseline_health = ((baseline_diag.get("reconstructed", {}) or {}).get("output_health", {}) or {})
    hook_health = ((hook_diag.get("reconstructed", {}) or {}).get("output_health", {}) or {})
    health_delta = _compare_output_health(hook_health, baseline_health)

    status = "parked_negative_smoke"
    if hook_recon_geom > baseline_recon_geom and bool(health_delta.get("no_regression", False)):
        status = "review_reopen_candidate"

    summary = {
        "status": status,
        "reason": "Weak true-hook revisit on stronger retrieval state only; non-blocking low-priority smoke",
        "base_config": str(cfg_path),
        "checkpoint_dir": str(checkpoint_dir),
        "max_rows": int(args.max_rows),
        "baseline": baseline,
        "weak_truehook": hook,
        "comparisons": {
            "decode_geom_delta_vs_baseline": round(hook_decode_geom - baseline_decode_geom, 4),
            "reconstructed_geom_delta_vs_baseline": round(hook_recon_geom - baseline_recon_geom, 4),
            "reconstructed_health_delta_vs_baseline": health_delta,
        },
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    write_json(report_dir / "summary.json", summary)

    report_lines = [
        "# Winner RK True-Hook Revisit Smoke",
        "",
        f"- status: `{status}`",
        "- reason: stronger retrieval state revisit only; low-priority and non-blocking",
        f"- checkpoint_dir: `{checkpoint_dir}`",
        f"- max_rows: `{int(args.max_rows)}`",
        "",
        "## baseline",
        "",
        f"- decode geom: `{baseline_decode_geom:.4f}`",
        f"- reconstructed geom: `{baseline_recon_geom:.4f}`",
        "",
        "## weak_truehook",
        "",
        f"- rk_bias_strength: `{float(args.rk_bias_strength):.2f}`",
        f"- rk_max_bias_steps: `{int(args.rk_max_bias_steps)}`",
        f"- decode geom: `{hook_decode_geom:.4f}`",
        f"- reconstructed geom: `{hook_recon_geom:.4f}`",
        f"- reconstructed delta vs baseline: `{(hook_recon_geom - baseline_recon_geom):+.4f}`",
        f"- health no_regression vs baseline: `{bool(health_delta.get('no_regression', False))}`",
        "",
    ]
    write_text(report_dir / "gate_report.md", "\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
