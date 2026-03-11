from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from taskform_phase12_common import resolve_path, write_json, write_text
from taskform_winner_a1_continue_probe_flow import (
    _compare_output_health,
    _compute_hard_subset_metrics,
    _evaluate_candidate,
    _load_anchor_reference,
    _load_json,
    _train_if_needed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_wlite_cfg(base_config_path: Path, label: str, report_dir: Path) -> Path:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    cfg["name"] = f"taskform_winner_a1r_wlite_{label}"
    paths_cfg = (cfg.get("paths", {}) or {}).copy()
    paths_cfg["run_dir"] = f"runs/TASKFORM_WINNER_A1R_WLITE_{label.upper()}_20260310"
    cfg["paths"] = paths_cfg
    out_dir = report_dir / "generated_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"taskform_winner_a1r_wlite_{label}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-summary", default="reports/taskform_winner_a1_continue_build_20260310/summary.json")
    ap.add_argument("--probe-summary", default="reports/taskform_winner_a1_continue_probe_20260310/summary.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--predict-batch-size", type=int, default=16)
    ap.add_argument("--anchor-samples", type=int, default=64)
    ap.add_argument("--anchor-gate", type=float, default=0.15)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_continue_wlite_20260310")
    args = ap.parse_args()

    build_summary = _load_json(resolve_path(args.build_summary, REPO_ROOT / "reports"))
    probe_summary = _load_json(resolve_path(args.probe_summary, REPO_ROOT / "reports"))
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")

    best_label = str(probe_summary.get("best_ratio_label", "internal_only_matched"))
    best_status = str(probe_summary.get("best_ratio_status", "control_only"))
    refs = _load_anchor_reference()

    if best_status != "review_to_wlite" or best_label == "internal_only_matched":
        summary = {
            "status": "skipped_no_probe_gate",
            "reason": "probe best did not pass review_to_wlite gate",
            "best_ratio_label": best_label,
            "best_ratio_status": best_status,
            "probe_summary_path": str(resolve_path(args.probe_summary, REPO_ROOT / "reports")),
        }
        write_json(report_dir / "summary.json", summary)
        write_text(
            report_dir / "gate_report.md",
            "# A1 Continue W-lite\n\n- status: `skipped_no_probe_gate`\n- reason: probe best did not pass `review_to_wlite`\n",
        )
        print(f"OK: wrote {report_dir / 'summary.json'}")
        return

    build_map = (build_summary.get("builds", {}) or {})
    control_cfg_path = Path(str((build_map.get("internal_only_matched", {}) or {}).get("config_path", "")))
    best_cfg_path = Path(str((build_map.get(best_label, {}) or {}).get("config_path", "")))
    if not control_cfg_path.exists() or not best_cfg_path.exists():
        raise FileNotFoundError("Missing continue config paths from build summary")

    control_w_cfg = _write_wlite_cfg(control_cfg_path, "internal_only", report_dir)
    best_w_cfg = _write_wlite_cfg(best_cfg_path, best_label, report_dir)

    _train_if_needed(control_w_cfg, int(args.fold), int(args.max_steps), int(args.eval_steps))
    _train_if_needed(best_w_cfg, int(args.fold), int(args.max_steps), int(args.eval_steps))

    control_eval = _evaluate_candidate(
        label="internal_only_wlite",
        config_path=control_w_cfg,
        fold=int(args.fold),
        hard_ids_csv=hard_ids_csv,
        tag="taskform_winner_a1r_internal_only_wlite_anchor64_20260310",
        predict_batch_size=int(args.predict_batch_size),
        anchor_samples=int(args.anchor_samples),
        max_steps=int(args.max_steps),
        eval_steps=int(args.eval_steps),
    )
    best_eval = _evaluate_candidate(
        label=f"{best_label}_wlite",
        config_path=best_w_cfg,
        fold=int(args.fold),
        hard_ids_csv=hard_ids_csv,
        tag=f"taskform_winner_a1r_{best_label}_wlite_anchor64_20260310",
        predict_batch_size=int(args.predict_batch_size),
        anchor_samples=int(args.anchor_samples),
        max_steps=int(args.max_steps),
        eval_steps=int(args.eval_steps),
    )

    anchor_delta = round(
        float(best_eval["anchor64"]["eval_geom"]) - float(control_eval["anchor64"]["eval_geom"]),
        4,
    )
    hard_delta = round(
        float(best_eval["hard_subset"]["eval_geom"]) - float(control_eval["hard_subset"]["eval_geom"]),
        4,
    )
    health_delta = _compare_output_health(best_eval, control_eval)
    recon_health_delta = _compare_output_health(
        {"output_health": best_eval.get("reconstructed_health", {}) or {}},
        {"output_health": control_eval.get("reconstructed_health", {}) or {}},
    )
    status = "review_stop"
    if (
        anchor_delta >= float(args.anchor_gate)
        and hard_delta >= 0.0
        and bool(health_delta["no_regression"])
        and bool(recon_health_delta["no_regression"])
    ):
        status = "proceed_fullval"

    summary = {
        "status": status,
        "reason": "matched long compare at 400 steps from shared upstream raw W-lite adapter",
        "best_ratio_label": best_label,
        "best_ratio_status_from_probe": best_status,
        "control": control_eval,
        "candidate": best_eval,
        "comparisons": {
            "delta_geom_vs_control_anchor64": anchor_delta,
            "delta_geom_vs_control_hard_subset": hard_delta,
            "delta_geom_vs_incumbent_anchor64": round(
                float(best_eval["anchor64"]["eval_geom"]) - float((refs.get("incumbent_anchor64", {}) or {}).get("eval_geom", 0.0)),
                4,
            ),
            "delta_geom_vs_frozen_candidate_anchor64": round(
                float(best_eval["anchor64"]["eval_geom"]) - float((refs.get("frozen_candidate", {}) or {}).get("anchor64_reconstructed_geom", 0.0)),
                4,
            ),
            "health_delta_vs_control": health_delta,
            "reconstructed_health_delta_vs_control": recon_health_delta,
        },
        "incumbent_anchor64": refs.get("incumbent_anchor64", {}) or {},
        "frozen_candidate": refs.get("frozen_candidate", {}) or {},
        "artifacts": {
            "control_config_path": str(control_w_cfg),
            "candidate_config_path": str(best_w_cfg),
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A1 Continue W-lite",
        "",
        f"- status: `{status}`",
        f"- best ratio label: `{best_label}`",
        f"- control anchor64 geom: `{float(control_eval['anchor64']['eval_geom']):.4f}`",
        f"- candidate anchor64 geom: `{float(best_eval['anchor64']['eval_geom']):.4f}`",
        f"- candidate hard geom: `{float(best_eval['hard_subset']['eval_geom']):.4f}`",
        f"- delta anchor vs control: `{anchor_delta:.4f}`",
        f"- delta hard vs control: `{hard_delta:.4f}`",
        f"- health no_regression: `{bool(health_delta['no_regression'])}`",
        f"- reconstructed health no_regression: `{bool(recon_health_delta['no_regression'])}`",
        "",
    ]
    write_text(report_dir / "gate_report.md", "\n".join(lines))
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
