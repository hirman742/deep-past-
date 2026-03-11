#!/usr/bin/env python3
from __future__ import annotations

import copy
import time
import traceback
from pathlib import Path
from typing import Any

from taskform_a2_a1_flow import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _compute_hard_subset_metrics,
    _ensure_incumbent_anchor,
    _load_json,
    _load_yaml,
    _materialize_cfg,
    _resolve_path,
    _run,
    _run_candidate_probe,
)
from taskform_phase12_common import safe_text, write_json, write_text


def _write_status(path: Path, *, stage: str, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def _load_frozen_candidate() -> dict[str, Any]:
    frozen_path = REPO_ROOT / "reports" / "taskform_winner_a2_freeze_20260310" / "summary.json"
    if not frozen_path.exists():
        return {}
    payload = _load_json(frozen_path)
    frozen = ((payload.get("scoreboard", {}) or {}).get("fallback_180", {}) or {})
    return {
        "source": str(frozen_path),
        "anchor64_reconstructed_geom": float(frozen.get("anchor64_reconstructed_geom", 0.0) or 0.0),
        "fullval_reconstructed_geom": float(frozen.get("fullval_reconstructed_geom", 0.0) or 0.0),
        "hard_geom": float(frozen.get("hard_geom", 0.0) or 0.0),
        "status": safe_text(frozen.get("status", "")),
    }


def _compare_output_health(candidate_health: dict[str, Any], baseline_health: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "empty_prediction_ratio_pct",
        "copy_source_ratio_pct",
        "pred_shorter_than_half_ref_ratio_pct",
        "unique_prediction_ratio_pct",
        "has_bad_token_regex_ratio_pct",
        "exact_extra_id_0_ratio_pct",
        "repeat_prediction_ratio_pct",
        "internal_repeat_trigram_ratio_pct",
    ]
    deltas = {
        key: round(float(candidate_health.get(key, 0.0) or 0.0) - float(baseline_health.get(key, 0.0) or 0.0), 4)
        for key in keys
    }
    no_regression = (
        float(candidate_health.get("empty_prediction_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("empty_prediction_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("copy_source_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("copy_source_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("pred_shorter_than_half_ref_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("pred_shorter_than_half_ref_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("has_bad_token_regex_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("has_bad_token_regex_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("repeat_prediction_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("repeat_prediction_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("internal_repeat_trigram_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("internal_repeat_trigram_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("unique_prediction_ratio_pct", 0.0) or 0.0)
        >= float(baseline_health.get("unique_prediction_ratio_pct", 0.0) or 0.0)
    )
    deltas["no_regression"] = bool(no_regression)
    return deltas


def _build_combo_if_needed(
    *,
    python_exec: str,
    base_cfg_path: Path,
    combo_base_processed_dir: Path,
    combo_processed_dir: Path,
    combo_build_report_dir: Path,
    fold: int,
) -> dict[str, Any]:
    summary_path = combo_build_report_dir / "summary.json"
    if (combo_processed_dir / "train_proc.csv").exists() and (combo_processed_dir / "folds.csv").exists() and summary_path.exists():
        payload = _load_json(summary_path)
        payload["reused_existing_build"] = True
        return payload

    combo_build_report_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            python_exec,
            str(SCRIPTS_DIR / "taskform_build_retrieval_hint_processed.py"),
            "--base-config",
            str(base_cfg_path),
            "--base-processed-dir",
            str(combo_base_processed_dir),
            "--fold",
            str(int(fold)),
            "--top-k",
            "1",
            "--output-dir",
            str(combo_processed_dir),
            "--report-dir",
            str(combo_build_report_dir),
        ]
    )
    payload = _load_json(summary_path)
    payload["reused_existing_build"] = False
    return payload


def _evaluate_candidate(
    *,
    python_exec: str,
    base_cfg: dict[str, Any],
    generated_cfg_dir: Path,
    base_checkpoint_dir: Path,
    processed_dir: Path,
    run_name: str,
    label: str,
    tag: str,
    fold: int,
    max_steps: int,
    eval_steps: int,
    hard_ids_csv: Path,
) -> dict[str, Any]:
    cfg_path = generated_cfg_dir / f"{run_name.lower()}.yaml"
    cfg = _materialize_cfg(
        base_cfg=copy.deepcopy(base_cfg),
        output_path=cfg_path,
        processed_dir=processed_dir,
        run_dir=f"runs/{run_name}",
    )
    probe = _run_candidate_probe(
        python_exec=python_exec,
        cfg_path=cfg_path,
        cfg=cfg,
        init_adapter_dir=base_checkpoint_dir,
        fold=int(fold),
        max_steps=int(max_steps),
        eval_steps=int(eval_steps),
        tag=tag,
    )
    diag_payload = _load_json(Path(str(probe["diagnose_summary_path"])))
    reconstructed = (diag_payload.get("reconstructed", {}) or {})
    reconstructed_csv = Path(str(((reconstructed.get("artifacts", {}) or {}).get("reconstructed_csv", ""))))
    run_summary = _load_json(Path(str(probe["run_summary_path"])))
    probe["label"] = label
    probe["processed_dir"] = str(processed_dir)
    probe["reconstructed_health"] = (reconstructed.get("output_health", {}) or {})
    probe["hard_subset"] = _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv)
    probe["train_runtime_seconds"] = float(run_summary.get("train_runtime_seconds", 0.0) or 0.0)
    probe["best_eval_loss"] = float(run_summary.get("best_eval_loss", 0.0) or 0.0)
    return probe


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml",
    )
    ap.add_argument(
        "--base-checkpoint-dir",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model",
    )
    ap.add_argument("--control-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0")
    ap.add_argument("--combo-base-processed-dir", default="data/processed_taskform_replay25_fold0")
    ap.add_argument("--combo-processed-dir", default="data/processed_taskform_replay25_retrieval_top1_fold0")
    ap.add_argument("--combo-build-report-dir", default="reports/taskform_winner_combo_retrieval_replay25_build_20260311")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--probe-max-steps", type=int, default=180)
    ap.add_argument("--probe-eval-steps", type=int, default=45)
    ap.add_argument("--anchor-gate", type=float, default=0.25)
    ap.add_argument("--hard-floor", type=float, default=0.0)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_combo_retrieval_replay25_probe_20260311")
    args = ap.parse_args()

    python_exec = str(REPO_ROOT / ".venv-deeppast" / "bin" / "python")
    started = time.time()

    base_cfg_path = _resolve_path(
        args.base_config,
        REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_wlite_20260310" / "generated_configs" / "taskform_winner_a2_retrieval_top1_wlite.yaml",
    )
    base_checkpoint_dir = _resolve_path(
        args.base_checkpoint_dir,
        REPO_ROOT / "runs" / "TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0" / "best_model",
    )
    control_processed_dir = _resolve_path(
        args.control_processed_dir,
        REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0",
    )
    combo_base_processed_dir = _resolve_path(
        args.combo_base_processed_dir,
        REPO_ROOT / "data" / "processed_taskform_replay25_fold0",
    )
    combo_processed_dir = _resolve_path(
        args.combo_processed_dir,
        REPO_ROOT / "data" / "processed_taskform_replay25_retrieval_top1_fold0",
    )
    combo_build_report_dir = _resolve_path(
        args.combo_build_report_dir,
        REPO_ROOT / "reports" / "taskform_winner_combo_retrieval_replay25_build_20260311",
    )
    hard_ids_csv = _resolve_path(
        args.hard_ids_csv,
        REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv",
    )
    out_dir = _resolve_path(
        args.report_dir,
        REPO_ROOT / "reports" / "taskform_winner_combo_retrieval_replay25_probe_20260311",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = out_dir / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    base_cfg = _load_yaml(base_cfg_path)

    try:
        _write_status(
            status_path,
            stage="started",
            status="running",
            extra={"report_dir": str(out_dir), "run_suffix": str(args.run_suffix)},
        )

        _write_status(status_path, stage="refs", status="running")
        incumbent_anchor = _ensure_incumbent_anchor(
            python_exec=python_exec,
            base_cfg_path=REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
            checkpoint_dir=REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
            fold=int(args.fold),
            out_dir=out_dir,
        )
        frozen_candidate = _load_frozen_candidate()
        _write_status(
            status_path,
            stage="refs",
            status="done",
            extra={"incumbent_anchor64_geom": float(incumbent_anchor.get("eval_geom", 0.0) or 0.0)},
        )

        _write_status(status_path, stage="build_combo", status="running")
        combo_build = _build_combo_if_needed(
            python_exec=python_exec,
            base_cfg_path=base_cfg_path,
            combo_base_processed_dir=combo_base_processed_dir,
            combo_processed_dir=combo_processed_dir,
            combo_build_report_dir=combo_build_report_dir,
            fold=int(args.fold),
        )
        _write_status(
            status_path,
            stage="build_combo",
            status="done",
            extra={"combo_processed_dir": str(combo_processed_dir)},
        )

        specs = [
            (
                "ctrl",
                control_processed_dir,
                f"TASKFORM_WINNER_RETRIEVAL_CTRL_COMBO_{safe_text(str(args.run_suffix)).upper()}",
                f"taskform_winner_retrieval_ctrl_combo_anchor64_{args.run_suffix}",
            ),
            (
                "combo",
                combo_processed_dir,
                f"TASKFORM_WINNER_RETRIEVAL_REPLAY25_COMBO_{safe_text(str(args.run_suffix)).upper()}",
                f"taskform_winner_retrieval_replay25_combo_anchor64_{args.run_suffix}",
            ),
        ]

        candidates: dict[str, Any] = {}
        for label, processed_dir, run_name, tag in specs:
            _write_status(status_path, stage=f"probe_{label}", status="running")
            candidate = _evaluate_candidate(
                python_exec=python_exec,
                base_cfg=base_cfg,
                generated_cfg_dir=generated_cfg_dir,
                base_checkpoint_dir=base_checkpoint_dir,
                processed_dir=processed_dir,
                run_name=run_name,
                label=label,
                tag=tag,
                fold=int(args.fold),
                max_steps=int(args.probe_max_steps),
                eval_steps=int(args.probe_eval_steps),
                hard_ids_csv=hard_ids_csv,
            )
            candidate["delta_geom_vs_incumbent_anchor64"] = round(
                float(candidate["eval_geom"]) - float(incumbent_anchor.get("eval_geom", 0.0) or 0.0),
                4,
            )
            candidate["delta_geom_vs_frozen_anchor64"] = round(
                float(candidate["eval_geom"]) - float(frozen_candidate.get("anchor64_reconstructed_geom", 0.0) or 0.0),
                4,
            )
            candidates[label] = candidate
            _write_status(
                status_path,
                stage=f"probe_{label}",
                status="done",
                extra={f"{label}_anchor64_geom": float(candidate["eval_geom"])},
            )

        control = candidates["ctrl"]
        combo = candidates["combo"]
        anchor_delta = round(float(combo["eval_geom"]) - float(control["eval_geom"]), 4)
        hard_delta = round(
            float((combo.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0)
            - float((control.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0),
            4,
        )
        train_runtime_ratio = round(
            float(combo.get("train_runtime_seconds", 0.0) or 0.0)
            / max(1e-9, float(control.get("train_runtime_seconds", 0.0) or 0.0)),
            4,
        )
        health_delta = _compare_output_health(combo.get("output_health", {}) or {}, control.get("output_health", {}) or {})
        recon_health_delta = _compare_output_health(
            combo.get("reconstructed_health", {}) or {},
            control.get("reconstructed_health", {}) or {},
        )
        health_all_green = bool(health_delta["no_regression"]) and bool(recon_health_delta["no_regression"])

        best_label = "combo" if float(combo["eval_geom"]) > float(control["eval_geom"]) else "ctrl"
        best_status = "control_only"
        overall_status = "reject_stop"
        reason = "combo arm fails the matched probe gate; retrieval control remains best"

        if best_label == "combo":
            if anchor_delta >= float(args.anchor_gate) and hard_delta >= float(args.hard_floor) and health_all_green:
                best_status = "review_to_candidate_pool"
                overall_status = "review_to_candidate_pool"
                reason = "retrieval + replay25 combo clears the written probe gate against retrieval control"
            elif anchor_delta > 0.0 and hard_delta >= float(args.hard_floor) and health_all_green:
                best_status = "review_stop"
                overall_status = "review_stop"
                reason = "combo arm is locally positive but does not yet clear the candidate-pool gate"

        comparisons = {
            "combo": {
                "delta_geom_vs_control_anchor64": anchor_delta,
                "delta_geom_vs_control_hard_subset": hard_delta,
                "train_runtime_ratio_vs_control": train_runtime_ratio,
                "health_delta_vs_control": health_delta,
                "reconstructed_health_delta_vs_control": recon_health_delta,
                "status": best_status if best_label == "combo" else "control_only",
            }
        }

        summary = {
            "line": "winner_retrieval_replay25_combo_probe",
            "status": overall_status,
            "reason": reason,
            "base_config_path": str(base_cfg_path),
            "base_checkpoint_dir": str(base_checkpoint_dir),
            "control_processed_dir": str(control_processed_dir),
            "combo_base_processed_dir": str(combo_base_processed_dir),
            "combo_processed_dir": str(combo_processed_dir),
            "combo_build": combo_build,
            "fold": int(args.fold),
            "probe_max_steps": int(args.probe_max_steps),
            "probe_eval_steps": int(args.probe_eval_steps),
            "anchor_gate": float(args.anchor_gate),
            "hard_floor": float(args.hard_floor),
            "report_dir": str(out_dir),
            "incumbent_anchor64": incumbent_anchor,
            "frozen_candidate": frozen_candidate,
            "candidates": candidates,
            "comparisons": comparisons,
            "best_label": best_label,
            "best_status": best_status,
            "eligible_for_postprobe_fullval": best_label == "combo" and anchor_delta > 0.0 and hard_delta >= float(args.hard_floor) and health_all_green,
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        }
        write_json(out_dir / "summary.json", summary)

        lines = [
            "# Winner Retrieval + Replay25 Combo Probe",
            "",
            f"- status: `{overall_status}`",
            f"- reason: {reason}",
            f"- incumbent anchor64 geom: `{float(incumbent_anchor.get('eval_geom', 0.0) or 0.0):.4f}`",
            f"- frozen fallback anchor/fullval/hard: `{float(frozen_candidate.get('anchor64_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('fullval_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('hard_geom', 0.0) or 0.0):.4f}`",
            f"- combo build reused existing: `{bool(combo_build.get('reused_existing_build', False))}`",
            "",
            "## Control",
            "",
            f"- anchor64 geom: `{float(control['eval_geom']):.4f}`",
            f"- hard geom: `{float((control.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0):.4f}`",
            "",
            "## Combo",
            "",
            f"- anchor64 geom: `{float(combo['eval_geom']):.4f}`",
            f"- hard geom: `{float((combo.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0):.4f}`",
            f"- delta anchor vs control: `{anchor_delta:+.4f}`",
            f"- delta hard vs control: `{hard_delta:+.4f}`",
            f"- train runtime ratio vs control: `{train_runtime_ratio:.4f}`",
            f"- health no_regression: `{bool(health_delta['no_regression'])}`",
            f"- reconstructed health no_regression: `{bool(recon_health_delta['no_regression'])}`",
            f"- delta vs incumbent anchor64: `{float(combo['delta_geom_vs_incumbent_anchor64']):+.4f}`",
            f"- delta vs frozen anchor64: `{float(combo['delta_geom_vs_frozen_anchor64']):+.4f}`",
            f"- status: `{comparisons['combo']['status']}`",
            f"- eligible_for_postprobe_fullval: `{bool(summary['eligible_for_postprobe_fullval'])}`",
            "",
        ]
        write_text(out_dir / "gate_report.md", "\n".join(lines) + "\n")

        _write_status(
            status_path,
            stage="completed",
            status=overall_status,
            extra={
                "best_label": best_label,
                "summary_json": str(out_dir / "summary.json"),
                "gate_report_md": str(out_dir / "gate_report.md"),
            },
        )
    except Exception as exc:  # pragma: no cover - operational path
        _write_status(
            status_path,
            stage="failed",
            status="failed",
            extra={"error": safe_text(str(exc)), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
