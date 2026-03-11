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


def _build_replay_if_needed(
    *,
    python_exec: str,
    base_cfg_path: Path,
    base_processed_dir: Path,
    out_dir: Path,
    fold: int,
    ratio: float,
) -> dict[str, Any]:
    audit_path = out_dir / "audit_hardcase_replay.json"
    if (out_dir / "train_proc.csv").exists() and (out_dir / "folds.csv").exists() and audit_path.exists():
        payload = _load_json(audit_path)
        payload["reused_existing_build"] = True
        return payload
    _run(
        [
            python_exec,
            str(SCRIPTS_DIR / "build_taskform_probe_datasets.py"),
            "replay",
            "--config",
            str(base_cfg_path),
            "--base-processed-dir",
            str(base_processed_dir),
            "--out-dir",
            str(out_dir),
            "--fold",
            str(int(fold)),
            "--ratio",
            str(float(ratio)),
            "--seed",
            "42",
        ]
    )
    payload = _load_json(audit_path)
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
    ap.add_argument("--base-config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--base-checkpoint-dir", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250")
    ap.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap.add_argument("--replay15-dir", default="data/processed_taskform_replay15_fold0")
    ap.add_argument("--replay20-dir", default="data/processed_taskform_replay20_fold0")
    ap.add_argument("--replay30-dir", default="data/processed_taskform_replay30_fold0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--probe-max-steps", type=int, default=180)
    ap.add_argument("--probe-eval-steps", type=int, default=45)
    ap.add_argument("--anchor-gate", type=float, default=0.25)
    ap.add_argument("--hard-floor", type=float, default=0.0)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_replay_band_probe_20260311")
    args = ap.parse_args()

    python_exec = str(REPO_ROOT / ".venv-deeppast" / "bin" / "python")
    started = time.time()

    base_cfg_path = _resolve_path(
        args.base_config,
        REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
    )
    base_checkpoint_dir = _resolve_path(
        args.base_checkpoint_dir,
        REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
    )
    base_processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    replay15_dir = _resolve_path(args.replay15_dir, REPO_ROOT / "data" / "processed_taskform_replay15_fold0")
    replay20_dir = _resolve_path(args.replay20_dir, REPO_ROOT / "data" / "processed_taskform_replay20_fold0")
    replay30_dir = _resolve_path(args.replay30_dir, REPO_ROOT / "data" / "processed_taskform_replay30_fold0")
    hard_ids_csv = _resolve_path(
        args.hard_ids_csv,
        REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv",
    )
    out_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_replay_band_probe_20260311")
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
            base_cfg_path=base_cfg_path,
            checkpoint_dir=base_checkpoint_dir,
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

        build_specs = [
            ("replay15", replay15_dir, 0.15),
            ("replay20", replay20_dir, 0.20),
            ("replay30", replay30_dir, 0.30),
        ]
        builds: dict[str, Any] = {}
        for label, out_build_dir, ratio in build_specs:
            _write_status(status_path, stage=f"build_{label}", status="running")
            builds[label] = _build_replay_if_needed(
                python_exec=python_exec,
                base_cfg_path=base_cfg_path,
                base_processed_dir=base_processed_dir,
                out_dir=out_build_dir,
                fold=int(args.fold),
                ratio=float(ratio),
            )
            _write_status(status_path, stage=f"build_{label}", status="done")

        specs = [
            ("ctrl", base_processed_dir, f"TASKFORM_WINNER_REPLAY_CTRL_BAND_{safe_text(str(args.run_suffix)).upper()}", f"taskform_winner_replay_ctrl_band_anchor64_{args.run_suffix}"),
            ("replay15", replay15_dir, f"TASKFORM_WINNER_REPLAY15_{safe_text(str(args.run_suffix)).upper()}", f"taskform_winner_replay15_anchor64_{args.run_suffix}"),
            ("replay20", replay20_dir, f"TASKFORM_WINNER_REPLAY20_{safe_text(str(args.run_suffix)).upper()}", f"taskform_winner_replay20_anchor64_{args.run_suffix}"),
            ("replay30", replay30_dir, f"TASKFORM_WINNER_REPLAY30_{safe_text(str(args.run_suffix)).upper()}", f"taskform_winner_replay30_anchor64_{args.run_suffix}"),
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
        comparisons: dict[str, Any] = {}
        best_label = "ctrl"
        best_eval_geom = float(control["eval_geom"])

        for label in ("replay15", "replay20", "replay30"):
            candidate = candidates[label]
            anchor_delta = round(float(candidate["eval_geom"]) - float(control["eval_geom"]), 4)
            hard_delta = round(
                float((candidate.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0)
                - float((control.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0),
                4,
            )
            train_runtime_ratio = round(
                float(candidate.get("train_runtime_seconds", 0.0) or 0.0)
                / max(1e-9, float(control.get("train_runtime_seconds", 0.0) or 0.0)),
                4,
            )
            health_delta = _compare_output_health(candidate.get("output_health", {}) or {}, control.get("output_health", {}) or {})
            recon_health_delta = _compare_output_health(
                candidate.get("reconstructed_health", {}) or {},
                control.get("reconstructed_health", {}) or {},
            )
            health_all_green = bool(health_delta["no_regression"]) and bool(recon_health_delta["no_regression"])
            status = "reject_stop"
            if anchor_delta >= float(args.anchor_gate) and hard_delta >= float(args.hard_floor) and health_all_green:
                status = "review_to_candidate_pool"
            elif anchor_delta > 0.0 and hard_delta >= float(args.hard_floor) and health_all_green:
                status = "review_stop"
            comparisons[label] = {
                "delta_geom_vs_control_anchor64": anchor_delta,
                "delta_geom_vs_control_hard_subset": hard_delta,
                "train_runtime_ratio_vs_control": train_runtime_ratio,
                "health_delta_vs_control": health_delta,
                "reconstructed_health_delta_vs_control": recon_health_delta,
                "status": status,
            }
            if float(candidate["eval_geom"]) > best_eval_geom:
                best_label = label
                best_eval_geom = float(candidate["eval_geom"])

        best_status = comparisons.get(best_label, {}).get("status", "control_only")
        if best_status == "review_to_candidate_pool":
            overall_status = "review_to_candidate_pool"
            reason = "best replay-band arm clears the written probe gate against matched control"
        elif best_status == "review_stop":
            overall_status = "review_stop"
            reason = "best replay-band arm is locally positive but does not yet clear the candidate-pool gate"
        else:
            overall_status = "reject_stop"
            reason = "replay-band arms fail the matched probe gate; control remains best"

        summary = {
            "line": "winner_replay_band_probe",
            "status": overall_status,
            "reason": reason,
            "base_config_path": str(base_cfg_path),
            "base_checkpoint_dir": str(base_checkpoint_dir),
            "base_processed_dir": str(base_processed_dir),
            "replay15_dir": str(replay15_dir),
            "replay20_dir": str(replay20_dir),
            "replay30_dir": str(replay30_dir),
            "builds": builds,
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
            "eligible_for_postprobe_fullval": best_label != "ctrl"
            and float(comparisons.get(best_label, {}).get("delta_geom_vs_control_anchor64", 0.0)) > 0.0
            and float(comparisons.get(best_label, {}).get("delta_geom_vs_control_hard_subset", -1.0)) >= float(args.hard_floor)
            and bool((comparisons.get(best_label, {}).get("health_delta_vs_control", {}) or {}).get("no_regression", False))
            and bool((comparisons.get(best_label, {}).get("reconstructed_health_delta_vs_control", {}) or {}).get("no_regression", False)),
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        }
        write_json(out_dir / "summary.json", summary)

        lines = [
            "# Winner Replay Narrow Probe",
            "",
            f"- status: `{overall_status}`",
            f"- reason: {reason}",
            f"- incumbent anchor64 geom: `{float(incumbent_anchor.get('eval_geom', 0.0) or 0.0):.4f}`",
            f"- frozen fallback anchor/fullval/hard: `{float(frozen_candidate.get('anchor64_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('fullval_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('hard_geom', 0.0) or 0.0):.4f}`",
            "",
            "## Control",
            "",
            f"- anchor64 geom: `{float(control['eval_geom']):.4f}`",
            f"- hard geom: `{float((control.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0):.4f}`",
            "",
        ]
        for label in ("replay15", "replay20", "replay30"):
            candidate = candidates[label]
            comp = comparisons[label]
            lines.extend(
                [
                    f"## {label}",
                    "",
                    f"- anchor64 geom: `{float(candidate['eval_geom']):.4f}`",
                    f"- hard geom: `{float((candidate.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0):.4f}`",
                    f"- delta anchor vs control: `{float(comp['delta_geom_vs_control_anchor64']):+.4f}`",
                    f"- delta hard vs control: `{float(comp['delta_geom_vs_control_hard_subset']):+.4f}`",
                    f"- train runtime ratio vs control: `{float(comp['train_runtime_ratio_vs_control']):.4f}`",
                    f"- health no_regression: `{bool(comp['health_delta_vs_control']['no_regression'])}`",
                    f"- reconstructed health no_regression: `{bool(comp['reconstructed_health_delta_vs_control']['no_regression'])}`",
                    f"- delta vs incumbent anchor64: `{float(candidate['delta_geom_vs_incumbent_anchor64']):+.4f}`",
                    f"- delta vs frozen anchor64: `{float(candidate['delta_geom_vs_frozen_anchor64']):+.4f}`",
                    f"- status: `{comp['status']}`",
                    "",
                ]
            )
        lines.extend(
            [
                "## Queue Eligibility",
                "",
                f"- best label: `{best_label}`",
                f"- eligible_for_postprobe_fullval: `{bool(summary['eligible_for_postprobe_fullval'])}`",
                "",
            ]
        )
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
