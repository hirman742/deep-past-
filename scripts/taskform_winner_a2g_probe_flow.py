from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import REPO_ROOT, resolve_path, safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import (
    _compute_hard_subset_metrics,
    _compute_targeted_bucket_metrics,
    _load_val_meta,
)


SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _compare_output_health(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
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
    candidate_health = candidate.get("output_health", {}) or {}
    baseline_health = baseline.get("output_health", {}) or {}
    deltas = {
        key: round(float(candidate_health.get(key, 0.0) or 0.0) - float(baseline_health.get(key, 0.0) or 0.0), 4)
        for key in keys
    }
    no_regression = (
        float(candidate_health.get("empty_prediction_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("empty_prediction_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("pred_shorter_than_half_ref_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("pred_shorter_than_half_ref_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("repeat_prediction_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("repeat_prediction_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("internal_repeat_trigram_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("internal_repeat_trigram_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("copy_source_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("copy_source_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("has_bad_token_regex_ratio_pct", 0.0) or 0.0)
        <= float(baseline_health.get("has_bad_token_regex_ratio_pct", 0.0) or 0.0)
        and float(candidate_health.get("unique_prediction_ratio_pct", 0.0) or 0.0)
        >= float(baseline_health.get("unique_prediction_ratio_pct", 0.0) or 0.0)
    )
    deltas["no_regression"] = bool(no_regression)
    return deltas


def _candidate_run_dir(config_path: Path, fold: int) -> Path:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_root = resolve_path((cfg.get("paths", {}) or {}).get("run_dir"), REPO_ROOT / "runs" / "missing")
    return run_root.parent / f"{run_root.name}_fold{fold}"


def _train_if_needed(config_path: Path, fold: int, max_steps: int, eval_steps: int) -> tuple[Path, dict[str, Any]]:
    run_dir = _candidate_run_dir(config_path, fold)
    best_model = run_dir / "best_model"
    if not best_model.exists():
        _run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_mt5_lora.py"),
                "--config",
                str(config_path),
                "--fold",
                str(fold),
                "--max-steps",
                str(max_steps),
                "--eval-steps",
                str(eval_steps),
                "--skip-final-predict",
            ]
        )
    return run_dir, _load_json(run_dir / "run_summary.json")


def _evaluate_candidate(
    *,
    label: str,
    config_path: Path,
    fold: int,
    max_steps: int,
    eval_steps: int,
    predict_batch_size: int,
    anchor_samples: int,
    hard_ids_csv: Path,
    meta_df: pd.DataFrame,
    token_freq: dict[str, int],
    tag: str,
) -> dict[str, Any]:
    run_dir, run_summary = _train_if_needed(config_path, fold, max_steps, eval_steps)
    checkpoint_dir = run_dir / "best_model"
    decode_best_path = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    diagnose_summary_path = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"

    if not decode_best_path.exists():
        _run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "eval_decode_grid.py"),
                "--config",
                str(config_path),
                "--fold",
                str(fold),
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
                str(predict_batch_size),
                "--max-val-samples",
                str(anchor_samples),
                "--aggregate-by-parent",
                "auto",
                "--aggregate-original-only",
            ]
        )
    decode_payload = _load_json(decode_best_path)

    if not diagnose_summary_path.exists():
        _run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "diagnose_val_outputs.py"),
                "--config",
                str(config_path),
                "--fold",
                str(fold),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--tag",
                tag,
                "--predict-batch-size",
                str(predict_batch_size),
                "--num-beams",
                str(int(decode_payload.get("num_beams", 4))),
                "--length-penalty",
                str(float(decode_payload.get("length_penalty", 0.7))),
                "--no-repeat-ngram-size",
                str(int(decode_payload.get("no_repeat_ngram_size", 0))),
                "--min-new-tokens",
                str(int(decode_payload.get("min_new_tokens", 0))),
                "--max-new-tokens",
                str(int(decode_payload.get("max_new_tokens", 384))),
                "--aggregate-by-parent",
                "auto",
                "--aggregate-original-only",
                "--max-rows",
                str(anchor_samples),
            ]
        )
    diag_payload = _load_json(diagnose_summary_path)

    reconstructed_csv = Path(
        (((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    )
    pred_csv = Path((((diag_payload.get("artifacts", {}) or {}).get("predictions_csv", ""))))

    return {
        "label": label,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "run_summary_path": str(run_dir / "run_summary.json"),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "anchor64": {
            "eval_geom": float(decode_payload.get("eval_geom", 0.0) or 0.0),
            "eval_bleu": float(decode_payload.get("eval_bleu", 0.0) or 0.0),
            "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0) or 0.0),
            "elapsed_seconds": float(decode_payload.get("elapsed_seconds", 0.0) or 0.0),
        },
        "output_health": diag_payload.get("output_health", {}) or {},
        "reconstructed_health": (((diag_payload.get("reconstructed", {}) or {}).get("output_health", {}) or {})),
        "hard_subset": _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv),
        "targeted_buckets": _compute_targeted_bucket_metrics(pred_csv, meta_df, token_freq),
        "train_runtime_seconds": float(run_summary.get("train_runtime_seconds", 0.0) or 0.0),
        "best_eval_loss": float(run_summary.get("best_eval_loss", 0.0) or 0.0),
        "gpu_peak_utilization_pct": float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0),
        "init_adapter_dir": safe_text(run_summary.get("init_adapter_dir", "")),
    }


def _load_refs() -> dict[str, Any]:
    incumbent_path = REPO_ROOT / "reports" / "taskform_a2_a1_20260310" / "incumbent_anchor64_summary.json"
    frozen_path = REPO_ROOT / "reports" / "taskform_winner_a2_freeze_20260310" / "summary.json"
    payload: dict[str, Any] = {"incumbent_anchor64": {}, "frozen_candidate": {}}
    if incumbent_path.exists():
        data = _load_json(incumbent_path)
        payload["incumbent_anchor64"] = {
            "source": str(incumbent_path),
            "eval_geom": float(data.get("eval_geom", 0.0) or 0.0),
            "eval_bleu": float(data.get("eval_bleu", 0.0) or 0.0),
            "eval_chrfpp": float(data.get("eval_chrfpp", 0.0) or 0.0),
            "output_health": data.get("output_health", {}) or {},
        }
    if frozen_path.exists():
        data = _load_json(frozen_path)
        frozen = ((data.get("scoreboard", {}) or {}).get("fallback_180", {}) or {})
        payload["frozen_candidate"] = {
            "source": str(frozen_path),
            "anchor64_reconstructed_geom": float(frozen.get("anchor64_reconstructed_geom", 0.0) or 0.0),
            "fullval_reconstructed_geom": float(frozen.get("fullval_reconstructed_geom", 0.0) or 0.0),
            "hard_geom": float(frozen.get("hard_geom", 0.0) or 0.0),
        }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-summary", default="reports/taskform_winner_a2g_build_20260311/summary.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=180)
    ap.add_argument("--eval-steps", type=int, default=45)
    ap.add_argument("--predict-batch-size", type=int, default=16)
    ap.add_argument("--anchor-samples", type=int, default=64)
    ap.add_argument("--anchor-gate", type=float, default=0.25)
    ap.add_argument("--hard-floor", type=float, default=-0.10)
    ap.add_argument("--latency-gate", type=float, default=1.8)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2g_probe_20260311")
    args = ap.parse_args()

    build_summary = _load_json(
        resolve_path(args.build_summary, REPO_ROOT / "reports" / "taskform_winner_a2g_build_20260311" / "summary.json")
    )
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2g_probe_20260311")
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv")
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_map = build_summary.get("candidates", {}) or {}
    control_cfg = Path(str(candidate_map["ctrl"]["config_path"]))
    meta_df, token_freq = _load_val_meta(control_cfg, int(args.fold))

    control = _evaluate_candidate(
        label="ctrl",
        config_path=control_cfg,
        fold=int(args.fold),
        max_steps=int(args.max_steps),
        eval_steps=int(args.eval_steps),
        predict_batch_size=int(args.predict_batch_size),
        anchor_samples=int(args.anchor_samples),
        hard_ids_csv=hard_ids_csv,
        meta_df=meta_df,
        token_freq=token_freq,
        tag="taskform_winner_a2g_ctrl_anchor64_20260311",
    )

    candidates: dict[str, Any] = {"ctrl": control}
    comparisons: dict[str, Any] = {}
    best_label = "ctrl"
    best_score = float(control["anchor64"]["eval_geom"])

    for label in ("g35", "g50", "g65"):
        build = candidate_map.get(label)
        if not build:
            continue
        candidate = _evaluate_candidate(
            label=label,
            config_path=Path(str(build["config_path"])),
            fold=int(args.fold),
            max_steps=int(args.max_steps),
            eval_steps=int(args.eval_steps),
            predict_batch_size=int(args.predict_batch_size),
            anchor_samples=int(args.anchor_samples),
            hard_ids_csv=hard_ids_csv,
            meta_df=meta_df,
            token_freq=token_freq,
            tag=f"taskform_winner_a2g_{label}_anchor64_20260311",
        )
        candidates[label] = candidate

        anchor_delta = round(float(candidate["anchor64"]["eval_geom"]) - float(control["anchor64"]["eval_geom"]), 4)
        hard_delta = round(float(candidate["hard_subset"]["eval_geom"]) - float(control["hard_subset"]["eval_geom"]), 4)
        latency_ratio = round(
            float(candidate["anchor64"]["elapsed_seconds"]) / max(1e-9, float(control["anchor64"]["elapsed_seconds"])),
            4,
        )
        health_delta = _compare_output_health(candidate, control)
        recon_health_delta = _compare_output_health(
            {"output_health": candidate.get("reconstructed_health", {}) or {}},
            {"output_health": control.get("reconstructed_health", {}) or {}},
        )
        targeted_deltas = {
            bucket: round(
                float((candidate["targeted_buckets"].get(bucket, {}) or {}).get("eval_geom", 0.0))
                - float((control["targeted_buckets"].get(bucket, {}) or {}).get("eval_geom", 0.0)),
                4,
            )
            for bucket in ("rare_name", "measure", "formula", "marker_rich")
        }
        health_all_green = bool(health_delta["no_regression"]) and bool(recon_health_delta["no_regression"])
        rare_formula_ok = not (
            float(targeted_deltas["rare_name"]) < 0.0 and float(targeted_deltas["formula"]) < 0.0
        )
        measure_or_marker_positive = bool(
            float(targeted_deltas["measure"]) > 0.0 or float(targeted_deltas["marker_rich"]) > 0.0
        )

        status = "reject_stop"
        if (
            anchor_delta >= float(args.anchor_gate)
            and hard_delta >= float(args.hard_floor)
            and latency_ratio <= float(args.latency_gate)
            and health_all_green
            and rare_formula_ok
        ):
            status = "review_to_w"
        elif anchor_delta > 0.0 and hard_delta >= float(args.hard_floor) and latency_ratio <= float(args.latency_gate) and health_all_green:
            status = "review_stop"

        comparisons[label] = {
            "delta_geom_vs_control_anchor64": anchor_delta,
            "delta_geom_vs_control_hard_subset": hard_delta,
            "decode_latency_ratio_vs_control": latency_ratio,
            "health_delta_vs_control": health_delta,
            "reconstructed_health_delta_vs_control": recon_health_delta,
            "targeted_bucket_deltas_vs_control": targeted_deltas,
            "measure_or_marker_positive": measure_or_marker_positive,
            "rare_formula_ok": rare_formula_ok,
            "status": status,
        }

        candidate_score = float(candidate["anchor64"]["eval_geom"])
        if candidate_score > best_score:
            best_label = label
            best_score = candidate_score

    best_status = comparisons.get(best_label, {}).get("status", "control_only")
    if best_status == "review_to_w":
        overall_status = "review_to_w"
        reason = "best selective gate clears the written probe gate against retrieval always-on control"
    elif best_status == "review_stop":
        overall_status = "review_stop"
        reason = "best selective gate is locally positive but does not clear the full probe gate"
    else:
        overall_status = "reject_stop"
        reason = "selective retrieval gates fail the matched probe gate against retrieval always-on control"

    refs = _load_refs()
    incumbent = refs.get("incumbent_anchor64", {}) or {}
    frozen = refs.get("frozen_candidate", {}) or {}

    summary = {
        "line": "A2g_retrieval_selective_gate",
        "status": overall_status,
        "reason": reason,
        "fold": int(args.fold),
        "max_steps": int(args.max_steps),
        "eval_steps": int(args.eval_steps),
        "anchor_samples": int(args.anchor_samples),
        "anchor_gate": float(args.anchor_gate),
        "hard_floor": float(args.hard_floor),
        "latency_gate": float(args.latency_gate),
        "control": control,
        "candidates": candidates,
        "comparisons": comparisons,
        "best_gate_label": best_label,
        "best_gate_status": best_status,
        "incumbent_anchor64": incumbent,
        "frozen_candidate": frozen,
    }
    if best_label in candidates:
        summary["best_gate_vs_incumbent_anchor64"] = round(
            float(candidates[best_label]["anchor64"]["eval_geom"]) - float(incumbent.get("eval_geom", 0.0) or 0.0),
            4,
        )
        summary["best_gate_vs_frozen_anchor64"] = round(
            float(candidates[best_label]["anchor64"]["eval_geom"])
            - float(frozen.get("anchor64_reconstructed_geom", 0.0) or 0.0),
            4,
        )

    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A2g Probe Gate Report",
        "",
        f"- status: `{overall_status}`",
        f"- reason: {reason}",
        f"- control anchor64 geom: `{float(control['anchor64']['eval_geom']):.4f}`",
        f"- control hard geom: `{float(control['hard_subset']['eval_geom']):.4f}`",
        f"- best gate: `{best_label}`",
        f"- best gate status: `{best_status}`",
        "",
    ]
    for label in ("g35", "g50", "g65"):
        if label not in comparisons:
            continue
        cand = candidates[label]
        comp = comparisons[label]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- anchor64 geom: `{float(cand['anchor64']['eval_geom']):.4f}`",
                f"- hard geom: `{float(cand['hard_subset']['eval_geom']):.4f}`",
                f"- delta anchor vs control: `{float(comp['delta_geom_vs_control_anchor64']):.4f}`",
                f"- delta hard vs control: `{float(comp['delta_geom_vs_control_hard_subset']):.4f}`",
                f"- latency ratio vs control: `{float(comp['decode_latency_ratio_vs_control']):.4f}`",
                f"- health no_regression: `{bool(comp['health_delta_vs_control']['no_regression'])}`",
                f"- reconstructed health no_regression: `{bool(comp['reconstructed_health_delta_vs_control']['no_regression'])}`",
                f"- targeted deltas: `rare={float(comp['targeted_bucket_deltas_vs_control']['rare_name']):.4f}, measure={float(comp['targeted_bucket_deltas_vs_control']['measure']):.4f}, formula={float(comp['targeted_bucket_deltas_vs_control']['formula']):.4f}, marker={float(comp['targeted_bucket_deltas_vs_control']['marker_rich']):.4f}`",
                f"- status: `{comp['status']}`",
                "",
            ]
        )
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
