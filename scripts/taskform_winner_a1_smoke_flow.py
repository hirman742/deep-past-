from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from taskform_phase12_common import evaluate_frame, resolve_path, safe_text, write_json, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _candidate_run_dir(cfg: dict[str, Any], fold: int) -> Path:
    run_root = resolve_path((cfg.get("paths", {}) or {}).get("run_dir"), REPO_ROOT / "runs" / "missing")
    return run_root.parent / f"{run_root.name}_fold{fold}"


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


def _compute_hard_subset_metrics(reconstructed_csv: Path, hard_ids_csv: Path) -> dict[str, Any]:
    pred_df = pd.read_csv(reconstructed_csv)
    hard_df = pd.read_csv(hard_ids_csv)
    hard_ids = set(hard_df["oare_id"].fillna("").astype(str).tolist())
    id_col = "oare_id" if "oare_id" in pred_df.columns else "id"
    subset = pred_df.loc[pred_df[id_col].fillna("").astype(str).isin(hard_ids)].copy().reset_index(drop=True)
    summary = evaluate_frame(
        subset,
        prediction_col="prediction",
        reference_col="reference",
        tag="hard_subset",
        subset_name="hard_subset",
        note="hard subset = ids intersect routed_full",
    )
    summary["rows"] = int(len(subset))
    return summary


def _load_incumbent_anchor() -> dict[str, Any]:
    path = REPO_ROOT / "reports" / "taskform_a2_a1_20260310" / "incumbent_anchor64_summary.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    return {
        "source": str(path),
        "eval_geom": float(payload.get("eval_geom", 0.0)),
        "eval_bleu": float(payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(payload.get("eval_chrfpp", 0.0)),
        "output_health": payload.get("output_health", {}) or {},
    }


def _train_if_needed(cfg_path: Path, *, fold: int, max_steps: int, eval_steps: int) -> tuple[Path, dict[str, Any]]:
    cfg = _load_yaml(cfg_path)
    run_dir = _candidate_run_dir(cfg, fold)
    best_model = run_dir / "best_model"
    if not best_model.exists():
        _run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_mt5_lora.py"),
                "--config",
                str(cfg_path),
                "--fold",
                str(fold),
                "--max-steps",
                str(max_steps),
                "--eval-steps",
                str(eval_steps),
                "--skip-final-predict",
            ]
        )
    run_summary = _load_json(run_dir / "run_summary.json")
    return run_dir, run_summary


def _evaluate_candidate(
    *,
    label: str,
    cfg_path: Path,
    fold: int,
    hard_ids_csv: Path,
    tag: str,
    predict_batch_size: int,
    anchor_samples: int,
    max_steps: int,
    eval_steps: int,
) -> dict[str, Any]:
    run_dir, run_summary = _train_if_needed(cfg_path, fold=fold, max_steps=max_steps, eval_steps=eval_steps)
    checkpoint_dir = run_dir / "best_model"
    decode_best_path = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    diagnose_summary_path = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"

    if not decode_best_path.exists():
        _run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "eval_decode_grid.py"),
                "--config",
                str(cfg_path),
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
                str(cfg_path),
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

    reconstructed_csv_str = (((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    reconstructed_csv = Path(reconstructed_csv_str)
    hard_subset = _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv)

    return {
        "label": label,
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "run_summary_path": str(run_dir / "run_summary.json"),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "anchor64": {
            "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
            "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
            "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
            "elapsed_seconds": float(decode_payload.get("elapsed_seconds", 0.0) or 0.0),
        },
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "reconstructed_health": (((diag_payload.get("reconstructed", {}) or {}).get("output_health", {}) or {})),
        "hard_subset": hard_subset,
        "train_runtime_seconds": float(run_summary.get("train_runtime_seconds", 0.0) or 0.0),
        "best_eval_loss": float(run_summary.get("best_eval_loss", 0.0) or 0.0),
        "gpu_peak_utilization_pct": float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-build-summary", default="reports/taskform_winner_a1_mix_build_20260310/summary.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--eval-steps", type=int, default=55)
    ap.add_argument("--predict-batch-size", type=int, default=16)
    ap.add_argument("--anchor-samples", type=int, default=64)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_smoke_20260310")
    args = ap.parse_args()

    mix_build_summary = _load_json(
        resolve_path(args.mix_build_summary, REPO_ROOT / "reports" / "taskform_winner_a1_mix_build_20260310" / "summary.json")
    )
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a1_smoke_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        ("internal_only_matched", "taskform_winner_a1_internal_only_anchor64_20260310"),
        ("e10", "taskform_winner_a1_e10_anchor64_20260310"),
        ("e30", "taskform_winner_a1_e30_anchor64_20260310"),
        ("e50", "taskform_winner_a1_e50_anchor64_20260310"),
    ]

    candidates: dict[str, dict[str, Any]] = {}
    for label, tag in labels:
        cfg_path = Path(mix_build_summary["builds"][label]["config_path"])
        candidates[label] = _evaluate_candidate(
            label=label,
            cfg_path=cfg_path,
            fold=int(args.fold),
            hard_ids_csv=hard_ids_csv,
            tag=tag,
            predict_batch_size=int(args.predict_batch_size),
            anchor_samples=int(args.anchor_samples),
            max_steps=int(args.max_steps),
            eval_steps=int(args.eval_steps),
        )

    control = candidates["internal_only_matched"]
    incumbent = _load_incumbent_anchor()
    comparisons: dict[str, Any] = {}
    best_label = "internal_only_matched"
    best_anchor = float(control["anchor64"]["eval_geom"])

    for label in ("e10", "e30", "e50"):
        candidate = candidates[label]
        delta_anchor = float(candidate["anchor64"]["eval_geom"]) - float(control["anchor64"]["eval_geom"])
        delta_hard = float(candidate["hard_subset"]["eval_geom"]) - float(control["hard_subset"]["eval_geom"])
        health_delta = _compare_output_health(candidate, control)
        reconstructed_health_delta = _compare_output_health(
            {"output_health": candidate.get("reconstructed_health", {})},
            {"output_health": control.get("reconstructed_health", {})},
        )
        status = "reject_stop"
        if delta_anchor >= 0.30 and delta_hard >= 0.0 and bool(health_delta["no_regression"]) and bool(reconstructed_health_delta["no_regression"]):
            if incumbent and float(candidate["anchor64"]["eval_geom"]) >= float(incumbent.get("eval_geom", 0.0) or 0.0):
                status = "accept_to_w"
            else:
                status = "review_to_wlite"
        elif delta_anchor > 0.0 and delta_hard >= 0.0:
            status = "review_stop"

        if float(candidate["anchor64"]["eval_geom"]) > best_anchor:
            best_label = label
            best_anchor = float(candidate["anchor64"]["eval_geom"])

        comparisons[label] = {
            "delta_geom_vs_control_anchor64": round(delta_anchor, 4),
            "delta_geom_vs_control_hard_subset": round(delta_hard, 4),
            "delta_geom_vs_incumbent_anchor64": round(
                float(candidate["anchor64"]["eval_geom"]) - float(incumbent.get("eval_geom", 0.0) or 0.0),
                4,
            )
            if incumbent
            else None,
            "health_delta_vs_control": health_delta,
            "reconstructed_health_delta_vs_control": reconstructed_health_delta,
            "status": status,
        }

    best_status = comparisons.get(best_label, {}).get("status", "control_only")
    if best_label == "internal_only_matched":
        overall_status = "reject_stop"
        reason = "none of the external ratios beat the internal-only matched control"
    elif best_status == "accept_to_w":
        overall_status = "accept_to_w"
        reason = "best external ratio beats control and clears the incumbent anchor gate"
    elif best_status == "review_to_wlite":
        overall_status = "review_to_wlite"
        reason = "best external ratio clears the local smoke gate but still needs W-lite confirmation"
    elif best_status == "review_stop":
        overall_status = "review_stop"
        reason = "best external ratio is locally positive but fails the full A1 smoke gate"
    else:
        overall_status = "reject_stop"
        reason = "external ratios fail the A1 smoke gate"

    summary = {
        "line": "A1_external_mix",
        "status": overall_status,
        "reason": reason,
        "fold": int(args.fold),
        "max_steps": int(args.max_steps),
        "eval_steps": int(args.eval_steps),
        "anchor_samples": int(args.anchor_samples),
        "control": control,
        "candidates": candidates,
        "comparisons": comparisons,
        "best_ratio_label": best_label,
        "best_ratio_status": best_status,
        "incumbent_anchor64": incumbent,
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A1 Smoke Gate Report",
        "",
        f"- status: `{overall_status}`",
        f"- reason: {reason}",
        f"- control anchor64 geom: `{control['anchor64']['eval_geom']:.4f}`",
        f"- control hard geom: `{control['hard_subset']['eval_geom']:.4f}`",
        f"- best ratio: `{best_label}`",
        f"- best ratio status: `{best_status}`",
        "",
    ]
    for label in ("e10", "e30", "e50"):
        comp = comparisons[label]
        cand = candidates[label]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- anchor64 geom: `{cand['anchor64']['eval_geom']:.4f}`",
                f"- hard geom: `{cand['hard_subset']['eval_geom']:.4f}`",
                f"- delta anchor vs control: `{comp['delta_geom_vs_control_anchor64']:.4f}`",
                f"- delta hard vs control: `{comp['delta_geom_vs_control_hard_subset']:.4f}`",
                f"- health no_regression: `{bool(comp['health_delta_vs_control']['no_regression'])}`",
                f"- reconstructed health no_regression: `{bool(comp['reconstructed_health_delta_vs_control']['no_regression'])}`",
                f"- status: `{comp['status']}`",
                "",
            ]
        )
    if incumbent:
        lines.append(f"- incumbent anchor64 geom: `{float(incumbent['eval_geom']):.4f}`")
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
