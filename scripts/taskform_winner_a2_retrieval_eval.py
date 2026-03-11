from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from taskform_phase12_common import UNIT_WORDS, evaluate_frame, formula_count, resolve_path, safe_text, write_json, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MARKER_RICH_RE = re.compile(r"<gap>|[\[\]{}<>]|[₀₁₂₃₄₅₆₇₈₉]|(?<!\w)x(?!\w)", flags=re.IGNORECASE)
SOURCE_MEASURE_RE = re.compile(r"\b(?:ma-na|ma\.na|gin2|g[íi]n|gur|qa|sila3|iku|sar)\b", flags=re.IGNORECASE)
FORMULA_SOURCE_RE = re.compile(r"\b(?:um-ma|q[ií]-bi|ki-a-am|li-mu-um)\b", flags=re.IGNORECASE)
STRIP_TOKEN_RE = re.compile(r"^[\[\]{}()<>.,;:\"'`]+|[\[\]{}()<>.,;:\"'`]+$")


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


def _strip_task_prefix(text: str, task_prefix: str) -> str:
    value = safe_text(text).strip()
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value


def _normalize_source_token(token: str) -> str:
    value = STRIP_TOKEN_RE.sub("", safe_text(token).strip().lower())
    if not value or value == "<gap>":
        return ""
    return value


def _source_tokens(text: str) -> list[str]:
    return [_normalize_source_token(token) for token in safe_text(text).split()]


def _compute_source_token_freq(texts: list[str]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for text in texts:
        for token in _source_tokens(text):
            if not token:
                continue
            freq[token] += 1
    return freq


def _has_rare_source_token(text: str, token_freq: Counter[str]) -> bool:
    for token in _source_tokens(text):
        if not token:
            continue
        if len(token) < 4:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        if token_freq.get(token, 0) <= 2:
            return True
    return False


def _bucket_flags(frame: pd.DataFrame, token_freq: Counter[str]) -> pd.DataFrame:
    out = frame.copy()
    out["bucket_marker_rich"] = out["source_original"].fillna("").astype(str).str.contains(MARKER_RICH_RE, regex=True)
    out["bucket_measure"] = (
        out["source_original"].fillna("").astype(str).str.contains(SOURCE_MEASURE_RE, regex=True)
        | out["reference_original"].fillna("").astype(str).str.lower().map(
            lambda text: any(unit in text for unit in UNIT_WORDS)
        )
    )
    out["bucket_formula"] = (
        out["reference_original"].fillna("").astype(str).map(lambda text: formula_count(text) > 0)
        | out["source_original"].fillna("").astype(str).str.contains(FORMULA_SOURCE_RE, regex=True)
    )
    out["bucket_rare_name"] = out["source_original"].fillna("").astype(str).map(lambda text: _has_rare_source_token(text, token_freq))
    return out


def _compare_output_health(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_health = candidate.get("output_health", {}) or {}
    baseline_health = baseline.get("output_health", {}) or {}
    keys = [
        "empty_prediction_ratio_pct",
        "copy_source_ratio_pct",
        "pred_shorter_than_half_ref_ratio_pct",
        "unique_prediction_ratio_pct",
        "has_bad_token_regex_ratio_pct",
        "exact_extra_id_0_ratio_pct",
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


def _compute_targeted_bucket_metrics(pred_csv: Path, meta_df: pd.DataFrame, token_freq: Counter[str]) -> dict[str, Any]:
    pred_df = pd.read_csv(pred_csv)
    merged = pred_df.merge(
        meta_df,
        on=["oare_id", "parent_oare_id"],
        how="left",
        suffixes=("", "_meta"),
    )
    merged["reference_original"] = merged["reference_original"].fillna(merged["reference"]).astype(str)
    merged = _bucket_flags(merged, token_freq)

    results: dict[str, Any] = {}
    for label, flag_col in (
        ("rare_name", "bucket_rare_name"),
        ("measure", "bucket_measure"),
        ("formula", "bucket_formula"),
        ("marker_rich", "bucket_marker_rich"),
    ):
        subset = merged.loc[merged[flag_col].fillna(False)].copy().reset_index(drop=True)
        summary = evaluate_frame(
            subset,
            prediction_col="prediction",
            reference_col="reference_original",
            tag=f"targeted_{label}",
            subset_name=f"targeted_{label}",
            note="audit-only bucket on anchor64 diagnostic rows",
        )
        summary["rows"] = int(len(subset))
        results[label] = summary
    return results


def _load_val_meta(control_cfg_path: Path, fold: int) -> tuple[pd.DataFrame, Counter[str]]:
    cfg = _load_yaml(control_cfg_path)
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    task_prefix = safe_text(preprocess_cfg.get("task_prefix", "")).strip()
    if task_prefix and not task_prefix.endswith(" "):
        task_prefix += " "
    processed_dir = resolve_path((cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_proc = pd.read_csv(processed_dir / "train_proc.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")
    merged = train_proc.merge(folds[["oare_id", "fold"]], on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(fold)].copy().reset_index(drop=True)
    train_texts = train_visible["source"].fillna("").astype(str).map(lambda text: _strip_task_prefix(text, task_prefix)).tolist()
    token_freq = _compute_source_token_freq(train_texts)
    meta_df = val_visible[["oare_id", "parent_oare_id", "source", "target"]].copy()
    meta_df["source_original"] = meta_df["source"].fillna("").astype(str).map(lambda text: _strip_task_prefix(text, task_prefix))
    meta_df["reference_original"] = meta_df["target"].fillna("").astype(str)
    return meta_df[["oare_id", "parent_oare_id", "source_original", "reference_original"]], token_freq


def _evaluate_candidate(
    *,
    label: str,
    cfg_path: Path,
    fold: int,
    tag: str,
    hard_ids_csv: Path,
    meta_df: pd.DataFrame,
    token_freq: Counter[str],
) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    run_dir = _candidate_run_dir(cfg, fold)
    checkpoint_dir = run_dir / "best_model"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing best_model for {label}: {checkpoint_dir}")
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
                "16",
                "--max-val-samples",
                "64",
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
                "16",
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
                "64",
            ]
        )
    diag_payload = _load_json(diagnose_summary_path)
    run_summary_path = run_dir / "run_summary.json"
    run_summary = _load_json(run_summary_path)

    reconstructed_csv_str = (((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    pred_csv_str = ((diag_payload.get("artifacts", {}) or {}).get("predictions_csv", ""))
    reconstructed_csv = Path(reconstructed_csv_str)
    pred_csv = Path(pred_csv_str)

    hard_subset = _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv)
    targeted_buckets = _compute_targeted_bucket_metrics(pred_csv, meta_df, token_freq)

    return {
        "label": label,
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "run_summary_path": str(run_summary_path),
        "anchor64": {
            "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
            "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
            "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
            "elapsed_seconds": float(decode_payload.get("elapsed_seconds", 0.0) or 0.0),
        },
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "reconstructed_health": (((diag_payload.get("reconstructed", {}) or {}).get("output_health", {}) or {})),
        "hard_subset": hard_subset,
        "targeted_buckets": targeted_buckets,
        "train_runtime_seconds": float(run_summary.get("train_runtime_seconds", 0.0) or 0.0),
        "gpu_peak_utilization_pct": float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--control-config",
        default="reports/taskform_winner_a2_retrieval_20260310/generated_configs/taskform_winner_a2_retrieval_control.yaml",
    )
    ap.add_argument(
        "--retrieval-config",
        default="reports/taskform_winner_a2_retrieval_20260310/generated_configs/taskform_winner_a2_retrieval_top1.yaml",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_retrieval_eval_20260310")
    args = ap.parse_args()

    control_cfg_path = resolve_path(args.control_config, REPO_ROOT / "reports")
    retrieval_cfg_path = resolve_path(args.retrieval_config, REPO_ROOT / "reports")
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_eval_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)

    meta_df, token_freq = _load_val_meta(control_cfg_path, args.fold)
    control = _evaluate_candidate(
        label="R0_control",
        cfg_path=control_cfg_path,
        fold=args.fold,
        tag="taskform_winner_a2_r0_anchor64_20260310",
        hard_ids_csv=hard_ids_csv,
        meta_df=meta_df,
        token_freq=token_freq,
    )
    retrieval = _evaluate_candidate(
        label="R1_retrieval_top1",
        cfg_path=retrieval_cfg_path,
        fold=args.fold,
        tag="taskform_winner_a2_r1_anchor64_20260310",
        hard_ids_csv=hard_ids_csv,
        meta_df=meta_df,
        token_freq=token_freq,
    )
    incumbent = _load_incumbent_anchor()

    delta_anchor = float(retrieval["anchor64"]["eval_geom"]) - float(control["anchor64"]["eval_geom"])
    delta_hard = float(retrieval["hard_subset"]["eval_geom"]) - float(control["hard_subset"]["eval_geom"])
    latency_ratio = float(retrieval["anchor64"]["elapsed_seconds"]) / max(1e-9, float(control["anchor64"]["elapsed_seconds"]))
    health_delta = _compare_output_health(retrieval, control)
    targeted_bucket_deltas = {
        bucket: round(
            float((retrieval["targeted_buckets"].get(bucket, {}) or {}).get("eval_geom", 0.0))
            - float((control["targeted_buckets"].get(bucket, {}) or {}).get("eval_geom", 0.0)),
            4,
        )
        for bucket in ("rare_name", "measure", "formula", "marker_rich")
    }
    positive_buckets = [bucket for bucket, delta in targeted_bucket_deltas.items() if delta > 0.0]

    if delta_anchor >= 0.25 and delta_hard >= 0.0 and bool(positive_buckets) and latency_ratio <= 1.8 and bool(health_delta["no_regression"]):
        status = "accept_to_w"
        reason = "retrieval top1 clears anchor64, hard subset, targeted bucket, latency, and health gates"
    elif delta_anchor > 0.0 and delta_hard >= 0.0 and latency_ratio <= 1.8:
        status = "review_stop"
        reason = "retrieval is locally positive but does not clear the full P gate"
    else:
        status = "reject_stop"
        reason = "retrieval top1 fails the matched smoke gate against R0"

    summary = {
        "line": "A2_retrieval",
        "status": status,
        "reason": reason,
        "fold": int(args.fold),
        "control": control,
        "retrieval_top1": retrieval,
        "incumbent_anchor64": incumbent,
        "delta_geom_r1_vs_r0_anchor64": round(delta_anchor, 4),
        "delta_geom_r1_vs_r0_hard_subset": round(delta_hard, 4),
        "delta_geom_r1_vs_i0_anchor64": round(
            float(retrieval["anchor64"]["eval_geom"]) - float(incumbent.get("eval_geom", 0.0) or 0.0),
            4,
        )
        if incumbent
        else None,
        "decode_latency_ratio_r1_vs_r0": round(latency_ratio, 4),
        "health_delta_r1_vs_r0": health_delta,
        "targeted_bucket_deltas_r1_vs_r0": targeted_bucket_deltas,
        "positive_targeted_buckets": positive_buckets,
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A2 Retrieval Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- control anchor64 geom: `{control['anchor64']['eval_geom']:.4f}`",
        f"- retrieval anchor64 geom: `{retrieval['anchor64']['eval_geom']:.4f}`",
        f"- delta geom R1-R0 anchor64: `{delta_anchor:.4f}`",
        f"- control hard geom: `{control['hard_subset']['eval_geom']:.4f}`",
        f"- retrieval hard geom: `{retrieval['hard_subset']['eval_geom']:.4f}`",
        f"- delta geom R1-R0 hard: `{delta_hard:.4f}`",
        f"- latency ratio R1/R0: `{latency_ratio:.4f}`",
        f"- health no_regression: `{bool(health_delta['no_regression'])}`",
        f"- positive targeted buckets: `{', '.join(positive_buckets) if positive_buckets else 'none'}`",
    ]
    if incumbent:
        lines.append(f"- incumbent anchor64 geom: `{float(incumbent['eval_geom']):.4f}`")
        lines.append(
            f"- delta geom R1-I0 anchor64: `{float(retrieval['anchor64']['eval_geom']) - float(incumbent['eval_geom']):.4f}`"
        )
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
