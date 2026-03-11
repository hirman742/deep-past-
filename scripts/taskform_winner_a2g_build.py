from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from generation_utils import apply_task_prefix, normalize_task_prefix
from taskform_phase12_common import REPO_ROOT, UNIT_WORDS, resolve_path, safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import (
    FORMULA_SOURCE_RE,
    MARKER_RICH_RE,
    SOURCE_MEASURE_RE,
    _compute_source_token_freq,
    _has_rare_source_token,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_thresholds(spec: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for piece in safe_text(spec).split(","):
        chunk = piece.strip()
        if not chunk:
            continue
        label, value = chunk.split(":", 1)
        threshold = float(value)
        items.append(
            {
                "label": label.strip().lower(),
                "threshold": threshold,
            }
        )
    if not items:
        raise ValueError("No valid thresholds parsed")
    return items


def _first_score(value: Any) -> float:
    text = safe_text(value).strip()
    if not text:
        return 0.0
    try:
        return float(text.split("|", 1)[0].strip())
    except ValueError:
        return 0.0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _bucket_flags(frame: pd.DataFrame, token_freq: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    out["bucket_marker_rich"] = out["source_raw"].fillna("").astype(str).str.contains(MARKER_RICH_RE, regex=True)
    out["bucket_measure"] = (
        out["source_raw"].fillna("").astype(str).str.contains(SOURCE_MEASURE_RE, regex=True)
        | out["target_raw"].fillna("").astype(str).str.lower().map(lambda text: any(unit in text for unit in UNIT_WORDS))
    )
    out["bucket_formula"] = (
        out["source_raw"].fillna("").astype(str).str.contains(FORMULA_SOURCE_RE, regex=True)
        | out["target_raw"].fillna("").astype(str).str.contains(r"Seal of|Sealed by", regex=True)
    )
    out["bucket_rare_name"] = out["source_raw"].fillna("").astype(str).map(
        lambda text: _has_rare_source_token(text, token_freq)
    )
    return out


def _write_processed_dir(train_df: pd.DataFrame, folds_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_proc.csv", index=False)
    folds_df.to_csv(out_dir / "folds.csv", index=False)


def _clone_cfg(
    *,
    base_cfg: dict[str, Any],
    label: str,
    processed_dir: Path,
    run_dir_name: str,
    init_adapter_dir: Path,
    config_dir: Path,
) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["name"] = f"taskform_winner_a2g_{label}"
    paths_cfg = (cfg.get("paths", {}) or {}).copy()
    paths_cfg["processed_dir"] = _rel(processed_dir)
    paths_cfg["run_dir"] = f"runs/{run_dir_name}"
    cfg["paths"] = paths_cfg
    tapt_cfg = (cfg.get("tapt", {}) or {}).copy()
    tapt_cfg["init_adapter_dir"] = _rel(init_adapter_dir)
    cfg["tapt"] = tapt_cfg
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"taskform_winner_a2g_{label}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return cfg_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="reports/taskform_winner_a2_retrieval_20260310/generated_configs/taskform_winner_a2_retrieval_top1.yaml",
    )
    ap.add_argument(
        "--base-processed-dir",
        default="data/processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0",
    )
    ap.add_argument(
        "--init-adapter-dir",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thresholds", default="g35:0.35,g50:0.50,g65:0.65")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2g_build_20260311")
    ap.add_argument("--config-dir", default="reports/taskform_winner_a2g_build_20260311/generated_configs")
    args = ap.parse_args()

    base_cfg_path = resolve_path(
        args.base_config,
        REPO_ROOT
        / "reports"
        / "taskform_winner_a2_retrieval_20260310"
        / "generated_configs"
        / "taskform_winner_a2_retrieval_top1.yaml",
    )
    base_cfg = _load_yaml(base_cfg_path)
    base_processed_dir = resolve_path(
        args.base_processed_dir,
        REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0",
    )
    init_adapter_dir = resolve_path(
        args.init_adapter_dir,
        REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
    )
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2g_build_20260311")
    config_dir = resolve_path(args.config_dir, report_dir / "generated_configs")
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.thresholds)
    preprocess_cfg = base_cfg.get("preprocess", {}) or {}
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))

    base_train = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(base_processed_dir / "folds.csv")
    merged = base_train.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    merged["top1_score"] = merged["retrieval_scores"].map(_first_score).astype(float)
    merged["source_raw"] = merged["source_raw"].fillna("").astype(str)
    merged["target_raw"] = merged["target_raw"].fillna("").astype(str)

    train_visible = merged.loc[merged["fold"] != int(args.fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(args.fold)].copy().reset_index(drop=True)

    token_freq = _compute_source_token_freq(train_visible["source_raw"].tolist())
    bucketed = _bucket_flags(merged, token_freq)

    score_quantiles = {
        "all": {
            "p05": round(float(bucketed["top1_score"].quantile(0.05)), 4),
            "p50": round(float(bucketed["top1_score"].quantile(0.50)), 4),
            "p95": round(float(bucketed["top1_score"].quantile(0.95)), 4),
            "mean": round(float(bucketed["top1_score"].mean()), 4),
        },
        "train_visible": {
            "p05": round(float(train_visible["top1_score"].quantile(0.05)), 4),
            "p50": round(float(train_visible["top1_score"].quantile(0.50)), 4),
            "p95": round(float(train_visible["top1_score"].quantile(0.95)), 4),
            "mean": round(float(train_visible["top1_score"].mean()), 4),
        },
        "val_visible": {
            "p05": round(float(val_visible["top1_score"].quantile(0.05)), 4),
            "p50": round(float(val_visible["top1_score"].quantile(0.50)), 4),
            "p95": round(float(val_visible["top1_score"].quantile(0.95)), 4),
            "mean": round(float(val_visible["top1_score"].mean()), 4),
        },
    }

    coverage_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    control_cfg_path = _clone_cfg(
        base_cfg=base_cfg,
        label="ctrl",
        processed_dir=base_processed_dir,
        run_dir_name="TASKFORM_WINNER_A2G_CTRL_20260311",
        init_adapter_dir=init_adapter_dir,
        config_dir=config_dir,
    )
    control_payload = {
        "label": "ctrl",
        "threshold": None,
        "processed_dir": str(base_processed_dir),
        "config_path": str(control_cfg_path),
        "run_dir": "runs/TASKFORM_WINNER_A2G_CTRL_20260311",
        "train_visible_enabled_rows": int(train_visible.shape[0]),
        "val_visible_enabled_rows": int(val_visible.shape[0]),
    }
    candidates["ctrl"] = control_payload
    manifest_rows.append(control_payload)

    for item in thresholds:
        label = str(item["label"])
        threshold = float(item["threshold"])
        gated = base_train.copy()
        gated["top1_score"] = gated["retrieval_scores"].map(_first_score).astype(float)
        gated["retrieval_gate_threshold"] = threshold
        gated["retrieval_gate_enabled"] = gated["top1_score"] >= threshold
        gated["source"] = [
            current_source if enabled else apply_task_prefix(source_raw, task_prefix)
            for current_source, enabled, source_raw in zip(
                gated["source"].fillna("").astype(str).tolist(),
                gated["retrieval_gate_enabled"].fillna(False).astype(bool).tolist(),
                gated["source_raw"].fillna("").astype(str).tolist(),
            )
        ]
        gated.loc[~gated["retrieval_gate_enabled"], "retrieval_hint_source"] = ""
        gated.loc[~gated["retrieval_gate_enabled"], "retrieval_hint_target"] = ""
        gated.loc[~gated["retrieval_gate_enabled"], "retrieval_mode"] = "gate_off"

        out_dir = REPO_ROOT / f"data/processed_byt5_chunks_align_gc_cost14_retrieval_top1_{label}_fold0"
        _write_processed_dir(gated, folds_df.copy(), out_dir)
        cfg_path = _clone_cfg(
            base_cfg=base_cfg,
            label=label,
            processed_dir=out_dir,
            run_dir_name=f"TASKFORM_WINNER_A2G_{label.upper()}_20260311",
            init_adapter_dir=init_adapter_dir,
            config_dir=config_dir,
        )

        gated_merged = gated.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
        train_gate = gated_merged.loc[gated_merged["fold"] != int(args.fold), "retrieval_gate_enabled"].fillna(False).astype(bool)
        val_gate = gated_merged.loc[gated_merged["fold"] == int(args.fold), "retrieval_gate_enabled"].fillna(False).astype(bool)

        payload = {
            "label": label,
            "threshold": threshold,
            "processed_dir": str(out_dir),
            "config_path": str(cfg_path),
            "run_dir": f"runs/TASKFORM_WINNER_A2G_{label.upper()}_20260311",
            "train_visible_enabled_rows": int(train_gate.sum()),
            "train_visible_disabled_rows": int((~train_gate).sum()),
            "val_visible_enabled_rows": int(val_gate.sum()),
            "val_visible_disabled_rows": int((~val_gate).sum()),
        }
        candidates[label] = payload
        manifest_rows.append(payload)

        for split_name, split_df in (
            ("train_visible", bucketed.loc[bucketed["fold"] != int(args.fold)].copy()),
            ("val_visible", bucketed.loc[bucketed["fold"] == int(args.fold)].copy()),
        ):
            enabled = split_df["top1_score"] >= threshold
            coverage_rows.append(
                {
                    "gate_label": label,
                    "threshold": threshold,
                    "split": split_name,
                    "total_rows": int(len(split_df)),
                    "enabled_rows": int(enabled.sum()),
                    "disabled_rows": int((~enabled).sum()),
                    "enabled_ratio_pct": round(100.0 * float(enabled.mean()), 4) if len(split_df) else 0.0,
                }
            )
            for bucket_name, flag_col in (
                ("rare_name", "bucket_rare_name"),
                ("measure", "bucket_measure"),
                ("formula", "bucket_formula"),
                ("marker_rich", "bucket_marker_rich"),
            ):
                subset = split_df.loc[split_df[flag_col].fillna(False)].copy()
                if subset.empty:
                    bucket_rows.append(
                        {
                            "gate_label": label,
                            "threshold": threshold,
                            "split": split_name,
                            "bucket": bucket_name,
                            "total_rows": 0,
                            "enabled_rows": 0,
                            "disabled_rows": 0,
                            "enabled_ratio_pct": 0.0,
                        }
                    )
                    continue
                subset_enabled = subset["top1_score"] >= threshold
                bucket_rows.append(
                    {
                        "gate_label": label,
                        "threshold": threshold,
                        "split": split_name,
                        "bucket": bucket_name,
                        "total_rows": int(len(subset)),
                        "enabled_rows": int(subset_enabled.sum()),
                        "disabled_rows": int((~subset_enabled).sum()),
                        "enabled_ratio_pct": round(100.0 * float(subset_enabled.mean()), 4),
                    }
                )

    audit_df = bucketed[["oare_id", "parent_oare_id", "fold", "top1_score"]].copy()
    for item in thresholds:
        label = str(item["label"])
        threshold = float(item["threshold"])
        audit_df[f"{label}_enabled"] = audit_df["top1_score"] >= threshold
    audit_df.to_csv(report_dir / "score_gate_rowwise.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(report_dir / "score_gate_coverage.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(report_dir / "score_gate_bucket_coverage.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(report_dir / "processed_dir_manifest.csv", index=False)

    summary = {
        "status": "ready_for_probe",
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "init_adapter_dir": str(init_adapter_dir),
        "fold": int(args.fold),
        "thresholds": thresholds,
        "rows": {
            "total": int(merged.shape[0]),
            "train_visible": int(train_visible.shape[0]),
            "val_visible": int(val_visible.shape[0]),
        },
        "score_quantiles": score_quantiles,
        "artifacts": {
            "score_gate_audit_json": str(report_dir / "score_gate_audit.json"),
            "score_gate_coverage_csv": str(report_dir / "score_gate_coverage.csv"),
            "score_gate_bucket_coverage_csv": str(report_dir / "score_gate_bucket_coverage.csv"),
            "score_gate_rowwise_csv": str(report_dir / "score_gate_rowwise.csv"),
            "processed_dir_manifest_csv": str(report_dir / "processed_dir_manifest.csv"),
        },
        "candidates": candidates,
    }
    write_json(report_dir / "score_gate_audit.json", summary)
    write_json(report_dir / "summary.json", summary)
    write_json(report_dir / "manifest.json", summary)

    lines = [
        "# A2g Build Report",
        "",
        f"- status: `{summary['status']}`",
        f"- base processed dir: `{summary['base_processed_dir']}`",
        f"- init adapter dir: `{summary['init_adapter_dir']}`",
        f"- total rows: `{summary['rows']['total']}`",
        f"- train visible rows: `{summary['rows']['train_visible']}`",
        f"- val visible rows: `{summary['rows']['val_visible']}`",
        f"- val top1 score p50: `{summary['score_quantiles']['val_visible']['p50']:.4f}`",
        f"- val top1 score p95: `{summary['score_quantiles']['val_visible']['p95']:.4f}`",
        "",
        "## Candidates",
        "",
        f"- ctrl: `{candidates['ctrl']['processed_dir']}`",
    ]
    for item in thresholds:
        label = str(item["label"])
        payload = candidates[label]
        lines.extend(
            [
                f"- {label}: threshold=`{payload['threshold']:.2f}`, train_enabled=`{payload['train_visible_enabled_rows']}`, val_enabled=`{payload['val_visible_enabled_rows']}`",
            ]
        )
    write_text(report_dir / "report.md", "\n".join(lines) + "\n")

    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
