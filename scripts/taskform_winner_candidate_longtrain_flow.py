#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import time
import traceback
from pathlib import Path
from typing import Any

from taskform_a2_a1_flow import (
    REPO_ROOT,
    _ensure_incumbent_anchor,
    _load_json,
    _load_yaml,
    _materialize_cfg,
    _resolve_path,
    _run,
)
from taskform_phase12_common import safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import (
    _compare_output_health,
    _evaluate_candidate,
    _load_val_meta,
)
from taskform_winner_a2_retrieval_wlite_eval import _evaluate_fullval_candidate, _load_incumbent_fullval


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


def _run_train_if_needed(
    *,
    python_exec: str,
    cfg_path: Path,
    fold: int,
    max_steps: int,
    eval_steps: int,
    init_adapter_dir: Path,
) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    run_root = _resolve_path((cfg.get("paths", {}) or {}).get("run_dir"), REPO_ROOT / "runs" / "missing")
    run_dir = run_root.parent / f"{run_root.name}_fold{fold}"
    best_model_dir = run_dir / "best_model"
    run_summary_path = run_dir / "run_summary.json"
    if best_model_dir.exists() and run_summary_path.exists():
        return {
            "reused_existing_run": True,
            "run_dir": str(run_dir),
            "best_model_dir": str(best_model_dir),
            "run_summary_path": str(run_summary_path),
        }

    _run(
        [
            python_exec,
            str(REPO_ROOT / "scripts" / "train_mt5_lora.py"),
            "--config",
            str(cfg_path),
            "--fold",
            str(fold),
            "--max-steps",
            str(max_steps),
            "--eval-steps",
            str(eval_steps),
            "--skip-final-predict",
            "--init-adapter-dir",
            str(init_adapter_dir),
        ]
    )
    return {
        "reused_existing_run": False,
        "run_dir": str(run_dir),
        "best_model_dir": str(best_model_dir),
        "run_summary_path": str(run_summary_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--init-adapter-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument(
        "--incumbent-config",
        default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml",
    )
    ap.add_argument(
        "--incumbent-checkpoint-dir",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250",
    )
    args = ap.parse_args()

    python_exec = str(REPO_ROOT / ".venv-deeppast" / "bin" / "python")
    started = time.time()

    base_cfg_path = _resolve_path(args.base_config, REPO_ROOT / "reports")
    processed_dir = _resolve_path(args.processed_dir, REPO_ROOT / "data" / "processed")
    init_adapter_dir = _resolve_path(args.init_adapter_dir, REPO_ROOT / "runs")
    hard_ids_csv = _resolve_path(
        args.hard_ids_csv,
        REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv",
    )
    incumbent_cfg_path = _resolve_path(
        args.incumbent_config,
        REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
    )
    incumbent_checkpoint_dir = _resolve_path(
        args.incumbent_checkpoint_dir,
        REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
    )
    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = report_dir / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "status.json"
    summary_path = report_dir / "summary.json"
    gate_report_path = report_dir / "gate_report.md"

    base_cfg = _load_yaml(base_cfg_path)
    cfg_path = generated_cfg_dir / f"{safe_text(args.label)}.yaml"
    cfg = _materialize_cfg(
        base_cfg=copy.deepcopy(base_cfg),
        output_path=cfg_path,
        processed_dir=processed_dir,
        run_dir=f"runs/{safe_text(args.run_name)}",
    )

    try:
        _write_status(
            status_path,
            stage="started",
            status="running",
            extra={
                "label": safe_text(args.label),
                "report_dir": str(report_dir),
                "config_path": str(cfg_path),
            },
        )

        _write_status(status_path, stage="refs", status="running")
        incumbent_anchor = _ensure_incumbent_anchor(
            python_exec=python_exec,
            base_cfg_path=incumbent_cfg_path,
            checkpoint_dir=incumbent_checkpoint_dir,
            fold=int(args.fold),
            out_dir=report_dir,
        )
        incumbent_fullval = _load_incumbent_fullval(hard_ids_csv)
        frozen_candidate = _load_frozen_candidate()
        _write_status(
            status_path,
            stage="refs",
            status="done",
            extra={
                "incumbent_anchor64_geom": float(incumbent_anchor.get("eval_geom", 0.0) or 0.0),
                "incumbent_fullval_geom": float(incumbent_fullval.get("eval_geom", 0.0) or 0.0),
            },
        )

        _write_status(status_path, stage="train", status="running")
        train_info = _run_train_if_needed(
            python_exec=python_exec,
            cfg_path=cfg_path,
            fold=int(args.fold),
            max_steps=int(args.max_steps),
            eval_steps=int(args.eval_steps),
            init_adapter_dir=init_adapter_dir,
        )
        run_summary = _load_json(Path(str(train_info["run_summary_path"])))
        _write_status(
            status_path,
            stage="train",
            status="done",
            extra={
                "run_dir": str(train_info["run_dir"]),
                "reused_existing_run": bool(train_info["reused_existing_run"]),
                "best_eval_loss": float(run_summary.get("best_eval_loss", 0.0) or 0.0),
            },
        )

        meta_df, token_freq = _load_val_meta(cfg_path, int(args.fold))

        _write_status(status_path, stage="anchor64_eval", status="running")
        anchor_eval = _evaluate_candidate(
            label=safe_text(args.label),
            cfg_path=cfg_path,
            fold=int(args.fold),
            tag=f"{safe_text(args.label)}_anchor64_{safe_text(args.run_suffix)}",
            hard_ids_csv=hard_ids_csv,
            meta_df=meta_df,
            token_freq=token_freq,
        )
        _write_status(
            status_path,
            stage="anchor64_eval",
            status="done",
            extra={
                "anchor64_geom": float(((anchor_eval.get("anchor64", {}) or {}).get("eval_geom", 0.0) or 0.0)),
            },
        )

        _write_status(status_path, stage="fullval_eval", status="running")
        fullval_eval = _evaluate_fullval_candidate(
            cfg_path=cfg_path,
            fold=int(args.fold),
            tag=f"{safe_text(args.label)}_fullval_{safe_text(args.run_suffix)}",
            hard_ids_csv=hard_ids_csv,
        )
        _write_status(
            status_path,
            stage="fullval_eval",
            status="done",
            extra={"fullval_geom": float(fullval_eval.get("eval_geom", 0.0) or 0.0)},
        )

        anchor_geom = float(((anchor_eval.get("anchor64", {}) or {}).get("eval_geom", 0.0) or 0.0))
        anchor_hard = float(((anchor_eval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))
        fullval_geom = float(fullval_eval.get("eval_geom", 0.0) or 0.0)
        fullval_hard = float(((fullval_eval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))

        comparisons = {
            "anchor64_vs_incumbent": round(anchor_geom - float(incumbent_anchor.get("eval_geom", 0.0) or 0.0), 4),
            "anchor64_vs_frozen": round(anchor_geom - float(frozen_candidate.get("anchor64_reconstructed_geom", 0.0) or 0.0), 4),
            "fullval_vs_incumbent": round(fullval_geom - float(incumbent_fullval.get("eval_geom", 0.0) or 0.0), 4),
            "fullval_vs_frozen": round(fullval_geom - float(frozen_candidate.get("fullval_reconstructed_geom", 0.0) or 0.0), 4),
            "hard_vs_incumbent": round(
                fullval_hard - float(((incumbent_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0)),
                4,
            ),
            "hard_vs_frozen": round(fullval_hard - float(frozen_candidate.get("hard_geom", 0.0) or 0.0), 4),
        }

        health = {
            "anchor64_vs_incumbent": _compare_output_health(anchor_eval, incumbent_anchor),
            "fullval_vs_incumbent": _compare_output_health(fullval_eval, incumbent_fullval),
        }

        summary = {
            "line": "winner_candidate_longtrain",
            "label": safe_text(args.label),
            "status": "completed_review_pending",
            "reason": "long-train candidate completed with anchor64/fullval/hard measurements",
            "base_config_path": str(base_cfg_path),
            "generated_config_path": str(cfg_path),
            "processed_dir": str(processed_dir),
            "init_adapter_dir": str(init_adapter_dir),
            "run_name": safe_text(args.run_name),
            "fold": int(args.fold),
            "max_steps": int(args.max_steps),
            "eval_steps": int(args.eval_steps),
            "report_dir": str(report_dir),
            "train": train_info,
            "run_summary": run_summary,
            "incumbent_anchor64": incumbent_anchor,
            "incumbent_fullval": incumbent_fullval,
            "frozen_candidate": frozen_candidate,
            "anchor_eval": anchor_eval,
            "fullval_eval": fullval_eval,
            "comparisons": comparisons,
            "health": health,
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        }
        write_json(summary_path, summary)

        lines = [
            f"# Winner Candidate Long Train: {safe_text(args.label)}",
            "",
            "- status: `completed_review_pending`",
            "- reason: long-train candidate completed with anchor64/fullval/hard measurements",
            f"- summary: `{summary_path}`",
            "",
            "## Candidate",
            "",
            f"- init adapter: `{init_adapter_dir}`",
            f"- run dir: `{train_info['run_dir']}`",
            f"- max steps: `{int(args.max_steps)}`",
            f"- eval steps: `{int(args.eval_steps)}`",
            f"- reused existing run: `{bool(train_info['reused_existing_run'])}`",
            "",
            "## Anchor64",
            "",
            f"- geom: `{anchor_geom:.4f}`",
            f"- hard geom: `{anchor_hard:.4f}`",
            f"- delta vs incumbent: `{comparisons['anchor64_vs_incumbent']:+.4f}`",
            f"- delta vs frozen: `{comparisons['anchor64_vs_frozen']:+.4f}`",
            f"- health no_regression vs incumbent: `{bool((health['anchor64_vs_incumbent'] or {}).get('no_regression', False))}`",
            "",
            "## Fullval",
            "",
            f"- geom: `{fullval_geom:.4f}`",
            f"- hard geom: `{fullval_hard:.4f}`",
            f"- delta vs incumbent: `{comparisons['fullval_vs_incumbent']:+.4f}`",
            f"- delta vs frozen: `{comparisons['fullval_vs_frozen']:+.4f}`",
            f"- hard delta vs incumbent: `{comparisons['hard_vs_incumbent']:+.4f}`",
            f"- hard delta vs frozen: `{comparisons['hard_vs_frozen']:+.4f}`",
            f"- health no_regression vs incumbent: `{bool((health['fullval_vs_incumbent'] or {}).get('no_regression', False))}`",
            "",
        ]
        write_text(gate_report_path, "\n".join(lines) + "\n")

        _write_status(
            status_path,
            stage="completed",
            status="completed_review_pending",
            extra={
                "summary_json": str(summary_path),
                "gate_report_md": str(gate_report_path),
                "anchor64_geom": anchor_geom,
                "fullval_geom": fullval_geom,
            },
        )
    except Exception as exc:  # pragma: no cover - operational path
        _write_status(
            status_path,
            stage="failed",
            status="failed",
            extra={
                "error": safe_text(str(exc)),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
