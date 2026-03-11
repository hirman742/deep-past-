#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from taskform_a2_a1_flow import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _build_monolingual_assets,
    _build_synthetic_mix_dir,
    _compute_hard_subset_metrics,
    _deep_merge,
    _ensure_incumbent_anchor,
    _load_json,
    _load_yaml,
    _materialize_cfg,
    _maybe_run_fullval,
    _resolve_path,
    _run,
    _run_candidate_probe,
)
from taskform_phase12_common import markdown_table, safe_text, write_json, write_text


def _write_status(path: Path, *, stage: str, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)


def _run_winner_tapt_smoke(
    *,
    python_exec: str,
    base_cfg: dict[str, Any],
    out_dir: Path,
    mono_csv: Path,
    max_rows: int,
    run_suffix: str,
) -> dict[str, Any]:
    cfg_path = out_dir / "generated_configs" / f"taskform_winner_pseudotarget_tapt_smoke_{run_suffix}.yaml"
    run_root = REPO_ROOT / "runs" / f"TASKFORM_WINNER_A1_PSEUDOTARGET_TAPT_SMOKE_{run_suffix}"
    summary_path = run_root / "tapt_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        return {
            "config_path": str(cfg_path),
            "run_dir": str(run_root),
            "summary_path": str(summary_path),
            "best_model_dir": str(summary.get("best_model_dir", "")),
            "train_rows": int(summary.get("train_rows", 0)),
            "eval_rows": int(summary.get("eval_rows", 0)),
            "eval_loss": float((summary.get("eval_metrics", {}) or {}).get("eval_loss", 0.0) or 0.0),
        }
    cfg = _deep_merge(
        copy.deepcopy(base_cfg),
        {
            "name": f"taskform_winner_pseudotarget_tapt_smoke_{run_suffix.lower()}",
            "tapt": {
                "run_dir": str(run_root.relative_to(REPO_ROOT)),
                "max_source_length": 640,
                "max_target_length": 384,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1.0e-4,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "lr_scheduler_type": "cosine",
                "num_train_epochs": 1,
                "max_steps": 300,
                "eval_steps": 100,
                "fp16": False,
                "bf16": True,
            },
        },
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _run(
        [
            python_exec,
            str(SCRIPTS_DIR / "tapt_denoise.py"),
            "--config",
            str(cfg_path),
            "--corpus-csvs",
            str(mono_csv),
            "--output-run-dir",
            str(run_root),
            "--max-rows",
            str(int(max_rows)),
        ]
    )
    summary = _load_json(summary_path)
    return {
        "config_path": str(cfg_path),
        "run_dir": str(run_root),
        "summary_path": str(summary_path),
        "best_model_dir": str(summary.get("best_model_dir", "")),
        "train_rows": int(summary.get("train_rows", 0)),
        "eval_rows": int(summary.get("eval_rows", 0)),
        "eval_loss": float((summary.get("eval_metrics", {}) or {}).get("eval_loss", 0.0) or 0.0),
    }


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


def _probe_hard_metrics(probe: dict[str, Any]) -> dict[str, Any]:
    diag = _load_json(Path(str(probe["diagnose_summary_path"])))
    reconstructed_csv = Path(
        (((diag.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    )
    hard_csv = REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv"
    return _compute_hard_subset_metrics(reconstructed_csv, hard_csv)


def _finalize_probe_status(
    *,
    incumbent_anchor: dict[str, Any],
    frozen_candidate: dict[str, Any],
    probe: dict[str, Any],
    fullval: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    probe_delta_inc = round(float(probe["eval_geom"]) - float(incumbent_anchor.get("eval_geom", 0.0) or 0.0), 4)
    probe_delta_frozen = round(
        float(probe["eval_geom"]) - float(frozen_candidate.get("anchor64_reconstructed_geom", 0.0) or 0.0),
        4,
    )
    comparisons = {
        "probe_delta_geom_vs_incumbent_anchor64": probe_delta_inc,
        "probe_delta_geom_vs_frozen_anchor64": probe_delta_frozen,
    }
    if fullval is None:
        if probe_delta_inc >= 0.05:
            return "review_continue", "anchor64 is locally positive but full-val was not executed", comparisons
        return "reject_stop", "anchor64 smoke is not positive against incumbent anchor baseline", comparisons

    fullval_delta_frozen = round(
        float(fullval.get("eval_geom", 0.0) or 0.0) - float(frozen_candidate.get("fullval_reconstructed_geom", 0.0) or 0.0),
        4,
    )
    hard_delta_frozen = round(
        float(((fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))
        - float(frozen_candidate.get("hard_geom", 0.0) or 0.0),
        4,
    )
    comparisons["fullval_delta_geom_vs_frozen"] = fullval_delta_frozen
    comparisons["fullval_hard_delta_vs_frozen"] = hard_delta_frozen
    if fullval_delta_frozen >= 0.15 and hard_delta_frozen >= -0.10:
        return "review_to_candidate_pool", "full-val clears the written frozen compare gate for an orthogonal candidate", comparisons
    if probe_delta_inc >= 0.05:
        return "review_stop", "anchor64 smoke was positive but full-val does not clear the frozen compare gate", comparisons
    return "reject_stop", "full-val and anchor64 both fail to establish a useful orthogonal candidate", comparisons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--base-checkpoint-dir", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250")
    ap.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tapt-max-rows", type=int, default=6000)
    ap.add_argument("--pseudo-limit", type=int, default=512)
    ap.add_argument("--probe-max-steps", type=int, default=180)
    ap.add_argument("--probe-eval-steps", type=int, default=90)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_pseudotarget_smoke_20260311")
    args = ap.parse_args()

    python_exec = sys.executable
    started = time.time()

    base_cfg_path = _resolve_path(args.base_config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml")
    base_checkpoint_dir = _resolve_path(args.base_checkpoint_dir, REPO_ROOT / "runs" / "missing_checkpoint")
    base_processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    out_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a1_pseudotarget_smoke_20260311")
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
            extra={
                "report_dir": str(out_dir),
                "run_suffix": str(args.run_suffix),
            },
        )

        _write_status(status_path, stage="monolingual_inventory", status="running")
        monolingual_summary = _build_monolingual_assets(out_dir=out_dir)
        _write_status(
            status_path,
            stage="monolingual_inventory",
            status="done",
            extra={"mono_rows": int(monolingual_summary["mono_total_rows"])},
        )

        _write_status(status_path, stage="incumbent_anchor64", status="running")
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
            stage="incumbent_anchor64",
            status="done",
            extra={"incumbent_anchor64_geom": float(incumbent_anchor["eval_geom"])},
        )

        _write_status(status_path, stage="tapt_smoke", status="running")
        tapt_summary = _run_winner_tapt_smoke(
            python_exec=python_exec,
            base_cfg=base_cfg,
            out_dir=out_dir,
            mono_csv=Path(str(monolingual_summary["mono_corpus_csv"])),
            max_rows=int(args.tapt_max_rows),
            run_suffix=str(args.run_suffix),
        )
        _write_status(
            status_path,
            stage="tapt_smoke",
            status="done",
            extra={"tapt_best_model_dir": str(tapt_summary["best_model_dir"])},
        )

        pseudo_csv = out_dir / "a1_pseudo_targets.csv"
        _write_status(status_path, stage="pseudo_targets", status="running")
        if not (pseudo_csv.exists() and pseudo_csv.stat().st_size > 0):
            _run(
                [
                    python_exec,
                    str(SCRIPTS_DIR / "taskform_generate_pseudo_targets.py"),
                    "--config",
                    str(base_cfg_path),
                    "--checkpoint-dir",
                    str(base_checkpoint_dir),
                    "--input-csv",
                    str(monolingual_summary["pseudo_pool_csv"]),
                    "--output-csv",
                    str(pseudo_csv),
                    "--id-col",
                    "asset_id",
                    "--source-col",
                    "transliteration",
                    "--limit",
                    str(int(args.pseudo_limit)),
                    "--predict-batch-size",
                    "8",
                    "--num-beams",
                    "4",
                    "--length-penalty",
                    "0.7",
                    "--max-new-tokens",
                    "384",
                    "--no-repeat-ngram-size",
                    "0",
                ]
            )
        _write_status(
            status_path,
            stage="pseudo_targets",
            status="done",
            extra={"pseudo_targets_csv": str(pseudo_csv)},
        )

        _write_status(status_path, stage="synthetic_mix", status="running")
        mix_summary = _build_synthetic_mix_dir(
            source_processed_dir=base_processed_dir,
            pseudo_csv=pseudo_csv,
            out_dir=out_dir,
            mix_label=f"winner_a1_pseudotarget_synthmix_{safe_text(str(args.run_suffix)).lower()}",
        )
        _write_status(
            status_path,
            stage="synthetic_mix",
            status="done",
            extra={
                "synthetic_rows": int(mix_summary["synthetic_rows"]),
                "mixed_rows": int(mix_summary["mixed_rows"]),
            },
        )

        _write_status(status_path, stage="probe", status="running")
        probe_cfg_path = generated_cfg_dir / f"taskform_winner_a1_pseudotarget_probe_{args.run_suffix}.yaml"
        probe_cfg = _materialize_cfg(
            base_cfg=base_cfg,
            output_path=probe_cfg_path,
            processed_dir=Path(str(mix_summary["processed_dir"])),
            run_dir=f"runs/TASKFORM_WINNER_A1_PSEUDOTARGET_PROBE_{safe_text(str(args.run_suffix)).upper()}",
        )
        probe = _run_candidate_probe(
            python_exec=python_exec,
            cfg_path=probe_cfg_path,
            cfg=probe_cfg,
            init_adapter_dir=Path(str(tapt_summary["best_model_dir"])),
            fold=int(args.fold),
            max_steps=int(args.probe_max_steps),
            eval_steps=int(args.probe_eval_steps),
            tag=f"taskform_winner_a1_pseudotarget_anchor64_{args.run_suffix}",
        )
        probe["label"] = "competition_only_pseudotarget_tapt_continue"
        probe["hard_subset"] = _probe_hard_metrics(probe)
        probe["delta_geom_vs_incumbent_anchor64"] = round(
            float(probe["eval_geom"]) - float(incumbent_anchor.get("eval_geom", 0.0) or 0.0),
            4,
        )
        probe["delta_geom_vs_frozen_anchor64"] = round(
            float(probe["eval_geom"]) - float(frozen_candidate.get("anchor64_reconstructed_geom", 0.0) or 0.0),
            4,
        )
        _write_status(
            status_path,
            stage="probe",
            status="done",
            extra={"probe_anchor64_geom": float(probe["eval_geom"])},
        )

        fullval: dict[str, Any] | None = None
        if float(probe["delta_geom_vs_incumbent_anchor64"]) >= 0.05:
            _write_status(status_path, stage="fullval", status="running")
            fullval = _maybe_run_fullval(
                python_exec=python_exec,
                cfg_path=probe_cfg_path,
                cfg=probe_cfg,
                checkpoint_dir=Path(str(probe["checkpoint_dir"])),
                fold=int(args.fold),
                tag=f"taskform_winner_a1_pseudotarget_fullval_{args.run_suffix}",
                out_dir=out_dir,
            )
            _write_status(
                status_path,
                stage="fullval",
                status="done",
                extra={"fullval_geom": float(fullval.get("eval_geom", 0.0) or 0.0)},
            )

        overall_status, reason, comparisons = _finalize_probe_status(
            incumbent_anchor=incumbent_anchor,
            frozen_candidate=frozen_candidate,
            probe=probe,
            fullval=fullval,
        )

        summary = {
            "line": "winner_competition_only_pseudotarget_smoke",
            "status": overall_status,
            "reason": reason,
            "base_config_path": str(base_cfg_path),
            "base_checkpoint_dir": str(base_checkpoint_dir),
            "base_processed_dir": str(base_processed_dir),
            "report_dir": str(out_dir),
            "incumbent_anchor64": incumbent_anchor,
            "frozen_candidate": frozen_candidate,
            "monolingual_inventory": monolingual_summary,
            "tapt_smoke": tapt_summary,
            "pseudo_targets_csv": str(pseudo_csv),
            "synthetic_mix": mix_summary,
            "probe": probe,
            "fullval": fullval,
            "comparisons": comparisons,
            "official_metric_bridge": {
                "status": "pending_parallel_tmux",
                "note": "separate tmux session runs official metric probe/report glue",
                "report_dir": str(REPO_ROOT / "reports" / "taskform_winner_bridge_20260311"),
            },
            "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        }
        write_json(out_dir / "summary.json", summary)

        table = [
            {
                "inc_anchor": f"{float(incumbent_anchor.get('eval_geom', 0.0) or 0.0):.4f}",
                "probe_anchor": f"{float(probe['eval_geom']):.4f}",
                "probe_hard": f"{float((probe['hard_subset'] or {}).get('eval_geom', 0.0) or 0.0):.4f}",
                "delta_vs_inc": f"{float(probe['delta_geom_vs_incumbent_anchor64']):+.4f}",
                "delta_vs_frozen": f"{float(probe['delta_geom_vs_frozen_anchor64']):+.4f}",
            }
        ]
        lines = [
            "# Winner Competition-only Pseudo-target Smoke",
            "",
            f"- status: `{overall_status}`",
            f"- reason: {reason}",
            f"- frozen fallback anchor/fullval/hard: `{float(frozen_candidate.get('anchor64_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('fullval_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen_candidate.get('hard_geom', 0.0) or 0.0):.4f}`",
            "",
            "## Probe",
            "",
            markdown_table(table, ["inc_anchor", "probe_anchor", "probe_hard", "delta_vs_inc", "delta_vs_frozen"]),
            "",
            f"- TAPT best model: `{tapt_summary['best_model_dir']}`",
            f"- pseudo rows: `{int(mix_summary['synthetic_rows'])}`",
            f"- mixed rows: `{int(mix_summary['mixed_rows'])}`",
        ]
        if fullval is not None:
            lines.extend(
                [
                    "",
                    "## Full-val",
                    "",
                    f"- local geom: `{float(fullval.get('eval_geom', 0.0) or 0.0):.4f}`",
                    f"- hard geom: `{float(((fullval.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0)):.4f}`",
                    f"- delta vs frozen fullval: `{float(comparisons.get('fullval_delta_geom_vs_frozen', 0.0) or 0.0):+.4f}`",
                    f"- delta vs frozen hard: `{float(comparisons.get('fullval_hard_delta_vs_frozen', 0.0) or 0.0):+.4f}`",
                ]
            )
        write_text(out_dir / "gate_report.md", "\n".join(lines) + "\n")

        _write_status(
            status_path,
            stage="completed",
            status=overall_status,
            extra={
                "summary_json": str(out_dir / "summary.json"),
                "gate_report_md": str(out_dir / "gate_report.md"),
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
