from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_tag(value: str) -> str:
    cleaned = "_".join(chunk for chunk in str(value).strip().replace("/", " ").replace("\\", " ").split() if chunk)
    return cleaned or "stage"


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    cursor = cfg
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = copy.deepcopy(value)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if "." in key:
            _set_nested(merged, key, value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _materialize_candidate_config(candidate: dict[str, Any], generated_dir: Path) -> tuple[Path, dict[str, Any]]:
    base_config_path = _resolve_path(
        str(candidate.get("base_config", "")),
        REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml",
    )
    base_cfg = _load_yaml(base_config_path)
    merged_cfg = _deep_merge(base_cfg, candidate.get("overrides") or {})
    merged_cfg["name"] = str(merged_cfg.get("name") or candidate.get("name") or base_cfg.get("name") or "steer_candidate")

    output_path = generated_dir / f"{_sanitize_tag(str(candidate.get('name', merged_cfg['name'])))}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(merged_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path, merged_cfg


def _run_prepare_commands(
    *,
    candidate: dict[str, Any],
    cfg_path: Path,
    python_executable: str,
    dry_run: bool,
) -> list[list[str]]:
    prepared: list[list[str]] = []
    for raw_cmd in candidate.get("prepare_commands") or []:
        if not isinstance(raw_cmd, list) or not raw_cmd:
            raise ValueError(f"prepare_commands entries must be non-empty lists: {raw_cmd!r}")
        cmd = [
            str(part).format(
                python_executable=python_executable,
                config_path=str(cfg_path),
                repo_root=str(REPO_ROOT),
                candidate_name=str(candidate.get("name", "candidate")),
            )
            for part in raw_cmd
        ]
        _run_command(cmd, dry_run=dry_run)
        prepared.append(cmd)
    return prepared


def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for item in run_dir.glob("checkpoint-*"):
        try:
            step = int(item.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, item))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def _candidate_run_dir(cfg: dict[str, Any], fold: int) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    return run_root.parent / f"{run_root.name}_fold{fold}"


def _stage_tag(candidate_name: str, stage_name: str) -> str:
    return _sanitize_tag(f"{candidate_name}_{stage_name}")


def _incumbent_tag(stage_name: str) -> str:
    return _sanitize_tag(f"steer_incumbent_{stage_name}")


def _decode_best_path(run_dir: Path, tag: str) -> Path:
    return run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"


def _diagnose_summary_path(run_dir: Path, tag: str) -> Path:
    return run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"


def _stage_summary_snapshot_path(run_dir: Path, tag: str) -> Path:
    return run_dir / "diagnostics" / f"run_summary_{tag}.json"


def _copy_stage_summary(run_dir: Path, tag: str, *, dry_run: bool) -> Path | None:
    source = run_dir / "run_summary.json"
    target = _stage_summary_snapshot_path(run_dir, tag)
    if dry_run:
        return target
    if not source.exists():
        raise FileNotFoundError(f"Missing run summary to snapshot: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _run_training_stage(
    *,
    python_executable: str,
    cfg_path: Path,
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    stage: dict[str, Any],
    fold: int,
    resume_from_checkpoint: Path | None,
    dry_run: bool,
) -> Path:
    run_dir = _candidate_run_dir(cfg, fold)
    train_cfg = stage.get("train", {}) or {}
    cmd = [
        python_executable,
        str(REPO_ROOT / "scripts" / "train_mt5_lora.py"),
        "--config",
        str(cfg_path),
        "--fold",
        str(fold),
        "--max-steps",
        str(int(train_cfg.get("max_steps", -1))),
        "--eval-steps",
        str(int(train_cfg.get("eval_steps", -1))),
        "--skip-final-predict",
    ]
    if int(train_cfg.get("max_train_rows", -1)) > 0:
        cmd.extend(["--max-train-rows", str(int(train_cfg.get("max_train_rows", -1)))])
    if int(train_cfg.get("max_val_rows", -1)) > 0:
        cmd.extend(["--max-val-rows", str(int(train_cfg.get("max_val_rows", -1)))])

    if resume_from_checkpoint is not None:
        cmd.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    else:
        init_adapter_dir = str(candidate.get("init_adapter_dir", "")).strip()
        if init_adapter_dir:
            cmd.extend(["--init-adapter-dir", str(_resolve_path(init_adapter_dir, run_dir / "missing_init_adapter_dir"))])

    _run_command(cmd, dry_run=dry_run)
    return run_dir


def _run_decode_stage(
    *,
    python_executable: str,
    cfg_path: Path,
    fold: int,
    checkpoint_dir: Path,
    stage: dict[str, Any],
    tag: str,
    dry_run: bool,
) -> Path:
    decode_cfg = stage.get("decode", {}) or {}
    cmd = [
        python_executable,
        str(REPO_ROOT / "scripts" / "eval_decode_grid.py"),
        "--config",
        str(cfg_path),
        "--fold",
        str(fold),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--tag",
        tag,
        "--beams",
        str(decode_cfg.get("beams", "4")),
        "--length-penalties",
        str(decode_cfg.get("length_penalties", "0.8,1.0,1.2")),
        "--no-repeat-ngram-sizes",
        str(decode_cfg.get("no_repeat_ngram_sizes", "0,2")),
        "--min-new-tokens-list",
        str(decode_cfg.get("min_new_tokens_list", "0")),
        "--max-new-tokens-list",
        str(decode_cfg.get("max_new_tokens_list", "384")),
        "--predict-batch-size",
        str(int(decode_cfg.get("predict_batch_size", 16))),
        "--aggregate-by-parent",
        str(decode_cfg.get("aggregate_by_parent", "auto")),
    ]
    max_val_samples = int(decode_cfg.get("max_val_samples", 0))
    if max_val_samples > 0:
        cmd.extend(["--max-val-samples", str(max_val_samples)])
    if bool(decode_cfg.get("aggregate_original_only", True)):
        cmd.append("--aggregate-original-only")
    else:
        cmd.append("--no-aggregate-original-only")

    _run_command(cmd, dry_run=dry_run)
    run_dir = _candidate_run_dir(_load_yaml(cfg_path), fold)
    return _decode_best_path(run_dir, tag)


def _run_diagnose_stage(
    *,
    python_executable: str,
    cfg_path: Path,
    fold: int,
    checkpoint_dir: Path,
    stage: dict[str, Any],
    tag: str,
    decode_best: dict[str, Any],
    dry_run: bool,
) -> Path:
    diagnose_cfg = stage.get("diagnose", {}) or {}
    cmd = [
        python_executable,
        str(REPO_ROOT / "scripts" / "diagnose_val_outputs.py"),
        "--config",
        str(cfg_path),
        "--fold",
        str(fold),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--tag",
        tag,
        "--predict-batch-size",
        str(int(diagnose_cfg.get("predict_batch_size", 16))),
        "--num-beams",
        str(int(decode_best.get("num_beams", 4))),
        "--length-penalty",
        str(float(decode_best.get("length_penalty", 1.0))),
        "--no-repeat-ngram-size",
        str(int(decode_best.get("no_repeat_ngram_size", 0))),
        "--min-new-tokens",
        str(int(decode_best.get("min_new_tokens", 0))),
        "--max-new-tokens",
        str(int(decode_best.get("max_new_tokens", 384))),
        "--aggregate-by-parent",
        str(diagnose_cfg.get("aggregate_by_parent", "auto")),
    ]
    max_rows = int(diagnose_cfg.get("max_rows", 0))
    if max_rows > 0:
        cmd.extend(["--max-rows", str(max_rows)])
    if bool(diagnose_cfg.get("aggregate_original_only", True)):
        cmd.append("--aggregate-original-only")
    else:
        cmd.append("--no-aggregate-original-only")

    _run_command(cmd, dry_run=dry_run)
    run_dir = _candidate_run_dir(_load_yaml(cfg_path), fold)
    return _diagnose_summary_path(run_dir, tag)


def _maybe_load_json(path: Path, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {}
    return _load_json(path)


def _metric_value(decode_best: dict[str, Any], key: str) -> float:
    value = decode_best.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _health_value(diag_summary: dict[str, Any], key: str) -> float:
    health = diag_summary.get("output_health", {}) or {}
    value = health.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _evaluate_stage(
    *,
    stage: dict[str, Any],
    baseline_decode: dict[str, Any],
    decode_best: dict[str, Any],
    diag_summary: dict[str, Any],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    constraints = stage.get("constraints", {}) or {}
    issues: list[str] = []
    warnings: list[str] = []

    metric_rules = [
        ("eval_geom", "max_geom_drop_abs", "min_geom_gain_abs"),
        ("eval_bleu", "max_bleu_drop_abs", "min_bleu_gain_abs"),
        ("eval_chrfpp", "max_chrfpp_drop_abs", "min_chrfpp_gain_abs"),
    ]
    deltas: dict[str, float] = {}
    for metric_key, drop_key, gain_key in metric_rules:
        baseline_value = _metric_value(baseline_decode, metric_key)
        candidate_value = _metric_value(decode_best, metric_key)
        if baseline_value == baseline_value and candidate_value == candidate_value:
            delta = float(candidate_value - baseline_value)
            deltas[metric_key] = delta
            if drop_key in constraints and delta < -float(constraints[drop_key]):
                issues.append(
                    f"{metric_key} dropped by {delta:.4f}, worse than -{float(constraints[drop_key]):.4f}"
                )
            if gain_key in constraints and delta < float(constraints[gain_key]):
                issues.append(
                    f"{metric_key} gain {delta:.4f} is below required {float(constraints[gain_key]):.4f}"
                )

    health_rules = [
        ("exact_extra_id_0_ratio_pct", "max_exact_extra_id_0_ratio_pct"),
        ("pred_shorter_than_half_ref_ratio_pct", "max_pred_shorter_than_half_ref_ratio_pct"),
        ("empty_prediction_ratio_pct", "max_empty_prediction_ratio_pct"),
        ("copy_source_ratio_pct", "max_copy_source_ratio_pct"),
    ]
    for health_key, constraint_key in health_rules:
        if constraint_key not in constraints:
            continue
        value = _health_value(diag_summary, health_key)
        if value == value and value > float(constraints[constraint_key]):
            issues.append(
                f"{health_key}={value:.4f} exceeds {float(constraints[constraint_key]):.4f}"
            )

    gpu_util = float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0)
    if "min_gpu_peak_utilization_pct" in constraints and gpu_util < float(constraints["min_gpu_peak_utilization_pct"]):
        issues.append(
            f"gpu_peak_utilization_pct={gpu_util:.2f} is below floor {float(constraints['min_gpu_peak_utilization_pct']):.2f}"
        )
    if "max_gpu_peak_utilization_pct" in constraints and gpu_util > float(constraints["max_gpu_peak_utilization_pct"]):
        issues.append(
            f"gpu_peak_utilization_pct={gpu_util:.2f} exceeds ceiling {float(constraints['max_gpu_peak_utilization_pct']):.2f}"
        )
    if "target_gpu_peak_utilization_pct" in constraints and gpu_util < float(constraints["target_gpu_peak_utilization_pct"]):
        warnings.append(
            f"gpu_peak_utilization_pct={gpu_util:.2f} is below target {float(constraints['target_gpu_peak_utilization_pct']):.2f}"
        )

    accepted = not issues
    return {
        "accepted": bool(accepted),
        "issues": issues,
        "warnings": warnings,
        "deltas_vs_incumbent": deltas,
        "gpu_peak_utilization_pct": gpu_util,
    }


def _ensure_incumbent_baseline(
    *,
    plan: dict[str, Any],
    stage: dict[str, Any],
    python_executable: str,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    incumbent = plan.get("incumbent", {}) or {}
    cfg_path = _resolve_path(str(incumbent.get("config", "")), REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    fold = int(incumbent.get("fold", 0))
    run_dir = _candidate_run_dir(cfg, fold)
    checkpoint_dir = _resolve_path(str(incumbent.get("checkpoint_dir", "")), run_dir / "best_model")
    tag = _incumbent_tag(str(stage.get("name", "stage")))
    decode_path = _decode_best_path(run_dir, tag)
    diagnose_path = _diagnose_summary_path(run_dir, tag)

    if not skip_existing or not decode_path.exists():
        _run_decode_stage(
            python_executable=python_executable,
            cfg_path=cfg_path,
            fold=fold,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            tag=tag,
            dry_run=dry_run,
        )
    decode_best = _maybe_load_json(decode_path, dry_run=dry_run)

    if not skip_existing or not diagnose_path.exists():
        _run_diagnose_stage(
            python_executable=python_executable,
            cfg_path=cfg_path,
            fold=fold,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            tag=tag,
            decode_best=decode_best,
            dry_run=dry_run,
        )
    diagnose_summary = _maybe_load_json(diagnose_path, dry_run=dry_run)

    return {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decode_best": decode_best,
        "diagnose_summary": diagnose_summary,
        "decode_best_path": str(decode_path),
        "diagnose_summary_path": str(diagnose_path),
    }


def _select_candidates(plan: dict[str, Any], requested: str) -> list[dict[str, Any]]:
    candidates = plan.get("candidates") or []
    if not requested.strip():
        return list(candidates)
    selected_names = {chunk.strip() for chunk in requested.split(",") if chunk.strip()}
    selected = [candidate for candidate in candidates if str(candidate.get("name", "")) in selected_names]
    missing = sorted(selected_names - {str(candidate.get("name", "")) for candidate in selected})
    if missing:
        raise ValueError(f"Unknown candidates requested: {missing}")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/cloud_stage2_steer.yaml")
    ap.add_argument("--candidates", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    plan_path = _resolve_path(args.plan, REPO_ROOT / "configs" / "cloud_stage2_steer.yaml")
    plan = _load_yaml(plan_path)
    python_executable = str(plan.get("python_executable") or sys.executable)
    fold = int(plan.get("fold", (plan.get("incumbent", {}) or {}).get("fold", 0)))
    generated_dir = _resolve_path(
        str(plan.get("generated_config_dir", "runs/STEER/generated_configs")),
        REPO_ROOT / "runs" / "STEER" / "generated_configs",
    )
    summary_path = _resolve_path(
        str(plan.get("summary_path", "runs/STEER/steer_summary.json")),
        REPO_ROOT / "runs" / "STEER" / "steer_summary.json",
    )

    stages = plan.get("stages") or []
    if not stages:
        raise ValueError("Plan must define at least one stage")

    candidates = _select_candidates(plan, args.candidates)
    overall_records: list[dict[str, Any]] = []
    baseline_cache: dict[str, dict[str, Any]] = {}

    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        baseline_cache[stage_name] = _ensure_incumbent_baseline(
            plan=plan,
            stage=stage,
            python_executable=python_executable,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )

    for candidate in candidates:
        candidate_name = str(candidate.get("name", "candidate"))
        cfg_path, cfg = _materialize_candidate_config(candidate, generated_dir)
        run_dir = _candidate_run_dir(cfg, fold)
        run_dir.mkdir(parents=True, exist_ok=True)

        prepare_commands = _run_prepare_commands(
            candidate=candidate,
            cfg_path=cfg_path,
            python_executable=python_executable,
            dry_run=args.dry_run,
        )

        resume_checkpoint: Path | None = None
        stage_records: list[dict[str, Any]] = []
        stopped = False
        for idx, stage in enumerate(stages):
            stage_name = str(stage.get("name", f"stage{idx + 1}"))
            tag = _stage_tag(candidate_name, stage_name)
            decode_path = _decode_best_path(run_dir, tag)
            diagnose_path = _diagnose_summary_path(run_dir, tag)
            stage_summary_snapshot = _stage_summary_snapshot_path(run_dir, tag)

            if args.skip_existing and decode_path.exists() and diagnose_path.exists() and stage_summary_snapshot.exists():
                print(f"SKIP: reuse candidate={candidate_name} stage={stage_name}")
                decode_best = _load_json(decode_path) if not args.dry_run else {}
                diagnose_summary = _load_json(diagnose_path) if not args.dry_run else {}
                run_summary = _load_json(stage_summary_snapshot) if not args.dry_run else {}
            else:
                if idx > 0:
                    resume_checkpoint = _find_latest_checkpoint(run_dir)
                    if resume_checkpoint is None:
                        raise FileNotFoundError(
                            f"Missing checkpoint to resume candidate={candidate_name} stage={stage_name} in {run_dir}"
                        )
                _run_training_stage(
                    python_executable=python_executable,
                    cfg_path=cfg_path,
                    cfg=cfg,
                    candidate=candidate,
                    stage=stage,
                    fold=fold,
                    resume_from_checkpoint=resume_checkpoint,
                    dry_run=args.dry_run,
                )
                checkpoint_dir = run_dir / "best_model"
                _run_decode_stage(
                    python_executable=python_executable,
                    cfg_path=cfg_path,
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                    stage=stage,
                    tag=tag,
                    dry_run=args.dry_run,
                )
                decode_best = _maybe_load_json(decode_path, dry_run=args.dry_run)
                _run_diagnose_stage(
                    python_executable=python_executable,
                    cfg_path=cfg_path,
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                    stage=stage,
                    tag=tag,
                    decode_best=decode_best,
                    dry_run=args.dry_run,
                )
                diagnose_summary = _maybe_load_json(diagnose_path, dry_run=args.dry_run)
                _copy_stage_summary(run_dir, tag, dry_run=args.dry_run)
                run_summary = _maybe_load_json(stage_summary_snapshot, dry_run=args.dry_run)

            baseline = baseline_cache[stage_name]
            evaluation = _evaluate_stage(
                stage=stage,
                baseline_decode=baseline.get("decode_best", {}) or {},
                decode_best=decode_best,
                diag_summary=diagnose_summary,
                run_summary=run_summary,
            )
            stage_record = {
                "stage": stage_name,
                "tag": tag,
                "decode_best_path": str(decode_path),
                "diagnose_summary_path": str(diagnose_path),
                "run_summary_snapshot_path": str(stage_summary_snapshot),
                "decode_best": decode_best,
                "diagnose_summary": diagnose_summary,
                "run_summary": run_summary,
                "baseline": baseline,
                "evaluation": evaluation,
            }
            stage_records.append(stage_record)

            if not evaluation.get("accepted", False):
                stopped = True
                break

        final_stage = stage_records[-1] if stage_records else {}
        final_eval = final_stage.get("evaluation", {}) if isinstance(final_stage, dict) else {}
        final_decode = final_stage.get("decode_best", {}) if isinstance(final_stage, dict) else {}
        overall_records.append(
            {
                "candidate": candidate_name,
                "config_path": str(cfg_path),
                "run_dir": str(run_dir),
                "prepare_commands": prepare_commands,
                "stopped_early": bool(stopped),
                "accepted": bool(final_eval.get("accepted", False)) if stage_records else False,
                "final_stage": final_stage.get("stage", "") if isinstance(final_stage, dict) else "",
                "final_eval_geom": float(final_decode.get("eval_geom", 0.0) or 0.0),
                "final_eval_bleu": float(final_decode.get("eval_bleu", 0.0) or 0.0),
                "final_eval_chrfpp": float(final_decode.get("eval_chrfpp", 0.0) or 0.0),
                "final_gpu_peak_utilization_pct": float(final_eval.get("gpu_peak_utilization_pct", 0.0) or 0.0),
                "stage_records": stage_records,
            }
        )
        _write_json(
            summary_path,
            {
                "plan_path": str(plan_path),
                "python_executable": python_executable,
                "generated_config_dir": str(generated_dir),
                "records": overall_records,
                "updated_at_unix": time.time(),
            },
        )

    overall_records.sort(
        key=lambda item: (
            0 if item.get("accepted") else 1,
            -float(item.get("final_eval_geom", 0.0)),
            -float(item.get("final_eval_bleu", 0.0)),
        )
    )
    payload = {
        "plan_path": str(plan_path),
        "python_executable": python_executable,
        "generated_config_dir": str(generated_dir),
        "records": overall_records,
        "updated_at_unix": time.time(),
    }
    _write_json(summary_path, payload)
    print(f"OK: wrote {summary_path}")
    if overall_records:
        leader = overall_records[0]
        print(
            "INFO: leader="
            f"{leader['candidate']} geom={float(leader.get('final_eval_geom', 0.0)):.4f} "
            f"bleu={float(leader.get('final_eval_bleu', 0.0)):.4f} "
            f"chrfpp={float(leader.get('final_eval_chrfpp', 0.0)):.4f} "
            f"accepted={bool(leader.get('accepted', False))}"
        )


if __name__ == "__main__":
    main()
