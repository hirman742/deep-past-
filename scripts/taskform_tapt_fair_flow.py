#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metrics_utils import compute_translation_metrics
from taskform_phase12_common import normalize_whitespace, safe_text, tokenize_words, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=REPO_ROOT)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _candidate_run_dir(cfg: dict[str, Any], fold: int) -> Path:
    run_root = _resolve_path((cfg.get("paths", {}) or {}).get("run_dir"), REPO_ROOT / "runs" / "MISSING")
    return run_root.parent / f"{run_root.name}_fold{fold}"


def _word_len(text: str) -> int:
    return len(tokenize_words(safe_text(text)))


def _materialize_cfg(
    *,
    base_cfg: dict[str, Any],
    output_path: Path,
    processed_dir: Path,
    run_dir: str,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("paths", {})["processed_dir"] = str(processed_dir.relative_to(REPO_ROOT))
    cfg.setdefault("paths", {})["run_dir"] = run_dir
    cfg = _deep_merge(cfg, extra_overrides or {})
    _write_yaml(output_path, cfg)
    return cfg


def _prepare_asset_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    source_kind: str,
    genre_label: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["transliteration"] = out["transliteration"].fillna("").astype(str).map(normalize_whitespace)
    out = out.loc[out["transliteration"].astype(str).str.strip() != ""].copy()
    out = out.drop_duplicates(subset=["transliteration"], keep="first").reset_index(drop=True)
    out["asset_id"] = out[id_col].fillna("").astype(str).map(lambda value: f"{source_kind}::{value}")
    out["oare_id"] = out[id_col].fillna("").astype(str)
    out["source_kind"] = source_kind
    out["genre_label"] = genre_label
    out["source_word_len"] = out["transliteration"].map(_word_len)
    return out[["asset_id", "oare_id", "source_kind", "genre_label", "transliteration", "source_word_len"]]


def _concat_dedup_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            text = safe_text(row.get("transliteration"))
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["asset_id", "oare_id", "source_kind", "genre_label", "transliteration", "source_word_len"])
    out["source_word_len"] = out["transliteration"].map(_word_len)
    out = out.sort_values(["source_word_len", "asset_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def _build_fair_mono_assets(
    *,
    base_processed_dir: Path,
    fold: int,
    out_dir: Path,
) -> dict[str, Any]:
    train_proc = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(base_processed_dir / "folds.csv")
    raw_train = pd.read_csv(REPO_ROOT / "data" / "raw" / "train.csv")
    raw_test = pd.read_csv(REPO_ROOT / "data" / "raw" / "test.csv")
    published = pd.read_csv(REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv")

    merged = train_proc.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    parent_folds = merged[["parent_oare_id", "fold"]].drop_duplicates().reset_index(drop=True)
    dup_parent = parent_folds.duplicated(subset=["parent_oare_id"], keep=False)
    if dup_parent.any():
        duplicates = parent_folds.loc[dup_parent, "parent_oare_id"].astype(str).tolist()[:10]
        raise ValueError(f"Parent fold mapping is inconsistent for {duplicates}")

    raw_train = raw_train.merge(parent_folds, left_on="oare_id", right_on="parent_oare_id", how="left")
    if raw_train["fold"].isna().any():
        missing = raw_train.loc[raw_train["fold"].isna(), "oare_id"].astype(str).tolist()[:10]
        raise ValueError(f"Missing fold assignment for raw train rows: {missing}")

    raw_train["transliteration"] = raw_train["transliteration"].fillna("").astype(str).map(normalize_whitespace)
    raw_test["transliteration"] = raw_test["transliteration"].fillna("").astype(str).map(normalize_whitespace)
    published["transliteration"] = published["transliteration"].fillna("").astype(str).map(normalize_whitespace)

    val_train = raw_train.loc[raw_train["fold"].astype(int) == int(fold)].copy().reset_index(drop=True)
    trainfold_train = raw_train.loc[raw_train["fold"].astype(int) != int(fold)].copy().reset_index(drop=True)

    val_sources = set(val_train["transliteration"].astype(str).tolist())
    test_sources = set(raw_test["transliteration"].astype(str).tolist())

    trainfold_assets = _prepare_asset_frame(
        trainfold_train.sort_values(["oare_id"]).reset_index(drop=True),
        id_col="oare_id",
        source_kind="trainfold_source_only",
        genre_label="trainfold_parallel",
    )
    trainfold_overlap_mask = trainfold_assets["transliteration"].astype(str).isin(val_sources)
    removed_rows: list[dict[str, Any]] = []
    if trainfold_overlap_mask.any():
        for row in trainfold_assets.loc[trainfold_overlap_mask].to_dict(orient="records"):
            removed_rows.append(
                {
                    "asset_id": row["asset_id"],
                    "oare_id": row["oare_id"],
                    "source_kind": row["source_kind"],
                    "transliteration": row["transliteration"],
                    "remove_reason": "val_overlap",
                }
            )
    trainfold_assets = trainfold_assets.loc[~trainfold_overlap_mask].copy().reset_index(drop=True)
    trainfold_sources = set(trainfold_assets["transliteration"].astype(str).tolist())

    kept_published_rows: list[dict[str, Any]] = []
    seen_published_texts: set[str] = set()
    published_nonempty = published.loc[published["transliteration"].astype(str).str.strip() != ""].copy()
    published_nonempty = published_nonempty.sort_values(["oare_id"]).reset_index(drop=True)
    for row in published_nonempty.to_dict(orient="records"):
        text = safe_text(row.get("transliteration"))
        remove_reason = ""
        if not text:
            continue
        if text in val_sources:
            remove_reason = "val_overlap"
        elif text in test_sources:
            remove_reason = "test_source"
        elif text in trainfold_sources or text in seen_published_texts:
            remove_reason = "published_duplicate"

        if remove_reason:
            removed_rows.append(
                {
                    "asset_id": f"published_source_only::{safe_text(row.get('oare_id'))}",
                    "oare_id": safe_text(row.get("oare_id")),
                    "source_kind": "published_source_only",
                    "transliteration": text,
                    "remove_reason": remove_reason,
                }
            )
            continue

        seen_published_texts.add(text)
        kept_published_rows.append(
            {
                "asset_id": f"published_source_only::{safe_text(row.get('oare_id'))}",
                "oare_id": safe_text(row.get("oare_id")),
                "source_kind": "published_source_only",
                "genre_label": safe_text(row.get("genre_label")) or "published",
                "transliteration": text,
                "source_word_len": _word_len(text),
            }
        )

    published_nooverlap = pd.DataFrame(kept_published_rows)
    if published_nooverlap.empty:
        published_nooverlap = pd.DataFrame(
            columns=["asset_id", "oare_id", "source_kind", "genre_label", "transliteration", "source_word_len"]
        )
    else:
        published_nooverlap = published_nooverlap.sort_values(["source_word_len", "asset_id"], ascending=[False, True]).reset_index(drop=True)

    test_assets = _prepare_asset_frame(
        raw_test.sort_values(["id"]).reset_index(drop=True),
        id_col="id",
        source_kind="test_source_only",
        genre_label="test",
    )
    for row in test_assets.to_dict(orient="records"):
        removed_rows.append(
            {
                "asset_id": row["asset_id"],
                "oare_id": row["oare_id"],
                "source_kind": row["source_kind"],
                "transliteration": row["transliteration"],
                "remove_reason": "test_source",
            }
        )

    fair_offline = _concat_dedup_frames([trainfold_assets, published_nooverlap])

    published_submit_unique = _prepare_asset_frame(
        published.loc[published["transliteration"].astype(str).str.strip() != ""].sort_values(["oare_id"]).reset_index(drop=True),
        id_col="oare_id",
        source_kind="published_source_only",
        genre_label="published",
    )
    all_train_assets = _prepare_asset_frame(
        raw_train.sort_values(["oare_id"]).reset_index(drop=True),
        id_col="oare_id",
        source_kind="train_all_source",
        genre_label="train_parallel",
    )
    submit_mono_all = _concat_dedup_frames([all_train_assets, test_assets, published_submit_unique])

    removed_df = pd.DataFrame(removed_rows)
    if removed_df.empty:
        removed_df = pd.DataFrame(columns=["asset_id", "oare_id", "source_kind", "transliteration", "remove_reason"])

    trainfold_csv = out_dir / "fair_offline_mono_fold0_trainfold_source_only.csv"
    published_csv = out_dir / "fair_offline_mono_fold0_published_nooverlap.csv"
    fair_csv = out_dir / "fair_offline_mono_fold0.csv"
    submit_csv = out_dir / "submit_mono_all.csv"
    removed_csv = out_dir / "fair_offline_removed_rows.csv"

    trainfold_assets.to_csv(trainfold_csv, index=False)
    published_nooverlap.to_csv(published_csv, index=False)
    fair_offline.to_csv(fair_csv, index=False)
    submit_mono_all.to_csv(submit_csv, index=False)
    removed_df.to_csv(removed_csv, index=False)

    removal_counts = removed_df["remove_reason"].value_counts().to_dict()
    base_val_parent_rows = int(merged.loc[merged["fold"] == int(fold), "parent_oare_id"].nunique())
    base_val_chunk_rows = int((merged["fold"] == int(fold)).sum())
    manifest = {
        "fold": int(fold),
        "base_processed_dir": str(base_processed_dir),
        "fair_offline_mono_fold0_csv": str(fair_csv),
        "trainfold_source_only_csv": str(trainfold_csv),
        "trainfold_plus_published_nooverlap_csv": str(fair_csv),
        "published_nooverlap_csv": str(published_csv),
        "submit_mono_all_csv": str(submit_csv),
        "removed_rows_csv": str(removed_csv),
        "fold0_val_rows": int(len(val_train)),
        "fold0_val_parent_rows": int(len(val_train)),
        "fold0_val_unique_sources": int(len(val_sources)),
        "fold0_val_parent_rows_from_processed": int(base_val_parent_rows),
        "fold0_val_chunk_rows_from_processed": int(base_val_chunk_rows),
        "inventory": {
            "trainfold_source_only_rows": int(len(trainfold_assets)),
            "published_nooverlap_rows": int(len(published_nooverlap)),
            "fair_offline_mono_fold0_rows": int(len(fair_offline)),
            "submit_mono_all_rows": int(len(submit_mono_all)),
        },
        "removals": {
            "val_overlap_rows": int(removal_counts.get("val_overlap", 0)),
            "test_rows": int(removal_counts.get("test_source", 0)),
            "published_duplicate_rows": int(removal_counts.get("published_duplicate", 0)),
        },
        "checks": {
            "fold0_val_row_count_matches_base": int(len(val_train)) == int(base_val_parent_rows),
            "fold0_val_chunk_rows_unchanged": int(base_val_chunk_rows) == int((merged["fold"] == int(fold)).sum()),
            "trainfold_exact_overlap_with_val_unique_sources": int(len(set(trainfold_assets["transliteration"].astype(str).tolist()) & val_sources)),
            "fair_offline_exact_overlap_with_val_unique_sources": int(len(set(fair_offline["transliteration"].astype(str).tolist()) & val_sources)),
            "fair_offline_exact_overlap_with_test_unique_sources": int(len(set(fair_offline["transliteration"].astype(str).tolist()) & test_sources)),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _ensure_incumbent_anchor(
    *,
    python_exec: str,
    base_cfg_path: Path,
    checkpoint_dir: Path,
    fold: int,
    out_dir: Path,
) -> dict[str, Any]:
    legacy_summary_path = REPO_ROOT / "reports" / "taskform_a2_a1_20260310" / "incumbent_anchor64_summary.json"
    if legacy_summary_path.exists():
        legacy = _load_json(legacy_summary_path)
        summary = {
            "label": "I0_incumbent",
            "tag": safe_text(legacy.get("tag")) or "taskform_tapt_fair_i0_anchor64_20260310",
            "decode_best_path": str(legacy.get("decode_best_path", "")),
            "diagnose_summary_path": str(legacy.get("diagnose_summary_path", "")),
            "eval_geom": float(legacy.get("eval_geom", 0.0)),
            "eval_bleu": float(legacy.get("eval_bleu", 0.0)),
            "eval_chrfpp": float(legacy.get("eval_chrfpp", 0.0)),
            "output_health": (legacy.get("output_health", {}) or {}),
            "reuse_source": str(legacy_summary_path),
        }
        write_json(out_dir / "incumbent_anchor64_summary.json", summary)
        return summary

    tag = "taskform_tapt_fair_i0_anchor64_20260310"
    base_cfg = _load_yaml(base_cfg_path)
    run_dir = _candidate_run_dir(base_cfg, fold)
    decode_best = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    diagnose_summary = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"
    if not decode_best.exists():
        _run(
            [
                python_exec,
                str(SCRIPTS_DIR / "eval_decode_grid.py"),
                "--config",
                str(base_cfg_path),
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
    decode_payload = _load_json(decode_best)
    if not diagnose_summary.exists():
        _run(
            [
                python_exec,
                str(SCRIPTS_DIR / "diagnose_val_outputs.py"),
                "--config",
                str(base_cfg_path),
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
    diag_payload = _load_json(diagnose_summary)
    summary = {
        "label": "I0_incumbent",
        "tag": tag,
        "decode_best_path": str(decode_best),
        "diagnose_summary_path": str(diagnose_summary),
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "output_health": (diag_payload.get("output_health", {}) or {}),
    }
    write_json(out_dir / "incumbent_anchor64_summary.json", summary)
    return summary


def _run_tapt_smoke(
    *,
    python_exec: str,
    base_cfg: dict[str, Any],
    out_dir: Path,
    mono_csv: Path,
    label: str,
    max_steps: int,
    eval_steps: int,
) -> dict[str, Any]:
    tapt_cfg_path = out_dir / "generated_configs" / f"taskform_tapt_fair_smoke_{label}.yaml"
    run_root = REPO_ROOT / "runs" / f"TASKFORM_TAPT_FAIR_{label.upper()}_20260310"
    overrides = {
        "name": f"taskform_tapt_fair_smoke_{label}",
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
            "max_steps": int(max_steps),
            "eval_steps": int(eval_steps),
            "fp16": False,
            "bf16": True,
        },
    }
    _write_yaml(tapt_cfg_path, _deep_merge(base_cfg, overrides))
    _run(
        [
            python_exec,
            str(SCRIPTS_DIR / "tapt_denoise.py"),
            "--config",
            str(tapt_cfg_path),
            "--corpus-csvs",
            str(mono_csv),
            "--output-run-dir",
            str(run_root),
        ]
    )
    summary_path = run_root / "tapt_summary.json"
    payload = _load_json(summary_path)
    record = {
        "label": label,
        "corpus_csv": str(mono_csv),
        "config_path": str(tapt_cfg_path),
        "run_dir": str(run_root),
        "summary_path": str(summary_path),
        "best_model_dir": str(payload.get("best_model_dir", "")),
        "raw_text_rows": int(payload.get("raw_text_rows", 0)),
        "train_rows": int(payload.get("train_rows", 0)),
        "eval_rows": int(payload.get("eval_rows", 0)),
        "train_loss": float((payload.get("train_metrics", {}) or {}).get("train_loss", math.nan) or math.nan),
        "eval_loss": float((payload.get("eval_metrics", {}) or {}).get("eval_loss", math.nan) or math.nan),
    }
    return record


def _select_best_tapt(records: list[dict[str, Any]]) -> dict[str, Any]:
    healthy = [
        row
        for row in records
        if row.get("best_model_dir") and not math.isnan(float(row.get("eval_loss", math.nan) or math.nan))
    ]
    if not healthy:
        raise ValueError("No healthy TAPT smoke run completed")
    best = min(healthy, key=lambda row: (float(row["eval_loss"]), -int(row.get("raw_text_rows", 0)), str(row.get("label"))))
    selected = copy.deepcopy(best)
    selected["selection_reason"] = "lowest_eval_loss_as_smoke_health_tiebreaker_only"
    return selected


def _run_supervised_probe(
    *,
    python_exec: str,
    cfg_path: Path,
    cfg: dict[str, Any],
    fold: int,
    max_steps: int,
    eval_steps: int,
    tag: str,
    init_adapter_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = _candidate_run_dir(cfg, fold)
    train_cmd = [
        python_exec,
        str(SCRIPTS_DIR / "train_mt5_lora.py"),
        "--config",
        str(cfg_path),
        "--fold",
        str(fold),
        "--max-steps",
        str(int(max_steps)),
        "--eval-steps",
        str(int(eval_steps)),
        "--skip-final-predict",
    ]
    if init_adapter_dir is not None:
        train_cmd.extend(["--init-adapter-dir", str(init_adapter_dir)])
    _run(train_cmd)

    checkpoint_dir = run_dir / "best_model"
    _run(
        [
            python_exec,
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
    decode_best_path = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    decode_payload = _load_json(decode_best_path)

    _run(
        [
            python_exec,
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
    diagnose_summary_path = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"
    diag_payload = _load_json(diagnose_summary_path)
    run_summary_path = run_dir / "run_summary.json"
    run_summary = _load_json(run_summary_path)
    return {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "run_summary_path": str(run_summary_path),
        "used_init_adapter": bool(init_adapter_dir is not None),
        "init_adapter_dir": str(init_adapter_dir) if init_adapter_dir is not None else "",
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "gpu_peak_utilization_pct": float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0),
    }


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
    subset = pred_df.loc[pred_df["oare_id"].fillna("").astype(str).isin(hard_ids)].copy().reset_index(drop=True)
    metrics = compute_translation_metrics(
        predictions=subset["prediction"].fillna("").astype(str).tolist(),
        references=subset["reference"].fillna("").astype(str).tolist(),
    )
    return {
        "rows": int(len(subset)),
        "eval_geom": float(metrics["geom"]),
        "eval_bleu": float(metrics["bleu"]),
        "eval_chrfpp": float(metrics["chrfpp"]),
        "note": "hard subset = ids intersect routed_full",
    }


def _maybe_run_fullval(
    *,
    python_exec: str,
    cfg_path: Path,
    cfg: dict[str, Any],
    checkpoint_dir: Path,
    fold: int,
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    _run(
        [
            python_exec,
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
            "--aggregate-by-parent",
            "auto",
            "--aggregate-original-only",
        ]
    )
    decode_best_path = _candidate_run_dir(cfg, fold) / "diagnostics" / f"decode_grid_best_{tag}.json"
    decode_payload = _load_json(decode_best_path)
    _run(
        [
            python_exec,
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
        ]
    )
    reconstructed_csv = _candidate_run_dir(cfg, fold) / "diagnostics" / f"val_predictions_reconstructed_{tag}.csv"
    hard_metrics = _compute_hard_subset_metrics(
        reconstructed_csv,
        REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv",
    )
    fullval = {
        "decode_best_path": str(decode_best_path),
        "reconstructed_csv": str(reconstructed_csv),
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "hard_subset": hard_metrics,
        "official_like_note": "same as local until official bridge lands",
    }
    write_json(out_dir / "fullval_compare_summary.json", fullval)
    return fullval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--base-checkpoint-dir", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250")
    ap.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/taskform_tapt_fair_20260310")
    ap.add_argument("--tapt-max-steps", type=int, default=240)
    ap.add_argument("--tapt-eval-steps", type=int, default=80)
    ap.add_argument("--supervised-max-steps", type=int, default=220)
    ap.add_argument("--supervised-eval-steps", type=int, default=110)
    ap.add_argument("--run-fullval-if-t0-beats-i0", action="store_true")
    args = ap.parse_args()

    started = time.time()
    python_exec = sys.executable

    base_cfg_path = _resolve_path(args.base_config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml")
    base_cfg = _load_yaml(base_cfg_path)
    base_checkpoint_dir = _resolve_path(args.base_checkpoint_dir, REPO_ROOT / "runs" / "missing_checkpoint")
    base_processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_tapt_fair_20260310")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = out_dir / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_fair_mono_assets(
        base_processed_dir=base_processed_dir,
        fold=int(args.fold),
        out_dir=out_dir,
    )
    incumbent_anchor = _ensure_incumbent_anchor(
        python_exec=python_exec,
        base_cfg_path=base_cfg_path,
        checkpoint_dir=base_checkpoint_dir,
        fold=int(args.fold),
        out_dir=out_dir,
    )

    tapt_records = [
        _run_tapt_smoke(
            python_exec=python_exec,
            base_cfg=base_cfg,
            out_dir=out_dir,
            mono_csv=Path(manifest["trainfold_source_only_csv"]),
            label="trainfold_source_only",
            max_steps=int(args.tapt_max_steps),
            eval_steps=int(args.tapt_eval_steps),
        ),
        _run_tapt_smoke(
            python_exec=python_exec,
            base_cfg=base_cfg,
            out_dir=out_dir,
            mono_csv=Path(manifest["trainfold_plus_published_nooverlap_csv"]),
            label="trainfold_plus_published_nooverlap",
            max_steps=int(args.tapt_max_steps),
            eval_steps=int(args.tapt_eval_steps),
        ),
    ]
    best_tapt = _select_best_tapt(tapt_records)

    c0_cfg_path = generated_cfg_dir / "taskform_tapt_fair_c0.yaml"
    c0_cfg = _materialize_cfg(
        base_cfg=base_cfg,
        output_path=c0_cfg_path,
        processed_dir=base_processed_dir,
        run_dir="runs/TASKFORM_TAPT_FAIR_C0_20260310",
    )
    c0_probe = _run_supervised_probe(
        python_exec=python_exec,
        cfg_path=c0_cfg_path,
        cfg=c0_cfg,
        fold=int(args.fold),
        max_steps=int(args.supervised_max_steps),
        eval_steps=int(args.supervised_eval_steps),
        tag="taskform_tapt_fair_c0_anchor64_20260310",
        init_adapter_dir=None,
    )
    c0_probe["label"] = "C0_no_tapt"

    t0_cfg_path = generated_cfg_dir / "taskform_tapt_fair_t0.yaml"
    t0_cfg = _materialize_cfg(
        base_cfg=base_cfg,
        output_path=t0_cfg_path,
        processed_dir=base_processed_dir,
        run_dir="runs/TASKFORM_TAPT_FAIR_T0_20260310",
    )
    t0_probe = _run_supervised_probe(
        python_exec=python_exec,
        cfg_path=t0_cfg_path,
        cfg=t0_cfg,
        fold=int(args.fold),
        max_steps=int(args.supervised_max_steps),
        eval_steps=int(args.supervised_eval_steps),
        tag="taskform_tapt_fair_t0_anchor64_20260310",
        init_adapter_dir=Path(best_tapt["best_model_dir"]),
    )
    t0_probe["label"] = "T0_tapt_then_supervised"
    t0_probe["tapt_init_label"] = best_tapt["label"]

    compare = {
        "delta_geom_t0_vs_c0": round(float(t0_probe["eval_geom"]) - float(c0_probe["eval_geom"]), 4),
        "delta_geom_t0_vs_i0": round(float(t0_probe["eval_geom"]) - float(incumbent_anchor["eval_geom"]), 4),
        "health_t0_vs_c0": _compare_output_health(t0_probe, c0_probe),
        "health_t0_vs_i0": _compare_output_health(t0_probe, incumbent_anchor),
    }
    if compare["delta_geom_t0_vs_c0"] >= 0.30 and compare["health_t0_vs_c0"]["no_regression"]:
        compare["status"] = "tapt_positive_offline"
        if compare["delta_geom_t0_vs_i0"] <= -0.50:
            compare["status"] = "review_stop"
        elif compare["delta_geom_t0_vs_i0"] >= 0.0:
            compare["status"] = "eligible_fullval_compare"
        else:
            compare["status"] = "eligible_wlite"
    else:
        compare["status"] = "review_stop"

    fullval_summary: dict[str, Any] | None = None
    if args.run_fullval_if_t0_beats_i0 and float(compare["delta_geom_t0_vs_i0"]) >= 0.0:
        fullval_summary = _maybe_run_fullval(
            python_exec=python_exec,
            cfg_path=t0_cfg_path,
            cfg=t0_cfg,
            checkpoint_dir=Path(t0_probe["checkpoint_dir"]),
            fold=int(args.fold),
            tag="taskform_tapt_fair_t0_fullval_20260310",
            out_dir=out_dir,
        )

    summary = {
        "base_config_path": str(base_cfg_path),
        "base_checkpoint_dir": str(base_checkpoint_dir),
        "base_processed_dir": str(base_processed_dir),
        "manifest_path": str(out_dir / "manifest.json"),
        "fair_offline_manifest": manifest,
        "incumbent_anchor64": incumbent_anchor,
        "tapt_smoke": {
            "records": tapt_records,
            "selected": best_tapt,
        },
        "matched_supervised": {
            "c0": c0_probe,
            "t0": t0_probe,
        },
        "comparison": compare,
        "fullval_compare": fullval_summary,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    write_json(out_dir / "summary.json", summary)

    print(
        json.dumps(
            {
                "status": compare["status"],
                "delta_geom_t0_vs_c0": compare["delta_geom_t0_vs_c0"],
                "delta_geom_t0_vs_i0": compare["delta_geom_t0_vs_i0"],
                "selected_tapt": best_tapt["label"],
                "elapsed_minutes": summary["elapsed_minutes"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
