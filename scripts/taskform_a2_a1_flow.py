#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from metrics_utils import compute_translation_metrics
from taskform_phase12_common import (
    internal_repeat_score,
    markdown_table,
    normalize_whitespace,
    safe_text,
    tokenize_words,
    write_json,
    write_text,
)


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


def _task_prefix(cfg: dict[str, Any]) -> str:
    prefix = safe_text(((cfg.get("preprocess", {}) or {}).get("task_prefix", ""))).strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "
    return prefix


def _strip_task_prefix(text: str, task_prefix: str) -> str:
    value = safe_text(text)
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value


def _word_len(text: str) -> int:
    return len(tokenize_words(text))


def _noise_score_row(row: pd.Series, *, task_prefix: str) -> dict[str, Any]:
    source = _strip_task_prefix(safe_text(row.get("source")), task_prefix)
    target = safe_text(row.get("target"))
    source_len = max(1, _word_len(source))
    target_len = max(1, _word_len(target))
    ratio = float(target_len) / float(source_len)
    ratio_penalty = 0.0
    if ratio < 0.30 or ratio > 2.20:
        ratio_penalty = 1.0
    elif ratio < 0.45 or ratio > 1.70:
        ratio_penalty = 0.5

    is_short_aligned = safe_text(row.get("chunk_mode")).startswith("short_aligned")
    align_type = safe_text(row.get("align_type"))
    align_cost = float(row.get("align_cost")) if pd.notna(row.get("align_cost")) else 0.0
    target_repeat = int(internal_repeat_score(target, ngram_size=3, min_count=3) > 0)
    source_repeat = int(internal_repeat_score(source, ngram_size=3, min_count=3) > 0)
    gap_count = source.count("<gap>")
    chunk_total = int(row.get("chunk_total", 1) or 1)

    score = 0.0
    score += 1.50 if is_short_aligned else 0.0
    score += min(2.00, float(align_cost) / 0.40)
    score += 0.75 if align_type == "1:2" else 0.0
    score += 0.25 if align_type == "2:1" else 0.0
    score += 0.50 * float(target_repeat)
    score += 0.25 * float(source_repeat)
    score += float(ratio_penalty)
    score += 0.25 if gap_count >= 3 else 0.0
    score += 0.25 if chunk_total >= 8 else 0.0

    return {
        "source_word_len": int(source_len),
        "target_word_len": int(target_len),
        "len_ratio": round(ratio, 4),
        "ratio_penalty": float(ratio_penalty),
        "is_short_aligned_proxy": bool(is_short_aligned),
        "align_cost_proxy": round(float(align_cost), 6),
        "target_repeat_proxy": int(target_repeat),
        "source_repeat_proxy": int(source_repeat),
        "gap_count_proxy": int(gap_count),
        "noise_score": round(float(score), 6),
    }


def _build_a2_variants(
    *,
    base_cfg: dict[str, Any],
    base_processed_dir: Path,
    out_dir: Path,
    keep_fracs: list[float],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_df = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(base_processed_dir / "folds.csv")
    task_prefix = _task_prefix(base_cfg)

    features = train_df.apply(lambda row: _noise_score_row(row, task_prefix=task_prefix), axis=1, result_type="expand")
    scored = pd.concat([train_df.copy(), features], axis=1)
    scored["rank_noise"] = scored["noise_score"].rank(method="first", ascending=True)
    scored.to_csv(out_dir / "a2_noise_rows.csv", index=False)

    variants: list[dict[str, Any]] = []
    total_rows = int(len(scored))
    for keep_frac in keep_fracs:
        label = f"keep{int(round(keep_frac * 100)):02d}"
        keep_rows = min(total_rows, max(32, int(round(float(total_rows) * float(keep_frac)))))
        kept = scored.nsmallest(keep_rows, columns=["noise_score", "rank_noise"]).copy().reset_index(drop=True)
        kept_ids = set(kept["oare_id"].astype(str).tolist())
        kept_folds = folds_df.loc[folds_df["oare_id"].astype(str).isin(kept_ids)].copy().reset_index(drop=True)

        processed_dir = REPO_ROOT / "data" / f"processed_byt5_chunks_align_gc_cost14_a2_{label}"
        processed_dir.mkdir(parents=True, exist_ok=True)
        kept.to_csv(processed_dir / "train_proc.csv", index=False)
        kept_folds.to_csv(processed_dir / "folds.csv", index=False)

        variant = {
            "label": label,
            "keep_frac": float(keep_frac),
            "processed_dir": str(processed_dir),
            "rows": int(len(kept)),
            "removed_rows": int(total_rows - len(kept)),
            "mean_noise_score": round(float(kept["noise_score"].mean()), 6),
            "p95_noise_score": round(float(kept["noise_score"].quantile(0.95)), 6),
            "short_aligned_rows": int(kept["is_short_aligned_proxy"].sum()),
            "short_aligned_ratio_pct": round(100.0 * float(kept["is_short_aligned_proxy"].mean()), 2),
            "align12_rows": int((kept["align_type"].fillna("").astype(str) == "1:2").sum()),
        }
        variants.append(variant)

    pd.DataFrame(variants).to_csv(out_dir / "a2_variants.csv", index=False)
    return scored, variants


def _build_monolingual_assets(*, out_dir: Path) -> dict[str, Any]:
    train_df = pd.read_csv(REPO_ROOT / "data" / "raw" / "train.csv")
    test_df = pd.read_csv(REPO_ROOT / "data" / "raw" / "test.csv")
    published_df = pd.read_csv(REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv")

    published_df["transliteration"] = published_df["transliteration"].fillna("").astype(str)
    published_df = published_df.loc[published_df["transliteration"].str.strip() != ""].copy().reset_index(drop=True)

    train_oares = set(train_df["oare_id"].astype(str).tolist())
    train_translit = set(train_df["transliteration"].fillna("").astype(str).tolist())

    published_source_only = published_df.loc[~published_df["oare_id"].astype(str).isin(train_oares)].copy().reset_index(drop=True)
    published_source_only = published_source_only.loc[
        ~published_source_only["transliteration"].astype(str).isin(train_translit)
    ].copy().reset_index(drop=True)

    mono_rows: list[dict[str, Any]] = []
    for row in train_df.to_dict(orient="records"):
        mono_rows.append(
            {
                "asset_id": f"train::{safe_text(row['oare_id'])}",
                "oare_id": safe_text(row["oare_id"]),
                "source_kind": "train_parallel_source",
                "genre_label": "train_parallel",
                "transliteration": normalize_whitespace(safe_text(row["transliteration"])),
            }
        )
    for row in test_df.to_dict(orient="records"):
        mono_rows.append(
            {
                "asset_id": f"test::{safe_text(row['id'])}",
                "oare_id": safe_text(row["id"]),
                "source_kind": "test_source_only",
                "genre_label": "test",
                "transliteration": normalize_whitespace(safe_text(row["transliteration"])),
            }
        )
    for row in published_source_only.to_dict(orient="records"):
        mono_rows.append(
            {
                "asset_id": f"pub::{safe_text(row['oare_id'])}",
                "oare_id": safe_text(row["oare_id"]),
                "source_kind": "published_source_only",
                "genre_label": safe_text(row.get("genre_label", "unknown")) or "unknown",
                "transliteration": normalize_whitespace(safe_text(row["transliteration"])),
            }
        )

    mono = pd.DataFrame(mono_rows)
    mono = mono.loc[mono["transliteration"].astype(str).str.strip() != ""].copy()
    mono = mono.drop_duplicates(subset=["transliteration"]).reset_index(drop=True)
    mono["source_word_len"] = mono["transliteration"].map(_word_len)
    mono = mono.sort_values(["source_word_len", "asset_id"], ascending=[False, True]).reset_index(drop=True)

    mono_corpus_csv = out_dir / "a1_monolingual_corpus.csv"
    mono.to_csv(mono_corpus_csv, index=False)

    pseudo_pool = mono.loc[mono["source_kind"].isin(["published_source_only", "test_source_only"])].copy().reset_index(drop=True)
    pseudo_pool_csv = out_dir / "a1_pseudo_source_pool.csv"
    pseudo_pool.to_csv(pseudo_pool_csv, index=False)

    summary = {
        "mono_corpus_csv": str(mono_corpus_csv),
        "pseudo_pool_csv": str(pseudo_pool_csv),
        "train_parallel_source_rows": int((mono["source_kind"] == "train_parallel_source").sum()),
        "test_source_only_rows": int((mono["source_kind"] == "test_source_only").sum()),
        "published_source_only_rows": int((mono["source_kind"] == "published_source_only").sum()),
        "mono_total_rows": int(len(mono)),
        "pseudo_pool_rows": int(len(pseudo_pool)),
        "top_genres": (
            mono["genre_label"].value_counts().head(10).rename_axis("genre_label").reset_index(name="rows").to_dict(orient="records")
        ),
    }
    write_json(out_dir / "a1_monolingual_inventory.json", summary)
    return summary


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


def _ensure_incumbent_anchor(
    *,
    python_exec: str,
    base_cfg_path: Path,
    checkpoint_dir: Path,
    fold: int,
    out_dir: Path,
) -> dict[str, Any]:
    tag = "taskform_a2_a1_incumbent_anchor64_20260310"
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


def _run_candidate_probe(
    *,
    python_exec: str,
    cfg_path: Path,
    cfg: dict[str, Any],
    init_adapter_dir: Path,
    fold: int,
    max_steps: int,
    eval_steps: int,
    tag: str,
) -> dict[str, Any]:
    run_dir = _candidate_run_dir(cfg, fold)
    checkpoint_dir = run_dir / "best_model"
    decode_best_path = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    diagnose_summary_path = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"
    run_summary_path = run_dir / "run_summary.json"

    if not (checkpoint_dir.exists() and run_summary_path.exists()):
        _run(
            [
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
                "--init-adapter-dir",
                str(init_adapter_dir),
                "--skip-final-predict",
            ]
        )

    if not decode_best_path.exists():
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
    decode_payload = _load_json(decode_best_path)

    if not diagnose_summary_path.exists():
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
    diag_payload = _load_json(diagnose_summary_path)
    run_summary = _load_json(run_summary_path)
    return {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "run_summary_path": str(run_summary_path),
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "gpu_peak_utilization_pct": float(run_summary.get("gpu_peak_utilization_pct", 0.0) or 0.0),
    }


def _run_tapt_smoke(
    *,
    python_exec: str,
    base_cfg: dict[str, Any],
    base_cfg_path: Path,
    out_dir: Path,
    mono_csv: Path,
) -> dict[str, Any]:
    tapt_cfg_path = out_dir / "generated_configs" / "taskform_a1_tapt_smoke.yaml"
    tapt_overrides = {
        "name": "taskform_a1_tapt_smoke",
        "tapt": {
            "run_dir": "runs/TASKFORM_A1_TAPT_SMOKE_20260310",
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
    }
    _write_yaml(tapt_cfg_path, _deep_merge(base_cfg, tapt_overrides))
    run_root = REPO_ROOT / "runs" / "TASKFORM_A1_TAPT_SMOKE_20260310"
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
            "--max-rows",
            "6000",
        ]
    )
    summary_path = run_root / "tapt_summary.json"
    summary = _load_json(summary_path)
    return {
        "config_path": str(tapt_cfg_path),
        "run_dir": str(run_root),
        "summary_path": str(summary_path),
        "best_model_dir": str(summary.get("best_model_dir", "")),
        "train_rows": int(summary.get("train_rows", 0)),
        "eval_rows": int(summary.get("eval_rows", 0)),
        "eval_loss": float((summary.get("eval_metrics", {}) or {}).get("eval_loss", 0.0) or 0.0),
    }


def _select_best_a2(records: list[dict[str, Any]], incumbent: dict[str, Any]) -> dict[str, Any]:
    best = max(records, key=lambda item: item["eval_geom"])
    best = copy.deepcopy(best)
    best["delta_geom_vs_incumbent"] = round(float(best["eval_geom"]) - float(incumbent["eval_geom"]), 4)
    return best


def _build_synthetic_mix_dir(
    *,
    source_processed_dir: Path,
    pseudo_csv: Path,
    out_dir: Path,
    mix_label: str,
) -> dict[str, Any]:
    train_df = pd.read_csv(source_processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(source_processed_dir / "folds.csv")
    pseudo_df = pd.read_csv(pseudo_csv)
    pseudo_df = pseudo_df.loc[pseudo_df["pseudo_target"].fillna("").astype(str).str.strip() != ""].copy().reset_index(drop=True)
    pseudo_df = pseudo_df.head(min(512, max(64, int(round(len(train_df) * 0.12))))).copy().reset_index(drop=True)

    synth_rows = pd.DataFrame(
        {
            "oare_id": pseudo_df["asset_id"].astype(str).map(lambda value: f"{value}__synthetic"),
            "transliteration": pseudo_df["transliteration"].fillna("").astype(str),
            "translation": pseudo_df["pseudo_target"].fillna("").astype(str),
            "source_raw": pseudo_df["transliteration"].fillna("").astype(str),
            "target_raw": pseudo_df["pseudo_target"].fillna("").astype(str),
            "source": pseudo_df["source"].fillna("").astype(str),
            "target": pseudo_df["pseudo_target"].fillna("").astype(str),
            "parent_oare_id": pseudo_df["asset_id"].astype(str).map(lambda value: f"{value}__synthetic_parent"),
            "chunk_index": 0,
            "chunk_total": 1,
            "is_chunk": False,
            "chunk_mode": "synthetic_source_mono",
            "is_short_aligned": False,
            "short_align_mode": "",
            "source_oare_id": pseudo_df["oare_id"].fillna("").astype(str),
            "align_type": "",
            "align_cost": math.nan,
        }
    )
    synth_folds = pd.DataFrame(
        {
            "oare_id": synth_rows["oare_id"].astype(str),
            "fold": -1,
            "group_key": "synthetic_source_mono",
            "group_kind": "fixed_train_only",
            "group_source": "synthetic",
            "parent_oare_id": synth_rows["parent_oare_id"].astype(str),
            "chunk_index": 0,
            "chunk_total": 1,
            "chunk_mode": "synthetic_source_mono",
            "short_align_mode": "",
            "align_type": "",
        }
    )

    mixed_train = pd.concat([train_df, synth_rows], ignore_index=True)
    mixed_folds = pd.concat([folds_df, synth_folds], ignore_index=True)

    processed_dir = REPO_ROOT / "data" / f"processed_byt5_chunks_align_gc_cost14_{mix_label}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    mixed_train.to_csv(processed_dir / "train_proc.csv", index=False)
    mixed_folds.to_csv(processed_dir / "folds.csv", index=False)
    return {
        "processed_dir": str(processed_dir),
        "base_rows": int(len(train_df)),
        "synthetic_rows": int(len(synth_rows)),
        "mixed_rows": int(len(mixed_train)),
    }


def _first_existing_col(frame: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of the candidate columns found: {candidates}")


def _compute_hard_subset_metrics(reconstructed_csv: Path, hard_ids_csv: Path) -> dict[str, Any]:
    pred_df = pd.read_csv(reconstructed_csv)
    hard_df = pd.read_csv(hard_ids_csv)
    pred_id_col = _first_existing_col(pred_df, ["oare_id", "parent_oare_id", "id"])
    hard_id_col = _first_existing_col(hard_df, ["oare_id", "parent_oare_id", "id"])
    hard_ids = set(hard_df[hard_id_col].fillna("").astype(str).tolist())
    subset = pred_df.loc[pred_df[pred_id_col].fillna("").astype(str).isin(hard_ids)].copy().reset_index(drop=True)
    if subset.empty:
        return {
            "rows": 0,
            "eval_geom": 0.0,
            "eval_bleu": 0.0,
            "eval_chrfpp": 0.0,
            "note": f"hard subset empty using pred_id_col={pred_id_col} hard_id_col={hard_id_col}",
        }
    metrics = compute_translation_metrics(
        predictions=subset["prediction"].fillna("").astype(str).tolist(),
        references=subset["reference"].fillna("").astype(str).tolist(),
    )
    return {
        "rows": int(len(subset)),
        "eval_geom": float(metrics["geom"]),
        "eval_bleu": float(metrics["bleu"]),
        "eval_chrfpp": float(metrics["chrfpp"]),
        "note": f"hard subset = routed_full parent ids via pred_id_col={pred_id_col} hard_id_col={hard_id_col}",
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
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "hard_subset": hard_metrics,
        "reconstructed_csv": str(reconstructed_csv),
        "official_like_note": "pending_official_bridge_same_as_local",
    }
    write_json(out_dir / "a1_fullval_summary.json", fullval)
    return fullval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--base-checkpoint-dir", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250")
    ap.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/taskform_a2_a1_20260310")
    args = ap.parse_args()

    python_exec = sys.executable
    started = time.time()

    base_cfg_path = _resolve_path(args.base_config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml")
    base_cfg = _load_yaml(base_cfg_path)
    base_checkpoint_dir = _resolve_path(args.base_checkpoint_dir, REPO_ROOT / "runs" / "missing_checkpoint")
    base_processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_a2_a1_20260310")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_cfg_dir = out_dir / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    scored_df, variants = _build_a2_variants(
        base_cfg=base_cfg,
        base_processed_dir=base_processed_dir,
        out_dir=out_dir,
        keep_fracs=[1.0, 0.97, 0.94],
    )
    a2_variants = [row for row in variants if row["label"] != "keep100"]
    monolingual_summary = _build_monolingual_assets(out_dir=out_dir)
    incumbent_anchor = _ensure_incumbent_anchor(
        python_exec=python_exec,
        base_cfg_path=base_cfg_path,
        checkpoint_dir=base_checkpoint_dir,
        fold=int(args.fold),
        out_dir=out_dir,
    )
    tapt_summary = _run_tapt_smoke(
        python_exec=python_exec,
        base_cfg=base_cfg,
        base_cfg_path=base_cfg_path,
        out_dir=out_dir,
        mono_csv=Path(monolingual_summary["mono_corpus_csv"]),
    )

    a2_probe_records: list[dict[str, Any]] = []
    for variant in a2_variants:
        label = safe_text(variant["label"])
        cfg_path = generated_cfg_dir / f"taskform_a2_{label}.yaml"
        cfg = _materialize_cfg(
            base_cfg=base_cfg,
            output_path=cfg_path,
            processed_dir=Path(variant["processed_dir"]),
            run_dir=f"runs/TASKFORM_A2_{label.upper()}_20260310",
        )
        record = _run_candidate_probe(
            python_exec=python_exec,
            cfg_path=cfg_path,
            cfg=cfg,
            init_adapter_dir=base_checkpoint_dir,
            fold=int(args.fold),
            max_steps=180,
            eval_steps=90,
            tag=f"taskform_a2_{label}_anchor64_20260310",
        )
        record["label"] = label
        record["processed_dir"] = variant["processed_dir"]
        record["delta_geom_vs_incumbent"] = round(float(record["eval_geom"]) - float(incumbent_anchor["eval_geom"]), 4)
        a2_probe_records.append(record)

    best_a2 = _select_best_a2(a2_probe_records, incumbent_anchor)
    a2_status = "accept_to_a1" if float(best_a2["delta_geom_vs_incumbent"]) >= 0.05 else "review_continue" if float(best_a2["delta_geom_vs_incumbent"]) >= -0.20 else "reject_stop"

    pseudo_csv = out_dir / "a1_pseudo_targets.csv"
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
            "512",
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

    a1_executed = False
    a1_probe: dict[str, Any] | None = None
    a1_fullval: dict[str, Any] | None = None
    mix_summary: dict[str, Any] | None = None
    if a2_status in {"accept_to_a1", "review_continue"}:
        mix_summary = _build_synthetic_mix_dir(
            source_processed_dir=Path(best_a2["processed_dir"]),
            pseudo_csv=pseudo_csv,
            out_dir=out_dir,
            mix_label=f"a1_synthmix_{safe_text(best_a2['label'])}",
        )
        a1_cfg_path = generated_cfg_dir / f"taskform_a1_tapt_{safe_text(best_a2['label'])}.yaml"
        a1_cfg = _materialize_cfg(
            base_cfg=base_cfg,
            output_path=a1_cfg_path,
            processed_dir=Path(mix_summary["processed_dir"]),
            run_dir=f"runs/TASKFORM_A1_TAPT_{safe_text(best_a2['label']).upper()}_20260310",
        )
        a1_probe = _run_candidate_probe(
            python_exec=python_exec,
            cfg_path=a1_cfg_path,
            cfg=a1_cfg,
            init_adapter_dir=Path(tapt_summary["best_model_dir"]),
            fold=int(args.fold),
            max_steps=180,
            eval_steps=90,
            tag=f"taskform_a1_tapt_{safe_text(best_a2['label'])}_anchor64_20260310",
        )
        a1_probe["label"] = f"tapt_{safe_text(best_a2['label'])}"
        a1_probe["delta_geom_vs_incumbent"] = round(float(a1_probe["eval_geom"]) - float(incumbent_anchor["eval_geom"]), 4)
        a1_probe["delta_geom_vs_best_a2"] = round(float(a1_probe["eval_geom"]) - float(best_a2["eval_geom"]), 4)
        a1_executed = True
        if float(a1_probe["delta_geom_vs_incumbent"]) >= 0.05:
            a1_fullval = _maybe_run_fullval(
                python_exec=python_exec,
                cfg_path=a1_cfg_path,
                cfg=a1_cfg,
                checkpoint_dir=Path(a1_probe["checkpoint_dir"]),
                fold=int(args.fold),
                tag=f"taskform_a1_tapt_{safe_text(best_a2['label'])}_fullval_20260310",
                out_dir=out_dir,
            )

    summary = {
        "base_config_path": str(base_cfg_path),
        "base_checkpoint_dir": str(base_checkpoint_dir),
        "base_processed_dir": str(base_processed_dir),
        "incumbent_anchor64": incumbent_anchor,
        "a2": {
            "noise_rows_csv": str(out_dir / "a2_noise_rows.csv"),
            "variants_csv": str(out_dir / "a2_variants.csv"),
            "variants": variants,
            "probe_records": a2_probe_records,
            "best_variant": best_a2,
            "status": a2_status,
        },
        "a1": {
            "monolingual_inventory": monolingual_summary,
            "tapt_smoke": tapt_summary,
            "pseudo_targets_csv": str(pseudo_csv),
            "executed_probe": bool(a1_executed),
            "synthetic_mix": mix_summary,
            "probe": a1_probe,
            "fullval": a1_fullval,
        },
        "official_metric_bridge": {
            "status": "missing_bridge",
            "note": "official-like remains same as local until bridge lands",
        },
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    write_json(out_dir / "summary.json", summary)

    a2_table = [
        {
            "label": row["label"],
            "geom": f"{row['eval_geom']:.4f}",
            "delta_vs_inc": f"{row['delta_geom_vs_incumbent']:+.4f}",
            "gpu_util": f"{row['gpu_peak_utilization_pct']:.1f}",
        }
        for row in a2_probe_records
    ]
    sections = [
        "# Taskform A2-A1 Summary",
        "",
        f"- incumbent anchor64 geom: `{incumbent_anchor['eval_geom']:.4f}`",
        f"- A2 status: `{a2_status}`",
        f"- A1 probe executed: `{a1_executed}`",
        "",
        "## A2 probe",
        "",
        markdown_table(a2_table, ["label", "geom", "delta_vs_inc", "gpu_util"]) if a2_table else "_no rows_",
        "",
        "## A1",
        "",
        f"- mono rows: `{monolingual_summary['mono_total_rows']}`",
        f"- TAPT best model: `{tapt_summary['best_model_dir']}`",
    ]
    if a1_probe:
        sections.extend(
            [
                f"- A1 anchor64 geom: `{a1_probe['eval_geom']:.4f}`",
                f"- delta vs incumbent: `{a1_probe['delta_geom_vs_incumbent']:+.4f}`",
                f"- delta vs best A2: `{a1_probe['delta_geom_vs_best_a2']:+.4f}`",
            ]
        )
    if a1_fullval:
        sections.extend(
            [
                f"- A1 full-val local geom: `{a1_fullval['eval_geom']:.4f}`",
                f"- A1 hard geom: `{a1_fullval['hard_subset']['eval_geom']:.4f}`",
                "- official-like: `same as local until bridge lands`",
            ]
        )
    write_text(out_dir / "gate_report.md", "\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
