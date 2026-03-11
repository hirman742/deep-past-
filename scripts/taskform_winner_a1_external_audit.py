from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cleaning.normalize import normalize_source  # noqa: E402
from generation_utils import apply_task_prefix, normalize_task_prefix  # noqa: E402


INLINE_WS_RE = re.compile(r"[^\S\n]+", flags=re.UNICODE)


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


def _load_cleaning_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "cleaning" / "configs" / "cleaning.t0.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _normalize_text(
    text: str,
    *,
    apply_t0: bool,
    cleaning_cfg: dict[str, Any],
    strip_text: bool,
    fold_ws: bool,
    lowercase: bool,
    task_prefix: str,
) -> str:
    value = text if isinstance(text, str) else ""
    if apply_t0:
        value, _ = normalize_source(value, config=cleaning_cfg)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if fold_ws:
        value = INLINE_WS_RE.sub(" ", value)
    if strip_text:
        value = value.strip()
    if lowercase:
        value = value.lower()
    return apply_task_prefix(value, task_prefix)


def _strip_task_prefix(text: str, task_prefix: str) -> str:
    value = text if isinstance(text, str) else ""
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value.strip()


def _detect_columns(frame: pd.DataFrame) -> tuple[str, str]:
    source_candidates = ["source", "transliteration", "akkadian", "input_text", "text"]
    target_candidates = ["target", "translation", "english", "output_text"]
    src_col = next((c for c in source_candidates if c in frame.columns), None)
    tgt_col = next((c for c in target_candidates if c in frame.columns), None)
    if src_col is None or tgt_col is None:
        raise KeyError(f"Unable to detect source/target columns from: {list(frame.columns)}")
    return src_col, tgt_col


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml",
    )
    ap.add_argument("--base-processed-dir", default="")
    ap.add_argument("--raw-train-csv", default="data/raw/train.csv")
    ap.add_argument("--raw-test-csv", default="data/raw/test.csv")
    ap.add_argument("--external-csv", default="data/external/oracc_parallel.csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_20260310")
    args = ap.parse_args()

    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a1_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = _resolve_path(
        args.base_config,
        REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
    )
    base_cfg = _load_yaml(base_cfg_path)
    base_processed_dir = _resolve_path(
        args.base_processed_dir,
        _resolve_path((base_cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed"),
    )

    raw_train_path = _resolve_path(args.raw_train_csv, REPO_ROOT / "data" / "raw" / "train.csv")
    raw_test_path = _resolve_path(args.raw_test_csv, REPO_ROOT / "data" / "raw" / "test.csv")
    external_path = _resolve_path(args.external_csv, REPO_ROOT / "data" / "external" / "oracc_parallel.csv")

    preprocess_cfg = base_cfg.get("preprocess", {}) or {}
    cleaning_cfg = _load_cleaning_config()
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))
    apply_t0 = bool(preprocess_cfg.get("apply_t0_normalize", False))
    strip_text = bool(preprocess_cfg.get("strip_text", True))
    fold_ws = bool(preprocess_cfg.get("fold_inline_whitespace", True))
    lowercase_source = bool(preprocess_cfg.get("lowercase_source", False))

    train_proc = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds = pd.read_csv(base_processed_dir / "folds.csv")
    merged = train_proc.merge(folds[["oare_id", "fold"]], on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(args.fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(args.fold)].copy().reset_index(drop=True)

    raw_train = pd.read_csv(raw_train_path)
    raw_test = pd.read_csv(raw_test_path)

    val_sources = (
        val_visible["source"]
        .fillna("")
        .astype(str)
        .map(lambda x: _strip_task_prefix(x, task_prefix))
        .tolist()
    )
    val_source_set = set(val_sources)
    train_visible_sources = (
        train_visible["source"]
        .fillna("")
        .astype(str)
        .map(lambda x: _strip_task_prefix(x, task_prefix))
        .tolist()
    )

    raw_train_norm = raw_train["transliteration"].fillna("").astype(str).map(
        lambda x: _normalize_text(
            x,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
            task_prefix=task_prefix,
        )
    )
    raw_train_norm = raw_train_norm.map(lambda x: _strip_task_prefix(x, task_prefix))

    raw_test_norm = raw_test["transliteration"].fillna("").astype(str).map(
        lambda x: _normalize_text(
            x,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
            task_prefix=task_prefix,
        )
    )
    raw_test_norm = raw_test_norm.map(lambda x: _strip_task_prefix(x, task_prefix))

    source_registry_rows = [
        {
            "source_origin": "competition_internal",
            "license_note": "competition_train_visible_only",
            "raw_parent_rows": int(raw_train.shape[0]),
            "post_clean_rows": int(raw_train_norm.str.strip().ne("").sum()),
            "post_chunk_rows": int(train_visible.shape[0]),
            "post_shortalign_rows": int(
                train_visible.get("is_short_aligned", pd.Series(False)).fillna(False).astype(str).str.lower().isin({"1", "true"}).sum()
            ),
            "status": "ready",
            "path": str(base_processed_dir),
        }
    ]

    mix_plan_rows = []
    for ratio in (0.10, 0.30, 0.50):
        mix_plan_rows.append(
            {
                "label": f"E{int(round(ratio * 100)):02d}",
                "ratio_vs_train_visible_rows": float(ratio),
                "base_train_visible_rows": int(train_visible.shape[0]),
                "target_external_rows": int(round(train_visible.shape[0] * ratio)),
                "status": "pending_external_audit",
            }
        )

    overlap_audit: dict[str, Any] = {
        "status": "blocked_missing_external_parallel",
        "external_csv": str(external_path),
        "reason": "external_parallel_csv_not_found",
        "checks": {
            "fold0_val_parent_rows": int(val_visible["parent_oare_id"].astype(str).nunique()) if "parent_oare_id" in val_visible.columns else None,
            "fold0_val_chunk_rows": int(val_visible.shape[0]),
            "train_visible_rows": int(train_visible.shape[0]),
            "base_processed_dir": str(base_processed_dir),
        },
    }

    manifest: dict[str, Any] = {
        "status": "blocked_missing_external_parallel",
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "fold": int(args.fold),
        "internal": {
            "train_visible_rows": int(train_visible.shape[0]),
            "val_visible_rows": int(val_visible.shape[0]),
            "val_parent_rows": int(val_visible["parent_oare_id"].astype(str).nunique()) if "parent_oare_id" in val_visible.columns else None,
            "raw_train_rows": int(raw_train.shape[0]),
            "raw_test_rows": int(raw_test.shape[0]),
        },
        "paths": {
            "source_registry_csv": str(report_dir / "source_registry.csv"),
            "mix_plan_csv": str(report_dir / "mix_plan.csv"),
            "overlap_audit_json": str(report_dir / "overlap_audit.json"),
            "dedup_manifest_json": str(report_dir / "dedup_manifest.json"),
            "manifest_json": str(report_dir / "manifest.json"),
        },
    }

    dedup_manifest: dict[str, Any] = {
        "status": "blocked_missing_external_parallel",
        "fold": int(args.fold),
        "internal": {
            "train_visible_rows": int(train_visible.shape[0]),
            "train_visible_unique_source_norm_rows": int(pd.Series(train_visible_sources).nunique()),
            "train_visible_duplicate_source_norm_rows": int(train_visible.shape[0] - pd.Series(train_visible_sources).nunique()),
            "val_visible_rows": int(val_visible.shape[0]),
            "val_visible_unique_source_norm_rows": int(pd.Series(val_sources).nunique()),
            "val_visible_duplicate_source_norm_rows": int(val_visible.shape[0] - pd.Series(val_sources).nunique()),
        },
        "external": {
            "path": str(external_path),
            "status": "missing_csv",
        },
    }

    if external_path.exists():
        external_df = pd.read_csv(external_path)
        src_col, tgt_col = _detect_columns(external_df)
        external = external_df[[src_col, tgt_col]].copy()
        external.columns = ["source", "target"]
        external = external.fillna("")
        external = external.loc[
            (external["source"].astype(str).str.strip() != "")
            & (external["target"].astype(str).str.strip() != "")
        ].copy().reset_index(drop=True)

        external["source_norm"] = external["source"].astype(str).map(
            lambda x: _normalize_text(
                x,
                apply_t0=apply_t0,
                cleaning_cfg=cleaning_cfg,
                strip_text=strip_text,
                fold_ws=fold_ws,
                lowercase=lowercase_source,
                task_prefix=task_prefix,
            )
        )
        external["source_norm"] = external["source_norm"].map(lambda x: _strip_task_prefix(x, task_prefix))

        raw_train_set = set(raw_train_norm.tolist())
        raw_test_set = set(raw_test_norm.tolist())
        ext_norm_values = external["source_norm"].fillna("").astype(str)

        exact_overlap_val = int(ext_norm_values.isin(val_source_set).sum())
        exact_overlap_test = int(ext_norm_values.isin(raw_test_set).sum())
        exact_overlap_train = int(ext_norm_values.isin(raw_train_set).sum())
        unique_external_rows = int(ext_norm_values.nunique())
        filtered_external = external.loc[
            ~ext_norm_values.isin(val_source_set | raw_test_set | raw_train_set)
        ].copy().reset_index(drop=True)

        source_registry_rows.append(
            {
                "source_origin": "external_parallel",
                "license_note": "user_supplied_csv",
                "raw_parent_rows": int(external_df.shape[0]),
                "post_clean_rows": int(external.shape[0]),
                "post_chunk_rows": None,
                "post_shortalign_rows": None,
                "status": "ready",
                "path": str(external_path),
            }
        )

        overlap_audit = {
            "status": "ready",
            "external_csv": str(external_path),
            "checks": {
                "external_rows_raw": int(external_df.shape[0]),
                "external_rows_post_clean": int(external.shape[0]),
                "external_rows_unique_source_norm": unique_external_rows,
                "fold0_val_exact_overlap_rows": exact_overlap_val,
                "test_exact_overlap_rows": exact_overlap_test,
                "train_exact_overlap_rows": exact_overlap_train,
                "external_rows_after_overlap_filter": int(filtered_external.shape[0]),
            },
        }
        manifest["status"] = "ready_for_mix_build"
        manifest["external"] = {
            "path": str(external_path),
            "rows_raw": int(external_df.shape[0]),
            "rows_post_clean": int(external.shape[0]),
            "rows_after_overlap_filter": int(filtered_external.shape[0]),
        }
        dedup_manifest = {
            "status": "ready_for_mix_build",
            "fold": int(args.fold),
            "internal": {
                "train_visible_rows": int(train_visible.shape[0]),
                "train_visible_unique_source_norm_rows": int(pd.Series(train_visible_sources).nunique()),
                "train_visible_duplicate_source_norm_rows": int(train_visible.shape[0] - pd.Series(train_visible_sources).nunique()),
                "val_visible_rows": int(val_visible.shape[0]),
                "val_visible_unique_source_norm_rows": int(pd.Series(val_sources).nunique()),
                "val_visible_duplicate_source_norm_rows": int(val_visible.shape[0] - pd.Series(val_sources).nunique()),
            },
            "external": {
                "path": str(external_path),
                "rows_raw": int(external_df.shape[0]),
                "rows_post_clean": int(external.shape[0]),
                "unique_source_norm_rows": int(unique_external_rows),
                "duplicate_source_norm_rows": int(external.shape[0] - unique_external_rows),
                "rows_after_overlap_filter": int(filtered_external.shape[0]),
                "duplicate_rows_after_overlap_filter": int(filtered_external.shape[0] - filtered_external["source_norm"].nunique()),
            },
        }
        for row in mix_plan_rows:
            row["available_external_rows_post_filter"] = int(filtered_external.shape[0])
            row["status"] = (
                "ready"
                if int(filtered_external.shape[0]) >= int(row["target_external_rows"])
                else "insufficient_external_rows"
            )

    pd.DataFrame(source_registry_rows).to_csv(report_dir / "source_registry.csv", index=False)
    pd.DataFrame(mix_plan_rows).to_csv(report_dir / "mix_plan.csv", index=False)
    _write_json(report_dir / "overlap_audit.json", overlap_audit)
    _write_json(report_dir / "dedup_manifest.json", dedup_manifest)
    _write_json(report_dir / "manifest.json", manifest)

    print(f"OK: wrote {report_dir / 'source_registry.csv'}")
    print(f"OK: wrote {report_dir / 'mix_plan.csv'}")
    print(f"OK: wrote {report_dir / 'overlap_audit.json'}")
    print(f"OK: wrote {report_dir / 'dedup_manifest.json'}")
    print(f"OK: wrote {report_dir / 'manifest.json'}")
    print(f"INFO: status={manifest['status']}")


if __name__ == "__main__":
    main()
