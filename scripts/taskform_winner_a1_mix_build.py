from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from generation_utils import apply_task_prefix, normalize_task_prefix
from taskform_phase12_common import resolve_path, safe_text, write_json, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _stable_fold(key: str) -> int:
    digest = hashlib.md5(safe_text(key).encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % 4)


def _bucket_key(text: str) -> str:
    tokens = safe_text(text).lower().split()
    if not tokens:
        return "external|empty"
    head = "|".join(tokens[:3])
    return f"external|{head}|l{min(len(tokens) // 8, 15)}"


def _prepare_parent_input(frame: pd.DataFrame, task_prefix: str) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["orig_oare_id"] = out["oare_id"].fillna("").astype(str)
    out["oare_id"] = out["orig_oare_id"].map(lambda value: f"extpub__{value}")
    out["source_raw"] = out["source"].fillna("").astype(str)
    out["target_raw"] = out["target"].fillna("").astype(str)
    out["source"] = out["source_raw"].map(lambda value: apply_task_prefix(value, task_prefix))
    out["target"] = out["target_raw"]
    out["is_short_aligned"] = False
    out["short_align_mode"] = ""
    out["source_oare_id"] = out["orig_oare_id"]
    out["align_type"] = ""
    out["align_cost"] = ""
    keep_cols = [
        "oare_id",
        "orig_oare_id",
        "source_raw",
        "target_raw",
        "source",
        "target",
        "source_origin",
        "license_note",
        "row_origin",
        "genre_label",
        "label",
        "anchor_token",
        "anchor_type",
        "source_token_len",
        "target_word_len",
        "builder",
        "is_short_aligned",
        "short_align_mode",
        "source_oare_id",
        "align_type",
        "align_cost",
    ]
    available = [col for col in keep_cols if col in out.columns]
    return out[available].copy()


def _build_parent_folds(parent_input: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parent_id, group in parent_input.groupby("oare_id", sort=False):
        source_for_bucket = safe_text(group["source_raw"].iloc[0])
        rows.append(
            {
                "oare_id": str(parent_id),
                "parent_oare_id": str(parent_id),
                "fold": int(_stable_fold(str(parent_id))),
                "group_key": _bucket_key(source_for_bucket),
                "group_kind": "external_parent",
                "group_source": "silver_external",
            }
        )
    return pd.DataFrame(rows)


def _prepare_sentence_rows(frame: pd.DataFrame, task_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy().reset_index(drop=True)
    work["orig_oare_id"] = work["oare_id"].fillna("").astype(str)
    work["sentence_uuid"] = work["sentence_uuid"].fillna("").astype(str)
    fallback_ids = [f"row{i:05d}" for i in range(len(work))]
    work["sentence_uid"] = [
        sent if sent else fallback
        for sent, fallback in zip(work["sentence_uuid"].tolist(), fallback_ids)
    ]
    work["parent_oare_id"] = work["orig_oare_id"].map(lambda value: f"extpub__{value}")
    work["oare_id"] = [
        f"{parent}__s{sent}"
        for parent, sent in zip(work["parent_oare_id"].tolist(), work["sentence_uid"].tolist())
    ]
    work["source_raw"] = work["source"].fillna("").astype(str)
    work["target_raw"] = work["target"].fillna("").astype(str)
    work["source"] = work["source_raw"].map(lambda value: apply_task_prefix(value, task_prefix))
    work["target"] = work["target_raw"]
    work["chunk_index"] = 0
    work["chunk_total"] = 1
    work["is_chunk"] = False
    work["chunk_mode"] = "external_sentence_silver"
    work["is_short_aligned"] = False
    work["short_align_mode"] = ""
    work["source_oare_id"] = work["orig_oare_id"]
    work["align_type"] = ""
    work["align_cost"] = ""
    train_keep = [
        "oare_id",
        "source_raw",
        "target_raw",
        "source",
        "target",
        "parent_oare_id",
        "chunk_index",
        "chunk_total",
        "is_chunk",
        "chunk_mode",
        "is_short_aligned",
        "short_align_mode",
        "source_oare_id",
        "align_type",
        "align_cost",
        "source_origin",
        "license_note",
        "row_origin",
        "genre_label",
        "label",
        "anchor_token",
        "anchor_type",
        "source_token_len",
        "target_word_len",
        "builder",
        "orig_oare_id",
        "sentence_uuid",
    ]
    train_rows = work[[col for col in train_keep if col in work.columns]].copy()

    folds_rows = pd.DataFrame(
        {
            "oare_id": train_rows["oare_id"].astype(str),
            "fold": train_rows["parent_oare_id"].astype(str).map(_stable_fold).astype(int),
            "group_key": train_rows["source_raw"].astype(str).map(_bucket_key),
            "group_kind": "external_sentence",
            "group_source": "silver_external",
            "parent_oare_id": train_rows["parent_oare_id"].astype(str),
            "chunk_index": 0,
            "chunk_total": 1,
            "chunk_mode": "external_sentence_silver",
            "short_align_mode": "",
            "align_type": "",
        }
    )
    return train_rows, folds_rows


def _sample_parent_prefix_sizes(group_sizes: list[int], desired_rows: int) -> int:
    if desired_rows <= 0 or not group_sizes:
        return 0
    cumulative = 0
    best_k = 0
    best_gap = abs(desired_rows)
    for idx, size in enumerate(group_sizes, start=1):
        cumulative += int(size)
        gap = abs(desired_rows - cumulative)
        if gap <= best_gap:
            best_gap = gap
            best_k = idx
    return best_k


def _write_processed_dir(train_df: pd.DataFrame, folds_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train_proc.csv", index=False)
    folds_df.to_csv(out_dir / "folds.csv", index=False)


def _actual_external_rows(frame: pd.DataFrame, internal_ids: set[str]) -> int:
    return int((~frame["oare_id"].astype(str).isin(internal_ids)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--external-csv", default="data/external/oracc_parallel.csv")
    ap.add_argument("--mix-plan-csv", default="reports/taskform_winner_a1_20260310/mix_plan.csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sentence-min-source-tokens", type=int, default=6)
    ap.add_argument("--sentence-min-target-words", type=int, default=6)
    ap.add_argument("--sentence-target-share", type=float, default=0.35)
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_mix_build_20260310")
    ap.add_argument("--config-dir", default="reports/taskform_winner_a1_mix_build_20260310/generated_configs")
    args = ap.parse_args()

    random.seed(int(args.seed))

    base_cfg_path = resolve_path(
        args.base_config,
        REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
    )
    base_cfg = _load_yaml(base_cfg_path)
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a1_mix_build_20260310")
    config_dir = resolve_path(args.config_dir, report_dir / "generated_configs")
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    task_prefix = normalize_task_prefix(((base_cfg.get("preprocess", {}) or {}).get("task_prefix", "")))
    base_processed_dir = resolve_path(
        ((base_cfg.get("paths", {}) or {}).get("processed_dir", "")),
        REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14",
    )
    external_csv = resolve_path(args.external_csv, REPO_ROOT / "data" / "external" / "oracc_parallel.csv")
    mix_plan_csv = resolve_path(args.mix_plan_csv, REPO_ROOT / "reports" / "taskform_winner_a1_20260310" / "mix_plan.csv")

    base_train = pd.read_csv(base_processed_dir / "train_proc.csv")
    base_folds = pd.read_csv(base_processed_dir / "folds.csv")
    base_merged = base_train.merge(base_folds[["oare_id", "fold"]], on="oare_id", how="inner")
    val_visible = base_merged.loc[base_merged["fold"] == int(args.fold)].copy().reset_index(drop=True)
    train_visible = base_merged.loc[base_merged["fold"] != int(args.fold)].copy().reset_index(drop=True)
    mix_plan = pd.read_csv(mix_plan_csv)
    external_df = pd.read_csv(external_csv)

    sentence_df = external_df.loc[external_df["row_origin"].fillna("") == "published_sentence_silver"].copy().reset_index(drop=True)
    sentence_df = sentence_df.loc[
        (sentence_df["source_token_len"].fillna(0).astype(float) >= float(args.sentence_min_source_tokens))
        & (sentence_df["target_word_len"].fillna(0).astype(float) >= float(args.sentence_min_target_words))
    ].copy().reset_index(drop=True)
    parent_df = external_df.loc[external_df["row_origin"].fillna("") == "published_agg_parent"].copy().reset_index(drop=True)

    parent_input = _prepare_parent_input(parent_df, task_prefix)
    parent_folds = _build_parent_folds(parent_input)

    temp_dir = report_dir / "_tmp_parent_chunk"
    temp_dir.mkdir(parents=True, exist_ok=True)
    parent_input_csv = temp_dir / "parent_input.csv"
    parent_folds_csv = temp_dir / "parent_folds.csv"
    parent_chunk_csv = temp_dir / "parent_chunk_train.csv"
    parent_chunk_folds_csv = temp_dir / "parent_chunk_folds.csv"
    parent_report_json = temp_dir / "parent_chunk_report.json"
    parent_map_csv = temp_dir / "parent_chunk_map.csv"
    parent_input.to_csv(parent_input_csv, index=False)
    parent_folds.to_csv(parent_folds_csv, index=False)

    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_long_chunks.py"),
            "--config",
            str(base_cfg_path),
            "--input-train",
            str(parent_input_csv),
            "--input-folds",
            str(parent_folds_csv),
            "--output-train",
            str(parent_chunk_csv),
            "--output-folds",
            str(parent_chunk_folds_csv),
            "--output-map-train",
            str(parent_map_csv),
            "--report-json",
            str(parent_report_json),
            "--source-col",
            "source",
            "--target-col",
            "target",
        ]
    )

    parent_chunks = pd.read_csv(parent_chunk_csv)
    parent_chunk_folds = pd.read_csv(parent_chunk_folds_csv)
    sentence_rows, sentence_folds = _prepare_sentence_rows(sentence_df, task_prefix)

    parent_chunks["row_origin"] = parent_chunks["row_origin"].fillna("published_agg_parent").astype(str)
    parent_chunks["source_origin"] = parent_chunks["source_origin"].fillna("oare_published_silver").astype(str)
    parent_chunks["license_note"] = parent_chunks["license_note"].fillna("local_oare_assets_for_offline_research_only").astype(str)

    parent_groups = []
    for parent_id, group in parent_chunks.groupby("parent_oare_id", sort=False):
        parent_groups.append((str(parent_id), group.copy().reset_index(drop=True), int(len(group))))
    parent_rng = random.Random(int(args.seed) + 101)
    parent_rng.shuffle(parent_groups)
    parent_group_sizes = [size for _, _, size in parent_groups]
    parent_group_frames = {parent_id: group for parent_id, group, _ in parent_groups}
    parent_group_ids = [parent_id for parent_id, _, _ in parent_groups]
    parent_cum = []
    running = 0
    for size in parent_group_sizes:
        running += int(size)
        parent_cum.append(running)

    sentence_rng = random.Random(int(args.seed) + 202)
    sentence_indices = list(range(len(sentence_rows)))
    sentence_rng.shuffle(sentence_indices)
    sentence_rows = sentence_rows.iloc[sentence_indices].reset_index(drop=True)
    sentence_folds = sentence_folds.iloc[sentence_indices].reset_index(drop=True)

    internal_only_dir = REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_a1_internal_only_fold0"
    _write_processed_dir(base_train.copy(), base_folds.copy(), internal_only_dir)

    def build_mix(label: str, target_external_rows: int) -> dict[str, Any]:
        desired_sentence_rows = int(round(float(target_external_rows) * float(args.sentence_target_share)))
        desired_parent_rows = int(target_external_rows) - desired_sentence_rows
        parent_prefix = _sample_parent_prefix_sizes(parent_group_sizes, desired_parent_rows)
        selected_parent_ids = parent_group_ids[:parent_prefix]
        selected_parent_rows = int(parent_cum[parent_prefix - 1]) if parent_prefix > 0 else 0
        sentence_take = max(0, int(target_external_rows) - selected_parent_rows)
        selected_parent_df = pd.concat(
            [parent_group_frames[parent_id] for parent_id in selected_parent_ids],
            ignore_index=True,
        ) if selected_parent_ids else parent_chunks.iloc[0:0].copy()
        selected_parent_fold_df = parent_chunk_folds.loc[
            parent_chunk_folds["parent_oare_id"].astype(str).isin(set(selected_parent_ids))
        ].copy().reset_index(drop=True)
        selected_sentence_df = sentence_rows.head(sentence_take).copy().reset_index(drop=True)
        selected_sentence_folds_df = sentence_folds.head(sentence_take).copy().reset_index(drop=True)

        ext_train = pd.concat([selected_parent_df, selected_sentence_df], ignore_index=True)
        ext_folds = pd.concat([selected_parent_fold_df, selected_sentence_folds_df], ignore_index=True)
        ext_train = ext_train.reset_index(drop=True)
        ext_folds = ext_folds.reset_index(drop=True)

        mixed_train = pd.concat([base_train.copy(), ext_train], ignore_index=True)
        mixed_folds = pd.concat([base_folds.copy(), ext_folds], ignore_index=True)

        processed_dir = REPO_ROOT / f"data/processed_byt5_chunks_align_gc_cost14_a1_{label.lower()}_fold0"
        _write_processed_dir(mixed_train, mixed_folds, processed_dir)

        cfg = json.loads(json.dumps(base_cfg))
        cfg["name"] = f"taskform_winner_a1_{label.lower()}"
        paths_cfg = (cfg.get("paths", {}) or {}).copy()
        paths_cfg["processed_dir"] = str(processed_dir.relative_to(REPO_ROOT))
        paths_cfg["run_dir"] = f"runs/TASKFORM_WINNER_A1_{label}_20260310"
        cfg["paths"] = paths_cfg
        cfg_path = config_dir / f"taskform_winner_a1_{label.lower()}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")

        mixed_train_merged = mixed_train.merge(mixed_folds[["oare_id", "fold"]], on="oare_id", how="inner")
        val_after = mixed_train_merged.loc[mixed_train_merged["fold"] == int(args.fold)].copy().reset_index(drop=True)
        val_unchanged = bool(val_after["oare_id"].astype(str).tolist() == val_visible["oare_id"].astype(str).tolist())
        ext_fold0_rows = int(
            mixed_train_merged.loc[
                (mixed_train_merged["fold"] == int(args.fold))
                & (~mixed_train_merged["oare_id"].astype(str).isin(set(base_train["oare_id"].astype(str))))
            ].shape[0]
        )
        row_origin_counts = {
            key: int(value)
            for key, value in ext_train.get("row_origin", pd.Series(dtype=str)).fillna("").value_counts().to_dict().items()
        }
        chunk_mode_counts = {
            key: int(value)
            for key, value in ext_train.get("chunk_mode", pd.Series(dtype=str)).fillna("").value_counts().to_dict().items()
        }
        return {
            "label": label,
            "processed_dir": str(processed_dir),
            "config_path": str(cfg_path),
            "target_external_rows": int(target_external_rows),
            "actual_external_rows": int(len(ext_train)),
            "actual_ratio_vs_train_visible": round(float(len(ext_train)) / float(max(1, len(train_visible))), 4),
            "desired_sentence_rows": int(desired_sentence_rows),
            "actual_sentence_rows": int(len(selected_sentence_df)),
            "actual_parent_chunk_rows": int(len(selected_parent_df)),
            "actual_sentence_share": round(float(len(selected_sentence_df)) / float(max(1, len(ext_train))), 4),
            "selected_parent_groups": int(len(selected_parent_ids)),
            "ext_fold0_rows": ext_fold0_rows,
            "val_rows_unchanged": bool(val_unchanged),
            "row_origin_counts": row_origin_counts,
            "chunk_mode_counts": chunk_mode_counts,
        }

    builds = {
        "internal_only_matched": {
            "label": "internal_only_matched",
            "processed_dir": str(internal_only_dir),
            "config_path": str(config_dir / "taskform_winner_a1_internal_only_matched.yaml"),
            "target_external_rows": 0,
            "actual_external_rows": 0,
            "actual_ratio_vs_train_visible": 0.0,
            "desired_sentence_rows": 0,
            "actual_sentence_rows": 0,
            "actual_parent_chunk_rows": 0,
            "actual_sentence_share": 0.0,
            "selected_parent_groups": 0,
            "ext_fold0_rows": 0,
            "val_rows_unchanged": True,
            "row_origin_counts": {},
            "chunk_mode_counts": {},
        }
    }

    internal_cfg = json.loads(json.dumps(base_cfg))
    internal_cfg["name"] = "taskform_winner_a1_internal_only_matched"
    internal_paths = (internal_cfg.get("paths", {}) or {}).copy()
    internal_paths["processed_dir"] = str(internal_only_dir.relative_to(REPO_ROOT))
    internal_paths["run_dir"] = "runs/TASKFORM_WINNER_A1_INTERNAL_ONLY_20260310"
    internal_cfg["paths"] = internal_paths
    internal_cfg_path = config_dir / "taskform_winner_a1_internal_only_matched.yaml"
    internal_cfg_path.write_text(yaml.safe_dump(internal_cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    builds["internal_only_matched"]["config_path"] = str(internal_cfg_path)

    for _, row in mix_plan.iterrows():
        label = safe_text(row.get("label", "")).strip()
        if not label:
            continue
        builds[label.lower()] = build_mix(label, int(row["target_external_rows"]))

    summary = {
        "status": "built",
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "external_csv": str(external_csv),
        "fold": int(args.fold),
        "seed": int(args.seed),
        "sentence_min_source_tokens": int(args.sentence_min_source_tokens),
        "sentence_min_target_words": int(args.sentence_min_target_words),
        "sentence_target_share": float(args.sentence_target_share),
        "internal": {
            "train_visible_rows": int(len(train_visible)),
            "val_visible_rows": int(len(val_visible)),
            "val_parent_rows": int(val_visible["parent_oare_id"].fillna("").astype(str).nunique()),
            "base_chunk_mode_counts_train_visible": {
                key: int(value) for key, value in train_visible["chunk_mode"].fillna("").value_counts().to_dict().items()
            },
        },
        "external_pool": {
            "sentence_rows_after_length_filter": int(len(sentence_rows)),
            "parent_rows_raw": int(len(parent_df)),
            "parent_chunk_rows": int(len(parent_chunks)),
            "parent_chunk_report_json": str(parent_report_json),
            "sentence_pool_row_origin_counts": {
                key: int(value) for key, value in sentence_rows["row_origin"].fillna("").value_counts().to_dict().items()
            },
        },
        "builds": builds,
    }
    write_json(report_dir / "summary.json", summary)
    pd.DataFrame([builds[key] for key in ("internal_only_matched", "e10", "e30", "e50") if key in builds]).to_csv(
        report_dir / "mix_builds.csv",
        index=False,
    )

    lines = [
        "# A1 Mix Build Report",
        "",
        f"- status: `{summary['status']}`",
        f"- base train_visible rows: `{summary['internal']['train_visible_rows']}`",
        f"- val rows fixed: `{summary['internal']['val_visible_rows']}`",
        f"- sentence pool after length filter: `{summary['external_pool']['sentence_rows_after_length_filter']}`",
        f"- parent chunk rows: `{summary['external_pool']['parent_chunk_rows']}`",
        f"- sentence target share: `{summary['sentence_target_share']:.2f}`",
        "",
    ]
    for key in ("internal_only_matched", "e10", "e30", "e50"):
        build = builds.get(key)
        if not build:
            continue
        lines.extend(
            [
                f"## {build['label']}",
                "",
                f"- processed_dir: `{build['processed_dir']}`",
                f"- config_path: `{build['config_path']}`",
                f"- actual external rows: `{build['actual_external_rows']}`",
                f"- actual ratio vs train_visible: `{build['actual_ratio_vs_train_visible']}`",
                f"- sentence rows: `{build['actual_sentence_rows']}`",
                f"- parent chunk rows: `{build['actual_parent_chunk_rows']}`",
                f"- external fold0 rows: `{build['ext_fold0_rows']}`",
                f"- val rows unchanged: `{build['val_rows_unchanged']}`",
                "",
            ]
        )
    write_text(report_dir / "report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
