from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from metrics_utils import compute_translation_metrics


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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _strip_task_prefix(text: str, task_prefix: str) -> str:
    value = _safe_text(text).strip()
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _bucket_name(row: pd.Series) -> str:
    chunk_total = int(row.get("chunk_total", 1) or 1)
    source = _safe_text(row.get("source"))
    if chunk_total >= 7:
        return "chunk7plus"
    if chunk_total >= 4:
        return "chunk4_6"
    if "<gap>" in source or "{" in source or "[" in source:
        return "chunk2_3_tag_rich"
    return "chunk1_3_plain"


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
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_retrieval_20260310")
    args = ap.parse_args()

    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_20260310")
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
    task_prefix = _safe_text(((base_cfg.get("preprocess", {}) or {}).get("task_prefix", ""))).strip()
    if task_prefix and not task_prefix.endswith(" "):
        task_prefix += " "

    train_proc = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds = pd.read_csv(base_processed_dir / "folds.csv")
    merged = train_proc.merge(folds[["oare_id", "fold"]], on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(args.fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(args.fold)].copy().reset_index(drop=True)

    train_visible["source_retrieval"] = train_visible["source"].fillna("").astype(str).map(
        lambda x: _strip_task_prefix(x, task_prefix)
    )
    val_visible["source_retrieval"] = val_visible["source"].fillna("").astype(str).map(
        lambda x: _strip_task_prefix(x, task_prefix)
    )
    train_visible["target_text"] = train_visible["target"].fillna("").astype(str)
    val_visible["target_text"] = val_visible["target"].fillna("").astype(str)
    val_visible["bucket"] = val_visible.apply(_bucket_name, axis=1)

    # Char + word TF-IDF combined via feature concatenation by text doubling.
    train_corpus = (
        train_visible["source_retrieval"].fillna("").astype(str)
        + " ||| "
        + train_visible["source_retrieval"].fillna("").astype(str)
    ).tolist()
    val_corpus = (
        val_visible["source_retrieval"].fillna("").astype(str)
        + " ||| "
        + val_visible["source_retrieval"].fillna("").astype(str)
    ).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_corpus)
    val_matrix = vectorizer.transform(val_corpus)
    sim = linear_kernel(val_matrix, train_matrix)

    top_k = max(1, int(args.top_k))
    nearest_rows: list[dict[str, Any]] = []
    top1_scores: list[float] = []
    top3_scores: list[float] = []
    top1_target_scores: list[float] = []

    train_exact_ids = set(train_visible["oare_id"].astype(str).tolist())
    val_exact_ids = set(val_visible["oare_id"].astype(str).tolist())
    overlap_ids = len(train_exact_ids & val_exact_ids)

    has_gap = val_visible["source_retrieval"].astype(str).str.contains("<gap>", regex=False)
    has_bracket = val_visible["source_retrieval"].astype(str).str.contains(r"[\[\]{}]", regex=True)
    short_aligned_val = _bool_series(val_visible, "is_short_aligned")
    short_aligned_train = _bool_series(train_visible, "is_short_aligned")

    for row_idx in range(val_visible.shape[0]):
        row_scores = sim[row_idx]
        order = np.argsort(-row_scores)[:top_k]
        best_idx = int(order[0])
        best_score = float(row_scores[best_idx])
        top1_scores.append(best_score)
        top3_scores.append(float(np.mean(row_scores[order])))

        val_target = _safe_text(val_visible.iloc[row_idx]["target_text"])
        best_target = _safe_text(train_visible.iloc[best_idx]["target_text"])
        target_metric = compute_translation_metrics(predictions=[best_target], references=[val_target])
        top1_target_scores.append(float(target_metric["chrfpp"]))

        nearest = train_visible.iloc[order]
        nearest_rows.append(
            {
                "oare_id": _safe_text(val_visible.iloc[row_idx]["oare_id"]),
                "parent_oare_id": _safe_text(val_visible.iloc[row_idx].get("parent_oare_id")),
                "bucket": _safe_text(val_visible.iloc[row_idx]["bucket"]),
                "source": _safe_text(val_visible.iloc[row_idx]["source_retrieval"]),
                "target": val_target,
                "top1_neighbor_oare_id": _safe_text(train_visible.iloc[best_idx]["oare_id"]),
                "top1_neighbor_source": _safe_text(train_visible.iloc[best_idx]["source_retrieval"]),
                "top1_neighbor_target": best_target,
                "top1_score": best_score,
                "top3_mean_score": float(np.mean(row_scores[order])),
                "top1_target_chrfpp": float(target_metric["chrfpp"]),
                "topk_neighbor_ids": "|".join(nearest["oare_id"].astype(str).tolist()),
                "topk_scores": "|".join(f"{float(row_scores[int(i)]):.4f}" for i in order),
            }
        )

    nearest_df = pd.DataFrame(nearest_rows)
    nearest_df.to_csv(report_dir / "retrieval_bucket_audit.csv", index=False)

    bucket_summary = (
        nearest_df.groupby("bucket", as_index=False)
        .agg(
            rows=("oare_id", "count"),
            mean_top1_score=("top1_score", "mean"),
            p95_top1_score=("top1_score", lambda s: float(np.percentile(s, 95))),
            mean_top1_target_chrfpp=("top1_target_chrfpp", "mean"),
        )
        .sort_values(["mean_top1_target_chrfpp", "mean_top1_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    bucket_summary.to_csv(report_dir / "retrieval_bucket_summary.csv", index=False)

    datastore_manifest = {
        "status": "ready_internal_only",
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "fold": int(args.fold),
        "datastore_rows": int(train_visible.shape[0]),
        "datastore_unique_oare_id": int(train_visible["oare_id"].astype(str).nunique()),
        "datastore_unique_parent_oare_id": int(train_visible["parent_oare_id"].astype(str).nunique())
        if "parent_oare_id" in train_visible.columns
        else None,
        "datastore_short_aligned_rows": int(short_aligned_train.sum()),
        "val_rows": int(val_visible.shape[0]),
        "val_short_aligned_rows": int(short_aligned_val.sum()),
        "artifacts": {
            "retrieval_bucket_audit_csv": str(report_dir / "retrieval_bucket_audit.csv"),
            "retrieval_bucket_summary_csv": str(report_dir / "retrieval_bucket_summary.csv"),
        },
    }
    _write_json(report_dir / "datastore_manifest.json", datastore_manifest)

    overlap_audit = {
        "status": "ready_internal_only",
        "train_val_exact_id_overlap": int(overlap_ids),
        "train_val_exact_source_overlap_rows": int(
            val_visible["source_retrieval"].astype(str).isin(set(train_visible["source_retrieval"].astype(str).tolist())).sum()
        ),
        "val_has_gap_rows": int(has_gap.sum()),
        "val_has_bracket_rows": int(has_bracket.sum()),
        "checks": {
            "datastore_from_train_visible_only": True,
            "fold0_val_not_in_exact_ids": overlap_ids == 0,
        },
    }
    _write_json(report_dir / "retrieval_overlap_audit.json", overlap_audit)

    summary = {
        "status": "ready_internal_only",
        "fold": int(args.fold),
        "rows": {
            "train_visible": int(train_visible.shape[0]),
            "val_visible": int(val_visible.shape[0]),
        },
        "scores": {
            "top1_score_mean": float(np.mean(top1_scores)) if top1_scores else 0.0,
            "top1_score_p50": float(np.percentile(top1_scores, 50)) if top1_scores else 0.0,
            "top1_score_p95": float(np.percentile(top1_scores, 95)) if top1_scores else 0.0,
            "top3_score_mean": float(np.mean(top3_scores)) if top3_scores else 0.0,
            "top1_target_chrfpp_mean": float(np.mean(top1_target_scores)) if top1_target_scores else 0.0,
        },
        "targeted_slices": {
            "gap_rows": {
                "rows": int(has_gap.sum()),
                "top1_score_mean": float(nearest_df.loc[has_gap.values, "top1_score"].mean()) if int(has_gap.sum()) > 0 else 0.0,
                "top1_target_chrfpp_mean": float(nearest_df.loc[has_gap.values, "top1_target_chrfpp"].mean())
                if int(has_gap.sum()) > 0
                else 0.0,
            },
            "bracket_rows": {
                "rows": int(has_bracket.sum()),
                "top1_score_mean": float(nearest_df.loc[has_bracket.values, "top1_score"].mean())
                if int(has_bracket.sum()) > 0
                else 0.0,
                "top1_target_chrfpp_mean": float(nearest_df.loc[has_bracket.values, "top1_target_chrfpp"].mean())
                if int(has_bracket.sum()) > 0
                else 0.0,
            },
            "short_aligned_val_rows": {
                "rows": int(short_aligned_val.sum()),
                "top1_score_mean": float(nearest_df.loc[short_aligned_val.values, "top1_score"].mean())
                if int(short_aligned_val.sum()) > 0
                else 0.0,
                "top1_target_chrfpp_mean": float(nearest_df.loc[short_aligned_val.values, "top1_target_chrfpp"].mean())
                if int(short_aligned_val.sum()) > 0
                else 0.0,
            },
        },
        "artifacts": {
            "datastore_manifest_json": str(report_dir / "datastore_manifest.json"),
            "retrieval_overlap_audit_json": str(report_dir / "retrieval_overlap_audit.json"),
            "retrieval_bucket_audit_csv": str(report_dir / "retrieval_bucket_audit.csv"),
            "retrieval_bucket_summary_csv": str(report_dir / "retrieval_bucket_summary.csv"),
        },
    }
    _write_json(report_dir / "summary.json", summary)

    print(f"OK: wrote {report_dir / 'datastore_manifest.json'}")
    print(f"OK: wrote {report_dir / 'retrieval_overlap_audit.json'}")
    print(f"OK: wrote {report_dir / 'retrieval_bucket_audit.csv'}")
    print(f"OK: wrote {report_dir / 'retrieval_bucket_summary.csv'}")
    print(f"OK: wrote {report_dir / 'summary.json'}")
    print(f"INFO: top1_score_mean={summary['scores']['top1_score_mean']:.4f}")


if __name__ == "__main__":
    main()
