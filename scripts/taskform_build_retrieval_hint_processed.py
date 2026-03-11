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


def _task_prefix(cfg: dict[str, Any]) -> str:
    prefix = _safe_text(((cfg.get("preprocess", {}) or {}).get("task_prefix", ""))).strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "
    return prefix


def _strip_task_prefix(text: str, task_prefix: str) -> str:
    value = _safe_text(text).strip()
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value


def _truncate(text: str, max_chars: int) -> str:
    value = _safe_text(text).strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 1)].rstrip() + "…"


def _compact_neighbor_block(
    *,
    neighbor_sources: list[str],
    neighbor_targets: list[str],
    scores: list[float],
    max_source_chars: int,
    max_target_chars: int,
    score_decimals: int,
) -> str:
    rows: list[str] = []
    for rank, (source_text, target_text, score) in enumerate(zip(neighbor_sources, neighbor_targets, scores), start=1):
        source_piece = _truncate(source_text, max_source_chars)
        target_piece = _truncate(target_text, max_target_chars)
        segments: list[str] = []
        if source_piece:
            segments.append(source_piece)
        if target_piece:
            segments.append(f"=> {target_piece}")
        if not segments:
            continue
        rows.append(f"[{rank}|{score:.{score_decimals}f}] " + " ".join(segments))
    return "\n".join(rows).strip()


def _build_augmented_source(
    *,
    base_source: str,
    hint_source: str,
    hint_target: str,
    neighbor_sources: list[str],
    neighbor_targets: list[str],
    scores: list[float],
    hint_format: str,
    hint_source_max_chars: int,
    hint_target_max_chars: int,
    compact_score_decimals: int,
) -> str:
    if hint_format == "compact_triplets":
        compact_block = _compact_neighbor_block(
            neighbor_sources=neighbor_sources,
            neighbor_targets=neighbor_targets,
            scores=scores,
            max_source_chars=hint_source_max_chars,
            max_target_chars=hint_target_max_chars,
            score_decimals=compact_score_decimals,
        )
        if compact_block:
            return f"{base_source}\n\nRetrieved neighbors:\n{compact_block}".strip()
        return base_source
    if hint_source or hint_target:
        return (
            f"{base_source}\n\n"
            f"Retrieved source: {hint_source}\n"
            f"Retrieved English hint: {hint_target}"
        ).strip()
    return base_source


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
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--hint-source-max-chars", type=int, default=160)
    ap.add_argument("--hint-target-max-chars", type=int, default=220)
    ap.add_argument("--hint-format", choices=["split_fields", "compact_triplets"], default="split_fields")
    ap.add_argument("--compact-score-decimals", type=int, default=2)
    ap.add_argument("--output-dir", default="data/processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_retrieval_top1_build_20260310")
    args = ap.parse_args()

    base_cfg_path = _resolve_path(
        args.base_config,
        REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
    )
    base_cfg = _load_yaml(base_cfg_path)
    task_prefix = _task_prefix(base_cfg)
    base_processed_dir = _resolve_path(
        args.base_processed_dir,
        _resolve_path((base_cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed"),
    )
    output_dir = _resolve_path(args.output_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_retrieval_top1_fold0")
    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_top1_build_20260310")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_proc = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds = pd.read_csv(base_processed_dir / "folds.csv")
    merged = train_proc.merge(folds[["oare_id", "fold"]], on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(args.fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(args.fold)].copy().reset_index(drop=True)

    train_visible["source_plain"] = train_visible["source"].fillna("").astype(str).map(lambda x: _strip_task_prefix(x, task_prefix))
    val_visible["source_plain"] = val_visible["source"].fillna("").astype(str).map(lambda x: _strip_task_prefix(x, task_prefix))
    train_visible["target_plain"] = train_visible["target"].fillna("").astype(str)
    val_visible["target_plain"] = val_visible["target"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_visible["source_plain"].fillna("").astype(str).tolist())
    val_matrix = vectorizer.transform(val_visible["source_plain"].fillna("").astype(str).tolist())

    train_train_sim = linear_kernel(train_matrix, train_matrix)
    val_train_sim = linear_kernel(val_matrix, train_matrix)
    top_k = max(1, int(args.top_k))

    train_hints: dict[str, dict[str, Any]] = {}
    for row_idx in range(train_visible.shape[0]):
        scores = train_train_sim[row_idx]
        order = np.argsort(-scores)
        selected: list[int] = []
        for idx in order:
            if int(idx) == int(row_idx):
                continue
            selected.append(int(idx))
            if len(selected) >= top_k:
                break
        if not selected:
            continue
        neighbors = train_visible.iloc[selected]
        train_hints[_safe_text(train_visible.iloc[row_idx]["oare_id"])] = {
            "neighbor_ids": neighbors["oare_id"].astype(str).tolist(),
            "neighbor_sources": neighbors["source_plain"].astype(str).tolist(),
            "neighbor_targets": neighbors["target_plain"].astype(str).tolist(),
            "scores": [float(scores[i]) for i in selected],
        }

    val_hints: dict[str, dict[str, Any]] = {}
    for row_idx in range(val_visible.shape[0]):
        scores = val_train_sim[row_idx]
        order = np.argsort(-scores)[:top_k]
        neighbors = train_visible.iloc[order]
        val_hints[_safe_text(val_visible.iloc[row_idx]["oare_id"])] = {
            "neighbor_ids": neighbors["oare_id"].astype(str).tolist(),
            "neighbor_sources": neighbors["source_plain"].astype(str).tolist(),
            "neighbor_targets": neighbors["target_plain"].astype(str).tolist(),
            "scores": [float(scores[int(i)]) for i in order],
        }

    full_hints: dict[str, dict[str, Any]] = {}
    full_hints.update(train_hints)
    full_hints.update(val_hints)

    out = train_proc.copy()
    retrieval_neighbor_ids: list[str] = []
    retrieval_scores: list[str] = []
    retrieval_hint_targets: list[str] = []
    retrieval_hint_sources: list[str] = []
    retrieval_modes: list[str] = []
    augmented_sources: list[str] = []
    fold_map = folds.set_index("oare_id")["fold"].to_dict()

    for _, row in out.iterrows():
        oare_id = _safe_text(row.get("oare_id"))
        hint = full_hints.get(oare_id, {})
        neighbor_ids = [str(x) for x in hint.get("neighbor_ids", [])]
        neighbor_sources = [str(x) for x in hint.get("neighbor_sources", [])]
        neighbor_targets = [str(x) for x in hint.get("neighbor_targets", [])]
        scores = [float(x) for x in hint.get("scores", [])]

        mode = "val_to_train" if int(fold_map.get(oare_id, -1)) == int(args.fold) else "train_to_train"
        hint_source = " || ".join(_truncate(x, int(args.hint_source_max_chars)) for x in neighbor_sources if x)
        hint_target = " || ".join(_truncate(x, int(args.hint_target_max_chars)) for x in neighbor_targets if x)
        base_source = _safe_text(row.get("source"))
        augmented = _build_augmented_source(
            base_source=base_source,
            hint_source=hint_source,
            hint_target=hint_target,
            neighbor_sources=neighbor_sources,
            neighbor_targets=neighbor_targets,
            scores=scores,
            hint_format=str(args.hint_format),
            hint_source_max_chars=int(args.hint_source_max_chars),
            hint_target_max_chars=int(args.hint_target_max_chars),
            compact_score_decimals=max(0, int(args.compact_score_decimals)),
        )

        augmented_sources.append(augmented)
        retrieval_neighbor_ids.append("|".join(neighbor_ids))
        retrieval_scores.append("|".join(f"{x:.4f}" for x in scores))
        retrieval_hint_targets.append(hint_target)
        retrieval_hint_sources.append(hint_source)
        retrieval_modes.append(mode)

    out["source"] = augmented_sources
    out["retrieval_neighbor_ids"] = retrieval_neighbor_ids
    out["retrieval_scores"] = retrieval_scores
    out["retrieval_hint_source"] = retrieval_hint_sources
    out["retrieval_hint_target"] = retrieval_hint_targets
    out["retrieval_mode"] = retrieval_modes

    out.to_csv(output_dir / "train_proc.csv", index=False)
    folds.to_csv(output_dir / "folds.csv", index=False)

    sample_audit = out[["oare_id", "source", "retrieval_neighbor_ids", "retrieval_scores", "retrieval_mode"]].copy()
    sample_audit.to_csv(report_dir / "retrieval_augmented_samples.csv", index=False)

    summary = {
        "status": "ready",
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "output_dir": str(output_dir),
        "fold": int(args.fold),
        "top_k": int(top_k),
        "hint_format": str(args.hint_format),
        "rows": {
            "total": int(out.shape[0]),
            "train_visible": int(train_visible.shape[0]),
            "val_visible": int(val_visible.shape[0]),
        },
        "hints": {
            "train_rows_with_hint": int(sum(bool(train_hints.get(x)) for x in train_visible["oare_id"].astype(str).tolist())),
            "val_rows_with_hint": int(sum(bool(val_hints.get(x)) for x in val_visible["oare_id"].astype(str).tolist())),
        },
        "artifacts": {
            "train_proc_csv": str(output_dir / "train_proc.csv"),
            "folds_csv": str(output_dir / "folds.csv"),
            "retrieval_augmented_samples_csv": str(report_dir / "retrieval_augmented_samples.csv"),
        },
    }
    _write_json(report_dir / "summary.json", summary)

    print(f"OK: wrote {output_dir / 'train_proc.csv'}")
    print(f"OK: wrote {output_dir / 'folds.csv'}")
    print(f"OK: wrote {report_dir / 'retrieval_augmented_samples.csv'}")
    print(f"OK: wrote {report_dir / 'summary.json'}")
    print(f"INFO: train_rows_with_hint={summary['hints']['train_rows_with_hint']}, val_rows_with_hint={summary['hints']['val_rows_with_hint']}")


if __name__ == "__main__":
    main()
