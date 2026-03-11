from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import LogitsProcessor, LogitsProcessorList


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def task_prefix_from_config(cfg: dict[str, Any]) -> str:
    prefix = safe_text(((cfg.get("preprocess", {}) or {}).get("task_prefix", ""))).strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "
    return prefix


def strip_task_prefix(text: str, task_prefix: str) -> str:
    value = safe_text(text).strip()
    if task_prefix and value.startswith(task_prefix):
        return value[len(task_prefix) :].strip()
    return value


def batch_token_ids(tokenizer, texts: list[str], batch_size: int = 128) -> list[list[int]]:
    out: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        out.extend([list(map(int, ids)) for ids in encoded["input_ids"]])
    return out


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    arr = np.clip(np.asarray(scores, dtype=float), a_min=0.0, a_max=None)
    total = float(arr.sum())
    if total <= 1e-12:
        return [1.0 / float(len(scores)) for _ in scores]
    return [float(x / total) for x in arr]


@dataclass
class QueryNeighbors:
    neighbor_ids: list[str]
    neighbor_scores: list[float]
    neighbor_weights: list[float]
    neighbor_targets: list[str]
    neighbor_token_ids: list[list[int]]


class RetrievalBiasLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        query_neighbors: list[QueryNeighbors],
        num_beams: int,
        bias_strength: float,
        max_bias_steps: int,
        eos_token_id: int | None,
    ) -> None:
        self.query_neighbors = query_neighbors
        self.num_beams = max(1, int(num_beams))
        self.bias_strength = float(bias_strength)
        self.max_bias_steps = int(max_bias_steps)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        if self.bias_strength <= 0.0:
            return scores
        step_idx = int(input_ids.shape[1] - 1)
        if step_idx < 0 or step_idx >= self.max_bias_steps:
            return scores
        for row_idx in range(scores.shape[0]):
            query_idx = int(row_idx // self.num_beams)
            if query_idx >= len(self.query_neighbors):
                continue
            query = self.query_neighbors[query_idx]
            votes: dict[int, float] = {}
            for weight, token_ids in zip(query.neighbor_weights, query.neighbor_token_ids):
                if step_idx < len(token_ids):
                    token_id = int(token_ids[step_idx])
                    votes[token_id] = votes.get(token_id, 0.0) + float(weight)
                elif self.eos_token_id is not None and step_idx == len(token_ids):
                    eos_id = int(self.eos_token_id)
                    votes[eos_id] = votes.get(eos_id, 0.0) + float(weight)
            if not votes:
                continue
            for token_id, weight in votes.items():
                if 0 <= int(token_id) < scores.shape[1]:
                    scores[row_idx, int(token_id)] = scores[row_idx, int(token_id)] + (
                        self.bias_strength * float(weight)
                    )
        return scores


def load_train_visible_frame(*, processed_dir: Path, fold: int, task_prefix: str) -> pd.DataFrame:
    train_proc = pd.read_csv(processed_dir / "train_proc.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")[["oare_id", "fold"]]
    merged = train_proc.merge(folds, on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(fold)].copy().reset_index(drop=True)
    train_visible["source_plain"] = (
        train_visible["source"].fillna("").astype(str).map(lambda text: strip_task_prefix(text, task_prefix))
    )
    train_visible["target_plain"] = train_visible["target"].fillna("").astype(str)
    return train_visible


def _dedup_neighbors(
    *,
    train_visible: pd.DataFrame,
    order: np.ndarray,
    score_row: np.ndarray,
    target_token_ids: list[list[int]],
    query_target_plain: str,
    raw_pool_k: int,
    final_k: int,
) -> QueryNeighbors:
    seen_targets: dict[str, int] = {}
    neighbor_ids: list[str] = []
    neighbor_scores: list[float] = []
    neighbor_targets: list[str] = []
    neighbor_token_rows: list[list[int]] = []

    for idx in order[: max(int(raw_pool_k), int(final_k))]:
        row = train_visible.iloc[int(idx)]
        target_text = safe_text(row["target_plain"])
        if not target_text or target_text == query_target_plain:
            continue
        if target_text in seen_targets:
            pos = seen_targets[target_text]
            neighbor_scores[pos] += float(score_row[int(idx)])
            continue
        seen_targets[target_text] = len(neighbor_ids)
        neighbor_ids.append(safe_text(row["oare_id"]))
        neighbor_scores.append(float(score_row[int(idx)]))
        neighbor_targets.append(target_text)
        neighbor_token_rows.append(target_token_ids[int(idx)])
        if len(neighbor_ids) >= int(final_k):
            break

    weights = normalize_scores(neighbor_scores)
    return QueryNeighbors(
        neighbor_ids=neighbor_ids,
        neighbor_scores=[round(float(x), 6) for x in neighbor_scores],
        neighbor_weights=[round(float(x), 6) for x in weights],
        neighbor_targets=neighbor_targets,
        neighbor_token_ids=neighbor_token_rows,
    )


def build_query_neighbors_for_frame(
    *,
    train_visible: pd.DataFrame,
    query_df: pd.DataFrame,
    tokenizer,
    task_prefix: str,
    raw_pool_k: int,
    final_k: int,
) -> tuple[list[QueryNeighbors], dict[str, Any], pd.DataFrame]:
    if query_df.empty:
        return [], {"status": "empty_query"}, pd.DataFrame()

    query_work = query_df.copy().reset_index(drop=True)
    query_work["source_plain"] = query_work["source"].fillna("").astype(str).map(
        lambda text: strip_task_prefix(text, task_prefix)
    )
    query_work["target_plain"] = query_work["target"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    index_start = time.perf_counter()
    train_matrix = vectorizer.fit_transform(train_visible["source_plain"].fillna("").astype(str).tolist())
    query_matrix = vectorizer.transform(query_work["source_plain"].fillna("").astype(str).tolist())
    sim = linear_kernel(query_matrix, train_matrix)
    index_elapsed = time.perf_counter() - index_start

    tokenize_start = time.perf_counter()
    train_target_token_ids = batch_token_ids(tokenizer, train_visible["target_plain"].fillna("").astype(str).tolist())
    tokenize_elapsed = time.perf_counter() - tokenize_start

    query_neighbors: list[QueryNeighbors] = []
    query_rows: list[dict[str, Any]] = []
    neighbor_counts: list[int] = []
    all_neighbor_scores: list[float] = []

    for row_idx in range(query_work.shape[0]):
        score_row = sim[row_idx]
        order = np.argsort(-score_row)
        query = query_work.iloc[row_idx]
        deduped = _dedup_neighbors(
            train_visible=train_visible,
            order=order,
            score_row=score_row,
            target_token_ids=train_target_token_ids,
            query_target_plain=safe_text(query["target_plain"]),
            raw_pool_k=int(raw_pool_k),
            final_k=int(final_k),
        )
        query_neighbors.append(deduped)
        neighbor_counts.append(int(len(deduped.neighbor_ids)))
        all_neighbor_scores.extend([float(x) for x in deduped.neighbor_scores])
        query_rows.append(
            {
                "oare_id": safe_text(query.get("oare_id")),
                "parent_oare_id": safe_text(query.get("parent_oare_id")),
                "neighbor_count": int(len(deduped.neighbor_ids)),
                "neighbor_ids": "|".join(deduped.neighbor_ids),
                "neighbor_scores": "|".join(f"{float(x):.4f}" for x in deduped.neighbor_scores),
                "neighbor_weights": "|".join(f"{float(x):.4f}" for x in deduped.neighbor_weights),
            }
        )

    metadata = {
        "status": "ready",
        "train_visible_rows": int(train_visible.shape[0]),
        "query_rows": int(query_work.shape[0]),
        "index_build_seconds": round(float(index_elapsed), 4),
        "target_tokenize_seconds": round(float(tokenize_elapsed), 4),
        "neighbor_count_mean": round(float(np.mean(np.asarray(neighbor_counts, dtype=float))), 4)
        if neighbor_counts
        else 0.0,
        "neighbor_count_p50": round(float(np.percentile(np.asarray(neighbor_counts, dtype=float), 50)), 4)
        if neighbor_counts
        else 0.0,
        "neighbor_count_p95": round(float(np.percentile(np.asarray(neighbor_counts, dtype=float), 95)), 4)
        if neighbor_counts
        else 0.0,
        "neighbor_score_mean": round(float(np.mean(np.asarray(all_neighbor_scores, dtype=float))), 4)
        if all_neighbor_scores
        else 0.0,
    }
    return query_neighbors, metadata, pd.DataFrame(query_rows)


def build_batch_logits_processor(
    *,
    query_neighbors: list[QueryNeighbors] | None,
    batch_start_idx: int,
    batch_size: int,
    num_beams: int,
    bias_strength: float,
    max_bias_steps: int,
    eos_token_id: int | None,
):
    if not query_neighbors or bias_strength <= 0.0:
        return None
    batch_queries = query_neighbors[batch_start_idx : batch_start_idx + batch_size]
    if not batch_queries:
        return None
    return LogitsProcessorList(
        [
            RetrievalBiasLogitsProcessor(
                query_neighbors=batch_queries,
                num_beams=int(num_beams),
                bias_strength=float(bias_strength),
                max_bias_steps=int(max_bias_steps),
                eos_token_id=eos_token_id,
            )
        ]
    )
