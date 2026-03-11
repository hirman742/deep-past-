from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoConfig, AutoTokenizer

from metrics_utils import compute_translation_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _batch_token_lengths(tokenizer: AutoTokenizer, texts: list[str], batch_size: int = 128) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False, return_length=True)
        lengths.extend(int(x) for x in encoded["length"])
    return lengths


def _batch_token_ids(tokenizer: AutoTokenizer, texts: list[str], batch_size: int = 128) -> list[list[int]]:
    out: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        out.extend([list(map(int, ids)) for ids in encoded["input_ids"]])
    return out


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    clipped = np.clip(np.asarray(scores, dtype=float), a_min=0.0, a_max=None)
    total = float(clipped.sum())
    if total <= 1e-12:
        return [1.0 / float(len(scores)) for _ in scores]
    return [float(x / total) for x in clipped]


def _top_vote_candidates(
    *,
    neighbor_token_ids: list[list[int]],
    weights: list[float],
    gold_token_ids: list[int],
    max_steps: int,
    top_n: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    preview: list[dict[str, Any]] = []
    hits_at_1 = 0
    hits_at_5 = 0
    total_steps = 0
    limit = min(int(max_steps), len(gold_token_ids))
    for step in range(limit):
        votes: dict[int, float] = {}
        for weight, token_ids in zip(weights, neighbor_token_ids):
            if step >= len(token_ids):
                continue
            token_id = int(token_ids[step])
            votes[token_id] = votes.get(token_id, 0.0) + float(weight)
        if not votes:
            continue
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        gold_token_id = int(gold_token_ids[step])
        top_ids = [token_id for token_id, _ in ranked[: max(5, top_n)]]
        total_steps += 1
        if top_ids and gold_token_id == int(top_ids[0]):
            hits_at_1 += 1
        if gold_token_id in top_ids[:5]:
            hits_at_5 += 1
        preview.append(
            {
                "step": int(step),
                "gold_token_id": gold_token_id,
                "top_token_votes": [
                    {"token_id": int(token_id), "weight": round(float(weight), 6)}
                    for token_id, weight in ranked[:top_n]
                ],
            }
        )
    if total_steps == 0:
        return 0.0, 0.0, preview
    return float(hits_at_1) / float(total_steps), float(hits_at_5) / float(total_steps), preview


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _load_fold_split(base_cfg: dict[str, Any], base_processed_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_prefix = _task_prefix(base_cfg)
    train_proc = pd.read_csv(base_processed_dir / "train_proc.csv")
    folds = pd.read_csv(base_processed_dir / "folds.csv")
    merged = train_proc.merge(folds[["oare_id", "fold"]], on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(fold)].copy().reset_index(drop=True)
    for frame in (train_visible, val_visible):
        frame["source_plain"] = frame["source"].fillna("").astype(str).map(lambda x: _strip_task_prefix(x, task_prefix))
        frame["target_plain"] = frame["target"].fillna("").astype(str)
    return train_visible, val_visible


def _build_r3_processed(
    *,
    base_cfg_path: Path,
    base_cfg: dict[str, Any],
    base_processed_dir: Path,
    fold: int,
    output_dir: Path,
    report_dir: Path,
    config_out_path: Path,
    top_k: int,
    hint_source_max_chars: int,
    hint_target_max_chars: int,
) -> dict[str, Any]:
    build_report_dir = report_dir / "r3_build"
    build_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "taskform_build_retrieval_hint_processed.py"),
        "--base-config",
        str(base_cfg_path),
        "--base-processed-dir",
        str(base_processed_dir),
        "--fold",
        str(fold),
        "--top-k",
        str(top_k),
        "--hint-format",
        "compact_triplets",
        "--hint-source-max-chars",
        str(hint_source_max_chars),
        "--hint-target-max-chars",
        str(hint_target_max_chars),
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(build_report_dir),
    ]
    _run(build_cmd)

    cfg = json.loads(json.dumps(base_cfg))
    cfg["name"] = "taskform_winner_a2_retrieval_top3_comp"
    cfg.setdefault("paths", {})["processed_dir"] = _repo_relative(output_dir)
    cfg["paths"]["run_dir"] = "runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP3_COMP_20260310"
    config_out_path.parent.mkdir(parents=True, exist_ok=True)
    config_out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str((base_cfg.get("model", {}) or {}).get("name", "google/byt5-small")))
    base_train = pd.read_csv(base_processed_dir / "train_proc.csv")
    r3_train = pd.read_csv(output_dir / "train_proc.csv")
    if len(base_train) != len(r3_train):
        raise ValueError("R3 build row count mismatch against base processed data")

    base_sources = base_train["source"].fillna("").astype(str).tolist()
    r3_sources = r3_train["source"].fillna("").astype(str).tolist()
    base_chars = [len(text) for text in base_sources]
    r3_chars = [len(text) for text in r3_sources]
    delta_chars = [float(b - a) for a, b in zip(base_chars, r3_chars)]

    print("INFO: auditing R3 source length overhead", flush=True)
    token_start = time.perf_counter()
    base_token_lengths = _batch_token_lengths(tokenizer, base_sources)
    r3_token_lengths = _batch_token_lengths(tokenizer, r3_sources)
    token_elapsed = time.perf_counter() - token_start

    delta_tokens = [float(b - a) for a, b in zip(base_token_lengths, r3_token_lengths)]
    model_max_source_length = int((base_cfg.get("model", {}) or {}).get("max_source_length", 640))

    audit_df = pd.DataFrame(
        {
            "oare_id": r3_train["oare_id"].fillna("").astype(str),
            "base_source_chars": base_chars,
            "r3_source_chars": r3_chars,
            "delta_source_chars": delta_chars,
            "base_source_tokens": base_token_lengths,
            "r3_source_tokens": r3_token_lengths,
            "delta_source_tokens": delta_tokens,
            "retrieval_neighbor_count": r3_train["retrieval_neighbor_ids"]
            .fillna("")
            .astype(str)
            .map(lambda text: 0 if not text else len([item for item in text.split("|") if item])),
            "retrieval_hint_char_len": r3_train["retrieval_hint_target"].fillna("").astype(str).str.len(),
        }
    )
    audit_df.to_csv(report_dir / "r3_length_audit.csv", index=False)
    sample_cols = [
        "oare_id",
        "source",
        "retrieval_neighbor_ids",
        "retrieval_scores",
        "retrieval_hint_source",
        "retrieval_hint_target",
    ]
    r3_train[sample_cols].head(32).to_csv(report_dir / "r3_sample_rows.csv", index=False)

    summary = {
        "status": "ready_top3_compact",
        "top_k": int(top_k),
        "hint_format": "compact_triplets",
        "build_report_dir": str(build_report_dir),
        "processed_dir": str(output_dir),
        "config_path": str(config_out_path),
        "rows": int(len(r3_train)),
        "length_overhead": {
            "delta_chars_mean": round(_mean(delta_chars), 4),
            "delta_chars_p95": round(_pctl(delta_chars, 95), 4),
            "delta_tokens_mean": round(_mean(delta_tokens), 4),
            "delta_tokens_p95": round(_pctl(delta_tokens, 95), 4),
            "base_tokens_p95": round(_pctl([float(x) for x in base_token_lengths], 95), 4),
            "r3_tokens_p95": round(_pctl([float(x) for x in r3_token_lengths], 95), 4),
        },
        "overflow_risk": {
            "model_max_source_length": int(model_max_source_length),
            "base_rows_over_limit": int(sum(int(x) > model_max_source_length for x in base_token_lengths)),
            "r3_rows_over_limit": int(sum(int(x) > model_max_source_length for x in r3_token_lengths)),
        },
        "runtime_seconds": {
            "token_length_audit": round(float(token_elapsed), 4),
        },
        "artifacts": {
            "processed_train_csv": str(output_dir / "train_proc.csv"),
            "processed_folds_csv": str(output_dir / "folds.csv"),
            "config_yaml": str(config_out_path),
            "length_audit_csv": str(report_dir / "r3_length_audit.csv"),
            "sample_rows_csv": str(report_dir / "r3_sample_rows.csv"),
        },
    }
    _write_json(report_dir / "r3_summary.json", summary)
    print("INFO: R3 summary ready", flush=True)
    return summary


def _build_rk_dry_run(
    *,
    base_cfg: dict[str, Any],
    train_visible: pd.DataFrame,
    val_visible: pd.DataFrame,
    report_dir: Path,
    max_anchor_queries: int,
    rk_k: int,
    max_step_probe: int,
    preview_queries: int,
    preview_steps: int,
) -> dict[str, Any]:
    model_name = str((base_cfg.get("model", {}) or {}).get("name", "google/byt5-small"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cfg = AutoConfig.from_pretrained(model_name)
    hidden_size = int(getattr(model_cfg, "d_model", getattr(model_cfg, "hidden_size", 0)) or 0)

    anchor_visible = val_visible.iloc[: max(1, int(max_anchor_queries))].copy().reset_index(drop=True)

    print(
        f"INFO: building RK tfidf index train_rows={train_visible.shape[0]} anchor_rows={anchor_visible.shape[0]} rk_k={rk_k}",
        flush=True,
    )
    index_start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_visible["source_plain"].fillna("").astype(str).tolist())
    index_elapsed = time.perf_counter() - index_start

    query_start = time.perf_counter()
    anchor_matrix = vectorizer.transform(anchor_visible["source_plain"].fillna("").astype(str).tolist())
    sim = linear_kernel(anchor_matrix, train_matrix)
    query_elapsed = time.perf_counter() - query_start

    print("INFO: tokenizing RK datastore targets", flush=True)
    train_token_start = time.perf_counter()
    train_target_token_ids = _batch_token_ids(tokenizer, train_visible["target_plain"].fillna("").astype(str).tolist())
    train_target_token_elapsed = time.perf_counter() - train_token_start
    anchor_gold_token_ids = _batch_token_ids(tokenizer, anchor_visible["target_plain"].fillna("").astype(str).tolist())

    train_target_token_lens = [len(ids) for ids in train_target_token_ids]
    total_train_target_tokens = int(sum(train_target_token_lens))
    fp16_datastore_gib = 0.0
    if hidden_size > 0 and total_train_target_tokens > 0:
        fp16_datastore_gib = float(total_train_target_tokens * hidden_size * 2) / float(1024**3)
    value_store_gib = float(total_train_target_tokens * 4) / float(1024**3)

    query_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    retrieval_latencies_ms: list[float] = []
    assembly_latencies_ms: list[float] = []
    top1_scores: list[float] = []
    mean_topk_scores: list[float] = []
    top1_target_chrfpp: list[float] = []
    oracle_topk_target_chrfpp: list[float] = []
    oracle_recall_at1: list[float] = []
    oracle_recall_at5: list[float] = []
    neighbor_target_lens_mean: list[float] = []

    per_query_search_ms = 1000.0 * float(query_elapsed) / float(max(1, anchor_visible.shape[0]))

    print("INFO: assembling RK query cache previews", flush=True)
    for row_idx in range(anchor_visible.shape[0]):
        score_row = sim[row_idx]
        retrieve_start = time.perf_counter()
        order = np.argsort(-score_row)[: max(1, int(rk_k))]
        retrieval_latencies_ms.append(per_query_search_ms + (1000.0 * (time.perf_counter() - retrieve_start)))

        query_row = anchor_visible.iloc[row_idx]
        gold_target = _safe_text(query_row["target_plain"])
        gold_token_ids = anchor_gold_token_ids[row_idx]
        neighbor_scores = [float(score_row[int(idx)]) for idx in order]
        normalized_scores = _normalize_scores(neighbor_scores)
        top1_scores.append(float(neighbor_scores[0]) if neighbor_scores else 0.0)
        mean_topk_scores.append(float(np.mean(neighbor_scores)) if neighbor_scores else 0.0)

        assembly_start = time.perf_counter()
        neighbor_indices = [int(idx) for idx in order]
        neighbor_rows = train_visible.iloc[neighbor_indices]
        neighbor_ids = neighbor_rows["oare_id"].fillna("").astype(str).tolist()
        neighbor_targets = neighbor_rows["target_plain"].fillna("").astype(str).tolist()
        neighbor_sources = neighbor_rows["source_plain"].fillna("").astype(str).tolist()
        neighbor_token_ids = [train_target_token_ids[int(idx)] for idx in neighbor_indices]
        neighbor_target_lens = [len(ids) for ids in neighbor_token_ids]
        neighbor_target_lens_mean.append(_mean([float(x) for x in neighbor_target_lens]))

        top1_metric = compute_translation_metrics(
            predictions=[neighbor_targets[0] if neighbor_targets else ""],
            references=[gold_target],
        )
        top1_target_chrfpp.append(float(top1_metric["chrfpp"]))

        oracle_scores = [
            float(compute_translation_metrics(predictions=[candidate], references=[gold_target])["chrfpp"])
            for candidate in neighbor_targets
        ]
        oracle_topk_target_chrfpp.append(max(oracle_scores) if oracle_scores else 0.0)

        recall_at_1, recall_at_5, vote_preview = _top_vote_candidates(
            neighbor_token_ids=neighbor_token_ids,
            weights=normalized_scores,
            gold_token_ids=gold_token_ids,
            max_steps=max_step_probe,
            top_n=5,
        )
        oracle_recall_at1.append(recall_at_1)
        oracle_recall_at5.append(recall_at_5)
        assembly_latencies_ms.append(1000.0 * (time.perf_counter() - assembly_start))

        row_summary = {
            "oare_id": _safe_text(query_row["oare_id"]),
            "parent_oare_id": _safe_text(query_row.get("parent_oare_id")),
            "top1_score": round(float(neighbor_scores[0]) if neighbor_scores else 0.0, 6),
            "topk_score_mean": round(float(np.mean(neighbor_scores)) if neighbor_scores else 0.0, 6),
            "top1_target_chrfpp": round(float(top1_metric["chrfpp"]), 4),
            "oracle_topk_target_chrfpp": round(max(oracle_scores) if oracle_scores else 0.0, 4),
            "oracle_next_token_recall_at1": round(float(recall_at_1), 4),
            "oracle_next_token_recall_at5": round(float(recall_at_5), 4),
            "neighbor_target_len_mean": round(_mean([float(x) for x in neighbor_target_lens]), 4),
            "retrieval_latency_ms": round(retrieval_latencies_ms[-1], 4),
            "cache_assembly_latency_ms": round(assembly_latencies_ms[-1], 4),
            "neighbor_ids": "|".join(neighbor_ids),
            "neighbor_scores": "|".join(f"{float(score):.4f}" for score in neighbor_scores),
        }
        query_rows.append(row_summary)

        if len(preview_rows) < int(preview_queries):
            preview_rows.append(
                {
                    "query_oare_id": _safe_text(query_row["oare_id"]),
                    "query_source": _safe_text(query_row["source_plain"]),
                    "query_target": gold_target,
                    "neighbors": [
                        {
                            "rank": int(rank + 1),
                            "oare_id": neighbor_ids[rank],
                            "score": round(float(neighbor_scores[rank]), 6),
                            "weight": round(float(normalized_scores[rank]), 6),
                            "source": neighbor_sources[rank],
                            "target": neighbor_targets[rank],
                            "target_token_len": int(neighbor_target_lens[rank]),
                        }
                        for rank in range(min(len(neighbor_ids), int(rk_k)))
                    ],
                    "step_vote_preview": vote_preview[: int(preview_steps)],
                }
            )
        if (row_idx + 1) % 16 == 0 or (row_idx + 1) == anchor_visible.shape[0]:
            print(
                f"INFO: RK query progress {row_idx + 1}/{anchor_visible.shape[0]}",
                flush=True,
            )

    query_df = pd.DataFrame(query_rows)
    query_metrics_path = report_dir / "rk_anchor64_query_metrics.csv"
    preview_path = report_dir / "rk_anchor64_cache_preview.jsonl"
    query_df.to_csv(query_metrics_path, index=False)
    _write_jsonl(preview_path, preview_rows)

    summary = {
        "status": "dry_run_only",
        "ready_for_training": False,
        "ready_for_inference_hook": False,
        "notes": [
            "Dry-run stops at retrieval cache assembly and token-vote simulation.",
            "Main generation path still lacks a custom logits interpolation hook.",
        ],
        "anchor_queries": int(anchor_visible.shape[0]),
        "rk_k": int(rk_k),
        "token_vote_probe_steps": int(max_step_probe),
        "datastore": {
            "rows": int(train_visible.shape[0]),
            "total_target_tokens": int(total_train_target_tokens),
            "target_token_len_mean": round(_mean([float(x) for x in train_target_token_lens]), 4),
            "target_token_len_p95": round(_pctl([float(x) for x in train_target_token_lens], 95), 4),
            "hidden_size": int(hidden_size),
            "estimated_fp16_key_store_gib": round(float(fp16_datastore_gib), 4),
            "estimated_value_store_gib": round(float(value_store_gib), 4),
        },
        "latency": {
            "index_build_seconds": round(float(index_elapsed), 4),
            "anchor_query_seconds": round(float(query_elapsed), 4),
            "train_target_tokenize_seconds": round(float(train_target_token_elapsed), 4),
            "retrieval_latency_ms_mean": round(_mean(retrieval_latencies_ms), 4),
            "retrieval_latency_ms_p95": round(_pctl(retrieval_latencies_ms, 95), 4),
            "cache_assembly_latency_ms_mean": round(_mean(assembly_latencies_ms), 4),
            "cache_assembly_latency_ms_p95": round(_pctl(assembly_latencies_ms, 95), 4),
        },
        "neighbor_quality": {
            "top1_score_mean": round(_mean(top1_scores), 4),
            "topk_score_mean": round(_mean(mean_topk_scores), 4),
            "top1_target_chrfpp_mean": round(_mean(top1_target_chrfpp), 4),
            "oracle_topk_target_chrfpp_mean": round(_mean(oracle_topk_target_chrfpp), 4),
            "oracle_topk_gain_over_top1_chrfpp": round(
                _mean([b - a for a, b in zip(top1_target_chrfpp, oracle_topk_target_chrfpp)]),
                4,
            ),
            "oracle_next_token_recall_at1_mean": round(_mean(oracle_recall_at1), 4),
            "oracle_next_token_recall_at5_mean": round(_mean(oracle_recall_at5), 4),
            "neighbor_target_len_mean": round(_mean(neighbor_target_lens_mean), 4),
        },
        "artifacts": {
            "anchor64_query_metrics_csv": str(query_metrics_path),
            "anchor64_cache_preview_jsonl": str(preview_path),
            "anchor64_query_metrics_csv_bytes": int(query_metrics_path.stat().st_size),
            "anchor64_cache_preview_jsonl_bytes": int(preview_path.stat().st_size),
        },
    }
    _write_json(report_dir / "rk_summary.json", summary)
    print("INFO: RK summary ready", flush=True)
    return summary


def _render_report(r3: dict[str, Any], rk: dict[str, Any]) -> str:
    lines = [
        "# A2 Retrieval Infra Probe",
        "",
        "## R3 top3 compressed",
        "",
        f"- processed_dir: `{r3['processed_dir']}`",
        f"- config: `{r3['config_path']}`",
        f"- delta tokens mean / p95: `{r3['length_overhead']['delta_tokens_mean']}` / `{r3['length_overhead']['delta_tokens_p95']}`",
        f"- rows over max_source_length: `{r3['overflow_risk']['base_rows_over_limit']} -> {r3['overflow_risk']['r3_rows_over_limit']}`",
        "",
        "## RK dry-run",
        "",
        f"- anchor queries: `{rk['anchor_queries']}`",
        f"- datastore target tokens: `{rk['datastore']['total_target_tokens']}`",
        f"- estimated fp16 key store GiB: `{rk['datastore']['estimated_fp16_key_store_gib']}`",
        f"- retrieval latency mean / p95 ms: `{rk['latency']['retrieval_latency_ms_mean']}` / `{rk['latency']['retrieval_latency_ms_p95']}`",
        f"- cache assembly mean / p95 ms: `{rk['latency']['cache_assembly_latency_ms_mean']}` / `{rk['latency']['cache_assembly_latency_ms_p95']}`",
        f"- top1 target chrF++ mean: `{rk['neighbor_quality']['top1_target_chrfpp_mean']}`",
        f"- oracle topk target chrF++ mean: `{rk['neighbor_quality']['oracle_topk_target_chrfpp_mean']}`",
        f"- oracle token recall@1 / @5: `{rk['neighbor_quality']['oracle_next_token_recall_at1_mean']}` / `{rk['neighbor_quality']['oracle_next_token_recall_at5_mean']}`",
        "",
        "## Feasibility",
        "",
    ]
    if rk.get("ready_for_inference_hook"):
        lines.append("- RK is infra-complete for generation-side interpolation.")
    else:
        lines.append("- RK dry-run is positive only for datastore/cache shape; generation hook still needs custom logits interpolation.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml",
    )
    ap.add_argument("--base-processed-dir", default="")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--r3-output-dir", default="data/processed_byt5_chunks_align_gc_cost14_retrieval_top3_comp_fold0")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_retrieval_probe_20260310")
    ap.add_argument("--r3-top-k", type=int, default=3)
    ap.add_argument("--rk-k", type=int, default=8)
    ap.add_argument("--max-anchor-queries", type=int, default=64)
    ap.add_argument("--max-step-probe", type=int, default=32)
    ap.add_argument("--preview-queries", type=int, default=12)
    ap.add_argument("--preview-steps", type=int, default=8)
    ap.add_argument("--hint-source-max-chars", type=int, default=96)
    ap.add_argument("--hint-target-max-chars", type=int, default=128)
    args = ap.parse_args()

    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_retrieval_probe_20260310")
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
    r3_output_dir = _resolve_path(
        args.r3_output_dir,
        REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14_retrieval_top3_comp_fold0",
    )
    generated_config_dir = report_dir / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)
    r3_config_path = generated_config_dir / "taskform_winner_a2_retrieval_top3_comp.yaml"

    train_visible, val_visible = _load_fold_split(base_cfg, base_processed_dir, int(args.fold))
    r3_summary = _build_r3_processed(
        base_cfg_path=base_cfg_path,
        base_cfg=base_cfg,
        base_processed_dir=base_processed_dir,
        fold=int(args.fold),
        output_dir=r3_output_dir,
        report_dir=report_dir,
        config_out_path=r3_config_path,
        top_k=int(args.r3_top_k),
        hint_source_max_chars=int(args.hint_source_max_chars),
        hint_target_max_chars=int(args.hint_target_max_chars),
    )
    rk_summary = _build_rk_dry_run(
        base_cfg=base_cfg,
        train_visible=train_visible,
        val_visible=val_visible,
        report_dir=report_dir,
        max_anchor_queries=int(args.max_anchor_queries),
        rk_k=int(args.rk_k),
        max_step_probe=int(args.max_step_probe),
        preview_queries=int(args.preview_queries),
        preview_steps=int(args.preview_steps),
    )

    summary = {
        "status": "completed",
        "line": "A2_retrieval_infra_probe",
        "fold": int(args.fold),
        "base_config_path": str(base_cfg_path),
        "base_processed_dir": str(base_processed_dir),
        "r3": r3_summary,
        "rk": rk_summary,
        "artifacts": {
            "report_md": str(report_dir / "report.md"),
            "summary_json": str(report_dir / "summary.json"),
        },
    }
    _write_json(report_dir / "summary.json", summary)
    _write_text(report_dir / "report.md", _render_report(r3_summary, rk_summary))
    print(f"OK: wrote {report_dir / 'summary.json'}")
    print(f"OK: wrote {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
