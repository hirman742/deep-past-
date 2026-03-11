from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from generation_utils import build_bad_words_ids, build_generate_kwargs, resolve_generation_settings
from taskform_phase12_common import build_health, evaluate_predictions, resolve_path, write_json, write_text
from taskform_winner_a2_retrieval_r3_rk_probe import _batch_token_ids


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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


def _load_fold_split(base_cfg: dict[str, Any], processed_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_prefix = _task_prefix(base_cfg)
    train_proc = pd.read_csv(processed_dir / "train_proc.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")[["oare_id", "fold"]]
    merged = train_proc.merge(folds, on="oare_id", how="inner")
    train_visible = merged.loc[merged["fold"] != int(fold)].copy().reset_index(drop=True)
    val_visible = merged.loc[merged["fold"] == int(fold)].copy().reset_index(drop=True)
    for frame in (train_visible, val_visible):
        frame["source_plain"] = frame["source"].fillna("").astype(str).map(lambda text: _strip_task_prefix(text, task_prefix))
        frame["target_plain"] = frame["target"].fillna("").astype(str)
    return train_visible, val_visible


def _normalize_scores(scores: list[float]) -> list[float]:
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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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
                    votes[int(self.eos_token_id)] = votes.get(int(self.eos_token_id), 0.0) + float(weight)
            if not votes:
                continue
            for token_id, weight in votes.items():
                if 0 <= int(token_id) < scores.shape[1]:
                    scores[row_idx, int(token_id)] = scores[row_idx, int(token_id)] + (self.bias_strength * float(weight))
        return scores


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
        target_text = _safe_text(row["target_plain"])
        if not target_text or target_text == query_target_plain:
            continue
        key = target_text
        if key in seen_targets:
            score_pos = seen_targets[key]
            neighbor_scores[score_pos] += float(score_row[int(idx)])
            continue
        seen_targets[key] = len(neighbor_ids)
        neighbor_ids.append(_safe_text(row["oare_id"]))
        neighbor_scores.append(float(score_row[int(idx)]))
        neighbor_targets.append(target_text)
        neighbor_token_rows.append(target_token_ids[int(idx)])
        if len(neighbor_ids) >= int(final_k):
            break

    weights = _normalize_scores(neighbor_scores)
    return QueryNeighbors(
        neighbor_ids=neighbor_ids,
        neighbor_scores=[round(float(x), 6) for x in neighbor_scores],
        neighbor_weights=[round(float(x), 6) for x in weights],
        neighbor_targets=neighbor_targets,
        neighbor_token_ids=neighbor_token_rows,
    )


def _aggregate_parent_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if "parent_oare_id" not in frame.columns:
        grouped = frame[["oare_id", "source", "reference", "prediction"]].copy()
        grouped = grouped.rename(columns={"oare_id": "id"})
        return grouped
    ordered = frame.copy()
    sort_cols = ["parent_oare_id"]
    if "chunk_index" in ordered.columns:
        sort_cols.append("chunk_index")
    ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
    grouped = (
        ordered.groupby("parent_oare_id", as_index=False)
        .agg(
            source=("source", lambda s: "\n".join(str(x).strip() for x in s.tolist() if str(x).strip())),
            reference=("reference", lambda s: "\n".join(str(x).strip() for x in s.tolist() if str(x).strip())),
            prediction=("prediction", lambda s: "\n".join(str(x).strip() for x in s.tolist() if str(x).strip())),
        )
        .rename(columns={"parent_oare_id": "id"})
    )
    return grouped


def _load_baseline_anchor64() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    diag_dir = REPO_ROOT / "runs" / "TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0" / "diagnostics"
    decode_best_path = diag_dir / "decode_grid_best_taskform_winner_a2_r1_wlite_anchor64_20260310.json"
    diag_pred_path = diag_dir / "val_predictions_diagnostic_taskform_winner_a2_r1_wlite_anchor64_20260310.csv"
    recon_path = diag_dir / "val_predictions_reconstructed_taskform_winner_a2_r1_wlite_anchor64_20260310.csv"
    if not decode_best_path.exists():
        raise FileNotFoundError(f"Missing baseline anchor64 decode summary: {decode_best_path}")
    payload = json.loads(decode_best_path.read_text(encoding="utf-8"))
    return payload, pd.read_csv(diag_pred_path), pd.read_csv(recon_path)


def _evaluate_probe_predictions(
    *,
    probe_df: pd.DataFrame,
    baseline_payload: dict[str, Any],
    baseline_diag_df: pd.DataFrame,
    baseline_recon_df: pd.DataFrame,
    elapsed_seconds: float,
) -> dict[str, Any]:
    chunk_summary = evaluate_predictions(
        predictions=probe_df["prediction"].fillna("").astype(str).tolist(),
        references=probe_df["reference"].fillna("").astype(str).tolist(),
        tag="rk_probe_chunk",
        subset_name="anchor64_chunk",
        note="proxy logits interpolation anchor64 probe",
    )
    recon_df = _aggregate_parent_predictions(probe_df)
    recon_summary = evaluate_predictions(
        predictions=recon_df["prediction"].fillna("").astype(str).tolist(),
        references=recon_df["reference"].fillna("").astype(str).tolist(),
        tag="rk_probe_reconstructed",
        subset_name="anchor64_reconstructed",
        note="proxy logits interpolation anchor64 probe",
    )
    baseline_chunk_summary = evaluate_predictions(
        predictions=baseline_diag_df["prediction"].fillna("").astype(str).tolist(),
        references=baseline_diag_df["reference"].fillna("").astype(str).tolist(),
        tag="baseline_chunk",
        subset_name="anchor64_chunk",
    )
    baseline_recon_summary = evaluate_predictions(
        predictions=baseline_recon_df["prediction"].fillna("").astype(str).tolist(),
        references=baseline_recon_df["reference"].fillna("").astype(str).tolist(),
        tag="baseline_reconstructed",
        subset_name="anchor64_reconstructed",
    )
    return {
        "elapsed_seconds": round(float(elapsed_seconds), 4),
        "chunk": chunk_summary,
        "reconstructed": recon_summary,
        "baseline_chunk": baseline_chunk_summary,
        "baseline_reconstructed": baseline_recon_summary,
        "delta_geom_vs_baseline_chunk": round(
            float(chunk_summary["eval_geom"]) - float(baseline_chunk_summary["eval_geom"]),
            4,
        ),
        "delta_geom_vs_baseline_reconstructed": round(
            float(recon_summary["eval_geom"]) - float(baseline_recon_summary["eval_geom"]),
            4,
        ),
        "latency_ratio_vs_baseline_anchor64": round(
            float(elapsed_seconds) / float(max(1e-9, float(baseline_payload.get("elapsed_seconds", 0.0) or 0.0))),
            4,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml",
    )
    ap.add_argument("--checkpoint-dir", default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_rk_anchor64_probe_20260310")
    ap.add_argument("--predict-batch-size", type=int, default=8)
    ap.add_argument("--max-anchor-queries", type=int, default=64)
    ap.add_argument("--rk-k", type=int, default=8)
    ap.add_argument("--raw-pool-k", type=int, default=48)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--length-penalty", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--max-bias-steps", type=int, default=192)
    ap.add_argument("--bias-strengths", default="1.5,3.0,4.5")
    args = ap.parse_args()

    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a2_rk_anchor64_probe_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = resolve_path(args.config, REPO_ROOT / "reports")
    checkpoint_dir = resolve_path(args.checkpoint_dir, REPO_ROOT / "runs")
    cfg = _load_yaml(cfg_path)

    processed_dir = resolve_path((cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_visible, val_visible = _load_fold_split(cfg, processed_dir, int(args.fold))
    anchor_visible = val_visible.iloc[: max(1, int(args.max_anchor_queries))].copy().reset_index(drop=True)
    if anchor_visible.empty:
        raise ValueError("Anchor64 subset is empty")

    model_cfg = cfg.get("model", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)
    generation_settings["num_beams"] = int(args.num_beams)
    generation_settings["length_penalty"] = float(args.length_penalty)
    generation_settings["max_new_tokens"] = int(args.max_new_tokens)
    generation_settings["min_new_tokens"] = int(args.min_new_tokens)
    generation_settings["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)

    model_name = str(model_cfg.get("name", "google/byt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 640))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(
        f"INFO: RK probe device={device} anchor_rows={anchor_visible.shape[0]} train_visible_rows={train_visible.shape[0]}",
        flush=True,
    )
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=False,
        sublinear_tf=True,
    )
    index_start = time.perf_counter()
    train_matrix = vectorizer.fit_transform(train_visible["source_plain"].fillna("").astype(str).tolist())
    anchor_matrix = vectorizer.transform(anchor_visible["source_plain"].fillna("").astype(str).tolist())
    sim = linear_kernel(anchor_matrix, train_matrix)
    index_elapsed = time.perf_counter() - index_start

    print("INFO: tokenizing RK datastore targets", flush=True)
    tokenize_start = time.perf_counter()
    train_target_token_ids = _batch_token_ids(tokenizer, train_visible["target_plain"].fillna("").astype(str).tolist())
    tokenize_elapsed = time.perf_counter() - tokenize_start

    query_neighbors: list[QueryNeighbors] = []
    query_rows: list[dict[str, Any]] = []
    for row_idx in range(anchor_visible.shape[0]):
        score_row = sim[row_idx]
        order = np.argsort(-score_row)
        query = anchor_visible.iloc[row_idx]
        deduped = _dedup_neighbors(
            train_visible=train_visible,
            order=order,
            score_row=score_row,
            target_token_ids=train_target_token_ids,
            query_target_plain=_safe_text(query["target_plain"]),
            raw_pool_k=int(args.raw_pool_k),
            final_k=int(args.rk_k),
        )
        query_neighbors.append(deduped)
        query_rows.append(
            {
                "oare_id": _safe_text(query["oare_id"]),
                "parent_oare_id": _safe_text(query.get("parent_oare_id")),
                "neighbor_count": int(len(deduped.neighbor_ids)),
                "neighbor_ids": "|".join(deduped.neighbor_ids),
                "neighbor_scores": "|".join(f"{float(x):.4f}" for x in deduped.neighbor_scores),
                "neighbor_weights": "|".join(f"{float(x):.4f}" for x in deduped.neighbor_weights),
            }
        )
    pd.DataFrame(query_rows).to_csv(report_dir / "query_neighbors.csv", index=False)

    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )
    sources = anchor_visible["source"].fillna("").astype(str).tolist()
    references = anchor_visible["target"].fillna("").astype(str).tolist()
    baseline_payload, baseline_diag_df, baseline_recon_df = _load_baseline_anchor64()

    bias_strengths = [float(item.strip()) for item in str(args.bias_strengths).split(",") if item.strip()]
    sweep_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_alpha: float | None = None

    for alpha in bias_strengths:
        print(f"INFO: generating alpha={alpha}", flush=True)
        start = time.perf_counter()
        predictions: list[str] = []
        processor = RetrievalBiasLogitsProcessor(
            query_neighbors=query_neighbors,
            num_beams=int(generation_settings["num_beams"]),
            bias_strength=float(alpha),
            max_bias_steps=int(args.max_bias_steps),
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            batch_start_idx = 0
            for batch_sources in _chunk(sources, max(1, int(args.predict_batch_size))):
                tokenized = tokenizer(
                    batch_sources,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_source_length,
                    padding=True,
                )
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                batch_queries = query_neighbors[batch_start_idx : batch_start_idx + len(batch_sources)]
                batch_processor = LogitsProcessorList(
                    [
                        RetrievalBiasLogitsProcessor(
                            query_neighbors=batch_queries,
                            num_beams=int(generation_settings["num_beams"]),
                            bias_strength=float(alpha),
                            max_bias_steps=int(args.max_bias_steps),
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    ]
                )
                generated = model.generate(
                    **tokenized,
                    **build_generate_kwargs(
                        num_beams=int(generation_settings["num_beams"]),
                        length_penalty=float(generation_settings["length_penalty"]),
                        max_new_tokens=int(generation_settings["max_new_tokens"]),
                        min_new_tokens=int(generation_settings["min_new_tokens"]),
                        no_repeat_ngram_size=int(generation_settings["no_repeat_ngram_size"]),
                        bad_words_ids=bad_words_ids,
                    ),
                    logits_processor=batch_processor,
                )
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.extend([text.strip() for text in decoded])
                batch_start_idx += len(batch_sources)
        elapsed = time.perf_counter() - start
        probe_df = anchor_visible.copy()
        probe_df["reference"] = references
        probe_df["prediction"] = predictions
        if "oare_id" not in probe_df.columns:
            probe_df["oare_id"] = [str(idx) for idx in range(len(probe_df))]
        out_cols = [col for col in ["oare_id", "parent_oare_id", "chunk_index", "source", "reference", "prediction"] if col in probe_df.columns]
        probe_df[out_cols].to_csv(report_dir / f"rk_anchor64_predictions_alpha{str(alpha).replace('.', 'p')}.csv", index=False)

        eval_summary = _evaluate_probe_predictions(
            probe_df=probe_df[out_cols].copy(),
            baseline_payload=baseline_payload,
            baseline_diag_df=baseline_diag_df,
            baseline_recon_df=baseline_recon_df,
            elapsed_seconds=elapsed,
        )
        alpha_key = str(alpha).replace(".", "p")
        write_json(report_dir / f"rk_anchor64_eval_alpha{alpha_key}.json", eval_summary)

        row = {
            "alpha": float(alpha),
            "elapsed_seconds": round(float(elapsed), 4),
            "chunk_geom": round(float(eval_summary["chunk"]["eval_geom"]), 4),
            "reconstructed_geom": round(float(eval_summary["reconstructed"]["eval_geom"]), 4),
            "delta_geom_vs_baseline_chunk": float(eval_summary["delta_geom_vs_baseline_chunk"]),
            "delta_geom_vs_baseline_reconstructed": float(eval_summary["delta_geom_vs_baseline_reconstructed"]),
            "chunk_unique_prediction_ratio_pct": round(
                100.0 * float(pd.Series(predictions).nunique()) / float(max(1, len(predictions))),
                4,
            ),
            "chunk_repeat_prediction_ratio_pct": round(
                float(build_health(predictions, references)["repeat_prediction_ratio_pct"]),
                4,
            ),
            "reconstructed_repeat_prediction_ratio_pct": round(
                float(eval_summary["reconstructed"]["health"]["repeat_prediction_ratio_pct"]),
                4,
            ),
            "latency_ratio_vs_baseline_anchor64": float(eval_summary["latency_ratio_vs_baseline_anchor64"]),
        }
        sweep_rows.append(row)
        if best_result is None or row["reconstructed_geom"] > float(best_result["reconstructed"]["eval_geom"]):
            best_result = eval_summary
            best_alpha = float(alpha)

    sweep_df = pd.DataFrame(sweep_rows).sort_values(["reconstructed_geom", "chunk_geom"], ascending=False).reset_index(drop=True)
    sweep_df.to_csv(report_dir / "alpha_sweep.csv", index=False)
    if best_result is None or best_alpha is None:
        raise RuntimeError("RK probe produced no results")

    summary = {
        "status": "completed_proxy_probe",
        "line": "A2_RK_anchor64_proxy_logits_interpolation",
        "notes": [
            "This is a generation-side proxy probe, not a full decoder-state kNN-MT datastore.",
            "Bias is injected from TF-IDF-retrieved target token votes at each decode step.",
        ],
        "config_path": str(cfg_path),
        "checkpoint_dir": str(checkpoint_dir),
        "fold": int(args.fold),
        "anchor_rows": int(anchor_visible.shape[0]),
        "rk_k": int(args.rk_k),
        "raw_pool_k": int(args.raw_pool_k),
        "max_bias_steps": int(args.max_bias_steps),
        "index_build_seconds": round(float(index_elapsed), 4),
        "target_tokenize_seconds": round(float(tokenize_elapsed), 4),
        "baseline_anchor64": {
            "elapsed_seconds": float(baseline_payload.get("elapsed_seconds", 0.0) or 0.0),
            "eval_geom": float(baseline_payload.get("eval_geom", 0.0) or 0.0),
            "eval_bleu": float(baseline_payload.get("eval_bleu", 0.0) or 0.0),
            "eval_chrfpp": float(baseline_payload.get("eval_chrfpp", 0.0) or 0.0),
        },
        "best_alpha": float(best_alpha),
        "best_result": best_result,
        "artifacts": {
            "query_neighbors_csv": str(report_dir / "query_neighbors.csv"),
            "alpha_sweep_csv": str(report_dir / "alpha_sweep.csv"),
            "summary_json": str(report_dir / "summary.json"),
            "report_md": str(report_dir / "report.md"),
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# RK Anchor64 Proxy Probe",
        "",
        f"- status: `{summary['status']}`",
        f"- anchor rows: `{summary['anchor_rows']}`",
        f"- rk_k / raw_pool_k: `{summary['rk_k']}` / `{summary['raw_pool_k']}`",
        f"- index build seconds: `{summary['index_build_seconds']}`",
        f"- target tokenize seconds: `{summary['target_tokenize_seconds']}`",
        f"- baseline anchor64 geom: `{summary['baseline_anchor64']['eval_geom']:.4f}`",
        f"- best alpha: `{summary['best_alpha']}`",
        f"- best probe chunk geom: `{summary['best_result']['chunk']['eval_geom']:.4f}`",
        f"- best probe reconstructed geom: `{summary['best_result']['reconstructed']['eval_geom']:.4f}`",
        f"- delta vs baseline chunk: `{summary['best_result']['delta_geom_vs_baseline_chunk']}`",
        f"- delta vs baseline reconstructed: `{summary['best_result']['delta_geom_vs_baseline_reconstructed']}`",
        f"- latency ratio vs baseline anchor64: `{summary['best_result']['latency_ratio_vs_baseline_anchor64']}`",
        "",
        "## Notes",
        "",
        "- This probe is proxy-only and should not be described as full kNN-MT.",
        "- Positive result means the generation-side interpolation idea is worth integrating into the main decode path.",
    ]
    write_text(report_dir / "report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
