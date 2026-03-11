from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from fusion_flow_common import (
    combine_chunk_texts,
    compute_eval_summary,
    generate_predictions,
    load_yaml,
    normalize_whitespace,
    resolve_path,
    sanitize_draft,
    safe_text,
    title_case,
    write_csv,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _id_col(frame: pd.DataFrame) -> str:
    if "oare_id" in frame.columns:
        return "oare_id"
    if "id" in frame.columns:
        return "id"
    raise KeyError("Expected `oare_id` or `id` column")


def _load_processed_val(processed_dir: Path, fold: int, max_val_parents: int) -> pd.DataFrame:
    train_df = pd.read_csv(processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(processed_dir / "folds.csv")
    merged = train_df.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    val_df = merged.loc[merged["fold"].astype(int) == int(fold)].copy()
    if "route_rank" in val_df.columns:
        val_df["route_rank"] = val_df["route_rank"].fillna(999999).astype(int)
        val_df = val_df.sort_values(["route_rank", "oare_id"]).reset_index(drop=True)
    else:
        val_df = val_df.sort_values("oare_id").reset_index(drop=True)
    if max_val_parents > 0:
        val_df = val_df.head(int(max_val_parents)).reset_index(drop=True)
    return val_df


def _write_predictions_csv(frame: pd.DataFrame, predictions: list[str], out_csv: Path) -> None:
    out = frame.copy()
    out["prediction"] = predictions
    keep_cols = [
        col
        for col in [
            "oare_id",
            "parent_oare_id",
            "route_rank",
            "route_score",
            "route_reason",
            "orig_chunk_total",
            "parent_ref_tok",
            "source",
            "target",
            "prediction",
        ]
        if col in out.columns
    ]
    renamed = out[keep_cols].rename(columns={"target": "reference"})
    write_csv(out_csv, renamed)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_subset_meta(metadata_df: pd.DataFrame, *, scope: str, max_parents: int) -> pd.DataFrame:
    if str(scope).strip().lower() == "all_val":
        subset_meta = metadata_df.loc[metadata_df["split"] == "val"].copy()
    else:
        subset_meta = metadata_df.loc[
            (metadata_df["split"] == "val") & (metadata_df["is_routed_hard"] == True)
        ].copy()
    subset_meta["route_rank"] = subset_meta["route_rank"].fillna(999999).astype(int)
    subset_meta = subset_meta.sort_values(["route_rank", "parent_oare_id"]).reset_index(drop=True)
    if int(max_parents) > 0:
        subset_meta = subset_meta.head(int(max_parents)).reset_index(drop=True)
    return subset_meta


def _length_stats(tokenizer, texts: list[str], *, max_new_tokens: int, max_target_length: int) -> dict[str, Any]:
    lengths = [len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) for text in texts]
    series = pd.Series(lengths, dtype="float64")
    rows = int(len(lengths))
    over_new = int(sum(length > int(max_new_tokens) for length in lengths))
    over_target = int(sum(length > int(max_target_length) for length in lengths))
    return {
        "rows": rows,
        "tok_p50": float(series.quantile(0.5)) if rows else 0.0,
        "tok_p90": float(series.quantile(0.9)) if rows else 0.0,
        "tok_p95": float(series.quantile(0.95)) if rows else 0.0,
        "tok_max": int(max(lengths)) if rows else 0,
        "over_max_new_tokens": over_new,
        "over_max_new_tokens_pct": 100.0 * (float(over_new) / float(rows)) if rows else 0.0,
        "over_max_target_length": over_target,
        "over_max_target_length_pct": 100.0 * (float(over_target) / float(rows)) if rows else 0.0,
    }


def _dedup_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _compose_parent_prediction(
    chunk_rows: list[dict[str, Any]],
    *,
    mode: str,
    text_field: str,
    sanitize_chunks: bool,
    sanitize_max_words: int,
    sanitize_tail_words: int,
    dedup_strategy: str,
) -> tuple[str, str]:
    if sanitize_chunks:
        pieces = []
        ordered = sorted(chunk_rows, key=lambda item: int(item["chunk_index"]))
        fallback_field = "chunk_target" if mode == "oracle" else "draft_prediction"
        for chunk in ordered:
            raw = safe_text(chunk.get(text_field, "") or chunk.get(fallback_field, ""))
            text = sanitize_draft(
                raw,
                max_words=int(sanitize_max_words),
                tail_words=int(sanitize_tail_words),
            ) if raw else ""
            if text:
                pieces.append(text)
        if dedup_strategy == "consecutive_exact":
            pieces = [value for value in pieces if value]
            out: list[str] = []
            for value in pieces:
                if out and out[-1] == value:
                    continue
                out.append(value)
            pieces = out
        elif dedup_strategy == "global_exact":
            pieces = _dedup_keep_order([value for value in pieces if value])
        prediction = normalize_whitespace(" ".join(pieces))
        source = prediction
        return source, prediction

    combined = combine_chunk_texts(
        chunk_rows,
        mode=mode,
        draft_field=text_field,
        dedup_consecutive=(dedup_strategy == "consecutive_exact"),
    )
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if dedup_strategy == "global_exact":
        lines = _dedup_keep_order(lines)
    prediction = normalize_whitespace(" ".join(lines))
    return combined, prediction


def matched_baseline(args: argparse.Namespace, *, line_name: str) -> None:
    metadata_df = pd.read_csv(args.metadata_csv)
    subset_meta = _select_subset_meta(
        metadata_df,
        scope=str(args.scope),
        max_parents=int(args.max_parents),
    )
    ids = set(subset_meta["parent_oare_id"].astype(str).tolist())

    baseline_df = pd.read_csv(args.baseline_csv)
    id_col = _id_col(baseline_df)
    baseline_df[id_col] = baseline_df[id_col].astype(str)
    subset = baseline_df.loc[baseline_df[id_col].isin(ids)].copy()
    subset = subset.set_index(id_col).loc[subset_meta["parent_oare_id"].astype(str).tolist()].reset_index()
    summary = compute_eval_summary(
        predictions=subset["prediction"].fillna("").astype(str).tolist(),
        references=subset["reference"].fillna("").astype(str).tolist(),
        sources=subset["source"].fillna("").astype(str).tolist() if "source" in subset.columns else [""] * len(subset),
        tokenizer_name=args.tokenizer_name,
        tag=args.tag,
        checkpoint_dir=args.checkpoint_dir,
        subset_name=args.subset_name,
        eval_rows=int(len(subset)),
        extra={
            "baseline_csv": str(args.baseline_csv),
            "selected_parent_count": int(len(subset_meta)),
            "scope": str(args.scope),
        },
    )
    if args.out_csv:
        write_csv(Path(args.out_csv), subset)
    write_json(Path(args.out_json), summary)


def target_length_audit(args: argparse.Namespace, *, line_name: str) -> None:
    processed_dir = resolve_path(args.processed_dir, REPO_ROOT / "data" / "processed_taskform_dan1_pred_smoke_fold0")
    train_df = pd.read_csv(processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(processed_dir / "folds.csv")
    merged = train_df.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")
    merged["fold"] = merged["fold"].astype(int)
    if "route_rank" in merged.columns:
        merged["route_rank"] = merged["route_rank"].fillna(999999).astype(int)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    val_df = merged.loc[merged["fold"] == int(args.fold)].copy()
    if "route_rank" in val_df.columns:
        val_df = val_df.sort_values(["route_rank", "oare_id"]).reset_index(drop=True)
    else:
        val_df = val_df.sort_values("oare_id").reset_index(drop=True)
    if int(args.max_val_parents) > 0:
        val_df = val_df.head(int(args.max_val_parents)).reset_index(drop=True)

    train_split_df = merged.loc[merged["fold"] != int(args.fold)].copy()

    summary = {
        "line": line_name,
        "tag": args.tag,
        "processed_dir": str(processed_dir),
        "fold": int(args.fold),
        "tokenizer_name": args.tokenizer_name,
        "thresholds": {
            "max_new_tokens": int(args.max_new_tokens),
            "max_target_length": int(args.max_target_length),
        },
        "train_targets": _length_stats(
            tokenizer,
            train_split_df["target"].fillna("").astype(str).tolist(),
            max_new_tokens=int(args.max_new_tokens),
            max_target_length=int(args.max_target_length),
        ),
        "val_targets": _length_stats(
            tokenizer,
            val_df["target"].fillna("").astype(str).tolist(),
            max_new_tokens=int(args.max_new_tokens),
            max_target_length=int(args.max_target_length),
        ),
        "train_sources": _length_stats(
            tokenizer,
            train_split_df["source"].fillna("").astype(str).tolist(),
            max_new_tokens=int(args.max_new_tokens),
            max_target_length=int(args.max_target_length),
        ),
        "val_sources": _length_stats(
            tokenizer,
            val_df["source"].fillna("").astype(str).tolist(),
            max_new_tokens=int(args.max_new_tokens),
            max_target_length=int(args.max_target_length),
        ),
    }
    summary["gate"] = {
        "needs_budget_repair": bool(
            float(summary["val_targets"]["over_max_new_tokens_pct"]) > float(args.max_over_new_pct)
            or float(summary["val_targets"]["over_max_target_length_pct"]) > float(args.max_over_target_pct)
        ),
        "max_over_new_pct": float(args.max_over_new_pct),
        "max_over_target_pct": float(args.max_over_target_pct),
    }
    write_json(Path(args.out_json), summary)


def synthetic_baseline(args: argparse.Namespace, *, line_name: str) -> None:
    metadata_df = pd.read_csv(args.metadata_csv)
    subset_meta = _select_subset_meta(
        metadata_df,
        scope=str(args.scope),
        max_parents=int(args.max_parents),
    )
    cache_df = pd.read_csv(args.draft_cache_csv)
    cache_df["parent_oare_id"] = cache_df["parent_oare_id"].astype(str)

    rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    for parent_id in subset_meta["parent_oare_id"].astype(str).tolist():
        group_df = cache_df.loc[cache_df["parent_oare_id"] == parent_id].copy()
        if group_df.empty:
            continue
        chunk_rows = group_df.sort_values("chunk_index").to_dict(orient="records")
        source, prediction = _compose_parent_prediction(
            chunk_rows,
            mode=str(args.mode),
            text_field=str(args.text_field),
            sanitize_chunks=bool(args.sanitize_chunks),
            sanitize_max_words=int(args.sanitize_max_words),
            sanitize_tail_words=int(args.sanitize_tail_words),
            dedup_strategy=str(args.dedup_strategy),
        )
        reference = safe_text(group_df["parent_translation"].iloc[0])
        meta = subset_meta.loc[subset_meta["parent_oare_id"].astype(str) == parent_id].iloc[0]
        predictions.append(prediction)
        references.append(reference)
        rows.append(
            {
                "oare_id": parent_id,
                "parent_oare_id": parent_id,
                "route_rank": int(meta["route_rank"]),
                "route_score": float(meta["route_score"]),
                "route_reason": safe_text(meta["route_reason"]),
                "orig_chunk_total": int(meta["chunk_total"]),
                "parent_ref_tok": int(meta["parent_ref_tok"]),
                "source": source,
                "reference": reference,
                "prediction": prediction,
            }
        )

    summary = compute_eval_summary(
        predictions=predictions,
        references=references,
        sources=[""] * len(predictions),
        tokenizer_name=args.tokenizer_name,
        tag=args.tag,
        checkpoint_dir=args.checkpoint_dir,
        subset_name=args.subset_name,
        eval_rows=int(len(predictions)),
        extra={
            "draft_cache_csv": str(args.draft_cache_csv),
            "text_field": str(args.text_field),
            "mode": str(args.mode),
            "dedup_strategy": str(args.dedup_strategy),
            "sanitize_chunks": bool(args.sanitize_chunks),
        },
    )
    write_csv(Path(args.out_csv), pd.DataFrame(rows))
    write_json(Path(args.out_json), summary)


def eval_model(args: argparse.Namespace, *, line_name: str) -> None:
    cfg = load_yaml(resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "taskform_dan1_routed_probe.yaml"))
    processed_dir = resolve_path((cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed_taskform_dan1_routed_fold0")
    model_cfg = cfg.get("model", {}) or {}
    val_df = _load_processed_val(processed_dir, int(args.fold), int(args.max_val_parents))
    checkpoint_dir = resolve_path(args.checkpoint_dir, REPO_ROOT / "runs" / "missing")
    predictions = generate_predictions(
        model_name=str(model_cfg.get("name", "google/flan-t5-small")),
        checkpoint_dir=checkpoint_dir,
        sources=val_df["source"].fillna("").astype(str).tolist(),
        max_source_length=int(model_cfg.get("max_source_length", 1024)),
        predict_batch_size=int(args.predict_batch_size),
        num_beams=int(args.num_beams),
        length_penalty=float(args.length_penalty),
        max_new_tokens=int(args.max_new_tokens),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        bad_tokens_regex=str((cfg.get("generation", {}) or {}).get("bad_tokens_regex", r"<extra_id_\d+>")),
        suppress_extra_ids=bool((cfg.get("generation", {}) or {}).get("suppress_extra_ids", True)),
    )
    summary = compute_eval_summary(
        predictions=predictions,
        references=val_df["target"].fillna("").astype(str).tolist(),
        sources=val_df["source"].fillna("").astype(str).tolist(),
        tokenizer_name=str(model_cfg.get("name", "google/flan-t5-small")),
        tag=args.tag,
        checkpoint_dir=str(checkpoint_dir),
        subset_name=args.subset_name,
        eval_rows=int(len(val_df)),
        extra={
            "config_path": str(resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "taskform_dan1_routed_probe.yaml")),
            "processed_dir": str(processed_dir),
        },
    )
    _write_predictions_csv(val_df, predictions, Path(args.out_csv))
    write_json(Path(args.out_json), summary)


def gate_report(args: argparse.Namespace, *, line_name: str) -> None:
    matched = _load_json(args.matched_json)
    candidates = [_load_json(path) for path in args.candidate_jsons]
    winner = max(candidates, key=lambda item: float(item["eval_geom"]))
    delta = float(winner["eval_geom"]) - float(matched["eval_geom"])
    empty_delta = float(winner["health"]["empty_prediction_ratio_pct"]) - float(matched["health"]["empty_prediction_ratio_pct"])
    short_delta = float(winner["health"]["pred_shorter_than_half_ref_ratio_pct"]) - float(matched["health"]["pred_shorter_than_half_ref_ratio_pct"])
    repeat_delta = float(winner["health"]["repeat_prediction_ratio_pct"]) - float(matched["health"]["repeat_prediction_ratio_pct"])
    go = (
        delta >= float(args.min_delta)
        and empty_delta <= float(args.max_empty_worse)
        and short_delta <= float(args.max_short_worse)
        and repeat_delta <= float(args.max_repeat_worse)
    )
    if go:
        stage_decision = args.accept_label
    elif delta > 0.0:
        stage_decision = "review_stop"
    else:
        stage_decision = "reject_stop"
    summary = {
        "line": line_name,
        "matched_geom": float(matched["eval_geom"]),
        "winner_geom": float(winner["eval_geom"]),
        "winner_tag": safe_text(winner["tag"]),
        "winner_checkpoint_dir": safe_text(winner["checkpoint_dir"]),
        "delta_geom": delta,
        "health_deltas": {
            "empty_prediction_ratio_pct": empty_delta,
            "pred_shorter_than_half_ref_ratio_pct": short_delta,
            "repeat_prediction_ratio_pct": repeat_delta,
        },
        "thresholds": {
            "min_delta": float(args.min_delta),
            "max_empty_worse": float(args.max_empty_worse),
            "max_short_worse": float(args.max_short_worse),
            "max_repeat_worse": float(args.max_repeat_worse),
        },
        "go": go,
        "stage_decision": stage_decision,
    }
    write_json(Path(args.out_json), summary)
    title = args.title or f"{title_case(line_name)} Gate"
    md = "\n".join(
        [
            f"# {title}",
            "",
            f"- matched geom: `{summary['matched_geom']:.4f}`",
            f"- winner geom: `{summary['winner_geom']:.4f}`",
            f"- winner tag: `{summary['winner_tag']}`",
            f"- delta geom: `{summary['delta_geom']:+.4f}`",
            f"- empty delta: `{summary['health_deltas']['empty_prediction_ratio_pct']:+.2f}`",
            f"- short delta: `{summary['health_deltas']['pred_shorter_than_half_ref_ratio_pct']:+.2f}`",
            f"- repeat delta: `{summary['health_deltas']['repeat_prediction_ratio_pct']:+.2f}`",
            f"- go: `{'true' if summary['go'] else 'false'}`",
            f"- stage_decision: `{summary['stage_decision']}`",
            "",
        ]
    )
    write_text(Path(args.out_md), md)


def mix_predictions(args: argparse.Namespace, *, line_name: str) -> None:
    metadata_df = pd.read_csv(args.metadata_csv)
    routed = metadata_df.loc[
        (metadata_df["split"] == "val") & (metadata_df["is_routed_hard"] == True)
    ].copy()
    routed_ids = set(routed["parent_oare_id"].astype(str).tolist())
    baseline_df = pd.read_csv(args.baseline_csv)
    candidate_df = pd.read_csv(args.candidate_csv)
    base_id_col = _id_col(baseline_df)
    cand_id_col = _id_col(candidate_df)
    baseline_df[base_id_col] = baseline_df[base_id_col].astype(str)
    candidate_df[cand_id_col] = candidate_df[cand_id_col].astype(str)
    candidate_map = {row[cand_id_col]: safe_text(row["prediction"]) for row in candidate_df.to_dict(orient="records")}

    mixed = baseline_df.copy()
    mixed["prediction_pass_a"] = mixed["prediction"].fillna("").astype(str)
    mixed["prediction_pass_b"] = mixed[base_id_col].map(candidate_map).fillna("")
    mixed["used_pass_b"] = mixed[base_id_col].map(lambda item: item in routed_ids and item in candidate_map)
    mixed["prediction"] = mixed.apply(
        lambda row: safe_text(row["prediction_pass_b"]) if bool(row["used_pass_b"]) else safe_text(row["prediction_pass_a"]),
        axis=1,
    )
    overall = compute_eval_summary(
        predictions=mixed["prediction"].fillna("").astype(str).tolist(),
        references=mixed["reference"].fillna("").astype(str).tolist(),
        sources=mixed["source"].fillna("").astype(str).tolist() if "source" in mixed.columns else [""] * len(mixed),
        tokenizer_name=args.tokenizer_name,
        tag=args.tag,
        checkpoint_dir=safe_text(args.candidate_checkpoint_dir),
        subset_name=args.subset_name,
        eval_rows=int(len(mixed)),
    )
    routed_subset = mixed.loc[mixed["used_pass_b"] == True].copy()
    easy_subset = mixed.loc[mixed["used_pass_b"] == False].copy()
    overall["bridge"] = {
        "routed_parent_count": int(len(routed_subset)),
        "easy_parent_count": int(len(easy_subset)),
        "routed_geom": float(
            compute_eval_summary(
                predictions=routed_subset["prediction"].fillna("").astype(str).tolist(),
                references=routed_subset["reference"].fillna("").astype(str).tolist(),
                sources=routed_subset["source"].fillna("").astype(str).tolist() if "source" in routed_subset.columns else [""] * len(routed_subset),
                tokenizer_name=args.tokenizer_name,
                tag=args.tag,
                checkpoint_dir=safe_text(args.candidate_checkpoint_dir),
                subset_name="routed_only",
                eval_rows=int(len(routed_subset)),
            )["eval_geom"]
        )
        if not routed_subset.empty
        else 0.0,
        "easy_geom": float(
            compute_eval_summary(
                predictions=easy_subset["prediction"].fillna("").astype(str).tolist(),
                references=easy_subset["reference"].fillna("").astype(str).tolist(),
                sources=easy_subset["source"].fillna("").astype(str).tolist() if "source" in easy_subset.columns else [""] * len(easy_subset),
                tokenizer_name=args.tokenizer_name,
                tag=args.tag,
                checkpoint_dir=safe_text(args.candidate_checkpoint_dir),
                subset_name="easy_only",
                eval_rows=int(len(easy_subset)),
            )["eval_geom"]
        )
        if not easy_subset.empty
        else 0.0,
    }
    write_csv(Path(args.out_csv), mixed)
    write_json(Path(args.out_json), overall)
    if args.out_md:
        md = "\n".join(
            [
                f"# {args.title or f'{title_case(line_name)} Bucket Bridge'}",
                "",
                f"- mixed geom: `{overall['eval_geom']:.4f}`",
                f"- routed geom: `{overall['bridge']['routed_geom']:.4f}`",
                f"- easy geom: `{overall['bridge']['easy_geom']:.4f}`",
                f"- routed parent count: `{overall['bridge']['routed_parent_count']}`",
                f"- easy parent count: `{overall['bridge']['easy_parent_count']}`",
                "",
            ]
        )
        write_text(Path(args.out_md), md)


def promote_compare(args: argparse.Namespace, *, line_name: str) -> None:
    matched = _load_json(args.matched_json)
    candidate = _load_json(args.candidate_json)
    delta = float(candidate["eval_geom"]) - float(matched["eval_geom"])
    empty_delta = float(candidate["health"]["empty_prediction_ratio_pct"]) - float(matched["health"]["empty_prediction_ratio_pct"])
    short_delta = float(candidate["health"]["pred_shorter_than_half_ref_ratio_pct"]) - float(matched["health"]["pred_shorter_than_half_ref_ratio_pct"])
    repeat_delta = float(candidate["health"]["repeat_prediction_ratio_pct"]) - float(matched["health"]["repeat_prediction_ratio_pct"])
    go = (
        delta >= float(args.min_delta)
        and empty_delta <= float(args.max_empty_worse)
        and short_delta <= float(args.max_short_worse)
        and repeat_delta <= float(args.max_repeat_worse)
    )
    summary = {
        "line": line_name,
        "matched_geom": float(matched["eval_geom"]),
        "candidate_geom": float(candidate["eval_geom"]),
        "delta_geom": delta,
        "health_deltas": {
            "empty_prediction_ratio_pct": empty_delta,
            "pred_shorter_than_half_ref_ratio_pct": short_delta,
            "repeat_prediction_ratio_pct": repeat_delta,
        },
        "go": go,
        "stage_decision": args.accept_label if go else ("review_stop" if delta > 0.0 else "reject_stop"),
    }
    write_json(Path(args.out_json), summary)
    md = "\n".join(
        [
            f"# {args.title or f'{title_case(line_name)} Promote Compare'}",
            "",
            f"- matched geom: `{summary['matched_geom']:.4f}`",
            f"- candidate geom: `{summary['candidate_geom']:.4f}`",
            f"- delta geom: `{summary['delta_geom']:+.4f}`",
            f"- empty delta: `{summary['health_deltas']['empty_prediction_ratio_pct']:+.2f}`",
            f"- short delta: `{summary['health_deltas']['pred_shorter_than_half_ref_ratio_pct']:+.2f}`",
            f"- repeat delta: `{summary['health_deltas']['repeat_prediction_ratio_pct']:+.2f}`",
            f"- go: `{'true' if summary['go'] else 'false'}`",
            f"- stage_decision: `{summary['stage_decision']}`",
            "",
        ]
    )
    write_text(Path(args.out_md), md)


def ablation_compare(args: argparse.Namespace, *, line_name: str) -> None:
    baseline = _load_json(args.baseline_json)
    candidate = _load_json(args.candidate_json)
    delta = float(candidate["eval_geom"]) - float(baseline["eval_geom"])
    summary = {
        "line": line_name,
        "baseline_geom": float(baseline["eval_geom"]),
        "candidate_geom": float(candidate["eval_geom"]),
        "delta_geom": delta,
        "stage_decision": "candidate_clear_better" if delta >= float(args.min_delta) else "candidate_not_clear",
    }
    write_json(Path(args.out_json), summary)
    md = "\n".join(
        [
            f"# {args.title or f'{title_case(line_name)} Ablation Compare'}",
            "",
            f"- baseline geom: `{summary['baseline_geom']:.4f}`",
            f"- candidate geom: `{summary['candidate_geom']:.4f}`",
            f"- delta geom: `{summary['delta_geom']:+.4f}`",
            f"- stage_decision: `{summary['stage_decision']}`",
            "",
        ]
    )
    write_text(Path(args.out_md), md)


def main(*, line_name: str, include_ablation: bool) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_m = sub.add_parser("matched_baseline")
    ap_m.add_argument("--metadata-csv", required=True)
    ap_m.add_argument("--baseline-csv", required=True)
    ap_m.add_argument("--out-json", required=True)
    ap_m.add_argument("--out-csv", default="")
    ap_m.add_argument("--tag", required=True)
    ap_m.add_argument("--subset-name", default="routed_full")
    ap_m.add_argument("--checkpoint-dir", default="pass_a_winner")
    ap_m.add_argument("--tokenizer-name", default="google/byt5-small")
    ap_m.add_argument("--max-parents", type=int, default=0)
    ap_m.add_argument("--scope", default="routed", choices=["routed", "all_val"])

    ap_t = sub.add_parser("target_length_audit")
    ap_t.add_argument("--processed-dir", required=True)
    ap_t.add_argument("--fold", type=int, default=0)
    ap_t.add_argument("--out-json", required=True)
    ap_t.add_argument("--tag", required=True)
    ap_t.add_argument("--tokenizer-name", default="google/flan-t5-small")
    ap_t.add_argument("--max-val-parents", type=int, default=64)
    ap_t.add_argument("--max-new-tokens", type=int, default=1024)
    ap_t.add_argument("--max-target-length", type=int, default=1536)
    ap_t.add_argument("--max-over-new-pct", type=float, default=10.0)
    ap_t.add_argument("--max-over-target-pct", type=float, default=5.0)

    ap_s = sub.add_parser("synthetic_baseline")
    ap_s.add_argument("--metadata-csv", required=True)
    ap_s.add_argument("--draft-cache-csv", required=True)
    ap_s.add_argument("--out-json", required=True)
    ap_s.add_argument("--out-csv", required=True)
    ap_s.add_argument("--tag", required=True)
    ap_s.add_argument("--subset-name", default="routed_anchor64")
    ap_s.add_argument("--checkpoint-dir", default="pass_a_synthetic")
    ap_s.add_argument("--tokenizer-name", default="google/flan-t5-small")
    ap_s.add_argument("--max-parents", type=int, default=64)
    ap_s.add_argument("--scope", default="routed", choices=["routed", "all_val"])
    ap_s.add_argument("--mode", default="pred", choices=["pred", "oracle"])
    ap_s.add_argument("--text-field", default="draft_prediction")
    ap_s.add_argument("--sanitize-chunks", action="store_true")
    ap_s.add_argument("--sanitize-max-words", type=int, default=18)
    ap_s.add_argument("--sanitize-tail-words", type=int, default=6)
    ap_s.add_argument("--dedup-strategy", default="none", choices=["none", "consecutive_exact", "global_exact"])

    ap_e = sub.add_parser("eval_model")
    ap_e.add_argument("--config", required=True)
    ap_e.add_argument("--fold", type=int, default=0)
    ap_e.add_argument("--checkpoint-dir", required=True)
    ap_e.add_argument("--out-json", required=True)
    ap_e.add_argument("--out-csv", required=True)
    ap_e.add_argument("--tag", required=True)
    ap_e.add_argument("--subset-name", default="routed_full")
    ap_e.add_argument("--max-val-parents", type=int, default=0)
    ap_e.add_argument("--predict-batch-size", type=int, default=16)
    ap_e.add_argument("--num-beams", type=int, default=4)
    ap_e.add_argument("--length-penalty", type=float, default=0.7)
    ap_e.add_argument("--max-new-tokens", type=int, default=384)
    ap_e.add_argument("--no-repeat-ngram-size", type=int, default=0)

    ap_g = sub.add_parser("gate_report")
    ap_g.add_argument("--matched-json", required=True)
    ap_g.add_argument("--candidate-jsons", nargs="+", required=True)
    ap_g.add_argument("--out-json", required=True)
    ap_g.add_argument("--out-md", required=True)
    ap_g.add_argument("--title", default="")
    ap_g.add_argument("--min-delta", type=float, default=0.25)
    ap_g.add_argument("--max-empty-worse", type=float, default=0.0)
    ap_g.add_argument("--max-short-worse", type=float, default=2.0)
    ap_g.add_argument("--max-repeat-worse", type=float, default=2.0)
    ap_g.add_argument("--accept-label", default="accept_to_wlite")

    ap_mix = sub.add_parser("mix_predictions")
    ap_mix.add_argument("--metadata-csv", required=True)
    ap_mix.add_argument("--baseline-csv", required=True)
    ap_mix.add_argument("--candidate-csv", required=True)
    ap_mix.add_argument("--out-json", required=True)
    ap_mix.add_argument("--out-csv", required=True)
    ap_mix.add_argument("--out-md", default="")
    ap_mix.add_argument("--title", default="")
    ap_mix.add_argument("--tag", required=True)
    ap_mix.add_argument("--subset-name", default="mixed_full")
    ap_mix.add_argument("--candidate-checkpoint-dir", default="")
    ap_mix.add_argument("--tokenizer-name", default="google/byt5-small")

    ap_p = sub.add_parser("promote_compare")
    ap_p.add_argument("--matched-json", required=True)
    ap_p.add_argument("--candidate-json", required=True)
    ap_p.add_argument("--out-json", required=True)
    ap_p.add_argument("--out-md", required=True)
    ap_p.add_argument("--title", default="")
    ap_p.add_argument("--min-delta", type=float, default=0.0)
    ap_p.add_argument("--max-empty-worse", type=float, default=0.0)
    ap_p.add_argument("--max-short-worse", type=float, default=2.0)
    ap_p.add_argument("--max-repeat-worse", type=float, default=2.0)
    ap_p.add_argument("--accept-label", default="review_win_taskform_full")

    if include_ablation:
        ap_a = sub.add_parser("ablation_compare")
        ap_a.add_argument("--baseline-json", required=True)
        ap_a.add_argument("--candidate-json", required=True)
        ap_a.add_argument("--out-json", required=True)
        ap_a.add_argument("--out-md", required=True)
        ap_a.add_argument("--title", default="")
        ap_a.add_argument("--min-delta", type=float, default=0.25)

    args = ap.parse_args()
    if args.cmd == "matched_baseline":
        matched_baseline(args, line_name=line_name)
    elif args.cmd == "target_length_audit":
        target_length_audit(args, line_name=line_name)
    elif args.cmd == "synthetic_baseline":
        synthetic_baseline(args, line_name=line_name)
    elif args.cmd == "eval_model":
        eval_model(args, line_name=line_name)
    elif args.cmd == "gate_report":
        gate_report(args, line_name=line_name)
    elif args.cmd == "mix_predictions":
        mix_predictions(args, line_name=line_name)
    elif args.cmd == "promote_compare":
        promote_compare(args, line_name=line_name)
    elif args.cmd == "ablation_compare" and include_ablation:
        ablation_compare(args, line_name=line_name)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")
