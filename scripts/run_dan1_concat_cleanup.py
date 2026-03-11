#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from fusion_flow_common import (
    compute_eval_summary,
    normalize_whitespace,
    safe_text,
    write_csv,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

FORMULA_PATTERNS = (
    "Seal of",
    "Sealed by",
    "send me the silver",
    "will pay the silver",
    "not paid the silver",
)


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _select_subset_meta(metadata_df: pd.DataFrame, *, scope: str, max_parents: int) -> pd.DataFrame:
    if str(scope).strip().lower() == "all_val":
        subset = metadata_df.loc[metadata_df["split"] == "val"].copy()
    else:
        subset = metadata_df.loc[
            (metadata_df["split"] == "val") & (metadata_df["is_routed_hard"] == True)
        ].copy()
    subset["route_rank"] = subset["route_rank"].fillna(999999).astype(int)
    subset = subset.sort_values(["route_rank", "parent_oare_id"]).reset_index(drop=True)
    if int(max_parents) > 0:
        subset = subset.head(int(max_parents)).reset_index(drop=True)
    return subset


def _tokenize_words(text: str) -> list[str]:
    return [token for token in normalize_whitespace(text).split() if token]


def _repair_gap_markers(text: str) -> str:
    value = safe_text(text)
    value = re.sub(r"(?<!<)\bgap>", "<gap>", value, flags=re.IGNORECASE)
    value = re.sub(r"<gap>\s*<gap>", "<gap>", value)
    return normalize_whitespace(value)


def _clamp_consecutive_repeated_spans(
    text: str,
    *,
    max_span: int = 12,
    max_occurrences: int = 2,
) -> str:
    words = _tokenize_words(text)
    if not words:
        return ""
    out: list[str] = []
    i = 0
    n_words = len(words)
    while i < n_words:
        matched = False
        max_try = min(int(max_span), (n_words - i) // 2)
        for span in range(max_try, 0, -1):
            pattern = words[i : i + span]
            reps = 1
            while i + ((reps + 1) * span) <= n_words and words[i + (reps * span) : i + ((reps + 1) * span)] == pattern:
                reps += 1
            if reps > int(max_occurrences):
                for _ in range(int(max_occurrences)):
                    out.extend(pattern)
                i += reps * span
                matched = True
                break
        if matched:
            continue
        out.append(words[i])
        i += 1
    return " ".join(out).strip()


def _collapse_formula_loops(text: str, *, max_repeats: int = 3) -> str:
    value = normalize_whitespace(text)
    if not value:
        return ""
    words = value.split()
    spans = [2, 3, 4]
    for stem in FORMULA_PATTERNS:
        stem_words = stem.split()
        stem_len = len(stem_words)
        idx = 0
        new_words: list[str] = []
        while idx < len(words):
            if words[idx : idx + stem_len] != stem_words:
                new_words.append(words[idx])
                idx += 1
                continue
            matched_any = False
            for span in spans:
                if idx + (span * stem_len) > len(words):
                    continue
                unit = words[idx : idx + span]
                reps = 1
                while idx + ((reps + 1) * len(unit)) <= len(words) and words[idx + (reps * len(unit)) : idx + ((reps + 1) * len(unit))] == unit:
                    reps += 1
                if reps > max_repeats:
                    for _ in range(max_repeats):
                        new_words.extend(unit)
                    idx += reps * len(unit)
                    matched_any = True
                    break
            if not matched_any:
                new_words.extend(words[idx : idx + stem_len])
                idx += stem_len
        words = new_words
    return normalize_whitespace(" ".join(words))


def _formula_count(text: str) -> int:
    value = safe_text(text)
    return sum(value.count(stem) for stem in FORMULA_PATTERNS)


def _internal_repeat_score(text: str, *, ngram_size: int = 3, min_count: int = 4) -> int:
    words = _tokenize_words(text)
    if len(words) < max(1, ngram_size):
        return 0
    grams = Counter(
        tuple(words[idx : idx + ngram_size])
        for idx in range(0, len(words) - ngram_size + 1)
    )
    return int(sum(max(0, count - (min_count - 1)) for count in grams.values() if count >= min_count))


def _candidate_score(raw_text: str, candidate_text: str) -> tuple[float, dict[str, Any]]:
    raw_words = max(1, len(_tokenize_words(raw_text)))
    cand_words = len(_tokenize_words(candidate_text))
    ratio = float(cand_words) / float(raw_words)
    gap_bad = candidate_text.count("gap>")
    internal_repeat = _internal_repeat_score(candidate_text)
    formula_count = _formula_count(candidate_text)
    label_bad = 1 if re.search(r"\bc\d+:", candidate_text) else 0
    short_penalty = max(0.0, 0.82 - ratio) * 120.0
    score = (
        short_penalty
        + (18.0 * float(gap_bad))
        + (4.0 * float(internal_repeat))
        + (0.8 * float(max(0, formula_count - 18)))
        + (40.0 * float(label_bad))
    )
    features = {
        "word_ratio_vs_raw": ratio,
        "gap_bad_count": int(gap_bad),
        "internal_repeat_score": int(internal_repeat),
        "formula_count": int(formula_count),
        "label_bad": int(label_bad),
        "short_penalty": float(short_penalty),
    }
    return float(score), features


def _bucket_name(meta_row: dict[str, Any]) -> str:
    chunk_total = int(meta_row["chunk_total"])
    parent_ref_tok = int(meta_row["parent_ref_tok"])
    marker_count = int(meta_row["marker_count"])
    if 4 <= chunk_total <= 6:
        return "chunk4_6"
    if chunk_total >= 7:
        return "chunk7plus"
    if 2 <= chunk_total <= 3 and (parent_ref_tok >= 129 or marker_count >= 2):
        return "chunk2_3_long_or_tag"
    return "other_routed"


def _build_candidates_for_parent(group_df: pd.DataFrame) -> dict[str, str]:
    ordered = group_df.sort_values("chunk_index")
    raw_chunks = [normalize_whitespace(text) for text in ordered["draft_prediction"].fillna("").astype(str).tolist() if normalize_whitespace(text)]
    raw = normalize_whitespace(" ".join(raw_chunks))

    gapfix_chunks = [_repair_gap_markers(text) for text in raw_chunks]
    gapfix = normalize_whitespace(" ".join(gapfix_chunks))

    looptrim_chunks = [
        _collapse_formula_loops(
            _clamp_consecutive_repeated_spans(_repair_gap_markers(text), max_span=12, max_occurrences=2),
            max_repeats=2,
        )
        for text in raw_chunks
    ]
    looptrim = normalize_whitespace(
        _collapse_formula_loops(
            _clamp_consecutive_repeated_spans(" ".join(looptrim_chunks), max_span=12, max_occurrences=2),
            max_repeats=2,
        )
    )

    dedup_chunks: list[str] = []
    for chunk_text in looptrim_chunks:
        if not chunk_text:
            continue
        if dedup_chunks and dedup_chunks[-1] == chunk_text:
            continue
        dedup_chunks.append(chunk_text)
    looptrim_chunkdedup = normalize_whitespace(
        _collapse_formula_loops(
            _clamp_consecutive_repeated_spans(" ".join(dedup_chunks), max_span=12, max_occurrences=2),
            max_repeats=2,
        )
    )
    return {
        "raw_concat": raw,
        "gapfix_concat": gapfix,
        "looptrim_concat": looptrim,
        "looptrim_chunkdedup_concat": looptrim_chunkdedup,
    }


def _build_subset_frame(
    *,
    metadata_df: pd.DataFrame,
    draft_cache_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    scope: str,
    max_parents: int,
) -> pd.DataFrame:
    subset_meta = _select_subset_meta(metadata_df, scope=scope, max_parents=max_parents)
    baseline_df = baseline_df.copy()
    baseline_id_col = "oare_id" if "oare_id" in baseline_df.columns else "id"
    baseline_df[baseline_id_col] = baseline_df[baseline_id_col].astype(str)
    baseline_index = baseline_df.set_index(baseline_id_col)
    rows: list[dict[str, Any]] = []
    for meta in subset_meta.to_dict(orient="records"):
        parent_id = safe_text(meta["parent_oare_id"])
        base_row = baseline_index.loc[parent_id]
        group_df = draft_cache_df.loc[draft_cache_df["parent_oare_id"].astype(str) == parent_id].copy()
        candidates = _build_candidates_for_parent(group_df)
        rerank_features: dict[str, dict[str, Any]] = {}
        rerank_scored: list[tuple[float, str]] = []
        for name, text in candidates.items():
            score, features = _candidate_score(candidates["raw_concat"], text)
            rerank_features[name] = features
            rerank_scored.append((score, name))
        rerank_scored.sort(key=lambda item: (item[0], 0 if item[1] == "raw_concat" else 1, item[1]))
        rerank_name = rerank_scored[0][1]
        rows.append(
            {
                "oare_id": parent_id,
                "parent_oare_id": parent_id,
                "route_rank": int(meta["route_rank"]),
                "route_score": float(meta["route_score"]),
                "route_reason": safe_text(meta["route_reason"]),
                "orig_chunk_total": int(meta["chunk_total"]),
                "parent_ref_tok": int(meta["parent_ref_tok"]),
                "marker_count": int(meta["marker_count"]),
                "bucket": _bucket_name(meta),
                "source": safe_text(base_row["source"]),
                "reference": safe_text(base_row["reference"]),
                "pass_a_prediction": safe_text(base_row["prediction"]),
                **candidates,
                "rerank_choice": rerank_name,
                "rerank_prediction": candidates[rerank_name],
                "rerank_score": float(rerank_scored[0][0]),
                "raw_internal_repeat_score": int(_internal_repeat_score(candidates["raw_concat"])),
                "rerank_internal_repeat_score": int(_internal_repeat_score(candidates[rerank_name])),
                "raw_formula_count": int(_formula_count(candidates["raw_concat"])),
                "rerank_formula_count": int(_formula_count(candidates[rerank_name])),
                "raw_word_count": int(len(_tokenize_words(candidates["raw_concat"]))),
                "rerank_word_count": int(len(_tokenize_words(candidates[rerank_name]))),
                "raw_gap_bad_count": int(candidates["raw_concat"].count("gap>")),
                "rerank_gap_bad_count": int(candidates[rerank_name].count("gap>")),
                "rerank_features_json": str(rerank_features[rerank_name]),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_candidates(frame: pd.DataFrame, *, tokenizer_name: str, checkpoint_tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in [
        "raw_concat",
        "gapfix_concat",
        "looptrim_concat",
        "looptrim_chunkdedup_concat",
        "rerank_prediction",
        "pass_a_prediction",
    ]:
        out[col] = compute_eval_summary(
            predictions=frame[col].fillna("").astype(str).tolist(),
            references=frame["reference"].fillna("").astype(str).tolist(),
            sources=frame["source"].fillna("").astype(str).tolist(),
            tokenizer_name=tokenizer_name,
            tag=f"{checkpoint_tag}_{col}",
            checkpoint_dir=checkpoint_tag,
            subset_name=checkpoint_tag,
            eval_rows=int(len(frame)),
        )
    return out


def _bucket_metrics(frame: pd.DataFrame, *, candidate_col: str, tokenizer_name: str, tag_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("bucket", sort=False):
        summary = compute_eval_summary(
            predictions=group[candidate_col].fillna("").astype(str).tolist(),
            references=group["reference"].fillna("").astype(str).tolist(),
            sources=group["source"].fillna("").astype(str).tolist(),
            tokenizer_name=tokenizer_name,
            tag=f"{tag_prefix}_{bucket}_{candidate_col}",
            checkpoint_dir=tag_prefix,
            subset_name=bucket,
            eval_rows=int(len(group)),
        )
        rows.append(
            {
                "bucket": bucket,
                "candidate": candidate_col,
                "n": int(len(group)),
                "geom": float(summary["eval_geom"]),
                "bleu": float(summary["eval_bleu"]),
                "chrfpp": float(summary["eval_chrfpp"]),
                "short_pct": float(summary["health"]["pred_shorter_than_half_ref_ratio_pct"]),
                "internal_repeat_trigram_pct": float(summary["health"].get("internal_repeat_trigram_ratio_pct", 0.0)),
                "avg_raw_formula_count": float(group["raw_formula_count"].mean()),
                "avg_candidate_formula_count": float(group["rerank_formula_count"].mean())
                if candidate_col == "rerank_prediction"
                else float(group["raw_formula_count"].mean()),
            }
        )
    return rows


def _pick_metric(summary_map: dict[str, Any], key: str) -> float:
    return float(summary_map[key]["eval_geom"])


def _write_bucket_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# dan-1 Concat Bucket Audit",
        "",
        "| bucket | candidate | n | geom | bleu | chrfpp | short_pct | internal_repeat_trigram_pct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bucket']} | {row['candidate']} | {row['n']} | "
            f"{row['geom']:.4f} | {row['bleu']:.4f} | {row['chrfpp']:.4f} | "
            f"{row['short_pct']:.2f} | {row['internal_repeat_trigram_pct']:.2f} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def _mix_with_official(
    *,
    baseline_df: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    candidate_col: str,
    tokenizer_name: str,
) -> dict[str, Any]:
    baseline = baseline_df.copy()
    id_col = "oare_id" if "oare_id" in baseline.columns else "id"
    baseline[id_col] = baseline[id_col].astype(str)
    candidate_map = {
        safe_text(row["oare_id"]): safe_text(row[candidate_col])
        for row in candidate_frame.to_dict(orient="records")
    }
    mixed = baseline.copy()
    mixed["prediction_pass_a"] = mixed["prediction"].fillna("").astype(str)
    mixed["prediction_pass_b"] = mixed[id_col].map(candidate_map).fillna("")
    mixed["used_pass_b"] = mixed[id_col].map(lambda item: item in candidate_map)
    mixed["prediction"] = mixed.apply(
        lambda row: safe_text(row["prediction_pass_b"]) if bool(row["used_pass_b"]) else safe_text(row["prediction_pass_a"]),
        axis=1,
    )
    summary = compute_eval_summary(
        predictions=mixed["prediction"].fillna("").astype(str).tolist(),
        references=mixed["reference"].fillna("").astype(str).tolist(),
        sources=mixed["source"].fillna("").astype(str).tolist(),
        tokenizer_name=tokenizer_name,
        tag=f"official_mix_{candidate_col}",
        checkpoint_dir=candidate_col,
        subset_name="official_full_mix",
        eval_rows=int(len(mixed)),
        extra={
            "routed_parent_count": int(mixed["used_pass_b"].sum()),
            "easy_parent_count": int((~mixed["used_pass_b"]).sum()),
        },
    )
    return {"summary": summary, "predictions": mixed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default="data/taskform_dan1_edit_v1_fold0/parent_metadata.csv")
    ap.add_argument("--draft-cache-csv", default="data/taskform_dan1_edit_v1_fold0/draft_cache_pred.csv")
    ap.add_argument(
        "--baseline-csv",
        default=(
            "runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/"
            "val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv"
        ),
    )
    ap.add_argument("--out-dir", default="reports/taskform_dan1_b1_b2_b4")
    ap.add_argument("--tokenizer-name", default="google/flan-t5-small")
    args = ap.parse_args()

    metadata_csv = _resolve_path(args.metadata_csv, REPO_ROOT / "data" / "taskform_dan1_edit_v1_fold0" / "parent_metadata.csv")
    draft_cache_csv = _resolve_path(args.draft_cache_csv, REPO_ROOT / "data" / "taskform_dan1_edit_v1_fold0" / "draft_cache_pred.csv")
    baseline_csv = _resolve_path(
        args.baseline_csv,
        REPO_ROOT
        / "runs"
        / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0"
        / "diagnostics"
        / "val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv",
    )
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_csv)
    draft_cache_df = pd.read_csv(draft_cache_csv)
    baseline_df = pd.read_csv(baseline_csv)

    anchor64 = _build_subset_frame(
        metadata_df=metadata_df,
        draft_cache_df=draft_cache_df,
        baseline_df=baseline_df,
        scope="routed",
        max_parents=64,
    )
    routed_full = _build_subset_frame(
        metadata_df=metadata_df,
        draft_cache_df=draft_cache_df,
        baseline_df=baseline_df,
        scope="routed",
        max_parents=0,
    )

    anchor64_eval = _evaluate_candidates(anchor64, tokenizer_name=args.tokenizer_name, checkpoint_tag="anchor64")
    routed_full_eval = _evaluate_candidates(routed_full, tokenizer_name=args.tokenizer_name, checkpoint_tag="routed_full")

    anchor64_compare = {
        name: float(summary["eval_geom"])
        for name, summary in anchor64_eval.items()
    }
    routed_full_compare = {
        name: float(summary["eval_geom"])
        for name, summary in routed_full_eval.items()
    }

    bucket_rows = (
        _bucket_metrics(routed_full, candidate_col="raw_concat", tokenizer_name=args.tokenizer_name, tag_prefix="bucket")
        + _bucket_metrics(routed_full, candidate_col="rerank_prediction", tokenizer_name=args.tokenizer_name, tag_prefix="bucket")
    )

    official_mix = _mix_with_official(
        baseline_df=baseline_df,
        candidate_frame=routed_full,
        candidate_col="rerank_prediction",
        tokenizer_name=args.tokenizer_name,
    )

    write_csv(out_dir / "anchor64_predictions.csv", anchor64)
    write_csv(out_dir / "routed_full_predictions.csv", routed_full)
    write_json(out_dir / "anchor64_eval.json", {key: value for key, value in anchor64_eval.items()})
    write_json(out_dir / "routed_full_eval.json", {key: value for key, value in routed_full_eval.items()})
    write_json(out_dir / "anchor64_compare.json", anchor64_compare)
    write_json(out_dir / "routed_full_compare.json", routed_full_compare)
    write_json(out_dir / "bucket_audit.json", {"rows": bucket_rows})
    write_csv(out_dir / "bucket_audit.csv", pd.DataFrame(bucket_rows))
    write_text(
        out_dir / "compare.md",
        "\n".join(
            [
                "# dan-1 Concat Cleanup Compare",
                "",
                "## anchor64",
                *(f"- {name}: `{value:.4f}`" for name, value in anchor64_compare.items()),
                "",
                "## routed_full",
                *(f"- {name}: `{value:.4f}`" for name, value in routed_full_compare.items()),
                "",
            ]
        )
        + "\n",
    )
    _write_bucket_md(out_dir / "bucket_audit.md", bucket_rows)
    write_json(out_dir / "official_mix_rerank_summary.json", official_mix["summary"])
    write_csv(out_dir / "official_mix_rerank_predictions.csv", official_mix["predictions"])

    overall = {
        "anchor64": anchor64_compare,
        "routed_full": routed_full_compare,
        "official_mix_rerank_geom": float(official_mix["summary"]["eval_geom"]),
        "official_mix_rerank_bleu": float(official_mix["summary"]["eval_bleu"]),
        "official_mix_rerank_chrfpp": float(official_mix["summary"]["eval_chrfpp"]),
        "rerank_choice_counts_anchor64": anchor64["rerank_choice"].value_counts().to_dict(),
        "rerank_choice_counts_routed_full": routed_full["rerank_choice"].value_counts().to_dict(),
    }
    write_json(out_dir / "summary.json", overall)
    print(f"OK: wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
