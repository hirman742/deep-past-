from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DIGIT_PATTERN = re.compile(r"\d+")


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _decode_delimiter(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _parse_delimiters(raw: str, fallback: list[str]) -> list[str]:
    if not str(raw or "").strip():
        return fallback
    tokens = [_decode_delimiter(x.strip()) for x in str(raw).split(",") if x.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out or fallback


def _split_source(text: str, delimiters: list[str], line_number_regex: str) -> list[str]:
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return []
    escaped = [re.escape(x) for x in delimiters if x]
    pattern = "|".join(escaped) if escaped else "\n+"
    chunks = [x.strip() for x in re.split(pattern, value) if x and x.strip()]
    if not chunks:
        return []
    if not line_number_regex.strip():
        return chunks
    line_re = re.compile(line_number_regex)
    out: list[str] = []
    for chunk in chunks:
        parts = [x.strip() for x in line_re.split(chunk) if x and x.strip()]
        if parts:
            out.extend(parts)
        else:
            out.append(chunk)
    return [x for x in out if x]


def _split_target(text: str, sentence_regex: str) -> list[str]:
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return []
    regex = sentence_regex.strip() or r"(?:\n+|(?<=[\.\!\?;:])\s+)"
    parts = [x.strip() for x in re.split(regex, value) if x and x.strip()]
    return parts


def _pair_cost(
    *,
    source_text: str,
    target_text: str,
    alpha: float,
    merge_penalty: float,
) -> float:
    src_len = max(1, len(source_text))
    tgt_len = max(1, len(target_text))
    ratio = float(tgt_len) / float(src_len)
    cost = abs(math.log(max(1e-6, ratio / max(1e-6, alpha))))
    return float(cost + merge_penalty)


def _gale_church_align(
    *,
    source_segments: list[str],
    target_segments: list[str],
    alpha: float,
    merge_penalty: float,
) -> list[dict[str, Any]] | None:
    n_src = len(source_segments)
    n_tgt = len(target_segments)
    if n_src == 0 or n_tgt == 0:
        return None

    inf = 1e12
    dp = np.full((n_src + 1, n_tgt + 1), inf, dtype=np.float64)
    back: list[list[tuple[int, int, int, int, float] | None]] = [[None] * (n_tgt + 1) for _ in range(n_src + 1)]
    dp[0, 0] = 0.0
    transitions = [
        (1, 1, 0.0, "1:1"),
        (1, 2, float(merge_penalty), "1:2"),
        (2, 1, float(merge_penalty), "2:1"),
    ]

    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            base_cost = float(dp[i, j])
            if base_cost >= inf:
                continue
            for step_src, step_tgt, merge_cost, _ in transitions:
                ni = i + step_src
                nj = j + step_tgt
                if ni > n_src or nj > n_tgt:
                    continue
                src_text = " ".join(source_segments[i:ni]).strip()
                tgt_text = " ".join(target_segments[j:nj]).strip()
                if not src_text or not tgt_text:
                    continue
                step_cost = _pair_cost(
                    source_text=src_text,
                    target_text=tgt_text,
                    alpha=alpha,
                    merge_penalty=merge_cost,
                )
                cand = base_cost + step_cost
                if cand < float(dp[ni, nj]):
                    dp[ni, nj] = cand
                    back[ni][nj] = (i, j, step_src, step_tgt, step_cost)

    if back[n_src][n_tgt] is None:
        return None

    out: list[dict[str, Any]] = []
    i = n_src
    j = n_tgt
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            return None
        pi, pj, step_src, step_tgt, step_cost = prev
        src_text = " ".join(source_segments[pi:i]).strip()
        tgt_text = " ".join(target_segments[pj:j]).strip()
        out.append(
            {
                "source": src_text,
                "target": tgt_text,
                "src_span": int(step_src),
                "tgt_span": int(step_tgt),
                "align_type": f"{step_src}:{step_tgt}",
                "align_cost": float(step_cost),
            }
        )
        i, j = pi, pj
    out.reverse()
    return out


def _sample_indices(
    rng: np.random.Generator,
    *,
    total: int,
    take: int,
    allow_replacement: bool,
) -> tuple[np.ndarray, bool]:
    if take <= 0 or total <= 0:
        return np.asarray([], dtype=np.int64), False
    if take <= total:
        return rng.choice(total, size=take, replace=False), False
    if bool(allow_replacement):
        return rng.choice(total, size=take, replace=True), True
    return rng.choice(total, size=total, replace=False), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/byt5_small_lora_chunked_stage1_r8_qv.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--input-folds", default="")
    ap.add_argument("--output-train", default="")
    ap.add_argument("--output-folds", default="")
    ap.add_argument("--report-json", default="")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--source-delimiters", default="\\n,;,|")
    ap.add_argument("--source-line-number-regex", default=r"(?:^|\s)\d+\.\s+")
    ap.add_argument("--target-sentence-regex", default=r"(?:\n+|(?<=[\.\!\?;:])\s+)")
    ap.add_argument("--max-source-segments", type=int, default=80)
    ap.add_argument("--max-target-segments", type=int, default=80)
    ap.add_argument("--max-align-cost", type=float, default=1.2)
    ap.add_argument("--merge-penalty", type=float, default=0.25)
    ap.add_argument("--min-source-tokens", type=int, default=3)
    ap.add_argument("--min-target-tokens", type=int, default=3)
    ap.add_argument("--max-source-tokens", type=int, default=256)
    ap.add_argument("--max-target-tokens", type=int, default=256)
    ap.add_argument("--min-length-ratio", type=float, default=0.4)
    ap.add_argument("--max-length-ratio", type=float, default=2.5)
    ap.add_argument("--require-digit-coverage", action="store_true")
    ap.add_argument("--mix-ratio", type=float, default=0.5)
    ap.add_argument("--max-extra-rows", type=int, default=0)
    ap.add_argument("--allow-replacement", dest="allow_replacement", action="store_true")
    ap.add_argument("--no-allow-replacement", dest="allow_replacement", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="")
    ap.set_defaults(allow_replacement=False)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "byt5_small_lora_chunked_stage1_r8_qv.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")

    train_path = _resolve_path(args.input_train, processed_dir / "train_proc.csv")
    folds_path = _resolve_path(args.input_folds, processed_dir / "folds.csv")
    output_train = _resolve_path(args.output_train, processed_dir / "train_proc_shortalign_gc.csv")
    output_folds = _resolve_path(args.output_folds, processed_dir / "folds_shortalign_gc.csv")
    report_json = _resolve_path(args.report_json, processed_dir / "short_align_gc_report.json")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train csv: {train_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing folds csv: {folds_path}")

    source_delimiters = _parse_delimiters(args.source_delimiters, fallback=["\n", ";", "|"])
    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    if args.source_col not in train_df.columns or args.target_col not in train_df.columns:
        raise KeyError(f"Missing source/target columns: {args.source_col}, {args.target_col}")
    if "oare_id" not in train_df.columns or "oare_id" not in folds_df.columns:
        raise KeyError("Both train and folds files must include oare_id")

    model_name = str(args.model_name or model_cfg.get("name", "google/byt5-small"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use dataset-level length ratio as alpha.
    ratio_series = (
        (train_df[args.target_col].fillna("").astype(str).str.len() + 1.0)
        / (train_df[args.source_col].fillna("").astype(str).str.len() + 1.0)
    )
    alpha = float(ratio_series.median()) if not ratio_series.empty else 1.0
    alpha = float(min(max(alpha, 0.2), 5.0))

    fold_map = folds_df.set_index("oare_id").to_dict(orient="index")

    candidates: list[dict[str, Any]] = []
    align_type_stats: dict[str, int] = {}
    parent_mode_stats: dict[str, int] = {}

    for _, row in train_df.iterrows():
        source = str(row.get(args.source_col, "") or "")
        target = str(row.get(args.target_col, "") or "")
        src_segments = _split_source(source, source_delimiters, str(args.source_line_number_regex))
        tgt_segments = _split_target(target, str(args.target_sentence_regex))
        if not src_segments or not tgt_segments:
            continue
        if len(src_segments) > int(args.max_source_segments) or len(tgt_segments) > int(args.max_target_segments):
            continue

        aligned = _gale_church_align(
            source_segments=src_segments,
            target_segments=tgt_segments,
            alpha=alpha,
            merge_penalty=float(args.merge_penalty),
        )
        if not aligned:
            continue

        base_id = str(row["oare_id"])
        parent_id = str(row.get("parent_oare_id", base_id))
        parent_mode_stats["gale_church"] = int(parent_mode_stats.get("gale_church", 0) + 1)
        seg_total = len(aligned)
        for seg_idx, pair in enumerate(aligned):
            if float(pair["align_cost"]) > float(args.max_align_cost):
                continue
            source_pair = str(pair["source"]).strip()
            target_pair = str(pair["target"]).strip()
            if not source_pair or not target_pair:
                continue
            src_tok = len(tokenizer.encode(source_pair, add_special_tokens=True))
            tgt_tok = len(tokenizer.encode(target_pair, add_special_tokens=True))
            if src_tok < int(args.min_source_tokens) or tgt_tok < int(args.min_target_tokens):
                continue
            if src_tok > int(args.max_source_tokens) or tgt_tok > int(args.max_target_tokens):
                continue
            ratio = float(tgt_tok) / float(max(1, src_tok))
            if ratio < float(args.min_length_ratio) or ratio > float(args.max_length_ratio):
                continue
            if bool(args.require_digit_coverage):
                src_digits = set(DIGIT_PATTERN.findall(source_pair))
                if src_digits:
                    tgt_digits = set(DIGIT_PATTERN.findall(target_pair))
                    if not tgt_digits:
                        continue

            align_type = str(pair["align_type"])
            align_type_stats[align_type] = int(align_type_stats.get(align_type, 0) + 1)
            candidates.append(
                {
                    "base_id": base_id,
                    "parent_id": parent_id,
                    "segment_index": int(seg_idx),
                    "segment_total": int(seg_total),
                    "source": source_pair,
                    "target": target_pair,
                    "align_type": align_type,
                    "align_cost": float(pair["align_cost"]),
                    "row": row.to_dict(),
                }
            )

    if not candidates:
        output_train.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_train, index=False)
        folds_df.to_csv(output_folds, index=False)
        report = {
            "config_path": str(cfg_path),
            "input_train": str(train_path),
            "input_folds": str(folds_path),
            "output_train": str(output_train),
            "output_folds": str(output_folds),
            "rows_input_train": int(len(train_df)),
            "rows_input_folds": int(len(folds_df)),
            "candidates_before_token_filter": 0,
            "rows_extra_pool": 0,
            "rows_extra_selected": 0,
            "rows_output_train": int(len(train_df)),
            "rows_output_folds": int(len(folds_df)),
            "mix_ratio": float(args.mix_ratio),
            "requested_extra_rows": 0,
            "effective_extra_rows": 0,
            "allow_replacement": bool(args.allow_replacement),
            "used_replacement": False,
            "alignment_alpha": float(alpha),
            "candidate_parent_rows_by_mode": parent_mode_stats,
            "candidate_rows_by_align_type": align_type_stats,
        }
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("WARN: no gale-church candidates found; wrote passthrough outputs.")
        print(f"OK: wrote {output_train}")
        print(f"OK: wrote {output_folds}")
        print(f"OK: wrote {report_json}")
        return

    extra_rows: list[dict[str, Any]] = []
    extra_folds: list[dict[str, Any]] = []
    skipped_missing_fold = 0
    for item in candidates:
        base_id = str(item["base_id"])
        seg_idx = int(item["segment_index"])
        seg_total = int(item["segment_total"])
        new_id = f"{base_id}__gc{seg_idx + 1}of{seg_total}"
        out_row = dict(item["row"])
        out_row["oare_id"] = new_id
        out_row[args.source_col] = item["source"]
        out_row[args.target_col] = item["target"]
        out_row["parent_oare_id"] = str(item["parent_id"])
        out_row["chunk_index"] = seg_idx
        out_row["chunk_total"] = seg_total
        out_row["is_chunk"] = True
        out_row["chunk_mode"] = "short_aligned_gale_church"
        out_row["is_short_aligned"] = True
        out_row["short_align_mode"] = "gale_church"
        out_row["source_oare_id"] = base_id
        out_row["align_type"] = str(item["align_type"])
        out_row["align_cost"] = float(item["align_cost"])
        extra_rows.append(out_row)

        if base_id not in fold_map:
            skipped_missing_fold += 1
            continue
        fold_row = dict(fold_map[base_id])
        fold_row["oare_id"] = new_id
        if "parent_oare_id" not in fold_row:
            fold_row["parent_oare_id"] = str(item["parent_id"])
        fold_row["chunk_index"] = seg_idx
        fold_row["chunk_total"] = seg_total
        fold_row["chunk_mode"] = "short_aligned_gale_church"
        fold_row["short_align_mode"] = "gale_church"
        fold_row["align_type"] = str(item["align_type"])
        extra_folds.append(fold_row)

    extra_df = pd.DataFrame(extra_rows)
    extra_folds_df = pd.DataFrame(extra_folds)
    if extra_df.empty or extra_folds_df.empty:
        raise ValueError("No gale-church rows left after fold mapping.")

    extra_fold_map = extra_folds_df.set_index("oare_id").to_dict(orient="index")
    aligned_ids = [x for x in extra_df["oare_id"].astype(str).tolist() if x in extra_fold_map]
    if not aligned_ids:
        raise ValueError("No gale-church aligned ids with fold mapping.")
    aligned_extra_df = extra_df.set_index("oare_id").loc[aligned_ids].reset_index()

    requested_extra_rows = max(0, int(round(float(args.mix_ratio) * float(len(train_df)))))
    if int(args.max_extra_rows) > 0:
        requested_extra_rows = min(requested_extra_rows, int(args.max_extra_rows))

    rng = np.random.default_rng(int(args.seed))
    selected_indices, used_replacement = _sample_indices(
        rng,
        total=int(len(aligned_extra_df)),
        take=int(requested_extra_rows),
        allow_replacement=bool(args.allow_replacement),
    )
    selected_extra_df = (
        aligned_extra_df.iloc[selected_indices].reset_index(drop=True)
        if requested_extra_rows > 0
        else aligned_extra_df.iloc[0:0].copy()
    )
    selected_folds_df = (
        pd.DataFrame([{"oare_id": oid, **extra_fold_map[str(oid)]} for oid in selected_extra_df["oare_id"].astype(str).tolist()])
        if not selected_extra_df.empty
        else folds_df.iloc[0:0].copy()
    )

    output_train_df = pd.concat([train_df, selected_extra_df], ignore_index=True)
    output_folds_df = pd.concat([folds_df, selected_folds_df], ignore_index=True)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_train_df.to_csv(output_train, index=False)
    output_folds_df.to_csv(output_folds, index=False)

    report = {
        "config_path": str(cfg_path),
        "input_train": str(train_path),
        "input_folds": str(folds_path),
        "output_train": str(output_train),
        "output_folds": str(output_folds),
        "rows_input_train": int(len(train_df)),
        "rows_input_folds": int(len(folds_df)),
        "candidates_before_token_filter": int(len(candidates)),
        "rows_extra_pool": int(len(aligned_extra_df)),
        "rows_extra_selected": int(len(selected_extra_df)),
        "rows_extra_selected_unique_oare_id": int(selected_extra_df["oare_id"].astype(str).nunique()),
        "rows_output_train": int(len(output_train_df)),
        "rows_output_folds": int(len(output_folds_df)),
        "mix_ratio": float(args.mix_ratio),
        "requested_extra_rows": int(requested_extra_rows),
        "effective_extra_rows": int(len(selected_extra_df)),
        "max_extra_rows": int(args.max_extra_rows),
        "allow_replacement": bool(args.allow_replacement),
        "used_replacement": bool(used_replacement),
        "alignment_alpha": float(alpha),
        "max_align_cost": float(args.max_align_cost),
        "candidate_parent_rows_by_mode": parent_mode_stats,
        "candidate_rows_by_align_type": align_type_stats,
        "avg_align_cost_pool": float(aligned_extra_df["align_cost"].mean()) if not aligned_extra_df.empty else 0.0,
        "avg_align_cost_selected": float(selected_extra_df["align_cost"].mean()) if not selected_extra_df.empty else 0.0,
        "skipped_missing_fold_rows": int(skipped_missing_fold),
        "source_delimiters": source_delimiters,
        "target_sentence_regex": str(args.target_sentence_regex),
        "source_line_number_regex": str(args.source_line_number_regex),
        "model_name": model_name,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    print(f"OK: wrote {output_folds}")
    print(f"OK: wrote {report_json}")
    print(
        "INFO: gc_pool/selected="
        f"{len(aligned_extra_df)}/{len(selected_extra_df)}, "
        f"used_replacement={used_replacement}, "
        f"rows_out={len(output_train_df)}"
    )


if __name__ == "__main__":
    main()
