from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from transformers import AutoTokenizer


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
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _decode_delimiter(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _parse_delimiters(raw: str, fallback: list[str]) -> list[str]:
    if not str(raw or "").strip():
        return fallback
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    decoded = [_decode_delimiter(x) for x in values]
    unique: list[str] = []
    seen: set[str] = set()
    for token in decoded:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique or fallback


def _split_by_delimiters(text: str, delimiters: list[str]) -> list[str]:
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return []
    escaped = [re.escape(x) for x in delimiters if x]
    if not escaped:
        return [value]
    pattern = "|".join(escaped)
    pieces = [x.strip() for x in re.split(pattern, value) if x and x.strip()]
    return pieces or [value]


def _split_by_word_count(text: str, n_chunks: int) -> list[str]:
    words = [x for x in (text or "").split(" ") if x]
    if not words:
        return [""]
    n_chunks = max(1, min(n_chunks, len(words)))
    out: list[str] = []
    start = 0
    for i in range(n_chunks):
        end = round((i + 1) * len(words) / n_chunks)
        piece = " ".join(words[start:end]).strip()
        out.append(piece)
        start = end
    return [x for x in out if x]


def _split_target_chunks(target: str, n_chunks: int, delimiters: list[str]) -> list[str]:
    segments = _split_by_delimiters(target, delimiters)
    if len(segments) < n_chunks:
        return _split_by_word_count(target, n_chunks)
    target_chars = max(1, sum(len(x) for x in segments))
    budget = max(1, target_chars / n_chunks)
    out: list[str] = []
    current: list[str] = []
    current_len = 0
    remaining = len(segments)
    for seg in segments:
        remaining -= 1
        current.append(seg)
        current_len += len(seg)
        needed = n_chunks - len(out) - 1
        if (current_len >= budget and needed <= remaining) or remaining < needed:
            out.append(" ".join(current).strip())
            current = []
            current_len = 0
    if current:
        out.append(" ".join(current).strip())
    if len(out) < n_chunks:
        return _split_by_word_count(target, n_chunks)
    return out


def _split_source_by_ratio(source: str, ratios: list[float]) -> list[str]:
    words = [x for x in (source or "").split(" ") if x]
    if not words:
        return ["" for _ in ratios]
    ratios = [max(0.0, float(x)) for x in ratios]
    total = sum(ratios)
    if total <= 0:
        ratios = [1.0 / max(1, len(ratios)) for _ in ratios]
    else:
        ratios = [x / total for x in ratios]
    out: list[str] = []
    start = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            end = len(words)
        else:
            end = min(len(words), start + max(1, round(len(words) * ratio)))
        out.append(" ".join(words[start:end]).strip())
        start = end
    if start < len(words):
        tail = " ".join(words[start:]).strip()
        if out:
            out[-1] = f"{out[-1]} {tail}".strip()
        else:
            out = [tail]
    return out


def _trim_or_expand_chunks(chunks: list[str], desired: int) -> list[str]:
    if desired <= 0:
        return chunks
    if len(chunks) == desired:
        return chunks
    merged = " ".join([x for x in chunks if x]).strip()
    if not merged:
        return ["" for _ in range(desired)]
    return _split_by_word_count(merged, desired)


def _token_lengths(*, tokenizer, values: list[str], as_target: bool) -> list[int]:
    if as_target:
        ids = tokenizer(text_target=values, truncation=False, add_special_tokens=True)["input_ids"]
    else:
        ids = tokenizer(values, truncation=False, add_special_tokens=True)["input_ids"]
    return [len(x) for x in ids]


def _split_one_chunk(value: str) -> tuple[str, str]:
    pieces = _split_by_word_count(value, 2)
    if len(pieces) >= 2:
        return pieces[0], pieces[1]
    text = (value or "").strip()
    if not text:
        return "", ""
    half = max(1, len(text) // 2)
    return text[:half].strip(), text[half:].strip()


def _enforce_pair_limits(
    *,
    source_chunks: list[str],
    target_chunks: list[str],
    tokenizer,
    src_limit: int,
    tgt_limit: int,
    max_chunks: int,
) -> tuple[list[str], list[str]]:
    src_chunks = list(source_chunks)
    tgt_chunks = list(target_chunks)
    while len(src_chunks) == len(tgt_chunks) and len(src_chunks) < max_chunks:
        src_lens = _token_lengths(tokenizer=tokenizer, values=src_chunks, as_target=False)
        tgt_lens = _token_lengths(tokenizer=tokenizer, values=tgt_chunks, as_target=True)
        violating = [
            idx
            for idx, (src_len, tgt_len) in enumerate(zip(src_lens, tgt_lens))
            if src_len > src_limit or tgt_len > tgt_limit
        ]
        if not violating:
            break
        idx = max(violating, key=lambda i: max(src_lens[i] - src_limit, tgt_lens[i] - tgt_limit))
        src_left, src_right = _split_one_chunk(src_chunks[idx])
        tgt_left, tgt_right = _split_one_chunk(tgt_chunks[idx])
        src_chunks = src_chunks[:idx] + [src_left, src_right] + src_chunks[idx + 1 :]
        tgt_chunks = tgt_chunks[:idx] + [tgt_left, tgt_right] + tgt_chunks[idx + 1 :]
    return src_chunks, tgt_chunks


def _enforce_source_limits(
    *,
    source_chunks: list[str],
    tokenizer,
    src_limit: int,
    max_chunks: int,
) -> list[str]:
    chunks = list(source_chunks)
    while len(chunks) < max_chunks:
        src_lens = _token_lengths(tokenizer=tokenizer, values=chunks, as_target=False)
        violating = [idx for idx, value in enumerate(src_lens) if value > src_limit]
        if not violating:
            break
        idx = max(violating, key=lambda i: src_lens[i] - src_limit)
        left, right = _split_one_chunk(chunks[idx])
        chunks = chunks[:idx] + [left, right] + chunks[idx + 1 :]
    return chunks


def _append_train_row(
    *,
    out_rows: list[dict[str, Any]],
    row: pd.Series,
    parent_id: str,
    source_col: str,
    target_col: str,
    source_value: str,
    target_value: str,
    chunk_idx: int,
    chunk_total: int,
    split_mode: str,
) -> None:
    item = row.to_dict()
    if chunk_total > 1:
        item["oare_id"] = f"{parent_id}__c{chunk_idx + 1}of{chunk_total}"
    else:
        item["oare_id"] = parent_id
    item[source_col] = source_value.strip()
    item[target_col] = target_value.strip()
    item["parent_oare_id"] = parent_id
    item["chunk_index"] = int(chunk_idx)
    item["chunk_total"] = int(chunk_total)
    item["is_chunk"] = bool(chunk_total > 1)
    item["chunk_mode"] = str(split_mode)
    out_rows.append(item)


def _append_test_row(
    *,
    out_rows: list[dict[str, Any]],
    row: pd.Series,
    parent_id: str,
    source_col: str,
    source_value: str,
    chunk_idx: int,
    chunk_total: int,
    split_mode: str,
) -> None:
    item = row.to_dict()
    if chunk_total > 1:
        item["id"] = f"{parent_id}__c{chunk_idx + 1}of{chunk_total}"
    else:
        item["id"] = parent_id
    item[source_col] = source_value.strip()
    item["parent_id"] = parent_id
    item["chunk_index"] = int(chunk_idx)
    item["chunk_total"] = int(chunk_total)
    item["is_chunk"] = bool(chunk_total > 1)
    item["chunk_mode"] = str(split_mode)
    out_rows.append(item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--input-test", default="")
    ap.add_argument("--input-folds", default="")
    ap.add_argument("--output-train", default="")
    ap.add_argument("--output-test", default="")
    ap.add_argument("--output-folds", default="")
    ap.add_argument("--output-map-train", default="")
    ap.add_argument("--output-map-test", default="")
    ap.add_argument("--report-json", default="")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--src-limit", type=int, default=-1)
    ap.add_argument("--tgt-limit", type=int, default=-1)
    ap.add_argument("--min-chunks", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--always-chunk", action="store_true")
    ap.add_argument("--alignment-mode", default="auto", choices=["auto", "delimiter", "ratio"])
    ap.add_argument("--source-delimiters", default="\\n,;,|")
    ap.add_argument("--target-delimiters", default="\\n,;,.,!,?")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    chunk_cfg = cfg.get("chunking", {}) or {}
    train_path = _resolve_path(args.input_train, processed_dir / "train_proc.csv")
    test_path = _resolve_path(args.input_test, processed_dir / "test_proc.csv")
    folds_path = _resolve_path(args.input_folds, processed_dir / "folds.csv")
    output_train = _resolve_path(args.output_train, processed_dir / "train_proc_chunked.csv")
    output_test = _resolve_path(args.output_test, processed_dir / "test_proc_chunked.csv")
    output_folds = _resolve_path(args.output_folds, processed_dir / "folds_chunked.csv")
    output_map_train = _resolve_path(args.output_map_train, processed_dir / "chunk_map_train.csv")
    output_map_test = _resolve_path(args.output_map_test, processed_dir / "chunk_map_test.csv")
    report_json = _resolve_path(args.report_json, processed_dir / "long_chunk_report.json")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    src_limit = int(args.src_limit if args.src_limit > 0 else chunk_cfg.get("src_limit", model_cfg.get("max_source_length", 256)))
    tgt_limit = int(args.tgt_limit if args.tgt_limit > 0 else chunk_cfg.get("tgt_limit", model_cfg.get("max_target_length", 192)))
    min_chunks = max(2, int(args.min_chunks if args.min_chunks > 0 else chunk_cfg.get("min_chunks", 2)))
    max_chunks = max(min_chunks, int(args.max_chunks if args.max_chunks > 0 else chunk_cfg.get("max_chunks", 8)))
    alignment_mode = str(args.alignment_mode if args.alignment_mode else chunk_cfg.get("alignment_mode", "auto"))
    source_delimiters = _parse_delimiters(
        str(args.source_delimiters if args.source_delimiters else chunk_cfg.get("source_delimiters", "\\n,;,|")),
        fallback=["\n", ";", "|"],
    )
    target_delimiters = _parse_delimiters(
        str(args.target_delimiters if args.target_delimiters else chunk_cfg.get("target_delimiters", "\\n,;,.,!,?")),
        fallback=["\n", ";", ".", "!", "?"],
    )

    frame = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    folds_df = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    if args.source_col not in frame.columns or args.target_col not in frame.columns:
        raise KeyError(f"Missing source/target columns: {args.source_col}, {args.target_col}")
    if "oare_id" not in frame.columns:
        raise KeyError("train file must include oare_id")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_values = frame[args.source_col].fillna("").astype(str).tolist()
    tgt_values = frame[args.target_col].fillna("").astype(str).tolist()
    src_lens = _token_lengths(tokenizer=tokenizer, values=src_values, as_target=False)
    tgt_lens = _token_lengths(tokenizer=tokenizer, values=tgt_values, as_target=True)

    out_rows: list[dict[str, Any]] = []
    train_map_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    test_map_rows: list[dict[str, Any]] = []
    chunked_parent = 0
    total_chunk_rows = 0
    delimiter_aligned_rows = 0

    for idx, row in frame.iterrows():
        src_text = str(row[args.source_col]) if isinstance(row[args.source_col], str) else ""
        tgt_text = str(row[args.target_col]) if isinstance(row[args.target_col], str) else ""
        src_len = int(src_lens[idx])
        tgt_len = int(tgt_lens[idx])

        src_chunks_needed = int(math.ceil(src_len / max(1, src_limit)))
        tgt_chunks_needed = int(math.ceil(tgt_len / max(1, tgt_limit)))
        chunks_needed = max(src_chunks_needed, tgt_chunks_needed, 1)

        should_chunk = args.always_chunk or (chunks_needed > 1)
        if not should_chunk:
            parent_id = str(row["oare_id"])
            _append_train_row(
                out_rows=out_rows,
                row=row,
                parent_id=parent_id,
                source_col=args.source_col,
                target_col=args.target_col,
                source_value=src_text,
                target_value=tgt_text,
                chunk_idx=0,
                chunk_total=1,
                split_mode="none",
            )
            train_map_rows.append(
                {
                    "oare_id": parent_id,
                    "parent_oare_id": parent_id,
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "chunk_mode": "none",
                }
            )
            continue

        split_mode = "ratio"
        source_chunks: list[str] = []
        target_chunks: list[str] = []

        src_delim_chunks = _split_by_delimiters(src_text, source_delimiters)
        tgt_delim_chunks = _split_by_delimiters(tgt_text, target_delimiters)
        can_use_delimiter = (
            alignment_mode in {"auto", "delimiter"}
            and len(src_delim_chunks) >= min_chunks
            and len(src_delim_chunks) == len(tgt_delim_chunks)
        )
        if can_use_delimiter:
            chunk_total_delim = min(max_chunks, len(src_delim_chunks))
            source_chunks = _trim_or_expand_chunks(src_delim_chunks, chunk_total_delim)
            target_chunks = _trim_or_expand_chunks(tgt_delim_chunks, chunk_total_delim)
            split_mode = "delimiter_aligned"
            delimiter_aligned_rows += 1
        else:
            chunk_total = min(max_chunks, max(min_chunks, chunks_needed))
            target_chunks = _split_target_chunks(tgt_text, chunk_total, target_delimiters)
            target_chunks = _trim_or_expand_chunks(target_chunks, chunk_total)
            ratios = [max(1, len(x)) for x in target_chunks]
            source_chunks = _split_source_by_ratio(src_text, ratios)
            source_chunks = _trim_or_expand_chunks(source_chunks, chunk_total)
            split_mode = "ratio"

        chunk_total = min(len(source_chunks), len(target_chunks))
        if chunk_total <= 0:
            chunk_total = 1
            source_chunks = [src_text]
            target_chunks = [tgt_text]
            split_mode = "fallback_single"
        source_chunks, target_chunks = _enforce_pair_limits(
            source_chunks=source_chunks[:chunk_total],
            target_chunks=target_chunks[:chunk_total],
            tokenizer=tokenizer,
            src_limit=src_limit,
            tgt_limit=tgt_limit,
            max_chunks=max_chunks,
        )
        chunk_total = min(len(source_chunks), len(target_chunks))

        parent_id = str(row["oare_id"])
        chunked_parent += 1
        total_chunk_rows += chunk_total

        for chunk_idx in range(chunk_total):
            src_chunk = source_chunks[chunk_idx]
            tgt_chunk = target_chunks[chunk_idx]
            _append_train_row(
                out_rows=out_rows,
                row=row,
                parent_id=parent_id,
                source_col=args.source_col,
                target_col=args.target_col,
                source_value=src_chunk,
                target_value=tgt_chunk,
                chunk_idx=chunk_idx,
                chunk_total=chunk_total,
                split_mode=split_mode,
            )
            train_map_rows.append(
                {
                    "oare_id": f"{parent_id}__c{chunk_idx + 1}of{chunk_total}" if chunk_total > 1 else parent_id,
                    "parent_oare_id": parent_id,
                    "chunk_index": int(chunk_idx),
                    "chunk_total": int(chunk_total),
                    "chunk_mode": split_mode,
                }
            )

    if not test_df.empty and args.source_col in test_df.columns and "id" in test_df.columns:
        test_sources = test_df[args.source_col].fillna("").astype(str).tolist()
        test_src_lens = _token_lengths(tokenizer=tokenizer, values=test_sources, as_target=False)
        for idx, row in test_df.iterrows():
            src_text = str(row[args.source_col]) if isinstance(row[args.source_col], str) else ""
            src_len = int(test_src_lens[idx])
            chunks_needed = max(1, int(math.ceil(src_len / max(1, src_limit))))
            should_chunk = args.always_chunk or (chunks_needed > 1)
            parent_id = str(row["id"])

            if not should_chunk:
                _append_test_row(
                    out_rows=test_rows,
                    row=row,
                    parent_id=parent_id,
                    source_col=args.source_col,
                    source_value=src_text,
                    chunk_idx=0,
                    chunk_total=1,
                    split_mode="none",
                )
                test_map_rows.append(
                    {
                        "id": parent_id,
                        "parent_id": parent_id,
                        "chunk_index": 0,
                        "chunk_total": 1,
                        "chunk_mode": "none",
                    }
                )
                continue

            src_delim_chunks = _split_by_delimiters(src_text, source_delimiters)
            if len(src_delim_chunks) >= min_chunks and alignment_mode in {"auto", "delimiter"}:
                chunk_total = min(max_chunks, len(src_delim_chunks))
                source_chunks = _trim_or_expand_chunks(src_delim_chunks, chunk_total)
                split_mode = "delimiter_only"
            else:
                chunk_total = min(max_chunks, max(min_chunks, chunks_needed))
                source_chunks = _trim_or_expand_chunks(_split_by_word_count(src_text, chunk_total), chunk_total)
                split_mode = "ratio_only"
            source_chunks = _enforce_source_limits(
                source_chunks=source_chunks,
                tokenizer=tokenizer,
                src_limit=src_limit,
                max_chunks=max_chunks,
            )

            for chunk_idx in range(len(source_chunks)):
                _append_test_row(
                    out_rows=test_rows,
                    row=row,
                    parent_id=parent_id,
                    source_col=args.source_col,
                    source_value=source_chunks[chunk_idx],
                    chunk_idx=chunk_idx,
                    chunk_total=len(source_chunks),
                    split_mode=split_mode,
                )
                test_map_rows.append(
                    {
                        "id": f"{parent_id}__c{chunk_idx + 1}of{len(source_chunks)}" if len(source_chunks) > 1 else parent_id,
                        "parent_id": parent_id,
                        "chunk_index": int(chunk_idx),
                        "chunk_total": int(len(source_chunks)),
                        "chunk_mode": split_mode,
                    }
                )

    out_df = pd.DataFrame(out_rows)
    out_test_df = pd.DataFrame(test_rows)
    train_map_df = pd.DataFrame(train_map_rows)
    test_map_df = pd.DataFrame(test_map_rows)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_train, index=False)
    if not out_test_df.empty:
        out_test_df.to_csv(output_test, index=False)
    train_map_df.to_csv(output_map_train, index=False)
    if not test_map_df.empty:
        test_map_df.to_csv(output_map_test, index=False)

    output_train_src_lens = _token_lengths(
        tokenizer=tokenizer,
        values=out_df[args.source_col].fillna("").astype(str).tolist(),
        as_target=False,
    )
    output_train_tgt_lens = _token_lengths(
        tokenizer=tokenizer,
        values=out_df[args.target_col].fillna("").astype(str).tolist(),
        as_target=True,
    )
    out_src_trunc = 100.0 * float(sum(1 for x in output_train_src_lens if x > src_limit)) / float(max(1, len(output_train_src_lens)))
    out_tgt_trunc = 100.0 * float(sum(1 for x in output_train_tgt_lens if x > tgt_limit)) / float(max(1, len(output_train_tgt_lens)))

    folds_written = False
    if not folds_df.empty and "oare_id" in folds_df.columns:
        train_map_for_merge = train_map_df.copy()
        train_map_for_merge["parent_oare_id"] = train_map_for_merge["parent_oare_id"].astype(str)
        if "parent_oare_id" in folds_df.columns:
            parent_cols = ["parent_oare_id", "fold", "group_key", "group_kind", "group_source"]
            parent_available = [x for x in parent_cols if x in folds_df.columns]
            parent_folds = folds_df[parent_available].drop_duplicates(subset=["parent_oare_id"]).copy()
        else:
            parent_folds = folds_df.rename(columns={"oare_id": "parent_oare_id"}).copy()
        parent_folds["parent_oare_id"] = parent_folds["parent_oare_id"].astype(str)
        chunk_folds = train_map_for_merge.merge(parent_folds, on="parent_oare_id", how="left")
        if chunk_folds["fold"].isna().any():
            missing = int(chunk_folds["fold"].isna().sum())
            raise ValueError(f"Missing fold mapping for {missing} chunk rows")
        chunk_folds = chunk_folds.rename(columns={"oare_id": "chunk_oare_id"})
        chunk_folds["oare_id"] = chunk_folds["chunk_oare_id"].astype(str)
        keep_cols = ["oare_id", "fold", "group_key", "group_kind", "group_source", "parent_oare_id", "chunk_index", "chunk_total", "chunk_mode"]
        available_cols = [x for x in keep_cols if x in chunk_folds.columns]
        chunk_folds = chunk_folds[available_cols].copy()
        chunk_folds.to_csv(output_folds, index=False)
        folds_written = True

    report = {
        "config_path": str(cfg_path),
        "input_train": str(train_path),
        "input_test": str(test_path) if test_path.exists() else "",
        "input_folds": str(folds_path) if folds_path.exists() else "",
        "output_train": str(output_train),
        "output_test": str(output_test) if not out_test_df.empty else "",
        "output_folds": str(output_folds) if folds_written else "",
        "output_map_train": str(output_map_train),
        "output_map_test": str(output_map_test) if not test_map_df.empty else "",
        "rows_input": int(len(frame)),
        "rows_output": int(len(out_df)),
        "rows_test_input": int(len(test_df)),
        "rows_test_output": int(len(out_test_df)),
        "chunked_parent_rows": int(chunked_parent),
        "added_rows_from_chunking": int(len(out_df) - len(frame)),
        "avg_chunk_total_on_chunked_rows": float(
            out_df[out_df["is_chunk"] == True]["chunk_total"].mean() if chunked_parent > 0 else 0.0
        ),
        "delimiter_aligned_parent_rows": int(delimiter_aligned_rows),
        "delimiter_aligned_ratio_pct": 100.0 * float(delimiter_aligned_rows) / float(max(1, chunked_parent)),
        "src_limit": int(src_limit),
        "tgt_limit": int(tgt_limit),
        "min_chunks": int(min_chunks),
        "max_chunks": int(max_chunks),
        "alignment_mode": alignment_mode,
        "source_delimiters": source_delimiters,
        "target_delimiters": target_delimiters,
        "before_truncation_ratio_pct": {
            "source": 100.0 * float(sum(1 for x in src_lens if x > src_limit)) / float(max(1, len(src_lens))),
            "target": 100.0 * float(sum(1 for x in tgt_lens if x > tgt_limit)) / float(max(1, len(tgt_lens))),
        },
        "after_truncation_ratio_pct": {
            "source": out_src_trunc,
            "target": out_tgt_trunc,
        },
    }
    if not out_test_df.empty:
        output_test_src_lens = _token_lengths(
            tokenizer=tokenizer,
            values=out_test_df[args.source_col].fillna("").astype(str).tolist(),
            as_target=False,
        )
        report["after_test_source_truncation_ratio_pct"] = 100.0 * float(
            sum(1 for x in output_test_src_lens if x > src_limit)
        ) / float(max(1, len(output_test_src_lens)))

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    if not out_test_df.empty:
        print(f"OK: wrote {output_test}")
    if folds_written:
        print(f"OK: wrote {output_folds}")
    print(f"OK: wrote {output_map_train}")
    if not test_map_df.empty:
        print(f"OK: wrote {output_map_test}")
    print(f"OK: wrote {report_json}")
    print(
        "INFO: chunked_parent_rows="
        f"{chunked_parent}, rows_in/out={len(frame)}/{len(out_df)}, added={len(out_df) - len(frame)}"
    )
    print(
        "INFO: trunc_before(src/tgt)="
        f"{report['before_truncation_ratio_pct']['source']:.2f}/{report['before_truncation_ratio_pct']['target']:.2f}, "
        "trunc_after(src/tgt)="
        f"{report['after_truncation_ratio_pct']['source']:.2f}/{report['after_truncation_ratio_pct']['target']:.2f}"
    )


if __name__ == "__main__":
    main()
