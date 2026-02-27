from __future__ import annotations

import argparse
import json
import math
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


def _split_sent_like(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    current = []
    separators = set(";:.!?")
    for char in text:
        current.append(char)
        if char in separators:
            piece = "".join(current).strip()
            if piece:
                chunks.append(piece)
            current = []
    tail = "".join(current).strip()
    if tail:
        chunks.append(tail)
    if not chunks:
        return [text]
    return chunks


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


def _split_target_chunks(target: str, n_chunks: int) -> list[str]:
    segments = _split_sent_like(target)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--output-train", default="")
    ap.add_argument("--report-json", default="")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--src-limit", type=int, default=-1)
    ap.add_argument("--tgt-limit", type=int, default=-1)
    ap.add_argument("--min-chunks", type=int, default=2)
    ap.add_argument("--always-chunk", action="store_true")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_path = _resolve_path(args.input_train, processed_dir / "train_proc.csv")
    output_train = _resolve_path(args.output_train, processed_dir / "train_proc_chunked.csv")
    report_json = _resolve_path(args.report_json, processed_dir / "long_chunk_report.json")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    src_limit = int(args.src_limit if args.src_limit > 0 else model_cfg.get("max_source_length", 256))
    tgt_limit = int(args.tgt_limit if args.tgt_limit > 0 else model_cfg.get("max_target_length", 192))
    min_chunks = max(2, int(args.min_chunks))

    frame = pd.read_csv(train_path)
    if args.source_col not in frame.columns or args.target_col not in frame.columns:
        raise KeyError(f"Missing source/target columns: {args.source_col}, {args.target_col}")
    if "oare_id" not in frame.columns:
        raise KeyError("train file must include oare_id")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    src_ids = tokenizer(frame[args.source_col].fillna("").astype(str).tolist(), truncation=False)["input_ids"]
    tgt_ids = tokenizer(
        text_target=frame[args.target_col].fillna("").astype(str).tolist(),
        truncation=False,
    )["input_ids"]

    out_rows: list[dict[str, Any]] = []
    chunked_parent = 0
    total_chunk_rows = 0

    for idx, row in frame.iterrows():
        src_text = str(row[args.source_col]) if isinstance(row[args.source_col], str) else ""
        tgt_text = str(row[args.target_col]) if isinstance(row[args.target_col], str) else ""
        src_len = len(src_ids[idx])
        tgt_len = len(tgt_ids[idx])

        src_chunks_needed = int(math.ceil(src_len / max(1, src_limit)))
        tgt_chunks_needed = int(math.ceil(tgt_len / max(1, tgt_limit)))
        chunks_needed = max(src_chunks_needed, tgt_chunks_needed, 1)

        should_chunk = args.always_chunk or (chunks_needed > 1)
        if not should_chunk:
            item = row.to_dict()
            item["parent_oare_id"] = str(row["oare_id"])
            item["chunk_index"] = 0
            item["chunk_total"] = 1
            item["is_chunk"] = False
            out_rows.append(item)
            continue

        chunk_total = max(min_chunks, chunks_needed)
        target_chunks = _split_target_chunks(tgt_text, chunk_total)
        ratios = [max(1, len(x)) for x in target_chunks]
        source_chunks = _split_source_by_ratio(src_text, ratios)
        source_chunks = source_chunks[: len(target_chunks)]
        while len(source_chunks) < len(target_chunks):
            source_chunks.append("")

        parent_id = str(row["oare_id"])
        chunked_parent += 1
        total_chunk_rows += len(target_chunks)

        for chunk_idx, (src_chunk, tgt_chunk) in enumerate(zip(source_chunks, target_chunks)):
            item = row.to_dict()
            item["parent_oare_id"] = parent_id
            item["oare_id"] = f"{parent_id}__c{chunk_idx + 1}of{len(target_chunks)}"
            item[args.source_col] = src_chunk.strip()
            item[args.target_col] = tgt_chunk.strip()
            item["chunk_index"] = int(chunk_idx)
            item["chunk_total"] = int(len(target_chunks))
            item["is_chunk"] = True
            out_rows.append(item)

    out_df = pd.DataFrame(out_rows)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_train, index=False)

    report = {
        "config_path": str(cfg_path),
        "input_train": str(train_path),
        "output_train": str(output_train),
        "rows_input": int(len(frame)),
        "rows_output": int(len(out_df)),
        "chunked_parent_rows": int(chunked_parent),
        "added_rows_from_chunking": int(len(out_df) - len(frame)),
        "avg_chunk_total_on_chunked_rows": float(
            out_df[out_df["is_chunk"] == True]["chunk_total"].mean() if chunked_parent > 0 else 0.0
        ),
        "src_limit": int(src_limit),
        "tgt_limit": int(tgt_limit),
        "min_chunks": int(min_chunks),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    print(f"OK: wrote {report_json}")
    print(
        "INFO: chunked_parent_rows="
        f"{chunked_parent}, rows_in/out={len(frame)}/{len(out_df)}, added={len(out_df) - len(frame)}"
    )


if __name__ == "__main__":
    main()
