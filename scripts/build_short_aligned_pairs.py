from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
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
    return pieces


def _split_by_word_count(text: str, n_chunks: int) -> list[str]:
    words = [x for x in str(text or "").split(" ") if x]
    if not words:
        return []
    n_chunks = max(1, min(int(n_chunks), len(words)))
    out: list[str] = []
    start = 0
    for i in range(n_chunks):
        end = len(words) if i == n_chunks - 1 else round((i + 1) * len(words) / n_chunks)
        piece = " ".join(words[start:end]).strip()
        if piece:
            out.append(piece)
        start = end
    return out


def _sample_indices(rng: np.random.Generator, *, total: int, take: int) -> np.ndarray:
    if take <= 0 or total <= 0:
        return np.asarray([], dtype=np.int64)
    if take <= total:
        return rng.choice(total, size=take, replace=False)
    return rng.choice(total, size=take, replace=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/byt5_small_lora_chunked.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--input-folds", default="")
    ap.add_argument("--output-train", default="")
    ap.add_argument("--output-folds", default="")
    ap.add_argument("--report-json", default="")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--source-delimiters", default="\\n,;,|")
    ap.add_argument("--target-delimiters", default="\\n,;,.,!,?")
    ap.add_argument("--min-segments", type=int, default=2)
    ap.add_argument("--max-segments", type=int, default=12)
    ap.add_argument("--min-source-chars", type=int, default=4)
    ap.add_argument("--min-target-chars", type=int, default=3)
    ap.add_argument("--max-source-tokens", type=int, default=256)
    ap.add_argument("--max-target-tokens", type=int, default=256)
    ap.add_argument("--fallback-parts", type=int, default=2)
    ap.add_argument("--fallback-equal-split", dest="fallback_equal_split", action="store_true")
    ap.add_argument("--no-fallback-equal-split", dest="fallback_equal_split", action="store_false")
    ap.add_argument("--mix-ratio", type=float, default=3.0)
    ap.add_argument("--max-extra-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", default="")
    ap.set_defaults(fallback_equal_split=True)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "byt5_small_lora_chunked.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")

    train_path = _resolve_path(args.input_train, processed_dir / "train_proc.csv")
    folds_path = _resolve_path(args.input_folds, processed_dir / "folds.csv")
    output_train = _resolve_path(args.output_train, processed_dir / "train_proc_shortalign.csv")
    output_folds = _resolve_path(args.output_folds, processed_dir / "folds_shortalign.csv")
    report_json = _resolve_path(args.report_json, processed_dir / "short_align_report.json")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train csv: {train_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing folds csv: {folds_path}")

    source_delimiters = _parse_delimiters(args.source_delimiters, fallback=["\n", ";", "|"])
    target_delimiters = _parse_delimiters(args.target_delimiters, fallback=["\n", ";", ".", "!", "?"])

    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    if args.source_col not in train_df.columns or args.target_col not in train_df.columns:
        raise KeyError(f"Missing source/target columns: {args.source_col}, {args.target_col}")
    if "oare_id" not in train_df.columns:
        raise KeyError("train file must include oare_id")
    if "oare_id" not in folds_df.columns:
        raise KeyError("folds file must include oare_id")

    model_name = str(args.model_name or model_cfg.get("name", "google/byt5-small"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    min_segments = max(2, int(args.min_segments))
    max_segments = max(min_segments, int(args.max_segments))
    min_source_chars = max(1, int(args.min_source_chars))
    min_target_chars = max(1, int(args.min_target_chars))
    max_source_tokens = max(1, int(args.max_source_tokens))
    max_target_tokens = max(1, int(args.max_target_tokens))
    fallback_parts = max(2, int(args.fallback_parts))
    fallback_equal_split = bool(args.fallback_equal_split)
    seed = int(args.seed)

    fold_info = folds_df.set_index("oare_id").to_dict(orient="index")

    candidates: list[dict[str, Any]] = []
    candidate_mode_stats: dict[str, int] = {}
    for _, row in train_df.iterrows():
        source = str(row.get(args.source_col, "") or "")
        target = str(row.get(args.target_col, "") or "")
        src_segments = _split_by_delimiters(source, source_delimiters)
        tgt_segments = _split_by_delimiters(target, target_delimiters)

        align_mode = ""
        if (
            src_segments
            and tgt_segments
            and len(src_segments) == len(tgt_segments)
            and len(src_segments) >= min_segments
            and len(src_segments) <= max_segments
        ):
            align_mode = "delimiter"
        elif fallback_equal_split:
            src_segments = _split_by_word_count(source, fallback_parts)
            tgt_segments = _split_by_word_count(target, fallback_parts)
            if (
                src_segments
                and tgt_segments
                and len(src_segments) == len(tgt_segments)
                and len(src_segments) >= min_segments
                and len(src_segments) <= max_segments
            ):
                align_mode = "fallback_equal"
        if not align_mode:
            continue

        parent_id = str(row.get("parent_oare_id", row["oare_id"]))
        base_id = str(row["oare_id"])
        total = len(src_segments)
        candidate_mode_stats[align_mode] = int(candidate_mode_stats.get(align_mode, 0) + 1)
        for seg_idx, (src_seg, tgt_seg) in enumerate(zip(src_segments, tgt_segments)):
            src_seg = src_seg.strip()
            tgt_seg = tgt_seg.strip()
            if len(src_seg) < min_source_chars or len(tgt_seg) < min_target_chars:
                continue
            candidates.append(
                {
                    "base_id": base_id,
                    "parent_id": parent_id,
                    "segment_index": int(seg_idx),
                    "segment_total": int(total),
                    "source": src_seg,
                    "target": tgt_seg,
                    "align_mode": align_mode,
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
        }
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("WARN: no aligned candidates found; wrote passthrough outputs.")
        print(f"OK: wrote {output_train}")
        print(f"OK: wrote {output_folds}")
        print(f"OK: wrote {report_json}")
        return

    candidate_sources = [item["source"] for item in candidates]
    candidate_targets = [item["target"] for item in candidates]
    src_tok_lens = [len(x) for x in tokenizer(candidate_sources, truncation=False, add_special_tokens=True)["input_ids"]]
    tgt_tok_lens = [
        len(x) for x in tokenizer(text_target=candidate_targets, truncation=False, add_special_tokens=True)["input_ids"]
    ]

    extra_rows: list[dict[str, Any]] = []
    extra_folds: list[dict[str, Any]] = []
    skipped_missing_fold = 0
    for idx, item in enumerate(candidates):
        if src_tok_lens[idx] > max_source_tokens or tgt_tok_lens[idx] > max_target_tokens:
            continue
        base_id = str(item["base_id"])
        seg_idx = int(item["segment_index"])
        seg_total = int(item["segment_total"])
        new_id = f"{base_id}__sa{seg_idx + 1}of{seg_total}"

        out_row = dict(item["row"])
        out_row["oare_id"] = new_id
        out_row[args.source_col] = item["source"]
        out_row[args.target_col] = item["target"]
        out_row["parent_oare_id"] = str(item["parent_id"])
        out_row["chunk_index"] = seg_idx
        out_row["chunk_total"] = seg_total
        out_row["is_chunk"] = True
        out_row["chunk_mode"] = f"short_aligned_{item['align_mode']}"
        out_row["is_short_aligned"] = True
        out_row["short_align_mode"] = str(item["align_mode"])
        out_row["source_oare_id"] = base_id
        extra_rows.append(out_row)

        if base_id not in fold_info:
            skipped_missing_fold += 1
            continue
        fold_row = dict(fold_info[base_id])
        fold_row["oare_id"] = new_id
        if "parent_oare_id" not in fold_row:
            fold_row["parent_oare_id"] = str(item["parent_id"])
        fold_row["chunk_index"] = seg_idx
        fold_row["chunk_total"] = seg_total
        fold_row["chunk_mode"] = f"short_aligned_{item['align_mode']}"
        fold_row["short_align_mode"] = str(item["align_mode"])
        extra_folds.append(fold_row)

    extra_df = pd.DataFrame(extra_rows)
    extra_folds_df = pd.DataFrame(extra_folds)
    if extra_df.empty or extra_folds_df.empty:
        raise ValueError("No extra aligned rows left after token and fold filters.")

    extra_fold_map = extra_folds_df.set_index("oare_id").to_dict(orient="index")
    aligned_ids = [x for x in extra_df["oare_id"].astype(str).tolist() if x in extra_fold_map]
    if not aligned_ids:
        raise ValueError("No aligned ids with fold mapping.")

    aligned_extra_df = extra_df.set_index("oare_id").loc[aligned_ids].reset_index()

    target_extra_rows = int(round(float(args.mix_ratio) * float(len(train_df))))
    if int(args.max_extra_rows) > 0:
        target_extra_rows = min(target_extra_rows, int(args.max_extra_rows))
    target_extra_rows = max(0, target_extra_rows)

    rng = np.random.default_rng(seed)
    selected_indices = _sample_indices(
        rng,
        total=int(len(aligned_extra_df)),
        take=target_extra_rows,
    )
    selected_extra_df = aligned_extra_df.iloc[selected_indices].reset_index(drop=True) if target_extra_rows > 0 else aligned_extra_df.iloc[0:0].copy()
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
        "rows_output_train": int(len(output_train_df)),
        "rows_output_folds": int(len(output_folds_df)),
        "mix_ratio": float(args.mix_ratio),
        "max_extra_rows": int(args.max_extra_rows),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "max_source_tokens": int(max_source_tokens),
        "max_target_tokens": int(max_target_tokens),
        "fallback_equal_split": bool(fallback_equal_split),
        "fallback_parts": int(fallback_parts),
        "candidate_parent_rows_by_mode": candidate_mode_stats,
        "skipped_missing_fold_rows": int(skipped_missing_fold),
        "model_name": model_name,
        "source_delimiters": source_delimiters,
        "target_delimiters": target_delimiters,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    print(f"OK: wrote {output_folds}")
    print(f"OK: wrote {report_json}")
    print(
        "INFO: extra_pool/selected="
        f"{len(aligned_extra_df)}/{len(selected_extra_df)}, "
        f"rows_out={len(output_train_df)}"
    )


if __name__ == "__main__":
    main()
