from __future__ import annotations

import argparse
import json
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
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _length_stats(lengths: list[int]) -> dict[str, float]:
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb_e2_len.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb_e2_len.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    input_train = _resolve_path(args.input_train, processed_dir / "train_proc.csv")
    out_json = _resolve_path(
        args.out_json,
        REPO_ROOT / "runs" / f"tokenizer_health_{Path(cfg_path).stem}.json",
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_train)
    if args.max_rows > 0:
        frame = frame.head(int(args.max_rows)).copy()

    source = frame[args.source_col].fillna("").astype(str).tolist()
    target = frame[args.target_col].fillna("").astype(str).tolist()
    model_name = str(model_cfg.get("name", "google/mt5-small"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    unk_id = tokenizer.unk_token_id

    src_encoded = tokenizer(source, truncation=False, add_special_tokens=True)["input_ids"]
    tgt_encoded = tokenizer(text_target=target, truncation=False, add_special_tokens=True)["input_ids"]

    src_lens = [len(x) for x in src_encoded]
    tgt_lens = [len(x) for x in tgt_encoded]

    src_total = sum(src_lens)
    tgt_total = sum(tgt_lens)
    src_unk = 0
    tgt_unk = 0
    if unk_id is not None and int(unk_id) >= 0:
        src_unk = sum(sum(1 for token in seq if token == unk_id) for seq in src_encoded)
        tgt_unk = sum(sum(1 for token in seq if token == unk_id) for seq in tgt_encoded)

    report = {
        "config_path": str(cfg_path),
        "input_train": str(input_train),
        "model_name": model_name,
        "rows": int(len(frame)),
        "source": {
            "token_length": _length_stats(src_lens),
            "unk_ratio_pct": 100.0 * float(src_unk) / float(max(1, src_total)),
        },
        "target": {
            "token_length": _length_stats(tgt_lens),
            "unk_ratio_pct": 100.0 * float(tgt_unk) / float(max(1, tgt_total)),
        },
        "model_limits": {
            "max_source_length": int(model_cfg.get("max_source_length", 0)),
            "max_target_length": int(model_cfg.get("max_target_length", 0)),
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {out_json}")
    print(
        "INFO:"
        f" src_p95/p99={report['source']['token_length']['p95']:.2f}/{report['source']['token_length']['p99']:.2f},"
        f" tgt_p95/p99={report['target']['token_length']['p95']:.2f}/{report['target']['token_length']['p99']:.2f},"
        f" src_unk={report['source']['unk_ratio_pct']:.4f}%,"
        f" tgt_unk={report['target']['unk_ratio_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
