from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from transformers import AutoTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cleaning.normalize import normalize_source  # noqa: E402
from generation_utils import apply_task_prefix, normalize_task_prefix  # noqa: E402


INLINE_WS_RE = re.compile(r"[^\S\n]+", flags=re.UNICODE)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _load_cleaning_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "cleaning" / "configs" / "cleaning.t0.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _normalize_text(
    text: str,
    *,
    apply_t0: bool,
    cfg: dict[str, Any],
    strip_text: bool,
    fold_ws: bool,
    lowercase: bool,
) -> str:
    output = text if isinstance(text, str) else ""
    if apply_t0:
        output, _ = normalize_source(output, config=cfg)
    output = output.replace("\r\n", "\n").replace("\r", "\n")
    if fold_ws:
        output = INLINE_WS_RE.sub(" ", output)
    if strip_text:
        output = output.strip()
    if lowercase:
        output = output.lower()
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb_e2_len.yaml")
    ap.add_argument("--input-train", default="")
    ap.add_argument("--output-train", default="data/interim/feasibility_zero_trunc_train.csv")
    ap.add_argument("--report-json", default="runs/feasibility_zero_trunc_report.json")
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb_e2_len.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    input_train = _resolve_path(args.input_train, REPO_ROOT / paths_cfg.get("train_csv", "data/interim/t0_train.csv"))
    output_train = _resolve_path(args.output_train, REPO_ROOT / "data" / "interim" / "feasibility_zero_trunc_train.csv")
    report_json = _resolve_path(args.report_json, REPO_ROOT / "runs" / "feasibility_zero_trunc_report.json")
    output_train.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_train)
    required = {"oare_id", "transliteration", "translation"}
    missing = sorted(required - set(train_df.columns))
    if missing:
        raise KeyError(f"Missing train columns: {missing}")

    apply_t0 = bool(preprocess_cfg.get("apply_t0_normalize", False))
    strip_text = bool(preprocess_cfg.get("strip_text", True))
    fold_ws = bool(preprocess_cfg.get("fold_inline_whitespace", True))
    lowercase_source = bool(preprocess_cfg.get("lowercase_source", False))
    lowercase_target = bool(preprocess_cfg.get("lowercase_target", False))
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))
    cleaning_cfg = _load_cleaning_config()

    train_df = train_df.copy()
    train_df["source_norm"] = train_df["transliteration"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
        )
    ).map(lambda s: apply_task_prefix(s, task_prefix))
    train_df["target_norm"] = train_df["translation"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_target,
        )
    )

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 384))
    max_target_length = int(model_cfg.get("max_target_length", 256))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    src_ids = tokenizer(train_df["source_norm"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]
    tgt_ids = tokenizer(text_target=train_df["target_norm"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]
    train_df["source_tok_len"] = [len(x) for x in src_ids]
    train_df["target_tok_len"] = [len(x) for x in tgt_ids]

    mask = (train_df["source_tok_len"] <= max_source_length) & (train_df["target_tok_len"] <= max_target_length)
    feasible_df = train_df[mask].copy()
    if args.max_rows > 0 and len(feasible_df) > args.max_rows:
        feasible_df = feasible_df.sample(n=int(args.max_rows), random_state=int(args.seed)).reset_index(drop=True)

    out_df = feasible_df.drop(columns=["source_norm", "target_norm"])
    out_df.to_csv(output_train, index=False)

    report = {
        "config_path": str(cfg_path),
        "input_train": str(input_train),
        "output_train": str(output_train),
        "model_name": model_name,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "input_rows": int(len(train_df)),
        "feasible_rows_before_cap": int(mask.sum()),
        "output_rows": int(len(out_df)),
        "output_ratio_pct": 100.0 * float(len(out_df)) / float(max(1, len(train_df))),
        "source_trunc_pct_input": 100.0 * float((train_df["source_tok_len"] > max_source_length).mean()),
        "target_trunc_pct_input": 100.0 * float((train_df["target_tok_len"] > max_target_length).mean()),
        "source_trunc_pct_output": 100.0 * float((out_df["source_tok_len"] > max_source_length).mean()),
        "target_trunc_pct_output": 100.0 * float((out_df["target_tok_len"] > max_target_length).mean()),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    print(f"OK: wrote {report_json}")
    print(
        "INFO: rows input/feasible/output="
        f"{report['input_rows']}/{report['feasible_rows_before_cap']}/{report['output_rows']}"
    )
    print(
        "INFO: trunc_pct input(src/tgt)="
        f"{report['source_trunc_pct_input']:.2f}/{report['target_trunc_pct_input']:.2f}"
    )
    print(
        "INFO: trunc_pct output(src/tgt)="
        f"{report['source_trunc_pct_output']:.2f}/{report['target_trunc_pct_output']:.2f}"
    )


if __name__ == "__main__":
    main()
