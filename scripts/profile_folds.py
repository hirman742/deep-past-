from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING_MARKER_RE = re.compile(r"(?:\bx+\b|\.{3,}|[\[\]⸢⸣])", flags=re.IGNORECASE)
SPECIAL_SYMBOL_RE = re.compile(r"[\{\}\[\]\-]")


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


def _safe_density(text: str) -> float:
    if not text:
        return 0.0
    return float(len(SPECIAL_SYMBOL_RE.findall(text))) / float(max(1, len(text)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_path = processed_dir / "train_proc.csv"
    folds_path = processed_dir / "folds.csv"

    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    merged = train_df.merge(folds_df[["oare_id", "fold", "group_key"]], on="oare_id", how="inner")
    if merged.empty:
        raise ValueError("Merged train/folds frame is empty")

    tokenizer_name = str(model_cfg.get("name", "google/mt5-small"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    merged["source"] = merged["source"].fillna("").astype(str)
    merged["target"] = merged["target"].fillna("").astype(str)
    merged["src_char_len"] = merged["source"].map(len)
    merged["tgt_char_len"] = merged["target"].map(len)
    merged["src_tok_len"] = merged["source"].map(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
    merged["tgt_tok_len"] = merged["target"].map(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
    merged["special_symbol_density"] = merged["source"].map(_safe_density)
    merged["missing_marker"] = merged["source"].map(lambda x: bool(MISSING_MARKER_RE.search(x)))

    rows: list[dict[str, Any]] = []
    for fold, frame in merged.groupby("fold"):
        group_counts = frame["group_key"].astype(str).value_counts()
        top_group_share_pct = 100.0 * float(group_counts.iloc[0]) / float(len(frame))
        rows.append(
            {
                "fold": int(fold),
                "rows": int(len(frame)),
                "group_unique": int(frame["group_key"].nunique()),
                "top_group_share_pct": float(top_group_share_pct),
                "src_char_mean": float(frame["src_char_len"].mean()),
                "src_char_p95": float(frame["src_char_len"].quantile(0.95)),
                "tgt_char_mean": float(frame["tgt_char_len"].mean()),
                "tgt_char_p95": float(frame["tgt_char_len"].quantile(0.95)),
                "src_tok_mean": float(frame["src_tok_len"].mean()),
                "src_tok_p95": float(frame["src_tok_len"].quantile(0.95)),
                "tgt_tok_mean": float(frame["tgt_tok_len"].mean()),
                "tgt_tok_p95": float(frame["tgt_tok_len"].quantile(0.95)),
                "special_symbol_density_mean": float(frame["special_symbol_density"].mean()),
                "missing_marker_ratio_pct": 100.0 * float(frame["missing_marker"].mean()),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    out_dir = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0").parent
    csv_path = out_dir / "fold_profile.csv"
    json_path = out_dir / "fold_profile_summary.json"
    result_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "config_path": str(cfg_path),
                "tokenizer_name": tokenizer_name,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"OK: wrote {csv_path}")
    print(f"OK: wrote {json_path}")


if __name__ == "__main__":
    main()
