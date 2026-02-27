from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, KFold
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


def _normalize_text(text: str, *, apply_t0: bool, cfg: dict[str, Any], strip_text: bool, fold_ws: bool, lowercase: bool) -> str:
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


def _build_source_buckets(df: pd.DataFrame, *, source_col: str = "source") -> pd.Series:
    values = df[source_col].fillna("").astype(str)
    compact = values.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    head_tokens = compact.str.split().str[:3].str.join("|")
    char_bin = (compact.str.len() // 120).astype(int).clip(lower=0, upper=50).astype(str)
    symbol_bin = (
        compact.str.count(r"[\[\]{}<>\-]")
        .astype(int)
        .floordiv(5)
        .clip(lower=0, upper=50)
        .astype(str)
    )
    return head_tokens.fillna("NA") + "|l" + char_bin + "|s" + symbol_bin


def _choose_group_column(df: pd.DataFrame, strategy: str, metadata_path: Path | None) -> tuple[pd.Series, str]:
    id_col = "oare_id"
    if strategy not in {"auto", "oare_prefix", "kfold", "publication_catalog", "source_bucket"}:
        raise ValueError(f"Unsupported group_strategy={strategy!r}")

    if strategy == "kfold":
        return pd.Series(np.arange(len(df), dtype=np.int64), index=df.index), "row_index"

    if strategy == "oare_prefix":
        return df[id_col].astype(str).str.slice(0, 8), "oare_id_prefix8"

    if strategy == "publication_catalog":
        if metadata_path is None or not metadata_path.exists():
            raise FileNotFoundError("group_strategy=publication_catalog requires a metadata csv path")
        meta = pd.read_csv(metadata_path, usecols=["oare_id", "publication_catalog"])
        merged = df[[id_col]].merge(meta, on=id_col, how="left")
        groups = merged["publication_catalog"].fillna("NA").astype(str)
        return groups, "publication_catalog"

    if strategy == "source_bucket":
        return _build_source_buckets(df, source_col="source"), "source_bucket"

    def _group_quality(groups: pd.Series) -> float:
        if len(groups) == 0:
            return 1.0
        return float(groups.nunique(dropna=False)) / float(len(groups))

    direct_candidates = ["text_id", "publication_catalog", "cdli_id"]
    for col in direct_candidates:
        if col in df.columns and df[col].nunique(dropna=True) >= 5:
            groups = df[col].fillna("NA").astype(str)
            if _group_quality(groups) < 0.98:
                return groups, col

    if metadata_path is not None and metadata_path.exists():
        use_cols = ["oare_id", "publication_catalog"]
        meta = pd.read_csv(metadata_path, usecols=use_cols)
        merged = df[[id_col]].merge(meta, on=id_col, how="left")
        groups = merged["publication_catalog"].fillna("NA").astype(str)
        if groups.nunique() >= 5 and _group_quality(groups) < 0.98:
            return groups, "publication_catalog"

    prefix_groups = df[id_col].astype(str).str.slice(0, 8)
    if _group_quality(prefix_groups) < 0.98:
        return prefix_groups, "oare_id_prefix8"
    return _build_source_buckets(df, source_col="source"), "source_bucket"


def _assign_folds(df: pd.DataFrame, *, n_splits: int, seed: int, groups: pd.Series) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    folds = np.full(len(df), fill_value=-1, dtype=np.int64)

    unique_ratio = float(groups.nunique(dropna=False)) / float(max(1, len(groups)))
    use_group = groups.nunique() >= n_splits and unique_ratio < 0.995
    if use_group:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(df)

    for fold_id, (_, val_idx) in enumerate(split_iter):
        folds[val_idx] = fold_id

    out = pd.DataFrame(
        {
            "oare_id": df["oare_id"].astype(str).values,
            "fold": folds,
            "group_key": groups.astype(str).values,
            "group_kind": "groupkfold" if use_group else "kfold",
        }
    )
    return out


def _percentile(values: list[int], pct: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def _build_length_stats(
    tokenizer_name: str,
    source_lens: list[int],
    target_lens: list[int],
    *,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    src_trunc_ratio = 100.0 * float(np.mean(np.asarray(source_lens) > max_source_length))
    tgt_trunc_ratio = 100.0 * float(np.mean(np.asarray(target_lens) > max_target_length))

    return {
        "tokenizer_name": tokenizer_name,
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "source": {
            "p95": _percentile(source_lens, 95),
            "p99": _percentile(source_lens, 99),
            "max": int(max(source_lens) if source_lens else 0),
            "truncation_ratio_pct": src_trunc_ratio,
        },
        "target": {
            "p95": _percentile(target_lens, 95),
            "p99": _percentile(target_lens, 99),
            "max": int(max(target_lens) if target_lens else 0),
            "truncation_ratio_pct": tgt_trunc_ratio,
        },
    }


def _compute_token_lengths(
    *,
    tokenizer_name: str,
    train_df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
) -> tuple[list[int], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    src_ids = tokenizer(
        train_df[src_col].tolist(),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    tgt_ids = tokenizer(
        text_target=train_df[tgt_col].tolist(),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    return [len(x) for x in src_ids], [len(x) for x in tgt_ids]


def _parse_length_candidates(
    *,
    preprocess_cfg: dict[str, Any],
    default_src_len: int,
    default_tgt_len: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = [(int(default_src_len), int(default_tgt_len))]
    raw = preprocess_cfg.get("length_candidates") or []
    for item in raw:
        src_len = None
        tgt_len = None
        if isinstance(item, dict):
            src_len = item.get("max_source_length")
            tgt_len = item.get("max_target_length")
        elif isinstance(item, str) and "x" in item.lower():
            lhs, rhs = item.lower().split("x", 1)
            src_len = lhs.strip()
            tgt_len = rhs.strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            src_len, tgt_len = item
        if src_len is None or tgt_len is None:
            continue
        try:
            pair = (int(src_len), int(tgt_len))
        except (TypeError, ValueError):
            continue
        if pair not in out:
            out.append(pair)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-test-rows", type=int, default=0)
    args = ap.parse_args()

    config_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(config_path)

    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    train_path = _resolve_path(paths_cfg.get("train_csv"), REPO_ROOT / "data" / "interim" / "t0_train.csv")
    test_path = _resolve_path(paths_cfg.get("test_csv"), REPO_ROOT / "data" / "interim" / "t0_test.csv")
    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    metadata_path = _resolve_path(paths_cfg.get("metadata_csv"), REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv")

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if args.max_train_rows > 0:
        train_df = train_df.iloc[: args.max_train_rows].copy()
    if args.max_test_rows > 0:
        test_df = test_df.iloc[: args.max_test_rows].copy()

    required_train = {"oare_id", "transliteration", "translation"}
    required_test = {"id", "transliteration"}
    missing_train = sorted(required_train - set(train_df.columns))
    missing_test = sorted(required_test - set(test_df.columns))
    if missing_train:
        raise KeyError(f"Missing train columns: {missing_train}")
    if missing_test:
        raise KeyError(f"Missing test columns: {missing_test}")

    apply_t0 = bool(preprocess_cfg.get("apply_t0_normalize", False))
    strip_text = bool(preprocess_cfg.get("strip_text", True))
    fold_ws = bool(preprocess_cfg.get("fold_inline_whitespace", True))
    lowercase_source = bool(preprocess_cfg.get("lowercase_source", False))
    lowercase_target = bool(preprocess_cfg.get("lowercase_target", False))
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))
    cleaning_cfg = _load_cleaning_config()

    train_df["source_raw"] = train_df["transliteration"].fillna("").astype(str)
    train_df["target_raw"] = train_df["translation"].fillna("").astype(str)
    train_df["source"] = train_df["source_raw"].map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
        )
    ).map(lambda s: apply_task_prefix(s, task_prefix))
    train_df["target"] = train_df["target_raw"].map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_target,
        )
    )

    test_df["source_raw"] = test_df["transliteration"].fillna("").astype(str)
    test_df["source"] = test_df["source_raw"].map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
        )
    ).map(lambda s: apply_task_prefix(s, task_prefix))

    folds = int(preprocess_cfg.get("folds", 5))
    seed = int(cfg.get("seed", 42))
    group_strategy = str(preprocess_cfg.get("group_strategy", "auto"))
    groups, group_source = _choose_group_column(train_df, group_strategy, metadata_path if metadata_path.exists() else None)
    folds_df = _assign_folds(train_df, n_splits=folds, seed=seed, groups=groups)
    folds_df["group_source"] = group_source

    tokenizer_name = str(model_cfg.get("name", "google/mt5-small"))
    max_src_len = int(model_cfg.get("max_source_length", 256))
    max_tgt_len = int(model_cfg.get("max_target_length", 192))
    src_lens, tgt_lens = _compute_token_lengths(
        tokenizer_name=tokenizer_name,
        train_df=train_df,
        src_col="source",
        tgt_col="target",
    )
    length_stats = _build_length_stats(
        tokenizer_name,
        src_lens,
        tgt_lens,
        max_source_length=max_src_len,
        max_target_length=max_tgt_len,
    )
    length_stats["group_source"] = group_source
    length_stats["task_prefix"] = task_prefix
    length_stats["group_unique"] = int(groups.nunique(dropna=False))
    length_stats["group_unique_ratio"] = float(groups.nunique(dropna=False)) / float(max(1, len(groups)))
    length_stats["group_kind"] = str(folds_df["group_kind"].iloc[0])
    length_candidates = _parse_length_candidates(
        preprocess_cfg=preprocess_cfg,
        default_src_len=max_src_len,
        default_tgt_len=max_tgt_len,
    )
    length_stats["candidates"] = [
        _build_length_stats(
            tokenizer_name,
            src_lens,
            tgt_lens,
            max_source_length=src_len,
            max_target_length=tgt_len,
        )
        for src_len, tgt_len in length_candidates
    ]

    train_out = processed_dir / "train_proc.csv"
    test_out = processed_dir / "test_proc.csv"
    folds_out = processed_dir / "folds.csv"
    stats_out = processed_dir / "length_stats.json"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    folds_df.to_csv(folds_out, index=False)
    stats_out.write_text(json.dumps(length_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {train_out}")
    print(f"OK: wrote {test_out}")
    print(f"OK: wrote {folds_out}")
    print(f"OK: wrote {stats_out}")
    print(f"INFO: group_source={group_source}, group_kind={folds_df['group_kind'].iloc[0]}")
    print(
        "INFO: truncation_pct(src/tgt)="
        f"{length_stats['source']['truncation_ratio_pct']:.2f}/{length_stats['target']['truncation_ratio_pct']:.2f}"
    )
    print(
        "INFO: group_unique_ratio="
        f"{length_stats['group_unique_ratio']:.4f}, task_prefix={'ON' if task_prefix else 'OFF'}"
    )


if __name__ == "__main__":
    main()
