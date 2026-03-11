from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generation_utils import apply_task_prefix, build_bad_words_ids, build_generate_kwargs, normalize_task_prefix  # noqa: E402


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


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_base_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(processed_dir / "folds.csv")
    merged = train_df.merge(folds_df, on="oare_id", how="inner", suffixes=("", "_fold"))
    return train_df, folds_df, merged


def _prefix_from_config(cfg: dict[str, Any]) -> str:
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    return normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))


def _strip_prefix(text: str, prefix: str) -> str:
    value = text if isinstance(text, str) else ""
    if prefix and value.startswith(prefix):
        return value[len(prefix) :].lstrip()
    return value


def _concat_chunks(values: list[str]) -> str:
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    return "\n".join(cleaned).strip()


def _token_len(tokenizer, text: str, *, as_target: bool) -> int:
    if as_target:
        ids = tokenizer(text_target=[text], truncation=False, add_special_tokens=True)["input_ids"][0]
    else:
        ids = tokenizer([text], truncation=False, add_special_tokens=True)["input_ids"][0]
    return int(len(ids))


def _copy_optional(src: pd.Series, name: str, default: Any = "") -> Any:
    return src[name] if name in src.index else default


def _make_row(
    *,
    oare_id: str,
    parent_oare_id: str,
    source_text: str,
    target_text: str,
    task_prefix: str,
    chunk_index: int,
    chunk_total: int,
    chunk_mode: str,
    transliteration: str = "",
    translation: str = "",
) -> dict[str, Any]:
    translit = transliteration or source_text
    trans = translation or target_text
    return {
        "oare_id": oare_id,
        "transliteration": translit,
        "translation": trans,
        "source_raw": source_text,
        "target_raw": target_text,
        "source": apply_task_prefix(source_text, task_prefix),
        "target": target_text,
        "parent_oare_id": parent_oare_id,
        "chunk_index": int(chunk_index),
        "chunk_total": int(chunk_total),
        "is_chunk": True,
        "chunk_mode": chunk_mode,
        "is_short_aligned": "",
        "short_align_mode": "",
        "source_oare_id": "",
        "align_type": "",
        "align_cost": "",
    }


def _make_fold_row(
    *,
    oare_id: str,
    template: pd.Series,
    parent_oare_id: str,
    chunk_index: int,
    chunk_total: int,
    chunk_mode: str,
) -> dict[str, Any]:
    return {
        "oare_id": oare_id,
        "fold": int(_copy_optional(template, "fold", 0)),
        "group_key": str(_copy_optional(template, "group_key", "")),
        "group_kind": str(_copy_optional(template, "group_kind", "groupkfold")),
        "group_source": str(_copy_optional(template, "group_source", "source_bucket")),
        "parent_oare_id": parent_oare_id,
        "chunk_index": int(chunk_index),
        "chunk_total": int(chunk_total),
        "chunk_mode": chunk_mode,
        "short_align_mode": "",
        "align_type": "",
    }


def _best_window_end(
    *,
    rows: list[dict[str, Any]],
    start: int,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
    max_chunks: int,
) -> int:
    best_end = start + 1
    upper = min(len(rows), start + max_chunks)
    for end in range(start + 1, upper + 1):
        source_text = _concat_chunks([rows[idx]["source_piece"] for idx in range(start, end)])
        target_text = _concat_chunks([rows[idx]["target_piece"] for idx in range(start, end)])
        if not source_text or not target_text:
            continue
        src_len = _token_len(tokenizer, source_text, as_target=False)
        tgt_len = _token_len(tokenizer, target_text, as_target=True)
        if src_len <= max_source_length and tgt_len <= max_target_length:
            best_end = end
        else:
            break
    return best_end


def build_parentpack(args: argparse.Namespace) -> None:
    cfg = _load_yaml(_resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml"))
    processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "data" / "processed_taskform_parentpack_fold0")
    train_df, folds_df, merged = _load_base_processed(processed_dir)

    model_cfg = cfg.get("model", {}) or {}
    task_prefix = _prefix_from_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg.get("name", "google/byt5-small")))
    max_source_length = int(model_cfg.get("max_source_length", 640))
    max_target_length = int(model_cfg.get("max_target_length", 640))

    work = merged.copy()
    work["source_piece"] = work["source"].fillna("").astype(str).map(lambda x: _strip_prefix(x, task_prefix))
    work["target_piece"] = work["target"].fillna("").astype(str)
    if "chunk_index" in work.columns:
        work["chunk_index"] = work["chunk_index"].fillna(0).astype(int)
    else:
        work["chunk_index"] = 0
    if "parent_oare_id" not in work.columns:
        work["parent_oare_id"] = work["oare_id"].astype(str)

    out_rows: list[dict[str, Any]] = []
    out_folds: list[dict[str, Any]] = []
    mode_stats: dict[str, int] = {"single": 0, "parentpack_2_3": 0, "parentwindow_3ofN": 0}
    pack_size_stats: list[int] = []

    for parent_id, group in work.groupby("parent_oare_id", sort=False):
        ordered = group.sort_values("chunk_index").reset_index(drop=True)
        payload = ordered.to_dict(orient="records")
        windows: list[tuple[int, int, str]] = []
        if len(payload) <= 1:
            windows.append((0, 1, "single"))
        elif len(payload) <= 3:
            start = 0
            while start < len(payload):
                end = _best_window_end(
                    rows=payload,
                    start=start,
                    tokenizer=tokenizer,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    max_chunks=3,
                )
                windows.append((start, end, "parentpack_2_3"))
                start = end
        else:
            for start in range(len(payload)):
                end = _best_window_end(
                    rows=payload,
                    start=start,
                    tokenizer=tokenizer,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    max_chunks=3,
                )
                windows.append((start, end, "parentwindow_3ofN"))
        deduped: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int, str]] = set()
        for window in windows:
            if window not in seen:
                seen.add(window)
                deduped.append(window)

        chunk_total = len(deduped)
        for pack_idx, (start, end, mode) in enumerate(deduped):
            subset = payload[start:end]
            source_text = _concat_chunks([row["source_piece"] for row in subset])
            target_text = _concat_chunks([row["target_piece"] for row in subset])
            oare_id = f"{parent_id}__p{pack_idx + 1}of{chunk_total}"
            first = ordered.iloc[0]
            out_rows.append(
                _make_row(
                    oare_id=oare_id,
                    parent_oare_id=str(parent_id),
                    source_text=source_text,
                    target_text=target_text,
                    task_prefix=task_prefix,
                    chunk_index=pack_idx,
                    chunk_total=chunk_total,
                    chunk_mode=mode,
                    transliteration=source_text,
                    translation=target_text,
                )
            )
            out_folds.append(
                _make_fold_row(
                    oare_id=oare_id,
                    template=first,
                    parent_oare_id=str(parent_id),
                    chunk_index=pack_idx,
                    chunk_total=chunk_total,
                    chunk_mode=mode,
                )
            )
            mode_stats[mode] = mode_stats.get(mode, 0) + 1
            pack_size_stats.append(int(end - start))

    out_train = pd.DataFrame(out_rows)
    out_folds_df = pd.DataFrame(out_folds)
    _write_csv(out_dir / "train_proc.csv", out_train)
    _write_csv(out_dir / "folds.csv", out_folds_df)

    audit = {
        "mode": "parentpack",
        "base_processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "parents": int(work["parent_oare_id"].nunique()),
        "rows_in": int(len(train_df)),
        "rows_out": int(len(out_train)),
        "mode_stats": mode_stats,
        "avg_pack_size": float(sum(pack_size_stats) / max(1, len(pack_size_stats))),
        "max_pack_size": int(max(pack_size_stats) if pack_size_stats else 0),
        "fold": int(args.fold),
    }
    (out_dir / "audit_parentpack.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_dir/'train_proc.csv'}")
    print(f"OK: wrote {out_dir/'folds.csv'}")
    print(f"OK: wrote {out_dir/'audit_parentpack.json'}")


def build_replay(args: argparse.Namespace) -> None:
    cfg = _load_yaml(_resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml"))
    processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "data" / "processed_taskform_replay_fold0")
    train_df, folds_df, merged = _load_base_processed(processed_dir)

    model_cfg = cfg.get("model", {}) or {}
    task_prefix = _prefix_from_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg.get("name", "google/byt5-small")))

    work = merged.copy()
    work["source_piece"] = work["source"].fillna("").astype(str).map(lambda x: _strip_prefix(x, task_prefix))
    work["target_piece"] = work["target"].fillna("").astype(str)
    work["chunk_total"] = work["chunk_total"].fillna(1).astype(int)
    train_mask = work["fold"].astype(int) != int(args.fold)
    train_only = work.loc[train_mask].reset_index(drop=True)
    source_lens = [int(len(x)) for x in tokenizer(train_only["source_piece"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]]
    target_lens = [int(len(x)) for x in tokenizer(text_target=train_only["target_piece"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]]
    hard_mask = (
        (train_only["chunk_total"] >= 4)
        | (pd.Series(target_lens) >= 129)
        | train_only["source_piece"].fillna("").astype(str).str.contains(r"<gap>|\{|\[", regex=True)
    )
    hard_rows = train_only.loc[hard_mask].copy().reset_index(drop=True)
    if hard_rows.empty:
        raise ValueError("hard-case replay produced empty candidate set")

    extra_n = max(1, int(round(len(train_only) * float(args.ratio))))
    sampled = hard_rows.sample(n=extra_n, replace=True, random_state=int(args.seed)).reset_index(drop=True)

    dup_train_rows: list[dict[str, Any]] = []
    dup_fold_rows: list[dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        new_id = f"{row['oare_id']}__replay{idx + 1:05d}"
        row_dict = row[train_df.columns].to_dict()
        row_dict["oare_id"] = new_id
        dup_train_rows.append(row_dict)

        fold_row = {
            "oare_id": new_id,
            "fold": int(row["fold"]),
            "group_key": str(_copy_optional(row, "group_key", "")),
            "group_kind": str(_copy_optional(row, "group_kind", "groupkfold")),
            "group_source": str(_copy_optional(row, "group_source", "source_bucket")),
            "parent_oare_id": str(_copy_optional(row, "parent_oare_id", "")),
            "chunk_index": int(_copy_optional(row, "chunk_index", 0)),
            "chunk_total": int(_copy_optional(row, "chunk_total", 1)),
            "chunk_mode": str(_copy_optional(row, "chunk_mode", "")),
            "short_align_mode": str(_copy_optional(row, "short_align_mode", "")),
            "align_type": str(_copy_optional(row, "align_type", "")),
        }
        dup_fold_rows.append(fold_row)

    out_train = pd.concat([train_df, pd.DataFrame(dup_train_rows)], ignore_index=True)
    out_folds_df = pd.concat([folds_df, pd.DataFrame(dup_fold_rows)], ignore_index=True)
    _write_csv(out_dir / "train_proc.csv", out_train)
    _write_csv(out_dir / "folds.csv", out_folds_df)

    audit = {
        "mode": "hardcase_replay",
        "base_processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "fold": int(args.fold),
        "ratio": float(args.ratio),
        "train_rows_before": int(len(train_only)),
        "hard_rows": int(len(hard_rows)),
        "extra_rows_added": int(extra_n),
        "criteria": {
            "chunk_total_ge_4": int((train_only["chunk_total"] >= 4).sum()),
            "target_len_ge_129": int((pd.Series(target_lens) >= 129).sum()),
            "has_gap_or_brace_or_bracket": int(
                train_only["source_piece"].fillna("").astype(str).str.contains(r"<gap>|\{|\[", regex=True).sum()
            ),
        },
    }
    (out_dir / "audit_hardcase_replay.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_dir/'train_proc.csv'}")
    print(f"OK: wrote {out_dir/'folds.csv'}")
    print(f"OK: wrote {out_dir/'audit_hardcase_replay.json'}")


def build_hardcase_selector(args: argparse.Namespace) -> None:
    cfg = _load_yaml(_resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml"))
    processed_dir = _resolve_path(args.processed_dir, REPO_ROOT / "data" / "processed_taskform_replay25_fold0")
    out_dir = _resolve_path(args.out_dir, REPO_ROOT / "data" / "processed_taskform_replay25_hardselector_fold0")
    train_df, folds_df, merged = _load_base_processed(processed_dir)

    model_cfg = cfg.get("model", {}) or {}
    task_prefix = _prefix_from_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg.get("name", "google/byt5-small")))

    work = merged.copy()
    work["source_piece"] = work["source"].fillna("").astype(str).map(lambda x: _strip_prefix(x, task_prefix))
    work["target_piece"] = work["target"].fillna("").astype(str)
    work["chunk_total"] = work["chunk_total"].fillna(1).astype(int)
    if "parent_oare_id" not in work.columns:
        work["parent_oare_id"] = work["oare_id"].astype(str)

    train_only = work.loc[work["fold"].astype(int) != int(args.fold)].copy()
    val_only = work.loc[work["fold"].astype(int) == int(args.fold)].copy().reset_index(drop=True)
    if val_only.empty:
        raise ValueError(f"Fold {args.fold} has empty validation split in {processed_dir}")

    src_lens = [
        int(len(x))
        for x in tokenizer(val_only["source_piece"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]
    ]
    tgt_lens = [
        int(len(x))
        for x in tokenizer(text_target=val_only["target_piece"].tolist(), truncation=False, add_special_tokens=True)["input_ids"]
    ]
    marker_mask = val_only["source_piece"].fillna("").astype(str).str.contains(r"<gap>|\{|\[", regex=True)
    chunk_mask = val_only["chunk_total"] >= int(args.min_chunk_total)
    tgt_mask = pd.Series(tgt_lens, index=val_only.index) >= int(args.target_len_threshold)

    val_only["selector_score"] = (
        (4.0 * chunk_mask.astype(float))
        + (2.0 * tgt_mask.astype(float))
        + (1.0 * marker_mask.astype(float))
        + pd.Series([min(1.5, t / max(1, int(args.target_len_threshold))) for t in tgt_lens], index=val_only.index)
    )
    val_only["target_tok_len"] = tgt_lens
    val_only["source_tok_len"] = src_lens
    val_only["selector_bucket"] = [
        "|".join(
            part
            for part, enabled in (
                ("chunk4plus", bool(is_chunk)),
                ("long_tgt", bool(is_long)),
                ("marker", bool(has_marker)),
            )
            if enabled
        )
        or "other"
        for is_chunk, is_long, has_marker in zip(chunk_mask.tolist(), tgt_mask.tolist(), marker_mask.tolist())
    ]

    ranked = val_only.sort_values(
        ["selector_score", "target_tok_len", "chunk_total", "source_tok_len"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    selector_max_rows = int(args.max_val_rows)
    if selector_max_rows > 0:
        ranked = ranked.head(selector_max_rows).reset_index(drop=True)

    keep_ids = set(train_only["oare_id"].astype(str).tolist()) | set(ranked["oare_id"].astype(str).tolist())
    out_train = train_df.loc[train_df["oare_id"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    out_folds_df = folds_df.loc[folds_df["oare_id"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    _write_csv(out_dir / "train_proc.csv", out_train)
    _write_csv(out_dir / "folds.csv", out_folds_df)

    audit = {
        "mode": "hardcase_selector",
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "fold": int(args.fold),
        "train_rows_kept": int(len(train_only)),
        "val_rows_before": int(len(val_only)),
        "val_rows_after": int(len(ranked)),
        "criteria": {
            "min_chunk_total": int(args.min_chunk_total),
            "target_len_threshold": int(args.target_len_threshold),
            "marker_regex": r"<gap>|\{|\[",
        },
        "bucket_counts": ranked["selector_bucket"].value_counts().to_dict(),
        "score_stats": {
            "score_p50": float(ranked["selector_score"].median()) if not ranked.empty else 0.0,
            "target_tok_p50": float(ranked["target_tok_len"].median()) if not ranked.empty else 0.0,
            "target_tok_p90": float(ranked["target_tok_len"].quantile(0.9)) if not ranked.empty else 0.0,
        },
    }
    (out_dir / "audit_hardcase_selector.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_dir/'train_proc.csv'}")
    print(f"OK: wrote {out_dir/'folds.csv'}")
    print(f"OK: wrote {out_dir/'audit_hardcase_selector.json'}")


def _score_proxy_row(text: str, genre: str) -> float:
    value = text if isinstance(text, str) else ""
    words = [x for x in value.split() if x]
    has_gap = 1.0 if "<gap>" in value else 0.0
    has_brace = 1.0 if "{" in value else 0.0
    has_sub = 1.0 if re.search(r"[₀₁₂₃₄₅₆₇₈₉ₓ]", value) else 0.0
    word_score = min(4.0, len(words) / 40.0)
    genre_norm = (genre or "").strip().lower()
    genre_boost = 1.0 if genre_norm in {"letter", "debt note", "account", "memorandum"} else 0.0
    return (4.0 * has_gap) + (2.0 * has_brace) + has_sub + word_score + genre_boost


def _generate_pseudo_predictions(
    *,
    cfg: dict[str, Any],
    checkpoint_dir: Path,
    transliterations: list[str],
    batch_size: int,
    num_beams: int,
    length_penalty: float,
    max_new_tokens: int,
) -> list[str]:
    model_cfg = cfg.get("model", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    model_name = str(model_cfg.get("name", "google/byt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 640))
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(gen_cfg.get("suppress_extra_ids", True)),
        bad_tokens_regex=str(gen_cfg.get("bad_tokens_regex", r"<extra_id_\d+>")),
    )
    generate_kwargs = build_generate_kwargs(
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=0,
        no_repeat_ngram_size=0,
        bad_words_ids=bad_words_ids,
    )

    prefixed = [apply_task_prefix(str(x).strip(), task_prefix) for x in transliterations]
    predictions: list[str] = []
    with torch.no_grad():
        for i in range(0, len(prefixed), max(1, batch_size)):
            batch = prefixed[i : i + max(1, batch_size)]
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            generated = model.generate(**tokenized, **generate_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([x.strip() for x in decoded])
    return predictions


def build_proxy_mix(args: argparse.Namespace) -> None:
    cfg = _load_yaml(_resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml"))
    processed_dir = _resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    published_path = _resolve_path(
        args.published_csv,
        REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv",
    )
    checkpoint_dir = _resolve_path(
        args.winner_checkpoint_dir,
        REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
    )
    out_root = _resolve_path(args.out_root, REPO_ROOT / "data" / "processed_taskform_proxymix")

    train_df, folds_df, merged = _load_base_processed(processed_dir)
    base_train_rows = int((merged["fold"].astype(int) != int(args.fold)).sum())

    published = pd.read_csv(published_path)
    if "transliteration" not in published.columns:
        raise KeyError(f"published csv missing transliteration column: {published_path}")
    if "oare_id" not in published.columns:
        raise KeyError(f"published csv missing oare_id column: {published_path}")

    existing_ids = set(train_df["parent_oare_id"].fillna(train_df["oare_id"]).astype(str).tolist())
    pool = published[["oare_id", "transliteration", "genre_label"]].copy()
    pool["transliteration"] = pool["transliteration"].fillna("").astype(str)
    pool = pool[pool["transliteration"].str.strip() != ""].reset_index(drop=True)
    pool = pool[~pool["oare_id"].astype(str).isin(existing_ids)].reset_index(drop=True)
    pool["score"] = [
        _score_proxy_row(text, genre)
        for text, genre in zip(pool["transliteration"].tolist(), pool["genre_label"].fillna("").astype(str).tolist())
    ]
    pool["word_count"] = pool["transliteration"].map(lambda x: len([w for w in str(x).split() if w]))
    pool = pool.sort_values(["score", "word_count"], ascending=[False, False]).reset_index(drop=True)

    ratios = [float(x) for x in str(args.ratios).split(",") if str(x).strip()]
    max_extra = max(max(1, int(round(base_train_rows * ratio))) for ratio in ratios)
    pool_size = min(len(pool), max(args.min_pool, max_extra * 3))
    pool = pool.head(pool_size).reset_index(drop=True)

    predictions = _generate_pseudo_predictions(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        transliterations=pool["transliteration"].tolist(),
        batch_size=int(args.predict_batch_size),
        num_beams=int(args.num_beams),
        length_penalty=float(args.length_penalty),
        max_new_tokens=int(args.max_new_tokens),
    )
    pool["pseudo_translation"] = predictions
    pool = pool[pool["pseudo_translation"].fillna("").astype(str).str.strip() != ""].reset_index(drop=True)

    task_prefix = _prefix_from_config(cfg)
    audit: dict[str, Any] = {
        "mode": "proxy_mix",
        "published_csv": str(published_path),
        "checkpoint_dir": str(checkpoint_dir),
        "base_processed_dir": str(processed_dir),
        "ratios": ratios,
        "pool_size": int(len(pool)),
        "base_train_rows": base_train_rows,
        "outputs": {},
    }

    for ratio in ratios:
        extra_n = max(1, int(round(base_train_rows * ratio)))
        sample = pool.head(extra_n).copy().reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        ratio_tag = f"{int(round(ratio * 1000)):03d}"
        out_dir = out_root.parent / f"{out_root.name}_{ratio_tag}"
        for idx, row in sample.iterrows():
            oare_id = f"proxy_{ratio_tag}_{idx + 1:05d}_{row['oare_id']}"
            source_text = str(row["transliteration"]).strip()
            target_text = str(row["pseudo_translation"]).strip()
            rows.append(
                _make_row(
                    oare_id=oare_id,
                    parent_oare_id=oare_id,
                    source_text=source_text,
                    target_text=target_text,
                    task_prefix=task_prefix,
                    chunk_index=0,
                    chunk_total=1,
                    chunk_mode=f"proxy_mix_{ratio_tag}",
                    transliteration=source_text,
                    translation=target_text,
                )
            )
            fold_rows.append(
                {
                    "oare_id": oare_id,
                    "fold": 99,
                    "group_key": f"proxy_mix_{ratio_tag}",
                    "group_kind": "proxy_mix",
                    "group_source": "published_texts_pseudo",
                    "parent_oare_id": oare_id,
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "chunk_mode": f"proxy_mix_{ratio_tag}",
                    "short_align_mode": "",
                    "align_type": "",
                }
            )
        out_train = pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)
        out_folds_df = pd.concat([folds_df, pd.DataFrame(fold_rows)], ignore_index=True)
        _write_csv(out_dir / "train_proc.csv", out_train)
        _write_csv(out_dir / "folds.csv", out_folds_df)
        audit["outputs"][ratio_tag] = {
            "ratio": float(ratio),
            "out_dir": str(out_dir),
            "rows_added": int(len(rows)),
        }
        print(f"OK: wrote {out_dir/'train_proc.csv'}")
        print(f"OK: wrote {out_dir/'folds.csv'}")

    (out_root.parent / f"{out_root.name}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_root.parent / f'{out_root.name}_audit.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_parent = sub.add_parser("parentpack")
    ap_parent.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap_parent.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap_parent.add_argument("--out-dir", required=True)
    ap_parent.add_argument("--fold", type=int, default=0)

    ap_replay = sub.add_parser("replay")
    ap_replay.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap_replay.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap_replay.add_argument("--out-dir", required=True)
    ap_replay.add_argument("--fold", type=int, default=0)
    ap_replay.add_argument("--ratio", type=float, required=True)
    ap_replay.add_argument("--seed", type=int, default=42)

    ap_hsel = sub.add_parser("hardcase_selector")
    ap_hsel.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap_hsel.add_argument("--processed-dir", required=True)
    ap_hsel.add_argument("--out-dir", required=True)
    ap_hsel.add_argument("--fold", type=int, default=0)
    ap_hsel.add_argument("--min-chunk-total", type=int, default=4)
    ap_hsel.add_argument("--target-len-threshold", type=int, default=129)
    ap_hsel.add_argument("--max-val-rows", type=int, default=192)

    ap_proxy = sub.add_parser("proxy_mix")
    ap_proxy.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap_proxy.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap_proxy.add_argument("--winner-checkpoint-dir", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250")
    ap_proxy.add_argument("--published-csv", default="deep-past-initiative-machine-translation/published_texts.csv")
    ap_proxy.add_argument("--out-root", required=True)
    ap_proxy.add_argument("--fold", type=int, default=0)
    ap_proxy.add_argument("--ratios", default="0.05,0.10")
    ap_proxy.add_argument("--predict-batch-size", type=int, default=8)
    ap_proxy.add_argument("--num-beams", type=int, default=4)
    ap_proxy.add_argument("--length-penalty", type=float, default=0.7)
    ap_proxy.add_argument("--max-new-tokens", type=int, default=384)
    ap_proxy.add_argument("--min-pool", type=int, default=300)

    args = ap.parse_args()
    if args.cmd == "parentpack":
        build_parentpack(args)
    elif args.cmd == "replay":
        build_replay(args)
    elif args.cmd == "hardcase_selector":
        build_hardcase_selector(args)
    elif args.cmd == "proxy_mix":
        build_proxy_mix(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
