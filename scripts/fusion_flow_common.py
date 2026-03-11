from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generation_utils import build_bad_words_ids, build_generate_kwargs
from metrics_utils import build_metric_signatures, compute_translation_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_KEYS = ("has_gap", "has_bracket", "has_subscript", "has_x")


def resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), max(1, size))]


def marker_flags(text: str) -> dict[str, bool]:
    value = safe_text(text)
    return {
        "has_gap": bool(re.search(r"<gap>|\{[^}]*\}|\{", value)),
        "has_bracket": bool(re.search(r"\[|\]", value)),
        "has_subscript": bool(re.search(r"[₀₁₂₃₄₅₆₇₈₉ₓ]", value)),
        "has_x": bool(re.search(r"(?<!\w)x(?!\w)", value.lower())),
    }


def head_tail_words(text: str, n_words: int) -> tuple[str, str]:
    words = [token for token in safe_text(text).split() if token]
    if not words:
        return "", ""
    head = " ".join(words[:n_words])
    tail = " ".join(words[-n_words:])
    return head, tail


def strip_prompt_markup(text: str) -> str:
    value = safe_text(text)
    patterns = [
        r"<parent_chunk_total>\d+</parent_chunk_total>",
        r"</?chunk_\d+>",
        r"/chunk_\d+>",
        r"\bchunk_idx\s*=\s*\d+\b",
        r"\bchunk_\d+\b",
        r"\bFinal translation:\b",
    ]
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return normalize_whitespace(value)


def collapse_repeated_word_spans(text: str, *, max_span: int = 8) -> str:
    words = [token for token in normalize_whitespace(text).split() if token]
    if not words:
        return ""
    out: list[str] = []
    i = 0
    n_words = len(words)
    while i < n_words:
        kept = False
        max_try = min(max_span, (n_words - i) // 2)
        for span in range(max_try, 0, -1):
            left = words[i : i + span]
            right = words[i + span : i + (2 * span)]
            if left == right:
                out.extend(left)
                i += span
                while i + span <= n_words and words[i - span : i] == words[i : i + span]:
                    i += span
                kept = True
                break
        if kept:
            continue
        if not out or out[-1] != words[i]:
            out.append(words[i])
        i += 1
    return " ".join(out).strip()


def dedupe_consecutive_texts(texts: list[str]) -> list[str]:
    out: list[str] = []
    for raw in texts:
        text = normalize_whitespace(raw)
        if not text:
            continue
        if out and out[-1] == text:
            continue
        out.append(text)
    return out


def shorten_words(text: str, *, max_words: int, tail_words: int = 0) -> str:
    words = [token for token in normalize_whitespace(text).split() if token]
    if max_words <= 0 or len(words) <= max_words:
        return " ".join(words).strip()
    if tail_words <= 0 or tail_words >= max_words:
        return " ".join(words[:max_words]).strip()
    head_words = max(1, max_words - tail_words)
    return " ".join(words[:head_words] + ["..."] + words[-tail_words:]).strip()


def sanitize_draft(
    text: str,
    *,
    max_words: int,
    tail_words: int = 0,
) -> str:
    value = strip_prompt_markup(text)
    value = collapse_repeated_word_spans(value)
    value = normalize_whitespace(value)
    if not value:
        return ""
    return shorten_words(value, max_words=max_words, tail_words=tail_words)


def combine_chunk_texts(
    chunk_rows: list[dict[str, Any]],
    *,
    mode: str,
    draft_field: str,
    dedup_consecutive: bool = False,
) -> str:
    ordered = sorted(chunk_rows, key=lambda item: int(item["chunk_index"]))
    fallback_field = "chunk_target" if mode == "oracle" else "draft_prediction"
    texts: list[str] = []
    for chunk in ordered:
        text = safe_text(chunk.get(draft_field, "") or chunk.get(fallback_field, ""))
        text = normalize_whitespace(text)
        if text:
            texts.append(text)
    if dedup_consecutive:
        texts = dedupe_consecutive_texts(texts)
    return "\n".join(texts).strip()


def load_base_merged(processed_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(processed_dir / "train_proc.csv")
    folds_df = pd.read_csv(processed_dir / "folds.csv")
    return train_df.merge(folds_df, on="oare_id", how="inner", suffixes=("", "_fold"))


def _word_len(text: str) -> int:
    return int(len([token for token in safe_text(text).split() if token]))


def _route_reason(
    *,
    chunk_total: int,
    parent_ref_tok: int,
    marker_count: int,
    route_ref_tok_threshold: int,
    route_marker_ref_tok_threshold: int,
) -> str:
    reasons: list[str] = []
    if chunk_total >= 4:
        reasons.append("chunk4plus")
    if parent_ref_tok >= route_ref_tok_threshold:
        reasons.append("ref_tok129plus")
    if marker_count >= 2 and parent_ref_tok >= route_marker_ref_tok_threshold:
        reasons.append("tag_rich_long")
    return "|".join(reasons) if reasons else "pass_a_only"


def build_parent_payloads(
    *,
    merged: pd.DataFrame,
    fold: int,
    base_tokenizer_name: str,
    route_ref_tok_threshold: int,
    route_marker_ref_tok_threshold: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for parent_id, group in merged.groupby("parent_oare_id", sort=False):
        ordered = group.sort_values("chunk_index").reset_index(drop=True)
        parent_target = (
            safe_text(ordered["translation"].iloc[0])
            or safe_text(ordered["target_raw"].iloc[0])
            or "\n".join(ordered["target"].fillna("").astype(str).tolist()).strip()
        )
        parent_transliteration = (
            safe_text(ordered["transliteration"].iloc[0])
            or safe_text(ordered["source_raw"].iloc[0])
            or "\n".join(ordered["source_raw"].fillna("").astype(str).tolist()).strip()
        )
        parent_flags = marker_flags(parent_transliteration)
        marker_count = sum(int(parent_flags[key]) for key in STRUCTURE_KEYS)
        chunk_total = int(ordered["chunk_total"].max())
        parent_ref_tok = _word_len(parent_target)
        is_routed_hard = bool(
            chunk_total >= 4
            or parent_ref_tok >= route_ref_tok_threshold
            or (marker_count >= 2 and parent_ref_tok >= route_marker_ref_tok_threshold)
        )
        route_score = (
            (120.0 if chunk_total >= 4 else 0.0)
            + (120.0 if parent_ref_tok >= route_ref_tok_threshold else 0.0)
            + (45.0 * marker_count)
            + min(float(parent_ref_tok), 512.0) / 8.0
            + float(chunk_total * 2)
        )
        chunks: list[dict[str, Any]] = []
        for row in ordered.to_dict(orient="records"):
            chunk_source_raw = safe_text(row.get("source_raw"))
            chunk_flags = marker_flags(chunk_source_raw)
            chunk_head, chunk_tail = head_tail_words(chunk_source_raw, 12)
            chunks.append(
                {
                    "oare_id": safe_text(row.get("oare_id")),
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "chunk_total": int(row.get("chunk_total", 1)),
                    "source": safe_text(row.get("source")),
                    "source_raw": chunk_source_raw,
                    "target": safe_text(row.get("target")),
                    "target_raw": safe_text(row.get("target_raw")),
                    "transliteration": safe_text(row.get("transliteration")),
                    "translation": safe_text(row.get("translation")),
                    "head": chunk_head,
                    "tail": chunk_tail,
                    **chunk_flags,
                }
            )
        payloads.append(
            {
                "parent_oare_id": safe_text(parent_id),
                "oare_id": safe_text(parent_id),
                "fold": int(ordered["fold"].iloc[0]),
                "split": "val" if int(ordered["fold"].iloc[0]) == int(fold) else "train",
                "group_key": safe_text(ordered.get("group_key", pd.Series([""])).iloc[0]),
                "group_kind": safe_text(ordered.get("group_kind", pd.Series(["groupkfold"])).iloc[0]),
                "group_source": safe_text(ordered.get("group_source", pd.Series(["source_bucket"])).iloc[0]),
                "chunk_total": chunk_total,
                "parent_ref_tok": parent_ref_tok,
                "marker_count": marker_count,
                "is_routed_hard": is_routed_hard,
                "route_reason": _route_reason(
                    chunk_total=chunk_total,
                    parent_ref_tok=parent_ref_tok,
                    marker_count=marker_count,
                    route_ref_tok_threshold=route_ref_tok_threshold,
                    route_marker_ref_tok_threshold=route_marker_ref_tok_threshold,
                ),
                "route_score": round(route_score, 4),
                "parent_transliteration": parent_transliteration,
                "parent_translation": parent_target,
                "chunks": chunks,
                **parent_flags,
            }
        )

    for split in ("train", "val"):
        routed = [
            payload
            for payload in payloads
            if payload["split"] == split and bool(payload["is_routed_hard"])
        ]
        routed.sort(
            key=lambda item: (
                -float(item["route_score"]),
                -int(item["parent_ref_tok"]),
                -int(item["chunk_total"]),
                item["parent_oare_id"],
            )
        )
        for rank, payload in enumerate(routed, start=1):
            payload["route_rank"] = rank

    for payload in payloads:
        payload.setdefault("route_rank", 0)
    return payloads


def payloads_to_metadata_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        rows.append(
            {
                "parent_oare_id": payload["parent_oare_id"],
                "oare_id": payload["oare_id"],
                "fold": int(payload["fold"]),
                "split": payload["split"],
                "group_key": payload["group_key"],
                "group_kind": payload["group_kind"],
                "group_source": payload["group_source"],
                "chunk_total": int(payload["chunk_total"]),
                "parent_ref_tok": int(payload["parent_ref_tok"]),
                "marker_count": int(payload["marker_count"]),
                "is_routed_hard": bool(payload["is_routed_hard"]),
                "route_rank": int(payload["route_rank"]),
                "route_score": float(payload["route_score"]),
                "route_reason": payload["route_reason"],
                "has_gap": bool(payload["has_gap"]),
                "has_bracket": bool(payload["has_bracket"]),
                "has_subscript": bool(payload["has_subscript"]),
                "has_x": bool(payload["has_x"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["split", "is_routed_hard", "route_rank", "parent_oare_id"],
        ascending=[True, False, True, True],
    )


def load_val_chunk_drafts(diagnostic_csv: Path) -> dict[str, str]:
    frame = pd.read_csv(diagnostic_csv)
    return {
        safe_text(row["oare_id"]): safe_text(row["prediction"])
        for row in frame.to_dict(orient="records")
        if safe_text(row.get("oare_id"))
    }


def _build_model_for_generate(model_name: str, checkpoint_dir: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    if checkpoint_dir is not None:
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_predictions(
    *,
    model_name: str,
    checkpoint_dir: Path | None,
    sources: list[str],
    max_source_length: int,
    predict_batch_size: int,
    num_beams: int,
    length_penalty: float,
    max_new_tokens: int,
    no_repeat_ngram_size: int = 0,
    bad_tokens_regex: str = r"<extra_id_\d+>",
    suppress_extra_ids: bool = True,
) -> list[str]:
    tokenizer, model, device = _build_model_for_generate(model_name, checkpoint_dir)
    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(suppress_extra_ids),
        bad_tokens_regex=str(bad_tokens_regex),
    )
    generate_kwargs = build_generate_kwargs(
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=0,
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        bad_words_ids=bad_words_ids,
    )
    outputs: list[str] = []
    with torch.no_grad():
        for batch in chunked(sources, max(1, predict_batch_size)):
            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            generated = model.generate(**tokenized, **generate_kwargs)
            outputs.extend([safe_text(text) for text in tokenizer.batch_decode(generated, skip_special_tokens=True)])
    return outputs


def build_dan1_prompt(chunk_rows: list[dict[str, Any]], *, mode: str) -> str:
    lines = [
        "Fuse the chunk drafts into one final English translation for the full parent.",
        "Keep all information, remove cross-chunk repetition, and preserve uncertainty markers like <gap>.",
        f"<parent_chunk_total>{len(chunk_rows)}</parent_chunk_total>",
    ]
    for idx, chunk in enumerate(sorted(chunk_rows, key=lambda item: int(item["chunk_index"])), start=1):
        draft_text = safe_text(chunk["chunk_target"] if mode == "oracle" else chunk["draft_prediction"]) or "<empty>"
        lines.extend(
            [
                f"<chunk_{idx}>",
                f"chunk_idx={idx}",
                draft_text,
                f"</chunk_{idx}>",
            ]
        )
    return "\n".join(lines).strip()


def build_dan1_flat_prompt(
    chunk_rows: list[dict[str, Any]],
    *,
    mode: str,
    draft_field: str,
) -> str:
    lines = [
        "Merge chunk drafts into one English translation for the whole parent.",
        "Remove repeated boilerplate across chunks. Keep <gap> when uncertain.",
        f"Parent chunks: {len(chunk_rows)}",
    ]
    ordered = sorted(chunk_rows, key=lambda item: int(item["chunk_index"]))
    grouped: list[dict[str, Any]] = []
    group_index: dict[str, int] = {}
    for idx, chunk in enumerate(ordered, start=1):
        fallback_field = "chunk_target" if mode == "oracle" else "draft_prediction"
        text = safe_text(chunk.get(draft_field, "") or chunk.get(fallback_field, ""))
        if not text:
            continue
        if text in group_index:
            grouped[group_index[text]]["indices"].append(idx)
            continue
        group_index[text] = len(grouped)
        grouped.append({"indices": [idx], "text": text})
    for group in grouped:
        indices = group["indices"]
        label = ",".join(str(i) for i in indices[:6])
        if len(indices) > 6:
            label = f"{label},..."
        lines.append(f"c{label}: {group['text']}")
    lines.append("Final translation:")
    return "\n".join(lines).strip()


def build_dan1_edit_prompt(
    chunk_rows: list[dict[str, Any]],
    *,
    mode: str,
    draft_field: str,
) -> str:
    combined = combine_chunk_texts(
        chunk_rows,
        mode=mode,
        draft_field=draft_field,
        dedup_consecutive=True,
    ) or "<empty>"
    lines = [
        "Edit the combined draft into one coherent English translation for the whole parent.",
        "Preserve details and <gap> markers. Keep repeated wording unless it is clearly accidental noise.",
        "Do not mention chunks, labels, or notes.",
        "Combined draft:",
        combined,
        "Final translation:",
    ]
    return "\n".join(lines).strip()


def build_dan2_prompt(chunk_rows: list[dict[str, Any]]) -> str:
    lines = [
        "Fuse the chunk drafts into one final English translation for the full parent.",
        "Use the source hints to resolve structure and broken text. Keep <gap> markers when the draft is uncertain.",
        f"<parent_chunk_total>{len(chunk_rows)}</parent_chunk_total>",
    ]
    ordered = sorted(chunk_rows, key=lambda item: int(item["chunk_index"]))
    for idx, chunk in enumerate(ordered, start=1):
        lines.extend(
            [
                f"<chunk_{idx}>",
                f"chunk_idx={idx}",
                f"draft: {safe_text(chunk['draft_prediction']) or '<empty>'}",
                (
                    "flags: "
                    f"has_gap={str(bool(chunk['has_gap'])).lower()} "
                    f"has_bracket={str(bool(chunk['has_bracket'])).lower()} "
                    f"has_subscript={str(bool(chunk['has_subscript'])).lower()} "
                    f"has_x={str(bool(chunk['has_x'])).lower()}"
                ),
                f"translit_head: {safe_text(chunk['head'])}",
                f"translit_tail: {safe_text(chunk['tail'])}",
                f"</chunk_{idx}>",
            ]
        )
    return "\n".join(lines).strip()


def build_processed_rows(
    *,
    parent_groups: dict[str, list[dict[str, Any]]],
    metadata_frame: pd.DataFrame,
    chunk_mode: str,
    prompt_builder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    metadata_index = metadata_frame.set_index("parent_oare_id")
    for parent_id in sorted(parent_groups.keys()):
        meta = metadata_index.loc[parent_id]
        chunk_rows = parent_groups[parent_id]
        chunk_rows = sorted(chunk_rows, key=lambda item: int(item["chunk_index"]))
        target = safe_text(chunk_rows[0]["parent_translation"])
        source = prompt_builder(chunk_rows)
        rows.append(
            {
                "oare_id": parent_id,
                "transliteration": safe_text(chunk_rows[0]["parent_transliteration"]),
                "translation": target,
                "source_raw": source,
                "target_raw": target,
                "source": source,
                "target": target,
                "parent_oare_id": parent_id,
                "chunk_index": 0,
                "chunk_total": 1,
                "is_chunk": False,
                "chunk_mode": chunk_mode,
                "is_short_aligned": False,
                "short_align_mode": "",
                "source_oare_id": "",
                "align_type": "",
                "align_cost": "",
                "route_rank": int(meta["route_rank"]),
                "route_score": float(meta["route_score"]),
                "route_reason": safe_text(meta["route_reason"]),
                "orig_chunk_total": int(meta["chunk_total"]),
                "parent_ref_tok": int(meta["parent_ref_tok"]),
                "marker_count": int(meta["marker_count"]),
                "has_gap": bool(meta["has_gap"]),
                "has_bracket": bool(meta["has_bracket"]),
                "has_subscript": bool(meta["has_subscript"]),
                "has_x": bool(meta["has_x"]),
            }
        )
        fold_rows.append(
            {
                "oare_id": parent_id,
                "fold": int(meta["fold"]),
                "group_key": safe_text(meta["group_key"]),
                "group_kind": safe_text(meta["group_kind"]),
                "group_source": safe_text(meta["group_source"]),
                "parent_oare_id": parent_id,
                "chunk_index": 0,
                "chunk_total": 1,
                "chunk_mode": chunk_mode,
                "short_align_mode": "",
                "align_type": "",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return float(num) / float(den)


def _has_internal_repeat_ngram(text: str, *, ngram_size: int = 3, min_count: int = 4) -> bool:
    words = [token for token in normalize_whitespace(text).split() if token]
    if len(words) < max(ngram_size, 1):
        return False
    grams = Counter(
        tuple(words[idx : idx + ngram_size])
        for idx in range(0, len(words) - ngram_size + 1)
    )
    return any(count >= int(min_count) for count in grams.values())


def compute_eval_summary(
    *,
    predictions: list[str],
    references: list[str],
    sources: list[str],
    tokenizer_name: str,
    tag: str,
    checkpoint_dir: str,
    subset_name: str,
    eval_rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pred_texts = [safe_text(text) for text in predictions]
    ref_texts = [safe_text(text) for text in references]
    src_texts = [safe_text(text) for text in sources]
    metrics = compute_translation_metrics(predictions=pred_texts, references=ref_texts)
    pred_tok_lens = [len(tokenizer.encode(text, add_special_tokens=True)) for text in pred_texts]
    ref_tok_lens = [len(tokenizer.encode(text, add_special_tokens=True)) for text in ref_texts]
    pred_char_lens = [len(text) for text in pred_texts]
    ref_char_lens = [len(text) for text in ref_texts]
    empty_count = sum(1 for text in pred_texts if not text.strip())
    copy_count = sum(1 for src, pred in zip(src_texts, pred_texts) if src.strip() == pred.strip() and pred.strip())
    short_count = sum(
        1
        for pred_len, ref_len in zip(pred_char_lens, ref_char_lens)
        if ref_len > 0 and _safe_div(float(pred_len), float(ref_len)) < 0.5
    )
    counts = Counter(pred_texts)
    repeat_rows = sum(count for text, count in counts.items() if text and count > 1)
    internal_repeat_rows = sum(1 for text in pred_texts if _has_internal_repeat_ngram(text))
    summary = {
        "tag": tag,
        "subset_name": subset_name,
        "checkpoint_dir": checkpoint_dir,
        "eval_rows": int(eval_rows),
        "eval_bleu": float(metrics["bleu"]),
        "eval_chrfpp": float(metrics["chrfpp"]),
        "eval_geom": float(metrics["geom"]),
        "eval_bleu_01": float(metrics["bleu_01"]),
        "eval_chrfpp_01": float(metrics["chrfpp_01"]),
        "eval_geom_01": float(metrics["geom_01"]),
        "metric_signatures": build_metric_signatures(),
        "health": {
            "empty_prediction_ratio_pct": 100.0 * _safe_div(float(empty_count), float(len(pred_texts))),
            "copy_source_ratio_pct": 100.0 * _safe_div(float(copy_count), float(len(pred_texts))),
            "pred_shorter_than_half_ref_ratio_pct": 100.0 * _safe_div(float(short_count), float(len(pred_texts))),
            "unique_prediction_ratio_pct": 100.0 * _safe_div(float(len(set(pred_texts))), float(len(pred_texts))),
            "repeat_prediction_ratio_pct": 100.0 * _safe_div(float(repeat_rows), float(len(pred_texts))),
            "internal_repeat_trigram_ratio_pct": 100.0 * _safe_div(float(internal_repeat_rows), float(len(pred_texts))),
            "top_repeated_predictions": [
                {"text": text, "count": int(count)}
                for text, count in counts.most_common(10)
                if text
            ],
        },
        "length_stats": {
            "pred_tok_mean": float(pd.Series(pred_tok_lens).mean()) if pred_tok_lens else 0.0,
            "ref_tok_mean": float(pd.Series(ref_tok_lens).mean()) if ref_tok_lens else 0.0,
            "pred_tok_p95": float(pd.Series(pred_tok_lens).quantile(0.95)) if pred_tok_lens else 0.0,
            "ref_tok_p95": float(pd.Series(ref_tok_lens).quantile(0.95)) if ref_tok_lens else 0.0,
            "pred_char_p95": float(pd.Series(pred_char_lens).quantile(0.95)) if pred_char_lens else 0.0,
            "ref_char_p95": float(pd.Series(ref_char_lens).quantile(0.95)) if ref_char_lens else 0.0,
        },
    }
    if extra:
        summary.update(extra)
    return summary


def title_case(line_name: str) -> str:
    return line_name.upper().replace("_", "-")
