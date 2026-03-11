from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from metrics_utils import build_metric_signatures, compute_translation_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
TERM_SPLIT_LABELS = ("tune", "holdout", "report")
UNIT_WORDS = (
    "mina",
    "minas",
    "shekel",
    "shekels",
    "talent",
    "talents",
    "textile",
    "textiles",
    "litre",
    "litres",
    "silver",
    "tin",
    "copper",
)


def resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    return " ".join(safe_text(text).replace("\r", "\n").split())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def stable_split(key: str) -> str:
    digest = hashlib.md5(safe_text(key).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 4:
        return "tune"
    if bucket < 7:
        return "holdout"
    return "report"


def attach_stable_split(frame: pd.DataFrame, *, id_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["stage_split"] = out[id_col].map(stable_split)
    return out


def tokenize_words(text: str) -> list[str]:
    return [token for token in normalize_whitespace(text).split() if token]


def word_count(text: str) -> int:
    return len(tokenize_words(text))


def internal_repeat_score(text: str, *, ngram_size: int = 3, min_count: int = 4) -> int:
    words = tokenize_words(text)
    if len(words) < max(1, ngram_size):
        return 0
    grams = Counter(tuple(words[idx : idx + ngram_size]) for idx in range(0, len(words) - ngram_size + 1))
    return int(sum(max(0, count - (min_count - 1)) for count in grams.values() if count >= min_count))


def formula_count(text: str) -> int:
    value = safe_text(text)
    patterns = ("Seal of", "Sealed by", "send me the silver", "will pay the silver", "not paid the silver")
    return int(sum(value.count(item) for item in patterns))


def build_health(predictions: list[str], references: list[str]) -> dict[str, float]:
    pred_values = [safe_text(text) for text in predictions]
    ref_values = [safe_text(text) for text in references]
    pred_counts = Counter(pred_values)
    pred_word_lens = [word_count(text) for text in pred_values]
    ref_word_lens = [word_count(text) for text in ref_values]
    short = 0
    repeated = 0
    trigram_repeat = 0
    empty = 0
    for pred, ref_len in zip(pred_values, ref_word_lens):
        pred_len = word_count(pred)
        if pred_len == 0:
            empty += 1
        if pred_len < max(1, ref_len / 2.0):
            short += 1
        if internal_repeat_score(pred) > 0:
            trigram_repeat += 1
        if pred_counts[pred] > 1:
            repeated += 1
    total = max(1, len(pred_values))
    return {
        "empty_prediction_ratio_pct": 100.0 * float(empty) / float(total),
        "pred_shorter_than_half_ref_ratio_pct": 100.0 * float(short) / float(total),
        "repeat_prediction_ratio_pct": 100.0 * float(repeated) / float(total),
        "internal_repeat_trigram_ratio_pct": 100.0 * float(trigram_repeat) / float(total),
        "pred_word_mean": float(sum(pred_word_lens)) / float(total) if pred_word_lens else 0.0,
        "ref_word_mean": float(sum(ref_word_lens)) / float(total) if ref_word_lens else 0.0,
    }


def evaluate_predictions(
    *,
    predictions: list[str],
    references: list[str],
    tag: str,
    subset_name: str,
    note: str = "",
) -> dict[str, Any]:
    if not predictions or not references:
        metrics = {
            "bleu": 0.0,
            "chrfpp": 0.0,
            "geom": 0.0,
            "bleu_01": 0.0,
            "chrfpp_01": 0.0,
            "geom_01": 0.0,
        }
    else:
        metrics = compute_translation_metrics(predictions=predictions, references=references)
    payload = {
        "tag": tag,
        "subset_name": subset_name,
        "rows": int(len(predictions)),
        "eval_bleu": float(metrics["bleu"]),
        "eval_chrfpp": float(metrics["chrfpp"]),
        "eval_geom": float(metrics["geom"]),
        "eval_bleu_01": float(metrics["bleu_01"]),
        "eval_chrfpp_01": float(metrics["chrfpp_01"]),
        "eval_geom_01": float(metrics["geom_01"]),
        "metric_signatures": build_metric_signatures(),
        "health": build_health(predictions, references),
    }
    if note:
        payload["note"] = note
    return payload


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    reference_col: str = "reference",
    tag: str,
    subset_name: str,
    note: str = "",
) -> dict[str, Any]:
    return evaluate_predictions(
        predictions=frame[prediction_col].fillna("").astype(str).tolist(),
        references=frame[reference_col].fillna("").astype(str).tolist(),
        tag=tag,
        subset_name=subset_name,
        note=note,
    )


def delta_geom(summary: dict[str, Any], baseline_summary: dict[str, Any]) -> float:
    return float(summary.get("eval_geom", 0.0)) - float(baseline_summary.get("eval_geom", 0.0))


def repair_gap_markers(text: str) -> str:
    value = safe_text(text)
    value = re.sub(r"(?<!<)\bgap>", "<gap>", value, flags=re.IGNORECASE)
    value = re.sub(r"<gap>\s*<gap>", "<gap>", value)
    return normalize_whitespace(value)


def clamp_consecutive_repeated_spans(
    text: str,
    *,
    max_span: int = 12,
    max_occurrences: int = 2,
) -> str:
    words = tokenize_words(text)
    if not words:
        return ""
    out: list[str] = []
    idx = 0
    total = len(words)
    while idx < total:
        matched = False
        max_try = min(int(max_span), (total - idx) // 2)
        for span in range(max_try, 0, -1):
            pattern = words[idx : idx + span]
            reps = 1
            while idx + ((reps + 1) * span) <= total and words[idx + (reps * span) : idx + ((reps + 1) * span)] == pattern:
                reps += 1
            if reps > int(max_occurrences):
                for _ in range(int(max_occurrences)):
                    out.extend(pattern)
                idx += reps * span
                matched = True
                break
        if matched:
            continue
        out.append(words[idx])
        idx += 1
    return normalize_whitespace(" ".join(out))


def collapse_formula_loops(text: str, *, max_repeats: int = 2) -> str:
    value = normalize_whitespace(text)
    if not value:
        return ""
    words = value.split()
    stems = (
        "Seal of",
        "Sealed by",
        "send me the silver",
        "will pay the silver",
        "not paid the silver",
    )
    spans = [2, 3, 4]
    for stem in stems:
        stem_words = stem.split()
        stem_len = len(stem_words)
        idx = 0
        new_words: list[str] = []
        while idx < len(words):
            if words[idx : idx + stem_len] != stem_words:
                new_words.append(words[idx])
                idx += 1
                continue
            collapsed = False
            for span in spans:
                if idx + (span * stem_len) > len(words):
                    continue
                unit = words[idx : idx + span]
                reps = 1
                while idx + ((reps + 1) * len(unit)) <= len(words) and words[idx + (reps * len(unit)) : idx + ((reps + 1) * len(unit))] == unit:
                    reps += 1
                if reps > max_repeats:
                    for _ in range(max_repeats):
                        new_words.extend(unit)
                    idx += reps * len(unit)
                    collapsed = True
                    break
            if not collapsed:
                new_words.extend(words[idx : idx + stem_len])
                idx += stem_len
        words = new_words
    return normalize_whitespace(" ".join(words))


def load_term_lexicon(path: Path | None) -> list[dict[str, str]]:
    if not path or not path.exists():
        return []
    frame = pd.read_csv(path, sep="\t")
    rows: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "term_type": safe_text(row.get("term_type")),
                "canonical": safe_text(row.get("canonical")),
                "action": safe_text(row.get("action")),
            }
        )
    return rows


def apply_term_lexicon(
    text: str,
    lexicon_rows: list[dict[str, str]],
    *,
    apply_builtin_rules: bool = True,
    normalize_output: bool = True,
) -> str:
    value = safe_text(text)
    if not value:
        return ""
    patched = value
    if apply_builtin_rules:
        patched = re.sub(r"(?i)\bseal o\b", "Seal of", patched)
        patched = re.sub(r"(?i)\bseal of\b", "Seal of", patched)
        patched = re.sub(r"(?i)\bsealed by\b", "Sealed by", patched)
        patched = re.sub(
            r"\b(\d+)\s+(\d{4})(?=\s+(?:mina|minas|shekel|shekels|talent|talents|textile|textiles|litre|litres)\b)",
            r"\1.\2",
            patched,
        )
    for row in lexicon_rows:
        canonical = safe_text(row.get("canonical"))
        if not canonical:
            continue
        escaped = re.escape(canonical)
        patched = re.sub(rf"(?i)\b{escaped}\b", canonical, patched)
        if canonical.startswith("Seal of "):
            tail = re.escape(canonical[len("Seal of ") :])
            patched = re.sub(rf"(?i)\bseal o(?:f)?\s+{tail}\b", canonical, patched)
    return normalize_whitespace(patched) if normalize_output else patched


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return "\n".join([header, sep, *body])
