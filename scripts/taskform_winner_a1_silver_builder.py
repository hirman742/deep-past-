from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generation_utils import normalize_task_prefix  # noqa: E402


INLINE_WS_RE = re.compile(r"[^\S\n]+", flags=re.UNICODE)
SUBSCRIPT_DIGIT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
TOKEN_STRIP_RE = re.compile(r"^[^\w<>{}\[\]\.-]+|[^\w<>{}\[\]\.-]+$", flags=re.UNICODE)


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_text(text: str, *, task_prefix: str = "") -> str:
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = INLINE_WS_RE.sub(" ", value).strip()
    if task_prefix and value.startswith(task_prefix):
        value = value[len(task_prefix) :].strip()
    return value


def _normalize_token(token: str) -> str:
    value = unicodedata.normalize("NFC", str(token or ""))
    value = value.translate(SUBSCRIPT_DIGIT_MAP).lower().strip()
    value = value.replace("…", "").replace('"', "").replace("'", "")
    value = TOKEN_STRIP_RE.sub("", value)
    return value


def _tokenize_source(text: str) -> list[str]:
    return [tok for tok in (_normalize_token(piece) for piece in str(text or "").split()) if tok]


def _aggregate_parent_pairs(
    *,
    published_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
) -> pd.DataFrame:
    sentence_work = sentence_df.loc[
        sentence_df["translation"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    sentence_work["sentence_obj_in_text"] = pd.to_numeric(
        sentence_work["sentence_obj_in_text"], errors="coerce"
    ).fillna(10**9)
    sentence_work = sentence_work.sort_values(["text_uuid", "sentence_obj_in_text"])

    aggregated = sentence_work.groupby("text_uuid", as_index=False).agg(
        target=("translation", lambda s: "\n".join(str(x).strip() for x in s if str(x).strip()))
    )
    parent = published_df.merge(aggregated, left_on="oare_id", right_on="text_uuid", how="inner")
    parent["source"] = parent["transliteration"].fillna("").astype(str).map(_normalize_text)
    parent["target"] = parent["target"].fillna("").astype(str).map(_normalize_text)
    parent["row_origin"] = "published_agg_parent"
    parent["sentence_uuid"] = ""
    parent["anchor_token"] = ""
    parent["anchor_type"] = ""
    parent["anchor_found"] = True
    parent["source_token_len"] = parent["source"].astype(str).str.split().map(len).astype(int)
    parent["target_word_len"] = parent["target"].astype(str).str.split().map(len).astype(int)
    return parent[
        [
            "oare_id",
            "sentence_uuid",
            "source",
            "target",
            "row_origin",
            "genre_label",
            "label",
            "anchor_token",
            "anchor_type",
            "anchor_found",
            "source_token_len",
            "target_word_len",
        ]
    ].copy()


def _sentence_candidates(
    *,
    published_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
    blocked_oare_ids: set[str],
    max_source_tokens: int,
    max_target_words: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    published_tokens = {
        str(row.oare_id): _tokenize_source(str(row.transliteration))
        for row in published_df[["oare_id", "transliteration"]].itertuples(index=False)
    }
    published_meta = (
        published_df[["oare_id", "genre_label", "label"]]
        .drop_duplicates(subset=["oare_id"])
        .set_index("oare_id")
        .to_dict(orient="index")
    )

    rows: list[dict[str, Any]] = []
    texts_with_any_anchor = 0
    total_texts = 0
    total_sentences = 0
    found_sentences = 0

    for text_uuid, group in sentence_df.groupby("text_uuid", sort=False):
        text_uuid = str(text_uuid)
        if text_uuid in blocked_oare_ids:
            continue
        source_tokens = published_tokens.get(text_uuid) or []
        if not source_tokens:
            continue
        total_texts += 1
        group = group.sort_values("sentence_obj_in_text")
        anchors: list[int | None] = []
        anchor_token_values: list[str] = []
        anchor_types: list[str] = []
        translations: list[str] = []
        sentence_ids: list[str] = []
        search_start = 0

        for row in group.itertuples(index=False):
            total_sentences += 1
            candidates: list[tuple[str, str]] = []
            first_word_spelling = str(getattr(row, "first_word_spelling", "") or "").strip()
            first_word_transcription = str(getattr(row, "first_word_transcription", "") or "").strip()
            if first_word_spelling:
                for token in _tokenize_source(first_word_spelling):
                    candidates.append((token, "first_word_spelling"))
            if first_word_transcription:
                for token in _tokenize_source(first_word_transcription):
                    candidates.append((token, "first_word_transcription"))

            deduped: list[tuple[str, str]] = []
            seen_tokens: set[str] = set()
            for token, token_type in candidates:
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                deduped.append((token, token_type))

            anchor_index: int | None = None
            anchor_token = ""
            anchor_type = ""
            for idx in range(search_start, len(source_tokens)):
                current = source_tokens[idx]
                matched = next(((token, token_type) for token, token_type in deduped if token == current), None)
                if matched is None:
                    continue
                anchor_index = idx
                anchor_token, anchor_type = matched
                search_start = idx
                break

            anchors.append(anchor_index)
            anchor_token_values.append(anchor_token)
            anchor_types.append(anchor_type)
            translations.append(_normalize_text(str(getattr(row, "translation", "") or "")))
            sentence_ids.append(str(getattr(row, "sentence_uuid", "") or ""))

        found_here = sum(anchor is not None for anchor in anchors)
        if found_here > 0:
            texts_with_any_anchor += 1
            found_sentences += found_here

        meta = published_meta.get(text_uuid, {})
        for idx, start in enumerate(anchors):
            if start is None:
                continue
            end: int | None = None
            for later in anchors[idx + 1 :]:
                if later is not None and later > start:
                    end = later
                    break
            span_tokens = source_tokens[start:end] if end is not None else source_tokens[start:]
            if not span_tokens:
                continue
            target = translations[idx]
            if not target:
                continue
            if len(span_tokens) > int(max_source_tokens):
                continue
            if len(target.split()) > int(max_target_words):
                continue
            rows.append(
                {
                    "oare_id": text_uuid,
                    "sentence_uuid": sentence_ids[idx],
                    "source": " ".join(span_tokens).strip(),
                    "target": target,
                    "row_origin": "published_sentence_silver",
                    "genre_label": str(meta.get("genre_label", "") or ""),
                    "label": str(meta.get("label", "") or ""),
                    "anchor_token": anchor_token_values[idx],
                    "anchor_type": anchor_types[idx],
                    "anchor_found": True,
                    "source_token_len": int(len(span_tokens)),
                    "target_word_len": int(len(target.split())),
                }
            )

    if rows:
        sentence_pairs = pd.DataFrame(rows)
    else:
        sentence_pairs = pd.DataFrame(
            columns=[
                "oare_id",
                "sentence_uuid",
                "source",
                "target",
                "row_origin",
                "genre_label",
                "label",
                "anchor_token",
                "anchor_type",
                "anchor_found",
                "source_token_len",
                "target_word_len",
            ]
        )

    stats = {
        "texts_considered": int(total_texts),
        "texts_with_any_anchor": int(texts_with_any_anchor),
        "total_sentence_rows": int(total_sentences),
        "sentence_rows_with_anchor": int(found_sentences),
        "anchor_recall_pct": round(100.0 * float(found_sentences) / float(max(1, total_sentences)), 4),
        "kept_sentence_pairs": int(len(sentence_pairs)),
    }
    return sentence_pairs, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml",
    )
    ap.add_argument(
        "--published-csv",
        default="deep-past-initiative-machine-translation/published_texts.csv",
    )
    ap.add_argument(
        "--sentences-csv",
        default="deep-past-initiative-machine-translation/Sentences_Oare_FirstWord_LinNum.csv",
    )
    ap.add_argument("--raw-train-csv", default="data/raw/train.csv")
    ap.add_argument("--raw-test-csv", default="data/raw/test.csv")
    ap.add_argument("--output-csv", default="data/external/oracc_parallel.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_silver_build_20260310")
    ap.add_argument("--max-source-tokens", type=int, default=160)
    ap.add_argument("--max-target-words", type=int, default=160)
    args = ap.parse_args()

    report_dir = _resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_a1_silver_build_20260310")
    report_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(
        _resolve_path(
            args.base_config,
            REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
        )
    )
    task_prefix = normalize_task_prefix(((base_cfg.get("preprocess", {}) or {}).get("task_prefix", "")))
    processed_dir = _resolve_path(
        ((base_cfg.get("paths", {}) or {}).get("processed_dir", "")),
        REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14",
    )

    published_path = _resolve_path(args.published_csv, REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv")
    sentences_path = _resolve_path(args.sentences_csv, REPO_ROOT / "deep-past-initiative-machine-translation" / "Sentences_Oare_FirstWord_LinNum.csv")
    raw_train_path = _resolve_path(args.raw_train_csv, REPO_ROOT / "data" / "raw" / "train.csv")
    raw_test_path = _resolve_path(args.raw_test_csv, REPO_ROOT / "data" / "raw" / "test.csv")
    output_csv = _resolve_path(args.output_csv, REPO_ROOT / "data" / "external" / "oracc_parallel.csv")

    published_df = pd.read_csv(
        published_path,
        usecols=["oare_id", "label", "genre_label", "transliteration"],
    )
    sentence_df = pd.read_csv(
        sentences_path,
        usecols=[
            "text_uuid",
            "sentence_uuid",
            "sentence_obj_in_text",
            "translation",
            "first_word_transcription",
            "first_word_spelling",
        ],
    )
    raw_train = pd.read_csv(raw_train_path, usecols=["oare_id", "transliteration"])
    raw_test = pd.read_csv(raw_test_path, usecols=["transliteration"])
    processed_train = pd.read_csv(processed_dir / "train_proc.csv", usecols=["source"])

    blocked_oare_ids = set(raw_train["oare_id"].fillna("").astype(str))
    blocked_sources = set(raw_train["transliteration"].fillna("").astype(str).map(_normalize_text))
    blocked_sources.update(raw_test["transliteration"].fillna("").astype(str).map(_normalize_text))
    blocked_sources.update(
        processed_train["source"].fillna("").astype(str).map(lambda x: _normalize_text(x, task_prefix=task_prefix))
    )

    parent_pairs = _aggregate_parent_pairs(published_df=published_df, sentence_df=sentence_df)
    sentence_pairs, sentence_stats = _sentence_candidates(
        published_df=published_df,
        sentence_df=sentence_df,
        blocked_oare_ids=blocked_oare_ids,
        max_source_tokens=int(args.max_source_tokens),
        max_target_words=int(args.max_target_words),
    )

    combined = pd.concat([parent_pairs, sentence_pairs], axis=0, ignore_index=True)
    combined["source"] = combined["source"].fillna("").astype(str).map(_normalize_text)
    combined["target"] = combined["target"].fillna("").astype(str).map(_normalize_text)
    combined["source_norm"] = combined["source"]
    combined = combined.loc[
        combined["source"].astype(str).str.strip().ne("")
        & combined["target"].astype(str).str.strip().ne("")
    ].copy()

    combined["blocked_by_oare_id"] = combined["oare_id"].fillna("").astype(str).isin(blocked_oare_ids)
    combined["blocked_by_source_overlap"] = combined["source_norm"].isin(blocked_sources)
    pre_filter_rows = int(len(combined))
    blocked_oare_id_rows = int(combined["blocked_by_oare_id"].sum())
    blocked_source_rows = int(combined["blocked_by_source_overlap"].sum())
    filtered = combined.loc[
        ~combined["blocked_by_oare_id"] & ~combined["blocked_by_source_overlap"]
    ].copy()

    filtered["priority"] = filtered["row_origin"].map(
        {"published_agg_parent": 0, "published_sentence_silver": 1}
    ).fillna(9)
    exact_pair_rows_before = int(len(filtered))
    filtered = filtered.sort_values(["source_norm", "priority"]).drop_duplicates(
        subset=["source_norm"], keep="first"
    )
    filtered = filtered.reset_index(drop=True)

    filtered["source_origin"] = "oare_published_silver"
    filtered["license_note"] = "local_oare_assets_for_offline_research_only"
    filtered["builder"] = "taskform_winner_a1_silver_builder"
    filtered["target"] = filtered["target"].astype(str)

    output_df = filtered[
        [
            "source",
            "target",
            "source_origin",
            "license_note",
            "row_origin",
            "oare_id",
            "sentence_uuid",
            "genre_label",
            "label",
            "anchor_token",
            "anchor_type",
            "source_token_len",
            "target_word_len",
            "builder",
        ]
    ].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    output_df.head(200).to_csv(report_dir / "samples.csv", index=False)

    summary = {
        "status": "built",
        "base_config_path": str(
            _resolve_path(
                args.base_config,
                REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml",
            )
        ),
        "published_csv": str(published_path),
        "sentences_csv": str(sentences_path),
        "output_csv": str(output_csv),
        "counts": {
            "published_rows": int(len(published_df)),
            "sentence_rows": int(len(sentence_df)),
            "parent_pairs_raw": int(len(parent_pairs)),
            "sentence_pairs_raw": int(len(sentence_pairs)),
            "combined_rows_before_filter": int(pre_filter_rows),
            "blocked_by_train_oare_id_rows": int(blocked_oare_id_rows),
            "blocked_by_competition_source_overlap_rows": int(blocked_source_rows),
            "rows_after_overlap_filter_before_dedupe": int(exact_pair_rows_before),
            "rows_final": int(len(output_df)),
            "unique_source_rows_final": int(output_df["source"].nunique()),
            "row_origin_counts_final": {
                key: int(value) for key, value in output_df["row_origin"].value_counts().to_dict().items()
            },
            "unique_oare_id_final": int(output_df["oare_id"].fillna("").astype(str).nunique()),
        },
        "sentence_alignment": sentence_stats,
        "sweep_capacity": {
            "e10_target_rows": 406,
            "e30_target_rows": 1219,
            "e50_target_rows": 2032,
            "supports_e10": bool(len(output_df) >= 406),
            "supports_e30": bool(len(output_df) >= 1219),
            "supports_e50": bool(len(output_df) >= 2032),
        },
        "artifacts": {
            "samples_csv": str(report_dir / "samples.csv"),
            "summary_json": str(report_dir / "summary.json"),
        },
    }
    _write_json(report_dir / "summary.json", summary)

    lines = [
        "# A1 Silver External Build",
        "",
        "- status: `built`",
        f"- output csv: `{output_csv}`",
        f"- final rows: `{len(output_df)}`",
        f"- final unique source rows: `{output_df['source'].nunique()}`",
        "",
        "## Origins",
        "",
    ]
    for key, value in summary["counts"]["row_origin_counts_final"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Sentence Alignment",
            "",
            f"- texts considered: `{sentence_stats['texts_considered']}`",
            f"- texts with any anchor: `{sentence_stats['texts_with_any_anchor']}`",
            f"- sentence anchor recall pct: `{sentence_stats['anchor_recall_pct']}`",
            f"- kept sentence pairs: `{sentence_stats['kept_sentence_pairs']}`",
            "",
            "## Sweep Capacity",
            "",
            f"- supports E10: `{summary['sweep_capacity']['supports_e10']}`",
            f"- supports E30: `{summary['sweep_capacity']['supports_e30']}`",
            f"- supports E50: `{summary['sweep_capacity']['supports_e50']}`",
        ]
    )
    _write_text(report_dir / "report.md", "\n".join(lines) + "\n")

    print(f"OK: wrote {output_csv}", flush=True)
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)
    print(f"INFO: final_rows={len(output_df)}", flush=True)


if __name__ == "__main__":
    main()
