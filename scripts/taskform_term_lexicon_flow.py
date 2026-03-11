#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import (
    TERM_SPLIT_LABELS,
    apply_term_lexicon,
    attach_stable_split,
    delta_geom,
    evaluate_frame,
    markdown_table,
    normalize_whitespace,
    resolve_path,
    safe_text,
    write_json,
    write_text,
    write_tsv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMULA_RE = re.compile(r"\b(?:Seal of|Sealed by)\s+[^,.;:\n\"]+", flags=re.IGNORECASE)
UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s+(?:mina(?:s)?|shekel(?:s)?|talent(?:s)?|textile(?:s)?|litre(?:s)?)\b(?:\s+of\s+(?:silver|tin|copper|grain))?",
    flags=re.IGNORECASE,
)
HEADER_RE = re.compile(r"\b(?:To|From)\s+([^:\n]+)", flags=re.IGNORECASE)
NAME_TOKEN_RE = re.compile(r"^[A-ZĀĪŪŠṢṬḪ][A-Za-zĀĪŪŠṢṬḪāīūšṣṭḫ'`\-<>]+$")


def _extract_formula_terms(text: str) -> list[str]:
    out: list[str] = []
    for match in FORMULA_RE.findall(safe_text(text)):
        cleaned = normalize_whitespace(match).strip(" ,.;:")
        if len(cleaned) >= 8:
            out.append(cleaned)
    return out


def _extract_unit_terms(text: str) -> list[str]:
    out: list[str] = []
    for match in UNIT_RE.findall(safe_text(text)):
        cleaned = normalize_whitespace(match).strip(" ,.;:")
        if cleaned:
            out.append(cleaned)
    return out


def _extract_name_terms(text: str) -> list[str]:
    names: list[str] = []
    for match in HEADER_RE.findall(safe_text(text)):
        head = normalize_whitespace(match)
        head = re.sub(r"\b(?:to|from)\b", " ", head, flags=re.IGNORECASE)
        for chunk in re.split(r",|\band\b", head):
            tokens = [token.strip() for token in chunk.split() if token.strip()]
            if 0 < len(tokens) <= 4 and all(NAME_TOKEN_RE.match(token) for token in tokens):
                names.append(" ".join(tokens))
    return names


def _collect_terms(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for value in _extract_formula_terms(text):
        rows.append(("formula", value))
    for value in _extract_unit_terms(text):
        rows.append(("measure", value))
    for value in _extract_name_terms(text):
        rows.append(("name_or_place", value))
    return rows


def _term_present(text: str, canonical: str) -> bool:
    return canonical.lower() in safe_text(text).lower()


def _select_term_rows(
    frame: pd.DataFrame,
    *,
    allowed_term_types: set[str],
    min_ref_count: int,
    min_miss_count: int,
    min_miss_rate: float,
    min_pred_hit_count: int,
    max_terms: int,
) -> list[dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        reference = safe_text(row["reference"])
        prediction = safe_text(row["prediction"])
        seen: set[tuple[str, str]] = set()
        for term_type, canonical in _collect_terms(reference):
            key = (term_type, canonical)
            if key in seen:
                continue
            seen.add(key)
            payload = stats.setdefault(
                key,
                {
                    "term_type": term_type,
                    "canonical": canonical,
                    "ref_count": 0,
                    "pred_hit_count": 0,
                    "miss_count": 0,
                },
            )
            payload["ref_count"] += 1
            if _term_present(prediction, canonical):
                payload["pred_hit_count"] += 1
            else:
                payload["miss_count"] += 1

    rows: list[dict[str, Any]] = []
    for (_, _), payload in stats.items():
        term_type = safe_text(payload["term_type"])
        if allowed_term_types and term_type not in allowed_term_types:
            continue
        ref_count = int(payload["ref_count"])
        miss_count = int(payload["miss_count"])
        pred_hit_count = int(payload["pred_hit_count"])
        miss_rate = float(miss_count) / float(max(1, ref_count))
        if ref_count < int(min_ref_count):
            continue
        if miss_count < int(min_miss_count):
            continue
        if miss_rate < float(min_miss_rate):
            continue
        if pred_hit_count < int(min_pred_hit_count):
            continue
        action = "canonicalize"
        if term_type == "name_or_place":
            action = "protect_case"
        rows.append(
            {
                "term_type": term_type,
                "canonical": payload["canonical"],
                "action": action,
                "ref_count": ref_count,
                "pred_hit_count": pred_hit_count,
                "miss_count": miss_count,
                "miss_rate": round(miss_rate, 4),
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item["pred_hit_count"]),
            -int(item["ref_count"]),
            float(item["miss_rate"]),
            item["term_type"],
            item["canonical"],
        )
    )
    if int(max_terms) > 0:
        rows = rows[: int(max_terms)]
    return rows


def _apply_patch_frame(
    frame: pd.DataFrame,
    lexicon_rows: list[dict[str, Any]],
    *,
    apply_builtin_rules: bool,
    normalize_output: bool,
) -> pd.DataFrame:
    out = frame.copy()
    out["patched_prediction"] = out["prediction"].map(
        lambda text: apply_term_lexicon(
            safe_text(text),
            lexicon_rows,
            apply_builtin_rules=apply_builtin_rules,
            normalize_output=normalize_output,
        )
    )
    out["changed"] = out["patched_prediction"] != out["prediction"].fillna("").astype(str)
    return out


def _targeted_mask(frame: pd.DataFrame, lexicon_rows: list[dict[str, Any]]) -> pd.Series:
    canonicals = [safe_text(row["canonical"]) for row in lexicon_rows if safe_text(row["canonical"])]
    if not canonicals:
        return pd.Series(False, index=frame.index)
    lowered = [canonical.lower() for canonical in canonicals]
    refs = frame["reference"].fillna("").astype(str).str.lower()
    return refs.map(lambda text: any(canonical in text for canonical in lowered))


def _subset_eval_bundle(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    routed_ids: set[str],
    targeted_mask: pd.Series,
    split_name: str,
) -> dict[str, Any]:
    local = evaluate_frame(frame, prediction_col=pred_col, tag=f"{split_name}_local", subset_name=split_name)
    official_like = evaluate_frame(
        frame,
        prediction_col=pred_col,
        tag=f"{split_name}_official_like",
        subset_name=split_name,
        note="pending_official_metric_bridge_same_as_local",
    )

    hard_rows = frame.loc[frame["id"].astype(str).isin(routed_ids)].copy()
    targeted_rows = frame.loc[targeted_mask].copy()
    hard = evaluate_frame(
        hard_rows if not hard_rows.empty else frame.head(0),
        prediction_col=pred_col,
        tag=f"{split_name}_hard",
        subset_name=f"{split_name}_routed_hard",
        note="hard subset = ids intersect routed_full",
    )
    targeted = evaluate_frame(
        targeted_rows if not targeted_rows.empty else frame.head(0),
        prediction_col=pred_col,
        tag=f"{split_name}_targeted",
        subset_name=f"{split_name}_term_targeted",
        note="targeted subset = reference contains selected lexicon term",
    )
    return {
        "local": local,
        "official_like": official_like,
        "hard": hard,
        "targeted": targeted,
    }


def _status_from_holdout(
    *,
    baseline_bundle: dict[str, Any],
    patched_bundle: dict[str, Any],
) -> tuple[str, str]:
    local_delta = delta_geom(patched_bundle["local"], baseline_bundle["local"])
    hard_delta = delta_geom(patched_bundle["hard"], baseline_bundle["hard"])
    targeted_delta = delta_geom(patched_bundle["targeted"], baseline_bundle["targeted"])
    if local_delta >= 0.10 and targeted_delta >= 0.20:
        return "accept_to_w", "holdout local+targeted positive"
    if local_delta >= -0.05 and (targeted_delta >= 0.20 or hard_delta >= 0.20):
        return "review_stop", "targeted/hard positive but global gain not yet stable"
    return "reject_stop", "holdout gain insufficient"


def _evaluate_term_types(frame: pd.DataFrame, lexicon_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term_type in sorted({safe_text(row["term_type"]) for row in lexicon_rows if safe_text(row["term_type"])}):
        subset_rows = [row for row in lexicon_rows if safe_text(row["term_type"]) == term_type]
        patched = _apply_patch_frame(frame, subset_rows)
        summary = evaluate_frame(
            patched,
            prediction_col="patched_prediction",
            tag=f"fullval_{term_type}",
            subset_name=f"fullval_{term_type}",
        )
        rows.append(
            {
                "term_type": term_type,
                "terms": len(subset_rows),
                "geom": round(float(summary["eval_geom"]), 4),
                "bleu": round(float(summary["eval_bleu"]), 4),
                "chrfpp": round(float(summary["eval_chrfpp"]), 4),
                "changed_rows": int(patched["changed"].sum()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--official-csv",
        default=(
            "runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/"
            "val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv"
        ),
    )
    ap.add_argument("--routed-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--out-dir", default="reports/taskform_l2_term_phase12")
    ap.add_argument("--allowed-term-types", default="measure")
    ap.add_argument("--min-ref-count", type=int, default=5)
    ap.add_argument("--min-miss-count", type=int, default=2)
    ap.add_argument("--min-miss-rate", type=float, default=0.4)
    ap.add_argument("--min-pred-hit-count", type=int, default=2)
    ap.add_argument("--max-terms", type=int, default=12)
    ap.add_argument("--apply-builtin-rules", action="store_true")
    ap.add_argument("--normalize-output", action="store_true")
    ap.add_argument("--auto-promote", action="store_true")
    args = ap.parse_args()

    official_csv = resolve_path(args.official_csv, REPO_ROOT / "runs" / "missing.csv")
    routed_csv = resolve_path(args.routed_csv, REPO_ROOT / "reports" / "missing.csv")
    out_dir = resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_l2_term_phase12")
    out_dir.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(official_csv).rename(columns={"id": "oare_id"})
    official["id"] = official["oare_id"].fillna("").astype(str)
    official["reference"] = official["reference"].fillna("").astype(str)
    official["prediction"] = official["prediction"].fillna("").astype(str)
    official = attach_stable_split(official, id_col="id")

    routed = pd.read_csv(routed_csv)
    routed_ids = set(routed["oare_id"].astype(str).tolist())

    tune = official.loc[official["stage_split"] == "tune"].copy()
    holdout = official.loc[official["stage_split"] == "holdout"].copy()
    report = official.loc[official["stage_split"] == "report"].copy()

    allowed_term_types = {item.strip() for item in safe_text(args.allowed_term_types).split(",") if item.strip()}
    lexicon_rows = _select_term_rows(
        tune,
        allowed_term_types=allowed_term_types,
        min_ref_count=int(args.min_ref_count),
        min_miss_count=int(args.min_miss_count),
        min_miss_rate=float(args.min_miss_rate),
        min_pred_hit_count=int(args.min_pred_hit_count),
        max_terms=int(args.max_terms),
    )

    lexicon_path = out_dir / "term_lexicon.tsv"
    glossary_path = out_dir / "glossary_min.csv"
    write_tsv(
        lexicon_path,
        lexicon_rows,
        fieldnames=["term_type", "canonical", "action", "ref_count", "pred_hit_count", "miss_count", "miss_rate"],
    )
    glossary_frame = pd.DataFrame(lexicon_rows, columns=["term_type", "canonical", "action"])
    glossary_frame.to_csv(glossary_path, index=False)

    patched_full = _apply_patch_frame(
        official,
        lexicon_rows,
        apply_builtin_rules=bool(args.apply_builtin_rules),
        normalize_output=bool(args.normalize_output),
    )
    targeted_mask_full = _targeted_mask(patched_full, lexicon_rows)

    bundles: dict[str, Any] = {}
    for split_name in TERM_SPLIT_LABELS:
        split_frame = patched_full.loc[patched_full["stage_split"] == split_name].copy()
        split_mask = targeted_mask_full.loc[split_frame.index]
        bundles[split_name] = {
            "baseline": _subset_eval_bundle(
                split_frame,
                pred_col="prediction",
                routed_ids=routed_ids,
                targeted_mask=split_mask,
                split_name=split_name,
            ),
            "patched": _subset_eval_bundle(
                split_frame,
                pred_col="patched_prediction",
                routed_ids=routed_ids,
                targeted_mask=split_mask,
                split_name=f"{split_name}_patched",
            ),
            "rows": int(len(split_frame)),
            "changed_rows": int(split_frame["changed"].sum()),
        }

    full_bundle = {
        "baseline": _subset_eval_bundle(
            patched_full,
            pred_col="prediction",
            routed_ids=routed_ids,
            targeted_mask=targeted_mask_full,
            split_name="fullval",
        ),
        "patched": _subset_eval_bundle(
            patched_full,
            pred_col="patched_prediction",
            routed_ids=routed_ids,
            targeted_mask=targeted_mask_full,
            split_name="fullval_patched",
        ),
        "rows": int(len(patched_full)),
        "changed_rows": int(patched_full["changed"].sum()),
    }

    status, reason = _status_from_holdout(
        baseline_bundle=bundles["holdout"]["baseline"],
        patched_bundle=bundles["holdout"]["patched"],
    )

    per_type_rows: list[dict[str, Any]] = []
    training_feedback_plan: dict[str, Any] = {}
    if bool(args.auto_promote) and status in {"accept_to_w", "review_stop"} and lexicon_rows:
        per_type_rows = _evaluate_term_types(patched_full, lexicon_rows)
        training_feedback_plan = {
            "allow_feedback": status == "accept_to_w",
            "approved_paths": [
                "source_variant_normalization",
                "small_target_consistency_boost",
                "reuse_for_constrained_decoding",
            ],
            "blocked_paths": [
                "rewrite_training_targets",
                "use_holdout_terms_as_training_labels",
                "semantic_rewrite",
            ],
            "terms_by_type": Counter(row["term_type"] for row in lexicon_rows),
        }

    summary = {
        "line": "L2",
        "status": status,
        "reason": reason,
        "lexicon_terms": int(len(lexicon_rows)),
        "lexicon_path": str(lexicon_path),
        "glossary_path": str(glossary_path),
        "selection_config": {
            "allowed_term_types": sorted(allowed_term_types),
            "min_ref_count": int(args.min_ref_count),
            "min_miss_count": int(args.min_miss_count),
            "min_miss_rate": float(args.min_miss_rate),
            "min_pred_hit_count": int(args.min_pred_hit_count),
            "max_terms": int(args.max_terms),
            "apply_builtin_rules": bool(args.apply_builtin_rules),
            "normalize_output": bool(args.normalize_output),
        },
        "splits": bundles,
        "fullval": full_bundle,
        "per_type": per_type_rows,
        "training_feedback_plan": training_feedback_plan,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "local_eval.json", {"splits": bundles, "fullval": full_bundle})
    write_json(
        out_dir / "official_like_eval.json",
        {
            "splits": {name: payload["patched"]["official_like"] for name, payload in bundles.items()},
            "fullval": full_bundle["patched"]["official_like"],
            "note": "pending_official_metric_bridge_same_as_local",
        },
    )
    write_json(
        out_dir / "hard_eval.json",
        {
            "splits": {name: payload["patched"]["hard"] for name, payload in bundles.items()},
            "fullval": full_bundle["patched"]["hard"],
        },
    )
    patched_full[["id", "stage_split", "reference", "prediction", "patched_prediction", "changed"]].to_csv(
        out_dir / "fullval_predictions.csv", index=False
    )
    holdout_rows = patched_full.loc[patched_full["stage_split"] == "holdout", ["id", "reference", "prediction", "patched_prediction", "changed"]]
    holdout_rows.to_csv(out_dir / "holdout_predictions.csv", index=False)

    gate_lines = [
        "# L2 Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- lexicon terms: `{len(lexicon_rows)}`",
        "",
        "## Holdout",
        f"- baseline local geom: `{bundles['holdout']['baseline']['local']['eval_geom']:.4f}`",
        f"- patched local geom: `{bundles['holdout']['patched']['local']['eval_geom']:.4f}`",
        f"- baseline targeted geom: `{bundles['holdout']['baseline']['targeted']['eval_geom']:.4f}`",
        f"- patched targeted geom: `{bundles['holdout']['patched']['targeted']['eval_geom']:.4f}`",
        f"- baseline hard geom: `{bundles['holdout']['baseline']['hard']['eval_geom']:.4f}`",
        f"- patched hard geom: `{bundles['holdout']['patched']['hard']['eval_geom']:.4f}`",
        "",
        "## Fullval",
        f"- baseline local geom: `{full_bundle['baseline']['local']['eval_geom']:.4f}`",
        f"- patched local geom: `{full_bundle['patched']['local']['eval_geom']:.4f}`",
        f"- baseline hard geom: `{full_bundle['baseline']['hard']['eval_geom']:.4f}`",
        f"- patched hard geom: `{full_bundle['patched']['hard']['eval_geom']:.4f}`",
    ]
    if per_type_rows:
        gate_lines.extend(["", "## Per Type", markdown_table(per_type_rows, ["term_type", "terms", "geom", "bleu", "chrfpp", "changed_rows"])])
    write_text(out_dir / "gate_report.md", "\n".join(gate_lines) + "\n")

    print(f"OK: wrote {out_dir/'summary.json'}")
    print(f"OK: wrote {lexicon_path}")
    print(f"OK: wrote {glossary_path}")


if __name__ == "__main__":
    main()
