#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu

from taskform_phase12_common import (
    evaluate_predictions,
    formula_count,
    internal_repeat_score,
    resolve_path,
    safe_text,
    tokenize_words,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "oare_id"
    if "reference" not in frame.columns or "prediction" not in frame.columns:
        raise KeyError(f"Missing reference/prediction in {path}")
    out = frame[[id_col, "source", "reference", "prediction"]].copy()
    out = out.rename(columns={id_col: "id"})
    out["id"] = out["id"].astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    out["reference"] = out["reference"].fillna("").astype(str)
    out["prediction"] = out["prediction"].fillna("").astype(str)
    return out


def _sentence_geom(prediction: str, reference: str) -> float:
    bleu = sacrebleu.metrics.BLEU().sentence_score(prediction, [reference]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).sentence_score(prediction, [reference]).score
    return math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))


def _heuristic_pick(source: str, left: str, right: str) -> tuple[str, str]:
    def score(text: str) -> tuple[float, float, float]:
        words = len(tokenize_words(text))
        src_words = max(1, len(tokenize_words(source)))
        short_penalty = max(0.0, (0.12 * float(src_words)) - float(words))
        return (
            -4.0 * float(internal_repeat_score(text)),
            -2.0 * float(formula_count(text)),
            -float(short_penalty),
        )

    left_score = score(left)
    right_score = score(right)
    if left_score > right_score:
        return left, "baseline"
    if right_score > left_score:
        return right, "probe"
    if len(left) >= len(right):
        return left, "baseline"
    return right, "probe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--probe-csv", required=True)
    ap.add_argument("--out-dir", default="reports/taskform_l1_lite_phase12")
    args = ap.parse_args()

    baseline_csv = resolve_path(args.baseline_csv, REPO_ROOT / "runs" / "missing.csv")
    probe_csv = resolve_path(args.probe_csv, REPO_ROOT / "runs" / "missing.csv")
    out_dir = resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_l1_lite_phase12")
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_frame(baseline_csv)
    probe = _load_frame(probe_csv)
    merged = baseline.merge(probe, on=["id", "source", "reference"], suffixes=("_baseline", "_probe"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between baseline/probe")

    baseline_summary = evaluate_predictions(
        predictions=merged["prediction_baseline"].tolist(),
        references=merged["reference"].tolist(),
        tag="baseline",
        subset_name="diversity_probe",
    )
    probe_summary = evaluate_predictions(
        predictions=merged["prediction_probe"].tolist(),
        references=merged["reference"].tolist(),
        tag="probe",
        subset_name="diversity_probe",
    )

    oracle_predictions: list[str] = []
    heuristic_predictions: list[str] = []
    heuristic_choice_counter: Counter[str] = Counter()
    disagreement = 0
    baseline_win = 0
    probe_win = 0
    oracle_row_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        baseline_pred = safe_text(row["prediction_baseline"])
        probe_pred = safe_text(row["prediction_probe"])
        reference = safe_text(row["reference"])
        source = safe_text(row["source"])
        if baseline_pred != probe_pred:
            disagreement += 1
        baseline_geom = _sentence_geom(baseline_pred, reference)
        probe_geom = _sentence_geom(probe_pred, reference)
        if baseline_geom >= probe_geom:
            oracle_predictions.append(baseline_pred)
            baseline_win += 1
            oracle_choice = "baseline"
        else:
            oracle_predictions.append(probe_pred)
            probe_win += 1
            oracle_choice = "probe"
        heuristic_pred, heuristic_choice = _heuristic_pick(source, baseline_pred, probe_pred)
        heuristic_predictions.append(heuristic_pred)
        heuristic_choice_counter[heuristic_choice] += 1
        oracle_row_rows.append(
            {
                "id": row["id"],
                "baseline_geom": round(baseline_geom, 4),
                "probe_geom": round(probe_geom, 4),
                "oracle_choice": oracle_choice,
                "heuristic_choice": heuristic_choice,
            }
        )

    oracle_summary = evaluate_predictions(
        predictions=oracle_predictions,
        references=merged["reference"].tolist(),
        tag="oracle_upper_bound",
        subset_name="diversity_probe",
        note="oracle per-row upper bound, not deployable",
    )
    heuristic_summary = evaluate_predictions(
        predictions=heuristic_predictions,
        references=merged["reference"].tolist(),
        tag="heuristic_selector",
        subset_name="diversity_probe",
        note="selector prefers lower repeat/formula pressure",
    )

    disagreement_ratio = 100.0 * float(disagreement) / float(max(1, len(merged)))
    oracle_delta = float(oracle_summary["eval_geom"]) - max(float(baseline_summary["eval_geom"]), float(probe_summary["eval_geom"]))
    heuristic_delta = float(heuristic_summary["eval_geom"]) - max(float(baseline_summary["eval_geom"]), float(probe_summary["eval_geom"]))
    probe_drop = float(probe_summary["eval_geom"]) - float(baseline_summary["eval_geom"])

    if probe_drop >= -1.0 and (oracle_delta >= 0.50 or heuristic_delta >= 0.15) and disagreement_ratio >= 15.0:
        status = "accept_to_w"
        reason = "probe has enough complementarity to justify wider OOF work"
    elif probe_drop >= -1.0 and (oracle_delta >= 0.30 or disagreement_ratio >= 12.0):
        status = "review_stop"
        reason = "some complementarity exists but practical gain still uncertain"
    else:
        status = "reject_stop"
        reason = "complementarity too weak for immediate OOF expansion"

    summary = {
        "line": "L1-lite",
        "status": status,
        "reason": reason,
        "rows": int(len(merged)),
        "baseline": baseline_summary,
        "probe": probe_summary,
        "heuristic_selector": heuristic_summary,
        "oracle_upper_bound": oracle_summary,
        "disagreement_ratio_pct": disagreement_ratio,
        "baseline_oracle_wins": int(baseline_win),
        "probe_oracle_wins": int(probe_win),
        "heuristic_choice_counts": dict(heuristic_choice_counter),
        "oracle_delta_geom_vs_best_single": oracle_delta,
        "heuristic_delta_geom_vs_best_single": heuristic_delta,
        "probe_delta_geom_vs_baseline": probe_drop,
    }
    write_json(out_dir / "summary.json", summary)
    pd.DataFrame(oracle_row_rows).to_csv(out_dir / "rowwise.csv", index=False)

    lines = [
        "# L1-lite Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- rows: `{len(merged)}`",
        f"- disagreement ratio: `{disagreement_ratio:.2f}%`",
        f"- baseline geom: `{baseline_summary['eval_geom']:.4f}`",
        f"- probe geom: `{probe_summary['eval_geom']:.4f}`",
        f"- heuristic selector geom: `{heuristic_summary['eval_geom']:.4f}`",
        f"- oracle upper bound geom: `{oracle_summary['eval_geom']:.4f}`",
        f"- oracle delta vs best single: `{oracle_delta:.4f}`",
        f"- heuristic delta vs best single: `{heuristic_delta:.4f}`",
    ]
    write_text(out_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
