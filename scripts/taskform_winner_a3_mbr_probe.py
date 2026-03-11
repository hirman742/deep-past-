from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu

from taskform_phase12_common import (
    build_health,
    formula_count,
    internal_repeat_score,
    resolve_path,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "id" if "id" in frame.columns else "oare_id"
    needed = {id_col, "source", "reference", "prediction"}
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    out = frame[[id_col, "source", "reference", "prediction"]].copy()
    out = out.rename(
        columns={
            id_col: "id",
            "source": f"source__{label}",
            "reference": f"reference__{label}",
            "prediction": f"prediction__{label}",
        }
    )
    out["id"] = out["id"].fillna("").astype(str)
    for col in out.columns:
        if col != "id":
            out[col] = out[col].fillna("").astype(str)
    return out


def _merge_frames(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for _, frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="id", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping ids across candidate files")
    labels = [label for label, _ in frames]
    merged["source"] = merged[[f"source__{label}" for label in labels]].bfill(axis=1).iloc[:, 0].fillna("")
    merged["reference"] = merged[[f"reference__{label}" for label in labels]].bfill(axis=1).iloc[:, 0].fillna("")
    return merged


def _sentence_geom(prediction: str, reference: str) -> float:
    bleu = sacrebleu.metrics.BLEU().sentence_score(prediction, [reference]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).sentence_score(prediction, [reference]).score
    return math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))


def _corpus_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    bleu = sacrebleu.metrics.BLEU().corpus_score(predictions, [references]).score
    chrfpp = sacrebleu.metrics.CHRF(word_order=2).corpus_score(predictions, [references]).score
    geom = math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))
    return {"bleu": float(bleu), "chrfpp": float(chrfpp), "geom": float(geom)}


def _diagnostic_health(predictions: list[str], references: list[str], sources: list[str]) -> dict[str, Any]:
    pred_series = pd.Series(predictions, dtype=str)
    total = max(1, len(predictions))
    top_repeated = []
    counts = pred_series.value_counts()
    for text, count in counts.head(10).items():
        if text.strip() and int(count) > 1:
            top_repeated.append({"text": str(text), "count": int(count)})
    health = {
        "num_rows": int(len(predictions)),
        "empty_prediction_ratio_pct": round(100.0 * float(sum(1 for pred in predictions if not pred.strip())) / float(total), 4),
        "copy_source_ratio_pct": round(
            100.0 * float(sum(1 for src, pred in zip(sources, predictions) if src.strip() == pred.strip())) / float(total),
            4,
        ),
        "pred_shorter_than_half_ref_ratio_pct": round(
            100.0 * float(sum(1 for pred, ref in zip(predictions, references) if len(pred) < max(1.0, len(ref) / 2.0))) / float(total),
            4,
        ),
        "unique_prediction_ratio_pct": round(100.0 * float(pred_series.nunique()) / float(total), 4),
        "has_bad_token_regex_ratio_pct": 0.0,
        "exact_extra_id_0_ratio_pct": round(100.0 * float(sum(1 for pred in predictions if pred.strip() == "<extra_id_0>")) / float(total), 4),
        "repeat_prediction_ratio_pct": round(
            100.0 * float(sum(1 for pred in predictions if counts.get(pred, 0) > 1)) / float(total),
            4,
        ),
        "top_repeated_predictions": top_repeated,
    }
    aux = build_health(predictions, references)
    health["aux_repeat_prediction_ratio_pct"] = float(aux["repeat_prediction_ratio_pct"])
    health["aux_internal_repeat_trigram_ratio_pct"] = float(aux["internal_repeat_trigram_ratio_pct"])
    health["aux_pred_word_mean"] = float(aux["pred_word_mean"])
    health["aux_ref_word_mean"] = float(aux["ref_word_mean"])
    return health


def _row_candidate_stats(text: str) -> tuple[int, int, int]:
    return (internal_repeat_score(text), formula_count(text), len(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--incumbent-csv",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_reconstructed_taskform_a2_a1_incumbent_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--retrieval-smoke-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_20260310_fold0/diagnostics/val_predictions_reconstructed_taskform_winner_a2_r1_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--retrieval-wlite-csv",
        default="runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_predictions_reconstructed_taskform_winner_a2_r1_wlite_anchor64_20260310.csv",
    )
    ap.add_argument(
        "--retrieval-wlite-repaired-csv",
        default="reports/taskform_winner_a2_health_review_20260310/repeat_trim_long_only_anchor64_reconstructed.csv",
    )
    ap.add_argument("--report-dir", default="reports/taskform_winner_a3_mbr_probe_20260310")
    ap.add_argument("--health-tie-margin", type=float, default=0.35)
    args = ap.parse_args()

    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_specs = [
        ("incumbent", resolve_path(args.incumbent_csv, REPO_ROOT / "runs")),
        ("retrieval_smoke", resolve_path(args.retrieval_smoke_csv, REPO_ROOT / "runs")),
        ("retrieval_wlite", resolve_path(args.retrieval_wlite_csv, REPO_ROOT / "runs")),
    ]
    repaired_path = resolve_path(args.retrieval_wlite_repaired_csv, REPO_ROOT / "reports")
    if repaired_path.exists():
        candidate_specs.append(("retrieval_wlite_repaired", repaired_path))

    frames = [(label, _load_predictions(path, label)) for label, path in candidate_specs]
    labels = [label for label, _ in frames]
    merged = _merge_frames(frames).sort_values("id").reset_index(drop=True)

    references = merged["reference"].fillna("").astype(str).tolist()
    sources = merged["source"].fillna("").astype(str).tolist()
    row_records: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    selected_predictions: list[str] = []
    selected_from: list[str] = []
    health_tie_rows = 0

    for _, row in merged.iterrows():
        preds = {label: str(row[f"prediction__{label}"]) for label in labels}
        utilities: dict[str, float] = {}
        for label in labels:
            others = [other for other in labels if other != label]
            if not others:
                utilities[label] = 0.0
                continue
            scores = [_sentence_geom(preds[label], preds[other]) for other in others]
            utilities[label] = float(sum(scores) / float(max(1, len(scores))))
        best_utility = max(utilities.values())
        near_best = [label for label in labels if (best_utility - utilities[label]) <= float(args.health_tie_margin)]
        if len(near_best) > 1:
            health_tie_rows += 1
        chosen = sorted(
            near_best,
            key=lambda label: (
                _row_candidate_stats(preds[label])[0],
                _row_candidate_stats(preds[label])[1],
                -utilities[label],
                _row_candidate_stats(preds[label])[2],
                label,
            ),
        )[0]
        selected_predictions.append(preds[chosen])
        selected_from.append(chosen)

        row_out: dict[str, Any] = {
            "id": str(row["id"]),
            "reference": str(row["reference"]),
            "selected_from": chosen,
        }
        for label in labels:
            row_out[f"utility__{label}"] = round(float(utilities[label]), 4)
            row_out[f"prediction__{label}"] = preds[label]
        row_out["selected_prediction"] = preds[chosen]
        row_records.append(row_out)

    rowwise_df = pd.DataFrame(row_records)
    rowwise_path = report_dir / "rowwise.csv"
    rowwise_df.to_csv(rowwise_path, index=False)

    for left, right in itertools.combinations(labels, 2):
        exact_overlap = merged[f"prediction__{left}"].eq(merged[f"prediction__{right}"])
        pairwise_rows.append(
            {
                "left": left,
                "right": right,
                "exact_overlap_ratio_pct": round(100.0 * float(exact_overlap.mean()), 4),
            }
        )
    pd.DataFrame(pairwise_rows).to_csv(report_dir / "pairwise_overlap.csv", index=False)

    candidate_metrics: dict[str, Any] = {}
    best_single_label = ""
    best_single_geom = -1.0
    for label in labels:
        preds = merged[f"prediction__{label}"].fillna("").astype(str).tolist()
        metrics = _corpus_metrics(preds, references)
        health = _diagnostic_health(preds, references, sources)
        candidate_metrics[label] = {
            "metrics": {
                "bleu": round(float(metrics["bleu"]), 4),
                "chrfpp": round(float(metrics["chrfpp"]), 4),
                "geom": round(float(metrics["geom"]), 4),
            },
            "output_health": health,
        }
        if float(metrics["geom"]) > best_single_geom:
            best_single_geom = float(metrics["geom"])
            best_single_label = label

    mbr_metrics = _corpus_metrics(selected_predictions, references)
    mbr_health = _diagnostic_health(selected_predictions, references, sources)
    mbr_summary = {
        "metrics": {
            "bleu": round(float(mbr_metrics["bleu"]), 4),
            "chrfpp": round(float(mbr_metrics["chrfpp"]), 4),
            "geom": round(float(mbr_metrics["geom"]), 4),
        },
        "output_health": mbr_health,
    }

    selection_counts = pd.Series(selected_from, dtype=str).value_counts().to_dict()
    gain_vs_best_single = round(float(mbr_metrics["geom"]) - float(best_single_geom), 4)
    health_not_worse = (
        float(mbr_health["empty_prediction_ratio_pct"]) <= float(candidate_metrics[best_single_label]["output_health"]["empty_prediction_ratio_pct"])
        and float(mbr_health["copy_source_ratio_pct"]) <= float(candidate_metrics[best_single_label]["output_health"]["copy_source_ratio_pct"])
        and float(mbr_health["pred_shorter_than_half_ref_ratio_pct"]) <= float(candidate_metrics[best_single_label]["output_health"]["pred_shorter_than_half_ref_ratio_pct"])
        and float(mbr_health["unique_prediction_ratio_pct"]) >= float(candidate_metrics[best_single_label]["output_health"]["unique_prediction_ratio_pct"])
    )
    status = "review_stop"
    if gain_vs_best_single >= 0.2 and bool(health_not_worse):
        status = "promote_to_pairwise"

    manifest = {
        "status": status,
        "candidate_pool": [
            {"label": label, "path": str(path)}
            for label, path in candidate_specs
        ],
        "best_single_label": best_single_label,
        "best_single_geom": round(best_single_geom, 4),
        "mbr_geom": round(float(mbr_metrics["geom"]), 4),
        "delta_geom_vs_best_single": gain_vs_best_single,
        "health_not_worse_vs_best_single": bool(health_not_worse),
        "health_tie_rows": int(health_tie_rows),
        "selection_counts": {key: int(value) for key, value in selection_counts.items()},
        "artifacts": {
            "rowwise_csv": str(rowwise_path),
            "pairwise_overlap_csv": str(report_dir / "pairwise_overlap.csv"),
        },
    }
    write_json(report_dir / "candidate_pool_manifest.json", manifest)

    summary = {
        **manifest,
        "candidate_metrics": candidate_metrics,
        "mbr": mbr_summary,
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A3 MBR Probe",
        "",
        f"- status: `{status}`",
        f"- best single: `{best_single_label}`",
        f"- best single geom: `{best_single_geom:.4f}`",
        f"- MBR geom: `{float(mbr_metrics['geom']):.4f}`",
        f"- delta vs best single: `{gain_vs_best_single}`",
        f"- health not worse vs best single: `{bool(health_not_worse)}`",
        f"- health tie rows: `{health_tie_rows}`",
    ]
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
