from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import resolve_path, safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _candidate_run_dir,
    _compare_output_health,
    _compute_hard_subset_metrics,
    _evaluate_candidate,
    _load_incumbent_anchor,
    _load_json,
    _load_val_meta,
    _load_yaml,
    _run,
)


def _evaluate_fullval_candidate(
    *,
    cfg_path: Path,
    fold: int,
    tag: str,
    hard_ids_csv: Path,
) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    run_dir = _candidate_run_dir(cfg, fold)
    checkpoint_dir = run_dir / "best_model"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing best_model: {checkpoint_dir}")

    decode_best_path = run_dir / "diagnostics" / f"decode_grid_best_{tag}.json"
    diagnose_summary_path = run_dir / "diagnostics" / f"val_diagnostic_summary_{tag}.json"

    if not decode_best_path.exists():
        _run(
            [
                str(REPO_ROOT / ".venv-deeppast" / "bin" / "python"),
                str(SCRIPTS_DIR / "eval_decode_grid.py"),
                "--config",
                str(cfg_path),
                "--fold",
                str(fold),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--tag",
                tag,
                "--beams",
                "4",
                "--length-penalties",
                "0.7",
                "--no-repeat-ngram-sizes",
                "0",
                "--min-new-tokens-list",
                "0",
                "--max-new-tokens-list",
                "384",
                "--predict-batch-size",
                "16",
                "--aggregate-by-parent",
                "auto",
                "--aggregate-original-only",
            ]
        )
    decode_payload = _load_json(decode_best_path)

    if not diagnose_summary_path.exists():
        _run(
            [
                str(REPO_ROOT / ".venv-deeppast" / "bin" / "python"),
                str(SCRIPTS_DIR / "diagnose_val_outputs.py"),
                "--config",
                str(cfg_path),
                "--fold",
                str(fold),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--tag",
                tag,
                "--predict-batch-size",
                "16",
                "--num-beams",
                str(int(decode_payload.get("num_beams", 4))),
                "--length-penalty",
                str(float(decode_payload.get("length_penalty", 0.7))),
                "--no-repeat-ngram-size",
                str(int(decode_payload.get("no_repeat_ngram_size", 0))),
                "--min-new-tokens",
                str(int(decode_payload.get("min_new_tokens", 0))),
                "--max-new-tokens",
                str(int(decode_payload.get("max_new_tokens", 384))),
                "--aggregate-by-parent",
                "auto",
                "--aggregate-original-only",
            ]
        )
    diag_payload = _load_json(diagnose_summary_path)
    reconstructed_csv = Path(
        (((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    )
    hard_subset = _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv)

    return {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decode_best_path": str(decode_best_path),
        "diagnose_summary_path": str(diagnose_summary_path),
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "elapsed_seconds": float(decode_payload.get("elapsed_seconds", 0.0) or 0.0),
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "reconstructed_health": (((diag_payload.get("reconstructed", {}) or {}).get("output_health", {}) or {})),
        "hard_subset": hard_subset,
        "reconstructed_csv": str(reconstructed_csv),
        "official_like_note": "same as local until official bridge lands",
    }


def _load_incumbent_fullval(hard_ids_csv: Path) -> dict[str, Any]:
    run_dir = REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0"
    decode_best_path = run_dir / "diagnostics" / "decode_grid_best_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json"
    diagnose_summary_path = run_dir / "diagnostics" / "val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json"
    decode_payload = _load_json(decode_best_path)
    diag_payload = _load_json(diagnose_summary_path)
    reconstructed_csv = Path(
        (((diag_payload.get("reconstructed", {}) or {}).get("artifacts", {}) or {}).get("reconstructed_csv", ""))
    )
    hard_subset = _compute_hard_subset_metrics(reconstructed_csv, hard_ids_csv)
    return {
        "source_decode_best_path": str(decode_best_path),
        "source_diagnose_summary_path": str(diagnose_summary_path),
        "eval_geom": float(decode_payload.get("eval_geom", 0.0)),
        "eval_bleu": float(decode_payload.get("eval_bleu", 0.0)),
        "eval_chrfpp": float(decode_payload.get("eval_chrfpp", 0.0)),
        "elapsed_seconds": float(decode_payload.get("elapsed_seconds", 0.0) or 0.0),
        "output_health": (diag_payload.get("output_health", {}) or {}),
        "reconstructed_health": (((diag_payload.get("reconstructed", {}) or {}).get("output_health", {}) or {})),
        "hard_subset": hard_subset,
        "reconstructed_csv": str(reconstructed_csv),
        "official_like_note": "same as local until official bridge lands",
    }


def _build_cache_hit_stats(cfg_path: Path, fold: int) -> dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    processed_dir = resolve_path((cfg.get("paths", {}) or {}).get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_proc = pd.read_csv(processed_dir / "train_proc.csv")
    folds = pd.read_csv(processed_dir / "folds.csv")[["oare_id", "fold"]]
    merged = train_proc.merge(folds, on="oare_id", how="inner")

    stats: dict[str, Any] = {}
    for label, subset in (
        ("train_visible", merged.loc[merged["fold"] != int(fold)].copy()),
        ("val_visible", merged.loc[merged["fold"] == int(fold)].copy()),
    ):
        hint_source = subset["retrieval_hint_source"].fillna("").astype(str)
        hint_target = subset["retrieval_hint_target"].fillna("").astype(str)
        neighbor_ids = subset["retrieval_neighbor_ids"].fillna("").astype(str)
        score_series = pd.to_numeric(subset["retrieval_scores"], errors="coerce")
        hit_mask = hint_source.str.len().gt(0) & hint_target.str.len().gt(0) & neighbor_ids.str.len().gt(0)
        hit_scores = score_series.loc[hit_mask & score_series.notna()]
        stats[label] = {
            "rows": int(len(subset)),
            "rows_with_hint": int(hit_mask.sum()),
            "hit_ratio_pct": round(100.0 * float(hit_mask.mean() if len(subset) else 0.0), 4),
            "score_mean": round(float(hit_scores.mean()), 4) if len(hit_scores) else None,
            "score_p50": round(float(hit_scores.quantile(0.5)), 4) if len(hit_scores) else None,
            "score_p95": round(float(hit_scores.quantile(0.95)), 4) if len(hit_scores) else None,
        }
    return stats


def _build_neighbor_quality_audit(retrieval_report_dir: Path) -> tuple[dict[str, Any], str]:
    audit_summary = _load_json(retrieval_report_dir / "summary.json")
    bucket_df = pd.read_csv(retrieval_report_dir / "retrieval_bucket_audit.csv")
    bucket_df["source"] = bucket_df["source"].fillna("").astype(str)
    bucket_df["target"] = bucket_df["target"].fillna("").astype(str)
    bucket_df["top1_neighbor_source"] = bucket_df["top1_neighbor_source"].fillna("").astype(str)
    bucket_df["top1_neighbor_target"] = bucket_df["top1_neighbor_target"].fillna("").astype(str)
    bucket_df["top1_target_chrfpp"] = pd.to_numeric(bucket_df["top1_target_chrfpp"], errors="coerce")

    valid = bucket_df.loc[bucket_df["top1_target_chrfpp"].notna()].copy().reset_index(drop=True)
    best_rows = valid.sort_values("top1_target_chrfpp", ascending=False).head(5).copy().reset_index(drop=True)
    worst_rows = valid.sort_values("top1_target_chrfpp", ascending=True).head(5).copy().reset_index(drop=True)

    def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        cols = [
            "oare_id",
            "bucket",
            "top1_score",
            "top1_target_chrfpp",
            "source",
            "target",
            "top1_neighbor_oare_id",
            "top1_neighbor_source",
            "top1_neighbor_target",
        ]
        records: list[dict[str, Any]] = []
        for record in frame[cols].to_dict(orient="records"):
            trimmed = {}
            for key, value in record.items():
                if isinstance(value, str):
                    trimmed[key] = safe_text(value)[:240]
                else:
                    trimmed[key] = value
            records.append(trimmed)
        return records

    summary = {
        "retrieval_audit_summary": audit_summary,
        "best_neighbor_examples": _records(best_rows),
        "worst_neighbor_examples": _records(worst_rows),
    }

    lines = [
        "# A2 Retrieval Neighbor Quality Audit",
        "",
        "- source: `reports/taskform_winner_a2_retrieval_20260310/retrieval_bucket_audit.csv`",
        f"- top1 score mean: `{float((audit_summary.get('scores', {}) or {}).get('top1_score_mean', 0.0)):.4f}`",
        f"- top1 target chrF++ mean: `{float((audit_summary.get('scores', {}) or {}).get('top1_target_chrfpp_mean', 0.0)):.4f}`",
        "",
        "## Best top1 neighbors",
        "",
    ]
    for record in summary["best_neighbor_examples"]:
        lines.extend(
            [
                f"- `oare_id={record['oare_id']}` bucket=`{record['bucket']}` score=`{record['top1_score']:.4f}` chrF++=`{record['top1_target_chrfpp']:.4f}`",
                f"  - source: {record['source']}",
                f"  - target: {record['target']}",
                f"  - nn source: {record['top1_neighbor_source']}",
                f"  - nn target: {record['top1_neighbor_target']}",
            ]
        )
    lines.extend(["", "## Worst top1 neighbors", ""])
    for record in summary["worst_neighbor_examples"]:
        lines.extend(
            [
                f"- `oare_id={record['oare_id']}` bucket=`{record['bucket']}` score=`{record['top1_score']:.4f}` chrF++=`{record['top1_target_chrfpp']:.4f}`",
                f"  - source: {record['source']}",
                f"  - target: {record['target']}",
                f"  - nn source: {record['top1_neighbor_source']}",
                f"  - nn target: {record['top1_neighbor_target']}",
            ]
        )
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wlite-config",
        default="reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml",
    )
    ap.add_argument(
        "--smoke-summary",
        default="reports/taskform_winner_a2_retrieval_eval_20260310/summary.json",
    )
    ap.add_argument(
        "--retrieval-report-dir",
        default="reports/taskform_winner_a2_retrieval_20260310",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a2_retrieval_wlite_eval_20260310")
    args = ap.parse_args()

    wlite_cfg_path = resolve_path(args.wlite_config, REPO_ROOT / "reports")
    smoke_summary_path = resolve_path(args.smoke_summary, REPO_ROOT / "reports")
    retrieval_report_dir = resolve_path(args.retrieval_report_dir, REPO_ROOT / "reports")
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    smoke_summary = _load_json(smoke_summary_path)
    meta_df, token_freq = _load_val_meta(wlite_cfg_path, args.fold)
    wlite_anchor = _evaluate_candidate(
        label="R1_retrieval_top1_wlite_anchor64",
        cfg_path=wlite_cfg_path,
        fold=args.fold,
        tag="taskform_winner_a2_r1_wlite_anchor64_20260310",
        hard_ids_csv=hard_ids_csv,
        meta_df=meta_df,
        token_freq=token_freq,
    )
    wlite_fullval = _evaluate_fullval_candidate(
        cfg_path=wlite_cfg_path,
        fold=args.fold,
        tag="taskform_winner_a2_r1_wlite_fullval_20260310",
        hard_ids_csv=hard_ids_csv,
    )
    incumbent_anchor = smoke_summary.get("incumbent_anchor64") or _load_incumbent_anchor()
    incumbent_fullval = _load_incumbent_fullval(hard_ids_csv)
    cache_hit_stats = _build_cache_hit_stats(wlite_cfg_path, args.fold)
    neighbor_quality_summary, neighbor_quality_md = _build_neighbor_quality_audit(retrieval_report_dir)

    smoke_anchor = (smoke_summary.get("retrieval_top1", {}) or {}).get("anchor64", {}) or {}
    smoke_hard = (smoke_summary.get("retrieval_top1", {}) or {}).get("hard_subset", {}) or {}
    delta_anchor_wlite_vs_smoke = float(wlite_anchor["anchor64"]["eval_geom"]) - float(smoke_anchor.get("eval_geom", 0.0) or 0.0)
    delta_anchor_wlite_vs_i0 = float(wlite_anchor["anchor64"]["eval_geom"]) - float(incumbent_anchor.get("eval_geom", 0.0) or 0.0)
    delta_fullval_wlite_vs_i0 = float(wlite_fullval["eval_geom"]) - float(incumbent_fullval.get("eval_geom", 0.0) or 0.0)
    delta_hard_wlite_vs_i0 = float(wlite_fullval["hard_subset"]["eval_geom"]) - float(
        incumbent_fullval["hard_subset"].get("eval_geom", 0.0) or 0.0
    )
    delta_hard_wlite_vs_smoke = float(wlite_fullval["hard_subset"]["eval_geom"]) - float(smoke_hard.get("eval_geom", 0.0) or 0.0)
    fullval_health_delta_vs_i0 = _compare_output_health(
        {"output_health": wlite_fullval["output_health"]},
        {"output_health": incumbent_fullval["output_health"]},
    )
    anchor_health_delta_vs_smoke = _compare_output_health(wlite_anchor, smoke_summary.get("retrieval_top1", {}) or {})

    if delta_fullval_wlite_vs_i0 >= 0.0 and delta_hard_wlite_vs_i0 >= 0.0 and bool(fullval_health_delta_vs_i0["no_regression"]):
        status = "accept_to_f"
        reason = "W-lite beats incumbent on full-val and hard without output-health regression"
    elif delta_fullval_wlite_vs_i0 >= 0.0 and delta_hard_wlite_vs_i0 >= 0.0:
        status = "review_for_f"
        reason = "W-lite beats incumbent on score, but output-health requires manual review"
    elif delta_fullval_wlite_vs_i0 > -0.5 and delta_hard_wlite_vs_i0 >= 0.0:
        status = "review_stop"
        reason = "W-lite narrows the incumbent gap on full-val and stays non-negative on hard"
    else:
        status = "candidate_pool_only"
        reason = "W-lite does not clear the incumbent replacement gate on full-val"

    summary = {
        "line": "A2_retrieval_wlite",
        "status": status,
        "reason": reason,
        "fold": int(args.fold),
        "smoke_summary_path": str(smoke_summary_path),
        "retrieval_report_dir": str(retrieval_report_dir),
        "wlite_anchor64": wlite_anchor,
        "wlite_fullval": wlite_fullval,
        "smoke_retrieval_anchor64": smoke_summary.get("retrieval_top1"),
        "incumbent_anchor64": incumbent_anchor,
        "incumbent_fullval": incumbent_fullval,
        "cache_hit_stats": cache_hit_stats,
        "neighbor_quality_summary": neighbor_quality_summary,
        "delta_geom_wlite_vs_smoke_anchor64": round(delta_anchor_wlite_vs_smoke, 4),
        "delta_geom_wlite_vs_i0_anchor64": round(delta_anchor_wlite_vs_i0, 4),
        "delta_geom_wlite_vs_i0_fullval": round(delta_fullval_wlite_vs_i0, 4),
        "delta_geom_wlite_vs_i0_hard_subset": round(delta_hard_wlite_vs_i0, 4),
        "delta_geom_wlite_hard_vs_smoke_hard": round(delta_hard_wlite_vs_smoke, 4),
        "anchor_health_delta_wlite_vs_smoke": anchor_health_delta_vs_smoke,
        "fullval_health_delta_wlite_vs_i0": fullval_health_delta_vs_i0,
        "official_like_note": "same as local until official bridge lands",
    }
    write_json(report_dir / "summary.json", summary)
    write_text(report_dir / "neighbor_quality_audit.md", neighbor_quality_md)

    lines = [
        "# A2 Retrieval W-lite Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- W-lite anchor64 geom: `{wlite_anchor['anchor64']['eval_geom']:.4f}`",
        f"- smoke anchor64 geom: `{float(smoke_anchor.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- incumbent anchor64 geom: `{float(incumbent_anchor.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- delta W-lite vs smoke anchor64: `{delta_anchor_wlite_vs_smoke:.4f}`",
        f"- delta W-lite vs incumbent anchor64: `{delta_anchor_wlite_vs_i0:.4f}`",
        f"- W-lite full-val geom: `{wlite_fullval['eval_geom']:.4f}`",
        f"- incumbent full-val geom: `{incumbent_fullval['eval_geom']:.4f}`",
        f"- delta W-lite vs incumbent full-val: `{delta_fullval_wlite_vs_i0:.4f}`",
        f"- W-lite hard geom: `{wlite_fullval['hard_subset']['eval_geom']:.4f}`",
        f"- incumbent hard geom: `{incumbent_fullval['hard_subset']['eval_geom']:.4f}`",
        f"- delta W-lite vs incumbent hard: `{delta_hard_wlite_vs_i0:.4f}`",
        f"- train-visible retrieval hit ratio: `{cache_hit_stats['train_visible']['hit_ratio_pct']:.2f}%`",
        f"- val-visible retrieval hit ratio: `{cache_hit_stats['val_visible']['hit_ratio_pct']:.2f}%`",
        f"- full-val health no_regression vs I0: `{bool(fullval_health_delta_vs_i0['no_regression'])}`",
        "- official-like note: `same as local until official bridge lands`",
    ]
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
