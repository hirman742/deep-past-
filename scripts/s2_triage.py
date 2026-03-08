from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _concat_chunks(values: list[str]) -> str:
    cleaned = [str(x).strip() for x in values if str(x).strip()]
    return "\n".join(cleaned).strip()


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _fmt_path(path: Path | None) -> str:
    if path is None:
        return "NA"
    return str(path)


def _safe_delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "NA"
    delta = float(current) - float(baseline)
    return f"{delta:+.4f}"


def _resolve_run_dir(config_path: Path, fold: int) -> Path:
    cfg = _load_yaml(config_path)
    paths_cfg = cfg.get("paths", {}) or {}
    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    return run_root.parent / f"{run_root.name}_fold{fold}"


def _reconstruct_health(
    *,
    rowwise_csv: Path,
    checkpoint_dir: Path,
    prediction_col: str,
) -> dict[str, float | None]:
    if not rowwise_csv.exists():
        return {
            "empty_prediction_ratio_pct": None,
            "copy_source_ratio_pct": None,
            "pred_shorter_than_half_ref_ratio_pct": None,
            "pred_tok_p95": None,
        }

    frame = pd.read_csv(rowwise_csv)
    if frame.empty or "parent_oare_id" not in frame.columns:
        return {
            "empty_prediction_ratio_pct": None,
            "copy_source_ratio_pct": None,
            "pred_shorter_than_half_ref_ratio_pct": None,
            "pred_tok_p95": None,
        }

    sort_cols = ["parent_oare_id"]
    if "chunk_index" in frame.columns:
        sort_cols.append("chunk_index")
    grouped = (
        frame.sort_values(sort_cols)
        .groupby("parent_oare_id", as_index=False)
        .agg(
            source=("source", lambda s: _concat_chunks(s.tolist())),
            reference=("reference", lambda s: _concat_chunks(s.tolist())),
            prediction=(prediction_col, lambda s: _concat_chunks(s.tolist())),
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    predictions = grouped["prediction"].fillna("").astype(str).tolist()
    references = grouped["reference"].fillna("").astype(str).tolist()
    sources = grouped["source"].fillna("").astype(str).tolist()
    pred_tok_lens = [len(tokenizer.encode(text, add_special_tokens=True)) for text in predictions]

    total = max(1, len(predictions))
    empty_count = sum(1 for text in predictions if not text.strip())
    copy_count = sum(1 for src, pred in zip(sources, predictions) if src.strip() == pred.strip())
    shorter_half = sum(
        1
        for pred, ref in zip(predictions, references)
        if len(ref.strip()) > 0 and (len(pred.strip()) / max(1, len(ref.strip()))) < 0.5
    )

    return {
        "empty_prediction_ratio_pct": 100.0 * float(empty_count) / float(total),
        "copy_source_ratio_pct": 100.0 * float(copy_count) / float(total),
        "pred_shorter_than_half_ref_ratio_pct": 100.0 * float(shorter_half) / float(total),
        "pred_tok_p95": float(pd.Series(pred_tok_lens).quantile(0.95)) if pred_tok_lens else None,
    }


def _baseline_section(run_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "diagnostics" / "val_diagnostic_summary_s1_final.json"
    rowwise_csv = run_dir / "diagnostics" / "val_predictions_diagnostic_s1_final.csv"
    reconstructed_csv = run_dir / "diagnostics" / "val_predictions_reconstructed_s1_final.csv"
    if not summary_path.exists():
        return {
            "status": "missing",
            "summary_path": summary_path,
            "rowwise_csv": rowwise_csv,
            "reconstructed_csv": reconstructed_csv,
            "metrics": None,
            "health": None,
        }

    summary = _load_json(summary_path)
    metrics = (summary.get("reconstructed", {}) or {}).get("metrics", {}) or {}
    health = _reconstruct_health(
        rowwise_csv=rowwise_csv,
        checkpoint_dir=checkpoint_dir,
        prediction_col="prediction",
    )
    return {
        "status": "ok",
        "summary_path": summary_path,
        "rowwise_csv": rowwise_csv,
        "reconstructed_csv": reconstructed_csv,
        "metrics": metrics,
        "health": health,
    }


def _decode_section(run_dir: Path) -> dict[str, Any]:
    best_json = run_dir / "decode_grid_best.json"
    metrics_csv = run_dir / "decode_grid_metrics.csv"
    if not best_json.exists():
        return {
            "status": "pending",
            "best_json": best_json,
            "metrics_csv": metrics_csv,
            "metrics": None,
            "health": None,
        }

    best = _load_json(best_json)
    metrics = {
        "geom": float(best.get("eval_geom", 0.0)),
        "bleu": float(best.get("eval_bleu", 0.0)),
        "chrfpp": float(best.get("eval_chrfpp", 0.0)),
        "rows": int(best.get("eval_rows", 0)),
        "num_beams": int(best.get("num_beams", 0)),
        "length_penalty": float(best.get("length_penalty", 0.0)),
        "no_repeat_ngram_size": int(best.get("no_repeat_ngram_size", 0)),
        "min_new_tokens": int(best.get("min_new_tokens", 0)),
        "max_new_tokens": int(best.get("max_new_tokens", 0)),
        "metric_level": str(best.get("metric_level", "")),
    }
    return {
        "status": "ok",
        "best_json": best_json,
        "metrics_csv": metrics_csv,
        "metrics": metrics,
        "health": None,
    }


def _rerank_section(run_dir: Path, checkpoint_dir: Path, tag: str) -> dict[str, Any]:
    diag_dir = run_dir / "diagnostics"
    summary_path = diag_dir / f"nbest_rerank_summary_{tag}.json"
    rowwise_csv = diag_dir / f"nbest_rerank_rowwise_{tag}.csv"
    candidates_csv = diag_dir / f"nbest_rerank_candidates_{tag}.csv"
    reconstructed_csv = diag_dir / f"nbest_rerank_reconstructed_{tag}.csv"
    if not summary_path.exists():
        return {
            "status": "pending",
            "summary_path": summary_path,
            "rowwise_csv": rowwise_csv,
            "candidates_csv": candidates_csv,
            "reconstructed_csv": reconstructed_csv,
            "metrics": None,
            "health": None,
        }

    summary = _load_json(summary_path)
    metrics = ((summary.get("reconstructed", {}) or {}).get("metrics", {})) or {}
    health = _reconstruct_health(
        rowwise_csv=rowwise_csv,
        checkpoint_dir=checkpoint_dir,
        prediction_col="rerank_prediction",
    )
    return {
        "status": "ok",
        "summary_path": summary_path,
        "rowwise_csv": rowwise_csv,
        "candidates_csv": candidates_csv,
        "reconstructed_csv": reconstructed_csv,
        "metrics": metrics,
        "health": health,
    }


def _write_report(
    *,
    report_path: Path,
    config_path: Path,
    checkpoint_dir: Path,
    baseline: dict[str, Any],
    decode: dict[str, Any],
    rerank_n8: dict[str, Any],
) -> None:
    baseline_metrics = baseline.get("metrics") or {}
    decode_metrics = decode.get("metrics") or {}
    rerank_metrics = rerank_n8.get("metrics") or {}
    baseline_health = baseline.get("health") or {}
    rerank_health = rerank_n8.get("health") or {}

    worth_continue = "pending"
    rerank_geom = rerank_metrics.get("geom")
    decode_geom = decode_metrics.get("geom")
    baseline_geom = baseline_metrics.get("geom")
    if rerank_geom is not None:
        anchor = decode_geom if decode_geom is not None else baseline_geom
        if anchor is not None:
            worth_continue = "yes" if (float(rerank_geom) - float(anchor)) >= 0.1 else "no"
        else:
            worth_continue = "unknown"

    lines = [
        "# Cloud S2 Triage",
        "",
        f"- config_path: {config_path}",
        f"- checkpoint_dir: {checkpoint_dir}",
        f"- worth_continue_rerank: {worth_continue}",
        "",
        "## Baseline",
        f"- status: {baseline['status']}",
        f"- summary_path: {_fmt_path(baseline.get('summary_path'))}",
        f"- rowwise_csv: {_fmt_path(baseline.get('rowwise_csv'))}",
        f"- reconstructed_csv: {_fmt_path(baseline.get('reconstructed_csv'))}",
        f"- reconstructed_geom: {_fmt_float(baseline_metrics.get('geom'))}",
        f"- reconstructed_bleu: {_fmt_float(baseline_metrics.get('bleu'))}",
        f"- reconstructed_chrfpp: {_fmt_float(baseline_metrics.get('chrfpp'))}",
        f"- health_empty_pct: {_fmt_float(baseline_health.get('empty_prediction_ratio_pct'))}",
        f"- health_copy_pct: {_fmt_float(baseline_health.get('copy_source_ratio_pct'))}",
        f"- health_shorter_half_pct: {_fmt_float(baseline_health.get('pred_shorter_than_half_ref_ratio_pct'))}",
        f"- health_pred_tok_p95: {_fmt_float(baseline_health.get('pred_tok_p95'))}",
        "",
        "## Decode Grid",
        f"- status: {decode['status']}",
        f"- best_json: {_fmt_path(decode.get('best_json'))}",
        f"- metrics_csv: {_fmt_path(decode.get('metrics_csv'))}",
        f"- reconstructed_geom: {_fmt_float(decode_metrics.get('geom'))}",
        f"- reconstructed_bleu: {_fmt_float(decode_metrics.get('bleu'))}",
        f"- reconstructed_chrfpp: {_fmt_float(decode_metrics.get('chrfpp'))}",
        f"- delta_geom_vs_baseline: {_safe_delta(decode_metrics.get('geom'), baseline_metrics.get('geom'))}",
        f"- delta_bleu_vs_baseline: {_safe_delta(decode_metrics.get('bleu'), baseline_metrics.get('bleu'))}",
        f"- delta_chrfpp_vs_baseline: {_safe_delta(decode_metrics.get('chrfpp'), baseline_metrics.get('chrfpp'))}",
        f"- metric_level: {decode_metrics.get('metric_level', 'NA')}",
        f"- health_empty_pct: NA",
        f"- health_copy_pct: NA",
        f"- health_shorter_half_pct: NA",
        f"- health_pred_tok_p95: NA",
        "",
        "## Rerank N8",
        f"- status: {rerank_n8['status']}",
        f"- summary_path: {_fmt_path(rerank_n8.get('summary_path'))}",
        f"- rowwise_csv: {_fmt_path(rerank_n8.get('rowwise_csv'))}",
        f"- candidates_csv: {_fmt_path(rerank_n8.get('candidates_csv'))}",
        f"- reconstructed_csv: {_fmt_path(rerank_n8.get('reconstructed_csv'))}",
        f"- reconstructed_geom: {_fmt_float(rerank_metrics.get('geom'))}",
        f"- reconstructed_bleu: {_fmt_float(rerank_metrics.get('bleu'))}",
        f"- reconstructed_chrfpp: {_fmt_float(rerank_metrics.get('chrfpp'))}",
        f"- delta_geom_vs_baseline: {_safe_delta(rerank_metrics.get('geom'), baseline_metrics.get('geom'))}",
        f"- delta_geom_vs_decode_grid: {_safe_delta(rerank_metrics.get('geom'), decode_metrics.get('geom'))}",
        f"- delta_bleu_vs_baseline: {_safe_delta(rerank_metrics.get('bleu'), baseline_metrics.get('bleu'))}",
        f"- delta_chrfpp_vs_baseline: {_safe_delta(rerank_metrics.get('chrfpp'), baseline_metrics.get('chrfpp'))}",
        f"- health_empty_pct: {_fmt_float(rerank_health.get('empty_prediction_ratio_pct'))}",
        f"- health_copy_pct: {_fmt_float(rerank_health.get('copy_source_ratio_pct'))}",
        f"- health_shorter_half_pct: {_fmt_float(rerank_health.get('pred_shorter_than_half_ref_ratio_pct'))}",
        f"- health_pred_tok_p95: {_fmt_float(rerank_health.get('pred_tok_p95'))}",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloud_stage1_len512_lr2e4.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/best_model")
    ap.add_argument("--report-path", default="docs/cloud_s2_triage.md")
    ap.add_argument("--n8-tag", default="stepBEST_beam8_n8_lp1p2_m512")
    args = ap.parse_args()

    config_path = _resolve_path(args.config, REPO_ROOT / "configs" / "cloud_stage1_len512_lr2e4.yaml")
    checkpoint_dir = _resolve_path(args.checkpoint_dir, REPO_ROOT / "runs" / "CLOUD_STAGE1_LEN512_LR2E4_fold0" / "best_model")
    run_dir = _resolve_run_dir(config_path, args.fold)
    report_path = _resolve_path(args.report_path, REPO_ROOT / "docs" / "cloud_s2_triage.md")

    baseline = _baseline_section(run_dir, checkpoint_dir)
    decode = _decode_section(run_dir)
    rerank_n8 = _rerank_section(run_dir, checkpoint_dir, args.n8_tag)
    _write_report(
        report_path=report_path,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        baseline=baseline,
        decode=decode,
        rerank_n8=rerank_n8,
    )

    print(f"report_path={report_path}")
    print(f"baseline_status={baseline['status']}")
    print(f"decode_status={decode['status']}")
    print(f"rerank_n8_status={rerank_n8['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
