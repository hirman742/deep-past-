#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from taskform_phase12_common import resolve_path, safe_text, write_json, write_text
from taskform_winner_a2_retrieval_eval import _compare_output_health
from taskform_winner_a2_retrieval_wlite_eval import _evaluate_fullval_candidate, _load_incumbent_fullval
from taskform_winner_candidate_longtrain_flow import _load_frozen_candidate
from taskform_winner_a2_retrieval_eval import _load_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _eligible_probe_candidate(
    *,
    family: str,
    label: str,
    summary: dict[str, Any],
    candidate: dict[str, Any],
    comp: dict[str, Any],
) -> dict[str, Any] | None:
    anchor_delta = float(comp.get("delta_geom_vs_control_anchor64", 0.0) or 0.0)
    hard_delta = float(comp.get("delta_geom_vs_control_hard_subset", 0.0) or 0.0)
    health_ok = bool((comp.get("health_delta_vs_control", {}) or {}).get("no_regression", False))
    recon_health_ok = bool((comp.get("reconstructed_health_delta_vs_control", {}) or {}).get("no_regression", False))
    if not (anchor_delta > 0.0 and hard_delta >= 0.0 and health_ok and recon_health_ok):
        return None
    return {
        "family": family,
        "label": label,
        "anchor_delta_vs_control": round(anchor_delta, 4),
        "hard_delta_vs_control": round(hard_delta, 4),
        "summary_path": str(summary.get("_summary_path", "")),
        "config_path": str(candidate.get("config_path", "")),
        "checkpoint_dir": str(candidate.get("checkpoint_dir", "")),
        "run_dir": str(candidate.get("run_dir", "")),
        "anchor64_geom": float(candidate.get("eval_geom", 0.0) or 0.0),
    }


def _load_probe_candidates(combo_summary_path: Path, replay_summary_path: Path) -> list[dict[str, Any]]:
    combo_summary = _load_json(combo_summary_path)
    combo_summary["_summary_path"] = str(combo_summary_path)
    replay_summary = _load_json(replay_summary_path)
    replay_summary["_summary_path"] = str(replay_summary_path)
    candidates: list[dict[str, Any]] = []

    combo_candidate = (combo_summary.get("candidates", {}) or {}).get("combo", {}) or {}
    combo_comp = (combo_summary.get("comparisons", {}) or {}).get("combo", {}) or {}
    eligible = _eligible_probe_candidate(
        family="retrieval_replay25_combo",
        label="combo",
        summary=combo_summary,
        candidate=combo_candidate,
        comp=combo_comp,
    )
    if eligible:
        candidates.append(eligible)

    replay_best_label = safe_text(replay_summary.get("best_label", ""))
    if replay_best_label and replay_best_label != "ctrl":
        replay_candidate = (replay_summary.get("candidates", {}) or {}).get(replay_best_label, {}) or {}
        replay_comp = (replay_summary.get("comparisons", {}) or {}).get(replay_best_label, {}) or {}
        eligible = _eligible_probe_candidate(
            family="replay_band",
            label=replay_best_label,
            summary=replay_summary,
            candidate=replay_candidate,
            comp=replay_comp,
        )
        if eligible:
            candidates.append(eligible)
    candidates.sort(key=lambda row: (float(row["anchor_delta_vs_control"]), float(row["hard_delta_vs_control"])), reverse=True)
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo-summary", default="reports/taskform_winner_combo_retrieval_replay25_probe_20260311/summary.json")
    ap.add_argument("--replay-summary", default="reports/taskform_winner_replay_band_probe_20260311/summary.json")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--run-suffix", default="20260311")
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_postprobe_fullval_20260311")
    args = ap.parse_args()

    started = time.time()
    combo_summary_path = resolve_path(
        args.combo_summary,
        REPO_ROOT / "reports" / "taskform_winner_combo_retrieval_replay25_probe_20260311" / "summary.json",
    )
    replay_summary_path = resolve_path(
        args.replay_summary,
        REPO_ROOT / "reports" / "taskform_winner_replay_band_probe_20260311" / "summary.json",
    )
    hard_ids_csv = resolve_path(
        args.hard_ids_csv,
        REPO_ROOT / "reports" / "taskform_dan1_b1_b2_b4" / "routed_full_predictions.csv",
    )
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports" / "taskform_winner_postprobe_fullval_20260311")
    report_dir.mkdir(parents=True, exist_ok=True)

    ranked = _load_probe_candidates(combo_summary_path, replay_summary_path)
    selected = ranked[: max(0, int(args.top_k))]
    incumbent_fullval = _load_incumbent_fullval(hard_ids_csv)
    frozen_candidate = _load_frozen_candidate()

    if not selected:
        summary = {
            "status": "skipped_no_probe_winner",
            "reason": "No probe winner satisfies the written post-probe fullval eligibility rule",
            "ranked_probe_candidates": ranked,
            "top_k": int(args.top_k),
        }
        write_json(report_dir / "summary.json", summary)
        write_text(
            report_dir / "gate_report.md",
            "# Winner Post-Probe Fullval Queue\n\n- status: `skipped_no_probe_winner`\n- reason: no eligible probe winners\n",
        )
        return

    results: list[dict[str, Any]] = []
    for idx, candidate in enumerate(selected, start=1):
        cfg_path = Path(str(candidate["config_path"]))
        tag = f"winner_postprobe_{safe_text(candidate['family'])}_{safe_text(candidate['label'])}_fullval_{safe_text(args.run_suffix)}_{idx}"
        fullval_eval = _evaluate_fullval_candidate(
            cfg_path=cfg_path,
            fold=int(args.fold),
            tag=tag,
            hard_ids_csv=hard_ids_csv,
        )
        fullval_geom = float(fullval_eval.get("eval_geom", 0.0) or 0.0)
        fullval_hard = float(((fullval_eval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))
        health_vs_incumbent = _compare_output_health(fullval_eval, incumbent_fullval)
        results.append(
            {
                **candidate,
                "rank": int(idx),
                "fullval_eval": fullval_eval,
                "fullval_geom": round(fullval_geom, 4),
                "fullval_hard_geom": round(fullval_hard, 4),
                "fullval_delta_vs_incumbent": round(fullval_geom - float(incumbent_fullval.get("eval_geom", 0.0) or 0.0), 4),
                "fullval_delta_vs_frozen": round(fullval_geom - float(frozen_candidate.get("fullval_reconstructed_geom", 0.0) or 0.0), 4),
                "hard_delta_vs_incumbent": round(
                    fullval_hard
                    - float(((incumbent_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0)),
                    4,
                ),
                "hard_delta_vs_frozen": round(fullval_hard - float(frozen_candidate.get("hard_geom", 0.0) or 0.0), 4),
                "health_vs_incumbent": health_vs_incumbent,
            }
        )

    best_label = max(results, key=lambda row: float(row["fullval_geom"]))["label"]
    summary = {
        "status": "completed_review_pending",
        "reason": "Post-probe fullval queue completed for top eligible winners only",
        "top_k": int(args.top_k),
        "ranked_probe_candidates": ranked,
        "selected_candidates": results,
        "best_label": best_label,
        "incumbent_fullval": incumbent_fullval,
        "frozen_candidate": frozen_candidate,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# Winner Post-Probe Fullval Queue",
        "",
        f"- status: `{summary['status']}`",
        f"- reason: {summary['reason']}",
        f"- top_k: `{int(args.top_k)}`",
        f"- best_label: `{best_label}`",
        "",
    ]
    for row in results:
        lines.extend(
            [
                f"## {row['family']} / {row['label']}",
                "",
                f"- probe anchor delta vs control: `{float(row['anchor_delta_vs_control']):+.4f}`",
                f"- probe hard delta vs control: `{float(row['hard_delta_vs_control']):+.4f}`",
                f"- fullval geom: `{float(row['fullval_geom']):.4f}`",
                f"- fullval hard geom: `{float(row['fullval_hard_geom']):.4f}`",
                f"- fullval delta vs incumbent: `{float(row['fullval_delta_vs_incumbent']):+.4f}`",
                f"- fullval delta vs frozen: `{float(row['fullval_delta_vs_frozen']):+.4f}`",
                f"- hard delta vs incumbent: `{float(row['hard_delta_vs_incumbent']):+.4f}`",
                f"- hard delta vs frozen: `{float(row['hard_delta_vs_frozen']):+.4f}`",
                f"- health no_regression vs incumbent: `{bool((row['health_vs_incumbent'] or {}).get('no_regression', False))}`",
                "",
            ]
        )
    write_text(report_dir / "gate_report.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
