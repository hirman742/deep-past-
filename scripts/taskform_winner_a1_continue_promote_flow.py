from __future__ import annotations

import argparse
from pathlib import Path

from taskform_phase12_common import resolve_path, write_json, write_text
from taskform_winner_a1_continue_probe_flow import _load_json
from taskform_winner_a2_retrieval_wlite_eval import _evaluate_fullval_candidate, _load_incumbent_fullval, _run


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlite-summary", default="reports/taskform_winner_a1_continue_wlite_20260310/summary.json")
    ap.add_argument("--freeze-summary", default="reports/taskform_winner_a2_freeze_20260310/summary.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--hard-ids-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_continue_promote_20260310")
    args = ap.parse_args()

    wlite_summary = _load_json(resolve_path(args.wlite_summary, REPO_ROOT / "reports"))
    freeze_summary = _load_json(resolve_path(args.freeze_summary, REPO_ROOT / "reports"))
    report_dir = resolve_path(args.report_dir, REPO_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    hard_ids_csv = resolve_path(args.hard_ids_csv, REPO_ROOT / "reports")

    if str(wlite_summary.get("status", "")) != "proceed_fullval":
        summary = {
            "status": "skipped_no_wlite_gate",
            "reason": "A1 W-lite did not pass proceed_fullval gate",
            "wlite_summary_path": str(resolve_path(args.wlite_summary, REPO_ROOT / "reports")),
        }
        write_json(report_dir / "summary.json", summary)
        write_text(
            report_dir / "gate_report.md",
            "# A1 Continue Promote\n\n- status: `skipped_no_wlite_gate`\n- reason: W-lite did not pass `proceed_fullval`\n",
        )
        print(f"OK: wrote {report_dir / 'summary.json'}")
        return

    best_label = str(wlite_summary.get("best_ratio_label", "candidate"))
    control_cfg = Path(str(((wlite_summary.get("artifacts", {}) or {}).get("control_config_path", ""))))
    candidate_cfg = Path(str(((wlite_summary.get("artifacts", {}) or {}).get("candidate_config_path", ""))))
    if not control_cfg.exists() or not candidate_cfg.exists():
        raise FileNotFoundError("Missing W-lite config paths")

    control_fullval = _evaluate_fullval_candidate(
        cfg_path=control_cfg,
        fold=int(args.fold),
        tag="taskform_winner_a1r_internal_only_wlite_fullval_20260310",
        hard_ids_csv=hard_ids_csv,
    )
    candidate_fullval = _evaluate_fullval_candidate(
        cfg_path=candidate_cfg,
        fold=int(args.fold),
        tag="taskform_winner_a1r_candidate_wlite_fullval_20260310",
        hard_ids_csv=hard_ids_csv,
    )
    incumbent_fullval = _load_incumbent_fullval(hard_ids_csv)

    candidate_run_dir = Path(str(candidate_fullval["run_dir"]))
    candidate_anchor_csv = Path(
        str(
            candidate_run_dir
            / "diagnostics"
            / f"val_predictions_diagnostic_taskform_winner_a1r_{best_label}_wlite_anchor64_20260310.csv"
        )
    )
    candidate_fullval_csv = Path(
        str(candidate_run_dir / "diagnostics" / "val_predictions_diagnostic_taskform_winner_a1r_candidate_wlite_fullval_20260310.csv")
    )
    if not candidate_anchor_csv.exists() or not candidate_fullval_csv.exists():
        raise FileNotFoundError("Missing candidate diagnostic csvs for surgical review")

    surgical_report_dir = report_dir / "health_surgical"
    _run(
        [
            str(REPO_ROOT / ".venv-deeppast" / "bin" / "python"),
            str(SCRIPTS_DIR / "taskform_winner_a2_health_surgical_probe.py"),
            "--anchor64-wlite-csv",
            str(candidate_anchor_csv),
            "--fullval-wlite-csv",
            str(candidate_fullval_csv),
            "--anchor64-incumbent-csv",
            str(REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "diagnostics" / "val_predictions_diagnostic_taskform_a2_a1_incumbent_anchor64_20260310.csv"),
            "--fullval-incumbent-csv",
            str(REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "diagnostics" / "val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv"),
            "--incumbent-fullval-summary",
            str(REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "diagnostics" / "val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json"),
            "--hard-ids-csv",
            str(hard_ids_csv),
            "--report-dir",
            str(surgical_report_dir),
        ]
    )
    surgical_summary = _load_json(surgical_report_dir / "summary.json")
    selected_variant = str(surgical_summary.get("recommended_variant", "raw"))
    selected_payload = ((surgical_summary.get("variant_payloads", {}) or {}).get(selected_variant, {}) or {})
    selected_fullval = (selected_payload.get("fullval", {}) or {})
    selected_anchor = (selected_payload.get("anchor64", {}) or {})
    selected_health_vs_i0 = (selected_payload.get("fullval_health_vs_i0", {}) or {})

    frozen_score = (freeze_summary.get("scoreboard", {}) or {}).get("fallback_180", {}) or {}
    frozen_fullval = float(frozen_score.get("fullval_reconstructed_geom", 0.0) or 0.0)
    frozen_hard = float(frozen_score.get("hard_geom", 0.0) or 0.0)

    selected_fullval_geom = float(
        (((selected_fullval.get("reconstructed", {}) or {}).get("metrics", {}) or {}).get("geom", 0.0) or 0.0)
    )
    selected_hard_geom = float((((selected_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0)) or 0.0))
    selected_anchor_geom = float(
        (((selected_anchor.get("reconstructed", {}) or {}).get("metrics", {}) or {}).get("geom", 0.0) or 0.0)
    )

    control_fullval_geom = float(control_fullval.get("eval_geom", 0.0) or 0.0)
    control_hard_geom = float(((control_fullval.get("hard_subset", {}) or {}).get("eval_geom", 0.0) or 0.0))
    candidate_raw_fullval_geom = float(candidate_fullval.get("eval_geom", 0.0) or 0.0)

    status = "review_stop"
    if (
        selected_fullval_geom >= (frozen_fullval + 0.15)
        and selected_hard_geom >= (frozen_hard - 0.10)
        and bool(selected_health_vs_i0.get("no_regression"))
    ):
        status = "manual_promote_recommended"
    elif selected_fullval_geom > control_fullval_geom and bool(selected_health_vs_i0.get("no_regression")):
        status = "candidate_pool_only"

    summary = {
        "status": status,
        "reason": "A1 continue candidate compared against control, incumbent, and frozen fallback_180",
        "control_fullval": control_fullval,
        "candidate_fullval_raw": candidate_fullval,
        "incumbent_fullval": incumbent_fullval,
        "frozen_candidate": {
            "fullval_reconstructed_geom": frozen_fullval,
            "hard_geom": frozen_hard,
            "anchor64_reconstructed_geom": float(frozen_score.get("anchor64_reconstructed_geom", 0.0) or 0.0),
        },
        "health_surgical": {
            "summary_json": str(surgical_report_dir / "summary.json"),
            "recommended_variant": selected_variant,
            "selected_health_vs_i0": selected_health_vs_i0,
        },
        "selected_candidate": {
            "variant": selected_variant,
            "anchor64_reconstructed_geom": round(selected_anchor_geom, 4),
            "fullval_reconstructed_geom": round(selected_fullval_geom, 4),
            "hard_geom": round(selected_hard_geom, 4),
        },
        "deltas": {
            "selected_vs_control": {
                "fullval_reconstructed_geom": round(selected_fullval_geom - control_fullval_geom, 4),
                "hard_geom": round(selected_hard_geom - control_hard_geom, 4),
            },
            "selected_vs_frozen_candidate": {
                "fullval_reconstructed_geom": round(selected_fullval_geom - frozen_fullval, 4),
                "hard_geom": round(selected_hard_geom - frozen_hard, 4),
            },
            "raw_vs_control": {
                "fullval_reconstructed_geom": round(candidate_raw_fullval_geom - control_fullval_geom, 4),
            },
        },
        "artifacts": {
            "candidate_anchor_csv": str(candidate_anchor_csv),
            "candidate_fullval_csv": str(candidate_fullval_csv),
            "surgical_report_dir": str(surgical_report_dir),
        },
    }
    write_json(report_dir / "summary.json", summary)

    lines = [
        "# A1 Continue Promote",
        "",
        f"- status: `{status}`",
        f"- selected variant: `{selected_variant}`",
        f"- control full-val geom: `{control_fullval_geom:.4f}`",
        f"- raw candidate full-val geom: `{candidate_raw_fullval_geom:.4f}`",
        f"- selected candidate full-val geom: `{selected_fullval_geom:.4f}`",
        f"- selected candidate hard geom: `{selected_hard_geom:.4f}`",
        f"- frozen fallback_180 full-val geom: `{frozen_fullval:.4f}`",
        f"- delta selected vs control full-val: `{(selected_fullval_geom - control_fullval_geom):.4f}`",
        f"- delta selected vs frozen full-val: `{(selected_fullval_geom - frozen_fullval):.4f}`",
        f"- health no_regression vs incumbent: `{bool(selected_health_vs_i0.get('no_regression'))}`",
        "",
    ]
    write_text(report_dir / "gate_report.md", "\n".join(lines))
    print(f"OK: wrote {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
