#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from taskform_a2_a1_flow import REPO_ROOT, _load_json, _resolve_path
from taskform_phase12_common import write_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="reports/taskform_winner_replay_probe_20260311/summary.json")
    ap.add_argument("--doc-path", default="docs/taskform_winner_replay_probe_2026-03-11.md")
    args = ap.parse_args()

    summary_path = _resolve_path(args.summary, REPO_ROOT / "reports" / "taskform_winner_replay_probe_20260311" / "summary.json")
    doc_path = _resolve_path(args.doc_path, REPO_ROOT / "docs" / "taskform_winner_replay_probe_2026-03-11.md")
    summary = _load_json(summary_path)

    incumbent = summary.get("incumbent_anchor64", {}) or {}
    frozen = summary.get("frozen_candidate", {}) or {}
    candidates = summary.get("candidates", {}) or {}
    comparisons = summary.get("comparisons", {}) or {}
    control = candidates.get("ctrl", {}) or {}

    lines = [
        "# Winner Replay / Curriculum Probe（2026-03-11）",
        "",
        f"- status: `{summary.get('status', '')}`",
        f"- reason: {summary.get('reason', '')}",
        f"- summary: `{summary_path}`",
        f"- gate report: `{Path(str(summary.get('report_dir', ''))) / 'gate_report.md'}`",
        f"- best label: `{summary.get('best_label', '')}`",
        "",
        "## Baselines",
        "",
        f"- incumbent anchor64 geom: `{float(incumbent.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- frozen fallback anchor/fullval/hard: `{float(frozen.get('anchor64_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen.get('fullval_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen.get('hard_geom', 0.0) or 0.0):.4f}`",
        "",
        "## Control",
        "",
        f"- control anchor64 geom: `{float(control.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- control hard geom: `{float(((control.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0)):.4f}`",
    ]

    for label in ("replay25", "replay40"):
        candidate = candidates.get(label, {}) or {}
        comp = comparisons.get(label, {}) or {}
        if not candidate:
            continue
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- anchor64 geom: `{float(candidate.get('eval_geom', 0.0) or 0.0):.4f}`",
                f"- hard geom: `{float(((candidate.get('hard_subset', {}) or {}).get('eval_geom', 0.0) or 0.0)):.4f}`",
                f"- delta anchor vs control: `{float(comp.get('delta_geom_vs_control_anchor64', 0.0) or 0.0):+.4f}`",
                f"- delta hard vs control: `{float(comp.get('delta_geom_vs_control_hard_subset', 0.0) or 0.0):+.4f}`",
                f"- health no_regression: `{bool(((comp.get('health_delta_vs_control', {}) or {}).get('no_regression', False)))}`",
                f"- reconstructed health no_regression: `{bool(((comp.get('reconstructed_health_delta_vs_control', {}) or {}).get('no_regression', False)))}`",
                f"- delta vs incumbent anchor64: `{float(candidate.get('delta_geom_vs_incumbent_anchor64', 0.0) or 0.0):+.4f}`",
                f"- delta vs frozen anchor64: `{float(candidate.get('delta_geom_vs_frozen_anchor64', 0.0) or 0.0):+.4f}`",
                f"- status: `{comp.get('status', '')}`",
            ]
        )
    write_text(doc_path, "\n".join(lines) + "\n")
    print(f"OK: wrote {doc_path}")


if __name__ == "__main__":
    main()
