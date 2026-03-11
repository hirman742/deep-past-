#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from taskform_a2_a1_flow import REPO_ROOT, _load_json, _resolve_path
from taskform_phase12_common import write_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="reports/taskform_winner_a1_pseudotarget_smoke_20260311/summary.json")
    ap.add_argument("--doc-path", default="docs/taskform_winner_pseudotarget_smoke_2026-03-11.md")
    args = ap.parse_args()

    summary_path = _resolve_path(args.summary, REPO_ROOT / "reports" / "taskform_winner_a1_pseudotarget_smoke_20260311" / "summary.json")
    doc_path = _resolve_path(args.doc_path, REPO_ROOT / "docs" / "taskform_winner_pseudotarget_smoke_2026-03-11.md")
    summary = _load_json(summary_path)

    probe = summary.get("probe", {}) or {}
    hard = (probe.get("hard_subset", {}) or {})
    fullval = summary.get("fullval", {}) or {}
    fullval_hard = (fullval.get("hard_subset", {}) or {})
    cmp_data = summary.get("comparisons", {}) or {}
    mono = summary.get("monolingual_inventory", {}) or {}
    tapt = summary.get("tapt_smoke", {}) or {}
    mix = summary.get("synthetic_mix", {}) or {}
    frozen = summary.get("frozen_candidate", {}) or {}

    lines = [
        "# Winner Competition-only Pseudo-target Smoke（2026-03-11）",
        "",
        f"- status: `{summary.get('status', '')}`",
        f"- reason: {summary.get('reason', '')}",
        f"- summary: `{summary_path}`",
        f"- gate report: `{Path(str(summary.get('report_dir', ''))) / 'gate_report.md'}`",
        "",
        "## Probe",
        "",
        f"- incumbent anchor64 geom: `{float((summary.get('incumbent_anchor64', {}) or {}).get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- probe anchor64 geom: `{float(probe.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- probe hard geom: `{float(hard.get('eval_geom', 0.0) or 0.0):.4f}`",
        f"- delta vs incumbent anchor64: `{float(cmp_data.get('probe_delta_geom_vs_incumbent_anchor64', 0.0) or 0.0):+.4f}`",
        f"- delta vs frozen anchor64: `{float(cmp_data.get('probe_delta_geom_vs_frozen_anchor64', 0.0) or 0.0):+.4f}`",
        "",
        "## Assets",
        "",
        f"- monolingual rows: `{int(mono.get('mono_total_rows', 0) or 0)}`",
        f"- pseudo pool rows: `{int(mono.get('pseudo_pool_rows', 0) or 0)}`",
        f"- TAPT best model: `{tapt.get('best_model_dir', '')}`",
        f"- synthetic rows mixed in: `{int(mix.get('synthetic_rows', 0) or 0)}`",
        f"- mixed rows total: `{int(mix.get('mixed_rows', 0) or 0)}`",
        "",
        "## Compare",
        "",
        f"- frozen fallback anchor/fullval/hard: `{float(frozen.get('anchor64_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen.get('fullval_reconstructed_geom', 0.0) or 0.0):.4f} / {float(frozen.get('hard_geom', 0.0) or 0.0):.4f}`",
    ]
    if fullval:
        lines.extend(
            [
                f"- full-val local geom: `{float(fullval.get('eval_geom', 0.0) or 0.0):.4f}`",
                f"- full-val hard geom: `{float(fullval_hard.get('eval_geom', 0.0) or 0.0):.4f}`",
                f"- full-val delta vs frozen: `{float(cmp_data.get('fullval_delta_geom_vs_frozen', 0.0) or 0.0):+.4f}`",
                f"- hard delta vs frozen: `{float(cmp_data.get('fullval_hard_delta_vs_frozen', 0.0) or 0.0):+.4f}`",
            ]
        )
    else:
        lines.append("- full-val: `not run`")
    write_text(doc_path, "\n".join(lines) + "\n")
    print(f"OK: wrote {doc_path}")


if __name__ == "__main__":
    main()
