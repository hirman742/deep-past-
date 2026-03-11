#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from taskform_phase12_common import resolve_path, write_json, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATTERNS = [
    "metrics.py",
    "**/metrics.py",
    "eval.json",
    "**/eval.json",
    "**/submission_log.md",
    "**/kaggle_infer.ipynb",
]
EXCLUDED_PARTS = {
    ".venv-deeppast",
    ".cache",
    "__pycache__",
    "site-packages",
    ".git",
}


def _candidate_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in CANDIDATE_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            parts = set(path.relative_to(root).parts)
            if parts & EXCLUDED_PARTS:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            rows.append(
                {
                    "path": rel,
                    "name": path.name,
                    "size_bytes": int(path.stat().st_size),
                }
            )
    rows.sort(key=lambda item: item["path"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/taskform_phase12")
    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_phase12")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_rows(REPO_ROOT)
    status = "bridge_ready" if any(row["name"] in {"metrics.py", "eval.json"} for row in candidates) else "missing_bridge"
    summary = {
        "status": status,
        "reason": "repo scan for official metric bridge artifacts",
        "candidates": candidates,
        "recommendation": (
            "wire official metric into L2/L3 gates next"
            if status == "bridge_ready"
            else "no official metric bridge files found; keep official-like layer and add bridge later"
        ),
    }
    write_json(out_dir / "official_metric_probe.json", summary)

    lines = [
        "# Official Metric Probe",
        "",
        f"- status: `{status}`",
        f"- candidates found: `{len(candidates)}`",
        f"- recommendation: {summary['recommendation']}",
    ]
    if candidates:
        lines.append("")
        lines.append("## Candidates")
        for row in candidates:
            lines.append(f"- `{row['path']}`")
    write_text(out_dir / "official_metric_probe.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {out_dir/'official_metric_probe.json'}")


if __name__ == "__main__":
    main()
