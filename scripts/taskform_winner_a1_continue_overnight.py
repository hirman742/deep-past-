from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> None:
    print(f"[{_utc_now()}] RUN {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[{_utc_now()}] DONE {' '.join(cmd)}", flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="reports/taskform_winner_a1_continue_overnight_20260310")
    args = ap.parse_args()

    report_dir = REPO_ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    status_path = report_dir / "status.json"

    status: dict[str, Any] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "stages": [],
    }
    _write_json(status_path, status)

    def record(stage: str, stage_status: str, payload: dict[str, Any]) -> None:
        status["stages"].append(
            {
                "stage": stage,
                "status": stage_status,
                "timestamp_utc": _utc_now(),
                **payload,
            }
        )
        status["status"] = stage_status if stage_status.startswith("stopped_") else "running"
        _write_json(status_path, status)

    build_summary = REPO_ROOT / "reports" / "taskform_winner_a1_continue_build_20260310" / "summary.json"
    probe_summary = REPO_ROOT / "reports" / "taskform_winner_a1_continue_probe_20260310" / "summary.json"
    wlite_summary = REPO_ROOT / "reports" / "taskform_winner_a1_continue_wlite_20260310" / "summary.json"
    promote_summary = REPO_ROOT / "reports" / "taskform_winner_a1_continue_promote_20260310" / "summary.json"

    _run([sys.executable, str(SCRIPTS_DIR / "taskform_winner_a1_continue_build.py")])
    record("build", "completed", {"summary_json": str(build_summary)})

    _run([sys.executable, str(SCRIPTS_DIR / "taskform_winner_a1_continue_probe_flow.py")])
    probe = _load_json(probe_summary)
    record(
        "probe",
        "completed",
        {
            "summary_json": str(probe_summary),
            "probe_status": probe.get("status", ""),
            "best_ratio_label": probe.get("best_ratio_label", ""),
            "best_ratio_status": probe.get("best_ratio_status", ""),
        },
    )

    if str(probe.get("best_ratio_status", "")) != "review_to_wlite":
        status["status"] = "stopped_after_probe"
        status["finished_at_utc"] = _utc_now()
        _write_json(status_path, status)
        print(f"[{_utc_now()}] STOP after probe: best_ratio_status={probe.get('best_ratio_status','')}", flush=True)
        return

    _run([sys.executable, str(SCRIPTS_DIR / "taskform_winner_a1_continue_wlite_flow.py")])
    wlite = _load_json(wlite_summary)
    record(
        "wlite",
        "completed",
        {
            "summary_json": str(wlite_summary),
            "wlite_status": wlite.get("status", ""),
            "best_ratio_label": wlite.get("best_ratio_label", ""),
        },
    )

    if str(wlite.get("status", "")) != "proceed_fullval":
        status["status"] = "stopped_after_wlite"
        status["finished_at_utc"] = _utc_now()
        _write_json(status_path, status)
        print(f"[{_utc_now()}] STOP after wlite: status={wlite.get('status','')}", flush=True)
        return

    _run([sys.executable, str(SCRIPTS_DIR / "taskform_winner_a1_continue_promote_flow.py")])
    promote = _load_json(promote_summary)
    record(
        "promote",
        "completed",
        {
            "summary_json": str(promote_summary),
            "promote_status": promote.get("status", ""),
            "selected_variant": ((promote.get("selected_candidate", {}) or {}).get("variant", "")),
        },
    )

    status["status"] = "completed"
    status["finished_at_utc"] = _utc_now()
    status["final"] = {
        "build_summary_json": str(build_summary),
        "probe_summary_json": str(probe_summary),
        "wlite_summary_json": str(wlite_summary),
        "promote_summary_json": str(promote_summary),
    }
    _write_json(status_path, status)
    print(f"[{_utc_now()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
