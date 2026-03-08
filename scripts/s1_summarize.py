from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "configs/cloud_stage1_len512_lr2e4.yaml",
    "configs/cloud_stage1_len512_lr1e4.yaml",
]
DEFAULT_REPORT = "docs/cloud_s1_report.md"


@dataclass
class RunReport:
    config_path: Path
    config_name: str
    summary_path: Path | None
    diagnostic_summary_path: Path | None
    run_dir: Path | None
    best_model_path: Path | None
    status: str
    eval_geom: float | None
    eval_bleu: float | None
    eval_chrfpp: float | None
    train_runtime: float | None


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_python() -> Path:
    preferred = REPO_ROOT / ".venv-deeppast" / "bin" / "python"
    if preferred.exists():
        return preferred
    return Path(sys.executable).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_run_summary(config_name: str, fold: int) -> Path | None:
    pattern = f"*_fold{fold}/run_summary.json"
    candidates: list[Path] = []
    for summary_path in (REPO_ROOT / "runs").glob(pattern):
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue
        summary_config = Path(str(summary.get("config_path", ""))).name
        if summary_config == config_name:
            candidates.append(summary_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_latest_diag(run_dir: Path) -> Path | None:
    diag_dir = run_dir / "diagnostics"
    if not diag_dir.exists():
        return None
    candidates = list(diag_dir.glob("val_diagnostic_summary*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_diagnostics(config_path: Path, fold: int, best_model_path: Path) -> Path:
    cmd = [
        str(_resolve_python()),
        str(REPO_ROOT / "scripts" / "diagnose_val_outputs.py"),
        "--config",
        str(config_path),
        "--fold",
        str(fold),
        "--checkpoint-dir",
        str(best_model_path),
        "--tag",
        "s1_final",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    diag_path = _find_latest_diag(best_model_path.parent)
    if diag_path is None:
        raise FileNotFoundError(f"Missing diagnostic summary under {best_model_path.parent / 'diagnostics'}")
    return diag_path


def _extract_train_runtime(summary: dict[str, Any]) -> float | None:
    train_metrics = summary.get("train_metrics", {}) or {}
    value = train_metrics.get("train_runtime")
    if value is None:
        return None
    return float(value)


def _extract_eval_metrics(summary: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    eval_metrics = summary.get("eval_metrics", {}) or {}
    geom = eval_metrics.get("eval_geom")
    bleu = eval_metrics.get("eval_bleu")
    chrfpp = eval_metrics.get("eval_chrfpp")
    if geom is None or bleu is None or chrfpp is None:
        return None, None, None
    return float(geom), float(bleu), float(chrfpp)


def _extract_diag_metrics(summary: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    metrics = summary.get("metrics", {}) or {}
    geom = metrics.get("geom")
    bleu = metrics.get("bleu")
    chrfpp = metrics.get("chrfpp")
    if geom is None or bleu is None or chrfpp is None:
        return None, None, None
    return float(geom), float(bleu), float(chrfpp)


def _build_run_report(config_path: Path, fold: int) -> RunReport:
    summary_path = _find_latest_run_summary(config_path.name, fold)
    if summary_path is None:
        return RunReport(
            config_path=config_path,
            config_name=config_path.stem,
            summary_path=None,
            diagnostic_summary_path=None,
            run_dir=None,
            best_model_path=None,
            status="pending",
            eval_geom=None,
            eval_bleu=None,
            eval_chrfpp=None,
            train_runtime=None,
        )

    summary = _load_json(summary_path)
    run_dir = Path(str(summary.get("run_dir", summary_path.parent))).resolve()
    best_model_path = run_dir / "best_model"
    train_runtime = _extract_train_runtime(summary)
    eval_geom, eval_bleu, eval_chrfpp = _extract_eval_metrics(summary)

    diagnostic_summary_path: Path | None = None
    status = "ok"

    if eval_geom is None:
        if not best_model_path.exists():
            status = "missing_best_model"
        else:
            diagnostic_summary_path = _find_latest_diag(run_dir)
            if diagnostic_summary_path is None:
                diagnostic_summary_path = _run_diagnostics(config_path, fold, best_model_path)
            diag_summary = _load_json(diagnostic_summary_path)
            eval_geom, eval_bleu, eval_chrfpp = _extract_diag_metrics(diag_summary)
            if eval_geom is None:
                status = "missing_decode_metrics"
            else:
                status = "ok"

    return RunReport(
        config_path=config_path,
        config_name=config_path.stem,
        summary_path=summary_path,
        diagnostic_summary_path=diagnostic_summary_path,
        run_dir=run_dir,
        best_model_path=best_model_path,
        status=status,
        eval_geom=eval_geom,
        eval_bleu=eval_bleu,
        eval_chrfpp=eval_chrfpp,
        train_runtime=train_runtime,
    )


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _fmt_path(path: Path | None) -> str:
    if path is None:
        return "NA"
    return str(path)


def _write_report(path: Path, reports: list[RunReport]) -> None:
    lines = [
        "# Cloud S1 Report",
        "",
        "| config | status | eval_geom | eval_bleu | eval_chrfpp | train_runtime_s | run_dir | best_model |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for report in reports:
        lines.append(
            "| "
            + " | ".join(
                [
                    report.config_name,
                    report.status,
                    _fmt_float(report.eval_geom),
                    _fmt_float(report.eval_bleu),
                    _fmt_float(report.eval_chrfpp),
                    _fmt_float(report.train_runtime),
                    _fmt_path(report.run_dir),
                    _fmt_path(report.best_model_path),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Details", ""])
    for report in reports:
        lines.extend(
            [
                f"### {report.config_name}",
                f"- status: {report.status}",
                f"- config_path: {report.config_path}",
                f"- run_dir: {_fmt_path(report.run_dir)}",
                f"- best_model: {_fmt_path(report.best_model_path)}",
                f"- run_summary: {_fmt_path(report.summary_path)}",
                f"- diagnostic_summary: {_fmt_path(report.diagnostic_summary_path)}",
                f"- eval_geom: {_fmt_float(report.eval_geom)}",
                f"- eval_bleu: {_fmt_float(report.eval_bleu)}",
                f"- eval_chrfpp: {_fmt_float(report.eval_chrfpp)}",
                f"- train_runtime_s: {_fmt_float(report.train_runtime)}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--config", action="append", dest="configs")
    args = parser.parse_args()

    config_values = args.configs or DEFAULT_CONFIGS
    config_paths = [_resolve_path(value) for value in config_values]
    reports = [_build_run_report(config_path, args.fold) for config_path in config_paths]

    report_path = _resolve_path(args.report_path)
    _write_report(report_path, reports)

    print(f"report_path={report_path}")
    for report in reports:
        print(f"config={report.config_name}")
        print(f"run_dir={_fmt_path(report.run_dir)}")
        print(f"best_model={_fmt_path(report.best_model_path)}")
        print(f"status={report.status}")

    all_ready = all(report.summary_path is not None and report.eval_geom is not None for report in reports)
    return 0 if all_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
