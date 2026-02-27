from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_folds(value: str, default_n_folds: int) -> list[int]:
    if not value.strip():
        return list(range(default_n_folds))
    out: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("No valid folds parsed")
    return sorted(set(out))


def _to_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(number) or math.isinf(number):
        return float("nan")
    return number


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--folds", default="")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}

    default_n_folds = int(preprocess_cfg.get("folds", 5))
    folds = _parse_folds(args.folds, default_n_folds)

    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    train_script = REPO_ROOT / "scripts" / "train_mt5_lora.py"

    records: list[dict[str, Any]] = []
    for fold in folds:
        run_dir = run_root.parent / f"{run_root.name}_fold{fold}"
        summary_path = run_dir / "run_summary.json"

        if args.skip_existing and summary_path.exists():
            print(f"SKIP: fold={fold} summary exists at {summary_path}")
        else:
            cmd = [
                sys.executable,
                str(train_script),
                "--config",
                str(cfg_path),
                "--fold",
                str(fold),
            ]
            if args.max_steps >= 0:
                cmd.extend(["--max-steps", str(args.max_steps)])
            print(f"RUN: fold={fold} {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing run summary for fold {fold}: {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        eval_metrics = summary.get("eval_metrics", {}) or {}
        train_metrics = summary.get("train_metrics", {}) or {}
        records.append(
            {
                "fold": int(fold),
                "run_dir": str(run_dir),
                "eval_bleu": _to_float(eval_metrics.get("eval_bleu")),
                "eval_chrfpp": _to_float(eval_metrics.get("eval_chrfpp")),
                "eval_geom": _to_float(eval_metrics.get("eval_geom")),
                "eval_loss": _to_float(eval_metrics.get("eval_loss")),
                "train_loss": _to_float(train_metrics.get("train_loss")),
                "peak_gpu_memory_mb": _to_float(summary.get("peak_gpu_memory_mb")),
                "trainable_ratio_pct": _to_float(summary.get("trainable_ratio_pct")),
            }
        )

    df = pd.DataFrame(records).sort_values("fold").reset_index(drop=True)
    cv_dir = run_root.parent
    cv_tag = run_root.name
    csv_path = cv_dir / f"{cv_tag}_cv_metrics.csv"
    json_path = cv_dir / f"{cv_tag}_cv_summary.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "config_path": str(cfg_path),
        "folds": [int(x) for x in df["fold"].tolist()],
        "num_folds": int(len(df)),
        "mean": {
            "eval_bleu": float(np.nanmean(df["eval_bleu"].to_numpy(dtype=float))),
            "eval_chrfpp": float(np.nanmean(df["eval_chrfpp"].to_numpy(dtype=float))),
            "eval_geom": float(np.nanmean(df["eval_geom"].to_numpy(dtype=float))),
            "eval_loss": float(np.nanmean(df["eval_loss"].to_numpy(dtype=float))),
            "peak_gpu_memory_mb": float(np.nanmean(df["peak_gpu_memory_mb"].to_numpy(dtype=float))),
        },
        "std": {
            "eval_bleu": float(np.nanstd(df["eval_bleu"].to_numpy(dtype=float))),
            "eval_chrfpp": float(np.nanstd(df["eval_chrfpp"].to_numpy(dtype=float))),
            "eval_geom": float(np.nanstd(df["eval_geom"].to_numpy(dtype=float))),
            "eval_loss": float(np.nanstd(df["eval_loss"].to_numpy(dtype=float))),
            "peak_gpu_memory_mb": float(np.nanstd(df["peak_gpu_memory_mb"].to_numpy(dtype=float))),
        },
        "rows": records,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {csv_path}")
    print(f"OK: wrote {json_path}")
    print(
        "INFO: CV mean geom/bleu/chrfpp="
        f"{summary['mean']['eval_geom']:.4f}/"
        f"{summary['mean']['eval_bleu']:.4f}/"
        f"{summary['mean']['eval_chrfpp']:.4f}"
    )


if __name__ == "__main__":
    main()
