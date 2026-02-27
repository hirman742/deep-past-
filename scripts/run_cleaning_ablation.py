from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _variant_overrides(name: str) -> dict[str, Any]:
    if name == "baseline":
        return {}
    if name == "ws_off":
        return {"preprocess": {"fold_inline_whitespace": False}}
    if name == "t0_on":
        return {"preprocess": {"apply_t0_normalize": True}}
    if name == "lower_src":
        return {"preprocess": {"lowercase_source": True}}
    if name == "lower_tgt":
        return {"preprocess": {"lowercase_target": True}}
    raise ValueError(f"Unsupported ablation variant: {name}")


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--variants", default="baseline,ws_off,t0_on")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=10)
    args = ap.parse_args()

    base_cfg_path = _resolve_path(args.base_config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    base_cfg = _load_yaml(base_cfg_path)

    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not variant_names:
        raise ValueError("No variants provided")

    exp_root = REPO_ROOT / "runs" / "B2_CLEAN_ABLATION"
    config_dir = exp_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    preprocess_script = REPO_ROOT / "scripts" / "preprocess.py"
    train_script = REPO_ROOT / "scripts" / "train_mt5_lora.py"

    records: list[dict[str, Any]] = []

    for variant in variant_names:
        cfg = copy.deepcopy(base_cfg)
        _deep_update(cfg, _variant_overrides(variant))

        paths_cfg = cfg.setdefault("paths", {})
        paths_cfg["processed_dir"] = f"data/processed_{variant}"
        paths_cfg["run_dir"] = f"runs/B2_{variant}"

        cfg_path = config_dir / f"{variant}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        _run([sys.executable, str(preprocess_script), "--config", str(cfg_path)])
        _run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(cfg_path),
                "--fold",
                str(args.fold),
                "--max-steps",
                str(args.max_steps),
            ]
        )

        run_root = _resolve_path(cfg["paths"]["run_dir"], REPO_ROOT / "runs" / "B2_TMP")
        run_dir = run_root.parent / f"{run_root.name}_fold{args.fold}"
        summary_path = run_dir / "run_summary.json"
        stats_path = _resolve_path(cfg["paths"]["processed_dir"], REPO_ROOT / "data" / "processed") / "length_stats.json"

        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing length stats: {stats_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        eval_metrics = summary.get("eval_metrics", {}) or {}

        records.append(
            {
                "variant": variant,
                "config_path": str(cfg_path),
                "run_dir": str(run_dir),
                "eval_bleu": float(eval_metrics.get("eval_bleu", float("nan"))),
                "eval_chrfpp": float(eval_metrics.get("eval_chrfpp", float("nan"))),
                "eval_geom": float(eval_metrics.get("eval_geom", float("nan"))),
                "eval_loss": float(eval_metrics.get("eval_loss", float("nan"))),
                "src_truncation_ratio_pct": float(stats["source"]["truncation_ratio_pct"]),
                "tgt_truncation_ratio_pct": float(stats["target"]["truncation_ratio_pct"]),
            }
        )

    result_df = pd.DataFrame(records).sort_values(["eval_geom", "eval_bleu"], ascending=False).reset_index(drop=True)
    csv_path = exp_root / "ablation_metrics.csv"
    json_path = exp_root / "ablation_best.json"
    result_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result_df.iloc[0].to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {csv_path}")
    print(f"OK: wrote {json_path}")
    print(
        "INFO: best variant="
        f"{result_df.iloc[0]['variant']}, "
        f"geom/bleu/chrfpp="
        f"{result_df.iloc[0]['eval_geom']:.4f}/"
        f"{result_df.iloc[0]['eval_bleu']:.4f}/"
        f"{result_df.iloc[0]['eval_chrfpp']:.4f}"
    )


if __name__ == "__main__":
    main()
