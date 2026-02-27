from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from generation_utils import resolve_generation_settings


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


def _assert_equal(label: str, left: Any, right: Any) -> None:
    if left != right:
        raise AssertionError(f"{label} mismatch: left={left!r}, right={right!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-summary", required=True)
    ap.add_argument("--diagnose-summary", default="")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    expected = resolve_generation_settings(
        model_cfg=cfg.get("model", {}) or {},
        gen_cfg=cfg.get("generation", {}) or {},
    )

    run_summary_path = _resolve_path(args.run_summary, REPO_ROOT / args.run_summary)
    run_summary = _load_json(run_summary_path)
    run_gen = run_summary.get("generation_settings", {}) or {}

    for key in [
        "num_beams",
        "length_penalty",
        "max_new_tokens",
        "min_new_tokens",
        "no_repeat_ngram_size",
        "suppress_extra_ids",
        "bad_tokens_regex",
    ]:
        _assert_equal(f"run_summary.{key}", run_gen.get(key), expected.get(key))

    if args.diagnose_summary.strip():
        diag_summary_path = _resolve_path(args.diagnose_summary, REPO_ROOT / args.diagnose_summary)
        diag_summary = _load_json(diag_summary_path)
        diag_decode = (diag_summary.get("decode", {}) or {}).copy()
        for key in [
            "num_beams",
            "length_penalty",
            "max_new_tokens",
            "min_new_tokens",
            "no_repeat_ngram_size",
            "suppress_extra_ids",
            "bad_tokens_regex",
        ]:
            _assert_equal(f"diagnose.decode.{key}", diag_decode.get(key), expected.get(key))

    print("OK: decode settings are consistent across config/run/diagnose.")


if __name__ == "__main__":
    main()
