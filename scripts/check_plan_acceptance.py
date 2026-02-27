from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_json(path_str: str) -> dict:
    path = _resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_le(label: str, value: float, threshold: float) -> None:
    if value > threshold:
        raise AssertionError(f"{label}={value:.6f} exceeds threshold {threshold:.6f}")


def _assert_ge(label: str, value: float, threshold: float) -> None:
    if value < threshold:
        raise AssertionError(f"{label}={value:.6f} below threshold {threshold:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose-summary", required=True)
    ap.add_argument("--length-stats", required=True)
    ap.add_argument("--oracc-audit", default="")
    ap.add_argument("--max-extra-id0-ratio", type=float, default=1.0)
    ap.add_argument("--max-shorter-half-ratio", type=float, default=55.0)
    ap.add_argument("--max-src-trunc", type=float, default=10.0)
    ap.add_argument("--max-tgt-trunc", type=float, default=8.0)
    ap.add_argument("--min-oracc-removed", type=int, default=1)
    args = ap.parse_args()

    diagnose = _load_json(args.diagnose_summary)
    length_stats = _load_json(args.length_stats)
    health = diagnose.get("output_health", {}) or {}

    _assert_le(
        "exact_extra_id_0_ratio_pct",
        float(health.get("exact_extra_id_0_ratio_pct", 100.0)),
        float(args.max_extra_id0_ratio),
    )
    _assert_le(
        "pred_shorter_than_half_ref_ratio_pct",
        float(health.get("pred_shorter_than_half_ref_ratio_pct", 100.0)),
        float(args.max_shorter_half_ratio),
    )
    _assert_le(
        "src_truncation_ratio_pct",
        float((length_stats.get("source", {}) or {}).get("truncation_ratio_pct", 100.0)),
        float(args.max_src_trunc),
    )
    _assert_le(
        "tgt_truncation_ratio_pct",
        float((length_stats.get("target", {}) or {}).get("truncation_ratio_pct", 100.0)),
        float(args.max_tgt_trunc),
    )

    if args.oracc_audit.strip():
        audit = _load_json(args.oracc_audit)
        removed_exact = int(audit.get("rows_oracc_input", 0)) - int(audit.get("rows_oracc_after_exact_dedupe", 0))
        removed_sim = int(audit.get("rows_oracc_removed_similarity", 0))
        _assert_ge("oracc_removed_total", float(removed_exact + removed_sim), float(args.min_oracc_removed))

    print("OK: acceptance checks passed.")


if __name__ == "__main__":
    main()
