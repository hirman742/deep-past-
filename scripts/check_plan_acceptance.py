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
    ap.add_argument("--profile", choices=["smoke", "baseline"], default="smoke")
    ap.add_argument("--max-extra-id0-ratio", type=float, default=None)
    ap.add_argument("--max-shorter-half-ratio", type=float, default=None)
    ap.add_argument("--max-empty-pred-ratio", type=float, default=None)
    ap.add_argument("--max-copy-source-ratio", type=float, default=None)
    ap.add_argument("--max-src-trunc", type=float, default=None)
    ap.add_argument("--max-tgt-trunc", type=float, default=None)
    ap.add_argument("--min-oracc-removed", type=int, default=1)
    args = ap.parse_args()

    profile_defaults = {
        "smoke": {
            "max_extra_id0_ratio": 1.0,
            "max_shorter_half_ratio": 55.0,
            "max_empty_pred_ratio": 5.0,
            "max_copy_source_ratio": 15.0,
            "max_src_trunc": 10.0,
            "max_tgt_trunc": 8.0,
        },
        "baseline": {
            "max_extra_id0_ratio": 0.5,
            "max_shorter_half_ratio": 45.0,
            "max_empty_pred_ratio": 2.0,
            "max_copy_source_ratio": 10.0,
            "max_src_trunc": 8.0,
            "max_tgt_trunc": 6.0,
        },
    }
    selected = profile_defaults[str(args.profile)]

    max_extra_id0_ratio = float(
        selected["max_extra_id0_ratio"] if args.max_extra_id0_ratio is None else args.max_extra_id0_ratio
    )
    max_shorter_half_ratio = float(
        selected["max_shorter_half_ratio"] if args.max_shorter_half_ratio is None else args.max_shorter_half_ratio
    )
    max_empty_pred_ratio = float(
        selected["max_empty_pred_ratio"] if args.max_empty_pred_ratio is None else args.max_empty_pred_ratio
    )
    max_copy_source_ratio = float(
        selected["max_copy_source_ratio"] if args.max_copy_source_ratio is None else args.max_copy_source_ratio
    )
    max_src_trunc = float(selected["max_src_trunc"] if args.max_src_trunc is None else args.max_src_trunc)
    max_tgt_trunc = float(selected["max_tgt_trunc"] if args.max_tgt_trunc is None else args.max_tgt_trunc)

    diagnose = _load_json(args.diagnose_summary)
    length_stats = _load_json(args.length_stats)
    health = diagnose.get("output_health", {}) or {}

    _assert_le(
        "exact_extra_id_0_ratio_pct",
        float(health.get("exact_extra_id_0_ratio_pct", 100.0)),
        max_extra_id0_ratio,
    )
    _assert_le(
        "pred_shorter_than_half_ref_ratio_pct",
        float(health.get("pred_shorter_than_half_ref_ratio_pct", 100.0)),
        max_shorter_half_ratio,
    )
    _assert_le(
        "empty_prediction_ratio_pct",
        float(health.get("empty_prediction_ratio_pct", 100.0)),
        max_empty_pred_ratio,
    )
    _assert_le(
        "copy_source_ratio_pct",
        float(health.get("copy_source_ratio_pct", 100.0)),
        max_copy_source_ratio,
    )
    _assert_le(
        "src_truncation_ratio_pct",
        float((length_stats.get("source", {}) or {}).get("truncation_ratio_pct", 100.0)),
        max_src_trunc,
    )
    _assert_le(
        "tgt_truncation_ratio_pct",
        float((length_stats.get("target", {}) or {}).get("truncation_ratio_pct", 100.0)),
        max_tgt_trunc,
    )

    if args.oracc_audit.strip():
        audit = _load_json(args.oracc_audit)
        removed_exact = int(audit.get("rows_oracc_input", 0)) - int(audit.get("rows_oracc_after_exact_dedupe", 0))
        removed_sim = int(audit.get("rows_oracc_removed_similarity", 0))
        _assert_ge("oracc_removed_total", float(removed_exact + removed_sim), float(args.min_oracc_removed))

    print(
        "OK: acceptance checks passed. "
        f"profile={args.profile}, "
        f"thresholds(extra_id0/shorter/empty/copy/src_trunc/tgt_trunc)="
        f"{max_extra_id0_ratio}/{max_shorter_half_ratio}/{max_empty_pred_ratio}/"
        f"{max_copy_source_ratio}/{max_src_trunc}/{max_tgt_trunc}"
    )


if __name__ == "__main__":
    main()
