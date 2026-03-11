#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import (
    apply_term_lexicon,
    attach_stable_split,
    clamp_consecutive_repeated_spans,
    collapse_formula_loops,
    delta_geom,
    evaluate_frame,
    formula_count,
    internal_repeat_score,
    load_term_lexicon,
    markdown_table,
    normalize_whitespace,
    repair_gap_markers,
    resolve_path,
    safe_text,
    tokenize_words,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_ORDER = ["pass_a_prediction", "selective_patch_light", "looptrim_concat", "raw_concat"]


def _source_bucket(chunk_total: int, marker_count: int) -> str:
    if int(chunk_total) >= 7:
        return "chunk7plus"
    if 4 <= int(chunk_total) <= 6:
        return "chunk4_6"
    if 2 <= int(chunk_total) <= 3 and int(marker_count) >= 2:
        return "chunk2_3_marked"
    return "chunk2_3_other"


def _build_selective_patch(text: str, lexicon_rows: list[dict[str, str]]) -> str:
    patched = repair_gap_markers(safe_text(text))
    patched = clamp_consecutive_repeated_spans(patched, max_span=12, max_occurrences=2)
    patched = collapse_formula_loops(patched, max_repeats=2)
    if lexicon_rows:
        patched = apply_term_lexicon(patched, lexicon_rows)
    return normalize_whitespace(patched)


def _choose_action(frame: pd.DataFrame, *, bucket: str) -> dict[str, Any]:
    subset = frame.loc[frame["policy_bucket"] == bucket].copy()
    rows: list[dict[str, Any]] = []
    for action in ACTION_ORDER:
        summary = evaluate_frame(
            subset,
            prediction_col=action,
            tag=f"tune_{bucket}_{action}",
            subset_name=f"tune_{bucket}_{action}",
        )
        rows.append(
            {
                "bucket": bucket,
                "action": action,
                "geom": float(summary["eval_geom"]),
                "bleu": float(summary["eval_bleu"]),
                "chrfpp": float(summary["eval_chrfpp"]),
            }
        )
    rows.sort(key=lambda item: (-item["geom"], ACTION_ORDER.index(item["action"])))
    return {"winner": rows[0]["action"], "candidates": rows}


def _apply_policy(frame: pd.DataFrame, *, policy: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    out["policy_action"] = out["policy_bucket"].map(lambda bucket: policy.get(bucket, "pass_a_prediction"))
    out["policy_prediction"] = out.apply(lambda row: safe_text(row[row["policy_action"]]), axis=1)
    return out


def _is_safe_switch(
    row: pd.Series,
    *,
    min_length_ratio: float,
    min_repeat_gain: int,
) -> bool:
    pass_a = safe_text(row.get("pass_a_prediction"))
    candidate = safe_text(row.get("policy_prediction"))
    if not candidate or candidate == pass_a:
        return False
    pass_a_len = max(1, len(tokenize_words(pass_a)))
    candidate_len = len(tokenize_words(candidate))
    repeat_gain = int(internal_repeat_score(pass_a)) - int(internal_repeat_score(candidate))
    formula_gain = int(formula_count(pass_a)) - int(formula_count(candidate))
    if float(candidate_len) < float(min_length_ratio) * float(pass_a_len):
        return False
    return repeat_gain >= int(min_repeat_gain) or formula_gain > 0


def _apply_safe_policy(
    frame: pd.DataFrame,
    *,
    policy: dict[str, str],
    allowed_route_buckets: set[str],
    min_length_ratio: float,
    min_repeat_gain: int,
) -> pd.DataFrame:
    out = _apply_policy(frame, policy=policy)
    out["policy_allowed_bucket"] = out["policy_bucket"].map(lambda bucket: bucket in allowed_route_buckets)
    out["policy_prediction"] = out.apply(
        lambda row: safe_text(row["policy_prediction"])
        if bool(row["policy_allowed_bucket"]) and _is_safe_switch(row, min_length_ratio=min_length_ratio, min_repeat_gain=min_repeat_gain)
        else safe_text(row["pass_a_prediction"]),
        axis=1,
    )
    out["policy_switched"] = out["policy_prediction"] != out["pass_a_prediction"].fillna("").astype(str)
    return out


def _random_policy_benchmark(
    frame: pd.DataFrame,
    *,
    buckets: list[str],
    allowed_route_buckets: set[str],
    iterations: int,
    seed: int,
    min_length_ratio: float,
    min_repeat_gain: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(int(iterations)):
        policy = {bucket: (rng.choice(ACTION_ORDER) if bucket in allowed_route_buckets else "pass_a_prediction") for bucket in buckets}
        mixed = _apply_safe_policy(
            frame,
            policy=policy,
            allowed_route_buckets=allowed_route_buckets,
            min_length_ratio=min_length_ratio,
            min_repeat_gain=min_repeat_gain,
        )
        summary = evaluate_frame(
            mixed,
            prediction_col="policy_prediction",
            tag=f"holdout_random_{idx}",
            subset_name="holdout_random_policy",
        )
        rows.append(
            {
                "iter": idx,
                "geom": float(summary["eval_geom"]),
                "bleu": float(summary["eval_bleu"]),
                "chrfpp": float(summary["eval_chrfpp"]),
                "policy": json.dumps(policy, ensure_ascii=False),
            }
        )
    rows.sort(key=lambda item: item["geom"])
    return rows


def _status_from_holdout(
    *,
    holdout_policy: dict[str, Any],
    holdout_pass_a: dict[str, Any],
    random_rows: list[dict[str, Any]],
) -> tuple[str, str, float]:
    pass_a_delta = float(holdout_policy["eval_geom"]) - float(holdout_pass_a["eval_geom"])
    random_p95 = float(random_rows[min(len(random_rows) - 1, int(round(0.95 * max(0, len(random_rows) - 1))))]["geom"]) if random_rows else 0.0
    if float(holdout_policy["eval_geom"]) > random_p95 and pass_a_delta >= 0.05:
        return "accept_to_w", "holdout policy beats pass_a and random95", random_p95
    if float(holdout_policy["eval_geom"]) > random_p95 and pass_a_delta >= -0.05:
        return "review_stop", "policy beats random95 but not enough over pass_a", random_p95
    return "reject_stop", "policy not strong enough on holdout", random_p95


def _bucket_report_rows(frame: pd.DataFrame, *, prediction_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in sorted(frame["policy_bucket"].dropna().astype(str).unique()):
        subset = frame.loc[frame["policy_bucket"] == bucket].copy()
        summary = evaluate_frame(
            subset,
            prediction_col=prediction_col,
            tag=f"{bucket}_{prediction_col}",
            subset_name=f"{bucket}_{prediction_col}",
        )
        rows.append(
            {
                "bucket": bucket,
                "n": int(len(subset)),
                "geom": round(float(summary["eval_geom"]), 4),
                "bleu": round(float(summary["eval_bleu"]), 4),
                "chrfpp": round(float(summary["eval_chrfpp"]), 4),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-csv", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv")
    ap.add_argument("--routed-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--lexicon-tsv", default="")
    ap.add_argument("--out-dir", default="reports/taskform_l3_source_policy_phase12")
    ap.add_argument("--allowed-route-buckets", default="chunk4_6")
    ap.add_argument("--safety-min-length-ratio", type=float, default=0.8)
    ap.add_argument("--safety-min-repeat-gain", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto-promote", action="store_true")
    args = ap.parse_args()

    official_csv = resolve_path(args.official_csv, REPO_ROOT / "runs" / "missing.csv")
    routed_csv = resolve_path(args.routed_csv, REPO_ROOT / "reports" / "missing.csv")
    lexicon_tsv = resolve_path(args.lexicon_tsv, REPO_ROOT / "reports" / "missing.tsv") if args.lexicon_tsv else None
    out_dir = resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_l3_source_policy_phase12")
    out_dir.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(official_csv).rename(columns={"id": "oare_id"})
    official["id"] = official["oare_id"].fillna("").astype(str)
    official["reference"] = official["reference"].fillna("").astype(str)
    official["prediction"] = official["prediction"].fillna("").astype(str)
    official = attach_stable_split(official, id_col="id")
    official_index = official.set_index("id")

    routed = pd.read_csv(routed_csv)
    routed["id"] = routed["oare_id"].fillna("").astype(str)
    routed["policy_bucket"] = routed.apply(
        lambda row: _source_bucket(int(row.get("orig_chunk_total", 1)), int(row.get("marker_count", 0))),
        axis=1,
    )
    routed["source_char_len"] = routed["source"].fillna("").astype(str).map(len)
    routed["source_tok_len"] = routed["source"].fillna("").astype(str).map(lambda text: len(tokenize_words(text)))
    lexicon_rows = load_term_lexicon(lexicon_tsv)
    routed["selective_patch_light"] = routed["pass_a_prediction"].map(lambda text: _build_selective_patch(text, lexicon_rows))
    routed = attach_stable_split(routed, id_col="id")

    tune = routed.loc[routed["stage_split"] == "tune"].copy()
    holdout = routed.loc[routed["stage_split"] == "holdout"].copy()
    report = routed.loc[routed["stage_split"] == "report"].copy()

    buckets = sorted(routed["policy_bucket"].dropna().astype(str).unique())
    allowed_route_buckets = {item.strip() for item in safe_text(args.allowed_route_buckets).split(",") if item.strip()}
    bucket_choices: list[dict[str, Any]] = []
    policy: dict[str, str] = {}
    for bucket in buckets:
        choice = _choose_action(tune, bucket=bucket)
        policy[bucket] = str(choice["winner"]) if bucket in allowed_route_buckets else "pass_a_prediction"
        bucket_choices.extend(choice["candidates"])

    holdout_policy_frame = _apply_safe_policy(
        holdout,
        policy=policy,
        allowed_route_buckets=allowed_route_buckets,
        min_length_ratio=float(args.safety_min_length_ratio),
        min_repeat_gain=int(args.safety_min_repeat_gain),
    )
    report_policy_frame = _apply_safe_policy(
        report,
        policy=policy,
        allowed_route_buckets=allowed_route_buckets,
        min_length_ratio=float(args.safety_min_length_ratio),
        min_repeat_gain=int(args.safety_min_repeat_gain),
    )
    routed_policy_frame = _apply_safe_policy(
        routed,
        policy=policy,
        allowed_route_buckets=allowed_route_buckets,
        min_length_ratio=float(args.safety_min_length_ratio),
        min_repeat_gain=int(args.safety_min_repeat_gain),
    )

    random_rows = _random_policy_benchmark(
        holdout,
        buckets=buckets,
        allowed_route_buckets=allowed_route_buckets,
        iterations=int(args.iterations),
        seed=int(args.seed),
        min_length_ratio=float(args.safety_min_length_ratio),
        min_repeat_gain=int(args.safety_min_repeat_gain),
    )
    holdout_policy = evaluate_frame(
        holdout_policy_frame,
        prediction_col="policy_prediction",
        tag="holdout_policy",
        subset_name="holdout_policy",
    )
    holdout_pass_a = evaluate_frame(
        holdout_policy_frame,
        prediction_col="pass_a_prediction",
        tag="holdout_pass_a",
        subset_name="holdout_pass_a",
    )
    status, reason, random_p95 = _status_from_holdout(
        holdout_policy=holdout_policy,
        holdout_pass_a=holdout_pass_a,
        random_rows=random_rows,
    )

    summary: dict[str, Any] = {
        "line": "L3",
        "status": status,
        "reason": reason,
        "allowed_route_buckets": sorted(allowed_route_buckets),
        "safety_gate": {
            "min_length_ratio": float(args.safety_min_length_ratio),
            "min_repeat_gain": int(args.safety_min_repeat_gain),
        },
        "policy": policy,
        "random_p95_geom": random_p95,
        "bucket_choices": bucket_choices,
        "holdout_policy": holdout_policy,
        "holdout_pass_a": holdout_pass_a,
        "holdout_switched_rows": int(holdout_policy_frame["policy_switched"].sum()),
        "holdout_looptrim": evaluate_frame(
            holdout_policy_frame,
            prediction_col="looptrim_concat",
            tag="holdout_looptrim",
            subset_name="holdout_looptrim",
        ),
        "holdout_random_best": random_rows[-1] if random_rows else {},
    }

    report_rows: list[dict[str, Any]] = []
    if bool(args.auto_promote) and status in {"accept_to_w", "review_stop"}:
        routed_policy = evaluate_frame(
            routed_policy_frame,
            prediction_col="policy_prediction",
            tag="routed_full_policy",
            subset_name="routed_full_policy",
        )
        routed_pass_a = evaluate_frame(
            routed_policy_frame,
            prediction_col="pass_a_prediction",
            tag="routed_full_pass_a",
            subset_name="routed_full_pass_a",
        )
        mixed = official.copy()
        routed_pred_map = dict(zip(routed_policy_frame["id"].astype(str), routed_policy_frame["policy_prediction"].astype(str)))
        mixed["mixed_prediction"] = mixed.apply(
            lambda row: routed_pred_map.get(str(row["id"]), safe_text(row["prediction"])),
            axis=1,
        )
        mixed_policy = evaluate_frame(
            mixed,
            prediction_col="mixed_prediction",
            tag="official_full_mixed_policy",
            subset_name="official_full_mixed_policy",
            note="easy rows kept official pass_a; routed rows replaced by source-only policy",
        )
        hard_ids = set(routed["id"].astype(str).tolist())
        hard_mixed = mixed.loc[mixed["id"].astype(str).isin(hard_ids)].copy()
        hard_policy = evaluate_frame(
            hard_mixed,
            prediction_col="mixed_prediction",
            tag="official_hard_mixed_policy",
            subset_name="official_hard_mixed_policy",
        )
        report_rows = _bucket_report_rows(report_policy_frame, prediction_col="policy_prediction")
        summary.update(
            {
                "report_policy": evaluate_frame(
                    report_policy_frame,
                    prediction_col="policy_prediction",
                    tag="report_policy",
                    subset_name="report_policy",
                ),
                "routed_full_policy": routed_policy,
                "routed_full_pass_a": routed_pass_a,
                "official_full_mixed_policy": mixed_policy,
                "official_hard_mixed_policy": hard_policy,
                "report_bucket_rows": report_rows,
            }
        )
        mixed[["id", "reference", "prediction", "mixed_prediction"]].to_csv(out_dir / "official_mixed_predictions.csv", index=False)

    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "local_eval.json",
        {
            "holdout_policy": holdout_policy,
            "holdout_pass_a": holdout_pass_a,
            "report_policy": summary.get("report_policy", {}),
            "routed_full_policy": summary.get("routed_full_policy", {}),
            "official_full_mixed_policy": summary.get("official_full_mixed_policy", {}),
        },
    )
    write_json(
        out_dir / "official_like_eval.json",
        {
            "holdout_policy": {**holdout_policy, "note": "pending_official_metric_bridge_same_as_local"},
            "report_policy": {**summary.get("report_policy", {}), "note": "pending_official_metric_bridge_same_as_local"},
            "official_full_mixed_policy": {
                **summary.get("official_full_mixed_policy", {}),
                "note": "pending_official_metric_bridge_same_as_local",
            },
        },
    )
    write_json(
        out_dir / "hard_eval.json",
        {
            "holdout_policy": holdout_policy,
            "routed_full_policy": summary.get("routed_full_policy", {}),
            "official_hard_mixed_policy": summary.get("official_hard_mixed_policy", {}),
        },
    )
    pd.DataFrame(bucket_choices).to_csv(out_dir / "bucket_tune_choices.csv", index=False)
    pd.DataFrame(random_rows).to_csv(out_dir / "holdout_random_policies.csv", index=False)
    routed_policy_frame[
        [
            "id",
            "policy_bucket",
            "stage_split",
            "pass_a_prediction",
            "raw_concat",
            "looptrim_concat",
            "selective_patch_light",
            "policy_action",
            "policy_prediction",
        ]
    ].to_csv(out_dir / "routed_policy_predictions.csv", index=False)

    report_lines = [
        "# L3 Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- random p95 geom: `{random_p95:.4f}`",
        f"- holdout pass_a geom: `{holdout_pass_a['eval_geom']:.4f}`",
        f"- holdout policy geom: `{holdout_policy['eval_geom']:.4f}`",
        "",
        "## Policy",
        markdown_table(
            [{"bucket": bucket, "action": action} for bucket, action in sorted(policy.items())],
            ["bucket", "action"],
        ),
        "",
        "## Tune Winners",
        markdown_table(bucket_choices, ["bucket", "action", "geom", "bleu", "chrfpp"]),
    ]
    if report_rows:
        report_lines.extend(["", "## Report Buckets", markdown_table(report_rows, ["bucket", "n", "geom", "bleu", "chrfpp"])])
    write_text(out_dir / "gate_report.md", "\n".join(report_lines) + "\n")

    print(f"OK: wrote {out_dir/'summary.json'}")
    print(f"OK: wrote {out_dir/'gate_report.md'}")


if __name__ == "__main__":
    main()
