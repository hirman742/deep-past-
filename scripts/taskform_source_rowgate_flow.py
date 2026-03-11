#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path
from typing import Any

import pandas as pd

from taskform_phase12_common import (
    apply_term_lexicon,
    attach_stable_split,
    clamp_consecutive_repeated_spans,
    collapse_formula_loops,
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
ACTION_PRESETS = {
    "looptrim_only": ["looptrim_concat"],
    "looptrim_selective": ["looptrim_concat", "selective_patch_light"],
    "all": ["looptrim_concat", "raw_concat", "selective_patch_light"],
}


def _build_selective_patch(text: str, lexicon_rows: list[dict[str, str]]) -> str:
    patched = repair_gap_markers(safe_text(text))
    patched = clamp_consecutive_repeated_spans(patched, max_span=12, max_occurrences=2)
    patched = collapse_formula_loops(patched, max_repeats=2)
    if lexicon_rows:
        patched = apply_term_lexicon(patched, lexicon_rows, apply_builtin_rules=False, normalize_output=False)
    return normalize_whitespace(patched)


def _pick_candidate(
    row: pd.Series,
    *,
    action_cols: list[str],
    min_length_ratio: float,
    min_repeat_gain: int,
) -> tuple[str, str, float]:
    pass_a = safe_text(row.get("pass_a_prediction"))
    pass_a_len = max(1, len(tokenize_words(pass_a)))
    pass_a_repeat = int(internal_repeat_score(pass_a))
    pass_a_formula = int(formula_count(pass_a))

    best_action = "pass_a_prediction"
    best_prediction = pass_a
    best_score = 0.0
    for action in action_cols:
        candidate = safe_text(row.get(action))
        if not candidate or candidate == pass_a:
            continue
        candidate_len = len(tokenize_words(candidate))
        if float(candidate_len) < float(min_length_ratio) * float(pass_a_len):
            continue
        repeat_gain = pass_a_repeat - int(internal_repeat_score(candidate))
        formula_gain = pass_a_formula - int(formula_count(candidate))
        if repeat_gain < int(min_repeat_gain) and formula_gain <= 0:
            continue
        score = float(repeat_gain) + (0.5 * float(formula_gain)) + (0.002 * float(candidate_len - pass_a_len))
        if score > best_score:
            best_score = score
            best_action = action
            best_prediction = candidate
    return best_action, best_prediction, best_score


def _apply_rowgate(
    frame: pd.DataFrame,
    *,
    action_cols: list[str],
    min_length_ratio: float,
    min_repeat_gain: int,
    min_score_threshold: float,
    max_switch_rows: int,
) -> pd.DataFrame:
    out = frame.copy()
    picks = out.apply(
        lambda row: _pick_candidate(
            row,
            action_cols=action_cols,
            min_length_ratio=min_length_ratio,
            min_repeat_gain=min_repeat_gain,
        ),
        axis=1,
    )
    out["candidate_action"] = [item[0] for item in picks]
    out["candidate_prediction"] = [item[1] for item in picks]
    out["candidate_score"] = [float(item[2]) for item in picks]
    out["policy_eligible"] = (
        (out["candidate_action"] != "pass_a_prediction")
        & (out["candidate_score"] >= float(min_score_threshold))
    )
    out["policy_rank"] = math.inf
    eligible = out.loc[out["policy_eligible"]].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(["candidate_score", "id"], ascending=[False, True]).reset_index()
        eligible["policy_rank"] = eligible.index + 1
        selected = eligible
        if int(max_switch_rows) > 0:
            selected = eligible.head(int(max_switch_rows)).copy()
        selected_ids = set(selected["id"].astype(str).tolist())
        out["policy_rank"] = out["id"].astype(str).map(
            dict(zip(eligible["id"].astype(str), eligible["policy_rank"].astype(int)))
        ).fillna(math.inf)
    else:
        selected_ids = set()
    out["policy_action"] = out.apply(
        lambda row: safe_text(row["candidate_action"]) if str(row["id"]) in selected_ids else "pass_a_prediction",
        axis=1,
    )
    out["policy_prediction"] = out.apply(
        lambda row: safe_text(row["candidate_prediction"]) if str(row["id"]) in selected_ids else safe_text(row["pass_a_prediction"]),
        axis=1,
    )
    out["policy_score"] = out.apply(
        lambda row: float(row["candidate_score"]) if str(row["id"]) in selected_ids else 0.0,
        axis=1,
    )
    out["policy_switched"] = out["policy_action"] != "pass_a_prediction"
    return out


def _chunk_band(chunk_total: int) -> str:
    value = int(chunk_total)
    if value >= 7:
        return "chunk7plus"
    if value >= 4:
        return "chunk4_6"
    return "chunk2_3"


def _random_rowgate_benchmark(
    holdout_frame: pd.DataFrame,
    *,
    switch_count: int,
    iterations: int,
    seed: int,
    stratify_cols: list[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = holdout_frame.loc[holdout_frame["policy_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    chosen = holdout_frame.loc[holdout_frame["policy_switched"]].copy()
    strata_counts: dict[tuple[str, ...], int] = {}
    for row in chosen.to_dict(orient="records"):
        key = tuple(safe_text(row.get(col)) for col in stratify_cols)
        strata_counts[key] = strata_counts.get(key, 0) + 1
    for idx in range(int(iterations)):
        chosen_ids: set[str] = set()
        for key, need in strata_counts.items():
            mask = pd.Series(True, index=candidates.index)
            for col, value in zip(stratify_cols, key):
                mask = mask & (candidates[col].astype(str) == value)
            pool = candidates.loc[mask, "id"].astype(str).tolist()
            if not pool:
                continue
            take = min(int(need), len(pool))
            chosen_ids.update(rng.sample(pool, k=take))
        if len(chosen_ids) < int(switch_count):
            remaining_pool = [value for value in candidates["id"].astype(str).tolist() if value not in chosen_ids]
            need = min(int(switch_count) - len(chosen_ids), len(remaining_pool))
            if need > 0:
                chosen_ids.update(rng.sample(remaining_pool, k=need))
        mixed = holdout_frame.copy()
        mixed["random_prediction"] = mixed.apply(
            lambda row: safe_text(row["candidate_prediction"]) if str(row["id"]) in chosen_ids else safe_text(row["pass_a_prediction"]),
            axis=1,
        )
        summary = evaluate_frame(
            mixed,
            prediction_col="random_prediction",
            tag=f"holdout_random_rowgate_{idx}",
            subset_name="holdout_random_rowgate",
        )
        rows.append(
            {
                "iter": idx,
                "geom": float(summary["eval_geom"]),
                "bleu": float(summary["eval_bleu"]),
                "chrfpp": float(summary["eval_chrfpp"]),
                "switch_count": int(len(chosen_ids)),
                "mode": "stratified_eligible_pool",
            }
        )
    rows.sort(key=lambda item: item["geom"])
    return rows


def _status_from_holdout(
    *,
    holdout_policy: dict[str, Any],
    holdout_pass_a: dict[str, Any],
    random_rows: list[dict[str, Any]],
    random_margin: float,
) -> tuple[str, str, float]:
    delta = float(holdout_policy["eval_geom"]) - float(holdout_pass_a["eval_geom"])
    random_p95 = float(random_rows[min(len(random_rows) - 1, int(round(0.95 * max(0, len(random_rows) - 1))))]["geom"]) if random_rows else 0.0
    if float(holdout_policy["eval_geom"]) >= random_p95 + float(random_margin) and delta >= 0.05:
        return "accept_to_w", "holdout row-gate beats pass_a and random95 with margin", random_p95
    if float(holdout_policy["eval_geom"]) > random_p95 and delta > 0.0:
        return "review_stop", "row-gate beats pass_a and random95 but margin is still narrow", random_p95
    return "reject_stop", "row-gate not strong enough on holdout", random_p95


def _band_report_rows(frame: pd.DataFrame, *, prediction_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["chunk2_3", "chunk4_6", "chunk7plus"]:
        subset = frame.loc[frame["chunk_band"] == band].copy()
        if subset.empty:
            continue
        summary = evaluate_frame(
            subset,
            prediction_col=prediction_col,
            tag=f"{band}_{prediction_col}",
            subset_name=f"{band}_{prediction_col}",
        )
        rows.append(
            {
                "chunk_band": band,
                "rows": int(len(subset)),
                "geom": round(float(summary["eval_geom"]), 4),
                "bleu": round(float(summary["eval_bleu"]), 4),
                "chrfpp": round(float(summary["eval_chrfpp"]), 4),
                "switched_rows": int(subset["policy_switched"].sum()) if "policy_switched" in subset.columns else 0,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-csv", default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv")
    ap.add_argument("--routed-csv", default="reports/taskform_dan1_b1_b2_b4/routed_full_predictions.csv")
    ap.add_argument("--lexicon-tsv", default="")
    ap.add_argument("--out-dir", default="reports/taskform_l3_rowgate_phase12")
    ap.add_argument("--action-presets", default="looptrim_only,looptrim_selective,all")
    ap.add_argument("--length-ratios", default="0.8,0.85,0.9")
    ap.add_argument("--repeat-gains", default="1,2")
    ap.add_argument("--score-thresholds", default="0.0,1.0,2.0")
    ap.add_argument("--max-switch-rows", default="6,8,10,-1")
    ap.add_argument("--selection-tolerance", type=float, default=0.03)
    ap.add_argument("--random-margin", type=float, default=0.03)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    official_csv = resolve_path(args.official_csv, REPO_ROOT / "runs" / "missing.csv")
    routed_csv = resolve_path(args.routed_csv, REPO_ROOT / "reports" / "missing.csv")
    lexicon_tsv = resolve_path(args.lexicon_tsv, REPO_ROOT / "reports" / "missing.tsv") if args.lexicon_tsv else None
    out_dir = resolve_path(args.out_dir, REPO_ROOT / "reports" / "taskform_l3_rowgate_phase12")
    out_dir.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(official_csv).rename(columns={"id": "oare_id"})
    official["id"] = official["oare_id"].fillna("").astype(str)
    official["reference"] = official["reference"].fillna("").astype(str)
    official["prediction"] = official["prediction"].fillna("").astype(str)

    routed = pd.read_csv(routed_csv)
    routed["id"] = routed["oare_id"].fillna("").astype(str)
    routed["chunk_band"] = routed["orig_chunk_total"].fillna(1).astype(int).map(_chunk_band)
    lexicon_rows = load_term_lexicon(lexicon_tsv)
    routed["selective_patch_light"] = routed["pass_a_prediction"].map(lambda text: _build_selective_patch(text, lexicon_rows))
    routed = attach_stable_split(routed, id_col="id")

    tune = routed.loc[routed["stage_split"] == "tune"].copy()
    holdout = routed.loc[routed["stage_split"] == "holdout"].copy()
    report = routed.loc[routed["stage_split"] == "report"].copy()

    preset_names = [item.strip() for item in safe_text(args.action_presets).split(",") if item.strip()]
    length_ratios = [float(item.strip()) for item in safe_text(args.length_ratios).split(",") if item.strip()]
    repeat_gains = [int(item.strip()) for item in safe_text(args.repeat_gains).split(",") if item.strip()]
    score_thresholds = [float(item.strip()) for item in safe_text(args.score_thresholds).split(",") if item.strip()]
    max_switch_rows_list = [int(float(item.strip())) for item in safe_text(args.max_switch_rows).split(",") if item.strip()]

    tune_search_rows: list[dict[str, Any]] = []
    for preset_name in preset_names:
        action_cols = ACTION_PRESETS.get(preset_name)
        if not action_cols:
            continue
        for length_ratio in length_ratios:
            for repeat_gain in repeat_gains:
                for score_threshold in score_thresholds:
                    for max_switch_rows in max_switch_rows_list:
                        tuned = _apply_rowgate(
                            tune,
                            action_cols=action_cols,
                            min_length_ratio=length_ratio,
                            min_repeat_gain=repeat_gain,
                            min_score_threshold=score_threshold,
                            max_switch_rows=max_switch_rows,
                        )
                        summary = evaluate_frame(
                            tuned,
                            prediction_col="policy_prediction",
                            tag=f"tune_{preset_name}_{length_ratio}_{repeat_gain}_{score_threshold}_{max_switch_rows}",
                            subset_name="tune_rowgate",
                        )
                        row = {
                            "preset": preset_name,
                            "action_cols": ",".join(action_cols),
                            "min_length_ratio": float(length_ratio),
                            "min_repeat_gain": int(repeat_gain),
                            "min_score_threshold": float(score_threshold),
                            "max_switch_rows": int(max_switch_rows),
                            "geom": float(summary["eval_geom"]),
                            "bleu": float(summary["eval_bleu"]),
                            "chrfpp": float(summary["eval_chrfpp"]),
                            "switched_rows": int(tuned["policy_switched"].sum()),
                            "eligible_rows": int(tuned["policy_eligible"].sum()),
                        }
                        tune_search_rows.append(row)

    if not tune_search_rows:
        raise ValueError("No valid row-gate config produced")

    max_tune_geom = max(float(row["geom"]) for row in tune_search_rows)
    shortlist = [
        row
        for row in tune_search_rows
        if float(row["geom"]) >= max_tune_geom - float(args.selection_tolerance)
    ]
    shortlist.sort(
        key=lambda row: (
            int(row["switched_rows"]),
            int(row["max_switch_rows"]) if int(row["max_switch_rows"]) > 0 else 10**9,
            -float(row["min_score_threshold"]),
            -float(row["min_length_ratio"]),
            -float(row["geom"]),
            row["preset"],
        )
    )
    best_config = shortlist[0]

    best_action_cols = [item for item in safe_text(best_config["action_cols"]).split(",") if item]
    holdout_policy_frame = _apply_rowgate(
        holdout,
        action_cols=best_action_cols,
        min_length_ratio=float(best_config["min_length_ratio"]),
        min_repeat_gain=int(best_config["min_repeat_gain"]),
        min_score_threshold=float(best_config["min_score_threshold"]),
        max_switch_rows=int(best_config["max_switch_rows"]),
    )
    report_policy_frame = _apply_rowgate(
        report,
        action_cols=best_action_cols,
        min_length_ratio=float(best_config["min_length_ratio"]),
        min_repeat_gain=int(best_config["min_repeat_gain"]),
        min_score_threshold=float(best_config["min_score_threshold"]),
        max_switch_rows=int(best_config["max_switch_rows"]),
    )
    routed_policy_frame = _apply_rowgate(
        routed,
        action_cols=best_action_cols,
        min_length_ratio=float(best_config["min_length_ratio"]),
        min_repeat_gain=int(best_config["min_repeat_gain"]),
        min_score_threshold=float(best_config["min_score_threshold"]),
        max_switch_rows=int(best_config["max_switch_rows"]),
    )

    random_rows = _random_rowgate_benchmark(
        holdout_policy_frame,
        switch_count=int(holdout_policy_frame["policy_switched"].sum()),
        iterations=int(args.iterations),
        seed=int(args.seed),
        stratify_cols=["chunk_band", "candidate_action"],
    )
    holdout_policy = evaluate_frame(
        holdout_policy_frame,
        prediction_col="policy_prediction",
        tag="holdout_rowgate_policy",
        subset_name="holdout_rowgate_policy",
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
        random_margin=float(args.random_margin),
    )

    routed_policy = evaluate_frame(
        routed_policy_frame,
        prediction_col="policy_prediction",
        tag="routed_full_rowgate_policy",
        subset_name="routed_full_rowgate_policy",
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
        tag="official_full_mixed_rowgate_policy",
        subset_name="official_full_mixed_rowgate_policy",
        note="easy rows kept official pass_a; routed rows replaced by row-level safety gate",
    )
    hard_ids = set(routed["id"].astype(str).tolist())
    hard_mixed = mixed.loc[mixed["id"].astype(str).isin(hard_ids)].copy()
    hard_policy = evaluate_frame(
        hard_mixed,
        prediction_col="mixed_prediction",
        tag="official_hard_mixed_rowgate_policy",
        subset_name="official_hard_mixed_rowgate_policy",
    )

    summary: dict[str, Any] = {
        "line": "L3-rowgate",
        "status": status,
        "reason": reason,
        "best_config": best_config,
        "selection_tolerance": float(args.selection_tolerance),
        "random_margin": float(args.random_margin),
        "random_p95_geom": random_p95,
        "holdout_policy": holdout_policy,
        "holdout_pass_a": holdout_pass_a,
        "holdout_eligible_rows": int(holdout_policy_frame["policy_eligible"].sum()),
        "holdout_switched_rows": int(holdout_policy_frame["policy_switched"].sum()),
        "holdout_random_best": random_rows[-1] if random_rows else {},
        "routed_full_policy": routed_policy,
        "routed_full_pass_a": routed_pass_a,
        "official_full_mixed_policy": mixed_policy,
        "official_hard_mixed_policy": hard_policy,
        "report_policy": evaluate_frame(
            report_policy_frame,
            prediction_col="policy_prediction",
            tag="report_rowgate_policy",
            subset_name="report_rowgate_policy",
        ),
        "report_pass_a": evaluate_frame(
            report_policy_frame,
            prediction_col="pass_a_prediction",
            tag="report_pass_a",
            subset_name="report_pass_a",
        ),
        "random_baseline_summary": {
            "iterations": int(args.iterations),
            "mode": "stratified_eligible_pool",
            "eligible_rows_holdout": int(holdout_policy_frame["policy_eligible"].sum()),
            "switched_rows_holdout": int(holdout_policy_frame["policy_switched"].sum()),
            "random_p50_geom": float(random_rows[len(random_rows) // 2]["geom"]) if random_rows else 0.0,
            "random_p95_geom": random_p95,
            "random_max_geom": float(random_rows[-1]["geom"]) if random_rows else 0.0,
        },
        "tune_search_top": sorted(tune_search_rows, key=lambda item: (-item["geom"], item["preset"]))[:20],
        "report_band_rows": _band_report_rows(report_policy_frame, prediction_col="policy_prediction"),
        "holdout_band_rows": _band_report_rows(holdout_policy_frame, prediction_col="policy_prediction"),
    }

    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "local_eval.json",
        {
            "holdout_policy": holdout_policy,
            "holdout_pass_a": holdout_pass_a,
            "routed_full_policy": routed_policy,
            "official_full_mixed_policy": mixed_policy,
            "official_hard_mixed_policy": hard_policy,
        },
    )
    write_json(
        out_dir / "official_like_eval.json",
        {
            "holdout_policy": {**holdout_policy, "note": "pending_official_metric_bridge_same_as_local"},
            "routed_full_policy": {**routed_policy, "note": "pending_official_metric_bridge_same_as_local"},
            "official_full_mixed_policy": {**mixed_policy, "note": "pending_official_metric_bridge_same_as_local"},
        },
    )
    write_json(
        out_dir / "hard_eval.json",
        {
            "holdout_policy": holdout_policy,
            "routed_full_policy": routed_policy,
            "official_hard_mixed_policy": hard_policy,
        },
    )
    pd.DataFrame(tune_search_rows).sort_values(["geom", "preset"], ascending=[False, True]).to_csv(out_dir / "tune_search.csv", index=False)
    pd.DataFrame(random_rows).to_csv(out_dir / "holdout_random_policies.csv", index=False)
    holdout_policy_frame[
        [
            "id",
            "orig_chunk_total",
            "marker_count",
            "chunk_band",
            "candidate_action",
            "candidate_score",
            "policy_eligible",
            "policy_action",
            "policy_score",
            "policy_switched",
        ]
    ].to_csv(out_dir / "eligible_rows.csv", index=False)
    write_json(out_dir / "random_baseline_summary.json", summary["random_baseline_summary"])
    routed_policy_frame[
        [
            "id",
            "orig_chunk_total",
            "marker_count",
            "chunk_band",
            "stage_split",
            "pass_a_prediction",
            "looptrim_concat",
            "raw_concat",
            "selective_patch_light",
            "candidate_action",
            "candidate_prediction",
            "candidate_score",
            "policy_eligible",
            "policy_action",
            "policy_prediction",
            "policy_score",
            "policy_switched",
        ]
    ].to_csv(out_dir / "routed_policy_predictions.csv", index=False)
    mixed[["id", "reference", "prediction", "mixed_prediction"]].to_csv(out_dir / "official_mixed_predictions.csv", index=False)

    lines = [
        "# L3 Row-Gate Report",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- best preset: `{best_config['preset']}`",
        f"- action cols: `{best_config['action_cols']}`",
        f"- min length ratio: `{best_config['min_length_ratio']}`",
        f"- min repeat gain: `{best_config['min_repeat_gain']}`",
        f"- min score threshold: `{best_config['min_score_threshold']}`",
        f"- max switch rows: `{best_config['max_switch_rows']}`",
        f"- holdout eligible rows: `{summary['holdout_eligible_rows']}`",
        f"- holdout switched rows: `{summary['holdout_switched_rows']}`",
        f"- random p95 geom: `{random_p95:.4f}`",
        f"- holdout pass_a geom: `{holdout_pass_a['eval_geom']:.4f}`",
        f"- holdout row-gate geom: `{holdout_policy['eval_geom']:.4f}`",
        f"- routed full pass_a geom: `{routed_pass_a['eval_geom']:.4f}`",
        f"- routed full row-gate geom: `{routed_policy['eval_geom']:.4f}`",
        f"- official mixed geom: `{mixed_policy['eval_geom']:.4f}`",
        f"- official hard mixed geom: `{hard_policy['eval_geom']:.4f}`",
        "",
        "## Tune Top",
        markdown_table(summary["tune_search_top"], ["preset", "action_cols", "min_length_ratio", "min_repeat_gain", "geom", "bleu", "chrfpp", "switched_rows"]),
    ]
    if summary["holdout_band_rows"]:
        lines.extend(["", "## Holdout Bands", markdown_table(summary["holdout_band_rows"], ["chunk_band", "rows", "geom", "bleu", "chrfpp", "switched_rows"])])
    write_text(out_dir / "gate_report.md", "\n".join(lines) + "\n")
    print(f"OK: wrote {out_dir/'summary.json'}")
    print(f"OK: wrote {out_dir/'gate_report.md'}")


if __name__ == "__main__":
    main()
