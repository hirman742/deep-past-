#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fusion_flow_common import (
    build_dan1_edit_prompt,
    build_dan1_flat_prompt,
    build_dan1_prompt,
    build_parent_payloads,
    build_processed_rows,
    generate_predictions,
    load_base_merged,
    load_val_chunk_drafts,
    load_yaml,
    normalize_whitespace,
    payloads_to_metadata_frame,
    resolve_path,
    sanitize_draft,
    safe_text,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _annotate_sanitized_columns(
    frame: pd.DataFrame,
    *,
    sanitize_pred_drafts: bool,
    pred_draft_max_words: int,
    pred_draft_tail_words: int,
    sanitize_oracle_drafts: bool,
    oracle_draft_max_words: int,
    oracle_draft_tail_words: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["draft_prediction"] = work["draft_prediction"].fillna("").astype(str)
    work["chunk_target"] = work["chunk_target"].fillna("").astype(str)
    work["draft_prediction_sanitized"] = work["draft_prediction"].map(
        lambda text: sanitize_draft(
            text,
            max_words=int(pred_draft_max_words),
            tail_words=int(pred_draft_tail_words),
        )
        if sanitize_pred_drafts
        else normalize_whitespace(text)
    )
    work["chunk_target_sanitized"] = work["chunk_target"].map(
        lambda text: sanitize_draft(
            text,
            max_words=int(oracle_draft_max_words),
            tail_words=int(oracle_draft_tail_words),
        )
        if sanitize_oracle_drafts
        else normalize_whitespace(text)
    )
    return work


def _build_groups_from_cache(
    *,
    draft_cache_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    smoke_train_parents: int,
    smoke_val_parents: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    full_groups: dict[str, list[dict]] = {}
    smoke_groups: dict[str, list[dict]] = {}
    routed_meta = metadata_df.loc[metadata_df["is_routed_hard"] == True].copy()
    meta_index = routed_meta.set_index("parent_oare_id")
    for parent_id, group in draft_cache_df.groupby("parent_oare_id", sort=False):
        if parent_id not in meta_index.index:
            continue
        rows = group.sort_values("chunk_index").to_dict(orient="records")
        full_groups[str(parent_id)] = rows
        meta = meta_index.loc[parent_id]
        keep_for_smoke = (
            (meta["split"] == "train" and int(meta["route_rank"]) <= int(smoke_train_parents))
            or (meta["split"] == "val" and int(meta["route_rank"]) <= int(smoke_val_parents))
        )
        if keep_for_smoke:
            smoke_groups[str(parent_id)] = rows
    return full_groups, smoke_groups


def _group_chunk_rows(
    *,
    payloads: list[dict],
    draft_map: dict[str, str],
    metadata_df: pd.DataFrame,
    mode: str,
    smoke_train_parents: int,
    smoke_val_parents: int,
    sanitize_pred_drafts: bool,
    pred_draft_max_words: int,
    pred_draft_tail_words: int,
    sanitize_oracle_drafts: bool,
    oracle_draft_max_words: int,
    oracle_draft_tail_words: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], pd.DataFrame]:
    metadata_index = metadata_df.set_index("parent_oare_id")
    full_groups: dict[str, list[dict]] = {}
    smoke_groups: dict[str, list[dict]] = {}
    draft_rows: list[dict] = []
    for payload in payloads:
        if not bool(payload["is_routed_hard"]):
            continue
        chunk_rows: list[dict] = []
        meta = metadata_index.loc[payload["parent_oare_id"]]
        for chunk in payload["chunks"]:
            draft_prediction = safe_text(chunk["target"] if mode == "oracle" else draft_map.get(chunk["oare_id"], ""))
            chunk_target = safe_text(chunk["target"])
            pred_sanitized = sanitize_draft(
                draft_prediction,
                max_words=int(pred_draft_max_words),
                tail_words=int(pred_draft_tail_words),
            ) if sanitize_pred_drafts else normalize_whitespace(draft_prediction)
            oracle_sanitized = sanitize_draft(
                chunk_target,
                max_words=int(oracle_draft_max_words),
                tail_words=int(oracle_draft_tail_words),
            ) if sanitize_oracle_drafts else normalize_whitespace(chunk_target)
            row = {
                "parent_oare_id": payload["parent_oare_id"],
                "oare_id": chunk["oare_id"],
                "fold": int(payload["fold"]),
                "split": payload["split"],
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_total": int(chunk["chunk_total"]),
                "draft_prediction": draft_prediction,
                "draft_prediction_sanitized": pred_sanitized,
                "chunk_target": chunk_target,
                "chunk_target_sanitized": oracle_sanitized,
                "chunk_source_model": safe_text(chunk["source"]),
                "chunk_source_raw": safe_text(chunk["source_raw"]),
                "parent_translation": safe_text(payload["parent_translation"]),
                "parent_transliteration": safe_text(payload["parent_transliteration"]),
                "route_rank": int(payload["route_rank"]),
                "route_score": float(payload["route_score"]),
                "route_reason": safe_text(payload["route_reason"]),
                "has_gap": bool(chunk["has_gap"]),
                "has_bracket": bool(chunk["has_bracket"]),
                "has_subscript": bool(chunk["has_subscript"]),
                "has_x": bool(chunk["has_x"]),
                "head": safe_text(chunk["head"]),
                "tail": safe_text(chunk["tail"]),
            }
            chunk_rows.append(row)
            if mode == "pred":
                draft_rows.append(row)
        full_groups[payload["parent_oare_id"]] = chunk_rows
        keep_for_smoke = (
            (payload["split"] == "train" and int(payload["route_rank"]) <= int(smoke_train_parents))
            or (payload["split"] == "val" and int(payload["route_rank"]) <= int(smoke_val_parents))
        )
        if keep_for_smoke:
            smoke_groups[payload["parent_oare_id"]] = chunk_rows
        metadata_df.loc[metadata_df["parent_oare_id"] == payload["parent_oare_id"], "smoke_keep"] = bool(keep_for_smoke)
        metadata_df.loc[metadata_df["parent_oare_id"] == payload["parent_oare_id"], "line_mode"] = mode
        metadata_df.loc[metadata_df["parent_oare_id"] == payload["parent_oare_id"], "route_rank"] = int(meta["route_rank"])
    return full_groups, smoke_groups, pd.DataFrame(draft_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml")
    ap.add_argument("--base-processed-dir", default="data/processed_byt5_chunks_align_gc_cost14")
    ap.add_argument(
        "--winner-checkpoint-dir",
        default="runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250",
    )
    ap.add_argument(
        "--winner-val-diagnostic-csv",
        default=(
            "runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/"
            "val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv"
        ),
    )
    ap.add_argument("--out-root", default="data/taskform_dan1_fold0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--route-ref-tok-threshold", type=int, default=129)
    ap.add_argument("--route-marker-ref-tok-threshold", type=int, default=96)
    ap.add_argument("--smoke-train-parents", type=int, default=256)
    ap.add_argument("--smoke-val-parents", type=int, default=64)
    ap.add_argument("--predict-batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--length-penalty", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--routed-processed-dir", default="data/processed_taskform_dan1_routed_fold0")
    ap.add_argument("--pred-smoke-processed-dir", default="data/processed_taskform_dan1_pred_smoke_fold0")
    ap.add_argument("--oracle-smoke-processed-dir", default="data/processed_taskform_dan1_oracle_smoke_fold0")
    ap.add_argument("--prompt-style", choices=["xml", "flat", "edit_combined"], default="xml")
    ap.add_argument("--sanitize-pred-drafts", action="store_true")
    ap.add_argument("--pred-draft-max-words", type=int, default=16)
    ap.add_argument("--pred-draft-tail-words", type=int, default=4)
    ap.add_argument("--sanitize-oracle-drafts", action="store_true")
    ap.add_argument("--oracle-draft-max-words", type=int, default=24)
    ap.add_argument("--oracle-draft-tail-words", type=int, default=6)
    ap.add_argument("--reuse-metadata-csv", default="")
    ap.add_argument("--reuse-draft-cache-csv", default="")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config, REPO_ROOT / "runs" / "STEER" / "generated_configs" / "continue_s4_bs24_len640_seg5.yaml"))
    processed_dir = resolve_path(args.base_processed_dir, REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14")
    winner_checkpoint_dir = resolve_path(
        args.winner_checkpoint_dir,
        REPO_ROOT / "runs" / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0" / "checkpoint-250",
    )
    winner_val_diagnostic_csv = resolve_path(
        args.winner_val_diagnostic_csv,
        REPO_ROOT
        / "runs"
        / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0"
        / "diagnostics"
        / "val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv",
    )
    out_root = resolve_path(args.out_root, REPO_ROOT / "data" / "taskform_dan1_fold0")
    out_root.mkdir(parents=True, exist_ok=True)

    reuse_metadata_csv = resolve_path(args.reuse_metadata_csv, out_root / "missing_metadata.csv")
    reuse_draft_cache_csv = resolve_path(args.reuse_draft_cache_csv, out_root / "missing_draft_cache.csv")

    if args.reuse_metadata_csv and args.reuse_draft_cache_csv:
        metadata_df = pd.read_csv(reuse_metadata_csv)
        draft_cache_df = pd.read_csv(reuse_draft_cache_csv)
        draft_cache_df = _annotate_sanitized_columns(
            draft_cache_df,
            sanitize_pred_drafts=bool(args.sanitize_pred_drafts),
            pred_draft_max_words=int(args.pred_draft_max_words),
            pred_draft_tail_words=int(args.pred_draft_tail_words),
            sanitize_oracle_drafts=bool(args.sanitize_oracle_drafts),
            oracle_draft_max_words=int(args.oracle_draft_max_words),
            oracle_draft_tail_words=int(args.oracle_draft_tail_words),
        )
        pred_groups, pred_smoke_groups = _build_groups_from_cache(
            draft_cache_df=draft_cache_df,
            metadata_df=metadata_df,
            smoke_train_parents=int(args.smoke_train_parents),
            smoke_val_parents=int(args.smoke_val_parents),
        )
        oracle_groups = pred_groups
        oracle_smoke_groups = pred_smoke_groups
    else:
        merged = load_base_merged(processed_dir)
        base_model_name = str((cfg.get("model", {}) or {}).get("name", "google/byt5-small"))
        max_source_length = int((cfg.get("model", {}) or {}).get("max_source_length", 640))
        payloads = build_parent_payloads(
            merged=merged,
            fold=int(args.fold),
            base_tokenizer_name=base_model_name,
            route_ref_tok_threshold=int(args.route_ref_tok_threshold),
            route_marker_ref_tok_threshold=int(args.route_marker_ref_tok_threshold),
        )
        metadata_df = payloads_to_metadata_frame(payloads)

        val_draft_map = load_val_chunk_drafts(winner_val_diagnostic_csv)
        train_chunks = []
        for payload in payloads:
            if payload["split"] != "train" or not bool(payload["is_routed_hard"]):
                continue
            for chunk in payload["chunks"]:
                train_chunks.append({"oare_id": chunk["oare_id"], "source": safe_text(chunk["source"])})
        if train_chunks:
            train_predictions = generate_predictions(
                model_name=base_model_name,
                checkpoint_dir=winner_checkpoint_dir,
                sources=[row["source"] for row in train_chunks],
                max_source_length=max_source_length,
                predict_batch_size=int(args.predict_batch_size),
                num_beams=int(args.num_beams),
                length_penalty=float(args.length_penalty),
                max_new_tokens=int(args.max_new_tokens),
                bad_tokens_regex=str((cfg.get("generation", {}) or {}).get("bad_tokens_regex", r"<extra_id_\d+>")),
                suppress_extra_ids=bool((cfg.get("generation", {}) or {}).get("suppress_extra_ids", True)),
            )
            train_draft_map = {
                row["oare_id"]: safe_text(prediction)
                for row, prediction in zip(train_chunks, train_predictions)
            }
        else:
            train_draft_map = {}
        pred_draft_map = dict(val_draft_map)
        pred_draft_map.update(train_draft_map)

        pred_groups, pred_smoke_groups, draft_cache_df = _group_chunk_rows(
            payloads=payloads,
            draft_map=pred_draft_map,
            metadata_df=metadata_df.copy(),
            mode="pred",
            smoke_train_parents=int(args.smoke_train_parents),
            smoke_val_parents=int(args.smoke_val_parents),
            sanitize_pred_drafts=bool(args.sanitize_pred_drafts),
            pred_draft_max_words=int(args.pred_draft_max_words),
            pred_draft_tail_words=int(args.pred_draft_tail_words),
            sanitize_oracle_drafts=bool(args.sanitize_oracle_drafts),
            oracle_draft_max_words=int(args.oracle_draft_max_words),
            oracle_draft_tail_words=int(args.oracle_draft_tail_words),
        )
        oracle_groups, oracle_smoke_groups, _ = _group_chunk_rows(
            payloads=payloads,
            draft_map=pred_draft_map,
            metadata_df=metadata_df.copy(),
            mode="oracle",
            smoke_train_parents=int(args.smoke_train_parents),
            smoke_val_parents=int(args.smoke_val_parents),
            sanitize_pred_drafts=bool(args.sanitize_pred_drafts),
            pred_draft_max_words=int(args.pred_draft_max_words),
            pred_draft_tail_words=int(args.pred_draft_tail_words),
            sanitize_oracle_drafts=bool(args.sanitize_oracle_drafts),
            oracle_draft_max_words=int(args.oracle_draft_max_words),
            oracle_draft_tail_words=int(args.oracle_draft_tail_words),
        )

    if args.prompt_style == "flat":
        pred_prompt_builder = lambda chunk_rows: build_dan1_flat_prompt(
            chunk_rows, mode="pred", draft_field="draft_prediction_sanitized" if args.sanitize_pred_drafts else "draft_prediction"
        )
        oracle_prompt_builder = lambda chunk_rows: build_dan1_flat_prompt(
            chunk_rows, mode="oracle", draft_field="chunk_target_sanitized" if args.sanitize_oracle_drafts else "chunk_target"
        )
    elif args.prompt_style == "edit_combined":
        pred_prompt_builder = lambda chunk_rows: build_dan1_edit_prompt(
            chunk_rows, mode="pred", draft_field="draft_prediction_sanitized" if args.sanitize_pred_drafts else "draft_prediction"
        )
        oracle_prompt_builder = lambda chunk_rows: build_dan1_edit_prompt(
            chunk_rows, mode="oracle", draft_field="chunk_target_sanitized" if args.sanitize_oracle_drafts else "chunk_target"
        )
    else:
        pred_prompt_builder = lambda chunk_rows: build_dan1_prompt(chunk_rows, mode="pred")
        oracle_prompt_builder = lambda chunk_rows: build_dan1_prompt(chunk_rows, mode="oracle")

    pred_rows, pred_folds = build_processed_rows(
        parent_groups=pred_groups,
        metadata_frame=metadata_df.loc[metadata_df["is_routed_hard"] == True].copy(),
        chunk_mode="dan1_pred_fuse",
        prompt_builder=pred_prompt_builder,
    )
    pred_smoke_rows, pred_smoke_folds = build_processed_rows(
        parent_groups=pred_smoke_groups,
        metadata_frame=metadata_df.loc[metadata_df["parent_oare_id"].isin(pred_smoke_groups.keys())].copy(),
        chunk_mode="dan1_pred_fuse_smoke",
        prompt_builder=pred_prompt_builder,
    )
    oracle_smoke_rows, oracle_smoke_folds = build_processed_rows(
        parent_groups=oracle_smoke_groups,
        metadata_frame=metadata_df.loc[metadata_df["parent_oare_id"].isin(oracle_smoke_groups.keys())].copy(),
        chunk_mode="dan1_oracle_fuse_smoke",
        prompt_builder=oracle_prompt_builder,
    )

    routed_dir = resolve_path(args.routed_processed_dir, REPO_ROOT / "data" / "processed_taskform_dan1_routed_fold0")
    pred_smoke_dir = resolve_path(args.pred_smoke_processed_dir, REPO_ROOT / "data" / "processed_taskform_dan1_pred_smoke_fold0")
    oracle_smoke_dir = resolve_path(args.oracle_smoke_processed_dir, REPO_ROOT / "data" / "processed_taskform_dan1_oracle_smoke_fold0")
    write_csv(routed_dir / "train_proc.csv", pred_rows)
    write_csv(routed_dir / "folds.csv", pred_folds)
    write_csv(pred_smoke_dir / "train_proc.csv", pred_smoke_rows)
    write_csv(pred_smoke_dir / "folds.csv", pred_smoke_folds)
    write_csv(oracle_smoke_dir / "train_proc.csv", oracle_smoke_rows)
    write_csv(oracle_smoke_dir / "folds.csv", oracle_smoke_folds)

    metadata_path = out_root / "parent_metadata.csv"
    draft_cache_path = out_root / "draft_cache_pred.csv"
    write_csv(metadata_path, metadata_df)
    write_csv(draft_cache_path, draft_cache_df.sort_values(["split", "route_rank", "parent_oare_id", "chunk_index"]))

    routed_meta = metadata_df.loc[metadata_df["is_routed_hard"] == True].copy()
    summary = {
        "line": "dan1",
        "fold": int(args.fold),
        "winner_checkpoint_dir": str(winner_checkpoint_dir),
        "winner_val_diagnostic_csv": str(winner_val_diagnostic_csv),
        "routing": {
            "route_ref_tok_threshold": int(args.route_ref_tok_threshold),
            "route_marker_ref_tok_threshold": int(args.route_marker_ref_tok_threshold),
            "definition": "chunk_total>=4 OR parent_ref_tok>=129 OR (marker_count>=2 AND parent_ref_tok>=96)",
        },
        "prompt": {
            "prompt_style": str(args.prompt_style),
            "sanitize_pred_drafts": bool(args.sanitize_pred_drafts),
            "pred_draft_max_words": int(args.pred_draft_max_words),
            "pred_draft_tail_words": int(args.pred_draft_tail_words),
            "sanitize_oracle_drafts": bool(args.sanitize_oracle_drafts),
            "oracle_draft_max_words": int(args.oracle_draft_max_words),
            "oracle_draft_tail_words": int(args.oracle_draft_tail_words),
        },
        "counts": {
            "all_parents": int(len(metadata_df)),
            "routed_parents": int(len(routed_meta)),
            "routed_train_parents": int((routed_meta["split"] == "train").sum()),
            "routed_val_parents": int((routed_meta["split"] == "val").sum()),
            "routed_train_chunks": int((draft_cache_df["split"] == "train").sum()),
            "routed_val_chunks": int((draft_cache_df["split"] == "val").sum()),
            "smoke_train_parents": int(sum(1 for pid in pred_smoke_groups if pred_smoke_groups[pid][0]["split"] == "train")),
            "smoke_val_parents": int(sum(1 for pid in pred_smoke_groups if pred_smoke_groups[pid][0]["split"] == "val")),
        },
        "artifacts": {
            "metadata_csv": str(metadata_path),
            "draft_cache_csv": str(draft_cache_path),
            "routed_processed_dir": str(routed_dir),
            "pred_smoke_processed_dir": str(pred_smoke_dir),
            "oracle_smoke_processed_dir": str(oracle_smoke_dir),
        },
    }
    write_json(out_root / "summary.json", summary)
    print(f"OK: wrote {metadata_path}")
    print(f"OK: wrote {draft_cache_path}")
    print(f"OK: wrote {routed_dir/'train_proc.csv'}")
    print(f"OK: wrote {pred_smoke_dir/'train_proc.csv'}")
    print(f"OK: wrote {oracle_smoke_dir/'train_proc.csv'}")
    print(f"OK: wrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
