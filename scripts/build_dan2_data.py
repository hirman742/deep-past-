#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fusion_flow_common import (
    build_dan2_prompt,
    build_processed_rows,
    resolve_path,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _group_by_parent(frame: pd.DataFrame) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for parent_id, group in frame.groupby("parent_oare_id", sort=False):
        groups[str(parent_id)] = group.sort_values("chunk_index").to_dict(orient="records")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dan1-root", default="data/taskform_dan1_fold0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-root", default="data/taskform_dan2_fold0")
    ap.add_argument("--smoke-train-parents", type=int, default=256)
    ap.add_argument("--smoke-val-parents", type=int, default=64)
    args = ap.parse_args()

    dan1_root = resolve_path(args.dan1_root, REPO_ROOT / "data" / "taskform_dan1_fold0")
    out_root = resolve_path(args.out_root, REPO_ROOT / "data" / "taskform_dan2_fold0")
    out_root.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(dan1_root / "parent_metadata.csv")
    draft_cache_df = pd.read_csv(dan1_root / "draft_cache_pred.csv")
    routed_meta = metadata_df.loc[metadata_df["is_routed_hard"] == True].copy()

    full_groups = _group_by_parent(draft_cache_df)
    smoke_keep = routed_meta.apply(
        lambda row: (
            (row["split"] == "train" and int(row["route_rank"]) <= int(args.smoke_train_parents))
            or (row["split"] == "val" and int(row["route_rank"]) <= int(args.smoke_val_parents))
        ),
        axis=1,
    )
    smoke_meta = routed_meta.loc[smoke_keep].copy()
    smoke_groups = {parent_id: rows for parent_id, rows in full_groups.items() if parent_id in set(smoke_meta["parent_oare_id"].tolist())}

    hint_rows, hint_folds = build_processed_rows(
        parent_groups=full_groups,
        metadata_frame=routed_meta,
        chunk_mode="dan2_hint_fuse",
        prompt_builder=build_dan2_prompt,
    )
    hint_smoke_rows, hint_smoke_folds = build_processed_rows(
        parent_groups=smoke_groups,
        metadata_frame=smoke_meta,
        chunk_mode="dan2_hint_fuse_smoke",
        prompt_builder=build_dan2_prompt,
    )

    routed_dir = REPO_ROOT / "data" / "processed_taskform_dan2_routed_fold0"
    smoke_dir = REPO_ROOT / "data" / "processed_taskform_dan2_hint_smoke_fold0"
    write_csv(routed_dir / "train_proc.csv", hint_rows)
    write_csv(routed_dir / "folds.csv", hint_folds)
    write_csv(smoke_dir / "train_proc.csv", hint_smoke_rows)
    write_csv(smoke_dir / "folds.csv", hint_smoke_folds)

    summary = {
        "line": "dan2",
        "fold": int(args.fold),
        "dan1_root": str(dan1_root),
        "counts": {
            "routed_parents": int(len(routed_meta)),
            "routed_train_parents": int((routed_meta["split"] == "train").sum()),
            "routed_val_parents": int((routed_meta["split"] == "val").sum()),
            "smoke_train_parents": int((smoke_meta["split"] == "train").sum()),
            "smoke_val_parents": int((smoke_meta["split"] == "val").sum()),
        },
        "artifacts": {
            "metadata_csv": str(dan1_root / "parent_metadata.csv"),
            "draft_cache_csv": str(dan1_root / "draft_cache_pred.csv"),
            "routed_processed_dir": str(routed_dir),
            "hint_smoke_processed_dir": str(smoke_dir),
        },
    }
    write_json(out_root / "summary.json", summary)
    print(f"OK: wrote {routed_dir/'train_proc.csv'}")
    print(f"OK: wrote {smoke_dir/'train_proc.csv'}")
    print(f"OK: wrote {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
