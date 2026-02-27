from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "cleaning" / "configs" / "cleaning.t0.yaml"


def _load_keep_n() -> int:
    if not CONFIG_PATH.exists():
        return 3
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return int(((cfg.get("logging") or {}).get("keep_examples_per_rule") or 3))


def _input_path(split: str) -> Path:
    return REPO_ROOT / "data" / "interim" / f"t0_{split}_edit_log.jsonl"


def _output_path(split: str) -> Path:
    return REPO_ROOT / "runs" / f"GATE0_T0_{split}" / "hit_stats.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()

    split: str = args.split
    in_path = _input_path(split)
    out_path = _output_path(split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing required input: {in_path}")

    keep_n = _load_keep_n()
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0

    for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        total += 1
        rec = json.loads(line)
        rid = rec.get("rule_id") or "<missing_rule_id>"
        counts[rid] += 1
        ex_list = examples[rid]
        if len(ex_list) < keep_n:
            ex_list.append(
                {
                    "sample_id": rec.get("sample_id", ""),
                    "before": rec.get("before", ""),
                    "after": rec.get("after", ""),
                    "note": rec.get("note", ""),
                }
            )

    hit_stats = {
        "split": split,
        "total_edits": total,
        "rules": [
            {"rule_id": rid, "count": counts[rid], "examples": examples.get(rid, [])}
            for rid in sorted(counts.keys())
        ],
    }

    out_path.write_text(json.dumps(hit_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()

