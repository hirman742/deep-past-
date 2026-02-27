from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "cleaning" / "configs" / "cleaning.t0.yaml"

sys.path.insert(0, str(REPO_ROOT))
from cleaning.normalize import normalize_source  # noqa: E402


def _open_text(path: Path, mode: str):
    return path.open(mode, encoding="utf-8", errors="replace", newline="")


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _input_path(split: str) -> Path:
    return REPO_ROOT / "data" / "raw" / f"{split}.csv"


def _output_path(split: str) -> Path:
    return REPO_ROOT / "data" / "interim" / f"t0_{split}.csv"


def _edit_log_path(split: str) -> Path:
    return REPO_ROOT / "data" / "interim" / f"t0_{split}_edit_log.jsonl"


def _id_key(split: str) -> str:
    return "oare_id" if split == "train" else "id"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()

    split: str = args.split
    in_path = _input_path(split)
    out_path = _output_path(split)
    log_path = _edit_log_path(split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing required input: {in_path}")

    config = _load_config()
    id_key = _id_key(split)

    with _open_text(in_path, "r") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{in_path} missing header row")
        if "transliteration" not in reader.fieldnames:
            raise KeyError(f"{in_path} missing required column: transliteration")
        if id_key not in reader.fieldnames:
            raise KeyError(f"{in_path} missing required id column: {id_key}")

        with _open_text(out_path, "w") as fout, _open_text(log_path, "w") as flog:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                sample_id = (row.get(id_key) or "").strip()
                before = row.get("transliteration") or ""
                after, edits = normalize_source(before, config=config)
                row["transliteration"] = after
                writer.writerow(row)

                for e in edits:
                    payload = {"split": split, "sample_id": sample_id, **e}
                    flog.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"OK: wrote {out_path}")
    print(f"OK: wrote {log_path}")


if __name__ == "__main__":
    main()
