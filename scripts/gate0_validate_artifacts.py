from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")


def _validate_inventory(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a JSON object keyed by symbol")
    for symbol, payload in data.items():
        if not isinstance(symbol, str) or len(symbol) == 0:
            raise ValueError(f"{path}: invalid symbol key")
        if not isinstance(payload, dict):
            raise TypeError(f"{path}: symbol payload must be an object")
        for k in ("count", "codepoint", "name", "examples"):
            if k not in payload:
                raise KeyError(f"{path}: missing key {k} for symbol {symbol!r}")
        if not isinstance(payload["examples"], list):
            raise TypeError(f"{path}: examples must be a list for symbol {symbol!r}")


def _validate_patterns_md(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    headings = [ln for ln in text.splitlines() if ln.startswith("## ")]
    if len(headings) < 10:
        raise ValueError(f"{path}: expected >= 10 pattern headings, got {len(headings)}")


def _validate_symbols_only(path: Path) -> None:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty")
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 4:
            raise ValueError(f"{path}: expected >= 4 tab-separated fields, got: {ln!r}")


def main() -> None:
    paths = {
        "train_inv": ARTIFACTS_DIR / "symbol_inventory_train.json",
        "test_inv": ARTIFACTS_DIR / "symbol_inventory_test.json",
        "patterns_train": ARTIFACTS_DIR / "patterns_train.md",
        "patterns_test": ARTIFACTS_DIR / "patterns_test.md",
        "only_in_test": ARTIFACTS_DIR / "symbols_only_in_test.txt",
    }

    for p in paths.values():
        _require(p)

    _validate_inventory(paths["train_inv"])
    _validate_inventory(paths["test_inv"])
    _validate_patterns_md(paths["patterns_train"])
    _validate_patterns_md(paths["patterns_test"])
    _validate_symbols_only(paths["only_in_test"])

    print("OK: Gate0 artifacts validated.")


if __name__ == "__main__":
    main()

