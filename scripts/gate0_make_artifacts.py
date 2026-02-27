from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


PATTERNS: list[tuple[str, str]] = [
    ("braces", r"\{[^}]*\}"),
    ("brackets", r"\[[^\]]*\]"),
    ("parens", r"\([^)]*\)"),
    ("angle", r"<[^>]*>"),
    ("pipe", r"\|[^|]*\|"),
    ("at_line", r"(^|\n)\s*@\S+"),
    ("dollar_line", r"(^|\n)\s*\$\s*\S+"),
    ("hash_line", r"(^|\n)\s*#\S+"),
    ("percent_code", r"%[a-zA-Z]{1,4}"),
    ("subscript_digits", r"[₀₁₂₃₄₅₆₇₈₉]"),
]


def _codepoint(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def _uname(ch: str) -> str:
    return unicodedata.name(ch, "<UNKNOWN>")


def _display_char(ch: str) -> str:
    if ch == "\t":
        return "\\t"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if unicodedata.category(ch).startswith("C"):
        return f"\\u{ord(ch):04x}"
    return ch


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _snippet_around(text: str, needle: str, max_len: int = 160) -> str:
    if not text:
        return ""
    idx = text.find(needle)
    if idx < 0:
        return _truncate(text, max_len)
    half = max_len // 2
    start = max(0, idx - half)
    end = min(len(text), start + max_len)
    start = max(0, end - max_len)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _build_symbol_inventory_from_counter(
    counter: Counter[str], examples: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    items = []
    for ch, count in counter.items():
        items.append(
            (
                ord(ch),
                ch,
                {
                    "count": int(count),
                    "codepoint": _codepoint(ch),
                    "name": _uname(ch),
                    "examples": examples.get(ch, [])[:5],
                },
            )
        )
    items.sort(key=lambda t: t[0])

    inv: dict[str, Any] = {}
    for _, ch, payload in items:
        inv[ch] = payload
    return inv


def _process_split(
    path: Path, *, label: str, required_cols: list[str], row_id_key: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    compiled = [(pid, rx, re.compile(rx, flags=re.MULTILINE)) for pid, rx in PATTERNS]
    pattern_report: dict[str, Any] = {}
    for pid, rx, _ in compiled:
        pattern_report[pid] = {"regex": rx, "match_count": 0, "examples": []}

    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")

    opened = False
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=enc, errors="replace", newline="") as f:
                opened = True
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"{label} missing header row")
                missing = [c for c in required_cols if c not in set(reader.fieldnames)]
                if missing:
                    raise KeyError(f"{label} missing required columns: {missing}")

                for row in reader:
                    row_id = (row.get(row_id_key) or "").strip()
                    text = row.get("transliteration") or ""
                    counter.update(text)

                    for ch in set(text):
                        ex_list = examples[ch]
                        if len(ex_list) < 5:
                            ex_list.append(
                                {
                                    "row_id": row_id,
                                    "fragment": _truncate(_snippet_around(text, ch, 160), 160),
                                }
                            )

                    for pid, _, cre in compiled:
                        for m in cre.finditer(text):
                            pattern_report[pid]["match_count"] += 1
                            ex_list = pattern_report[pid]["examples"]
                            if len(ex_list) < 10:
                                frag = text[
                                    max(0, m.start() - 50) : min(len(text), m.end() + 50)
                                ]
                                ex_list.append({"row_id": row_id, "fragment": _truncate(frag, 160)})
        except UnicodeError:
            continue
        break

    if not opened:
        raise UnicodeError(f"Unable to decode CSV as utf-8/utf-8-sig: {path}")

    inventory = _build_symbol_inventory_from_counter(counter, examples)
    return inventory, pattern_report


def _write_patterns_md(path: Path, split: str, report: dict[str, Any]) -> None:
    lines: list[str] = [f"# Patterns — {split}", ""]
    for pid, payload in report.items():
        lines.append(f"## {pid}")
        lines.append(f"- regex: `{payload['regex']}`")
        lines.append(f"- match_count: {payload['match_count']}")
        lines.append("")
        exs = payload.get("examples") or []
        if not exs:
            lines.append("_examples: none_")
        else:
            lines.append("### examples")
            for ex in exs:
                rid = ex.get("row_id", "")
                frag = ex.get("fragment", "")
                lines.append(f"- `{rid}`: `{frag}`")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "raw" / "train.csv"
    test_path = DATA_DIR / "raw" / "test.csv"

    train_inv, train_patterns = _process_split(
        train_path,
        label="train.csv",
        required_cols=["oare_id", "transliteration", "translation"],
        row_id_key="oare_id",
    )
    test_inv, test_patterns = _process_split(
        test_path,
        label="test.csv",
        required_cols=["id", "text_id", "line_start", "line_end", "transliteration"],
        row_id_key="id",
    )

    (ARTIFACTS_DIR / "symbol_inventory_train.json").write_text(
        json.dumps(train_inv, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "symbol_inventory_test.json").write_text(
        json.dumps(test_inv, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_patterns_md(ARTIFACTS_DIR / "patterns_train.md", "train", train_patterns)
    _write_patterns_md(ARTIFACTS_DIR / "patterns_test.md", "test", test_patterns)

    train_chars = set(train_inv.keys())
    test_chars = set(test_inv.keys())
    only_in_test = sorted(test_chars - train_chars, key=lambda ch: ord(ch))

    out_lines = ["codepoint\tchar\tname\ttest_count"]
    for ch in only_in_test:
        out_lines.append(
            "\t".join(
                [
                    _codepoint(ch),
                    _display_char(ch),
                    _uname(ch),
                    str(test_inv[ch]["count"]),
                ]
            )
        )
    (ARTIFACTS_DIR / "symbols_only_in_test.txt").write_text(
        "\n".join(out_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
