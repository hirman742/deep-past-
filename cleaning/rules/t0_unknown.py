from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


_NON_VISIBLE = {" ", "\t", "\n", "\r"}


def _is_visible_char(ch: str) -> bool:
    if ch in _NON_VISIBLE:
        return False
    category = unicodedata.category(ch)
    return not category.startswith("C")


def _load_known_visible_charset(config: dict[str, Any]) -> set[str]:
    cache_key = "_known_visible_charset_cache"
    if cache_key in config:
        return set(config[cache_key])

    path_str = (config or {}).get("known_visible_charset_train")
    if not path_str:
        config[cache_key] = set()
        return set()

    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    data = json.loads(path.read_text(encoding="utf-8"))
    charset = {ch for ch in data.keys() if _is_visible_char(ch)}
    config[cache_key] = charset
    return charset


def t0_unknown_visible_char_alert(text: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not text:
        return []

    known = _load_known_visible_charset(config)
    if not known:
        return []

    hits: dict[str, dict[str, Any]] = {}
    for idx, ch in enumerate(text):
        if not _is_visible_char(ch) or ch in known:
            continue
        start = max(0, idx - 20)
        end = min(len(text), idx + 21)
        payload = hits.setdefault(
            ch,
            {
                "count": 0,
                "pos": idx,
                "context": text[start:end],
            },
        )
        payload["count"] += 1

    out: list[dict[str, Any]] = []
    for ch, payload in hits.items():
        out.append(
            {
                "rule_id": "t0_unknown_visible_char_alert",
                "before": ch,
                "after": ch,
                "note": (
                    f"unknown visible char {ch!r} ({unicodedata.name(ch, 'UNKNOWN')}, "
                    f"U+{ord(ch):04X}) count={payload['count']}"
                ),
                "pos": payload["pos"],
                "context": payload["context"],
                "severity": "warn",
            }
        )
    return out
