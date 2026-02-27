from __future__ import annotations

from typing import Any

from .rules.t0_unicode import t0_remove_invisible, t0_unicode_nfc
from .rules.t0_whitespace import t0_whitespace_normalize


def normalize_source(text: str, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rules_cfg = (config or {}).get("rules", {}) or {}

    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    edit_log: list[dict[str, Any]] = []

    def apply_rule(rule_id: str, enabled_key: str, fn):
        nonlocal text
        if not bool(rules_cfg.get(enabled_key, True)):
            return
        before = text
        after, note = fn(text)
        if after != before:
            edit_log.append(
                {
                    "rule_id": rule_id,
                    "before": before,
                    "after": after,
                    "note": note,
                }
            )
            text = after

    apply_rule("t0_unicode_nfc", "t0_unicode_nfc", t0_unicode_nfc)
    apply_rule("t0_remove_invisible", "t0_remove_invisible", t0_remove_invisible)
    apply_rule("t0_whitespace_normalize", "t0_whitespace_normalize", t0_whitespace_normalize)

    return text, edit_log

