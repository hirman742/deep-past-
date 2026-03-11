from __future__ import annotations

from typing import Any

from .rules.t0_unknown import t0_unknown_visible_char_alert
from .rules.t0_unicode import t0_remove_invisible, t0_unicode_nfc
from .rules.t1_structure import t1_brace_whitespace_canonicalize, t1_gap_tag_canonicalize
from .rules.t0_whitespace import t0_whitespace_normalize


def normalize_source(text: str, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rules_cfg = (config or {}).get("rules", {}) or {}

    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    edit_log: list[dict[str, Any]] = []

    def apply_rule(rule_id: str, enabled_key: str, fn, default_enabled: bool = True):
        nonlocal text
        if not bool(rules_cfg.get(enabled_key, default_enabled)):
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

    def apply_log_rule(enabled_key: str, fn, default_enabled: bool = True):
        if not bool(rules_cfg.get(enabled_key, default_enabled)):
            return
        edit_log.extend(fn(text, config or {}))

    apply_rule("t0_unicode_nfc", "t0_unicode_nfc", t0_unicode_nfc)
    apply_rule("t0_remove_invisible", "t0_remove_invisible", t0_remove_invisible)
    apply_rule("t0_whitespace_normalize", "t0_whitespace_normalize", t0_whitespace_normalize)
    apply_log_rule("t0_unknown_visible_char_alert", t0_unknown_visible_char_alert)
    apply_rule(
        "t1_gap_tag_canonicalize",
        "t1_gap_tag_canonicalize",
        t1_gap_tag_canonicalize,
        default_enabled=False,
    )
    apply_rule(
        "t1_brace_whitespace_canonicalize",
        "t1_brace_whitespace_canonicalize",
        t1_brace_whitespace_canonicalize,
        default_enabled=False,
    )

    return text, edit_log
