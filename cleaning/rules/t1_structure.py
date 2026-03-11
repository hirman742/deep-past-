from __future__ import annotations

import re


_BIG_GAP_RE = re.compile(r"<\s*big(?:[\s_]+)gap\s*>", flags=re.IGNORECASE)
_GAP_RE = re.compile(r"<\s*gap\s*>", flags=re.IGNORECASE)
_BRACE_WS_RE = re.compile(r"\{\s*([^{}\n]*?)\s*\}")


def t1_gap_tag_canonicalize(text: str) -> tuple[str, str]:
    if not text:
        return text, "canonicalize <gap>/<big_gap> tags"
    out = _BIG_GAP_RE.sub("<big_gap>", text)
    out = _GAP_RE.sub("<gap>", out)
    return out, "canonicalize <gap>/<big_gap> tags"


def t1_brace_whitespace_canonicalize(text: str) -> tuple[str, str]:
    if not text:
        return text, "trim inner brace whitespace"
    out = _BRACE_WS_RE.sub(lambda m: "{" + m.group(1).strip() + "}", text)
    return out, "trim inner brace whitespace"
