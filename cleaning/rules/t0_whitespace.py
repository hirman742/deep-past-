from __future__ import annotations

import re


_INLINE_WS = re.compile(r"[^\S\n]+", flags=re.UNICODE)


def t0_whitespace_normalize(text: str) -> tuple[str, str]:
    if not text:
        return text, "normalize newlines + fold inline whitespace"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    return text, "normalize newlines + fold inline whitespace"

