from __future__ import annotations

import unicodedata


def t0_unicode_nfc(text: str) -> tuple[str, str]:
    return unicodedata.normalize("NFC", text), "unicodedata.normalize('NFC')"


_INVISIBLE = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
}


def t0_remove_invisible(text: str) -> tuple[str, str]:
    if not text:
        return text, "remove ZWSP/ZWNJ/ZWJ/BOM"
    out = "".join(ch for ch in text if ch not in _INVISIBLE)
    return out, "remove ZWSP/ZWNJ/ZWJ/BOM"

