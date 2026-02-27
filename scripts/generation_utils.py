from __future__ import annotations

import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_generation_settings(
    *,
    model_cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
) -> dict[str, Any]:
    max_target_length = int(model_cfg.get("max_target_length", 192))
    settings = {
        "num_beams": int(gen_cfg.get("num_beams", 4)),
        "length_penalty": float(gen_cfg.get("length_penalty", 1.0)),
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", max_target_length)),
        "min_new_tokens": int(gen_cfg.get("min_new_tokens", 0)),
        "no_repeat_ngram_size": int(gen_cfg.get("no_repeat_ngram_size", 0)),
        "suppress_extra_ids": bool(gen_cfg.get("suppress_extra_ids", False)),
        "bad_tokens_regex": str(gen_cfg.get("bad_tokens_regex", r"<extra_id_\d+>")),
    }
    return settings


def build_bad_words_ids(
    *,
    tokenizer,
    suppress_extra_ids: bool,
    bad_tokens_regex: str,
) -> list[list[int]] | None:
    if not suppress_extra_ids:
        return None
    pattern = re.compile(bad_tokens_regex)
    blocked_ids: set[int] = set()
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if pattern.search(token):
            blocked_ids.add(int(token_id))
    for token in tokenizer.all_special_tokens:
        if pattern.search(token):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and int(token_id) >= 0:
                blocked_ids.add(int(token_id))

    protected = {
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
    }
    filtered = sorted(x for x in blocked_ids if x is not None and x >= 0 and x not in protected)
    if not filtered:
        return None
    return [[x] for x in filtered]


def build_generate_kwargs(
    *,
    num_beams: int,
    length_penalty: float,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    no_repeat_ngram_size: int = 0,
    bad_words_ids: list[list[int]] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_beams": int(num_beams),
        "length_penalty": float(length_penalty),
        "max_new_tokens": int(max_new_tokens),
    }
    if int(min_new_tokens) > 0:
        kwargs["min_new_tokens"] = int(min_new_tokens)
    if int(no_repeat_ngram_size) > 0:
        kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if bad_words_ids:
        kwargs["bad_words_ids"] = bad_words_ids
    return kwargs


def normalize_task_prefix(prefix: str | None) -> str:
    if not prefix:
        return ""
    normalized = str(prefix).strip()
    if not normalized:
        return ""
    if not normalized.endswith(" "):
        normalized += " "
    return normalized


def apply_task_prefix(text: str, task_prefix: str) -> str:
    if not task_prefix:
        return text
    value = text if isinstance(text, str) else ""
    if value.startswith(task_prefix):
        return value
    return f"{task_prefix}{value}"
