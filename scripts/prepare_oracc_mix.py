from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cleaning.normalize import normalize_source  # noqa: E402
from generation_utils import apply_task_prefix, normalize_task_prefix  # noqa: E402


INLINE_WS_RE = re.compile(r"[^\S\n]+", flags=re.UNICODE)


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_cleaning_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "cleaning" / "configs" / "cleaning.t0.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _normalize_text(
    text: str,
    *,
    apply_t0: bool,
    cleaning_cfg: dict[str, Any],
    strip_text: bool,
    fold_ws: bool,
    lowercase: bool,
    task_prefix: str,
) -> str:
    value = text if isinstance(text, str) else ""
    if apply_t0:
        value, _ = normalize_source(value, config=cleaning_cfg)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if fold_ws:
        value = INLINE_WS_RE.sub(" ", value)
    if strip_text:
        value = value.strip()
    if lowercase:
        value = value.lower()
    value = apply_task_prefix(value, task_prefix)
    return value


def _detect_columns(frame: pd.DataFrame) -> tuple[str, str]:
    source_candidates = ["source", "transliteration", "akkadian", "input_text", "text"]
    target_candidates = ["target", "translation", "english", "output_text"]
    src_col = next((c for c in source_candidates if c in frame.columns), None)
    tgt_col = next((c for c in target_candidates if c in frame.columns), None)
    if src_col is None or tgt_col is None:
        raise KeyError(
            f"Unable to detect source/target columns. Found: {list(frame.columns)}"
        )
    return src_col, tgt_col


def _exact_dedupe_mask(values: pd.Series, existing: set[str]) -> pd.Series:
    return values.map(lambda x: str(x) not in existing)


def _build_signature(text: str) -> tuple[str, int]:
    value = (text or "").strip().lower()
    prefix = value[:24]
    length_bin = len(value) // 40
    return prefix, length_bin


def _jaccard_char_ngrams(a: str, b: str, n: int = 4) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) < n or len(b) < n:
        return 0.0
    a_set = {a[i : i + n] for i in range(len(a) - n + 1)}
    b_set = {b[i : i + n] for i in range(len(b) - n + 1)}
    if not a_set or not b_set:
        return 0.0
    return float(len(a_set & b_set)) / float(len(a_set | b_set))


def _high_similarity_filter(
    values: list[str],
    *,
    reference_values: list[str],
    threshold: float,
) -> list[bool]:
    buckets: dict[tuple[str, int], list[str]] = {}
    for ref in reference_values:
        buckets.setdefault(_build_signature(ref), []).append(ref)

    keep_mask: list[bool] = []
    for value in values:
        sig = _build_signature(value)
        candidates = buckets.get(sig, [])
        should_keep = True
        for candidate in candidates:
            if _jaccard_char_ngrams(value, candidate, n=4) >= threshold:
                should_keep = False
                break
        keep_mask.append(should_keep)
    return keep_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--oracc-csv", default="")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--output-train", default="")
    ap.add_argument("--audit-json", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--similarity-threshold", type=float, default=0.92)
    ap.add_argument("--disable-similarity-filter", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)
    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}

    train_path = _resolve_path(paths_cfg.get("train_csv"), REPO_ROOT / "data" / "interim" / "t0_train.csv")
    test_path = _resolve_path(paths_cfg.get("test_csv"), REPO_ROOT / "data" / "interim" / "t0_test.csv")
    default_oracc = REPO_ROOT / "data" / "external" / "oracc_parallel.csv"
    oracc_path = _resolve_path(args.oracc_csv, default_oracc)
    if not oracc_path.exists():
        raise FileNotFoundError(f"Missing ORACC csv: {oracc_path}")

    output_train = _resolve_path(args.output_train, REPO_ROOT / "data" / "interim" / "oracc_mix_train.csv")
    audit_json = _resolve_path(args.audit_json, REPO_ROOT / "runs" / "oracc_mix_audit.json")

    apply_t0 = bool(preprocess_cfg.get("apply_t0_normalize", False))
    strip_text = bool(preprocess_cfg.get("strip_text", True))
    fold_ws = bool(preprocess_cfg.get("fold_inline_whitespace", True))
    lowercase_source = bool(preprocess_cfg.get("lowercase_source", False))
    lowercase_target = bool(preprocess_cfg.get("lowercase_target", False))
    task_prefix = normalize_task_prefix(preprocess_cfg.get("task_prefix", ""))
    cleaning_cfg = _load_cleaning_config()

    comp_train = pd.read_csv(train_path)
    comp_test = pd.read_csv(test_path)
    oracc = pd.read_csv(oracc_path)

    if not {"oare_id", "transliteration", "translation"}.issubset(comp_train.columns):
        raise KeyError("competition train must contain oare_id/transliteration/translation")
    if "transliteration" not in comp_test.columns:
        raise KeyError("competition test must contain transliteration")

    src_col, tgt_col = _detect_columns(oracc)
    ext = oracc[[src_col, tgt_col]].copy()
    ext.columns = ["transliteration", "translation"]
    ext = ext.fillna("")
    ext = ext[(ext["transliteration"].str.strip() != "") & (ext["translation"].str.strip() != "")].reset_index(drop=True)
    ext["oare_id"] = [f"oracc_{i:08d}" for i in range(len(ext))]

    comp_train_norm_src = comp_train["transliteration"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
            task_prefix=task_prefix,
        )
    )
    comp_test_norm_src = comp_test["transliteration"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
            task_prefix=task_prefix,
        )
    )
    ext_norm_src = ext["transliteration"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_source,
            task_prefix=task_prefix,
        )
    )
    ext_norm_tgt = ext["translation"].fillna("").astype(str).map(
        lambda s: _normalize_text(
            s,
            apply_t0=apply_t0,
            cleaning_cfg=cleaning_cfg,
            strip_text=strip_text,
            fold_ws=fold_ws,
            lowercase=lowercase_target,
            task_prefix="",
        )
    )

    ext["transliteration"] = ext_norm_src
    ext["translation"] = ext_norm_tgt
    ext = ext[(ext["transliteration"].str.strip() != "") & (ext["translation"].str.strip() != "")].reset_index(drop=True)

    comp_reference = set(comp_train_norm_src.tolist()) | set(comp_test_norm_src.tolist())
    keep_exact = _exact_dedupe_mask(ext["transliteration"], comp_reference)
    ext_exact = ext[keep_exact].reset_index(drop=True)

    similarity_removed = 0
    if args.disable_similarity_filter:
        ext_filtered = ext_exact
    else:
        sim_mask = _high_similarity_filter(
            ext_exact["transliteration"].tolist(),
            reference_values=list(comp_reference),
            threshold=float(args.similarity_threshold),
        )
        ext_filtered = ext_exact[pd.Series(sim_mask)].reset_index(drop=True)
        similarity_removed = int(len(ext_exact) - len(ext_filtered))

    target_external_rows = int(round(len(comp_train) * max(0.0, args.ratio)))
    target_external_rows = min(target_external_rows, len(ext_filtered))
    if target_external_rows <= 0:
        sampled_ext = ext_filtered.iloc[0:0].copy()
    else:
        sampled_ext = ext_filtered.sample(n=target_external_rows, random_state=args.seed).reset_index(drop=True)

    comp_train_out = comp_train[["oare_id", "transliteration", "translation"]].copy()
    comp_train_out["origin"] = "competition"
    sampled_ext_out = sampled_ext[["oare_id", "transliteration", "translation"]].copy()
    sampled_ext_out["origin"] = "oracc"
    mixed = pd.concat([comp_train_out, sampled_ext_out], axis=0, ignore_index=True)

    output_train.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(output_train, index=False)

    audit = {
        "config_path": str(cfg_path),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "oracc_csv": str(oracc_path),
        "rows_competition_train": int(len(comp_train_out)),
        "rows_oracc_input": int(len(oracc)),
        "rows_oracc_after_basic_clean": int(len(ext)),
        "rows_oracc_after_exact_dedupe": int(len(ext_exact)),
        "rows_oracc_removed_similarity": int(similarity_removed),
        "rows_oracc_sampled": int(len(sampled_ext_out)),
        "rows_mixed_output": int(len(mixed)),
        "ratio_requested": float(args.ratio),
        "ratio_realized": float(len(sampled_ext_out)) / float(max(1, len(comp_train_out))),
        "similarity_filter_enabled": bool(not args.disable_similarity_filter),
        "similarity_threshold": float(args.similarity_threshold),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {output_train}")
    print(f"OK: wrote {audit_json}")
    print(
        "INFO: competition/oracc/mixed="
        f"{len(comp_train_out)}/{len(sampled_ext_out)}/{len(mixed)} "
        f"(ratio={audit['ratio_realized']:.3f})"
    )


if __name__ == "__main__":
    main()
