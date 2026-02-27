from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_lora_modules(model: torch.nn.Module, requested: list[str]) -> list[str]:
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    fallback_map = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
    }
    resolved: list[str] = []
    for module in requested:
        if module in module_names:
            resolved.append(module)
            continue
        mapped = fallback_map.get(module)
        if mapped and mapped in module_names:
            resolved.append(mapped)
    resolved = sorted(set(resolved))
    if not resolved:
        raise ValueError(f"Unable to resolve TAPT LoRA target modules from {requested}")
    return resolved


def _resolve_precision(*, model_name: str, request_fp16: bool, request_bf16: bool) -> tuple[bool, bool]:
    use_fp16 = bool(request_fp16)
    use_bf16 = bool(request_bf16)
    if not torch.cuda.is_available():
        return False, False
    if "t5" in model_name.lower() and use_fp16 and not use_bf16:
        if torch.cuda.is_bf16_supported():
            print("WARN: TAPT T5 + fp16 can be unstable; auto-switched to bf16.")
            return False, True
        print("WARN: TAPT T5 + fp16 can be unstable; fp16 disabled.")
        return False, False
    if use_fp16 and use_bf16:
        return False, True
    return use_fp16, use_bf16


class Seq2SeqListDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _find_text_column(frame: pd.DataFrame) -> str:
    for col in ["source", "transliteration", "text", "input_text"]:
        if col in frame.columns:
            return col
    raise KeyError(f"No text column found in dataframe columns={list(frame.columns)}")


def _sentinel_factory(tokenizer) -> callable:
    vocab = tokenizer.get_vocab()

    def _sentinel(idx: int) -> str:
        token = f"<extra_id_{idx}>"
        if token in vocab:
            return token
        return "<mask>"

    return _sentinel


def _build_denoise_pair(
    text: str,
    *,
    rng: random.Random,
    mask_ratio: float,
    max_span_len: int,
    sentinel_for,
) -> tuple[str, str]:
    words = [x for x in (text or "").split(" ") if x]
    if len(words) <= 4:
        return text, text

    target_mask_tokens = max(1, int(round(len(words) * mask_ratio)))
    mask = [False] * len(words)
    masked = 0
    while masked < target_mask_tokens:
        start = rng.randint(0, len(words) - 1)
        if mask[start]:
            continue
        span_len = rng.randint(1, max(1, max_span_len))
        for i in range(start, min(len(words), start + span_len)):
            if not mask[i]:
                mask[i] = True
                masked += 1
            if masked >= target_mask_tokens:
                break

    source_parts: list[str] = []
    target_parts: list[str] = []
    span_idx = 0
    i = 0
    while i < len(words):
        if not mask[i]:
            source_parts.append(words[i])
            i += 1
            continue
        j = i
        while j < len(words) and mask[j]:
            j += 1
        sentinel = sentinel_for(span_idx)
        source_parts.append(sentinel)
        target_parts.append(sentinel)
        target_parts.extend(words[i:j])
        span_idx += 1
        i = j

    source = " ".join(source_parts).strip()
    target = " ".join(target_parts).strip()
    if not source or not target:
        return text, text
    return source, target


def _to_rows(
    frame: pd.DataFrame,
    *,
    tokenizer,
    source_col: str,
    target_col: str,
    max_source_length: int,
    max_target_length: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in zip(frame[source_col].tolist(), frame[target_col].tolist()):
        model_inputs = tokenizer(
            str(source),
            truncation=True,
            max_length=max_source_length,
            add_special_tokens=True,
        )
        labels = tokenizer(
            text_target=str(target),
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        rows.append(model_inputs)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--corpus-csvs", default="")
    ap.add_argument("--output-run-dir", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    lora_cfg = cfg.get("lora", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    tapt_cfg = cfg.get("tapt", {}) or {}

    default_corpus = [
        _resolve_path(paths_cfg.get("train_csv"), REPO_ROOT / "data" / "interim" / "t0_train.csv"),
        _resolve_path(paths_cfg.get("test_csv"), REPO_ROOT / "data" / "interim" / "t0_test.csv"),
    ]
    corpus_paths: list[Path] = []
    if args.corpus_csvs.strip():
        for item in args.corpus_csvs.split(","):
            item = item.strip()
            if item:
                corpus_paths.append(_resolve_path(item, REPO_ROOT / item))
    else:
        corpus_paths.extend(default_corpus)
        extra_corpus = tapt_cfg.get("oracc_source_csv", "")
        if extra_corpus:
            corpus_paths.append(_resolve_path(extra_corpus, REPO_ROOT / str(extra_corpus)))
    corpus_paths = [x for x in corpus_paths if x.exists()]
    if not corpus_paths:
        raise FileNotFoundError("No TAPT corpus csv found. Use --corpus-csvs to provide files.")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(tapt_cfg.get("max_source_length", model_cfg.get("max_source_length", 256)))
    max_target_length = int(tapt_cfg.get("max_target_length", model_cfg.get("max_target_length", 192)))

    run_root = _resolve_path(
        args.output_run_dir or str(tapt_cfg.get("run_dir", "")),
        REPO_ROOT / "runs" / "TAPT_MT5",
    )
    run_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentinel_for = _sentinel_factory(tokenizer)
    rng = random.Random(seed)

    all_texts: list[str] = []
    for csv_path in corpus_paths:
        frame = pd.read_csv(csv_path)
        text_col = _find_text_column(frame)
        all_texts.extend(frame[text_col].fillna("").astype(str).tolist())
    all_texts = [x.strip() for x in all_texts if x and x.strip()]
    all_texts = list(dict.fromkeys(all_texts))
    if args.max_rows > 0:
        all_texts = all_texts[: args.max_rows]
    if len(all_texts) < 32:
        raise ValueError("TAPT corpus too small after filtering")

    mask_ratio = float(tapt_cfg.get("mask_ratio", 0.15))
    max_span_len = int(tapt_cfg.get("max_span_length", 3))
    pairs = [
        _build_denoise_pair(
            text,
            rng=rng,
            mask_ratio=mask_ratio,
            max_span_len=max_span_len,
            sentinel_for=sentinel_for,
        )
        for text in all_texts
    ]
    frame = pd.DataFrame({"source": [x[0] for x in pairs], "target": [x[1] for x in pairs]})
    frame = frame[(frame["source"].str.strip() != "") & (frame["target"].str.strip() != "")].reset_index(drop=True)

    val_ratio = float(tapt_cfg.get("val_ratio", 0.02))
    val_rows = int(round(len(frame) * val_ratio))
    val_rows = max(0, min(val_rows, len(frame) // 5))
    if val_rows > 0:
        val_frame = frame.iloc[:val_rows].reset_index(drop=True)
        train_frame = frame.iloc[val_rows:].reset_index(drop=True)
    else:
        val_frame = frame.iloc[0:0].copy()
        train_frame = frame.copy()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    requested_modules = [str(x) for x in (lora_cfg.get("target_modules") or ["q_proj", "v_proj"])]
    resolved_modules = _resolve_lora_modules(model, requested_modules)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=int(tapt_cfg.get("lora_r", lora_cfg.get("r", 4))),
        lora_alpha=int(tapt_cfg.get("lora_alpha", lora_cfg.get("alpha", 4))),
        lora_dropout=float(tapt_cfg.get("lora_dropout", lora_cfg.get("dropout", 0.0))),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=resolved_modules,
    )
    model = get_peft_model(model, peft_config)
    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = _to_rows(
        train_frame,
        tokenizer=tokenizer,
        source_col="source",
        target_col="target",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    val_rows_data = _to_rows(
        val_frame,
        tokenizer=tokenizer,
        source_col="source",
        target_col="target",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    train_dataset = Seq2SeqListDataset(train_rows)
    eval_dataset = Seq2SeqListDataset(val_rows_data) if val_rows_data else None
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    eval_steps = max(1, int(tapt_cfg.get("eval_steps", 100)))
    max_steps = int(tapt_cfg.get("max_steps", 400))
    use_fp16, use_bf16 = _resolve_precision(
        model_name=model_name,
        request_fp16=bool(tapt_cfg.get("fp16", train_cfg.get("fp16", False))),
        request_bf16=bool(tapt_cfg.get("bf16", train_cfg.get("bf16", True))),
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_root),
        per_device_train_batch_size=int(tapt_cfg.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(tapt_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(tapt_cfg.get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps", 8))),
        learning_rate=float(tapt_cfg.get("learning_rate", train_cfg.get("learning_rate", 2e-4))),
        warmup_ratio=float(tapt_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tapt_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tapt_cfg.get("lr_scheduler_type", "cosine")),
        num_train_epochs=float(tapt_cfg.get("num_train_epochs", 1.0)),
        max_steps=max_steps,
        fp16=bool(use_fp16),
        bf16=bool(use_bf16),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=max(10, eval_steps // 2),
        load_best_model_at_end=bool(eval_dataset is not None),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        report_to="none",
        seed=seed,
        do_train=True,
        do_eval=bool(eval_dataset is not None),
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate() if eval_dataset is not None else {}

    best_dir = run_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)

    summary = {
        "config_path": str(cfg_path),
        "run_dir": str(run_root),
        "best_model_dir": str(best_dir),
        "corpus_csvs": [str(x) for x in corpus_paths],
        "raw_text_rows": int(len(all_texts)),
        "tapt_pairs_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "eval_rows": int(len(val_frame)),
        "mask_ratio": float(mask_ratio),
        "max_span_length": int(max_span_len),
        "max_source_length": int(max_source_length),
        "max_target_length": int(max_target_length),
        "resolved_lora_modules": resolved_modules,
        "precision": {"fp16": bool(use_fp16), "bf16": bool(use_bf16)},
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }
    summary_path = run_root / "tapt_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: best TAPT adapter saved at {best_dir}")
    print(f"OK: summary saved at {summary_path}")


if __name__ == "__main__":
    main()
