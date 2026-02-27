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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from generation_utils import build_bad_words_ids, build_generate_kwargs, resolve_generation_settings
from metrics_utils import build_metric_signatures, compute_translation_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


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


class Seq2SeqListDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _build_dataset_rows(
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


def _compute_eval_steps(
    train_size: int,
    *,
    per_device_batch: int,
    grad_accum: int,
    eval_fraction: float,
) -> int:
    steps_per_epoch = math.ceil(train_size / max(1, per_device_batch * grad_accum))
    return max(1, int(math.ceil(steps_per_epoch * eval_fraction)))


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
        raise ValueError(f"Unable to resolve LoRA target modules from {requested}")
    return resolved


def _trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--init-adapter-dir", default="")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    lora_cfg = cfg.get("lora", {}) or {}
    train_cfg = cfg.get("training", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    tapt_cfg = cfg.get("tapt", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    train_path = processed_dir / "train_proc.csv"
    folds_path = processed_dir / "folds.csv"
    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    run_dir = run_root.parent / f"{run_root.name}_fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    folds_df = pd.read_csv(folds_path)
    merged = train_df.merge(folds_df[["oare_id", "fold"]], on="oare_id", how="inner")

    train_split = merged[merged["fold"] != args.fold].reset_index(drop=True)
    val_split = merged[merged["fold"] == args.fold].reset_index(drop=True)
    if train_split.empty or val_split.empty:
        raise ValueError(f"Fold {args.fold} produced empty train/val split")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))
    max_target_length = int(model_cfg.get("max_target_length", 192))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    requested_modules = [str(x) for x in (lora_cfg.get("target_modules") or ["q_proj", "v_proj"])]
    resolved_modules = _resolve_lora_modules(base_model, requested_modules)

    init_adapter_cfg = str(tapt_cfg.get("init_adapter_dir", ""))
    init_adapter_dir = _resolve_path(
        args.init_adapter_dir or init_adapter_cfg,
        run_dir / "missing_init_adapter_dir",
    )
    used_init_adapter = bool(args.init_adapter_dir or init_adapter_cfg)
    if used_init_adapter:
        if not init_adapter_dir.exists():
            raise FileNotFoundError(f"Missing init adapter dir: {init_adapter_dir}")
        model = PeftModel.from_pretrained(base_model, init_adapter_dir, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=int(lora_cfg.get("r", 4)),
            lora_alpha=int(lora_cfg.get("alpha", 4)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=resolved_modules,
        )
        model = get_peft_model(base_model, peft_config)

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = _build_dataset_rows(
        train_split,
        tokenizer=tokenizer,
        source_col="source",
        target_col="target",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    val_rows = _build_dataset_rows(
        val_split,
        tokenizer=tokenizer,
        source_col="source",
        target_col="target",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    train_dataset = Seq2SeqListDataset(train_rows)
    val_dataset = Seq2SeqListDataset(val_rows)

    per_device_train_batch_size = int(train_cfg.get("per_device_train_batch_size", 4))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    eval_fraction = float(train_cfg.get("eval_fraction_of_epoch", 0.25))
    eval_steps = _compute_eval_steps(
        len(train_dataset),
        per_device_batch=per_device_train_batch_size,
        grad_accum=grad_accum,
        eval_fraction=eval_fraction,
    )

    requested_fp16 = bool(train_cfg.get("fp16", True))
    requested_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = requested_fp16
    use_bf16 = requested_bf16

    if not torch.cuda.is_available():
        use_fp16 = False
        use_bf16 = False
    elif "t5" in model_name.lower() and use_fp16 and not use_bf16:
        if torch.cuda.is_bf16_supported():
            use_fp16 = False
            use_bf16 = True
            print("WARN: T5 + fp16 can be unstable; auto-switched to bf16.")
        else:
            use_fp16 = False
            print("WARN: T5 + fp16 can be unstable; fp16 disabled.")

    if use_fp16 and use_bf16:
        use_fp16 = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None,
    )

    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )
    trainer_generate_kwargs = build_generate_kwargs(
        num_beams=int(generation_settings["num_beams"]),
        length_penalty=float(generation_settings["length_penalty"]),
        max_new_tokens=int(generation_settings["max_new_tokens"]),
        min_new_tokens=int(generation_settings["min_new_tokens"]),
        no_repeat_ngram_size=int(generation_settings["no_repeat_ngram_size"]),
        bad_words_ids=bad_words_ids,
    )
    model.generation_config.num_beams = int(generation_settings["num_beams"])
    model.generation_config.length_penalty = float(generation_settings["length_penalty"])
    model.generation_config.max_new_tokens = int(generation_settings["max_new_tokens"])
    model.generation_config.no_repeat_ngram_size = int(generation_settings["no_repeat_ngram_size"])
    if int(generation_settings["min_new_tokens"]) > 0:
        model.generation_config.min_new_tokens = int(generation_settings["min_new_tokens"])
    if bad_words_ids:
        model.generation_config.bad_words_ids = bad_words_ids

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions).astype(np.int64)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_texts = [x.strip() for x in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        ref_texts = [x.strip() for x in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return compute_translation_metrics(predictions=pred_texts, references=ref_texts)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metric_for_best_model = str(train_cfg.get("metric_for_best_model", "geom"))
    greater_is_better = bool(train_cfg.get("greater_is_better", True))
    callbacks = []
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(train_cfg.get("early_stopping_threshold", 0.0)),
            )
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        max_steps=int(args.max_steps),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        report_to="none",
        seed=seed,
        do_train=True,
        do_eval=True,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer._gen_kwargs = dict(trainer_generate_kwargs)

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)

    prediction_output = trainer.predict(val_dataset)
    val_predictions = prediction_output.predictions
    if isinstance(val_predictions, tuple):
        val_predictions = val_predictions[0]
    val_predictions = np.asarray(val_predictions)
    if val_predictions.ndim == 3:
        val_predictions = np.argmax(val_predictions, axis=-1)
    val_predictions = np.where(val_predictions < 0, tokenizer.pad_token_id, val_predictions).astype(np.int64)
    decoded_predictions = tokenizer.batch_decode(val_predictions, skip_special_tokens=True)

    val_pred_df = pd.DataFrame(
        {
            "oare_id": val_split["oare_id"].astype(str).tolist(),
            "reference": val_split["target"].astype(str).tolist(),
            "prediction": [x.strip() for x in decoded_predictions],
        }
    )
    val_pred_path = run_dir / "val_predictions.csv"
    val_pred_df.to_csv(val_pred_path, index=False)

    trainable, total = _trainable_params(model)
    summary = {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "fold": int(args.fold),
        "resolved_lora_modules": resolved_modules,
        "init_adapter_dir": str(init_adapter_dir) if used_init_adapter else "",
        "train_rows": int(len(train_split)),
        "val_rows": int(len(val_split)),
        "eval_steps": int(eval_steps),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_ratio_pct": 100.0 * float(trainable) / float(total),
        "precision": {"fp16": bool(use_fp16), "bf16": bool(use_bf16)},
        "generation_settings": generation_settings,
        "trainer_generate_kwargs": trainer_generate_kwargs,
        "suppressed_bad_word_ids_count": int(len(bad_words_ids or [])),
        "early_stopping_patience": int(early_stopping_patience),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "linear")),
        "metrics_scale": {
            "bleu": "0-100 (sacrebleu.score)",
            "chrfpp": "0-100 (sacrebleu.score)",
            "geom": "sqrt(bleu * chrfpp), same 0-100 scale family",
            "bleu_01": "bleu/100",
            "chrfpp_01": "chrfpp/100",
            "geom_01": "sqrt(bleu_01 * chrfpp_01)",
        },
        "metric_signatures": build_metric_signatures(),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0,
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    resolved_cfg_path = run_dir / "resolved_config.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"OK: best model saved at {best_dir}")
    print(f"OK: val predictions saved at {val_pred_path}")
    print(f"OK: summary saved at {summary_path}")
    print(
        "INFO: eval geom/bleu/chrfpp="
        f"{eval_result.get('eval_geom', 0.0):.4f}/"
        f"{eval_result.get('eval_bleu', 0.0):.4f}/"
        f"{eval_result.get('eval_chrfpp', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
