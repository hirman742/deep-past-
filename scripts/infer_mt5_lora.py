from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generation_utils import build_bad_words_ids, build_generate_kwargs, resolve_generation_settings


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


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--submission-path", default="")
    ap.add_argument("--predict-batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=-1)
    ap.add_argument("--length-penalty", type=float, default=-1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=-1)
    ap.add_argument("--min-new-tokens", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=-1)
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    paths_cfg = cfg.get("paths", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    test_path = processed_dir / "test_proc.csv"
    sample_submission_path = _resolve_path(
        paths_cfg.get("sample_submission_csv"), REPO_ROOT / "data" / "raw" / "sample_submission.csv"
    )

    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    run_dir = run_root.parent / f"{run_root.name}_fold{args.fold}"
    checkpoint_dir = _resolve_path(args.checkpoint_dir, run_dir / "best_model")
    submission_path = _resolve_path(args.submission_path, run_dir / "submission.csv")
    prediction_path = run_dir / "test_predictions.csv"

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))

    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_submission_path)
    if "id" not in test_df.columns:
        raise KeyError("test_proc.csv missing id column")
    if "source" not in test_df.columns:
        raise KeyError("test_proc.csv missing source column")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    if args.num_beams > 0:
        generation_settings["num_beams"] = int(args.num_beams)
    if args.length_penalty > 0:
        generation_settings["length_penalty"] = float(args.length_penalty)
    if args.no_repeat_ngram_size >= 0:
        generation_settings["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    if args.min_new_tokens >= 0:
        generation_settings["min_new_tokens"] = int(args.min_new_tokens)
    if args.max_new_tokens >= 0:
        generation_settings["max_new_tokens"] = int(args.max_new_tokens)

    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=bool(generation_settings["suppress_extra_ids"]),
        bad_tokens_regex=str(generation_settings["bad_tokens_regex"]),
    )
    generate_kwargs = build_generate_kwargs(
        num_beams=int(generation_settings["num_beams"]),
        length_penalty=float(generation_settings["length_penalty"]),
        max_new_tokens=int(generation_settings["max_new_tokens"]),
        min_new_tokens=int(generation_settings["min_new_tokens"]),
        no_repeat_ngram_size=int(generation_settings["no_repeat_ngram_size"]),
        bad_words_ids=bad_words_ids,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    sources = test_df["source"].fillna("").astype(str).tolist()
    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, args.predict_batch_size)):
            tokenized = tokenizer(
                batch_sources,
                return_tensors="pt",
                truncation=True,
                max_length=max_source_length,
                padding=True,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            generated = model.generate(
                **tokenized,
                **generate_kwargs,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([x.strip() for x in decoded])

    pred_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(str).tolist(),
            "prediction": predictions,
        }
    )
    pred_df.to_csv(prediction_path, index=False)

    id_to_pred = {str(k): v for k, v in zip(pred_df["id"].tolist(), pred_df["prediction"].tolist())}
    submission = sample_df.copy()
    submission["id"] = submission["id"].astype(str)
    missing = [x for x in submission["id"].tolist() if x not in id_to_pred]
    if missing:
        raise ValueError(f"Missing predictions for ids: {missing[:5]}")
    submission["translation"] = submission["id"].map(id_to_pred)

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    print(f"OK: wrote {prediction_path}")
    print(f"OK: wrote {submission_path}")
    print(
        "INFO: generation "
        f"beams={generation_settings['num_beams']} "
        f"lp={generation_settings['length_penalty']} "
        f"max_new={generation_settings['max_new_tokens']} "
        f"min_new={generation_settings['min_new_tokens']} "
        f"no_repeat={generation_settings['no_repeat_ngram_size']} "
        f"suppress_extra_ids={generation_settings['suppress_extra_ids']}"
    )


if __name__ == "__main__":
    main()
