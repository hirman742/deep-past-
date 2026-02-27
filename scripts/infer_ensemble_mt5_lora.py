from __future__ import annotations

import argparse
import json
from collections import Counter
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


def _parse_folds(value: str, default_n_folds: int) -> list[int]:
    if not value.strip():
        return list(range(default_n_folds))
    folds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not folds:
        raise ValueError("No valid folds parsed")
    return sorted(set(folds))


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    best_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == best_count]
    winners = sorted(winners, key=lambda x: (len(x), x))
    return winners[0]


def _weighted_vote(values: list[str], weights: list[float]) -> str:
    if len(values) != len(weights):
        raise ValueError("values/weights length mismatch")
    scores: dict[str, float] = {}
    for text, weight in zip(values, weights):
        scores[text] = scores.get(text, 0.0) + float(weight)
    best = max(scores.values())
    winners = [k for k, v in scores.items() if abs(v - best) <= 1e-12]
    winners = sorted(winners, key=lambda x: (len(x), x))
    return winners[0]


def _predict_for_fold(
    *,
    model_name: str,
    checkpoint_dir: Path,
    sources: list[str],
    max_source_length: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    suppress_extra_ids: bool,
    bad_tokens_regex: str,
    predict_batch_size: int,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    bad_words_ids = build_bad_words_ids(
        tokenizer=tokenizer,
        suppress_extra_ids=suppress_extra_ids,
        bad_tokens_regex=bad_tokens_regex,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for batch_sources in _chunk(sources, max(1, predict_batch_size)):
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
                **build_generate_kwargs(
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                ),
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend([x.strip() for x in decoded])

    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mt5_small_lora_8gb.yaml")
    ap.add_argument("--folds", default="")
    ap.add_argument("--predict-batch-size", type=int, default=32)
    ap.add_argument("--submission-path", default="")
    ap.add_argument("--weights-json", default="")
    args = ap.parse_args()

    cfg_path = _resolve_path(args.config, REPO_ROOT / "configs" / "mt5_small_lora_8gb.yaml")
    cfg = _load_yaml(cfg_path)

    paths_cfg = cfg.get("paths", {}) or {}
    preprocess_cfg = cfg.get("preprocess", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    generation_settings = resolve_generation_settings(model_cfg=model_cfg, gen_cfg=gen_cfg)

    default_n_folds = int(preprocess_cfg.get("folds", 5))
    folds = _parse_folds(args.folds, default_n_folds)
    fold_tag = "-".join(str(x) for x in folds)

    processed_dir = _resolve_path(paths_cfg.get("processed_dir"), REPO_ROOT / "data" / "processed")
    test_path = processed_dir / "test_proc.csv"
    sample_submission_path = _resolve_path(
        paths_cfg.get("sample_submission_csv"), REPO_ROOT / "data" / "raw" / "sample_submission.csv"
    )
    run_root = _resolve_path(paths_cfg.get("run_dir"), REPO_ROOT / "runs" / "A1_MT5_FOLD0")
    ensemble_dir = run_root.parent / f"{run_root.name}_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    predictions_out = ensemble_dir / f"test_predictions_folds_{fold_tag}.csv"
    summary_out = ensemble_dir / f"ensemble_summary_folds_{fold_tag}.json"
    submission_path = _resolve_path(args.submission_path, ensemble_dir / f"submission_folds_{fold_tag}.csv")

    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_submission_path)
    if "id" not in test_df.columns or "source" not in test_df.columns:
        raise KeyError("test_proc.csv missing required columns id/source")

    model_name = str(model_cfg.get("name", "google/mt5-small"))
    max_source_length = int(model_cfg.get("max_source_length", 256))
    max_new_tokens = int(generation_settings["max_new_tokens"])
    min_new_tokens = int(generation_settings["min_new_tokens"])
    num_beams = int(generation_settings["num_beams"])
    length_penalty = float(generation_settings["length_penalty"])
    no_repeat_ngram_size = int(generation_settings["no_repeat_ngram_size"])
    suppress_extra_ids = bool(generation_settings["suppress_extra_ids"])
    bad_tokens_regex = str(generation_settings["bad_tokens_regex"])

    sources = test_df["source"].fillna("").astype(str).tolist()
    fold_predictions: dict[int, list[str]] = {}
    for fold in folds:
        checkpoint_dir = run_root.parent / f"{run_root.name}_fold{fold}" / "best_model"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {checkpoint_dir}")
        print(f"RUN: fold={fold} checkpoint={checkpoint_dir}")
        fold_predictions[fold] = _predict_for_fold(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            sources=sources,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_extra_ids=suppress_extra_ids,
            bad_tokens_regex=bad_tokens_regex,
            predict_batch_size=args.predict_batch_size,
        )

    fold_weights = [1.0 for _ in folds]
    if args.weights_json.strip():
        weights_path = _resolve_path(args.weights_json, ensemble_dir / "weights.json")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights json: {weights_path}")
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weight_map = payload.get("fold_weights") or payload.get("weights") or {}
        fold_weights = []
        for fold in folds:
            key = str(fold)
            if key in weight_map:
                fold_weights.append(float(weight_map[key]))
            elif f"fold{fold}" in weight_map:
                fold_weights.append(float(weight_map[f"fold{fold}"]))
            else:
                fold_weights.append(1.0)
        if sum(max(0.0, w) for w in fold_weights) <= 0:
            fold_weights = [1.0 for _ in folds]

    ensemble_predictions: list[str] = []
    vote_strength: list[int] = []
    unanimous_count = 0
    for idx in range(len(sources)):
        candidates = [fold_predictions[fold][idx] for fold in folds]
        if args.weights_json.strip():
            voted = _weighted_vote(candidates, fold_weights)
        else:
            voted = _majority_vote(candidates)
        ensemble_predictions.append(voted)
        counts = Counter(candidates)
        best_count = max(counts.values())
        vote_strength.append(int(best_count))
        if best_count == len(folds):
            unanimous_count += 1

    prediction_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(str).tolist(),
            "prediction": ensemble_predictions,
            "vote_strength": vote_strength,
        }
    )
    prediction_df.to_csv(predictions_out, index=False)

    sample_df = sample_df.copy()
    sample_df["id"] = sample_df["id"].astype(str)
    id_to_pred = {str(k): v for k, v in zip(prediction_df["id"], prediction_df["prediction"])}
    missing = [x for x in sample_df["id"].tolist() if x not in id_to_pred]
    if missing:
        raise ValueError(f"Missing predictions for ids: {missing[:5]}")
    sample_df["translation"] = sample_df["id"].map(id_to_pred)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(submission_path, index=False)

    summary = {
        "config_path": str(cfg_path),
        "folds": folds,
        "num_samples": int(len(sources)),
        "unanimous_ratio_pct": 100.0 * float(unanimous_count) / float(max(1, len(sources))),
        "avg_vote_strength": float(sum(vote_strength)) / float(max(1, len(vote_strength))),
        "fold_weights": {str(f): float(w) for f, w in zip(folds, fold_weights)},
        "predictions_path": str(predictions_out),
        "submission_path": str(submission_path),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {predictions_out}")
    print(f"OK: wrote {submission_path}")
    print(f"OK: wrote {summary_out}")


if __name__ == "__main__":
    main()
