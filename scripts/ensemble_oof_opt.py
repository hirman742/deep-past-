from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sacrebleu


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _parse_specs(raw: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid model spec: {chunk!r}, expected name=path")
        name, value = chunk.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid model spec: {chunk!r}")
        specs.append((name, value))
    if not specs:
        raise ValueError("No model specs parsed")
    return specs


def _collect_oof_files(path_spec: str) -> list[Path]:
    raw = Path(path_spec)
    candidates: list[Path] = []
    if any(x in path_spec for x in ["*", "?", "["]):
        wildcard_path = Path(path_spec)
        if wildcard_path.is_absolute():
            candidates = sorted(wildcard_path.parent.glob(wildcard_path.name))
        else:
            candidates = sorted(REPO_ROOT.glob(path_spec))
    else:
        path = _resolve_path(path_spec, REPO_ROOT / path_spec)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            direct = sorted(path.glob("*_fold*/val_predictions.csv"))
            recursive = sorted(path.rglob("val_predictions.csv"))
            candidates = direct if direct else recursive
        else:
            candidates = sorted(path.parent.glob(f"{path.name}_fold*/val_predictions.csv"))
    unique = []
    seen = set()
    for file in candidates:
        file = file.resolve()
        if file not in seen:
            unique.append(file)
            seen.add(file)
    return unique


def _load_oof_dataframe(path_spec: str) -> pd.DataFrame:
    files = _collect_oof_files(path_spec)
    if not files:
        raise FileNotFoundError(f"No OOF files found for spec: {path_spec}")
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file)
        if not {"oare_id", "prediction"}.issubset(frame.columns):
            raise KeyError(f"OOF file missing columns oare_id/prediction: {file}")
        if "reference" not in frame.columns:
            alt_ref = "target" if "target" in frame.columns else ""
            if alt_ref:
                frame = frame.rename(columns={alt_ref: "reference"})
            else:
                raise KeyError(f"OOF file missing reference column: {file}")
        frames.append(frame[["oare_id", "reference", "prediction"]].copy())
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged["oare_id"] = merged["oare_id"].astype(str)
    merged["reference"] = merged["reference"].fillna("").astype(str)
    merged["prediction"] = merged["prediction"].fillna("").astype(str)
    merged = merged.drop_duplicates(subset=["oare_id"], keep="first").reset_index(drop=True)
    return merged


def _load_test_dataframe(path_spec: str) -> pd.DataFrame:
    file = _resolve_path(path_spec, REPO_ROOT / path_spec)
    if not file.exists():
        raise FileNotFoundError(f"Missing test prediction file: {file}")
    frame = pd.read_csv(file)
    if not {"id", "prediction"}.issubset(frame.columns):
        raise KeyError(f"Test prediction file missing columns id/prediction: {file}")
    out = frame[["id", "prediction"]].copy()
    out["id"] = out["id"].astype(str)
    out["prediction"] = out["prediction"].fillna("").astype(str)
    return out


def _compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    bleu = float(sacrebleu.corpus_bleu(predictions, [references]).score)
    chrfpp = float(sacrebleu.corpus_chrf(predictions, [references], word_order=2).score)
    bleu_01 = bleu / 100.0
    chrfpp_01 = chrfpp / 100.0
    geom = math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))
    geom_01 = math.sqrt(max(bleu_01, 0.0) * max(chrfpp_01, 0.0))
    return {
        "bleu": bleu,
        "chrfpp": chrfpp,
        "geom": geom,
        "bleu_01": bleu_01,
        "chrfpp_01": chrfpp_01,
        "geom_01": geom_01,
    }


def _weighted_vote(values: list[str], weights: list[float]) -> str:
    scores: dict[str, float] = {}
    for text, weight in zip(values, weights):
        scores[text] = scores.get(text, 0.0) + float(weight)
    best = max(scores.values())
    winners = [k for k, v in scores.items() if abs(v - best) <= 1e-12]
    winners = sorted(winners, key=lambda x: (len(x), x))
    return winners[0]


def _ensemble_predictions(
    *,
    rows: list[list[str]],
    weights: list[float],
) -> list[str]:
    return [_weighted_vote(values, weights) for values in rows]


def _normalize(weights: np.ndarray) -> list[float]:
    values = np.asarray(weights, dtype=np.float64)
    values = np.clip(values, a_min=0.0, a_max=None)
    total = float(values.sum())
    if total <= 0:
        values[:] = 1.0 / max(1, len(values))
    else:
        values /= total
    return [float(x) for x in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        required=True,
        help="comma list name=OOF_path_or_pattern, path can be csv, dir, glob, or run_root prefix",
    )
    ap.add_argument(
        "--test-models",
        default="",
        help="optional comma list name=test_prediction_csv, names must match --models",
    )
    ap.add_argument("--sample-submission", default="")
    ap.add_argument("--search-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--output-dir", default="runs/OOF_ENSEMBLE_OPT")
    args = ap.parse_args()

    output_dir = _resolve_path(args.output_dir, REPO_ROOT / "runs" / "OOF_ENSEMBLE_OPT")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = _parse_specs(args.models)
    model_names = [name for name, _ in model_specs]
    oof_frames: dict[str, pd.DataFrame] = {}
    for name, path_spec in model_specs:
        oof_frames[name] = _load_oof_dataframe(path_spec)

    merged = None
    for name in model_names:
        frame = oof_frames[name].rename(columns={"prediction": f"prediction__{name}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["oare_id", "reference"], how="inner")
    if merged is None or merged.empty:
        raise ValueError("No OOF rows available after merge")
    merged = merged.sort_values("oare_id").reset_index(drop=True)

    prediction_cols = [f"prediction__{name}" for name in model_names]
    references = merged["reference"].fillna("").astype(str).tolist()
    prediction_rows = merged[prediction_cols].fillna("").astype(str).values.tolist()

    model_metrics = {}
    for name in model_names:
        preds = merged[f"prediction__{name}"].fillna("").astype(str).tolist()
        model_metrics[name] = _compute_metrics(preds, references)

    rng = np.random.default_rng(args.seed)
    candidates: list[dict[str, Any]] = []

    equal_weights = _normalize(np.ones(len(model_names), dtype=np.float64))
    equal_preds = _ensemble_predictions(rows=prediction_rows, weights=equal_weights)
    equal_metrics = _compute_metrics(equal_preds, references)
    candidates.append({"label": "equal", "weights": equal_weights, "metrics": equal_metrics})

    for idx, name in enumerate(model_names):
        one_hot = np.zeros(len(model_names), dtype=np.float64)
        one_hot[idx] = 1.0
        weights = _normalize(one_hot)
        preds = _ensemble_predictions(rows=prediction_rows, weights=weights)
        metrics = _compute_metrics(preds, references)
        candidates.append({"label": f"single_{name}", "weights": weights, "metrics": metrics})

    for run_idx in range(max(0, int(args.search_iters))):
        sample = rng.dirichlet(np.ones(len(model_names), dtype=np.float64))
        weights = _normalize(sample)
        preds = _ensemble_predictions(rows=prediction_rows, weights=weights)
        metrics = _compute_metrics(preds, references)
        candidates.append({"label": f"rand_{run_idx}", "weights": weights, "metrics": metrics})

    scored = sorted(
        candidates,
        key=lambda x: (
            float(x["metrics"]["geom_01"]),
            float(x["metrics"]["bleu_01"]),
            float(x["metrics"]["chrfpp_01"]),
        ),
        reverse=True,
    )
    best = scored[0]
    best_weights = [float(x) for x in best["weights"]]
    best_preds = _ensemble_predictions(rows=prediction_rows, weights=best_weights)

    oof_out = pd.DataFrame(
        {
            "oare_id": merged["oare_id"].astype(str).tolist(),
            "reference": references,
            "prediction": best_preds,
        }
    )
    oof_out_path = output_dir / "oof_ensemble_predictions.csv"
    oof_out.to_csv(oof_out_path, index=False)

    rows = []
    for item in scored[: max(1, int(args.top_k))]:
        row = {"label": item["label"]}
        for i, name in enumerate(model_names):
            row[f"w_{name}"] = float(item["weights"][i])
        row.update(
            {
                "eval_bleu": float(item["metrics"]["bleu"]),
                "eval_chrfpp": float(item["metrics"]["chrfpp"]),
                "eval_geom": float(item["metrics"]["geom"]),
                "eval_bleu_01": float(item["metrics"]["bleu_01"]),
                "eval_chrfpp_01": float(item["metrics"]["chrfpp_01"]),
                "eval_geom_01": float(item["metrics"]["geom_01"]),
            }
        )
        rows.append(row)
    top_df = pd.DataFrame(rows)
    top_path = output_dir / "oof_weight_search_topk.csv"
    top_df.to_csv(top_path, index=False)

    pairwise_agreement: list[dict[str, Any]] = []
    for i, left in enumerate(model_names):
        for j in range(i + 1, len(model_names)):
            right = model_names[j]
            left_preds = merged[f"prediction__{left}"].fillna("").astype(str)
            right_preds = merged[f"prediction__{right}"].fillna("").astype(str)
            agree = float((left_preds == right_preds).mean())
            pairwise_agreement.append({"left": left, "right": right, "exact_match_ratio": agree})

    summary = {
        "num_rows_oof": int(len(merged)),
        "models": model_names,
        "model_metrics": model_metrics,
        "best": {
            "label": str(best["label"]),
            "weights": {name: float(best_weights[i]) for i, name in enumerate(model_names)},
            "metrics": best["metrics"],
        },
        "equal_baseline_metrics": equal_metrics,
        "pairwise_exact_match_ratio": pairwise_agreement,
        "artifacts": {
            "oof_ensemble_predictions_csv": str(oof_out_path),
            "topk_csv": str(top_path),
        },
    }

    if args.test_models.strip():
        test_specs = _parse_specs(args.test_models)
        test_map = {name: path for name, path in test_specs}
        missing = [name for name in model_names if name not in test_map]
        if missing:
            raise ValueError(f"test model specs missing names: {missing}")

        merged_test = None
        for name in model_names:
            frame = _load_test_dataframe(test_map[name]).rename(columns={"prediction": f"prediction__{name}"})
            if merged_test is None:
                merged_test = frame
            else:
                merged_test = merged_test.merge(frame, on="id", how="inner")
        if merged_test is None or merged_test.empty:
            raise ValueError("No test rows available after merge")
        merged_test = merged_test.sort_values("id").reset_index(drop=True)
        test_rows = merged_test[[f"prediction__{name}" for name in model_names]].astype(str).values.tolist()
        test_preds = _ensemble_predictions(rows=test_rows, weights=best_weights)
        test_out = pd.DataFrame({"id": merged_test["id"].astype(str).tolist(), "prediction": test_preds})
        test_out_path = output_dir / "test_ensemble_predictions.csv"
        test_out.to_csv(test_out_path, index=False)
        summary["artifacts"]["test_ensemble_predictions_csv"] = str(test_out_path)

        if args.sample_submission.strip():
            sample_path = _resolve_path(args.sample_submission, REPO_ROOT / "data" / "raw" / "sample_submission.csv")
            sample_df = pd.read_csv(sample_path)
            if "id" not in sample_df.columns:
                raise KeyError(f"sample submission missing id column: {sample_path}")
            sample_df = sample_df.copy()
            sample_df["id"] = sample_df["id"].astype(str)
            id_to_pred = {str(k): v for k, v in zip(test_out["id"], test_out["prediction"])}
            if "translation" not in sample_df.columns:
                sample_df["translation"] = ""
            sample_df["translation"] = sample_df["id"].map(id_to_pred)
            submission_path = output_dir / "submission.csv"
            sample_df.to_csv(submission_path, index=False)
            summary["artifacts"]["submission_csv"] = str(submission_path)

    summary_path = output_dir / "oof_ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"OK: wrote {oof_out_path}")
    print(f"OK: wrote {top_path}")
    print(f"OK: wrote {summary_path}")
    print(
        "INFO: best geom/bleu/chrfpp="
        f"{best['metrics']['geom']:.4f}/"
        f"{best['metrics']['bleu']:.4f}/"
        f"{best['metrics']['chrfpp']:.4f}"
    )


if __name__ == "__main__":
    main()
