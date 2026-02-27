# Deep Past MT (Akkadian → English)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Deep%20Past%20MT-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](env.yml)
[![Conda Env](https://img.shields.io/badge/Conda-deeppast--cleaning-44A833?logo=anaconda&logoColor=white)](env.yml)
[![Last Commit](https://img.shields.io/github/last-commit/hirman742/deep-past-)](https://github.com/hirman742/deep-past-)

Reproducible Kaggle pipeline for the Deep Past Initiative machine translation task, focused on Akkadian to English sequence-to-sequence modeling with mT5/ByT5, LoRA, ORACC mix training, TAPT, and OOF-weighted ensemble.

## Project Overview
- Objective: improve translation quality under low-resource constraints with robust preprocessing and controlled experiments.
- Main stack: `transformers`, `peft` (LoRA), `sacrebleu` metrics, Conda-managed training environment.
- Current workflow: preprocessing -> fold training -> validation diagnostics -> decode grid -> inference -> ensemble.

## Repository Layout
- `configs/`: baseline and aggressive experiment YAML configs.
- `scripts/`: preprocess/train/infer/diagnose plus ORACC mix, chunking, TAPT, and ensemble scripts.
- `data/`: raw/interim/processed datasets (large generated artifacts are ignored by Git).
- `runs/`: experiment outputs, diagnostics, CV summaries, and submission artifacts.
- `references/` and `docs/`: background materials and cleaning/normalization specs.

## Environment
Windows (GPU by default):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

CPU mode:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -ComputeTarget cpu
```

Manual:
```bash
conda env create -f env.yml
conda activate deeppast-cleaning
```

## Core Pipeline
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb.yaml
conda run -n deeppast-cleaning python scripts/train_mt5_lora.py --config configs/mt5_small_lora_8gb.yaml --fold 0
conda run -n deeppast-cleaning python scripts/diagnose_val_outputs.py --config configs/mt5_small_lora_8gb.yaml --fold 0
conda run -n deeppast-cleaning python scripts/infer_mt5_lora.py --config configs/mt5_small_lora_8gb.yaml --fold 0
```

## Aggressive 7-Day Plan (Implemented Scripts)
### Day1: chain fix + decode alignment
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml
conda run -n deeppast-cleaning python scripts/train_mt5_lora.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --fold 0
conda run -n deeppast-cleaning python scripts/diagnose_val_outputs.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --fold 0
```

Decode grid (no retrain):
```bash
conda run -n deeppast-cleaning python scripts/eval_decode_grid.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --fold 0 --beams 4,6,8 --length-penalties 0.8,1.0,1.2,1.4 --no-repeat-ngram-sizes 0,2,3 --min-new-tokens-list 0,8,16
```

### Day2: length expansion
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb_e1_len_512_320.yaml
conda run -n deeppast-cleaning python scripts/train_folds_mt5_lora.py --config configs/mt5_small_lora_8gb_e1_len_512_320.yaml --folds 0,1,2,3,4
```

Optional long context:
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb_e1_len_640_384.yaml
```

### Day3: long-sample chunking
```bash
conda run -n deeppast-cleaning python scripts/build_long_chunks.py --config configs/mt5_small_lora_8gb_e2_chunks.yaml --input-train data/processed_e1_512_320/train_proc.csv --output-train data/processed_e2_chunks/train_proc_chunked.csv --report-json runs/E2_chunk_report.json
```

### Day4: ORACC mix + dedupe
```bash
conda run -n deeppast-cleaning python scripts/prepare_oracc_mix.py --config configs/mt5_small_lora_8gb_e3_oracc10.yaml --ratio 0.10 --oracc-csv data/external/oracc_parallel.csv --output-train data/interim/oracc_mix_train_r10.csv --audit-json runs/oracc_mix_audit_r10.json
conda run -n deeppast-cleaning python scripts/prepare_oracc_mix.py --config configs/mt5_small_lora_8gb_e3_oracc30.yaml --ratio 0.30 --oracc-csv data/external/oracc_parallel.csv --output-train data/interim/oracc_mix_train_r30.csv --audit-json runs/oracc_mix_audit_r30.json
conda run -n deeppast-cleaning python scripts/prepare_oracc_mix.py --config configs/mt5_small_lora_8gb_e3_oracc50.yaml --ratio 0.50 --oracc-csv data/external/oracc_parallel.csv --output-train data/interim/oracc_mix_train_r50.csv --audit-json runs/oracc_mix_audit_r50.json
```

### Day5: TAPT + supervised fine-tune
```bash
conda run -n deeppast-cleaning python scripts/tapt_denoise.py --config configs/mt5_small_lora_8gb_e4_tapt.yaml
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb_e4_tapt.yaml
conda run -n deeppast-cleaning python scripts/train_mt5_lora.py --config configs/mt5_small_lora_8gb_e4_tapt.yaml --fold 0 --init-adapter-dir runs/TAPT_MT5_E4/best_model
```

### Day6: second model family (ByT5)
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/byt5_small_lora_aggressive.yaml
conda run -n deeppast-cleaning python scripts/train_mt5_lora.py --config configs/byt5_small_lora_aggressive.yaml --fold 0
```

### Day7: OOF weight optimization and submission
```bash
conda run -n deeppast-cleaning python scripts/ensemble_oof_opt.py --models mt5=runs/E4_MT5_TAPT_SFT,oracc=runs/E3_MT5_ORACC10,byt5=runs/E5_BYT5 --test-models mt5=runs/E4_MT5_TAPT_SFT_ensemble/test_predictions_folds_0-1-2-3-4.csv,oracc=runs/E3_MT5_ORACC10_ensemble/test_predictions_folds_0-1-2-3-4.csv,byt5=runs/E5_BYT5_ensemble/test_predictions_folds_0-1-2-3-4.csv --sample-submission data/raw/sample_submission.csv --output-dir runs/E6_ENSEMBLE_OPT
```

## New Config Interfaces
- `preprocess.task_prefix`
- `generation.min_new_tokens`
- `generation.no_repeat_ngram_size`
- `generation.suppress_extra_ids`
- `generation.bad_tokens_regex`
- `preprocess.length_candidates`
- `training.lr_scheduler_type`
- `training.early_stopping_patience`

## Notes
- `scripts/train_mt5_lora.py`, `scripts/infer_mt5_lora.py`, `scripts/diagnose_val_outputs.py`, and `scripts/eval_decode_grid.py` now share one decode config source.
- `<extra_id_*>` suppression is controlled by config and applied via `bad_words_ids`.
- Grouped CV falls back automatically when groups are near-unique to avoid degenerate `GroupKFold`.

## Validation Commands
Decode consistency (regression check):
```bash
conda run -n deeppast-cleaning python scripts/check_decode_consistency.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --run-summary runs/E0_MT5_CHAINFIX_fold0/run_summary.json --diagnose-summary runs/E0_MT5_CHAINFIX_fold0/diagnostics/val_diagnostic_summary.json
```

16-sample integration smoke:
```bash
conda run -n deeppast-cleaning python scripts/preprocess.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --max-train-rows 16 --max-test-rows 16
conda run -n deeppast-cleaning python scripts/train_mt5_lora.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --fold 0 --max-steps 10
conda run -n deeppast-cleaning python scripts/diagnose_val_outputs.py --config configs/mt5_small_lora_8gb_e0_chainfix.yaml --fold 0 --sample-size 16
```

Acceptance threshold check:
```bash
conda run -n deeppast-cleaning python scripts/check_plan_acceptance.py --diagnose-summary runs/E0_MT5_CHAINFIX_fold0/diagnostics/val_diagnostic_summary.json --length-stats data/processed_e0/length_stats.json --oracc-audit runs/oracc_mix_audit_r10.json

# stricter gate for baseline-quality runs
conda run -n deeppast-cleaning python scripts/check_plan_acceptance.py --profile baseline --diagnose-summary runs/E0_MT5_CHAINFIX_fold0/diagnostics/val_diagnostic_summary.json --length-stats data/processed_e0/length_stats.json --oracc-audit runs/oracc_mix_audit_r10.json
```
