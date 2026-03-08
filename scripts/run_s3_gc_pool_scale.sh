#!/usr/bin/env bash
set -euo pipefail

cd /workspace/deep-past-

export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TQDM=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$HF_HOME" logs docs \
  data/processed_byt5_chunks_align_gc_cost14 \
  data/processed_byt5_chunks_align_gc_cost16 \
  data/processed_byt5_chunks_align_gc_cost18

# S3 candidate A: relaxed GC pool, max_align_cost=1.4
.venv-deeppast/bin/python scripts/build_short_aligned_pairs_galechurch.py \
  --config configs/cloud_stage1_len512_lr2e4.yaml \
  --mix-ratio 0.5 \
  --no-allow-replacement \
  --max-align-cost 1.4 \
  --output-train data/processed_byt5_chunks_align_gc_cost14/train_proc.csv \
  --output-folds data/processed_byt5_chunks_align_gc_cost14/folds.csv \
  --report-json runs/S3_GC_pool_scale_cost14.json

# S3 candidate B: relaxed GC pool, max_align_cost=1.6
.venv-deeppast/bin/python scripts/build_short_aligned_pairs_galechurch.py \
  --config configs/cloud_stage1_len512_lr2e4.yaml \
  --mix-ratio 0.5 \
  --no-allow-replacement \
  --max-align-cost 1.6 \
  --output-train data/processed_byt5_chunks_align_gc_cost16/train_proc.csv \
  --output-folds data/processed_byt5_chunks_align_gc_cost16/folds.csv \
  --report-json runs/S3_GC_pool_scale_cost16.json

# S3 candidate C: relaxed GC pool, max_align_cost=1.8
.venv-deeppast/bin/python scripts/build_short_aligned_pairs_galechurch.py \
  --config configs/cloud_stage1_len512_lr2e4.yaml \
  --mix-ratio 0.5 \
  --no-allow-replacement \
  --max-align-cost 1.8 \
  --output-train data/processed_byt5_chunks_align_gc_cost18/train_proc.csv \
  --output-folds data/processed_byt5_chunks_align_gc_cost18/folds.csv \
  --report-json runs/S3_GC_pool_scale_cost18.json
