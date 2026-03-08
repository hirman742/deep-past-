#!/usr/bin/env bash
set -euo pipefail

cd /workspace/deep-past-

export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TQDM=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$HF_HOME" logs docs

# Before running:
# 1) pick the GC pool variant from S3
# 2) point configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml at that processed_dir if needed

.venv-deeppast/bin/python scripts/train_mt5_lora.py \
  --config configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml \
  --fold 0 \
  --init-adapter-dir runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/best_model \
  --max-steps 4000 \
  --eval-steps 500 \
  --max-val-rows 900 \
  --skip-final-predict
