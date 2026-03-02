# ByT5 Chunk Round2 Execution Report (2026-03-01)

## Scope
- Implemented and executed the requested improvement track around `ByT5 + chunk`, focusing on:
  - training-to-convergence behavior,
  - LoRA capacity/injection comparison,
  - short-aligned data expansion,
  - parent-level sampling control,
  - decode grid on reconstructed parent-level metrics.

## Code & Config Changes
- `scripts/train_mt5_lora.py`
  - Added CLI: `--max-train-rows`, `--max-val-rows`, `--eval-steps`, `--skip-final-predict`.
  - Added parent-level sampling controls from config:
    - `training.max_chunks_per_parent`
    - `training.parent_inverse_sample`
  - Added parent sampling stats into `run_summary.json` (`parent_sampling`).
- `scripts/build_short_aligned_pairs.py` (new)
  - Builds high-confidence short aligned pairs from chunked train data.
  - Supports delimiter alignment + optional equal-split fallback.
  - Writes mixed `train_proc.csv`, mixed `folds.csv`, and a report JSON.
- New configs:
  - `configs/byt5_small_lora_chunked_stage1_r8_qv.yaml`
  - `configs/byt5_small_lora_chunked_stage1_r16_qv.yaml`
  - `configs/byt5_small_lora_chunked_stage1_r16_qvo.yaml`
  - `configs/byt5_small_lora_chunked_stage2_align_r16_qvo.yaml`

## Data Build
- Ran short-aligned pair builder on `data/processed_byt5_chunks`.
- Report: `runs/E8_short_align_report.json`
  - `rows_extra_pool=7238`
  - `rows_extra_selected=10923`
  - `rows_output_train=14564`

## Executed Runs

### 1) Stage1: LoRA r=16, q/v/o, 1000 steps
- Run: `runs/E7_BYT5_STAGE1_R16_QVO_fold0`
- Train eval (`run_summary`): `geom=4.2944`, `bleu=1.5863`, `chrfpp=11.6255`
- Reconstructed diagnostics:
  - `step1000_fixeddecode_256`: `geom=6.5057`, `bleu=1.7147`, `chrfpp=24.6823`
  - `step1000_fixeddecode_384`: `geom=6.5135`, `bleu=1.7165`, `chrfpp=24.7162`
  - `pred_shorter_than_half_ref_ratio_pct=1.0638`

### 2) Stage2: short-aligned mix + parent cap, LoRA r=16 q/v/o, 1000 steps
- Run: `runs/E8_BYT5_STAGE2_ALIGN_R16_QVO_fold0`
- Train eval (`run_summary`): `geom=4.1765`, `bleu=1.4601`, `chrfpp=11.9465`
- Reconstructed diagnostics:
  - `step1000_fixeddecode_256`: `geom=5.0195`, `bleu=1.2589`, `chrfpp=20.0143`
  - `step1000_fixeddecode_384`: `geom=5.0224`, `bleu=1.2596`, `chrfpp=20.0265`
  - `pred_shorter_than_half_ref_ratio_pct=23.4043`

### 3) Stage1: LoRA r=8, q/v, 3000 steps (convergence check)
- Run: `runs/E7_BYT5_STAGE1_R8_QV_fold0`
- Train eval (`run_summary`): `geom=4.3665`, `bleu=1.5293`, `chrfpp=12.4671`
- Reconstructed diagnostics:
  - `step3000_fixeddecode_256`: `geom=7.7482`, `bleu=2.3100`, `chrfpp=25.9890`
  - `step3000_fixeddecode_384`: `geom=7.7508`, `bleu=2.3102`, `chrfpp=26.0045`
  - `pred_shorter_than_half_ref_ratio_pct=3.1915`

## Decode Grid Results

### E7 (stage1 r8 q/v, 3000-step checkpoint)
- Best: `runs/E7_BYT5_STAGE1_R8_QV_fold0/decode_grid_best.json`
- `num_beams=4`, `length_penalty=1.2`, `no_repeat_ngram_size=3`, `max_new_tokens=384`
- Reconstructed best:
  - `geom=7.9771`
  - `bleu=2.3728`
  - `chrfpp=26.8176`

### E8 (stage2 align r16 q/v/o)
- Best: `runs/E8_BYT5_STAGE2_ALIGN_R16_QVO_fold0/decode_grid_best.json`
- `num_beams=4`, `length_penalty=1.2`, `no_repeat_ngram_size=3`, `max_new_tokens=384`
- Reconstructed best:
  - `geom=6.1958`
  - `bleu=1.6777`
  - `chrfpp=22.8814`

## Decision
- Current strongest line is still `E7_STAGE1_R8_QV` with longer training and tuned decode.
- The current short-aligned mixing recipe (`E8`) did not beat stage1 baseline and should not replace the main line yet.
- Parent-level sampling controls are now available for further balancing experiments.

## Recommended Next Run Order
1. Add the missing `300-step` point for the same `E7_STAGE1_R8_QV` setup to complete the 300/1000/3000 curve.
2. Keep `E7_STAGE1_R8_QV` as main baseline and run 2-fold verification (grouped by parent).
3. Revisit short-aligned mix with lower mix ratio and stricter segment filtering before another full 1000-step comparison.

