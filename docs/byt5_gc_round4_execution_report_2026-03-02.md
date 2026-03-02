# ByT5 GC Round4 Execution Report (2026-03-02)

## Goal
- Execute the next high-ROI diagnostics to raise score:
  1. Keep `E7` checkpoint as the stage1 anchor and validate cheap `n-best` rerank gain.
  2. Replace fallback-only short align with a no-replacement Gale–Church pool.
  3. Run a short stage2 curriculum sanity train from stage1 adapter.

## Code / Config Added
- `scripts/eval_nbest_rerank.py`
  - Chunk-level `n-best` generation and rule rerank (`model_score`, digit match, repetition penalty, length outlier penalty).
  - Outputs rowwise/candidate CSV and summary JSON with reconstructed metrics.
- `scripts/build_short_aligned_pairs_galechurch.py`
  - Monotonic length alignment (`1:1`, `1:2`, `2:1`) with confidence filters and no-replacement sampling.
- `configs/byt5_small_lora_chunked_stage1_r8_qv_cont8k.yaml`
- `configs/byt5_small_lora_chunked_stage1_r8_qv_cont8k_inv.yaml`
- `configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml`
  - uses `data/processed_byt5_chunks_align_gc_relaxed`
  - run dir `runs/E10_BYT5_STAGE2_GC_RELAXED`

## Executed Runs

### 1) E9 stage1 (already completed) + n-best rerank
- Base diagnostic (`beam=4 lp=1.2 no_repeat=3 max_new=384`):
  - reconstructed `geom=10.0616`
  - file: `runs/E9_BYT5_STAGE1_R8_QV_CONT8K_fold0/diagnostics/val_diagnostic_summary_step5000_fixeddecode_384.json`
- `n-best` rerank (`num_return_sequences=4`):
  - reconstructed `geom=10.1286`
  - delta vs top1 on chunk metrics: `+0.0565 geom`
  - file: `runs/E9_BYT5_STAGE1_R8_QV_CONT8K_fold0/diagnostics/nbest_rerank_summary_step5000_beam4_n4_lp1p2_m384.json`

### 2) Gale–Church short-align pool build
- strict profile:
  - `pool=1645`, `selected=1645`, `used_replacement=false`
  - file: `runs/E10_short_align_gc_report.json`
- relaxed profile (chosen for stage2 sanity):
  - `pool=2737`, `selected=1820`, `used_replacement=false`
  - align types: `1:1=1802`, `1:2=935`
  - file: `runs/E10_short_align_gc_report_relaxed.json`

### 3) E10 stage2 curriculum sanity run (from E9 adapter)
- train command:
  - `python scripts/train_mt5_lora.py --config configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml --fold 0 --init-adapter-dir runs/E9_BYT5_STAGE1_R8_QV_CONT8K_fold0/best_model --max-steps 1000 --eval-steps 250 --max-val-rows 256 --skip-final-predict`
- trainer best eval (internal):
  - `eval_geom=6.1194`, `eval_bleu=2.3861`, `eval_chrfpp=15.6941`
  - file: `runs/E10_BYT5_STAGE2_GC_RELAXED_fold0/run_summary.json`
- fixed decode diagnostic:
  - reconstructed `geom=10.1777`
  - file: `runs/E10_BYT5_STAGE2_GC_RELAXED_fold0/diagnostics/val_diagnostic_summary_step1000_fixeddecode_384.json`
- fixed decode + `n-best` rerank:
  - reconstructed `geom=10.2125`
  - file: `runs/E10_BYT5_STAGE2_GC_RELAXED_fold0/diagnostics/nbest_rerank_summary_step1000_beam4_n4_lp1p2_m384.json`

## Result Summary
- Best reconstructed score in this round:
  - `10.2125` (E10 + rerank)
- Improvement over E9 baseline diagnostic:
  - `10.2125 - 10.0616 = +0.1509`
- Improvement over E9 rerank:
  - `10.2125 - 10.1286 = +0.0839`

## Decision
- Direction is valid: stage1 anchor + GC-based stage2 + cheap rerank gives consistent positive gain.
- Next run priority:
  1. Extend stage1 continuation to `8k` steps with same fixed training decode.
  2. Re-run stage2 curriculum from the stronger stage1 checkpoint.
  3. Keep `mix_ratio <= 0.5` and `used_replacement=false` as hard constraints.
