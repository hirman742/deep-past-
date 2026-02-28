# E6 ByT5 Chunked Round Report (2026-03-01)

## Scope
- Objective: verify the chunked training path can remove truncation bottlenecks and improve length health on fold0.
- Config: `configs/byt5_small_lora_chunked.yaml`
- Run dir: `runs/E6_BYT5_CHUNKS_fold0`

## What was executed
1. Rebuilt chunked dataset and folds mapping (`scripts/build_long_chunks.py`).
2. Trained fold0 on GPU with `--max-steps 30` (`scripts/train_mt5_lora.py`).
3. Ran decode grid on chunked validation with parent reconstruction scoring (`scripts/eval_decode_grid.py`).
4. Ran diagnostic inference with best decode settings (`scripts/diagnose_val_outputs.py`).

## Key results
- Chunking report (`runs/E6_BYT5_CHUNKS_chunk_report.json`):
  - Train rows: `1561 -> 3641` (added `2080` chunk rows)
  - Truncation ratio: source `59.71% -> 0.00%`, target `51.12% -> 0.00%`
- Training summary (`runs/E6_BYT5_CHUNKS_fold0/run_summary.json`):
  - Eval (chunk-level): `geom=1.3211`, `bleu=0.4160`, `chrfpp=4.1956`
  - Reconstructed (parent-level): `geom=1.9287`, `bleu=0.5987`, `chrfpp=6.2132`
  - Precision: `bf16=true`, `fp16=false`
- Decode grid best (`runs/E6_BYT5_CHUNKS_fold0/decode_grid_best.json`):
  - `num_beams=2`, `no_repeat_ngram_size=0`, `max_new_tokens=512`
  - Reconstructed score: `geom=2.3440`, `bleu=0.7818`, `chrfpp=7.0279`
- Diagnostic with best decode on 256 rows (`runs/E6_BYT5_CHUNKS_fold0/diagnostics/val_diagnostic_summary_chunk_step30_gridbest_256.json`):
  - Chunk-level shorter-half ratio: `7.81%`
  - Reconstructed shorter-half ratio: `1.06%`
  - Empty/copy ratio (reconstructed): `0.00% / 0.00%`

## Decision
- This direction is **viable**: truncation bottleneck has been removed and length health improved sharply after chunk reconstruction.
- Next step: keep chunk pipeline fixed, increase training steps/epochs on fold0, then start 5-fold OOF on the same chunked recipe.
