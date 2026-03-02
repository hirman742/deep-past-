# E8 Diagnostic Round3 (2026-03-02)

## Goal
- Execute the requested diagnosis fixes to validate why `E8` underperformed `E7`.
- Focus items:
  1. Reconstruction comparability (`original chunk only` vs unfiltered).
  2. Mix-ratio downshift without replacement sampling.
  3. High-confidence alignment check (`no fallback_equal_split`).

## Code Changes
- `scripts/diagnose_val_outputs.py`
  - Added `--aggregate-original-only/--no-aggregate-original-only` (default: on).
  - Reconstruction now supports filtering out `is_short_aligned` / `chunk_mode=short_aligned_*`.
  - Added reconstruction filter stats to summary JSON.
- `scripts/eval_decode_grid.py`
  - Added `--aggregate-original-only/--no-aggregate-original-only` (default: on).
  - Parent reconstruction path now supports original-chunk-only filtering and logs filtered row count in output rows.
- `scripts/train_mt5_lora.py`
  - Final reconstructed eval now auto-filters short-aligned rows when marker columns exist.
  - `run_summary.json` reconstructed block now records filter stats.
- `scripts/build_short_aligned_pairs.py`
  - Added replacement control:
    - `--allow-replacement` / `--no-allow-replacement` (default: no replacement).
  - When requested extra rows exceed pool and replacement is disabled, sampling now caps at pool size.
  - Report now includes `requested_extra_rows`, `effective_extra_rows`, `used_replacement`, and unique selected count.

## Experiment 1 — Reconstruction comparability

### A) Same decode, 256 rows (historical setting)
- `e8_cmp_filtered` vs `e8_cmp_unfiltered`
- Result: identical reconstructed metrics.
- Reason: first 256 rows contain no short-aligned rows (`filtered_rows=0`).

### B) Same decode, 900 rows (includes short-aligned rows)
- Command setting: `beams=1`, `lp=1.0`, `no_repeat=0`, `max_new=128`, `max_rows=900`.
- Filtered (`aggregate_original_only=true`):
  - `geom=4.3794`, `bleu=1.3282`, `chrfpp=14.4402`
  - `shorter_half=23.9617`
  - filter stats: `rows_before=900`, `rows_after=820`, `filtered_rows=80`
- Unfiltered (`aggregate_original_only=false`):
  - `geom=4.5195`, `bleu=1.3879`, `chrfpp=14.7171`
  - `shorter_half=21.4058`

Interpretation:
- In this checkpoint/slice, unfiltered score is slightly higher, so the current `E8 < E7` gap is **not explained** by reconstruction contamination alone.
- Comparability guard is now implemented and measurable.

## Experiment 2 — Mix-ratio downshift (no replacement)

All runs used `build_short_aligned_pairs.py` with default `--no-allow-replacement`.

- `mix=0.3`: `pool=7238`, `selected=1092`, `used_replacement=false`
  - report: `runs/E8_diag_mix03_report.json`
- `mix=0.5`: `pool=7238`, `selected=1820`, `used_replacement=false`
  - report: `runs/E8_diag_mix05_report.json`
- `mix=1.0`: `pool=7238`, `selected=3641`, `used_replacement=false`
  - report: `runs/E8_diag_mix10_report.json`
- `mix=3.0`: `pool=7238`, `selected=7238` (capped), `used_replacement=false`
  - report: `runs/E8_diag_mix30_report.json`

Interpretation:
- Replacement oversampling is now removed by default.
- Large-ratio duplication risk is controlled at data-construction stage.

## Experiment 3 — Disable fallback alignment

- Run: `mix=0.5` + `--no-fallback-equal-split`
- Result: no candidates found (`pool=0`, `selected=0`).
- report: `runs/E8_diag_mix05_nofb_report.json`

Interpretation:
- Current short-aligned expansion is entirely coming from `fallback_equal` mode, not delimiter high-confidence alignment.
- `candidate_parent_rows_by_mode` currently shows only fallback mode in available pool.

## Diagnosis Outcome
- Confirmed root behavior:
  - `E8` expansion pool quality is weak (fallback-only).
  - Previous replacement amplification issue existed and is now fixed.
- Not confirmed:
  - Reconstruction contamination as the dominant cause of `E8 < E7`.

## Next Actions (minimal and high-signal)
1. Train a short run on `mix=0.3` and `mix=0.5` (no replacement), keep all other knobs fixed.
2. Keep `aggregate_original_only=true` for all decode-grid and diagnostic comparisons.
3. Improve delimiter-based segmentation rules before attempting `no-fallback` training again.
