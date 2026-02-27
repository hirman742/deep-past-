# Changelog

## 2026-02-28

### Engineering updates
- Hardened generation token suppression to avoid accidental full-vocab blocking when `bad_tokens_regex` is empty or too broad.
- Added shared metric utilities and persisted SacreBLEU signatures for reproducibility across train/diagnose/decode-grid/ensemble.
- Added decode-grid resilience: incremental CSV/JSON writes, progress metadata, and resume/skip of completed decode combinations.
- Updated default decode settings in experiment configs to less restrictive values (`min_new_tokens=0`, `no_repeat_ngram_size=0`).
- Pinned core environment/tooling versions in `env.yml` and aligned setup script install behavior with pinned PyTorch.

### Latest run snapshot (fold0, `E2_MT5_LEN`)
- Training summary: `eval_geom=0.3925`, `eval_bleu=0.0842`, `eval_chrfpp=1.8290`.
- Diagnostic summary: `geom=0.2939`, `bleu=0.0548`, `chrfpp=1.5759`.
- Output health: `empty_prediction_ratio=0%`, `exact_extra_id_0_ratio=0%`, `has_bad_token_regex_ratio=0%`.
- Remaining gap: `pred_shorter_than_half_ref_ratio=67.73%` (length adequacy remains the main bottleneck).
