# A2 Freeze Bundle

- status: `candidate_frozen_manual_promote_recommended`
- selected candidate: `fallback_180`
- candidate type: `posthoc_incumbent_fallback_on_repeated_generic_chunks`
- freeze date: `2026-03-10`

## Decision

- promote compare candidate is `fallback_180`
- raw retrieval W-lite remains score ceiling reference only
- this is a post-hoc health-safe candidate, not a new retrained checkpoint

## Scoreboard

- incumbent full-val / hard / anchor64: `14.3323 / 13.7161 / 16.5057`
- raw W-lite full-val / hard / anchor64: `19.9908 / 20.8360 / 23.5415`
- frozen candidate full-val / hard / anchor64: `19.9035 / 20.7888 / 23.5415`
- frozen vs incumbent full-val delta: `+5.5713`
- frozen vs raw W-lite full-val delta: `-0.0872`

## Health

- no_regression vs incumbent: `True`
- unique delta vs incumbent: `+0.0000`
- short delta vs incumbent: `-0.4898`
- empty delta vs incumbent: `-0.0816`

## Official-like

- status: `template_ready`
- bridge probe status: `missing_bridge`
- note: official-like remains local proxy until bridge lands
- recommendation: no official metric bridge files found; keep official-like layer and add bridge later

## Support

- support bundle status: `ready_partial_bundle`
- cache hit stats: `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/retrieval_cache_hit_stats.json`
- neighbor quality audit: `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/nearest_neighbor_quality_audit.json`
- latency report: `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/latency_report.json`
- memory report: `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/memory_usage_report.json`

## Changed Rows

- changed chunk rows: `58` / `1225` (`4.7347%`)
- changed parents: `29`
- changed original rows: `8`
- changed hard rows / parents: `25` / `12`

## Frozen Artifacts

- full-val chunk csv: `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/fallback_180_fullval_chunk.csv`
- full-val reconstructed csv: `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/fallback_180_fullval_reconstructed.csv`
- raw full-val chunk csv: `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/raw_fullval_chunk.csv`
- raw full-val reconstructed csv: `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/raw_fullval_reconstructed.csv`
- changed rows csv: `/workspace/deep-past-/reports/taskform_winner_a2_promote_compare_20260310/changed_rows.csv`
- repeat summary json: `/workspace/deep-past-/reports/taskform_winner_a2_promote_compare_20260310/repeat_group_summary.json`
