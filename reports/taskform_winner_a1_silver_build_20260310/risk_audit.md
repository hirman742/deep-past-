# A1 Silver Risk Audit

- status: `completed`
- external csv: `/workspace/deep-past-/data/external/oracc_parallel.csv`

## Exact Overlap

- vs raw train rows: `0`
- vs raw test rows: `0`
- vs processed train rows: `0`
- vs fold0 val rows: `0`

## High-Similarity Audit

- p95 max similarity: `0.0`
- p99 max similarity: `0.0`
- rows >= 0.85: `0`
- rows >= 0.90: `0`
- rows >= 0.92: `0`
- rows >= 0.95: `0`

## Verdict

- fits winner paradigm: `True`
- external set is bimodal in a useful way: sentence_silver rows resemble short supervision, aggregated parent rows can be chunked by the existing pipeline
- exact overlap with train/test/fold0 val is zero under normalized source comparison
- high-similarity near-duplicate counts vs competition sources are zero at 0.85/0.90/0.92/0.95 thresholds under the current audit

## Caveats

- aggregated parent rows are longer than internal chunk rows and rely on A1_P1 chunking to stay within the winner recipe
- sentence_silver rows are shorter than internal short_aligned rows, so mix ratio and row-origin balance still need A1_P2 gate validation
- silver targets come from sentence aggregation/anchoring and are not gold-aligned in the same sense as competition train
