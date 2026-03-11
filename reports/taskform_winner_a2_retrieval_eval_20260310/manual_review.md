# A2 Retrieval Manual Review

- status: `manual_promote_to_w`
- rationale: `R1` clears the written `A2` gate on score, hard subset, targeted buckets, and latency; automatic `review_stop` was triggered only by a small `unique_prediction_ratio` drop.

## Core Evidence

- `R0 anchor64 geom = 15.8741`
- `R1 anchor64 geom = 22.2513`
- `delta geom R1-R0 = +6.3772`
- `hard subset geom = 15.2172 -> 22.9848`
- `delta geom R1-I0 = +5.7455`
- latency ratio `R1 / R0 = 1.0110`
- targeted positive buckets:
  - `rare_name`
  - `formula`

## Health Review

- `empty_prediction_ratio_pct` improved:
  - `1.5625 -> 0.0`
- `pred_shorter_than_half_ref_ratio_pct` improved:
  - `14.0625 -> 12.5`
- `unique_prediction_ratio_pct` worsened slightly:
  - `98.4375 -> 96.875`

The only duplicated row-level prediction appears `3` times and is concentrated inside a single long parent:

- parent id:
  - `092043d6-49a6-40cd-b867-5771e4babd95`
- duplicated chunks:
  - `c5of13`
  - `c11of13`
  - `c12of13`

This is a real localized failure, but not a global degeneration:

- no repeated parent-level reconstructed predictions
- reconstructed `short` improved to `0.0%`
- gains are distributed across multiple large positive deltas, not one lucky row

## Decision

Manual decision:

- promote `R1 retrieval-top1` into `W-lite`

Immediate next action:

- run `retrieval-top1 W-lite @ 400 steps`
