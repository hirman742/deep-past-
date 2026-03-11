# Winner A3 Revisit

- status: `completed_revisit_audit`
- rows: `17`
- pool unique candidate ratio: `100.0`
- rows all same: `0.0`
- rows all unique: `100.0`
- best oracle label: `replay15`
- best oracle delta vs best single: `2.1503`

## Candidate uniqueness

- `retrieval_raw_longtrain` unique prediction ratio: `100.0`
- `retrieval_raw_longtrain` sentence geom mean: `17.9762`
- `replay15` unique prediction ratio: `100.0`
- `replay15` sentence geom mean: `17.436`

## Pairwise overlap

- `retrieval_raw_longtrain__vs__replay15` exact overlap: `0.0`
- `retrieval_raw_longtrain__vs__replay15` bottom25 overlap / jaccard: `23.5294` / `0.6667`

## Oracle Pairwise

- `retrieval_raw_longtrain vs replay15` disagreement ratio: `100.0`
- `retrieval_raw_longtrain vs replay15` oracle delta vs best single: `2.1503`
- `retrieval_raw_longtrain vs replay15` heuristic delta vs best single: `0.0`
