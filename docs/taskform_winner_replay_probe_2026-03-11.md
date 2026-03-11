# Winner Replay / Curriculum Probe（2026-03-11）

- status: `review_to_candidate_pool`
- reason: best replay arm clears the written probe gate against matched control
- summary: `/workspace/deep-past-/reports/taskform_winner_replay_probe_20260311/summary.json`
- gate report: `/workspace/deep-past-/reports/taskform_winner_replay_probe_20260311/gate_report.md`
- best label: `replay25`

## Baselines

- incumbent anchor64 geom: `16.5057`
- frozen fallback anchor/fullval/hard: `23.5415 / 19.9035 / 20.7888`

## Control

- control anchor64 geom: `15.3962`
- control hard geom: `14.5055`

## replay25

- anchor64 geom: `15.9949`
- hard geom: `15.9245`
- delta anchor vs control: `+0.5987`
- delta hard vs control: `+1.4190`
- health no_regression: `True`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `-0.5108`
- delta vs frozen anchor64: `-7.5466`
- status: `review_to_candidate_pool`

## replay40

- anchor64 geom: `15.5201`
- hard geom: `15.0942`
- delta anchor vs control: `+0.1239`
- delta hard vs control: `+0.5887`
- health no_regression: `False`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `-0.9857`
- delta vs frozen anchor64: `-8.0214`
- status: `reject_stop`
