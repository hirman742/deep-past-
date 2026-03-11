# Winner Replay / Curriculum Probe

- status: `review_to_candidate_pool`
- reason: best replay arm clears the written probe gate against matched control
- incumbent anchor64 geom: `16.5057`
- frozen fallback anchor/fullval/hard: `23.5415 / 19.9035 / 20.7888`

## Control

- anchor64 geom: `15.3962`
- hard geom: `14.5055`

## replay25

- anchor64 geom: `15.9949`
- hard geom: `15.9245`
- delta anchor vs control: `+0.5987`
- delta hard vs control: `+1.4190`
- train runtime ratio vs control: `0.0000`
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
- train runtime ratio vs control: `0.0000`
- health no_regression: `False`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `-0.9857`
- delta vs frozen anchor64: `-8.0214`
- status: `reject_stop`

