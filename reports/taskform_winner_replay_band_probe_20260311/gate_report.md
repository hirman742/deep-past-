# Winner Replay Narrow Probe

- status: `review_to_candidate_pool`
- reason: best replay-band arm clears the written probe gate against matched control
- incumbent anchor64 geom: `15.6636`
- frozen fallback anchor/fullval/hard: `23.5415 / 19.9035 / 20.7888`

## Control

- anchor64 geom: `15.5527`
- hard geom: `14.7113`

## replay15

- anchor64 geom: `15.8344`
- hard geom: `15.6694`
- delta anchor vs control: `+0.2817`
- delta hard vs control: `+0.9581`
- train runtime ratio vs control: `0.0000`
- health no_regression: `True`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `+0.1708`
- delta vs frozen anchor64: `-7.7071`
- status: `review_to_candidate_pool`

## replay20

- anchor64 geom: `15.5928`
- hard geom: `14.8164`
- delta anchor vs control: `+0.0402`
- delta hard vs control: `+0.1051`
- train runtime ratio vs control: `0.0000`
- health no_regression: `False`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `-0.0708`
- delta vs frozen anchor64: `-7.9487`
- status: `reject_stop`

## replay30

- anchor64 geom: `15.2102`
- hard geom: `14.9073`
- delta anchor vs control: `-0.3424`
- delta hard vs control: `+0.1960`
- train runtime ratio vs control: `0.0000`
- health no_regression: `True`
- reconstructed health no_regression: `True`
- delta vs incumbent anchor64: `-0.4534`
- delta vs frozen anchor64: `-8.3313`
- status: `reject_stop`

## Queue Eligibility

- best label: `replay15`
- eligible_for_postprobe_fullval: `True`

