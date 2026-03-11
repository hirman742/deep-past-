# Taskform 轻量实验 3：proxy_mix 候选池审计（2026-03-09）

## 总览

- filtered + ranked pool size: `1218`
- base_train_rows: `4064`
- mix outputs: `050, 100`
- word_count p50/p90/p95: `95.0 / 183.0 / 221.1`
- marker rates (%): `gap=100.00, brace=84.56, subscript=0.00`

## 输出目录

- `050`: `203` rows -> `/workspace/deep-past-/data/processed_taskform_proxymix_050`
- `100`: `406` rows -> `/workspace/deep-past-/data/processed_taskform_proxymix_100`

## Top genre_label

| genre_label | count |
| --- | ---: |
| letter | 713 |
| unknown | 263 |
| debt note | 60 |
| note | 38 |
| memo | 30 |
| agreement (contract) | 22 |
| testimony | 16 |
| list | 16 |
| legal writing(s) | 14 |
| contract of sale | 8 |
| legal challenge | 6 |
| administrative regulation | 4 |

## 解读

- 这份审计用于判断 `P3` 的 proxy 样本是否真的偏向 hard bucket，而不是只是随便混入额外噪声。
- 如果 marker rate 和 genre 分布明显偏向复杂样本，说明 `P3` 的失败更可能是任务不适配，而不是候选池完全失真。
