# Taskform 轻量实验 4：winner 桶表 + matched baseline 参考（2026-03-09）

- matched baseline anchor32 geom: `18.3354`

| level | bucket | n | geom | short_pct | cap_hit_pct | repeat_pct | matched_baseline_anchor32_geom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| row | ref_tok129_256 | 714 | 9.4343 | 6.58 | 26.61 | 85.43 | 18.3354 |
| row | ref_tok257+ | 267 | 9.1207 | 14.61 | 35.58 | 80.90 | 18.3354 |
| row | parent_chunks=2_3 | 401 | 15.8690 | 5.74 | 21.95 | 65.09 | 18.3354 |
| row | parent_chunks=4_6 | 380 | 9.8544 | 3.95 | 27.11 | 85.00 | 18.3354 |
| row | parent_chunks>=7 | 420 | 7.2947 | 12.86 | 27.14 | 84.76 | 18.3354 |
| parent | ref_tok<=128 | 231 | 16.7779 | 5.63 | 0.00 | 77.49 | 18.3354 |
| parent | ref_tok129_256 | 62 | 13.1505 | 0.00 | 0.00 | 100.00 | 18.3354 |
| parent | ref_tok257_512 | 16 | 11.2519 | 0.00 | 0.00 | 100.00 | 18.3354 |
| parent | ref_tok513+ | 4 | 7.3026 | 50.00 | 0.00 | 100.00 | 18.3354 |
| parent | parent_chunks=2_3 | 161 | 18.5538 | 4.97 | 0.00 | 71.43 | 18.3354 |
| parent | parent_chunks=4_6 | 82 | 12.4272 | 1.22 | 0.00 | 100.00 | 18.3354 |
| parent | parent_chunks>=7 | 46 | 12.0495 | 4.35 | 0.00 | 100.00 | 18.3354 |

## 解读

- 这不是 taskform 结果表，而是后面 `P1/P2/P3` 做 matched baseline 对照时的参考面。
- 重点看 `2-3 chunk` 与 `4+ chunk` 的断层，以及长 parent 桶的系统性退化。
