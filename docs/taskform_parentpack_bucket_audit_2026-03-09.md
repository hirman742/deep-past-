# Taskform 轻量实验 1：parentpack 桶审计（2026-03-09）

## 总览

- 原始行数：`5289`
- parentpack 后行数：`4579`
- parent 数：`1561`
- 平均 pack 大小：`1.99`
- 最大 pack 大小：`3`

## 按 chunk_mode

| chunk_mode | rows | parents | avg_pack_rows | avg_src_chars | avg_tgt_chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| parentwindow_3ofN | 3379 | 549 | 2.12 | 341.0 | 417.8 |
| parentpack_2_3 | 845 | 657 | 1.84 | 321.8 | 312.6 |
| single | 355 | 355 | 1.00 | 216.8 | 209.7 |

## 按 pack 行数

| pack_rows | rows | parents | avg_src_chars | avg_tgt_chars |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1327 | 1075 | 187.3 | 217.8 |
| 2 | 2016 | 1049 | 348.3 | 402.9 |
| 3 | 1236 | 481 | 445.4 | 525.2 |

## 解读

- `parentwindow_3ofN` 是主力模式，说明这轮任务形式升级的主体其实是“多 chunk parent 的窗口化重组”。
- 平均 pack 大小接近 `2`，意味着数据并没有被粗暴拼成超长输入，而是仍然保持短窗口策略。
- 这个审计主要用于解释后面 `P1` 的 matched baseline 和 probe 结果。
