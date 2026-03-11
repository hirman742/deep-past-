# Taskform 轻量实验 2：replay25 / replay40 覆盖率审计（2026-03-09）

| variant | ratio | train_rows_before | hard_rows | extra_rows_added | rows_after | hard_reweight_mult | extra_over_train_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| replay25 | 0.25 | 4064 | 3682 | 1016 | 5080 | 1.28 | 25.00% |
| replay40 | 0.40 | 4064 | 3682 | 1626 | 5690 | 1.44 | 40.01% |

## 难例覆盖口径（原 train 内的基数）

| variant | chunk_total>=4 | target_len>=129 | gap/brace/bracket |
| --- | ---: | ---: | ---: |
| replay25 | 1058 | 3350 | 1686 |
| replay40 | 1058 | 3350 | 1686 |

## 解读

- `replay40` 比 `replay25` 更激进，本质上是把当前 hard bucket 的权重再放大一档。
- 这份表主要用于后面解释 `P2`：如果 `replay40` 更差，说明 hard-case 重加权过头；如果更好，则说明主问题确实集中在这些难例桶。
