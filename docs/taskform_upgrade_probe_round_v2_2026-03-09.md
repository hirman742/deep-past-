# Taskform Probe Round v2（2026-03-09）

## 口径

- `P1` 保留为任务形式升级主线。
- `P2` 已重构为 `replay + hardselector`，只看 hardest-bucket selector 子集。
- `P3` 已降级为 side-track，本轮不进入主队列。
- 本轮所有 `geom` 都是：
  - `32-sample`
  - `metric_level = chunk_or_sample`
  - `aggregate-by-parent = off`
- 因此本轮分数只用于**同口径 matched baseline 对比**，不直接和正式 full-val `14.3323` 横比。

## 数据口径

### P1：parent-packed / parent-windowed

- 数据目录：
  - [processed_taskform_parentpack_fold0](/workspace/deep-past-/data/processed_taskform_parentpack_fold0)
- 审计：
  - [audit_parentpack.json](/workspace/deep-past-/data/processed_taskform_parentpack_fold0/audit_parentpack.json)
- 关键事实：
  - `rows_out = 4579`
  - `parentpack_2_3 = 845`
  - `parentwindow_3ofN = 3379`

### P2：replay + hardselector

- 数据目录：
  - [processed_taskform_replay25_hardselector_fold0](/workspace/deep-past-/data/processed_taskform_replay25_hardselector_fold0)
  - [processed_taskform_replay40_hardselector_fold0](/workspace/deep-past-/data/processed_taskform_replay40_hardselector_fold0)
- 审计：
  - [replay25 hardselector audit](/workspace/deep-past-/data/processed_taskform_replay25_hardselector_fold0/audit_hardcase_selector.json)
  - [replay40 hardselector audit](/workspace/deep-past-/data/processed_taskform_replay40_hardselector_fold0/audit_hardcase_selector.json)
- hardest-bucket selector 条件：
  - `chunk_total >= 4`
  - `target_len >= 129`
  - source 含 `<gap>` 或 `{` 或 `[`
  - val 只保留 `192` 条 hardest rows

## 结果摘要

### P1：parent-packed / parent-windowed

- matched baseline：
  - [decode_grid_best_taskform_p1_matched_baseline_anchor32_v2.json](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/decode_grid_best_taskform_p1_matched_baseline_anchor32_v2.json)
  - `geom = 9.0096`
- anchor：
  - [ckpt150](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/decode_grid_best_taskform_p1_ckpt150_anchor32_v2.json)
    - `9.2586`
  - [ckpt200](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/decode_grid_best_taskform_p1_ckpt200_anchor32_v2.json)
    - `9.5190`
  - [ckpt250](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/decode_grid_best_taskform_p1_ckpt250_anchor32_v2.json)
    - `8.9377`
- `diag32`：
  - [val_diagnostic_summary_taskform_p1_diag32_v2.json](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/val_diagnostic_summary_taskform_p1_diag32_v2.json)
  - `geom = 9.5190`
  - `empty = 0.00%`
  - `copy = 0.00%`
  - `pred_shorter_than_half_ref = 18.75%`
  - `unique = 93.75%`
  - `pred_tok_p95 = 385`
- 训练侧：
  - [run_summary.json](/workspace/deep-past-/runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/run_summary.json)
  - `best_model_checkpoint = checkpoint-250`
  - `best_metric(eval_loss) = 0.8423`

结论：

- `P1` 在自己的任务口径下，确实打赢了 matched baseline：
  - `9.0096 -> 9.5190`
  - 绝对增益约 `+0.51 geom`
- 但健康还不够稳：
  - 明显存在重复拖尾
  - `short` 仍偏高
- 这说明：
  - `P1` 方向成立
  - 但只能进入 `W-lite`
  - 还不够直接进 `promote-lite`

### P2：replay + hardselector

- matched baseline：
  - [replay25](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/decode_grid_best_taskform_p2_replay25_hardselector_matched_anchor32_v2.json)
    - `8.2732`
  - [replay40](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY40_HARDSELECTOR_PROBE_fold0/diagnostics/decode_grid_best_taskform_p2_replay40_hardselector_matched_anchor32_v2.json)
    - `8.2732`
- smoke winner：
  - `replay25_hardselector`
- anchor：
  - [ckpt150](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/decode_grid_best_taskform_p2_ckpt150_anchor32_v2.json)
    - `8.2602`
  - [ckpt200](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/decode_grid_best_taskform_p2_ckpt200_anchor32_v2.json)
    - `9.2412`
  - [ckpt250](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/decode_grid_best_taskform_p2_ckpt250_anchor32_v2.json)
    - `9.1064`
- `diag32`：
  - [val_diagnostic_summary_taskform_p2_diag32_v2.json](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/val_diagnostic_summary_taskform_p2_diag32_v2.json)
  - `geom = 9.2412`
  - `empty = 0.00%`
  - `copy = 0.00%`
  - `pred_shorter_than_half_ref = 6.25%`
  - `unique = 96.88%`
  - `pred_tok_p95 = 385`
- 训练侧：
  - [run_summary.json](/workspace/deep-past-/runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/run_summary.json)
  - `best_model_checkpoint = checkpoint-150`
  - `best_metric(eval_loss) = 0.9256`

结论：

- `P2` 也在 hardest-bucket 口径下打赢了 matched baseline：
  - `8.2732 -> 9.2412`
  - 绝对增益约 `+0.97 geom`
- 它的相对增益比 `P1` 更大，健康也略稳。
- 但它本质上仍是“在原任务上补 hardest buckets”，不是直接解决多 chunk 整合。
- 因此当前定位应是：
  - `review / reserve`
  - 不先于 `P1` 进入 `W-lite`

## 交叉解释

这轮最重要的结论不是“谁分更高”，而是：

1. `P1` 和 `P2` 都给了真信号，不是噪声。
2. `P1` 更接近根因，因为它直接改变了多 chunk 的组织方式。
3. `P2` 更像补救 hardest buckets 的保底线。
4. 两条线都还远不到“直接替换正式 winner”的程度。

换句话说：

- `P1` 解决的是**结构性瓶颈**
- `P2` 解决的是**训练分布偏置**

## 阶段判断

- `P1`
  - `stage_decision = review_to_wlite`
  - 下一步：进入 `W-lite`
- `P2`
  - `stage_decision = review_hold`
  - 下一步：保留为 reserve line，暂不先行 warmup
- `P3`
  - `stage_decision = sidetrack`
  - 下一步：不进入当前主执行链

## 下一步

当前最合理的下一动作是：

1. 只让 `P1` 进入 `W-lite`
2. `P2` 保留为 reserve / 对照线
3. `W-lite` 结束后由脚本自动判定：
   - 是否进入 `promote-lite`
   - 还是停在 `review`
