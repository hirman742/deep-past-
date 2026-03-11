# Winner Hook 阶段报告（2026-03-10）

## 1. 目标

本轮任务是把 `RK_true_hook` 从 proxy/单独脚本状态，接入正式 `decode / diagnose` 主路径，并做最小 smoke 对照，判断它是否值得进入更大口径 probe。

主路径接线文件：

- `scripts/retrieval_logits_hook.py`
- `scripts/eval_decode_grid.py`
- `scripts/diagnose_val_outputs.py`

## 2. 本轮做了什么

### 2.1 主路径接线

新增共享模块：

- `scripts/retrieval_logits_hook.py`

功能包括：

- train-visible retrieval frame 加载
- query side neighbor 构造
- target token vote 聚合
- decoder-time `RetrievalBiasLogitsProcessor`
- batch 级 hook 装配

正式脚本新增参数：

- `--rk-enabled`
- `--rk-k`
- `--rk-raw-pool-k`
- `--rk-bias-strength`
- `--rk-max-bias-steps`
- `--rk-report-dir`

因此现在 `eval_decode_grid.py` 与 `diagnose_val_outputs.py` 都可以直接跑真实 hook，而不是只靠单独 probe 脚本。

### 2.2 最小 smoke 对照

模型与口径：

- config:
  - `reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml`
- checkpoint:
  - `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model`
- decode:
  - `beam=4 / lp=0.7 / max_new_tokens=384`
- subset:
  - `max_rows = 8`

共跑了三组：

1. baseline no-hook
2. true-hook `alpha=1.5, k=8, raw_pool_k=48, max_bias_steps=192`
3. weak true-hook `alpha=0.5, k=8, raw_pool_k=48, max_bias_steps=32`

## 3. 结果

### 3.1 baseline

- reconstructed `geom / bleu / chrfpp`
  - `18.4946 / 9.1878 / 37.2288`

文件：

- `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_diagnostic_summary_taskform_winner_rk_baseline_smoke8_20260310.json`

### 3.2 true-hook（alpha=1.5）

- reconstructed `geom / bleu / chrfpp`
  - `16.6067 / 8.1902 / 33.6723`
- 相对 baseline
  - `geom -1.8879`

文件：

- `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/val_diagnostic_summary_taskform_winner_rk_truehook_smoke8_20260310.json`

### 3.3 weak true-hook（alpha=0.5, steps=32）

- eval `geom / bleu / chrfpp`
  - `17.5879 / 8.7474 / 35.3631`
- 相对 baseline
  - `geom -0.9067`

文件：

- `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/diagnostics/decode_grid_best_taskform_winner_rk_truehook_smoke8_a05s32_20260310.json`

### 3.4 与旧 proxy probe 合并判断

此前 `anchor64 proxy` sweep 已经显示：

- `alpha=1.5`: reconstructed `23.5422 -> 21.9962`
- `alpha=3.0`: reconstructed `23.5422 -> 22.4150`
- `alpha=4.5`: reconstructed `23.5422 -> 22.4518`

即 reconstructed 主分在已测区间内始终为负。

对应文件：

- `reports/taskform_winner_a2_rk_anchor64_probe_20260310/summary.json`
- `reports/taskform_winner_a2_rk_anchor64_probe_20260310/alpha_sweep.csv`

## 4. 失败方式

当前 true-hook 不是“接不进去”，而是“接进去了但当前 formulation 会伤主分”。

从 8-row 对照看，5/8 行预测发生变化，主要坏法是：

- 把原本较长的 chunk 预测明显压短
- 过早贴近近邻 target 模板
- 对 parent 重构的全局连贯性造成伤害

这说明当前 token-vote bias 更像“强词表先验”，而不是稳定的 decoder-state kNN-MT。

## 5. 阶段结论

本轮可以下两个明确结论：

1. `RK_true_hook` 主路径接线已完成
2. 当前参数与当前 formulation 下，`RK_true_hook` 不应进入更大口径 GPU 主线

当前状态应记为：

- `parked_negative_smoke`

## 6. 下一步建议

当前不建议直接开：

- `RK_true_hook anchor64`
- `RK_true_hook W-lite`
- 任意基于当前 hook 的 full-val

若后续重开，只建议走更窄的改法：

- selective hook
  - 只对短 chunk / 高 generic 风险 chunk 启用
- retrieval fallback
  - 只在明确重复 generic 输出上触发
- 更像真正 kNN-MT 的 decoder-state / prefix-conditioned 检索，而不是当前静态 token vote

在此之前，主序仍应回到：

- `A2` promote / official-like / artifact freeze
- 等 `A1` 外部数据解锁
