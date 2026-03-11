# Taskform Winner 阶段实验报告（更新于 2026-03-10）

## 0. 结论先行

截至当前仓库快照，结论已经比较清楚：

- `A1` 外部平行数据主线已经从资产阻塞解锁：本地 `silver external` 已生成并通过 `A1_P0` 审计，当前状态是 `ready_for_mix_build`。
- `A1` 已跑过一轮 `plain fresh-base smoke`，但这轮比较口径错误，现已判为无效，不得用于 winner 主线判断。
- `A2` retrieval 主线已经跑出当前最强的 offline 单模型候选：`retrieval-top1 W-lite` 在 `anchor64 / full-val / hard` 三个口径上都明显超过 incumbent。
- `A2` 的 health surgical review 已经给出可 promote compare 的 health-safe 候选：`fallback_180` 以 `-0.0872` full-val 代价换回 health gate 过线，且 freeze bundle 已落盘，当前状态是 `candidate_frozen_manual_promote_recommended`。
- `A3_P0` diversity audit 是正的，但 `A3_P1/P2` 当前 formulation 已经跑负：`MBR` 没有打过 best single，因此 `A3_P3` 不能继续。
- `RK` 方向的 infra 是正面的，但当前这轮 proxy probe 在 promote 相关的 reconstructed 指标上为负，不能作为下一条主优先 GPU 线。

因此，当前最准确的判断不是“项目又回到无主线状态”，而是：

1. `A2 retrieval-top1 W-lite` 已经证明 retrieval 主线成立，而 `fallback_180` 成为当前更适合 promote compare 的 health-safe 候选。
2. 下一步不该再开 `R3`，也不该沿当前 `MBR` formulation 继续烧预算。
3. `A2 promote compare` 已经收尾；下一条不是恢复旧 `A1_P1/P2`，而是改成 `continue-on-wlite` 的 `A1R_P1/P2 -> W-lite -> promote`。

## 1. 当前记分板

当前正式 incumbent 仍是：

- checkpoint:
  - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- decode:
  - `beam=4 / lp=0.7 / max_new_tokens=384`

当前最重要的分数对比如下：

- incumbent full-val reconstructed:
  - `geom = 14.3323`
- incumbent hard:
  - `geom = 13.7161`
- incumbent anchor64:
  - `geom = 16.5057`

- retrieval-top1 smoke anchor64:
  - `geom = 22.2513`
- retrieval-top1 W-lite anchor64:
  - `geom = 23.5422`
- retrieval-top1 W-lite full-val reconstructed:
  - `geom = 19.9956`
- retrieval-top1 W-lite hard:
  - `geom = 20.8432`

对应增益：

- `W-lite vs incumbent anchor64 = +7.0365`
- `W-lite vs incumbent full-val = +5.6633`
- `W-lite vs incumbent hard = +7.1271`

主分层面，这已经不是“小正”，而是当前仓库中最明确的一次大正。

## 2. A1：外部平行数据 / 域扩展混训

这条线现在已经完成了 `silver external build + A1_P0`，产物如下：

- `/workspace/deep-past-/reports/taskform_winner_a1_silver_build_20260310/summary.json`
- `/workspace/deep-past-/data/external/oracc_parallel.csv`
- `/workspace/deep-past-/reports/taskform_winner_a1_20260310/manifest.json`
- `/workspace/deep-past-/reports/taskform_winner_a1_20260310/source_registry.csv`
- `/workspace/deep-past-/reports/taskform_winner_a1_20260310/mix_plan.csv`
- `/workspace/deep-past-/reports/taskform_winner_a1_20260310/overlap_audit.json`
- `/workspace/deep-past-/reports/taskform_winner_a1_20260310/dedup_manifest.json`

已确认的 internal 基线规模：

- `train_visible_rows = 4064`
- `val_visible_rows = 1225`
- `val_parent_rows = 313`

当前状态：

- `status = ready_for_mix_build`

这意味着：

- `silver external` 已经长出并落盘。
- `A1_P0` 已确认：
  - `fold0_val_exact_overlap_rows = 0`
  - `test_exact_overlap_rows = 0`
  - `train_exact_overlap_rows = 0`
- `E10 / E30 / E50` 三档规模都已够量。

但这里必须补一个纠错：

- 已执行的 `reports/taskform_winner_a1_smoke_20260310/summary.json`
  - 用的是 plain processed dir
  - 从 fresh `ByT5-small + LoRA` 起训
  - 没接当前 `retrieval-top1 W-lite` adapter
- 因此那轮 `3.x geom` 只能视为“比例噪声先验”，不能拿来判 `A1` 能不能提升当前 winner

所以对 `A1` 的正确处理现在不是直接沿旧 smoke 继续，而是改成：

- `A1R_P1`: plain mixed rows -> retrieval mixed rows 构造
- `A1R_P2`: matched continue probe
- upstream init:
  - `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model`
- ratio sweep:
  - `C0 / E5 / E10 / E15`
- long path:
  - `A1R_Wlite -> A1R_F/promote`

## 3. A2：retrieval / 记忆增强 chunk 建模

### 3.1 已完成内容

这条线当前已经完整走过：

- `A2_P0`: internal-only retrieval audit
- `A2_P1/P2`: `R0` matched control 与 `R1 retrieval-top1` smoke
- `A2_P3`: targeted bucket audit
- `W-lite`: `retrieval-top1 @ 400 steps`
- full-val / hard / latency / memory / support bundle

主要产物：

- smoke 对照：
  - `/workspace/deep-past-/reports/taskform_winner_a2_retrieval_eval_20260310/summary.json`
- W-lite 总结：
  - `/workspace/deep-past-/reports/taskform_winner_a2_retrieval_wlite_eval_20260310/summary.json`
- gate report：
  - `/workspace/deep-past-/reports/taskform_winner_a2_retrieval_wlite_eval_20260310/gate_report.md`
- support bundle：
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/summary.json`

### 3.2 为什么说它已经是当前最强单模型候选

和 incumbent 比，`retrieval-top1 W-lite` 的主分是显著领先的：

- `anchor64: 23.5422 vs 16.5057`
- `full-val reconstructed: 19.9956 vs 14.3323`
- `hard: 20.8432 vs 13.7161`

检索支撑证据也成立：

- train-visible retrieval hit ratio:
  - `100.00%`
- val-visible retrieval hit ratio:
  - `100.00%`
- val top1 score mean:
  - `0.3864`
- val top1 target chrF++ mean:
  - `28.5124`

说明这不是“偶然多训了一轮”，而是 retrieval 记忆本身给到了有效信号。

### 3.3 为什么它还没有被直接写成正式新 winner

当前 gate 结果是：

- `status = review_for_f`

原因不是分数不够，而是输出健康度没有自动过关。

full-val 侧最关键的健康项：

- incumbent `unique_prediction_ratio_pct = 64.6531`
- W-lite `unique_prediction_ratio_pct = 63.1020`
- incumbent `pred_shorter_than_half_ref_ratio_pct = 8.7347`
- W-lite `pred_shorter_than_half_ref_ratio_pct = 8.8163`

也就是说：

- W-lite 的主分大幅更高。
- 但 full-val 重复和唯一性并没有自动优于 incumbent。
- 自动 gate 因而选择保守地停在 `review_for_f`。

这类风险不该被忽略，但也不能误读成“这条线不成立”。它更准确的含义是：

- `A2` 已经成功把单模型上限抬高。
- 接下来要么人工复核并放入正式 promote compare，
- 要么把它先作为 `A3` 的 strongest single candidate，利用多样候选选择器去进一步化解重复问题。

### 3.4 latency / memory / bundle 状态

`A2` 的支撑件已经齐了：

- cache hit:
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/retrieval_cache_hit_stats.json`
- neighbor quality:
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/nearest_neighbor_quality_audit.json`
- latency:
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/latency_report.json`
- memory:
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/memory_usage_report.json`
- official-like template:
  - `/workspace/deep-past-/reports/taskform_winner_a2_support_20260310/official_like_template.json`

其中最需要记住的工程代价是：

- incumbent full-val decode:
  - `2892.38s`
- retrieval W-lite full-val decode:
  - `4970.81s`
- latency ratio:
  - `1.7186x`

这仍然在纪律里的 `<= 1.8x` 门内，但已经接近上界，说明后续 `A3/MBR` 必须尽量复用缓存，不能无节制扩 decode 成本。

### 3.5 A2 health surgical review：为什么当前推荐的是 `fallback_180`

health surgical probe 与 promote compare 已经完成，主产物如下：

- `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/summary.json`
- `/workspace/deep-past-/reports/taskform_winner_a2_health_surgical_20260310/manual_review.md`
- `/workspace/deep-past-/reports/taskform_winner_a2_promote_compare_20260310/summary.json`
- `/workspace/deep-past-/reports/taskform_winner_a2_freeze_20260310/summary.json`
- `/workspace/deep-past-/reports/taskform_winner_a2_freeze_20260310/gate_report.md`

最终推荐的不是 raw `W-lite`，而是：

- `fallback_180`
  - 对 full-val 中高频重复的 generic chunk 输出，回退到 incumbent 对应 chunk
  - 触发条件：
    - `repeat >= 5`
    - 或 `repeat >= 4` 且预测长度 `<= 180`

这条候选的关键指标是：

- raw W-lite full-val reconstructed:
  - `19.9908`
- `fallback_180` full-val reconstructed:
  - `19.9035`
- incumbent full-val reconstructed:
  - `14.3323`
- `fallback_180` hard:
  - `20.7888`
- `fallback_180` full-val health `no_regression vs incumbent`:
  - `True`

换句话说：

- 它只牺牲了 `0.0872` 的 full-val reconstructed `geom`
- 但把 health gate 从红灯拉回了可比较状态

这个修复也不是大面积改写：

- changed chunk rows:
  - `58 / 1225 = 4.7347%`
- changed parent rows:
  - `29`
- changed original rows:
  - `8`
- changed ratio rows:
  - `23`
- changed short-aligned rows:
  - `27`

也就是：

- 真正动到 original chunk 的只是一小撮高频 generic 输出
- 大多数替换发生在 ratio / short-aligned 行上

这条线也有代价，且必须承认：

- 在被替换的 `58` 行上，raw W-lite 的局部 `geom = 13.2549`
- `fallback_180` / incumbent 在同一 changed subset 上是 `11.5200`

所以它不是“更会翻这些行”，而是“为了过 health gate，接受在少量高频 generic 行上回退到更稳的 incumbent 表达”。这也是为什么它应该被写成：

- promote compare 的 frozen candidate
- 而不是新的训练 checkpoint

- promote compare candidate
- 而不是新训练出的正式 checkpoint

## 4. A3：多样候选池 / MBR / ensemble

`A3_P0` 已经完成，结果是正面的：

- `/workspace/deep-past-/reports/taskform_winner_a3_diversity_20260310/summary.json`

关键结果：

- pool `unique_candidate_ratio_pct = 92.1569`
- `rows_all_unique_ratio_pct = 76.4706`
- `incumbent vs retrieval_smoke exact_overlap_ratio_pct = 0.0`
- `incumbent vs retrieval_wlite exact_overlap_ratio_pct = 0.0`
- `retrieval_smoke vs retrieval_wlite exact_overlap_ratio_pct = 23.5294`

这说明：

- incumbent 与 retrieval 系列候选确实互补，不是同一句话换个标点。
- `A3` 的前置 gate 已经过线。

但后续真正跑完的 `A3_P1/P2` 结果是负的：

- `/workspace/deep-past-/reports/taskform_winner_a3_mbr_probe_20260310/summary.json`
- best single:
  - `retrieval_wlite_repaired`
  - `anchor64 geom = 24.6330`
- `MBR geom = 23.6760`
- `delta = -0.9570`
- `status = review_stop`

也就是说：

- diversity evidence 本身没错
- 但当前这套 `MBR` utility / candidate mix 并没有把多样性真正变成更高分
- 因此 `A3_P3` pairwise ensemble 当前不该继续

对 `A3` 的正确结论应写成：

- `A3_P0` 给了“以后值得再回来的证据”
- `A3_P1/P2` 当前 formulation 已证伪
- 在出现新的强候选轴之前，先 park，不抢主序

## 5. RK：kNN-MT logits interpolation proxy probe

这条线的 infra 结论要分开看。

正面部分：

- datastore `4064` 行
- target token 总数 `827132`
- 估算 fp16 key store `2.27 GiB`
- retrieval latency `1.05 ms/query`
- cache assembly `11.69 ms/query`
- `top1 target chrF++ mean = 26.19`
- `oracle topk target chrF++ mean = 32.61`

这些结果说明：

- 真正的 decoder-side `kNN-MT` 方向在工程上可做
- 而且存在可观 oracle gap

但当前这轮 proxy probe 的实验结论是负的：

- baseline anchor64 reconstructed geom:
  - `23.5422`
- best alpha proxy reconstructed geom:
  - `22.4518`
- `delta = -1.0904`

也就是说：

- 现在这条 generation-side proxy interpolation 没有打过当前 retrieval W-lite baseline
- 它不能作为下一条主优先级实验

所以对 `RK` 的正确处理是：

- 保留为“后续真实 decoder hook”方向
- 当前不抢 `A3` 的主序 GPU 预算

## 6. 为什么下一步不是 R3，也不是继续烧 RK

`R3` 当前 formulation 已经被 probe 明确卡住：

- 平均输入增量约 `+742` token
- `p95` 约 `+814`
- `5017 / 5289` 行超过 `max_source_length = 640`

这说明不是“top3 没价值”，而是“当前 hint 格式不合格”。
所以它现在只能重做压缩策略，不能直接开 smoke。

`RK` 当前也不该优先，因为：

- 只是 proxy，不是真正 decoder-state hook
- reconstructed 指标明确低于 baseline

因此，当前最该避免的是：

- 在 `R3` 上继续浪费训练预算
- 把 `RK proxy` 的 infra-positive 误写成实验正收益

## 7. 阶段性判断与下一步

阶段性判断：

1. 当前最强训练模型已经从 incumbent 变成了 `A2 retrieval-top1 W-lite`。
2. 当前最适合进入 promote compare 的候选不是 raw W-lite，而是 `fallback_180`。
3. `fallback_180` 的 promote / freeze 包已经落盘，`A2_F_review` 不再是待办项。
4. `A3_P0` 虽然为正，但 `A3_P1/P2` 当前 formulation 已经失败，不能继续推到 `A3_P3`。
5. `A1` 已经从资产缺失切换到 `ready_for_mix_build`。
6. `RK` 暂停在“后续 selective / stateful hook 再开”的状态。

下一步优先级应当写成：

1. `A1R_P1`: 构造 `plain mixed -> retrieval mixed` 的 continue-ready processed_dir
2. `A1R_P2`: 跑 matched continue probe，对照 `C0`
3. `A1R_Wlite`: 只放行最佳比例进入 400-step long compare
4. `A1R_F/promote`: full-val raw + repaired candidate compare，对照 `fallback_180`
5. `RK_true_hook_revisit`: 只有在 selective gating / fallback routing / 更强 retrieval state 条件下才重开
6. `A3`: 只有出现新的强候选轴后才回看，不再沿当前 formulation 继续烧预算
7. `A4`: 继续锁住，不前置

一句话总结：

当前主线已经从“先证明 retrieval 有没有用”切换成“retrieval 已经成立，health-safe 候选已冻结；而 `A1` 若要再开，必须走 `continue-on-wlite` 的公平口径，而不是再看那轮 fresh-base smoke。” 
