# Winner 下一条主线 P/W/F 方案与执行回填（Selective Retrieval Gate，2026-03-11）

## 执行回填（2026-03-11）

这份 `P/W/F` 方案已经实际执行到 `P2 probe`，结果应视为止于 `g35` 的负结果归档，不再继续按原计划推进 `g50 / g65 / W / F`。

已完成：

- `P0 score coverage audit`
- `P1 gated build`
- `P2 probe` 的 `ctrl` 与 `g35`

实际结果：

- `ctrl anchor64 geom = 22.3155`
- `ctrl hard geom = 23.1564`
- `g35 anchor64 geom = 18.8861`
- `g35 hard geom = 19.4744`
- `g35 - ctrl anchor64 = -3.4294`
- `g35 - ctrl hard = -3.6820`
- `g35 latency ratio vs ctrl = 1.0097`
- `g35 health no_regression = false`
- targeted buckets 全为负：
  - `rare=-0.8144`
  - `measure=-1.0547`
  - `formula=-0.5161`
  - `marker=-0.9130`

执行结论：

- `A2g selective gate` 当前判负
- `g50 / g65` 不再继续补跑
- 不再继续开 retrieval 同质 gate 变体的 GPU 训练
- 当前工作基线继续保留：
  - retrieval always-on `ctrl`
  - 长口径 compare 仍以 `fallback_180` 为 health-safe baseline

结果归档：

- `reports/taskform_winner_a2g_build_20260311/summary.json`
- `reports/taskform_winner_a2g_probe_20260311/summary.json`
- `reports/taskform_winner_a2g_probe_20260311/gate_report.md`

下一条主线：

- `competition-only pseudo-target / denoising continue smoke`

切线更新（`2026-03-11` 夜间）：

- 这条 pseudo-target 线已经实际跑完并判负：
  - `incumbent anchor64 = 16.5057`
  - `probe anchor64 = 2.5066`
  - `probe hard = 2.7485`
  - `delta vs incumbent = -13.9992`
- 输出形态出现明显塌缩：
  - 重复字符
  - `<gap>` 堆叠
  - instruction echo
- 因此它只说明：
  - 原始 pseudo-target synthetic mix 当前不可放行
  - 不说明 `TAPT / denoising` 本身无效

下一条主线现已更新为：

- `competition-only denoising-only continue smoke`

再次更新（`2026-03-11` 当前执行口径）：

- 仓库里已存在更干净的 fair compare：
  - `reports/taskform_tapt_fair_20260310/summary.json`
- 其中：
  - `T0_tapt_then_supervised - C0_no_tapt anchor64 = -0.5435`
  - `health_t0_vs_c0 no_regression = false`
- 因此 `denoising-only` 也不值得重复开跑

当前真正的下一条主线改为：

- `winner replay / curriculum probe`

## 0. 决策结论

当前不应继续把主 GPU 时间投到：

- `A1 external mix continue-on-wlite`
- 当前候选池上的 `MBR`
- 旧 `L2/L3` 后处理主线

原因已经足够清楚：

- `A1R_P2` 已经跑完，`e5 / e10 / e15` 全部输给 matched control
- 当前 `MBR` 相对 best single 仍为负
- 当前真正稳定、显著转正的主线仍只有 `A2 retrieval-top1`

因此下一条主线改写为：

- **`A2g = retrieval-top1 selective gate`**

目标不是：

- 换 backbone
- 再引入 external mix
- 再堆更多近亲候选给 `MBR`

而是：

- 保留当前 retrieval 主收益
- 压掉低置信近邻带来的坏 hint
- 在不破坏 `rare_name / formula` recall 的前提下，改善 full-val health 与弱桶表现

## 1. 当前冻结基线

正式历史基线仍保持：

- `I0 = incumbent`
  - checkpoint:
    - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
  - decode:
    - `beam=4 / lp=0.7 / max_new_tokens=384`

当前工作基线改为：

- `B0 = fallback_180`
  - 来源：
    - `/workspace/deep-past-/reports/taskform_winner_a2_freeze_20260310/summary.json`
  - 当前分数：
    - `anchor64 reconstructed geom = 23.5415`
    - `full-val reconstructed geom = 19.9035`
    - `hard geom = 20.7888`
  - 当前状态：
    - `candidate_frozen_manual_promote_recommended`

从这一轮起，所有新线都要同时对两套参照汇报：

- 对 `I0`：
  - 作为历史正式 winner 对照
- 对 `B0`：
  - 作为当前最强、最健康的工作基线

纪律补充：

- **不允许把 `fallback_180` 当训练起点**
- `fallback_180` 只用于 compare / promote gate

## 2. 为什么是 retrieval selective gate

当前证据链如下：

- `A2 retrieval-top1` 已显著有效：
  - `R1 vs R0 anchor64 geom = +6.3772`
  - `R1 vs R0 hard subset geom = +7.7677`
  - latency ratio 仅 `1.011x`
- `A2 W-lite` 已大幅超过 `I0`：
  - `full-val reconstructed geom = 19.9956`
  - `hard geom = 20.8432`
- 但 raw `W-lite` full-val health 仍发红：
  - `unique_prediction_ratio_pct` 相对 `I0` 有回落
- `fallback_180` 说明问题主要不是“retrieval 无效”，而是“少量坏输出需要约束”
- targeted bucket 里当前 retrieval 的强弱已分化：
  - 相对 `R0` 为正：
    - `rare_name`
    - `formula`
  - 相对 `R0` 为负：
    - `measure`
    - `marker_rich`

因此下一条最合理的问题不是：

- “要不要继续加更多 retrieval”

而是：

- “能不能只在高置信行上保留 retrieval hint，把低置信 hint 关掉，从而保住主收益并减少副作用”

## 3. 本线唯一变量

这条线只允许改变一个变量：

- **一行是否接收 retrieval top1 hint**

控制条件固定不变：

- model family:
  - `ByT5-small chunk`
- 当前 chunk pipeline
- 当前 datastore：
  - internal parallel train-visible only
- 同 seed
- 同 lr
- 同 batch size
- 同 eval_steps
- 同 decode
- 同 val split

当前轮次明确不做：

- `top3 compressed`
- `kNN interpolation`
- mixed datastore
- external mix
- backbone 改动

## 4. P 阶段

### 4.1 `P0`: score coverage audit（CPU）

先对当前 top1 retrieval score 做覆盖率审计，固定三档 gate：

- `G35`:
  - `top1_score >= 0.35`
- `G50`:
  - `top1_score >= 0.50`
- `G65`:
  - `top1_score >= 0.65`

选择这三档的原因：

- 当前 `val top1_score p50 ~= 0.364`
- 当前 `val top1_score p95 ~= 0.667`

必须输出：

- `score_gate_audit.json`
- `score_gate_coverage.csv`
- `score_gate_bucket_coverage.csv`

必须至少统计：

- train-visible coverage
- val-visible coverage
- `rare_name / measure / formula / marker_rich` coverage
- 各 gate 下 hint 被关闭的比例

`P0` 只是审计，不做 gate。

### 4.2 `P1`: gated processed_dir build

构造四套数据臂：

- `C0 = retrieval_top1_always_on`
- `T35 = retrieval_top1_gate035`
- `T50 = retrieval_top1_gate050`
- `T65 = retrieval_top1_gate065`

构造纪律：

- retrieval datastore 仍只来自 internal parallel
- `fold0 val` 绝对不变
- 训练可见行数与 row order 只允许因 hint on/off 不同而变化
- 同一行在 gate 关闭时：
  - 回退为 no-hint source
- 不允许引入新的 top-k hint 格式
- 不允许把旧 external mix 行带入本线

必须落盘：

- `manifest.json`
- `gate_build_summary.json`
- `processed_dir_manifest.csv`

### 4.3 `P2`: matched smoke

四臂统一跑：

- `180 steps`
- `eval_steps = 45`
- `anchor64 reconstructed`
- `hard subset`
- targeted bucket audit
- output health

只认以下 compare：

- `T35 - C0`
- `T50 - C0`
- `T65 - C0`

本阶段必须额外输出：

- `rare_name / measure / formula / marker_rich`
- `empty / copy / short / repeat / unique`
- `latency`

### 4.4 `P gate`

定义：

- `Tbest = argmax(anchor64 geom among T35/T50/T65)`

放行条件：

- `Tbest - C0 anchor64 geom >= +0.25`
- `Tbest - C0 hard subset geom >= -0.10`
- `health no_regression vs C0 = true`
- `rare_name` 与 `formula` 不能同时明显恶化

优先 review 信号：

- `measure` 或 `marker_rich` 至少一类转正
- 或 full-val proxy health 明显更绿

停机规则：

- 若三组 gate 全部不满足：
  - `reject_stop`
- 若 `0 < delta_anchor < +0.25` 且 health 全绿：
  - `review_stop`
- 若明确过线：
  - `review_to_w`

## 5. W 阶段

### 5.1 `W0`: official bridge 并行补齐（CPU）

从这一轮开始，不再允许把 official bridge 无限后推。

因此 `W` 阶段必须并行补齐：

- `official_metric_probe.json`
- bridge 接线说明
- 若仍未能接入：
  - 必须明确阻塞点

### 5.2 `W1`: 400-step matched compare

只有 `P` 过线才继续。

只保留两臂：

- `C0_W = retrieval_top1_always_on @ 400`
- `Gbest_W = best selective gate @ 400`

统一要求：

- 同 upstream recipe
- 同 budget
- 同 decode
- full-val reconstructed
- official-like
- hard subset
- targeted bucket audit
- latency audit

### 5.3 `W gate`

放行条件：

- `Gbest_W - C0_W full-val reconstructed geom >= +0.25`
- `Gbest_W - C0_W hard geom >= -0.10`
- `health no_regression vs C0_W = true`
- `rare_name / formula` 主收益仍在

review 信号：

- `measure / marker_rich` 任一类改善
- 或重复/唯一性健康明显优于 `C0_W`

停机规则：

- 若 full-val 不过线：
  - `reject_stop`
- 若只在 local 微正、hard 无改善：
  - `review_stop`
- 若 full-val 明显成立：
  - `review_to_f`

## 6. F 阶段

### 6.1 compare 参照

进入 `F` 后必须三路比较：

- `I0 = incumbent`
- `B0 = fallback_180`
- `Gbest_F = selective gate best long candidate`

### 6.2 必须产出

固定产出：

- `full-val reconstructed`
- `official-like`
- `hard subset`
- `targeted bucket audit`
- `changed_rows.csv`
- `gate_report.md`
- `manifest.json`

### 6.3 health surgical

若 `Gbest_F raw` health 发红：

- 必须补一轮与 `A2` 同口径的 health surgical compare
- repaired candidate 与 raw candidate 同时写入 compare 表

### 6.4 promote gate

正式 promote 条件：

- `full-val reconstructed geom >= B0 + 0.15`
- `hard geom >= B0 - 0.10`
- `health no_regression vs I0 = true`

同时必须补充判断：

- 若相对 `B0` 只有局部微正，但错误模式明确互补：
  - 可降级进入 candidate pool
  - 不得直接 promote

停机规则：

- 未超过 `B0`：
  - `review_stop` 或 `candidate_pool_only`
- 明显低于 `B0`：
  - `reject_stop`

## 7. 并行执行纪律

推荐 tmux session：

- `winner_rg_audit`
- `winner_rg_build`
- `winner_rg_probe`
- `winner_rg_wlite`
- `winner_rg_bridge`
- `winner_rg_report`

并行原则：

- GPU 重训练仍串行，轻训练尽量多步并行。
- CPU 审计、bridge、report、bucket 汇总并行
- 每个 session 都必须单独落日志

推荐日志：

- `logs/winner_rg_audit_20260311.log`
- `logs/winner_rg_probe_20260311.log`
- `logs/winner_rg_wlite_20260311.log`
- `logs/winner_rg_bridge_20260311.log`

## 8. 产物命名建议

推荐 report 目录：

- `reports/taskform_winner_a2g_score_audit_20260311`
- `reports/taskform_winner_a2g_build_20260311`
- `reports/taskform_winner_a2g_probe_20260311`
- `reports/taskform_winner_a2g_wlite_20260311`
- `reports/taskform_winner_a2g_promote_20260311`

推荐实验标签：

- `taskform_winner_a2g_ctrl`
- `taskform_winner_a2g_g35`
- `taskform_winner_a2g_g50`
- `taskform_winner_a2g_g65`

## 9. 时间预算

单卡串行、CPU 审计并行条件下，建议预算写成：

- `P0 audit + P1 build`：
  - `20-40 分钟`
- `P2 probe`：
  - `1-1.5 小时`
- `W1 400-step + full-val eval`：
  - `2-3 小时`
- `F compare + surgical + report`：
  - `2-3.5 小时`

因此 wall-clock 预估：

- 只到 `P` 判定：
  - `1.5-2 小时`
- 跑到 `W` 判定：
  - `3.5-5 小时`
- 一路到 `F/promote`：
  - `5.5-8.5 小时`

## 10. 本线实际结果与切线

`A2g` 已于 `2026-03-11` 实际判负，并在 `g35` 后人工止损。

已确认：

- `g35` 相对 `ctrl` 为明显负收益
- 输出健康未通过 `no_regression`
- `rare_name / measure / formula / marker_rich` 全部未给出正信号
- 因此 `g50 / g65` 不再有继续消耗 GPU 预算的必要

下一条主线第一次切到：

- `competition-only pseudo-target / denoising continue smoke`

但该线已在 `2026-03-11` 夜间实际判负，因此当前再切线更新为：

- `competition-only denoising-only continue smoke`

理由：

- `pseudo-target` 的失败形态高度像 synthetic supervision 污染，而不是单纯的 `TAPT` 失败
- 当前最便宜、最干净的下一问不是“再造更多伪标签”
- 而是“只保留 denoising continue，看看 monolingual signal 本身是否有任何可保留增益”

但这条 `denoising-only` 问题实际上也已经被历史 fair compare 回答了：

- `T0_tapt_then_supervised` 相对 `C0_no_tapt` 为负
- 输出健康未通过 `no_regression`

因此当前再往前切一层，真正要执行的是：

- `winner replay / curriculum probe`

## 11. 当前已批准执行队列（2026-03-11）

这份旧 `A2g` 文档当前只保留为切线记录；真正执行口径已经改成：

- 当前主序保持不动：
  - `raw retrieval W-lite long train`
  - `replay25 candidate-pool long train`
- 小实验队列排在主序后面，仍然只做 `probe`
- 再单独挂一条 `post-probe full-val decode` 队列
- 这条 decode 队列最多只吃 `top 1-2` 个赢家

当前批准的小实验：

- `retrieval-top1 + replay25 combo 180-step probe`
- `replay15 / replay20 / replay30` 窄扫
- `A3 cheap revisit`
- `RK_true_hook weak revisit smoke`

桥接规则：

- `combo probe` 直接接 `raw retrieval longtrain`
- `replay band` 直接接 `replay25 longtrain`
- `A3` 只在新的 probe 候选落盘后做 cheap audit
- `post-probe decode` 不追求面面俱到，只追求值卡效率

边界不变：

- 不重开 `A1 external mix`
- 不重开 `A2g selective gate`
- 不重开 `competition-only mono`
- 不重开旧 `MBR formulation`
- `RK_true_hook` 这次只允许最小 formal smoke，不允许扩成新主线
