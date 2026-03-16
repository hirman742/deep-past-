# 公开模型上游机制逆向与训练计划
## 2026-03-14 training plan, updated 2026-03-15

本稿承接：

- [public_model_public_weight_next_step_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_public_weight_next_step_plan_2026-03-14.md)
- [public_model_public_weight_anchor_and_longtrain_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_public_weight_anchor_and_longtrain_plan_2026-03-13.md)
- [public_model_r15_strict_repro_correction_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_r15_strict_repro_correction_plan_2026-03-13.md)
- [public_model_repro_design_2026-03-13.md](/workspace/deep-past-/docs/public_model_repro_design_2026-03-13.md)
- [next_step_discipline_2026-03-13.md](/workspace/deep-past-/docs/next_step_discipline_2026-03-13.md)

本稿只做两件事：

1. 明确“接下来怎么问上游问题，才是在搞懂 `public model` 怎么来”
2. 基于当前证据，给出新的 `Track B` 训练计划

本稿不直接启动训练，不改写当前主线，不把 `inconclusive` 写成机制结论。

当前文档结构改为两层：

1. `Update` 区域负责补充已经落地的训练记录与冻结口径
2. 正文同时保留：
   - `2026-03-15-2` 的当前下一步计划
   - 当前已冻结事实与解释边界的统一口径
   - `2026-03-14` 的历史训练计划

## Update · 2026-03-14T17:06:24+00:00

`U3` 现已落地，本稿应视为 `U3` 之前的计划快照。

- `U3` 上游：[/workspace/deep-past-/reports/public_model_r19_public_taptbroad_20260314/route_decision.md](/workspace/deep-past-/reports/public_model_r19_public_taptbroad_20260314/route_decision.md)
- `U3` 下游：[/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/route_decision.md](/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/route_decision.md)
- 正式 freeze：`ckpt300 / geom 40.1336 / delta vs plain continuation pilot = -0.0930 / verdict = inconclusive`
- 后续判断改看：[/workspace/deep-past-/docs/public_model_u3_freeze_and_next_step_2026-03-14.md](/workspace/deep-past-/docs/public_model_u3_freeze_and_next_step_2026-03-14.md)

## Update · 2026-03-14T21:16:23+00:00

`U4/H4-H6` 队列与 guarded `R26` 现已全部落地，本稿可作为 `R26` 之后的决策快照使用。

先说明一个编号问题：

- 本稿前文把 `H5` 写成 `unlabeled corpus scope`，把 `H6` 写成 `architecture / training shape`
- 但 `4.4` 标题又把“更接近公开对象的训练形态”写成了 `H5`
- 以下回填统一按执行口径理解：
  - `H4` = normalization / task-form
  - `H5` = unlabeled corpus scope
  - `H6` = training shape / wider adaptation proxy
  - `R26` 虽命名为 `H5-proxy long confirm`，实际是在对当前 `H6 proxy` 候选做 guarded long confirm

### U1~U3 原始记录快照

这里补一份便于下一轮决策的“原始三枪”快照。先说明编号：

- 按原始 `2026-03-14` 计划标签，已实际落地的是：
  - `U1 = R18 official-only medium`
  - `U3 = R19 broader text-only medium`
  - `U3-strong = R20 broader text-only strong`
  - 原计划 `U2 strong official-only` 没有单独执行
- 若只按已经实际发生的三次 `Track B` 主训练流水顺序记，则可把 `R18 / R19 / R20` 视为“实际执行的 `U1 / U2 / U3`”

为避免遗漏，这里同时保留“原始计划标签”和“实际执行顺序”两层说明。

1. 实际执行 `U1`（`R18 / 原始计划 U1`）：
   - 方向：`official-only medium -> same pilot`
   - 结果：`ckpt300 / geom 39.4724 / delta vs plain continuation pilot = -0.7542 / verdict = negative`
   - 结论：`H3a official-only medium` 不成立为默认后续
   - 文件：[/workspace/deep-past-/reports/public_model_r18_public_taptmed_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r18_public_taptmed_cont_pilot_20260314/driver_results.json)
2. 实际执行 `U2`（`R19 / 原始计划 U3`）：
   - 方向：`broader text-only medium -> same pilot`
   - 结果：`ckpt300 / geom 40.1336 / delta vs plain continuation pilot = -0.0930 / verdict = inconclusive`
   - 结论：它明显好于 `R18`，但仍未超过 plain continuation pilot，只能冻结为弱方向信号
   - 文件：[/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/driver_results.json)
3. 实际执行 `U3`（`R20 / 原始计划 U3-strong`）：
   - 方向：`broader text-only strong -> same pilot`
   - 结果：`ckpt300 / geom 39.5334 / delta vs plain continuation pilot = -0.6932 / verdict = negative`
   - 结论：把 `H3b` 预算拉强后没有继续变好，反而重新回到负面
   - 文件：[/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_cont_pilot_20260314/driver_results.json)

这三枪合在一起的冻结表述应是：

- `U1~U3` 都已训练过
- 它们都提供了方向信息
- 但它们都没有给出足以替代 `public-weight continuation` 的收益
- 因此它们的价值主要在于缩小假设空间，而不是直接成为新主线

### 已落地结果快照

1. `H3a official-only medium` (`R18 / U1`)：
   - 结果：`ckpt300 / geom 39.4724 / delta vs plain continuation pilot = -0.7542 / verdict = negative`
   - 文件：[/workspace/deep-past-/reports/public_model_r18_public_taptmed_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r18_public_taptmed_cont_pilot_20260314/driver_results.json)
2. `H3b broader text-only medium` (`R19 / U3`)：
   - 结果：`ckpt300 / geom 40.1336 / delta vs plain continuation pilot = -0.0930 / verdict = inconclusive`
   - 文件：[/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/driver_results.json)
3. `H3b broader text-only strong` (`R20 / U3-strong`)：
   - 结果：`ckpt300 / geom 39.5334 / delta vs plain continuation pilot = -0.6932 / verdict = negative`
   - 文件：[/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_cont_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_cont_pilot_20260314/driver_results.json)
4. `H4` 单轴 probe (`R21/R22/R23`)：
   - `R21 normprobe`：`geom 40.2266 / delta 0.0000 / verdict inconclusive`
   - `R22 task-form probe`：`geom 40.2266 / delta 0.0000 / verdict inconclusive`
   - `R23 norm + task-form combo`：`geom 40.2266 / delta 0.0000 / verdict inconclusive`
   - 汇总：[/workspace/deep-past-/reports/public_model_r21plus_h4h6_queue_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r21plus_h4h6_queue_20260314/driver_results.json)
5. `H6 proxy` (`R24`)：
   - 结果：`ckpt300 / geom 40.5412 / delta vs plain continuation pilot = +0.3146 / verdict = positive`
   - 文件：[/workspace/deep-past-/reports/public_model_r24_public_h6proxy_ffn_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r24_public_h6proxy_ffn_pilot_20260314/driver_results.json)
6. `H4 x H6 combo` (`R25`)：
   - 结果：`ckpt300 / geom 40.5412 / delta vs plain continuation pilot = +0.3146 / verdict = positive`
   - 解释：与 `R24` 持平，没有显示出 `H4` 的额外叠加收益
   - 文件：[/workspace/deep-past-/reports/public_model_r25_public_h4h6_combo_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r25_public_h4h6_combo_pilot_20260314/driver_results.json)
7. guarded long confirm (`R26`, source = `R24`)：
   - 结果：`ckpt600 / geom 40.4099 / delta vs plain continuation pilot = +0.1833 / delta vs incumbent long 40.4028 = +0.0071 / verdict = inconclusive`
   - 文件：[/workspace/deep-past-/reports/public_model_r26_public_h5proxy_longconfirm_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r26_public_h5proxy_longconfirm_20260314/driver_results.json)

### 对下一轮决策的直接含义

当前可以先冻结成下面四条：

1. `H3` 当前不成立为主要解释轴
   - `official-only medium` 明确为负
   - `broader text-only medium` 只有弱方向信号
   - 一旦加大到 `strong`，结果重新回落为负
2. 当前安全范围内的 `H4 normalization / task-form` 没有给出净增益
   - `R21/R22/R23` 全部与 plain continuation pilot 打平
   - 因此目前没有证据支持“`H4` 是公开强度主要来源”
3. `H6 / training-shape proxy` 是目前最强的上游线索
   - `R24` 在同预算 pilot 上给出明确正信号
   - 这说明更宽的参数塑形历史，至少比单独 `H3/H4` 更接近有用方向
4. 但 `H6 proxy` 还不足以改写主线
   - `R26` 虽然健康，而且 long best `40.4099` 贴住了 incumbent `40.4028`
   - 但相对 incumbent long 的增量只有 `+0.0071`
   - 按既定 `+0.2` promote 线，它只能写成 `inconclusive`，不能 promote

因此，如果下一轮仍以 `Track B upstream reverse-engineering` 为目标，更合理的决策应是：

1. 正式 freeze：
   - `H3` 当前不成立为默认后续
   - `H4` 当前不成立为默认后续
   - `H6 proxy = pilot positive, long inconclusive`
2. 下一轮优先考虑：
   - 更高保真的 `H6 / training-shape` 审计或 proxy 设计
   - 而不是回头继续堆更弱的 `H3/H4` 轻量变体
3. 主线口径继续保持：
   - `public-weight continuation` 仍是 stable incumbent
   - `Track B` 新结果只能写成“更接近因果线索”，不能写成“已经复现公开 monster”

以下正文仍保留 `2026-03-14` 当时的原始计划脉络；若与上方 `Update` 区域冲突，以最新时间戳的 `Update` 为准。

## Update · 2026-03-15T03:45:23+00:00

`2026-03-15` 新计划中的 `V1~V4` 已实际落地，`V5` 冻结文档也已补齐；本稿现在可视为 `R27` 收口后的决策快照。

- `V1` 审计与设计：[/workspace/deep-past-/docs/public_model_h6_audit_and_r27_design_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_audit_and_r27_design_2026-03-15.md)
- `V2~V4` 自动队列：[/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md](/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md)
- `R27` 正式结果：[/workspace/deep-past-/reports/public_model_r27_public_h6proxy_ffn_rank32_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r27_public_h6proxy_ffn_rank32_pilot_20260315/driver_results.json)
- `R28` 未启动；原因是 `R27` pilot 未通过 `healthy + positive vs plain continuation pilot + not weaker than R24` gate
- 正式 freeze：`ckpt300 / geom 40.3165 / delta vs plain continuation pilot = +0.0899 / delta vs R24 pilot = -0.2247 / verdict = inconclusive / healthy = true`
- 后续判断改看：[/workspace/deep-past-/docs/public_model_h6_r27_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r27_freeze_and_next_step_2026-03-15.md)

### 对下一轮决策的直接含义

这里可以先冻结成下面四条：

1. `R24` 仍是当前 `H6` 线上最强 pilot
   - `geom 40.5412` 依然高于 `R27` 的 `40.3165`
2. “只把当前 FFN LoRA recipe 的容量做大”不是更好的高保真 proxy
   - `R27` 相对 plain continuation pilot 仍有弱正信号
   - 但它没有放大 `R24` 的收益，反而回落了 `0.2247`
3. 因此当前 `H6` 轴应冻结为：
   - `R24 = pilot positive`
   - `R26 = long inconclusive`
   - `R27 = higher-capacity pilot inconclusive, stop`
4. 如果还要继续 `Track B`
   - 下一枪必须换新的 `training-shape` 单轴变量
   - 而不是继续在同一套 `FFN LoRA rank/alpha` 上做更大的容量扩张

若与本次回填快照冲突，以本次 `Update · 2026-03-15T03:45:23+00:00` 为准。

## Update · 2026-03-15T05:09:22+00:00

`R29` 现已完成；本稿现在可视为 `R29` 收口后的 `Track B / H6` 决策快照。

- `R29` quick analysis：[/workspace/deep-past-/docs/public_model_h6_r29_inproj_quick_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_inproj_quick_analysis_2026-03-15.md)
- `R29` 正式结果：[/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/driver_results.json)
- `R29` 正式 freeze：[/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md)
- 正式 freeze：`ckpt300 / geom 40.4669 / delta vs plain continuation pilot = +0.2403 / delta vs R24 pilot = -0.0743 / delta vs R27 pilot = +0.1504 / verdict = positive / healthy = true`
- `R29` 没有进入 long confirm；原因不是健康或 verdict 问题，而是它仍未超过当前 `R24`

### 现在的结论能够说明什么

当前可以比较硬地说明下面四条：

1. `H6` 的正信号不是“必须整块 FFN 一起动”才会出现
   - `R29 = q/k/v/o + wi_0/wi_1` 仍然相对 plain continuation pilot 给出 `+0.2403`
   - 因此 `R24` 的正信号里，`wi_0/wi_1` 这部分很可能已经解释了主要份额
2. 但 `wi_0/wi_1` 还不能解释 `R24` 的全部收益
   - `R29` 仍然低于 `R24` `0.0743`
   - 所以剩余收益仍可能来自：
     - `wo`
     - 或“整块 FFN 一起动”的组合效应
3. `module placement` 当前看起来比“继续把同一 recipe 容量做大”更重要
   - `R27` 只是放大 `rank/alpha`，结果掉到 `40.3165`
   - `R29` 回到 `R24` 的容量口径，但改成更干净的 module split，结果回升到 `40.4669`
   - 这更像是在说：当前真正有信息增益的不是 raw capacity，而是 `H6` 内部的模块覆盖/placement
4. `Track B` 目前仍然只能写成“更接近因果线索”
   - `R29` 是 `positive`
   - 但它没有改写 `R24` 仍是 best pilot 这一事实
   - 因此当前还不能写成“已经解释了 public model 的形成机制”

### 现在的结论还不能说明什么

当前仍然不能直接说明：

1. `wo` 一定就是剩余增益的唯一来源
2. 只要把 `wi_0/wi_1` 挂上，就已经足以复制 `R24`
3. 当前 `H6` 线已经值得开新的 long confirm

更准确的口径应是：

- `R29` 证明了 `R24` 的正信号主要不是来自“纯 attention-only baseline”，而是明确包含了 FFN 输入侧模块的贡献；但它还没有把剩余增益归因到单一残差模块上

### 对下一轮决策的直接含义

这里可以再冻结成下面三条：

1. 当前 `H6` 轴的局部排序变为：
   - `R24 = 40.5412`
   - `R29 = 40.4669`
   - `R27 = 40.3165`
   - baseline = `40.2266`
2. 因此如果继续 `Track B`
   - 下一枪最值得做的是 `wo-only` 或同等级别的剩余 module split
   - 而不是 long confirm
   - 也不是继续扩 `rank/alpha`
3. 主线纪律保持不变：
   - 只有当新 `H6` pilot 至少不弱于 `R24`
   - 才值得继续谈 long confirm

若与本次回填快照冲突，以本次 `Update · 2026-03-15T05:09:22+00:00` 为准。

## Update · 2026-03-15T05:38:02+00:00

`2026-03-15-2` 现已按计划启动，当前 live probe 为 `R30`。

- live session: `pub_h6outproj_pilot`
- `R30` quick analysis：[/workspace/deep-past-/docs/public_model_h6_r30_outproj_quick_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_outproj_quick_analysis_2026-03-15.md)
- `R30` live config：[/workspace/deep-past-/configs/public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml)
- `R30` live report dir：[/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315)
- live status：[/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_status.json](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_status.json)
- 当前阶段：`train_pilot`
- 已通过 preflight：`passed`

`R30` 的固定设计口径是：

- `q/k/v/o + wo`
- 不挂 `wi_0/wi_1`
- `r=16 / alpha=32`
- 同预算 `300-step pilot`

在 `R30` 完成前，这一条 `Update` 只表示“已启动且已通过 preflight”，不表示结果结论。

## Update · 2026-03-15T06:03:02+00:00

`R30` 现已完成；本稿现在可视为 `R30` 收口后的 `Track B / H6` 决策快照。

- `R30` quick analysis：[/workspace/deep-past-/docs/public_model_h6_r30_outproj_quick_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_outproj_quick_analysis_2026-03-15.md)
- `R30` 正式结果：[/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_results.json)
- `R30` 正式 freeze：[/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md)
- 正式 freeze：`ckpt300 / geom 40.4032 / delta vs plain continuation pilot = +0.1766 / delta vs R24 pilot = -0.1380 / delta vs R29 pilot = -0.0637 / delta vs R27 pilot = +0.0867 / verdict = inconclusive / healthy = true`
- `R30` 没有进入 long confirm；原因不是健康问题，而是它既未达到 `positive` 门槛，也明显没有超过 `R24`

### 现在的结论能够说明什么

当前可以比较硬地说明下面四条：

1. `wo-only` 并不是完全空的
   - `R30` 相对 plain continuation pilot 仍有 `+0.1766`
   - 说明 `wo` 可能有补充贡献
2. 但 `wo-only` 当前也不足以解释 `R24` 的主效应
   - `R30 40.4032` 低于 `R29 40.4669`
   - 这更支持 `wi_0/wi_1` 是 `R24` 的主要正增益来源
3. 当前更像是：
   - `wi_0/wi_1` 提供主效应
   - `wo` 提供次效应，或参与 `wi + wo` 组合效应
4. `H6` 的局部排序现在可冻结为：
   - `R24 = 40.5412`
   - `R29 = 40.4669`
   - `R30 = 40.4032`
   - `R27 = 40.3165`
   - baseline = `40.2266`

### 现在的结论还不能说明什么

当前仍然不能直接说明：

1. `wo` 完全没有贡献
2. `R24 - R29` 的剩余增益已经被确定归因为纯组合效应
3. 当前 `H6` 线已经值得开新的 long confirm
4. 我们已经解释了 `public model` 的形成机制

更准确的口径应是：

- `R29 + R30` 这对 split 一起把 `R24` 的解释空间压缩到了“主效应更像 wi，剩余更像 wo 的补充项或 wi+wo 组合效应”，但还没有形成对 `public model` 的完整机制解释

### 对下一轮决策的直接含义

这里可以再冻结成下面三条：

1. `R30` 完成后，当前 cheapest `H6` split 对照组已基本齐备
2. 因此如果继续 `Track B`
   - 不应再做更大 `rank/alpha`
   - 也不应因为 `R30` 接近 incumbent long 就开新的 long confirm
3. 更合理的后续只剩两类：
   - 把当前 `R24 / R29 / R30` 快照作为阶段性结论冻结
   - 仅在存在新的 interaction-focused `H6` 设计时，再考虑下一枪

若与本次回填快照冲突，以本次 `Update · 2026-03-15T06:03:02+00:00` 为准。

## Update · 2026-03-15T06:25:47+00:00

`R24 / R29 / R30` 的 checkpoint-level adapter audit 现已完成；当前对 `public model` 的解释已从“模块 split 结果”推进到“热点子回路”层面。

- root-cause audit memo：[/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
- audit report：[/workspace/deep-past-/reports/public_model_h6_adapter_audit_20260315/adapter_audit.json](/workspace/deep-past-/reports/public_model_h6_adapter_audit_20260315/adapter_audit.json)

当前最值得冻结的新增结论是：

1. 当前 `H6` 的主信号确实是 `FFN` 主导，不是 attention 主导
2. 当前热层并不分散，而是高度集中在：
   - `decoder block 5`
   - `decoder block 4`
   - `encoder block 13~17`
3. `wi_0/wi_1` 更像主 computation branch
4. `wo` 更像 readout / consolidation branch

这意味着：

- 当前对 `public model` 的最佳解释，已经更像“特定 FFN 子回路的训练形态塑形”
- 而不再只是“某个模块 split 恰好分更高”

## Update · 2026-03-15T06:36:00+00:00

`decoder block 5` 与 `encoder 13~17` 的 layer-local circuit audit 现已完成；当前口径已进一步收紧到“FFN 主导 + attention 输出侧共现”的局部回路层面。

- local circuit memo：[/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)
- local circuit report：[/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json](/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json)

当前最关键的新增结论是：

1. `decoder block 5 + encoder 13~17` 合计占：
   - `R24 61.08%`
   - `R29 61.38%`
   - `R30 58.00%`
   的 total adapter energy
2. 在这些层里，FFN 始终主导，但 attention `o/v` 与 FFN 总量呈高度同步
3. `o` consistently 强于 `v`
4. `R24` 的局部 profile 其实更像 `R30`，但性能更像 `R29`

这条反差意味着：

- `wo` 更像吸收大量局部更新、让 profile 看起来更像 full probe 的 readout 支路
- `wi` 更像真正决定收益的高因果效率 computation 支路

## Update · 2026-03-15T07:08:40+00:00

`decoder block 5` 与 `encoder 16/17` 的 module-pair audit 现已完成；当前 `H6` 口径已从“热点层 + 局部 profile”进一步压缩到“哪些 `FFN <-> attention` pair 更像高收益 interaction clue”。

- module-pair memo：[/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md)
- module-pair report：[/workspace/deep-past-/reports/public_model_h6_module_pair_audit_20260315/module_pair_audit.json](/workspace/deep-past-/reports/public_model_h6_module_pair_audit_20260315/module_pair_audit.json)

当前最值得冻结的新增结论是：

1. `decoder block 5` 里，`cross.o` 与 `self.o` 的确稳定贴着热点 FFN 走
   - 但更接近收益主支路的是 `wi_1 + o`
   - 更接近 full-probe 局部形状的是 `wo + o`
2. `encoder 16/17` 里，`self.o` 在 `R24 / R29 / R30` 三条线上都 consistently 强于 `self.v`
   - 说明 `o` 更像热点 FFN 的同层 readout companion
   - `v` 更像次级响应
3. pair-level 候选优先级已可初步冻结为：
   - `encoder 17 / wi_1 + self.o`
   - `encoder 16 / wi_0 + self.o`
   - `decoder 5 / wi_1 + cross.o`
4. 最不该误判成主效应候选的是 `wo`-anchored pair
   - 它们在 `R30` 中往往更大
   - 但 `R30` 的总收益仍低于 `R29`

这意味着：

- 如果还要继续 `Track B`，下一枪必须是新的 `interaction-focused H6` 设计
- 且应优先围绕 `wi`-anchored pair，而不是 `wo`-anchored pair
- 当前最合理的问题已经不是“哪一支更大”，而是“哪一个 `wi + o` interaction 更接近真正的高因果效率 computation branch”

## Update · 2026-03-15T07:22:52+00:00

热点层 attention compensation audit 现已完成；当前对 `attention o/v` 的口径已从“稳定共现”进一步收紧到“更像补偿性陪跑，而不是主驱动”。

- attention compensation memo：[/workspace/deep-past-/docs/public_model_h6_attention_compensation_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_attention_compensation_analysis_2026-03-15.md)
- attention compensation report：[/workspace/deep-past-/reports/public_model_h6_attention_compensation_audit_20260315/attention_compensation_audit.json](/workspace/deep-past-/reports/public_model_h6_attention_compensation_audit_20260315/attention_compensation_audit.json)

当前最值得冻结的新增结论是：

1. 在 `decoder block 5 + encoder 13~17` 六个热点层里，`R24` 的 attention `o/v total` 在 `6/6` 层里都是最低
2. 这不是因为 `R24` 的局部更新整体更小
   - `R24 local_total` 只在 `1/6` 层里最低
   - `R24 ffn_total` 在 `6/6` 层里都不是最低
3. 六层合计的 attention share 进一步说明：
   - `R24 = 17.39%`
   - `R29 = 30.66%`
   - `R30 = 30.49%`
4. 因此当前更像是：
   - split runs 需要更多 attention 陪跑来补位
   - full probe `R24` 反而在更低 attention 陪跑下拿到更高收益

这意味着：

- 当前 attention `o/v` 更像热点 FFN 不完整时的补偿性 readout / routing 项
- 而不是当前最接近根因的主 computation source
- 如果后续还要设计新的 `H6` interaction probe，更不该把“attention 更大”当成正向目标

## Update · 2026-03-15T08:44:26+00:00

热点层 branch completeness / synergy audit 现已完成；当前对 `R24` 的最佳解释已从“pair-level 候选 + attention compensation”进一步压缩到“更像局部 branch-complete circuit，而不只是共同响应同一个上游 shaping factor”。

- branch synergy memo：[/workspace/deep-past-/docs/public_model_h6_branch_synergy_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_branch_synergy_analysis_2026-03-15.md)
- branch synergy report：[/workspace/deep-past-/reports/public_model_h6_branch_synergy_audit_20260315/branch_synergy_audit.json](/workspace/deep-past-/reports/public_model_h6_branch_synergy_audit_20260315/branch_synergy_audit.json)

当前最值得冻结的新增结论是：

1. 六个热点层全部满足：
   - `branch_completeness_sum > 1`
   - `attn_relief_vs_split_min < 1`
   - 即 `R24` 在同一层里保住了超过“一条 split 线等价量”的 branch mass，同时 attention 补位低于任一 split
2. 汇总指标为：
   - `mean branch_completeness_sum = 1.1933`
   - `mean attn_relief_vs_split_min = 0.6408`
   - `synergy_pattern_hits = 6 / 6`
3. 在 `5/6` 个热点层里：
   - `R24 ffn_total >= max(R29, R30)`
   - 但 `R24 local_total < max(R29, R30)`
4. 因此当前更像是：
   - `wi` 与 `wo` 在热点层里把 FFN 子回路补完整
   - attention 陪跑需求随之下降
   - `R24` 靠更高效的局部分配，而不是更大的局部总更新，拿到更高收益

这意味着：

- 当前静态证据更偏向 `branch completeness / local synergy`
- 比“纯 shared shaping factor, no local synergy”更符合现有数据
- 但这仍是更强的推断，不是最终机制定论

## Update · 2026-03-15T09:12:14+00:00

`decoder block 4/5` 与 `encoder 13~17` 的结构扩展分析现已完成；当前热点层口径已从“少数层很热”进一步收紧到“decoder readout corridor + encoder FFN corridor”的结构层面。

- structure memo：[/workspace/deep-past-/docs/public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md)
- expanded pair report：[/workspace/deep-past-/reports/public_model_h6_module_pair_expanded_audit_20260315/module_pair_expanded_audit.json](/workspace/deep-past-/reports/public_model_h6_module_pair_expanded_audit_20260315/module_pair_expanded_audit.json)

当前最值得冻结的新增结论是：

1. decoder 侧不是只有 `decoder block 5`
   - `decoder block 4` 也是同一 decoder hotspot family 的成员
   - 但更像次级 staging / handoff hotspot
2. `cross.o` 在 `decoder 4/5` 都 consistently 强于 `self.o`
   - 说明 decoder 侧更关键的 attention companion 仍是 cross readout
3. encoder 侧也不是均匀热区
   - 当前更像 odd layers `13 / 15 / 17` 偏 `wi_1`
   - even layers `14 / 16` 偏 `wi_0`
4. 因此当前最值得继续盯的仍然是：
   - `decoder 5 / wi_1 + cross.o`
   - `encoder 16 / wi_0 + self.o`
   - `encoder 17 / wi_1 + self.o`
   - `decoder 4 / wi_1 + cross.o` 作为 decoder corridor 的补充位点

这意味着：

- 当前热点更像一条 decoder readout corridor 加一段 encoder FFN corridor
- 但真正最有因果密度的层，仍然是 `decoder 5` 与 `encoder 16/17`

## Update · 2026-03-15T09:12:15+00:00

checkpoint trajectory 与方向审计现已完成；当前口径已从“热点静态快照”进一步推进到“哪些方向是共享的、哪些分差来自局部效率”的轨迹层面。

- trajectory + direction memo：[/workspace/deep-past-/docs/public_model_h6_trajectory_direction_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_trajectory_direction_analysis_2026-03-15.md)
- trajectory report：[/workspace/deep-past-/reports/public_model_h6_checkpoint_trajectory_audit_20260315/checkpoint_trajectory_audit.json](/workspace/deep-past-/reports/public_model_h6_checkpoint_trajectory_audit_20260315/checkpoint_trajectory_audit.json)
- direction report：[/workspace/deep-past-/reports/public_model_h6_direction_audit_20260315/direction_audit.json](/workspace/deep-past-/reports/public_model_h6_direction_audit_20260315/direction_audit.json)

当前最值得冻结的新增结论是：

1. 三条 `H6` 线的热点模块方向在早期就已经基本定型
   - `100 -> 200` 的 within-run cosine 已在 `~0.96-0.98`
   - `200 -> 300` 几乎全部到 `~0.999`
2. `R24 / R29 / R30` 的共享模块方向是同形的，但不是纯缩放关系
   - `R24 vs R29 / wi_0, wi_1 ~= 0.67`
   - `R24 vs R30 / wo ~= 0.85`
   - attention `o/v` 多在 `~0.64-0.78`
3. `R30` 的收益更像 early readout saturation
   - `geom 39.7488 -> 40.3905 -> 40.4032`
4. `R24` 的优势更像 late refinement of a branch-complete circuit
   - `geom 40.3239 -> 40.3395 -> 40.5412`
   - 且热点 attention share 始终低于 split runs

这意味着：

- shared shaping direction 的确存在
- 但当前分差更像来自 `branch completeness / local synergy`
- 而不是来自“把同一方向简单放大得更大”

## Update · 2026-03-15T09:31:54+00:00

当前阶段的 `H6` 根因分析已足以冻结为“高置信局部根因”；主任务定义因此从“继续压缩局部根因”切换为“有限机制验证 + 复现准备判定”。

- local root-cause memo：[/workspace/deep-past-/docs/public_model_h6_high_confidence_local_root_cause_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_high_confidence_local_root_cause_2026-03-15.md)

当前最值得冻结的新增结论是：

1. 当前已经可以高置信地把局部根因写成：
   - 一个被 `training-shape` 塑形的稀疏 FFN-anchored local circuit
   - 其核心位于 `decoder 5`、`decoder 4`、`encoder 13~17`
2. 这个局部回路内部当前最像：
   - `wi` = 主 computation branch
   - `wo` = readout / completion branch
   - attention `o/v` = 配套 / 补偿项
3. 当前最有因果密度的局部位点已收敛到：
   - `decoder 5 / wi_1 -> wo -> cross.o`
   - `encoder 16 / wi_0 + self.o`
   - `encoder 17 / wi_1 + self.o`
4. 因此下一阶段不应再以“继续补静态分析”为主
   - 而应转入有上限的机制验证
   - 并在局部根因足够硬后转入 `replication prep`

这意味着：

- 当前已经足够冻结“高置信局部根因”
- 但还不够写成“完整形成机制已证实”

## Update · 2026-03-15T13:06:39+00:00

`encoder 16/17 necessary-core` 的 full-row 验证现已完成；这条支线现在可以正式从“待验证核心候选”降为“已否掉的必要核心假说”。

- full-row verification memo：[/workspace/deep-past-/docs/public_model_h6_encoder_core_verification_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_encoder_core_verification_2026-03-15.md)
- eval results：[/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/mechanism_eval_results.json](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/mechanism_eval_results.json)
- sample slice：[/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/sample_slice_summary.json)

当前最值得冻结的新增结论是：

1. `M1` full-row 已足够单独否掉 `encoder 16/17 necessary core`
   - `e16 pair = +0.0500`
   - `e17 pair = +0.0329`
   - `pair union = +0.0654`
   - `e16 FFN-all = -0.0320`
   - `e17 FFN-all = +0.0792`
   - `e16+17 FFN-all = +0.1101`
   - 其中最大手术质量占比已到 `12.51%`
2. `M2` donor reversion 只支持“弱局部专属性”，不支持“必要核心”
   - `r29 -> r24` 的 `e17 / union` 出现轻微负项：
     - `e17 pair = -0.0935`
     - `union = -0.1238`
   - 但更苛刻的 `r30` harsher reversion 全部转正：
     - `e16 pair = +0.0520`
     - `e17 pair = +0.0988`
     - `union = +0.1366`
   - 因此当前更像 donor-specific mismatch，而不是承重必要性
3. winning-row slice 也没有把 encoder 升格成核心
   - `r24` 真正赢过 `r29/r30` 的样本共有 `64` 条
   - `r24 = 41.8744`
   - `r29/r30 ~= 40.05`
   - 所有 encoder 变体仍落在 `41.3535 ~ 41.7578`
   - 说明它们只削掉一部分优势，无法抹掉主优势本身
4. 因而当前最准确的冻结口径应更新为：
   - `encoder 16/17 = companion / corridor`
   - `encoder 17` 可能带有弱局部专属性
   - 但两者都不是当前已证实的局部根因必要核心

这意味着：

- `encoder 16/17 necessary core` 这条支线可以正式关闭
- `complete mechanism check` 的主因焦点应继续压回 `decoder 5 / FFN`，尤其是 `wi_1 + wo`
- 如果未来还要回头问 encoder role，也只能作为 subordinate 的 softer corridor intervention，而不应再把它当成 root-cause 主假说

## Update · 2026-03-15T15:38:26+00:00

围绕当前最高置信局部根因 `decoder 5 / FFN wi_1 + wo` 的 `replication prep` 现已从“设计”进入“自动执行”阶段；当前主目标不再是继续追求完整机制闭环，而是用最小但系统化的一组 training proxy，把公开模型的局部根因压成可复现准备的操作面。

- replication queue flow：[/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/route_decision.md](/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/route_decision.md)
- live status：[/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/driver_status.json](/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/driver_status.json)
- live log：[/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/logs/queue.log](/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/logs/queue.log)
- queue script：[/workspace/deep-past-/scripts/public_model_r31_decoder5_rootcause_replication_queue.py](/workspace/deep-past-/scripts/public_model_r31_decoder5_rootcause_replication_queue.py)
- tmux launcher：[/workspace/deep-past-/scripts/public_model_r31_decoder5_rootcause_replication_tmux.sh](/workspace/deep-past-/scripts/public_model_r31_decoder5_rootcause_replication_tmux.sh)

当前这一轮固定采用四个 system-level proxy 候选：

1. `R31 = q/k/v/o + wi_1 + wo`
2. `R32 = q/k/v/o + wi_1`
3. `R33 = wi_1 + wo`
4. `R34 = o_proj + wi_1 + wo`

对应 config 为：

- [/workspace/deep-past-/configs/public_model_r31_public_h6proxy_qkvo_wi1wo_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r31_public_h6proxy_qkvo_wi1wo_c0_pilot_20260315.yaml)
- [/workspace/deep-past-/configs/public_model_r32_public_h6proxy_qkvo_wi1_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r32_public_h6proxy_qkvo_wi1_c0_pilot_20260315.yaml)
- [/workspace/deep-past-/configs/public_model_r33_public_h6proxy_ffn_wi1wo_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r33_public_h6proxy_ffn_wi1wo_c0_pilot_20260315.yaml)
- [/workspace/deep-past-/configs/public_model_r34_public_h6proxy_o_wi1wo_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r34_public_h6proxy_o_wi1wo_c0_pilot_20260315.yaml)

当前自动 promotion gate 固定为：

- `healthy`
- `delta_vs_plain >= 0`
- `geom >= max(R29, R30) = 40.4669`
- `delta_vs_r24 >= -0.10`

只有过门槛的 best pilot 才会自动进入：

- `R35 long confirm`：[/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315](/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315)

当前最值得冻结的新增状态是：

1. `replication prep` 的设计口径已经正式切换为“围绕 `decoder 5 / FFN wi_1 + wo` 的系统 proxy bundle”
   - 不再继续把主要资源投到 `encoder 16/17 necessary-core` 这条已否掉的支线
2. 四个候选 config 的 preflight 都已通过
   - 与公开评测配置的 `task_prefix / folds / generation wiring` 全部匹配
3. 自动执行已实际启动
   - live `tmux session = pm_r31_dec5_rootprep`
   - 当前 flow stage = `pilot_r31_qkvo_wi1wo`
   - 当前 candidate stage = `train_pilot`
4. 启动器与状态机之间的接口缺口已补上
   - queue 端现在允许空 JSON 初始化
   - `tmux` 启动脚本也会主动写入 `{}` 与 `# Route Decision`
   - 因而这套 bundle 现在可以稳定复跑，不再依赖手工修补状态文件

这意味着：

- 当前已经进入真正的 `replication prep`，而不是继续在局部机制分数上做小幅纠缠
- 下一次需要更新的，不应再是“是否继续分析 encoder”
- 而应是四个 proxy pilot 的排序，以及是否有候选足够接近 `R24` 从而值得进入 `R35 long confirm`

## Update · 2026-03-15T18:03:04+00:00

`R31~R35` decoder5 replication-prep queue 现已全部落地；本稿现在可视为 `R35` 收口后的 `replication prep / root-cause follow-up` 决策快照。

- replication queue summary：[/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r31_decoder5_rootcause_replication_flow_20260315/driver_results.json)
- `R34` 正式结果：[/workspace/deep-past-/reports/public_model_r34_public_h6proxy_o_wi1wo_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r34_public_h6proxy_o_wi1wo_pilot_20260315/driver_results.json)
- `R35` 正式结果：[/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315/driver_results.json)
- `R35` route decision：[/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315/route_decision.md](/workspace/deep-past-/reports/public_model_r35_public_h6proxy_decoder5rep_long_20260315/route_decision.md)
- 正式 freeze：
  - `R34 = ckpt100 / geom 40.5498 / delta vs plain continuation pilot = +0.3232 / delta vs R24 pilot = +0.0086 / verdict = positive / healthy = true`
  - `R35 = ckpt1200 / geom 40.2733 / delta vs plain continuation pilot = +0.0467 / delta vs incumbent long = -0.1295 / verdict = inconclusive / healthy = true`

### 对当前 `replication prep` 的直接含义

当前可以比较硬地冻结下面四条：

1. `R34 = o_proj + wi_1 + wo` 是当前最强的 system-level proxy
   - 它在 pilot 上不只是过 gate
   - 而且以 `40.5498` 略高于 `R24` 的 `40.5412`
2. 当前证据不支持“把 `q/k/v` scaffold 一起带上会更接近根因”
   - `R31 = q/k/v/o + wi_1 + wo` 只有 `40.3710`
   - `R32 = q/k/v/o + wi_1` 甚至掉回 `40.0868`
   - 这更像是在说：当前真正重要的不是更大的 attention coverage，而是 `o_proj + wi_1 + wo` 这条更窄的 branch placement
3. 但当前 best pilot 仍没有在 long confirm 里站住
   - `R35` 虽然健康
   - 但 long best 只有 `40.2733`
   - 低于 incumbent long `40.4028` `0.1295`
4. 因而当前最严格的统一口径应更新为：
   - `decoder 5 / FFN wi_1 + wo` 仍是最高置信局部根因
   - `o_proj + wi_1 + wo` 是当前最强的 pilot-level replication proxy
   - 但“稳定的上游训练形态复现”仍未闭环

### 如果未来还要继续排查根因，理论上还缺什么（仅作备忘，当前不继续执行）

截至 `2026-03-15`，这部分只保留为理论缺口备忘，不再作为当前推荐动作。

如果未来确实还要回头追根因，而不是再开一轮宽泛 proxy，最值得保留的工作应收缩到下面四类：

1. 解释 `R34` 的“pilot 强、long 掉”的时间维问题
   - 先比较 `ckpt100 -> 200 -> 400 -> 600 -> 1200` 的 trajectory
   - 看当前收益是在什么阶段开始流失
   - 这一步优先于再造新 recipe
2. 围绕 `winning rows` 继续找 `decoder5 triple` 之外的弱补位因子
   - 也就是继续问：为什么 `o_proj + wi_1 + wo` 已经够 pilot winning，却在 long 上保不住
   - 当前最像的方向仍应是 case-conditional / corridor-style companion，而不是回头把整套 `q/k/v` 再挂回去
3. 把后续 root-cause 审计继续限制在“局部 companion”而不是“更大 coverage”
   - `R31/R32` 已经说明更宽的 attention scaffold 不是当前缺口
   - 因此后续不应优先继续堆 `q/k/v/o`、更大 `rank/alpha`、或新的宽 recipe
4. 在新的 long 预算之前，先补一轮低成本稳定性检查
   - 至少确认 `R34` 的 early-win 不是偶然 checkpoint 噪声
   - 然后再决定是否值得围绕同一抓手开下一枪 long

一句话说：

- `replication prep` 已经把最优 proxy 压到 `o_proj + wi_1 + wo`
- 但根因排查还差最后一层：解释为什么它能在 pilot 上贴住甚至略超 `R24`，却不能在 long 上稳定承重

## Update · 2026-03-15T20:06:49+00:00

围绕 `R34 = o_proj + wi_1 + wo` 的低成本稳定性复核现已全部落地；本稿现在可视为 `R35` 之后、`R34 stability pack` 收口后的 `root-cause follow-up` 决策快照。

- stability-pack summary：[/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/driver_results.json)
- stability-pack route：[/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/route_decision.md](/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/route_decision.md)
- checkpoint trajectory audit：[/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/audits/r34_r35_checkpoint_trajectory.json](/workspace/deep-past-/reports/public_model_r34_stability_pack_20260315/audits/r34_r35_checkpoint_trajectory.json)
- 正式 freeze：
  - `fold0 seed42 = ckpt100 / geom 40.5498 / verdict positive`
  - `fold0 seed43 = ckpt200 / geom 40.1945 / verdict inconclusive`
  - `fold0 seed44 = ckpt100 / geom 40.4182 / verdict inconclusive`
  - `fold0 seed summary = geom mean/std 40.3875 / 0.1466`

这里还要补一个解释边界：

- `fold1 seed42 = 48.7263`
- `fold2 seed42 = 37.2435`

但由于 `fold1 / fold2` 对应的是不同验证切片，这两个绝对 geom 不能直接拿来替代 `fold0` gate 做 promote 判断；当前它们最有价值的意义是：

- 这条 `R34` 线在跨 fold 上仍表现出较大的异质性

### 对当前 `R34` 抓手的直接含义

当前可以先冻结成下面四条：

1. `R34` 的 `ckpt100` 早赢并不是伪信号
   - `fold0 seed42` 和 `fold0 seed44` 都在 `ckpt100` 见到 best
   - 因而“最优点前移”本身不是单次偶然读数
2. 但 `R34` 还不具备足够硬的 fold0 seed 稳定性
   - `seed42 = 40.5498`
   - `seed43 = 40.1945`
   - `seed44 = 40.4182`
   - 三个 seed 的 spread 已到 `0.3553`
   - 因而当前不能把 `R34` 写成已经稳定成立的 promote recipe
3. `R34` 当前最需要解释的缺口是“早 checkpoint 敏感性”，不是“还要不要再把 `q/k/v` 挂回来”
   - `R31/R32` 已经否掉了更宽 attention scaffold 的优先级
   - 而 checkpoint trajectory audit 也确认 `R34 pilot` 在 `fold0 seed42` 上是 `ckpt100 > ckpt200 > ckpt300`
4. 因而 `R34` 现在最准确的口径应更新为：
   - `pilot best proxy, but early-peak and not yet seed-stable`

### 如果未来还要继续排查根因，现在真正缺的工作是什么（仅作备忘，当前不继续执行）

截至 `2026-03-15`，这部分也只保留为理论缺口备忘，不再作为当前推荐动作。

当前最值得保留的 root-cause follow-up 工作，应继续收缩到下面四类：

1. 解释为什么 `R34` 的收益集中在早 checkpoint
   - 优先审计 `ckpt100 -> 200 -> 300` 的分数回落与模块轨迹
   - 这是当前最直接的因果缺口
2. 解释为什么 `fold0 seed43` 会掉到接近 baseline，而 `seed42/44` 仍能保住一部分早赢
   - 也就是把问题从“recipe 对不对”收缩成“这个局部回路为什么对 seed 敏感”
3. 围绕 `winning rows` 比较 `fold0 seed42` 与 `seed43/44`
   - 看真正丢失的是哪一类样本
   - 以及这些样本是否需要 `decoder5 triple` 之外的 case-conditional companion
4. 在解释完 `R34` 的 early-peak / seed-sensitivity 之前，不再优先开新的更宽 recipe 或新的 long
   - 当前更像是稳定性与承重问题
   - 不是 coverage 还不够大

一句话说：

- `R34` 已经告诉我们当前最像根因的 system-level 抓手确实落在 `o_proj + wi_1 + wo`
- 但真正还没排完的，不是“主抓手在哪”，而是“为什么这个抓手只能早赢，且还不够 seed-stable”

## 5. 2026-03-15-2 执行计划（已执行）

本节对应的一枪 `R30` 已于 `2026-03-15T06:03:02+00:00` 完成；当前结果口径以上方最新 `Update` 为准。

### 5.1 统一编号与口径

为避免后续文档再次混淆，本稿从这里开始统一按下面口径理解：

1. `H4` = normalization / task-form
2. `H5` = unlabeled corpus scope
3. `H6` = training-shape / wider adaptation proxy
4. `R26` 虽然文件名里仍写着 `h5proxy`，但按当前统一口径，它实际应视为：
   - `R24/H6` 候选的 guarded long confirm

### 5.2 当前已冻结事实

当前应先把下面五条写死：

1. 比较锚点固定为：
   - public anchor `39.1025`
   - plain continuation pilot `40.2266`
   - incumbent long `40.4028`
2. `H3` 当前不成立为默认后续：
   - `R18 = 39.4724 / negative`
   - `R19 = 40.1336 / inconclusive`
   - `R20 = 39.5334 / negative`
3. `H4` 当前不成立为默认后续：
   - `R21/R22/R23 = 40.2266 / inconclusive`
4. `H6` 是当前唯一持续给出正信号的上游轴：
   - `R24 = 40.5412 / positive / current best pilot`
   - `R26 = 40.4099 / guarded long inconclusive`
   - `R27 = 40.3165 / higher-capacity inconclusive`
   - `R29 = 40.4669 / inproj split positive`
   - `R30 = 40.4032 / outproj split inconclusive`
5. `public-weight continuation` 仍是 stable incumbent：
   - `Track B` 到现在为止还没有拿到足以 promote 的替代候选

### 5.3 这些事实能够说明什么

当前可以比较严格地说明下面四条：

1. 当前最像 `public model` 上游强度线索的是 `H6 / training-shape`
   - 不是 `H3`
   - 也不是当前安全范围内的 `H4`
2. `R29 + R30` 一起说明 `R24` 的正信号并不是由单一残差分支完整解释
   - `wi_0/wi_1` 当前更像主效应
   - `wo` 当前更像补充贡献或组合项
3. `R27 < R30 < R29 < R24` 更支持：
   - 当前有信息密度的是 `module placement / coverage`
   - 不是继续把同一套 recipe 的 `rank/alpha` 做大
4. `U1~U3` 与 `H4` 并不是“没价值”
   - 它们的价值主要在于排除假设空间
   - 而不是提供新的收益主线

### 5.4 这些事实还不能说明什么

当前仍然不能直接说明：

1. `wo` 一定就是 `R24 - R29` 剩余增益的唯一来源
2. `R29` 已经完整解释了 `R24`
3. 当前 `H6` 已经值得再开 long confirm
4. 我们已经解释了 `public model` 的形成机制

因此，当前最严格的统一口径应是：

- 我们拿到的是一组越来越清晰的上游因果线索，而不是对 `public model` 的完整复现或完整归因

在这个边界内，最新 module-pair audit 已把“如果还要继续分析或设计下一枪，该先问什么”进一步压缩到下面三条：

1. `encoder 17 / wi_1 + self.o`
   - 当前是最强的 `wi`-anchored pair 候选
   - `R29 = 64.44% local`
   - `R30 = 15.84% local`
2. `encoder 16 / wi_0 + self.o`
   - 当前更像 `encoder 16` 的主 pair
   - `R29 = 58.51% local`
   - `R30 = 16.35% local`
3. `decoder 5 / wi_1 + cross.o`
   - 当前更像 decoder 侧最值得问的 interaction
   - `R29 = 51.36% local`
   - `R30 = 11.20% local`

因此，如果后续真的还要开新的 `H6` probe，最严格的设计约束应是：

1. 必须是 `interaction-focused`，而不是新的 `rank/alpha` 扩张
2. 必须优先围绕 `wi`-anchored pair，而不是 `wo`-anchored pair
3. 必须把核心问题写成：
   - `o` 是否只是稳定陪跑项
   - 还是某个特定 `wi + o` pair 才是当前最接近根因的局部 interaction clue

同时，最新 attention compensation audit 进一步加上一条约束：

1. 不能把更高的 attention `o/v` 质量误读成更接近 `R24`
2. 当前更像是：
   - `wi + wo` 越不完整
   - attention 补位越重
3. 因此下一轮如果还要问 interaction
   - 应问“哪个 `wi + o` pair 最有因果效率”
   - 而不是“怎样把 attention `o/v` 再做大”

最新 branch synergy audit 则把这条纪律再向前推了一步：

1. 当前更像是在问：
   - `wi` 与 `wo` 是否共同补全了同一热点 FFN 子回路
   - 而不只是共同响应同一个更上游因素
2. 因此后续如果继续做分析或设计 probe
   - 更值得问串联/协同关系
   - 不值得再回到“单支是不是主因”这种已经被压缩过的问题
3. 当前最合理的口径应是：
   - `wi` 是主 computation branch
   - `wo` 是 readout completion branch
   - attention 更像 branch 不完整时上升的补偿项

最新 structure + trajectory/direction 分析则进一步把 probe 设计约束收紧成：

1. decoder 侧如果要继续问 interaction
   - 第一优先级应是 `decoder 5 / wi_1 -> wo -> cross.o`
   - 第二优先级才是 `decoder 4 / wi_1 + cross.o`
2. encoder 侧如果要继续问 interaction
   - 应把 `encoder 16 / wi_0 + self.o`
   - 与 `encoder 17 / wi_1 + self.o`
   当成一对非对称配对问题来问
3. 不应再把下一枪定义成：
   - 只看单支大小
   - 只看更高 attention
   - 或只看 shared shaping 有没有存在
4. 当前最合理的下一问已经收敛成：
   - `decoder 5` 上 `wi_1 -> wo -> cross.o` 更像串联协同
   - 还是更像并联共同响应
   - focused memo：[/workspace/deep-past-/docs/public_model_h6_decoder5_serial_parallel_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder5_serial_parallel_analysis_2026-03-15.md)

### 5.5 该轮计划实际做什么

这轮计划正式命名为：

- `2026-03-15-2`

它实际限定为一枪：

- `R30 = H6 wo-only split pilot`

问题定义是：

- 在固定 `public model -> official-only continuation`、`raw-row fold0 313` 与当前 decode 不变的前提下，测试 `q/k/v/o + wo`、而不挂 `wi_0/wi_1`，是否仍能保住 `R24` 的正信号。

固定不变项：

- start model: `public model`
- downstream: `official-only continuation`
- gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- `r=16 / alpha=32`
- 不叠加 `H3`
- 不叠加 `H4`

比较对象固定为：

1. baseline `40.2266`
2. `R24 = 40.5412`
3. `R29 = 40.4669`

### 5.6 为什么下一枪是 `wo-only`

原因应写清楚：

1. `R29` 已经把 `wi_0/wi_1` 这支单独拉出来了
2. `R24` 相对 `R29` 的剩余差值只剩 `0.0743`
3. 当前最便宜、最直接、信息密度最高的下一问，就是：
   - 剩余差值主要落在 `wo`
   - 还是落在 `wi + wo` 的组合效应
4. 这一步比继续做 long、继续放大 capacity、或回头补弱轴更值得做

### 5.7 `R30` 跑完后应该怎么解释

下一步建议按下面三种分支理解：

1. 如果 `R30` 接近 `R24`，且明显高于 `R29`
   - 更支持“剩余增益主要落在 `wo`”
2. 如果 `R30` 为正，但明显低于 `R29`
   - 更支持“`wi_0/wi_1` 是主效应，`wo` 是次效应”
3. 如果 `R30` 接近 baseline 或转负
   - 更支持“`wo` 单独不够，剩余更像 `wi + wo` 组合效应”

只有当新 pilot 至少不弱于 `R24`，才重新讨论 long confirm。

### 5.8 当前明确不做什么

当前不建议：

1. 开新的 long confirm
2. 继续扩大 `rank/alpha`
3. 把 `H3/H4` 重新叠回 `H6`
4. 回头补跑已经冻结的弱轴轻量变体

## 6. 2026-03-15-3 机制检查执行与复现准备快照

本节最初对应 `2026-03-15-3` 的完整机制检查计划；在 `6.8/6.9` 实际执行完成后，当前用途已改为：

1. 保存这轮机制验证的执行快照
2. 冻结已经足够用于复现准备的局部根因
3. 把“完整机制闭环”降为可选未来支线，而不是当前必须目标

当前默认目标不再是：

- 必须把当前局部根因推进成完整机制闭环

而是：

- 用最少且最有因果密度的检查，把当前“高置信局部根因”加硬到足以支持 `replication prep`

### 6.1 当前阶段性结论

进入 `2026-03-15-3` 前，先把当前阶段性结论写死：

1. 已冻结为高置信局部根因的不是：
   - `H3`
   - `H4`
   - attention 主导
   - 更大 `rank/alpha`
2. 已冻结为高置信局部根因的是：
   - 一个被 `training-shape` 塑形的稀疏 FFN-anchored local circuit
   - 主 computation branch 更像 `wi`
   - readout completion branch 更像 `wo`
   - attention `o/v` 更像配套 / 补偿项
3. 当前最有信息密度的位点是：
   - `decoder 5`
   - `decoder 4`
   - `encoder 16/17`

以下 `6.2 ~ 6.7` 保留为当时的执行设计脉络；当前状态与结论，以 `6.8 ~ 6.10` 为准。

### 6.2 `2026-03-15-3` 的核心问题

这一轮不再问“大方向对不对”，而是只问下面四个完整机制检查问题：

1. 必要性：
   - 如果拿掉 `decoder 5 / wi_1`
   - 或拿掉 `decoder 5 / wo`
   - `R24` 的收益会掉多少
2. 充分性：
   - 如果只保留 `decoder 5 + encoder 16/17` 的关键分支
   - 是否已经能保住 `R24` 的主要收益
3. interaction：
   - `decoder 5 / wi_1 -> wo -> cross.o` 更像串联协同
   - 还是并联共同响应
4. corridor structure：
   - `decoder 4 -> decoder 5`
   - 与 `encoder 16 / 17`
   是否构成最小可解释回路

### 6.3 工作包

这一轮完整机制检查应按下面顺序执行：

1. `M1` 必要性检查
   - 对 `R24` 做 layer/module ablation
   - 优先：
     - `decoder 5 / wi_1`
     - `decoder 5 / wo`
     - `encoder 16 / wi_0`
     - `encoder 17 / wi_1`
2. `M2` 充分性检查
   - 对 `R24` 做 restricted transplant / keep-only 检查
   - 先看：
     - `decoder 5`
     - `decoder 5 + encoder 16/17`
     - `decoder 4/5 + encoder 16/17`
3. `M3` interaction 检查
   - 围绕 `decoder 5 / wi_1 -> wo -> cross.o`
   - 判断更像：
     - serial synergy
     - parallel shared response
4. `M4` encoder 非对称检查
   - 把 `encoder 16 / wi_0 + self.o`
   - 与 `encoder 17 / wi_1 + self.o`
   当成一对配对问题验证
5. `M5` sample-level slice 检查
   - 只看 `R24` 相对 `R29/R30` 真正赢下的样本
   - 验证收益是否集中在上述局部回路最活跃的样本上

### 6.4 首选实现顺序

如需最小化新工作量，首选顺序应是：

1. 先做离线 `adapter surgery / transplant`
   - 因为它比新训练更便宜
   - 但更接近因果
2. 只有当离线 surgery 仍无法区分串联/并联时
   - 才设计新的 `interaction-focused H6 probe`
3. 不应默认直接进入：
   - 新 long confirm
   - 更大 `rank/alpha`
   - 或更广的新热层扫描

### 6.5 最小通过标准

只有当下面三类结果同时成立，才允许把口径从“高置信局部根因”升级为“完整机制检查基本通过”：

1. 必要性成立
   - 拿掉关键 branch / key layer 会明显打掉 `R24` 主要收益
2. 充分性近似成立
   - 保留关键局部回路后，仍能保住 `R24` 的主要份额
3. interaction 判别成立
   - `decoder 5 / wi_1 -> wo -> cross.o` 能被收敛成更明确的协同结构

### 6.6 当前不做什么

`2026-03-15-3` 当前不建议：

1. 回头补更多同类静态分析
2. 把 attention 更大当成正向目标
3. 再开新的 long confirm
4. 再做更大 `rank/alpha`
5. 把 `H3/H4` 重新混回 `H6`

### 6.7 一句话任务定义

`2026-03-15-3` 的一句话定义应是：

- 在固定 `public model -> official-only continuation` 与当前 decode/gate 不变的前提下，围绕 `decoder 5 / decoder 4 / encoder 16/17` 做必要性、充分性与 interaction-focused 检查，以验证当前高置信局部根因是否足以组成最小完整机制解释。

### 6.8 `2026-03-15-3` 首轮执行结果（`subsample64`）

首轮系统执行已完成，主产物见：

- [public_model_h6_mechanism_subsample64_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_mechanism_subsample64_analysis_2026-03-15.md)
- [mechanism_summary.md](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/mechanism_summary.md)
- [sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/sample_slice_summary.json)

本轮必须先带一个口径边界：

1. 这是 `fold0 / 64-row` 机制筛查子样本，不是 full-row 终判。
2. 该子样本上的 absolute ranking 是：
   - `r29_ref > r30_ref > r24_ref`
3. 因而本轮只能用于读：
   - 必要性掉分模式
   - keep-only 保留模式
   - interaction 增益模式
   - `R24` 真正赢下样本上的保真模式

在这个边界下，第一轮最重要的执行结果是：

1. `M1` 必要性：
   - `decoder 5 / wi_1` 去掉后相对 `r24_ref` 为 `-0.3342`
   - `decoder 5 / wo` 去掉后相对 `r24_ref` 为 `-0.1636`
   - `encoder 16/17` 的单点与联合消融都未表现出必要性
2. `M2` 充分性：
   - `keep decoder 5 only` 是最能保分的 keep-only 方案
   - `keep decoder 5 + encoder 16/17` 与 `keep decoder 4/5 + encoder 16/17` 并未更好
3. `M3` interaction：
   - `wi_1 only` 与 `wo only` 接近
   - `wi_1 + wo` 比任一单支高 `+0.5745`
   - 再加 `cross.o` 只再提升 `+0.1099`
4. `M4` encoder pair：
   - `encoder 16 pair > encoder 17 pair > union`
   - 没有看到 additive union
5. `M5` sample slice：
   - 在 `R24` 真正赢下的 `15` 个样本上
   - `decoder 5` 手术比 `encoder` 手术更直接地打掉这些优势
   - `keep decoder 5 only` 在这些样本上的保真度明显高于 `R29/R30`

因此，`2026-03-15-3` 首轮执行后，最值得冻结的新口径应更新为：

- 当前最有因果密度、且已被第一轮机制检查显著加硬的局部机制，已经可以进一步压到 `decoder block 5 / FFN`，尤其是 `wi_1 + wo` 的组合；`wi_1` 更像主 computation branch，`wo` 更像 readout / completion branch，`cross.o` 更像次级补位项；`encoder 16/17` 在这一轮里仍更像 companion / corridor 假说，而不是已经被证明的必要核心。

同时必须明确：

1. 这还不是“完整机制已通过”
2. 这还不能把 `encoder 16/17` 升级成必要核心
3. 这还不能把 keep-only 下 corridor 表现不佳解释为“encoder 有害”

因此，`2026-03-15-3` 当前的最准确状态不是：

- 计划已结束

而是：

- 计划已完成第一轮系统执行
- `decoder 5 / FFN wi_1 + wo` 已拿到更强的局部因果支持
- 完整机制闭环仍需更高保真验证

从投入产出比看，后续最高价值动作应收缩到：

1. full-row 只重跑最关键少数变体：
   - `m1_ablate_d5_wi1`
   - `m1_ablate_d5_wo`
   - `m2_keep_d5_all`
   - `m3_keep_d5_wi1_only`
   - `m3_keep_d5_wo_only`
   - `m3_keep_d5_wi1_wo`
   - `m3_keep_d5_wi1_wo_crosso`
2. 围绕 `left_beats_all` 的 winning rows 做更细粒度 case audit
3. 若要继续验证 encoder role
   - 应设计比 hard keep-only 更温和的 corridor intervention
   - 而不是继续堆更多静态热点分析

### 6.9 `2026-03-15-3` 第二轮执行结果（`decoder5 full-row`）

这轮 full-row bundle 已完成，主产物见：

- [public_model_h6_decoder5_fullrow_verification_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder5_fullrow_verification_2026-03-15.md)
- [mechanism_eval_results.json](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_eval_results.json)
- [mechanism_eval_table.csv](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_eval_table.csv)
- [sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/sample_slice_summary.json)
- [mechanism_summary.md](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_summary.md)

这轮 bundle 覆盖：

1. refs
   - `r24_ref`
   - `r29_ref`
   - `r30_ref`
2. `M1`
   - `m1_ablate_d5_wi1`
   - `m1_ablate_d5_wo`
   - `m1_ablate_d5_wi1_wo`
3. `M2`
   - `m2_keep_d5_all`
   - `m2_keep_d5_e16e17_all`
   - `m2_keep_d4d5_e16e17_all`
4. `M3`
   - `m3_keep_d5_wi1_only`
   - `m3_keep_d5_wo_only`
   - `m3_keep_d5_crosso_only`
   - `m3_keep_d5_wi1_wo`
   - `m3_keep_d5_wi1_crosso`
   - `m3_keep_d5_wo_crosso`
   - `m3_keep_d5_wi1_wo_crosso`

这轮之后，最值得冻结的新口径应进一步更新为：

1. `M1` 必要性已经在 full-row 上再次坐实：
   - `m1_ablate_d5_wi1 = 40.3317`
   - `m1_ablate_d5_wo = 40.3587`
   - `m1_ablate_d5_wi1_wo = 40.1334`
   - 因而 `decoder 5 / wi_1` 与 `decoder 5 / wo` 都具备必要性，且联合去除伤害更大
2. `M2` 充分性仍然不成立：
   - `m2_keep_d5_e16e17_all = 39.2931`
   - `m2_keep_d5_all = 39.2882`
   - `m2_keep_d4d5_e16e17_all = 39.1161`
   - 说明当前最好的 compact sufficiency proxy 仍是 `decoder 5` 本身；`encoder 16/17` 不能把它升级成已证实核心
3. `M3` interaction 给出的最准确读法，不再是“简单硬串联”，而是：
   - best single = `m3_keep_d5_crosso_only = 39.1298`
   - best pair = `m3_keep_d5_wi1_wo = 39.1906`
   - triple = `m3_keep_d5_wi1_wo_crosso = 39.4729`
   - `wi_1 + crosso = 38.8120` 明显偏差
   - `wo + crosso = 39.1275` 基本只回到 `crosso_only`
   - 所以 `cross.o` 更像条件性 export/readout port，而不是稳定的第二主 branch
4. `M5` winning-row slice 继续压低可下结论边界：
   - `left_beats_all` 共 `64` 条
   - `r24_ref = 41.8744`
   - `r29_ref = 40.0582`
   - `r30_ref = 40.0349`
   - `M1` 变体会直接吃掉这批主优势
   - 但所有 keep-only 子图都保不回这批 winning rows

因此，这轮 full-row 后，当前最高置信局部根因的最准确表述应是：

- `decoder 5 / FFN wi_1 + wo` 可以升级为当前最强的局部必要核心；
- `cross.o` 仍重要，但其作用更像在 `wi_1 + wo` 已同时存在后的条件性 readout/export companion；
- `encoder 16/17` 依然只能保留为非核心 corridor companion，而不是必要核心。

但也必须把边界写死：

1. 这还不是“完整机制已通过”
2. `m3_keep_d5_wi1_wo_crosso` 虽然是当前最好的 compact local subgraph
   - 但仍比 `r24_ref` 低 `-1.0683`
3. 在真正 `left_beats_all` 的 winning rows 上
   - 它也没有把 `r24` 的主优势保回来

所以，`2026-03-15-3` 当前最准确的状态不是：

- 根因分析已彻底结束

而是：

- `decoder 5 / FFN wi_1 + wo` 已被 full-row 进一步加硬为高置信局部必要核心
- `cross.o` 被降格为条件性 companion
- `encoder 16/17` 继续保持非核心 corridor 口径
- 完整机制闭环仍需更高保真检查

这轮之后，剩余最高价值工作应继续收缩到：

1. `winning-row` case audit
   - 重点对比：
     - `m1_ablate_d5_wi1`
     - `m1_ablate_d5_wo`
     - `m1_ablate_d5_wi1_wo`
     - `m2_keep_d5_e16e17_all`
     - `m3_keep_d5_wi1_wo_crosso`
2. 更温和的 corridor intervention
   - 不再做更重的 hard keep-only
   - 而是围绕 `decoder5 triple + weaker corridor mix` 看 `encoder 17 / self.o` 或局部 `decoder4 -> decoder5` 是否只在 winning rows 上补位
3. case-conditional interaction 检查
   - 重点不是再问“`cross.o` 要不要更大”
   - 而是问：
     - 为什么 `cross.o` 只在 `wi_1 + wo` 同时存在时才更有帮助
     - 以及为什么这种帮助仍不足以闭合 winning rows 上的主优势

### 6.10 当前冻结快照：实验结果、结论与状态

本节用于把截至目前所有与 `public model upstream reverse-engineering` 直接相关的结果，压缩成一个可停止、可交接、可进入 `replication prep` 的统一快照。

#### 6.10.1 当前实验结果快照

可以压缩成下面五组：

1. `Track A / public-weight continuation`
   - stable incumbent 仍是 `geom = 40.4028`
   - `stability pack` 的 `fold0` seed 汇总为 `40.5055 ± 0.1281`
   - 这条线已经完成 `pilot -> long -> stability pack`
2. `Track B / H3-H4` 轻量 upstream probe
   - `TAPT-lite -> continuation` 同预算 pilot 为 `40.1407`
   - 相对 plain continuation pilot `40.2266` 为 `-0.0859`
   - 正式 verdict 仍只能写 `inconclusive`
3. `Track B / H6` 的大方向筛查
   - `R27` freeze 为：
     - `geom = 40.3165`
     - `delta vs plain continuation pilot = +0.0899`
     - `delta vs R24 pilot = -0.2247`
     - `verdict = inconclusive`
   - 因而更大 `rank/alpha`、继续堆同配方，不是当前主线
4. `encoder 16/17 necessary-core` full-row 验证
   - 已完成否证
   - 当前统一口径为：
     - `encoder 16/17` 不是必要核心
     - 最多是非核心 `companion / corridor`
5. `decoder5` full-row 机制验证
   - `M1`
     - `m1_ablate_d5_wi1 = -0.2095`
     - `m1_ablate_d5_wo = -0.1825`
     - `m1_ablate_d5_wi1_wo = -0.4077`
   - `M2`
     - `m2_keep_d5_e16e17_all = 39.2931`
     - `m2_keep_d5_all = 39.2882`
     - `m2_keep_d4d5_e16e17_all = 39.1161`
   - `M3`
     - best single = `m3_keep_d5_crosso_only = 39.1298`
     - best pair = `m3_keep_d5_wi1_wo = 39.1906`
     - best compact local subgraph = `m3_keep_d5_wi1_wo_crosso = 39.4729`
     - `m3_keep_d5_wi1_crosso = 38.8120`
     - `m3_keep_d5_wo_crosso = 39.1275`
   - `winning rows`
     - `left_beats_all` 共 `64` 条
     - `r24_ref = 41.8744`
     - `m3_keep_d5_wi1_wo_crosso = 39.2398`

#### 6.10.2 当前冻结结论

截至当前，可以正式冻结下面六条：

1. 当前最强的局部必要核心在 `decoder 5 / FFN wi_1 + wo`。
2. `wi_1` 与 `wo` 都具备必要性，且联合去除伤害更大。
3. `cross.o` 不是稳定第二主 branch，更像条件性 readout/export companion：
   - 单独保留不够
   - 与 `wi_1` 配对明显不佳
   - 与 `wo` 配对也几乎不增益
   - 但加在 `wi_1 + wo` 上时会继续补分
4. `encoder 16/17` 不是必要核心，只保留为非核心 corridor companion。
5. 当前探索已经足够支撑“局部根因驱动的复现准备”。
6. 当前探索不再需要把“完整机制闭环”当作继续推进的前置条件。

#### 6.10.3 当前状态：是否足以进入复现准备

当前最准确的状态应写成：

1. `high-confidence local root cause` 已完成冻结。
2. 这一级冻结已经足够指导 `replication prep`：
   - 优先围绕 `decoder 5 / FFN wi_1 + wo`
   - 把 `cross.o` 视为条件性 companion
   - 不再把 `encoder 16/17` 当核心复现目标
3. 这还不是对 `public model` 形成机制的完整复现。
4. 但如果目标只是“带着明确机制抓手进入下一阶段复现设计”，当前证据已经够用。

因此，本稿当前不再把主任务定义为：

- 继续追完整机制闭环

而应改写为：

- 在冻结 `decoder 5 / FFN wi_1 + wo` 为当前最高置信局部根因的前提下，进入 `replication prep`

#### 6.10.4 完整闭环的突破口只保留一句话

如果未来还要回头追“完整机制闭环”，当前最可能的突破口应只保留为：

- 围绕 `winning rows`，检查为什么 `wi_1 + wo + cross.o` 虽然已是最强 compact local subgraph，却仍保不回 `r24` 的主优势；也就是继续找 `decoder5 triple` 之外、但只在真正 winning cases 上提供补位的弱 corridor / case-conditional factor。

## 0. 结论先行（`2026-03-14` 历史判断，现以文末 `# 工作判断` 为准）

以下 `0/1/2/3/4/6/7/8` 各节保留的是 `R34 stability pack` 完成前后的历史判断与原始计划。按 `2026-03-15` 已冻结的新事实，当前不再推荐继续展开这些 `Track B` follow-up；当前推荐执行口径已收敛到文末 `# 工作判断`。

接下来如果我们说“要搞懂 `public model` 怎么来”，就不能再把问题说成：

- 这条 continuation 再训久一点会不会更高
- 这条新线能不能顺便替代当前 incumbent
- 再叠一点 external / cleaning / decode 会不会起飞

更准确的问法应该是：

- 在固定 `public model -> official-only continuation` 主线与 `raw-row fold0 313` 主 gate 不变的前提下，哪一种 **上游机制** 能在同预算 downstream pilot 上给出稳定、可归因的增量？

也就是说，我们现在追的不是：

- full reproducible

而是：

- partial causal map
- partial reproducible upstream clue

当前证据已经足够说明：

1. `Track A` 不轻
   - `public-weight continuation` 已经完成 `pilot -> long -> stability pack`
   - 当前 stable incumbent 是 `geom = 40.4028`
   - `stability pack` 的 `fold0` seed 汇总是 `40.5055 ± 0.1281`
2. `Track B` 还轻
   - `M025` 已证明 `healthy but no gain -> stop`
   - `TAPT-lite -> continuation` 只做了一个很轻的 upstream probe
   - 它的同预算 pilot 结果是 `40.1407`
   - 相对 plain continuation pilot `40.2266` 的差值是 `-0.0859`
   - 当前正式 verdict 只能写成 `inconclusive`

因此，下一步如果要加“整体实验力度”，应该加在：

- upstream mechanism probe

而不是：

- downstream continuation 再盲目拉长

## 1. 现在“上游”到底指什么

这里的“上游”，专指发生在：

- `official-only supervised continuation`

之前的机制层。

对当前项目，最相关的上游层有四类：

1. `H3`: continued pretraining
   - 例如 `TAPT / DAPT`
   - 问题不是“能不能跑”，而是“会不会给同预算 downstream continuation 带来净增益”
2. `H4`: normalization / task form
   - 例如 `Tier-0` 之外的更强但仍安全的输入规范化
   - 或 prompt / task form 的稳定差异
3. `H5`: unlabeled corpus scope
   - 不同 source-side / text-only 语料范围
   - 不是 supervised external mix
4. `H6`: architecture / training shape
   - 更强 backbone
   - full-model vs adapter-only history
   - 是否存在更接近公开对象形成方式的训练形态

这里必须继续守住一个边界：

- 我们已经证明了“拿到 `public model` 以后，`official-only continuation` 可以稳定增益”
- 我们还没有证明“自己已经复现了 `public model` 的生成机制”

所以从现在开始，`Track B` 的目标应写成：

- 逐步建立 `public model` 强度来源的因果线索图

而不是：

- 直接宣布已经复现公开 monster

## 2. 接下来应该怎么对这个问题提要求

以后如果要让我做“上游机制”工作，建议直接按下面这种句式提：

### 2.1 推荐提法

1. 请把任务定义为 `Track B upstream reverse-engineering`
2. 固定以下不变项：
   - `public model`
   - `official-only continuation`
   - `raw-row fold0 313`
   - `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
3. 只允许改一个上游轴
4. 用同预算 downstream pilot 与 plain continuation pilot 比
5. 结果只能判：
   - `positive`
   - `inconclusive`
   - `negative`

### 2.2 推荐工作请求模板

你可以直接这样说：

> 把当前任务定义成 `Track B`。固定 `public model -> official-only continuation` 主线和 `raw-row fold0 313` 主 gate，不要追当前主线得分；只做一个上游单轴 probe，并用同预算 pilot 对比 plain continuation。

或者更具体一点：

> 请把当前问题写成 `H3a / H3b / H4` 假设树，先告诉我这次只动哪一个上游轴、为什么动它、比较对象是谁、停线条件是什么，再落 config / driver / tmux。

### 2.3 不推荐提法

不建议再说：

- 再训一版看看
- 把 TAPT、mix、cleaning 一起上
- 反正怪物都 40+ 了，继续加大训练
- 这次如果涨一点就直接开 long

这些说法的问题是：

- 它们会把 `Track A` 和 `Track B` 混在一起
- 会把“方向判定”和“主线 promote”混在一起
- 会让结果无法归因

## 3. 当前已经知道什么，不知道什么

### 3.1 已经知道的事

当前可以硬认：

1. `public-weight continuation` 是稳定主线
2. `M025 supervised mix` 不成立为默认后继
3. `TAPT-lite -> continuation` 在同预算 pilot 下没有给出净增益
4. 当前 `TAPT-lite` 的 best 不是出在更晚 step，而是：
   - `ckpt200 = 40.1407`
   - `ckpt300 = 40.0367`

这点很重要，因为它说明：

- 当前 probe 不能简单解释成“downstream continuation 训得不够”

### 3.2 还不知道的事

当前仍然不知道：

1. 如果把 upstream continued pretraining 做得明显更强，是否会开始给出净增益
2. 如果扩大 unlabeled text 范围，是否会改变结果方向
3. normalization / task form 是否属于公开强度的重要来源
4. 更接近公开 monster 的形成方式，到底是：
   - stronger backbone
   - stronger continued pretraining
   - 更强 preprocessing / task form
   - 还是多者叠加

## 4. 新的 `Track B` 假设树（历史原案，现冻结不执行）

接下来只推荐按下面四个假设来推进：

### 4.1 `H3a`: 强度主要来自 stronger continued pretraining intensity

问法：

- 如果只增强 official-only source-side `TAPT` 的训练强度，而 downstream continuation 保持同预算，是否会稳定高于 plain continuation pilot？

这是当前最应该优先回答的问题。

原因：

- `TAPT-lite` 已经证明链路能接通
- 但它太轻，不能代表“continued pretraining 这个上游机制本身已经被充分检验”

### 4.2 `H3b`: 强度主要来自 broader unlabeled corpus scope

问法：

- 如果 continued pretraining 不只吃 official source-side text，而是吃更广的 text-only 语料，是否会给同一条 downstream continuation 带来增益？

注意这里仍然是：

- unlabeled / text-only

不是：

- supervised external mix

### 4.3 `H4`: 强度主要来自 normalization / task-form

问法：

- 在不改模型、不改 decode、不改数据监督边界的前提下，更强但安全的 preprocessing / task-form 变体，是否会给主线带来可重复增益？

这条线应晚于 `H3a / H3b`。

原因：

- 现在继续同时碰 `TAPT + normalization + mix` 会立刻失去归因能力

### 4.4 `H5`: 强度主要来自更接近公开对象的训练形态

问法：

- 如果真正的怪物强度主要来自更重的 backbone / 更接近 full-model shaping 的历史，那么仅靠 current continuation 系列 probe 可能永远只是在怪物表面加分，而不是接近其形成机制。

这条线现在只做：

- 审计
- 设计
- 预算判断

先不直接开大训。

## 6. 2026-03-14 训练计划（历史原案，现冻结不执行）

训练计划按 `U0 -> U1 -> U2 -> U3 -> U4` 五段走。

### 6.1 `U0`: 冻结当前事实

先冻结当前三条结论：

1. `public-weight continuation` 是主线
2. `M025 mix` 是 `healthy but no gain -> stop`
3. `R17 TAPT-lite -> continuation pilot` 是 `inconclusive`

其中 `R17` 当前正式结论应固定为：

- best checkpoint: `ckpt200`
- `geom = 40.1407`
- delta vs plain continuation pilot: `-0.0859`
- verdict: `inconclusive`

在这一步完成前，不要把 `TAPT-lite` 误写成正线。

### 6.2 `U1`: `TAPT-medium official-only -> same pilot`

这是下一枪，优先级最高。

目标：

- 回答 `H3a`
- 判断“当前 upstream continued pretraining 之所以没赢，是不是因为 `lite` 预算太轻”

固定不变项：

- start model: `public model`
- downstream: `official-only continuation`
- pilot gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- LoRA target/modules: 维持当前 `q/k/v/o, r=16, alpha=32`
- 不引入 external supervised mix
- 不同时改 cleaning / tokenizer / decode

upstream 训练建议：

- 语料：full official source-side text
- 不再加 `max_rows=512` cap
- `tapt.max_steps = 300`
- `tapt.eval_steps = 100`
- `mask_ratio = 0.15`
- `max_span_length = 3`
- 其余 LoRA / precision 设置保持与 `R17 TAPT-lite` 一致

downstream continuation 建议：

- 继续沿用当前同预算 pilot
- `max_steps = 300`
- `eval_steps = 100`
- sweep: `100 / 200 / 300`
- 主评测仍为 `raw-row fold0 313`
- 保留 `trunc640 + diag32`

比较对象：

- plain continuation pilot
- `/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/pilot_public_eval_best.json`
- `geom = 40.2266`

判定规则：

1. health 先过线
2. 相对 plain continuation pilot：
   - `geom >= +0.2`: `positive`
   - `-0.2 < delta < +0.2`: `inconclusive`
   - `geom <= -0.2`: `negative`

建议文件名：

- `configs/public_model_r18_public_tapt_medium_20260314.yaml`
- `configs/public_model_r18_public_taptmed_cont_c0_pilot_20260314.yaml`
- `scripts/public_model_r18_public_taptmed_cont_driver.py`

tmux 建议：

- `pub_taptmed`
- `pub_taptmed_pilot`

### 6.3 `U2`: `TAPT-strong official-only -> same pilot`

只有当 `U1 = inconclusive` 时，才升级到这一步。

目标：

- 继续回答 `H3a`
- 把“是不是上游预算太轻”这个解释真正打实或打掉

固定不变项：

- 与 `U1` 完全一致
- 只增强 upstream budget

upstream 训练建议：

- 语料仍是 full official source-side text
- `tapt.max_steps = 900`
- `tapt.eval_steps = 300`
- 其余设置不变

downstream continuation：

- 仍然只跑同预算 `300-step pilot`
- 不升级到 long

原因：

- 如果 `U2` 还没在同预算 pilot 上赢过 plain continuation，就没有资格把 downstream long 也卷进来

判定规则：

- 与 `U1` 相同

建议文件名：

- `configs/public_model_r18_public_tapt_strong_20260314.yaml`
- `configs/public_model_r18_public_taptstrong_cont_c0_pilot_20260314.yaml`

tmux 建议：

- `pub_taptstrong`
- `pub_taptstrong_pilot`

### 6.4 `U3`: `broader text-only corpus -> same pilot`

只有在 `U1 / U2` 没给出明确 `positive` 时，才进入这一步。

目标：

- 回答 `H3b`
- 判断“问题是不是 continued pretraining 语料范围太窄，而不是 continued pretraining 本身无效”

固定不变项：

- downstream continuation 不变
- 主 gate 不变
- decode 不变
- 不引入 external supervised labels

允许变化项：

- 只扩 upstream text-only 语料范围

推荐语料组合：

1. official source-side text
2. 公开可审计的 text-only published/nooverlap source

不推荐：

- 把 `oracc_parallel` 直接当 supervised mix 再混进去
- 一步把 `text-only` 和 `supervised external` 绑成同一实验

训练预算建议：

- 先沿用 `U1 medium` 档
- 如果仍为 `inconclusive`，再考虑 strong 档

### 6.5 `U4`: normalization / task-form probe

只有在 `H3` 梯队跑完后，才考虑这一步。

目标：

- 回答 `H4`

原则：

- 一次只动一个 preprocessing / task-form 轴
- 不和 `TAPT` 同时叠加
- 不和 supervised mix 同时叠加

推荐第一枪：

- 在 plain continuation pilot 上做一个单轴 normalization probe
- 只比较：
  - current official-compatible preprocessing
  - stronger but still safe normalization variant

不推荐第一枪就做：

- tokenizer 改造
- 多个 normalization 规则一起开
- task-form + TAPT 联动

## 7. 这份 `2026-03-14` 计划要怎么执行（历史原案）

### 7.1 先后顺序

严格按这个顺序：

1. `U0` 冻结当前事实
2. `U1` 跑 `TAPT-medium official-only -> same pilot`
3. `U1` 若 `positive`，再决定是否进入同轴 long confirm
4. `U1` 若 `inconclusive`，进入 `U2`
5. `U2` 若仍不正，再进入 `U3`
6. `H3` 梯队跑完后，再考虑 `U4`

### 7.2 当前不允许做的事

在这套 `Track B` 计划下，当前不允许：

1. 因为 `TAPT-lite` 没赢，就直接把 downstream continuation 拉更长
2. 把 `U1/U2` 和 external supervised mix 绑在一起
3. 把 `U1/U2` 和 normalization/task-form 绑在一起
4. 因为差值很小，就直接开 long
5. 把任何单 fold 单 seed 的 tiny positive 写成“已经找到原因”

## 8. 口径纪律（历史口径；当前以文末 `# 工作判断` 为准）

接下来文档里建议统一这样写：

- 我们已经证明 `public-weight continuation` 是稳定成立的交付主线。
- 我们还没有证明自己已经复现了 `public model` 的形成机制。
- 当前 `TAPT-lite -> continuation` 只说明：
  - `continued pretraining` 这个上游方向值得继续 probe
  - 但这一次 `lite` 预算没有给出净增益
- 下一步应该加大的是 upstream mechanism probe，而不是盲目加长 downstream continuation。

一句话总结：

- `Track A` 现在解决的是“怎样在 monster 底座上稳定继续推高”
- `Track B` 接下来要解决的是“monster 到底为什么会变成 monster”
- `Track B` 的正确做法不是乱堆 recipe，而是用固定 downstream pilot 去逐个检验 upstream 机制

# 工作判断（2026-03-15 冻结版）

本节覆盖上文仍保留的 `2026-03-14` 历史原案。按目前事实，`Track B` 不再继续做更细 checkpoint 审计、`winning rows` companion 深挖、或新的大预算 root-cause 宽搜；这些只保留为以后确有必要时的理论缺口。当前工程重点应从“把根因解释得更圆”切回“复现一条更强的 `published-like` 主线，然后嫁接已验证有效的继续训练内容”。

## 1. `Track A` 这段时间做了什么

1. 以公开 `published/public weights` 为起点，固定 `official-only supervised continuation`、固定 `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`、固定 `raw-row fold0 313` 本地 gate。
2. 已经完成 `pilot -> long -> stability pack` 全链条。
3. 当前稳定 incumbent 仍是 `R16 long = 40.4028`；`stability pack` 的 `fold0` seed 汇总为 `40.5055 ± 0.1281`。
4. 这条线回答的是：拿到现成强公开底座以后，怎样继续训练，并稳定拿到不低于 baseline 的实用增益。

## 2. `Track B` 这段时间做了什么

1. 做了上游机制筛查：
   - `H3/H4` 轻量 upstream probe 没有给出足以 promote 的净增益；
   - 这说明“随便加 upstream 预算/变量就会赢”不成立。
2. 做了局部机制压缩：
   - `encoder16/17` 被压到 `companion / corridor`，不再是必要核心；
   - 当前最高置信局部根因冻结在 `decoder5 / FFN wi_1 + wo`。
3. 做了 system-level replication proxy 筛查：
   - `R34 = o_proj + wi_1 + wo` 是当前最强 proxy；
   - `pilot` 最优点在 `ckpt100 = 40.5498`，略高于 `R24 pilot = 40.5412`；
   - `R35 long` 只到 `40.2733`，没有站住；
   - `R34 stability pack` 的 `fold0 seed42/43/44 = 40.5498 / 40.1945 / 40.4182`，`mean/std = 40.3875 / 0.1466`。
4. 需要单独记一条边界：
   - `fold1 seed42 = 48.7263`、`fold2 seed42 = 37.2435` 由于对应不同验证切片，不能拿来替代 `fold0` promote gate；
   - 它们当前只说明 `R34` 跨 fold 异质性仍大。
5. 这条线回答的是：公开强度最像从哪里来；它还没有回答：能不能直接把这套上游 recipe 变成新的稳定主线。

## 3. 当前对 `published model` 继续训练和逆向工程的结论

1. `published model` 上继续训练这件事，已经有稳定工程结论：
   - `Track A` 是当前交付主线；
   - 它不需要再借助 `Track B` 来证明自己。
2. `published model` 的逆向工程这件事，只拿到了“高置信局部线索 + 当前最强 proxy”：
   - 局部线索是 `decoder5 / FFN wi_1 + wo`；
   - system-level proxy 是 `o_proj + wi_1 + wo`；
   - 但这还不是 seed-stable、long-stable 的可 promote 训练 recipe。
3. 因而当前最合理的冻结口径是：
   - `Track A` 已经证明“继续训练能稳定做”；
   - `Track B` 已经证明“根因最像落在什么局部回路”；
   - 但 `Track B` 还没有证明“我们已经复现了 `published model` 的形成机制”。
4. 额外需要改正的一点是：
   - 不能再把“`ckpt300` 最优”当作泛化口径；
   - 它只适用于部分旧 plain-cont pilot；
   - 至少在 `R34` 这条线上，最优点已经明确前移到 `ckpt100`。
5. 所以当前最合理的工程判断不是继续在 `Track B` 上吹毛求疵，而是把它冻结成设计约束，反过来指导下一条更强复现线。

## 4. 下一步最合理的工程路线：先复现强 `published-like` 线，再嫁接继续训练

### 4.1 清洗与数据边界

- 清洗基线固定为 `Gate 0-A / Tier-0`。
- 本轮不把 `Tier-1 / Tier-2`、额外 normalization、tokenizer 改造混进来。
- 监督数据先沿用仓库已验证过的三池结构：
  - 原始对
  - chunk 化样本
  - `Gale-Church` 风格 short-aligned 增广
- 第一版不要再把 external supervised mix、TAPT、cleaning 变化绑成同一枪。

### 4.2 特征工程与 decode

- 特征工程继续沿用当前正式 `chunk + GC` 资产。
- decode 从第一枪开始就固定为公开模型兼容口径：
  - `beam=8`
  - `length_penalty=1.0`
  - `repetition_penalty=1.1`
  - `max_new_tokens=640`
- 旧 `beam=4 / lp=0.7 / max384` 只保留作历史对照，不再作为 promote baseline。

### 4.3 复现主线超参与流程

推荐把“复现 `published-like` 强模型”拆成两段，而不是继续在 `Track B` 上抠 branch：

1. `Stage R1`：先做强 supervised reproduction line
   - 默认主线：`ByT5-base len640 q/v`
   - 推荐先沿用轻量 LoRA 口径启动：`q/v, r=8, alpha=16, dropout=0.0`
   - `bs/grad_acc` 按显存 smoke 先试 `8x3`，不行再试 `6x4`
   - 先只跑 `250-step probe`，看 `ckpt100/150/200/250`
   - 只有当它显著接近或超过当前 matched baseline，才进 full compare
2. `Stage R2`：只有 `R1` 为正，才开更大 adapter 容量
   - `ByT5-base len640 q/k/v/o`
   - `r=16, alpha=32`
   - 显存 smoke 先试 `bs=4, grad_acc=6`，不行再试 `bs=3, grad_acc=8`
   - 目标不是盲目加大，而是回答“大 backbone 之外，大 adapter 是否也必要”
3. `Stage R3`：在复现 winner 上嫁接“新继续训练内容”
   - 起点不再是公开权重，而是 `R1/R2` 的 winner
   - 下游继续训练内容直接复用 `Track A` 已证明有效的 `official-only continuation` 边界
   - 长度继续固定 `len640`
   - 学习率建议先从旧 winner 的 continue 口径起步：`lr=5e-5`
   - 若采用更像 `R16` 的 adapter 形态，再比较 `lr=1e-4` 是否更稳；但不在第一轮同时扫很多超参
   - decode 与评测口径保持不变，不再换表
4. promote 纪律
   - 先 `P`：smoke / `250-step probe`
   - 再 `F`：single winner full compare
   - 只有新 reproduced line 自己站住，才值得再做 continuation graft
   - 只有 graft 后接近或超过现有 `Track A` 基线，才值得考虑替主线

### 4.4 为什么当前路线应这样收敛

- `Track B` 已经提供了局部结构约束，但还没给出稳定 recipe。
- 旧强线已经告诉我们 `chunk + GC + staged continue` 这类工程栈是有产出的。
- 第二阶段复现设计已经说明：如果目标是更像 `published model` 的强线，优先级应是更强 backbone + 新 decode baseline，而不是继续在 `R34` 上反复补解释。
- 因此当前最划算的路线不是“把根因讲到 100% 闭环”，而是“用已知强工程栈做出一个更像 `published model` 的强底座，再把 `Track A` 的 continuation 内容嫁接上去”。

## 5. 旧模型超参与训练策略速查

旧 strong route 的信息现在主要散在这几个位置：

- 文档总览：`docs/public_model_repro_design_2026-03-13.md`
- `Stage1` 配置：`configs/cloud_stage1_len512_lr2e4.yaml`
- `Stage2` 配置：`configs/cloud_stage2_gc_curriculum_cost14_from_s1win.yaml`
- `Continue winner stage` 只在文档里留下了 `continue_s4_bs24_len640_seg5.yaml` 这个生成名；仓库当前没找到这份 YAML，本节把关键超参摘出来备查。

### 5.1 旧主线结构

- backbone: `google/byt5-small`
- LoRA target: `q_proj / v_proj`
- `r=8`
- `alpha=16`
- `dropout=0.0`
- `bias=none`

### 5.2 旧三段训练链

1. `Stage1`
   - `len=512`
   - `lr=2e-4`
   - `per_device_train_batch_size=16`
   - `gradient_accumulation_steps=2`
   - `epochs=30`
   - processed_dir: `data/processed_byt5_chunks`
2. `Stage2`
   - `len=512`
   - `lr=1e-4`
   - `per_device_train_batch_size=16`
   - `gradient_accumulation_steps=2`
   - `epochs=8`
   - processed_dir: `data/processed_byt5_chunks_align_gc_cost14`
3. `Continue winner stage`
   - init: stage2 winner
   - `len=640`
   - `lr=5e-5`
   - `bs=24`
   - `grad_acc=1`
   - `epochs=8`
   - `bf16=true`
   - `gradient_checkpointing=true`

### 5.3 旧数据与 decode

- 数据是三池混合：
  - 原始对
  - chunk
  - short-aligned 局部对齐监督
- 旧默认 decode：
  - `beam=4`
  - `length_penalty=0.7`
  - `max_new_tokens=384`

### 5.4 当前 `Track A` continuation 对照

- 配置：`configs/public_model_r16_public_cont_c0_long_20260313.yaml`
- start model：公开 `published/public model`
- `max_source_length = 640`
- `max_target_length = 640`
- LoRA target：`q/k/v/o`
- `r=16`
- `alpha=32`
- `dropout=0.05`
- `per_device_train_batch_size=8`
- `gradient_accumulation_steps=2`
- `lr=1e-4`
- `epochs=8`
- decode：`beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`

### 5.5 旧路线里真正值得继承和必须升级的部分

- 真正值得继承的是：
  - `Gate 0-A / Tier-0` 边界内的清洗纪律
  - `chunk + GC short-aligned` 的特征工程
  - `staged continue` 的训练流程
- 真正应该升级的是：
  - backbone 强度
  - decode baseline
  - 然后再把 `Track A` 已验证的 continuation 内容嫁接上去
