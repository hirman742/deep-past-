# Cloud Stage2 下一阶段纪律（架构升级轮，2026-03-09）

> 状态更新：
> 本文档对应的“架构升级轮”已在本机完整跑完，三条 `ByT5-base / mT5-base` 主线都在 `probe` 阶段 `reject`。
> 当前有效的下一阶段执行口径，请改读：
> [next-step_taskform_discipline_2026-03-09.md](/workspace/deep-past-/docs/next-step_taskform_discipline_2026-03-09.md)
> 任何仍出现在本文中的 `ByT5-base / mT5-base` 方案，都只代表“已失败归档线”，不应再按它们启动新主队列。

## 0. 文档定位

这份文档用于接替当前 `ByT5-small + len640 + early checkpoint + lp=0.7` 主线之后的下一轮实验。

这轮不再把重点放在：

- 更细的 checkpoint 微扫
- 更细的 `lp` 微扫
- `beam=6`
- `max_new_tokens > 384`
- 最小 Tier-1 清洗

原因已经在本机实验里得到基本确认：

- 当前正式 winner 已经闭环：
  - `len640 seg5 ckpt250 @ beam=4 / lp=0.7 / max_new_tokens=384`
  - full-val `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`
- 当前这条 `ByT5-small` 路线还可以抠出一些波动，但已经进入边际收益很低区。
- 新一轮应转向“架构级或表征级升级”，而不是继续做小修小补。

## 1. 执行前必须先读的文档

新 Codex 或任何接手本轮实验的人，在执行前必须先读：

1. [二期云端 Steer 自迭代方案（修订执行版）](/workspace/deep-past-/docs/cloud_stage2_steer_plan_2026-03-08.md)
   - 这是总纪律来源。
   - `probe / warmup / promote` 的硬停、复核、业务状态定义以它为准。

2. [Cloud Stage2 本机实验清单（2026-03-09）](/workspace/deep-past-/docs/cloud_stage2_machine_experiment_inventory_2026-03-09.md)
   - 这是当前机器上所有已做实验的台账。
   - 先读它，避免重复跑已经证伪的方向。

3. [Cloud Stage2 Tier1 Cleaning Gate Report 2026-03-09](/workspace/deep-past-/docs/cloud_stage2_tier1_cleaning_gate_report_2026-03-09.md)
   - 明确知道：最小 Tier-1 清洗在当前 competition-only 训练条件下没有增益。

4. [semantic_cleaning_spec.md](/workspace/deep-past-/docs/semantic_cleaning_spec.md)
   - 只用于理解为什么当前不把 Tier-1 拉回主线。
   - 不是本轮主线的执行依据。

5. 本文档：
   - [next-step_discipline.md](/workspace/deep-past-/docs/next-step_discipline.md)

## 2. 当前冻结对照线

### 2.1 当前正式赢家

正式 winner 不是 incumbent，而是：

- checkpoint 参考：
  - `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- full-val best：
  - [decode_grid_best_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/decode_grid_best_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json)
- full-val diagnose：
  - [val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json)

冻结 decode 参数：

- `beam=4`
- `lp=0.7`
- `max_new_tokens=384`

### 2.2 当前不再作为主线的方向

以下方向已在当前机器上给出弱证据或负证据，本轮不再优先：

- `beam=6`
- `max_new_tokens=448/512`
- `ckpt260+`
- `qkvo / r16 / dropout=0.05` 在 `ByT5-small` 上的继续扩展
- `parent-balanced / inverse-sample`
- `mixed / progressive length`
- 最小 Tier-1 清洗

## 3. 架构升级轮总纪律

### 3.1 统一阶段命名

每条主线都严格按三段执行：

- `P` = `probe`
- `W` = `warmup`
- `F` = `full/promote-lite`

命名格式：

- 主线 1：
  - `P1_1, P1_2, ...`
  - `W1_1, W1_2, ...`
  - `F1_1, F1_2, ...`
- 主线 2：
  - `P2_1, P2_2, ...`
  - `W2_1, W2_2, ...`
  - `F2_1, F2_2, ...`
- 主线 3：
  - `P3_1, P3_2, ...`
  - `W3_1, W3_2, ...`
  - `F3_1, F3_2, ...`

### 3.2 统一 gate

阶段决策仍沿用旧计划：

- `stage_decision ∈ {accept, review, reject}`
- `business_status ∈ {observe, strong_signal, freeze_candidate, ...}`

具体硬停与复核阈值，直接沿用：

- [cloud_stage2_steer_plan_2026-03-08.md](/workspace/deep-past-/docs/cloud_stage2_steer_plan_2026-03-08.md)

但本轮增加一个执行细则：

- 每条新主线都必须先在同一子集上补一条“当前正式 winner 的 matched baseline”。
- `probe(32)` 和 `warmup(64)` 的判断，不直接拿历史旧 run 的分数比较，而拿“同子集、同 decode 口径下的当前 winner”比较。

### 3.3 统一 decode 口径

除非某条主线另有明确说明，否则本轮默认 decode 口径统一为：

- `beam=4`
- `lp=0.7`
- `max_new_tokens=384`

理由：

- 这是当前正式 winner 的真实最优点。
- `beam=6`、`448/512` 已基本证伪为主杠杆。

### 3.4 统一资源纪律

- 默认只用现有脚本：
  - `scripts/train_mt5_lora.py`
  - `scripts/eval_decode_grid.py`
  - `scripts/diagnose_val_outputs.py`
- 不重开整套 steer controller。
- 长任务一律放 `tmux`。
- 每轮最多并行：
  - `2` 条训练
  - 或 `1` 条训练 + `2` 条小 decode/diagnose
- 不允许在同一轮同时开 `3` 条大训练。

## 4. 主线一：`ByT5-base len640 q/v`

### 4.1 定位

这是最保守的架构升级线。

核心问题：

- 只换 backbone 容量，其他尽量保持与当前 winner 同风格。
- 目标是回答：`ByT5-small` 的瓶颈是不是纯容量上限。

### 4.2 基本参数

- backbone: `google/byt5-base`
- `max_source_length = 640`
- `max_target_length = 640`
- LoRA target: `q/v`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- 优先精度：`bf16`
- 目标有效 batch：`24`

说明：

- 不尝试把 `small` adapter 直接迁到 `base`。
- 这是冷起线，不要伪装成 warm-start。

### 4.3 Probe 簇

- `P1_1`: 显存 smoke，`bs=8, grad_acc=3`
- `P1_2`: 显存 smoke，`bs=6, grad_acc=4`
- `P1_3`: 用通过 smoke 的组合跑 `250 steps`
- `P1_4`: `ckpt100 @ 32-sample anchor`
- `P1_5`: `ckpt150 @ 32-sample anchor`
- `P1_6`: probe winner `diag32`

Probe 决策：

- 如果 `P1_3` 训练本身就不能稳定吃到有效 batch `24`，且 GPU 利用率长期过低，直接 `reject`
- 如果 `P1` 的 best anchor 仍明显低于 matched baseline，直接 `reject`
- 如果 `P1` 只给弱正信号，但健康正常，记 `review`
- 只有 `P1` 明显优于 matched baseline，才进 `W1`

### 4.4 Warmup 纪律

- `W1_1`: 累计到 `600 steps`
- `W1_2`: 比较 `ckpt200 / 300 / 400 / 500 / 600`
- `W1_3`: best checkpoint 跑 `64-sample anchor`
- `W1_4`: winner `diag64`

Warmup 决策：

- 如果冷起线在 `600` 步还没有完全起飞，但 loss、长度和健康都向好，可以 `review`
- 如果 warmup anchor 明显打平或超过当前正式 winner 的本地 matched baseline，进入 `F1`

### 4.5 Full 纪律

- `F1_1`: winner 只跑单点 full-val decode
- `F1_2`: winner full-val diagnose
- `F1_3`: 和当前正式 winner 比较是否切主线

Full 决策：

- 若 full-val `geom > 14.3323` 且健康不差于当前 winner，记 `accept`
- 若 full-val `geom >= 14.5`，记 `business_status = strong_signal`
- 若只小幅领先但健康变差，记 `review`

## 5. 主线二：`ByT5-base len640 qkvo_r16`

### 5.1 定位

这是最激进的容量升级线。

核心问题：

- 如果单纯换 backbone 还不够，这条线验证“更大 backbone + 更大 adapter 容量”是否能一起打开上限。

### 5.2 基本参数

- backbone: `google/byt5-base`
- `max_source_length = 640`
- `max_target_length = 640`
- LoRA target: `q/k/v/o`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.0`
- 目标有效 batch：`24`

说明：

- 这是高风险、高波动线。
- 只有在主线一的 smoke 说明 `ByT5-base` 资源可控时才启动。

### 5.3 Probe 簇

- `P2_1`: 显存 smoke，`bs=4, grad_acc=6`
- `P2_2`: 显存 smoke，`bs=3, grad_acc=8`
- `P2_3`: 用通过 smoke 的组合跑 `250 steps`
- `P2_4`: `ckpt100 @ 32-sample anchor`
- `P2_5`: `ckpt150 @ 32-sample anchor`
- `P2_6`: `ckpt200 @ 32-sample anchor`
- `P2_7`: probe winner `diag32`

Probe 决策：

- 如果这条线连显存和吞吐都太差，不要硬撑到 warmup
- 如果 `P2` 明显低于 `P1`，直接停
- 只有当 `P2` 至少明显优于 `P1`，或与 `P1` 打平但健康更稳，才进 `W2`

### 5.4 Warmup 纪律

- `W2_1`: 累计到 `600 steps`
- `W2_2`: 比较 `ckpt200 / 300 / 400 / 500 / 600`
- `W2_3`: best checkpoint 跑 `64-sample anchor`
- `W2_4`: winner `diag64`

Warmup 决策：

- 这条线只在明显形成上限迹象时才值得进 full
- 否则即使略有正信号，也应先挂起，不抢主线一

### 5.5 Full 纪律

- `F2_1`: winner 单点 full-val decode
- `F2_2`: winner full-val diagnose
- `F2_3`: 只有在 full-val 明显超过 `F1` 时才切主线

Full 决策：

- 若 `F2` 不能稳定打穿 `14.5`，不建议为这条高成本线切主线

## 6. 主线三：`mT5-base len640 q/v`

### 6.1 定位

这是 tokenizer family 切换线。

核心问题：

- 当前 `ByT5` 的字符级路径已经被榨得很深。
- 需要回答“问题是不是出在表征粒度，而不只是 backbone 容量”。

### 6.2 基本参数

- backbone: `google/mt5-base`
- `max_source_length = 640`
- `max_target_length = 640`
- LoRA target: `q/v`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- 目标有效 batch：`24`

说明：

- 这是表征变化线，不应和 `ByT5-base` 在同一轮直接混战。
- 只有在 `ByT5-base` 证明“大 backbone 确实有用”或“仍然不够”后，才有必要认真跑。

### 6.3 Probe 簇

- `P3_1`: tokenizer / throughput smoke
- `P3_2`: 显存 smoke，`bs=6, grad_acc=4`
- `P3_3`: `250 steps` probe
- `P3_4`: `ckpt100 @ 32-sample anchor`
- `P3_5`: `ckpt150 @ 32-sample anchor`
- `P3_6`: best checkpoint `diag32`

Probe 决策：

- 如果 `mT5-base` 在 `probe` 就不能靠近当前 `ByT5` winner 的 matched baseline，直接停
- 不要因为“换 family”就自动给更多 budget

### 6.4 Warmup 纪律

- `W3_1`: 累计到 `600 steps`
- `W3_2`: `ckpt200 / 300 / 400 / 500 / 600`
- `W3_3`: best checkpoint `64-sample anchor`
- `W3_4`: winner `diag64`

### 6.5 Full 纪律

- `F3_1`: 单点 full-val decode
- `F3_2`: full-val diagnose
- `F3_3`: 只在其明显优于当前 `ByT5` 最优线时切主线

## 7. 三条主线的推荐顺序

按优先级：

1. 主线一 `ByT5-base len640 q/v`
2. 主线二 `ByT5-base len640 qkvo_r16`
3. 主线三 `mT5-base len640 q/v`

执行顺序纪律：

- 第一轮只跑主线一
- 只有主线一给出明确正信号，才决定是否开主线二
- 主线三不在第一轮与主线一并跑

## 8. 给新 Codex 的交接说明

### 8.1 先看什么

新 Codex 接手时，先按顺序读：

1. [cloud_stage2_machine_experiment_inventory_2026-03-09.md](/workspace/deep-past-/docs/cloud_stage2_machine_experiment_inventory_2026-03-09.md)
2. [cloud_stage2_tier1_cleaning_gate_report_2026-03-09.md](/workspace/deep-past-/docs/cloud_stage2_tier1_cleaning_gate_report_2026-03-09.md)
3. [cloud_stage2_steer_plan_2026-03-08.md](/workspace/deep-past-/docs/cloud_stage2_steer_plan_2026-03-08.md)
4. [next-step_discipline.md](/workspace/deep-past-/docs/next-step_discipline.md)

### 8.2 已训练好的关键模型与位置

- incumbent baseline adapter：
  - `/workspace/deep-past-/runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/best_model`
  - 作用：一切旧主线 warm-start 的起点

- 当前正式 winner 所在 run：
  - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0`
  - 关键 checkpoint：
    - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
  - 作用：当前正式最优 recipe 的冻结参考

- `lr=4e-5 peak240` run：
  - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_LR4E5_PEAK240_fold0`
  - 作用：证明 `225-235` 有早峰，但尚未超过正式主线

- `bs32_len512` 控制线：
  - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS32_LEN512_fold0`
  - 作用：保底对照，不是当前主线

### 8.3 数据与清洗位置

- 当前正式 processed 数据：
  - `/workspace/deep-past-/data/processed_byt5_chunks_align_gc_cost14`

- Tier-0 processed：
  - `/workspace/deep-past-/data/processed_byt5_chunks_align_gc_cost14_t0`

- Tier-1 processed：
  - `/workspace/deep-past-/data/processed_byt5_chunks_align_gc_cost14_t1`

说明：

- `t0/t1` 已生成，但本轮不建议把 `t1` 当主线输入。

### 8.4 代码与配置位置

- 训练脚本：
  - `/workspace/deep-past-/scripts/train_mt5_lora.py`
- decode 脚本：
  - `/workspace/deep-past-/scripts/eval_decode_grid.py`
- diagnose 脚本：
  - `/workspace/deep-past-/scripts/diagnose_val_outputs.py`
- 清洗实现：
  - `/workspace/deep-past-/cleaning/normalize.py`
  - `/workspace/deep-past-/cleaning/rules/`
- 生成配置目录：
  - `/workspace/deep-past-/runs/STEER/generated_configs`

### 8.5 交接后的第一动作

新 Codex 不应先重开旧实验，也不应先扫新 decode 网格。

第一动作应该是：

1. 校对当前 GPU/磁盘空载状态
2. 用一条 `ByT5-base len640 q/v` 的 smoke 线验证显存与 batch
3. 只有 smoke 稳定，再按 `P1 -> W1 -> F1` 启动架构升级轮

## 9. 最终拍板

这轮之后，默认主线不再是“继续抠 `ByT5-small` 的 decode / checkpoint”。

新的实验主线是：

- 先做 `主线一：ByT5-base len640 q/v`
- 再决定是否需要 `主线二：ByT5-base qkvo_r16`
- 最后才考虑 `主线三：mT5-base`

若没有新的架构级正信号，不要再把预算优先投回 `small` 的局部细调。

## 附录 A：智能分组 / 并线原则

本轮架构升级 probe 默认采用以下分组原则：

1. `matched baseline` 单独先跑  
   原因：后续所有 `probe` 决策都要和同子集、同 decode 口径下的当前正式 winner 比较。

2. 所有 smoke 先串行  
   原因：`ByT5-base`、`mT5-base`、`qkvo_r16` 的训练峰值显存未知，先用最小代价探边界。

3. 通过 smoke 的 `250-step` probe 训练仍然串行  
   原因：训练是本轮最重任务，串行能避免因为 OOM 或调度抖动导致整轮 probe 失真。

4. anchor decode 再并行  
   原因：decode 显存通常远低于训练，且对 GPU burst 更友好，适合按 2-3 条一组并行。

5. 每条主线只对 line winner 跑一次 `diag32`  
   原因：probe 的职责是快速筛线，不应把 diagnose 成本扩散到全部 checkpoint。

6. 如果显存或功耗已经接近上限，优先保证训练任务，缩减 decode 并线数  
   原因：probe 的核心证据来自训练后 line winner，不来自 decode 并行数量本身。

## 附录 B：本轮架构升级 Probe 结果

<!-- NEXT_STEP_ARCH_PROBE_RESULTS_START -->
- matched baseline anchor32: 18.3354 / 11.1352 / 30.1913

### 主线一：ByT5-base len640 q/v
- 通过 smoke: `P1_1, P1_2`
- `ckpt100`: `3.5886 / 1.4989 / 8.5920`
- line winner: `ckpt100` -> `3.5886 / 1.4989 / 8.5920`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界

### 主线二：ByT5-base len640 qkvo_r16
- 通过 smoke: `P2_1, P2_2`
- `ckpt100`: `3.7618 / 1.7423 / 8.1222`
- `ckpt200`: `3.1917 / 1.2308 / 8.2763`
- line winner: `ckpt100` -> `3.7618 / 1.7423 / 8.1222`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界

### 主线三：mT5-base len640 q/v
- 通过 smoke: `P3_1, P3_2`
- `ckpt100`: `0.1437 / 0.0163 / 1.2681`
- line winner: `ckpt100` -> `0.1437 / 0.0163 / 1.2681`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界
<!-- NEXT_STEP_ARCH_PROBE_RESULTS_END -->
