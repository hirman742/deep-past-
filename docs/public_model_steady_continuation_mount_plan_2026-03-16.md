# Public Model Steady Continuation Mount Plan
## 2026-03-16

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_public_weight_next_step_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_public_weight_next_step_plan_2026-03-14.md)
- [driver_status.json](/workspace/deep-past-/reports/public_model_repro_himax_20260315/driver_status.json)

本稿只做两件事：

1. 把“当前 `pub_repro_himax` 结束后，如何自动挂一条稳步推进的继续训练”冻结成可执行方案
2. 明确这条挂接线为什么必须回到 `Track A`，而不能把 `pub_repro_himax` 当成新 donor

本稿不改写 `2026-03-14` 冻结判断，不把 `Track B` 临时结果包装成 continuation 主线，不启动新的宽搜。

## 0. 结论先行

当前最合理的挂接方案是：

1. 把 [public_model_repro_himax_20260315](/workspace/deep-past-/reports/public_model_repro_himax_20260315) 只当作当前 GPU 占用方
2. 等它正常写完 `driver_status.json = completed`
3. 立刻自动启动一条保守的 `Track A step-up continuation`
4. donor 固定为 [PUBLIC_MODEL_R16_PUBLIC_CONT_C0_LONG_20260313_fold0/best_model](/workspace/deep-past-/runs/PUBLIC_MODEL_R16_PUBLIC_CONT_C0_LONG_20260313_fold0/best_model)
5. 数据、decode、adapter 形态全部沿用已验证稳定的 `R16` 主线，只把学习率和预算下调成“小步延长”

一句话说：

- `pub_repro_himax` 在这次挂接里只负责“让出卡”
- 真正继续训练的 donor 仍然是已经证明稳定的 `R16 long winner`

## 1. 当前快照

截至 `2026-03-16T00:38:07Z`：

- [public_model_repro_himax_20260315/driver_status.json](/workspace/deep-past-/reports/public_model_repro_himax_20260315/driver_status.json) 仍是 `status=running`
- 当前 stage 是 `stage2_eval`
- [himax_stage2_owi1wo_best.json](/workspace/deep-past-/reports/public_model_repro_himax_20260315/himax_stage2_owi1wo_best.json) 尚未写出

因此，当前最合理的动作不是插队起新训练，而是先把后续 continuation queue 挂好，等 `stage2_eval` 完成后自动接手。

## 2. 为什么 donor 不能用 `pub_repro_himax`

这条边界必须写死：

1. [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md) 的 `2026-03-15 冻结版` 已明确：
   - `Track A` 是继续训练主线
   - `Track B` 只提供局部结构线索，不是 promote-ready recipe
2. 当前 `pub_repro_himax` 属于高规格 reproduction probe，不是稳定 continuation recipe
3. 这次挂接的目标是“稳步推进”，不是再赌一次 root-cause line 会自动转正
4. 因而挂接时必须把：
   - blocker：`pub_repro_himax`
   - donor：`R16 long best_model`
   分开

这也修正了一个常见误区：

- “当前正在跑的东西结束了，所以顺手拿它继续训”并不等于低风险
- 对这次任务来说，低风险恰恰意味着不要把 donor 和 blocker 混为一谈

## 3. 挂接后的继续训练主线

### 3.1 donor

- [PUBLIC_MODEL_R16_PUBLIC_CONT_C0_LONG_20260313_fold0/best_model](/workspace/deep-past-/runs/PUBLIC_MODEL_R16_PUBLIC_CONT_C0_LONG_20260313_fold0/best_model)
- 对应 incumbent 最优点见 [long_public_eval_best.json](/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/long_public_eval_best.json)
- 当前 incumbent:
  - `geom = 40.4028`
  - `checkpoint = ckpt600`

### 3.2 数据与 decode

固定不变：

- processed_dir:
  [processed_public_eval_byt5_akkadian_mbr_local](/workspace/incoming/public_eval_byt5_akkadian_mbr/data/processed_public_eval_byt5_akkadian_mbr_local)
- task / fold / group strategy:
  - `translate Akkadian to English:`
  - `fold0`
  - `group_strategy=auto`
- decode:
  - `beam=8`
  - `length_penalty=1.0`
  - `repetition_penalty=1.1`
  - `max_new_tokens=640`

不做的事：

- 不切到 retrieval-heavy 特征
- 不引入 external mix
- 不改 adapter target
- 不再换 decode 表

### 3.3 超参与预算

新配置：

- [public_model_r16_public_cont_stepup_20260316.yaml](/workspace/deep-past-/configs/public_model_r16_public_cont_stepup_20260316.yaml)

冻结口径：

- LoRA target: `q/k/v/o`
- `r=16`
- `alpha=32`
- `dropout=0.05`
- train batch / eval batch / grad_accum: `8 / 16 / 2`
- `lr = 5e-5`
- `warmup_ratio = 0.02`
- `bf16 = true`
- `gradient_checkpointing = true`

执行预算：

- full train budget: `400 steps`
- eval cadence: `100 steps`
- decode checkpoints: `100 / 200 / 300 / 400`

这个预算是故意保守的：

- 它不是第二条 long run
- 它只是把已经站住的 `R16 winner` 再向前推一小段，检查是否还能在不学坏的前提下拿到非负延展

## 4. stop 条件与验收口径

继续沿用当前健康性红线：

1. `top_repeat_count > 5`
2. `unique_prediction_ratio_pct < 90`
3. `max_len_hit_ratio_pct >= 50`

结果解释时再看两个差值：

1. `delta vs anchor`
2. `delta vs incumbent long`

本轮的文档口径应保持保守：

- 只要健康且接近 incumbent，就算这条“稳态延长”没有学坏
- 只有明显超过 incumbent，才值得讨论后续 promote
- 如果明显回落或出现健康性警报，就把它记为“step-up 不值继续拉长”，而不是继续往下堆

## 5. 可执行挂载实现

### 5.1 continuation driver

- [public_model_r16_public_cont_stepup_driver.py](/workspace/deep-past-/scripts/public_model_r16_public_cont_stepup_driver.py)

职责：

1. 跑 preflight
2. 从 `R16 long best_model` 启动 step-up 训练
3. 自动扫 `ckpt100/200/300/400`
4. 写 `driver_status.json / driver_results.json / route_decision.md`
5. 补 `trunc640` 与 `diag32` 诊断

默认 report dir：

- [public_model_r16_public_cont_stepup_20260316](/workspace/deep-past-/reports/public_model_r16_public_cont_stepup_20260316)

### 5.2 wait-for-himax queue

- [public_model_r16_public_cont_after_himax_queue.py](/workspace/deep-past-/scripts/public_model_r16_public_cont_after_himax_queue.py)

职责：

1. 轮询 [public_model_repro_himax_20260315/driver_status.json](/workspace/deep-past-/reports/public_model_repro_himax_20260315/driver_status.json)
2. 一旦状态变成 `completed`，立即起 step-up continuation
3. 如果 blocker 失败，则 queue 直接停下，不自动乱接

默认 queue report dir：

- [public_model_r16_public_cont_stepup_queue_20260316](/workspace/deep-past-/reports/public_model_r16_public_cont_stepup_queue_20260316)

### 5.3 tmux launcher

- [public_model_r16_public_cont_after_himax_tmux.sh](/workspace/deep-past-/scripts/public_model_r16_public_cont_after_himax_tmux.sh)

默认 session：

- `pub_cont_stepup_wait`

tmux 内会开这些观察窗：

- queue
- route
- status
- results
- cont-status
- cont-results
- cont-train
- cont-eval
- gpu

## 6. 估时

这里分成两段：

1. 等 blocker 结束
   - 无法从当前 `stage2_eval` 精确反推剩余时间
   - 但 queue 每 `15s` 轮询一次，因此 blocker 结束到 continuation 启动的额外延迟不超过一个轮询周期
2. continuation 自身
   - 训练 `400 steps` 预计约 `5-8` 分钟
   - decode sweep + diagnostics 预计约 `8-15` 分钟
   - 因此从真正启动开始，到完整写完结果，大致 `15-25` 分钟

## 7. 这次方案刻意不做什么

本次挂接明确不做：

1. 不把 `pub_repro_himax` 结果当 continuation init
2. 不在挂接阶段同时切换数据、特征工程、adapter 规模和 decode
3. 不在 queue 里串更多后续分支
4. 不把这次 step-up 写成“已经找到更强主线”

这次方案的目标很窄：

- 先把一条真正低风险、可随 blocker 自动接手的继续训练线挂起来
- 再用结果决定是否值得继续向前推
