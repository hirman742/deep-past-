# H6 R27 Freeze And Next Step
## higher-capacity H6 proxy freeze, 2026-03-15

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_h6_audit_and_r27_design_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_audit_and_r27_design_2026-03-15.md)
- [route_decision.md](/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md)

本稿只做两件事：

1. 正式冻结 `R27 = same FFN coverage + higher LoRA capacity` 的结论
2. 基于该结论，给出下一轮 `H6` 线的最小决策边界

本稿不启动训练，不改写 `Track A` 主线，不把 `inconclusive` 写成机制结论。

## 0. 结论先行

`R27` 的正式冻结表述应为：

- `same H6 FFN coverage + higher LoRA capacity = healthy, still slightly above plain continuation pilot, but weaker than R24 -> inconclusive`

这句话包含四层意思：

1. `H6` 这条大轴没有被 `R27` 否掉
2. 但“只把当前 adapter capacity 做大”不是更好的高保真 proxy
3. `R24` 仍是当前最强的 `H6` pilot
4. `R27` 不应进入 `R28 long confirm`

因此当前更合理的冻结顺序是：

1. 保持 `R24 = current best H6 pilot`
2. 保持 `R26 = long inconclusive`
3. 冻结 `R27 = higher-capacity H6 pilot inconclusive`
4. 如果继续追 `H6`，下一枪必须换新的 `training-shape` 单轴变量，而不是继续放大当前 `rank/alpha`

## 1. `R27` 在回答什么问题

`R27` 回答的是：

- 如果保留 `R24` 的 FFN coverage，不改数据、不改 preprocess、不改 decode，只把 LoRA capacity 从 `r=16, alpha=32` 提到 `r=32, alpha=64`，是否会让 `H6` proxy 在同预算 pilot 上继续变强？

固定不变项没有变：

- start model: `public model`
- downstream: `official-only continuation`
- gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- LoRA target modules: `q_proj, k_proj, v_proj, o_proj, wi_0, wi_1, wo`

## 2. 已落地事实

`R27` report dir：

- `/workspace/deep-past-/reports/public_model_r27_public_h6proxy_ffn_rank32_pilot_20260315`

自动 flow report dir：

- `/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315`

`R27` best checkpoint：

- `ckpt300`
- `geom / BLEU / chrF++ = 40.3165 / 31.8018 / 51.1109`
- `top_repeat_count = 3`
- `unique_prediction_ratio_pct = 99.0415`
- `max_len_hit_ratio_pct = 11.1821`
- health verdict: `passed`

比较对象：

- plain continuation pilot: `40.2266`
- current best `H6` pilot `R24`: `40.5412`
- incumbent long: `40.4028`

关键差值：

- delta vs plain continuation pilot: `+0.0899`
- delta vs `R24` pilot: `-0.2247`
- delta vs incumbent long: `-0.0863`

## 3. 正式 verdict

`R27` 的正式 verdict 只能写成：

- `inconclusive`

理由很直接：

1. 它是健康的
2. 它相对 plain continuation pilot 仍有弱正信号
3. 但它没有达到 `R24 pilot geom 40.5412`
4. 自动 gate 明确要求：
   - `healthy`
   - `positive vs plain continuation pilot`
   - `not weaker than R24`

因此当前不应写成：

- `positive`
- `higher capacity 已证明更接近 public model shaping`
- `值得继续开 R28`

也不应写成：

- `negative`
- `H6 方向已被否掉`

更准确的是：

- `higher adapter capacity alone does not strengthen the current H6 proxy enough to justify long confirm`

### 3.1 自动 stop 结果

`R27 -> R28` flow 的正式结果是：

- `pilot_gate.pass = false`
- `launched_r28 = false`

停止原因在：

- [route_decision.md](/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md)

原文口径是：

- `pilot did not beat-or-match r24 under health and verdict constraints`

## 4. 这些结果意味着什么

### 4.1 已经知道的事

当前可以硬认：

1. `R24` 仍是当前 `H6` 线上最强 pilot
2. `R27` 没有把 `R24` 的收益放大
3. 当前 `H6` 线上真正成立的有用信号，仍然是：
   - `FFN coverage matters`
4. 当前没有证据支持：
   - `same FFN coverage + larger LoRA capacity` 会更接近 `public model`

### 4.2 还不知道的事

当前仍不知道：

1. 更高保真的 `training-shape` 变量是否会继续提升 `R24`
2. `R24` 的正信号究竟更接近：
   - wider module coverage
   - adapter placement
   - adaptation history
   - 还是这些因素中的一部分组合
3. 是否存在比“当前 LoRA recipe 放大容量”更像公开对象形成方式的单轴 proxy

## 5. 对下一步的最小边界

### 5.1 当前不建议做什么

当前不建议：

1. 启动 `R28`
2. 重跑 `R27`
3. 继续在同一套 `FFN rank/alpha` recipe 上往更大容量扩张
4. 把 `H3/H4` 重新叠到这个 recipe 上试图“混出正结果”

### 5.2 如果继续 `H6`，应该怎么约束

如果还要继续 `Track B` 的 `H6` 线，建议只接受下面这类动作：

1. 新设计必须是新的 `training-shape` 单轴变量
2. 比较对象必须同时包含：
   - plain continuation pilot `40.2266`
   - `R24` pilot `40.5412`
3. 如果新 pilot 连 `R24` 都打不过，就直接 stop

一句话说：

- `R27` 冻结后，`H6` 线真正留下来的不是“继续把容量做大”，而是“必须重新定义更像 public model shaping 的单轴变量”。
