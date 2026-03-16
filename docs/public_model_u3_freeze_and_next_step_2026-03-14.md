# U3 Freeze And Next Step
## broader text-only corpus freeze, 2026-03-14

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_public_weight_next_step_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_public_weight_next_step_plan_2026-03-14.md)
- [route_decision.md](/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314/route_decision.md)

本稿只做两件事：

1. 正式冻结 `U3 = broader text-only corpus -> same pilot` 的结论
2. 基于该结论，定义下一条最值得做的单轴动作

本稿不启动训练，不改写 `Track A` 主线，不把 `inconclusive` 写成机制结论。

## 0. 结论先行

`U3` 的正式冻结表述应为：

- `broader text-only corpus -> same pilot = healthy, materially better than U1 official-only medium, but still no net gain vs plain continuation pilot -> inconclusive`

这句话同时包含三层意思：

1. `U3` 不是负例
2. `U3` 也还不是 promote 证据
3. 下一步仍应留在 `H3` 轴内，而不是立刻跳去 `U4`

因此当前更合理的顺序是：

1. 冻结 `U3` 为 `H3b inconclusive`
2. 保持 `public-weight continuation` 为 `Track A` 主线
3. 下一步优先做 `U3-strong broader text-only -> same pilot`
4. 只有 `H3` 梯队收口后，再考虑 `U4 normalization / task-form`

## 1. `U3` 正式冻结

### 1.1 `U3` 在回答什么问题

`U3` 回答的是：

- `H3b`: 如果 continued pretraining 不只吃 official source-side text，而是吃更广的 `text-only` no-overlap 语料，是否会在同预算 downstream pilot 上带来净增益？

固定不变项没有变：

- start model: `public model`
- downstream: `official-only continuation`
- gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`

### 1.2 已落地事实

上游 `R19 TAPT-broad`：

- official unique source rows: `1559`
- published nooverlap rows: `7610`
- combined rows: `9169`
- best adapter: `/workspace/deep-past-/runs/PUBLIC_MODEL_R19_PUBLIC_TAPT_BROAD_20260314/best_model`

下游同预算 pilot：

- report: `/workspace/deep-past-/reports/public_model_r19_public_taptbroad_cont_pilot_20260314`
- best checkpoint: `ckpt300`
- `geom / BLEU / chrF++ = 40.1336 / 31.9504 / 50.4125`
- `top_repeat_count = 2`
- `unique_prediction_ratio_pct = 99.3610`
- `max_len_hit_ratio_pct = 18.5304`
- health verdict: `passed`

比较对象：

- plain continuation pilot: `40.2266`
- `R18 U1 TAPT-medium official-only pilot`: `39.4724`

关键差值：

- delta vs plain continuation pilot: `-0.0930`
- delta vs `R18 U1`: `+0.6612`

### 1.3 正式 verdict

`U3` 的正式 verdict 只能写成：

- `inconclusive`

理由很直接：

1. 它没有坏
2. 它明显好于 `U1 official-only medium`
3. 但它仍然没有赢过 plain continuation pilot
4. 差值 `-0.0930` 落在当前既定的 `-0.2 < delta < +0.2` 区间内

因此当前不应写成：

- `positive`
- `H3b 已成立`
- `broader corpus 已证明是公开强度来源`

也不应写成：

- `negative`
- `broader corpus 无效`

更准确的是：

- `broader text-only scope is a stronger H3 probe than official-only medium, but current medium budget still does not provide net gain over plain continuation`

### 1.4 对路线的影响

`U3` 落地后，路线应固定为：

1. `Track A` 不变
   - `public-weight continuation` 仍是当前 stable incumbent
   - 当前 incumbent 仍是 `geom = 40.4028`
2. `Track B` 也不应 promote `U3`
   - `U3` 不进入 long confirm
   - `U3` 不进入 promotion pack

一句话说：

- `U3` 改善了方向判断，但还没有改写主线

## 2. 这些结果意味着什么

### 2.1 已经知道的事

当前可以硬认：

1. `H3a official-only medium` 不成立为下一条强候选
2. `H3b broader text-only medium` 比 `H3a official-only medium` 更接近有用方向
3. `H3b` 目前仍停在 `inconclusive`

### 2.2 还不知道的事

当前仍不知道：

1. `H3b` 如果加大 upstream budget，是否会越过 `+0.2` promote 线
2. `H3b` 的改善来自更广语料范围本身，还是来自更接近公开对象形成方式的其他伴随因素
3. `U4 normalization / task-form` 是否会比 `H3b strong` 更值得优先做

## 3. 下一步做什么

### 3.1 推荐下一步：`U3-strong broader text-only -> same pilot`

下一步最值得做的单轴动作是：

- 保持 `U3` 语料轴不变
- 只把 upstream budget 从 medium 提升到 strong

建议原因：

1. 计划文档对 `U3` 已经写明：medium 若仍为 `inconclusive`，再考虑 strong 档
2. `U3` 已经证明 broader text-only scope 比 `U1 official-only medium` 更有希望
3. 在这个节点跳去 `U4`，会过早离开当前最接近正信号的 `H3b` 轴

### 3.2 为什么不是 `U4`

当前不建议立刻转 `U4`，原因是：

1. `H3` 梯队还没有在当前最强信号轴上收口
2. `U3` 距 plain continuation pilot 只差 `0.0930`
3. 现在换到 normalization / task-form，会让“是 corpus 还是 preprocessing 起作用”重新混在一起

### 3.3 为什么也不是回头补 `U2`

当前也不建议回头优先补 `U2 strong official-only`，原因是：

1. `U1 official-only medium` 已经是明确 `negative`
2. `U3 broader text-only medium` 已经把结果从 `39.4724` 拉到 `40.1336`
3. 预算优先级应先给当前更接近 promote 线的 `H3b`，而不是回到更弱的 `H3a official-only` 轴

这不是说 `U2` 永远不该做，而是：

- 在当前排序里，`U3-strong` 比 `U2` 更值得先做

## 4. `U3-strong` 的执行边界

如果继续往下做，建议只改以下一项：

- upstream TAPT budget

其余保持不变：

- start model: `public model`
- unlabeled corpus: `official source-side + published nooverlap text-only`
- downstream continuation: `official-only`
- pilot gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- LoRA target/modules: 维持当前 `q/k/v/o, r=16, alpha=32`
- 不引入 external supervised mix
- 不同时改 normalization / tokenizer / decode

建议 strong 档：

- `tapt.max_steps = 900`
- `tapt.eval_steps = 300`
- downstream 仍只跑 `300-step same pilot`
- checkpoint sweep 仍为 `100 / 200 / 300`

判定规则保持不变：

1. health 先过线
2. 相对 plain continuation pilot：
   - `geom >= +0.2`: `positive`
   - `-0.2 < delta < +0.2`: `inconclusive`
   - `geom <= -0.2`: `negative`

只有当 `U3-strong = positive` 时，才值得继续谈 long confirm。

### 4.1 本次落地命名

本次 `U3-strong` 已按下列命名落地：

- upstream config: [public_model_r20_public_tapt_broad_strong_20260314.yaml](/workspace/deep-past-/configs/public_model_r20_public_tapt_broad_strong_20260314.yaml)
- downstream config: [public_model_r20_public_taptbroadstrong_cont_c0_pilot_20260314.yaml](/workspace/deep-past-/configs/public_model_r20_public_taptbroadstrong_cont_c0_pilot_20260314.yaml)
- upstream driver: [public_model_r20_public_taptbroadstrong_driver.py](/workspace/deep-past-/scripts/public_model_r20_public_taptbroadstrong_driver.py)
- downstream driver: [public_model_r20_public_taptbroadstrong_cont_driver.py](/workspace/deep-past-/scripts/public_model_r20_public_taptbroadstrong_cont_driver.py)
- upstream report dir: `/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_20260314`
- downstream report dir: `/workspace/deep-past-/reports/public_model_r20_public_taptbroadstrong_cont_pilot_20260314`
- tmux upstream session: `pub_taptbroadstrong`
- tmux pilot session: `pub_taptbroadstrong_pilot`

## 5. 冻结后的统一口径

接下来文档建议统一这样写：

- 我们已经证明 `public-weight continuation` 是稳定成立的交付主线。
- 我们还没有证明自己已经复现了 `public model` 的形成机制。
- `U3 broader text-only -> same pilot` 当前正式结论是：
  - `healthy`
  - `better than U1 official-only medium`
  - `still no net gain vs plain continuation pilot`
  - `verdict = inconclusive`
- 下一步优先继续 `H3b` strong，而不是立刻跳到 `U4`
