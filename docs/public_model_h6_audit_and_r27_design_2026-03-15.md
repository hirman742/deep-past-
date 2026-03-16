# Public Model H6 Audit And R27 Design
## V1 audit + next single-axis proxy, 2026-03-15

本稿服务于 [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md) 中 `2026-03-15` 新计划的 `V1`。

本稿只做三件事：

1. 审计 `R24/R25/R26` 到底改了什么
2. 解释当前 `H6 proxy` 为什么能在 pilot 上转正、但 long 没有拉开
3. 给出下一枪 `R27` 的单轴设计

## Update · 2026-03-15T03:45:23+00:00

`R27` 现已实际跑完；本稿中的设计部分应视为历史设计记录。

- `R27` report: [/workspace/deep-past-/reports/public_model_r27_public_h6proxy_ffn_rank32_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r27_public_h6proxy_ffn_rank32_pilot_20260315/driver_results.json)
- `R27 -> R28` flow: [/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md](/workspace/deep-past-/reports/public_model_r27_then_r28_h6fidelity_flow_20260315/route_decision.md)
- 实际结果：`ckpt300 / geom 40.3165 / delta vs plain continuation pilot = +0.0899 / delta vs R24 pilot = -0.2247 / verdict = inconclusive`
- `R28` 未启动，因为 `R27` 没达到 `not weaker than R24` gate
- 正式冻结口径改看：[/workspace/deep-past-/docs/public_model_h6_r27_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r27_freeze_and_next_step_2026-03-15.md)

## 0. 结论先行

当前 `H6` 线上已经知道三件事：

1. `R24` 相对 plain continuation pilot 的确是单轴改动
   - 真正发生的改动只有：LoRA target modules 从 `q/k/v/o` 扩到 `q/k/v/o + wi_0/wi_1/wo`
   - 其余数据、preprocess、decode、步数口径都与 plain continuation pilot 对齐
2. `R25` 没有提供额外收益
   - 说明当前安全范围内的 `H4` 叠加，不是 `R24` 正信号的来源
3. `R26` long 贴住 incumbent，但没有拉开
   - 这更像是：当前 `H6 proxy` 已经接近一条有用方向
   - 但当前 proxy 容量还不足以形成新的稳定 ceiling

因此 `R27` 最合理的设计是：

- 继续留在 `H6` 单轴
- 保持 `R24` 的 FFN target coverage 不变
- 只提高 adapter capacity：`r=16 -> 32`，`alpha=32 -> 64`

## 1. `R24` 到底改了什么

以 plain continuation pilot 为基准：

- baseline config: [public_model_r16_public_cont_c0_pilot_20260313.yaml](/workspace/deep-past-/configs/public_model_r16_public_cont_c0_pilot_20260313.yaml)
- `R24` config: [public_model_r24_public_h6proxy_ffn_c0_pilot_20260314.yaml](/workspace/deep-past-/configs/public_model_r24_public_h6proxy_ffn_c0_pilot_20260314.yaml)

关键 diff 只有一项：

- LoRA target modules：
  - baseline: `q_proj, k_proj, v_proj, o_proj`
  - `R24`: `q_proj, k_proj, v_proj, o_proj, wi_0, wi_1, wo`

没有改的项：

- start model：`public model`
- train/val data
- preprocess：`apply_t0_normalize=false`
- task prefix：`translate Akkadian to English:`
- decode：`beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- pilot budget：`300 steps / eval 100 / ckpt100,200,300`

因此，`R24` 的 pilot 正信号可以被解释为：

- wider adaptation coverage into FFN blocks is a real single-axis gain over plain continuation pilot

## 2. `R24/R25/R26` 结果一起看意味着什么

### 2.1 `R24` pilot

- report: [/workspace/deep-past-/reports/public_model_r24_public_h6proxy_ffn_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r24_public_h6proxy_ffn_pilot_20260314/driver_results.json)
- best: `ckpt300 / geom 40.5412 / BLEU 32.1176 / chrF++ 51.1741`
- delta vs plain continuation pilot `40.2266`: `+0.3146`
- health: `passed`

这说明：

- 训练形态 proxy 不是空方向
- 在 pilot 预算下，FFN coverage 比纯 attention-only LoRA 更接近有用方向

### 2.2 `R25` pilot

- report: [/workspace/deep-past-/reports/public_model_r25_public_h4h6_combo_pilot_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r25_public_h4h6_combo_pilot_20260314/driver_results.json)
- best: `ckpt300 / geom 40.5412`
- delta vs plain continuation pilot: `+0.3146`

它与 `R24` 持平，意味着：

- `H4` 当前没有解释掉 `R24` 的正信号
- 当前更值得继续追的是 `H6` 本身，而不是 `H4xH6` 叠加

### 2.3 `R26` long

- report: [/workspace/deep-past-/reports/public_model_r26_public_h5proxy_longconfirm_20260314/driver_results.json](/workspace/deep-past-/reports/public_model_r26_public_h5proxy_longconfirm_20260314/driver_results.json)
- best: `ckpt600 / geom 40.4099 / BLEU 31.8887 / chrF++ 51.2080`
- delta vs plain continuation pilot: `+0.1833`
- delta vs incumbent long `40.4028`: `+0.0071`
- verdict: `inconclusive`

long decode table显示：

- `ckpt600` 最好
- `ckpt800/1000/1200` 明显回落

这说明：

1. `H6 proxy` 的增益不是“越训越高”型
2. 当前这版 proxy 在中段 checkpoint 已接近 plateau
3. 如果继续沿同一容量直接拉更长，信息价值很低

## 3. 为什么下一枪不是别的方向

当前不建议：

- 回头补 `H3`
  - `R18` 和 `R20` 已经给出负面结果
- 再做 `H4`
  - `R21/R22/R23` 全部打平
- 做新的 `H4xH6`
  - `R25` 已经说明当前 `H4` 叠加没有增益

因此下一枪最值得测试的是：

- 同一条 `H6` 线在更高 adapter capacity 下，pilot 能否继续超过 `R24`

## 4. `R27` 设计

`R27` 只改一项：

- LoRA capacity

具体建议：

- base line: `R24`
- target_modules: 维持 `q_proj, k_proj, v_proj, o_proj, wi_0, wi_1, wo`
- `r = 32`
- `alpha = 64`
- 其余保持不变

不改：

- 数据
- preprocess
- task prefix
- decode
- pilot budget

这样做的理由是：

1. 单轴最干净
2. 比 `R24` 更接近“更重的 shaping history”
3. 不会把 `H4` 或 `H3` 重新混进来

## 5. `R27 -> R28` 的放行规则

`R27` pilot 只有同时满足以下条件，才放行 `R28` long：

1. `healthy`
2. 相对 plain continuation pilot：`positive`
3. 相对 `R24 pilot geom 40.5412`：`not weaker`

如果 `R27` 没达到第 3 条，就应直接 stop。

原因：

- `R24` 已经是当前 `H6` 线上最强 pilot
- 新设计如果连 `R24` 都打不过，就没有资格继续消耗 long 预算
