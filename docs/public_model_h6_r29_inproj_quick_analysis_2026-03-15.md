# Public Model H6 R29 Inproj Quick Analysis
## cheap analysis snapshot after launch, 2026-03-15

本稿服务于当前继续执行的 `Track B`：

- live session: `pub_h6inproj_pilot`
- live config: [public_model_r29_public_h6proxy_ffn_inproj_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r29_public_h6proxy_ffn_inproj_c0_pilot_20260315.yaml)
- live report dir: [/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315](/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315)

本稿只做一件事：

1. 在不消耗额外 GPU 预算的前提下，解释为什么 `R29` 是当前最值得挂上的新 `H6` 单轴变量

本稿不写训练结果，不把未完成的 pilot 写成结论。

## Update · 2026-03-15T05:09:22+00:00

`R29` 现已实际跑完；本稿中的主体内容应视为训练前的低成本分析记录。

- `R29` report: [/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/driver_results.json)
- 实际结果：`ckpt300 / geom 40.4669 / delta vs plain continuation pilot = +0.2403 / delta vs R24 pilot = -0.0743 / verdict = positive`
- 正式冻结口径改看：[/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md)

## 0. 结论先行

`R29` 的设计口径是：

- `baseline attention + wi_0/wi_1 only`

也就是：

- 保留 baseline 的 `q/k/v/o`
- 只加入 `wi_0/wi_1`
- 明确不加入 `wo`
- 维持 `R24` 的 `r=16, alpha=32`

这样做的价值在于：

1. 它是对 `R24` 的真正拆半，而不是又做一次超参扩张
2. 它保持 `R24` 的容量口径不变，避免再次混入 `R27` 那种 capacity confound
3. 它能直接回答：`R24` 的正信号更像来自 FFN 输入侧扩展，还是必须要整块 FFN 一起动

## 1. 当前 H6 线上已知事实

当前已知：

1. baseline `q/k/v/o`：
   - `geom 40.2266`
   - trainable params `4,423,680`
2. `R24 = q/k/v/o + wi_0/wi_1/wo`：
   - `geom 40.5412`
   - trainable params `10,764,288`
3. `R27 = R24 same coverage + r32/alpha64`：
   - `geom 40.3165`
   - trainable params `21,528,576`

这三点合起来说明：

1. `H6` 方向本身不是空的
2. 但“继续加大同一 recipe 的容量”不是下一步
3. 下一步应该回到 `R24` 这条正信号本身，做 placement / module-split 拆解

## 2. 这次 cheap analysis 得到的模型级事实

对当前 `public model` 做静音审计后，相关 suffix 计数是：

- `q/k/v/o`: 各 `30`
- `wi_0/wi_1/wo`: 各 `24`

对应的 trainable params 对比如下：

| recipe | target modules | trainable params | trainable ratio |
| --- | --- | ---: | ---: |
| baseline | `q,k,v,o` | `4,423,680` | `0.7548%` |
| `R29` planned | `q,k,v,o,wi_0,wi_1` | `8,650,752` | `1.4655%` |
| `R24` | `q,k,v,o,wi_0,wi_1,wo` | `10,764,288` | `1.8170%` |
| `R27` | `R24` with `r32/alpha64` | `21,528,576` | `3.5692%` |
| hypothetical split | `q,k,v,o,wo` | `6,537,216` | `1.1114%` |

这里最关键的是：

1. `R29` 相对 baseline 新增参数量是：
   - `8,650,752 - 4,423,680 = 4,227,072`
2. `R24` 相对 baseline 新增参数量是：
   - `10,764,288 - 4,423,680 = 6,340,608`
3. 因此 `R29` 已经覆盖了 `R24` 额外参数中的约 `66.7%`
4. 而未挂载的 `wo-only` split 只对应剩余约 `33.3%`

这意味着：

- 如果 `R29` 还能保住 `R24` 的大部分增益，那么 `R24` 的正信号更可能主要来自 `wi_0/wi_1`
- 如果 `R29` 明显掉回 baseline，则 `wo` 或“整块 FFN 一起动”才更关键

## 3. 为什么这枪比别的 H6 变体更值得先做

当前不优先做别的原因是：

1. 不做更大 rank
   - `R27` 已经给出 `inconclusive`
2. 不做 `H4xH6`
   - `R25` 已经说明当前 `H4` 不提供额外收益
3. 不做混合轴
   - 当前最缺的是解释 `R24`，不是继续堆新变量

因此，`R29` 比较像一把“便宜但有信息增益”的刀：

1. 不需要改训练脚本
2. 不需要新 long queue
3. 结果无论正负都能直接约束下一步

## 4. `R29` 能回答什么

`R29` 完成后，最关键的是看三条线：

1. 相对 baseline `40.2266` 是否仍为 `positive`
2. 相对 `R24 40.5412` 是否接近、明显落后，还是直接回到 baseline
3. 输出健康是否保持 `top_repeat_count <= 5`、`unique >= 90%`、`max_len_hit < 50%`

因此它能把下一步决策压缩成下面三类：

1. `R29 ~ R24`
   - 下一步优先考虑 `wi_0/wi_1` 这一支是主效应来源
2. `baseline < R29 < R24`
   - 说明 `R24` 的正信号可能是分块累加，不该再做容量扩张，应该继续做 module split
3. `R29 ~ baseline`
   - 说明 `wo` 或 full-FFN coverage 更值得被单独审计

一句话说：

- `R29` 是当前最便宜、同时又最能解释 `R24` 正信号来源的 `H6` 单轴 pilot。
