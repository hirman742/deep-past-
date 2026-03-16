# Public Model H6 R30 Outproj Quick Analysis
## cheap analysis snapshot before pilot result, 2026-03-15

本稿服务于当前继续执行的 `Track B`：

- live session: `pub_h6outproj_pilot`
- live config: [public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml)
- live report dir: [/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315)

本稿只做一件事：

1. 在不消耗额外设计复杂度的前提下，说明为什么 `R30` 是当前最值得补上的 `H6` 单轴变量

本稿不写训练结果，不把未完成的 pilot 写成结论。

## Update · 2026-03-15T06:03:02+00:00

`R30` 现已实际跑完；本稿中的主体内容应视为训练前的低成本分析记录。

- `R30` report: [/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_results.json](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_results.json)
- 实际结果：`ckpt300 / geom 40.4032 / delta vs plain continuation pilot = +0.1766 / delta vs R24 pilot = -0.1380 / delta vs R29 pilot = -0.0637 / verdict = inconclusive`
- 正式冻结口径改看：[/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md)

## 0. 结论先行

`R30` 的设计口径是：

- `baseline attention + wo only`

也就是：

- 保留 baseline 的 `q/k/v/o`
- 只加入 `wo`
- 明确不加入 `wi_0/wi_1`
- 维持 `R24` 的 `r=16, alpha=32`

这样做的价值在于：

1. 它是 `R29` 的互补拆解，而不是另一轮 capacity 扩张
2. 它能直接回答 `R24 - R29` 的剩余差值更像来自 `wo`，还是来自 `wi + wo` 组合效应
3. 它保持 `R24/R29` 的其余条件完全一致，因而最容易解释

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
4. `R29 = q/k/v/o + wi_0/wi_1`：
   - `geom 40.4669`
   - trainable params `8,650,752`

这四点合起来说明：

1. `H6` 方向本身是有效线索
2. “继续把同一 recipe 做大”已经被 `R27` 否掉
3. 当前最该问的是 `R24` 的剩余增益到底落在哪个模块分支上

## 2. 这次 cheap analysis 的关键事实

对当前 `public model`，相关 suffix 计数仍是：

- `q/k/v/o`: 各 `30`
- `wi_0/wi_1/wo`: 各 `24`

对应的 trainable params 对比如下：

| recipe | target modules | trainable params | trainable ratio |
| --- | --- | ---: | ---: |
| baseline | `q,k,v,o` | `4,423,680` | `0.7548%` |
| `R30` planned | `q,k,v,o,wo` | `6,537,216` | `1.1114%` |
| `R29` | `q,k,v,o,wi_0,wi_1` | `8,650,752` | `1.4655%` |
| `R24` | `q,k,v,o,wi_0,wi_1,wo` | `10,764,288` | `1.8170%` |
| `R27` | `R24` with `r32/alpha64` | `21,528,576` | `3.5692%` |

这里最关键的是：

1. `R30` 相对 baseline 新增参数量是：
   - `6,537,216 - 4,423,680 = 2,113,536`
2. `R24` 相对 baseline 新增参数量是：
   - `10,764,288 - 4,423,680 = 6,340,608`
3. 因此 `R30` 只覆盖了 `R24` 额外参数中的约 `33.3%`
4. 而 `R29` 已经覆盖了 `R24` 额外参数中的约 `66.7%`

这意味着：

- 如果 `R30` 依然接近 `R24`，那么剩余主效应更可能落在 `wo`
- 如果 `R30` 明显弱于 `R29`，那么当前更支持 `wi_0/wi_1` 是主效应、`wo` 是次效应
- 如果 `R30` 接近 baseline，则剩余更像 `wi + wo` 组合效应，而不是 `wo` 单独贡献

## 3. 为什么现在必须补这枪

当前不优先做别的原因是：

1. 不做更大 rank
   - `R27` 已经说明那不是信息密度最高的方向
2. 不做 long confirm
   - `R29` 仍未超过 `R24`
3. 不做混合轴
   - 当前最缺的是干净归因，不是继续堆变量

因此，`R30` 是当前最便宜、也最有解释力的补枪：

1. 不需要新 driver
2. 不需要改 decode
3. 结果无论正负，都能直接约束 `R24` 的剩余解释空间

## 4. `R30` 跑完后最关键看什么

`R30` 完成后，最关键看三条线：

1. 相对 baseline `40.2266` 是否仍为 `positive`
2. 相对 `R29 40.4669` 是更强、更弱，还是直接回到 baseline
3. 输出健康是否继续保持 `top_repeat_count <= 5`、`unique >= 90%`、`max_len_hit < 50%`

因此它能把下一步决策收敛成下面三类：

1. `R30 ~ R24`
   - 更支持 `wo` 是剩余主效应来源
2. `baseline < R30 < R29`
   - 更支持 `wi_0/wi_1` 是主效应，`wo` 是次效应
3. `R30 ~ baseline`
   - 更支持 `wo` 单独不够，剩余更像 `wi + wo` 组合效应

一句话说：

- `R30` 是当前最便宜、同时又最能检验 `R24` 剩余增益去向的 `H6` 单轴 pilot。
