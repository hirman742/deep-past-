# H6 R30 Freeze And Next Step
## outproj-only split freeze, 2026-03-15

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_h6_r30_outproj_quick_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_outproj_quick_analysis_2026-03-15.md)
- [route_decision.md](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/route_decision.md)

本稿只做两件事：

1. 正式冻结 `R30 = q/k/v/o + wo` 的结论
2. 基于该结论，写清楚它现在到底说明了什么

本稿不启动训练，不改写 `Track A` 主线，不把 `inconclusive` 写成机制定论。

## 0. 结论先行

`R30` 的正式冻结表述应为：

- `attention + wo only = healthy, slightly above baseline, weaker than R29 and R24, inconclusive`

这句话同时包含四层意思：

1. `wo` 单独不是完全无效
2. 但它也没有给出足以单独站住的 `positive`
3. 当前 `wi_0/wi_1` 比 `wo` 更像 `R24` 的主效应来源
4. 因此 `R24` 的剩余收益更像次效应或组合效应，而不是 `wo` 单支统治

## 1. `R30` 在回答什么问题

`R30` 回答的是：

- 如果保留 baseline 的 `q/k/v/o`，只加入 `wo`，而明确不加入 `wi_0/wi_1`，是否仍能保住 `R24` 的 `H6` 正信号？

固定不变项没有变：

- start model: `public model`
- downstream: `official-only continuation`
- gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- `r=16 / alpha=32`

## 2. 已落地事实

`R30` report dir：

- `/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315`

`R30` best checkpoint：

- `ckpt300`
- `geom / BLEU / chrF++ = 40.4032 / 32.0022 / 51.0095`
- `top_repeat_count = 3`
- `unique_prediction_ratio_pct = 99.0415`
- `max_len_hit_ratio_pct = 13.4185`
- health verdict: `passed`

比较对象：

- plain continuation pilot: `40.2266`
- `R24`: `40.5412`
- `R29`: `40.4669`
- `R27`: `40.3165`
- incumbent long: `40.4028`

关键差值：

- delta vs plain continuation pilot: `+0.1766`
- delta vs `R24`: `-0.1380`
- delta vs `R29`: `-0.0637`
- delta vs `R27`: `+0.0867`
- delta vs incumbent long: `+0.0004`

## 3. 正式 verdict

`R30` 的正式 verdict 应写成：

- `inconclusive`

理由是：

1. 它健康
2. 它相对 plain continuation pilot 有弱正增量
3. 但它没有达到 `+0.2` 的 `positive` 门槛

因此更准确的写法是：

- `R30 is a healthy but inconclusive H6 split probe; it narrows attribution, but it is not a new candidate`

## 4. 现在的结论能够说明什么

当前可以比较稳地说明下面四条：

1. `wo` 单独并不是完全空的
   - `R30` 相对 plain continuation pilot 仍有 `+0.1766`
2. 但 `wo` 当前也不是 `R24` 的主效应来源
   - `R30 40.4032` 低于 `R29 40.4669`
   - 这更支持 `wi_0/wi_1` 是主要正增益来源
3. `R24` 的剩余收益目前更像次效应或组合效应
   - `R24` 同时高于 `R29` 与 `R30`
   - 因而剩余解释空间更像 `wo` 的补充贡献，或 `wi + wo` 的组合效应
4. 当前 `H6` 局部排序已更清楚
   - `R24 > R29 > R30 > R27 > baseline`

## 5. 现在的结论还不能说明什么

当前仍然不能直接说明：

1. `wo` 完全没有贡献
2. `R24 - R29` 的剩余增益已经被确定归因为纯组合效应
3. `Track B` 现在就值得开新的 long confirm
4. 我们已经理解 `public model` 的形成机制

因此，当前最严格的口径仍应是：

- `R30` 让我们更确定了 `wi_0/wi_1` 比 `wo` 更像主效应，但它还没有把 `R24` 的剩余增益完全归因为单一机制`

## 6. 对下一步的最小边界

当前不建议：

1. 把 `R30` 直接开 long
2. 继续做更大的 `rank/alpha`
3. 回去重跑已经冻结的弱轴

如果还要继续 `Track B` 的 `H6` 线，当前更合理的下一步应是：

1. 把 `R24 / R29 / R30` 作为一组完整 split 快照冻结
2. 只在存在新的 interaction-focused 设计时，再考虑下一枪

一句话说：

- `R30` 现在告诉我们：`wo` 有弱贡献，但当前最像 `R24` 主效应的仍是 `wi_0/wi_1`；剩余解释空间更像补充项或组合效应，而不是 `wo-only` 本身。
