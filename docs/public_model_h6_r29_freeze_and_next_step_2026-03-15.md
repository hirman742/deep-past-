# H6 R29 Freeze And Next Step
## inproj-only split freeze, 2026-03-15

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_h6_r29_inproj_quick_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_inproj_quick_analysis_2026-03-15.md)
- [route_decision.md](/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/route_decision.md)

本稿只做两件事：

1. 正式冻结 `R29 = q/k/v/o + wi_0/wi_1` 的结论
2. 基于该结论，写清楚它现在到底说明了什么

本稿不启动训练，不改写 `Track A` 主线，不把 `positive` 写成机制定论。

## 0. 结论先行

`R29` 的正式冻结表述应为：

- `attention + wi_0/wi_1 only = healthy, positive vs plain continuation pilot, better than R27, close to but still below R24`

这句话同时包含四层意思：

1. `R24` 的正信号并不要求“整块 FFN 全挂上”才出现
2. `wi_0/wi_1` 这支很可能已经解释了 `R24` 的主要增益
3. 但 `R29` 还没有完全等于 `R24`
4. 因此现在最合理的下一枪不是 long，而是剩余 module split

## 1. `R29` 在回答什么问题

`R29` 回答的是：

- 如果保留 baseline 的 `q/k/v/o`，只加入 `wi_0/wi_1`，而明确不加入 `wo`，是否仍能保住 `R24` 的 `H6` 正信号？

固定不变项没有变：

- start model: `public model`
- downstream: `official-only continuation`
- gate: `raw-row fold0 313`
- decode: `beam=8 / lp=1.0 / rep=1.1 / max_new_tokens=640`
- `r=16 / alpha=32`

## 2. 已落地事实

`R29` report dir：

- `/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315`

`R29` best checkpoint：

- `ckpt300`
- `geom / BLEU / chrF++ = 40.4669 / 32.0107 / 51.1568`
- `top_repeat_count = 3`
- `unique_prediction_ratio_pct = 99.0415`
- `max_len_hit_ratio_pct = 12.1406`
- health verdict: `passed`

比较对象：

- plain continuation pilot: `40.2266`
- `R24`: `40.5412`
- `R27`: `40.3165`
- incumbent long: `40.4028`

关键差值：

- delta vs plain continuation pilot: `+0.2403`
- delta vs `R24`: `-0.0743`
- delta vs `R27`: `+0.1504`
- delta vs incumbent long: `+0.0641`

## 3. 正式 verdict

`R29` 的正式 verdict 可以写成：

- `positive`

理由是：

1. 它健康
2. 它相对 plain continuation pilot 的增量已经超过 `+0.2`
3. 它没有出现输出健康退化

但这不等于：

- `R29` 已经取代 `R24`
- `R29` 已经解释完 `R24`
- `R29` 已经够资格直接 long confirm

更准确的写法是：

- `R29 is a positive H6 split probe, but it is still a mechanism clue rather than a new incumbent candidate`

## 4. 现在的结论能够说明什么

当前可以比较稳地说明下面四条：

1. `R24` 的正信号主要不是来自“单纯 attention-only baseline”
   - 因为 `R29` 在不挂 `wo` 的情况下，仍然给出 `+0.2403`
2. `wi_0/wi_1` 很可能已经解释了 `R24` 的主要正增益
   - `R29 40.4669` 与 `R24 40.5412` 的差距只剩 `0.0743`
3. `module split / placement` 当前比 raw capacity 更有解释力
   - `R27` 扩 capacity 反而只有 `40.3165`
   - `R29` 回到 `R24` 原容量并做 placement 拆分，结果回升到 `40.4669`
4. 当前最有信息价值的不是继续做更大 rank
   - 而是继续把 `R24` 拆成剩余模块因子

## 5. 现在的结论还不能说明什么

当前仍然不能直接说明：

1. `wo` 一定是剩余 `0.0743` 增益的唯一来源
2. `R29` 已经足够代表完整 `H6`
3. `Track B` 现在就值得开新的 long confirm
4. 我们已经理解 `public model` 的形成机制

因此，当前最严格的口径仍应是：

- `R29` 让我们更接近解释 `R24` 的因果来源，但还没有完成对残差增益的单点归因`

## 6. 对下一步的最小边界

当前不建议：

1. 把 `R29` 直接开 long
2. 回去重跑 `R27`
3. 再做更大的 `rank/alpha`

如果继续 `Track B` 的 `H6` 线，最合理的下一枪是：

1. `wo-only` split
2. 或同等级别的剩余 module split

比较对象应保持：

1. baseline `40.2266`
2. `R24 40.5412`
3. `R29 40.4669`

一句话说：

- `R29` 现在告诉我们：`R24` 的主效应大概率已经落在 `wi_0/wi_1` 这一支上；下一步该审计的不是更大容量，而是剩余的 `wo` 残差。
