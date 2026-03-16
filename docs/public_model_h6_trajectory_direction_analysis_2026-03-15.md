# Public Model H6 Trajectory Direction Analysis
## checkpoint evolution + directionality audit, 2026-03-15

本稿承接：

- [public_model_h6_attention_compensation_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_attention_compensation_analysis_2026-03-15.md)
- [public_model_h6_branch_synergy_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_branch_synergy_analysis_2026-03-15.md)
- [checkpoint_trajectory_audit.json](/workspace/deep-past-/reports/public_model_h6_checkpoint_trajectory_audit_20260315/checkpoint_trajectory_audit.json)
- [direction_audit.json](/workspace/deep-past-/reports/public_model_h6_direction_audit_20260315/direction_audit.json)

本稿只做三件事：

1. 看 `100 -> 200 -> 300` checkpoint 里，`wi / wo / attention o/v` 谁先起、谁后补位
2. 看共享模块的更新方向是不是同形
3. 在此基础上，把 `shared shaping` 与 `local synergy` 的边界再压实一步

本稿不新增训练，也不把这些静态/轨迹结果误写成最终机制定论。

## 0. 结论先行

当前最值得冻结的新口径是：

- 三条 `H6` 线的热点模块方向在训练早期就已经基本定型，后期主要不是“换方向”，而是“沿既有方向继续放大或微调”。
- `R24 / R29 / R30` 之间共享模块的方向确实是同形的，但只是中高相似，不是同一个向量的简单缩放。
- 因此当前最合理的统一解释不是二选一，而是：
  - 存在 shared shaping direction
  - 但性能差异更像来自局部 branch completeness / allocation efficiency
  - 而不只是来自“谁把同一方向放大得更大”

更直白地说：

- `shared shaping` 解释了为什么三条线都在打同一批热点层、方向也同形
- `local synergy` 解释了为什么 `R24` 在 attention 更低、局部总更新不更大的情况下仍然分更高

## 1. checkpoint 轨迹在说什么

### 1.1 三条线的大部分质量都在 `100 -> 200` 先长出来

热点层里，三条线的 `wi / wo / attention` 主体增长都集中在早期窗口。

例如 `decoder block 5`：

1. `R24 / 100 -> 200`
   - `wi += 12.396`
   - `wo += 11.501`
   - `attn += 4.304`
2. `R29 / 100 -> 200`
   - `wi += 19.325`
   - `attn += 8.965`
3. `R30 / 100 -> 200`
   - `wo += 19.861`
   - `attn += 8.165`

`200 -> 300` 时，这些增长还在继续，但都缩小了一个量级。

这说明：

- 主 circuit 在早期就被打亮
- 后期更多是 refinement，不是另起一套新结构

### 1.2 `R24` 的收益是“晚出”的，`R30` 的收益是“早饱和”的

public eval `geom`：

1. `R24`
   - `100 = 40.3239`
   - `200 = 40.3395`
   - `300 = 40.5412`
2. `R29`
   - `100 = 40.2439`
   - `200 = 40.3281`
   - `300 = 40.4669`
3. `R30`
   - `100 = 39.7488`
   - `200 = 40.3905`
   - `300 = 40.4032`

这条最值得注意：

1. `R30` 的大部分收益在 `100 -> 200` 已经拿到
   - `+0.6417`
   - `200 -> 300` 只再涨 `+0.0127`
2. `R24` 则相反
   - `100 -> 200` 只涨 `+0.0156`
   - `200 -> 300` 再涨 `+0.2017`

这更像是在说：

- `wo`-heavy 线路能更快把 readout 形状立起来
- 但也更快进入平台
- `wi + wo` 同时存在时，真正的收益优势在后期 refinement 才显出来

### 1.3 `R24` 的 attention share 从一开始就更低，而且一路继续下降

热点层合计 attention share：

1. `R24`
   - `100 = 22.11%`
   - `200 = 19.08%`
   - `300 = 18.62%`
2. `R29`
   - `100 = 37.52%`
   - `200 = 34.21%`
   - `300 = 33.57%`
3. `R30`
   - `100 = 34.21%`
   - `200 = 31.49%`
   - `300 = 31.14%`

这说明：

- `R24` 不是到了后期才学会“少用 attention”
- 它从早期就是更低 attention 的线路
- 后期只是把这条低-attention、branch-complete 的线路进一步做稳

## 2. 方向分析在说什么

### 2.1 单条 run 内，方向在早期就基本定型

`100 -> 200` 的模块方向 cosine 已经很高：

1. `R24`
   - `ffn.wi_0 = 0.9591`
   - `ffn.wi_1 = 0.9644`
   - `ffn.wo = 0.9807`
   - `self.o = 0.9771`
2. `R29`
   - `ffn.wi_0 = 0.9606`
   - `ffn.wi_1 = 0.9646`
   - `self.o = 0.9772`
3. `R30`
   - `ffn.wo = 0.9778`
   - `self.o = 0.9734`

`200 -> 300` 更接近完全同向：

- 各 family 几乎都在 `0.9992 ~ 0.9998`

这意味着：

- 后期主要不是“学出一个新方向”
- 而是在既有方向上继续拉强/微调

### 2.2 run 与 run 之间，方向同形但不完全重合

final checkpoint 的热点共享模块方向平均 cosine：

1. `R24 vs R29`
   - `ffn.wi_0 = 0.6774`
   - `ffn.wi_1 = 0.6711`
   - `self.o = 0.6444`
   - `cross.o = 0.6977`
2. `R24 vs R30`
   - `ffn.wo = 0.8488`
   - `self.o = 0.6371`
   - `cross.o = 0.7057`
3. `R29 vs R30`
   - `self.o = 0.6929`
   - `cross.o = 0.7218`

这组数字说明：

1. 三条线不是在学彼此正交或相反的东西
2. 但它们也不是同一个向量的简单缩放
3. `wo` 的共享方向最稳定
4. attention 侧方向也同形，但弱于 `wo`

### 2.3 跨 run 方向相似度随训练只小幅上升

例如：

1. `R24 vs R29 / ffn.wi_0`
   - `step100 = 0.6134`
   - `step200 = 0.6685`
   - `step300 = 0.6774`
2. `R24 vs R30 / ffn.wo`
   - `step100 = 0.8421`
   - `step200 = 0.8479`
   - `step300 = 0.8488`

这更像是在说：

- shared shaping direction 在早期就已经存在
- 后期并没有出现 dramatic reorientation
- 因此后面的性能差异，更像分支配置与局部效率差异

## 3. 这组轨迹和方向合起来说明什么

### 3.1 不能只用“shared shaping factor”解释完

如果只说“大家都被同一个 shaping factor 沿同一方向推”，那最自然的预期会是：

1. 更高分的线应该主要是更大的总更新
2. 或者更高分的线应该在 attention、FFN 等各支都更大

但当前事实不是这样：

1. `R24` 的热点 attention 一直更低
2. `R24` 的局部总更新通常不更大
3. `R24` 的方向虽然和 split runs 同形，但不是纯缩放关系

因此：

- 纯 shared shaping 不够解释现有差异

### 3.2 也不能把 shared shaping 完全丢掉

反过来，也不能说三条线在学完全不同的东西。

因为：

1. 共享模块方向都是正向中高相似
2. within-run 方向稳定得非常早
3. 跨 run 相似度还会随步数轻微上升

因此更准确的说法应是：

- shared shaping 是背景
- local synergy / branch completeness 是分差来源

### 3.3 当前最接近的机制判别口径

把轨迹、方向、attention compensation 和 branch synergy 合在一起，当前更严格的表述应是：

- `R24 / R29 / R30` 共享一组相近的热点更新方向，说明它们确实都在追同一类上游 shaping clue；但 `R24` 不是简单把这些方向全量放大，而是在 `wi + wo` 同时存在时，以更低的 attention 补位需求和更高的 branch-complete 效率，实现了更高收益。

## 4. 对下一步的直接含义

如果后续还继续分析或设计 probe，当前更合理的纪律是：

1. 不要再把“方向同形”误读成“shared shaping 已经解释完全部分差”
2. 不要再把“哪个分支更大”误读成“哪个分支就是主因”
3. 当前最值得继续压实的是：
   - `decoder 5 / wi_1 -> wo -> cross.o` 到底是不是串联协同
   - 而不是回头再问更粗的单支问题

一句话说：

- 这一轮轨迹与方向分析把结论推进到了：共享 shaping 方向确实存在，但当前分差更像 branch completeness / local synergy，而不是纯缩放。 
