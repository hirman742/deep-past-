# Public Model H6 Local Circuit Analysis
## decoder block 5 + encoder 13~17, 2026-03-15

本稿承接：

- [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
- [local_circuit_audit.json](/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json)

本稿只做一件事：

1. 把 `decoder block 5` 与 `encoder 13~17` 里 `wi_0 / wi_1 / wo` 和 attention `o / v` 的局部共现关系拆清楚

本稿不新增训练，不把局部相关性误写成完整机制定论。

## 0. 结论先行

当前最值得冻结的局部电路口径是：

- `public model` 当前最关键的热点子回路，表现为“FFN 主导、attention o/v 陪跑共现”的局部结构；其中 `wi` 更像主 computation source，`wo` 更像 FFN readout，而 attention `o` 比 `v` 更像同层配套的输出整理项。

这个口径包含四层意思：

1. `decoder block 5` 与 `encoder 13~17` 不是普通热点，而是当前 `H6` 能量最密集的局部区域
2. 这些层里 attention `o / v` 并没有缺席，但它们主要是跟着 FFN 热层一起亮
3. 在这组局部回路里，`o` 的共现强度 consistently 高于 `v`
4. `R24` 的 full probe 不是“谁能更像它的局部能量形状谁就赢”，而是“谁真正保住了更关键的 `wi` 计算支路谁更接近它的收益”

## 1. 范围与方法

本次只分析六个目标层：

1. `decoder block 5`
2. `encoder block 13`
3. `encoder block 14`
4. `encoder block 15`
5. `encoder block 16`
6. `encoder block 17`

比较对象仍是：

1. `R24`
2. `R29`
3. `R30`

局部统计项包括：

1. 每层 `ffn.wi_0 / ffn.wi_1 / ffn.wo / self.o / self.v / cross.o / cross.v` 的 `delta energy`
2. 每层 FFN 与 attention `o/v` 的局部占比
3. 每层 `FFN / attention` 比值
4. `R24` 与 `R29/R30` 在这些层的局部 profile 相似度

## 2. 硬事实

### 2.1 这六层本身就占了大部分 adapter 能量

目标层合计占 total adapter energy：

1. `R24 = 61.08%`
2. `R29 = 61.38%`
3. `R30 = 58.00%`

这说明：

- 当前最该盯的不是更大范围，而就是这一组局部回路

### 2.2 这些层里 FFN 始终主导，但 attention `o/v` 稳定共现

目标层内平均 `FFN share within local`：

1. `R24 = 84.14%`
2. `R29 = 70.65%`
3. `R30 = 72.29%`

同时，逐层 `FFN total` 与 `attention o/v total` 的 layerwise Pearson 很高：

1. `R24 = 0.9989`
2. `R29 = 0.9907`
3. `R30 = 0.9972`

这组数字合起来的含义是：

1. attention `o/v` 不是主导项
2. 但它们也不是随机噪声
3. 更像是和同层 FFN 热点一起同步抬升

### 2.3 在局部共现里，`o` 明显比 `v` 更强

目标层合计：

1. `R24 self.o / self.v = 1.9138`
2. `R29 self.o / self.v = 2.1959`
3. `R30 self.o / self.v = 2.0480`

对 `decoder block 5` 的 cross attention：

1. `R24 cross.o / cross.v = 1.4151`
2. `R29 cross.o / cross.v = 1.2723`
3. `R30 cross.o / cross.v = 1.1622`

因此当前更像的是：

- attention 侧真正和 FFN 热点同步的，优先是 `o`
- `v` 也在动，但 consistently 弱于 `o`

这更像 readout / projection coordination，而不是 value-side 本身在主导新计算。

## 3. `decoder block 5` 在说什么

`decoder block 5` 是最热的单层。

### 3.1 `R24`

`decoder block 5` 占 `R24` 总能量 `32.74%`，层内结构是：

1. `FFN share = 80.99%`
2. `attention share = 19.01%`
3. top modules:
   - `ffn.wo = 36.56%`
   - `ffn.wi_1 = 26.55%`
   - `ffn.wi_0 = 17.89%`
   - `cross.o = 6.84%`

这说明 `block 5` 里不是“只有 FFN”，而是：

- 一个 FFN 主导回路
- 外加一个 nontrivial 的 decoder attention readout 陪跑项

### 3.2 `R29`

`decoder block 5` 占 `R29` 总能量 `30.31%`，层内结构是：

1. `FFN share = 67.26%`
2. `attention share = 32.74%`
3. top modules:
   - `ffn.wi_1 = 39.99%`
   - `ffn.wi_0 = 27.27%`
   - `cross.o = 11.37%`
   - `self.o = 9.64%`

这更像：

- 只保留 `wi` 时，attention `o/v` 会明显抬头补位
- 但真正主导收益的，仍是 `wi`

### 3.3 `R30`

`decoder block 5` 占 `R30` 总能量 `30.20%`，层内结构是：

1. `FFN share = 65.79%`
2. `attention share = 34.21%`
3. top modules:
   - `ffn.wo = 65.79%`
   - `cross.o = 11.20%`
   - `self.o = 10.20%`
   - `cross.v = 9.64%`

这说明：

- `wo-only` 也会把 `block 5` 打得很亮
- 但它更像把局部能量吸到 readout 侧，而不是把主要 computation 建出来

## 4. `encoder 13~17` 在说什么

这五层表现出非常稳定的共同结构。

### 4.1 `R24`

在 `encoder 13~17`：

1. 每层 `FFN share` 都在 `82%~87%`
2. 每层 `FFN / attention` 比值都在 `4.59~6.58`
3. `wo` 通常是单支最大项，但 `wi_0 + wi_1` 与 `wo` 大致五五开

这说明 full probe 的 encoder 热层更像：

- `wi` 与 `wo` 协同
- 但 attention 只占次级位置

### 4.2 `R29`

在 `encoder 13~17`：

1. 每层 `FFN share` 降到 `66.9%~75.0%`
2. `FFN / attention` 比值降到 `2.02~3.01`
3. 仍然由 `wi_0 / wi_1` 主导局部 FFN

这说明：

- 去掉 `wo` 后，attention `o/v` 会接过更多局部份额
- 但注意力补位并没有让 `R29` 崩掉
- 这进一步支持 `wi` 是主 computation source

### 4.3 `R30`

在 `encoder 13~17`：

1. 每层 `FFN share` 约 `72%~74%`
2. `FFN / attention` 比值约 `2.62~2.90`
3. 局部 FFN 完全由 `wo` 独占

这说明：

- `wo` 能稳定占住这些热层
- 但光占住 readout 位，并不足以把收益推到 `R29` 那个级别

## 5. 最关键的反直觉事实

如果只看局部 profile 相似度，`R24` 反而更像 `R30`：

1. `R24 vs R29` 六层平均：
   - cosine = `0.6287`
   - pearson = `0.2552`
2. `R24 vs R30` 六层平均：
   - cosine = `0.7991`
   - pearson = `0.7253`

但性能排序却是：

1. `R24 = 40.5412`
2. `R29 = 40.4669`
3. `R30 = 40.4032`

这条非常重要。它说明：

1. “局部能量形状更像 `R24`”并不等于“收益更像 `R24`”
2. `wo` 可能吸收了大量局部更新能量，使 profile 更像 full probe
3. 但 `wi` 的每单位局部更新，更可能具有更高的 causal efficiency

更直白地说：

- `wo` 看起来更像 full probe
- `wi` 用起来更像 full probe

这正是当前最接近根因的地方。

## 6. 这组局部共现关系意味着什么

当前更合理的机制解释应是：

1. hotspot layers 里真正负责“把有用计算做出来”的，仍更像 `wi`
2. `wo` 负责把这些计算更有效地读出和传出去
3. attention `o` 与 `v` 则像局部配套项
   - 其中 `o` 比 `v` 更重要
   - 更像同层输出整理/投影协同

因此当前最准确的局部机制表述不是：

1. attention 是主因
2. `wo` 是主因
3. `wi` 单支就解释了一切

而是：

- 一个以 `wi` 为主 computation source、`wo` 为 readout source、attention `o/v` 为局部配套项的稀疏 FFN-anchored circuit，在 `decoder block 5` 与 `encoder 13~17` 被反复打亮。

## 7. 接下来该怎么继续逼近根因

如果后续还要继续做“分析”而不是立刻新训，最值得做的是：

1. 对 `decoder block 5` 做层内 module-pair 审计
   - `wi_1 <-> cross.o`
   - `wi_1 <-> self.o`
   - `wo <-> cross.o`
2. 对 `encoder 17` 和 `encoder 16` 做同样审计
   - `wi_0 / wi_1 / wo` 与 `self.o / self.v`
3. 目标不是再看谁大，而是看：
   - `o` 是否总是贴着热点 FFN 走
   - `v` 是否更多只是次级响应

一句话说：

- 当前局部电路分析把问题推进到了“FFN 主导、attention 输出侧配套共现”的层面；它比单纯看分数更接近根因，也更能约束接下来的思路。
