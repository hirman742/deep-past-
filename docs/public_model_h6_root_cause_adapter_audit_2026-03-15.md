# Public Model H6 Root Cause Adapter Audit
## checkpoint-level causal-clue audit, 2026-03-15

本稿承接：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
- [public_model_h6_r29_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md)
- [public_model_h6_r30_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md)
- [adapter_audit.json](/workspace/deep-past-/reports/public_model_h6_adapter_audit_20260315/adapter_audit.json)

本稿只做两件事：

1. 把 `R24 / R29 / R30` 的 best checkpoint adapter 直接拆开，查看真正被更新的是哪些模块、哪些层
2. 在不新增训练的前提下，把当前对 `public model` 的解释推进到更接近根因的层面

本稿不把 adapter audit 误写成“已经复现 public model 形成机制”。

## 0. 结论先行

当前最像根因层解释的表述应是：

- `public model` 的额外强度线索，当前最像是某种更强的 `training-shape / adaptation history` 在一个稀疏 FFN 子回路上留下的痕迹；这个子回路主要集中在 `decoder block 5` 与上部 `encoder FFN`，其中 `wi_0/wi_1` 更像主效应，`wo` 更像读出/补充项。

这句话包含五层意思：

1. 当前主信号确实是 `H6`
2. 当前主信号确实主要落在 `FFN`，不是 attention
3. 这个信号不是全层均匀铺开的，而是高度集中在少数热层
4. `wi_0/wi_1` 比 `wo` 更像“产生有用计算”的那一支
5. `wo` 不是没用，但更像把已有计算读出、传递、整理出去的分支

## 1. 审计对象与方法

本次审计对象固定为三条 `H6` 线的 best checkpoint：

1. `R24 = q/k/v/o + wi_0/wi_1/wo`
2. `R29 = q/k/v/o + wi_0/wi_1`
3. `R30 = q/k/v/o + wo`

对应结果分别是：

1. `R24 = 40.5412`
2. `R29 = 40.4669`
3. `R30 = 40.4032`

静态审计方法是：

1. 读取每个 checkpoint 的 `adapter_model.safetensors`
2. 对每个 LoRA 模块恢复 `delta = B @ A * (alpha / r)`
3. 统计：
   - 每个 family 的 `delta energy`
   - 每层 FFN 的 energy 集中度
   - `R24` 与 `R29/R30` 在共享模块上的 energy 分布相似度

这里的 `delta energy` 不是直接等于“因果贡献”，但它能告诉我们：

- 更新到底集中在哪里
- 哪些分支更像主计算通道
- 单轴 split 与 full probe 是不是在打同一批层

## 2. 硬事实

### 2.1 主信号是 FFN，不是 attention

三条线的 FFN energy share of total 分别是：

1. `R24 = 0.7840`
2. `R29 = 0.6512`
3. `R30 = 0.6367`

这说明当前 `H6` 正信号的 adapter 变化，主要不是 attention 主导，而是 FFN 主导。

### 2.2 在 `R24` 内部，FFN 主体更偏向 `wi`，不是 `wo`

`R24` 的 family share of total：

1. `wi_0 = 0.2038`
2. `wi_1 = 0.2290`
3. `wo = 0.3512`

如果只看 FFN 内部：

1. `wi_0 + wi_1 = 55.2%`
2. `wo = 44.8%`

因此更准确的表述不是“`wo` 最大，所以 `wo` 是主因”，而是：

- `wo` 是最大的单支
- 但输入侧两支 `wi_0/wi_1` 合在一起，仍然是 `R24` FFN 主体

这与性能对照是一致的：

1. `R29` 保住了 `R24` 大部分收益
2. `R30` 只有弱正、且未过 `positive` 门槛

### 2.3 信号是稀疏热层，不是全模型均匀抬升

FFN energy 的层集中度非常高：

`R24`：

1. top-1 FFN layer share = `33.82%`
2. top-3 FFN layers share = `52.11%`
3. top-8 FFN layers share = `76.94%`

`R29`：

1. top-1 = `31.31%`
2. top-3 = `50.28%`
3. top-8 = `75.12%`

`R30`：

1. top-1 = `31.21%`
2. top-3 = `53.90%`
3. top-8 = `79.88%`

三条线共同的热点层基本一致：

1. `decoder block 5`
2. `decoder block 4`
3. `encoder block 17`
4. `encoder block 16`
5. `encoder block 15`
6. `encoder block 14`
7. `encoder block 13`

这说明当前最有信息密度的不是“全模型更强”，而是：

- 一个很窄、很稳定的 FFN 热层子回路被反复打亮

### 2.4 `R24` 与 `R29/R30` 在共享模块上的层分布几乎是同一张图

共享模块 energy 分布相似度：

1. `R24 vs R29 / shared_ffn_wi`
   - cosine = `0.9897`
   - pearson = `0.9857`
2. `R24 vs R30 / shared_ffn_wo`
   - cosine = `0.9977`
   - pearson = `0.9976`

这条很关键。它说明：

1. `R29` 和 `R30` 并不是各自找到了一套完全不同的层
2. 它们与 `R24` 命中的，本质上是同一批热点层
3. 差别主要不在“打哪里”，而在“打到这些层之后，哪一支更能产生有用收益”

## 3. 这组事实真正说明什么

### 3.1 `wi_0/wi_1` 更像主计算支路

当前最重要的因果线索是：

1. `R29` 的得分明显高于 `R30`
2. `R24` 内部 FFN 能量主体仍偏向 `wi_0 + wi_1`
3. `R29` 和 `R24` 在共享 `wi` 模块上的层分布高度一致

这更像是在说：

- `wi_0/wi_1` 负责把有用计算“做出来”

### 3.2 `wo` 更像读出/整理/传输支路

`wo` 也不是噪声，理由是：

1. `R30` 仍然高于 baseline `+0.1766`
2. `R24` 与 `R30` 的共享 `wo` 模块分布几乎完全同形
3. `R24` 的 top module 里，`wo` 占了多个高位

但问题在于：

1. `wo-only` 没能单独站住
2. 它更接近 incumbent long，而不是更接近 `R24`

因此当前更像是：

- `wo` 能把某些有用东西读出来
- 但这些“有用东西”的主要来源，不像是 `wo` 自己单独制造出来的

也就是说，`wo` 当前更像：

- readout
- consolidation
- transport

而不是主 computation origin

### 3.3 `R24` 的优势更像协同，不像单支统治

当前最严格的机制解释不是：

1. `wo` 是唯一来源
2. `wi` 是唯一来源

而是：

1. `wi_0/wi_1` 更像主效应
2. `wo` 更像补充读出项
3. `R24` 的最优表现来自：
   - 同一组热层上
   - `wi` 先把 useful computation 做出来
   - `wo` 再把这部分计算更有效地传出去

所以当前更接近的根因口径应是：

- `public model` 的强度线索，不像是“多一个单独模块就赢”，而像是“一个集中在少数 FFN 热层上的训练形态，把输入侧与输出侧 FFN 支路同时塑形成了协同回路”。

## 4. 这对“public model 怎么来的”意味着什么

这组证据把问题从“recipe 排行榜”推进到了更接近根因的位置：

1. 它不像 `H3`
   - 因为更广或更强的 continued pretraining 没有给出对应收益
2. 它不像 `H4`
   - 因为 normalization / task-form 没有净增益
3. 它也不像“同一 LoRA recipe 继续加大容量”
   - 因为 `R27` 已经说明那条路不对

当前最像的解释是：

- `public model` 之所以强，不是因为多吃了一点普通文本，也不是因为 decode 或清洗更好，而更像因为它经历过一种更强的 shaping history，使少数关键 FFN 层学会了更有效的内部变换与输出读出。

如果把这句话再压缩一点：

- 当前最像根因的不是 `more data`，而是 `specific FFN circuit shaping`

## 5. 这还不能说明什么

当前仍然不能直接说明：

1. 这就是 `public model` 的完整形成机制
2. 公开对象一定经历过 full-model training，而不是别的高保真 shaping history
3. 当前热层就是唯一关键层
4. 只要复制这些模块，就一定能完全复制 `public model`

因此，这仍然是：

- 更接近根因的机制线索

而不是：

- 最终机制定论

## 6. 这意味着接下来该怎么做分析

如果目标是继续逼近根因，而不是继续刷实验数，下一步更合理的是低成本分析，而不是立刻新训：

1. 对 `R24 / R29 / R30` 的热点层做更细的层内审计
   - 先看 `decoder block 5`
   - 再看 `encoder 13~17`
2. 在这些热层上比较：
   - `wi_0 / wi_1 / wo` 的 layer-local delta 规模
   - 与 attention `o / v` 的共现关系
3. 目标不是再问“谁高”，而是问：
   - `wi` 产生的 computation 是不是被 `wo` 读出
   - 还是二者只是共同响应同一个更上游因素

如果后续还要开新训练，唯一合理的方向也应是：

1. interaction-focused `H6`
2. 而不是更大 rank
3. 也不是新的 long confirm

一句话说：

- 当前对 `public model` 的最佳根因解释，是“少数 FFN 热层上的训练形态塑形”，而不是“更大、更久、更多数据”。
