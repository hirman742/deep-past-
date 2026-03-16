# Public Model H6 Attention Compensation Analysis
## hotspot-layer attention relief audit, 2026-03-15

本稿承接：

- [public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)
- [public_model_h6_module_pair_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md)
- [attention_compensation_audit.json](/workspace/deep-past-/reports/public_model_h6_attention_compensation_audit_20260315/attention_compensation_audit.json)

本稿只做一件事：

1. 直接回答一个前面还没压实的问题：
   - attention `o/v` 到底更像主驱动
   - 还是当 `wi` 或 `wo` 被拆开后出现的补偿性抬升

本稿不新增训练，也不把该结论误写成 `public model` 的完整形成机制。

## 0. 结论先行

当前最值得冻结的新口径是：

- 在 `decoder block 5 + encoder 13~17` 这六个热点层里，`R24` 虽然性能最高，但它的 attention `o/v` 总量反而是三条线里最低的。
- 因此当前 attention 更像：
  - 跟着热点 FFN 走的配套 readout / routing 项
  - 在 `wi` 或 `wo` 被拆开时出现的补偿性抬升
  - 而不是主效应来源
- 更直白地说：
  - `R29` 与 `R30` 需要更多 attention 陪跑
  - `R24` 反而用更少的 attention，拿到更高的收益

这进一步支持：

- `wi` 更像主 computation source
- `wo` 更像 readout branch
- attention `o/v` 更像对 FFN 分支完整度的响应，而不是自己单独把收益推上去

## 1. 方法与范围

本次只复用：

- [local_circuit_audit.json](/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json)

比较对象固定为：

1. `R24`
2. `R29`
3. `R30`

固定层仍是六个热点层：

1. `decoder block 5`
2. `encoder block 13`
3. `encoder block 14`
4. `encoder block 15`
5. `encoder block 16`
6. `encoder block 17`

这次只看四类量：

1. 每层 `attention o/v total`
2. 每层 `attention share within local`
3. `R24` 是否在每层都是最小 attention 分支
4. 这种“attention 更低”是否只是因为 `R24` 总更新更少

## 2. 硬事实

### 2.1 `R24` 在六个热点层里，attention 总量都是最低

六层逐层看：

1. `decoder 5`
   - `R24 attn = 14.206`
   - `R29 attn = 23.917`
   - `R30 attn = 22.054`
2. `encoder 13`
   - `R24 = 1.089`
   - `R29 = 2.268`
   - `R30 = 1.719`
3. `encoder 14`
   - `R24 = 1.370`
   - `R29 = 2.855`
   - `R30 = 2.155`
4. `encoder 15`
   - `R24 = 1.987`
   - `R29 = 4.567`
   - `R30 = 3.021`
5. `encoder 16`
   - `R24 = 2.698`
   - `R29 = 5.575`
   - `R30 = 4.098`
6. `encoder 17`
   - `R24 = 2.887`
   - `R29 = 6.160`
   - `R30 = 4.697`

汇总成一句话就是：

- `R24 attn_total` 在 `6/6` 个热点层里都是最低

### 2.2 这不是“R24 什么都更小”，因为它的 local total 并没有普遍最低

同样看这六层：

- `R24 local_total` 只有 `1/6` 层是最低
- `R24 ffn_total` 在 `6/6` 层里都不是最低

也就是说：

- `R24` 变小的不是“整个局部都塌了”
- 真正系统性变小的是 attention 侧

这条非常关键，因为它排除了最弱解释：

- 不是 `R24` 只是“更新总量更小，所以 attention 也顺便更小”

### 2.3 attention share 在 `R24` 里几乎被压到 split runs 的一半

六层合计：

1. `R24`
   - `attn_total_sum = 24.238`
   - `local_total_sum = 139.384`
   - `attn share = 17.39%`
2. `R29`
   - `attn_total_sum = 45.342`
   - `local_total_sum = 147.908`
   - `attn share = 30.66%`
3. `R30`
   - `attn_total_sum = 37.743`
   - `local_total_sum = 123.790`
   - `attn share = 30.49%`

因此：

- `R24` 的热点层 attention 占比明显低于两条 split 线
- `R29` 与 `R30` 的 attention share 反而几乎一样高

### 2.4 `self.o / self.v` 以及 decoder 侧 `cross.o / cross.v`，在 `R24` 里也都是最低

统计结果是：

1. `self.o`
   - `R24` 在 `6/6` 个热点层里都低于 `R29`
   - 也在 `6/6` 个热点层里都低于 `R30`
2. `self.v`
   - `R24` 同样在 `6/6` 个热点层里都低于 `R29`
   - 也都低于 `R30`
3. `decoder block 5 / cross.o`
   - `R24 < R29`
   - `R24 < R30`
4. `decoder block 5 / cross.v`
   - `R24 < R29`
   - `R24 < R30`

这说明：

- 不是只有某一支 attention 被压低
- 而是整个热点 attention accompaniment 都在 `R24` 中被系统性压缩

## 3. 这组事实真正说明什么

### 3.1 attention 更像补偿项，而不是主驱动

如果 attention `o/v` 是当前主驱动，最自然的预期会是：

- 分数最高的 `R24` 至少不该在热点层里 consistently 拿到最低的 attention 总量

但事实恰好相反：

- `R24` 得分最高
- `R24` 的热点 attention 反而最低

因此当前更合理的解释是：

- 当 `wi` 或 `wo` 被拆开时，attention `o/v` 会抬升来补位
- 当 `wi + wo` 同时存在时，FFN 子回路本身更完整，attention 陪跑需求下降

### 3.2 `o` 仍然重要，但它更像“随 FFN 完整度变化的配套项”

这并不意味着 `o` 不重要。

更准确的表述是：

1. `o` 仍然稳定贴着热点 FFN 走
2. 但“更多的 `o`”本身并不等于“更高的收益”
3. 很多时候，更多的 `o` 更像是在暴露：
   - 当前 FFN 主 computation branch 不够完整
   - 于是 attention 侧需要更多 readout / routing 补位

这也解释了为什么：

- `R30` 的局部 profile 更像 `R24`
- 但性能更像次优解

因为：

- `wo + o` 可以把 readout 形状做得很满
- 但真正决定收益的 computation branch，仍然更像 `wi`

### 3.3 当前最接近的机制口径又收紧了一步

把前面的 root-cause audit、local circuit audit、module-pair audit 与这次 attention compensation audit 合在一起，当前最严格的表述应是：

- `public model` 当前最像是某种更强的 `training-shape / adaptation history`，在少数热点 FFN 层上形成了一个以 `wi` 为主 computation source、`wo` 为 readout source、attention `o/v` 为可补偿配套项的稀疏局部回路。

这里 attention `o/v` 的角色更接近：

- responsive accompaniment
- readout / routing relief valve

而不是：

- dominant causal origin

## 4. 对下一步的直接含义

如果继续分析或设计下一枪，当前最重要的约束应是：

1. 不要再把“谁的 attention 更大”当成正向证据
2. 新的 interaction 问题应优先问：
   - 哪个 `wi + o` pair 最像高因果效率支路
   - 而不是哪个 `o/v` bucket 更大
3. `wo` 与 `o` 的大幅共现，现在更该被解释成：
   - readout-heavy / compensation-heavy 形状
   - 不是主 computation clue

一句话说：

- 这一轮分析把 attention 从“可能主因”进一步压回到了“热点 FFN 不完整时会抬升的配套补偿项”。
