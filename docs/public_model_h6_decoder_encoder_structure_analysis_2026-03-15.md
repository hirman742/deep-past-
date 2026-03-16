# Public Model H6 Decoder Encoder Structure Analysis
## decoder 4/5 roles + encoder 13~17 asymmetry, 2026-03-15

本稿承接：

- [public_model_h6_module_pair_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md)
- [module_pair_expanded_audit.json](/workspace/deep-past-/reports/public_model_h6_module_pair_expanded_audit_20260315/module_pair_expanded_audit.json)

本稿只做两件事：

1. 补齐 `decoder block 4` 的层内角色分析
2. 把 `encoder 13~17` 的层间差异从“16/17 特例”收紧成更完整的结构口径

本稿不新增训练，也不把这些局部结构误写成完整机制定论。

## 0. 结论先行

当前最值得冻结的结构口径是：

- `decoder block 4` 与 `decoder block 5` 都是 FFN-anchored decoder hotspot，但它们不是完全同一种角色。
- 两层里 `cross.o` 都 consistently 强于 `self.o`，说明 decoder 侧更关键的 attention companion 仍是 cross readout，而不是 self readout。
- `decoder block 4` 更像次一级的 cross-attention handoff / staging 层；`decoder block 5` 仍是最热、最直接贴近收益主支路的主 hotspot。
- `encoder 13~17` 也不是“所有层都同一种 FFN 子分支结构”。
  - 当前更像 odd/even 交替偏置：
  - `13 / 15 / 17` 更偏 `wi_1`
  - `14 / 16` 更偏 `wi_0`
- 其中：
  - `encoder 16` 是最清楚的 `wi_0 + self.o` 层
  - `encoder 17` 是最清楚的 `wi_1 + self.o` 层

## 1. `decoder 4` 补充了什么

### 1.1 `decoder 4` 不是噪声热点

`decoder 4` 在三条线上都保留了和 `decoder 5` 同类的结构：

1. `R24`
   - `ffn.wo = 46.35% local`
   - `ffn.wi_1 = 16.29%`
   - `cross.o = 9.46%`
   - `self.o = 8.07%`
2. `R29`
   - `ffn.wi_1 = 26.32%`
   - `cross.o = 20.55%`
   - `ffn.wi_0 = 17.75%`
   - `self.o = 15.73%`
3. `R30`
   - `ffn.wo = 66.01%`
   - `cross.o = 12.19%`
   - `self.o = 9.25%`

这说明：

- `decoder 4` 不是外围噪声
- 它仍然是同一个 decoder hotspot family 的成员

### 1.2 `cross.o` consistently 强于 `self.o`

`decoder 4`：

1. `R24 / cross.o:self.o = 1.173`
2. `R29 / cross.o:self.o = 1.306`
3. `R30 / cross.o:self.o = 1.318`

`decoder 5`：

1. `R24 / cross.o:self.o = 1.225`
2. `R29 / cross.o:self.o = 1.180`
3. `R30 / cross.o:self.o = 1.098`

这意味着：

- `cross.o` 在两个 decoder hotspot 都持续压过 `self.o`
- `decoder 4` 尤其在 split runs 中更偏 cross-attention side

因此当前更合理的 decoder 侧口径是：

- `cross.o` 是更强的 decoder-side readout companion
- `self.o` 仍在场，但更像次一级配套项

### 1.3 `decoder 4` 比 `decoder 5` 更像辅助 staging 层

pair-level 对比也支持这个分工：

1. `decoder 4 / R29`
   - `wi_1 + cross.o = 46.86% local`
   - `wi_1 + self.o = 42.05% local`
2. `decoder 5 / R29`
   - `wi_1 + cross.o = 51.36% local`
   - `wi_1 + self.o = 49.63% local`

3. `decoder 4 / R30`
   - `wo + cross.o = 78.20% local`
4. `decoder 5 / R30`
   - `wo + cross.o = 76.99% local`

这里更像是在说：

- `decoder 4` 不是次要到可以忽略
- 但真正最贴近主收益支路的，仍然是 `decoder 5`
- `decoder 4` 更像在同一个 decoder readout corridor 里提供上游 handoff / staging

## 2. `cross.o` 和 `self.o` 现在更像怎么分工

把 `decoder 4/5` 合在一起，当前更严格的口径应是：

1. `cross.o` 比 `self.o` 更 consistently 贴着热点 FFN 走
2. 尤其在 split runs 中，`cross.o` 更像承接不完整 FFN 子回路的主要 decoder-side companion
3. `self.o` 也稳定共现，但更像局部补充 readout / stabilization 项

因此如果后续还继续做 decoder-side interaction 分析，优先级更该是：

1. `wi_1 + cross.o`
2. `wo + cross.o`
3. `wi_1 + self.o`

而不是把 `self.o` 放到和 `cross.o` 完全等价的位置。

## 3. `encoder 13~17` 真正显示了什么

### 3.1 不是只有 `16/17` 有差异，`13~17` 整段都有结构

在最能暴露 `wi` 主效应的 `R29` 上，`wi_1 + self.o` 与 `wi_0 + self.o` 的 local pair 差值分别是：

1. `encoder 13 = +1.93 pts`
2. `encoder 14 = -3.80 pts`
3. `encoder 15 = +2.19 pts`
4. `encoder 16 = -6.46 pts`
5. `encoder 17 = +22.14 pts`

在 `R24` 上，这个模式被压平，但方向基本还在：

1. `encoder 13 = +3.03 pts`
2. `encoder 14 = -1.69 pts`
3. `encoder 15 = +1.71 pts`
4. `encoder 16 = -6.61 pts`
5. `encoder 17 = +1.78 pts`

这说明：

- `encoder 13~17` 不是同构复制
- 它更像一段有内部角色分工的 FFN corridor

### 3.2 当前更像 odd/even 交替偏置

最简洁的结构口径是：

1. odd layers `13 / 15 / 17` 更偏 `wi_1`
2. even layers `14 / 16` 更偏 `wi_0`

而且不是所有 odd/even 偏置强度都一样：

1. `encoder 17` 是最强 `wi_1` 层
2. `encoder 16` 是最强 `wi_0` 层
3. `13 / 14 / 15` 更像较弱但稳定的前导结构

### 3.3 `wo` 在 `R24` 里始终是大支，但不等于因果主支

`R24` 的 `encoder 13~17` 中，`wo + self.o` 通常都接近或超过 `50% local`：

1. `encoder 13 = 51.01%`
2. `encoder 14 = 46.46%`
3. `encoder 15 = 52.57%`
4. `encoder 16 = 51.78%`
5. `encoder 17 = 49.74%`

但与此同时：

- `R29` 的高信息密度 pair 仍落在 `wi_0 / wi_1 + self.o`
- 尤其 `encoder 16 / wi_0 + self.o`
- 与 `encoder 17 / wi_1 + self.o`

因此这组数据继续支持：

- `wo` 在结构上始终很大
- 但更像 branch-complete circuit 的 readout branch
- 真正决定“哪层更值钱”的仍然是 `wi` 支路的局部偏置

## 4. 当前最严格的结构口径

把这次扩展 pair audit 和前面的 module-pair、attention compensation、branch synergy 分析合在一起，当前最严格的结构表述应是：

- decoder 侧不是只有一个 `decoder block 5`，而是一条以 `decoder 5` 为主 hotspot、`decoder 4` 为次级 staging hotspot 的 cross-readout corridor；其中 `cross.o` consistently 强于 `self.o`。
- encoder 侧不是一个均匀的 `13~17` 热区，而是一段存在内部偏置的 FFN corridor；当前最像的模式是 odd layers 偏 `wi_1`、even layers 偏 `wi_0`，并在 `encoder 16/17` 达到最清晰的结构分化。

## 5. 对下一步的直接含义

如果后续继续分析或设计新的 interaction probe，当前更合理的优先级应是：

1. `decoder 5 / wi_1 + cross.o`
2. `encoder 16 / wi_0 + self.o`
3. `encoder 17 / wi_1 + self.o`
4. `decoder 4 / wi_1 + cross.o`

其中第 4 条不是因为它比前三条更像主效应，而是因为：

- 它最可能帮助确认 decoder 侧到底是单热点主导
- 还是 `decoder 4 -> decoder 5` 的串联 corridor

一句话说：

- 这一轮结构补充说明：热点并不是“一个点”，而是一条 decoder readout corridor 加一段 encoder FFN corridor；但真正最有因果密度的，仍然是 `decoder 5` 与 `encoder 16/17`。
