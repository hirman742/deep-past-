# Public Model H6 Branch Synergy Analysis
## branch completeness vs shared-shaping audit, 2026-03-15

本稿承接：

- [public_model_h6_module_pair_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md)
- [public_model_h6_attention_compensation_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_attention_compensation_analysis_2026-03-15.md)
- [branch_synergy_audit.json](/workspace/deep-past-/reports/public_model_h6_branch_synergy_audit_20260315/branch_synergy_audit.json)

本稿只做一件事：

1. 继续压实一个前面还没完全收口的问题：
   - `R24` 更像 `wi + wo` 的局部协同/完整性效应
   - 还是更像同一个上游 shaping factor 在不同分支上的共同响应

本稿不新增训练，也不把该结论误写成 `public model` 的完整形成机制。

## 0. 结论先行

当前最值得冻结的新口径是：

- 仅从热点层静态审计看，`R24` 的模式已经不太像“`wi` 与 `wo` 各自响应同一个上游因子，然后 attention 也一起跟着变大”。
- 它更像：
  - `wi` 与 `wo` 在同一热点层里共同把 FFN 子回路补完整
  - 随后 attention 陪跑需求下降
  - 因而在更低的 attention 补位下拿到更高收益

更严格地说，这不是机制定论，而是一个更强的推断：

- 当前证据更偏向 `branch completeness / synergy`
- 比“纯 shared shaping factor, no local synergy”更符合数据

## 1. 方法与指标

本次只复用：

- [local_circuit_audit.json](/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json)

固定层仍是六个热点层：

1. `decoder block 5`
2. `encoder block 13`
3. `encoder block 14`
4. `encoder block 15`
5. `encoder block 16`
6. `encoder block 17`

对每一层，定义四个关键指标：

1. `wi_retention_vs_r29 = R24.wi_sum / R29.wi_sum`
2. `wo_retention_vs_r30 = R24.wo / R30.wo`
3. `branch_completeness_sum = wi_retention_vs_r29 + wo_retention_vs_r30`
4. `attn_relief_vs_split_min = R24.attn_total / min(R29.attn_total, R30.attn_total)`

这里的直观含义是：

1. 如果 `branch_completeness_sum > 1`
   - 说明 `R24` 在同一层里，合起来保住了超过“一条 split 线等价量”的 branch mass
2. 如果同时 `attn_relief_vs_split_min < 1`
   - 说明 `R24` 在保住 branch mass 的同时，用了比任一 split 更少的 attention 陪跑

这组指标本身不能证明因果，但它能直接回答：

- 当前更像“补完整 FFN 子回路后 attention 压下去”
- 还是“大家一起被同一个因子抬起来”

## 2. 硬事实

### 2.1 六个热点层全部出现“branch completeness + attention relief”

六层全部满足：

1. `branch_completeness_sum > 1`
2. `attn_relief_vs_split_min < 1`

汇总指标是：

1. `mean branch_completeness_sum = 1.1933`
2. `mean attn_relief_vs_split_min = 0.6408`
3. `synergy_pattern_hits = 6 / 6`

这说明：

- `R24` 在每个热点层里，都不是靠“少一支 branch、靠更多 attention 补上”
- 相反，它更像在两支 FFN branch 都部分保留的情况下，把 attention 补位压低了

### 2.2 在 `5/6` 个热点层里，`R24` 的 FFN 总量还不低于任一 split

汇总指标：

1. `mean ffn_vs_split_max = 1.0756`
2. `layers_with_ffn_vs_split_max_ge_1 = 5 / 6`

也就是说：

- 在大多数热点层里，`R24` 不只是“同时有一点 wi 和一点 wo”
- 而是 FFN 总量直接达到或超过了最佳 split 线

同时：

1. `mean local_vs_split_max = 0.9062`
2. `layers_with_local_vs_split_max_lt_1 = 5 / 6`

这说明：

- `R24` 在更多时候不是靠更大的局部总更新取胜
- 而是靠更高效的 FFN/attention 分配取胜

### 2.3 两个最有信息密度的例子

`decoder block 5`：

1. `branch_completeness_sum = 1.3198`
2. `attn_relief_vs_split_min = 0.6441`
3. `ffn_vs_split_max = 1.2316`

这更像：

- `wi` 与 `wo` 同时存在后，decoder 侧热点 FFN 回路显著更完整
- attention 补位需求则下降到 split 线的约 `64%`

`encoder block 16`：

1. `branch_completeness_sum = 1.0736`
2. `attn_relief_vs_split_min = 0.6584`
3. `ffn_vs_split_max = 1.0533`
4. `local_vs_split_max = 0.8933`

这更像：

- 在总局部更新更少的前提下，`R24` 仍保住了不低于 split 线的 FFN mass
- attention 则被进一步压低

## 3. 这组事实更支持什么，不支持什么

### 3.1 更支持：branch completeness / local synergy

当前更合理的推断是：

1. `R29` 暴露了 `wi` 主 computation branch
2. `R30` 暴露了 `wo` readout branch
3. `R24` 把两支放回同一热点层后
   - FFN 回路更完整
   - attention 补位更少
   - 整体收益更高

这条链条和现有数据是相互一致的。

### 3.2 较不支持：纯共同响应、无局部协同

如果只是“同一个上游 shaping factor 让 `wi`、`wo`、`attention` 一起响应”，而没有明显局部协同，那么更自然的图景会是：

1. `R24` 的热点 attention 不应系统性低于两条 split
2. `R24` 的局部优势更可能来自更大的总更新，而不是更高效的更新分配

但我们看到的是：

1. attention 在 `R24` 中系统性更低
2. FFN 通常不弱于 split
3. local total 反而通常更低

因此更像是：

- `R24` 在热点层里实现了更高效的 branch-complete circuit

这里需要明确，这是一条推断，不是最终定论。

更准确地说：

- 当前静态证据把概率更多推向“局部协同/完整性效应”
- 但还不能完全排除“更上游 shaping factor 同时塑形多支分支”的解释

## 4. 当前最严格的统一口径

把 root-cause audit、local circuit audit、module-pair audit、attention compensation audit 与这次 branch synergy audit 合在一起，当前最严格的表述应是：

- `public model` 当前最像是某种更强的 `training-shape / adaptation history`，在少数热点 FFN 层里，把 `wi` computation branch 与 `wo` readout branch 共同塑形成了一个更完整的局部回路；attention `o/v` 仍会共现，但在回路更完整时，它更像被压低的配套补偿项，而不是主 computation origin。

## 5. 对下一步的直接含义

如果后续还继续分析，而不是立刻训练，当前最值得做的不是再问：

- `wo` 到底有没有贡献
- attention 是不是越大越好

而是更具体地问：

1. `decoder 5` 的 `wi_1 -> wo -> cross.o` 更像串联协同，还是并联共同响应
2. `encoder 16` 的 `wi_0 + self.o`
3. `encoder 17` 的 `wi_1 + self.o`

如果后续真的要设计新的 `H6` probe，当前更合理的纪律是：

1. 继续围绕 `wi`-anchored pair
2. 把 `wo` 理解成 branch-complete circuit 里的 readout 补全项
3. 不要把“更高 attention”误写成目标

一句话说：

- 这一轮分析把问题进一步从“同一上游共同响应”压向了“热点 FFN 子回路的 branch completeness / local synergy”。
