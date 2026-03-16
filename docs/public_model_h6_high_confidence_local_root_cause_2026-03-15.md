# Public Model H6 High-Confidence Local Root Cause
## frozen local-root-cause statement, 2026-03-15

本稿承接：

- [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
- [public_model_h6_module_pair_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_module_pair_analysis_2026-03-15.md)
- [public_model_h6_attention_compensation_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_attention_compensation_analysis_2026-03-15.md)
- [public_model_h6_branch_synergy_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_branch_synergy_analysis_2026-03-15.md)
- [public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md)
- [public_model_h6_trajectory_direction_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_trajectory_direction_analysis_2026-03-15.md)
- [public_model_h6_decoder5_serial_parallel_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder5_serial_parallel_analysis_2026-03-15.md)

本稿只做两件事：

1. 给出当前阶段可以冻结的“高置信局部根因”口径
2. 明确这不是“完整形成机制已经证实”

## 0. 高置信局部根因

当前最严格、且已经可以高置信冻结的局部根因表述应是：

- `public model` 当前最像是某种更强的 `training-shape / adaptation history`，在一个稀疏 FFN-anchored local circuit 上留下了稳定塑形痕迹；这个局部回路的核心位于 `decoder block 5`、`decoder block 4` 和 `encoder block 13~17`，其中：
  - `wi_0 / wi_1` 更像主 computation branch
  - `wo` 更像 readout / completion branch
  - attention `o/v` 更像配套 readout / routing 项
  - 且当 `wi + wo` 同时存在、局部回路更完整时，attention 补位需求会下降

如果把它进一步压缩成一句话：

- 当前最像 `public model` 局部根因的，不是更多普通数据、不是 `H4`、不是 attention 主导、也不是单支模块放大，而是一个被 `training-shape` 塑形的稀疏 FFN 子回路，在 `decoder 5` 与 `encoder 16/17` 最具因果密度，在 `decoder 4` 与 `encoder 13~15` 形成走廊式配套结构。

## 1. 为什么这个口径已经可以高置信冻结

当前已经有六类证据彼此收敛：

1. recipe / score 层
   - `H6` 是唯一持续给出正信号的上游轴
   - `R24 > R29 > R30 > R27 > baseline`
2. adapter family 层
   - 主信号是 `FFN`，不是 attention
   - `wi_0/wi_1` 比 `wo` 更像主效应
3. hotspot layer 层
   - 热点高度集中在 `decoder 5`、`decoder 4`、`encoder 13~17`
4. pair-level 层
   - `decoder 5 / wi_1 + cross.o`
   - `encoder 16 / wi_0 + self.o`
   - `encoder 17 / wi_1 + self.o`
   是最有信息密度的局部 pair
5. compensation / synergy 层
   - `R24` 在热点层 attention 更低
   - 但收益更高
   - 更支持 `branch completeness / local synergy`
6. trajectory / direction 层
   - shared shaping direction 的确存在
   - 但性能差异不像简单方向缩放
   - 更像更高效的局部回路配置

这六条证据已经足够把口径从“机制线索”推进到：

- 高置信局部根因

而不是仅仅：

- 一个有趣但还很松的统计相关性

## 2. 当前可以明确断言什么

当前可以比较硬地断言：

1. `H6 / training-shape` 是当前主解释轴，不是 `H3/H4`
2. 主信号是稀疏 FFN 子回路，不是 attention 主导
3. `wi` 是主 computation branch
4. `wo` 是 readout / completion branch
5. attention `o/v` 是配套/补偿项，而不是主 computation origin
6. 当前分差更像 `branch completeness / local synergy`
   - 不是更大 `rank/alpha`
   - 不是更大 attention
   - 也不是 shared shaping direction 的简单放大

## 3. 当前仍然不能断言什么

当前仍然不能直接断言：

1. 我们已经解释了 `public model` 的完整形成机制
2. `decoder 5 / wi_1 -> wo -> cross.o` 已被严格证明为唯一因果链
3. `R24 - R29` 的剩余增益已被唯一归因
4. 只复制这些热点层或这些模块，就一定能完整复制 `public model`
5. shared shaping 已被完全排除

因此，当前最准确的位置是：

- 高置信局部根因已可冻结
- 完整机制仍需检查计划继续验证

## 4. 对下一阶段的直接含义

既然当前已经拿到高置信局部根因，下一阶段的工作目标就不应再是：

1. 继续补同类静态分析
2. 继续盲开新 pilot
3. 继续争论 `wi` 或 `wo` 谁“更大”

下一阶段唯一合理的目标应是：

- 把当前“高置信局部根因”推进成“经过必要性 / 充分性 / interaction 测试的完整机制检查结果”

一句话说：

- 当前已经足够说“根因主要落在一个被 `training-shape` 塑形的稀疏 FFN 局部回路”，但还不够说“完整形成机制已证实”。
