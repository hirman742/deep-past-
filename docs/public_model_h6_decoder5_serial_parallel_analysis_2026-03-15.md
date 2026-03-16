# Public Model H6 Decoder5 Serial Parallel Analysis
## `wi_1 -> wo -> cross.o` or parallel response?, 2026-03-15

本稿承接：

- [public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_decoder_encoder_structure_analysis_2026-03-15.md)
- [public_model_h6_trajectory_direction_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_trajectory_direction_analysis_2026-03-15.md)
- [module_pair_expanded_audit.json](/workspace/deep-past-/reports/public_model_h6_module_pair_expanded_audit_20260315/module_pair_expanded_audit.json)
- [checkpoint_trajectory_audit.json](/workspace/deep-past-/reports/public_model_h6_checkpoint_trajectory_audit_20260315/checkpoint_trajectory_audit.json)

本稿只回答一个问题：

1. `decoder block 5` 上的 `wi_1 -> wo -> cross.o` 更像串联协同
2. 还是更像几支分支并联响应同一个上游 shaping factor

本稿不新增训练，也不把当前证据误写成最终机制定论。

## 0. 结论先行

当前最严格的口径是：

- `decoder 5` 不像一个“纯并联、谁都只是一起变大”的结构。
- 但它也不像一个“谁在上游，后面每一支都会单调变大”的硬串联链。
- 当前更像的是：
  - `wi_1` 负责主 computation
  - `wo` 负责 downstream readout completion
  - `cross.o` 是最终 export/readout port
  - 当 `wi_1` 与 `wo` 同时存在时，`cross.o` 的补位需求反而下降

因此更准确地说：

- 这是一个带串联角色分工的协同回路
- 但不是“越串联越要求 `cross.o` 更大”的简单单调链

## 1. 为什么它不像纯并联共同响应

如果 `wi_1`、`wo`、`cross.o` 只是共同响应同一个上游因子，而没有明显结构关系，更自然的预期会是：

1. `R24` 里三者都更大
2. 或至少 `cross.o` 不会在 `R24` 中低于 split runs

但实际不是这样：

1. `R29 / decoder 5 / wi_1 + cross.o = 51.36% local`
2. `R30 / decoder 5 / wo + cross.o = 76.99% local`
3. `R24 / decoder 5 / wi_1 + cross.o = 33.39% local`
4. `R24 / decoder 5 / wo + cross.o = 43.40% local`

同时：

1. `R24 decoder 5 / attn_total = 14.206`
2. `R29 = 23.917`
3. `R30 = 22.054`

也就是说：

- 分数最高的 `R24`
- 并没有把 `cross.o` 和 attention 总量一起推得更高

这不支持“只是并联共同响应、谁都一起变大”的最简单解释。

## 2. 为什么它也不像简单硬串联

如果是简单硬串联：

- `wi_1` 做出 computation
- `wo` 读出
- `cross.o` 再继续往外传

那么最直观的预期会是：

- 当 `wi_1` 与 `wo` 同时存在时，`cross.o` 至少不该比 split runs 更低

但实际观察到的是：

1. `R24` 同时拥有 `wi_1` 和 `wo`
2. `R24` 也是性能最高
3. 但 `R24 / cross.o` 反而低于 `R29` 和 `R30`

因此：

- `cross.o` 的角色不像串联链中必须继续扩大的瓶颈
- 它更像当上游 branch 不完整时，需要被额外拉高的 export / readout port

## 3. 轨迹为什么更支持“协同 + relief”

`decoder 5` 的 checkpoint 轨迹：

1. `R24 / 100 -> 200`
   - `wi += 12.396`
   - `wo += 11.501`
   - `cross.o += 1.667`
2. `R29 / 100 -> 200`
   - `wi += 19.325`
   - `cross.o += 3.205`
3. `R30 / 100 -> 200`
   - `wo += 19.861`
   - `cross.o += 2.660`

这条对比最重要：

- 不管是 `wi` 线还是 `wo` 线，单支 split 都需要更大的 `cross.o` 增长
- `R24` 同时拥有两支 branch，但 `cross.o` 增长反而更小

这更像：

- `wi_1` 和 `wo` 共同把局部回路补完整
- 完整后，`cross.o` 只需更少的补位即可完成 export

## 4. 方向分析为什么支持“角色分工存在”

方向 audit 显示：

1. `R24 vs R29 / cross.o mean cosine = 0.6977`
2. `R24 vs R30 / cross.o mean cosine = 0.7057`
3. `100 -> 200` 与 `200 -> 300` 的 `cross.o` within-run cosine 都很高

这说明：

1. `cross.o` 不是三条线各学各的随机东西
2. 它们确实都在学相近的 decoder-side readout direction
3. 但分差不在“有没有这条 direction”
4. 更在“这条 direction 需要被拉高到什么程度”

这也支持：

- `cross.o` 更像相同读出端口上的补位强度差异
- 而不是主 computation origin

## 5. 当前最准确的口径

因此，当前对 `decoder 5` 最准确的表述不是：

1. 完全并联共同响应
2. 简单硬串联单调放大

而是：

- 一个以 `wi_1` 为主 computation branch、`wo` 为 readout completion branch、`cross.o` 为最终 export/readout port 的协同回路；当 `wi_1 + wo` 同时存在时，`cross.o` 仍然重要，但它需要承担的补偿强度会下降。

## 6. 对下一步的直接含义

如果后续还要继续分析或设计 interaction probe，这条问题现在应该被写成：

1. `decoder 5` 的关键不是“`cross.o` 要不要更大”
2. 而是：
   - `wi_1` 是否先产生关键 computation
   - `wo` 是否再把它补全成更完整 readout
   - `cross.o` 是否只是最后的 export port

一句话说：

- 当前 `decoder 5` 更像“带串联角色分工的协同回路”，而不是“纯并联共同响应”或“单调硬串联链”。 
