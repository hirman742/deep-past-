# Public Model H6 Encoder Core Verification
## full-row falsification of `encoder 16/17 necessary core`, 2026-03-15

本稿基于以下产物：

- [mechanism_eval_results.json](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/mechanism_eval_results.json)
- [mechanism_eval_table.csv](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/mechanism_eval_table.csv)
- [sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/sample_slice_summary.json)
- [sample_slice_rows.csv](/workspace/deep-past-/reports/public_model_h6_encoder_core_full_20260315/sample_slice_rows.csv)

本稿只回答一个问题：

- `encoder 16/17` 能不能从此前的 `companion / corridor` 假说，升级成已经被 full-row 证实的 `necessary core`

结论先写在前面：

- 不能。
- `encoder 16/17` 现在可以更硬地冻结为：最多是带有弱局部专属性的 `companion / corridor`，不是当前 `public model` 局部根因里的必要核心。

## 0. 判据

这轮 full-row 验证使用三层判据：

1. `M1` 必要性：
   - 直接对 `R24` 做 `encoder 16/17` pair 与 FFN-all ablation。
   - 如果这些位点真是必要核心，full-row 应出现稳定、明显掉分。
2. `M2` donor reversion：
   - 把 `r29` 或 `r30` 在这些位点的局部状态回填进 `r24`。
   - 如果这些位点承载 `r24` 的关键优势，则替换成 weaker donor 的局部状态应稳定拉坏 `r24`。
3. winning-row slice：
   - 只看 `r24` 真正赢过 `r29/r30` 的样本。
   - 排除“均值掩盖局部塌陷”的可能。

full-row references 固定为：

- `r24_ref = 40.5412`
- `r29_ref = 40.4669`
- `r30_ref = 40.4032`

## 1. `M1` full-row 必要性

`M1` 的 six-shot 结果如下：

1. pair ablation
   - `ec_m1_ablate_e16_pair`，手术质量占比 `2.19%`：`40.5912`，相对 `r24_ref` `+0.0500`
   - `ec_m1_ablate_e17_pair`，手术质量占比 `2.68%`：`40.5741`，相对 `r24_ref` `+0.0329`
   - `ec_m1_ablate_e16e17_pair_union`，手术质量占比 `4.87%`：`40.6066`，相对 `r24_ref` `+0.0654`
2. FFN-all ablation
   - `ec_m1_ablate_e16_ffn_all`，手术质量占比 `5.43%`：`40.5092`，相对 `r24_ref` `-0.0320`
   - `ec_m1_ablate_e17_ffn_all`，手术质量占比 `7.08%`：`40.6204`，相对 `r24_ref` `+0.0792`
   - `ec_m1_ablate_e16e17_ffn_all`，手术质量占比 `12.51%`：`40.6513`，相对 `r24_ref` `+0.1101`

这组结果已经足够单独否掉“必要核心”：

1. `pair` 三枪全部不降反升。
2. 把 `encoder 16/17` 两层整个 `FFN` 六支都清掉，full-row 仍然是全表最高值之一。
3. 唯一的负项 `ec_m1_ablate_e16_ffn_all = -0.0320` 也只是轻微扰动，完全不符合“必要核心被打掉”的轮廓。

因此，`M1` full-row 本身已经给出系统性反证：

- `encoder 16/17` 不是当前 `R24` 收益的必要承重结构。

## 2. `M2` donor reversion

`M2` 把 `r29/r30` 在这些位点的局部状态回填到 `r24`，结果分成两类。

### 2.1 `r29 -> r24`

1. `ec_r29_revert_e16_pair`，占比 `2.19%`：`40.5721`，相对 `r24_ref` `+0.0309`
2. `ec_r29_revert_e17_pair`，占比 `2.68%`：`40.4477`，相对 `r24_ref` `-0.0935`
3. `ec_r29_revert_e16e17_pair_union`，占比 `4.87%`：`40.4174`，相对 `r24_ref` `-0.1238`

### 2.2 `r30 -> r24`

这里是更强的 stress test，因为 `r30` 缺失对应 `wi`，因此：

- `e16` / `e17` reversion 实际是 `replaced=1, zeroed=1`
- `union` reversion 实际是 `replaced=2, zeroed=2`

结果为：

1. `ec_r30_revert_e16_pair`，占比 `2.19%`：`40.5931`，相对 `r24_ref` `+0.0520`
2. `ec_r30_revert_e17_pair`，占比 `2.68%`：`40.6400`，相对 `r24_ref` `+0.0988`
3. `ec_r30_revert_e16e17_pair_union`，占比 `4.87%`：`40.6778`，相对 `r24_ref` `+0.1366`

`M2` 的正确解释不是“必要性翻盘”，而是：

1. `e16` 在 donor reversion 下两次都不伤，说明它没有承重地位。
2. `r29` 的 `e17 / union` 出现了轻微负项，说明这些位点上可能带有一点 `r24` 局部专属性。
3. 但这种专属性并不稳健：
   - 更苛刻的 `r30` harsher reversion 反而全部转正
   - 因此更像是 `r29` donor-specific mismatch
   - 不是“只要换掉这些位点，`r24` 就会被打坏”的必要核心轮廓

所以，`M2` 最多支持：

- `encoder 17` 可能带有弱局部专属性

但不支持：

- `encoder 16/17` 是必要核心

## 3. winning-row sample slice

`left_beats_all` 切片只保留 `r24` 真正赢过 `r29/r30` 的样本，共 `64` 条。

这个切片上的 baseline 是：

- `r24_ref = 41.8744`
- `r29_ref = 40.0582`
- `r30_ref = 40.0349`

也就是：

- `r24` 在这些真赢样本上，仍对 `r29/r30` 保有约 `1.82~1.84` 的 corpus geom 优势

在这 `64` 个样本上，所有 encoder-core 变体都落在：

- `41.3535 ~ 41.7578`

也就是说：

1. 这些手术会削掉一部分 `r24` 的 winning-row 优势。
2. 但削掉的量级只有 `-0.1166 ~ -0.5209`。
3. 这远小于 `r24` 相对 `r29/r30` 的原始优势。
4. 没有任何一个 encoder 变体把 `r24` 压回 `r29/r30` 的水平。

因此，winning-row slice 给出的读法是：

- `encoder 16/17` 不是“完全没用”
- 但它们也不是决定 `r24` 主优势能否存在的必要核心
- 更像是在 `r24` 已经成立的主回路上，提供次级 companion / corridor refinement

## 4. 这轮之后可以冻结什么

这轮 full-row 验证后，可以把口径收紧成下面四条：

1. `encoder 16/17 necessary core` 假说已被 full-row 否掉。
2. `encoder 16` 可以更硬地写成：
   - 非必要核心
   - 连稳定局部专属性都没有显示出来
3. `encoder 17` 可以更保守地写成：
   - 可能带有弱局部专属性
   - 但不具备必要核心证据
4. 因此当前最准确的统一口径应是：
   - `encoder 16/17 = companion / corridor`
   - 不是当前已证实的局部根因核心

## 5. 这轮不能说明什么

当前仍然不能直接说明：

1. `encoder 16/17` 在所有更温和 intervention 下都完全无关
2. encoder corridor 不存在任何上下文依赖作用
3. 完整机制已经闭环

因此，最准确的阶段性结论不是：

- encoder 完全无用

而是：

- encoder 不是必要核心，最多是弱 companion / corridor

## 6. 对下一步的直接含义

这轮之后，最合理的资源分配应是：

1. 关闭 `encoder 16/17 necessary core` 这条支线，不再继续投入更多 hard ablation / hard reversion。
2. 把 `complete mechanism check` 的主因焦点继续压回：
   - `decoder 5 / FFN`
   - 尤其是 `wi_1 + wo`
3. 如果未来还要回头问 encoder role：
   - 只能作为 subordinate 的 `softer corridor intervention`
   - 而不是再把它当成 root-cause 主假说

一句话冻结：

- `encoder 16/17` 现在不能升级为 `necessary core`；它们最多是带有弱局部专属性的 `companion / corridor`，而当前更高置信的局部根因核心仍然留在 `decoder 5 / FFN`。
