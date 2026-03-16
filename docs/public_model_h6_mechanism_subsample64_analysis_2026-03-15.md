# Public Model H6 Mechanism Subsample64 Analysis
## first execution readout for `2026-03-15-3`

本稿基于以下产物：

- [mechanism_eval_results.json](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/mechanism_eval_results.json)
- [sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/sample_slice_summary.json)
- [mechanism_summary.json](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/mechanism_summary.json)
- [mechanism_summary.md](/workspace/deep-past-/reports/public_model_h6_mechanism_subsample64_20260315/mechanism_summary.md)

本稿目标不是宣称“完整机制已经证实”，而是把 `2026-03-15-3` 第一轮系统执行的可冻结结论写清楚。

## 0. 范围与口径

这轮结果必须带着下面三个边界阅读：

1. 这是 `fold0 / max_rows=64` 的机制筛查子样本，不是 full-row 终判。
2. 这个子样本上的 reference 排序是：
   - `r29_ref = 37.1995`
   - `r30_ref = 37.0050`
   - `r24_ref = 36.9186`
   因而不能把这里的绝对排序当成 full-fold 证据。
3. 本轮应只使用：
   - 必要性掉分模式
   - keep-only 保留模式
   - interaction 增益模式
   - sample-slice 上的相对保真模式

另外，本轮中途修正了两个工具口径问题：

1. `mechanism_eval.py` 最初把 `delta_vs_r24/r29/r30` 对到了 full pilot reference
   - 现已修正为优先使用本轮实际跑出的 `r24_ref/r29_ref/r30_ref`
2. `sample_slice.py` 最初没有把 `project_name` 的预测列 merge 进总表
   - 现已修正并完成重跑

因此，本稿采用的数值均已按本轮子样本 reference 重算。

## 1. `M1` 必要性

`M1` 的核心结果是：

- `m1_ablate_d5_wi1`: `36.5843`，相对 `r24_ref` 为 `-0.3342`
- `m1_ablate_d5_wo`: `36.7550`，相对 `r24_ref` 为 `-0.1636`
- `m1_ablate_d5_wi1_wo`: `36.6259`，相对 `r24_ref` 为 `-0.2927`
- `m1_ablate_e16_wi0`: `37.0053`，相对 `r24_ref` 为 `+0.0867`
- `m1_ablate_e17_wi1`: `36.9732`，相对 `r24_ref` 为 `+0.0546`
- `m1_ablate_e16e17_wi`: `36.9792`，相对 `r24_ref` 为 `+0.0606`

可冻结解释：

1. `decoder 5 / FFN` 在这个子样本上表现出局部必要性。
2. `wi_1` 的必要性强于 `wo`。
3. `encoder 16/17` 在这个子样本上没有表现出单点必要性，也没有表现出联合必要性。

因此，这一轮支持：

- `wi` 更像主 computation branch
- `wo` 更像 readout / completion branch

但这一轮不支持直接写成：

- `encoder 16/17` 已经被证明是必要核心

## 2. `M2` 充分性

`M2` 的核心结果是：

- `m2_keep_d5_all`: `36.2074`，相对 `r24_ref` 为 `-0.7111`
- `m2_keep_d5_e16e17_all`: `35.8820`，相对 `r24_ref` 为 `-1.0365`
- `m2_keep_d4d5_e16e17_all`: `35.8914`，相对 `r24_ref` 为 `-1.0272`

对应的 surgery 能量占比分别约为：

- `decoder 5 only`: `0.3439`
- `decoder 5 + encoder 16/17`: `0.4970`
- `decoder 4/5 + encoder 16/17`: `0.5982`

可冻结解释：

1. 在 hard keep-only 手术下，`decoder 5` 是最紧凑、最能保分的局部载体。
2. 单纯把 `encoder 16/17` 或 `decoder 4/5 corridor` 一起保留下来，并没有自动补全机制。
3. 这更像说明：
   - `decoder 5` 具有最高因果密度
   - `encoder/corridor` 可能需要更完整上下文才能发挥作用

因此，这一轮支持：

- 最小高密度 sufficiency proxy 更像 `decoder 5`

但这一轮不支持直接写成：

- `decoder 4/5 + encoder 16/17` 已经作为 keep-only 充分回路成立

## 3. `M3` interaction

`M3` 的核心结果是：

- `m3_keep_d5_wi1_only`: `35.4211`
- `m3_keep_d5_wo_only`: `35.4029`
- `m3_keep_d5_wi1_wo`: `35.9956`
- `m3_keep_d5_wi1_wo_crosso`: `36.1055`

对应 interaction 增益为：

- `pair_gain_over_best_single = +0.5745`
- `triple_gain_over_best_pair = +0.1099`

可冻结解释：

1. `wi_1 only` 与 `wo only` 单支保分能力接近。
2. `wi_1 + wo` 比任一单支明显更强。
3. `+ cross.o` 还能再抬一点，但抬升远小于 `wi_1 + wo` 组合本身。

因此，这一轮最支持的 interaction 口径是：

- 主效应落在 `decoder 5 / FFN wi_1 + wo` 的组合
- `cross.o` 更像 companion / compensation，而不是 primary origin

这和此前“attention 是配套项”的口径一致，而且因果强度更高。

## 4. `M4` encoder 非对称

`M4` 的核心结果是：

- `m4_keep_e16_wi0_selfo`: `35.5167`
- `m4_keep_e17_wi1_selfo`: `35.1997`
- `m4_keep_e16e17_pair_union`: `35.1339`

可冻结解释：

1. 如果强行压成 tiny encoder-side proxy，`encoder 16` pair 强于 `encoder 17` pair。
2. 两个 pair 的 union 没有带来加性恢复，反而更低。

因此，这一轮支持：

- `encoder 16` 比 `encoder 17` 更像高价值 companion pair

但这一轮不支持：

- `encoder 16 + 17` 作为独立、可加性的 mini-circuit

## 5. `M5` sample-level slice

`left_beats_all` 切片结果为：

- 样本数：`15`
- `r24_ref` 相对 `r29/r30` 的平均优势边际：`+1.0893`

在这 `15` 个 `r24` 真正赢下来的样本上：

- `r24_ref`: `34.7162`
- `r29_ref`: `33.3202`
- `r30_ref`: `33.5943`
- `m1_ablate_d5_wi1`: `33.2461`
- `m1_ablate_d5_wo`: `33.5374`
- `m1_ablate_e16_wi0`: `34.4471`
- `m1_ablate_e17_wi1`: `34.3441`
- `m1_ablate_e16e17_wi`: `34.6321`
- `m2_keep_d5_all`: `34.2327`

可冻结解释：

1. `R24` 真正比 `R29/R30` 赢得更多的样本，和 `decoder 5` 绑定更紧。
2. 去掉 `decoder 5 / wi_1` 或 `wo`，这些 winning rows 的优势会明显缩小。
3. `encoder` 侧消融对这些 rows 的损害远小于 `decoder 5` 消融。
4. `keep decoder 5 only` 在这些 rows 上的保真度明显高于 `R29/R30`。

这说明：

- 第一轮样本级证据也把主要因果密度压回了 `decoder 5`

## 6. 这一轮后可以冻结的“高置信局部机制结论”

在保守口径下，本轮之后最值得冻结的新结论是：

- 在 `2026-03-15-3` 第一轮系统机制检查里，`public model` 当前最稳定、最有因果密度的局部载体，已经可以进一步压缩到 `decoder block 5 / FFN`，尤其是 `wi_1 + wo` 的组合；`wi_1` 更像主 computation branch，`wo` 更像 readout / completion branch，`cross.o` 只表现出次级补位收益；`encoder 16/17` 在当前子样本上更像 companion / corridor structure，而不是已经被证明的必要核心。

这比此前“稀疏 FFN-anchored local circuit”更进一步，但仍然是：

- 高置信局部机制结论

而不是：

- 完整机制已经通过

## 7. 这一轮仍然不能断言什么

当前仍然不能断言：

1. `decoder 5 + encoder 16/17` 已经被完整证明为最小闭环
2. `encoder 16/17` 已被证明是必要核心
3. `keep-only` 里 corridor 表现更差，等价于“encoder 在真实机制里有害”
4. 这个 `64-row` 子样本上的 `r29 > r24` 绝对排序，能推翻 full-fold 上的总体判断
5. 完整机制检查已经通过

因此，最准确的阶段性 verdict 应是：

- `decoder 5 / FFN wi_1 + wo` 已经拿到高置信局部机制支持
- `encoder 16/17` 仍保留为 companion / corridor 假说
- 完整机制仍需更高保真验证

## 8. 后续最高价值动作

如果继续往根因方向推进，而不是回头做泛化搜索，最高价值动作应是：

1. 用 full-row 只重跑最关键的少数变体：
   - `r24_ref`
   - `r29_ref`
   - `r30_ref`
   - `m1_ablate_d5_wi1`
   - `m1_ablate_d5_wo`
   - `m2_keep_d5_all`
   - `m3_keep_d5_wi1_only`
   - `m3_keep_d5_wo_only`
   - `m3_keep_d5_wi1_wo`
   - `m3_keep_d5_wi1_wo_crosso`
2. 对 `left_beats_all` 的 `15` 个 winning rows 做更细粒度 projection / case audit
   - 看 `decoder 5` 手术究竟改坏了哪些局部翻译决策
3. 如果还要验证 encoder role
   - 应优先设计比 hard keep-only 更温和的 corridor intervention
   - 而不是继续堆更多静态热点分析

一句话说：

- 第一轮完整机制检查已经把“主根因落在 `decoder 5 / FFN wi_1 + wo`”这件事显著加硬了，但还没有把 `encoder 16/17` 从 companion 假说推进成必要核心，也还不足以宣称完整机制闭环已经证实。
