# Public Model H6 Decoder5 Full-Row Verification
## full-row necessity / sufficiency / interaction check, 2026-03-15

本稿基于以下产物：

- [mechanism_eval_results.json](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_eval_results.json)
- [mechanism_eval_table.csv](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_eval_table.csv)
- [sample_slice_summary.json](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/sample_slice_summary.json)
- [sample_slice_rows.csv](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/sample_slice_rows.csv)
- [mechanism_summary.md](/workspace/deep-past-/reports/public_model_h6_decoder5_full_20260315/mechanism_summary.md)

本稿只回答一个问题：

- 在 `public model` 的当前局部根因里，`decoder block 5 / FFN` 尤其是 `wi_1 + wo` 能否通过 full-row 检查，被升级成更高置信的局部核心；`cross.o` 与 `encoder 16/17` 又该怎么降格或保留。

## 0. 结论先行

这轮 full-row 之后，可以更硬地冻结下面四条：

1. `decoder 5 / wi_1` 与 `decoder 5 / wo` 都有明确必要性。
2. 其中更高置信的局部必要核心，不是 `encoder 16/17`，而是 `decoder 5 / FFN` 的 `wi_1 + wo`。
3. `cross.o` 不是独立主 computation branch，而是带明显条件性的 export / readout companion：
   - 单独保留它时只能保住有限份额；
   - 与 `wi_1` 或 `wo` 的单边配对都不强；
   - 但加在 `wi_1 + wo` 上时，确实还能继续增益。
4. 这还不足以宣告“完整机制已闭环”：
   - 最强 compact proxy 仍然离 `r24_ref` 有明显差距；
   - 在 `left_beats_all` winning rows 上，也没有任何 keep-only 子图能把 `r24` 的主优势保回来。

一句话冻结：

- 当前最高置信的局部根因核心已经可以压到 `decoder 5 / FFN wi_1 + wo`；`cross.o` 是条件性 readout/export port；`encoder 16/17` 继续保留为非核心 corridor companion，而不是必要核心。

## 1. `M1` 必要性：`wi_1` 与 `wo` 都成立，联合更强

full-row `M1` 结果：

1. `m1_ablate_d5_wi1 = 40.3317`
   - 相对 `r24_ref = 40.5412` 为 `-0.2095`
2. `m1_ablate_d5_wo = 40.3587`
   - 相对 `r24_ref` 为 `-0.1825`
3. `m1_ablate_d5_wi1_wo = 40.1334`
   - 相对 `r24_ref` 为 `-0.4077`

这组结果足够把必要性口径收紧成：

1. `wi_1` 与 `wo` 都不是可随意去掉的旁枝。
2. `wi_1` 单独去掉时伤害略大于 `wo`。
3. 联合去掉时伤害明显放大，说明当前 `R24` 的局部收益不是由其中任意一支单独承重。

在 `left_beats_all` 的 `64` 个 winning rows 上，这个判断更硬：

1. `r24_ref = 41.8744`
2. `m1_ablate_d5_wi1 = 40.8642`
3. `m1_ablate_d5_wo = 40.8744`
4. `m1_ablate_d5_wi1_wo = 40.2960`

也就是说：

- 在 `R24` 真正赢下 `r29/r30` 的样本上，`wi_1` 与 `wo` 的去除都会直接啃掉主优势，而联合去除的破坏更大。

## 2. `M2` 充分性：`decoder5` 是最佳 compact proxy，但仍明显不够

full-row `M2` 结果：

1. `m2_keep_d5_e16e17_all = 39.2931`
2. `m2_keep_d5_all = 39.2882`
3. `m2_keep_d4d5_e16e17_all = 39.1161`

对 `r24_ref` 的差值分别是：

1. `-1.2481`
2. `-1.2530`
3. `-1.4251`

正确读法是：

1. `decoder 5` 仍然是当前最佳 compact sufficiency proxy。
2. 把 `encoder 16/17` 硬塞回 keep-only，full-row 上几乎不带来可见增益。
3. 把 `decoder 4/5 + encoder 16/17` 一起保留，反而更差。

因此，这轮不支持：

- `encoder 16/17` 是把 `decoder5` 变成局部充分核心所必需的关键承重层。

但 `left_beats_all` 切片上有一个需要保留的弱信号：

1. `m2_keep_d5_all = 39.3968`
2. `m2_keep_d5_e16e17_all = 39.6240`

这说明：

- `encoder 16/17` 也许在真正 winning rows 上提供弱 companion 帮助；
- 但这种帮助既不稳定，也远不足以把它升级成已证实核心。

## 3. `M3` interaction：最佳 pair 是 `wi_1 + wo`，`cross.o` 的增益是条件性的

### 3.1 singles

1. `m3_keep_d5_crosso_only = 39.1298`
2. `m3_keep_d5_wo_only = 39.0354`
3. `m3_keep_d5_wi1_only = 39.0232`

这里最重要的不是 `cross.o` 单支略高，而是：

- 三个单支都离 `r24_ref` 很远；
- 没有任何单支能构成足够高保真的局部解释。

### 3.2 pairs

1. `m3_keep_d5_wi1_wo = 39.1906`
2. `m3_keep_d5_wo_crosso = 39.1275`
3. `m3_keep_d5_wi1_crosso = 38.8120`

这三枪给出的结构信息非常清楚：

1. 最佳 pair 是 `wi_1 + wo`。
2. `wo + crosso` 基本只回到 `crosso_only` 水平。
3. `wi_1 + crosso` 反而是全表最差 pair。

因此，`cross.o` 不是“给任何一个 FFN branch 配上去都能补全”的通用伙伴。

### 3.3 triple

`m3_keep_d5_wi1_wo_crosso = 39.4729`

这比：

1. 最佳 pair `m3_keep_d5_wi1_wo = 39.1906`
   - 高 `+0.2823`
2. `m2_keep_d5_all = 39.2882`
   - 高 `+0.1846`

这说明：

1. `wi_1 + wo + cross.o` 是当前 full-row 下最好的 compact local subgraph。
2. `cross.o` 在 `wi_1 + wo` 已经同时存在时，仍然会继续提供非零增益。
3. 但这种增益是强条件性的：
   - 它不会稳定地帮助单边 pair；
   - 它更像完整 FFN 双支已经就位后的 export/readout port。

所以这轮 `M3` 最准确的口径不是：

- `cross.o` 不重要

也不是：

- `cross.o` 才是真正主核

而是：

- `cross.o` 是条件性 companion；真正更像局部必要核心的仍是 `wi_1 + wo`。

## 4. winning-row slice：必要核心更硬，但最小充分机制仍未闭环

`left_beats_all` 切片共有 `64` 条样本，其 baseline 为：

1. `r24_ref = 41.8744`
2. `r29_ref = 40.0582`
3. `r30_ref = 40.0349`

也就是：

- `R24` 在这些真正赢样本上，保有大约 `1.82~1.84` 的 corpus geom 优势。

这一切片上的关键观察有两条：

1. `M1` 仍然直接击穿优势：
   - `m1_ablate_d5_wi1 = 40.8642`
   - `m1_ablate_d5_wo = 40.8744`
   - `m1_ablate_d5_wi1_wo = 40.2960`
2. 所有 keep-only 子图都保不回这批主优势样本：
   - `m2_keep_d5_e16e17_all = 39.6240`
   - `m2_keep_d5_all = 39.3968`
   - `m3_keep_d5_wi1_wo_crosso = 39.2398`

也就是说：

1. `decoder5` 局部回路对 winning rows 确实有必要性。
2. 但当前拿到的 compact subgraph 还不足以构成最小充分机制。
3. 因此最准确的状态仍然是：
   - `high-confidence local core` 已更硬
   - `complete mechanism` 仍未闭环

## 5. 这轮之后可以冻结什么

这轮 full-row 之后，可以把口径收紧成下面五条：

1. `decoder 5 / wi_1` 与 `decoder 5 / wo` 都是当前 `R24` 局部收益的必要 branch。
2. `decoder 5 / wi_1 + wo` 是当前最高置信的局部必要核心组合。
3. `cross.o` 不是主 computation origin，而是条件性 readout/export companion：
   - 单独保留不够；
   - 单边 pair 不稳；
   - 在 `wi_1 + wo` 同时存在时才更像有效补位项。
4. `encoder 16/17` 仍不能从此前的 `companion / corridor` 降格中翻盘。
5. 因此当前更准确的统一表述应是：
   - 高置信局部根因核心在 `decoder 5 / FFN wi_1 + wo`
   - `cross.o` 是条件性 companion
   - `encoder 16/17` 是非核心 corridor companion

## 6. 这轮还不能说明什么

当前仍然不能直接说明：

1. `decoder 5` 的完整最小机制已经被 compact subgraph 解释完。
2. `cross.o` 可以被简化成“完全不重要”。
3. `encoder corridor` 在所有更温和 intervention 下都无关。

因此，这轮最准确的收口不是：

- 机制已完成

而是：

- 当前最强的局部根因核心已显著收敛；
- 但完整机制仍需要更高保真、尤其是 winning-row 导向的补充检查。
