# Public Model H6 Module Pair Analysis
## decoder block 5 + encoder 16/17 pair audit, 2026-03-15

本稿承接：

- [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
- [public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)
- [module_pair_audit.json](/workspace/deep-past-/reports/public_model_h6_module_pair_audit_20260315/module_pair_audit.json)

本稿只做一件事：

1. 把 `decoder block 5` 与 `encoder 16/17` 的指定 `FFN <-> attention o/v` pair 固定下来，直接看：
   - `o` 是否稳定贴着热点 FFN 走
   - `v` 是否更多只是次级响应
   - `wi` 与 `wo` 哪一支更像高因果效率的主支路

本稿不新增训练，也不把 pair audit 误写成 `public model` 的完整形成机制。

## 0. 结论先行

当前最值得冻结的 pair-level 口径是：

- `decoder block 5` 里，`cross.o` 与 `self.o` 都稳定贴着热点 FFN 走；但更接近高收益支路的是 `wi_1 + o`，更接近 full-probe 局部形状的是 `wo + o`。
- `encoder 16/17` 里，`self.o` 在 `R24 / R29 / R30` 三条线上都 consistently 强于 `self.v`，说明 `o` 仍更像热点 FFN 的配套 readout 项，`v` 更像次级响应。
- `R24` 与尤其 `R30` 可以在局部上表现得更 `wo`-heavy，但性能排序仍是 `R24 > R29 > R30`；这进一步支持：
  - `wi` 更像主 computation source
  - `wo` 更像 readout / transport / consolidation
- `encoder 16` 与 `encoder 17` 对 `wi` 的偏好并不完全相同：
  - `encoder 16` 更像偏 `wi_0`
  - `encoder 17` 更像偏 `wi_1`

## 1. 方法与范围

本次 pair audit 不重读 checkpoint，只复用：

- [adapter_audit.json](/workspace/deep-past-/reports/public_model_h6_adapter_audit_20260315/adapter_audit.json)

审计对象固定为：

1. `R24 = q/k/v/o + wi_0/wi_1/wo`
2. `R29 = q/k/v/o + wi_0/wi_1`
3. `R30 = q/k/v/o + wo`

固定 pair 只看文档已经点名的那些：

1. `decoder block 5`
   - `wi_1 <-> cross.o`
   - `wi_1 <-> self.o`
   - `wo <-> cross.o`
2. `encoder block 16`
   - `wi_0 / wi_1 / wo` 与 `self.o / self.v`
3. `encoder block 17`
   - `wi_0 / wi_1 / wo` 与 `self.o / self.v`

这里的 pair 指标主要看：

1. pair 占 layer-local total 的比例
2. pair 两端分别占各自 bucket 的比例
3. `o` 相对 `v` 的稳定性
4. `wi`-anchored pair 与 `wo`-anchored pair 的局部形状差异

## 2. `decoder block 5` 在说明什么

这里最直接的事实是：

1. `wi_1 + o` 在高收益支路上非常稳定
   - `R24 / wi_1 <-> cross.o = 33.39% local`
   - `R24 / wi_1 <-> self.o = 32.13% local`
   - `R29 / wi_1 <-> cross.o = 51.36% local`
   - `R29 / wi_1 <-> self.o = 49.63% local`
2. `o` 在 attention bucket 里始终压过 `v`
   - `cross.o` 占 cross bucket：
     - `R24 = 58.59%`
     - `R29 = 55.99%`
     - `R30 = 53.75%`
   - `self.o` 占 self bucket：
     - `R24 = 76.09%`
     - `R29 = 77.52%`
     - `R30 = 76.33%`
3. `wo + cross.o` 更像 full-probe/readout 形状
   - `R24 / wo <-> cross.o = 43.40% local`
   - `R30 / wo <-> cross.o = 76.99% local`
   - 但 `R30` 仍低于 `R29`

这更像是在说：

- `o` 的确总是贴着热点 FFN 走
- 但真正更接近收益主效应的，不是单纯让 `wo + o` 吃掉更多局部能量
- 而是让 `wi_1` 这支 computation branch 带着 `o` 一起亮

## 3. `encoder 16/17` 在说明什么

### 3.1 `self.o` consistently 强于 `self.v`

`encoder 16`：

1. `R24`
   - `self.o` 占 self bucket `54.75%`
   - `self.v` 占 self bucket `45.25%`
2. `R29`
   - `self.o = 66.00%`
   - `self.v = 34.00%`
3. `R30`
   - `self.o = 63.32%`
   - `self.v = 36.68%`

`encoder 17`：

1. `R24`
   - `self.o = 60.14%`
   - `self.v = 39.86%`
2. `R29`
   - `self.o = 63.47%`
   - `self.v = 36.53%`
3. `R30`
   - `self.o = 57.29%`
   - `self.v = 42.71%`

这说明当前 pair-level 证据仍然支持：

- `o` 更像热点 FFN 的同层 readout companion
- `v` 并没有缺席，但更像次级响应

### 3.2 `encoder 16` 更像偏 `wi_0`，`encoder 17` 更像偏 `wi_1`

在最能暴露 `wi` 主效应的 `R29` 上：

1. `encoder 16`
   - `wi_0 <-> self.o = 58.51% local`
   - `wi_1 <-> self.o = 52.05% local`
2. `encoder 17`
   - `wi_0 <-> self.o = 42.30% local`
   - `wi_1 <-> self.o = 64.44% local`

这说明上部 encoder 热层并不是“所有层都同一种 FFN 子分支结构”，而是：

- `encoder 16` 更像 `wi_0` 偏重
- `encoder 17` 更像 `wi_1` 偏重

### 3.3 `wo` 可以吃掉更多局部质量，但不等于更高收益

仍看 `encoder 16/17`：

1. `R24`
   - `wo <-> self.o`
     - `encoder 16 = 51.78% local`
     - `encoder 17 = 49.74% local`
2. `R30`
   - `wo <-> self.o`
     - `encoder 16 = 90.53% local`
     - `encoder 17 = 88.19% local`
3. 但性能排序仍是：
   - `R24 = 40.5412`
   - `R29 = 40.4669`
   - `R30 = 40.4032`

这条反差仍然最重要：

- `wo` 可以把局部 pair 形状做得非常满
- 但 `wo`-heavy 并不自动转成更高收益
- 因此 pair-level 证据继续指向：
  - `wi` 更像高因果效率的 computation source
  - `wo` 更像 readout-heavy branch

## 4. 当前最严格的统一口径

把 root-cause audit、local circuit audit 与这次 pair audit 合在一起，当前最严格的表述应是：

- `public model` 当前最像是某种更强的 `training-shape / adaptation history`，在 `decoder block 5` 与 `encoder 16/17` 这些热点层上，把一个以 `wi` 为主 computation source、`wo` 为 readout source、attention `o` 为稳定配套项、`v` 为次级响应的稀疏 FFN-anchored circuit 反复打亮。

这里仍然不能写成：

1. 我们已经解释了 `public model` 的完整形成机制
2. `wo` 完全不重要
3. 只要复制这些 pair 就能完整复制 `public model`

## 5. 哪些 pair 最值得优先进入 interaction 问题

如果把这次 pair audit 进一步当成“候选筛选器”，最有价值的不是看谁绝对值最大，而是看：

1. 哪些 pair 在 `R29` 明显强、但在 `R30` 明显弱
2. 哪些 pair 与更高收益支路一起抬升
3. 哪些 pair 是 `o` 优先，而不是 `v` 优先

按这个标准，当前最值得优先冻结的候选顺序是：

1. `encoder 17 / wi_1 + self.o`
   - `R29 = 64.44% local`
   - `R30 = 15.84% local`
   - `R29 - R30 = +48.60 pts`
2. `encoder 16 / wi_0 + self.o`
   - `R29 = 58.51% local`
   - `R30 = 16.35% local`
   - `R29 - R30 = +42.16 pts`
3. `decoder 5 / wi_1 + cross.o`
   - `R29 = 51.36% local`
   - `R30 = 11.20% local`
   - `R29 - R30 = +40.15 pts`
4. `decoder 5 / wi_1 + self.o`
   - `R29 = 49.63% local`
   - `R30 = 10.20% local`
   - `R29 - R30 = +39.42 pts`

这组排序的含义是：

- 如果下一枪真要问 interaction，最该问的是 `wi`-anchored pair
- 而不是 `wo`-anchored pair
- 且 `o` 优先级高于 `v`

相反，最不该误判成主效应候选的是：

1. `encoder 17 / wo + self.o`
2. `encoder 16 / wo + self.o`
3. `decoder 5 / wo + cross.o`

因为这些 pair 在 `R30` 中都异常大，但 `R30` 的总收益并没有超过 `R29`。

## 6. 对下一步的直接含义

如果继续做分析，而不是立刻训练，当前最合理的动作是：

1. 把这份 pair audit 作为固定输入，与现有 `R24 / R29 / R30` 冻结文档一起使用
2. 若必须设计新 probe，只能问新的 interaction question，例如：
   - `decoder block 5 / wi_1 + cross.o`
   - `encoder 16 / wi_0 + self.o`
   - `encoder 17 / wi_1 + self.o`
3. 不应因为 `wo` pair 很亮，就回到：
   - 新的 long confirm
   - 更大 `rank/alpha`
   - 把 `H3/H4` 重新混回 `H6`

一句话说：

- pair-level 结果进一步说明：`o` 的确贴着热点 FFN 走，但更像决定收益的是 `wi` 支路，`wo` 更像把这些热点更新读出并放大的 readout 支路。
