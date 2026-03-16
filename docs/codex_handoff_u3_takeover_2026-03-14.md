# Codex Handoff
## Track B H6 root-cause bridge, updated 2026-03-15

这份手册现在不再是 `U3 takeover checklist`。

它的用途改为：

1. 让新的 Codex 在上下文不足时，快速接上当前 `Track B / public model` 的真实阶段
2. 明确哪些结论已经冻结，哪些地方还不能过度解释
3. 给出下一步最合理的分析入口，避免重新走回“继续盲开训练”的旧路

## 0. 当前状态

当前阶段已经不是 `U3`。

目前 `Track B` 已完成：

1. `U1~U3`
2. `H4` probes
3. `H6` 主线 `R24 / R26 / R27 / R29 / R30`
4. checkpoint-level adapter audit
5. layer-local circuit audit

当前没有活跃训练：

1. `R30` 已完成，见 [driver_status.json](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_status.json)
2. GPU 当前空闲
3. tmux 里仍保留一些历史 session，但不是 live training source of truth

新的 Codex 接手时，先默认：

- 当前工作重点是分析与冻结，不是立刻继续挂训练

## 1. 先守住的工作约束

1. 术语固定用 `public model` 与 `public-weight continuation`
2. 不要把 `public model` 叫成 `base model`
3. 不要把当前结论写成“已经解释了 public model 的形成机制”
4. 不要做 `git` 同步、提交或回滚用户已有改动
5. 不要默认继续开 long confirm
6. 不要默认继续做更大 `rank/alpha`
7. 不要回头把 `H3/H4` 重新混回 `H6`

## 2. 先看的主入口

新的 Codex 应按这个顺序看：

1. 主计划总表：
   - [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
2. `R29` 冻结：
   - [public_model_h6_r29_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r29_freeze_and_next_step_2026-03-15.md)
3. `R30` 冻结：
   - [public_model_h6_r30_freeze_and_next_step_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_r30_freeze_and_next_step_2026-03-15.md)
4. 根因层 adapter audit：
   - [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
5. 局部电路分析：
   - [public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)

如果只够读一份，先读：

- [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)

## 3. 当前已冻结事实

比较锚点：

1. public anchor = `39.1025`
2. plain continuation pilot = `40.2266`
3. incumbent long = `40.4028`

`H3 / H4`：

1. `R18 = 39.4724 / negative`
2. `R19 = 40.1336 / inconclusive`
3. `R20 = 39.5334 / negative`
4. `R21/R22/R23 = 40.2266 / inconclusive`

`H6`：

1. `R24 = 40.5412 / positive / current best pilot`
2. `R26 = 40.4099 / guarded long inconclusive`
3. `R27 = 40.3165 / higher-capacity inconclusive`
4. `R29 = 40.4669 / inproj split positive`
5. `R30 = 40.4032 / outproj split inconclusive`

当前 `H6` 排序已冻结为：

- `R24 > R29 > R30 > R27 > baseline`

## 4. 当前最重要的结论

现在最重要的不是单次实验分数，而是下面四条：

1. 当前最像 `public model` 上游强度来源的是 `H6 / training-shape`
2. `wi_0/wi_1` 比 `wo` 更像主效应
3. `wo` 有贡献，但更像补充 readout / consolidation 支路
4. 当前最像根因的不是 `more data`，而是 `specific FFN circuit shaping`

更严格的口径是：

- `public model` 当前最像是某种更强的 `training-shape / adaptation history` 在一个稀疏 FFN 子回路上留下的痕迹

## 5. 根因层分析结论

adapter audit 已经把解释推进到比“module split 分数”更深一层。

关键事实，见 [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)：

1. `R24 / R29 / R30` 的主信号都以 `FFN` 为主，不是 attention 主导
   - `R24` FFN share of total adapter energy = `0.7840`
   - `R29` FFN share of total adapter energy = `0.6512`
   - `R30` FFN share of total adapter energy = `0.6367`
2. 热点层高度集中在：
   - `decoder block 5`
   - `decoder block 4`
   - `encoder block 13~17`
3. `R24` 的 FFN 内部：
   - `wi_0 + wi_1 = 55.2%`
   - `wo = 44.8%`
4. `R24 vs R29 / shared_ffn_wi` 高度同形
   - cosine = `0.9897`
   - pearson = `0.9857`
5. `R24 vs R30 / shared_ffn_wo` 也高度同形
   - cosine = `0.9977`
   - pearson = `0.9976`

这意味着：

1. `R29` 和 `R30` 命中的不是不同热点层
2. 它们命中的本质上是同一组热点层
3. 关键差别在于：
   - `wi` 更像把有用 computation 做出来
   - `wo` 更像把已有 computation 读出和传出去

## 6. 局部电路层分析结论

layer-local circuit audit 已进一步把问题压缩到 `decoder block 5 + encoder 13~17`。

关键事实，见 [public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)：

1. 这六层合计占 total adapter energy：
   - `R24 = 61.08%`
   - `R29 = 61.38%`
   - `R30 = 58.00%`
2. `decoder block 5` 本身就是最热单层：
   - `R24 = 32.74%`
   - `R29 = 30.31%`
   - `R30 = 30.20%`
3. 这些层里 FFN 始终主导，但 attention `o/v` 与 FFN 总量高度同步
   - local FFN share: `R24 = 84.14% / R29 = 70.65% / R30 = 72.29%`
   - layerwise Pearson of `FFN total` vs `attention o/v total`:
     - `R24 = 0.9989`
     - `R29 = 0.9907`
     - `R30 = 0.9972`
4. attention 侧 `o` consistently 强于 `v`
   - self-attn `o / v`:
     - `R24 = 1.9138`
     - `R29 = 2.1959`
     - `R30 = 2.0480`
5. `R24` 的局部 profile 其实更像 `R30`
   - `R24 vs R29` 六层平均：cosine `0.6287` / pearson `0.2552`
   - `R24 vs R30` 六层平均：cosine `0.7991` / pearson `0.7253`
6. 但 `R24` 的性能更像 `R29`

这条反直觉事实目前最有价值：

- `wo` 更像吸收大量局部更新、让 profile 看起来更像 full probe 的 readout 支路
- `wi` 更像真正决定收益的高因果效率 computation 支路

当前最准确的局部电路口径是：

- 一个以 `wi` 为主 computation source、`wo` 为 readout source、attention `o/v` 为局部配套项的稀疏 FFN-anchored circuit，在 `decoder block 5` 与 `encoder 13~17` 被反复打亮

## 7. 当前不该做什么

新的 Codex 接手时，不建议默认做这些事：

1. 再开新的 long confirm
2. 继续做更大 `rank/alpha`
3. 因为 `R30` 接近 incumbent long，就误判为值得 promote
4. 把 `H3/H4` 重新混回 `H6`
5. 把当前证据写成“已经理解 public model 完整机制”

## 8. 当前最合理的下一步

如果继续分析，而不是立刻训练，下一步最合理的是：

1. 对 `decoder block 5` 做 module-pair 审计
   - `wi_1 <-> cross.o`
   - `wi_1 <-> self.o`
   - `wo <-> cross.o`
2. 对 `encoder 16/17` 做同样审计
   - `wi_0 / wi_1 / wo` 与 `self.o / self.v`
3. 核心问题不是“谁更大”，而是：
   - `o` 是否总是贴着热点 FFN 走
   - `v` 是否只是次级响应
   - `wi` 与 `wo` 到底是串联关系，还是共同响应更上游 shaping factor
4. 进入下一轮分析前，新的 Codex 不需要重跑前置训练
   - 直接以 `R24 / R29 / R30 + adapter_audit + local_circuit_audit` 为固定输入即可

如果要继续训练，前提应是：

1. 先明确新的训练要回答哪一个 interaction question
2. 而不是“继续挂一个新 pilot 看看”

## 9. 报告与代码入口

关键报告：

1. `R24`：
   - [driver_results.json](/workspace/deep-past-/reports/public_model_r24_public_h6proxy_ffn_pilot_20260314/driver_results.json)
2. `R29`：
   - [driver_results.json](/workspace/deep-past-/reports/public_model_r29_public_h6proxy_ffn_inproj_pilot_20260315/driver_results.json)
3. `R30`：
   - [driver_results.json](/workspace/deep-past-/reports/public_model_r30_public_h6proxy_ffn_outproj_pilot_20260315/driver_results.json)
4. adapter audit：
   - [adapter_audit.json](/workspace/deep-past-/reports/public_model_h6_adapter_audit_20260315/adapter_audit.json)
5. local circuit audit：
   - [local_circuit_audit.json](/workspace/deep-past-/reports/public_model_h6_local_circuit_audit_20260315/local_circuit_audit.json)

关键脚本：

1. 通用 pilot driver：
   - [public_model_publicprobe_driver.py](/workspace/deep-past-/scripts/public_model_publicprobe_driver.py)
2. adapter audit：
   - [public_model_h6_adapter_audit.py](/workspace/deep-past-/scripts/public_model_h6_adapter_audit.py)
3. local circuit audit：
   - [public_model_h6_local_circuit_audit.py](/workspace/deep-past-/scripts/public_model_h6_local_circuit_audit.py)

关键配置：

1. `R24`：
   - [public_model_r24_public_h6proxy_ffn_c0_pilot_20260314.yaml](/workspace/deep-past-/configs/public_model_r24_public_h6proxy_ffn_c0_pilot_20260314.yaml)
2. `R29`：
   - [public_model_r29_public_h6proxy_ffn_inproj_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r29_public_h6proxy_ffn_inproj_c0_pilot_20260315.yaml)
3. `R30`：
   - [public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml](/workspace/deep-past-/configs/public_model_r30_public_h6proxy_ffn_outproj_c0_pilot_20260315.yaml)

## 10. 新 Codex 接手时的最小动作

1. 先看 [public_model_upstream_reverse_engineering_train_plan_2026-03-14.md](/workspace/deep-past-/docs/public_model_upstream_reverse_engineering_train_plan_2026-03-14.md)
2. 再看 [public_model_h6_root_cause_adapter_audit_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_root_cause_adapter_audit_2026-03-15.md)
3. 再看 [public_model_h6_local_circuit_analysis_2026-03-15.md](/workspace/deep-past-/docs/public_model_h6_local_circuit_analysis_2026-03-15.md)
4. 再决定：
   - 是继续分析 module-pair
   - 还是设计一个真正有 interaction 假设的新 `H6` probe

一句话总结：

- 当前阶段已经从“试哪个方向”推进到“热点 FFN 子回路怎么被训练形态塑形”；新的 Codex 不该回到 `U3` 时代的任务定义。
