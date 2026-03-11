# 架构升级 Probe 轮报告（2026-03-09）

## 1. 目标

本轮用于执行 [next-step_discipline.md](/workspace/deep-past-/docs/next-step_discipline.md) 中定义的全部 `probe` 任务，验证：

- `ByT5-base len640 q/v`
- `ByT5-base len640 qkvo_r16`
- `mT5-base len640 q/v`

是否能在 `probe` 阶段给出足够强的结构性正信号。

## 2. 对照线

matched baseline 使用当前正式 winner：

- `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- decode 口径：
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`
  - `max_val_samples=32`

## 3. Probe 任务范围

### 3.1 主线一：`ByT5-base len640 q/v`

- `P1_1`
- `P1_2`
- `P1_3`
- `P1_4`
- `P1_5`
- `P1_6`

### 3.2 主线二：`ByT5-base len640 qkvo_r16`

- `P2_1`
- `P2_2`
- `P2_3`
- `P2_4`
- `P2_5`
- `P2_6`
- `P2_7`

### 3.3 主线三：`mT5-base len640 q/v`

- `P3_1`
- `P3_2`
- `P3_3`
- `P3_4`
- `P3_5`
- `P3_6`

## 4. 智能分组原则

本轮采用“重训练串行、轻 decode 并行”的保守编排：

- Group A：
  - 先跑 `matched baseline`
- Group B：
  - 所有 smoke 串行
- Group C：
  - 所有通过 smoke 的 `250-step` probe 训练串行
- Group D：
  - anchor decode 按显存上限分批并行
- Group E：
  - 每条主线只对 line winner 跑 `diag32`

这样做的原因：

- `ByT5-base` 与 `mT5-base` 的训练峰值显存未知
- `qkvo_r16` 是最高风险线
- 与其三条大训练同时赌显存，不如保证每条 probe 都跑完整

## 5. 当前状态

本轮 `probe` 已全部完成。

- `Group A` 已完成：
  - `matched baseline anchor32`
- `Group B` 已完成：
  - `P1_1`
  - `P1_2`
  - `P2_1`
  - `P2_2`
  - `P3_1`
  - `P3_2`
- `Group C` 已完成：
  - `P1_3`
  - `P2_3`
  - `P3_3`
- `Group D` 已完成：
  - 全部 anchor decode
- `Group E` 已完成：
  - 每条主线的 line winner `diag32`

当前判断：

- 本轮不是下载卡住。
- `mT5-base` 的环境依赖已补齐并完成补跑。
- 三条架构升级主线都在 `probe` 阶段命中硬停边界。

## 6. 结果回写

<!-- ARCH_PROBE_RESULTS_START -->
matched baseline:
- 18.3354 / 11.1352 / 30.1913

### 主线一：ByT5-base len640 q/v
- 通过 smoke: `P1_1, P1_2`
- `ckpt100`: `3.5886 / 1.4989 / 8.5920`
- line winner: `ckpt100` -> `3.5886 / 1.4989 / 8.5920`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界

### 主线二：ByT5-base len640 qkvo_r16
- 通过 smoke: `P2_1, P2_2`
- `ckpt100`: `3.7618 / 1.7423 / 8.1222`
- `ckpt200`: `3.1917 / 1.2308 / 8.2763`
- line winner: `ckpt100` -> `3.7618 / 1.7423 / 8.1222`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界

### 主线三：mT5-base len640 q/v
- 通过 smoke: `P3_1, P3_2`
- `ckpt100`: `0.1437 / 0.0163 / 1.2681`
- line winner: `ckpt100` -> `0.1437 / 0.0163 / 1.2681`
- diag32 reconstructed: `0.0000 / 0.0000 / 0.0000`
- diag32 health: `empty=0.00% copy=0.00% short=0.00%`
- `stage_decision = reject`
- reason: 命中 probe 硬停边界
<!-- ARCH_PROBE_RESULTS_END -->
