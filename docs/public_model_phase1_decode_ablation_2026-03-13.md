# 公开模型第一阶段对照结论
## byt5 akkadian mbr decode ablation，2026-03-13

## 0. 范围

本结论承接：

- [public_model_fast_repro_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_fast_repro_plan_2026-03-13.md)

本阶段只回答一件事：

- 这个公开高分模型在本地验证上的优势，主要来自模型本体，还是来自 decode 设定

本阶段未做：

- 官方 hidden test 复现
- 训练复现
- MBR / rerank / submission glue 复现

## 1. 已完成的三组对照

本地工作区：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr`

验证集：

- local `fold0`
- `313` 行

统一评分口径：

- corpus BLEU
- corpus chrF++
- geometric mean

### 1.1 默认公开配置

- tag:
  - `default_beam8_rep11`
- decode:
  - `beam=8`
  - `length_penalty=1.0`
  - `repetition_penalty=1.1`
  - `max_new_tokens=640`

结果：

- `BLEU = 30.8114`
- `chrF++ = 49.6247`
- `geom = 39.1025`
- `pred_shorter_than_half_ref_ratio_pct = 10.8626`
- `unique_prediction_ratio_pct = 99.3610`
- `elapsed_seconds = 784.98`

### 1.2 用我们旧 winner 风格 decode

- tag:
  - `compare_beam4_lp07`
- decode:
  - `beam=4`
  - `length_penalty=0.7`
  - `repetition_penalty=1.1`
  - `max_new_tokens=384`

结果：

- `BLEU = 21.3210`
- `chrF++ = 41.8817`
- `geom = 29.8824`
- `pred_shorter_than_half_ref_ratio_pct = 24.9201`
- `unique_prediction_ratio_pct = 99.3610`
- `elapsed_seconds = 557.70`

### 1.3 去掉 repetition penalty

- tag:
  - `ablate_no_rep_beam8`
- decode:
  - `beam=8`
  - `length_penalty=1.0`
  - `repetition_penalty=1.0`
  - `max_new_tokens=640`

结果：

- `BLEU = 30.8130`
- `chrF++ = 49.6426`
- `geom = 39.1106`
- `pred_shorter_than_half_ref_ratio_pct = 11.1821`
- `unique_prediction_ratio_pct = 99.3610`
- `elapsed_seconds = 804.61`

## 2. 关键结论

### 2.1 真正重要的不是 `repetition_penalty=1.1`

从结果看：

- `default_beam8_rep11 geom = 39.1025`
- `ablate_no_rep_beam8 geom = 39.1106`

二者几乎相同。

这说明：

- 对这个公开模型而言，`repetition_penalty=1.1` 不是主要增益来源
- 至少在这套 local `fold0` 验证上，它不是决定性因素

### 2.2 真正重要的是 decode 框架本身

把 decode 改成我们旧 winner 风格之后：

- `geom: 39.1025 -> 29.8824`
- `delta = -9.2201`

同时：

- `pred_shorter_than_half_ref_ratio_pct: 10.8626 -> 24.9201`

这说明：

- 这个公开模型的高分表现高度依赖：
  - 更宽的 beam
  - 更自然的长度释放
  - 更高的生成上限
- 我们旧主线常用的：
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`
  - 会显著压坏它的本地表现

### 2.3 主干强度仍然是基础

虽然 decode 很关键，但不能误读成：

- 只要把 decode 改成 `beam=8` 就能达到公开模型水平

更准确的结论是：

- 强单模是基础
- 宽 beam / 更长生成上限是放大器
- 两者叠加才构成当前高分与健康表现

## 3. 对我们当前主线的直接含义

### 3.1 我们旧 decode 习惯本身就在压分

这次对照已经说明：

- 即使是公开强模型
- 套上我们近期常用的 `beam4/lp0.7/max384`
- 分数也会被明显压低

所以当前不能再继续默认：

- `beam=4 / lp=0.7 / max384`
- 是普适主配置

### 3.2 当前最优路径不是“只换更大模型”

第一阶段结果说明，快速复现公开高分路径至少需要同时复现：

1. 更强 ByT5 主干
2. 更合理的 decode 释放

不能只做其中一半。

### 3.3 我们近期 repair/chooser 路线的优先级应继续下降

这次对照更支持以下判断：

- 当前最大的增益不在：
  - chooser
  - fallback 阈值
  - replay rescue
  - term patch
- 而在：
  - 更强主干
  - 更正确的 decode 策略

## 4. 第一阶段后的执行建议

### 4.1 立即固定公开路线 decode baseline

从这版开始，公开复现实验的默认 baseline 应改成：

- `beam=8`
- `length_penalty=1.0`
- `max_new_tokens=640`

`repetition_penalty` 作为可选项保留，不再当成关键主因。

### 4.2 新训练主线必须同时满足两条

1. 更大 ByT5 主干
2. 新 decode baseline 验证

如果只满足其中一条：

- 仍不能算作在复现公开高分路线

### 4.3 旧 decode baseline 应降级为对照项

以下配置不再应被当作默认 promote decode：

- `beam=4`
- `length_penalty=0.7`
- `max_new_tokens=384`

它接下来只适合保留为：

- historical compare baseline

## 5. 最短结论

第一阶段已经给出足够明确的结果：

1. 公开模型的高分不主要来自 `repetition_penalty=1.1`
2. 公开模型的高分高度依赖：
   - 更强主干
   - 更宽 beam
   - 更长生成上限
3. 我们旧 winner 风格 decode 会显著压坏它的得分与健康
4. 所以快速复现路径必须写成：
   - 强单模 first
   - 新 decode baseline second
   - rerank / MBR later
