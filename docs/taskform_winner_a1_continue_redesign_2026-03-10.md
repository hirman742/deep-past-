# A1 Continue-On-Wlite 重设计（2026-03-10）

## 1. 为什么旧 A1 结果无效

`reports/taskform_winner_a1_smoke_20260310/summary.json` 这一轮不能再用于主线判断，原因只有一条但足够致命：

- 它没有接当前 `A2 retrieval-top1 W-lite` 的训练口径

具体表现：

- 用的是 plain processed dir，而不是 retrieval recipe
- 从 fresh `ByT5-small + LoRA` 起训
- 只跑了 `220 steps`
- control 自己就只有 `anchor64 geom = 3.0869`

所以它回答的问题其实是：

- “在一个很弱的 fresh-base smoke 里，external silver 有没有一点局部增益？”

而不是：

- “external silver 能不能继续提升当前已经跑到 `~20 geom` 的 retrieval winner？”

因此这轮只能保留一个很弱的先验：

- 高比例 `E30 / E50` 不值得再先开
- 若要重试，比例应该降到 `E5 / E10 / E15`

## 2. 正确的 probe 口径

新的 `A1` 只认 `continue-on-wlite`：

- upstream adapter:
  - `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model`
- compare reference:
  - `reports/taskform_winner_a2_freeze_20260310/summary.json`
  - 也就是当前 frozen promote candidate `fallback_180`

### 2.1 P1：build

先构 plain mixed rows，再统一重建 retrieval hint：

1. plain base:
   - `data/processed_byt5_chunks_align_gc_cost14`
2. ratios:
   - `C0 / E5 / E10 / E15`
3. retrieval rebuild:
   - `scripts/taskform_build_retrieval_hint_processed.py`
4. continue configs:
   - 克隆 `taskform_winner_a2_retrieval_top1_wlite.yaml`
   - 写入新的 `processed_dir`
   - 写入 `tapt.init_adapter_dir = raw W-lite best_model`

### 2.2 P2：probe

每个候选都跑：

- `180 steps`
- `eval_steps = 45`
- decode 固定：
  - `beam=4 / lp=0.7 / max_new_tokens=384`
- 只看：
  - `anchor64 reconstructed`
  - `hard subset`
  - `health delta vs C0`

放行门槛：

- `Ebest - C0 anchor64 >= +0.25`
- `hard subset >= 0`
- `short / empty / repeat / unique` 不恶化

## 3. W-lite 长实验

只有 `A1R_P2` 过线才继续。

长实验不是单跑候选，而是 matched compare：

- `C0_W = internal-only continue @ 400 steps`
- `Ebest_W = best external continue @ 400 steps`

若 `400 steps` 仍明显成立，才允许再扩：

- `600 steps`
- 或 `ckpt averaging`

## 4. Promote 判定

`A1` 长实验进入 `F` 后，必须和三路比较：

- `I0 incumbent`
- `A2 frozen candidate = fallback_180`
- `A1 best long raw`

若 raw `A1 best` health 发红：

- 先跑一轮和 `A2` 同类的 health surgical compare
- 再拿 repaired candidate 进 promote compare

推荐 promote 门槛：

- `full-val reconstructed >= fallback_180 + 0.15`
- `hard >= fallback_180 - 0.10`
- `health no_regression vs incumbent = true`

若没超过 `fallback_180`，但互补错误模式明显：

- 降级进 `A3 candidate pool`

## 5. 预估时间

单卡串行、CPU diagnose 不并发抢 GPU 的条件下：

- `A1R_P1 build`
  - `15-25 分钟`
- `A1R_P2 probe`
  - `55-75 分钟`
  - 四臂：`C0 / E5 / E10 / E15`
- `A1R_Wlite @ 400`
  - `35-50 分钟`
  - 两臂：`C0_W / Ebest_W`
- `A1R_F/promote`
  - raw full-val decode: `85-95 分钟`
  - full-val diagnose: `45-70 分钟`
  - health compare + freeze: `10-20 分钟`

所以 wall-clock 预算应写成：

- 只到 probe 判定：
  - `1.25-1.75 小时`
- probe 过线并跑到 W-lite：
  - `2-2.5 小时`
- 一路跑到 promote 判定：
  - `4.5-6 小时`

所有超过 `10 分钟` 的阶段都应挂 `tmux`：

- `winner_a1r_build`
- `winner_a1r_probe`
- `winner_a1r_wlite`
- `winner_a1r_fullval`
