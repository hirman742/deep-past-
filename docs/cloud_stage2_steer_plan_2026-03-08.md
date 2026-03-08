# 二期云端 Steer 自迭代方案（2026-03-08）

## 0. 文档定位

- 本文档维护在本地仓库中，用于后续上云执行二期 steer 自迭代。
- 当前离比赛结束约 10 天，目标不是做一套完美研究方案，而是在高强度租机节奏下，用最少的无效算力换取最大的涨分机会。
- 本方案明确采用激进打分思路。当前 `13.8639` 只是 incumbent，不是目标，更不是天花板。
- 北极星目标可以继续定到 `30`，但自动控制器不能直接拿 `30` 做停机阈值，否则只会把所有候选一锅端掉。控制器必须使用“分阶段存活阈值 + 激进晋级阈值 + 人工终审”三层逻辑。
- 你已经明确说明，真实租机强度往往是单日 `8` 小时以上，且 credits 投入预期较高。因此本方案不走“极端保守省钱”路线，而走“高利用率、快裁撤、快迭代、保留主线证据链”的路线。

## 1. 当前基线与仓库事实

### 1.1 当前正式基线

当前正式 best 仍然是：

- `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/decode_grid_best.json`
- `eval_geom = 13.86392507487608`
- `eval_bleu = 7.1960007050296815`
- `eval_chrfpp = 26.710450201517165`

对应 decode 组合：

- `num_beams = 4`
- `length_penalty = 0.8`
- `no_repeat_ngram_size = 0`
- `min_new_tokens = 0`
- `max_new_tokens = 384`

### 1.2 当前主线模型是什么

当前正式主线仍然是：

- 基座模型：`google/byt5-small`
- 输入长度：`max_source_length = 512`
- 输出长度：`max_target_length = 512`
- LoRA 模块：`[q_proj, v_proj]`
- LoRA 秩：`r = 8`
- LoRA alpha：`16`
- 精度：`bf16 = true`
- 训练阶段 best 选择依据：`eval_loss`
- 正式比较 winner 的依据：`decode 后的 geom`

这说明当前主线有一个结构性错位：

- 训练过程用 `eval_loss` 选 checkpoint。
- 比赛结果却用 `geom` 说话。
- 二期 steer 的核心任务之一，就是用中间 gate 把这两套目标尽量对齐。

### 1.3 当前真实耗时结构

从仓库产物能直接读出两个关键事实：

1. 训练本身并不昂贵。

- 文件：`runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/run_summary.json`
- `4000 steps` 训练总耗时约 `1577.89 秒`
- 约等于 `26.3 分钟`
- 平均约 `2.535 steps/s`

2. decode 才是主要成本项。

- 文件：`runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/decode_grid_best.json`
- 当前正式 best 单组合全量 decode 的 `elapsed_seconds = 2063.34`
- 约等于 `34.4 分钟`

结论：

- 训练不是当前最贵的部分。
- 租机成本的主要消耗来自 `decode + diagnose + 重复 gate`。
- 所以真正要优化的不是“怎么把训练步数压到极致”，而是“怎么尽早识别不值得继续 decode 的候选”。

### 1.4 长度仍然是硬瓶颈

历史长度统计文件 `data/processed_byt5_chunks/length_stats.json` 显示：

- 在 `384/384` 下，`source truncation_ratio_pct = 59.71%`
- 在 `384/384` 下，`target truncation_ratio_pct = 51.12%`
- 即使把 source 提到 `512`，历史统计里的 source 截断率仍有 `43.88%`

这不是当前 `512/512` 配方的实时精确截断率，但足以说明：

- 长度仍然是主瓶颈之一。
- `len640` 分支是有历史证据支撑的，不是拍脑袋。

## 2. 激进目标如何落成可执行控制逻辑

### 2.1 为什么不能直接把自动阈值写成 30

比赛总目标可以非常激进，甚至继续盯着 `30`。

但自动化控制器必须承认现实：

- 当前单折正式 best 是 `13.8639`
- 任何一次小改超参都不可能用一步把自动 gate 从 `13.8` 跳到 `30`
- 如果直接把 `30` 写成自动停机条件，控制器只会把所有路线都判死刑

因此要把“激进目标”拆成多级里程碑，而不是一个不可能一步命中的终点值。

### 2.2 建议的分层目标

建议把目标分成四层：

| 层级 | 含义 | 建议阈值 |
|---|---|---|
| 生存阈值 | 候选值得继续活下去 | promote 阶段相对 incumbent `geom +0.30` 以上 |
| 强信号阈值 | 候选已显示出真实突破趋势 | 单折相对 incumbent `geom +0.50` 以上 |
| 激进胜出阈值 | 候选足以冻结 recipe 并转人工决策 | 绝对 `geom >= 20.0`，且 BLEU/chrF++ 与输出健康指标没有炸裂 |
| 北极星目标 | 比赛后段综合目标 | `geom ≈ 30`，更可能依赖 recipe 跃迁、多折、集成、rerank 和高质量增量数据 |

这里最关键的一点是：

- `+0.30` 不是最终目标，只是“自动晋级到下一决策层”的最低门槛。
- 真正的激进要求体现在 `+0.50` 以上的强信号，以及绝对 `geom >= 20.0` 的冻结级结果上。

## 3. 为什么允许“训练一段 -> gate -> 再训练”

### 3.1 允许的情况

如果同时满足下面三点，那么这是合法的“分段训练”，不是偷换实验：

- 还是同一个候选
- 还是同一个 `run_dir`
- 还是同一套权重形状与优化状态，用 `--resume-from-checkpoint` 接着跑

### 3.2 必须新开候选的情况

如果 gate 后改了任何核心条件，就必须新开候选：

- 学习率
- batch size
- gradient accumulation
- max source / target length
- LoRA target modules
- LoRA rank / alpha / dropout
- 数据源或数据筛选规则

### 3.3 本方案的立场

- 同候选分段续训：允许，而且应该成为主流程。
- 改核心超参后继续：允许，但必须新建候选。
- 不允许一边改 recipe，一边把结果伪装成原主线自然续跑。

## 4. 已落到仓库的脚本能力

### 4.1 `scripts/train_mt5_lora.py`

已补上的能力：

- 恢复遵守 `training.predict_with_generate`
- 增加 `--resume-from-checkpoint`
- 当 `predict_with_generate: false` 时，不再假装 decode 指标在 trainer eval 中可直接用
- `run_summary.json` 中补充更完整的 trainer-state 与 GPU 摘要字段
- 新代码将显存利用率口径向 `reserved` 靠拢，避免仅看 `allocated` 低估显存占用

### 4.2 `scripts/eval_decode_grid.py`

已补上的能力：

- 支持 `--checkpoint-dir`
- 支持 `--tag`
- stage 产物输出到 `diagnostics/`，不会覆盖正式 best 文件

### 4.3 `scripts/steer_stage2_cloud.py`

当前控制器骨架已经能做：

- 根据 YAML 生成候选配置
- 对 incumbent 跑同设置 stage baseline
- 对候选执行“训练 -> decode -> diagnose -> 比较 -> 决策”
- 汇总各阶段结果到总 summary

### 4.4 当前仍然存在的缺口

这些缺口需要明确写出来，不回避：

- 当前 gate 仍然主要 decode `best_model`，还没有正式做 `best_model + latest_checkpoint` 双 checkpoint 竞赛
- warmup 子集仍不是分层抽样 manifest
- 历史老 run 的显存摘要是旧口径，因此你看到的 `3GB+` 与 `nvidia-smi` 的 `10GB+` 并不冲突

## 5. 当前推荐候选与顺序

| 顺序 | 候选名 | 类型 | 主要目标 | 为什么现在就该跑 | 主要风险 |
|---|---|---|---|---|---|
| 1 | `continue_s4_bs32_len512` | warm-start | 先把显存和吞吐吃满 | 最便宜，最能回答“低显存利用率是不是 batch 太保守” | 分数不一定涨，可能只是利用率好看 |
| 2 | `continue_s4_bs24_len640` | warm-start | 用显存换更长上下文 | 长度瓶颈有历史证据支撑，且 BLEU 偏弱时最有希望 | decode 更慢，长度稳定性更难控 |
| 3 | `cold_qkvo_r16_len640` | cold-start | 同时增加容量和上下文 | 上限可能最高，是最激进的分支 | 冷启动、LoRA 结构变化大、最容易烧 credits |

### 5.1 为什么不把 `byt5-base` 放进首轮

不是说 `byt5-base` 永远不值得做，而是说当前优先级不如下面三条：

- 吃满显存
- 拉长长度
- 提高 LoRA 容量

理由：

- 这三条更便宜
- 更容易形成阶段性结论
- 更适合在 10 天窗口内做快速淘汰

如果这三条都没有给出明显涨分方向，再考虑是否值得单开 `byt5-base` 风险线。

## 6. 新的三段式 Steer 结构

本轮文档与 YAML 已切成三段：

1. `probe`
2. `warmup`
3. `promote`

这样做是为了同时满足两个目标：

- 激进地加快迭代
- 不因为 32 样本噪声太大而误判主线

### 6.1 Probe 阶段

当前 YAML 中，probe 设置为：

- `max_steps = 250`
- `eval_steps = 100`
- decode 样本数：`32`
- diagnose 样本数：`32`
- decode 组合：`beam=4, lp=0.8, no_repeat_ngram=0, max_new_tokens=384`

Probe 的定位不是决定 winner，而是：

- 用极小成本砍掉明显跑偏的候选
- 尽早暴露 OOM、异常输出、完全不吃显存这类问题
- 为高 credits 强度下的快速迭代提供第一道闸门

### 6.2 Warmup 阶段

当前 YAML 中，warmup 设置为：

- `max_steps = 600`
- `eval_steps = 200`
- decode 样本数：`64`
- diagnose 样本数：`64`
- decode 组合仍保留 `1` 组：`beam=4, lp=0.8`

Warmup 用于决定候选是否值得进入 promote，重点看质量、生存约束和资源利用率的联合结果。

### 6.3 Promote 阶段

当前 YAML 中，promote 设置为：

- `max_steps = 3000`
- `eval_steps = 300`
- decode 组合数：`4`
  - `beam = 4, 6`
  - `length_penalty = 0.8, 1.0`
  - `no_repeat_ngram_size = 0`
  - `max_new_tokens = 384`
- decode 用全量验证折

这一步也比上一版更激进：

- promote 总步数从 `1800` 提到 `3000`
- 这样做的原因很简单：训练便宜，decode 才贵
- 既然 promote 已经决定要花 decode 大钱，就没有必要把训练段压得过短

## 7. 边界条件：硬停、软警告、人工复核

### 7.1 适用范围说明

GPU 利用率边界只适用于下面两类任务：

- 训练任务
- 模型 decode / diagnose 任务

不适用于纯 CPU 更合理的任务，例如：

- 数据拷贝
- CSV 清洗
- 结果汇总
- 某些轻量 rerank 分析

这些 CPU 型任务不应被强行套进 GPU 利用率框架，否则会误判。

### 7.2 Probe 硬停边界

只要命中任意一条，probe 直接失败：

- `geom` 相对 incumbent 同设置 baseline 下降超过 `1.20`
- `BLEU` 下降超过 `1.00`
- `chrF++` 下降超过 `2.00`
- `exact_extra_id_0_ratio_pct > 1.0`
- `pred_shorter_than_half_ref_ratio_pct > 60.0`
- `empty_prediction_ratio_pct > 3.0`
- `copy_source_ratio_pct > 12.0`
- `gpu_peak_utilization_pct < 50.0`
- `gpu_peak_utilization_pct > 95.0`
- OOM
- decode 或 diagnose 缺失

Probe 之所以允许相对更宽的文本质量波动，是因为：

- `32` 样本本身方差更大
- probe 的职责是砍掉明显坏线，而不是决定最终 winner

### 7.3 Warmup 硬停边界

只要命中任意一条，warmup 直接失败：

- `geom` 相对 incumbent 同设置 baseline 下降超过 `0.60`
- `BLEU` 下降超过 `0.50`
- `chrF++` 下降超过 `1.20`
- `exact_extra_id_0_ratio_pct > 0.5`
- `pred_shorter_than_half_ref_ratio_pct > 45.0`
- `empty_prediction_ratio_pct > 2.0`
- `copy_source_ratio_pct > 10.0`
- `gpu_peak_utilization_pct < 50.0`
- `gpu_peak_utilization_pct > 95.0`
- OOM
- decode 或 diagnose 缺失

这组区间用于保证候选尽量吃满 GPU，但不越稳定性边界。

### 7.4 Warmup 软警告区

命中后不必立刻停，但必须标黄：

- `50% <= gpu_peak_utilization_pct < 70%`
- `geom` 没明显提升，但 `BLEU` 或 `chrF++` 出现单边改善
- `eval_loss` 在下降，但 decode 指标基本不动
- 训练吞吐显著变慢，但没有换来相应质量提升

低于 `50%` 的候选直接视为 warmup 失败；进入软警告区的候选必须结合质量信号决定是否继续。

### 7.5 Warmup 人工复核区

满足任一情形，就不要让 AI 单独拍板：

- `geom` 落在 `[-0.10, +0.30)` 之间
- `BLEU` 明显上涨，但 `geom` 尚未形成清晰优势
- 输出健康指标正常，但 GPU 利用率只是勉强达标
- 冷启动候选在 600 步还没完全起飞，但 loss、长度和输出健康信号在变好

这部分建议正式落成三态：

- `accept`
- `review`
- `reject`

### 7.6 Promote 接受边界

候选进入 promote 后，只有同时满足下面条件，才有资格挑战 incumbent：

- `geom` 相对 incumbent 全量 baseline 至少提升 `+0.30`
- `BLEU` 不得下降超过 `0.20`
- `chrF++` 不得下降超过 `0.60`
- `exact_extra_id_0_ratio_pct <= 0.5`
- `pred_shorter_than_half_ref_ratio_pct <= 45.0`
- `empty_prediction_ratio_pct <= 2.0`
- `copy_source_ratio_pct <= 10.0`
- `60% <= gpu_peak_utilization_pct <= 95%`

这里也直接回应你的评论：

- 上一版写成 `+0.10`，确实太弱
- 现在把自动晋级线提升到 `+0.30`
- `+0.10` 最多只能算“值得保留痕迹”，不配当作 aggressive mainline winner

### 7.7 强信号与激进胜出边界

建议在文档层面再加两层业务判断：

1. 强信号候选：

- 单折相对 incumbent `geom +0.50` 以上
- 或绝对 `geom >= 14.5` 且 BLEU / chrF++ 同步向上

2. 激进胜出候选：

- 绝对 `geom >= 20.0`
- 且 BLEU / chrF++ / 输出健康指标没有炸裂
- 且显存利用率与吞吐没有明显倒退

一旦出现激进胜出候选，就不应该继续无上限扩 recipe，而应先冻结它，转入人类决策：

- 继续追单折
- 还是转向 `fold1`
- 还是做多折 / rerank / ensemble

## 8. 推荐执行顺序与停机策略

### 8.1 推荐顺序

建议严格按下面顺序推进：

1. `continue_s4_bs32_len512`
2. `continue_s4_bs24_len640`
3. `cold_qkvo_r16_len640`

原因：

- 第一条最便宜，先确认 GPU 利用率问题是不是 batch 保守导致
- 第二条最有可能把长度瓶颈转成实际涨分
- 第三条最贵，必须压到最后

### 8.2 具体停机规则

- 第一条若 probe 就失败，优先看第二条，不要急着烧第三条
- 第一条若 warmup 通过但只给弱信号，可以先挂起，再看第二条能否形成更强趋势
- 第二条若 promote 后出现 `+0.50` 以上强信号，应优先做同 recipe 复核与补证，但不因为强信号本身就自动终止全局搜索
- 第三条只有在前两条没有给出强信号时才值得认真烧完

### 8.3 什么时候先停下来交给人类

建议把“人工优先接管”和“冻结 recipe”分开写：

1. 进入人工优先复核区：

- 单折 `geom` 相对 incumbent 稳定提升 `+0.50` 以上
- 或绝对 `geom >= 15.0`
- `BLEU / chrF++ / 输出健康指标` 没有炸裂
- GPU 利用率和吞吐没有明显倒退

这时更合理的动作是：优先复核这条 recipe，但不必因为它刚出现强信号就强制停掉全部搜索。

2. 直接视为激进 winner：

- 绝对 `geom >= 20.0`
- 且 `BLEU / chrF++ / 输出健康指标` 没有炸裂
- 且 GPU 利用率与吞吐没有明显倒退

此时更合理的动作不是继续刷无穷实验，而是：

- 冻结 recipe
- 复核边界条件
- 决定是否进入 `fold1 / CV / ensemble / rerank` 阶段

### 8.4 CV 升级档位与边际效应处理

建议把交叉验证升级规则写成分档决策，而不是一次性铺开：

- `Tier0`：`geom < 25`
  - 继续单折自迭代，不做 CV。
- `Tier1`：`25 <= geom < 30`
  - 进入 `2-fold CV`。
  - 目标是验证 recipe 是否具备迁移性。
- `Tier2`：`30 <= geom < 35`
  - 仍以 `2-fold CV` 为主，并开始准备 OOF / ensemble 管线。
  - 若 `2-fold` 复现稳定，再决定是否继续扩折。
- `Tier3`：`geom >= 35`
  - 直接进入 `5-fold CV`。
  - 这一档的目标不再是找方向，而是尽快榨取最终成绩。

执行前提：

- 这里的 `geom` 指单折 full-val 主指标。
- 升档前至少完成一次同 recipe 复核，避免把偶然波动直接放大成多折成本。

若连续几次迭代出现边际效应，应立即暂停扩线并做根因分析。建议把边际效应定义为满足任一条件：

- 连续 `2` 个 promote 候选，`geom` 提升都 `< +0.20`
- 连续 `3` 个候选都只改善 GPU 利用率或吞吐，不改善质量
- `BLEU / chrF++` 持续小幅摆动，但没有形成结构性突破

根因分析优先级：

- checkpoint 选择错位：`eval_loss` 与 decode winner 不一致
- decode 瓶颈：模型有改进，但推理参数限制了分数
- 长度瓶颈：长度扩展仍不足，截断继续压分
- 容量瓶颈：`byt5-small + 当前 LoRA` 已接近上限
- 数据瓶颈：高质量增量样本不足
- 验证偏差：过拟合 `fold0` 或过拟合公榜反馈
## 9. 租机时长与 credits / 金额评估

### 9.1 估算原则

你的实际约束不是“没钱”，而是：

- credits 投入会比较高
- 但不允许浪费分毫算力
- 要求最大化利用率，同时保证快速迭代

因此这里不再沿用“极端省钱”的思路，而采用：

- 愿意为强信号候选花够时长
- 但绝不允许把 credits 烧在弱候选的重复 decode 上

### 9.2 估算依据

本估算基于仓库中的真实记录：

- S4 stage2 训练 `4000 steps` 约 `26.3 分钟`
- 单组合全量 decode 约 `34.4 分钟`
- 新版控制器已把最贵的早期 gate 缩成 `probe=32`、`warmup=64`
- promote 仍然贵，因为它要跑全量验证折与 `4` 个 decode 组合
- 启动实例后，你还有约 `1 小时` 固定损耗：上传数据、下载插件、拷贝数据到 git 工作目录

### 9.3 分阶段耗时估计

#### 固定准备开销

- `1.0 小时`

#### Stage baseline 开销

控制器第一次执行时，还要为 incumbent 计算各阶段 baseline：

- `probe baseline`：约 `0.05 - 0.15 小时`
- `warmup baseline`：约 `0.10 - 0.20 小时`
- `promote baseline`：约 `2.6 - 3.4 小时`

合计：

- 约 `2.75 - 3.75 小时`

#### 单个候选开销

1. Probe 就失败：

- 约 `0.10 - 0.20 小时`

2. 通过 probe，但 warmup 失败：

- 约 `0.30 - 0.55 小时`

3. 完整走到 promote 结束：

- probe + warmup：约 `0.35 - 0.60 小时`
- promote 训练追加：约 `0.20 - 0.35 小时`
- promote 全量 decode + diagnose：约 `2.7 - 3.6 小时`
- 合计：约 `3.25 - 4.55 小时`

这也再次说明：

- 训练不是成本大头
- 全量 promote decode 才是 credits 黑洞

### 9.4 推荐租机时长

#### 方案 A：快速筛三条线

目标：

- 做完准备
- 做完 baseline
- 三条候选至少都经过 `probe + warmup`

推荐租机：

- `6 - 8 小时`

适用情况：

- 你当前最关心的是判断“哪条线值得继续烧”
- 不追求当日就把所有 promote 全跑完

#### 方案 B：高强度主线推进

目标：

- 做完准备
- 做完 baseline
- 完整推进前两条主候选
- 至少让 `1` 条强候选完整走完 promote

推荐租机：

- `12 - 16 小时`

这是当前最推荐的首轮租机区间。

理由：

- 你已经明确单日会高强度使用算力
- 这个区间既不寒酸，也不至于无脑包太长
- 足够支撑 aggressive 探索，同时仍保留早停节制

#### 方案 C：激进全跑

目标：

- 三条候选都给完整 promote 机会
- 保留重试与人工复核窗口

推荐租机：

- `18 - 24 小时`

适用情况：

- 你确认要把 `cold_qkvo_r16_len640` 也认真打到底
- 并愿意接受它是高波动高成本分支

### 9.5 credits / 金额换算

如果平台按 credits 计费，本质上只是把“小时单价”换成“每小时消耗多少 credits”。

因此通用公式仍然是：

- `总消耗 = 时长 × 每小时单价`

下面继续用 `$ / h` 只是为了直观比较，不影响 credits 模型。

#### 12 小时表

| 单价 | 12 小时总消耗 |
|---|---|
| `$0.35/h` | `$4.20` |
| `$1/h` | `$12` |
| `$2/h` | `$24` |
| `$4/h` | `$48` |
| `$10/h` | `$120` |
| `$40/h` | `$480` |

#### 16 小时表

| 单价 | 16 小时总消耗 |
|---|---|
| `$0.35/h` | `$5.60` |
| `$1/h` | `$16` |
| `$2/h` | `$32` |
| `$4/h` | `$64` |
| `$10/h` | `$160` |
| `$40/h` | `$640` |

#### 24 小时表

| 单价 | 24 小时总消耗 |
|---|---|
| `$0.35/h` | `$8.40` |
| `$1/h` | `$24` |
| `$2/h` | `$48` |
| `$4/h` | `$96` |
| `$10/h` | `$240` |
| `$40/h` | `$960` |

### 9.6 最终租机建议

如果现在就租，我建议：

- 首轮按 `12 - 16 小时` 规划
- 如果前两条主候选给出强信号，再视情况续到 `18 - 24 小时`

理由不是“钱多任性”，而是：

- 你本来就会高强度使用算力
- 太短的时长会迫使你在关键 promote decode 上半途收手
- 真正浪费 credits 的不是租长一点，而是让弱候选跑完整套昂贵 decode

## 10. 如何最大化 credits 效率，而不是机械省钱

### 10.1 三条基本原则

- 不让 GPU 在训练/解码窗口里闲着
- 不让弱候选走到昂贵 full decode
- 不把纯 CPU 任务误塞进 GPU 利用率框架

### 10.2 decode 节流原则

对于 decode，我赞同慎重，但这里的“慎重”不是少跑，而是只把 decode 花在有资格的对象上：

- `probe` 只跑 `32`
- `warmup` 只跑 `64`
- 只有进入 `promote` 的候选，才配跑全量验证折和 4 组 decode 组合

这样做的逻辑是：

- credits 可以高投入
- 但高投入必须有资格门槛
- decode 必须从“默认全跑”改成“晋级后再花钱”

### 10.3 如果还要进一步压榨效率

若租机时长非常紧张，可优先裁剪：

- `beam=6` 的 promote decode

不要优先裁剪：

- probe
- warmup
- 训练总步数本身

原因：

- probe / warmup 是筛选器
- 训练本身不贵
- 最贵的是 full-val decode 组合扩张

## 11. 推荐执行命令

### 11.1 本地 dry-run

```bash
python scripts/steer_stage2_cloud.py --plan configs/cloud_stage2_steer.yaml --dry-run --candidates continue_s4_bs32_len512
```

### 11.2 云端顺序执行

```bash
python scripts/steer_stage2_cloud.py --plan configs/cloud_stage2_steer.yaml --skip-existing --candidates continue_s4_bs32_len512
python scripts/steer_stage2_cloud.py --plan configs/cloud_stage2_steer.yaml --skip-existing --candidates continue_s4_bs24_len640
python scripts/steer_stage2_cloud.py --plan configs/cloud_stage2_steer.yaml --skip-existing --candidates cold_qkvo_r16_len640
```

不建议一开始就无脑三条线串满。

更合理的流程是：

- 第一条先过 `probe`
- 再看是否值得走 `warmup`
- 再决定是否给它完整 `promote`
- 之后再决定第二条与第三条的优先级

## 12. 下一批必须直接落代码的补强

这一节不再停留在“保持补充精神”的抽象口号，而直接落成待办。

### 12.1 三态决策落代码

需要把当前二元 `accept / reject` 扩成三态：

- `accept`
- `review`
- `reject`

具体要落的字段：

- `decision`
- `decision_reason`
- `hard_failures`
- `soft_warnings`
- `review_flags`

### 12.2 双 checkpoint gate

每个阶段至少比两份 checkpoint：

- `best_model`
- `latest_checkpoint`

实际执行上：

- 两边都 decode
- 取较强者参与 gate
- 同时把较弱者也保留在 summary 中，供人复核“是不是 trainer 选错 checkpoint”

### 12.3 分层 warmup 子集 manifest

不要再默认用固定切片当 warmup 子集。

建议落一个 manifest 文件，至少分层考虑：

- parent 粒度
- 长度桶
- 文本难度或 chunk 数

这样可以减少 warmup 因样本偏而误判路线。

### 12.4 显存口径统一

统一记录三类数据：

- `allocated`
- `reserved`
- `nvidia-smi peak`

并在 summary 中明确：

- 哪个数字用于自动 gate
- 哪个数字只作诊断参考

当前建议以后续自动 gate 主要使用：

- `reserved`
- 并在人工复核时交叉看 `nvidia-smi`

### 12.5 晋级与停搜规则代码化

应把下面这套业务规则直接写入控制器逻辑：

- 绝对 `geom >= 25.0` 才允许自动 accept 到主线挑战层
- 单折相对 incumbent `geom +1.0` 以上标记为强信号
- 绝对 `geom >= 30.0` 标记为 Tier2 候选，优先进入稳定性复核与 `2-fold CV` 排队
- 绝对 `geom >= 35.0` 标记为激进 winner，并直接进入 `5-fold CV` 决策层
- 出现激进 winner 后，默认暂停 recipe 搜索，等待人工确认是否转 `5-fold CV / ensemble`

建议在 summary 中同时落盘：

- `tier`
- `decision`
- `decision_reason`

这些参数未来仍然会在自迭代中根据分数走势、边际效应和算力使用情况继续调整，不应视为永久固定常量。
## 13. 最终结论

二期最合理的打法不是继续在一期路径上做无穷无尽的小修小补，而是：

- 用 incumbent 守住证据链
- 用 `probe(32) -> warmup(64) -> promote(3000)` 建立高裁撤率、高利用率的激进赛制
- 用更高的 GPU 利用率门槛和更强的单折晋级阈值保证“激进但不乱烧”
- 用分层目标把“冲 30”从口号拆成可执行里程碑
- 一旦出现激进 winner，或某条强信号路线持续压制其他候选，就先停下来，交给人类决定是否转折验证、多折、集成或 rerank

这套方案的核心不是“尽量少花钱”，而是“在高 credits 投入前提下，不浪费分毫算力，把每一小时都花在最可能涨分的地方”。







