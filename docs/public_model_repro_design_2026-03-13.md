# 公开高分路线复现设计
## 第二阶段修订版，供审计

本方案承接：

- [public_model_fast_repro_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_fast_repro_plan_2026-03-13.md)
- [public_model_phase1_decode_ablation_2026-03-13.md](/workspace/deep-past-/docs/public_model_phase1_decode_ablation_2026-03-13.md)
- [next_step_discipline_2026-03-13.md](/workspace/deep-past-/docs/next_step_discipline_2026-03-13.md)

当前更具体、可直接审计的 `R1 raw-row + supervised mix` 执行稿见：

- [public_model_r1_rawrow_supervised_mix_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_r1_rawrow_supervised_mix_plan_2026-03-13.md)

这版设计只做三件事：

1. 把仓库当前已经发生过的事实写实，不再泛化
2. 把旧主线的结构、参数、前后处理层级说清楚
3. 把“更大主干 + P/F + 串并行 smoke”收敛成可以执行的第二阶段方案

## 0. 事实边界

### 0.1 当前事实清洗基线

当前仓库已经落地、且能被事实核验的 source-side 清洗基线，不是泛指的“规范化”，而是：

- [semantic_cleaning_spec.md](/workspace/deep-past-/docs/semantic_cleaning_spec.md) 中明确写出的 `Gate 0-A`
- 以及该文档定义的 `Tier-0` 安全规范化基线

可以硬确认的事实：

- `Gate 0-A` 已完成：inventory / patterns / test-only 差集 / Tier-0 可落地
- 当前清洗宪法要求三层清洗：`Tier-0 / Tier-1 / Tier-2`
- 但当前可直接作为主线事实基线的，是 `Tier-0`
- `Tier-0` 只允许：
  - 空白/不可见字符/Unicode 形态规范化
  - 极低风险统一
  - 且不得改变可见语义标记

因此，后续所有“更严格但不丧失语义”的说法，都必须落回这条事实边界：

- 当前仓库不是已经在大规模使用 `Tier-1 / Tier-2` 做主线训练
- 当前可审计、可直接复现的主线清洗事实，是 `Gate 0-A` 支撑下的 `Tier-0`

### 0.2 当前 official 评测边界

当前仓库已经有 `official-like` 口径，但正式 bridge 仍缺失。

因此本方案仍坚持：

- 本地固定口径：`official-formula local`
- 指标定义：`geom = sqrt(corpus BLEU * corpus chrF++)`
- 它是官方公式兼容，不是隐藏测试集真实官方分数

## 1. 旧主线的结构性说明

### 1.1 旧主线不是单点 tweak，而是一条完整链路

从 [taskform_experiment_report_2026-03-10.md](/workspace/deep-past-/docs/taskform_experiment_report_2026-03-10.md)、[taskform_winner_stage_report_2026-03-10.md](/workspace/deep-past-/docs/taskform_winner_stage_report_2026-03-10.md)、[taskform_winner_retrieval_gate_pwf_2026-03-11.md](/workspace/deep-past-/docs/taskform_winner_retrieval_gate_pwf_2026-03-11.md) 可以把旧主线还原成六层：

1. `Gate 0-A / Tier-0` 约束下的 source-side 清洗
2. chunk 化训练集构造
3. `Gale-Church` 风格 short-aligned 扩容
4. `ByT5-small + LoRA(q,v,r=8)` 三段 continue 训练
5. retrieval / replay / fallback / chooser / term patch 等后置层
6. `beam=4 / lp=0.7 / max_new_tokens=384` 风格 decode 和本地代理评测

真正的问题不是“旧主线没有工程”，而是：

- 前四层把一个轻主干推到了局部最优
- 后两层承担了过多“救输出、补健康、修主分”的职责

### 1.2 旧主线的数据与特征工程

旧主线训练时，数据和特征工程并不弱，主要包括：

- source/target 标准列重建
- chunk 化切分
- ratio 模式 chunk
- `Gale-Church` 风格 short-aligned pair 增广
- source 侧任务前缀统一
- `suppress_extra_ids = true`
- `bad_tokens_regex = <extra_id_\\d+>`

其中最关键的两条不是小规则，而是：

1. chunk 化  
把超长 parent 拆成可训练的局部子问题

2. short-aligned augmentation  
用简化 `Gale-Church` DP 在 source/target 片段间找 `1:1 / 1:2 / 2:1` 对齐对，再把这些对齐段作为额外监督样本混入训练集

当前 winner processed 数据中，这部分不是边角料，而是显著占比：

- `chunk_mode = ratio`: `3026`
- `chunk_mode = short_aligned_gale_church`: `1648`
- `chunk_mode = none`: `615`

这说明旧主线的真实特征工程不是“只有原始比赛对”，而是：

- 原始对
- chunk
- short-aligned 局部对齐监督

三者混合。

### 1.3 旧主线模型架构

旧主线的核心基座是 `google/byt5-small`，关键结构参数为：

- `d_model = 1472`
- `d_ff = 3584`
- `num_layers = 12`
- `num_decoder_layers = 4`
- `num_heads = 6`
- `vocab_size = 384`

训练形态不是全参，而是极轻量 LoRA：

- target modules: `q / v`
- `r = 8`
- `alpha = 16`
- `dropout = 0.0`
- `bias = none`

旧 winner 量级：

- `total_params = 301,362,176`
- `trainable_params = 593,920`
- `trainable_ratio_pct = 0.1971%`

因此旧主线的本质不是“大模型训练”，而是：

- byte-level ByT5 小主干
- 极低比例 LoRA continue

### 1.4 旧主线训练链

旧 winner 是三段链，不是单 run：

1. `Stage1`
   - config: `configs/cloud_stage1_len512_lr2e4.yaml`
   - backbone: `google/byt5-small`
   - LoRA: `q/v, r=8, alpha=16`
   - `len = 512`
   - `lr = 2e-4`
   - `bs = 16`
   - `grad_accum = 2`
   - `epochs = 30`

2. `Stage2`
   - config: `configs/cloud_stage2_gc_curriculum_cost14_from_s1win.yaml`
   - init adapter: stage1 winner
   - processed_dir: `data/processed_byt5_chunks_align_gc_cost14`
   - `len = 512`
   - `lr = 1e-4`
   - `bs = 16`
   - `grad_accum = 2`
   - `epochs = 8`

3. `Continue winner stage`
   - generated config: `continue_s4_bs24_len640_seg5.yaml`
   - init adapter: stage2 winner
   - `len = 640`
   - `lr = 5e-5`
   - `bs = 24`
   - `grad_accum = 1`
   - `epochs = 8`
   - `bf16 = true`
   - `gradient_checkpointing = true`

换句话说，旧主线的真正 recipe 是：

- `ByT5-small`
- LoRA continue
- chunk
- GC short-aligned augmentation
- 降学习率继续训

### 1.5 旧主线 decode 与后置层

旧主线常用 decode 是：

- `beam = 4`
- `length_penalty = 0.7`
- `max_new_tokens = 384`

旧主线后置层包括：

- retrieval top1 / W-lite
- replay15 / replay25
- fallback routing
- chooser / pairwise gate
- term-aware patch
- 当前 formulation 的 `MBR`

这些层在仓库里的真实定位应当写成：

- 它们是建立在小主干上的后置增强或后置修补
- 其中有些局部正，有些已被判负
- 它们不是一条已经和官方高分路线对齐的主提交链路

### 1.6 旧主线参考的仓库先验

当前仓库里明确能读到的先验/参考方向，不应写成“神秘黑箱经验”，而应写成以下几类：

- byte-level `ByT5`
  - 目标：减轻 transliteration 任务上的分词脆弱性
- `LoRA`
  - 目标：在有限算力下做 continue / adapter 微调
- `Gale-Church` 风格短对齐
  - 目标：补局部对齐监督
- `SacreBLEU + chrF++`
  - 目标：用 corpus 级指标做统一评测
- retrieval / `kNN-MT`
  - 目标：引入可检索记忆
- efficient `MBR`
  - 目标：在候选层做重排或聚合

仓库事实更支持如下判断：

- 这些先验方向本身没有被整体证伪
- 被证伪的是当前仓库最近这一版具体 formulation

## 2. 第一阶段已经确认的新事实

### 2.1 公开模型不是 adapter-only

已解包公开对象 `byt5 akkadian mbr · default` 是完整模型，不是 LoRA adapter。

可确认结构：

- `T5ForConditionalGeneration`
- `ByT5Tokenizer`
- `d_model = 1536`
- `d_ff = 3968`
- `num_layers = 18`
- `num_decoder_layers = 6`
- `num_heads = 12`

与旧主线对照：

- 它是更强的完整 ByT5 主干
- 我们是小主干 + 极轻 LoRA

### 2.2 第一阶段 decode 对照结论

第一阶段本地验证已确认：

- `beam=8 / lp=1.0 / max640` 是关键 decode 框架
- `repetition_penalty=1.1` 不是主增益来源
- 旧 `beam=4 / lp=0.7 / max384` 会显著压坏强模型表现

因此第二阶段必须把“更强主干”和“新 decode baseline”绑定，而不是分开理解。

## 3. 当前困境的准确判定

当前困境不是“没有工程”，而是下面四条叠加：

1. 事实清洗基线只是 `Gate 0-A / Tier-0`
   - 当前主线并不是建立在更强的、已正式验证通过的 `Tier-1 / Tier-2` 上

2. 主干仍偏弱
   - `ByT5-small + q/v LoRA r8` 更像成本优化线，不像公开高分线

3. 后置层承担了过多职责
   - retrieval / replay / fallback / chooser / patch / MBR 被推到了主序前面

4. official bridge 缺失下仍用 local proxy 驱动主序
   - 这会把优化导向本地局部最优，而不是官方提交最优

## 4. 第二阶段设计原则

第二阶段必须同时遵守四条纪律：

1. 事实先于猜测  
   先按仓库已证实结构复盘，再改架构

2. 强单模先于后置层  
   先站住 backbone 与 decode，再谈 rerank / MBR

3. `P/F` 先于 full rollout  
   先 smoke / probe，再 full

4. 训练串行、轻量 decode 可并行  
   不允许一上来把多条高风险训练线同时压满

## 5. 新主线默认 decode baseline

从第二阶段开始，新主线默认 decode baseline 固定为：

- `num_beams = 8`
- `length_penalty = 1.0`
- `max_new_tokens = 640`
- `min_new_tokens = 0`
- `no_repeat_ngram_size = 0`
- `repetition_penalty = 1.0`
- `suppress_extra_ids = true`
- `bad_tokens_regex = <extra_id_\\d+>`

旧 baseline：

- `beam=4 / lp=0.7 / max384`

降级为：

- historical compare only

不再允许作为：

- 新 checkpoint promote baseline
- 新主线默认 decode

## 6. 更大主干到底多大

### 6.1 第二阶段的“更大主干”定义

第二阶段不把“更大主干”写成一句空话，而是分三档：

1. `P1 主线`
   - `ByT5-base len640 q/v`
   - 这是最近、最可执行、与仓库既有脚本最兼容的第一条升级线

2. `P2 强化线`
   - `ByT5-base len640 q/k/v/o, r=16, alpha=32`
   - 这是容量进一步打开的高风险线

3. `P3 备选 family 线`
   - `mT5-base len640 q/v`
   - 不是首轮主线，只是 reserve

### 6.2 为什么先这样分，而不是直接追公开包大小

公开高分包在结构上更接近“更强完整 ByT5”，但我们目前不知道其完整训练 recipe。

因此最快、最稳的复现策略不是：

- 直接跳到未知的 full finetune 大线

而是：

1. 先用仓库内已有纪律验证 `ByT5-base` 是否给正信号
2. 若 `ByT5-base` 也不给正信号，再反推是否需要更靠近公开包的 full-model 方案

这符合仓库现有实验纪律，也更利于审计。

## 7. 第二阶段的 P/F 方案

### 7.1 P/F 含义

- `P = Probe / Smoke`
- `F = Full`

纪律含义：

- `P` 只回答“这条线有没有资格继续”
- `F` 直接回答“它能不能进入主序”

这次明确取消单独 `W` 层，原因只有一条：

- 过去的 `P/W/F` 在这个仓库里已经开始出现官僚化和口径漂移

新的 `P/F` 原则更硬：

- `P` 不过，立即停
- `P` 过了，直接进 `F`
- 不再保留一个容易无限延长、又容易和 `F` 口径重叠的中间层

### 7.2 P1 主线：`ByT5-base len640 q/v`

这是第二阶段唯一默认先开的主线。

固定条件：

- 清洗：沿用当前事实基线 `Gate 0-A / Tier-0`
- processed 数据：先沿用当前正式 chunk + GC 资产
- decode：统一用新 baseline
- 指标：统一 `official-formula local`

`P1` 分三步：

1. `P1_1`
   - 显存 smoke：`bs=8, grad_acc=3`

2. `P1_2`
   - 显存 smoke：`bs=6, grad_acc=4`

3. `P1_3`
   - 用通过 smoke 的组合跑 `250 steps`

`P1` 产物必须包含：

- generated config
- run_summary
- `ckpt100 / 150 / 200 / 250` anchor 评测
- line winner `diag32`
- health 指标

`P1` 放行条件：

- 至少显著接近或超过当前 matched baseline
- 输出健康不出现明显塌缩
- 资源成本可控

若 `P1` 不过：

- 第二阶段不直接跳去大规模 full
- 先复盘输入构造与 backbone 适配

若 `P1` 过线：

- 直接进入 `F1`
- 不再插入单独 warmup 层

`F1` 内容固定为：

- line winner 单点 full-val decode
- full-val diagnose
- 与当前正式 baseline 做 promote compare
- 记录 full-val health 与成本

### 7.3 P2 强化线：`ByT5-base len640 qkvo_r16`

只有当 `P1` 证明 `ByT5-base` 资源可控且方向为正，才允许开 `P2`。

固定参数：

- backbone: `google/byt5-base`
- `max_source_length = 640`
- `max_target_length = 640`
- LoRA target: `q/k/v/o`
- `r = 16`
- `alpha = 32`

`P2` 结构：

- `P2_1`: 显存 smoke，`bs=4, grad_acc=6`
- `P2_2`: 显存 smoke，`bs=3, grad_acc=8`
- `P2_3`: `250 steps` probe

`P2` 只回答一个问题：

- 更大 backbone 之外，更大 adapter 容量是否也必要

若 `P2` 过线：

- 直接进入 `F2`
- 用与 `F1` 同口径的 full-val compare 决定是否切主线

### 7.4 P3 reserve 线：`mT5-base len640 q/v`

`P3` 不在首轮与 `P1` 并跑。

只有当下面任一条件成立才放行：

- `P1` 正但上限明显不够
- `P1/P2` 都无法稳定形成正信号
- 有明确证据说明 byte-level family 之外需要另一种分词粒度

## 8. 串行/并行纪律

### 8.1 训练 smoke 一律串行

第二阶段训练任务遵守：

1. 所有显存 smoke 串行
2. 所有 `250-step` probe 训练串行
3. 所有 full 训练也串行
4. 同时只允许一条训练主线处于 `in_progress`

原因很简单：

- 当前大 backbone 峰值显存未知
- 训练是最高成本环节
- 并行训练会把 `OOM / 调度抖动 / I/O 干扰` 混进结论

### 8.2 decode / diagnose 可有限并行

只有轻量任务允许有限并行：

- anchor decode
- `diag32`
- `diag64`

并行前提：

- 不挤占当前训练主进程
- GPU 峰值和吞吐仍在安全边界内

## 9. P/F 原则在第二阶段怎么落

这里的 `P/F` 核心只有三条：

- 变量隔离
- 失败即止损
- 过线后直接按同口径 full compare 决策

### 9.1 `P` 阶段

只允许改 backbone / adapter 容量这一类主变量。

不允许在 `P` 阶段同时引入：

- 新清洗层
- 新 retrieval 逻辑
- 新 chooser
- 新 MBR
- 新 term patch

`P` 阶段必须产出：

- generated config
- run summary
- anchor 对照
- `diag32`
- health 指标
- 成本记录

### 9.2 `F` 阶段

只有 `P` 阶段过线，才允许做：

- 单点 full-val decode
- full-val diagnose
- 与当前正式 baseline 的 promote compare

`F` 阶段不再承担“继续观察一下”的功能。

它只回答：

- 这条线能不能进入主序
- 这条线是否值得替换当前正式 baseline

## 10. 快速复现公开高分/健康的最优路径

按优先级，第二阶段之后的最短路径应写成：

1. 固定新 decode baseline  
   不再让旧 `beam4/lp0.7/max384` 扼住强模型

2. 固定事实清洗基线  
   先用 `Gate 0-A / Tier-0`，不在这一轮把清洗变量混进去

3. 只开 `P1: ByT5-base len640 q/v`  
   先判断更大 backbone 是否马上给正信号

4. 若 `P1 -> F1` 已给出明确正信号，再决定是否开 `P2`  
   不并跑，不抢主序

5. 只有强单模站住，再把 retrieval / rerank / MBR 接回去  
   后置层回到“增益工具”的位置

## 11. 本轮明确冻结或降级的旧方向

以下方向在第二阶段不再抢主序：

- `A20` 阈值继续扫
- chooser 再叠规则
- replay band 同质窄扫
- term-aware patch 同质扩臂
- 当前 formulation 的 `MBR` 重开
- 当前 external continue / TAPT-lite 再训一版看看

这些方向的定位改成：

- 归档证据
- reserve 诊断工具
- 只有在强单模已成立后，才可能被重新接入

## 12. 审计时应重点盯的五件事

1. 是否已经把当前事实清洗基线准确写成 `Gate 0-A / Tier-0`
2. 是否已经把旧主线的特征工程、训练结构、后置层拆开说明
3. “更大主干”是否被收敛成可执行的三档，而不是空泛口号
4. `P/F` 与串行 smoke 纪律是否足够硬
5. 新主线是否真的把强单模放回第一优先级
