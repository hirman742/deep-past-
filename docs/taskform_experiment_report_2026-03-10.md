# Taskform 实验总报告（基于当前仓库快照，更新于 2026-03-10）

## 0. 报告范围与证据边界

这份报告基于当前仓库中仍可读取的以下证据编写：

- `docs/next-step_taskform_discipline_2026-03-10.md`
- `docs/next-step_taskform_discipline_2026-03-09.md`
- `docs/taskform_upgrade_probe_round_v2_2026-03-09.md`
- `docs/taskform_p1_wlite_gate_2026-03-09.md`
- `logs/steer_incumbent_promote_decode.log`
- `logs/architecture_probe_round_20260309_summary.json`
- `logs/taskform_probe_round_v2_20260309_summary.json`
- `logs/taskform_p1_wlite_gate_20260309_summary.json`
- `reports/taskform_*` 下当前保留的 `summary.json` / `gate_report.md` / 诊断文件
- `runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/run_summary.json`
- `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/run_summary.json`
- `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/run_summary.json`
- `runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml`
- `configs/cloud_stage1_len512_lr2e4.yaml`
- `configs/cloud_stage2_gc_curriculum_cost14_from_s1win.yaml`
- `scripts/preprocess.py`
- `scripts/build_long_chunks.py`
- `scripts/build_short_aligned_pairs_galechurch.py`
- `scripts/train_mt5_lora.py`
- `scripts/generation_utils.py`
- `scripts/metrics_utils.py`
- `scripts/taskform_a2_a1_flow.py`
- `cleaning/normalize.py`
- `cleaning/configs/cleaning.t0.yaml`

重要边界：

- 本报告只对“当前仓库快照中仍有产物或脚本可复查”的实验做精确复盘。
- 对已经在纪律文档中被归档、但仓库里没有完整产物的旧线，只写能被现有证据支持的结论，不做虚构补细节。
- 报告里的“失败原因”分为两类：
  - 由日志和指标直接支持的硬结论
  - 结合脚本实现与结果做出的工程归因

## 1. 先给结论

到 `2026-03-10` 这版仓库为止，可以成立的总体判断不是“项目整体失败”，而是：

1. 当前正式 winner 仍然是全仓库里最稳、最难被替代的单模型方案。
2. 最近绝大多数新线都没有创造出更强的训练信号，而是在 winner 输出上做局部 patch、融合、过滤或再加工，因此上限天然受限。
3. 真正把线跑死的，不只是“分数没涨”，而是反复出现下面三种失败模式：
   - 训练 `eval_loss` 更好，但 decode 更差。
   - 输出长度/重复控制崩坏，导致 `geom` 明显下滑。
   - 验证口径被污染，浪费了判断机会。
4. 已经足够证伪、应停止继续消耗主线时间的方向包括：
   - 当前 formulation 下的 `dan-1` parent rewrite / edit
   - `ByT5-base / mT5-base` 新架构探测
   - 当前 `A2` 过滤器
   - 当前 fair 对照下的 `TAPT -> matched supervised`
   - `L2` 术语词表和 `L3` row-gate 作为主线提分方案
5. 当前 winner 之所以一直赢，不是因为它没有问题，而是因为它在一个“错误可控、输出行为更稳”的点上；最近多数新线没有拿到比它更强的泛化能力，只拿到了更多失控模式。

## 2. 当前正式 winner 是什么

### 2.1 冻结结论

根据 `docs/next-step_taskform_discipline_2026-03-10.md`，当前冻结正式主线是：

- model family: `ByT5-small chunk`
- frozen checkpoint: `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- frozen decode:
  - `beam=4`
  - `length_penalty=0.7`
  - `max_new_tokens=384`

当前正式分数：

- full-val reconstructed:
  - `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`
- hard routed full:
  - `geom / bleu / chrfpp = 13.7161 / 7.1669 / 26.2499`

### 2.2 一个非常关键的事实

`runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/run_summary.json` 里，trainer 侧按 `eval_loss` 选出的 best checkpoint 实际是：

- `best_model_checkpoint = checkpoint-50`
- `best_metric = 0.8546958565711975`

但当前被冻结的正式 winner 却是：

- `checkpoint-250`

这说明这条线从实践上已经证明一件事：  
**训练期 `eval_loss` 最优，不等于最终 decode 指标最优。**

这也是后续很多“loss 看起来健康、decode 反而变差”实验的总背景。

### 2.3 当前 winner 的 anchor64 对照分

`reports/taskform_a2_a1_20260310/incumbent_anchor64_summary.json` 给出了当前 winner 在 anchor64 上的对照分：

- `geom = 16.5057`
- `bleu = 9.8606`
- `chrfpp = 27.6291`

输出健康度：

- `empty_prediction_ratio_pct = 1.5625`
- `pred_shorter_than_half_ref_ratio_pct = 12.5`
- `unique_prediction_ratio_pct = 100.0`

这组数很重要，因为很多后续实验并不是在 full-val 上直接挑战 winner，而是在 anchor32 / anchor64 / routed hard 等子口径上先做 matched 对照。所有这类 probe 的上限，都必须至少先能靠近这组 incumbent 对照。

## 3. 当前 winner 的模型架构与训练参数

### 3.1 基座模型

当前 winner 的 base model 是 `google/byt5-small`。

通过本地 `transformers.AutoConfig.from_pretrained("google/byt5-small")` 读取到的关键结构参数为：

- `model_type = t5`
- `d_model = 1472`
- `d_ff = 3584`
- `num_layers = 12`
- `num_decoder_layers = 4`
- `num_heads = 6`
- `d_kv = 64`
- `vocab_size = 384`
- `dropout_rate = 0.1`
- `feed_forward_proj = gated-gelu`
- `tie_word_embeddings = True`

这说明当前 winner 不是词级 subword 模型，而是 byte-level T5 家族模型；对 Akkadian transliteration 这种字符形态变化多、符号密度高、分词不稳定的任务，这类模型往往比标准 subword MT 更稳。

### 3.2 LoRA 形态

当前 winner 配置文件 `runs/STEER/generated_configs/continue_s4_bs24_len640_seg5.yaml` 显示：

- LoRA target modules:
  - `q_proj`
  - `v_proj`
- 实际在 T5 模块名上解析后，训练脚本会落到：
  - `q`
  - `v`
- LoRA 超参数：
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.0`
  - `bias = none`

`run_summary.json` 显示参数量：

- `total_params = 301,362,176`
- `trainable_params = 593,920`
- `trainable_ratio_pct = 0.1971%`

也就是说，这条 winner 不是全参训练，而是非常轻量的 LoRA continue。

### 3.3 当前 winner continue 阶段训练参数

`continue_s4_bs24_len640_seg5.yaml`：

- `max_source_length = 640`
- `max_target_length = 640`
- `per_device_train_batch_size = 24`
- `per_device_eval_batch_size = 48`
- `gradient_accumulation_steps = 1`
- `learning_rate = 5e-5`
- `warmup_ratio = 0.03`
- `weight_decay = 0.0`
- `num_train_epochs = 8`
- `lr_scheduler_type = cosine`
- `bf16 = true`
- `fp16 = false`
- `gradient_checkpointing = true`
- `eval_fraction_of_epoch = 0.5`
- `predict_with_generate = false`
- `metric_for_best_model = eval_loss`

运行摘要：

- `train_rows = 4064`
- `val_rows = 1225`
- `eval_steps = 5`
- `train_loss = 0.9902`
- `eval_loss = 0.8547`
- `global_step = 300`
- `peak_gpu_memory_allocated_mb = 4272.48`

### 3.4 winner 不是凭空出现的，而是三段链路产物

当前主线是一个三段 continue 链：

1. Stage1
   - config: `configs/cloud_stage1_len512_lr2e4.yaml`
   - processed_dir: `data/processed_byt5_chunks`
   - base model: `google/byt5-small`
   - LoRA: `q/v, r=8, alpha=16`
   - `max_source_length = 512`
   - `max_target_length = 512`
   - `lr = 2e-4`
   - `bs = 16`
   - `grad_accum = 2`
   - `epochs = 30`
   - `train_rows = 2821`
   - `val_rows = 820`
   - `eval_loss = 0.8658`

2. Stage2
   - config: `configs/cloud_stage2_gc_curriculum_cost14_from_s1win.yaml`
   - init adapter: `runs/CLOUD_STAGE1_LEN512_LR2E4_fold0/best_model`
   - processed_dir: `data/processed_byt5_chunks_align_gc_cost14`
   - `max_source_length = 512`
   - `max_target_length = 512`
   - `lr = 1e-4`
   - `bs = 16`
   - `grad_accum = 2`
   - `epochs = 8`
   - `train_rows = 4064`
   - `val_rows = 900`
   - `eval_loss = 0.8584`

3. Continue winner stage
   - init adapter: `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/best_model`
   - processed_dir 保持 `data/processed_byt5_chunks_align_gc_cost14`
   - 主要变化是：
     - 长度预算从 `512 -> 640`
     - train batch 从 `16x2` 改为 `24x1`
     - learning rate 继续降到 `5e-5`
   - `train_rows = 4064`
   - `val_rows = 1225`
   - `eval_loss = 0.8547`

这条链路本身已经说明当前 winner 的真正优势不是某个单点 tweak，而是：

- chunk 化
- 再加 Gale-Church 风格的 short aligned augmentation
- 再加一段低学习率 continue

## 4. 我们当前的整体训练策略是什么

### 4.1 原始数据到 `train_proc/test_proc/folds` 的预处理

入口脚本是 `scripts/preprocess.py`。

它做的事情依次是：

1. 读取 `train_csv` 和 `test_csv`
2. 检查字段
   - train 至少要求：
     - `oare_id`
     - `transliteration`
     - `translation`
   - test 至少要求：
     - `id`
     - `transliteration`
3. 生成标准列：
   - `source`
   - `target`
   - 并保留原列
4. 按配置决定是否做：
   - `apply_t0_normalize`
   - `strip_text`
   - `fold_inline_whitespace`
   - `lowercase_source`
   - `lowercase_target`
5. 给 source 加任务前缀：
   - 当前 winner 用的是 `translate Akkadian to English:`
6. 生成 `folds.csv`
   - 优先尝试 group-aware split
   - 否则退回 `KFold`
7. 可选统计 token 长度分布

### 4.2 当前 winner 实际启用的清洗/规范化

winner 配置里：

- `apply_t0_normalize = false`
- `strip_text = true`
- `fold_inline_whitespace = true`
- `lowercase_source = false`
- `lowercase_target = false`
- `task_prefix = "translate Akkadian to English:"`

这意味着当前正式 winner 并没有启用完整 T0 规则链，只启用了：

- 换行标准化
- 行内空白折叠
- 首尾 strip
- 任务前缀

### 4.3 仓库里定义了哪些规范化规则

`cleaning/configs/cleaning.t0.yaml` 定义了 T0 规则：

- `t0_unicode_nfc = true`
- `t0_remove_invisible = true`
- `t0_whitespace_normalize = true`
- `t0_unknown_visible_char_alert = true`

`cleaning/normalize.py` 中真正的规则执行顺序是：

1. `t0_unicode_nfc`
2. `t0_remove_invisible`
3. `t0_whitespace_normalize`
4. `t0_unknown_visible_char_alert`
5. 可选但默认关闭：
   - `t1_gap_tag_canonicalize`
   - `t1_brace_whitespace_canonicalize`

因此，仓库里有比当前 winner 更强的标准化能力，但 winner 当前并没有把这一整套打开。  
这是一个重要事实，因为之后有些数据清洗/后处理实验，其实是在已经很稳的 winner 之上补规则，而不是从训练入口就重写 source 规范化。

### 4.4 fold 划分纪律

`scripts/preprocess.py` 会根据 group strategy 选择：

- `GroupKFold`
- 或 `KFold`

候选分组来源包括：

- `text_id`
- `publication_catalog`
- `cdli_id`
- `oare_id` 前缀
- source bucket

当前 winner config 用的是：

- `folds = 5`
- `group_strategy = auto`

### 4.5 chunk 化策略

`scripts/build_long_chunks.py` 是当前主线最核心的数据工程之一。

它做的事情是：

1. 读取已经过 preprocess 的 `train_proc.csv / test_proc.csv / folds.csv`
2. 根据 `src_limit / tgt_limit` 计算一条 parent 是否需要切块
3. 优先尝试 delimiter 对齐切块
   - source 和 target 分段数一致时，走 `delimiter_aligned`
4. 否则走 ratio-based 切块
   - 先按 target 分块
   - 再按 target 块长度比例切 source
5. 若任何块仍超预算，就继续二分切开
6. 为每个 chunk 写入：
   - `parent_oare_id`
   - `chunk_index`
   - `chunk_total`
   - `is_chunk`
   - `chunk_mode`

当前快照中最终 winner 用的 processed dir 是：

- `data/processed_byt5_chunks_align_gc_cost14`

这个目录的 `train_proc.csv` 当前统计为：

- `rows = 5289`
- `unique_parent_oare_id = 1561`
- `avg_rows_per_parent = 3.3882`
- `max_rows_per_parent = 18`

chunk/augmentation 构成：

- `chunk_mode = ratio`: `3026`
- `chunk_mode = short_aligned_gale_church`: `1648`
- `chunk_mode = none`: `615`

这说明当前 winner 并不只是“长文本切块”，而是“切块 + 大量短对齐增广”的混合训练集。

### 4.6 Gale-Church 风格 short aligned augmentation

`scripts/build_short_aligned_pairs_galechurch.py` 会：

1. 先把 source 按分隔符和行号模式切成 source segments
2. 把 target 按句子边界切成 target segments
3. 用一个简化的 Gale-Church 风格 DP 对齐
4. 允许转移：
   - `1:1`
   - `1:2`
   - `2:1`
5. 为每一对齐段计算 `align_cost`
6. 只保留满足阈值的 pair
7. 将这些 pair 作为额外训练样本混入原训练集

新增样本会被显式标记为：

- `chunk_mode = short_aligned_gale_church`
- `is_short_aligned = True`
- `short_align_mode = gale_church`
- `align_type = 1:1 / 1:2 / 2:1`
- `align_cost = ...`

在当前 winner processed dir 中：

- short aligned 总行数：`1648`
- 其中：
  - `align_type = 1:1`: `1148`
  - `align_type = 1:2`: `500`

这点非常重要，因为后面的 `A2` 过滤器正是对这些样本施加了过强惩罚，直接砍掉了 winner 很可能依赖的一部分有效监督。

### 4.7 编码、训练、解码到底怎么做

训练脚本是 `scripts/train_mt5_lora.py`。

当前训练链路的实际行为是：

1. 按 fold 读取 processed train / val
2. source 列作为 encoder 输入
3. target 列作为 decoder labels
4. 用 `AutoTokenizer` 对 source 和 target 做 seq2seq tokenization
5. 构造 HF `Seq2SeqTrainer`
6. 模型是：
   - base `AutoModelForSeq2SeqLM`
   - 再包一层 LoRA
7. 启用：
   - `gradient_checkpointing`
   - `bf16`
8. 用 teacher-forcing 的标准 seq2seq 目标训练
9. 如果 `predict_with_generate=false`，训练过程中不做真正 decode 评估
10. 最终模型保存为：
   - `best_model`
   - `run_summary.json`

### 4.8 当前主线真正的“正则化”

如果严格按显式超参数看，当前 winner 的传统 regularization 并不重：

- `lora_dropout = 0.0`
- `weight_decay = 0.0`
- `no_repeat_ngram_size = 0`（训练配置默认生成参数）

真正起主要作用的，不是显式正则，而是下面这些结构性约束：

- byte-level ByT5 编码，减轻分词脆弱性
- chunk 切分，把超长 parent 分解成可控子问题
- Gale-Church short aligned augmentation，补充局部对齐监督
- source 前缀统一任务表述
- `suppress_extra_ids = true`，显式屏蔽 `<extra_id_*>` 之类坏 token
- decode 时再单独做 beam / length penalty 搜索，而不是相信训练期 loss

### 4.9 评估和 decode 策略

`scripts/generation_utils.py` 和 `scripts/metrics_utils.py` 定义了当前 decode/eval 规范。

decode 可控参数包括：

- `num_beams`
- `length_penalty`
- `max_new_tokens`
- `min_new_tokens`
- `no_repeat_ngram_size`
- `suppress_extra_ids`
- `bad_tokens_regex`

当前主线常见设置：

- `suppress_extra_ids = true`
- `bad_tokens_regex = <extra_id_\\d+>`

指标定义：

- `BLEU`
- `chrF++`
- `geom = sqrt(BLEU * chrF++)`

当前仓库里的 `official-like` 仍然只是本地代理层。  
`reports/taskform_phase12/official_metric_probe.json` 直接写明：

- `status = missing_bridge`

也就是说，到这份报告为止，仓库里还没有真正接入官方 metric bridge。

### 4.10 parent reconstruction 的纪律

一个经常被忽视、但极其关键的实现细节在 `train_mt5_lora.py`：

- 做 parent reconstruction 时，会先过滤掉：
  - `is_short_aligned = True`
  - 或 `chunk_mode` 以 `short_aligned` 开头的样本

也就是说：

- short aligned augmentation 只用于训练监督
- 不应被当作 parent reconstruction 的真实 chunk 参与验证重建

这也是为什么后面出现“整份 processed_dir 被重写”的实验会污染比较口径：  
一旦训练集和 fold 结构被整体重写，就不再是对同一组 val parent 集合的干净对比。

## 5. 当前 winner 为什么强

从现有证据看，当前 winner 强在四点：

1. 它把超长 parent 问题拆回了 chunk 级子问题，避免了 parent-level rewrite 模型常见的长度崩坏。
2. 它的 base family 选得对：`ByT5-small` 比后面试过的 `ByT5-base / mT5-base / flan 风格 parent fusion` 更稳。
3. 它在训练集层面吃到了 `chunk + short aligned` 这类真正改变监督信号的增益，而不是只在 decode 端做小修小补。
4. 它的正式 checkpoint 和 decode 最终是按 decode 结果冻结，而不是盲信 training loss。

简化地说，winner 赢在“任务分解”和“输出可控”，而不是赢在参数量或者 fancy 后处理。

## 6. 我们近期到底尝试了哪些路径

下面按时间与逻辑顺序复盘。

### 6.1 架构探测：ByT5-base / mT5-base

证据：

- `logs/architecture_probe_round_20260309_summary.json`

对照基线：

- matched baseline `geom = 18.3354`

尝试了三条架构线：

1. `P1: ByT5-base len640 q/v`
   - best `geom = 3.5886`
   - `stage_decision = reject`
2. `P2: ByT5-base len640 qkvo r16`
   - best `geom = 3.7618`
   - `stage_decision = reject`
3. `P3: mT5-base len640 q/v`
   - best `geom = 0.1437`
   - `stage_decision = reject`

失败原因：

- 这不是“小负”，而是直接命中 probe 硬停边界。
- 说明在当前数据规模、当前 recipe、当前 decode 预算下，换大模型或换家族并没有带来更强可用能力，反而更不稳定。
- 也说明“winner 的成功”不是来自简单的模型更大，而是来自匹配对了数据形态和任务分解。

结论：

- 新架构探测这条线已经足够证伪。
- 纪律文档里将 `ByT5-base / mT5-base` 再起新探测列为暂停项，是合理的。

### 6.2 Taskform Probe Round v2：P1 parent-packed / P2 replay+hardselector

证据：

- `docs/taskform_upgrade_probe_round_v2_2026-03-09.md`
- `runs/TASKFORM_P1_PARENTPACK_PROBE_fold0/diagnostics/*`
- `runs/TASKFORM_P2_REPLAY25_HARDSELECTOR_PROBE_fold0/diagnostics/*`

#### P1：parent-packed / parent-windowed

数据侧：

- 数据目录：`data/processed_taskform_parentpack_fold0`
- `rows_out = 4579`
- `parentpack_2_3 = 845`
- `parentwindow_3ofN = 3379`

结果：

- matched baseline `geom = 9.0096`
- `ckpt150 = 9.2586`
- `ckpt200 = 9.5190`
- `ckpt250 = 8.9377`

健康度：

- `empty = 0.00%`
- `copy = 0.00%`
- `short = 18.75%`
- `unique = 93.75%`
- `pred_tok_p95 = 385`

阶段判断：

- 自己的 probe 口径里，确实赢了 matched baseline
- 但长度与重复健康不够稳，只能进入 `W-lite`

#### P2：replay25 + hardselector

数据侧 hardest bucket 条件：

- `chunk_total >= 4`
- `target_len >= 129`
- source 含 `<gap>` 或 `{` 或 `[`
- val hardest rows 只有 `192`

结果：

- matched baseline `geom = 8.2732`
- `ckpt150 = 8.2602`
- `ckpt200 = 9.2412`
- `ckpt250 = 9.1064`

健康度：

- `empty = 0.00%`
- `copy = 0.00%`
- `short = 6.25%`
- `unique = 96.88%`

阶段判断：

- 这条线在 hardest-bucket probe 上也给出了真信号
- 但它解决的是 hardest bucket 补丁，不是全局主任务替代
- 因而被保留为 reserve，而不是主线

#### 这一轮为什么没有走成最终主线

P1/P2 的共同问题是：

- 它们只在小样本、子口径、matched baseline 下给出有限增益
- 还远没有证明能在 full-val reconstructed 上替代正式 winner

所以这轮的正确解读是：

- 它们不是“完全没信号”
- 但它们也不是“已经证明可晋升”

### 6.3 P1 W-lite：parentpack 进入更严格 gate 后失败

证据：

- `docs/taskform_p1_wlite_gate_2026-03-09.md`
- `logs/taskform_p1_wlite_gate_20260309_summary.json`

结果：

- matched baseline64 `geom = 7.6886`
- winner checkpoint `checkpoint-300`
- winner anchor64 `geom = 7.5264`
- `delta = -0.1622`

健康度：

- `empty = 0.00%`
- `copy = 0.00%`
- `short = 15.625%`
- `unique = 95.3125%`
- `pred_tok_p95 = 385`

门槛：

- `min_delta_geom = 0.4`
- `max_short = 15.0`

失败原因：

- 分数从小正增益变成小负
- `short` 达到 `15.625%`，越过 gate
- 说明 P1 的“结构上似乎有帮助”并没有稳到足以进入 promote

结论：

- P1 是“probe 有信号，W-lite 失败”的典型。
- 它说明 parent-packed 不是纯噪声，但也说明该 formulation 不够稳。

### 6.4 dan-1 A1/A3：concat / dedup / oracle / edit smoke

证据：

- `reports/taskform_dan1_a1_a3/summary.json`
- `docs/next-step_taskform_discipline_2026-03-09.md`

结果：

- matched baseline `geom = 13.2798`
- `concat_pred_geom = 11.4449`
- `dedup_concat_pred_geom = 9.1836`
- `concat_oracle_geom = 78.5647`
- `edit_smoke_geom = 6.3961`

长度健康：

- `concat_pred_short = 4.6875`
- `dedup_concat_pred_short = 42.1875`
- `edit_smoke_short = 28.125`

这组结果非常有信息量：

1. `concat_oracle_geom = 78.5647`
   - 说明这条线的“理论上限”非常高
   - 也就是说，Pass-A chunk drafts 里并不是没有信息
2. 但实际无训练 concat、去重 concat、以及 learned edit 全都没把这个潜力转成可用分数
3. `edit_smoke` 比 matched baseline 直接低了 `-6.8837`

失败原因：

- 不是“信息不够”，而是任务 formulation 错了。
- learned edit 模型没有学会“保留已有 draft 信息并只做局部修复”，而是把 parent-level 输出控制做坏了。
- `dedup_concat` 和 `edit_smoke` 的短句率大幅恶化，说明模型/规则在去重时误伤了真实正文。

结论：

- `dan-1` 线不是没有 ceiling，而是当前定义成了错误问题。
- 在现有 formulation 下，这条线已经不值得继续当主线。

### 6.5 dan-1 P：真正训练的 routed probe 直接失败

证据：

- `reports/taskform_dan1_p/summary.json`
- `reports/taskform_dan1_p/gate_report.md`

结果：

- matched `geom = 13.2798`
- winner `geom = 4.5109`
- `delta = -8.7689`

健康恶化：

- `short delta = +21.875`

失败原因：

- 这是一次非常彻底的失败，不是边界噪声。
- parent-level draft fusion/edit 模型在 routed hardest parents 上没有学会“整合 chunk 信息”，反而大幅增加了短输出。
- 这说明当前 `Pass-B` parent rewrite/edit 训练目标不成立，至少不适合当时那套 prompt、预算和训练定义。

结论：

- `dan-1` 训练线直接 `reject_stop` 是正确的。

### 6.6 dan-1 B1/B2/B4：cleanup / rerank 也没打过 Pass-A

证据：

- `reports/taskform_dan1_b1_b2_b4/summary.json`
- `reports/taskform_dan1_b1_b2_b4/compare.md`
- `reports/taskform_dan1_b1_b2_b4/bucket_audit.md`

anchor64：

- `raw_concat = 11.5044`
- `looptrim_concat = 12.5232`
- `looptrim_chunkdedup_concat = 12.3262`
- `pass_a_prediction = 13.2798`

routed_full：

- `raw_concat = 11.7911`
- `looptrim_concat = 13.1063`
- `looptrim_chunkdedup_concat = 12.9053`
- `rerank_prediction = 12.9132`
- `pass_a_prediction = 13.7161`

一个表面上“好看”的数字是：

- `official_mix_rerank_geom = 14.1843`

但它不能当成主线成功，原因是：

- deployable 口径下，纯 routed hardest 替换并没有稳定打过 `pass_a_prediction`
- cleanup 和 rerank 更像“混合场景里有限止损”，不是可独立晋升的 Pass-B

bucket audit 进一步说明了问题：

- `chunk7plus` 上 `rerank_prediction` 只有 `7.5238`
- 远低于 `chunk4_6` 和 `chunk2_3_long_or_tag`

失败原因：

- 这类 concat cleanup 只能在某些桶里局部修复重复，但最难的超长 bucket 仍然控制不住。
- 也就是说，后处理没有把 hardest case 的根因解决掉。

### 6.7 L2：术语词表后处理

证据：

- `reports/taskform_l2_term_phase12/gate_report.md`
- `reports/taskform_phase12/phase12_l2_l3_summary.json`
- `reports/taskform_phase12/phase12_l2_l3_tight_summary.json`

#### loose 版

- 词表项数：`57`
- holdout local:
  - baseline `13.4444`
  - patched `13.3485`
- fullval local:
  - baseline `14.3323`
  - patched `14.2150`
- fullval hard:
  - baseline `13.7161`
  - patched `13.6312`
- fullval changed rows：`249`

结论：

- 改动很多行，但总体掉分。

#### tight 版

- 词表项数：`10`
- `allowed_term_types = ["measure"]`
- fullval local:
  - baseline `14.332266680617607`
  - patched `14.332266680617607`
- holdout changed rows：`0`
- fullval changed rows：`0`

结论：

- 一收紧到安全范围，几乎就不改任何东西了。

失败原因：

- loose 版说明词表修正会误伤正常生成。
- tight 版说明只靠安全词表，几乎拿不到足够覆盖率。

这条线最终只证明了一件事：

- `L2` 可以当止损工具研究
- 但不能当主线提分方案

### 6.8 L3：source-only policy / row-gate

这条线分三次看比较清楚。

#### 第一版 source-policy

从 `reports/taskform_phase12/phase12_l2_l3_summary.json` 看：

- holdout pass_a `13.3577`
- holdout policy `13.0298`

结论：

- 第一版直接在 holdout 上掉分。

#### v2 row-gate

证据：

- `reports/taskform_l3_rowgate_phase12_v2/gate_report.md`
- `reports/taskform_l3_rowgate_phase12_v2/summary.json`
- `reports/taskform_l3_rowgate_phase12_v2/random_baseline_summary.json`

最佳 preset：

- `looptrim_only`
- `min_length_ratio = 0.8`
- `min_repeat_gain = 1`
- `min_score_threshold = 2.0`
- `max_switch_rows = 8`

holdout：

- pass_a `13.3577`
- row-gate `13.4387`
- random `p95 = 13.4401`

这很关键：

- 真实策略虽然比 pass_a 高一点
- 但没有高过随机 95 分位

这意味着：

- 看起来像增益
- 但在统计上不够强，不能宣称有效

#### tight 版

从 `reports/taskform_phase12/phase12_l2_l3_tight_summary.json`：

- holdout pass_a `13.3577`
- holdout policy `13.4186`
- random `p95 = 13.4186`

也就是说，tight 版正好等于随机最好值。

失败原因：

- `L3` 的问题不是一点用都没有，而是强度太弱。
- 一旦补随机化对照，它就无法证明自己真比随机切换强。

结论：

- `L3` 作为后处理主线已经足够证伪。

### 6.9 A2：数据质量过滤 / curriculum

证据：

- `reports/taskform_a2_a1_20260310/summary.json`
- `reports/taskform_a2_a1_20260310/gate_report.md`
- `scripts/taskform_a2_a1_flow.py`

#### A2 到底做了什么

`scripts/taskform_a2_a1_flow.py` 的 `_noise_score_row` 明确对以下条件加惩罚：

- `is_short_aligned` 直接 `+1.50`
- `align_cost / 0.40`，最多加到 `+2.00`
- `align_type == 1:2` 再 `+0.75`
- `align_type == 2:1` 再 `+0.25`
- target/source 重复 proxy
- 长度比异常
- `<gap>` 多
- `chunk_total >= 8`

而在 `_build_a2_variants` 里，它不是“只对 train fold 过滤”，而是：

- 直接在整个 `train_proc.csv` 上按 `noise_score` 排序
- 再把保留下来的 `oare_id` 同步写回新的 `train_proc.csv` 和 `folds.csv`

这带来了两个问题。

第一个问题是方向错了：

- `keep100` 总行数：`5289`
- `keep97` 总行数：`5130`
  - removed `159`
  - short aligned 占比从 `31.16% -> 29.03%`
  - `1:2` 行从 `500 -> 391`
- `keep94` 总行数：`4972`
  - removed `317`
  - short aligned 占比从 `31.16% -> 26.77%`
  - `1:2` 行从 `500 -> 295`

这说明过滤器系统性打击了：

- short aligned
- `1:2`
- 高 `align_cost`

而这些正是当前 winner 家族很可能依赖的有效监督。

第二个问题是比较口径被污染：

- 过滤是重写整份 processed train/folds
- 而不是“只动 train fold，不动 fold0 val 对应 parent 集”

所以这轮不仅分差，还不够干净。

#### A2 的实际结果

gate report：

- incumbent anchor64 `16.5057`
- `keep97 = 12.2241`
- `keep94 = 12.2050`
- `status = reject_stop`

失败原因：

1. 过滤器把对 winner 有用的监督砍掉了。
2. 过滤粒度不干净，比较口径被污染。

结论：

- 当前 A2 方向可以判失败。
- 失败不代表“所有数据过滤都错”，而是这版 proxy 设计和执行方式都不成立。

### 6.10 A1 旧版 TAPT smoke

证据：

- `runs/TASKFORM_A1_TAPT_SMOKE_20260310/tapt_summary.json`
- `reports/taskform_a2_a1_20260310/summary.json`

旧版 A1 只做了 TAPT smoke，没有做 matched supervised 对照。

结果：

- `raw_text_rows = 6000`
- `train_rows = 5880`
- `eval_rows = 120`
- `mask_ratio = 0.15`
- `max_span_length = 3`
- `max_source_length = 640`
- `max_target_length = 384`
- `train_loss = 15.7933`
- `eval_loss = 2.3605`

为什么不能用它判 TAPT 生死：

1. 它看的只是去噪预训练 `eval_loss`
2. 它没有和 no-TAPT matched supervised 做 C0/T0 因果对照
3. 这轮还是和受污染的 A2 流程绑在一起的

所以旧版 A1 只能说明：

- TAPT 能跑

不能说明：

- TAPT 最终对翻译分有净增益

### 6.11 fair TAPT 三臂对照

证据：

- `reports/taskform_tapt_fair_20260310/manifest.json`
- `reports/taskform_tapt_fair_20260310/summary.json`

这是当前对 TAPT 最公平、也最有结论力的一轮。

#### 先看公平性检查

manifest 显示：

- `fold0_val_parent_rows = 313`
- `fold0_val_chunk_rows_from_processed = 1225`
- `val_overlap_rows removed = 69`
- `test_rows removed = 4`
- `published_duplicate_rows removed = 274`
- checks:
  - `fold0_val_row_count_matches_base = true`
  - `fold0_val_chunk_rows_unchanged = true`
  - `fair_offline_exact_overlap_with_val_unique_sources = 0`
  - `fair_offline_exact_overlap_with_test_unique_sources = 0`

这说明 fair 版本真正遵守了：

- val 不变
- test 不混入
- mono 语料去掉与 val/test 的 exact overlap

#### TAPT smoke

两套轻量 mono：

1. `trainfold_source_only`
2. `trainfold_plus_published_nooverlap`

被选中的 TAPT smoke 是：

- `trainfold_plus_published_nooverlap`
- `raw_text_rows = 8857`
- `train_rows = 8680`
- `eval_rows = 177`
- `eval_loss = 2.7963`

#### matched supervised 对照

三臂定义：

- `I0 = incumbent`
- `C0 = no-TAPT matched control`
- `T0 = TAPT -> matched supervised`

结果：

- `I0 geom = 16.5057`
- `C0 geom = 3.0869`
- `T0 geom = 2.5434`
- `delta(T0 - C0) = -0.5435`
- `delta(T0 - I0) = -13.9623`

输出健康变化：

- `pred_shorter_than_half_ref_ratio_pct` 相比 C0 反而略好 `-1.5625`
- 但 `unique_prediction_ratio_pct` 下降了 `-10.9375`
- `no_regression = false`
- `status = review_stop`

失败原因：

- 在当前 matched recipe 下，TAPT 没有给 C0 带来净增益，反而让输出多样性明显变差。
- 它不仅没有追近 incumbent，甚至连 no-TAPT control 都没打过。

结论：

- 到 `2026-03-10` 这版 fair 对照为止，当前 formulation 下的 `TAPT -> supervised` 可以判负。
- 这里要强调“当前 formulation 下”五个字：
  - 这不等于所有单语方法永久死亡
  - 但等于这条具体主线现在不该再继续烧卡

### 6.12 继续训 incumbent：`steer_incumbent_promote`

证据：

- `logs/steer_incumbent_promote_decode.log`
- `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/diagnostics/decode_grid_best_steer_incumbent_promote.json`
- `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/diagnostics/val_diagnostic_summary_steer_incumbent_promote.json`

decode grid 实际试了四组：

- `beam=4 lp=0.8`
- `beam=4 lp=1.0`
- `beam=6 lp=0.8`
- `beam=6 lp=1.0`

最佳是：

- `beam=6`
- `lp=0.8`
- `max_new_tokens=384`
- full-val reconstructed `geom = 13.8738`

和正式 winner 比：

- current winner full-val reconstructed `geom = 14.3323`
- promote best `geom = 13.8738`
- 仍然落后

更关键的是 chunk-level output health：

- `unique_prediction_ratio_pct = 64.4082`
- `empty_prediction_ratio_pct = 0.0816`
- `pred_shorter_than_half_ref_ratio_pct = 6.6122`

top repeated predictions 里充满明显循环：

- “If you have not paid the silver ...”
- “Seal of the silver and the silver ...”
- “miss simmiss simmiss ...”

失败原因：

- 这是“训练 loss 健康，但 decode 行为失控”的典型案例。
- 模型不是完全不会翻，而是陷入了高重复、模板化、模式坍缩。
- 从 metrics 看，它在 parent reconstructed 上输了；从 health 看，它在 chunk level 上已经明显坏了。

结论：

- 继续训 incumbent 这条 promote 线不能靠“再训一会儿”解决。
- 这轮非常强地支持一个更大的判断：
  - 当前优化目标与最终 `geom` 并不完全对齐。

### 6.13 2026-03-11 winner replay / curriculum probe

证据：

- `reports/taskform_winner_replay_probe_20260311/summary.json`
- `reports/taskform_winner_replay_probe_20260311/gate_report.md`
- `docs/taskform_winner_replay_probe_2026-03-11.md`

这是把旧 `P2 replay hardselector` 思路，按 current winner 的 matched continue 口径重新跑了一次。

三臂定义：

- `ctrl = current winner matched continue control`
- `replay25 = replay 25%`
- `replay40 = replay 40%`

结果：

- `ctrl anchor64 / hard = 15.3962 / 14.5055`
- `replay25 anchor64 / hard = 15.9949 / 15.9245`
- `replay25 - ctrl anchor64 / hard = +0.5987 / +1.4190`
- `replay40 anchor64 / hard = 15.5201 / 15.0942`
- `replay40 - ctrl anchor64 / hard = +0.1239 / +0.5887`

健康度：

- `replay25 health no_regression = true`
- `replay25 reconstructed health no_regression = true`
- `replay40 health no_regression = false`

和全局强基线相比：

- `replay25 - incumbent anchor64 = -0.5108`
- `replay25 - frozen retrieval anchor64 = -7.5466`

阶段判断：

- 这次 `replay25` 不再只是 hardest bucket 局部信号，而是在 current winner 的 matched control 下给出了干净正增益。
- 但它还没有强到可以直接改写“当前最强单模型是谁”这个答案。
- `replay40` 说明这条轴不是“越重越好”，有用的区间更像温和 replay，而不是 aggressive replay。

结论：

- `replay25` 可以判为当前仍活着的正交候选轴。
- 它值得进入一条激进的 `candidate_pool_long_train`，用真实长训练去摸清 `full-val / hard / raw / reconstructed` 的最终落点。
- 但在跑出这些实测结果前，它不能被写成：
  - 新 winner
  - `F/promote`
  - `fallback_180` 的替代者

## 7. 为什么这些路径最后都败给了旧 winner

### 7.1 第一类失败：换了问题定义，但没保住输出控制

代表实验：

- `dan-1` edit / fusion
- parentpack W-lite

共性：

- 把 chunk 任务抬回 parent-level 整体生成或编辑
- 理论上更直接，但实际更难控长文本输出

常见后果：

- 短输出升高
- 重复拖尾
- 容易把真实正文误当作 boilerplate 去掉

旧 winner 为什么赢：

- 它把问题拆成 chunk 级，降低了每次生成的难度
- 最后再做 parent reconstruction，而不是直接要求模型一口气写完整个 parent

### 7.2 第二类失败：训练信号没有变强，只是在 winner 输出上做 patch

代表实验：

- `L2`
- `L3`
- concat cleanup / rerank

共性：

- 不改变底层模型能力
- 只在输出端修词、删重复、切换候选

为什么最后打不过 winner：

- 这些方法最多能做局部止损
- 但无法系统性提升 hardest rows 的建模能力
- 一旦加上 holdout 和 random baseline，对照强度不够就会暴露

### 7.3 第三类失败：损失函数和最终指标错位

代表实验：

- `steer_incumbent_promote`
- winner 自己训练时 `best_model_checkpoint=50` 但冻结 winner 是 `checkpoint-250`

共性：

- 训练侧看的是 `eval_loss`
- 真正 deploy 看的是 decode 后的 `BLEU / chrF++ / geom`

为什么败给旧 winner：

- 旧 winner 已经是“decode 后验证过”的 checkpoint
- 新实验很多还停留在“训练 loss 看着还行”的层面

### 7.4 第四类失败：把有效监督当噪声砍掉了

代表实验：

- `A2` filtering

共性：

- 过度惩罚 `short_aligned`
- 过度惩罚 `1:2`
- 过度惩罚 `align_cost`

为什么败给旧 winner：

- 旧 winner 恰恰建立在 chunk + short aligned augmentation 上
- 过滤器切掉的是它赖以成功的一部分监督

### 7.5 第五类失败：比较口径不干净

代表实验：

- 旧 `A2/A1`
- 旧 TAPT smoke

共性：

- 把 train/val/fold 结构一起改了
- 或只看去噪 loss，不看 matched supervised translate score

为什么败给旧 winner：

- 不是旧 winner 特别强，而是新实验连“公平比”都没建立起来

### 7.6 第六类失败：subset 有信号，不等于能晋升正式 winner

代表实验：

- `P1 parentpack probe`
- `P2 replay hardselector`

共性：

- 在 anchor32 / hardest bucket / small matched probe 上能给真信号
- 但一旦进入更严 gate，就不再稳

为什么败给旧 winner：

- 正式 winner 是 full-val reconstructed 的稳定方案
- subset 改进如果不能扩展到 full-val，就只能算 reserve，不算替代

## 8. 到今天为止，哪些结论可以硬判，哪些不能

### 8.1 可以硬判失败的

- `ByT5-base / mT5-base` 架构探测
- 当前 `dan-1` parent rewrite / edit formulation
- `P1 parentpack` 作为晋升线
- `L2` 词表后处理作为主线
- `L3` row-gate 作为主线
- 当前 `A2` 过滤器
- 当前 fair 对照下的 `TAPT -> supervised`

### 8.2 只能判“当前 formulation 失败”，不能外推过头的

- 所有单语方法永久无效
- 所有数据过滤永久无效
- 所有 parent-level 方法永久无效

更精确的说法应当是：

- 当前这些具体实现失败了
- 它们失败的方式已经足够清楚
- 但不能把“一个坏 formulation 的失败”直接写成“研究方向永死”

### 8.3 当前真正还活着的价值判断

从纪律文档和现有证据看，后续更值得投入的方向，是那些真正可能改变训练信号或候选质量的方向：

- competition-only 主模型增强
- 更严格但公平的数据资产构造
- replay / curriculum 这类能在 matched control 下给出真增益的训练信号重排
- efficient MBR
- retrieval / kNN-MT
- OOF / ensemble

而不是继续在已经被证伪的 `L2/L3/dan-1/current-A2/current-TAPT-formulation` 上做微修。

### 8.4 当前仍可接受的激进行动

如果要激进，不是完全没有空间，但必须激进在对的地方。

当前唯一有证据支持的激进行动是：

- 对 `replay25` 直接开一条 `candidate_pool_long_train`

理由：

- 它已经对 matched control 给出明确正增益
- `hard` 也同步为正
- health 没坏
- 它和当前 retrieval frozen candidate 代表的是不同轴，不是同质小修

同时必须保留边界：

- 可以跳过当轮 `W-lite`
- 不能把这条长训练写成 `F/promote`
- 不能因为它在 local probe 为正，就宣称它已经替代 `fallback_180`
- 最合理的配套动作，是在它长训练的同时，用更便宜的 `P/probe` 去继续找别的正交候选轴

### 8.5 已批准的后台执行队列（2026-03-11）

当前已批准并应落成后台 `tmux` 队列的，不是“把所有东西同时抢卡跑”，而是：

- 主序保持不动：
  - `raw retrieval W-lite long train`
  - `replay25 candidate-pool long train`
- 主序之后接一条 `probe-only` 小实验队列
- 小实验之后再接一条 `post-probe full-val decode` 队列

写死的执行纪律：

- 小实验只做 `probe`
- `A3` 只做便宜审计
- `RK_true_hook` 只做最小 formal smoke revisit
- `post-probe decode` 最多只吃 `top 1-2` 个赢家
- 不把所有小实验都送去 `full-val`

当前批准的小实验臂：

- `retrieval-top1 + replay25 combo 180-step probe`
  - 作为 retrieval 主胜轴与 replay 次胜轴的真正组合检查
- `replay15 / replay20 / replay30` 窄扫
  - 只看 `25%` 邻域，不再回到 `40%+`
- `A3 cheap revisit`
  - 只看 overlap / bottom25 / oracle ceiling
- `RK_true_hook weak revisit smoke`
  - 只因“更强 retrieval state”这一条件，允许做一次低优先级重开

配套桥接：

- `combo probe` 接 `raw retrieval longtrain`
- `replay band probe` 接 `replay25 longtrain`
- `A3` 只在两个 probe 都落盘后做候选池审计
- `post-probe decode` 再只挑最值得花预算的 `1-2` 个赢家

## 9. 为什么会让人感觉“项目没有信心了”

如果只看最近几天，会很容易产生这个感觉，因为：

- 很多实验确实输了
- 而且不是小输，是明确输
- 还有几条线因为口径污染，浪费了判断机会

但更冷静的结论是：

- 不是“整个项目没有任何有效资产”
- 而是“旧 winner 是真实资产，而最近很多新线不是”

真正让人疲惫的不是没有 winner，而是：

- winner 明明在
- 但很多后续尝试没有建立在它真正成功的原因上

winner 的成功原因是：

- 任务拆解正确
- base family 匹配数据
- 训练信号真实增强
- decode 端做了最终验证

而最近很多失败线做的是：

- 把问题重新定义得更难
- 或只在输出端补丁
- 或把有用监督当噪声删掉

所以这不是“项目已经没路”，而是“最近这批路线大多不值得再信”。

## 10. 最终总结

截至 `2026-03-10` 当前仓库快照，可以把 taskform 主线复盘成一句话：

> 我们已经证明，当前正式 winner 之所以还是 winner，不是靠偶然，而是因为它同时满足了数据形态、任务分解、训练信号和 decode 稳定性；最近绝大多数新实验要么没有比它更强的监督，要么把输出控制做坏了，要么比较口径本身不干净，因此最终都败给了它。

如果要把这份报告转成执行纪律，最重要的不是“再多跑一点”，而是：

1. 不再把已经硬证伪的线继续当主线。
2. 不再把训练 loss 当成 deploy 成功代理。
3. 任何新线都必须先建立公平对照口径。
4. 只有真正改变训练信号或候选质量的方案，才值得抢主线时间。
