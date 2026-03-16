# 公开模型对标 R1 计划
## raw-row + supervised external mix，2026-03-13

本稿不是继续解释上一轮 `P1`，而是把下一条真正对标公开模型本地 `39.10` 的主线一次写清。

本稿承接：

- [public_model_fast_repro_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_fast_repro_plan_2026-03-13.md)
- [public_model_phase1_decode_ablation_2026-03-13.md](/workspace/deep-past-/docs/public_model_phase1_decode_ablation_2026-03-13.md)
- [public_model_repro_design_2026-03-13.md](/workspace/deep-past-/docs/public_model_repro_design_2026-03-13.md)

## 0. 结论先行

上一轮 `PUBLIC_MODEL_REPRO_P1_BYT5_BASE_LEN640_QV_PLAINGC_20260313` 不是公开高分路线复现线，而是：

- 去掉 retrieval hint 污染后的 clean control
- 旧 `chunk + GC + reconstructed` 任务形式上的 `byt5-base + LoRA(q,v,r=8)` 短程 probe

它能回答：

- retrieval hint 污染是否会直接学坏输出
- `byt5-base` 在当前仓库训练链里是不是纯负线

它不能回答：

- 我们离公开模型本地 `39.10` 还差多少
- `raw-row` 主口径下的强单模是否能接近公开路线

因此，从这版开始，主线改成：

- `R1 = raw-row official_formula_local`
- `R1 = official supervision + external supervised parallel mix`
- `R1 = byt5-base stronger adapter`
- `R1 = beam=8 / lp=1.0 / max_new_tokens=640`

## 1. Evidence

### 1.1 公开模型的本地 `39.10` 口径是什么

公开模型 `byt5 akkadian mbr` 在本地拿到的：

- `BLEU = 30.8114`
- `chrF++ = 49.6247`
- `geom = 39.1025`

证据：

- [summary_default_beam8_rep11.json](/workspace/incoming/public_eval_byt5_akkadian_mbr/reports/local_eval_byt5_akkadian_mbr/fold0/summary_default_beam8_rep11.json)

这份分数的口径不是仓库旧的 `chunk/reconstructed`，而是：

- `train.csv` 原始行级数据
- `fold0` raw validation
- `313` 行
- corpus BLEU
- corpus chrF++
- `geom = sqrt(BLEU * chrF++)`

证据：

- [public_eval_byt5_akkadian_mbr_local.yaml](/workspace/incoming/public_eval_byt5_akkadian_mbr/configs/public_eval_byt5_akkadian_mbr_local.yaml)
- [preprocess.py](/workspace/incoming/public_eval_byt5_akkadian_mbr/scripts/preprocess.py)
- [metrics_utils.py](/workspace/deep-past-/scripts/metrics_utils.py)

### 1.2 上一轮 `plaingc P1` 为什么不能拿来对标 `39.10`

上一轮 clean `P1` 的真实形态是：

- `google/byt5-base`
- `LoRA(q,v,r=8, alpha=16)`
- `0.19%` trainable params
- `250 steps`
- `processed_dir = data/processed_byt5_chunks_align_gc_cost14`

证据：

- [public_model_repro_p1_byt5_base_len640_qv_plaingc_20260313.yaml](/workspace/deep-past-/configs/public_model_repro_p1_byt5_base_len640_qv_plaingc_20260313.yaml)
- [run_summary.json](/workspace/deep-past-/runs/PUBLIC_MODEL_REPRO_P1_BYT5_BASE_LEN640_QV_PLAINGC_20260313_fold0/run_summary.json)

它的结果是：

- reconstructed eval `geom = 5.9010`
- `anchor64 ckpt250 geom = 6.0504`
- `diag32 geom = 3.7151`
- `diag32 reconstructed geom = 7.1399`

证据：

- [run_summary.json](/workspace/deep-past-/runs/PUBLIC_MODEL_REPRO_P1_BYT5_BASE_LEN640_QV_PLAINGC_20260313_fold0/run_summary.json)
- [ckpt250 anchor64](/workspace/deep-past-/runs/PUBLIC_MODEL_REPRO_P1_BYT5_BASE_LEN640_QV_PLAINGC_20260313_fold0/diagnostics/decode_grid_best_public_model_repro_p1_plaingc_ckpt250_anchor64_beam8_lp10_20260313.json)
- [diag32](/workspace/deep-past-/runs/PUBLIC_MODEL_REPRO_P1_BYT5_BASE_LEN640_QV_PLAINGC_20260313_fold0/diagnostics/val_diagnostic_summary_public_model_repro_p1_plaingc_linewinner_ckpt250_diag32_beam8_lp10_20260313.json)

这说明：

1. 这轮不是公开路线复现，而是 clean control negative
2. 它仍然在旧 `chunk + reconstructed + anchor64` 任务形式里
3. 它的训练容量和训练强度都明显低于公开完整强单模

### 1.3 仓库里已经有可用的 external supervised parallel

当前仓库已有外部平行资产：

- [oracc_parallel.csv](/workspace/deep-past-/data/external/oracc_parallel.csv)

该资产的构建结果：

- `rows_final = 7483`
- `published_sentence_silver = 6321`
- `published_agg_parent = 1162`

证据：

- [taskform_winner_a1_silver_build_20260310/summary.json](/workspace/deep-past-/reports/taskform_winner_a1_silver_build_20260310/summary.json)

Overlap 审计结果：

- `fold0_val_exact_overlap_rows = 0`
- `test_exact_overlap_rows = 0`
- `train_exact_overlap_rows = 0`

证据：

- [overlap_audit.json](/workspace/deep-past-/reports/taskform_winner_a1_20260310/overlap_audit.json)
- [source_registry.csv](/workspace/deep-past-/reports/taskform_winner_a1_20260310/source_registry.csv)

这说明：

- 仓库不是没有扩容数据
- 真正缺的是把它变成 raw-row 主口径下的正式 supervised mix 主线

### 1.4 历史 `published_nooverlap` 负例不等于否定 supervised mix

仓库确实已有 `published_nooverlap` 负例，但那次是：

- `TAPT / mono continue`
- 不是 external parallel supervised mix

证据：

- [taskform_tapt_fair_20260310/summary.json](/workspace/deep-past-/reports/taskform_tapt_fair_20260310/summary.json)
- [taskform_tapt_fair_smoke_trainfold_plus_published_nooverlap.yaml](/workspace/deep-past-/reports/taskform_tapt_fair_20260310/generated_configs/taskform_tapt_fair_smoke_trainfold_plus_published_nooverlap.yaml)

因此不能把那次结论写成：

- `published_nooverlap` 都没用

更准确的写法应是：

- `mono TAPT` 在当前 matched recipe 下为负
- 不能直接外推到 `supervised external parallel mix`

### 1.5 decode 侧已经有明确证据

公开模型本地对照已经证明：

- `beam=8 / lp=1.0 / max_new_tokens=640` 是关键主配置
- `repetition_penalty=1.1` 不是主增益来源
- 我们旧 `beam=4 / lp=0.7 / max384` 会把公开模型本地分从 `39.10` 压到 `29.88`

证据：

- [public_model_phase1_decode_ablation_2026-03-13.md](/workspace/deep-past-/docs/public_model_phase1_decode_ablation_2026-03-13.md)

## 2. R1 的目标定义

`R1` 的目标不是再做一条 `anchor64` 内部 probe，而是：

- 用仓库本地可复现资产，尽可能接近公开模型的 raw-row 本地口径
- 先看 strong single model 能不能站起来
- 再决定是否值得进入更重的复现与 Kaggle 提交链

`R1` 的主目标固定为：

1. raw-row `fold0`
2. `313` 行 official-compatible local eval
3. corpus BLEU
4. corpus chrF++
5. `geom = sqrt(BLEU * chrF++)`

`anchor64 / reconstructed / diag32` 的定位改为：

- 诊断指标
- 不是 R1 主 gate

## 3. 训练集怎么来

### 3.1 监督根

官方监督根固定为：

- [data/raw/train.csv](/workspace/deep-past-/data/raw/train.csv)

基本事实：

- `1561` 行
- 列：`oare_id / transliteration / translation`

### 3.2 外部平行根

外部平行根固定为：

- [data/external/oracc_parallel.csv](/workspace/deep-past-/data/external/oracc_parallel.csv)

不直接用的外部源：

- `published_nooverlap mono`
- 词典定义
- lexicon gloss
- retrieval hint

这些不进 `R1` 主训练集。

### 3.3 val split 怎么固定

`R1` 的验证集必须先固定，再混外部行。

执行原则：

1. 只用官方 `train.csv` 生成 fold
2. fold 逻辑对齐隔离区公开模型本地评测
3. `fold0 val` 固定为原始官方行级 `313` 行
4. external rows 一律 `train-only`

技术边界：

- 不能把 mixed train 整体重新做 5-fold
- 否则 external rows 会进入 val，主口径污染

现有训练脚本行为证据：

- [train_mt5_lora.py](/workspace/deep-past-/scripts/train_mt5_lora.py)
  - `train_split = merged[fold != args.fold]`
  - `val_split = merged[fold == args.fold]`

因此 `R1` 的 `folds.csv` 应按下面写法构造：

- 官方 train 行：`fold in {0..4}`
- external train-only 行：`fold = -1`

这样：

- external 行自动进入 `train_split`
- external 行不会进入 `val_split`

### 3.4 external mix 怎么做

不从零发明，直接建立在现有脚本能力上：

- [prepare_oracc_mix.py](/workspace/deep-past-/scripts/prepare_oracc_mix.py)

该脚本已经具备：

- exact dedupe
- 基于规范化 source 的 overlap 去重
- 4-gram Jaccard similarity filter
- ratio 采样

默认过滤事实：

- similarity threshold = `0.92`
- 和 competition train/test 的规范化 source 去重

`R1` 的 external mix 直接用这套原则，不再引入 retrieval / hint / target-side feature。

## 4. 用什么样的特征工程

### 4.1 R1 base feature engineering

`R1` 的 base feature engineering 只允许最小、公正、可复现的 source-side 处理。

允许：

- 保留原始 transliteration
- `strip_text = true`
- `fold_inline_whitespace = true`
- `lowercase_source = false`
- `lowercase_target = false`
- 固定任务前缀：
  - `translate Akkadian to English:`
- 禁止 `<extra_id_*>`

不允许：

- retrieval hint
- retrieved source
- retrieved English hint
- lexicon definition 注入
- dictionary gloss 注入
- prompt 中加入 target-side 线索
- chunk / short-aligned / parent-packed 任务形式主线化

### 4.2 `Tier-0` 怎么处理

仓库当前能被事实确认的清洗基线是：

- `Gate 0-A / Tier-0`

证据：

- [semantic_cleaning_spec.md](/workspace/deep-past-/docs/semantic_cleaning_spec.md)

但 `R1` 的初始主线不直接把 `Tier-0` 打开成主变量，原因只有一个：

- 公开模型 `39.10` 的本地验证是在 raw-row 最小预处理口径下测得的
- 如果第一枪同时改任务形式、外部数据、清洗强度、训练容量，就无法归因

因此 `R1` 主线的清洗策略写成：

- `R1-main`: minimal raw-literal preprocessing
- `Tier-0`: 第二位 audit ablation，不作为第一枪主变量

## 5. 模型与训练怎么做

### 5.1 backbone

主 backbone 固定为：

- `google/byt5-base`

理由：

- 公开模型属于更强完整 ByT5 主干
- 上一轮 `byt5-base` clean control 已证明它不是纯负线
- 继续回到 `byt5-small` 没有对标价值

### 5.2 adapter 形态

`R1` 主 adapter 直接上 stronger adapter：

- `target_modules = [q_proj, k_proj, v_proj, o_proj]`
- `r = 16`
- `alpha = 32`
- `dropout = 0.05`

仓库前例证据：

- steer 候选里已有 `cold_qkvo_r16_len640`
  - [cloud_stage2_steer.yaml](/workspace/deep-past-/configs/cloud_stage2_steer.yaml)
- 小主干上已有 `q/v/o r16` 历史前例
  - [byt5_small_lora_chunked_stage1_r16_qvo.yaml](/workspace/deep-past-/configs/byt5_small_lora_chunked_stage1_r16_qvo.yaml)
  - [byt5_small_lora_chunked_stage1_r16_qv.yaml](/workspace/deep-past-/configs/byt5_small_lora_chunked_stage1_r16_qv.yaml)

这条线的写法要明确：

- `qkvo r16` 在本仓库是“已有候选前例 + 现在应主线化”
- 不是完全凭空新造

### 5.3 训练超参

`R1` 首轮统一如下：

- `max_source_length = 640`
- `max_target_length = 640`
- `bf16 = true`
- `gradient_checkpointing = true`
- `learning_rate = 1e-4`
- `warmup_ratio = 0.03`
- `weight_decay = 0.0`
- `lr_scheduler_type = cosine`

批次建议：

- primary:
  - `per_device_train_batch_size = 8`
  - `per_device_eval_batch_size = 16`
  - `gradient_accumulation_steps = 2`
- fallback if OOM:
  - `per_device_train_batch_size = 6`
  - `per_device_eval_batch_size = 12`
  - `gradient_accumulation_steps = 3`

### 5.4 训练时长

不再用 `250-step` 读主线。

`R1` 主实验首轮统一为：

- `max_steps = 1200`
- `eval_steps = 100`

理由：

- 官方 train `1561` 行
- 若做 `1.0x external mix`，总监督约 `3122` 行
- 以 `bs8 / ga2` 粗算，每 epoch 约 `195` 步
- `1200` 步约等于 `6.1` epochs

这才是有资格读主分的训练强度。

### 5.5 checkpoint 选择

trainer 内部不再把 `eval_loss best` 当主赢家。

`R1` 的 line winner 选择规则应改成：

1. 跑完 `ckpt200 / 400 / 600 / 800 / 1000 / 1200`
2. 用 raw-row `fold0 313` 主口径做 decode 对照
3. 选 `raw-row geom` 最好的 checkpoint
4. 再跑 `diag32`

## 6. decode 怎么做

`R1` decode 直接固定公开高分路线验证出的 baseline：

- `num_beams = 8`
- `length_penalty = 1.0`
- `max_new_tokens = 640`
- `min_new_tokens = 0`
- `no_repeat_ngram_size = 0`

这不是猜的，是已经被本地公开模型对照验证过的。

证据：

- [public_model_phase1_decode_ablation_2026-03-13.md](/workspace/deep-past-/docs/public_model_phase1_decode_ablation_2026-03-13.md)

关于 `repetition_penalty`：

- 当前不作为首要主变量
- 首轮默认不单独 sweep

原因：

- 公开模型对照里它不是主增益来源

## 7. 评测怎么做

### 7.1 主评测

`R1` 主评测固定为：

- raw-row `fold0`
- `313` 行
- `aggregate-by-parent = off`
- corpus BLEU
- corpus chrF++
- geom

### 7.2 健康评测

`R1` 的健康评测固定为：

- `diag32`
- `aggregate-by-parent = off`

重点看：

- empty
- copy-source
- shorter-than-half-ref
- unique ratio
- top repeated predictions

### 7.3 旧仓库诊断保留但降级

以下指标保留，但降级为辅助：

- `anchor64`
- `reconstructed`
- `diag32 reconstructed`

它们的作用是：

- 看这条线和仓库旧工作流相比是否健康
- 不是拿来直接对标公开 `39.10`

## 8. R1 实验矩阵

### 8.1 必跑

1. `R1-C0 official-only`
   - 只用官方 `1561`
   - raw-row 主口径
   - 作用：给 external mix 提供真正可比的 clean control

2. `R1-M050`
   - 官方 `1561` + external `0.5x`
   - 约 `780` external 行
   - 作用：中等强度扩容

3. `R1-M100`
   - 官方 `1561` + external `1.0x`
   - 约 `1561` external 行
   - 作用：最快接近公开路线的监督规模

### 8.2 不跑

这一轮明确不跑：

- retrieval hint mixed train
- `published_nooverlap mono TAPT`
- lexicon/dictionary target hints
- chunk/GC 任务形式主线
- `byt5-small`
- `q/v r8` 的弱 adapter 再试一版

## 9. 结果判定

`R1` 不再用旧 `anchor64` 口径判通过，而是：

### 9.1 Fail

出现任一条即 fail：

- raw-row `geom` 明显低于 `C0`
- `BLEU` 不涨，只靠 `chrF++` 表面抬分
- `diag32` 出现明显 loop / source echo / template collapse

### 9.2 Pass to next

满足以下才允许进入更重训练或提交准备：

- `M050` 或 `M100` 在 raw-row `geom` 上高于 `C0`
- `BLEU` 也正增益
- `diag32` 没有明显模式塌缩

### 9.3 Ceiling reference

公开模型本地 `39.1025` 仅作为 ceiling reference：

- 它是当前上界
- 不是要求 `R1` 第一天就命中

但若 `R1` 仍停在：

- `geom < 15`

就不应继续自我安慰为“离公开路线只差一点”。

## 10. 产物要求

建议新目录：

- `reports/public_model_r1_rawrow_supervised_mix_20260313/`

至少产出：

- `data_build_audit.json`
- `fold_manifest.csv`
- `train_mix_manifest.json`
- `rawrow_decode_table.csv`
- `rawrow_diag32_summary.json`
- `secondary_anchor64_table.csv`
- `route_decision.md`

## 11. 最短结论

下一条真正应该跑的，不是：

- `plaingc q/v r8 250-step probe`

而是：

- `raw-row official_formula_local`
- `official + external supervised parallel mix`
- `byt5-base + qkvo r16`
- `beam=8 / lp=1.0 / max640`
- `raw-row 313` 作为主 gate

这才是一条能和公开 `39.10` 正面对话的主线。

## 12. 执行排布与估时

### 12.1 总原则

执行顺序不再是“边跑边改”，而是：

1. 先把 `R1` 的数据与主评测口径搭好
2. 先跑 `C0 official-only`
3. 再跑 `M050`
4. 只有 `M050` 是正信号，才继续 `M100`

也就是说：

- 不会一开始同时开 `C0 / M050 / M100`
- 不会再让 `anchor64` 充当主 gate
- 不会再让 `250-step smoke` 充当主实验

### 12.2 Step A: R1 数据与评测管线搭建

目标：

- 固定 raw-row `fold0 313` 主口径
- 生成 `official-only / M050 / M100` 三套 train manifest
- 让训练与评测脚本都能读这个新口径

执行内容：

1. 新建 `R1` raw-row fold manifest 生成脚本
2. 新建 `R1` external mix 生成脚本
3. 新建 `R1` raw-row eval 入口
4. 新建 `R1-C0 / R1-M050 / R1-M100` 配置
5. 产出：
   - `fold_manifest.csv`
   - `train_mix_manifest.json`
   - `data_build_audit.json`

估时：

- `45-90 分钟`

### 12.3 Step B: 显存 smoke

目标：

- 验证 `byt5-base + qkvo r16 + len640`
- 在当前 `32GB 5090` 上能否稳定训练

执行内容：

1. 先对 `R1-C0` 跑 `10-20 step` VRAM smoke
2. 首发参数：
   - `bs=8`
   - `eval batch=16`
   - `grad_accum=2`
3. 若不稳，则回退到：
   - `bs=6`
   - `eval batch=12`
   - `grad_accum=3`

估时：

- 首发 smoke：`5-10 分钟`
- 若需一次回退：额外 `5-10 分钟`

### 12.4 Step C: R1-C0 official-only 主实验

目标：

- 先得到真正可比的 clean raw-row control

执行内容：

1. 用官方 `1561` 行 raw-row 训练
2. 训练到：
   - `max_steps = 1200`
   - `eval_steps = 100`
3. 对 `ckpt200/400/600/800/1000/1200` 跑 raw-row `fold0 313` decode
4. 选 raw-row geom best checkpoint
5. 对 line winner 跑：
   - `diag32`
   - secondary `anchor64`

估时：

- 训练：`20-35 分钟`
- raw-row checkpoint decode sweep：`60-100 分钟`
- `diag32 + secondary anchor64`：`15-30 分钟`
- 小结审计：`10-15 分钟`

合计：

- `1小时45分 - 3小时`

### 12.5 Step D: R1-M050 主实验

目标：

- 判断中等强度 external supervised mix 是否带来真实正增益

训练集定义：

- 官方 `1561`
- external 约 `0.5x`
- external 约 `780` 行

执行内容与 `C0` 相同：

1. 训练到 `1200 steps`
2. 跑 raw-row checkpoint sweep
3. 选 raw-row line winner
4. 跑 `diag32 + secondary anchor64`

估时：

- `1小时45分 - 3小时`

### 12.6 Step E: M050 gate

只有满足以下条件，才开 `M100`：

1. raw-row `geom` 高于 `C0`
2. `BLEU` 也高于 `C0`
3. `diag32` 没有明显 loop / mode collapse

如果 `M050` 不满足三条：

- 直接停在 `M050`
- 不再继续烧 `M100`

人工判读估时：

- `10-20 分钟`

### 12.7 Step F: R1-M100 主实验

目标：

- 用更强的 supervised external mix 看能否进一步逼近公开高分路线

训练集定义：

- 官方 `1561`
- external 约 `1.0x`
- external 约 `1561` 行

执行内容与 `C0/M050` 相同。

估时：

- `1小时45分 - 3小时`

### 12.8 整体估时

若只做到 `C0 + M050`：

- Step A: `45-90 分钟`
- Step B: `5-20 分钟`
- Step C: `1小时45分 - 3小时`
- Step D: `1小时45分 - 3小时`
- Step E: `10-20 分钟`

总计：

- `4小时30分 - 7小时30分`

若继续做到 `M100`：

- 再加 `1小时45分 - 3小时`

总计：

- `6小时15分 - 10小时30分`

### 12.9 tmux 排布

建议会话名：

- `r1_rawrow`

窗口安排：

1. `build`
   - 数据构建、manifest、audit
2. `train`
   - 当前主实验训练
3. `watch`
   - `nvidia-smi` + train log
4. `eval`
   - raw-row decode sweep
5. `diag`
   - `diag32 + secondary anchor64`

执行纪律：

- `C0 / M050 / M100` 严格串行
- 只允许 `watch` 并行
- decode 与训练不并行抢 GPU

### 12.10 我会怎么实际推进

实际推进顺序固定为：

1. 先把 `Step A` 落完
2. 立刻跑 `C0 smoke`
3. `C0` full run
4. 读 raw-row 主结果
5. 再决定 `M050`

也就是说，下一步我不会直接开：

- `M050`
- `M100`
- `large`

而是先把 `R1` 的主口径真正落地。
