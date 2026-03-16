# 公开现成权重锚点清单与长训规划
## public weight anchor + long-train plan, 2026-03-13

本稿只做两件事：

1. 明确“现成公开权重”现在有哪些可直接使用的锚点材料
2. 在此基础上定义后续更科学的长训顺序

本稿不启动训练，不恢复旧 `R1/M050/M100`。

## 0. 结论先行

当前更科学的顺序是：

1. 先用 `/workspace/incoming/public_eval_byt5_akkadian_mbr/model` 做锚点
2. 先确认本地推理链、评测链、输出健康度都能接稳
3. 再做基于现成权重的继续训练
4. 最后才回头研究从头复现、recipe 病灶、external mix

原因很简单：

- 当前从头 `byt5-base + LoRA` 线已经证明，会把“链路没接好”和“训练学坏了”混在一起
- 现成权重锚点能先把“推理/评测链是否可信”单独回答掉
- 只有锚点稳定，后面任何长训结果才有参照

## 1. 现成权重清单

### 1.1 权重本体

公开现成权重位置：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/model`

关键文件：

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `added_tokens.json`

模型身份：

- `architectures = T5ForConditionalGeneration`
- `tokenizer_class = ByT5Tokenizer`
- `model_type = t5`
- `tie_word_embeddings = false`

这意味着它可以作为：

- 直接 eval 锚点
- 继续训练起点
- LoRA 增量适配底座

证据：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/model/config.json`
- `/workspace/incoming/public_eval_byt5_akkadian_mbr/model/generation_config.json`

### 1.2 公开本地 eval 配置

公开本地配置：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/configs/public_eval_byt5_akkadian_mbr_local.yaml`

固定口径：

- `task_prefix = "translate Akkadian to English:"`
- `folds = 5`
- `group_strategy = auto`
- `max_source_length = 640`
- `max_target_length = 640`
- `beam = 8`
- `length_penalty = 1.0`
- `repetition_penalty = 1.1`
- `max_new_tokens = 640`
- `suppress_extra_ids = false`
- `bad_tokens_regex = "<extra_id_\\d+>"`

### 1.3 公开本地 processed 数据

公开本地 processed 数据位置：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/data/processed_public_eval_byt5_akkadian_mbr_local`

关键文件：

- `train_proc.csv`
- `folds.csv`
- `length_stats.json`

这套数据的价值不是“训练集本身更神奇”，而是：

- 它定义了公开本地 eval 实际用的 fold
- 它可以作为锚点重跑时的直接输入

### 1.4 公开本地结果锚点

默认主锚点结果：

- 文件：
  - `/workspace/incoming/public_eval_byt5_akkadian_mbr/reports/local_eval_byt5_akkadian_mbr/fold0/summary_default_beam8_rep11.json`
- 分数：
  - `BLEU = 30.8114`
  - `chrF++ = 49.6247`
  - `geom = 39.1025`

辅助对照结果：

- 文件：
  - `/workspace/incoming/public_eval_byt5_akkadian_mbr/reports/local_eval_byt5_akkadian_mbr/fold0/summary_ablate_no_rep_beam8.json`
- 分数：
  - `BLEU = 30.8130`
  - `chrF++ = 49.6426`
  - `geom = 39.1106`

这说明至少对这个公开现成模型而言：

- `rep=1.1` 和 `rep=1.0` 在本地分数上几乎不敏感
- 它并不会像我们当前从头训练线那样 collapse

## 2. 现成权重锚点执行清单

### 2.0 Stage A 是否可以先改成极小 smoke

可以，但只能改成：

- `A0 = 极小 smoke`
- `A1 = 完整锚点`

不能把 `A0` 直接当作 `A1` 的替代。

原因：

- `A0` 适合回答：
  - 权重能不能加载
  - tokenizer / generate / bad token suppression / decode profile 是否接通
  - 输出是否立刻崩成空串、bad token、显著异常
- `A0` 不适合回答：
  - 是否真的复现了 `geom ≈ 39`
  - `313` 全量上的健康度是否稳定
  - 和公开 summary 的偏差是不是只来自抽样噪声

所以我建议把 Stage A 改写成：

1. `A0 smoke`：`max_rows = 32`
2. `A1 full anchor`：全量 `313`

默认执行顺序：

- 先 `A0`
- `A0` 通过再挂后台跑 `A1`

## 2.1 第一条线：incoming 原生线

### 2.1 第一条线：incoming 原生线

目的：

- 不通过我们仓库，直接验证 `/workspace/incoming` 自带 eval 包仍然可复现

`A0 smoke` 命令：

```bash
cd /workspace/incoming/public_eval_byt5_akkadian_mbr
.venv/bin/python scripts/eval_public_full_model_local.py \
  --config configs/public_eval_byt5_akkadian_mbr_local.yaml \
  --model-dir model \
  --fold 0 \
  --max-rows 32 \
  --predict-batch-size 2 \
  --repetition-penalty 1.1 \
  --max-new-tokens 640 \
  --tag default_beam8_rep11_smoke32
```

`A1 full anchor` 命令：

```bash
cd /workspace/incoming/public_eval_byt5_akkadian_mbr
.venv/bin/python scripts/eval_public_full_model_local.py \
  --config configs/public_eval_byt5_akkadian_mbr_local.yaml \
  --model-dir model \
  --fold 0 \
  --predict-batch-size 2 \
  --repetition-penalty 1.1 \
  --max-new-tokens 640 \
  --tag default_beam8_rep11_rerun
```

`A0` 通过标准：

- 正常完成，不报依赖或权重加载错误
- decode 字段与公开 config 一致
- 预测文件正常生成
- 不出现明显 collapse：
  - 大面积空串
  - `<extra_id_*>` 泄漏
  - 明显 bad token

`A1` 通过标准：

- `geom` 与现成 summary 相差不超过约 `0.2`
- decode 字段与公开 config 一致
- 预测文件正常生成

主要看：

- `reports/local_eval_byt5_akkadian_mbr/fold0/summary_default_beam8_rep11_smoke32.json`
- `reports/local_eval_byt5_akkadian_mbr/fold0/summary_default_beam8_rep11_rerun.json`
- `reports/local_eval_byt5_akkadian_mbr/fold0/val_predictions_default_beam8_rep11_rerun.csv`

### 2.2 第二条线：no-rep 对照线

目的：

- 用公开现成模型确认“无重复惩罚”本身不会导致 collapse

`A0 smoke` 命令：

```bash
cd /workspace/incoming/public_eval_byt5_akkadian_mbr
.venv/bin/python scripts/eval_public_full_model_local.py \
  --config configs/public_eval_byt5_akkadian_mbr_local.yaml \
  --model-dir model \
  --fold 0 \
  --max-rows 32 \
  --predict-batch-size 2 \
  --num-beams 8 \
  --length-penalty 1.0 \
  --repetition-penalty 1.0 \
  --max-new-tokens 640 \
  --tag ablate_no_rep_beam8_smoke32
```

`A1 full anchor` 命令：

```bash
cd /workspace/incoming/public_eval_byt5_akkadian_mbr
.venv/bin/python scripts/eval_public_full_model_local.py \
  --config configs/public_eval_byt5_akkadian_mbr_local.yaml \
  --model-dir model \
  --fold 0 \
  --predict-batch-size 2 \
  --num-beams 8 \
  --length-penalty 1.0 \
  --repetition-penalty 1.0 \
  --max-new-tokens 640 \
  --tag ablate_no_rep_beam8_rerun
```

通过标准：

- 分数继续维持在 `geom ≈ 39` 级别
- 输出不出现我们当前从头线那种大规模人名串重复

### 2.3 第三条线：deep-past 兼容线

目的：

- 让我们自己的仓库也能直接吃“完整现成权重”，而不是只会吃 LoRA checkpoint

当前事实：

- 现在的 `scripts/diagnose_val_outputs.py` 还不能直接加载完整公开模型
- 它固定走：
  - `AutoModelForSeq2SeqLM.from_pretrained(model_name)`
  - `PeftModel.from_pretrained(base_model, checkpoint_dir)`
- 也就是说它要求 `checkpoint_dir` 是 LoRA adapter，而不是完整 model dir

这一步必须补上，补法建议：

1. 新增一个 full-model eval shim
2. 或给 `diagnose_val_outputs.py` 增加 `--full-model-dir` 模式

完成标准：

- 用我们仓库的 eval 脚本加载 `/workspace/incoming/public_eval_byt5_akkadian_mbr/model`
- 结果与 incoming 原生线差异很小

在这一步完成前，不要声称“我们仓库已经完全兼容公开现成权重”。

建议 smoke 口径：

- `max_rows = 32`
- 主看：
  - 是否成功加载 full model
  - decode 字段是否和 incoming 一致
  - 预测是否正常落盘

## 2.4 Stage A 估时

根据公开包已有记录：

- full `313` 原生 eval：
  - `rep=1.1` 约 `785s`
  - `rep=1.0` 约 `805s`
  - 即大约 `13-14` 分钟 / 次

因此更实际的估时是：

- `A0 incoming smoke32`：约 `2-4` 分钟
- `A0 no-rep smoke32`：约 `2-4` 分钟
- `A0 deep-past compat smoke32`：约 `2-5` 分钟
- `A1 incoming full 313`：约 `13-15` 分钟
- `A1 deep-past compat full 313`：约 `13-18` 分钟

如果只先做最快可用验证：

- `A0 incoming smoke32 + A0 compat smoke32`
- 总等待大概 `5-10` 分钟

## 2.5 Stage A 失败返回策略

### A0 incoming smoke 失败

直接返回：

- 不进入 deep-past compat
- 不进入任何训练
- 先查：
  - 环境依赖
  - 权重目录完整性
  - CUDA / transformers 兼容

### A0 incoming 成功，但 A0 compat 失败

直接返回：

- 不进入 `A1`
- 不进入训练
- 只修 deep-past 的 full-model 兼容层

### A1 incoming 成功，但 A1 compat 偏差大

偏差定义建议：

- `geom` 漂移超过 `0.3`
- 或 decode 字段不一致
- 或输出健康度明显异常

直接返回：

- 不进入训练
- 先修本仓 eval / generate 实现

### A1 全都成功

才进入 Stage B。

## 3. 为什么先现成权重、再长训

因为它把三个问题拆开了：

1. 如果 incoming 原生线都跑不稳：
   - 问题在环境或依赖

2. 如果 incoming 原生线稳，但 deep-past 兼容线不稳：
   - 问题在我们自己的 eval / generate 实现

3. 如果两条锚点线都稳，而继续训练后学坏：
   - 问题才真正落在训练目标、数据分布或训练 recipe

这比现在继续赌“从头训练会不会突然学会停止”更科学。

## 4. 后续长训规划

长训顺序改成四段。

### 4.1 Stage A：锚点完成

目标：

- 稳定复现公开现成权重在本地 `fold0 raw-row 313` 的主分和输出健康度

子阶段：

1. `A0 smoke`
2. `A1 full anchor`

通过条件：

- incoming 原生线稳定
- deep-past 兼容线稳定

只有 Stage A 通过，才进入训练。

### 4.2 Stage B：现成权重继续训练的短 pilot

实验名建议：

- `PUBLIC_MODEL_R16_PUBLIC_CONT_C0_PILOT_20260314_fold0`

定义：

- 起点模型：`/workspace/incoming/public_eval_byt5_akkadian_mbr/model`
- 数据：official-only
- 训练方式：LoRA q/k/v/o, `r=16`, `alpha=32`
- 主 eval：`raw-row fold0 313`
- 辅助 eval：
  - `trunc640`
  - `diag32`
  - 健康指标

首发长度和 decode：

- 不改公开 decode profile
- `beam=8`
- `length_penalty=1.0`
- `repetition_penalty=1.1`
- `max_new_tokens=640`

步数：

- 先只跑 `max_steps = 300`
- 只扫 `ckpt100/200/300`

估时：

- smoke：`5-10` 分钟
- train `300 steps`：约 `20-35` 分钟
- checkpoint sweep `100/200/300`：约 `20-30` 分钟
- `trunc640 + diag32`：约 `10-20` 分钟
- 总计：约 `55-95` 分钟

目标：

- 回答“从一个健康公开模型出发，训练一动手会不会立刻把停止能力训坏”

硬 stop 条件：

- `top_repeat_count > 5`
- 或 `max_len_hit_ratio_pct >= 50`
- 或 `unique_prediction_ratio_pct < 90`

只要短 pilot 就已经 collapse，先停，别长训。

失败返回策略：

- `ckpt100` 就 unhealthy：
  - 立即停
  - 回到 Stage A，先查训练接入方式
- `ckpt200` 后开始 collapse：
  - 停在 `ckpt200`
  - 不开 long run
- `ckpt300` 健康但主分明显掉：
  - 不直接判死
  - 先看 `trunc640` 和 `diag32`

### 4.3 Stage C：现成权重继续训练的长 official-only 线

只有 Stage B 健康，才开这一步。

实验名建议：

- `PUBLIC_MODEL_R16_PUBLIC_CONT_C0_LONG_20260314_fold0`

定义：

- 仍然 official-only
- `max_steps = 900` 到 `1200`
- checkpoint sweep：`200/400/600/800/1000/1200`

估时：

- train `900-1200 steps`：约 `70-110` 分钟
- sweep `6` 个 checkpoint：约 `80-110` 分钟
- `trunc640 + diag32`：约 `10-20` 分钟
- 总计：约 `2.7-4.0` 小时

目标：

- 看继续训练到底能不能在不破坏停止能力的前提下提升本地主分

如果这一步比公开现成权重掉得厉害，含义非常明确：

- 不是 eval 错
- 而是“当前训练目标 / 数据 / 继续训练 recipe”本身会把健康模型训坏

失败返回策略：

- `ckpt200` 已 unhealthy：
  - 直接停 long run
  - 不进入 mix
- `ckpt400` 比锚点明显掉分且 unhealthy：
  - 直接停
- `ckpt600+` 仍健康但分数持续低于锚点：
  - 记为“继续训练有害”
  - 不进入 mix

### 4.4 Stage D：只在 C 健康后再谈 mix

只有 Stage C 健康，才进入 mix。

顺序：

1. `M025`
2. `M050`
3. 视情况才到 `M100`

不要直接从 `M050` 起。

原因：

- 当前最大的未决问题不是“external 量够不够”
- 而是“训练一介入会不会重新产生重复和不会停”

mix gate：

1. `geom >= C0 public-cont`
2. `BLEU >= C0 public-cont`
3. `top_repeat_count <= 5`
4. `max_len_hit_ratio_pct < 50`
5. `unique_prediction_ratio_pct >= 90`

估时：

- `M025`：约 `2-3` 小时
- `M050`：约 `2.5-4` 小时
- `M100`：约 `3-4.5` 小时

失败返回策略：

- `M025` 不过 gate：
  - 不进 `M050`
- `M050` 不过 gate：
  - 不进 `M100`
- 任一 mix 线出现明显 repeat / max-len saturation：
  - 直接停在该线
  - 回 official-only 分析

## 5. 当前不建议做的事

现在不建议：

1. 再开从头 `byt5-base + LoRA` 长训
2. 继续旧 `M100`
3. 把“对齐公开 eval”与“发明新 recipe”混在一起
4. 在没有现成权重锚点的情况下讨论 external 比例优化

## 6. 明早默认动作

如果只做一件事，默认动作应该是：

1. 先重跑 incoming 原生锚点
2. 再补 deep-past 的 full-model 兼容 eval
3. 只有两条锚点都稳，才开 `PUBLIC_MODEL_R16_PUBLIC_CONT_C0_PILOT`

## 7. tmux 挂后台计划

tmux 方案也拆成和阶段一致的 session，不把所有东西塞进一个 session。

### 7.1 Stage A0 smoke

session 建议：

- `pub_anchor_a0_incoming`
- `pub_anchor_a0_compat`

策略：

- 先跑 incoming smoke
- 通过后再跑 compat smoke
- 两个 session 都是短任务，跑完自动退出

### 7.2 Stage A1 full anchor

session 建议：

- `pub_anchor_a1_incoming`
- `pub_anchor_a1_compat`

策略：

- 可以顺序跑，不建议并行抢同一张卡
- `A1 incoming` 过后再开 `A1 compat`

### 7.3 Stage B/C 训练

session 建议：

- `pub_cont_pilot`
- `pub_cont_long`

每个 session 内部再拆日志：

- `train.log`
- `eval.log`
- `diag.log`
- `route_decision.md`
- `driver_status.json`

### 7.4 Stage D mix

session 建议：

- `pub_mix_m025`
- `pub_mix_m050`
- `pub_mix_m100`

### 7.5 默认后台顺序

更稳的默认后台顺序：

1. `A0 incoming smoke`
2. `A0 compat smoke`
3. `A1 incoming full`
4. `A1 compat full`
5. `B pilot`
6. `C long`
7. `D mix`

不要跳步，不要在 `A0/A1` 没钉住前直接挂 `B/C/D`。

一句话总结：

- 先用现成权重把“链”钉住
- 再用短 pilot 判断“训”会不会把好模型训坏
- 最后才决定要不要长训和混 external
