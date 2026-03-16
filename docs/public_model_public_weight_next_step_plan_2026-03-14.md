# 公开现成权重后续路线与下一步计划
## public weight next-step plan, 2026-03-14

本稿只做两件事：

1. 把 `2026-03-14` 已经落地的 `public model` 证据冻结成新的前提
2. 在此基础上定义下一条最值得做的执行顺序

本稿不启动训练，不恢复旧 `M050/M100`，不把 `M025` 改写成正线。

## Update · 2026-03-14T17:06:24+00:00

`Track B` 后续已推进到 `R19 U3 broader text-only -> same pilot`。

- 本稿应视为 `U3` 之前的路线快照
- `U3` 正式 freeze 改看：[/workspace/deep-past-/docs/public_model_u3_freeze_and_next_step_2026-03-14.md](/workspace/deep-past-/docs/public_model_u3_freeze_and_next_step_2026-03-14.md)

## 0. 结论先行

当前更科学的顺序是：

1. 正式冻结 `public model -> official-only continuation` 为当前主线
2. 正式冻结 `M025 mix = healthy but no gain -> stop`
3. 下一条强候选改成：`TAPT-lite -> official-only supervised continuation`
4. 继续把“交付增益”与“机制逆向”拆成两条线
5. 只有 `TAPT-lite` 在单轴 pilot 上给出明确正信号，才进入 long run 和 promotion test

原因很简单：

- `public-weight continuation` 现在已经不只是单次 lucky run，而是有了 seed / fold 稳定性证据
- 当前这版 `M025` 虽然健康，但在已完成的可比对上持续低于 `official-only continuation`
- 如果下一步还继续加码 `M050/M100`，本质上是在对一个已经给出负面信号的方向继续下注
- 当前真正缺的不是“再证明一次 continuation 是正的”，而是下一个单轴强候选
- 这个单轴强候选不应该再是 external mix，而应该是更上游、但仍然能和当前主线严谨对照的 `TAPT-lite`

## 1. 03-14 已冻结的事实

### 1.1 `public model` 锚点已经钉住

`public model` 固定为：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/model`

固定本地主锚点：

- `/workspace/incoming/public_eval_byt5_akkadian_mbr/reports/local_eval_byt5_akkadian_mbr/fold0/summary_default_beam8_rep11.json`
- `BLEU = 30.8114`
- `chrF++ = 49.6247`
- `geom = 39.1025`

固定主 gate：

- `raw-row fold0 313`
- 不是手工抽样
- 来源于公开模型自带本地 eval 口径
- `5-fold + group_strategy=auto`
- 实际落地是 `source_bucket + GroupKFold`

### 1.2 `public-weight continuation` 已经从 pilot 走到稳定 incumbent

当前稳定 incumbent：

- `/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/long_public_eval_best.json`
- best checkpoint: `C-long ckpt600`
- `BLEU = 31.8374`
- `chrF++ = 51.2726`
- `geom = 40.4028`

它相对 `public model` 锚点的提升是：

- `BLEU +1.0260`
- `chrF++ +1.6479`
- `geom +1.3003`

这说明：

- 我们还没有证明自己能独立长出 `public model`
- 但已经证明：把 `public model` 作为底座后，继续训练能稳定抬高

### 1.3 `stability pack` 已经跑完，而且结论很清楚

正式完成文件：

- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/driver_status.json`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/route_decision.md`

核心结果：

- `cont fold0 geom mean/std = 40.5055 / 0.1281`
- `mix fold0 geom mean/std = 39.9578 / 0.1775`
- `cont 3-fold OOF geom = 41.6565`
- `mix 3-fold OOF geom = 41.0411`

证据：

- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/summaries/cont_fold0_seed_summary.json`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/summaries/mix_fold0_seed_summary.json`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/summaries/cont_fold0_1_2_oof_summary.json`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/summaries/mix_fold0_1_2_oof_summary.json`

已完成的 5 组可比对上，结论都是同一个方向：

- `official-only continuation > current M025 mix`

### 1.4 paired significance 的结论也支持停掉 `M025`

正式 paired significance 目录：

- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/significance`

BLEU paired bootstrap p-value：

- `fold0_seed42 = 0.2617`
- `fold0_seed43 = 0.0090`
- `fold0_seed44 = 0.0490`
- `fold1_seed42 = 0.0060`
- `fold2_seed42 = 0.0180`

这意味着：

- 5 组里有 4 组在 BLEU 上达到 `p < 0.05`
- 唯一不显著的是 `fold0_seed42`

同时也要明确写清：

- 这次 `chrF++` paired-bs 全部 fallback
- 所以不能写成“所有指标都显著”

### 1.5 `M025` 的问题不是坏掉，而是“没有净增益”

当前最准确的冻结表述是：

- `M025 pilot = healthy but no gain -> stop`

不能写成：

- `M025 collapse`
- `mix 完全无效`

更准确是：

- 它没有坏
- 但它没有赢过当前这条 `official-only continuation`

而且目前最像真的解释不是“训练链错了”，而是：

- external 数据和 official 分布不够对齐
- 尤其长度分布明显更短

审计证据：

- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/audits/external_domain/summary.json`

关键长度事实：

- official target token `p95 = 1385`
- external target token `p95 = 531`

这条观察现在只能写成：

- working hypothesis

不能写成：

- 已经证明 mix 负效应完全由长度分布导致

## 2. 这些结果到底意味着什么

### 2.1 已经证明的事

现在已经可以硬认两件事：

1. `public model` 本地锚点是稳定、可复用的
2. `public-weight continuation` 是真正稳定成立的增益方向

所以从 03-14 开始：

- `official-only continuation` 不再只是一个候选
- 它就是当前主线

### 2.2 还没有证明的事

现在仍然没有证明：

- 我们已经理解 `public model` 为什么这么强
- 我们已经独立复现了它的形成机制
- external mix 一定不该用
- `TAPT-lite` 一定会正

所以后续文档口径必须继续保持：

- 对已经成立的结果敢认
- 对机制和成因不抢答

### 2.3 为什么下一步不该是 `M050/M100`

因为当前 `M025` 已经回答了最重要的问题：

- `mix` 不会立刻 collapse
- 但当前这版 mix 没有赢过 `official-only continuation`

在这种情况下直接去 `M050/M100`，相当于：

- 对一个已经显示负面实用信号的方向继续加码

这和 03-13 那份计划的前提已经不同。

03-13 的 `Stage D` 还是：

- 只有 `Stage C` 健康，才进入 `M025`

而 03-14 之后的新事实是：

- `M025` 已经完成，而且不过 promote 逻辑

所以从这版开始：

- `mix` 保留为次级候选
- 不再默认晋升到 `M050/M100`

## 3. 03-14 之后的执行顺序

从这版开始，执行顺序改成六段。

### 3.1 Stage E0：冻结当前主线

目标：

- 把 03-14 已经跑完的结果固定成新的路线前提

需要冻结的文件：

- `/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/route_decision.md`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/route_decision.md`
- `/workspace/deep-past-/reports/public_model_r16_stability_pack_20260314/dual_track_route_map_20260314.md`
- `/workspace/deep-past-/reports/public_model_r16_public_mix_m025_20260314/public_vs_current_model_snapshot_20260314.md`

完成标准：

- `public-weight continuation` 被明确写成主线
- `M025` 被明确写成 `healthy but no gain -> stop`
- 文档里不再把 `M050/M100` 写成默认后续步骤

在这一步完成前，不要直接开新实验。

### 3.2 Stage E1：定义最小化 `TAPT-lite` 候选

下一条强候选固定为：

- `TAPT-lite -> official-only supervised continuation`

它的作用不是替代当前主线，而是回答新的单轴问题：

- 如果先做一个极小的 continued pretraining，再走同一条 `official-only continuation`，会不会在当前主线上给出实用级别正信号？

为了避免把多个变量混在一起，这个候选必须锁住以下不变项：

1. 起点仍然是：
   - `/workspace/incoming/public_eval_byt5_akkadian_mbr/model`
2. 下游监督阶段仍然是：
   - official-only
3. 主 gate 仍然是：
   - `raw-row fold0 313`
4. decode 仍然锁定：
   - `beam=8`
   - `length_penalty=1.0`
   - `repetition_penalty=1.1`
   - `max_new_tokens=640`
5. 不引入 external supervised mix
6. 不同时改 cleaning / tokenizer / decode / adapter 规模

默认最小 TAPT-lite 语料建议：

- 先只用 official source-side text
- 不先混 external
- 不先混 test
- 不把 `TAPT-lite` 和 `external domain expansion` 绑在一起

原因：

- 当前我们要回答的是 `H3 = continued pretraining` 是否有用
- 不是重新回到 `H2 = external mix` 这条已给负信号的线

建议新配置名：

- `configs/public_model_r17_public_tapt_lite_20260314.yaml`
- `configs/public_model_r17_public_taptlite_cont_c0_pilot_20260314.yaml`
- `configs/public_model_r17_public_taptlite_cont_c0_long_20260314.yaml`

建议新 driver：

- `scripts/public_model_r17_public_taptlite_cont_driver.py`

可复用工具：

- `/workspace/deep-past-/scripts/tapt_denoise.py`
- `/workspace/deep-past-/scripts/train_mt5_lora.py`
- `/workspace/deep-past-/scripts/public_model_r1_rawrow_eval.py`

### 3.3 Stage E2：先做 `TAPT-lite` smoke，不开长训

目标：

- 只回答这条新链能不能顺利接通

建议实验名：

- `PUBLIC_MODEL_R17_PUBLIC_TAPT_LITE_SMOKE_20260314`
- `PUBLIC_MODEL_R17_PUBLIC_TAPTLITE_CONT_C0_SMOKE_20260314_fold0`

smoke 只看：

- `TAPT-lite` 是否能正常产出 `best_model`
- downstream continuation 是否能正常读取 `init_adapter_dir`
- eval 是否正常落盘
- 输出是否明显 collapse

推荐 hard stop：

- `top_repeat_count > 5`
- 或 `max_len_hit_ratio_pct >= 50`
- 或 `unique_prediction_ratio_pct < 90`

只要 smoke 不健康，直接停，不进入 pilot。

### 3.4 Stage E3：做 directional pilot，而不是直接 long

只有 smoke 通过，才进入这一步。

目标：

- 判断 `TAPT-lite` 是否在当前主线之上带来实用级别信号

推荐比较对象不是 `public model` 锚点，而是当前 plain continuation 的同预算 pilot：

- `/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/route_decision.md`
- `Stage B ckpt300 = geom 40.2266`

原因：

- 现在 `public model` 锚点已经太低
- 只比锚点高，不足以证明这条新方向值得继续
- 新候选必须先赢过“同预算 plain continuation”

pilot 定义建议：

- 下游监督仍然只跑 `300 steps`
- 只扫 `ckpt100/200/300`
- 主 eval 仍为 `raw-row fold0 313`
- 仍保留 `trunc640 + diag32`

建议判定规则：

1. health 先过线
2. 相对 plain continuation pilot：
   - `geom >= +0.2`：记为 `positive`
   - `-0.2 < delta < +0.2`：记为 `inconclusive`
   - `geom <= -0.2`：记为 `negative`

这里故意不用小数点后很小的波动做 promote。

### 3.5 Stage E4：只有 pilot 正，才开单次 long confirm

只有 `Stage E3 = positive`，才进入 long。

目标：

- 判断 `TAPT-lite` 是否有机会真正替代当前 stable incumbent

比较对象：

- 当前 stable incumbent
- `/workspace/deep-past-/reports/public_model_r16_public_cont_20260313/long_public_eval_best.json`
- `geom = 40.4028`

建议 long 定义：

- 下游监督仍然 official-only
- `max_steps = 900` 到 `1200`
- sweep：`200/400/600/800/1000/1200`
- 先只跑 `fold0 seed42`

建议判定规则：

1. health 先过线
2. 相对当前 `C-long`：
   - `geom >= +0.2`：进入 promotion pack
   - `-0.2 < delta < +0.2`：记为 `inconclusive`
   - `geom <= -0.2`：记为 `negative`

如果 long 只是贴着 incumbent 小幅波动，不要硬升格。

### 3.6 Stage E5：只有 long 正，才开 promotion pack

只有 `Stage E4 = positive`，才进入这一步。

promotion pack 结构直接复用 03-14 这次已经跑通的逻辑：

1. `fold0` 三个 seed：`42 / 43 / 44`
2. `fold1 / fold2` 的 seed42 对照
3. paired significance
4. `3-fold OOF` 汇总
5. seed stability 汇总

promotion pack 的比较对象不再是 `public model` 锚点，而是：

- 当前 plain `official-only continuation`

只有 promotion pack 也为正，才能把主线从：

- `public-weight continuation`

升级为：

- `public-weight TAPT-lite -> continuation`

## 4. Track B 默认只做什么

03-14 之后，`Track B` 的任务不是立刻开新长训，而是先把研究问题定义清楚。

默认只做三件事：

1. 保留 `public-weight continuation` 这条线的机制边界说明
2. 把 external 长度 / 分布不对齐继续写成审计事实，不写成定论
3. 把 `TAPT-lite` 视为新的 `H3` probe，而不是“已经找到成因”

在 `Track A` 的新候选没有结果前，不建议再同时开：

- tokenizer / normalization 线
- external mix 比例线
- 多轴 recipe 猜测线

## 5. 当前不建议做的事

现在不建议：

1. 继续 `M050/M100`
2. 重开旧 `raw-row supervised mix` 主线
3. 再开从头 `byt5-base + LoRA` 长训
4. 把 `TAPT-lite` 和 external mix 绑成一个实验
5. 同时改 decode、cleaning、adapter 规模和语料
6. 把 tiny positive 当成 promote 证据

## 6. 现在默认动作

如果只做一件事，默认动作应该是：

1. 先把 03-14 的 route 和 snapshot 结论冻结
2. 再写出 `R17 TAPT-lite -> official-only continuation` 的最小 config / driver
3. 先跑 smoke
4. smoke 通过后，再跑 directional pilot

不要跳步，不要直接开 long，不要先开 `M050/M100`。

## 7. tmux 挂后台计划

tmux 也继续按阶段拆，不把所有任务塞进一个 session。

### 7.1 Stage E2 smoke

session 建议：

- `pub_taptlite_smoke`
- `pub_taptlite_cont_smoke`

日志建议：

- `tapt.log`
- `train.log`
- `eval.log`
- `route_decision.md`
- `driver_status.json`

### 7.2 Stage E3 pilot

session 建议：

- `pub_taptlite_pilot`

### 7.3 Stage E4 long confirm

session 建议：

- `pub_taptlite_long`

### 7.4 Stage E5 promotion pack

session 建议：

- `pub_taptlite_stability_pack`

一句话总结：

- 03-13 的计划解决了“先把现成公开权重锚点钉住，再看 continuation 会不会学坏”
- 03-14 的计划则是在这个基础上继续推进：
- 主线正式冻结为 `public-weight continuation`
- `M025` 正式停掉
- 下一条最值得做的单轴强候选是 `TAPT-lite -> official-only supervised continuation`
- 而且必须按 `smoke -> pilot -> long confirm -> promotion pack` 的顺序走
