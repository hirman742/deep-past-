# 公开模型对标 R1.5 纠偏计划
## strict reproduction correction，2026-03-13

本稿承接：

- [public_model_r1_rawrow_supervised_mix_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_r1_rawrow_supervised_mix_plan_2026-03-13.md)
- [public_model_phase1_decode_ablation_2026-03-13.md](/workspace/deep-past-/docs/public_model_phase1_decode_ablation_2026-03-13.md)
- [public_model_fast_repro_plan_2026-03-13.md](/workspace/deep-past-/docs/public_model_fast_repro_plan_2026-03-13.md)

本稿只定义下一步怎么纠偏，不实施代码，不恢复当前已挂起进程。

## 0. 结论先行

当前 `R1 raw-row + external supervised mix` 线先冻结，不继续默认推进。

冻结理由不是“raw-row 313 口径错了”，而是：

1. 当前线没有完整对齐 `/workspace/incoming` 公开包的本地 eval profile
2. 当前线存在明确的训练/评测长度不一致
3. `C0` 和 `M050` 都显示出同一种核心病灶：
   - 不会停
   - 顶着 `640` 输出
   - 重复串塌缩

因此，明早第一枪不应是旧 `M100`，也不应是继续赌 external 比例，而应是：

- `R1.5-C0-realign`
- `official-only`
- `raw-row fold0 313` 主口径不变
- 先把对齐误差和健康诊断补齐

## 1. 当前线为什么冻结

### 1.1 `313` 口径不是问题，必须保留

当前主口径：

- `train.csv` raw-row
- `fold0`
- `313` 行
- corpus BLEU
- corpus chrF++
- `geom = sqrt(BLEU * chrF++)`

这套口径和公开本地 eval 对齐，不是临时抽样。

证据：

- [data_build_audit.json](/workspace/deep-past-/reports/public_model_r1_rawrow_supervised_mix_20260313/data_build_audit.json)
- [public_eval_byt5_akkadian_mbr_local.yaml](/workspace/incoming/public_eval_byt5_akkadian_mbr/configs/public_eval_byt5_akkadian_mbr_local.yaml)

因此，纠偏时不改 `313`，否则直接失去和公开本地结果的可比性。

### 1.2 当前线并没有诚实对齐 `/workspace/incoming`

当前线和公开包的差异至少有三层：

1. 模型本体不同
   - 公开包本地 eval 用的是完整导出的公开模型
   - 当前线用的是 `google/byt5-base + LoRA`

2. generation profile 没完全对齐
   - 公开包：
     - `beam=8`
     - `length_penalty=1.0`
     - `repetition_penalty=1.1`
     - `max_new_tokens=640`
     - `suppress_extra_ids=false`
   - 当前线：
     - `beam=8`
     - `length_penalty=1.0`
     - `max_new_tokens=640`
     - `suppress_extra_ids=true`
     - 未显式设置 `repetition_penalty`

3. `/workspace/incoming` 提供的是本地 eval/ablation 包，不是公开训练 recipe
   - 因此当前最多只能说“部分 eval 对齐”
   - 不能说“已经训练复现了公开模型”

证据：

- [public_eval_byt5_akkadian_mbr_local.yaml](/workspace/incoming/public_eval_byt5_akkadian_mbr/configs/public_eval_byt5_akkadian_mbr_local.yaml)
- [generation_config.json](/workspace/incoming/public_eval_byt5_akkadian_mbr/model/generation_config.json)
- [run_phase1_ablation.sh](/workspace/incoming/public_eval_byt5_akkadian_mbr/run_phase1_ablation.sh)

### 1.3 当前线存在长度口径不一致

当前训练和评测的长度定义不一致：

- 训练时 label 截断到 `max_target_length=640`
- 评测时 reference 用完整 `target`

对当前任务，这不是小偏差，而是系统性偏差。

已测得的 ByT5 token 长度事实：

- official-only train 中，`356 / 1248` 个 target 超过 `640`
- fold0 val 中，`91 / 313` 个 target 超过 `640`
- M050 train 中，`380 / 2028` 个 target 超过 `640`

这意味着模型在大量样本上根本没被监督“完整答案如何结束”，却被要求在 full-ref 指标上负责。

### 1.4 当前失败不是偶然波动，而是稳定坏相

`C0` 最优仅到：

- `geom = 3.2255`
- `BLEU = 1.2940`
- `chrF++ = 8.0400`

且 `diag32 healthy = false`。

`M050` 到 `ckpt600` 仍然是：

- `geom = 2.5669`
- `BLEU = 0.9014`
- `chrF++ = 7.3098`
- `unique_prediction_ratio_pct = 70.6070`
- `pred_tok_mean = 638.69`
- top repeated prediction `count = 19`

这说明 external 0.5x 目前没有把主病灶治掉。

证据：

- [route_decision.md](/workspace/deep-past-/reports/public_model_r1_rawrow_supervised_mix_20260313/route_decision.md)
- [val_diagnostic_summary_c0_rawrow_ckpt600.json](/workspace/deep-past-/runs/PUBLIC_MODEL_R1_RAWR0W_C0_20260313_fold0/diagnostics/val_diagnostic_summary_c0_rawrow_ckpt600.json)
- [val_diagnostic_summary_m050_rawrow_ckpt600.json](/workspace/deep-past-/runs/PUBLIC_MODEL_R1_RAWR0W_M050_20260313_fold0/diagnostics/val_diagnostic_summary_m050_rawrow_ckpt600.json)

## 2. R1.5 的目标

`R1.5` 不是继续 external mix，也不是新 recipe。

`R1.5` 只回答两个问题：

1. 当前负结果里，有多少是对齐误差和诊断缺口造成的
2. 在不换研究问题的前提下，`official-only + raw-row 313` 能否先学会正常停止、避免明显重复

`R1.5` 不回答：

- external `0.5x -> 1.0x` 是否继续增益
- continuation / 分段训练是否更强
- 新 recipe 能否打过公开线

这些问题都推迟到 `R1.5` 之后。

## 3. 固定边界

以下边界在 `R1.5` 中不动：

1. 主评测口径仍是 `raw-row fold0 313`
2. 主指标仍是 `BLEU / chrF++ / geom`
3. fold 划分仍以官方 `train.csv` 为根
4. 首轮只跑 `official-only`
5. `diag32` 仍保留，但只作为健康诊断

以下内容不进入 `R1.5`：

1. external supervised mix
2. `M100`
3. continuation / segmented target 训练
4. chooser / replay / fallback / repair
5. MBR / rerank

## 4. 明早第一枪：`R1.5-C0-realign`

### 4.1 实验定义

实验名：

- `PUBLIC_MODEL_R15_RAWR0W_C0_REALIGN_20260314_fold0`

训练集：

- 官方 `1561`
- 不混 external

模型：

- 先保持 `google/byt5-base + LoRA(q/k/v/o, r=16, alpha=32)`
- 不在第一枪同时改 adapter 容量

理由：

- 第一枪目标是定位对齐误差和健康病灶
- 不是同时重新搜索模型容量

### 4.2 generation profile

`R1.5` 明确拆成两个 profile：

1. `public_eval_profile`
   - `beam=8`
   - `length_penalty=1.0`
   - `repetition_penalty=1.1`
   - `max_new_tokens=640`
   - `min_new_tokens=0`
   - `no_repeat_ngram_size=0`
   - `suppress_extra_ids=false`

2. `no_rep_diag_profile`
   - 与上面相同
   - 仅把 `repetition_penalty=1.0`

规则：

- checkpoint 选择与主 gate 只看 `public_eval_profile`
- `no_rep_diag_profile` 只做辅助诊断，不能拿来选 winner

### 4.3 新增辅助 eval

除了主评测 `raw-row fullref 313`，新增一个辅助 eval：

- `trunc640 eval`

定义：

- 对验证参考做同 tokenizer、同 `640` 截断
- 只用于判断模型在“训练覆盖到的长度区间”里是否学会基本映射

规则：

- `trunc640` 不是主分
- 不能拿来替代 fullref
- 但如果 `trunc640` 明显改善而 fullref 仍极差，则可证明长度口径是重要病灶

## 5. 健康诊断硬门槛

`R1.5` 不再只看 `geom`。

必须新增并显式记录这些指标：

1. `max_len_hit_ratio_pct`
   - 预测长度恰好打到 `max_new_tokens` 的比例

2. `eos_before_limit_ratio_pct`
   - 在未触顶前正常结束的比例

3. `top_repeat_count`
   - 最常见重复输出出现次数

4. `unique_prediction_ratio_pct`
   - 唯一预测占比

5. `pred_tok_mean`
   - 预测 token 平均长度

### 5.1 R1.5 的 unhealthy 判定

满足任一条，就判为 unhealthy：

1. `top_repeat_count > 5`
2. `unique_prediction_ratio_pct < 90`
3. `pred_tok_mean >= 630`
4. `max_len_hit_ratio_pct >= 50`

原因：

- 当前 `C0/M050` 的坏相不是“略短”或“略低分”
- 而是“顶满上限 + 重复串塌缩”
- 这些现象必须成为硬门槛

## 6. 训练与 checkpoint sweep 规则

`R1.5` 第一枪不再完整跑 `200/400/600/800/1000/1200` 全扫。

先收缩为：

- `checkpoint = 200 / 400 / 600`

原因：

- 当前坏相在 `400` 前后已经显著出现
- 没必要再为一个明显 unhealthy 的配置烧完整轮 sweep

### 6.1 早停规则

若 `ckpt400` 同时满足：

1. `raw-row geom < 3.0`
2. `top_repeat_count > 5`
3. `pred_tok_mean >= 630` 或 `max_len_hit_ratio_pct >= 50`

则直接停止该线，不继续 `ckpt600+`。

## 7. R1.5 通过与失败的定义

### 7.1 通过

满足以下全部，才允许继续到下一步：

1. `raw-row geom > C0 best = 3.2255`
2. `raw-row BLEU > C0 best = 1.2940`
3. `unhealthy = false`
4. `trunc640 eval` 不显示明显崩坏

若通过，下一步才允许讨论：

- `R1.5-M050-realign`

### 7.2 失败

若不满足以上条件，则结论写死为：

- 当前 `official-only + byt5-base + len640 raw-row` 线在纠偏后仍未站稳
- 问题不再主要是 external 比例
- 应转入新 recipe 线

## 8. 若 R1.5 失败，后续只开一条新线

若 `R1.5-C0-realign` 失败，不再继续旧 `M050/M100`，也不继续做同比例 mix。

只开一条明确命名的新线：

- `R2 stop-aware recipe`

但这条线不在本稿实施，只先定义边界：

1. 仍保留 `raw-row 313` 主口径
2. 目标优先级从“分数”改成“先学会停止、先消除重复”
3. 第一枪仍从 `official-only` 开，不直接混 external

## 9. 明早实际交付物

明早真正要产出的不是“更多日志”，而是以下清单：

1. `R1.5` 计划配置
2. `public_eval_profile` 与 `/workspace/incoming` 差异审计表
3. `trunc640 eval` 入口
4. 新健康指标汇总表
5. `R1.5-C0-realign` 的 stop/go route decision

## 10. 最短结论

明早最值得做的不是：

- 恢复旧 `M050`
- 继续旧 `M100`
- 再赌 external 比例

明早最值得做的是：

- 保留 `raw-row 313` 主口径
- 回到 `official-only`
- 把 `/workspace/incoming` 的本地 eval profile 对齐干净
- 把“不会停、会重复”升格为硬门槛
- 用 `R1.5-C0-realign` 先判定这条线到底是“实现没对齐”，还是“recipe 本身站不住”
