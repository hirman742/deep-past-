# Cloud Stage2 本机实验清单（2026-03-09）

## 1. 文档目的

本文记录本机在本轮 Cloud Stage2 steering 中实际完成或正在进行的训练、decode、diagnose 与小型对照实验，作为下一阶段正式实验的机器侧台账。

## 2. 基线闭环

### 2.1 incumbent baseline

- 基线权重：
  - `runs/CLOUD_STAGE2_GC_CURRICULUM_COST14_FROM_S1WIN_fold0/best_model`
- full-val decode grid 已完整跑完：
  - `beam=4, lp=0.8, max_new_tokens=384`
  - `beam=4, lp=1.0, max_new_tokens=384`
  - `beam=6, lp=0.8, max_new_tokens=384`
  - `beam=6, lp=1.0, max_new_tokens=384`
- best decode：
  - `beam=6, lp=0.8, max_new_tokens=384`
  - `geom / bleu / chrfpp = 13.8738 / 7.3377 / 26.2317`
- full-val diagnose 已完成

### 2.2 基线结论

- incumbent 侧 decode 网格只能带来极小提升。
- `beam=6` 不是主提分杠杆。
- 当前正式对照线固定为：
  - `13.8738 / 7.3377 / 26.2317`

## 3. 候选主线与控制线

### 3.1 `continue_s4_bs32_len512`

方法：

- 从 incumbent `best_model` warm-start
- `max_source_length=max_target_length=512`
- `per_device_train_batch_size=32`
- LoRA：`q/v`, `r=8`, `alpha=16`, `dropout=0`
- 本机已完成：
  - smoke/probe 训练
  - probe decode/diagnose
  - warmup 训练
  - warmup anchor decode/diagnose

关键结果：

- probe reconstructed：
  - `16.9907 / 9.5997 / 30.0722`
- warmup anchor32：
  - `17.1785 / 9.8179 / 30.0575`
- warmup diag32：
  - `empty=0`
  - `pred_shorter_than_half_ref=6.25%`

结论：

- 这是可靠的控制线/保底线。
- 它优于 incumbent probe，但不如当前最佳 `len640` 主线。

### 3.2 `continue_s4_bs24_len640`

方法：

- 从 incumbent `best_model` warm-start
- `max_source_length=max_target_length=640`
- `per_device_train_batch_size=24`
- LoRA：`q/v`, `r=8`, `alpha=16`, `dropout=0`
- 本机已完成：
  - smoke/probe 训练
  - probe decode/diagnose
  - warmup 训练
  - warmup decode/diagnose
  - reprobe
  - `seg25`
  - `seg5`

关键结果：

- 原始 probe：
  - `17.5342 / 10.1074 / 30.4182`
- warmup：
  - `15.0523 / 8.2085 / 27.6018`
- reprobe `ckpt250 @ 4 / 0.8 / 384`：
  - `17.5369 / 10.0172 / 30.7013`

结论：

- `len640` 的强信号可复现。
- 增益集中在较早 checkpoint，不在后段 warmup。
- 问题核心转移到了 checkpoint 选择与 decode 稳定性。

## 4. checkpoint 窗口实验

### 4.1 `seg25`

方法：

- 细化早期 checkpoint 窗口
- anchor decode 固定：
  - `beam=4, lp=0.8, max_new_tokens=384`
- 比较：
  - `ckpt200`
  - `ckpt225`
  - `ckpt275`

结果：

- `ckpt200`: `17.1296 / 9.6497 / 30.4072`
- `ckpt225`: `17.2515 / 9.8209 / 30.3043`
- `ckpt275`: `17.0462 / 9.4837 / 30.6394`

结论：

- 最优区间明显早于 `300`。
- `225-250` 是应继续细扫的窗口。

### 4.2 `seg5`

方法：

- 更细的 checkpoint 间隔实验
- anchor decode 固定：
  - `beam=4, lp=0.8, max_new_tokens=384`
- 比较：
  - `ckpt225`
  - `ckpt230`
  - `ckpt240`
  - `ckpt250`
  - `ckpt300`

结果：

- `ckpt225`: `17.5905 / 10.0275 / 30.8579`
- `ckpt230`: `17.1833 / 9.6634 / 30.5551`
- `ckpt240`: `17.0285 / 9.4045 / 30.8332`
- `ckpt250`: `17.8351 / 10.3338 / 30.7815`
- `ckpt300`: `16.6431 / 9.1337 / 30.3266`

结论：

- 当前局部最优是 `ckpt250`。
- 继续训练到 `300` 会明显退化。

## 5. decode 侧实验

### 5.1 cap sweep（`ckpt250`）

固定：

- `beam=4`
- `lp=0.8`

比较：

- `max_new_tokens=384`
- `max_new_tokens=448`
- `max_new_tokens=512`

结果：

- `384`: `17.8351 / 10.3338 / 30.7815`
- `448`: `17.2342 / 9.7633 / 30.4217`
- `512`: `16.6463 / 9.2265 / 30.0331`

结论：

- 对当前最佳 checkpoint，`384` 是正确上限。
- 更大的 cap 只会掉分。

### 5.2 局部 length penalty sweep（`ckpt250`）

固定：

- `beam=4`
- `max_new_tokens=384`

比较：

- `lp=0.7`
- `lp=0.72`
- `lp=0.75`
- `lp=0.8`
- `lp=0.9`

结果：

- `lp=0.7`: `18.3354 / 11.1352 / 30.1913`
- `lp=0.72`: `18.1134 / 10.8611 / 30.2083`
- `lp=0.75`: `18.0362 / 10.6497 / 30.5459`
- `lp=0.8`: `17.8351 / 10.3338 / 30.7815`
- `lp=0.9`: `15.9255 / 8.3337 / 30.4335`

结论：

- 最强 decode 点是 `lp=0.7`。
- `0.72/0.75` 接近，但仍弱于 `0.7`。
- `0.9` 明显过高。

## 6. promote-lite full-val

方法：

- 用 `seg5 ckpt250` 做 full-val 单点对决
- 比较：
  - `beam=4, lp=0.7, max_new_tokens=384`
  - `beam=4, lp=0.8, max_new_tokens=384`

结果：

- `lp=0.7`：
  - `14.3323 / 7.7369 / 26.5499`
- `lp=0.8`：
  - `13.9214 / 7.2415 / 26.7631`
- incumbent best：
  - `13.8738 / 7.3377 / 26.2317`

结论：

- `len640 ckpt250 @ 4 / 0.7 / 384` 是当前本机最强主线。
- 相对 incumbent full-val 提升约 `+0.4585 geom`。

## 7. diagnose / 健康度

### 7.1 incumbent promote full-val diagnose

- reconstructed：
  - `13.8738 / 7.3377 / 26.2317`
- health：
  - `empty=0.0`
  - `copy=0.0`
  - `pred_shorter_than_half_ref=3.83%`

### 7.2 `len640 ckpt250 lp0.7` full-val diagnose

- overall：
  - `11.2464 / 6.1041 / 20.7207`
- reconstructed：
  - `14.3323 / 7.7369 / 26.5499`
- reconstructed health：
  - `empty=0.0`
  - `copy=0.0`
  - `pred_shorter_than_half_ref=3.83%`

结论：

- `lp=0.7` 的 full-val winner 已经健康闭环。
- reconstructed 健康项不比 incumbent 更差。

### 7.3 `lp0.7` / `lp0.75` 小样本 diagnose

- `lp0.7 diag64`
  - reconstructed：`16.5057 / 9.8606 / 27.6291`
  - `empty=1.56%`
  - `pred_shorter_than_half_ref=12.5%`
- `lp0.75 diag64`
  - reconstructed：`16.1847 / 9.3801 / 27.9256`
  - `empty=1.56%`
  - `pred_shorter_than_half_ref=7.81%`

结论：

- `lp0.7` 更强，但小样本上更激进。
- 最终 full-val diagnose 已经证明它仍可接受。

## 8. micro 实验总结

### 8.1 值得保留

- `micro03 len640 lr=4e-5`
  - `17.8730 / 10.4592 / 30.5419`
  - 给出轻微正信号
- `micro14 lp=0.72`
  - `18.1134 / 10.8611 / 30.2083`
  - 可作 decode 保守备选
- `micro02 bs32 warmup diag32`
  - 证明 `bs32` 保底线有效

### 8.2 中性，不再扩展

- `micro04 qkvo`
- `micro05 r16`
- `micro10 dropout=0.05`
- `micro11 soup(225,240,250)`
- `micro12 soup(240,250,255)`

这些要么持平，要么只有很弱信号，不值得继续烧大实验。

### 8.3 负向，停止

- `micro06 parent-balanced(max_chunks_per_parent=2)`
- `micro07 inverse-sample`
- `micro08 progressive 512->640`
- `micro09 mixed 512/640`
- `micro13 ckpt260`

## 9. orthogonal 小实验总结

方法：

- 在 `micro03 len640 lr=4e-5` 基础上，分别测试：
  - 去掉 task prefix
  - 打开 `apply_t0_normalize`
  - source lowercase
  - exact pair dedup

结果：

- `ortho01 noprefix`: `17.8730 / 10.4592 / 30.5419`
- `ortho02 t0norm`: `17.8730 / 10.4592 / 30.5419`
- `ortho03 lowercase_source`: `17.8730 / 10.4592 / 30.5419`
- `ortho04 dedup_pair`: `15.6025 / 9.3345 / 26.0792`

结论：

- `noprefix / t0norm / lowercase_source` 在当前 competition-only 路径上几乎不产生变化。
- `dedup exact pair` 明显负向，应停止。

## 10. 最新训练补充

### 10.1 `continue_s4_bs24_len640_lr4e5_peak240`

方法：

- `len640`
- `bs24`
- `lr=4e-5`
- 训练到 `240` 步
- `eval/save=5`
- 目标：验证更低 LR 下，最优点是否稳定前移到 `220-230`

当前结果：

- 已完成
- `best_model_checkpoint`: `checkpoint-230`
- `best_eval_loss = 0.8547`

结论：

- `lr=4e-5` 的最优点进一步前移到 `225-230` 区间。
- 相比此前 `seg5 @ lr=5e-5`，这条线说明“更低 LR + 更早停”是当前最可信的训练侧优化方向。

### 10.2 `continue_s4_bs24_len640_lr4e5_peak240_t0`

方法：

- 与主线完全一致
- 唯一差异：`apply_t0_normalize=true`
- 目标：判断完整 Tier-0 开关在当前 competition-only 主线中是否带来差异

当前结果：

- 已完成
- `best_model_checkpoint`: `checkpoint-230`
- `best_eval_loss = 0.8547`

结论：

- 在当前 competition-only 路径下，打开 `Tier-0` 并未观察到可见训练收益或损失。
- 这进一步支持：Tier-1/更完整清洗不应阻塞下一轮正式主线训练。

### 10.3 `continue_s4_bs24_len640_lr3e5_peak240`

方法：

- `len640`
- `bs24`
- `lr=3e-5`
- 训练到 `240` 步
- `eval/save=5`
- 目标：验证学习率再降一档后，是否能进一步稳定早 checkpoint

结果：

- 已完成
- `best_model_checkpoint`: `checkpoint-90`
- `best_eval_loss = 0.8546802`

结论：

- `lr=3e-5` 并没有在 loss 上明显优于 `lr=4e-5`。
- 最优点过早前移到 `checkpoint-90`，更像是训练过早饱和，而不是更稳的提升。

### 10.4 `continue_s4_bs24_len640_lr4e5_peak240_seed43`

方法：

- `len640`
- `bs24`
- `lr=4e-5`
- `seed=43`
- 训练到 `240` 步
- `eval/save=5`
- 目标：验证当前 `lr=4e-5` 方向是否对随机种子稳健

结果：

- 已完成
- `best_model_checkpoint`: `checkpoint-155`
- `best_eval_loss = 0.8547641`

结论：

- `seed43` 的 loss 与主 seed 极接近，说明 `lr=4e-5` 的方向不是偶然噪声。
- 但最优 checkpoint 从 `230` 前移到 `155`，说明“早 checkpoint 更好”的趋势很稳，而“最优点位置”本身仍有种子敏感性。

### 10.5 `continue_s4_bs32_len512_lr4e5_peak240`

方法：

- `len512`
- `bs32`
- `lr=4e-5`
- 训练到 `240` 步
- `eval/save=5`
- 目标：在同一优化制度下，验证 `bs32_len512` 是否能成为可信控制线或替代线

结果：

- 已完成
- `best_model_checkpoint`: `checkpoint-75`
- `best_eval_loss = 0.8537620`

结论：

- 这条线在训练 loss 上优于两条 `len640` 线。
- 但它目前还只有训练信号，没有 decode 证据，因此不能直接取代 `len640 ckpt250 lp0.7` 的 full-val 主线地位。

## 11. 当前机器侧结论

- 当前最强正式候选：
  - `len640 seg5 ckpt250 @ 4 / 0.7 / 384`
- 当前可信控制线：
  - `bs32_len512 @ lr=4e-5`
- 当前最值得继续追的训练侧变化：
  - `lr=4e-5`
- 新补充实验说明：
  - `lr=3e-5` 没有给出明确优势
  - `lr=4e-5 (seed43)` 证明该方向具有可复现性
  - `bs32_len512 @ lr=4e-5` 在 loss 上最强，但仍缺 decode 证据
- 下一阶段应优先做 selector / anchor decode：
  - `len640 lr=4e-5 peak240` 的 `ckpt225 / ckpt230 @ 4 / 0.7 / 384`
  - `len640 lr=4e-5 seed43` 的 `ckpt155 @ 4 / 0.7 / 384`
  - `bs32_len512 lr=4e-5` 的 `ckpt75 @ 4 / 0.7 / 384` 与 `4 / 0.8 / 384`
- selector 通过后再决定是否进入新的 full-val promote-lite
- 当前可以暂停的方向：
  - `beam=6`
  - `max_new_tokens > 384`
  - `ckpt260+`
  - `qkvo / r16 / dropout=0.05`
  - `parent-balanced / inverse-sample`
  - `mixed / progressive length`
  - `dedup exact pair`
  - `noprefix / t0norm / lowercase_source`（至少在当前 competition-only 设定下）

## 12. 下一阶段 selector 轮

已挂起任务：

- `len640 lr=4e-5 peak240`:
  - `checkpoint-225 @ 4 / 0.7 / 384`
  - `checkpoint-230 @ 4 / 0.7 / 384`
- `len640 lr=4e-5 peak240 seed43`:
  - `checkpoint-155 @ 4 / 0.7 / 384`
- `bs32_len512 lr=4e-5 peak240`:
  - `checkpoint-75 @ 4 / 0.7 / 384`
  - `checkpoint-75 @ 4 / 0.8 / 384`

selector 规则：

- 先比较 32-sample anchor decode 的 `geom`
- 选出单一 winner
- 只对 winner 追加 `64-sample diagnose`

目的：

- 判断 `lr=4e-5` 主线是否应从 `seg5 ckpt250` 切换到更早 checkpoint
- 判断 `seed43` 是否能在 decode 侧复现主线优势
- 判断 `bs32_len512 @ lr=4e-5` 是否能从“训练 loss 更强”提升为真正可竞争的 decode 候选

结果：

- `len640 lr=4e-5 peak240 ckpt225 @ 4 / 0.7 / 384`
  - `18.2427 / 11.1410 / 29.8712`
- `len640 lr=4e-5 peak240 ckpt230 @ 4 / 0.7 / 384`
  - `18.2495 / 11.1487 / 29.8730`
- `len640 lr=4e-5 seed43 ckpt155 @ 4 / 0.7 / 384`
  - `17.2665 / 10.2320 / 29.1372`
- `bs32_len512 lr=4e-5 ckpt75`
  - winner 为 `lp=0.7`
  - `16.7370 / 9.6038 / 29.1683`

selector winner：

- `len640 lr=4e-5 peak240 ckpt230 @ 4 / 0.7 / 384`

winner `diag64`：

- overall：
  - `11.5391 / 6.8934 / 19.3158`
- reconstructed：
  - `16.3310 / 9.6932 / 27.5143`
- reconstructed health：
  - `empty=0.0`
  - `copy=0.0`
  - `pred_shorter_than_half_ref=0.0`

结论：

- `lr=4e-5 peak240` 的 decode 最优点确实在 `225-230`，并且 `230` 略优于 `225`。
- 但这条 selector winner 仍未超过当前正式主线 `seg5 ckpt250 @ 4 / 0.7 / 384`。
- `seed43` 没有在 decode 侧复现主 seed 的强度。
- `bs32_len512 @ lr=4e-5` 在训练 loss 上更强，但 decode 侧仍显著弱于 `len640` 主线。

## 13. 清洗策略落地状态

当前已从“仅 Tier-0”推进到“Tier-0 + 最小可执行 Tier-1”。

### 13.1 已落地规则

- `cleaning/normalize.py`
  - Tier-0:
    - `t0_unicode_nfc`
    - `t0_remove_invisible`
    - `t0_whitespace_normalize`
    - `t0_unknown_visible_char_alert`
  - Tier-1:
    - `t1_gap_tag_canonicalize`
    - `t1_brace_whitespace_canonicalize`
- 配置：
  - `cleaning/configs/cleaning.t0.yaml`
  - `cleaning/configs/cleaning.t1.yaml`
- 说明：
  - `t0_unknown_visible_char_alert` 只告警、不改写
  - Tier-1 当前只处理：
    - `<gap>/<big_gap>` 的大小写/空格/下划线规范化
    - 花括号内边界空白规整

### 13.2 仍未落地

- `„` 的权威定性与专门冻结证据
- Tier-1/Tier-2 的完整 rule registry / 单测闭环
- 更高风险的 semantic merging

### 13.3 当前数据清洗与对照准备进度

- 数据清洗窗口：
  - `clean-tier01`
- 当前状态：
  - `t0_train.csv / t0_test.csv`：已生成
  - `t1_train.csv / t1_test.csv`：已生成
  - `data/processed_byt5_chunks_align_gc_cost14_t0`：已生成
  - `data/processed_byt5_chunks_align_gc_cost14_t1`：已生成

结论：

- Tier-1 现在已经进入可执行状态，但仍是“最小结构规范化版”，不是完整宪法实现。
- 下一步应先看 `t0/t1` 清洗集在同一训练 recipe 下的对照结果，再决定是否把 Tier-1 并入正式主线。

## 14. `lr=4e-5` 更细早停窗口补充

方法：

- 在已有 `peak240` 结果基础上，再向局部窗口补点：
  - 主 seed：`checkpoint-220`、`checkpoint-235`
  - `seed43`：`checkpoint-150`、`checkpoint-160`
- decode 固定：
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`
  - `max_val_samples=32`

结果：

- `ckpt220`: `18.2021 / 11.1151 / 29.8080`
- `ckpt225`: `18.2427 / 11.1410 / 29.8712`
- `ckpt230`: `18.2495 / 11.1487 / 29.8730`
- `ckpt235`: `18.2495 / 11.1487 / 29.8730`
- `seed43 ckpt150`: `17.3494 / 10.2365 / 29.4045`
- `seed43 ckpt160`: `17.3473 / 10.3221 / 29.1538`

结论：

- `lr=4e-5` 的局部最优基本锁在 `230-235`。
- 但这一带的最好 anchor 仍未超过当前正式主线 `seg5 ckpt250 @ 4 / 0.7 / 384`。
- `seed43` 继续证明“早 checkpoint 有效”，但 decode 强度仍弱于主 seed。

## 15. `T0-clean vs T1-clean` 大实验结果

目标：

- 在当前正式赢家
  - `len640 seg5 ckpt250 @ 4 / 0.7 / 384`
  已闭环的前提下，验证最小 Tier-1 清洗是否能在相同训练制度下带来新的稳定增益。

当前并行训练：

- `gate-t0clean-train`
  - `continue_s4_bs24_len640_lr4e5_peak240_t0clean_gate.yaml`
- `gate-t1clean-train`
  - `continue_s4_bs24_len640_lr4e5_peak240_t1clean_gate.yaml`

共同 recipe：

- `len640`
- `bs24`
- `lr=4e-5`
- `max_steps=240`
- `eval/save=5`
- `LoRA q/v r=8`

Gate 队列：

- `gate-cleaning-queue`
- 训练完成后自动比较：
  - `ckpt225`
  - `ckpt230`
  - `ckpt235`
- selector decode 固定：
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`
  - `max_val_samples=32`
- 若 `T1-best geom >= T0-best geom + 0.15`
  - 自动补 `diag64`
  - 自动补单点 full-val `decode + diagnose`
- 若未达到门槛
  - 停在 selector

训练结果：

- `T0-clean`
  - best checkpoint: `checkpoint-165`
  - best metric: `0.794761`
- `T1-clean`
  - best checkpoint: `checkpoint-165`
  - best metric: `0.794761`

selector 结果（`ckpt225/230/235 @ 4 / 0.7 / 384 / anchor32`）：

- `T0-clean`
  - `ckpt225`: `6.1271 / 2.5117 / 14.9468`
  - `ckpt230`: `6.1271 / 2.5117 / 14.9468`
  - `ckpt235`: `6.1271 / 2.5117 / 14.9468`
- `T1-clean`
  - `ckpt225`: `6.1271 / 2.5117 / 14.9468`
  - `ckpt230`: `6.1271 / 2.5117 / 14.9468`
  - `ckpt235`: `6.1271 / 2.5117 / 14.9468`

Gate 结果：

- `t0_best = 6.1271`
- `t1_best = 6.1271`
- `T1 - T0 = 0.0000`
- 未达到 `+0.15 geom` 门槛
- 自动停止在 selector，未进入 `diag64` 与 full-val

结论：

- 最小 Tier-1 清洗在当前 competition-only 训练条件下没有给出增益。
- 当前主线不应切到 `T1-clean`。
- 这轮结果也说明：如果要继续升级清洗，必须先引入更能触发结构规则的数据面，或先做更严格的规则收益验证。

## 16. 架构升级 Probe 轮（2026-03-09）

matched baseline：
- `seg5 ckpt250 @ 4 / 0.7 / 384 / anchor32`: `18.3354 / 11.1352 / 30.1913`

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

总判断：

- 三条架构升级主线都在 `probe` 阶段失败。
- `ByT5-base` 与 `mT5-base` 在当前数据面和任务形式下都没有接近 matched baseline。
- 当前不应进入 `W1/W2/W3` 或 `F1/F2/F3`。

## 17. 当前正式 Winner 误差桶报告

已补一份独立报告：

- [winner_error_bucket_report_2026-03-09.md](/workspace/deep-past-/docs/winner_error_bucket_report_2026-03-09.md)

核心结论：

- 当前 winner 的主要问题集中在：
  - 长样本
  - 多 chunk parent
  - 结构标记丰富样本
- 主要失败模式不是空输出，而是：
  - 重复模板化
  - 贴长度上限
- 这进一步支持下一轮主线优先切到：
  - `parent-aware / parent-packed`

## 18. 任务形式升级轮纪律

在架构升级轮给出负证据后，下一阶段主线已经切换为“任务形式升级轮”。

执行口径见：

- [next-step_taskform_discipline_2026-03-09.md](/workspace/deep-past-/docs/next-step_taskform_discipline_2026-03-09.md)

本轮不再优先：

- 更细 decode 微调
- 更大 backbone 冷起
- Tier-1 再次主线化

而优先：

- `parent-packed / parent-windowed`
- `hard-case replay / bucket curriculum`
- `proxy / published_texts controlled mix`

补充口径（基于首轮 taskform 执行后的重构）：

- `P1` 保留为任务形式升级主线
- `P2` 已重构为 `replay + hardselector` 口径
  - 训练集保留 replay
  - 验证集切到 hardest-bucket selector 子集
- `P3` 已从主队列降级为 side-track
  - 保留数据构建与候选池审计
  - 不再进入默认 probe 主队列

执行纪律以更新后的：

- [next-step_taskform_discipline_2026-03-09.md](/workspace/deep-past-/docs/next-step_taskform_discipline_2026-03-09.md)

为准。

## 19. Taskform Probe v2 最终结果与下一动作

独立报告：

- [taskform_upgrade_probe_round_v2_2026-03-09.md](/workspace/deep-past-/docs/taskform_upgrade_probe_round_v2_2026-03-09.md)

最终结论：

- `P1 = parent-packed / parent-windowed`
  - matched baseline: `9.0096`
  - best anchor: `ckpt200 -> 9.5190`
  - `diag32`: `empty=0.00%`, `copy=0.00%`, `short=18.75%`, `unique=93.75%`
  - 判断：`review_to_wlite`
- `P2 = replay25_hardselector`
  - matched baseline: `8.2732`
  - best anchor: `ckpt200 -> 9.2412`
  - `diag32`: `empty=0.00%`, `copy=0.00%`, `short=6.25%`, `unique=96.88%`
  - 判断：`review_hold`
- `P3`
  - side-track，不进主队列

本机当前推荐顺序：

1. 仅让 `P1` 进入 `W-lite`
2. `P2` 保留为 reserve line
3. `P3` 继续 side-track

`W-lite` 的默认执行口径：

- `matched baseline anchor64`
- 从 `P1 ckpt200` 继续训练到总步数 `400`
- 比较 `ckpt250 / 300 / 350 / 400 @ anchor64`
- line winner `diag64`

`W-lite` 之后由脚本自动 gate：

- 若分数和健康同时过线，则进入 `taskform promote-lite`
- 否则停在 `review`，不直接推到 `promote`
