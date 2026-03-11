# Cloud Stage2 Tier1 Cleaning Gate Report 2026-03-09

## 目标

在当前正式赢家

- `len640 seg5 ckpt250 @ beam=4 / lp=0.7 / max_new_tokens=384`

已经闭环的前提下，验证最小 Tier-1 清洗是否能在同一训练制度下带来新的稳定增益。

## 实验设计

并行训练两条线：

- `T0-clean`
- `T1-clean`

共同 recipe：

- `len640`
- `bs24`
- `lr=4e-5`
- `max_steps=240`
- `eval/save=5`
- `LoRA q/v r=8`

## Gate

1. 训练结束后，只比较：
   - `ckpt225`
   - `ckpt230`
   - `ckpt235`
2. selector decode 固定：
   - `beam=4`
   - `lp=0.7`
   - `max_new_tokens=384`
   - `max_val_samples=32`
3. 若 `T1-best geom >= T0-best geom + 0.15`
   - 先补 `diag64`
   - 再跑单点 full-val `decode + diagnose`
4. 若不满足
   - 停在 selector，不进入 full-val

## 估时

- 两条训练并行：约 `12-18` 分钟
- selector 六个 anchor decode：约 `10-15` 分钟
- 若 T1 过 gate：
  - `diag64`: `6-10` 分钟
  - full-val decode: `85-100` 分钟
  - full-val diagnose: `75-90` 分钟

总计：

- 若 T1 不过 gate：约 `25-35` 分钟
- 若 T1 过 gate：约 `3小时 - 3小时40分`

## 产物路径

- 训练配置：
  - `runs/STEER/generated_configs/continue_s4_bs24_len640_lr4e5_peak240_t0clean_gate.yaml`
  - `runs/STEER/generated_configs/continue_s4_bs24_len640_lr4e5_peak240_t1clean_gate.yaml`
- 队列脚本：
  - `tmp/run_tier_cleaning_gate_20260309.sh`
  - `tmp/launch_tier_cleaning_gate_20260309.sh`

## 当前状态

本轮已完成，未进入 `diag64` 与 full-val 阶段。

训练结果：

- `T0-clean`
  - best checkpoint: `checkpoint-165`
  - best metric (`eval_loss`): `0.794761`
- `T1-clean`
  - best checkpoint: `checkpoint-165`
  - best metric (`eval_loss`): `0.794761`

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
- 队列已自动 `SKIP T1 full-val gate not passed`

当前日志：

- `logs/steer_continue_s4_bs24_len640_lr4e5_peak240_t0clean_gate_train.log`
- `logs/steer_continue_s4_bs24_len640_lr4e5_peak240_t1clean_gate_train.log`
- `logs/tier_cleaning_gate_20260309.log`

结论：

- 最小 Tier-1 清洗在当前 competition-only 训练条件下没有给出可见增益。
- 就这一轮证据看，Tier-1 目前不应进入正式主线。
