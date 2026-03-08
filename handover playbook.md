# Deep Past Cloud Handover Playbook

> 更新时间：2026-03-03  
> 交接基线分支：`main`  
> 远程仓库：`https://github.com/hirman742/deep-past-.git`  
> 当前最新提交：`cfbcebc`（`Add GC short-align pipeline and n-best rerank diagnostics`）

---

## 0. 交接目标（一句话）

将当前已验证有效的主线  
**ByT5 + chunk + parent重建评测 + GC高置信短对齐stage2 + n-best rerank**  
在云算力上规模化执行，优先完成 `S0~S6`（见第7节），并保持评测口径一致与实验可追溯。

---

## 1. 当前仓库状态总览

### 1.1 代码状态
- 分支：`main`
- 工作区：干净（可直接交接）
- 最近关键提交链：
  - `cfbcebc`：新增 GC 对齐脚本 + n-best rerank + round4报告
  - `92a32f7`：E8诊断防护（original-only重建、no-replacement）
  - `193757e`：ByT5 chunk stage2 工具与 round2 报告
  - `b80ad63`：chunk化 ByT5 主流水线

### 1.2 环境依赖（已锁版本）
- 环境文件：`env.yml`
- Conda 环境名：`deeppast-cleaning`
- 关键版本：
  - `python=3.11`
  - `pytorch=2.10.0`
  - `transformers==5.2.0`
  - `peft==0.18.1`
  - `accelerate==1.12.0`
  - `sacrebleu==2.6.0`

### 1.3 被 `.gitignore` 忽略的关键目录（交接必读）
- `data/interim/`
- `data/processed*/`
- `runs/*`（绝大部分实验产物）

**含义**：仅克隆仓库无法复现实验，需要额外同步数据与产物（见第6节与第8节）。

---

## 2. 与《云算力冲分实验方案》对齐说明

外部方案（`C:\Users\Hirman\obsidion\kaggle\云算力冲分实验方案.md`）核心结论是：
- 不改主路线；
- 云上重点做：`长训 + len512 + 大候选池 rerank + GC池扩容 + 2fold→5fold`；
- 保持口径：`parent重建 + aggregate_original_only=true`。

本仓库当前实现状态：
- ✅ `aggregate_original_only` 已在诊断/网格/训练重建链路支持并默认可用；
- ✅ `n-best rerank` 已可执行（`scripts/eval_nbest_rerank.py`）；
- ✅ `GC对齐` 已可执行（`scripts/build_short_aligned_pairs_galechurch.py`）；
- ✅ `stage2 curriculum` 配置已就位（`configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml`）；
- ⏳ 尚未开始云上 `len512 + 20k/40k` 主线长训（这是下一接手人主任务）。

---

## 3. 实验进展与结论（给接手人的“当前最佳认知”）

## 3.1 阶段性结果（按时间）

| 阶段 | 关键文件 | 结论 |
|---|---|---|
| E6 chunk冒烟 | `docs/e6_byt5_chunked_round_report.md` | 彻底解决截断学习（训练数据 trunc 到 0/0），方向可行 |
| E7/E8 round2 | `docs/byt5_chunk_round2_execution_report_2026-03-01.md` | E7 stage1 明显优于 E8（当时短对齐池质量不足） |
| E8 round3诊断 | `docs/e8_diagnostic_round3_2026-03-02.md` | 锁定 E8下滑根因：fallback噪声 + 采样策略；已加防护 |
| GC round4 | `docs/byt5_gc_round4_execution_report_2026-03-02.md` | GC短对齐+stage2+rerkank 带来持续正增益（小幅但稳定） |

## 3.2 当前“可用最好链路”指标快照（fold0）

- E9 固定解码重建：`geom=10.0616`  
  文件：`runs/E9_BYT5_STAGE1_R8_QV_CONT8K_fold0/diagnostics/val_diagnostic_summary_step5000_fixeddecode_384.json`
- E9 + rerank 重建：`geom=10.1286`  
  文件：`runs/E9_BYT5_STAGE1_R8_QV_CONT8K_fold0/diagnostics/nbest_rerank_summary_step5000_beam4_n4_lp1p2_m384.json`
- E10（GC relaxed stage2）固定解码重建：`geom=10.1777`  
  文件：`runs/E10_BYT5_STAGE2_GC_RELAXED_fold0/diagnostics/val_diagnostic_summary_step1000_fixeddecode_384.json`
- E10 + rerank 重建：`geom=10.2125`（当前本地最佳）  
  文件：`runs/E10_BYT5_STAGE2_GC_RELAXED_fold0/diagnostics/nbest_rerank_summary_step1000_beam4_n4_lp1p2_m384.json`

**关键结论**：  
GC 高置信短对齐 + 小步 stage2 + rerank 已被证明“正收益”，但增益还在小幅区间，下一阶段主要靠云端长训和候选池放大。

---

## 4. 关键脚本与配置地图（接手人最先看的文件）

### 4.1 训练与评测主链路
- 训练：`scripts/train_mt5_lora.py`
- 诊断：`scripts/diagnose_val_outputs.py`
- 解码网格：`scripts/eval_decode_grid.py`
- 推理：`scripts/infer_mt5_lora.py`

### 4.2 新增增益模块
- n-best rerank：`scripts/eval_nbest_rerank.py`
- GC短对齐构建：`scripts/build_short_aligned_pairs_galechurch.py`

### 4.3 当前重点配置
- E9 stage1主配置：`configs/byt5_small_lora_chunked_stage1_r8_qv_cont8k.yaml`
- E9 parent逆采样对照：`configs/byt5_small_lora_chunked_stage1_r8_qv_cont8k_inv.yaml`
- E10 stage2配置：`configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml`

---

## 5. 统一评测口径（禁止偏离）

所有实验必须执行以下口径，否则结果不可比：
- 指标级别：**parent重建**
- 过滤策略：`aggregate_original_only=true`
- 指标集合：
  - `geom / bleu / chrfpp`
  - `empty_prediction_ratio_pct`
  - `copy_source_ratio_pct`
  - `pred_shorter_than_half_ref_ratio_pct`
  - `pred_tok_p95 / ref_tok_p95`
- 签名记录：SacreBLEU/chrF++ signature（已在 summary 中落盘）

---

## 6. 云端交接执行：先做什么

### 6.1 仓库克隆
```bash
git clone https://github.com/hirman742/deep-past-.git
cd deep-past-
git checkout main
git pull --ff-only
```

### 6.2 环境创建（Linux云机推荐）
```bash
conda env create -f env.yml
conda activate deeppast-cleaning
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 6.3 缓存建议
```bash
export HF_HOME="$PWD/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
```

### 6.4 数据同步（必须）
由于 `.gitignore`，下面目录需从本地机或对象存储同步到云端：
- `data/interim/t0_train.csv`
- `data/interim/t0_test.csv`
- `data/processed_byt5_chunks/`
- （可选）`data/processed_byt5_chunks_align_gc_relaxed/`（如直接跑stage2）

建议命令（本地→云端）：
```bash
rsync -avP data/interim/ <cloud_user>@<cloud_host>:<repo_path>/data/interim/
rsync -avP data/processed_byt5_chunks/ <cloud_user>@<cloud_host>:<repo_path>/data/processed_byt5_chunks/
```

---

## 7. 云算力冲分 Runbook（按优先级）

## 7.1 S0：云端冒烟
```bash
python scripts/train_mt5_lora.py \
  --config configs/byt5_small_lora_chunked_stage1_r8_qv_cont8k.yaml \
  --fold 0 \
  --max-train-rows 256 \
  --max-val-rows 128 \
  --max-steps 100 \
  --eval-steps 50 \
  --skip-final-predict
```
通过条件：能产出 `run_summary.json` 且 eval 指标非空，诊断无大面积空输出。

## 7.2 S1：主线长训（云端核心）
按方案新增两个云版配置（`len512`，`lr=2e-4/1e-4`），然后执行：
```bash
python scripts/train_mt5_lora.py --config configs/cloud_stage1_len512_lr2e4.yaml --fold 0 --max-steps 20000 --eval-steps 500 --skip-final-predict
python scripts/train_mt5_lora.py --config configs/cloud_stage1_len512_lr1e4.yaml --fold 0 --max-steps 20000 --eval-steps 500 --skip-final-predict
```
对比标准：parent重建 `geom` + 健康指标稳定性。

## 7.3 S2：大候选池 rerank
```bash
python scripts/eval_nbest_rerank.py \
  --config configs/cloud_stage1_len512_lr2e4.yaml \
  --fold 0 \
  --checkpoint-dir runs/<stage1_run>_fold0/best_model \
  --tag stepBEST_beam8_n8_lp1p2_m512 \
  --max-rows 900 \
  --num-beams 8 \
  --num-return-sequences 8 \
  --length-penalty 1.2 \
  --no-repeat-ngram-size 3 \
  --max-new-tokens 512 \
  --aggregate-original-only
```

## 7.4 S3：GC池扩容（目标10k+）
```bash
python scripts/build_short_aligned_pairs_galechurch.py \
  --config configs/cloud_stage1_len512_lr2e4.yaml \
  --mix-ratio 0.5 \
  --no-allow-replacement \
  --max-align-cost 1.4 \
  --report-json runs/GC_pool_scale_report_cost14.json
```
关键约束：`used_replacement=false` 不得破坏。

## 7.5 S4：stage2 curriculum（从强stage1初始化）
```bash
python scripts/train_mt5_lora.py \
  --config configs/byt5_small_lora_chunked_stage2_gc_curriculum.yaml \
  --fold 0 \
  --init-adapter-dir runs/<strong_stage1>_fold0/best_model \
  --max-steps 4000 \
  --eval-steps 500 \
  --max-val-rows 900 \
  --skip-final-predict
```
通过条件：stage2 后不能压低 stage1-only 可比口径结果。

## 7.6 S5/S6：2fold验证 → 5fold OOF
先 fold0+fold1 检查泛化，再扩到 5fold 才做最终提交。

---

## 8. 交接后首日 Checklist

- [ ] 云机 `git pull` 到 `cfbcebc` 之后无冲突  
- [ ] `conda activate deeppast-cleaning` 后核心包可 import  
- [ ] `torch.cuda.is_available()==True`  
- [ ] `data/interim` 与 `data/processed_byt5_chunks` 已同步  
- [ ] S0 冒烟通过  
- [ ] 输出目录命名规范（如 `runs/CLOUD_*`）已确认  
- [ ] 评测口径 `aggregate_original_only=true` 已确认

---

## 9. 高风险坑位与止损规则

### 9.1 常见坑位
- 数据没同步全：训练直接空跑或报文件缺失
- 评测口径漂移：没用 parent重建或没开 original-only
- 不小心启用 replacement：短对齐池被重复样本污染
- 训练期过度频繁做重解码：GPU时间被评测吞噬

### 9.2 止损规则（建议执行）
- 任一新改动若使 `reconstructed geom` 连续两次显著下降，立即回滚
- stage2 若不能稳定超过强 stage1，先停 stage2，回主线长训
- rerank 若增益趋零，优先扩大候选池多样性，再改权重

---

## 10. Git 提交与推送：命令 + 必要信息

## 10.1 必要身份信息（首次云机必须配置）
```bash
git config --global user.name "Hirman"
git config --global user.email "<你的GitHub注册邮箱>"
git config --global core.autocrlf input
```

校验：
```bash
git config --global --get user.name
git config --global --get user.email
git remote -v
```

## 10.2 推荐分支策略
- 主分支：`main`
- 云实验分支命名：`exp/cloud-<date>-<topic>`
  - 示例：`exp/cloud-20260303-stage1-len512`

创建并推送分支：
```bash
git checkout -b exp/cloud-20260303-stage1-len512
git push -u origin exp/cloud-20260303-stage1-len512
```

## 10.3 标准提交流程（每次实验）
```bash
git status
git add <改动文件1> <改动文件2> ...
git commit -m "exp: stage1 len512 lr2e-4 fold0 to 20k steps (parent reconstructed eval)"
git push origin <当前分支名>
```

## 10.4 推荐提交信息模板
```text
<type>: <one-line summary>

Context:
- why this change

Changes:
- config/script/report paths

Results:
- reconstructed geom/bleu/chrfpp
- health metrics

Next:
- what to run next
```

`type` 推荐：`exp` / `fix` / `feat` / `docs` / `chore`

## 10.5 合并到 main（如走 PR）
```bash
git checkout main
git pull --ff-only
git merge --no-ff <实验分支>
git push origin main
```

---

## 11. 当前已知“最优接力点”（供接手人直接开始）

从以下点继续最省时间：
- 代码基线：`cfbcebc`
- 阶段：E10 已证明正增益，但幅度有限，下一步应把预算投入云端 stage1 长训与 len512
- 第一优先任务：执行 S1（20k steps, lr sweep, len512）

---

## 12. 附：参考文件索引

- 总体说明：`README.md`
- round2：`docs/byt5_chunk_round2_execution_report_2026-03-01.md`
- round3：`docs/e8_diagnostic_round3_2026-03-02.md`
- round4：`docs/byt5_gc_round4_execution_report_2026-03-02.md`
- 方案来源：`C:\Users\Hirman\obsidion\kaggle\云算力冲分实验方案.md`

