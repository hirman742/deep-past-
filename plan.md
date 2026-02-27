```md
# 8GB 显存专用极限配置版（可一键复制）

> 目标：在 8GB 显存上稳定跑通训练/评估/推理/提交，并能做最小 ablation。

---

## 0. 统一硬约束（所有实验默认）

- fp16：开启
- gradient checkpointing：开启
- 只训 LoRA：开启（冻结 backbone）
- dataloader：尽量使用 dynamic padding（按 batch 内最长 pad）
- 日志必记：显存峰值、step time、tokens/s、trainable params、截断比例

---

## 1. 数据与 CV（先固化，再训练）

### 1.1 预处理脚本要求（preprocess.py）
- 归一化规则全部做成开关（便于消融）
- 输出：
  - `train_proc.csv`
  - `folds.csv`（5 fold，GroupKFold 优先）

### 1.2 截断比例统计（必须）
训练前先统计：
- `src_len_p95 / p99`
- `tgt_len_p95 / p99`
- 计算在 `max_source_length/max_target_length` 下的截断比例（%）

> 截断比例超过 3%：先降清洗损失或调 max_length，而不是硬训。

---

## 2. mT5-small + LoRA（8GB 起步模板）

### 2.1 默认推荐（稳）
- `model = google/mt5-small`
- `batch_size = 4`（OOM 就降 2）
- `grad_accum = 8`（等效 batch = 32）
- `max_source_length = 256`（先保守）
- `max_target_length = 192`
- `lr = 2e-4`（LoRA-only 常用起步）
- `warmup_ratio = 0.03`
- `weight_decay = 0.0~0.01`
- `epochs = 3`（先短跑闭环）
- `eval_every = 0.25 epoch`（至少每 epoch 评一次）

### 2.2 LoRA（极限实用配置）
- `target_modules = ["q_proj","v_proj"]`
- `r = 4`
- `lora_alpha = 4`
- `lora_dropout = 0.0`（需要正则再调到 0.05）
- `bias = "none"`

> 显存紧张时优先：`r=2` 或只注入 `q_proj`（牺牲一点效果换稳定）。

### 2.3 解码（推理省显存/省时）
- `num_beams = 4`
- `length_penalty = 1.0`
- `max_new_tokens = max_target_length`
- 推理 batch：尽量大（不会 OOM 的前提下），通常 16~64

---

## 3. ByT5-small + LoRA（8GB 必须降档）

> ByT5 序列长，显存与时间都更吃紧，必须按“能跑就行”模式起步。

### 3.1 推荐起步
- `model = google/byt5-small`
- `batch_size = 1`
- `grad_accum = 16~32`（等效 batch 16~32）
- `max_source_length = 512`（先保守，截断太高再加）
- `max_target_length = 256`
- `lr = 2e-4`
- `epochs = 2~3`

### 3.2 必开项
- gradient checkpointing：开
- dynamic padding：开
- 训练时尽量关闭不必要的缓存（如 `use_cache=False`）

### 3.3 OOM 处理顺序（只按这个顺序改）
1. `batch_size -> 1`
2. `max_source_length -> 384 -> 256`
3. `r -> 2 -> 1`
4. `grad_accum` 上调保持等效 batch 不太小

---

## 4. 8GB “必做”最小闭环计划（一天内完成）

1. 固化 `train_proc.csv + folds.csv`
2. mT5-small + LoRA：fold0 跑 1~2 epoch，能产出 submission
3. ByT5-small + LoRA：fold0 跑 1 epoch 对照
4. 记录：BLEU、chrF++、几何均值 + 截断比例 + 训练吞吐

---

## 5. 8GB 常见坑与止损

- 训练很稳但分不涨：优先做归一化/解码网格，不要先扩模型
- 截断比例高：先降 max_length 以外的损失（清洗/拆分/句子化），再上 max_length
- ByT5 不涨且慢：及时止损，回到 mT5 + 数据策略

---

---


# 冲榜强化版实验路线图（从 baseline 到 leaderboard）

> 目标：用最少试验次数、最强“单变量”迭代，尽快冲上有效分数段。

---

## 阶段 A：强 baseline（必须先完成）

### A1. 单折闭环（mT5 + LoRA）
- 固定 fold0 + seed42
- 固定预处理版本 v0
- 固定解码：beam=4, lp=1.0
- 跑出可提交模型与日志

### A2. 5-fold 训练（只在 A1 可靠后）
- 每折保存 best ckpt
- 预测时做 5 折 ensemble（logits 或生成结果投票/选择最高置信）

> 如果算力不够：先 3 折 ensemble（0/1/2）。

---

## 阶段 B：最赚钱的 3 个方向（按优先级）

### B1. 解码网格（ROI 极高，成本极低）
固定模型不变，扫：
- `num_beams: 4, 5, 8`
- `length_penalty: 0.8, 1.0, 1.2, 1.4`
- （可选）`no_repeat_ngram_size: 2, 3`

输出：
- 每组参数的 BLEU/chrF++/总分
- 选择在 CV 上最稳的，不要盲追单折峰值

---

### B2. 归一化/清洗消融（高 ROI）
只改一个开关，其余全部固定。

建议消融顺序：
1. 空白/重复空格归一
2. 标点与分隔符处理（保守 vs 激进）
3. determinatives / 特殊标记处理（保留 vs 规范化）
4. 仅对 source 做规则 vs source+target 同步规则

判定：
- 如果 BLEU 上升但 chrF++ 下降（或反之），不要急着否定，先看总分与错误类型。

---

### B3. 外部 ORACC 混训（高上限，但风险大）
先做 10% 试水：
- 只在 mT5 最佳配置上加 ORACC
- 严格去泄漏（test/source 精确重复剔除）
- 验证集只用竞赛 fold

方向正确再扫比例：
- 10% / 30% / 50%

止损信号：
- CV 上涨但公榜不涨：立刻降比例或做子域筛选

---

## 阶段 C：ByT5 与 tokenizer 方向（定位“符号噪声”收益）

### C1. 同预算 A/B（ByT5 vs mT5）
固定：
- fold、seed、训练步数、LoRA、解码
只允许为显存调整 batch/accum/max_length。

如果 ByT5 明显更好：
- 下一步将清洗策略改为“更保守”（保留符号信息）

---

## 阶段 D：LoRA 强化（在数据与解码稳定后再做）

只做 3 点：
- r: 2 / 4 / 8
- dropout: 0.0 / 0.05
- target_modules: (q,v) vs (q,k,v,o)（显存允许再做）

> 未先稳定数据与解码就扫 LoRA，通常浪费试验预算。

---

## 阶段 E：最终融合（冲榜阶段）

- 选择 2~3 个最强“配置族”：
  - （例）mT5 + best decode + best norm
  - （例）mT5 + ORACC(10~30%) + best decode
  - （例）ByT5 + best norm（若确有收益）
- 做 3 折或 5 折 ensemble
- 输出多份 submission（不同 family），用公榜选择最终方案

---

## 冲榜节奏建议（最少试验次数）

1) mT5 baseline（1 折）  
2) 解码网格（固定模型）  
3) 清洗消融（2~4 次）  
4) ORACC 10%（1 次）  
5) ORACC 比例 sweep（2 次）  
6) 3 折 ensemble（1 次）  
7) 只在最后做 LoRA r/target_modules 小扫（2~3 次）

---

## 交付物（保证可复现与可回滚）

- `configs/`：每个实验一个 yaml（命名含版本号）
- `logs/`：记录 score、截断比例、显存峰值、decode 参数
- `external_data_manifest.md`：外部数据版本与许可
- `rules_checklist.md`：规则条款摘录
- `submissions/`：每次提交保存对应 config 与 ckpt hash
```

