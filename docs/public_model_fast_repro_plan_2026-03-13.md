# 公开高分模型快速复现规划
## 基于 2026-03-13 纪律重排

## 0. 目的

这份规划用于承接：

- [next_step_discipline_2026-03-13.md](/workspace/deep-past-/docs/next_step_discipline_2026-03-13.md)

目标不是继续修当前 winner，而是明确：

- 当前仓库真实现状
- 公开高分模型已验证出的关键信号
- 我们接下来“最快复现高分/健康”的路径

## 1. 当前现状

### 1.1 仓库状态

当前 `/workspace/deep-past-` 不是完整训练工作树，而是：

- 代码、配置、报告、纪律文档仍在
- 关键 winner 资产大量封存在 `release/*.tgz`
- 当前工作树中的 `runs/` 不完整
- `official metric bridge` 仍缺失

这意味着：

- 当前仓库仍足够支持分析、对照、重构流程
- 但不适合继续把现有工作树当作完整训练态向前滚动

### 1.2 当前主线的真实水平

当前仓库最近主线的核心形态是：

- `google/byt5-small`
- `LoRA(q, v, r=8, alpha=16)`
- retrieval augmentation
- local winner surgery
- chooser / replay / fallback / repair

代表性结论已经在近期文档中写明：

- 本地 `A18 / A20` 可以把 local reconstructed geom 推到 `21.x`
- 但真实外部信号是：
  - `Version 6 public score = 7.6`
- `official-like` 与 `public leaderboard` 之间没有正式 bridge
- 当前仓库这版 `MBR` formulation 为负
- 当前仓库这版 `TAPT / external continue` 已证伪

因此当前主线的问题不是“没有工程”，而是：

- 主干偏弱
- 本地代理指标权重过高
- 后处理与候选修补过早占据主序

### 1.3 公开高分模型带来的新证据

已解包并本地验证的公开对象：

- `byt5 akkadian mbr · default`

公开包实物特征：

- 完整模型导出，不是 adapter-only
- `model.safetensors ≈ 2.2G`
- `T5ForConditionalGeneration`
- `ByT5Tokenizer`
- `d_model = 1536`
- `d_ff = 3968`
- `num_layers = 18`
- `num_decoder_layers = 6`
- `num_heads = 12`

默认生成配置：

- `num_beams = 8`
- `repetition_penalty = 1.1`

这和我们当前主线形成直接对照：

- 它是更强的完整 ByT5 主干
- 我们是 `byt5-small + LoRA` 的轻量 continue

### 1.4 本地验证结果

已在隔离工作区对公开模型做本地 `fold0` 验证：

- 工作区：
  - `/workspace/incoming/public_eval_byt5_akkadian_mbr`
- 结果：
  - `BLEU = 30.8114`
  - `chrF++ = 49.6247`
  - `geom = 39.1025`
  - `elapsed_seconds = 784.98`
  - `device = cuda`

输出健康：

- `empty_prediction_ratio_pct = 0.0`
- `copy_source_ratio_pct = 0.0`
- `pred_shorter_than_half_ref_ratio_pct = 10.86`
- `unique_prediction_ratio_pct = 99.36`

这说明：

- 公开高分路线的核心优势首先来自强单模本体
- 它不是靠局部 chooser / fallback 才站住
- 它的输出健康度也明显不是当前仓库近期那种“高度依赖修补链”的状态

## 2. 我们当前困境的准确判定

### 2.1 不是 family 选错

当前证据反而更支持：

- `ByT5` 路线是对的
- byte-level 架构对 Akkadian transliteration 仍然合适

当前错的不是：

- `ByT5`
- `MBR`
- rerank 这些大类本身

当前错的是：

- 在较弱主干上，过早把局部修补链当成主线

### 2.2 不是工程不够，而是工程重心错了

仓库在这些方面并不弱：

- cleaning / normalization
- chunking / GC 扩容
- retrieval augmentation
- candidate pool
- replay/repair/chooser
- n-best / rerank / MBR probe

但问题在于：

- 这些工程大多围绕 `byt5-small + LoRA` 主干在补短板
- 不是围绕“强单模 first”在放大上限

### 2.3 当前主线的核心失配

当前最准确的技术判定是：

1. 主干偏弱
2. decode 约束偏晚
3. `MBR` 用在了错误顺序
4. `official bridge` 缺失下仍继续用 local proxy 驱动主序

换句话说：

- 我们不是缺一个新规则
- 我们缺的是“更强主干 + 更正确的顺序”

## 3. 快速复现高分/健康的最优路径

### 3.1 总原则

从现在开始，路线改成：

1. 强单模优先
2. decode 健康优先
3. official-compatible local eval 固定
4. rerank / MBR 放到第二阶段

不再使用的旧主序是：

1. 小主干先跑
2. local geom 上涨就继续
3. 输出坏了再靠 fallback / chooser / repair 回补

### 3.2 第一阶段：公开模型复现实验补全

第一阶段目标不是训练，而是把已知高分对象看透。

必须完成：

1. 同一公开模型做 decode 对照
   - `beam=8 / repetition_penalty=1.1`
   - `beam=4 / lp=0.7 / max_new_tokens=384`
   - 如有必要再加：
     - 去掉 `repetition_penalty`
2. 固定 official-compatible local eval 口径
   - corpus BLEU
   - corpus chrF++
   - geometric mean
3. 判断公开模型优势主要来自：
   - 主干
   - decode
   - 还是二者叠加

这一步的意义是：

- 先把“为什么它高分”拆清楚
- 不要盲目开始重训

### 3.3 第二阶段：新主线只开“强单模复现线”

新的训练主线应当只允许这一类：

- 更大 ByT5 主干
- 基于现有清洗和数据资产
- 目标是复现“强单模 + 健康输出”

不应继续作为主线的方向：

- `A20` 类阈值扫
- chooser 再套规则
- replay band 继续窄扫
- term-aware patch 同质扩臂
- 当前 formulation 的 `MBR` 重开

### 3.4 第三阶段：把我们已有资产重新摆位

#### 保留并复用

- cleaning 规则
- raw/train/test 数据治理
- GC / short-aligned 扩容流程
- 本地评测与日志体系
- retrieval 资产

#### 降级为第二阶段使用

- replay
- chooser
- fallback
- term patch
- rerank
- MBR

这些资产的正确角色应是：

- 强单模站住后再接入
- 用于增益
- 不再承担“救火主线”的职责

## 4. 我们的数据/清洗/特征工程接下来怎么做

### 4.1 清洗原则

下一条主线的数据治理必须更严格，但不能丧失语义。

允许加强的方向：

- 空白与控制字符统一
- 异常符号与噪声字符治理
- 断裂、缺字、标记符号规范化
- prompt / prefix 输入规范化
- 明确坏 token 抑制策略

不允许加强成“过清洗”的方向：

- 把阿卡德语形式信息洗平
- 把人名、计量、公式化结构做不可逆弱化
- 为了表面整洁牺牲语义和文本结构

标准应是：

- 规范化形式
- 不扁平化内容

### 4.2 GC 扩容的定位

当前 GC / short-aligned 资产仍是有效的。

但它接下来不应被当成：

- 当前小主干继续提分的主要借口

而应被当成：

- 更强 ByT5 主干的训练补充资产

### 4.3 official-compatible local eval 固定

从这版开始，本地主评价口径固定为：

- corpus BLEU
- corpus chrF++
- geometric mean

并明确写成：

- `official_formula_local`

不再让这些口径混乱地和：

- `parent reconstructed`
- 各类局部 surgery 指标
- 非官方 bridge 的代理统计

混成一个主序。

## 5. 具体执行规划

### P1：公开模型对照补全

目标：

- 完成同一公开模型的 decode ablation

产物：

- `reports/.../summary_compare_beam4_lp07.json`
- `reports/.../summary_default_beam8_rep11.json`
- 对照结论文档

### P2：冻结旧 winner 微调链

目标：

- 停止 `A20 / chooser / term patch / replay rescue` 同质继续

产物：

- 新纪律确认文档
- 旧线归档说明

### P3：新建强单模主线

目标：

- 立一个“更大 ByT5 主干”主线

最低要求：

- 不再以 `byt5-small + LoRA(q,v,r8)` 为唯一主干假设
- 首先追求单模质量和输出健康

### P4：第二阶段再接 rerank / MBR

进入条件：

- 强单模已站住
- 本地 official-compatible eval 明显转正
- 输出健康不依赖 fallback/chooser 修补

未达条件前：

- 不重开当前 formulation 的 `MBR`

## 6. 最短路径结论

当前最快、最正确的路径不是：

- 继续修当前 winner
- 继续加 chooser
- 继续在小主干上做 repair

当前最快、最正确的路径是：

1. 看透公开高分模型的主干与 decode 贡献
2. 固定 official-compatible local eval
3. 冻结旧的小主干救火链
4. 新建更大 ByT5 强单模主线
5. 等强单模成立后，再把 retrieval / rerank / MBR 接回去

一句话总结：

- 我们现在不是“缺更多小修补”
- 我们是“该把主干升级，并把修补链降级”
