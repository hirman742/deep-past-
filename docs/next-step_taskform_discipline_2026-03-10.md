# Cloud Stage2 下一步纪律（Taskform，2026-03-10）

## 0. 结论先行

从 `2026-03-10` 起，`taskform` 的主线不再是 `dan-1 / dan-2` 式 parent-level fusion。

原因已经足够明确：

- 当前 `dan-1` / `dan-2` 没有给出可推广的正信号
- 当前分桶 routing 实现使用了 target-side 特征，不能当作可部署策略
- 当前正式 winner family 仍只有 `fold0`，多折 OOF / ensemble 这条高价值主线并未真正跑完

二期目标不变：

- 不是只做流程补全
- 不是只追求稳一点
- 而是要**显著拉升 `geom`**

因此所有新线都按“是否可能带来结构性 `geom` 提升”排序，而不是按工程完整度排序。

因此从 `2026-03-10` 晚间起，主线再次转向。

新的优先级不再是 `L2/L3` 后处理，而是研究依据更强、且更有希望带来结构性 `geom` 提升的**非 LLM** 主线：

1. competition-only `denoising pretrain + iterative back-translation + multilingual/transfer fine-tune + model averaging`
2. 并行数据噪声过滤 / curriculum / alignment-quality gate
3. efficient `MBR` decoding / candidate reranking
4. `kNN-MT / dynamic retrieval`（仅基于官方并行数据建 datastore）
5. `L1-lite` OOF / ensemble、`L5` constrained decoding、`L4` ORACC mix 作为次主线

当前冻结正式主线：

- model family:
  - `ByT5-small chunk`
- checkpoint:
  - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- decode:
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`

当前正式分数：

- full-val reconstructed:
  - `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`
- hard routed full:
  - `geom / bleu / chrfpp = 13.7161 / 7.1669 / 26.2499`

## 1. 共性纪律

### 1.1 路由纪律

从这一版开始，任何可部署 routing / bucket / policy 都必须满足：

- 只能使用推理时可见特征
- 禁止使用任何 target/reference 派生特征
- 禁止使用当前验证集结果反向定义 bucket

明确禁止：

- `parent_ref_tok`
- `reference length`
- `gold bucket`
- 任何由 `translation / target / reference` 直接或间接导出的 routing 特征

允许使用的特征：

- `chunk_total`
- source 侧长度：
  - `source_char_len`
  - `source_tok_len`
- 结构标记：
  - `has_gap`
  - `has_bracket`
  - `has_subscript`
  - `has_x`
  - `marker_count`
- `Pass-A` 预测可观测统计：
  - draft 长度
  - 重复率
  - `gap>` 异常率
  - 内部 trigram repeat

### 1.2 评估纪律

所有新策略统一采用三段式：

- `tune`
- `holdout`
- `report`

解释：

- `tune` 只用于选阈值、选桶、选规则
- `holdout` 只用于决定是否放行
- `report` 只用于最后汇报，不再调参

额外强制要求：

- 对 routing / rerank / cleanup 这类策略，必须补随机化对照
- 固定覆盖率后，至少做 `200` 次随机 routing / random action baseline
- 若真实策略没有显著高于随机 `95%` 分位：
  - 不得宣称有效

### 1.3 评分纪律

从这一版开始，不允许只报单一口径。

每条线统一输出 3 套分数：

- `local`
  - 当前本地主口径：
    - `BLEU / chrF++ / geom`
    - 以 full-val reconstructed 为主
- `official-like`
  - 在本地可复现条件下尽量贴近官方 metric
  - 本轮先不抢在最前面接入
  - 但在 `L2/L3` 首轮稳定后必须评估是否接入
- `hard`
  - 困难句子/困难 parent 口径
  - 至少包括：
    - routed hard full
    - 长 parent / 多 chunk / 结构标记重样本

统一要求：

- 不允许只报其中一套
- promote / reject 不能只靠单一分数决定
- 任何 summary / gate report 都必须同时写出这三套结果

当前自动化约束：

- `L2` 与 `L3` 由统一后台链路执行
- `L2` 先产出 `term_lexicon.tsv / glossary_min.csv`
- `L3` 默认复用 `L2` 词表后再做 source-only policy
- 首轮执行脚本固定为：
  - `/workspace/deep-past-/tmp/run_taskform_l2_l3_20260310.sh`
  - `/workspace/deep-past-/tmp/launch_taskform_l2_l3_20260310.sh`

### 1.4 官方 metric 接入纪律

官方 metric 相关接入不放在第一步，但也不能无限期搁置。

执行规则：

- 先把 `L2-P` 与 `L3-P` 做出第一轮稳定产物
- 之后立即评估是否接入官方 metric bridge
- 若接入成本低且规则清晰：
  - 纳入后续所有 gate
- 若接入成本高：
  - 先保留 `official-like` 适配层
  - 但必须保留后续切换入口

当前结论：

- 先不让官方 metric 接入阻塞 `L2/L3`
- 但在二者首轮结果出来后，官方 metric 接入必须进入待办前列

### 1.5 并行纪律

统一仍按：

- `P -> W -> F`

并行规则：

- 同条线内不同段不抢跑
- GPU 重训练任务串行
- CPU 分析、词表、audit、report 可以并行
- 轻量 decode 和后处理若不抢卡，可后台并行

### 1.6 暂停项

以下方向先归档，不再作为主线：

- `dan-1 / dan-2` full rewrite fusion
- target-aware routing
- 新的 `flan-t5-small` parent fusion 训练
- 把 `no_repeat_ngram_size` 当作默认修复
- 再起新的 `ByT5-base / mT5-base` 架构探测

### 1.7 主线转向（Research-Backed, Non-LLM）

从这一版开始，以下结论视为正式纪律：

- `L2` 术语小词表线归档：
  - 当前已证明可止损，但没有形成可放行增益
- `L3` bucket / row-gate 后处理线归档：
  - 当前 corrected-random 下仍未显著高于随机对照
- 后处理线保留结果，不再继续消耗主线时间
- 接下来只优先投入“可能显著拉升 `geom`”的模型侧与 decoding 侧主线

本次转向的研究依据：

- `MBR` 已有 `2024` 年高效近似实现，说明它不再只是理论上有效、工程上太慢的方向
- `2024` / `2025` 低资源 MT shared-task 系统反复证明：
  - `denoising pretrain`
  - `back-translation`
  - `multilingual/transfer`
  - `model averaging`
  - `kNN-MT`
  是现实有效的组合
- `2024` 文档级 MT 研究也提示：
  - 在低资源场景中，强 sentence-level 模型可能优于 context-aware DocNMT
  - 因此不应把大量时间再次投到 document-level 架构改造

## 1.8 新的主线执行顺序（非 LLM）

### A1. competition-only 主模型增强

目标：

- 走最可能产生结构性增益的主线
- 仍基于当前正式 family，而不是另起不受控架构
- 与仓库现实保持一致：
  - 在 reverse system 尚未存在前，先执行可落地的 `competition-only source-mono synthetic-data smoke`
  - 不把“必须先有 reverse BT”当成启动主线的阻塞条件

执行顺序：

- `A1_P1`: competition-only monolingual 资产盘点
- `A1_P2`: `denoising pretrain` smoke
- `A1_P3`: synthetic-data smoke
  - 若 reverse model 已就绪：
    - 跑标准 `back-translation`
  - 若 reverse model 未就绪：
    - 先跑 `source-mono -> current winner pseudo-target -> filtered continue` 的 competition-only fallback
- `A1_P4`: multilingual / transfer continue
- `A1_P5`: pair-specific fine-tune + `model averaging`

当前夜间执行口径：

- `A1_P1` 与 `A2_P1/P2` 可并行准备
- `A1_P2` 与 `A2` probe 串行占用单卡
- `A1_P3` 只在以下条件之一满足时自动接续：
  - `A2` best smoke 非负或仅微负（未出现明显坏信号）
  - `A1_P2` 自身训练与长度健康正常
- 以上安排是为了最大化算力利用，而不是放松 gate

当前自动化入口：

- 主链 orchestration：
  - `/workspace/deep-past-/scripts/taskform_a2_a1_flow.py`
- synthetic target 生成：
  - `/workspace/deep-past-/scripts/taskform_generate_pseudo_targets.py`
- 后台运行：
  - `/workspace/deep-past-/tmp/run_taskform_a2_a1_20260310.sh`
  - `/workspace/deep-past-/tmp/launch_taskform_a2_a1_20260310.sh`

gate：

- smoke 若不能在 `anchor64` 上给出正信号：
  - `reject_stop`
- 若 full-val local / hard 都无改善：
  - `reject_stop`

### A2. 数据质量与 curriculum

目标：

- 在任何重训练前先减少“坏平行句 + 错配 + 近 raw 噪声”带来的训练污染

执行顺序：

- `A2_P1`: 对当前 processed train 做 alignment/noise proxy 打分
- `A2_P2`: 做多阈值过滤对照
- `A2_P3`: 只放行最优阈值到主训练链

当前实现要求：

- `A2` 只用训练时可见字段做 proxy：
  - `chunk_mode`
  - `short_align_mode`
  - `align_type`
  - `align_cost`
  - source/target 长度比
  - source/target 自身重复特征
- 不允许用任何 reference-side 评估结果回写 filter
- `A2` 首轮至少保留一个“轻过滤”候选，避免一上来把有用增广样本全砍掉

gate：

- 若过滤后只降数据量、不提分：
  - `reject_stop`
- 若轻过滤即可改善：
  - 进入 `A1`

### A3. Efficient MBR

目标：

- 不换模型，直接在 decode 端争取真实收益
- 只使用非 LLM utility：
  - `BLEU`
  - `chrF++`
  - 它们的组合

执行顺序：

- `A3_P1`: candidate diversity audit
- `A3_P2`: current winner family 上做 sampled / diverse candidate set
- `A3_P3`: `MBR` 近似评估
- `A3_P4`: 对比 beam winner / looptrim / row-gate / MBR

gate：

- 若 MBR 不能超过当前正式 decode：
  - `reject_stop`
- 若 only local 提升、hard 不提升：
  - `review_stop`

### A4. kNN-MT / dynamic retrieval

目标：

- 用官方并行数据自身构建非参数记忆
- 在不依赖 LLM 的情况下争取 domain recall 和术语复现

执行顺序：

- `A4_P1`: official parallel datastore smoke
- `A4_P2`: static kNN-MT
- `A4_P3`: dynamic retrieval / skip retrieval
- `A4_P4`: decode latency vs gain audit

gate：

- 若 gain 太小或延迟太高：
  - `reject_stop`
- 若 hard subset 提升明显：
  - 进入 wider evaluation

### A5. 明确降级项

以下方向从主线降级：

- document-level / parent-level 新架构探测
- `L2` 词表 patch 主线化
- `L3` post-processing 主线化

原因：

- 现有证据更像“小修补”而不是“大幅提分”
- 当前阶段不符合二期目标

## 2. 主线一-lite：当前正式 winner family 的多样性探针与条件 OOF / ensemble

### 2.1 目标

验证当前正式 winner family 是否存在足够的折间 / seed 间互补性。

这条线降级为 `lite`，原因很现实：

- 当前正式主线只有 `fold0`
- 但当前误差更像系统性偏差，不像纯方差问题
- 因此它更像“上限探针”和“提交基础设施线”
- 不应先于更对症的 `L2 / L3`

本线的目标不是默认追求 `geom` 大涨，而是回答：

- 当前正式 family 值不值得继续投到 `3-fold / 5-fold`
- 以及 ensemble 的现实上限大约在哪一档

### 2.2 固定 recipe

固定不改：

- family:
  - `STEER_S4_CONTINUE_BS24_LEN640_SEG5`
- decode:
  - `beam=4 / lp=0.7 / max_new_tokens=384`

允许微调的仅限：

- 每折 best checkpoint 选择
- 每折 anchor decode 校验
- ensemble 权重搜索

### 2.3 P 纪律

- `P1_1`: 复核 `fold0` 正式 winner 产物
- `P1_2`: 只补一个 diversity probe：
  - `fold1`
  - 或同 recipe 新 seed
- `P1_3`: 跑 anchor32 / anchor64 decode
- `P1_4`: 计算 error overlap / 互补性
- `P1_5`: 做 pairwise ensemble 上限测试

P gate：

- 任一 probe 若相对 `fold0` anchor64 `geom` 下降超过 `1.0`
  - `review_stop`
- 若 pairwise ensemble 相对 best single model 提升小于 `+0.15`
  - `review_stop`
- 若 error overlap 过高、互补性太弱：
  - `review_stop`
- 只有当 pairwise probe 明确显示互补性时：
  - 才进入 `W`

### 2.4 W 纪律

- `W1_1`: 补到 `3-fold OOF`
- `W1_2`: 重新做 ensemble 权重搜索
- `W1_3`: 再决定是否扩到 `5-fold`
- `W1_4`: 产出 `OOF bridge report`

W gate：

- `3-fold OOF ensemble` 必须仍有正收益
- 若扩到 `5-fold` 不增反降或显著更不稳定：
  - 回退到 `3-fold`

### 2.5 F 纪律

- `F1_1`: 用固定每折 winner 对 test 推理
- `F1_2`: 导出 `submission.csv`
- `F1_3`: 记录 `submission_log.md`
- `F1_4`: 固化 `kaggle_infer.ipynb`

统一产物：

- `oof_predictions_folds_*.csv`
- `ensemble_search.json`
- `ensemble_search.md`
- `submission_log.md`
- `kaggle_infer.ipynb`

## 3. 主线二：术语词表与一致性后处理

### 3.1 目标

做最小可审计词表，并把当前已知现状直接写入术语线。

这条线要明确面对的不是抽象“术语问题”，而是当前 winner 的具体现实：

- 长文本和多 chunk 有系统性重复
- 高频公式化片段反复出现
- 名称、地点、度量单位和公式术语的一致性没有被单独治理

因此本线优先只修：

- 人名
- 地名
- 度量单位
- 常见公式化术语

当前优先 term buckets：

- `Seal of X`
- `mina / shekel / talent / textile`
- 高频发信人 / 收信人名
- 高频地名与商贸地点

当前已知基线现状也直接写进术语线：

- 正式主线 full-val reconstructed：
  - `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`
- 困难 routed hard full：
  - `geom / bleu / chrfpp = 13.7161 / 7.1669 / 26.2499`
- 二期目标不是“修一点一致性”而已：
  - 而是要求术语线先拿到稳定正信号，再考虑反哺训练去推高 `geom`

这条线优先级高，因为：

- 仓库已有错误桶和清洗规范
- 但没有真正落地的 `term_lexicon.tsv / glossary_min.csv`
- 这是低风险、可部署、可解释的 test-time 增益方向
- 若给出正信号，还可以**反哺训练**

### 3.2 纪律

只允许高置信规则：

- exact match
- 明确变体归并
- 明确“不译 / 保持一致”规则

禁止：

- 大范围自由替换
- 依赖 gold target 构词
- 模糊匹配到低置信词
- 无条件 whitespace/layout 正规化

### 3.3 P 纪律

- `P2_1`: 从正式 winner 错误样本抽取 top term buckets
- `P2_2`: 生成人工可审计词表：
  - `term_lexicon.tsv`
  - `glossary_min.csv`
- `P2_3`: 实现最小一致性 patcher
- `P2_4`: 跑 tune split targeted eval
- `P2_5`: 跑 holdout split targeted eval

P gate：

- 词表必须完全可审计
- 只允许高置信触发
- holdout targeted bucket 上必须有稳定正信号
- 若全局 mixed 下降：
  - `reject_stop`

### 3.4 W / F 纪律

- `W2_1`: 按 term type 分开扩词：
  - name
  - place
  - measure
  - formula
- `W2_2`: 对正式 full-val mixed 评估
- `W2_3`: 若 inference-only 正信号成立，再设计最小 training feedback
- `F2_1`: 只对高置信命中样本启用

training feedback 只允许走这三条：

- source 侧变体规范化
- 极小范围 target 一致性增强
- 供 `L5` constrained decoding 复用

明确禁止：

- 用 val / holdout target 反向造词表
- 大范围重写训练 target
- 把词表线直接升级成 semantic rewrite

放行标准：

- full-val mixed `geom >= +0.10`
- 且错误替换率可控
- 且未命中样本完全不改

本线首轮自动化产物固定为：

- `summary.json`
- `gate_report.md`
- `term_lexicon.tsv`
- `glossary_min.csv`
- `local_eval.json`
- `official_like_eval.json`
- `hard_eval.json`

当前收紧版默认值：

- 只保留高精度小词表
- 默认只开 `measure`
- `min_ref_count >= 5`
- `min_pred_hit_count >= 2`
- `max_terms <= 12`
- 默认关闭 builtin term normalizer
- 默认禁止输出 whitespace 折叠

## 4. 主线三：source-only routing + 非生成式 cleanup / selective patch

### 4.1 目标

把当前 exploratory 的 concat / rerank 线改造成可部署版本，并把它从“可疑提分术”改成“可信方法学”。

这条线不是继续做 fusion 训练，而是：

- source-only routing
- 非生成式 cleanup
- selective patch

### 4.2 当前问题

当前 routing 危险点已经确认：

- 使用了 target-side `parent_ref_tok`
- `route_rank` 也依赖该特征
- 当前结果只能算 exploratory，不可直接 promote

因此这一条线的第一任务不是提分，而是先消除泄漏。

只有方法学先站稳，后面的 `geom` 才有意义。

### 4.3 动作空间

只允许在以下候选动作中选：

- `pass_a_prediction`
- `raw_concat`
- `looptrim_concat`
- `selective_patch_light`

说明：

- `selective_patch_light` 只修坏 span：
  - `gap>`
  - 明显重复串
  - 术语不一致
- 不允许整段重写

### 4.4 P 纪律

- `P3_1`: 重建 source-only metadata
- `P3_2`: 定义 frozen buckets
- `P3_3`: 建 `tune / holdout / report` 三段 split
- `P3_4`: 在 `tune` 上选 bucket action policy
- `P3_5`: 做随机 routing / random action 对照
- `P3_6`: 在 `holdout` 上评估 mixed output

P gate：

- 任何策略若仍使用 target-side 特征：
  - `hard_stop`
- 若真实 policy 不高于随机 `95%` 分位：
  - `reject_stop`
- 若 holdout mixed 不高于正式 `Pass-A`：
  - `review_stop`

### 4.5 W / F 纪律

- `W3_1`: 扩到 full-val report split
- `W3_2`: 按桶输出 bridge report
- `F3_1`: 只在通过的桶中启用

额外硬纪律：

- `7+ chunk` 若没有独立正信号：
  - 默认继续走 `Pass-A`
- 不允许因为某一小桶正收益就放大全路由

当前收紧版默认值：

- 默认只允许 `chunk4_6` 进入候选切换
- `chunk7plus` 回退 `Pass-A`
- `2-3 chunk` 默认回退 `Pass-A`
- 允许切换前必须再过行级安全门：
  - 候选长度不得明显短于 `Pass-A`
  - 且内部重复或公式循环必须确实下降

当前进一步收束：

- `bucket policy` 只保留作对照
- 当前更优的探索方向是 `row-level safety gate`
- 不再先按桶决定动作
- 而是逐条样本只在候选满足以下条件时切换：
  - `looptrim_concat` 或其他候选能降低内部重复 / 公式循环
  - 且长度保持安全

当前 row-gate 最佳探索配置：

- `looptrim_only`
- `min_length_ratio = 0.8`
- `min_repeat_gain = 1`

当前 row-gate 结果（`2026-03-10`）：

- 第一版 row-gate 因随机池实现错误，不再作为主结论
- corrected-random v2 才是当前有效口径：
  - holdout local：
    - `13.3577 -> 13.4387`
  - routed hard full：
    - `13.7161 -> 13.7373`
  - official mixed：
    - `14.3323 -> 14.4400`

但当前仍未 promote，原因是：

- corrected `random p95 = 13.4401`
- 当前 holdout `13.4387` 仍略低于它
- 说明这版 selector 虽然接近成立，但还没有证明自己显著优于“从 eligible 池里分层随机抽相同数量样本”

本线首轮自动化产物固定为：

- `summary.json`
- `gate_report.md`
- `bucket_tune_choices.csv`
- `holdout_random_policies.csv`
- `routed_policy_predictions.csv`
- `local_eval.json`
- `official_like_eval.json`
- `hard_eval.json`

## 5. 主线四：ORACC mix 在当前 winner family 上重开

### 5.1 目标

把已有 ORACC 混训基础设施真正对接到当前正式 winner family，而不是停留在旧计划层。

### 5.2 前置条件

只有同时满足以下条件才可启动：

- 官方规则已再次核实
- `external_data_manifest.md` 已生成
- `rules_checklist.md` 已生成
- dedupe / similarity filter audit 完整

### 5.3 P 纪律

- `P4_1`: 刷新 ORACC manifest 与规则清单
- `P4_2`: 用现有 dedupe + similarity filter 重做审计
- `P4_3`: 在当前 winner family 上跑 `10% / 20% / 30%` fold0 smoke
- `P4_4`: anchor32 / anchor64 比较
- `P4_5`: full-val 单点 only if smoke positive

P gate：

- 若规则口径不清晰：
  - `hard_stop`
- 若 dedupe / similarity filter 不充分：
  - `hard_stop`
- 若 smoke 相对 competition-only baseline 无正信号：
  - `reject_stop`

### 5.4 W / F 纪律

- `W4_1`: 只放行最优 ratio
- `W4_2`: 先补 `2-fold`
- `W4_3`: 若稳定，再进 `3-fold / 5-fold`
- `F4_1`: 只作为 ensemble family 候选，不直接替代主线

放行标准：

- smoke `geom >= +0.20`
- 或长度健康显著改善且分数不降
- 若公榜风险高：
  - 仅作为备选 family

## 6. 主线五：真正的 constrained decoding PoC

### 6.1 目标

实现真正的最小正向约束 PoC，而不是仅靠当前的 `bad_words_ids` 负约束。

当前仓库已有：

- `bad_words_ids`

当前仓库没有：

- `force_words_ids`
- trie / prefix 约束
- 高置信术语约束解码

### 6.2 约束范围

只允许在极小范围试：

- 高频人名
- 高频地名
- 高频度量单位
- 高频且形式稳定的公式术语

只对高置信触发样本启用。

### 6.3 P 纪律

- `P5_1`: 复用主线二的 `term_lexicon.tsv`
- `P5_2`: 实现最小 constrained decoding PoC
- `P5_3`: 只在命中样本子集上做 A/B
- `P5_4`: 对 mixed output 做无害性检查

P gate：

- decode 成功率必须接近 `100%`
- 未命中样本必须完全不受影响
- 若只在 targeted subset 好、mixed 全局变差：
  - `review_stop`

### 6.4 W / F 纪律

- `W5_1`: 扩词表到 `<= 20` 个高置信 term
- `W5_2`: 扩到 `<= 50`，但仍保持高置信
- `F5_1`: 只作为 test-time optional layer

放行标准：

- mixed full-val `geom >= +0.05`
- 且约束命中样本不出现 decode 崩坏

## 7. 新的统一优先级

### 7.1 立刻执行

- 主线二 `term lexicon`
- 主线三 `source-only routing audit`
- 主线一-lite `diversity probe`

解释：

- 主线二和主线三更对症当前误差形态
- 主线一-lite 保留，但只做上限探针，不抢第一优先级
- 二期目标是显著抬升 `geom`，因此优先做更可能带来结构性提升的线

### 7.2 条件执行

- 主线五 `constrained decoding`
- 主线四 `ORACC mix`

解释：

- 主线五依赖主线二先产出词表
- 主线四依赖规则确认与 manifest

## 8. 建议执行顺序

第一阶段：

- `L2-P`
- `L3-P`
- `L1-lite-P`

并行方式：

- `L2 / L3` 先行
- `L1-lite` 只占低优先级 GPU 窗口
- 若 GPU 预算紧：
  - 先不扩 `L1`

第二阶段：

- 若 `L2` 给正信号：
  - 接 `L2-W`
- 若 `L3` 完成 source-only holdout：
  - 再决定是否放行 mixed policy
- 若 `L1-lite` 明确显示互补性：
  - 再决定是否扩到 `L1-W`

第三阶段：

- `L5` 只在词表稳定后启动
- `L4` 只在规则与 manifest 完成后启动

## 9. 当前要补的缺口文件

这轮要求补齐以下产物：

- `external_data_manifest.md`
- `rules_checklist.md`
- `term_lexicon.tsv`
- `glossary_min.csv`
- `submission_log.md`
- `kaggle_infer.ipynb`
- `official_metric_report.json`
- `hard_bucket_report.json`

## 10. 一句话版本

冻结当前 winner 当正式主线，停止 target-aware fusion 扩线，先把：

- 术语一致性
- source-only routing
- `L1-lite` 多样性探针

这三条更对症、更可能拉升 `geom` 的线做完，再决定要不要扩到完整 OOF / ensemble、ORACC 和 constrained decoding。
