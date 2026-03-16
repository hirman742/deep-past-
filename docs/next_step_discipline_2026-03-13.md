# Cloud Stage2 下一步纪律
## Winner 四期停训大整改版，2026-03-13

## 0. 结论先行

2026-03-12 这轮结论已经足够明确：

- 当前官方公榜提交版本 `Version 6` 的公开分数只有 `7.6`
- 这不是本地 `21.x fullval geom` 的轻微偏差
- 这是“主评估目标错位”的系统性失配

同时，公开模型页已经给出足够强的外部反证：

- `byt5 akkadian mbr`：best public score = `35.3`
- `byt5 akkadian mbr v2`：best public score = `35.4`
- `akkadian-byt5-small-translator`：best public score = `28.9`
- `Gemma 3 / gemma-3-4b-it`：best public score = `32.6`
- `flan-t5 / base`：best public score = `28.4`
- `byt5 adafactor comp dataset train`：best public score = `29.1`

因此，从 2026-03-13 起，项目正式进入：

- Winner 四期停训大整改

这期的首要目标不再是：

- 继续推高本地 `313-row fullval reconstructed geom`
- 继续在 `A18 / A20 / chooser / term patch` 上做小修补
- 继续把 `official-like` 当成主排序依据

这期的首要目标改为：

- 重新对齐官方公榜目标
- 下载公开模型与公开方案
- 做可复现实物审计
- 重新决定主线

## 1. 当前仓库现状

### 1.1 这是一个“紧急恢复态”仓库，不是完整训练工作树

截至 2026-03-13 当前工作树，仓库更接近：

- 代码、配置、纪律文档、评估脚本、阶段性 `reports/` 基本仍在
- 许多关键训练实物没有直接展开在当前工作树
- 多数重要 winner 资产被打包进 `release/*.tgz` 与 `deep_past_rescue_20260306_191331.tgz`

当前可直接看到的仓库骨架仍然完整：

- `docs/` 文档仍较全
- `scripts/` 训练、诊断、评估、taskform/winner 流程脚本仍在
- `configs/` 的 mT5 / ByT5 / cloud / chunk / TAPT 配置仍在
- `reports/` 中保留了大量 2026-03-10 至 2026-03-11 的总结与证据
- 原始比赛数据与 `data/external/oracc_parallel.csv` 仍在

但当前工作树下的 `runs/` 很不完整，只剩很少量早期或轻量痕迹，例如：

- `runs/E2_MT5_LEN_fold0/metrics.json`
- `runs/E5_BYT5_fold0/metrics.json`
- `runs/FEAS_MT5_ZERO_TRUNC_fold0/metrics.json`
- `runs/OVERFIT512_MT5_fold0/metrics.json`

这说明：

- 现在仓库里“结论和报表”多于“可直接续训的展开态实物”
- 不能把当前工作树误当成完整可继续训练的原始环境

### 1.2 关键 winner 实物还在，但主要在归档包里

`docs/archive_manifest_2026-03-11.md` 已经明确说明，这是一次分层归档：

- `release/deep_past_winner_core_20260311.tgz`
- `release/deep_past_research_tail_20260311.tgz`
- `release/deep_past_legacy_taskform_history_20260311.tgz`

其中最关键的是 `winner_core`。归档清单表明它内部仍包含：

- `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/`
- `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/`
- `best_model/adapter_model.safetensors`
- `checkpoint-250` / `checkpoint-400`
- `trainer_state.json`
- `run_summary.json`
- 大量 `diagnostics/` 文件

也就是说：

- 关键 winner / incumbent / retrieval W-lite 的 LoRA 与诊断资产并没有彻底丢失
- 但它们当前没有解包回工作树
- 所以文档中大量 `runs/...` 路径在当前仓库状态下是“引用存在，实物未展开”

### 1.3 当前仓库最实际的问题不是“完全没证据”，而是“证据与实物分离”

现在仓库中最完整保留下来的是：

- 文档化结论
- JSON/CSV 级别的总结
- 流程脚本
- 归档包

相对缺失或受影响的是：

- 直接可运行的完整 `runs/` 目录
- 训练中间态目录
- 一部分可以原地续训、原地复算的工作树实物

典型例子：

- 归档恢复顺序文档要求先看 `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/run_summary.json`
- 但这个路径在当前工作树中是缺失的
- 该文件实际上存在于 `release/deep_past_winner_core_20260311.tgz` 内部

因此当前仓库的真实状态应表述为：

- 不是“核心 winner 资产已经消失”
- 而是“核心 winner 资产被封存在 release/rescue 包中，当前工作树未展开”

### 1.4 评估桥接层确实缺失

已有本地 probe 明确写出：

- `status = missing_bridge`
- `candidates found = 0`
- `recommendation = no official metric bridge files found; keep official-like layer and add bridge later`

这说明当前仓库至少有一个关键结构性缺口：

- `official/public leaderboard` 与本地评估之间缺乏正式 bridge

因此近期很多判断虽然有完整本地分析链条，但仍然只能停留在：

- `official-like`
- `reconstructed geom`
- 局部 fullval / hard 口径

这正是这轮路线失配的核心背景之一。

### 1.5 当前仓库还能支撑什么，不能支撑什么

当前仓库仍然足以支撑：

- 回读最近一轮 winner 线的决策史
- 定位 A1 / A2 / A3 / replay / TAPT / retrieval / term patch 的结论
- 利用已有脚本与报告重新做审计框架
- 从 release 包中恢复关键实物
- 转向公开模型下载与 recipe 逆向

当前仓库不适合直接做的事情是：

- 在未解包关键 runs 资产前直接恢复原链路长训
- 在 `missing_bridge` 未解决前继续把本地 geom 当主指标滚动优化
- 在恢复态工作树上假设“所有训练记录都还原无损”

所以，从仓库资产治理角度，当前项目已进入：

- `documentation-rich but unpacked-asset recovery state`

换成中文就是：

- 结论很多
- 过程证据不少
- 关键实物并未完全丢失
- 但训练工作树不是展开态
- 桥接评估层缺失
- 直接续原主线风险很高

## 2. 这次为什么会走错

这次不是“模型 family 选错”，而是“评价与决策框架选错”。

最核心的错误只有四条：

1. 把代理指标当成了主目标
   - 最近主线一直围绕：
   - `313-row fullval reconstructed geom`
   - 这套口径可以用于本地分析
   - 不能再单独驱动主序决策
2. 在官方 bridge 缺失的前提下继续滚动优化
   - 仓库现有 probe 已经明确写出：
   - `status = missing_bridge`
   - 但后续仍继续做 `geom-first winner` 微调
3. 把“当前仓库 formulation 失败”误读成了“大类路线失败”
   - 当前失败的是：
   - 我们仓库这一版 `MBR / chooser / repair / external continue` 配方
   - 不是：
   - `ByT5`
   - `LoRA`
   - `MBR`
   - 公开 `ByT5 recipe`
4. 外部 sanity check 介入过晚
   - 公开模型公榜分数远高于当前官方 `7.6`
   - 这本应更早触发路线重置

## 3. 从这版开始的硬纪律

### 3.1 停训

从这版开始，先停：

- 新的 retrieval 长训
- 新的 DAPT / TAPT-lite
- 新的 `A18 / A20` 同质 repair sweep
- 新的 deploy subtree chooser 微调
- 新的 term patch sweep

只有当“公开模型复现审计”完成后，才允许恢复训练。

### 3.2 指标主次重排

从这版开始，所有候选按以下顺序排序：

1. official/public leaderboard score
2. official-like BLEU
3. official-like chrF++
4. 本地 geom

其中：

- `geom` 降级为诊断指标
- 不能再单独作为 promote 依据

### 3.3 撤销错误冻结

此前文档里对以下大类的冻结表述需要纠正：

- 不能再写成：
  - `MBR` 大类冻结
- 应改成：
  - 当前仓库已证伪的 `MBR formulation` 冻结
  - 公开强基线中的 `MBR` 为优先审计对象

同理：

- 不能把 `ByT5 + LoRA` 当成负线
- 当前负线只是：
  - 我们最近这条 `geom-first` 本地 winner 工作流

## 4. 当前资产的新定位

### 4.1 保留但降级

- `A20`
  - 保留为：
  - 本地 `313-row` 诊断参考
  - 不再视作当前 promote 候选
- `A18`
  - 保留为：
  - exploratory best local geom
  - 不再视作部署方向
- `A3C replay15`
  - 保留为：
  - 候选池互补性证据
  - 不再单独推动主序

### 4.2 直接归档

- 当前 formulation 的 `A1X_P1 DAPT/TAPT-lite`
- 当前 formulation 的 term-aware patch
- 当前 formulation 的 `A20 deploy subtree chooser probe`

归档含义：

- 结果可保留
- 不再继续同质扩臂

## 5. 四期唯一主任务：公开模型下载与评定

2026-03-13 这一天的任务，不是训练，而是：

- 下载公开模型
- 识别公开 recipe
- 做可复现性评定

### 5.1 要审计的公开对象

第一批优先对象：

1. `byt5 akkadian mbr`
2. `byt5 akkadian mbr v2`
3. `akkadian-byt5-small-translator`

第二批补充对象：

1. `byt5 adafactor comp dataset train`
2. `flan-t5 base`
3. `Gemma 3 / gemma-3-4b-it`

### 5.2 审计问题

每个公开对象必须回答：

1. 能否下载到：
   - 权重
   - adapter
   - tokenizer
   - inference notebook / script
   - postprocess / submission glue
2. 它属于哪一类：
   - full reproducible
   - partial reproducible
   - weights-only
   - notebook-only
   - listing-only
3. 它的 recipe 核心是什么：
   - base model
   - LoRA 还是 full finetune
   - chunking / prompt 格式
   - decode 设置
   - 是否使用 MBR
   - 是否有 rerank / candidate pool
   - submission 后处理
4. 它和当前仓库主线差异在哪：
   - 数据
   - 模型
   - decode
   - MBR
   - rerank
   - 评估目标

### 5.3 评定输出

明天结束前至少要产出：

- 能否复现
- 值得直接移植的部分
- 不值得跟的部分
- 下一条主线建议

## 6. 明天的一天计划

### 6.1 P4_1：公开模型 inventory

产出：

- 所有目标公开模型的 listing 表
- 公榜分数对照表
- 可下载资产清单

### 6.2 P4_2：公开资产下载

目标：

- 能下多少下多少
- 优先拿：
  - 推理代码
  - notebook
  - model card
  - config
  - tokenizer
  - LoRA / checkpoint

### 6.3 P4_3：recipe 逆向

重点不是“偷参数”，而是：

- 识别公开 submission path
- 识别 `MBR / rerank / postprocess` 结构
- 识别当前仓库缺失的关键环节

### 6.4 P4_4：可复现实验判级

对每个公开对象分级：

- A级：可直接本地复现或近复现
- B级：只能部分复现，但关键 recipe 清晰
- C级：只能看分数和简介，无法实用复现

### 6.5 P4_5：新主线决策

明天结束时只允许选以下三类之一进入后续：

1. 公开 `ByT5 / MBR recipe` 复现主线
2. 公开 baseline + 我们检索资产的混合主线
3. 公开方案不可复现，则转 paper / baseline 复现主线

不允许重新回到：

- `A20` 继续扫阈值
- `A1X` 继续训一版看看
- chooser 再加一层规则

## 7. Gate

四期第一天的 gate 不是“拿更高本地分”，而是：

1. 至少定位并分类完首批公开模型
2. 至少下载到一条高分公开路线的实物资产或代码资产
3. 至少明确一条：
   - 可以本地复现
   - 或可以部分复现并足以移植 recipe

如果三条都做不到：

- 必须写清楚 blocker
- 再决定是否转 paper 复现

## 8. 产物要求

建议新目录：

- `reports/taskform_winner_phase4_public_model_audit_20260313/`

至少产出：

- `model_inventory.md`
- `model_download_manifest.json`
- `reproducibility_matrix.csv`
- `recipe_diff.md`
- `official_vs_local_gap_report.md`
- `next_route_decision.md`
- `status.json`

## 9. 最短结论

从 2026-03-13 起：

- 停训
- 停本地 `geom-first` 微调
- 先下载公开模型
- 先审公开 recipe
- 先把官方目标重新对齐

这期不是再“修当前 winner”。

这期是：

- 推翻错误评估主序
- 重建 winner 主线

## 附录 A：2026-03-10 至 2026-03-12 近期训练与实验回填

### A.1 A2R hybrid 本地 winner 线

这段时间主训练内容，主体不是新 backbone，而是围绕：

- raw
- fallback_80
- loop_to_hint
- replay15 rescue

做本地 `313-row reconstructed winner routing sweep`。

阶段性结果可以概括为：

- `A11`
  - 把 `loop_to_hint` 误伤压到 `0`
  - 但主 gate 仍然没有过 `I0`
- `A15`
  - 把 fallback 残余再压一层
  - `fullval / hard = 21.0807 / 21.4195`
- `A18`
  - 成为 score-best exploratory winner
  - `fullval / hard = 21.1205 / 21.4195`
  - 相对 `fallback_80` 提升约 `+0.8325 / +0.8058`
  - 但 `health_no_regression_vs_i0 = false`
- `A20`
  - 在执行单中被记为：
  - `current deploy-style preferred candidate`
  - 原因不是它分数最高，而是它比 `A18` 更稳
  - `fullval / hard = 21.1100 / 21.4195`
  - 保住了 `A15` 的健康面，不再额外恶化 short

这条线的本质：

- 它确实把本地 `313-row reconstructed geom` 推高了
- 但它一直没有把 `I0` 健康门拉绿
- 它属于局部 row-level surgery，不是官方提交路径复现

### A.2 A3C replay15 与 deploy chooser probe

这段时间还做了两类 chooser 试验。

第一类是 raw vs replay15 pair scope：

- 最佳安全变体是：
  - `C5_tree_gate_raw_repeat_or_conf45_formula1`
  - `fullval / hard = 20.2418 / 20.5735`
  - 相对 raw 有：
  - `+0.1519 / +0.1076`
  - `health_no_regression_vs_raw = true`

这说明：

- replay15 不是完全没价值
- 但价值只在很窄的 pair-overlap 子集上成立

第二类是把 chooser 接到 `A20 deploy subtree` 之后再测：

- 最佳仍然是：
  - `D0_ctrl_a20`
  - `changed_rows_vs_a20 = 0`

这说明：

- 广义 deploy chooser 并没有在 `A20` 上继续拿到净增益
- pair scope 的互补性，不能直接推导成 full deploy 增益

### A.3 A2R term-aware probe

还做了 term-aware / no-repeat patch probe。

结果是：

- 最佳仍然是：
  - `T0_ctrl_a20`
  - `patched_rows = 0`

这说明：

- 当前这版 term-aware patch 没有形成有效干预
- 它不是当前低分问题的主因，也不是当前救火主线

### A.4 A1X_P1 外部数据 DAPT/TAPT-lite smoke

这条线是这段时间最明确的红灯结果。

它的设置是：

- curated rows available = `1871`
- selected rows = `406`
- 在 `anchor64 reconstructed probe` 上比较：
  - `C0 = matched supervised only`
  - `D0 = DAPT-only`
  - `T0 = DAPT -> TAPT -> matched supervised`

结果是：

- incumbent anchor64：
  - `geom / bleu / chrf++ = 16.5057 / 9.8606 / 27.6291`
- `C0`：
  - `3.3953 / 1.4060 / 8.1993`
- `D0`：
  - `2.6390 / 0.8029 / 8.6734`
- `T0`：
  - `2.8349 / 0.8739 / 9.1965`

因此：

- `D0 vs C0 = -0.7563 geom`
- `T0 vs C0 = -0.5604 geom`

最终状态是：

- `review_stop`

这条线没有进入 fullval，并且已经证伪：

- 当前 formulation 的外部数据 continue training

## 附录 B：为何架构看起来类似高分段，但分数仍然很低

### B.1 不是 family 对了，recipe 就自动对

公开模型已经说明：

- `ByT5`
- `LoRA`
- `MBR`

这些大类本身不是负线。

问题在于：

- 同样叫 `ByT5 + LoRA`
- 并不代表：
  - 数据相同
  - chunking 相同
  - prompt 相同
  - decode 相同
  - candidate pool 相同
  - `MBR` 相同
  - rerank 相同
  - submission glue 相同

也就是说：

- 我们最近失败的只是：
  - 当前仓库这一版 recipe
- 不是：
  - `ByT5 + LoRA` 这个 family 本身不行

### B.2 最近主优化目标和官方公榜目标并不一致

最近主线推动的是：

- 本地 `313-row fullval reconstructed geom`

例如：

- `A18 = 21.1205`
- `A20 = 21.1100`

但官方给出的真实外部信号是：

- `Version 6 public score = 7.6`

这两者不是同一量纲。

更关键的是，本地历史 `official-like BLEU` 本来就只有：

- `7.7369`

所以真正的戏剧性断层，不是：

- `official-like BLEU 7.7` 对 `public 7.6`

而是：

- `local reconstructed geom 21.x` 对 `public 7.6`

这说明：

- 最近本地 winner 线优化到的，主要是本地代理信号
- 不是已经验证能映射到官方 leaderboard 的主目标

### B.3 我们最近做的是局部 surgery，不是高分提交链路

`A18 / A20` 的有效动作，本质上是：

- 在 `raw / fallback_80 / loop_to_hint / replay15`
- 之间做很窄的行级切换

这种做法的特点是：

- 可以在本地小评估集上显著推高诊断指标
- 但它没有回答：
  - 官方高分方案如何组织输入
  - 是否使用完整 `MBR`
  - 候选池如何生成
  - `rerank / postprocess / submission` 具体怎么做

因此：

- 即使局部 surgery 看起来连续涨分
- 也不能等价理解为：
  - 官方公榜提交路径已经被改好

### B.4 过早冻结了错误对象

最近还有一个技术判断错误：

- 把“当前仓库 formulation 的 `MBR / chooser / repair` 没打通”
- 误读成了“这类路线本身不值得优先”

现在看应当纠正为：

- 冻结的是：
  - 当前仓库已证伪 formulation
- 不应冻结的是：
  - 公开强基线里的 `MBR` 路线

换句话说：

- 近期不是架构 family 选错了
- 而是：
  - 评估主序错了
  - 提交 recipe 没对齐
  - 本地代理信号被过度放大了

## 附录 C：方差是否爆了，输出是否爆了

### C.1 不是优化数值爆炸，不是训练 loss 发散

从 `A1X_P1` 的训练日志看：

- DAPT 阶段 eval_loss：
  - `6.592 -> 6.334`
- C0 阶段 eval_loss：
  - `2.582 -> 2.254`
- D0 阶段 eval_loss：
  - `2.128 -> 2.072`
- T0 阶段 eval_loss：
  - `2.051 -> 1.996`

这些信号说明：

- 没有出现 `NaN`
- 没有出现 loss 突然爆炸
- 没有出现典型 optimizer variance run-away

所以：

- 这不是“训练数值不稳定导致的偶发坏种子”
- 不是通常意义上的“方差爆了”

### C.2 但输出端确实发生了明显塌缩与退化

真正爆掉的是输出模式，而不是训练 loss。

`A1X_P1` 三条 probe 都出现了明显的 decoder degeneration：

- prompt echo
  - `translate Akkadian to English ...`
  - `Akkadian to English ...`
- source / transliteration mode 侵入
  - 大量保留阿卡德语转写串
- 重复字符 loop
  - `ḫḫḫḫ...`
  - `ṣṣṣṣ...`
- 重复短语 loop
  - 同一句在一条输出中滚动复写

这不是轻微质量下降，而是输出模式切换到了错误轨道。

### C.3 不是纯空输出，也不是纯 copy-source，而是“模式错位”

如果只是空输出或纯拷源，诊断会更简单。

但这次不是。

以 `A1X_P1` 为例：

- `empty_prediction_ratio_pct` 仍只有：
  - `1.5625%`
- `copy_source_ratio_pct` 仍是：
  - `0.0%`

所以分数奇低，不是因为：

- 大面积空白
- 大面积逐字拷源

真正的问题是：

- 模型仍然在输出大量 token
- 但输出语义模式错了
- 它更像：
  - prompt echo
  - transliteration continuation
  - 字符/短语级 loop

这类错误会同时伤：

- BLEU
- chrF++
- reconstructed 聚合后的整体分数

### C.4 T0 比 D0 略高，但健康更差，说明不是随机抖动

`T0` 相对 `D0` 只回收了少量 geom：

- `delta_geom_t0_vs_d0 = +0.1959`

但它的健康面没有恢复，反而进一步说明模式塌缩具有方向性：

- `unique_prediction_ratio_pct`
  - 从 `C0` 的 `98.4375%`
  - 掉到 `T0` 的 `82.8125%`
- `health_vs_c0.no_regression = false`

这说明：

- 不是单次 decode 随机波动
- 不是简单 sampling 方差
- 而是模型被外部数据 continue training 推进到了错误输出盆地

### C.5 当前对“分数奇低”的技术判定

当前最合理的技术结论是：

1. `A1X_P1` 的极低分不是 optimizer 数值爆炸
2. 也不是单纯 seed 方差爆炸
3. 主要原因是：
   - 训练后输出模式塌缩
   - prompt echo / transliteration continuation / loop 增多
4. 再往上一层看，最近整条主线的低官方分，还叠加了：
   - 本地代理指标与官方目标错位
   - 提交 recipe 未对齐公开高分路线

因此：

- 近期低分既有：
  - 宏观上的评估主序错误
  - 也有：
    - 微观上的 decoder 输出模式退化
- 两者叠加，才形成了现在看到的：
  - 架构 family 看起来不离谱
  - 但真实公榜分数极低
