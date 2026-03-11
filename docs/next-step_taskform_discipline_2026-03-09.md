# Cloud Stage2 下一步纪律（Taskform Draft Fusion，2026-03-09）

## 0. 总原则

从现在开始，`taskform` 主线不再是重训当前 chunk 主模型。

冻结规则：

- `Pass-A` 固定为当前正式 winner：
  - checkpoint:
    - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
  - decode 固定：
    - `beam=4`
    - `lp=0.7`
    - `max_new_tokens=384`
- `Pass-B` 才是新模型。
- 主线不再重做已经稳定的 easy parents：
  - `1 chunk` 全部继续走 `Pass-A`
  - 大部分 `2-3 chunk` 继续走 `Pass-A`
  - 只有 routed hard parents 才进入 `Pass-B`
- 在任何新的 `Pass-B` 训练前，必须先过两类 feasibility gate：
  - 输出长度可行性
  - concat/edit baseline 可行性
- `Pass-B` 不再默认定义成“full rewrite 生成器”。
  - 优先定义成：
    - `Pass-A draft` 上的 merge/edit 模型
  - 目标是：
    - 尽量保留已有 draft 信息
    - 只做去重、补缝、顺序整理、少量重写

当前冻结正式 full-val reconstructed 指标：

- `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`

## 1. 路由纪律

### 1.1 目标

`Pass-B` 只处理这几类 hard parents：

- `4+ chunk`
- `ref_tok129+`
- `gap/tag-rich`

目标是只修 hardest buckets，不把已经做得好的 easy 区域重做坏。

### 1.2 操作化定义

为避免把 `has_gap` 一类高覆盖结构样本整体打进 `Pass-B`，本轮 routing 采用下面的**收紧实现**：

- `chunk_total >= 4`
- 或 `parent_ref_tok >= 129`
- 或 `marker_count >= 2` 且 `parent_ref_tok >= 96`

其中：

- `marker_count` 统计：
  - `has_gap`
  - `has_bracket`
  - `has_subscript`
  - `has_x`
- `parent_ref_tok` 在本轮 routing 里按 parent reference 的词级长度统计，不直接复用旧 bucket 报告里的 tokenizer 口径

说明：

- 这条 `tag-rich` 定义是本轮的工程化实现，不是语义上的唯一解释。
- 这样做的原因是：若直接用 `has_gap` 或 `has_subscript`，会把过多 `2-3 chunk` parent 一并路由，破坏“`1 chunk` 和大部分 `2-3 chunk` 继续走 `Pass-A`”这条总原则。
- 本轮 routing 基数以基础 processed split 为准，不再沿用旧 bucket 报告里已经漂移的 `parent_chunks` 统计。

## 2. 并行纪律

统一阶段纪律：

- `P -> W -> F`
- 不允许跨段抢跑：
  - `P` 没 gate 通过，不进 `W`
  - `W-lite` 没 gate 通过，不进 `F-lite`

同段内的并行纪律：

- 如果子任务没有内承关系，可以放后台
- 但本机当前只有 `1` 张 `RTX 5090 32GB`
- 因此本轮默认策略是：
  - GPU 训练主链串行
  - CPU/轻量汇总并行
  - 独立 decode/summary 可后台

## 3. dan-1：Draft Fusion

### 3.1 目标

验证：

- 当前 winner 的 chunk draft
- 再加一个 parent-level merge/edit 模型

能否吃掉 routed hardest buckets。

### 3.2 Pass-A

`Pass-A` 负责：

- 用当前正式 winner 生成 routed hard parents 的 chunk-level English draft
- 产出 draft cache

本轮明确限制：

- 不生成全训练集 draft cache
- 只生成 routed hard-parent 子集
- val 端优先复用现成 winner full-val 诊断产物

### 3.3 Pass-B

模型优先级：

1. `google/flan-t5-small`
2. 若环境不顺再退 `t5-small`

本机当前已确认 `flan-t5-small` 可用，因此先按 `flan-t5-small` 执行。

`Pass-B` 的任务定义从这一版开始收紧为两层：

1. 无训练 baseline：
   - `concat_pred_drafts`
   - `dedup_concat_pred_drafts`
2. learned edit：
   - 在 concat/dedup draft 基础上做 parent-level edit

输入纪律：

- 不再默认把 `chunk_idx` 标签直接喂给模型
- 不再默认使用 `c1:` / `c2:` / XML chunk tag 这类容易泄漏到输出的格式
- 不再用“remove repeated boilerplate”作为核心指令
  - 因为本任务中大量重复表面句式本身就是正文
- 优先输入：
  - `combined draft`
  - 必要时再加：
    - 极简 chunk boundary 提示
    - 极简 source hint

输出纪律：

- 输出必须是：
  - `parent-level final English`
- 不得输出：
  - `c1:` 等 chunk 标签
  - prompt 中间标记
  - 对 chunk 的逐条抄录

长度预算纪律：

- `Pass-B` 配置不再复用 chunk 任务的默认输出长度
- 训练前必须先统计：
  - val anchor target token 分布
  - smoke train target token 分布
- 若 `target > max_new_tokens` 或 `target > max_target_length` 的比例超阈值：
  - 先修预算，再训练
- 当前 parent 级默认起点应至少满足：
  - `max_target_length >= 1024`
  - `max_new_tokens >= 768`
  - 或按当轮 `p95` 动态设定

### 3.4 P 纪律

从这一版开始，`P` 段拆成三层：

- `P0`: feasibility + no-train baselines
- `P1`: edit smoke
- `P2`: probe250

#### P0：feasibility + no-train baselines

- `P0_1`: `target_length_audit`
  - 统计：
    - `ref_over_max_new_tokens`
    - `ref_over_max_target_length`
- `P0_2`: `concat_pred_baseline_anchor64`
  - 直接按原 chunk 顺序拼接 `Pass-A` drafts
- `P0_3`: `dedup_concat_pred_baseline_anchor64`
  - 在拼接前仅做轻量去重/清理
- `P0_4`: `concat_oracle_ceiling_anchor64`
  - 只用于测可达上限

`P0` gate：

- 若 `ref_over_max_new_tokens > 10%`
  - `hard_stop_repair_budget`
- 若 `ref_over_max_target_length > 5%`
  - `hard_stop_repair_budget`
- 若 learned 模型的目标仍未定义清楚：
  - 不进入训练

#### P1：edit smoke

- `P1_1`: `edit_smoke_from_concat_pred`
- `P1_2`: `edit_smoke_from_dedup_concat_pred`
- `P1_3`: `prompt_format_ablation`
  - 比较：
    - `flat unlabeled`
    - `bullet merge`
    - `combined-draft edit`
- `P1_4`: `copy_suppression_ablation`
  - prompt 明示：
    - 不要输出 chunk 标签
    - 不要逐条复述 drafts
- `P1_5`: `best_smoke_vs_baselines`

`P1` gate：

- 基准不再只看 old `pred_smoke`
- 必须同时看：
  - routed matched baseline
  - `concat_pred_baseline`
  - `dedup_concat_pred_baseline`
- 若 best smoke 连 `concat_pred_baseline` 都打不过：
  - `reject_stop`
- 若 best smoke 仅略好于 concat，但仍明显低于 matched：
  - `review_stop`
- 只有当 best smoke：
  - `>= concat_pred_baseline + 0.5`
  - 且 `short` 不比 concat baseline 恶化 `+5.0` 以上
  - 且与 matched baseline 的差距收敛到 `-1.5` 以内
  - 才允许进 `probe250`

#### P2：probe250

- `P2_1`: `routed_edit_probe250`
- `P2_2`: `ckpt150 / ckpt200 / ckpt250 @ anchor64`
- `P2_3`: `winner diag64`

Probe gate：

- 默认 matched baseline：
  - routed full
  - routed anchor64
- 若 `anchor64 geom` 低于 `concat_pred_baseline`：
  - `reject_stop`
- 若高于 concat 但相对 matched 没有稳定正信号：
  - `review_stop`
- 若 `anchor64 geom` 有明确正信号，且 `empty/short/repeat` 未明显恶化：
  - `accept_to_wlite`

本轮自动 gate 阈值：

- `delta_geom >= +0.25`
- `empty` 不恶化
- `short` 允许至多 `+2.0`
- `repeat` 允许至多 `+2.0`

### 3.5 W-lite 纪律

- `W1_1`: 从 probe winner 继续到 `400 steps`
- `W1_2`: 比较 `ckpt250 / ckpt300 / ckpt350 / ckpt400`
- `W1_3`: `anchor96`
- `W1_4`: `diag96`

自动 gate：

- 相对 routed matched baseline `geom >= +0.5`
- `repeat / short / empty` 不恶化

通过后：

- `go_to_flite`

否则：

- `review_stop` 或 `reject_stop`

### 3.6 F-lite 纪律

- `F1_1`: routed subset full-val
- `F1_2`: mixed output full-val
  - easy parents 用 `Pass-A`
  - hard parents 用 `Pass-B`
- `F1_3`: diagnose
- `F1_4`: bucket bridge report

正式 promote 纪律：

- 脚本自动生成：
  - `summary.json`
  - `gate_report.md`
  - `promote_compare.json`
  - `promote_compare.md`
- 最终是否替换正式主线：
  - 仍保留人工确认

## 4. dan-2：Source-Aware Draft Fusion

### 4.1 目标

验证：

- 只看 English draft 不够
- 需要同时看 source 结构提示

### 4.2 Pass-A

同 `dan-1`：

- 继续复用当前正式 winner
- 继续复用 `dan-1` 的 routed draft cache

### 4.3 Pass-B

模型：

- 同样优先 `flan-t5-small`

`dan-2` 不再建立在“per-chunk full rewrite fusion”上，而建立在：

- `dan-1` 已稳定的 edit 目标
- 再增加最小必要 source hint

输入：

- `combined draft`
- `source hint`
- `transliteration` 头尾片段
- `has_gap / has_bracket / has_subscript / has_x`

输出：

- `parent-level final English`

### 4.4 P 纪律

- `P2_1`: `draft+hint_edit_smoke`
- `P2_2`: `draft_only_vs_hint_ablation`
  - 同子集比较：
    - `dan-1 best edit smoke`
    - `dan-2 hint edit smoke`
- `P2_3`: `routed_pred_hint_probe250`
- `P2_4`: `ckpt150 / ckpt200 / ckpt250 @ anchor64`
- `P2_5`: `winner diag64`

### 4.5 执行优先级

`dan-2` 不与 `dan-1` 同时起 full chain。

本轮执行策略固定为：

- 先跑 `dan-1`
- 只有 `dan-1` edit smoke 和 probe 都给出真实正信号，才放行 `dan-2` probe
- `dan-2` 是否值得进入更深阶段，先看：
  - `P2_2 draft_only_vs_hint_ablation`
  - `P2` 相对 matched baseline 的正信号

### 4.6 F-lite 前置条件

`dan-2` 只有在**明显优于 `dan-1`** 时，才值得跑自己的 `F-lite`。

本轮优先级：

- `dan-2 probe` 可以直接挂后台等待
- `dan-2` 的更深阶段不抢在 `dan-1` 全链之前

## 5. 运行与产物

本轮已固定的脚本：

- 数据构建：
  - `/workspace/deep-past-/scripts/build_dan1_data.py`
  - `/workspace/deep-past-/scripts/build_dan2_data.py`
- 评估与 gate：
  - `/workspace/deep-past-/scripts/evaluate_dan1_flow.py`
  - `/workspace/deep-past-/scripts/evaluate_dan2_flow.py`
- concat cleanup / rerank：
  - `/workspace/deep-past-/scripts/run_dan1_concat_cleanup.py`
- 后台队列：
  - `/workspace/deep-past-/tmp/run_dan1_probe.sh`
  - `/workspace/deep-past-/tmp/run_dan2_probe.sh`

统一配置：

- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan1_oracle_smoke.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan1_pred_smoke.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan1_routed_probe.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan1_routed_wlite.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan2_hint_smoke.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan2_routed_probe.yaml`
- `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan2_routed_wlite.yaml`

脚本自动负责：

- 生成 matched baseline
- 选择 checkpoint winner
- 判断：
  - `accept_to_wlite`
  - `review_stop`
  - `reject_stop`
- 再判断：
  - `go_to_flite`

统一报告目录：

- `dan-1 probe`:
  - `/workspace/deep-past-/reports/taskform_dan1_p`
- `dan-1 w-lite`:
  - `/workspace/deep-past-/reports/taskform_dan1_w`
- `dan-1 f-lite`:
  - `/workspace/deep-past-/reports/taskform_dan1_f`
- `dan-2 probe`:
  - `/workspace/deep-past-/reports/taskform_dan2_p`
- `dan-1 concat cleanup / rerank`:
  - `/workspace/deep-past-/reports/taskform_dan1_b1_b2_b4`

## 6. 当前执行结论

### 6.1 已完成结论

`dan-1` 首轮 `probe` 已跑完并在 `P` 段停止：

- routed matched `anchor64 geom = 13.2798`
- `oracle_fuse_smoke geom = 29.8351`
- `pred_fuse_smoke geom = 3.0628`
- `ckpt150 / 200 / 250 = 4.5109 / 4.2934 / 4.1993`
- 结论：
  - `reject_stop`

解释：

- fusion 假设本身不是死路：
  - `oracle` 上限高
- 失败点主要在：
  - `Pass-A draft -> Pass-B prompt`

### 6.2 2026-03-10 修正重跑（R1 + R2 + P1b）

本轮已完成一次只改输入、不进新 probe 的修正 smoke：

- `R1 Prompt Audit`
  - XML prompt 改为 flat prompt
  - 加固定出口：
    - `Final translation:`
- `R2 Draft Sanitize`
  - 清理 prompt markup
  - 压重复词串
  - 按 chunk 做词数预算截断
- `P1b ShortPrompt Smoke`
  - 只重跑 `pred_fuse_smoke`
  - 不进入 `probe250`

v2 结果：

- 新配置：
  - `/workspace/deep-past-/runs/STEER/generated_configs/taskform_dan1_pred_smoke_v2.yaml`
- 新脚本：
  - `/workspace/deep-past-/tmp/run_dan1_pred_smoke_v2.sh`
- 新报告目录：
  - `/workspace/deep-past-/reports/taskform_dan1_p_v2`

长度统计已明显修复：

- `rows = 320`
- `src_tok_p50 / p95 / max = 188 / 333 / 466`
- `over_512 = 0`
- `over_768 = 0`
- `over_1024 = 0`

但指标仍未恢复：

- old `pred_smoke geom = 3.0628`
- new `pred_smoke geom = 3.9179`
- `delta new vs old = +0.8551`
- `delta new vs matched = -9.3619`
- `short = 84.375%`

当前判断：

- 输入长度超限已经不是主矛盾
- 但输出长度预算仍是硬问题
- `Pass-B` 仍在复述 chunk drafts，而不是生成 parent final translation
- 现象包括：
  - 输出保留 `c1: / c2:` 结构
  - 高频 `Seal of ...` 样式重复
  - `<gap>` 被劣化成 `gap>`

因此：

- `dan-1` 仍然停在 `P` 段
- `dan-2` 继续 blocked
- 现阶段不允许重开 `W-lite / F-lite`

### 6.3 反常低分的定性结论

本轮已经确认，当前异常不是简单的“prompt 太长”，而是多层问题叠加：

- 现有 `Pass-B` 比 no-train concat baseline 更差
  - `concat_pred_baseline_anchor64 geom = 11.4449`
  - v2 best smoke 只有 `3.9179`
- hard anchor 的 parent targets 明显长于当前输出预算
  - `27 / 64` 超过 `max_new_tokens=384`
  - `8 / 64` 超过 `max_target_length=640`
- prompt 把模型引向了“逐条摘录 chunk”而不是“合并成 parent final translation”

因此当前主结论改为：

- 失败点首先是：
  - 任务定义错位
  - 输出预算错位
- 不是：
  - fusion 假设已经被否定

### 6.4 修订后的下一步方案

下一轮固定改为四步：

- `A1 Feasibility Audit`
  - 先出：
    - `target_length_audit`
    - `concat_pred_baseline`
    - `dedup_concat_pred_baseline`
    - `concat_oracle_ceiling`
- `A2 Budget Repair`
  - 把 `Pass-B` 输出预算切到 parent 级
  - 不再沿用 chunk 任务默认值
- `A3 Edit Smoke`
  - 不再做 per-chunk rewrite prompt
  - 改做 combined-draft edit
  - 去掉 chunk 标签暴露
- `A4 Bucket Split`
  - 分开验证：
    - `4-6 chunk`
    - `7+ chunk`
    - `2-3 chunk 且 ref129+/tag-rich`

新的硬 gate：

- 若 `A1` 发现输出预算仍不覆盖大多数 anchor targets：
  - 不得训练
- 若 `A3` best smoke 仍打不过 `concat_pred_baseline`：
  - 不得进入 `probe250`
- 若 `A3` 只在单一桶里有效：
  - 只放行该桶，不放大全路由

`dan-2` 的放行条件同步上调：

- 只有 `dan-1` 先证明：
  - edit 目标可行
  - 输出预算可行
  - smoke 至少超过 concat baseline
- `dan-2` 才值得起自己的 hint ablation

### 6.5 A1-A3 实跑结论（2026-03-10）

本轮已按修订纪律完成：

- `A1 Feasibility Audit`
- `A2 Budget Repair`
- `A3 Edit Smoke`

结果：

- `target_length_audit`
  - val anchor `ref_over_max_new_tokens = 3 / 64 = 4.69%`
  - val anchor `ref_over_max_target_length = 0 / 64 = 0%`
  - 说明：
    - parent 级预算已基本可行
- `concat_pred_baseline_anchor64 geom = 11.4449`
- `dedup_concat_pred_baseline_anchor64 geom = 9.1836`
- `concat_oracle_ceiling_anchor64 geom = 78.5647`
- `edit_smoke_anchor64 geom = 6.3961`

解释：

- `A2` 修掉了预算错位
- `A3` 也去掉了 chunk label 泄漏
- 但 learned edit 仍明显打不过 `concat_pred_baseline`

这说明当前主矛盾已经进一步收敛为：

- 不是：
  - 输入长度不够
  - chunk 标签泄漏
- 而是：
  - 模型在训练目标上仍倾向于高频公式自激循环
  - learned edit 仍在破坏 `Pass-A` 已有信号

快速反例：

- 对同一 `best_model` 仅加 decode 约束：
  - `no_repeat_ngram_size = 3`
- 结果：
  - `geom = 0.9837`
  - `short = 98.44%`

因此明确禁止：

- 把硬 `no_repeat_ngram` 当作默认修复方案

保留的新增能力：

- 评估已新增样本内循环检测：
  - `internal_repeat_trigram_ratio_pct`

下一轮应继续坚持：

- 先拿 no-train baseline 做主参考
- learned model 必须先超过 `concat_pred_baseline`
- 若做新尝试，优先改：
  - 训练目标
  - candidate 生成方式
  - edit 约束形式
  - 而不是先堆更硬的 decode 惩罚

一句话：

- 冻结当前 winner 做 `Pass-A`
- `dan-1` 先修任务定义和输出预算，再决定是否重开 probe
- `dan-2` 继续等待，不能抢跑

### 6.6 B1-B2-B4 实跑结论（2026-03-10）

本轮已执行：

- `B1 Bucket Audit`
- `B2 Concat Cleanup`
- `B4 Candidate Rerank`

核心结论：

- `concat_pred` 方向是对的
- 但必须按桶处理
- 当前 cleanup/rerank 仍不能整体替代正式 `Pass-A`

anchor64 对比：

- `raw_concat geom = 11.5044`
- `looptrim_concat geom = 12.5232`
- `looptrim_chunkdedup_concat geom = 12.3262`
- `rerank_prediction geom = 12.3262`
- `pass_a_prediction geom = 13.2798`

routed full 对比：

- `raw_concat geom = 11.7911`
- `looptrim_concat geom = 13.1063`
- `looptrim_chunkdedup_concat geom = 12.9053`
- `rerank_prediction geom = 12.9132`
- `pass_a_prediction geom = 13.7161`

正式 mixed full-val 对比：

- 当前正式主线：
  - `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`
- `official_mix_rerank`：
  - `geom / bleu / chrfpp = 14.1843 / 7.7388 / 25.9980`

解释：

- `looptrim_concat` 是当前最强 no-train cleanup 候选
- 但它仍低于 routed hard 上的 `Pass-A`
- `rerank` 也没有把 routed full 拉过 `Pass-A`
- 因此本轮结论是：
  - `review_stop`
  - 不 promote
  - 不替换正式主线

桶级结果：

- `chunk4_6`
  - `raw_concat geom = 11.7089`
  - `rerank_prediction geom = 14.3570`
- `chunk2_3_long_or_tag`
  - `raw_concat geom = 12.7530`
  - `rerank_prediction geom = 14.0183`
- `chunk7plus`
  - `raw_concat geom = 10.1750`
  - `rerank_prediction geom = 7.5238`

因此新增硬纪律：

- 不允许再对 `7+ chunk` 使用当前这版 `looptrim/rerank`
- 若后续继续沿 `concat cleanup` 线推进：
  - 默认只允许在
    - `4-6 chunk`
    - `2-3 chunk 且 ref129+/tag-rich`
  - 这两桶内继续试
- `7+ chunk` 必须改走：
  - 更保守的 `raw concat`
  - 或新的 selective edit / hierarchical merge

### 6.7 下一步执行顺序

下一轮不重开 learned full edit smoke。

固定顺序改为：

- `B3 Selective Edit`
  - 只修被检测出的坏 span
  - 不重写整段
- `B5 Bucketed Routing`
  - 把 cleanup/edit 放行限制在有效桶
- `B6 Hierarchical Merge`
  - 只服务 `7+ chunk`
  - 先局部合并，再 parent 合并

放行纪律：

- 任一新候选必须先超过同桶 `raw_concat`
- 若要替代正式 routed `Pass-A`，还必须超过同桶 `Pass-A`
- 若只对单桶有效：
  - 只允许该桶进入 mixed output

这份文档从此替换旧的 `parentpack / replay / proxy` 主线描述。
