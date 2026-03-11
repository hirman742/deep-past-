# Cloud Stage2 下一步纪律（Winner 再版，2026-03-10）

## 0. 结论先行

这次再版后的主线不再围绕：

- `dan-1 / dan-2` parent-level fusion
- `L2 / L3` 后处理修补
- 当前 `A2` 过滤器
- 当前 fair 对照下已经判负的 `TAPT -> supervised`
- 在旧 recipe 上重开 `ByT5-base / mT5-base`

原因已经足够清楚：

- 这些线要么已经被硬证伪
- 要么只显示了局部、弱、不可推广的信号
- 要么根本没有触及当前 winner 真正成功的原因

当前正式 winner 仍冻结为：

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

截至本版纪律实际执行完的状态回填如下：

- `A1_P0` 已完成：
  - `source_registry / overlap_audit / dedup_manifest / mix_plan` 已落盘
  - `silver external csv` 已落盘：
    - `data/external/oracc_parallel.csv`
  - 当前状态：
    - `ready_for_mix_build`
  - 当前外部规模：
    - `rows_post_clean = 7481`
    - `fold0_val_exact_overlap_rows = 0`
    - `test_exact_overlap_rows = 0`
    - `train_exact_overlap_rows = 0`
    - `E10 / E30 / E50 = all ready`
- `A1` 旧 smoke 已判无效：
  - 产物：
    - `reports/taskform_winner_a1_smoke_20260310/summary.json`
  - 问题：
    - 它使用的是 plain processed dir
    - 从 fresh `ByT5-small + LoRA` 起训
    - 没接当前 `retrieval-top1 W-lite` adapter
  - 纪律结论：
    - 这轮只可作为“silver 比例先验”的弱信号
    - 不可用于 winner 主线判定
    - 后续 `A1` 只认 `continue-on-wlite` 口径
- `A1 continue-on-wlite overnight` 已完成但当前不放行：
  - 产物：
    - `reports/taskform_winner_a1_continue_build_20260310/summary.json`
    - `reports/taskform_winner_a1_continue_probe_20260310/summary.json`
    - `reports/taskform_winner_a1_continue_overnight_20260310/status.json`
  - probe 结论：
    - `e5 / e10 / e15` 全部输给 `internal_only_matched`
    - best ratio status:
      - `control_only`
  - 当前状态：
    - `stopped_after_probe`
  - 纪律结论：
    - 当前 `external mix continue-on-wlite` 先归档为负
    - 不再继续自动接 `W-lite / promote`
- `A2 retrieval-top1 W-lite` 已完成到 full-val：
  - `anchor64 geom = 23.5422`
  - `full-val reconstructed geom = 19.9956`
  - `hard geom = 20.8432`
  - 相对 incumbent：
    - `anchor64 +7.0365`
    - `full-val +5.6633`
    - `hard +7.1271`
  - 当前状态：
    - `review_for_f`
  - 原因：
    - 主分显著转正，但 full-val 输出健康仍需人工复核
- `A2 health surgical` 已完成：
  - 推荐候选：
    - `fallback_180`
  - `full-val reconstructed geom = 19.9035`
  - `hard geom = 20.7888`
  - 相对 raw W-lite：
    - `full-val -0.0872`
  - health：
    - `full-val no_regression vs I0 = true`
  - 当前状态：
    - `candidate_frozen_manual_promote_recommended`
  - freeze bundle：
    - `reports/taskform_winner_a2_freeze_20260310/summary.json`
- `A2g retrieval selective gate` 已完成到 partial probe，但当前不放行：
  - 产物：
    - `reports/taskform_winner_a2g_build_20260311/summary.json`
    - `reports/taskform_winner_a2g_probe_20260311/summary.json`
  - 当前对照：
    - `ctrl anchor64 / hard = 22.3155 / 23.1564`
    - `g35 anchor64 / hard = 18.8861 / 19.4744`
    - `g35 - ctrl anchor64 / hard = -3.4294 / -3.6820`
  - 额外信号：
    - `health no_regression = false`
    - `rare / measure / formula / marker_rich` 全负
  - 当前状态：
    - `stopped_after_g35`
  - 纪律结论：
    - 不再继续补跑 `g50 / g65`
    - 不再继续在 retrieval 同质 gate 变体上横向加臂
- `competition-only pseudo-target / denoising continue smoke` 已完成并判负：
  - 产物：
    - `reports/taskform_winner_a1_pseudotarget_smoke_20260311/summary.json`
    - `reports/taskform_winner_a1_pseudotarget_smoke_20260311/gate_report.md`
    - `docs/taskform_winner_pseudotarget_smoke_2026-03-11.md`
  - probe 对照：
    - `incumbent anchor64 geom = 16.5057`
    - `probe anchor64 geom = 2.5066`
    - `probe hard geom = 2.7485`
    - `delta vs incumbent anchor64 = -13.9992`
    - `delta vs frozen anchor64 = -21.0349`
  - 失败形态：
    - 输出明显塌缩为重复字符 / `<gap>` 堆叠 / instruction echo
    - 问题出在 pseudo-target synthetic supervision，而不是“还没来得及 full-val”
  - 当前状态：
    - `completed`
    - `reject_stop`
  - 纪律结论：
    - 不补 `full-val`
    - 不进入 `candidate_pool_long_train`
    - 当前 pseudo-target synthetic line 先归档为负
    - 下一条主线改为：
      - `competition-only denoising-only continue smoke`
- `competition-only denoising-only continue smoke` 已有公平对照负证据：
  - 产物：
    - `reports/taskform_tapt_fair_20260310/summary.json`
  - 当前对照：
    - `T0_tapt_then_supervised - C0_no_tapt anchor64 = -0.5435`
    - `T0_tapt_then_supervised - I0 anchor64 = -13.9623`
  - 额外信号：
    - `health_t0_vs_c0 no_regression = false`
    - 输出同样出现 `<gap>` 堆叠 / 重复字符 / instruction echo
  - 当前状态：
    - `review_stop`
  - 纪律结论：
    - 当前 `competition-only mono` 轴整体不再重跑同口径
    - 不再继续 `denoising-only` / 原始 `pseudo-target synthetic mix` 变体
    - 下一条主线改为：
      - `winner replay / curriculum probe`
- `winner replay / curriculum probe` 已完成并给出 candidate-pool 级正信号：
  - 产物：
    - `reports/taskform_winner_replay_probe_20260311/summary.json`
    - `reports/taskform_winner_replay_probe_20260311/gate_report.md`
    - `docs/taskform_winner_replay_probe_2026-03-11.md`
  - 当前对照：
    - `ctrl anchor64 / hard = 15.3962 / 14.5055`
    - `replay25 anchor64 / hard = 15.9949 / 15.9245`
    - `replay25 - ctrl anchor64 / hard = +0.5987 / +1.4190`
    - `replay25 - incumbent anchor64 = -0.5108`
    - `replay25 - frozen anchor64 = -7.5466`
    - `replay40 anchor64 / hard = 15.5201 / 15.0942`
    - `replay40 - ctrl anchor64 / hard = +0.1239 / +0.5887`
  - 额外信号：
    - `replay25 health no_regression = true`
    - `replay25 reconstructed_health no_regression = true`
    - `replay40 health no_regression = false`
  - 当前状态：
    - `review_to_candidate_pool`
    - `best_label = replay25`
  - 纪律结论：
    - `replay25` 可以跳过当轮 `W-lite / full-val`，直接进入：
      - `candidate_pool_long_train`
    - 这条长训练只用于实测全口径表现，不得写成：
      - `F/promote`
    - `replay40` 当前归档，不再作为长训练候选
- `winner_train_main_20260311` 已全部完成：
  - `raw retrieval W-lite long train`
    - `anchor64 / full-val / hard = 23.2666 / 20.0899 / 20.4659`
    - 相对 `fallback_180`：
      - `full-val +0.1864`
      - `hard -0.3229`
    - 额外信号：
      - `health no_regression vs incumbent = false`
    - 当前状态：
      - `completed_review_pending`
    - 纪律结论：
      - 它是当前最强的 raw 单模型 score ceiling
      - 但还不是可直接 promote 的 health-safe 候选
      - 当前更合理的定位是：
        - `research ceiling`
        - `candidate_pool / compare reference`
  - `replay25 candidate-pool long train`
    - `anchor64 / full-val / hard = 15.6636 / 14.3271 / 13.6466`
    - 相对 incumbent：
      - `anchor64 -0.8422`
      - `full-val -0.0052`
      - `hard -0.0695`
    - 相对 `fallback_180`：
      - `full-val -5.5764`
      - `hard -7.1422`
    - 当前状态：
      - `completed_review_pending`
    - 纪律结论：
      - `replay25` 作为单模型主线不成立
      - 只能保留为：
        - `candidate_pool complementary axis`
- `orthogonal small probes` 已全部完成：
  - `retrieval-top1 + replay25 combo probe`
    - `combo - ctrl anchor64 / hard = -0.1518 / -0.1463`
    - 当前状态：
      - `reject_stop`
    - 纪律结论：
      - 不再继续同口径 combo long train
  - `replay15 / replay20 / replay30` 窄扫
    - `replay15`：
      - `anchor64 / hard delta vs ctrl = +0.2817 / +0.9581`
      - `health no_regression = true`
    - `replay20`：
      - 轻微分数变化但 `health no_regression = false`
    - `replay30`：
      - health 不脏，但 anchor 增益不足
    - 当前状态：
      - `review_to_candidate_pool`
      - `best_label = replay15`
    - 纪律结论：
      - replay 轴只在轻量配比上保留研究价值
      - 不再往更重 replay 比例扩展
  - `post-probe full-val decode`
    - 只实际评到了：
      - `replay15`
    - `replay15 full-val / hard = 14.0434 / 13.7119`
    - 相对 incumbent：
      - `full-val -0.2888`
      - `hard -0.0042`
      - `health no_regression = true`
    - 当前状态：
      - `completed_review_pending`
    - 纪律结论：
      - `replay15` 说明 replay 轴有 health-safe 局部信号
      - 但不足以成为新的单模型 winner
  - `A3 cheap revisit`
    - `retrieval_raw_longtrain` 与 `replay15` 在共同 17 行上：
      - `exact overlap = 0%`
      - `oracle delta vs best single = +2.1503`
      - `heuristic delta vs best single = 0.0`
    - 当前状态：
      - `completed_revisit_audit`
    - 纪律结论：
      - 候选池互补性是真实存在的
      - 问题不在“有没有互补”，而在“当前 selector 还吃不到它”
- `official bridge / report glue` 已后台跑完：
  - session：
    - `winner_bridge_20260311`
  - 产物：
    - `reports/taskform_winner_bridge_20260311/official_metric_probe.json`
  - 当前结果：
    - `status = missing_bridge`
  - 纪律结论：
    - 当前仍只能把 `official-like` 当本地代理层
    - 不能把它写成“等同官方”
- 当前后台状态：
  - 本轮 `winner_train_main_20260311`
  - `winner_postmain_probe_20260311`
  - `winner_postprobe_decode_20260311`
  - `winner_a3_revisit_20260311`
  - `winner_smallprep_20260311`
  - 均已收尾
  - 当前无活跃 GPU 训练 / decode 会话
- `A3_P0` 已完成并为正：
  - `unique_candidate_ratio_pct = 92.1569`
  - `incumbent vs retrieval_wlite exact overlap = 0%`
- `A3_P1/P2` 已完成但当前不放行：
  - `best_single geom = 24.6330`
  - `MBR geom = 23.6760`
  - `delta = -0.9570`
  - 当前状态：
    - `review_stop`
- `RK` proxy probe 已完成但当前不放行：
  - 虽有 infra 正信号，但 best proxy reconstructed `22.4518 < 23.5422`
  - 不能抢占下一条主优先 GPU 线
- `RK_true_hook` 主路径接线与 smoke 已完成但当前仍不放行：
  - formal `decode / diagnose` 已打通
  - 8-row baseline reconstructed `geom = 18.4946`
  - true-hook `alpha=1.5` reconstructed `geom = 16.6067`
  - weak true-hook `alpha=0.5, steps=32` 也只有 `geom = 17.5879`
  - 当前状态：
    - `parked_negative_smoke`
  - 结论：
    - 当前 token-vote interpolation formulation 不应进入更大口径 GPU 主线

### 0.1 历史尝试总账

截至 `2026-03-11` 当前仓库快照，历史尝试可以压缩成三类：

- 已明确证伪或归档的线：
  - `dan-1 / dan-2` parent-level fusion
  - `L2 / L3` 后处理主线
  - 当前 fair 对照下的 `TAPT -> supervised`
  - `A1 external mix continue-on-wlite`
  - `A2g retrieval selective gate`
  - `competition-only pseudo-target synthetic mix`
  - `competition-only denoising-only continue`
  - `retrieval + replay25 combo`
  - 当前 formulation 的 `A3_P1/P2 MBR`
  - `RK proxy / RK true-hook token-vote`
- 已证明有效，但当前只适合保留为基线 / 候选池证据的线：
  - `A2 retrieval-top1 W-lite`
    - 证明 retrieval 主线成立
  - `A2 health surgical / fallback_180`
    - 证明 retrieval 主线可以被修成 health-safe compare baseline
  - `winner replay / curriculum probe`
    - 证明轻 replay 有局部正信号
  - `A3 cheap revisit`
    - 证明 retrieval 与 replay 之间确实存在互补空间
- 已完成但未成为 promote 候选的长线：
  - `raw retrieval longtrain`
    - 分数最高，但 `hard / health` 未同时过线
  - `replay25 candidate longtrain`
    - 逼近 incumbent，但未赢 incumbent，更远低于 frozen retrieval
  - `replay15 post-probe fullval`
    - health 绿，但单模型分数不够

### 0.2 当前现状

当前状态已经不是“还有后台长训练在跑”，而是：

- 本轮云端训练与后评估已全部收尾
- 当前工作基线仍应保持：
  - `fallback_180`
- 当前 raw 单模型 score ceiling 是：
  - `winner_retrieval_raw_wlite_longtrain`
- 当前 replay 轴的最准确定性是：
  - `candidate-pool complementary signal`
  - 不是新的单模型主胜轴
- 当前 `A3` 的最准确定性是：
  - 互补性存在
  - 但当前 selector / MBR formulation 不成立
- 当前 `official bridge` 仍缺失：
  - 只能继续使用 `official-like` 本地代理

因此，当前阶段应明确切换为：

- 先下云端算力
- 做离线分析 / 调研
- 明确下一轮只回云端做**真正正交**的新迭代

### 0.3 当前仍可能的突破口

基于现有证据，下一轮真正值得赌的突破口只剩三类：

1. `retrieval` 主线上的 health-constrained continue / surgical repair
   - 证据：
     - `raw retrieval longtrain` 已把 `full-val` 抬到 `20.0899`
     - 但 `hard vs frozen = -0.3229`
     - `health no_regression vs incumbent = false`
   - 解释：
     - 问题不是 retrieval 无效
     - 而是 retrieval 继续训练后，少量 generic / repeat / 短输出行开始拖后腿
   - 下一轮若重开 retrieval，必须以：
     - `health-aware objective`
     - 或 `changed-row surgical repair`
     - 或 `fallback-compatible selector`
     为中心，而不是再做同质 gate / 同质长训
2. `retrieval_raw_longtrain + replay15` 的 candidate selector / pairwise chooser
   - 证据：
     - `A3 cheap revisit` 显示二者 `exact overlap = 0%`
     - `oracle delta vs best single = +2.1503`
     - 但当前 heuristic `delta = 0.0`
   - 解释：
     - 候选池不是没互补
     - 而是当前还没有一个足够好的可部署选择器
   - 这条线比继续堆新的 replay 单模型更值得
3. 轻量 replay 作为 retrieval 主线的辅因子，而不是独立主线
   - 证据：
     - `replay15` probe 为正且 health 绿
     - 但一进 full-val 就没有赢 incumbent
     - `replay25 longtrain` 也没有转成新 winner
   - 解释：
     - replay 更像 regularizer / candidate source
     - 不是当前最强的单模型终局方向

不再建议继续赌的方向也应写死：

- 不回到 `A1 external mix`
- 不回到 `A2g selective gate`
- 不回到 `competition-only mono`
- 不回到当前 `MBR` formulation
- 不回到当前 `RK token-vote`

当前轮次的总目标也重新写死：

- 不是再做“小修能不能涨 0.1”
- 不是再赌一个未经证明的重写模型
- 而是要**狠狠干一条最可能把 `geom` 抬到下一层的主线**

因此从这版开始，winner 提分主线按下面顺序重排：

1. 外部平行数据 / 域扩展混训
2. retrieval / `kNN-MT` / 记忆增强式 chunk 建模
3. 多样候选池 + efficient `MBR` + 真正有互补性的 `OOF / ensemble`
4. 只有在更强 recipe 已被证明有效后，才允许重开新的 backbone 探测

这四条的共同原则是：

- 不推翻当前 winner 的 chunk formulation
- 不破坏 `short_aligned` 这一真实有效监督结构
- 不再把问题抬回已被证明容易失控的 parent rewrite
- 不再把时间耗在只会局部止损、不能改上限的 patch 线上

## 1. 共性纪律

### 1.1 winner 冻结与公平对照纪律

从这一版开始，任何新线都必须把当前 winner 当成固定 `I0`：

- `I0 = incumbent`
  - checkpoint 固定：
    - `/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
  - decode 固定：
    - `beam=4 / lp=0.7 / max_new_tokens=384`

所有新线都必须尽量构造成三臂：

- `I0`: incumbent
- `C0`: matched control
  - 与 treatment 完全同 recipe / 同 budget / 同 decode
  - 唯一差别是不加本轮新因子
- `T0`: treatment

统一要求：

- 验证集绝不能变化
- `fold0` 的 val parent 集必须与 incumbent 完全一致
- `fold0` 的 val chunk 行数必须与 incumbent 完全一致
- 任何训练过滤、混训、增广、retrieval datastore 扩展都只能作用于 train-visible 数据
- 任何新线如果做不到 matched control：
  - 不得直接拿去和 `I0` 宣称输赢

### 1.2 外部数据纪律

从这一版开始，外部平行语料是第一主线，但必须严格满足：

- 合规
- 可审计
- train-only
- 不污染 val / test

强制要求：

- 所有外部语料必须建立 `source_registry.csv`
- 每条语料都必须带：
  - `source_origin`
  - `license_note`
  - `raw_parent_rows`
  - `post_clean_rows`
  - `post_chunk_rows`
  - `post_shortalign_rows`
- 必须输出 `manifest.json`
- 必须输出 overlap audit：
  - exact source overlap with `fold0 val`
  - exact source overlap with `test`
  - normalized source overlap with `fold0 val`
  - normalized source overlap with `test`
  - exact pair duplicate rows

明确禁止：

- 把外部平行数据一股脑全灌进去
- 不做来源分层、不做比例 sweep 就直接 promote
- 让某个单一外部来源在首轮混训里占外部部分的绝对多数
- 用外部 dev/benchmark 结果倒灌本地 gate

首轮固定 sweep：

- 外部增量比例：
  - `10%`
  - `30%`
  - `50%`

这里的比例定义为：

- 相对于当前 internal processed train rows 的增量比例
- 不是相对于原始 parent 行数

### 1.3 retrieval / datastore 纪律

retrieval 是第二主线，但必须是**可部署、无泄漏、chunk-level** 的实现。

统一要求：

- datastore 只能来自 train-visible parallel
- offline 验证时：
  - `fold0 val` 不得出现在 datastore 中
- test 端：
  - 不得把任何 test 自身 target 或未来不可见字段写入 datastore
- datastore 每条记录必须保留：
  - `source`
  - `target`
  - `source_origin`
  - `parent_oare_id`
  - `chunk_index`
  - `chunk_total`
- retrieval 只能在 chunk 级工作
- 不允许把 retrieval 重新包装成 parent rewrite

明确禁止：

- 用 val/test target 做近邻建库
- 用 reference 长度、reference bucket、gold hard/easy 标签做 retrieval 路由
- 用 retrieval 结果做不可解释的大范围重写

### 1.4 candidate / MBR / ensemble 纪律

从这一版开始，`MBR` 和 ensemble 只接受“真实多样性”的候选池。

候选多样性的合法来源包括：

- internal-only winner
- external-mix winner
- retrieval-on / retrieval-off 版本
- 不同 fold
- 不同 seed
- 不同 source-mix ratio
- 只有在 `A4` 放行后，才允许加入新的 backbone family

明确禁止：

- 只拿同一个模型的几条相似 beam 假装是多样候选池
- 只拿相邻 checkpoint 做“伪 ensemble”
- 没做相关性 / error overlap 审计就宣称 ensemble 有价值

统一要求：

- 先做 candidate diversity audit
- 再做 `MBR`
- 再做 ensemble / rerank

### 1.5 评分纪律

所有主线统一输出 3 套分数：

- `local`
  - full-val reconstructed 为主
- `official-like`
  - 在官方 bridge 就绪前，保持最接近官方的本地代理
- `hard`
  - 至少包含：
    - routed hard full
    - 长 parent / 多 chunk / marker-rich 子集

所有主线最终归档统一输出：

- `anchor64`
- `full-val`
- `hard subset`
- 输出健康度：
  - `empty`
  - `copy`
  - `short`
  - `repeat`
  - `unique`

外部混训线还必须额外输出：

- `source_origin` slice
- `internal-only slice`
- `external-heavy slice`

当前规则：

- 不允许只报 tune 小口径
- 不允许只报 local 不报 hard
- `promote / reject` 不能只由单一分数决定

但探索期值卡纪律改成：

- `P/probe` 阶段的强制产物只有：
  - `anchor64`
  - `hard subset`
  - 输出健康度
- `full-val` 不再作为每条新线的默认后续动作
- 未跑 `full-val` 的长训练，只能记为：
  - `candidate_pool_long_train`
  - 不能记为 `F/promote`
- 同一轮探索里，默认只允许**最强且最正交的一条候选**消耗 `full-val` 预算
- 其他候选只允许：
  - `reject_stop`
  - `review_stop`
  - `candidate_pool_long_train`

### 1.6 官方 metric bridge 纪律

这一版不再允许无限期拖延官方 bridge。

执行顺序：

- `P` 段可以先用 `official-like`
- 但任何进入 `F` 段的主线，都必须补：
  - official bridge
  - 或明确写出“为什么本轮仍只能 official-like”

若 bridge 接入成本可控：

- 直接纳入所有 `W/F` gate

若 bridge 仍未就绪：

- 必须输出 `official_metric_probe.json`
- 且不能把 `official-like` 的结果写成“等同官方”

### 1.7 算力最大化 + 并行 tmux 纪律

从这一版开始，默认按“单卡最大利用率 + 多 tmux 后台并行”执行。

统一仍按：

- `P -> W -> F`

GPU 纪律：

- GPU 重训练任务串行
- 同一时刻只允许一个主训练占卡
- decode 若不抢卡，可与 CPU 审计并行
- 若时间/精力不足：
  - 优先保 `probe`
  - 砍掉大多数 `full-val`
  - 不允许因为想省时间，就把 `candidate_pool_long_train` 误写成 `F/promote`

CPU / IO 纪律：

- overlap audit
- registry 构建
- retrieval datastore 构建
- candidate cache 汇总
- report / manifest

必须放入独立 tmux session 并行跑。

默认 session 命名：

- `winner_data`
- `winner_train`
- `winner_decode`
- `winner_retrieval`
- `winner_mbr`
- `winner_report`

统一要求：

- 每个 session 都要落日志
- 每条主线都要落：
  - `summary.json`
  - `gate_report.md`
  - `manifest.json`

### 1.8 暂停项

以下方向先归档，不再作为当前提分主线：

- `dan-1 / dan-2` parent-level rewrite / fusion
- 当前 `A2` alignment/noise filter
- 当前 fair 结果为负的 `TAPT -> supervised`
- `L2` 词表 patch 主线化
- `L3` row-gate / source-policy 主线化
- 在旧 recipe 上重开 `ByT5-base / mT5-base`
- 把 `no_repeat_ngram_size` 当默认修复
- 任何 target-aware routing

## 2. 主线一：外部平行数据 / 域扩展混训（最高优先级）

### 2.1 目标

这是当前最值得狠狠干的主线。

目标不是：

- 换更大的模型
- 重新定义任务

而是：

- 在保留 winner 成功结构的前提下，直接扩大并增强监督

本线固定保留：

- `ByT5-small chunk`
- 当前 chunk formulation
- 当前 `short_aligned` 监督结构
- 当前 decode gate

本线唯一优先改变的是：

- train 侧可见监督规模
- train 侧监督来源多样性

### 2.2 固定 recipe

第一阶段固定不改：

- base family:
  - `ByT5-small`
- LoRA:
  - `q/v`
  - `r=8`
  - `alpha=16`
- current chunk pipeline
- current short aligned augmentation
- current decode:
  - `beam=4 / lp=0.7 / max_new_tokens=384`

也就是说，首轮外部混训要回答的是：

- **只扩监督，不换架构，能不能提分**

而不是把变量一次性搅在一起。

### 2.3 数据构造纪律

外部平行语料进入主线前，必须经过同一套处理：

1. preprocess
2. overlap / dedup audit
3. chunking
4. 必要时 short aligned augmentation
5. source-origin tagging

首轮强制要求：

- `exact source overlap with fold0 val = 0`
- `exact source overlap with test = 0`
- `normalized source overlap with fold0 val = 0`
- `normalized source overlap with test = 0`
- `exact pair duplicates removed`
- `source_origin` 明确标注

首轮外部短对齐纪律：

- 外部 parallel 若要启用 `short_aligned`
  - 必须走与 internal 一致的 pipeline
  - 但阈值更严
- 若外部 `short_aligned` 在 smoke 中造成明显重复或短句恶化：
  - 立即回退为“只保留 external chunk rows”

首轮比例 sweep 固定为：

- `E10`
- `E30`
- `E50`

每个比例都必须有 matched control：

- `C0 = internal-only matched control`
- `E10 = internal + external 10%`
- `E30 = internal + external 30%`
- `E50 = internal + external 50%`

### 2.4 P 纪律

#### `A1_P0`: 资产盘点与合法性审计

- 建 `source_registry.csv`
- 建 `manifest.json`
- 建 `overlap_audit.json`
- 建 `mix_plan.csv`

必须记录：

- 每个来源 raw rows
- 清洗后 rows
- chunk 后 rows
- short aligned 后 rows
- 被剔除的 overlap rows
- 被剔除的 duplicate rows

当前结果回填：

- 已完成并落盘：
  - `source_registry.csv`
  - `manifest.json`
  - `overlap_audit.json`
  - `mix_plan.csv`
  - `dedup_manifest.json`
- 当前 internal 基线规模：
  - `train_visible_rows = 4064`
  - `val_visible_rows = 1225`
  - `val_parent_rows = 313`
- 当前状态：
  - `ready_for_mix_build`
- 纪律解释：
  - `A1` 现阶段已从资产阻塞解锁，可以直接进入 `A1_P1`

#### `A1_P1r`: continue-ready mixed processed_dir 构造

旧的 `A1_P1/P2` plain-base smoke 已归档，不再作为主线。

当前只允许构造 4 套 `continue-on-wlite` 数据：

- `internal_only_matched`
- `external_mix_e5`
- `external_mix_e10`
- `external_mix_e15`

要求：

- 第一步必须先在 plain processed rows 上混入 external：
  - 基底固定为 `data/processed_byt5_chunks_align_gc_cost14`
- 第二步必须对每一套 plain mixed rows 重新生成 retrieval-top1 hint：
  - 不允许在已有 retrieval source 上二次叠 hint
- `fold0 val` 的 raw row 集合完全不变
- 训练改动只作用于 `fold != 0`
- internal rows 全保留
- 外部增量只按 sweep 比例加入
- 训练 config 必须克隆当前 retrieval winner recipe：
  - `reports/taskform_winner_a2_retrieval_wlite_20260310/generated_configs/taskform_winner_a2_retrieval_top1_wlite.yaml`
- 所有候选都必须从同一个 upstream adapter 初始化：
  - `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model`
- 不允许把 `fallback_180` 当训练起点：
  - 它是 post-hoc compare candidate，不是 checkpoint

#### `A1_P2r`: matched continue probe

每套都跑同 budget smoke：

- `180 steps`
- 同 seed
- 同 batch size
- 同 lr
- 同 eval_steps
- 同 decode

只允许一个变量不同：

- 训练数据来源
- upstream train data mixture

probe 输出只认：

- `anchor64 reconstructed`
- `hard subset`
- `health delta vs C0`

绝不再拿：

- fresh-base 绝对分
- plain non-retrieval smoke
- `E30 / E50` 这种已被坏先验提示过的高比例

#### `A1_P3`: source-tag ablation

当前不自动开启。

只有当 `A1_P2r` 的最佳比例已经明确正于 `C0`，才允许在最佳比例上补一个最小 `tag_on/tag_off` 对照。

### 2.5 gate

`A1` 首轮 gate 改成 winner-continue 口径：

- `Ebest - C0` 在 `anchor64 geom >= +0.25`
- `hard subset` 不得为负
- `short / empty / repeat / unique` 不恶化
- 与当前 frozen candidate 的 `anchor64` 差距不能继续恶化

若三组比例全部不满足：

- `reject_stop`

若最佳比例只有弱正：

- `0 < delta_anchor < +0.25`
- 但 health 全绿

则：

- `review_stop`

若最佳比例过线：

- `review_to_wlite`

### 2.6 `W-lite` 纪律

`W-lite` 不再是“从零再训一版”，而是：

- `C0_W = internal-only continue @ 400 steps`
- `Ebest_W = best external continue @ 400 steps`

都从同一个 upstream adapter 出发：

- `TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/best_model`

首轮只跑：

- `400 steps`
- `anchor64`
- `hard subset`

只有当 `Ebest_W` 在 400 steps 下仍明显强于 `C0_W` 时，才允许扩到：

- `600 steps`
- 或 `ckpt averaging`

`W-lite` 放行条件：

- 相对 `C0` full-val `geom >= +0.25`
- `hard subset` 不掉
- `health no_regression vs C0 = true`

若仍远落后 `I0`：

- `review_stop`

### 2.7 `F / promote` 判定长实验

只有当 `Ebest_W` 已经对 `C0_W` 明显成立时，才进入长实验。

`F` 段的 compare 参照不再只有 incumbent，而是三路：

- `I0 incumbent`
- `A2 frozen candidate = fallback_180`
- `A1 best long candidate`

长实验固定产出：

- full-val reconstructed
- official-like
- hard subset
- source-origin slices

若 raw `A1 best` health 仍发红：

- 必须补一轮 `A2` 同类的 health surgical compare
- 然后再拿 repaired candidate 进 promote compare

`A1` 进入 promote 的门槛改成：

- `full-val reconstructed geom >= fallback_180 + 0.15`
- `hard geom >= fallback_180 - 0.10`
- `health no_regression vs incumbent = true`

若没超过 `fallback_180`，但有明显互补错误模式：

- 可降级转入 `A3` 的 candidate pool
- 但不得直接 promote 为新 winner

## 3. 主线二：retrieval / kNN-MT / 记忆增强式 chunk 建模

### 3.1 目标

这条线的核心不是重写模型，而是给每个 chunk 更像的训练记忆。

目标：

- 保留 winner 的 chunk 级稳定性
- 提升专名、术语、公式化搭配和低频结构的 recall
- 避开 parent rewrite 已知失败模式

这条线必须显式回答的问题是：

- retrieval 能不能在不破坏输出可控性的情况下，抬高 chunk 级候选质量

### 3.2 datastore 纪律

首轮 datastore 固定只用：

- internal parallel train-visible rows

只有当 `A1` 外部混训已经给出正信号后，才允许加第二套 datastore：

- `internal + approved external`

datastore 记录粒度：

- chunk-level source
- chunk-level target
- `source_origin`
- `parent_oare_id`
- `chunk_index`
- `chunk_total`

必须输出：

- `datastore_manifest.json`
- `retrieval_overlap_audit.json`
- `retrieval_bucket_audit.csv`

### 3.3 P 纪律

#### `A2_P0`: 检索基线审计

先做不训练的 retrieval audit：

- `BM25`
- char-ngram similarity
- token overlap

目标：

- 看 top-k 近邻是不是语义/术语上真的有帮助

#### `A2_P1`: datastore smoke

构建两套候选：

- `R0 = no retrieval control`
- `R1 = retrieval hint top1`
- `R3 = retrieval hint top3 compressed`

如果 infra 足够稳定，再追加：

- `RK = kNN-MT logits interpolation`

当前结果回填：

- `R0` matched control anchor64：
  - `geom = 15.8741`
- `R1` smoke anchor64：
  - `geom = 22.2513`
- `R1` smoke hard：
  - `geom = 22.9848`
- `R3` 当前 formulation 不放行：
  - `5017 / 5289` 行超过 `max_source_length = 640`
  - 需要改成更短、去重、按预算裁切的 hint 格式
- `RK` proxy probe 不放行：
  - infra-positive
  - 但 best proxy reconstructed `22.4518`，低于 baseline `23.5422`

#### `A2_P2`: matched supervised smoke

所有 retrieval 版本必须同 budget：

- `180-250 steps`
- 同 seed
- 同 batch size
- 同 lr
- 同 decode

只允许一个变量不同：

- retrieval on/off / top-k / interpolation

#### `A2_P3`: targeted bucket audit

必须额外单报：

- rare name bucket
- measure bucket
- formula bucket
- marker-rich bucket

目的：

- 判断 retrieval 是真的补 recall，还是只是碰巧改写了几句

### 3.4 gate

首轮放行门槛：

- 相对 `R0`：
  - `anchor64 geom >= +0.25`
- `hard subset` 必须非负
- rare/measure/formula 至少一类 targeted bucket 明显转正
- 推理延迟不得失控

当前默认 latency gate：

- 平均 decode+retrieval latency 不得超过 `R0` 的 `1.8x`

若 retrieval 只有 local 小正、hard 为负：

- `review_stop`

若 retrieval 明显改善 hard 或 targeted recall：

- 进入 `W`

### 3.5 W 纪律

`W` 段只保留最佳 retrieval setting：

- `top1 vs top3`
- `hint vs kNN interpolation`
- `internal-only datastore vs mixed datastore`

统一要求：

- full-val local / hard 同时评估
- 输出 retrieval cache hit 统计
- 输出 nearest-neighbor quality 样本审计

若 retrieval 最终仍未超过 `I0`，但与 `I0` 错误分布互补：

- 降级进入 `A3` candidate pool

当前结果回填：

- 当前 best retrieval setting 已经明确是：
  - `retrieval hint top1`
- `W-lite` 已跑完：
  - `anchor64 geom = 23.5422`
  - `full-val reconstructed geom = 19.9956`
  - `hard geom = 20.8432`
- 当前支撑件已齐：
  - retrieval cache hit
  - nearest-neighbor quality audit
  - latency report
  - memory usage report
  - official-like template
- 当前 health surgical 已完成：
  - 推荐候选：
    - `fallback_180`
  - `full-val reconstructed geom = 19.9035`
  - `hard geom = 20.7888`
  - 相对 raw W-lite：
    - `full-val -0.0872`
  - health：
    - `no_regression vs I0 = true`
  - changed rows：
    - `58 / 1225`
    - 其中 original 仅 `8` 行
- 当前状态：
  - `candidate_frozen_manual_promote_recommended`
- 原因：
  - raw `W-lite` 主分最高，但 health 为红
  - `fallback_180` 以很小代价换回 health gate 过线
  - 因此当前 `A2_F_review` 已以 `fallback_180` 完成 promote freeze，而不是直接 promote raw `W-lite`

### 3.6 F 纪律

只有当 retrieval line 本身已可与 `I0` 正面对比时才进 `F`：

- full-val reconstructed
- official-like
- hard subset
- latency report
- memory usage report

## 4. 主线三：多样候选池 + efficient MBR + 真正有互补性的 OOF / ensemble

### 4.1 目标

如果单模继续抠上限不够，这条线就是撬开单兵上限的主线。

这条线的重点不是：

- 再做一个输出 patch

而是：

- 造真正多样化的候选池
- 用更强选择器选候选
- 用真正互补的模型做 ensemble

### 4.2 候选多样性纪律

候选池的合法来源，按优先级如下：

1. `I0` incumbent
2. `A1` 外部混训 best
3. `A2` retrieval best
4. 同 recipe 不同 seed / fold
5. 同 recipe 但不同 mix ratio
6. 只有 `A4` 放行后，才允许新 backbone

明确禁止：

- 只拿同一模型几条高度相似 beam 当多样候选池
- 只拿相邻 checkpoint 当 ensemble 主体

统一要求：

- 先跑 candidate diversity audit
- 再跑 `MBR`
- 再跑 ensemble / rerank

### 4.3 P 纪律

#### `A3_P0`: diversity audit

必须先统计：

- unique candidate ratio
- self-BLEU / self-chrF
- exact match overlap
- error overlap

当前结果回填：

- `A3_P0` 已完成，且已过本线前置 gate：
  - pool `unique_candidate_ratio_pct = 92.1569`
  - `rows_all_unique_ratio_pct = 76.4706`
  - `incumbent vs retrieval_smoke exact overlap = 0%`
  - `incumbent vs retrieval_wlite exact overlap = 0%`
- `A3_P1/P2` 已完成：
  - best single：
    - `retrieval_wlite_repaired`
    - `anchor64 geom = 24.6330`
  - `MBR geom = 23.6760`
  - `delta = -0.9570`
  - 当前状态：
    - `review_stop`
- 当前结论：
  - incumbent 与 retrieval 系列候选存在真实互补
  - 但当前 `MBR` formulation 没有把互补性转成更高分
  - `A3_P3` 当前不放行
  - 只有在出现新的强候选轴后，才允许重开 `A3`

#### `A3_P1`: candidate generation

候选来源至少覆盖两个不同轴：

- model axis
- data axis
- retrieval axis
- fold/seed axis

beam 只作为补充，不是主多样性来源。

#### `A3_P2`: efficient MBR

首轮 utility 固定只用非 LLM：

- `BLEU`
- `chrF++`
- 及其组合

必须对比：

- beam winner
- best single candidate
- `MBR` winner

#### `A3_P3`: pairwise ensemble probe

只允许先做小规模 pairwise：

- `I0 + A1_best`
- `I0 + A2_best`
- `A1_best + A2_best`

### 4.4 gate

`A3` 的候选池必须先满足：

- `unique candidate ratio >= 85%`
- 候选间不是高度同质

`MBR` / ensemble 放行门槛：

- 相对 best single model：
  - `anchor64 geom >= +0.20`
- `hard subset >= +0.15`
- `short / empty / repeat` 不恶化

若 pairwise ensemble 对 best single 只给出 `< +0.15`：

- `review_stop`

若 `MBR` 提升只出现在 local，不出现在 hard：

- `review_stop`

### 4.5 W 纪律

只有当 pairwise 确认真互补时，才扩到：

- `3-model pool`
- `OOF weight search`
- `ckpt averaging`
- `MBR + ensemble` 组合

统一要求：

- 输出 `ensemble_search.json`
- 输出 `correlation_audit.csv`
- 输出 `candidate_pool_manifest.json`

### 4.6 F 纪律

进入 `F` 前必须固定：

- 模型名单
- 权重
- decode 参数
- cache 目录
- fallback 策略

最终提交线只接受：

- 可复现
- 互补性有证据
- 明确优于 best single model

## 5. 主线四：更强 recipe 上的二次架构探测（条件放行）

### 5.1 立场

这条线不是当前主攻，只是条件放行项。

当前纪律明确写死：

- 在旧 recipe 上重开 `ByT5-base / mT5-base`
  - 不允许

因为现有证据已经表明：

- 在当前 recipe 下，它们不是“小负”
- 而是直接硬失败

### 5.2 只有在以下条件满足时，才允许重开 backbone

至少满足其一：

- `A1` 外部混训已稳定转正
- `A2` retrieval line 已稳定转正
- `A3` 已证明现有候选池仍受 backbone 同质性限制

并且同时满足：

- official bridge 已接入
- current decode gate 已稳定
- 新 backbone 的对照预算可 matched

### 5.3 放行后只允许的做法

首轮只允许：

- matched smoke
- anchor64
- 同 budget
- 同 decode

若 `100-150 steps` 就明显落后 matched control：

- `hard_stop`

也就是说：

- backbone 只能在更强监督、更强候选、更干净评估都补齐后再开
- 不允许再拿它当前置替代方案

## 6. 执行顺序（算力最大化版）

### 6.1 当前总顺序

在当前已执行结果基础上，下一阶段执行顺序更新为：

1. `A1` 当前归档
   - `continue-on-wlite` probe 已判负
   - 不再继续 `E5 / E10 / E15` 同口径扩展
2. `A2g` 当前归档
   - `g35` 已明显负于 retrieval always-on control
   - 不再继续 `g50 / g65` 或同质 selective gate 扩展
3. `competition-only mono` 当前归档
   - `pseudo-target synthetic mix` 判负
   - `denoising-only fair compare` 也判负
   - 不再重复消耗 GPU 跑同口径
4. `A2 retrieval`
   - `fallback_180` 继续保留为当前工作基线 / promote compare 基线
   - `raw retrieval longtrain` 保留为当前 raw score ceiling
   - 下一轮若重开 retrieval，只允许围绕 health repair / selector / surgical 设计
5. `replay` 轴
   - `replay25` 与 `replay15` 只保留为 candidate-pool complementary signal
   - 不再把 replay 当新的单模型主线
6. `A3 revisit`
   - `cheap revisit` 已证明 retrieval 与 replay 之间存在真实互补
   - 但当前 selector / MBR formulation 仍未成立
7. `official bridge / report glue`
   - 继续作为 CPU 侧补件
   - 不阻塞下一轮核心设计
8. 当前进入：
   - 下云端算力
   - 离线分析 / 调研
   - 明确下一轮只回云端做新的、正交的迭代
9. `RK_true_hook`
   - 继续 park
   - 只保留代码接线与 error analysis 产物
10. `A4`
   - 继续锁住，不前置

解释：

- `A1`、`A2g`、`competition-only mono` 都已经给出足够明确的负证据
- `A2 retrieval` 是当前唯一真正成立的主胜轴
- `raw retrieval longtrain` 说明 retrieval 还有 ceiling，但 health / hard 还没被解决
- `replay` 说明存在互补性，但还不足以独立赢 single-model compare
- `A3 cheap revisit` 把问题进一步收紧为：
  - 不是“有没有互补候选”
  - 而是“能不能设计出真正吃到互补的 selector”

### 6.2 本轮后台队列回填（已结束）

本轮已完成的后台队列如下：

- `winner_train_main_20260311`
  - 已完成
  - 结果：
    - `raw retrieval W-lite long train`
    - `replay25 candidate-pool long train`
- `winner_postmain_probe_20260311`
  - 已完成
  - 结果：
    - `combo probe`
    - `replay15 / replay20 / replay30` 窄扫
- `winner_a3_revisit_20260311`
  - 已完成
  - 结果：
    - `A3 cheap revisit`
- `winner_postprobe_decode_20260311`
  - 已完成
  - 结果：
    - `replay15 post-probe full-val`
- `winner_smallprep_20260311`
  - 已完成
  - 结果：
    - 相关 processed dir / config 预构建

本轮后台执行后的最终事实是：

- 当前无活跃 GPU 训练 / decode 会话
- 当前不再有“今晚还在自动排队”的任务
- 当前阶段已经从：
  - `云端执行`
  切换到：
  - `结果分析`
  - `下一轮设计`

### 6.3 自动分叉纪律

若 `A1` 先给出明显正信号：

- `A2` 可加第二套 mixed datastore
- `A3` 候选池必须纳入 `A1_best`

若 `A1` 仍处于 blocked：

- `A1` 不占主训练预算
- `A2/A3` 继续按 internal-only 证据推进

若 `A1` 再次出现 `ready_for_mix_build` 这一旧条件：

- 不再自动回到 GPU 主序第一位
- 必须先有新的非同质设计，再允许重开 `A1`

若 `A2` 已转正但 `A2g` 为负：

- retrieval always-on 继续作为工作基线
- 不再自动开启 retrieval 同质 gate 训练
- retrieval 只在出现真正不同的机制变量时才允许重开

若 `A2` 已有 health-safe compare 候选：

- `A2_F_review` 立即收口为 freeze bundle
- 当前 promote candidate 固定用 `fallback_180`
- raw `W-lite` 只保留为 score ceiling reference
- 在新候选轴出现前，不再自动开启新的 `A2` GPU 训练

若 `raw retrieval longtrain` 已出现“full-val 高于 frozen，但 hard / health 未同时过线”的情况：

- 不得直接写成：
  - `F/promote`
- 只允许写成：
  - `research ceiling`
  - `candidate_pool / compare reference`
- 下一轮若重开 retrieval：
  - 只能围绕 health repair / selector / surgical compare
  - 不再继续同口径裸长训

若 `A2g` 在 `g35` 已明显负于 `ctrl`：

- `g50 / g65` 不补跑
- `A2g` 立即归档
- 下一条主线直接切到正交候选轴

若 `competition-only pseudo-target / denoising continue smoke` 已实际 `probe` 判负：

- 立即停线
- 不补 `full-val`
- 当前 pseudo-target synthetic line 归档
- GPU 主序切到：
  - `competition-only denoising-only continue smoke`

若 `competition-only denoising-only continue smoke` 已有公平对照负证据：

- `competition-only mono` 轴整体归档
- 不再继续原始 pseudo-target / synthetic curriculum 扩展
- GPU 主序切到：
  - `winner replay / curriculum probe`

若 `winner replay / curriculum probe` 的 best arm 为弱正：

- 可以跳过当轮 `full-val`
- 直接转入：
  - `candidate_pool_long_train`
- 但不得宣称进入 `F/promote`

若 `winner replay / curriculum probe` 的 best arm 为强正：

- 先看它是否是当轮**最强且最正交**候选
- 若是：
  - 才允许消耗 `full-val` 预算
- 若不是：
  - 仍只进入 `candidate_pool_long_train`

若 `winner replay / curriculum probe` 已出现“对 matched control 强正、但仍低于 incumbent / frozen”的情况：

- 允许采用激进模式：
  - 跳过当轮 `W-lite`
  - 直接启动一条 `candidate_pool_long_train`
- 这条长训练的目的只允许写成：
  - 真实摸清该轴在 `full-val / hard / raw / reconstructed` 上的最终落点
- 不允许写成：
  - `F/promote`
- 同时允许并行开启少量新的正交 `P/probe`
- 但这些小实验必须满足：
  - 比当前长训练更便宜
  - 不与 `A2g` / `competition-only mono` / `A1` 旧 external mix 同质
  - 不抢占唯一需要保留的长训练 GPU 时段

若 `replay25 candidate-pool long train` 已完成，且 `full-val` 只逼近 incumbent、但仍明显低于 frozen retrieval：

- replay 轴不得再被写成新的单模型主线
- replay 只保留为：
  - `candidate_pool complementary axis`
  - `regularizer / selector signal source`
- 后续若重开 replay：
  - 必须依附于 retrieval 主线或 selector 设计
  - 不再单独扩成长训练主序

若当前已批准执行 `winner_train_main_20260311 -> winner_postmain_probe_20260311 -> winner_postprobe_decode_20260311`：

- 不得打断主序当前的 `raw retrieval -> replay25` 长训练
- 小实验只允许排在主序后面
- `winner_smallprep_20260311` 只做 CPU / IO 预构建
- `winner_postprobe_decode_20260311` 必须满足：
  - `top_k <= 2`
  - 只吃 probe winner
  - 必须先过：
    - `anchor_delta_vs_control > 0`
    - `hard_delta_vs_control >= 0`
    - `health no_regression = true`
- `A3 cheap revisit` 不得冒充 `MBR promote`
- `RK_true_hook` 这次只允许：
  - `weak smoke revisit`
  - 不得扩到 `anchor64 / W-lite / full-val`

若 `winner replay / curriculum probe` 的 best arm 为负：

- 当前 replay / curriculum 口径归档
- GPU 主序再切到下一条真正不同的结构轴

若 `winner replay15 post-probe full-val` 已健康过线，但 `full-val <= incumbent`：

- `replay15` 只允许保留在候选池中
- 不得再往 `single-model promote` 推进
- 这类结果应优先送入：
  - `A3 cheap revisit`
  - `selector / pairwise chooser` 设计

若 `A3_P1/P2` 为负：

- `A3_P3` 禁止继续
- `A3` 只有在出现新候选轴后才允许 revisit

若 `A3 cheap revisit` 已显示：

- `exact overlap ~= 0`
- `oracle delta vs best single > 0`
- 但 `heuristic delta ~= 0`

则下一轮 `A3` 只允许重开：

- 更强的 row-level selector / chooser
- 与 health surgical 兼容的 pairwise repair

不允许重开：

- 当前同 formulation 的 `MBR`
- 没有新 selector 的纯 decode-level 投票

若 `RK` 只有 infra-positive，但 reconstructed 指标为负：

- `RK` 立即 park
- 不得跳过 `A2_F_review` 或其 freeze 结果抢占 GPU 主序

若 `RK_true_hook` 已接入 formal `decode / diagnose`，但 smoke 仍为负：

- `RK_true_hook` 立即 park
- 只允许保留代码接线与 error analysis 产物
- 不得继续扩到 `anchor64 / W-lite / full-val`

## 7. 放行门槛总表

### 7.1 `A1` 外部混训

- 旧 plain-base smoke 一律不计
- 只认 `continue-on-wlite` 口径
- `anchor64 geom >= C0 + 0.25`
- `hard subset` 非负
- `short / empty / repeat` 不恶化

### 7.2 `A2` retrieval

- `anchor64 geom >= R0 + 0.25`
- targeted rare/measure/formula 至少一类明显转正
- latency `<= 1.8x`

### 7.3 `A3` MBR / ensemble

- `anchor64 geom >= best_single + 0.20`
- `hard subset >= +0.15`
- pairwise ensemble gain 不得小于 `+0.15`

### 7.4 `A4` backbone

- 仅条件放行
- `100-150 steps` 若落后 matched control 明显：
  - `hard_stop`

## 8. 这版纪律的核心判断

从这版开始，winner 提分不再赌：

- 更大的模型会不会自己突然变强
- 后处理会不会侥幸把分补回来
- 再做一个 parent rewrite 会不会刚好学对

而是明确赌三件更像“真能抬上限”的事：

1. retrieval 主线上的 health-constrained gain
2. retrieval 与 replay 之间真正可部署的候选选择器
3. 只在新机制变量成立后，才引入新的训练信号

如果这三条仍然都不给信号，再谈“当前 winner family 是否已到天花板”才有意义。  
在那之前，继续烧旧坑，不叫激进，只叫重复犯错。
