# ORACC Authority Index for Akkadian — Deep Past MT (MVP)
# ORACC Authority Index for Akkadian (Deep Past MT) — MVP v0.1

## 1. What this index is for
- 本索引是 `semantic_cleaning_spec.md` 的“权威来源目录（authority index）”，用于支撑清洗宪法的每一条“必须/允许/禁止”规则，而不是语言学综述。
- 用途一：为 **Protected Markers（semantic markers）** 建立可审计依据：哪些符号/结构是语义标记必须保留（例如 ATF 控制行、方括号/花括号、行号、方言 `%xx`、`#lem:` 结构等）。
- 用途二：为 **Tier 边界（Tier-0/1/2）** 提供“哪里写过/怎么写的”证据，确保 Tier-0 只做安全规范化（safe normalization），不改变语义标记是否存在。
- 用途三：指导 **冻结证据（evidence freeze）**：对 P0 页面做截图/PDF/MD 快照（snapshot），将证据固定进仓库，避免后续“漫游式采集”和页面变更带来的不可复现。
- 用途四：为清洗规则实现提供“rule hooks”：每条规则绑定一个或多个权威页面，方便 ablation、edit_log 审计与回滚。
- 重要原则：**不臆造页面内容**；只基于你给的 PDF/MD 中出现的链接与其附近上下文做最小推断；不确定一律标注“需二次核验”。

## 2. Extracted links (partial, deduped)
> 说明：以下为从 PDF+MD 能稳定抽取到的链接（去重后）。Akkadian roots 在 PDF 中出现大量 `#anchor`（#alef/#b/...），此处仅保留代表性条目，避免刷屏。

- Akkadian stylesheet — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/akkadianstylesheet/index.html — PDF/MD 多处提到：建议遵循 Akkadian transliteration 样式以复用成熟词汇表（glossary）与格式约定。
- Akkadian roots (base page) — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/akkadianroots/index.html — MD 提到：roots 需在项目 Akkadian glossary（`akk.glo`）中分配；PDF 还列出按字母分区 anchors。
- Akkadian roots (example anchor: #alef) — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/akkadianroots/index.html#alef — PDF roots 列表分区锚点之一（其余锚点同类）。
- Projects and Emacs (glossary annotation) — https://oracc.museum.upenn.edu/doc/help/..index.html#h_addinglinguisticannotationstotheglossary — PDF/MD 提到：在 glossary 中添加语言学标注（linguistic annotations）的入口（链接看起来像“..index.html”，需二次核验是否重定向）。
- Akkadian linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/index.html — MD/primer 中作为 Akkadian 专用标注规范入口（含 dialect/normalization/citation form 相关提示）。
- Akkadian citation forms (CF) section — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/index.html#h_citationformscf — PDF 提到：CACF/CF 规则与 normalization 中 mimation 等处理（高风险，需二次核验具体条款）。
- ATF Primer — https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/index.html — PDF/MD 指向：ATF 编辑入门（对控制行/结构符号的保留极关键）。
- ATF Inline Tutorial: Languages section — https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/inlinetutorial/index.html#h_languages — MD 指向：语言/方言代码（dialect codes），含行内 `%xx` 示例。
- Lemmatisation primer (Oracc linguistic annotation intro) — https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html — MD/QPN 文中假定读者已知：#lem 行结构、分隔符、歧义与一致性原则等。
- Lemmatisation primer: proper nouns section — https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html#h_propernouns — QPN 文中引用：解释 explicit vs POS-only lemmatization（与 `#lem:` 结构强相关）。
- Nammu and Emacs (index) — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/index.html — PDF 提到：用 Emacs/atf-mode 编辑与 lemmatise，强调一致性（对“可审计/可回滚”流程有价值）。
- Preparing to work with Emacs — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/emacssetup/index.html — PDF 提到：安装/配置 Emacs 以开展 Oracc 工作。
- Aquamacs (Emacs for Mac) — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/aquamacs/index.html — PDF 提到：Mac 下编辑/lemmatise ATF 的工具说明。
- EmacsW32 (Emacs for Windows) — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/emacsw32/index.html — PDF 提到：Windows 下编辑/lemmatise ATF 的工具说明。
- Working with ATF in Emacs using atf-mode — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/emacsforatf/index.html — PDF 提到：atf-mode 提供模板生成与检查（对结构约束与错误检测相关）。
- Managing projects with Emacs (index) — https://oracc.museum.upenn.edu/doc/help/..index.html — PDF 提到：与项目管理相关（链接形态不完整，需二次核验具体落点）。
- Project Management with Emacs (harvesting lemmatization data) — https://oracc.museum.upenn.edu/doc/help/..index.html#h_harvestingnelemmatisation%20data — primer 中提到：Harvest Notices/复核长格式 lemmatization（含 `+`）等流程。
- Building a Portal Website — https://oracc.museum.upenn.edu/doc/help/portals/index.html — PDF “See also” 提到：与项目门户构建相关（对清洗宪法不是核心，但可保留）。
- Proper nouns linguistic annotation page (QPN) — https://oracc.museum.upenn.edu/doc/help/languages/propernouns/index.html — MD/QPN 文本本体：专名转写与标注约定（含 `{m}/{f}`、连字符、双连字符等提示）。
- Aramaic linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/aramaic/index.html — primer “Next steps” 列表中的语言页（旁证：不同语言页可能对标记/结构有差异）。
- Elamite linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/elamite/index.html — 同上（语言差异旁证）。
- Greek linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/greek/index.html — 同上（语言差异旁证）。
- Old Persian linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/oldpersian/index.html — 同上（语言差异旁证）。
- Sumerian linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/sumerian/index.html — MD 提到：苏美语形态标注细节更多（旁证：某些符号/结构在别语种更复杂）。
- Ugaritic linguistic annotation page — https://oracc.museum.upenn.edu/doc/help/languages/ugaritic/index.html — 同上（语言差异旁证）。

- [TODO: missing links due to rendering] PDF/MD 可能含更多 ATF quick reference / CDLI ATF 等链接，但当前抽取未覆盖；后续补齐。
- 备注：若链接出现 `..index.html` / 锚点疑似失效或重定向风险，冻结时优先抓“页面标题 + URL + 示例段”以保证可追溯。

## 3. Priority shortlist (P0 candidates)
> P0 候选：最可能直接决定“语义标记必须保留”与 Tier-0 边界的页面（8–12 条）。用途不确定处已标“需二次核验”。

### P0-01 ATF Primer
- why：ATF 的控制行/结构标记定义了“不可动的语义骨架”，Tier-0 必须保证它们不被清洗破坏。
- likely semantic markers：`# @ $ &`（控制行/元数据）、行号与 `.`、`{ } [ ] ( )`、`-`、`=`、下标数字（如 `₂`）、可能还有 `*`。
- tier impact：Tier-0 / Tier-1
- capture target：capture page header + sections describing ATF control lines / structure + examples section（需二次核验具体小节名）

URL: https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/index.html

### P0-02 ATF Inline Tutorial — Languages section
- why：方言/语言代码（dialect codes）直接出现在行内（如 `%na/%sb`），属于必须保留的语义标记；误删会改变文本语言层信息。
- likely semantic markers：`#atf: lang ...`、`%xx`（行内方言切换）、可能与 `@`/结构行共现。
- tier impact：Tier-0 / Tier-1
- capture target：capture “Languages” 小节 + dialect code 表/列表 + 至少 1 个带 `%xx` 的示例段

URL: https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/inlinetutorial/index.html#h_languages

### P0-03 Lemmatisation primer (intro)
- why：`#lem:` 行结构（separator / ambiguity / 1:1 对齐）决定了清洗时哪些字符是结构性（structural）而非噪声；误处理会造成解析错误或语义丢失。
- likely semantic markers：`#lem:`、`; `（分隔符）、`|`（歧义分隔）、`+`（长格式 lemmatization）、`$`、`#`（形态/特征片段）、`[ ]`（词义/标签）等。
- tier impact：Tier-0 / Tier-1
- capture target：capture “#lem: lines” + “Separator” + “Ambiguity” + any examples showing `; ` and `|`

URL: https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html

### P0-04 Lemmatisation primer — proper nouns section (anchor)
- why：专名 lemmatization 牵涉 explicit vs POS-only 的结构差异；清洗若改动标记会破坏可比性与一致性。
- likely semantic markers：`#lem:`、`PN`、`|`、`+`、`$`、连字符 `-`、花括号 `{ }`（determinatives）等。
- tier impact：Tier-1（为主）/ Tier-0（保留标记）
- capture target：capture anchor section header + definition of explicit vs POS-only + 1–2 examples

URL: https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html#h_propernouns

### P0-05 Akkadian linguistic annotation page
- why：Akkadian 专用页面往往定义 dialect/normalization/CF 等“高收益但高风险”的规范；至少要作为 Tier-1/2 的边界依据。
- likely semantic markers：方言标记 `%xx`、citation form（CF）相关标记、可能涉及 `+ ... [gloss]POS$...#...` 结构片段。
- tier impact：Tier-1 / Tier-2（部分条款可能触及 Tier-0 边界，需二次核验）
- capture target：capture page header + sections referencing dialects/normalization/CF（不确定则 capture header + TOC + examples）

URL: https://oracc.museum.upenn.edu/doc/help/languages/akkadian/index.html

### P0-06 Akkadian citation forms section (anchor)
- why：CF/CACF 与 normalization（例如 mimation、形态规范化）一旦介入就容易变成 Tier-2；但必须先明确“哪些东西绝不能在 Tier-0 做”。
- likely semantic markers：`+`（lem headword）、`$`、`#`、下标数字（如 `₂`）、连字符 `-`、可能涉及 logograms 与分词符号。
- tier impact：Tier-2（主）/ Tier-1（边界定义）
- capture target：capture section header + any example lines showing CACF vs instance spelling（需二次核验）

URL: https://oracc.museum.upenn.edu/doc/help/languages/akkadian/index.html#h_citationformscf

### P0-07 Akkadian stylesheet
- why：样式表是“什么算合法/推荐写法”的硬依据；清洗要么遵循其容忍范围，要么显式声明“不触碰该语义层”。
- likely semantic markers：转写规范中的 `{ } [ ]`、`-`、`.`、`=`、`#atf:`/`@`/`$` 等与 ATF 格式交叉的符号（需二次核验页面是否含这些点）。
- tier impact：Tier-0（边界）/ Tier-1（格式一致化）
- capture target：capture sections describing transliteration conventions + any “do/don’t” examples

URL: https://oracc.museum.upenn.edu/doc/help/languages/akkadian/akkadianstylesheet/index.html

### P0-08 Proper nouns linguistic annotation page (QPN)
- why：明确 `{m}/{f}` 等 determinatives、连字符/双连字符等在专名里是“语义/结构”，清洗误折叠会改变实体边界。
- likely semantic markers：`{m} {f} {d}`、`--`（双连字符）、`-`（内部边界）、大小写（CF 大写规则）、`+ ... PN$`。
- tier impact：Tier-0（保留）/ Tier-1（结构一致化）
- capture target：capture “Transliteration” + “Lemmatization” 段落 + 任何提到 `--` 的示例

URL: https://oracc.museum.upenn.edu/doc/help/languages/propernouns/index.html

### P0-09 Nammu and Emacs (index)
- why：强调“你是在给计算机做标注”，并且提到自动检查/一致性；可作为宪法中“auditability（可审计）/edit_log 必须”部分的旁证。
- likely semantic markers：不直接定义符号，但涉及 `|`、`+`、ATF/lemmatise 工作流与检查点。
- tier impact：Tier-0（流程约束）/ Tier-1（工具辅助一致化）
- capture target：capture page header + sections describing consistency checks / atf-mode tooling mentions

URL: https://oracc.museum.upenn.edu/doc/help/nammuandemacs/index.html

### P0-10 Project Management with Emacs (harvesting lemmatization data)
- why：给出“复核/收集/检查”流程线索，可用于宪法条款：哪些结构错误会触发检查、为何必须保留 1:1 对齐等。
- likely semantic markers：`+`（长格式）、与 `#lem:` 结构相关的检查点提示（需二次核验页面内容/重定向）。
- tier impact：Tier-0（审计）/ Tier-1（结构一致化）
- capture target：capture page header + any section mentioning harvesting notices / checking lemmatization

URL: https://oracc.museum.upenn.edu/doc/help/..index.html#h_harvestingnelemmatisation%20data

## 4. Freeze checklist (only for P0 candidates)
> 冻结格式建议：优先 PDF（打印为 PDF 或浏览器导出）+ 关键段截图（包含 URL/标题）。若页面可稳定渲染，再补 MD 快照。

- P0-01 ATF Primer
  - file_name_suggestion: `references/oracc_P0_atf_primer_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + ATF 控制行/结构示例段可见 + URL 可追溯

- P0-02 ATF Inline Tutorial (Languages)
  - file_name_suggestion: `references/oracc_P0_atf_inlinetutorial_languages_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + dialect code 列表/表格可见 + 至少 1 个 `%xx` 示例可见 + URL 可追溯

- P0-03 Lemmatisation primer
  - file_name_suggestion: `references/oracc_P0_lemmatisation_primer_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + “#lem: lines / Separator / Ambiguity” 段可见 + 示例中 `; ` 与 `|` 可见 + URL 可追溯

- P0-04 Lemmatisation primer (proper nouns anchor)
  - file_name_suggestion: `references/oracc_P0_lemmatisation_primer_propernouns_YYYYMMDD.pdf`
  - done_definition: 锚点所在小节标题可见 + explicit vs POS-only 的说明段可见 + URL（含 #anchor）可追溯

- P0-05 Akkadian linguistic annotation page
  - file_name_suggestion: `references/oracc_P0_akkadian_ling_annotation_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + 至少 1 段与 dialect/normalization/CF 相关内容可见（不确定则 TOC+示例）+ URL 可追溯

- P0-06 Akkadian citation forms section (anchor)
  - file_name_suggestion: `references/oracc_P0_akkadian_citationforms_cf_YYYYMMDD.pdf`
  - done_definition: 锚点小节标题可见 + CACF/CF 的规则或示例段可见 + URL（含 #anchor）可追溯

- P0-07 Akkadian stylesheet
  - file_name_suggestion: `references/oracc_P0_akkadian_stylesheet_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + transliteration conventions 的关键条款段可见 + URL 可追溯

- P0-08 Proper nouns linguistic annotation (QPN)
  - file_name_suggestion: `references/oracc_P0_propernouns_qpn_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + `{m}/{f}` 与连字符/双连字符规则段可见 + 至少 1 条示例可见 + URL 可追溯

- P0-09 Nammu and Emacs
  - file_name_suggestion: `references/oracc_P0_nammuandemacs_index_YYYYMMDD.pdf`
  - done_definition: 页面标题可见 + 强调一致性/检查点的段落可见 + URL 可追溯

- P0-10 Project Management with Emacs (harvesting lemmatization data)
  - file_name_suggestion: `references/oracc_P0_projectmgmt_emacs_harvesting_YYYYMMDD.pdf`
  - done_definition: 页面标题可见（含重定向后的真实标题）+ harvesting/checking 相关段落可见 + URL 可追溯

## 5. Hooks into semantic_cleaning_spec.md (minimal)
> 最小 hook：先覆盖“必须保护的语义标记 + Tier-0 边界 + edit_log 可审计”，不做语言学深水区归并。

- rule_id: `t0_preserve_atf_control_lines`
  - what_it_protects: 保留 ATF 控制行与其行首符号（如 `#...`, `@...`, `$...`, `&...`）的存在与行序（line structure）
  - authority: (P0-01) ATF Primer — https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/index.html
  - risk_note: 误删/合并控制行会破坏文本结构与元数据，属于不可逆语义损失（Tier-0 禁区）

- rule_id: `t0_preserve_dialect_codes`
  - what_it_protects: 方言声明 `#atf: lang ...` 与行内 `%xx` 方言切换标记
  - authority: (P0-02) ATF Inline Tutorial (Languages) — https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/inlinetutorial/index.html#h_languages
  - risk_note: 清洗若删 `%xx` 或改写其位置，会改变语言层信息（semantic marker）

- rule_id: `t0_preserve_brackets_and_dets`
  - what_it_protects: `{ } [ ]` 等括号类结构标记（含 determinatives 如 `{m}/{f}/{d}`）仅允许 Unicode/空白层面的安全规范化
  - authority: (P0-08) Proper nouns (QPN) — https://oracc.museum.upenn.edu/doc/help/languages/propernouns/index.html
  - risk_note: 括号缺失/替换会改变专名/词界/破损标注等语义；Tier-0 禁止“改变是否存在”

- rule_id: `t1_lem_separator_semicolon_space_preserve`
  - what_it_protects: `#lem:` 行中 `; `（分号+空格）作为 lemmatization 分隔符的精确形式
  - authority: (P0-03) Lemmatisation primer — https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html
  - risk_note: 折叠空格或替换分隔符会破坏 `#lem:` 与转写行的 1:1 对齐，导致解析/训练噪声

- rule_id: `t1_lem_ambiguity_pipe_preserve`
  - what_it_protects: `#lem:` 行中 `|`（vertical bar）作为歧义/多候选分隔的结构性标记
  - authority: (P0-03) Lemmatisation primer — https://oracc.museum.upenn.edu/doc/help/lemmatising/primer/index.html
  - risk_note: 若把 `|` 当作噪声清理，会合并多个 lemmatization，语义直接坍塌

- rule_id: `t1_preserve_longform_plus`
  - what_it_protects: lemmatization 长格式中的 `+`（以及与之相关的复核流程线索）
  - authority: (P0-10) Project Management with Emacs (harvesting) — https://oracc.museum.upenn.edu/doc/help/..index.html#h_harvestingnelemmatisation%20data
  - risk_note: 删除/移动 `+` 会影响后续复核与一致性检查路径（auditability），并可能改变解析

- rule_id: `t1_hyphenation_policy_propernouns`
  - what_it_protects: 专名内部边界的 `-` 与可能出现的 `--`（双连字符）不做自动折叠（除非显式 Tier-1 配置并有 ablation）
  - authority: (P0-08) Proper nouns (QPN) — https://oracc.museum.upenn.edu/doc/help/languages/propernouns/index.html
  - risk_note: 盲目把 `--`→`-` 会改变内部词界编码；建议默认“只保留/不改写”

- rule_id: `t1_atf_unicode_consistency_nfc`
  - what_it_protects: 在不改变字符序列语义的前提下，统一 Unicode 形态（NFC/NFKC 需谨慎）以减少同形异码噪声；同时保持 ATF/`#atf: use unicode` 语义线索不受损
  - authority: (P0-02) ATF Inline Tutorial (Languages) — https://oracc.museum.upenn.edu/doc/help/editinginatf/primer/inlinetutorial/index.html#h_languages
  - risk_note: 对下标数字（如 `₂`）等符号做错误归一会改变 token；Tier-0 只能做“极低风险”规范化并记录 edit_log

- rule_id: `t2_block_cf_normalization_by_default`
  - what_it_protects: 默认不在 Tier-0/1 做 CF/CACF 与 mimation 等语义归并；仅在 Tier-2 且有 variant_map/glossary 支撑时允许
  - authority: (P0-06) Akkadian CF section — https://oracc.museum.upenn.edu/doc/help/languages/akkadian/index.html#h_citationformscf
  - risk_note: 这类归并会改变可见字符并引入不可控偏差；必须晚于 A2 稳定后再做 A3（高风险）

- rule_id: `t0_audit_log_required`
  - what_it_protects: 所有 Tier-0/1 规则必须输出 edit_log（可定位、可回滚），并保留原始行结构以便审计
  - authority: (P0-09) Nammu and Emacs — https://oracc.museum.upenn.edu/doc/help/nammuandemacs/index.html
  - risk_note: 无审计日志的清洗会让 OOF 回退时无法定位责任规则，破坏实验纪律


