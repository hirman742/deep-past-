# Semantic Cleaning Spec
# semantic_cleaning_spec.md — Deep Past MT (Akkadian transliteration → English)

> Cleaning Constitution / 宪法级先验  

> 版本：v0.1.1（在 v0.1.0 基础上补齐“字符红线 + 异常字符条款 + 规则条款模板”）  

> 日期：2026-02-01 (UTC+8)  

> 评分函数：Score = sqrt(BLEU * chrF++)（所有清洗决策以 OOF 为准）

  

---

  

## Changelog（变更摘要）

- v0.1.1（本次）

  - 新增：**字符级红线表**（Tier-0/1 禁改清单 + 空白规则边界）

  - 强化：**未知/异常可见字符**（含 `„`）的“透传 + 告警 + 禁止自动映射”条款

  - 新增：**规则条款模板**（rule_id → tier → allowed/forbidden → authority_anchor → unit tests）

- v0.1.0（上一版）

  - Gate 0-A 完成：inventory / patterns / test-only 差集 + Tier-0 可落地

  

---

  

## 0. 本文档的作用域（Scope）

  

本文档定义 **source-side（transliteration）** 的清洗与规范化（normalization）准则，用于：

  

- 约束 `cleaning/normalize.py: normalize_source(text, config) -> (normalized_text, edit_log)`

- 约束三层清洗：Tier-0 / Tier-1 / Tier-2（每条规则可开关、可 ablation）

- 约束实验纪律：A0 RAW → A1 Tier-0 → A2 Tier-0+Tier-1 → A3 Tier-0+Tier-1+Tier-2  

- 保证：**train 与 inference 完全一致**，且任何规则都有 **edit_log 可定位、可回滚**

  

⚠️ 非作用域（Explicitly out of scope）：

- 不改模型结构、不调 LoRA、不改解码超参

- 不对英文目标做“润色”或风格后处理

- 不引入不合规外部平行数据

  

---

  

## 1. 宪法级不变量（Non-negotiables）

  

### 1.1 语义标记不可被“消失或凭空出现”

在 **Tier-0** 中，禁止任何操作导致下列事情发生：

- 某类“语义标记（semantic marker）”从文本中被删除（是否存在改变）

- 某类语义标记被新增（是否存在改变）

  

> Tier-0 只允许：空白/不可见字符/Unicode 形态规范化/极低风险统一（且不改变可见语义标记）

  

### 1.2 可审计、可回滚、可复现

- 每条规则必须有 `rule_id`，可在 config 中独立启用/禁用

- 每次命中必须写入 `edit_log`（包含 rule_id、before/after、上下文或备注）

- normalize 必须 **幂等（idempotent）**：`normalize(normalize(x)) == normalize(x)`（同一 config）

  

### 1.3 “空白不是无辜的”：分隔符不可被吞掉

Tier-0/Tier-1 的任何空白规则必须满足：

- **不得**把两个原本被空白分隔的 token 合并为一个（即：不得删除“最后一个分隔空白”）

- **允许**：把 `多个空白/Tab` 折叠成 **1 个空格**；把 `\r\n` 统一为 `\n`

- **不得**：跨行合并（不得移除 `\n` 的存在性）

  

---

  

## 2. 证据清单与数据事实（Evidence & Facts）

  

### 2.1 Gate 0-A 产物（当前已存在）

以下文件构成本宪法的“证据附件”（audit trail）：

- `artifacts/symbol_inventory_train.json`

- `artifacts/symbol_inventory_test.json`

- `artifacts/patterns_train.md`

- `artifacts/patterns_test.md`

- `artifacts/symbols_only_in_test.txt`

  

建议记录（可选）：

- sha256（前 16 位）  

  - symbol_inventory_train.json: `01a500d5921156ef`  

  - symbol_inventory_test.json : `c728c232eff77aab`  

  - patterns_train.md          : `315f848dea377e21`  

  - patterns_test.md           : `b2c1fa472aac4361`  

  - symbols_only_in_test.txt   : `cea7cce57ff559ea`

  

### 2.2 竞争数据（train/test）符号事实（来自 symbol inventory）

- train：**90** 个不同字符（unique symbols）

- test ：**42** 个不同字符

- test-only（仅出现在 test，不在 train）：**1 个字符**

  - `„` (U+201E DOUBLE LOW-9 QUOTATION MARK)，出现 **6** 次  

  - 样例片段（来自 test inventory）：`qí-bi„-ma`、`u„-mì-im` 等

  

### 2.3 关键结构模式事实（来自 patterns）

train 的主要结构模式计数：

- `subscript_digits`（₀₁₂₃₄₅₆₇₈₉）：**3668**

- `parens`（(...)）：**1098**

- `brackets`（[...]）：**210**

- 其余（{...}, <...>, |...|, @/$/#/% 控制行）：在 competition train/test 中 **0**（但见 2.4）

  

test 的结构模式计数：

- `parens`：2

- 其余：0

  

### 2.4 旁路语料（proxy）提醒：published_texts 的结构标记（强建议纳入风险评估）

如果你使用 `published_texts.csv` 作为“隐藏 test 的近似 proxy”，可观察到：

- `<...>`（angle tags，如 `<big_gap>`, `<gap>`）出现 **32961** 次

- `{...}`（如 `{TÚG}` / `{large break}` 一类标记）出现 **6111** 次

- `_`（下划线，常用于 `<big_gap>`）大量存在

  

结论：即便 competition train/test 中没出现 `<...>` 或 `{...}`，它们在同源语料中非常常见，应被视为 **潜在语义标记**，至少在 Tier-0/Tier-1 中严格保护。

  

### 2.5 权威证据冻结（Evidence Freeze）——把“链接”变成“可审计证据”

本项目所有“语义标记的定义/允许与禁止”最终应能落到 **冻结证据** 上（截图/PDF/MD 快照），并由 `references/oracc_authority_index.md` 索引。

  

约定（MVP）：

- 每个权威条目对应一个 `authority_anchor`（稳定 ID），例如：

  - `ORACC-ATF-UNICODE@20260201`

  - `ORACC-ATF-BROKEN-MARKERS@20260201`

  - `ORACC-ATF-TAGS-GAP@20260201`

- 快照建议放置路径（示例）：

  - `references/freeze/<authority_anchor>.pdf`

  - `references/freeze/<authority_anchor>.png`（关键段落截图）

  - `references/freeze/<authority_anchor>.md`（页面正文快照）

  

---

  

## 3. 术语与分类（Definitions）

  

### 3.1 语义标记（semantic marker）

满足任一条件即为语义标记：

- 承载音位/字母差异：如 `š, ḫ, ṣ, ṭ, ʾ` 及带重音/变体 `á/à, í/ì, ú/ù` 等

- 承载 sign index：如下标数字 `₀…₉`（例：`PUZUR₄`, `il₅`）

- 承载破损/缺失/不确定：如 `[...]`, `x`, `…`, `<gap>`, `<big_gap>` 等

- 承载结构/注释：如 `{TÚG}`, `{large break}` 或 ORACC/ATF 控制行 `#...`, `@...`, `%...`, `$...`（即便当前 train/test 未出现，也按权威格式保护）

  

### 3.2 纯噪声（pure noise）

满足下列条件且不影响语义标记存在性，可归为“纯噪声”：

- 不可见字符：如 `\u200b \u200c \u200d \ufeff` 等（零宽/ BOM）

- 不稳定的行内空白冗余：多空格、制表符（但换行是否存在要谨慎；见 Tier-0）

  

---

  

## 4. Tier 分层边界（Hard boundaries）

  

## 4.1 Tier-0（安全规范化 / safe normalization）

**目标**：消除不可见字符与非决定性的 Unicode 形式差异，让同一信息更一致；  

**禁止**：删除/新增任何语义标记；更改字母/符号本体；“纠错式改写”。

  

允许操作（Allowed Ops）：

1) Unicode 规范化：**NFC**（禁止 NFKC）

2) 移除不可见字符：零宽字符、BOM

3) 空白规范化：  

   - 统一换行 `\r\n`→`\n`  

   - 行内空白折叠为单空格（保留换行本身）

   - 不做“自动补齐括号/配对修复”

  

> Tier-0 的输出应保证所有可见符号集合不变（除了空白/不可见字符）。

  

## 4.2 Tier-1（结构显式化 / structure canonicalization）

**目标**：对“已在证据中出现的结构变体”做一致化表达；  

**原则**：信息量不降低（不得通过删除解决问题）。

  

Tier-1 只能基于 `patterns_*.md` 或其它明确证据提出规则。  

当前可考虑但默认不启用的方向（示例）：

- 规范化 `<gap>` / `<big_gap>` 标签的大小写、空格、下划线（需要 proxy 证据 + 单测）

- 规范化 `{TÚG}` 等花括号标记的周围空白（不改变 `{...}` 内容）

  

⚠️ Tier-1 仍然禁止：

- 合并/删除 `[...]`、`x`、`…`

- 把下标数字 `₄` 改写为普通数字 `4`（语义会变）

- 自动平衡括号（因为数据中存在不平衡计数：`(` 与 `)` 频次不等）

  

## 4.3 Tier-2（语义归并 / semantic merging，高风险）

**禁止默认启用**。只有在具备：

- `variant_map.tsv`（来源可解释、可审计）

- `glossary_min.csv`（最小词表/变体归并表）

并且 A2 稳定（Tier-0 + Tier-1 已通过 smoke 与 OOF）后，才允许进入 A3。

  

---

  

## 5. 受保护的语义标记清单（Protected Markers）

  

### 5.0 字符级红线表（Tier-0/1 禁改）

这张表的目标：把“保护”从宣言变成可执行约束。

  

**红线含义（Tier-0/1）**：

- **不得**删除/新增这些符号

- **不得**把它们替换为“看起来相近”的符号（包括 ASCII 近似替换）

- **不得**做大小写折叠（除非你能证明大小写不承载信息；当前默认“承载信息”）

- **空白规则不得改变 token 边界**：不得吞掉最后一个分隔空白；不得人为插入围绕符号的新空格

  

| 类别 | 红线对象（示例） | Tier-0/1 禁止操作 | 备注（风险） |

|---|---|---|---|

| 音位/字母差异 | `š ḫ ṣ ṭ ʾ`；`á à é è í ì ú ù`（含大写） | 去音标、ASCII 近似替换、大小写折叠 | 直接改音位/词形，等价于改文本内容 |

| sign index | `₀₁₂₃₄₅₆₇₈₉`；（以及数据中出现的 `ₓ`） | `₄→4`、删除下标、把下标当普通字符清洗掉 | 直接改 sign 编号，语义变化巨大 |

| 破损/不确定 | `[...]`、`x`、`…` | 删除/合并/替换为普通标点 | 破损/缺失是关键语义标记 |

| 结构分隔符 | `-`、`.`、`+` | 删除、替换、或“规范化”成其他连字符/点号；向两侧插入新空格 | `-` 和 `.` 在转写里常是结构分隔，不是普通英文标点 |

| 括号系统 | `( )`、`[ ]`、（以及罕见的 `⌈` 等） | 自动补齐/自动配平；替换括号形态；删除括号 | 数据里存在不平衡现象，不能“修复式清洗” |

| 结构标签（proxy 强提示） | `<...>`、`{...}`、`|...|`、`# @ % $` 控制行 | 删除、替换、内容改写；在 tag 内做“纠错式”空白/大小写变化 | competition 里未必出现，但 proxy 极常见，必须先保护 |

  

> 工程落地要求：实现层必须提供一个 `PROTECTED_CHARSET`（可配置）与一个“未知字符告警”规则（见 5.5 & 6.3），并在 edit_log 中记录命中。

  

### 5.1 音位/字母类（必须保留）

来自 train/test inventory 的非 ASCII 字母与重音（示例，不限于）：

- `š, ḫ, ṣ, ṭ, ʾ`（含大写变体）

- `á, à, í, ì, ú, ù, é, è` 及其大写变体（如 `Ù, Í, É`）

  

**禁止**：去音标、大小写折叠、替换为近似 ASCII（如 `š`→`s`）。

  

### 5.2 下标数字（必须保留）

`₀₁₂₃₄₅₆₇₈₉`（train 中出现频繁；subscript_digits=3668）  

例：`PUZUR₄`, `il₅`, `tur₄`, `en₆` 等。

  

补充（来自实际字符集合）：`ₓ`（LATIN SUBSCRIPT SMALL LETTER X）在 train 中出现极少但属于“可见符号”，一律按红线保护。

  

**禁止**：`₄ -> 4`。

  

### 5.3 破损/缺失/不确定标记（必须保留）

- `[...]`（train brackets=210）

- `x`（train 中频繁出现；常代表不确定符号）

- `…`（ellipsis；常用于省略/缺失）

- proxy 中强出现的：`<gap>`, `<big_gap>` 及 `<...>` 标签

  

### 5.4 结构标记（必须保留）

- `(...)`（train parens=1098；常见如 `(ki)`）

- `-`（hyphen-minus；train 中极高频，属于结构分隔符）

- `.`（full stop；在转写中既用于数值小数点，也可能用于 logogram/缩写结构）

- `+`（plus；虽低频但真实出现，应保护）

- `{...}`（proxy 中强出现，如 `{TÚG}`）

- `|...|`（proxy 中偶见，仍视为结构标记）

- `⌈`（LEFT CEILING；train 中极罕见，默认按“未知/罕见但可见符号”冻结保护）

  

### 5.5 unknown / rare visible symbols（默认“透传 + 告警”，禁止自动映射）

**核心原则**：任何“可见字符”的自动纠正/映射都属于 **Tier-2 风险**，除非你能给出权威证据 + ablation 证明收益。

  

#### 5.5.1 未知字符（unknown）定义

- unknown = “当前输入中出现，但 **不在 train 的可见字符集合** 中的字符”

- 对 unknown：Tier-0/Tier-1 **一律不改写、不删除**，仅写入告警日志（edit_log）

  

> 注意：unknown 的处理是“模型鲁棒性与审计”的问题，不是“纠错”的借口。

  

#### 5.5.2 test-only 字符 `„`（U+201E）的专门条款

当前 test-only 集合：`„`，出现 6 次，且不在 train 中出现。

  

Tier-0/Tier-1 处理策略（必须）：

- **透传（passthrough）**：输出中原样保留 `„`

- **强制告警**：edit_log 写入 `rule_id=t0_unknown_visible_char_alert`（或等价 rule_id），附：

  - 字符、codepoint、命中次数、上下文片段（<=80 字符）

- **禁止**：自动替换为 `"`、`'` 或任何“看起来合理”的符号（因为缺少可验证语义等价）

  

若要“修复/映射”（例如统一为 `"`），必须满足（Tier-2 门槛）：

1) 有 `authority_anchor` 对应的冻结证据，证明该字符属于编码噪声且映射等价  

2) 有 `variant_map.tsv` 或明确可审计映射表（来源说明 + 版本号）  

3) 先做 A/B smoke + 正式 OOF，对比 BLEU/chrF++/Score  

4) 默认放在 Tier-2（高风险），并且必须可一键回滚

  

---

  

## 6. 规则体系与接口契约（Rules & API Contract）

  

### 6.0 规则条款模板（每条规则必须这样写清楚）

每条规则在 spec 与实现里都应具备以下字段（缺一不可）：

  

- `rule_id`：稳定唯一（例如 `t0_unicode_nfc`）

- `tier`：t0 / t1 / t2

- `default_on`：true/false（默认是否启用）

- `type`：`transform` 或 `log_only`

- `allowed_ops`：允许做什么（精确到字符层面）

- `forbidden_ops`：禁止做什么（写成“硬红线”）

- `scope_guard`：必须绕开的受保护对象（例如“不得触碰 PROTECTED_CHARSET；不得改写 `{...}` 内容”）

- `authority_anchor`：权威证据锚点（没有则写 `TBD`，但不得空缺）

- `unit_tests`：至少 3 组（should-change / should-not-change / ambiguous-only-log）

  

> 实现层要求：`normalize_source` 必须能输出（normalized_text, edit_log），且 edit_log 至少记录 rule_id、before、after、note。

  

### 6.1 normalize_source 签名

`normalize_source(text, config) -> (normalized_text, edit_log)`

  

- `config` 必须包含：

  - `tier`：t0/t1/t2

  - `rules`: `{rule_id: bool}`

  - `logging`: 控制日志粒度（可选）

  - （推荐）`known_visible_charset_train`：用于 unknown 字符告警（来自 symbol_inventory_train）

  

### 6.2 edit_log schema（最小字段）

每条 edit log 记录至少包含：

- `rule_id`

- `before`（被修改片段；log_only 可为空或与 after 相同）

- `after`（替换后片段；log_only 可为空或与 before 相同）

- `note`（简短备注：原因/类型/位置）

  

建议扩展字段（可选但推荐）：

- `pos`（字符位置或行列）

- `context`（前后窗口，<=80 字符）

- `sample_id`（oare_id 或 test id）

- `severity`（info/warn/block）

  

### 6.3 Tier-0 规则列表（建议默认启用）

下面把 Tier-0 的每条规则都写成“条款模板”格式（可直接对应实现）。

  

#### Rule: `t0_unicode_nfc`

- tier: t0

- default_on: true

- type: transform

- allowed_ops:

  - 对全字符串执行 `unicodedata.normalize("NFC", text)`

- forbidden_ops:

  - 禁止 NFKC/NFKD（可能引入兼容分解导致语义变化）

- scope_guard:

  - 不得删除/新增任何可见字符；NFC 仅改变组合形式

- authority_anchor: `ORACC-ATF-UNICODE@20260201`（TBD：需冻结证据）

- unit_tests:

  - should-change: 含“分解重音”的等价写法应归一

  - should-not-change: `š ḫ ṣ ṭ ʾ ₄ …` 等字符保持本体

  - ambiguous-only-log: 无

  

#### Rule: `t0_remove_invisible`

- tier: t0

- default_on: true

- type: transform

- allowed_ops:

  - 删除：`\u200b \u200c \u200d \ufeff`（零宽/ZWJ/BOM）

- forbidden_ops:

  - 禁止删除任何“可见字符”（包括组合附标、下标数字、标点）

- scope_guard:

  - 仅对上述 codepoint 生效（白名单删除）

- authority_anchor: `TBD`（无需强权威，但建议记录为“工程噪声清理”）

- unit_tests:

  - should-change: `"a\u200b-b"` → `"a-b"`

  - should-not-change: `"PUZUR₄"` 不得受影响

  - ambiguous-only-log: 无

  

#### Rule: `t0_whitespace_normalize`

- tier: t0

- default_on: true

- type: transform

- allowed_ops:

  - `\r\n` → `\n`

  - 将 `\t` 视为一个空白并折叠

  - 将“行内连续空白”折叠成 **单空格**

- forbidden_ops:

  - 禁止跨行合并（不得删除 `\n`）

  - 禁止删除“最后一个分隔空白”（不得把两个 token 粘连）

  - 禁止插入“围绕结构符号的新空格”（例如不得把 `a-b` 变成 `a - b`）

- scope_guard:

  - 不得改变 5.0 红线对象的字符序列（除空白折叠本身）

- authority_anchor: `TBD`（工程规范；不直接依赖 ORACC）

- unit_tests:

  - should-change: `"a\t\tb"` → `"a b"`

  - should-not-change: `"qí-bi„-ma"`（不触碰 `„`，且不得插入 `qí-bi „ -ma` 这种空格）

  - ambiguous-only-log: 若输入出现异常“无空格粘连”，只告警不自动插空格（避免纠错式清洗）

  

#### Rule: `t0_unknown_visible_char_alert`

- tier: t0

- default_on: true

- type: log_only

- allowed_ops:

  - 扫描文本：若出现不在 `known_visible_charset_train` 中的可见字符 → 写入 edit_log（warn）

- forbidden_ops:

  - 禁止修改文本（log_only）

- scope_guard:

  - unknown 字符必须原样透传

- authority_anchor: `TBD`（其价值来自审计与风控）

- unit_tests:

  - should-change: 无（不改文本）

  - should-not-change: 输入输出完全一致

  - ambiguous-only-log: `„` 命中时应记录 codepoint 与上下文

  

Tier-0 明确禁止的规则（即便你想到也不得放进 Tier-0）：

- `t0_strip_diacritics`（去音标）——绝对禁止

- `t0_subscript_to_digit`（₄→4）——绝对禁止

- `t0_bracket_drop`（删除 [...] 或 x）——绝对禁止

- `t0_auto_balance_parens`（自动补括号）——绝对禁止

- `t0_auto_insert_spaces`（自动插空格修复）——绝对禁止（这是纠错式改写）

  

---

  

## 7. Gate / Ablation 实验纪律（与主项目对齐）

  

### Gate 0：训练前清洗（不跑模型也可完成）

必须完成：

- 本宪法（semantic_cleaning_spec.md）

- symbol inventories / patterns / test-only 差集

- normalize() 工程骨架（规则开关 + edit_log）

- unknown 字符告警（至少 log_only）

  

### Gate 1：smoke baseline（RAW）

- 1 折 / 固定 split

- 少步数

- RAW 输入

- 产出 BLEU/chrF++/Score + 最基本错误样例桶

  

### Gate 2：Tier-0（A0 vs A1）

- A0: RAW

- A1: Tier-0

- 预期：chrF++ 上升；BLEU 不应明显崩；Score 不应更差（至少噪声范围内）

  

### Gate 3：Full baseline CV（正式 OOF）

当你要合入 Tier-1/Tier-2、tokenizer 对照、模型对照时必须跑。

  

---

  

## 8. 单测与验收（Tests）

  

### 8.1 必须通过的性质测试（property tests）

- 幂等：`norm(norm(x)) == norm(x)`

- Tier-0 可见字符集合不变（除了空白/不可见字符）  

  - 可见字符 = 除 `space/tab/newline` 与不可见字符外的所有字符

- 受保护标记存在性不变（至少覆盖 5.0 红线）：

  - `- . + ( ) [ ] { } < >`、`₀…₉`、`š ḫ ṣ ṭ ʾ`、`x`、`…`

- unknown 字符策略：

  - unknown 出现时：输出必须原样透传 + edit_log(warn)

  

### 8.2 样例测试（来自 inventory/patterns 的片段）

以下样例必须在 Tier-0 后仍保持关键标记：

- `PUZUR₄-a-šur`（下标数字保留；不得 `₄→4`）

- `[...] x x x`（破损标记保留；空白可折叠但不可消失）

- `aa a-lim(ki)`（括号保留；不得自动配平/改写）

- `qí-bi„-ma`（`„` 默认保留；只告警不替换）

- `0.33333`（小数点保留；不得改成逗号或别的点号）

- `a-lá+lá-x`（`+` 保留；不得替换/删除）

  

---

  

## 9. 附录：文件指针（Repository Pointers）

  

- Gate 0-A 证据产物：

  - `artifacts/symbol_inventory_train.json`

  - `artifacts/symbol_inventory_test.json`

  - `artifacts/patterns_train.md`

  - `artifacts/patterns_test.md`

  - `artifacts/symbols_only_in_test.txt`

  

- 权威来源索引（用于后续“冻结证据”与 Tier-1/2 合法性说明）：

  - `references/oracc_authority_index.md`

  - `references/minimal_linguistics_patch_oracc.md`（若存在）

  

---

  

## 10. 当前开放 TODO（允许存在，但必须可闭环）

- [TODO] 完成 `authority_anchor` 的冻结证据：至少覆盖 Unicode 规范、破损标记、gap/tags、花括号标记的定义

- [TODO] 对 test-only 字符 `„` 做权威定性：是编码噪声还是可解释标记  

  - 未定性前：严格执行 5.5（透传 + 告警 + 禁止自动映射）

- [TODO] Tier-1 规则候选清单：只允许基于 patterns + 权威证据提出，并逐条单测与 smoke 验证

  

（v0.1.1 结束）

---

  

## Gate 0-A Evidence Attachments (Auto)

  

Evidence inputs (repo-relative):

  

- `artifacts/symbol_inventory_train.json`

- `artifacts/symbol_inventory_test.json`

- `artifacts/patterns_train.md`

- `artifacts/patterns_test.md`

- `artifacts/symbols_only_in_test.txt`

  

Generated facts (do not hand-edit):

  

- `docs/spec_facts.json`

  

### Charset Summary

  

- train_unique_symbols: `90`

- test_unique_symbols: `42`

- test_only_symbols_count: `1`

  

### Test-Only Symbols (from `artifacts/symbols_only_in_test.txt`)

  

| codepoint | char | name | test_count |

| --- | --- | --- | --- |

| U+201E | „ | DOUBLE LOW-9 QUOTATION MARK | 6 |

  
  

### Patterns Frequency (selected)

  

| pattern_id | train_match_count | test_match_count |

| --- | --- | --- |

| braces | 0 | 0 |

| brackets | 210 | 0 |

| parens | 1098 | 2 |

| angle | 0 | 0 |

| pipe | 0 | 0 |

| at_line | 0 | 0 |

| dollar_line | 0 | 0 |

| hash_line | 0 | 0 |

| percent_code | 0 | 0 |

| subscript_digits | 3668 | 0 |

  

### Evidence Fingerprints (sha256 first16)

  

| file | sha256_first16 |

| --- | --- |

| patterns_test.md | b2c1fa472aac4361 |

| patterns_train.md | 315f848dea377e21 |

| symbol_inventory_test.json | c728c232eff77aab |

| symbol_inventory_train.json | 01a500d5921156ef |

| symbols_only_in_test.txt | cea7cce57ff559ea |

  

Regenerate:

  

```bash

python scripts/_extract_facts_for_spec.py

python scripts/_build_semantic_cleaning_spec.py

```

  

## Rule Registry (Tier-0 defaults)

  

| rule_id | tier | default | what it does | invariants | unit-test idea |

| --- | --- | --- | --- | --- | --- |

| `t0_unicode_nfc` | t0 | on | Unicode NFC normalize only | Never use NFKC; visible symbols preserved (except safe normalization) | NFC stability; no token loss |

| `t0_remove_invisible` | t0 | on | Remove ZWSP/ZWNJ/ZWJ/BOM | Only removes `​ ‌ ‍ �` | Input containing these shrinks; others unchanged |

| `t0_whitespace_normalize` | t0 | on | `

  

`→`

`; fold inline whitespace; keep newlines | Newlines remain; no cross-line merges | Preserve line count; collapse spaces within lines |

  

## Tier-0 Forbidden Operations (hard)

  

- NFKC/NFKD normalization

- Removing or rewriting any bracket/brace/paren/angle marker: `{}` `[]` `()` `<>`

- Auto-fixing or adding missing markers

- Mapping subscript digits (e.g. `₄`) to ASCII digits (`4`)

- Removing diacritics / accent folding

- Deleting ATF control lines or prefixes (e.g. lines starting with `@`, `$`, `#`)

  

## Audit & Rollback

  

`normalize_source(text, config) -> (normalized_text, edit_log)` must emit `edit_log` entries with at least:

  

- `rule_id`

- `before`

- `after`

- `note`

  

`runs/` tracking policy (must match `.gitignore`): track only:

  

- `runs/**/metrics.json`

- `runs/**/hit_stats.json`

- `runs/**/worst_examples.md`

  

## How To Reproduce Gate 0-A + Tier-0

  

```bash

python scripts/gate0_make_artifacts.py

python scripts/gate0_validate_artifacts.py

  

python scripts/normalize_dataset.py --split train

python scripts/normalize_dataset.py --split test

  

python scripts/summarize_edit_log.py --split train

python scripts/summarize_edit_log.py --split test

```

