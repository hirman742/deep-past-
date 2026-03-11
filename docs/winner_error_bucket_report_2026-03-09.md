# 当前正式 Winner 误差桶报告（2026-03-09）

## 1. 对象

本报告分析当前正式 winner：

- checkpoint:
  - `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/checkpoint-250`
- decode:
  - `beam=4`
  - `lp=0.7`
  - `max_new_tokens=384`

对应正式产物：

- [full-val decode best](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/decode_grid_best_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json)
- [full-val diagnose summary](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_diagnostic_summary_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.json)

正式 full-val 指标：

- overall `geom / bleu / chrfpp = 11.2464 / 6.1041 / 20.7207`
- reconstructed `geom / bleu / chrfpp = 14.3323 / 7.7369 / 26.5499`

## 2. 数据与方法

分析输入：

- [row-level diagnostic csv](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_diagnostic_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv)
- [parent-level reconstructed csv](/workspace/deep-past-/runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/diagnostics/val_predictions_reconstructed_continue_s4_bs24_len640_seg5_ckpt250_lp07_fullval_20260309.csv)
- [bucket json](/workspace/deep-past-/logs/winner_error_bucket_report_20260309.json)

样本规模：

- row-level: `1225`
- parent-level: `313`

分桶口径：

- 长度桶：
  - row: `ref_tok<=64 / 65-128 / 129-256 / 257+`
  - parent: `ref_tok<=128 / 129-256 / 257-512 / 513+`
- parent chunk 数桶：
  - `1 / 2-3 / 4-6 / 7+`
- 结构标记桶：
  - `gap/braces/angle`
  - `parens`
  - `subscript`
  - `brackets`
  - standalone `x`
- 失败模式桶：
  - `short_pred`
  - `cap_hit`
  - `repeat_pred`

注意：

- 本报告中的桶内 `geom/bleu/chrf` 是**句子级 proxy 平均值**，用于定位误差集中区。
- 它不是正式 corpus 指标，不应直接替代官方 full-val 指标。

## 3. 总体结论

最重要的结论只有三条：

1. 当前 winner 的主要问题不在短样本，而在**中长样本和多 chunk parent**。
2. 当前 winner 的主要失败模式不是空输出，而是**重复扩写 + 贴近长度上限**。
3. 当前 winner 的上限瓶颈更像**任务形式/上下文组织问题**，而不是继续抠 decode 或 backbone 容量。

## 4. 总体健康

row-level proxy：

- `row_geom_mean = 11.03`
- `row_empty_pct = 0.08%`
- `row_short_pct = 7.76%`
- `row_cap_hit_pct = 25.22%`
- `row_repeat_pct = 78.20%`

parent-level proxy：

- `parent_geom_mean = 15.6558`
- `parent_empty_pct = 0.00%`
- `parent_short_pct = 4.79%`
- `parent_cap_hit_pct = 0.00%`
- `parent_repeat_pct = 83.39%`

解读：

- 空输出不是当前主问题。
- `cap_hit` 和 `repeat_pred` 的占比都很高，尤其在 row-level。
- 到了 parent-level，虽然不再出现 token 级 `cap_hit`，但**重复模板化**仍然非常严重。

## 5. 长度桶

### 5.1 row-level

- `ref_tok<=64`
  - `geom=13.7739`
  - `cap_hit=13.08%`
  - `repeat=58.88%`
- `ref_tok65_128`
  - `geom=20.9241`
  - 这是表现最好的桶
- `ref_tok129_256`
  - `geom=9.4343`
  - `cap_hit=26.61%`
  - `repeat=85.43%`
- `ref_tok257+`
  - `geom=9.1207`
  - `short=14.61%`
  - `cap_hit=35.58%`
  - `repeat=80.90%`

结论：

- 模型最能处理的是中短样本，尤其 `65-128` token 参考长度。
- 一旦进入 `129+` token 区间，分数明显塌下去。
- 超长 row 最典型的问题是：
  - 贴上限
  - 过度重复
  - 以及一部分“明显过短”

### 5.2 parent-level

- `ref_tok<=128`
  - `geom=16.7779`
- `ref_tok129_256`
  - `geom=13.1505`
  - `repeat=100%`
- `ref_tok257_512`
  - `geom=11.2519`
  - `repeat=100%`
- `ref_tok513+`
  - `geom=7.3026`
  - `short=50.00%`
  - `repeat=100%`

结论：

- parent 一长，winner 就不是“偶尔出错”，而是几乎系统性进入重复模板区。
- 最长 parent 桶已经非常接近任务形式上限。

## 6. Parent Chunk 数桶

### 6.1 row-level

- `parent_chunks=1`
  - `geom=14.1578`
- `parent_chunks=2_3`
  - `geom=15.8690`
  - 这是表现最好的桶
- `parent_chunks=4_6`
  - `geom=9.8544`
  - `repeat=85.00%`
- `parent_chunks>=7`
  - `geom=7.2947`
  - `short=12.86%`
  - `repeat=84.76%`

### 6.2 parent-level

- `parent_chunks=1`
  - `geom=14.1578`
- `parent_chunks=2_3`
  - `geom=18.5538`
  - 这是全报告最强桶之一
- `parent_chunks=4_6`
  - `geom=12.4272`
  - `repeat=100%`
- `parent_chunks>=7`
  - `geom=12.0495`
  - `repeat=100%`

结论：

- 当前主线非常适合 `2-3 chunk parent`。
- 一旦 parent 超过 `4` 个 chunk，表现急剧恶化。
- 这比单纯长度桶更能说明问题：
  - 当前瓶颈不是 token 容量本身
  - 而是**多 chunk 信息整合失败**

## 7. 结构标记桶

### 7.1 gap/tag

row-level：

- `has_gap_tag`
  - `n=514`
  - `geom=10.8084`
  - `short=8.56%`
  - `cap_hit=20.23%`
  - `repeat=78.21%`

parent-level：

- `has_gap_tag`
  - `n=211`
  - `geom=13.5842`
  - `repeat=92.42%`

结论：

- 带 `<gap> / {..}` 一类结构标记的样本明显更难。
- 但它们的问题也不是“完全不会”，而是更容易落到重复模板。

### 7.2 subscript

row-level：

- `has_subscript`
  - `n=606`
  - `geom=11.3974`
  - `cap_hit=29.37%`
  - `repeat=82.51%`

parent-level：

- `has_subscript`
  - `n=232`
  - `geom=14.0855`
  - `repeat=93.53%`

结论：

- 下标本身不是直接灾难点。
- 但带下标的样本往往也是结构更复杂、长度更长的样本，因此更容易伴随重复与贴上限。

### 7.3 brackets / standalone x

本次 full-val 切片里：

- `has_brackets = 0`
- `has_x_marker = 0`

结论：

- 当前 winner 的 full-val 切片不适合拿来判断 `[...]` 或 standalone `x` 的行为。
- 如果下一轮要专门处理破损文本，应该单独做 targeted bucket，不应依赖这次切片。

## 8. 失败模式

### 8.1 主要失败模式排序

按当前数据看，失败模式排序是：

1. `repeat_pred`
2. `cap_hit`
3. `short_pred`
4. `empty_pred`

这和正式 diagnose summary 里的 top repeated predictions 是一致的：

- `"Seal of the silver ..."`
- `"If you have not paid the silver ..."`
- `"miss sistems sistems ..."`

### 8.2 最坏样本形态

最差的 parent 样本主要是这几类：

- 破损很多、`<gap>` 很多，但 reference 很短  
  结果：模型反而输出冗长重复的“Seal of ...”

- 极短 reference，但 source 仍带大量结构与专名  
  结果：模型用固定书信/封印模板过度扩写

- 多 chunk 财务/清单类文本  
  结果：不断重复“silver / seal / minas / shekels”模板

代表例子见：

- [bucket json](/workspace/deep-past-/logs/winner_error_bucket_report_20260309.json)
  - `worst_rows`
  - `worst_parents`

## 9. 对下一阶段实验的含义

这份误差桶报告支持下面的判断：

1. 不该继续把主力放在 `beam / lp / cap / checkpoint` 微调上。  
   因为主要失败模式不是局部 decode 没拧对，而是多 chunk/长样本整合失败。

2. 不该优先继续做“更大 backbone 冷起”。  
   架构升级 probe 已经给出负证据，而误差桶又显示问题集中在任务组织层。

3. 下一阶段最值得切的主线是 `parent-aware / parent-packed`。  
   证据就是：
   - `2-3 chunk` 很强
   - `4+ chunk` 明显崩
   - 说明模型不是完全不会翻，而是不会跨太多 chunk 整合

4. `hard-case replay` 可以作为第二优先级。  
   因为现在最差样本桶已经很清楚，可以直接从这些桶中抽 replay list。

## 10. 结论

一句话总结：

- 当前正式 winner 已经把 `ByT5-small + chunk` 体系的局部最优基本榨干了。
- 它的主要误差不在短样本，而在**长样本、多 chunk parent、结构标记丰富样本**。
- 它的主失败模式不是空输出，而是**重复模板化与贴长度上限**。
- 因此下一轮主线应该优先切到 **parent-aware / parent-packed**，而不是继续做 decode 或 backbone 细调。
