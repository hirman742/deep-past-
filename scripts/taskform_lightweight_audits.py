from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pack_rows(value: str) -> int:
    text = value if isinstance(value, str) else ""
    return max(1, len([x for x in text.splitlines() if x.strip()]))


def _score_proxy_row(text: str, genre: str) -> float:
    value = text if isinstance(text, str) else ""
    words = [x for x in value.split() if x]
    has_gap = 1.0 if "<gap>" in value else 0.0
    has_brace = 1.0 if "{" in value else 0.0
    has_sub = 1.0 if re.search(r"[₀₁₂₃₄₅₆₇₈₉ₓ]", value) else 0.0
    word_score = min(4.0, len(words) / 40.0)
    genre_norm = (genre or "").strip().lower()
    genre_boost = 1.0 if genre_norm in {"letter", "debt note", "account", "memorandum"} else 0.0
    return (4.0 * has_gap) + (2.0 * has_brace) + has_sub + word_score + genre_boost


def run_parentpack(args: argparse.Namespace) -> None:
    train = pd.read_csv(REPO_ROOT / "data" / "processed_taskform_parentpack_fold0" / "train_proc.csv")
    audit = json.loads((REPO_ROOT / "data" / "processed_taskform_parentpack_fold0" / "audit_parentpack.json").read_text(encoding="utf-8"))
    train["pack_rows"] = train["source_raw"].fillna("").astype(str).map(_pack_rows)
    train["src_chars"] = train["source_raw"].fillna("").astype(str).str.len()
    train["tgt_chars"] = train["target_raw"].fillna("").astype(str).str.len()

    by_mode = (
        train.groupby("chunk_mode", dropna=False)
        .agg(rows=("oare_id", "count"), parents=("parent_oare_id", "nunique"), avg_pack_rows=("pack_rows", "mean"), avg_src_chars=("src_chars", "mean"), avg_tgt_chars=("tgt_chars", "mean"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    by_pack = (
        train.groupby("pack_rows", dropna=False)
        .agg(rows=("oare_id", "count"), parents=("parent_oare_id", "nunique"), avg_src_chars=("src_chars", "mean"), avg_tgt_chars=("tgt_chars", "mean"))
        .reset_index()
        .sort_values("pack_rows")
    )

    payload = {
        "overall": audit,
        "by_mode": by_mode.to_dict(orient="records"),
        "by_pack_rows": by_pack.to_dict(orient="records"),
    }
    _write_json(REPO_ROOT / "logs" / "taskform_parentpack_bucket_audit_20260309.json", payload)

    lines = [
        "# Taskform 轻量实验 1：parentpack 桶审计（2026-03-09）",
        "",
        "## 总览",
        "",
        f"- 原始行数：`{audit['rows_in']}`",
        f"- parentpack 后行数：`{audit['rows_out']}`",
        f"- parent 数：`{audit['parents']}`",
        f"- 平均 pack 大小：`{audit['avg_pack_size']:.2f}`",
        f"- 最大 pack 大小：`{audit['max_pack_size']}`",
        "",
        "## 按 chunk_mode",
        "",
        "| chunk_mode | rows | parents | avg_pack_rows | avg_src_chars | avg_tgt_chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_mode.to_dict(orient="records"):
        lines.append(
            f"| {row['chunk_mode']} | {int(row['rows'])} | {int(row['parents'])} | {row['avg_pack_rows']:.2f} | {row['avg_src_chars']:.1f} | {row['avg_tgt_chars']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 按 pack 行数",
            "",
            "| pack_rows | rows | parents | avg_src_chars | avg_tgt_chars |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in by_pack.to_dict(orient="records"):
        lines.append(
            f"| {int(row['pack_rows'])} | {int(row['rows'])} | {int(row['parents'])} | {row['avg_src_chars']:.1f} | {row['avg_tgt_chars']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- `parentwindow_3ofN` 是主力模式，说明这轮任务形式升级的主体其实是“多 chunk parent 的窗口化重组”。",
            "- 平均 pack 大小接近 `2`，意味着数据并没有被粗暴拼成超长输入，而是仍然保持短窗口策略。",
            "- 这个审计主要用于解释后面 `P1` 的 matched baseline 和 probe 结果。"
        ]
    )
    _write_text(REPO_ROOT / "docs" / "taskform_parentpack_bucket_audit_2026-03-09.md", "\n".join(lines) + "\n")


def run_replay(args: argparse.Namespace) -> None:
    paths = {
        "replay25": REPO_ROOT / "data" / "processed_taskform_replay25_fold0" / "audit_hardcase_replay.json",
        "replay40": REPO_ROOT / "data" / "processed_taskform_replay40_fold0" / "audit_hardcase_replay.json",
    }
    audits = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}

    rows = []
    for name, audit in audits.items():
        hard_rows = max(1, int(audit["hard_rows"]))
        extra = int(audit["extra_rows_added"])
        before = int(audit["train_rows_before"])
        rows.append(
            {
                "name": name,
                "ratio": float(audit["ratio"]),
                "train_rows_before": before,
                "hard_rows": hard_rows,
                "extra_rows_added": extra,
                "rows_after": before + extra,
                "hard_reweight_mult": round((hard_rows + extra) / hard_rows, 4),
                "extra_over_train_pct": round(100.0 * extra / max(1, before), 2),
                "chunk_total_ge_4": int(audit["criteria"]["chunk_total_ge_4"]),
                "target_len_ge_129": int(audit["criteria"]["target_len_ge_129"]),
                "has_gap_or_brace_or_bracket": int(audit["criteria"]["has_gap_or_brace_or_bracket"]),
            }
        )

    payload = {"rows": rows}
    _write_json(REPO_ROOT / "logs" / "taskform_replay_coverage_audit_20260309.json", payload)

    lines = [
        "# Taskform 轻量实验 2：replay25 / replay40 覆盖率审计（2026-03-09）",
        "",
        "| variant | ratio | train_rows_before | hard_rows | extra_rows_added | rows_after | hard_reweight_mult | extra_over_train_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['ratio']:.2f} | {row['train_rows_before']} | {row['hard_rows']} | {row['extra_rows_added']} | {row['rows_after']} | {row['hard_reweight_mult']:.2f} | {row['extra_over_train_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 难例覆盖口径（原 train 内的基数）",
            "",
            "| variant | chunk_total>=4 | target_len>=129 | gap/brace/bracket |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['chunk_total_ge_4']} | {row['target_len_ge_129']} | {row['has_gap_or_brace_or_bracket']} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- `replay40` 比 `replay25` 更激进，本质上是把当前 hard bucket 的权重再放大一档。",
            "- 这份表主要用于后面解释 `P2`：如果 `replay40` 更差，说明 hard-case 重加权过头；如果更好，则说明主问题确实集中在这些难例桶。"
        ]
    )
    _write_text(REPO_ROOT / "docs" / "taskform_replay_coverage_audit_2026-03-09.md", "\n".join(lines) + "\n")


def run_proxy(args: argparse.Namespace) -> None:
    audit_path = REPO_ROOT / "data" / "processed_taskform_proxymix_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    published = pd.read_csv(REPO_ROOT / "deep-past-initiative-machine-translation" / "published_texts.csv")
    train = pd.read_csv(REPO_ROOT / "data" / "processed_byt5_chunks_align_gc_cost14" / "train_proc.csv")

    existing_ids = set(train["parent_oare_id"].fillna(train["oare_id"]).astype(str).tolist())
    pool = published[["oare_id", "transliteration", "genre_label"]].copy()
    pool["transliteration"] = pool["transliteration"].fillna("").astype(str)
    pool = pool[pool["transliteration"].str.strip() != ""].reset_index(drop=True)
    pool = pool[~pool["oare_id"].astype(str).isin(existing_ids)].reset_index(drop=True)
    pool["score"] = [
        _score_proxy_row(text, genre)
        for text, genre in zip(pool["transliteration"].tolist(), pool["genre_label"].fillna("").astype(str).tolist())
    ]
    pool["word_count"] = pool["transliteration"].map(lambda x: len([w for w in str(x).split() if w]))
    pool = pool.sort_values(["score", "word_count"], ascending=[False, False]).reset_index(drop=True)
    pool = pool.head(int(audit["pool_size"])).reset_index(drop=True)

    genre_counts = Counter(pool["genre_label"].fillna("")).most_common(12)
    has_gap = float(pool["transliteration"].str.contains("<gap>", regex=False).mean() * 100.0)
    has_brace = float(pool["transliteration"].str.contains("{", regex=False).mean() * 100.0)
    has_sub = float(pool["transliteration"].str.contains(r"[₀₁₂₃₄₅₆₇₈₉ₓ]", regex=True).mean() * 100.0)
    payload = {
        "audit": audit,
        "pool_size": int(len(pool)),
        "word_count_quantiles": {
            "p50": float(pool["word_count"].quantile(0.50)),
            "p90": float(pool["word_count"].quantile(0.90)),
            "p95": float(pool["word_count"].quantile(0.95)),
        },
        "marker_rates_pct": {
            "has_gap": round(has_gap, 2),
            "has_brace": round(has_brace, 2),
            "has_subscript": round(has_sub, 2),
        },
        "top_genres": [{"genre_label": genre, "count": count} for genre, count in genre_counts],
    }
    _write_json(REPO_ROOT / "logs" / "taskform_proxy_mix_candidate_audit_20260309.json", payload)

    lines = [
        "# Taskform 轻量实验 3：proxy_mix 候选池审计（2026-03-09）",
        "",
        "## 总览",
        "",
        f"- filtered + ranked pool size: `{len(pool)}`",
        f"- base_train_rows: `{audit['base_train_rows']}`",
        f"- mix outputs: `{', '.join(sorted(audit['outputs'].keys()))}`",
        f"- word_count p50/p90/p95: `{payload['word_count_quantiles']['p50']:.1f} / {payload['word_count_quantiles']['p90']:.1f} / {payload['word_count_quantiles']['p95']:.1f}`",
        f"- marker rates (%): `gap={payload['marker_rates_pct']['has_gap']:.2f}, brace={payload['marker_rates_pct']['has_brace']:.2f}, subscript={payload['marker_rates_pct']['has_subscript']:.2f}`",
        "",
        "## 输出目录",
        "",
    ]
    for ratio_tag, meta in sorted(audit["outputs"].items()):
        lines.append(f"- `{ratio_tag}`: `{meta['rows_added']}` rows -> `{meta['out_dir']}`")
    lines.extend(["", "## Top genre_label", "", "| genre_label | count |", "| --- | ---: |"])
    for row in payload["top_genres"]:
        lines.append(f"| {row['genre_label'] or '(blank)'} | {row['count']} |")
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 这份审计用于判断 `P3` 的 proxy 样本是否真的偏向 hard bucket，而不是只是随便混入额外噪声。",
            "- 如果 marker rate 和 genre 分布明显偏向复杂样本，说明 `P3` 的失败更可能是任务不适配，而不是候选池完全失真。"
        ]
    )
    _write_text(REPO_ROOT / "docs" / "taskform_proxy_mix_candidate_audit_2026-03-09.md", "\n".join(lines) + "\n")


def run_winner(args: argparse.Namespace) -> None:
    bucket = json.loads((REPO_ROOT / "logs" / "winner_error_bucket_report_20260309.json").read_text(encoding="utf-8"))
    matched = json.loads(
        (
            REPO_ROOT
            / "runs"
            / "STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0"
            / "diagnostics"
            / "decode_grid_best_arch_probe_matched_baseline_anchor32_20260309.json"
        ).read_text(encoding="utf-8")
    )
    selected = [
        ("row", "ref_tok129_256"),
        ("row", "ref_tok257+"),
        ("row", "parent_chunks=2_3"),
        ("row", "parent_chunks=4_6"),
        ("row", "parent_chunks>=7"),
        ("parent", "ref_tok<=128"),
        ("parent", "ref_tok129_256"),
        ("parent", "ref_tok257_512"),
        ("parent", "ref_tok513+"),
        ("parent", "parent_chunks=2_3"),
        ("parent", "parent_chunks=4_6"),
        ("parent", "parent_chunks>=7"),
    ]
    rows = []
    for level, name in selected:
        source = bucket["row_buckets"] if level == "row" else bucket["parent_buckets"]
        entry = source.get(name, {})
        rows.append(
            {
                "level": level,
                "bucket": name,
                "n": int(entry.get("n", 0)),
                "geom": float(entry.get("geom", 0.0)),
                "short_pct": float(entry.get("short_pct", 0.0)),
                "cap_hit_pct": float(entry.get("cap_hit_pct", 0.0)),
                "repeat_pct": float(entry.get("repeat_pct", 0.0)),
                "matched_baseline_anchor32_geom": float(matched["eval_geom"]),
            }
        )

    payload = {
        "matched_baseline_anchor32": matched,
        "selected_buckets": rows,
    }
    _write_json(REPO_ROOT / "logs" / "winner_bucket_baseline_compare_20260309.json", payload)

    lines = [
        "# Taskform 轻量实验 4：winner 桶表 + matched baseline 参考（2026-03-09）",
        "",
        f"- matched baseline anchor32 geom: `{matched['eval_geom']:.4f}`",
        "",
        "| level | bucket | n | geom | short_pct | cap_hit_pct | repeat_pct | matched_baseline_anchor32_geom |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['level']} | {row['bucket']} | {row['n']} | {row['geom']:.4f} | {row['short_pct']:.2f} | {row['cap_hit_pct']:.2f} | {row['repeat_pct']:.2f} | {row['matched_baseline_anchor32_geom']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 这不是 taskform 结果表，而是后面 `P1/P2/P3` 做 matched baseline 对照时的参考面。",
            "- 重点看 `2-3 chunk` 与 `4+ chunk` 的断层，以及长 parent 桶的系统性退化。"
        ]
    )
    _write_text(REPO_ROOT / "docs" / "winner_bucket_baseline_compare_2026-03-09.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["parentpack", "replay", "proxy", "winner"])
    args = parser.parse_args()
    if args.task == "parentpack":
        run_parentpack(args)
    elif args.task == "replay":
        run_replay(args)
    elif args.task == "proxy":
        run_proxy(args)
    elif args.task == "winner":
        run_winner(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
