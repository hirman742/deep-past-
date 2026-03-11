# Archive Manifest (2026-03-11)

## Scope

As of 2026-03-11, the local workspace contains a winner freeze chain that is not yet represented on `origin/main` (`https://github.com/hirman742/deep-past-.git`).

The remote baseline checked for this manifest is `origin/main` at commit `836c14d`.

This manifest splits the workspace into four layers:

1. `repo_sync_20260311.lst`
2. `winner_core_20260311.lst`
3. `research_tail_20260311.lst`
4. `legacy_taskform_history_20260311.lst`

The goal is to keep Git history readable while preserving the non-regenerable winner assets outside normal source control history.

## Bundle Map

### `repo_sync_20260311.lst`

Use this list for normal Git commits and pushes. It contains:

- code and config changes under `scripts/`, `configs/`, `cleaning/`, and `artifacts/`
- winner and research docs under `docs/`
- compact `reports/` directories that explain the freeze and the rejected branches
- the small external asset `data/external/oracc_parallel.csv`
- the archive helper script and the archive manifests themselves

### `winner_core_20260311.lst`

Build this archive as the primary frozen release. It contains:

- the incumbent checkpoint and diagnostics
- the promoted retrieval W-lite checkpoint and diagnostics
- the retrieval processed dataset that W-lite depends on
- the full A2 freeze report chain, bridge report, and decision docs
- the code and raw inputs needed to interpret the frozen assets

Recommended output name:

```text
release/deep_past_winner_core_20260311.tgz
```

### `research_tail_20260311.lst`

Build this archive as the secondary research bundle. It contains:

- A1 external / continue-on-wlite evidence
- pseudo-target and TAPT-negative evidence
- replay and combo probe evidence
- revisit artifacts and research-only logs

Recommended output name:

```text
release/deep_past_research_tail_20260311.tgz
```

### `legacy_taskform_history_20260311.lst`

Build this archive as the legacy evidence bundle for pre-freeze taskform lines. It contains:

- archived taskform and phase12 report directories
- DAN1-era summary artifacts and compact data sidecars
- the `taskform_tapt_fair_smokecheck` boundary package
- the older docs that explain why these lines were superseded

Recommended output name:

```text
release/deep_past_legacy_taskform_history_20260311.tgz
```

## Explicit Exclusions

Do not add these to the new archives:

- `deep_past_rescue_20260306_191331.tgz`
- `.git/`
- `.cache/`
- `.venv-deeppast/`
- `_cloud_reparse/`
- the full `runs/` tree
- the full `data/` tree
- `pre_knowledge/`

## Execution

### 1. Stage the Git-sync layer

Review the list first, then stage it:

```bash
while IFS= read -r path; do
  [ -z "$path" ] && continue
  case "$path" in
    \#*) continue ;;
  esac
  git add -- "$path"
done < manifests/archive/repo_sync_20260311.lst
```

### 2. Validate manifests without building archives

```bash
scripts/build_archive_from_manifest.sh --check-only \
  manifests/archive/winner_core_20260311.lst \
  release/deep_past_winner_core_20260311.tgz

scripts/build_archive_from_manifest.sh --check-only \
  manifests/archive/research_tail_20260311.lst \
  release/deep_past_research_tail_20260311.tgz

scripts/build_archive_from_manifest.sh --check-only \
  manifests/archive/legacy_taskform_history_20260311.lst \
  release/deep_past_legacy_taskform_history_20260311.tgz
```

### 3. Build the archives

```bash
scripts/build_archive_from_manifest.sh \
  manifests/archive/winner_core_20260311.lst \
  release/deep_past_winner_core_20260311.tgz

scripts/build_archive_from_manifest.sh \
  manifests/archive/research_tail_20260311.lst \
  release/deep_past_research_tail_20260311.tgz

scripts/build_archive_from_manifest.sh \
  manifests/archive/legacy_taskform_history_20260311.lst \
  release/deep_past_legacy_taskform_history_20260311.tgz

sha256sum \
  release/deep_past_winner_core_20260311.tgz \
  release/deep_past_research_tail_20260311.tgz \
  release/deep_past_legacy_taskform_history_20260311.tgz \
  > release/SHA256SUMS_20260311.txt
```

### 4. Publish

Preferred publication path:

- push the `repo_sync` layer through normal Git history
- upload the three `.tgz` files and `SHA256SUMS_20260311.txt` as GitHub Release assets

Fallback path:

- if you need the archives in Git history, add new LFS tracking rules before commit

## Recovery Order

When restoring from scratch, read assets in this order:

1. `docs/taskform_winner_stage_report_2026-03-10.md`
2. `docs/next_step_displine_winner_2026-03-10.md`
3. `reports/taskform_winner_a2_freeze_20260310/summary.json`
4. `reports/taskform_winner_a2_support_20260310/summary.json`
5. `runs/STEER_S4_CONTINUE_BS24_LEN640_SEG5_fold0/run_summary.json`
6. `runs/TASKFORM_WINNER_A2_RETRIEVAL_TOP1_WLITE_20260310_fold0/run_summary.json`

If you need the earlier taskform history after that, unpack `deep_past_legacy_taskform_history_20260311.tgz` and start from:

1. `docs/taskform_experiment_report_2026-03-10.md`
2. `docs/next-step_taskform_discipline_2026-03-10.md`
3. `reports/taskform_phase12/phase12_l2_l3_summary.json`
