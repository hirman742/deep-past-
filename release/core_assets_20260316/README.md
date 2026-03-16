# Core Assets Backup 2026-03-16

This folder contains release archives for the current high-value working assets before repository cleanup.

## Classification Basis

- Scope counted from real untracked files, not folded `git status` directory entries.
- Current working-set snapshot:
  - 3 tracked files modified
  - 1046 real untracked files
  - 33255 real ignored files

## What Is Treated As Core

- Experiment source of truth:
  - modified tracked scripts
  - new scripts
  - new configs
- Docs tracked directly in `docs/`:
  - new analysis and planning markdown files kept as raw files for Git review
- Deliverable/export bundle:
  - `exports/public_model_r16_stepup_kaggle_20260316`
- Result summaries:
  - untracked `reports/` files with `.json`, `.csv`, `.md`, `.txt`, `.yaml`
- Heavy report model artifacts:
  - untracked `reports/` files with `.safetensors`, `.bin`
- Selected run snapshots:
  - ignored `runs/` files under `best_model/`
  - ignored `runs/` `resolved_config.yaml`
  - ignored `runs/` `run_summary.json`

## What Is Excluded

- `.venv-deeppast/`
- full ignored `runs/` trees outside the selected snapshot subset
- ignored logs, caches, and temp folders
- ignored processed data trees

These are mostly reproducible or operational artifacts and would explode backup size without adding much cleanup safety.

## Archives

The Git-tracked release format is `.tgz`. Local `.zip` copies are left untracked for convenience only.

- `core_source_bundle_20260316.tgz`
  - modified tracked scripts + untracked `configs/`, `scripts/`
- `core_export_bundle_20260316.tgz`
  - 26 files
  - current Kaggle/export deliverable bundle
- `core_report_summaries_20260316.tgz`
  - 802 files
  - report summaries, tables, manifests, narrative notes
- `core_report_model_artifacts_20260316.tgz`
  - 72 files
  - heavy adapter/model surgery artifacts stored under `reports/`
- `core_run_snapshots_20260316.tgz`
  - 434 files
  - selected `best_model` snapshots plus per-run resolved config and run summary

Checksums are recorded in `SHA256SUMS_20260316.txt`.
