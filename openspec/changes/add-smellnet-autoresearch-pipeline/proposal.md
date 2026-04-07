# Change: Add A SmellNet Autoresearch Pipeline

## Why
The current repository exposes the paper-era training code and run logs, but it does not ship a reproducible, inference-ready training harness for autonomous iteration. The best paper-matching run is only recorded in JSONL metadata, and the repo does not persist the preprocessing state that Raspberry Pi inference needs.

## What Changes
- Add a SmellNet-specific autoresearch harness modeled after `autoresearch-macos`, but grounded in SmellNet's six-channel paper-faithful preprocessing path.
- Save inference-ready artifacts alongside checkpoints, including channel ordering, scaler statistics, label ordering, and model configuration.
- Document the fixed baseline and artifact contract through OpenSpec so future optimization work does not silently drift away from the deployable path.

## Impact
- Affected specs: `autoresearch-harness`, `inference-artifacts`
- Affected code:
  - `autoresearch_smellnet/prepare.py`
  - `autoresearch_smellnet/train.py`
  - `autoresearch_smellnet/program.md`
  - `autoresearch_smellnet/README.md`
  - `openspec/project.md`
