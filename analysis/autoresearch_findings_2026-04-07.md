# SmellNet Autoresearch Findings

Date: 2026-04-07

This note captures the main findings from the SmellNet baseline audit and the first autoresearch runs. It is meant as a working reference, not a paper-ready summary.

## 1. Exact-upstream benchmark contract

The benchmark-faithful track uses:

- Dataset: `SMELLNET-BASE`
- Split: official `data/offline_training` vs `data/offline_testing`
- Input channels, in order: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`
- Dropped channels: `Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`, `Altitude`
- Preprocessing: subtract first row, `diff(periods=25)`, sliding windows
- Window size: `100`
- Stride: `50`
- Standardization: fit on training windows only
- Task for the benchmark track: single-model, sensor-only, window-level classification

The processed training tensor shape is effectively `(batch, 100, 6)`.

## 2. File vs window

- A `file` is one full recording CSV for one smell trial.
- A `window` is one fixed-length sliding slice cut from a single preprocessed file.

Important consequence:

- `window-level` evaluation asks the model to classify each local chunk independently.
- `file-level` evaluation runs the model on multiple windows from the same file and aggregates them into one final file prediction.

This means file-level numbers are not directly comparable to benchmark window-level numbers. File-level evaluation is more realistic if deployment predicts once at the end of a recording, but it says less about short-horizon or streaming prediction quality.

## 3. Why grouped file-level validation makes sense

Windows from the same CSV are highly correlated. If train and validation are split at the window level, windows from the same source file can leak across splits. Grouped file-level validation avoids that by holding out whole files.

This is methodologically stronger and should be kept even if the primary metric remains window-level.

## 4. Validation locking and ensembles

The old public `SmellNet` baseline flow is simple:

- train on `offline_training`
- evaluate on `offline_testing`
- no explicit validation split
- no file-level aggregation in the main benchmark path
- no ensemble

The improved protocol discussed later is:

- carve grouped validation files from `offline_training`
- choose checkpoint on validation only
- choose any file-level aggregator on validation only
- choose any ensemble members on validation only
- touch `offline_testing` once at the end

That is better experimental hygiene. The main caution is that file-level and ensemble results should not be presented as if they were the same task as the original single-model window-level benchmark.

## 5. Original repo checkpoint status

The original repo contains the training code and run logs, but not the exact paper-matching checkpoint artifact for the best logged contrastive transformer run.

The logged best paper-like run is:

- model: `transformer`
- contrastive: `on`
- gradient period: `25`
- window size: `100`
- batch size: `32`
- learning rate: `0.0003`
- Top-1: `58.5657`

The run metadata points at a checkpoint path, but the repo snapshot does not include that file, and the checked-in training code logs checkpoint paths without actually calling `torch.save(...)` in the main runner.

## 6. Autoresearch harness added here

We added a local benchmark-faithful autoresearch harness under `autoresearch_smellnet/` with three main entry points:

- `train.py`: exact-upstream baseline trainer
- `search.py`: phase-1 validation-locked search
- `search_phase2.py`: phase-2 promoted search with a larger hyperparameter surface

We also added an OpenSpec change under `openspec/changes/add-smellnet-autoresearch-pipeline/`.

## 7. Baseline and search results so far

### 7.1 Exact-upstream smoke baseline

Single baseline run, 5 epochs:

- Test Top-1: `52.99`
- Test Top-5: `89.24`

This was mainly a smoke check for the new harness and preprocessing contract.

### 7.2 Phase 1 search

Phase 1 search settings:

- Track: exact-upstream
- Model family: transformer only
- Contrastive learning: off
- Device: `mps`
- Budget: `3h`
- Validation: grouped, `1` held-out file per class
- Search objective: validation window-level `acc@1`
- Test: touched once at the end

Outcome:

- Trials completed: `127`
- Best validation config:
  - `model_dim = 512`
  - `num_heads = 8`
  - `num_layers = 5`
  - `dropout = 0.0`
  - `lr = 0.0001`
- Best validation metrics:
  - Top-1: `54.71`
  - Top-5: `88.18`
- Final test metrics:
  - Top-1: `56.57`
  - Top-5: `87.65`

Interpretation:

- The exact-upstream transformer can be pushed into the original paper's sensor-only performance band with honest validation-locked search.
- Naive hyperparameter search alone does not appear to open a large new gap beyond the original benchmark.

### 7.3 Phase 2 search

Phase 2 search adds:

- two-stage search
- promotion from cheap stage 1 to longer stage 2
- rotated grouped validation folds
- extra search knobs:
  - `weight_decay`
  - `label_smoothing`
  - `warmup_ratio`
  - plus model size/depth/dropout/LR

Phase 2 run settings:

- Device: `mps`
- Budget: `2h`
- Stage 1 epochs: `6`
- Stage 2 epochs: `20`
- Stage 2 fold count: `3`

Outcome:

- Stage 1 trials completed: `14`
- Stage 2 candidates evaluated: `1`
- Best promoted config:
  - `model_dim = 512`
  - `num_heads = 8`
  - `num_layers = 6`
  - `dropout = 0.05`
  - `lr = 0.0003`
  - `weight_decay = 0.0`
  - `label_smoothing = 0.0`
  - `warmup_ratio = 0.0`
- Aggregate stage-2 validation metrics:
  - Top-1: `53.85`
  - Top-5: `85.94`
- Final single test evaluation:
  - Top-1: `57.97`
  - Top-5: `88.05`

Interpretation:

- Phase 2 improved over phase 1 on Top-1 by about `+1.39` points.
- The best exact-upstream single-model score currently tracked in this repo is `57.97` Top-1.
- The phase-2 run did not need nonzero weight decay, label smoothing, or warmup to win under this budget.

## 8. What this does and does not show

What it shows:

- The benchmark-faithful data path is working.
- Grouped validation plus validation-locked search is viable.
- Exact-upstream single-model transformer performance is reproducible and can be improved modestly with search.

What it does not show:

- It does not prove that file-level aggregation is a good default for a fast or streaming product.
- It does not prove that short-window inference will perform as well as full-file inference.
- It does not justify comparing file-level ensemble results directly to the original benchmark window-level single-model numbers.

## 9. Recommended next measurement, not yet implemented

Before leaning into file-level metrics for deployment, add prefix evaluation for the same trained models:

- first window only
- first 2 windows
- first 3 windows
- full file

That will tell us whether the stronger file-level protocols still help when prediction must happen early.

## 10. Current working take

- Keep grouped file-level validation.
- Keep validation-locked checkpoint selection.
- Keep exact-upstream single-model window-level results as the clean benchmark anchor.
- Treat file-level aggregation and heterogeneous ensembles as a separate, production-oriented or max-accuracy track.
- Do not assume that a better full-file result implies a better early or streaming detector.
