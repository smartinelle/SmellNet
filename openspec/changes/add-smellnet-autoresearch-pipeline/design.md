## Context
`autoresearch-macos` provides a useful workflow shape, but its implementation is hardcoded for language-model pretraining: tokenized text data, BPE tokenizers, GPT-style training, and `val_bpb` as the optimization target. SmellNet needs a different fixed harness: six-channel sensor windows, paper-faithful preprocessing, classification metrics, and artifact persistence for Raspberry Pi inference.

The current SmellNet codebase also has a reproducibility gap: `models/run.py` logs checkpoint paths into run metadata, but it does not persist those checkpoints or the associated preprocessing state.

## Goals
- Provide a fixed SmellNet baseline harness that autonomous search can optimize against.
- Keep the first implementation paper-faithful to the best supervised sensor-classification path.
- Persist all inference-critical state as explicit files.

## Non-Goals
- Rebuild the full cross-modal contrastive training stack in the first iteration.
- Match every historical training utility under `models/`.
- Optimize Raspberry Pi latency in the first scaffold.

## Decisions

### Fixed baseline
The initial harness uses the paper-matching six-channel setup:
- channels: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`
- dropped columns: `Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`, `Altitude`
- preprocessing: subtract first row, `diff(periods=25)`, `window_size=100`, `stride=50`
- model: transformer classifier

### Artifact contract
Every baseline run should save:
- model checkpoint
- metrics JSON
- preprocessing JSON
- label names JSON

`preprocessing.json` is the stable handoff contract for Raspberry Pi inference. It must include:
- input channel ordering
- dropped columns
- gradient period
- window size and stride
- scaler mean and std
- label names in encoded order
- model architecture parameters

### Dependency boundary
The new harness is intentionally self-contained. It reuses the transformer model definition from `models/models.py`, but it does not import the existing `models/train.py` or `models/load_data.py` modules because those modules rely on path-sensitive imports and were not designed to persist deployment artifacts.
