# SmellNet Autoresearch Program

## Goal
Improve SmellNet sensor classification while preserving an inference-ready artifact contract for Raspberry Pi deployment.

## Fixed baseline
- Dataset: `data/offline_training` and `data/offline_testing`
- Track: `exact-upstream`
- Input channels: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`
- Preprocessing: subtract first row, drop non-model columns, `diff(periods=25)`, `window_size=100`, `stride=50`, train-only standardization
- Baseline model: transformer classifier
- Contrastive learning: disabled
- Primary score: held-out window-level `acc@1`

## Optimization target
Primary objective:
- maximize held-out `acc@1`

Secondary objectives:
- preserve `acc@5`
- reduce CPU inference cost where possible
- never break the saved artifact contract required for Raspberry Pi inference

## Guardrails
- `prepare.py` defines the fixed data contract and artifact format. Keep it stable unless you are intentionally changing the contract.
- Always save checkpoint, metrics, preprocessing metadata, and label ordering.
- Do not silently change channel order, differencing period, or label ordering.
- Keep paper-faithful baselines comparable before introducing exploratory variants.
- Treat file-level aggregation and ensembling as later-stage evaluation features, not the first optimization target.

## Phase 2
- Keep the model family fixed to the transformer track.
- Use a promoted two-stage search instead of a flat random loop.
- Search optimization-policy knobs before adding new architectures:
  `lr`, `weight_decay`, `label_smoothing`, `warmup_ratio`, `model_dim`, `num_heads`, `num_layers`, `dropout`
- Reduce validation noise with rotated grouped validation folds.

## Contrastive track
- Use the same exact-upstream sensor preprocessing contract as the supervised baseline.
- Load `data/gcms_dataframe.csv` as a fixed 50-class GC-MS bank aligned to the sensor label order.
- Train a transformer sensor encoder and GC-MS MLP encoder with cross-modal contrastive loss.
- Keep grouped validation and validation-locked final test evaluation.
- Treat contrastive search as a separate track from the supervised benchmark anchor.
