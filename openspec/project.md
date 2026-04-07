# Project Context

## Purpose
SmellNet is a multimodal smell-recognition research repository built around gas-sensor time series, GC-MS descriptors, and classification / contrastive-learning experiments.

## Tech Stack
- Python research code under `models/` and `preprocessing/`
- PyTorch for models and training
- Pandas / NumPy / scikit-learn for preprocessing
- Arduino + serial collection scripts under `Arduino/` and `data_collection/`
- OpenSpec for spec-driven planning under `openspec/`

## Project Conventions

### Data and preprocessing
- The paper-matching sensor path uses six channels in this fixed order: `NO2`, `C2H5OH`, `VOC`, `CO`, `Alcohol`, `LPG`.
- Raw collection data may include `timestamp`, `State`, and additional environmental channels, but the paper-matching training path drops those before model input.
- For the best reported transformer run, preprocessing is: subtract first row, drop unused channels, `diff(periods=25)`, sliding windows of size `100` with stride `50`, and train-only `StandardScaler` fitting.

### Research workflow
- Preserve reproducibility: training code must emit all artifacts required for offline evaluation and Raspberry Pi inference.
- Prefer explicit saved artifacts over in-memory preprocessing state.
- Keep a paper-faithful baseline separate from exploratory variations.

### Deployment goals
- Inference artifacts must be CPU-friendly and loadable without reconstructing hidden preprocessing state.
- Saved outputs should include checkpoint weights, scaler statistics, label ordering, and model configuration.
