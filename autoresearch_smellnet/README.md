# SmellNet Autoresearch

This directory adapts the `autoresearch-macos` workflow shape to SmellNet.

The first baseline is intentionally narrow:
- paper-faithful six-channel sensor preprocessing
- supervised transformer classifier
- explicit artifact saving for Raspberry Pi inference

The baseline writes:
- `checkpoint.pt`
- `metrics.json`
- `preprocessing.json`
- `labels.json`

Run from the repo root:

```bash
python autoresearch_smellnet/train.py \
  --train-dir data/offline_training \
  --test-dir data/offline_testing \
  --output-dir autoresearch_runs/baseline
```
