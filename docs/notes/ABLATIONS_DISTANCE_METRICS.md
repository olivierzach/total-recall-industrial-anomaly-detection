# Ablation: distance metrics and whitening for PatchCore

Motivation: kNN distance in embedding space can be distorted by correlated/redundant embedding dimensions. This ablation compares:

- kNN with **Euclidean** distance
- kNN with **Cosine** distance
- kNN with **PCA-whitened** embeddings + Euclidean distance

## What changed in this repo

`PatchCoreConfig` now supports:
- `distance_metric`: `euclidean|cosine`
- `pca_dim`: `0` disables, otherwise projects to `pca_dim`
- `pca_whiten`: whether to whiten along principal components

The fitted PCA state (if enabled) is saved alongside the model in:
- `pca_state.npz`
- `pca_state.json`

## Run on MVTec

```bash
python3 scripts/ablate_distance_metrics_mvtec.py \
  --mvtec-root /path/to/mvtec_ad \
  --category bottle \
  --device cpu \
  --coreset-ratio 0.1 \
  --pca-dim 256 \
  --out outputs/ablations/distance_mvtec_bottle.json
```

This produces per-variant JSON result files next to the output, plus a combined JSON.

## Notes / interpretation

- If `l2_normalize=True` and you use Euclidean distance, Euclidean and cosine become monotonic-equivalent for unit vectors.
  - In that regime, any observed differences between `euclidean` and `cosine` are likely coming from implementation details or non-unit vectors.
- PCA whitening is expected to matter most when embedding dimensions are strongly correlated and the nominal manifold is anisotropic.

Next step: add a `recall@fixed-FPR` plot + latency/memory summaries to match industrial operating points.
