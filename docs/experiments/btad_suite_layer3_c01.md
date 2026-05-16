# BTAD rigor suite (layer3, coreset=0.01)

Status: **in progress**. This doc will be updated as runs complete.

## Goal

Evaluate PatchCore variants on BTAD with:
- no train/test leakage
- explicit operating-point calibration (threshold chosen on held-out nominal)
- reporting beyond AUROC (pixel + PRO + recall@fixed-FPR)

## Dataset

- BTAD root: `data/btad/`
- Train images: 1400 (nominal)
- Test images: 511

## Protocol

- Backbone: `wide_resnet50_2`
- Layers: `layer3`
- Image size: 256
- Memory bank coreset ratio: 0.01 (k-center greedy)
- Distance variants:
  - euclidean
  - cosine
  - PCA(256)+whiten + euclidean

### Threshold calibration

We do **not** set thresholds on the test set.

- Hold out `calib_fraction=0.2` of nominal train images.
- Compute image scores on that holdout.
- Choose threshold at `quantile = 1 - target_fpr`.
- Report recall/precision/FPR at that threshold on the test set.

## Runs

Outputs are written under:
- `outputs/rigor/btad_suite_layer3_c01/`

### 1) kNN euclidean (completed)

Command:
```bash
.venv/bin/python scripts/eval_btad_patchcore.py \
  --btad-root data/btad \
  --device mps --batch 16 --num-workers 0 \
  --coreset-ratio 0.01 --layers layer3 \
  --distance-metric euclidean \
  --target-fpr 0.01 --calib-fraction 0.2 \
  --out outputs/rigor/btad_suite_layer3_c01/knn_euclidean.json
```

Key results (see JSON for full detail):
- image AUROC: 0.8812
- pixel AUROC: 0.7872
- PRO AUC: 0.2127
- threshold calibrated to ~1% FPR on nominal holdout
  - achieved test FPR: 0.0133
  - recall: 0.6034

Timing notes:
- coreset selection dominates runtime (naive O(kN) farthest-first).

### 2) kNN cosine (completed)

Output:
- `outputs/rigor/btad_suite_layer3_c01/knn_cosine.json`

Key results:
- image AUROC: 0.8812
- pixel AUROC: 0.7878
- PRO AUC: 0.2134
- operational (calibrated on nominal holdout):
  - recall: 0.6034
  - achieved test FPR: 0.0133
  - threshold: 0.2181

### 3) PCA(256)+whiten + euclidean (completed)

Output:
- `outputs/rigor/btad_suite_layer3_c01/knn_pca256_euclidean.json`

Key results:
- image AUROC: 0.8761
- pixel AUROC: 0.7790
- PRO AUC: 0.2067
- operational (calibrated on nominal holdout):
  - recall: 0.5655
  - achieved test FPR: 0.0044
  - threshold: 23.9661

Timing highlight:
- PCA reduced coreset time dramatically vs naive k-center on raw embeddings.
  - coreset_s: 339s (PCA256) vs 1275s (euclidean baseline)

## Notes / caveats

- BTAD images are large BMPs; I/O is heavy.
- Our current coreset selection is naive; for larger banks we should implement batching and/or FAISS.
