# Unsupervised defect memory bank (bootstrap "known failure modes")

Goal: bootstrap a "defect memory bank" even when you only have *anomalous images* and no curated defect taxonomy yet.

This is useful for:
- grouping recurring anomalies by visual similarity
- quickly creating a first-pass defect taxonomy by human review
- building a retrieval-backed defect lookup stage later

## What this is (and is not)

- **Is**: clustering of *high-scoring PatchCore patches* taken from anomalous images.
- **Is not**: ground-truth defect classification.

You should treat clusters as "candidate failure mode groups".
A human should review and assign names (e.g., "cracked capacitor", "missing solder", etc.).

## Script

- `scripts/build_defect_bank_unsupervised_mvtec.py`

Pipeline:
1) Load a fitted PatchCore model.
2) For each anomalous image, embed patches and compute patch anomaly scores.
3) Take top-k highest-scoring patches per image as candidate defect patches.
4) (Optional) apply the model's PCA transform if present.
5) k-means cluster those defect patch embeddings.
6) Write:
   - defect embeddings (`defect_embeddings.npy`)
   - patch metadata (`defect_metadata.json`)
   - cluster report with top exemplars (`defect_cluster_report.json`)

## Example (MVTec smoke dataset)

```bash
python3 scripts/build_defect_bank_unsupervised_mvtec.py \
  --mvtec-root data/mvtec_smoke \
  --category bottle \
  --model outputs/models/bottle \
  --out outputs/defect_bank/mvtec_bottle_unsup \
  --top-k-patches 5 \
  --clusters 8
```

## Dependencies

This script requires the same runtime deps as the PatchCore pipeline (torch/torchvision, etc.).
Run inside your project environment/venv.

## How to use output

- Start with `defect_cluster_report.json`.
- For each cluster, review the `top_exemplars` images/patch coordinates.
- Assign a human label.

Once labeled, you can:
- train a linear head (SVM/logistic) on patch embeddings
- build a defect retrieval index (kNN/IVF) for the lookup stage

## Caveats

- If the anomaly detector highlights nuisance regions (edges, specularities), your clusters may group nuisances rather than true defects.
- Tune `top-k-patches` and add filtering rules to avoid systematic false positives.
- Consider routing by product/component/camera before clustering, to avoid mixing modes.
