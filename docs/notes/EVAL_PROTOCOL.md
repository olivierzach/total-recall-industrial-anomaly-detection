# Evaluation protocol (industrial anomaly detection)

This doc specifies a default evaluation protocol for PatchCore-style models in an industrial QA setting.

## 1) Dataset splits

### Recommended primary split (choose what matches deployment)
- **Time-based**: train on earliest window, validate on next, test on latest.
- **Part-based**: split by serial/part ID (all captures of the same unit stay in one split).
- **Batch/lot-based**: train on lots A..C, test on lot D.
- **Camera/site-based**: train on camera 1, test on camera 2.

### Leakage checks
- Near-duplicate detection across splits (recommended if multi-capture):
  - perceptual hash (pHash) or embedding similarity
  - reject pairs above a similarity threshold

## 2) Metrics

### Research-standard metrics
- Image AUROC
- Pixel AUROC
- PRO (Per-Region Overlap), with explicit implementation details (kernel size, thresholds)

### Deployment metrics
- **Recall @ fixed FPR** (pick FPR from inspection budget)
- **Precision @ required recall** (if recall is the KPI)
- Throughput: images/sec and ms/image on target hardware

## 3) Threshold selection

A model is not deployable without a threshold procedure.

- Use a validation set that matches deployment shifts.
- Choose threshold to satisfy one of:
  - fixed FPR target
  - fixed “inspections per hour” target

Report:
- selected threshold
- sensitivity of results to threshold drift

## 4) Confidence intervals / stability

- Report variability across:
  - random seeds (if any)
  - different nominal subsets
  - different days/batches

Simple practice:
- bootstrap confidence intervals on image-level metrics

## 5) Error analysis requirements

For every run:
- top 20 false positives (with heatmaps)
- top 20 false negatives (with heatmaps)
- cluster FP modes (lighting, edges, specularities, etc.)

## 6) Shift tests

Maintain a shift suite:
- lighting perturbations
- small affine misalignment
- blur/noise
- camera swap (if available)

## 7) Reporting template

Every report should include:
- dataset version/hash
- split definition
- model config
- thresholding rule
- metrics + operational point
- qualitative panel (TP/FP/FN heatmaps)
