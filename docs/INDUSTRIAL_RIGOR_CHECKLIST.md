# Industrial rigor checklist (vision anomaly detection)

This is a practical checklist for presenting and shipping an industrial vision anomaly detector (e.g., PatchCore).

## 0) Define the *operational* problem (before metrics)

- **What is a unit?** (image, part, region, time-window)
- **What is a defect?** (taxonomy, severity thresholds, “cosmetic vs functional”)
- **What is the action?** (stop line, rework, human inspection queue)
- **What is the cost model?**
  - False negative cost (missed defect)
  - False positive cost (extra inspection / scrap)
  - Localization error cost (mis-triage)

Deliverable: 1 slide / 1 page describing the real objective.

## 1) Data collection reality check

- Cameras: fixed vs handheld; single vs multi-site
- Lighting: controlled vs variable; known changes (shift schedules, bulb types)
- Pose: rigid fixturing vs variable pose; rotation/scale range
- Resolution requirements: smallest defect size in pixels
- “Nuisance factors”: reflections, dust, fingerprints, packaging, motion blur

Deliverable: a *dataset datasheet* (camera/lighting/pose, known nuisances).

## 2) Split strategy (avoid vision-specific leakage)

Leakage in vision is often unintentional. Use the split that matches deployment.

Choose at least one:
- **By physical unit** (serial number / part ID) across captures
- **By time** (train on earlier, test on later)
- **By batch/lot** (train lots A..C, test lot D)
- **By station/camera** (train camera 1, test camera 2)

Add a near-duplicate filter if needed:
- perceptual hash or embedding similarity to prevent “same photo twice” across splits

Deliverable: explicit split definition + checks.

## 3) Baseline ladder (confidence via discipline)

Build a ladder you can explain:
1) trivial baseline (e.g., SSIM / simple reconstruction error / mean template)
2) pretrained features + simple distance (kNN without coreset)
3) PatchCore full (coreset, aggregation, local smoothing)
4) optional: EfficientAD / distillation / learned head for latency

Deliverable: table/plot showing each step’s benefit and complexity.

## 4) Metrics that match QA reality

Report both “paper metrics” and “shop-floor metrics.”

Paper metrics:
- Image-level AUROC
- Pixel-level AUROC
- PRO (per-region overlap)

Operational metrics (often more important):
- **Recall at fixed FPR** (or fixed inspections/hour)
- **Precision at required recall** (if recall is the KPI)
- **Time-to-detect** (if scanning sequences)
- **Worst-case / tail risk** across product variants and shifts

Deliverable: a chosen operating point + how it’s selected.

## 5) Thresholding and calibration (first-class)

- Define a threshold selection procedure:
  - validation set separate from test
  - choose threshold for required FPR / inspection budget
- Track stability:
  - score distributions over time
  - per-variant thresholds vs global threshold trade-offs

Deliverable: thresholding protocol + plots.

## 6) Interpretability artifacts (how you debug)

Always keep:
- anomaly heatmaps overlaid on images
- top false positives / false negatives (with explanations)
- “what changed” comparisons (lighting/pose changes)

Deliverable: a recurring “model review” report.

## 7) Robustness plan

- Stress tests:
  - lighting perturbations
  - blur/noise
  - small misalignment
- Domain shift evaluation:
  - camera swap
  - new lot
  - new operator

Deliverable: explicit shift tests + pass/fail criteria.

## 8) Deployment considerations

- Latency and throughput targets
- Memory footprint (PatchCore memory bank)
- Update protocol:
  - how to add new nominal data
  - how to quarantine suspicious nominal data
- Monitoring:
  - drift detection on embeddings/scores
  - alerting thresholds

Deliverable: deployment runbook.

## 9) Communication pattern that lands with rigor

When presenting, say:
- “Here is the operational objective.”
- “Here is how we prevent leakage.”
- “Here is the baseline ladder.”
- “Here are the failure modes and our test plan.”

If you can do those 4 cleanly, you’ll read as confident even in a new domain.
