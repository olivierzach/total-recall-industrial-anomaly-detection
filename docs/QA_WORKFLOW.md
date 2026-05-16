# QA workflow: training on nominal images, scoring new builds, and iterating on errors

This doc answers the practical question:

> "For prime time, I want to train on my own *nominal* images and then look up errors. Is that the right path?"

Yes: **PatchCore is designed exactly for this cold-start / nominal-only workflow.**

It is best thought of as a *retrieval-style* anomaly detector:
- learn a representation using a pretrained backbone (ImageNet)
- build a reference set (memory bank) from nominal data
- score new images by how far their patch embeddings are from nominal patches

---

## 1) What PatchCore is (operationally)

PatchCore training does **not** mean training a deep network end-to-end.

Instead, it builds a **memory bank** of patch embeddings from nominal data:
1. Extract patch embeddings from nominal images using a pretrained CNN.
2. Optionally subsample via a coreset (k-Center Greedy) so memory fits and is representative.
3. At inference time, compute nearest-neighbor distances from test patches to the memory bank.

This fits industrial QA because:
- you often have many good/nominal examples
- true defects are rare and diverse
- you want localization heatmaps (where it looks wrong)

---

## 2) Prime-time workflow (recommended)

### Step A — Curate a nominal reference set

Nominal images should represent:
- the full range of *acceptable* variation: lighting, camera position, product tolerances
- known benign artifacts (dust, mild glare) if they appear in production

Avoid:
- images with true defects (or as many as possible)
- distribution shifts that won't exist at inference time (e.g., temporary test rig)

**Practical tip:** treat nominal curation as part of the model.
PatchCore's memory bank is effectively a learned definition of "normal".

### Step B — Fit the model (memory bank)

Run `fit_*` to build the coreset memory bank.

This repo writes a model directory containing:
- `config.json`
- `memory_bank.npy`

This is the artifact you deploy.

### Step C — Score new images (QA inference)

Use `score_images.py` to score:
- a batch of new parts
- a new production day
- a new supplier run

Outputs a JSONL with per-image anomaly scores.

### Step D — Thresholding for operations

AUC is not the operational target. In QA we care about:
- **recall at fixed false positive rate** (or fixed inspection budget)
- stable behavior under drift

Operational approach:
- choose a threshold based on validation nominal sets (or a controlled defect set if available)
- periodically recalibrate thresholds as nominal distribution shifts

### Step E — Look up errors, diagnose, and iterate

PatchCore is good for "poking" because:
- high scores can be localized (anomaly map)
- you can retrieve nearest nominal patches to see what it matched

Typical failure modes:
1) **False positives due to benign artifacts** (glare, dust)
2) **False negatives on subtle defects** (small scratches)
3) **Domain shift** (lighting/camera changes)

Mitigations:
- Expand nominal reference set to include benign artifacts.
- Adjust backbone/layers for finer texture sensitivity.
- Tune coreset ratio (bigger memory can improve recall but costs RAM/time).
- Add simple preprocessing (illumination normalization) if stable.

---

## 3) What you should log for "prime time"

At minimum for each image:
- image score
- top-k patch scores
- anomaly map (downsampled or compressed)
- metadata: product, line, camera, timestamp

This enables error triage and drift monitoring.

---

## 4) Common misconceptions

### "Do we need a pretrained PatchCore checkpoint?"
No.
PatchCore uses a pretrained backbone (ImageNet) plus a nominal memory bank built from *your* data.

### "Is it supervised?"
Not in the classic sense.
It's nominal-only (one-class) with optional defect labels for evaluation.

### "Will it work on a new product?"
Yes, if nominal images capture the acceptable variability. Expect some iteration.

---

## 5) Recommended next additions to this repo

To support a real QA workflow, we should add:
- anomaly map visualization + overlays
- retrieval: show nearest nominal patches for top anomalies
- operational metrics: recall@FPR, PR curves, expected inspections
- dataset abstractions for "your" data (folder of nominal, folder of scored)

