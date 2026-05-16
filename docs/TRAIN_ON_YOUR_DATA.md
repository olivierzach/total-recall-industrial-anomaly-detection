# Training PatchCore on *your* nominal images (elite MLE guide)

This repo implements **PatchCore** as a *nominal-only* anomaly detector.

What “training” means here:
- You **do not** fine-tune a deep network.
- You **build a memory bank** of patch embeddings from *nominal (good)* images.
- At inference, anomalies are patches that are far from the nominal memory bank.

This is why PatchCore is attractive for industrial QA: you can deploy with **no defect labels**.

---

## 0) Inputs / outputs

### Inputs
- A directory of **nominal** images for one product + one camera setup.
  - Prefer consistent image geometry, lighting, and framing.
  - Include *acceptable* variation (tolerances) and common benign artifacts.

### Output model artifact
A model directory containing:
- `config.json`
- `memory_bank.npy` (coreset of nominal patch embeddings)
- `backbone_state.pt` (the exact backbone weights used to build the memory bank)

This artifact is what you deploy to score new images.

---

## 1) Minimum viable pipeline (what to run)

### Step 1 — Create a nominal dataset folder

Example:
```
/data/nominal_widgets_camA/
  0001.png
  0002.png
  ...
```

### Step 2 — Fit a memory bank on nominal images

Use the custom fitter:

```bash
python3 scripts/fit_nominal_patchcore.py \
  --nominal /data/nominal_widgets_camA \
  --out outputs/models/widgets_camA \
  --device cpu \
  --image-size 256 \
  --coreset-ratio 0.02
```

### Step 3 — Score new images (QA)

```bash
python3 scripts/score_images.py \
  --model outputs/models/widgets_camA \
  --images /data/new_widgets_camA \
  --out outputs/scores.jsonl \
  --save-maps outputs/maps

python3 scripts/viz_anomaly_maps.py \
  --images /data/new_widgets_camA \
  --maps outputs/maps \
  --out outputs/overlays
```

### Step 4 — Calibrate a threshold from nominal calibration images

Take a *separate* nominal set from a different day/shift (same product/camera), score it, then pick a threshold at your desired FPR:

```bash
python3 scripts/score_images.py --model outputs/models/widgets_camA --images /data/nominal_calib_camA --out outputs/calib.jsonl
python3 scripts/calibrate_threshold.py --scores outputs/calib.jsonl --target-fpr 0.001
```

Interpretation:
- `target-fpr=0.001` means “about 0.1% of nominal images are allowed to trigger.”
- Convert that into an inspection budget (alerts/day).

---

## 2) How many nominal images do we need?

There is no single number because PatchCore is essentially a **coverage problem** in embedding space.

You need enough nominal examples to cover:
- part-to-part manufacturing variation
- lighting drift
- camera alignment drift
- acceptable surface texture variation
- common benign artifacts

### Practical guidance (start here)

**Fast pilot:**
- 100–300 nominal images
- coreset_ratio ~ 0.01–0.05
- purpose: validate plumbing + sanity-check score distributions

**Usable QA baseline:**
- 1,000–5,000 nominal images
- at least 2–3 days/shifts worth if the process drifts

**High-stakes / high-recall:**
- 10,000+ nominal images (if variation is high)
- plus ongoing refresh / monitoring

### The key: diversity beats raw count
1000 near-identical images does less than 200 that span the real nuisance factors.

### What to measure to know if you have enough nominal data
On a held-out nominal set (different day/shift):
- distribution of scores (median, p99, max)
- stability across time buckets
- top false positives (visual inspection): are they benign artifacts you should include?

If p99/max explode on the held-out nominal set, you are under-covering nominal variation.

---

## 3) Recommended data splits (production-minded)

Even without defects, you should structure data like an MLE:

- `nominal_train`: used to build the memory bank
- `nominal_calib`: used to set threshold / FPR
- `nominal_monitor`: used for drift monitoring over time

If you have defects later:
- keep a `defect_eval` set for measuring recall (do not use for thresholding).

---

## 4) Common failure modes and fixes

### False positives (benign but “rare”)
Symptoms:
- high scores on nominal images
- anomaly maps highlight reflections, edges, dust

Fixes:
- add those patterns to nominal_train (or a separate “benign artifacts” nominal set)
- consider a slightly larger image size or different backbone layers
- increase coreset_ratio (more coverage)

### False negatives (subtle defects)
Symptoms:
- anomalies not scored high

Fixes:
- choose layers with more texture sensitivity (often earlier/mid layers)
- increase image_size
- reduce aggressive coreset subsampling (increase memory)

### Domain shift (camera/lighting changes)
Fixes:
- treat each camera/product as separate model, unless you have strong invariance
- add per-camera nominal calibration and thresholds

---

## 5) Deployment guidance

### What to store
- model dir (`config.json`, `memory_bank.npy`)
- threshold(s) per product/camera
- logs: scores + anomaly maps for triage

### How to alert
Use operational metrics:
- fixed FPR on nominal calibration
- recall tracked on known defect examples

### Monitoring
- daily nominal score distribution drift
- alert rate drift

---

## 6) Advanced improvements (optional)

Once the baseline works:
- add FAISS for faster NN
- swap backbone to a ViT and compare (keep same pipeline)
- implement nearest-neighbor patch retrieval for explanations
- add PRO metric for localization evaluation (on datasets with masks)
