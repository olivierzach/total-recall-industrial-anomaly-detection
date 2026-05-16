# PLAN — PatchCore (“Towards Total Recall…”) implementation + experiments

## 0) Guiding constraint
We want a **buildable QA-style system**: high recall, controllable false positives, interpretable localization, reasonable latency, and robust behavior under cold start (nominal-only training).

## 1) Paper summary (operational)
PatchCore:
- Use a pretrained backbone (e.g., WideResNet50) as a feature extractor.
- Form patch-level embeddings (multi-scale / mid-level features).
- Build a **memory bank** of embeddings from nominal images.
- Reduce bank via **coreset sampling** (k-Center Greedy in embedding space).
- Score new images by nearest-neighbor distances to the memory bank:
  - image score = max (or aggregated) patch anomaly score
  - pixel score = upsampled patch anomaly map

## 2) Milestones

### Current status (updated)
- Implemented a self-owned PatchCore MVP and a minimal MVTec loader.
- Added dataset unpack helper, model save/load, and a QA-style scoring script.
- Added pytest suite for core utilities.
- Next: pixel-level anomaly maps + PRO metric + second dataset pipeline.
- Added QA workflow doc: `docs/QA_WORKFLOW.md`.
- Added ViT backbone scaffolding doc: `docs/VIT.md`.

### Progress log (BTAD)
- ✅ BTAD downloaded + extracted automatically.
- ✅ ViT-B/16 baseline ran end-to-end on BTAD (coreset_ratio=0.0005): image AUROC ~0.8285, pixel AUROC ~0.7427.
- ⏳ WRN50-2 BTAD run was interrupted (SIGTERM); will rerun with better progress logging + fast-smoke mode.

### M1 — Minimal baseline
- [x] Dataset loader: MVTec AD (train=nominal, test=mixed)
- [x] Backbone feature extraction (torchvision WRN50 / ResNet)
- [x] Patch embedding extraction (layer selection + concatenation)
- [x] Memory bank construction
- [x] Scoring (kNN; FAISS optional later)
- [~] Metrics: image AUROC (done), pixel AUROC + PRO (next)
- [ ] Deterministic run config + seeds (next)
- [x] QA pipeline: fit → save model → score arbitrary images
- [x] Test suite for core components (coreset/embedding/loader)

### M2 — Faithful reproduction knobs
- [x] Implement k-Center Greedy coreset (bank subsampling)
- [ ] Add Gaussian smoothing / normalization as needed
- [ ] Compare layer choices, embedding dims, patch strides
- [ ] Validate against reported numbers (within tolerance)

### M3 — Industrial QA extensions
- [ ] Thresholding for high recall at fixed FPR
- [ ] Open-set / shift diagnostics (embedding distance stats)
- [ ] Fast incremental updates (add nominal points, update coreset)
- [ ] Latency/memory benchmarks

### M4 — Research playground
- [ ] Compare to PaDiM, SPADE, STFPM, FastFlow, Reverse Distillation, EfficientAD
- [ ] Few-shot defect adaptation
- [ ] Synthetic anomaly augmentation experiments

## 3) Repo layout (proposed)
- `src/patchcore/` — implementation
- `src/data/` — datasets + transforms
- `scripts/` — train/eval utilities
- `experiments/` — YAML configs + outputs
- `docs/` — notes, ablations, references

## 4) Implementation details to decide (early)
- Backbones: WRN50-2 baseline, plus a smaller ResNet18 for speed
- Feature layers: choose mid-level feature maps; confirm shapes
- Patch embedding: flatten H×W locations, concat channels across layers
- NN backend: start with brute-force torch / sklearn; add FAISS for speed
- Coreset ratio: default 1–10% with k-Center Greedy

## 5) Deliverables
- Reproducible evaluation script
- One-command download/prep for datasets (where licensing allows)
- A results table generator
- Plotting utilities (anomaly map overlays)

