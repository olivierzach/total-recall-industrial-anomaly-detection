# Core reading for *Towards Total Recall in Industrial Anomaly Detection* (PatchCore)

This is a curated, “minimum sufficient” reading list to understand and *apply* PatchCore-style industrial anomaly detection.

## 0) The primary source (read first)

- **Roth et al. (CVPR 2022)** — *Towards Total Recall in Industrial Anomaly Detection* (**PatchCore**)
  - arXiv: https://arxiv.org/abs/2106.08265
  - CVF PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
- **Official code** (reference implementation + hyperparams):
  - https://github.com/amazon-science/patchcore-inspection

What to extract from PatchCore:
- The **patch embedding** idea (mid-level features from pretrained backbones).
- The **memory bank** of nominal patch features.
- **Coreset subsampling** to reduce memory and speed up kNN.
- The **scoring pipeline**: kNN distance → image score; patchwise distances → anomaly map.
- Practical knobs: backbone layer(s), patch size, NN count `k`, sampling rate, preprocessing.

## 1) Baseline “family tree” (know what PatchCore is competing with)

These establish the canonical *nominal-only* setting and common evaluation conventions.

### Patch embedding + nearest-neighbor / distribution modeling
- **SPADE** (Cohen & Hoshen, 2020): patch-level nearest-neighbor / distribution estimation.
- **PaDiM** (Defard et al., ICPR 2021): multivariate Gaussian modeling over patch embeddings.

Why read: these clarify why patch embeddings work, and what PatchCore changes (memory bank + coreset + local aggregation).

### Reconstruction / distillation baselines
- **STFPM** (Wang et al., ICCV 2021): student–teacher feature pyramid matching.
- **DRAEM** (Zavrtanik et al., ICCV 2021): synthetic anomalies + reconstruction.
- **Reverse Distillation** (Deng & Li, CVPR 2022): teacher → student distillation for anomaly.

Why read: many production-ish settings end up using distillation for speed; useful to know when PatchCore is *not* the right tool.

### Flow / density-on-features
- **FastFlow** (Yu et al., 2021/2022): normalizing flow on pretrained features.

Why read: clarifies the “density estimation on embeddings” line vs “kNN memory bank” line.

### Recent efficiency-focused representatives
- **EfficientAD** (2023): strong speed/quality trade-off.
- **SimpleNet** (2023): simplified architecture + strong baseline.

Why read: you’ll likely compare against these if you care about latency.

## 2) The math/CS tricks PatchCore leans on

PatchCore is conceptually simple; the “trick” is in the *engineering-friendly approximations*.

### 2.1 kNN anomaly detection as a general pattern
- Nearest-neighbor anomaly detection / distance-to-manifold intuition.

What to internalize:
- Why **distance in feature space** works better than pixel space.
- The difference between **image-level score** and **pixel/patch-level maps**.
- Why calibration is hard: distances are not probabilities.

### 2.2 Coreset / k-center greedy (farthest-first)
PatchCore’s memory bank is reduced via a greedy procedure that approximates k-center selection.

Read/know:
- **k-center objective** and **farthest-first traversal** (a.k.a. greedy k-center).
- Why it gives coverage guarantees in metric spaces.

Keywords to search:
- “farthest-first traversal k-center greedy 2-approx”
- “coreset selection for kNN / facility location”

### 2.3 Random projection + Johnson–Lindenstrauss (JL) lemma
PatchCore uses random projections during coreset selection to make distance computations cheaper.

Read/know:
- **Johnson–Lindenstrauss lemma**: random linear maps approximately preserve pairwise distances.

Keywords to search:
- “Johnson–Lindenstrauss lemma random projection distance preservation”

### 2.4 Approximate nearest neighbor (ANN) for scaling
If you go beyond MVTec or want low latency, you’ll likely need ANN.

- **FAISS** (Facebook AI Similarity Search): https://github.com/facebookresearch/faiss

Why read: to understand index choices (IVF, HNSW, PQ) and recall/latency trade-offs.

## 3) Data + metrics you must understand to “apply” PatchCore

### 3.1 Benchmarks
- **MVTec AD**: object-wise splits; both image-level and pixel-level evaluation.
- **VisA**, **MPDD**, **BTAD**: common follow-on datasets with different failure modes.

See this repo’s dataset notes:
- `docs/DATASETS.md`

### 3.2 Metrics (and what they hide)
PatchCore papers report AUROC heavily, but industrial QA often needs tail metrics.

Must understand:
- Image-level AUROC vs pixel-level AUROC
- **PRO** (per-region overlap) and how it’s computed
- Operational metrics: **recall at fixed FPR**, “inspections/hour”, thresholding strategy

Why: deployment almost always becomes a threshold + monitoring problem.

## 4) A practical mental model (how to reason about PatchCore)

Think of PatchCore as:

1. A pretrained backbone provides a **feature space** where nominal patches cluster.
2. A memory bank approximates the nominal manifold.
3. Anomalies are **out-of-distribution patches** → large kNN distances.
4. Coreset subsampling keeps the bank representative enough while staying fast.

Failure modes to watch:
- Domain shift (lighting/camera/texture changes) → many false positives.
- “Near anomalies” that remain close in feature space.
- Poor calibration across products: thresholds don’t transfer.

## 5) Suggested reading order (fastest path)

1) PatchCore paper + skim official repo flags.
2) MVTec AD evaluation + PRO.
3) k-center greedy / farthest-first (enough to understand the coreset intuition).
4) JL lemma (why random projections are safe-ish here).
5) One competitor in each family: PaDiM + STFPM + EfficientAD.

---

If you want, I can turn this into a `docs/reading/` folder with BibTeX + local PDFs, but this is the high-signal minimum.
