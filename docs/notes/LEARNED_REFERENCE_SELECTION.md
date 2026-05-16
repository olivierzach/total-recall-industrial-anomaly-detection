# Learned reference selection / supervised coreset for PatchCore

This note records a concrete plan to use *supervision* (or weak supervision) to choose a **better nominal reference set** for PatchCore-style kNN scoring.

## 0) Clarify: what PatchCore is doing (nominal-only)

PatchCore is primarily a **nearest-neighbor anomaly detector in feature space**.

- We embed each image into a set of **patch embeddings** using a pretrained backbone.
- We build a memory bank `M` from **nominal training patches only**.
- For a test image, for each test patch embedding `e`, we compute a distance like:

  `d(e) = min_{m in M} || e - m ||`  (or kNN variants)

- Large distance means the patch looks **unlike anything seen in nominal data**, i.e. likely anomalous.

### Why compare to "nominal" at all?
Because anomalies are defined as **deviation from nominal**.

The nominal reference set is the *definition* of normality for the detector:
- If the nominal set is too small / unrepresentative, you get false positives.
- If it’s too broad / noisy, anomalies can get absorbed and you lose recall.

So the memory bank is both:
1) the **baseline manifold** we measure distance to, and
2) the **computational object** that sets runtime (more points = slower kNN).

The kNN is the whole scoring mechanism; it’s not optional.

## 1) Motivation: learned / supervised reference selection

We can use supervision not to directly predict anomaly, but to **select** a better subset of nominal patches to store and compare against.

Goal:
- improve localization (PRO) and/or AUROC
- while shrinking the memory bank

This is essentially a learned coreset / prototype set tuned for detection.

## 2) Minimal, high-leverage approach: linear SVM in embedding space

Assume we have a small labeled set with either:
- pixel masks (best), or
- image-level anomaly labels (okay), or
- weak labels / heuristics

### 2.1 Data construction
Create a training dataset over patch embeddings:
- `x`: patch embedding
- `y`: 0 nominal, 1 anomalous

If you have masks:
- sample patch embeddings from anomalous regions as positives
- sample nominal patch embeddings as negatives

If you only have image-level labels:
- use MIL-style sampling: treat patches from anomalous images as “noisy positives” (harder)

### 2.2 Train linear model
Train a linear classifier (SVM or logistic regression) on patch embeddings.

Why linear often suffices:
- pretrained embeddings are already linearly separable for many defect types
- simple models are stable and easy to debug

### 2.3 Use the model to select the nominal memory bank
We want a nominal memory bank that is:
- diverse (covers nominal variation)
- “hard” enough to preserve discriminative boundaries

Two practical selectors:

**(A) Hard-negative nominal patches**
- score each nominal patch by proximity to boundary (low margin)
- keep top-K hardest nominal patches

**(B) Support-vector style**
- for SVM, the support vectors identify “important” points; keep nominal SVs

Add a diversity term (optional but recommended):
- cluster nominal patches, select hard points per cluster
- or mix: 70% hard + 30% random/diverse

### 2.4 PatchCore scoring unchanged
After selection, PatchCore stays nominal-only:
- distance to this selected nominal set

So this is a *drop-in refinement* of the memory bank.

## 3) Two-stage retrieval variant (optional)

Use the linear model to *shortlist* which nominal prototypes are relevant for a test image/patch, then compute exact kNN distance only within that shortlist.

This can improve both:
- runtime (fewer comparisons)
- accuracy (avoid irrelevant nominal modes)

## 4) Evaluation protocol (must be explicit)

### Metrics
- image AUROC
- pixel AUROC
- PRO AUC (FPR<=0.3)

### Ablations
- baseline PatchCore (random/coreset)
- learned-selection PatchCore
- same memory bank size across methods (fairness)

### Key plots
- PRO vs memory bank size
- PRO vs runtime

## 5) Implementation plan (repo-level tasks)

1) Add an embedding dump utility:
   - save patch embeddings + patch locations for a subset

2) Add a patch-label builder:
   - from masks OR weak labels

3) Train a linear model:
   - sklearn `LinearSVC` or `LogisticRegression`

4) Build selected memory bank:
   - output indices into nominal patch pool

5) Run PatchCore with that bank:
   - compare metrics + timing

## 6) What supervision is needed?

- **Best:** pixel masks for anomalies (even a small number of images)
- **Okay:** image-level anomaly labels (we can still do something, but noisier)
- **None:** you can still do unsupervised coreset; it’s just not this approach

## 7) Why this preserves the “nominal-only” character

At inference time, the detector still uses:
- pretrained feature extractor
- distances to a **nominal-only** memory bank

Supervision is used only to construct that nominal bank more intelligently.

---

If you tell me what labels you have (pixel masks vs image-level) and roughly counts, we can choose the simplest variant that’s likely to move PRO meaningfully.
