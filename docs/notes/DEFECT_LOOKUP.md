# Extending PatchCore with defect lookup ("known failure mode" memory)

Goal: keep PatchCore’s strength (nominal-only anomaly detection + localization), but when an anomaly is found, also answer:

- *“Which known failure mode does this most resemble?”* (e.g., cracked capacitor)

This is analogous to retrieval-augmented classification: detect OOD regions, then retrieve prototypes.

## 1) Proposed two-stage pipeline

### Stage A — nominal anomaly detection (PatchCore)
- Build nominal patch memory bank (possibly coreset).
- For a test image, compute patch anomaly scores and an anomaly map.
- Select candidate anomalous patches/regions:
  - top-k patches by anomaly score
  - or connected components on the heatmap

Output: a small set of candidate patch embeddings {x_i} + their image coordinates.

### Stage B — defect lookup / classification
Maintain a *defect memory bank* D, organized by defect type.

At inference:
- For each candidate embedding x:
  - retrieve nearest neighbors in D
  - aggregate votes/scores per defect type

Return:
- ranked list of defect types + evidence patches
- optional: confidence score / reject option (“unknown defect”)

## 2) Data requirements (what you must collect)

- Labeled defect examples for each failure mode.
- Consistent capture conditions if possible (or explicit shift handling).

Practical tip:
- Store defect exemplars as *patches* or *regions* + their embeddings.
- Keep the original images for human review.

## 3) How to avoid over-claiming

Defect lookup should have a reject option:
- If defect memory neighbors are weak / inconsistent, return “anomalous, unknown failure mode.”

This avoids mislabeling novel defects.

## 4) Scoring / calibration

You can calibrate defect-type scores using:
- distance margin between top-1 and top-2 defect types
- similarity to defect prototypes vs nominal prototypes

## 5) Learned retrieval vs pure kNN

kNN is the simplest baseline, but you can learn the lookup stage:

### Option 1 — Linear classifier on embeddings (SVM / logistic)
- Train one-vs-rest linear SVM on patch embeddings for defect types.
- Works surprisingly well; fast; easy to explain.
- Add a rejection rule based on max score / calibration.

### Option 2 — Metric learning / embedding adaptation
- Train a small projection head so “same defect” patches cluster.
- Triplet / contrastive losses.
- Beware: needs enough defect diversity; can overfit to nuisance cues.

### Option 3 — Prototype networks / nearest centroid
- Maintain centroids per defect type and compare to centroids.
- Often good when data is small.

## 6) Relationship to RAG intuition

- Nominal PatchCore memory is like the **document store** for “what normal looks like.”
- Defect memory is like the **labeled retrieval set**.
- The “Karpathy-style SVM in the lookup stage” maps to a **linear head on frozen embeddings**:
  - retrieval-free classification, still grounded in the same representation.

## 7) Recommended implementation order

1) PatchCore anomaly map + candidate patch extraction (top-k)
2) Defect memory bank + kNN lookup + reject rule
3) Add linear SVM baseline on embeddings
4) If needed, consider learned projection heads / metric learning

Deliverables:
- confusion matrix over known defects
- unknown-defect rejection curve (trade-off)
- qualitative panels with retrieved exemplars
