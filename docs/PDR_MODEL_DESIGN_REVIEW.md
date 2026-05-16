# Model Design Review (PDR) — Towards Total Recall Industrial Anomaly Detection (TTRIAD)

**Project:** TTRIAD (PatchCore-based industrial anomaly detection + known-failure lookup)

**Doc status:** Draft (v0.1)

**Owner:** Zach

**Last updated:** 2026-03-28

---

## 0) Executive summary

We propose a two-head anomaly detection system for industrial visual inspection:

- **Head A (Nominal anomaly detection):** PatchCore-style nearest-neighbor anomaly scoring against a *nominal memory bank* built from “good” images.
- **Head B (Known failure lookup):** retrieval against a *defect memory bank* of known failure exemplars, producing a ranked list of likely failure modes **with a reject option** ("unknown defect").

The system is designed to be:
- **high recall** at a controllable false positive rate (inspection budget)
- **explainable** (localization maps + retrieved exemplars)
- **extensible** across products/modes via multiple banks and routing
- **efficient** via coreset selection and optional dimensionality reduction (PCA/whitening or JL-style random projection)

---

## 1) Problem statement

Given images from one or more industrial stations/products, detect defects under class imbalance where:
- “good” examples are abundant
- defects are rare, diverse, and evolve over time

**Primary output:** `nominal` vs `anomalous` decision at an operating point chosen for the inspection budget.

**Secondary outputs:**
- anomaly localization heatmap
- explanation of why (nearest nominal neighbors)
- if anomalous: likely defect type(s) when defect resembles a known failure mode

---

## 2) Scope and non-goals

### In-scope
- image-level anomaly detection + patch-level localization
- reproducible evaluation (AUROC + operational metrics)
- multi-mode nominal handling (routing / multi-bank)
- defect-bank retrieval for known failure modes

### Non-goals (for PDR phase)
- end-to-end finetuning of large backbones on production data
- replacing human-in-the-loop review for novel defect discovery
- real-time embedded inference constraints (unless specified later)

---

## 3) Requirements (what success means)

### 3.1 Functional requirements
- Detect anomalous images with controllable FPR.
- Provide localization map (heatmap) to support review.
- For known failure modes, return top-K defect labels and exemplar evidence.
- Provide an “unknown defect” rejection pathway.

### 3.2 Performance requirements
- Throughput: report ms/image on target hardware.
- Memory: quantify memory bank size (MB) and runtime RAM.
- Stability: variance across seeds / nominal subset / minor shifts.

### 3.3 Rigor / reporting requirements
- Split definition prevents leakage (part/time/batch/camera).
- Thresholding procedure defined and repeatable.
- Automated logs/manifests include dataset identity and config.

---

## 4) Data and split design

### 4.1 Data units
- **Image**: a single capture.
- **Unit/Part**: all captures associated with one physical item (serial).

### 4.2 Splits (choose one primary; others as shift tests)
- Time-based (preferred for deployment realism)
- Part/serial-based grouping
- Batch/lot-based
- Camera/site-based

### 4.3 Leakage checks
- near-duplicate detection across splits
- enforce unit grouping constraints

**Reference:** `docs/EVAL_PROTOCOL.md`

---

## 5) System architecture (two-head)

### 5.1 Common preprocessing contract
- Resize to fixed `image_size` (default 256).
- Normalize (ImageNet mean/std for torchvision backbones).
- Optional ROI masking (fixture borders / irrelevant regions).

### 5.2 Head A — PatchCore nominal anomaly detection

**Inputs:** nominal (“good”) images.

**Training artifacts:**
- backbone config + layer hooks
- nominal patch embedding matrix `X` (implicit)
- **memory bank** `M` created by coreset selection
- optional preprocessing transform on embeddings:
  - PCA(+whitening) OR
  - JL-style random projection (RP)

**Inference:**
- compute patch embeddings for query
- apply same embedding transform
- score by kNN distance to nominal memory
- aggregate to image score (`max`/`mean`)
- produce anomaly map (patch grid)

**Key knobs:**
- backbone / layers
- coreset ratio and method (k-center / random / kmeans prototypes)
- metric (euclidean/cosine)
- embedding transform (none / PCA / RP)

### 5.3 Thresholding
- calibrate threshold on held-out nominal set to meet target FPR or inspection budget.

### 5.4 Head B — Known failure lookup (defect bank)

**Goal:** given anomalous patches/regions, retrieve nearest known defect exemplars.

**Defect bank sources:**
- supervised: labeled defect exemplars by failure mode
- unsupervised bootstrap: cluster high-scoring anomaly patches (human names clusters)

**Inference:**
- select candidate anomalous patches (top-k or region proposals)
- retrieve in defect embedding space
- aggregate votes/scores per defect type
- reject as unknown if confidence low

**Reject rules (MVP):**
- min distance threshold to defect bank
- or margin between top1/top2 defect types

**Reference:** `docs/DEFECT_LOOKUP.md`, `docs/DEFECT_BANK_UNSUPERVISED.md`

---

## 6) JL / dimensionality reduction design decision

### 6.1 Why we want it
- k-center coreset selection is expensive at high embedding dimension.
- defect-bank retrieval can grow large.

### 6.2 Options
- PCA (data-dependent; may improve metric by removing correlated directions)
- PCA + whitening (changes metric to Mahalanobis-like)
- Random projection (JL-style; data-independent; distance-preserving for finite sets)

### 6.3 Validation
- neighbor-overlap@k between full-D and reduced-D on representative samples
- stability of AUROC and recall@FPR across seeds

---

## 7) Evaluation plan

### 7.1 Metrics
Research:
- image AUROC
- pixel AUROC
- PRO AUC

Operational:
- recall @ fixed FPR
- precision @ required recall
- alert rate / inspection volume

### 7.2 Protocol
- strict splits + leakage checks
- threshold calibration on nominal holdout
- report runtime metrics and memory

### 7.3 Error analysis
- top false positives with maps
- top false negatives with maps
- cluster FP modes (lighting, edges, speculars, etc.)

---

## 8) Risks and mitigations

### 8.1 Domain shift (lighting/camera drift)
- Mitigation: shift test suite + capture guidelines + ROI masks

### 8.2 False explanations (nearest neighbor not semantically similar)
- Mitigation: show multiple neighbors; sanity-check retrieval; consider whitening/RP

### 8.3 Defect lookup over-claims
- Mitigation: reject option; calibrate confidence; maintain “unknown” bucket

### 8.4 Parasitic leakage in data split
- Mitigation: explicit unit grouping; dedup checks

---

## 9) Implementation status (repo)

### Implemented
- PatchCore baseline harness + experiments
- evaluation scripts (MVTec/BTAD)
- query-adaptive nominal routing (IVF-like)
- unsupervised defect-bank bootstrap script
- run manifests / provenance utilities

### Planned / next
- a first-class defect lookup scorer integrated into the scoring pipeline
- labeled defect-bank artifact format
- optional FAISS-based indexing when banks scale

---

## 10) Milestones

### M0 — Baseline
- Reproduce PatchCore baseline on at least one dataset with thresholded operational metrics.

### M1 — Robustness + speed
- Add dimensionality reduction ablation (PCA vs RP) + report speed/quality trade.

### M2 — Known-defect lookup MVP
- Defect bank (labeled or clustered) + retrieval + reject rule + qualitative evidence panel.

### M3 — Multi-mode support
- Multi-bank routing and/or explicit product-ID routing; demonstrate prevents cross-product false positives.

---

## Appendix A: Primary scripts (current repo)

- Fit nominal: `scripts/fit_nominal_patchcore.py`
- Evaluate: `scripts/eval_btad_patchcore.py`, `scripts/eval_mvtec_patchcore.py`
- Score images: `scripts/score_images.py`
- Routed scoring: `scripts/score_images_routed.py`
- Defect bank bootstrap: `scripts/build_defect_bank_unsupervised_mvtec.py`

