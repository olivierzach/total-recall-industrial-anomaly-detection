# Towards Total Recall in Industrial Anomaly Detection (PatchCore) — local implementation

This repo is a practical, local re-implementation + experimentation harness for:

- **Roth et al. (CVPR 2022)**: *Towards Total Recall in Industrial Anomaly Detection* (a.k.a. **PatchCore**)
  - arXiv: https://arxiv.org/abs/2106.08265
  - CVF PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
  - Code/datasets page: https://www.amazon.science/code-and-datasets/towards-total-recall-in-industrial-anomaly-detection

## What we’re building

- A clean PatchCore baseline (feature extraction → memory bank → coreset sampling → NN scoring)
- Reproducible evaluation on common benchmarks (start with **MVTec AD**)
- A two-head extension for industrial QA:
  - head A: nominal anomaly detection against a nominal memory bank
  - head B: known-failure lookup against a defect / failure memory bank with an unknown-defect reject path
- A set of ablations and knobs for “industrial QA” settings:
  - sensitivity/recall under rare defects
  - latency / memory trade-offs
  - domain shifts, new products, few-shot updates

## Current Repo Evidence

This repo already contains fitted artifacts and saved experiment outputs under `outputs/`.

- **MVTec bottle baseline:** `outputs/ablations/mvtec_full_bottle_layer3_euclidean_c01.json`
  - `wide_resnet50_2`, `layer3`, coreset `0.01`
  - saved result: image AUROC `1.0`
- **BTAD rigor suite:** `outputs/rigor/btad_suite_layer3_c01/`
  - `knn_euclidean.json`: image AUROC `0.8812`, pixel AUROC `0.7872`, PRO AUC `0.2127`, recall `0.6034` at ~1% target FPR
  - `knn_rp256_euclidean.json`: image AUROC `0.8966`, pixel AUROC `0.7901`, PRO AUC `0.2118`, much faster coreset build than raw euclidean
- **Saved fitted models:** `outputs/models/`
  - nominal-only banks such as `outputs/models/bottle_layer3_kmeans_c01`
  - routed nominal bank such as `outputs/models/bottle_routed_smoke`
  - BTAD nominal banks such as `outputs/models/btad_nominal_kmeans_c01_layer3`

## Two-Head Memory-Bank Direction

The main applied idea in this repo is not just “PatchCore as anomaly detection,” but a **two-head memory-bank system**:

- **Head A: nominal memory bank**
  - answers: “does this look unlike known-good production output?”
- **Head B: defect / known-failure bank**
  - answers: “if this is anomalous, does it resemble a known failure mode?”

Why this is innovative:

- it keeps the **open-set strength** of PatchCore instead of collapsing the problem into a closed-set defect classifier
- it separates two different jobs that factories usually need at the same time:
  - detect anything off-nominal with very high recall
  - name recurring failure families when enough examples exist
- it adds an **explicit reject path** for novel defects, so the system can say “this is bad, but not like any stored failure mode yet”
- it is **retrieval-first rather than training-heavy**:
  - Head A uses a nominal memory bank built from good images
  - Head B uses a small support library of labeled failures
  - new failure classes can be added by storing exemplars instead of retraining a large classifier
- it is naturally **explainable**:
  - anomaly heatmaps show where the image departs from nominal
  - nearest nominal neighbors show what “good” looked like
  - nearest failure supports show why a known-failure label was proposed
- it fits real deployment better than a single monolithic model because nominal routing, multiple banks, and unsupervised defect-bank bootstrap can all be layered onto the same retrieval pipeline

Why this works:

- **Head A is strong because industrial QA usually has plenty of good data.**
  - PatchCore uses nearest-neighbor distance to a nominal patch memory bank, which is a good match for the real data regime: abundant normal images, rare and changing defects
- **Thresholding is disciplined instead of ad hoc.**
  - the anomaly gate is calibrated on a held-out nominal set, so the operating point can be chosen around inspection budget / target FPR rather than eyeballed
- **Head B looks at the right evidence.**
  - failure descriptors are built from the most anomalous patches plus global image context, so lookup focuses on the suspicious region without losing the broader part appearance
- **Known-vs-unknown is handled explicitly.**
  - a support is only accepted as a known failure when it is close enough to the failure bank and sufficiently separated from competing labels; otherwise the sample stays `unknown_anomaly`
- **The decomposition reduces interference.**
  - Head A answers “is this off-nominal?”
  - Head B answers “if yes, which known failure is it closest to?”
  - this is more robust than forcing one classifier to learn normal detection, defect taxonomy, and open-set rejection all at once
- **It is few-shot friendly.**
  - recurring failures do not need large balanced training sets; a small number of support examples per failure family can already improve triage and routing
- **The repo’s saved artifacts already support the direction.**
  - nominal-only PatchCore is strong on simple MVTec settings
  - the hybrid artifacts show materially better known-failure handling than nominal-only status prediction
  - the remaining hard problem is open-set rejection quality for truly novel defects, which is exactly the right place for the system to still be conservative

### Technical details of the implemented two-head system

The concrete implementation lives under `hybrid_memory_demo/` and is intentionally built as a thin layer on top of the base PatchCore code in `src/patchcore/`.

- **Shared backbone / embedding pipeline**
  - both heads use the same frozen vision backbone and the same extracted patch embedding tensor
  - default hybrid config is in `hybrid_memory_demo/model.py`:
    - `layers=("layer2", "layer3")`
    - `image_size=256`
    - `l2_normalize=True`
    - `image_score="max"`
- **Head A training artifact = nominal patch memory**
  - nominal train images are embedded patch-wise
  - all nominal patches are concatenated into one large matrix
  - k-center coreset selection reduces that matrix to a representative nominal memory bank
  - a standard `PatchCoreModel` is then fit as an exact kNN index over the retained nominal embeddings
- **Head A inference output**
  - each query patch gets an anomaly score equal to its nearest-neighbor distance to the nominal bank
  - image score is the max patch score by default
  - the anomaly threshold is not hand-tuned; it is calibrated from a held-out nominal calibration split using a quantile rule (`anomaly_quantile`, default `0.99`)
- **Head B training artifact = failure descriptor bank**
  - each labeled failure support image is first scored by Head A against the nominal bank
  - the support descriptor is then built as:
    - mean embedding of the top anomalous patches (`failure_top_k_patches`, default `8`)
    - concatenated with the global mean embedding over all patches
    - followed by L2 normalization
  - this produces one descriptor per support image, plus a `SupportRecord(label, path, anomaly_score)` for traceability
- **Why the failure descriptor has two parts**
  - the salient half concentrates on the suspicious local region
  - the global half keeps part-level context so lookup does not overfit to a tiny texture fragment
  - this is the main implementation choice that makes the second head more than a plain global classifier
- **Head B decision rule**
  - at inference, the query image gets the same anomaly-weighted descriptor
  - descriptor-to-support distances are computed against the failure bank
  - the nearest supports are grouped by label, and each label score is the mean distance of its retrieved supports
  - the best label is accepted only if both conditions hold:
    - best label distance is below the calibrated known-failure threshold
    - top-2 label separation is large enough, measured as `second_best / best >= margin_threshold`
- **Known-vs-unknown calibration**
  - `known_failure_threshold` is estimated from same-label support distances using a quantile rule (`known_failure_quantile`, default `0.95`)
  - `margin_threshold` is estimated from the ratio between nearest different-label and nearest same-label support distances, with a floor of `min_margin_ratio` (default `1.05`)
  - this means the reject path is explicit in the artifact, not bolted on after prediction
- **Final runtime states**
  - `normal`: image anomaly score is below the nominal threshold
  - `known_failure`: image is anomalous and the failure-bank lookup passes both distance and margin checks
  - `unknown_anomaly`: image is anomalous but no known failure label is accepted
- **Persisted artifact contents**
  - a fitted hybrid artifact stores:
    - `nominal_memory.npy`
    - `failure_descriptors.npy`
    - `support_records.json`
    - `thresholds.json`
    - `config.json`
  - this is important operationally because new failure supports can be added by rebuilding the defect-bank side of the artifact without redesigning the whole pipeline

In code terms, the two-head system is therefore:

1. Patch-level nominal retrieval for open-set detection.
2. Anomaly-conditioned descriptor construction for suspicious evidence.
3. Support-set retrieval with calibrated distance and margin rejection for known-failure naming.

That decomposition is a key design choice in this repo: the first head protects recall on anything off-nominal, while the second head adds practical failure-family lookup without collapsing the whole system into a closed-world classifier.

Current saved evidence for that direction:

- **Hybrid benchmark:** `outputs/hybrid_benchmark/mvtec_bottle/summary.md`
  - baseline nominal-only status accuracy: `0.482`
  - hybrid known-failure bank with 8 supports/class: status accuracy `0.706`, known-failure recall `0.821`, known-label accuracy `0.654`
  - caveat: unknown-anomaly recall is still only `0.336`, so the open-set rejection problem is not solved yet
- **Single-run hybrid demo artifact:** `outputs/hybrid_memory_demo/mvtec_bottle/report.json`
  - status accuracy `0.773`
  - known-failure recall `0.647`
  - unknown-anomaly recall `0.810`
- **Unsupervised defect bank bootstrap:** `outputs/defect_bank/mvtec_bottle_unsup/`
  - stores clustered high-scoring anomaly patches to bootstrap a known-failure taxonomy
- **Multi-bank nominal routing demo:** `docs/experiments/multibank_bottle_btad_routing.md`
  - proof-of-concept that separate nominal banks can be routed correctly across bottle vs BTAD domains

Interpretation:

- the repo already has concrete evidence for:
  - strong nominal-only detection on a simple MVTec setting
  - operational BTAD evaluation with threshold calibration
  - routed / multi-bank nominal memory
  - a hybrid nominal-bank + known-failure-bank prototype
- the open problem is not whether the idea exists in code, but how robustly the second head handles **novel defects vs known failures** without over-forcing novel issues into a known label

## Quickstart

- New machine + your own nominal images: `docs/notes/FROM_SCRATCH.md`
- Fast dev iteration: `docs/notes/DEV_CONFIG.md`
- Mac mini GPU (MPS): `docs/notes/GPU_ACCEL.md`
- Sweeps (BTAD): `docs/notes/SWEEPS.md`
- Datasets + official MVTec links: `docs/notes/DATASETS.md`

Mac mini note:
- MPS works on the tested Apple Silicon machine when launched from a normal shell / `tmux`.
- The sandboxed agent runtime may report `mps_available=False` even though the machine itself supports MPS.

## Next steps

Start here:
- `docs/notes/PLAN.md`
- `docs/notes/CORE_READING.md` (what to read to understand/apply PatchCore)
- `docs/notes/PRODUCTION_DEPLOYMENT_DESIGN.md` (concrete deployment design for real factory rollout)
- `docs/notes/PATCHCORE_VS_LLM.md` (trade-offs: PatchCore memory bank vs. LLM/VLM one-shot)
- `docs/notes/LLM_TRIAGE_INTEGRATION.md` (how to bolt an LLM/VLM triage stage onto the two-head system)
- `docs/notes/INDUSTRIAL_RIGOR_CHECKLIST.md` (how to ship/present with rigor)
- `docs/notes/EVAL_PROTOCOL.md`
- `docs/notes/BASELINE_LADDER.md`
- `docs/notes/DEFECT_LOOKUP.md` (extending anomaly detection with known failure-mode lookup)
- `docs/notes/ABLATIONS_DISTANCE_METRICS.md` (euclidean vs cosine vs PCA-whitened kNN)
- `docs/notes/QUERY_ADAPTIVE_MEMORY.md` (query-adaptive nominal selection; patch vs image routing)
- `docs/notes/IVF.md` (what IVF indexing is; how it maps to PatchCore retrieval)
- `docs/notes/DEFECT_BANK_UNSUPERVISED.md` (cluster anomalies to bootstrap defect memory bank)
- `docs/notes/LIGHTING_AND_CAPTURE.md` (lighting/camera risks + early go/no-go tests)
- `docs/notes/RUN_STABILITY.md` (avoid SIGKILL; caching + iteration knobs)
- `docs/notes/RELATED_WORK.md`
