# Production Deployment Design

**Project:** TTRIAD (PatchCore-based industrial anomaly detection + known-failure lookup)

**Doc status:** Draft (v0.1)

**Last updated:** 2026-04-06

---

## 0) Purpose

This document turns the repo from a research/evaluation harness into a concrete production deployment design for a high-recall industrial inspection setting.

The intended operating model is:

- **Primary detector:** PatchCore-style nominal anomaly detection
- **Optional second head:** known-failure retrieval / defect bank lookup
- **Primary goal:** maximize recall subject to a bounded inspection budget
- **Primary control:** threshold calibrated on held-out nominal data
- **Primary safeguard:** human review for alerts and novel defects

This is the design I would use for a Starlink-like manufacturing environment with fixed stations, fixed cameras, high throughput, and high cost of missed defects.

---

## 1) Deployment stance

This system should not be deployed initially as a single autonomous ship / no-ship gate.

Recommended production posture:

- **Phase 1:** advisory mode
- **Phase 2:** inspection-routing mode
- **Phase 3:** constrained gating mode for tightly validated stations only

Practical meaning:

- In phase 1, the model scores parts and writes evidence, but humans still make all decisions.
- In phase 2, only flagged parts are routed for secondary inspection.
- In phase 3, the system may block obviously bad parts, but only after station-specific validation and a stable drift-monitoring history.

---

## 2) System boundary

This repo should own the model and evidence layer, not the entire factory control system.

### In scope for this repo

- model training on nominal images
- threshold calibration
- image scoring
- anomaly map generation
- nearest-neighbor explanation artifacts
- known-failure retrieval artifacts
- run manifests and provenance
- review-set generation for false positives / false negatives

### Out of scope for this repo

- camera triggering
- PLC / MES integration
- physical reject actuators
- operator UI beyond simple review artifacts
- plant-wide scheduling / orchestration

Those integrations should consume the artifacts this repo writes rather than reimplement model logic elsewhere.

---

## 3) Deployment topology

### 3.1 Logical components

**A. Capture service**

- acquires station images
- tags them with `station_id`, `camera_id`, `product_id`, `process_rev`, `timestamp`, `unit_serial`
- writes raw images to durable storage

**B. Inference worker**

- loads the station-specific model artifact
- scores new images with `scripts/score_images.py`
- writes JSONL scores, anomaly maps, and manifests

**C. Threshold / routing policy**

- applies the station-specific threshold artifact
- decides whether the unit is:
  - pass
  - review
  - hard-fail

**D. Review workflow**

- shows flagged images, anomaly maps, and nearest nominal neighbors
- captures reviewer decision and defect taxonomy if known

**E. Retraining / requalification pipeline**

- ingests approved nominal data
- rebuilds model and threshold
- runs qualification suite
- publishes versioned artifacts

### 3.2 Recommended ownership split

- factory systems own image capture and unit traceability
- this repo owns model training, inference, manifests, and review artifacts
- deployment glue owns model serving, storage movement, and alert routing

---

## 4) Model boundary definition

Do not train one broad model across loosely related operating modes just because the images look similar.

Default model granularity should be:

- one model per `product_family`
- one model per `station`
- one model per `camera`
- one model per major `process_rev`

Collapse across these only after proving that:

- nominal score distributions remain stable
- false positives do not spike
- review burden stays acceptable

If the process is multi-modal but still operationally one station, use routed / multi-bank variants only after the single-bank baseline is understood.

---

## 5) Artifact contract

The production unit of deployment is a versioned artifact bundle.

### 5.1 Required model artifacts

For nominal anomaly detection:

- `config.json`
- `memory_bank.npy`
- `backbone_state.pt`
- `memory_metadata.json`
- `artifact_info.json`
- `run_manifest.json`

Optional but recommended:

- threshold artifact JSON
- saved review panels from qualification
- calibration score distribution summary

### 5.2 Versioning requirements

Each deployed model version must be traceable to:

- raw training image set
- nominal calibration image set
- code commit SHA
- exact config
- threshold selection rule
- qualification report

This repo already supports that direction through run manifests. Deployment should treat manifests as mandatory, not optional.

---

## 6) Offline lifecycle

### 6.1 Data splits

Use these production-minded buckets:

- `nominal_train`
- `nominal_calib`
- `nominal_monitor`
- `defect_eval`

Definitions:

- `nominal_train`: used to build the memory bank
- `nominal_calib`: used to choose operating threshold
- `nominal_monitor`: recent nominal data used only for drift monitoring
- `defect_eval`: known defect data used only for evaluating recall and failure taxonomy support

### 6.2 Recommended commands

Train nominal model:

```bash
python3 scripts/fit_nominal_patchcore.py \
  --nominal /data/station_A/cam_03/nominal_train \
  --out outputs/models/station_A_cam_03_rev12 \
  --device cpu \
  --image-size 256 \
  --coreset-ratio 0.02 \
  --seed 0
```

Score calibration nominal set:

```bash
python3 scripts/score_images.py \
  --model outputs/models/station_A_cam_03_rev12 \
  --images /data/station_A/cam_03/nominal_calib \
  --out outputs/calib/station_A_cam_03_rev12.jsonl
```

Calibrate threshold:

```bash
python3 scripts/calibrate_threshold.py \
  --scores outputs/calib/station_A_cam_03_rev12.jsonl \
  --target-fpr 0.001 \
  --out outputs/thresholds/station_A_cam_03_rev12.json
```

Generate qualitative review set:

```bash
python3 scripts/review_dataset_examples.py \
  --dataset mvtec \
  --root /data/review_like_layout \
  --category bottle \
  --model outputs/models/station_A_cam_03_rev12 \
  --threshold outputs/thresholds/station_A_cam_03_rev12.json \
  --select fp \
  --top-n 25 \
  --outdir outputs/review/station_A_cam_03_rev12_fp
```

The exact dataset wrapper will depend on local layout, but the qualification flow should follow this shape.

### 6.3 Qualification gates before deployment

Require all of the following:

- target nominal FPR achieved on `nominal_calib`
- acceptable alert rate on `nominal_monitor`
- acceptable recall on `defect_eval`
- review of top false positives
- review of top false negatives
- no known station drift between training and deployment image conditions

---

## 7) Online inference flow

### 7.1 Step-by-step runtime flow

1. capture image and metadata
2. verify image quality / basic capture health
3. route to station-specific model
4. score image with nominal model
5. if score < threshold: mark pass
6. if score >= threshold: mark review
7. if known-failure head exists: attach likely failure types
8. persist score, anomaly map path, model version, manifest path, and unit metadata

### 7.2 Required runtime metadata per scored image

- `unit_serial`
- `station_id`
- `camera_id`
- `product_id`
- `process_rev`
- `timestamp`
- `model_version`
- `threshold_version`
- `score`
- `threshold`
- `is_anomaly`
- `run_id`
- path to anomaly map / evidence panel if generated

### 7.3 Hard-fail policy

Do not hard-fail solely from anomaly score during first deployment.

Safer rollout:

- score below threshold: pass
- score above threshold but below critical band: manual review
- score far above threshold plus high-confidence known failure: candidate hard-fail, but still station-specific

The critical band should be learned from actual factory review data, not chosen arbitrarily.

---

## 8) Strict nominal data curation

This is the most important operational discipline in a PatchCore-style system.

### 8.1 What it means

**Strict nominal data curation** means:

- only images that are genuinely acceptable production output go into the nominal memory bank
- every nominal image is trusted enough that you would be comfortable teaching the model "this is good"
- uncertain, borderline, mislabeled, or process-transition images are excluded or quarantined

This is not just "collect a lot of good-ish images." It is controlled construction of the definition of normal.

### 8.2 What must be excluded from nominal training

Never include:

- known defective parts
- suspected defects
- rework images
- startup / warmup transients
- maintenance windows
- camera misfocus / blur / exposure failures
- images with incorrect framing or missing part presence
- images from mixed product revisions unless intentionally supported
- images captured during upstream process excursions

Also exclude by default:

- ambiguous cosmetic issues that reviewers disagree on
- units without traceability
- data from days with uninvestigated score spikes

### 8.3 Required curation states

Every candidate nominal image should be in one of these states:

- `accepted_nominal`
- `rejected_defect`
- `rejected_capture_issue`
- `rejected_process_excursion`
- `quarantine_uncertain`

Do not allow unlabeled images to flow straight into `nominal_train`.

### 8.4 How to curate in practice

Minimum practical workflow:

1. collect candidate good production images from stable line operation
2. sample across shifts, lots, and acceptable nuisance variation
3. review a statistically meaningful subset manually
4. remove obvious capture failures and process transitions
5. quarantine any borderline images
6. build the first model
7. inspect top false positives on held-out nominal data
8. either:
   - add reviewer-approved benign variation to nominal training
   - or tighten the capture/process spec if the variation should not exist

### 8.5 What “strict” means quantitatively

For a high-stakes deployment, I would want:

- a documented acceptance rule for nominal inclusion
- a per-batch rejection log
- at least one recent cross-shift nominal holdout
- top false positives reviewed before release
- a quarantine bucket for uncertain data rather than forced binary labeling

### 8.6 The main anti-pattern

The biggest mistake is to suppress false positives by throwing everything into the nominal bank.

That creates two failure modes:

- the model learns true defects as normal
- the model learns unstable process drift as normal

Short-term alert rate improves; long-term recall degrades quietly.

### 8.7 Good nominal curation policy

Good policy is:

- include acceptable variation that production truly wants to pass
- exclude variation caused by broken capture or unstable process
- quarantine borderline cases until a domain owner decides whether they represent acceptable product or a real defect class

### 8.8 Nominal curation checklist

- same product family
- same camera geometry
- same station / similar optics
- stable lighting condition
- traceable unit metadata
- no known defects
- no unresolved reviewer disagreement
- no capture corruption
- no process-transition window
- balanced coverage across normal nuisance factors

---

## 9) Monitoring and requalification

### 9.1 Daily monitors

Track at least:

- score median
- score p95
- score p99
- alert rate
- top recurring alert clusters
- percent of images failing capture-quality checks

Bucket by:

- station
- camera
- product
- shift
- lot / batch where available

### 9.2 Drift triggers

Trigger investigation when:

- p99 shifts materially from calibration baseline
- alert rate exceeds inspection budget
- new false-positive mode becomes common
- camera is serviced or replaced
- process recipe / material / coating changes

### 9.3 Requalification triggers

Retrain and requalify when:

- process revision changes
- optics or lighting changes materially
- fixture or background changes
- nominal distribution drifts for multiple days
- new benign variation becomes operationally common

---

## 10) Failure modes and controls

### 10.1 Domain shift

Failure mode:

- lighting drift, focus drift, camera replacement, background change

Controls:

- per-station models
- capture health checks
- nominal monitor set
- mandatory requalification after hardware change

### 10.2 Contaminated nominal bank

Failure mode:

- defects in nominal training reduce recall

Controls:

- strict nominal curation
- quarantine state
- reviewer signoff
- audit trail from artifact to raw files

### 10.3 Inspection overload

Failure mode:

- target recall drives false positives too high for operations

Controls:

- calibrate threshold to explicit inspection budget
- review top false positives before rollout
- use routing / multi-bank only after baseline is stable

### 10.4 Shortcut learning on fixture/background

Failure mode:

- detector responds to tray, fixture, tape, screws, glare, or crop boundary

Controls:

- ROI masking where appropriate
- consistent framing
- review anomaly maps on top false positives

### 10.5 Novel defect overconfidence

Failure mode:

- known-failure head forces a bad label on a truly novel anomaly

Controls:

- keep unknown-defect reject path
- require confidence / margin threshold
- log retrieval evidence, not just class label

---

## 11) Rollout plan

### Phase 0: shadow data

- collect raw images and metadata
- do not influence production decisions
- establish capture-health baseline

### Phase 1: offline qualification

- build nominal model
- calibrate threshold
- review false positives and false negatives
- verify inspection budget

### Phase 2: silent online scoring

- score every unit in production
- no routing decisions yet
- compare alerts with downstream quality findings

### Phase 3: review routing

- flagged parts routed to secondary inspection
- capture reviewer outcomes
- build known-failure bank from confirmed failures

### Phase 4: constrained gating

- limited hard-fail policy for clearly validated failure bands
- station-by-station approval only

---

## 12) What this repo should support next

If this repo is to be used in production, the next engineering additions should be:

- a folder-based production scoring wrapper that accepts external metadata
- explicit capture-quality prechecks
- richer threshold artifact schema with station/product identifiers
- drift dashboard generation from score JSONL outputs
- curation manifest format for nominal dataset approval
- support for ROI masks and station-specific preprocessing contracts

---

## 13) Bottom line

This repo is already close to a usable model-and-evidence layer for production, but the real deployment success will depend less on the nearest-neighbor algorithm and more on operational discipline:

- strict nominal data curation
- station-specific model boundaries
- explicit thresholds tied to inspection budget
- drift monitoring
- review workflow
- controlled rollout

If those pieces are weak, PatchCore will look good offline and fail operationally. If those pieces are strong, this approach is a credible production inspection backbone.
