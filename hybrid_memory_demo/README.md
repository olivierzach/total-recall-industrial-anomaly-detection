# Hybrid Failure Memory Demo

This is an additive prototype layered on top of the existing PatchCore-style code in this repo.

Goal:
- keep a **nominal memory bank** for open-set anomaly detection
- keep a **labeled failure memory bank** for known defect retrieval/classification
- return either:
  - `normal`
  - `known_failure` with the closest stored failure mode
  - `unknown_anomaly` when the image is off-nominal but does not fit the stored failure library

## Why this matches the product need

The current repo is nominal-only. That is strong for total-recall anomaly detection, but it does not answer:

- which already-known failure mode does this resemble?
- is this a new failure family, or one we have seen before?

This prototype adds that second stage without changing the baseline code:

1. PatchCore nominal bank scores whether an image is off-nominal.
2. An anomaly-weighted descriptor is built from the most anomalous patches.
3. A nearest-neighbor search over labeled failure supports decides:
   - close to a known class -> `known_failure`
   - far from known classes -> `unknown_anomaly`

## Files

- `hybrid_memory_demo/model.py`
  - pure numpy logic for descriptors, thresholds, and known-vs-unknown decisions
- `hybrid_memory_demo/layout.py`
  - folder-layout loader for your own nominal/calibration/known-failure dataset
- `hybrid_memory_demo/pipeline.py`
  - training, artifact I/O, and runtime inference
- `hybrid_memory_demo/demo_common.py`
  - shared report/example helpers used by dataset-specific demo runners
- `hybrid_memory_demo/fit_folder_hybrid_memory.py`
  - CLI to fit a hybrid artifact from your own folder layout
- `hybrid_memory_demo/run_mvtec_bottle_demo.py`
  - public benchmark example on `data/mvtec/bottle`
- `hybrid_memory_demo/run_btad_demo.py`
  - public benchmark example on `data/btad`
- `hybrid_memory_demo/web.py`
  - lightweight browser UI with multi-dataset artifact switching
- `hybrid_memory_demo/app.html`
  - frontend for uploads, dataset examples, and embedding-space heatmaps
- `hybrid_memory_demo/demo_manifest.json`
  - browser manifest for loading both the MVTec and BTAD demo artifacts
- `hybrid_memory_demo/router.py`
  - learned nominal-domain router over image embeddings
- `hybrid_memory_demo/build_nominal_router_demo.py`
  - builds the bottle-vs-BTAD nominal router used by viewer auto mode
- `hybrid_memory_demo/SETUP_KNOWN_FAILURES.md`
  - step-by-step guide for setting up the known-failure bank on your own data

## Public Benchmark Demos

This prototype supports two public benchmark demo profiles:

- `MVTec AD / bottle`
  - known failures: `broken_large`, `broken_small`
  - held-out novel failure: `contamination`
- `BTAD / component-conditioned defects`
  - BTAD only exposes `ok/ko`, so the demo uses component-conditioned labels (`component_01`, `component_02`) as the known failure bank
  - held-out unknown component anomalies default to `component_03`
  - the viewer manifest points at the stronger BTAD profile: `wide_resnet50_2`, `layer3`, `image_size=256`, `coreset_ratio=0.01`
  - interpret BTAD primarily as an open-set support-family retrieval problem:
    - `status_accuracy`, `known_failure_recall`, `unknown_anomaly_recall`, and `novel_as_known_rate` are the primary metrics
    - `known_label_accuracy` is secondary because the labels are support-family identifiers rather than named defect semantics

### MVTec bottle

Nominal calibration comes from a held-out subset of `train/good`.

Run:

```bash
.venv/bin/python hybrid_memory_demo/run_mvtec_bottle_demo.py \
  --mvtec-root data/mvtec \
  --category bottle \
  --device cpu \
  --out outputs/hybrid_memory_demo/mvtec_bottle
```

### BTAD

Run:

```bash
.venv/bin/python hybrid_memory_demo/run_btad_demo.py \
  --btad-root data/btad \
  --device cpu \
  --out outputs/hybrid_memory_demo/btad_components_demo
```

### Launch the browser UI

Use the manifest to load both artifacts in one app:

```bash
.venv/bin/python hybrid_memory_demo/web.py \
  --manifest-json hybrid_memory_demo/demo_manifest.json \
  --device cpu
```

The viewer now supports two inference modes:

- `Use Selected Dataset`
  - score the image with the currently selected artifact only
- `Use Learned Nominal Router`
  - compare the image against all loaded nominal profiles
  - expose the learned router suggestion
  - choose the active profile by best normalized nominal fit across the loaded artifacts

Open:

```text
http://127.0.0.1:8765
```

## Generic usage on your own data

The cleanest path is the folder-based CLI:

```bash
.venv/bin/python hybrid_memory_demo/fit_folder_hybrid_memory.py \
  --data-root /path/to/your_hybrid_dataset \
  --out outputs/hybrid_memory_demo/your_line_v1 \
  --device cpu
```

Expected layout:

```text
your_hybrid_dataset/
  nominal/
    train/
    calibration/
  failures/
    crack/
    scratch/
    contamination/
```

The lower-level training API in `hybrid_memory_demo/pipeline.py` expects:

- `nominal_train_paths`: good images used to build the nominal bank
- `nominal_calibration_paths`: good images from a separate split/day for thresholding
- `labeled_failure_paths`: dictionary of `failure_label -> list[image paths]`

That is the shape you want in production if operators can upload new negative/failure classes over time.

For the full process, see:

- `hybrid_memory_demo/SETUP_KNOWN_FAILURES.md`

## Practical guidance

- Use at least 2-4 support images per known failure class if you want unknown-vs-known gating to work well.
- Keep failure supports visually consistent with deployment cameras.
- Rebuild thresholds when adding a materially new failure family.
- Treat each product/camera setup as its own artifact unless you have evidence of transfer.
- The app now renders three complementary views per query:
  - input image
  - threshold-aware anomaly heatmap
  - embedding-space heatmap relative to the nearest retrieved failure memory
- BTAD should be treated carefully: the labels available to this demo are ok/ko rather than named defect families, so the BTAD demo labels are component-conditioned anomaly families rather than defect semantics.
- The viewer now surfaces the protocol explicitly:
  - MVTec uses `Predicted Label`
  - BTAD uses `Predicted Support Family`
  - the dataset summary cards show the evaluation protocol and open-set metrics for the selected artifact
