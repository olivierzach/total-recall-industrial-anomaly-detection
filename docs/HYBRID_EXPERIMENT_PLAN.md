# Hybrid Known-Failure Benchmark Plan

## Goal

Turn the hybrid known-failure idea into a **measurable experiment** rather than a demo-only concept.

The benchmark question is:

> Compared with a nominal-only PatchCore-style detector, does adding a known-failure memory bank improve operational usefulness while preserving open-set behavior on unseen defect families?

## Local data available now

### Primary benchmark dataset

- `data/mvtec/bottle`

Why this is the cleanest benchmark in the current repo:

- it is already present locally
- it has three explicit defect families
  - `broken_large`
  - `broken_small`
  - `contamination`
- it has a clean normal train/test split

### Why BTAD is not the primary benchmark here

BTAD is present locally, but in the current repo state it is not yet the best dataset for this hybrid benchmark because the local setup does not expose a clean fine-grained failure taxonomy comparable to MVTec bottle’s named defect families. BTAD remains useful for exploratory out-of-domain probing in the UI.

## Protocol

### Nominal split

- `train/good`
  - split into:
    - `nominal_train`
    - `nominal_calibration`

### Defect protocol

Use **leave-one-defect-family-out** evaluation.

For MVTec bottle, run 3 folds:

1. unknown = `broken_large`, known = `broken_small`, `contamination`
2. unknown = `broken_small`, known = `broken_large`, `contamination`
3. unknown = `contamination`, known = `broken_large`, `broken_small`

Within each fold:

- take `k` support images per known class for the known-failure memory bank
- evaluate on:
  - all test `good` images -> `normal`
  - remaining images from known classes -> `known_failure`
  - all images from held-out class -> `unknown_anomaly`

## Baseline and hybrid systems

### Baseline

- nominal-only PatchCore-style detector
- outputs:
  - `normal`
  - `unknown_anomaly`
- cannot emit a known failure class

### Hybrid

- same nominal detector
- plus labeled known-failure bank
- outputs:
  - `normal`
  - `known_failure`
  - `unknown_anomaly`

## Metrics

### Anomaly detection

- image AUROC

### Operational 3-state decision

- `status_accuracy`
- `normal_recall`
- `known_failure_recall`
- `unknown_anomaly_recall`

### Known-class recognition

- `known_label_accuracy`
- `known_label_accuracy_when_predicted_known`

### Failure mode safety

- `novel_as_known_rate`
  - unseen failures incorrectly forced into known classes
- `normal_false_alarm_rate`
  - normal images incorrectly escalated

## Ablations

The first benchmark ablation in this repo is:

- support images per known class: `1, 2, 4, 8`

This asks the most practical first question:

- how much labeled defect memory do we need before the hybrid system becomes useful?

## Runnable implementation

The benchmark is implemented in:

- `hybrid_memory_demo/benchmark.py`
- `hybrid_memory_demo/benchmark_mvtec_hybrid.py`

Run:

```bash
.venv/bin/python hybrid_memory_demo/benchmark_mvtec_hybrid.py \
  --mvtec-root data/mvtec \
  --category bottle \
  --device cpu \
  --out outputs/hybrid_benchmark/mvtec_bottle
```

Artifacts written:

- `protocol.json`
- `runs.json`
- `summary.json`
- `summary.md`

Current local benchmark output path:

- `outputs/hybrid_benchmark/mvtec_bottle/summary.md`

## Current local snapshot

Executed on the local `data/mvtec/bottle` copy with:

- backbone: `resnet18`
- image size: `160`
- coreset ratio: `0.05`
- leave-one-defect-family-out over:
  - `broken_large`
  - `broken_small`
  - `contamination`

Current mean results across the 3 folds:

| Method | Support/Class | Image AUROC | Status Acc | Normal Recall | Known Recall | Unknown Recall | Known Label Acc | Novel->Known | False Alarm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_nominal_only | 0 | 1.000 | 0.482 | 0.950 | 0.000 | 1.000 | 0.000 | 0.000 | 0.050 |
| hybrid_known_failure_bank | 1 | 1.000 | 0.658 | 0.950 | 0.751 | 0.209 | 0.417 | 0.791 | 0.050 |
| hybrid_known_failure_bank | 2 | 1.000 | 0.679 | 0.950 | 0.745 | 0.303 | 0.472 | 0.697 | 0.050 |
| hybrid_known_failure_bank | 4 | 1.000 | 0.671 | 0.950 | 0.725 | 0.317 | 0.568 | 0.683 | 0.050 |
| hybrid_known_failure_bank | 8 | 1.000 | 0.706 | 0.950 | 0.821 | 0.336 | 0.654 | 0.664 | 0.050 |

Interpretation:

- The hybrid bank clearly improves `known_failure_recall` and `known_label_accuracy` over the nominal-only baseline.
- On the current implementation, that gain comes with a large `novel_as_known_rate`, meaning many held-out defect families are still being over-forced into known classes.
- So the local data already supports the central hypothesis that the hybrid idea is useful, but it also exposes the main research problem: **better open-set rejection on unseen defect families**.

## Interpretation guidance

The hybrid method should be considered successful if it improves:

- `known_failure_recall`
- `known_label_accuracy`

without badly degrading:

- `unknown_anomaly_recall`
- `normal_false_alarm_rate`

The main failure mode to watch is:

- high `novel_as_known_rate`

That would mean the added failure memory bank is over-classifying novel defects instead of preserving open-set behavior.

## Next experiments after this benchmark

Once the first benchmark is stable, the next research-grade ablations should be:

1. prototype bank vs instance bank
2. support descriptor variants
3. class-specific thresholds vs global threshold
4. backbone swap: `resnet18` vs stronger pretrained features
5. day/shift/camera split when real plant data is available
