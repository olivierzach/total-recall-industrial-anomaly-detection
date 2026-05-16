# Baseline ladder (what to build in what order)

Goal: a sequence of baselines where each step adds one idea, so gains are attributable and you can defend the approach.

## Ladder

### Level 0 — sanity baselines
- Pixel/template baselines (cheap but weak):
  - SSIM vs nominal template(s)
  - per-pixel mean/std z-score (if alignment is strong)

Deliverable: proves the data pipeline works and shows how hard the task is.

### Level 1 — pretrained embedding distance (no coreset)
- Extract patch embeddings from a pretrained backbone.
- Score by distance to nominal patches (exact kNN against full memory bank).

Deliverable: shows the “feature space” idea works before adding sampling/engineering.

### Level 2 — PatchCore baseline
- Add:
  - coreset subsampling (k-center greedy / approx)
  - embedding aggregation / smoothing
  - image-level scoring via max/top-k

Deliverable: the first real PatchCore model.

### Level 3 — calibration + operational evaluation
- Fix the operating point:
  - choose threshold at fixed FPR / inspection budget
  - report recall, precision, and stability

Deliverable: turns a research metric into a QA system.

### Level 4 — efficiency (if needed)
Pick based on bottleneck:
- ANN index (FAISS) for faster retrieval
- smaller backbone / fewer layers
- distillation (e.g., EfficientAD-style) for speed

Deliverable: meets latency/throughput constraints.

### Level 5 — defect lookup / known failure mode classification (optional)
- Add a defect memory bank + classifier head / retrieval for known failures.

Deliverable: not just “anomaly,” but “likely failure mode.”

## How to present this ladder

- Show one plot/table per rung:
  - performance (AUROC + recall@fixed-FPR)
  - latency (ms/image)
  - memory (MB)
- Show representative heatmaps + top errors at each rung.

If each rung is crisp, you’ll come across as rigorous even without prior vision track record.
