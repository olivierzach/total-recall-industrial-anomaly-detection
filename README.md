# Towards Total Recall in Industrial Anomaly Detection (PatchCore) — local implementation

This repo is a practical, local re-implementation + experimentation harness for:

- **Roth et al. (CVPR 2022)**: *Towards Total Recall in Industrial Anomaly Detection* (a.k.a. **PatchCore**)
  - arXiv: https://arxiv.org/abs/2106.08265
  - CVF PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf
  - Code/datasets page: https://www.amazon.science/code-and-datasets/towards-total-recall-in-industrial-anomaly-detection

## What we’re building

- A clean PatchCore baseline (feature extraction → memory bank → coreset sampling → NN scoring)
- Reproducible evaluation on common benchmarks (start with **MVTec AD**)
- A set of ablations and knobs for “industrial QA” settings:
  - sensitivity/recall under rare defects
  - latency / memory trade-offs
  - domain shifts, new products, few-shot updates

## Quickstart

- New machine + your own nominal images: `docs/FROM_SCRATCH.md`
- Fast dev iteration: `docs/DEV_CONFIG.md`
- Mac mini GPU (MPS): `docs/GPU_ACCEL.md`
- Sweeps (BTAD): `docs/SWEEPS.md`
- Datasets + official MVTec links: `docs/DATASETS.md`

Mac mini note:
- MPS works on the tested Apple Silicon machine when launched from a normal shell / `tmux`.
- The sandboxed agent runtime may report `mps_available=False` even though the machine itself supports MPS.

## Next steps

Start here:
- `docs/PLAN.md`
- `docs/RELATED_WORK.md`
