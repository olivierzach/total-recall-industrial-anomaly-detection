# Related work — industrial anomaly detection (vision)

Core setting: **nominal-only training**, detect + localize anomalies at test time.

## Patch-based embedding + kNN family
- **PatchCore** (Roth et al., CVPR 2022): https://arxiv.org/abs/2106.08265
- **PaDiM** (Defard et al., ICPR 2021): probabilistic modeling over patch embeddings
- **SPADE** (Cohen & Hoshen, 2020): patch distribution estimation / nearest neighbors

## Feature reconstruction / distillation
- **STFPM** (Wang et al., ICCV 2021): student–teacher feature pyramid matching
- **Reverse Distillation** (Deng & Li, CVPR 2022): distill from teacher to student (reverse)
- **DRAEM** (Zavrtanik et al., ICCV 2021): synthetic anomaly + reconstruction

## Normalizing flows / density models
- **FastFlow** (Yu et al., 2021/2022): normalizing flow on features for anomaly maps

## Recent efficiency-focused
- **EfficientAD** (2023): lightweight student/teacher for speed
- **SimpleNet** (2023): simplification + strong baseline results

## Benchmarks / datasets
- **MVTec AD**: classic industrial anomaly detection dataset
- **VisA** (Vision Anomaly): larger dataset with realistic anomalies
- **MPDD**: metal defect dataset

## Evaluation metrics
- Image-level AUROC
- Pixel-level AUROC
- PRO (per-region overlap) used heavily in MVTec literature
- Operational metrics for QA: recall@fixed-FPR, expected inspections per hour

## Notes / practical cautions
- Many papers optimize AUROC; QA often needs calibrated thresholds and tail-risk control.
- Beware dataset leakage: per-object splits, transform differences, test-time normalization.
- kNN memory banks scale; coreset sampling + ANN (FAISS) are practical levers.

