# Paper

- Title: Towards Total Recall in Industrial Anomaly Detection
- Alias: PatchCore
- arXiv: https://arxiv.org/abs/2106.08265
- CVPR 2022 PDF: https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf

## Checklist for faithful implementation
- backbone (paper default)
- which layers used for patch embeddings
- patch embedding concatenation + normalization
- coreset sampling (k-Center Greedy)
- distance metric (L2 in embedding space)
- image scoring aggregation
- anomaly map upsampling/smoothing
- evaluation: AUROC (image/pixel), PRO

