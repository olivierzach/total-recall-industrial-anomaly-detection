# BTAD (BeanTech Anomaly Detection Dataset)

BTAD is a smaller industrial anomaly detection dataset that is often easier to access than MVTec.

References:
- DatasetNinja overview: https://datasetninja.com/btad
- DatasetNinja repo: https://github.com/dataset-ninja/btad
- Kaggle mirror (may require Kaggle auth): https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection

## Plan for this repo

- Provide `scripts/btad_get.py` similar to `mvtec_get.py`:
  - download from a user-provided URL or use a local archive
  - extract into `data/btad/`
- Implement a loader under `src/data/btad.py` with a similar API to the MVTec loader.
- Add an eval script `scripts/eval_btad_patchcore.py` that outputs:
  - image-level AUROC
  - pixel-level AUROC (if masks are present)
  - PRO (optional)

