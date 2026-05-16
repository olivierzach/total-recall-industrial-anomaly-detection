# Datasets

## MVTec AD

PatchCore is commonly evaluated on **MVTec AD**.

Official links:
- Dataset page: https://www.mvtec.com/company/research/datasets/mvtec-ad
- Downloads page: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

### Directory layout expected by this repo

After extraction, we expect:

```
<MVTEC_ROOT>/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*_mask.png
  cable/
  capsule/
  ...
```

### Download

MVTec AD has licensing/terms and is typically downloaded from the official MVTec website.
The official downloads page currently provides both the full archive and per-category archives.

This repo provides a helper script that can:
- download from a **user-provided URL** (e.g. your internal mirror / signed URL), and/or
- unpack an existing local archive.

Run:

```bash
python3 scripts/mvtec_get.py --help
```

### Example: official `bottle` category URL

If you want a smaller end-to-end smoke/eval path, the official `bottle` archive can be downloaded directly with:

```bash
python3 scripts/mvtec_get.py \
  --url https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz \
  --download-to data/raw/bottle.tar.xz \
  --out data/mvtec
```

Then fit/evaluate:

```bash
python3 scripts/fit_mvtec_patchcore.py \
  --mvtec-root data/mvtec \
  --category bottle \
  --device cpu \
  --seed 0 \
  --out outputs/models/bottle

python3 scripts/eval_mvtec_patchcore.py \
  --mvtec-root data/mvtec \
  --category bottle \
  --device cpu \
  --seed 0 \
  --out outputs/mvtec_bottle_eval.json
```

## BTAD (easy-access alternative)

BTAD (BeanTech Anomaly Detection Dataset) can be downloaded without login via DatasetNinja/Supervisely hosting.

```bash
python3 scripts/btad_get.py --out data/btad
```

This fetches a tar in Supervisely project format (see `docs/BTAD.md`). By default we store the downloaded archive under `data/raw/` and extract into `data/btad/`.

### Disk hygiene

If you already have a complete extracted dataset under `data/btad/` (e.g., `data/btad/train/img`, `data/btad/test/img`, etc.), you can delete the large raw archive (commonly named like `data/raw/btad-DatasetNinja.tar`) and re-fetch it later by re-running `scripts/btad_get.py`.

Recommended workflow:
1) Download the dataset archive manually (or from your company mirror).
2) Place it somewhere like `data/raw/mvtec_ad.zip`.
3) Extract to `data/mvtec`:

```bash
python3 scripts/mvtec_get.py --archive data/raw/mvtec_ad.zip --out data/mvtec
```

Then evaluate:

```bash
python3 scripts/eval_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu
```
