# Datasets

## MVTec AD

PatchCore is commonly evaluated on **MVTec AD**.

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

This repo provides a helper script that can:
- download from a **user-provided URL** (e.g. your internal mirror / signed URL), and/or
- unpack an existing local archive.

Run:

```bash
python3 scripts/mvtec_get.py --help
```

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
