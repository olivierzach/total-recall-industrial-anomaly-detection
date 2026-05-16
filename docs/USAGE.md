# Usage

## 1) Get MVTec AD

See `docs/DATASETS.md`.

Quickstart (after you have an archive):

```bash
python3 scripts/mvtec_get.py --archive data/raw/mvtec_ad.zip --out data/mvtec
```

## 2) Fit a PatchCore memory bank

```bash
python3 scripts/fit_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu --out outputs/models/bottle
```

## 3) Evaluate on MVTec test split

```bash
python3 scripts/eval_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu --coreset-ratio 0.1
```

## 4) Score new images (QA-style pipeline)

```bash
python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/images --out outputs/scores.jsonl
```

Output JSONL example:
```json
{"path":"/path/to/img.png","score":1.234}
```

## 5) Run tests

```bash
python3 -m pip install -e .[dev]
pytest -q
```
