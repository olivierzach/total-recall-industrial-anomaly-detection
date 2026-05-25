# Usage

## 1) Get MVTec AD

See `docs/DATASETS.md`.

Quickstart (after you have an archive):

```bash
python3 scripts/mvtec_get.py --archive data/raw/mvtec_ad.zip --out data/mvtec
```

## 2) Fit a PatchCore memory bank

### On your own nominal folder (recommended)

```bash
python3 scripts/fit_nominal_patchcore.py --nominal /path/to/nominal --device cpu --seed 0 --out outputs/models/my_product
```

### On MVTec (benchmark)

```bash
python3 scripts/fit_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu --seed 0 --out outputs/models/bottle
```

## 3) Evaluate on MVTec test split

```bash
python3 scripts/eval_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu --coreset-ratio 0.1
```

## 4) Score new images (QA-style pipeline)

```bash
python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/images --out outputs/scores.jsonl
```

Optionally also save patch-grid anomaly maps for triage:

```bash
python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/images \
  --out outputs/scores.jsonl --save-maps outputs/maps

python3 scripts/viz_anomaly_maps.py --images /path/to/images --maps outputs/maps --out outputs/overlays
```

Calibrate an operational threshold from nominal calibration images:

```bash
python3 scripts/calibrate_threshold.py --scores outputs/scores.jsonl --target-fpr 0.001 --out outputs/threshold.json
python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/images --threshold outputs/threshold.json --out outputs/scored_with_decisions.jsonl
```

Output JSONL example:
```json
{"path":"/path/to/img.png","score":1.234}
```

Optional explanation output for the highest-scoring patches:
```bash
python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/images \
  --top-k-neighbors 3 --top-k-patches 3 --out outputs/scores_with_explanations.jsonl
```

## 5) Gather nominal images from the web (optional)

Only do this when you have rights to use the images. Avoid scraping sources whose terms forbid it.

Workflow:
1) In chat, use web search to find a few relevant product gallery pages.
2) Feed those page URLs into:

```bash
python3 scripts/fetch_nominal_web_images.py --pages <url1> <url2> --out data/web_nominal/my_product --limit 300
```

Then fit:

```bash
python3 scripts/fit_nominal_patchcore.py --nominal data/web_nominal/my_product --out outputs/models/my_product
```

See also `docs/DEV_CONFIG.md` for a fast-iteration path.
See also `docs/GPU_ACCEL.md` for running on Apple Silicon GPUs via PyTorch MPS.

## 6) Run tests

```bash
python3 -m pip install -e .[dev]
pytest -q
```

## 7) Plot and review experiments

Aggregate eval JSONs and sweep summaries into comparable plots:

```bash
python3 scripts/plot_experiments.py outputs --outdir outputs/reports/all_runs
```

Extract qualitative examples from a labeled dataset split:

```bash
python3 scripts/review_dataset_examples.py \
  --dataset mvtec --root data/mvtec --category bottle \
  --model outputs/models/bottle \
  --threshold outputs/threshold.json \
  --select fp --top-n 12 \
  --outdir outputs/review/bottle_fp
```
