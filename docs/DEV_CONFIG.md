# Dev config (fast iteration path)

Goal: run the full end-to-end pipeline in **minutes**, not hours, while keeping a "faithful" path for final numbers.

## Why this exists
Full PatchCore evaluation can be slow on CPU because feature extraction dominates.
A fast dev path lets you iterate on:
- data loading and transforms
- memory bank construction
- kNN scoring
- metrics code (image/pixel/PRO)
- tooling (overlays, calibration)

## Recommended dev defaults

### Backbone
Use a smaller backbone for dev iteration:
- `resnet18`

### Image size
- 224 (works for ViT) or 256 (CNN)

### Coreset
Start tiny, then scale:
- `coreset_ratio = 0.0005` (very small coreset; good for smoke)

### Caps
During development:
- `--max-train 256`
- `--max-test 256`

## Example commands

### Quick smoke on BTAD

```bash
python3 scripts/eval_btad_patchcore.py \
  --btad-root data/btad \
  --backbone resnet18 \
  --image-size 256 \
  --coreset-ratio 0.0005 \
  --max-train 256 --max-test 256 \
  --log-every 10 \
  --out outputs/btad_smoke_resnet18.json
```

### Faithful-ish baseline

```bash
python3 scripts/eval_btad_patchcore.py \
  --btad-root data/btad \
  --backbone wide_resnet50_2 \
  --image-size 256 \
  --coreset-ratio 0.0005 \
  --out outputs/btad_wrn50_full.json
```

### ViT baseline

```bash
python3 scripts/eval_btad_patchcore.py \
  --btad-root data/btad \
  --backbone vit_b_16 \
  --image-size 224 \
  --coreset-ratio 0.0005 \
  --out outputs/btad_vitb16_full.json
```
