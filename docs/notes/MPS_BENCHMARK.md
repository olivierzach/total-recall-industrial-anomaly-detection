# MPS acceleration benchmark (BTAD)

This doc records measured speedups from using **PyTorch MPS** on the Mac mini.

## Setup
- Dataset: BTAD
- Backbone: vit_b_16
- image_size=224
- coreset_ratio=0.0005
- max_train=256, max_test=256 (smoke)

## Results

We ran the same command twice, changing only `--device`.

### Total wall time
- CPU: **80.12s**
- MPS: **55.73s**
- Overall speedup: **1.44×**

### Timing breakdown (seconds)
| stage | CPU | MPS | speedup |
|---|---:|---:|---:|
| feature_train_s | 14.42 | 3.60 | 4.01× |
| feature_test_s  | 13.07 | 4.44 | 2.94× |
| coreset_s       | 1.62  | 1.03 | 1.58× |
| knn_fit_s       | 0.035 | 0.001 | 28× *(tiny absolute)* |
| metric_s        | 10.91 | 10.83 | 1.01× |

Interpretation:
- MPS accelerates **feature extraction** substantially (3–4× here).
- Overall end-to-end speedup is smaller because **metrics/PRO** are CPU-bound.

## Command lines

CPU:
```bash
python3 scripts/eval_btad_patchcore.py --btad-root data/btad --device cpu --batch 16 --num-workers 0 \
  --coreset-ratio 0.0005 --backbone vit_b_16 --image-size 224 \
  --max-train 256 --max-test 256 --log-every 10 \
  --out outputs/speed_cpu_vit_smoke.json
```

MPS:
```bash
python3 scripts/eval_btad_patchcore.py --btad-root data/btad --device mps --batch 16 --num-workers 0 \
  --coreset-ratio 0.0005 --backbone vit_b_16 --image-size 224 \
  --max-train 256 --max-test 256 --log-every 10 \
  --out outputs/speed_mps_vit_smoke.json
```

