# GPU acceleration on Apple Silicon

This repo can use Apple Silicon GPUs via **PyTorch MPS**.

## Why MPS (not MLX)
- MLX is great for transformer experimentation, but torchvision backbones + the broader PyTorch ecosystem are easiest via **MPS**.
- PatchCore does not “train” a deep net; it mostly does **feature extraction** and kNN. MPS accelerates the feature extraction step.

## Requirements
- Apple Silicon Mac
- PyTorch built with MPS support
- A terminal session that can access Metal

## Environment Checks

MPS availability depends on both the PyTorch build and the active execution environment. Check both before launching longer jobs.

Quick check:

```bash
python3 - <<'PY'
import torch
print('mps_available', torch.backends.mps.is_available())
print('mps_built', torch.backends.mps.is_built())
PY
```

Recommended environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip
pip install torch torchvision scikit-learn
```

## Usage

Just pass `--device mps`.

Example (BTAD smoke):

```bash
python3 scripts/eval_btad_patchcore.py \
  --btad-root data/btad \
  --device mps \
  --backbone vit_b_16 \
  --image-size 224 \
  --coreset-ratio 0.0005 \
  --max-train 256 --max-test 256 \
  --out outputs/btad_smoke_vit_mps.json
```

MVTec smoke on Apple Silicon:

```bash
.venv311/bin/python scripts/eval_mvtec_patchcore.py \
  --mvtec-root data/mvtec \
  --category bottle \
  --backbone resnet18 \
  --image-size 224 \
  --device mps \
  --seed 0 \
  --num-workers 0 \
  --max-train 32 --max-test 16 \
  --log-every 2 \
  --out outputs/mvtec_bottle_mps_smoke_resnet18_eval.json
```

Longer runs are best launched in `tmux`:

```bash
tmux new-session -d -s mvtec_bottle_mps \
  ".venv311/bin/python scripts/eval_mvtec_patchcore.py \
    --mvtec-root data/mvtec \
    --category bottle \
    --backbone resnet18 \
    --image-size 224 \
    --device mps \
    --seed 0 \
    --num-workers 0 \
    --log-every 5 \
    --out outputs/mvtec_bottle_mps_eval.json \
    2>&1 | tee outputs/logs/mvtec_bottle_mps_eval.log"
```

## Compatibility

MPS is compatible with the repo paths that run feature extraction through PyTorch and accept `--device`:
- `scripts/fit_nominal_patchcore.py`
- `scripts/fit_mvtec_patchcore.py`
- `scripts/eval_mvtec_patchcore.py`
- `scripts/eval_btad_patchcore.py`
- `scripts/score_images.py`
- `scripts/review_dataset_examples.py`

Important note:
- The backbone forward pass runs on MPS.
- kNN, coreset, JSON/report generation, and most plotting remain CPU-side (`numpy` / `scikit-learn` / `matplotlib`).
- So MPS speeds up the expensive feature-extraction stage, but it does not move the entire pipeline to GPU.

## Notes
- kNN/coreset are CPU-side (numpy/sklearn). The heavy part is feature extraction, which MPS accelerates.
- If you want full GPU kNN, the next step is adding a FAISS backend or torch-based kNN.
