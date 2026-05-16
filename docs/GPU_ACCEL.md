# GPU acceleration on Mac mini (Apple Silicon)

This repo can use the Mac mini GPU via **PyTorch MPS**.

## Why MPS (not MLX)
- MLX is great for transformer experimentation, but torchvision backbones + the broader PyTorch ecosystem are easiest via **MPS**.
- PatchCore does not “train” a deep net; it mostly does **feature extraction** and kNN. MPS accelerates the feature extraction step.

## Requirements
- Apple Silicon Mac
- PyTorch built with MPS support

Quick check:

```bash
python3 - <<'PY'
import torch
print('mps_available', torch.backends.mps.is_available())
print('mps_built', torch.backends.mps.is_built())
PY
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

## Notes
- kNN/coreset are CPU-side (numpy/sklearn). The heavy part is feature extraction, which MPS accelerates.
- If you want full GPU kNN, the next step is adding a FAISS backend or torch-based kNN.
