# GPU acceleration on Mac mini (Apple Silicon)

This repo can use the Mac mini GPU via **PyTorch MPS**.

## Why MPS (not MLX)
- MLX is great for transformer experimentation, but torchvision backbones + the broader PyTorch ecosystem are easiest via **MPS**.
- PatchCore does not “train” a deep net; it mostly does **feature extraction** and kNN. MPS accelerates the feature extraction step.

## Requirements
- Apple Silicon Mac
- PyTorch built with MPS support
- A normal local shell / tmux session that can access Metal

## What we found on the Mac mini

Tested machine:
- Apple M4 Pro Mac mini
- macOS 15.3.1

Observed behavior:
- `torch.backends.mps.is_built() == True` in both Python 3.14 and Python 3.11 environments
- `torch.backends.mps.is_available() == False` inside the sandboxed agent runtime
- `torch.backends.mps.is_available() == True` outside the sandbox in a normal shell / `tmux` session using a fresh Python 3.11 venv

Conclusion:
- The Mac mini and PyTorch build are MPS-capable.
- The blocker was the **execution environment**, not the hardware.
- For reliable MPS runs on this machine, use a normal terminal or `tmux`, not the sandboxed agent runtime.

Quick check:

```bash
python3 - <<'PY'
import torch
print('mps_available', torch.backends.mps.is_available())
print('mps_built', torch.backends.mps.is_built())
PY
```

Recommended tested environment on this machine:

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

MVTec smoke on the Mac mini GPU:

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
