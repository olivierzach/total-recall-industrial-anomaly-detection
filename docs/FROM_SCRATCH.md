# From-scratch setup + running on your own nominal images

This is the practical path for getting PatchCore running on a **new computer** and then fitting a memory bank on **your own nominal (good) images**.

## 0) Prereqs
- Python 3.11+ (3.12 recommended)
- git

Optional (Mac GPU): Apple Silicon + PyTorch MPS (see `docs/GPU_ACCEL.md`).
On the tested Mac mini setup, the most reliable MPS path was a dedicated Python 3.11 env run from a normal shell / `tmux`.

## 1) Install

CPU / general install:

```bash
git clone <REPO_URL>
cd total-recall-industrial-anomaly-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Mac mini GPU (tested path):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip
pip install torch torchvision scikit-learn
pip install -e . --no-deps
```

## 2) Fit a model (build a nominal memory bank)

PatchCore is “nominal-only”: you build a memory bank from **good** examples.

Assume you have:
- `~/data/nominal_dishes/` containing good dish images (`.jpg/.png/...`)

Run:

```bash
.venv311/bin/python scripts/fit_nominal_patchcore.py \
  --nominal ~/data/nominal_dishes \
  --out outputs/models/dishes_camA \
  --device mps \
  --backbone vit_b_16 --image-size 224 \
  --coreset-ratio 0.02
```

What you get:
- `outputs/models/dishes_camA/` which includes:
  - the PatchCore config (backbone, layers, image_size, etc.)
  - the fitted memory bank (coreset embeddings)
  - the exact backbone weights used for scoring

Notes:
- The first run will auto-download pretrained backbone weights via torchvision.
- If you change backbone/image_size/layers, you must rebuild the memory bank.

## 3) Score new images

```bash
.venv311/bin/python scripts/score_images.py \
  --model outputs/models/dishes_camA \
  --images ~/data/new_dishes_batch \
  --device mps \
  --out outputs/dishes_scores.jsonl \
  --save-maps outputs/dishes_maps
```

This writes:
- `outputs/dishes_scores.jsonl` (one JSON per image with an anomaly score)
- `outputs/dishes_maps/*.anomaly_patchgrid.npy` (patch-grid anomaly maps for triage)

## 4) Visualize anomaly maps

See `scripts/viz_anomaly_maps.py`.

## Troubleshooting

- If `--device mps` errors, fall back to `--device cpu`.
- If `mps_built=True` but `mps_available=False`, verify you are running from a normal local shell / `tmux`; the sandboxed agent runtime on this machine could not access MPS even though the hardware supported it.
- If you’re on Linux with an NVIDIA GPU, use `--device cuda`.
