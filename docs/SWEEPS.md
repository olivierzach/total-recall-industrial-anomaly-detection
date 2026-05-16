# Sweeps

This repo includes a sweep runner for PatchCore on BTAD.

## What the sweep varies
By default, `scripts/sweep_btad.py` runs a small grid over:
- backbone ∈ {vit_b_16, wide_resnet50_2} *(unless you specify `--backbone`)*
- image_size ∈ {224, 256} *(unless specified)*
- coreset_ratio ∈ {0.0005, 0.001} *(unless specified)*
- num_neighbors (k) ∈ {1, 5}
- image_score ∈ {max, mean}
- l2_normalize ∈ {true} (toggle via `--no-l2-normalize`)

Each run writes a JSON artifact; the sweep writes a markdown summary table.

## Run via Make

Smoke (fast iteration):

```bash
make sweep_btad_smoke DEVICE=mps
```

Full dataset:

```bash
make sweep_btad_full DEVICE=mps
```

## Run via Python

```bash
python3 scripts/sweep_btad.py --btad-root data/btad --device mps --outdir outputs/sweeps/btad_full
```

## Outputs

In the sweep output directory:
- `summary.md` — markdown table sorted by PRO (then image AUROC)
- `summary.json` — machine-readable rows
- `btad_<...>.json` — one artifact per run
