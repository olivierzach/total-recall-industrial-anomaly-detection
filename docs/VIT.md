# ViT backbone (PatchCore + Vision Transformer)

PatchCore is a *pipeline*; the backbone can be swapped.

This repo supports using a torchvision ViT backbone (e.g. `vit_b_16`) in a self-contained way.

## Caveat
Our current implementation extracts **feature maps via forward hooks on CNN layers** (e.g. `layer2`, `layer3`).
ViTs don't expose the same intermediate 2D feature maps by default.

So today we support ViT **as a backbone download + model loading option**, but to use it properly we still need
an embedding extractor that:
- takes ViT token embeddings
- reshapes patch tokens into an H×W grid
- optionally concatenates multiple block outputs

## Why still include it now?
- Sets up the dependency + checkpoint handling.
- Makes it easy to add ViT patch-token extraction next without refactoring config/CLI.

## Planned implementation (next)
- Add `src/patchcore/vit_embedding.py`:
  - run ViT forward
  - extract patch tokens `[B, N_patches, D]`
  - reshape to `[B, H, W, D]` and return `[B, H*W, D]`
- Update scripts to automatically choose CNN vs ViT extraction.

