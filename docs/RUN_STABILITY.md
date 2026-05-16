# Run stability (avoiding SIGKILL / long-run failure)

This repo contains a few operations that can take a long time on large memory banks:
- naive k-center greedy coreset selection (O(kN))
- repeated evaluation runs across many variants

If your agent runner or environment kills long-running processes (SIGKILL), use these patterns.

## 1) Avoid parallel heavy runs

Do not run multiple full evals at once (e.g. cosine + PCA variants in parallel). Run sequentially.

Reason: you can saturate CPU/RAM and get killed.

## 2) Use cached memory banks for ablations

Both `eval_mvtec_patchcore.py` and `eval_btad_patchcore.py` support:
- `--cache-memory`

This caches the computed coreset memory bank (post-PCA) to `<out>.memory_cache.npz`.
Subsequent ablation runs with the same config can reuse it and skip the expensive coreset step.

## 3) Use a fast coreset baseline when iterating

Use:
- `--coreset-method random`

This gives you a fast baseline that still exercises the full pipeline.
Then, when ready, switch back to `kcenter`.

## 4) Reduce memory/compute knobs

- prefer `--layers layer3` (fewer patches)
- reduce `--image-size`
- reduce `--coreset-ratio`
- consider PCA with `--pca-dim` to speed distance computations

## 5) If k-center is the bottleneck

For production-scale runs, consider:
- switching to a faster sampler (k-means prototypes)
- implementing ANN-backed coreset selection / FAISS

(We treat k-center as a correctness baseline, not an always-on production primitive.)
