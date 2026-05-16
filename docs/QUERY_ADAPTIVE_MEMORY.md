# Query-adaptive memory / learned-ish sample selection (IVF-style)

Motivation: PatchCore’s coreset is a *global* subset that must cover all nominal modes.
If your nominal set contains different products/components/modes, a global coreset can feel wasteful.

Instead, we can do **query-adaptive selection**:
- given a query embedding (patch or image), pick a *small candidate subset* of nominal memory to search.

This is analogous to RAG: retrieve candidates, then do the expensive comparison only inside the candidate set.

## What we implement

We implement an IVF-like routing index using k-means clustering.

### Patch routing
- Cluster nominal *patch embeddings* into K clusters.
- For each query patch, select top-`probes` closest clusters.
- Search kNN only within the union of those cluster members.

### Image routing
- Compute a per-image nominal embedding (mean of patch embeddings).
- Cluster nominal *image embeddings* into K clusters.
- For a query image, select top-`probes` image clusters.
- Search patches only from the nominal images routed to those clusters.

This addresses: “if I have different products/components, search within the relevant nominal bucket.”

## Training

Use:
- `scripts/fit_nominal_patchcore_routed.py`

Artifacts saved:
- Standard PatchCore artifact (`memory_bank.npy`, config, backbone state, optional PCA)
- `routing_state.npz` and `routing_state.json` containing centroids and membership lists.

## Scoring

Use:
- `scripts/score_images_routed.py`

Routing modes:
- `--routing patch`
- `--routing image`

Key knob:
- `--probes r` (multi-probe; helps avoid cluster-boundary misses)

## Notes and caveats

- This is primarily a **speed/efficiency** method, but it can also reduce false positives when the query is compared only against the right nominal mode.
- Too few clusters or too few probes can cause *missed nearest neighbors* (false anomalies). Multi-probe is the fix.
- The current implementation uses a small dependency-free k-means (Lloyd). For huge banks, use FAISS or MiniBatchKMeans.

## Next step

Add an evaluation script that compares:
- global coreset vs query-adaptive routing
- latency/memory trade-offs
- recall@fixed-FPR
