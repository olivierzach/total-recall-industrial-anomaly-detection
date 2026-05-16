# IVF (Inverted File) indexing for fast similarity search (PatchCore / memory banks)

This note explains IVF-style retrieval and how it maps to PatchCore’s memory-bank kNN.

## Problem

PatchCore-style scoring needs repeated nearest-neighbor queries:
- For a test image, we embed patches into vectors `x_i ∈ R^D`.
- For each patch, we compute distance to the nearest nominal memory vector `m_j`.

Naively, this is O(P·N·D) per image (P patches, N memory vectors). That can be too slow for on-the-line use.

## IVF idea (partition + search only relevant partitions)

IVF is a standard approximate nearest neighbor (ANN) scheme:

1) **Train a coarse quantizer** (usually k-means) on the database vectors.
   - centroids `c_1..c_K`

2) **Assign each database vector** to its nearest centroid.
   - this creates K *inverted lists* (buckets)

3) **At query time**, pick the top `nprobe` closest centroids (multi-probe).

4) Search only the vectors in those selected lists.

So total comparisons drop from `N` to roughly `N * (nprobe / K)` (plus overhead). If K is large and nprobe small, huge speedup.

### Why multi-probe matters
If you probe only 1 centroid, a query near a boundary might miss its true nearest neighbor.
Multi-probe (nprobe>1) mitigates this at small extra cost.

## Mapping to this repo

We implement the *IVF routing concept* (partition + multi-probe) in `docs/QUERY_ADAPTIVE_MEMORY.md`:
- **Patch routing**: cluster patch embeddings; route each query patch.
- **Image routing**: cluster per-image embeddings; route once per query image, then search only patches from those routed images.

These are IVF-like. The current implementation:
- uses k-means to build partitions
- uses brute-force search within the routed candidate set

This is already useful for:
- lowering comparisons
- keeping “product/component modes” separate so queries search in relevant nominal buckets

## Production-grade IVF (FAISS)

For real on-the-line latency, you typically use a high-performance ANN library such as FAISS.

Common FAISS index types:
- `IndexFlatL2` / `IndexFlatIP`: exact (still can be fast with BLAS, but O(N))
- `IndexIVFFlat`: IVF partitioning + exact scan within lists
- `IndexIVFPQ`: IVF + product quantization (smaller, faster, approximate)
- `HNSW`: graph-based ANN; often very fast with good recall

IVF knobs:
- `nlist` (K): number of partitions
- `nprobe` (probes): how many partitions to scan at query time

Rule of thumb:
- bigger `nlist` → smaller lists, faster per-probe scans, but requires higher `nprobe` for recall.

## Latency model (back-of-envelope)

Total time per image roughly =
- `T_backbone` (embed the image)
- + `T_routing` (choose clusters)
- + `T_search` (distance computations to candidates)

Routing is typically negligible.
The dominant cost is (a) backbone forward and (b) search.

If you’re CPU-bound, a smaller backbone or fewer layers may matter more than ANN.
If you’re search-bound (large memory bank), IVF/FAISS is the lever.

## Gotchas

- IVF is approximate; always measure recall/false negatives at your operating point.
- If you route too aggressively (small `nprobe`), you can create false anomalies by missing true nearest nominal neighbors.
- If you have multiple products, routing can be *more* important than coreset: it prevents irrelevant nominal modes from contaminating distances.

## Recommended next steps

1) Keep the current routing baseline for correctness and experimentation.
2) Add a FAISS backend for the candidate search step.
3) Evaluate trade-offs at an operational metric (recall@fixed-FPR + latency).
