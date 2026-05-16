# Multi-bank nominal routing demo: MVTec bottle + BTAD

Goal: demonstrate a simple multi-product setup:

- maintain **separate nominal banks** per product/dataset
- route each incoming image to the closest bank (in embedding space)
- score using that bank

This is a minimal proof-of-concept for a "single visualization app" that can score
multiple products correctly by selecting the right nominal reference.

## Models

We require both banks to share the same backbone/layers/image_size.

- Bottle bank (MVTec bottle nominal):
  - `outputs/models/bottle_layer3_kmeans_c01`
  - trained on: `data/mvtec/bottle/train/good`
  - backbone: `wide_resnet50_2`
  - layers: `layer3`
  - coreset: `kmeans`, ratio=0.01

- BTAD bank (BTAD nominal):
  - `outputs/models/btad_nominal_kmeans_c01_layer3`
  - trained on: `data/btad/train/img`
  - backbone: `wide_resnet50_2`
  - layers: `layer3`
  - coreset: `kmeans`, ratio=0.01

## Routing rule

For a query image:
1) compute patch embeddings
2) take the mean patch embedding (image embedding)
3) choose the bank whose **memory centroid** has highest cosine similarity

This is intentionally simple.

## Run

```bash
.venv/bin/python scripts/score_images_multibank.py \
  --models bottle=outputs/models/bottle_layer3_kmeans_c01 btad=outputs/models/btad_nominal_kmeans_c01_layer3 \
  --images data/mvtec/bottle/test data/btad/test/img \
  --device mps \
  --out outputs/multibank_scores_bottle_btad.jsonl
```

## Results

Routing counts (sanity check):
- total images scored: 824
- MVTec bottle test images (83/83) routed to **bottle** bank
- BTAD test images (741/741) routed to **btad** bank

So for this two-domain toy demo, routing is perfectly separating the sources.

## Caveats

- This is not a claim of general multi-product performance. It's a sanity test.
- We routed between two *very different* domains; real products may be less separable.
- In production you likely have an explicit product ID; learned routing is for when you don't.
