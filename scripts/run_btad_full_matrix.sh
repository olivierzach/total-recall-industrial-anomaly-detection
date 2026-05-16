#!/usr/bin/env bash
set -euo pipefail

BTAD_ROOT=${BTAD_ROOT:-data/btad}
DEVICE=${DEVICE:-mps}
BATCH=${BATCH:-16}
NUM_WORKERS=${NUM_WORKERS:-0}
CORESET_RATIO=${CORESET_RATIO:-0.0005}

OUTDIR=${OUTDIR:-outputs/matrix_btad}
mkdir -p "$OUTDIR"

# A small but useful baseline matrix. Expand as needed.
BACKBONES=(
  "vit_b_16:224"
  "wide_resnet50_2:256"
  "resnet18:256"
)

for spec in "${BACKBONES[@]}"; do
  bb=${spec%%:*}
  sz=${spec##*:}
  out="$OUTDIR/btad_${bb}_img${sz}_coreset${CORESET_RATIO}_${DEVICE}.json"
  echo "[run] backbone=$bb image_size=$sz device=$DEVICE coreset=$CORESET_RATIO -> $out"
  python3 scripts/eval_btad_patchcore.py \
    --btad-root "$BTAD_ROOT" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --num-workers "$NUM_WORKERS" \
    --coreset-ratio "$CORESET_RATIO" \
    --backbone "$bb" \
    --image-size "$sz" \
    --out "$out"
done

echo "[done] wrote results to $OUTDIR"