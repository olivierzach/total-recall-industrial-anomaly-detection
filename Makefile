.PHONY: help test sweep_btad_smoke sweep_btad_full

PY ?= .venv/bin/python

help:
	@echo "Targets:"
	@echo "  test              - run pytest in the local .venv"
	@echo "  sweep_btad_smoke  - small sweep with caps for quick iteration"
	@echo "  sweep_btad_full   - full dataset sweep (can take a while)"
	@echo ""
	@echo "Variables:"
	@echo "  PY=<path>         - python executable (default: .venv/bin/python)"

# Defaults can be overridden: make sweep_btad_smoke DEVICE=cpu
DEVICE ?= mps
BTAD_ROOT ?= data/btad
BATCH ?= 16
NUM_WORKERS ?= 0

test:
	$(PY) -m pytest -q

sweep_btad_smoke:
	$(PY) scripts/sweep_btad.py \
	  --btad-root $(BTAD_ROOT) \
	  --device $(DEVICE) \
	  --batch $(BATCH) \
	  --num-workers $(NUM_WORKERS) \
	  --outdir outputs/sweeps/btad_smoke \
	  --max-train 256 --max-test 256

sweep_btad_full:
	$(PY) scripts/sweep_btad.py \
	  --btad-root $(BTAD_ROOT) \
	  --device $(DEVICE) \
	  --batch $(BATCH) \
	  --num-workers $(NUM_WORKERS) \
	  --outdir outputs/sweeps/btad_full
