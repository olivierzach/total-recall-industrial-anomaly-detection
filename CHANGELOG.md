# Changelog

This is a human-readable changelog for the repository.

Rationale: git history is the source of truth, but a curated log helps when you’re iterating fast and want a narrative of what changed and why.

## Unreleased

- Add BTAD sweep runner (`scripts/sweep_btad.py`) and expose more eval knobs in `eval_btad_patchcore.py`.

## 2026-03-11

- Implement PRO metric (region-based localization metric) and integrate into BTAD eval output.
- Add dev fast-iteration documentation and commands.
- Add Mac mini GPU acceleration notes (PyTorch MPS) and a measured benchmark + speed compare tool.
- Add docs for from-scratch setup and fitting/scoring on your own nominal image folder.
- Add learned reference selection plan doc.
