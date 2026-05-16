#!/usr/bin/env python3
"""Calibrate an anomaly-score threshold on a nominal-only reference set.

For QA usage you often want: choose threshold s.t. FPR ~= target on nominal images.

Example:
  # Score a nominal calibration directory
  python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/nominal_calib --out outputs/calib_scores.jsonl

  # Choose threshold at 0.1% FPR
  python3 scripts/calibrate_threshold.py --scores outputs/calib_scores.jsonl --target-fpr 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="JSONL from score_images.py")
    ap.add_argument("--target-fpr", type=float, default=0.001)
    args = ap.parse_args()

    scores = []
    with Path(args.scores).open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            scores.append(float(o["score"]))

    if not scores:
        raise SystemExit("no scores")

    scores = np.asarray(scores)
    q = 1.0 - float(args.target_fpr)
    thr = float(np.quantile(scores, q))

    out = {
        "n": int(scores.shape[0]),
        "target_fpr": float(args.target_fpr),
        "quantile": q,
        "threshold": thr,
        "score_min": float(scores.min()),
        "score_med": float(np.median(scores)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(scores.max()),
    }

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
