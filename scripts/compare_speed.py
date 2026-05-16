#!/usr/bin/env python3
"""Compare timing breakdowns between two eval outputs.

Example:
  python3 scripts/compare_speed.py outputs/speed_cpu_vit_smoke.json outputs/speed_mps_vit_smoke.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str):
    o = json.loads(Path(p).read_text())
    return o.get("timing", {}), o


def fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:8.3f}"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_speed.py <cpu.json> <mps.json>")

    t0, o0 = load(sys.argv[1])
    t1, o1 = load(sys.argv[2])

    keys = sorted(set(t0) | set(t1))
    print(f"left={sys.argv[1]}\nright={sys.argv[2]}")
    print(f"backbone={o0.get('cfg',{}).get('backbone')} image_size={o0.get('cfg',{}).get('image_size')} max_train/max_test used")
    print("\nmetric                cpu_s     mps_s   speedup")
    print("-------------------  --------  --------  -------")
    for k in keys:
        a = t0.get(k)
        b = t1.get(k)
        sp = None
        if a is not None and b is not None and b > 0:
            sp = a / b
        sp_s = f"{sp:7.2f}x" if sp is not None else "   -  "
        print(f"{k:19s}  {fmt(a)}  {fmt(b)}  {sp_s}")


if __name__ == "__main__":
    main()
