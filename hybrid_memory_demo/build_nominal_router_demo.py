from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_memory_demo.router import NominalRouterConfig, fit_nominal_router, save_nominal_router


def _sample(paths: list[Path], cap: int, seed: int) -> list[Path]:
    rows = list(paths)
    if int(cap) <= 0 or len(rows) <= int(cap):
        return rows
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    return rows[: int(cap)]


def _btad_nominal(btad_root: Path) -> list[Path]:
    return [path for path in sorted((btad_root / "train" / "img").glob("*")) if path.is_file() and "_ok_" in path.name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", default="data/mvtec")
    ap.add_argument("--mvtec-category", default="bottle")
    ap.add_argument("--btad-root", default="data/btad")
    ap.add_argument("--out", default="outputs/hybrid_memory_demo/nominal_router_v1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-label", type=int, default=256)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--layers", nargs="*", default=["layer3"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--router-iters", type=int, default=300)
    ap.add_argument("--router-l2", type=float, default=1.0)
    ap.add_argument("--router-lr", type=float, default=0.1)
    args = ap.parse_args()

    mvtec_nominal = sorted((Path(args.mvtec_root) / args.mvtec_category / "train" / "good").glob("*"))
    btad_nominal = _btad_nominal(Path(args.btad_root))
    label_to_paths = {
        "mvtec_bottle": _sample(mvtec_nominal, int(args.max_per_label), int(args.seed)),
        "btad_components": _sample(btad_nominal, int(args.max_per_label), int(args.seed) + 1),
    }
    cfg = NominalRouterConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        router_iters=int(args.router_iters),
        router_l2=float(args.router_l2),
        router_lr=float(args.router_lr),
    )
    artifact = fit_nominal_router(
        label_to_paths=label_to_paths,
        label_display_names={
            "mvtec_bottle": "MVTec AD / bottle",
            "btad_components": "BTAD / component-conditioned defects",
        },
        cfg=cfg,
        device=args.device,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        artifact_info={
            "datasets": {label: len(paths) for label, paths in label_to_paths.items()},
            "mvtec_category": args.mvtec_category,
        },
    )
    save_nominal_router(args.out, artifact)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "labels": artifact.labels,
                "train_accuracy": artifact.artifact_info.get("train_accuracy"),
                "datasets": artifact.artifact_info.get("datasets"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
