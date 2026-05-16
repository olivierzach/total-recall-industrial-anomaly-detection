from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.collate import collate_batch
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.learned_router import fit_linear_router
from src.patchcore.patchcore import to_numpy


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class NominalRouterConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    layers: tuple[str, ...] = ("layer3",)
    image_size: int = 224
    l2_normalize: bool = True
    router_iters: int = 200
    router_l2: float = 1.0
    router_lr: float = 0.1


@dataclass(frozen=True)
class NominalRouterArtifact:
    cfg: NominalRouterConfig
    labels: list[str]
    label_display_names: dict[str, str]
    router_W: np.ndarray
    router_b: np.ndarray
    centroids: np.ndarray
    backbone_state: dict | None
    artifact_info: dict | None


class _LabeledPathDataset(Dataset):
    def __init__(self, pairs: list[tuple[str | Path, int]], transform):
        self.rows = [(Path(path), int(label)) for path, label in pairs]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, label = self.rows[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)

        class Item:
            pass

        item = Item()
        item.image = tensor
        item.label = label
        item.mask = None
        item.path = str(path)
        return item


def iter_image_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [root_path]
    return [p for p in sorted(root_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _extract_image_embeddings(
    *,
    pairs: list[tuple[str | Path, int]],
    cfg: NominalRouterConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = _LabeledPathDataset(pairs, transform=_make_transform(cfg.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch,
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    rows: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            emb, _ = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=batch.image.to(device),
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)
            rows.extend(emb_np.mean(axis=1).astype(np.float32, copy=False))
            labels.extend(batch.label.tolist())
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int64), backbone.state_dict()


def fit_nominal_router(
    *,
    label_to_paths: dict[str, list[str | Path]],
    label_display_names: dict[str, str] | None = None,
    cfg: NominalRouterConfig,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 0,
    artifact_info: dict | None = None,
) -> NominalRouterArtifact:
    labels = sorted([label for label, paths in label_to_paths.items() if paths])
    if len(labels) < 2:
        raise ValueError("Nominal router requires at least two non-empty label groups")

    pairs: list[tuple[str | Path, int]] = []
    for idx, label in enumerate(labels):
        pairs.extend((path, idx) for path in label_to_paths[label])

    features, y, backbone_state = _extract_image_embeddings(
        pairs=pairs,
        cfg=cfg,
        device=torch.device(device),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    router = fit_linear_router(
        features,
        y,
        iters=int(cfg.router_iters),
        l2=float(cfg.router_l2),
        lr=float(cfg.router_lr),
        rng=np.random.default_rng(int(seed)),
    )
    centroids = np.stack([features[y == idx].mean(axis=0) for idx in range(len(labels))], axis=0).astype(np.float32, copy=False)

    info = dict(artifact_info or {})
    preds = np.argmax(features @ router.W.T + router.b[None, :], axis=1)
    info.update(
        {
            "n_images": int(features.shape[0]),
            "labels": labels,
            "train_accuracy": float(np.mean(preds == y)),
            "seed": int(seed),
        }
    )

    display = dict(label_display_names or {})
    for label in labels:
        display.setdefault(label, label)

    return NominalRouterArtifact(
        cfg=cfg,
        labels=labels,
        label_display_names=display,
        router_W=router.W.astype(np.float32, copy=False),
        router_b=router.b.astype(np.float32, copy=False),
        centroids=centroids,
        backbone_state=backbone_state,
        artifact_info=info,
    )


def save_nominal_router(out_dir: str | Path, artifact: NominalRouterArtifact) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.json").write_text(json.dumps(asdict(artifact.cfg), indent=2, sort_keys=True))
    (out_path / "labels.json").write_text(
        json.dumps({"labels": artifact.labels, "label_display_names": artifact.label_display_names}, indent=2, sort_keys=True)
    )
    np.save(out_path / "router_W.npy", artifact.router_W.astype(np.float32, copy=False))
    np.save(out_path / "router_b.npy", artifact.router_b.astype(np.float32, copy=False))
    np.save(out_path / "centroids.npy", artifact.centroids.astype(np.float32, copy=False))
    if artifact.backbone_state is not None:
        torch.save(artifact.backbone_state, out_path / "backbone_state.pt")
    if artifact.artifact_info is not None:
        (out_path / "artifact_info.json").write_text(json.dumps(artifact.artifact_info, indent=2, sort_keys=True))


def load_nominal_router(model_dir: str | Path) -> NominalRouterArtifact:
    model_path = Path(model_dir)
    cfg = NominalRouterConfig(**json.loads((model_path / "config.json").read_text()))
    labels_payload = json.loads((model_path / "labels.json").read_text())
    state_path = model_path / "backbone_state.pt"
    backbone_state = None
    if state_path.exists():
        try:
            backbone_state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            backbone_state = torch.load(state_path, map_location="cpu")
    info = None
    info_path = model_path / "artifact_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    return NominalRouterArtifact(
        cfg=cfg,
        labels=list(labels_payload["labels"]),
        label_display_names=dict(labels_payload.get("label_display_names", {})),
        router_W=np.load(model_path / "router_W.npy").astype(np.float32, copy=False),
        router_b=np.load(model_path / "router_b.npy").astype(np.float32, copy=False),
        centroids=np.load(model_path / "centroids.npy").astype(np.float32, copy=False),
        backbone_state=backbone_state,
        artifact_info=info,
    )


class NominalRouterRuntime:
    def __init__(self, artifact: NominalRouterArtifact, *, device: str = "cpu"):
        self.artifact = artifact
        self.device = torch.device(device)
        self.backbone = load_backbone(artifact.cfg.backbone, pretrained=artifact.cfg.pretrained and artifact.backbone_state is None)
        if artifact.backbone_state is not None:
            self.backbone.load_state_dict(artifact.backbone_state)
        self.backbone = self.backbone.to(self.device)
        self.hooks = None if is_vit_backbone(artifact.cfg.backbone) else FeatureHooks(self.backbone, list(artifact.cfg.layers))
        self.transform = _make_transform(artifact.cfg.image_size)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb, _ = extract_patch_embeddings(
                backbone_name=self.artifact.cfg.backbone,
                model=self.backbone,
                hooks=self.hooks,
                x=x,
                layers=self.artifact.cfg.layers,
                l2_normalize=self.artifact.cfg.l2_normalize,
                return_hw=True,
            )
        patch_embeddings = to_numpy(emb[0])
        return patch_embeddings.mean(axis=0).astype(np.float32, copy=False)

    def route_image(self, image: Image.Image) -> dict:
        embedding = self._embed_image(image)
        logits = embedding[None, :] @ self.artifact.router_W.T + self.artifact.router_b[None, :]
        logits = logits[0]
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / np.maximum(np.sum(probs), 1e-8)
        order = np.argsort(-probs)
        ranked = [
            {
                "rank": int(rank + 1),
                "label": self.artifact.labels[int(index)],
                "display_name": self.artifact.label_display_names.get(self.artifact.labels[int(index)], self.artifact.labels[int(index)]),
                "probability": float(probs[int(index)]),
                "logit": float(logits[int(index)]),
            }
            for rank, index in enumerate(order.tolist())
        ]
        return {
            "predicted_label": ranked[0]["label"],
            "predicted_display_name": ranked[0]["display_name"],
            "probabilities": ranked,
        }
