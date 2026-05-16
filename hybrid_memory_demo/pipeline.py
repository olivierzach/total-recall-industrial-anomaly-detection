from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hybrid_memory_demo.model import (
    FailureDecision,
    HybridMemoryArtifact,
    HybridMemoryConfig,
    HybridPrediction,
    SupportRecord,
    build_failure_descriptor,
    calibrate_anomaly_threshold,
    calibrate_known_failure_threshold,
    classify_failure_descriptor,
)
from src.data.collate import collate_batch
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.utils.image_viz import overlay_heatmap, upsample_score_map


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def iter_image_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [root_path]
    return [p for p in sorted(root_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def compute_embedding_similarity_map(
    patch_embeddings: np.ndarray,
    hw: tuple[int, int],
    reference_descriptor: np.ndarray,
) -> np.ndarray:
    if patch_embeddings.ndim != 2:
        raise ValueError(f"Expected [P,D] patch embeddings, got shape {patch_embeddings.shape}")
    height, width = int(hw[0]), int(hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid patch grid: {hw}")
    if patch_embeddings.shape[0] != height * width:
        raise ValueError("Patch count does not match requested spatial grid")
    if reference_descriptor.ndim != 1:
        raise ValueError("reference_descriptor must be rank-1")
    if reference_descriptor.shape[0] < patch_embeddings.shape[1]:
        raise ValueError("reference_descriptor is smaller than the patch embedding dim")

    ref = reference_descriptor[: patch_embeddings.shape[1]].astype(np.float32, copy=False)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm <= 0.0:
        return np.zeros((height, width), dtype=np.float32)

    emb = patch_embeddings.astype(np.float32, copy=False)
    emb_norm = np.linalg.norm(emb, axis=1)
    denom = np.maximum(emb_norm * ref_norm, 1e-8)
    similarity = np.sum(emb * ref[None, :], axis=1) / denom
    return similarity.reshape(height, width).astype(np.float32, copy=False)


def compute_patch_embedding_projection(
    patch_embeddings: np.ndarray,
    patch_scores: np.ndarray,
    nominal_memory: np.ndarray,
    *,
    nominal_sample_size: int = 512,
    top_k_patches: int = 24,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if patch_embeddings.ndim != 2:
        raise ValueError(f"Expected [P,D] patch embeddings, got shape {patch_embeddings.shape}")
    if patch_scores.ndim != 1:
        raise ValueError(f"Expected [P] patch scores, got shape {patch_scores.shape}")
    if patch_embeddings.shape[0] != patch_scores.shape[0]:
        raise ValueError("patch_embeddings and patch_scores must have matching patch counts")
    if nominal_memory.ndim != 2:
        raise ValueError("nominal_memory must be rank-2")
    if nominal_memory.shape[1] != patch_embeddings.shape[1]:
        raise ValueError("nominal_memory and patch_embeddings must share feature dimension")

    rng = np.random.default_rng(int(seed))
    if nominal_memory.shape[0] > int(nominal_sample_size):
        idx = rng.choice(nominal_memory.shape[0], size=int(nominal_sample_size), replace=False)
        nominal_sample = nominal_memory[idx]
    else:
        nominal_sample = nominal_memory

    combined = np.concatenate([nominal_sample, patch_embeddings], axis=0).astype(np.float32, copy=False)
    centered = combined - combined.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        nominal_proj = np.zeros((nominal_sample.shape[0], 2), dtype=np.float32)
        query_proj = np.zeros((patch_embeddings.shape[0], 2), dtype=np.float32)
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[: min(2, vh.shape[0])].T
        proj = centered @ basis
        if proj.shape[1] < 2:
            proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])), mode="constant")
        nominal_proj = proj[: nominal_sample.shape[0]].astype(np.float32, copy=False)
        query_proj = proj[nominal_sample.shape[0] :].astype(np.float32, copy=False)

    k = max(1, min(int(top_k_patches), int(patch_embeddings.shape[0])))
    top_idx = np.argsort(-patch_scores)[:k]
    top_mask = np.zeros(patch_embeddings.shape[0], dtype=bool)
    top_mask[top_idx] = True
    return {
        "nominal_projection": nominal_proj,
        "query_projection": query_proj,
        "top_query_mask": top_mask,
    }


def _make_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _PathDataset(Dataset):
    def __init__(self, paths: Iterable[str | Path], transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)

        class Item:
            pass

        item = Item()
        item.image = tensor
        item.label = 0
        item.mask = None
        item.path = str(path)
        return item


def _patchcore_cfg(cfg: HybridMemoryConfig):
    from src.patchcore.config import PatchCoreConfig

    return PatchCoreConfig(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        layers=tuple(cfg.layers),
        image_size=int(cfg.image_size),
        l2_normalize=bool(cfg.l2_normalize),
        coreset_ratio=float(cfg.coreset_ratio),
        num_neighbors=int(cfg.num_neighbors),
        image_score=str(cfg.image_score),
    )


def _extract_batches(
    *,
    paths: list[str | Path],
    cfg: HybridMemoryConfig,
    backbone,
    hooks: FeatureHooks | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[tuple[str, np.ndarray, tuple[int, int]]]:
    dataset = _PathDataset(paths, transform=_make_transform(cfg.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch,
    )

    rows: list[tuple[str, np.ndarray, tuple[int, int]]] = []
    with torch.no_grad():
        for batch in dataloader:
            emb, hw = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=batch.image.to(device),
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)
            for path, item_emb in zip(batch.path, emb_np):
                rows.append((str(path), item_emb, hw))
    return rows


def fit_hybrid_memory(
    *,
    nominal_train_paths: list[str | Path],
    nominal_calibration_paths: list[str | Path],
    labeled_failure_paths: dict[str, list[str | Path]],
    cfg: HybridMemoryConfig,
    device: str = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 0,
    artifact_info: dict | None = None,
) -> HybridMemoryArtifact:
    if not nominal_train_paths:
        raise ValueError("nominal_train_paths is empty")
    if not nominal_calibration_paths:
        raise ValueError("nominal_calibration_paths is empty")

    rng = np.random.default_rng(int(seed))
    torch_device = torch.device(device)
    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(torch_device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    nominal_rows = _extract_batches(
        paths=nominal_train_paths,
        cfg=cfg,
        backbone=backbone,
        hooks=hooks,
        device=torch_device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    nominal_bank = np.concatenate([emb for _, emb, _ in nominal_rows], axis=0)
    indices = KCenterGreedy().select(nominal_bank, ratio=cfg.coreset_ratio, rng=rng)
    nominal_memory = nominal_bank[indices].astype(np.float32, copy=False)
    nominal_model = PatchCoreModel.fit(_patchcore_cfg(cfg), nominal_memory)

    calib_rows = _extract_batches(
        paths=nominal_calibration_paths,
        cfg=cfg,
        backbone=backbone,
        hooks=hooks,
        device=torch_device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    calib_scores = np.array(
        [float(nominal_model.score_image(emb)) for _, emb, _ in calib_rows],
        dtype=np.float32,
    )
    anomaly_threshold = calibrate_anomaly_threshold(calib_scores, cfg.anomaly_quantile)

    support_records: list[SupportRecord] = []
    failure_descriptors: list[np.ndarray] = []
    for label, paths in sorted(labeled_failure_paths.items()):
        if not paths:
            continue
        failure_rows = _extract_batches(
            paths=paths,
            cfg=cfg,
            backbone=backbone,
            hooks=hooks,
            device=torch_device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for path, emb, _ in failure_rows:
            patch_scores = nominal_model.score_patches(emb)
            descriptor = build_failure_descriptor(
                emb,
                patch_scores,
                top_k_patches=cfg.failure_top_k_patches,
            )
            failure_descriptors.append(descriptor)
            support_records.append(
                SupportRecord(
                    label=label,
                    path=str(path),
                    anomaly_score=float(np.max(patch_scores)),
                )
            )

    failure_array = np.stack(failure_descriptors, axis=0) if failure_descriptors else np.zeros((0, 0), dtype=np.float32)
    known_failure_threshold, margin_threshold, failure_stats = calibrate_known_failure_threshold(
        failure_array,
        [rec.label for rec in support_records],
        quantile=cfg.known_failure_quantile,
        min_margin_ratio=cfg.min_margin_ratio,
    ) if len(support_records) >= 2 else (float("inf"), float(cfg.min_margin_ratio), {})

    info = dict(artifact_info or {})
    info.update(
        {
            "seed": int(seed),
            "nominal_train_images": len(nominal_train_paths),
            "nominal_calibration_images": len(nominal_calibration_paths),
            "failure_support_images": len(support_records),
            "calibration_score_p50": float(np.median(calib_scores)),
            "calibration_score_p95": float(np.quantile(calib_scores, 0.95)),
            "failure_stats": failure_stats,
        }
    )

    return HybridMemoryArtifact(
        cfg=cfg,
        nominal_memory=nominal_memory,
        failure_descriptors=failure_array.astype(np.float32, copy=False),
        support_records=support_records,
        anomaly_threshold=float(anomaly_threshold),
        known_failure_threshold=float(known_failure_threshold),
        margin_threshold=float(margin_threshold),
        backbone_state=backbone.state_dict(),
        artifact_info=info,
    )


def save_artifact(out_dir: str | Path, artifact: HybridMemoryArtifact) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.json").write_text(json.dumps(asdict(artifact.cfg), indent=2, sort_keys=True))
    np.save(out_path / "nominal_memory.npy", artifact.nominal_memory.astype(np.float32, copy=False))
    np.save(out_path / "failure_descriptors.npy", artifact.failure_descriptors.astype(np.float32, copy=False))
    (out_path / "support_records.json").write_text(
        json.dumps([asdict(record) for record in artifact.support_records], indent=2, sort_keys=True)
    )
    (out_path / "thresholds.json").write_text(
        json.dumps(
            {
                "anomaly_threshold": artifact.anomaly_threshold,
                "known_failure_threshold": artifact.known_failure_threshold,
                "margin_threshold": artifact.margin_threshold,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if artifact.backbone_state is not None:
        torch.save(artifact.backbone_state, out_path / "backbone_state.pt")
    if artifact.artifact_info is not None:
        (out_path / "artifact_info.json").write_text(json.dumps(artifact.artifact_info, indent=2, sort_keys=True))


def load_artifact(model_dir: str | Path) -> HybridMemoryArtifact:
    model_path = Path(model_dir)
    cfg = HybridMemoryConfig(**json.loads((model_path / "config.json").read_text()))
    nominal_memory = np.load(model_path / "nominal_memory.npy")
    failure_descriptors = np.load(model_path / "failure_descriptors.npy")
    support_records = [SupportRecord(**row) for row in json.loads((model_path / "support_records.json").read_text())]
    thresholds = json.loads((model_path / "thresholds.json").read_text())
    state_path = model_path / "backbone_state.pt"
    backbone_state = None
    if state_path.exists():
        try:
            backbone_state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            backbone_state = torch.load(state_path, map_location="cpu")
    artifact_info = None
    info_path = model_path / "artifact_info.json"
    if info_path.exists():
        artifact_info = json.loads(info_path.read_text())

    return HybridMemoryArtifact(
        cfg=cfg,
        nominal_memory=nominal_memory.astype(np.float32, copy=False),
        failure_descriptors=failure_descriptors.astype(np.float32, copy=False),
        support_records=support_records,
        anomaly_threshold=float(thresholds["anomaly_threshold"]),
        known_failure_threshold=float(thresholds["known_failure_threshold"]),
        margin_threshold=float(thresholds["margin_threshold"]),
        backbone_state=backbone_state,
        artifact_info=artifact_info,
    )


class HybridMemoryRuntime:
    def __init__(self, artifact: HybridMemoryArtifact, *, device: str = "cpu"):
        self.artifact = artifact
        self.device = torch.device(device)
        self.backbone = load_backbone(artifact.cfg.backbone, pretrained=artifact.cfg.pretrained and artifact.backbone_state is None)
        if artifact.backbone_state is not None:
            self.backbone.load_state_dict(artifact.backbone_state)
        self.backbone = self.backbone.to(self.device)
        self.hooks = None if is_vit_backbone(artifact.cfg.backbone) else FeatureHooks(self.backbone, list(artifact.cfg.layers))
        self.nominal_model = PatchCoreModel.fit(_patchcore_cfg(artifact.cfg), artifact.nominal_memory)
        self.transform = _make_transform(artifact.cfg.image_size)

    def _extract(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb, hw = extract_patch_embeddings(
                backbone_name=self.artifact.cfg.backbone,
                model=self.backbone,
                hooks=self.hooks,
                x=x,
                layers=self.artifact.cfg.layers,
                l2_normalize=self.artifact.cfg.l2_normalize,
                return_hw=True,
            )
        return to_numpy(emb[0]), hw

    def predict_image(self, image: Image.Image) -> tuple[HybridPrediction, np.ndarray]:
        prediction, score_map, _, _, _ = self.predict_image_with_diagnostics(image)
        return prediction, score_map

    def predict_image_with_diagnostics(
        self,
        image: Image.Image,
    ) -> tuple[HybridPrediction, np.ndarray, np.ndarray | None, dict | None, dict[str, np.ndarray]]:
        patch_embeddings, hw = self._extract(image)
        patch_scores = self.nominal_model.score_patches(patch_embeddings)
        if self.artifact.cfg.image_score == "mean":
            anomaly_score = float(np.mean(patch_scores))
        else:
            anomaly_score = float(np.max(patch_scores))

        descriptor = build_failure_descriptor(
            patch_embeddings,
            patch_scores,
            top_k_patches=self.artifact.cfg.failure_top_k_patches,
        )
        failure_decision = classify_failure_descriptor(
            descriptor,
            self.artifact.failure_descriptors,
            self.artifact.support_records,
            neighbors=self.artifact.cfg.classifier_neighbors,
            known_failure_threshold=self.artifact.known_failure_threshold,
            margin_threshold=self.artifact.margin_threshold,
        )
        prediction = self._compose_prediction(anomaly_score, failure_decision)
        score_map = patch_scores.reshape(hw).astype(np.float32, copy=False)
        embedding_projection = compute_patch_embedding_projection(
            patch_embeddings,
            patch_scores,
            self.artifact.nominal_memory,
            top_k_patches=self.artifact.cfg.failure_top_k_patches,
        )
        embedding_map = None
        embedding_reference = None
        if failure_decision.neighbors:
            best_neighbor = failure_decision.neighbors[0]
            support_index = int(best_neighbor.get("support_index", -1))
            if 0 <= support_index < len(self.artifact.support_records):
                embedding_map = compute_embedding_similarity_map(
                    patch_embeddings,
                    hw,
                    self.artifact.failure_descriptors[support_index],
                )
                embedding_reference = {
                    "label": best_neighbor["label"],
                    "path": best_neighbor["path"],
                    "distance": float(best_neighbor["distance"]),
                    "mode": "cosine_similarity_to_nearest_support_salient_descriptor",
                }
        return prediction, score_map, embedding_map, embedding_reference, embedding_projection

    def predict_path(self, path: str | Path) -> tuple[HybridPrediction, np.ndarray]:
        return self.predict_image(Image.open(path).convert("RGB"))

    def predict_bytes(self, data: bytes) -> tuple[HybridPrediction, np.ndarray, Image.Image]:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        prediction, score_map = self.predict_image(image)
        return prediction, score_map, image

    def predict_bytes_with_diagnostics(
        self,
        data: bytes,
    ) -> tuple[HybridPrediction, np.ndarray, np.ndarray | None, dict | None, dict[str, np.ndarray], Image.Image]:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        prediction, score_map, embedding_map, embedding_reference, embedding_projection = self.predict_image_with_diagnostics(image)
        return prediction, score_map, embedding_map, embedding_reference, embedding_projection, image

    def render_overlay(self, image: Image.Image, score_map: np.ndarray) -> Image.Image:
        upsampled = upsample_score_map(score_map, image.size)
        shifted = np.maximum(upsampled - float(self.artifact.anomaly_threshold), 0.0)
        if float(np.max(shifted)) <= 0.0:
            return image.convert("RGB")
        return overlay_heatmap(image, shifted, alpha=0.45)

    def render_embedding_map(self, image: Image.Image, embedding_map: np.ndarray | None) -> Image.Image:
        width, height = image.size
        if embedding_map is None:
            return Image.new("RGB", (width, height), color=(18, 24, 30))

        upsampled = upsample_score_map(embedding_map, (width, height))
        smin = float(np.min(upsampled))
        smax = float(np.max(upsampled))
        if smax > smin:
            norm = (upsampled - smin) / (smax - smin)
        else:
            norm = np.zeros_like(upsampled, dtype=np.float32)
        return overlay_heatmap(Image.new("RGB", (width, height), color=(24, 30, 36)), norm, alpha=0.9)

    def render_embedding_space(self, projection: dict[str, np.ndarray], *, size: tuple[int, int] = (480, 480)) -> Image.Image:
        width, height = int(size[0]), int(size[1])
        canvas = Image.new("RGB", (width, height), color=(14, 20, 26))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width - 1, height - 1), outline=(52, 70, 86), width=1)

        nominal = projection.get("nominal_projection", np.zeros((0, 2), dtype=np.float32))
        query = projection.get("query_projection", np.zeros((0, 2), dtype=np.float32))
        top_mask = projection.get("top_query_mask", np.zeros((0,), dtype=bool))

        combined = [arr for arr in [nominal, query] if arr.size > 0]
        if not combined:
            return canvas
        all_points = np.concatenate(combined, axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)

        def to_xy(points: np.ndarray) -> list[tuple[float, float]]:
            if points.size == 0:
                return []
            norm = (points - mins[None, :]) / spans[None, :]
            xs = 24 + norm[:, 0] * (width - 48)
            ys = 24 + (1.0 - norm[:, 1]) * (height - 48)
            return list(zip(xs.tolist(), ys.tolist()))

        for x, y in to_xy(nominal):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(88, 104, 122))

        query_xy = to_xy(query)
        for idx, (x, y) in enumerate(query_xy):
            color = (255, 159, 67) if idx < len(top_mask) and bool(top_mask[idx]) else (101, 196, 255)
            radius = 3 if idx < len(top_mask) and bool(top_mask[idx]) else 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        draw.text((16, 10), "Nominal memory patches", fill=(136, 152, 170))
        draw.text((16, 28), "Query patches", fill=(101, 196, 255))
        draw.text((16, 46), "Top anomalous query patches", fill=(255, 159, 67))
        return canvas

    def _compose_prediction(self, anomaly_score: float, failure_decision: FailureDecision) -> HybridPrediction:
        if anomaly_score < self.artifact.anomaly_threshold:
            status = "normal"
            predicted_label = None
        elif failure_decision.is_known_failure:
            status = "known_failure"
            predicted_label = failure_decision.predicted_label
        else:
            status = "unknown_anomaly"
            predicted_label = None
        return HybridPrediction(
            status=status,
            anomaly_score=float(anomaly_score),
            anomaly_threshold=float(self.artifact.anomaly_threshold),
            predicted_label=predicted_label,
            best_failure_distance=failure_decision.best_distance,
            failure_margin_ratio=failure_decision.margin_ratio,
            is_known_failure=bool(failure_decision.is_known_failure),
            nearest_failures=failure_decision.neighbors,
        )
