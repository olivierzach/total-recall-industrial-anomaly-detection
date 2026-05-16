from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hybrid_memory_demo.model import HybridMemoryConfig
from hybrid_memory_demo.model import build_failure_descriptor, calibrate_anomaly_threshold, calibrate_known_failure_threshold, classify_failure_descriptor
from hybrid_memory_demo.pipeline import HybridMemoryRuntime, fit_hybrid_memory, iter_image_files
from src.data.collate import collate_batch
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.config import PatchCoreConfig
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.metrics import auroc
from src.patchcore.patchcore import PatchCoreModel, to_numpy


@dataclass(frozen=True)
class EvalRow:
    path: str
    ground_truth_status: str
    ground_truth_label: str | None


@dataclass(frozen=True)
class ProtocolFold:
    unknown_class: str
    known_classes: tuple[str, ...]
    support_per_class: int
    eval_rows: list[EvalRow]
    support_paths: dict[str, list[Path]]


class _PathDataset(Dataset):
    def __init__(self, paths: list[str | Path], image_size: int):
        self.paths = [Path(p) for p in paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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


def load_mvtec_category_protocol(
    *,
    mvtec_root: str | Path,
    category: str,
    calibration_good: int,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, list[Path]], list[Path]]:
    root = Path(mvtec_root) / category
    nominal_train = list(iter_image_files(root / "train" / "good"))
    if len(nominal_train) <= int(calibration_good):
        raise ValueError("Not enough nominal images for the requested calibration split")

    rng = random.Random(int(seed))
    rng.shuffle(nominal_train)
    nominal_calibration = nominal_train[: int(calibration_good)]
    nominal_fit = nominal_train[int(calibration_good) :]

    defect_paths: dict[str, list[Path]] = {}
    for defect_dir in sorted((root / "test").iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        paths = list(iter_image_files(defect_dir))
        rng.shuffle(paths)
        defect_paths[defect_dir.name] = paths

    good_test = list(iter_image_files(root / "test" / "good"))
    return nominal_fit, nominal_calibration, defect_paths, good_test


def extract_embedding_cache(
    *,
    paths: list[str | Path],
    cfg: HybridMemoryConfig,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    dataset = _PathDataset(paths, image_size=cfg.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_batch,
    )

    torch_device = torch.device(device)
    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(torch_device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    cache: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in dataloader:
            emb, _ = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=batch.image.to(torch_device),
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)
            for path, item_emb in zip(batch.path, emb_np):
                cache[str(path)] = item_emb

    return cache


def _patchcore_cfg(cfg: HybridMemoryConfig) -> PatchCoreConfig:
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


def build_cached_predictor(
    *,
    cache: dict[str, np.ndarray],
    nominal_train: list[Path],
    nominal_calibration: list[Path],
    support_paths: dict[str, list[Path]],
    cfg: HybridMemoryConfig,
    seed: int,
) -> dict:
    nominal_bank = np.concatenate([cache[str(path)] for path in nominal_train], axis=0)
    indices = KCenterGreedy().select(nominal_bank, ratio=cfg.coreset_ratio, rng=np.random.default_rng(int(seed)))
    nominal_memory = nominal_bank[indices].astype(np.float32, copy=False)
    nominal_model = PatchCoreModel.fit(_patchcore_cfg(cfg), nominal_memory)

    calib_scores = np.array(
        [float(nominal_model.score_image(cache[str(path)])) for path in nominal_calibration],
        dtype=np.float32,
    )
    anomaly_threshold = calibrate_anomaly_threshold(calib_scores, cfg.anomaly_quantile)

    support_records = []
    failure_descriptors = []
    for label, paths in sorted(support_paths.items()):
        for path in paths:
            emb = cache[str(path)]
            patch_scores = nominal_model.score_patches(emb)
            failure_descriptors.append(
                build_failure_descriptor(emb, patch_scores, top_k_patches=cfg.failure_top_k_patches)
            )
            support_records.append({"label": label, "path": str(path), "anomaly_score": float(np.max(patch_scores))})

    if failure_descriptors:
        descriptor_array = np.stack(failure_descriptors, axis=0).astype(np.float32, copy=False)
        known_failure_threshold, margin_threshold, _ = calibrate_known_failure_threshold(
            descriptor_array,
            [row["label"] for row in support_records],
            quantile=cfg.known_failure_quantile,
            min_margin_ratio=cfg.min_margin_ratio,
        )
    else:
        descriptor_array = np.zeros((0, 0), dtype=np.float32)
        known_failure_threshold = float("inf")
        margin_threshold = float(cfg.min_margin_ratio)

    return {
        "cfg": cfg,
        "nominal_model": nominal_model,
        "anomaly_threshold": float(anomaly_threshold),
        "failure_descriptors": descriptor_array,
        "support_records": support_records,
        "known_failure_threshold": float(known_failure_threshold),
        "margin_threshold": float(margin_threshold),
    }


def predict_with_cached_predictor(predictor: dict, patch_embeddings: np.ndarray) -> dict:
    nominal_model = predictor["nominal_model"]
    cfg = predictor["cfg"]

    patch_scores = nominal_model.score_patches(patch_embeddings)
    anomaly_score = float(np.mean(patch_scores)) if cfg.image_score == "mean" else float(np.max(patch_scores))
    descriptor = build_failure_descriptor(
        patch_embeddings,
        patch_scores,
        top_k_patches=cfg.failure_top_k_patches,
    )
    decision = classify_failure_descriptor(
        descriptor,
        predictor["failure_descriptors"],
        [
            type("SupportRecordLike", (), row)()
            for row in predictor["support_records"]
        ],
        neighbors=cfg.classifier_neighbors,
        known_failure_threshold=predictor["known_failure_threshold"],
        margin_threshold=predictor["margin_threshold"],
    )
    if anomaly_score < predictor["anomaly_threshold"]:
        status = "normal"
        predicted_label = None
    elif decision.is_known_failure:
        status = "known_failure"
        predicted_label = decision.predicted_label
    else:
        status = "unknown_anomaly"
        predicted_label = None
    return {
        "status": status,
        "anomaly_score": float(anomaly_score),
        "anomaly_threshold": float(predictor["anomaly_threshold"]),
        "predicted_label": predicted_label,
        "best_failure_distance": decision.best_distance,
        "failure_margin_ratio": decision.margin_ratio,
        "is_known_failure": bool(decision.is_known_failure),
        "nearest_failures": decision.neighbors,
    }


def evaluate_cached_predictor(predictor: dict, cache: dict[str, np.ndarray], eval_rows: list[EvalRow]) -> dict:
    predictions = []
    for row in eval_rows:
        prediction = predict_with_cached_predictor(predictor, cache[row.path])
        predictions.append(
            {
                "path": row.path,
                "ground_truth_status": row.ground_truth_status,
                "ground_truth_label": row.ground_truth_label,
                "prediction": prediction,
            }
        )

    y_true = np.array([0 if row["ground_truth_status"] == "normal" else 1 for row in predictions], dtype=np.int64)
    y_score = np.array([row["prediction"]["anomaly_score"] for row in predictions], dtype=np.float32)

    confusion = Counter((row["ground_truth_status"], row["prediction"]["status"]) for row in predictions)
    status_accuracy = float(
        sum(int(row["ground_truth_status"] == row["prediction"]["status"]) for row in predictions) / max(len(predictions), 1)
    )

    recalls = {}
    for status in ["normal", "known_failure", "unknown_anomaly"]:
        subset = [row for row in predictions if row["ground_truth_status"] == status]
        recalls[status] = float(
            sum(int(row["prediction"]["status"] == status) for row in subset) / max(len(subset), 1)
        )

    known_rows = [row for row in predictions if row["ground_truth_status"] == "known_failure"]
    known_label_accuracy = float(
        sum(int(row["prediction"]["predicted_label"] == row["ground_truth_label"]) for row in known_rows) / max(len(known_rows), 1)
    )
    predicted_known_rows = [row for row in known_rows if row["prediction"]["status"] == "known_failure"]
    known_label_accuracy_when_predicted_known = float(
        sum(int(row["prediction"]["predicted_label"] == row["ground_truth_label"]) for row in predicted_known_rows)
        / max(len(predicted_known_rows), 1)
    )

    novel_as_known_rate = float(
        sum(
            int(row["prediction"]["status"] == "known_failure")
            for row in predictions
            if row["ground_truth_status"] == "unknown_anomaly"
        )
        / max(sum(int(row["ground_truth_status"] == "unknown_anomaly") for row in predictions), 1)
    )
    normal_false_alarm_rate = float(
        sum(
            int(row["prediction"]["status"] != "normal")
            for row in predictions
            if row["ground_truth_status"] == "normal"
        )
        / max(sum(int(row["ground_truth_status"] == "normal") for row in predictions), 1)
    )

    return {
        "n_eval": len(predictions),
        "image_auroc": float(auroc(y_true, y_score)),
        "status_accuracy": status_accuracy,
        "normal_recall": recalls["normal"],
        "known_failure_recall": recalls["known_failure"],
        "unknown_anomaly_recall": recalls["unknown_anomaly"],
        "known_label_accuracy": known_label_accuracy,
        "known_label_accuracy_when_predicted_known": known_label_accuracy_when_predicted_known,
        "novel_as_known_rate": novel_as_known_rate,
        "normal_false_alarm_rate": normal_false_alarm_rate,
        "confusion": [
            {"ground_truth_status": gt, "predicted_status": pred, "count": count}
            for (gt, pred), count in sorted(confusion.items())
        ],
        "predictions": predictions,
    }


def build_fold(
    *,
    defect_paths: dict[str, list[Path]],
    good_test: list[Path],
    unknown_class: str,
    support_per_class: int,
) -> ProtocolFold:
    known_classes = tuple(sorted(label for label in defect_paths if label != unknown_class))
    support_paths: dict[str, list[Path]] = {}
    eval_rows: list[EvalRow] = []

    for label in known_classes:
        paths = defect_paths[label]
        support = paths[: int(support_per_class)]
        query = paths[int(support_per_class) :]
        if len(support) < int(support_per_class):
            raise ValueError(f"Class {label} does not have {support_per_class} support images")
        support_paths[label] = support
        for path in query:
            eval_rows.append(EvalRow(path=str(path), ground_truth_status="known_failure", ground_truth_label=label))

    for path in defect_paths[unknown_class]:
        eval_rows.append(EvalRow(path=str(path), ground_truth_status="unknown_anomaly", ground_truth_label=unknown_class))
    for path in good_test:
        eval_rows.append(EvalRow(path=str(path), ground_truth_status="normal", ground_truth_label=None))

    return ProtocolFold(
        unknown_class=unknown_class,
        known_classes=known_classes,
        support_per_class=int(support_per_class),
        eval_rows=eval_rows,
        support_paths=support_paths,
    )


def evaluate_runtime(runtime: HybridMemoryRuntime, eval_rows: list[EvalRow]) -> dict:
    predictions = []
    for row in eval_rows:
        prediction, _ = runtime.predict_path(row.path)
        predictions.append(
            {
                "path": row.path,
                "ground_truth_status": row.ground_truth_status,
                "ground_truth_label": row.ground_truth_label,
                "prediction": asdict(prediction),
            }
        )

    y_true = np.array([0 if row["ground_truth_status"] == "normal" else 1 for row in predictions], dtype=np.int64)
    y_score = np.array([row["prediction"]["anomaly_score"] for row in predictions], dtype=np.float32)

    confusion = Counter((row["ground_truth_status"], row["prediction"]["status"]) for row in predictions)
    status_accuracy = float(
        sum(int(row["ground_truth_status"] == row["prediction"]["status"]) for row in predictions) / max(len(predictions), 1)
    )

    recalls = {}
    for status in ["normal", "known_failure", "unknown_anomaly"]:
        subset = [row for row in predictions if row["ground_truth_status"] == status]
        recalls[status] = float(
            sum(int(row["prediction"]["status"] == status) for row in subset) / max(len(subset), 1)
        )

    known_rows = [row for row in predictions if row["ground_truth_status"] == "known_failure"]
    known_label_accuracy = float(
        sum(int(row["prediction"]["predicted_label"] == row["ground_truth_label"]) for row in known_rows) / max(len(known_rows), 1)
    )
    predicted_known_rows = [row for row in known_rows if row["prediction"]["status"] == "known_failure"]
    known_label_accuracy_when_predicted_known = float(
        sum(int(row["prediction"]["predicted_label"] == row["ground_truth_label"]) for row in predicted_known_rows)
        / max(len(predicted_known_rows), 1)
    )

    novel_as_known_rate = float(
        sum(
            int(row["prediction"]["status"] == "known_failure")
            for row in predictions
            if row["ground_truth_status"] == "unknown_anomaly"
        )
        / max(sum(int(row["ground_truth_status"] == "unknown_anomaly") for row in predictions), 1)
    )
    normal_false_alarm_rate = float(
        sum(
            int(row["prediction"]["status"] != "normal")
            for row in predictions
            if row["ground_truth_status"] == "normal"
        )
        / max(sum(int(row["ground_truth_status"] == "normal") for row in predictions), 1)
    )

    return {
        "n_eval": len(predictions),
        "image_auroc": float(auroc(y_true, y_score)),
        "status_accuracy": status_accuracy,
        "normal_recall": recalls["normal"],
        "known_failure_recall": recalls["known_failure"],
        "unknown_anomaly_recall": recalls["unknown_anomaly"],
        "known_label_accuracy": known_label_accuracy,
        "known_label_accuracy_when_predicted_known": known_label_accuracy_when_predicted_known,
        "novel_as_known_rate": novel_as_known_rate,
        "normal_false_alarm_rate": normal_false_alarm_rate,
        "confusion": [
            {"ground_truth_status": gt, "predicted_status": pred, "count": count}
            for (gt, pred), count in sorted(confusion.items())
        ],
        "predictions": predictions,
    }


def aggregate_results(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["support_per_class"]))].append(row)

    summary = []
    for (method, support_per_class), items in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        metrics = [item["metrics"] for item in items]
        confusion = Counter()
        for metric in metrics:
            for row in metric["confusion"]:
                confusion[(row["ground_truth_status"], row["predicted_status"])] += int(row["count"])

        def mean(name: str) -> float:
            return float(np.mean([metric[name] for metric in metrics]))

        summary.append(
            {
                "method": method,
                "support_per_class": support_per_class,
                "n_folds": len(items),
                "image_auroc_mean": mean("image_auroc"),
                "status_accuracy_mean": mean("status_accuracy"),
                "normal_recall_mean": mean("normal_recall"),
                "known_failure_recall_mean": mean("known_failure_recall"),
                "unknown_anomaly_recall_mean": mean("unknown_anomaly_recall"),
                "known_label_accuracy_mean": mean("known_label_accuracy"),
                "known_label_accuracy_when_predicted_known_mean": mean("known_label_accuracy_when_predicted_known"),
                "novel_as_known_rate_mean": mean("novel_as_known_rate"),
                "normal_false_alarm_rate_mean": mean("normal_false_alarm_rate"),
                "confusion_total": [
                    {"ground_truth_status": gt, "predicted_status": pred, "count": count}
                    for (gt, pred), count in sorted(confusion.items())
                ],
            }
        )
    return summary


def summary_to_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "# Hybrid Benchmark Summary",
        "",
        "| Method | Support/Class | Image AUROC | Status Acc | Normal Recall | Known Recall | Unknown Recall | Known Label Acc | Novel->Known | False Alarm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["support_per_class"]),
                    f'{row["image_auroc_mean"]:.3f}',
                    f'{row["status_accuracy_mean"]:.3f}',
                    f'{row["normal_recall_mean"]:.3f}',
                    f'{row["known_failure_recall_mean"]:.3f}',
                    f'{row["unknown_anomaly_recall_mean"]:.3f}',
                    f'{row["known_label_accuracy_mean"]:.3f}',
                    f'{row["novel_as_known_rate_mean"]:.3f}',
                    f'{row["normal_false_alarm_rate_mean"]:.3f}',
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `baseline_nominal_only` uses the same nominal memory and threshold but no labeled failure bank.")
    lines.append("- `hybrid_known_failure_bank` adds labeled support images for the known classes.")
    lines.append("- `Known Label Acc` is measured over all ground-truth known failures.")
    lines.append("- `Novel->Known` is the rate at which truly unseen failure classes are incorrectly forced into a known class.")
    return "\n".join(lines) + "\n"


def save_benchmark_report(out_dir: str | Path, *, protocol: dict, runs: list[dict], summary: list[dict]) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True))
    (out_path / "runs.json").write_text(json.dumps(runs, indent=2, sort_keys=True))
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_path / "summary.md").write_text(summary_to_markdown(summary))


def fit_runtime(
    *,
    nominal_train: list[Path],
    nominal_calibration: list[Path],
    support_paths: dict[str, list[Path]],
    cfg: HybridMemoryConfig,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> HybridMemoryRuntime:
    artifact = fit_hybrid_memory(
        nominal_train_paths=nominal_train,
        nominal_calibration_paths=nominal_calibration,
        labeled_failure_paths=support_paths,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return HybridMemoryRuntime(artifact, device=device)
