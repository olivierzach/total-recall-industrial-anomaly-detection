from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_memory_demo.pipeline import HybridMemoryRuntime, iter_image_files, load_artifact
from hybrid_memory_demo.router import NominalRouterRuntime, load_nominal_router


@dataclass(frozen=True)
class DemoEntry:
    key: str
    display_name: str
    artifact_dir: Path
    examples_json: Path | None
    report_json: Path | None
    note: str | None
    artifact_info: dict
    report: dict
    runtime: HybridMemoryRuntime
    bank: list[dict]


def _pil_to_data_url(image: Image.Image) -> str:
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_examples(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return json.loads(path.read_text())


def _load_report(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _infer_bank_source(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "mvtec" in parts:
        idx = parts.index("mvtec")
        category = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        label = parts[idx + 3] if idx + 3 < len(parts) else category
        return f"mvtec/{category}", label
    if "btad" in parts:
        stem = path.stem
        tokens = stem.split("_")
        component = tokens[0] if tokens else "btad"
        label = "ok" if "_ok_" in stem else "ko"
        return f"btad/{component}", label
    return "local", path.parent.name


def _entry_note(entry: DemoEntry) -> str:
    dataset = entry.artifact_info.get("dataset")
    if dataset == "btad":
        return (
            entry.note
            or "BTAD is shown here with component-conditioned failure labels because the local dataset exposes ok/ko labels rather than named defect families."
        )
    if dataset == "mvtec":
        return entry.note or "MVTec profile with named defect families."
    return entry.note or "Hybrid memory demo profile."


def _artifact_profile(entry: DemoEntry) -> str:
    cfg = entry.runtime.artifact.cfg
    layers = ",".join(cfg.layers)
    return f"{cfg.backbone} | layers {layers} | image {cfg.image_size} | coreset {cfg.coreset_ratio}"


def _report_path(*, examples_json: Path | None, artifact_dir: Path) -> Path | None:
    if examples_json is not None:
        candidate = examples_json.parent / "report.json"
        if candidate.exists():
            return candidate
    candidate = artifact_dir.parent / "report.json"
    return candidate if candidate.exists() else None


def _label_display_name(entry: DemoEntry) -> str:
    if entry.artifact_info.get("dataset") == "btad":
        return "Predicted Support Family"
    return "Predicted Label"


def _protocol_summary(entry: DemoEntry) -> str:
    task_type = entry.report.get("task_type")
    if task_type == "open_set_support_family_retrieval":
        return "Open-set support-family retrieval"
    if task_type == "open_set_named_failure_retrieval":
        return "Open-set named failure retrieval"
    dataset = entry.artifact_info.get("dataset")
    if dataset == "btad":
        return "Open-set support-family retrieval"
    if dataset == "mvtec":
        return "Open-set named failure retrieval"
    return "Hybrid open-set retrieval"


def _primary_interpretation(entry: DemoEntry) -> str:
    if entry.report.get("primary_interpretation"):
        return str(entry.report["primary_interpretation"])
    if entry.artifact_info.get("dataset") == "btad":
        return "Treat BTAD as open-set retrieval over support families. Component labels are not named defect semantics."
    return "Treat the predicted label as the nearest stored known failure class when status is known_failure."


def _evaluation_snapshot(entry: DemoEntry) -> dict:
    report = entry.report
    if not report:
        return {}
    return {
        "status_accuracy": report.get("status_accuracy"),
        "known_failure_recall": report.get("known_failure_recall"),
        "unknown_anomaly_recall": report.get("unknown_anomaly_recall"),
        "known_label_accuracy": report.get("known_label_accuracy"),
        "novel_as_known_rate": report.get("novel_as_known_rate"),
        "normal_false_alarm_rate": report.get("normal_false_alarm_rate"),
        "n_eval": report.get("n_eval"),
    }


def _profile_bank_roots(entry: DemoEntry, *, repo_root: Path) -> list[Path]:
    dataset = str(entry.artifact_info.get("dataset") or "")
    category = entry.artifact_info.get("category")
    if dataset == "mvtec" and category:
        root = repo_root / "data" / "mvtec" / str(category)
        if root.exists():
            return [root]
    if dataset == "btad":
        root = repo_root / "data" / "btad"
        if root.exists():
            return [root]

    roots = []
    for rel in ["data/mvtec", "data/btad"]:
        root = repo_root / rel
        if root.exists():
            roots.append(root)
    return roots


def _build_random_bank(roots: list[Path], *, limit: int, seed: int) -> list[dict]:
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(iter_image_files(root))

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    selected = candidates[: min(int(limit), len(candidates))]

    bank = []
    for idx, path in enumerate(selected):
        source, label = _infer_bank_source(path)
        bank.append(
            {
                "id": f"bank_{idx}",
                "path": str(path),
                "source": source,
                "label": label,
            }
        )
    return bank


def _load_catalog(
    *,
    manifest_json: str | Path | None,
    artifact_dir: str | Path | None,
    examples_json: str | Path | None,
    device: str,
) -> tuple[str, dict[str, DemoEntry], NominalRouterRuntime | None]:
    repo_root = Path(__file__).resolve().parents[1]
    router_runtime = None
    if manifest_json is not None:
        manifest_path = Path(manifest_json).resolve()
        payload = json.loads(manifest_path.read_text())
        base_dir = manifest_path.parent
    elif artifact_dir is not None:
        payload = {
            "default_dataset": "default",
            "datasets": [
                {
                    "key": "default",
                    "display_name": "Hybrid Memory Demo",
                    "artifact_dir": str(artifact_dir),
                    "examples_json": str(examples_json) if examples_json is not None else None,
                }
            ],
        }
        base_dir = Path.cwd()
    else:
        default_manifest = Path(__file__).resolve().parent / "demo_manifest.json"
        payload = json.loads(default_manifest.read_text())
        base_dir = default_manifest.parent

    router_path = payload.get("router_artifact")
    if router_path is not None:
        router_runtime = NominalRouterRuntime(load_nominal_router(_resolve_path(router_path, base_dir=base_dir)), device=device)

    datasets = payload.get("datasets", [])
    if not datasets:
        raise ValueError("Dataset catalog is empty")

    entries: dict[str, DemoEntry] = {}
    for row in datasets:
        key = str(row["key"])
        resolved_artifact = _resolve_path(row["artifact_dir"], base_dir=base_dir)
        resolved_examples = _resolve_path(row.get("examples_json"), base_dir=base_dir)
        resolved_report = _report_path(examples_json=resolved_examples, artifact_dir=resolved_artifact)
        artifact = load_artifact(resolved_artifact)
        info = dict(artifact.artifact_info or {})
        runtime = HybridMemoryRuntime(artifact, device=device)
        stub = DemoEntry(
            key=key,
            display_name=str(row.get("display_name") or key),
            artifact_dir=resolved_artifact,
            examples_json=resolved_examples,
            report_json=resolved_report,
            note=row.get("note"),
            artifact_info=info,
            report=_load_report(resolved_report),
            runtime=runtime,
            bank=[],
        )
        entries[key] = DemoEntry(
            key=stub.key,
            display_name=stub.display_name,
            artifact_dir=stub.artifact_dir,
            examples_json=stub.examples_json,
            report_json=stub.report_json,
            note=stub.note,
            artifact_info=stub.artifact_info,
            report=stub.report,
            runtime=stub.runtime,
            bank=_build_random_bank(_profile_bank_roots(stub, repo_root=repo_root), limit=1000, seed=0),
        )

    default_dataset = str(payload.get("default_dataset") or next(iter(entries)))
    if default_dataset not in entries:
        default_dataset = next(iter(entries))
    return default_dataset, entries, router_runtime


def serve_demo(
    *,
    artifact_dir: str | Path | None,
    examples_json: str | Path | None,
    manifest_json: str | Path | None,
    host: str,
    port: int,
    device: str,
) -> None:
    default_dataset, catalog, router_runtime = _load_catalog(
        manifest_json=manifest_json,
        artifact_dir=artifact_dir,
        examples_json=examples_json,
        device=device,
    )
    app_html = (Path(__file__).resolve().parent / "app.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_png(self, image: Image.Image) -> None:
            from io import BytesIO

            buf = BytesIO()
            image.save(buf, format="PNG")
            raw = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _resolve_entry(self, parsed) -> DemoEntry | None:
            params = parse_qs(parsed.query)
            dataset = params.get("dataset", [default_dataset])[0]
            return catalog.get(dataset)

        def _resolve_mode(self, parsed) -> str:
            params = parse_qs(parsed.query)
            mode = params.get("mode", ["selected"])[0]
            return "auto" if mode == "auto" else "selected"

        def _catalog_rows(self) -> list[dict]:
            rows = []
            for key, entry in catalog.items():
                artifact = entry.runtime.artifact
                rows.append(
                    {
                        "key": key,
                        "display_name": entry.display_name,
                        "artifact_info": entry.artifact_info,
                        "thresholds": {
                            "anomaly_threshold": artifact.anomaly_threshold,
                            "known_failure_threshold": artifact.known_failure_threshold,
                            "margin_threshold": artifact.margin_threshold,
                        },
                        "example_count": len(_load_examples(entry.examples_json)),
                        "support_library_count": len(artifact.support_records),
                        "bank_count": len(entry.bank),
                        "note": _entry_note(entry),
                        "profile_summary": _artifact_profile(entry),
                        "protocol_summary": _protocol_summary(entry),
                        "label_display_name": _label_display_name(entry),
                        "primary_interpretation": _primary_interpretation(entry),
                        "evaluation_snapshot": _evaluation_snapshot(entry),
                    }
                )
            return rows

        def _prediction_payload(
            self,
            *,
            requested_entry: DemoEntry,
            entry: DemoEntry,
            mode: str,
            routing_decision,
            profile_comparison,
            image: Image.Image,
            prediction,
            score_map,
            embedding_map,
            embedding_reference,
            embedding_projection,
        ) -> dict:
            overlay = entry.runtime.render_overlay(image, score_map)
            embedding = entry.runtime.render_embedding_map(image, embedding_map)
            embedding_space = entry.runtime.render_embedding_space(embedding_projection)
            nearest = []
            for item in prediction.nearest_failures:
                support_image = Image.open(item["path"]).convert("RGB")
                nearest.append(
                    {
                        **item,
                        "image_url": _pil_to_data_url(support_image),
                    }
                )
            return {
                "requested_dataset": requested_entry.key,
                "dataset": entry.key,
                "display_name": entry.display_name,
                "mode": mode,
                "note": _entry_note(entry),
                "label_display_name": _label_display_name(entry),
                "protocol_summary": _protocol_summary(entry),
                "primary_interpretation": _primary_interpretation(entry),
                "dataset_evaluation": _evaluation_snapshot(entry),
                "routing_decision": routing_decision,
                "profile_comparison": profile_comparison,
                "prediction": asdict(prediction),
                "input_image_url": _pil_to_data_url(image),
                "overlay_image_url": _pil_to_data_url(overlay),
                "embedding_image_url": _pil_to_data_url(embedding),
                "embedding_space_image_url": _pil_to_data_url(embedding_space),
                "embedding_reference": embedding_reference,
                "nearest_failures": nearest,
            }

        def _profile_comparison(self, image: Image.Image) -> list[dict]:
            rows = []
            for key, candidate in catalog.items():
                prediction, _ = candidate.runtime.predict_image(image)
                rows.append(
                    {
                        "dataset": key,
                        "display_name": candidate.display_name,
                        "status": prediction.status,
                        "predicted_label": prediction.predicted_label,
                        "anomaly_score": float(prediction.anomaly_score),
                        "anomaly_threshold": float(prediction.anomaly_threshold),
                        "anomaly_ratio": float(prediction.anomaly_score / max(prediction.anomaly_threshold, 1e-8)),
                    }
                )
            return rows

        def _choose_entry(
            self,
            *,
            requested_entry: DemoEntry,
            image: Image.Image,
            mode: str,
            profile_comparison: list[dict],
        ) -> tuple[DemoEntry, dict | None]:
            if mode != "auto" or router_runtime is None:
                return requested_entry, None
            routing = router_runtime.route_image(image)
            best_profile = min(profile_comparison, key=lambda row: (row["anomaly_ratio"], row["anomaly_score"]))
            chosen = catalog.get(best_profile["dataset"], requested_entry)
            routing["selected_by_nominal_fit"] = {
                "dataset": best_profile["dataset"],
                "display_name": best_profile["display_name"],
                "anomaly_ratio": float(best_profile["anomaly_ratio"]),
            }
            return chosen, routing

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(app_html)
                return
            if parsed.path == "/api/catalog":
                self._send_json({"default_dataset": default_dataset, "datasets": self._catalog_rows()})
                return
            if parsed.path == "/api/examples":
                entry = self._resolve_entry(parsed)
                if entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                examples = _load_examples(entry.examples_json)
                artifact = entry.runtime.artifact
                self._send_json(
                    {
                        "dataset": entry.key,
                        "display_name": entry.display_name,
                        "note": _entry_note(entry),
                        "examples": [
                            {
                                **row,
                                "image_url": _pil_to_data_url(Image.open(row["path"]).convert("RGB")),
                            }
                            for row in examples
                        ],
                        "support_library": [
                            {
                                "support_index": idx,
                                **asdict(rec),
                                "image_url": _pil_to_data_url(Image.open(rec.path).convert("RGB")),
                            }
                            for idx, rec in enumerate(artifact.support_records)
                        ],
                        "thresholds": {
                            "anomaly_threshold": artifact.anomaly_threshold,
                            "known_failure_threshold": artifact.known_failure_threshold,
                            "margin_threshold": artifact.margin_threshold,
                        },
                        "artifact_info": entry.artifact_info,
                        "profile_summary": _artifact_profile(entry),
                        "protocol_summary": _protocol_summary(entry),
                        "label_display_name": _label_display_name(entry),
                        "primary_interpretation": _primary_interpretation(entry),
                        "dataset_evaluation": _evaluation_snapshot(entry),
                    }
                )
                return
            if parsed.path == "/api/bank":
                entry = self._resolve_entry(parsed)
                if entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                self._send_json(
                    {
                        "dataset": entry.key,
                        "count": len(entry.bank),
                        "bank": entry.bank,
                        "note": _entry_note(entry),
                    }
                )
                return
            if parsed.path == "/api/bank_image":
                entry = self._resolve_entry(parsed)
                if entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                params = parse_qs(parsed.query)
                bank_id = params.get("id", [None])[0]
                bank_by_id = {row["id"]: row for row in entry.bank}
                row = bank_by_id.get(bank_id)
                if row is None:
                    self._send_json({"error": f"unknown bank id: {bank_id}"}, status=404)
                    return
                image = Image.open(row["path"]).convert("RGB").resize((128, 128))
                self._send_png(image)
                return
            if parsed.path == "/api/predict_example":
                requested_entry = self._resolve_entry(parsed)
                if requested_entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                mode = self._resolve_mode(parsed)
                params = parse_qs(parsed.query)
                example_id = params.get("id", [None])[0]
                examples = _load_examples(requested_entry.examples_json)
                example_by_id = {row["id"]: row for row in examples}
                row = example_by_id.get(example_id)
                if row is None:
                    self._send_json({"error": f"unknown example id: {example_id}"}, status=404)
                    return
                image = Image.open(row["path"]).convert("RGB")
                profile_comparison = self._profile_comparison(image)
                entry, routing_decision = self._choose_entry(requested_entry=requested_entry, image=image, mode=mode, profile_comparison=profile_comparison)
                prediction, score_map, embedding_map, embedding_reference, embedding_projection = entry.runtime.predict_image_with_diagnostics(image)
                payload = self._prediction_payload(
                    requested_entry=requested_entry,
                    entry=entry,
                    mode=mode,
                    routing_decision=routing_decision,
                    profile_comparison=profile_comparison,
                    image=image,
                    prediction=prediction,
                    score_map=score_map,
                    embedding_map=embedding_map,
                    embedding_reference=embedding_reference,
                    embedding_projection=embedding_projection,
                )
                payload["example"] = row
                self._send_json(payload)
                return
            if parsed.path == "/api/predict_support":
                requested_entry = self._resolve_entry(parsed)
                if requested_entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                mode = self._resolve_mode(parsed)
                params = parse_qs(parsed.query)
                support_index = int(params.get("index", [-1])[0])
                records = requested_entry.runtime.artifact.support_records
                if support_index < 0 or support_index >= len(records):
                    self._send_json({"error": f"unknown support index: {support_index}"}, status=404)
                    return
                rec = records[support_index]
                image = Image.open(rec.path).convert("RGB")
                profile_comparison = self._profile_comparison(image)
                entry, routing_decision = self._choose_entry(requested_entry=requested_entry, image=image, mode=mode, profile_comparison=profile_comparison)
                prediction, score_map, embedding_map, embedding_reference, embedding_projection = entry.runtime.predict_image_with_diagnostics(image)
                payload = self._prediction_payload(
                    requested_entry=requested_entry,
                    entry=entry,
                    mode=mode,
                    routing_decision=routing_decision,
                    profile_comparison=profile_comparison,
                    image=image,
                    prediction=prediction,
                    score_map=score_map,
                    embedding_map=embedding_map,
                    embedding_reference=embedding_reference,
                    embedding_projection=embedding_projection,
                )
                payload["support_item"] = {"support_index": support_index, **asdict(rec)}
                self._send_json(payload)
                return
            if parsed.path == "/api/predict_bank":
                requested_entry = self._resolve_entry(parsed)
                if requested_entry is None:
                    self._send_json({"error": "unknown dataset"}, status=404)
                    return
                mode = self._resolve_mode(parsed)
                params = parse_qs(parsed.query)
                bank_id = params.get("id", [None])[0]
                bank_by_id = {row["id"]: row for row in requested_entry.bank}
                row = bank_by_id.get(bank_id)
                if row is None:
                    self._send_json({"error": f"unknown bank id: {bank_id}"}, status=404)
                    return
                image = Image.open(row["path"]).convert("RGB")
                profile_comparison = self._profile_comparison(image)
                entry, routing_decision = self._choose_entry(requested_entry=requested_entry, image=image, mode=mode, profile_comparison=profile_comparison)
                prediction, score_map, embedding_map, embedding_reference, embedding_projection = entry.runtime.predict_image_with_diagnostics(image)
                payload = self._prediction_payload(
                    requested_entry=requested_entry,
                    entry=entry,
                    mode=mode,
                    routing_decision=routing_decision,
                    profile_comparison=profile_comparison,
                    image=image,
                    prediction=prediction,
                    score_map=score_map,
                    embedding_map=embedding_map,
                    embedding_reference=embedding_reference,
                    embedding_projection=embedding_projection,
                )
                payload["bank_item"] = row
                self._send_json(payload)
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/predict":
                self._send_json({"error": "not found"}, status=404)
                return

            requested_entry = self._resolve_entry(parsed)
            if requested_entry is None:
                self._send_json({"error": "unknown dataset"}, status=404)
                return
            mode = self._resolve_mode(parsed)

            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            if not payload:
                self._send_json({"error": "empty request body"}, status=400)
                return

            try:
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                profile_comparison = self._profile_comparison(image)
                entry, routing_decision = self._choose_entry(requested_entry=requested_entry, image=image, mode=mode, profile_comparison=profile_comparison)
                prediction, score_map, embedding_map, embedding_reference, embedding_projection = entry.runtime.predict_image_with_diagnostics(image)
                response_payload = self._prediction_payload(
                    requested_entry=requested_entry,
                    entry=entry,
                    mode=mode,
                    routing_decision=routing_decision,
                    profile_comparison=profile_comparison,
                    image=image,
                    prediction=prediction,
                    score_map=score_map,
                    embedding_map=embedding_map,
                    embedding_reference=embedding_reference,
                    embedding_projection=embedding_projection,
                )
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(response_payload)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(
        json.dumps(
            {
                "host": host,
                "port": int(port),
                "default_dataset": default_dataset,
                "datasets": [
                    {
                        "key": entry.key,
                        "display_name": entry.display_name,
                        "artifact_dir": str(entry.artifact_dir),
                        "bank_count": len(entry.bank),
                    }
                    for entry in catalog.values()
                ],
            },
            indent=2,
        )
    )
    print(f"Open http://{host}:{int(port)} in a browser")
    server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--examples-json", default=None)
    ap.add_argument("--manifest-json", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    serve_demo(
        artifact_dir=args.artifact_dir,
        examples_json=args.examples_json,
        manifest_json=args.manifest_json,
        host=args.host,
        port=int(args.port),
        device=args.device,
    )


if __name__ == "__main__":
    main()
