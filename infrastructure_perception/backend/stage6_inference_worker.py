from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _device(value: str) -> str | int:
    if value != "auto":
        return int(value) if value.isdigit() else value
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _prediction_payload(result: Any) -> list[dict[str, Any]]:
    height, width = (int(value) for value in result.orig_shape)
    if height <= 0 or width <= 0 or result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    predictions: list[dict[str, Any]] = []
    for box, confidence, class_id in zip(boxes, confidences, classes, strict=True):
        x_min, y_min, x_max, y_max = box
        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [
                    max(0.0, min(1.0, float(x_min) / width)),
                    max(0.0, min(1.0, float(y_min) / height)),
                    max(0.0, min(1.0, float(x_max) / width)),
                    max(0.0, min(1.0, float(y_max) / height)),
                ],
            }
        )
    return predictions


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("The declared Stage 6 checkpoint is unavailable.")
    observed_sha = _sha256(checkpoint)
    if observed_sha != args.checkpoint_sha256.upper():
        raise RuntimeError("The Stage 6 checkpoint hash does not match its model package.")

    from ultralytics import YOLO

    selected_device = _device(args.device)
    started = time.perf_counter()
    model = YOLO(str(checkpoint))
    load_seconds = time.perf_counter() - started

    if args.self_test:
        import numpy as np

        source: Any = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        source_identity = "synthetic_zero_image_no_project_pixels"
    else:
        if not args.image:
            raise ValueError("An image is required unless --self-test is used.")
        image = Path(args.image).resolve()
        if not image.is_file():
            raise FileNotFoundError("The requested Stage 6 image is unavailable.")
        source = str(image)
        source_identity = image.name

    inference_started = time.perf_counter()
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_detections,
        device=selected_device,
        verbose=False,
        save=False,
        stream=False,
    )
    inference_seconds = time.perf_counter() - inference_started
    if len(results) != 1:
        raise RuntimeError("The Stage 6 worker expected exactly one inference result.")
    result = results[0]
    names = result.names if isinstance(result.names, dict) else {}
    predictions = _prediction_payload(result)
    return {
        "worker_contract_version": "1.0.0",
        "status": "self_test_passed" if args.self_test else "inference_complete",
        "checkpoint_sha256": observed_sha,
        "source_identity": source_identity,
        "device": str(selected_device),
        "image_shape": [int(value) for value in result.orig_shape],
        "class_names": {str(key): str(value) for key, value in names.items()},
        "prediction_count": len(predictions),
        "predictions": predictions,
        "timing_ms": {
            "model_load": round(load_seconds * 1000, 3),
            "worker_inference": round(inference_seconds * 1000, 3),
            **{
                str(key): round(float(value), 3)
                for key, value in (result.speed or {}).items()
                if isinstance(value, (int, float))
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated Ultralytics worker for the TWM-InfraDet Stage 6 adapter."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output).resolve()
    try:
        payload = run(args)
    except Exception as error:  # the parent process records the bounded failure detail
        _write_json(
            output,
            {
                "worker_contract_version": "1.0.0",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            },
        )
        print(f"Stage 6 inference worker failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    _write_json(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
