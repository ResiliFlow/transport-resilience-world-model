from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from infrastructure_perception.backend.stage7_fusion_r07 import (
    CLASS_NAMES as R07_CLASS_NAMES,
    fuse_r07,
)


CLASS_NAMES = (
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_cracking",
    "uneven_or_broken_footpath",
    "broken_kerb",
    "standing_water",
    "debris_or_vegetation_obstruction",
)

VIEW_CROPS = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "lower_70": (0.0, 0.30, 1.0, 1.0),
    "lower_55": (0.0, 0.45, 1.0, 1.0),
    "road_centre": (0.16, 0.38, 0.92, 1.0),
    "road_left": (0.0, 0.38, 0.64, 1.0),
    "road_right": (0.36, 0.38, 1.0, 1.0),
    "crack_mid": (0.18, 0.40, 0.92, 0.82),
    "crack_focus_left": (0.16, 0.46, 0.56, 0.84),
    "crack_focus_centre": (0.34, 0.46, 0.74, 0.84),
    "crack_focus_right": (0.54, 0.46, 0.96, 0.84),
    "side_edge_left": (0.0, 0.34, 0.42, 1.0),
    "side_edge_right": (0.58, 0.34, 1.0, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _intersection(first: list[float], second: list[float]) -> list[float] | None:
    box = [
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    ]
    if box[2] - box[0] <= 0.0 or box[3] - box[1] <= 0.0:
        return None
    if (box[2] - box[0]) * (box[3] - box[1]) < 0.004:
        return None
    return box


def _nms(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(
            candidate["class_id"] == previous["class_id"]
            and _iou(candidate["box"], previous["box"]) >= threshold
            for previous in retained
        ):
            continue
        retained.append(candidate)
    return retained


def _prepare_inputs(image_path: Path) -> tuple[list[np.ndarray], list[dict[str, Any]], tuple[int, int]]:
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        arrays: list[np.ndarray] = []
        variants: list[dict[str, Any]] = []
        for name, (x1, y1, x2, y2) in VIEW_CROPS.items():
            arrays.append(
                np.asarray(
                    rgb.crop(
                        (
                            round(x1 * width),
                            round(y1 * height),
                            round(x2 * width),
                            round(y2 * height),
                        )
                    )
                )
            )
            variants.append({"name": name, "crop": [x1, y1, x2, y2]})
    return arrays, variants, (width, height)


def _model_candidates(
    checkpoint: Path,
    arrays: list[np.ndarray],
    variants: list[dict[str, Any]],
    *,
    confidence: float,
    imgsz: int,
    iou: float,
    max_det: int,
    device: str,
    isolated_variants: set[str] | None = None,
    preserve_variant_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded_at = time.perf_counter()
    model = YOLO(str(checkpoint), task="detect")
    observed_names = model.names
    normalised_names = [str(observed_names[index]) for index in range(len(observed_names))]
    if normalised_names != list(CLASS_NAMES):
        raise RuntimeError("The Stage 7 ensemble component class map is invalid.")
    load_seconds = time.perf_counter() - loaded_at
    predicted_at = time.perf_counter()
    results = model.predict(
        source=arrays,
        imgsz=imgsz,
        conf=confidence,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
        save=False,
    )
    prediction_seconds = time.perf_counter() - predicted_at
    candidates: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    maxima: dict[str, float | None] = {}
    for variant, result in zip(variants, results, strict=True):
        crop = variant["crop"]
        crop_width = crop[2] - crop[0]
        crop_height = crop[3] - crop[1]
        variant_candidates: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxyn.detach().cpu().tolist()
            confidences = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, score, class_id in zip(boxes, confidences, classes, strict=True):
                mapped = [
                    crop[0] + crop_width * float(box[0]),
                    crop[1] + crop_height * float(box[1]),
                    crop[0] + crop_width * float(box[2]),
                    crop[1] + crop_height * float(box[3]),
                ]
                mapped = [min(1.0, max(0.0, value)) for value in mapped]
                if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
                    continue
                variant_candidates.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "box": mapped,
                        "variant": variant["name"],
                    }
                )
        counts[variant["name"]] = len(variant_candidates)
        maxima[variant["name"]] = (
            max(item["confidence"] for item in variant_candidates)
            if variant_candidates
            else None
        )
        candidates.extend(variant_candidates)
    del results
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if preserve_variant_candidates:
        return candidates, {
            "load_seconds": load_seconds,
            "prediction_seconds": prediction_seconds,
            "variant_candidate_counts": counts,
            "variant_max_confidence": maxima,
            "variant_candidates_preserved": True,
        }
    isolated = isolated_variants or set()
    primary_candidates = [item for item in candidates if item["variant"] not in isolated]
    isolated_candidates = [item for item in candidates if item["variant"] in isolated]
    return [*_nms(primary_candidates, 0.55), *_nms(isolated_candidates, 0.55)], {
        "load_seconds": load_seconds,
        "prediction_seconds": prediction_seconds,
        "variant_candidate_counts": counts,
        "variant_max_confidence": maxima,
    }


def _consensus(
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    *,
    agreement_iou: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    used_teacher: set[int] = set()
    for anchor_item in sorted(anchor, key=lambda item: item["confidence"], reverse=True):
        matches = [
            (index, item, _iou(anchor_item["box"], item["box"]))
            for index, item in enumerate(teacher)
            if index not in used_teacher and item["class_id"] == anchor_item["class_id"]
        ]
        matches = [item for item in matches if item[2] >= agreement_iou]
        if not matches:
            continue
        teacher_index, teacher_item, overlap = max(
            matches,
            key=lambda item: (item[2], item[1]["confidence"]),
        )
        shared_box = _intersection(anchor_item["box"], teacher_item["box"])
        if shared_box is None:
            continue
        used_teacher.add(teacher_index)
        fused.append(
            {
                "class_id": anchor_item["class_id"],
                "confidence": math.sqrt(
                    anchor_item["confidence"] * teacher_item["confidence"]
                ),
                "box": shared_box,
                "agreement_iou": overlap,
                "support": {
                    "yolo26s": anchor_item["confidence"],
                    "rtdetr_l": teacher_item["confidence"],
                    "yolo_variant": anchor_item["variant"],
                    "rtdetr_variant": teacher_item["variant"],
                },
            }
        )
    retained: list[dict[str, Any]] = []
    for candidate in sorted(fused, key=lambda item: item["confidence"], reverse=True):
        if any(_iou(candidate["box"], previous["box"]) >= 0.72 for previous in retained):
            continue
        retained.append(candidate)
        if len(retained) >= max_detections:
            break
    return retained


def _appearance_metrics(image_path: Path, box: list[float]) -> dict[str, float]:
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    x1 = max(0, min(width - 1, round(box[0] * width)))
    y1 = max(0, min(height - 1, round(box[1] * height)))
    x2 = max(x1 + 1, min(width, round(box[2] * width)))
    y2 = max(y1 + 1, min(height, round(box[3] * height)))
    region = rgb[y1:y2, x1:x2]
    luminance = 0.2126 * region[..., 0] + 0.7152 * region[..., 1] + 0.0722 * region[..., 2]
    channel_max = region.max(axis=2)
    channel_min = region.min(axis=2)
    saturation = np.divide(
        channel_max - channel_min,
        np.maximum(channel_max, 1.0),
    )
    horizontal = np.abs(np.diff(luminance, axis=1)).mean() if luminance.shape[1] > 1 else 0.0
    vertical = np.abs(np.diff(luminance, axis=0)).mean() if luminance.shape[0] > 1 else 0.0
    return {
        "mean_luminance": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "dark_fraction_lt_90": float((luminance < 90.0).mean()),
        "bright_fraction_gt_210": float((luminance > 210.0).mean()),
        "mean_saturation": float(saturation.mean()),
        "mean_gradient": float((horizontal + vertical) / 2.0),
        "mean_red": float(region[..., 0].mean()),
        "mean_green": float(region[..., 1].mean()),
        "mean_blue": float(region[..., 2].mean()),
    }


def _road_surface_grid(image_path: Path) -> list[dict[str, Any]]:
    """Numeric-only diagnostics for the mid/far road; no provider pixels leave the worker."""

    cells: list[dict[str, Any]] = []
    x_start, x_step = 0.20, 0.10
    y_start, y_step = 0.30, 0.072
    for row in range(5):
        for column in range(6):
            box = [
                x_start + column * x_step,
                y_start + row * y_step,
                x_start + (column + 1) * x_step,
                y_start + (row + 1) * y_step,
            ]
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "box": box,
                    "appearance": _appearance_metrics(image_path, box),
                }
            )
    return cells


def _road_crack_structure_diagnostics(image_path: Path) -> dict[str, Any]:
    """Return numeric road-line evidence without retaining provider pixels."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"state": "image_unavailable", "segment_count": 0, "segments": []}
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(cv2.GaussianBlur(gray, (3, 3), 0))[0]
    if detected is None:
        return {"state": "complete", "segment_count": 0, "segments": []}

    def road_half_width(y_normalised: float) -> float:
        return min(0.49, 0.08 + max(0.0, y_normalised - 0.42) * 1.05)

    segments: list[dict[str, Any]] = []
    for raw in detected.reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in raw)
        dx, dy = x2 - x1, y2 - y1
        length_pixels = math.hypot(dx, dy)
        if length_pixels < 10.0:
            continue
        midpoint_x = (x1 + x2) / (2.0 * width)
        midpoint_y = (y1 + y2) / (2.0 * height)
        if not 0.45 <= midpoint_y <= 0.86:
            continue
        half_width = road_half_width(midpoint_y)
        if not 0.52 - half_width <= midpoint_x <= 0.52 + half_width:
            continue

        sample_count = max(8, min(96, round(length_pixels)))
        samples = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        centre_x = np.clip(np.rint(x1 + samples * dx).astype(np.int32), 0, width - 1)
        centre_y = np.clip(np.rint(y1 + samples * dy).astype(np.int32), 0, height - 1)
        norm = max(length_pixels, 1.0)
        perpendicular_x = -dy / norm
        perpendicular_y = dx / norm
        side_values: list[np.ndarray] = []
        for offset in (3.0, -3.0, 5.0, -5.0):
            side_x = np.clip(
                np.rint(centre_x + perpendicular_x * offset).astype(np.int32),
                0,
                width - 1,
            )
            side_y = np.clip(
                np.rint(centre_y + perpendicular_y * offset).astype(np.int32),
                0,
                height - 1,
            )
            side_values.append(gray[side_y, side_x].astype(np.float32))
        centre_values = gray[centre_y, centre_x].astype(np.float32)
        side_mean = float(np.concatenate(side_values).mean())
        centre_mean = float(centre_values.mean())
        dark_contrast = side_mean - centre_mean

        pad = 4
        bx1 = max(0, math.floor(min(x1, x2)) - pad)
        by1 = max(0, math.floor(min(y1, y2)) - pad)
        bx2 = min(width, math.ceil(max(x1, x2)) + pad + 1)
        by2 = min(height, math.ceil(max(y1, y2)) + pad + 1)
        region_saturation = float(hsv[by1:by2, bx1:bx2, 1].mean() / 255.0)
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        acute_angle = min(angle, 180.0 - angle)
        orientation = (
            "transverse"
            if acute_angle <= 22.0
            else "longitudinal"
            if acute_angle >= 28.0
            else "diagonal_boundary"
        )
        x_at_horizon: float | None = None
        if abs(dy) >= 2.0:
            horizon_y = 0.42 * height
            x_at_horizon = float((x1 + (horizon_y - y1) * dx / dy) / width)
        horizon_consistent = (
            x_at_horizon is not None and -0.10 <= x_at_horizon <= 1.10
        )
        length_normalised = length_pixels / math.hypot(width, height)
        road_line_candidate = (
            length_normalised >= 0.022
            and dark_contrast >= 2.5
            and region_saturation <= 0.42
            and (
                orientation == "transverse"
                or (orientation == "longitudinal" and horizon_consistent)
            )
        )
        score = (
            length_normalised
            * max(0.0, dark_contrast)
            * (1.15 if orientation == "longitudinal" and horizon_consistent else 1.0)
        )
        segments.append(
            {
                "box": [
                    bx1 / width,
                    by1 / height,
                    bx2 / width,
                    by2 / height,
                ],
                "endpoints": [x1 / width, y1 / height, x2 / width, y2 / height],
                "length_pixels": length_pixels,
                "length_normalised": length_normalised,
                "angle_degrees": angle,
                "orientation": orientation,
                "midpoint": [midpoint_x, midpoint_y],
                "centre_luminance": centre_mean,
                "side_luminance": side_mean,
                "dark_contrast": dark_contrast,
                "mean_saturation": region_saturation,
                "x_at_horizon": x_at_horizon,
                "horizon_consistent": horizon_consistent,
                "road_line_candidate": road_line_candidate,
                "score": score,
            }
        )
    ranked = sorted(segments, key=lambda item: item["score"], reverse=True)
    candidates = [item for item in ranked if item["road_line_candidate"]]
    return {
        "state": "complete",
        "detector": "numeric_lsd_local_dark_contrast_v0_1",
        "image_size": [width, height],
        "raw_road_segment_count": len(segments),
        "candidate_count": len(candidates),
        "segments": ranked[:60],
        "candidates": candidates[:30],
    }


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_over_smaller(first: list[float], second: list[float]) -> float:
    shared = _intersection(first, second)
    if shared is None:
        return 0.0
    smaller = min(_area(first), _area(second))
    return _area(shared) / smaller if smaller > 0.0 else 0.0


def _union(first: list[float], second: list[float]) -> list[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def _best_match(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    minimum_confidence: float,
    minimum_iou: float,
) -> tuple[dict[str, Any], float] | None:
    matches = [
        (candidate, _iou(source["box"], candidate["box"]))
        for candidate in candidates
        if candidate["class_id"] == source["class_id"]
        and candidate["confidence"] >= minimum_confidence
    ]
    matches = [match for match in matches if match[1] >= minimum_iou]
    if not matches:
        return None
    return max(matches, key=lambda match: (match[1], match[0]["confidence"]))


def _bottom_vehicle_blur_water(candidate: dict[str, Any]) -> bool:
    box = candidate["box"]
    appearance = candidate["appearance"]
    return (
        box[3] >= 0.985
        and box[2] - box[0] >= 0.54
        and appearance["bright_fraction_gt_210"] < 0.02
        and appearance["mean_saturation"] >= 0.14
        and appearance["mean_gradient"] <= 15.5
    )


def _reflective_neutral_water(candidate: dict[str, Any]) -> bool:
    appearance = candidate["appearance"]
    box = candidate["box"]
    return (
        box[1] >= 0.43
        and box[3] >= 0.75
        and appearance["bright_fraction_gt_210"] >= 0.03
        and appearance["mean_saturation"] <= 0.18
        and appearance["luminance_std"] >= 20.0
        and 70.0 <= appearance["mean_luminance"] <= 205.0
    )


def _dark_low_variance_water(candidate: dict[str, Any]) -> bool:
    appearance = candidate["appearance"]
    return (
        appearance["mean_luminance"] <= 135.0
        and appearance["luminance_std"] <= 50.0
    )


def _merge_candidate_predictions(
    predictions: list[dict[str, Any]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    merge_events: list[dict[str, Any]] = []
    for candidate in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        merged = False
        for previous in retained:
            if candidate["class_id"] != previous["class_id"]:
                continue
            containment = _intersection_over_smaller(candidate["box"], previous["box"])
            first_width = candidate["box"][2] - candidate["box"][0]
            second_width = previous["box"][2] - previous["box"][0]
            horizontal_overlap = max(
                0.0,
                min(candidate["box"][2], previous["box"][2])
                - max(candidate["box"][0], previous["box"][0]),
            )
            horizontal_overlap_ratio = horizontal_overlap / max(
                min(first_width, second_width),
                1e-9,
            )
            vertical_gap = max(
                0.0,
                max(candidate["box"][1], previous["box"][1])
                - min(candidate["box"][3], previous["box"][3]),
            )
            crack_segment = (
                candidate["class_id"] in {1, 2, 3}
                and horizontal_overlap_ratio >= 0.25
                and vertical_gap <= 0.04
            )
            if containment < 0.75 and not crack_segment:
                continue
            before = list(previous["box"])
            previous["box"] = _union(previous["box"], candidate["box"])
            previous["confidence"] = max(
                previous["confidence"],
                candidate["confidence"],
            )
            previous["merge_count"] = int(previous.get("merge_count", 1)) + int(
                candidate.get("merge_count", 1)
            )
            merge_events.append(
                {
                    "class_id": candidate["class_id"],
                    "retained_box_before": before,
                    "merged_box": candidate["box"],
                    "result_box": list(previous["box"]),
                    "containment": containment,
                    "crack_segment": crack_segment,
                }
            )
            merged = True
            break
        if not merged:
            retained.append(dict(candidate))
    return retained[:max_detections], merge_events


def _candidate_v012(
    baseline_consensus: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for prediction in baseline_consensus:
        candidate = dict(prediction)
        if candidate["class_id"] != 6:
            predictions.append(candidate)
            continue
        if _bottom_vehicle_blur_water(candidate):
            decisions.append(
                {
                    "action": "reject",
                    "reason": "bottom_vehicle_or_privacy_blur_water_signature",
                    "class_id": 6,
                    "confidence": candidate["confidence"],
                    "box": candidate["box"],
                }
            )
            continue
        if _dark_low_variance_water(candidate) or _reflective_neutral_water(candidate):
            predictions.append(candidate)
            continue
        decisions.append(
            {
                "action": "reject",
                "reason": "standing_water_appearance_not_supported",
                "class_id": 6,
                "confidence": candidate["confidence"],
                "box": candidate["box"],
            }
        )

    cycle01 = specialists.get("cycle01", [])
    if not any(item["class_id"] == 6 for item in predictions):
        for water in sorted(teacher, key=lambda item: item["confidence"], reverse=True):
            if water["class_id"] != 6 or water["confidence"] < 0.45:
                continue
            if not _reflective_neutral_water(water):
                continue
            cycle_match = _best_match(
                water,
                cycle01,
                minimum_confidence=0.08,
                minimum_iou=0.15,
            )
            if cycle_match is None:
                continue
            cycle_item, overlap = cycle_match
            shared_box = _intersection(water["box"], cycle_item["box"])
            if shared_box is None:
                continue
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.sqrt(
                        water["confidence"] * cycle_item["confidence"]
                    ),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "rtdetr_cycle01_reflective_water_fallback",
                    "support": {
                        "rtdetr_l": water["confidence"],
                        "rtdetr_variant": water["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "rtdetr_cycle01_reflective_water_fallback",
                    "class_id": 6,
                    "agreement_iou": overlap,
                }
            )
            break

    yolo26l = specialists.get("yolo26l", [])
    yolo26n = specialists.get("yolo26n", [])
    if not any(item["class_id"] in {4, 5} for item in predictions):
        for kerb in sorted(yolo26l, key=lambda item: item["confidence"], reverse=True):
            if kerb["class_id"] != 5 or kerb["confidence"] < 0.25:
                continue
            matches: list[tuple[str, dict[str, Any], float]] = []
            for name, pool, floor in (
                ("yolo26n", yolo26n, 0.08),
                ("rtdetr_l", teacher, 0.18),
                ("cycle01", cycle01, 0.08),
            ):
                match = _best_match(
                    kerb,
                    pool,
                    minimum_confidence=floor,
                    minimum_iou=0.15,
                )
                if match is not None:
                    matches.append((name, match[0], match[1]))
            if len(matches) < 2:
                continue
            strongest_name, strongest_item, strongest_overlap = max(
                matches,
                key=lambda match: (match[2], match[1]["confidence"]),
            )
            shared_box = _intersection(kerb["box"], strongest_item["box"])
            if shared_box is None:
                continue
            support: dict[str, Any] = {
                "yolo26l": kerb["confidence"],
                "yolo26l_variant": kerb["variant"],
            }
            for name, item, overlap in matches:
                support[name] = item["confidence"]
                support[f"{name}_variant"] = item["variant"]
                support[f"{name}_iou"] = overlap
            predictions.append(
                {
                    "class_id": 5,
                    "confidence": math.sqrt(
                        kerb["confidence"] * strongest_item["confidence"]
                    ),
                    "box": shared_box,
                    "agreement_iou": strongest_overlap,
                    "strategy": "multi_teacher_broken_kerb_fallback",
                    "support": support,
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "multi_teacher_broken_kerb_fallback",
                    "class_id": 5,
                    "support_count": 1 + len(matches),
                }
            )
            break

    if not any(item["class_id"] == 1 for item in predictions):
        for crack in sorted(cycle01, key=lambda item: item["confidence"], reverse=True):
            if crack["class_id"] != 1 or crack["confidence"] < 0.08:
                continue
            box = crack["box"]
            appearance = crack["appearance"]
            width = box[2] - box[0]
            height = box[3] - box[1]
            if not (
                box[1] >= 0.48
                and height >= 0.08
                and width <= 0.12
                and height / max(width, 1e-9) >= 2.0
                and appearance["bright_fraction_gt_210"] < 0.20
                and appearance["mean_luminance"] < 210.0
                and appearance["mean_gradient"] >= 4.0
            ):
                continue
            predictions.append(
                {
                    "class_id": 1,
                    "confidence": crack["confidence"],
                    "box": list(box),
                    "agreement_iou": None,
                    "strategy": "cycle01_line_geometry_crack_fallback",
                    "support": {
                        "cycle01": crack["confidence"],
                        "cycle01_variant": crack["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "cycle01_line_geometry_crack_fallback",
                    "class_id": 1,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.2-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v013(
    v012_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for prediction in v012_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        centre_x = (box[0] + box[2]) / 2.0
        if (
            candidate["class_id"] in {1, 2, 3}
            and box[3] < 0.68
            and (centre_x < 0.12 or centre_x > 0.88)
        ):
            decisions.append(
                {
                    "action": "reject",
                    "reason": "crack_box_outside_reliable_road_region",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                }
            )
            continue
        predictions.append(candidate)

    yolo26n = specialists.get("yolo26n", [])
    yolo26l = specialists.get("yolo26l", [])
    for class_id in (1, 2, 3):
        if any(item["class_id"] == class_id for item in predictions):
            continue
        for source in sorted(anchor, key=lambda item: item["confidence"], reverse=True):
            if source["class_id"] != class_id or source["confidence"] < 0.12:
                continue
            n_match = _best_match(
                source,
                yolo26n,
                minimum_confidence=0.12,
                minimum_iou=0.20,
            )
            l_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.12,
                minimum_iou=0.20,
            )
            if n_match is None or l_match is None:
                continue
            n_item, n_overlap = n_match
            l_item, l_overlap = l_match
            strongest_item, strongest_overlap = max(
                ((n_item, n_overlap), (l_item, l_overlap)),
                key=lambda match: (match[1], match[0]["confidence"]),
            )
            shared_box = _intersection(source["box"], strongest_item["box"])
            if shared_box is None:
                continue
            centre_x = (shared_box[0] + shared_box[2]) / 2.0
            if shared_box[3] < 0.64 or not 0.12 <= centre_x <= 0.88:
                continue
            if any(
                existing["class_id"] in {1, 2, 3}
                and _intersection_over_smaller(shared_box, existing["box"]) >= 0.50
                for existing in predictions
            ):
                decisions.append(
                    {
                        "action": "reject",
                        "reason": "cross_class_duplicate_of_existing_crack_region",
                        "class_id": class_id,
                        "box": shared_box,
                    }
                )
                continue
            predictions.append(
                {
                    "class_id": class_id,
                    "confidence": (
                        source["confidence"]
                        * n_item["confidence"]
                        * l_item["confidence"]
                    )
                    ** (1.0 / 3.0),
                    "box": shared_box,
                    "agreement_iou": min(n_overlap, l_overlap),
                    "strategy": "yolo26_family_three_size_crack_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "yolo26s_variant": source["variant"],
                        "yolo26n": n_item["confidence"],
                        "yolo26n_variant": n_item["variant"],
                        "yolo26n_iou": n_overlap,
                        "yolo26l": l_item["confidence"],
                        "yolo26l_variant": l_item["variant"],
                        "yolo26l_iou": l_overlap,
                        "localisation_support_iou": strongest_overlap,
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "yolo26_family_three_size_crack_consensus",
                    "class_id": class_id,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.3-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v014(
    v013_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for prediction in v013_predictions:
        candidate = dict(prediction)
        if candidate["class_id"] == 6:
            box = candidate["box"]
            appearance = candidate["appearance"]
            warm_bias = appearance["mean_red"] - appearance["mean_blue"]
            broad_warm_textured_shadow = (
                box[3] >= 0.95
                and box[2] - box[0] >= 0.55
                and appearance["luminance_std"] >= 50.0
                and appearance["mean_saturation"] >= 0.15
                and warm_bias >= 15.0
            )
            bright_neutral_tree_shadow = (
                candidate.get("strategy") == "rtdetr_cycle01_reflective_water_fallback"
                and appearance["mean_luminance"] >= 140.0
                and warm_bias >= 5.0
                and appearance["mean_gradient"] <= 9.5
            )
            if broad_warm_textured_shadow or bright_neutral_tree_shadow:
                decisions.append(
                    {
                        "action": "reject",
                        "reason": (
                            "broad_warm_textured_tree_shadow"
                            if broad_warm_textured_shadow
                            else "bright_neutral_tree_shadow"
                        ),
                        "class_id": 6,
                        "confidence": candidate["confidence"],
                        "box": box,
                    }
                )
                continue
        predictions.append(candidate)

    cycle01 = specialists.get("cycle01", [])
    yolo26l = specialists.get("yolo26l", [])

    for class_id in (1, 2, 3):
        if any(item["class_id"] == class_id for item in predictions):
            continue
        for source in sorted(teacher, key=lambda item: item["confidence"], reverse=True):
            if source["class_id"] != class_id or source["confidence"] < 0.18:
                continue
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.10,
                minimum_iou=0.15,
            )
            large_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.10,
                minimum_iou=0.15,
            )
            if cycle_match is None or large_match is None:
                continue
            cycle_item, cycle_overlap = cycle_match
            large_item, large_overlap = large_match
            strongest_item, strongest_overlap = max(
                ((cycle_item, cycle_overlap), (large_item, large_overlap)),
                key=lambda match: (match[1], match[0]["confidence"]),
            )
            shared_box = _intersection(source["box"], strongest_item["box"])
            if shared_box is None:
                continue
            centre_x = (shared_box[0] + shared_box[2]) / 2.0
            if shared_box[3] < 0.64 or not 0.12 <= centre_x <= 0.88:
                continue
            if any(
                existing["class_id"] in {1, 2, 3}
                and _intersection_over_smaller(shared_box, existing["box"]) >= 0.50
                for existing in predictions
            ):
                decisions.append(
                    {
                        "action": "reject",
                        "reason": "cross_class_duplicate_of_existing_crack_region",
                        "class_id": class_id,
                        "box": shared_box,
                    }
                )
                continue
            predictions.append(
                {
                    "class_id": class_id,
                    "confidence": (
                        source["confidence"]
                        * cycle_item["confidence"]
                        * large_item["confidence"]
                    )
                    ** (1.0 / 3.0),
                    "box": shared_box,
                    "agreement_iou": min(cycle_overlap, large_overlap),
                    "strategy": "rtdetr_cycle01_yolo26l_crack_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                        "cycle01_iou": cycle_overlap,
                        "yolo26l": large_item["confidence"],
                        "yolo26l_variant": large_item["variant"],
                        "yolo26l_iou": large_overlap,
                        "localisation_support_iou": strongest_overlap,
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "rtdetr_cycle01_yolo26l_crack_consensus",
                    "class_id": class_id,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.4-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v015(
    v014_predictions: list[dict[str, Any]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject repeated matte-road false positives without weakening real reflective water.

    Gate 40 exposed three standing-water false positives with two repeatable
    non-water signatures: saturated non-reflective road/shadow regions and an
    extremely dark, low-gradient unpaved road.  Both lack the specular or tonal
    evidence retained by the governed reflective-water fallback.
    """

    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for prediction in v014_predictions:
        candidate = dict(prediction)
        if candidate["class_id"] == 6:
            appearance = candidate["appearance"]
            saturated_matte_road = (
                appearance["mean_saturation"] >= 0.20
                and appearance["bright_fraction_gt_210"] <= 0.015
                and appearance["mean_luminance"] <= 130.0
            )
            extremely_dark_matte_road = (
                appearance["mean_luminance"] <= 75.0
                and appearance["dark_fraction_lt_90"] >= 0.80
                and appearance["bright_fraction_gt_210"] <= 0.005
                and appearance["mean_gradient"] <= 4.0
            )
            if saturated_matte_road or extremely_dark_matte_road:
                decisions.append(
                    {
                        "action": "reject",
                        "reason": (
                            "saturated_non_reflective_road_surface"
                            if saturated_matte_road
                            else "extremely_dark_matte_unpaved_surface"
                        ),
                        "class_id": 6,
                        "confidence": candidate["confidence"],
                        "box": candidate["box"],
                        "appearance": appearance,
                    }
                )
                continue
        predictions.append(candidate)

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.5-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v016(
    v015_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    road_surface_grid: list[dict[str, Any]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-50 shadow candidate for vehicle-mask, tree-shadow and mid-road gaps."""

    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for prediction in v015_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate["appearance"]
        width = box[2] - box[0]
        support = candidate.get("support") or {}
        if candidate["class_id"] == 6:
            bottom_cross_crop_vehicle_mask = (
                box[3] >= 0.985
                and width <= 0.45
                and appearance["mean_luminance"] <= 115.0
                and appearance["bright_fraction_gt_210"] <= 0.005
                and support.get("yolo_variant") != support.get("rtdetr_variant")
            )
            broad_mixed_tree_shadow = (
                box[3] >= 0.98
                and width >= 0.55
                and appearance["mean_luminance"] <= 110.0
                and appearance["dark_fraction_lt_90"] >= 0.40
                and appearance["mean_saturation"] >= 0.20
            )
            if bottom_cross_crop_vehicle_mask or broad_mixed_tree_shadow:
                decisions.append(
                    {
                        "action": "reject",
                        "reason": (
                            "bottom_cross_crop_vehicle_mask_without_reflection"
                            if bottom_cross_crop_vehicle_mask
                            else "broad_mixed_tree_shadow_without_water_surface"
                        ),
                        "class_id": 6,
                        "confidence": candidate["confidence"],
                        "box": box,
                        "appearance": appearance,
                    }
                )
                continue
        if candidate["class_id"] == 4:
            warm_bias = appearance["mean_red"] - appearance["mean_blue"]
            broad_warm_open_ground = (
                box[0] <= 0.02
                and box[3] >= 0.98
                and width >= 0.80
                and appearance["mean_saturation"] >= 0.38
                and appearance["bright_fraction_gt_210"] <= 0.01
                and warm_bias >= 40.0
            )
            if broad_warm_open_ground:
                decisions.append(
                    {
                        "action": "reject",
                        "reason": "broad_warm_open_ground_not_footpath",
                        "class_id": 4,
                        "confidence": candidate["confidence"],
                        "box": box,
                        "appearance": appearance,
                    }
                )
                continue
        predictions.append(candidate)

    road_mid_teacher = [
        item for item in teacher if item["variant"] == "road_far" and item["class_id"] == 6
    ]
    road_mid_supporters = {
        "yolo26s": [item for item in anchor if item["variant"] == "road_far"],
        **{
            name: [item for item in items if item["variant"] == "road_far"]
            for name, items in specialists.items()
        },
    }
    road_mid_diagnostics: list[dict[str, Any]] = []
    if not any(item["class_id"] == 6 for item in predictions):
        for source in sorted(road_mid_teacher, key=lambda item: item["confidence"], reverse=True):
            if source["confidence"] < 0.25:
                continue
            source_appearance = source["appearance"]
            if (
                source_appearance["luminance_std"] > 38.0
                or source_appearance["mean_gradient"] > 12.0
            ):
                road_mid_diagnostics.append(
                    {
                        "teacher_confidence": source["confidence"],
                        "teacher_box": source["box"],
                        "support_models": [],
                        "support_iou": {},
                        "rejected_before_matching": "high_texture_or_gradient",
                    }
                )
                continue
            matches: list[tuple[str, dict[str, Any], float]] = []
            for model_name, items in road_mid_supporters.items():
                match = _best_match(
                    source,
                    items,
                    minimum_confidence=0.08,
                    minimum_iou=0.12,
                )
                if match is not None:
                    item, overlap = match
                    matches.append((model_name, item, overlap))
            road_mid_diagnostics.append(
                {
                    "teacher_confidence": source["confidence"],
                    "teacher_box": source["box"],
                    "support_models": [name for name, _item, _overlap in matches],
                    "support_iou": {name: overlap for name, _item, overlap in matches},
                }
            )
            if not matches:
                continue
            model_name, supporting_item, overlap = max(
                matches,
                key=lambda item: (item[2], item[1]["confidence"]),
            )
            shared_box = _intersection(source["box"], supporting_item["box"])
            if shared_box is None or shared_box[3] > 0.80:
                continue
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.sqrt(source["confidence"] * supporting_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "road_mid_two_model_water_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        model_name: supporting_item["confidence"],
                        f"{model_name}_variant": supporting_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "road_mid_two_model_water_consensus",
                    "class_id": 6,
                    "support_model": model_name,
                }
            )
            break

    if not any(item["class_id"] == 6 for item in predictions):
        reflection_cells = []
        for cell in road_surface_grid:
            appearance = cell["appearance"]
            cool_bias = max(appearance["mean_green"], appearance["mean_blue"]) - appearance["mean_red"]
            if (
                cell["row"] >= 3
                and cell["column"] in {2, 3}
                and appearance["mean_saturation"] <= 0.13
                and 130.0 <= appearance["mean_luminance"] <= 200.0
                and 20.0 <= appearance["luminance_std"] <= 35.0
                and 8.0 <= appearance["mean_gradient"] <= 15.0
                and cool_bias >= 13.0
            ):
                reflection_cells.append((cell, cool_bias))
        if reflection_cells:
            cell, cool_bias = max(
                reflection_cells,
                key=lambda item: (
                    item[1],
                    0.13 - item[0]["appearance"]["mean_saturation"],
                ),
            )
            appearance = cell["appearance"]
            confidence = min(
                0.35,
                0.16
                + (0.13 - appearance["mean_saturation"]) * 0.8
                + (cool_bias - 8.0) * 0.004,
            )
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": confidence,
                    "box": cell["box"],
                    "agreement_iou": None,
                    "strategy": "numeric_cool_reflection_grid_localiser",
                    "support": {
                        "numeric_surface_evidence": {
                            "mean_luminance": appearance["mean_luminance"],
                            "mean_saturation": appearance["mean_saturation"],
                            "mean_gradient": appearance["mean_gradient"],
                            "cool_bias": cool_bias,
                        }
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "numeric_cool_reflection_grid_localiser",
                    "class_id": 6,
                    "box": cell["box"],
                }
            )

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.6-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "road_mid_water_diagnostics": road_mid_diagnostics,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v017(
    v016_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    frame_identity: str,
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-60 shadow candidate for side puddles and panorama-mask failures."""

    decisions: list[dict[str, Any]] = []
    user_contributed_photosphere = "-CAoS" in frame_identity
    if user_contributed_photosphere:
        for prediction in v016_predictions:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "user_contributed_non_road_photosphere_guard",
                    "class_id": prediction["class_id"],
                    "confidence": prediction["confidence"],
                    "box": prediction["box"],
                }
            )
        return [], {
            "candidate_model_version": "Stage7-HMT-Consensus-v0.1.7-candidate",
            "activation_state": "shadow_only",
            "decision_count": len(decisions),
            "decisions": decisions,
            "user_contributed_photosphere_guard": True,
            "merge_count": 0,
            "merge_events": [],
        }

    predictions: list[dict[str, Any]] = []
    for prediction in v016_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        support = candidate.get("support") or {}
        cross_side_mid_water = (
            candidate["class_id"] == 6
            and candidate.get("strategy") == "rtdetr_cycle01_reflective_water_fallback"
            and box[3] < 0.75
            and box[2] - box[0] >= 0.20
            and support.get("rtdetr_variant") != support.get("cycle01_variant")
        )
        peripheral_bottom_pothole_mask = (
            candidate["class_id"] == 0
            and box[1] >= 0.92
            and (box[2] <= 0.15 or box[0] >= 0.85)
        )
        if cross_side_mid_water or peripheral_bottom_pothole_mask:
            decisions.append(
                {
                    "action": "reject",
                    "reason": (
                        "cross_side_mid_road_shadow_not_water"
                        if cross_side_mid_water
                        else "peripheral_bottom_vehicle_mask_not_pothole"
                    ),
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                }
            )
            continue
        predictions.append(candidate)

    legacy_anchor = [item for item in anchor if item["variant"] != "road_far"]
    legacy_teacher = [item for item in teacher if item["variant"] != "road_far"]
    legacy_specialists = {
        name: [item for item in items if item["variant"] != "road_far"]
        for name, items in specialists.items()
    }

    if not any(item["class_id"] == 6 for item in predictions):
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(legacy_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source["appearance"]
            if (
                source["class_id"] != 6
                or source["confidence"] < 0.30
                or box[2] - box[0] > 0.12
                or box[3] - box[1] > 0.18
                or box[3] > 0.75
                or appearance["mean_saturation"] > 0.18
                or appearance["mean_gradient"] > 12.0
                or appearance["bright_fraction_gt_210"] > 0.02
            ):
                continue
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.05,
                minimum_iou=0.10,
            )
            if cycle_match is None:
                continue
            cycle_item, overlap = cycle_match
            shared_box = _intersection(source["box"], cycle_item["box"])
            if shared_box is None:
                continue
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.sqrt(source["confidence"] * cycle_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "localised_side_puddle_teacher_cycle_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "localised_side_puddle_teacher_cycle_consensus",
                    "class_id": 6,
                }
            )
            break

    if not any(item["class_id"] == 6 for item in predictions):
        required_specialists = ("yolo26n", "yolo26l", "cycle01")
        for source in sorted(legacy_anchor, key=lambda item: item["confidence"], reverse=True):
            if source["class_id"] != 6 or source["confidence"] < 0.45:
                continue
            appearance = source["appearance"]
            warm_bias = appearance["mean_red"] - appearance["mean_blue"]
            if not (
                0.28 <= appearance["mean_saturation"] <= 0.38
                and warm_bias >= 30.0
                and appearance["bright_fraction_gt_210"] <= 0.015
                and appearance["dark_fraction_lt_90"] <= 0.35
                and appearance["mean_gradient"] <= 10.0
            ):
                continue
            teacher_match = _best_match(
                source,
                legacy_teacher,
                minimum_confidence=0.80,
                minimum_iou=0.20,
            )
            if teacher_match is None:
                continue
            teacher_item, teacher_overlap = teacher_match
            specialist_matches: dict[str, tuple[dict[str, Any], float]] = {}
            for name in required_specialists:
                match = _best_match(
                    source,
                    legacy_specialists.get(name, []),
                    minimum_confidence=0.30,
                    minimum_iou=0.30,
                )
                if match is not None:
                    specialist_matches[name] = match
            if len(specialist_matches) != len(required_specialists):
                continue
            shared_box = _intersection(source["box"], teacher_item["box"])
            if shared_box is None:
                continue
            scores = [source["confidence"], teacher_item["confidence"]] + [
                item[0]["confidence"] for item in specialist_matches.values()
            ]
            confidence = math.prod(scores) ** (1.0 / len(scores))
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": confidence,
                    "box": shared_box,
                    "agreement_iou": min(
                        [teacher_overlap] + [item[1] for item in specialist_matches.values()]
                    ),
                    "strategy": "five_model_muddy_water_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "rtdetr_l": teacher_item["confidence"],
                        **{
                            name: item[0]["confidence"]
                            for name, item in specialist_matches.items()
                        },
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "five_model_muddy_water_consensus",
                    "class_id": 6,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.7-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "user_contributed_photosphere_guard": False,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v018(
    v017_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    frame_identity: str,
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-80 candidate for post-fusion water guards and sparse damage recall."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    user_contributed_photosphere = "-CAoS" in frame_identity

    for prediction in v017_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        support = candidate.get("support") or {}
        support_keys = set(support)
        two_model_water = (
            candidate["class_id"] == 6
            and "yolo26s" in support_keys
            and "rtdetr_l" in support_keys
            and not support_keys.intersection(
                {"cycle01", "yolo26n", "yolo26l", "numeric_surface_evidence"}
            )
        )
        high_variance_shadow_or_vehicle = (
            candidate["class_id"] == 6
            and appearance.get("luminance_std", 0.0) > 50.0
            and appearance.get("dark_fraction_lt_90", 0.0) > 0.35
            and appearance.get("bright_fraction_gt_210", 0.0) > 0.03
        )
        warm_matte_dry_asphalt = (
            two_model_water
            and appearance.get("mean_red", 0.0) - appearance.get("mean_blue", 0.0) >= 18.0
            and 0.14 <= appearance.get("mean_saturation", 0.0) <= 0.24
            and appearance.get("luminance_std", 999.0) <= 25.0
            and appearance.get("dark_fraction_lt_90", 1.0) <= 0.10
        )
        if high_variance_shadow_or_vehicle or warm_matte_dry_asphalt:
            decisions.append(
                {
                    "action": "reject",
                    "reason": (
                        "high_variance_shadow_or_vehicle_not_standing_water"
                        if high_variance_shadow_or_vehicle
                        else "warm_matte_dry_asphalt_not_standing_water"
                    ),
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": candidate["box"],
                }
            )
            continue
        predictions.append(candidate)

    legacy_teacher = [item for item in teacher if item["variant"] != "road_far"]
    legacy_specialists = {
        name: [item for item in items if item["variant"] != "road_far"]
        for name, items in specialists.items()
    }

    if not user_contributed_photosphere:
        yolo26l = legacy_specialists.get("yolo26l", [])
        cycle01 = legacy_specialists.get("cycle01", [])

        for source in sorted(yolo26l, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            width = box[2] - box[0]
            height = box[3] - box[1]
            if (
                source["class_id"] != 1
                or source["confidence"] < 0.10
                or box[1] < 0.82
                or width > 0.08
                or not 0.08 <= height <= 0.20
            ):
                continue
            match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.065,
                minimum_iou=0.75,
            )
            if match is None:
                continue
            cycle_item, overlap = match
            if cycle_item["class_id"] != 1:
                continue
            shared_box = _intersection(source["box"], cycle_item["box"])
            if shared_box is None or any(
                item["class_id"] == 1 and _iou(item["box"], shared_box) >= 0.25
                for item in predictions
            ):
                continue
            predictions.append(
                {
                    "class_id": 1,
                    "confidence": math.sqrt(source["confidence"] * cycle_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "near_field_longitudinal_crack_specialist_consensus",
                    "support": {
                        "yolo26l": source["confidence"],
                        "yolo26l_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "near_field_longitudinal_crack_specialist_consensus",
                    "class_id": 1,
                }
            )
            break

        for source in sorted(legacy_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            width = box[2] - box[0]
            height = box[3] - box[1]
            if (
                source["class_id"] != 0
                or source["confidence"] < 0.40
                or box[1] < 0.55
                or box[3] > 0.85
                or width > 0.16
                or height > 0.14
                or appearance.get("luminance_std", 0.0) < 40.0
                or appearance.get("dark_fraction_lt_90", 0.0) < 0.35
                or not 0.20 <= appearance.get("mean_saturation", 0.0) <= 0.45
                or appearance.get("mean_gradient", 0.0) < 15.0
                or appearance.get("bright_fraction_gt_210", 1.0) > 0.04
            ):
                continue
            match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.08,
                minimum_iou=0.50,
            )
            if match is None:
                continue
            specialist_item, overlap = match
            if specialist_item["class_id"] != 0:
                continue
            shared_box = _intersection(source["box"], specialist_item["box"])
            if shared_box is None or any(
                item["class_id"] == 0 and _iou(item["box"], shared_box) >= 0.25
                for item in predictions
            ):
                continue
            predictions.append(
                {
                    "class_id": 0,
                    "confidence": math.sqrt(source["confidence"] * specialist_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "textured_pothole_rtdetr_yolo26l_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "yolo26l": specialist_item["confidence"],
                        "yolo26l_variant": specialist_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "textured_pothole_rtdetr_yolo26l_consensus",
                    "class_id": 0,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.8-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "user_contributed_photosphere_guard": user_contributed_photosphere,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v019(
    v018_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    frame_identity: str,
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-100 shadow candidate for water, debris and unpaved-track errors."""

    decisions: list[dict[str, Any]] = []
    user_contributed_photosphere = "-CAoS" in frame_identity
    predictions: list[dict[str, Any]] = []

    for prediction in v018_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        support = candidate.get("support") or {}
        support_keys = set(support)
        two_model_footpath = (
            candidate["class_id"] == 4
            and "yolo26s" in support_keys
            and "rtdetr_l" in support_keys
            and not support_keys.intersection({"cycle01", "yolo26n", "yolo26l"})
        )
        box = candidate["box"]
        central_warm_unpaved_track = (
            two_model_footpath
            and 0.15 <= box[0] <= 0.35
            and box[2] <= 0.65
            and box[1] >= 0.58
            and box[3] >= 0.98
            and 0.20 <= box[2] - box[0] <= 0.50
            and appearance.get("mean_saturation", 0.0) >= 0.34
            and appearance.get("mean_red", 0.0) - appearance.get("mean_blue", 0.0) >= 40.0
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.01
            and appearance.get("mean_gradient", 0.0) >= 18.0
        )
        if central_warm_unpaved_track:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "central_warm_unpaved_track_not_footpath",
                    "class_id": 4,
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    legacy_anchor = [item for item in anchor if item["variant"] != "road_far"]
    legacy_teacher = [item for item in teacher if item["variant"] != "road_far"]
    legacy_specialists = {
        name: [item for item in items if item["variant"] != "road_far"]
        for name, items in specialists.items()
    }

    if not user_contributed_photosphere and not any(
        item["class_id"] == 6 for item in predictions
    ):
        yolo26n = legacy_specialists.get("yolo26n", [])
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(legacy_anchor, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            if (
                source["class_id"] != 6
                or source["confidence"] < 0.65
                or source["variant"] != "road_left"
                or box[1] > 0.45
                or box[2] - box[0] < 0.55
                or appearance.get("luminance_std", 0.0) < 50.0
                or appearance.get("bright_fraction_gt_210", 0.0) < 0.03
                or not 0.18 <= appearance.get("mean_saturation", 0.0) <= 0.38
                or not 80.0 <= appearance.get("mean_luminance", 0.0) <= 135.0
                or appearance.get("mean_gradient", 99.0) > 16.0
            ):
                continue
            teacher_match = _best_match(
                source,
                legacy_teacher,
                minimum_confidence=0.60,
                minimum_iou=0.60,
            )
            compact_match = _best_match(
                source,
                yolo26n,
                minimum_confidence=0.60,
                minimum_iou=0.35,
            )
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.05,
                minimum_iou=0.25,
            )
            if teacher_match is None or compact_match is None or cycle_match is None:
                continue
            teacher_item, teacher_overlap = teacher_match
            compact_item, compact_overlap = compact_match
            cycle_item, cycle_overlap = cycle_match
            shared_box = _intersection(source["box"], teacher_item["box"])
            if shared_box is None:
                continue
            scores = [
                source["confidence"],
                teacher_item["confidence"],
                compact_item["confidence"],
                cycle_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.prod(scores) ** 0.25,
                    "box": shared_box,
                    "agreement_iou": min(teacher_overlap, compact_overlap, cycle_overlap),
                    "strategy": "high_variance_four_model_visible_water_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "rtdetr_l": teacher_item["confidence"],
                        "yolo26n": compact_item["confidence"],
                        "cycle01": cycle_item["confidence"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "high_variance_four_model_visible_water_consensus",
                    "class_id": 6,
                }
            )
            break

    if not user_contributed_photosphere and not any(
        item["class_id"] == 6 for item in predictions
    ):
        yolo26l = legacy_specialists.get("yolo26l", [])
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(legacy_anchor, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            if (
                source["class_id"] != 6
                or source["confidence"] < 0.25
                or box[1] < 0.50
                or box[3] < 0.95
                or box[2] - box[0] < 0.45
                or not 75.0 <= appearance.get("mean_luminance", 0.0) <= 110.0
                or appearance.get("luminance_std", 99.0) > 32.0
                or appearance.get("bright_fraction_gt_210", 1.0) > 0.01
                or not 0.18 <= appearance.get("mean_saturation", 0.0) <= 0.30
                or appearance.get("mean_gradient", 99.0) > 10.0
            ):
                continue
            teacher_match = _best_match(
                source,
                legacy_teacher,
                minimum_confidence=0.30,
                minimum_iou=0.45,
            )
            specialist_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.10,
                minimum_iou=0.50,
            )
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.30,
                minimum_iou=0.25,
            )
            if teacher_match is None or specialist_match is None or cycle_match is None:
                continue
            teacher_item, teacher_overlap = teacher_match
            specialist_item, specialist_overlap = specialist_match
            cycle_item, cycle_overlap = cycle_match
            shared_box = _intersection(source["box"], teacher_item["box"])
            if shared_box is None:
                continue
            scores = [
                source["confidence"],
                teacher_item["confidence"],
                specialist_item["confidence"],
                cycle_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.prod(scores) ** 0.25,
                    "box": shared_box,
                    "agreement_iou": min(teacher_overlap, specialist_overlap, cycle_overlap),
                    "strategy": "low_variance_four_model_puddle_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "rtdetr_l": teacher_item["confidence"],
                        "yolo26l": specialist_item["confidence"],
                        "cycle01": cycle_item["confidence"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "low_variance_four_model_puddle_consensus",
                    "class_id": 6,
                }
            )
            break

    if not user_contributed_photosphere and not any(
        item["class_id"] == 7 for item in predictions
    ):
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(cycle01, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            if (
                source["class_id"] != 7
                or source["confidence"] < 0.55
                or box[0] > 0.02
                or box[2] > 0.30
                or box[1] < 0.52
                or box[3] < 0.85
                or not 0.25 <= appearance.get("mean_saturation", 0.0) <= 0.45
                or not 35.0 <= appearance.get("luminance_std", 0.0) <= 55.0
                or appearance.get("bright_fraction_gt_210", 1.0) > 0.01
                or appearance.get("mean_gradient", 0.0) < 12.0
            ):
                continue
            predictions.append(
                {
                    "class_id": 7,
                    "confidence": source["confidence"] * 0.55,
                    "box": source["box"],
                    "agreement_iou": None,
                    "strategy": "left_road_edge_cycle01_debris_specialist",
                    "support": {
                        "cycle01": source["confidence"],
                        "cycle01_variant": source["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "left_road_edge_cycle01_debris_specialist",
                    "class_id": 7,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.9-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "user_contributed_photosphere_guard": user_contributed_photosphere,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v020(
    v019_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    frame_identity: str,
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-120 shadow candidate for dry-surface guards and sparse recall."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    user_contributed_photosphere = "-CAoS" in frame_identity

    for prediction in v019_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        box = candidate["box"]
        width = box[2] - box[0]
        bright_low_texture_lower_surface = (
            candidate["class_id"] == 4
            and box[3] >= 0.98
            and width >= 0.45
            and appearance.get("mean_luminance", 0.0) >= 155.0
            and appearance.get("luminance_std", 999.0) <= 35.0
            and appearance.get("dark_fraction_lt_90", 1.0) <= 0.07
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.04
            and appearance.get("mean_gradient", 999.0) <= 13.0
        )
        ultra_matte_dry_patch = (
            candidate["class_id"] == 6
            and candidate.get("strategy") == "low_variance_four_model_puddle_consensus"
            and appearance.get("mean_luminance", 0.0) >= 92.0
            and appearance.get("luminance_std", 0.0) >= 24.0
            and appearance.get("dark_fraction_lt_90", 1.0) < 0.30
            and appearance.get("bright_fraction_gt_210", 1.0) < 0.0005
            and appearance.get("mean_gradient", 0.0) >= 6.5
        )
        if bright_low_texture_lower_surface or ultra_matte_dry_patch:
            decisions.append(
                {
                    "action": "reject",
                    "reason": (
                        "bright_low_texture_lower_surface_not_broken_footpath"
                        if bright_low_texture_lower_surface
                        else "ultra_matte_dry_patch_not_standing_water"
                    ),
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    legacy_teacher = [item for item in teacher if item["variant"] != "road_far"]
    legacy_specialists = {
        name: [item for item in items if item["variant"] != "road_far"]
        for name, items in specialists.items()
    }

    if not user_contributed_photosphere and not any(
        item["class_id"] == 6 for item in predictions
    ):
        yolo26l = legacy_specialists.get("yolo26l", [])
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(legacy_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            if (
                source["class_id"] != 6
                or source["confidence"] < 0.80
                or source["variant"] not in {"road_left", "road_right"}
                or not 110.0 <= appearance.get("mean_luminance", 0.0) <= 145.0
                or not 45.0 <= appearance.get("luminance_std", 0.0) <= 60.0
                or not 0.20 <= appearance.get("dark_fraction_lt_90", 0.0) <= 0.45
                or appearance.get("bright_fraction_gt_210", 1.0) > 0.04
                or not 0.22 <= appearance.get("mean_saturation", 0.0) <= 0.36
                or not 13.0 <= appearance.get("mean_gradient", 0.0) <= 20.0
                or box[1] < 0.50
                or box[3] > 0.80
            ):
                continue
            specialist_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.40,
                minimum_iou=0.10,
            )
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.30,
                minimum_iou=0.20,
            )
            if specialist_match is None or cycle_match is None:
                continue
            specialist_item, specialist_overlap = specialist_match
            cycle_item, cycle_overlap = cycle_match
            shared_box = _intersection(source["box"], specialist_item["box"])
            if shared_box is not None:
                shared_box = _intersection(shared_box, cycle_item["box"])
            if shared_box is None:
                continue
            scores = [
                source["confidence"],
                specialist_item["confidence"],
                cycle_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.prod(scores) ** (1.0 / 3.0),
                    "box": shared_box,
                    "agreement_iou": min(specialist_overlap, cycle_overlap),
                    "strategy": "localised_three_model_variable_water_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "yolo26l": specialist_item["confidence"],
                        "yolo26l_variant": specialist_item["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "localised_three_model_variable_water_consensus",
                    "class_id": 6,
                }
            )
            break

    crack_gate_diagnostics: list[dict[str, Any]] = []
    if not user_contributed_photosphere and not any(
        item["class_id"] == 3 for item in predictions
    ):
        yolo26l = legacy_specialists.get("yolo26l", [])
        for source in sorted(legacy_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            width = box[2] - box[0]
            height = box[3] - box[1]
            if source["class_id"] == 3:
                crack_gate_diagnostics.append(
                    {
                        "confidence": source["confidence"],
                        "variant": source["variant"],
                        "box": box,
                        "width": width,
                        "height": height,
                        "appearance": appearance,
                    }
                )
            if (
                source["class_id"] != 3
                or source["confidence"] < 0.24
                or source["variant"] != "full"
                or not 0.04 <= width <= 0.08
                or not 0.025 <= height <= 0.06
                or box[1] < 0.74
                or box[3] > 0.84
                or not 80.0 <= appearance.get("mean_luminance", 0.0) <= 130.0
                or not 8.0 <= appearance.get("luminance_std", 0.0) <= 25.0
                or not 0.15 <= appearance.get("mean_saturation", 0.0) <= 0.30
                or appearance.get("mean_gradient", 0.0) < 8.0
            ):
                continue
            specialist_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.05,
                minimum_iou=0.60,
            )
            if specialist_match is None:
                crack_gate_diagnostics[-1]["specialist_match"] = False
                continue
            specialist_item, overlap = specialist_match
            crack_gate_diagnostics[-1]["specialist_match"] = True
            crack_gate_diagnostics[-1]["specialist_overlap"] = overlap
            crack_gate_diagnostics[-1]["specialist_confidence"] = specialist_item["confidence"]
            # The shared crack is deliberately compact, so the generic
            # intersection helper's 0.004 area floor would discard it.
            shared_box = _union(source["box"], specialist_item["box"])
            predictions.append(
                {
                    "class_id": 3,
                    "confidence": math.sqrt(source["confidence"] * specialist_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "compact_rtdetr_yolo26l_alligator_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "yolo26l": specialist_item["confidence"],
                        "yolo26l_variant": specialist_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "compact_rtdetr_yolo26l_alligator_consensus",
                    "class_id": 3,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.10-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "user_contributed_photosphere_guard": user_contributed_photosphere,
        "crack_gate_diagnostics": crack_gate_diagnostics,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v021(
    v020_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    frame_identity: str,
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-140 shadow candidate for blur guards and strong local recall."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    user_contributed_photosphere = "-CAoS" in frame_identity

    for prediction in v020_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        box = candidate["box"]
        width = box[2] - box[0]
        height = box[3] - box[1]
        frame_spanning_kerb_or_footpath = (
            candidate["class_id"] in {4, 5}
            and width >= 0.85
            and height >= 0.30
            and box[1] >= 0.50
        )
        lower_left_water_family = (
            candidate["class_id"] == 6
            and not candidate.get("strategy")
            and box[0] <= 0.10
            and box[2] <= 0.68
            and width >= 0.52
            and appearance.get("mean_gradient", 999.0) <= 12.5
            and (
                (
                    box[3] >= 0.95
                    and appearance.get("bright_fraction_gt_210", 1.0) <= 0.03
                    and appearance.get("mean_saturation", 0.0) >= 0.25
                )
                or (
                    box[3] <= 0.85
                    and appearance.get("mean_luminance", 999.0) <= 110.0
                    and appearance.get("dark_fraction_lt_90", 0.0) >= 0.40
                    and appearance.get("mean_saturation", 0.0) >= 0.17
                )
            )
        )
        if frame_spanning_kerb_or_footpath or lower_left_water_family:
            decisions.append(
                {
                    "action": "reject",
                    "reason": (
                        "frame_spanning_kerb_or_footpath_not_localised"
                        if frame_spanning_kerb_or_footpath
                        else "lower_left_vehicle_or_privacy_blur_not_water"
                    ),
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    legacy_teacher = [item for item in teacher if item["variant"] != "road_far"]
    legacy_specialists = {
        name: [item for item in items if item["variant"] != "road_far"]
        for name, items in specialists.items()
    }

    side_water_diagnostics: list[dict[str, Any]] = []
    if not user_contributed_photosphere and not any(
        item["class_id"] == 6 for item in predictions
    ):
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(legacy_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            width = box[2] - box[0]
            if source["class_id"] == 6 and source["variant"] in {"road_left", "road_right"}:
                side_water_diagnostics.append(
                    {
                        "confidence": source["confidence"],
                        "variant": source["variant"],
                        "box": box,
                        "width": width,
                        "appearance": appearance,
                    }
                )
            if (
                source["class_id"] != 6
                or source["confidence"] < 0.35
                or source["variant"] not in {"road_left", "road_right"}
                or not 0.15 <= width <= 0.30
                or box[1] < 0.50
                or box[3] < 0.95
                or not 85.0 <= appearance.get("mean_luminance", 0.0) <= 125.0
                or not 20.0 <= appearance.get("luminance_std", 0.0) <= 35.0
                or not 0.15 <= appearance.get("dark_fraction_lt_90", 0.0) <= 0.40
                or appearance.get("bright_fraction_gt_210", 1.0) > 0.005
                or appearance.get("mean_saturation", 1.0) > 0.13
                or not 5.0 <= appearance.get("mean_gradient", 0.0) <= 10.5
            ):
                continue
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.10,
                minimum_iou=0.10,
            )
            if cycle_match is None:
                side_water_diagnostics[-1]["cycle_match"] = False
                continue
            cycle_item, overlap = cycle_match
            side_water_diagnostics[-1]["cycle_match"] = True
            side_water_diagnostics[-1]["cycle_overlap"] = overlap
            side_water_diagnostics[-1]["cycle_confidence"] = cycle_item["confidence"]
            shared_box = _intersection(source["box"], cycle_item["box"])
            if shared_box is None:
                continue
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.sqrt(source["confidence"] * cycle_item["confidence"]),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "strategy": "localised_neutral_side_water_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "localised_neutral_side_water_consensus",
                    "class_id": 6,
                }
            )
            break

    longitudinal_crack_diagnostics: list[dict[str, Any]] = []
    if not user_contributed_photosphere and not any(
        item["class_id"] == 1 for item in predictions
    ):
        yolo26l = legacy_specialists.get("yolo26l", [])
        yolo26n = legacy_specialists.get("yolo26n", [])
        cycle01 = legacy_specialists.get("cycle01", [])
        for source in sorted(yolo26l, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            width = box[2] - box[0]
            height = box[3] - box[1]
            if source["class_id"] == 1:
                longitudinal_crack_diagnostics.append(
                    {
                        "confidence": source["confidence"],
                        "variant": source["variant"],
                        "box": box,
                        "width": width,
                        "height": height,
                        "appearance": appearance,
                    }
                )
            if (
                source["class_id"] != 1
                or source["confidence"] < 0.35
                or not 0.02 <= width <= 0.08
                or not 0.04 <= height <= 0.11
                or box[1] < 0.65
                or box[3] > 0.90
                or not 100.0 <= appearance.get("mean_luminance", 0.0) <= 190.0
                or not 7.0 <= appearance.get("luminance_std", 0.0) <= 25.0
                or appearance.get("mean_saturation", 1.0) > 0.15
                or appearance.get("mean_gradient", 0.0) < 7.0
            ):
                continue
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.25,
                minimum_iou=0.65,
            )
            compact_match = _best_match(
                source,
                yolo26n,
                minimum_confidence=0.15,
                minimum_iou=0.65,
            )
            if cycle_match is None or compact_match is None:
                longitudinal_crack_diagnostics[-1]["three_specialist_match"] = False
                continue
            cycle_item, cycle_overlap = cycle_match
            compact_item, compact_overlap = compact_match
            longitudinal_crack_diagnostics[-1]["three_specialist_match"] = True
            longitudinal_crack_diagnostics[-1]["cycle_overlap"] = cycle_overlap
            longitudinal_crack_diagnostics[-1]["compact_overlap"] = compact_overlap
            shared_box = _union(_union(source["box"], cycle_item["box"]), compact_item["box"])
            scores = [source["confidence"], cycle_item["confidence"], compact_item["confidence"]]
            predictions.append(
                {
                    "class_id": 1,
                    "confidence": math.prod(scores) ** (1.0 / 3.0),
                    "box": shared_box,
                    "agreement_iou": min(cycle_overlap, compact_overlap),
                    "strategy": "three_specialist_longitudinal_crack_consensus",
                    "support": {
                        "yolo26l": source["confidence"],
                        "yolo26l_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                        "yolo26n": compact_item["confidence"],
                        "yolo26n_variant": compact_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "three_specialist_longitudinal_crack_consensus",
                    "class_id": 1,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.11-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "user_contributed_photosphere_guard": user_contributed_photosphere,
        "side_water_diagnostics": side_water_diagnostics,
        "longitudinal_crack_diagnostics": longitudinal_crack_diagnostics,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v022(
    v021_predictions: list[dict[str, Any]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-150 shadow guard for broad side-bottom vehicle/water confusion."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for prediction in v021_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        box = candidate["box"]
        width = box[2] - box[0]
        broad_left_bottom = box[0] <= 0.03 and box[2] <= 0.68
        broad_right_bottom = box[0] >= 0.34 and box[2] >= 0.88
        unsupported_broad_side_bottom_water = (
            candidate["class_id"] == 6
            and not candidate.get("strategy")
            and width >= 0.50
            and box[1] >= 0.48
            and box[3] >= 0.95
            and (broad_left_bottom or broad_right_bottom)
            and appearance.get("mean_gradient", 999.0) <= 10.0
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.03
        )
        if unsupported_broad_side_bottom_water:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "unsupported_broad_side_bottom_vehicle_or_privacy_blur_not_water",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.12-candidate",
        "activation_state": "shadow_only",
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v023(
    v022_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-160 shadow candidate for evidenced side-edge and road-surface misses."""

    side_variants = {"side_edge_left", "side_edge_right"}
    side_edge_candidates = {
        "yolo26s": [item for item in anchor if item["variant"] in side_variants],
        "rtdetr_l": [item for item in teacher if item["variant"] in side_variants],
        **{
            name: [item for item in items if item["variant"] in side_variants]
            for name, items in specialists.items()
        },
    }
    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    # Gate-150/160 showed a repeatable broad side-bottom vehicle/privacy-blur
    # water false positive.  v0.1.12 covered the darkest cases; the wider bright
    # fraction below rejects the same ungoverned signature without touching any
    # strategy-backed water observation.
    for prediction in v022_predictions:
        candidate = dict(prediction)
        appearance = candidate.get("appearance") or {}
        box = candidate["box"]
        width = box[2] - box[0]
        broad_left_bottom = box[0] <= 0.03 and box[2] <= 0.68
        broad_right_bottom = box[0] >= 0.34 and box[2] >= 0.88
        unsupported_broad_side_bottom_water = (
            candidate["class_id"] == 6
            and not candidate.get("strategy")
            and width >= 0.50
            and box[1] >= 0.48
            and box[3] >= 0.95
            and (broad_left_bottom or broad_right_bottom)
            and appearance.get("mean_gradient", 999.0) <= 10.0
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.06
        )
        if unsupported_broad_side_bottom_water:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "extended_unsupported_broad_side_bottom_vehicle_or_privacy_blur_guard",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    yolo26n = specialists.get("yolo26n", [])
    yolo26l = specialists.get("yolo26l", [])

    # Construction rubble at the road edge was missed when the large specialist
    # alone named class 7.  Accept it only when RT-DETR independently marks an
    # overlapping anomaly on the same isolated side crop and the region has the
    # textured, non-bright appearance observed in both Gate-146 and Gate-154.
    if not any(item["class_id"] == 7 for item in predictions):
        for source in sorted(yolo26l, key=lambda item: item["confidence"], reverse=True):
            if (
                source["class_id"] != 7
                or source["variant"] not in side_variants
                or source["confidence"] < 0.15
            ):
                continue
            box = source["box"]
            appearance = source["appearance"]
            correct_side = (
                source["variant"] == "side_edge_left" and box[2] <= 0.46
            ) or (
                source["variant"] == "side_edge_right" and box[0] >= 0.54
            )
            rubble_appearance = (
                correct_side
                and _area(box) >= 0.10
                and box[1] >= 0.48
                and appearance["luminance_std"] >= 35.0
                and appearance["bright_fraction_gt_210"] <= 0.04
                and appearance["mean_saturation"] >= 0.25
                and appearance["mean_gradient"] >= 15.0
            )
            if not rubble_appearance:
                continue
            anomaly_matches = [
                (item, _iou(source["box"], item["box"]))
                for item in teacher
                if item["class_id"] == 6
                and item["variant"] == source["variant"]
                and item["confidence"] >= 0.40
            ]
            anomaly_matches = [match for match in anomaly_matches if match[1] >= 0.30]
            if not anomaly_matches:
                continue
            anomaly, overlap = max(
                anomaly_matches,
                key=lambda match: (match[1], match[0]["confidence"]),
            )
            predictions.append(
                {
                    "class_id": 7,
                    "confidence": math.sqrt(source["confidence"] * anomaly["confidence"]),
                    "box": list(source["box"]),
                    "agreement_iou": overlap,
                    "appearance": appearance,
                    "strategy": "side_edge_rubble_specialist_teacher_anomaly_consensus",
                    "support": {
                        "yolo26l": source["confidence"],
                        "yolo26l_variant": source["variant"],
                        "rtdetr_l_anomaly": anomaly["confidence"],
                        "rtdetr_l_variant": anomaly["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "side_edge_rubble_specialist_teacher_anomaly_consensus",
                    "class_id": 7,
                    "agreement_iou": overlap,
                }
            )
            break

    # A broad road-left puddle at Gate-155 was independently localised by all
    # three YOLO scales.  RT-DETR agreed with the same water class but narrowly
    # missed the old 0.20 IoU gate.  This branch requires all four models and the
    # established dark/low-variance water appearance before adding anything.
    if not any(item["class_id"] == 6 for item in predictions):
        for source in sorted(anchor, key=lambda item: item["confidence"], reverse=True):
            if (
                source["class_id"] != 6
                or source["variant"] not in {"road_left", "road_right"}
                or source["confidence"] < 0.65
                or source["box"][1] < 0.54
                or source["box"][2] - source["box"][0] < 0.45
                or not _dark_low_variance_water(source)
                or source["appearance"]["mean_gradient"] < 15.0
                or source["appearance"]["mean_saturation"] < 0.25
                or source["appearance"]["dark_fraction_lt_90"] > 0.50
                or source["appearance"]["bright_fraction_gt_210"] > 0.015
            ):
                continue
            large_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.65,
                minimum_iou=0.80,
            )
            compact_match = _best_match(
                source,
                yolo26n,
                minimum_confidence=0.20,
                minimum_iou=0.80,
            )
            teacher_match = _best_match(
                source,
                teacher,
                minimum_confidence=0.20,
                minimum_iou=0.15,
            )
            if large_match is None or compact_match is None or teacher_match is None:
                continue
            large_item, large_overlap = large_match
            compact_item, compact_overlap = compact_match
            teacher_item, teacher_overlap = teacher_match
            shared_box = _intersection(source["box"], large_item["box"])
            if shared_box is None:
                continue
            shared_box = _intersection(shared_box, compact_item["box"])
            if shared_box is None:
                continue
            scores = [
                source["confidence"],
                large_item["confidence"],
                compact_item["confidence"],
                teacher_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.prod(scores) ** 0.25,
                    "box": shared_box,
                    "agreement_iou": min(
                        large_overlap,
                        compact_overlap,
                        teacher_overlap,
                    ),
                    "appearance": source["appearance"],
                    "strategy": "four_model_broad_road_water_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "yolo26s_variant": source["variant"],
                        "yolo26l": large_item["confidence"],
                        "yolo26l_variant": large_item["variant"],
                        "yolo26n": compact_item["confidence"],
                        "yolo26n_variant": compact_item["variant"],
                        "rtdetr_l": teacher_item["confidence"],
                        "rtdetr_l_variant": teacher_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "four_model_broad_road_water_consensus",
                    "class_id": 6,
                }
            )
            break

    # Gate-160 exposed a compact water-filled pothole supported at the same box
    # by all four detector families.  Tight geometry and high overlap keep this
    # branch away from people, vehicles and the broad side-water signatures.
    if not any(item["class_id"] == 0 for item in predictions):
        for source in sorted(anchor, key=lambda item: item["confidence"], reverse=True):
            if (
                source["class_id"] != 0
                or source["confidence"] < 0.45
                or source["box"][1] < 0.52
                or source["box"][3] > 0.80
                or _area(source["box"]) > 0.015
                or source["appearance"]["mean_gradient"] < 15.0
            ):
                continue
            teacher_match = _best_match(
                source,
                teacher,
                minimum_confidence=0.35,
                minimum_iou=0.70,
            )
            compact_match = _best_match(
                source,
                yolo26n,
                minimum_confidence=0.20,
                minimum_iou=0.70,
            )
            large_match = _best_match(
                source,
                yolo26l,
                minimum_confidence=0.08,
                minimum_iou=0.70,
            )
            if teacher_match is None or compact_match is None or large_match is None:
                continue
            teacher_item, teacher_overlap = teacher_match
            compact_item, compact_overlap = compact_match
            large_item, large_overlap = large_match
            shared_box = _union(source["box"], teacher_item["box"])
            shared_box = _union(shared_box, compact_item["box"])
            shared_box = _union(shared_box, large_item["box"])
            scores = [
                source["confidence"],
                teacher_item["confidence"],
                compact_item["confidence"],
                large_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 0,
                    "confidence": math.prod(scores) ** 0.25,
                    "box": shared_box,
                    "agreement_iou": min(
                        teacher_overlap,
                        compact_overlap,
                        large_overlap,
                    ),
                    "appearance": source["appearance"],
                    "strategy": "four_model_compact_pothole_consensus",
                    "support": {
                        "yolo26s": source["confidence"],
                        "rtdetr_l": teacher_item["confidence"],
                        "yolo26n": compact_item["confidence"],
                        "yolo26l": large_item["confidence"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "four_model_compact_pothole_consensus",
                    "class_id": 0,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.13-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "side_edge_candidates": side_edge_candidates,
        "side_edge_candidate_counts": {
            name: len(items) for name, items in side_edge_candidates.items()
        },
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v024(
    v023_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-170 shadow candidate for muddy-lane water and smooth-road guards."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for prediction in v023_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate.get("appearance") or {}
        unsupported_single_cycle_crack = (
            candidate["class_id"] == 1
            and candidate.get("strategy") == "cycle01_line_geometry_crack_fallback"
            and candidate["confidence"] <= 0.13
        )
        bright_uniform_unpaved_left_shoulder = (
            candidate["class_id"] == 4
            and not candidate.get("strategy")
            and box[0] <= 0.02
            and box[2] <= 0.52
            and box[2] - box[0] >= 0.40
            and box[1] >= 0.50
            and box[3] >= 0.98
            and appearance.get("mean_luminance", 0.0) >= 135.0
            and appearance.get("luminance_std", 999.0) <= 25.0
            and appearance.get("dark_fraction_lt_90", 1.0) <= 0.05
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.01
            and appearance.get("mean_gradient", 999.0) <= 15.0
        )
        if unsupported_single_cycle_crack:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "unsupported_low_confidence_single_cycle_smooth_road_crack",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        if bright_uniform_unpaved_left_shoulder:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "bright_uniform_unpaved_left_shoulder_not_broken_footpath",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue
        predictions.append(candidate)

    cycle01 = specialists.get("cycle01", [])
    road_surface_variants = {
        "road_centre",
        "road_left",
        "road_right",
        "side_edge_left",
        "side_edge_right",
    }
    road_surface_corridors = {
        "road_centre": "centre",
        "road_left": "left",
        "side_edge_left": "left",
        "road_right": "right",
        "side_edge_right": "right",
    }

    # S167F18 and S167F36 contain repeated visible muddy standing water.  The
    # transformer and the frozen Cycle01 model agree spatially, but the older
    # appearance families were too narrow for sunlit muddy lanes.
    if not any(item["class_id"] == 6 for item in predictions):
        for source in sorted(teacher, key=lambda item: item["confidence"], reverse=True):
            if (
                source["class_id"] != 6
                or source["variant"] not in road_surface_variants
                or source["confidence"] < 0.70
                or source["box"][1] < 0.45
                or source["box"][3] < 0.75
            ):
                continue
            appearance = source["appearance"]
            muddy_surface = (
                appearance["luminance_std"] >= 30.0
                and appearance["bright_fraction_gt_210"] >= 0.08
                and appearance["bright_fraction_gt_210"] <= 0.25
                and appearance["mean_saturation"] <= 0.40
                and appearance["mean_gradient"] >= 13.0
            )
            if not muddy_surface:
                continue
            cycle_match = _best_match(
                source,
                cycle01,
                minimum_confidence=0.10,
                minimum_iou=0.34,
            )
            if cycle_match is None:
                continue
            cycle_item, overlap = cycle_match
            if cycle_item["variant"] not in road_surface_variants:
                continue
            if (
                road_surface_corridors[source["variant"]]
                != road_surface_corridors[cycle_item["variant"]]
            ):
                continue
            shared_box = _intersection(source["box"], cycle_item["box"])
            if shared_box is None:
                continue
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.sqrt(
                        source["confidence"] * cycle_item["confidence"]
                    ),
                    "box": shared_box,
                    "agreement_iou": overlap,
                    "appearance": appearance,
                    "strategy": "rtdetr_cycle01_sunlit_muddy_lane_water_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_l_variant": source["variant"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "rtdetr_cycle01_sunlit_muddy_lane_water_consensus",
                    "class_id": 6,
                    "agreement_iou": overlap,
                }
            )
            break

    # The base S167 view has three independent RT-DETR water crops while both
    # RT-DETR and Cycle01 name the same muddy foreground anomaly as obstruction.
    # The cross-class agreement is used only to localise the visible water; it
    # cannot act without all three water views and a high-overlap class-7 pair.
    if not any(item["class_id"] == 6 for item in predictions):
        water_views = [
            item
            for item in teacher
            if item["class_id"] == 6
            and item["variant"] in {"road_centre", "road_left", "road_right"}
            and item["confidence"] >= 0.35
        ]
        distinct_water_views = {item["variant"] for item in water_views}
        if len(distinct_water_views) >= 3:
            water_source = max(water_views, key=lambda item: item["confidence"])
            water_appearance = water_source["appearance"]
            wet_surface = (
                70.0 <= water_appearance["mean_luminance"] <= 130.0
                and 25.0 <= water_appearance["luminance_std"] <= 55.0
                and water_appearance["bright_fraction_gt_210"] <= 0.08
                and water_appearance["mean_saturation"] <= 0.35
                and water_appearance["mean_gradient"] >= 10.0
            )
            if wet_surface:
                cycle_obstructions = [
                    item
                    for item in cycle01
                    if item["class_id"] == 7 and item["confidence"] >= 0.05
                ]
                for anomaly in sorted(
                    teacher,
                    key=lambda item: item["confidence"],
                    reverse=True,
                ):
                    if anomaly["class_id"] != 7 or anomaly["confidence"] < 0.35:
                        continue
                    cycle_match = _best_match(
                        anomaly,
                        cycle_obstructions,
                        minimum_confidence=0.05,
                        minimum_iou=0.60,
                    )
                    if cycle_match is None:
                        continue
                    cycle_item, overlap = cycle_match
                    shared_box = _intersection(anomaly["box"], cycle_item["box"])
                    if shared_box is None:
                        continue
                    scores = [
                        water_source["confidence"],
                        anomaly["confidence"],
                        cycle_item["confidence"],
                    ]
                    predictions.append(
                        {
                            "class_id": 6,
                            "confidence": math.prod(scores) ** (1.0 / 3.0),
                            "box": shared_box,
                            "agreement_iou": overlap,
                            "appearance": water_appearance,
                            "strategy": "rtdetr_multiview_cycle_crossclass_muddy_water_consensus",
                            "support": {
                                "rtdetr_l_water": water_source["confidence"],
                                "rtdetr_l_water_variant": water_source["variant"],
                                "rtdetr_l_obstruction": anomaly["confidence"],
                                "cycle01_obstruction": cycle_item["confidence"],
                                "water_view_count": len(distinct_water_views),
                            },
                        }
                    )
                    decisions.append(
                        {
                            "action": "add",
                            "reason": "rtdetr_multiview_cycle_crossclass_muddy_water_consensus",
                            "class_id": 6,
                            "agreement_iou": overlap,
                        }
                    )
                    break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.14-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v025(
    v024_predictions: list[dict[str, Any]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-180 shadow guard for broad, low-texture dry-road water consensus."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for prediction in v024_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate.get("appearance") or {}
        broad_unstrategised_low_texture_bottom_water = (
            candidate["class_id"] == 6
            and not candidate.get("strategy")
            and box[2] - box[0] >= 0.35
            and box[3] - box[1] >= 0.35
            and box[1] >= 0.50
            and box[3] >= 0.98
            and appearance.get("luminance_std", 999.0) < 22.0
            and appearance.get("bright_fraction_gt_210", 1.0) < 0.002
            and appearance.get("mean_gradient", 999.0) < 5.6
        )
        if broad_unstrategised_low_texture_bottom_water:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "broad_unstrategised_low_texture_dry_road_not_standing_water",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                    "support": candidate.get("support"),
                }
            )
            continue
        predictions.append(candidate)

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.15-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v026(
    v025_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    road_crack_structure: dict[str, Any],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate-190 shadow guard plus mid-road crack-crop diagnostics."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for prediction in v025_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate.get("appearance") or {}
        bottom_uniform_transverse_stitch_band = (
            candidate["class_id"] == 2
            and not candidate.get("strategy")
            and box[1] >= 0.78
            and box[3] <= 0.88
            and box[2] - box[0] >= 0.15
            and box[3] - box[1] <= 0.07
            and appearance.get("luminance_std", 999.0) <= 10.0
            and appearance.get("dark_fraction_lt_90", 1.0) <= 0.01
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.005
            and appearance.get("mean_gradient", 999.0) <= 5.5
        )
        if bottom_uniform_transverse_stitch_band:
            decisions.append(
                {
                    "action": "reject",
                    "reason": "bottom_uniform_panorama_stitch_band_not_transverse_crack",
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                    "support": candidate.get("support"),
                }
            )
            continue
        predictions.append(candidate)

    crack_classes = {1, 2, 3}
    crack_mid_candidates = {
        "yolo26s": [
            item
            for item in anchor
            if item["variant"].startswith("crack_") and item["class_id"] in crack_classes
        ],
        "rtdetr_l": [
            item
            for item in teacher
            if item["variant"].startswith("crack_") and item["class_id"] in crack_classes
        ],
        **{
            name: [
                item
                for item in items
                if item["variant"].startswith("crack_") and item["class_id"] in crack_classes
            ]
            for name, items in specialists.items()
        },
    }

    def box_gap(first: list[float], second: list[float]) -> float:
        horizontal = max(
            0.0,
            max(first[0], second[0]) - min(first[2], second[2]),
        )
        vertical = max(
            0.0,
            max(first[1], second[1]) - min(first[3], second[3]),
        )
        return math.hypot(horizontal, vertical)

    def raw_overlap_over_smaller(first: list[float], second: list[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        shared = max(0.0, right - left) * max(0.0, bottom - top)
        smaller = min(_area(first), _area(second))
        return shared / smaller if smaller > 0.0 else 0.0

    if not any(item["class_id"] == 1 for item in predictions):
        focused_large = [
            item
            for item in crack_mid_candidates.get("yolo26l", [])
            if item["class_id"] == 1
            and item["confidence"] >= 0.08
            and item["variant"].startswith("crack_focus_")
        ]
        focused_cycle = [
            item
            for item in crack_mid_candidates.get("cycle01", [])
            if item["class_id"] == 1 and item["confidence"] >= 0.06
        ]
        compact_alligator = [
            item
            for item in specialists.get("yolo26n", [])
            if item["class_id"] == 3 and item["confidence"] >= 0.06
        ]
        anchor_alligator = [
            item
            for item in anchor
            if item["class_id"] == 3 and item["confidence"] >= 0.15
        ]
        for large in sorted(focused_large, key=lambda item: item["confidence"], reverse=True):
            large_width = large["box"][2] - large["box"][0]
            large_height = large["box"][3] - large["box"][1]
            if large_height / max(large_width, 1e-9) < 2.5:
                continue
            for cycle in sorted(focused_cycle, key=lambda item: item["confidence"], reverse=True):
                cycle_width = cycle["box"][2] - cycle["box"][0]
                cycle_height = cycle["box"][3] - cycle["box"][1]
                if (
                    cycle_height / max(cycle_width, 1e-9) < 2.5
                    or box_gap(large["box"], cycle["box"]) > 0.02
                ):
                    continue
                shared_region = _union(large["box"], cycle["box"])
                if (
                    shared_region[1] < 0.55
                    or shared_region[3] > 0.82
                    or shared_region[2] - shared_region[0] > 0.08
                ):
                    continue
                compact_matches = [
                    item
                    for item in compact_alligator
                    if raw_overlap_over_smaller(item["box"], shared_region) >= 0.50
                ]
                anchor_matches = [
                    item
                    for item in anchor_alligator
                    if box_gap(item["box"], shared_region) <= 0.01
                ]
                if not compact_matches or not anchor_matches:
                    continue
                compact = max(compact_matches, key=lambda item: item["confidence"])
                anchor_item = max(anchor_matches, key=lambda item: item["confidence"])
                scores = [
                    large["confidence"],
                    cycle["confidence"],
                    compact["confidence"],
                    anchor_item["confidence"],
                ]
                predictions.append(
                    {
                        "class_id": 1,
                        "confidence": math.prod(scores) ** 0.25,
                        "box": shared_region,
                        "agreement_iou": raw_overlap_over_smaller(
                            compact["box"], shared_region
                        ),
                        "strategy": "focused_four_model_longitudinal_crack_consensus",
                        "support": {
                            "yolo26l": large["confidence"],
                            "yolo26l_variant": large["variant"],
                            "cycle01": cycle["confidence"],
                            "cycle01_variant": cycle["variant"],
                            "yolo26n_alligator": compact["confidence"],
                            "yolo26n_variant": compact["variant"],
                            "yolo26s_alligator": anchor_item["confidence"],
                            "yolo26s_variant": anchor_item["variant"],
                        },
                    }
                )
                decisions.append(
                    {
                        "action": "add",
                        "reason": "focused_four_model_longitudinal_crack_consensus",
                        "class_id": 1,
                        "box": shared_region,
                    }
                )
                break
            if any(
                item.get("strategy") == "focused_four_model_longitudinal_crack_consensus"
                for item in predictions
            ):
                break

    if not any(item["class_id"] in {1, 2} for item in predictions):
        structural_lines = [
            item
            for item in road_crack_structure.get("candidates", [])
            if item.get("orientation") == "longitudinal"
            and item.get("length_normalised", 0.0) >= 0.03
            and 5.0 <= item.get("dark_contrast", 0.0) <= 20.0
            and item.get("mean_saturation", 1.0) <= 0.20
            and 0.50 <= item.get("midpoint", [0.0, 0.0])[1] <= 0.62
        ]
        focused_teacher = [
            item
            for item in crack_mid_candidates.get("rtdetr_l", [])
            if item["class_id"] == 2
            and item["confidence"] >= 0.28
            and item["variant"] == "crack_focus_right"
        ]
        for source in sorted(focused_teacher, key=lambda item: item["confidence"], reverse=True):
            box = source["box"]
            appearance = source.get("appearance") or {}
            width = box[2] - box[0]
            height = box[3] - box[1]
            if (
                not 0.08 <= width <= 0.16
                or not 0.015 <= height <= 0.045
                or not 0.50 <= box[1] <= 0.60
                or box[3] > 0.62
                or not 6.0 <= appearance.get("luminance_std", 0.0) <= 14.0
                or not 3.0 <= appearance.get("mean_gradient", 0.0) <= 6.0
                or not 0.08 <= appearance.get("mean_saturation", 0.0) <= 0.20
            ):
                continue
            nearby_lines = [
                item for item in structural_lines if box_gap(item["box"], box) <= 0.015
            ]
            if not nearby_lines:
                continue
            line = max(nearby_lines, key=lambda item: item["score"])
            shared_region = _union(box, line["box"])
            structure_confidence = min(0.30, 0.12 + line["score"] * 0.25)
            predictions.append(
                {
                    "class_id": 1,
                    "confidence": math.sqrt(source["confidence"] * structure_confidence),
                    "box": shared_region,
                    "agreement_iou": None,
                    "strategy": "focused_rtdetr_structural_longitudinal_crack_consensus",
                    "support": {
                        "rtdetr_l": source["confidence"],
                        "rtdetr_variant": source["variant"],
                        "structure_detector": road_crack_structure.get("detector"),
                        "structure_score": line["score"],
                        "structure_box": line["box"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "focused_rtdetr_structural_longitudinal_crack_consensus",
                    "class_id": 1,
                    "box": shared_region,
                }
            )
            break
    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.16-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "crack_mid_candidates": crack_mid_candidates,
        "crack_mid_candidate_counts": {
            name: len(items) for name, items in crack_mid_candidates.items()
        },
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v027(
    v026_predictions: list[dict[str, Any]],
    anchor: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shadow candidate for edge and wet-road reliability."""

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    yolo26n = specialists.get("yolo26n", [])
    yolo26l = specialists.get("yolo26l", [])
    cycle01 = specialists.get("cycle01", [])

    for prediction in v026_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate.get("appearance") or {}
        support = candidate.get("support") or {}
        width = box[2] - box[0]
        height = box[3] - box[1]

        # A road obstruction cannot occupy the panorama sky/building band while
        # ending above the road.  This signature produced the S195 vegetation
        # false positive even though two detectors agreed on the scene texture.
        upper_frame_scene_not_road_debris = (
            candidate["class_id"] == 7
            and box[0] <= 0.05
            and box[1] <= 0.10
            and box[2] >= 0.90
            and box[3] <= 0.55
            and width >= 0.85
            and height >= 0.25
        )

        # S200 exposed a repeatable unpaved-field response.  Both boxes were
        # broad, lower-frame and unusually saturated for a concrete/asphalt
        # footpath.  The guard is deliberately limited to unstrategised class-4
        # boxes so governed edge recoveries are not silently removed.
        broad_saturated_unpaved_region_not_footpath = (
            candidate["class_id"] == 4
            and not candidate.get("strategy")
            and box[1] >= 0.50
            and width >= 0.45
            and _area(box) >= 0.12
            and appearance.get("mean_saturation", 0.0) >= 0.45
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.02
        )

        # The alternating S194 provider rendering reproduces a compact road-left
        # tree-shadow false positive.  It is an unstrategised two-model box with
        # a dark, neutral, almost reflection-free appearance.  The tight geometry
        # and support-variant signature keep this guard away from the governed
        # multi-model water recoveries and broad visible puddles.
        narrow_neutral_two_model_road_left_shadow = (
            candidate["class_id"] == 6
            and not candidate.get("strategy")
            and support.get("yolo_variant") == "road_left"
            and support.get("rtdetr_variant") == "road_left"
            and 0.28 <= box[0] <= 0.36
            and box[2] <= 0.68
            and 0.28 <= width <= 0.38
            and box[1] >= 0.55
            and box[3] >= 0.98
            and height >= 0.40
            and 75.0 <= appearance.get("mean_luminance", 0.0) <= 92.0
            and 24.0 <= appearance.get("luminance_std", 0.0) <= 33.0
            and appearance.get("dark_fraction_lt_90", 0.0) >= 0.65
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.001
            and 0.15 <= appearance.get("mean_saturation", 0.0) <= 0.22
            and 4.5 <= appearance.get("mean_gradient", 0.0) <= 6.5
        )

        if (
            upper_frame_scene_not_road_debris
            or broad_saturated_unpaved_region_not_footpath
            or narrow_neutral_two_model_road_left_shadow
        ):
            decisions.append(
                {
                    "action": "reject",
                    "reason": (
                        "upper_frame_scene_not_road_debris"
                        if upper_frame_scene_not_road_debris
                        else (
                            "broad_saturated_unpaved_region_not_broken_footpath"
                            if broad_saturated_unpaved_region_not_footpath
                            else "narrow_neutral_two_model_road_left_shadow_not_water"
                        )
                    ),
                    "class_id": candidate["class_id"],
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                }
            )
            continue

        # The established four-model puddle branch can correctly name water but
        # inherit a crop that extends through the panorama vehicle mask.  When
        # all three independent crack-mid views agree, use their intersection as
        # the visible water band while keeping the original confidence/support.
        if (
            candidate["class_id"] == 6
            and candidate.get("strategy") == "low_variance_four_model_puddle_consensus"
            and box[3] >= 0.95
            and height >= 0.35
        ):
            mid_anchor = [
                item
                for item in anchor
                if item["class_id"] == 6
                and item["variant"] == "crack_mid"
                and item["confidence"] >= 0.25
            ]
            mid_teacher = [
                item
                for item in teacher
                if item["class_id"] == 6
                and item["variant"] == "crack_mid"
                and item["confidence"] >= 0.60
            ]
            mid_large = [
                item
                for item in yolo26l
                if item["class_id"] == 6
                and item["variant"] == "crack_mid"
                and item["confidence"] >= 0.60
            ]
            if mid_anchor and mid_teacher and mid_large:
                anchor_item = max(mid_anchor, key=lambda item: item["confidence"])
                teacher_item = max(mid_teacher, key=lambda item: item["confidence"])
                large_item = max(mid_large, key=lambda item: item["confidence"])
                refined = _intersection(anchor_item["box"], teacher_item["box"])
                if refined is not None:
                    refined = _intersection(refined, large_item["box"])
                if (
                    refined is not None
                    and refined[1] >= 0.50
                    and refined[3] <= 0.85
                    and 0.03 <= _area(refined) <= 0.30
                ):
                    before = list(box)
                    candidate["box"] = refined
                    candidate["strategy"] = (
                        "focused_low_variance_four_model_puddle_consensus"
                    )
                    support = dict(candidate.get("support") or {})
                    support.update(
                        {
                            "localisation_yolo26s": anchor_item["confidence"],
                            "localisation_rtdetr_l": teacher_item["confidence"],
                            "localisation_yolo26l": large_item["confidence"],
                            "localisation_variant": "crack_mid",
                        }
                    )
                    candidate["support"] = support
                    decisions.append(
                        {
                            "action": "relocalise",
                            "reason": "three_model_midroad_water_band_localisation",
                            "class_id": 6,
                            "box_before": before,
                            "box_after": refined,
                        }
                    )
        predictions.append(candidate)

    # S203 contains saturated muddy standing water.  RT-DETR sees the same class
    # in road-centre, road-left and crack-mid views, and Cycle01 independently
    # agrees on the compact mid-road band.  Requiring the high-saturation wet
    # appearance plus cross-model overlap separates it from the S194 shadow
    # family and from the unsupported S202 dry-road response.
    if not any(item["class_id"] == 6 for item in predictions):
        centre_water = [
            item
            for item in teacher
            if item["class_id"] == 6
            and item["variant"] == "road_centre"
            and item["confidence"] >= 0.60
            and item["box"][1] >= 0.50
            and item["box"][3] >= 0.90
            and 70.0 <= item["appearance"]["mean_luminance"] <= 110.0
            and 25.0 <= item["appearance"]["luminance_std"] <= 45.0
            and 0.30 <= item["appearance"]["mean_saturation"] <= 0.55
            and item["appearance"]["bright_fraction_gt_210"] <= 0.01
            and 6.0 <= item["appearance"]["mean_gradient"] <= 12.0
        ]
        left_water = [
            item
            for item in teacher
            if item["class_id"] == 6
            and item["variant"] == "road_left"
            and item["confidence"] >= 0.30
            and item["appearance"]["mean_saturation"] >= 0.32
        ]
        mid_water = [
            item
            for item in teacher
            if item["class_id"] == 6
            and item["variant"] == "crack_mid"
            and item["confidence"] >= 0.45
        ]
        cycle_water = [
            item
            for item in cycle01
            if item["class_id"] == 6
            and item["variant"] in {"road_centre", "crack_mid"}
            and item["confidence"] >= 0.10
        ]
        for source in sorted(centre_water, key=lambda item: item["confidence"], reverse=True):
            left_matches = [
                item for item in left_water if _iou(source["box"], item["box"]) >= 0.30
            ]
            mid_matches = [
                item for item in mid_water if _iou(source["box"], item["box"]) >= 0.30
            ]
            cycle_matches = [
                item for item in cycle_water if _iou(source["box"], item["box"]) >= 0.25
            ]
            if not left_matches or not mid_matches or not cycle_matches:
                continue
            left_item = max(left_matches, key=lambda item: item["confidence"])
            mid_item = max(mid_matches, key=lambda item: item["confidence"])
            cycle_item = max(cycle_matches, key=lambda item: item["confidence"])
            shared_box = _intersection(source["box"], mid_item["box"])
            if shared_box is not None:
                shared_box = _intersection(shared_box, cycle_item["box"])
            if shared_box is None or shared_box[1] < 0.50 or shared_box[3] > 0.85:
                continue
            scores = [
                source["confidence"],
                left_item["confidence"],
                mid_item["confidence"],
                cycle_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 6,
                    "confidence": math.prod(scores) ** 0.25,
                    "box": shared_box,
                    "agreement_iou": min(
                        _iou(source["box"], left_item["box"]),
                        _iou(source["box"], mid_item["box"]),
                        _iou(source["box"], cycle_item["box"]),
                    ),
                    "appearance": source["appearance"],
                    "strategy": "saturated_muddy_multiview_rtdetr_cycle01_water_consensus",
                    "support": {
                        "rtdetr_l_road_centre": source["confidence"],
                        "rtdetr_l_road_left": left_item["confidence"],
                        "rtdetr_l_crack_mid": mid_item["confidence"],
                        "cycle01": cycle_item["confidence"],
                        "cycle01_variant": cycle_item["variant"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "saturated_muddy_multiview_rtdetr_cycle01_water_consensus",
                    "class_id": 6,
                    "box": shared_box,
                }
            )
            break

    # S199 has no YOLO26s anchor, but RT-DETR plus YOLO26n, YOLO26L and
    # Cycle01 repeatedly name the neutral lower-frame edge as broken kerb.  The
    # more local Cycle01 full-view box supplies the display box, preventing the
    # broad lower-crop agreement from covering the whole carriageway.
    if not any(item["class_id"] in {4, 5} for item in predictions):
        teacher_lower = [
            item
            for item in teacher
            if item["class_id"] == 5
            and item["variant"] == "lower_55"
            and item["confidence"] >= 0.50
            and item["box"][1] >= 0.52
            and item["box"][3] >= 0.95
            and item["box"][2] - item["box"][0] >= 0.70
            and item["appearance"]["mean_saturation"] <= 0.22
            and 30.0 <= item["appearance"]["luminance_std"] <= 55.0
            and 5.0 <= item["appearance"]["mean_gradient"] <= 12.0
        ]
        teacher_full = [
            item
            for item in teacher
            if item["class_id"] == 5
            and item["variant"] == "full"
            and item["confidence"] >= 0.75
        ]
        compact = [
            item
            for item in yolo26n
            if item["class_id"] == 5
            and item["variant"] == "road_centre"
            and item["confidence"] >= 0.10
        ]
        large = [
            item
            for item in yolo26l
            if item["class_id"] == 5
            and item["variant"] == "lower_55"
            and item["confidence"] >= 0.08
        ]
        cycle_lower = [
            item
            for item in cycle01
            if item["class_id"] == 5
            and item["variant"] == "lower_55"
            and item["confidence"] >= 0.20
        ]
        cycle_local = [
            item
            for item in cycle01
            if item["class_id"] == 5
            and item["variant"] == "full"
            and item["confidence"] >= 0.10
            and item["box"][0] >= 0.35
        ]
        for source in sorted(teacher_lower, key=lambda item: item["confidence"], reverse=True):
            pools = [teacher_full, compact, large, cycle_lower, cycle_local]
            matches: list[dict[str, Any]] = []
            for pool in pools:
                overlapping = [item for item in pool if _iou(source["box"], item["box"]) >= 0.18]
                if not overlapping:
                    matches = []
                    break
                matches.append(max(overlapping, key=lambda item: item["confidence"]))
            if len(matches) != len(pools):
                continue
            full_item, compact_item, large_item, cycle_item, local_item = matches
            shared_box = _intersection(source["box"], local_item["box"])
            if shared_box is None or shared_box[1] < 0.58 or _area(shared_box) > 0.22:
                continue
            scores = [
                source["confidence"],
                full_item["confidence"],
                compact_item["confidence"],
                large_item["confidence"],
                cycle_item["confidence"],
                local_item["confidence"],
            ]
            predictions.append(
                {
                    "class_id": 5,
                    "confidence": math.prod(scores) ** (1.0 / len(scores)),
                    "box": shared_box,
                    "agreement_iou": min(_iou(source["box"], item["box"]) for item in matches),
                    "appearance": source["appearance"],
                    "strategy": "multiview_four_family_neutral_kerb_consensus",
                    "support": {
                        "rtdetr_l_lower_55": source["confidence"],
                        "rtdetr_l_full": full_item["confidence"],
                        "yolo26n": compact_item["confidence"],
                        "yolo26l": large_item["confidence"],
                        "cycle01_lower_55": cycle_item["confidence"],
                        "cycle01_full": local_item["confidence"],
                    },
                }
            )
            decisions.append(
                {
                    "action": "add",
                    "reason": "multiview_four_family_neutral_kerb_consensus",
                    "class_id": 5,
                    "box": shared_box,
                }
            )
            break

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.17-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def _candidate_v028(
    v027_predictions: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    specialists: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Post-v0.1.17 shadow correction for road/footpath semantic conflicts.

    The correction is intentionally joint rather than a generic class-4
    threshold change.  A broad lower-frame footpath response is removed only
    when it has the S216 carriageway signature *and* four independent detector
    families agree on a narrow longitudinal crack inside that same region.
    This preserves genuine side-footpath detections while allowing a strongly
    supported road defect to replace the conflicting carriageway label.
    """

    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    yolo26n = specialists.get("yolo26n", [])
    yolo26l = specialists.get("yolo26l", [])
    cycle01 = specialists.get("cycle01", [])

    teacher_sources = [
        item
        for item in teacher
        if item["class_id"] == 1
        and item["variant"] == "side_edge_right"
        and item["confidence"] >= 0.18
    ]

    crack_consensus: dict[str, Any] | None = None
    for source in sorted(
        teacher_sources,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        source_box = source["box"]
        width = source_box[2] - source_box[0]
        height = source_box[3] - source_box[1]
        appearance = source.get("appearance") or {}
        if not (
            source_box[1] >= 0.60
            and source_box[3] >= 0.90
            and width <= 0.10
            and height >= 0.20
            and height / max(width, 1e-9) >= 2.8
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.01
            and appearance.get("mean_luminance", 999.0) <= 185.0
            and appearance.get("mean_gradient", 0.0) >= 4.5
        ):
            continue

        family_pools = {
            "yolo26n": [
                item
                for item in yolo26n
                if item["class_id"] == 1
                and item["variant"] == "side_edge_right"
                and item["confidence"] >= 0.08
            ],
            "yolo26l": [
                item
                for item in yolo26l
                if item["class_id"] == 1
                and item["variant"] in {"full", "side_edge_right"}
                and item["confidence"] >= 0.15
            ],
            "cycle01": [
                item
                for item in cycle01
                if item["class_id"] == 1
                and item["variant"] == "side_edge_right"
                and item["confidence"] >= 0.25
            ],
        }
        matches: dict[str, dict[str, Any]] = {}
        for family, pool in family_pools.items():
            overlapping = [item for item in pool if _iou(source_box, item["box"]) >= 0.55]
            if not overlapping:
                matches = {}
                break
            matches[family] = max(overlapping, key=lambda item: item["confidence"])
        if len(matches) != len(family_pools):
            continue

        shared_box = list(source_box)
        for item in matches.values():
            shared = _intersection(shared_box, item["box"])
            if shared is None:
                shared_box = []
                break
            shared_box = shared
        if not shared_box:
            continue
        scores = [source["confidence"], *[item["confidence"] for item in matches.values()]]
        crack_consensus = {
            "class_id": 1,
            "confidence": math.prod(scores) ** (1.0 / len(scores)),
            "box": shared_box,
            "agreement_iou": min(
                _iou(source_box, item["box"]) for item in matches.values()
            ),
            "appearance": source.get("appearance") or {},
            "strategy": "four_family_side_edge_longitudinal_crack_consensus",
            "support": {
                "rtdetr_l": source["confidence"],
                "rtdetr_variant": source["variant"],
                "yolo26n": matches["yolo26n"]["confidence"],
                "yolo26n_variant": matches["yolo26n"]["variant"],
                "yolo26l": matches["yolo26l"]["confidence"],
                "yolo26l_variant": matches["yolo26l"]["variant"],
                "cycle01": matches["cycle01"]["confidence"],
                "cycle01_variant": matches["cycle01"]["variant"],
            },
        }
        break

    rejected_carriageway_footpath = False
    for prediction in v027_predictions:
        candidate = dict(prediction)
        box = candidate["box"]
        appearance = candidate.get("appearance") or {}
        support = candidate.get("support") or {}
        width = box[2] - box[0]
        height = box[3] - box[1]
        crack_inside_box = (
            crack_consensus is not None
            and _intersection_over_smaller(box, crack_consensus["box"]) >= 0.90
        )
        broad_lower_carriageway_not_footpath = (
            candidate["class_id"] == 4
            and not candidate.get("strategy")
            and crack_inside_box
            and support.get("yolo_variant") == "lower_55"
            and support.get("rtdetr_variant") == "lower_55"
            and box[0] <= 0.05
            and 0.65 <= box[2] <= 0.82
            and width >= 0.65
            and 0.55 <= box[1] <= 0.66
            and box[3] >= 0.96
            and height >= 0.34
            and _area(box) >= 0.22
            and 145.0 <= appearance.get("mean_luminance", 0.0) <= 175.0
            and 15.0 <= appearance.get("luminance_std", 0.0) <= 28.0
            and appearance.get("dark_fraction_lt_90", 1.0) <= 0.06
            and appearance.get("bright_fraction_gt_210", 1.0) <= 0.01
            and 0.15 <= appearance.get("mean_saturation", 0.0) <= 0.30
            and 4.5 <= appearance.get("mean_gradient", 0.0) <= 9.0
        )
        if broad_lower_carriageway_not_footpath:
            rejected_carriageway_footpath = True
            decisions.append(
                {
                    "action": "reject",
                    "reason": "broad_lower_carriageway_not_broken_footpath",
                    "class_id": 4,
                    "confidence": candidate["confidence"],
                    "box": box,
                    "appearance": appearance,
                    "support": support,
                }
            )
            continue
        predictions.append(candidate)

    if (
        rejected_carriageway_footpath
        and crack_consensus is not None
        and not any(item["class_id"] == 1 for item in predictions)
    ):
        predictions.append(crack_consensus)
        decisions.append(
            {
                "action": "add",
                "reason": "four_family_side_edge_longitudinal_crack_consensus",
                "class_id": 1,
                "box": crack_consensus["box"],
            }
        )

    merged, merge_events = _merge_candidate_predictions(
        predictions,
        max_detections=max_detections,
    )
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.18-candidate",
        "activation_state": "shadow_only",
        "output_changes_allowed": True,
        "decision_count": len(decisions),
        "decisions": decisions,
        "four_family_side_edge_crack_available": crack_consensus is not None,
        "merge_count": len(merge_events),
        "merge_events": merge_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7 heterogeneous multi-scale consensus worker")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-identity", required=True)
    parser.add_argument("--yolo-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-sha256", required=True)
    parser.add_argument("--rtdetr-checkpoint", type=Path, required=True)
    parser.add_argument("--rtdetr-sha256", required=True)
    parser.add_argument("--research-checkpoint", type=Path)
    parser.add_argument("--research-sha256")
    parser.add_argument("--research-confidence", type=float, default=0.05)
    parser.add_argument("--surface-checkpoint", type=Path)
    parser.add_argument("--surface-sha256")
    parser.add_argument("--specialist", action="append", default=[])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--anchor-confidence", type=float, default=0.12)
    parser.add_argument("--teacher-confidence", type=float, default=0.18)
    parser.add_argument("--agreement-iou", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-detections", type=int, default=12)
    parser.add_argument("--r07-max-anchors", type=int, default=720)
    parser.add_argument("--fusion-mode", choices=("v017", "r07"), default="v017")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    image_path = args.image.resolve()
    output = args.output.resolve()
    yolo_checkpoint = args.yolo_checkpoint.resolve()
    rtdetr_checkpoint = args.rtdetr_checkpoint.resolve()
    device = "0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    try:
        if _sha256(yolo_checkpoint) != args.yolo_sha256.upper():
            raise RuntimeError("The YOLO26s Stage 7 component hash does not match.")
        if _sha256(rtdetr_checkpoint) != args.rtdetr_sha256.upper():
            raise RuntimeError("The RT-DETR-L Stage 7 component hash does not match.")
        research_checkpoint = args.research_checkpoint.resolve() if args.research_checkpoint else None
        if research_checkpoint and (
            not args.research_sha256
            or _sha256(research_checkpoint) != args.research_sha256.upper()
        ):
            raise RuntimeError("The TWM-InfraDet research diagnostic hash does not match.")
        surface_checkpoint = args.surface_checkpoint.resolve() if args.surface_checkpoint else None
        if surface_checkpoint and (
            not args.surface_sha256
            or _sha256(surface_checkpoint) != args.surface_sha256.upper()
        ):
            raise RuntimeError("The R04 surface proposal checkpoint hash does not match.")
        if args.fusion_mode == "r07" and (not research_checkpoint or not surface_checkpoint):
            raise RuntimeError("R07 requires both frozen research and R04 surface checkpoints.")
        arrays, variants, image_size = _prepare_inputs(image_path)
        side_edge_shadow_variants = {"side_edge_left", "side_edge_right"}
        crack_shadow_variants = {
            "crack_mid",
            "crack_focus_left",
            "crack_focus_centre",
            "crack_focus_right",
        }
        isolated_shadow_variants = {*crack_shadow_variants, *side_edge_shadow_variants}
        preserve_r07 = args.fusion_mode == "r07"
        anchor, anchor_diagnostics = _model_candidates(
            yolo_checkpoint,
            arrays,
            variants,
            confidence=0.001 if preserve_r07 else args.anchor_confidence,
            imgsz=args.imgsz,
            iou=args.iou,
            max_det=100,
            device=device,
            isolated_variants=isolated_shadow_variants,
            preserve_variant_candidates=preserve_r07,
        )
        # Always run the teacher so a zero-anchor result can still be recovered
        # by the governed heterogeneous v0.1.2 fallback rules.
        teacher, teacher_diagnostics = _model_candidates(
            rtdetr_checkpoint,
            arrays,
            variants,
            confidence=0.005 if preserve_r07 else args.teacher_confidence,
            imgsz=args.imgsz,
            iou=args.iou,
            max_det=100,
            device=device,
            isolated_variants=isolated_shadow_variants,
            preserve_variant_candidates=preserve_r07,
        )
        if not preserve_r07:
            for candidate in anchor:
                candidate["appearance"] = _appearance_metrics(image_path, candidate["box"])
            for candidate in teacher:
                candidate["appearance"] = _appearance_metrics(image_path, candidate["box"])
        if research_checkpoint:
            research, research_diagnostics = _model_candidates(
                research_checkpoint,
                arrays,
                variants,
                confidence=0.55 if preserve_r07 else args.research_confidence,
                imgsz=args.imgsz,
                iou=args.iou,
                max_det=100,
                device=device,
                isolated_variants=isolated_shadow_variants,
                preserve_variant_candidates=preserve_r07,
            )
            if not preserve_r07:
                for candidate in research:
                    candidate["appearance"] = _appearance_metrics(image_path, candidate["box"])
        else:
            research = []
            research_diagnostics = {"skipped": True, "reason": "research_checkpoint_unavailable"}
        specialist_candidates: dict[str, list[dict[str, Any]]] = {}
        specialist_diagnostics: dict[str, dict[str, Any]] = {}
        specialist_component_sha256: dict[str, str] = {}
        for specification in args.specialist:
            parts = specification.split("|", 3)
            if len(parts) != 4:
                raise RuntimeError("The Stage 7 specialist diagnostic specification is invalid.")
            specialist_name, specialist_path_text, specialist_sha256, specialist_confidence_text = parts
            if not specialist_name or not specialist_name.replace("_", "").isalnum():
                raise RuntimeError("The Stage 7 specialist diagnostic name is invalid.")
            specialist_path = Path(specialist_path_text).resolve()
            if _sha256(specialist_path) != specialist_sha256.upper():
                raise RuntimeError(f"The {specialist_name} diagnostic hash does not match.")
            specialist, specialist_run = _model_candidates(
                specialist_path,
                arrays,
                variants,
                confidence=0.001 if preserve_r07 else float(specialist_confidence_text),
                imgsz=args.imgsz,
                iou=args.iou,
                max_det=100,
                device=device,
                isolated_variants=isolated_shadow_variants,
                preserve_variant_candidates=preserve_r07,
            )
            if not preserve_r07:
                for candidate in specialist:
                    candidate["appearance"] = _appearance_metrics(image_path, candidate["box"])
            specialist_candidates[specialist_name] = specialist
            specialist_diagnostics[specialist_name] = specialist_run
            specialist_component_sha256[specialist_name] = specialist_sha256.upper()
        if preserve_r07:
            surface, surface_diagnostics = _model_candidates(
                surface_checkpoint,
                arrays,
                variants,
                confidence=0.001,
                imgsz=args.imgsz,
                iou=args.iou,
                max_det=100,
                device=device,
                isolated_variants=isolated_shadow_variants,
                preserve_variant_candidates=True,
            )
            source_candidates = {
                "r04_exact": [item for item in surface if item["variant"] == "full"],
                # The frozen R07 development evidence includes the full variant in
                # the view-crop proposal package, so the live adapter mirrors it.
                "r04_viewcrop": surface,
                "yolo26s": anchor,
                "rtdetr_l": teacher,
                "twm_infradet_v1_5_1": research,
                **{
                    name: specialist_candidates.get(name, [])
                    for name in ("yolo26n", "yolo26l", "cycle01")
                },
            }
            predictions, r07_diagnostics = fuse_r07(
                source_candidates,
                max_detections=min(args.max_detections, 5),
                max_anchors=args.r07_max_anchors,
            )
            payload = {
                "schema_version": 1,
                "status": "inference_complete",
                "model_id": "twm-infradet-stage7-r07-high-recall-v0.1.19",
                "model_name": "TWM-InfraDet Stage 7 R07 High-Recall Fusion",
                "model_version": "Stage7-R07-HighRecall-v0.1.19-candidate03",
                "class_names": {
                    str(index): name for index, name in enumerate(R07_CLASS_NAMES)
                },
                "image_size": list(image_size),
                "image_sha256": _sha256(image_path),
                "component_sha256": {
                    "yolo26s": args.yolo_sha256.upper(),
                    "rtdetr_l": args.rtdetr_sha256.upper(),
                    "twm_infradet_v1_5_1": args.research_sha256.upper(),
                    "yolo26l_surface_r04_e11": args.surface_sha256.upper(),
                    **specialist_component_sha256,
                },
                "prediction_count": len(predictions),
                "predictions": predictions,
                "diagnostics": {
                    **r07_diagnostics,
                    "strategy": "r07_category_agnostic_multi_origin_spatial_fusion_with_balanced_high_recall_selection",
                    "fusion_mode": "r07",
                    "source_candidate_counts": {
                        name: len(items) for name, items in source_candidates.items()
                    },
                    "source_runs": {
                        "yolo26s": anchor_diagnostics,
                        "rtdetr_l": teacher_diagnostics,
                        "twm_infradet_v1_5_1": research_diagnostics,
                        "yolo26l_surface_r04_e11": surface_diagnostics,
                        **specialist_diagnostics,
                    },
                    "raw_proposals_persisted": False,
                    "provider_pixels_persisted": False,
                },
                "device": str(device),
                "timing_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return
        active_anchor = [
            item for item in anchor if item["variant"] not in isolated_shadow_variants
        ]
        active_teacher = [
            item for item in teacher if item["variant"] not in isolated_shadow_variants
        ]
        active_specialist_candidates = {
            name: [
                item for item in items if item["variant"] not in isolated_shadow_variants
            ]
            for name, items in specialist_candidates.items()
        }
        # Gate-170 and Gate-180 predecessors intentionally use the established
        # side-edge crops, but they must never see the new v0.1.16 crack crops.
        # Otherwise a shadow-only crop can change a predecessor's best-match
        # ordering and silently mutate the active v0.1.15 output.
        pre_v026_anchor = [
            item for item in anchor if item["variant"] not in crack_shadow_variants
        ]
        pre_v026_teacher = [
            item for item in teacher if item["variant"] not in crack_shadow_variants
        ]
        pre_v026_specialist_candidates = {
            name: [
                item for item in items if item["variant"] not in crack_shadow_variants
            ]
            for name, items in specialist_candidates.items()
        }
        shadow_only_variants = {"road_mid", "road_far", *isolated_shadow_variants}
        legacy_anchor = [item for item in anchor if item["variant"] not in shadow_only_variants]
        legacy_teacher = [item for item in teacher if item["variant"] not in shadow_only_variants]
        legacy_specialist_candidates = {
            name: [item for item in items if item["variant"] not in shadow_only_variants]
            for name, items in specialist_candidates.items()
        }
        predictions = _consensus(
            legacy_anchor,
            legacy_teacher,
            agreement_iou=args.agreement_iou,
            max_detections=args.max_detections,
        )
        appearance_rejections: list[dict[str, Any]] = []
        for prediction in predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        consensus_before_appearance = [dict(prediction) for prediction in predictions]
        candidate_v012_predictions, candidate_v012_diagnostics = _candidate_v012(
            consensus_before_appearance,
            legacy_teacher,
            legacy_specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v012_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v012_diagnostics["prediction_count"] = len(candidate_v012_predictions)
        candidate_v012_diagnostics["predictions"] = candidate_v012_predictions
        candidate_v013_predictions, candidate_v013_diagnostics = _candidate_v013(
            candidate_v012_predictions,
            legacy_anchor,
            legacy_specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v013_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v013_diagnostics["prediction_count"] = len(candidate_v013_predictions)
        candidate_v013_diagnostics["predictions"] = candidate_v013_predictions
        candidate_v014_predictions, candidate_v014_diagnostics = _candidate_v014(
            candidate_v013_predictions,
            legacy_teacher,
            legacy_specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v014_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v014_diagnostics["prediction_count"] = len(candidate_v014_predictions)
        candidate_v014_diagnostics["predictions"] = candidate_v014_predictions
        candidate_v015_predictions, candidate_v015_diagnostics = _candidate_v015(
            candidate_v014_predictions,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v015_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v015_diagnostics["prediction_count"] = len(candidate_v015_predictions)
        candidate_v015_diagnostics["predictions"] = candidate_v015_predictions
        road_surface_grid = _road_surface_grid(image_path)
        road_crack_structure = _road_crack_structure_diagnostics(image_path)
        candidate_v016_predictions, candidate_v016_diagnostics = _candidate_v016(
            candidate_v015_predictions,
            active_anchor,
            active_teacher,
            active_specialist_candidates,
            road_surface_grid,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v016_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v016_diagnostics["prediction_count"] = len(candidate_v016_predictions)
        candidate_v016_diagnostics["predictions"] = candidate_v016_predictions
        candidate_v017_predictions, candidate_v017_diagnostics = _candidate_v017(
            candidate_v016_predictions,
            active_anchor,
            active_teacher,
            active_specialist_candidates,
            args.frame_identity,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v017_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v017_diagnostics["prediction_count"] = len(candidate_v017_predictions)
        candidate_v017_diagnostics["predictions"] = candidate_v017_predictions
        candidate_v018_predictions, candidate_v018_diagnostics = _candidate_v018(
            candidate_v017_predictions,
            active_teacher,
            active_specialist_candidates,
            args.frame_identity,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v018_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v018_diagnostics["prediction_count"] = len(candidate_v018_predictions)
        candidate_v018_diagnostics["predictions"] = candidate_v018_predictions
        candidate_v019_predictions, candidate_v019_diagnostics = _candidate_v019(
            candidate_v018_predictions,
            active_anchor,
            active_teacher,
            active_specialist_candidates,
            args.frame_identity,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v019_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v019_diagnostics["prediction_count"] = len(candidate_v019_predictions)
        candidate_v019_diagnostics["predictions"] = candidate_v019_predictions
        candidate_v020_predictions, candidate_v020_diagnostics = _candidate_v020(
            candidate_v019_predictions,
            active_teacher,
            active_specialist_candidates,
            args.frame_identity,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v020_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v020_diagnostics["prediction_count"] = len(candidate_v020_predictions)
        candidate_v020_diagnostics["predictions"] = candidate_v020_predictions
        candidate_v021_predictions, candidate_v021_diagnostics = _candidate_v021(
            candidate_v020_predictions,
            active_teacher,
            active_specialist_candidates,
            args.frame_identity,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v021_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v021_diagnostics["prediction_count"] = len(candidate_v021_predictions)
        candidate_v021_diagnostics["predictions"] = candidate_v021_predictions
        candidate_v022_predictions, candidate_v022_diagnostics = _candidate_v022(
            candidate_v021_predictions,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v022_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v022_diagnostics["prediction_count"] = len(candidate_v022_predictions)
        candidate_v022_diagnostics["predictions"] = candidate_v022_predictions
        candidate_v023_predictions, candidate_v023_diagnostics = _candidate_v023(
            candidate_v022_predictions,
            pre_v026_anchor,
            pre_v026_teacher,
            pre_v026_specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v023_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v023_diagnostics["prediction_count"] = len(candidate_v023_predictions)
        candidate_v023_diagnostics["predictions"] = candidate_v023_predictions
        candidate_v024_predictions, candidate_v024_diagnostics = _candidate_v024(
            candidate_v023_predictions,
            pre_v026_teacher,
            pre_v026_specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v024_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v024_diagnostics["prediction_count"] = len(candidate_v024_predictions)
        candidate_v024_diagnostics["predictions"] = candidate_v024_predictions
        candidate_v025_predictions, candidate_v025_diagnostics = _candidate_v025(
            candidate_v024_predictions,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v025_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v025_diagnostics["prediction_count"] = len(candidate_v025_predictions)
        candidate_v025_diagnostics["predictions"] = candidate_v025_predictions
        candidate_v026_predictions, candidate_v026_diagnostics = _candidate_v026(
            candidate_v025_predictions,
            anchor,
            teacher,
            specialist_candidates,
            road_crack_structure,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v026_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v026_diagnostics["prediction_count"] = len(candidate_v026_predictions)
        candidate_v026_diagnostics["predictions"] = candidate_v026_predictions
        candidate_v027_predictions, candidate_v027_diagnostics = _candidate_v027(
            candidate_v026_predictions,
            anchor,
            teacher,
            specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v027_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v027_diagnostics["prediction_count"] = len(candidate_v027_predictions)
        candidate_v027_diagnostics["predictions"] = candidate_v027_predictions
        candidate_v028_predictions, candidate_v028_diagnostics = _candidate_v028(
            candidate_v027_predictions,
            teacher,
            specialist_candidates,
            max_detections=args.max_detections,
        )
        for prediction in candidate_v028_predictions:
            prediction["appearance"] = _appearance_metrics(image_path, prediction["box"])
        candidate_v028_diagnostics["prediction_count"] = len(candidate_v028_predictions)
        candidate_v028_diagnostics["predictions"] = candidate_v028_predictions
        retained_predictions: list[dict[str, Any]] = []
        for prediction in predictions:
            appearance = prediction["appearance"]
            if prediction["class_id"] == 6 and (
                appearance["mean_luminance"] > 135.0
                or appearance["luminance_std"] > 50.0
            ):
                appearance_rejections.append(
                    {
                        "class_id": prediction["class_id"],
                        "confidence": prediction["confidence"],
                        "box": prediction["box"],
                        "agreement_iou": prediction.get("agreement_iou"),
                        "support": prediction.get("support"),
                        "reason": "standing_water_region_inconsistent_with_dark_low_variance_surface",
                        "appearance": appearance,
                    }
                )
                continue
            retained_predictions.append(prediction)
        legacy_v011_predictions = retained_predictions
        predictions = candidate_v027_predictions
        candidate_v012_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v013_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v014_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v015_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v016_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v017_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v018_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v019_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v020_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v021_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v022_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v023_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v024_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v025_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v026_diagnostics["activation_state"] = "accepted_predecessor"
        candidate_v027_diagnostics["activation_state"] = "active"
        candidate_v028_diagnostics["activation_state"] = "shadow_only"
        payload = {
            "schema_version": 1,
            "status": "inference_complete",
            "model_id": "twm-infradet-stage7-heterogeneous-consensus-v0.1.17",
            "model_name": "TWM-InfraDet Stage 7 Heterogeneous Consensus",
            "model_version": "Stage7-HMT-Consensus-v0.1.17",
            "class_names": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "image_size": list(image_size),
            "image_sha256": _sha256(image_path),
            "component_sha256": {
                "yolo26s": args.yolo_sha256.upper(),
                "rtdetr_l": args.rtdetr_sha256.upper(),
                **specialist_component_sha256,
            },
            "prediction_count": len(predictions),
            "predictions": predictions,
            "diagnostics": {
                "strategy": "governed_multiscale_heterogeneous_teacher_fusion_v0_1_17",
                "legacy_v011_prediction_count": len(legacy_v011_predictions),
                "anchor_candidate_count": len(anchor),
                "teacher_candidate_count": len(teacher),
                "research_candidate_count": len(research),
                "class_names": {str(index): name for index, name in enumerate(CLASS_NAMES)},
                "anchor_candidates": anchor,
                "teacher_candidates": teacher,
                "research_candidates": research,
                "specialist_candidates": specialist_candidates,
                "consensus_before_appearance": consensus_before_appearance,
                "candidate_v012": candidate_v012_diagnostics,
                "candidate_v013": candidate_v013_diagnostics,
                "candidate_v014": candidate_v014_diagnostics,
                "candidate_v015": candidate_v015_diagnostics,
                "candidate_v016": candidate_v016_diagnostics,
                "candidate_v017": candidate_v017_diagnostics,
                "candidate_v018": candidate_v018_diagnostics,
                "candidate_v019": candidate_v019_diagnostics,
                "candidate_v020": candidate_v020_diagnostics,
                "candidate_v021": candidate_v021_diagnostics,
                "candidate_v022": candidate_v022_diagnostics,
                "candidate_v023": candidate_v023_diagnostics,
                "candidate_v024": candidate_v024_diagnostics,
                "candidate_v025": candidate_v025_diagnostics,
                "candidate_v026": candidate_v026_diagnostics,
                "candidate_v027": candidate_v027_diagnostics,
                "candidate_v028": candidate_v028_diagnostics,
                "road_surface_grid": road_surface_grid,
                "road_crack_structure": road_crack_structure,
                "anchor": anchor_diagnostics,
                "teacher": teacher_diagnostics,
                "research": research_diagnostics,
                "specialists": specialist_diagnostics,
                "research_confidence_floor": args.research_confidence,
                "agreement_iou_threshold": args.agreement_iou,
                "anchor_confidence_floor": args.anchor_confidence,
                "teacher_confidence_floor": args.teacher_confidence,
                "appearance_filter": {
                    "standing_water_max_mean_luminance": 135.0,
                    "standing_water_max_luminance_std": 50.0,
                    "rejected_count": len(appearance_rejections),
                    "rejections": appearance_rejections,
                },
            },
            "device": str(device),
            "timing_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        }
    except Exception as error:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "runtime": {
                "python_executable": sys.executable,
                "torch_version": getattr(torch, "__version__", None),
                "torch_file": getattr(torch, "__file__", None),
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
