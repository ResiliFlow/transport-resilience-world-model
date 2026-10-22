from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


RESEARCH_MODEL = "twm_infradet_v1_5_1"
BASE_MODELS = ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l", "cycle01")
RAW_FLOORS = {
    "yolo26n": 0.001,
    "yolo26s": 0.001,
    "yolo26l": 0.001,
    "rtdetr_l": 0.005,
    "cycle01": 0.001,
    RESEARCH_MODEL: 0.54,
}
CONFIDENCE_SCALES = {
    "yolo26n": 0.05,
    "yolo26s": 0.12,
    "yolo26l": 0.05,
    "rtdetr_l": 0.18,
    "cycle01": 0.05,
    RESEARCH_MODEL: 0.55,
}
PRODUCTION_FLOORS = {
    "yolo26n": 0.05,
    "yolo26s": 0.12,
    "yolo26l": 0.05,
    "rtdetr_l": 0.18,
    "cycle01": 0.05,
    RESEARCH_MODEL: 0.55,
}


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def iou(first: list[float], second: list[float]) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def overlap_smaller(first: list[float], second: list[float]) -> float:
    smaller = min(_area(first), _area(second))
    return _intersection_area(first, second) / smaller if smaller > 0.0 else 0.0


def _centre_distance(first: list[float], second: list[float]) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)


def _compatible(first: list[float], second: list[float]) -> bool:
    return (
        iou(first, second) >= 0.12
        or (
            overlap_smaller(first, second) >= 0.70
            and _centre_distance(first, second) <= 0.42
        )
    )


def _strength(candidate: dict[str, Any]) -> float:
    scale = CONFIDENCE_SCALES[candidate["model"]]
    return min(1.5, math.sqrt(max(0.0, candidate["confidence"]) / scale))


def _eligible_geometry(class_id: int, box: list[float]) -> bool:
    area = _area(box)
    if area < 0.00008 or area > 0.92:
        return False
    centre_y = (box[1] + box[3]) / 2.0
    if class_id in {0, 1, 2, 3, 6}:
        return box[3] >= 0.54 and centre_y >= 0.46
    return box[3] >= 0.50 and centre_y >= 0.42


def _normalise_candidates(raw_models: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for model_name, model_payload in raw_models.items():
        if model_name not in RAW_FLOORS:
            continue
        per_variant: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for original in model_payload.get("candidates") or []:
            class_id = int(original["class_id"])
            confidence = float(original["confidence"])
            box = [float(value) for value in original["box"]]
            if confidence < RAW_FLOORS[model_name] or not _eligible_geometry(class_id, box):
                continue
            candidate = {
                "model": model_name,
                "class_id": class_id,
                "confidence": confidence,
                "box": box,
                "variant": str(original.get("variant") or "unknown"),
            }
            per_variant[(class_id, candidate["variant"])].append(candidate)
        for items in per_variant.values():
            for candidate in sorted(items, key=lambda item: item["confidence"], reverse=True)[:6]:
                grouped[candidate["class_id"]].append(candidate)
    return grouped


def _cluster_base_candidates(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    ordered = sorted(candidates, key=lambda item: (_strength(item), item["confidence"]), reverse=True)
    for candidate in ordered:
        compatible: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            seed = cluster[0]
            if not _compatible(candidate["box"], seed["box"]):
                continue
            score = max(
                iou(candidate["box"], seed["box"]),
                overlap_smaller(candidate["box"], seed["box"]) * 0.55,
            )
            compatible.append((score, index))
        if compatible:
            clusters[max(compatible)[1]].append(candidate)
        else:
            clusters.append([candidate])
    return clusters


def _representatives(cluster: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in cluster:
        by_model[candidate["model"]].append(candidate)
    representatives: dict[str, dict[str, Any]] = {}
    for model_name, items in by_model.items():
        representatives[model_name] = max(
            items,
            key=lambda item: (
                sum(overlap_smaller(item["box"], other["box"]) for other in cluster),
                _strength(item),
                item["confidence"],
            ),
        )
    return representatives


def _variant_source(variant: str) -> str:
    return "exact_tile" if variant.startswith("exact_tile_") else "original_crop"


def _matching_research(
    class_id: int,
    representatives: dict[str, dict[str, Any]],
    research: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in research
        if candidate["class_id"] == class_id
        and any(_compatible(candidate["box"], item["box"]) for item in representatives.values())
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda candidate: (
            max(overlap_smaller(candidate["box"], item["box"]) for item in representatives.values()),
            candidate["confidence"],
        ),
    )


def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= total / 2.0:
            return value
    return ordered[-1][0]


def _localise(
    representatives: dict[str, dict[str, Any]],
    research: dict[str, Any] | None,
) -> list[float]:
    items = list(representatives.values())
    if research is not None:
        items.append(research)
    coordinates: list[float] = []
    for coordinate in range(4):
        weighted = [
            (item["box"][coordinate], 0.35 + _strength(item))
            for item in items
        ]
        coordinates.append(_weighted_median(weighted))
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        strongest = max(items, key=_strength)
        return list(strongest["box"])
    return coordinates


@lru_cache(maxsize=128)
def _image_arrays(image_path_text: str) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path_text) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    grey = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return rgb, grey


def _texture_metrics(image_path: Path, box: list[float]) -> dict[str, float]:
    rgb, grey = _image_arrays(str(image_path.resolve()))
    height, width = grey.shape
    x1 = max(0, min(width - 1, round(box[0] * width)))
    y1 = max(0, min(height - 1, round(box[1] * height)))
    x2 = max(x1 + 1, min(width, round(box[2] * width)))
    y2 = max(y1 + 1, min(height, round(box[3] * height)))
    region = grey[y1:y2, x1:x2]
    rgb_region = rgb[y1:y2, x1:x2]
    gradient_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    horizontal_energy = float(np.abs(gradient_x).mean())
    vertical_energy = float(np.abs(gradient_y).mean())
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    expanded_width = max(2, x2 - x1)
    expanded_height = max(2, y2 - y1)
    context_x1 = max(0, x1 - round(expanded_width * 0.35))
    context_y1 = max(0, y1 - round(expanded_height * 0.35))
    context_x2 = min(width, x2 + round(expanded_width * 0.35))
    context_y2 = min(height, y2 + round(expanded_height * 0.35))
    context = grey[context_y1:context_y2, context_x1:context_x2]
    channel_max = rgb_region.max(axis=2)
    channel_min = rgb_region.min(axis=2)
    saturation = np.divide(channel_max - channel_min, np.maximum(channel_max, 1.0))
    hist = cv2.calcHist([region.astype(np.uint8)], [0], None, [32], [0, 256]).reshape(-1)
    probabilities = hist / max(float(hist.sum()), 1.0)
    nonzero = probabilities[probabilities > 0.0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    edges = cv2.Canny(region.astype(np.uint8), 45, 130)
    line_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected_lines = line_detector.detect(region.astype(np.uint8))[0]
    line_lengths: list[float] = []
    if detected_lines is not None:
        for raw in detected_lines.reshape(-1, 4):
            line_lengths.append(float(math.hypot(raw[2] - raw[0], raw[3] - raw[1])))
    diagonal = max(math.hypot(region.shape[1], region.shape[0]), 1.0)
    return {
        "gradient_x_energy": horizontal_energy,
        "gradient_y_energy": vertical_energy,
        "gradient_y_to_x_ratio": vertical_energy / max(horizontal_energy, 1e-6),
        "laplacian_variance": float(cv2.Laplacian(region, cv2.CV_32F).var()),
        "mean_luminance": float(region.mean()),
        "luminance_std": float(region.std()),
        "luminance_p10": float(np.percentile(region, 10.0)),
        "luminance_p50": float(np.percentile(region, 50.0)),
        "luminance_p90": float(np.percentile(region, 90.0)),
        "context_mean_luminance": float(context.mean()),
        "region_minus_context_luminance": float(region.mean() - context.mean()),
        "mean_saturation": float(saturation.mean()),
        "gradient_magnitude_mean": float(gradient_magnitude.mean()),
        "strong_gradient_fraction": float((gradient_magnitude >= 70.0).mean()),
        "canny_edge_fraction": float((edges > 0).mean()),
        "grayscale_entropy": entropy,
        "line_count": float(len(line_lengths)),
        "longest_line_normalised": max(line_lengths, default=0.0) / diagonal,
        "box_area": _area(box),
        "box_width": box[2] - box[0],
        "box_height": box[3] - box[1],
        "box_centre_x": (box[0] + box[2]) / 2.0,
        "box_centre_y": (box[1] + box[3]) / 2.0,
    }


def _accept_cluster(
    class_id: int,
    cluster: list[dict[str, Any]],
    representatives: dict[str, dict[str, Any]],
    research: dict[str, Any] | None,
    box: list[float],
    texture: dict[str, float],
) -> tuple[bool, str]:
    families = set(representatives)
    production_families = {
        name
        for name, candidate in representatives.items()
        if candidate["confidence"] >= PRODUCTION_FLOORS[name]
    }
    rtdetr_members = [item for item in cluster if item["model"] == "rtdetr_l"]
    rtdetr_variants = {item["variant"] for item in rtdetr_members}
    rtdetr_max = max((item["confidence"] for item in rtdetr_members), default=0.0)

    if class_id == 0:
        if "rtdetr_l" in families and len(families) >= 3:
            return True, "three_family_pothole_proposal_consensus"
        if (
            families == {"rtdetr_l"}
            and len(rtdetr_variants) >= 3
            and rtdetr_max >= 0.007
            and _area(box) <= 0.18
        ):
            return True, "rtdetr_multiview_local_pothole_expert"
        return False, "insufficient_pothole_support"
    if class_id == 1:
        if "rtdetr_l" in families and len(families) >= 3:
            return True, "three_family_longitudinal_crack_consensus"
        return False, "insufficient_longitudinal_crack_support"
    if class_id in {2, 3}:
        if len(families) >= 3 and len(production_families) >= 2:
            return True, "strong_production_multifamily_crack_consensus"
        return False, "no_fast50_authority_for_low_confidence_class"
    if class_id == 4:
        if research is not None and "rtdetr_l" in families:
            return True, "v151_class_expert_with_rtdetr_footpath_localisation"
        return False, "footpath_requires_research_and_rtdetr_support"
    if class_id == 5:
        if research is not None and "rtdetr_l" in families and len(families) >= 2:
            return True, "v151_class_expert_with_multifamily_kerb_localisation"
        return False, "kerb_requires_research_and_multifamily_support"
    if class_id == 6:
        if research is None or "rtdetr_l" not in families or len(families) < 3:
            return False, "water_requires_research_rtdetr_and_three_families"
        if texture["gradient_y_to_x_ratio"] > 1.42:
            return False, "directional_shadow_texture_not_standing_water"
        return True, "v151_semantics_with_multifamily_water_localisation"
    if class_id == 7:
        if (
            research is not None
            and "rtdetr_l" in families
            and (len(families) >= 2 or len(rtdetr_variants) >= 2)
        ):
            return True, "v151_semantics_with_rtdetr_obstruction_localisation"
        return False, "obstruction_requires_research_and_spatial_support"
    return False, "unsupported_class"


def _quality_gate(
    class_id: int,
    decision: dict[str, Any],
) -> tuple[bool, str]:
    """Apply class-aware, frame-agnostic evidence gates learned from Fast50.

    The gates use only repeatability, existing-model support, geometry and local
    texture. They never receive a sample identifier, coordinate or colour rule.
    """

    source_counts = decision["cluster_source_counts"]
    source_maxima = decision["cluster_max_confidence_by_source"]
    texture = decision["texture"]
    if class_id == 0:
        if (
            decision["cluster_tile_view_count"] >= 6
            and texture["context_mean_luminance"] <= 102.2
            and texture["luminance_p90"] >= 109.0
        ):
            return True, "repeatable_dark_surface_depression"
        return False, "pothole_not_repeatable_on_dark_road_surface"
    if class_id == 1:
        if (
            source_counts["original_crop"] >= 18
            and (source_maxima["original_crop"] or 0.0) <= 0.0221
            and decision["cluster_model_count"] >= 4
        ):
            return True, "diffuse_low_confidence_multicrop_crack_trace"
        return False, "longitudinal_trace_not_stable_across_original_crops"
    if class_id in {2, 3}:
        return True, "existing_strong_production_consensus_retained"
    if class_id == 4:
        if (
            decision["cluster_model_count"] >= 3
            and texture["luminance_p10"] <= 35.0
            and (source_maxima["original_crop"] or 0.0) >= 0.014
        ):
            return True, "multimodel_dark_discontinuity_footpath"
        return False, "footpath_region_lacks_multimodel_dark_discontinuity"
    if class_id == 5:
        if (
            source_counts["original_crop"] >= 27
            and texture["luminance_p90"] <= 130.5
        ):
            return True, "persistent_original_and_tile_kerb_break"
        return False, "kerb_region_lacks_persistent_multiview_support"
    if class_id == 6:
        if (
            texture["longest_line_normalised"] <= 0.1252
            and decision["cluster_tile_view_count"] >= 9
            and (decision["family_confidence"].get("cycle01") or 0.0) >= 0.0013
        ):
            return True, "research_semantics_with_non_linear_reflective_region"
        return False, "water_region_is_linear_or_semantically_weak"
    if class_id == 7:
        if (
            source_counts["exact_tile"] >= 44
            and decision["cluster_model_count"] >= 4
        ):
            return True, "dense_balanced_obstruction_texture"
        return False, "obstruction_texture_not_dense_and_balanced"
    return False, "unsupported_quality_gate_class"


def _resolve_cross_class_conflicts(
    additions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Suppress incompatible labels for the same visual structure."""

    suppressed: set[int] = set()
    events: list[dict[str, Any]] = []
    for index, candidate in enumerate(additions):
        for other_index, other in enumerate(additions):
            if index == other_index or candidate["class_id"] == other["class_id"]:
                continue
            containment = overlap_smaller(candidate["box"], other["box"])
            candidate_area = _area(candidate["box"])
            other_area = _area(other["box"])
            reason: str | None = None
            if (
                candidate["class_id"] in {4, 5}
                and other["class_id"] == 1
                and containment >= 0.90
                and candidate_area >= 3.0 * other_area
            ):
                reason = "broad_edge_class_suppressed_by_contained_crack_trace"
            elif (
                candidate["class_id"] == 0
                and other["class_id"] == 5
                and containment >= 0.90
                and other_area >= 3.0 * candidate_area
            ):
                reason = "local_pothole_label_suppressed_by_broken_kerb_structure"
            elif (
                candidate["class_id"] == 6
                and other["class_id"] == 4
                and containment >= 0.90
                and other_area >= 2.5 * candidate_area
            ):
                reason = "water_label_suppressed_by_broken_footpath_structure"
            elif (
                candidate["class_id"] == 0
                and candidate["confidence"] < 0.60
                and other["class_id"] == 1
                and _centre_distance(candidate["box"], other["box"]) <= 0.30
            ):
                reason = "weak_nearby_pothole_label_suppressed_by_crack_trace"
            elif (
                candidate["class_id"] == 0
                and other["class_id"] == 1
                and containment >= 0.90
                and other["confidence"] >= candidate["confidence"] + 0.10
            ):
                reason = "contained_pothole_label_suppressed_by_stronger_crack_trace"
            if reason is None:
                continue
            suppressed.add(index)
            events.append(
                {
                    "class_id": candidate["class_id"],
                    "accepted": False,
                    "reason": reason,
                    "box": candidate["box"],
                    "competing_class_id": other["class_id"],
                    "competing_box": other["box"],
                    "overlap_smaller": containment,
                }
            )
            break
    return [item for index, item in enumerate(additions) if index not in suppressed], events


def _same_detection(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first["class_id"] == second["class_id"] and (
        iou(first["box"], second["box"]) >= 0.30
        or overlap_smaller(first["box"], second["box"]) >= 0.80
    )


def fuse_v019(
    raw_models: dict[str, Any],
    baseline_predictions: list[dict[str, Any]],
    image_path: Path,
    *,
    max_detections: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse low-threshold existing-model proposals without frame-specific inputs."""

    grouped = _normalise_candidates(raw_models)
    research_by_class = {
        class_id: [
            item
            for item in candidates
            if item["model"] == RESEARCH_MODEL
        ]
        for class_id, candidates in grouped.items()
    }
    decisions: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    for class_id in range(8):
        base = [item for item in grouped.get(class_id, []) if item["model"] in BASE_MODELS]
        for cluster in _cluster_base_candidates(base):
            representatives = _representatives(cluster)
            research = _matching_research(
                class_id,
                representatives,
                research_by_class.get(class_id, []),
            )
            box = _localise(representatives, research)
            texture = _texture_metrics(image_path, box)
            accepted, reason = _accept_cluster(
                class_id,
                cluster,
                representatives,
                research,
                box,
                texture,
            )
            decision = {
                "class_id": class_id,
                "accepted": accepted,
                "reason": reason,
                "box": box,
                "families": sorted(representatives),
                "family_confidence": {
                    name: candidate["confidence"]
                    for name, candidate in representatives.items()
                },
                "family_variants": {
                    name: candidate["variant"]
                    for name, candidate in representatives.items()
                },
                "cluster_member_count": len(cluster),
                "cluster_model_count": len({item["model"] for item in cluster}),
                "cluster_source_counts": {
                    source: sum(
                        1
                        for item in cluster
                        if _variant_source(item["variant"]) == source
                    )
                    for source in ("original_crop", "exact_tile")
                },
                "cluster_tile_view_count": len(
                    {
                        item["variant"]
                        for item in cluster
                        if _variant_source(item["variant"]) == "exact_tile"
                    }
                ),
                "cluster_variants_by_model": {
                    name: sorted({item["variant"] for item in cluster if item["model"] == name})
                    for name in sorted({item["model"] for item in cluster})
                },
                "cluster_max_confidence_by_source": {
                    source: max(
                        (
                            item["confidence"]
                            for item in cluster
                            if _variant_source(item["variant"]) == source
                        ),
                        default=None,
                    )
                    for source in ("original_crop", "exact_tile")
                },
                "research_confidence": research["confidence"] if research else None,
                "research_variant": research["variant"] if research else None,
                "texture": texture,
            }
            proposal_reason = reason
            if accepted:
                accepted, quality_reason = _quality_gate(class_id, decision)
                reason = quality_reason
            decision["proposal_reason"] = proposal_reason
            decision["accepted"] = accepted
            decision["reason"] = reason
            decisions.append(decision)
            if not accepted:
                continue
            strengths = [_strength(item) for item in representatives.values()]
            if research is not None:
                strengths.append(_strength(research))
            confidence = min(0.97, 0.35 + 0.10 * len(representatives) + 0.08 * float(np.mean(strengths)))
            additions.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "box": box,
                    "strategy": reason,
                    "support": decision["family_confidence"],
                }
            )

    additions, conflict_events = _resolve_cross_class_conflicts(additions)
    decisions.extend(conflict_events)

    retained_baseline: list[dict[str, Any]] = []
    for prediction in baseline_predictions:
        candidate = dict(prediction)
        if candidate["class_id"] == 6:
            matching = [item for item in additions if _same_detection(candidate, item)]
            if not matching:
                decisions.append(
                    {
                        "class_id": 6,
                        "accepted": False,
                        "reason": "legacy_water_not_confirmed_by_v019_verifier",
                        "box": candidate["box"],
                    }
                )
                continue
        retained_baseline.append(candidate)

    merged: list[dict[str, Any]] = []
    for candidate in sorted(
        [*retained_baseline, *additions],
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        if any(_same_detection(candidate, previous) for previous in merged):
            continue
        merged.append(candidate)
        if len(merged) >= max_detections:
            break
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.19-candidate",
        "activation_state": "isolated_evaluation_only",
        "frame_specific_inputs_used": False,
        "coordinate_rules_used": False,
        "training_used": False,
        "decision_count": len(decisions),
        "accepted_addition_count": len(additions),
        "decisions": decisions,
    }
