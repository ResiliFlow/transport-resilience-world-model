from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from infrastructure_perception.backend.stage7_fusion_r07 import (
    EIGHT_CLASS_NAMES,
    PRODUCTION_FLOORS,
    RESEARCH_MODEL,
    SPARSE_EXPERTS,
    _normalise_sources,
    _normalised_confidence,
    area,
    centre_distance,
    iou,
    overlap_smaller,
)
from infrastructure_perception.backend.stage7_fusion_r08_candidate01 import (
    CLASS_MAXIMUM_AREAS,
    EDGE_VARIANTS,
    RELIABLE_FAMILIES,
    ROAD_VARIANTS,
    SOURCE_WEIGHTS,
)


CANDIDATE_MODEL_VERSION = "Stage7-R09-LearnedProposalVerifier-v0.2.1-candidate02"
CALIBRATION_SEED = 20260915
DENSE_FAMILIES = {"rtdetr_l", RESEARCH_MODEL}
MAXIMUM_ANCHORS_PER_CLASS = 140
MAXIMUM_RECORDS_PER_FRAME = 1600

# The learned verifier is deliberately class-agnostic. This prevents the sparse
# calibration labels from silently disabling any of the eight supported classes.
FEATURE_NAMES = (
    "family_fraction",
    "reliable_family_fraction",
    "sparse_family_fraction",
    "dense_family_fraction",
    "strong_family_fraction",
    "r04_present",
    "r04_exact_present",
    "r04_view_variant_fraction",
    "unique_variant_fraction",
    "full_variant_present",
    "mean_spatial_support",
    "minimum_spatial_support",
    "maximum_spatial_support",
    "mean_normalised_confidence",
    "maximum_normalised_confidence",
    "minimum_normalised_confidence",
    "box_source_weight",
    "box_source_normalised_confidence",
    "box_source_production_strong",
    "box_mean_iou_to_representatives",
    "box_max_iou_to_representatives",
    "box_mean_overlap_smaller",
    "box_area_fraction",
    "box_width",
    "box_height",
    "box_centre_y",
    "box_horizontal_edge_proximity",
    "coordinate_dispersion",
    "log_area_ratio",
    "edge_variant_share",
    "road_variant_share",
)


def _pair_support(first: list[float], second: list[float]) -> tuple[bool, float]:
    pair_iou = iou(first, second)
    containment = overlap_smaller(first, second)
    distance = centre_distance(first, second)
    first_area = area(first)
    second_area = area(second)
    ratio = max(first_area, second_area) / max(min(first_area, second_area), 1e-9)
    supported = pair_iou >= 0.10 or (
        containment >= 0.60 and distance <= 0.24 and ratio <= 6.0
    )
    return supported, max(pair_iou, 0.72 * containment) if supported else 0.0


def _anchor_priority(item: dict[str, Any]) -> tuple[float, float, float]:
    family = str(item["family"])
    return (
        SOURCE_WEIGHTS.get(family, 0.70) + 0.18 * min(2.0, _normalised_confidence(item)),
        -area(item["box"]),
        float(item["confidence"]),
    )


def _anchors(items: list[dict[str, Any]], class_id: int) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in items
        if int(item["class_id"]) == class_id and item["family"] in RELIABLE_FAMILIES
    ]
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=_anchor_priority, reverse=True):
        if any(
            iou(candidate["box"], previous["box"]) >= 0.92
            or (
                overlap_smaller(candidate["box"], previous["box"]) >= 0.98
                and centre_distance(candidate["box"], previous["box"]) <= 0.04
            )
            for previous in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= MAXIMUM_ANCHORS_PER_CLASS:
            break
    return retained


def _representatives(
    anchor: dict[str, Any],
    same_class_items: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], float]]:
    selected: dict[str, tuple[dict[str, Any], float]] = {}
    for item in same_class_items:
        supported, spatial = _pair_support(anchor["box"], item["box"])
        if not supported:
            continue
        family = str(item["family"])
        value = spatial + 0.08 * min(2.0, _normalised_confidence(item))
        current = selected.get(family)
        if current is None or value > current[1]:
            selected[family] = (item, value)
    return selected


def _cap_dimension(box: list[float], maximum_width: float, maximum_height: float) -> list[float]:
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    width = min(x2 - x1, maximum_width)
    height = min(y2 - y1, maximum_height)
    return [
        max(0.0, centre_x - width / 2.0),
        max(0.42, centre_y - height / 2.0),
        min(1.0, centre_x + width / 2.0),
        min(1.0, centre_y + height / 2.0),
    ]


def _cap_area(box: list[float], maximum_area: float) -> list[float]:
    current = area(box)
    if current <= maximum_area:
        return box
    scale = math.sqrt(maximum_area / max(current, 1e-9))
    x1, y1, x2, y2 = box
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    width = (x2 - x1) * scale
    height = (y2 - y1) * scale
    return [
        max(0.0, centre_x - width / 2.0),
        max(0.42, centre_y - height / 2.0),
        min(1.0, centre_x + width / 2.0),
        min(1.0, centre_y + height / 2.0),
    ]


def _professional_box(class_id: int, raw_box: list[float]) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in raw_box]
    y1 = max(0.42, y1)
    box = _cap_dimension([x1, y1, x2, y2], 0.86, 0.58)
    box = _cap_area(box, CLASS_MAXIMUM_AREAS[class_id])
    if area(box) < 0.0015:
        return None
    width = box[2] - box[0]
    height = box[3] - box[1]
    centre_y = (box[1] + box[3]) / 2.0
    if box[3] < 0.70 or centre_y < 0.54:
        return None
    if width >= 0.88 or height >= 0.62:
        return None
    return box


def _host_surface(
    class_id: int,
    box: list[float],
    representatives: dict[str, tuple[dict[str, Any], float]],
) -> tuple[str, dict[str, Any]]:
    if class_id <= 3:
        return "road", {"rule": "road_defect_taxonomy"}
    if class_id == 4:
        return "footpath", {"rule": "footpath_defect_taxonomy"}
    if class_id == 5:
        return "kerb", {"rule": "kerb_defect_taxonomy"}
    variants = [str(item["variant"]) for item, _score in representatives.values()]
    edge_votes = sum(variant in EDGE_VARIANTS for variant in variants)
    road_votes = sum(variant in ROAD_VARIANTS for variant in variants)
    centre_x = (box[0] + box[2]) / 2.0
    width = box[2] - box[0]
    extreme_edge = centre_x <= 0.20 or centre_x >= 0.80
    very_extreme_tight_edge = (centre_x <= 0.14 or centre_x >= 0.86) and width <= 0.24
    host = (
        "footpath"
        if very_extreme_tight_edge
        or (extreme_edge and width <= 0.30 and edge_votes >= max(2, road_votes))
        else "road"
    )
    return host, {
        "rule": "multi_host_geometry_and_crop_consensus",
        "edge_votes": edge_votes,
        "road_votes": road_votes,
        "extreme_edge": extreme_edge,
        "very_extreme_tight_edge": very_extreme_tight_edge,
    }


def _coordinate_dispersion(boxes: list[list[float]]) -> float:
    if len(boxes) <= 1:
        return 0.0
    dispersion = 0.0
    for coordinate in range(4):
        values = [box[coordinate] for box in boxes]
        mean = sum(values) / len(values)
        dispersion += math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        )
    return dispersion / 4.0


def _feature_vector(
    candidate_item: dict[str, Any],
    box: list[float],
    representatives: dict[str, tuple[dict[str, Any], float]],
    matched_items: list[dict[str, Any]],
) -> tuple[list[float], dict[str, float]]:
    families = set(representatives)
    reliable = families.intersection(RELIABLE_FAMILIES)
    sparse = families.intersection(SPARSE_EXPERTS)
    dense = families.intersection(DENSE_FAMILIES)
    representative_items = [item for item, _score in representatives.values()]
    representative_boxes = [item["box"] for item in representative_items]
    spatial_scores = [min(1.0, score) for _item, score in representatives.values()]
    normalised_confidences = [
        min(3.0, _normalised_confidence(item)) / 3.0 for item in representative_items
    ]
    strong_count = sum(
        item["confidence"] >= PRODUCTION_FLOORS.get(item["source"], 0.05)
        for item in representative_items
    )
    all_variants = {str(item["variant"]) for item in matched_items}
    r04_view_variants = {
        str(item["variant"])
        for item in matched_items
        if item["source"] == "r04_viewcrop"
    }
    r04_exact_present = any(item["source"] == "r04_exact" for item in matched_items)
    ious = [iou(box, other) for other in representative_boxes]
    overlaps = [overlap_smaller(box, other) for other in representative_boxes]
    areas = [area(other) for other in representative_boxes]
    variants = [str(item["variant"]) for item in representative_items]
    width = box[2] - box[0]
    height = box[3] - box[1]
    centre_x = (box[0] + box[2]) / 2.0
    centre_y = (box[1] + box[3]) / 2.0
    feature_map = {
        "family_fraction": len(families) / 7.0,
        "reliable_family_fraction": len(reliable) / 5.0,
        "sparse_family_fraction": len(sparse) / 4.0,
        "dense_family_fraction": len(dense) / 2.0,
        "strong_family_fraction": strong_count / 7.0,
        "r04_present": float("r04" in families),
        "r04_exact_present": float(r04_exact_present),
        "r04_view_variant_fraction": min(12, len(r04_view_variants)) / 12.0,
        "unique_variant_fraction": min(12, len(all_variants)) / 12.0,
        "full_variant_present": float("full" in all_variants),
        "mean_spatial_support": sum(spatial_scores) / len(spatial_scores),
        "minimum_spatial_support": min(spatial_scores),
        "maximum_spatial_support": max(spatial_scores),
        "mean_normalised_confidence": sum(normalised_confidences)
        / len(normalised_confidences),
        "maximum_normalised_confidence": max(normalised_confidences),
        "minimum_normalised_confidence": min(normalised_confidences),
        "box_source_weight": SOURCE_WEIGHTS.get(str(candidate_item["family"]), 0.70)
        / 1.30,
        "box_source_normalised_confidence": min(
            3.0, _normalised_confidence(candidate_item)
        )
        / 3.0,
        "box_source_production_strong": float(
            candidate_item["confidence"]
            >= PRODUCTION_FLOORS.get(candidate_item["source"], 0.05)
        ),
        "box_mean_iou_to_representatives": sum(ious) / len(ious),
        "box_max_iou_to_representatives": max(ious),
        "box_mean_overlap_smaller": sum(overlaps) / len(overlaps),
        "box_area_fraction": area(box) / 0.24,
        "box_width": width,
        "box_height": height,
        "box_centre_y": centre_y,
        "box_horizontal_edge_proximity": min(centre_x, 1.0 - centre_x) / 0.50,
        "coordinate_dispersion": _coordinate_dispersion(representative_boxes),
        "log_area_ratio": math.log(
            max(areas) / max(min(areas), 1e-9)
        )
        / math.log(100.0),
        "edge_variant_share": sum(variant in EDGE_VARIANTS for variant in variants)
        / len(variants),
        "road_variant_share": sum(variant in ROAD_VARIANTS for variant in variants)
        / len(variants),
    }
    return [feature_map[name] for name in FEATURE_NAMES], feature_map


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    box = record["box"]
    centre_x = (box[0] + box[2]) / 2.0
    centre_y = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        int(record["class_id"]),
        str(record["host_surface"]),
        round(centre_x / 0.025),
        round(centre_y / 0.025),
        round(width / 0.04),
        round(height / 0.04),
    )


def build_candidate_records(
    source_candidates: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _normalise_sources(source_candidates)
    records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    anchor_count = 0
    cluster_count = 0
    for class_id in range(len(EIGHT_CLASS_NAMES)):
        same_class_items = [item for item in items if int(item["class_id"]) == class_id]
        # Every normalised raw proposal can be the candidate box. Dense families
        # still cannot pass alone because the cluster must include a reliable
        # R04/YOLO/Cycle01 family. This preserves a tight RT-DETR or v1.5.1 box
        # when it is more accurate than the reliable anchor that discovered it.
        for anchor in same_class_items:
            anchor_count += 1
            matched_items: list[dict[str, Any]] = []
            for item in same_class_items:
                supported, _spatial = _pair_support(anchor["box"], item["box"])
                if supported:
                    matched_items.append(item)
            representatives = _representatives(anchor, same_class_items)
            families = set(representatives)
            if len(families) < 3:
                continue
            if not families.intersection(RELIABLE_FAMILIES):
                continue
            if not families.intersection(DENSE_FAMILIES):
                continue
            cluster_count += 1
            for family, (candidate_item, spatial_score) in representatives.items():
                box = _professional_box(class_id, candidate_item["box"])
                if box is None:
                    continue
                host_surface, host_diagnostics = _host_surface(
                    class_id, box, representatives
                )
                features, feature_map = _feature_vector(
                    candidate_item,
                    box,
                    representatives,
                    matched_items,
                )
                pre_score = (
                    2.0 * len(families)
                    + 1.2 * len(families.intersection(RELIABLE_FAMILIES))
                    + 0.8 * sum(
                        item["confidence"]
                        >= PRODUCTION_FLOORS.get(item["source"], 0.05)
                        for item, _score in representatives.values()
                    )
                    + min(1.0, spatial_score)
                    - 1.5 * area(box)
                )
                record = {
                    "class_id": class_id,
                    "class_name": EIGHT_CLASS_NAMES[class_id],
                    "host_surface": host_surface,
                    "box": box,
                    "features": features,
                    "feature_map": feature_map,
                    "pre_score": pre_score,
                    "support": {
                        "box_source": candidate_item["source"],
                        "box_family": family,
                        "box_variant": candidate_item["variant"],
                        "families": sorted(families),
                        "host": host_diagnostics,
                    },
                }
                key = _record_key(record)
                current = records_by_key.get(key)
                if current is None or pre_score > current["pre_score"]:
                    records_by_key[key] = record

    records = sorted(
        records_by_key.values(),
        key=lambda item: (item["pre_score"], -area(item["box"])),
        reverse=True,
    )[:MAXIMUM_RECORDS_PER_FRAME]
    return records, {
        "normalised_candidate_count": len(items),
        "anchor_count": anchor_count,
        "cluster_count": cluster_count,
        "candidate_record_count": len(records),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def score_record(record: dict[str, Any], calibrator: dict[str, Any]) -> float:
    if tuple(calibrator["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Candidate02 calibrator feature contract changed.")
    means = [float(value) for value in calibrator["feature_means"]]
    scales = [float(value) for value in calibrator["feature_scales"]]
    weights = [float(value) for value in calibrator["weights"]]
    if not len(record["features"]) == len(means) == len(scales) == len(weights):
        raise RuntimeError("Candidate02 calibrator dimensions are invalid.")
    logit = float(calibrator["bias"])
    for value, mean, scale, weight in zip(
        record["features"], means, scales, weights, strict=True
    ):
        logit += ((float(value) - mean) / max(scale, 1e-6)) * weight
    return _sigmoid(logit)


def _same_problem(first: dict[str, Any], second: dict[str, Any]) -> bool:
    same_class = int(first["class_id"]) == int(second["class_id"])
    same_host = str(first["host_surface"]) == str(second["host_surface"])
    pair_iou = iou(first["box"], second["box"])
    containment = overlap_smaller(first["box"], second["box"])
    if same_class and same_host:
        return pair_iou >= 0.46 or (
            containment >= 0.82
            and centre_distance(first["box"], second["box"]) <= 0.12
        )
    return pair_iou >= 0.72 or (
        containment >= 0.95
        and centre_distance(first["box"], second["box"]) <= 0.07
    )


def select_scored_records(
    scored_records: Iterable[dict[str, Any]],
    *,
    threshold: float,
    maximum_detections: int = 12,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for record in sorted(
        (item for item in scored_records if float(item["verifier_score"]) >= threshold),
        key=lambda item: (
            item["verifier_score"],
            item["pre_score"],
            -area(item["box"]),
        ),
        reverse=True,
    ):
        if any(_same_problem(record, previous) for previous in retained):
            continue
        retained.append(record)
        if len(retained) >= maximum_detections:
            break
    return retained


def fuse_candidate02(
    source_candidates: dict[str, list[dict[str, Any]]],
    calibrator: dict[str, Any],
    *,
    maximum_detections: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, build_diagnostics = build_candidate_records(source_candidates)
    scored: list[dict[str, Any]] = []
    for record in records:
        score = score_record(record, calibrator)
        scored.append({**record, "verifier_score": score})
    threshold = float(calibrator["threshold"])
    selected = select_scored_records(
        scored,
        threshold=threshold,
        maximum_detections=maximum_detections,
    )
    detections = [
        {
            "class_id": int(record["class_id"]),
            "class_name": str(record["class_name"]),
            "host_surface": str(record["host_surface"]),
            "confidence": float(record["verifier_score"]),
            "box": [float(value) for value in record["box"]],
            "strategy": "r09_learned_class_agnostic_proposal_verifier",
            "evidence_score": float(record["pre_score"]),
            "support": record["support"],
        }
        for record in selected
    ]
    return detections, {
        "candidate_model_version": CANDIDATE_MODEL_VERSION,
        "activation_state": "isolated_candidate02_only",
        "parent_model_version": "Stage7-R07-HighRecall-v0.1.19-candidate03",
        "calibrator_identity": calibrator["identity"],
        "calibrator_parent_composite_sha256": calibrator[
            "parent_composite_sha256"
        ],
        "threshold": threshold,
        "training_type": "proposal_metadata_linear_calibration_only",
        "provider_pixels_used_for_fitting": False,
        "openai_used": False,
        "class_identity_used_as_learned_feature": False,
        "fixed_output_quota_used": False,
        "generic_anomaly_output_used": False,
        "sample_coordinate_colour_or_road_name_rules_used": False,
        **build_diagnostics,
        "selected_count": len(detections),
    }
