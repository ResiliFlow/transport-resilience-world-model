from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from infrastructure_perception.backend.stage7_fusion_r07 import (
    ALL_EXPERTS,
    EIGHT_CLASS_NAMES,
    PRODUCTION_FLOORS,
    R04_EXACT,
    R04_VIEWCROP,
    RESEARCH_MODEL,
    SPARSE_EXPERTS,
    _normalise_sources,
    _normalised_confidence,
    area,
    centre_distance,
    iou,
    overlap_smaller,
)


CANDIDATE_MODEL_VERSION = "Stage7-R08-ClassAwareTightFusion-v0.2.0-candidate01"
RELIABLE_FAMILIES = {"r04", *SPARSE_EXPERTS}
DENSE_FAMILIES = {"rtdetr_l", RESEARCH_MODEL}


@dataclass(frozen=True)
class FusionConfig:
    config_id: str
    minimum_families: int
    minimum_reliable_families: int
    minimum_standing_water_families: int
    minimum_strong_families: int = 1
    pair_iou: float = 0.18
    pair_overlap_smaller: float = 0.72
    pair_centre_distance: float = 0.18
    require_r04: bool = False
    maximum_anchors_per_class: int = 180
    maximum_detections: int = 12


# This compact development grid is frozen before Candidate01 Random50 results.
# It varies only general consensus strictness; no sample, coordinate, road name,
# colour or target-specific values are available to the fusion function.
DEVELOPMENT_CONFIGS = (
    FusionConfig("balanced_f4_r2_w4", 4, 2, 4),
    FusionConfig("balanced_r04_f4_r2_w4", 4, 2, 4, require_r04=True),
    FusionConfig("strict_f5_r2_w5", 5, 2, 5),
    FusionConfig("strict_r04_f5_r2_w5", 5, 2, 5, require_r04=True),
    FusionConfig("precision_f5_r3_w5", 5, 3, 5),
    FusionConfig(
        "precision_r04_f5_r3_w6",
        5,
        3,
        6,
        pair_iou=0.20,
        pair_overlap_smaller=0.76,
        pair_centre_distance=0.16,
        require_r04=True,
    ),
)


SOURCE_WEIGHTS = {
    "r04": 1.30,
    "yolo26n": 1.00,
    "yolo26s": 1.05,
    "yolo26l": 1.10,
    "cycle01": 1.00,
    "rtdetr_l": 0.90,
    RESEARCH_MODEL: 0.78,
}

CLASS_MAXIMUM_AREAS = {
    0: 0.12,
    1: 0.10,
    2: 0.10,
    3: 0.14,
    4: 0.15,
    5: 0.10,
    6: 0.22,
    7: 0.24,
}

ROAD_VARIANTS = {"road_centre", "road_left", "road_right", "crack_mid"}
EDGE_VARIANTS = {"side_edge_left", "side_edge_right"}


def _pair_support(
    first: list[float],
    second: list[float],
    config: FusionConfig,
) -> tuple[bool, float]:
    pair_iou = iou(first, second)
    containment = overlap_smaller(first, second)
    distance = centre_distance(first, second)
    first_area = area(first)
    second_area = area(second)
    ratio = max(first_area, second_area) / max(min(first_area, second_area), 1e-9)
    supported = pair_iou >= config.pair_iou or (
        containment >= config.pair_overlap_smaller
        and distance <= config.pair_centre_distance
        and ratio <= 4.0
    )
    score = max(pair_iou, 0.74 * containment) if supported else 0.0
    return supported, score


def _anchor_priority(item: dict[str, Any]) -> tuple[float, float, float]:
    family = str(item["family"])
    family_weight = SOURCE_WEIGHTS.get(family, 0.70)
    production_strength = min(2.0, _normalised_confidence(item))
    compactness = -max(0.0, area(item["box"]) - 0.18)
    return family_weight + 0.22 * production_strength, compactness, float(item["confidence"])


def _anchor_pool(
    items: list[dict[str, Any]],
    class_id: int,
    config: FusionConfig,
) -> list[dict[str, Any]]:
    # Dense RT-DETR and v1.5.1 boxes can corroborate a location, but they never
    # create a location on their own. This removes the R07 dense-noise path.
    candidates = [
        item
        for item in items
        if item["class_id"] == class_id and item["family"] in RELIABLE_FAMILIES
    ]
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=_anchor_priority, reverse=True):
        if any(
            iou(candidate["box"], previous["box"]) >= 0.90
            or (
                overlap_smaller(candidate["box"], previous["box"]) >= 0.97
                and centre_distance(candidate["box"], previous["box"]) <= 0.05
            )
            for previous in retained
        ):
            continue
        retained.append(candidate)
        if len(retained) >= config.maximum_anchors_per_class:
            break
    return retained


def _best_by_family(
    anchor: dict[str, Any],
    matches: list[tuple[dict[str, Any], float]],
) -> dict[str, tuple[dict[str, Any], float]]:
    representatives: dict[str, tuple[dict[str, Any], float]] = {}
    for item, spatial in matches:
        family = str(item["family"])
        score = (
            spatial
            + 0.08 * min(2.0, _normalised_confidence(item))
            + 0.04 * SOURCE_WEIGHTS.get(family, 0.70)
            - 0.04 * max(0.0, area(item["box"]) - area(anchor["box"]))
        )
        current = representatives.get(family)
        if current is None:
            representatives[family] = (item, score)
            continue
        if score > current[1]:
            representatives[family] = (item, score)
    return representatives


def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _value, weight in ordered)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2.0:
            return value
    return ordered[-1][0]


def _cap_box_area(box: list[float], maximum_area: float) -> list[float]:
    current_area = area(box)
    if current_area <= maximum_area:
        return box
    scale = math.sqrt(maximum_area / max(current_area, 1e-9))
    centre_x = (box[0] + box[2]) / 2.0
    centre_y = (box[1] + box[3]) / 2.0
    half_width = (box[2] - box[0]) * scale / 2.0
    half_height = (box[3] - box[1]) * scale / 2.0
    x1 = max(0.0, centre_x - half_width)
    x2 = min(1.0, centre_x + half_width)
    y1 = max(0.0, centre_y - half_height)
    y2 = min(1.0, centre_y + half_height)
    return [x1, y1, x2, y2]


def _tight_consensus_box(
    class_id: int,
    representatives: dict[str, tuple[dict[str, Any], float]],
) -> list[float] | None:
    boxes = [item["box"] for item, _score in representatives.values()]
    if not boxes:
        return None
    sorted_areas = sorted(area(box) for box in boxes)
    median_area = sorted_areas[len(sorted_areas) // 2]
    usable = [
        (family, item, score)
        for family, (item, score) in representatives.items()
        if 0.30 * median_area <= area(item["box"]) <= 2.80 * median_area
    ]
    if len(usable) < 3:
        usable = [
            (family, item, score)
            for family, (item, score) in representatives.items()
        ]

    coordinates: list[float] = []
    for coordinate_index in range(4):
        values: list[tuple[float, float]] = []
        for family, item, score in usable:
            weight = SOURCE_WEIGHTS.get(family, 0.70) * (0.65 + 0.35 * min(score, 1.0))
            values.append((float(item["box"][coordinate_index]), weight))
        coordinates.append(_weighted_median(values))

    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    width = x2 - x1
    height = y2 - y1
    # A small percentage pad preserves the visible boundary while avoiding the
    # broad union boxes that made R07 look unprofessional.
    x_padding = min(0.012, 0.035 * width)
    y_padding = min(0.012, 0.035 * height)
    box = [
        max(0.0, x1 - x_padding),
        max(0.42, y1 - y_padding),
        min(1.0, x2 + x_padding),
        min(1.0, y2 + y_padding),
    ]
    box = _cap_box_area(box, CLASS_MAXIMUM_AREAS[class_id])
    if area(box) < 0.0015:
        return None
    width = box[2] - box[0]
    height = box[3] - box[1]
    centre_y = (box[1] + box[3]) / 2.0
    if box[3] < 0.70 or centre_y < 0.55:
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
    # Broad edge obstructions frequently occupy the road shoulder. Only a tight,
    # extreme-edge consensus is classified as footpath for multi-host classes.
    host = (
        "footpath"
        if extreme_edge and width <= 0.30 and edge_votes >= max(2, road_votes)
        else "road"
    )
    return host, {
        "rule": "multi_host_geometry_and_crop_consensus",
        "edge_votes": edge_votes,
        "road_votes": road_votes,
        "extreme_edge": extreme_edge,
    }


def _proposal_for_anchor(
    anchor: dict[str, Any],
    same_class_items: list[dict[str, Any]],
    config: FusionConfig,
) -> dict[str, Any] | None:
    matches: list[tuple[dict[str, Any], float]] = []
    for item in same_class_items:
        supported, spatial = _pair_support(anchor["box"], item["box"], config)
        if supported:
            matches.append((item, spatial))
    representatives = _best_by_family(anchor, matches)
    families = set(representatives)
    reliable_families = families.intersection(RELIABLE_FAMILIES)
    strong_families = {
        family
        for family, (item, _score) in representatives.items()
        if item["confidence"] >= PRODUCTION_FLOORS.get(item["source"], 0.05)
    }
    class_id = int(anchor["class_id"])
    minimum_families = (
        config.minimum_standing_water_families
        if class_id == 6
        else config.minimum_families
    )
    if len(families) < minimum_families:
        return None
    if len(reliable_families) < config.minimum_reliable_families:
        return None
    if len(strong_families) < config.minimum_strong_families:
        return None
    if config.require_r04 and "r04" not in families:
        return None
    if not families.intersection(DENSE_FAMILIES):
        return None

    box = _tight_consensus_box(class_id, representatives)
    if box is None:
        return None
    # Footpath and kerb defects must be spatially edge-supported, not promoted
    # from a central carriageway response.
    centre_x = (box[0] + box[2]) / 2.0
    variants = [str(item["variant"]) for item, _score in representatives.values()]
    edge_votes = sum(variant in EDGE_VARIANTS for variant in variants)
    if class_id in {4, 5} and 0.32 < centre_x < 0.68 and edge_votes < 2:
        return None

    host_surface, host_diagnostics = _host_surface(class_id, box, representatives)
    coherence = sum(min(1.0, score) for _item, score in representatives.values()) / len(
        representatives
    )
    evidence_score = (
        1.55 * len(families)
        + 1.10 * len(reliable_families)
        + 0.65 * len(strong_families)
        + 0.80 * float("r04" in families)
        + 0.75 * coherence
        - 1.30 * area(box)
    )
    confidence = min(
        0.96,
        0.40
        + 0.055 * len(families)
        + 0.035 * len(reliable_families)
        + 0.025 * len(strong_families)
        + 0.045 * coherence,
    )
    return {
        "class_id": class_id,
        "class_name": EIGHT_CLASS_NAMES[class_id],
        "host_surface": host_surface,
        "confidence": confidence,
        "box": box,
        "strategy": "r08_same_class_heterogeneous_tight_consensus",
        "evidence_score": evidence_score,
        "support": {
            "anchor_source": anchor["source"],
            "anchor_variant": anchor["variant"],
            "families": sorted(families),
            "reliable_families": sorted(reliable_families),
            "strong_families": sorted(strong_families),
            "coherence": coherence,
            "host": host_diagnostics,
        },
    }


def _same_problem(first: dict[str, Any], second: dict[str, Any]) -> bool:
    same_class = int(first["class_id"]) == int(second["class_id"])
    pair_iou = iou(first["box"], second["box"])
    containment = overlap_smaller(first["box"], second["box"])
    if same_class:
        return pair_iou >= 0.48 or (
            containment >= 0.84
            and centre_distance(first["box"], second["box"]) <= 0.12
        )
    return pair_iou >= 0.68 or (
        containment >= 0.93
        and centre_distance(first["box"], second["box"]) <= 0.08
    )


def _deduplicate(
    proposals: Iterable[dict[str, Any]],
    maximum_detections: int,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (item["evidence_score"], item["confidence"], -area(item["box"])),
        reverse=True,
    ):
        if any(_same_problem(proposal, previous) for previous in retained):
            continue
        retained.append(proposal)
        if len(retained) >= maximum_detections:
            break
    return retained


def fuse_candidate01(
    source_candidates: dict[str, list[dict[str, Any]]],
    *,
    config: FusionConfig = DEVELOPMENT_CONFIGS[0],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only class-stable, host-eligible, tight heterogeneous consensus.

    The function has no sample identifier, coordinates, road name, colour, image
    pixels, ground truth or OpenAI input. It cannot force a detection count and
    permits an empty result when no supported eight-class issue is present.
    """

    items = _normalise_sources(source_candidates)
    proposals: list[dict[str, Any]] = []
    anchor_count = 0
    for class_id in range(len(EIGHT_CLASS_NAMES)):
        same_class_items = [item for item in items if item["class_id"] == class_id]
        anchors = _anchor_pool(items, class_id, config)
        anchor_count += len(anchors)
        for anchor in anchors:
            proposal = _proposal_for_anchor(anchor, same_class_items, config)
            if proposal is not None:
                proposals.append(proposal)

    retained = _deduplicate(proposals, config.maximum_detections)
    return retained, {
        "candidate_model_version": CANDIDATE_MODEL_VERSION,
        "activation_state": "isolated_random50_development_only",
        "parent_model_version": "Stage7-R07-HighRecall-v0.1.19-candidate03",
        "training_used": False,
        "openai_used": False,
        "fixed_output_quota_used": False,
        "generic_anomaly_output_used": False,
        "sample_coordinate_colour_or_road_name_rules_used": False,
        "config": asdict(config),
        "normalised_candidate_count": len(items),
        "anchor_count": anchor_count,
        "accepted_before_nms": len(proposals),
        "retained_after_nms": len(retained),
    }
