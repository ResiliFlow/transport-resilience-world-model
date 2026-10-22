from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


EIGHT_CLASS_NAMES = (
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_cracking",
    "uneven_or_broken_footpath",
    "broken_kerb",
    "standing_water",
    "debris_or_vegetation_obstruction",
)
VISIBLE_ANOMALY_CLASS_ID = len(EIGHT_CLASS_NAMES)
VISIBLE_ANOMALY_NAME = "visible_road_anomaly"
CLASS_NAMES = (*EIGHT_CLASS_NAMES, VISIBLE_ANOMALY_NAME)

R04_EXACT = "r04_exact"
R04_VIEWCROP = "r04_viewcrop"
RESEARCH_MODEL = "twm_infradet_v1_5_1"
SPARSE_EXPERTS = {"yolo26n", "yolo26s", "yolo26l", "cycle01"}
ALL_EXPERTS = {*SPARSE_EXPERTS, "rtdetr_l", RESEARCH_MODEL}

PRODUCTION_FLOORS = {
    "yolo26n": 0.05,
    "yolo26s": 0.12,
    "yolo26l": 0.05,
    "rtdetr_l": 0.18,
    "cycle01": 0.05,
    RESEARCH_MODEL: 0.60,
    R04_EXACT: 0.03,
    R04_VIEWCROP: 0.03,
}

RAW_FLOORS = {
    "yolo26n": 0.001,
    "yolo26s": 0.001,
    "yolo26l": 0.001,
    "rtdetr_l": 0.005,
    "cycle01": 0.001,
    RESEARCH_MODEL: 0.55,
    R04_EXACT: 0.001,
    R04_VIEWCROP: 0.001,
}


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def iou(first: list[float], second: list[float]) -> float:
    shared = intersection_area(first, second)
    union = area(first) + area(second) - shared
    return shared / union if union > 0.0 else 0.0


def overlap_smaller(first: list[float], second: list[float]) -> float:
    smaller = min(area(first), area(second))
    return intersection_area(first, second) / smaller if smaller > 0.0 else 0.0


def centre_distance(first: list[float], second: list[float]) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)


def spatial_support(first: list[float], second: list[float]) -> tuple[bool, float]:
    pair_iou = iou(first, second)
    containment = overlap_smaller(first, second)
    distance = centre_distance(first, second)
    supported = pair_iou >= 0.08 or (containment >= 0.55 and distance <= 0.30)
    return supported, max(pair_iou, 0.72 * containment) if supported else 0.0


def _valid_road_geometry(box: list[float]) -> bool:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        return False
    if not 0.0 <= box[0] < box[2] <= 1.0 or not 0.0 <= box[1] < box[3] <= 1.0:
        return False
    box_area = area(box)
    width = box[2] - box[0]
    height = box[3] - box[1]
    centre_y = (box[1] + box[3]) / 2.0
    if not 0.003 <= box_area <= 0.58:
        return False
    if width >= 0.96 and height >= 0.74:
        return False
    return box[3] >= 0.72 and centre_y >= 0.48


def _localise_to_visible_road_band(box: list[float]) -> list[float] | None:
    """Tighten a supported proposal to the visible road/road-edge band.

    Street-view road defects can occur at the left or right edge, so this is not
    a central-lane mask.  It only removes the upper scene content and caps very
    broad anchors that otherwise turn a valid lower-image proposal into a box
    spanning sky, trees or buildings.  It uses normalised geometry only.
    """

    x1, y1, x2, y2 = box
    y1 = max(y1, 0.42, y2 - 0.48)
    if y2 - y1 < 0.035:
        return None
    centre_x = (x1 + x2) / 2.0
    if x2 - x1 > 0.58:
        x1 = max(0.0, centre_x - 0.29)
        x2 = min(1.0, centre_x + 0.29)
    tightened = [x1, y1, x2, y2]
    if area(tightened) > 0.26:
        target_width = min(x2 - x1, 0.26 / max(y2 - y1, 1e-9))
        x1 = max(0.0, centre_x - target_width / 2.0)
        x2 = min(1.0, centre_x + target_width / 2.0)
        tightened = [x1, y1, x2, y2]
    if area(tightened) < 0.003 or tightened[3] < 0.72:
        return None
    return tightened


def _family(source: str) -> str:
    return "r04" if source in {R04_EXACT, R04_VIEWCROP} else source


def _origin(source: str) -> str:
    if source == R04_EXACT:
        return "exact_full"
    if source == R04_VIEWCROP:
        return "view_crop"
    return "heterogeneous_expert"


def _normalised_confidence(item: dict[str, Any]) -> float:
    floor = max(PRODUCTION_FLOORS.get(item["source"], 0.05), 1e-6)
    return min(3.0, float(item["confidence"]) / floor)


def _normalise_sources(source_candidates: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for source, raw_items in source_candidates.items():
        if source not in RAW_FLOORS:
            continue
        floor = RAW_FLOORS[source]
        per_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in raw_items or []:
            try:
                box = [float(value) for value in raw["box"]]
                confidence = float(raw["confidence"])
                class_id = int(raw["class_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= class_id < len(EIGHT_CLASS_NAMES):
                continue
            if confidence < floor or not _valid_road_geometry(box):
                continue
            item = {
                "source": source,
                "family": _family(source),
                "origin": _origin(source),
                "class_id": class_id,
                "confidence": confidence,
                "box": box,
                "variant": str(raw.get("variant") or "full"),
            }
            per_variant[item["variant"]].append(item)

        # These limits retain every recorded low-threshold proposal for the dense
        # 100-per-view components while preventing malformed evidence from causing
        # unbounded quadratic work.
        per_variant_limit = 120 if source in {"rtdetr_l", RESEARCH_MODEL} else 80
        for items in per_variant.values():
            normalised.extend(
                sorted(items, key=lambda value: value["confidence"], reverse=True)[
                    :per_variant_limit
                ]
            )
    return normalised


def _anchor_priority(item: dict[str, Any]) -> float:
    source = item["source"]
    base = 4.0 if source == R04_EXACT else 3.4 if source == R04_VIEWCROP else 2.3
    if source in {"rtdetr_l", RESEARCH_MODEL}:
        base = 1.8
    return base + 0.35 * _normalised_confidence(item) - 0.25 * max(0.0, area(item["box"]) - 0.42)


def _anchor_pool(items: list[dict[str, Any]], max_anchors: int) -> list[dict[str, Any]]:
    raw_anchors = []
    for item in items:
        source = item["source"]
        if source in {R04_EXACT, R04_VIEWCROP} or source in SPARSE_EXPERTS:
            raw_anchors.append(item)
        elif source == "rtdetr_l" and item["confidence"] >= PRODUCTION_FLOORS[source]:
            raw_anchors.append(item)
        elif source == RESEARCH_MODEL and item["confidence"] >= PRODUCTION_FLOORS[source]:
            raw_anchors.append(item)

    kept: list[dict[str, Any]] = []
    for candidate in sorted(raw_anchors, key=_anchor_priority, reverse=True):
        duplicate = False
        for previous in kept:
            pair_iou = iou(candidate["box"], previous["box"])
            containment = overlap_smaller(candidate["box"], previous["box"])
            ratio = max(area(candidate["box"]), area(previous["box"])) / max(
                min(area(candidate["box"]), area(previous["box"])), 1e-9
            )
            if pair_iou >= 0.90 or (containment >= 0.985 and ratio <= 1.35):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= max_anchors:
            break
    return kept


def _best_by_family(matches: list[tuple[dict[str, Any], float]]) -> dict[str, tuple[dict[str, Any], float]]:
    representatives: dict[str, tuple[dict[str, Any], float]] = {}
    for item, spatial in matches:
        value = (spatial + 0.08 * _normalised_confidence(item), item["confidence"])
        current = representatives.get(item["family"])
        if current is None:
            representatives[item["family"]] = (item, spatial)
            continue
        current_item, current_spatial = current
        current_value = (
            current_spatial + 0.08 * _normalised_confidence(current_item),
            current_item["confidence"],
        )
        if value > current_value:
            representatives[item["family"]] = (item, spatial)
    return representatives


def _label(
    representatives: dict[str, tuple[dict[str, Any], float]],
) -> tuple[int, dict[str, Any]]:
    family_weights = {
        "r04": 1.25,
        "yolo26n": 1.0,
        "yolo26s": 1.0,
        "yolo26l": 1.05,
        "rtdetr_l": 0.90,
        "cycle01": 1.0,
        RESEARCH_MODEL: 0.72,
    }
    votes: dict[int, float] = defaultdict(float)
    voters: dict[int, list[str]] = defaultdict(list)
    for family, (item, spatial) in representatives.items():
        weight = family_weights.get(family, 0.75) * (0.45 + 0.55 * spatial)
        weight *= 0.70 + 0.30 * min(1.5, _normalised_confidence(item))
        votes[item["class_id"]] += weight
        voters[item["class_id"]].append(family)
    if not votes:
        return VISIBLE_ANOMALY_CLASS_ID, {"votes": {}, "reason": "no_stable_eight_class_vote"}
    winning_class, winning_weight = max(votes.items(), key=lambda value: value[1])
    total = sum(votes.values())
    stable = winning_weight / max(total, 1e-9) >= 0.55 and len(set(voters[winning_class])) >= 2
    return (
        winning_class if stable else VISIBLE_ANOMALY_CLASS_ID,
        {
            "votes": {str(key): value for key, value in sorted(votes.items())},
            "voters": {str(key): sorted(value) for key, value in sorted(voters.items())},
            "winning_class": winning_class,
            "winning_fraction": winning_weight / max(total, 1e-9),
            "reason": "stable_eight_class_vote" if stable else "visible_location_with_unstable_class_vote",
        },
    )


def _proposal_for_anchor(
    anchor: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches: list[tuple[dict[str, Any], float]] = []
    for item in items:
        supported, spatial = spatial_support(anchor["box"], item["box"])
        if supported:
            matches.append((item, spatial))
    representatives = _best_by_family(matches)
    families = set(representatives)
    expert_families = families.intersection(ALL_EXPERTS)
    matched_items = [item for item, _spatial in matches]
    exact_count = sum(item["source"] == R04_EXACT for item in matched_items)
    r04_variants = {
        item["variant"] for item in matched_items if item["source"] == R04_VIEWCROP
    }
    strong_families = {
        family
        for family, (item, _spatial) in representatives.items()
        if item["confidence"] >= PRODUCTION_FLOORS.get(item["source"], 0.05)
    }
    v151_variants = {
        item["variant"] for item in matched_items if item["source"] == RESEARCH_MODEL
    }
    rtdetr_variants = {
        item["variant"] for item in matched_items if item["source"] == "rtdetr_l"
    }
    has_r04_exact = exact_count > 0
    has_r04_view = bool(r04_variants)
    has_dense_pair = RESEARCH_MODEL in families and "rtdetr_l" in families

    accepted = (
        (has_r04_exact and len(expert_families) >= 1)
        or (len(r04_variants) >= 2 and len(expert_families) >= 1)
        or len(r04_variants) >= 4
        or (
            has_dense_pair
            and len(v151_variants) >= 2
            and len(rtdetr_variants) >= 2
            and len(families) >= 2
        )
        or (len(expert_families) >= 4 and len(strong_families) >= 1)
    )
    if not accepted:
        return None

    output_box = _localise_to_visible_road_band(anchor["box"])
    if output_box is None:
        return None

    family_spatial = sum(spatial for _item, spatial in representatives.values())
    family_count = len(families)
    expert_count = len(expert_families)
    coherence = family_spatial / max(family_count, 1)
    score = (
        2.9 * float(has_r04_exact)
        + 0.72 * min(len(r04_variants), 8)
        + 0.86 * min(expert_count, 6)
        + 0.42 * min(len(strong_families), 5)
        + 0.70 * coherence
        + 0.26 * _normalised_confidence(anchor)
        + 0.32 * float("rtdetr_l" in families)
        + 0.24 * float(RESEARCH_MODEL in families)
        + 0.20 * float(anchor["source"] in {R04_EXACT, R04_VIEWCROP})
        - 0.55 * max(0.0, area(anchor["box"]) - 0.42) / 0.16
    )
    class_id, label_diagnostics = _label(representatives)
    confidence = min(
        0.96,
        0.36
        + 0.045 * min(family_count, 7)
        + 0.025 * min(len(r04_variants), 8)
        + 0.035 * min(len(strong_families), 5)
        + 0.04 * min(coherence, 1.0),
    )
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": confidence,
        "box": output_box,
        "strategy": (
            "r07_stable_eight_class_after_spatial_fusion"
            if class_id != VISIBLE_ANOMALY_CLASS_ID
            else "r07_visible_road_anomaly_spatial_fallback"
        ),
        "evidence_score": score,
        "support": {
            "anchor_source": anchor["source"],
            "anchor_variant": anchor["variant"],
            "anchor_class_id": anchor["class_id"],
            "families": sorted(families),
            "expert_family_count": expert_count,
            "strong_families": sorted(strong_families),
            "r04_exact_support_count": exact_count,
            "r04_view_variants": sorted(r04_variants),
            "rtdetr_variants": sorted(rtdetr_variants),
            "v151_variants": sorted(v151_variants),
            "coherence": coherence,
            "label": label_diagnostics,
        },
    }


def _same_spatial_problem(first: dict[str, Any], second: dict[str, Any]) -> bool:
    pair_iou = iou(first["box"], second["box"])
    containment = overlap_smaller(first["box"], second["box"])
    first_area = area(first["box"])
    second_area = area(second["box"])
    ratio = max(first_area, second_area) / max(min(first_area, second_area), 1e-9)
    return pair_iou >= 0.76 or (
        containment >= 0.975 and ratio <= 2.5 and centre_distance(first["box"], second["box"]) <= 0.16
    )


def _deduplicate(
    proposals: Iterable[dict[str, Any]],
    *,
    max_detections: int,
) -> list[dict[str, Any]]:
    spatial_clusters: list[dict[str, Any]] = []
    for proposal in sorted(
        proposals,
        key=lambda value: (value["evidence_score"], value["confidence"]),
        reverse=True,
    ):
        if any(
            _same_spatial_problem(proposal, previous)
            for previous in spatial_clusters
        ):
            continue
        spatial_clusters.append(proposal)

    # High-recall selection is spatially balanced after category-agnostic fusion.
    # This prevents a dense group of high-scoring centre-road proposals from
    # consuming every output slot while a genuine kerb, footpath or obstruction at
    # either road edge is silently discarded. These are image-box coordinates only;
    # no sample, geographic coordinate, road name or colour is available here.
    candidates = [
        proposal
        for proposal in spatial_clusters
        if 0.008 <= area(proposal["box"]) <= 0.26
    ]
    selected: list[dict[str, Any]] = []

    def add_best(pool: list[dict[str, Any]]) -> None:
        for candidate in sorted(
            pool,
            key=lambda value: (value["evidence_score"], value["confidence"]),
            reverse=True,
        ):
            if candidate not in selected:
                selected.append(candidate)
                return

    add_best(candidates)
    add_best(
        [
            candidate
            for candidate in candidates
            if candidate["box"][0] <= 0.12
            and (candidate["box"][1] + candidate["box"][3]) / 2.0 >= 0.64
        ]
    )
    add_best(
        [
            candidate
            for candidate in candidates
            if candidate["box"][2] >= 0.88
            and (candidate["box"][1] + candidate["box"][3]) / 2.0 >= 0.64
        ]
    )
    add_best(
        [
            candidate
            for candidate in candidates
            if (candidate["box"][1] + candidate["box"][3]) / 2.0 >= 0.78
        ]
    )
    while len(selected) < max_detections:
        remaining = [candidate for candidate in candidates if candidate not in selected]
        if not remaining:
            break
        maximum_score = max(candidate["evidence_score"] for candidate in remaining)
        candidate = max(
            remaining,
            key=lambda value: (
                0.55 * value["evidence_score"] / max(maximum_score, 1e-9)
                + min(
                    centre_distance(value["box"], previous["box"])
                    for previous in selected
                )
            ),
        )
        selected.append(candidate)
    return selected[:max_detections]


def fuse_r07(
    source_candidates: dict[str, list[dict[str, Any]]],
    baseline_predictions: list[dict[str, Any]] | None = None,
    *,
    max_detections: int = 12,
    max_anchors: int = 720,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse existing proposal sources by location before assigning a class.

    The function accepts no sample identifier, coordinate, colour or road name.
    It never treats the dense research detector as an unguarded source. Every
    retained location must have either an R04 exact/view-crop anchor or spatial
    agreement across heterogeneous expert families.
    """

    items = _normalise_sources(source_candidates)
    anchors = _anchor_pool(items, max_anchors=max_anchors)
    proposals = [
        proposal
        for anchor in anchors
        if (proposal := _proposal_for_anchor(anchor, items)) is not None
    ]
    for baseline in baseline_predictions or []:
        try:
            box = [float(value) for value in baseline["box"]]
            class_id = int(baseline["class_id"])
            confidence = float(baseline["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= class_id < len(EIGHT_CLASS_NAMES) or not _valid_road_geometry(box):
            continue
        proposals.append(
            {
                **baseline,
                "box": box,
                "class_id": class_id,
                "class_name": EIGHT_CLASS_NAMES[class_id],
                "confidence": confidence,
                "strategy": str(baseline.get("strategy") or "v017_preserved_expert_output"),
                "evidence_score": float(baseline.get("evidence_score") or 8.0),
                "support": baseline.get("support") or {"source": "v017_preserved_output"},
            }
        )
    retained = _deduplicate(proposals, max_detections=max_detections)
    return retained, {
        "candidate_model_version": "Stage7-R07-HighRecall-v0.1.19-candidate03",
        "activation_state": "isolated_development_only",
        "strategy": "category_agnostic_multi_origin_spatial_anchor_then_balanced_selection_then_class",
        "visible_road_anomaly_class_id": VISIBLE_ANOMALY_CLASS_ID,
        "visible_road_anomaly_name": VISIBLE_ANOMALY_NAME,
        "training_used": False,
        "sample_coordinate_colour_or_road_name_rules_used": False,
        "normalised_candidate_count": len(items),
        "anchor_count": len(anchors),
        "accepted_before_global_nms": len(proposals),
        "retained_after_global_nms": len(retained),
        "max_detections": max_detections,
    }
