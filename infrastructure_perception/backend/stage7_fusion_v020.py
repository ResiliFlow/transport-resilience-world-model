from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from stage7_fusion_v019 import (
    BASE_MODELS,
    PRODUCTION_FLOORS,
    RESEARCH_MODEL,
    _area,
    _cluster_base_candidates,
    _compatible,
    _localise,
    _matching_research,
    _normalise_candidates,
    _representatives,
    _same_detection,
    _strength,
    _texture_metrics,
    iou,
    overlap_smaller,
)


def _box_centre(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _centre_distance(first: list[float], second: list[float]) -> float:
    first_x, first_y = _box_centre(first)
    second_x, second_y = _box_centre(second)
    return math.hypot(first_x - second_x, first_y - second_y)


def _median_box(items: list[dict[str, Any]]) -> list[float] | None:
    if not items:
        return None
    return [float(np.median([item["box"][coordinate] for item in items])) for coordinate in range(4)]


def _evidence(
    cluster: list[dict[str, Any]],
    representatives: dict[str, dict[str, Any]],
    box: list[float],
    research: dict[str, Any] | None,
) -> dict[str, Any]:
    def source(item: dict[str, Any]) -> str:
        if item["variant"].startswith("exact_tile_"):
            return "exact_tile"
        if item["variant"].startswith("tta_"):
            return "tta"
        if item["variant"].startswith("roi_"):
            return "roi"
        return "original_crop"

    source_counts = {
        source_name: sum(source(item) == source_name for item in cluster)
        for source_name in ("original_crop", "exact_tile", "tta", "roi")
    }
    tile_views = {
        item["variant"]
        for item in cluster
        if item["variant"].startswith("exact_tile_")
    }
    tta_views = {item["variant"] for item in cluster if item["variant"].startswith("tta_")}
    roi_views = {item["variant"] for item in cluster if item["variant"].startswith("roi_")}
    production_models = {
        model_name
        for model_name, item in representatives.items()
        if item["confidence"] >= PRODUCTION_FLOORS[model_name]
    }
    normalised_support = sum(
        min(2.0, item["confidence"] / PRODUCTION_FLOORS[model_name])
        for model_name, item in representatives.items()
    )
    representative_items = list(representatives.values())
    representative_pairs = list(combinations(representative_items, 2))
    representative_ious = [iou(first["box"], second["box"]) for first, second in representative_pairs]
    representative_overlaps = [
        overlap_smaller(first["box"], second["box"])
        for first, second in representative_pairs
    ]
    representative_distances = [
        _centre_distance(first["box"], second["box"])
        for first, second in representative_pairs
    ]
    member_ious = [iou(item["box"], box) for item in cluster]
    member_overlaps = [overlap_smaller(item["box"], box) for item in cluster]
    member_distances = [_centre_distance(item["box"], box) for item in cluster]
    original_items = [item for item in cluster if source(item) == "original_crop"]
    tile_items = [item for item in cluster if source(item) == "exact_tile"]
    tta_items = [item for item in cluster if source(item) == "tta"]
    roi_items = [item for item in cluster if source(item) == "roi"]
    original_box = _median_box(original_items)
    tile_box = _median_box(tile_items)
    models_with_both_sources = {
        model_name
        for model_name in representatives
        if any(item["model"] == model_name for item in original_items)
        and any(item["model"] == model_name for item in tile_items)
    }
    models_with_original_and_tta = {
        model_name
        for model_name in representatives
        if any(item["model"] == model_name for item in original_items)
        and any(item["model"] == model_name for item in tta_items)
    }
    tta_variant_counts_by_model = {
        model_name: len({item["variant"] for item in tta_items if item["model"] == model_name})
        for model_name in representatives
    }
    models_with_original_and_roi = {
        model_name
        for model_name in representatives
        if any(item["model"] == model_name for item in original_items)
        and any(item["model"] == model_name for item in roi_items)
    }
    roi_variant_counts_by_model = {
        model_name: len({item["variant"] for item in roi_items if item["model"] == model_name})
        for model_name in representatives
    }
    tta_box = _median_box(tta_items)
    roi_box = _median_box(roi_items)
    return {
        "model_count": len(representatives),
        "non_rtdetr_model_count": sum(name != "rtdetr_l" for name in representatives),
        "production_model_count": len(production_models),
        "production_models": sorted(production_models),
        "member_count": len(cluster),
        "source_counts": source_counts,
        "tile_view_count": len(tile_views),
        "tta_view_count": len(tta_views),
        "roi_view_count": len(roi_views),
        "normalised_support": normalised_support,
        "models_with_both_sources": len(models_with_both_sources),
        "models_with_original_and_tta": len(models_with_original_and_tta),
        "models_with_tta_2plus": sum(value >= 2 for value in tta_variant_counts_by_model.values()),
        "models_with_tta_3plus": sum(value >= 3 for value in tta_variant_counts_by_model.values()),
        "models_with_tta_4": sum(value >= 4 for value in tta_variant_counts_by_model.values()),
        "models_with_original_and_roi": len(models_with_original_and_roi),
        "models_with_roi_2plus": sum(value >= 2 for value in roi_variant_counts_by_model.values()),
        "models_with_roi_4plus": sum(value >= 4 for value in roi_variant_counts_by_model.values()),
        "models_with_roi_8plus": sum(value >= 8 for value in roi_variant_counts_by_model.values()),
        "representative_pair_count": len(representative_pairs),
        "representative_iou_mean": float(np.mean(representative_ious)) if representative_ious else 1.0,
        "representative_iou_min": min(representative_ious, default=1.0),
        "representative_overlap_mean": float(np.mean(representative_overlaps)) if representative_overlaps else 1.0,
        "representative_overlap_min": min(representative_overlaps, default=1.0),
        "representative_compatible_fraction": (
            sum(_compatible(first["box"], second["box"]) for first, second in representative_pairs)
            / len(representative_pairs)
            if representative_pairs
            else 1.0
        ),
        "representative_centre_distance_mean": float(np.mean(representative_distances)) if representative_distances else 0.0,
        "representative_centre_distance_max": max(representative_distances, default=0.0),
        "member_iou_mean": float(np.mean(member_ious)) if member_ious else 0.0,
        "member_iou_p25": float(np.percentile(member_ious, 25.0)) if member_ious else 0.0,
        "member_overlap_mean": float(np.mean(member_overlaps)) if member_overlaps else 0.0,
        "member_overlap_p25": float(np.percentile(member_overlaps, 25.0)) if member_overlaps else 0.0,
        "member_centre_distance_mean": float(np.mean(member_distances)) if member_distances else 0.0,
        "member_centre_distance_p90": float(np.percentile(member_distances, 90.0)) if member_distances else 0.0,
        "original_tile_iou": iou(original_box, tile_box) if original_box and tile_box else 0.0,
        "original_tile_overlap": overlap_smaller(original_box, tile_box) if original_box and tile_box else 0.0,
        "original_tta_iou": iou(original_box, tta_box) if original_box and tta_box else 0.0,
        "original_tta_overlap": overlap_smaller(original_box, tta_box) if original_box and tta_box else 0.0,
        "tile_tta_iou": iou(tile_box, tta_box) if tile_box and tta_box else 0.0,
        "tile_tta_overlap": overlap_smaller(tile_box, tta_box) if tile_box and tta_box else 0.0,
        "original_roi_iou": iou(original_box, roi_box) if original_box and roi_box else 0.0,
        "original_roi_overlap": overlap_smaller(original_box, roi_box) if original_box and roi_box else 0.0,
        "tile_roi_iou": iou(tile_box, roi_box) if tile_box and roi_box else 0.0,
        "tile_roi_overlap": overlap_smaller(tile_box, roi_box) if tile_box and roi_box else 0.0,
        "tta_roi_iou": iou(tta_box, roi_box) if tta_box and roi_box else 0.0,
        "tta_roi_overlap": overlap_smaller(tta_box, roi_box) if tta_box and roi_box else 0.0,
        "research_iou": iou(research["box"], box) if research else 0.0,
        "research_overlap": overlap_smaller(research["box"], box) if research else 0.0,
    }


def _generic_gate(
    class_id: int,
    evidence: dict[str, Any],
    research: dict[str, Any] | None,
    box: list[float],
    texture: dict[str, float],
) -> tuple[bool, str]:
    models = evidence["model_count"]
    non_rtdetr = evidence["non_rtdetr_model_count"]
    production = evidence["production_model_count"]
    members = evidence["member_count"]
    tiles = evidence["tile_view_count"]
    original = evidence["source_counts"]["original_crop"]
    support = evidence["normalised_support"]
    research_confidence = research["confidence"] if research else 0.0
    area = _area(box)

    if class_id == 0:
        accepted = (
            (models >= 4 and tiles >= 3 and members >= 12)
            or (models >= 3 and production >= 1 and (tiles >= 2 or original >= 6))
            or (models >= 2 and production >= 2 and support >= 2.2)
        ) and area <= 0.32
        return accepted, "generic_multiview_pothole_consensus"
    if class_id == 1:
        accepted = (
            models >= 4
            and non_rtdetr >= 2
            and (tiles >= 3 or original >= 12)
            and area <= 0.18
        )
        return accepted, "generic_multiview_longitudinal_crack_consensus"
    if class_id in {2, 3}:
        accepted = (
            models >= 4
            and non_rtdetr >= 3
            and (production >= 1 or members >= 60)
            and tiles >= 3
            and area <= 0.42
        )
        return accepted, "generic_multifamily_network_crack_consensus"
    if class_id == 4:
        accepted = research is not None and (
            (models >= 3 and tiles >= 2)
            or (models >= 2 and production >= 1)
            or (
                models >= 1
                and tiles >= 3
                and research_confidence >= 0.60
                and area >= 0.035
                and texture["grayscale_entropy"] >= 3.5
            )
        )
        return accepted, "generic_research_verified_footpath_consensus"
    if class_id == 5:
        accepted = research is not None and (
            (models >= 3 and tiles >= 2)
            or (models >= 2 and production >= 1 and tiles >= 1)
        ) and area >= 0.015
        return accepted, "generic_research_verified_kerb_consensus"
    if class_id == 6:
        accepted = research is not None and (
            (models >= 4 and tiles >= 3)
            or (models >= 3 and production >= 1 and tiles >= 2)
            or (models >= 2 and production >= 2 and support >= 2.4)
        ) and area >= 0.012
        return accepted, "generic_research_verified_water_consensus"
    if class_id == 7:
        accepted = research is not None and (
            (models >= 3 and tiles >= 2)
            or (models >= 2 and tiles >= 2 and research_confidence >= 0.60)
            or (
                models >= 1
                and tiles >= 4
                and research_confidence >= 0.64
                and area >= 0.04
                and texture["grayscale_entropy"] >= 4.0
            )
        )
        return accepted, "generic_research_verified_obstruction_consensus"
    return False, "unsupported_class"


def _score(
    class_id: int,
    evidence: dict[str, Any],
    research: dict[str, Any] | None,
    box: list[float],
    texture: dict[str, float],
) -> float:
    research_confidence = research["confidence"] if research else 0.0
    score = (
        0.16 * evidence["model_count"]
        + 0.10 * evidence["production_model_count"]
        + 0.055 * min(evidence["tile_view_count"], 8)
        + 0.045 * min(math.log1p(evidence["member_count"]), 5.0)
        + 0.055 * min(evidence["normalised_support"], 5.0)
        + (0.20 * research_confidence if class_id >= 4 or class_id == 6 else 0.0)
    )
    if class_id in {0, 1, 2, 3, 6}:
        score += 0.04 * max(0.0, min(1.0, (box[3] - 0.50) / 0.50))
    if class_id in {4, 5, 7}:
        score += 0.025 * min(texture["grayscale_entropy"], 5.0)
    return score


def _deduplicate_ranked(items: list[dict[str, Any]], *, per_class_limit: int = 1) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for item in sorted(items, key=lambda value: value["evidence_score"], reverse=True):
        if counts.get(item["class_id"], 0) >= per_class_limit:
            continue
        if any(
            item["class_id"] == previous["class_id"]
            and (iou(item["box"], previous["box"]) >= 0.22 or overlap_smaller(item["box"], previous["box"]) >= 0.78)
            for previous in kept
        ):
            continue
        kept.append(item)
        counts[item["class_id"]] = counts.get(item["class_id"], 0) + 1
    return kept


def fuse_v020(
    raw_models: dict[str, Any],
    baseline_predictions: list[dict[str, Any]],
    image_path: Path,
    *,
    max_detections: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _normalise_candidates(raw_models)
    research_by_class = {
        class_id: [item for item in candidates if item["model"] == RESEARCH_MODEL]
        for class_id, candidates in grouped.items()
    }
    decisions = []
    additions = []
    for class_id in range(8):
        base = [item for item in grouped.get(class_id, []) if item["model"] in BASE_MODELS]
        for cluster in _cluster_base_candidates(base):
            representatives = _representatives(cluster)
            research = _matching_research(class_id, representatives, research_by_class.get(class_id, []))
            box = _localise(representatives, research)
            texture = _texture_metrics(image_path, box)
            evidence = _evidence(cluster, representatives, box, research)
            accepted, reason = _generic_gate(class_id, evidence, research, box, texture)
            evidence_score = _score(class_id, evidence, research, box, texture)
            decision = {
                "class_id": class_id,
                "accepted": accepted,
                "reason": reason,
                "box": box,
                "families": sorted(representatives),
                "family_confidence": {name: item["confidence"] for name, item in representatives.items()},
                "family_variants": {name: item["variant"] for name, item in representatives.items()},
                "research_confidence": research["confidence"] if research else None,
                "research_variant": research["variant"] if research else None,
                "evidence": evidence,
                "evidence_score": evidence_score,
                "texture": texture,
            }
            decisions.append(decision)
            if not accepted:
                continue
            strengths = [_strength(item) for item in representatives.values()]
            if research is not None:
                strengths.append(_strength(research))
            additions.append(
                {
                    "class_id": class_id,
                    "confidence": min(0.95, 0.28 + 0.22 * min(1.0, evidence_score / 2.0) + 0.08 * float(np.mean(strengths))),
                    "box": box,
                    "strategy": reason,
                    "support": decision["family_confidence"],
                    "evidence_score": evidence_score,
                }
            )

    ranked = _deduplicate_ranked(additions, per_class_limit=1)
    retained_baseline = [dict(item) for item in baseline_predictions]
    merged = []
    for candidate in sorted(
        [*retained_baseline, *ranked],
        key=lambda item: (item.get("evidence_score", 0.0), item["confidence"]),
        reverse=True,
    ):
        if any(_same_detection(candidate, previous) for previous in merged):
            continue
        merged.append(candidate)
        if len(merged) >= max_detections:
            break
    return merged, {
        "candidate_model_version": "Stage7-HMT-Consensus-v0.1.20-candidate",
        "activation_state": "isolated_development_only",
        "frame_specific_inputs_used": False,
        "coordinate_rules_used": False,
        "training_used": False,
        "decision_count": len(decisions),
        "accepted_before_rank_count": len(additions),
        "accepted_after_rank_count": len(ranked),
        "decisions": decisions,
    }
