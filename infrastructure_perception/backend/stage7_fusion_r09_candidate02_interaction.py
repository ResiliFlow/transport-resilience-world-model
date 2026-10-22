from __future__ import annotations

import math
from typing import Any

from infrastructure_perception.backend.stage7_fusion_r09_candidate02 import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    build_candidate_records,
    select_scored_records,
)


CANDIDATE_MODEL_VERSION = "Stage7-R09-InteractionProposalVerifier-v0.2.2-candidate02"
CALIBRATION_SEED = 20260916
INTERACTION_FEATURE_NAMES = tuple(BASE_FEATURE_NAMES) + tuple(
    f"{BASE_FEATURE_NAMES[first]}*{BASE_FEATURE_NAMES[second]}"
    for first in range(len(BASE_FEATURE_NAMES))
    for second in range(first, len(BASE_FEATURE_NAMES))
)


def expand_standardised_features(values: list[float]) -> list[float]:
    if len(values) != len(BASE_FEATURE_NAMES):
        raise RuntimeError("Candidate02 interaction base-feature dimensions are invalid.")
    expanded = [float(value) for value in values]
    expanded.extend(
        float(values[first]) * float(values[second])
        for first in range(len(values))
        for second in range(first, len(values))
    )
    if len(expanded) != len(INTERACTION_FEATURE_NAMES):
        raise RuntimeError("Candidate02 interaction expansion dimensions are invalid.")
    return expanded


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def score_interaction_record(
    record: dict[str, Any], calibrator: dict[str, Any]
) -> float:
    if tuple(calibrator["base_feature_names"]) != tuple(BASE_FEATURE_NAMES):
        raise RuntimeError("Candidate02 interaction base-feature contract changed.")
    if tuple(calibrator["interaction_feature_names"]) != INTERACTION_FEATURE_NAMES:
        raise RuntimeError("Candidate02 interaction feature contract changed.")
    base_means = [float(value) for value in calibrator["base_feature_means"]]
    base_scales = [float(value) for value in calibrator["base_feature_scales"]]
    base_values = [float(value) for value in record["features"]]
    if not len(base_values) == len(base_means) == len(base_scales):
        raise RuntimeError("Candidate02 interaction base standardisation is invalid.")
    base_standardised = [
        (value - mean) / max(scale, 1e-6)
        for value, mean, scale in zip(
            base_values, base_means, base_scales, strict=True
        )
    ]
    interaction_values = expand_standardised_features(base_standardised)
    interaction_means = [
        float(value) for value in calibrator["interaction_feature_means"]
    ]
    interaction_scales = [
        float(value) for value in calibrator["interaction_feature_scales"]
    ]
    weights = [float(value) for value in calibrator["weights"]]
    if not (
        len(interaction_values)
        == len(interaction_means)
        == len(interaction_scales)
        == len(weights)
    ):
        raise RuntimeError("Candidate02 interaction calibrator dimensions are invalid.")
    logit = float(calibrator["bias"])
    for value, mean, scale, weight in zip(
        interaction_values,
        interaction_means,
        interaction_scales,
        weights,
        strict=True,
    ):
        logit += ((value - mean) / max(scale, 1e-6)) * weight
    return _sigmoid(logit)


def fuse_candidate02_interaction(
    source_candidates: dict[str, list[dict[str, Any]]],
    calibrator: dict[str, Any],
    *,
    maximum_detections: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, build_diagnostics = build_candidate_records(source_candidates)
    scored = [
        {
            **record,
            "verifier_score": score_interaction_record(record, calibrator),
        }
        for record in records
    ]
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
            "strategy": "r09_class_agnostic_interaction_proposal_verifier",
            "evidence_score": float(record["pre_score"]),
            "support": record["support"],
        }
        for record in selected
    ]
    return detections, {
        "candidate_model_version": CANDIDATE_MODEL_VERSION,
        "activation_state": "isolated_candidate02_interaction_only",
        "parent_model_version": "Stage7-R07-HighRecall-v0.1.19-candidate03",
        "calibrator_identity": calibrator["identity"],
        "calibrator_parent_composite_sha256": calibrator[
            "parent_composite_sha256"
        ],
        "threshold": threshold,
        "training_type": "proposal_metadata_interaction_calibration_only",
        "provider_pixels_used_for_fitting": False,
        "openai_used": False,
        "class_identity_used_as_learned_feature": False,
        "fixed_output_quota_used": False,
        "generic_anomaly_output_used": False,
        "sample_coordinate_colour_or_road_name_rules_used": False,
        **build_diagnostics,
        "selected_count": len(detections),
    }
