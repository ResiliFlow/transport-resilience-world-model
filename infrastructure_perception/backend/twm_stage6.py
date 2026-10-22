from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = PERCEPTION_ROOT.parent
INTEGRATION_RECORD = (
    PERCEPTION_ROOT / "stage6" / "twm_infradet_v0_1_1" / "integration_record.json"
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
SURFACE_OPTIONS = {
    0: ("road",),
    1: ("road", "footpath"),
    2: ("road", "footpath"),
    3: ("road",),
    4: ("footpath",),
    5: ("kerb",),
    6: ("road", "footpath", "kerb"),
    7: ("road", "footpath", "kerb"),
}
LOCATION_METHODS = {
    "source_metadata",
    "survey_control",
    "network_match",
    "manual_review",
}
EVIDENCE_FIELDS = {
    "source_id",
    "asset_id",
    "captured_at",
    "licence_status",
    "permission_record_id",
    "attribution",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(relative_path: str) -> Path:
    return PLATFORM_ROOT / relative_path


def _verified_file(relative_path: str, expected_sha256: str) -> bool:
    path = _resolve(relative_path)
    return path.is_file() and _sha256(path) == expected_sha256.upper()


def _first_cache_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError("The real-inference cache has no records.")


def _valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
        return False
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    return 0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0


def _normalise_location(location: Any) -> dict[str, Any] | None:
    if not isinstance(location, dict):
        return None
    allowed = {"latitude", "longitude", "method", "accuracy_m"}
    if not {"latitude", "longitude", "method"}.issubset(location):
        return None
    if set(location) - allowed:
        return None
    latitude = location["latitude"]
    longitude = location["longitude"]
    method = location["method"]
    if not isinstance(latitude, (int, float)) or not math.isfinite(latitude):
        return None
    if not isinstance(longitude, (int, float)) or not math.isfinite(longitude):
        return None
    if not -90.0 <= float(latitude) <= 90.0:
        return None
    if not -180.0 <= float(longitude) <= 180.0:
        return None
    if method not in LOCATION_METHODS:
        return None
    result: dict[str, Any] = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "method": method,
    }
    if "accuracy_m" in location:
        accuracy = location["accuracy_m"]
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
            return None
        if float(accuracy) < 0.0:
            return None
        result["accuracy_m"] = float(accuracy)
    return result


def _valid_source_evidence(evidence: Any) -> bool:
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_FIELDS:
        return False
    if evidence["source_id"] not in {
        "licensed_public_dataset",
        "institution_authorised",
        "kartaview_cc_by_sa",
        "google_maps_core",
        "copernicus_sentinel",
        "individual_user_upload",
        "unknown_or_unverified",
    }:
        return False
    if evidence["licence_status"] != "verified":
        return False
    return all(
        isinstance(evidence[field], str) and bool(evidence[field].strip())
        for field in EVIDENCE_FIELDS - {"licence_status"}
    )


def _abstention(
    *,
    reasons: list[str],
    image: str,
    prediction_count: int,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    return {
        "result_type": "abstention",
        "state": "insufficient_evidence",
        "reasons": reasons,
        "image": image,
        "prediction_count": prediction_count,
        "model": {"name": model_name, "version": model_version},
        "review": {"status": "pending"},
        "automatic_operational_action_enabled": False,
    }


def adapt_predictions(
    *,
    image: str,
    predictions: list[dict[str, Any]],
    source_evidence: dict[str, Any],
    location: dict[str, Any] | None,
    surface_type: str | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert detector predictions to contract observations or safely abstain."""

    integration = record or _load_json(INTEGRATION_RECORD)
    model_name = integration["model_name"]
    model_version = integration["model_version"]
    normalised_location = _normalise_location(location)
    if normalised_location is None:
        return _abstention(
            reasons=["missing_verified_point_location"],
            image=image,
            prediction_count=len(predictions),
            model_name=model_name,
            model_version=model_version,
        )
    if not _valid_source_evidence(source_evidence):
        return _abstention(
            reasons=["invalid_source_evidence"],
            image=image,
            prediction_count=len(predictions),
            model_name=model_name,
            model_version=model_version,
        )

    threshold = float(integration["adapter_policy"]["minimum_detection_confidence"])
    retained: list[tuple[int, float, int, list[float], str]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            return _abstention(
                reasons=["invalid_prediction_payload"],
                image=image,
                prediction_count=len(predictions),
                model_name=model_name,
                model_version=model_version,
            )
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            return _abstention(
                reasons=["invalid_prediction_payload"],
                image=image,
                prediction_count=len(predictions),
                model_name=model_name,
                model_version=model_version,
            )
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            return _abstention(
                reasons=["invalid_prediction_payload"],
                image=image,
                prediction_count=len(predictions),
                model_name=model_name,
                model_version=model_version,
            )
        if confidence < threshold:
            continue
        class_id = prediction.get("class_id")
        box = prediction.get("box")
        if not isinstance(class_id, int) or class_id not in SURFACE_OPTIONS or not _valid_box(box):
            return _abstention(
                reasons=["invalid_prediction_payload"],
                image=image,
                prediction_count=len(predictions),
                model_name=model_name,
                model_version=model_version,
            )
        allowed_surfaces = SURFACE_OPTIONS[class_id]
        resolved_surface = prediction.get("surface_type") or surface_type
        if resolved_surface is None and len(allowed_surfaces) == 1:
            resolved_surface = allowed_surfaces[0]
        if resolved_surface not in allowed_surfaces:
            return _abstention(
                reasons=["ambiguous_surface_without_verified_surface_metadata"],
                image=image,
                prediction_count=len(predictions),
                model_name=model_name,
                model_version=model_version,
            )
        retained.append(
            (index, confidence, class_id, [float(value) for value in box], resolved_surface)
        )

    if not retained:
        return _abstention(
            reasons=["no_detection_at_or_above_minimum_confidence"],
            image=image,
            prediction_count=len(predictions),
            model_name=model_name,
            model_version=model_version,
        )

    observations: list[dict[str, Any]] = []
    model_card_id = _load_json(_resolve(integration["model_card"]))["model_card_id"]
    for index, confidence, class_id, box, resolved_surface in retained:
        token = json.dumps(
            [image, index, class_id, box, model_version],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        observation_id = f"twm-infradet-{hashlib.sha256(token).hexdigest()[:20]}"
        x_min, y_min, x_max, y_max = box
        observations.append(
            {
                "contract_version": integration["output_contract"]["contract_version"],
                "observation_id": observation_id,
                "surface_type": resolved_surface,
                "damage_type": CLASS_NAMES[class_id],
                "evidence_region": {
                    "type": "bounding_box",
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
                "confidence": confidence,
                "uncertainty": {
                    "state": "review_required",
                    "reasons": [
                        "research_prototype",
                        "raw_detector_confidence_not_posthoc_calibrated",
                    ],
                },
                "location": normalised_location,
                "evidence": dict(source_evidence),
                "model": {
                    "name": model_name,
                    "version": model_version,
                    "model_card_id": model_card_id,
                },
                "review": {"status": "pending"},
            }
        )
    return {
        "result_type": "observations",
        "observations": observations,
        "automatic_operational_action_enabled": False,
    }


def _smoke_trace_matches(record: dict[str, Any], cache_record: dict[str, Any]) -> bool:
    smoke = record["smoke_inference_evidence"]
    if cache_record.get("image") != smoke["image"]:
        return False
    predictions = cache_record.get("predictions")
    index = smoke["prediction_index"]
    if not isinstance(predictions, list) or not 0 <= index < len(predictions):
        return False
    actual = predictions[index]
    expected = smoke["raw_detection"]
    return (
        actual.get("class_id") == expected["class_id"]
        and actual.get("confidence") == expected["confidence"]
        and actual.get("box") == expected["box_xyxy_normalised"]
    )


def _smoke_manifest_matches(record: dict[str, Any], manifest: dict[str, Any]) -> bool:
    smoke = record["smoke_inference_evidence"]
    asset = next(
        (item for item in manifest.get("assets", []) if item.get("asset_id") == smoke["manifest_asset_id"]),
        None,
    )
    return bool(
        asset
        and asset.get("source_id") == smoke["source_id"]
        and asset.get("relative_image", "").endswith(smoke["image"])
        and asset.get("sha256", "").upper() == smoke["manifest_asset_sha256"]
    )


def _smoke_source_matches(record: dict[str, Any], registry: dict[str, Any]) -> bool:
    smoke = record["smoke_inference_evidence"]
    source = next(
        (item for item in registry.get("sources", []) if item.get("id") == smoke["source_id"]),
        None,
    )
    evidence = smoke["source_evidence"]
    return bool(
        source
        and source.get("licence_verified") is True
        and source.get("training_eligibility") == "permitted"
        and source.get("attribution") == evidence["attribution"]
    )


def stage6_twm_status() -> dict[str, Any]:
    record = _load_json(INTEGRATION_RECORD)
    cache_path = _resolve(record["real_inference_cache"])
    manifest_path = _resolve(record["provenance"]["dataset_manifest"])
    registry_path = _resolve(record["provenance"]["source_registry"])
    schema_path = _resolve(record["output_contract"]["path"])

    cache_record = _first_cache_record(cache_path)
    manifest = _load_json(manifest_path)
    registry = _load_json(registry_path)
    schema = _load_json(schema_path)
    smoke = record["smoke_inference_evidence"]
    smoke_result = adapt_predictions(
        image=cache_record["image"],
        predictions=cache_record["predictions"],
        source_evidence=smoke["source_evidence"],
        location=None,
        record=record,
    )

    required_observation_fields = {
        "contract_version",
        "observation_id",
        "surface_type",
        "damage_type",
        "evidence_region",
        "confidence",
        "uncertainty",
        "location",
        "evidence",
        "model",
        "review",
    }
    expected_abstention = smoke["expected_adapter_result"]
    checks = {
        "checkpoint_verified": _verified_file(record["checkpoint"], record["checkpoint_sha256"]),
        "real_inference_cache_verified": _verified_file(
            record["real_inference_cache"], record["real_inference_cache_sha256"]
        ),
        "model_card_verified": _verified_file(record["model_card"], record["model_card_sha256"]),
        "runtime_adapter_verified": _verified_file(
            record["runtime_adapter"]["path"], record["runtime_adapter"]["sha256"]
        ),
        "observation_contract_verified": _verified_file(
            record["output_contract"]["path"], record["output_contract"]["sha256"]
        ),
        "dataset_manifest_verified": _verified_file(
            record["provenance"]["dataset_manifest"],
            record["provenance"]["dataset_manifest_sha256"],
        ),
        "source_registry_verified": _verified_file(
            record["provenance"]["source_registry"],
            record["provenance"]["source_registry_sha256"],
        ),
        "cycle01_diagnostic_verified": _verified_file(
            record["provenance"]["cycle01_diagnostic"],
            record["provenance"]["cycle01_diagnostic_sha256"],
        ),
        "cycle03_decision_manifest_verified": _verified_file(
            record["provenance"]["cycle03_decision_manifest"],
            record["provenance"]["cycle03_decision_manifest_sha256"],
        ),
        "smoke_prediction_trace_verified": _smoke_trace_matches(record, cache_record),
        "smoke_manifest_trace_verified": _smoke_manifest_matches(record, manifest),
        "smoke_source_trace_verified": _smoke_source_matches(record, registry),
        "output_contract_required_fields_verified": set(schema.get("required", []))
        == required_observation_fields,
        "raw_confidence_truthfully_labelled": record["adapter_policy"]["confidence_semantics"]
        == "raw_detector_score_not_posthoc_calibrated"
        and record["adapter_policy"]["posthoc_calibration_applied"] is False,
        "missing_location_safely_abstains": smoke_result["result_type"]
        == expected_abstention["result_type"]
        and smoke_result["state"] == expected_abstention["state"]
        and smoke_result["reasons"] == expected_abstention["reasons"],
        "human_review_required": record["adapter_policy"]["human_review_required"] is True,
        "automatic_operational_action_disabled": record["adapter_policy"][
            "automatic_operational_action_enabled"
        ]
        is False,
    }
    technical_ready = all(checks.values())
    return {
        **record,
        "integration_checks": checks,
        "smoke_adapter_result": smoke_result,
        "technical_integration_ready": technical_ready,
        "integration_ready": technical_ready,
        "readiness_scope": "isolated_research_prototype_technical_integration_only",
        "scientific_promotion_ready": record["readiness"]["scientific_promotion_ready"],
        "operational_ready": record["readiness"]["operational_ready"],
    }
