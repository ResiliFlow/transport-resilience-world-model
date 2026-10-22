from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from . import stage7_surface_precision as base
from . import stage7_surface_precision_v021 as candidate01
from . import stage7_surface_precision_v021_candidate06 as parent


MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.21-candidate07"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision v0.1.21 candidate07"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.21-candidate07"
PARENT_VERSION = parent.MODEL_VERSION

DEBRIS_CONTEXT_INSTRUCTIONS = """
You are the final highlighted-region verifier for debris or vegetation obstruction in a road-condition system.
The first image is the exact Google Street View frame, the second marks an untrusted proposed box in red,
the third is its enlarged context, and the fourth is a same-panorama detail view.

Accept debris_or_vegetation_obstruction only when the red box tightly covers a discrete pile or object of
loose rubble, refuse, fallen material or vegetation that physically lies on and obstructs an identifiable
paved road, kerb or permanent footpath. The eligible surface must be visibly present inside the box.

Reject general gravel, stones or roughness that form an unpaved shoulder, bare soil, construction ground,
vacant land or the ordinary roadside beyond the paved edge. Reject vendor carts, tables, stalls, vehicles,
people, merchandise, sacks, crates, ordinary bins, panorama masks and harmless small litter. Adjacency to a
road is not overlap with a road. If the paved boundary or obstruction is ambiguous, return uncertain.
""".strip()


class Stage7SurfacePrecisionCandidate07Error(parent.Stage7SurfacePrecisionCandidate06Error):
    pass


def debris_verification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "accept_discrete_supported_obstruction",
                    "reject_unpaved_shoulder_or_bare_ground",
                    "reject_mobile_or_commercial_object",
                    "reject_not_obstructing",
                    "reject_panorama_artifact",
                    "uncertain",
                ],
            },
            "discrete_loose_material_or_vegetation": {"type": "boolean"},
            "eligible_surface_visible_inside_box": {"type": "boolean"},
            "physically_overlaps_eligible_surface": {"type": "boolean"},
            "obstructs_usable_corridor": {"type": "boolean"},
            "unpaved_shoulder_or_bare_ground_dominated": {"type": "boolean"},
            "mobile_commercial_or_traffic_object": {"type": "boolean"},
            "panorama_mask_or_blur_dominated": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "verdict",
            "discrete_loose_material_or_vegetation",
            "eligible_surface_visible_inside_box",
            "physically_overlaps_eligible_surface",
            "obstructs_usable_corridor",
            "unpaved_shoulder_or_bare_ground_dominated",
            "mobile_commercial_or_traffic_object",
            "panorama_mask_or_blur_dominated",
            "reason",
        ],
    }


DEBRIS_CONTEXT_POLICY_SHA256 = hashlib.sha256(DEBRIS_CONTEXT_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
DEBRIS_VERIFICATION_SCHEMA_SHA256 = base._stable_hash(debris_verification_schema())


def checks() -> dict[str, bool]:
    return {
        "candidate_adapter_available": Path(__file__).is_file(),
        "candidate06_parent_verified": all(parent.checks().values()),
        "candidate06_version_verified": parent.component_identity()["version"] == PARENT_VERSION,
        "unpaved_shoulder_exclusion_present": "unpaved shoulder" in DEBRIS_CONTEXT_INSTRUCTIONS,
        "mobile_object_exclusion_present": "vendor carts" in DEBRIS_CONTEXT_INSTRUCTIONS,
        "eligible_surface_overlap_required": "physically lies on" in DEBRIS_CONTEXT_INSTRUCTIONS,
        "debris_schema_frozen": DEBRIS_VERIFICATION_SCHEMA_SHA256 == base._stable_hash(debris_verification_schema()),
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": base._sha256(Path(__file__)),
        "candidate06_parent": parent.component_identity()["checkpoint_sha256"],
        "debris_context_policy": DEBRIS_CONTEXT_POLICY_SHA256,
        "debris_verification_schema": DEBRIS_VERIFICATION_SCHEMA_SHA256,
        "overlay_policy": parent.OVERLAY_POLICY_SHA256,
    }
    composite = hashlib.sha256(
        (
            ":".join(components.values())
            + f":{parent.parent.DEFAULT_MODEL}:exact-current-view-v7-highlighted-footpath-and-debris-context-verifier"
        ).encode("ascii")
    ).hexdigest().upper()
    return {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "checkpoint_sha256": composite,
        "components": components,
        "activation_state": "shadow_candidate_unaccepted",
        "parent_version": PARENT_VERSION,
        "rollback_version": base.MODEL_VERSION,
    }


def _verification_accepts(verdict: dict[str, Any]) -> bool:
    return bool(
        verdict.get("verdict") == "accept_discrete_supported_obstruction"
        and verdict.get("discrete_loose_material_or_vegetation") is True
        and verdict.get("eligible_surface_visible_inside_box") is True
        and verdict.get("physically_overlaps_eligible_surface") is True
        and verdict.get("obstructs_usable_corridor") is True
        and verdict.get("unpaved_shoulder_or_bare_ground_dominated") is False
        and verdict.get("mobile_commercial_or_traffic_object") is False
        and verdict.get("panorama_mask_or_blur_dominated") is False
    )


def _verify_debris_detection(
    *,
    api_key: str,
    image_path: Path,
    tiles: dict[str, Path],
    frame_identity: str,
    detection: dict[str, Any],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    box = list(detection.get("box") or [])
    if len(box) != 4:
        raise Stage7SurfacePrecisionCandidate07Error("A debris hypothesis has an invalid box.")
    marked_image, crop_image = parent._make_context_images(image_path, box)
    centre_x = (float(box[0]) + float(box[2])) / 2.0
    detail_label = "left" if centre_x < 0.38 else "right" if centre_x > 0.62 else "centre"
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Verify highlighted debris hypothesis in exact frame {frame_identity}. "
                f"Normalised box: {json.dumps(box, separators=(',', ':'))}. "
                f"Earlier evidence is untrusted: {str(detection.get('evidence') or '')[:400]}"
            ),
        }
    ]
    content.extend(base._image_content(image_path, "unmodified_exact_frame"))
    content.extend(parent._image_bytes_content(marked_image, "exact_frame_with_untrusted_red_hypothesis"))
    content.extend(parent._image_bytes_content(crop_image, "enlarged_context_with_untrusted_red_hypothesis"))
    content.extend(base._image_content(tiles[detail_label], f"same_panorama_{detail_label}_detail"))
    try:
        return candidate01._call_structured(
            api_key=api_key,
            instructions=DEBRIS_CONTEXT_INSTRUCTIONS,
            content=content,
            schema=debris_verification_schema(),
            schema_name="stage7_v021_candidate07_debris_context_verdict",
            max_output_tokens=1000,
            timeout=timeout,
        )
    finally:
        marked_image = b""
        crop_image = b""
        content.clear()


def infer(
    *,
    api_key: str,
    image_path: Path,
    tiles: dict[str, Path],
    frame_identity: str,
    timeout: float = 150.0,
) -> dict[str, Any]:
    cleaned_key = str(api_key or "").strip()
    if len(cleaned_key) < 20:
        raise Stage7SurfacePrecisionCandidate07Error("A Stage 7 verifier key is required for this tab.")
    failed_checks = [name for name, passed in checks().items() if not passed]
    if failed_checks:
        raise Stage7SurfacePrecisionCandidate07Error(
            f"The Stage 7 v0.1.21 candidate07 is unavailable: {', '.join(failed_checks)}."
        )

    started = time.perf_counter()
    try:
        result = parent.infer(
            api_key=cleaned_key,
            image_path=image_path,
            tiles=tiles,
            frame_identity=frame_identity,
            timeout=timeout,
        )
        retained: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        verifier_meta: list[dict[str, Any]] = []
        for detection in result.get("detections") or []:
            if detection.get("damage_type") != "debris_or_vegetation_obstruction":
                retained.append(detection)
                continue
            verdict, meta = _verify_debris_detection(
                api_key=cleaned_key,
                image_path=image_path,
                tiles=tiles,
                frame_identity=frame_identity,
                detection=detection,
                timeout=timeout,
            )
            verifier_meta.append(meta)
            accepted = _verification_accepts(verdict)
            consensus = dict(detection.get("consensus") or {})
            consensus["highlighted_debris_context_verifier"] = {
                "accepted": accepted,
                "verdict": verdict.get("verdict"),
                "discrete_loose_material_or_vegetation": verdict.get("discrete_loose_material_or_vegetation"),
                "eligible_surface_visible_inside_box": verdict.get("eligible_surface_visible_inside_box"),
                "physically_overlaps_eligible_surface": verdict.get("physically_overlaps_eligible_surface"),
                "obstructs_usable_corridor": verdict.get("obstructs_usable_corridor"),
                "unpaved_shoulder_or_bare_ground_dominated": verdict.get("unpaved_shoulder_or_bare_ground_dominated"),
                "mobile_commercial_or_traffic_object": verdict.get("mobile_commercial_or_traffic_object"),
                "panorama_mask_or_blur_dominated": verdict.get("panorama_mask_or_blur_dominated"),
                "reason": str(verdict.get("reason") or "")[:500],
                "policy_sha256": DEBRIS_CONTEXT_POLICY_SHA256,
                "schema_sha256": DEBRIS_VERIFICATION_SCHEMA_SHA256,
            }
            detection["consensus"] = consensus
            if accepted:
                retained.append(detection)
            else:
                rejected.append(
                    {
                        "detection_id": detection.get("detection_id"),
                        "damage_type": detection.get("damage_type"),
                        "surface_type": detection.get("surface_type"),
                        "box": list(detection.get("box") or []),
                        "verdict": verdict.get("verdict"),
                        "reason": str(verdict.get("reason") or "")[:500],
                    }
                )
    finally:
        cleaned_key = ""

    identity = component_identity()
    for index, detection in enumerate(retained):
        token = json.dumps(
            [frame_identity, index, detection.get("damage_type"), detection.get("box"), MODEL_VERSION],
            separators=(",", ":"),
        ).encode("utf-8")
        detection["detection_id"] = f"stage7-detection-{hashlib.sha256(token).hexdigest()[:20]}"
    run = result["run"]
    run.update(
        {
            "run_id": f"stage7-surface-precision-v021-c07-run-{uuid.uuid4().hex}",
            "model_id": identity["id"],
            "model_name": identity["name"],
            "model_version": identity["version"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "component_sha256": identity["components"],
            "result_type": "detections" if retained else "no_detection",
            "detection_ids": [item["detection_id"] for item in retained],
        }
    )
    diagnostics = run.setdefault("diagnostics", {})
    diagnostics.update(
        {
            "strategy": "exact_current_view_v7_highlighted_footpath_and_debris_context_verifier",
            "parent_version": PARENT_VERSION,
            "debris_context_policy_sha256": DEBRIS_CONTEXT_POLICY_SHA256,
            "debris_verification_schema_sha256": DEBRIS_VERIFICATION_SCHEMA_SHA256,
            "debris_context_verifier_count": len(verifier_meta),
            "debris_context_rejected_count": len(rejected),
            "debris_context_rejections": rejected,
            "debris_context_provider_models": [item.get("model") for item in verifier_meta],
            "debris_context_provider_response_ids": [item.get("response_id") for item in verifier_meta],
            "debris_context_provider_usage": [item.get("usage") or {} for item in verifier_meta],
            "retained_box_count": len(retained),
            "elapsed_seconds": time.perf_counter() - started,
            "temporary_context_images_retained": False,
            "provider_credential_persisted": False,
        }
    )
    result["detections"] = retained
    return result
