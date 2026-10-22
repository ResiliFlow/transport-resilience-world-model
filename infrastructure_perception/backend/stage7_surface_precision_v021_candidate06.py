from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from . import stage7_surface_precision as base
from . import stage7_surface_precision_v021 as candidate01
from . import stage7_surface_precision_v021_candidate04 as parent


MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.21-candidate06"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision v0.1.21 candidate06"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.21-candidate06"
PARENT_VERSION = parent.MODEL_VERSION

FOOTPATH_CONTEXT_INSTRUCTIONS = """
You are the final region verifier for a road-condition system. Judge only the red highlighted hypothesis.
The first image is the unmodified exact Google Street View frame, the second shows the same frame with the
hypothesis outlined in red, the third is an enlarged context crop, and the fourth is a same-panorama detail
view. The red rectangle is an untrusted proposal and is not evidence.

Accept uneven_or_broken_footpath only when the highlighted physical material is visibly both:
1. a permanent, ground-integrated pedestrian walking surface that continues into the surrounding footpath;
2. genuinely fractured, displaced, missing or uneven inside the highlighted region.

Reject a detached, leaning, raised, bridging or portable board, shop access ramp, tabletop, pallet, lid, mat,
display platform, stall fixture, wall/base, doorway threshold, vehicle part, panorama mask, ordinary intact
surface, or loose object merely resting on the footpath. A sloped or rough surface is not damage by itself.
Do not infer permanence from the earlier label or explanation. Trace the visible plane and edges in the full
frame and crop. If the evidence is ambiguous, return uncertain rather than inventing fixed-surface continuity.
""".strip()

OVERLAY_POLICY = {
    "identity": "stage7-footpath-context-overlay-v1",
    "scope": "uneven_or_broken_footpath_only",
    "context_expansion": 2.5,
    "minimum_context_pixels": 128,
    "temporary_files_deleted_after_verification": True,
    "per_frame_coordinates_or_rules": False,
}


class Stage7SurfacePrecisionCandidate06Error(parent.Stage7SurfacePrecisionCandidate04Error):
    pass


def footpath_verification_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["accept_fixed_surface_defect", "reject_unsupported_object", "reject_no_visible_damage", "uncertain"],
            },
            "permanent_ground_integrated_walking_surface": {"type": "boolean"},
            "visible_physical_damage_inside_box": {"type": "boolean"},
            "detached_raised_leaning_or_portable_object": {"type": "boolean"},
            "ordinary_surface_or_panorama_artifact": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "verdict",
            "permanent_ground_integrated_walking_surface",
            "visible_physical_damage_inside_box",
            "detached_raised_leaning_or_portable_object",
            "ordinary_surface_or_panorama_artifact",
            "reason",
        ],
    }


FOOTPATH_CONTEXT_POLICY_SHA256 = hashlib.sha256(FOOTPATH_CONTEXT_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
FOOTPATH_VERIFICATION_SCHEMA_SHA256 = base._stable_hash(footpath_verification_schema())
OVERLAY_POLICY_SHA256 = base._stable_hash(OVERLAY_POLICY)


def checks() -> dict[str, bool]:
    return {
        "candidate_adapter_available": Path(__file__).is_file(),
        "candidate04_parent_verified": all(parent.checks().values()),
        "candidate04_version_verified": parent.component_identity()["version"] == PARENT_VERSION,
        "fixed_surface_binary_verifier_present": "permanent, ground-integrated" in FOOTPATH_CONTEXT_INSTRUCTIONS,
        "portable_fixture_rejection_present": "portable board" in FOOTPATH_CONTEXT_INSTRUCTIONS,
        "verification_schema_frozen": FOOTPATH_VERIFICATION_SCHEMA_SHA256 == base._stable_hash(footpath_verification_schema()),
        "overlay_policy_frozen": OVERLAY_POLICY_SHA256 == base._stable_hash(OVERLAY_POLICY),
        "temporary_overlay_cleanup_declared": OVERLAY_POLICY["temporary_files_deleted_after_verification"] is True,
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": base._sha256(Path(__file__)),
        "candidate04_parent": parent.component_identity()["checkpoint_sha256"],
        "footpath_context_policy": FOOTPATH_CONTEXT_POLICY_SHA256,
        "footpath_verification_schema": FOOTPATH_VERIFICATION_SCHEMA_SHA256,
        "overlay_policy": OVERLAY_POLICY_SHA256,
    }
    composite = hashlib.sha256(
        (
            ":".join(components.values())
            + f":{parent.DEFAULT_MODEL}:exact-current-view-v6-highlighted-footpath-context-verifier"
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


def _normalised_box_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(float(box[0]) * width))))
    y1 = max(0, min(height - 1, int(round(float(box[1]) * height))))
    x2 = max(x1 + 1, min(width, int(round(float(box[2]) * width))))
    y2 = max(y1 + 1, min(height, int(round(float(box[3]) * height))))
    return x1, y1, x2, y2


def _jpeg_bytes(image: Image.Image, *, quality: int) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality, subsampling=0)
    return stream.getvalue()


def _make_context_images(image_path: Path, box: list[float]) -> tuple[bytes, bytes]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = _normalised_box_to_pixels(box, width, height)
    line_width = max(3, round(min(width, height) * 0.008))

    marked = image.copy()
    ImageDraw.Draw(marked).rectangle((x1, y1, x2 - 1, y2 - 1), outline=(230, 30, 45), width=line_width)

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    context_width = max(int(box_width * OVERLAY_POLICY["context_expansion"]), OVERLAY_POLICY["minimum_context_pixels"])
    context_height = max(int(box_height * OVERLAY_POLICY["context_expansion"]), OVERLAY_POLICY["minimum_context_pixels"])
    centre_x = (x1 + x2) // 2
    centre_y = (y1 + y2) // 2
    crop_x1 = max(0, centre_x - context_width // 2)
    crop_y1 = max(0, centre_y - context_height // 2)
    crop_x2 = min(width, crop_x1 + context_width)
    crop_y2 = min(height, crop_y1 + context_height)
    crop_x1 = max(0, crop_x2 - context_width)
    crop_y1 = max(0, crop_y2 - context_height)
    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    ImageDraw.Draw(crop).rectangle(
        (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1 - 1, y2 - crop_y1 - 1),
        outline=(230, 30, 45),
        width=line_width,
    )
    return _jpeg_bytes(marked, quality=94), _jpeg_bytes(crop, quality=95)


def _image_bytes_content(data: bytes, label: str) -> list[dict[str, Any]]:
    encoded = base64.b64encode(data).decode("ascii")
    return [
        {"type": "input_text", "text": f"Image label: {label}"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}", "detail": "original"},
    ]


def _verification_accepts(verdict: dict[str, Any]) -> bool:
    return bool(
        verdict.get("verdict") == "accept_fixed_surface_defect"
        and verdict.get("permanent_ground_integrated_walking_surface") is True
        and verdict.get("visible_physical_damage_inside_box") is True
        and verdict.get("detached_raised_leaning_or_portable_object") is False
        and verdict.get("ordinary_surface_or_panorama_artifact") is False
    )


def _verify_footpath_detection(
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
        raise Stage7SurfacePrecisionCandidate06Error("A footpath hypothesis has an invalid box.")
    marked_image, crop_image = _make_context_images(image_path, box)
    centre_x = (float(box[0]) + float(box[2])) / 2.0
    detail_label = "left" if centre_x < 0.38 else "right" if centre_x > 0.62 else "centre"
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Verify highlighted footpath hypothesis in exact frame {frame_identity}. "
                f"Normalised box: {json.dumps(box, separators=(',', ':'))}. "
                f"Earlier evidence is untrusted: {str(detection.get('evidence') or '')[:400]}"
            ),
        }
    ]
    content.extend(base._image_content(image_path, "unmodified_exact_frame"))
    content.extend(_image_bytes_content(marked_image, "exact_frame_with_untrusted_red_hypothesis"))
    content.extend(_image_bytes_content(crop_image, "enlarged_context_with_untrusted_red_hypothesis"))
    content.extend(base._image_content(tiles[detail_label], f"same_panorama_{detail_label}_detail"))
    try:
        return candidate01._call_structured(
            api_key=api_key,
            instructions=FOOTPATH_CONTEXT_INSTRUCTIONS,
            content=content,
            schema=footpath_verification_schema(),
            schema_name="stage7_v021_candidate06_footpath_context_verdict",
            max_output_tokens=900,
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
        raise Stage7SurfacePrecisionCandidate06Error("A Stage 7 verifier key is required for this tab.")
    failed_checks = [name for name, passed in checks().items() if not passed]
    if failed_checks:
        raise Stage7SurfacePrecisionCandidate06Error(
            f"The Stage 7 v0.1.21 candidate06 is unavailable: {', '.join(failed_checks)}."
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
            if detection.get("damage_type") != "uneven_or_broken_footpath" or detection.get("surface_type") != "footpath":
                retained.append(detection)
                continue
            verdict, meta = _verify_footpath_detection(
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
            consensus["highlighted_footpath_context_verifier"] = {
                "accepted": accepted,
                "verdict": verdict.get("verdict"),
                "permanent_ground_integrated_walking_surface": verdict.get("permanent_ground_integrated_walking_surface"),
                "visible_physical_damage_inside_box": verdict.get("visible_physical_damage_inside_box"),
                "detached_raised_leaning_or_portable_object": verdict.get("detached_raised_leaning_or_portable_object"),
                "ordinary_surface_or_panorama_artifact": verdict.get("ordinary_surface_or_panorama_artifact"),
                "reason": str(verdict.get("reason") or "")[:500],
                "policy_sha256": FOOTPATH_CONTEXT_POLICY_SHA256,
                "schema_sha256": FOOTPATH_VERIFICATION_SCHEMA_SHA256,
                "overlay_policy_sha256": OVERLAY_POLICY_SHA256,
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
            "run_id": f"stage7-surface-precision-v021-c06-run-{uuid.uuid4().hex}",
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
            "strategy": "exact_current_view_v6_highlighted_footpath_context_verifier",
            "parent_version": PARENT_VERSION,
            "footpath_context_policy_sha256": FOOTPATH_CONTEXT_POLICY_SHA256,
            "footpath_verification_schema_sha256": FOOTPATH_VERIFICATION_SCHEMA_SHA256,
            "overlay_policy_sha256": OVERLAY_POLICY_SHA256,
            "footpath_context_verifier_count": len(verifier_meta),
            "footpath_context_rejected_count": len(rejected),
            "footpath_context_rejections": rejected,
            "footpath_context_provider_models": [item.get("model") for item in verifier_meta],
            "footpath_context_provider_response_ids": [item.get("response_id") for item in verifier_meta],
            "footpath_context_provider_usage": [item.get("usage") or {} for item in verifier_meta],
            "retained_box_count": len(retained),
            "elapsed_seconds": time.perf_counter() - started,
            "temporary_context_images_retained": False,
            "provider_credential_persisted": False,
        }
    )
    result["detections"] = retained
    return result
