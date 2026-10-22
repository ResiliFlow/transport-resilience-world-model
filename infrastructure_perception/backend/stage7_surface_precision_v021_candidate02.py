from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from . import stage7_surface_precision as parent
from . import stage7_surface_precision_v021 as candidate01


DEFAULT_MODEL = parent.DEFAULT_MODEL
MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.21-candidate02"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision v0.1.21 candidate02"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.21-candidate02"
PARENT_VERSION = candidate01.MODEL_VERSION
EIGHT_CLASSES = parent.EIGHT_CLASSES
SURFACES = parent.SURFACES

PROPOSER_INSTRUCTIONS = (
    candidate01.PROPOSER_INSTRUCTIONS
    + """

Object-exclusion rule: people, motorcycles, bicycles, cars, buses, parked vehicles, handcarts, vendor carts, tables, stalls, merchandise, sacks, crates, ordinary bins and other movable commercial or traffic objects are never infrastructure defects. Do not call any such object debris_or_vegetation_obstruction. That class is limited to loose debris, rubble or vegetation physically lying on and obstructing a road, kerb or footpath.
"""
).strip()

ADJUDICATOR_INSTRUCTIONS = (
    candidate01.ADJUDICATOR_INSTRUCTIONS
    + """

Enforce object identity before accepting any box. A street-vendor handcart, trolley, table, stall, vehicle, motorcycle, bicycle, person, merchandise, sack, crate or ordinary bin is an unsupported object even when it occupies the roadside. Reject it. For debris_or_vegetation_obstruction, the visible material itself must be loose rubble, refuse or vegetation lying on an identifiable road, kerb or footpath.

For every returned box, confirm that the road, kerb or footpath host surface is visibly identifiable at the defect location. Set host_surface_boundary_visible false if the box is instead on a building, wall, shopfront, vehicle, cart, person or ambiguous background. Set unsupported_mobile_or_commercial_object true whenever the proposed content is one of those excluded objects. Only return defects with host_surface_boundary_visible true and unsupported_mobile_or_commercial_object false.
"""
).strip()


class Stage7SurfacePrecisionCandidate02Error(candidate01.Stage7SurfacePrecisionV021Error):
    pass


def adjudication_schema() -> dict[str, Any]:
    schema = json.loads(json.dumps(candidate01.adjudication_schema()))
    defect = schema["properties"]["defects"]["items"]
    defect["properties"]["host_surface_boundary_visible"] = {"type": "boolean"}
    defect["properties"]["unsupported_mobile_or_commercial_object"] = {"type": "boolean"}
    defect["required"].extend(
        ["host_surface_boundary_visible", "unsupported_mobile_or_commercial_object"]
    )
    return schema


PROPOSER_POLICY_SHA256 = hashlib.sha256(PROPOSER_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
ADJUDICATOR_POLICY_SHA256 = hashlib.sha256(ADJUDICATOR_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
PROPOSER_SCHEMA_SHA256 = parent._stable_hash(parent.response_schema())
ADJUDICATOR_SCHEMA_SHA256 = parent._stable_hash(adjudication_schema())


def checks() -> dict[str, bool]:
    return {
        "candidate_adapter_available": Path(__file__).is_file(),
        "candidate01_parent_verified": all(candidate01.checks().values()),
        "candidate01_version_verified": candidate01.component_identity()["version"] == PARENT_VERSION,
        "object_exclusion_policy_present": "handcart" in ADJUDICATOR_INSTRUCTIONS,
        "host_surface_evidence_required": "host_surface_boundary_visible" in json.dumps(adjudication_schema()),
        "frozen_taxonomy_verified": len(EIGHT_CLASSES) == 8 and len(set(EIGHT_CLASSES)) == 8,
        "surface_contract_verified": set(SURFACES) == {"road", "kerb", "footpath"},
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": parent._sha256(Path(__file__)),
        "candidate01_parent": candidate01.component_identity()["checkpoint_sha256"],
        "proposer_policy": PROPOSER_POLICY_SHA256,
        "adjudicator_policy": ADJUDICATOR_POLICY_SHA256,
        "proposer_schema": PROPOSER_SCHEMA_SHA256,
        "adjudicator_schema": ADJUDICATOR_SCHEMA_SHA256,
        "bottom_mask_guard": parent.BOTTOM_MASK_GUARD_SHA256,
    }
    composite = hashlib.sha256(
        (":".join(components.values()) + f":{DEFAULT_MODEL}:exact-current-view-two-pass-v2-object-gate").encode("ascii")
    ).hexdigest().upper()
    return {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "checkpoint_sha256": composite,
        "components": components,
        "activation_state": "shadow_candidate_unaccepted",
        "parent_version": PARENT_VERSION,
        "rollback_version": parent.MODEL_VERSION,
    }


def _call_passes(
    *,
    api_key: str,
    frame_identity: str,
    exact: Path,
    tiles: dict[str, Path],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    proposer_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Generate high-recall proposals for exact current-view frame {frame_identity}. "
                "Return boxes only in the canonical exact image and exclude movable street objects."
            ),
        }
    ]
    proposer_content.extend(candidate01._image_set_content(exact, tiles))
    proposals, proposer_meta = candidate01._call_structured(
        api_key=api_key,
        instructions=PROPOSER_INSTRUCTIONS,
        content=proposer_content,
        schema=parent.response_schema(),
        schema_name="stage7_v021_candidate02_proposals",
        max_output_tokens=1800,
        timeout=timeout,
    )
    parent._validate_result(proposals)

    proposal_summary = json.dumps(
        {
            "frame_verdict": proposals.get("frame_verdict"),
            "eligible_surface_visible": proposals.get("eligible_surface_visible"),
            "defects": proposals.get("defects") or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    adjudicator_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Independently adjudicate frame {frame_identity}. The hypotheses below may contain "
                "unsupported movable objects and may miss real surface defects. Produce the final corrected set.\n"
                f"{proposal_summary}"
            ),
        }
    ]
    adjudicator_content.extend(candidate01._image_set_content(exact, tiles))
    final, adjudicator_meta = candidate01._call_structured(
        api_key=api_key,
        instructions=ADJUDICATOR_INSTRUCTIONS,
        content=adjudicator_content,
        schema=adjudication_schema(),
        schema_name="stage7_v021_candidate02_adjudication",
        max_output_tokens=2300,
        timeout=timeout,
    )
    _validate_adjudication(final)
    return proposals, final, [proposer_meta, adjudicator_meta]


def _validate_adjudication(result: dict[str, Any]) -> None:
    candidate01._validate_adjudication(result)
    for defect in result.get("defects") or []:
        if not isinstance(defect.get("host_surface_boundary_visible"), bool):
            raise Stage7SurfacePrecisionCandidate02Error("The adjudicator omitted host-surface evidence.")
        if not isinstance(defect.get("unsupported_mobile_or_commercial_object"), bool):
            raise Stage7SurfacePrecisionCandidate02Error("The adjudicator omitted object-exclusion status.")


def _apply_evidence_guards(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for defect in result.get("defects") or []:
        reason: str | None = None
        if defect.get("unsupported_mobile_or_commercial_object") is True:
            reason = "unsupported_mobile_or_commercial_object"
        elif defect.get("host_surface_boundary_visible") is not True:
            reason = "host_surface_not_visibly_identified"
        if reason:
            rejected.append(
                {
                    "defect_class": defect.get("defect_class"),
                    "surface_type": defect.get("surface_type"),
                    "canonical_box_0_1000": dict(defect.get("canonical_box_0_1000") or {}),
                    "reason": reason,
                }
            )
        else:
            eligible.append(defect)
    guarded, base_rejections = candidate01._apply_evidence_guards({"defects": eligible})
    rejected.extend(base_rejections)
    return guarded, rejected


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
        raise Stage7SurfacePrecisionCandidate02Error("A Stage 7 verifier key is required for this tab.")
    if not image_path.is_file() or set(tiles) != {"left", "centre", "right"}:
        raise Stage7SurfacePrecisionCandidate02Error("The exact current-view image set is incomplete.")
    if any(not path.is_file() for path in tiles.values()):
        raise Stage7SurfacePrecisionCandidate02Error("A current-view detail image is unavailable.")
    failed_checks = [name for name, passed in checks().items() if not passed]
    if failed_checks:
        raise Stage7SurfacePrecisionCandidate02Error(
            f"The Stage 7 v0.1.21 candidate02 is unavailable: {', '.join(failed_checks)}."
        )

    started_at = parent._timestamp()
    started = time.perf_counter()
    try:
        proposals, final, response_meta = _call_passes(
            api_key=cleaned_key,
            frame_identity=frame_identity,
            exact=image_path,
            tiles=tiles,
            timeout=timeout,
        )
    finally:
        cleaned_key = ""

    retained, rejected = _apply_evidence_guards(final)
    identity = component_identity()
    detections: list[dict[str, Any]] = []
    for index, defect in enumerate(retained):
        box = defect["canonical_box_0_1000"]
        normalised_box = [
            box["x1"] / 1000.0,
            box["y1"] / 1000.0,
            box["x2"] / 1000.0,
            box["y2"] / 1000.0,
        ]
        token = json.dumps(
            [frame_identity, index, defect["defect_class"], normalised_box, MODEL_VERSION],
            separators=(",", ":"),
        ).encode("utf-8")
        detections.append(
            {
                "detection_id": f"stage7-detection-{hashlib.sha256(token).hexdigest()[:20]}",
                "class_id": EIGHT_CLASSES.index(defect["defect_class"]),
                "damage_type": defect["defect_class"],
                "surface_type": defect["surface_type"],
                "confidence": float(defect["confidence"]),
                "tentative": bool(defect["tentative"]),
                "box": normalised_box,
                "source_support": list(defect["source_support"]),
                "evidence": str(defect["evidence"]),
                "review": {"status": "pending"},
                "consensus": {
                    "strategy": "two_pass_recall_surface_mask_and_object_adjudication_v2",
                    "host_surface_gate": True,
                    "host_surface_boundary_visible": True,
                    "unsupported_mobile_or_commercial_object": False,
                    "physical_defect_evidence": True,
                    "panorama_mask_or_blur_dominated": False,
                    "origin": defect["origin"],
                },
            }
        )

    run = {
        "run_id": f"stage7-surface-precision-v021-c02-run-{uuid.uuid4().hex}",
        "origin_stage": 7,
        "started_at": started_at,
        "completed_at": parent._timestamp(),
        "frame_identity": frame_identity,
        "image_sha256": parent._sha256(image_path),
        "model_id": identity["id"],
        "model_name": identity["name"],
        "model_version": identity["version"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "component_sha256": identity["components"],
        "result_type": "detections" if detections else "no_detection",
        "detection_ids": [item["detection_id"] for item in detections],
        "surface_metadata_used": False,
        "stage6_observation_persisted": False,
        "automatic_twm_transfer": False,
        "automatic_operational_action_enabled": False,
        "diagnostics": {
            "strategy": "exact_current_view_two_pass_recall_surface_mask_and_object_adjudication_v2",
            "parent_version": PARENT_VERSION,
            "proposer_policy_sha256": PROPOSER_POLICY_SHA256,
            "adjudicator_policy_sha256": ADJUDICATOR_POLICY_SHA256,
            "proposer_schema_sha256": PROPOSER_SCHEMA_SHA256,
            "adjudicator_schema_sha256": ADJUDICATOR_SCHEMA_SHA256,
            "bottom_mask_guard_sha256": parent.BOTTOM_MASK_GUARD_SHA256,
            "proposal_box_count": len(proposals.get("defects") or []),
            "adjudicated_box_count": len(final.get("defects") or []),
            "retained_box_count": len(retained),
            "rejected_evidence_mask_host_or_object_box_count": len(rejected),
            "rejections": rejected,
            "recovered_miss_count": sum(1 for item in retained if item.get("origin") == "recovered_miss"),
            "frame_verdict": final.get("frame_verdict"),
            "eligible_surface_visible": final.get("eligible_surface_visible"),
            "rejected_common_distractors": final.get("rejected_common_distractors") or [],
            "summary": final.get("summary"),
            "provider_models": [item.get("model") for item in response_meta],
            "provider_response_ids": [item.get("response_id") for item in response_meta],
            "provider_usage": [item.get("usage") or {} for item in response_meta],
            "elapsed_seconds": time.perf_counter() - started,
            "provider_credential_persisted": False,
        },
    }
    return {"run": run, "detections": detections}
