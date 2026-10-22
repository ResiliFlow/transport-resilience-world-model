from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import stage7_surface_precision as parent


DEFAULT_MODEL = parent.DEFAULT_MODEL
MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.21"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision v0.1.21"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.21-candidate01"
PARENT_VERSION = parent.MODEL_VERSION
EIGHT_CLASSES = parent.EIGHT_CLASSES
SURFACES = parent.SURFACES
IMAGE_LABELS = parent.IMAGE_LABELS
TRANSIENT_HTTP = parent.TRANSIENT_HTTP

PROPOSER_INSTRUCTIONS = """
You are the high-recall proposal stage for exact-current-view road-condition analysis.

The first image is the canonical Google Street View frame visible to the user. The next three images are higher-resolution left, centre and right views from the same panorama. Search the whole canonical frame for human-visible physical defects on road, kerb or footpath. The only permitted classes are pothole, longitudinal_crack, transverse_crack, alligator_cracking, uneven_or_broken_footpath, broken_kerb, standing_water, and debris_or_vegetation_obstruction.

Prioritise recall of visually clear defects. Inspect near-field road edges and the lower part of the image carefully. A real crack, cavity, broken edge, puddle or obstruction must not be omitted merely because it is close to the bottom edge or partly adjacent to panorama blur. At the same time, do not treat Google panorama stitching, vehicle masks, smooth blur, shadows, walls, buildings, vehicles, people, ordinary road markings or harmless texture as defects.

Return tight boxes in canonical coordinates 0..1000 around the physical defect itself. Detail views may confirm a defect but may never introduce content outside the canonical frame. If a real eligible defect is visible but its class is uncertain, choose the closest permitted class and set tentative true. Return an empty defects array only when no human-visible eligible defect can be located.
""".strip()

ADJUDICATOR_INSTRUCTIONS = """
You are the final independent visual adjudicator for an exact-current-view road-condition system. The user must receive small, defensible boxes on real defects, without boxes on buildings or panorama artefacts.

The first image is the canonical Google Street View frame. The next three images are same-panorama detail views. You will also receive a proposal list from a high-recall first pass. Treat those proposals only as hypotheses: reject false proposals, refine loose boxes, and independently scan the canonical frame for clearly visible eligible defects missed by the proposer.

Keep a defect only when all of these are true:
1. The physical defect itself is visibly supported in the canonical frame.
2. Its host is road, kerb or footpath.
3. It belongs to one of the eight permitted classes: pothole, longitudinal_crack, transverse_crack, alligator_cracking, uneven_or_broken_footpath, broken_kerb, standing_water, or debris_or_vegetation_obstruction.
4. The box tightly encloses the defect rather than the whole surface or surrounding background.

Panorama-mask rule: reject a proposal when the supposed defect is mainly a smooth, smeared, warped or textureless Google panorama/vehicle mask and no stable physical defect boundary or texture is visible. This rule applies to every class, not only standing water. Proximity to the bottom edge alone is not grounds for rejection. Preserve a genuine near-field road-edge crack, cavity, broken footpath, kerb break, puddle or obstruction when physical texture and boundaries remain visible next to a blurred region.

Reject buildings, walls, roofs, fences, vehicles, people, sky, signs, normal road markings, shadows, stains, ordinary construction joints and non-obstructing vegetation. Do not invent a generic anomaly class. For every returned item explicitly state whether real physical evidence is present and whether panorama blur dominates. Returned defects must have physical_defect_evidence true and panorama_mask_or_blur_dominated false. Mark recovered_miss when you add a clear defect absent from the proposal list.
""".strip()


class Stage7SurfacePrecisionV021Error(parent.Stage7SurfacePrecisionError):
    pass


def _box_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x1": {"type": "integer", "minimum": 0, "maximum": 1000},
            "y1": {"type": "integer", "minimum": 0, "maximum": 1000},
            "x2": {"type": "integer", "minimum": 0, "maximum": 1000},
            "y2": {"type": "integer", "minimum": 0, "maximum": 1000},
        },
        "required": ["x1", "y1", "x2", "y2"],
    }


def adjudication_schema() -> dict[str, Any]:
    defect = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "defect_class": {"type": "string", "enum": list(EIGHT_CLASSES)},
            "surface_type": {"type": "string", "enum": list(SURFACES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tentative": {"type": "boolean"},
            "canonical_box_0_1000": _box_schema(),
            "source_support": {
                "type": "array",
                "items": {"type": "string", "enum": list(IMAGE_LABELS)},
            },
            "evidence": {"type": "string"},
            "physical_defect_evidence": {"type": "boolean"},
            "panorama_mask_or_blur_dominated": {"type": "boolean"},
            "origin": {
                "type": "string",
                "enum": ["proposal_confirmed", "proposal_refined", "recovered_miss"],
            },
        },
        "required": [
            "defect_class",
            "surface_type",
            "confidence",
            "tentative",
            "canonical_box_0_1000",
            "source_support",
            "evidence",
            "physical_defect_evidence",
            "panorama_mask_or_blur_dominated",
            "origin",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frame_verdict": {
                "type": "string",
                "enum": ["positive", "clean", "no_eligible_surface", "uncertain"],
            },
            "eligible_surface_visible": {"type": "boolean"},
            "defects": {"type": "array", "items": defect},
            "rejected_common_distractors": {
                "type": "array",
                "items": {"type": "string"},
            },
            "summary": {"type": "string"},
        },
        "required": [
            "frame_verdict",
            "eligible_surface_visible",
            "defects",
            "rejected_common_distractors",
            "summary",
        ],
    }


PROPOSER_POLICY_SHA256 = hashlib.sha256(PROPOSER_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
ADJUDICATOR_POLICY_SHA256 = hashlib.sha256(ADJUDICATOR_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
PROPOSER_SCHEMA_SHA256 = parent._stable_hash(parent.response_schema())
ADJUDICATOR_SCHEMA_SHA256 = parent._stable_hash(adjudication_schema())


def checks() -> dict[str, bool]:
    parent_checks = parent.checks()
    return {
        "candidate_adapter_available": Path(__file__).is_file(),
        "parent_v020_package_verified": all(parent_checks.values()),
        "parent_version_verified": parent.component_identity()["version"] == PARENT_VERSION,
        "two_pass_policy_present": bool(PROPOSER_INSTRUCTIONS and ADJUDICATOR_INSTRUCTIONS),
        "schemas_present": bool(parent.response_schema() and adjudication_schema()),
        "frozen_taxonomy_verified": len(EIGHT_CLASSES) == 8 and len(set(EIGHT_CLASSES)) == 8,
        "surface_contract_verified": set(SURFACES) == {"road", "kerb", "footpath"},
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": parent._sha256(Path(__file__)),
        "parent_v020": parent.component_identity()["checkpoint_sha256"],
        "proposer_policy": PROPOSER_POLICY_SHA256,
        "adjudicator_policy": ADJUDICATOR_POLICY_SHA256,
        "proposer_schema": PROPOSER_SCHEMA_SHA256,
        "adjudicator_schema": ADJUDICATOR_SCHEMA_SHA256,
        "bottom_mask_guard": parent.BOTTOM_MASK_GUARD_SHA256,
    }
    composite = hashlib.sha256(
        (":".join(components.values()) + f":{DEFAULT_MODEL}:exact-current-view-two-pass-v2").encode("ascii")
    ).hexdigest().upper()
    return {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "checkpoint_sha256": composite,
        "components": components,
        "activation_state": "shadow_candidate_unaccepted",
        "parent_version": PARENT_VERSION,
        "rollback_version": PARENT_VERSION,
    }


def _call_structured(
    *,
    api_key: str,
    instructions: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_body = {
        "model": DEFAULT_MODEL,
        "store": False,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    request = Request(
        parent.API_ENDPOINT,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.8, 2.0, 4.0)):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = json.loads(parent._extract_output_text(payload))
            return result, {
                "response_id": payload.get("id"),
                "model": payload.get("model") or DEFAULT_MODEL,
                "usage": payload.get("usage") or {},
            }
        except HTTPError as error:
            last_error = parent._sanitised_http_error(error)
            if int(error.code) not in TRANSIENT_HTTP or attempt == 2:
                raise last_error from None
        except (TimeoutError, URLError, OSError, json.JSONDecodeError) as error:
            last_error = Stage7SurfacePrecisionV021Error(
                f"The Stage 7 v0.1.21 verifier transport failed ({type(error).__name__})."
            )
            if attempt == 2:
                raise last_error from None
        time.sleep(delay)
    raise last_error or Stage7SurfacePrecisionV021Error("The Stage 7 v0.1.21 verifier failed.")


def _image_set_content(exact: Path, tiles: dict[str, Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    content.extend(parent._image_content(exact, "exact"))
    for label in ("left", "centre", "right"):
        content.extend(parent._image_content(tiles[label], label))
    return content


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
                "Return boxes only in the canonical exact image."
            ),
        }
    ]
    proposer_content.extend(_image_set_content(exact, tiles))
    proposals, proposer_meta = _call_structured(
        api_key=api_key,
        instructions=PROPOSER_INSTRUCTIONS,
        content=proposer_content,
        schema=parent.response_schema(),
        schema_name="stage7_v021_high_recall_proposals",
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
                f"Independently adjudicate frame {frame_identity}. The first-pass hypotheses follow. "
                "They may contain false positives and may miss real defects. Produce the final corrected set.\n"
                f"{proposal_summary}"
            ),
        }
    ]
    adjudicator_content.extend(_image_set_content(exact, tiles))
    final, adjudicator_meta = _call_structured(
        api_key=api_key,
        instructions=ADJUDICATOR_INSTRUCTIONS,
        content=adjudicator_content,
        schema=adjudication_schema(),
        schema_name="stage7_v021_final_adjudication",
        max_output_tokens=2200,
        timeout=timeout,
    )
    _validate_adjudication(final)
    return proposals, final, [proposer_meta, adjudicator_meta]


def _validate_adjudication(result: dict[str, Any]) -> None:
    parent._validate_result(result)
    for defect in result.get("defects") or []:
        if not isinstance(defect.get("physical_defect_evidence"), bool):
            raise Stage7SurfacePrecisionV021Error("The adjudicator omitted physical-evidence status.")
        if not isinstance(defect.get("panorama_mask_or_blur_dominated"), bool):
            raise Stage7SurfacePrecisionV021Error("The adjudicator omitted panorama-mask status.")
        if defect.get("origin") not in {"proposal_confirmed", "proposal_refined", "recovered_miss"}:
            raise Stage7SurfacePrecisionV021Error("The adjudicator returned an invalid defect origin.")


def _apply_evidence_guards(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for defect in result.get("defects") or []:
        reason: str | None = None
        if defect.get("physical_defect_evidence") is not True:
            reason = "no_visible_physical_defect_evidence"
        elif defect.get("panorama_mask_or_blur_dominated") is True:
            reason = "panorama_mask_or_blur_dominated"
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

    guarded, old_mask_rejections = parent._apply_bottom_mask_guard({"defects": eligible})
    rejected.extend(old_mask_rejections)
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
        raise Stage7SurfacePrecisionV021Error("A Stage 7 verifier key is required for this tab.")
    if not image_path.is_file() or set(tiles) != {"left", "centre", "right"}:
        raise Stage7SurfacePrecisionV021Error("The exact current-view image set is incomplete.")
    if any(not path.is_file() for path in tiles.values()):
        raise Stage7SurfacePrecisionV021Error("A current-view detail image is unavailable.")
    failed_checks = [name for name, passed in checks().items() if not passed]
    if failed_checks:
        raise Stage7SurfacePrecisionV021Error(
            f"The Stage 7 v0.1.21 candidate is unavailable: {', '.join(failed_checks)}."
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
                    "strategy": "two_pass_recall_plus_surface_adjudication_v2",
                    "host_surface_gate": True,
                    "physical_defect_evidence": True,
                    "panorama_mask_or_blur_dominated": False,
                    "origin": defect["origin"],
                },
            }
        )

    usage = [item.get("usage") or {} for item in response_meta]
    run = {
        "run_id": f"stage7-surface-precision-v021-run-{uuid.uuid4().hex}",
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
            "strategy": "exact_current_view_two_pass_recall_plus_surface_adjudication_v2",
            "parent_version": PARENT_VERSION,
            "proposer_policy_sha256": PROPOSER_POLICY_SHA256,
            "adjudicator_policy_sha256": ADJUDICATOR_POLICY_SHA256,
            "proposer_schema_sha256": PROPOSER_SCHEMA_SHA256,
            "adjudicator_schema_sha256": ADJUDICATOR_SCHEMA_SHA256,
            "bottom_mask_guard_sha256": parent.BOTTOM_MASK_GUARD_SHA256,
            "proposal_box_count": len(proposals.get("defects") or []),
            "adjudicated_box_count": len(final.get("defects") or []),
            "retained_box_count": len(retained),
            "rejected_evidence_or_mask_box_count": len(rejected),
            "rejections": rejected,
            "recovered_miss_count": sum(1 for item in retained if item.get("origin") == "recovered_miss"),
            "frame_verdict": final.get("frame_verdict"),
            "eligible_surface_visible": final.get("eligible_surface_visible"),
            "rejected_common_distractors": final.get("rejected_common_distractors") or [],
            "summary": final.get("summary"),
            "provider_models": [item.get("model") for item in response_meta],
            "provider_response_ids": [item.get("response_id") for item in response_meta],
            "provider_usage": usage,
            "elapsed_seconds": time.perf_counter() - started,
            "provider_credential_persisted": False,
        },
    }
    return {"run": run, "detections": detections}
