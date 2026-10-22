from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_EVIDENCE = (
    PERCEPTION_ROOT
    / "stage7"
    / "evidence"
    / "s7_sp_unseen101_200_highrecall_maskguard_r01_adjudication.runtime.json"
)
ACCEPTANCE_EVIDENCE_SHA256 = "6ACF3D575E1305C9D9CF30B7D103C22E0EE874B442D6092E25FC00B346439454"
API_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-sol"
MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.20"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.20-candidate01"
PREVIOUS_ACTIVE_VERSION = "Stage7-R07-HighRecall-v0.1.19-candidate03"
TRANSIENT_HTTP = {408, 409, 429, 500, 502, 503, 504}
EIGHT_CLASSES = (
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_cracking",
    "uneven_or_broken_footpath",
    "broken_kerb",
    "standing_water",
    "debris_or_vegetation_obstruction",
)
SURFACES = ("road", "kerb", "footpath")
IMAGE_LABELS = ("exact", "left", "centre", "right")

SYSTEM_INSTRUCTIONS = """
You are a conservative road-infrastructure defect verifier for Stage 7 of a transport research platform.

The first image is the canonical exact Google Street View frame currently visible to the user. The next three images are higher-resolution left, centre and right views from the same panorama. Use the three detail views only to verify content that is also inside the canonical exact frame. Never report content visible only outside the canonical frame.

An eligible detection must be a human-visible physical defect located on road, kerb or footpath and must belong to exactly one of these frozen classes: pothole, longitudinal_crack, transverse_crack, alligator_cracking, uneven_or_broken_footpath, broken_kerb, standing_water, debris_or_vegetation_obstruction.

Do not report the road, kerb or footpath itself. Reject buildings, walls, roofs, fences, vehicles, people, vegetation that is not obstructing an eligible walking or road corridor, sky, signs, normal road markings, ordinary construction joints, shadows, stains and harmless texture. A clean frame must have an empty defects array. Do not invent a generic or visible-anomaly class.

For each retained defect, return a tight bounding box in the canonical exact image using integer coordinates from 0 to 1000: x increases left-to-right and y increases top-to-bottom. The box must cover the defect itself, not a whole surface. If a class is uncertain but a real eligible defect is clear, choose the most plausible frozen class and set tentative true. If the defect itself is not clear, return no detection.
""".strip()

PANORAMA_BOTTOM_MASK_GUARD = {
    "identity": "stage7-panorama-bottom-mask-guard-v1",
    "scope": "standing_water_only",
    "canonical_coordinates": "0..1000",
    "reject_if": {
        "defect_class": "standing_water",
        "canonical_box_y1_gte": 880,
        "canonical_box_y2_gte": 990,
    },
    "reason": (
        "A water proposal wholly confined to the textureless Google panorama bottom mask "
        "and touching the lower image boundary has no independently visible physical water support."
    ),
    "per_frame_coordinates_or_rules": False,
}


class Stage7SurfacePrecisionError(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def response_schema() -> dict[str, Any]:
    box_schema = {
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
    defect_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "defect_class": {"type": "string", "enum": list(EIGHT_CLASSES)},
            "surface_type": {"type": "string", "enum": list(SURFACES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "tentative": {"type": "boolean"},
            "canonical_box_0_1000": box_schema,
            "source_support": {
                "type": "array",
                "items": {"type": "string", "enum": list(IMAGE_LABELS)},
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "defect_class",
            "surface_type",
            "confidence",
            "tentative",
            "canonical_box_0_1000",
            "source_support",
            "evidence",
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
            "defects": {"type": "array", "items": defect_schema},
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


SEMANTIC_POLICY_SHA256 = hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest().upper()
RESPONSE_SCHEMA_SHA256 = _stable_hash(response_schema())
BOTTOM_MASK_GUARD_SHA256 = _stable_hash(PANORAMA_BOTTOM_MASK_GUARD)


def checks() -> dict[str, bool]:
    return {
        "surface_precision_adapter_available": Path(__file__).is_file(),
        "acceptance_evidence_verified": (
            ACCEPTANCE_EVIDENCE.is_file()
            and _sha256(ACCEPTANCE_EVIDENCE) == ACCEPTANCE_EVIDENCE_SHA256
        ),
        "semantic_policy_verified": (
            SEMANTIC_POLICY_SHA256
            == "D4E4895CD40F17E67030C954C6F64CA77F52A686472F8740A43E76A4C7066399"
        ),
        "response_schema_verified": (
            RESPONSE_SCHEMA_SHA256
            == "4FC1157F6241E8F001C6F813613944D91D4837EDF869B79B9046D2E36D6002EA"
        ),
        "bottom_mask_guard_verified": (
            BOTTOM_MASK_GUARD_SHA256
            == "97C76B8B7A51F9D3D0B1B02C57962A6578CC93DCD1F8099E91166AB48FE0BD27"
        ),
        "frozen_taxonomy_verified": len(EIGHT_CLASSES) == 8 and len(set(EIGHT_CLASSES)) == 8,
        "surface_contract_verified": set(SURFACES) == {"road", "kerb", "footpath"},
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": _sha256(Path(__file__)),
        "semantic_policy": SEMANTIC_POLICY_SHA256,
        "response_schema": RESPONSE_SCHEMA_SHA256,
        "panorama_bottom_mask_guard": BOTTOM_MASK_GUARD_SHA256,
        "blind_acceptance": ACCEPTANCE_EVIDENCE_SHA256,
    }
    composite = hashlib.sha256(
        (":".join(components.values()) + f":{DEFAULT_MODEL}:exact-current-view-four-image-v1").encode("ascii")
    ).hexdigest().upper()
    return {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "checkpoint_sha256": composite,
        "components": components,
        "activation_state": "registry_controlled",
        "rollback_version": PREVIOUS_ACTIVE_VERSION,
    }


def _image_content(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        with Image.open(path) as image:
            image_format = str(image.format or "JPEG").upper()
    except OSError as error:
        raise Stage7SurfacePrecisionError("A Stage 7 current-view image is unreadable.") from error
    mime = "image/png" if image_format == "PNG" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "input_text", "text": f"Image label: {label}"},
        {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{encoded}",
            "detail": "original",
        },
    ]


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and str(content.get("text") or "").strip():
                return str(content["text"])
    raise Stage7SurfacePrecisionError("The Stage 7 verifier returned no structured output.")


def _sanitised_http_error(error: HTTPError) -> Stage7SurfacePrecisionError:
    message = f"The Stage 7 verifier returned HTTP {int(error.code)}."
    try:
        body = json.loads(error.read().decode("utf-8", errors="replace"))
        detail = str(((body.get("error") or {}).get("message") or "")).strip()
        if detail:
            detail = re.sub(r"(?:sk-proj-|sk-)[A-Za-z0-9_-]{12,}", "[redacted]", detail)
            message = f"{message} {detail[:400]}"
    except Exception:
        pass
    return Stage7SurfacePrecisionError(message)


def _call_openai(
    *,
    api_key: str,
    frame_identity: str,
    exact: Path,
    tiles: dict[str, Path],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"Analyse exact current-view frame {frame_identity}. Apply the frozen policy. "
                "The exact image is canonical; return only canonical-image boxes."
            ),
        }
    ]
    content.extend(_image_content(exact, "exact"))
    for label in ("left", "centre", "right"):
        content.extend(_image_content(tiles[label], label))
    request_body = {
        "model": DEFAULT_MODEL,
        "store": False,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 1800,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "stage7_surface_verifier",
                "strict": True,
                "schema": response_schema(),
            }
        },
    }
    request = Request(
        API_ENDPOINT,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.8, 2.0, 4.0)):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result = json.loads(_extract_output_text(payload))
            return result, {
                "response_id": payload.get("id"),
                "model": payload.get("model") or DEFAULT_MODEL,
                "usage": payload.get("usage") or {},
            }
        except HTTPError as error:
            last_error = _sanitised_http_error(error)
            if int(error.code) not in TRANSIENT_HTTP or attempt == 2:
                raise last_error from None
        except (TimeoutError, URLError, OSError, json.JSONDecodeError) as error:
            last_error = Stage7SurfacePrecisionError(
                f"The Stage 7 verifier transport failed ({type(error).__name__})."
            )
            if attempt == 2:
                raise last_error from None
        time.sleep(delay)
    raise last_error or Stage7SurfacePrecisionError("The Stage 7 verifier failed.")


def _validate_result(result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid result object.")
    verdict = result.get("frame_verdict")
    if verdict not in {"positive", "clean", "no_eligible_surface", "uncertain"}:
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid frame verdict.")
    if not isinstance(result.get("eligible_surface_visible"), bool):
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned invalid surface visibility.")
    defects = result.get("defects")
    if not isinstance(defects, list):
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned no defects array.")
    if verdict in {"clean", "no_eligible_surface"} and defects:
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an inconsistent verdict.")
    if verdict == "positive" and not defects:
        raise Stage7SurfacePrecisionError("The Stage 7 verifier returned a positive verdict without a defect.")
    for defect in defects:
        if not isinstance(defect, dict):
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid defect object.")
        if defect.get("defect_class") not in EIGHT_CLASSES:
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned a non-frozen class.")
        if defect.get("surface_type") not in SURFACES:
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid host surface.")
        confidence = defect.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned invalid confidence.")
        if not isinstance(defect.get("tentative"), bool):
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid tentative flag.")
        support = defect.get("source_support")
        if (
            not isinstance(support, list)
            or not support
            or "exact" not in support
            or any(item not in IMAGE_LABELS for item in support)
        ):
            raise Stage7SurfacePrecisionError(
                "The Stage 7 verifier returned a defect without canonical-frame support."
            )
        if not isinstance(defect.get("evidence"), str):
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned invalid defect evidence.")
        box = defect.get("canonical_box_0_1000") or {}
        if not (
            isinstance(box.get("x1"), int)
            and isinstance(box.get("y1"), int)
            and isinstance(box.get("x2"), int)
            and isinstance(box.get("y2"), int)
            and 0 <= box["x1"] < box["x2"] <= 1000
            and 0 <= box["y1"] < box["y2"] <= 1000
        ):
            raise Stage7SurfacePrecisionError("The Stage 7 verifier returned an invalid box.")


def _apply_bottom_mask_guard(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for defect in result.get("defects") or []:
        box = defect["canonical_box_0_1000"]
        if (
            defect["defect_class"] == "standing_water"
            and box["y1"] >= 880
            and box["y2"] >= 990
        ):
            rejected.append(
                {
                    "defect_class": defect["defect_class"],
                    "surface_type": defect["surface_type"],
                    "canonical_box_0_1000": dict(box),
                    "reason": "panorama_bottom_mask_guard",
                }
            )
            continue
        retained.append(defect)
    return retained, rejected


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
        raise Stage7SurfacePrecisionError("A Stage 7 verifier key is required for this tab.")
    if not image_path.is_file() or set(tiles) != {"left", "centre", "right"}:
        raise Stage7SurfacePrecisionError("The exact current-view image set is incomplete.")
    if any(not path.is_file() for path in tiles.values()):
        raise Stage7SurfacePrecisionError("A current-view detail image is unavailable.")
    failed_checks = [name for name, passed in checks().items() if not passed]
    if failed_checks:
        raise Stage7SurfacePrecisionError(
            f"The Stage 7 surface-precision package is unavailable: {', '.join(failed_checks)}."
        )
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        result, response_meta = _call_openai(
            api_key=cleaned_key,
            frame_identity=frame_identity,
            exact=image_path,
            tiles=tiles,
            timeout=timeout,
        )
    finally:
        cleaned_key = ""
    _validate_result(result)
    retained, rejected = _apply_bottom_mask_guard(result)
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
                    "strategy": "exact_current_view_four_image_surface_verifier_v1",
                    "host_surface_gate": True,
                },
            }
        )
    run = {
        "run_id": f"stage7-surface-precision-run-{uuid.uuid4().hex}",
        "origin_stage": 7,
        "started_at": started_at,
        "completed_at": _timestamp(),
        "frame_identity": frame_identity,
        "image_sha256": _sha256(image_path),
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
            "strategy": "exact_current_view_four_image_surface_verifier_v1",
            "semantic_policy_sha256": SEMANTIC_POLICY_SHA256,
            "response_schema_sha256": RESPONSE_SCHEMA_SHA256,
            "bottom_mask_guard_sha256": BOTTOM_MASK_GUARD_SHA256,
            "raw_box_count": len(result.get("defects") or []),
            "retained_box_count": len(retained),
            "rejected_bottom_mask_box_count": len(rejected),
            "rejections": rejected,
            "frame_verdict": result.get("frame_verdict"),
            "eligible_surface_visible": result.get("eligible_surface_visible"),
            "rejected_common_distractors": result.get("rejected_common_distractors") or [],
            "summary": result.get("summary"),
            "provider_model": response_meta.get("model"),
            "provider_response_id": response_meta.get("response_id"),
            "provider_usage": response_meta.get("usage") or {},
            "elapsed_seconds": time.perf_counter() - started,
            "provider_credential_persisted": False,
        },
    }
    return {"run": run, "detections": detections}
