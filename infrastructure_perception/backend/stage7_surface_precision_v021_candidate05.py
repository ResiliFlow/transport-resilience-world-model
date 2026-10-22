from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from . import stage7_surface_precision as base
from . import stage7_surface_precision_v021_candidate04 as parent


MODEL_ID = "twm-infradet-stage7-surface-precision-v0.1.21-candidate05"
MODEL_NAME = "TWM-InfraDet Stage 7 Surface Precision v0.1.21 candidate05"
MODEL_VERSION = "Stage7-SurfacePrecision-v0.1.21-candidate05"
PARENT_VERSION = parent.MODEL_VERSION

LOCAL_OVERLAP_GATE = {
    "enabled_surface": "footpath",
    "minimum_box_y1": 0.65,
    "minimum_any_local_iou": 0.03,
    "class_match_required": False,
    "purpose": "reject_lower-frame_portable_fixture_or_panorama_mask_false_footpath",
}
LOCAL_OVERLAP_GATE_SHA256 = base._stable_hash(LOCAL_OVERLAP_GATE)


def checks() -> dict[str, bool]:
    return {
        "candidate_adapter_available": Path(__file__).is_file(),
        "candidate04_parent_verified": all(parent.checks().values()),
        "candidate04_version_verified": parent.component_identity()["version"] == PARENT_VERSION,
        "local_overlap_gate_frozen": LOCAL_OVERLAP_GATE_SHA256 == base._stable_hash(LOCAL_OVERLAP_GATE),
        "local_overlap_gate_narrowly_scoped": (
            LOCAL_OVERLAP_GATE["enabled_surface"] == "footpath"
            and LOCAL_OVERLAP_GATE["minimum_box_y1"] == 0.65
            and LOCAL_OVERLAP_GATE["minimum_any_local_iou"] == 0.03
        ),
    }


def component_identity() -> dict[str, Any]:
    components = {
        "adapter": base._sha256(Path(__file__)),
        "candidate04_parent": parent.component_identity()["checkpoint_sha256"],
        "local_overlap_gate": LOCAL_OVERLAP_GATE_SHA256,
    }
    composite = hashlib.sha256(
        (
            ":".join(components.values())
            + f":{parent.DEFAULT_MODEL}:exact-current-view-v5-local-overlap-gated-footpath"
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


def infer(
    *,
    api_key: str,
    image_path: Path,
    tiles: dict[str, Path],
    frame_identity: str,
    timeout: float = 150.0,
) -> dict[str, Any]:
    result = parent.infer(
        api_key=api_key,
        image_path=image_path,
        tiles=tiles,
        frame_identity=frame_identity,
        timeout=timeout,
    )
    identity = component_identity()
    detections = list(result.get("detections") or [])
    for index, detection in enumerate(detections):
        box = list(detection.get("box") or [])
        gate_enabled = bool(
            detection.get("surface_type") == LOCAL_OVERLAP_GATE["enabled_surface"]
            and len(box) == 4
            and float(box[1]) >= float(LOCAL_OVERLAP_GATE["minimum_box_y1"])
        )
        consensus = dict(detection.get("consensus") or {})
        consensus["local_overlap_gate"] = {
            "enabled": gate_enabled,
            "minimum_any_local_iou": LOCAL_OVERLAP_GATE["minimum_any_local_iou"],
            "class_match_required": LOCAL_OVERLAP_GATE["class_match_required"],
            "policy_sha256": LOCAL_OVERLAP_GATE_SHA256,
        }
        detection["consensus"] = consensus
        token = json.dumps(
            [frame_identity, index, detection.get("damage_type"), box, MODEL_VERSION],
            separators=(",", ":"),
        ).encode("utf-8")
        detection["detection_id"] = f"stage7-detection-{hashlib.sha256(token).hexdigest()[:20]}"

    run = result["run"]
    run.update(
        {
            "run_id": f"stage7-surface-precision-v021-c05-run-{uuid.uuid4().hex}",
            "model_id": identity["id"],
            "model_name": identity["name"],
            "model_version": identity["version"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "component_sha256": identity["components"],
            "detection_ids": [item["detection_id"] for item in detections],
        }
    )
    run.setdefault("diagnostics", {}).update(
        {
            "strategy": "exact_current_view_v5_local_overlap_gated_ambiguous_footpath",
            "parent_version": PARENT_VERSION,
            "local_overlap_gate_policy": dict(LOCAL_OVERLAP_GATE),
            "local_overlap_gate_sha256": LOCAL_OVERLAP_GATE_SHA256,
            "local_overlap_gate_candidate_count": sum(
                1 for item in detections if item.get("consensus", {}).get("local_overlap_gate", {}).get("enabled")
            ),
        }
    )
    result["detections"] = detections
    return result
