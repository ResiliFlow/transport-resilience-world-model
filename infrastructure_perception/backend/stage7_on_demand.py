from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request as UrlRequest, urlopen

from PIL import Image, UnidentifiedImageError

from .stage6_deployment import Stage6Deployment, Stage6DeploymentError
from .stage7_ensemble import stage7_ensemble
from . import stage7_classless_v022 as classless_v022
from . import stage7_classless_local_v023 as classless_local_v023
from . import stage7_classless_local_v024 as classless_local_v024
from .stage7_surface_precision import (
    Stage7SurfacePrecisionError,
    infer as infer_surface_precision,
)
from .stage7_surface_precision_v021 import infer as infer_surface_precision_v021
from .stage7_surface_precision_v021_candidate02 import (
    infer as infer_surface_precision_v021_candidate02,
)
from .stage7_surface_precision_v021_candidate03 import (
    infer as infer_surface_precision_v021_candidate03,
)
from .stage7_surface_precision_v021_candidate04 import (
    infer as infer_surface_precision_v021_candidate04,
)
from .stage7_surface_precision_v021_candidate05 import (
    infer as infer_surface_precision_v021_candidate05,
)
from .stage7_surface_precision_v021_candidate06 import (
    infer as infer_surface_precision_v021_candidate06,
)
from .stage7_surface_precision_v021_candidate07 import (
    infer as infer_surface_precision_v021_candidate07,
)


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = PERCEPTION_ROOT / "contracts" / "stage7_on_demand_contract.json"
DEFAULT_STORAGE_ROOT = PERCEPTION_ROOT / "storage"
OUTBOUND_USER_AGENT = "ResiliFlow-TWM-Stage7/1.0 (https://www.unsw.edu.au/research/rciti; research prototype)"
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
NETWORK_TIMEOUT_SECONDS = 25
GOOGLE_STREET_VIEW_STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"
KARTAVIEW_ATTRIBUTION = "© Grab and KartaView Contributors · CC BY-SA 4.0"
KARTAVIEW_PERMISSION_RECORD = "kartaview-terms-2025-06-17-cc-by-sa-4.0"
R58_V021_SHADOW_FLAG = PERCEPTION_ROOT.parent / ".runtime" / "stage7_r58_v021_shadow.enabled"


class Stage7Error(RuntimeError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage7Error(f"The Stage 7 file is unavailable or invalid: {path.name}") from error
    if not isinstance(payload, dict):
        raise Stage7Error(f"The Stage 7 file is not a JSON object: {path.name}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _empty_activity() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": None, "analyses": []}


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _fuse_local_and_multimodal(
    local_inference: dict[str, Any] | None,
    multimodal_inference: dict[str, Any],
    *,
    local_error: str | None = None,
) -> dict[str, Any]:
    """Keep semantic-verifier boxes while recording same-frame local support."""
    fused = copy.deepcopy(multimodal_inference)
    local_detections = list((local_inference or {}).get("detections") or [])
    multimodal_detections = list(fused.get("detections") or [])
    supported_count = 0
    gated_out: list[dict[str, Any]] = []
    retained_multimodal: list[dict[str, Any]] = []
    for detection in multimodal_detections:
        box = detection.get("box") or []
        overlaps: list[tuple[float, dict[str, Any]]] = []
        if isinstance(box, list) and len(box) == 4:
            for local_detection in local_detections:
                local_box = local_detection.get("box") or []
                if not isinstance(local_box, list) or len(local_box) != 4:
                    continue
                overlaps.append((_box_iou(box, local_box), local_detection))
        overlaps.sort(key=lambda item: item[0], reverse=True)
        best_iou = overlaps[0][0] if overlaps else 0.0
        supporting = [
            candidate
            for iou, candidate in overlaps
            if iou >= 0.05
            and candidate.get("damage_type")
            in {detection.get("damage_type"), "visible_road_anomaly"}
        ]
        if supporting:
            supported_count += 1
        consensus = dict(detection.get("consensus") or {})
        consensus.update(
            {
                "strategy": "local_r07_plus_openai_surface_verifier_v1",
                "local_model_support": bool(supporting),
                "local_best_iou": best_iou,
                "local_detection_ids": [item.get("detection_id") for item in supporting],
            }
        )
        detection["consensus"] = consensus
        local_overlap_gate = consensus.get("local_overlap_gate") or {}
        if (
            local_overlap_gate.get("enabled") is True
            and best_iou < float(local_overlap_gate.get("minimum_any_local_iou", 0.0))
        ):
            gated_out.append(
                {
                    "detection_id": detection.get("detection_id"),
                    "damage_type": detection.get("damage_type"),
                    "surface_type": detection.get("surface_type"),
                    "box": list(detection.get("box") or []),
                    "local_best_iou": best_iou,
                    "reason": "candidate_opt_in_local_overlap_gate",
                }
            )
            continue
        retained_multimodal.append(detection)
    fused["detections"] = retained_multimodal
    run = fused.setdefault("run", {})
    run["detection_ids"] = [item.get("detection_id") for item in retained_multimodal]
    run["result_type"] = "detections" if retained_multimodal else "no_detection"
    diagnostics = run.setdefault("diagnostics", {})
    diagnostics.update(
        {
            "inference_mode": "enhanced",
            "strategy": "local_r07_plus_openai_surface_verifier_v1",
            "local_model_version": (local_inference or {}).get("run", {}).get("model_version"),
            "local_checkpoint_sha256": (local_inference or {}).get("run", {}).get("checkpoint_sha256"),
            "local_detection_count": len(local_detections),
            "multimodal_detection_count": len(multimodal_detections),
            "retained_multimodal_detection_count": len(retained_multimodal),
            "local_overlap_gate_rejected_count": len(gated_out),
            "local_overlap_gate_rejections": gated_out,
            "multimodal_detections_with_local_support": supported_count,
            "local_inference_error": local_error,
            "final_box_policy": "multimodal_surface_verified_boxes_with_same_frame_local_support_metadata",
        }
    )
    return fused


class Stage7OnDemand:
    def __init__(
        self,
        *,
        contract_path: Path = DEFAULT_CONTRACT_PATH,
        storage_root: Path = DEFAULT_STORAGE_ROOT,
    ) -> None:
        self.contract_path = Path(contract_path)
        self.storage_root = Path(storage_root)
        self.activity_path = self.storage_root / "stage7_activity.json"
        self.runtime_root = self.storage_root / "stage7_runtime"
        self._activity_lock = threading.RLock()

    def contract(self) -> dict[str, Any]:
        contract = _load_json(self.contract_path)
        if contract.get("schema_version") != 1 or contract.get("stage") != 7:
            raise Stage7Error("The Stage 7 on-demand contract has an invalid identity.")
        return contract

    def status(self, stage6_status: dict[str, Any]) -> dict[str, Any]:
        contract = self.contract()
        active = stage6_status.get("active_model") if isinstance(stage6_status, dict) else {}
        enhanced_checks = stage7_ensemble.checks()
        enhanced_identity = stage7_ensemble.component_identity()
        local_adapter = (
            classless_local_v024 if classless_local_v024.active("local")
            else classless_local_v023 if classless_local_v023.active("local")
            else classless_v022
        )
        classless_local = local_adapter.active("local")
        classless_runtime = local_adapter.worker.warmup() if classless_local else None
        local_checks = local_adapter.checks("local") if classless_local else stage7_ensemble.checks(candidate=True)
        local_identity = local_adapter.component_identity("local") if classless_local else stage7_ensemble.component_identity(candidate=True)
        stage6_ready = bool(
            isinstance(active, dict)
            and active.get("package_ready") is True
            and stage6_status.get("status") == "research_deployment_ready"
        )
        local_ready = stage6_ready and all(local_checks.values())
        enhanced_ready = stage6_ready and all(enhanced_checks.values())
        return {
            "stage": 7,
            "status": "on_demand_runtime_ready" if local_ready else "blocked",
            "contract_id": contract["contract_id"],
            "contract_status": contract.get("status"),
            "acceptance_status": contract.get("acceptance", {}).get("status"),
            "contract_sha256": _sha256(self.contract_path),
            "accepted_result_states": [item["id"] for item in contract["accepted_result_states"]],
            "active_model": enhanced_identity["version"] if enhanced_ready else local_identity["version"],
            "active_checkpoint_sha256": enhanced_identity["checkpoint_sha256"] if enhanced_ready else local_identity["checkpoint_sha256"],
            "base_stage6_model": active.get("model_version") if isinstance(active, dict) else None,
            "ensemble_components": enhanced_identity["components"],
            "ensemble_checks": enhanced_checks,
            "inference_modes": {
                "local": {
                    "ready": local_ready,
                    "default": True,
                    "requires_openai_key": False,
                    "model_version": local_identity["version"],
                    "checkpoint_sha256": local_identity["checkpoint_sha256"],
                    "checks": local_checks,
                    "display_class_names": False if classless_local else True,
                    "worker": classless_runtime,
                },
                "enhanced": {
                    "ready": enhanced_ready,
                    "default": False,
                    "requires_user_openai_key": True,
                    "model_version": enhanced_identity["version"],
                    "checkpoint_sha256": enhanced_identity["checkpoint_sha256"],
                    "checks": enhanced_checks,
                },
            },
            "credential_isolation": {
                "assistant_session_lookup": False,
                "assistant_credential_reuse": False,
                "server_openai_key_fallback": False,
                "user_key_persisted_or_logged": False,
                "enhanced_key_scope": "single_perception_tab_request_memory_only",
            },
            "target_area_detection_proof_confirmed": contract.get("status") == "exact_current_view_accepted",
            "completion_boundary_met": contract.get("status") == "exact_current_view_accepted",
            "arbitrary_point_image_coverage_guaranteed": False,
            "provider_capabilities": {
                "kartaview_cc_by_sa": "enabled_with_attribution_and_share_alike_controls",
                "wikimedia_commons_open": "enabled_for_geotagged_road_files_with_verified_open_licences",
                "google_current_panorama_frame": "enabled_for_user_initiated_exact_current_view_inference",
                "google_street_view_insights": "not_onboarded",
            },
            "guarantees": {
                "empty_response_allowed": False,
                "explicit_analysis_state": True,
                "temporary_provider_pixels_deleted": True,
                "nearby_image_substitution": False,
                "surface_metadata_required_for_exact_view": False,
                "human_review_required": True,
                "automatic_twm_transfer": False,
                "automatic_operational_action": False,
            },
            "completion_boundary": contract["acceptance"]["completion_rule"],
        }

    @staticmethod
    def _approved_provider_url(value: Any, source_id: str) -> str:
        url = str(value or "").strip()
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if source_id == "kartaview_cc_by_sa":
            approved = (
                hostname == "kartaview.org"
                or hostname.endswith(".kartaview.org")
                or hostname == "openstreetcam.org"
                or hostname.endswith(".openstreetcam.org")
            )
        elif source_id == "wikimedia_commons_open":
            approved = hostname == "upload.wikimedia.org"
        else:
            approved = False
        if parsed.scheme != "https" or not approved or parsed.username or parsed.password:
            raise Stage7Error("The selected provider image URL is not an approved HTTPS source.")
        return url

    @staticmethod
    def _approved_kartaview_url(value: Any) -> str:
        return Stage7OnDemand._approved_provider_url(value, "kartaview_cc_by_sa")

    def _download_image(self, image_url: str, source_id: str = "kartaview_cc_by_sa") -> tuple[Path, dict[str, Any]]:
        approved_url = self._approved_provider_url(image_url, source_id)
        request = UrlRequest(
            approved_url,
            headers={"User-Agent": OUTBOUND_USER_AGENT, "Accept": "image/jpeg,image/png,image/webp"},
            method="GET",
        )
        body = b""
        content_type = ""
        for attempt in range(3):
            try:
                with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                    self._approved_provider_url(response.geturl(), source_id)
                    declared = int(response.headers.get("Content-Length") or 0)
                    content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
                    if declared > MAX_IMAGE_BYTES:
                        raise Stage7Error("The selected provider image exceeds the 30 MB safety limit.")
                    body = response.read(MAX_IMAGE_BYTES + 1)
                break
            except Stage7Error:
                raise
            except HTTPError as error:
                if error.code == 429 and attempt < 2:
                    retry_header = str(error.headers.get("Retry-After") or "").strip() if error.headers else ""
                    try:
                        retry_seconds = float(retry_header)
                    except ValueError:
                        retry_seconds = 6.0 * (attempt + 1)
                    time.sleep(min(15.0, max(6.0, retry_seconds)))
                    continue
                raise Stage7Error(f"The selected provider returned HTTP {error.code} for the image.") from error
            except (URLError, TimeoutError, OSError) as error:
                raise Stage7Error("The selected provider image is temporarily unavailable.") from error
        if len(body) > MAX_IMAGE_BYTES:
            raise Stage7Error("The selected provider image exceeds the 30 MB safety limit.")
        if not body:
            raise Stage7Error("The selected provider image was empty.")
        try:
            with Image.open(io.BytesIO(body)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise Stage7Error("The selected provider asset is not a readable image.") from error
        if width < 32 or height < 32 or width * height > MAX_IMAGE_PIXELS:
            raise Stage7Error("The selected provider image dimensions are outside the safe inference range.")
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
        if suffix is None or (content_type and not content_type.startswith("image/")):
            raise Stage7Error("The selected provider asset has an unsupported image type.")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        prefix = "stage7-commons-" if source_id == "wikimedia_commons_open" else "stage7-kartaview-"
        descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.runtime_root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return Path(temporary_name), {
            "byte_count": len(body),
            "width": width,
            "height": height,
            "format": image_format,
        }

    def _download_google_frame(
        self,
        *,
        api_key: str,
        pano_id: str,
        heading: float,
        pitch: float,
        fov: float,
        width: int,
        height: int,
        referrer: str | None,
    ) -> tuple[Path, dict[str, Any]]:
        """Acquire one exact camera frame; the credential is never returned or persisted."""

        query = urlencode(
            {
                "size": f"{width}x{height}",
                "pano": pano_id,
                "heading": f"{heading:.6f}",
                "pitch": f"{pitch:.6f}",
                "fov": f"{fov:.6f}",
                "return_error_code": "true",
                "key": api_key,
            }
        )
        headers = {
            "User-Agent": OUTBOUND_USER_AGENT,
            "Accept": "image/jpeg,image/png",
        }
        if referrer:
            headers["Referer"] = referrer
        request = UrlRequest(f"{GOOGLE_STREET_VIEW_STATIC_URL}?{query}", headers=headers, method="GET")
        try:
            with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                declared = int(response.headers.get("Content-Length") or 0)
                content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
                if declared > MAX_IMAGE_BYTES:
                    raise Stage7Error("The current Google panorama frame exceeds the 30 MB safety limit.")
                body = response.read(MAX_IMAGE_BYTES + 1)
        except HTTPError as error:
            try:
                google_detail = error.read(8192).decode("utf-8", errors="replace")
            except Exception:
                google_detail = ""
            google_detail = google_detail.replace(api_key, "[redacted]")
            google_detail = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", google_detail)
            google_detail = " ".join(google_detail.split())[:700]
            suffix = f" Google response: {google_detail}" if google_detail else ""
            raise Stage7Error(
                f"Google Street View Static API returned HTTP {error.code}.{suffix}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise Stage7Error("The exact current Google panorama frame is temporarily unavailable.") from error
        if len(body) > MAX_IMAGE_BYTES:
            raise Stage7Error("The current Google panorama frame exceeds the 30 MB safety limit.")
        if not body:
            raise Stage7Error("Google returned an empty current panorama frame.")
        try:
            with Image.open(io.BytesIO(body)) as image:
                image_format = str(image.format or "").upper()
                observed_width, observed_height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise Stage7Error("Google did not return a readable image for the current panorama frame.") from error
        suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image_format)
        if (
            suffix is None
            or (content_type and not content_type.startswith("image/"))
            or observed_width < 32
            or observed_height < 32
            or observed_width * observed_height > MAX_IMAGE_PIXELS
        ):
            raise Stage7Error("The current Google panorama frame has an unsupported image response.")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="stage7-google-current-frame-",
            suffix=suffix,
            dir=self.runtime_root,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return Path(temporary_name), {
            "byte_count": len(body),
            "width": observed_width,
            "height": observed_height,
            "format": image_format,
            "preview_data_url": f"data:{content_type or 'image/jpeg'};base64,{base64.b64encode(body).decode('ascii')}",
        }

    def _activity(self) -> dict[str, Any]:
        if not self.activity_path.exists():
            return _empty_activity()
        payload = _load_json(self.activity_path)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("analyses"), list):
            raise Stage7Error("The Stage 7 activity store is invalid.")
        return payload

    def _record(self, response: dict[str, Any]) -> None:
        imagery = dict(response.get("imagery") or {})
        imagery.pop("preview_data_url", None)
        compact = {
            "analysis_id": response["analysis_id"],
            "requested_at": response["requested_at"],
            "completed_at": response["completed_at"],
            "selected_point": response["selected_point"],
            "requested_provider": response["requested_provider"],
            "inference_mode": response.get("inference_mode"),
            "state": response["state"],
            "reason_codes": list(response.get("reason_codes") or []),
            "message": response["message"],
            "imagery": imagery or None,
            "model": response.get("model"),
            "diagnostics": response.get("diagnostics") or {},
            "observation_ids": [item.get("observation_id") for item in response.get("observations") or []],
            "detection_ids": [item.get("detection_id") for item in response.get("detections") or []],
            "frame": response.get("frame"),
            "automatic_twm_transfer": False,
            "automatic_operational_action": False,
        }
        with self._activity_lock:
            activity = self._activity()
            activity["analyses"].append(compact)
            activity["updated_at"] = response["completed_at"]
            _atomic_json(self.activity_path, activity)

    def activity(self, *, limit: int = 100) -> dict[str, Any]:
        with self._activity_lock:
            activity = self._activity()
        records = list(reversed(activity["analyses"][-limit:]))
        return {
            "updated_at": activity["updated_at"],
            "analysis_count": len(activity["analyses"]),
            "returned_count": len(records),
            "analyses": records,
        }

    def _response(
        self,
        *,
        analysis_id: str,
        requested_at: str,
        latitude: float,
        longitude: float,
        radius_m: int,
        provider: str,
        state: str,
        message: str,
        reason_codes: list[str] | None = None,
        imagery: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        observations: list[dict[str, Any]] | None = None,
        inference_run: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = {
            "analysis_id": analysis_id,
            "requested_at": requested_at,
            "completed_at": _timestamp(),
            "selected_point": {"latitude": latitude, "longitude": longitude, "radius_m": radius_m},
            "requested_provider": provider,
            "state": state,
            "processed": state in {"detections_found_review_required", "analysis_complete_no_supported_detection"},
            "message": message,
            "reason_codes": reason_codes or [],
            "imagery": imagery,
            "model": model,
            "observations": observations or [],
            "inference_run": inference_run,
            "diagnostics": diagnostics or {},
            "human_review_required": bool(observations),
            "automatic_twm_transfer": False,
            "automatic_operational_action": False,
        }
        self._record(response)
        return response

    def _current_view_response(
        self,
        *,
        analysis_id: str,
        requested_at: str,
        frame: dict[str, Any],
        state: str,
        message: str,
        reason_codes: list[str] | None = None,
        imagery: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        detections: list[dict[str, Any]] | None = None,
        inference_run: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        inference_mode: str = "local",
    ) -> dict[str, Any]:
        detections = detections or []
        response = {
            "analysis_id": analysis_id,
            "requested_at": requested_at,
            "completed_at": _timestamp(),
            "selected_point": {
                "latitude": frame["latitude"],
                "longitude": frame["longitude"],
            },
            "requested_provider": "google_current_panorama_frame",
            "inference_mode": inference_mode,
            "frame": frame,
            "state": state,
            "processed": state in {
                "detections_found_review_required",
                "analysis_complete_no_supported_detection",
            },
            "message": message,
            "reason_codes": reason_codes or [],
            "imagery": imagery,
            "model": model,
            "detections": detections,
            "observations": [],
            "inference_run": inference_run,
            "diagnostics": diagnostics or {},
            "human_review_required": bool(detections),
            "surface_metadata_used": False,
            "nearby_image_substitution_used": False,
            "provider_credential_persisted": False,
            "automatic_twm_transfer": False,
            "automatic_operational_action": False,
        }
        self._record(response)
        return response

    def analyse_current_view(
        self,
        *,
        api_key: str,
        stage7_api_key: str | None,
        inference_mode: str,
        frame_id: str,
        pano_id: str,
        latitude: float,
        longitude: float,
        heading: float,
        pitch: float,
        zoom: float,
        fov: float,
        width: int,
        height: int,
        viewport_width: int,
        viewport_height: int,
        frame_revision: int,
        referrer: str | None,
        deployment: Stage6Deployment,
        candidate_model: str | None = None,
    ) -> dict[str, Any]:
        analysis_id = f"stage7-current-view-{uuid.uuid4().hex}"
        requested_at = _timestamp()
        frame = {
            "frame_id": frame_id,
            "pano_id": pano_id,
            "latitude": latitude,
            "longitude": longitude,
            "heading": heading,
            "pitch": pitch,
            "zoom": zoom,
            "fov": fov,
            "width": width,
            "height": height,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
            "frame_revision": frame_revision,
        }
        temporary: Path | None = None
        detail_temporaries: list[Path] = []
        imagery: dict[str, Any] | None = None
        flag_shadow_v021 = (
            candidate_model is None
            and inference_mode == "enhanced"
            and R58_V021_SHADOW_FLAG.is_file()
        )
        active_surface_precision_v021_candidate06 = (
            candidate_model is None
            and inference_mode == "enhanced"
            and stage7_ensemble.active_mode() == "surface_precision_v021_candidate06"
        )
        shadow_v021 = (
            candidate_model == "surface_precision_v021"
            or flag_shadow_v021
            or active_surface_precision_v021_candidate06
        )
        selected_mode = (
            "enhanced"
            if candidate_model in {"surface_precision", "surface_precision_v021"} or shadow_v021
            else inference_mode
        )
        if candidate_model in {"r07", "classless_local_v023", "classless_local_v024"}:
            selected_mode = "local"
        local_v024_selected = selected_mode == "local" and (
            candidate_model == "classless_local_v024"
            or (candidate_model is None and classless_local_v024.active("local"))
        )
        local_v023_selected = selected_mode == "local" and (
            candidate_model == "classless_local_v023"
            or (candidate_model is None and classless_local_v023.active("local"))
        )
        classless_selected = local_v024_selected or local_v023_selected or candidate_model == "classless_v022" or (
            candidate_model is None and classless_v022.active(selected_mode)
        )
        classless_adapter = (
            classless_local_v024 if local_v024_selected
            else classless_local_v023 if local_v023_selected else classless_v022
        )
        try:
            temporary, image_meta = self._download_google_frame(
                api_key=api_key,
                pano_id=pano_id,
                heading=heading,
                pitch=pitch,
                fov=fov,
                width=width,
                height=height,
                referrer=referrer,
            )
            imagery = {
                "provider": "Google Street View",
                "source_id": "google_current_panorama_frame",
                "pano_id": pano_id,
                "frame_id": frame_id,
                "latitude": latitude,
                "longitude": longitude,
                "details_url": f"https://www.google.com/maps?q={latitude:.7f},{longitude:.7f}&layer=c",
                "attribution": "Google Street View",
                "licence": "Google Maps Platform terms",
                "temporary": True,
                **image_meta,
            }
            if classless_selected:
                tile_fov = max(10.0, min(60.0, fov * 4.0 / 9.0))
                tile_offset = fov / 3.0
                detail_tiles = {}
                download_error = None
                # Three views of this exact panorama/camera, never nearby substitutes.
                # All successful temporary files are tracked even if another request fails.
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {
                        name: pool.submit(self._download_google_frame,
                            api_key=api_key, pano_id=pano_id,
                            heading=(heading + offset) % 360.0, pitch=pitch,
                            fov=tile_fov, width=640, height=640, referrer=referrer)
                        for name, offset in (("left", -tile_offset), ("centre", 0.0), ("right", tile_offset))
                    }
                    for name, future in futures.items():
                        try:
                            path, _meta = future.result()
                            detail_temporaries.append(path)
                            detail_tiles[name] = path
                        except Exception as error:
                            download_error = error
                if download_error is not None:
                    if isinstance(download_error, Stage7Error):
                        raise download_error
                    raise Stage7Error("The exact-camera detail images could not be acquired.") from None
                inference = classless_adapter.infer(
                    image_path=temporary, tiles=detail_tiles, camera=frame,
                    frame_identity=frame_id, mode=selected_mode,
                    api_key=stage7_api_key if selected_mode == "enhanced" else None,
                )
                inference["run"]["diagnostics"].update(
                    exact_current_view_only=True, canonical_fov=fov, detail_fov=tile_fov,
                    detail_heading_offsets=[-tile_offset, 0.0, tile_offset],
                    candidate_shadow=candidate_model in {"classless_v022", "classless_local_v023", "classless_local_v024"},
                )
            elif selected_mode == "enhanced":
                verifier_key = str(stage7_api_key or "").strip()
                if len(verifier_key) < 20:
                    raise Stage7SurfacePrecisionError(
                        "Enable Stage 7 multimodal enhancement and provide the user's OpenAI key for this perception request. Platform Assistant credentials are never reused."
                    )
                local_inference: dict[str, Any] | None = None
                local_error: str | None = None
                try:
                    local_inference = stage7_ensemble.infer(
                        image_path=temporary,
                        frame_identity=frame_id,
                        candidate=True,
                    )
                except Stage6DeploymentError as error:
                    local_error = str(error)[:500]
                try:
                    tile_fov = max(10.0, min(60.0, fov * 4.0 / 9.0))
                    tile_offset = fov / 3.0
                    detail_tiles: dict[str, Path] = {}
                    for tile_name, heading_offset in (
                        ("left", -tile_offset),
                        ("centre", 0.0),
                        ("right", tile_offset),
                    ):
                        detail_path, _detail_meta = self._download_google_frame(
                            api_key=api_key,
                            pano_id=pano_id,
                            heading=(heading + heading_offset) % 360.0,
                            pitch=pitch,
                            fov=tile_fov,
                            width=640,
                            height=640,
                            referrer=referrer,
                        )
                        detail_temporaries.append(detail_path)
                        detail_tiles[tile_name] = detail_path
                    shadow_version = (
                        R58_V021_SHADOW_FLAG.read_text(encoding="utf-8").strip()
                        if flag_shadow_v021 and R58_V021_SHADOW_FLAG.is_file()
                        else ""
                    )
                    selected_surface_infer = infer_surface_precision
                    if active_surface_precision_v021_candidate06:
                        selected_surface_infer = infer_surface_precision_v021_candidate06
                    elif candidate_model == "surface_precision_v021":
                        selected_surface_infer = infer_surface_precision_v021
                    elif flag_shadow_v021:
                        if shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate07":
                            selected_surface_infer = infer_surface_precision_v021_candidate07
                        elif shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate06":
                            selected_surface_infer = infer_surface_precision_v021_candidate06
                        elif shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate05":
                            selected_surface_infer = infer_surface_precision_v021_candidate05
                        elif shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate04":
                            selected_surface_infer = infer_surface_precision_v021_candidate04
                        elif shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate03":
                            selected_surface_infer = infer_surface_precision_v021_candidate03
                        elif shadow_version == "Stage7-SurfacePrecision-v0.1.21-candidate02":
                            selected_surface_infer = infer_surface_precision_v021_candidate02
                        else:
                            selected_surface_infer = infer_surface_precision_v021
                    multimodal_inference = selected_surface_infer(
                        api_key=verifier_key,
                        image_path=temporary,
                        tiles=detail_tiles,
                        frame_identity=frame_id,
                    )
                    inference = _fuse_local_and_multimodal(
                        local_inference,
                        multimodal_inference,
                        local_error=local_error,
                    )
                    inference["run"].setdefault("diagnostics", {}).update(
                        {
                            "canonical_fov": fov,
                            "detail_fov": tile_fov,
                            "detail_heading_offsets": [-tile_offset, 0.0, tile_offset],
                            "exact_current_view_only": True,
                            "openai_used": True,
                            "assistant_credential_reused": False,
                            "r58_v021_shadow_route": shadow_v021,
                            "r58_v021_shadow_version": shadow_version or None,
                            "active_surface_precision_version": (
                                "Stage7-SurfacePrecision-v0.1.21-candidate06"
                                if active_surface_precision_v021_candidate06
                                else None
                            ),
                        }
                    )
                except (Stage7Error, Stage7SurfacePrecisionError) as error:
                    if local_inference is None:
                        raise
                    selected_mode = "local_fallback"
                    inference = local_inference
                    inference["run"].setdefault("diagnostics", {}).update(
                        {
                            "inference_mode": "local_fallback",
                            "openai_used": False,
                            "assistant_credential_reused": False,
                            "enhancement_error": str(error)[:500],
                        }
                    )
                finally:
                    verifier_key = ""
            else:
                inference = stage7_ensemble.infer(
                    image_path=temporary,
                    frame_identity=frame_id,
                    candidate=True,
                )
                inference["run"].setdefault("diagnostics", {}).update(
                    {
                        "inference_mode": "local",
                        "openai_used": False,
                        "assistant_credential_reused": False,
                    }
                )
            run = inference["run"]
            detections = inference["detections"]
            imagery["frame_sha256"] = run["image_sha256"]
            model = {
                "id": run["model_id"],
                "name": run["model_name"],
                "version": run["model_version"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "components": run.get("component_sha256") or {},
                "strategy": (
                    run.get("diagnostics", {}).get("strategy")
                    or "governed_multiscale_heterogeneous_teacher_fusion_v0_1_4"
                ),
            }
            if detections:
                fallback = selected_mode == "local_fallback"
                return self._current_view_response(
                    analysis_id=analysis_id,
                    requested_at=requested_at,
                    frame=frame,
                    state="detections_found_review_required",
                    message=(
                        f"The exact current Google Street View frame was processed and produced "
                        f"{len(detections)} "
                        + ("possible road-problem region(s)." if classless_selected else "class-coloured road-problem detection(s).")
                        + (" OpenAI enhancement was unavailable, so the isolated local result was returned." if fallback else "")
                    ),
                    reason_codes=["enhancement_unavailable_local_result_returned"] if fallback else None,
                    imagery=imagery,
                    model=model,
                    detections=detections,
                    inference_run=run,
                    inference_mode=selected_mode,
                )
            return self._current_view_response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                frame=frame,
                state="analysis_complete_no_supported_detection",
                message=(
                    "The exact current Google Street View frame was processed successfully; "
                    "the active road-problem model retained zero detections in this view."
                    + (" OpenAI enhancement was unavailable, so this is the isolated local result." if selected_mode == "local_fallback" else "")
                ),
                reason_codes=(
                    ["no_detection_at_or_above_model_threshold", "enhancement_unavailable_local_result_returned"]
                    if selected_mode == "local_fallback"
                    else ["no_detection_at_or_above_model_threshold"]
                ),
                imagery=imagery,
                model=model,
                inference_run=run,
                inference_mode=selected_mode,
            )
        except (Stage7Error, Stage6DeploymentError, Stage7SurfacePrecisionError) as error:
            return self._current_view_response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                frame=frame,
                state="analysis_not_possible",
                message=(
                    "The exact current Google Street View frame could not be processed; "
                    "no nearby or substitute image was used."
                ),
                reason_codes=["exact_current_frame_acquisition_or_model_failure"],
                imagery=imagery,
                diagnostics={"detail": str(error)[:500]},
                inference_mode=selected_mode,
            )
        finally:
            for detail_temporary in detail_temporaries:
                detail_temporary.unlink(missing_ok=True)
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def analyse(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_m: int,
        provider: str,
        surface_type: str | None,
        photo_id: str | None,
        photos: list[dict[str, Any]],
        provider_error: str | None,
        deployment: Stage6Deployment,
    ) -> dict[str, Any]:
        analysis_id = f"stage7-analysis-{uuid.uuid4().hex}"
        requested_at = _timestamp()
        if provider == "google_maps_core":
            return self._response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius_m,
                provider=provider,
                state="analysis_not_possible",
                message="The Google panorama can be viewed interactively, but it was not submitted to the local model.",
                reason_codes=["google_maps_core_local_inference_not_permitted"],
                diagnostics={"next_provider_path": "separately onboard Google Street View Insights with a Vertex AI-resident model"},
            )
        if provider_error:
            return self._response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius_m,
                provider=provider,
                state="analysis_not_possible",
                message="No image was processed because the permitted open-imagery catalogues could not be reached.",
                reason_codes=["permitted_provider_temporarily_unavailable"],
                diagnostics={"provider": "open imagery", "detail": provider_error[:500]},
            )
        selected = None
        if photo_id:
            selected = next((item for item in photos if str(item.get("id")) == photo_id), None)
            if selected is None:
                return self._response(
                    analysis_id=analysis_id,
                    requested_at=requested_at,
                    latitude=latitude,
                    longitude=longitude,
                    radius_m=radius_m,
                    provider=provider,
                    state="analysis_not_possible",
                    message="The requested open-licence photograph was not returned by the bounded point search, so no image was processed.",
                    reason_codes=["requested_permitted_photo_not_found"],
                )
        elif photos:
            selected = photos[0]
        if selected is None:
            return self._response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius_m,
                provider=provider,
                state="analysis_not_possible",
                message=f"No permitted geotagged road image was found within {radius_m:,} m, so this point was not analysed.",
                reason_codes=["no_permitted_street_imagery_within_radius"],
                diagnostics={"google_maps_core_role": "interactive human-viewed context only"},
            )

        source_id = str(selected.get("source_id") or "kartaview_cc_by_sa")
        evidence_source_id = "licensed_public_dataset" if source_id == "wikimedia_commons_open" else source_id
        provider_name = str(selected.get("provider") or ("Wikimedia Commons" if source_id == "wikimedia_commons_open" else "KartaView"))
        licence = str(selected.get("licence") or ("CC BY-SA 4.0" if source_id == "kartaview_cc_by_sa" else "verified open licence"))
        attribution = str(selected.get("attribution") or KARTAVIEW_ATTRIBUTION)
        permission_record_id = str(selected.get("permission_record_id") or KARTAVIEW_PERMISSION_RECORD)
        imagery = {
            "provider": provider_name,
            "source_id": source_id,
            "photo_id": str(selected["id"]),
            "title": selected.get("title") or f"{provider_name} image {selected['id']}",
            "image_url": selected["image_url"],
            "thumbnail_url": selected.get("thumbnail_url") or selected["image_url"],
            "details_url": selected.get("details_url") or "https://kartaview.org/map",
            "captured_at": selected.get("captured_at") or "unknown_not_provided",
            "latitude": float(selected["latitude"]),
            "longitude": float(selected["longitude"]),
            "distance_m": selected.get("distance_m"),
            "licence": licence,
            "attribution": attribution,
        }
        temporary: Path | None = None
        try:
            temporary, image_meta = self._download_image(selected["image_url"], source_id)
            imagery.update(image_meta)
            asset_id = str(selected["id"])
            safe_asset_id = "".join(character for character in asset_id if character.isalnum() or character in "_-")[:80] or "unknown"
            image_record = {
                "id": f"{source_id}:{asset_id}",
                "source_asset_id": asset_id,
                "original_name": str(selected.get("title") or f"{provider_name} image {asset_id}"),
                "stored_name": f"stage7-{safe_asset_id}{temporary.suffix}",
                "image_url": selected["image_url"],
                "provider": source_id,
                "source_id": source_id,
                "analysis_authorised": True,
                "licence_status": "verified",
                "captured_at": selected.get("captured_at") or "unknown_not_provided",
                "latitude": float(selected["latitude"]),
                "longitude": float(selected["longitude"]),
                "location_method": str(selected.get("location_method") or "source metadata"),
                "source_evidence": {
                    "source_id": evidence_source_id,
                    "asset_id": asset_id,
                    "captured_at": str(selected.get("captured_at") or "unknown_not_provided"),
                    "licence_status": "verified",
                    "permission_record_id": permission_record_id,
                    "attribution": attribution,
                },
            }
            inference = deployment.infer_evidence_image(
                image_record=image_record,
                image_path=temporary,
                surface_type=surface_type,
                origin_stage=7,
            )
            result = inference["result"]
            observations = result.get("observations") or []
            model = {
                "id": inference["run"]["model_id"],
                "name": inference["run"]["model_name"],
                "version": inference["run"]["model_version"],
                "checkpoint_sha256": inference["run"]["checkpoint_sha256"],
            }
            if observations:
                return self._response(
                    analysis_id=analysis_id,
                    requested_at=requested_at,
                    latitude=latitude,
                    longitude=longitude,
                    radius_m=radius_m,
                    provider=provider,
                    state="detections_found_review_required",
                    message=f"The permitted {provider_name} road image was processed and produced {len(observations)} review-required detection(s).",
                    imagery=imagery,
                    model=model,
                    observations=observations,
                    inference_run=inference["run"],
                )
            reasons = list(result.get("reasons") or [])
            if reasons == ["no_detection_at_or_above_minimum_confidence"]:
                return self._response(
                    analysis_id=analysis_id,
                    requested_at=requested_at,
                    latitude=latitude,
                    longitude=longitude,
                    radius_m=radius_m,
                    provider=provider,
                    state="analysis_complete_no_supported_detection",
                    message=f"The permitted {provider_name} road image was processed successfully, but no supported defect was retained at the frozen model threshold.",
                    reason_codes=reasons,
                    imagery=imagery,
                    model=model,
                    inference_run=inference["run"],
                )
            return self._response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius_m,
                provider=provider,
                state="analysis_not_possible",
                message="The image reached the governed model adapter, but the evidence contract required abstention.",
                reason_codes=reasons or ["governed_model_abstention"],
                imagery=imagery,
                model=model,
                inference_run=inference["run"],
            )
        except (Stage7Error, Stage6DeploymentError) as error:
            return self._response(
                analysis_id=analysis_id,
                requested_at=requested_at,
                latitude=latitude,
                longitude=longitude,
                radius_m=radius_m,
                provider=provider,
                state="analysis_not_possible",
                message="No model result was produced for this point; the failure is recorded explicitly.",
                reason_codes=["provider_image_or_model_runtime_failure"],
                imagery=imagery,
                diagnostics={"detail": str(error)[:500]},
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


stage7 = Stage7OnDemand()
