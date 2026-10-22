from __future__ import annotations

import copy
import html
import json
import hashlib
import math
import os
import re
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset_v1 import (
    DATASET_ASSET_SCHEMA,
    SOURCE_REGISTRY,
    STAGE3_CONTRACT,
    dataset_readiness_report,
    load_current_manifest,
    run_dataset_contract_checks,
    validate_manifest,
)
from .model_v1 import (
    load_model_card,
    load_stage4_contract,
    stage4_checks,
    stage4_report,
    visualisation_path,
)
from .validation_v1 import (
    REPORT_PATH as STAGE5_REPORT_PATH,
    load_stage5_contract,
    stage5_checks,
    stage5_report,
    visualisation_path as stage5_visualisation_path,
)
from .validation_v2 import (
    COMPLETION_PATH as STAGE5_V2_COMPLETION_PATH,
    REPORT_PATH as STAGE5_V2_REPORT_PATH,
    load_stage5_v2_contract,
    stage5_v2_checks,
    stage5_v2_report,
)
from .twm_stage6 import stage6_twm_status
from .stage6_deployment import (
    Stage6ConflictError,
    Stage6DeploymentError,
    Stage6NotFoundError,
    deployment as stage6_deployment,
)
from .stage7_on_demand import Stage7Error, stage7 as stage7_service
from .stage8_annotations import (
    Stage8AnnotationConflictError,
    Stage8AnnotationError,
    Stage8AnnotationNotFoundError,
    annotations as stage8_annotations,
)
from . import stage7_classless_v022
from .python_runtime import configure_perception_python


configure_perception_python(stage7_classless_v022)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = WORKSPACE_ROOT.parent
FRONTEND_ROOT = WORKSPACE_ROOT / "frontend"
CONTRACT_ROOT = WORKSPACE_ROOT / "contracts"
STORAGE_ROOT = WORKSPACE_ROOT / "storage"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
CATALOGUE_PATH = STORAGE_ROOT / "catalogue.json"
SURVEY_PROJECT_PATH = STORAGE_ROOT / "survey_project.json"
MAPLIBRE_VENDOR = PLATFORM_ROOT / "frontend" / "vendor"
STAGE2_CONTRACT_PATH = CONTRACT_ROOT / "stage2_ai_contract.json"
OBSERVATION_SCHEMA_PATH = CONTRACT_ROOT / "observation.schema.json"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 30 * 1024 * 1024
DEFAULT_MAP_CENTRE = {"latitude": 23.2599, "longitude": 77.4126, "zoom": 12.0}
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
COPERNICUS_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1/"
COPERNICUS_STAC_SEARCH_URL = urljoin(COPERNICUS_STAC_ROOT, "search")
KARTAVIEW_PHOTO_URL = "https://api.openstreetcam.org/2.0/photo/"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
NETWORK_TIMEOUT_SECONDS = 18
MAX_EXTERNAL_RESPONSE_BYTES = 20 * 1024 * 1024
OUTBOUND_USER_AGENT = "ResiliFlow-TWM-Infrastructure-Perception/1.0 (https://www.unsw.edu.au/research/rciti; research prototype)"

SATELLITE_COLLECTIONS = {
    "sentinel-2-l2a": {
        "name": "Sentinel-2 Level-2A",
        "sensor": "Multispectral optical",
        "resolution": "10 m visible and near-infrared; 20/60 m additional bands",
        "best_for": "Broad surface, water, vegetation and large-area change screening",
        "cloud_filter": True,
    },
    "sentinel-1-grd": {
        "name": "Sentinel-1 GRD",
        "sensor": "C-band synthetic aperture radar",
        "resolution": "Approximately 10 m for common IW products; mode dependent",
        "best_for": "All-weather flood, moisture and broad surface-change screening",
        "cloud_filter": False,
    },
}

_preview_sources: dict[str, str] = {}
_preview_lock = threading.RLock()
_location_cache: dict[str, list[dict[str, Any]]] = {}

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
_catalogue_lock = threading.RLock()


class LocationUpdate(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    direction: float | None = Field(default=None, ge=0, lt=360)


class AnalysisAuthorisationUpdate(BaseModel):
    confirmed: bool
    declaration: str = Field(min_length=12, max_length=500)


class ContractValidationRequest(BaseModel):
    observation: dict[str, Any]


class Stage6InferenceRequest(BaseModel):
    surface_type: str | None = None

    @model_validator(mode="after")
    def validate_surface(self) -> "Stage6InferenceRequest":
        if self.surface_type not in {None, "road", "footpath", "kerb"}:
            raise ValueError("Choose road, footpath, kerb or automatic surface resolution.")
        return self


class Stage7PointAnalysisRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_m: int = Field(default=1000, ge=25, le=2000)
    provider: str = "auto"
    surface_type: str | None = None
    photo_id: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_stage7_request(self) -> "Stage7PointAnalysisRequest":
        if self.provider not in {"auto", "kartaview_cc_by_sa", "wikimedia_commons_open", "google_maps_core"}:
            raise ValueError("Choose automatic, KartaView, Wikimedia Commons or Google Maps Core provider handling.")
        if self.surface_type not in {None, "road", "footpath", "kerb"}:
            raise ValueError("Choose road, footpath, kerb or automatic surface resolution.")
        if self.photo_id is not None:
            self.photo_id = self.photo_id.strip() or None
        return self


class Stage7CurrentViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=20, max_length=500, repr=False)
    stage7_api_key: str | None = Field(default=None, min_length=20, max_length=500, repr=False)
    inference_mode: str = "local"
    frame_id: str = Field(min_length=1, max_length=240)
    pano_id: str = Field(min_length=1, max_length=300)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    heading: float = Field(ge=0, lt=360)
    pitch: float = Field(ge=-90, le=90)
    zoom: float = Field(ge=0, le=10)
    fov: float = Field(ge=10, le=120)
    width: int = Field(ge=128, le=640)
    height: int = Field(ge=128, le=640)
    viewport_width: int = Field(ge=128, le=10000)
    viewport_height: int = Field(ge=128, le=10000)
    frame_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_current_view(self) -> "Stage7CurrentViewRequest":
        self.api_key = self.api_key.strip()
        if self.stage7_api_key is not None:
            self.stage7_api_key = self.stage7_api_key.strip() or None
        self.inference_mode = self.inference_mode.strip().lower()
        self.frame_id = self.frame_id.strip()
        self.pano_id = self.pano_id.strip()
        if len(self.api_key) < 20 or not self.frame_id or not self.pano_id:
            raise ValueError("A complete current-view frame identity and Google key are required.")
        if self.inference_mode not in {"local", "enhanced"}:
            raise ValueError("Choose local or enhanced Stage 7 inference.")
        if self.inference_mode == "enhanced" and self.stage7_api_key is None:
            raise ValueError("Enhanced Stage 7 inference requires the user's OpenAI key for this request.")
        return self


STAGE8_SURFACES = {"road", "kerb", "footpath"}
STAGE8_DEFECT_CLASSES = {
    "alligator_cracking",
    "longitudinal_crack",
    "transverse_crack",
    "pothole",
    "standing_water",
    "broken_kerb",
    "uneven_or_broken_footpath",
    "debris_or_vegetation_obstruction",
}


class Stage8FrameIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1, max_length=240)
    pano_id: str = Field(min_length=1, max_length=300)
    frame_revision: int = Field(ge=0)
    frame_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    heading: float = Field(ge=0, lt=360)
    pitch: float = Field(ge=-90, le=90)
    zoom: float = Field(ge=0, le=10)
    fov: float = Field(ge=10, le=120)
    width: int = Field(ge=128, le=640)
    height: int = Field(ge=128, le=640)
    viewport_width: int = Field(ge=128, le=10000)
    viewport_height: int = Field(ge=128, le=10000)


class Stage8SelectedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_m: int | None = Field(default=None, ge=25, le=2000)


class Stage8OriginalDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_id: str | None = Field(default=None, max_length=240)
    damage_type: str = Field(min_length=1, max_length=120)
    surface_type: str | None = Field(default=None, max_length=80)
    confidence: float | None = Field(default=None, ge=0, le=1)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    evidence_region: dict[str, Any] | None = None
    consensus: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_original_box(self) -> "Stage8OriginalDetection":
        if self.box is None and not isinstance(self.evidence_region, dict):
            raise ValueError("An original detection requires a normalised box or evidence region.")
        if self.box is not None:
            _validate_stage8_box(self.box)
        return self


def _validate_stage8_box(box: list[float]) -> None:
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError("A Stage 8 box must contain four finite coordinates.")
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        raise ValueError("A Stage 8 box must be a positive normalised region inside the frame.")


class Stage8AnnotationBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_box_id: str = Field(min_length=1, max_length=160)
    origin: str = Field(pattern=r"^(model|human)$")
    damage_type: str | None = Field(default=None, max_length=120)
    surface_type: str | None = Field(default=None, max_length=80)
    uncertain: bool = False
    box: list[float] = Field(min_length=4, max_length=4)
    source_detection_id: str | None = Field(default=None, max_length=240)
    model_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_annotation(self) -> "Stage8AnnotationBox":
        _validate_stage8_box(self.box)
        if self.uncertain:
            if self.damage_type not in {None, "uncertain", *STAGE8_DEFECT_CLASSES}:
                raise ValueError("Choose a frozen defect class or uncertain.")
            if self.surface_type not in {None, *STAGE8_SURFACES}:
                raise ValueError("Choose road, kerb, footpath or leave an uncertain surface unresolved.")
            return self
        if self.damage_type not in STAGE8_DEFECT_CLASSES:
            raise ValueError("Every resolved annotation requires one of the frozen eight defect classes.")
        if self.surface_type not in STAGE8_SURFACES:
            raise ValueError("Every resolved annotation requires road, kerb or footpath.")
        return self


class Stage8ReviewSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str = Field(min_length=8, max_length=160)
    draft_started_at: str | None = Field(default=None, max_length=80)
    frame: Stage8FrameIdentity
    selected_point: Stage8SelectedPoint
    inference_mode: str = Field(pattern=r"^(local|enhanced|local_fallback)$")
    source_model: dict[str, Any]
    original_detections: list[Stage8OriginalDetection] = Field(default_factory=list, max_length=100)
    annotations: list[Stage8AnnotationBox] = Field(default_factory=list, max_length=100)
    pending_annotations: list[Stage8AnnotationBox] = Field(default_factory=list, max_length=100)
    submission_scope: str = Field(
        default="complete_review",
        pattern=r"^(partial_review|complete_review|clean_frame)$",
    )
    ignore_unreviewed_regions: bool = False
    clean_frame: bool = False
    reviewer_id: str = Field(default="human-reviewer", min_length=1, max_length=160)
    note: str | None = Field(default=None, max_length=1000)
    change_log: list[dict[str, Any]] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_review(self) -> "Stage8ReviewSaveRequest":
        if self.clean_frame and (self.annotations or self.pending_annotations):
            raise ValueError("A clean frame cannot retain confirmed or pending annotation boxes.")
        if not self.clean_frame and not self.annotations:
            raise ValueError("Apply at least one corrected annotation before confirming, or mark the exact frame clean.")
        if self.clean_frame and self.ignore_unreviewed_regions:
            raise ValueError("A confirmed clean frame cannot contain ignored unreviewed regions.")
        if not self.clean_frame and self.submission_scope == "clean_frame":
            raise ValueError("The clean-frame submission scope requires an explicitly clean frame.")
        if self.submission_scope == "partial_review":
            if not self.pending_annotations:
                raise ValueError("A partial review requires at least one separately stored pending annotation.")
            if not self.ignore_unreviewed_regions:
                raise ValueError("A partial review must ignore unreviewed regions instead of using them as negatives.")
            if any(item.uncertain for item in self.annotations):
                raise ValueError("Only resolved confirmed annotations can enter a partial training unit.")
        elif self.pending_annotations:
            raise ValueError("Pending annotations require the partial-review submission scope.")
        identifiers = [
            item.annotation_box_id
            for item in [*self.annotations, *self.pending_annotations]
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Confirmed and pending annotation box identifiers must be unique within a review.")
        version = str(self.source_model.get("version") or "").strip()
        checkpoint = str(self.source_model.get("checkpoint_sha256") or "").strip()
        if not version or not re.fullmatch(r"[A-Fa-f0-9]{64}", checkpoint):
            raise ValueError("The source model version and checkpoint identity are required.")
        return self


class Stage8QueueStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1, max_length=80)
    model_version: str | None = Field(default=None, max_length=160)


class Stage6ReviewRequest(BaseModel):
    status: str
    reviewer_id: str = Field(min_length=1, max_length=160)
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_review(self) -> "Stage6ReviewRequest":
        if self.status not in {"confirmed", "rejected", "uncertain"}:
            raise ValueError("Choose confirmed, rejected or uncertain.")
        return self


class Stage6ActivationRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=160)
    expected_checkpoint_sha256: str = Field(min_length=64, max_length=64)
    actor: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=500)
    confirmed: bool = False


class Stage6RollbackRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=500)
    confirmed: bool = False


class DatasetValidationRequest(BaseModel):
    manifest: dict[str, Any]


class BoundingBox(BaseModel):
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def validate_extent(self) -> "BoundingBox":
        if self.east <= self.west or self.north <= self.south:
            raise ValueError("The research area must have positive width and height.")
        if self.east - self.west > 5 or self.north - self.south > 5:
            raise ValueError("Choose a research area no larger than 5 degrees in either direction.")
        return self

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class SatelliteSearchRequest(BaseModel):
    aoi: BoundingBox
    collection: str = "sentinel-2-l2a"
    start_date: date
    end_date: date
    max_cloud_cover: float = Field(default=40, ge=0, le=100)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_search(self) -> "SatelliteSearchRequest":
        if self.collection not in SATELLITE_COLLECTIONS:
            raise ValueError("Choose a supported Sentinel collection.")
        if self.end_date < self.start_date:
            raise ValueError("The end date must not be earlier than the start date.")
        if (self.end_date - self.start_date).days > 3660:
            raise ValueError("Choose a date range no longer than ten years.")
        return self


class SceneSnapshot(BaseModel):
    id: str = Field(min_length=1, max_length=240)
    collection: str = Field(min_length=1, max_length=80)
    captured_at: str | None = Field(default=None, max_length=80)
    cloud_cover: float | None = Field(default=None, ge=0, le=100)
    bbox: list[float] = Field(default_factory=list, max_length=4)


class SurveyProjectState(BaseModel):
    name: str = Field(default="Untitled imagery survey", min_length=1, max_length=120)
    location_label: str | None = Field(default=None, max_length=240)
    map_center: list[float] = Field(default_factory=list, max_length=2)
    map_zoom: float | None = Field(default=None, ge=0, le=24)
    aoi: BoundingBox | None = None
    collection: str | None = Field(default=None, max_length=80)
    start_date: date | None = None
    end_date: date | None = None
    max_cloud_cover: float | None = Field(default=None, ge=0, le=100)
    selected_scenes: list[SceneSnapshot] = Field(default_factory=list, max_length=2)
    notes: str | None = Field(default=None, max_length=2000)


def _empty_catalogue() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": None, "images": []}


def _load_catalogue() -> dict[str, Any]:
    with _catalogue_lock:
        if not CATALOGUE_PATH.exists():
            return _empty_catalogue()
        try:
            payload = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _empty_catalogue()
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            return _empty_catalogue()
        return payload


def _save_catalogue(payload: dict[str, Any]) -> None:
    with _catalogue_lock:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = CATALOGUE_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, CATALOGUE_PATH)


def _ratio(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        numerator = getattr(value, "numerator", 0)
        denominator = getattr(value, "denominator", 1)
        return float(numerator) / float(denominator or 1)


def _dms_to_decimal(values: Any, reference: Any) -> float | None:
    try:
        degrees, minutes, seconds = (_ratio(item) for item in values)
    except (TypeError, ValueError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    ref = reference.decode(errors="ignore") if isinstance(reference, bytes) else str(reference or "")
    if ref.upper() in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _extract_metadata(image_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "captured_at": None,
        "direction": None,
        "width": None,
        "height": None,
    }
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        metadata["width"], metadata["height"] = image.size
        exif = image.getexif()
        if not exif:
            return metadata
        metadata["captured_at"] = exif.get(36867) or exif.get(36868) or exif.get(306)
        try:
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError, TypeError):
            gps = exif.get(34853) or {}
        if isinstance(gps, dict):
            latitude = _dms_to_decimal(gps.get(2), gps.get(1)) if gps.get(2) else None
            longitude = _dms_to_decimal(gps.get(4), gps.get(3)) if gps.get(4) else None
            if latitude is not None and longitude is not None:
                metadata["latitude"] = latitude
                metadata["longitude"] = longitude
            if gps.get(17) is not None:
                metadata["direction"] = round(_ratio(gps.get(17)) % 360.0, 2)
    return metadata


def _public_record(record: dict[str, Any], distance_m: float | None = None) -> dict[str, Any]:
    result = dict(record)
    result["image_url"] = f"uploads/{record['stored_name']}"
    if distance_m is not None:
        result["distance_m"] = round(distance_m, 1)
    return result


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch_json(url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": OUTBOUND_USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = UrlRequest(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_EXTERNAL_RESPONSE_BYTES:
                raise HTTPException(status_code=502, detail="The external catalogue response was unexpectedly large.")
            body = response.read(MAX_EXTERNAL_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise HTTPException(status_code=502, detail=f"The external imagery service returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(status_code=502, detail="The external imagery service is temporarily unavailable.") from error
    if len(body) > MAX_EXTERNAL_RESPONSE_BYTES:
        raise HTTPException(status_code=502, detail="The external catalogue response was unexpectedly large.")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail="The external imagery service returned an unreadable response.") from error


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _approved_kartaview_url(value: Any) -> str | None:
    url = str(value or "").strip()
    if not url or "{{sizeprefix}}" in url:
        return None
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    approved = hostname == "kartaview.org" or hostname.endswith(".kartaview.org") or hostname == "openstreetcam.org" or hostname.endswith(".openstreetcam.org")
    return url if parsed.scheme == "https" and approved else None


def _kartaview_records(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    result = payload.get("result")
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if isinstance(data, dict):
        return [data]
    return [item for item in data or [] if isinstance(item, dict)]


def _normalise_kartaview_photo(item: dict[str, Any], *, query_latitude: float | None = None, query_longitude: float | None = None) -> dict[str, Any] | None:
    photo_id = str(item.get("id") or item.get("photoId") or "").strip()
    sequence_id = str(item.get("sequenceId") or item.get("sequence_id") or "").strip()
    latitude = _optional_float(item.get("lat") if item.get("lat") is not None else item.get("latitude"))
    longitude = _optional_float(item.get("lng") if item.get("lng") is not None else item.get("longitude"))
    image_url = next(
        (
            candidate
            for candidate in (
                _approved_kartaview_url(item.get("imageProcUrl")),
                _approved_kartaview_url(item.get("imageLthUrl")),
                _approved_kartaview_url(item.get("imageThUrl")),
                _approved_kartaview_url(item.get("fileurlProc")),
                _approved_kartaview_url(item.get("fileurlLTh")),
                _approved_kartaview_url(item.get("fileurlTh")),
            )
            if candidate
        ),
        None,
    )
    if not photo_id or latitude is None or longitude is None or image_url is None:
        return None
    distance = _optional_float(item.get("distance"))
    if distance is None and query_latitude is not None and query_longitude is not None:
        distance = _haversine_m(query_latitude, query_longitude, latitude, longitude)
    heading = _optional_float(item.get("heading"))
    field_of_view = _optional_float(item.get("fieldOfView") or item.get("field_of_view"))
    sequence_index = _optional_int(item.get("sequenceIndex") or item.get("sequence_index"))
    details_url = f"https://kartaview.org/details/{sequence_id}/{sequence_index or 0}/track-info" if sequence_id else "https://kartaview.org/map"
    return {
        "id": photo_id,
        "source_id": "kartaview_cc_by_sa",
        "provider": "KartaView",
        "sequence_id": sequence_id or None,
        "sequence_index": sequence_index,
        "captured_at": item.get("shotDate") or item.get("dateAdded") or item.get("date_added"),
        "latitude": latitude,
        "longitude": longitude,
        "direction": round(heading % 360, 2) if heading is not None else None,
        "field_of_view": field_of_view,
        "is_360": bool(field_of_view is not None and field_of_view >= 300) or str(item.get("projection") or "").upper() == "SPHERE",
        "image_url": image_url,
        "thumbnail_url": _approved_kartaview_url(item.get("imageThUrl")) or image_url,
        "width": _optional_int(item.get("width")),
        "height": _optional_int(item.get("height")),
        "distance_m": round(distance, 1) if distance is not None else None,
        "details_url": details_url,
        "source": "KartaView open street imagery",
        "licence": "CC BY-SA 4.0",
        "attribution": "© Grab and KartaView Contributors · CC BY-SA 4.0",
        "permission_record_id": "kartaview-terms-2025-06-17-cc-by-sa-4.0",
        "location_method": "network match",
    }


def _plain_metadata(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())


def _approved_wikimedia_url(value: Any, *, details: bool = False) -> str | None:
    url = str(value or "").strip()
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    approved = hostname == "upload.wikimedia.org" if not details else hostname in {"commons.wikimedia.org", "upload.wikimedia.org"}
    return url if parsed.scheme == "https" and approved and not parsed.username and not parsed.password else None


def _wikimedia_open_licence(value: Any) -> str | None:
    licence = _plain_metadata(value)
    normalised = licence.casefold()
    if not licence or "noncommercial" in normalised or "no derivatives" in normalised or "-nc" in normalised or "-nd" in normalised:
        return None
    allowed = normalised.startswith("cc by ") or normalised.startswith("cc by-sa ") or normalised.startswith("cc0") or normalised.startswith("public domain")
    return licence if allowed else None


def _normalise_wikimedia_photo(
    page: dict[str, Any],
    *,
    query_latitude: float,
    query_longitude: float,
) -> dict[str, Any] | None:
    coordinates = page.get("coordinates") if isinstance(page.get("coordinates"), list) else []
    image_info = page.get("imageinfo") if isinstance(page.get("imageinfo"), list) else []
    if not coordinates or not image_info or not isinstance(image_info[0], dict):
        return None
    info = image_info[0]
    metadata = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    latitude = _optional_float(coordinates[0].get("lat"))
    longitude = _optional_float(coordinates[0].get("lon"))
    licence = _wikimedia_open_licence(metadata.get("LicenseShortName") or metadata.get("License"))
    original_url = _approved_wikimedia_url(info.get("url"))
    thumbnail_url = _approved_wikimedia_url(info.get("thumburl"))
    image_url = thumbnail_url or original_url
    details_url = _approved_wikimedia_url(info.get("descriptionurl"), details=True)
    mime = str(info.get("mime") or "").lower()
    page_id = str(page.get("pageid") or "").strip()
    title = str(page.get("title") or "").removeprefix("File:").strip()
    road_context = " ".join(
        [
            title,
            _plain_metadata(metadata.get("ObjectName")),
            _plain_metadata(metadata.get("ImageDescription")),
            _plain_metadata(metadata.get("Categories")),
        ]
    ).casefold()
    road_terms = (
        "road", "street", "footpath", "pavement", "sidewalk", "kerb", "curb", "bridge",
        "traffic", "highway", "road barrier",
    )
    if (
        latitude is None
        or longitude is None
        or licence is None
        or image_url is None
        or details_url is None
        or not page_id
        or mime not in {"image/jpeg", "image/png", "image/webp"}
        or not any(term in road_context for term in road_terms)
    ):
        return None
    artist = _plain_metadata(metadata.get("Artist")) or "Wikimedia Commons contributor"
    captured_at = (
        _plain_metadata(metadata.get("DateTimeOriginal"))
        or _plain_metadata(metadata.get("DateTimeDigitized"))
        or _plain_metadata(metadata.get("Date"))
        or "unknown_not_provided"
    )
    distance = _haversine_m(query_latitude, query_longitude, latitude, longitude)
    permission_token = re.sub(r"[^a-z0-9]+", "-", licence.casefold()).strip("-") or "open-licence"
    return {
        "id": f"commons-{page_id}",
        "source_id": "wikimedia_commons_open",
        "provider": "Wikimedia Commons",
        "title": title or f"Wikimedia Commons file {page_id}",
        "captured_at": captured_at,
        "latitude": latitude,
        "longitude": longitude,
        "image_url": image_url,
        "thumbnail_url": image_url,
        "width": _optional_int(info.get("thumbwidth") or info.get("width")),
        "height": _optional_int(info.get("thumbheight") or info.get("height")),
        "distance_m": round(distance, 1),
        "details_url": details_url,
        "source": "Wikimedia Commons geotagged open imagery",
        "licence": licence,
        "attribution": f"{artist} · {licence} · Wikimedia Commons",
        "permission_record_id": f"wikimedia-commons:{page_id}:{permission_token}",
        "location_method": "source metadata",
    }


def _wikimedia_nearby_photos(*, latitude: float, longitude: float, radius_m: int) -> list[dict[str, Any]]:
    search_params = urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "geosearch",
            "gsprimary": "all",
            "gsnamespace": 6,
            "gsradius": radius_m,
            "gscoord": f"{latitude:.7f}|{longitude:.7f}",
            "gslimit": 50,
        }
    )
    search_payload = _fetch_json(f"{WIKIMEDIA_API_URL}?{search_params}")
    query = search_payload.get("query") if isinstance(search_payload, dict) else {}
    candidates = query.get("geosearch") if isinstance(query, dict) else []
    page_ids = [str(item.get("pageid")) for item in candidates or [] if isinstance(item, dict) and item.get("pageid")][:50]
    if not page_ids:
        return []
    details_params = urlencode(
        {
            "action": "query",
            "format": "json",
            "pageids": "|".join(page_ids),
            "prop": "coordinates|imageinfo",
            "coprimary": "all",
            "colimit": "max",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 640,
        }
    )
    details_payload = _fetch_json(f"{WIKIMEDIA_API_URL}?{details_params}")
    detail_query = details_payload.get("query") if isinstance(details_payload, dict) else {}
    pages = detail_query.get("pages") if isinstance(detail_query, dict) else {}
    photos = [
        photo
        for page in (pages or {}).values()
        if isinstance(page, dict)
        and (photo := _normalise_wikimedia_photo(page, query_latitude=latitude, query_longitude=longitude))
    ]
    photos.sort(key=lambda item: (item.get("distance_m") is None, item.get("distance_m") or 0, item.get("id") or ""))
    return photos


def _preview_asset(assets: dict[str, Any]) -> str | None:
    preferred_keys = ("thumbnail", "preview", "rendered_preview")
    for key in preferred_keys:
        asset = assets.get(key)
        if isinstance(asset, dict) and asset.get("href"):
            return urljoin(COPERNICUS_STAC_ROOT, str(asset["href"]))
    for asset in assets.values():
        if not isinstance(asset, dict) or not asset.get("href"):
            continue
        media_type = str(asset.get("type") or "").lower()
        roles = {str(role).lower() for role in asset.get("roles") or []}
        href = str(asset["href"])
        raster_preview = media_type in {"image/jpeg", "image/png", "image/webp"} or href.lower().split("?")[0].endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
        if raster_preview or {"thumbnail", "overview"} & roles:
            return urljoin(COPERNICUS_STAC_ROOT, str(asset["href"]))
    return None


def _normalise_scene(feature: dict[str, Any], fallback_collection: str) -> dict[str, Any] | None:
    item_id = str(feature.get("id") or "").strip()
    if not item_id:
        return None
    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    collection = str(feature.get("collection") or fallback_collection)
    bbox = feature.get("bbox") if isinstance(feature.get("bbox"), list) and len(feature["bbox"]) >= 4 else []
    bbox = [round(float(value), 7) for value in bbox[:4]] if bbox else []
    cloud = properties.get("eo:cloud_cover")
    try:
        cloud = round(float(cloud), 2) if cloud is not None else None
    except (TypeError, ValueError):
        cloud = None
    gsd = properties.get("gsd")
    try:
        gsd = round(float(gsd), 2) if gsd is not None else None
    except (TypeError, ValueError):
        gsd = None
    remote_preview = _preview_asset(feature.get("assets") if isinstance(feature.get("assets"), dict) else {})
    preview_url = None
    if remote_preview:
        token = hashlib.sha256(f"{collection}:{item_id}:{remote_preview}".encode("utf-8")).hexdigest()[:28]
        with _preview_lock:
            if len(_preview_sources) >= 300:
                for stale in list(_preview_sources)[:100]:
                    _preview_sources.pop(stale, None)
            _preview_sources[token] = remote_preview
        preview_url = f"api/satellite/previews/{token}"
    captured_at = properties.get("datetime") or properties.get("start_datetime") or properties.get("created")
    meta = SATELLITE_COLLECTIONS.get(collection, {})
    return {
        "id": item_id,
        "title": str(properties.get("title") or item_id),
        "collection": collection,
        "collection_name": meta.get("name", collection),
        "sensor": meta.get("sensor", "Earth observation"),
        "captured_at": captured_at,
        "cloud_cover": cloud,
        "gsd_m": gsd,
        "resolution": meta.get("resolution", "See product metadata"),
        "best_for": meta.get("best_for", "Broad-area screening"),
        "bbox": bbox,
        "preview_url": preview_url,
        "browser_url": f"https://browser.stac.dataspace.copernicus.eu/collections/{collection}/items/{item_id}",
        "provider": "Copernicus Data Space Ecosystem",
        "licence": "Copernicus Sentinel data: free, full and open access with attribution",
        "analysis_allowed": True,
        "scientific_limit": "Satellite screening does not confirm individual potholes or footpath defects.",
    }


def _allowed_preview_host(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    allowed_suffixes = (
        ".dataspace.copernicus.eu",
        ".copernicus.eu",
        ".cloudferro.com",
        ".amazonaws.com",
    )
    return hostname == "dataspace.copernicus.eu" or any(hostname.endswith(suffix) for suffix in allowed_suffixes)


def _fetch_preview(url: str) -> tuple[bytes, str]:
    if not _allowed_preview_host(url):
        raise HTTPException(status_code=403, detail="This preview source is not an approved Copernicus host.")
    request = UrlRequest(url, headers={"User-Agent": OUTBOUND_USER_AGENT, "Accept": "image/*"})
    try:
        with urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            content_type = str(response.headers.get("Content-Type") or "image/jpeg").split(";", 1)[0]
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=502, detail="The Copernicus preview was not an image.")
            body = response.read(MAX_EXTERNAL_RESPONSE_BYTES + 1)
    except HTTPException:
        raise
    except HTTPError as error:
        raise HTTPException(status_code=502, detail=f"The Copernicus preview returned HTTP {error.code}.") from error
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(status_code=502, detail="The Copernicus preview is temporarily unavailable.") from error
    if len(body) > MAX_EXTERNAL_RESPONSE_BYTES:
        raise HTTPException(status_code=502, detail="The Copernicus preview exceeds the safe display limit.")
    return body, content_type


def _load_contract_document(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The Stage 2 contract document could not be loaded: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"The Stage 2 contract document must be a JSON object: {path.name}")
    return payload


STAGE2_CONTRACT = _load_contract_document(STAGE2_CONTRACT_PATH)
OBSERVATION_SCHEMA = _load_contract_document(OBSERVATION_SCHEMA_PATH)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_stage2_observation(observation: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def add(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    required = set(OBSERVATION_SCHEMA.get("required", []))
    for field in sorted(required - observation.keys()):
        add("missing_required_field", field, f"{field} is required by the Stage 2 observation contract.")

    allowed_root_fields = set(OBSERVATION_SCHEMA.get("properties", {}))
    for field in sorted(observation.keys() - allowed_root_fields):
        code = "unsupported_engineering_measurement" if field == "engineering_measurement" else "unsupported_field"
        message = (
            "Engineering measurements are outside V1 unless a later calibrated contract explicitly enables them."
            if field == "engineering_measurement"
            else f"{field} is not part of the Stage 2 observation contract."
        )
        add(code, field, message)

    if observation.get("contract_version") != STAGE2_CONTRACT.get("contract_version"):
        add("contract_version_mismatch", "contract_version", "The observation must use the active contract version.")
    if not isinstance(observation.get("observation_id"), str) or not observation.get("observation_id", "").strip():
        add("invalid_observation_id", "observation_id", "A non-empty observation identifier is required.")

    surface_ids = {item["id"] for item in STAGE2_CONTRACT["surfaces"]}
    defect_map = {item["id"]: item for item in STAGE2_CONTRACT["defects"]}
    surface_type = observation.get("surface_type")
    damage_type = observation.get("damage_type")
    if surface_type not in surface_ids:
        add("invalid_surface_label", "surface_type", "The surface label is not in the frozen V1 taxonomy.")
    if damage_type not in defect_map:
        add("invalid_damage_label", "damage_type", "The damage label is not in the frozen V1 taxonomy.")
    elif surface_type in surface_ids and surface_type not in defect_map[damage_type]["applicable_surfaces"]:
        add("surface_damage_mismatch", "damage_type", "This damage label is not valid for the selected surface.")

    region = observation.get("evidence_region")
    if not isinstance(region, dict):
        add("invalid_evidence_region", "evidence_region", "A normalised bounding box or polygon is required.")
    elif region.get("type") == "bounding_box":
        values = [region.get(key) for key in ("x_min", "y_min", "x_max", "y_max")]
        if not all(_is_number(value) and 0 <= float(value) <= 1 for value in values):
            add("invalid_evidence_region", "evidence_region", "Bounding-box coordinates must be numeric values from 0 to 1.")
        elif not (float(values[0]) < float(values[2]) and float(values[1]) < float(values[3])):
            add("invalid_evidence_region", "evidence_region", "Bounding-box minimum coordinates must be smaller than maximum coordinates.")
    elif region.get("type") == "polygon":
        points = region.get("points")
        valid_points = isinstance(points, list) and len(points) >= 3 and all(
            isinstance(point, list)
            and len(point) == 2
            and all(_is_number(value) and 0 <= float(value) <= 1 for value in point)
            for point in (points or [])
        )
        if not valid_points:
            add("invalid_evidence_region", "evidence_region", "A polygon requires at least three normalised [x, y] points.")
    else:
        add("invalid_evidence_region", "evidence_region.type", "Region type must be bounding_box or polygon.")

    confidence = observation.get("confidence")
    if not _is_number(confidence) or not 0 <= float(confidence) <= 1:
        add("confidence_out_of_range", "confidence", "Confidence must be a numeric value from 0 to 1.")

    uncertainty = observation.get("uncertainty")
    uncertainty_states = {"review_required", "out_of_scope", "insufficient_evidence"}
    if not isinstance(uncertainty, dict) or uncertainty.get("state") not in uncertainty_states or not isinstance(uncertainty.get("reasons"), list):
        add("invalid_uncertainty", "uncertainty", "Uncertainty state and a reasons list are required.")

    location = observation.get("location")
    if not isinstance(location, dict):
        add("missing_location_provenance", "location", "GPS coordinates and a location method are required.")
    else:
        latitude, longitude = location.get("latitude"), location.get("longitude")
        if not _is_number(latitude) or not -90 <= float(latitude) <= 90:
            add("invalid_latitude", "location.latitude", "Latitude must be between -90 and 90.")
        if not _is_number(longitude) or not -180 <= float(longitude) <= 180:
            add("invalid_longitude", "location.longitude", "Longitude must be between -180 and 180.")
        if location.get("method") not in {"source_metadata", "survey_control", "network_match", "manual_review"}:
            add("invalid_location_method", "location.method", "A declared location method is required.")

    evidence = observation.get("evidence")
    source_policies = {item["id"]: item for item in STAGE2_CONTRACT["source_policies"]}
    if not isinstance(evidence, dict):
        add("missing_provenance", "evidence", "Source, asset, capture time, permission and attribution are required.")
    else:
        for field in ("source_id", "asset_id", "captured_at", "licence_status", "permission_record_id", "attribution"):
            if not isinstance(evidence.get(field), str) or not evidence.get(field, "").strip():
                add("missing_provenance", f"evidence.{field}", f"Evidence {field} is required.")
        source_id = evidence.get("source_id")
        source_policy = source_policies.get(source_id)
        if source_policy is None:
            add("unknown_source", "evidence.source_id", "The imagery source is not registered in the Stage 2 contract.")
        elif source_policy.get("automated_inference") not in {"permitted", "conditional"}:
            add("unapproved_source", "evidence.source_id", "This source is not approved for automated Function 1 inference.")
        if evidence.get("licence_status") != "verified":
            add("unverified_licence", "evidence.licence_status", "Licence status must be verified before inference.")

    model = observation.get("model")
    if not isinstance(model, dict) or any(not isinstance(model.get(field), str) or not model.get(field, "").strip() for field in ("name", "version", "model_card_id")):
        add("missing_model_provenance", "model", "Model name, version and model-card identifier are required.")

    review = observation.get("review")
    review_states = {item["id"] for item in STAGE2_CONTRACT["review_states"]}
    if not isinstance(review, dict) or review.get("status") not in review_states:
        add("invalid_review_state", "review.status", "A valid human-review state is required.")

    return errors


def _run_stage2_contract_checks() -> list[dict[str, Any]]:
    sample = STAGE2_CONTRACT["sample_observation"]
    cases: list[tuple[str, str, dict[str, Any], bool, str | None]] = []
    cases.append(("valid_sample", "Valid observation passes", copy.deepcopy(sample), True, None))

    invalid_label = copy.deepcopy(sample)
    invalid_label["damage_type"] = "invented_damage"
    cases.append(("invalid_label", "Unknown defect label is rejected", invalid_label, False, "invalid_damage_label"))

    invalid_confidence = copy.deepcopy(sample)
    invalid_confidence["confidence"] = 1.4
    cases.append(("confidence_range", "Confidence outside 0–1 is rejected", invalid_confidence, False, "confidence_out_of_range"))

    missing_provenance = copy.deepcopy(sample)
    missing_provenance.pop("evidence")
    cases.append(("missing_provenance", "Missing evidence provenance is rejected", missing_provenance, False, "missing_provenance"))

    unapproved_source = copy.deepcopy(sample)
    unapproved_source["evidence"]["source_id"] = "google_maps_core"
    cases.append(("unapproved_source", "Google or other unapproved AI source is rejected", unapproved_source, False, "unapproved_source"))

    unsupported_measurement = copy.deepcopy(sample)
    unsupported_measurement["engineering_measurement"] = {"depth_mm": 35}
    cases.append(("unsupported_measurement", "Uncalibrated engineering measurement is rejected", unsupported_measurement, False, "unsupported_engineering_measurement"))

    results: list[dict[str, Any]] = []
    for check_id, label, observation, should_be_valid, expected_error in cases:
        errors = _validate_stage2_observation(observation)
        valid = not errors
        error_codes = {error["code"] for error in errors}
        passed = valid == should_be_valid and (expected_error is None or expected_error in error_codes)
        results.append({
            "id": check_id,
            "label": label,
            "passed": passed,
            "observed_valid": valid,
            "error_codes": sorted(error_codes),
        })
    return results


app = FastAPI(
    title="AI-based Transport Infrastructure Perception Workspace",
    version="1.2.0-stage8-human-annotation",
    description="Functionally isolated imagery, governed point inference, optional human annotation and a versioned training-data queue within the TWM platform.",
)


@app.middleware("http")
async def disable_perception_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    content_type = str(response.headers.get("content-type") or "").lower()
    if (
        "text/html" in content_type
        or "text/css" in content_type
        or "javascript" in content_type
        or path.endswith((".html", ".css", ".js"))
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Clear-Site-Data"] = '"cache"'
    return response

app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="perception-static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_ROOT), name="perception-uploads")
app.mount("/vendor", StaticFiles(directory=MAPLIBRE_VENDOR), name="maplibre-vendor")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    if not stage6_deployment.registry_path.is_file():
        return {
            "status": "degraded",
            "workspace": "infrastructure-perception",
            "reason": "Model packages and private research datasets are not included in the source release.",
            "ai_contract_ready": True,
            "model_trained": False,
            "ai_detection_enabled": False,
            "stage6_research_deployment_ready": False,
            "stage7_on_demand_runtime_ready": False,
            "stage7_exact_current_view_runtime_ready": False,
            "stage8_human_annotation_ready": True,
            "automatic_operational_action_enabled": False,
        }
    dataset_status = dataset_readiness_report(verify_files=False)
    model_status = stage4_report(verify_artifacts=False)
    validation_status = stage5_report(verify_artifacts=False)
    validation_v2_status = stage5_v2_report(verify_artifacts=False) if STAGE5_V2_COMPLETION_PATH.is_file() else None
    stage6_status = stage6_deployment.status()
    stage6_ready = stage6_status["status"] == "research_deployment_ready"
    stage7_status = stage7_service.status(stage6_status)
    stage8_status = stage8_annotations.status()
    return {
        "status": "healthy",
        "workspace": "infrastructure-perception",
        "stage": 7,
        "capability": "exact-current-view Google Street View analysis, frame-locked class-coloured detections, explicit three-state outcomes and versioned TWM-InfraDet inference",
        "storage": "isolated",
        "ai_contract_ready": True,
        "dataset_contract_ready": True,
        "dataset_ready": model_status["model_trained"],
        "dataset_asset_count": dataset_status["stats"]["asset_count"],
        "model_trained": model_status["model_trained"],
        "independent_validation_complete": validation_status["independent_validation_executed"],
        "stage5_remediation_v2_complete": validation_v2_status is not None,
        "stage6_eligible": validation_v2_status["stage6_eligible"] if validation_v2_status else False,
        "stage6_research_deployment_ready": stage6_ready,
        "stage6_active_model": stage6_status["active_model"]["model_version"],
        "stage7_on_demand_runtime_ready": stage7_status["status"] == "on_demand_runtime_ready",
        "stage7_exact_current_view_runtime_ready": stage7_status["status"] == "on_demand_runtime_ready",
        "stage7_target_area_detection_proof_confirmed": stage7_status["target_area_detection_proof_confirmed"],
        "stage7_arbitrary_target_area_completion_confirmed": stage7_status["completion_boundary_met"],
        "stage7_completion_boundary": stage7_status["completion_boundary"],
        "stage8_human_annotation_ready": stage8_status["status"] == "human_annotation_ready",
        "stage8_training_queue_summary": stage8_status["summary"],
        "target_area_validation_complete": validation_status["target_area_validation_complete"],
        "ai_detection_enabled": stage6_ready,
        "ai_detection_scope": "authorised Stage 6 evidence plus user-initiated exact-current-view Stage 7 visual frames",
        "twm_transfer_enabled": False,
        "twm_confirmed_observation_export_ready": stage6_ready,
        "automatic_operational_action_enabled": False,
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    if not stage6_deployment.registry_path.is_file():
        unavailable = {"version": None, "status": "not_installed", "status_label": "Not included in the source release", "stage6_eligible": False}
        return {
            "workspace_name": "AI-based Transport Infrastructure Perception",
            "default_location": "Default map view",
            "map": DEFAULT_MAP_CENTRE,
            "accepted_formats": sorted(ALLOWED_SUFFIXES),
            "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
            "satellite_collections": SATELLITE_COLLECTIONS,
            "ai_contract": {"version": STAGE2_CONTRACT["contract_version"], "status": STAGE2_CONTRACT["status"], "status_label": STAGE2_CONTRACT["status_label"]},
            "dataset_v1": dict(unavailable), "model_v1": dict(unavailable),
            "validation_v1": dict(unavailable), "validation_v2": dict(unavailable),
            "stage6": {**unavailable, "active_model": None, "capabilities": {}},
            "stage7": {**unavailable, "inference_modes": {"local": {"ready": False}, "enhanced": {"ready": False}}},
            "stage8": stage8_annotations.status(),
            "boundary": "Review and mapping source is included. Automated detection requires separately licensed model packages and permitted imagery.",
        }
    validation_v2_status = stage5_v2_report(verify_artifacts=False) if STAGE5_V2_COMPLETION_PATH.is_file() else None
    stage6_status = stage6_deployment.status()
    stage7_status = stage7_service.status(stage6_status)
    stage8_status = stage8_annotations.status()
    return {
        "workspace_name": "AI-based Transport Infrastructure Perception",
        "default_location": "Default map view",
        "map": DEFAULT_MAP_CENTRE,
        "accepted_formats": sorted(ALLOWED_SUFFIXES),
        "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
        "satellite_collections": SATELLITE_COLLECTIONS,
        "ai_contract": {
            "version": STAGE2_CONTRACT["contract_version"],
            "status": STAGE2_CONTRACT["status"],
            "status_label": STAGE2_CONTRACT["status_label"],
        },
        "dataset_v1": {
            "version": STAGE3_CONTRACT["contract_version"],
            "status": STAGE3_CONTRACT["status"],
            "status_label": STAGE3_CONTRACT["status_label"],
        },
        "model_v1": {
            "version": load_stage4_contract()["contract_version"],
            "status": stage4_report(verify_artifacts=False)["status"],
            "status_label": stage4_report(verify_artifacts=False)["status_label"],
        },
        "validation_v1": {
            "version": load_stage5_contract()["contract_version"],
            "status": stage5_report(verify_artifacts=False)["status"],
            "status_label": stage5_report(verify_artifacts=False)["status_label"],
            "stage6_eligible": stage5_report(verify_artifacts=False)["stage6_eligible"],
        },
        "validation_v2": {
            "version": load_stage5_v2_contract()["contract_version"],
            "status": validation_v2_status["status"] if validation_v2_status else "stage5_v2_in_progress",
            "status_label": validation_v2_status["status_label"] if validation_v2_status else "Stage 5 v2 detector comparison and validation in progress",
            "stage6_eligible": validation_v2_status["stage6_eligible"] if validation_v2_status else False,
        },
        "stage6": {
            "status": stage6_status["status"],
            "scope": stage6_status["scope"],
            "active_model": stage6_status["active_model"],
            "capabilities": stage6_status["capabilities"],
            "automatic_operational_action_enabled": False,
        },
        "stage7": stage7_status,
        "stage8": stage8_status,
        "boundary": "Stage 7 returns an explicit outcome in the same Google road-selection workspace. Permitted KartaView or geotagged open Wikimedia Commons road imagery may be analysed and is shown with its evidence coordinates and distance. Every detection requires human review and no automatic operational action is permitted.",
    }


@app.get("/api/ai-contract")
def ai_contract() -> dict[str, Any]:
    payload = copy.deepcopy(STAGE2_CONTRACT)
    payload["schema_url"] = "api/ai-contract/schema"
    payload["validation_url"] = "api/ai-contract/validate"
    payload["checks_url"] = "api/ai-contract/checks"
    return payload


@app.get("/api/ai-contract/schema")
def ai_contract_schema() -> dict[str, Any]:
    return copy.deepcopy(OBSERVATION_SCHEMA)


@app.post("/api/ai-contract/validate")
def validate_ai_contract(request: ContractValidationRequest) -> dict[str, Any]:
    errors = _validate_stage2_observation(request.observation)
    return {
        "contract_version": STAGE2_CONTRACT["contract_version"],
        "valid": not errors,
        "errors": errors,
    }


@app.post("/api/ai-contract/checks")
def run_ai_contract_checks() -> dict[str, Any]:
    checks = _run_stage2_contract_checks()
    passed = sum(1 for check in checks if check["passed"])
    return {
        "contract_version": STAGE2_CONTRACT["contract_version"],
        "status": STAGE2_CONTRACT["status"],
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
    }


@app.get("/api/dataset-v1")
def dataset_v1() -> dict[str, Any]:
    payload = copy.deepcopy(STAGE3_CONTRACT)
    payload["schema_url"] = "api/dataset-v1/schema"
    payload["source_registry_url"] = "api/dataset-v1/sources"
    payload["manifest_url"] = "api/dataset-v1/manifest"
    payload["validation_url"] = "api/dataset-v1/validate"
    payload["checks_url"] = "api/dataset-v1/checks"
    payload["readiness"] = dataset_readiness_report(verify_files=True)
    return payload


@app.get("/api/dataset-v1/schema")
def dataset_v1_schema() -> dict[str, Any]:
    return copy.deepcopy(DATASET_ASSET_SCHEMA)


@app.get("/api/dataset-v1/sources")
def dataset_v1_sources() -> dict[str, Any]:
    return copy.deepcopy(SOURCE_REGISTRY)


@app.get("/api/dataset-v1/manifest")
def dataset_v1_manifest() -> dict[str, Any]:
    return load_current_manifest()


@app.post("/api/dataset-v1/validate")
def validate_dataset_v1(request: DatasetValidationRequest) -> dict[str, Any]:
    result = validate_manifest(request.manifest, verify_files=False)
    return {
        "contract_version": STAGE3_CONTRACT["contract_version"],
        **result,
    }


@app.post("/api/dataset-v1/checks")
def run_dataset_v1_checks() -> dict[str, Any]:
    validator_checks = run_dataset_contract_checks()
    passed = sum(1 for check in validator_checks if check["passed"])
    readiness = dataset_readiness_report(verify_files=True)
    return {
        "contract_version": STAGE3_CONTRACT["contract_version"],
        "validator_passed": passed,
        "validator_total": len(validator_checks),
        "validator_all_passed": passed == len(validator_checks),
        "validator_checks": validator_checks,
        "readiness": readiness,
    }


@app.get("/api/model-v1")
def model_v1() -> dict[str, Any]:
    return stage4_report(verify_artifacts=True)


@app.get("/api/model-v1/contract")
def model_v1_contract() -> dict[str, Any]:
    return load_stage4_contract()


@app.get("/api/model-v1/model-card")
def model_v1_card() -> dict[str, Any]:
    return load_model_card()


@app.post("/api/model-v1/checks")
def run_model_v1_checks() -> dict[str, Any]:
    return stage4_checks()


@app.get("/api/model-v1/visualisation/{name}")
def model_v1_visualisation(name: str) -> FileResponse:
    path = visualisation_path(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown Stage 4 visualisation.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/validation-v1")
def validation_v1() -> dict[str, Any]:
    return stage5_report(verify_artifacts=True)


@app.get("/api/validation-v1/contract")
def validation_v1_contract() -> dict[str, Any]:
    return load_stage5_contract()


@app.post("/api/validation-v1/checks")
def run_validation_v1_checks() -> dict[str, Any]:
    return stage5_checks()


@app.get("/api/validation-v1/report")
def validation_v1_report() -> FileResponse:
    if not STAGE5_REPORT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Stage 5 report is unavailable.")
    return FileResponse(STAGE5_REPORT_PATH, media_type="text/markdown", filename="STAGE5_REPORT.md")


@app.get("/api/validation-v1/visualisation/{name}")
def validation_v1_visualisation(name: str) -> FileResponse:
    record = stage5_visualisation_path(name)
    if record is None or not record[0].is_file():
        raise HTTPException(status_code=404, detail="Unknown Stage 5 visualisation.")
    return FileResponse(record[0], media_type=record[1])


@app.get("/api/validation-v2")
def validation_v2() -> dict[str, Any]:
    if not STAGE5_V2_COMPLETION_PATH.is_file():
        raise HTTPException(status_code=404, detail="Stage 5 v2 is still in progress; no frozen completion record exists.")
    return stage5_v2_report(verify_artifacts=True)

@app.get("/api/twm-infradet-stage6")
def twm_infradet_stage6_status() -> dict[str, Any]:
    payload = stage6_twm_status()
    payload["deployment"] = stage6_deployment.status()
    return payload


def _raise_stage6_http(error: Stage6DeploymentError) -> None:
    if isinstance(error, Stage6NotFoundError):
        status_code = 404
    elif isinstance(error, Stage6ConflictError):
        status_code = 409
    else:
        status_code = 503
    raise HTTPException(status_code=status_code, detail=str(error)) from error


@app.get("/api/stage6")
def stage6_status() -> dict[str, Any]:
    try:
        return stage6_deployment.status()
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.post("/api/stage6/self-test")
def stage6_self_test() -> dict[str, Any]:
    try:
        return stage6_deployment.self_test()
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.post("/api/stage6/images/{image_id}/infer")
def stage6_infer_image(
    image_id: str,
    request: Stage6InferenceRequest,
) -> dict[str, Any]:
    catalogue = _load_catalogue()
    record = next((item for item in catalogue["images"] if item.get("id") == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    try:
        return stage6_deployment.infer_uploaded_image(
            image_record=record,
            image_path=UPLOAD_ROOT / record["stored_name"],
            surface_type=request.surface_type,
        )
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.get("/api/stage6/activity")
def stage6_activity(image_id: str | None = Query(default=None, max_length=64)) -> dict[str, Any]:
    try:
        return stage6_deployment.activity(image_id=image_id)
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.patch("/api/stage6/observations/{observation_id}/review")
def stage6_review_observation(
    observation_id: str,
    request: Stage6ReviewRequest,
) -> dict[str, Any]:
    try:
        return stage6_deployment.review_observation(
            observation_id,
            status=request.status,
            reviewer_id=request.reviewer_id,
            note=request.note,
        )
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.get("/api/stage6/observations/export")
def stage6_export_observations(
    review_status: str = Query(default="confirmed"),
) -> dict[str, Any]:
    if review_status not in {"pending", "confirmed", "rejected", "uncertain", "all"}:
        raise HTTPException(status_code=422, detail="Unknown review status.")
    try:
        activity = stage6_deployment.activity()
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)
    envelopes = activity["observations"]
    if review_status != "all":
        envelopes = [
            item
            for item in envelopes
            if item["observation"]["review"]["status"] == review_status
        ]
    return {
        "contract_version": "1.0.0",
        "review_status": review_status,
        "count": len(envelopes),
        "observations": [item["observation"] for item in envelopes],
        "automatic_transfer_executed": False,
        "automatic_operational_action_enabled": False,
    }


@app.post("/api/stage6/models/activate")
def stage6_activate_model(request: Stage6ActivationRequest) -> dict[str, Any]:
    try:
        state = stage6_deployment.activate(
            request.model_id,
            expected_checkpoint_sha256=request.expected_checkpoint_sha256,
            actor=request.actor,
            reason=request.reason,
            confirmed=request.confirmed,
        )
        return {"activated": True, "state": state}
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.post("/api/stage6/models/rollback")
def stage6_rollback_model(request: Stage6RollbackRequest) -> dict[str, Any]:
    try:
        state = stage6_deployment.rollback(
            actor=request.actor,
            reason=request.reason,
            confirmed=request.confirmed,
        )
        return {"rolled_back": True, "state": state}
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)


@app.get("/api/validation-v2/contract")
def validation_v2_contract() -> dict[str, Any]:
    return load_stage5_v2_contract()


@app.post("/api/validation-v2/checks")
def run_validation_v2_checks() -> dict[str, Any]:
    if not STAGE5_V2_COMPLETION_PATH.is_file():
        raise HTTPException(status_code=404, detail="Stage 5 v2 is still in progress; no frozen completion record exists.")
    return stage5_v2_checks()


@app.get("/api/validation-v2/report")
def validation_v2_report() -> FileResponse:
    if not STAGE5_V2_REPORT_PATH.is_file():
        raise HTTPException(status_code=404, detail="Stage 5 v2 report is unavailable.")
    return FileResponse(STAGE5_V2_REPORT_PATH, media_type="text/markdown", filename="STAGE5_V2_REPORT.md")


@app.get("/api/stage7")
def stage7_status() -> dict[str, Any]:
    try:
        return stage7_service.status(stage6_deployment.status())
    except (Stage6DeploymentError, Stage7Error) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/stage7/contract")
def stage7_contract() -> dict[str, Any]:
    try:
        return stage7_service.contract()
    except Stage7Error as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/stage7/activity")
def stage7_activity(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    try:
        return stage7_service.activity(limit=limit)
    except Stage7Error as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/stage7/views/analyse")
def stage7_analyse_current_view(
    payload: Stage7CurrentViewRequest,
    request: Request,
    candidate_model: str | None = Query(default=None),
) -> dict[str, Any]:
    if candidate_model not in {None, "r07", "surface_precision", "surface_precision_v021", "classless_v022", "classless_local_v023", "classless_local_v024"}:
        raise HTTPException(status_code=400, detail="Unsupported Stage 7 candidate model.")
    referrer = None
    raw_referrer = str(request.headers.get("referer") or "").strip()
    if raw_referrer:
        parsed = urlparse(raw_referrer)
        if parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in {
            "127.0.0.1",
            "localhost",
        }:
            referrer = raw_referrer.split("?", 1)[0].split("#", 1)[0]
    inference_mode = payload.inference_mode
    if candidate_model in {"r07", "classless_local_v023", "classless_local_v024"}:
        inference_mode = "local"
    elif candidate_model in {"surface_precision", "surface_precision_v021"}:
        inference_mode = "enhanced"
    verifier_key = payload.stage7_api_key if inference_mode == "enhanced" else None
    try:
        return stage7_service.analyse_current_view(
            api_key=payload.api_key,
            stage7_api_key=verifier_key,
            inference_mode=inference_mode,
            frame_id=payload.frame_id,
            pano_id=payload.pano_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            heading=payload.heading,
            pitch=payload.pitch,
            zoom=payload.zoom,
            fov=payload.fov,
            width=payload.width,
            height=payload.height,
            viewport_width=payload.viewport_width,
            viewport_height=payload.viewport_height,
            frame_revision=payload.frame_revision,
            referrer=referrer,
            deployment=stage6_deployment,
            candidate_model=candidate_model,
        )
    except Stage7Error as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        verifier_key = None


@app.post("/api/stage7/points/analyse")
def stage7_analyse_point(request: Stage7PointAnalysisRequest) -> dict[str, Any]:
    photos: list[dict[str, Any]] = []
    provider_errors: list[str] = []
    if request.provider in {"auto", "kartaview_cc_by_sa"}:
        params = urlencode(
            {
                "lat": f"{request.latitude:.7f}",
                "lng": f"{request.longitude:.7f}",
                "radius": request.radius_m,
                "page": 1,
                "itemsPerPage": 50,
            }
        )
        try:
            payload = _fetch_json(f"{KARTAVIEW_PHOTO_URL}?{params}")
            photos = [
                photo
                for item in _kartaview_records(payload)
                if (
                    photo := _normalise_kartaview_photo(
                        item,
                        query_latitude=request.latitude,
                        query_longitude=request.longitude,
                    )
                )
            ]
            photos.sort(
                key=lambda item: (
                    item.get("distance_m") is None,
                    item.get("distance_m") or 0,
                    item.get("captured_at") or "",
                )
            )
        except HTTPException as error:
            provider_errors.append(f"KartaView: {error.detail}")
    requested_photo_missing = bool(
        request.photo_id and not any(str(item.get("id")) == request.photo_id for item in photos)
    )
    if request.provider in {"auto", "wikimedia_commons_open"} and (not photos or requested_photo_missing):
        try:
            photos.extend(
                _wikimedia_nearby_photos(
                    latitude=request.latitude,
                    longitude=request.longitude,
                    radius_m=request.radius_m,
                )
            )
        except HTTPException as error:
            provider_errors.append(f"Wikimedia Commons: {error.detail}")
    provider_error = "; ".join(provider_errors) if not photos and provider_errors else None
    try:
        return stage7_service.analyse(
            latitude=request.latitude,
            longitude=request.longitude,
            radius_m=request.radius_m,
            provider=request.provider,
            surface_type=request.surface_type,
            photo_id=request.photo_id,
            photos=photos,
            provider_error=provider_error,
            deployment=stage6_deployment,
        )
    except Stage7Error as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/stage7/observations/export")
def stage7_export_confirmed_observations() -> dict[str, Any]:
    try:
        activity = stage6_deployment.activity()
    except Stage6DeploymentError as error:
        _raise_stage6_http(error)
    envelopes = [
        item
        for item in activity["observations"]
        if item.get("origin_stage") == 7
        and item.get("observation", {}).get("review", {}).get("status") == "confirmed"
    ]
    return {
        "contract_version": "1.0.0",
        "review_status": "confirmed",
        "count": len(envelopes),
        "observations": [item["observation"] for item in envelopes],
        "automatic_transfer_executed": False,
        "automatic_operational_action_enabled": False,
    }


def _raise_stage8_http(error: Stage8AnnotationError) -> None:
    if isinstance(error, Stage8AnnotationNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, Stage8AnnotationConflictError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/stage8")
def stage8_status() -> dict[str, Any]:
    try:
        status = stage8_annotations.status()
        stage7_status = stage7_service.status(stage6_deployment.status())
    except (Stage6DeploymentError, Stage7Error, Stage8AnnotationError) as error:
        if isinstance(error, Stage8AnnotationError):
            _raise_stage8_http(error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        **status,
        "stage7_dependency_ready": stage7_status.get("status") == "on_demand_runtime_ready",
        "stage7_local_model": stage7_status.get("inference_modes", {}).get("local", {}).get("model_version"),
        "stage7_enhanced_model": stage7_status.get("inference_modes", {}).get("enhanced", {}).get("model_version"),
        "stage7_inference_changed": False,
        "stage6_changed": False,
        "model_registry_changed": False,
    }


@app.get("/api/stage8/contract")
def stage8_contract() -> dict[str, Any]:
    try:
        return stage8_annotations.contract()
    except Stage8AnnotationError as error:
        _raise_stage8_http(error)


@app.get("/api/stage8/queue")
def stage8_queue(
    limit: int = Query(default=100, ge=1, le=500),
    status: str | None = Query(default=None, max_length=80),
    frame_key: str | None = Query(default=None, min_length=64, max_length=64),
) -> dict[str, Any]:
    try:
        return stage8_annotations.list_records(limit=limit, status=status, frame_key=frame_key)
    except Stage8AnnotationError as error:
        _raise_stage8_http(error)


@app.post("/api/stage8/annotations")
def stage8_save_annotation(payload: Stage8ReviewSaveRequest) -> dict[str, Any]:
    try:
        return stage8_annotations.save_review(payload.model_dump(mode="json"))
    except Stage8AnnotationError as error:
        _raise_stage8_http(error)


@app.get("/api/stage8/annotations/{annotation_id}")
def stage8_get_annotation(annotation_id: str) -> dict[str, Any]:
    try:
        return stage8_annotations.get(annotation_id)
    except Stage8AnnotationError as error:
        _raise_stage8_http(error)


@app.patch("/api/stage8/annotations/{annotation_id}/status")
def stage8_update_annotation_status(
    annotation_id: str,
    payload: Stage8QueueStatusRequest,
) -> dict[str, Any]:
    try:
        return stage8_annotations.update_status(
            annotation_id,
            status=payload.status,
            model_version=payload.model_version,
        )
    except Stage8AnnotationError as error:
        _raise_stage8_http(error)


@app.get("/api/imagery/providers")
def imagery_providers() -> dict[str, Any]:
    return {
        "providers": [
            {
                "id": "copernicus",
                "name": "Copernicus Sentinel",
                "status": "connected",
                "access": "Free, full and open Sentinel catalogue access",
                "mode": "In-platform search and preview",
                "analysis_allowed": True,
                "attribution": "Contains modified Copernicus Sentinel data",
            },
            {
                "id": "bhuvan",
                "name": "ISRO Bhuvan Open Data Archive",
                "status": "catalogue-link",
                "access": "Selected Indian satellite data and derived products are free to download",
                "mode": "Open official archive; product-specific availability and terms apply",
                "analysis_allowed": "Depends on the selected product terms",
                "url": "https://bhuvan-app3.nrsc.gov.in/data/download/index.php",
            },
            {
                "id": "google-maps",
                "name": "Google Maps, satellite, Places (New) and Street View",
                "status": "user-connect",
                "access": "Restricted browser key supplied for the current tab only",
                "mode": "Road and satellite basemap, Places search, interactive outdoor Street View and exact-current-frame Stage 7 inference",
                "analysis_allowed": "user-initiated exact current panorama frame",
                "credential_storage": "none; the key is passed ephemerally for the current frame request and is never persisted",
                "attribution": "Google Maps Platform terms and in-product attribution apply",
            },
            {
                "id": "kartaview",
                "name": "KartaView open street imagery",
                "status": "connected",
                "access": "Public community street imagery; no API key is required for discovery",
                "mode": "Click-to-search, continuous sequence browsing and Stage 7 on-demand analysis",
                "analysis_allowed": True,
                "analysis_controls": "CC BY-SA 4.0 attribution and share-alike controls",
                "attribution": "© Grab and KartaView Contributors · CC BY-SA 4.0",
                "url": "https://kartaview.org/",
            },
            {
                "id": "wikimedia-commons",
                "name": "Wikimedia Commons geotagged road evidence",
                "status": "connected",
                "access": "Open files discovered by coordinate with per-file licence metadata",
                "mode": "Stage 7 fallback for road-relevant evidence in the selected-point workspace",
                "analysis_allowed": True,
                "analysis_controls": "Only verified CC BY, CC BY-SA, CC0 or public-domain files",
                "attribution": "Per-file creator and licence returned by Wikimedia Commons",
                "url": "https://commons.wikimedia.org/",
            },
            {
                "id": "user-ground",
                "name": "User ground imagery",
                "status": "connected",
                "access": "JPEG, PNG and WebP uploads",
                "mode": "EXIF location or manual map placement",
                "analysis_allowed": "Only where the uploader has appropriate rights",
            },
        ]
    }


@app.get("/api/street-imagery/nearby")
def nearby_street_imagery(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
    radius_m: int = Query(default=1000, ge=25, le=2000),
    limit: int = Query(default=30, ge=1, le=50),
) -> dict[str, Any]:
    params = urlencode({"lat": f"{latitude:.7f}", "lng": f"{longitude:.7f}", "radius": radius_m, "page": 1, "itemsPerPage": limit})
    payload = _fetch_json(f"{KARTAVIEW_PHOTO_URL}?{params}")
    photos = [
        photo
        for item in _kartaview_records(payload)
        if (photo := _normalise_kartaview_photo(item, query_latitude=latitude, query_longitude=longitude))
    ]
    photos.sort(key=lambda item: (item.get("distance_m") is None, item.get("distance_m") or 0, item.get("captured_at") or ""))
    return {
        "query": {"latitude": latitude, "longitude": longitude, "radius_m": radius_m},
        "count": len(photos[:limit]),
        "photos": photos[:limit],
        "source": "KartaView public street-imagery catalogue",
        "coverage": "Community-contributed coverage varies by location and date. An empty result is not a service error.",
        "scientific_limit": "Images are displayed for evidence discovery only. AI defect detection and TWM transfer are not enabled.",
    }


@app.get("/api/street-imagery/sequences/{sequence_id}")
def street_imagery_sequence(sequence_id: str, limit: int = Query(default=150, ge=1, le=150)) -> dict[str, Any]:
    safe_id = sequence_id.strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not safe_id or len(safe_id) > 80 or any(character not in allowed for character in safe_id):
        raise HTTPException(status_code=422, detail="The KartaView sequence identifier is invalid.")
    params = urlencode({"sequenceId": safe_id, "page": 1, "itemsPerPage": limit})
    payload = _fetch_json(f"{KARTAVIEW_PHOTO_URL}?{params}")
    photos = [photo for item in _kartaview_records(payload) if (photo := _normalise_kartaview_photo(item))]
    photos.sort(key=lambda item: (item.get("sequence_index") is None, item.get("sequence_index") or 0, item.get("captured_at") or ""))
    return {
        "sequence_id": safe_id,
        "count": len(photos[:limit]),
        "photos": photos[:limit],
        "source": "KartaView public street-imagery sequence",
        "scientific_limit": "Sequence browsing supports visual inspection only; it is not an inferred road-condition result.",
    }


@app.get("/api/locations/search")
def search_locations(q: str = Query(min_length=2, max_length=160), limit: int = Query(default=5, ge=1, le=8)) -> dict[str, Any]:
    query = " ".join(q.split())
    key = f"{query.casefold()}:{limit}"
    if key in _location_cache:
        return {"query": query, "count": len(_location_cache[key]), "results": _location_cache[key]}
    params = urlencode({"q": query, "format": "jsonv2", "addressdetails": 1, "limit": limit})
    payload = _fetch_json(f"{NOMINATIM_SEARCH_URL}?{params}")
    if not isinstance(payload, list):
        raise HTTPException(status_code=502, detail="The location service returned an unexpected response.")
    results: list[dict[str, Any]] = []
    for item in payload[:limit]:
        if not isinstance(item, dict):
            continue
        try:
            latitude = float(item["lat"])
            longitude = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_bbox = item.get("boundingbox") or []
        bbox = None
        if len(raw_bbox) == 4:
            try:
                south, north, west, east = (float(value) for value in raw_bbox)
                bbox = {"west": west, "south": south, "east": east, "north": north}
            except (TypeError, ValueError):
                bbox = None
        results.append(
            {
                "display_name": str(item.get("display_name") or query),
                "latitude": latitude,
                "longitude": longitude,
                "type": str(item.get("type") or item.get("category") or "place"),
                "bbox": bbox,
                "source": "OpenStreetMap Nominatim",
            }
        )
    _location_cache[key] = results
    return {"query": query, "count": len(results), "results": results}


@app.post("/api/satellite/search")
def search_satellite_scenes(search: SatelliteSearchRequest) -> dict[str, Any]:
    collection_meta = SATELLITE_COLLECTIONS[search.collection]
    payload: dict[str, Any] = {
        "collections": [search.collection],
        "bbox": search.aoi.as_list(),
        "datetime": f"{search.start_date.isoformat()}T00:00:00Z/{search.end_date.isoformat()}T23:59:59Z",
        "limit": search.limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if collection_meta["cloud_filter"]:
        payload["query"] = {"eo:cloud_cover": {"lte": search.max_cloud_cover}}
    response = _fetch_json(COPERNICUS_STAC_SEARCH_URL, payload=payload)
    if not isinstance(response, dict) or not isinstance(response.get("features"), list):
        raise HTTPException(status_code=502, detail="Copernicus returned an unexpected catalogue response.")
    scenes: list[dict[str, Any]] = []
    for feature in response["features"]:
        if not isinstance(feature, dict):
            continue
        scene = _normalise_scene(feature, search.collection)
        if scene:
            scenes.append(scene)
    scenes.sort(key=lambda scene: scene.get("captured_at") or "", reverse=True)
    return {
        "query": {
            "aoi": search.aoi.model_dump(),
            "collection": search.collection,
            "start_date": search.start_date.isoformat(),
            "end_date": search.end_date.isoformat(),
            "max_cloud_cover": search.max_cloud_cover if collection_meta["cloud_filter"] else None,
        },
        "count": len(scenes),
        "scenes": scenes,
        "source": "Copernicus Data Space Ecosystem STAC catalogue",
        "licence": "Sentinel data are available on a free, full and open basis; attribution is required.",
        "freshness": "Acquisition dates are reported per scene. Satellite imagery is not real-time.",
    }


@app.get("/api/satellite/previews/{token}")
def satellite_preview(token: str) -> Response:
    with _preview_lock:
        remote_url = _preview_sources.get(token)
    if remote_url is None:
        raise HTTPException(status_code=404, detail="Preview expired. Search the Sentinel catalogue again.")
    body, content_type = _fetch_preview(remote_url)
    return Response(content=body, media_type=content_type, headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/survey-project")
def get_survey_project() -> dict[str, Any]:
    if not SURVEY_PROJECT_PATH.exists():
        return {"saved": False, "project": None}
    try:
        project = json.loads(SURVEY_PROJECT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="The saved imagery survey could not be read.")
    return {"saved": True, "project": project}


@app.put("/api/survey-project")
def save_survey_project(project: SurveyProjectState) -> dict[str, Any]:
    payload = project.model_dump(mode="json")
    payload["schema_version"] = 1
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    temporary = SURVEY_PROJECT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, SURVEY_PROJECT_PATH)
    return {"saved": True, "project": payload}


@app.delete("/api/survey-project")
def clear_survey_project() -> dict[str, Any]:
    SURVEY_PROJECT_PATH.unlink(missing_ok=True)
    return {"cleared": True}


@app.get("/api/images")
def list_images() -> dict[str, Any]:
    records = [_public_record(record) for record in _load_catalogue()["images"]]
    return {
        "count": len(records),
        "located": sum(record.get("latitude") is not None and record.get("longitude") is not None for record in records),
        "unlocated": sum(record.get("latitude") is None or record.get("longitude") is None for record in records),
        "images": records,
    }


@app.post("/api/images/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    source: str = Form(default="User upload"),
    analysis_authorised: bool = Form(default=False),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one image.")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Upload no more than 100 images at once.")

    catalogue = _load_catalogue()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for upload in files:
        original_name = Path(upload.filename or "image").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            rejected.append({"name": original_name, "reason": "Unsupported image format"})
            continue
        data = await upload.read(MAX_FILE_BYTES + 1)
        if not data or len(data) > MAX_FILE_BYTES:
            rejected.append({"name": original_name, "reason": "Empty image or file exceeds 30 MB"})
            continue

        identifier = uuid.uuid4().hex
        stored_name = f"{identifier}{suffix}"
        image_path = UPLOAD_ROOT / stored_name
        image_path.write_bytes(data)
        try:
            metadata = _extract_metadata(image_path)
        except (UnidentifiedImageError, OSError, ValueError):
            image_path.unlink(missing_ok=True)
            rejected.append({"name": original_name, "reason": "The file is not a readable image"})
            continue

        record = {
            "id": identifier,
            "original_name": original_name,
            "stored_name": stored_name,
            "source": (source or "User upload").strip()[:120],
            "source_id": "individual_user_upload",
            "analysis_authorised": analysis_authorised,
            "licence_status": "verified" if analysis_authorised else "unverified",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "captured_at": metadata["captured_at"],
            "latitude": metadata["latitude"],
            "longitude": metadata["longitude"],
            "direction": metadata["direction"],
            "width": metadata["width"],
            "height": metadata["height"],
            "location_method": "image metadata" if metadata["latitude"] is not None else "not located",
        }
        catalogue["images"].append(record)
        accepted.append(_public_record(record))

    if accepted:
        _save_catalogue(catalogue)
    return {"accepted": accepted, "rejected": rejected}


@app.patch("/api/images/{image_id}/location")
def update_location(image_id: str, update: LocationUpdate) -> dict[str, Any]:
    catalogue = _load_catalogue()
    record = next((item for item in catalogue["images"] if item.get("id") == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    record["latitude"] = update.latitude
    record["longitude"] = update.longitude
    if update.direction is not None:
        record["direction"] = update.direction
    record["location_method"] = "manual map placement"
    _save_catalogue(catalogue)
    return _public_record(record)


@app.patch("/api/images/{image_id}/analysis-authorisation")
def update_analysis_authorisation(
    image_id: str,
    update: AnalysisAuthorisationUpdate,
) -> dict[str, Any]:
    if update.confirmed is not True:
        raise HTTPException(status_code=422, detail="Analysis rights must be explicitly confirmed.")
    catalogue = _load_catalogue()
    record = next((item for item in catalogue["images"] if item.get("id") == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    record["source_id"] = "individual_user_upload"
    record["analysis_authorised"] = True
    record["licence_status"] = "verified"
    record["analysis_authorisation"] = {
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "declaration": update.declaration.strip(),
    }
    _save_catalogue(catalogue)
    return _public_record(record)


@app.get("/api/images/nearby")
def nearby_images(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
    radius_km: float = Query(default=2.0, gt=0, le=100),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for record in _load_catalogue()["images"]:
        if record.get("latitude") is None or record.get("longitude") is None:
            continue
        distance = _haversine_m(latitude, longitude, float(record["latitude"]), float(record["longitude"]))
        if distance <= radius_km * 1000:
            ranked.append((distance, record))
    ranked.sort(key=lambda item: (item[0], item[1].get("captured_at") or item[1].get("uploaded_at") or ""))
    return {
        "query": {"latitude": latitude, "longitude": longitude, "radius_km": radius_km},
        "count": min(len(ranked), limit),
        "images": [_public_record(record, distance) for distance, record in ranked[:limit]],
    }


@app.delete("/api/images/{image_id}")
def delete_image(image_id: str) -> dict[str, Any]:
    catalogue = _load_catalogue()
    record = next((item for item in catalogue["images"] if item.get("id") == image_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    catalogue["images"] = [item for item in catalogue["images"] if item.get("id") != image_id]
    (UPLOAD_ROOT / record["stored_name"]).unlink(missing_ok=True)
    _save_catalogue(catalogue)
    return {"deleted": image_id}
