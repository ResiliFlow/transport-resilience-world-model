from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = PERCEPTION_ROOT / "contracts" / "stage8_annotation_contract.json"
DEFAULT_STORAGE_ROOT = PERCEPTION_ROOT / "storage"


class Stage8AnnotationError(RuntimeError):
    pass


class Stage8AnnotationNotFoundError(Stage8AnnotationError):
    pass


class Stage8AnnotationConflictError(Stage8AnnotationError):
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
    except FileNotFoundError as error:
        raise Stage8AnnotationError(f"Required Stage 8 file is missing: {path.name}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise Stage8AnnotationError(f"Required Stage 8 file is unreadable: {path.name}") from error
    if not isinstance(payload, dict):
        raise Stage8AnnotationError(f"Required Stage 8 file is not a JSON object: {path.name}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _empty_queue() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue_id": "prototype5-stage8-annotation-queue-v1",
        "updated_at": None,
        "records": [],
    }


def _frame_key(frame: dict[str, Any]) -> str:
    identity = json.dumps(
        {
            "frame_id": frame["frame_id"],
            "frame_revision": frame["frame_revision"],
            "frame_sha256": frame["frame_sha256"],
            "pano_id": frame["pano_id"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest().upper()


def _safe_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(model.get(key))
        for key in ("id", "name", "version", "checkpoint_sha256", "components", "strategy")
        if model.get(key) is not None
    }


def _safe_detection(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item.get(key))
        for key in (
            "detection_id",
            "damage_type",
            "surface_type",
            "confidence",
            "box",
            "evidence_region",
            "consensus",
        )
        if item.get(key) is not None
    }


def _safe_annotation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_box_id": item["annotation_box_id"],
        "origin": item["origin"],
        "damage_type": item.get("damage_type"),
        "surface_type": item.get("surface_type"),
        "uncertain": bool(item.get("uncertain")),
        "box": [float(value) for value in item["box"]],
        "source_detection_id": item.get("source_detection_id"),
        "model_confidence": item.get("model_confidence"),
    }


def _assert_no_credentials(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    if "sk-" in encoded or "AIza" in encoded:
        raise Stage8AnnotationError("Credentials must never enter the Stage 8 annotation queue.")


class Stage8AnnotationQueue:
    def __init__(
        self,
        *,
        contract_path: Path = DEFAULT_CONTRACT_PATH,
        storage_root: Path = DEFAULT_STORAGE_ROOT,
    ) -> None:
        self.contract_path = Path(contract_path)
        self.storage_root = Path(storage_root)
        self.queue_path = self.storage_root / "stage8_annotation_queue.json"
        self._lock = threading.RLock()

    def contract(self) -> dict[str, Any]:
        contract = _load_json(self.contract_path)
        if (
            contract.get("schema_version") != 1
            or contract.get("stage") != 8
            or contract.get("contract_id") != "prototype5-stage8-human-annotation-v1"
        ):
            raise Stage8AnnotationError("The Stage 8 annotation contract has an invalid identity.")
        return contract

    def _queue(self) -> dict[str, Any]:
        if not self.queue_path.exists():
            return _empty_queue()
        queue = _load_json(self.queue_path)
        if (
            queue.get("schema_version") != 1
            or queue.get("queue_id") != "prototype5-stage8-annotation-queue-v1"
            or not isinstance(queue.get("records"), list)
        ):
            raise Stage8AnnotationError("The Stage 8 annotation queue is invalid.")
        return queue

    @staticmethod
    def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = {
            "human_reviewed": 0,
            "ready_for_training": 0,
            "partial_ready_for_training": 0,
            "used_in_model_version": 0,
            "needs_resolution": 0,
            "rejected": 0,
            "clean_negative_frame": 0,
        }
        for record in records:
            status = str(record.get("status") or "")
            if status in statuses:
                statuses[status] += 1
        training_records = [
            record
            for record in records
            if record.get("training_ready") and record.get("status") != "rejected"
        ]
        training_annotations = sum(
            len(record.get("human_review", {}).get("annotations") or [])
            for record in training_records
        )
        pending_annotations = sum(
            len(record.get("human_review", {}).get("pending_annotations") or [])
            for record in records
            if record.get("status") != "rejected"
        )
        return {
            "total": len(records),
            "positive": sum(1 for record in records if record.get("human_review", {}).get("annotations")),
            "clean": statuses["clean_negative_frame"],
            "needs_resolution": statuses["needs_resolution"],
            "ready_for_training": statuses["ready_for_training"] + statuses["partial_ready_for_training"],
            "complete_ready_for_training": statuses["ready_for_training"],
            "partial_ready_for_training": statuses["partial_ready_for_training"],
            "training_units": len(training_records),
            "training_annotations": training_annotations,
            "pending_annotations": pending_annotations,
            "used_in_model_version": statuses["used_in_model_version"],
            "rejected": statuses["rejected"],
            "by_status": statuses,
        }

    def status(self) -> dict[str, Any]:
        contract = self.contract()
        with self._lock:
            queue = self._queue()
        return {
            "stage": 8,
            "status": "human_annotation_ready",
            "contract_id": contract["contract_id"],
            "contract_version": contract["version"],
            "contract_sha256": _sha256(self.contract_path),
            "queue_storage": "isolated_local_versioned_metadata",
            "provider_pixels_persisted": False,
            "api_credentials_accepted": False,
            "model_training_enabled": False,
            "automatic_model_activation_enabled": False,
            "summary": self._summary(queue["records"]),
        }

    def list_records(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
        frame_key: str | None = None,
    ) -> dict[str, Any]:
        self.contract()
        with self._lock:
            queue = self._queue()
            records = list(queue["records"])
        if status:
            records = [record for record in records if record.get("status") == status]
        if frame_key:
            records = [record for record in records if record.get("frame_key") == frame_key]
        records.sort(key=lambda record: str(record.get("saved_at") or ""), reverse=True)
        return {
            "queue_id": queue["queue_id"],
            "count": min(len(records), limit),
            "records": records[:limit],
            "summary": self._summary(queue["records"]),
        }

    def get(self, annotation_id: str) -> dict[str, Any]:
        with self._lock:
            queue = self._queue()
            record = next(
                (item for item in queue["records"] if item.get("annotation_id") == annotation_id),
                None,
            )
        if record is None:
            raise Stage8AnnotationNotFoundError("The requested Stage 8 annotation does not exist.")
        return copy.deepcopy(record)

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        contract = self.contract()
        _assert_no_credentials(payload)
        now = _timestamp()
        frame = copy.deepcopy(payload["frame"])
        key = _frame_key(frame)
        annotations = [_safe_annotation(item) for item in payload.get("annotations") or []]
        pending_annotations = [
            _safe_annotation(item) for item in payload.get("pending_annotations") or []
        ]
        uncertain = any(item["uncertain"] for item in annotations)
        clean_frame = bool(payload.get("clean_frame"))
        submission_scope = str(payload.get("submission_scope") or "complete_review")
        if clean_frame:
            submission_scope = "clean_frame"
        if clean_frame:
            status = "clean_negative_frame"
            training_ready = True
        elif uncertain:
            status = "needs_resolution"
            training_ready = False
        elif submission_scope == "partial_review":
            status = "partial_ready_for_training"
            training_ready = True
        else:
            status = "ready_for_training"
            training_ready = True

        with self._lock:
            queue = self._queue()
            replay = next(
                (
                    item
                    for item in queue["records"]
                    if item.get("submission_id") == payload["submission_id"]
                ),
                None,
            )
            if replay is not None:
                return {
                    "record": copy.deepcopy(replay),
                    "summary": self._summary(queue["records"]),
                    "idempotent_replay": True,
                }
            frame_versions = [
                int(item.get("review_version") or 0)
                for item in queue["records"]
                if item.get("frame_key") == key
            ]
            review_version = max(frame_versions, default=0) + 1
            annotation_id = f"stage8-annotation-{uuid.uuid4().hex}"
            workflow = [
                {"status": "draft", "at": payload.get("draft_started_at") or now},
                {"status": "human_reviewed", "at": now},
            ]
            if status != "human_reviewed":
                workflow.append({"status": status, "at": now})
            record = {
                "schema_version": 1,
                "annotation_id": annotation_id,
                "submission_id": payload["submission_id"],
                "frame_key": key,
                "review_version": review_version,
                "submission_scope": submission_scope,
                "status": status,
                "training_ready": training_ready,
                "used_in_model_version": None,
                "saved_at": now,
                "frame": frame,
                "selected_point": copy.deepcopy(payload["selected_point"]),
                "inference_mode": payload["inference_mode"],
                "source_model": _safe_model(payload["source_model"]),
                "original_model_output": {
                    "immutable": True,
                    "detections": [
                        _safe_detection(item) for item in payload.get("original_detections") or []
                    ],
                },
                "human_review": {
                    "reviewer_id": payload["reviewer_id"],
                    "clean_frame": clean_frame,
                    "annotations": annotations,
                    "pending_annotations": pending_annotations,
                    "change_log": copy.deepcopy(payload.get("change_log") or []),
                    "note": payload.get("note"),
                },
                "training_unit": {
                    "unit_id": annotation_id,
                    "state": (
                        "clean_negative"
                        if clean_frame
                        else ("partial" if submission_scope == "partial_review" else "complete")
                    ),
                    "annotation_count": len(annotations),
                    "pending_annotation_count": len(pending_annotations),
                    "confirmed_annotations_only": True,
                    "unreviewed_regions_ignored": bool(
                        payload.get("ignore_unreviewed_regions")
                    ),
                    "unreviewed_regions_used_as_negative": False,
                },
                "workflow": workflow,
                "governance": {
                    "contract_id": contract["contract_id"],
                    "contract_version": contract["version"],
                    "provider_pixels_persisted": False,
                    "api_credentials_persisted": False,
                    "confirm_is_submission_boundary": True,
                    "apply_to_image_writes_queue": False,
                    "pending_annotations_used_as_negatives": False,
                    "unreviewed_regions_used_as_negatives": False,
                    "model_training_started": False,
                    "model_registry_changed": False,
                },
            }
            queue["records"].append(record)
            queue["updated_at"] = now
            _atomic_json(self.queue_path, queue)
        return {
            "record": copy.deepcopy(record),
            "summary": self._summary(queue["records"]),
            "idempotent_replay": False,
        }

    def update_status(
        self,
        annotation_id: str,
        *,
        status: str,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        contract = self.contract()
        allowed = set(contract["queue_statuses"])
        if status not in allowed or status == "draft":
            raise Stage8AnnotationConflictError("The requested Stage 8 queue status is invalid.")
        if status == "used_in_model_version" and not model_version:
            raise Stage8AnnotationConflictError("A model version is required for a used annotation.")
        now = _timestamp()
        with self._lock:
            queue = self._queue()
            record = next(
                (item for item in queue["records"] if item.get("annotation_id") == annotation_id),
                None,
            )
            if record is None:
                raise Stage8AnnotationNotFoundError("The requested Stage 8 annotation does not exist.")
            if status == "ready_for_training" and not record.get("training_ready"):
                raise Stage8AnnotationConflictError("Resolve uncertain labels before marking this review ready.")
            record["status"] = status
            record["used_in_model_version"] = model_version if status == "used_in_model_version" else None
            record.setdefault("workflow", []).append(
                {"status": status, "at": now, "model_version": model_version}
            )
            queue["updated_at"] = now
            _atomic_json(self.queue_path, queue)
            summary = self._summary(queue["records"])
        return {"record": copy.deepcopy(record), "summary": summary}


annotations = Stage8AnnotationQueue()
