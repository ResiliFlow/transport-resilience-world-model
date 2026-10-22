from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .twm_stage6 import adapt_predictions


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = PERCEPTION_ROOT.parent
REGISTRY_PATH = PERCEPTION_ROOT / "stage6" / "model_registry.json"
DEFAULT_STORAGE_ROOT = PERCEPTION_ROOT / "storage"


class Stage6DeploymentError(RuntimeError):
    pass


class Stage6NotFoundError(Stage6DeploymentError):
    pass


class Stage6ConflictError(Stage6DeploymentError):
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
        raise Stage6DeploymentError(f"Required Stage 6 file is missing: {path.name}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise Stage6DeploymentError(f"Required Stage 6 file is unreadable: {path.name}") from error
    if not isinstance(payload, dict):
        raise Stage6DeploymentError(f"Required Stage 6 file is not a JSON object: {path.name}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _empty_activity() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": None,
        "inference_runs": [],
        "observations": [],
    }


class Stage6Deployment:
    def __init__(
        self,
        *,
        registry_path: Path = REGISTRY_PATH,
        storage_root: Path = DEFAULT_STORAGE_ROOT,
        platform_root: Path = PLATFORM_ROOT,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.storage_root = Path(storage_root)
        self.platform_root = Path(platform_root)
        self.active_state_path = self.storage_root / "stage6_active_model.json"
        self.activity_path = self.storage_root / "stage6_activity.json"
        self.runtime_root = self.storage_root / "stage6_runtime"
        self._state_lock = threading.RLock()
        self._activity_lock = threading.RLock()
        self._inference_lock = threading.Lock()

    def _registry(self) -> dict[str, Any]:
        registry = _load_json(self.registry_path)
        packages = registry.get("packages")
        if registry.get("schema_version") != 1 or not isinstance(packages, list) or not packages:
            raise Stage6DeploymentError("The Stage 6 model registry is invalid.")
        identifiers = [item.get("id") for item in packages if isinstance(item, dict)]
        if len(identifiers) != len(packages) or len(set(identifiers)) != len(identifiers):
            raise Stage6DeploymentError("Stage 6 model package identifiers must be unique.")
        if registry.get("default_model_id") not in identifiers:
            raise Stage6DeploymentError("The Stage 6 default model is not registered.")
        return registry

    def _package(self, model_id: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
        registry = registry or self._registry()
        package = next(
            (item for item in registry["packages"] if item.get("id") == model_id),
            None,
        )
        if package is None:
            raise Stage6NotFoundError("The requested Stage 6 model package is not registered.")
        return package

    def _project_path(self, relative_path: str) -> Path:
        path = (self.platform_root / relative_path).resolve()
        try:
            path.relative_to(self.platform_root.resolve())
        except ValueError as error:
            raise Stage6DeploymentError("A Stage 6 model package path escapes the platform root.") from error
        return path

    def _active_state(self, registry: dict[str, Any] | None = None) -> dict[str, Any]:
        registry = registry or self._registry()
        if not self.active_state_path.exists():
            return {
                "schema_version": 1,
                "active_model_id": registry["default_model_id"],
                "previous_model_id": None,
                "activated_at": None,
                "activated_by": "registry_default",
                "reason": "immutable_registry_default",
            }
        state = _load_json(self.active_state_path)
        if state.get("schema_version") != 1:
            raise Stage6DeploymentError("The active Stage 6 model pointer is invalid.")
        self._package(str(state.get("active_model_id") or ""), registry)
        previous = state.get("previous_model_id")
        if previous is not None:
            self._package(str(previous), registry)
        return state

    def _python_executable(self, package: dict[str, Any]) -> Path:
        runtime = package["runtime"]
        variable = str(runtime.get("python_environment_variable") or "")
        configured = os.environ.get(variable) if variable else None
        return Path(configured or runtime["default_python"]).resolve()

    def _package_checks(self, package: dict[str, Any], registry: dict[str, Any]) -> dict[str, bool]:
        runtime = package.get("runtime") if isinstance(package.get("runtime"), dict) else {}
        checkpoint = self._project_path(str(package.get("checkpoint") or ""))
        model_card = self._project_path(str(package.get("model_card") or ""))
        integration = self._project_path(str(package.get("integration_record") or ""))
        worker = self._project_path(str(runtime.get("worker") or ""))
        contract = self._project_path(str(registry["contract"]["path"]))

        def verified(path: Path, expected: str) -> bool:
            return path.is_file() and bool(expected) and _sha256(path) == expected.upper()

        return {
            "checkpoint_verified": verified(checkpoint, str(package.get("checkpoint_sha256") or "")),
            "model_card_verified": verified(model_card, str(package.get("model_card_sha256") or "")),
            "integration_record_verified": verified(
                integration,
                str(package.get("integration_record_sha256") or ""),
            ),
            "worker_verified": verified(worker, str(runtime.get("worker_sha256") or "")),
            "observation_contract_verified": verified(
                contract,
                str(registry["contract"].get("sha256") or ""),
            ),
            "isolated_python_available": self._python_executable(package).is_file(),
            "teacher_free_runtime": runtime.get("type") == "isolated_ultralytics_subprocess",
            "automatic_operational_action_disabled": registry["policy"].get(
                "automatic_operational_action_enabled"
            )
            is False,
            "human_review_required": registry["policy"].get("human_review_required") is True,
        }

    def status(self) -> dict[str, Any]:
        registry = self._registry()
        state = self._active_state(registry)
        packages: list[dict[str, Any]] = []
        for package in registry["packages"]:
            checks = self._package_checks(package, registry)
            packages.append(
                {
                    "id": package["id"],
                    "display_name": package["display_name"],
                    "model_name": package["model_name"],
                    "model_version": package["model_version"],
                    "channel": package["channel"],
                    "active": package["id"] == state["active_model_id"],
                    "activation_eligible": package["activation_eligible"],
                    "scientific_promotion_ready": package["scientific_promotion_ready"],
                    "operational_ready": package["operational_ready"],
                    "checkpoint_sha256": package["checkpoint_sha256"],
                    "checks": checks,
                    "package_ready": all(checks.values()),
                    "boundary": package["boundary"],
                }
            )
        active = next(item for item in packages if item["active"])
        return {
            "stage": 6,
            "status": "research_deployment_ready" if active["package_ready"] else "blocked",
            "registry_id": registry["registry_id"],
            "registry_sha256": _sha256(self.registry_path),
            "active_model": active,
            "active_pointer": state,
            "packages": packages,
            "contract": registry["contract"],
            "policy": registry["policy"],
            "capabilities": {
                "authorised_upload_inference": active["package_ready"],
                "safe_abstention": True,
                "observation_persistence": True,
                "human_review": True,
                "atomic_model_activation": True,
                "rollback": True,
                "automatic_operational_action": False,
            },
            "scope": "human_reviewed_research_deployment",
        }

    def _run_worker(
        self,
        package: dict[str, Any],
        *,
        image_path: Path | None = None,
        self_test: bool = False,
    ) -> dict[str, Any]:
        registry = self._registry()
        checks = self._package_checks(package, registry)
        if not all(checks.values()):
            failed = ", ".join(key for key, passed in checks.items() if not passed)
            raise Stage6DeploymentError(f"The active Stage 6 model package failed: {failed}.")
        checkpoint = self._project_path(package["checkpoint"])
        worker = self._project_path(package["runtime"]["worker"])
        python = self._python_executable(package)
        config = package["inference"]
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        descriptor, output_name = tempfile.mkstemp(
            prefix="stage6-result-",
            suffix=".json",
            dir=self.runtime_root,
        )
        os.close(descriptor)
        output = Path(output_name)
        output.unlink(missing_ok=True)
        command = [
            str(python),
            str(worker),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            package["checkpoint_sha256"],
            "--output",
            str(output),
            "--imgsz",
            str(config["imgsz"]),
            "--confidence",
            str(config["confidence"]),
            "--iou",
            str(config["iou"]),
            "--max-detections",
            str(config["max_detections"]),
            "--device",
            str(config["device"]),
        ]
        if self_test:
            command.append("--self-test")
        elif image_path is not None:
            command.extend(["--image", str(image_path.resolve())])
        else:
            raise Stage6DeploymentError("The Stage 6 worker requires an image or a self-test.")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["YOLO_CONFIG_DIR"] = str((self.runtime_root / "ultralytics").resolve())
        try:
            with self._inference_lock:
                completed = subprocess.run(
                    command,
                    cwd=self.platform_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=int(config["timeout_seconds"]),
                    check=False,
                )
            payload = _load_json(output) if output.is_file() else {}
            if completed.returncode != 0 or payload.get("status") == "failed":
                detail = str(payload.get("error") or completed.stderr or "unknown worker failure")
                raise Stage6DeploymentError(f"Stage 6 inference failed: {detail[:500]}")
            expected_status = "self_test_passed" if self_test else "inference_complete"
            if payload.get("status") != expected_status:
                raise Stage6DeploymentError("The Stage 6 worker returned an invalid status.")
            if payload.get("checkpoint_sha256") != package["checkpoint_sha256"]:
                raise Stage6DeploymentError("The Stage 6 worker used an unexpected checkpoint.")
            return payload
        except subprocess.TimeoutExpired as error:
            raise Stage6DeploymentError("Stage 6 inference exceeded its bounded runtime.") from error
        finally:
            output.unlink(missing_ok=True)

    def self_test(self, model_id: str | None = None) -> dict[str, Any]:
        registry = self._registry()
        if model_id is None:
            model_id = self._active_state(registry)["active_model_id"]
        package = self._package(model_id, registry)
        result = self._run_worker(package, self_test=True)
        return {
            "model_id": model_id,
            "model_version": package["model_version"],
            "status": result["status"],
            "device": result["device"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "timing_ms": result["timing_ms"],
            "synthetic_input_only": True,
        }

    @staticmethod
    def _location(record: dict[str, Any]) -> dict[str, Any] | None:
        latitude = record.get("latitude")
        longitude = record.get("longitude")
        if latitude is None or longitude is None:
            return None
        method = {
            "image metadata": "source_metadata",
            "source metadata": "source_metadata",
            "network match": "network_match",
            "survey control": "survey_control",
            "manual map placement": "manual_review",
            "manual review": "manual_review",
        }.get(str(record.get("location_method") or "").casefold(), "manual_review")
        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "method": method,
        }

    @staticmethod
    def _source_evidence(record: dict[str, Any]) -> dict[str, Any]:
        explicit = record.get("source_evidence")
        if isinstance(explicit, dict):
            return dict(explicit)
        authorised = record.get("analysis_authorised") is True
        uploaded_at = str(record.get("uploaded_at") or "unknown_not_provided")
        captured_at = str(record.get("captured_at") or uploaded_at)
        asset_id = str(record.get("id") or "unknown-upload")
        return {
            "source_id": "individual_user_upload",
            "asset_id": asset_id,
            "captured_at": captured_at,
            "licence_status": "verified" if authorised else "unverified",
            "permission_record_id": (
                f"user-upload:{asset_id}:rights-confirmed"
                if authorised
                else f"user-upload:{asset_id}:rights-not-confirmed"
            ),
            "attribution": str(record.get("source") or "Individual user upload")[:500],
        }

    def _activity(self) -> dict[str, Any]:
        if not self.activity_path.exists():
            return _empty_activity()
        payload = _load_json(self.activity_path)
        if (
            payload.get("schema_version") != 1
            or not isinstance(payload.get("inference_runs"), list)
            or not isinstance(payload.get("observations"), list)
        ):
            raise Stage6DeploymentError("The Stage 6 activity store is invalid.")
        return payload

    def _record_result(
        self,
        *,
        run: dict[str, Any],
        result: dict[str, Any],
        image_record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        now = _timestamp()
        with self._activity_lock:
            activity = self._activity()
            activity["inference_runs"].append(run)
            envelopes: list[dict[str, Any]] = []
            by_id = {
                item.get("observation", {}).get("observation_id"): item
                for item in activity["observations"]
                if isinstance(item, dict) and isinstance(item.get("observation"), dict)
            }
            for observation in result.get("observations", []):
                observation_id = observation["observation_id"]
                existing = by_id.get(observation_id)
                if existing is None:
                    envelope = {
                        "image_id": image_record["id"],
                        "image_name": image_record.get("original_name") or image_record["id"],
                        "image_url": image_record.get("image_url") or f"uploads/{image_record['stored_name']}",
                        "origin_stage": int(run.get("origin_stage") or 6),
                        "provider": image_record.get("provider") or image_record.get("source_id") or "individual_user_upload",
                        "created_at": now,
                        "last_seen_at": now,
                        "review_note": None,
                        "review_history": [],
                        "observation": observation,
                    }
                    activity["observations"].append(envelope)
                    by_id[observation_id] = envelope
                else:
                    envelope = existing
                    previous_review = existing["observation"].get("review", {"status": "pending"})
                    observation["review"] = previous_review
                    envelope["observation"] = observation
                    envelope["last_seen_at"] = now
                envelopes.append(envelope)
            activity["updated_at"] = now
            _atomic_json(self.activity_path, activity)
            return envelopes

    def infer_uploaded_image(
        self,
        *,
        image_record: dict[str, Any],
        image_path: Path,
        surface_type: str | None = None,
    ) -> dict[str, Any]:
        return self.infer_evidence_image(
            image_record=image_record,
            image_path=image_path,
            surface_type=surface_type,
            origin_stage=6,
        )

    def infer_visual_frame(
        self,
        *,
        image_path: Path,
        frame_identity: str,
        origin_stage: int = 7,
    ) -> dict[str, Any]:
        """Run the active detector on a transient visual frame without evidence adaptation.

        Stage 7 uses this deliberately narrow entry point for the exact image currently
        visible to the user. It validates the immutable Stage 6 package and raw detector
        contract, but it neither asks for surface metadata nor writes Stage 6 observations.
        """

        if not image_path.is_file():
            raise Stage6NotFoundError("The selected visual frame is unavailable.")
        frame_identity = frame_identity.strip()
        if not frame_identity:
            raise Stage6DeploymentError("A visual-frame identity is required.")
        registry = self._registry()
        state = self._active_state(registry)
        package = self._package(state["active_model_id"], registry)
        checks = self._package_checks(package, registry)
        if not all(checks.values()):
            raise Stage6DeploymentError("The active Stage 6 model package is not ready.")

        started_at = _timestamp()
        worker_result = self._run_worker(package, image_path=image_path)
        expected_names = package["class_names"]
        observed_names = [
            worker_result["class_names"].get(str(index))
            for index in range(len(expected_names))
        ]
        if observed_names != expected_names:
            raise Stage6DeploymentError("The worker class map does not match the Stage 6 package.")

        detections: list[dict[str, Any]] = []
        for index, prediction in enumerate(worker_result.get("predictions") or []):
            if not isinstance(prediction, dict):
                raise Stage6DeploymentError("The visual-frame worker returned an invalid prediction.")
            class_id = prediction.get("class_id")
            confidence = prediction.get("confidence")
            box = prediction.get("box")
            valid_box = (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
                and 0.0 <= float(box[0]) < float(box[2]) <= 1.0
                and 0.0 <= float(box[1]) < float(box[3]) <= 1.0
            )
            if (
                not isinstance(class_id, int)
                or not 0 <= class_id < len(expected_names)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
                or not valid_box
            ):
                raise Stage6DeploymentError("The visual-frame worker returned an invalid prediction.")
            normalised_box = [float(value) for value in box]
            token = json.dumps(
                [frame_identity, index, class_id, normalised_box, package["model_version"]],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            detections.append(
                {
                    "detection_id": f"stage7-detection-{hashlib.sha256(token).hexdigest()[:20]}",
                    "class_id": class_id,
                    "damage_type": expected_names[class_id],
                    "confidence": float(confidence),
                    "box": normalised_box,
                    "review": {"status": "pending"},
                }
            )

        run = {
            "run_id": f"stage{origin_stage}-visual-run-{uuid.uuid4().hex}",
            "origin_stage": origin_stage,
            "started_at": started_at,
            "completed_at": _timestamp(),
            "frame_identity": frame_identity,
            "image_sha256": _sha256(image_path),
            "model_id": package["id"],
            "model_name": package["model_name"],
            "model_version": package["model_version"],
            "checkpoint_sha256": package["checkpoint_sha256"],
            "result_type": "detections" if detections else "no_detection",
            "detection_ids": [item["detection_id"] for item in detections],
            "worker": {
                "device": worker_result["device"],
                "prediction_count": worker_result["prediction_count"],
                "timing_ms": worker_result["timing_ms"],
            },
            "surface_metadata_used": False,
            "stage6_observation_persisted": False,
            "automatic_operational_action_enabled": False,
        }
        return {"run": run, "detections": detections}

    def infer_evidence_image(
        self,
        *,
        image_record: dict[str, Any],
        image_path: Path,
        surface_type: str | None = None,
        origin_stage: int = 6,
    ) -> dict[str, Any]:
        if surface_type not in {None, "road", "footpath", "kerb"}:
            raise Stage6DeploymentError("The selected surface type is invalid.")
        if not image_path.is_file():
            raise Stage6NotFoundError("The selected evidence image is unavailable.")
        registry = self._registry()
        state = self._active_state(registry)
        package = self._package(state["active_model_id"], registry)
        checks = self._package_checks(package, registry)
        if not all(checks.values()):
            raise Stage6DeploymentError("The active Stage 6 model package is not ready.")
        integration = _load_json(self._project_path(package["integration_record"]))
        location = self._location(image_record)
        source_evidence = self._source_evidence(image_record)
        started_at = _timestamp()
        run_id = f"stage{origin_stage}-run-{uuid.uuid4().hex}"
        worker_result: dict[str, Any] | None = None
        image_identity = str(
            image_record.get("stored_name")
            or image_record.get("source_asset_id")
            or image_record.get("id")
            or image_path.name
        )

        if location is None or source_evidence.get("licence_status") != "verified":
            result = adapt_predictions(
                image=image_identity,
                predictions=[],
                source_evidence=source_evidence,
                location=location,
                surface_type=surface_type,
                record=integration,
            )
        else:
            worker_result = self._run_worker(package, image_path=image_path)
            expected_names = package["class_names"]
            observed_names = [
                worker_result["class_names"].get(str(index))
                for index in range(len(expected_names))
            ]
            if observed_names != expected_names:
                raise Stage6DeploymentError("The worker class map does not match the Stage 6 package.")
            result = adapt_predictions(
                image=image_identity,
                predictions=worker_result["predictions"],
                source_evidence=source_evidence,
                location=location,
                surface_type=surface_type,
                record=integration,
            )

        completed_at = _timestamp()
        run = {
            "run_id": run_id,
            "origin_stage": origin_stage,
            "started_at": started_at,
            "completed_at": completed_at,
            "image_id": image_record["id"],
            "image_name": image_record.get("original_name") or image_identity,
            "provider": image_record.get("provider") or image_record.get("source_id") or "individual_user_upload",
            "image_sha256": _sha256(image_path),
            "model_id": package["id"],
            "model_name": package["model_name"],
            "model_version": package["model_version"],
            "checkpoint_sha256": package["checkpoint_sha256"],
            "result_type": result["result_type"],
            "observation_ids": [
                item["observation_id"] for item in result.get("observations", [])
            ],
            "abstention_reasons": result.get("reasons", []),
            "worker": (
                {
                    "device": worker_result["device"],
                    "prediction_count": worker_result["prediction_count"],
                    "timing_ms": worker_result["timing_ms"],
                }
                if worker_result
                else None
            ),
            "automatic_operational_action_enabled": False,
        }
        persisted = self._record_result(
            run=run,
            result=result,
            image_record=image_record,
        )
        return {
            "run": run,
            "result": result,
            "persisted_observations": persisted,
        }

    def activity(self, *, image_id: str | None = None) -> dict[str, Any]:
        with self._activity_lock:
            activity = self._activity()
        observations = activity["observations"]
        runs = activity["inference_runs"]
        if image_id is not None:
            observations = [item for item in observations if item.get("image_id") == image_id]
            runs = [item for item in runs if item.get("image_id") == image_id]
        return {
            "updated_at": activity["updated_at"],
            "observation_count": len(observations),
            "run_count": len(runs),
            "observations": list(reversed(observations)),
            "inference_runs": list(reversed(runs)),
        }

    def review_observation(
        self,
        observation_id: str,
        *,
        status: str,
        reviewer_id: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"confirmed", "rejected", "uncertain"}:
            raise Stage6DeploymentError("Choose confirmed, rejected or uncertain.")
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise Stage6DeploymentError("A reviewer identifier is required.")
        now = _timestamp()
        with self._activity_lock:
            activity = self._activity()
            envelope = next(
                (
                    item
                    for item in activity["observations"]
                    if item.get("observation", {}).get("observation_id") == observation_id
                ),
                None,
            )
            if envelope is None:
                raise Stage6NotFoundError("The requested Stage 6 observation was not found.")
            previous = dict(envelope["observation"].get("review") or {"status": "pending"})
            review = {
                "status": status,
                "reviewer_id": reviewer_id[:160],
                "reviewed_at": now,
            }
            envelope["observation"]["review"] = review
            envelope["review_note"] = note.strip()[:1000] if note and note.strip() else None
            envelope.setdefault("review_history", []).append(
                {
                    "changed_at": now,
                    "previous": previous,
                    "current": review,
                    "note": envelope["review_note"],
                }
            )
            activity["updated_at"] = now
            _atomic_json(self.activity_path, activity)
            return envelope

    def activate(
        self,
        model_id: str,
        *,
        expected_checkpoint_sha256: str,
        actor: str,
        reason: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        if not confirmed:
            raise Stage6ConflictError("Explicit model-activation confirmation is required.")
        registry = self._registry()
        package = self._package(model_id, registry)
        if package.get("activation_eligible") is not True:
            raise Stage6ConflictError("This model package is not activation eligible.")
        if expected_checkpoint_sha256.upper() != package["checkpoint_sha256"]:
            raise Stage6ConflictError("The model-activation checkpoint confirmation does not match.")
        checks = self._package_checks(package, registry)
        if not all(checks.values()):
            raise Stage6ConflictError("The requested model package failed integrity checks.")
        self_test = self.self_test(model_id)
        now = _timestamp()
        with self._state_lock:
            current = self._active_state(registry)
            previous_model_id = (
                current.get("previous_model_id")
                if current["active_model_id"] == model_id
                else current["active_model_id"]
            )
            state = {
                "schema_version": 1,
                "active_model_id": model_id,
                "previous_model_id": previous_model_id,
                "activated_at": now,
                "activated_by": actor.strip()[:160],
                "reason": reason.strip()[:500],
                "checkpoint_sha256": package["checkpoint_sha256"],
                "pre_activation_self_test": self_test,
            }
            _atomic_json(self.active_state_path, state)
            return state

    def rollback(self, *, actor: str, reason: str, confirmed: bool) -> dict[str, Any]:
        if not confirmed:
            raise Stage6ConflictError("Explicit rollback confirmation is required.")
        registry = self._registry()
        with self._state_lock:
            current = self._active_state(registry)
            target_id = current.get("previous_model_id")
            if not target_id:
                raise Stage6ConflictError("No previous Stage 6 model is available for rollback.")
            package = self._package(target_id, registry)
            self_test = self.self_test(target_id)
            state = {
                "schema_version": 1,
                "active_model_id": target_id,
                "previous_model_id": current["active_model_id"],
                "activated_at": _timestamp(),
                "activated_by": actor.strip()[:160],
                "reason": reason.strip()[:500],
                "checkpoint_sha256": package["checkpoint_sha256"],
                "pre_activation_self_test": self_test,
                "rollback": True,
            }
            _atomic_json(self.active_state_path, state)
            return state


deployment = Stage6Deployment()
