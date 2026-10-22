from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .stage6_deployment import Stage6DeploymentError
from .stage7_surface_precision import (
    checks as surface_precision_checks,
    component_identity as surface_precision_component_identity,
)
from .stage7_surface_precision_v021_candidate06 import (
    checks as surface_precision_v021_candidate06_checks,
    component_identity as surface_precision_v021_candidate06_component_identity,
)


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = PERCEPTION_ROOT.parent
DEFAULT_STORAGE_ROOT = PERCEPTION_ROOT / "storage"
WORKER_PATH = PERCEPTION_ROOT / "backend" / "stage7_ensemble_worker.py"
R07_FUSION_PATH = PERCEPTION_ROOT / "backend" / "stage7_fusion_r07.py"
MODEL_REGISTRY_PATH = PERCEPTION_ROOT / "contracts" / "stage7_model_registry.json"
YOLO26S_CHECKPOINT = (
    PERCEPTION_ROOT
    / "datasets"
    / "stage5_v2"
    / "artifacts"
    / "runs"
    / "final_yolo26s_seed20260810"
    / "weights"
    / "best.pt"
)
RTDETR_L_CHECKPOINT = (
    PERCEPTION_ROOT
    / "datasets"
    / "stage5_v2"
    / "artifacts"
    / "runs"
    / "final_rtdetr-l_seed20260810"
    / "weights"
    / "best.pt"
)
YOLO26S_SHA256 = "1C5BBA8177839B3AF76CBD932A92411D8B957B623AAE6FA11CC60D04781EA9A6"
RTDETR_L_SHA256 = "D35C29A8D2282D3385CCFCAFD978A01AFB5CD6942402163FF65BDD0765C49A9D"
ACTIVE_SPECIALISTS = {
    "yolo26n": (
        PERCEPTION_ROOT / "datasets" / "stage5_v2" / "artifacts" / "runs" / "final_yolo26n_seed20260810" / "weights" / "best.pt",
        "F21CBA5926057CD0982460646F3A6BC3439CB1BBFF4425D49B2E12043DB85455",
        0.05,
    ),
    "yolo26l": (
        PERCEPTION_ROOT / "datasets" / "stage5_v2" / "artifacts" / "runs" / "final_yolo26l_seed20260810" / "weights" / "best.pt",
        "B765013E6C36EF09FFE23EC6AA9BD3E968D487995EC66AEA96A2B681341E67A5",
        0.05,
    ),
    "cycle01": (
        PERCEPTION_ROOT / "datasets" / "stage5_v2" / "artifacts" / "runs" / "twm_infradet_s1_cycle01_seed20260822" / "weights" / "best.pt",
        "FF6F0739793E30DE24A9C1F9D998CE895064D96357F2042C558B7AD0D310FA7B",
        0.05,
    ),
}
R04_SURFACE_CHECKPOINT = (
    PLATFORM_ROOT
    / ".runtime"
    / "stage7-surface-proposal-r04-cont"
    / "yolo26l_surface_proposal_r04_logical10_11"
    / "weights"
    / "last.pt"
)
R04_SURFACE_SHA256 = "7DB292C3616C458A216B7A79C879FEA13879EBE479566C8B77A3E7AA70ECF470"
RESEARCH_CHECKPOINT = PLATFORM_ROOT / ".runtime/twm-v151-p05-r04/weights/best.pt"
RESEARCH_SHA256 = "C399F66777DD3C7FFCBEB76A77FD9D28090E04F5917B22018B853643A59D2B06"
MODEL_ID = "twm-infradet-stage7-heterogeneous-consensus-v0.1.17"
MODEL_NAME = "TWM-InfraDet Stage 7 Heterogeneous Consensus"
MODEL_VERSION = "Stage7-HMT-Consensus-v0.1.17"
CLASS_NAMES = (
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_cracking",
    "uneven_or_broken_footpath",
    "broken_kerb",
    "standing_water",
    "debris_or_vegetation_obstruction",
)
R07_MODEL_ID = "twm-infradet-stage7-r07-high-recall-v0.1.19"
R07_MODEL_NAME = "TWM-InfraDet Stage 7 R07 High-Recall Fusion"
R07_MODEL_VERSION = "Stage7-R07-HighRecall-v0.1.19-candidate03"
R07_CLASS_NAMES = (*CLASS_NAMES, "visible_road_anomaly")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@lru_cache(maxsize=32)
def _sha256_matches_fingerprint(
    path_text: str,
    size: int,
    modified_ns: int,
    expected_sha256: str,
) -> bool:
    del size, modified_ns
    return _sha256(Path(path_text)) == expected_sha256


def _verified_immutable_file(path: Path, expected_sha256: str) -> bool:
    try:
        metadata = path.stat()
    except OSError:
        return False
    if not path.is_file():
        return False
    return _sha256_matches_fingerprint(
        str(path.resolve()),
        metadata.st_size,
        metadata.st_mtime_ns,
        expected_sha256,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage6DeploymentError("The Stage 7 ensemble worker returned no readable result.") from error
    if not isinstance(payload, dict):
        raise Stage6DeploymentError("The Stage 7 ensemble worker returned an invalid result.")
    return payload


def _registry_fusion_mode() -> str:
    payload = _load_json(MODEL_REGISTRY_PATH)
    models = payload.get("models")
    active_key = payload.get("active_model_key")
    if (
        payload.get("schema_version") != 1
        or not isinstance(models, dict)
        or active_key not in models
    ):
        raise Stage6DeploymentError("The Stage 7 model registry is invalid.")
    fusion_mode = models[active_key].get("fusion_mode")
    if fusion_mode not in {
        "v017",
        "r07",
        "surface_precision_v020",
        "surface_precision_v021_candidate06",
        "classless_v022",
    }:
        raise Stage6DeploymentError("The active Stage 7 registry model is invalid.")
    return str(fusion_mode)


def _registry_candidate_active() -> bool:
    return _registry_fusion_mode() == "r07"


class Stage7Ensemble:
    def __init__(self, *, storage_root: Path = DEFAULT_STORAGE_ROOT) -> None:
        self.storage_root = Path(storage_root)
        self.runtime_root = self.storage_root / "stage7_ensemble_runtime"
        self._lock = threading.Lock()

    @staticmethod
    def _python_executable() -> Path:
        return Path(os.environ.get("TWM_PERCEPTION_PYTHON") or sys.executable)

    def checks(self, *, candidate: bool | None = None) -> dict[str, bool]:
        active_mode = self.active_mode() if candidate is None else None
        if candidate is None and active_mode == "classless_v022":
            from .stage7_classless_v022 import checks
            return checks("enhanced")
        if candidate is None and active_mode in {
            "surface_precision_v020",
            "surface_precision_v021_candidate06",
        }:
            payload = _load_json(MODEL_REGISTRY_PATH)
            entry = payload["models"][payload["active_model_key"]]
            if active_mode == "surface_precision_v021_candidate06":
                identity = surface_precision_v021_candidate06_component_identity()
                result = surface_precision_v021_candidate06_checks()
            else:
                identity = surface_precision_component_identity()
                result = surface_precision_checks()
            result.update(
                {
                    "surface_precision_adapter_registry_verified": (
                        entry.get("adapter_sha256") == identity["components"]["adapter"]
                    ),
                    "surface_precision_checkpoint_registry_verified": (
                        entry.get("checkpoint_sha256") == identity["checkpoint_sha256"]
                    ),
                }
            )
            return result
        if candidate is None:
            candidate = _registry_candidate_active()
        python = self._python_executable()
        checks = {
            "python_available": python.is_file(),
            "worker_available": WORKER_PATH.is_file(),
            "registry_available": MODEL_REGISTRY_PATH.is_file(),
            "yolo26s_verified": _verified_immutable_file(YOLO26S_CHECKPOINT, YOLO26S_SHA256),
            "rtdetr_l_verified": _verified_immutable_file(RTDETR_L_CHECKPOINT, RTDETR_L_SHA256),
        }
        for specialist_name, (
            specialist_path,
            specialist_sha256,
            _specialist_confidence,
        ) in ACTIVE_SPECIALISTS.items():
            checks[f"{specialist_name}_verified"] = (
                _verified_immutable_file(specialist_path, specialist_sha256)
            )
        if candidate:
            checks.update(
                {
                    "r04_surface_verified": _verified_immutable_file(
                        R04_SURFACE_CHECKPOINT, R04_SURFACE_SHA256
                    ),
                    "research_v151_verified": _verified_immutable_file(
                        RESEARCH_CHECKPOINT, RESEARCH_SHA256
                    ),
                    "r07_fusion_available": R07_FUSION_PATH.is_file(),
                }
            )
        return checks

    @staticmethod
    def component_identity(*, candidate: bool | None = None) -> dict[str, Any]:
        if candidate is None:
            active_mode = _registry_fusion_mode()
            if active_mode == "classless_v022":
                from .stage7_classless_v022 import component_identity
                return component_identity("enhanced")
            if active_mode == "surface_precision_v020":
                return surface_precision_component_identity()
            if active_mode == "surface_precision_v021_candidate06":
                identity = dict(surface_precision_v021_candidate06_component_identity())
                identity["activation_state"] = "active_accepted_surface_precision"
                return identity
        registry_selected = candidate is None
        if candidate is None:
            candidate = _registry_candidate_active()
        if candidate:
            fusion_sha256 = _sha256(R07_FUSION_PATH)
            components = {
                "yolo26s": YOLO26S_SHA256,
                "rtdetr_l": RTDETR_L_SHA256,
                **{
                    name: sha256
                    for name, (_path, sha256, _confidence) in ACTIVE_SPECIALISTS.items()
                },
                "twm_infradet_v1_5_1": RESEARCH_SHA256,
                "yolo26l_surface_r04_e11": R04_SURFACE_SHA256,
                "r07_fusion": fusion_sha256,
            }
            composite = hashlib.sha256(
                (
                    ":".join(components.values())
                    + ":r07-category-agnostic-spatial-road-band-high-recall-candidate03"
                ).encode("ascii")
            ).hexdigest().upper()
            return {
                "id": R07_MODEL_ID,
                "name": R07_MODEL_NAME,
                "version": R07_MODEL_VERSION,
                "checkpoint_sha256": composite,
                "components": components,
                "activation_state": "active" if registry_selected else "candidate_shadow",
                "rollback_version": MODEL_VERSION,
            }
        composite = hashlib.sha256(
            (
                f"{YOLO26S_SHA256}:{RTDETR_L_SHA256}:"
                + ":".join(
                    specialist_sha256
                    for _path, specialist_sha256, _confidence in ACTIVE_SPECIALISTS.values()
                )
                + ":governed-multiscale-heterogeneous-fusion-v1.17"
            ).encode("ascii")
        ).hexdigest().upper()
        return {
            "id": MODEL_ID,
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "checkpoint_sha256": composite,
            "components": {
                "yolo26s": YOLO26S_SHA256,
                "rtdetr_l": RTDETR_L_SHA256,
                **{
                    name: sha256
                    for name, (_path, sha256, _confidence) in ACTIVE_SPECIALISTS.items()
                },
            },
            "activation_state": "active",
        }

    def infer(
        self,
        *,
        image_path: Path,
        frame_identity: str,
        candidate: bool | None = None,
    ) -> dict[str, Any]:
        if candidate is None and self.active_mode() in {
            "surface_precision_v020",
            "surface_precision_v021_candidate06",
        }:
            raise Stage6DeploymentError(
                "The active Stage 7 surface-precision adapter requires exact-frame detail views."
            )
        if candidate is None:
            candidate = _registry_candidate_active()
        if not image_path.is_file():
            raise Stage6DeploymentError("The exact Stage 7 frame is unavailable.")
        checks = self.checks(candidate=candidate)
        if not all(checks.values()):
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise Stage6DeploymentError(f"The Stage 7 ensemble is unavailable: {failed}.")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        descriptor, output_name = tempfile.mkstemp(
            prefix="stage7-ensemble-result-",
            suffix=".json",
            dir=self.runtime_root,
        )
        os.close(descriptor)
        output = Path(output_name)
        output.unlink(missing_ok=True)
        identity = self.component_identity(candidate=candidate)
        expected_class_names = R07_CLASS_NAMES if candidate else CLASS_NAMES
        command = [
            str(self._python_executable()),
            "-I",
            str(WORKER_PATH),
            "--image",
            str(image_path.resolve()),
            "--output",
            str(output),
            "--frame-identity",
            frame_identity,
            "--yolo-checkpoint",
            str(YOLO26S_CHECKPOINT),
            "--yolo-sha256",
            YOLO26S_SHA256,
            "--rtdetr-checkpoint",
            str(RTDETR_L_CHECKPOINT),
            "--rtdetr-sha256",
            RTDETR_L_SHA256,
            "--imgsz",
            "640",
            "--anchor-confidence",
            "0.12",
            "--teacher-confidence",
            "0.18",
            "--agreement-iou",
            "0.20",
            "--max-detections",
            "5" if candidate else "12",
            "--fusion-mode",
            "r07" if candidate else "v017",
            "--device",
            "auto",
        ]
        if candidate:
            command.extend(
                [
                    "--research-checkpoint",
                    str(RESEARCH_CHECKPOINT),
                    "--research-sha256",
                    RESEARCH_SHA256,
                    "--research-confidence",
                    "0.55",
                    "--surface-checkpoint",
                    str(R04_SURFACE_CHECKPOINT),
                    "--surface-sha256",
                    R04_SURFACE_SHA256,
                    "--r07-max-anchors",
                    "720",
                ]
            )
        for specialist_name, (
            specialist_path,
            specialist_sha256,
            specialist_confidence,
        ) in ACTIVE_SPECIALISTS.items():
            command.extend(
                [
                    "--specialist",
                    f"{specialist_name}|{specialist_path}|{specialist_sha256}|{specialist_confidence}",
                ]
            )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONUTF8"] = "1"
        environment["YOLO_CONFIG_DIR"] = str((self.runtime_root / "ultralytics").resolve())
        try:
            with self._lock:
                completed = subprocess.run(
                    command,
                    cwd=PLATFORM_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=360 if candidate else 180,
                    check=False,
                )
            payload = _load_json(output) if output.is_file() else {}
            if completed.returncode != 0 or payload.get("status") != "inference_complete":
                detail = str(payload.get("error") or completed.stderr or "unknown failure")
                if payload.get("runtime"):
                    detail = f"{detail}; runtime={json.dumps(payload['runtime'], ensure_ascii=True)}"
                raise Stage6DeploymentError(f"Stage 7 ensemble inference failed: {detail[:500]}")
            expected_names = [
                payload.get("class_names", {}).get(str(index))
                for index in range(len(expected_class_names))
            ]
            if expected_names != list(expected_class_names):
                raise Stage6DeploymentError("The Stage 7 ensemble class map is invalid.")
            if (
                payload.get("model_id") != identity["id"]
                or payload.get("model_version") != identity["version"]
            ):
                raise Stage6DeploymentError("The Stage 7 ensemble model identity is invalid.")
            detections: list[dict[str, Any]] = []
            for index, prediction in enumerate(payload.get("predictions") or []):
                class_id = prediction.get("class_id")
                score = prediction.get("confidence")
                box = prediction.get("box")
                if (
                    not isinstance(class_id, int)
                    or not 0 <= class_id < len(expected_class_names)
                    or not isinstance(score, (int, float))
                    or not 0.0 <= float(score) <= 1.0
                    or not isinstance(box, list)
                    or len(box) != 4
                    or not all(isinstance(value, (int, float)) for value in box)
                    or not 0.0 <= float(box[0]) < float(box[2]) <= 1.0
                    or not 0.0 <= float(box[1]) < float(box[3]) <= 1.0
                ):
                    raise Stage6DeploymentError("The Stage 7 ensemble returned an invalid prediction.")
                normalised_box = [float(value) for value in box]
                token = json.dumps(
                    [frame_identity, index, class_id, normalised_box, identity["version"]],
                    separators=(",", ":"),
                ).encode("utf-8")
                detections.append(
                    {
                        "detection_id": f"stage7-detection-{hashlib.sha256(token).hexdigest()[:20]}",
                        "class_id": class_id,
                        "damage_type": expected_class_names[class_id],
                        "confidence": float(score),
                        "box": normalised_box,
                        "review": {"status": "pending"},
                        "consensus": {
                            "agreement_iou": prediction.get("agreement_iou"),
                            "support": prediction.get("support"),
                        },
                    }
                )
            run = {
                "run_id": f"stage7-ensemble-run-{uuid.uuid4().hex}",
                "origin_stage": 7,
                "started_at": _timestamp(),
                "completed_at": _timestamp(),
                "frame_identity": frame_identity,
                "image_sha256": payload["image_sha256"],
                "model_id": identity["id"],
                "model_name": identity["name"],
                "model_version": identity["version"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "component_sha256": identity["components"],
                "result_type": "detections" if detections else "no_detection",
                "detection_ids": [item["detection_id"] for item in detections],
                "worker": {
                    "device": payload["device"],
                    "prediction_count": payload["prediction_count"],
                    "timing_ms": payload["timing_ms"],
                },
                "diagnostics": payload.get("diagnostics") or {},
                "surface_metadata_used": False,
                "stage6_observation_persisted": False,
                "automatic_operational_action_enabled": False,
            }
            return {"run": run, "detections": detections}
        except subprocess.TimeoutExpired as error:
            timeout_seconds = 360 if candidate else 180
            raise Stage6DeploymentError(
                f"Stage 7 ensemble inference exceeded {timeout_seconds} seconds."
            ) from error
        finally:
            output.unlink(missing_ok=True)

    @staticmethod
    def active_mode() -> str:
        return _registry_fusion_mode()


stage7_ensemble = Stage7Ensemble()
