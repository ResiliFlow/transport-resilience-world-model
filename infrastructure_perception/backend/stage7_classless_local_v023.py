"""Versioned Local adapter; no OpenAI credentials or Enhanced routing."""
from __future__ import annotations
import atexit
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from . import stage7_classless_v022 as base
from . import stage7_classless_geometry_v022 as geometry

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "infrastructure_perception/models/classless_local_v023"
WORKER = Path(__file__).with_name("stage7_classless_local_worker_v023.py")
REGISTRY = base.REGISTRY
RUNTIME = ROOT / ".runtime/stage7_classless_local_v023"
MANIFEST_SHA = "A61235E87D7F621308806D6C83DAE974794CBF9909A8B52E5C9E06C93031B4FA"
FUSION_MODE = "classless_local_v023"
Stage7ClasslessError = base.Stage7ClasslessError
sha = base.sha


def manifest():
    if not base.verified(PACKAGE / "manifest.json", MANIFEST_SHA):
        raise Stage7ClasslessError("The Local model manifest failed integrity verification.")
    return json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))


def active(mode="local"):
    if mode != "local":
        return False
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return data.get("models", {}).get(data.get("local_model_key"), {}).get("fusion_mode") == FUSION_MODE


def component_identity(mode="local"):
    if mode != "local":
        raise Stage7ClasslessError("This model package is Local only.")
    m = manifest()
    components = {**m["files"], "manifest": MANIFEST_SHA, "adapter": sha(__file__),
        "worker": sha(WORKER), "frozen_worker": sha(base.WORKER),
        "frozen_transport": sha(base.__file__), "geometry_policy": sha(geometry.__file__)}
    digest = hashlib.sha256(json.dumps({"components": components, "mode": "local"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return {"id": "twm-classless-local-v023", "name": "TWM compact road-problem detection",
        "version": m["package_version"], "checkpoint_sha256": digest, "components": components,
        "activation_state": "active" if active() else "isolated_candidate",
        "parent_version": m["parent_local_version"], "display_class_names": False,
        "requires_openai_key": False}


def checks(mode="local"):
    try:
        m = manifest()
        result = {"local_only": mode == "local", "manifest_verified": True,
            "python_available": base.PYTHON.is_file(), "isolated_worker_available": WORKER.is_file(),
            "frozen_worker_verified": base.verified(base.WORKER, m["frozen_base_worker_sha256"]),
            **{name + "_verified": base.verified(PACKAGE / name, digest)
               for name, digest in m["files"].items()}}
        if active(mode):
            registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
            result["active_registry_verified"] = (
                registry["models"][registry["local_model_key"]]["checkpoint_sha256"] ==
                component_identity()["checkpoint_sha256"])
        return result
    except (OSError, ValueError, KeyError, Stage7ClasslessError):
        return {"manifest_verified": False}


class ResidentWorker(base.ResidentWorker):
    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if not all(checks().values()):
            raise Stage7ClasslessError("The isolated Local model package is not ready.")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.log = (RUNTIME / "worker.log").open("ab")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for name in ("OPENAI_API_KEY", "STAGE7_VISION_API_KEY", "GOOGLE_MAPS_API_KEY", "GOOGLE_API_KEY"):
            env.pop(name, None)
        self.process = subprocess.Popen(
            [str(base.PYTHON), "-X", "utf8", "-B", "-u", str(WORKER), "--manifest-sha", MANIFEST_SHA],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            encoding="utf-8", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output, pipe = queue.Queue(), self.process.stdout
        self.output = output

        def read_lines():
            try:
                for line in pipe:
                    try:
                        output.put(json.loads(line))
                    except json.JSONDecodeError:
                        output.put({"ok": False, "error": "Invalid worker protocol response"})
            finally:
                output.put(None)

        threading.Thread(target=read_lines, daemon=True).start()
        try:
            ready = output.get(timeout=180)
            if (not ready or ready.get("ready") is not True or
                    ready.get("manifest_sha256") != MANIFEST_SHA or
                    ready.get("model_version") != manifest()["package_version"]):
                self.close()
                raise Stage7ClasslessError("The isolated Local worker failed to load its verified package.")
            self.ready = ready
            (RUNTIME / "worker_identity.json").write_text(json.dumps(ready), encoding="utf-8")
        except queue.Empty:
            self.close()
            raise Stage7ClasslessError("The isolated Local worker timed out while loading.") from None

    def infer(self, original, native, mode):
        if mode != "local":
            raise Stage7ClasslessError("Enhanced requests cannot use the Local-only worker.")
        result = super().infer(original, native, mode)
        m = manifest()
        if (result.get("manifest_sha256") != MANIFEST_SHA or
                result.get("checkpoint_sha256") != m["dense_source_sha256"] or
                result.get("roi_checkpoint_sha256") != m["roi_source_sha256"] or
                result.get("roi_parameter_sha256") != m["roi_parameter_sha256"] or
                result.get("cross_frame_cache") is not False or result.get("provider_calls") != 0):
            self.close()
            raise Stage7ClasslessError("The Local worker returned a different model identity.")
        return result


worker = ResidentWorker()
atexit.register(worker.close)


def infer(*, image_path, tiles, camera, frame_identity, mode="local", api_key=None):
    if mode != "local" or api_key:
        raise Stage7ClasslessError("Local analysis does not use an OpenAI key.")
    if not all(checks().values()):
        raise Stage7ClasslessError("The Local model package failed verification.")
    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    identity = component_identity()
    run_id = "classless-current-view-" + uuid.uuid4().hex
    original = native = None
    try:
        original, native, transforms = geometry.aligned_detail(image_path, tiles, camera)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="twm-frame-", dir=RUNTIME) as directory:
            native_path = Path(directory) / "native.png"
            native.save(native_path, format="PNG")
            local = worker.infer(Path(image_path), native_path, "local")
            detections = []
            for index, prediction in enumerate(local["predictions"]):
                if not geometry.bounds(prediction["box"]):
                    raise ValueError("Invalid final box")
                detections.append({**prediction, "detection_id": f"{run_id}:box:{index}",
                    "class_id": 8, "damage_type": "visible_road_anomaly", "display_label": None,
                    "classless": True, "confidence_is_calibrated": False, "tentative": True,
                    "review": {"status": "pending"},
                    "consensus": {"strategy": "trained130_roi132_classless_local",
                                  "host_estimate_requires_review": True}})
            run = {"run_id": run_id, "origin_stage": 7, "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(), "frame_identity": frame_identity,
                "frame_revision": camera.get("frame_revision"), "image_sha256": local["original_sha256"],
                "native_sha256": local["native_sha256"], "model_id": identity["id"],
                "model_name": identity["name"], "model_version": identity["version"],
                "checkpoint_sha256": identity["checkpoint_sha256"], "component_sha256": identity["components"],
                "result_type": "detections" if detections else "no_detection",
                "detection_ids": [d["detection_id"] for d in detections],
                "surface_metadata_used": False, "stage6_observation_persisted": False,
                "automatic_twm_transfer": False, "automatic_operational_action_enabled": False,
                "diagnostics": {"strategy": "trained130_roi132_classless_local", "inference_mode": "local",
                    "actual_model_execution": True, "worker_run_id": local["run_id"],
                    "worker_pid": local["worker_pid"], "worker_execution_count": local["execution_count"],
                    "dense_checkpoint_sha256": local["checkpoint_sha256"],
                    "dense_parameter_sha256": local["parameter_sha256"],
                    "roi_checkpoint_sha256": local["roi_checkpoint_sha256"],
                    "roi_parameter_sha256": local["roi_parameter_sha256"],
                    "forward_images": local["forward_images"], "roi_execution": local["roi_execution"],
                    "surface_support_floor": local["surface_gate_threshold"],
                    "model_load_seconds": local["model_load_seconds"], "local_seconds": local["seconds"],
                    "elapsed_seconds": time.perf_counter()-start, "openai_used": False,
                    "provider_calls": 0, "provider_results": [], "cross_frame_cache_used": False,
                    "assistant_credential_reused": False, "provider_credential_persisted": False,
                    "display_class_names": False, "native_size": list(native.size),
                    "same_camera_transforms": transforms, "gpu_peak_bytes": local["gpu_peak_bytes"]}}
            return {"run": run, "detections": detections}
    except Stage7ClasslessError:
        raise
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise Stage7ClasslessError("The exact-frame Local result failed validation: " + type(error).__name__) from None
    finally:
        if original is not None:
            original.close()
        if native is not None:
            native.close()
