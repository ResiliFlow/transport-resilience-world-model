"""Versioned classless Local and optional Enhanced adapter; credentials stay request-local."""
from __future__ import annotations
import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from PIL import Image
from .stage6_deployment import Stage6DeploymentError
from . import stage7_classless_geometry_v022 as geometry

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "infrastructure_perception/models/classless_v022"
WORKER = Path(__file__).with_name("stage7_classless_worker_v022.py")
PYTHON = Path(sys.executable)
REGISTRY = ROOT / "infrastructure_perception/contracts/stage7_model_registry.json"
MANIFEST_SHA = "C48B14EEA86B99931D6A6DA6F13B4701DB948F5D36D513144D725C0C9F42EF37"
FUSION_MODE = "classless_v022"
RUNTIME = ROOT / ".runtime/stage7_classless_v022"


class Stage7ClasslessError(Stage6DeploymentError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest().upper()


@lru_cache(maxsize=32)
def verified_fingerprint(path, size, mtime_ns, expected):
    return sha(Path(path)) == expected


def verified(path, expected):
    try:
        st = path.stat()
        return path.is_file() and verified_fingerprint(str(path), st.st_size, st.st_mtime_ns, expected)
    except OSError:
        return False


def manifest():
    if not verified(PACKAGE/"manifest.json", MANIFEST_SHA):
        raise Stage7ClasslessError("The classless model package manifest failed integrity verification.")
    return json.loads((PACKAGE/"manifest.json").read_text(encoding="utf-8"))


def active(mode):
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    key = data.get("local_model_key" if mode == "local" else "active_model_key")
    return data.get("models", {}).get(key, {}).get("fusion_mode") == FUSION_MODE


def component_identity(mode="enhanced"):
    m = manifest()
    components = {**m["files"], "adapter": sha(__file__), "worker": sha(WORKER),
                  "geometry_policy": sha(geometry.__file__), "manifest": MANIFEST_SHA}
    version = "TWM-D124-R129-Local-v0.2.2" if mode == "local" else "TWM-D124-NativeCompletion-Enhanced-v0.2.2"
    digest = hashlib.sha256(json.dumps({"components": components, "mode": mode},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    return {"id": "twm-classless-"+mode+"-v022", "name": "TWM compact road-problem detection",
            "version": version, "checkpoint_sha256": digest, "components": components,
            "activation_state": "active" if active(mode) else "isolated_candidate",
            "parent_version": "R07" if mode == "local" else "C06",
            "display_class_names": False, "requires_openai_key": mode == "enhanced"}


def checks(mode="enhanced"):
    try:
        m = manifest()
        result = {"manifest_verified": True, "python_available": PYTHON.is_file(),
                "isolated_worker_available": WORKER.is_file(),
                **{name+"_verified": verified(PACKAGE/name, digest) for name, digest in m["files"].items()}}
        if active(mode):
            registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
            key = registry["local_model_key" if mode == "local" else "active_model_key"]
            result["active_registry_verified"] = registry["models"][key].get("checkpoint_sha256") == component_identity(mode)["checkpoint_sha256"]
        return result
    except (OSError, ValueError, Stage7ClasslessError):
        return {"manifest_verified": False}


class ResidentWorker:
    def __init__(self):
        self._lock = threading.RLock()
        self.process = None
        self.output = None
        self.log = None
        self.ready = None
        self.warm_thread = None
        self.warmup_error = None

    def warmup(self):
        """One-shot model loading, not polling; never performs inference or API calls."""
        if self.ready and self.process is not None and self.process.poll() is None:
            return {"state": "ready", "pid": self.ready["pid"], "load_seconds": self.ready["load_seconds"]}
        if self.warm_thread is None:
            def load_once():
                try:
                    with self._lock:
                        self.start()
                except Exception as error:
                    self.warmup_error = type(error).__name__
            self.warm_thread = threading.Thread(target=load_once, name="twm-classless-warmup", daemon=True)
            self.warm_thread.start()
        return {"state": "loading" if self.warm_thread.is_alive() else "not_loaded",
                "error_type": self.warmup_error, "provider_calls": 0}

    def close(self):
        with self._lock:
            process = self.process
            self.process = None
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(json.dumps({"command": "shutdown"})+"\n")
                        process.stdin.flush()
                        process.wait(timeout=4)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                for handle in (process.stdin, process.stdout):
                    if handle:
                        handle.close()
            if self.log:
                self.log.close()
                self.log = None
            self.output = None
            self.ready = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if not all(checks("local").values()):
            raise Stage7ClasslessError("The isolated classless model package is not ready.")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.log = (RUNTIME/"worker.log").open("ab")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for name in ("OPENAI_API_KEY", "STAGE7_VISION_API_KEY", "GOOGLE_MAPS_API_KEY", "GOOGLE_API_KEY"):
            env.pop(name, None)
        self.process = subprocess.Popen([str(PYTHON), "-X", "utf8", "-B", "-u", str(WORKER)],
                                        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, encoding="utf-8", bufsize=1, env=env,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output, pipe = queue.Queue(), self.process.stdout
        self.output = output
        def read_lines():
            try:
                for line in pipe:
                    try:
                        output.put(json.loads(line))
                    except json.JSONDecodeError:
                        output.put({"ok": False, "error": "Worker emitted an invalid protocol response"})
            finally:
                output.put(None)
        threading.Thread(target=read_lines, daemon=True).start()
        try:
            ready = output.get(timeout=180)
            if not ready or ready.get("ready") is not True or ready.get("manifest_sha256") != MANIFEST_SHA:
                self.close()
                raise Stage7ClasslessError("The isolated model worker failed to load the verified package.")
            self.ready = ready
            (RUNTIME/"worker_identity.json").write_text(json.dumps(ready), encoding="utf-8")
        except queue.Empty:
            self.close()
            raise Stage7ClasslessError("The isolated model worker timed out during loading.") from None

    def infer(self, original, native, mode):
        with self._lock:
            self.start()
            request = {"request_id": uuid.uuid4().hex, "original": str(original), "native": str(native),
                       "original_sha256": sha(original), "native_sha256": sha(native),
                       "mode": "local" if mode == "local" else "proposals"}
            try:
                self.process.stdin.write(json.dumps(request)+"\n")
                self.process.stdin.flush()
                result = self.output.get(timeout=150)
                if (not result or result.get("request_id") != request["request_id"] or
                        result.get("ok") is not True or result.get("actual_model_execution") is not True or
                        result.get("original_sha256") != request["original_sha256"] or
                        result.get("native_sha256") != request["native_sha256"]):
                    self.close()
                    raise Stage7ClasslessError("The isolated model did not complete this exact frame.")
                return result
            except (queue.Empty, BrokenPipeError, OSError):
                self.close()
                raise Stage7ClasslessError("The isolated model worker did not return this frame; no cached result was substituted.") from None


worker = ResidentWorker()
atexit.register(worker.close)


def _provider_call(key, instructions, content, schema, name):
    rules = geometry.policy()
    payload = {"model": rules["provider_model"], "store": False, "service_tier": "default",
               "reasoning": {"effort": rules["reasoning_effort"]}, "instructions": instructions,
               "input": [{"role": "user", "content": content}], "max_output_tokens": rules["max_output_tokens"],
               "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}}
    req = Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                  headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as response:
            body = response.read(8*1024*1024+1)
        if len(body) > 8*1024*1024:
            raise Stage7ClasslessError("OpenAI returned an oversized response.")
        value = json.loads(body)
        if (value.get("status") != "completed" or not value.get("id") or
                value.get("model") != rules["provider_model"] or
                (value.get("reasoning") or {}).get("effort") != rules["reasoning_effort"]):
            raise Stage7ClasslessError("OpenAI did not complete the requested image analysis.")
        text = "".join(c.get("text", "") for item in value.get("output", [])
                       for c in item.get("content", []) if c.get("type") == "output_text")
        parsed = json.loads(text)
        return parsed, {"response_id": value["id"], "model": value["model"], "usage": value.get("usage", {}),
                        "seconds": time.perf_counter()-start, "phase": name,
                        "reasoning_effort": rules["reasoning_effort"], "actual_provider_execution": True}
    except HTTPError as error:
        code = error.code
        error.close()
        raise Stage7ClasslessError(f"OpenAI image analysis failed (HTTP {code}); no automatic retry or Local substitution.") from None
    except (URLError, TimeoutError):
        raise Stage7ClasslessError("OpenAI image analysis could not reach the service; no automatic retry or Local substitution.") from None
    except (ValueError, KeyError, TypeError):
        raise Stage7ClasslessError("OpenAI returned an invalid or incomplete structured image result.") from None
    finally:
        key = ""
        req = None
        payload.clear()
        content.clear()


def infer(*, image_path, tiles, camera, frame_identity, mode, api_key=None):
    if mode not in ("local", "enhanced"):
        raise Stage7ClasslessError("Unsupported classless inference mode.")
    key = str(api_key or "").strip() if mode == "enhanced" else ""
    if mode == "enhanced" and len(key) < 20:
        raise Stage7ClasslessError("Enter an OpenAI key in this perception tab to enable Enhanced analysis.")
    start = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if not all(checks(mode).values()):
        raise Stage7ClasslessError("The selected classless package does not match its verified registry.")
    identity = component_identity(mode)
    run_id = "classless-current-view-"+uuid.uuid4().hex
    original = native = None
    try:
        original, native, transforms = geometry.aligned_detail(image_path, tiles, camera)
        with tempfile.TemporaryDirectory(prefix="twm-frame-", dir=RUNTIME if RUNTIME.is_dir() else None) as directory:
            native_path = Path(directory)/"native.png"
            native.save(native_path, format="PNG")
            local = worker.infer(Path(image_path), native_path, mode)
            provider = []
            if mode == "local":
                predictions = local["predictions"]
            else:
                rules = geometry.policy()
                first, meta = _provider_call(key, rules["first_instructions"],
                    geometry.selector_content(native, local["proposals"], rules),
                    rules["first_schema"], "b262_discrete_ground_boxes")
                provider.append(meta)
                predictions = geometry.validate_selector(first, local["proposals"], rules, run_id)
                if predictions:
                    content, views = geometry.review_content(native, predictions, rules)
                    reviewed, meta = _provider_call(key, rules["review_instructions"], content,
                        rules["review_schema"], "native_completion_v2")
                    provider.append(meta)
                    predictions = geometry.validate_review(reviewed, predictions, views, native.size, run_id)
            detections = []
            for index, prediction in enumerate(predictions):
                b = prediction["box"]
                if not geometry.bounds(b):
                    raise ValueError("Invalid final box")
                detections.append({**prediction, "detection_id": f"{run_id}:box:{index}",
                                   "class_id": 8, "damage_type": "visible_road_anomaly",
                                   "display_label": None, "classless": True,
                                   "confidence_is_calibrated": False, "tentative": True,
                                   "review": {"status": "pending"},
                                   "consensus": {"strategy": "trained124_classless_"+mode,
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
                   "diagnostics": {"strategy": "trained124_classless_"+mode, "inference_mode": mode,
                        "actual_model_execution": True, "worker_run_id": local["run_id"],
                        "worker_pid": local["worker_pid"], "worker_execution_count": local["execution_count"],
                        "dense_checkpoint_sha256": local["checkpoint_sha256"],
                        "dense_parameter_sha256": local["parameter_sha256"],
                        "forward_images": local["forward_images"], "roi_execution": local["roi_execution"],
                        "model_load_seconds": local["model_load_seconds"], "local_seconds": local["seconds"],
                        "elapsed_seconds": time.perf_counter()-start, "openai_used": mode == "enhanced",
                        "provider_calls": len(provider), "provider_results": provider,
                        "cross_frame_cache_used": False, "assistant_credential_reused": False,
                        "provider_credential_persisted": False, "display_class_names": False,
                        "native_size": list(native.size), "same_camera_transforms": transforms,
                        "gpu_peak_bytes": local["gpu_peak_bytes"]}}
            return {"run": run, "detections": detections}
    except Stage7ClasslessError:
        raise
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise Stage7ClasslessError("The exact-frame classless result failed validation: "+type(error).__name__) from None
    finally:
        key = ""
        if original is not None: original.close()
        if native is not None: native.close()
