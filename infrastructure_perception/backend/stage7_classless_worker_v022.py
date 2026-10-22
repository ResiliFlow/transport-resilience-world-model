"""Isolated, resident D124/ROI129 inference. No provider credentials or training."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

def startup_event(stage):
    print(json.dumps({"event": "classless_worker_startup", "stage": stage,
                      "pid": os.getpid(), "time": time.time()}), file=sys.stderr, flush=True)

startup_event("import_dependencies")
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from transformers import SiglipVisionConfig, SiglipVisionModel
startup_event("dependencies_imported")

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "infrastructure_perception/models/classless_v022"
MANIFEST_SHA = "C48B14EEA86B99931D6A6DA6F13B4701DB948F5D36D513144D725C0C9F42EF37"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def tensor_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def crop_tensor(image, box, factor):
    w, h = image.size
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1+x2)*w/2, (y1+y2)*h/2
    bw, bh = max(2., (x2-x1)*w)*factor, max(2., (y2-y1)*h)*factor
    rect = (max(0, math.floor(cx-bw/2)), max(0, math.floor(cy-bh/2)),
            min(w, math.ceil(cx+bw/2)), min(h, math.ceil(cy+bh/2)))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError("Empty ROI")
    with image.crop(rect) as cropped:
        with cropped.resize((256, 256), Image.Resampling.BILINEAR) as resized:
            a = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)/127.5-1
    return torch.from_numpy(a.copy())


class RegionModel(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        cfg = SiglipVisionConfig(**weights["vision_config"])
        cfg._attn_implementation = "sdpa"
        self.vision = SiglipVisionModel(cfg)
        self.classes = torch.nn.Linear(1536, 9)
        self.hosts = torch.nn.Linear(1536, 3)
        for name in ("vision", "classes", "hosts"):
            getattr(self, name).load_state_dict(weights[name], strict=True)

    def forward(self, pixels):
        with torch.autocast("cuda", dtype=torch.float16):
            values = self.vision(pixel_values=pixels).pooler_output
        features = F.normalize(values.float(), dim=1).reshape(len(pixels)//2, 1536)
        return {"logits": self.classes(features), "hosts": self.hosts(features)}


def positions(length):
    if length <= 640:
        return [0]
    count = math.ceil((length-640)/320)+1
    return list(dict.fromkeys(round(i*(length-640)/(count-1)) for i in range(count)))


def tensor_views(original, native):
    ow, oh = original.size
    w, h = native.size
    ratio = min(640/ow, 640/oh)
    rw, rh = round(ow*ratio), round(oh*ratio)
    pad_x, pad_y = (640-rw)//2, (640-rh)//2
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    resized = original.resize((rw, rh), Image.Resampling.BILINEAR) if (rw, rh) != original.size else original
    canvas.paste(resized, (pad_x, pad_y))
    arrays = [np.asarray(canvas).copy()]
    maps = [{"x": 0, "y": 0, "rw": rw, "rh": rh, "px": pad_x, "py": pad_y,
             "full": True, "width": ow, "height": oh}]
    if resized is not original:
        resized.close()
    canvas.close()
    for y in positions(h):
        for x in positions(w):
            tw, th = min(640, w-x), min(640, h-y)
            px, py = (640-tw)//2, (640-th)//2
            canvas = Image.new("RGB", (640, 640), (114, 114, 114))
            with native.crop((x, y, x+tw, y+th)) as tile:
                canvas.paste(tile, (px, py))
            arrays.append(np.asarray(canvas).copy())
            maps.append({"x": x, "y": y, "rw": tw, "rh": th, "px": px, "py": py,
                         "full": False, "width": w, "height": h})
            canvas.close()
    if len(arrays) > 25:
        raise ValueError("Current frame aspect ratio exceeds the supported inference window")
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255).cuda()
    return tensor, maps


def decode(decoded, logits, maps, run_id):
    confidence = torch.logsumexp(logits.float(), dim=1).sigmoid()
    parts = []
    for view, m in enumerate(maps):
        mask = confidence[view] >= .005
        local = decoded[view][mask].float()
        scores = confidence[view][mask].float()
        mapped = local.clone()
        if not m["full"]:
            mapped[:, [0, 2]] = mapped[:, [0, 2]].clamp(m["px"], m["px"]+m["rw"])
            mapped[:, [1, 3]] = mapped[:, [1, 3]].clamp(m["py"], m["py"]+m["rh"])
        mapped[:, [0, 2]] = (mapped[:, [0, 2]]-m["px"])/m["rw"]
        mapped[:, [1, 3]] = (mapped[:, [1, 3]]-m["py"])/m["rh"]
        if not m["full"]:
            mapped[:, [0, 2]] = (mapped[:, [0, 2]]*m["rw"]+m["x"])/m["width"]
            mapped[:, [1, 3]] = (mapped[:, [1, 3]]*m["rh"]+m["y"])/m["height"]
        mapped.clamp_(0, 1)
        valid = torch.isfinite(mapped).all(1) & torch.isfinite(scores)
        valid &= (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
        mapped, scores, local = mapped[valid], scores[valid], local[valid]
        indices = torch.arange(logits.shape[2], device="cuda")[mask][valid]
        edge = torch.zeros(len(local), dtype=torch.bool, device="cuda")
        if not m["full"]:
            if m["x"] > 0: edge |= local[:, 0] <= m["px"]+1
            if m["y"] > 0: edge |= local[:, 1] <= m["py"]+1
            if m["x"]+m["rw"] < m["width"]: edge |= local[:, 2] >= m["px"]+m["rw"]-1
            if m["y"]+m["rh"] < m["height"]: edge |= local[:, 3] >= m["py"]+m["rh"]-1
        keep = nms(mapped, scores, .60)[:300]
        parts.append((mapped[keep], scores[keep], torch.full((len(keep),), view, device="cuda"),
                      edge[keep], indices[keep]))
    boxes = torch.cat([p[0] for p in parts])
    scores = torch.cat([p[1] for p in parts])
    views = torch.cat([p[2] for p in parts])
    edges = torch.cat([p[3] for p in parts])
    anchors = torch.cat([p[4] for p in parts])
    keep = nms(boxes, scores, .60)[:300]
    return [{"box": boxes[k].tolist(), "confidence": float(scores[k]), "source_view": int(views[k]),
             "touches_artificial_crop_edge": bool(edges[k]), "source_anchor": int(anchors[k]),
             "detection_id": f"{run_id}:proposal:{i}", "display_label": None}
            for i, k in enumerate(keep.tolist())]


class Engine:
    def __init__(self):
        started = time.perf_counter()
        startup_event("verify_package")
        if sha(PACKAGE/"manifest.json") != MANIFEST_SHA:
            raise ValueError("Package manifest integrity mismatch")
        self.manifest = json.loads((PACKAGE/"manifest.json").read_text(encoding="utf-8"))
        for name, digest in self.manifest["files"].items():
            if sha(PACKAGE/name) != digest:
                raise ValueError("Package asset integrity mismatch: "+name)
        if not torch.cuda.is_available():
            raise RuntimeError("The isolated GPU inference runtime is unavailable")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(20260914273)
        weights = torch.load(PACKAGE/"dense.pt", map_location="cpu", weights_only=False)
        startup_event("dense_weights_loaded")
        self.dense = weights["model"].float().eval().cuda().requires_grad_(False)
        self.dense.model[-1].max_det = 300
        self.parameter_sha256 = tensor_hash(self.dense)
        if self.parameter_sha256 != self.manifest["dense_parameter_sha256"]:
            raise ValueError("Loaded detector parameters differ from valid epoch124")
        roi = torch.load(PACKAGE/"roi.pt", map_location="cpu", weights_only=False)
        self.roi = RegionModel(roi).eval().cuda().requires_grad_(False)
        startup_event("roi_weights_loaded")
        self.load_seconds = time.perf_counter()-started
        self.execution_count = 0

    @torch.inference_mode()
    def infer(self, request):
        original_path, native_path = Path(request["original"]), Path(request["native"])
        original_sha, native_sha = sha(original_path), sha(native_path)
        if original_sha != request["original_sha256"] or native_sha != request["native_sha256"]:
            raise ValueError("Current frame identity changed before inference")
        started = time.perf_counter()
        run_id = "d124-worker-"+uuid.uuid4().hex
        with Image.open(original_path) as image:
            original = image.convert("RGB")
        with Image.open(native_path) as image:
            native = image.convert("RGB")
        try:
            tensor, mapping = tensor_views(original, native)
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
            out = self.dense(tensor)
            raw = out[1] if isinstance(out, tuple) else out
            branch = raw["one2many"]
            boxes = self.dense.model[-1]._get_decode_boxes(branch).transpose(1, 2)
            proposals = decode(boxes, branch["scores"], mapping, run_id)
            torch.cuda.synchronize()
            dense_seconds = time.perf_counter()-forward_start
            del tensor, out, raw, branch, boxes
            predictions = []
            roi_started = time.perf_counter()
            if request["mode"] == "local":
                zs, hosts = [], []
                for offset in range(0, len(proposals), 8):
                    rows = proposals[offset:offset+8]
                    pixels = torch.stack([crop_tensor(native, p["box"], factor)
                                          for p in rows for factor in (1., 1.8)]).cuda()
                    output = self.roi(pixels)
                    zs.extend((torch.logsumexp(output["logits"][:, :8], 1)-output["logits"][:, 8]).cpu().tolist())
                    hosts.extend(output["hosts"].float().cpu().tolist())
                    del pixels, output
                candidates = []
                for p, z, host in zip(proposals, zs, hosts, strict=True):
                    b = p["box"]
                    if z >= 1.0096738338470461 and (b[2]-b[0])*(b[3]-b[1]) <= .25:
                        candidates.append(dict(p, proposal_confidence=p["confidence"], foreground_logit=z,
                                               confidence=1/(1+math.exp(-max(-60, min(60, z)))),
                                               surface_type=("road", "kerb", "footpath")[int(np.argmax(host))],
                                               host_logits=host, host_estimate_uncalibrated=True))
                if candidates:
                    keep = nms(torch.tensor([p["box"] for p in candidates]),
                               torch.tensor([p["foreground_logit"] for p in candidates]), .5)[:30].tolist()
                    predictions = [candidates[i] for i in keep]
            elif request["mode"] != "proposals":
                raise ValueError("Unknown isolated inference mode")
            torch.cuda.synchronize()
            self.execution_count += 1
            return {"request_id": request["request_id"], "run_id": run_id, "worker_pid": os.getpid(),
                    "execution_count": self.execution_count, "actual_model_execution": True,
                    "checkpoint_sha256": self.manifest["dense_source_sha256"],
                    "parameter_sha256": self.parameter_sha256, "model_load_seconds": self.load_seconds,
                    "original_sha256": original_sha, "native_sha256": native_sha,
                    "dense_seconds": dense_seconds, "roi_seconds": time.perf_counter()-roi_started,
                    "seconds": time.perf_counter()-started, "forward_images": len(mapping),
                    "roi_execution": request["mode"] == "local", "proposals": proposals,
                    "predictions": predictions, "cross_frame_cache": False,
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(), "provider_calls": 0}
        finally:
            original.close()
            native.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine()
    if args.parity:
        ref = torch.load(PACKAGE/"parity.pt", map_location="cpu", weights_only=False)
        with torch.inference_mode():
            actual = engine.roi(ref["pixels"].cuda())
        errors = {k: float((actual[k].cpu()-ref[k]).abs().max()) for k in ("logits", "hosts")}
        if max(errors.values()) > 1e-5:
            raise ValueError("Export changed ROI inference: "+str(errors))
        print(json.dumps({"passed": True, "roi_max_absolute_errors": errors,
                          "dense_parameter_sha256": engine.parameter_sha256,
                          "load_seconds": engine.load_seconds}), flush=True)
        return
    print(json.dumps({"ready": True, "pid": os.getpid(), "manifest_sha256": MANIFEST_SHA,
                      "parameter_sha256": engine.parameter_sha256,
                      "load_seconds": engine.load_seconds}), flush=True)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                break
            with contextlib.redirect_stdout(sys.stderr):
                result = engine.infer(request)
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "request_id": request.get("request_id"),
                              "error_type": type(error).__name__, "error": str(error)[:300]}), flush=True)


if __name__ == "__main__":
    main()
