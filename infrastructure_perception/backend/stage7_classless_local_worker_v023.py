"""Resident D130/ROI132 Local inference with the frozen trained surface gate."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infrastructure_perception.backend import stage7_classless_worker_v022 as base

torch = base.torch
PACKAGE = ROOT / "infrastructure_perception/models/classless_local_v023"


class SupportedRegionModel(base.RegionModel):
    def __init__(self, weights):
        super().__init__(weights)
        self.trace = []

    def forward(self, pixels):
        output = super().forward(pixels)
        support = output["hosts"].sigmoid().max(1).values
        admitted = support >= .2
        foreground = torch.logsumexp(output["logits"][:, :8], 1) - output["logits"][:, 8]
        self.trace.extend({"foreground_logit": float(z), "support": float(s), "admitted": bool(a)}
            for z, s, a in zip(foreground.cpu(), support.cpu(), admitted.cpu(), strict=True))
        logits = output["logits"].clone()
        logits[~admitted, :8] = -60
        logits[~admitted, 8] = 0
        return dict(output, logits=logits)


class Engine(base.Engine):
    def __init__(self, manifest_sha):
        started = time.perf_counter()
        if not re.fullmatch("[A-F0-9]{64}", manifest_sha) or base.sha(PACKAGE / "manifest.json") != manifest_sha:
            raise ValueError("Local package manifest integrity mismatch")
        self.manifest_sha256 = manifest_sha
        self.manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
        if base.sha(base.__file__) != self.manifest["frozen_base_worker_sha256"]:
            raise ValueError("The shared frozen inference implementation changed")
        for name, digest in self.manifest["files"].items():
            if base.sha(PACKAGE / name) != digest:
                raise ValueError("Local package asset integrity mismatch: " + name)
        if self.manifest["local_policy"] != {"foreground_logit": 1.0096738338470461,
                "max_area_fraction": .25, "nms_iou": .5, "max_boxes": 30, "support_floor": .2}:
            raise ValueError("Local support or selection policy mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("The isolated GPU inference runtime is unavailable")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(20260915319)
        weights = torch.load(PACKAGE / "dense.pt", map_location="cpu", weights_only=False)
        self.dense = weights["model"].float().eval().cuda().requires_grad_(False)
        self.dense.model[-1].max_det = 300
        self.parameter_sha256 = base.tensor_hash(self.dense)
        if self.parameter_sha256 != self.manifest["dense_parameter_sha256"]:
            raise ValueError("Loaded detector parameters do not match Dense130")
        weights = torch.load(PACKAGE / "roi.pt", map_location="cpu", weights_only=False)
        self.roi = SupportedRegionModel(weights).eval().cuda().requires_grad_(False)
        if base.tensor_hash(self.roi) != self.manifest["roi_export_parameter_sha256"]:
            raise ValueError("Loaded region parameters do not match ROI132")
        self.execution_count = 0
        self.load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def infer(self, request):
        if request.get("mode") != "local":
            raise ValueError("This package supports Local analysis only")
        self.roi.trace = []
        result = super().infer(request)
        if len(self.roi.trace) != len(result["proposals"]):
            raise ValueError("Incomplete surface inference trace")
        old_id = result["run_id"]
        result["run_id"] = old_id.replace("d124-worker-", "d130-r132-worker-", 1)
        for prediction in result["proposals"] + result["predictions"]:
            prediction["detection_id"] = prediction["detection_id"].replace(old_id, result["run_id"], 1)
        result.update(model_version=self.manifest["package_version"],
            manifest_sha256=self.manifest_sha256, surface_gate_trace=self.roi.trace,
            surface_gate_threshold=.2, roi_checkpoint_sha256=self.manifest["roi_source_sha256"],
            roi_parameter_sha256=self.manifest["roi_parameter_sha256"],
            roi_export_parameter_sha256=self.manifest["roi_export_parameter_sha256"])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha", required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(args.manifest_sha)
    print(json.dumps({"ready": True, "pid": os.getpid(), "manifest_sha256": args.manifest_sha,
        "parameter_sha256": engine.parameter_sha256, "model_version": engine.manifest["package_version"],
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
