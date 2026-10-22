"""Same-camera compositing and frozen classless set-review policy."""
from __future__ import annotations
import base64
import io
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

POLICY_PATH = Path(__file__).resolve().parents[1] / "models/classless_v022/policy.json"


def policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rotation_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def aligned_detail(image_path, tiles, camera):
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    if original.size != (int(camera["width"]), int(camera["height"])):
        original.close()
        raise ValueError("Canonical camera size does not match the actual frame")
    fov, pitch = float(camera["fov"]), math.radians(float(camera["pitch"]))
    if not 0 < fov < 180 or not -math.pi/2 <= pitch <= math.pi/2:
        original.close()
        raise ValueError("Unsupported perspective camera")
    outw, outh = 1280, round(original.height*1280/original.width)
    out = original.resize((outw, outh), Image.Resampling.LANCZOS)
    focal = outw/(2*math.tan(math.radians(fov/2)))
    inverse = [[1/focal, 0, -outw/(2*focal)], [0, 1/focal, -outh/(2*focal)], [0, 0, 1]]
    transforms = {}
    for name, offset in (("centre", 0), ("left", -fov/3), ("right", fov/3)):
        with Image.open(tiles[name]) as source:
            tile = source.convert("RGB")
        tf = tile.width/(2*math.tan(math.radians(max(10, min(60, fov*4/9))/2)))
        intrinsic = [[tf, 0, tile.width/2], [0, tf, tile.height/2], [0, 0, 1]]
        rotation = mul(mul(rotation_x(-pitch), rotation_y(-math.radians(offset))), rotation_x(pitch))
        matrix = mul(mul(intrinsic, rotation), inverse)
        normal = matrix[2][2]
        if abs(normal) < 1e-10:
            raise ValueError("Degenerate perspective mapping")
        coeffs = [matrix[i][j]/normal for i, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1))]
        warped = tile.transform((outw, outh), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
        valid = Image.new("L", tile.size, 0)
        footer = max(32, round(32*tile.height/640))
        ImageDraw.Draw(valid).rectangle((3, 3, tile.width-4, tile.height-footer-4), fill=255)
        mask = valid.transform((outw, outh), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.NEAREST)
        out.paste(warped, (0, 0), mask)
        transforms[name] = coeffs
        for image in (tile, warped, valid, mask):
            image.close()
    strip = max(16, round(16*outw/original.width))
    with original.resize((outw, outh), Image.Resampling.LANCZOS) as resized:
        with resized.crop((0, outh-strip, outw, outh)) as credit:
            out.paste(credit, (0, outh-strip))
    return original, out, transforms


def image_item(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return {"type": "input_image", "image_url": "data:image/png;base64,"+
                base64.b64encode(buffer.getvalue()).decode(), "detail": "high"}


def bounds(box):
    return (isinstance(box, list) and len(box) == 4 and
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[2] > box[0] and box[3] > box[1])


def validate_metadata(host, confidence):
    if (host not in ("road", "kerb", "footpath") or type(confidence) not in (int, float) or
            not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Invalid host/confidence")


def mapped_box(view, values, width, height):
    if not isinstance(values, list) or len(values) != 4 or not all(type(x) is int and 0 <= x <= 1000 for x in values):
        raise ValueError("Invalid local box coordinates")
    x0, y0, x1, y1 = view
    box = [(x0+values[0]/1000*(x1-x0))/width, (y0+values[1]/1000*(y1-y0))/height,
           (x0+values[2]/1000*(x1-x0))/width, (y0+values[3]/1000*(y1-y0))/height]
    if not bounds(box):
        raise ValueError("Invalid mapped rectangle")
    return box


def selector_content(image, proposals, rules):
    content = []
    for name, affine in rules["views"].items():
        px = [round(v*(image.width if i % 2 == 0 else image.height)) for i, v in enumerate(affine)]
        with image.crop(px) as crop:
            item = image_item(crop)
        content.extend([{"type": "input_text", "text": f"CURRENT SAME FRAME view_id={name}, local0..1000axes. Use full to establish supported ground."}, item])
    table = [{"id": i+1, "xyxy": [round(v*1000, 2) for v in p["box"]]} for i, p in enumerate(proposals)]
    content.append({"type": "input_text", "text":
        "Untrusted alternative boxes. xyxy coordinates always FULL view; select IDs only for complete real problems. Do not classify defects. Independently recover missing issues. TABLE="+json.dumps(table, separators=(",", ":"))})
    return content


def validate_selector(raw, proposals, rules, run_id):
    if not isinstance(raw, dict) or set(raw) != {"selected", "added", "scan_complete"} or raw["scan_complete"] is not True:
        raise ValueError("Incomplete full-view scan")
    if not isinstance(raw["selected"], list) or not isinstance(raw["added"], list) or len(raw["selected"])+len(raw["added"]) > 40:
        raise ValueError("Invalid selector output count")
    result, used = [], set()
    for item in raw["selected"]:
        if (set(item) != {"id", "host", "confidence"} or type(item["id"]) is not int or
                not 1 <= item["id"] <= len(proposals) or item["id"] in used):
            raise ValueError("Invalid or repeated trained proposal ID")
        used.add(item["id"])
        source = proposals[item["id"]-1]
        result.append({"box": list(source["box"]), "surface_type": item["host"],
                       "confidence": item["confidence"], "source_detection_id": source["detection_id"],
                       "geometry_from_trained_detector": True})
    for item in raw["added"]:
        if set(item) != {"view_id", "xyxy", "host", "confidence"} or item["view_id"] not in rules["views"]:
            raise ValueError("Invalid selector recovery source")
        box = mapped_box(rules["views"][item["view_id"]], item["xyxy"], 1, 1)
        result.append({"box": box, "surface_type": item["host"], "confidence": item["confidence"],
                       "geometry_from_trained_detector": False, "independent_recovery": True})
    for i, item in enumerate(result):
        validate_metadata(item["surface_type"], item["confidence"])
        if not bounds(item["box"]):
            raise ValueError("Invalid selector geometry")
        item.update(detection_id=f"{run_id}:initial:{i}", display_label=None, confidence_is_calibrated=False)
    return result


def contexts(predictions, width, height):
    views = {"full": [0, 0, width, height]}
    for i, prediction in enumerate(predictions, 1):
        x0, y0, x1, y1 = prediction["box"]
        cx, cy = (x0+x1)*width/2, (y0+y1)*height/2
        w, h = min(width, max(192, (x1-x0)*width*2)), min(height, max(192, (y1-y0)*height*2))
        left, top = max(0, min(width-w, cx-w/2)), max(0, min(height-h, cy-h/2))
        views["r"+str(i)] = [round(left), round(top), round(left+w), round(top+h)]
    return views


def review_content(image, predictions, rules):
    if not 0 < len(predictions) <= 40:
        raise ValueError("Review requires a bounded nonempty first pass")
    width, height = image.size
    views = contexts(predictions, width, height)
    content = [{"type": "input_text", "text": "full: exact unchanged whole image for host and ground context."},
               image_item(image)]
    for i, prediction in enumerate(predictions, 1):
        name, b = "r"+str(i), prediction["box"]
        px = views[name]
        local = [(b[j]*(width if j % 2 == 0 else height)-px[j % 2])/
                 (px[2+j % 2]-px[j % 2])*1000 for j in range(4)]
        with image.crop(px) as crop:
            shown = crop.convert("RGB")
        ImageDraw.Draw(shown).rectangle([local[j]/1000*(shown.width if j % 2 == 0 else shown.height)
                                        for j in range(4)], outline="#fff231", width=1)
        item = image_item(shown)
        shown.close()
        content.extend([{"type": "input_text", "text":
            f"{name}: proposed id={i}, yellow UNTRUSTED region local xyxy={list(map(lambda v:round(v,2),local))}. Output replacements local to a named view. Previous confidence/class not supplied."}, item])
    for name, affine in rules["views"].items():
        if name == "full":
            continue
        px = [round(v*(width if j % 2 == 0 else height)) for j, v in enumerate(affine)]
        views[name] = px
        with image.crop(px) as crop:
            item = image_item(crop)
        content.extend([{"type": "input_text", "text":
            f"{name}: fixed native-detail strip of the same whole view; local0..1000 coordinates. Check completeness on supported ground, not a separate scene."}, item])
    return content, views


def validate_review(raw, predictions, views, size, run_id):
    if not isinstance(raw, dict) or set(raw) != {"decisions", "complete", "recovered"} or raw["complete"] is not True:
        raise ValueError("Incomplete material review")
    ds = raw["decisions"]
    if not isinstance(ds, list) or sorted(d.get("id", -1) for d in ds) != list(range(1, len(predictions)+1)):
        raise ValueError("Missing or repeated review decisions")
    width, height = size
    result = []
    for item in ds:
        if set(item) != {"id", "action", "reason", "host", "confidence", "replacements"}:
            raise ValueError("Unexpected review fields")
        validate_metadata(item["host"], item["confidence"])
        action = item["action"]
        if action not in ("keep", "replace", "remove") or (action == "replace") != bool(item["replacements"]):
            raise ValueError("Invalid review action")
        if ((action == "remove" and item["reason"] == "visible_problem") or
                (action != "remove" and item["reason"] != "visible_problem")):
            raise ValueError("Inconsistent physical review evidence")
        if action == "remove":
            continue
        source = predictions[item["id"]-1]
        if action == "keep":
            boxes = [list(source["box"])]
        else:
            if len(item["replacements"]) > 5:
                raise ValueError("Too many replacement regions")
            boxes = []
            for replacement in item["replacements"]:
                if set(replacement) != {"view_id", "xyxy"} or replacement["view_id"] not in views:
                    raise ValueError("Invalid replacement view")
                boxes.append(mapped_box(views[replacement["view_id"]], replacement["xyxy"], width, height))
        for box in boxes:
            result.append({"box": box, "surface_type": item["host"], "confidence": item["confidence"],
                           "review_action": action, "source_detection_id": source["detection_id"]})
    if not isinstance(raw["recovered"], list) or len(raw["recovered"]) > 12:
        raise ValueError("Invalid recovery count")
    for item in raw["recovered"]:
        if set(item) != {"view_id", "xyxy", "host", "confidence"} or item["view_id"] not in views:
            raise ValueError("Invalid completeness recovery")
        validate_metadata(item["host"], item["confidence"])
        box = mapped_box(views[item["view_id"]], item["xyxy"], width, height)
        result.append({"box": box, "surface_type": item["host"], "confidence": item["confidence"],
                       "review_action": "independent_recovery"})
    for i, item in enumerate(result):
        if not bounds(item["box"]):
            raise ValueError("Invalid final rectangle")
        item.update(detection_id=f"{run_id}:final:{i}", display_label=None, confidence_is_calibrated=False)
    return result
