from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = WORKSPACE_ROOT / "contracts"
STAGE3_ROOT = WORKSPACE_ROOT / "datasets" / "stage3"
STAGE2_CONTRACT_PATH = CONTRACT_ROOT / "stage2_ai_contract.json"
STAGE3_CONTRACT_PATH = CONTRACT_ROOT / "stage3_dataset_contract.json"
DATASET_ASSET_SCHEMA_PATH = CONTRACT_ROOT / "dataset_asset.schema.json"
SOURCE_REGISTRY_PATH = STAGE3_ROOT / "source_registry.json"
DATASET_MANIFEST_PATH = STAGE3_ROOT / "dataset_manifest.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The Stage 3 document could not be loaded: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"The Stage 3 document must be a JSON object: {path.name}")
    return payload


STAGE2_CONTRACT = _load_json_object(STAGE2_CONTRACT_PATH)
STAGE3_CONTRACT = _load_json_object(STAGE3_CONTRACT_PATH) if STAGE3_CONTRACT_PATH.is_file() else {}
DATASET_ASSET_SCHEMA = _load_json_object(DATASET_ASSET_SCHEMA_PATH)
SOURCE_REGISTRY = _load_json_object(SOURCE_REGISTRY_PATH) if SOURCE_REGISTRY_PATH.is_file() else {}


def load_current_manifest() -> dict[str, Any]:
    return _load_json_object(DATASET_MANIFEST_PATH)


def deterministic_split(source_id: str, split_group_id: str, seed: str | None = None) -> str:
    active_seed = seed or str(STAGE3_CONTRACT["split_policy"]["seed"])
    digest = hashlib.sha256(f"{active_seed}|{source_id}|{split_group_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    ratios = STAGE3_CONTRACT["split_policy"]["ratios"]
    train_edge = float(ratios["train"])
    validation_edge = train_edge + float(ratios["validation"])
    if fraction < train_edge:
        return "train"
    if fraction < validation_edge:
        return "validation"
    return "test"


def validate_source_registry(registry: dict[str, Any] | None = None) -> list[dict[str, str]]:
    active = SOURCE_REGISTRY if registry is None else registry
    errors: list[dict[str, str]] = []

    def add(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    stage2_surfaces = {item["id"] for item in STAGE2_CONTRACT.get("surfaces", [])}
    stage2_defects = {item["id"] for item in STAGE2_CONTRACT.get("defects", [])}
    if STAGE3_CONTRACT.get("stage2_contract_version") != STAGE2_CONTRACT.get("contract_version"):
        add("stage2_contract_version", "stage2_contract_version", "Stage 3 must reference the active Stage 2 contract version.")
    if set(STAGE3_CONTRACT.get("required_surfaces", [])) != stage2_surfaces:
        add("surface_contract_mismatch", "required_surfaces", "Stage 3 surfaces must exactly match the frozen Stage 2 taxonomy.")
    if set(STAGE3_CONTRACT.get("required_defects", [])) != stage2_defects:
        add("defect_contract_mismatch", "required_defects", "Stage 3 defects must exactly match the frozen Stage 2 taxonomy.")

    if active.get("registry_version") != "1.0.0":
        add("registry_version", "registry_version", "The active source registry must use version 1.0.0.")
    sources = active.get("sources")
    if not isinstance(sources, list):
        add("registry_sources", "sources", "The source registry requires a sources list.")
        return errors

    seen: set[str] = set()
    by_id: dict[str, dict[str, Any]] = {}
    required_fields = {
        "id",
        "name",
        "record_url",
        "record_version",
        "publisher",
        "licence_id",
        "licence_url",
        "licence_verified",
        "training_eligibility",
        "privacy_review",
        "ingestion_status",
        "label_mapping",
        "attribution",
    }
    for index, source in enumerate(sources):
        path = f"sources[{index}]"
        if not isinstance(source, dict):
            add("invalid_source", path, "Each source registry entry must be an object.")
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            add("source_id", f"{path}.id", "Each source requires a non-empty identifier.")
            continue
        if source_id in seen:
            add("duplicate_source_id", f"{path}.id", "Source identifiers must be unique.")
        seen.add(source_id)
        by_id[source_id] = source
        for field in sorted(required_fields - source.keys()):
            add("missing_source_field", f"{path}.{field}", f"Source field {field} is required.")
        eligibility = source.get("training_eligibility")
        if eligibility not in {"conditional", "permitted", "prohibited"}:
            add("training_eligibility", f"{path}.training_eligibility", "Training eligibility is invalid.")
        if eligibility in {"conditional", "permitted"}:
            if source.get("licence_verified") is not True:
                add("unverified_source_licence", f"{path}.licence_verified", "A candidate training source needs verified licence metadata.")
            archive = source.get("archive")
            if not isinstance(archive, dict):
                add("missing_archive", f"{path}.archive", "A candidate source requires an exact archive record.")
            else:
                checksum = archive.get("checksum")
                algorithm = archive.get("checksum_algorithm")
                expected_length = 32 if algorithm == "md5" else 64 if algorithm == "sha256" else 0
                if not isinstance(checksum, str) or len(checksum) != expected_length or any(character not in "0123456789abcdef" for character in checksum):
                    add("invalid_archive_checksum", f"{path}.archive.checksum", "The official archive checksum is missing or invalid.")
                if not isinstance(archive.get("bytes"), int) or archive.get("bytes", 0) < 1:
                    add("invalid_archive_size", f"{path}.archive.bytes", "A candidate source requires the official positive archive size.")
                if not isinstance(archive.get("download_url"), str) or not archive.get("download_url", "").startswith("https://"):
                    add("invalid_archive_url", f"{path}.archive.download_url", "A candidate source requires an official HTTPS archive URL.")
            label_mapping = source.get("label_mapping")
            if not isinstance(label_mapping, dict) or not label_mapping:
                add("missing_label_mapping", f"{path}.label_mapping", "A candidate source requires an explicit label mapping.")
            elif not set(label_mapping.values()).issubset(stage2_defects):
                add("invalid_label_mapping", f"{path}.label_mapping", "Every mapped label must belong to the frozen Stage 2 taxonomy.")
            if source.get("surface_mapping") not in stage2_surfaces:
                add("invalid_surface_mapping", f"{path}.surface_mapping", "A candidate source requires a frozen Stage 2 surface mapping.")

    required_ids = {"google_maps_core", "individual_user_upload"}
    for source_id in sorted(required_ids - by_id.keys()):
        add("missing_required_source", "sources", f"Required source policy {source_id} is missing.")
    for source_id in ("google_maps_core", "individual_user_upload"):
        if by_id.get(source_id, {}).get("training_eligibility") != "prohibited":
            add("prohibited_source_opened", f"sources.{source_id}", f"{source_id} must remain prohibited for default training ingestion.")
    return errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_asset_path(relative_path: Any, stage_root: Path = STAGE3_ROOT) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        return None
    pure_path = PurePosixPath(relative_path)
    if not pure_path.parts or pure_path.is_absolute() or ".." in pure_path.parts or pure_path.parts[0] != "raw":
        return None
    raw_root = stage_root / "raw"
    resolved = (stage_root / Path(*pure_path.parts)).resolve()
    try:
        resolved.relative_to(raw_root.resolve())
    except ValueError:
        return None
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _validate_region(region: Any, path: str, add: Any) -> None:
    if not isinstance(region, dict):
        add("invalid_annotation_region", path, "An annotation requires a normalised bounding box or polygon.")
        return
    if region.get("type") == "bounding_box":
        values = [region.get(key) for key in ("x_min", "y_min", "x_max", "y_max")]
        if not all(_is_number(value) and 0 <= float(value) <= 1 for value in values):
            add("invalid_annotation_region", path, "Bounding-box coordinates must be numeric values from 0 to 1.")
        elif not (float(values[0]) < float(values[2]) and float(values[1]) < float(values[3])):
            add("invalid_annotation_region", path, "Bounding-box minimum coordinates must be smaller than maximum coordinates.")
        return
    if region.get("type") == "polygon":
        points = region.get("points")
        valid = isinstance(points, list) and len(points) >= 3 and all(
            isinstance(point, list)
            and len(point) == 2
            and all(_is_number(value) and 0 <= float(value) <= 1 for value in point)
            for point in (points or [])
        )
        if not valid:
            add("invalid_annotation_region", path, "A polygon requires at least three normalised [x, y] points.")
        return
    add("invalid_annotation_region", f"{path}.type", "Region type must be bounding_box or polygon.")


def _near_duplicate_pairs(records: list[dict[str, Any]], maximum_distance: int = 3) -> list[tuple[str, str]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        perceptual_hash = record["perceptual_hash"]
        for chunk_index in range(4):
            shift = chunk_index * 16
            chunk = (perceptual_hash >> shift) & 0xFFFF
            key = (chunk_index, chunk)
            for other_index in buckets.get(key, []):
                pairs.add((other_index, index))
            buckets.setdefault(key, []).append(index)

    leaked: list[tuple[str, str]] = []
    for left_index, right_index in pairs:
        left, right = records[left_index], records[right_index]
        if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
            continue
        if (left["perceptual_hash"] ^ right["perceptual_hash"]).bit_count() <= maximum_distance:
            leaked.append((left["asset_id"], right["asset_id"]))
    return leaked


def validate_manifest(manifest: dict[str, Any], *, verify_files: bool, stage_root: Path = STAGE3_ROOT) -> dict[str, Any]:
    errors: list[dict[str, str]] = []

    def add(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    for field in ("manifest_version", "dataset_id", "dataset_status", "contract_version", "split_seed", "frozen", "assets"):
        if field not in manifest:
            add("missing_manifest_field", field, f"Manifest field {field} is required.")
    if manifest.get("manifest_version") != "1.0.0":
        add("manifest_version", "manifest_version", "The Dataset V1 manifest must use version 1.0.0.")
    if manifest.get("dataset_id") != STAGE3_CONTRACT["contract_id"]:
        add("dataset_id", "dataset_id", "The manifest dataset identifier does not match the Stage 3 contract.")
    if manifest.get("contract_version") != STAGE3_CONTRACT["contract_version"]:
        add("dataset_contract_version", "contract_version", "The manifest must use the active Stage 3 contract version.")
    if manifest.get("split_seed") != STAGE3_CONTRACT["split_policy"]["seed"]:
        add("split_seed", "split_seed", "The manifest must use the frozen deterministic split seed.")
    if not isinstance(manifest.get("frozen"), bool):
        add("frozen_flag", "frozen", "The frozen field must be a Boolean value.")

    assets = manifest.get("assets")
    if not isinstance(assets, list):
        add("assets_type", "assets", "The manifest assets field must be a list.")
        assets = []

    source_by_id = {source["id"]: source for source in SOURCE_REGISTRY.get("sources", []) if isinstance(source, dict) and source.get("id")}
    surface_ids = {item["id"] for item in STAGE2_CONTRACT["surfaces"]}
    defect_by_id = {item["id"]: item for item in STAGE2_CONTRACT["defects"]}
    asset_ids: set[str] = set()
    checksums: dict[str, str] = {}
    group_splits: dict[tuple[str, str], str] = {}
    perceptual_records: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    surface_counts = {surface_id: 0 for surface_id in sorted(surface_ids)}
    class_counts = {defect_id: 0 for defect_id in sorted(defect_by_id)}
    class_groups: dict[str, set[str]] = {defect_id: set() for defect_id in sorted(defect_by_id)}
    class_split_counts = {defect_id: {"train": 0, "validation": 0, "test": 0} for defect_id in sorted(defect_by_id)}
    annotation_count = 0

    required_asset_fields = set(DATASET_ASSET_SCHEMA.get("required", []))
    allowed_asset_fields = set(DATASET_ASSET_SCHEMA.get("properties", {}))
    for index, asset in enumerate(assets):
        prefix = f"assets[{index}]"
        if not isinstance(asset, dict):
            add("asset_type", prefix, "Every dataset asset must be an object.")
            continue
        for field in sorted(required_asset_fields - asset.keys()):
            add("missing_asset_field", f"{prefix}.{field}", f"Asset field {field} is required.")
        for field in sorted(asset.keys() - allowed_asset_fields):
            add("unsupported_asset_field", f"{prefix}.{field}", f"Asset field {field} is not part of Dataset V1.")

        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            add("asset_id", f"{prefix}.asset_id", "A non-empty asset identifier is required.")
            asset_id = f"invalid-{index}"
        elif asset_id in asset_ids:
            add("duplicate_asset_id", f"{prefix}.asset_id", "Asset identifiers must be unique.")
        asset_ids.add(asset_id)

        source_id = asset.get("source_id")
        source = source_by_id.get(source_id)
        if source is None:
            add("unknown_training_source", f"{prefix}.source_id", "The asset source is not registered.")
        elif source.get("training_eligibility") not in {"conditional", "permitted"}:
            add("prohibited_training_source", f"{prefix}.source_id", "This source is prohibited for Dataset V1 training ingestion.")

        surface_type = asset.get("surface_type")
        if surface_type not in surface_ids:
            add("invalid_surface_label", f"{prefix}.surface_type", "The surface label is not in the frozen Stage 2 taxonomy.")
        else:
            surface_counts[surface_type] += 1

        split_group_id = asset.get("split_group_id")
        if not isinstance(split_group_id, str) or not split_group_id.strip():
            add("split_group_id", f"{prefix}.split_group_id", "A route, sequence, site or time-location split group is required.")
        split = asset.get("split")
        if split not in split_counts:
            add("invalid_split", f"{prefix}.split", "Split must be train, validation or test.")
        else:
            split_counts[split] += 1
        if isinstance(source_id, str) and isinstance(split_group_id, str) and split_group_id.strip() and split in split_counts:
            expected_split = deterministic_split(source_id, split_group_id, manifest.get("split_seed"))
            if split != expected_split:
                add("non_deterministic_split", f"{prefix}.split", f"This split group must be assigned to {expected_split} by the frozen policy.")
            group_key = (source_id, split_group_id)
            previous_split = group_splits.get(group_key)
            if previous_split is not None and previous_split != split:
                add("split_group_leakage", f"{prefix}.split_group_id", "The same source split group appears in more than one split.")
            group_splits[group_key] = split

        provenance = asset.get("provenance")
        if not isinstance(provenance, dict):
            add("missing_asset_provenance", f"{prefix}.provenance", "Asset-level source, version, licence, attribution and privacy review are required.")
        else:
            for field in ("source_record", "source_version", "licence_id", "attribution", "privacy_review_status"):
                if not isinstance(provenance.get(field), str) or not provenance.get(field, "").strip():
                    add("missing_asset_provenance", f"{prefix}.provenance.{field}", f"Provenance field {field} is required.")
            if source is not None and provenance.get("licence_id") != source.get("licence_id"):
                add("licence_mismatch", f"{prefix}.provenance.licence_id", "The asset licence does not match the registered source version.")
            if source is not None and provenance.get("source_version") != source.get("record_version"):
                add("source_version_mismatch", f"{prefix}.provenance.source_version", "The asset source version does not match the registered version.")
            if provenance.get("privacy_review_status") != "passed":
                add("privacy_review_incomplete", f"{prefix}.provenance.privacy_review_status", "Every retained training image must pass privacy review.")

        annotations = asset.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            add("missing_annotations", f"{prefix}.annotations", "At least one visible-condition annotation is required.")
        else:
            seen_annotations: set[str] = set()
            for annotation_index, annotation in enumerate(annotations):
                annotation_count += 1
                annotation_path = f"{prefix}.annotations[{annotation_index}]"
                if not isinstance(annotation, dict):
                    add("annotation_type", annotation_path, "Each annotation must be an object.")
                    continue
                annotation_id = annotation.get("annotation_id")
                if not isinstance(annotation_id, str) or not annotation_id.strip():
                    add("annotation_id", f"{annotation_path}.annotation_id", "A non-empty annotation identifier is required.")
                elif annotation_id in seen_annotations:
                    add("duplicate_annotation_id", f"{annotation_path}.annotation_id", "Annotation identifiers must be unique within an asset.")
                seen_annotations.add(annotation_id)
                damage_type = annotation.get("damage_type")
                defect = defect_by_id.get(damage_type)
                if defect is None:
                    add("invalid_damage_label", f"{annotation_path}.damage_type", "The damage label is not in the frozen Stage 2 taxonomy.")
                else:
                    class_counts[damage_type] += 1
                    if isinstance(source_id, str) and isinstance(split_group_id, str) and split_group_id.strip():
                        class_groups[damage_type].add(f"{source_id}|{split_group_id}")
                    if split in split_counts:
                        class_split_counts[damage_type][split] += 1
                    if surface_type in surface_ids and surface_type not in defect["applicable_surfaces"]:
                        add("surface_damage_mismatch", f"{annotation_path}.damage_type", "This damage label is not valid for the asset surface.")
                _validate_region(annotation.get("region"), f"{annotation_path}.region", add)

        relative_path = asset.get("relative_path")
        asset_path = _safe_asset_path(relative_path, stage_root)
        if asset_path is None:
            add("unsafe_asset_path", f"{prefix}.relative_path", "Asset paths must be safe POSIX-style paths below datasets/stage3/raw.")
        sha256 = asset.get("sha256")
        if not _valid_sha256(sha256):
            add("invalid_sha256", f"{prefix}.sha256", "A lowercase SHA-256 checksum is required.")
        elif sha256 in checksums:
            add("exact_duplicate", f"{prefix}.sha256", f"This file duplicates asset {checksums[sha256]}.")
        else:
            checksums[sha256] = asset_id

        width, height = asset.get("width"), asset.get("height")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            add("invalid_image_size", f"{prefix}.width", "Image width must be a positive integer.")
        if not isinstance(height, int) or isinstance(height, bool) or height < 1:
            add("invalid_image_size", f"{prefix}.height", "Image height must be a positive integer.")

        if verify_files and asset_path is not None:
            if not asset_path.is_file():
                add("missing_asset_file", f"{prefix}.relative_path", "The frozen asset file does not exist.")
            else:
                try:
                    computed_sha256 = _sha256_file(asset_path)
                    with Image.open(asset_path) as image:
                        image.verify()
                    with Image.open(asset_path) as image:
                        actual_width, actual_height = image.size
                    if computed_sha256 != sha256:
                        add("asset_checksum_mismatch", f"{prefix}.sha256", "The file SHA-256 does not match the manifest.")
                    if actual_width != width or actual_height != height:
                        add("asset_dimension_mismatch", f"{prefix}.width", "The file dimensions do not match the manifest.")
                    perceptual_records.append({
                        "asset_id": asset_id,
                        "split": split,
                        "sha256": sha256,
                        "perceptual_hash": _difference_hash(asset_path),
                    })
                except (OSError, UnidentifiedImageError, ValueError):
                    add("unreadable_asset_file", f"{prefix}.relative_path", "The asset is not a readable image.")

    for left_id, right_id in _near_duplicate_pairs(perceptual_records):
        add("near_duplicate_split_leakage", "assets", f"Near-duplicate assets {left_id} and {right_id} occur in different splits.")

    return {
        "valid": not errors,
        "errors": errors,
        "stats": {
            "asset_count": len(assets),
            "annotation_count": annotation_count,
            "split_counts": split_counts,
            "surface_counts": surface_counts,
            "class_counts": class_counts,
            "class_group_counts": {defect: len(groups) for defect, groups in class_groups.items()},
            "class_split_counts": class_split_counts,
        },
    }


def dataset_readiness_report(manifest: dict[str, Any] | None = None, *, verify_files: bool = True) -> dict[str, Any]:
    active_manifest = manifest or load_current_manifest()
    registry_errors = validate_source_registry()
    validation = validate_manifest(active_manifest, verify_files=verify_files)
    errors = validation["errors"]
    stats = validation["stats"]
    codes = {error["code"] for error in errors}
    asset_count = stats["asset_count"]
    split_counts = stats["split_counts"]
    missing_surfaces = [surface for surface, count in stats["surface_counts"].items() if count == 0]
    missing_defects = [defect for defect, count in stats["class_counts"].items() if count == 0]
    coverage = STAGE3_CONTRACT["minimum_coverage"]
    coverage_shortfalls: list[str] = []
    for surface, count in stats["surface_counts"].items():
        if count < coverage["assets_per_surface"]:
            coverage_shortfalls.append(f"{surface}: {count}/{coverage['assets_per_surface']} assets")
    for defect, count in stats["class_counts"].items():
        if count < coverage["annotations_per_defect"]:
            coverage_shortfalls.append(f"{defect}: {count}/{coverage['annotations_per_defect']} annotations")
        group_count = stats["class_group_counts"][defect]
        if group_count < coverage["independent_split_groups_per_defect"]:
            coverage_shortfalls.append(f"{defect}: {group_count}/{coverage['independent_split_groups_per_defect']} groups")
        validation_count = stats["class_split_counts"][defect]["validation"]
        if validation_count < coverage["validation_annotations_per_defect"]:
            coverage_shortfalls.append(f"{defect}: {validation_count}/{coverage['validation_annotations_per_defect']} validation annotations")
        test_count = stats["class_split_counts"][defect]["test"]
        if test_count < coverage["test_annotations_per_defect"]:
            coverage_shortfalls.append(f"{defect}: {test_count}/{coverage['test_annotations_per_defect']} test annotations")

    def gate(gate_id: str, passed: bool, detail: str) -> dict[str, Any]:
        return {"id": gate_id, "passed": passed, "detail": detail}

    file_codes = {"missing_asset_file", "asset_checksum_mismatch", "asset_dimension_mismatch", "unreadable_asset_file", "unsafe_asset_path", "invalid_sha256"}
    provenance_codes = {"missing_asset_provenance", "licence_mismatch", "source_version_mismatch", "unknown_training_source", "prohibited_training_source"}
    label_codes = {"invalid_surface_label", "invalid_damage_label", "surface_damage_mismatch", "invalid_annotation_region", "missing_annotations"}
    split_codes = {"invalid_split", "non_deterministic_split"}
    gates = [
        gate("source_registry_valid", not registry_errors, "Official source versions, licences and archive checksums are registered." if not registry_errors else f"{len(registry_errors)} source-registry errors remain."),
        gate("manifest_frozen", active_manifest.get("frozen") is True, "The manifest is frozen." if active_manifest.get("frozen") is True else "The manifest remains open because no accepted Dataset V1 has been frozen."),
        gate("assets_present", asset_count > 0, f"{asset_count} assets are registered." if asset_count else "No training assets have been ingested."),
        gate("files_verified", verify_files and asset_count > 0 and not (codes & file_codes), "Every local file matches its checksum and dimensions." if verify_files and asset_count > 0 and not (codes & file_codes) else "Local files are absent or a full checksum and image validation has not run."),
        gate("provenance_complete", asset_count > 0 and not (codes & provenance_codes), "Every asset has source-version and licence provenance." if asset_count > 0 and not (codes & provenance_codes) else "Asset-level provenance is incomplete or no assets exist."),
        gate("privacy_review_complete", asset_count > 0 and "privacy_review_incomplete" not in codes, "Every retained image passed privacy review." if asset_count > 0 and "privacy_review_incomplete" not in codes else "Privacy review is incomplete or no assets exist."),
        gate("labels_valid", asset_count > 0 and not (codes & label_codes), "All annotations match the frozen Stage 2 taxonomy." if asset_count > 0 and not (codes & label_codes) else "Annotations are absent or invalid."),
        gate("split_complete", all(split_counts.values()) and not (codes & split_codes), "Train, validation and test splits are populated by deterministic groups." if all(split_counts.values()) and not (codes & split_codes) else "All three leakage-resistant splits are not yet populated."),
        gate("no_exact_duplicates", asset_count > 0 and "exact_duplicate" not in codes, "No exact duplicate files were detected." if asset_count > 0 and "exact_duplicate" not in codes else "Duplicate checks require accepted assets."),
        gate("no_near_duplicate_leakage", verify_files and asset_count > 0 and "near_duplicate_split_leakage" not in codes, "No near-duplicate files cross split boundaries." if verify_files and asset_count > 0 and "near_duplicate_split_leakage" not in codes else "Near-duplicate leakage checks require accepted local files."),
        gate("no_group_leakage", asset_count > 0 and "split_group_leakage" not in codes, "No route, sequence or site group crosses split boundaries." if asset_count > 0 and "split_group_leakage" not in codes else "Group leakage checks require accepted assets."),
        gate("class_coverage_complete", asset_count > 0 and not coverage_shortfalls, "Every frozen surface and defect meets the minimum asset, annotation, group and held-out coverage floors." if asset_count > 0 and not coverage_shortfalls else f"{len(coverage_shortfalls)} minimum coverage requirements remain unmet."),
    ]
    candidate_sources = [
        {
            "id": source["id"],
            "name": source["name"],
            "role": source["role"],
            "record_url": source["record_url"],
            "doi": source.get("doi"),
            "record_version": source["record_version"],
            "licence_id": source["licence_id"],
            "ingestion_status": source["ingestion_status"],
            "archive_bytes": source.get("archive", {}).get("bytes"),
            "mapped_defects": sorted(set(source.get("label_mapping", {}).values())),
        }
        for source in SOURCE_REGISTRY.get("sources", [])
        if source.get("training_eligibility") in {"conditional", "permitted"}
    ]
    return {
        "stage": 3,
        "contract_version": STAGE3_CONTRACT["contract_version"],
        "status": "dataset_ready" if all(item["passed"] for item in gates) else STAGE3_CONTRACT["status"],
        "status_label": "Dataset V1 ready · model not trained" if all(item["passed"] for item in gates) else STAGE3_CONTRACT["status_label"],
        "dataset_ready": all(item["passed"] for item in gates),
        "model_trained": False,
        "ai_inference_enabled": False,
        "twm_transfer_enabled": False,
        "manifest_status": active_manifest.get("dataset_status"),
        "verification_mode": "full_local_files" if verify_files else "metadata_only",
        "stats": stats,
        "missing_surfaces": missing_surfaces,
        "missing_defects": missing_defects,
        "coverage_requirements": copy.deepcopy(coverage),
        "coverage_shortfalls": coverage_shortfalls,
        "gates_passed": sum(1 for item in gates if item["passed"]),
        "gates_total": len(gates),
        "gates": gates,
        "validation_errors": errors,
        "source_registry_errors": registry_errors,
        "candidate_sources": candidate_sources,
        "legal_note": STAGE3_CONTRACT["legal_note"],
    }


def _sample_asset() -> dict[str, Any]:
    """Synthetic metadata for contract checks, not a distributed research image."""
    source_id = "licensed_road_fixture"
    group_id = "example-route-001-session-001"
    return {
        "asset_id": "road-fixture-000001",
        "source_id": source_id,
        "relative_path": "raw/road_fixture/images/000001.jpg",
        "sha256": "1" * 64,
        "width": 600,
        "height": 600,
        "surface_type": "road",
        "split_group_id": group_id,
        "split": deterministic_split(source_id, group_id),
        "provenance": {
            "source_record": "https://example.org/datasets/road-fixture",
            "source_version": "1",
            "licence_id": "CC-BY-4.0",
            "attribution": "Synthetic contract-check fixture",
            "privacy_review_status": "passed",
            "country": "Synthetic test area",
            "route_or_site": "route-001",
            "capture_session": "session-001"
        },
        "annotations": [
            {
                "annotation_id": "damage-001",
                "damage_type": "pothole",
                "region": {"type": "bounding_box", "x_min": 0.2, "y_min": 0.3, "x_max": 0.5, "y_max": 0.7},
                "source_label": "D40",
                "source_severity": None
            }
        ]
    }


def _sample_manifest(asset: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "manifest_version": "1.0.0",
        "dataset_id": "twm-training-dataset-v1",
        "dataset_status": "building",
        "contract_version": "1.0.0",
        "split_seed": "twm-stage3-dataset-v1",
        "frozen": False,
        "assets": [asset or _sample_asset()],
    }


def run_dataset_contract_checks() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    valid = validate_manifest(_sample_manifest(), verify_files=False)
    cases.append({"id": "valid_asset", "label": "Valid governed asset passes", "passed": valid["valid"], "error_codes": sorted({error["code"] for error in valid["errors"]})})

    prohibited = _sample_asset()
    prohibited["source_id"] = "google_maps_core"
    prohibited["split"] = deterministic_split("google_maps_core", prohibited["split_group_id"])
    result = validate_manifest(_sample_manifest(prohibited), verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "prohibited_source", "label": "Provider-restricted input is rejected", "passed": "prohibited_training_source" in codes, "error_codes": sorted(codes)})

    invalid_label = _sample_asset()
    invalid_label["annotations"][0]["damage_type"] = "invented_damage"
    result = validate_manifest(_sample_manifest(invalid_label), verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "invalid_label", "label": "Unknown damage label is rejected", "passed": "invalid_damage_label" in codes, "error_codes": sorted(codes)})

    invalid_region = _sample_asset()
    invalid_region["annotations"][0]["region"]["x_max"] = 1.4
    result = validate_manifest(_sample_manifest(invalid_region), verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "invalid_region", "label": "Out-of-bounds annotation is rejected", "passed": "invalid_annotation_region" in codes, "error_codes": sorted(codes)})

    missing_provenance = _sample_asset()
    missing_provenance.pop("provenance")
    result = validate_manifest(_sample_manifest(missing_provenance), verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "missing_provenance", "label": "Missing asset provenance is rejected", "passed": "missing_asset_provenance" in codes, "error_codes": sorted(codes)})

    wrong_split = _sample_asset()
    wrong_split["split"] = next(split for split in ("train", "validation", "test") if split != wrong_split["split"])
    result = validate_manifest(_sample_manifest(wrong_split), verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "deterministic_split", "label": "Non-deterministic split is rejected", "passed": "non_deterministic_split" in codes, "error_codes": sorted(codes)})

    duplicate_left = _sample_asset()
    duplicate_right = copy.deepcopy(duplicate_left)
    duplicate_right["asset_id"] = "road-fixture-000002"
    duplicate_right["split_group_id"] = "example-route-002-session-001"
    duplicate_right["split"] = deterministic_split(duplicate_right["source_id"], duplicate_right["split_group_id"])
    duplicate_manifest = _sample_manifest(duplicate_left)
    duplicate_manifest["assets"].append(duplicate_right)
    result = validate_manifest(duplicate_manifest, verify_files=False)
    codes = {error["code"] for error in result["errors"]}
    cases.append({"id": "exact_duplicate", "label": "Exact duplicate asset is rejected", "passed": "exact_duplicate" in codes, "error_codes": sorted(codes)})

    stable = deterministic_split("licensed_road_fixture", "fixed-group") == deterministic_split("licensed_road_fixture", "fixed-group")
    cases.append({"id": "stable_assignment", "label": "Group split assignment is reproducible", "passed": stable, "error_codes": []})
    return cases
