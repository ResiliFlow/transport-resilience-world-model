from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = WORKSPACE_ROOT / "contracts" / "stage4_training_contract.json"
STAGE3_ROOT = WORKSPACE_ROOT / "datasets" / "stage3"
STAGE4_ROOT = WORKSPACE_ROOT / "datasets" / "stage4"
ARTIFACT_ROOT = STAGE4_ROOT / "artifacts"
COMPLETION_PATH = ARTIFACT_ROOT / "stage4_completion_record.json"
MODEL_CARD_PATH = ARTIFACT_ROOT / "model_card.json"
VISUAL_ROOT = ARTIFACT_ROOT / "visualisations"
VISUALISATIONS = {
    "confusion-matrix": VISUAL_ROOT / "confusion_matrix.png",
    "per-class-metrics": VISUAL_ROOT / "per_class_metrics.png",
    "held-out-failures": VISUAL_ROOT / "held_out_failures.png",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The Stage 4 document could not be loaded: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"The Stage 4 document must be a JSON object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage4_contract() -> dict[str, Any]:
    return _load_json(CONTRACT_PATH)


def load_model_card() -> dict[str, Any]:
    return _load_json(MODEL_CARD_PATH)


def stage4_report(verify_artifacts: bool = True) -> dict[str, Any]:
    contract = load_stage4_contract()
    completion = _load_json(COMPLETION_PATH)
    errors: list[dict[str, str]] = []

    def add(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    if completion.get("contract_sha256") != _sha256(CONTRACT_PATH):
        add("contract_hash_mismatch", "contract_sha256", "The frozen Stage 4 contract does not match the completed run.")
    manifest_path = STAGE3_ROOT / "dataset_manifest.json"
    audit_path = STAGE3_ROOT / "dataset_v1_audit.json"
    dataset = completion.get("dataset", {})
    if dataset.get("manifest_sha256") != _sha256(manifest_path):
        add("manifest_hash_mismatch", "dataset.manifest_sha256", "The Stage 3 manifest changed after Stage 4 training.")
    if dataset.get("audit_sha256") != _sha256(audit_path):
        add("audit_hash_mismatch", "dataset.audit_sha256", "The Stage 3 audit changed after Stage 4 training.")
    if dataset.get("split_counts") != contract["dataset"]["split_counts"]:
        add("split_count_mismatch", "dataset.split_counts", "The completed run did not use the frozen train, validation and test counts.")
    gates = completion.get("gates", [])
    if not isinstance(gates, list) or len(gates) != len(contract["completion_gates"]):
        add("completion_gate_count", "gates", "The Stage 4 completion gate record is incomplete.")
    elif {gate.get("id") for gate in gates} != set(contract["completion_gates"]):
        add("completion_gate_ids", "gates", "The Stage 4 completion gate identifiers do not match the frozen contract.")
    elif any(gate.get("passed") is not True for gate in gates):
        add("completion_gate_failed", "gates", "At least one Stage 4 completion gate failed.")
    if completion.get("test_evaluation_passes") != 1 or completion.get("test_used_during_training") is not False:
        add("test_isolation", "test_evaluation_passes", "The held-out test must be excluded from fitting and evaluated once.")
    if completion.get("ai_inference_enabled") is not False or completion.get("twm_transfer_enabled") is not False:
        add("downstream_boundary", "ai_inference_enabled", "Stage 4 must not enable AI inference or TWM transfer.")
    artifact_checks: list[dict[str, Any]] = []
    if verify_artifacts:
        for name, metadata in completion.get("artifacts", {}).items():
            try:
                relative_path = Path(str(metadata["relative_path"]))
                path = STAGE4_ROOT / relative_path
                passed = path.is_file() and _sha256(path) == metadata.get("sha256")
            except (KeyError, OSError, TypeError, ValueError):
                path = STAGE4_ROOT
                passed = False
            artifact_checks.append({"id": name, "passed": passed, "detail": str(path.relative_to(WORKSPACE_ROOT)) if path != STAGE4_ROOT else "invalid artefact record"})
            if not passed:
                add("artifact_hash_mismatch", f"artifacts.{name}", "A required Stage 4 artefact is missing or changed.")
    metrics = completion.get("held_out_test_metrics", {})
    expected_classes = set(contract["task"]["defects"])
    if set(metrics.get("per_class", {})) != expected_classes:
        add("class_metrics_incomplete", "held_out_test_metrics.per_class", "All eight frozen defect classes must be reported.")
    ready = not errors
    return {
        "stage": 4,
        "contract_version": contract["contract_version"],
        "status": completion.get("status") if ready else "stage4_integrity_failed",
        "status_label": completion.get("status_label") if ready else "Stage 4 integrity check failed",
        "model_trained": ready and completion.get("model_trained") is True,
        "independent_validation_complete": False,
        "target_area_validation_complete": False,
        "ai_inference_enabled": False,
        "twm_transfer_enabled": False,
        "dataset": copy.deepcopy(dataset),
        "model": copy.deepcopy(completion.get("model", {})),
        "environment": copy.deepcopy(completion.get("environment", {})),
        "validation_seed_sensitivity": copy.deepcopy(completion.get("validation_seed_sensitivity", {})),
        "held_out_test_metrics": copy.deepcopy(metrics),
        "simple_baseline_test_metrics": copy.deepcopy(completion.get("simple_baseline_test_metrics", {})),
        "gates_passed": sum(1 for gate in gates if gate.get("passed") is True),
        "gates_total": len(gates),
        "gates": copy.deepcopy(gates),
        "artifact_checks": artifact_checks,
        "verification_mode": "full_artifact_hashes" if verify_artifacts else "metadata_only",
        "errors": errors,
        "limitations": copy.deepcopy(completion.get("limitations", [])),
        "model_card_url": "api/model-v1/model-card",
        "visualisations": {
            "confusion_matrix": "api/model-v1/visualisation/confusion-matrix",
            "per_class_metrics": "api/model-v1/visualisation/per-class-metrics",
            "held_out_failures": "api/model-v1/visualisation/held-out-failures",
        },
        "scientific_boundary": contract["scientific_boundary"],
        "distribution_status": contract["distribution_status"],
    }


def stage4_checks() -> dict[str, Any]:
    report = stage4_report(verify_artifacts=True)
    checks = list(report["gates"]) + [
        {
            "id": "runtime_artifact_integrity",
            "passed": not report["errors"],
            "detail": f"{len(report['artifact_checks'])} generated artefacts and the frozen contract/dataset hashes were checked.",
        }
    ]
    passed = sum(1 for check in checks if check.get("passed") is True)
    return {
        "stage": 4,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
        "report": report,
    }


def visualisation_path(name: str) -> Path | None:
    return VISUALISATIONS.get(name)
