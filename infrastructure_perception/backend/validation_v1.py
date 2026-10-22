from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = WORKSPACE_ROOT / "contracts" / "stage5_validation_contract.json"
STAGE5_ROOT = WORKSPACE_ROOT / "datasets" / "stage5"
STAGE4_ROOT = WORKSPACE_ROOT / "datasets" / "stage4"
REGISTRY_PATH = STAGE5_ROOT / "source_registry.json"
ARTIFACT_ROOT = STAGE5_ROOT / "artifacts"
COMPLETION_PATH = ARTIFACT_ROOT / "stage5_completion_record.json"
REPORT_PATH = ARTIFACT_ROOT / "STAGE5_REPORT.md"
BASELINE_PATH = STAGE4_ROOT / "artifacts" / "twm_scl_cnn_v1_baseline.pt"
VISUALISATIONS = {
    "exact-class-f1": (ARTIFACT_ROOT / "visualisations" / "independent_exact_class_f1.png", "image/png"),
    "low-distress-alerts": (ARTIFACT_ROOT / "visualisations" / "low_distress_alert_curve.png", "image/png"),
    "operational-gates": (ARTIFACT_ROOT / "visualisations" / "operational_gate_summary.png", "image/png"),
    "cross_source-failures": (ARTIFACT_ROOT / "visualisations" / "cross_source_failure_examples.jpg", "image/jpeg"),
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The Stage 5 document could not be loaded: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"The Stage 5 document must be a JSON object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage5_contract() -> dict[str, Any]:
    return _load_json(CONTRACT_PATH)


def stage5_report(verify_artifacts: bool = True) -> dict[str, Any]:
    contract = load_stage5_contract()
    completion = _load_json(COMPLETION_PATH)
    errors: list[dict[str, str]] = []

    def add(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "path": path, "message": message})

    if completion.get("contract_sha256") != _sha256(CONTRACT_PATH):
        add("contract_hash_mismatch", "contract_sha256", "The frozen Stage 5 contract changed after validation.")
    if completion.get("source_registry_sha256") != _sha256(REGISTRY_PATH):
        add("source_registry_hash_mismatch", "source_registry_sha256", "The Stage 5 source registry changed after validation.")
    if _sha256(BASELINE_PATH) != contract["frozen_stage4_model"]["sha256"]:
        add("stage4_weight_hash_mismatch", "frozen_stage4_model.sha256", "The frozen Stage 4 baseline changed after Stage 5.")

    scientific = completion.get("scientific_execution_gates", [])
    expected_scientific = set(contract["scientific_execution_gates"])
    if {record.get("id") for record in scientific} != expected_scientific:
        add("scientific_gate_ids", "scientific_execution_gates", "Scientific execution gates do not match the frozen contract.")
    elif any(record.get("passed") is not True for record in scientific):
        add("scientific_gate_failed", "scientific_execution_gates", "At least one Stage 5 scientific execution gate failed.")

    operational = completion.get("operational_eligibility_gates", [])
    expected_operational = {record["id"] for record in contract["operational_eligibility_gates"]}
    if {record.get("id") for record in operational} != expected_operational:
        add("operational_gate_ids", "operational_eligibility_gates", "Operational gates do not match the frozen contract.")
    calculated_eligible = bool(operational) and all(record.get("passed") is True for record in operational)
    if completion.get("stage6_eligible") is not calculated_eligible:
        add("eligibility_decision_mismatch", "stage6_eligible", "The Stage 6 decision does not match the operational gate results.")
    if completion.get("ai_inference_enabled") is not False or completion.get("twm_transfer_enabled") is not False:
        add("downstream_boundary", "ai_inference_enabled", "Stage 5 must not enable AI inference or TWM transfer.")
    if completion.get("model_promotion") != "not_promoted" and not calculated_eligible:
        add("candidate_promotion_boundary", "model_promotion", "A failed Stage 5 candidate cannot be promoted.")

    artifact_checks: list[dict[str, Any]] = []
    if verify_artifacts:
        for name, metadata in completion.get("artifacts", {}).items():
            try:
                path = STAGE5_ROOT / Path(str(metadata["relative_path"]))
                passed = path.is_file() and path.stat().st_size == metadata.get("bytes") and _sha256(path) == metadata.get("sha256")
                detail = str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
            except (KeyError, OSError, TypeError, ValueError):
                passed = False
                detail = "invalid artefact record"
            artifact_checks.append({"id": name, "passed": passed, "detail": detail})
            if not passed:
                add("artifact_hash_mismatch", f"artifacts.{name}", "A required Stage 5 artefact is missing or changed.")

    ready = not errors
    return {
        "stage": 5,
        "contract_version": contract["contract_version"],
        "status": completion.get("status") if ready else "stage5_integrity_failed",
        "status_label": completion.get("status_label") if ready else "Stage 5 integrity check failed",
        "scientific_execution_complete": ready and completion.get("scientific_execution_passed") == completion.get("scientific_execution_total"),
        "scientific_execution_passed": completion.get("scientific_execution_passed", 0),
        "scientific_execution_total": completion.get("scientific_execution_total", 0),
        "operational_gates_passed": completion.get("operational_gates_passed", 0),
        "operational_gates_total": completion.get("operational_gates_total", 0),
        "stage6_eligible": ready and completion.get("stage6_eligible") is True,
        "independent_validation_executed": ready,
        "target_area_validation_complete": ready and completion.get("target_area_target_domain", {}).get("gate_passed") is True,
        "ai_inference_enabled": False,
        "twm_transfer_enabled": False,
        "model_promotion": completion.get("model_promotion"),
        "adaptation": copy.deepcopy(completion.get("adaptation", {})),
        "independent_results": copy.deepcopy(completion.get("independent_results", {})),
        "operational_eligibility_gates": copy.deepcopy(operational),
        "scientific_execution_gates": copy.deepcopy(scientific),
        "source_integrity": copy.deepcopy(completion.get("source_integrity", [])),
        "source_data_quality_exclusions": copy.deepcopy(completion.get("source_data_quality_exclusions", {})),
        "duplicate_scan": copy.deepcopy(completion.get("duplicate_scan", {})),
        "quarantined_source_audit": copy.deepcopy(completion.get("quarantined_source_audit", {})),
        "target_area_target_domain": copy.deepcopy(completion.get("target_area_target_domain", {})),
        "next_action": completion.get("next_action"),
        "artifact_checks": artifact_checks,
        "verification_mode": "full_artifact_hashes" if verify_artifacts else "metadata_only",
        "errors": errors,
        "report_url": "api/validation-v1/report",
        "visualisations": {
            "exact_class_f1": "api/validation-v1/visualisation/exact-class-f1",
            "low_distress_alerts": "api/validation-v1/visualisation/low-distress-alerts",
            "operational_gates": "api/validation-v1/visualisation/operational-gates",
            "cross_source_failures": "api/validation-v1/visualisation/cross_source-failures",
        },
        "completion_semantics": contract["completion_semantics"],
        "protected_boundary": copy.deepcopy(contract["protected_boundary"]),
    }


def stage5_checks() -> dict[str, Any]:
    report = stage5_report(verify_artifacts=True)
    checks = list(report["scientific_execution_gates"]) + [
        {
            "id": "runtime_artifact_integrity",
            "passed": not report["errors"],
            "detail": f"{len(report['artifact_checks'])} Stage 5 artefacts and all frozen hashes were checked.",
        }
    ]
    passed = sum(1 for record in checks if record.get("passed") is True)
    return {
        "stage": 5,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
        "report": report,
    }


def visualisation_path(name: str) -> tuple[Path, str] | None:
    return VISUALISATIONS.get(name)
