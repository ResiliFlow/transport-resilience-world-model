from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = WORKSPACE_ROOT / "contracts" / "stage5_remediation_v2_contract.json"
STAGE5_V2_ROOT = WORKSPACE_ROOT / "datasets" / "stage5_v2"
ARTIFACT_ROOT = STAGE5_V2_ROOT / "artifacts"
COMPLETION_PATH = ARTIFACT_ROOT / "stage5_v2_completion_record.json"
REPORT_PATH = ARTIFACT_ROOT / "STAGE5_V2_REPORT.md"
BASE_REQUIRED_CANDIDATES = {"yolo26n", "yolo26s", "yolo26l"}
CONDITIONAL_CANDIDATE = "yolo26x"
REQUIRED_SEEDS = {20260810, 20260811, 20260812}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"The Stage 5 v2 document could not be loaded: {path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"The Stage 5 v2 document must be a JSON object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage5_v2_contract() -> dict[str, Any]:
    return _load_json(CONTRACT_PATH)


def _verify_relative(relative_path: str, expected_hash: str) -> tuple[bool, str]:
    try:
        path = STAGE5_V2_ROOT / relative_path
        passed = path.is_file() and _sha256(path) == expected_hash
        detail = path.relative_to(WORKSPACE_ROOT).as_posix()
    except (OSError, TypeError, ValueError):
        passed = False
        detail = "invalid artefact record"
    return passed, detail


def stage5_v2_report(verify_artifacts: bool = True) -> dict[str, Any]:
    contract = load_stage5_v2_contract()
    completion = _load_json(COMPLETION_PATH)
    errors: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, detail: Any, message: str) -> None:
        checks.append({"id": identifier, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append({"code": identifier, "path": identifier, "message": message})

    frozen = completion.get("frozen_inputs", {})
    check("contract_version", completion.get("contract_version") == contract.get("contract_version") == "2.0.0", completion.get("contract_version"), "The completion and frozen contract versions do not match.")
    check("contract_hash", frozen.get("contract_sha256") == _sha256(CONTRACT_PATH), frozen.get("contract_sha256"), "The frozen Stage 5 v2 contract changed after training.")
    allowed_candidates = BASE_REQUIRED_CANDIDATES | {CONDITIONAL_CANDIDATE}
    check("candidate_identity", completion.get("selected_candidate") in allowed_candidates, completion.get("selected_candidate"), "The selected detector is not one of the frozen YOLO-priority candidates.")

    candidate_payloads = completion.get("candidate_comparison", {}).get("candidates", [])
    candidate_ids = {item.get("candidate") for item in candidate_payloads}
    valid_candidate_sets = (BASE_REQUIRED_CANDIDATES, BASE_REQUIRED_CANDIDATES | {CONDITIONAL_CANDIDATE})
    check("candidate_comparison_complete", candidate_ids in valid_candidate_sets, sorted(str(item) for item in candidate_ids), "The YOLO-priority candidate comparison is incomplete.")
    final_runs = completion.get("final_training_runs", [])
    final_seeds = {item.get("seed") for item in final_runs}
    check("three_seed_training_complete", final_seeds == REQUIRED_SEEDS, sorted(item for item in final_seeds if isinstance(item, int)), "The selected detector does not have all three frozen full-training seeds.")

    product_gates = completion.get("evaluation", {}).get("product_level_gates", [])
    calculated_passed = sum(item.get("passed") is True for item in product_gates)
    calculated_eligible = bool(product_gates) and calculated_passed == len(product_gates)
    check("product_gate_counts", completion.get("evaluation", {}).get("product_gates_passed") == calculated_passed and completion.get("evaluation", {}).get("product_gates_total") == len(product_gates), {"passed": calculated_passed, "total": len(product_gates)}, "The product-gate totals do not match the recorded decisions.")
    check("stage6_decision", completion.get("stage6_eligible") is calculated_eligible, completion.get("stage6_eligible"), "The Stage 6 decision does not match the product gates.")
    check("downstream_boundary", completion.get("ai_inference_enabled") is False and completion.get("twm_transfer_enabled") is False, {"ai_inference": completion.get("ai_inference_enabled"), "twm_transfer": completion.get("twm_transfer_enabled")}, "Stage 5 must not enable inference or TWM transfer.")

    artifact_checks: list[dict[str, Any]] = []
    if verify_artifacts:
        artifact_records = [
            ("candidate_comparison", completion.get("candidate_comparison", {}).get("report"), completion.get("candidate_comparison", {}).get("report_sha256")),
            ("evaluation_report", completion.get("evaluation", {}).get("report"), completion.get("evaluation", {}).get("report_sha256")),
            ("human_report", completion.get("human_report", {}).get("path"), completion.get("human_report", {}).get("sha256")),
            ("dataset_manifest", frozen.get("dataset_manifest"), frozen.get("dataset_manifest_sha256")),
            ("environment_lock", frozen.get("environment_lock"), frozen.get("environment_lock_sha256")),
            ("materialized_integrity", frozen.get("materialized_integrity"), frozen.get("materialized_integrity_sha256")),
            ("pilot_index_audit", frozen.get("pilot_index_audit"), frozen.get("pilot_index_audit_sha256")),
        ]
        for run in final_runs:
            artifact_records.extend(
                [
                    (f"seed_{run.get('seed')}_completion", run.get("completion_record"), run.get("completion_record_sha256")),
                    (f"seed_{run.get('seed')}_weights", run.get("best_weights"), run.get("best_weights_sha256")),
                    (f"seed_{run.get('seed')}_history", run.get("results_csv"), run.get("results_csv_sha256")),
                ]
            )
        for identifier, relative_path, expected_hash in artifact_records:
            passed, detail = _verify_relative(relative_path, expected_hash)
            artifact_checks.append({"id": identifier, "passed": passed, "detail": detail})
            if not passed:
                errors.append({"code": "artifact_hash_mismatch", "path": str(identifier), "message": f"A required Stage 5 v2 artefact is missing or changed: {detail}"})

    ready = not errors
    evaluation = completion.get("evaluation", {})
    return {
        "stage": 5,
        "contract_version": contract.get("contract_version"),
        "status": completion.get("status") if ready else "stage5_v2_integrity_failed",
        "status_label": completion.get("status_label") if ready else "Stage 5 v2 integrity check failed",
        "scientific_execution_complete": ready and completion.get("scientific_execution_complete") is True,
        "selected_candidate": completion.get("selected_candidate"),
        "selected_family": completion.get("selected_family"),
        "selection_rule": completion.get("selection_rule"),
        "candidate_comparison": copy.deepcopy(completion.get("candidate_comparison", {})),
        "final_training_runs": copy.deepcopy(final_runs),
        "selected_threshold": evaluation.get("selected_threshold"),
        "confidence_calibration": copy.deepcopy(evaluation.get("confidence_calibration", {})),
        "validation_metrics": copy.deepcopy(evaluation.get("validation_metrics", {})),
        "final_cohorts": copy.deepcopy(evaluation.get("final_cohorts", {})),
        "aggregate_final": copy.deepcopy(evaluation.get("aggregate_final", {})),
        "cross_source_cross_source": copy.deepcopy(evaluation.get("cross_source_cross_source", {})),
        "runtime": copy.deepcopy(evaluation.get("runtime", {})),
        "seed_stability": copy.deepcopy(evaluation.get("seed_stability", {})),
        "product_level_gates": copy.deepcopy(product_gates),
        "product_gates_passed": calculated_passed,
        "product_gates_total": len(product_gates),
        "failed_gates": [item.get("id") for item in product_gates if item.get("passed") is not True],
        "stage6_eligible": ready and calculated_eligible,
        "target_area_validated": ready and completion.get("target_area_validated") is True,
        "target_area_note": completion.get("target_area_note"),
        "ai_inference_enabled": False,
        "twm_transfer_enabled": False,
        "downstream_boundary": completion.get("downstream_boundary"),
        "integrity_checks": checks,
        "artifact_checks": artifact_checks,
        "verification_mode": "full_artifact_hashes" if verify_artifacts else "metadata_only",
        "errors": errors,
        "report_url": "api/validation-v2/report",
        "protected_boundary": copy.deepcopy(contract.get("protected_boundary", {})),
    }


def stage5_v2_checks() -> dict[str, Any]:
    report = stage5_v2_report(verify_artifacts=True)
    checks = list(report["integrity_checks"]) + [
        {
            "id": "runtime_artifact_integrity",
            "passed": not report["errors"],
            "detail": f"{len(report['artifact_checks'])} frozen artefacts were hashed.",
        }
    ]
    passed = sum(item.get("passed") is True for item in checks)
    return {
        "stage": 5,
        "contract_version": "2.0.0",
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
        "report": report,
    }
