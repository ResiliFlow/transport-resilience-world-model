from __future__ import annotations

"""Conditional multi-LLM review for mathematical model creation requests.

Normal navigation, analysis and scenario requests never invoke this workflow.
It activates only when the user explicitly asks to create, formulate or modify
a mathematical model.
"""

import re
from typing import Any

from .llm_providers import PROVIDER_SPECS, call_provider_json, get_credential, provider_status


MODEL_REQUEST_PATTERNS = (
    r"\b(?:build|create|formulate|design|derive|modify|rewrite)\s+(?:a\s+)?(?:new\s+)?(?:(?:mathematical|optimisation|optimization|simulation|traffic|transport|network)\s+){0,3}model\b",
    r"\b(?:change|extend|audit)\s+(?:the\s+)?(?:mathematical\s+)?formulation\b",
    r"(?:建立|创建|构建|设计|修改|扩充|推导|审查)(?:一个|这个|现有)?(?:数学|优化|仿真|交通|网络)?模型",
    r"(?:数学|优化)(?:建模|模型设计|模型修改|模型审查)",
)


def is_model_design_request(message: str) -> bool:
    cleaned = " ".join(str(message).strip().lower().split())
    return any(re.search(pattern, cleaned, flags=re.IGNORECASE) for pattern in MODEL_REQUEST_PATTERNS)


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name", "decision_question", "scope", "sets", "parameters", "variables",
        "objective", "constraints", "assumptions", "required_data", "solution_method",
        "verification_tests", "claim_boundary", "source_links",
    ],
    "properties": {
        "name": {"type": "string"},
        "decision_question": {"type": "string"},
        "scope": {"type": "string"},
        "sets": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "description"],
            "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
        }},
        "parameters": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "indices", "description", "unit"],
            "properties": {
                "name": {"type": "string"}, "indices": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}, "unit": {"type": "string"},
            },
        }},
        "variables": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "indices", "domain", "unit", "description"],
            "properties": {
                "name": {"type": "string"}, "indices": {"type": "array", "items": {"type": "string"}},
                "domain": {"type": "string", "enum": ["binary", "integer", "nonnegative", "continuous", "real"]},
                "unit": {"type": "string"}, "description": {"type": "string"},
            },
        }},
        "objective": {
            "type": "object", "additionalProperties": False,
            "required": ["sense", "expression", "units", "description"],
            "properties": {
                "sense": {"type": "string", "enum": ["min", "max"]}, "expression": {"type": "string"},
                "units": {"type": "string"}, "description": {"type": "string"},
            },
        },
        "constraints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "indices", "expression", "purpose", "left_units", "right_units"],
            "properties": {
                "name": {"type": "string"}, "indices": {"type": "array", "items": {"type": "string"}},
                "expression": {"type": "string"}, "purpose": {"type": "string"},
                "left_units": {"type": "string"}, "right_units": {"type": "string"},
            },
        }},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "required_data": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "unit", "role", "required"],
            "properties": {
                "field": {"type": "string"}, "unit": {"type": "string"}, "role": {"type": "string"},
                "required": {"type": "boolean"},
            },
        }},
        "solution_method": {
            "type": "object", "additionalProperties": False,
            "required": ["method", "termination", "complexity_note"],
            "properties": {
                "method": {"type": "string"}, "termination": {"type": "string"}, "complexity_note": {"type": "string"},
            },
        },
        "verification_tests": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "description", "pass_condition"],
            "properties": {
                "type": {"type": "string", "enum": ["unit", "invariant", "numerical", "counterexample"]},
                "description": {"type": "string"}, "pass_condition": {"type": "string"},
            },
        }},
        "claim_boundary": {"type": "array", "items": {"type": "string"}},
        "source_links": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["citation", "doi", "url", "used_for"],
            "properties": {
                "citation": {"type": "string"}, "doi": {"type": "string"}, "url": {"type": "string"},
                "used_for": {"type": "string"},
            },
        }},
    },
}


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "issues", "required_tests", "strengths"],
    "properties": {
        "verdict": {"type": "string", "enum": ["reject", "major_revision", "minor_revision", "provisionally_coherent"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["severity", "category", "claim", "reason", "fix", "test"],
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "category": {"type": "string"},
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "fix": {"type": "string"},
                    "test": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required_tests": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
    },
}


def deterministic_model_checks(proposal: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    required_sections = list(PROPOSAL_SCHEMA["required"])
    missing_sections = [name for name in required_sections if not proposal.get(name)]
    record("required_sections", not missing_sections, "Missing: " + ", ".join(missing_sections) if missing_sections else "All required sections are present.")

    variables = proposal.get("variables") if isinstance(proposal.get("variables"), list) else []
    variable_names = [str(item.get("name", "")).strip() for item in variables if isinstance(item, dict)]
    record("unique_variable_names", bool(variable_names) and len(variable_names) == len(set(variable_names)), "Variable names must be non-empty and unique.")
    domains_valid = all(
        str(item.get("domain", "")).lower() in {"binary", "integer", "nonnegative", "continuous", "real"}
        for item in variables if isinstance(item, dict)
    )
    record("declared_variable_domains", bool(variables) and domains_valid, "Every decision variable needs a recognised mathematical domain.")

    objective = proposal.get("objective") if isinstance(proposal.get("objective"), dict) else {}
    record("objective_sense", str(objective.get("sense", "")).lower() in {"min", "minimise", "minimize", "max", "maximise", "maximize"}, "Objective sense must be explicit.")
    record("objective_expression", bool(str(objective.get("expression", "")).strip()), "Objective expression must be explicit.")
    record("objective_units", bool(str(objective.get("units", "")).strip()), "Objective units or 'dimensionless' must be stated.")

    constraints = proposal.get("constraints") if isinstance(proposal.get("constraints"), list) else []
    complete_constraints = all(
        isinstance(item, dict)
        and str(item.get("name", "")).strip()
        and str(item.get("expression", "")).strip()
        and str(item.get("purpose", "")).strip()
        for item in constraints
    )
    record("constraint_structure", bool(constraints) and complete_constraints, "Every constraint needs a name, expression and purpose.")
    unit_checked = all(
        isinstance(item, dict)
        and str(item.get("left_units", "")).strip()
        and str(item.get("right_units", "")).strip()
        for item in constraints
    )
    record("constraint_unit_declarations", bool(constraints) and unit_checked, "Both sides of every constraint must declare units or 'dimensionless'.")

    required_data = proposal.get("required_data") if isinstance(proposal.get("required_data"), list) else []
    data_contract = all(
        isinstance(item, dict)
        and str(item.get("field", "")).strip()
        and str(item.get("unit", "")).strip()
        and str(item.get("role", "")).strip()
        for item in required_data
    )
    record("data_contract", bool(required_data) and data_contract, "Every input needs a field, unit and mathematical role.")

    tests = proposal.get("verification_tests") if isinstance(proposal.get("verification_tests"), list) else []
    test_types = {str(item.get("type", "")).lower() for item in tests if isinstance(item, dict)}
    required_test_types = {"unit", "invariant", "numerical", "counterexample"}
    record("verification_coverage", required_test_types.issubset(test_types), "Required test types: unit, invariant, numerical and counterexample.")

    sources = proposal.get("source_links") if isinstance(proposal.get("source_links"), list) else []
    traceable_sources = all(
        isinstance(item, dict)
        and str(item.get("citation", "")).strip()
        and (str(item.get("doi", "")).strip() or str(item.get("url", "")).strip())
        for item in sources
    )
    record("source_traceability", bool(sources) and traceable_sources, "Each theoretical source needs a citation plus DOI or primary URL.")
    return {
        "passed": all(item["passed"] for item in checks),
        "passed_count": sum(item["passed"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
    }


def build_and_review_model(
    *,
    session_id: str,
    primary_provider: str,
    user_request: str,
    project_context: dict[str, Any],
    model_registry: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_credential = get_credential(session_id, primary_provider)
    if primary_credential is None:
        raise ValueError("Configure the selected LLM provider before asking it to create or modify a mathematical model.")
    evidence = [
        source
        for model in model_registry
        for source in (model.get("literature_sources") or model.get("card", {}).get("primary_sources") or [])
    ]
    proposal = call_provider_json(
        provider=primary_provider,
        credential=primary_credential,
        system=(
            "You are the model-formulation lead for a disaster transport research platform. "
            "Produce an auditable mathematical proposal, not runnable code and not an operational order. "
            "Use only the supplied project fields and traceable source ledger. Never invent observed data, citations or calibration. "
            "Declare sets, indexed parameters, variables with domains and units, objective, every constraint, assumptions, data contract, solution method, tests and claim boundaries."
        ),
        user_payload={
            "request": user_request,
            "project": project_context,
            "traceable_sources": evidence,
            "required_test_types": ["unit", "invariant", "numerical", "counterexample"],
        },
        schema=PROPOSAL_SCHEMA,
    )
    deterministic = deterministic_model_checks(proposal)
    configured = [item for item in provider_status(session_id)["providers"] if item["configured"]]
    review_roles = [
        "mathematical consistency and notation auditor",
        "transport modelling and behavioural assumptions reviewer",
        "counterexample, identifiability and data sufficiency attacker",
    ]
    reviews: list[dict[str, Any]] = []
    for provider_item, role in zip(
        [item for item in configured if item["id"] != primary_provider][:3],
        review_roles,
    ):
        provider = provider_item["id"]
        credential = get_credential(session_id, provider)
        if credential is None:
            continue
        review = call_provider_json(
            provider=provider,
            credential=credential,
            system=(
                f"Act as an independent {role}. Do not rewrite the proposal. Attack unsupported claims, undefined symbols, "
                "dimensional errors, non-identifiability, infeasibility, invalid equilibrium/optimisation assumptions and missing tests. "
                "A literature citation is not proof. Return actionable issues and falsification tests."
            ),
            user_payload={"request": user_request, "proposal": proposal, "deterministic_checks": deterministic},
            schema=REVIEW_SCHEMA,
        )
        reviews.append({"provider": provider, "model": credential.model, "role": role, **review})

    issue_counts = {
        severity: sum(
            issue.get("severity") == severity
            for review in reviews
            for issue in review.get("issues", [])
        )
        for severity in ("critical", "major", "minor")
    }
    cross_review_complete = len(reviews) >= 2
    status = (
        "blocked_by_critical_review"
        if issue_counts["critical"]
        else "revision_required"
        if issue_counts["major"] or not deterministic["passed"]
        else "provisionally_coherent_not_validated"
    )
    return {
        "workflow": "conditional_multi_llm_model_review",
        "trigger": "explicit_model_creation_or_modification_request",
        "primary": {"provider": primary_provider, "model": primary_credential.model},
        "proposal": proposal,
        "deterministic_checks": deterministic,
        "independent_reviews": reviews,
        "review_coverage": {
            "configured_provider_count": len(configured),
            "independent_review_count": len(reviews),
            "cross_review_complete": cross_review_complete,
            "issue_counts": issue_counts,
        },
        "status": status,
        "execution_allowed": False,
        "next_gate": (
            "Resolve every critical/major issue, implement the declared deterministic tests, validate on held-out or external data, "
            "and register a versioned executable adapter before the proposal can run."
        ),
        "claim_boundary": "LLM agreement is not mathematical proof, empirical validation or operational certification.",
    }
