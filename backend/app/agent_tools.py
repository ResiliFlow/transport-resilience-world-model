from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from .api.schemas import (
    AgentAnalysisToolInput,
    AgentDataInspectionInput,
    AgentMatsimToolInput,
    AgentScenarioToolInput,
    AgentToolId,
    AnalysisRequest,
    MatsimRunRequest,
    ScenarioUpsertRequest,
)


AGENT_TOOL_PROTOCOL_VERSION = "1.0.0"


ANALYSIS_TOOL_TYPES: dict[str, str] = {
    "find_critical_roads": "critical_components",
    "find_critical_areas": "spatial_hotspots",
    "plan_recovery_priorities": "restoration_screening",
    "plan_routes_under_disruption": "route_detour_analysis",
    "test_network_resilience": "progressive_robustness",
}


_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "tool_id": "inspect_project_data",
        "title": "Inspect project data",
        "category": "data",
        "input_model": AgentDataInspectionInput,
        "output_kind": "project_profile",
        "saves_to_memory": False,
        "requires": ["project"],
    },
    {
        "tool_id": "find_critical_roads",
        "title": "Find critical roads",
        "category": "core_analysis",
        "input_model": AgentAnalysisToolInput,
        "output_kind": "analysis",
        "saves_to_memory": True,
        "requires": ["project", "routable_topology"],
    },
    {
        "tool_id": "find_critical_areas",
        "title": "Find critical areas",
        "category": "core_analysis",
        "input_model": AgentAnalysisToolInput,
        "output_kind": "analysis",
        "saves_to_memory": True,
        "requires": ["project", "mapped_entities"],
    },
    {
        "tool_id": "plan_recovery_priorities",
        "title": "Plan recovery priorities",
        "category": "core_analysis",
        "input_model": AgentAnalysisToolInput,
        "output_kind": "analysis",
        "saves_to_memory": True,
        "requires": ["project", "disrupted_roads_or_areas"],
    },
    {
        "tool_id": "plan_routes_under_disruption",
        "title": "Plan routes under disruption",
        "category": "core_analysis",
        "input_model": AgentAnalysisToolInput,
        "output_kind": "analysis",
        "saves_to_memory": True,
        "requires": ["project", "two_routable_endpoints"],
    },
    {
        "tool_id": "test_network_resilience",
        "title": "Test network resilience",
        "category": "core_analysis",
        "input_model": AgentAnalysisToolInput,
        "output_kind": "analysis",
        "saves_to_memory": True,
        "requires": ["project", "routable_topology"],
    },
    {
        "tool_id": "build_disruption_scenario",
        "title": "Build a disruption scenario",
        "category": "core_scenario",
        "input_model": AgentScenarioToolInput,
        "output_kind": "scenario",
        "saves_to_memory": True,
        "requires": ["project", "grounded_road_changes"],
    },
    {
        "tool_id": "run_matsim",
        "title": "Run MATSim",
        "category": "simulation",
        "input_model": AgentMatsimToolInput,
        "output_kind": "matsim",
        "saves_to_memory": True,
        "requires": ["project", "saved_project_scenario", "matsim_readiness"],
    },
)


def _definition(tool_id: str) -> dict[str, Any]:
    for definition in _TOOL_DEFINITIONS:
        if definition["tool_id"] == tool_id:
            return definition
    raise ValueError(f"Unknown Agent tool: {tool_id}.")


def agent_tool_catalog() -> dict[str, Any]:
    """Return a machine-readable catalogue without exposing Python objects."""

    tools: list[dict[str, Any]] = []
    for definition in _TOOL_DEFINITIONS:
        item = {key: deepcopy(value) for key, value in definition.items() if key != "input_model"}
        model: type[BaseModel] = definition["input_model"]
        item.update(
            {
                "input_schema": model.model_json_schema(),
                "task_contract": {
                    "status": ["queued", "running", "completed", "failed", "cancelled"],
                    "progress": "0-100 with stage and practitioner-readable message",
                    "cancellation": "cooperative; cancelled output is never persisted as completed",
                    "error": ["code", "message", "stage", "retryable", "details"],
                },
            }
        )
        tools.append(item)
    return {
        "protocol_version": AGENT_TOOL_PROTOCOL_VERSION,
        "tools": tools,
        "core_tool_count": 6,
        "tool_count": len(tools),
    }


def validate_agent_tool_input(tool_id: AgentToolId | str, raw_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise one tool input without executing any calculation."""

    definition = _definition(str(tool_id))
    model: type[BaseModel] = definition["input_model"]
    try:
        return model.model_validate(raw_input or {}).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Agent tool input is invalid: {exc}") from exc


def build_analysis_payload(
    *,
    project_id: str,
    tool_id: AgentToolId | str,
    normalised_input: dict[str, Any],
) -> dict[str, Any]:
    """Adapt an Agent core-tool input to the existing deterministic engine contract."""

    analysis_type = ANALYSIS_TOOL_TYPES.get(str(tool_id))
    if analysis_type is None:
        raise ValueError(f"{tool_id} is not an analysis-based Agent tool.")
    payload = {
        "project_id": project_id,
        "analysis_type": analysis_type,
        **normalised_input,
    }
    return AnalysisRequest.model_validate(payload).model_dump()


def build_scenario_request(
    *, project_id: str, normalised_input: dict[str, Any]
) -> ScenarioUpsertRequest:
    return ScenarioUpsertRequest.model_validate({"project_id": project_id, **normalised_input})


def build_matsim_request(normalised_input: dict[str, Any]) -> tuple[str, MatsimRunRequest]:
    scenario_id = str(normalised_input["scenario_id"])
    request = MatsimRunRequest.model_validate(
        {key: value for key, value in normalised_input.items() if key != "scenario_id"}
    )
    return scenario_id, request


def structured_agent_error(
    exc: Exception,
    *,
    stage: str,
    code: str = "tool_execution_failed",
    retryable: bool = False,
) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    return {
        "code": code,
        "message": message,
        "stage": stage,
        "retryable": bool(retryable),
        "details": {"exception_type": exc.__class__.__name__},
    }
