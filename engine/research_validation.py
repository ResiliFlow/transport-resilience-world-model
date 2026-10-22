"""Reviewer-facing validation of the Transport World Model evidence loop.

This module is deliberately isolated from the six operational tools.  It
reads the current project, invokes the public model functions through the
existing scientific-validation harness, and records independent baselines,
reproducibility information and claim boundaries.  It never changes a saved
scenario, map selection, model preference, Copilot conversation or analysis
run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable
from uuid import uuid4

from .matsim_adapter import assess_matsim_readiness
from .model_router import recommend_models
from .scientific_validation import LITERATURE, validate_project


PROTOCOL = "twm.research-validation.v2"
FIXED_SEEDS = [104729, 130363, 155921]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "storage" / "research_validation"


class ResearchValidationCancelled(RuntimeError):
    """Raised when the owning background job is cancelled."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    return connection


def _cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise ResearchValidationCancelled("Research validation was cancelled before a report was saved.")


def _notify(progress: Callable[[float, str], None] | None, value: float, message: str) -> None:
    if progress:
        progress(value, message)


def _temporal_profile(database_path: Path, maximum_points: int = 180) -> dict[str, Any]:
    """Aggregate observed metrics by timestamp without loading the fact table."""

    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT timestamp_epoch,
                   AVG(speed) AS mean_speed,
                   AVG(travel_time) AS mean_travel_time,
                   AVG(primary_value) AS mean_primary,
                   COUNT(*) AS observations
            FROM observations
            WHERE timestamp_epoch IS NOT NULL
            GROUP BY timestamp_epoch
            ORDER BY timestamp_epoch
            """
        ).fetchall()
    if not rows:
        return {"available": False, "points": [], "sampling": "No timestamped observations were uploaded."}
    stride = max(1, (len(rows) + maximum_points - 1) // maximum_points)
    sampled = rows[::stride]
    if sampled[-1]["timestamp_epoch"] != rows[-1]["timestamp_epoch"]:
        sampled.append(rows[-1])
    points = [
        {
            "timestamp": int(row["timestamp_epoch"]),
            "mean_speed": float(row["mean_speed"]) if row["mean_speed"] is not None else None,
            "mean_travel_time": float(row["mean_travel_time"]) if row["mean_travel_time"] is not None else None,
            "mean_primary": float(row["mean_primary"]) if row["mean_primary"] is not None else None,
            "observations": int(row["observations"]),
        }
        for row in sampled
    ]
    return {
        "available": True,
        "points": points,
        "source_point_count": len(rows),
        "display_point_count": len(points),
        "sampling": "All time points" if stride == 1 else f"Every {stride}th time point plus the final point",
    }


def _external_portability_evidence() -> dict[str, Any]:
    path = PROJECT_ROOT / "validation" / "external_data_matrix_report.json"
    if not path.exists():
        return {
            "status": "not_tested",
            "case_count": 0,
            "boundary": "No packaged schema-portability report is available.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return {
        "status": payload.get("status", "not_tested"),
        "case_count": int(summary.get("total") or summary.get("case_count") or 0),
        "passed": int(summary.get("passed") or 0),
        "summary": summary,
        "boundary": (
            "Synthetic schema and hazard variants test technical import portability only. "
            "They are not empirical cross-region or cross-hazard external validation."
        ),
    }


def _source_fingerprint(project: dict[str, Any]) -> dict[str, Any]:
    summary = project.get("summary", {})
    files = [
        {"name": item.get("name"), "sha256": item.get("sha256"), "size_bytes": item.get("size_bytes")}
        for item in summary.get("source_files", [])
    ]
    data_hash = _digest(files or {
        "project_id": project.get("project_id"),
        "observations": summary.get("observation_count"),
        "entities": summary.get("entity_count"),
    })
    return {"data_sha256": data_hash, "source_files": files}


def _evidence_grade(*, strong: bool = False, limited: bool = False, testable: bool = True) -> str:
    if not testable:
        return "not_testable"
    if strong and not limited:
        return "strong"
    if strong or limited:
        return "limited"
    return "attention"


def _core_tool_readiness(summary: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = summary.get("capabilities", {})
    tools = [
        ("Find critical roads", "topology_criticality"),
        ("Find critical areas", "spatial_hotspots"),
        ("Plan recovery priorities", "restoration_screening"),
        ("Plan routes under disruption", "route_detour_analysis"),
        ("Test network resilience", "progressive_robustness"),
        ("Build a disruption scenario", "closure_scenario"),
    ]
    return [
        {"tool": label, "capability": key, "status": capabilities.get(key, "unavailable")}
        for label, key in tools
    ]


def _selected_model_tiers(summary: dict[str, Any]) -> dict[str, Any]:
    routed = recommend_models(summary)
    models = routed.get("models", [])
    full = [item for item in models if item.get("executable") and not item.get("missing_for_high_fidelity")]
    limited = [item for item in models if item.get("executable") and item.get("missing_for_high_fidelity")]
    unavailable = [item for item in models if not item.get("executable")]
    return {
        "registered": len(models),
        "executable": len(full) + len(limited),
        "highest_fidelity_selected": len(full),
        "limited_tier_selected": len(limited),
        "not_executable": len(unavailable),
        "models": [
            {
                "model_id": item.get("model_id"),
                "goal": item.get("goal"),
                "executable": bool(item.get("executable")),
                "selected_tier": item.get("selected_tier"),
                "highest_tier": item.get("high_fidelity_tier"),
                "missing_for_highest_tier": item.get("missing_for_high_fidelity", []),
                "assumption_level": item.get("assumption_level"),
            }
            for item in models
        ],
    }


def _stage_records(
    project: dict[str, Any],
    core: dict[str, Any],
    temporal: dict[str, Any],
    matsim: dict[str, Any],
    model_tiers: dict[str, Any],
    context: dict[str, Any],
    portability: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = project.get("summary", {})
    quality = summary.get("quality", {})
    checks = {item["id"]: item for item in core.get("checks", [])}
    mapped = int(summary.get("mapped_entity_count") or 0)
    entities = int(summary.get("entity_count") or 0)
    mapping_coverage = mapped / max(1, entities)
    quality_flags = sum(int(quality.get(key) or 0) for key in (
        "invalid_coordinates", "invalid_times", "negative_speeds_invalid", "duplicate_entity_times", "missing_entity_ids"
    ))
    represent_limited = summary.get("geometry_kind") not in {"line", "polyline", "exact_link_geometry"}
    full_models = int(model_tiers.get("highest_fidelity_selected") or 0)
    executable_models = int(model_tiers.get("executable") or 0)
    registered_models = int(model_tiers.get("registered") or 0)
    tasks = int(context.get("agent_task_count") or 0)
    completed_tasks = int(context.get("completed_agent_task_count") or 0)
    runs = int(context.get("saved_run_count") or 0)
    tool_readiness = _core_tool_readiness(summary)
    any_tool_limited = any(item["status"] != "available" for item in tool_readiness)

    return [
        {
            "id": "perceive", "index": 1, "label": "Perceive",
            "claim": "The platform can read, map and quality-check the uploaded observations without changing the source files.",
            "method": "Semantic field mapping, coordinate/time validation and SQL aggregation over the complete imported fact table.",
            "equation": "Cₘₐₚ = Nₘₐₚₚₑd / Nₜₒₜₐₗ",
            "grade": _evidence_grade(strong=mapping_coverage == 1.0 and quality_flags == 0),
            "metrics": [
                {"label": "Mapping coverage", "value": mapping_coverage, "format": "percent"},
                {"label": "Observations read", "value": int(summary.get("observation_count") or 0), "format": "integer"},
                {"label": "Time points", "value": int(summary.get("time_value_count") or 0), "format": "integer"},
                {"label": "Invalid core records", "value": quality_flags, "format": "integer"},
            ],
            "evidence": ["Source-file SHA-256", "mapped/unmapped counts", "coordinate and timestamp checks", "observed metric timeline"],
            "interpretation": "The imported project is fully traceable to the uploaded file set." if mapping_coverage == 1.0 else "Some uploaded entities are not geographically mapped; topology analyses may still remain available.",
            "boundary": "A clean import does not establish disaster causality, exact road geometry or observed travel demand.",
        },
        {
            "id": "represent", "index": 2, "label": "Represent",
            "claim": "The uploaded records are represented as a time-indexed transport network with disclosed geometry and impedance assumptions.",
            "method": "Directed From→To graph reconstruction and adaptive model-tier selection from actual mapped fields.",
            "equation": "Gₜ = (V, E, wₜ)",
            "grade": _evidence_grade(strong=int(core.get("summary", {}).get("network_edges") or 0) > 1, limited=represent_limited),
            "metrics": [
                {"label": "Network nodes", "value": int(core.get("summary", {}).get("network_nodes") or 0), "format": "integer"},
                {"label": "Directed links", "value": int(core.get("summary", {}).get("network_edges") or 0), "format": "integer"},
                {"label": "Executable model tiers", "value": executable_models, "format": "integer"},
                {"label": "Highest-fidelity tiers", "value": full_models, "format": "integer"},
            ],
            "evidence": [summary.get("geometry_note") or summary.get("geometry_kind"), f"{executable_models}/{registered_models} registered models executable"],
            "interpretation": "The network is routable and model readiness is derived from this project rather than an unrelated project template.",
            "boundary": "Representative link points do not constitute surveyed road polylines." if represent_limited else "Network geometry is used at its uploaded precision.",
        },
        {
            "id": "predict", "index": 3, "label": "Predict",
            "claim": "Counterfactual network response is numerically reproducible under explicit disruptions.",
            "method": "Independent stress-curve recomputation, shortest-path oracle checks and MATSim readiness assessment.",
            "equation": "R(k) = E(Gₖ) / E(G₀)",
            "grade": _evidence_grade(
                strong=bool(checks.get("stress-independent-oracle", {}).get("passed")) and bool(checks.get("route-independent-oracle", {}).get("passed")),
                limited=matsim.get("tier") != "demand_calibrated",
            ),
            "metrics": [
                {"label": "Stress oracle error", "value": float(checks.get("stress-independent-oracle", {}).get("value") or 0.0), "format": "scientific"},
                {"label": "Route oracle error", "value": float(checks.get("route-independent-oracle", {}).get("value") or 0.0), "format": "scientific"},
                {"label": "Timeline available", "value": 1 if temporal.get("available") else 0, "format": "binary"},
            ],
            "evidence": [f"MATSim tier: {matsim.get('label') or matsim.get('tier')}", "Dijkstra–Bellman–Ford numerical agreement", "independent robustness recomputation"],
            "interpretation": "Counterfactual calculations are deterministic for the registered project and protocol.",
            "boundary": matsim.get("explanation") or "Prediction is conditional on the uploaded network and stated scenario operators.",
        },
        {
            "id": "decide", "index": 4, "label": "Decide",
            "claim": "Road and recovery rankings are compared with transparent traditional baselines rather than presented in isolation.",
            "method": "Network-efficiency ranking versus NetworkX betweenness, degree ablation, fixed-seed random orders and recovery-order baselines.",
            "equation": "ΔEₑ = [E(G) − E(G∖e)] / E(G)",
            "grade": _evidence_grade(
                strong=bool(checks.get("critical-vs-baselines", {}).get("passed")) and bool(checks.get("recovery-vs-baselines", {}).get("passed")),
                limited=any(item.get("missing_for_highest_tier") for item in model_tiers.get("models", []) if item.get("model_id") in {"P5-M2", "P5-M4"}),
            ),
            "metrics": [
                {"label": "Critical-road objective", "value": float(checks.get("critical-vs-baselines", {}).get("value") or 0.0), "format": "decimal"},
                {"label": "Best baseline", "value": float(checks.get("critical-vs-baselines", {}).get("threshold") or 0.0), "format": "decimal"},
                {"label": "Recovery AUC", "value": float(checks.get("recovery-vs-baselines", {}).get("value") or 0.0), "format": "decimal"},
            ],
            "evidence": ["NetworkX baseline", "degree-only ablation", "fixed-seed random baseline", "reverse recovery order"],
            "interpretation": "The registered rankings meet their declared objective on this project and tested parameter set.",
            "boundary": "Agreement with an aligned network-efficiency objective is not proof of universal superiority, observed demand impact or cost-optimal recovery.",
        },
        {
            "id": "act", "index": 5, "label": "Act",
            "claim": "Validated outputs can be passed to the six existing decision tools and the Agent tool layer without duplicating model logic.",
            "method": "Read-only capability audit of tool contracts, saved results and completed Agent tasks.",
            "equation": "a* = arg maxₐ ΔS(a | x, θ)",
            "grade": _evidence_grade(
                strong=all(item["status"] != "unavailable" for item in tool_readiness),
                limited=any_tool_limited or completed_tasks == 0,
            ),
            "metrics": [
                {"label": "Core tools addressable", "value": sum(item["status"] != "unavailable" for item in tool_readiness), "format": "integer"},
                {"label": "Completed Agent tasks", "value": completed_tasks, "format": "integer"},
                {"label": "Saved operational results", "value": runs, "format": "integer"},
            ],
            "evidence": ["Existing six-tool API contracts", f"{tasks} recorded Agent tasks", f"{runs} saved operational analyses"],
            "interpretation": "Research validation audits the existing action layer; it does not create a seventh operational tool.",
            "boundary": "Provider-dependent LLM language quality is not certified by this deterministic report.",
        },
        {
            "id": "feedback", "index": 6, "label": "Feedback",
            "claim": "Results remain inspectable under repetition, alternative baselines and a larger ranking depth.",
            "method": "Deterministic repeat run, Top-K prefix sensitivity, ablation and independent numerical oracles.",
            "equation": "J(Aₖ, A₂ₖ) = |Aₖ ∩ A₂ₖ| / |Aₖ ∪ A₂ₖ|",
            "grade": _evidence_grade(strong=bool(checks.get("repeatability", {}).get("passed")) and bool(checks.get("top-k-sensitivity", {}).get("passed"))),
            "metrics": [
                {"label": "Repeatability", "value": float(checks.get("repeatability", {}).get("value") or 0.0), "format": "percent"},
                {"label": "Top-K stability", "value": float(checks.get("top-k-sensitivity", {}).get("value") or 0.0), "format": "percent"},
                {"label": "Synthetic schema cases", "value": int(portability.get("case_count") or 0), "format": "integer"},
            ],
            "evidence": ["second-run SHA-256 equality", "Top-K sensitivity", "traditional baselines", "technical schema portability matrix"],
            "interpretation": "Numerical repeatability and local parameter stability are explicit rather than assumed.",
            "boundary": portability.get("boundary"),
        },
        {
            "id": "memory", "index": 7, "label": "Memory",
            "claim": "The evidence package can be reproduced from immutable source hashes, protocol settings and software identity.",
            "method": "Project provenance plus a deterministic report fingerprint over data, protocol, parameters and fixed seeds.",
            "equation": "Hᵣ = SHA-256(Hdₐₜₐ ∥ Hₘₒdₑₗ ∥ θ ∥ s)",
            "grade": _evidence_grade(strong=bool(summary.get("source_files"))),
            "metrics": [
                {"label": "Source files hashed", "value": sum(bool(item.get("sha256")) for item in summary.get("source_files", [])), "format": "integer"},
                {"label": "Fixed seeds recorded", "value": len(FIXED_SEEDS), "format": "integer"},
                {"label": "Saved results available", "value": runs, "format": "integer"},
            ],
            "evidence": ["data SHA-256", "software version", "protocol version", "parameters", "fixed validation seeds"],
            "interpretation": "The same registered inputs and protocol produce the same report fingerprint.",
            "boundary": "Reproducibility does not by itself establish external validity or causal disaster attribution.",
        },
    ]


def _world_model_stage_records(
    project: dict[str, Any],
    core: dict[str, Any],
    temporal: dict[str, Any],
    matsim: dict[str, Any],
    model_tiers: dict[str, Any],
    context: dict[str, Any],
    portability: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the reviewer trace in the platform's canonical eight-stage TWM order."""

    summary = project.get("summary", {})
    quality = summary.get("quality", {})
    checks = {item["id"]: item for item in core.get("checks", [])}
    mapped = int(summary.get("mapped_entity_count") or 0)
    entities = int(summary.get("entity_count") or 0)
    mapping_coverage = mapped / max(1, entities)
    quality_flags = sum(int(quality.get(key) or 0) for key in (
        "invalid_coordinates", "invalid_times", "negative_speeds_invalid", "duplicate_entity_times", "missing_entity_ids"
    ))
    exact_geometry = summary.get("geometry_kind") in {"line", "polyline", "exact_link_geometry"}
    observed_demand = int(matsim.get("od_rows") or 0) > 0
    full_models = int(model_tiers.get("highest_fidelity_selected") or 0)
    executable_models = int(model_tiers.get("executable") or 0)
    tasks = int(context.get("agent_task_count") or 0)
    completed_tasks = int(context.get("completed_agent_task_count") or 0)
    runs = int(context.get("saved_run_count") or 0)
    tool_readiness = _core_tool_readiness(summary)
    any_tool_limited = any(item["status"] != "available" for item in tool_readiness)
    passed_gates = int(core.get("summary", {}).get("passed") or 0)
    total_gates = int(core.get("summary", {}).get("total") or 0)
    stress_error = float(checks.get("stress-independent-oracle", {}).get("value") or 0.0)
    route_error = float(checks.get("route-independent-oracle", {}).get("value") or 0.0)
    planning_model_ids = {
        "TWM-METHOD-2", "TWM-METHOD-4", "TWM-METHOD-10", "TWM-METHOD-23",
        "P5-M2", "P5-M4", "P5-M10", "P5-M23",
    }
    planning_limited = any(
        item.get("missing_for_highest_tier")
        for item in model_tiers.get("models", [])
        if item.get("model_id") in planning_model_ids
    )
    matsim_can_run = bool(matsim.get("can_run"))
    matsim_full = matsim.get("tier") == "demand_calibrated"

    return [
        {
            "id": "perception", "index": 1, "label": "Perception",
            "claim": "Uploaded observations are converted into a quality-checked, time-indexed transport state; demand is included only when it is actually supplied.",
            "method": "Semantic field mapping, coordinate and time validation, directed network reconstruction and an explicit demand-evidence check.",
            "equation": "xₜ = (Gₜ, qₜ, dₜ),  Gₜ = (V, E, wₜ)",
            "grade": _evidence_grade(
                strong=mapping_coverage == 1.0 and quality_flags == 0 and exact_geometry and observed_demand,
                limited=mapping_coverage == 1.0 and quality_flags == 0 and (not exact_geometry or not observed_demand),
            ),
            "metrics": [
                {"label": "Mapping coverage", "value": mapping_coverage, "format": "percent"},
                {"label": "Observations read", "value": int(summary.get("observation_count") or 0), "format": "integer"},
                {"label": "Directed links", "value": int(core.get("summary", {}).get("network_edges") or 0), "format": "integer"},
                {"label": "Observed OD records", "value": int(matsim.get("od_rows") or 0), "format": "integer"},
            ],
            "evidence": [
                "Source-file SHA-256 and semantic field mappings",
                "coordinate, timestamp and entity checks",
                summary.get("geometry_note") or summary.get("geometry_kind"),
                f"Observed OD demand: {'available' if observed_demand else 'not uploaded'}",
            ],
            "interpretation": "The project state is traceable and analysis-ready at the precision of the uploaded network and demand evidence.",
            "boundary": "Representative points are not exact road polylines, and absent OD demand is not inferred from speeds, travel times or filenames.",
        },
        {
            "id": "prediction", "index": 2, "label": "Prediction",
            "claim": "Future or counterfactual network impacts are estimated under explicit disruption assumptions rather than presented as observed outcomes.",
            "method": "Registered response models, progressive network stress tests and shortest-path impact calculations over the uploaded time-dependent impedance.",
            "equation": "x̂ₜ₊τ = fθ(xₜ, uₜ),  R(k) = E(Gₖ) / E(G₀)",
            "grade": _evidence_grade(
                strong=bool(checks.get("stress-independent-oracle", {}).get("passed")) and bool(checks.get("route-independent-oracle", {}).get("passed")) and observed_demand,
                limited=bool(checks.get("stress-independent-oracle", {}).get("passed")) and bool(checks.get("route-independent-oracle", {}).get("passed")) and not observed_demand,
            ),
            "metrics": [
                {"label": "Stress oracle error", "value": stress_error, "format": "scientific"},
                {"label": "Route oracle error", "value": route_error, "format": "scientific"},
                {"label": "Time points", "value": int(summary.get("time_value_count") or 0), "format": "integer"},
            ],
            "evidence": ["independent robustness recomputation", "Dijkstra–Bellman–Ford numerical agreement", "uploaded performance timeline"],
            "interpretation": "The registered counterfactual response is numerically stable for the current project and declared scenario.",
            "boundary": "Without observed OD demand or external event labels, predictions are network-response estimates rather than calibrated behavioural forecasts or causal disaster attribution.",
        },
        {
            "id": "planning", "index": 3, "label": "Planning",
            "claim": "Critical-road, critical-area, recovery and route alternatives are produced with the strongest registered tier supported by the upload.",
            "method": "TWM planning objectives are compared with NetworkX betweenness, degree ablation, fixed-seed random orders and recovery-order baselines.",
            "equation": "π* = arg maxπ S(π | x̂, θ)",
            "grade": _evidence_grade(
                strong=bool(checks.get("critical-vs-baselines", {}).get("passed")) and bool(checks.get("recovery-vs-baselines", {}).get("passed")) and not planning_limited,
                limited=bool(checks.get("critical-vs-baselines", {}).get("passed")) and bool(checks.get("recovery-vs-baselines", {}).get("passed")) and planning_limited,
            ),
            "metrics": [
                {"label": "Executable model tiers", "value": executable_models, "format": "integer"},
                {"label": "Highest-fidelity tiers", "value": full_models, "format": "integer"},
                {"label": "Critical-road objective", "value": float(checks.get("critical-vs-baselines", {}).get("value") or 0.0), "format": "decimal"},
            ],
            "evidence": ["registered model-tier routing", "NetworkX baseline", "degree-only ablation", "fixed-seed random baseline", "reverse recovery order"],
            "interpretation": "Planning outputs are selected from the current project's actual data support rather than a fixed template.",
            "boundary": "A structurally valid plan is not automatically demand-weighted, cost-optimal or operationally safe when demand, repair-resource or capacity evidence is absent.",
        },
        {
            "id": "verification", "index": 4, "label": "Verification",
            "claim": "Registered outputs are checked against independent numerical oracles, transparent sources and exact repeat execution before they enter the evidence loop.",
            "method": "Dijkstra–Bellman–Ford comparison, independent stress recomputation, baseline and ablation tests, deterministic repetition and registered literature traceability.",
            "equation": "ε∞ = ‖yTWM − yoracle‖∞",
            "grade": _evidence_grade(strong=total_gates > 0 and passed_gates == total_gates),
            "metrics": [
                {"label": "Independent gates passed", "value": passed_gates, "format": "integer"},
                {"label": "Independent gates tested", "value": total_gates, "format": "integer"},
                {"label": "Largest oracle error", "value": max(stress_error, route_error), "format": "scientific"},
            ],
            "evidence": ["independent numerical oracles", "repeatability hash", "ablation and sensitivity checks", "registered model sources"],
            "interpretation": "The report separates verified numerical agreement from evidence that remains data-limited.",
            "boundary": "Verification establishes implementation consistency for the registered protocol; an LLM opinion is not treated as ground truth, and this is not external empirical validation.",
        },
        {
            "id": "simulation", "index": 5, "label": "Simulation",
            "claim": "Disruption scenarios are propagated through the available dynamic engine at the highest simulation tier supported by the uploaded network and demand.",
            "method": "MATSim readiness assessment plus registered closure, slowdown and travel-time scenario operators; demand-calibrated and exploratory tiers remain explicitly separated.",
            "equation": "τᵢ(s) = ∑ₑ∈pᵢ(s) tₑ(s)",
            "grade": _evidence_grade(strong=matsim_can_run and matsim_full, limited=matsim_can_run and not matsim_full, testable=matsim_can_run),
            "metrics": [
                {"label": "Routable links", "value": int(matsim.get("topology_links") or 0), "format": "integer"},
                {"label": "Observed OD records", "value": int(matsim.get("od_rows") or 0), "format": "integer"},
                {"label": "Capacity coverage", "value": float(matsim.get("capacity_coverage") or 0.0), "format": "percent"},
            ],
            "evidence": [f"MATSim tier: {matsim.get('label') or matsim.get('tier')}", "Java worker readiness", "network impedance coverage", "scenario operator registry"],
            "interpretation": "The simulation engine is exposed at its declared data-supported tier, without silently upgrading exploratory probes into observed traffic demand.",
            "boundary": matsim.get("explanation") or "Simulation is conditional on the uploaded network and stated scenario operators.",
        },
        {
            "id": "decision", "index": 6, "label": "Decision",
            "claim": "Verified predictions, plans and simulations can be compared through the six existing decision tools without duplicating their model logic.",
            "method": "Read-only audit of core-tool capability contracts, action-comparison outputs, saved analyses and completed Agent tasks.",
            "equation": "a* = arg maxₐ ΔS(a | x, θ)",
            "grade": _evidence_grade(
                strong=all(item["status"] == "available" for item in tool_readiness) and completed_tasks > 0,
                limited=all(item["status"] != "unavailable" for item in tool_readiness) and (any_tool_limited or completed_tasks == 0),
                testable=any(item["status"] != "unavailable" for item in tool_readiness),
            ),
            "metrics": [
                {"label": "Core tools addressable", "value": sum(item["status"] != "unavailable" for item in tool_readiness), "format": "integer"},
                {"label": "Completed Agent tasks", "value": completed_tasks, "format": "integer"},
                {"label": "Saved operational results", "value": runs, "format": "integer"},
            ],
            "evidence": ["existing six-tool API contracts", f"{tasks} recorded Agent tasks", f"{runs} saved operational analyses", "registered action-comparison outputs"],
            "interpretation": "The validation layer audits how evidence supports decisions; it does not create or replace an operational tool.",
            "boundary": "A ranked action remains decision support, not an automatic safety approval or proof of realised benefit.",
        },
        {
            "id": "feedback", "index": 7, "label": "Feedback",
            "claim": "Results remain open to revision through repeated runs, larger ranking depths, alternative baselines and explicit sensitivity checks.",
            "method": "Deterministic repetition, Top-K prefix sensitivity, ablation and technical schema-portability checks feed evidence back into model interpretation.",
            "equation": "J(Aₖ, A₂ₖ) = |Aₖ ∩ A₂ₖ| / |Aₖ ∪ A₂ₖ|",
            "grade": _evidence_grade(strong=bool(checks.get("repeatability", {}).get("passed")) and bool(checks.get("top-k-sensitivity", {}).get("passed"))),
            "metrics": [
                {"label": "Repeatability", "value": float(checks.get("repeatability", {}).get("value") or 0.0), "format": "percent"},
                {"label": "Top-K stability", "value": float(checks.get("top-k-sensitivity", {}).get("value") or 0.0), "format": "percent"},
                {"label": "Synthetic schema cases", "value": int(portability.get("case_count") or 0), "format": "integer"},
            ],
            "evidence": ["second-run SHA-256 equality", "Top-K sensitivity", "traditional baselines", "technical schema portability matrix"],
            "interpretation": "New evidence can revise the interpretation without overwriting the original trace.",
            "boundary": portability.get("boundary"),
        },
        {
            "id": "memory", "index": 8, "label": "Memory",
            "claim": "The evidence package can be reproduced from immutable source hashes, protocol settings and software identity.",
            "method": "Project provenance plus a deterministic report fingerprint over data, protocol, parameters and fixed seeds.",
            "equation": "Hᵣ = SHA-256(Hdata ∥ Hmodel ∥ θ ∥ s)",
            "grade": _evidence_grade(strong=bool(summary.get("source_files"))),
            "metrics": [
                {"label": "Source files hashed", "value": sum(bool(item.get("sha256")) for item in summary.get("source_files", [])), "format": "integer"},
                {"label": "Fixed seeds recorded", "value": len(FIXED_SEEDS), "format": "integer"},
                {"label": "Saved results available", "value": runs, "format": "integer"},
            ],
            "evidence": ["data SHA-256", "software version", "protocol version", "parameters", "fixed validation seeds"],
            "interpretation": "Cases, evidence traces and reproduction identities remain available for the next feedback cycle.",
            "boundary": "Reproducibility does not by itself establish external validity or causal disaster attribution.",
        },
    ]


def _chart_data(core: dict[str, Any], temporal: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = core.get("comparisons", {})
    critical = comparisons.get("critical_roads", {}).get("curves", {})
    recovery = comparisons.get("recovery_priorities", {}).get("curves", {})
    stress = comparisons.get("network_resilience", {})
    grade_score = {"strong": 1.0, "limited": 0.67, "attention": 0.33, "not_testable": 0.0}
    return {
        "timeline": temporal,
        "critical_baselines": [
            {"label": "TWM", "value": float(critical.get("platform", {}).get("mean_individual_loss") or 0.0)},
            {"label": "Betweenness", "value": float(critical.get("networkx_edge_betweenness", {}).get("mean_individual_loss") or 0.0)},
            {"label": "Degree", "value": float(critical.get("degree_only_ablation", {}).get("mean_individual_loss") or 0.0)},
            {"label": "Random", "value": float(critical.get("fixed_seed_random_mean", {}).get("mean_individual_loss") or 0.0)},
        ],
        "recovery_baselines": [
            {"label": "Adaptive TWM", "value": float(recovery.get("platform", {}).get("auc") or 0.0)},
            {"label": "Static order", "value": float(recovery.get("static_criticality_order", {}).get("auc") or 0.0)},
            {"label": "Reverse", "value": float(recovery.get("reverse_criticality_ablation", {}).get("auc") or 0.0)},
            {"label": "Random", "value": float(recovery.get("fixed_seed_random_mean", {}).get("auc") or 0.0)},
        ],
        "stress": {
            "platform": [float(value) for value in stress.get("platform_curve", [])],
            "independent": [float(value) for value in stress.get("independent_curve", [])],
        },
        "twm_loop": [
            {"label": stage["label"], "grade": stage["grade"], "score": grade_score.get(stage["grade"], 0.0)}
            for stage in stages
        ],
    }


def run_research_validation(
    project: dict[str, Any],
    *,
    software_version: str,
    model_registry_version: str,
    execution_context: dict[str, Any] | None = None,
    profile: str = "standard",
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the registered reviewer protocol and persist no operational state."""

    if profile != "standard":
        raise ValueError("Research validation currently uses one registered standard protocol.")
    started = time.perf_counter()
    summary = project.get("summary", {})
    database_path = Path(project["database_path"])
    if not database_path.exists():
        raise ValueError("The current project's imported data store is unavailable.")
    _notify(progress, 4, "Freezing the data, model and protocol identity")
    _cancelled(cancelled)
    source = _source_fingerprint(project)
    parameters = {
        "profile": profile,
        "critical_top_k": 10,
        "sensitivity_top_k": 20,
        "random_baseline_seeds": FIXED_SEEDS,
        "map_scope": "whole uploaded project",
        "map_selection_ignored": True,
    }
    input_fingerprint = _digest({
        "data": source["data_sha256"],
        "software_version": software_version,
        "model_registry_version": model_registry_version,
        "protocol": PROTOCOL,
        "parameters": parameters,
    })

    _notify(progress, 10, "Reading the complete uploaded history")
    temporal = _temporal_profile(database_path)
    _cancelled(cancelled)
    model_tiers = _selected_model_tiers(summary)
    matsim = assess_matsim_readiness(project)
    portability = _external_portability_evidence()

    _notify(progress, 18, "Running independent baselines, ablations and numerical oracles")
    core = validate_project(database_path, top_k=parameters["critical_top_k"])
    _cancelled(cancelled)
    _notify(progress, 76, "Building the TWM evidence loop and claim boundaries")
    context = execution_context or {}
    stages = _world_model_stage_records(project, core, temporal, matsim, model_tiers, context, portability)
    charts = _chart_data(core, temporal, stages)
    strong_count = sum(stage["grade"] == "strong" for stage in stages)
    limited_count = sum(stage["grade"] == "limited" for stage in stages)
    attention_count = sum(stage["grade"] == "attention" for stage in stages)
    not_testable_count = sum(stage["grade"] == "not_testable" for stage in stages)

    _notify(progress, 90, "Creating the reproducibility receipt and paper exports")
    report_id = f"rv_{uuid4().hex[:14]}"
    report = {
        "report_id": report_id,
        "protocol": PROTOCOL,
        "profile": profile,
        "generated_at": _now(),
        "project_id": project.get("project_id"),
        "project_name": project.get("name"),
        "hazard_type": project.get("hazard_type"),
        "status": "evidence_available" if attention_count == 0 else "attention_required",
        "summary": {
            "title": "Transport World Model research validation",
            "plain_language": (
                "The complete registered TWM evidence loop was evaluated with explicit baselines, sensitivity, reproducibility and claim boundaries."
                if attention_count == 0 else
                "The TWM evidence loop was evaluated, and at least one check requires attention before a publication claim is made."
            ),
            "strong_stages": strong_count,
            "limited_stages": limited_count,
            "attention_stages": attention_count,
            "not_testable_stages": not_testable_count,
            "stage_count": len(stages),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "network_nodes": int(core.get("summary", {}).get("network_nodes") or 0),
            "network_edges": int(core.get("summary", {}).get("network_edges") or 0),
            "observations": int(summary.get("observation_count") or 0),
        },
        "stages": stages,
        "charts": charts,
        "core_scientific_validation": core,
        "model_tiers": model_tiers,
        "core_tool_readiness": _core_tool_readiness(summary),
        "matsim_readiness": matsim,
        "technical_portability": portability,
        "literature": LITERATURE,
        "reproducibility": {
            "data_sha256": source["data_sha256"],
            "source_files": source["source_files"],
            "software_version": software_version,
            "model_registry_version": model_registry_version,
            "protocol": PROTOCOL,
            "parameters": parameters,
            "fixed_seeds": FIXED_SEEDS,
            "input_fingerprint_sha256": input_fingerprint,
        },
        "claim_boundary": (
            "This report establishes project-specific numerical correctness, repeatability, baseline comparison and evidence traceability. "
            "It does not establish causal disaster attribution, universal model superiority, observed demand where OD/flow data are absent, "
            "live operational safety, or empirical external validity beyond the uploaded case."
        ),
    }
    # The scientific evidence digest deliberately excludes report identity,
    # wall-clock runtime and generated timestamps.  Two runs with identical
    # data, model versions, parameters and fixed seeds therefore reproduce the
    # same digest even though each saved report has its own ID.
    stable_core = {
        "status": core.get("status"),
        "checks": core.get("checks"),
        "comparisons": core.get("comparisons"),
        "sensitivity_and_ablation": core.get("sensitivity_and_ablation"),
        "repeatability": core.get("repeatability"),
    }
    report["reproducibility"]["report_sha256"] = _digest({
        "input_fingerprint_sha256": input_fingerprint,
        "stages": stages,
        "charts": charts,
        "core_scientific_validation": stable_core,
        "model_tiers": model_tiers,
        "core_tool_readiness": report["core_tool_readiness"],
        "technical_portability": portability,
        "claim_boundary": report["claim_boundary"],
    })
    _cancelled(cancelled)
    _notify(progress, 98, "Saving the independent research evidence package")
    save_research_report(report)
    return report


def _report_dir(project_id: str) -> Path:
    safe = "".join(character for character in project_id if character.isalnum() or character in "_-.")
    if not safe:
        raise ValueError("A readable project ID is required for research validation.")
    return REPORT_ROOT / safe


def save_research_report(report: dict[str, Any]) -> Path:
    folder = _report_dir(str(report["project_id"]))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{report['report_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_research_reports(project_id: str) -> list[dict[str, Any]]:
    folder = _report_dir(project_id)
    if not folder.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(folder.glob("rv_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports.append({
            "report_id": payload.get("report_id"),
            "project_id": payload.get("project_id"),
            "project_name": payload.get("project_name"),
            "generated_at": payload.get("generated_at"),
            "status": payload.get("status"),
            "profile": payload.get("profile"),
            "summary": payload.get("summary", {}),
            "report_sha256": payload.get("reproducibility", {}).get("report_sha256"),
        })
    return reports


def load_research_report(report_id: str) -> dict[str, Any] | None:
    if not report_id.startswith("rv_") or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in report_id):
        return None
    if not REPORT_ROOT.exists():
        return None
    for path in REPORT_ROOT.glob(f"*/{report_id}.json"):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def report_csv(report: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["TWM stage", "Evidence grade", "Claim", "Method", "Metric", "Value", "Boundary"])
    for stage in report.get("stages", []):
        metrics = stage.get("metrics") or [{"label": "", "value": ""}]
        for metric in metrics:
            writer.writerow([
                stage.get("label"), stage.get("grade"), stage.get("claim"), stage.get("method"),
                metric.get("label"), metric.get("value"), stage.get("boundary"),
            ])
    return output.getvalue()


def report_latex(report: dict[str, Any]) -> str:
    def escape(value: Any) -> str:
        text = str(value or "")
        for source, target in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            text = text.replace(source, target)
        return text

    rows = "\n".join(
        (
            f"{escape(stage.get('label'))} & {escape(stage.get('grade'))} & "
            f"{escape(stage.get('claim'))} & {escape(stage.get('boundary'))} \\\\"
        )
        for stage in report.get("stages", [])
    )
    return "\n".join([
        r"% Requires \usepackage{booktabs,tabularx}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Transport World Model research-validation evidence for the uploaded project.}",
        r"\label{tab:twm-research-validation}",
        r"\begin{tabularx}{\linewidth}{@{}l l X X@{}}",
        r"\toprule",
        r"Stage & Evidence grade & Registered claim & Scientific boundary \\",
        r"\midrule",
        rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        f"% Protocol: {escape(report.get('protocol'))}",
        f"% Report SHA-256: {escape(report.get('reproducibility', {}).get('report_sha256'))}",
    ])
