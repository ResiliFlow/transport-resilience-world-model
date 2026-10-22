from __future__ import annotations

from typing import Any


def enrich_project_summary(source: dict[str, Any] | None) -> dict[str, Any]:
    """Add version-compatible capability flags without mutating stored metadata."""

    summary = {**(source or {})}
    capabilities = {**(summary.get("capabilities") or {})}
    metrics = summary.get("metrics") or {}
    quality = summary.get("quality") or {}

    mapped = summary.get("mapped_entity_count")
    if mapped is None:
        entity_count = int(summary.get("entity_count") or 0)
        invalid_coordinates = int(quality.get("invalid_coordinates") or 0)
        mapped = max(0, entity_count - invalid_coordinates) if summary.get("extent") else 0
        summary["mapped_entity_count"] = mapped
        summary["unmapped_entity_count"] = max(0, entity_count - mapped)

    has_time = len(summary.get("time_values") or []) > 1
    has_temporal_service = bool(
        metrics.get("speed", {}).get("available")
        or metrics.get("travel_time", {}).get("available")
    )
    has_any_metric = has_temporal_service or bool(metrics.get("primary", {}).get("available"))
    topology_status = capabilities.get("topology_criticality", "unavailable")
    has_topology = topology_status in {"available", "conditional"}
    capabilities.update(
        {
            "temporal_resilience": "conditional" if has_time and has_temporal_service else "unavailable",
            "spatial_hotspots": "available" if int(mapped) > 0 and has_any_metric else "unavailable",
            "progressive_robustness": topology_status if has_topology else "unavailable",
            "accessibility": "conditional" if has_topology else "unavailable",
            "traffic_assignment": (
                "available" if has_topology and all(metrics.get(name, {}).get("available") for name in ("flow", "capacity", "free_flow_time"))
                else "conditional" if has_topology else "unavailable"
            ),
            "resource_optimisation": (
                "available" if has_topology and all(metrics.get(name, {}).get("available") for name in ("repair_cost", "repair_duration"))
                else "conditional" if has_topology else "unavailable"
            ),
            "stochastic_reliability": (
                "available" if has_topology and metrics.get("failure_probability", {}).get("available")
                else "conditional" if has_topology else "unavailable"
            ),
            "disruption_propagation": topology_status if has_topology else "unavailable",
            "route_detour_analysis": topology_status if has_topology else "unavailable",
        }
    )
    summary["capabilities"] = capabilities
    return summary


def project_capability(project: dict[str, Any] | None, key: str) -> str:
    if not project:
        return "unavailable"
    return str(enrich_project_summary(project.get("summary"))["capabilities"].get(key, "unavailable"))
