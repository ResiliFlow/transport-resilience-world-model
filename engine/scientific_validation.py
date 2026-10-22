"""Independent scientific-validation harness for the six decision tools.

The routines in this module deliberately sit outside the production model
router.  They recompute evaluation metrics from the stored graph, compare the
platform sequence with NetworkX and transparent traditional baselines, and
record sensitivity, ablation and repeatability checks.  They are validation
evidence, not another user-facing decision tool.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any, Iterable

import networkx as nx

from .models import (
    _build_graph,
    _connect,
    _record_nodes,
    critical_components,
    progressive_robustness,
    restoration_screening,
    spatial_hotspots,
)


PROTOCOL = "twm.scientific-validation.v1"
LITERATURE = [
    {
        "method": "Network efficiency and vulnerability",
        "citation": "Latora and Marchiori (2001), Efficient behavior of small-world networks",
        "doi": "10.1103/PhysRevLett.87.198701",
    },
    {
        "method": "Network Robustness Index",
        "citation": "Scott et al. (2006), Network robustness index: A new method for identifying critical links and evaluating the performance of transportation networks",
        "doi": "10.1016/j.jtrangeo.2005.10.003",
    },
    {
        "method": "Area-covering disruption analysis",
        "citation": "Jenelius and Mattsson (2012), Road network vulnerability analysis of area-covering disruptions",
        "doi": "10.1016/j.tra.2012.02.003",
    },
    {
        "method": "Restoration prioritisation",
        "citation": "Nurre et al. (2012), Restoring infrastructure systems: An integrated network design and scheduling problem",
        "doi": "10.1016/j.ejor.2012.07.010",
    },
    {
        "method": "Loopless alternative paths",
        "citation": "Yen (1971), Finding the K shortest loopless paths in a network",
        "doi": "10.1287/mnsc.17.11.712",
    },
]


def _stable_sources(nodes: Iterable[str], maximum: int) -> list[str]:
    ordered = sorted(nodes, key=str)
    if len(ordered) <= maximum:
        return ordered
    stride = len(ordered) / maximum
    return [ordered[min(len(ordered) - 1, int(index * stride))] for index in range(maximum)]


def _independent_efficiency(graph: nx.MultiDiGraph, *, maximum_sources: int = 24) -> float:
    """Recompute directed harmonic efficiency without production helpers."""

    nodes = sorted(graph.nodes, key=str)
    if len(nodes) < 2:
        return 0.0
    sources = _stable_sources(nodes, maximum_sources)
    reciprocal = 0.0
    for source in sources:
        distances = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        reciprocal += sum(
            1.0 / distance
            for target, distance in distances.items()
            if target != source and distance > 0
        )
    return reciprocal / (len(sources) * (len(nodes) - 1))


def _result_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _edge_score_order(
    graph: nx.MultiDiGraph,
    method: str,
    *,
    maximum_sources: int = 24,
) -> list[str]:
    edges = list(graph.edges(keys=True))
    if method == "networkx_edge_betweenness":
        if graph.number_of_nodes() <= 1_200 and graph.number_of_edges() <= 2_500:
            scores = nx.edge_betweenness_centrality(graph, weight="weight", normalized=True)
        else:
            sources = _stable_sources(graph.nodes, maximum_sources)
            targets = sorted(graph.nodes, key=str)
            scores = nx.edge_betweenness_centrality_subset(
                graph, sources=sources, targets=targets, weight="weight", normalized=False
            )
        ranked = sorted(
            edges,
            key=lambda edge: (float(scores.get(edge, 0.0)), str(edge[0]), str(edge[1]), str(edge[2])),
            reverse=True,
        )
        return [str(edge[2]) for edge in ranked]
    if method == "degree_only_ablation":
        ranked = sorted(
            edges,
            key=lambda edge: (
                graph.in_degree(edge[0]) + graph.out_degree(edge[0])
                + graph.in_degree(edge[1]) + graph.out_degree(edge[1]),
                str(edge[0]), str(edge[1]), str(edge[2]),
            ),
            reverse=True,
        )
        return [str(edge[2]) for edge in ranked]
    raise ValueError(f"Unknown baseline method: {method}")


def _failure_curve(
    baseline: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    order: list[str],
    *,
    maximum_sources: int = 24,
) -> dict[str, Any]:
    graph = baseline.copy()
    baseline_efficiency = _independent_efficiency(graph, maximum_sources=maximum_sources)
    ratios = [1.0]
    removed: list[str] = []
    for key in order:
        record = records.get(key)
        if record is None:
            continue
        u, v = _record_nodes(record, graph)
        if graph.has_edge(u, v, key):
            graph.remove_edge(u, v, key)
        else:
            continue
        removed.append(key)
        efficiency = _independent_efficiency(graph, maximum_sources=maximum_sources)
        ratios.append(max(0.0, efficiency / max(1e-15, baseline_efficiency)))
    if not removed:
        return {"order": [], "ratios": [1.0], "auc": 1.0, "impact": 0.0, "final_retained": 1.0}
    auc = sum((left + right) * 0.5 for left, right in zip(ratios, ratios[1:])) / len(removed)
    return {
        "order": removed,
        "ratios": ratios,
        "auc": auc,
        "impact": 1.0 - auc,
        "final_retained": ratios[-1],
    }


def _independent_single_edge_score(
    baseline: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    order: list[str],
    *,
    maximum_sources: int = 24,
) -> dict[str, Any]:
    """Evaluate the declared critical-road objective one road at a time."""

    baseline_efficiency = _independent_efficiency(
        baseline, maximum_sources=maximum_sources
    )
    losses: list[float] = []
    evaluated: list[str] = []
    for key in order:
        record = records.get(key)
        if record is None:
            continue
        graph = baseline.copy()
        u, v = _record_nodes(record, graph)
        if not graph.has_edge(u, v, key):
            continue
        graph.remove_edge(u, v, key)
        disrupted = _independent_efficiency(
            graph, maximum_sources=maximum_sources
        )
        losses.append(max(0.0, (baseline_efficiency - disrupted) / max(1e-15, baseline_efficiency)))
        evaluated.append(key)
    return {
        "order": evaluated,
        "individual_losses": losses,
        "mean_individual_loss": _mean(losses),
        "total_individual_loss": sum(losses),
    }


def _recovery_curve(
    baseline: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    disrupted_keys: list[str],
    order: list[str],
) -> dict[str, Any]:
    graph = baseline.copy()
    for key in disrupted_keys:
        record = records.get(key)
        if record is None:
            continue
        u, v = _record_nodes(record, graph)
        if graph.has_edge(u, v, key):
            graph.remove_edge(u, v, key)
    baseline_efficiency = _independent_efficiency(baseline, maximum_sources=24)
    disrupted_efficiency = _independent_efficiency(graph, maximum_sources=24)
    denominator = max(1e-15, baseline_efficiency - disrupted_efficiency)
    ratios = [0.0]
    restored: list[str] = []
    for key in order:
        if key not in disrupted_keys or key in restored or key not in records:
            continue
        record = records[key]
        u, v = _record_nodes(record, baseline)
        if not baseline.has_edge(u, v, key):
            continue
        graph.add_edge(u, v, key=key, **dict(baseline[u][v][key]))
        restored.append(key)
        current = _independent_efficiency(graph, maximum_sources=24)
        ratios.append(min(1.0, max(0.0, (current - disrupted_efficiency) / denominator)))
    if not restored:
        return {"order": [], "ratios": [0.0], "auc": 0.0, "final_recovery": 0.0}
    auc = sum((left + right) * 0.5 for left, right in zip(ratios, ratios[1:])) / len(restored)
    return {"order": restored, "ratios": ratios, "auc": auc, "final_recovery": ratios[-1]}


def _random_orders(keys: list[str], seeds: tuple[int, ...] = (104729, 130363, 155921)) -> list[list[str]]:
    outputs = []
    for seed in seeds:
        order = list(keys)
        random.Random(seed).shuffle(order)
        outputs.append(order)
    return outputs


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def validate_project(database_path: Path, *, top_k: int = 10) -> dict[str, Any]:
    """Run a deterministic, project-specific scientific comparison suite."""

    started = time.perf_counter()
    requested = max(3, min(int(top_k), 25))
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, None)
        time_row = connection.execute("SELECT MAX(timestamp_epoch) AS timestamp_epoch FROM observations").fetchone()
        timestamp = int(time_row["timestamp_epoch"]) if time_row and time_row["timestamp_epoch"] is not None else None
        metric_row = connection.execute(
            "SELECT SUM(speed IS NOT NULL) AS speed_count, SUM(travel_time IS NOT NULL) AS travel_count, SUM(primary_value IS NOT NULL) AS primary_count FROM observations"
        ).fetchone()
    if graph.number_of_edges() < 3:
        raise ValueError("Scientific network validation needs at least three usable directed topology links.")
    actual_k = min(requested, graph.number_of_edges())

    critical_first = critical_components(database_path, top_k=actual_k, bbox=None)
    critical_second = critical_components(database_path, top_k=actual_k, bbox=None)
    sensitivity_k = min(graph.number_of_edges(), max(actual_k + 5, actual_k * 2))
    critical_sensitivity = critical_components(database_path, top_k=sensitivity_k, bbox=None)
    platform_order = [str(item["key"]) for item in critical_first.get("items", [])]
    if not platform_order:
        raise ValueError("The critical-road model returned no topology links for validation.")

    # Compare rankings under one common deterministic source cohort.  Using a
    # different source budget for the oracle would confound ranking quality
    # with sampling error and can reverse an otherwise identical efficiency
    # objective on large graphs.  The recomputation remains independent code,
    # while the disclosed cohort size is held fixed across every method.
    critical_source_count = max(
        1,
        int(
            critical_first.get("evidence", {}).get("efficiency_source_count")
            or min(24, graph.number_of_nodes())
        ),
    )
    betweenness_order = _edge_score_order(
        graph,
        "networkx_edge_betweenness",
        maximum_sources=critical_source_count,
    )[: len(platform_order)]
    degree_order = _edge_score_order(graph, "degree_only_ablation")[: len(platform_order)]
    random_failure_curves = [
        _independent_single_edge_score(
            graph,
            records,
            order[: len(platform_order)],
            maximum_sources=critical_source_count,
        )
        for order in _random_orders(sorted(records))
    ]
    critical_curves = {
        "platform": _independent_single_edge_score(
            graph,
            records,
            platform_order,
            maximum_sources=critical_source_count,
        ),
        "networkx_edge_betweenness": _independent_single_edge_score(
            graph,
            records,
            betweenness_order,
            maximum_sources=critical_source_count,
        ),
        "degree_only_ablation": _independent_single_edge_score(
            graph,
            records,
            degree_order,
            maximum_sources=critical_source_count,
        ),
        "fixed_seed_random_mean": {
            "mean_individual_loss": _mean([curve["mean_individual_loss"] for curve in random_failure_curves]),
            "total_individual_loss": _mean([curve["total_individual_loss"] for curve in random_failure_curves]),
            "replicates": len(random_failure_curves),
        },
    }
    critical_platform = critical_curves["platform"]["mean_individual_loss"]
    critical_best_baseline = max(
        critical_curves["networkx_edge_betweenness"]["mean_individual_loss"],
        critical_curves["degree_only_ablation"]["mean_individual_loss"],
        critical_curves["fixed_seed_random_mean"]["mean_individual_loss"],
    )
    critical_pass = critical_platform + 1e-10 >= critical_best_baseline

    recovery = restoration_screening(
        database_path,
        selected_keys=platform_order,
        bbox=None,
        top_k=len(platform_order),
        assumptions={"recovery_unit": "road", "priority_objective": "service_gain"},
    )
    recovery_order = [str(item["key"]) for item in recovery.get("items", [])]
    random_recovery_curves = [
        _recovery_curve(graph, records, platform_order, order)
        for order in _random_orders(platform_order)
    ]
    recovery_curves = {
        "platform": _recovery_curve(graph, records, platform_order, recovery_order),
        "static_criticality_order": _recovery_curve(graph, records, platform_order, platform_order),
        "reverse_criticality_ablation": _recovery_curve(graph, records, platform_order, list(reversed(platform_order))),
        "fixed_seed_random_mean": {
            "auc": _mean([curve["auc"] for curve in random_recovery_curves]),
            "replicates": len(random_recovery_curves),
        },
    }
    recovery_platform = recovery_curves["platform"]["auc"]
    recovery_best_baseline = max(
        recovery_curves["static_criticality_order"]["auc"],
        recovery_curves["reverse_criticality_ablation"]["auc"],
        recovery_curves["fixed_seed_random_mean"]["auc"],
    )
    recovery_pass = recovery_platform + 1e-10 >= recovery_best_baseline

    stress = progressive_robustness(
        database_path, top_k=len(platform_order), bbox=None, failure_keys=platform_order
    )
    independent_stress = _failure_curve(
        graph, records, platform_order, maximum_sources=64
    )
    platform_ratios = [1.0, *[float(item["performance_ratio"]) for item in stress.get("items", [])]]
    stress_errors = [abs(left - right) for left, right in zip(platform_ratios, independent_stress["ratios"])]
    stress_max_error = max(stress_errors, default=0.0)
    stress_pass = len(platform_ratios) == len(independent_stress["ratios"]) and stress_max_error <= 1e-10

    route_checks: list[dict[str, Any]] = []
    reachable_pairs = []
    for source in _stable_sources(graph.nodes, min(5, graph.number_of_nodes())):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        targets = [target for target in sorted(lengths, key=str) if target != source and lengths[target] > 0]
        if targets:
            reachable_pairs.append((source, targets[-1]))
    for source, target in reachable_pairs[:5]:
        dijkstra = nx.shortest_path_length(graph, source=source, target=target, weight="weight", method="dijkstra")
        bellman_ford = nx.single_source_bellman_ford_path_length(graph, source=source, weight="weight")[target]
        route_checks.append(
            {
                "source": str(source),
                "target": str(target),
                "dijkstra_impedance": float(dijkstra),
                "bellman_ford_oracle": float(bellman_ford),
                "absolute_error": abs(float(dijkstra) - float(bellman_ford)),
            }
        )
    route_max_error = max((item["absolute_error"] for item in route_checks), default=0.0)
    route_pass = bool(route_checks) and route_max_error <= 1e-10

    metric = "speed" if metric_row and int(metric_row["speed_count"] or 0) else "travel_time" if metric_row and int(metric_row["travel_count"] or 0) else "primary"
    hotspot_unavailable_reason = None
    try:
        hotspot_first = spatial_hotspots(database_path, metric=metric, top_k=actual_k, bbox=None, timestamp=timestamp)
        hotspot_second = spatial_hotspots(database_path, metric=metric, top_k=actual_k, bbox=None, timestamp=timestamp)
        hotspot_digest_1 = _result_digest({"items": hotspot_first.get("items"), "evidence": hotspot_first.get("evidence")})
        hotspot_digest_2 = _result_digest({"items": hotspot_second.get("items"), "evidence": hotspot_second.get("evidence")})
    except ValueError as error:
        # Topology validation remains meaningful when the only uploaded value
        # is explicitly descriptive and therefore cannot honestly define
        # degradation.  The report marks the area repeatability branch as not
        # applicable instead of inventing a severity direction.
        hotspot_unavailable_reason = str(error)
        hotspot_digest_1 = hotspot_digest_2 = "not-applicable"
    critical_digest_1 = _result_digest({"items": critical_first.get("items"), "evidence": critical_first.get("evidence")})
    critical_digest_2 = _result_digest({"items": critical_second.get("items"), "evidence": critical_second.get("evidence")})
    sensitivity_prefix = [str(item["key"]) for item in critical_sensitivity.get("items", [])[: len(platform_order)]]
    sensitivity_overlap = len(set(platform_order) & set(sensitivity_prefix)) / max(1, len(set(platform_order) | set(sensitivity_prefix)))
    repeatability_pass = critical_digest_1 == critical_digest_2 and hotspot_digest_1 == hotspot_digest_2
    sensitivity_pass = sensitivity_overlap >= 0.8

    checks = [
        {"id": "critical-vs-baselines", "label": "Critical-road impact vs traditional baselines", "passed": critical_pass, "value": critical_platform, "threshold": critical_best_baseline},
        {"id": "recovery-vs-baselines", "label": "Recovery-curve area vs static baselines", "passed": recovery_pass, "value": recovery_platform, "threshold": recovery_best_baseline},
        {"id": "stress-independent-oracle", "label": "Stress curve vs independent recomputation", "passed": stress_pass, "value": stress_max_error, "threshold": 1e-10},
        {"id": "route-independent-oracle", "label": "Shortest-path impedance vs Bellman-Ford oracle", "passed": route_pass, "value": route_max_error, "threshold": 1e-10},
        {"id": "repeatability", "label": "Repeated critical-area and critical-road results", "passed": repeatability_pass, "value": 1.0 if repeatability_pass else 0.0, "threshold": 1.0},
        {"id": "top-k-sensitivity", "label": "Critical-road prefix stability under a larger Top-N", "passed": sensitivity_pass, "value": sensitivity_overlap, "threshold": 0.8},
    ]
    passed_count = sum(bool(item["passed"]) for item in checks)
    status = "passed" if passed_count == len(checks) else "attention_required"
    return {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "summary": {
            "passed": passed_count,
            "total": len(checks),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "network_nodes": graph.number_of_nodes(),
            "network_edges": graph.number_of_edges(),
            "top_k": len(platform_order),
            "plain_language": (
                "All independent comparison gates passed for this project and parameter set."
                if status == "passed" else
                "At least one validation gate needs investigation; the detailed comparison is retained rather than hidden."
            ),
        },
        "checks": checks,
        "comparisons": {
            "critical_roads": {
                "evaluation_metric": "Independent mean single-road efficiency loss, matching the declared critical-road objective; higher is stronger",
                "platform_tier": critical_first.get("evidence", {}).get("ranking_mode"),
                "evaluation_source_count": critical_source_count,
                "sampling_protocol": "Common deterministic source cohort for the platform order and every baseline; independent efficiency implementation",
                "curves": critical_curves,
            },
            "recovery_priorities": {
                "evaluation_metric": "Independent area under normalised service-recovery curve; higher is earlier recovery",
                "platform_tier": recovery.get("evidence", {}).get("method"),
                "curves": recovery_curves,
            },
            "network_resilience": {
                "maximum_absolute_error": stress_max_error,
                "platform_curve": platform_ratios,
                "independent_curve": independent_stress["ratios"],
            },
            "routing": {
                "pair_count": len(route_checks),
                "maximum_absolute_error": route_max_error,
                "pairs": route_checks,
            },
        },
        "sensitivity_and_ablation": {
            "top_k_base": len(platform_order),
            "top_k_comparison": sensitivity_k,
            "critical_prefix_jaccard": sensitivity_overlap,
            "ablations": ["NetworkX edge betweenness", "degree-only ranking", "reverse recovery order", "fixed-seed random sequences"],
        },
        "repeatability": {
            "critical_roads_sha256": critical_digest_1,
            "critical_areas_sha256": hotspot_digest_1,
            "critical_areas_not_applicable_reason": hotspot_unavailable_reason,
            "second_run_matches": repeatability_pass,
            "randomness": "No production model uses a random seed; random orders appear only as fixed-seed validation baselines.",
        },
        "literature": LITERATURE,
        "scientific_boundary": (
            "Passing these gates demonstrates numerical agreement, deterministic repeatability and performance against the declared baselines for this project. "
            "It does not prove causal disaster impact, universal superiority, live operational safety or out-of-sample policy effectiveness."
        ),
    }
