from __future__ import annotations

"""Deterministic traffic redistribution utilities used by Prototype 5.

The module deliberately separates *what is observed* from *what is assumed*.
When an uploaded project does not contain an OD matrix, a small, reproducible
unit-demand OD design is generated from the uploaded topology.  It is a
network sensitivity experiment, not an estimate of real traffic demand.  The
same design is assigned to the baseline and disrupted networks so changes are
caused only by the confirmed scenario operators.

The lower-data tier follows Wardrop's shortest-route principle and the
all-or-nothing assignment used in classical traffic-assignment practice.  It
is intentionally deterministic: no random sampling, synthetic disaster
probability or invented population is introduced.
"""

from collections import defaultdict
import math
from typing import Any, Iterable

import networkx as nx


MAX_PROXY_OD_PAIRS = 800
MAX_PROXY_ANCHORS = 32
MAX_SEED_OD_PAIRS = 700
MAX_MAPPED_INDIRECT_LINKS = 750


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def collapse_to_route_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Keep the least-impedance parallel link for each directed node pair."""

    route_graph = nx.DiGraph()
    route_graph.graph.update(graph.graph)
    for u, v, edge_key, data in graph.edges(keys=True, data=True):
        weight = float(data.get("weight", 1.0))
        current = route_graph.get_edge_data(u, v)
        if current is not None and float(current["weight"]) <= weight:
            continue
        route_graph.add_edge(
            u,
            v,
            weight=weight,
            link_key=str(edge_key),
            capacity=float(data["capacity"]) if _finite_positive(data.get("capacity")) else None,
            observed_flow=float(data["observed_flow"]) if _finite_nonnegative(data.get("observed_flow")) else None,
            free_flow_weight=float(data["free_flow_weight"]) if _finite_positive(data.get("free_flow_weight")) else None,
        )
    return route_graph


def _node_order(route_graph: nx.DiGraph, records: dict[str, dict[str, Any]]) -> list[str]:
    """Return a stable geography-aware order without relying on random seeds."""

    positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in records.values():
        latitude = record.get("latitude")
        longitude = record.get("longitude")
        if latitude is None or longitude is None:
            continue
        point = (float(latitude), float(longitude))
        positions[str(record["graph_from"])].append(point)
        positions[str(record["graph_to"])].append(point)

    def key(node: Any) -> tuple[float, float, str]:
        points = positions.get(str(node), [])
        if not points:
            return (math.inf, math.inf, str(node))
        latitude = sum(point[0] for point in points) / len(points)
        longitude = sum(point[1] for point in points) / len(points)
        return (round(latitude, 7), round(longitude, 7), str(node))

    return [str(node) for node in sorted(route_graph.nodes, key=key)]


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    if count <= 1:
        return [values[0]]
    indices = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indices)]


def deterministic_proxy_od_pairs(
    route_graph: nx.DiGraph,
    records: dict[str, dict[str, Any]],
    *,
    maximum_pairs: int = MAX_PROXY_OD_PAIRS,
) -> list[dict[str, Any]]:
    """Construct a disclosed unit-demand design over the largest component.

    Anchors are geographically spread using a stable ordering.  Rotating each
    origin across the anchor list avoids the bias of pairing only adjacent
    sorted nodes.  Reachability is checked during assignment.
    """

    if route_graph.number_of_nodes() < 2 or route_graph.number_of_edges() == 0:
        return []
    components = list(nx.weakly_connected_components(route_graph))
    if not components:
        return []
    largest = max(components, key=lambda nodes: (len(nodes), sorted(str(node) for node in nodes)[0]))
    ordered = [node for node in _node_order(route_graph, records) if node in largest]
    if len(ordered) < 2:
        return []
    anchor_count = min(MAX_PROXY_ANCHORS, len(ordered), max(6, int(math.sqrt(maximum_pairs)) + 1))
    anchors = _evenly_spaced(ordered, anchor_count)
    pairs: list[dict[str, Any]] = []
    for rotation in range(1, len(anchors)):
        for index, origin in enumerate(anchors):
            destination = anchors[(index + rotation) % len(anchors)]
            if origin == destination:
                continue
            pairs.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "demand": 1.0,
                    "source": "deterministic_unit_od_proxy",
                }
            )
            if len(pairs) >= maximum_pairs:
                return pairs
    return pairs


def deterministic_seed_od_pairs(
    route_graph: nx.DiGraph,
    seed_keys: set[str],
    *,
    maximum_pairs: int = MAX_SEED_OD_PAIRS,
) -> list[dict[str, Any]]:
    """Add local OD evidence around user-selected initial disruptions.

    A globally spread anchor design can miss a selected road on a large or
    weakly connected network.  For each deterministically screened seed, this
    function adds its directed From node → To node relation.  The baseline
    shortest path must still choose the seed before the result counts it as
    covered; no route or demand is forced through the edge.
    """

    if not seed_keys or maximum_pairs <= 0:
        return []
    edge_by_key = {
        str(data.get("link_key")): (str(u), str(v))
        for u, v, data in route_graph.edges(data=True)
    }
    available = sorted(key for key in seed_keys if key in edge_by_key)
    screened = _evenly_spaced(available, min(maximum_pairs, len(available)))
    return [
        {
            "origin": edge_by_key[key][0],
            "destination": edge_by_key[key][1],
            "demand": 1.0,
            "source": "selected_seed_local_od_proxy",
            "seed_key": key,
        }
        for key in screened
        if edge_by_key[key][0] != edge_by_key[key][1]
    ]


def _merge_od_designs(
    seed_pairs: list[dict[str, Any]],
    global_pairs: list[dict[str, Any]],
    *,
    maximum_pairs: int,
) -> list[dict[str, Any]]:
    """Prefer seed-local evidence, then fill remaining capacity globally."""

    merged: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for pair in [*seed_pairs, *global_pairs]:
        identity = (str(pair["origin"]), str(pair["destination"]))
        if identity in identities:
            continue
        identities.add(identity)
        merged.append(pair)
        if len(merged) >= maximum_pairs:
            break
    return merged


def _path_link_keys(route_graph: nx.DiGraph, path: list[str]) -> list[str]:
    return [str(route_graph[u][v]["link_key"]) for u, v in zip(path, path[1:])]


def all_or_nothing_assignment(
    route_graph: nx.DiGraph,
    od_pairs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Assign each OD demand to a minimum-impedance path."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in od_pairs:
        grouped[str(pair["origin"])].append(pair)
    paths: dict[tuple[str, str], dict[str, Any]] = {}
    link_flow: dict[str, float] = defaultdict(float)
    reachable = 0
    disconnected = 0
    total_connected_cost = 0.0
    total_demand = 0.0
    for origin, pairs in grouped.items():
        if origin not in route_graph:
            disconnected += len(pairs)
            continue
        distances, node_paths = nx.single_source_dijkstra(route_graph, origin, weight="weight")
        for pair in pairs:
            destination = str(pair["destination"])
            demand = max(0.0, float(pair.get("demand", 1.0)))
            total_demand += demand
            identity = (origin, destination)
            if destination not in node_paths:
                disconnected += 1
                paths[identity] = {"reachable": False, "demand": demand}
                continue
            nodes = [str(node) for node in node_paths[destination]]
            links = _path_link_keys(route_graph, nodes)
            cost = float(distances[destination])
            for key in links:
                link_flow[key] += demand
            reachable += 1
            total_connected_cost += demand * cost
            paths[identity] = {
                "reachable": True,
                "nodes": nodes,
                "links": links,
                "cost": cost,
                "demand": demand,
            }
    return {
        "paths": paths,
        "link_flow": dict(link_flow),
        "reachable_pair_count": reachable,
        "disconnected_pair_count": disconnected,
        "total_connected_cost": total_connected_cost,
        "total_demand": total_demand,
    }


def _capacity_data_ready(route_graph: nx.DiGraph) -> bool:
    if route_graph.number_of_edges() == 0:
        return False
    complete = sum(
        _finite_positive(data.get("capacity"))
        and _finite_positive(data.get("free_flow_weight"))
        and data.get("observed_flow") is not None
        for _, _, data in route_graph.edges(data=True)
    )
    return complete / route_graph.number_of_edges() >= 0.8


def _calibrated_proxy_pairs(
    route_graph: nx.DiGraph,
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    unit = all_or_nothing_assignment(route_graph, pairs)
    assigned_link_units = sum(unit["link_flow"].values())
    observed_link_flow = sum(
        float(data.get("observed_flow") or 0.0)
        for _, _, data in route_graph.edges(data=True)
    )
    if assigned_link_units <= 0 or observed_link_flow <= 0:
        return pairs, 1.0
    scale = observed_link_flow / assigned_link_units
    calibrated = [{**pair, "demand": float(pair.get("demand", 1.0)) * scale} for pair in pairs]
    return calibrated, scale


def _set_bpr_weights(
    route_graph: nx.DiGraph,
    link_flow: dict[str, float],
    *,
    alpha: float,
    beta: float,
) -> None:
    for _, _, data in route_graph.edges(data=True):
        key = str(data["link_key"])
        t0 = float(data.get("free_flow_weight") or data["weight"])
        capacity = data.get("capacity")
        if _finite_positive(capacity):
            ratio = max(0.0, float(link_flow.get(key, 0.0))) / float(capacity)
            data["weight"] = t0 * (1.0 + alpha * ratio**beta)
        else:
            data["weight"] = t0


def _beckmann_step(
    route_graph: nx.DiGraph,
    current: dict[str, float],
    target: dict[str, float],
    *,
    alpha: float,
    beta: float,
) -> float:
    edge_data = {str(data["link_key"]): data for _, _, data in route_graph.edges(data=True)}

    def derivative(step: float) -> float:
        value = 0.0
        for key in set(current) | set(target):
            direction = float(target.get(key, 0.0)) - float(current.get(key, 0.0))
            if abs(direction) <= 1e-12:
                continue
            data = edge_data.get(key)
            if data is None:
                continue
            flow = max(0.0, float(current.get(key, 0.0)) + step * direction)
            t0 = float(data.get("free_flow_weight") or data["weight"])
            capacity = data.get("capacity")
            travel_time = (
                t0 * (1.0 + alpha * (flow / float(capacity)) ** beta)
                if _finite_positive(capacity)
                else t0
            )
            value += direction * travel_time
        return value

    low_value = derivative(0.0)
    high_value = derivative(1.0)
    if low_value >= 0:
        return 0.0
    if high_value <= 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = (low + high) / 2.0
        if derivative(middle) <= 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def capacity_constrained_assignment(
    route_graph: nx.DiGraph,
    od_pairs: list[dict[str, Any]],
    *,
    alpha: float = 0.15,
    beta: float = 4.0,
    maximum_iterations: int = 30,
    relative_gap_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Solve a Beckmann/BPR user-equilibrium approximation by Frank-Wolfe."""

    working = route_graph.copy()
    _set_bpr_weights(working, {}, alpha=alpha, beta=beta)
    initial = all_or_nothing_assignment(working, od_pairs)
    current = {key: float(value) for key, value in initial["link_flow"].items()}
    gap = math.inf
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        _set_bpr_weights(working, current, alpha=alpha, beta=beta)
        auxiliary_result = all_or_nothing_assignment(working, od_pairs)
        auxiliary = {key: float(value) for key, value in auxiliary_result["link_flow"].items()}
        direction_norm = sum(abs(auxiliary.get(key, 0.0) - current.get(key, 0.0)) for key in set(current) | set(auxiliary))
        flow_norm = max(1.0, sum(abs(value) for value in current.values()))
        gap = direction_norm / flow_norm
        if gap <= relative_gap_tolerance:
            break
        step = _beckmann_step(working, current, auxiliary, alpha=alpha, beta=beta)
        current = {
            key: current.get(key, 0.0) + step * (auxiliary.get(key, 0.0) - current.get(key, 0.0))
            for key in set(current) | set(auxiliary)
        }
    _set_bpr_weights(working, current, alpha=alpha, beta=beta)
    final_paths = all_or_nothing_assignment(working, od_pairs)
    final_paths.update(
        {
            "link_flow": current,
            "solver": "frank_wolfe_beckmann_bpr",
            "iterations": iteration,
            "relative_flow_gap": gap,
            "converged": gap <= relative_gap_tolerance,
            "bpr_alpha": alpha,
            "bpr_beta": beta,
        }
    )
    return final_paths


def compare_assignments(
    baseline_graph: nx.MultiDiGraph,
    scenario_graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    *,
    seed_keys: set[str],
    maximum_pairs: int = MAX_PROXY_OD_PAIRS,
) -> dict[str, Any]:
    """Run the same deterministic demand design before and after disruption."""

    baseline_route_graph = collapse_to_route_graph(baseline_graph)
    scenario_route_graph = collapse_to_route_graph(scenario_graph)
    seed_pairs = deterministic_seed_od_pairs(
        baseline_route_graph,
        seed_keys,
        maximum_pairs=min(MAX_SEED_OD_PAIRS, maximum_pairs),
    )
    global_pairs = deterministic_proxy_od_pairs(
        baseline_route_graph,
        records,
        maximum_pairs=maximum_pairs,
    )
    candidate_pairs = _merge_od_designs(
        seed_pairs,
        global_pairs,
        maximum_pairs=maximum_pairs,
    )
    baseline_reachability = all_or_nothing_assignment(baseline_route_graph, candidate_pairs)
    # Only baseline-reachable OD relations form the comparison set.  This
    # prevents a pre-existing directionality gap from being attributed to the
    # user-created disruption.
    comparable_pairs = [
        pair
        for pair in candidate_pairs
        if baseline_reachability["paths"].get((str(pair["origin"]), str(pair["destination"])), {}).get("reachable")
    ]
    sampled_seed_keys = {str(pair["seed_key"]) for pair in seed_pairs}
    covered_seed_keys = {
        str(pair["seed_key"])
        for pair in seed_pairs
        if baseline_reachability["paths"].get(
            (str(pair["origin"]), str(pair["destination"])), {}
        ).get("reachable")
        and str(pair["seed_key"])
        in baseline_reachability["paths"][(str(pair["origin"]), str(pair["destination"]))].get("links", [])
    }
    capacity_tier = _capacity_data_ready(baseline_route_graph) and _capacity_data_ready(scenario_route_graph)
    demand_scale = 1.0
    if capacity_tier:
        comparable_pairs, demand_scale = _calibrated_proxy_pairs(baseline_route_graph, comparable_pairs)
        baseline = capacity_constrained_assignment(baseline_route_graph, comparable_pairs)
        scenario = capacity_constrained_assignment(scenario_route_graph, comparable_pairs)
        tier = "observed_flow_calibrated_capacity_ue_proxy"
        theory = "Wardrop user equilibrium as a Beckmann program with the BPR link performance function"
        demand_basis = (
            "Selected-seed local OD relations plus deterministic geographic topology anchors, scaled so assigned baseline link-flow units match the uploaded observed-flow total"
        )
    else:
        baseline = all_or_nothing_assignment(baseline_route_graph, comparable_pairs)
        scenario = all_or_nothing_assignment(scenario_route_graph, comparable_pairs)
        tier = "deterministic_unit_od_aon"
        theory = "Wardrop shortest-route principle with classical all-or-nothing assignment"
        demand_basis = "Equal unit demand over selected-seed local OD relations plus deterministic, geographically spread topology anchors"

    rerouted = 0
    disconnected = 0
    connected_after = 0
    detour_ratios: list[float] = []
    for pair in comparable_pairs:
        identity = (str(pair["origin"]), str(pair["destination"]))
        before = baseline["paths"][identity]
        after = scenario["paths"].get(identity, {"reachable": False})
        if not after.get("reachable"):
            disconnected += 1
            continue
        connected_after += 1
        if before.get("links") != after.get("links"):
            rerouted += 1
        if float(before.get("cost", 0)) > 0:
            detour_ratios.append(max(0.0, float(after["cost"]) / float(before["cost"]) - 1.0))

    all_keys = set(baseline["link_flow"]) | set(scenario["link_flow"])
    changes: list[dict[str, Any]] = []
    for key in all_keys:
        before_flow = float(baseline["link_flow"].get(key, 0.0))
        after_flow = float(scenario["link_flow"].get(key, 0.0))
        delta = after_flow - before_flow
        if abs(delta) <= 1e-12:
            continue
        changes.append(
            {
                "key": key,
                "baseline_proxy_flow": before_flow,
                "scenario_proxy_flow": after_flow,
                "proxy_flow_change": delta,
                "relative_proxy_flow_change": delta / before_flow if before_flow > 0 else None,
                "seed": key in seed_keys,
            }
        )
    increases = sorted(
        (item for item in changes if item["proxy_flow_change"] > 0 and not item["seed"]),
        key=lambda item: (-item["proxy_flow_change"], -item["scenario_proxy_flow"], item["key"]),
    )
    relief = sorted(
        (item for item in changes if item["proxy_flow_change"] < 0 and not item["seed"]),
        key=lambda item: (item["proxy_flow_change"], item["key"]),
    )
    maximum_increase = max((item["proxy_flow_change"] for item in increases), default=0.0)
    for rank, item in enumerate(increases, start=1):
        item.update(
            {
                "rank": rank,
                "impact_role": "rerouting_pressure",
                "severity": item["proxy_flow_change"] / maximum_increase if maximum_increase > 0 else 0.0,
            }
        )
    for rank, item in enumerate(relief, start=1):
        item.update({"rank": rank, "impact_role": "traffic_relief"})

    baseline_cost = float(baseline["total_connected_cost"])
    scenario_cost = float(scenario["total_connected_cost"])
    connected_cost_change = (
        scenario_cost / baseline_cost - 1.0
        if baseline_cost > 0 and disconnected == 0
        else None
    )
    mean_detour = sum(detour_ratios) / len(detour_ratios) if detour_ratios else 0.0
    maximum_detour = max(detour_ratios, default=0.0)
    seed_dependency = [
        {
            "key": key,
            "baseline_proxy_flow": float(baseline["link_flow"].get(key, 0.0)),
            "scenario_proxy_flow": float(scenario["link_flow"].get(key, 0.0)),
            "local_od_sampled": key in sampled_seed_keys,
            "baseline_local_path_uses_seed": key in covered_seed_keys,
        }
        for key in sorted(seed_keys)
    ]
    seed_dependency.sort(key=lambda item: (-item["baseline_proxy_flow"], item["key"]))
    for rank, item in enumerate(seed_dependency, start=1):
        item["recovery_rank"] = rank

    solver_converged = bool(baseline.get("converged", True) and scenario.get("converged", True))
    return {
        "tier": tier,
        "theory": theory,
        "demand_basis": demand_basis,
        "demand_scale": demand_scale,
        "solver": baseline.get("solver", "dijkstra_all_or_nothing"),
        "baseline_solver_converged": baseline.get("converged", True),
        "scenario_solver_converged": scenario.get("converged", True),
        "baseline_solver_iterations": baseline.get("iterations", 1),
        "scenario_solver_iterations": scenario.get("iterations", 1),
        "candidate_pair_count": len(candidate_pairs),
        "global_anchor_pair_count": sum(pair.get("source") == "deterministic_unit_od_proxy" for pair in candidate_pairs),
        "seed_local_pair_count": sum(pair.get("source") == "selected_seed_local_od_proxy" for pair in candidate_pairs),
        "selected_seed_count": len(seed_keys),
        "sampled_seed_count": len(sampled_seed_keys),
        "covered_seed_count": len(covered_seed_keys),
        "uncovered_seed_count": max(0, len(seed_keys) - len(covered_seed_keys)),
        "seed_od_coverage_percent": 100.0 * len(covered_seed_keys) / len(seed_keys) if seed_keys else 100.0,
        "seed_sampling_limit": MAX_SEED_OD_PAIRS,
        "comparable_pair_count": len(comparable_pairs),
        "connected_after_count": connected_after,
        "rerouted_pair_count": rerouted,
        "disconnected_pair_count": disconnected,
        "rerouted_pair_percent": 100.0 * rerouted / len(comparable_pairs) if comparable_pairs else 0.0,
        "disconnected_pair_percent": 100.0 * disconnected / len(comparable_pairs) if comparable_pairs else 0.0,
        "mean_detour_percent": mean_detour * 100.0,
        "maximum_detour_percent": maximum_detour * 100.0,
        "connected_system_cost_change": connected_cost_change,
        "indirect_increases": increases[:MAX_MAPPED_INDIRECT_LINKS],
        "indirect_increase_count": len(increases),
        "indirect_relief": relief[:MAX_MAPPED_INDIRECT_LINKS],
        "indirect_relief_count": len(relief),
        "recovery_screening": seed_dependency,
        "randomness": "none",
        "verification": {
            "passed": bool(comparable_pairs)
            and baseline["reachable_pair_count"] == len(comparable_pairs)
            and scenario["reachable_pair_count"] + scenario["disconnected_pair_count"] == len(comparable_pairs)
            and solver_converged,
            "seed_coverage_complete": bool(sampled_seed_keys) and covered_seed_keys == sampled_seed_keys,
            "checks": [
                "one fixed OD design used for baseline and scenario",
                "deterministically screened selected seeds each receive a local From-to-To OD check",
                "baseline local shortest-path use is reported per sampled seed and is not fabricated for bypassed links",
                "every comparison OD pair reachable in baseline",
                "all assigned link keys grounded in uploaded topology",
                "flow conservation represented by complete path assignment",
                "no random sampling",
                "capacity solver converged when the Beckmann/BPR tier was activated",
            ],
        },
    }
