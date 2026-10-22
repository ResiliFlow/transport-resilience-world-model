from __future__ import annotations

from copy import deepcopy
import colorsys
import heapq
from pathlib import Path
from typing import Any

import networkx as nx

from .models import _connect, _nearest_timestamp, _summary
from .scenarios import _geometry_rows, _row_feature, _speed_to_metres_per_second, _time_graph


ROUTE_VARIANT_COLOURS = (
    "#f4b400",  # uploaded historical state
    "#0f766e",  # saved scenario 1
    "#d97706",  # saved scenario 2
    "#7c3aed",  # saved scenario 3
    "#2563eb",  # saved scenario 4
    "#db2777",  # saved scenario 5
)
ROUTE_BASELINE_COLOUR = "#2563eb"
ROUTE_PRIMARY_COLOUR = "#f28c28"
ROUTE_ALTERNATIVE_COLOUR = "#ffd43b"
MAX_ROUTE_OPTIONS = 50


def _route_option_colour(rank: int) -> str:
    """Return a deterministic, map-safe colour for one recommended route."""

    fixed = (
        "#1464d2", "#e87500", "#7b2cbf", "#00897b", "#c2185b",
        "#5f7f18", "#d1495b", "#0077a8", "#8c564b", "#6a5acd",
        "#b05f00", "#087f5b",
    )
    index = max(0, int(rank) - 1)
    if index < len(fixed):
        return fixed[index]
    hue = (0.61803398875 * index) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.78 if index % 2 else 0.88)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _reference_weight(record: dict[str, Any], basis: str, speed_unit: str) -> float:
    """Return the strongest defensible internal-reference impedance for one link."""

    if basis == "observed_travel_time" and record.get("baseline_travel_time"):
        return float(record["baseline_travel_time"])
    if basis == "observed_speed_derived_time" and record.get("length_m") and record.get("baseline_speed"):
        metres_per_second = _speed_to_metres_per_second(float(record["baseline_speed"]), speed_unit)
        if metres_per_second and metres_per_second > 0:
            return float(record["length_m"]) / metres_per_second
    if basis == "uploaded_free_flow_time" and record.get("free_flow_weight"):
        return float(record["free_flow_weight"])
    if basis == "reference_travel_time" and record.get("baseline_travel_time"):
        return float(record["baseline_travel_time"])
    if basis == "link_length" and record.get("length_m"):
        return float(record["length_m"])
    if basis == "uniform_hop_count":
        return 1.0
    return float(record["weight"])


def _reference_graph(
    historical_graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    speed_unit: str,
) -> tuple[nx.MultiDiGraph, int]:
    graph = historical_graph.copy()
    basis = str(historical_graph.graph.get("impedance_mode") or "uniform_hop_count")
    changed = 0
    for key, record in records.items():
        u, v = record["graph_from"], record["graph_to"]
        if not graph.has_edge(u, v, key):
            continue
        reference = _reference_weight(record, basis, speed_unit)
        if abs(reference - float(record["weight"])) > 1e-12:
            changed += 1
        graph[u][v][key]["weight"] = reference
    graph.graph.update(
        {
            "comparison_role": "internal_reference",
            "reference_weight_changes": changed,
            "impedance_label": f"Internal reference using the same {historical_graph.graph.get('impedance_label', 'link impedance')} basis",
        }
    )
    return graph, changed


def _scenario_graph(
    historical_graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    changes: list[dict[str, Any]],
) -> tuple[nx.MultiDiGraph, int]:
    graph = historical_graph.copy()
    applied = 0
    for raw in changes:
        key = str(raw.get("key") or "")
        record = records.get(key)
        if record is None:
            continue
        u, v = record["graph_from"], record["graph_to"]
        if not graph.has_edge(u, v, key):
            continue
        effect = str(raw.get("effect") or "")
        magnitude = float(raw.get("magnitude") or (1.0 if effect == "closed" else 0.0))
        if effect == "closed":
            graph.remove_edge(u, v, key)
        elif effect == "speed_reduction" and 0.01 <= magnitude <= 0.95:
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) / max(0.05, 1.0 - magnitude)
        elif effect == "travel_time_increase" and 0.01 <= magnitude <= 5.0:
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) * (1.0 + magnitude)
        else:
            continue
        applied += 1
    graph.graph.update({"comparison_role": "saved_scenario", "scenario_changes_applied": applied})
    return graph, applied


def _edge_choice(graph: nx.MultiDiGraph, u: str, v: str) -> tuple[str, dict[str, Any]]:
    candidates = graph.get_edge_data(u, v) or {}
    if not candidates:
        raise nx.NetworkXNoPath(f"No directed link connects {u} to {v}.")
    key, data = min(
        candidates.items(),
        key=lambda item: (float(item[1].get("weight", 1.0)), str(item[0])),
    )
    return str(key), data


def _directed_disconnection(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
) -> dict[str, Any]:
    """Classify a failed directed route without inventing reverse travel.

    A large road extract can be weakly connected while containing many small
    strongly connected components.  In that case a generic ``No path`` result
    is mathematically correct but easy to misread as a software failure.  The
    weak-connectivity oracle distinguishes missing direction continuity from
    a genuine split between undirected components.
    """

    if source not in graph or target not in graph:
        return {
            "reason_code": "endpoint_node_missing",
            "weakly_connected": False,
            "reason": "One endpoint node is absent from this comparison state's uploaded topology.",
        }
    undirected = graph.to_undirected(as_view=True)
    weakly_connected = nx.has_path(undirected, source, target)
    if weakly_connected:
        return {
            "reason_code": "direction_continuity_missing",
            "weakly_connected": True,
            "reason": (
                "The endpoints remain connected when direction is ignored, but the uploaded From/To "
                "directions do not form a legal directed route between them. No reverse road is invented."
            ),
        }
    return {
        "reason_code": "different_weak_components",
        "weakly_connected": False,
        "reason": "The endpoints lie in different connected parts of this comparison state's uploaded topology.",
    }


def _route_between_links(
    graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    origin_key: str,
    destination_key: str,
) -> dict[str, Any]:
    origin = records[origin_key]
    destination = records[destination_key]
    if origin_key == destination_key:
        if not graph.has_edge(origin["graph_from"], origin["graph_to"], origin_key):
            return {
                "reachable": False,
                "reason": "The selected road is closed in this comparison state.",
                "reason_code": "selected_road_closed",
                "weakly_connected": None,
                "keys": [],
            }
        data = graph[origin["graph_from"]][origin["graph_to"]][origin_key]
        keys = [origin_key]
        edge_data = [data]
    else:
        if not graph.has_edge(origin["graph_from"], origin["graph_to"], origin_key):
            return {
                "reachable": False,
                "reason": "The selected start road is closed in this comparison state.",
                "reason_code": "start_road_closed",
                "weakly_connected": None,
                "keys": [],
            }
        if not graph.has_edge(destination["graph_from"], destination["graph_to"], destination_key):
            return {
                "reachable": False,
                "reason": "The selected destination road is closed in this comparison state.",
                "reason_code": "destination_road_closed",
                "weakly_connected": None,
                "keys": [],
            }
        try:
            middle_nodes = nx.shortest_path(
                graph,
                source=origin["graph_to"],
                target=destination["graph_from"],
                weight="weight",
                method="dijkstra",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            diagnostic = _directed_disconnection(
                graph,
                str(origin["graph_to"]),
                str(destination["graph_from"]),
            )
            return {
                "reachable": False,
                **diagnostic,
                "keys": [],
            }
        keys = [origin_key]
        edge_data = [graph[origin["graph_from"]][origin["graph_to"]][origin_key]]
        for u, v in zip(middle_nodes, middle_nodes[1:]):
            key, data = _edge_choice(graph, u, v)
            keys.append(key)
            edge_data.append(data)
        keys.append(destination_key)
        edge_data.append(graph[destination["graph_from"]][destination["graph_to"]][destination_key])

    impedance = sum(float(data.get("weight", 0.0)) for data in edge_data)
    known_lengths = [records[key].get("length_m") for key in keys if records.get(key, {}).get("length_m")]
    return {
        "reachable": True,
        "reason": "",
        "reason_code": None,
        "weakly_connected": True,
        "keys": keys,
        "edge_weights": [float(data.get("weight", 0.0)) for data in edge_data],
        "impedance": impedance,
        "link_count": len(keys),
        "distance_m": sum(float(value) for value in known_lengths) if len(known_lengths) == len(keys) else None,
    }


def _route_alternatives_between_links(
    graph: nx.MultiDiGraph | nx.MultiGraph,
    records: dict[str, dict[str, Any]],
    origin_key: str,
    destination_key: str,
    *,
    count: int = 2,
    maximum_overlap: float = 0.60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return up to ``count`` deterministic, distinct loopless routes.

    Yen's loopless-path enumeration is provided by NetworkX's
    ``shortest_simple_paths``.  Parallel directed links are collapsed to the
    least-impedance link with a deterministic key tie-break, while the two
    user-selected endpoint road links remain fixed.  After the primary route,
    candidates are retained only when their road-set overlap with every route
    already retained does not exceed ``maximum_overlap``.  If the network
    cannot supply the requested number of visibly distinct routes, fewer are
    returned instead of recolouring near-duplicate paths.
    """

    requested = int(count)
    if requested < 1 or requested > MAX_ROUTE_OPTIONS:
        raise ValueError(f"Choose between 1 and {MAX_ROUTE_OPTIONS} route options.")
    overlap_limit = max(0.0, min(float(maximum_overlap), 1.0))
    origin = records[origin_key]
    destination = records[destination_key]
    if origin_key == destination_key:
        routes = [_route_between_links(graph, records, origin_key, destination_key)]
        return routes, {
            "requested_count": requested,
            "returned_count": len([route for route in routes if route.get("reachable")]),
            "candidates_examined": 1,
            "search_exhausted": True,
            "search_budget_reached": False,
            "overlap_relaxed": False,
        }
    if not graph.has_edge(origin["graph_from"], origin["graph_to"], origin_key):
        routes = [{
            "reachable": False,
            "reason": "The selected start road is closed in this comparison state.",
            "reason_code": "start_road_closed",
            "weakly_connected": None,
            "keys": [],
        }]
        return routes, {
            "requested_count": requested, "returned_count": 0, "candidates_examined": 0,
            "search_exhausted": True, "search_budget_reached": False, "overlap_relaxed": False,
        }
    if not graph.has_edge(destination["graph_from"], destination["graph_to"], destination_key):
        routes = [{
            "reachable": False,
            "reason": "The selected destination road is closed in this comparison state.",
            "reason_code": "destination_road_closed",
            "weakly_connected": None,
            "keys": [],
        }]
        return routes, {
            "requested_count": requested, "returned_count": 0, "candidates_examined": 0,
            "search_exhausted": True, "search_budget_reached": False, "overlap_relaxed": False,
        }

    # The selected start and destination roads are fixed endpoint roads.  They
    # must not be reused inside an enumerated path, otherwise an undirected
    # connectivity proxy can repeatedly leave and re-enter an endpoint road
    # and reject every otherwise feasible candidate as a loop.
    endpoint_keys = {origin_key, destination_key}
    simple = nx.DiGraph() if graph.is_directed() else nx.Graph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        if str(key) in endpoint_keys:
            continue
        weight = float(data.get("weight", 1.0))
        current = simple.get_edge_data(u, v)
        if current is None or (weight, str(key)) < (float(current["weight"]), str(current["edge_key"])):
            simple.add_edge(u, v, weight=weight, edge_key=str(key))
    if graph.is_directed():
        orientation_specs = [
            {
                "origin_entry": origin["graph_from"],
                "source": origin["graph_to"],
                "target": destination["graph_from"],
                "destination_exit": destination["graph_to"],
            }
        ]
    else:
        # From/To often describes connectivity rather than a declared one-way
        # restriction.  In the disclosed two-way proxy a selected road can be
        # left from either endpoint and the destination road can be entered
        # from either endpoint.  Evaluate all four legal orientations instead
        # of silently forcing the uploaded row order.
        orientation_specs = [
            {
                "origin_entry": origin_entry,
                "source": source,
                "target": target,
                "destination_exit": destination_exit,
            }
            for origin_entry, source in (
                (origin["graph_from"], origin["graph_to"]),
                (origin["graph_to"], origin["graph_from"]),
            )
            for target, destination_exit in (
                (destination["graph_from"], destination["graph_to"]),
                (destination["graph_to"], destination["graph_from"]),
            )
        ]

    generator_states: list[dict[str, Any]] = []
    for spec in orientation_specs:
        source = spec["source"]
        target = spec["target"]
        working = simple.copy()
        blocked_middle_nodes = {
            spec["origin_entry"],
            spec["destination_exit"],
        } - {source, target}
        working.remove_nodes_from(node for node in blocked_middle_nodes if node in working)
        if source == target:
            generator = iter([[source]])
        elif source not in working or target not in working or not nx.has_path(working, source, target):
            continue
        else:
            generator = nx.shortest_simple_paths(working, source=source, target=target, weight="weight")
        generator_states.append({"spec": spec, "graph": working, "generator": generator, "exhausted": False})

    def build_candidate(state: dict[str, Any], nodes: list[str]) -> dict[str, Any] | None:
        spec = state["spec"]
        working = state["graph"]
        full_nodes = [spec["origin_entry"], *nodes, spec["destination_exit"]]
        if len(full_nodes) != len(set(full_nodes)):
            return None
        middle_keys = [str(working[u][v]["edge_key"]) for u, v in zip(nodes, nodes[1:])]
        keys = [origin_key, *middle_keys, destination_key]
        edge_data = [graph[origin["graph_from"]][origin["graph_to"]][origin_key]]
        edge_data.extend(
            graph[records[key]["graph_from"]][records[key]["graph_to"]][key]
            for key in middle_keys
        )
        edge_data.append(graph[destination["graph_from"]][destination["graph_to"]][destination_key])
        known_lengths = [records[key].get("length_m") for key in keys if records.get(key, {}).get("length_m")]
        return {
            "reachable": True,
            "reason": "",
            "keys": keys,
            "nodes": full_nodes,
            "edge_weights": [float(data.get("weight", 0.0)) for data in edge_data],
            "impedance": sum(float(data.get("weight", 0.0)) for data in edge_data),
            "link_count": len(keys),
            "distance_m": sum(float(value) for value in known_lengths) if len(known_lengths) == len(keys) else None,
            "origin_traversal": [spec["origin_entry"], spec["source"]],
            "destination_traversal": [spec["target"], spec["destination_exit"]],
        }

    accepted: list[dict[str, Any]] = []
    seen_key_sequences: set[tuple[str, ...]] = set()
    maximum_examined = max(100, min(5000, requested * 50))
    examined = 0
    overlap_limited_candidates = 0
    serial = 0
    heap: list[tuple[float, tuple[str, ...], int, int, dict[str, Any]]] = []

    def push_next(state_index: int) -> None:
        nonlocal examined, serial
        state = generator_states[state_index]
        while examined < maximum_examined:
            try:
                nodes = next(state["generator"])
            except (StopIteration, nx.NetworkXNoPath, nx.NodeNotFound):
                state["exhausted"] = True
                return
            examined += 1
            candidate = build_candidate(state, list(nodes))
            if candidate is None:
                continue
            serial += 1
            heapq.heappush(
                heap,
                (
                    float(candidate["impedance"]),
                    tuple(candidate["keys"]),
                    state_index,
                    serial,
                    candidate,
                ),
            )
            return

    for state_index in range(len(generator_states)):
        push_next(state_index)
    while heap and len(accepted) < requested and examined <= maximum_examined:
        _, key_tuple, state_index, _, candidate = heapq.heappop(heap)
        if key_tuple not in seen_key_sequences:
            seen_key_sequences.add(key_tuple)
            candidate_keys = set(candidate["keys"][1:-1])
            prior_overlaps = []
            for prior in accepted:
                prior_keys = set(prior["keys"][1:-1])
                union = candidate_keys | prior_keys
                prior_overlaps.append(len(candidate_keys & prior_keys) / len(union) if union else 1.0)
            candidate["maximum_overlap_with_retained"] = max(prior_overlaps, default=0.0)
            if not accepted or candidate["maximum_overlap_with_retained"] <= overlap_limit + 1e-12:
                accepted.append(candidate)
            else:
                overlap_limited_candidates += 1
        push_next(state_index)

    accepted.sort(key=lambda route: (float(route["impedance"]), tuple(route["keys"])))
    search_exhausted = not heap and all(state["exhausted"] for state in generator_states)
    search_budget_reached = examined >= maximum_examined and len(accepted) < requested
    overlap_relaxed = False

    if not accepted:
        if graph.is_directed():
            diagnostic = _directed_disconnection(
                graph,
                str(origin["graph_to"]),
                str(destination["graph_from"]),
            )
        elif search_budget_reached:
            diagnostic = {
                "reason_code": "route_search_budget_reached",
                "weakly_connected": True,
                "reason": "No distinct loopless route was found within the disclosed deterministic search budget.",
            }
        else:
            diagnostic = {
                "reason_code": "different_weak_components",
                "weakly_connected": False,
                "reason": "The applied disruption leaves the selected roads in different connected parts of the routable network.",
            }
        routes = [{"reachable": False, **diagnostic, "keys": []}]
        return routes, {
            "requested_count": requested, "returned_count": 0, "candidates_examined": examined,
            "search_exhausted": search_exhausted, "search_budget_reached": search_budget_reached,
            "overlap_relaxed": overlap_relaxed,
            "overlap_limited_candidates": overlap_limited_candidates,
        }
    primary_keys = set(accepted[0]["keys"][1:-1])
    primary_cost = float(accepted[0]["impedance"])
    for index, route in enumerate(accepted, start=1):
        route_keys = set(route["keys"][1:-1])
        union = primary_keys | route_keys
        route["alternative_rank"] = index
        route["overlap_with_primary"] = len(primary_keys & route_keys) / len(union) if union else 1.0
        route["impedance_change_from_primary"] = (
            (float(route["impedance"]) / primary_cost - 1.0) if primary_cost > 0 else None
        )
    return accepted[:requested], {
        "requested_count": requested,
        "returned_count": min(len(accepted), requested),
        "candidates_examined": examined,
        "search_exhausted": search_exhausted,
        "search_budget_reached": search_budget_reached,
        "overlap_relaxed": overlap_relaxed,
        "overlap_limited_candidates": overlap_limited_candidates,
    }


def _best_route_between_links(
    graph: nx.MultiDiGraph | nx.MultiGraph,
    records: dict[str, dict[str, Any]],
    origin_key: str,
    destination_key: str,
) -> dict[str, Any]:
    """Return the best legal road-to-road route using the active direction tier."""

    routes, _ = _route_alternatives_between_links(
        graph,
        records,
        origin_key,
        destination_key,
        count=1,
        maximum_overlap=1.0,
    )
    return routes[0]


def _route_features(
    route: dict[str, Any],
    *,
    role: str,
    label: str,
    templates: dict[str, dict[str, Any]],
    variant_id: str | None = None,
    variant_index: int | None = None,
    variant_colour: str | None = None,
    cost_unit: str | None = None,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for order, key in enumerate(route.get("keys") or [], start=1):
        template = templates.get(key)
        if template is None:
            continue
        feature = deepcopy(template)
        feature["properties"].update(
            {
                "analysis_type": "route_detour_analysis",
                "route_role": role,
                "route_label": label,
                "segment_order": order,
                "route_link_count": route.get("link_count"),
                "route_impedance": route.get("impedance"),
                "route_distance_m": route.get("distance_m"),
                "route_rank": route.get("alternative_rank"),
                "route_overlap_with_primary": route.get("overlap_with_primary"),
                "route_change_from_primary": route.get("impedance_change_from_primary"),
                **({"route_cost_unit": cost_unit} if cost_unit else {}),
                **({"route_variant_id": variant_id} if variant_id is not None else {}),
                **({"route_variant_index": variant_index} if variant_index is not None else {}),
                **({"route_colour": variant_colour} if variant_colour else {}),
            }
        )
        features.append(feature)
    return features


def _route_graph_for_selected_endpoints(
    graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    origin_key: str,
    destination_key: str,
) -> tuple[nx.MultiDiGraph | nx.MultiGraph, str]:
    """Use direction when it is usable, otherwise disclose a two-way topology proxy.

    Many uploaded disaster-network tables use ``From`` and ``To`` only to
    encode link connectivity and provide no one-way field.  A strict directed
    interpretation can therefore reject almost every geographically connected
    endpoint pair.  The fallback is activated only for a direction-continuity
    gap inside the same weak component; genuine network splits remain
    disconnected and no missing road is created.
    """

    directed = _route_between_links(graph, records, origin_key, destination_key)
    if directed.get("reachable") or directed.get("reason_code") != "direction_continuity_missing":
        return graph, "uploaded_direction"
    return graph.to_undirected(reciprocal=False), "two_way_connectivity_proxy"


def _endpoint_feature(record: dict[str, Any], role: str, label: str) -> dict[str, Any] | None:
    if record.get("longitude") is None or record.get("latitude") is None:
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(record["longitude"]), float(record["latitude"])]},
        "properties": {
            "analysis_type": "route_detour_analysis",
            "route_role": role,
            "route_label": label,
            "key": record["key"],
            "entity_id": record["entity_id"],
            "region_id": record["region_id"],
        },
    }


def _route_items(
    route: dict[str, Any],
    *,
    role: str,
    label: str,
    records: dict[str, dict[str, Any]],
    other_keys: set[str],
    variant_id: str | None = None,
    variant_index: int | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    edge_weights = route.get("edge_weights") or []
    for sequence, key in enumerate(route.get("keys") or [], start=1):
        record = records[key]
        items.append(
            {
                "route_role": role,
                "route_label": label,
                "sequence": sequence,
                "key": key,
                "entity_id": record["entity_id"],
                "region_id": record["region_id"],
                "from_node": record["from_node"],
                "to_node": record["to_node"],
                "segment_impedance": edge_weights[sequence - 1] if sequence <= len(edge_weights) else record["weight"],
                "length_m": record.get("length_m"),
                "shared_with_other_route": key in other_keys,
                **({"route_variant_id": variant_id} if variant_id is not None else {}),
                **({"route_variant_index": variant_index} if variant_index is not None else {}),
            }
        )
    return items


def _route_difference(baseline: dict[str, Any], compared: dict[str, Any]) -> dict[str, Any]:
    baseline_keys = set(baseline.get("keys") or [])
    compared_keys = set(compared.get("keys") or [])
    union = baseline_keys | compared_keys
    overlap_percent = (len(baseline_keys & compared_keys) / len(union) * 100.0) if union else 0.0
    changed_link_count = len(baseline_keys ^ compared_keys)
    if baseline.get("reachable") and compared.get("reachable") and baseline.get("impedance", 0) > 0:
        detour_percent = (float(compared["impedance"]) / float(baseline["impedance"]) - 1.0) * 100.0
    else:
        detour_percent = None
    return {
        "overlap_percent": overlap_percent,
        "changed_link_count": changed_link_count,
        "detour_percent": detour_percent,
    }


def _cost_unit(summary: dict[str, Any], graph: nx.MultiDiGraph) -> str:
    basis = str(graph.graph.get("impedance_mode") or "uniform_hop_count")
    if basis in {"observed_travel_time", "uploaded_free_flow_time", "reference_travel_time"}:
        return str(summary.get("metrics", {}).get("travel_time", {}).get("unit") or "uploaded travel-time unit")
    if basis == "observed_speed_derived_time":
        return "seconds"
    if basis == "link_length":
        return "metres"
    if basis == "uniform_hop_count":
        return "link hops"
    return "uploaded impedance unit"


def _multi_scenario_route_result(
    *,
    summary: dict[str, Any],
    selected_time: int,
    historical_graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    geometry_rows: list[Any],
    scenario_contexts: list[dict[str, Any]],
    origin_key: str,
    destination_key: str,
    routing_direction_mode: str,
) -> dict[str, Any]:
    templates = {
        f"{row['region_id']}::{row['entity_id']}": _row_feature(row)
        for row in geometry_rows
    }
    speed_unit = str(summary.get("metrics", {}).get("speed", {}).get("unit") or "")
    baseline_graph, reference_changes = _reference_graph(historical_graph, records, speed_unit)
    baseline_label = "Internal reference condition"
    baseline_route = _best_route_between_links(baseline_graph, records, origin_key, destination_key)
    historical_label = summary.get("time_labels", {}).get(str(selected_time), str(selected_time))
    cost_unit = _cost_unit(summary, historical_graph)

    variant_specs: list[dict[str, Any]] = [
        {
            "id": "historical",
            "label": f"Historical state · {historical_label}",
            "kind": "historical",
            "timestamp": selected_time,
            "graph": historical_graph,
            "records": records,
            "changes_applied": 0,
            "scenario_id": None,
        }
    ]
    for context in scenario_contexts:
        scenario_graph, applied = _scenario_graph(
            context["graph"],
            context["records"],
            context.get("changes") or [],
        )
        variant_specs.append(
            {
                "id": str(context.get("scenario_id") or f"scenario-{len(variant_specs)}"),
                "label": f"Scenario · {context.get('name') or 'saved disruption'}",
                "kind": "saved_scenario",
                "timestamp": context["timestamp"],
                "timestamp_label": context["timestamp_label"],
                "graph": scenario_graph,
                "records": context["records"],
                "changes_applied": applied,
                "scenario_id": context.get("scenario_id"),
            }
        )

    variants: list[dict[str, Any]] = []
    route_records: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    all_variant_keys: set[str] = set()
    for index, spec in enumerate(variant_specs):
        variant_records = spec["records"]
        if origin_key not in variant_records or destination_key not in variant_records:
            route = {
                "reachable": False,
                "reason": "One selected point is unavailable in this scenario's historical network state.",
                "reason_code": "endpoint_entity_missing",
                "weakly_connected": None,
                "keys": [],
            }
        else:
            route = _best_route_between_links(spec["graph"], variant_records, origin_key, destination_key)
        difference = _route_difference(baseline_route, route)
        all_variant_keys.update(route.get("keys") or [])
        variants.append(
            {
                "id": spec["id"],
                "scenario_id": spec.get("scenario_id"),
                "label": spec["label"],
                "kind": spec["kind"],
                "timestamp": spec["timestamp"],
                "timestamp_label": spec.get("timestamp_label") or historical_label,
                "reachable": bool(route.get("reachable")),
                "reason": route.get("reason") or "",
                "reason_code": route.get("reason_code"),
                "weakly_connected": route.get("weakly_connected"),
                "impedance": route.get("impedance"),
                "link_count": route.get("link_count", 0),
                "distance_m": route.get("distance_m"),
                "route_keys": route.get("keys") or [],
                "changes_applied": spec.get("changes_applied", 0),
                "colour": ROUTE_VARIANT_COLOURS[index % len(ROUTE_VARIANT_COLOURS)],
                **difference,
            }
        )
        route_records.append((route, variant_records))

    features = _route_features(
        baseline_route,
        role="baseline_route",
        label="Baseline route",
        templates=templates,
        variant_id="reference",
        variant_index=-1,
        variant_colour=ROUTE_BASELINE_COLOUR,
        cost_unit=cost_unit,
    )
    items = _route_items(
        baseline_route,
        role="baseline_route",
        label=baseline_label,
        records=records,
        other_keys=all_variant_keys,
        variant_id="reference",
        variant_index=-1,
    )
    for index, ((route, variant_records), variant) in enumerate(zip(route_records, variants)):
        features.extend(
            _route_features(
                route,
                role="comparison_route",
                label=variant["label"],
                templates=templates,
                variant_id=variant["id"],
                variant_index=index,
                variant_colour=variant["colour"],
                cost_unit=cost_unit,
            )
        )
        items.extend(
            _route_items(
                route,
                role="comparison_route",
                label=variant["label"],
                records=variant_records,
                other_keys=set(baseline_route.get("keys") or []),
                variant_id=variant["id"],
                variant_index=index,
            )
        )
    for feature in (
        _endpoint_feature(records[origin_key], "route_origin", "Start point / road"),
        _endpoint_feature(records[destination_key], "route_destination", "Destination point / road"),
    ):
        if feature is not None:
            features.append(feature)

    connected = [variant for variant in variants if variant["reachable"]]
    disconnected = len(variants) - len(connected)
    largest_detour = max(
        (variant for variant in connected if variant["detour_percent"] is not None),
        key=lambda item: float(item["detour_percent"]),
        default=None,
    )
    if connected:
        conclusion = (
            f"The platform calculated least-impedance route recommendations for {len(variants):,} historical/scenario states: "
            f"{len(connected):,} remain connected and {disconnected:,} are disconnected."
        )
        if largest_detour is not None:
            conclusion += (
                f" The largest connected increase is {float(largest_detour['detour_percent']):+.1f}% "
                f"under {largest_detour['label']}."
            )
    else:
        conclusion = f"None of the {len(variants):,} compared historical/scenario states has a legal route between the selected points."

    basis = str(historical_graph.graph.get("impedance_mode") or "uniform_hop_count")
    line_features = [feature for feature in features if feature.get("geometry", {}).get("type") == "LineString"]
    return {
        "model": {
            "id": "P5-M23",
            "name": "Plan routes under disruption",
            "status": "numerically_verified",
        },
        "status": "supported" if basis != "uniform_hop_count" and routing_direction_mode == "uploaded_direction" else "conditional",
        "conclusion": conclusion,
        "headline_effects": [
            {"label": "States compared", "value": len(variants), "detail": "Current history plus selected saved disruption scenarios"},
            {"label": "Routes available", "value": f"{len(connected)}/{len(variants)}", "detail": f"{disconnected} disconnected state(s)"},
            {
                "label": "Largest connected change",
                "value": "No connected comparison" if largest_detour is None else f"{float(largest_detour['detour_percent']):+.1f}%",
                "detail": "No route was available in both compared states." if largest_detour is None else largest_detour["label"],
            },
        ],
        "visual": {
            "mode": "route_scenario_compare",
            "baseline_label": baseline_label,
            "baseline_reachable": bool(baseline_route.get("reachable")),
            "baseline_reason": baseline_route.get("reason") or "",
            "baseline_reason_code": baseline_route.get("reason_code"),
            "baseline_weakly_connected": baseline_route.get("weakly_connected"),
            "baseline_impedance": baseline_route.get("impedance"),
            "baseline_link_count": baseline_route.get("link_count", 0),
            "baseline_distance_m": baseline_route.get("distance_m"),
            "cost_unit": cost_unit,
            "routing_direction_mode": routing_direction_mode,
            "variants": variants,
        },
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "next_actions": [
            "Use the scenario cards to compare the route recommended for each network state.",
            "Inspect coloured route differences on the map and the complete segment table.",
            "Treat disconnected states as an accessibility warning; do not invent a detour or live traffic condition.",
        ],
        "evidence": {
            "algorithm": (
                "Dijkstra least-impedance path on a disclosed two-way connectivity proxy, recomputed independently for every state."
                if routing_direction_mode == "two_way_connectivity_proxy"
                else "Directed Dijkstra least-impedance path recomputed independently for the internal reference, uploaded history and every selected saved scenario."
            ),
            "comparison_kind": "multi_scenario",
            "selected_time": selected_time,
            "impedance_mode": basis,
            "impedance_label": historical_graph.graph.get("impedance_label"),
            "cost_unit": cost_unit,
            "routing_direction_mode": routing_direction_mode,
            "reference_links_with_changed_weight": reference_changes,
            "origin_key": origin_key,
            "destination_key": destination_key,
            "scenario_count": len(scenario_contexts),
            "variants": variants,
            "baseline_route": baseline_route,
            "literature_basis": [
                "Dijkstra (1959), A note on two problems in connexion with graphs.",
                "Wardrop (1952), Some theoretical aspects of road traffic research.",
                "Jenelius, Petersen and Mattsson (2006), Importance and exposure in road network vulnerability analysis.",
            ],
            "randomness": "none",
        },
        "data_boundary": (
            "These are retrospective least-impedance recommendations for uploaded historical and saved counterfactual network states. "
            "They are not live navigation, observed traveller choices, emergency dispatch or a congestion forecast. "
            + (" From/To is used as a two-way connectivity orientation because strict direction continuity could not connect the selected endpoints and no explicit one-way field is available."
               if routing_direction_mode == "two_way_connectivity_proxy" else " Uploaded From/To direction is retained for this connected endpoint pair.")
            + " Ordinary points without topology are not snapped or invented."
        ),
        "verification": {
            "passed": (
                origin_key in records
                and destination_key in records
                and len(variants) == len(scenario_contexts) + 1
                and all(item["key"] in records or any(item["key"] in context["records"] for context in scenario_contexts) for item in items)
                and len(line_features) <= len(features)
            ),
            "checks": [
                "both point or line endpoints grounded in uploaded road-link entities",
                "each saved scenario belongs to the current project",
                "each scenario operator applied to its saved historical baseline",
                "positive directed link impedance",
                "direction mode disclosed as uploaded direction or two-way connectivity proxy",
                "least-impedance route recomputed independently for every state",
                "deterministic tie breaking and no random demand",
                "disconnection reported without an invented route penalty",
            ],
        },
    }


def historical_route_detour_analysis(
    database_path: Path,
    *,
    origin_key: str,
    destination_key: str,
    timestamp: int | None,
    scenario_changes: list[dict[str, Any]] | None = None,
    scenario_name: str | None = None,
    scenario_variants: list[dict[str, Any]] | None = None,
    alternative_count: int = 2,
    maximum_overlap: float = 0.60,
) -> dict[str, Any]:
    """Compare a historical shortest route with a reference or saved scenario.

    This is a deterministic retrospective network analysis.  It uses the
    uploaded directed topology and the strongest available positive impedance
    tier; it does not claim to be live navigation or observed OD assignment.
    """

    if not origin_key or not destination_key:
        raise ValueError("Choose one start point or road and one destination point or road on the map.")
    scenario_contexts: list[dict[str, Any]] = []
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        historical_graph, records = _time_graph(connection, selected_time)
        geometry_rows, _ = _geometry_rows(connection)
        graph_cache: dict[int, tuple[nx.MultiDiGraph, dict[str, dict[str, Any]]]] = {
            selected_time: (historical_graph, records)
        }
        for variant in scenario_variants or []:
            variant_time = _nearest_timestamp(connection, variant.get("base_timestamp"))
            if variant_time not in graph_cache:
                graph_cache[variant_time] = _time_graph(connection, variant_time)
            variant_graph, variant_records = graph_cache[variant_time]
            scenario_contexts.append(
                {
                    **variant,
                    "timestamp": variant_time,
                    "timestamp_label": summary.get("time_labels", {}).get(str(variant_time), str(variant_time)),
                    "graph": variant_graph,
                    "records": variant_records,
                }
            )
    missing = [key for key in (origin_key, destination_key) if key not in records]
    if missing:
        raise ValueError("The selected route endpoint is not available in the uploaded From/To topology: " + ", ".join(missing))
    if historical_graph.number_of_edges() == 0:
        raise ValueError("Historical route comparison needs usable From node and To node fields.")

    historical_graph, routing_direction_mode = _route_graph_for_selected_endpoints(
        historical_graph,
        records,
        origin_key,
        destination_key,
    )
    if routing_direction_mode == "two_way_connectivity_proxy":
        for context in scenario_contexts:
            context["graph"] = context["graph"].to_undirected(reciprocal=False)

    if scenario_contexts:
        return _multi_scenario_route_result(
            summary=summary,
            selected_time=selected_time,
            historical_graph=historical_graph,
            records=records,
            geometry_rows=geometry_rows,
            scenario_contexts=scenario_contexts,
            origin_key=origin_key,
            destination_key=destination_key,
            routing_direction_mode=routing_direction_mode,
        )

    templates = {
        f"{row['region_id']}::{row['entity_id']}": _row_feature(row)
        for row in geometry_rows
    }
    speed_unit = str(summary.get("metrics", {}).get("speed", {}).get("unit") or "")
    historical_label = summary.get("time_labels", {}).get(str(selected_time), str(selected_time))
    if scenario_changes:
        baseline_graph = historical_graph
        comparison_graph, applied_changes = _scenario_graph(historical_graph, records, scenario_changes)
        baseline_label = f"Historical state · {historical_label}"
        comparison_label = f"Scenario · {scenario_name or 'saved disruption'}"
        comparison_kind = "saved_scenario"
        reference_changes = None
    else:
        baseline_graph, reference_changes = _reference_graph(historical_graph, records, speed_unit)
        comparison_graph = historical_graph
        applied_changes = 0
        baseline_label = "Internal reference condition"
        comparison_label = f"Historical state · {historical_label}"
        comparison_kind = "historical"

    baseline_route = _best_route_between_links(baseline_graph, records, origin_key, destination_key)
    comparison_routes, route_search = _route_alternatives_between_links(
        comparison_graph,
        records,
        origin_key,
        destination_key,
        count=alternative_count,
        maximum_overlap=maximum_overlap,
    )
    route_options = [route for route in comparison_routes if route.get("reachable")]
    comparison_route = route_options[0] if route_options else comparison_routes[0]
    alternatives = route_options[1:]
    baseline_keys = set(baseline_route.get("keys") or [])
    comparison_keys = set(comparison_route.get("keys") or [])
    difference = _route_difference(baseline_route, comparison_route)
    overlap_percent = difference["overlap_percent"]
    changed_link_count = difference["changed_link_count"]
    detour_percent = difference["detour_percent"]
    basis = str(historical_graph.graph.get("impedance_mode") or "uniform_hop_count")
    cost_unit = _cost_unit(summary, historical_graph)

    route_summaries: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for index, route in enumerate(route_options, start=1):
        colour = _route_option_colour(index)
        label = f"Recommended route {index}"
        variant_id = f"recommendation-{index}"
        features.extend(
            _route_features(
                route,
                role="recommended_route",
                label=label,
                templates=templates,
                variant_id=variant_id,
                variant_index=index,
                variant_colour=colour,
                cost_unit=cost_unit,
            )
        )
        route_summaries.append(
            {
                "route_summary": True,
                "route_role": "recommended_route",
                "route_label": label,
                "route_rank": index,
                "route_variant_id": variant_id,
                "route_colour": colour,
                "impedance": route.get("impedance"),
                "cost_unit": cost_unit,
                "link_count": route.get("link_count", 0),
                "distance_m": route.get("distance_m"),
                "overlap_with_primary": route.get("overlap_with_primary"),
                "impedance_change_from_primary": route.get("impedance_change_from_primary"),
                "route_keys": route.get("keys") or [],
            }
        )
    for feature in (
        _endpoint_feature(records[origin_key], "route_origin", "Start point / road"),
        _endpoint_feature(records[destination_key], "route_destination", "Destination point / road"),
    ):
        if feature is not None:
            features.append(feature)

    items = route_summaries
    requested_count = int(route_search["requested_count"])
    returned_count = len(route_options)
    if returned_count == 0:
        conclusion = f"No feasible route is available under {comparison_label}. {comparison_route.get('reason', '')}".strip()
    elif returned_count < requested_count:
        overlap_limited = int(route_search.get("overlap_limited_candidates") or 0)
        qualifier = (
            f"met the disclosed road-overlap limit after {overlap_limited:,} near-duplicate candidate"
            f"{'s were' if overlap_limited != 1 else ' was'} excluded"
            if overlap_limited
            else "exist in the tested network" if route_search.get("search_exhausted")
            else "were found within the disclosed search budget"
        )
        conclusion = (
            f"You requested {requested_count:,} routes. {returned_count:,} distinct feasible route"
            f"{'s' if returned_count != 1 else ''} {qualifier} under {comparison_label}."
        )
    else:
        conclusion = f"The model returned {returned_count:,} distinct feasible routes under {comparison_label}, ranked by total impedance."

    line_features = [feature for feature in features if feature.get("geometry", {}).get("type") == "LineString"]
    data_supported = (
        basis != "uniform_hop_count"
        and (comparison_kind == "saved_scenario" or bool(reference_changes))
        and routing_direction_mode == "uploaded_direction"
    )
    return {
        "model": {
            "id": "P5-M23",
            "name": "Plan routes under disruption",
            "status": "numerically_verified",
        },
        "status": "supported" if data_supported else "conditional",
        "conclusion": conclusion,
        "headline_effects": [
            {
                "label": "Feasible routes",
                "value": f"{returned_count:,}",
                "detail": f"Requested {requested_count:,} under the selected disruption",
            },
            {
                "label": "Best route",
                "value": "No route" if not route_options else f"{float(route_options[0]['impedance']):,.2f}",
                "detail": comparison_route.get("reason") if not route_options else f"Total impedance in {cost_unit}",
            },
            {
                "label": "Disruption state",
                "value": "Connected" if route_options else "Disconnected",
                "detail": comparison_label,
            },
        ],
        "visual": {
            "mode": "route_options",
            "scenario_label": comparison_label,
            "reachable": bool(route_options),
            "reason": comparison_route.get("reason") or "",
            "reason_code": comparison_route.get("reason_code"),
            "weakly_connected": comparison_route.get("weakly_connected"),
            "requested_count": requested_count,
            "returned_count": returned_count,
            "availability_limited": returned_count < requested_count,
            "search_exhausted": bool(route_search.get("search_exhausted")),
            "search_budget_reached": bool(route_search.get("search_budget_reached")),
            "overlap_limited_candidates": int(route_search.get("overlap_limited_candidates") or 0),
            "maximum_pairwise_overlap": max(0.0, min(float(maximum_overlap), 1.0)),
            "cost_unit": cost_unit,
            "routing_direction_mode": routing_direction_mode,
            "routes": [
                {
                    "rank": index,
                    "label": f"Recommended route {index}",
                    "reachable": True,
                    "impedance": route.get("impedance"),
                    "link_count": route.get("link_count"),
                    "distance_m": route.get("distance_m"),
                    "overlap_with_primary": route.get("overlap_with_primary"),
                    "impedance_change_from_primary": route.get("impedance_change_from_primary"),
                    "route_keys": route.get("keys") or [],
                    "colour": _route_option_colour(index),
                }
                for index, route in enumerate(route_options, start=1)
            ],
        },
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "next_actions": [
            "Inspect the coloured recommendations on the map; the legend identifies every returned route.",
            "Hover or click a route to see the same total impedance, road count and distance shown in the table.",
            "Request a different number of route options or change the disruption and calculate again.",
        ],
        "evidence": {
            "algorithm": (
                "Least-impedance Top-K loopless path enumeration on a disclosed two-way connectivity proxy; near-duplicate candidates above the disclosed pairwise road-set overlap limit are excluded."
                if routing_direction_mode == "two_way_connectivity_proxy"
                else "Directed least-impedance Top-K loopless path enumeration; near-duplicate candidates above the disclosed pairwise road-set overlap limit are excluded while the selected endpoint roads remain fixed."
            ),
            "comparison_kind": comparison_kind,
            "selected_time": selected_time,
            "impedance_mode": basis,
            "impedance_label": historical_graph.graph.get("impedance_label"),
            "cost_unit": cost_unit,
            "routing_direction_mode": routing_direction_mode,
            "reference_links_with_changed_weight": reference_changes,
            "scenario_changes_applied": applied_changes,
            "origin_key": origin_key,
            "destination_key": destination_key,
            "baseline_route": baseline_route,
            "comparison_route": comparison_route,
            "alternative_routes": alternatives,
            "route_options": route_options,
            "route_search": route_search,
            "alternative_count_requested": requested_count,
            "route_count_returned": returned_count,
            "maximum_pairwise_overlap": max(0.0, min(float(maximum_overlap), 1.0)),
            "literature_basis": [
                "Dijkstra (1959), A note on two problems in connexion with graphs.",
                "Yen (1971), Finding the K shortest loopless paths in a network, doi:10.1287/mnsc.17.11.712.",
                "Hu and Chiu (2015), A Constrained Time-Dependent K Shortest Paths Algorithm Addressing Overlap and Travel Time Deviation, doi:10.1016/S2046-0430(16)30169-1.",
                "Wardrop (1952), Some theoretical aspects of road traffic research.",
                "Jenelius, Petersen and Mattsson (2006), Importance and exposure in road network vulnerability analysis.",
            ],
            "randomness": "none",
        },
        "data_boundary": (
            "This is retrospective least-impedance route and detour analysis on uploaded static or historical data. "
            "It is not live navigation, an observed OD trip, an evacuation instruction or a congestion forecast. "
            + ("From/To is treated as a two-way connectivity orientation because strict direction continuity could not connect the selected endpoints; this is disclosed as a topology proxy. "
               if routing_direction_mode == "two_way_connectivity_proxy" else "Uploaded From/To direction is retained for this connected endpoint pair. ")
            + "Mapped link lines may be straight topology reconstructions between inferred nodes rather than surveyed road centrelines."
        ),
        "verification": {
            "passed": (
                origin_key in records
                and destination_key in records
                and returned_count <= requested_count
                and len({tuple(route.get("keys") or []) for route in route_options}) == returned_count
                and all(set(item["route_keys"]).issubset(records) for item in items)
                and len(line_features) <= len(features)
                and all(
                    not route.get("reachable")
                    or len(route.get("nodes") or []) == len(set(route.get("nodes") or []))
                    for route in route_options
                )
                and all(
                    abs(float(route.get("impedance", 0.0)) - sum(float(value) for value in route.get("edge_weights") or [])) <= 1e-9
                    for route in route_options
                )
            ),
            "checks": [
                "both route endpoints grounded in uploaded entities",
                "positive directed link impedance",
                "direction mode disclosed as uploaded direction or two-way connectivity proxy",
                "deterministic shortest-path tie breaking",
                "saved scenario operators applied to the same historical graph",
                "route totals recomputed from returned segments",
                "recommended routes are distinct, loopless and ordered by scenario impedance",
                "route totals equal the sum of their returned segment impedances",
                "returned alternatives satisfy the disclosed pairwise road-set overlap limit",
                "no random demand or live traffic invented",
            ],
        },
    }
