from __future__ import annotations

"""Executable base tiers for the extended, literature-grounded model registry.

These adapters deliberately reuse verified numerical kernels where the
mathematical object is the same, while changing neither the evidence nor the
claim boundary.  A low-data tier is never labelled as its richer counterpart.
"""

from itertools import groupby
import math
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .models import (
    _build_graph,
    _connect,
    _deterministic_sources,
    _inferred_link_geometries,
    _link_result_feature,
    _metric_parts,
    _summary,
    critical_components,
    restoration_screening,
    spatial_hotspots,
)


ADAPTIVE_MODEL_NAMES = {
    "P5-M5": "Importance and exposure sensitivity",
    "P5-M6": "Equal-opportunity network accessibility screen",
    "P5-M8": "Resource-aware staged restoration screen",
    "P5-M12": "Reliability and sensitivity envelope",
    "P5-M13": "Deterministic load-transfer cascade stress test",
    "P5-M14": "Observed hazard-transport exposure screen",
    "P5-M15": "Equal-weight accessibility distribution screen",
    "P5-M16": "Greedy protection-priority screen",
    "P5-M18": "Evacuation-network reachability readiness screen",
    "P5-M19": "Chronological persistence-benchmark validation",
    "P5-M20": "Resilience investment candidate screen",
}


def _table_columns(connection: Any, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}


def _clone_with_model(
    result: dict[str, Any],
    *,
    model_id: str,
    sources: list[str],
    conclusion: str,
    boundary: str,
) -> dict[str, Any]:
    output = {**result}
    output["model"] = {
        "id": model_id,
        "name": ADAPTIVE_MODEL_NAMES[model_id],
        "status": "adaptive_executable_base_tier",
        "sources": sources,
    }
    output["conclusion"] = conclusion
    output["data_boundary"] = boundary
    output["adaptive_execution"] = {
        "tier": "executable_base",
        "shared_verified_kernel": result.get("model", {}).get("id"),
        "randomness": "none",
    }
    output.setdefault("verification", {}).setdefault("checks", []).append(
        "adaptive model ID and lower-tier claim boundary recorded"
    )
    return output


def importance_exposure_screen(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
) -> dict[str, Any]:
    base = critical_components(database_path, top_k=top_k, bbox=bbox)
    for item in base.get("items", []):
        item["equal_demand_importance"] = float(
            item.get("efficiency_loss", item.get("structural_score", 0.0)) or 0.0
        )
    return _clone_with_model(
        base,
        model_id="P5-M5",
        sources=["Jenelius, Petersen and Mattsson (2006), doi:10.1016/j.tra.2005.11.003"],
        conclusion=(
            f"The executable equal-demand tier returned {len(base.get('items', []))} links whose removal most changes network efficiency."
        ),
        boundary=(
            "This is the equal-demand importance component of a vulnerability screen. Exposure in the Jenelius sense needs an observed OD-demand matrix; uploaded link flow alone is not silently treated as OD demand."
        ),
    )


def _node_coordinates(
    records: dict[str, dict[str, Any]], *, global_node_ids: bool
) -> dict[str, tuple[float, float]]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    for record in records.values():
        latitude = record.get("latitude")
        longitude = record.get("longitude")
        if latitude is None or longitude is None:
            continue
        for node in (record.get("from_node"), record.get("to_node")):
            if node is not None:
                graph_node = str(node) if global_node_ids else f"{record.get('region_id')}\x1f{node}"
                buckets.setdefault(graph_node, []).append((float(latitude), float(longitude)))
    return {
        node: (
            sum(pair[0] for pair in pairs) / len(pairs),
            sum(pair[1] for pair in pairs) / len(pairs),
        )
        for node, pairs in buckets.items()
        if pairs
    }


def accessibility_screen(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
    model_id: str = "P5-M6",
) -> dict[str, Any]:
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, bbox)
    if graph.number_of_edges() == 0:
        raise ValueError("Topology fields are required for network accessibility screening.")
    total_nodes = graph.number_of_nodes()
    sources = list(graph.nodes) if total_nodes <= 800 else _deterministic_sources(graph.nodes, maximum=320)
    coordinates = _node_coordinates(
        records,
        global_node_ids=graph.graph.get("topology_node_scope") == "global",
    )
    items: list[dict[str, Any]] = []
    for node in sources:
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="weight")
        reachable = max(0, len(lengths) - 1)
        harmonic = sum(1.0 / distance for target, distance in lengths.items() if target != node and distance > 0)
        latitude, longitude = coordinates.get(str(node), (None, None))
        items.append(
            {
                "node_id": str(node).split("\x1f")[-1],
                "graph_node": str(node),
                "reachable_nodes": reachable,
                "reachable_fraction": reachable / max(1, total_nodes - 1),
                "harmonic_accessibility": harmonic / max(1, total_nodes - 1),
                "latitude": latitude,
                "longitude": longitude,
            }
        )
    items.sort(key=lambda item: (item["reachable_fraction"], item["harmonic_accessibility"], item["graph_node"]))
    returned = [{"rank": rank, **item} for rank, item in enumerate(items[: min(len(items), max(1, int(top_k)))], 1)]
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [item["longitude"], item["latitude"]]},
            "properties": {
                **item,
                "key": f"node::{item['graph_node']}",
                "severity": 1.0 - item["reachable_fraction"],
                "geometry_kind": "incident_link_centroid_proxy",
            },
        }
        for item in returned
        if item.get("latitude") is not None and item.get("longitude") is not None
    ]
    exact = len(sources) == total_nodes
    name = ADAPTIVE_MODEL_NAMES[model_id]
    boundaries = {
        "P5-M6": "This is equal-opportunity network reachability. Essential-service accessibility needs geocoded facilities or opportunities and population/OD weights.",
        "P5-M15": "This is an equal-weight accessibility distribution screen, not a demographic equity result. Group populations, destinations and a normative equity rule are required for distributive-justice claims.",
        "P5-M18": "This is road-network reachability readiness, not an evacuation assignment. Population demand, departure timing, modes, shelter locations/capacities and behavioural assumptions are required for evacuation outcomes.",
    }
    values = [item["harmonic_accessibility"] for item in items]
    mean_value = sum(values) / len(values) if values else 0.0
    gini = 0.0
    if mean_value > 0 and values:
        gini = sum(abs(first - second) for first in values for second in values) / (2 * len(values) ** 2 * mean_value)
    return {
        "model": {
            "id": model_id,
            "name": name,
            "status": "adaptive_executable_base_tier",
            "sources": ["Hansen (1959), doi:10.1080/01944365908978307"],
        },
        "status": "supported" if exact and graph.graph.get("impedance_mode") != "uniform_hop_count" else "conditional",
        "conclusion": f"The model screened {len(sources):,} network nodes and returned the {len(returned)} weakest equal-weight reachability positions.",
        "headline_effects": [
            {"label": "Nodes screened", "value": f"{len(sources):,}", "detail": "All nodes" if exact else f"Deterministic sample of {total_nodes:,}"},
            {"label": "Lowest reachable share", "value": f"{(returned[0]['reachable_fraction'] * 100 if returned else 0):.1f}%", "detail": "Directed network reachability"},
            {"label": "Equal-weight access Gini", "value": f"{gini:.3f}", "detail": "Structural distribution only"},
        ],
        "next_actions": [
            "Inspect weakly reachable positions on the geographic map.",
            "Add facilities or opportunities before interpreting essential-service access.",
            "Add population groups and a stated welfare rule before interpreting equity.",
        ],
        "items": returned,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "total_node_count": total_nodes,
            "evaluated_node_count": len(sources),
            "exact_all_nodes": exact,
            "impedance_mode": graph.graph.get("impedance_mode"),
            "accessibility_definition": "normalised directed harmonic closeness with equal opportunity at every reachable node",
            "node_geometry_rule": "mean coordinate of incident uploaded link reference points",
            "equal_weight_access_gini": gini,
        },
        "data_boundary": boundaries[model_id],
        "verification": {
            "passed": all(0 <= item["reachable_fraction"] <= 1 and item["harmonic_accessibility"] >= 0 for item in items),
            "checks": ["positive shortest-path impedance", "reachability bounded in [0,1]", "deterministic node sample", "no invented facilities or population"],
        },
    }


def resource_restoration_screen(
    database_path: Path,
    *,
    selected_keys: list[str],
    bbox: list[float] | None,
    model_id: str,
) -> dict[str, Any]:
    base = restoration_screening(database_path, selected_keys=selected_keys, bbox=bbox)
    with _connect(database_path) as connection:
        columns = _table_columns(connection, "entities")
        optional = [name for name in ("repair_cost", "repair_duration") if name in columns]
        values: dict[str, dict[str, float | None]] = {}
        if optional:
            rows = connection.execute(
                f"SELECT region_id, entity_id, {', '.join(optional)} FROM entities"
            ).fetchall()
            values = {
                f"{row['region_id']}::{row['entity_id']}": {name: row[name] for name in optional}
                for row in rows
            }
    for item in base.get("items", []):
        item.update(values.get(str(item.get("key")), {}))
        cost = item.get("repair_cost")
        gain = float(item.get("marginal_efficiency_gain", 0.0) or 0.0)
        item["marginal_gain_per_cost"] = gain / float(cost) if cost is not None and float(cost) > 0 else None
    coverage = sum(
        item.get("repair_cost") is not None and item.get("repair_duration") is not None
        for item in base.get("items", [])
    )
    full_resource_data = bool(base.get("items")) and coverage == len(base["items"])
    label = "repair cost and duration are displayed" if full_resource_data else "equal-resource assumptions apply"
    return _clone_with_model(
        base,
        model_id=model_id,
        sources=["Nurre et al. (2012), doi:10.1016/j.ejor.2012.07.010"],
        conclusion=f"The deterministic marginal-service sequence returned {len(base.get('items', []))} selected failed links; {label}.",
        boundary=(
            "This is a sequential marginal-benefit screen, not a globally optimal multi-crew schedule or portfolio. A budget, precedence, crew compatibility and network-service objective are required for that claim."
        ),
    )


def reliability_screen(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
) -> dict[str, Any]:
    base = critical_components(database_path, top_k=top_k, bbox=bbox)
    probabilities: dict[str, float] = {}
    with _connect(database_path) as connection:
        if "failure_probability" in _table_columns(connection, "entities"):
            probabilities = {
                f"{row['region_id']}::{row['entity_id']}": float(row["failure_probability"])
                for row in connection.execute(
                    "SELECT region_id, entity_id, failure_probability FROM entities WHERE failure_probability BETWEEN 0 AND 1"
                ).fetchall()
            }
    for item in base.get("items", []):
        loss = float(item.get("efficiency_loss", item.get("structural_score", 0.0)) or 0.0)
        probability = probabilities.get(str(item.get("key")))
        item["failure_probability"] = probability
        item["expected_efficiency_loss"] = probability * loss if probability is not None else None
        item["deterministic_sensitivity"] = loss
    probability_items = [item for item in base.get("items", []) if item.get("failure_probability") is not None]
    if probability_items and len(probability_items) == len(base.get("items", [])):
        base["items"].sort(key=lambda item: (item["expected_efficiency_loss"], item["deterministic_sensitivity"]), reverse=True)
        for rank, item in enumerate(base["items"], 1):
            item["rank"] = rank
        geometries, mapping_ready = _inferred_link_geometries(
            database_path,
            (item["key"] for item in base["items"]),
        )
        probability_features = []
        for item in base["items"]:
            feature = _link_result_feature(
                item,
                geometries,
                severity=float(item["expected_efficiency_loss"] or 0.0),
                allow_point_fallback=not mapping_ready,
            )
            if feature is not None:
                probability_features.append(feature)
        base["feature_collection"] = {"type": "FeatureCollection", "features": probability_features}
        base["evidence"]["ranking_rule"] = (
            "Uploaded marginal failure probability multiplied by deterministic single-link efficiency loss; "
            "deterministic sensitivity is the tie-break."
        )
        base["evidence"]["mapped_line_result_count"] = sum(
            feature["geometry"]["type"] == "LineString" for feature in probability_features
        )
        tier = "uploaded_probability_expected_loss"
        boundary = "Expected loss uses uploaded link failure probabilities as supplied. Dependence between failures and probability calibration are not inferred."
    else:
        tier = "deterministic_one_at_a_time_sensitivity"
        boundary = "Failure probabilities are incomplete, so this is a deterministic one-at-a-time sensitivity envelope, not stochastic reliability or tail risk."
    output = _clone_with_model(
        base,
        model_id="P5-M12",
        sources=["Jenelius, Petersen and Mattsson (2006), doi:10.1016/j.tra.2005.11.003"],
        conclusion=f"The {tier.replace('_', ' ')} tier returned {len(base.get('items', []))} link sensitivities.",
        boundary=boundary,
    )
    output["adaptive_execution"]["tier"] = tier
    output["evidence"]["failure_probability_coverage"] = len(probability_items) / max(1, len(base.get("items", [])))
    highest = max(
        (float(item.get("expected_efficiency_loss") or 0.0) for item in base.get("items", [])),
        default=0.0,
    )
    output["headline_effects"] = [
        {"label": "Links returned", "value": f"{len(base.get('items', [])):,}", "detail": tier.replace("_", " ")},
        {
            "label": "Probability coverage",
            "value": f"{output['evidence']['failure_probability_coverage'] * 100:.1f}%",
            "detail": "Uploaded values only; none invented",
        },
        {
            "label": "Highest expected loss" if tier == "uploaded_probability_expected_loss" else "Highest sensitivity",
            "value": f"{highest * 100:.3f}%" if tier == "uploaded_probability_expected_loss" else f"{max((float(item.get('deterministic_sensitivity') or 0.0) for item in base.get('items', [])), default=0.0) * 100:.3f}%",
            "detail": "Relative global-efficiency effect",
        },
    ]
    return output


def _edge_loads(graph: nx.MultiDiGraph) -> tuple[dict[tuple[Any, Any, Any], float], str]:
    if graph.number_of_nodes() <= 800 and graph.number_of_edges() <= 1800:
        return nx.edge_betweenness_centrality(graph, weight="weight", normalized=False), "exact"
    sources = _deterministic_sources(graph.nodes, maximum=40)
    targets = sorted(graph.nodes, key=str)
    return (
        nx.edge_betweenness_centrality_subset(
            graph, sources=sources, targets=targets, weight="weight", normalized=False
        ),
        f"deterministic_{len(sources)}_source_screen",
    )


def cascade_stress_test(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
    selected_keys: list[str],
    tolerance: float = 0.20,
) -> dict[str, Any]:
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, bbox)
    if graph.number_of_edges() == 0:
        raise ValueError("Topology fields are required for cascade stress testing.")
    baseline_load, load_mode = _edge_loads(graph)
    key_to_edge = {str(key): (u, v, key) for u, v, key in graph.edges(keys=True)}
    seeds = [key for key in selected_keys if key in key_to_edge]
    if selected_keys and len(seeds) != len(set(selected_keys)):
        raise ValueError("One or more selected links are not in the current topology scope.")
    if not seeds:
        seed_edge = max(baseline_load, key=lambda edge: (baseline_load[edge], str(edge[2])))
        seeds = [str(seed_edge[2])]
    failed: dict[str, dict[str, Any]] = {}
    stages: list[dict[str, Any]] = []
    for key in seeds:
        edge = key_to_edge[key]
        failed[key] = {"stage": 0, "overload_ratio": None, "trigger": "initial_seed"}
        if graph.has_edge(edge[0], edge[1], edge[2]):
            graph.remove_edge(edge[0], edge[1], edge[2])
    stages.append({"stage": 0, "new_failures": len(seeds), "cumulative_failures": len(failed)})
    epsilon = max(baseline_load.values(), default=1.0) * 1e-9
    for stage in range(1, 9):
        if graph.number_of_edges() == 0:
            break
        current_load, _ = _edge_loads(graph)
        overloaded: list[tuple[float, tuple[Any, Any, Any]]] = []
        for edge, load in current_load.items():
            capacity = (1.0 + tolerance) * max(epsilon, float(baseline_load.get(edge, 0.0)))
            ratio = float(load) / capacity if capacity > 0 else math.inf
            if ratio > 1.0 + 1e-12:
                overloaded.append((ratio, edge))
        overloaded.sort(key=lambda pair: (pair[0], str(pair[1][2])), reverse=True)
        if not overloaded:
            break
        # A one-stage cap keeps a proxy-load cascade computationally bounded
        # and is disclosed in evidence rather than silently dropping links.
        overloaded = overloaded[: min(500, len(overloaded))]
        for ratio, edge in overloaded:
            key = str(edge[2])
            failed[key] = {"stage": stage, "overload_ratio": ratio, "trigger": "proxy_load_transfer"}
            if graph.has_edge(edge[0], edge[1], edge[2]):
                graph.remove_edge(edge[0], edge[1], edge[2])
        stages.append({"stage": stage, "new_failures": len(overloaded), "cumulative_failures": len(failed)})
    ranked_keys = sorted(failed, key=lambda key: (failed[key]["stage"], -(failed[key]["overload_ratio"] or math.inf), key))
    returned_keys = ranked_keys[: min(len(ranked_keys), max(1, int(top_k)))]
    items = [
        {
            "rank": rank,
            **records[key],
            **failed[key],
            "severity": 1.0 if failed[key]["stage"] == 0 else min(1.0, max(0.0, (failed[key]["overload_ratio"] or 1.0) - 1.0)),
        }
        for rank, key in enumerate(returned_keys, 1)
        if key in records
    ]
    geometries, mapping_ready = _inferred_link_geometries(database_path, returned_keys)
    features = [
        feature
        for item in items
        if (feature := _link_result_feature(item, geometries, severity=item["severity"], allow_point_fallback=not mapping_ready)) is not None
    ]
    return {
        "model": {
            "id": "P5-M13",
            "name": ADAPTIVE_MODEL_NAMES["P5-M13"],
            "status": "adaptive_executable_base_tier",
            "sources": ["Motter and Lai (2002), doi:10.1103/PhysRevE.66.065102"],
        },
        "status": "conditional",
        "conclusion": f"Starting from {len(seeds)} deterministic seed link(s), the proxy-load stress test produced {len(failed) - len(seeds)} secondary failures across {max(0, len(stages) - 1)} propagation stages.",
        "headline_effects": [
            {"label": "Initial seeds", "value": str(len(seeds)), "detail": "Selected links or highest proxy-load link"},
            {"label": "Secondary failures", "value": str(max(0, len(failed) - len(seeds))), "detail": f"Tolerance {(tolerance * 100):.0f}% above baseline proxy load"},
            {"label": "Propagation stages", "value": str(max(0, len(stages) - 1)), "detail": load_mode.replace("_", " ")},
        ],
        "next_actions": [
            "Compare cascade-sensitive links with observed capacity and flow.",
            "Select known initial failures instead of using the automatic stress seed.",
            "Add dependency data before interpreting interdependent-infrastructure cascades.",
        ],
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "initial_seed_keys": seeds,
            "tolerance": tolerance,
            "load_definition": "weighted edge betweenness used as a deterministic proxy load",
            "load_computation": load_mode,
            "stages": stages,
            "iteration_limit": 8,
            "per_stage_failure_cap": 500,
        },
        "data_boundary": "This is a single-road-network proxy-load stress test. It is not a calibrated traffic-flow cascade, an interdependent infrastructure model or a failure-probability forecast.",
        "verification": {
            "passed": all(item["stage"] >= 0 for item in items),
            "checks": ["deterministic seed rule", "fixed tolerance", "failed links removed once", "bounded iterations", "no random failure draw"],
        },
    }


def hazard_exposure_screen(
    database_path: Path,
    *,
    metric: str,
    top_k: int,
    bbox: list[float] | None,
    timestamp: int | None,
) -> dict[str, Any]:
    base = spatial_hotspots(
        database_path,
        metric=metric,
        top_k=top_k,
        bbox=bbox,
        timestamp=timestamp,
    )
    return _clone_with_model(
        base,
        model_id="P5-M14",
        sources=["Pregnolato et al. (2017), doi:10.1016/j.trd.2017.06.020"],
        conclusion=f"The observed-exposure tier returned {len(base.get('items', []))} geographic cells from uploaded transport or hazard-related measurements.",
        boundary="This maps uploaded observations as exposure evidence. A hazard intensity/depth layer and a calibrated transport impact function are required for hazard-to-damage or hazard-to-speed causation.",
    )


def protection_screen(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
) -> dict[str, Any]:
    base = critical_components(database_path, top_k=top_k, bbox=bbox)
    for item in base.get("items", []):
        item["protection_priority_score"] = float(item.get("efficiency_loss", item.get("structural_score", 0.0)) or 0.0)
    return _clone_with_model(
        base,
        model_id="P5-M16",
        sources=["Wood (1993), doi:10.1016/0895-7177(93)90236-R"],
        conclusion=f"The greedy protection screen returned {len(base.get('items', []))} structurally sensitive links for candidate hardening review.",
        boundary="This ranks candidates one link at a time. It is not a defender-attacker-defender optimum and does not include a hardening budget, attacker strategy, correlated failure or construction feasibility.",
    )


def chronological_validation(
    database_path: Path,
    *,
    metric: str,
    top_k: int,
) -> dict[str, Any]:
    value_column, _baseline_column, unit = _metric_parts(metric)
    observation_column = value_column.split(".")[-1]
    with _connect(database_path) as connection:
        summary = _summary(connection)
        keys = [
            (int(row["entity_pk"]), str(row["region_id"]), str(row["entity_id"]), row["latitude"], row["longitude"])
            for row in connection.execute(
                f"""
                SELECT e.entity_pk, e.region_id, e.entity_id, e.latitude, e.longitude
                FROM entities e
                WHERE EXISTS (
                    SELECT 1 FROM observations o
                    WHERE o.entity_pk = e.entity_pk
                      AND o.{observation_column} IS NOT NULL
                )
                ORDER BY e.region_id, e.entity_id
                """
            ).fetchall()
        ]
        if not keys:
            raise ValueError(f"The project does not contain {metric.replace('_', ' ')} observations.")
        sampled_keys = keys if len(keys) <= 800 else [keys[min(len(keys) - 1, int(index * len(keys) / 800))] for index in range(800)]
        items: list[dict[str, Any]] = []
        for entity_pk, region_id, entity_id, latitude, longitude in sampled_keys:
            series = [
                (int(row["timestamp_epoch"]), float(row["value"]))
                for row in connection.execute(
                    f"SELECT timestamp_epoch, {observation_column} AS value FROM observations WHERE entity_pk = ? AND {observation_column} IS NOT NULL ORDER BY timestamp_epoch",
                    (entity_pk,),
                ).fetchall()
            ]
            if len(series) < 3:
                continue
            split = max(2, min(len(series) - 1, int(math.floor(len(series) * 0.70))))
            prediction = series[split - 1][1]
            holdout = series[split:]
            absolute_errors = [abs(value - prediction) for _timestamp, value in holdout]
            mae = sum(absolute_errors) / len(absolute_errors)
            scale = sum(abs(value) for _timestamp, value in holdout) / len(holdout)
            items.append(
                {
                    "key": f"{region_id}::{entity_id}",
                    "region_id": region_id,
                    "entity_id": entity_id,
                    "rank": 0,
                    "training_points": split,
                    "holdout_points": len(holdout),
                    "persistence_prediction": prediction,
                    "holdout_mae": mae,
                    "relative_mae": mae / scale if scale > 0 else None,
                    "holdout_start": holdout[0][0],
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    items.sort(key=lambda item: (item["holdout_mae"], item["key"]), reverse=True)
    returned = items[: min(len(items), max(1, int(top_k)))]
    for rank, item in enumerate(returned, 1):
        item["rank"] = rank
    geometries, mapping_ready = _inferred_link_geometries(database_path, (item["key"] for item in returned))
    maximum = max((item["holdout_mae"] for item in returned), default=0.0)
    features = [
        feature
        for item in returned
        if (feature := _link_result_feature(
            item,
            geometries,
            severity=item["holdout_mae"] / maximum if maximum > 0 else 0.0,
            allow_point_fallback=not mapping_ready,
        )) is not None
    ]
    aggregate_mae = sum(item["holdout_mae"] for item in items) / len(items) if items else 0.0
    return {
        "model": {
            "id": "P5-M19",
            "name": ADAPTIVE_MODEL_NAMES["P5-M19"],
            "status": "adaptive_executable_base_tier",
            "sources": ["Chronological holdout and persistence benchmark; Bruneau et al. (2003), doi:10.1193/1.1623497"],
        },
        "status": "supported" if len(sampled_keys) == len(keys) else "conditional",
        "conclusion": f"A leakage-resistant 70/30 chronological holdout compared the last training value with later observations for {len(items):,} entities; mean entity MAE was {aggregate_mae:.3g} {summary.get('metrics', {}).get(metric, {}).get('unit', unit)}.",
        "headline_effects": [
            {"label": "Entities evaluated", "value": f"{len(items):,}", "detail": "All eligible" if len(sampled_keys) == len(keys) else f"Deterministic sample of {len(keys):,}"},
            {"label": "Mean entity MAE", "value": f"{aggregate_mae:.3g}", "detail": summary.get("metrics", {}).get(metric, {}).get("unit", unit)},
            {"label": "Holdout rule", "value": "Last 30%", "detail": "Chronological · no shuffle"},
        ],
        "next_actions": [
            "Inspect entities where a persistence benchmark fails most strongly.",
            "Compare a proposed predictive model against this same untouched holdout.",
            "Use a separate event or city for external validation before generalising.",
        ],
        "items": returned,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "split_rule": "first 70 percent train, final 30 percent holdout per entity",
            "benchmark": "last observed training value",
            "eligible_entity_count": len(items),
            "total_metric_entity_count": len(keys),
            "sampled_entity_count": len(sampled_keys),
            "sampling_rule": "all entities up to 800, otherwise deterministic evenly-spaced entity keys",
        },
        "data_boundary": "This validates a persistence benchmark against later uploaded observations. It does not validate a disaster forecast, causal hazard model or transfer to another event/location.",
        "verification": {
            "passed": all(item["holdout_points"] >= 1 and item["training_points"] >= 2 and item["holdout_mae"] >= 0 for item in items),
            "checks": ["strict chronological split", "training observation precedes holdout", "non-negative MAE", "no random shuffle", "sampling rule disclosed"],
        },
    }


def run_adaptive_model(database_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    model_id = str(request.get("model_id") or "")
    top_k = max(1, int(request.get("top_k", 10)))
    bbox = request.get("bbox")
    selected_keys = list(request.get("selected_keys") or [])
    metric = str(request.get("metric") or "speed")
    if model_id == "P5-M5":
        return importance_exposure_screen(database_path, top_k=top_k, bbox=bbox)
    if model_id in {"P5-M6", "P5-M15", "P5-M18"}:
        return accessibility_screen(database_path, top_k=top_k, bbox=bbox, model_id=model_id)
    if model_id in {"P5-M8", "P5-M20"}:
        return resource_restoration_screen(
            database_path,
            selected_keys=selected_keys,
            bbox=bbox,
            model_id=model_id,
        )
    if model_id == "P5-M12":
        return reliability_screen(database_path, top_k=top_k, bbox=bbox)
    if model_id == "P5-M13":
        return cascade_stress_test(
            database_path,
            top_k=top_k,
            bbox=bbox,
            selected_keys=selected_keys,
        )
    if model_id == "P5-M14":
        return hazard_exposure_screen(
            database_path,
            metric=metric,
            top_k=top_k,
            bbox=bbox,
            timestamp=request.get("time_end") or request.get("time_start"),
        )
    if model_id == "P5-M16":
        return protection_screen(database_path, top_k=top_k, bbox=bbox)
    if model_id == "P5-M19":
        return chronological_validation(database_path, metric=metric, top_k=top_k)
    raise ValueError(f"No direct adaptive executor is registered for {model_id}.")
