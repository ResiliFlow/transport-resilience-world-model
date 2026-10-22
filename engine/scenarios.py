from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .models import _component_metrics, _connect, _nearest_timestamp, _network_efficiency, _summary
from .network_mapping import MAPPING_LINK_TABLE, MAPPING_TABLE
from .transport_data import EXACT_GEOMETRY_TABLE
from .traffic_assignment import compare_assignments


MAXIMUM_SCENARIO_TARGETS = 5_000


def _table_exists(connection: Any, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone() is not None


def _normalise_bbox(bbox: list[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("Draw one rectangular area on the map.")
    west, south, east, north = (float(value) for value in bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("The selected area must use valid west, south, east and north coordinates.")
    return west, south, east, north


def _geometry_rows(
    connection: Any,
    *,
    bbox: list[float] | None = None,
    limit: int | None = None,
) -> tuple[list[Any], int]:
    clauses = ["e.latitude BETWEEN -90 AND 90", "e.longitude BETWEEN -180 AND 180"]
    parameters: list[Any] = []
    if bbox is not None:
        west, south, east, north = _normalise_bbox(bbox)
        clauses.extend(["e.longitude BETWEEN ? AND ?", "e.latitude BETWEEN ? AND ?"])
        parameters.extend([west, east, south, north])
    where = " AND ".join(clauses)
    total = int(connection.execute(f"SELECT COUNT(*) FROM entities e WHERE {where}", parameters).fetchone()[0])
    mapped = _table_exists(connection, MAPPING_TABLE) and _table_exists(connection, MAPPING_LINK_TABLE)
    exact = _table_exists(connection, EXACT_GEOMETRY_TABLE)
    if mapped:
        query = f"""
            SELECT e.region_id, e.entity_id, e.from_node, e.to_node, e.latitude, e.longitude,
                   nf.longitude AS from_longitude, nf.latitude AS from_latitude,
                   nt.longitude AS to_longitude, nt.latitude AS to_latitude,
                   {"g.geometry_json" if exact else "NULL"} AS exact_geometry_json,
                   {"g.geometry_kind" if exact else "NULL"} AS exact_geometry_kind
            FROM entities e
            LEFT JOIN {MAPPING_LINK_TABLE} ml
              ON ml.region_id = e.region_id AND ml.entity_id = e.entity_id
            LEFT JOIN {MAPPING_TABLE} nf ON nf.node_key = ml.from_node_key
            LEFT JOIN {MAPPING_TABLE} nt ON nt.node_key = ml.to_node_key
            {f"LEFT JOIN {EXACT_GEOMETRY_TABLE} g ON g.region_id = e.region_id AND g.entity_id = e.entity_id" if exact else ""}
            WHERE {where}
            ORDER BY e.region_id, e.entity_id
        """
    else:
        query = f"""
            SELECT e.region_id, e.entity_id, e.from_node, e.to_node, e.latitude, e.longitude,
                   NULL AS from_longitude, NULL AS from_latitude,
                   NULL AS to_longitude, NULL AS to_latitude,
                   {"g.geometry_json" if exact else "NULL"} AS exact_geometry_json,
                   {"g.geometry_kind" if exact else "NULL"} AS exact_geometry_kind
            FROM entities e
            {f"LEFT JOIN {EXACT_GEOMETRY_TABLE} g ON g.region_id = e.region_id AND g.entity_id = e.entity_id" if exact else ""}
            WHERE {where}
            ORDER BY e.region_id, e.entity_id
        """
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(int(limit))
    return connection.execute(query, parameters).fetchall(), total


def _row_feature(row: Any, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    key = f"{row['region_id']}::{row['entity_id']}"
    has_line = all(
        row[name] is not None
        for name in ("from_longitude", "from_latitude", "to_longitude", "to_latitude")
    )
    exact_geometry = None
    if row["exact_geometry_json"]:
        try:
            exact_geometry = json.loads(row["exact_geometry_json"])
        except (TypeError, json.JSONDecodeError):
            exact_geometry = None
    geometry = exact_geometry or (
        {
            "type": "LineString",
            "coordinates": [
                [float(row["from_longitude"]), float(row["from_latitude"])],
                [float(row["to_longitude"]), float(row["to_latitude"])],
            ],
        }
        if has_line
        else {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]}
    )
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "key": key,
            "region_id": str(row["region_id"]),
            "entity_id": str(row["entity_id"]),
            "from_node": row["from_node"],
            "to_node": row["to_node"],
            "geometry_kind": row["exact_geometry_kind"] if exact_geometry else "inferred_topology_line" if has_line else "representative_point",
            **(properties or {}),
        },
    }


def select_scenario_targets(
    database_path: Path,
    *,
    bbox: list[float],
    limit: int = MAXIMUM_SCENARIO_TARGETS,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit), MAXIMUM_SCENARIO_TARGETS))
    with _connect(database_path) as connection:
        rows, total = _geometry_rows(connection, bbox=bbox, limit=safe_limit + 1)
    truncated = len(rows) > safe_limit or total > safe_limit
    rows = rows[:safe_limit]
    features = [_row_feature(row) for row in rows]
    return {
        "keys": [feature["properties"]["key"] for feature in features],
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "selected_count": len(features),
        "total_count": total,
        "truncated": truncated,
        "limit": safe_limit,
        "message": (
            f"The area contains {total:,} mapped entities. Narrow the rectangle to select at most {safe_limit:,}."
            if truncated
            else f"The area contains {total:,} mapped entit{'y' if total == 1 else 'ies'}."
        ),
    }


def _speed_to_metres_per_second(speed: float, unit: str) -> float | None:
    cleaned = str(unit or "").strip().lower().replace(" ", "")
    if cleaned in {"km/h", "kmh", "kph"}:
        return speed / 3.6
    if cleaned in {"mph", "mi/h"}:
        return speed * 0.44704
    if cleaned in {"m/s", "mps"}:
        return speed
    return None


def _time_graph(connection: Any, timestamp: int) -> tuple[nx.MultiDiGraph, dict[str, dict[str, Any]]]:
    summary = _summary(connection)
    global_nodes = summary.get("topology_node_scope", "global") == "global"
    entity_columns = {row[1] for row in connection.execute("PRAGMA table_info(entities)")}
    observation_columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
    capacity_expression = "e.capacity" if "capacity" in entity_columns else "NULL"
    free_flow_expression = "e.free_flow_time" if "free_flow_time" in entity_columns else "NULL"
    observed_flow_expression = "o.flow" if "flow" in observation_columns else "NULL"
    rows = connection.execute(
        f"""
        SELECT e.region_id, e.entity_id, e.from_node, e.to_node, e.length_m,
               e.baseline_travel_time, e.baseline_speed, e.latitude, e.longitude,
               o.travel_time AS observed_travel_time, o.speed AS observed_speed,
               {capacity_expression} AS capacity,
               {free_flow_expression} AS free_flow_time,
               {observed_flow_expression} AS observed_flow
        FROM entities e
        LEFT JOIN observations o
          ON o.entity_pk = e.entity_pk AND o.timestamp_epoch = ?
        WHERE e.from_node IS NOT NULL AND e.from_node <> ''
          AND e.to_node IS NOT NULL AND e.to_node <> ''
        """,
        (timestamp,),
    ).fetchall()
    graph = nx.MultiDiGraph()
    records: dict[str, dict[str, Any]] = {}
    graph.graph["topology_node_scope"] = "global" if global_nodes else "per_region"
    travel_ready = bool(rows) and all(
        row["observed_travel_time"] is not None and float(row["observed_travel_time"]) > 0
        for row in rows
    )
    speed_unit = summary.get("metrics", {}).get("speed", {}).get("unit", "")
    speed_ready = bool(rows) and all(
        row["length_m"] is not None
        and float(row["length_m"]) > 0
        and row["observed_speed"] is not None
        and float(row["observed_speed"]) > 0
        and _speed_to_metres_per_second(float(row["observed_speed"]), speed_unit)
        for row in rows
    )
    baseline_ready = bool(rows) and all(
        row["baseline_travel_time"] is not None and float(row["baseline_travel_time"]) > 0
        for row in rows
    )
    free_flow_ready = bool(rows) and all(
        row["free_flow_time"] is not None and float(row["free_flow_time"]) > 0
        for row in rows
    )
    length_ready = bool(rows) and all(
        row["length_m"] is not None and float(row["length_m"]) > 0 for row in rows
    )
    if travel_ready:
        basis = "observed_travel_time"
        basis_label = "Observed travel time at the selected historical time"
    elif speed_ready:
        basis = "observed_speed_derived_time"
        basis_label = "Travel time derived from observed speed and uploaded link length"
    elif free_flow_ready:
        basis = "uploaded_free_flow_time"
        basis_label = "Uploaded free-flow travel time"
    elif baseline_ready:
        basis = "reference_travel_time"
        basis_label = "Internal reference travel time"
    elif length_ready:
        basis = "link_length"
        basis_label = "Uploaded link length as a relative impedance"
    else:
        basis = "uniform_hop_count"
        basis_label = "One equal impedance unit per directed link"
    graph.graph.update({"impedance_mode": basis, "impedance_label": basis_label, "timestamp": timestamp})
    for row in rows:
        if basis == "observed_travel_time":
            weight = float(row["observed_travel_time"])
        elif basis == "observed_speed_derived_time":
            speed_mps = _speed_to_metres_per_second(float(row["observed_speed"]), speed_unit)
            weight = float(row["length_m"]) / max(float(speed_mps or 0), 1e-12)
        elif basis == "uploaded_free_flow_time":
            weight = float(row["free_flow_time"])
        elif basis == "reference_travel_time":
            weight = float(row["baseline_travel_time"])
        elif basis == "link_length":
            weight = float(row["length_m"])
        else:
            weight = 1.0
        key = f"{row['region_id']}::{row['entity_id']}"
        graph_from = str(row["from_node"]) if global_nodes else f"{row['region_id']}\x1f{row['from_node']}"
        graph_to = str(row["to_node"]) if global_nodes else f"{row['region_id']}\x1f{row['to_node']}"
        record = {
            "key": key,
            "region_id": str(row["region_id"]),
            "entity_id": str(row["entity_id"]),
            "from_node": str(row["from_node"]),
            "to_node": str(row["to_node"]),
            "graph_from": graph_from,
            "graph_to": graph_to,
            "weight": float(weight),
            "length_m": float(row["length_m"]) if row["length_m"] is not None and float(row["length_m"]) > 0 else None,
            "baseline_travel_time": float(row["baseline_travel_time"]) if row["baseline_travel_time"] is not None and float(row["baseline_travel_time"]) > 0 else None,
            "baseline_speed": float(row["baseline_speed"]) if row["baseline_speed"] is not None and float(row["baseline_speed"]) > 0 else None,
            "observed_travel_time": float(row["observed_travel_time"]) if row["observed_travel_time"] is not None and float(row["observed_travel_time"]) > 0 else None,
            "observed_speed": float(row["observed_speed"]) if row["observed_speed"] is not None and float(row["observed_speed"]) > 0 else None,
            "free_flow_weight": float(row["free_flow_time"]) if row["free_flow_time"] is not None and float(row["free_flow_time"]) > 0 else None,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "capacity": float(row["capacity"]) if row["capacity"] is not None and float(row["capacity"]) > 0 else None,
            "observed_flow": float(row["observed_flow"]) if row["observed_flow"] is not None and float(row["observed_flow"]) >= 0 else None,
        }
        graph.add_edge(
            graph_from,
            graph_to,
            key=key,
            **{name: value for name, value in record.items() if name != "key"},
        )
        records[key] = record
    return graph, records


def _canonical_changes(changes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    for raw in changes:
        key = str(raw.get("key") or "").strip()
        effect = str(raw.get("effect") or "").strip()
        if not key or "::" not in key:
            raise ValueError("Every scenario change needs a mapped entity or road-link key.")
        if key in canonical:
            raise ValueError(f"{key} appears more than once. Keep one final change for each mapped entity.")
        if effect not in {"closed", "speed_reduction", "travel_time_increase"}:
            raise ValueError(f"{key} uses an unsupported disruption effect.")
        magnitude = 1.0 if effect == "closed" else float(raw.get("magnitude") or 0)
        if effect == "speed_reduction" and not 0.01 <= magnitude <= 0.95:
            raise ValueError("Speed reduction must be between 1% and 95%.")
        if effect == "travel_time_increase" and not 0.01 <= magnitude <= 5.0:
            raise ValueError("Travel-time increase must be between 1% and 500%.")
        canonical[key] = {"key": key, "effect": effect, "magnitude": magnitude}
    if not canonical:
        raise ValueError("Select at least one mapped location or road link for this scenario.")
    if len(canonical) > MAXIMUM_SCENARIO_TARGETS:
        raise ValueError(f"Keep one scenario to at most {MAXIMUM_SCENARIO_TARGETS:,} mapped entities.")
    return list(canonical.values())


def _scenario_features(connection: Any, changes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    by_key = {change["key"]: change for change in changes}
    rows, _ = _geometry_rows(connection)
    features: list[dict[str, Any]] = []
    grounded: set[str] = set()
    for row in rows:
        key = f"{row['region_id']}::{row['entity_id']}"
        change = by_key.get(key)
        if change is None:
            continue
        grounded.add(key)
        severity = 1.0 if change["effect"] == "closed" else min(1.0, float(change["magnitude"]))
        features.append(_row_feature(row, {**change, "severity": severity, "scenario": True}))
    return features, sorted(set(by_key) - grounded)


def _legacy_evaluate_disruption_scenario(
    database_path: Path,
    *,
    changes: list[dict[str, Any]],
    base_timestamp: int | None,
) -> dict[str, Any]:
    canonical = _canonical_changes(changes)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, base_timestamp)
        features, missing = _scenario_features(connection, canonical)
        if missing:
            raise ValueError("Some selected entities are no longer available: " + ", ".join(missing[:5]))
        graph, records = _time_graph(connection, selected_time)
    by_key = {change["key"]: change for change in canonical}
    topology_changes = [change for change in canonical if change["key"] in records]
    closure_count = sum(change["effect"] == "closed" for change in canonical)
    slower_count = sum(change["effect"] == "speed_reduction" for change in canonical)
    longer_count = sum(change["effect"] == "travel_time_increase" for change in canonical)
    base_result = {
        "model": {
            "id": "P5-M21",
            "name": "Deterministic multi-effect disruption scenario",
            "status": "numerically_verified",
        },
        "timestamp_epoch": selected_time,
        "timestamp_label": summary.get("time_labels", {}).get(str(selected_time), str(selected_time)),
        "affected_count": len(canonical),
        "topology_applied_count": len(topology_changes),
        "change_counts": {
            "closed": closure_count,
            "speed_reduction": slower_count,
            "travel_time_increase": longer_count,
        },
        "items": canonical,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "randomness": "none",
    }
    if graph.number_of_edges() == 0 or not topology_changes:
        return {
            **base_result,
            "status": "location_only",
            "impact_level": "mapped_locations",
            "plain_finding": (
                f"{len(canonical):,} mapped location{' is' if len(canonical) == 1 else 's are'} included. "
                "Network connectivity cannot be calculated because these data do not provide usable From and To node connections."
            ),
            "headline_effects": [
                {"label": "Affected locations", "value": f"{len(canonical):,}", "detail": "Shown on the geographic map"},
                {"label": "Network result", "value": "Not available", "detail": "Connectivity fields are missing"},
                {"label": "Random simulation", "value": "None", "detail": "Every change was selected by the user"},
            ],
            "visual": {
                "mode": "location_only",
                "baseline_efficiency_percent": None,
                "scenario_efficiency_percent": None,
                "baseline_components": None,
                "scenario_components": None,
            },
            "evidence": {
                "impedance_mode": None,
                "selected_time": selected_time,
                "communication_rule": "No qualitative network impact is inferred without topology.",
            },
            "data_boundary": "This scenario maps user-confirmed affected locations only. It does not infer network disruption, traffic redistribution or emergency consequences.",
            "verification": {
                "passed": len(features) == len(canonical),
                "checks": ["every change grounded in an uploaded entity", "no random sampling", "network claim withheld"],
            },
        }

    baseline_efficiency, exact, source_count = _network_efficiency(graph)
    baseline_components = _component_metrics(graph)
    for change in topology_changes:
        record = records[change["key"]]
        u, v, key = record["graph_from"], record["graph_to"], change["key"]
        if not graph.has_edge(u, v, key):
            continue
        if change["effect"] == "closed":
            graph.remove_edge(u, v, key)
        elif change["effect"] == "speed_reduction":
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) / max(
                0.05, 1.0 - float(change["magnitude"])
            )
        else:
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) * (
                1.0 + float(change["magnitude"])
            )
    scenario_efficiency, _, _ = _network_efficiency(graph)
    scenario_components = _component_metrics(graph)
    efficiency_loss = (
        max(0.0, (baseline_efficiency - scenario_efficiency) / baseline_efficiency)
        if baseline_efficiency > 0
        else 0.0
    )
    added_components = max(0, scenario_components["weak_components"] - baseline_components["weak_components"])
    component_retention = (
        scenario_components["largest_component_nodes"] / baseline_components["largest_component_nodes"]
        if baseline_components["largest_component_nodes"] > 0
        else 1.0
    )
    if added_components > 0:
        impact_level = "network_split"
        plain_finding = (
            f"The scenario creates {added_components} additional network part{'s' if added_components != 1 else ''}. "
            "Some places that were connected in the baseline may no longer reach each other through this directed topology."
        )
    elif efficiency_loss >= 0.15:
        impact_level = "large_impedance"
        plain_finding = "The network stays in the same number of parts, but movement through it becomes substantially harder."
    elif efficiency_loss >= 0.05:
        impact_level = "noticeable_impedance"
        plain_finding = "The network stays in the same number of parts, with a noticeable loss of topology-based movement efficiency."
    else:
        impact_level = "limited_topology_change"
        plain_finding = "The selected changes have a limited effect on whole-network topology efficiency, although local effects may still matter."
    monotone = scenario_efficiency <= baseline_efficiency + max(1e-12, abs(baseline_efficiency) * 1e-9)
    return {
        **base_result,
        "status": "supported" if exact and graph.graph.get("impedance_mode") != "uniform_hop_count" else "conditional",
        "impact_level": impact_level,
        "plain_finding": plain_finding,
        "headline_effects": [
            {
                "label": "Network movement retained",
                "value": f"{max(0.0, 1.0 - efficiency_loss) * 100:.1f}%",
                "detail": "Topology efficiency compared with the selected baseline time",
            },
            {
                "label": "Connected parts",
                "value": f"{baseline_components['weak_components']} → {scenario_components['weak_components']}",
                "detail": "Weakly connected components in the directed network",
            },
            {
                "label": "Changed road links",
                "value": f"{len(topology_changes):,}",
                "detail": f"{closure_count:,} closed · {slower_count + longer_count:,} impedance changes",
            },
        ],
        "visual": {
            "mode": "network_compare",
            "baseline_efficiency_percent": 100.0,
            "scenario_efficiency_percent": max(0.0, 1.0 - efficiency_loss) * 100.0,
            "baseline_components": baseline_components["weak_components"],
            "scenario_components": scenario_components["weak_components"],
            "largest_component_retained_percent": max(0.0, min(1.0, component_retention)) * 100.0,
        },
        "evidence": {
            "baseline_efficiency": baseline_efficiency,
            "scenario_efficiency": scenario_efficiency,
            "relative_efficiency_loss": efficiency_loss,
            "baseline_components": baseline_components,
            "scenario_components": scenario_components,
            "efficiency_exact": exact,
            "efficiency_source_count": source_count,
            "impedance_mode": graph.graph.get("impedance_mode"),
            "impedance_label": graph.graph.get("impedance_label"),
            "selected_time": selected_time,
            "communication_rule": (
                "Network split when weak-component count increases; otherwise large at 15% efficiency loss, "
                "noticeable at 5%, and limited below 5%. These are interface reading aids, not calibrated hazard probabilities."
            ),
        },
        "data_boundary": (
            "This is a deterministic network counterfactual. Speed and travel-time changes modify link impedance; closures remove selected directed links. "
            "No OD demand, congestion redistribution, probability, casualty estimate or operational instruction is inferred."
        ),
        "verification": {
            "passed": len(features) == len(canonical) and len(topology_changes) > 0 and monotone,
            "checks": [
                "every change grounded in an uploaded entity",
                "one final change per entity",
                "baseline and scenario recomputed on the same directed topology",
                "non-improving disruption check",
                "no random sampling of changes",
            ],
        },
    }


def evaluate_disruption_scenario(
    database_path: Path,
    *,
    changes: list[dict[str, Any]],
    base_timestamp: int | None,
) -> dict[str, Any]:
    """Evaluate direct disruption seeds and their deterministic network spillover.

    The three user-facing operators are grounded as follows: ``closed`` removes
    the directed link, ``speed_reduction`` applies ``t' = t / (1-r)``, and
    ``travel_time_increase`` applies ``t' = t(1+m)``.  A fixed, disclosed OD
    sensitivity design is then assigned to baseline and scenario networks so
    selected links are never confused with inferred downstream effects.
    """

    canonical = _canonical_changes(changes)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, base_timestamp)
        direct_features, missing = _scenario_features(connection, canonical)
        if missing:
            raise ValueError("Some selected entities are no longer available: " + ", ".join(missing[:5]))
        baseline_graph, records = _time_graph(connection, selected_time)
        geometry_rows, _ = _geometry_rows(connection)
        feature_templates = {
            f"{row['region_id']}::{row['entity_id']}": _row_feature(row)
            for row in geometry_rows
        }

    topology_changes = [change for change in canonical if change["key"] in records]
    closure_count = sum(change["effect"] == "closed" for change in canonical)
    slower_count = sum(change["effect"] == "speed_reduction" for change in canonical)
    longer_count = sum(change["effect"] == "travel_time_increase" for change in canonical)
    for rank, feature in enumerate(direct_features, start=1):
        feature["properties"].update(
            {
                "impact_role": "initial_disruption",
                "impact_label": "Initial disruption selected by user",
                "rank": rank,
                "initial_disruption": True,
            }
        )
    base_result = {
        "model": {
            "id": "P5-M22",
            "name": "Adaptive disruption propagation and rerouting scenario",
            "status": "numerically_verified",
        },
        "timestamp_epoch": selected_time,
        "timestamp_label": summary.get("time_labels", {}).get(str(selected_time), str(selected_time)),
        "affected_count": len(canonical),
        "initial_disruption_count": len(canonical),
        "topology_applied_count": len(topology_changes),
        "change_counts": {
            "closed": closure_count,
            "speed_reduction": slower_count,
            "travel_time_increase": longer_count,
        },
        "items": canonical,
        "feature_collection": {"type": "FeatureCollection", "features": direct_features},
        "randomness": "none",
    }
    if baseline_graph.number_of_edges() == 0 or not topology_changes:
        return {
            **base_result,
            "status": "location_only",
            "impact_level": "mapped_locations",
            "indirect_impact_count": 0,
            "indirect_items": [],
            "recovery_screening": [],
            "plain_finding": (
                f"{len(canonical):,} user-confirmed location{' is' if len(canonical) == 1 else 's are'} mapped as the initial disruption. "
                "Indirect road effects cannot be calculated because the upload does not provide usable From and To node connections."
            ),
            "headline_effects": [
                {"label": "Initial locations", "value": f"{len(canonical):,}", "detail": "Selected by the user and shown on the map"},
                {"label": "Indirect road effects", "value": "Not calculated", "detail": "From-node and To-node topology is missing"},
                {"label": "Random simulation", "value": "None", "detail": "No downstream claim is invented"},
            ],
            "visual": {
                "mode": "location_only",
                "baseline_efficiency_percent": None,
                "scenario_efficiency_percent": None,
                "initial_link_count": len(canonical),
                "indirect_link_count": 0,
            },
            "evidence": {
                "impedance_mode": None,
                "selected_time": selected_time,
                "communication_rule": "No network propagation is inferred without topology.",
            },
            "data_boundary": (
                "This tier maps confirmed disruption locations only. Add From node and To node fields, "
                "or use Map to links when a defensible mapping is available, to unlock propagation analysis."
            ),
            "verification": {
                "passed": len(direct_features) == len(canonical),
                "checks": ["every change grounded in an uploaded entity", "no random sampling", "unsupported network claim withheld"],
            },
        }

    scenario_graph = baseline_graph.copy()
    for change in topology_changes:
        record = records[change["key"]]
        u, v, key = record["graph_from"], record["graph_to"], change["key"]
        if not scenario_graph.has_edge(u, v, key):
            continue
        if change["effect"] == "closed":
            scenario_graph.remove_edge(u, v, key)
        elif change["effect"] == "speed_reduction":
            divisor = max(0.05, 1.0 - float(change["magnitude"]))
            scenario_graph[u][v][key]["weight"] = float(scenario_graph[u][v][key]["weight"]) / divisor
            if scenario_graph[u][v][key].get("free_flow_weight") is not None:
                scenario_graph[u][v][key]["free_flow_weight"] = float(
                    scenario_graph[u][v][key]["free_flow_weight"]
                ) / divisor
        else:
            multiplier = 1.0 + float(change["magnitude"])
            scenario_graph[u][v][key]["weight"] = float(scenario_graph[u][v][key]["weight"]) * multiplier
            if scenario_graph[u][v][key].get("free_flow_weight") is not None:
                scenario_graph[u][v][key]["free_flow_weight"] = float(
                    scenario_graph[u][v][key]["free_flow_weight"]
                ) * multiplier

    baseline_efficiency, exact, source_count = _network_efficiency(baseline_graph)
    baseline_components = _component_metrics(baseline_graph)
    scenario_efficiency, _, _ = _network_efficiency(scenario_graph)
    scenario_components = _component_metrics(scenario_graph)
    redistribution = compare_assignments(
        baseline_graph,
        scenario_graph,
        records,
        seed_keys={change["key"] for change in topology_changes},
    )
    indirect_items = redistribution["indirect_increases"]
    indirect_features: list[dict[str, Any]] = []
    for item in indirect_items:
        template = feature_templates.get(item["key"])
        if template is None:
            continue
        indirect_features.append(
            {
                "type": "Feature",
                "geometry": template["geometry"],
                "properties": {
                    **template["properties"],
                    **item,
                    "scenario": True,
                    "initial_disruption": False,
                    "impact_role": "rerouting_pressure",
                    "impact_label": "Indirect rerouting pressure",
                },
            }
        )
    combined_features = [*direct_features, *indirect_features]

    efficiency_loss = (
        max(0.0, (baseline_efficiency - scenario_efficiency) / baseline_efficiency)
        if baseline_efficiency > 0
        else 0.0
    )
    added_components = max(
        0,
        scenario_components["weak_components"] - baseline_components["weak_components"],
    )
    component_retention = (
        scenario_components["largest_component_nodes"] / baseline_components["largest_component_nodes"]
        if baseline_components["largest_component_nodes"] > 0
        else 1.0
    )
    if redistribution["disconnected_pair_count"] > 0:
        impact_level = "proxy_od_disconnection"
        plain_finding = (
            f"The {len(canonical):,} selected road change{'s' if len(canonical) != 1 else ''} are the initial disruption, not the complete impact. "
            f"The deterministic network test loses {redistribution['disconnected_pair_count']:,} of "
            f"{redistribution['comparable_pair_count']:,} baseline-reachable proxy routes and identifies "
            f"{redistribution['indirect_increase_count']:,} other links carrying additional rerouted paths."
        )
    elif redistribution["rerouted_pair_count"] > 0:
        impact_level = "rerouting_pressure"
        selected_plural = len(canonical) != 1
        rerouted_plural = redistribution["rerouted_pair_count"] != 1
        plain_finding = (
            f"The {len(canonical):,} selected road change{'s' if selected_plural else ''} "
            f"{'trigger' if selected_plural else 'triggers'} {redistribution['rerouted_pair_count']:,} rerouted proxy "
            f"route{'s' if rerouted_plural else ''} and {'shift' if selected_plural else 'shifts'} path load onto "
            f"{redistribution['indirect_increase_count']:,} other road links. Selected roads and indirect effects are shown separately."
        )
    elif redistribution["maximum_detour_percent"] > 1e-9:
        impact_level = "network_impedance"
        plain_finding = (
            f"The selected change{'s' if len(canonical) != 1 else ''} do not divert the tested routes, but they increase "
            f"mean route impedance by {redistribution['mean_detour_percent']:.1f}% "
            f"(maximum {redistribution['maximum_detour_percent']:.1f}%). No other link gains assigned load in this fixed OD design."
        )
    elif added_components > 0:
        impact_level = "network_split"
        plain_finding = (
            f"The disruption creates {added_components} additional weakly connected network part"
            f"{'s' if added_components != 1 else ''}; the fixed proxy routes did not identify a usable detour."
        )
    elif efficiency_loss >= 0.05:
        impact_level = "network_impedance"
        plain_finding = "The tested network remains connected but its topology-based movement efficiency decreases."
    else:
        impact_level = "limited_propagation"
        plain_finding = "The selected changes have limited effect on the fixed proxy route set; local unmodelled demand may still differ."

    if redistribution["covered_seed_count"] < redistribution["selected_seed_count"]:
        plain_finding += (
            f" {redistribution['covered_seed_count']:,} of {redistribution['selected_seed_count']:,} selected roads "
            "lie on their baseline local shortest-route checks; bypassed selected roads remain mapped as direct changes "
            "but do not receive invented proxy demand."
        )

    monotone = scenario_efficiency <= baseline_efficiency + max(1e-12, abs(baseline_efficiency) * 1e-9)
    assignment_supported = redistribution["verification"]["passed"]
    return {
        **base_result,
        "status": (
            "supported"
            if exact
            and baseline_graph.graph.get("impedance_mode") != "uniform_hop_count"
            and assignment_supported
            and redistribution["tier"] == "observed_flow_calibrated_capacity_ue_proxy"
            else "conditional"
        ),
        "impact_level": impact_level,
        "plain_finding": plain_finding,
        "indirect_impact_count": redistribution["indirect_increase_count"],
        "indirect_items": indirect_items,
        "relieved_items": redistribution["indirect_relief"],
        "recovery_screening": redistribution["recovery_screening"],
        "feature_collection": {"type": "FeatureCollection", "features": combined_features},
        "headline_effects": [
            {
                "label": "Initial roads route-tested",
                "value": f"{redistribution['covered_seed_count']:,}/{len(topology_changes):,}",
                "detail": f"Local From-to-To OD checks; deterministic cap {redistribution['seed_sampling_limit']:,}",
            },
            {
                "label": "Proxy routes rerouted",
                "value": f"{redistribution['rerouted_pair_percent']:.1f}%",
                "detail": f"{redistribution['rerouted_pair_count']:,} of {redistribution['comparable_pair_count']:,} baseline-reachable tests",
            },
            {
                "label": "Proxy routes disconnected",
                "value": f"{redistribution['disconnected_pair_percent']:.1f}%",
                "detail": f"{redistribution['disconnected_pair_count']:,} tests cannot complete after disruption",
            },
            {
                "label": "Other links under pressure",
                "value": f"{redistribution['indirect_increase_count']:,}",
                "detail": f"Beyond {len(topology_changes):,} user-selected initial links",
            },
        ],
        "visual": {
            "mode": "propagated_network_compare",
            "baseline_efficiency_percent": 100.0,
            "scenario_efficiency_percent": max(0.0, 1.0 - efficiency_loss) * 100.0,
            "baseline_components": baseline_components["weak_components"],
            "scenario_components": scenario_components["weak_components"],
            "largest_component_retained_percent": max(0.0, min(1.0, component_retention)) * 100.0,
            "rerouted_pair_percent": redistribution["rerouted_pair_percent"],
            "disconnected_pair_percent": redistribution["disconnected_pair_percent"],
            "mean_detour_percent": redistribution["mean_detour_percent"],
            "maximum_detour_percent": redistribution["maximum_detour_percent"],
            "initial_link_count": len(topology_changes),
            "indirect_link_count": redistribution["indirect_increase_count"],
        },
        "evidence": {
            "baseline_efficiency": baseline_efficiency,
            "scenario_efficiency": scenario_efficiency,
            "relative_efficiency_loss": efficiency_loss,
            "baseline_components": baseline_components,
            "scenario_components": scenario_components,
            "efficiency_exact": exact,
            "efficiency_source_count": source_count,
            "impedance_mode": baseline_graph.graph.get("impedance_mode"),
            "impedance_label": baseline_graph.graph.get("impedance_label"),
            "selected_time": selected_time,
            "assignment": redistribution,
            "operator_definitions": {
                "closed": "Remove the selected directed edge.",
                "speed_reduction": "Apply t' = t / (1-r), where r is the confirmed proportional speed loss.",
                "travel_time_increase": "Apply t' = t(1+m), where m is the confirmed proportional exogenous delay.",
            },
            "communication_rule": (
                "Selected roads are initial disruption seeds. An indirect road is returned only when the same fixed, "
                "baseline-reachable OD sensitivity design uses it more after the scenario operator is applied. "
                "The design combines globally spread anchors with local From-to-To checks for deterministically screened selected seeds."
            ),
        },
        "data_boundary": (
            "This is a deterministic capacity-constrained network counterfactual. No confirmed OD matrix is present, so selected-seed local OD checks and geographically spread proxy anchors are scaled only to the uploaded aggregate observed-link-flow total. The Beckmann/BPR tier identifies capacity-sensitive rerouting structure, not an observed OD assignment, hazard probability, casualties or a live operational instruction."
            if redistribution["tier"] == "observed_flow_calibrated_capacity_ue_proxy"
            else
            "This is a deterministic network counterfactual. Because no confirmed OD matrix is present, the rerouting layer uses equal unit demand over selected-seed local OD checks plus a deterministic, geographically spread set of topology anchors. It identifies route sensitivity and spillover structure, not actual traffic volume, hazard probability, casualties or a live operational instruction."
        ),
        "verification": {
            "passed": (
                len(direct_features) == len(canonical)
                and len(topology_changes) > 0
                and monotone
                and assignment_supported
                and all(feature["properties"].get("key") in records for feature in indirect_features)
            ),
            "checks": [
                "every change grounded in an uploaded entity",
                "one final change per entity",
                "baseline and scenario recomputed on the same directed topology",
                "non-improving disruption check",
                "same deterministic OD comparison design before and after disruption",
                "initial disruption and indirect rerouting links kept visually distinct",
                "no random sampling",
            ],
        },
    }
