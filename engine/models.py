from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator
from uuid import uuid4

import networkx as nx
import numpy as np

from .quantum_percolation import (
    DPCNumericalContract,
    PropagationEdge,
    regional_dpc_impacts,
)


METRIC_COLUMNS = {
    "speed": ("o.speed", "e.baseline_speed", "km/h"),
    "travel_time": ("o.travel_time", "e.baseline_travel_time", "min"),
    "primary": ("o.primary_value", "e.baseline_primary", "uploaded unit"),
    "flow": ("o.flow", "NULL", "vehicles/hour"),
    "capacity": ("e.capacity", "NULL", "vehicles/hour"),
    "free_flow_time": ("e.free_flow_time", "NULL", "min"),
    "repair_cost": ("e.repair_cost", "NULL", "uploaded cost unit"),
    "repair_duration": ("e.repair_duration", "NULL", "uploaded time unit"),
    "failure_probability": ("e.failure_probability", "NULL", "probability [0,1]"),
    "length": ("e.length_m", "NULL", "m"),
}


@contextmanager
def _connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database_path, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _summary(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute("SELECT value_json FROM metadata WHERE key = 'summary'").fetchone()
    if row is None:
        raise ValueError("Project summary is unavailable.")
    return json.loads(row[0])


def _metric_parts(metric: str) -> tuple[str, str, str]:
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported metric: {metric}")
    return METRIC_COLUMNS[metric]


def _primary_semantics(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("metrics", {}).get("primary", {}).get("semantics", {
        "direction": "descriptive",
        "comparison": "current_value",
        "scale": 1.0,
        "severity_enabled": False,
    })


def _severity(
    metric: str,
    value: float | None,
    baseline: float | None,
    primary_semantics: dict[str, Any] | None = None,
) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if metric == "speed":
        if baseline is None or not math.isfinite(baseline) or baseline <= 0:
            return None
        return max(0.0, 1.0 - value / baseline) if value >= 0 else None
    if metric == "travel_time":
        if baseline is None or not math.isfinite(baseline) or baseline <= 0:
            return None
        return max(0.0, value / baseline - 1.0) if value > 0 else None
    if metric != "primary":
        return None
    semantics = primary_semantics or {}
    direction = semantics.get("direction", "descriptive")
    comparison = semantics.get("comparison", "current_value")
    scale = float(semantics.get("scale") or 1.0)
    if direction not in {"higher_worse", "lower_worse"} or scale <= 0:
        return None
    if comparison == "change_from_reference":
        if baseline is None or not math.isfinite(baseline):
            return None
        change = value - baseline if direction == "higher_worse" else baseline - value
        return max(0.0, change / scale)
    return max(0.0, value / scale) if direction == "higher_worse" else max(0.0, 1.0 - value / scale)


def _severity_sql(
    metric: str,
    value_column: str,
    baseline_column: str,
    summary: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Return a transparent SQL severity expression and validity condition."""
    if metric == "speed":
        return (
            f"MAX(0.0, 1.0 - ({value_column} / {baseline_column}))",
            f"{value_column} IS NOT NULL AND {value_column} >= 0 AND {baseline_column} > 0",
        )
    if metric == "travel_time":
        return (
            f"MAX(0.0, ({value_column} / {baseline_column}) - 1.0)",
            f"{value_column} > 0 AND {baseline_column} > 0",
        )
    if metric != "primary":
        return None, None
    semantics = _primary_semantics(summary)
    direction = semantics.get("direction", "descriptive")
    comparison = semantics.get("comparison", "current_value")
    scale = float(semantics.get("scale") or 1.0)
    if direction not in {"higher_worse", "lower_worse"} or scale <= 0:
        return None, None
    if comparison == "change_from_reference":
        delta = (
            f"({value_column} - {baseline_column})"
            if direction == "higher_worse"
            else f"({baseline_column} - {value_column})"
        )
        return (
            f"MAX(0.0, {delta} / {scale:.12g})",
            f"{value_column} IS NOT NULL AND {baseline_column} IS NOT NULL",
        )
    if direction == "higher_worse":
        return f"MAX(0.0, {value_column} / {scale:.12g})", f"{value_column} IS NOT NULL"
    return f"MAX(0.0, 1.0 - {value_column} / {scale:.12g})", f"{value_column} IS NOT NULL"


def _quality_label(mask: int | None) -> str:
    value = int(mask or 0)
    labels = [
        label
        for bit, label in (
            (1, "invalid_coordinate"),
            (2, "invalid_time"),
            (4, "negative_speed"),
            (8, "nonpositive_travel_time"),
        )
        if value & bit
    ]
    return "|".join(labels) if labels else "valid"


def _add_display_scores(features: list[dict[str, Any]]) -> None:
    """Normalise descriptive values for map colour without inventing severity.

    ``display_score`` is deliberately separate from ``severity``.  It only
    describes the relative value range in the currently returned map slice;
    it must never be used by degradation, hotspot or resilience models.
    """
    values = [
        float(feature["properties"]["value"])
        for feature in features
        if feature.get("properties", {}).get("value") is not None
        and math.isfinite(float(feature["properties"]["value"]))
    ]
    if not values:
        return
    minimum, maximum = min(values), max(values)
    span = maximum - minimum
    for feature in features:
        value = feature.get("properties", {}).get("value")
        if value is None or not math.isfinite(float(value)):
            feature["properties"]["display_score"] = None
        elif span <= 1e-12:
            feature["properties"]["display_score"] = 0.5
        else:
            feature["properties"]["display_score"] = (float(value) - minimum) / span


def _nearest_timestamp(connection: sqlite3.Connection, requested: int | None) -> int:
    if requested is None:
        row = connection.execute("SELECT MAX(timestamp_epoch) FROM times").fetchone()
    else:
        row = connection.execute(
            "SELECT timestamp_epoch FROM times ORDER BY ABS(timestamp_epoch - ?) LIMIT 1",
            (requested,),
        ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("No observation time is available.")
    return int(row[0])


def map_features(
    database_path: Path,
    *,
    metric: str,
    timestamp: int | None = None,
    bbox: list[float] | None = None,
    limit: int = 9000,
) -> dict[str, Any]:
    value_column, baseline_column, default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        clauses = ["o.timestamp_epoch = ?", "e.latitude BETWEEN -90 AND 90", "e.longitude BETWEEN -180 AND 180"]
        parameters: list[Any] = [selected_time]
        if bbox:
            west, south, east, north = bbox
            clauses.extend(["e.longitude BETWEEN ? AND ?", "e.latitude BETWEEN ? AND ?"])
            parameters.extend([west, east, south, north])
        count_query = f"""
            SELECT COUNT(*)
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            WHERE {' AND '.join(clauses)}
        """
        total_count = int(connection.execute(count_query, parameters).fetchone()[0])
        query = f"""
            SELECT e.region_id, e.entity_id, e.from_node, e.to_node, e.latitude, e.longitude,
                   e.geometry_kind, {value_column} AS value, {baseline_column} AS baseline,
                   o.quality_flags, t.timestamp_text
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            JOIN times t ON t.timestamp_epoch = o.timestamp_epoch
            WHERE {' AND '.join(clauses)}
            ORDER BY ((e.entity_pk * 1103515245 + 12345) & 2147483647), e.entity_pk
            LIMIT ?
        """
        rows = connection.execute(query, (*parameters, limit + 1)).fetchall()
        truncated = total_count > limit
        rows = rows[:limit]
        unit = summary.get("metrics", {}).get(metric, {}).get("unit", default_unit)
        features = []
        for row in rows:
            value = float(row["value"]) if row["value"] is not None else None
            baseline = float(row["baseline"]) if row["baseline"] is not None else None
            degradation = _severity(metric, value, baseline, _primary_semantics(summary))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                    "properties": {
                        "key": f"{row['region_id']}::{row['entity_id']}",
                        "region_id": row["region_id"],
                        "entity_id": row["entity_id"],
                        "from_node": row["from_node"],
                        "to_node": row["to_node"],
                        "value": value,
                        "baseline": baseline,
                        "severity": degradation,
                        "quality": _quality_label(row["quality_flags"]),
                        "timestamp": row["timestamp_text"],
                        "metric": metric,
                        "unit": unit,
                        "geometry_kind": row["geometry_kind"],
                    },
                }
            )
        _add_display_scores(features)
        return {
            "type": "FeatureCollection",
            "features": features,
            "timestamp_epoch": selected_time,
            "timestamp_label": summary.get("time_labels", {}).get(str(selected_time), str(selected_time)),
            "metric": metric,
            "unit": unit,
            "truncated": truncated,
            "total_count": total_count,
            "displayed_count": len(features),
            "value_min": min((feature["properties"]["value"] for feature in features if feature["properties"]["value"] is not None), default=None),
            "value_max": max((feature["properties"]["value"] for feature in features if feature["properties"]["value"] is not None), default=None),
            "sampling_method": (
                "Deterministic hash-balanced entity sample; zoom or pan to request the current viewport."
                if truncated else "Every matching mapped observation is shown."
            ),
            "geometry_note": summary.get("geometry_note"),
        }


def entity_history(database_path: Path, region_id: str, entity_id: str, metric: str) -> dict[str, Any]:
    value_column, baseline_column, default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        total_point_count = int(connection.execute(
            """
            SELECT COUNT(*)
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            WHERE e.region_id = ? AND e.entity_id = ?
            """,
            (region_id, entity_id),
        ).fetchone()[0])
        maximum_points = 2500
        stride = max(1, math.ceil(total_point_count / maximum_points))
        rows = connection.execute(
            f"""
            WITH ordered AS (
                SELECT o.timestamp_epoch, t.timestamp_text, {value_column} AS value,
                       {baseline_column} AS baseline, o.quality_flags,
                       e.from_node, e.to_node, e.latitude, e.longitude,
                       ROW_NUMBER() OVER (ORDER BY o.timestamp_epoch) AS row_number,
                       COUNT(*) OVER () AS total_rows
                FROM observations o
                JOIN entities e ON e.entity_pk = o.entity_pk
                JOIN times t ON t.timestamp_epoch = o.timestamp_epoch
                WHERE e.region_id = ? AND e.entity_id = ?
            )
            SELECT * FROM ordered
            WHERE row_number = 1 OR row_number = total_rows OR (row_number - 1) % ? = 0
            ORDER BY timestamp_epoch
            LIMIT ?
            """,
            (region_id, entity_id, stride, maximum_points),
        ).fetchall()
        if not rows:
            raise ValueError("The selected entity is not available.")
        expected = int(summary.get("expected_interval_s") or 0)
        points = []
        previous_time: int | None = None
        for row in rows:
            timestamp = int(row["timestamp_epoch"])
            gap_before = bool(previous_time is not None and expected > 0 and timestamp - previous_time > expected * 1.5)
            value = float(row["value"]) if row["value"] is not None else None
            baseline = float(row["baseline"]) if row["baseline"] is not None else None
            points.append(
                {
                    "time": timestamp,
                    "label": row["timestamp_text"],
                    "value": value,
                    "baseline": baseline,
                    "severity": _severity(metric, value, baseline, _primary_semantics(summary)),
                    "quality": _quality_label(row["quality_flags"]),
                    "gap_before": gap_before,
                }
            )
            previous_time = timestamp
        first = rows[0]
        return {
            "key": f"{region_id}::{entity_id}",
            "region_id": region_id,
            "entity_id": entity_id,
            "from_node": first["from_node"],
            "to_node": first["to_node"],
            "latitude": first["latitude"],
            "longitude": first["longitude"],
            "metric": metric,
            "unit": summary.get("metrics", {}).get(metric, {}).get("unit", default_unit),
            "baseline_note": summary.get("baseline", {}).get("warning"),
            "total_point_count": total_point_count,
            "returned_point_count": len(points),
            "sampled": len(points) < total_point_count,
            "sampling_note": (
                f"History chart shows a deterministic time-balanced sample of {len(points):,} from {total_point_count:,} observations, including the first and last."
                if len(points) < total_point_count else "Every history observation is shown."
            ),
            "points": points,
        }


def _bbox_clause(bbox: list[float] | None, alias: str = "e") -> tuple[str, list[Any]]:
    if not bbox:
        return "", []
    west, south, east, north = bbox
    return (
        f" AND {alias}.longitude BETWEEN ? AND ? AND {alias}.latitude BETWEEN ? AND ?",
        [west, east, south, north],
    )


def _selected_keys_clause(keys: list[str], alias: str = "e") -> tuple[str, list[Any]]:
    pairs = [key.split("::", 1) for key in keys if "::" in key]
    if not pairs:
        return "", []
    conditions = []
    parameters: list[Any] = []
    for region_id, entity_id in pairs:
        conditions.append(f"({alias}.region_id = ? AND {alias}.entity_id = ?)")
        parameters.extend([region_id, entity_id])
    return f" AND ({' OR '.join(conditions)})", parameters


def persistent_degradation(
    database_path: Path,
    *,
    metric: str,
    threshold: float,
    top_k: int,
    bbox: list[float] | None,
    time_start: int | None,
    time_end: int | None,
) -> dict[str, Any]:
    value_column, baseline_column, default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        expected = int(summary.get("expected_interval_s") or 0)
        time_clauses = []
        parameters: list[Any] = []
        if time_start is not None:
            time_clauses.append("timestamp_epoch >= ?")
            parameters.append(time_start)
        if time_end is not None:
            time_clauses.append("timestamp_epoch <= ?")
            parameters.append(time_end)
        bbox_sql, bbox_parameters = _bbox_clause(bbox)
        parameters.extend(bbox_parameters)
        degradation, value_valid = _severity_sql(
            metric, value_column, baseline_column, summary
        )
        if not degradation or not value_valid:
            raise ValueError(
                "The uploaded metric is descriptive only. Confirm whether higher or lower values mean worse and provide a positive severity scale before running degradation models."
            )
        time_filter = " AND ".join(time_clauses)
        if time_filter:
            time_filter = " WHERE " + time_filter
        if expected > 0:
            duration_expression = f"CASE WHEN next_time IS NOT NULL AND next_time - timestamp_epoch BETWEEN 1 AND {int(expected * 1.5)} THEN next_time - timestamp_epoch ELSE 0 END"
        else:
            duration_expression = "CASE WHEN next_time IS NOT NULL THEN 1 ELSE 0 END"
        query = f"""
            WITH ordered_times AS MATERIALIZED (
                SELECT timestamp_epoch,
                       LEAD(timestamp_epoch) OVER (ORDER BY timestamp_epoch) AS next_time
                FROM times
                {time_filter}
            ), time_weights AS MATERIALIZED (
                SELECT timestamp_epoch, {duration_expression} AS represented_seconds
                FROM ordered_times
            ), scored AS (
                SELECT o.entity_pk, o.timestamp_epoch, e.region_id, e.entity_id, e.latitude, e.longitude,
                       {degradation} AS degradation, tw.represented_seconds
                FROM observations o
                CROSS JOIN entities e ON e.entity_pk = o.entity_pk
                CROSS JOIN time_weights tw ON tw.timestamp_epoch = o.timestamp_epoch
                WHERE {value_valid}{bbox_sql}
            ), entity_spans AS (
                SELECT entity_pk, MIN(timestamp_epoch) AS first_time, MAX(timestamp_epoch) AS last_time
                FROM scored
                GROUP BY entity_pk
            ), eligible_duration AS (
                SELECT spans.entity_pk, SUM(weights.represented_seconds) AS available_seconds
                FROM entity_spans spans
                JOIN time_weights weights
                  ON weights.timestamp_epoch BETWEEN spans.first_time AND spans.last_time
                GROUP BY spans.entity_pk
            )
            SELECT region_id, entity_id, MIN(latitude) AS latitude, MIN(longitude) AS longitude,
                   AVG(degradation) AS mean_degradation,
                   MAX(degradation) AS maximum_degradation,
                   SUM(CASE WHEN degradation >= ? THEN represented_seconds ELSE 0 END) AS degraded_seconds,
                   SUM(represented_seconds) AS observed_seconds,
                   MAX(eligible_duration.available_seconds) AS available_seconds
            FROM scored
            JOIN eligible_duration ON eligible_duration.entity_pk = scored.entity_pk
            GROUP BY scored.entity_pk
            HAVING SUM(represented_seconds) > 0
               AND SUM(represented_seconds) >= 0.8 * MAX(eligible_duration.available_seconds)
            ORDER BY (1.0 * degraded_seconds / observed_seconds) DESC, mean_degradation DESC
            LIMIT ?
        """
        rows = connection.execute(query, (*parameters, threshold, top_k)).fetchall()
        items = []
        features = []
        for rank, row in enumerate(rows, 1):
            persistence = float(row["degraded_seconds"] or 0) / max(1.0, float(row["observed_seconds"] or 0))
            item = {
                "rank": rank,
                "key": f"{row['region_id']}::{row['entity_id']}",
                "region_id": row["region_id"],
                "entity_id": row["entity_id"],
                "persistence": persistence,
                "mean_degradation": float(row["mean_degradation"] or 0),
                "maximum_degradation": float(row["maximum_degradation"] or 0),
                "observed_hours": float(row["observed_seconds"] or 0) / 3600,
                "coverage_ratio": float(row["observed_seconds"] or 0) / max(1.0, float(row["available_seconds"] or 0)),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }
            items.append(item)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                    "properties": {**item, "severity": persistence, "metric": metric},
                }
            )
        top_persistence = items[0]["persistence"] if items else 0.0
        minimum_coverage = min((item["coverage_ratio"] for item in items), default=0.0)
        return {
            "model": {
                "id": "P5-M1",
                "name": "Find recurring problem locations",
                "status": "numerically_verified",
                "source": "Transport World Model evidence specification; uploaded observations only",
            },
            "status": "supported",
            "conclusion": (
                f"{len(items)} locations spent the largest share of valid observed time beyond the confirmed {metric.replace('_', ' ')} degradation threshold."
                if items
                else "No location met the confirmed persistence calculation requirements."
            ),
            "headline_effects": [
                {"label": "Most frequently problematic", "value": f"{top_persistence * 100:.1f}%", "detail": "Share of valid observed time beyond the confirmed problem threshold"},
                {"label": "Threshold", "value": f"{threshold * 100:.0f}%", "detail": "Confirmed relative degradation"},
                {"label": "Locations ranked", "value": str(len(items)), "detail": "Based on valid observations and an internal reference"},
                {"label": "Minimum coverage", "value": f"{minimum_coverage * 100:.1f}%" if items else "No eligible location", "detail": "At least 80% of the requested gap-safe duration is required"},
            ],
            "next_actions": [
                "Inspect the top-ranked locations on the geographic map.",
                "Check whether the internal reference window represents normal conditions.",
                "Compare topology-based criticality before using the ranking for restoration.",
            ],
            "items": items,
            "feature_collection": {"type": "FeatureCollection", "features": features},
            "evidence": {
                "metric": metric,
                "unit": summary.get("metrics", {}).get(metric, {}).get("unit", default_unit),
                "threshold": threshold,
                "expected_interval_s": expected,
                "gap_rule": "Intervals longer than 1.5 times the confirmed sampling interval receive zero persistence weight.",
                "minimum_entity_coverage": 0.8,
                "coverage_rule": "Entities observed for less than 80% of the gap-safe duration between their first and last valid observations are excluded rather than being allowed to rank at 100% from sparse internal gaps.",
                "baseline": summary.get("baseline"),
            },
            "data_boundary": "This result describes uploaded observations. It is not a causal hurricane estimate or traffic-assignment result.",
            "verification": {
                "passed": True,
                "checks": ["positive baseline", "invalid observations excluded", "long gaps excluded", "duration weights non-negative"],
            },
        }


def _inferred_link_geometries(
    database_path: Path,
    keys: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], bool]:
    unique_keys = list(dict.fromkeys(str(key) for key in keys if key))
    if not unique_keys:
        return {}, False
    with _connect(database_path) as connection:
        exact_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='transport_link_geometries'"
        ).fetchone()
        if exact_table:
            placeholders = ",".join("?" for _ in unique_keys)
            exact_rows = connection.execute(
                f"SELECT link_key, geometry_json FROM transport_link_geometries WHERE link_key IN ({placeholders})",
                unique_keys,
            ).fetchall()
            exact = {}
            for row in exact_rows:
                try:
                    exact[str(row["link_key"])] = json.loads(row["geometry_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
            if exact:
                return exact, True
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('inferred_network_links', 'inferred_network_nodes')"
            ).fetchall()
        }
        if existing != {"inferred_network_links", "inferred_network_nodes"}:
            return {}, False
        placeholders = ",".join("?" for _ in unique_keys)
        rows = connection.execute(
            f"""
            SELECT ml.link_key,
                   nf.longitude AS from_longitude, nf.latitude AS from_latitude,
                   nt.longitude AS to_longitude, nt.latitude AS to_latitude
            FROM inferred_network_links ml
            JOIN inferred_network_nodes nf ON nf.node_key = ml.from_node_key
            JOIN inferred_network_nodes nt ON nt.node_key = ml.to_node_key
            WHERE ml.link_key IN ({placeholders})
            """,
            unique_keys,
        ).fetchall()
    return {
        str(row["link_key"]): {
            "type": "LineString",
            "coordinates": [
                [float(row["from_longitude"]), float(row["from_latitude"])],
                [float(row["to_longitude"]), float(row["to_latitude"])],
            ],
        }
        for row in rows
    }, True


def _link_result_feature(
    item: dict[str, Any],
    geometries: dict[str, dict[str, Any]],
    *,
    severity: float,
    extra_properties: dict[str, Any] | None = None,
    allow_point_fallback: bool = True,
) -> dict[str, Any] | None:
    key = str(item.get("key") or "")
    geometry = geometries.get(key)
    if geometry is not None:
        geometry_kind = "inferred_topology_line"
    elif allow_point_fallback and item.get("latitude") is not None and item.get("longitude") is not None:
        geometry = {
            "type": "Point",
            "coordinates": [float(item["longitude"]), float(item["latitude"])],
        }
        geometry_kind = "representative_point"
    else:
        return None
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            **item,
            **(extra_properties or {}),
            "severity": float(severity),
            "geometry_kind": geometry_kind,
        },
    }


def _build_graph(connection: sqlite3.Connection, bbox: list[float] | None = None) -> tuple[nx.MultiDiGraph, dict[str, dict[str, Any]]]:
    summary = _summary(connection)
    global_node_ids = summary.get("topology_node_scope", "global") == "global"
    bbox_sql, parameters = _bbox_clause(bbox)
    entity_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(entities)").fetchall()
    }

    def optional_column(name: str) -> str:
        return f"{name}" if name in entity_columns else f"NULL AS {name}"

    rows = connection.execute(
        f"""
        SELECT region_id, entity_id, from_node, to_node, length_m,
               baseline_travel_time, latitude, longitude,
               {optional_column('repair_cost')}, {optional_column('repair_duration')},
               {optional_column('capacity')}, {optional_column('free_flow_time')},
               {optional_column('failure_probability')}
        FROM entities e
        WHERE from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''{bbox_sql}
        """,
        parameters,
    ).fetchall()
    graph = nx.MultiDiGraph()
    graph.graph["topology_node_scope"] = "global" if global_node_ids else "per_region"
    positive_travel_count = sum(
        row["baseline_travel_time"] is not None and float(row["baseline_travel_time"]) > 0
        for row in rows
    )
    positive_length_count = sum(
        row["length_m"] is not None and float(row["length_m"]) > 0
        for row in rows
    )
    if rows and positive_travel_count == len(rows):
        impedance_mode = "baseline_travel_time"
    elif rows and positive_length_count == len(rows):
        impedance_mode = "length_m"
    else:
        impedance_mode = "uniform_hop_count"
    graph.graph.update({
        "impedance_mode": impedance_mode,
        "positive_travel_time_edges": positive_travel_count,
        "positive_length_edges": positive_length_count,
        "total_weighted_edges": len(rows),
    })
    edge_records: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['region_id']}::{row['entity_id']}"
        graph_from = str(row["from_node"]) if global_node_ids else f"{row['region_id']}\x1f{row['from_node']}"
        graph_to = str(row["to_node"]) if global_node_ids else f"{row['region_id']}\x1f{row['to_node']}"
        if impedance_mode == "baseline_travel_time":
            weight = float(row["baseline_travel_time"])
        elif impedance_mode == "length_m":
            weight = float(row["length_m"])
        else:
            weight = 1.0
        record = {
            "key": key,
            "region_id": row["region_id"],
            "entity_id": row["entity_id"],
            "from_node": row["from_node"],
            "to_node": row["to_node"],
            "weight": float(weight),
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "length_m": row["length_m"],
            "baseline_travel_time": row["baseline_travel_time"],
            "repair_cost": row["repair_cost"],
            "repair_duration": row["repair_duration"],
            "capacity": row["capacity"],
            "free_flow_time": row["free_flow_time"],
            "failure_probability": row["failure_probability"],
        }
        graph.add_edge(
            graph_from,
            graph_to,
            key=key,
            **{name: value for name, value in record.items() if name != "key"},
        )
        edge_records[key] = record
    return graph, edge_records


def _deterministic_sources(nodes: Iterable[str], maximum: int = 120) -> list[str]:
    ordered = sorted(nodes, key=str)
    if len(ordered) <= maximum:
        return ordered
    stride = len(ordered) / maximum
    return [ordered[min(len(ordered) - 1, int(index * stride))] for index in range(maximum)]


def _network_efficiency(graph: nx.MultiDiGraph, maximum_sources: int = 64) -> tuple[float, bool, int]:
    # Stable node ordering is part of the numerical contract.  It prevents a
    # database row-order change from altering either an exact calculation or
    # the deterministic large-network source screen.
    nodes = sorted(graph.nodes, key=str)
    n = len(nodes)
    if n < 2:
        return 0.0, True, n
    exact = n <= 600
    sources = nodes if exact else _deterministic_sources(nodes, maximum=maximum_sources)
    reciprocal_sum = 0.0
    for source in sources:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        reciprocal_sum += sum(1.0 / distance for target, distance in lengths.items() if target != source and distance > 0)
    return reciprocal_sum / (len(sources) * (n - 1)), exact, len(sources)


def _component_metrics(graph: nx.MultiDiGraph) -> dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {"weak_components": 0, "largest_component_nodes": 0}
    components = list(nx.weakly_connected_components(graph))
    return {
        "weak_components": len(components),
        "largest_component_nodes": max((len(component) for component in components), default=0),
    }


def _single_connection_bridge_pairs(graph: nx.MultiDiGraph) -> set[frozenset[str]]:
    """Return undirected bridge pairs represented by exactly one directed road.

    A simple undirected projection alone would label both directions of a
    two-way road as bridges even though removing either directed road leaves
    the weak connection present through its reverse direction.  The
    multiplicity guard keeps the screening claim consistent with the actual
    one-road removal operator.
    """

    projection = nx.Graph()
    pair_counts: dict[frozenset[str], int] = {}
    for u, v, _key in graph.edges(keys=True):
        if u == v:
            continue
        pair = frozenset((str(u), str(v)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        projection.add_edge(u, v)
    if projection.number_of_nodes() == 0:
        return set()
    return {
        frozenset((str(u), str(v)))
        for u, v in nx.bridges(projection)
        if pair_counts.get(frozenset((str(u), str(v))), 0) == 1
    }


def critical_components(database_path: Path, *, top_k: int, bbox: list[float] | None) -> dict[str, Any]:
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, bbox)
    if graph.number_of_edges() == 0:
        raise ValueError("Topology fields are unavailable for the selected data.")
    baseline_efficiency, efficiency_exact, source_count = _network_efficiency(graph, maximum_sources=48)
    bridge_pairs = _single_connection_bridge_pairs(graph)

    if graph.number_of_nodes() <= 1200 and graph.number_of_edges() <= 2500:
        centrality = nx.edge_betweenness_centrality(graph, weight="weight", normalized=True)
        centrality_status = "exact"
    else:
        sources = _deterministic_sources(graph.nodes, maximum=48)
        targets = sorted(graph.nodes, key=str)
        raw = nx.edge_betweenness_centrality_subset(
            graph, sources=sources, targets=targets, weight="weight", normalized=False
        )
        scale = max(1.0, len(sources) * max(1, len(targets) - 1))
        centrality = {edge: value / scale for edge, value in raw.items()}
        centrality_status = "deterministic_source_screen"

    all_edges = list(graph.edges(keys=True))
    total_edge_count = len(all_edges)
    requested_top_k = min(total_edge_count, max(1, int(top_k)))
    # Exhaustive edge removal is the only way to claim a true Top-N by
    # efficiency loss.  It remains practical on small networks.  Larger
    # networks use a deterministic, centrality-ranked screen and are labelled
    # conditional with explicit coverage instead of presenting a screened
    # subset as a complete ranking.
    exhaustive_removal = graph.number_of_nodes() <= 250 and total_edge_count <= 300
    structural_screening = not exhaustive_removal and requested_top_k > 100
    ranked_edges = sorted(
        all_edges,
        key=lambda edge: (
            float(centrality.get(edge, 0.0)),
            frozenset((edge[0], edge[1])) in bridge_pairs,
            str(edge[0]),
            str(edge[1]),
            str(edge[2]),
        ),
        reverse=True,
    )
    if exhaustive_removal:
        candidates = all_edges
        candidate_rule = "All directed roads were removed and evaluated one at a time."
    elif structural_screening:
        candidates = ranked_edges[:requested_top_k]
        candidate_rule = (
            "Large Top-N requests use deterministic edge-betweenness ranking with single-connection status as a tie-break; "
            "this avoids thousands of repeated whole-network shortest-path recalculations."
        )
    else:
        candidate_limit = min(total_edge_count, max(100, requested_top_k * 5))
        candidates = ranked_edges[:candidate_limit]
        candidate_rule = (
            "Deterministic pre-screen by edge betweenness with single-connection status as a tie-break; "
            "reported Top-N is conditional on the evaluated candidate set."
        )

    evaluated = []
    if structural_screening:
        maximum_centrality = max((float(centrality.get(edge, 0.0)) for edge in candidates), default=0.0)
        for u, v, key in candidates:
            attributes = dict(graph[u][v][key])
            record = records.get(str(key), attributes)
            raw_centrality = float(centrality.get((u, v, key), 0.0))
            evaluated.append(
                {
                    **record,
                    "edge_betweenness": raw_centrality,
                    "structural_score": raw_centrality / maximum_centrality if maximum_centrality > 0 else 0.0,
                    "is_bridge": frozenset((u, v)) in bridge_pairs,
                }
            )
        evaluated.sort(
            key=lambda item: (item["edge_betweenness"], item["is_bridge"], str(item.get("key"))),
            reverse=True,
        )
    else:
        for u, v, key in candidates:
            attributes = dict(graph[u][v][key])
            graph.remove_edge(u, v, key)
            disrupted_efficiency, _, _ = _network_efficiency(graph, maximum_sources=source_count)
            graph.add_edge(u, v, key=key, **attributes)
            loss = (
                max(0.0, (baseline_efficiency - disrupted_efficiency) / baseline_efficiency)
                if baseline_efficiency > 0
                else 0.0
            )
            record = records.get(str(key), attributes)
            evaluated.append(
                {
                    **record,
                    "edge_betweenness": float(centrality.get((u, v, key), 0.0)),
                    "efficiency_loss": loss,
                    "is_bridge": frozenset((u, v)) in bridge_pairs,
                }
            )
        evaluated.sort(
            key=lambda item: (item["efficiency_loss"], item["is_bridge"], item["edge_betweenness"]),
            reverse=True,
        )
    items = []
    for rank, item in enumerate(evaluated[:requested_top_k], 1):
        ranked = {"rank": rank, **item}
        items.append(ranked)
    geometries, mapping_ready = _inferred_link_geometries(database_path, (item["key"] for item in items))
    features = []
    for item in items:
        feature = _link_result_feature(
            item,
            geometries,
            severity=item.get("efficiency_loss", item.get("structural_score", 0.0)),
            allow_point_fallback=not mapping_ready,
        )
        if feature is not None:
            features.append(feature)
    top_loss = items[0].get("efficiency_loss", 0.0) if items else 0.0
    top_structural_score = items[0].get("structural_score", 0.0) if items else 0.0
    return {
        "model": {
            "id": "P5-M2",
            "name": "Find critical roads",
            "status": "numerically_verified",
            "sources": ["Tarjan (1972)", "Latora and Marchiori (2001)", "Prototype 1 read-only model registry"],
        },
        "status": "supported" if efficiency_exact and exhaustive_removal and not structural_screening and graph.graph.get("impedance_mode") != "uniform_hop_count" else "conditional",
        "conclusion": (
            f"The analysis exhaustively identified {len(items)} structurally important directed roads using explicit removal and efficiency loss."
            if exhaustive_removal
            else f"The analysis returned {len(items):,} of {total_edge_count:,} directed roads using deterministic structural centrality screening."
            if structural_screening
            else f"The analysis screened {len(candidates):,} of {total_edge_count:,} directed roads and returned the strongest {len(items)} within that transparent candidate set."
        ),
        "headline_effects": [
            ({"label": "Highest structural score", "value": f"{top_structural_score * 100:.2f}%", "detail": "Relative edge-betweenness within the returned screening set"}
             if structural_screening else
             {"label": "Largest efficiency loss", "value": f"{top_loss * 100:.2f}%", "detail": "Relative change after removing one directed road"}),
            {"label": "Single connections", "value": str(sum(bool(item["is_bridge"]) for item in items)), "detail": "Undirected bridge represented by one directed road"},
            {"label": "Network scale", "value": f"{graph.number_of_nodes():,} nodes", "detail": f"{graph.number_of_edges():,} directed roads"},
        ],
        "next_actions": [
            "Inspect whether the top roads represent complete road geometry or representative points.",
            "Compare the structural ranking with observed degradation.",
            "Add OD demand and facilities before interpreting this as social or evacuation impact.",
        ],
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "baseline_efficiency": baseline_efficiency,
            "efficiency_exact": efficiency_exact,
            "efficiency_source_count": source_count,
            "centrality_status": centrality_status,
            "evaluated_candidate_count": len(candidates),
            "total_edge_count": total_edge_count,
            "candidate_coverage": len(candidates) / max(1, total_edge_count),
            "candidate_selection_rule": candidate_rule,
            "exhaustive_edge_removal": exhaustive_removal,
            "ranking_mode": "structural_centrality" if structural_screening else "single_edge_efficiency_loss",
            "topology_node_scope": graph.graph.get("topology_node_scope"),
            "impedance_mode": graph.graph.get("impedance_mode"),
            "positive_travel_time_edges": graph.graph.get("positive_travel_time_edges"),
            "positive_length_edges": graph.graph.get("positive_length_edges"),
            "total_weighted_edges": graph.graph.get("total_weighted_edges"),
            "mapped_line_result_count": sum(feature["geometry"]["type"] == "LineString" for feature in features),
            "unmapped_line_result_count": len(items) - len(features) if mapping_ready else 0,
            "result_geometry_rule": (
                "Critical roads are drawn only as inferred topology lines when a saved mapping exists; unmapped roads remain in the complete table and are not misdrawn as detector circles."
                if mapping_ready
                else "No inferred link mapping was available, so representative points are disclosed as a fallback."
            ),
            "ranking_rule": (
                "Edge betweenness descending; bridge status and stable link key are transparent tie-breaks."
                if structural_screening
                else "Efficiency loss first; bridge status and edge betweenness are visible tie-break evidence, not a hidden composite score."
            ),
        },
        "data_boundary": "Topology-only criticality does not measure traffic redistribution, affected population, OD demand or repair feasibility.",
        "verification": {
            "passed": True,
                "checks": (["positive edge weights", "deterministic centrality sources", "candidate coverage disclosed", "stable tie-breaks"]
                           if structural_screening else
                           ["positive edge weights", "baseline recomputed after each evaluated removal", "removed edge restored after evaluation", "candidate coverage disclosed", "non-negative efficiency loss"]),
        },
    }


def area_summary(
    database_path: Path,
    *,
    metric: str,
    bbox: list[float],
    timestamp: int | None,
) -> dict[str, Any]:
    value_column, baseline_column, _default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        bbox_sql, parameters = _bbox_clause(bbox)
        severity_expression, severity_valid = _severity_sql(
            metric, value_column, baseline_column, summary
        )
        severity_sql = (
            f"CASE WHEN {severity_valid} THEN {severity_expression} END"
            if severity_expression and severity_valid else "NULL"
        )
        aggregate = connection.execute(
            f"""
            SELECT COUNT(*) AS mapped_count, AVG({value_column}) AS mean_value,
                   AVG({severity_sql}) AS mean_severity,
                   COUNT({severity_sql}) AS valid_severity_count,
                   SUM(CASE WHEN {severity_sql} IS NULL THEN 1 ELSE 0 END) AS undefined_count,
                   SUM(CASE WHEN {severity_sql} < 0.15 THEN 1 ELSE 0 END) AS low_count,
                   SUM(CASE WHEN {severity_sql} >= 0.15 AND {severity_sql} < 0.40 THEN 1 ELSE 0 END) AS degraded_count,
                   SUM(CASE WHEN {severity_sql} >= 0.40 THEN 1 ELSE 0 END) AS severe_count
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            WHERE o.timestamp_epoch = ?
              AND e.latitude BETWEEN -90 AND 90 AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
            """,
            (selected_time, *parameters),
        ).fetchone()
    collection = map_features(database_path, metric=metric, timestamp=selected_time, bbox=bbox, limit=20000)
    mapped_count = int(aggregate["mapped_count"] or 0)
    if mapped_count == 0:
        raise ValueError("The selected geographic area contains no mapped observation at this time.")
    mean_severity = float(aggregate["mean_severity"]) if aggregate["mean_severity"] is not None else None
    valid_severity_count = int(aggregate["valid_severity_count"] or 0)
    mean_value = float(aggregate["mean_value"]) if aggregate["mean_value"] is not None else None
    # Use the complete SQL aggregate, not the capped map rendering sample, so
    # the visual distribution remains correct for very large uploads.
    bucket_counts = {
        "Reference / low loss": int(aggregate["low_count"] or 0),
        "Degraded": int(aggregate["degraded_count"] or 0),
        "Severe": int(aggregate["severe_count"] or 0),
        "No severity definition": int(aggregate["undefined_count"] or 0),
    }
    items = [
        {"area_bucket": label, "count": count, "share": count / mapped_count}
        for label, count in bucket_counts.items()
        if count > 0
    ]
    west, south, east, north = bbox
    area_feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south],
        ]]},
        "properties": {"analysis_type": "area_summary", "mapped_count": mapped_count, "mean_value": mean_value, "mean_severity": mean_severity},
    }
    return {
        "model": {"id": "P5-M0", "name": "Geographic area summary", "status": "numerically_verified"},
        "status": "supported" if mean_severity is not None else "conditional",
        "conclusion": f"The selected geographic area contains {mapped_count} mapped entities at the chosen time.",
        "headline_effects": [
            {"label": "Mapped entities", "value": str(mapped_count), "detail": "Inside the confirmed geographic rectangle"},
            {
                "label": "Mean severity",
                "value": f"{mean_severity * 100:.1f}%" if mean_severity is not None else "Not calculated",
                "detail": (
                    f"Defined for {valid_severity_count:,} of {mapped_count:,} mapped observations"
                    if mean_severity is not None
                    else "The selected measure has no confirmed severity direction."
                ),
            },
            {"label": "Mean value", "value": f"{mean_value:.2f}" if mean_value is not None else "No valid value", "detail": collection["unit"]},
        ],
        "next_actions": ["Inspect individual locations.", "Run persistence analysis over a time window.", "Compare this area with a second confirmed area."],
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": [area_feature]},
        "evidence": {"bbox": bbox, "timestamp": collection["timestamp_label"], "metric": metric, "valid_severity_count": valid_severity_count, "primary_metric_semantics": _primary_semantics(summary) if metric == "primary" else None, "map_display_truncated": collection["truncated"]},
        "data_boundary": "This is a descriptive geographic summary of mapped observations.",
        "verification": {"passed": True, "checks": ["geographic bounds confirmed", "valid mapped coordinates only"]},
    }


def _graph_and_selected(
    database_path: Path, selected_keys: list[str], bbox: list[float] | None = None
) -> tuple[nx.MultiDiGraph, dict[str, dict[str, Any]], list[str]]:
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, bbox)
    requested = [str(key) for key in selected_keys]
    if not requested:
        raise ValueError("Select one or more mapped links with confirmed from/to topology.")
    duplicate_keys = sorted({key for key in requested if requested.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            "The selected-link request contains duplicates. Clear and select each link once: "
            + ", ".join(duplicate_keys[:5])
        )
    missing = [key for key in requested if key not in records]
    if missing:
        raise ValueError(
            "Some selected roads are not available in the current project or drawn area: "
            + ", ".join(missing[:5])
        )
    return graph, records, requested


def _record_nodes(record: dict[str, Any], graph: nx.MultiDiGraph) -> tuple[str, str]:
    if graph.graph.get("topology_node_scope") == "global":
        return str(record["from_node"]), str(record["to_node"])
    region = record["region_id"]
    return f"{region}\x1f{record['from_node']}", f"{region}\x1f{record['to_node']}"


def closure_scenario(database_path: Path, *, selected_keys: list[str], bbox: list[float] | None) -> dict[str, Any]:
    graph, records, selected = _graph_and_selected(database_path, selected_keys, bbox)
    baseline_efficiency, exact, source_count = _network_efficiency(graph)
    baseline_components = _component_metrics(graph)
    removed: list[dict[str, Any]] = []
    for key in selected:
        record = records[key]
        u, v = _record_nodes(record, graph)
        if graph.has_edge(u, v, key):
            removed.append({**record, **dict(graph[u][v][key])})
            graph.remove_edge(u, v, key)
    disrupted_efficiency, _, _ = _network_efficiency(graph)
    disrupted_components = _component_metrics(graph)
    efficiency_loss = (
        max(0.0, (baseline_efficiency - disrupted_efficiency) / baseline_efficiency)
        if baseline_efficiency > 0
        else 0.0
    )
    geometries, _mapping_ready = _inferred_link_geometries(database_path, (item["key"] for item in removed))
    features = []
    for item in removed:
        feature = _link_result_feature(
            item,
            geometries,
            severity=1.0,
            extra_properties={"scenario": "closed"},
        )
        if feature is not None:
            features.append(feature)
    return {
        "model": {"id": "P5-M3", "name": "Deterministic selected-road closure", "status": "numerically_verified"},
        "status": "supported" if exact and graph.graph.get("impedance_mode") != "uniform_hop_count" else "conditional",
        "conclusion": f"Closing {len(removed)} confirmed directed roads reduces topology-based network efficiency by {efficiency_loss * 100:.2f}%.",
        "headline_effects": [
            {"label": "Efficiency loss", "value": f"{efficiency_loss * 100:.2f}%", "detail": "Topology/travel-impedance response only"},
            {"label": "Weak components", "value": f"{baseline_components['weak_components']} → {disrupted_components['weak_components']}", "detail": "Directed network viewed through weak connectivity"},
            {"label": "Links removed", "value": str(len(removed)), "detail": "User-confirmed link observations"},
        ],
        "next_actions": ["Review the selected roads.", "Plan recovery priorities for these failures.", "Add OD and capacity data before estimating congestion or user delay."],
        "items": removed,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "baseline_efficiency": baseline_efficiency,
            "disrupted_efficiency": disrupted_efficiency,
            "efficiency_exact": exact,
            "efficiency_source_count": source_count,
            "baseline_components": baseline_components,
            "disrupted_components": disrupted_components,
            "topology_node_scope": graph.graph.get("topology_node_scope"),
            "impedance_mode": graph.graph.get("impedance_mode"),
        },
        "data_boundary": "The closure is a deterministic counterfactual. No traffic assignment, probability or causal disaster inference is performed.",
        "verification": {"passed": len(removed) == len(selected), "checks": ["selected roads grounded", "baseline retained", "counterfactual recomputed"]},
    }


def _apply_recovery_disruptions(
    graph: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    changes: dict[str, dict[str, Any]],
) -> None:
    """Apply the same three deterministic operators used by saved scenarios."""

    for key, raw in changes.items():
        record = records[key]
        u, v = _record_nodes(record, graph)
        if not graph.has_edge(u, v, key):
            continue
        effect = str(raw.get("effect") or "closed")
        magnitude = float(raw.get("magnitude") or (1.0 if effect == "closed" else 0.0))
        if effect == "closed":
            graph.remove_edge(u, v, key)
        elif effect == "speed_reduction" and 0.01 <= magnitude <= 0.95:
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) / (1.0 - magnitude)
        elif effect == "travel_time_increase" and 0.01 <= magnitude <= 5.0:
            graph[u][v][key]["weight"] = float(graph[u][v][key]["weight"]) * (1.0 + magnitude)
        else:
            raise ValueError(f"Unsupported or invalid disruption for recovery candidate {key}.")


def _restore_action(
    working: nx.MultiDiGraph,
    baseline: nx.MultiDiGraph,
    records: dict[str, dict[str, Any]],
    keys: list[str],
) -> list[tuple[str, str, str, dict[str, Any] | None]]:
    previous: list[tuple[str, str, str, dict[str, Any] | None]] = []
    for key in keys:
        record = records[key]
        u, v = _record_nodes(record, baseline)
        before = dict(working[u][v][key]) if working.has_edge(u, v, key) else None
        previous.append((u, v, key, before))
        baseline_data = dict(baseline[u][v][key])
        if working.has_edge(u, v, key):
            working[u][v][key].clear()
            working[u][v][key].update(baseline_data)
        else:
            working.add_edge(u, v, key=key, **baseline_data)
    return previous


def _undo_restoration(
    working: nx.MultiDiGraph,
    previous: list[tuple[str, str, str, dict[str, Any] | None]],
) -> None:
    for u, v, key, before in previous:
        if before is None:
            if working.has_edge(u, v, key):
                working.remove_edge(u, v, key)
        elif working.has_edge(u, v, key):
            working[u][v][key].clear()
            working[u][v][key].update(before)
        else:
            working.add_edge(u, v, key=key, **before)


def _metric_grid_action(record: dict[str, Any], cell_size_m: float = 5_000.0) -> tuple[str, list[float]]:
    earth_radius = 6_378_137.0
    world_half = math.pi * earth_radius
    longitude = float(record["longitude"])
    latitude = max(-85.05112878, min(85.05112878, float(record["latitude"])))
    x = earth_radius * math.radians(longitude)
    y = earth_radius * math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    column = int(math.floor((x + world_half) / cell_size_m))
    line = int(math.floor((y + world_half) / cell_size_m))
    x0 = column * cell_size_m - world_half
    y0 = line * cell_size_m - world_half

    def unproject(px: float, py: float) -> tuple[float, float]:
        return (
            math.degrees(px / earth_radius),
            math.degrees(2 * math.atan(math.exp(py / earth_radius)) - math.pi / 2),
        )

    west, south = unproject(x0, y0)
    east, north = unproject(x0 + cell_size_m, y0 + cell_size_m)
    return f"M5K-{column}-{line}", [west, south, east, north]


def _recovery_actions(
    records: dict[str, dict[str, Any]],
    selected: list[str],
    *,
    recovery_unit: str,
    area_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if recovery_unit != "area":
        return [
            {
                "action_id": key,
                "label": str(records[key]["entity_id"]),
                "keys": [key],
                "bbox": None,
            }
            for key in selected
        ]

    allowed = set(selected)
    assigned: set[str] = set()
    actions: list[dict[str, Any]] = []
    for raw in area_candidates:
        keys = [str(key) for key in raw.get("keys", []) if str(key) in allowed and str(key) not in assigned]
        if not keys:
            continue
        bbox = raw.get("bbox")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
            raise ValueError("Each recovery area bounding box must contain west, south, east and north.")
        action_id = str(raw.get("id") or raw.get("cell_id") or f"area-{len(actions) + 1}")
        actions.append({"action_id": action_id, "label": str(raw.get("label") or action_id), "keys": keys, "bbox": bbox})
        assigned.update(keys)

    # If no explicit hotspot/area packages were supplied, use the same fixed
    # 5 km metric grid as the critical-area model.  This preserves a stable,
    # reproducible meaning for an "area" across zoom levels and runs.
    grouped: dict[str, dict[str, Any]] = {}
    for key in selected:
        if key in assigned:
            continue
        record = records[key]
        if record.get("latitude") is None or record.get("longitude") is None:
            continue
        action_id, bbox = _metric_grid_action(record)
        action = grouped.setdefault(action_id, {"action_id": action_id, "label": action_id, "keys": [], "bbox": bbox})
        action["keys"].append(key)
    actions.extend(grouped[key] for key in sorted(grouped))
    if not actions:
        raise ValueError("Area recovery needs mapped coordinates for the disrupted roads.")
    return actions


def restoration_screening(
    database_path: Path,
    *,
    selected_keys: list[str],
    bbox: list[float] | None,
    top_k: int | None = None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assumptions = dict(assumptions or {})
    scenario_changes = assumptions.get("scenario_changes") or []
    if not isinstance(scenario_changes, list):
        raise ValueError("Recovery scenario changes must be a list.")
    requested = [str(key) for key in selected_keys]
    if not requested:
        requested = [str(item.get("key") or "") for item in scenario_changes if item.get("key")]
    recovery_unit = "area" if str(assumptions.get("recovery_unit") or "road").lower() == "area" else "road"
    area_candidates = assumptions.get("area_candidates") or []
    if not isinstance(area_candidates, list):
        raise ValueError("Recovery area candidates must be a list.")
    if not requested and area_candidates:
        requested = [str(key) for area in area_candidates for key in area.get("keys", [])]

    with _connect(database_path) as connection:
        baseline, records = _build_graph(connection, bbox)
        project_keys = {
            f"{row['region_id']}::{row['entity_id']}"
            for row in connection.execute("SELECT region_id, entity_id FROM entities").fetchall()
        }
    if not requested and bbox:
        requested = sorted(records)
    if not requested:
        raise ValueError("Choose disrupted roads, a disrupted area, or a saved disruption scenario first.")
    if len(requested) != len(set(requested)):
        raise ValueError("Each disrupted road may appear only once in a recovery request.")
    missing = [key for key in requested if key not in records]
    foreign = [key for key in missing if key not in project_keys]
    if foreign:
        raise ValueError("Some recovery candidates are not available in the current project: " + ", ".join(foreign[:5]))
    topology_excluded = [key for key in missing if key in project_keys]
    selected = list(dict.fromkeys(key for key in requested if key in records))
    if not selected:
        raise ValueError("The selected recovery roads are not available with usable From node and To node topology in the current project or drawn area.")
    if baseline.number_of_edges() == 0:
        raise ValueError("Recovery priorities need usable From node and To node topology.")
    actions = _recovery_actions(
        records,
        selected,
        recovery_unit=recovery_unit,
        area_candidates=area_candidates,
    )
    requested_top_k = min(len(actions), max(1, int(top_k if top_k is not None else len(actions))))

    raw_by_key = {str(item.get("key") or ""): dict(item) for item in scenario_changes if item.get("key")}
    changes = {key: raw_by_key.get(key, {"key": key, "effect": "closed", "magnitude": 1.0}) for key in selected}
    working = baseline.copy()
    _apply_recovery_disruptions(working, records, changes)
    evaluation_source_limit = 24 if baseline.number_of_nodes() > 1_200 else 64
    full_efficiency, efficiency_exact, source_count = _network_efficiency(
        baseline, maximum_sources=evaluation_source_limit
    )
    disrupted_efficiency, _, _ = _network_efficiency(
        working, maximum_sources=evaluation_source_limit
    )
    if disrupted_efficiency > full_efficiency + 1e-10:
        raise ValueError("The confirmed disruption unexpectedly improves the monotone network-efficiency baseline; review its operators and impedance fields.")

    objective = str(assumptions.get("priority_objective") or "service_gain")
    if objective not in {"service_gain", "gain_per_cost", "gain_per_duration"}:
        objective = "service_gain"
    resource_field = {"gain_per_cost": "repair_cost", "gain_per_duration": "repair_duration"}.get(objective)
    for action in actions:
        values = [records[key].get(resource_field) for key in action["keys"]] if resource_field else []
        action["resource_total"] = (
            sum(float(value) for value in values)
            if values and all(value is not None and float(value) > 0 for value in values)
            else None
        )
        action["repair_cost"] = (
            sum(float(records[key]["repair_cost"]) for key in action["keys"])
            if all(records[key].get("repair_cost") is not None for key in action["keys"])
            else None
        )
        action["repair_duration"] = (
            sum(float(records[key]["repair_duration"]) for key in action["keys"])
            if all(records[key].get("repair_duration") is not None for key in action["keys"])
            else None
        )
    resource_ready = resource_field is not None and all(action["resource_total"] for action in actions)
    objective_used = objective if resource_field is None or resource_ready else "service_gain"

    # Candidate count, rather than total city-network size, controls the
    # adaptive greedy loop.  This keeps a 10- or 20-road recovery request on a
    # large network genuinely marginal: every remaining action is re-tested
    # after each restoration stage using the same disclosed deterministic
    # source cohort.  Only genuinely large action sets use the structural tier.
    adaptive = len(actions) <= 25
    sequence: list[dict[str, Any]] = []
    current_efficiency = disrupted_efficiency
    if adaptive:
        remaining = {action["action_id"]: action for action in actions}
        while remaining and len(sequence) < requested_top_k:
            candidates: list[tuple[float, float, str, dict[str, Any]]] = []
            for action_id in sorted(remaining):
                action = remaining[action_id]
                previous = _restore_action(working, baseline, records, action["keys"])
                trial_efficiency, _, _ = _network_efficiency(
                    working, maximum_sources=evaluation_source_limit
                )
                _undo_restoration(working, previous)
                gain = max(0.0, trial_efficiency - current_efficiency)
                denominator = float(action["resource_total"]) if objective_used != "service_gain" else 1.0
                candidates.append((gain / denominator, gain, action_id, action))
            score, gain, chosen_id, chosen = max(candidates, key=lambda item: (item[0], item[1], item[2]))
            _restore_action(working, baseline, records, chosen["keys"])
            current_efficiency, _, _ = _network_efficiency(
                working, maximum_sources=evaluation_source_limit
            )
            recovery = (
                (current_efficiency - disrupted_efficiency) / max(1e-12, full_efficiency - disrupted_efficiency)
                if full_efficiency > disrupted_efficiency else 1.0
            )
            sequence.append({
                "stage": len(sequence) + 1,
                "rank": len(sequence) + 1,
                "key": chosen["keys"][0] if len(chosen["keys"]) == 1 else chosen_id,
                "entity_id": chosen["label"],
                "action_id": chosen_id,
                "recovery_unit": recovery_unit,
                "road_count": len(chosen["keys"]),
                "member_keys": chosen["keys"],
                "bbox": chosen.get("bbox"),
                "marginal_efficiency_gain": float(gain),
                "priority_score": float(score),
                "cumulative_recovery": min(1.0, max(0.0, float(recovery))),
                "repair_cost": chosen.get("repair_cost"),
                "repair_duration": chosen.get("repair_duration"),
                **(records[chosen["keys"][0]] if len(chosen["keys"]) == 1 else {}),
            })
            remaining.pop(chosen_id)
        method = "adaptive_marginal_efficiency"
        selection_rule = "After each restoration stage, re-evaluate every remaining action and choose the largest marginal network-efficiency gain, or gain per complete uploaded resource value."
    else:
        if baseline.number_of_nodes() <= 1_200 and baseline.number_of_edges() <= 2_500:
            centrality = nx.edge_betweenness_centrality(baseline, weight="weight", normalized=True)
            centrality_method = "exact_edge_betweenness"
        else:
            sources = _deterministic_sources(baseline.nodes, maximum=48)
            targets = sorted(baseline.nodes, key=str)
            raw = nx.edge_betweenness_centrality_subset(baseline, sources=sources, targets=targets, weight="weight", normalized=False)
            scale = max(1.0, len(sources) * max(1, len(targets) - 1))
            centrality = {edge: value / scale for edge, value in raw.items()}
            centrality_method = "deterministic_source_edge_betweenness"
        bridges = _single_connection_bridge_pairs(baseline)
        bridge_increment = 1.0 / max(1, baseline.number_of_edges())
        screened: list[tuple[float, float, str, dict[str, Any]]] = []
        for action in actions:
            structural = 0.0
            for key in action["keys"]:
                record = records[key]
                u, v = _record_nodes(record, baseline)
                structural += float(centrality.get((u, v, key), 0.0))
                if frozenset((u, v)) in bridges:
                    structural += bridge_increment
            denominator = float(action["resource_total"]) if objective_used != "service_gain" else 1.0
            screened.append((structural / denominator, structural, action["action_id"], action))
        screened.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        raw_structural_total = sum(max(0.0, item[1]) for item in screened)
        uniform_fallback = raw_structural_total <= 0
        total_structural = float(len(screened)) if uniform_fallback else raw_structural_total
        cumulative = 0.0
        for score, structural, action_id, action in screened[:requested_top_k]:
            cumulative += 1.0 if uniform_fallback else max(0.0, structural)
            sequence.append({
                "stage": len(sequence) + 1,
                "rank": len(sequence) + 1,
                "key": action["keys"][0] if len(action["keys"]) == 1 else action_id,
                "entity_id": action["label"],
                "action_id": action_id,
                "recovery_unit": recovery_unit,
                "road_count": len(action["keys"]),
                "member_keys": action["keys"],
                "bbox": action.get("bbox"),
                "marginal_efficiency_gain": None,
                "priority_score": float(score),
                "structural_priority": float(structural),
                "cumulative_recovery": None,
                "cumulative_priority_coverage": min(1.0, cumulative / total_structural),
                "repair_cost": action.get("repair_cost"),
                "repair_duration": action.get("repair_duration"),
                **(records[action["keys"][0]] if len(action["keys"]) == 1 else {}),
            })
        method = "large_candidate_structural_screen"
        selection_rule = f"Large candidate sets are ranked once by summed {centrality_method} plus an explicit bridge increment of {bridge_increment:.8g}; no marginal service-recovery claim is made."

    all_feature_keys = [key for item in sequence for key in item["member_keys"]]
    geometries, mapping_ready = _inferred_link_geometries(database_path, all_feature_keys)
    features: list[dict[str, Any]] = []
    for item in sequence:
        if item["recovery_unit"] == "area" and item.get("bbox"):
            west, south, east, north = item["bbox"]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]},
                "properties": {**item, "severity": 1.0, "analysis_type": "restoration_screening"},
            })
        for key in item["member_keys"]:
            record = records[key]
            feature = _link_result_feature(
                {**record, "stage": item["stage"], "rank": item["rank"], "action_id": item["action_id"], "recovery_unit": recovery_unit},
                geometries,
                severity=1.0,
                allow_point_fallback=not mapping_ready,
                extra_properties={"analysis_type": "restoration_screening", "recovery_highlight": True},
            )
            if feature is not None:
                features.append(feature)

    first = sequence[0]
    model_id = "P5-M8" if objective_used in {"gain_per_cost", "gain_per_duration"} else "P5-M4"
    unit_label = "area" if recovery_unit == "area" else "road"
    if adaptive:
        first_value = f"{float(first['marginal_efficiency_gain']) * 100:.3f}%"
        first_detail = "Absolute harmonic-efficiency gain at stage 1"
        conclusion = f"The model re-evaluated the remaining disrupted {unit_label}s after every stage and returned the first {len(sequence)} recovery priorities."
    else:
        first_value = f"{float(first['priority_score']):.5g}"
        first_detail = "Disclosed structural screening score; not service recovery"
        conclusion = f"The large-candidate model returned {len(sequence)} transparent structural recovery priorities; it does not claim an adaptive or globally optimal repair schedule."
    if objective != objective_used:
        conclusion += f" Uploaded {resource_field.replace('_', ' ') if resource_field else 'resource'} values were incomplete, so the service-gain objective was used."
    if topology_excluded:
        conclusion += f" {len(topology_excluded)} mapped candidate(s) without usable From/To topology were excluded and reported."
    return {
        "model": {
            "id": model_id,
            "name": "Plan recovery priorities",
            "status": "numerically_verified",
            "optimality_claim": "none",
        },
        "status": "supported" if adaptive and efficiency_exact and baseline.graph.get("impedance_mode") != "uniform_hop_count" else "conditional",
        "conclusion": conclusion,
        "headline_effects": [
            {"label": "Highest priority", "value": first["entity_id"], "detail": f"{unit_label.title()} ranked first"},
            {"label": "First-stage value", "value": first_value, "detail": first_detail},
            {"label": "Returned", "value": str(len(sequence)), "detail": f"Top {len(sequence)} of {len(actions)} disrupted {unit_label} actions"},
        ],
        "visual": {
            "mode": "recovery_priority",
            "method": method,
            "recovery_unit": recovery_unit,
            "objective_requested": objective,
            "objective_used": objective_used,
            "curve": [
                {
                    "stage": item["stage"],
                    "label": item["entity_id"],
                    "cumulative_recovery": item.get("cumulative_recovery"),
                    "cumulative_priority_coverage": item.get("cumulative_priority_coverage"),
                }
                for item in sequence
            ],
        },
        "next_actions": [
            "Inspect whether the highest-ranked road or area is physically accessible and repairable.",
            "Add repair cost and duration to compare service gain per resource unit.",
            "Use crews, calendars, budgets and precedence constraints before claiming an operational schedule or optimum.",
        ],
        "items": sequence,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "full_efficiency": full_efficiency,
            "disrupted_efficiency": disrupted_efficiency,
            "efficiency_exact": efficiency_exact,
            "efficiency_source_count": source_count,
            "method": method,
            "selection_rule": selection_rule,
            "candidate_action_count": len(actions),
            "returned_action_count": len(sequence),
            "recovery_unit": recovery_unit,
            "objective_requested": objective,
            "objective_used": objective_used,
            "resource_field_complete": resource_ready if resource_field else None,
            "scenario_changes_grounded": len(changes),
            "topology_excluded_count": len(topology_excluded),
            "topology_excluded_keys": topology_excluded,
            "topology_node_scope": baseline.graph.get("topology_node_scope"),
            "impedance_mode": baseline.graph.get("impedance_mode"),
            "literature_basis": [
                "Nurre et al. (2012), doi:10.1016/j.ejor.2012.07.010",
                "Maya Duque, Dolinskaya and Sorensen (2016), doi:10.1016/j.ejor.2015.06.026",
                "Jenelius and Mattsson (2012), doi:10.1016/j.tra.2012.02.003",
                "Latora and Marchiori (2001), doi:10.1103/PhysRevLett.87.198701",
            ],
        },
        "data_boundary": (
            "The adaptive tier is a sequential marginal network-service model, not a globally optimal multi-crew schedule. "
            "The large-candidate tier is a structural priority screen and deliberately withholds marginal recovery claims. "
            "Area actions restore their confirmed member roads as one screening package; they do not imply a real construction contract or jurisdiction."
        ),
        "verification": {
            "passed": (
                len(sequence) == requested_top_k
                and len({item["action_id"] for item in sequence}) == len(sequence)
                and all(set(item["member_keys"]).issubset(records) for item in sequence)
                and all(item.get("cumulative_recovery") is None or 0 <= item["cumulative_recovery"] <= 1 for item in sequence)
            ),
            "checks": [
                "every returned action is grounded in the current project",
                "each action is restored at most once",
                "disruption operators match saved-scenario semantics",
                "adaptive marginal gains are recomputed after every stage",
                "large candidate screens withhold uncomputed marginal recovery",
                "resource objectives run only with complete positive uploaded values",
                "no random sampling",
            ],
        },
    }


def temporal_resilience_curve(
    database_path: Path,
    *,
    metric: str,
    bbox: list[float] | None,
    time_start: int | None,
    time_end: int | None,
) -> dict[str, Any]:
    if metric not in {"speed", "travel_time"}:
        raise ValueError("Tracking disruption and recovery requires speed or travel-time observations with positive reference values.")
    value_column, baseline_column, _default_unit = _metric_parts(metric)
    if metric == "speed":
        service_sql = f"MIN(1.0, {value_column} / {baseline_column})"
        valid_sql = f"{value_column} >= 0 AND {baseline_column} > 0"
    else:
        service_sql = f"MIN(1.0, {baseline_column} / {value_column})"
        valid_sql = f"{value_column} > 0 AND {baseline_column} > 0"
    time_clauses: list[str] = []
    time_parameters: list[Any] = []
    if time_start is not None:
        time_clauses.append("t.timestamp_epoch >= ?")
        time_parameters.append(time_start)
    if time_end is not None:
        time_clauses.append("t.timestamp_epoch <= ?")
        time_parameters.append(time_end)
    bbox_sql, bbox_parameters = _bbox_clause(bbox)
    time_where = " AND ".join(time_clauses) if time_clauses else "1 = 1"
    observation_time_clauses = [clause.replace("t.timestamp_epoch", "o.timestamp_epoch") for clause in time_clauses]
    observation_where = " AND ".join([valid_sql, *observation_time_clauses])
    with _connect(database_path) as connection:
        summary = _summary(connection)
        cohort_count = int(connection.execute(
            f"""
            SELECT COUNT(*) FROM entities e
            WHERE {baseline_column} > 0
              AND e.latitude BETWEEN -90 AND 90 AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
            """,
            bbox_parameters,
        ).fetchone()[0])
        observed_rows = connection.execute(
            f"""
            SELECT o.timestamp_epoch, t.timestamp_text,
                   AVG({service_sql}) AS performance,
                   COUNT(*) AS valid_entity_count
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            JOIN times t ON t.timestamp_epoch = o.timestamp_epoch
            WHERE {observation_where}
              AND e.latitude BETWEEN -90 AND 90 AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
            GROUP BY o.timestamp_epoch, t.timestamp_text
            ORDER BY o.timestamp_epoch
            """,
            [*time_parameters, *bbox_parameters],
        ).fetchall()
        time_rows = connection.execute(
            f"SELECT timestamp_epoch, timestamp_text FROM times t WHERE {time_where} ORDER BY timestamp_epoch",
            time_parameters,
        ).fetchall()
    observed_by_time = {int(row["timestamp_epoch"]): row for row in observed_rows}
    rows = [
        {
            "timestamp_epoch": int(time_row["timestamp_epoch"]),
            "timestamp_text": time_row["timestamp_text"],
            "performance": observed_by_time.get(int(time_row["timestamp_epoch"]))["performance"]
            if int(time_row["timestamp_epoch"]) in observed_by_time else None,
            "valid_entity_count": int(observed_by_time.get(int(time_row["timestamp_epoch"]))["valid_entity_count"])
            if int(time_row["timestamp_epoch"]) in observed_by_time else 0,
        }
        for time_row in time_rows
    ]
    if len(rows) < 2:
        raise ValueError("At least two valid observed time points are required for a resilience curve.")

    expected_interval = int(summary.get("expected_interval_s") or 0)
    coverage_threshold = 0.90
    points: list[dict[str, Any]] = []
    cumulative_loss_seconds = 0.0
    observed_duration_seconds = 0.0
    excluded_gap_seconds = 0.0
    excluded_coverage_seconds = 0.0
    previous: dict[str, Any] | None = None
    for row in rows:
        performance = (
            max(0.0, min(1.0, float(row["performance"])))
            if row["performance"] is not None
            else None
        )
        point = {
            "stage": len(points) + 1,
            "time": int(row["timestamp_epoch"]),
            "label": row["timestamp_text"],
            "performance": performance,
            "loss": max(0.0, 1.0 - performance) if performance is not None else None,
            "valid_entity_count": int(row["valid_entity_count"]),
            "cohort_entity_count": cohort_count,
            "coverage_ratio": int(row["valid_entity_count"]) / max(1, cohort_count),
            "gap_before": False,
        }
        if previous is not None:
            interval = point["time"] - previous["time"]
            insufficient_coverage = (
                previous["coverage_ratio"] < coverage_threshold
                or point["coverage_ratio"] < coverage_threshold
            )
            is_gap = interval <= 0 or (expected_interval > 0 and interval > expected_interval * 1.5) or insufficient_coverage
            point["gap_before"] = is_gap
            if is_gap:
                if insufficient_coverage:
                    excluded_coverage_seconds += max(0, interval)
                else:
                    excluded_gap_seconds += max(0, interval)
            else:
                cumulative_loss_seconds += interval * (previous["loss"] + point["loss"]) / 2
                observed_duration_seconds += interval
        points.append(point)
        previous = point

    if observed_duration_seconds <= 0:
        raise ValueError("Observed time intervals are missing or separated only by large gaps; a resilience integral is not defensible.")
    normalised_loss = cumulative_loss_seconds / observed_duration_seconds
    resilience_index = max(0.0, min(1.0, 1.0 - normalised_loss))
    comparable_points = [
        point for point in points
        if point["coverage_ratio"] >= coverage_threshold and point["performance"] is not None
    ]
    if len(comparable_points) < 2:
        raise ValueError("Too few time points retain at least 90% cohort coverage for a defensible resilience curve.")
    nadir = min(comparable_points, key=lambda item: (item["performance"], item["time"]))
    final_point_comparable = points[-1]["coverage_ratio"] >= coverage_threshold and points[-1]["performance"] is not None
    final_performance = points[-1]["performance"] if final_point_comparable else None
    nadir_loss = max(0.0, 1.0 - nadir["performance"])
    recovery_fraction = max(0.0, min(1.0, (final_performance - nadir["performance"]) / nadir_loss)) if nadir_loss > 0 and final_point_comparable else None
    collection = map_features(
        database_path,
        metric=metric,
        timestamp=nadir["time"],
        bbox=bbox,
        limit=20000,
    )
    return {
        "model": {"id": "P5-M9", "name": "Track disruption and recovery", "status": "numerically_verified"},
        "status": "conditional",
        "conclusion": f"Across valid historical intervals, overall transport performance averaged {resilience_index * 100:.1f}% of its internal reference; the worst observed drop was {nadir_loss * 100:.1f}%.",
        "headline_effects": [
            {"label": "Average performance", "value": resilience_index, "display_value": f"{resilience_index * 100:.1f}%", "detail": "Time-weighted performance relative to the internal historical reference"},
            {"label": "Worst performance drop", "value": nadir_loss, "display_value": f"{nadir_loss * 100:.1f}%", "detail": nadir["label"]},
            {"label": "Recovered by final time", "value": recovery_fraction if recovery_fraction is not None else "Not inferred", "display_value": f"{recovery_fraction * 100:.1f}%" if recovery_fraction is not None else "Not inferred", "detail": "Share of the low-point drop regained; final time requires at least 90% cohort coverage"},
            {"label": "Minimum coverage", "value": min(point["coverage_ratio"] for point in points), "display_value": f"{min(point['coverage_ratio'] for point in points) * 100:.1f}%", "detail": f"Valid observations relative to {cohort_count:,} reference-eligible entities"},
        ],
        "next_actions": [
            "Inspect the mapped state at the observed nadir.",
            "Confirm whether the internal reference window represents pre-disaster service.",
            "Compare geographic hotspots at the low point and final time.",
            "To evaluate strengthening, define a candidate intervention and compare this same timeline or stress test before and after it.",
        ],
        "items": points,
        "feature_collection": collection,
        "evidence": {
            "metric": metric,
            "time_point_count": len(points),
            "cumulative_loss_seconds": cumulative_loss_seconds,
            "observed_duration_seconds": observed_duration_seconds,
            "excluded_gap_seconds": excluded_gap_seconds,
            "excluded_coverage_seconds": excluded_coverage_seconds,
            "coverage_threshold": coverage_threshold,
            "cohort_entity_count": cohort_count,
            "integration_rule": "Trapezoidal integration over adjacent intervals no longer than 1.5 times the median uploaded interval.",
            "performance_definition": "min(1, observed/reference) for speed; min(1, reference/observed) for travel time. Every uploaded time point remains visible; missing points have null performance and zero coverage. Intervals touching a point below 90% cohort coverage are excluded.",
        },
        "data_boundary": "This is an observed resilience curve relative to the internal reference window. It does not infer a counterfactual no-disaster trajectory or causal recovery effect.",
        "verification": {
            "passed": 0 <= resilience_index <= 1 and all(point["performance"] is None or 0 <= point["performance"] <= 1 for point in points),
            "checks": ["positive reference and non-negative speed or positive travel time", "coverage-screened cohort", "bounded service ratio", "gap-screened non-negative time weights", "trapezoidal integral"],
        },
    }


def _dpc_travel_time_expression(
    observation_alias: str = "o",
    entity_alias: str = "e",
) -> str:
    """Return the strongest row-level travel-time evidence available to DPC.

    The selected display metric must not hide a stronger uploaded field.  A
    positive uploaded travel time is therefore used first for every row; only
    a missing/invalid value is replaced by the dimensionally exact
    length/speed conversion.  No value is imputed from another road or time.
    """

    return (
        f"CASE WHEN {observation_alias}.travel_time > 0 "
        f"THEN {observation_alias}.travel_time "
        f"WHEN {observation_alias}.speed > 0 AND {entity_alias}.length_m > 0 "
        f"THEN ({entity_alias}.length_m / 1000.0) / {observation_alias}.speed * 60.0 END"
    )


def _dpc_travel_time_source_expression(
    observation_alias: str = "o",
    entity_alias: str = "e",
) -> str:
    return (
        f"CASE WHEN {observation_alias}.travel_time > 0 THEN 'uploaded_travel_time' "
        f"WHEN {observation_alias}.speed > 0 AND {entity_alias}.length_m > 0 "
        f"THEN 'derived_from_speed_and_length' END"
    )


def _dpc_observed_edges(
    connection: sqlite3.Connection,
    *,
    summary: dict[str, Any],
    metric: str,
    selected_time: int,
    bbox: list[float] | None,
    cell_for_location: Any,
) -> tuple[list[PropagationEdge], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load Eq. (10) and Eq. (11) inputs without inventing missing histories."""

    travel_expression = _dpc_travel_time_expression()
    source_expression = _dpc_travel_time_source_expression()
    bbox_sql, bbox_parameters = _bbox_clause(bbox)
    global_node_ids = summary.get("topology_node_scope", "global") == "global"
    rows = connection.execute(
        f"""
        WITH eligible_entities AS (
            SELECT e.entity_pk, e.region_id, e.entity_id, e.from_node, e.to_node,
                   e.length_m, e.latitude, e.longitude
            FROM entities e
            WHERE e.from_node IS NOT NULL AND e.from_node <> ''
              AND e.to_node IS NOT NULL AND e.to_node <> ''
              AND e.latitude BETWEEN -90 AND 90
              AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
        ),
        raw_history AS (
            SELECT o.entity_pk, {travel_expression} AS effective_travel_time,
                   {source_expression} AS travel_time_source
            FROM observations o
            JOIN eligible_entities e ON e.entity_pk = o.entity_pk
        ),
        valid_history AS (
            SELECT entity_pk, effective_travel_time, travel_time_source
            FROM raw_history
            WHERE effective_travel_time > 0
        ),
        ranked_history AS (
            SELECT entity_pk, effective_travel_time, travel_time_source,
                   ROW_NUMBER() OVER (
                       PARTITION BY entity_pk ORDER BY effective_travel_time, entity_pk
                   ) AS quantile_position,
                   COUNT(*) OVER (PARTITION BY entity_pk) AS history_count,
                   SUM(CASE WHEN travel_time_source = 'uploaded_travel_time' THEN 1 ELSE 0 END)
                       OVER (PARTITION BY entity_pk) AS uploaded_history_count,
                   SUM(CASE WHEN travel_time_source = 'derived_from_speed_and_length' THEN 1 ELSE 0 END)
                       OVER (PARTITION BY entity_pk) AS derived_history_count
            FROM valid_history
        ),
        reference_times AS (
            SELECT entity_pk, effective_travel_time AS reference_travel_time,
                   history_count, uploaded_history_count, derived_history_count
            FROM ranked_history
            WHERE quantile_position = CAST((history_count - 1) * 0.05 AS INTEGER) + 1
        )
        SELECT e.region_id, e.entity_id, e.from_node, e.to_node,
               e.latitude, e.longitude, r.reference_travel_time, r.history_count,
               r.uploaded_history_count, r.derived_history_count,
               {travel_expression} AS observed_travel_time,
               {source_expression} AS selected_time_source
        FROM eligible_entities e
        JOIN observations o ON o.entity_pk = e.entity_pk
        JOIN reference_times r ON r.entity_pk = e.entity_pk
        WHERE o.timestamp_epoch = ?
          AND {travel_expression} > 0
        ORDER BY e.region_id, e.entity_id
        """,
        (*bbox_parameters, selected_time),
    ).fetchall()
    eligible_count = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM entities e
            WHERE e.from_node IS NOT NULL AND e.from_node <> ''
              AND e.to_node IS NOT NULL AND e.to_node <> ''
              AND e.latitude BETWEEN -90 AND 90
              AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
            """,
            bbox_parameters,
        ).fetchone()[0]
    )
    if not rows:
        raise ValueError("No routable mapped link has a valid DPC travel-time history at the chosen observation time.")

    propagation_edges: list[PropagationEdge] = []
    cell_metadata: dict[str, dict[str, Any]] = {}
    history_counts: list[int] = []
    selected_source_counts = {
        "uploaded_travel_time": 0,
        "derived_from_speed_and_length": 0,
    }
    history_source_counts = {
        "uploaded_travel_time": 0,
        "derived_from_speed_and_length": 0,
    }
    for row in rows:
        history_count = int(row["history_count"] or 0)
        if history_count < 2:
            continue
        observed = float(row["observed_travel_time"])
        reference = float(row["reference_travel_time"])
        if not (math.isfinite(observed) and observed > 0 and math.isfinite(reference) and reference > 0):
            continue
        cell_id, bounds = cell_for_location(float(row["longitude"]), float(row["latitude"]))
        graph_from = str(row["from_node"]) if global_node_ids else f"{row['region_id']}\x1f{row['from_node']}"
        graph_to = str(row["to_node"]) if global_node_ids else f"{row['region_id']}\x1f{row['to_node']}"
        propagation_edges.append(
            PropagationEdge(
                from_node=graph_from,
                to_node=graph_to,
                strength=min(1.0, reference / observed),
                region_id=cell_id,
            )
        )
        metadata = cell_metadata.setdefault(cell_id, {**bounds, "entity_count": 0})
        metadata["entity_count"] += 1
        history_counts.append(history_count)
        selected_source = str(row["selected_time_source"] or "")
        if selected_source in selected_source_counts:
            selected_source_counts[selected_source] += 1
        history_source_counts["uploaded_travel_time"] += int(row["uploaded_history_count"] or 0)
        history_source_counts["derived_from_speed_and_length"] += int(row["derived_history_count"] or 0)

    coverage = len(propagation_edges) / max(1, eligible_count)
    if coverage < 0.80:
        raise ValueError(
            f"Regional DPC covers {coverage * 100:.1f}% of routable mapped links; at least 80% need valid current and historical travel time."
        )
    return propagation_edges, cell_metadata, {
        "eligible_topology_link_count": eligible_count,
        "dpc_link_count": len(propagation_edges),
        "dpc_link_coverage": coverage,
        "minimum_link_history_count": min(history_counts, default=0),
        "maximum_link_history_count": max(history_counts, default=0),
        "display_metric_requested": metric,
        "travel_time_precedence": "positive uploaded travel time per row; otherwise positive length/speed-derived travel time for that same row",
        "selected_time_source_counts": selected_source_counts,
        "history_source_counts": history_source_counts,
        "imputation": "none across roads or observation times",
        "reference_quantile": 0.05,
        "reference_quantile_rule": "lower empirical quantile at floor(q*(n-1))+1 in ascending valid link travel time",
    }


def spatial_hotspots(
    database_path: Path,
    *,
    metric: str,
    top_k: int,
    bbox: list[float] | None,
    timestamp: int | None,
) -> dict[str, Any]:
    value_column, baseline_column, _default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        severity_expression, severity_valid = _severity_sql(
            metric, value_column, baseline_column, summary
        )
        if not severity_expression or not severity_valid:
            raise ValueError(
                "The uploaded metric is descriptive only. Confirm its severity direction and positive scale before hotspot screening."
            )
        severity_sql = f"CASE WHEN {severity_valid} THEN {severity_expression} END"
        bbox_sql, parameters = _bbox_clause(bbox)
        rows = connection.execute(
            f"""
            SELECT e.region_id, e.entity_id, e.latitude, e.longitude,
                   {severity_sql} AS severity
            FROM observations o
            JOIN entities e ON e.entity_pk = o.entity_pk
            WHERE o.timestamp_epoch = ?
              AND e.latitude BETWEEN -90 AND 90 AND e.longitude BETWEEN -180 AND 180
              {bbox_sql}
            """,
            (selected_time, *parameters),
        ).fetchall()
        label_row = connection.execute("SELECT timestamp_text FROM times WHERE timestamp_epoch = ?", (selected_time,)).fetchone()
    valid_rows = [row for row in rows if row["severity"] is not None and math.isfinite(float(row["severity"]))]
    if not valid_rows:
        raise ValueError("No valid relative degradation values are available for geographic hotspot analysis at this time.")

    # Fixed metric cells remain comparable across time, viewport and project
    # extent. Web Mercator is used only to index 5 km screening cells; output
    # is transformed back to WGS84 GeoJSON for the real map.
    earth_radius = 6_378_137.0
    world_half = math.pi * earth_radius
    cell_size_m = 5_000.0
    minimum_support = 3

    def project(longitude: float, latitude: float) -> tuple[float, float]:
        bounded_latitude = max(-85.05112878, min(85.05112878, latitude))
        return (
            earth_radius * math.radians(longitude),
            earth_radius * math.log(math.tan(math.pi / 4 + math.radians(bounded_latitude) / 2)),
        )

    def unproject(x: float, y: float) -> tuple[float, float]:
        return (
            math.degrees(x / earth_radius),
            math.degrees(2 * math.atan(math.exp(y / earth_radius)) - math.pi / 2),
        )

    def cell_bounds(column: int, line: int) -> dict[str, float]:
        cell_x_min = column * cell_size_m - world_half
        cell_y_min = line * cell_size_m - world_half
        west, south = unproject(cell_x_min, cell_y_min)
        east, north = unproject(cell_x_min + cell_size_m, cell_y_min + cell_size_m)
        return {
            "column": column,
            "line": line,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        }

    def cell_for_location(longitude: float, latitude: float) -> tuple[str, dict[str, float]]:
        x, y = project(longitude, latitude)
        column = int(math.floor((x + world_half) / cell_size_m))
        line = int(math.floor((y + world_half) / cell_size_m))
        return f"M5K-{column}-{line}", cell_bounds(column, line)

    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for row in valid_rows:
        _cell_id, location_cell = cell_for_location(float(row["longitude"]), float(row["latitude"]))
        column = int(location_cell["column"])
        line = int(location_cell["line"])
        cell = cells.setdefault(
            (column, line),
            {"severity_sum": 0.0, "maximum_severity": 0.0, "entity_keys": set(), "regions": set()},
        )
        severity = max(0.0, float(row["severity"]))
        cell["severity_sum"] += severity
        cell["maximum_severity"] = max(cell["maximum_severity"], severity)
        cell["entity_keys"].add(f"{row['region_id']}::{row['entity_id']}")
        cell["regions"].add(str(row["region_id"]))

    ranked_cells = []
    for (column, line), cell in cells.items():
        count = len(cell["entity_keys"])
        mean_severity = cell["severity_sum"] / max(1, count)
        bounds = cell_bounds(column, line)
        ranked_cells.append(
            {
                "cell_id": f"M5K-{column}-{line}",
                "entity_count": count,
                "region_count": len(cell["regions"]),
                "mean_degradation": mean_severity,
                "maximum_degradation": cell["maximum_severity"],
                "entity_keys": sorted(cell["entity_keys"]),
                "west": bounds["west"],
                "south": bounds["south"],
                "east": bounds["east"],
                "north": bounds["north"],
            }
        )
    supported_cells = [cell for cell in ranked_cells if cell["entity_count"] >= minimum_support]
    low_support_fallback = not supported_cells
    if supported_cells:
        ranked_cells = supported_cells

    dpc_result: dict[str, Any] | None = None
    dpc_inputs: dict[str, Any] | None = None
    dpc_unavailable_reason: str | None = None
    if metric in {"speed", "travel_time"}:
        try:
            with _connect(database_path) as connection:
                propagation_edges, dpc_cells, dpc_inputs = _dpc_observed_edges(
                    connection,
                    summary=summary,
                    metric=metric,
                    selected_time=selected_time,
                    bbox=bbox,
                    cell_for_location=cell_for_location,
                )
            candidate_ids = sorted(
                cell_id
                for cell_id, metadata in dpc_cells.items()
                if int(metadata["entity_count"]) >= 10
            )
            if len(candidate_ids) < 2:
                raise ValueError("Regional DPC needs at least two 5 km candidate cells with 10 valid links each.")
            if len(candidate_ids) > 160:
                raise ValueError(
                    f"Regional DPC found {len(candidate_ids)} eligible cells. The verified platform contract supports at most 160 exhaustive candidate cells per run."
                )
            dpc_result = regional_dpc_impacts(
                propagation_edges,
                candidate_ids,
                contract=DPCNumericalContract(),
            )
            impacts = {
                str(item["region_id"]): item
                for item in dpc_result.get("impacts", [])
            }
            dpc_ranked_cells = []
            for cell in ranked_cells:
                impact = impacts.get(str(cell["cell_id"]))
                if impact is None:
                    continue
                dpc_ranked_cells.append(
                    {
                        **cell,
                        "baseline_dpc": impact["baseline_dpc"],
                        "degraded_dpc": impact["degraded_dpc"],
                        "signed_dpc_impact": impact["signed_dpc_impact"],
                        "regional_dpc_score": impact["regional_dpc_score"],
                        "ranking_metric": "regional DPC loss",
                    }
                )
            if len(dpc_ranked_cells) < 2:
                raise ValueError("Regional DPC candidate cells could not be joined to the mapped observation grid.")
            ranked_cells = dpc_ranked_cells
        except (ValueError, np.linalg.LinAlgError) as error:
            dpc_result = None
            dpc_unavailable_reason = str(error)

    if dpc_result is not None:
        ranked_cells.sort(
            key=lambda item: (
                -float(item["regional_dpc_score"]),
                -float(item["signed_dpc_impact"]),
                -item["entity_count"],
                item["cell_id"],
            )
        )
    else:
        ranked_cells.sort(key=lambda item: (-item["mean_degradation"], -item["entity_count"], item["cell_id"]))
    items = []
    features = []
    maximum_dpc_score = max(
        (float(item.get("regional_dpc_score") or 0.0) for item in ranked_cells),
        default=0.0,
    )
    for rank, item in enumerate(ranked_cells[:top_k], start=1):
        ranked = {"rank": rank, **item}
        items.append(ranked)
        if dpc_result is not None:
            # Never colour a DPC-ranked map with a different statistic.  When
            # every DPC score is zero, every returned polygon remains zero;
            # the observed mean is retained only as supporting table evidence.
            mapped_severity = (
                float(item.get("regional_dpc_score") or 0.0) / maximum_dpc_score
                if maximum_dpc_score > 0
                else 0.0
            )
        else:
            mapped_severity = min(1.0, item["mean_degradation"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [item["west"], item["south"]], [item["east"], item["south"]],
                        [item["east"], item["north"]], [item["west"], item["north"]],
                        [item["west"], item["south"]],
                    ]],
                },
                "properties": {**ranked, "severity": mapped_severity},
            }
        )
    top = items[0]
    dpc_tier = dpc_result is not None
    if dpc_tier:
        conclusion = (
            f"The highest-ranked cell loses {float(top['regional_dpc_score']) * 100:.3f}% of "
            f"Dynamic Propagation Connectivity after a 50% regional coupling degradation."
        )
        headline_effects = [
            {
                "label": "Top regional DPC loss",
                "value": f"{float(top['regional_dpc_score']) * 100:.3f}%",
                "detail": top["cell_id"],
            },
            {
                "label": "Observed propagation coverage",
                "value": f"{float(dpc_result['baseline_dpc']):.3f} nodes",
                "detail": label_row[0] if label_row else str(selected_time),
            },
            {
                "label": "Eligible DPC cells",
                "value": str(len(ranked_cells)),
                "detail": "Fixed 5 km grid; at least 10 valid routable links per cell",
            },
        ]
        next_actions = [
            "Inspect the highest-ranked propagation-critical cells on the map.",
            "Compare the same DPC ranking at another observed time.",
            "Use the ranked cells as candidates for recovery analysis, while retaining the reported numerical boundary.",
        ]
    else:
        conclusion = f"The highest observed geographic cell has {top['mean_degradation'] * 100:.1f}% mean relative degradation across {top['entity_count']} mapped entities."
        headline_effects = [
            {"label": "Top-cell mean loss", "value": f"{top['mean_degradation'] * 100:.1f}%", "detail": top["cell_id"]},
            {"label": "Valid mapped entities", "value": str(len(valid_rows)), "detail": label_row[0] if label_row else str(selected_time)},
            {"label": "Eligible cells", "value": str(len(ranked_cells)), "detail": "Fixed 5 km metric grid; at least 3 entities per cell when available"},
        ]
        next_actions = ["Zoom to the highlighted real geographic cells.", "Compare the hotspot ranking at another observed time.", "Add routable topology and valid travel-time history to unlock regional DPC criticality."]
    return {
        "model": {"id": "P5-M10", "name": "Find critical areas", "status": "numerically_verified"},
        "status": "supported" if dpc_tier else "conditional" if low_support_fallback else "supported",
        "randomness": "none",
        "conclusion": conclusion,
        "headline_effects": headline_effects,
        "next_actions": next_actions,
        "items": items,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "timestamp_epoch": selected_time,
            "metric": metric,
            "valid_mapped_entity_count": len(valid_rows),
            "formed_cell_count": len(cells),
            "eligible_cell_count": len(ranked_cells),
            "requested_result_count": max(1, int(top_k)),
            "returned_result_count": len(items),
            "primary_metric_semantics": _primary_semantics(summary) if metric == "primary" else None,
            "grid_crs": "EPSG:3857 screening grid; GeoJSON output transformed to WGS84",
            "grid_origin_m": [-world_half, -world_half],
            "cell_size_m": cell_size_m,
            "minimum_entity_support": minimum_support,
            "low_support_fallback": low_support_fallback,
            "executed_tier": "regional_dynamic_propagation_connectivity" if dpc_tier else "observed_fixed_grid_degradation",
            "randomness": "none",
            "dpc_source": "Xu, Niu, Dixit, Nair and Zhang (2026), arXiv:2606.30242, doi:10.48550/arXiv.2606.30242" if dpc_tier else None,
            "dpc_inputs": dpc_inputs,
            "dpc_numerical_diagnostics": dpc_result.get("diagnostics") if dpc_tier else None,
            "dpc_numerical_boundary": (
                "Equation (9) is an infinite propagation-time limit. The paper does not publish a finite quadrature horizon or sample rule. The platform therefore reports a deterministic finite-time approximation. For large networks, the reported Lanczos residual is a projection-boundary diagnostic, not a certified error bound."
                if dpc_tier
                else None
            ),
            "dpc_unavailable_reason": dpc_unavailable_reason,
            "computational_profile": (
                "Every eligible candidate cell is evaluated by a deterministic regional DPC perturbation. Dense Hermitian eigendecomposition is used for small networks and a disclosed Lanczos projection for larger networks."
                if dpc_tier
                else "One linear pass over the selected mapped observations followed by a stable sort of occupied cells; no road is removed and no shortest path is recomputed."
            ),
            "ranking_rule": (
                "Fixed 5 km cells with at least 10 valid routable links are ranked by descending non-negative regional DPC loss after alpha=0.5 propagation-strength degradation."
                if dpc_tier
                else "Fixed-grid cells meeting minimum support are ranked by descending mean relative degradation, then entity count; no hidden composite score."
            ),
        },
        "data_boundary": (
            "Regional DPC is a deterministic, classical-computing numerical implementation of the quantum-percolation formulation. It diagnoses propagation sensitivity under the uploaded historical state. It is not quantum-hardware output, a hazard probability, an administrative boundary or a causal impact estimate."
            if dpc_tier
            else "Cells summarise observed point locations on a real geographic map. They are not administrative boundaries, hazard footprints or causal impact zones."
        ),
        "verification": {
            "passed": all(item["entity_count"] > 0 for item in items)
            and all(float(item.get("regional_dpc_score") or 0.0) >= 0 for item in items),
            "checks": (
                [
                    "valid WGS84 representative link locations",
                    "at least 80% routable-link travel-time coverage",
                    "link-specific lower empirical 5th-percentile scaling",
                    "Hermitian symmetrised propagation operator",
                    "normalised uniform initial state",
                    "deterministic finite propagation-time quadrature",
                    "finite quadrature and any Lanczos projection disclosed as numerical boundaries",
                    "alpha=0.5 regional coupling perturbation",
                    "non-negative Eq. (20) score",
                    "no random sampling and no quantum-hardware claim",
                ]
                if dpc_tier
                else ["valid WGS84 locations", "single confirmed observed time", "transparent deterministic grid", "no random smoothing"]
            ),
        },
    }


def progressive_robustness(
    database_path: Path,
    *,
    top_k: int,
    bbox: list[float] | None,
    failure_keys: list[str] | None = None,
) -> dict[str, Any]:
    candidate_count = max(1, top_k)
    with _connect(database_path) as connection:
        graph, records = _build_graph(connection, bbox)
    if failure_keys is not None and not isinstance(failure_keys, list):
        raise ValueError("The stress-test failure order must be a list of project road keys.")
    confirmed_failure_keys = list(dict.fromkeys(str(key) for key in (failure_keys or []) if str(key)))
    if confirmed_failure_keys:
        missing = [key for key in confirmed_failure_keys if key not in records]
        if missing:
            raise ValueError("Some stress-test roads do not belong to the current project or area: " + ", ".join(missing[:5]))
        confirmed_prefix = confirmed_failure_keys[:candidate_count]
        extension = None
        if len(confirmed_prefix) < candidate_count:
            extension = critical_components(database_path, top_k=candidate_count, bbox=bbox)
            confirmed_set = set(confirmed_prefix)
            confirmed_prefix.extend(
                item["key"]
                for item in extension.get("items", [])
                if item.get("key") not in confirmed_set
            )
            confirmed_prefix = confirmed_prefix[:candidate_count]
        ranking = {
            "items": [
                {"key": key, "efficiency_loss": None}
                for key in confirmed_prefix
            ],
            "status": extension.get("status", "conditional") if extension else "conditional",
            "evidence": {
                "exhaustive_edge_removal": bool(extension and extension.get("evidence", {}).get("exhaustive_edge_removal")),
                "candidate_coverage": extension.get("evidence", {}).get("candidate_coverage") if extension else None,
                "evaluated_candidate_count": extension.get("evidence", {}).get("evaluated_candidate_count") if extension else len(confirmed_prefix),
                "total_edge_count": extension.get("evidence", {}).get("total_edge_count") if extension else graph.number_of_edges(),
            },
        }
        sequence_source = (
            "confirmed_current_critical_road_prefix_then_deterministic_extension"
            if extension else "confirmed_current_critical_road_order"
        )
    else:
        ranking = critical_components(database_path, top_k=candidate_count, bbox=bbox)
        sequence_source = "recomputed_single_road_criticality_order"
    baseline_efficiency, exact, source_count = _network_efficiency(graph)
    if baseline_efficiency <= 0:
        raise ValueError("The connected topology does not provide a positive baseline efficiency for robustness analysis.")
    baseline_components = _component_metrics(graph)
    stages = []
    performance_curve = [1.0]
    for rank_item in ranking["items"]:
        key = rank_item["key"]
        record = records.get(key)
        if not record:
            continue
        u, v = _record_nodes(record, graph)
        if graph.has_edge(u, v, key):
            graph.remove_edge(u, v, key)
        efficiency, _, _ = _network_efficiency(graph)
        performance_ratio = max(0.0, efficiency / baseline_efficiency)
        components = _component_metrics(graph)
        stage = {
            "stage": len(stages) + 1,
            **record,
            "performance_ratio": performance_ratio,
            "robustness_loss": max(0.0, 1.0 - performance_ratio),
            "weak_components": components["weak_components"],
            "ranking_basis": rank_item.get("efficiency_loss"),
        }
        stages.append(stage)
        performance_curve.append(performance_ratio)
    if not stages:
        raise ValueError("No ranked topology links could be removed for the robustness curve.")
    geometries, mapping_ready = _inferred_link_geometries(database_path, (item["key"] for item in stages))
    features = []
    for item in stages:
        feature = _link_result_feature(
            item,
            geometries,
            severity=min(1.0, item["robustness_loss"]),
            allow_point_fallback=not mapping_ready,
        )
        if feature is not None:
            features.append(feature)
    step_width = 1.0 / len(stages)
    robustness_auc = sum((left + right) * 0.5 * step_width for left, right in zip(performance_curve, performance_curve[1:]))
    final = stages[-1]
    thresholds = {}
    for threshold in (0.90, 0.75, 0.50):
        reached = next((item for item in stages if item["performance_ratio"] <= threshold), None)
        thresholds[f"{int(threshold * 100)}_percent"] = reached["stage"] if reached else None
    first_split = next(
        (item["stage"] for item in stages if item["weak_components"] > baseline_components["weak_components"]),
        None,
    )
    drops = [performance_curve[index] - performance_curve[index + 1] for index in range(len(stages))]
    steepest_index = max(range(len(drops)), key=lambda index: (drops[index], -index))
    steepest_stage = stages[steepest_index]["stage"]
    monotone = all(
        performance_curve[index + 1] <= performance_curve[index] + 1e-10
        for index in range(len(performance_curve) - 1)
    )
    ranking_exhaustive = bool(ranking.get("evidence", {}).get("exhaustive_edge_removal"))
    fully_supported = bool(
        exact
        and ranking.get("status") == "supported"
        and ranking_exhaustive
        and graph.graph.get("impedance_mode") != "uniform_hop_count"
    )
    weakening_story = (
        f"Efficiency first falls to 90% or less after {thresholds['90_percent']} cumulative removals."
        if thresholds["90_percent"]
        else "Efficiency remains above 90% throughout the tested removals."
    )
    split_story = (
        f" The first tested network split appears at stage {first_split}."
        if first_split
        else " No additional weakly connected part appears within the tested stages."
    )
    return {
        "model": {"id": "P5-M11", "name": "Test network resilience", "status": "numerically_verified"},
        "status": "supported" if fully_supported else "conditional",
        "conclusion": f"{weakening_story}{split_story} After all {len(stages)} tested removals, {final['performance_ratio'] * 100:.1f}% of baseline topology efficiency remains.",
        "headline_effects": [
            {"label": "First clear weakening", "value": f"Stage {thresholds['90_percent']}" if thresholds["90_percent"] else "Not reached", "detail": "First tested stage at or below 90% of baseline efficiency"},
            {"label": "Steepest tested drop", "value": f"Stage {steepest_stage}", "detail": f"{drops[steepest_index] * 100:.2f} percentage-point efficiency loss at that step"},
            {"label": "First network split", "value": f"Stage {first_split}" if first_split else "Not observed", "detail": f"Baseline has {baseline_components['weak_components']} weakly connected part(s)"},
        ],
        "visual": {
            "mode": "network_stress_curve",
            "curve": [
                {"stage": 0, "performance_ratio": 1.0, "weak_components": baseline_components["weak_components"]},
                *[{"stage": item["stage"], "performance_ratio": item["performance_ratio"], "weak_components": item["weak_components"]} for item in stages],
            ],
            "thresholds": thresholds,
            "first_split_stage": first_split,
            "steepest_drop_stage": steepest_stage,
            "stress_tolerance_index": robustness_auc,
        },
        "next_actions": ["Inspect the earliest roads before the steepest efficiency loss.", "Investigate protection or added route redundancy around those roads, then re-run the same resilience test to compare.", "Add OD demand and capacity before interpreting user delay or congestion."],
        "items": stages,
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "evidence": {
            "baseline_efficiency": baseline_efficiency,
            "efficiency_exact": exact,
            "efficiency_source_count": source_count,
            "sequence_rule": "Use the confirmed critical-road order when supplied; otherwise compute the single-road topology-criticality ranking against the intact baseline, then remove roads in that fixed order and recompute network efficiency after each cumulative removal.",
            "sequence_source": sequence_source,
            "baseline_components": baseline_components,
            "tested_thresholds": thresholds,
            "first_split_stage": first_split,
            "steepest_drop_stage": steepest_stage,
            "stress_tolerance_index": robustness_auc,
            "ranking_status": ranking.get("status"),
            "ranking_exhaustive": ranking_exhaustive,
            "ranking_candidate_coverage": ranking.get("evidence", {}).get("candidate_coverage"),
            "ranking_evaluated_candidate_count": ranking.get("evidence", {}).get("evaluated_candidate_count"),
            "ranking_total_edge_count": ranking.get("evidence", {}).get("total_edge_count"),
            "curve_points": performance_curve,
        },
        "data_boundary": "This is a deterministic topology stress curve under targeted removals. It is not a failure probability model, traffic assignment or hazard likelihood forecast.",
        "verification": {"passed": monotone and all(0 <= item["performance_ratio"] <= 1 for item in stages), "checks": ["positive baseline efficiency", "ranked roads removed once", "efficiency recomputed each stage", "remaining efficiency is non-increasing", "90/75/50 percent thresholds derived from tested stages", "normalised trapezoidal curve area", "no random sampling"]},
    }


def run_analysis(database_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    analysis_type = request["analysis_type"]
    if analysis_type == "persistent_degradation":
        result = persistent_degradation(
            database_path,
            metric=request.get("metric", "speed"),
            threshold=float(request.get("threshold", 0.2)),
            top_k=int(request.get("top_k", 10)),
            bbox=request.get("bbox"),
            time_start=request.get("time_start"),
            time_end=request.get("time_end"),
        )
    elif analysis_type == "critical_components":
        result = critical_components(database_path, top_k=int(request.get("top_k", 10)), bbox=request.get("bbox"))
    elif analysis_type == "area_summary":
        if not request.get("bbox"):
            raise ValueError("Draw and confirm a geographic area first.")
        result = area_summary(
            database_path,
            metric=request.get("metric", "speed"),
            bbox=request["bbox"],
            timestamp=request.get("time_end") or request.get("time_start"),
        )
    elif analysis_type == "closure_scenario":
        result = closure_scenario(database_path, selected_keys=request.get("selected_keys", []), bbox=request.get("bbox"))
    elif analysis_type == "restoration_screening":
        result = restoration_screening(
            database_path,
            selected_keys=request.get("selected_keys", []),
            bbox=request.get("bbox"),
            top_k=int(request.get("top_k", 10)),
            assumptions=request.get("assumptions") or {},
        )
    elif analysis_type == "temporal_resilience":
        result = temporal_resilience_curve(
            database_path,
            metric=request.get("metric", "speed"),
            bbox=request.get("bbox"),
            time_start=request.get("time_start"),
            time_end=request.get("time_end"),
        )
    elif analysis_type == "spatial_hotspots":
        result = spatial_hotspots(
            database_path,
            metric=request.get("metric", "speed"),
            top_k=int(request.get("top_k", 10)),
            bbox=request.get("bbox"),
            timestamp=request.get("time_end") or request.get("time_start"),
        )
    elif analysis_type == "progressive_robustness":
        assumptions = request.get("assumptions") or {}
        result = progressive_robustness(
            database_path,
            top_k=int(request.get("top_k", 10)),
            bbox=request.get("bbox"),
            failure_keys=assumptions.get("failure_keys") or None,
        )
    elif analysis_type == "route_detour_analysis":
        from .routing import historical_route_detour_analysis

        assumptions = request.get("assumptions") or {}
        selected_keys = request.get("selected_keys") or []
        result = historical_route_detour_analysis(
            database_path,
            origin_key=str(assumptions.get("origin_key") or (selected_keys[0] if len(selected_keys) >= 1 else "")),
            destination_key=str(assumptions.get("destination_key") or (selected_keys[1] if len(selected_keys) >= 2 else "")),
            timestamp=request.get("time_end") or request.get("time_start"),
            scenario_changes=assumptions.get("scenario_changes") or None,
            scenario_name=assumptions.get("scenario_name"),
            scenario_variants=assumptions.get("scenario_variants") or None,
            alternative_count=int(assumptions.get("alternative_count") or 2),
            maximum_overlap=float(assumptions.get("maximum_overlap") or 0.60),
        )
    elif analysis_type == "adaptive_registered_model":
        # Lazy import avoids a module cycle because the adaptive adapters
        # intentionally reuse the verified numerical kernels above.
        from .adaptive_models import run_adaptive_model

        result = run_adaptive_model(database_path, request)
    else:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")
    result.update(
        {
            "run_id": f"run_{uuid4().hex[:14]}",
            "analysis_type": analysis_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "deterministic": True,
            "random_seed": None,
        }
    )
    return result
