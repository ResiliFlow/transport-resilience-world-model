from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .models import (
    _add_display_scores,
    _connect,
    _metric_parts,
    _nearest_timestamp,
    _primary_semantics,
    _quality_label,
    _severity,
    _summary,
)
from .transport_data import EXACT_GEOMETRY_TABLE


MAPPING_METADATA_KEY = "network_mapping_v1"
MAPPING_TABLE = "inferred_network_nodes"
MAPPING_LINK_TABLE = "inferred_network_links"
MAPPING_METHOD = "incident_reference_point_coordinate_median_v1"
MINIMUM_LINE_LENGTH_M = 1.0
AMBIGUOUS_NODE_SPREAD_M = 10_000.0


def _node_key(region_id: str, node_id: str, global_node_ids: bool) -> str:
    return str(node_id) if global_node_ids else f"{region_id}\x1f{node_id}"


def _haversine_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    longitude_a, latitude_a = first
    longitude_b, latitude_b = second
    latitude_a_rad = math.radians(latitude_a)
    latitude_b_rad = math.radians(latitude_b)
    delta_latitude = latitude_b_rad - latitude_a_rad
    delta_longitude = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_latitude / 2.0) ** 2
        + math.cos(latitude_a_rad) * math.cos(latitude_b_rad) * math.sin(delta_longitude / 2.0) ** 2
    )
    return 6_371_008.8 * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))


def _mapping_table_exists(connection: Any, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _mapping_metadata(connection: Any) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT value_json FROM metadata WHERE key = ?",
        (MAPPING_METADATA_KEY,),
    ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_counts(connection: Any) -> tuple[int, int]:
    row = connection.execute(
        """
        SELECT
            SUM(from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''),
            SUM(from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''
                AND latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
        FROM entities
        """
    ).fetchone()
    return int(row[0] or 0), int(row[1] or 0)


def network_mapping_status(database_path: Path) -> dict[str, Any]:
    """Return whether a deterministic, saved point-to-link mapping can be used."""
    with _connect(database_path) as connection:
        summary = _summary(connection)
        topology_count, source_count = _source_counts(connection)
        metadata = _mapping_metadata(connection)
        table_exists = _mapping_table_exists(connection, MAPPING_TABLE) and _mapping_table_exists(
            connection, MAPPING_LINK_TABLE
        )
        exact_count = int(connection.execute(
            f"SELECT COUNT(*) FROM {EXACT_GEOMETRY_TABLE}"
        ).fetchone()[0]) if _mapping_table_exists(connection, EXACT_GEOMETRY_TABLE) else 0
        exact_kind_row = connection.execute(
            f"SELECT geometry_kind, COUNT(*) AS n FROM {EXACT_GEOMETRY_TABLE} GROUP BY geometry_kind ORDER BY n DESC LIMIT 1"
        ).fetchone() if exact_count else None

    available = exact_count > 0 or source_count > 0
    ready = exact_count > 0 or bool(table_exists and metadata and int(metadata.get("mapped_link_count") or 0) > 0)
    if exact_count:
        status = "ready"
        reason = "Uploaded, joined or topology-layout road geometry is ready."
    elif ready:
        status = "ready"
        reason = "A saved inferred link view is available."
    elif available:
        status = "available"
        reason = "From-node, to-node and representative coordinates can be mapped after user confirmation."
    elif topology_count:
        status = "unavailable"
        reason = "Topology fields exist, but no link has a valid WGS84 representative coordinate."
    else:
        status = "unavailable"
        reason = "Map both From node and To node fields to create an inferred link view."

    result: dict[str, Any] = {
        "available": available,
        "ready": ready,
        "status": status,
        "reason": reason,
        "topology_link_count": topology_count,
        "source_link_count": exact_count or source_count,
        "topology_node_scope": summary.get("topology_node_scope", "global"),
        "method": "uploaded_or_joined_transport_geometry_v1" if exact_count else MAPPING_METHOD,
        "geometry_kind": str(exact_kind_row[0]) if exact_kind_row else "inferred_topology_line" if ready else "representative_point",
        "exact_geometry": exact_count > 0,
        "mapped_link_count": exact_count if exact_count else int((metadata or {}).get("mapped_link_count") or 0),
        "warning": (
            summary.get("geometry_note")
            if exact_count else
            "Mapped links are straight topology reconstructions from uploaded reference points. "
            "They are not surveyed road centre-lines and are not snapped to OpenStreetMap."
        ),
    }
    if metadata:
        result.update(metadata)
        result.update({"available": available, "ready": ready, "status": status, "reason": reason})
    return result


def build_network_mapping(database_path: Path) -> dict[str, Any]:
    """Build and persist deterministic topology-node coordinates from incident link points."""
    existing = network_mapping_status(database_path)
    if existing.get("exact_geometry"):
        return existing
    with _connect(database_path) as connection:
        summary = _summary(connection)
        global_node_ids = summary.get("topology_node_scope", "global") == "global"
        rows = connection.execute(
            """
            SELECT region_id, entity_id, from_node, to_node, latitude, longitude
            FROM entities
            WHERE from_node IS NOT NULL AND from_node <> ''
              AND to_node IS NOT NULL AND to_node <> ''
              AND latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180
            ORDER BY region_id, entity_id
            """
        ).fetchall()
        if not rows:
            raise ValueError(
                "This project cannot create a link view. Confirm From node, To node, latitude and longitude mappings."
            )

        node_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
        node_identity: dict[str, tuple[str, str | None]] = {}
        for row in rows:
            region_id = str(row["region_id"])
            coordinate = (float(row["longitude"]), float(row["latitude"]))
            endpoint_ids = (str(row["from_node"]), str(row["to_node"]))
            for endpoint_id in dict.fromkeys(endpoint_ids):
                key = _node_key(region_id, endpoint_id, global_node_ids)
                node_samples[key].append(coordinate)
                node_identity[key] = (endpoint_id, None if global_node_ids else region_id)

        nodes: dict[str, dict[str, Any]] = {}
        for key, samples in node_samples.items():
            coordinate = (
                float(median(sample[0] for sample in samples)),
                float(median(sample[1] for sample in samples)),
            )
            spreads = [_haversine_m(coordinate, sample) for sample in samples]
            node_id, region_id = node_identity[key]
            nodes[key] = {
                "node_key": key,
                "node_id": node_id,
                "region_id": region_id,
                "longitude": coordinate[0],
                "latitude": coordinate[1],
                "support_count": len(samples),
                "median_spread_m": float(median(spreads)) if spreads else 0.0,
                "max_spread_m": float(max(spreads, default=0.0)),
            }

        mapped_link_count = 0
        degenerate_link_count = 0
        unresolved_link_count = 0
        inferred_lengths: list[float] = []
        mapped_links: list[tuple[str, str, str, str, str, float]] = []
        for row in rows:
            region_id = str(row["region_id"])
            from_key = _node_key(region_id, str(row["from_node"]), global_node_ids)
            to_key = _node_key(region_id, str(row["to_node"]), global_node_ids)
            first = nodes.get(from_key)
            second = nodes.get(to_key)
            if first is None or second is None:
                unresolved_link_count += 1
                continue
            length = _haversine_m(
                (float(first["longitude"]), float(first["latitude"])),
                (float(second["longitude"]), float(second["latitude"])),
            )
            if length < MINIMUM_LINE_LENGTH_M:
                degenerate_link_count += 1
                continue
            mapped_link_count += 1
            inferred_lengths.append(length)
            mapped_links.append(
                (
                    f"{region_id}::{row['entity_id']}",
                    region_id,
                    str(row["entity_id"]),
                    from_key,
                    to_key,
                    length,
                )
            )

        if mapped_link_count == 0:
            raise ValueError(
                "Topology IDs were found, but the uploaded reference points do not provide enough spatial separation "
                "to draw any reliable link. The original point view remains available."
            )

        ambiguous_node_count = sum(
            float(node["max_spread_m"]) > AMBIGUOUS_NODE_SPREAD_M for node in nodes.values()
        )
        coverage_ratio = mapped_link_count / len(rows)
        ambiguous_ratio = ambiguous_node_count / max(1, len(nodes))
        if coverage_ratio >= 0.95 and ambiguous_ratio <= 0.05:
            quality = "good_topology_reconstruction"
        elif coverage_ratio >= 0.75 and ambiguous_ratio <= 0.2:
            quality = "review_recommended"
        else:
            quality = "limited_geometry"

        result = {
            "available": True,
            "ready": True,
            "status": "ready",
            "reason": "A saved inferred link view is available.",
            "built_at": datetime.now(timezone.utc).isoformat(),
            "method": MAPPING_METHOD,
            "algorithm_version": 1,
            "geometry_kind": "inferred_topology_line",
            "topology_node_scope": "global" if global_node_ids else "per_region",
            "source_link_count": len(rows),
            "mapped_link_count": mapped_link_count,
            "excluded_link_count": degenerate_link_count + unresolved_link_count,
            "degenerate_link_count": degenerate_link_count,
            "unresolved_link_count": unresolved_link_count,
            "inferred_node_count": len(nodes),
            "ambiguous_node_count": ambiguous_node_count,
            "coverage_ratio": coverage_ratio,
            "quality": quality,
            "median_inferred_line_length_m": float(median(inferred_lengths)),
            "warning": (
                "Mapped links are straight topology reconstructions from uploaded reference points. "
                "They are not surveyed road centre-lines and are not snapped to OpenStreetMap."
            ),
        }

        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MAPPING_TABLE} (
                node_key TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                region_id TEXT,
                longitude REAL NOT NULL,
                latitude REAL NOT NULL,
                support_count INTEGER NOT NULL,
                median_spread_m REAL NOT NULL,
                max_spread_m REAL NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(f"DELETE FROM {MAPPING_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {MAPPING_TABLE}
                (node_key, node_id, region_id, longitude, latitude, support_count, median_spread_m, max_spread_m)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node["node_key"], node["node_id"], node["region_id"], node["longitude"], node["latitude"],
                    node["support_count"], node["median_spread_m"], node["max_spread_m"],
                )
                for node in nodes.values()
            ],
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MAPPING_LINK_TABLE} (
                link_key TEXT PRIMARY KEY,
                region_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                from_node_key TEXT NOT NULL,
                to_node_key TEXT NOT NULL,
                inferred_length_m REAL NOT NULL,
                UNIQUE(region_id, entity_id)
            ) WITHOUT ROWID
            """
        )
        connection.execute(f"DELETE FROM {MAPPING_LINK_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {MAPPING_LINK_TABLE}
                (link_key, region_id, entity_id, from_node_key, to_node_key, inferred_length_m)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            mapped_links,
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value_json) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (MAPPING_METADATA_KEY, json.dumps(result)),
        )
        connection.commit()
        return result


def network_features(
    database_path: Path,
    *,
    metric: str,
    timestamp: int | None = None,
    bbox: list[float] | None = None,
    limit: int = 20_000,
) -> dict[str, Any]:
    """Return current observations on saved inferred topology lines."""
    status = network_mapping_status(database_path)
    if not status["ready"]:
        raise ValueError("Create and save the link mapping before opening the mapped-link view.")
    if status.get("exact_geometry"):
        return _exact_network_features(
            database_path,
            metric=metric,
            timestamp=timestamp,
            bbox=bbox,
            limit=limit,
            status=status,
        )

    value_column, baseline_column, default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        clauses = [
            "o.timestamp_epoch = ?",
            "e.latitude BETWEEN -90 AND 90",
            "e.longitude BETWEEN -180 AND 180",
        ]
        parameters: list[Any] = [selected_time]
        if bbox:
            west, south, east, north = bbox
            clauses.extend(["e.longitude BETWEEN ? AND ?", "e.latitude BETWEEN ? AND ?"])
            parameters.extend([west, east, south, north])
        joins = f"""
            JOIN entities e ON e.entity_pk = o.entity_pk
            JOIN times t ON t.timestamp_epoch = o.timestamp_epoch
            JOIN {MAPPING_LINK_TABLE} ml ON ml.region_id = e.region_id AND ml.entity_id = e.entity_id
            JOIN {MAPPING_TABLE} nf ON nf.node_key = ml.from_node_key
            JOIN {MAPPING_TABLE} nt ON nt.node_key = ml.to_node_key
        """
        total_count = int(connection.execute(
            f"SELECT COUNT(*) FROM observations o {joins} WHERE {' AND '.join(clauses)}",
            parameters,
        ).fetchone()[0])
        rows = connection.execute(
            f"""
            SELECT e.region_id, e.entity_id, e.from_node, e.to_node,
                   e.latitude AS reference_latitude, e.longitude AS reference_longitude,
                   nf.longitude AS from_longitude, nf.latitude AS from_latitude,
                   nt.longitude AS to_longitude, nt.latitude AS to_latitude,
                   nf.support_count AS from_support_count, nt.support_count AS to_support_count,
                   nf.max_spread_m AS from_spread_m, nt.max_spread_m AS to_spread_m,
                   ml.inferred_length_m,
                   {value_column} AS value, {baseline_column} AS baseline,
                   o.quality_flags, t.timestamp_text
            FROM observations o
            {joins}
            WHERE {' AND '.join(clauses)}
            ORDER BY ((e.entity_pk * 1103515245 + 12345) & 2147483647), e.entity_pk
            LIMIT ?
            """,
            (*parameters, limit + 1),
        ).fetchall()
        truncated = total_count > limit
        rows = rows[:limit]
        unit = summary.get("metrics", {}).get(metric, {}).get("unit", default_unit)
        features = []
        for row in rows:
            value = float(row["value"]) if row["value"] is not None else None
            baseline = float(row["baseline"]) if row["baseline"] is not None else None
            first = (float(row["from_longitude"]), float(row["from_latitude"]))
            second = (float(row["to_longitude"]), float(row["to_latitude"]))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [list(first), list(second)]},
                    "properties": {
                        "key": f"{row['region_id']}::{row['entity_id']}",
                        "region_id": row["region_id"],
                        "entity_id": row["entity_id"],
                        "from_node": row["from_node"],
                        "to_node": row["to_node"],
                        "reference_latitude": float(row["reference_latitude"]),
                        "reference_longitude": float(row["reference_longitude"]),
                        "value": value,
                        "baseline": baseline,
                        "severity": _severity(metric, value, baseline, _primary_semantics(summary)),
                        "quality": _quality_label(row["quality_flags"]),
                        "timestamp": row["timestamp_text"],
                        "metric": metric,
                        "unit": unit,
                        "geometry_kind": "inferred_topology_line",
                        "mapping_method": MAPPING_METHOD,
                        "mapping_support": min(int(row["from_support_count"]), int(row["to_support_count"])),
                        "maximum_node_spread_m": max(float(row["from_spread_m"]), float(row["to_spread_m"])),
                        "inferred_line_length_m": float(row["inferred_length_m"]),
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
                "Deterministic hash-balanced link sample; zoom or pan to request the current viewport."
                if truncated else "Every matching inferred link observation is shown."
            ),
            "geometry_kind": "inferred_topology_line",
            "geometry_note": status["warning"],
            "mapping": status,
        }


def _exact_network_features(
    database_path: Path,
    *,
    metric: str,
    timestamp: int | None,
    bbox: list[float] | None,
    limit: int,
    status: dict[str, Any],
) -> dict[str, Any]:
    value_column, baseline_column, default_unit = _metric_parts(metric)
    with _connect(database_path) as connection:
        summary = _summary(connection)
        selected_time = _nearest_timestamp(connection, timestamp)
        clauses = ["o.timestamp_epoch = ?"]
        parameters: list[Any] = [selected_time]
        if bbox:
            west, south, east, north = bbox
            clauses.extend([
                "g.max_longitude >= ?", "g.min_longitude <= ?",
                "g.max_latitude >= ?", "g.min_latitude <= ?",
            ])
            parameters.extend([west, east, south, north])
        joins = f"""
            JOIN entities e ON e.entity_pk = o.entity_pk
            JOIN times t ON t.timestamp_epoch = o.timestamp_epoch
            JOIN {EXACT_GEOMETRY_TABLE} g ON g.region_id = e.region_id AND g.entity_id = e.entity_id
        """
        total_count = int(connection.execute(
            f"SELECT COUNT(*) FROM observations o {joins} WHERE {' AND '.join(clauses)}",
            parameters,
        ).fetchone()[0])
        rows = connection.execute(
            f"""
            SELECT e.region_id, e.entity_id, e.from_node, e.to_node, e.latitude, e.longitude,
                   g.geometry_json, g.geometry_kind,
                   {value_column} AS value, {baseline_column} AS baseline,
                   o.quality_flags, t.timestamp_text
            FROM observations o
            {joins}
            WHERE {' AND '.join(clauses)}
            ORDER BY ((e.entity_pk * 1103515245 + 12345) & 2147483647), e.entity_pk
            LIMIT ?
            """,
            (*parameters, limit + 1),
        ).fetchall()
        truncated = total_count > limit
        rows = rows[:limit]
        unit = summary.get("metrics", {}).get(metric, {}).get("unit", default_unit)
        features = []
        for row in rows:
            try:
                geometry = json.loads(row["geometry_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            value = float(row["value"]) if row["value"] is not None else None
            baseline = float(row["baseline"]) if row["baseline"] is not None else None
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "key": f"{row['region_id']}::{row['entity_id']}",
                    "region_id": row["region_id"], "entity_id": row["entity_id"],
                    "from_node": row["from_node"], "to_node": row["to_node"],
                    "reference_latitude": row["latitude"], "reference_longitude": row["longitude"],
                    "value": value, "baseline": baseline,
                    "severity": _severity(metric, value, baseline, _primary_semantics(summary)),
                    "quality": _quality_label(row["quality_flags"]), "timestamp": row["timestamp_text"],
                    "metric": metric, "unit": unit, "geometry_kind": row["geometry_kind"],
                    "mapping_method": status.get("method"),
                },
            })
        _add_display_scores(features)
        values = [feature["properties"]["value"] for feature in features if feature["properties"]["value"] is not None]
        return {
            "type": "FeatureCollection", "features": features,
            "timestamp_epoch": selected_time,
            "timestamp_label": summary.get("time_labels", {}).get(str(selected_time), str(selected_time)),
            "metric": metric, "unit": unit, "truncated": truncated, "total_count": total_count,
            "displayed_count": len(features), "value_min": min(values, default=None), "value_max": max(values, default=None),
            "sampling_method": "Deterministic hash-balanced link sample; zoom for local detail." if truncated else "Every matching road-link observation is shown.",
            "geometry_kind": status.get("geometry_kind"), "geometry_note": status.get("warning"), "mapping": status,
        }
