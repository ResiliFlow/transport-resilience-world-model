from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import gzip
import json
import math
from pathlib import Path
import shutil
import sqlite3
from statistics import median
import subprocess
import time
from typing import Any, Callable, Iterable
from xml.etree import ElementTree
from xml.sax.saxutils import quoteattr

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = PROJECT_ROOT / "matsim_worker"
WORKER_JAR = WORKER_ROOT / "target" / "twm-matsim-worker.jar"
MATSIM_STORAGE = PROJECT_ROOT / "storage" / "matsim"
DEFAULT_SEED = 41729


@dataclass(frozen=True)
class LinkRecord:
    key: str
    region_id: str
    entity_id: str
    from_node: str
    to_node: str
    length_m: float
    free_speed_mps: float
    capacity_vph: float
    coordinates: tuple[tuple[float, float], ...]
    capacity_uploaded: bool
    impedance_uploaded: bool


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    """Return available columns so older saved projects remain readable.

    Optional MATSim supply fields were added after the first project schema.
    Their absence means "not uploaded", not a server error.
    """
    if not _table_exists(connection, name):
        return set()
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{name}")')}


def _worker_runtime() -> dict[str, Any]:
    java = shutil.which("java")
    return {
        "java_ready": bool(java),
        "java_path": java,
        "worker_ready": WORKER_JAR.exists(),
        "worker_jar": str(WORKER_JAR),
    }


def assess_matsim_readiness(project: dict[str, Any]) -> dict[str, Any]:
    """Return a strict, user-facing MATSim data and runtime contract."""
    database_path = Path(project["database_path"])
    summary = project.get("summary", {})
    metrics = summary.get("metrics", {})
    with closing(_connect(database_path)) as connection:
        entity_columns = _table_columns(connection, "entities")
        topology_links = (
            int(connection.execute("SELECT COUNT(*) FROM inferred_network_links").fetchone()[0])
            if _table_exists(connection, "inferred_network_links") else 0
        )
        od_rows = (
            int(connection.execute(
                "SELECT COUNT(*) FROM transport_od WHERE demand IS NOT NULL AND demand > 0"
            ).fetchone()[0])
            if _table_exists(connection, "transport_od") else 0
        )
        if topology_links and "capacity" in entity_columns:
            capacity_links = int(connection.execute(
                """
                SELECT COUNT(*)
                FROM inferred_network_links l
                JOIN entities e ON e.region_id = l.region_id AND e.entity_id = l.entity_id
                WHERE e.capacity IS NOT NULL AND e.capacity > 0
                """
            ).fetchone()[0])
        else:
            capacity_links = 0
        impedance_sources = [
            column
            for column in ("free_flow_time", "baseline_speed", "baseline_travel_time")
            if column in entity_columns
        ]
        if topology_links and "length_m" in entity_columns and impedance_sources:
            impedance_predicate = " OR ".join(
                f"(e.{column} IS NOT NULL AND e.{column} > 0)" for column in impedance_sources
            )
            positive_impedance = int(connection.execute(
                f"""
                SELECT COUNT(*)
                FROM inferred_network_links l
                JOIN entities e ON e.region_id = l.region_id AND e.entity_id = l.entity_id
                WHERE e.length_m IS NOT NULL AND e.length_m > 0
                  AND ({impedance_predicate})
                """
            ).fetchone()[0])
        else:
            positive_impedance = 0

    runtime = _worker_runtime()
    missing_for_full: list[str] = []
    if not topology_links:
        missing_for_full.append("routable From/To topology")
    if positive_impedance < topology_links:
        missing_for_full.append(
            f"positive length plus speed or travel time on every routable link ({positive_impedance:,}/{topology_links:,})"
        )
    if capacity_links < topology_links:
        missing_for_full.append(f"road capacity on every routable link ({capacity_links:,}/{topology_links:,})")
    if not od_rows:
        missing_for_full.append("OD demand")

    full_supply_ready = bool(
        topology_links
        and positive_impedance == topology_links
        and capacity_links == topology_links
    )
    full_demand_ready = bool(od_rows and full_supply_ready)

    if topology_links < 2 or positive_impedance < 1:
        tier = "unavailable"
        label = "MATSim needs a routable network"
        status = "unavailable"
        explanation = (
            "Add From/To road connections and a positive length with speed or travel time. "
            "The existing network model remains available where supported."
        )
    elif full_demand_ready:
        tier = "demand_calibrated"
        label = "Full demand MATSim simulation"
        status = "ready" if runtime["java_ready"] and runtime["worker_ready"] else "runtime_missing"
        explanation = (
            "Uses uploaded OD demand and road supply. Baseline and disrupted networks run with "
            "the same agents, seed and sampling factor."
        )
    else:
        tier = "exploratory_probe"
        label = "Exploratory MATSim probe simulation"
        status = "ready" if runtime["java_ready"] and runtime["worker_ready"] else "runtime_missing"
        explanation = (
            "Uses a fixed set of probe trips to reveal rerouting and disconnection. Missing OD or "
            "capacity means it does not claim observed congestion, population impact or evacuation demand."
        )

    return {
        "status": status,
        "tier": tier,
        "label": label,
        "explanation": explanation,
        "can_run": tier != "unavailable" and runtime["java_ready"] and runtime["worker_ready"],
        "topology_links": topology_links,
        "positive_impedance_links": positive_impedance,
        "capacity_links": capacity_links,
        "capacity_coverage": capacity_links / topology_links if topology_links else 0.0,
        "impedance_coverage": positive_impedance / topology_links if topology_links else 0.0,
        "full_supply_ready": full_supply_ready,
        "od_rows": od_rows,
        "missing_for_full": missing_for_full,
        "available_metrics": [key for key, item in metrics.items() if item.get("available")],
        "runtime": runtime,
        "scientific_boundary": (
            "Full demand tier supports network loading for the uploaded or deterministically sampled demand. "
            "Exploratory tier supports comparative network probing only. Neither tier is live traffic control."
        ),
    }


def _haversine_metres(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(value)))


def _geometry_length(coordinates: tuple[tuple[float, float], ...]) -> float:
    if len(coordinates) < 2:
        return 1.0
    geographic = all(-180 <= point[0] <= 180 and -90 <= point[1] <= 90 for point in coordinates)
    distance = 0.0
    for first, second in zip(coordinates, coordinates[1:]):
        distance += _haversine_metres(first, second) if geographic else math.dist(first, second)
    return max(distance, 1.0)


def _line_coordinates(raw: str | None, fallback: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if not raw:
        return fallback
    try:
        geometry = json.loads(raw)
        if geometry.get("type") == "LineString":
            coordinates = tuple((float(item[0]), float(item[1])) for item in geometry.get("coordinates", []))
            if len(coordinates) >= 2:
                return coordinates
        if geometry.get("type") == "MultiLineString":
            lines = geometry.get("coordinates", [])
            longest = max(lines, key=len, default=[])
            coordinates = tuple((float(item[0]), float(item[1])) for item in longest)
            if len(coordinates) >= 2:
                return coordinates
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return fallback


def _load_links(database_path: Path) -> tuple[list[LinkRecord], dict[str, tuple[float, float]], dict[str, str]]:
    with closing(_connect(database_path)) as connection:
        entity_columns = _table_columns(connection, "entities")
        entity_value = lambda column: f"e.{column} AS {column}" if column in entity_columns else f"NULL AS {column}"
        geometry_join = (
            "LEFT JOIN transport_link_geometries g ON g.link_key = l.link_key"
            if _table_exists(connection, "transport_link_geometries") else ""
        )
        geometry_column = "g.geometry_json" if geometry_join else "NULL AS geometry_json"
        rows = connection.execute(f"""
            SELECT l.link_key, l.region_id, l.entity_id, l.from_node_key, l.to_node_key,
                   l.inferred_length_m, {entity_value('length_m')},
                   {entity_value('baseline_speed')}, {entity_value('baseline_travel_time')},
                   {entity_value('capacity')}, {entity_value('free_flow_time')},
                   fn.longitude AS from_lon, fn.latitude AS from_lat,
                   tn.longitude AS to_lon, tn.latitude AS to_lat,
                   {geometry_column}
            FROM inferred_network_links l
            JOIN entities e ON e.region_id = l.region_id AND e.entity_id = l.entity_id
            JOIN inferred_network_nodes fn ON fn.node_key = l.from_node_key
            JOIN inferred_network_nodes tn ON tn.node_key = l.to_node_key
            {geometry_join}
            ORDER BY l.link_key
        """).fetchall()
        node_rows = connection.execute(
            "SELECT node_key, node_id, longitude, latitude FROM inferred_network_nodes ORDER BY node_key"
        ).fetchall()

    nodes = {str(row["node_key"]): (float(row["longitude"]), float(row["latitude"])) for row in node_rows}
    node_ids = {str(row["node_key"]): str(row["node_id"]) for row in node_rows}
    links: list[LinkRecord] = []
    for row in rows:
        fallback = (
            (float(row["from_lon"]), float(row["from_lat"])),
            (float(row["to_lon"]), float(row["to_lat"])),
        )
        coordinates = _line_coordinates(row["geometry_json"], fallback)
        length = next((float(value) for value in (row["length_m"], row["inferred_length_m"]) if value is not None and float(value) > 0), None)
        length = length or _geometry_length(coordinates)
        free_flow_time = float(row["free_flow_time"]) if row["free_flow_time"] is not None else None
        baseline_speed = float(row["baseline_speed"]) if row["baseline_speed"] is not None else None
        baseline_time = float(row["baseline_travel_time"]) if row["baseline_travel_time"] is not None else None
        if free_flow_time and free_flow_time > 0:
            speed = length / (free_flow_time * 60.0)
        elif baseline_speed and baseline_speed > 0:
            speed = baseline_speed / 3.6
        elif baseline_time and baseline_time > 0:
            speed = length / (baseline_time * 60.0)
        else:
            speed = 13.89
        capacity_uploaded = row["capacity"] is not None and float(row["capacity"]) > 0
        capacity = float(row["capacity"]) if capacity_uploaded else 900.0
        links.append(LinkRecord(
            key=str(row["link_key"]), region_id=str(row["region_id"]), entity_id=str(row["entity_id"]),
            from_node=str(row["from_node_key"]), to_node=str(row["to_node_key"]),
            length_m=max(1.0, float(length)), free_speed_mps=max(0.5, min(55.0, speed)),
            capacity_vph=max(1.0, capacity), coordinates=coordinates,
            capacity_uploaded=capacity_uploaded,
            impedance_uploaded=bool(free_flow_time or baseline_speed or baseline_time),
        ))
    return links, nodes, node_ids


def _scenario_link(link: LinkRecord, changes: dict[str, dict[str, Any]]) -> LinkRecord | None:
    change = changes.get(link.key)
    if not change:
        return link
    effect = change.get("effect")
    magnitude = float(change.get("magnitude") or 0.0)
    if effect == "closed":
        # A static closure removes the road from the disrupted supply.  Using
        # near-zero queue capacity leaves a nominally open road in MATSim and
        # can make a few dozen affected agents wait for tens of thousands of
        # simulated hours.  Removal gives the router the intended choice:
        # detour on the remaining network or report no feasible route.
        return None
    if effect == "speed_reduction":
        speed = link.free_speed_mps * max(0.01, 1.0 - magnitude)
    elif effect == "travel_time_increase":
        speed = link.free_speed_mps / max(1.01, 1.0 + magnitude)
    else:
        speed = link.free_speed_mps
    return LinkRecord(**{**link.__dict__, "free_speed_mps": max(0.2, speed)})


def _matsim_dtd_uri(name: str) -> str:
    """Resolve MATSim XML DTDs from the packaged worker, never the network."""

    if name not in {"network_v2.dtd", "population_v6.dtd"}:
        raise ValueError(f"Unsupported MATSim DTD: {name}")
    return f"jar:{WORKER_JAR.resolve().as_uri()}!/dtd/{name}"


def _write_network(path: Path, nodes: dict[str, tuple[float, float]], links: Iterable[LinkRecord]) -> None:
    links = list(links)
    used_nodes = {link.from_node for link in links} | {link.to_node for link in links}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        stream.write(f'<!DOCTYPE network SYSTEM {quoteattr(_matsim_dtd_uri("network_v2.dtd"))}>\n')
        stream.write('<network>\n  <nodes>\n')
        for node_id in sorted(used_nodes):
            x, y = nodes[node_id]
            stream.write(f'    <node id={quoteattr(_matsim_id(node_id, "n"))} x={quoteattr(f"{x:.9f}")} y={quoteattr(f"{y:.9f}")} />\n')
        stream.write('  </nodes>\n  <links capperiod="01:00:00" effectivecellsize="7.5" effectivelanewidth="3.75">\n')
        for link in links:
            stream.write(
                f'    <link id={quoteattr(_matsim_id(link.key, "l"))} from={quoteattr(_matsim_id(link.from_node, "n"))} to={quoteattr(_matsim_id(link.to_node, "n"))} '
                f'length={quoteattr(f"{link.length_m:.3f}")} freespeed={quoteattr(f"{link.free_speed_mps:.6f}")} '
                f'capacity={quoteattr(f"{link.capacity_vph:.3f}")} permlanes="1.0" modes="car" />\n'
            )
        stream.write('  </links>\n</network>\n')


def _link_indexes(links: list[LinkRecord]) -> tuple[dict[str, list[LinkRecord]], dict[str, list[LinkRecord]], dict[str, LinkRecord]]:
    outgoing: dict[str, list[LinkRecord]] = defaultdict(list)
    incoming: dict[str, list[LinkRecord]] = defaultdict(list)
    by_key: dict[str, LinkRecord] = {}
    for link in links:
        outgoing[link.from_node].append(link)
        incoming[link.to_node].append(link)
        by_key[link.key] = link
    for values in outgoing.values():
        values.sort(key=lambda item: item.key)
    for values in incoming.values():
        values.sort(key=lambda item: item.key)
    return outgoing, incoming, by_key


def _resolve_node(value: str, region: str | None, node_ids: dict[str, str]) -> str | None:
    value = str(value).strip()
    if value in node_ids:
        return value
    candidates = [key for key, node_id in node_ids.items() if node_id == value]
    if region:
        scoped = [key for key in candidates if key.startswith(f"{region}\x1f")]
        if scoped:
            return sorted(scoped)[0]
    return sorted(candidates)[0] if candidates else None


def _allocate_agent_counts(demands: list[float], maximum: int) -> tuple[list[int], float]:
    total = sum(max(0.0, value) for value in demands)
    if total <= 0:
        return [0] * len(demands), 1.0
    target = min(maximum, max(1, int(round(total))))
    factor = min(1.0, target / total)
    raw = [max(0.0, value) * factor for value in demands]
    counts = [int(math.floor(value)) for value in raw]
    remaining = target - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts, factor


def _feasible_activity_link_pair(
    graph: nx.DiGraph,
    origin_links: Iterable[LinkRecord],
    destination_links: Iterable[LinkRecord],
) -> tuple[LinkRecord, LinkRecord] | None:
    """Choose a deterministic feasible activity-link pair for one OD row.

    Selecting the lexicographically first outgoing and incoming links can
    reject a valid OD relation when either node has more than one incident
    road.  This routine screens every endpoint-link combination and selects
    the fewest-hop feasible pair with stable link-key tie breaks.
    """

    feasible: list[tuple[int, str, str, LinkRecord, LinkRecord]] = []
    for origin_link in sorted(origin_links, key=lambda item: item.key):
        for destination_link in sorted(destination_links, key=lambda item: item.key):
            if origin_link.key == destination_link.key:
                continue
            try:
                hops = int(nx.shortest_path_length(
                    graph,
                    source=origin_link.to_node,
                    target=destination_link.from_node,
                ))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            feasible.append((hops, origin_link.key, destination_link.key, origin_link, destination_link))
    if not feasible:
        return None
    _, _, _, origin_link, destination_link = min(feasible, key=lambda item: item[:3])
    return origin_link, destination_link


def _uploaded_agents(
    connection: sqlite3.Connection,
    links: list[LinkRecord],
    node_ids: dict[str, str],
    maximum: int,
) -> tuple[list[tuple[str, str, float]], float, int]:
    if not _table_exists(connection, "transport_od"):
        return [], 1.0, 0
    rows = connection.execute(
        "SELECT region_id, origin, destination, demand FROM transport_od WHERE demand > 0 ORDER BY od_pk"
    ).fetchall()
    outgoing, incoming, _ = _link_indexes(links)
    graph = nx.DiGraph()
    graph.add_edges_from((link.from_node, link.to_node) for link in links)
    resolved: list[tuple[str, str, float]] = []
    for row in rows:
        origin = _resolve_node(str(row["origin"]), row["region_id"], node_ids)
        destination = _resolve_node(str(row["destination"]), row["region_id"], node_ids)
        if not origin or not destination or origin == destination:
            continue
        origin_links = outgoing.get(origin, [])
        destination_links = incoming.get(destination, [])
        if not origin_links or not destination_links:
            continue
        activity_links = _feasible_activity_link_pair(graph, origin_links, destination_links)
        if activity_links is None:
            continue
        origin_link, destination_link = activity_links
        resolved.append((origin_link.key, destination_link.key, float(row["demand"])))
    counts, factor = _allocate_agent_counts([item[2] for item in resolved], maximum)
    agents: list[tuple[str, str, float]] = []
    person_index = 0
    for (origin_link, destination_link, _), count in zip(resolved, counts):
        for _ in range(count):
            departure = 7 * 3600 + (person_index * 37) % (2 * 3600)
            agents.append((origin_link, destination_link, float(departure)))
            person_index += 1
    return agents, factor, len(rows)


def _probe_agents(links: list[LinkRecord], maximum: int = 240) -> list[tuple[str, str, float]]:
    outgoing, incoming, _ = _link_indexes(links)
    graph = nx.DiGraph()
    graph.add_edges_from((link.from_node, link.to_node) for link in links)
    candidates = sorted(
        (node for node in graph if outgoing.get(node) and incoming.get(node)),
        key=lambda node: (-graph.degree(node), node),
    )
    if len(candidates) < 2:
        return []
    stride = max(1, len(candidates) // max(2, int(math.sqrt(maximum))))
    anchors = candidates[::stride][: min(len(candidates), 80)]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for offset in range(1, len(anchors)):
        for index, origin in enumerate(anchors):
            destination = anchors[(index + offset) % len(anchors)]
            if origin == destination or (origin, destination) in seen:
                continue
            seen.add((origin, destination))
            activity_links = _feasible_activity_link_pair(
                graph,
                outgoing.get(origin, []),
                incoming.get(destination, []),
            )
            if activity_links is None:
                continue
            pairs.append((origin, destination))
            if len(pairs) >= maximum:
                break
        if len(pairs) >= maximum:
            break
    agents = []
    for index, (origin, destination) in enumerate(pairs):
        activity_links = _feasible_activity_link_pair(
            graph,
            outgoing.get(origin, []),
            incoming.get(destination, []),
        )
        if activity_links is None:
            continue
        origin_link, destination_link = activity_links
        agents.append((origin_link.key, destination_link.key, float(7 * 3600 + (index * 53) % 7200)))
    return agents


def _weighted_link_graph(links: list[LinkRecord]) -> nx.DiGraph:
    """Build the deterministic least-free-flow-time graph used for probe design."""

    graph = nx.DiGraph()
    for link in sorted(links, key=lambda item: item.key):
        weight = max(0.001, link.length_m / max(link.free_speed_mps, 0.001))
        current = graph.get_edge_data(link.from_node, link.to_node)
        if current is None or (weight, link.key) < (float(current["weight"]), str(current["link_key"])):
            graph.add_edge(
                link.from_node,
                link.to_node,
                weight=weight,
                link_key=link.key,
            )
    return graph


def _path_activity_pair(
    graph: nx.DiGraph,
    origin_node: str,
    destination_node: str,
    *,
    required_link: str | None = None,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Return activity links whose routed interior optionally crosses a selected link."""

    if origin_node == destination_node:
        return None
    try:
        nodes = nx.shortest_path(graph, origin_node, destination_node, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    link_keys = tuple(str(graph[first][second]["link_key"]) for first, second in zip(nodes, nodes[1:]))
    if len(link_keys) < 3:
        return None
    # MATSim activities sit on the first and final links. Requiring the
    # disrupted road in the routed interior makes the baseline exposure real
    # rather than merely placing an activity on the selected road.
    if required_link and required_link not in link_keys[1:-1]:
        return None
    if link_keys[0] == link_keys[-1]:
        return None
    return link_keys[0], link_keys[-1], link_keys


def _evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    indexes = {
        round(index * (len(items) - 1) / max(1, count - 1))
        for index in range(count)
    }
    return [items[index] for index in sorted(indexes)]


def _impact_probe_pairs(
    links: list[LinkRecord],
    changes: dict[str, dict[str, Any]],
    maximum: int,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Find baseline OD probes whose least-time route crosses a changed road."""

    graph = _weighted_link_graph(links)
    by_key = {link.key: link for link in links}
    changed = [by_key[key] for key in sorted(changes) if key in by_key]
    changed = _evenly_spaced(changed, min(len(changed), max(1, maximum)))
    if not changed or graph.number_of_edges() < 3:
        return [], set()

    reverse = graph.reverse(copy=False)
    per_change = max(1, math.ceil(maximum / len(changed)))
    candidates_by_change: list[list[tuple[str, str]]] = []
    nearby_keys: set[str] = set(changes)
    seen: set[tuple[str, str]] = set()

    for changed_link in changed:
        upstream_distances = nx.single_source_shortest_path_length(
            reverse, changed_link.from_node, cutoff=8
        )
        downstream_distances = nx.single_source_shortest_path_length(
            graph, changed_link.to_node, cutoff=8
        )
        upstream = [
            node for node, distance in sorted(
                upstream_distances.items(), key=lambda item: (-item[1], item[0])
            ) if distance >= 1
        ][:24]
        downstream = [
            node for node, distance in sorted(
                downstream_distances.items(), key=lambda item: (-item[1], item[0])
            ) if distance >= 1
        ][:24]
        for first, second, data in graph.in_edges(changed_link.from_node, data=True):
            nearby_keys.add(str(data["link_key"]))
        for first, second, data in graph.out_edges(changed_link.to_node, data=True):
            nearby_keys.add(str(data["link_key"]))

        local: list[tuple[str, str]] = []
        attempt_limit = max(16, per_change * 16)
        attempts = 0
        for offset in range(max(len(upstream), len(downstream), 1)):
            for index, origin in enumerate(upstream):
                if not downstream:
                    break
                destination = downstream[(index + offset) % len(downstream)]
                attempts += 1
                pair = _path_activity_pair(
                    graph,
                    origin,
                    destination,
                    required_link=changed_link.key,
                )
                if pair is not None:
                    activity_pair = (pair[0], pair[1])
                    if activity_pair not in seen:
                        seen.add(activity_pair)
                        local.append(activity_pair)
                if len(local) >= per_change + 2 or attempts >= attempt_limit:
                    break
            if len(local) >= per_change + 2 or attempts >= attempt_limit:
                break
        candidates_by_change.append(local)

    pairs: list[tuple[str, str]] = []
    for rank in range(max((len(values) for values in candidates_by_change), default=0)):
        for values in candidates_by_change:
            if rank < len(values):
                pairs.append(values[rank])
                if len(pairs) >= maximum:
                    return pairs, nearby_keys
    return pairs, nearby_keys


def _pair_route_keys(
    graph: nx.DiGraph,
    by_key: dict[str, LinkRecord],
    pair: tuple[str, str],
) -> tuple[str, ...]:
    origin = by_key.get(pair[0])
    destination = by_key.get(pair[1])
    if origin is None or destination is None:
        return ()
    try:
        nodes = nx.shortest_path(
            graph, origin.to_node, destination.from_node, weight="weight"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return ()
    return tuple(str(graph[first][second]["link_key"]) for first, second in zip(nodes, nodes[1:]))


def _impact_focused_probe_agents(
    links: list[LinkRecord],
    changes: dict[str, dict[str, Any]],
    target: int,
) -> tuple[list[tuple[str, str, float]], dict[str, Any]]:
    """Create reproducible probes concentrated on, near and away from the disruption."""

    target = max(1, min(200, int(target)))
    impact_target = int(round(target * 0.70))
    nearby_target = int(round(target * 0.20))
    control_target = target - impact_target - nearby_target
    impact_pairs, nearby_keys = _impact_probe_pairs(links, changes, max(impact_target * 2, 12))

    graph = _weighted_link_graph(links)
    by_key = {link.key: link for link in links}
    global_agents = _probe_agents(links, min(240, max(target * 3, 80)))
    global_pairs = list(dict.fromkeys((origin, destination) for origin, destination, _ in global_agents))
    nearby_pairs: list[tuple[str, str]] = []
    control_pairs: list[tuple[str, str]] = []
    impact_set = set(impact_pairs)
    changed_keys = set(changes)
    for pair in global_pairs:
        if pair in impact_set:
            continue
        route_keys = set(_pair_route_keys(graph, by_key, pair))
        if route_keys & changed_keys:
            impact_pairs.append(pair)
            impact_set.add(pair)
        elif route_keys & nearby_keys:
            nearby_pairs.append(pair)
        else:
            control_pairs.append(pair)

    available = {
        "impact_focused": list(dict.fromkeys(impact_pairs)),
        "nearby": list(dict.fromkeys(nearby_pairs)),
        "network_control": list(dict.fromkeys(control_pairs)),
    }
    requested = {
        "impact_focused": impact_target,
        "nearby": nearby_target,
        "network_control": control_target,
    }
    if not any(available.values()):
        return [], {
            "requested_agents": target,
            "generated_agents": 0,
            "group_counts": {key: 0 for key in requested},
            "unique_od_pairs": 0,
            "design": "impact_focused_deterministic_probe",
        }

    # Missing strata are reassigned deterministically instead of inventing an
    # invalid OD pair. Impact-exposed probes receive first priority.
    allocations = {key: (requested[key] if available[key] else 0) for key in requested}
    unallocated = target - sum(allocations.values())
    for key in ("impact_focused", "nearby", "network_control"):
        if available[key] and unallocated:
            allocations[key] += unallocated
            unallocated = 0

    agents: list[tuple[str, str, float]] = []
    group_counts: dict[str, int] = {key: 0 for key in requested}
    used_pairs: set[tuple[str, str]] = set()
    person_index = 0
    for group in ("impact_focused", "nearby", "network_control"):
        pairs = available[group]
        for index in range(allocations[group]):
            origin, destination = pairs[index % len(pairs)]
            departure = float(7 * 3600 + (person_index * 37) % (2 * 3600))
            agents.append((origin, destination, departure))
            group_counts[group] += 1
            used_pairs.add((origin, destination))
            person_index += 1

    return agents, {
        "requested_agents": target,
        "generated_agents": len(agents),
        "group_counts": group_counts,
        "unique_od_pairs": len(used_pairs),
        "changed_roads_targeted": len(changes),
        "design": "impact_focused_deterministic_probe",
        "baseline_exposure_rule": "Impact probes have a baseline least-time route crossing a confirmed changed road.",
    }


def _format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _scenario_reachable_agents(
    agents: list[tuple[str, str, float]],
    baseline_links: list[LinkRecord],
    scenario_links: list[LinkRecord],
    changes: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, str, float]], list[int], list[dict[str, Any]]]:
    """Keep routeable agents and record closure-disconnected agents explicitly.

    MATSim's plan preparation aborts the entire run when a static closure
    leaves an activity pair without a path.  Screening that condition before
    launch is both faster and more transparent: reachable members of the same
    deterministic probe population run in MATSim, while the omitted IDs are
    retained as no-route outcomes with a mapped approach road.
    """

    scenario_graph = _weighted_link_graph(scenario_links)
    scenario_by_key = {link.key: link for link in scenario_links}
    baseline_graph = _weighted_link_graph(baseline_links)
    baseline_by_key = {link.key: link for link in baseline_links}
    reachable_agents: list[tuple[str, str, float]] = []
    reachable_indexes: list[int] = []
    disconnected: list[dict[str, Any]] = []
    reachability_cache: dict[tuple[str, str], bool] = {}
    closed_keys = {
        key for key, change in changes.items() if change.get("effect") == "closed"
    }

    for index, agent in enumerate(agents):
        origin_key, destination_key, departure = agent
        origin = scenario_by_key.get(origin_key)
        destination = scenario_by_key.get(destination_key)
        pair = (origin_key, destination_key)
        routeable = reachability_cache.get(pair)
        if routeable is None:
            routeable = False
            if origin is not None and destination is not None:
                if origin.to_node == destination.from_node:
                    routeable = True
                else:
                    try:
                        routeable = nx.has_path(
                            scenario_graph, origin.to_node, destination.from_node
                        )
                    except nx.NodeNotFound:
                        routeable = False
            reachability_cache[pair] = routeable
        if routeable:
            reachable_agents.append(agent)
            reachable_indexes.append(index)
            continue

        baseline_route = [origin_key]
        baseline_route.extend(_pair_route_keys(baseline_graph, baseline_by_key, pair))
        baseline_route.append(destination_key)
        first_closed = next((key for key in baseline_route if key in closed_keys), None)
        approach_key = origin_key
        if first_closed is not None:
            closed_index = baseline_route.index(first_closed)
            if closed_index > 0:
                approach_key = baseline_route[closed_index - 1]
        disconnected.append({
            "person_id": f"twm-{index:06d}",
            "person_index": index,
            "origin_link_key": origin_key,
            "destination_link_key": destination_key,
            "departure_time": float(departure),
            "approach_link_key": approach_key,
            "closed_link_key": first_closed,
            "reason": "no_path_after_static_closure",
        })
    return reachable_agents, reachable_indexes, disconnected


def _write_population(
    path: Path,
    agents: list[tuple[str, str, float]],
    *,
    person_indexes: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indexes = person_indexes if person_indexes is not None else list(range(len(agents)))
    if len(indexes) != len(agents):
        raise ValueError("MATSim person indexes must match the exported agent population.")
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        stream.write(f'<!DOCTYPE population SYSTEM {quoteattr(_matsim_dtd_uri("population_v6.dtd"))}>\n')
        stream.write('<population>\n')
        for index, (origin, destination, departure) in zip(indexes, agents):
            stream.write(f'  <person id="twm-{index:06d}">\n    <plan selected="yes">\n')
            stream.write(f'      <activity type="home" link={quoteattr(_matsim_id(origin, "l"))} end_time={quoteattr(_format_time(departure))} />\n')
            stream.write('      <leg mode="car" />\n')
            stream.write(f'      <activity type="work" link={quoteattr(_matsim_id(destination, "l"))} />\n')
            stream.write('    </plan>\n  </person>\n')
        stream.write('</population>\n')


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matsim_id(value: str, prefix: str) -> str:
    """Encode platform-scoped IDs as XML-safe, deterministic MATSim IDs."""
    digest = sha256(str(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _validated_matsim_changes(
    links: list[LinkRecord],
    raw_changes: Any,
) -> dict[str, dict[str, Any]]:
    """Validate scenario operators again at the process-isolation boundary."""

    changes, excluded = _partition_matsim_changes(links, raw_changes, allow_empty=True)
    if excluded:
        raise ValueError(
            "MATSim cannot ground the selected disruption in the routable network: "
            f"{excluded[0]['key']}."
        )
    return changes


def _partition_matsim_changes(
    links: list[LinkRecord],
    raw_changes: Any,
    *,
    allow_empty: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Validate operators and separate map entities absent from the MATSim supply.

    A drawn area can include a small number of mapped observations whose inferred
    line is degenerate or unresolved.  Those observations remain valid map data,
    but they cannot receive a MATSim network operator.  One such observation must
    not invalidate every otherwise routable road in the confirmed area.
    """

    if not isinstance(raw_changes, list) or not raw_changes:
        raise ValueError("MATSim needs at least one confirmed road disruption.")
    available = {link.key for link in links}
    changes: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for raw in raw_changes:
        if not isinstance(raw, dict):
            raise ValueError("Every MATSim disruption must be a confirmed road-link change.")
        key = str(raw.get("key") or "").strip()
        if not key:
            raise ValueError("MATSim cannot ground a disruption with a missing road key.")
        if key in seen_keys:
            raise ValueError(f"MATSim received more than one disruption for {key}; keep one final operator per road.")
        seen_keys.add(key)
        effect = str(raw.get("effect") or "").strip()
        magnitude = 1.0 if effect == "closed" else float(raw.get("magnitude") or 0.0)
        if effect == "speed_reduction" and not 0.01 <= magnitude <= 0.95:
            raise ValueError("MATSim speed reduction must be between 1% and 95%.")
        if effect == "travel_time_increase" and not 0.01 <= magnitude <= 5.0:
            raise ValueError("MATSim travel-time increase must be between 1% and 500%.")
        if effect not in {"closed", "speed_reduction", "travel_time_increase"}:
            raise ValueError(f"MATSim does not support the disruption operator for {key}.")
        if key not in available:
            excluded.append({
                "key": key,
                "reason": "The mapped road is absent from the routable MATSim network.",
            })
            continue
        changes[key] = {"key": key, "effect": effect, "magnitude": magnitude}
    if not changes and not allow_empty:
        raise ValueError(
            "None of the selected roads can be represented in the routable MATSim network. "
            "Choose an area containing at least one mapped road with non-degenerate network geometry."
        )
    return changes, excluded


def _prepare_job(
    project: dict[str, Any],
    scenario: dict[str, Any],
    job_root: Path,
    *,
    max_agents: int,
    seed: int,
) -> dict[str, Any]:
    database_path = Path(project["database_path"])
    readiness = assess_matsim_readiness(project)
    if readiness["tier"] == "unavailable":
        raise ValueError(readiness["explanation"])
    links, nodes, node_ids = _load_links(database_path)
    changes, excluded_changes = _partition_matsim_changes(links, scenario.get("changes"))
    scenario_links = [changed for link in links if (changed := _scenario_link(link, changes)) is not None]
    if len(scenario_links) < 2:
        raise ValueError("The selected closures leave too little routable network for MATSim.")

    # Population activity links must exist in both baseline and scenario inputs.
    scenario_keys = {link.key for link in scenario_links}
    common_links = [link for link in links if link.key in scenario_keys]
    probe_profile: dict[str, Any] | None = None
    with closing(_connect(database_path)) as connection:
        if readiness["tier"] == "demand_calibrated":
            agents, sample_factor, source_od_rows = _uploaded_agents(connection, common_links, node_ids, max_agents)
            if not agents:
                raise ValueError("OD rows were detected but could not be joined to routable uploaded nodes.")
        else:
            agents, probe_profile = _impact_focused_probe_agents(
                links,
                changes,
                min(max_agents, 200),
            )
            sample_factor = 1.0
            source_od_rows = 0
    if not agents:
        raise ValueError("No baseline-reachable probe trip could be constructed from this directed network.")

    inputs = job_root / "inputs"
    baseline_network = inputs / "baseline-network.xml.gz"
    scenario_network = inputs / "scenario-network.xml.gz"
    baseline_population = inputs / "baseline-population.xml.gz"
    scenario_population = inputs / "scenario-population.xml.gz"
    scenario_agents, scenario_person_indexes, disconnected_agents = _scenario_reachable_agents(
        agents,
        links,
        scenario_links,
        changes,
    )
    _write_network(baseline_network, nodes, links)
    _write_network(scenario_network, nodes, scenario_links)
    _write_population(baseline_population, agents)
    _write_population(
        scenario_population,
        scenario_agents,
        person_indexes=scenario_person_indexes,
    )

    manifest = {
        "schema_version": "1.0",
        "project_id": project["project_id"],
        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario["name"],
        "tier": readiness["tier"],
        "seed": seed,
        "agent_count": len(agents),
        "scenario_agent_count": len(scenario_agents),
        "pre_simulation_disconnected_count": len(disconnected_agents),
        "pre_simulation_disconnected_agents": disconnected_agents,
        "source_od_rows": source_od_rows,
        "sample_factor": sample_factor,
        "baseline_link_count": len(links),
        "scenario_link_count": len(scenario_links),
        "closed_link_count": sum(1 for item in changes.values() if item.get("effect") == "closed"),
        "selected_change_count": len(changes) + len(excluded_changes),
        "changed_link_count": len(changes),
        "excluded_change_count": len(excluded_changes),
        "excluded_changes": excluded_changes,
        "probe_profile": probe_profile,
        "source_files": [
            {"name": item.get("name"), "sha256": item.get("sha256")}
            for item in project.get("summary", {}).get("source_files", [])
        ],
        "inputs": {
            "baseline_network_sha256": _file_digest(baseline_network),
            "scenario_network_sha256": _file_digest(scenario_network),
            "population_sha256": _file_digest(baseline_population),
            "baseline_population_sha256": _file_digest(baseline_population),
            "scenario_population_sha256": _file_digest(scenario_population),
        },
        "assumptions": (
            ["Uploaded OD demand is deterministically sampled; MATSim capacity factors use the same fraction."]
            if readiness["tier"] == "demand_calibrated" and sample_factor < 1.0
            else ["Uploaded OD demand and road supply are used without demand sampling."]
            if readiness["tier"] == "demand_calibrated"
            else [
                "Probe trips are deterministic and are not observed OD demand.",
                "Missing road capacities use 900 vehicles/hour solely to keep the queue simulation numerically defined.",
                "Interpret rerouting and disconnection only; do not interpret congestion or population impact.",
            ]
        ) + (
            [
                "Closed roads are removed from the disrupted MATSim network. Agents either reroute on the remaining "
                "network or are reported as unable to complete the trip when no feasible route remains."
            ]
            if any(item.get("effect") == "closed" for item in changes.values())
            else []
        ),
    }
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "readiness": readiness,
        "manifest": manifest,
        "links": links,
        "changes": changes,
        "baseline_network": baseline_network,
        "scenario_network": scenario_network,
        "population": baseline_population,
        "baseline_population": baseline_population,
        "scenario_population": scenario_population,
    }


def _run_worker(
    network: Path,
    population: Path,
    output: Path,
    *,
    iterations: int,
    seed: int,
    sample_factor: float,
    log_path: Path,
    cancelled: Callable[[], bool],
    timeout_seconds: int,
) -> None:
    java = shutil.which("java")
    if not java or not WORKER_JAR.exists():
        raise ValueError("The MATSim Java worker is not built on this installation.")
    command = [
        java, "-Xms512m", "-Xmx6g", "-jar", str(WORKER_JAR),
        "--network", str(network), "--plans", str(population), "--output", str(output),
        "--iterations", str(iterations), "--seed", str(seed),
        "--sample-factor", f"{max(0.000001, min(1.0, sample_factor)):.8f}",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=WORKER_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_lines: list[str] = []
    try:
        while process.poll() is None:
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("MATSim run cancelled.")
            if time.monotonic() - started > timeout_seconds:
                process.kill()
                raise TimeoutError(f"MATSim exceeded the {timeout_seconds}-second job limit.")
            line = process.stdout.readline() if process.stdout else ""
            if line:
                output_lines.append(line)
            else:
                time.sleep(0.15)
        if process.stdout:
            output_lines.extend(process.stdout.readlines())
    finally:
        if process.stdout:
            process.stdout.close()
    log_path.write_text("".join(output_lines), encoding="utf-8")
    if process.returncode != 0:
        tail = "".join(output_lines[-20:]).strip()
        raise RuntimeError(f"MATSim worker exited with code {process.returncode}. {tail[-1800:]}")


def _events_file(output: Path) -> Path:
    preferred = output / "output_events.xml.gz"
    if preferred.exists():
        return preferred
    candidates = sorted(output.rglob("*.events.xml.gz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError("MATSim completed without an events output.")
    return candidates[0]


def _parse_events(path: Path, *, bin_seconds: int = 300) -> dict[str, Any]:
    departures: dict[str, float] = {}
    link_entered: dict[tuple[str, str], float] = {}
    trip_times: list[float] = []
    link_times: dict[str, list[float]] = defaultdict(list)
    frames: dict[int, Counter[str]] = defaultdict(Counter)
    frame_vehicles: dict[int, list[tuple[str, str]]] = defaultdict(list)
    person_routes: dict[str, list[str]] = defaultdict(list)
    departed_people: set[str] = set()
    arrived_people: set[str] = set()
    stuck_people: set[str] = set()
    link_entry_counts: Counter[str] = Counter()
    arrivals = 0
    departures_count = 0
    stuck = 0
    link_entries = 0
    with gzip.open(path, "rb") as stream:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag != "event":
                element.clear()
                continue
            attributes = element.attrib
            event_type = str(attributes.get("type", "")).casefold()
            event_time = float(attributes.get("time", 0.0) or 0.0)
            person = str(attributes.get("person") or attributes.get("vehicle") or "")
            link = str(attributes.get("link") or "")
            if event_type == "departure":
                departures[person] = event_time
                if person:
                    departed_people.add(person)
                departures_count += 1
            elif event_type == "arrival":
                start = departures.pop(person, None)
                if start is not None and event_time >= start:
                    trip_times.append(event_time - start)
                arrivals += 1
                if person:
                    arrived_people.add(person)
            elif "entered link" in event_type or event_type == "linkenter":
                if person and link:
                    link_entered[(person, link)] = event_time
                    person_routes[person].append(link)
                    link_entry_counts[link] += 1
                    frame = int(event_time // bin_seconds) * bin_seconds
                    frames[frame][link] += 1
                    if len(frame_vehicles[frame]) < 400:
                        frame_vehicles[frame].append((person, link))
                    link_entries += 1
            elif "left link" in event_type or event_type == "linkleave":
                entered = link_entered.pop((person, link), None)
                if entered is not None and event_time >= entered:
                    link_times[link].append(event_time - entered)
            elif "stuck" in event_type or "abort" in event_type:
                stuck += 1
                if person:
                    stuck_people.add(person)
            element.clear()
    return {
        "departures": departures_count,
        "arrivals": arrivals,
        "stuck": stuck,
        "completed_trip_count": len(trip_times),
        "mean_trip_time_seconds": sum(trip_times) / len(trip_times) if trip_times else None,
        "median_trip_time_seconds": float(median(trip_times)) if trip_times else None,
        "link_entries": link_entries,
        "departed_people": sorted(departed_people),
        "arrived_people": sorted(arrived_people),
        "stuck_people": sorted(stuck_people),
        "person_routes": {key: value for key, value in person_routes.items()},
        "link_entry_counts": dict(link_entry_counts),
        "link_mean_times": {key: sum(values) / len(values) for key, values in link_times.items() if values},
        "frames": {str(key): dict(value) for key, value in frames.items()},
        "frame_vehicles": {str(key): value for key, value in frame_vehicles.items()},
    }


def _empty_event_parse() -> dict[str, Any]:
    """Return the event contract for a scenario with no routeable agents."""
    return {
        "departures": 0,
        "arrivals": 0,
        "stuck": 0,
        "completed_trip_count": 0,
        "mean_trip_time_seconds": None,
        "median_trip_time_seconds": None,
        "link_entries": 0,
        "departed_people": [],
        "arrived_people": [],
        "stuck_people": [],
        "person_routes": {},
        "link_entry_counts": {},
        "link_mean_times": {},
        "frames": {},
        "frame_vehicles": {},
    }


def _point_along(coordinates: tuple[tuple[float, float], ...], fraction: float) -> list[float]:
    if len(coordinates) < 2:
        return list(coordinates[0]) if coordinates else [0.0, 0.0]
    lengths = [math.dist(first, second) for first, second in zip(coordinates, coordinates[1:])]
    total = sum(lengths)
    if total <= 0:
        return list(coordinates[0])
    target = max(0.0, min(1.0, fraction)) * total
    traversed = 0.0
    for (first, second), length in zip(zip(coordinates, coordinates[1:]), lengths):
        if traversed + length >= target:
            local = (target - traversed) / max(length, 1e-12)
            return [first[0] + (second[0] - first[0]) * local, first[1] + (second[1] - first[1]) * local]
        traversed += length
    return list(coordinates[-1])


def _blocked_agent_features(
    disconnected_agents: list[dict[str, Any]],
    links: list[LinkRecord],
) -> list[dict[str, Any]]:
    """Map no-route agents to their last baseline approach road.

    These markers are explicitly labelled as network-screened outcomes rather
    than fabricated MATSim traffic events.
    """
    by_key = {link.key: link for link in links}
    features: list[dict[str, Any]] = []
    for item in disconnected_agents:
        approach_key = str(item.get("approach_link_key") or "")
        link = by_key.get(approach_key)
        if link is None:
            continue
        person_id = str(item.get("person_id") or "blocked-agent")
        digest = int(sha256(f"blocked|{person_id}|{approach_key}".encode()).hexdigest()[:8], 16)
        fraction = 0.70 + (digest % 2600) / 10000.0
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": _point_along(link.coordinates, fraction),
            },
            "properties": {
                "key": f"{person_id}@{approach_key}",
                "matsim_role": "vehicle",
                "link_key": approach_key,
                "impact_relation": "blocked",
                "agent_status": "No feasible route after closure",
                "evidence_kind": "pre_simulation_network_screen",
                "closed_link_key": item.get("closed_link_key"),
            },
        })
    return features


def _impact_zone_keys(
    links: list[LinkRecord],
    changes: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    changed_keys = set(changes)
    changed_links = [link for link in links if link.key in changed_keys]
    boundary_nodes = {
        node
        for link in changed_links
        for node in (link.from_node, link.to_node)
    }
    nearby_keys = {
        link.key
        for link in links
        if link.key not in changed_keys
        and (link.from_node in boundary_nodes or link.to_node in boundary_nodes)
    }
    return changed_keys, nearby_keys


def _animation_frames(
    parsed: dict[str, Any],
    links: list[LinkRecord],
    *,
    changes: dict[str, dict[str, Any]] | None = None,
    max_frames: int = 48,
    max_visible_agents: int = 240,
) -> list[dict[str, Any]]:
    by_key = {_matsim_id(link.key, "l"): link for link in links}
    changed_keys, nearby_keys = _impact_zone_keys(links, changes or {})
    raw_frame_times = sorted(int(value) for value in parsed.get("frames", {}))

    def frame_counts(frame_time: int) -> tuple[int, int, int, float]:
        counts = parsed.get("frames", {}).get(str(frame_time), {})
        changed_entries = 0
        nearby_entries = 0
        total_entries = 0
        for matsim_key, raw_count in counts.items():
            count = int(raw_count)
            total_entries += count
            link = by_key.get(matsim_key)
            if link and link.key in changed_keys:
                changed_entries += count
            elif link and link.key in nearby_keys:
                nearby_entries += count
        score = changed_entries * 4.0 + nearby_entries * 2.0 + total_entries
        return changed_entries, nearby_entries, total_entries, score

    peak_time = min(
        raw_frame_times,
        key=lambda value: (-frame_counts(value)[3], value),
        default=None,
    )
    frame_times = list(raw_frame_times)
    if len(frame_times) > max_frames:
        indexes = {round(index * (len(frame_times) - 1) / (max_frames - 1)) for index in range(max_frames)}
        if peak_time is not None:
            indexes.add(frame_times.index(peak_time))
        while len(indexes) > max_frames:
            removable = [index for index in sorted(indexes, reverse=True) if frame_times[index] != peak_time]
            indexes.remove(removable[0])
        frame_times = [frame_times[index] for index in sorted(indexes)]
    output = []
    for frame_time in frame_times:
        counts = parsed["frames"].get(str(frame_time), {})
        selected = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[:100]
        maximum_count = max((int(value) for _, value in selected), default=1)
        features: list[dict[str, Any]] = []
        for link_key, count in selected:
            link = by_key.get(link_key)
            if not link:
                continue
            relation = "changed" if link.key in changed_keys else "nearby" if link.key in nearby_keys else "spillover"
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(point) for point in link.coordinates]},
                "properties": {
                    "key": link.key,
                    "matsim_role": "link_load",
                    "vehicles": int(count),
                    "load_ratio": round(int(count) / max(1, maximum_count), 6),
                    "frame_time": frame_time,
                    "impact_relation": relation,
                },
            })
        vehicles = parsed.get("frame_vehicles", {}).get(str(frame_time), [])[:max_visible_agents]
        for index, (vehicle, link_key) in enumerate(vehicles):
            link = by_key.get(link_key)
            if not link:
                continue
            digest = int(sha256(f"{frame_time}|{vehicle}|{link_key}".encode()).hexdigest()[:8], 16)
            fraction = 0.08 + (digest % 8400) / 10000.0
            relation = "changed" if link.key in changed_keys else "nearby" if link.key in nearby_keys else "spillover"
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": _point_along(link.coordinates, fraction)},
                "properties": {
                    "key": f"{vehicle}@{link.key}",
                    "matsim_role": "vehicle",
                    "link_key": link.key,
                    "frame_time": frame_time,
                    "impact_relation": relation,
                },
            })
        changed_entries, nearby_entries, total_entries, impact_score = frame_counts(frame_time)
        output.append({
            "time_seconds": frame_time,
            "time_label": _format_time(frame_time),
            "vehicle_entries": total_entries,
            "changed_road_entries": changed_entries,
            "nearby_road_entries": nearby_entries,
            "spillover_road_entries": max(0, total_entries - changed_entries - nearby_entries),
            "impact_score": round(impact_score, 3),
            "is_peak_impact": frame_time == peak_time,
            "feature_collection": {"type": "FeatureCollection", "features": features},
        })
    return output


def _route_outcomes(
    baseline: dict[str, Any],
    scenario: dict[str, Any],
    expected_agents: int,
) -> dict[str, Any]:
    baseline_routes = {
        str(person): tuple(route)
        for person, route in baseline.get("person_routes", {}).items()
    }
    scenario_routes = {
        str(person): tuple(route)
        for person, route in scenario.get("person_routes", {}).items()
    }
    departed = set(scenario.get("departed_people", []))
    arrived = set(scenario.get("arrived_people", []))
    stuck = set(scenario.get("stuck_people", [])) - arrived
    rerouted = 0
    same_route = 0
    completed_unclassified = 0
    for person in sorted(arrived):
        baseline_route = baseline_routes.get(person, ())
        scenario_route = scenario_routes.get(person, ())
        if not baseline_route or not scenario_route:
            completed_unclassified += 1
        elif baseline_route == scenario_route:
            same_route += 1
        else:
            rerouted += 1
    unfinished = len(departed - arrived - stuck)
    unstarted = max(0, int(expected_agents) - len(departed))
    accounted = same_route + rerouted + completed_unclassified + len(stuck) + unfinished + unstarted
    if accounted < expected_agents:
        unfinished += expected_agents - accounted
    return {
        "same_route": same_route,
        "rerouted": rerouted,
        "completed_unclassified": completed_unclassified,
        "stuck_or_aborted": len(stuck),
        "unfinished": unfinished,
        "unstarted": unstarted,
        "agent_count": int(expected_agents),
        "rerouted_percent": round(rerouted / max(1, expected_agents) * 100.0, 3),
    }


def _top_link_shifts(
    baseline: dict[str, Any],
    scenario: dict[str, Any],
    links: list[LinkRecord],
    changes: dict[str, dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    by_matsim_key = {_matsim_id(link.key, "l"): link for link in links}
    changed_keys, nearby_keys = _impact_zone_keys(links, changes)
    baseline_counts = Counter({key: int(value) for key, value in baseline.get("link_entry_counts", {}).items()})
    scenario_counts = Counter({key: int(value) for key, value in scenario.get("link_entry_counts", {}).items()})
    rows: list[dict[str, Any]] = []
    for matsim_key in sorted(set(baseline_counts) | set(scenario_counts)):
        link = by_matsim_key.get(matsim_key)
        if link is None:
            continue
        before = int(baseline_counts[matsim_key])
        after = int(scenario_counts[matsim_key])
        additional = after - before
        if additional <= 0:
            continue
        relation = "changed" if link.key in changed_keys else "nearby" if link.key in nearby_keys else "spillover"
        rows.append({
            "key": link.key,
            "entity_id": link.entity_id,
            "label": f"Road {link.entity_id}",
            "relation": relation,
            "baseline_entries": before,
            "scenario_entries": after,
            "additional_entries": additional,
            "geometry": {
                "type": "LineString",
                "coordinates": [list(point) for point in link.coordinates],
            },
        })
    return sorted(
        rows,
        key=lambda item: (-item["additional_entries"], -item["scenario_entries"], item["key"]),
    )[:limit]


def _summary_metrics(parsed: dict[str, Any], *, expected_agents: int | None = None) -> dict[str, Any]:
    mean = parsed.get("mean_trip_time_seconds")
    median = parsed.get("median_trip_time_seconds")
    departures = int(parsed.get("departures") or 0)
    arrivals = int(parsed.get("arrivals") or 0)
    stuck = int(parsed.get("stuck") or 0)
    unstarted = max(0, int(expected_agents) - departures) if expected_agents is not None else None
    unfinished = max(0, departures - arrivals - stuck)
    return {
        "departures": departures,
        "arrivals": arrivals,
        "stuck": stuck,
        "unstarted": unstarted,
        "unfinished": unfinished,
        "completed_trips": int(parsed.get("completed_trip_count") or 0),
        "mean_trip_time_minutes": round(float(mean) / 60.0, 3) if mean is not None else None,
        "median_trip_time_minutes": round(float(median) / 60.0, 3) if median is not None else None,
        "link_entries": int(parsed.get("link_entries") or 0),
    }


def run_matsim_scenario(
    project: dict[str, Any],
    scenario: dict[str, Any],
    *,
    job_id: str,
    iterations: int = 3,
    max_agents: int = 4000,
    seed: int = DEFAULT_SEED,
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run paired baseline/scenario MATSim jobs and return map-ready evidence."""
    progress = progress or (lambda _value, _message: None)
    cancelled = cancelled or (lambda: False)
    started = time.perf_counter()
    job_root = MATSIM_STORAGE / scenario["scenario_id"] / job_id
    progress(5, "Checking MATSim data and runtime")
    prepared = _prepare_job(project, scenario, job_root, max_agents=max_agents, seed=seed)
    manifest = prepared["manifest"]
    progress(18, f"Prepared {manifest['agent_count']:,} identical baseline and scenario agents")

    baseline_output = job_root / "baseline-output"
    scenario_output = job_root / "scenario-output"
    progress(24, "Running MATSim baseline")
    _run_worker(
        prepared["baseline_network"], prepared["baseline_population"], baseline_output,
        iterations=iterations, seed=seed, sample_factor=manifest["sample_factor"],
        log_path=job_root / "baseline.log", cancelled=cancelled, timeout_seconds=timeout_seconds,
    )
    if manifest["scenario_agent_count"]:
        progress(56, "Running MATSim disruption for routeable members of the same agent group")
        _run_worker(
            prepared["scenario_network"], prepared["scenario_population"], scenario_output,
            iterations=iterations, seed=seed, sample_factor=manifest["sample_factor"],
            log_path=job_root / "scenario.log", cancelled=cancelled, timeout_seconds=timeout_seconds,
        )
    else:
        progress(76, "Every tested trip is disconnected by the confirmed closures")
    progress(84, "Reading MATSim events and building map playback")
    baseline_parsed = _parse_events(_events_file(baseline_output))
    scenario_parsed = (
        _parse_events(_events_file(scenario_output))
        if manifest["scenario_agent_count"]
        else _empty_event_parse()
    )
    baseline = _summary_metrics(baseline_parsed, expected_agents=manifest["agent_count"])
    disrupted = _summary_metrics(scenario_parsed, expected_agents=manifest["agent_count"])
    baseline_mean = baseline.get("mean_trip_time_minutes")
    scenario_mean = disrupted.get("mean_trip_time_minutes")
    change_percent = (
        (scenario_mean / baseline_mean - 1.0) * 100.0
        if baseline_mean not in (None, 0) and scenario_mean is not None else None
    )
    frames = _animation_frames(
        scenario_parsed,
        prepared["links"],
        changes=prepared["changes"],
    )
    blocked_features = _blocked_agent_features(
        manifest.get("pre_simulation_disconnected_agents", []),
        prepared["links"],
    )
    if blocked_features:
        if frames:
            target_index = next(
                (index for index, frame in enumerate(frames) if frame.get("is_peak_impact")),
                len(frames) - 1,
            )
            target_frame = frames[target_index]
            target_frame["feature_collection"]["features"].extend(blocked_features)
            target_frame["blocked_agent_count"] = len(blocked_features)
            target_frame["impact_score"] = round(
                float(target_frame.get("impact_score") or 0.0) + len(blocked_features) * 4.0,
                3,
            )
        else:
            departure_times = [
                float(item.get("departure_time") or 0.0)
                for item in manifest.get("pre_simulation_disconnected_agents", [])
            ]
            snapshot_time = int(median(departure_times)) if departure_times else 0
            frames = [{
                "time_seconds": snapshot_time,
                "time_label": _format_time(snapshot_time),
                "vehicle_entries": 0,
                "changed_road_entries": 0,
                "nearby_road_entries": 0,
                "spillover_road_entries": 0,
                "blocked_agent_count": len(blocked_features),
                "impact_score": float(len(blocked_features) * 4),
                "is_peak_impact": True,
                "feature_collection": {"type": "FeatureCollection", "features": blocked_features},
            }]
    peak_frame_index = next(
        (index for index, frame in enumerate(frames) if frame.get("is_peak_impact")),
        max(0, len(frames) - 1),
    )
    peak_frame = frames[peak_frame_index] if frames else {}
    route_outcomes = _route_outcomes(
        baseline_parsed,
        scenario_parsed,
        manifest["agent_count"],
    )
    top_link_shifts = _top_link_shifts(
        baseline_parsed,
        scenario_parsed,
        prepared["links"],
        prepared["changes"],
    )
    result = {
        "status": "completed",
        "job_id": job_id,
        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario["name"],
        "engine": "MATSim",
        "engine_version": "2026.0-2025w31",
        "worker_version": "1.0.0",
        "tier": manifest["tier"],
        "tier_label": prepared["readiness"]["label"],
        "baseline": baseline,
        "scenario": disrupted,
        "mean_trip_time_change_percent": round(change_percent, 3) if change_percent is not None else None,
        "additional_stuck_agents": max(0, disrupted["stuck"] - baseline["stuck"]),
        "additional_unstarted_agents": max(0, disrupted["unstarted"] - baseline["unstarted"]),
        "additional_unfinished_agents": max(0, disrupted["unfinished"] - baseline["unfinished"]),
        "route_outcomes": route_outcomes,
        "top_link_shifts": top_link_shifts,
        "animation_frames": frames,
        "frame_count": len(frames),
        "peak_impact_frame_index": peak_frame_index,
        "peak_impact_snapshot": {
            "frame_index": peak_frame_index,
            "time_seconds": peak_frame.get("time_seconds"),
            "time_label": peak_frame.get("time_label"),
            "probe_entries": int(peak_frame.get("vehicle_entries") or 0),
            "changed_road_entries": int(peak_frame.get("changed_road_entries") or 0),
            "nearby_road_entries": int(peak_frame.get("nearby_road_entries") or 0),
            "spillover_road_entries": int(peak_frame.get("spillover_road_entries") or 0),
            "blocked_agents": int(peak_frame.get("blocked_agent_count") or 0),
            "label": "Peak-impact agent distribution",
        },
        "manifest": manifest,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "plain_finding": (
            (
                f"MATSim used {manifest['changed_link_count']:,} of "
                f"{manifest['selected_change_count']:,} selected road changes; "
                f"{manifest['excluded_change_count']:,} mapped road"
                f"{' was' if manifest['excluded_change_count'] == 1 else 's were'} excluded because "
                "they are not represented in the routable MATSim network. "
            )
            if manifest["excluded_change_count"]
            else ""
        ) + (
            f"MATSim tested {manifest['agent_count']:,} "
            f"{'uploaded-demand' if manifest['tier'] == 'demand_calibrated' else 'impact-focused probe'} agents. "
            f"The disruption run retained {manifest['scenario_agent_count']:,} routeable agents; "
            f"{disrupted['completed_trips']:,} trips completed, {disrupted['stuck']:,} became stuck or aborted, "
            f"{manifest['pre_simulation_disconnected_count']:,} had no feasible route after the confirmed closures "
            f"and {disrupted['unfinished']:,} remained unfinished in the disruption run."
        ),
        "scientific_boundary": prepared["readiness"]["scientific_boundary"],
        "verification": {
            "passed": bool(
                manifest["agent_count"] > 0
                and baseline["departures"] <= manifest["agent_count"]
                and disrupted["departures"] <= manifest["agent_count"]
                and baseline["completed_trips"] <= baseline["departures"]
                and disrupted["completed_trips"] <= disrupted["departures"]
                and manifest["scenario_agent_count"]
                + manifest["pre_simulation_disconnected_count"]
                == manifest["agent_count"]
                and manifest["inputs"]["baseline_network_sha256"]
                != manifest["inputs"]["scenario_network_sha256"]
            ),
            "checks": [
                "scenario agents retain their baseline person IDs; closure-disconnected IDs are recorded explicitly",
                "baseline and scenario use the same deterministic seed",
                "scenario changes are applied only to confirmed uploaded link keys",
                "standard MATSim events were parsed from the final iteration",
                "route-change outcomes are compared person by person",
                "peak-impact playback frame is selected from event-derived road entries",
                "mapped roads absent from the routable MATSim supply are recorded rather than silently simulated",
                "tier and sampling assumptions are stored in the job manifest",
            ],
            "population_sha256": manifest["inputs"]["population_sha256"],
            "scenario_population_sha256": manifest["inputs"]["scenario_population_sha256"],
            "seed": seed,
        },
        "output_directory": str(job_root),
    }
    (job_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress(100, "MATSim comparison and playback ready")
    return result
