from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

import networkx as nx


PROFILE_ROW_LIMIT = 100_000
EXACT_GEOMETRY_TABLE = "transport_link_geometries"
NODE_TABLE = "transport_nodes"
OD_TABLE = "transport_od"
SCENARIO_TABLE = "transport_scenario_effects"


def _clean(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


ALIASES: dict[str, tuple[str, ...]] = {
    "record_type": ("recordtype", "featuretype", "objecttype", "rowtype", "tabletype", "datasettype"),
    "node_id": (
        "nodeid", "nodekey", "vertexid", "vertexkey", "junctionid", "intersectionid", "pointid",
        "nodecode", "noderef", "nodefid", "junctionref", "vertexref", "stopid", "stopcode",
        "stationid", "stationcode", "locationid", "siteid", "detectorid", "sensorid",
        "节点编号", "节点id", "站点编号", "检测器编号", "传感器编号", "node", "vertex",
    ),
    "link_id": (
        "linkid", "linkcode", "linkkey", "edgeid", "edgekey", "segmentid", "segmentuid", "roadid",
        "roadkey", "roadsegmentid", "roadsegment", "arcid", "sectionid", "sectionref", "wayid",
        "routeid", "streetid", "corridorid", "osmwayid", "osmid", "objectid", "fid",
        "路段编号", "道路编号", "链路编号", "边编号", "link", "edge",
    ),
    "from_node": (
        "fromnode", "fromnodeid", "fromid", "source", "sourceid", "sourcenode", "sourcenodeid",
        "tail", "tailnode", "startnode", "startnodeid", "startid", "beginnode", "originnode",
        "originnodeid", "fromstop", "fromstopid", "originstop", "originstopid", "startstop",
        "fromstation", "startstation", "beginpoint", "startpoint", "upstream", "upstreamnode",
        "origin", "start", "起点", "起始节点", "上游节点", "源节点", "anode", "inode", "from", "u",
    ),
    "to_node": (
        "tonode", "tonodeid", "toid", "target", "targetid", "targetnode", "targetnodeid",
        "head", "headnode", "endnode", "endnodeid", "endid", "destinationnode", "destinationnodeid",
        "tostop", "tostopid", "destinationstop", "destinationstopid", "endstop", "tostation",
        "endstation", "finishpoint", "endpoint", "downstream", "downstreamnode", "destination", "end",
        "终点", "终止节点", "下游节点", "目标节点", "bnode", "jnode", "to", "v",
    ),
    "longitude": ("longitude", "lon", "lng", "xcoord", "xcoordinate", "easting", "经度", "东坐标", "x"),
    "latitude": ("latitude", "lat", "ycoord", "ycoordinate", "northing", "纬度", "北坐标", "y"),
    "source_longitude": ("sourcelon", "sourcelng", "sourcelongitude", "sourcex", "fromlon", "fromlng", "fromlongitude", "fromeasting", "fromx", "startlon", "startlng", "startlongitude", "starteasting", "startx", "起点经度", "起点x", "xstart", "x1", "alon", "ax", "ulon", "ux"),
    "source_latitude": ("sourcelat", "sourcelatitude", "sourcey", "fromlat", "fromlatitude", "fromnorthing", "fromy", "startlat", "startlatitude", "startnorthing", "starty", "起点纬度", "起点y", "ystart", "y1", "alat", "ay", "ulat", "uy"),
    "target_longitude": ("targetlon", "targetlng", "targetlongitude", "targetx", "tolon", "tolng", "tolongitude", "toeasting", "tox", "endlon", "endlng", "endlongitude", "endeasting", "endx", "终点经度", "终点x", "xend", "x2", "blon", "bx", "vlon", "vx"),
    "target_latitude": ("targetlat", "targetlatitude", "targety", "tolat", "tolatitude", "tonorthing", "toy", "endlat", "endlatitude", "endnorthing", "endy", "终点纬度", "终点y", "yend", "y2", "blat", "by", "vlat", "vy"),
    "geometry": (
        "geometry", "geom", "thegeom", "wkt", "geomwkt", "geometrywkt", "shapewkt", "geojson",
        "linestring", "polyline", "coordinates", "coords", "shape", "linkgeometry", "roadgeometry",
        "nodegeometry", "pointgeometry", "几何", "道路几何", "路段几何",
    ),
    "region": ("regionid", "region", "zoneid", "zone", "areaid", "area", "county", "district"),
    "timestamp": ("datetime", "timestamp", "observationtime", "recordedat", "recordtime", "eventtime", "sampletime", "日期时间", "观测时间", "时间", "date"),
    "time_step": ("timestep", "simulationstep", "simstep", "timeindex", "stepindex"),
    "speed": ("speedkmh", "speedkph", "averagespeed", "avgspeed", "meanspeed", "meanvelocity", "velocity", "linkspeed", "speedmph", "speedmps", "平均速度", "行驶速度", "速度", "speed"),
    "travel_time": ("traveltimemin", "traveltimeseconds", "traveltime", "journeytime", "journeyduration", "linktime", "traversaltime", "旅行时间", "行程时间", "通行时间", "duration"),
    "length": ("lengthm", "lengthkm", "lengthmi", "linklength", "segmentlength", "roadlength", "distance", "路段长度", "道路长度", "长度", "length"),
    "flow": ("trafficflow", "linkflow", "observedflow", "trafficvolume", "vehiclecount", "vehiclesperhour", "vph", "交通流量", "交通量", "流量", "flow", "volume"),
    "capacity": ("linkcapacity", "roadcapacity", "segmentcapacity", "capacityvph", "hourlycapacity", "通行能力", "道路容量", "capacity"),
    "free_flow_time": ("freeflowtraveltime", "freeflowtime", "uncongestedtime", "freetraveltime", "fft", "t0"),
    "repair_cost": ("repaircost", "restorationcost", "recoverycost", "reconstructioncost", "costtorepair"),
    "repair_duration": ("repairduration", "repairtime", "restorationduration", "restorationtime", "recoveryduration", "timetorepair"),
    "failure_probability": ("failureprobability", "failureprob", "probabilityoffailure", "failurerisk", "fragilityprobability"),
    "primary_metric": (
        "floodexposure", "flooddepth", "waterdepth", "inundationdepth", "fireintensity", "burnseverity",
        "smokedensity", "windgust", "damage", "severity", "hazardvalue", "disruptionlevel", "closurestatus",
        "groundacceleration", "peakgroundacceleration", "pga", "failureprobability", "nodeweight", "metricvalue", "measurement",
        "waterlevel", "debrisdepth", "damageratio", "hazardscore", "riskscore", "impactscore", "roadcondition",
        "passability", "accessibility", "landslidesusceptibility", "liquefactionrisk", "bridgecondition",
    ),
    "origin": ("origin", "originid", "originnode", "fromzone", "ozone"),
    "destination": ("destination", "destinationid", "destinationnode", "tozone", "dzone"),
    "demand": ("oddemand", "tripdemand", "demand", "trips", "odflow"),
    "scenario_id": ("scenarioid", "scenarioname", "eventid", "eventname", "disruptionid", "hazardscenarioid"),
    "component_type": ("componenttype", "assettype", "entitytype"),
    "component_id": ("componentid", "assetid", "affectedid", "targetid", "linkid", "edgeid", "roadid", "segmentid"),
    "availability": ("availability", "available", "openratio", "operability", "isopen", "openstatus", "roadstatus", "closure", "closed"),
    "time_multiplier": ("timemultiplier", "traveltimemultiplier", "delayfactor", "timefactor", "speedreductionfactor"),
    "capacity_multiplier": ("capacitymultiplier", "capacityfactor", "remainingcapacityratio", "capacityreductionfactor"),
}

# Additional semantic families are kept in a clean extension block so the
# importer accepts common agency, GIS and multilingual headers without making
# any one spelling a required template.
_ALIAS_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "node_id": (
        "locationref", "siteref", "detectorref", "sensorref", "assetref", "assetid",
        "\u7ad9\u70b9\u7f16\u53f7", "\u68c0\u6d4b\u5668\u7f16\u53f7", "\u4f20\u611f\u5668\u7f16\u53f7",
    ),
    "longitude": (
        "centerlongitude", "centrelongitude", "centerlon", "centrelon",
        "\u4e2d\u5fc3\u7ecf\u5ea6", "\u7ecf\u5ea6", "\u4e1c\u5750\u6807",
    ),
    "latitude": (
        "centerlatitude", "centrelatitude", "centerlat", "centrelat",
        "\u4e2d\u5fc3\u7eac\u5ea6", "\u7eac\u5ea6", "\u5317\u5750\u6807",
    ),
    "region": ("regionname", "zonename", "\u533a\u57df\u540d\u79f0", "\u533a\u57df"),
    "timestamp": ("observedat", "\u89c2\u6d4b\u65f6\u95f4", "\u65e5\u671f\u65f6\u95f4"),
    "time_step": ("step", "\u65f6\u95f4\u6b65"),
    "primary_metric": ("\u6d2a\u6c34\u6df1\u5ea6", "\u707e\u5bb3\u5f3a\u5ea6", "\u635f\u4f24\u7a0b\u5ea6"),
}
for _role, _aliases in _ALIAS_EXTENSIONS.items():
    ALIASES[_role] = tuple(dict.fromkeys((*ALIASES[_role], *_aliases)))


RECORD_KIND_ALIASES: dict[str, set[str]] = {
    "node": {"node", "nodes", "vertex", "vertices", "junction", "junctions", "point", "points"},
    "link": {"link", "links", "edge", "edges", "road", "roads", "segment", "segments", "arc", "arcs"},
    "od": {"od", "oddemand", "trip", "trips", "demand", "odpair", "odpairs"},
    "scenario": {"scenario", "scenarios", "disruption", "disruptions", "event", "events", "hazard", "hazards"},
}


def _first_column(columns: Iterable[str], role: str) -> str | None:
    originals = [str(column) for column in columns]
    cleaned = {_clean(column): column for column in originals}
    for alias in ALIASES[role]:
        if alias in cleaned:
            return cleaned[alias]
    # Long aliases may safely appear inside unit-bearing headers such as
    # ``road_length_metres`` or ``average_speed_kmh``.
    for alias in ALIASES[role]:
        if len(alias) < 5:
            continue
        for normalised, original in cleaned.items():
            if role == "speed" and any(
                term in normalised
                for term in ("speedlimit", "postedspeed", "designspeed", "maximumspeed", "maxspeed")
            ):
                continue
            if alias in normalised:
                return original
    return None


def _has(row: dict[str, Any], column: str | None) -> bool:
    if not column:
        return False
    value = row.get(column)
    return value is not None and str(value).strip() not in {"", "nan", "None"}


def _value(row: dict[str, Any], column: str | None, default: Any = None) -> Any:
    return row.get(column, default) if column else default


def _number(value: Any) -> float | None:
    try:
        numeric = float(str(value).strip().replace("−", "-"))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record_kind(value: Any) -> str | None:
    cleaned = _clean(value)
    for kind, aliases in RECORD_KIND_ALIASES.items():
        if cleaned in aliases:
            return kind
    return None


def _geometry_record_kind(value: Any) -> str | None:
    """Return the transport entity implied by a geometry cell, if clear."""
    if value is None or not str(value).strip():
        return None
    text = re.sub(r"^SRID=\d+;", "", str(value).strip(), flags=re.IGNORECASE).lstrip()
    upper = text.upper()
    if upper.startswith(("LINESTRING", "MULTILINESTRING")):
        return "link"
    if upper.startswith(("POINT", "MULTIPOINT")):
        return "node"
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and parsed.get("type") == "Feature":
            parsed = parsed.get("geometry") or {}
        geometry_type = str(parsed.get("type") or "").casefold() if isinstance(parsed, dict) else ""
        if geometry_type in {"linestring", "multilinestring"}:
            return "link"
        if geometry_type in {"point", "multipoint"}:
            return "node"
        if isinstance(parsed, dict) and parsed.get("paths"):
            return "link"
        if isinstance(parsed, dict) and _number(parsed.get("x")) is not None and _number(parsed.get("y")) is not None:
            return "node"
        if isinstance(parsed, list) and parsed:
            if len(parsed) >= 2 and all(_number(item) is not None for item in parsed[:2]):
                return "node"
            if len(parsed) >= 2 and all(isinstance(item, list) and len(item) >= 2 for item in parsed):
                return "link"
    if len(re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)) >= 4 and any(separator in text for separator in (";", "|")):
        return "link"
    return None


def _classify_row(row: dict[str, Any], roles: dict[str, str | None]) -> str:
    explicit = _record_kind(_value(row, roles.get("record_type")))
    if explicit:
        return explicit
    geometry_kind = _geometry_record_kind(_value(row, roles.get("geometry")))
    if geometry_kind:
        return geometry_kind
    # Physical structure outranks optional OD or scenario attributes.  Road
    # inventories often carry demand, closure or recovery fields on the same
    # row; those attributes are extracted in addition to the Link later.
    endpoint_coordinates = all(
        _number(_value(row, roles.get(role))) is not None
        for role in ("source_longitude", "source_latitude", "target_longitude", "target_latitude")
    )
    if endpoint_coordinates:
        return "link"
    if (
        _has(row, roles.get("from_node")) and _has(row, roles.get("to_node"))
        and (
            _has(row, roles.get("link_id"))
            or any(_has(row, roles.get(role)) for role in (
                "length", "speed", "travel_time", "flow", "capacity", "free_flow_time", "primary_metric"
            ))
        )
    ):
        return "link"
    if _has(row, roles.get("component_id")) and any(
        _has(row, roles.get(role)) for role in ("scenario_id", "availability", "time_multiplier", "capacity_multiplier")
    ):
        return "scenario"
    if _has(row, roles.get("origin")) and _has(row, roles.get("destination")) and _has(row, roles.get("demand")):
        return "od"
    if _has(row, roles.get("from_node")) and _has(row, roles.get("to_node")):
        return "link"
    if _has(row, roles.get("node_id")) and (
        (_has(row, roles.get("longitude")) and _has(row, roles.get("latitude")))
        or not _has(row, roles.get("link_id"))
    ):
        return "node"
    if _number(_value(row, roles.get("longitude"))) is not None and _number(_value(row, roles.get("latitude"))) is not None:
        return "node"
    return "observation"


def _looks_datetime(values: list[str]) -> bool:
    if not values:
        return False
    hits = 0
    for value in values[:100]:
        text = str(value).strip()
        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}|T\d{2}:\d{2}", text):
            hits += 1
    return hits / min(len(values), 100) >= 0.7


def _fallback_numeric_metric(
    columns: list[str], sampled_rows: list[dict[str, Any]], roles: dict[str, str | None]
) -> str | None:
    """Choose one unambiguous numeric measure without guessing among peers."""
    used = {column for column in roles.values() if column}
    excluded_terms = (
        "id", "code", "name", "type", "class", "category", "latitude", "longitude", "lat", "lon", "lng",
        "easting", "northing", "geometry", "geom", "wkt", "node", "from", "to", "source", "target",
        "origin", "destination", "date", "timestamp", "timeindex", "timestep", "year",
    )
    signal_terms = (
        "hazard", "damage", "severity", "risk", "exposure", "impact", "condition", "loss", "disruption",
        "reliability", "access", "passab", "delay", "closure", "intensity", "probability", "score", "index",
        "depth", "level", "serviceability", "operability",
    )
    candidates: list[tuple[int, float, str]] = []
    for column in columns:
        if column in used:
            continue
        name = _clean(column)
        if not name or any(term in name for term in excluded_terms):
            continue
        values = [row.get(column) for row in sampled_rows[:500] if _has(row, column)]
        if not values:
            continue
        numeric_ratio = sum(_number(value) is not None for value in values) / len(values)
        if numeric_ratio < 0.8:
            continue
        candidates.append((int(any(term in name for term in signal_terms)), numeric_ratio, column))
    signalled = [item for item in candidates if item[0]]
    if len(signalled) == 1:
        return signalled[0][2]
    if not signalled and len(candidates) == 1:
        return candidates[0][2]
    return None


def _matrix_shape(columns: list[str], sampled_rows: list[dict[str, Any]]) -> tuple[str, list[str], float, float] | None:
    if len(columns) < 3 or not sampled_rows:
        return None
    origin_column = columns[0]
    destination_columns = columns[1:]
    tested = [
        row.get(column)
        for row in sampled_rows[:100]
        for column in destination_columns[:100]
        if str(row.get(column, "")).strip()
    ]
    if not tested:
        return None
    numeric_ratio = sum(_number(value) is not None for value in tested) / len(tested)
    origin_values = {_clean(row.get(origin_column)) for row in sampled_rows[:500] if _has(row, origin_column)}
    destination_values = {_clean(column) for column in destination_columns}
    header_overlap = len(origin_values & destination_values) / max(1, min(len(origin_values), len(destination_values)))
    return origin_column, destination_columns, numeric_ratio, header_overlap


def _network_matrix_columns(path: Path, columns: list[str], sampled_rows: list[dict[str, Any]]) -> tuple[str, list[str]] | None:
    shape = _matrix_shape(columns, sampled_rows)
    filename = _clean(path.stem)
    if not shape or not any(term in filename for term in ("adjacency", "connectivitymatrix", "networkmatrix", "edgematrix", "linkmatrix")):
        return None
    origin_column, destination_columns, numeric_ratio, header_overlap = shape
    return (origin_column, destination_columns) if numeric_ratio >= 0.8 and header_overlap >= 0.5 else None


def _od_matrix_columns(path: Path, columns: list[str], sampled_rows: list[dict[str, Any]]) -> tuple[str, list[str]] | None:
    if _network_matrix_columns(path, columns, sampled_rows):
        return None
    shape = _matrix_shape(columns, sampled_rows)
    if not shape:
        return None
    origin_column, destination_columns, numeric_ratio, header_overlap = shape
    cleaned_filename = _clean(path.stem)
    raw_filename = path.stem.casefold()
    filename_hint = bool(
        re.search(r"(^|[^a-z0-9])od([^a-z0-9]|$)", raw_filename)
        or any(term in cleaned_filename for term in ("odmatrix", "demandmatrix", "tripmatrix", "origindestination"))
    )
    origin_hint = _clean(origin_column) in {
        "origin", "originid", "originnode", "originzone", "fromzone", "sourcezone", "ozone", "taz",
    }
    if not (filename_hint or origin_hint or header_overlap >= 0.5):
        return None
    return (origin_column, destination_columns) if numeric_ratio >= 0.8 else None


def _roles_for_columns(columns: list[str], sampled_rows: list[dict[str, Any]], filename: str = "") -> dict[str, str | None]:
    roles = {role: _first_column(columns, role) for role in ALIASES}
    sampled_geometry_kinds = {
        kind for row in sampled_rows[:200]
        if (kind := _geometry_record_kind(_value(row, roles.get("geometry"))))
    }
    generic_type = next((column for column in columns if _clean(column) == "type"), None)
    if generic_type and not roles.get("record_type"):
        type_values = [_record_kind(row.get(generic_type)) for row in sampled_rows if _has(row, generic_type)]
        recognised = [value for value in type_values if value]
        if recognised and len(recognised) / max(1, len(type_values)) >= 0.7:
            roles["record_type"] = generic_type
    generic_id = next((column for column in columns if _clean(column) in {"id", "identifier", "objectid", "fid"}), None)
    if generic_id:
        filename_hint = _clean(Path(filename).stem)
        has_populated_topology = bool(
            roles.get("from_node")
            and roles.get("to_node")
            and any(
                _has(row, roles.get("from_node")) and _has(row, roles.get("to_node"))
                for row in sampled_rows[:200]
            )
        )
        has_link_shape = bool(has_populated_topology or "link" in sampled_geometry_kinds)
        has_node_shape = "node" in sampled_geometry_kinds
        has_record_types = bool(roles.get("record_type"))
        if not roles.get("link_id") and (has_link_shape or has_record_types or any(term in filename_hint for term in ("link", "edge", "road", "segment", "arc"))):
            roles["link_id"] = generic_id
        if not roles.get("node_id") and (has_record_types or has_node_shape or not has_link_shape or any(term in filename_hint for term in ("node", "vertex", "junction"))):
            roles["node_id"] = generic_id
    if "node" in sampled_geometry_kinds and not roles.get("node_id"):
        roles["node_id"] = generic_id or roles.get("link_id")
    if "link" in sampled_geometry_kinds and not roles.get("link_id"):
        roles["link_id"] = generic_id or roles.get("node_id")
    # A road/sensor identifier is also the correct stable point identifier
    # when the rows contain representative locations but no populated line
    # topology.  Reuse it instead of generating one identifier per row.
    has_point_coordinates = bool(roles.get("longitude") and roles.get("latitude"))
    has_populated_topology = bool(
        roles.get("from_node")
        and roles.get("to_node")
        and any(
            _has(row, roles.get("from_node")) and _has(row, roles.get("to_node"))
            for row in sampled_rows[:200]
        )
    )
    if has_point_coordinates and not has_populated_topology and not roles.get("node_id") and roles.get("link_id"):
        roles["node_id"] = roles["link_id"]
    # ``time`` is common in network tables and much more often means link
    # travel time than a timestamp.  Value patterns decide instead of a fixed
    # alias allocation.
    bare_time = next((column for column in columns if _clean(column) == "time"), None)
    if bare_time:
        values = [str(row.get(bare_time, "")).strip() for row in sampled_rows if _has(row, bare_time)]
        if _looks_datetime(values):
            roles["timestamp"] = roles.get("timestamp") or bare_time
        elif values and sum(_number(value) is not None for value in values[:200]) / min(len(values), 200) >= 0.8:
            roles["travel_time"] = roles.get("travel_time") or bare_time
    if not any(roles.get(role) for role in (
        "speed", "travel_time", "primary_metric", "flow", "capacity", "free_flow_time", "length",
        "repair_cost", "repair_duration", "failure_probability",
    )):
        roles["primary_metric"] = _fallback_numeric_metric(columns, sampled_rows, roles)
    return roles


def profile_transport_file(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    row_limit: int = PROFILE_ROW_LIMIT,
) -> dict[str, Any]:
    sampled_rows: list[dict[str, Any]] = []
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        for index, row in enumerate(reader):
            if index >= min(row_limit, 10_000):
                break
            sampled_rows.append(row)
    roles = _roles_for_columns(columns, sampled_rows, path.name)
    network_matrix = _network_matrix_columns(path, columns, sampled_rows)
    od_matrix = _od_matrix_columns(path, columns, sampled_rows)
    counts: Counter[str] = Counter()
    joint = Counter()
    sampled_map_points: list[list[float]] = []
    scanned = 0
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            if scanned >= row_limit:
                break
            scanned += 1
            if network_matrix:
                values = [_number(row.get(column)) for column in network_matrix[1]]
                counts["link"] += sum(value is not None and value != 0 for value in values)
                joint["link_topology"] += sum(value is not None and value != 0 for value in values)
                continue
            if od_matrix:
                counts["od"] += sum(_number(row.get(column)) is not None for column in od_matrix[1])
                continue
            raw_geometry = _value(row, roles.get("geometry"))
            if _has(row, roles.get("geometry")) and _geometry_record_kind(raw_geometry) is None:
                joint["unparsed_geometry_values"] += 1
            endpoint_present = [
                _has(row, roles.get(role))
                for role in ("source_longitude", "source_latitude", "target_longitude", "target_latitude")
            ]
            endpoint_numeric = [
                _number(_value(row, roles.get(role))) is not None
                for role in ("source_longitude", "source_latitude", "target_longitude", "target_latitude")
            ]
            if any(endpoint_present) and not all(endpoint_numeric):
                joint["incomplete_endpoint_rows"] += 1
            kind = _classify_row(row, roles)
            counts[kind] += 1
            if kind == "link":
                geometry_value = _value(row, roles.get("geometry"))
                parsed_geometry = parse_line_geometry(geometry_value)
                if parsed_geometry:
                    joint["link_geometry"] += 1
                    if len(sampled_map_points) < 5_000:
                        sampled_map_points.extend(_geometry_points(parsed_geometry)[:max(0, 5_000 - len(sampled_map_points))])
                elif _has(row, roles.get("geometry")):
                    joint["malformed_link_geometry"] += 1
                endpoint_values = [
                    _number(_value(row, roles.get(role)))
                    for role in ("source_longitude", "source_latitude", "target_longitude", "target_latitude")
                ]
                if all(value is not None for value in endpoint_values):
                    joint["link_endpoint_coordinates"] += 1
                    if len(sampled_map_points) < 5_000:
                        sampled_map_points.extend([
                            [float(endpoint_values[0]), float(endpoint_values[1])],
                            [float(endpoint_values[2]), float(endpoint_values[3])],
                        ])
                elif any(_has(row, roles.get(role)) for role in ("source_longitude", "source_latitude", "target_longitude", "target_latitude")):
                    joint["partial_link_endpoint_coordinates"] += 1
                if _has(row, roles.get("from_node")) and _has(row, roles.get("to_node")):
                    joint["link_topology"] += 1
                if _has(row, roles.get("longitude")) and _has(row, roles.get("latitude")):
                    joint["link_representative_coordinates"] += 1
                if not _has(row, roles.get("link_id")):
                    joint["generated_link_ids"] += 1
                if all(_has(row, roles.get(role)) for role in ("origin", "destination", "demand")):
                    joint["embedded_od"] += 1
                if any(_has(row, roles.get(role)) for role in ("availability", "time_multiplier", "capacity_multiplier")):
                    joint["embedded_scenario"] += 1
            if kind == "node":
                x = _number(_value(row, roles.get("longitude")))
                y = _number(_value(row, roles.get("latitude")))
                if x is not None and y is not None:
                    joint["located_nodes"] += 1
                    if len(sampled_map_points) < 5_000:
                        sampled_map_points.append([x, y])
                elif _geometry_record_kind(_value(row, roles.get("geometry"))) == "node":
                    joint["located_nodes"] += 1
                    joint["node_point_geometry"] += 1
                    if len(sampled_map_points) < 5_000:
                        point = parse_point_geometry(_value(row, roles.get("geometry")))
                        if point:
                            sampled_map_points.append(point)
                if not _has(row, roles.get("node_id")):
                    joint["generated_node_ids"] += 1

    kinds = {kind for kind, count in counts.items() if count}
    explicit_type = bool(roles.get("record_type")) and bool(kinds & {"node", "link", "od", "scenario"})
    direct_link = "link" in kinds and bool(joint["link_geometry"] or joint["link_endpoint_coordinates"])
    topology_only = "link" in kinds and not direct_link and bool(joint["link_topology"]) and (
        joint["link_representative_coordinates"] == 0 or explicit_type
    )
    mixed = len(kinds & {"node", "link", "od", "scenario"}) >= 2
    node_metric = any(roles.get(role) for role in (
        "speed", "travel_time", "primary_metric", "flow", "capacity", "length",
        "repair_cost", "repair_duration", "failure_probability",
    ))
    geometry_only_nodes = kinds == {"node"} and bool(joint["located_nodes"]) and not node_metric
    node_point_geometry = kinds == {"node"} and bool(joint["node_point_geometry"])
    coordinate_columns = [
        roles.get(role) or ""
        for role in ("longitude", "latitude", "source_longitude", "source_latitude", "target_longitude", "target_latitude", "geometry")
    ]
    coordinate_mode = _coordinate_transform(sampled_map_points, coordinate_columns)[1] if sampled_map_points else None
    node_coordinate_mode = coordinate_mode if joint["located_nodes"] else None
    projected_nodes = kinds == {"node"} and node_coordinate_mode in {"web_mercator_to_wgs84", "normalised_uploaded_geometry"}
    generated_nodes = kinds == {"node"} and bool(joint["generated_node_ids"] and joint["located_nodes"])
    adaptive = bool(direct_link or topology_only or mixed or explicit_type or geometry_only_nodes or node_point_geometry or projected_nodes or generated_nodes)
    auto_importable = bool(
        (counts["link"] and (joint["link_geometry"] or joint["link_endpoint_coordinates"] or joint["link_topology"]))
        or joint["located_nodes"]
    )
    if network_matrix:
        structure_type = "topology_matrix"
    elif od_matrix:
        structure_type = "od_matrix"
    elif mixed or explicit_type:
        structure_type = "mixed_transport_package"
    elif direct_link:
        structure_type = "direct_link_geometry"
    elif topology_only:
        structure_type = "topology_link_table"
    elif kinds == {"node"}:
        structure_type = "node_table"
    else:
        structure_type = "flat_observations"
    geography = (
        "uploaded_geometry" if joint["link_geometry"]
        else "endpoint_coordinates" if joint["link_endpoint_coordinates"]
        else "node_coordinates" if joint["located_nodes"]
        else "topology_layout" if topology_only else "representative_points"
    )
    has_measure = any(roles.get(role) for role in (
        "speed", "travel_time", "primary_metric", "flow", "capacity", "free_flow_time", "length",
        "repair_cost", "repair_duration", "failure_probability",
    ))
    steps: list[str] = []
    warnings: list[str] = []
    fatal_missing: list[str] = []
    if network_matrix:
        steps.append("Convert the adjacency/connectivity matrix into directed Links and a disclosed topology layout.")
    elif od_matrix:
        steps.append("Convert the wide OD matrix into origin-destination records.")
    elif joint["link_geometry"]:
        steps.append("Parse uploaded WKT, GeoJSON, coordinate-array or Esri line geometry into map Links.")
    elif joint["link_endpoint_coordinates"]:
        steps.append("Convert each pair of road endpoints into a LineString map Link.")
    elif joint["link_topology"] and joint["located_nodes"]:
        steps.append("Join Link From/To identifiers to uploaded Node coordinates.")
    elif topology_only:
        steps.append("Convert From/To connectivity into a deterministic non-geographic topology view.")
    elif joint["located_nodes"]:
        steps.append("Convert uploaded locations or Point geometry into the point map view.")
    if counts["link"] and not joint["link_topology"] and (joint["link_geometry"] or joint["link_endpoint_coordinates"]):
        steps.append("Derive stable topology endpoints from line coordinates when From/To fields are absent.")
    if joint["generated_link_ids"]:
        steps.append("Generate deterministic Link IDs because no source Link ID is available.")
    if joint["generated_node_ids"]:
        steps.append("Generate deterministic point IDs because no source asset ID is available.")
        warnings.append("Generated point IDs support this static map, but a stable uploaded ID is recommended for repeated observations across time.")
    if joint["embedded_od"]:
        steps.append("Keep embedded OD demand as additional records while preserving each road row as a Link.")
    if joint["embedded_scenario"]:
        steps.append("Keep embedded disruption fields as scenario effects while preserving each road row as a Link.")
    if joint["malformed_link_geometry"]:
        warnings.append(f"{joint['malformed_link_geometry']:,} geometry value(s) could not be parsed and will use valid endpoint or topology alternatives when available.")
    if joint["partial_link_endpoint_coordinates"]:
        warnings.append(f"{joint['partial_link_endpoint_coordinates']:,} Link row(s) have an incomplete endpoint-coordinate set.")
    if coordinate_mode == "normalised_uploaded_geometry":
        warnings.append("The coordinate reference system is not identifiable; connectivity will be shown in a clearly labelled non-geographic view.")
    if auto_importable and not has_measure:
        warnings.append("The network can be drawn, but condition analyses need speed, travel time, flow, capacity, length or a hazard/condition measure.")
    if not auto_importable:
        if joint["unparsed_geometry_values"]:
            fatal_missing.append("A geometry field was found, but it contains no readable Point, LineString, MultiLineString, Esri path or coordinate-pair geometry.")
        elif joint["incomplete_endpoint_rows"]:
            fatal_missing.append("Endpoint fields were found, but each Link needs two complete numeric coordinate pairs.")
        elif counts["od"] and not (counts["link"] or counts["node"]):
            fatal_missing.append("OD demand was detected, but a Node/Link network is needed for map visualisation.")
        elif counts["scenario"] and not (counts["link"] or counts["node"]):
            fatal_missing.append("Disruption records were detected, but a Node/Link network is needed for map visualisation.")
        else:
            fatal_missing.append("Provide one usable location pair, line geometry, four Link endpoint coordinates, or From/To connectivity.")
    target_view = (
        "geographic_links" if counts["link"] and coordinate_mode in {"wgs84", "swapped_wgs84", "web_mercator_to_wgs84"}
        else "non_geographic_links" if counts["link"]
        else "geographic_points" if joint["located_nodes"] and coordinate_mode in {"wgs84", "swapped_wgs84", "web_mercator_to_wgs84"}
        else "non_geographic_points" if joint["located_nodes"]
        else "not_visualisable"
    )
    return {
        "structure_type": structure_type,
        "adaptive": adaptive,
        "auto_importable": auto_importable,
        "preferred_view": "links" if counts["link"] else "points",
        "geography_mode": geography,
        "record_counts": dict(counts),
        "joint_counts": dict(joint),
        "roles": roles,
        "rows_profiled": scanned,
        "profile_sampled": scanned >= row_limit,
        "node_coordinate_mode": node_coordinate_mode,
        "coordinate_mode": coordinate_mode,
        "normalization": {
            "status": "ready" if auto_importable else "needs_key_data",
            "target_view": target_view,
            "steps": steps,
            "warnings": warnings,
            "fatal_missing": fatal_missing,
        },
        "od_matrix": {"origin_column": od_matrix[0], "destination_columns": od_matrix[1]} if od_matrix else None,
        "network_matrix": {"origin_column": network_matrix[0], "destination_columns": network_matrix[1]} if network_matrix else None,
        "explanation": (
            "A transport-network package was detected. Nodes, links, OD demand and scenario rows will be separated and joined automatically."
            if mixed or explicit_type
            else "Road-link records will open directly as a line view."
            if counts["link"]
            else "Point records will open as uploaded locations."
        ),
    }


def combine_transport_profiles(files: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [item.get("transport_structure") or {} for item in files]
    counts: Counter[str] = Counter()
    joint: Counter[str] = Counter()
    for profile in profiles:
        counts.update({key: int(value or 0) for key, value in (profile.get("record_counts") or {}).items()})
        joint.update({key: int(value or 0) for key, value in (profile.get("joint_counts") or {}).items()})
    has_links = counts["link"] > 0
    has_nodes = counts["node"] > 0
    adaptive = any(bool(profile.get("adaptive")) for profile in profiles) or (has_links and has_nodes) or (
        has_nodes and counts["od"] > 0
    )
    auto_importable = adaptive and (
        any(bool(profile.get("auto_importable")) for profile in profiles)
        or bool(has_links and joint["link_topology"] and joint["located_nodes"])
    )
    normalisation_steps = list(dict.fromkeys(
        step
        for profile in profiles
        for step in ((profile.get("normalization") or {}).get("steps") or [])
    ))
    normalisation_warnings = list(dict.fromkeys(
        warning
        for profile in profiles
        for warning in ((profile.get("normalization") or {}).get("warnings") or [])
    ))
    fatal_missing = [] if auto_importable else list(dict.fromkeys(
        issue
        for profile in profiles
        for issue in ((profile.get("normalization") or {}).get("fatal_missing") or [])
    ))
    return {
        "adaptive": adaptive,
        "auto_importable": auto_importable,
        "preferred_view": "links" if has_links else "points",
        "record_counts": dict(counts),
        "joint_counts": dict(joint),
        "structure_type": (
            "multi_file_transport_package" if len(files) > 1 and adaptive
            else next((profile.get("structure_type") for profile in profiles if profile.get("adaptive")), "flat_observations")
        ),
        "geography_mode": (
            "uploaded_geometry" if joint["link_geometry"]
            else "endpoint_coordinates" if joint["link_endpoint_coordinates"]
            else "node_coordinates" if joint["located_nodes"]
            else "topology_layout" if has_links else "representative_points"
        ),
        "normalization": {
            "status": "ready" if auto_importable else "needs_key_data",
            "target_view": "links" if has_links else "points" if has_nodes else "not_visualisable",
            "steps": normalisation_steps,
            "warnings": normalisation_warnings,
            "fatal_missing": fatal_missing,
        },
        "explanation": (
            "Network data detected. The importer will join compatible node, link, OD and scenario records; links open directly as lines."
            if auto_importable and has_links
            else "The standard point-observation importer will be used."
        ),
    }


def _parse_coordinate_sequence(text: str) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for pair in text.split(","):
        values = [part for part in pair.strip().split() if part]
        if len(values) < 2:
            continue
        first, second = _number(values[0]), _number(values[1])
        if first is not None and second is not None:
            coordinates.append([first, second])
    return coordinates


def parse_line_geometry(value: Any) -> dict[str, Any] | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and str(parsed.get("type") or "").casefold() == "feature":
            parsed = parsed.get("geometry") or {}
        if isinstance(parsed, dict):
            geometry_type = str(parsed.get("type") or "").casefold()
            if geometry_type in {"linestring", "multilinestring"}:
                canonical_type = "LineString" if geometry_type == "linestring" else "MultiLineString"
                return {"type": canonical_type, "coordinates": parsed.get("coordinates") or []}
            # ArcGIS/Esri JSON stores a polyline as one or more ``paths``.
            paths = parsed.get("paths")
            if isinstance(paths, list) and paths:
                valid_paths = [
                    [point[:2] for point in path if isinstance(point, list) and len(point) >= 2]
                    for path in paths if isinstance(path, list)
                ]
                valid_paths = [path for path in valid_paths if len(path) >= 2]
                if len(valid_paths) == 1:
                    return {"type": "LineString", "coordinates": valid_paths[0]}
                if valid_paths:
                    return {"type": "MultiLineString", "coordinates": valid_paths}
        if isinstance(parsed, list) and len(parsed) >= 2 and all(isinstance(point, list) and len(point) >= 2 for point in parsed):
            return {"type": "LineString", "coordinates": [point[:2] for point in parsed]}
        return None
    cleaned = re.sub(r"^SRID=\d+;", "", text, flags=re.IGNORECASE).strip()
    match = re.match(r"^LINESTRING\s*(?:ZM|Z|M)?\s*\((.*)\)$", cleaned, flags=re.IGNORECASE)
    if match:
        coordinates = _parse_coordinate_sequence(match.group(1))
        return {"type": "LineString", "coordinates": coordinates} if len(coordinates) >= 2 else None
    match = re.match(r"^MULTILINESTRING\s*(?:ZM|Z|M)?\s*\((.*)\)$", cleaned, flags=re.IGNORECASE)
    if match:
        parts = re.findall(r"\(([^()]*)\)", match.group(1))
        coordinates = [_parse_coordinate_sequence(part) for part in parts]
        coordinates = [part for part in coordinates if len(part) >= 2]
        return {"type": "MultiLineString", "coordinates": coordinates} if coordinates else None
    # Common delimited polyline cells: ``x y; x y`` or ``x,y|x,y``.
    if any(separator in cleaned for separator in (";", "|")):
        coordinates: list[list[float]] = []
        for pair in re.split(r"[;|]", cleaned):
            values = [item for item in re.split(r"[\s,]+", pair.strip()) if item]
            if len(values) < 2:
                continue
            x, y = _number(values[0]), _number(values[1])
            if x is not None and y is not None:
                coordinates.append([x, y])
        if len(coordinates) >= 2:
            return {"type": "LineString", "coordinates": coordinates}
    return None


def parse_point_geometry(value: Any) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and parsed.get("type") == "Feature":
            parsed = parsed.get("geometry") or {}
        if isinstance(parsed, dict) and str(parsed.get("type") or "").casefold() == "point":
            coordinates = parsed.get("coordinates") or []
        elif isinstance(parsed, dict) and _number(parsed.get("x")) is not None and _number(parsed.get("y")) is not None:
            coordinates = [parsed.get("x"), parsed.get("y")]
        elif isinstance(parsed, list):
            coordinates = parsed
        else:
            coordinates = []
        if len(coordinates) >= 2:
            x, y = _number(coordinates[0]), _number(coordinates[1])
            return [x, y] if x is not None and y is not None else None
        return None
    cleaned = re.sub(r"^SRID=\d+;", "", text, flags=re.IGNORECASE).strip()
    match = re.match(r"^POINT\s*(?:ZM|Z|M)?\s*\(([^()]*)\)$", cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    values = [part for part in re.split(r"[\s,]+", match.group(1).strip()) if part]
    if len(values) < 2:
        return None
    x, y = _number(values[0]), _number(values[1])
    return [x, y] if x is not None and y is not None else None


def _geometry_points(geometry: dict[str, Any] | None) -> list[list[float]]:
    if not geometry:
        return []
    if geometry.get("type") == "LineString":
        return [point[:2] for point in geometry.get("coordinates", []) if isinstance(point, list) and len(point) >= 2]
    if geometry.get("type") == "MultiLineString":
        return [point[:2] for line in geometry.get("coordinates", []) for point in line if isinstance(point, list) and len(point) >= 2]
    return []


def _transform_geometry(geometry: dict[str, Any], transform: Any) -> dict[str, Any]:
    if geometry["type"] == "LineString":
        return {"type": "LineString", "coordinates": [list(transform(*point[:2])) for point in geometry["coordinates"]]}
    return {
        "type": "MultiLineString",
        "coordinates": [[list(transform(*point[:2])) for point in line] for line in geometry["coordinates"]],
    }


def _coordinate_transform(points: list[list[float]], coordinate_columns: Iterable[str] = ()) -> tuple[Any, str, bool]:
    finite = [(float(point[0]), float(point[1])) for point in points if len(point) >= 2 and _number(point[0]) is not None and _number(point[1]) is not None]
    if finite and sum(-180 <= x <= 180 and -90 <= y <= 90 for x, y in finite) / len(finite) >= 0.95:
        return (lambda x, y: (float(x), float(y))), "wgs84", True
    if finite and sum(-90 <= x <= 90 and -180 <= y <= 180 for x, y in finite) / len(finite) >= 0.95:
        return (lambda x, y: (float(y), float(x))), "swapped_wgs84", True
    coordinate_hint = _clean(" ".join(str(column or "") for column in coordinate_columns))
    local_grid_hint = any(term in coordinate_hint for term in ("easting", "northing", "utm", "localx", "localy", "gridx", "gridy"))
    web_mercator_hint = any(term in coordinate_hint for term in ("webmercator", "epsg3857", "3857"))
    projected_ratio = sum(abs(x) <= 20_100_000 and abs(y) <= 20_100_000 and (abs(x) > 180 or abs(y) > 90) for x, y in finite) / len(finite) if finite else 0.0
    strong_web_mercator_range = bool(finite and max(max(abs(x), abs(y)) for x, y in finite) >= 10_000_000)
    if finite and projected_ratio >= 0.95 and not local_grid_hint and (web_mercator_hint or strong_web_mercator_range):
        def web_mercator(x: float, y: float) -> tuple[float, float]:
            longitude = float(x) / 20_037_508.34 * 180.0
            latitude = math.degrees(2.0 * math.atan(math.exp(float(y) / 6_378_137.0)) - math.pi / 2.0)
            return max(-180.0, min(180.0, longitude)), max(-85.0, min(85.0, latitude))
        return web_mercator, "web_mercator_to_wgs84", True
    if finite:
        xs, ys = [point[0] for point in finite], [point[1] for point in finite]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        span_x, span_y = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
        scale = min(24.0 / span_x, 12.0 / span_y)
        centre_x, centre_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        return (lambda x, y: ((float(x) - centre_x) * scale, (float(y) - centre_y) * scale)), "normalised_uploaded_geometry", False
    return (lambda x, y: (float(x), float(y))), "topology_layout", False


def _topology_positions(links: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    for link in links:
        origin, destination = str(link.get("from_node") or ""), str(link.get("to_node") or "")
        if origin and destination:
            graph.add_edge(origin, destination)
    if not graph:
        return {}
    if graph.number_of_nodes() <= 750:
        raw = nx.spring_layout(graph, seed=42, iterations=60, weight=None)
    else:
        ordered = graph.subgraph(sorted(graph.nodes())).copy()
        raw = nx.circular_layout(ordered)
    return {str(node): (float(value[0]) * 20.0, float(value[1]) * 10.0) for node, value in raw.items()}


def _midpoint(geometry: dict[str, Any]) -> tuple[float, float]:
    points = _geometry_points(geometry)
    middle = points[len(points) // 2]
    return float(middle[0]), float(middle[1])


def _coordinate_node_id(point: list[float]) -> str:
    """Create a stable internal endpoint ID without changing source data."""
    token = f"{float(point[0]):.6f},{float(point[1]):.6f}".encode("utf-8")
    return f"auto_node_{sha256(token).hexdigest()[:12]}"


def _geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    points = _geometry_points(geometry)
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _speed_kmh(value: Any, column: str | None) -> float | None:
    numeric = _number(value)
    if numeric is None:
        return None
    name = _clean(column)
    if "mph" in name:
        return numeric * 1.609344
    if "mps" in name or "ms" in name and "speed" in name:
        return numeric * 3.6
    return numeric


def _travel_minutes(value: Any, column: str | None) -> float | None:
    numeric = _number(value)
    if numeric is None:
        return None
    name = _clean(column)
    if "second" in name or name.endswith("sec"):
        return numeric / 60.0
    if "hour" in name or name.endswith("hr"):
        return numeric * 60.0
    return numeric


def _availability_number(value: Any, column: str | None) -> float | None:
    numeric = _number(value)
    name = _clean(column)
    if numeric is not None:
        if "closed" in name or name == "closure":
            return 0.0 if numeric > 0 else 1.0
        return numeric / 100.0 if numeric > 1.0 and numeric <= 100.0 else numeric
    text = _clean(value)
    if "closed" in name or name == "closure":
        if text in {"yes", "true", "closed", "blocked", "failed", "unavailable"}:
            return 0.0
        if text in {"no", "false", "open", "available", "operational", "passable"}:
            return 1.0
    if text in {"closed", "blocked", "failed", "unavailable", "no", "false", "yesclosed"}:
        return 0.0
    if text in {"open", "available", "operational", "passable", "yes", "true", "notclosed"}:
        return 1.0
    return None


def _length_metres(value: Any, column: str | None, *, speed: float | None, travel_time: float | None) -> float | None:
    numeric = _number(value)
    if numeric is None:
        return None
    name = _clean(column)
    if "km" in name:
        return numeric * 1000.0
    if "mile" in name or name.endswith("mi"):
        return numeric * 1609.344
    if name in {"length", "distance"} and speed and travel_time and numeric < 100:
        expected_minutes_if_km = numeric / speed * 60.0
        if abs(expected_minutes_if_km - travel_time) <= max(0.05, abs(travel_time) * 0.2):
            return numeric * 1000.0
    return numeric


@dataclass
class PreparedTransportPackage:
    rows: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    geometries: list[dict[str, Any]]
    od_rows: list[dict[str, Any]]
    scenario_rows: list[dict[str, Any]]
    source_files: list[dict[str, Any]]
    package_type: str
    geography_mode: str
    geographic: bool
    preferred_view: str
    warnings: list[str] = field(default_factory=list)

    @property
    def canonical_mapping(self) -> dict[str, str | None]:
        available = set(self.rows[0]) if self.rows else set()
        roles = {
            "entity_id": "entity_id", "from_node": "from_node", "to_node": "to_node", "length": "length_m",
            "latitude": "latitude", "longitude": "longitude", "time": "timestamp", "time_step": "time_step",
            "speed": "speed_kmh", "travel_time": "travel_time_min", "region": "region_id",
            "primary_metric": "primary_metric", "flow": "flow", "capacity": "capacity",
            "free_flow_time": "free_flow_time_min", "repair_cost": "repair_cost",
            "repair_duration": "repair_duration", "failure_probability": "failure_probability",
        }
        return {
            role: column
            if column in available and any(_has(row, column) for row in self.rows)
            else None
            for role, column in roles.items()
        }

    def write_canonical_csv(self, path: Path) -> None:
        fieldnames = [
            "entity_id", "from_node", "to_node", "length_m", "latitude", "longitude", "timestamp",
            "time_step", "speed_kmh", "travel_time_min", "primary_metric", "flow", "capacity", "free_flow_time_min",
            "repair_cost", "repair_duration", "failure_probability", "region_id",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{column: row.get(column, "") for column in fieldnames} for row in self.rows])


def prepare_transport_package(
    files: list[Path],
    *,
    previews: list[dict[str, Any]],
    entity_id_scope: str,
    confirmed_file_mappings: dict[str, dict[str, str | None]] | None = None,
) -> PreparedTransportPackage | None:
    combined = combine_transport_profiles(previews)
    if not combined.get("adaptive") or not combined.get("auto_importable"):
        return None

    raw_nodes: list[dict[str, Any]] = []
    raw_links: list[dict[str, Any]] = []
    od_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    all_raw_points: list[list[float]] = []

    for path, preview in zip(files, previews):
        profile = preview.get("transport_structure") or {}
        roles = dict(profile.get("roles") or {})
        confirmed = (confirmed_file_mappings or {}).get(path.name, {})
        bridge = {
            "from_node": "from_node", "to_node": "to_node", "length": "length",
            "latitude": "latitude", "longitude": "longitude", "time": "timestamp",
            "speed": "speed", "travel_time": "travel_time", "region": "region",
            "primary_metric": "primary_metric", "flow": "flow", "capacity": "capacity",
            "free_flow_time": "free_flow_time", "repair_cost": "repair_cost",
            "repair_duration": "repair_duration", "failure_probability": "failure_probability",
            "time_step": "time_step",
        }
        for standard_role, transport_role in bridge.items():
            if confirmed.get(standard_role):
                roles[transport_role] = confirmed[standard_role]
        if confirmed.get("entity_id"):
            link_records = int((profile.get("record_counts") or {}).get("link") or 0)
            node_records = int((profile.get("record_counts") or {}).get("node") or 0)
            if link_records > 0 and node_records == 0:
                roles["link_id"] = confirmed["entity_id"]
            if node_records > 0 and link_records == 0:
                roles["node_id"] = confirmed["entity_id"]
        encoding, delimiter = preview["encoding"], preview["delimiter"]
        source_files.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_hash(path),
            "encoding": encoding,
            "delimiter": delimiter,
            "detected_structure": profile.get("structure_type"),
            "detected_roles": roles,
        })
        with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            network_matrix = profile.get("network_matrix") or {}
            if network_matrix:
                origin_column = network_matrix.get("origin_column")
                destination_columns = list(network_matrix.get("destination_columns") or [])
                for row_number, row in enumerate(reader, 2):
                    origin = str(row.get(origin_column, "") or "").strip()
                    region = "global" if entity_id_scope == "global" else path.stem
                    if not origin:
                        continue
                    for destination in destination_columns:
                        weight = _number(row.get(destination))
                        if weight is None or weight == 0:
                            continue
                        raw_links.append({
                            "entity_id": f"{origin}->{destination}",
                            "from_node": origin,
                            "to_node": str(destination).strip(),
                            "region_id": region,
                            "raw_geometry": None,
                            "source_x": None,
                            "source_y": None,
                            "target_x": None,
                            "target_y": None,
                            "speed": None,
                            "travel_time": None,
                            "length_raw": None,
                            "length_column": None,
                            "timestamp": "",
                            "primary_metric": weight,
                            "flow": None,
                            "capacity": None,
                            "free_flow_time": None,
                            "repair_cost": None,
                            "repair_duration": None,
                            "failure_probability": None,
                            "source_file": path.name,
                            "source_row": row_number,
                        })
                continue
            od_matrix = profile.get("od_matrix") or {}
            if od_matrix:
                origin_column = od_matrix.get("origin_column")
                destination_columns = list(od_matrix.get("destination_columns") or [])
                for row_number, row in enumerate(reader, 2):
                    origin = str(row.get(origin_column, "") or "").strip()
                    region = "global" if entity_id_scope == "global" else path.stem
                    if not origin:
                        continue
                    for destination in destination_columns:
                        demand = _number(row.get(destination))
                        if demand is None or demand == 0:
                            continue
                        od_rows.append({
                            "origin": origin,
                            "destination": str(destination).strip(),
                            "demand": demand,
                            "region_id": region,
                            "source_file": path.name,
                            "source_row": row_number,
                            "attributes": {"matrix_origin_column": origin_column, "matrix_destination_column": destination},
                        })
                continue
            for row_number, row in enumerate(reader, 2):
                kind = _classify_row(row, roles)
                region = "global" if entity_id_scope == "global" else str(_value(row, roles.get("region"), "") or path.stem).strip()
                if kind == "node":
                    x, y = _number(_value(row, roles.get("longitude"))), _number(_value(row, roles.get("latitude")))
                    if x is None or y is None:
                        point = parse_point_geometry(_value(row, roles.get("geometry")))
                        if point:
                            x, y = point
                    uploaded_node_id = str(_value(row, roles.get("node_id"), "") or "").strip()
                    generated_node_id = (
                        _coordinate_node_id([x, y])
                        if x is not None and y is not None
                        else f"{path.stem}_point_{row_number - 1}"
                    )
                    record = {
                        "node_id": uploaded_node_id or generated_node_id,
                        "region_id": region, "raw_x": x, "raw_y": y,
                        "timestamp": str(_value(row, roles.get("timestamp"), "") or "").strip(),
                        "time_step": _number(_value(row, roles.get("time_step"))),
                        "speed": _speed_kmh(_value(row, roles.get("speed")), roles.get("speed")),
                        "travel_time": _travel_minutes(_value(row, roles.get("travel_time")), roles.get("travel_time")),
                        "primary_metric": _number(_value(row, roles.get("primary_metric"))),
                        "flow": _number(_value(row, roles.get("flow"))),
                        "capacity": _number(_value(row, roles.get("capacity"))),
                        "free_flow_time": _travel_minutes(_value(row, roles.get("free_flow_time")), roles.get("free_flow_time")),
                        "repair_cost": _number(_value(row, roles.get("repair_cost"))),
                        "repair_duration": _number(_value(row, roles.get("repair_duration"))),
                        "failure_probability": _number(_value(row, roles.get("failure_probability"))),
                        "source_file": path.name, "source_row": row_number,
                    }
                    if x is not None and y is not None:
                        all_raw_points.append([x, y])
                    raw_nodes.append(record)
                elif kind == "link":
                    geometry = parse_line_geometry(_value(row, roles.get("geometry")))
                    all_raw_points.extend(_geometry_points(geometry))
                    source_x = _number(_value(row, roles.get("source_longitude")))
                    source_y = _number(_value(row, roles.get("source_latitude")))
                    target_x = _number(_value(row, roles.get("target_longitude")))
                    target_y = _number(_value(row, roles.get("target_latitude")))
                    if None not in (source_x, source_y, target_x, target_y):
                        all_raw_points.extend([[source_x, source_y], [target_x, target_y]])
                    link_id = str(_value(row, roles.get("link_id"), "") or f"{path.stem}_link_{row_number - 1}").strip()
                    raw_links.append({
                        "entity_id": link_id,
                        "from_node": str(_value(row, roles.get("from_node"), "") or "").strip(),
                        "to_node": str(_value(row, roles.get("to_node"), "") or "").strip(),
                        "region_id": region, "raw_geometry": geometry,
                        "source_x": source_x, "source_y": source_y, "target_x": target_x, "target_y": target_y,
                        "speed": _speed_kmh(_value(row, roles.get("speed")), roles.get("speed")),
                        "travel_time": _travel_minutes(_value(row, roles.get("travel_time")), roles.get("travel_time")),
                        "length_raw": _number(_value(row, roles.get("length"))), "length_column": roles.get("length"),
                        "timestamp": str(_value(row, roles.get("timestamp"), "") or "").strip(),
                        "time_step": _number(_value(row, roles.get("time_step"))),
                        "primary_metric": _number(_value(row, roles.get("primary_metric"))),
                        "flow": _number(_value(row, roles.get("flow"))),
                        "capacity": _number(_value(row, roles.get("capacity"))),
                        "free_flow_time": _travel_minutes(_value(row, roles.get("free_flow_time")), roles.get("free_flow_time")),
                        "repair_cost": _number(_value(row, roles.get("repair_cost"))),
                        "repair_duration": _number(_value(row, roles.get("repair_duration"))),
                        "failure_probability": _number(_value(row, roles.get("failure_probability"))),
                        "source_file": path.name, "source_row": row_number,
                    })
                    # A physical road row may also carry OD or disruption
                    # attributes.  Preserve both meanings instead of forcing
                    # the row into one mutually exclusive table.
                    if all(_has(row, roles.get(role)) for role in ("origin", "destination", "demand")):
                        od_rows.append({
                            "origin": str(_value(row, roles.get("origin"), "") or "").strip(),
                            "destination": str(_value(row, roles.get("destination"), "") or "").strip(),
                            "demand": _number(_value(row, roles.get("demand"))),
                            "region_id": region, "source_file": path.name, "source_row": row_number,
                            "attributes": {"embedded_in_link": link_id},
                        })
                    if any(_has(row, roles.get(role)) for role in ("availability", "time_multiplier", "capacity_multiplier")):
                        scenario_rows.append({
                            "scenario_id": str(_value(row, roles.get("scenario_id"), "") or path.stem).strip(),
                            "component_type": str(_value(row, roles.get("component_type"), "") or "link").strip(),
                            "component_id": str(_value(row, roles.get("component_id"), "") or link_id).strip(),
                            "availability": _availability_number(_value(row, roles.get("availability")), roles.get("availability")),
                            "time_multiplier": _number(_value(row, roles.get("time_multiplier"))),
                            "capacity_multiplier": _number(_value(row, roles.get("capacity_multiplier"))),
                            "region_id": region, "source_file": path.name, "source_row": row_number,
                            "attributes": {"embedded_in_link": link_id},
                        })
                elif kind == "od":
                    od_rows.append({
                        "origin": str(_value(row, roles.get("origin"), "") or "").strip(),
                        "destination": str(_value(row, roles.get("destination"), "") or "").strip(),
                        "demand": _number(_value(row, roles.get("demand"))),
                        "region_id": region, "source_file": path.name, "source_row": row_number,
                        "attributes": {key: value for key, value in row.items() if str(value or "").strip()},
                    })
                elif kind == "scenario":
                    scenario_rows.append({
                        "scenario_id": str(_value(row, roles.get("scenario_id"), "") or path.stem).strip(),
                        "component_type": str(_value(row, roles.get("component_type"), "") or "link").strip(),
                        "component_id": str(_value(row, roles.get("component_id"), "") or "").strip(),
                        "availability": _availability_number(_value(row, roles.get("availability")), roles.get("availability")),
                        "time_multiplier": _number(_value(row, roles.get("time_multiplier"))),
                        "capacity_multiplier": _number(_value(row, roles.get("capacity_multiplier"))),
                        "region_id": region, "source_file": path.name, "source_row": row_number,
                        "attributes": {key: value for key, value in row.items() if str(value or "").strip()},
                    })

    coordinate_columns = [
        str(column)
        for source_file in source_files
        for role, column in (source_file.get("detected_roles") or {}).items()
        if column and role in {
            "longitude", "latitude", "source_longitude", "source_latitude",
            "target_longitude", "target_latitude", "geometry",
        }
    ]
    transform, coordinate_mode, geographic = _coordinate_transform(all_raw_points, coordinate_columns)
    nodes: list[dict[str, Any]] = []
    node_by_region: dict[tuple[str, str], tuple[float, float]] = {}
    global_candidates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for node in raw_nodes:
        if node["raw_x"] is None or node["raw_y"] is None or not node["node_id"]:
            continue
        longitude, latitude = transform(node["raw_x"], node["raw_y"])
        item = {**node, "longitude": longitude, "latitude": latitude}
        nodes.append(item)
        node_by_region[(node["region_id"], node["node_id"])] = (longitude, latitude)
        global_candidates[node["node_id"]].append((longitude, latitude))
    global_nodes = {node_id: values[0] for node_id, values in global_candidates.items() if len(set(values)) == 1}

    geometries: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for link in raw_links:
        geometry = _transform_geometry(link["raw_geometry"], transform) if link["raw_geometry"] else None
        if geometry is None and None not in (link["source_x"], link["source_y"], link["target_x"], link["target_y"]):
            first = transform(link["source_x"], link["source_y"])
            second = transform(link["target_x"], link["target_y"])
            geometry = {"type": "LineString", "coordinates": [list(first), list(second)]}
        if geometry is None and link["from_node"] and link["to_node"]:
            first = node_by_region.get((link["region_id"], link["from_node"])) or global_nodes.get(link["from_node"])
            second = node_by_region.get((link["region_id"], link["to_node"])) or global_nodes.get(link["to_node"])
            if first and second:
                geometry = {"type": "LineString", "coordinates": [list(first), list(second)]}
        if geometry is None:
            unresolved.append(link)
            continue
        points = _geometry_points(geometry)
        if points:
            if not link["from_node"]:
                link["from_node"] = _coordinate_node_id(points[0])
            if not link["to_node"]:
                link["to_node"] = _coordinate_node_id(points[-1])
        geometries.append({**link, "geometry": geometry})

    # Materialise genuine or derived endpoints so map, routing and resilience
    # models share the same normalised topology contract.
    known_nodes = {(str(item["region_id"]), str(item["node_id"])) for item in nodes if item.get("node_id")}
    for item in geometries:
        points = _geometry_points(item["geometry"])
        if not points:
            continue
        for node_id, point in ((item.get("from_node"), points[0]), (item.get("to_node"), points[-1])):
            key = (str(item["region_id"]), str(node_id or ""))
            if not key[1] or key in known_nodes:
                continue
            nodes.append({
                "node_id": key[1], "region_id": key[0], "longitude": float(point[0]), "latitude": float(point[1]),
                "primary_metric": None, "source_file": item.get("source_file") or "derived endpoint",
                "source_row": item.get("source_row") or 0,
            })
            known_nodes.add(key)

    # A pure edge list still deserves a readable deterministic topology view.
    # It is explicitly labelled as non-geographic and never promoted to a
    # geographic hotspot analysis.
    if raw_links and not geometries:
        positions = _topology_positions(raw_links)
        for link in raw_links:
            first, second = positions.get(link["from_node"]), positions.get(link["to_node"])
            if first and second:
                geometries.append({**link, "geometry": {"type": "LineString", "coordinates": [list(first), list(second)]}})
        coordinate_mode, geographic = "topology_layout", False
        nodes = [
            {"node_id": node_id, "region_id": "global", "longitude": point[0], "latitude": point[1], "primary_metric": None,
             "source_file": "derived topology", "source_row": 0}
            for node_id, point in positions.items()
        ]
    elif unresolved:
        # Preserve unresolved topology in the database, but do not fabricate
        # geographic positions beside a real map.
        pass

    rows: list[dict[str, Any]] = []
    if geometries:
        geometry_lookup = {(item["region_id"], item["entity_id"], item["source_file"], item["source_row"]): item for item in geometries}
        for link in raw_links:
            item = geometry_lookup.get((link["region_id"], link["entity_id"], link["source_file"], link["source_row"]))
            if item is None:
                continue
            longitude, latitude = _midpoint(item["geometry"])
            length_m = _length_metres(link["length_raw"], link["length_column"], speed=link["speed"], travel_time=link["travel_time"])
            rows.append({
                "entity_id": link["entity_id"], "from_node": link["from_node"], "to_node": link["to_node"],
                "length_m": length_m, "latitude": latitude, "longitude": longitude, "timestamp": link["timestamp"],
                "time_step": link.get("time_step"),
                "speed_kmh": link["speed"], "travel_time_min": link["travel_time"], "primary_metric": link["primary_metric"],
                "flow": link["flow"], "capacity": link["capacity"], "free_flow_time_min": link["free_flow_time"],
                "repair_cost": link["repair_cost"], "repair_duration": link["repair_duration"],
                "failure_probability": link["failure_probability"], "region_id": link["region_id"],
            })
    else:
        for node in nodes:
            rows.append({
                "entity_id": node["node_id"], "from_node": "", "to_node": "", "length_m": None,
                "latitude": node["latitude"], "longitude": node["longitude"], "timestamp": node.get("timestamp", ""),
                "time_step": node.get("time_step"),
                "speed_kmh": node.get("speed"), "travel_time_min": node.get("travel_time"), "primary_metric": node.get("primary_metric"),
                "flow": node.get("flow"), "capacity": node.get("capacity"), "free_flow_time_min": node.get("free_flow_time"),
                "repair_cost": node.get("repair_cost"), "repair_duration": node.get("repair_duration"),
                "failure_probability": node.get("failure_probability"), "region_id": node["region_id"],
            })
    if not rows:
        raise ValueError(
            "The files contain transport records, but no node location, line geometry, endpoint coordinates or usable topology could be formed."
        )
    warnings = []
    if unresolved:
        warnings.append(f"{len(unresolved):,} links had neither usable geometry nor resolvable endpoint coordinates and were retained only as source records.")
    return PreparedTransportPackage(
        rows=rows, nodes=nodes, geometries=geometries, od_rows=od_rows, scenario_rows=scenario_rows,
        source_files=source_files, package_type=combined["structure_type"], geography_mode=coordinate_mode,
        geographic=geographic, preferred_view="links" if geometries else "points", warnings=warnings,
    )


def install_prepared_package(
    database_path: Path,
    package: PreparedTransportPackage,
    summary: dict[str, Any],
    *,
    topology_node_scope: str,
) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        unique_geometries = {
            (str(item["region_id"]), str(item["entity_id"])): item
            for item in package.geometries
        }
        connection.executescript(f"""
            CREATE TABLE IF NOT EXISTS {EXACT_GEOMETRY_TABLE} (
                link_key TEXT PRIMARY KEY,
                region_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                geometry_json TEXT NOT NULL,
                geometry_kind TEXT NOT NULL,
                min_longitude REAL NOT NULL,
                min_latitude REAL NOT NULL,
                max_longitude REAL NOT NULL,
                max_latitude REAL NOT NULL,
                source_file TEXT,
                source_row INTEGER,
                UNIQUE(region_id, entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS {NODE_TABLE} (
                region_id TEXT NOT NULL, node_id TEXT NOT NULL, longitude REAL, latitude REAL,
                source_file TEXT, source_row INTEGER, PRIMARY KEY(region_id, node_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS {OD_TABLE} (
                od_pk INTEGER PRIMARY KEY, region_id TEXT, origin TEXT, destination TEXT, demand REAL,
                source_file TEXT, source_row INTEGER, attributes_json TEXT
            );
            CREATE TABLE IF NOT EXISTS {SCENARIO_TABLE} (
                effect_pk INTEGER PRIMARY KEY, region_id TEXT, scenario_id TEXT, component_type TEXT, component_id TEXT,
                availability REAL, time_multiplier REAL, capacity_multiplier REAL,
                source_file TEXT, source_row INTEGER, attributes_json TEXT
            );
        """)
        geometry_kind = (
            "uploaded_geographic_line" if package.geographic and package.geography_mode in {"wgs84", "swapped_wgs84", "web_mercator_to_wgs84"}
            else "topology_layout_line" if package.geography_mode == "topology_layout"
            else "normalised_uploaded_line" if not package.geographic
            else "joined_network_line"
        )
        geometry_rows = []
        for item in unique_geometries.values():
            bounds = _geometry_bounds(item["geometry"])
            geometry_rows.append((
                f"{item['region_id']}::{item['entity_id']}", item["region_id"], item["entity_id"],
                json.dumps(item["geometry"], separators=(",", ":")), geometry_kind, *bounds,
                item.get("source_file"), item.get("source_row"),
            ))
        connection.executemany(
            f"INSERT OR REPLACE INTO {EXACT_GEOMETRY_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", geometry_rows,
        )
        connection.executemany(
            f"INSERT OR REPLACE INTO {NODE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            [(item["region_id"], item["node_id"], item.get("longitude"), item.get("latitude"), item.get("source_file"), item.get("source_row")) for item in package.nodes if item.get("node_id")],
        )
        connection.executemany(
            f"INSERT INTO {OD_TABLE}(region_id, origin, destination, demand, source_file, source_row, attributes_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(item["region_id"], item["origin"], item["destination"], item["demand"], item["source_file"], item["source_row"], json.dumps(item["attributes"])) for item in package.od_rows],
        )
        connection.executemany(
            f"""INSERT INTO {SCENARIO_TABLE}
                (region_id, scenario_id, component_type, component_id, availability, time_multiplier, capacity_multiplier, source_file, source_row, attributes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(item["region_id"], item["scenario_id"], item["component_type"], item["component_id"], item["availability"], item["time_multiplier"], item["capacity_multiplier"], item["source_file"], item["source_row"], json.dumps(item["attributes"])) for item in package.scenario_rows],
        )

        # Seed the existing topology tables from genuine endpoints.  Numerical
        # models keep their established graph contract while the map uses the
        # complete uploaded line geometry.
        topology_links = [item for item in unique_geometries.values() if item.get("from_node") and item.get("to_node")]
        if topology_links:
            global_nodes = topology_node_scope == "global"
            node_positions: dict[str, tuple[str, str | None, float, float]] = {}
            mapped_links = []
            for item in topology_links:
                points = _geometry_points(item["geometry"])
                first, last = points[0], points[-1]
                region_id = item["region_id"]
                from_key = item["from_node"] if global_nodes else f"{region_id}\x1f{item['from_node']}"
                to_key = item["to_node"] if global_nodes else f"{region_id}\x1f{item['to_node']}"
                node_positions.setdefault(from_key, (item["from_node"], None if global_nodes else region_id, first[0], first[1]))
                node_positions.setdefault(to_key, (item["to_node"], None if global_nodes else region_id, last[0], last[1]))
                mapped_links.append((f"{region_id}::{item['entity_id']}", region_id, item["entity_id"], from_key, to_key, 0.0))
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS inferred_network_nodes (
                    node_key TEXT PRIMARY KEY, node_id TEXT NOT NULL, region_id TEXT, longitude REAL NOT NULL,
                    latitude REAL NOT NULL, support_count INTEGER NOT NULL, median_spread_m REAL NOT NULL,
                    max_spread_m REAL NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS inferred_network_links (
                    link_key TEXT PRIMARY KEY, region_id TEXT NOT NULL, entity_id TEXT NOT NULL,
                    from_node_key TEXT NOT NULL, to_node_key TEXT NOT NULL, inferred_length_m REAL NOT NULL,
                    UNIQUE(region_id, entity_id)
                ) WITHOUT ROWID;
                DELETE FROM inferred_network_nodes;
                DELETE FROM inferred_network_links;
            """)
            connection.executemany(
                "INSERT INTO inferred_network_nodes VALUES (?, ?, ?, ?, ?, 1, 0, 0)",
                [(key, *values) for key, values in node_positions.items()],
            )
            connection.executemany("INSERT INTO inferred_network_links VALUES (?, ?, ?, ?, ?, ?)", mapped_links)
            mapping = {
                "available": True, "ready": True, "status": "ready", "reason": "Uploaded or joined road-link geometry is ready.",
                "method": "uploaded_or_joined_transport_geometry_v1", "algorithm_version": 1,
                "geometry_kind": geometry_kind, "source_link_count": len(unique_geometries),
                "mapped_link_count": len(unique_geometries), "topology_link_count": len(topology_links),
                "excluded_link_count": max(0, len(unique_geometries) - len(topology_links)),
                "inferred_node_count": len(node_positions), "coverage_ratio": 1.0,
                "warning": (
                    "Uploaded geographic road geometry is displayed directly."
                    if package.geographic else "Topology layout only: positions are not geographic coordinates."
                ),
            }
            connection.execute(
                "INSERT INTO metadata(key, value_json) VALUES ('network_mapping_v1', ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                (json.dumps(mapping),),
            )

        summary = dict(summary)
        summary["preferred_map_view"] = package.preferred_view
        summary["geometry_kind"] = geometry_kind if package.geometries else "representative_point"
        summary["geometry_note"] = (
            "Uploaded geographic road geometry is displayed directly; no detector-to-link conversion is required."
            if package.geometries and package.geographic
            else "Topology layout only. Connectivity is preserved, but positions are not geographic locations."
            if package.geometries
            else "Uploaded node locations are displayed directly."
        )
        summary["transport_structure"] = {
            "package_type": package.package_type,
            "geography_mode": package.geography_mode,
            "geographic": package.geographic,
            "node_count": len(package.nodes),
            "link_count": len(unique_geometries),
            "od_count": len(package.od_rows),
            "scenario_effect_count": len(package.scenario_rows),
            "warnings": package.warnings,
        }
        summary["source_files"] = package.source_files
        summary.setdefault("mapping", {})["adaptive_transport_package"] = True
        if not package.geographic:
            summary.setdefault("capabilities", {})["spatial_hotspots"] = "unavailable"
        connection.execute(
            "INSERT INTO metadata(key, value_json) VALUES ('summary', ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (json.dumps(summary),),
        )
        connection.commit()
        return summary
    finally:
        connection.close()
