from __future__ import annotations

import csv
from contextlib import closing
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
from statistics import median
import tempfile
from typing import Any, Callable, Iterable
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


ProgressCallback = Callable[[float, str], None]


ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "entity_id": (
        "id", "linkid", "linkcode", "linkidentifier", "edgeid", "edgekey", "segmentid",
        "segmentcode", "segmentidentifier", "roadid", "roadlink", "sensorid", "sensorref",
        "detectorid", "detectorref", "stationid", "stationref", "siteid", "siteref", "assetid",
        "路段编号", "道路编号", "链路编号", "节点编号", "站点编号", "检测器编号", "传感器编号",
        "roadsegmentid", "segmentuid", "linknumber", "assetcode", "roadasset", "路段编号", "链路编号",
        "道路编号", "设施编号", "检测器编号", "传感器编号", "站点编号", "路段id",
    ),
    "from_node": (
        "from", "fromnode", "fromnodeid", "u", "source", "sourcenode", "tail", "startnode",
        "fromstop", "fromstopid", "originstop", "startstop", "fromstation",
        "起点", "起始节点", "上游节点", "源节点",
        "originnode", "upstreamnode", "upstreamjunction", "起点", "起始节点", "上游节点", "源节点",
    ),
    "to_node": (
        "to", "tonode", "tonodeid", "v", "target", "targetnode", "head", "endnode",
        "tostop", "tostopid", "destinationstop", "endstop", "tostation",
        "终点", "终止节点", "下游节点", "目标节点",
        "destinationnode", "downstreamnode", "downstreamjunction", "终点", "终止节点", "下游节点", "目标节点",
    ),
    "length": (
        "lengthm", "lengthkm", "lengthmi", "length", "distance", "distancem", "segmentlength",
        "长度", "路段长度", "道路长度", "距离",
        "linklength", "roadlength", "长度", "路段长度", "链路长度", "距离",
    ),
    "latitude": (
        "lat", "latitude", "y", "ycoord", "ycoordinate", "centroidlat", "centerlat",
        "纬度", "中心纬度", "质心纬度",
        "centrelatitude", "centrelat", "latdd", "latitudedeg", "纬度", "中心纬度", "质心纬度",
    ),
    "longitude": (
        "lon", "lng", "long", "longitude", "x", "xcoord", "xcoordinate", "centroidlon",
        "经度", "中心经度", "质心经度",
        "centerlon", "centrelon", "centrelongitude", "londd", "longitudedeg", "经度", "中心经度", "质心经度",
    ),
    "time": (
        "datetime", "timestamp", "time", "date", "observationtime", "recordedat", "recordtime",
        "日期时间", "时间", "时间戳", "观测时间", "记录时间",
        "eventtime", "sampletime", "observedat", "日期时间", "时间", "时间戳", "观测时间", "记录时间",
    ),
    "speed": (
        "speedkmh", "speedkph", "speedmph", "speed", "avgspeed", "averagespeed", "meanspeed",
        "速度", "平均速度", "行驶速度", "路段速度",
        "velocity", "velocitykph", "meanvelocity", "linkspeed", "速度", "平均速度", "行驶速度", "路段速度",
    ),
    "travel_time": (
        "traveltimemin", "traveltimeseconds", "traveltime", "tt", "journeytime",
        "旅行时间", "行程时间", "通行时间", "旅行时长", "行驶时间",
        "journeyduration", "journeyminutes", "durationmin", "linktime", "segmenttime", "traversaltime",
        "旅行时间", "行程时间", "通行时间", "旅行时长", "行驶时间",
    ),
    "time_step": (
        "timestep", "simulationstep", "simstep", "step", "sampleinterval", "siminterval", "timeindex",
        "intervalindex", "时间步", "时间步索引", "仿真步", "仿真步索引", "时段编号", "时段索引",
    ),
    "region": (
        "region", "regionid", "regionname", "area", "areaid", "zone", "zoneid", "county",
        "区域", "区域编号", "区域名称", "行政区", "分区",
        "district", "districtname", "jurisdiction", "区域", "区域编号", "区域名称", "行政区", "分区",
    ),
    "primary_metric": (
        "value", "metricvalue", "measurement", "severity", "damage", "hazardvalue",
        "观测值", "指标值", "灾害强度", "损伤", "严重程度",
        "flooddepth", "waterdepth", "fireintensity", "smokedensity", "windgust", "disruptionlevel",
        "closurestatus", "inundationdepth", "groundacceleration", "peakgroundacceleration", "pga",
        "观测值", "指标值", "灾害强度", "洪水深度", "积水深度",
        "火灾强度", "中断程度", "关闭状态",
    ),
    "flow": (
        "flow", "trafficflow", "linkflow", "volume", "trafficvolume", "vehiclesperhour",
        "流量", "交通流量", "交通量", "车辆数",
        "vehperhour", "vph", "vehiclecount", "demandflow", "observedflow",
        "流量", "交通流量", "车流量", "交通量", "车辆数",
    ),
    "capacity": (
        "capacity", "linkcapacity", "roadcapacity", "segmentcapacity", "hourlycapacity",
        "通行能力", "道路容量", "路段容量",
        "capacityvph", "vehperhourcapacity", "maxflow", "maximumflow", "通行能力", "道路容量", "路段容量",
    ),
    "free_flow_time": (
        "freeflowtime", "freeflowtraveltime", "fft", "t0", "uncongestedtime",
        "freetraveltime", "自由流时间", "自由流旅行时间",
    ),
    "repair_cost": (
        "repaircost", "restorationcost", "recoverycost", "reconstructioncost", "costtorepair",
        "修复成本", "恢复成本", "抢修成本",
    ),
    "repair_duration": (
        "repairduration", "repairtime", "restorationduration", "restorationtime", "recoveryduration",
        "timetorepair", "修复时长", "修复时间", "恢复时长",
    ),
    "failure_probability": (
        "failureprobability", "failureprob", "probabilityoffailure", "failurerisk", "fragilityprobability",
        "失效概率", "中断概率", "故障概率",
    ),
}


def _normalise_column(name: str) -> str:
    normalised = unicodedata.normalize("NFKC", str(name).casefold())
    return "".join(character for character in normalised if character.isalnum())


def _mapping_score(role: str, column: str) -> float:
    cleaned = _normalise_column(column)
    if not cleaned:
        return 0.0
    # A field name can contain a valid short alias while describing a
    # different quantity (for example ``speed_limit`` or ``time_zone``).
    # These semantic guard rails keep such metadata out of observations.
    guarded_terms: dict[str, tuple[str, ...]] = {
        "speed": ("speedlimit", "posted", "freeflow", "designspeed", "speedvariance", "speedstd", "speedaccuracy", "speederror"),
        "travel_time": ("reliability", "bufferindex", "planningtime", "variance", "standarddeviation", "traveltimeindex"),
        "time": ("timezone", "timeoffset", "utcoffset", "traveltime", "journeytime", "duration", "timestep", "timeindex", "interval"),
        "time_step": ("timestamp", "datetime", "eventtime", "recordtime", "observationtime", "timezone", "offset"),
        "region": ("geometry", "shape", "polygon", "boundary", "wkt", "geojson", "timezone", "timeoffset", "utcoffset"),
        "latitude": ("uncertainty", "accuracy", "error", "standarddeviation", "variance"),
        "longitude": ("uncertainty", "accuracy", "error", "standarddeviation", "variance"),
        "from_node": ("tonode", "target", "endnode", "destination", "downstream", "headnode"),
        "to_node": ("fromnode", "source", "startnode", "origin", "upstream", "tailnode"),
        "flow": ("workflow", "overflow", "flowdirection", "flowid", "capacity"),
        "capacity": ("utilisation", "utilization", "ratio", "remaining", "storage"),
        "free_flow_time": ("speed", "rate", "index", "ratio"),
        "repair_cost": ("costid", "costcategory", "costcode"),
        "repair_duration": ("durationid", "durationcategory"),
        "failure_probability": ("id", "category", "class"),
    }
    if any(term in cleaned for term in guarded_terms.get(role, ())):
        return 0.0
    scores: list[float] = []
    for alias in ROLE_ALIASES[role]:
        candidate = _normalise_column(alias)
        if cleaned == candidate:
            scores.append(1.0)
        elif len(candidate) >= 4 and candidate in cleaned:
            scores.append(min(0.96, 0.86 + len(candidate) / max(len(cleaned), 1) * 0.10))
        elif len(cleaned) >= 4 and cleaned in candidate:
            scores.append(0.80)
        else:
            ratio = SequenceMatcher(None, cleaned, candidate).ratio()
            if ratio >= 0.72:
                scores.append(ratio * 0.82)
    score = max(scores, default=0.0)
    if role == "time" and any(word in cleaned for word in ("travel", "journey", "duration", "step", "interval", "index")):
        score = min(score, 0.42)
    if role == "entity_id" and any(word in cleaned for word in ("fromnode", "tonode", "source", "target")):
        score = min(score, 0.35)
    return score


def suggest_mapping(columns: Iterable[str]) -> dict[str, Any]:
    originals = [str(column) for column in columns]
    ranked: dict[str, list[tuple[float, str]]] = {
        role: sorted(
            ((_mapping_score(role, column), column) for column in originals),
            key=lambda item: (-item[0], originals.index(item[1])),
        )
        for role in ROLE_ALIASES
    }
    roles: dict[str, str | None] = {}
    confidence: dict[str, str] = {}
    scores: dict[str, float] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    # Allocate the strongest role-column pairs globally.  A fixed role order
    # previously allowed a fuzzy ``time_step`` match to steal an exact
    # ``Timestamp`` field before the time role was considered.  Sorting all
    # edges first guarantees that exact aliases win and keeps one column from
    # silently serving two meanings.
    assignments: dict[str, tuple[float, str]] = {}
    used: set[str] = set()
    edges = sorted(
        (
            (score, role, column, originals.index(column))
            for role, candidates_for_role in ranked.items()
            for score, column in candidates_for_role
            if score >= 0.62
        ),
        key=lambda item: (-item[0], item[3], item[1]),
    )
    for score, role, column, _column_index in edges:
        if role in assignments or column in used:
            continue
        assignments[role] = (score, column)
        used.add(column)

    for role in ROLE_ALIASES:
        score, selected = assignments.get(role, (0.0, None))
        roles[role] = selected
        scores[role] = round(float(score), 3)
        confidence[role] = "confident" if score >= 0.90 else "please_check" if score >= 0.62 else "not_recognised"
        if selected:
            used.add(selected)
        candidates[role] = [
            {"column": column, "score": round(float(candidate_score), 3)}
            for candidate_score, column in ranked[role][:3]
            if candidate_score >= 0.45
        ]
    return {"roles": roles, "confidence": confidence, "scores": scores, "candidates": candidates}


def _detect_text_format(path: Path, encoding_override: str | None = None) -> tuple[str, str]:
    raw = path.read_bytes()[:65536]
    if encoding_override:
        try:
            decoded_override = raw.decode(encoding_override)
        except (LookupError, UnicodeError) as exc:
            raise ValueError(
                f"{path.name}: the confirmed {encoding_override} encoding cannot decode this file."
            ) from exc
        try:
            delimiter_override = csv.Sniffer().sniff(decoded_override, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter_override = ","
        return encoding_override, delimiter_override
    encoding = "utf-8-sig"
    decoded = ""
    # BOMs are definitive and should be honoured before statistical guesses.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ("utf-16",)
    elif raw.startswith(b"\xef\xbb\xbf"):
        candidates = ("utf-8-sig",)
    else:
        candidates = ("utf-8-sig", "utf-8")
    for candidate in candidates:
        try:
            decoded = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeError:
            continue
    else:
        # charset-normalizer is a transitive dependency in the supported web
        # environment.  It prevents GB18030/Big5 text from being accepted as
        # superficially decodable cp1252 mojibake.  The deterministic fallback
        # still supports common Windows and CJK encodings when it is absent.
        detected = None
        try:
            from charset_normalizer import from_bytes

            match = from_bytes(raw).best()
            detected = match.encoding if match else None
        except (ImportError, LookupError, UnicodeError):
            detected = None
        fallback_candidates = tuple(
            dict.fromkeys(
                candidate
                for candidate in (detected, "gb18030", "big5", "cp1252", "latin-1")
                if candidate
            )
        )
        decoded_candidates: list[tuple[tuple[float, float, float], str, str, str]] = []
        for candidate in fallback_candidates:
            try:
                candidate_text = raw.decode(candidate)
            except UnicodeError:
                continue
            try:
                candidate_delimiter = csv.Sniffer().sniff(candidate_text, delimiters=",;\t|").delimiter
            except csv.Error:
                candidate_delimiter = ","
            header_line = candidate_text.splitlines()[0] if candidate_text.splitlines() else ""
            try:
                candidate_columns = next(csv.reader([header_line], delimiter=candidate_delimiter))
            except (csv.Error, StopIteration):
                candidate_columns = []
            semantic = suggest_mapping(candidate_columns)
            core_roles = ("entity_id", "latitude", "longitude", "time", "time_step", "speed", "travel_time", "primary_metric")
            confident_count = sum(semantic["scores"].get(role, 0) >= 0.9 for role in core_roles)
            semantic_sum = sum(float(semantic["scores"].get(role, 0)) for role in core_roles)
            replacement_penalty = candidate_text.count("�") + candidate_text.count("\x00")
            # Ties retain charset-normalizer's first suggestion, while a
            # semantically readable CSV header decisively beats mojibake.
            decoded_candidates.append(
                ((float(confident_count), semantic_sum, -float(replacement_penalty)), candidate, candidate_text, candidate_delimiter)
            )
        if decoded_candidates:
            _score, encoding, decoded, delimiter = max(decoded_candidates, key=lambda item: item[0])
            return encoding, delimiter
        else:
            raise ValueError(
                f"{path.name}: text encoding could not be identified. Save the file as UTF-8 CSV and try again."
            )
    try:
        delimiter = csv.Sniffer().sniff(decoded, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def _sample_coordinate_diagnostics(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    mapping: dict[str, str | None],
    row_limit: int = 2500,
) -> dict[str, Any]:
    latitude_column = mapping.get("latitude")
    longitude_column = mapping.get("longitude")
    if not latitude_column or not longitude_column:
        return {
            "status": "not_mapped",
            "message": "Latitude and longitude still need to be matched.",
            "sample_count": 0,
            "valid_ratio": 0.0,
            "swapped_valid_ratio": 0.0,
            "recommended_swap": False,
        }

    pairs: list[tuple[float, float]] = []
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            if index >= row_limit:
                break
            try:
                latitude = float(str(row.get(latitude_column, "")).strip().replace("−", "-"))
                longitude = float(str(row.get(longitude_column, "")).strip().replace("−", "-"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(latitude) and math.isfinite(longitude):
                pairs.append((latitude, longitude))

    if not pairs:
        return {
            "status": "no_numeric_pairs",
            "message": "The matched coordinate fields contain no readable numeric pairs in the sample.",
            "sample_count": 0,
            "valid_ratio": 0.0,
            "swapped_valid_ratio": 0.0,
            "recommended_swap": False,
        }

    zero_count = sum(abs(latitude) < 1e-12 and abs(longitude) < 1e-12 for latitude, longitude in pairs)
    valid_count = sum(
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and not (abs(latitude) < 1e-12 and abs(longitude) < 1e-12)
        for latitude, longitude in pairs
    )
    swapped_count = sum(
        -90 <= longitude <= 90
        and -180 <= latitude <= 180
        and not (abs(latitude) < 1e-12 and abs(longitude) < 1e-12)
        for latitude, longitude in pairs
    )
    projected_count = sum(abs(latitude) > 180 or abs(longitude) > 180 for latitude, longitude in pairs)
    valid_ratio = valid_count / len(pairs)
    swapped_ratio = swapped_count / len(pairs)
    zero_ratio = zero_count / len(pairs)
    projected_ratio = projected_count / len(pairs)

    if projected_ratio >= 0.8:
        status = "projected_coordinates"
        message = (
            "Values look projected rather than WGS84. The transport normaliser will use a known safe transform "
            "when identifiable; otherwise it will create a clearly labelled non-geographic view."
        )
    elif swapped_ratio >= 0.8 and swapped_ratio >= valid_ratio + 0.5:
        status = "likely_swapped"
        message = "Latitude and longitude appear reversed. The preview has swapped the suggested fields for review."
    elif valid_ratio == 0:
        status = "invalid_coordinates"
        message = "No sampled coordinate pair falls inside valid WGS84 bounds."
    elif zero_ratio >= 0.8:
        status = "zero_island_warning"
        message = "Most coordinates are 0,0. Confirm that missing locations were not encoded as zero."
    elif zero_ratio > 0:
        status = "partial_coordinates"
        message = f"{zero_ratio:.0%} of sampled coordinate pairs are 0,0 and will be treated as missing; {valid_ratio:.0%} are mappable."
    elif valid_ratio < 0.8:
        status = "partial_coordinates"
        message = f"{valid_ratio:.0%} of sampled numeric coordinate pairs are mappable; invalid rows will be flagged."
    else:
        status = "valid"
        message = f"{valid_ratio:.0%} of sampled numeric coordinate pairs are valid WGS84 locations."
    return {
        "status": status,
        "message": message,
        "sample_count": len(pairs),
        "valid_pair_count": valid_count,
        "valid_ratio": round(valid_ratio, 4),
        "swapped_valid_ratio": round(swapped_ratio, 4),
        "zero_ratio": round(zero_ratio, 4),
        "projected_ratio": round(projected_ratio, 4),
        "recommended_swap": status == "likely_swapped",
        "latitude_column": latitude_column,
        "longitude_column": longitude_column,
    }


def _sample_type_diagnostics(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    mapping: dict[str, str | None],
    row_limit: int = 500,
) -> dict[str, Any]:
    """Check whether proposed semantic roles agree with sampled value types."""
    values: dict[str, list[str]] = {role: [] for role in mapping if mapping.get(role)}
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            if index >= row_limit:
                break
            for role, column in mapping.items():
                if column and role in values:
                    value = str(row.get(column, "") or "").strip()
                    if value:
                        values[role].append(value)

    numeric_roles = {
        "latitude", "longitude", "speed", "travel_time", "primary_metric",
        "length", "time_step",
    }
    details: dict[str, Any] = {}
    issues: list[str] = []
    for role, sampled in values.items():
        if not sampled:
            details[role] = {"sample_count": 0, "status": "empty"}
            if role in {"entity_id", "latitude", "longitude", "speed", "travel_time", "primary_metric"}:
                issues.append(f"{role.replace('_', ' ')} is empty in the sample")
            continue
        if role in numeric_roles:
            ratio = float(pd.to_numeric(pd.Series(sampled), errors="coerce").notna().mean())
            status = "compatible" if ratio >= 0.9 else "review" if ratio >= 0.65 else "mismatch"
            details[role] = {"sample_count": len(sampled), "numeric_ratio": round(ratio, 4), "status": status}
            if status != "compatible":
                issues.append(f"{role.replace('_', ' ')} is only {ratio:.0%} numeric in the sample")
        elif role == "time":
            parsed = pd.to_datetime(pd.Series(sampled), errors="coerce", format="mixed", utc=True)
            ratio = float(parsed.notna().mean())
            status = "compatible" if ratio >= 0.9 else "review" if ratio >= 0.65 else "mismatch"
            details[role] = {"sample_count": len(sampled), "datetime_ratio": round(ratio, 4), "status": status}
            if status != "compatible":
                issues.append(f"time is only {ratio:.0%} date/time-like in the sample")
        else:
            details[role] = {"sample_count": len(sampled), "status": "compatible"}
    return {"roles": details, "issues": issues, "sample_count": max((len(item) for item in values.values()), default=0)}


def preview_csv(path: Path, row_limit: int = 8, encoding_override: str | None = None) -> dict[str, Any]:
    from .transport_data import profile_transport_file

    encoding, delimiter = _detect_text_format(path, encoding_override)
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        rows = []
        line_lengths: list[int] = []
        for index, row in enumerate(reader):
            if index >= row_limit:
                break
            rows.append({column: row.get(column, "") for column in columns})
            line_lengths.append(sum(len(str(value)) for value in row.values()) + max(1, len(columns)))
    average_line = max(1, sum(line_lengths) / len(line_lengths)) if line_lengths else 100
    estimated_rows = max(0, int(path.stat().st_size / average_line) - 1)
    suggested_mapping = suggest_mapping(columns)
    transport_structure = profile_transport_file(path, encoding=encoding, delimiter=delimiter)
    if transport_structure.get("adaptive"):
        transport_roles = transport_structure.get("roles") or {}
        role_bridge = {
            "entity_id": "link_id" if (transport_structure.get("record_counts") or {}).get("link") else "node_id",
            "from_node": "from_node", "to_node": "to_node", "latitude": "latitude", "longitude": "longitude",
            "time": "timestamp", "speed": "speed", "travel_time": "travel_time", "length": "length",
            "region": "region", "primary_metric": "primary_metric", "flow": "flow", "capacity": "capacity",
            "free_flow_time": "free_flow_time", "repair_cost": "repair_cost",
            "repair_duration": "repair_duration", "failure_probability": "failure_probability",
        }
        for target_role, source_role in role_bridge.items():
            column = transport_roles.get(source_role)
            # Once a transport structure is recognised, its joint semantic
            # parser is authoritative.  Clear unrelated fuzzy matches (for
            # example ``observed_rank`` as time) instead of carrying them into
            # the confirmed normalisation contract.
            suggested_mapping["roles"][target_role] = column
            suggested_mapping["confidence"][target_role] = "confident" if column else "not_recognised"
            suggested_mapping["scores"][target_role] = 1.0 if column else 0.0
    coordinate_diagnostics = _sample_coordinate_diagnostics(
        path,
        encoding=encoding,
        delimiter=delimiter,
        mapping=suggested_mapping["roles"],
    )
    if coordinate_diagnostics["status"] == "likely_swapped":
        roles = suggested_mapping["roles"]
        roles["latitude"], roles["longitude"] = roles["longitude"], roles["latitude"]
        suggested_mapping["confidence"]["latitude"] = "please_check"
        suggested_mapping["confidence"]["longitude"] = "please_check"
        coordinate_diagnostics = _sample_coordinate_diagnostics(
            path,
            encoding=encoding,
            delimiter=delimiter,
            mapping=roles,
        ) | {
            "status": "review_swap",
            "message": "Values indicated reversed latitude/longitude, so the suggested fields were swapped. Please confirm them.",
            "recommended_swap": True,
        }

    type_diagnostics = _sample_type_diagnostics(
        path,
        encoding=encoding,
        delimiter=delimiter,
        mapping=suggested_mapping["roles"],
    )

    missing_required = [
        role for role in ("entity_id", "latitude", "longitude")
        if not suggested_mapping["roles"].get(role)
    ]
    measurable_roles = (
        "speed", "travel_time", "primary_metric", "flow", "capacity",
        "free_flow_time", "repair_cost", "repair_duration",
        "failure_probability", "length",
    )
    has_metric = any(suggested_mapping["roles"].get(role) for role in measurable_roles)
    blocked_coordinate = coordinate_diagnostics["status"] in {
        "projected_coordinates", "invalid_coordinates", "no_numeric_pairs", "not_mapped"
    }
    low_confidence_core = [
        role
        for role in ("entity_id", "latitude", "longitude", *measurable_roles, "time")
        if suggested_mapping["roles"].get(role)
        and suggested_mapping["confidence"].get(role) != "confident"
    ]
    type_mismatch_core = [
        role
        for role, detail in type_diagnostics["roles"].items()
        if detail.get("status") == "mismatch"
        and role in {"entity_id", "latitude", "longitude", "speed", "travel_time", "primary_metric", "time", "time_step"}
    ]
    needs_review = bool(
        coordinate_diagnostics["status"] in {"review_swap", "partial_coordinates", "zero_island_warning"}
        or low_confidence_core
        or type_diagnostics["issues"]
    )
    mapping_health = {
        "status": "blocked" if missing_required or not has_metric or blocked_coordinate or type_mismatch_core else "review" if needs_review else "ready",
        "missing_required": missing_required,
        "metric_detected": has_metric,
        "low_confidence_roles": low_confidence_core,
        "type_mismatch_roles": type_mismatch_core,
        "review_reasons": type_diagnostics["issues"],
    }
    if transport_structure.get("auto_importable") and transport_structure.get("adaptive"):
        mapping_health = {
            "status": "ready",
            "missing_required": [],
            "metric_detected": has_metric,
            "low_confidence_roles": [],
            "type_mismatch_roles": [],
            "review_reasons": [],
            "adaptive_transport_structure": True,
            "message": transport_structure.get("explanation"),
        }
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": columns,
        "rows": rows,
        "estimated_rows": estimated_rows,
        "suggested_mapping": suggested_mapping,
        "coordinate_diagnostics": coordinate_diagnostics,
        "type_diagnostics": type_diagnostics,
        "mapping_health": mapping_health,
        "transport_structure": transport_structure,
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_text_series(frame: pd.DataFrame, column: str | None, default: str = "") -> pd.Series:
    if column and column in frame.columns:
        return frame[column].astype("string").fillna(default).str.strip()
    return pd.Series([default] * len(frame), index=frame.index, dtype="string")


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _infer_column_unit(role: str, column: str | None, fallback: str) -> str:
    cleaned = _normalise_column(column or "")
    if role == "speed":
        if "mph" in cleaned or "mileperhour" in cleaned:
            return "mph"
        if any(term in cleaned for term in ("mps", "meterpersecond", "metrepersecond")):
            return "m/s"
        if any(term in cleaned for term in ("kmh", "kph", "kilometerperhour", "kilometreperhour")):
            return "km/h"
    if role == "travel_time":
        if any(term in cleaned for term in ("second", "seconds", "traveltimes", "durationsec")):
            return "s"
        if any(term in cleaned for term in ("hour", "hours", "durationhr")):
            return "h"
        if any(term in cleaned for term in ("minute", "minutes", "min")):
            return "min"
    if role == "length":
        if any(term in cleaned for term in ("kilometer", "kilometre", "lengthkm", "distancekm")):
            return "km"
        if any(term in cleaned for term in ("mile", "lengthmi", "distancemi")):
            return "mi"
        if any(term in cleaned for term in ("feet", "foot", "lengthft", "distanceft")):
            return "ft"
        return "m"
    return fallback


def _convert_unit(values: pd.Series, role: str, source: str, target: str) -> pd.Series:
    if role == "length":
        factors = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048}
        return values * factors.get(source, 1.0)
    if not target or target == "uploaded unit" or source == target:
        return values
    if role == "speed":
        to_metres_per_second = {"km/h": 1 / 3.6, "mph": 0.44704, "m/s": 1.0}
        if source in to_metres_per_second and target in to_metres_per_second:
            return values * to_metres_per_second[source] / to_metres_per_second[target]
    if role == "travel_time":
        to_seconds = {"s": 1.0, "min": 60.0, "h": 3600.0}
        if source in to_seconds and target in to_seconds:
            return values * to_seconds[source] / to_seconds[target]
    return values


def _timezone_is_unspecified(declared_timezone: str | None) -> bool:
    """Identify disclosure text that is not an executable IANA zone.

    Older project files used this field for notes such as ``local time`` or
    ``America/New_York (source convention; not converted)``. Only UTC/GMT or
    a clean IANA identifier is executable. Free text remains a disclosure and
    follows the documented UTC ordering convention; a clean but unknown IANA
    identifier is rejected later with a readable error.
    """
    name = str(declared_timezone or "").strip()
    folded = name.casefold()
    if folded in {"utc", "gmt", "etc/utc", "etc/gmt"}:
        return False
    return not bool(re.fullmatch(r"[A-Za-z0-9_+.-]+(?:/[A-Za-z0-9_+.-]+)+", name))


def _parse_datetime_series(values: pd.Series, declared_timezone: str) -> pd.Series:
    """Parse timestamps and apply a declared zone to naive local times.

    Offset-aware timestamps retain their own offsets.  A valid IANA zone such
    as ``Australia/Sydney`` is applied only when the uploaded values are
    naive.  ``unspecified`` preserves the legacy, explicitly disclosed UTC
    ordering convention.
    """
    text = values.astype("string").str.strip()
    timezone_name = str(declared_timezone or "unspecified").strip()
    unspecified = _timezone_is_unspecified(timezone_name)
    has_explicit_offset = text.str.contains(
        r"(?:z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False
    )
    if unspecified:
        return pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown time convention '{timezone_name}'. Use an IANA zone such as Australia/Sydney or UTC."
        ) from exc
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    try:
        if bool(has_explicit_offset.any()):
            parsed.loc[has_explicit_offset] = pd.to_datetime(
                text.loc[has_explicit_offset], errors="coerce", format="mixed", utc=True
            )
        naive_mask = ~has_explicit_offset
        if bool(naive_mask.any()):
            local = pd.to_datetime(
                text.loc[naive_mask], errors="coerce", format="mixed", utc=False
            )
            parsed.loc[naive_mask] = local.dt.tz_localize(
                zone, ambiguous="NaT", nonexistent="NaT"
            ).dt.tz_convert("UTC")
        return parsed
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Uploaded timestamps mix incompatible local and offset-aware formats. Split them or use explicit UTC offsets."
        ) from exc


def _quality_flags(
    entity_ids: pd.Series,
    latitude: pd.Series,
    longitude: pd.Series,
    time_valid: pd.Series,
    speed: pd.Series,
    travel_time: pd.Series,
) -> list[int]:
    invalid_coordinate = latitude.isna() | longitude.isna() | ~latitude.between(-90, 90) | ~longitude.between(-180, 180)
    # A reported speed of exactly zero can be a real observed closure and must
    # remain analyzable.  Negative speeds are invalid; zero speeds are counted
    # separately in project quality but are not hidden from disruption models.
    invalid_speed = speed.notna() & speed.lt(0)
    nonpositive_travel = travel_time.notna() & travel_time.le(0)
    # Compact bit masks avoid repeating long labels for every observation in
    # large projects.  The API expands them into readable labels.
    return [
        int(coordinate)
        | (int(invalid_time) << 1)
        | (int(bad_speed) << 2)
        | (int(bad_travel) << 3)
        | (int(missing_entity) << 4)
        for coordinate, invalid_time, bad_speed, bad_travel, missing_entity in zip(
            invalid_coordinate, ~time_valid, invalid_speed, nonpositive_travel, entity_ids.isna() | entity_ids.eq("")
        )
    ]


def _initialise_project_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"Project database already exists: {path}")
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE entities (
            entity_pk INTEGER PRIMARY KEY,
            region_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            from_node TEXT,
            to_node TEXT,
            length_m REAL,
            latitude REAL,
            longitude REAL,
            geometry_kind TEXT NOT NULL DEFAULT 'representative_point',
            baseline_speed REAL,
            baseline_travel_time REAL,
            baseline_primary REAL,
            capacity REAL,
            free_flow_time REAL,
            repair_cost REAL,
            repair_duration REAL,
            failure_probability REAL,
            UNIQUE(region_id, entity_id)
        );
        CREATE TABLE times (
            timestamp_epoch INTEGER PRIMARY KEY,
            timestamp_text TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE observations (
            entity_pk INTEGER NOT NULL,
            timestamp_epoch INTEGER NOT NULL,
            speed REAL,
            travel_time REAL,
            primary_value REAL,
            flow REAL,
            time_step REAL,
            quality_flags INTEGER NOT NULL DEFAULT 0,
            source_row_count INTEGER NOT NULL DEFAULT 1,
            speed_source_count INTEGER NOT NULL DEFAULT 0,
            travel_time_source_count INTEGER NOT NULL DEFAULT 0,
            primary_source_count INTEGER NOT NULL DEFAULT 0,
            flow_source_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(entity_pk, timestamp_epoch)
        ) WITHOUT ROWID;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


@dataclass
class ImportCounters:
    rows_seen: int = 0
    observations_inserted: int = 0
    duplicate_entity_times: int = 0
    invalid_coordinates: int = 0
    invalid_times: int = 0
    nonpositive_speeds: int = 0
    zero_speeds: int = 0
    negative_speeds: int = 0
    nonpositive_travel_times: int = 0
    missing_entity_ids: int = 0
    candidate_observation_rows: int = 0


def import_csv_project(
    *,
    files: list[Path],
    mapping: dict[str, str | None],
    file_mappings: dict[str, dict[str, str | None]] | None = None,
    file_encodings: dict[str, str] | None = None,
    units: dict[str, str],
    timezone: str,
    time_step_seconds: float = 300.0,
    topology_node_scope: str = "global",
    entity_id_scope: str = "per_region",
    primary_metric_direction: str = "descriptive",
    primary_metric_comparison: str = "current_value",
    primary_metric_scale: float = 1.0,
    database_path: Path,
    project_id: str,
    project_name: str,
    hazard_type: str,
    source: str,
    progress: ProgressCallback | None = None,
    _adaptive_checked: bool = False,
) -> dict[str, Any]:
    callback = progress or (lambda _value, _message: None)
    if not _adaptive_checked:
        from .transport_data import install_prepared_package, prepare_transport_package

        previews = [
            preview_csv(path, encoding_override=(file_encodings or {}).get(path.name))
            for path in files
        ]
        prepared = prepare_transport_package(
            files,
            previews=previews,
            entity_id_scope=entity_id_scope,
            confirmed_file_mappings=file_mappings,
        )
        if prepared is not None:
            callback(1.0, "Detected transport-network structure")
            with tempfile.TemporaryDirectory(prefix="twm_transport_") as temporary_folder:
                canonical_path = Path(temporary_folder) / "normalised_transport_records.csv"
                prepared.write_canonical_csv(canonical_path)
                canonical_mapping = prepared.canonical_mapping
                summary = import_csv_project(
                    files=[canonical_path],
                    mapping=canonical_mapping,
                    file_mappings={canonical_path.name: canonical_mapping},
                    file_encodings={canonical_path.name: "utf-8-sig"},
                    units={**units, "speed": "km/h", "travel_time": "min", "length": "m"},
                    timezone=timezone,
                    time_step_seconds=time_step_seconds,
                    topology_node_scope=topology_node_scope,
                    entity_id_scope=entity_id_scope,
                    primary_metric_direction=primary_metric_direction,
                    primary_metric_comparison=primary_metric_comparison,
                    primary_metric_scale=primary_metric_scale,
                    database_path=database_path,
                    project_id=project_id,
                    project_name=project_name,
                    hazard_type=hazard_type,
                    source=source,
                    progress=callback,
                    _adaptive_checked=True,
                )
            callback(98.0, "Installing uploaded road geometry and transport relations")
            summary = install_prepared_package(
                database_path,
                prepared,
                summary,
                topology_node_scope=topology_node_scope,
            )
            callback(100.0, "Project is ready")
            return summary
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _initialise_project_database(database_path)
    counters = ImportCounters()
    source_files: list[dict[str, Any]] = []
    total_size = sum(path.stat().st_size for path in files)
    bytes_completed = 0
    entity_pk_cache: dict[tuple[str, str], int] = {}
    confirmed_file_mappings = file_mappings or {}
    confirmed_file_encodings = file_encodings or {}

    try:
        for file_index, path in enumerate(files):
            file_preview = preview_csv(path, encoding_override=confirmed_file_encodings.get(path.name))
            columns = set(file_preview["columns"])
            suggestions = file_preview["suggested_mapping"]
            has_explicit_file_contract = path.name in confirmed_file_mappings
            confirmed_for_file = confirmed_file_mappings.get(path.name, mapping)
            confirmed_for_file = dict(confirmed_for_file)
            if not any(confirmed_for_file.get(role) for role in ("speed", "travel_time", "primary_metric")):
                fallback_role = next((
                    role for role in ("flow", "capacity", "free_flow_time", "repair_cost", "repair_duration", "failure_probability", "length")
                    if confirmed_for_file.get(role)
                ), None)
                if fallback_role:
                    # Preserve the specialised field and also expose it as a
                    # descriptive observation so static non-template uploads
                    # remain geographically viewable without inventing a
                    # degradation direction.
                    confirmed_for_file["primary_metric"] = confirmed_for_file[fallback_role]
            file_mapping: dict[str, str | None] = {}
            for role in ROLE_ALIASES:
                if has_explicit_file_contract and role in confirmed_for_file:
                    confirmed_column = confirmed_for_file.get(role)
                    file_mapping[role] = confirmed_column if confirmed_column in columns else None
                    continue
                confirmed_column = confirmed_for_file.get(role)
                if confirmed_column and confirmed_column in columns:
                    file_mapping[role] = confirmed_column
                    continue
                suggested_column = suggestions.get("roles", {}).get(role)
                suggested_score = float(suggestions.get("scores", {}).get(role, 0.0))
                file_mapping[role] = suggested_column if suggested_column in columns and suggested_score >= 0.62 else None
            missing_required = [
                role
                for role in ("entity_id", "latitude", "longitude")
                if not file_mapping.get(role)
            ]
            if missing_required:
                raise ValueError(f"{path.name} is missing mapped fields: {', '.join(missing_required)}")

            coordinate_diagnostics = _sample_coordinate_diagnostics(
                path,
                encoding=file_preview["encoding"],
                delimiter=file_preview["delimiter"],
                mapping=file_mapping,
            )
            if coordinate_diagnostics["status"] == "likely_swapped":
                raise ValueError(
                    f"{path.name}: latitude and longitude appear reversed. Return to field confirmation and swap them."
                )
            if coordinate_diagnostics["status"] == "projected_coordinates":
                raise ValueError(
                    f"{path.name}: coordinate values appear projected rather than WGS84. Reproject to longitude/latitude (EPSG:4326) before import."
                )
            if coordinate_diagnostics["status"] in {"invalid_coordinates", "no_numeric_pairs"}:
                raise ValueError(f"{path.name}: {coordinate_diagnostics['message']}")

            selected_columns = sorted({column for column in file_mapping.values() if column and column in columns})
            string_columns = {
                column: "string"
                for role in ("entity_id", "from_node", "to_node", "region")
                if (column := file_mapping.get(role)) and column in columns
            }
            region_fallback = path.stem if entity_id_scope == "per_region" else "global"
            source_units = {
                "speed": _infer_column_unit("speed", file_mapping.get("speed"), units.get("speed", "km/h")),
                "travel_time": _infer_column_unit("travel_time", file_mapping.get("travel_time"), units.get("travel_time", "min")),
                "free_flow_time": _infer_column_unit("travel_time", file_mapping.get("free_flow_time"), units.get("travel_time", "min")),
                "length": _infer_column_unit("length", file_mapping.get("length"), "m"),
            }
            callback(
                min(84.0, 2.0 + bytes_completed / max(1, total_size) * 82.0),
                f"Reading {path.name}",
            )

            reader = pd.read_csv(
                path,
                usecols=selected_columns,
                chunksize=100_000,
                encoding=file_preview["encoding"],
                sep=file_preview["delimiter"],
                dtype=string_columns or None,
                low_memory=False,
            )
            for chunk_index, frame in enumerate(reader):
                count = len(frame)
                if not count:
                    continue
                counters.rows_seen += count
                if entity_id_scope == "global":
                    # The user's global-ID confirmation means the same ID is
                    # one entity across every file, even when files carry a
                    # source-region column.  Keep one stable key namespace.
                    region = pd.Series(region_fallback, index=frame.index, dtype="string")
                else:
                    region = _safe_text_series(frame, file_mapping.get("region"), region_fallback).replace("", region_fallback)
                entity_id = _safe_text_series(frame, file_mapping.get("entity_id"))
                from_node = _safe_text_series(frame, file_mapping.get("from_node"))
                to_node = _safe_text_series(frame, file_mapping.get("to_node"))
                length = _numeric_series(frame, file_mapping.get("length"))
                latitude = _numeric_series(frame, file_mapping.get("latitude"))
                longitude = _numeric_series(frame, file_mapping.get("longitude"))
                speed = _numeric_series(frame, file_mapping.get("speed"))
                travel_time = _numeric_series(frame, file_mapping.get("travel_time"))
                primary = _numeric_series(frame, file_mapping.get("primary_metric"))
                flow = _numeric_series(frame, file_mapping.get("flow"))
                capacity = _numeric_series(frame, file_mapping.get("capacity"))
                free_flow_time = _numeric_series(frame, file_mapping.get("free_flow_time"))
                repair_cost = _numeric_series(frame, file_mapping.get("repair_cost"))
                repair_duration = _numeric_series(frame, file_mapping.get("repair_duration"))
                failure_probability = _numeric_series(frame, file_mapping.get("failure_probability"))
                time_step = _numeric_series(frame, file_mapping.get("time_step"))
                length = _convert_unit(length, "length", source_units["length"], "m")
                speed = _convert_unit(speed, "speed", source_units["speed"], units.get("speed", "km/h"))
                travel_time = _convert_unit(
                    travel_time,
                    "travel_time",
                    source_units["travel_time"],
                    units.get("travel_time", "min"),
                )
                free_flow_time = _convert_unit(
                    free_flow_time,
                    "travel_time",
                    source_units["free_flow_time"],
                    units.get("travel_time", "min"),
                )
                flow = flow.where(flow.ge(0))
                capacity = capacity.where(capacity.gt(0))
                free_flow_time = free_flow_time.where(free_flow_time.gt(0))
                repair_cost = repair_cost.where(repair_cost.ge(0))
                repair_duration = repair_duration.where(repair_duration.ge(0))
                failure_probability = failure_probability.where(failure_probability.between(0, 1))
                zero_coordinate = latitude.fillna(np.nan).eq(0) & longitude.fillna(np.nan).eq(0)
                latitude = latitude.mask(zero_coordinate)
                longitude = longitude.mask(zero_coordinate)

                time_column = file_mapping.get("time")
                if time_column and time_column in frame.columns:
                    parsed_time = _parse_datetime_series(frame[time_column], timezone)
                    time_valid = parsed_time.notna()
                    timestamp_text = parsed_time.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
                    timestamp_epoch = pd.Series(0, index=frame.index, dtype="int64")
                    if time_valid.any():
                        timestamp_epoch.loc[time_valid] = (
                            parsed_time.loc[time_valid].dt.as_unit("s").astype("int64")
                        )
                elif file_mapping.get("time_step") and file_mapping["time_step"] in frame.columns:
                    time_valid = time_step.notna() & np.isfinite(time_step)
                    timestamp_epoch = pd.Series(0, index=frame.index, dtype="int64")
                    if time_valid.any():
                        timestamp_epoch.loc[time_valid] = (
                            time_step.loc[time_valid].astype("float64") * float(time_step_seconds)
                        ).round().astype("int64")
                    timestamp_text = pd.Series("", index=frame.index, dtype="string")
                    timestamp_text.loc[time_valid] = time_step.loc[time_valid].map(
                        lambda value: f"Step {float(value):g}"
                    )
                else:
                    time_valid = pd.Series(True, index=frame.index)
                    timestamp_text = pd.Series("static", index=frame.index)
                    timestamp_epoch = pd.Series(0, index=frame.index, dtype="int64")

                flags = _quality_flags(entity_id, latitude, longitude, time_valid, speed, travel_time)
                counters.invalid_coordinates += int(
                    (latitude.isna() | longitude.isna() | ~latitude.between(-90, 90) | ~longitude.between(-180, 180)).sum()
                )
                counters.invalid_times += int((~time_valid).sum())
                counters.nonpositive_speeds += int((speed.notna() & speed.le(0)).sum())
                counters.zero_speeds += int((speed.notna() & speed.eq(0)).sum())
                counters.negative_speeds += int((speed.notna() & speed.lt(0)).sum())
                counters.nonpositive_travel_times += int((travel_time.notna() & travel_time.le(0)).sum())
                counters.missing_entity_ids += int((entity_id.isna() | entity_id.eq("")).sum())

                valid_entity = entity_id.ne("") & entity_id.notna()
                entity_frame = pd.DataFrame(
                    {
                        "region": region,
                        "entity": entity_id,
                        "from_node": from_node,
                        "to_node": to_node,
                        "length": length,
                        "latitude": latitude,
                        "longitude": longitude,
                        "capacity": capacity,
                        "free_flow_time": free_flow_time,
                        "repair_cost": repair_cost,
                        "repair_duration": repair_duration,
                        "failure_probability": failure_probability,
                    }
                ).loc[valid_entity]
                # Prefer the most complete row for an entity within this
                # chunk.  Previously a first row with missing coordinates
                # permanently hid a later valid location before the database
                # upsert could see it.
                entity_frame["_completeness"] = (
                    (entity_frame["latitude"].notna() & entity_frame["longitude"].notna()).astype(int) * 4
                    + entity_frame["from_node"].fillna("").ne("").astype(int)
                    + entity_frame["to_node"].fillna("").ne("").astype(int)
                    + entity_frame["length"].notna().astype(int)
                    + entity_frame["capacity"].notna().astype(int)
                    + entity_frame["free_flow_time"].notna().astype(int)
                )
                entity_frame = (
                    entity_frame.sort_values("_completeness", ascending=False, kind="stable")
                    .drop_duplicates(subset=["region", "entity"], keep="first")
                    .drop(columns=["_completeness"])
                )
                connection.executemany(
                    """
                    INSERT INTO entities
                    (region_id, entity_id, from_node, to_node, length_m, latitude, longitude,
                     capacity, free_flow_time, repair_cost, repair_duration, failure_probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(region_id, entity_id) DO UPDATE SET
                        from_node = CASE WHEN entities.from_node IS NULL OR entities.from_node = '' THEN excluded.from_node ELSE entities.from_node END,
                        to_node = CASE WHEN entities.to_node IS NULL OR entities.to_node = '' THEN excluded.to_node ELSE entities.to_node END,
                        length_m = COALESCE(entities.length_m, excluded.length_m),
                        latitude = COALESCE(entities.latitude, excluded.latitude),
                        longitude = COALESCE(entities.longitude, excluded.longitude),
                        capacity = COALESCE(entities.capacity, excluded.capacity),
                        free_flow_time = COALESCE(entities.free_flow_time, excluded.free_flow_time),
                        repair_cost = COALESCE(entities.repair_cost, excluded.repair_cost),
                        repair_duration = COALESCE(entities.repair_duration, excluded.repair_duration),
                        failure_probability = COALESCE(entities.failure_probability, excluded.failure_probability)
                    """,
                    entity_frame.itertuples(index=False, name=None),
                )

                # Resolve the compact integer key once per entity and retain it
                # across chunks.  Region-by-region queries keep this bounded for
                # uploads that contain many named areas.
                for region_value in entity_frame["region"].dropna().astype(str).unique():
                    for entity_pk, stored_region, stored_entity in connection.execute(
                        "SELECT entity_pk, region_id, entity_id FROM entities WHERE region_id = ?",
                        (region_value,),
                    ):
                        entity_pk_cache[(str(stored_region), str(stored_entity))] = int(entity_pk)

                entity_pks = pd.Series(
                    [
                        entity_pk_cache.get((str(region_value), str(entity_value)))
                        for region_value, entity_value in zip(region, entity_id)
                    ],
                    index=frame.index,
                    dtype="Int64",
                )

                time_frame = pd.DataFrame(
                    {"timestamp_epoch": timestamp_epoch, "timestamp_text": timestamp_text}
                ).loc[time_valid]
                time_frame = time_frame.drop_duplicates(subset=["timestamp_epoch"], keep="first")
                connection.executemany(
                    "INSERT OR IGNORE INTO times(timestamp_epoch, timestamp_text) VALUES (?, ?)",
                    time_frame.itertuples(index=False, name=None),
                )

                observation_frame = pd.DataFrame(
                    {
                        "entity_pk": entity_pks,
                        "timestamp_epoch": timestamp_epoch,
                        "speed": speed,
                        "travel_time": travel_time,
                        "primary": primary,
                        "flow": flow,
                        "time_step": time_step,
                        "flags": flags,
                        "source_row_count": 1,
                        "speed_source_count": speed.notna().astype("int64"),
                        "travel_time_source_count": travel_time.notna().astype("int64"),
                        "primary_source_count": primary.notna().astype("int64"),
                        "flow_source_count": flow.notna().astype("int64"),
                    }
                ).loc[valid_entity & time_valid & entity_pks.notna()]
                observation_frame["entity_pk"] = observation_frame["entity_pk"].astype("int64")
                counters.candidate_observation_rows += len(observation_frame)
                connection.executemany(
                    """
                    INSERT INTO observations
                    (entity_pk, timestamp_epoch, speed, travel_time, primary_value, flow, time_step, quality_flags,
                     source_row_count, speed_source_count, travel_time_source_count, primary_source_count, flow_source_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_pk, timestamp_epoch) DO UPDATE SET
                        speed = CASE
                            WHEN observations.speed_source_count + excluded.speed_source_count = 0 THEN NULL
                            ELSE (COALESCE(observations.speed, 0) * observations.speed_source_count
                                  + COALESCE(excluded.speed, 0) * excluded.speed_source_count)
                                 / (observations.speed_source_count + excluded.speed_source_count)
                        END,
                        travel_time = CASE
                            WHEN observations.travel_time_source_count + excluded.travel_time_source_count = 0 THEN NULL
                            ELSE (COALESCE(observations.travel_time, 0) * observations.travel_time_source_count
                                  + COALESCE(excluded.travel_time, 0) * excluded.travel_time_source_count)
                                 / (observations.travel_time_source_count + excluded.travel_time_source_count)
                        END,
                        primary_value = CASE
                            WHEN observations.primary_source_count + excluded.primary_source_count = 0 THEN NULL
                            ELSE (COALESCE(observations.primary_value, 0) * observations.primary_source_count
                                  + COALESCE(excluded.primary_value, 0) * excluded.primary_source_count)
                                 / (observations.primary_source_count + excluded.primary_source_count)
                        END,
                        flow = CASE
                            WHEN observations.flow_source_count + excluded.flow_source_count = 0 THEN NULL
                            ELSE (COALESCE(observations.flow, 0) * observations.flow_source_count
                                  + COALESCE(excluded.flow, 0) * excluded.flow_source_count)
                                 / (observations.flow_source_count + excluded.flow_source_count)
                        END,
                        time_step = COALESCE(observations.time_step, excluded.time_step),
                        quality_flags = observations.quality_flags | excluded.quality_flags,
                        source_row_count = observations.source_row_count + excluded.source_row_count,
                        speed_source_count = observations.speed_source_count + excluded.speed_source_count,
                        travel_time_source_count = observations.travel_time_source_count + excluded.travel_time_source_count,
                        primary_source_count = observations.primary_source_count + excluded.primary_source_count,
                        flow_source_count = observations.flow_source_count + excluded.flow_source_count
                    """,
                    observation_frame.itertuples(index=False, name=None),
                )
                connection.commit()

                # ``approximate_fraction`` is a ratio in [0, 1].  The previous
                # implementation mixed percentages and ratios, capped every
                # non-trivial file at 0.95%, and therefore left the UI near 3%
                # until the import completed.  Row progress is approximate,
                # but each reported value now reflects completed CSV chunks.
                approximate_fraction = min(
                    1.0,
                    (chunk_index + 1) * 100_000 / max(1, file_preview["estimated_rows"]),
                )
                overall = (
                    bytes_completed + path.stat().st_size * approximate_fraction
                ) / max(1, total_size)
                callback(min(84.0, 2.0 + overall * 82.0), f"Imported {counters.rows_seen:,} observations")

            bytes_completed += path.stat().st_size
            source_files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "region_id": region_fallback,
                    "sha256": file_sha256(path),
                    "encoding": file_preview["encoding"],
                    "delimiter": file_preview["delimiter"],
                    "resolved_mapping": file_mapping,
                    "source_units": source_units,
                }
            )

        callback(86.0, "Building map and time indexes")
        connection.executescript(
            """
            CREATE INDEX idx_observations_time ON observations(timestamp_epoch);
            CREATE INDEX idx_entities_coordinates ON entities(latitude, longitude);
            CREATE INDEX idx_entities_topology ON entities(from_node, to_node);
            """
        )
        connection.commit()
        counters.observations_inserted = int(
            connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        )
        counters.duplicate_entity_times = max(
            0, counters.candidate_observation_rows - counters.observations_inserted
        )

        time_values = [
            int(row[0])
            for row in connection.execute(
                "SELECT timestamp_epoch FROM times ORDER BY timestamp_epoch"
            ).fetchall()
        ]
        intervals = [
            later - earlier
            for earlier, later in zip(time_values, time_values[1:])
            if later > earlier
        ]
        expected_interval = int(median(intervals)) if intervals else 0
        baseline_cutoff = time_values[min(len(time_values) - 1, max(0, math.ceil(len(time_values) * 0.15) - 1))] if time_values else 0

        callback(91.0, "Calculating transparent internal reference values")
        connection.executescript(
            f"""
            CREATE TEMP TABLE baseline_values (
                entity_pk INTEGER PRIMARY KEY,
                baseline_speed REAL,
                baseline_travel_time REAL,
                baseline_primary REAL
            );
            INSERT INTO baseline_values
            SELECT entity_pk,
                   AVG(CASE WHEN speed > 0 THEN speed END) AS baseline_speed,
                   AVG(CASE WHEN travel_time > 0 THEN travel_time END) AS baseline_travel_time,
                   AVG(primary_value) AS baseline_primary
            FROM observations
            WHERE timestamp_epoch <= {int(baseline_cutoff)}
            GROUP BY entity_pk;

            UPDATE entities SET
                baseline_speed = (SELECT baseline_speed FROM baseline_values b
                                  WHERE b.entity_pk = entities.entity_pk),
                baseline_travel_time = (SELECT baseline_travel_time FROM baseline_values b
                                        WHERE b.entity_pk = entities.entity_pk),
                baseline_primary = (SELECT baseline_primary FROM baseline_values b
                                    WHERE b.entity_pk = entities.entity_pk)
            WHERE entity_pk IN (SELECT entity_pk FROM baseline_values);
            DROP TABLE baseline_values;
            """
        )
        connection.commit()

        entity_count = int(connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
        mapped_entity_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM entities WHERE latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180"
            ).fetchone()[0]
        )
        if mapped_entity_count == 0:
            raise ValueError(
                "No entity has a valid WGS84 latitude/longitude pair. Confirm field mapping or reproject coordinates to EPSG:4326."
            )
        region_count = int(connection.execute("SELECT COUNT(DISTINCT region_id) FROM entities").fetchone()[0])
        extent = connection.execute(
            "SELECT MIN(longitude), MIN(latitude), MAX(longitude), MAX(latitude) FROM entities WHERE latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180"
        ).fetchone()
        topology_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM entities WHERE from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''"
            ).fetchone()[0]
        )
        topology_weight_counts = connection.execute(
            """
            SELECT
                SUM(from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''
                    AND baseline_travel_time > 0),
                SUM(from_node IS NOT NULL AND from_node <> '' AND to_node IS NOT NULL AND to_node <> ''
                    AND length_m > 0)
            FROM entities
            """
        ).fetchone()
        metric_counts = connection.execute(
            """
            SELECT SUM(speed IS NOT NULL), SUM(travel_time IS NOT NULL), SUM(primary_value IS NOT NULL)
            FROM observations
            """
        ).fetchone()
        advanced_counts = connection.execute(
            """
            SELECT
                (SELECT SUM(flow IS NOT NULL) FROM observations),
                SUM(capacity IS NOT NULL),
                SUM(free_flow_time IS NOT NULL),
                SUM(repair_cost IS NOT NULL),
                SUM(repair_duration IS NOT NULL),
                SUM(failure_probability IS NOT NULL)
            FROM entities
            """
        ).fetchone()
        if counters.observations_inserted == 0:
            raise ValueError(
                "No valid observation was imported. Recheck the entity, time and measured-value field mappings."
            )
        geometry_only = not any(int(value or 0) > 0 for value in metric_counts)
        has_time = len(time_values) > 1 or (time_values and time_values[0] > 0)
        has_speed = bool(metric_counts[0])
        has_travel_time = bool(metric_counts[1])
        has_primary = bool(metric_counts[2])
        has_flow = bool(advanced_counts[0])
        has_capacity = bool(advanced_counts[1])
        has_free_flow_time = bool(advanced_counts[2])
        has_repair_cost = bool(advanced_counts[3])
        has_repair_duration = bool(advanced_counts[4])
        has_failure_probability = bool(advanced_counts[5])
        # Length is also a legitimate descriptive/map metric for point-only
        # uploads, so do not hide it merely because topology is absent.
        has_length = bool(
            connection.execute("SELECT SUM(length_m IS NOT NULL AND length_m > 0) FROM entities").fetchone()[0]
        )
        primary_severity_enabled = has_primary and primary_metric_direction in {"higher_worse", "lower_worse"}
        has_topology = topology_count >= max(1, int(entity_count * 0.8))
        has_complete_topology_impedance = bool(
            has_topology and (
                int(topology_weight_counts[0] or 0) == topology_count
                or int(topology_weight_counts[1] or 0) == topology_count
            )
        )

        time_labels = {
            int(row[0]): row[1]
            for row in connection.execute(
                "SELECT timestamp_epoch, timestamp_text FROM times ORDER BY timestamp_epoch"
            ).fetchall()
        }
        maximum_timeline_points = 2000
        if len(time_values) > maximum_timeline_points:
            timeline_indices = sorted({
                round(index * (len(time_values) - 1) / (maximum_timeline_points - 1))
                for index in range(maximum_timeline_points)
            })
            display_time_values = [time_values[index] for index in timeline_indices]
        else:
            display_time_values = time_values
        display_time_labels = {
            str(value): time_labels[value]
            for value in display_time_values
        }
        summary: dict[str, Any] = {
            "project_id": project_id,
            "name": project_name,
            "hazard_type": hazard_type,
            "source": source,
            "entity_count": entity_count,
            "mapped_entity_count": mapped_entity_count,
            "unmapped_entity_count": entity_count - mapped_entity_count,
            "observation_count": counters.observations_inserted,
            "region_count": region_count,
            "extent": list(extent) if extent and extent[0] is not None else None,
            "geometry_kind": "representative_point",
            "geometry_note": (
                "Each uploaded coordinate is a representative road-link point, not a topology node or exact road polyline."
                if has_topology
                else "Each uploaded coordinate is an entity location. No line geometry or topology is inferred."
            ),
            "time_values": display_time_values,
            "time_labels": display_time_labels,
            "time_value_count": len(time_values),
            "timeline_sampled": len(display_time_values) < len(time_values),
            "time_start": time_labels.get(time_values[0]) if time_values else "static",
            "time_end": time_labels.get(time_values[-1]) if time_values else "static",
            "expected_interval_s": expected_interval,
            "baseline": {
                "type": "internal_reference_window",
                "cutoff_epoch": baseline_cutoff,
                "cutoff_label": time_labels.get(baseline_cutoff, "static"),
                "warning": "The earliest 15% of uploaded time points are an internal reference, not automatically a pre-disaster baseline.",
            },
            "metrics": {
                "speed": {"available": has_speed, "unit": units.get("speed", "km/h")},
                "travel_time": {"available": has_travel_time, "unit": units.get("travel_time", "min")},
                "primary": {
                    "available": has_primary,
                    "unit": units.get("primary_metric", "uploaded unit"),
                    "semantics": {
                        "direction": primary_metric_direction,
                        "comparison": primary_metric_comparison,
                        "scale": float(primary_metric_scale),
                        "severity_enabled": primary_severity_enabled,
                    },
                },
                "flow": {"available": has_flow, "unit": units.get("flow", "vehicles/hour")},
                "capacity": {"available": has_capacity, "unit": units.get("capacity", "vehicles/hour")},
                "free_flow_time": {"available": has_free_flow_time, "unit": units.get("travel_time", "min")},
                "repair_cost": {"available": has_repair_cost, "unit": units.get("repair_cost", "uploaded cost unit")},
                "repair_duration": {"available": has_repair_duration, "unit": units.get("repair_duration", "uploaded time unit")},
                "failure_probability": {"available": has_failure_probability, "unit": "probability [0,1]"},
                "length": {"available": has_length, "unit": units.get("length", "m")},
            },
            "timezone": timezone,
            "timezone_application": (
                "No executable IANA zone was declared. Naive timestamps were treated as UTC; explicit offsets were preserved."
                if _timezone_is_unspecified(timezone)
                else "Naive timestamps were localised using the declared time zone and stored in UTC; explicit offsets were preserved."
            ),
            "time_step_seconds": float(time_step_seconds),
            "topology_node_scope": topology_node_scope,
            "entity_id_scope": entity_id_scope,
            "quality": {
                "invalid_coordinates": counters.invalid_coordinates,
                "invalid_times": counters.invalid_times,
                "nonpositive_speeds": counters.nonpositive_speeds,
                "zero_speeds_observed": counters.zero_speeds,
                "negative_speeds_invalid": counters.negative_speeds,
                "nonpositive_travel_times": counters.nonpositive_travel_times,
                "duplicate_entity_times": counters.duplicate_entity_times,
                "duplicate_resolution": "Rows sharing the same confirmed entity and time are combined by an order-independent numeric mean; quality flags are retained.",
                "missing_entity_ids": counters.missing_entity_ids,
                "geometry_only": geometry_only,
            },
            "capabilities": {
                "historical_reconstruction": "available" if has_time else "conditional",
                "persistent_degradation": "available" if has_time and (has_speed or has_travel_time or primary_severity_enabled) else "unavailable",
                "topology_criticality": "available" if has_complete_topology_impedance else "conditional" if has_topology else "unavailable",
                "closure_scenario": "available" if has_complete_topology_impedance else "conditional" if has_topology else "unavailable",
                "restoration_screening": "conditional" if has_topology else "unavailable",
                "temporal_resilience": "conditional" if has_time and (has_speed or has_travel_time) else "unavailable",
                "spatial_hotspots": "available" if mapped_entity_count > 0 and (has_speed or has_travel_time or primary_severity_enabled) else "unavailable",
                "progressive_robustness": "available" if has_complete_topology_impedance else "conditional" if has_topology else "unavailable",
                "accessibility": "unavailable",
                "traffic_assignment": (
                    "available" if has_topology and has_flow and has_capacity
                    else "conditional" if has_topology else "unavailable"
                ),
                "resource_optimisation": (
                    "available" if has_topology and has_repair_cost and has_repair_duration
                    else "conditional" if has_topology else "unavailable"
                ),
                "stochastic_reliability": "available" if has_topology and has_failure_probability else "conditional" if has_topology else "unavailable",
                "disruption_propagation": "available" if has_complete_topology_impedance else "conditional" if has_topology else "unavailable",
                "route_detour_analysis": "available" if has_complete_topology_impedance else "conditional" if has_topology else "unavailable",
            },
            "source_files": source_files,
            "mapping": {
                "default": mapping,
                "by_file": {
                    item["name"]: item["resolved_mapping"]
                    for item in source_files
                },
            },
        }
        connection.execute(
            "INSERT INTO metadata(key, value_json) VALUES ('summary', ?)", (json.dumps(summary),)
        )
        connection.commit()
        callback(100.0, "Project is ready")
        return summary
    finally:
        connection.close()


def read_project_summary(database_path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute("SELECT value_json FROM metadata WHERE key = 'summary'").fetchone()
    if row is None:
        raise ValueError("Project metadata is unavailable.")
    return json.loads(row[0])
