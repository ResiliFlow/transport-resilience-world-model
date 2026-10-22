from __future__ import annotations

import csv
from contextlib import closing
from datetime import datetime
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import networkx as nx
from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.core.storage import Repository
from engine.importers import import_csv_project
from engine.models import (
    closure_scenario,
    critical_components,
    map_features,
    persistent_degradation,
    restoration_screening,
    spatial_hotspots,
    temporal_resilience_curve,
)


BASE_FIELDS = [
    "id",
    "from",
    "to",
    "length_m",
    "lat",
    "lon",
    "datetime",
    "step",
    "speed_kmh",
    "travel_time_min",
    "region_id",
    "impact",
]

BASE_MAPPING: dict[str, str | None] = {
    "entity_id": "id",
    "from_node": "from",
    "to_node": "to",
    "length": "length_m",
    "latitude": "lat",
    "longitude": "lon",
    "time": "datetime",
    "time_step": "step",
    "speed": "speed_kmh",
    "travel_time": "travel_time_min",
    "region": "region_id",
    "primary_metric": "impact",
}


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    selected_fields = fields or BASE_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in selected_fields})


def import_rows(
    root: Path,
    name: str,
    rows: list[dict[str, object]],
    *,
    mapping: dict[str, str | None] | None = None,
    timezone: str = "UTC",
    time_step_seconds: float = 300.0,
    topology_node_scope: str = "global",
    primary_metric_direction: str = "descriptive",
    primary_metric_comparison: str = "current_value",
    primary_metric_scale: float = 1.0,
) -> tuple[Path, dict]:
    csv_path = root / f"{name}.csv"
    database = root / f"{name}.sqlite"
    write_rows(csv_path, rows)
    confirmed = dict(BASE_MAPPING if mapping is None else mapping)
    summary = import_csv_project(
        files=[csv_path],
        mapping=confirmed,
        file_mappings={csv_path.name: confirmed},
        units={"speed": "km/h", "travel_time": "min", "primary_metric": "index"},
        timezone=timezone,
        time_step_seconds=time_step_seconds,
        topology_node_scope=topology_node_scope,
        entity_id_scope="per_region",
        primary_metric_direction=primary_metric_direction,
        primary_metric_comparison=primary_metric_comparison,
        primary_metric_scale=primary_metric_scale,
        database_path=database,
        project_id=f"oracle_{name}",
        project_name=f"Oracle {name}",
        hazard_type="other",
        source="independent_release_regression_oracle",
    )
    return database, summary


def oracle_efficiency(graph: nx.MultiDiGraph) -> float:
    """Recompute directed global efficiency without using Prototype 5 helpers."""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return 0.0
    reciprocal_sum = 0.0
    for source in nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        reciprocal_sum += sum(
            1.0 / distance
            for target, distance in lengths.items()
            if target != source and distance > 0
        )
    return reciprocal_sum / (len(nodes) * (len(nodes) - 1))


class ImportAndScientificOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="p5_release_oracle_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_zero_speed_is_retained_as_a_valid_complete_disruption(self) -> None:
        rows = [
            {
                "id": "z1", "lat": -33.86, "lon": 151.20,
                "datetime": "2021-01-01 00:00:00", "speed_kmh": 60,
                "region_id": "r",
            },
            {
                "id": "z1", "lat": -33.86, "lon": 151.20,
                "datetime": "2021-01-01 00:10:00", "speed_kmh": 0,
                "region_id": "r",
            },
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        database, summary = import_rows(self.root, "zero_speed", rows, mapping=mapping)

        self.assertEqual(summary["quality"]["zero_speeds_observed"], 1)
        self.assertEqual(summary["quality"]["negative_speeds_invalid"], 0)
        with closing(sqlite3.connect(database)) as connection:
            speed, quality_flags = connection.execute(
                "SELECT speed, quality_flags FROM observations ORDER BY timestamp_epoch DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(speed, 0.0)
        self.assertEqual(quality_flags & 4, 0, "zero speed must not receive the negative-speed bit")

        latest = map_features(database, metric="speed", timestamp=summary["time_values"][-1], limit=100)
        self.assertEqual(latest["features"][0]["properties"]["value"], 0.0)
        self.assertEqual(latest["features"][0]["properties"]["severity"], 1.0)
        curve = temporal_resilience_curve(database, metric="speed", bbox=None, time_start=None, time_end=None)
        self.assertAlmostEqual(curve["items"][1]["performance"], 0.0, places=12)
        self.assertAlmostEqual(float(curve["headline_effects"][0]["value"]), 0.5, places=12)

    def test_missing_final_cohort_does_not_fabricate_recovery(self) -> None:
        rows = []
        for entity, lat in (("a", -33.86), ("b", -33.87)):
            rows.extend(
                [
                    {"id": entity, "lat": lat, "lon": 151.20, "datetime": "2021-01-01 00:00:00", "speed_kmh": 100, "region_id": "r"},
                    {"id": entity, "lat": lat, "lon": 151.20, "datetime": "2021-01-01 00:10:00", "speed_kmh": 50, "region_id": "r"},
                ]
            )
        rows.append(
            {"id": "a", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01 00:20:00", "speed_kmh": 100, "region_id": "r"}
        )
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        database, _summary = import_rows(self.root, "missing_recovery", rows, mapping=mapping)

        curve = temporal_resilience_curve(database, metric="speed", bbox=None, time_start=None, time_end=None)
        # Independent oracle: only the fully observed first interval is eligible.
        # Trapezoidal loss is (0 + 0.5) / 2 = 0.25, hence resilience = 0.75.
        self.assertAlmostEqual(curve["evidence"]["observed_duration_seconds"], 600.0, places=12)
        self.assertAlmostEqual(curve["evidence"]["excluded_coverage_seconds"], 600.0, places=12)
        self.assertAlmostEqual(float(curve["headline_effects"][0]["value"]), 0.75, places=12)
        self.assertEqual(curve["headline_effects"][2]["value"], "Not inferred")
        self.assertEqual(curve["items"][-1]["coverage_ratio"], 0.5)
        self.assertTrue(curve["items"][-1]["gap_before"])

    def test_all_missing_last_timestamp_is_retained_and_cannot_claim_recovery(self) -> None:
        rows = []
        for entity, latitude in (("a", -33.86), ("b", -33.87)):
            rows.extend(
                [
                    {"id": entity, "lat": latitude, "lon": 151.20, "datetime": "2021-01-01 00:00:00", "speed_kmh": 100, "region_id": "r"},
                    {"id": entity, "lat": latitude, "lon": 151.20, "datetime": "2021-01-01 00:10:00", "speed_kmh": 50, "region_id": "r"},
                    {"id": entity, "lat": latitude, "lon": 151.20, "datetime": "2021-01-01 00:20:00", "speed_kmh": 80, "region_id": "r"},
                    {"id": entity, "lat": latitude, "lon": 151.20, "datetime": "2021-01-01 00:30:00", "speed_kmh": "", "region_id": "r"},
                ]
            )
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        database, summary = import_rows(self.root, "all_missing_final", rows, mapping=mapping)
        curve = temporal_resilience_curve(database, metric="speed", bbox=None, time_start=None, time_end=None)

        self.assertEqual(len(curve["items"]), 4)
        self.assertEqual(curve["items"][-1]["time"], summary["time_values"][-1])
        self.assertIsNone(curve["items"][-1]["performance"])
        self.assertEqual(curve["items"][-1]["valid_entity_count"], 0)
        self.assertEqual(curve["items"][-1]["coverage_ratio"], 0.0)
        self.assertTrue(curve["items"][-1]["gap_before"])
        self.assertAlmostEqual(curve["evidence"]["observed_duration_seconds"], 1200.0, places=12)
        self.assertAlmostEqual(curve["evidence"]["excluded_coverage_seconds"], 600.0, places=12)
        self.assertEqual(curve["headline_effects"][2]["value"], "Not inferred")

    def test_time_step_is_scaled_to_seconds_and_preserved(self) -> None:
        rows = [
            {"id": "s1", "lat": 27.0, "lon": -81.0, "step": step, "speed_kmh": 50 - step * 5, "region_id": "r"}
            for step in (0, 1, 2)
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time": None, "travel_time": None, "primary_metric": None})
        database, summary = import_rows(
            self.root, "time_step", rows, mapping=mapping, time_step_seconds=900.0
        )
        self.assertEqual(summary["time_values"], [0, 900, 1800])
        self.assertEqual(summary["expected_interval_s"], 900)
        self.assertEqual(summary["time_labels"], {"0": "Step 0", "900": "Step 1", "1800": "Step 2"})
        with closing(sqlite3.connect(database)) as connection:
            stored = connection.execute(
                "SELECT timestamp_epoch, time_step FROM observations ORDER BY timestamp_epoch"
            ).fetchall()
        self.assertEqual(stored, [(0, 0.0), (900, 1.0), (1800, 2.0)])

    def test_two_time_steps_include_step_zero_in_expected_interval(self) -> None:
        rows = [
            {"id": "s1", "lat": 27.0, "lon": -81.0, "step": 0, "speed_kmh": 100, "region_id": "r"},
            {"id": "s1", "lat": 27.0, "lon": -81.0, "step": 1, "speed_kmh": 50, "region_id": "r"},
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time": None, "travel_time": None, "primary_metric": None})
        database, summary = import_rows(
            self.root, "two_time_steps", rows, mapping=mapping, time_step_seconds=600.0
        )
        self.assertEqual(summary["time_values"], [0, 600])
        self.assertEqual(summary["expected_interval_s"], 600)
        persistence = persistent_degradation(
            database,
            metric="speed",
            threshold=0.2,
            top_k=1,
            bbox=None,
            time_start=None,
            time_end=None,
        )
        self.assertEqual(len(persistence["items"]), 1)
        self.assertAlmostEqual(persistence["items"][0]["observed_hours"], 1 / 6, places=12)
        self.assertEqual(persistence["evidence"]["expected_interval_s"], 600)

    def test_iana_timezone_localises_naive_time_and_preserves_explicit_offset(self) -> None:
        rows = [
            {"id": "t1", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01 00:00:00", "speed_kmh": 60, "region_id": "r"},
            {"id": "t1", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01 01:00:00", "speed_kmh": 55, "region_id": "r"},
            {"id": "t1", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01T02:00:00+11:00", "speed_kmh": 50, "region_id": "r"},
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        _database, summary = import_rows(
            self.root, "timezone", rows, mapping=mapping, timezone="Australia/Sydney"
        )
        zone = ZoneInfo("Australia/Sydney")
        expected = [
            int(datetime(2021, 1, 1, hour, tzinfo=zone).timestamp())
            for hour in (0, 1, 2)
        ]
        self.assertEqual(summary["time_values"], expected)
        self.assertEqual(summary["expected_interval_s"], 3600)
        self.assertEqual(summary["time_start"], "2020-12-31T13:00:00Z")
        self.assertEqual(summary["time_end"], "2020-12-31T15:00:00Z")
        self.assertIn("localised", summary["timezone_application"])

    def test_parenthetical_timezone_disclosure_is_not_executed_as_iana(self) -> None:
        rows = [
            {"id": "legacy", "lat": 27.0, "lon": -81.0, "datetime": "2021-01-01 00:00:00", "speed_kmh": 60, "region_id": "r"},
            {"id": "legacy", "lat": 27.0, "lon": -81.0, "datetime": "2021-01-01 01:00:00", "speed_kmh": 55, "region_id": "r"},
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        _database, summary = import_rows(
            self.root,
            "parenthetical_timezone",
            rows,
            mapping=mapping,
            timezone="America/New_York (source convention; not converted)",
        )
        expected_start = int(datetime(2021, 1, 1, 0, 0, tzinfo=ZoneInfo("UTC")).timestamp())
        self.assertEqual(summary["time_values"], [expected_start, expected_start + 3600])
        self.assertEqual(summary["time_start"], "2021-01-01T00:00:00Z")
        self.assertIn("No executable IANA zone", summary["timezone_application"])

    def test_global_topology_nodes_connect_across_files(self) -> None:
        first = self.root / "west.csv"
        second = self.root / "east.csv"
        rows_by_file = {
            first: [{"id": "e1", "from": "A", "to": "B", "length_m": 1000, "lat": 0.0, "lon": 10.0, "travel_time_min": 1, "region_id": "west"}],
            second: [{"id": "e2", "from": "B", "to": "C", "length_m": 1000, "lat": 0.1, "lon": 10.1, "travel_time_min": 1, "region_id": "east"}],
        }
        mapping = dict(BASE_MAPPING)
        mapping.update({"time": None, "time_step": None, "speed": None, "primary_metric": None})
        for path, rows in rows_by_file.items():
            write_rows(path, rows)
        database = self.root / "global_topology.sqlite"
        summary = import_csv_project(
            files=[first, second],
            mapping=mapping,
            file_mappings={first.name: mapping, second.name: mapping},
            units={"travel_time": "min"},
            timezone="UTC",
            topology_node_scope="global",
            entity_id_scope="per_region",
            database_path=database,
            project_id="oracle_global_topology",
            project_name="Global topology oracle",
            hazard_type="other",
            source="independent_release_regression_oracle",
        )
        self.assertEqual(summary["topology_node_scope"], "global")
        result = critical_components(database, top_k=2, bbox=None)
        self.assertEqual(result["evidence"]["topology_node_scope"], "global")
        self.assertEqual(result["evidence"]["total_edge_count"], 2)
        self.assertEqual(result["headline_effects"][2]["value"], "3 nodes")
        self.assertTrue(result["evidence"]["exhaustive_edge_removal"])

    def test_global_entity_scope_merges_same_id_despite_source_regions(self) -> None:
        first = self.root / "global_entity_first.csv"
        second = self.root / "global_entity_second.csv"
        write_rows(
            first,
            [{"id": "shared", "lat": 27.0, "lon": -81.0, "datetime": "2021-01-01 00:00:00", "speed_kmh": 100, "region_id": "west"}],
        )
        write_rows(
            second,
            [{"id": "shared", "lat": 27.0, "lon": -81.0, "datetime": "2021-01-01 00:10:00", "speed_kmh": 50, "region_id": "east"}],
        )
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        database = self.root / "global_entity_scope.sqlite"
        summary = import_csv_project(
            files=[first, second],
            mapping=mapping,
            file_mappings={first.name: mapping, second.name: mapping},
            units={"speed": "km/h"},
            timezone="UTC",
            topology_node_scope="per_region",
            entity_id_scope="global",
            database_path=database,
            project_id="oracle_global_entity",
            project_name="Global entity oracle",
            hazard_type="other",
            source="independent_release_regression_oracle",
        )
        self.assertEqual(summary["entity_id_scope"], "global")
        self.assertEqual(summary["entity_count"], 1)
        self.assertEqual(summary["region_count"], 1)
        self.assertEqual(summary["observation_count"], 2)
        with closing(sqlite3.connect(database)) as connection:
            entities = connection.execute(
                "SELECT entity_pk, region_id, entity_id FROM entities"
            ).fetchall()
            observations = connection.execute(
                "SELECT DISTINCT entity_pk FROM observations"
            ).fetchall()
        self.assertEqual(entities, [(1, "global", "shared")])
        self.assertEqual(observations, [(1,)])

    def _build_small_topology(self) -> tuple[Path, list[dict[str, object]]]:
        edges: list[dict[str, object]] = [
            {"id": "e1", "from": "A", "to": "B", "travel_time_min": 1.0, "length_m": 1000, "lat": 1.00, "lon": 10.00, "region_id": "r"},
            {"id": "e2", "from": "B", "to": "C", "travel_time_min": 1.0, "length_m": 1000, "lat": 1.01, "lon": 10.01, "region_id": "r"},
            {"id": "e3", "from": "A", "to": "C", "travel_time_min": 5.0, "length_m": 1000, "lat": 1.02, "lon": 10.02, "region_id": "r"},
            {"id": "e4", "from": "C", "to": "D", "travel_time_min": 1.0, "length_m": 1000, "lat": 1.03, "lon": 10.03, "region_id": "r"},
            {"id": "e5", "from": "B", "to": "D", "travel_time_min": 5.0, "length_m": 1000, "lat": 1.04, "lon": 10.04, "region_id": "r"},
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"time": None, "time_step": None, "speed": None, "primary_metric": None})
        database, _summary = import_rows(self.root, "small_topology", edges, mapping=mapping)
        return database, edges

    def test_small_graph_critical_links_match_exhaustive_external_oracle(self) -> None:
        database, edges = self._build_small_topology()
        graph = nx.MultiDiGraph()
        for edge in edges:
            graph.add_edge(
                str(edge["from"]), str(edge["to"]),
                key=f"r::{edge['id']}", weight=float(edge["travel_time_min"]),
            )
        baseline = oracle_efficiency(graph)
        expected_losses: dict[str, float] = {}
        for u, v, key in list(graph.edges(keys=True)):
            attributes = dict(graph[u][v][key])
            graph.remove_edge(u, v, key)
            expected_losses[str(key)] = max(0.0, (baseline - oracle_efficiency(graph)) / baseline)
            graph.add_edge(u, v, key=key, **attributes)

        result = critical_components(database, top_k=len(edges), bbox=None)
        self.assertTrue(result["evidence"]["exhaustive_edge_removal"])
        self.assertEqual(result["evidence"]["candidate_coverage"], 1.0)
        self.assertEqual(result["evidence"]["evaluated_candidate_count"], len(edges))
        actual = {item["key"]: item["efficiency_loss"] for item in result["items"]}
        self.assertEqual(set(actual), set(expected_losses))
        for key, expected in expected_losses.items():
            self.assertAlmostEqual(actual[key], expected, places=12, msg=key)
        expected_top = max(expected_losses, key=expected_losses.get)
        self.assertEqual(result["items"][0]["key"], expected_top)

    def test_unknown_or_out_of_scope_links_are_strictly_rejected(self) -> None:
        database, _edges = self._build_small_topology()
        for model in (closure_scenario, restoration_screening):
            with self.subTest(model=model.__name__, case="unknown"):
                with self.assertRaisesRegex(ValueError, "not available"):
                    model(database, selected_keys=["r::unknown"], bbox=None)
            with self.subTest(model=model.__name__, case="outside_bbox"):
                with self.assertRaisesRegex(ValueError, "not available"):
                    model(database, selected_keys=["r::e1"], bbox=[100, 40, 101, 41])

    def test_primary_metric_zero_reference_respects_both_directions(self) -> None:
        for direction, disrupted_value in (("higher_worse", 2.0), ("lower_worse", -2.0)):
            with self.subTest(direction=direction):
                rows = []
                for index in range(3):
                    rows.extend(
                        [
                            {"id": f"p{index}", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01 00:00:00", "impact": 0.0, "region_id": "r"},
                            {"id": f"p{index}", "lat": -33.86, "lon": 151.20, "datetime": "2021-01-01 00:10:00", "impact": disrupted_value, "region_id": "r"},
                        ]
                    )
                mapping = dict(BASE_MAPPING)
                mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "speed": None, "travel_time": None})
                database, summary = import_rows(
                    self.root,
                    f"primary_{direction}",
                    rows,
                    mapping=mapping,
                    primary_metric_direction=direction,
                    primary_metric_comparison="change_from_reference",
                    primary_metric_scale=2.0,
                )
                with closing(sqlite3.connect(database)) as connection:
                    baselines = [row[0] for row in connection.execute("SELECT baseline_primary FROM entities")]
                self.assertEqual(baselines, [0.0, 0.0, 0.0])
                result = spatial_hotspots(
                    database, metric="primary", top_k=5, bbox=None,
                    timestamp=summary["time_values"][-1],
                )
                self.assertAlmostEqual(result["items"][0]["mean_degradation"], 1.0, places=12)
                self.assertEqual(result["items"][0]["entity_count"], 3)
                self.assertEqual(result["evidence"]["primary_metric_semantics"]["direction"], direction)

    def test_hotspot_uses_a_fixed_five_kilometre_metric_cell(self) -> None:
        longitude, latitude = 151.20, -33.86
        rows = []
        for index in range(3):
            rows.extend(
                [
                    {"id": f"h{index}", "lat": latitude, "lon": longitude, "datetime": "2021-01-01 00:00:00", "speed_kmh": 100, "region_id": "r"},
                    {"id": f"h{index}", "lat": latitude, "lon": longitude, "datetime": "2021-01-01 00:10:00", "speed_kmh": 50, "region_id": "r"},
                ]
            )
        mapping = dict(BASE_MAPPING)
        mapping.update({"from_node": None, "to_node": None, "length": None, "time_step": None, "travel_time": None, "primary_metric": None})
        database, summary = import_rows(self.root, "fixed_hotspot", rows, mapping=mapping)
        first = spatial_hotspots(database, metric="speed", top_k=5, bbox=None, timestamp=summary["time_values"][0])
        second = spatial_hotspots(
            database, metric="speed", top_k=5,
            bbox=[151.1, -34.0, 151.3, -33.8], timestamp=summary["time_values"][-1],
        )

        radius = 6_378_137.0
        world_half = math.pi * radius
        x = radius * math.radians(longitude)
        bounded_latitude = max(-85.05112878, min(85.05112878, latitude))
        y = radius * math.log(math.tan(math.pi / 4 + math.radians(bounded_latitude) / 2))
        column = math.floor((x + world_half) / 5000.0)
        line = math.floor((y + world_half) / 5000.0)
        expected_cell = f"M5K-{column}-{line}"
        self.assertEqual(first["items"][0]["cell_id"], expected_cell)
        self.assertEqual(second["items"][0]["cell_id"], expected_cell)
        self.assertEqual(first["feature_collection"]["features"][0]["geometry"], second["feature_collection"]["features"][0]["geometry"])
        self.assertEqual(second["evidence"]["cell_size_m"], 5000.0)
        self.assertEqual(second["evidence"]["minimum_entity_support"], 3)
        self.assertAlmostEqual(second["items"][0]["mean_degradation"], 0.5, places=12)


class AsyncAPIAndExportOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="p5_api_oracle_")
        self.root = Path(self.temporary.name)
        rows = [
            {"id": "e1", "from": "A", "to": "B", "travel_time_min": 1.0, "length_m": 1000, "lat": 1.00, "lon": 10.00, "region_id": "r"},
            {"id": "e2", "from": "B", "to": "C", "travel_time_min": 1.0, "length_m": 1000, "lat": 1.01, "lon": 10.01, "region_id": "r"},
            {"id": "e3", "from": "A", "to": "C", "travel_time_min": 5.0, "length_m": 1000, "lat": 1.02, "lon": 10.02, "region_id": "r"},
        ]
        mapping = dict(BASE_MAPPING)
        mapping.update({"time": None, "time_step": None, "speed": None, "primary_metric": None})
        self.database, self.summary = import_rows(self.root, "api_topology", rows, mapping=mapping)
        self.repository = Repository(self.root / "metadata.sqlite")
        self.repository.initialise()
        self.project_id = "api_oracle_project"
        self.repository.save_project(
            project_id=self.project_id,
            name="API oracle project",
            hazard_type="other",
            source="independent_release_regression_oracle",
            status="ready",
            database_path=str(self.database.resolve()),
            mapping=mapping,
            summary=self.summary,
        )
        self.repository_patch = patch.object(api_main, "repository", self.repository)
        self.repository_patch.start()
        self.client = TestClient(api_main.app)

    def tearDown(self) -> None:
        self.client.close()
        self.repository_patch.stop()
        self.temporary.cleanup()

    def _payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "analysis_type": "critical_components",
            "metric": "travel_time",
            "threshold": 0.2,
            "top_k": 3,
            "selected_keys": [],
            "bbox": None,
            "time_start": None,
            "time_end": None,
            "command_text": "Independent async release regression",
            "assumptions": {},
        }

    def _wait_for_terminal_job(self, job_id: str, timeout: float = 8.0) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            last = response.json()
            if last["status"] in {"completed", "failed", "cancelled"}:
                return last
            time.sleep(0.01)
        self.fail(f"job did not finish: {last}")

    def test_async_analysis_completes_verifies_and_is_saved_once(self) -> None:
        response = self.client.post("/api/analysis-jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        job = self._wait_for_terminal_job(response.json()["job_id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(job["progress"], 100.0)
        result = job["result"]
        self.assertEqual(result["analysis_type"], "critical_components")
        self.assertEqual(result["project_id"], self.project_id)
        self.assertGreaterEqual(result["runtime_seconds"], 0.0)
        self.assertEqual(len(result["provenance"]["analysis_source_sha256"]), 64)
        runs = self.repository.list_runs(project_id=self.project_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], result["run_id"])

    def test_disruption_scenario_crud_is_grounded_recomputed_and_project_scoped(self) -> None:
        target_response = self.client.get(
            f"/api/projects/{self.project_id}/scenario-targets",
            params={"bbox": "9.9,0.9,10.1,1.1"},
        )
        self.assertEqual(target_response.status_code, 200, target_response.text)
        self.assertEqual(set(target_response.json()["keys"]), {"r::e1", "r::e2", "r::e3"})

        payload = {
            "project_id": self.project_id,
            "name": "Bridge closure and slower approach",
            "hazard_type": "flood",
            "base_timestamp": None,
            "changes": [
                {"key": "r::e2", "effect": "closed", "magnitude": 1.0},
                {"key": "r::e3", "effect": "travel_time_increase", "magnitude": 0.5},
            ],
        }
        created_response = self.client.post("/api/scenarios", json=payload)
        self.assertEqual(created_response.status_code, 201, created_response.text)
        created = created_response.json()
        scenario_id = created["scenario_id"]
        self.assertEqual(created["result"]["scenario_id"], scenario_id)
        self.assertTrue(created["result"]["verification"]["passed"])
        self.assertEqual(created["result"]["randomness"], "none")
        self.assertLessEqual(
            created["result"]["evidence"]["scenario_efficiency"],
            created["result"]["evidence"]["baseline_efficiency"],
        )

        payload["name"] = "Updated deterministic flood scenario"
        payload["changes"] = [{"key": "r::e1", "effect": "speed_reduction", "magnitude": 0.4}]
        updated_response = self.client.post(f"/api/scenarios/{scenario_id}", json=payload)
        self.assertEqual(updated_response.status_code, 200, updated_response.text)
        updated = updated_response.json()
        self.assertEqual(updated["scenario_id"], scenario_id)
        self.assertEqual(updated["name"], payload["name"])
        self.assertEqual(updated["result"]["affected_count"], 1)

        listed = self.client.get("/api/scenarios", params={"project_id": self.project_id})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["scenario_id"] for item in listed.json()], [scenario_id])
        deleted = self.client.delete(f"/api/scenarios/{scenario_id}")
        self.assertEqual(deleted.status_code, 204, deleted.text)
        self.assertEqual(self.repository.list_scenarios(self.project_id), [])

    def test_route_api_accepts_project_owned_scenario_ids_and_returns_each_state(self) -> None:
        harmless_id = self.repository.save_scenario(
            project_id=self.project_id,
            name="Unrelated direct link closure",
            hazard_type="flood",
            base_timestamp=None,
            changes=[{"key": "r::e3", "effect": "closed", "magnitude": 1.0}],
            result={"verification": {"passed": True}},
        )
        endpoint_id = self.repository.save_scenario(
            project_id=self.project_id,
            name="Destination link closure",
            hazard_type="flood",
            base_timestamp=None,
            changes=[{"key": "r::e2", "effect": "closed", "magnitude": 1.0}],
            result={"verification": {"passed": True}},
        )
        payload = {
            "project_id": self.project_id,
            "analysis_type": "route_detour_analysis",
            "model_id": "P5-M23",
            "metric": "travel_time",
            "threshold": 0.2,
            "top_k": 1,
            "selected_keys": ["r::e1", "r::e2"],
            "bbox": None,
            "time_start": None,
            "time_end": None,
            "command_text": "Compare all saved scenario routes between two points.",
            "assumptions": {
                "origin_key": "r::e1",
                "destination_key": "r::e2",
                "scenario_ids": [harmless_id, endpoint_id],
            },
        }
        response = self.client.post("/api/analyses", json=payload)
        self.assertEqual(response.status_code, 201, response.text)
        result = response.json()
        self.assertEqual(result["visual"]["mode"], "route_scenario_compare")
        self.assertEqual(len(result["visual"]["variants"]), 3)
        by_id = {item["id"]: item for item in result["visual"]["variants"]}
        self.assertTrue(by_id["historical"]["reachable"])
        self.assertTrue(by_id[harmless_id]["reachable"])
        self.assertFalse(by_id[endpoint_id]["reachable"])
        self.assertTrue(result["verification"]["passed"])

    def test_cancelled_async_analysis_never_saves_a_completed_run(self) -> None:
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_execute(_project: dict, _payload: dict, *, save: bool = True) -> dict:
            self.assertFalse(save)
            started.set()
            release.wait(timeout=5.0)
            finished.set()
            return {
                "run_id": "run_should_not_be_saved",
                "verification": {"passed": True},
                "model": {"id": "P5-M2"},
                "items": [],
                "feature_collection": {"type": "FeatureCollection", "features": []},
            }

        with patch.object(api_main, "_execute_analysis", side_effect=blocking_execute):
            response = self.client.post("/api/analysis-jobs", json=self._payload())
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["job_id"]
            self.assertTrue(started.wait(timeout=3.0), "analysis worker did not begin")
            cancelled = self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 202)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            release.set()
            self.assertTrue(finished.wait(timeout=3.0), "analysis worker did not leave the blocked model")
            time.sleep(0.05)
        stored_job = self.repository.get_job(job_id)
        self.assertIsNotNone(stored_job)
        self.assertEqual(stored_job["status"], "cancelled")
        self.assertEqual(self.repository.list_runs(project_id=self.project_id), [])

    def test_cancel_wins_after_model_finishes_but_before_atomic_completion(self) -> None:
        completion_entered = threading.Event()
        release_completion = threading.Event()
        completion_returned = threading.Event()
        completion_outcomes: list[bool] = []
        original_complete = self.repository.complete_analysis_job

        def gated_complete(*args, **kwargs) -> bool:
            # This gate is deliberately after the deterministic model has
            # returned, at the exact old check-then-save race boundary.
            completion_entered.set()
            if not release_completion.wait(timeout=5.0):
                raise TimeoutError("test did not release atomic completion")
            try:
                outcome = original_complete(*args, **kwargs)
                completion_outcomes.append(outcome)
                return outcome
            finally:
                completion_returned.set()

        with patch.object(self.repository, "complete_analysis_job", side_effect=gated_complete):
            response = self.client.post("/api/analysis-jobs", json=self._payload())
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["job_id"]
            self.assertTrue(
                completion_entered.wait(timeout=5.0),
                "worker did not reach the post-model atomic-completion boundary",
            )
            cancelled = self.client.post(f"/api/jobs/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 202)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            release_completion.set()
            self.assertTrue(completion_returned.wait(timeout=5.0))

        self.assertEqual(completion_outcomes, [False])
        stored_job = self.repository.get_job(job_id)
        self.assertIsNotNone(stored_job)
        self.assertEqual(stored_job["status"], "cancelled")
        self.assertIsNone(stored_job["result"])
        self.assertEqual(self.repository.list_runs(project_id=self.project_id), [])

    def test_research_json_csv_and_geojson_exports_are_consistent(self) -> None:
        response = self.client.post("/api/analysis-jobs", json=self._payload())
        self.assertEqual(response.status_code, 202)
        job = self._wait_for_terminal_job(response.json()["job_id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        result = job["result"]
        run_id = result["run_id"]

        manifest_response = self.client.get(f"/api/runs/{run_id}/export.json")
        self.assertEqual(manifest_response.status_code, 200)
        manifest = manifest_response.json()
        self.assertEqual(manifest["schema"], "prototype5.research-run.v1")
        self.assertEqual(manifest["run"]["run_id"], run_id)
        self.assertEqual(manifest["project"]["project_id"], self.project_id)
        self.assertEqual(
            manifest["software"]["analysis_source_sha256"],
            result["provenance"]["analysis_source_sha256"],
        )
        self.assertTrue(manifest["software"]["source_digest_frozen_at_run"])
        self.assertEqual(len(manifest["project"]["summary"]["source_files"][0]["sha256"]), 64)
        self.assertEqual(manifest["model_record"]["id"], result["model"]["id"])
        for dependency in ("fastapi", "pydantic", "pandas", "numpy", "networkx"):
            self.assertIn(dependency, manifest["software"]["dependencies"])

        csv_response = self.client.get(f"/api/runs/{run_id}/export.csv")
        self.assertEqual(csv_response.status_code, 200)
        csv_rows = list(csv.DictReader(io.StringIO(csv_response.text)))
        self.assertEqual(len(csv_rows), len(result["items"]))
        self.assertEqual({row["key"] for row in csv_rows}, {item["key"] for item in result["items"]})
        self.assertTrue(all(row["efficiency_loss"] != "" for row in csv_rows))

        geojson_response = self.client.get(f"/api/runs/{run_id}/export.geojson")
        self.assertEqual(geojson_response.status_code, 200)
        collection = geojson_response.json()
        self.assertEqual(collection["type"], "FeatureCollection")
        self.assertEqual(collection, result["feature_collection"])
        self.assertEqual(len(collection["features"]), len(result["items"]))


if __name__ == "__main__":
    unittest.main()
