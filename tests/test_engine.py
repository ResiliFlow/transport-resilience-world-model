from __future__ import annotations

import csv
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import networkx as nx

from engine.importers import import_csv_project, suggest_mapping
from engine.command_parser import interpret_command
from engine.models import (
    area_summary,
    closure_scenario,
    critical_components,
    entity_history,
    map_features,
    persistent_degradation,
    progressive_robustness,
    restoration_screening,
    spatial_hotspots,
    temporal_resilience_curve,
    run_analysis,
    _single_connection_bridge_pairs,
)
from engine.network_mapping import build_network_mapping, network_features, network_mapping_status
from engine.scenarios import evaluate_disruption_scenario, select_scenario_targets
from engine.adaptive_models import run_adaptive_model
from engine.model_router import recommend_models


TRANSPORT_COLUMNS = [
    "id", "from", "to", "length_m", "lat", "lon", "datetime",
    "speed_kmh", "travel_time_min", "time_step", "region_id",
]

RENAMED_COLUMNS = [
    "Road Link Code", "Origin Node ID", "Destination Node ID",
    "Segment Length Metres", "Centre Latitude", "Centre Longitude",
    "Recorded At", "Mean Velocity KPH", "Journey Duration Minutes",
    "Simulation Step", "District Name", "Flood Depth m",
]


class TopologyInvariantTests(unittest.TestCase):
    def test_single_connection_bridge_excludes_parallel_or_reverse_roads(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("u", "v", key="forward", weight=1.0)
        graph.add_edge("v", "u", key="reverse", weight=1.0)
        graph.add_edge("v", "w", key="single", weight=1.0)
        bridges = _single_connection_bridge_pairs(graph)
        self.assertNotIn(frozenset(("u", "v")), bridges)
        self.assertIn(frozenset(("v", "w")), bridges)


class SemanticMappingTests(unittest.TestCase):
    def test_transport_schema_is_recognised(self) -> None:
        result = suggest_mapping(TRANSPORT_COLUMNS)
        self.assertEqual(result["roles"]["entity_id"], "id")
        self.assertEqual(result["roles"]["travel_time"], "travel_time_min")
        self.assertEqual(result["roles"]["time"], "datetime")
        self.assertTrue(all(result["scores"][role] >= 0.9 for role in ("entity_id", "latitude", "longitude")))

    def test_human_readable_alternative_schema_is_recognised(self) -> None:
        result = suggest_mapping(RENAMED_COLUMNS)
        expected = {
            "entity_id": "Road Link Code",
            "from_node": "Origin Node ID",
            "to_node": "Destination Node ID",
            "length": "Segment Length Metres",
            "latitude": "Centre Latitude",
            "longitude": "Centre Longitude",
            "time": "Recorded At",
            "speed": "Mean Velocity KPH",
            "travel_time": "Journey Duration Minutes",
            "time_step": "Simulation Step",
            "region": "District Name",
            "primary_metric": "Flood Depth m",
        }
        for role, column in expected.items():
            self.assertEqual(result["roles"][role], column, role)

    def test_junction_and_asset_schema_is_recognised(self) -> None:
        result = suggest_mapping([
            "road_asset",
            "upstream_junction",
            "downstream_junction",
            "distance_metres",
            "centre_lat",
            "centre_lon",
            "observation_time",
            "journey_minutes",
            "velocity_kph",
        ])
        expected = {
            "entity_id": "road_asset",
            "from_node": "upstream_junction",
            "to_node": "downstream_junction",
            "length": "distance_metres",
            "latitude": "centre_lat",
            "longitude": "centre_lon",
            "time": "observation_time",
            "travel_time": "journey_minutes",
            "speed": "velocity_kph",
        }
        for role, column in expected.items():
            self.assertEqual(result["roles"][role], column, role)


class ImportAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="prototype5_test_")
        self.root = Path(self.temporary.name)
        self.first = self.root / "template_schema.csv"
        self.second = self.root / "renamed_schema.csv"
        times = ["2021-08-01 00:00:00", "2021-08-01 00:10:00", "2021-08-01 00:20:00"]

        with self.first.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRANSPORT_COLUMNS)
            writer.writeheader()
            for link, from_node, to_node, lat, lon, speeds in (
                ("a1", "1", "2", 27.00, -81.00, [60, 42, 30]),
                ("a2", "2", "3", 27.05, -80.95, [55, 44, 28]),
            ):
                for step, (timestamp, speed) in enumerate(zip(times, speeds)):
                    writer.writerow(
                        {
                            "id": link,
                            "from": from_node,
                            "to": to_node,
                            "length_m": 1000,
                            "lat": lat,
                            "lon": lon,
                            "datetime": timestamp,
                            "speed_kmh": speed,
                            "travel_time_min": 60 / speed,
                            "time_step": step,
                            "region_id": "alpha",
                        }
                    )

        with self.second.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RENAMED_COLUMNS)
            writer.writeheader()
            for link, from_node, to_node, lat, lon, speeds in (
                ("b1", "1", "2", 28.00, -80.50, [50, 40, 25]),
                ("b2", "2", "3", 28.05, -80.45, [48, 38, 24]),
            ):
                for step, (timestamp, speed) in enumerate(zip(times, speeds)):
                    writer.writerow(
                        {
                            "Road Link Code": link,
                            "Origin Node ID": from_node,
                            "Destination Node ID": to_node,
                            "Segment Length Metres": 900,
                            "Centre Latitude": lat,
                            "Centre Longitude": lon,
                            "Recorded At": timestamp,
                            "Mean Velocity KPH": speed,
                            "Journey Duration Minutes": 54 / speed,
                            "Simulation Step": step,
                            "District Name": "beta",
                            "Flood Depth m": step * 0.15,
                        }
                    )

        self.database = self.root / "project.sqlite"
        mapping = suggest_mapping(TRANSPORT_COLUMNS)["roles"]
        self.progress_updates: list[tuple[float, str]] = []
        self.summary = import_csv_project(
            files=[self.first, self.second],
            mapping=mapping,
            units={"speed": "km/h", "travel_time": "min", "primary_metric": "m"},
            timezone="UTC",
            database_path=self.database,
            project_id="p5_test",
            project_name="Schema flexibility test",
            hazard_type="flood",
            source="automated_test",
            progress=lambda value, message: self.progress_updates.append((value, message)),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_heterogeneous_files_import_to_compact_schema(self) -> None:
        self.assertEqual(self.summary["entity_count"], 4)
        self.assertEqual(self.summary["observation_count"], 12)
        self.assertEqual(self.summary["region_count"], 2)
        self.assertEqual(self.summary["expected_interval_s"], 600)
        second_mapping = self.summary["source_files"][1]["resolved_mapping"]
        self.assertEqual(second_mapping["entity_id"], "Road Link Code")
        self.assertEqual(second_mapping["primary_metric"], "Flood Depth m")
        with closing(sqlite3.connect(self.database)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(observations)")}
            self.assertIn("entity_pk", columns)
            self.assertNotIn("entity_id", columns)
            self.assertNotIn("timestamp_text", columns)

    def test_import_progress_uses_real_chunk_fraction(self) -> None:
        values = [value for value, _message in self.progress_updates]
        self.assertEqual(values[-1], 100.0)
        self.assertEqual(values, sorted(values))
        imported_values = [value for value, message in self.progress_updates if message.startswith("Imported")]
        self.assertTrue(imported_values)
        self.assertGreater(max(imported_values), 20.0)
        self.assertLessEqual(max(imported_values), 84.0)

    def test_map_history_and_observed_models(self) -> None:
        latest = map_features(self.database, metric="speed", timestamp=None, limit=100)
        self.assertEqual(len(latest["features"]), 4)
        self.assertTrue(all(feature["properties"]["quality"] == "valid" for feature in latest["features"]))
        first_feature = latest["features"][0]
        self.assertEqual(first_feature["geometry"]["type"], "Point")
        self.assertEqual(first_feature["properties"]["geometry_kind"], "representative_point")
        self.assertTrue(first_feature["properties"]["from_node"])
        self.assertTrue(first_feature["properties"]["to_node"])
        self.assertIn("not a topology node", self.summary["geometry_note"])
        history = entity_history(self.database, "alpha", "a1", "speed")
        self.assertEqual(len(history["points"]), 3)
        self.assertGreater(history["points"][-1]["severity"], 0.4)
        persistence = persistent_degradation(
            self.database,
            metric="speed",
            threshold=0.2,
            top_k=4,
            bbox=None,
            time_start=None,
            time_end=None,
        )
        self.assertEqual(len(persistence["items"]), 4)
        area = area_summary(
            self.database,
            metric="speed",
            bbox=[-82, 26, -79, 29],
            timestamp=None,
        )
        self.assertEqual(area["headline_effects"][0]["value"], "4")

    def test_point_to_link_mapping_is_deterministic_saved_and_switchable(self) -> None:
        initial = network_mapping_status(self.database)
        self.assertTrue(initial["available"])
        self.assertFalse(initial["ready"])

        built = build_network_mapping(self.database)
        self.assertTrue(built["ready"])
        self.assertEqual(built["mapped_link_count"], 4)
        self.assertEqual(built["method"], "incident_reference_point_coordinate_median_v1")
        with closing(sqlite3.connect(self.database)) as connection:
            saved_node_count = connection.execute("SELECT COUNT(*) FROM inferred_network_nodes").fetchone()[0]
            self.assertGreater(saved_node_count, 0)

        first = network_features(self.database, metric="speed", timestamp=None, limit=100)
        self.assertEqual(len(first["features"]), 4)
        self.assertTrue(all(feature["geometry"]["type"] == "LineString" for feature in first["features"]))
        self.assertTrue(all(feature["properties"]["geometry_kind"] == "inferred_topology_line" for feature in first["features"]))
        self.assertTrue(all(feature["properties"]["inferred_line_length_m"] > 0 for feature in first["features"]))
        self.assertIn("not surveyed road centre-lines", first["geometry_note"])

        first_geometry = first["features"][0]["geometry"]
        rebuilt = build_network_mapping(self.database)
        second = network_features(self.database, metric="speed", timestamp=None, limit=100)
        self.assertEqual(rebuilt["mapped_link_count"], built["mapped_link_count"])
        self.assertEqual(second["features"][0]["geometry"], first_geometry)
        self.assertTrue(network_mapping_status(self.database)["ready"])

    def test_topology_closure_and_restoration_models(self) -> None:
        build_network_mapping(self.database)
        critical = critical_components(self.database, top_k=3, bbox=None)
        self.assertEqual(len(critical["items"]), 3)
        self.assertTrue(all(
            feature["geometry"]["type"] == "LineString"
            for feature in critical["feature_collection"]["features"]
        ))
        closure = closure_scenario(self.database, selected_keys=["alpha::a1"], bbox=None)
        self.assertEqual(len(closure["items"]), 1)
        self.assertEqual(closure["feature_collection"]["features"][0]["geometry"]["type"], "LineString")
        restoration = restoration_screening(
            self.database,
            selected_keys=["alpha::a1", "beta::b1"],
            bbox=None,
        )
        self.assertEqual(len(restoration["items"]), 2)
        self.assertEqual({item["key"] for item in restoration["items"]}, {"alpha::a1", "beta::b1"})

    def test_recovery_priorities_support_top_n_scenario_effects_areas_and_resources(self) -> None:
        build_network_mapping(self.database)
        selected = ["alpha::a1", "alpha::a2", "beta::b1", "beta::b2"]
        road_result = restoration_screening(
            self.database,
            selected_keys=selected,
            bbox=None,
            top_k=3,
            assumptions={
                "recovery_unit": "road",
                "scenario_changes": [
                    {"key": "alpha::a1", "effect": "closed", "magnitude": 1.0},
                    {"key": "alpha::a2", "effect": "speed_reduction", "magnitude": 0.5},
                    {"key": "beta::b1", "effect": "travel_time_increase", "magnitude": 0.5},
                    {"key": "beta::b2", "effect": "closed", "magnitude": 1.0},
                ],
            },
        )
        self.assertEqual(len(road_result["items"]), 3)
        self.assertEqual(road_result["visual"]["method"], "adaptive_marginal_efficiency")
        self.assertTrue(all(item["recovery_unit"] == "road" for item in road_result["items"]))
        self.assertTrue(all(0 <= item["cumulative_recovery"] <= 1 for item in road_result["items"]))

        area_result = restoration_screening(
            self.database,
            selected_keys=selected,
            bbox=None,
            top_k=2,
            assumptions={"recovery_unit": "area"},
        )
        self.assertEqual(len(area_result["items"]), 2)
        self.assertTrue(all(item["recovery_unit"] == "area" for item in area_result["items"]))
        self.assertTrue(any(feature["geometry"]["type"] == "Polygon" for feature in area_result["feature_collection"]["features"]))

        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("UPDATE entities SET repair_cost = 10 + entity_pk, repair_duration = 2 + entity_pk")
            connection.commit()
        resource_result = restoration_screening(
            self.database,
            selected_keys=selected,
            bbox=None,
            top_k=2,
            assumptions={"recovery_unit": "road", "priority_objective": "gain_per_cost"},
        )
        self.assertEqual(resource_result["model"]["id"], "P5-M8")
        self.assertEqual(resource_result["evidence"]["objective_used"], "gain_per_cost")
        self.assertTrue(all(item["repair_cost"] is not None for item in resource_result["items"]))

    def test_saved_scenario_engine_supports_multiple_deterministic_effects(self) -> None:
        build_network_mapping(self.database)
        targets = select_scenario_targets(
            self.database,
            bbox=[-81.2, 26.8, -80.8, 27.2],
        )
        self.assertEqual(set(targets["keys"]), {"alpha::a1", "alpha::a2"})
        self.assertFalse(targets["truncated"])
        self.assertTrue(all(
            feature["geometry"]["type"] == "LineString"
            for feature in targets["feature_collection"]["features"]
        ))

        scenario = evaluate_disruption_scenario(
            self.database,
            base_timestamp=None,
            changes=[
                {"key": "alpha::a1", "effect": "closed", "magnitude": 1.0},
                {"key": "alpha::a2", "effect": "speed_reduction", "magnitude": 0.5},
            ],
        )
        self.assertTrue(scenario["verification"]["passed"])
        self.assertEqual(scenario["affected_count"], 2)
        self.assertEqual(scenario["topology_applied_count"], 2)
        self.assertEqual(scenario["randomness"], "none")
        self.assertLessEqual(
            scenario["evidence"]["scenario_efficiency"],
            scenario["evidence"]["baseline_efficiency"],
        )
        features = scenario["feature_collection"]["features"]
        self.assertEqual(
            sum(feature["properties"].get("impact_role") == "initial_disruption" for feature in features),
            2,
        )
        self.assertEqual(
            len(features),
            scenario["initial_disruption_count"] + scenario["indirect_impact_count"],
        )

    def test_scenario_with_locations_only_maps_evidence_and_withholds_network_claim(self) -> None:
        source = self.root / "location_only.csv"
        fields = ["Sensor Ref", "Y Coordinate", "X Coordinate", "Observed Flood Depth"]
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([
                {"Sensor Ref": "g1", "Y Coordinate": -33.86, "X Coordinate": 151.20, "Observed Flood Depth": 0.4},
                {"Sensor Ref": "g2", "Y Coordinate": -33.87, "X Coordinate": 151.21, "Observed Flood Depth": 0.8},
            ])
        database = self.root / "location_only.sqlite"
        mapping = suggest_mapping(fields)["roles"]
        import_csv_project(
            files=[source],
            mapping=mapping,
            units={"primary_metric": "m"},
            timezone="Australia/Sydney",
            database_path=database,
            project_id="location_only",
            project_name="Location evidence only",
            hazard_type="flood",
            source="automated_test",
        )
        result = evaluate_disruption_scenario(
            database,
            base_timestamp=None,
            changes=[{"key": "location_only::g1", "effect": "travel_time_increase", "magnitude": 0.25}],
        )
        self.assertEqual(result["status"], "location_only")
        self.assertEqual(result["visual"]["mode"], "location_only")
        self.assertIsNone(result["visual"]["scenario_efficiency_percent"])
        self.assertIn("withheld", result["verification"]["checks"][-1])
        self.assertTrue(result["verification"]["passed"])

    def test_temporal_spatial_and_progressive_resilience_models(self) -> None:
        temporal = temporal_resilience_curve(
            self.database,
            metric="speed",
            bbox=None,
            time_start=None,
            time_end=None,
        )
        self.assertTrue(temporal["verification"]["passed"])
        self.assertEqual(len(temporal["items"]), 3)
        self.assertGreaterEqual(float(temporal["headline_effects"][0]["value"]), 0)
        hotspots = spatial_hotspots(
            self.database,
            metric="speed",
            top_k=5,
            bbox=None,
            timestamp=None,
        )
        self.assertTrue(hotspots["verification"]["passed"])
        self.assertGreater(len(hotspots["items"]), 0)
        self.assertTrue(all(feature["geometry"]["type"] == "Polygon" for feature in hotspots["feature_collection"]["features"]))
        robustness = progressive_robustness(self.database, top_k=3, bbox=None)
        self.assertTrue(robustness["verification"]["passed"])
        self.assertEqual(len(robustness["items"]), 3)
        self.assertTrue(all(0 <= item["performance_ratio"] <= 1 for item in robustness["items"]))
        self.assertEqual(robustness["visual"]["mode"], "network_stress_curve")
        self.assertEqual(set(robustness["visual"]["thresholds"]), {"90_percent", "75_percent", "50_percent"})
        ratios = [1.0, *(item["performance_ratio"] for item in robustness["items"])]
        self.assertTrue(all(right <= left + 1e-10 for left, right in zip(ratios, ratios[1:])))
        confirmed = progressive_robustness(
            self.database,
            top_k=2,
            bbox=None,
            failure_keys=["beta::b1", "alpha::a1"],
        )
        self.assertEqual([item["key"] for item in confirmed["items"]], ["beta::b1", "alpha::a1"])
        self.assertEqual(confirmed["evidence"]["sequence_source"], "confirmed_current_critical_road_order")
        extended = progressive_robustness(
            self.database,
            top_k=3,
            bbox=None,
            failure_keys=["beta::b1"],
        )
        self.assertEqual(len(extended["items"]), 3)
        self.assertEqual(extended["items"][0]["key"], "beta::b1")
        self.assertEqual(
            extended["evidence"]["sequence_source"],
            "confirmed_current_critical_road_prefix_then_deterministic_extension",
        )

    def test_extended_registry_models_have_real_executable_base_tiers(self) -> None:
        requests = {
            "P5-M5": {},
            "P5-M6": {},
            "P5-M8": {"selected_keys": ["alpha::a1", "alpha::a2"]},
            "P5-M12": {},
            "P5-M13": {"selected_keys": ["alpha::a1"]},
            "P5-M14": {"metric": "speed"},
            "P5-M15": {},
            "P5-M16": {},
            "P5-M18": {},
            "P5-M19": {"metric": "speed"},
            "P5-M20": {"selected_keys": ["alpha::a1", "alpha::a2"]},
        }
        for model_id, extra in requests.items():
            with self.subTest(model_id=model_id):
                result = run_adaptive_model(
                    self.database,
                    {"model_id": model_id, "top_k": 3, "bbox": None, **extra},
                )
                self.assertEqual(result["model"]["id"], model_id)
                self.assertTrue(result["verification"]["passed"])
                self.assertIn(result["status"], {"supported", "conditional"})
                self.assertTrue(result["data_boundary"])

    def test_probability_weighted_reliability_keeps_table_and_map_ranking_aligned(self) -> None:
        deterministic = run_adaptive_model(
            self.database,
            {"model_id": "P5-M12", "top_k": 4, "bbox": None},
        )
        first_key = deterministic["items"][0]["key"]
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("UPDATE entities SET failure_probability = 1.0")
            region_id, entity_id = first_key.split("::", 1)
            connection.execute(
                "UPDATE entities SET failure_probability = 0.0 WHERE region_id = ? AND entity_id = ?",
                (region_id, entity_id),
            )
            connection.commit()
        probability = run_adaptive_model(
            self.database,
            {"model_id": "P5-M12", "top_k": 4, "bbox": None},
        )
        self.assertEqual(probability["adaptive_execution"]["tier"], "uploaded_probability_expected_loss")
        expected_losses = [item["expected_efficiency_loss"] for item in probability["items"]]
        self.assertEqual(expected_losses, sorted(expected_losses, reverse=True))
        feature_keys = [feature["properties"]["key"] for feature in probability["feature_collection"]["features"]]
        self.assertEqual(feature_keys, [item["key"] for item in probability["items"]])
        self.assertEqual(
            [feature["properties"]["rank"] for feature in probability["feature_collection"]["features"]],
            list(range(1, len(probability["items"]) + 1)),
        )

    def test_model_router_covers_every_registered_family_without_promoting_missing_data(self) -> None:
        routing = recommend_models(self.summary, goal="cascade and recovery")
        self.assertEqual(routing["coverage"]["registered_model_count"], 24)
        self.assertTrue(routing["coverage"]["has_executable_path"])
        self.assertEqual(len(routing["models"]), 24)
        self.assertTrue(all(item.get("execution_adapter") for item in routing["models"]))
        od_model = next(item for item in routing["models"] if item["model_id"] == "P5-M5")
        self.assertEqual(od_model["selected_tier"], od_model["base_tier"])
        self.assertIn("od_demand", od_model["missing_for_high_fidelity"])

    def test_registered_model_runs_through_the_saved_analysis_dispatcher(self) -> None:
        result = run_analysis(
            self.database,
            {
                "analysis_type": "adaptive_registered_model",
                "model_id": "P5-M6",
                "metric": "speed",
                "top_k": 2,
                "bbox": None,
                "selected_keys": [],
            },
        )
        self.assertEqual(result["analysis_type"], "adaptive_registered_model")
        self.assertEqual(result["model"]["id"], "P5-M6")
        self.assertEqual(result["request"]["model_id"], "P5-M6")

    def test_natural_language_routes_extended_models_and_preserves_arbitrary_top_n(self) -> None:
        project = {"project_id": "p5_test", "name": "Test", "summary": self.summary}
        contract = interpret_command(
            text="Run a top 1000 cascading failure stress test.",
            project=project,
            selected_keys=["alpha::a1"],
            current_metric="speed",
        )
        self.assertTrue(contract["ready"])
        self.assertEqual(contract["analysis_type"], "adaptive_registered_model")
        self.assertEqual(contract["model_id"], "P5-M13")
        self.assertEqual(contract["top_k"], 1000)
        self.assertEqual(contract["selected_keys"], ["alpha::a1"])
        chinese = interpret_command(
            text="找出速度持续下降最严重的5条道路。",
            project=project,
            selected_keys=[],
            current_metric="speed",
        )
        self.assertTrue(chinese["ready"])
        self.assertEqual(chinese["analysis_type"], "persistent_degradation")
        self.assertEqual(chinese["top_k"], 5)

    def test_top_100_and_latest_scope_are_preserved_in_command_contract(self) -> None:
        project = {"project_id": "p5_test", "name": "test", "summary": self.summary}
        contract = interpret_command(
            text="Identify the Top 100 critical links across the whole network",
            project=project,
            selected_keys=["alpha::a1"],
            bbox=[-81.1, 26.9, -80.8, 27.2],
            current_metric="speed",
        )
        self.assertEqual(contract["top_k"], 100)
        self.assertEqual(contract["selected_keys"], [])
        self.assertIsNone(contract["bbox"])
        self.assertEqual(contract["grounding"]["scope_label"], "Whole project")
        closure = interpret_command(
            text="Assume the selected links are closed",
            project=project,
            selected_keys=["alpha::a1"],
            bbox=[-81.1, 26.9, -80.8, 27.2],
            current_metric="speed",
        )
        self.assertEqual(closure["selected_keys"], ["alpha::a1"])
        self.assertIsNone(closure["bbox"])
        dated_curve = interpret_command(
            text="Show the observed resilience curve from 2017",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        self.assertEqual(dated_curve["top_k"], 10)


if __name__ == "__main__":
    unittest.main()
