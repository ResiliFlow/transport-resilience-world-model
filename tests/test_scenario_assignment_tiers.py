from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from engine.importers import import_csv_project, suggest_mapping
from engine.scenarios import evaluate_disruption_scenario


class CapacityScenarioTierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="p5_capacity_scenario_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "capacity_network.csv"
        fields = [
            "Road Link Code", "Origin Node ID", "Destination Node ID",
            "Segment Length Metres", "Centre Latitude", "Centre Longitude",
            "Recorded At", "Mean Velocity KPH", "Journey Duration Minutes",
            "Traffic Flow", "Road Capacity", "Uncongested Time", "District Name",
        ]
        links = [
            ("a", "1", "2", 27.000, -81.000, 1.0),
            ("b", "2", "4", 27.010, -80.990, 1.0),
            ("c", "1", "3", 26.998, -80.998, 1.2),
            ("d", "3", "4", 27.012, -80.988, 1.2),
            ("e", "2", "3", 27.006, -80.994, 0.7),
            ("f", "3", "2", 27.005, -80.995, 0.7),
            ("g", "4", "1", 27.008, -80.992, 1.5),
        ]
        with self.source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for timestamp, multiplier in (("2024-09-01 00:00:00", 1.0), ("2024-09-01 00:05:00", 1.05)):
                for index, (link, origin, destination, latitude, longitude, free_time) in enumerate(links):
                    writer.writerow(
                        {
                            "Road Link Code": link,
                            "Origin Node ID": origin,
                            "Destination Node ID": destination,
                            "Segment Length Metres": 1000,
                            "Centre Latitude": latitude,
                            "Centre Longitude": longitude,
                            "Recorded At": timestamp,
                            "Mean Velocity KPH": 60 / free_time,
                            "Journey Duration Minutes": free_time * multiplier,
                            "Traffic Flow": 160 + index * 10,
                            "Road Capacity": 900 + index * 20,
                            "Uncongested Time": free_time,
                            "District Name": "r",
                        }
                    )
        mapping = suggest_mapping(fields)["roles"]
        self.assertEqual(mapping["flow"], "Traffic Flow")
        self.assertEqual(mapping["capacity"], "Road Capacity")
        self.assertEqual(mapping["free_flow_time"], "Uncongested Time")
        self.database = self.root / "project.sqlite"
        self.summary = import_csv_project(
            files=[self.source],
            mapping=mapping,
            units={"speed": "km/h", "travel_time": "min", "flow": "vehicles/hour", "capacity": "vehicles/hour"},
            timezone="UTC",
            database_path=self.database,
            project_id="capacity_project",
            project_name="Capacity scenario",
            hazard_type="hurricane",
            source="automated_test",
            topology_node_scope="global",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_closed_slower_and_longer_time_use_same_capacity_assignment_tier(self) -> None:
        cases = (
            ("closed", 1.0),
            ("speed_reduction", 0.50),
            ("travel_time_increase", 0.50),
        )
        for effect, magnitude in cases:
            with self.subTest(effect=effect):
                result = evaluate_disruption_scenario(
                    self.database,
                    base_timestamp=None,
                    changes=[{"key": "r::a", "effect": effect, "magnitude": magnitude}],
                )
                assignment = result["evidence"]["assignment"]
                self.assertEqual(assignment["tier"], "observed_flow_calibrated_capacity_ue_proxy")
                self.assertEqual(assignment["solver"], "frank_wolfe_beckmann_bpr")
                self.assertTrue(assignment["baseline_solver_converged"])
                self.assertTrue(assignment["scenario_solver_converged"])
                self.assertEqual(assignment["selected_seed_count"], 1)
                self.assertEqual(assignment["sampled_seed_count"], 1)
                self.assertEqual(assignment["covered_seed_count"], 1)
                self.assertEqual(assignment["seed_od_coverage_percent"], 100.0)
                self.assertGreaterEqual(
                    assignment["rerouted_pair_count"]
                    + assignment["disconnected_pair_count"]
                    + assignment["maximum_detour_percent"],
                    0.000001,
                )
                self.assertEqual(result["initial_disruption_count"], 1)
                self.assertEqual(result["randomness"], "none")
                self.assertTrue(result["verification"]["passed"])
                direct = [
                    feature for feature in result["feature_collection"]["features"]
                    if feature["properties"].get("impact_role") == "initial_disruption"
                ]
                indirect = [
                    feature for feature in result["feature_collection"]["features"]
                    if feature["properties"].get("impact_role") == "rerouting_pressure"
                ]
                self.assertEqual([feature["properties"]["key"] for feature in direct], ["r::a"])
                self.assertTrue(all(feature["properties"]["key"] != "r::a" for feature in indirect))
                if effect in {"closed", "speed_reduction"}:
                    self.assertGreater(len(indirect), 0)
                self.assertIn("No confirmed OD matrix", result["data_boundary"])


if __name__ == "__main__":
    unittest.main()
