from __future__ import annotations

import csv
from contextlib import closing
import gzip
from pathlib import Path
import sqlite3
import shutil
import tempfile
import unittest
from uuid import uuid4

from engine.importers import import_csv_project, preview_csv
from engine.matsim_adapter import (
    LinkRecord,
    _animation_frames,
    _impact_focused_probe_agents,
    _matsim_id,
    _matsim_dtd_uri,
    _partition_matsim_changes,
    _route_outcomes,
    _scenario_link,
    _scenario_reachable_agents,
    _write_network,
    _write_population,
    assess_matsim_readiness,
    run_matsim_scenario,
)


def _test_link(key: str, first: str, second: str, *, length: float = 100.0) -> LinkRecord:
    index = sum(ord(character) for character in key) % 100
    return LinkRecord(
        key=key,
        region_id="test",
        entity_id=key,
        from_node=first,
        to_node=second,
        length_m=length,
        free_speed_mps=10.0,
        capacity_vph=900.0,
        coordinates=((151.0 + index / 10000, -33.8), (151.001 + index / 10000, -33.799)),
        capacity_uploaded=False,
        impedance_uploaded=True,
    )


class MatsimProbeDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.links = [
            _test_link("e01", "n0", "n1"), _test_link("e10", "n1", "n0"),
            _test_link("e12", "n1", "n2"), _test_link("e21", "n2", "n1"),
            _test_link("e23", "n2", "n3"), _test_link("e32", "n3", "n2"),
            _test_link("e34", "n3", "n4"), _test_link("e43", "n4", "n3"),
            _test_link("e45", "n4", "n5"), _test_link("e54", "n5", "n4"),
            _test_link("e56", "n5", "n6"), _test_link("e65", "n6", "n5"),
            _test_link("e2a", "n2", "na", length=240),
            _test_link("ea4", "na", "n4", length=240),
            _test_link("e4a", "n4", "na", length=240),
            _test_link("ea2", "na", "n2", length=240),
        ]

    def test_probe_selector_generates_requested_reproducible_impact_focused_population(self) -> None:
        changes = {"e34": {"key": "e34", "effect": "speed_reduction", "magnitude": 0.5}}
        first, profile = _impact_focused_probe_agents(self.links, changes, 50)
        second, repeated_profile = _impact_focused_probe_agents(self.links, changes, 50)
        self.assertEqual(first, second)
        self.assertEqual(profile, repeated_profile)
        self.assertEqual(len(first), 50)
        self.assertEqual(profile["generated_agents"], 50)
        self.assertGreater(profile["group_counts"]["impact_focused"], 0)
        self.assertGreater(profile["unique_od_pairs"], 0)

    def test_area_changes_keep_routable_roads_and_report_unroutable_map_entities(self) -> None:
        raw_changes = [
            {"key": link.key, "effect": "speed_reduction", "magnitude": 0.5}
            for link in self.links[:6]
        ] + [{
            "key": "mapped-but-not-routable",
            "effect": "speed_reduction",
            "magnitude": 0.5,
        }]
        changes, excluded = _partition_matsim_changes(self.links, raw_changes)
        self.assertEqual(len(changes), 6)
        self.assertEqual(excluded, [{
            "key": "mapped-but-not-routable",
            "reason": "The mapped road is absent from the routable MATSim network.",
        }])

    def test_area_changes_fail_only_when_no_selected_road_is_routable(self) -> None:
        with self.assertRaisesRegex(ValueError, "None of the selected roads"):
            _partition_matsim_changes(self.links, [{
                "key": "mapped-but-not-routable",
                "effect": "travel_time_increase",
                "magnitude": 0.5,
            }])

    def test_peak_snapshot_is_event_derived_and_marks_agent_relation(self) -> None:
        changes = {"e34": {"key": "e34", "effect": "closed", "magnitude": 1.0}}
        changed = _matsim_id("e34", "l")
        nearby = _matsim_id("e23", "l")
        spillover = _matsim_id("e01", "l")
        parsed = {
            "frames": {
                "0": {spillover: 8},
                "300": {changed: 4, nearby: 5, spillover: 1},
                "600": {spillover: 12},
            },
            "frame_vehicles": {
                "0": [("p0", spillover)],
                "300": [("p1", changed), ("p2", nearby)],
                "600": [("p3", spillover)],
            },
        }
        frames = _animation_frames(parsed, self.links, changes=changes)
        peak = next(frame for frame in frames if frame["is_peak_impact"])
        self.assertEqual(peak["time_seconds"], 300)
        self.assertEqual(peak["changed_road_entries"], 4)
        relations = {
            feature["properties"].get("impact_relation")
            for feature in peak["feature_collection"]["features"]
            if feature["geometry"]["type"] == "Point"
        }
        self.assertEqual(relations, {"changed", "nearby"})

    def test_agent_outcomes_compare_same_population_person_by_person(self) -> None:
        baseline = {
            "person_routes": {"p1": ["a", "b"], "p2": ["a", "b"]},
        }
        scenario = {
            "person_routes": {"p1": ["a", "b"], "p2": ["a", "c"]},
            "departed_people": ["p1", "p2", "p3"],
            "arrived_people": ["p1", "p2"],
            "stuck_people": ["p3"],
        }
        outcomes = _route_outcomes(baseline, scenario, 4)
        self.assertEqual(outcomes["same_route"], 1)
        self.assertEqual(outcomes["rerouted"], 1)
        self.assertEqual(outcomes["stuck_or_aborted"], 1)
        self.assertEqual(outcomes["unstarted"], 1)

    def test_generated_xml_uses_packaged_dtds_without_a_network_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_matsim_dtd_") as folder:
            root = Path(folder)
            network = root / "network.xml.gz"
            population = root / "population.xml.gz"
            _write_network(
                network,
                {"n0": (151.0, -33.8), "n1": (151.001, -33.799)},
                [_test_link("e01", "n0", "n1")],
            )
            _write_population(population, [("e01", "e01", 7 * 3600.0)])
            with gzip.open(network, "rt", encoding="utf-8") as stream:
                network_xml = stream.read()
            with gzip.open(population, "rt", encoding="utf-8") as stream:
                population_xml = stream.read()

        self.assertIn(_matsim_dtd_uri("network_v2.dtd"), network_xml)
        self.assertIn(_matsim_dtd_uri("population_v6.dtd"), population_xml)
        self.assertNotIn("www.matsim.org/files/dtd", network_xml + population_xml)

    def test_static_closure_is_removed_and_disconnected_agents_are_preserved_as_no_route(self) -> None:
        chain = [
            _test_link("origin", "n0", "n1"),
            _test_link("closed", "n1", "n2"),
            _test_link("destination", "n2", "n3"),
        ]
        changes = {"closed": {"key": "closed", "effect": "closed", "magnitude": 1.0}}
        scenario_links = [
            changed for link in chain
            if (changed := _scenario_link(link, changes)) is not None
        ]
        self.assertNotIn("closed", {link.key for link in scenario_links})
        reachable, indexes, disconnected = _scenario_reachable_agents(
            [("origin", "destination", 7 * 3600.0)],
            chain,
            scenario_links,
            changes,
        )
        self.assertEqual(reachable, [])
        self.assertEqual(indexes, [])
        self.assertEqual(len(disconnected), 1)
        self.assertEqual(disconnected[0]["person_id"], "twm-000000")
        self.assertEqual(disconnected[0]["approach_link_key"], "origin")
        self.assertEqual(disconnected[0]["closed_link_key"], "closed")


@unittest.skipUnless(
    shutil.which("java") and (Path(__file__).resolve().parents[1] / "matsim_worker/target/twm-matsim-worker.jar").is_file(),
    "Requires Java and the separately built MATSim worker.",
)
class MatsimAdapterIntegrationTests(unittest.TestCase):
    def test_legacy_project_without_optional_supply_columns_is_data_limited_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_matsim_legacy_") as folder:
            database_path = Path(folder) / "legacy.sqlite"
            with closing(sqlite3.connect(database_path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE entities (
                        region_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        length_m REAL,
                        baseline_speed REAL,
                        baseline_travel_time REAL,
                        PRIMARY KEY (region_id, entity_id)
                    );
                    CREATE TABLE inferred_network_links (
                        link_key TEXT PRIMARY KEY,
                        region_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL
                    );
                    INSERT INTO entities VALUES ('r', 'a', 1000, 45, 1.3);
                    INSERT INTO entities VALUES ('r', 'b', 900, 40, 1.4);
                    INSERT INTO inferred_network_links VALUES ('r::a', 'r', 'a');
                    INSERT INTO inferred_network_links VALUES ('r::b', 'r', 'b');
                    """
                )
            readiness = assess_matsim_readiness(
                {
                    "project_id": "legacy",
                    "database_path": str(database_path),
                    "summary": {"metrics": {}},
                }
            )
            self.assertEqual(readiness["tier"], "exploratory_probe")
            self.assertEqual(readiness["positive_impedance_links"], 2)
            self.assertEqual(readiness["capacity_links"], 0)
            self.assertIn("road capacity on every routable link (0/2)", readiness["missing_for_full"])

    def test_uploaded_od_and_capacity_activate_real_demand_tier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_matsim_full_") as folder:
            root = Path(folder)
            source = root / "multi_hazard_transport_package.csv"
            fields = [
                "record_type", "node_id", "longitude", "latitude", "road_id", "from_node",
                "to_node", "geometry", "length_m", "free_flow_time_min", "capacity_vph",
                "origin", "destination", "trip_demand",
            ]
            rows = [
                {"record_type": "node", "node_id": "A", "longitude": 151.000, "latitude": -33.800},
                {"record_type": "node", "node_id": "B", "longitude": 151.010, "latitude": -33.800},
                {"record_type": "node", "node_id": "C", "longitude": 151.010, "latitude": -33.810},
                {"record_type": "node", "node_id": "D", "longitude": 151.020, "latitude": -33.805},
                {"record_type": "link", "road_id": "AB", "from_node": "A", "to_node": "B", "geometry": "LINESTRING(151 -33.8,151.01 -33.8)", "length_m": 1000, "free_flow_time_min": 1.2, "capacity_vph": 900},
                {"record_type": "link", "road_id": "BD", "from_node": "B", "to_node": "D", "geometry": "LINESTRING(151.01 -33.8,151.02 -33.805)", "length_m": 1100, "free_flow_time_min": 1.4, "capacity_vph": 850},
                {"record_type": "link", "road_id": "AC", "from_node": "A", "to_node": "C", "geometry": "LINESTRING(151 -33.8,151.01 -33.81)", "length_m": 1400, "free_flow_time_min": 1.8, "capacity_vph": 800},
                {"record_type": "link", "road_id": "CD", "from_node": "C", "to_node": "D", "geometry": "LINESTRING(151.01 -33.81,151.02 -33.805)", "length_m": 1100, "free_flow_time_min": 1.4, "capacity_vph": 800},
                {"record_type": "link", "road_id": "DA", "from_node": "D", "to_node": "A", "geometry": "LINESTRING(151.02 -33.805,151 -33.8)", "length_m": 2200, "free_flow_time_min": 2.7, "capacity_vph": 750},
                {"record_type": "link", "road_id": "BC", "from_node": "B", "to_node": "C", "geometry": "LINESTRING(151.01 -33.8,151.01 -33.81)", "length_m": 1000, "free_flow_time_min": 1.2, "capacity_vph": 700},
                {"record_type": "link", "road_id": "CB", "from_node": "C", "to_node": "B", "geometry": "LINESTRING(151.01 -33.81,151.01 -33.8)", "length_m": 1000, "free_flow_time_min": 1.2, "capacity_vph": 700},
                {"record_type": "od", "origin": "A", "destination": "D", "trip_demand": 120},
                {"record_type": "od", "origin": "B", "destination": "D", "trip_demand": 45},
            ]
            with source.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            preview = preview_csv(source)
            mapping = preview["suggested_mapping"]["roles"]
            database_path = root / "project.sqlite"
            summary = import_csv_project(
                files=[source], mapping=mapping, file_mappings={source.name: mapping},
                units={"travel_time": "min", "capacity": "vehicles/hour", "length": "m"},
                timezone="UTC", topology_node_scope="global", entity_id_scope="global",
                database_path=database_path, project_id="matsim_full_tier_test",
                project_name="MATSim full tier test", hazard_type="flood", source="integration_test",
            )
            project = {
                "project_id": "matsim_full_tier_test", "name": "MATSim full tier test",
                "hazard_type": "flood", "database_path": str(database_path),
                "summary": summary, "mapping": mapping,
            }
            readiness = assess_matsim_readiness(project)
            self.assertEqual(readiness["tier"], "demand_calibrated", readiness)
            self.assertTrue(readiness["can_run"], readiness)

            with closing(sqlite3.connect(database_path)) as connection:
                link_keys = dict(connection.execute(
                    "SELECT entity_id, link_key FROM inferred_network_links"
                ).fetchall())
            key = link_keys["BD"]
            scenario = {
                "scenario_id": "scenario_matsim_full_tier",
                "name": "Flood slowdown on BD",
                "changes": [{"key": key, "effect": "speed_reduction", "magnitude": 0.5}],
            }
            result = run_matsim_scenario(
                project, scenario, job_id=f"full_{uuid4().hex[:8]}",
                iterations=1, max_agents=60, timeout_seconds=240,
            )
            self.assertEqual(result["tier"], "demand_calibrated")
            self.assertEqual(result["manifest"]["source_od_rows"], 2)
            self.assertEqual(result["manifest"]["agent_count"], 60)
            self.assertTrue(result["verification"]["passed"])
            self.assertGreater(result["baseline"]["departures"], 0)
            self.assertGreaterEqual(result["frame_count"], 1)
            self.assertEqual(result["route_outcomes"]["agent_count"], 60)
            self.assertIn("peak_impact_frame_index", result)
            self.assertEqual(
                result["peak_impact_snapshot"]["frame_index"],
                result["peak_impact_frame_index"],
            )

            area_scenario = {
                "scenario_id": "scenario_matsim_area_full_tier",
                "name": "Flood slowdown across a selected area",
                "changes": [
                    {"key": link_keys[entity_id], "effect": "speed_reduction", "magnitude": 0.5}
                    for entity_id in ("AB", "BD", "BC")
                ],
            }
            area_result = run_matsim_scenario(
                project, area_scenario, job_id=f"area_{uuid4().hex[:8]}",
                iterations=1, max_agents=60, timeout_seconds=240,
            )
            self.assertEqual(area_result["manifest"]["selected_change_count"], 3)
            self.assertEqual(area_result["manifest"]["changed_link_count"], 3)
            self.assertEqual(area_result["route_outcomes"]["agent_count"], 60)
            self.assertTrue(area_result["verification"]["passed"])


if __name__ == "__main__":
    unittest.main()
