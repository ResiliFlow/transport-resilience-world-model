from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from engine.assistant import local_assistant_response
from engine.command_parser import interpret_command
from engine.importers import import_csv_project, suggest_mapping
from engine.network_mapping import build_network_mapping
from engine.routing import historical_route_detour_analysis


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "Road Link Code", "Origin Node", "Destination Node", "Length Metres",
    "Centre Lat", "Centre Lon", "Observed At", "Journey Duration Minutes", "District",
]


class HistoricalRouteDetourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="twm_route_detour_")
        self.root = Path(self.temporary.name)
        source = self.root / "renamed_historical_network.csv"
        links = [
            ("start", "A", "B", 1000, 27.00, -81.00, 1.0, 1.0),
            ("direct", "B", "D", 1000, 27.01, -80.99, 1.0, 5.0),
            ("alt-1", "B", "C", 1100, 27.02, -80.98, 1.0, 1.0),
            ("alt-2", "C", "D", 1100, 27.03, -80.97, 1.0, 1.0),
            ("finish", "D", "E", 1000, 27.04, -80.96, 1.0, 1.0),
        ]
        times = ("2021-08-01 00:00:00", "2021-08-01 00:10:00")
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for entity, origin, destination, length, lat, lon, first, second in links:
                for timestamp, travel_time in zip(times, (first, second)):
                    writer.writerow(
                        {
                            "Road Link Code": entity,
                            "Origin Node": origin,
                            "Destination Node": destination,
                            "Length Metres": length,
                            "Centre Lat": lat,
                            "Centre Lon": lon,
                            "Observed At": timestamp,
                            "Journey Duration Minutes": travel_time,
                            "District": "test",
                        }
                    )
        mapping = suggest_mapping(FIELDS)["roles"]
        self.database = self.root / "project.sqlite"
        self.summary = import_csv_project(
            files=[source],
            mapping=mapping,
            units={"travel_time": "min"},
            timezone="UTC",
            database_path=self.database,
            project_id="route_oracle",
            project_name="Historical detour oracle",
            hazard_type="flood",
            source="deterministic_test",
        )
        build_network_mapping(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_internal_reference_and_later_history_choose_expected_routes(self) -> None:
        result = historical_route_detour_analysis(
            self.database,
            origin_key="test::start",
            destination_key="test::finish",
            timestamp=self.summary["time_values"][-1],
        )
        baseline = result["evidence"]["baseline_route"]
        compared = result["evidence"]["comparison_route"]
        self.assertEqual(baseline["keys"], ["test::start", "test::direct", "test::finish"])
        self.assertEqual(compared["keys"], ["test::start", "test::alt-1", "test::alt-2", "test::finish"])
        self.assertEqual(result["visual"]["mode"], "route_options")
        self.assertEqual(result["visual"]["requested_count"], 2)
        self.assertEqual(result["visual"]["returned_count"], 2)
        self.assertEqual(result["visual"]["cost_unit"], "min")
        self.assertEqual(len(result["visual"]["routes"]), 2)
        self.assertLessEqual(result["visual"]["routes"][1]["overlap_with_primary"], 0.92)
        self.assertEqual(
            result["evidence"]["alternative_routes"][0]["keys"],
            ["test::start", "test::direct", "test::finish"],
        )
        self.assertEqual(
            len(result["evidence"]["alternative_routes"][0]["nodes"]),
            len(set(result["evidence"]["alternative_routes"][0]["nodes"])),
        )
        self.assertTrue(result["verification"]["passed"])
        self.assertTrue(all(feature["geometry"]["type"] in {"LineString", "Point"} for feature in result["feature_collection"]["features"]))

    def test_saved_closure_scenario_produces_a_real_detour(self) -> None:
        result = historical_route_detour_analysis(
            self.database,
            origin_key="test::start",
            destination_key="test::finish",
            timestamp=self.summary["time_values"][0],
            scenario_changes=[{"key": "test::direct", "effect": "closed", "magnitude": 1.0}],
            scenario_name="Direct road closure",
        )
        self.assertEqual(result["evidence"]["baseline_route"]["keys"], ["test::start", "test::direct", "test::finish"])
        self.assertEqual(result["evidence"]["comparison_route"]["keys"], ["test::start", "test::alt-1", "test::alt-2", "test::finish"])
        self.assertEqual(result["evidence"]["scenario_changes_applied"], 1)
        self.assertEqual(result["visual"]["returned_count"], 1)
        self.assertIn("1 distinct feasible route", result["conclusion"])

    def test_requested_route_count_and_map_attributes_match_returned_table(self) -> None:
        result = historical_route_detour_analysis(
            self.database,
            origin_key="test::start",
            destination_key="test::finish",
            timestamp=self.summary["time_values"][-1],
            scenario_changes=[{"key": "test::direct", "effect": "travel_time_increase", "magnitude": 0.5}],
            scenario_name="Travel-time increase",
            alternative_count=5,
        )
        self.assertEqual(result["visual"]["requested_count"], 5)
        self.assertEqual(result["visual"]["returned_count"], 2)
        self.assertTrue(result["visual"]["availability_limited"])
        self.assertTrue(result["visual"]["search_exhausted"])
        self.assertEqual(len(result["items"]), 2)
        features_by_route: dict[str, list[dict]] = {}
        for feature in result["feature_collection"]["features"]:
            variant_id = feature["properties"].get("route_variant_id")
            if variant_id:
                features_by_route.setdefault(variant_id, []).append(feature)
        for item in result["items"]:
            mapped = features_by_route[item["route_variant_id"]]
            self.assertTrue(mapped)
            self.assertTrue(all(feature["properties"]["route_colour"] == item["route_colour"] for feature in mapped))
            self.assertTrue(all(feature["properties"]["route_impedance"] == item["impedance"] for feature in mapped))
            self.assertTrue(all(feature["properties"]["route_link_count"] == item["link_count"] for feature in mapped))

    def test_closing_selected_start_returns_no_invented_route(self) -> None:
        result = historical_route_detour_analysis(
            self.database,
            origin_key="test::start",
            destination_key="test::finish",
            timestamp=self.summary["time_values"][-1],
            scenario_changes=[{"key": "test::start", "effect": "closed", "magnitude": 1.0}],
            scenario_name="Start road closure",
            alternative_count=4,
        )
        self.assertFalse(result["visual"]["reachable"])
        self.assertEqual(result["visual"]["returned_count"], 0)
        self.assertEqual(result["items"], [])
        self.assertFalse(any(feature["geometry"]["type"] == "LineString" for feature in result["feature_collection"]["features"]))
        self.assertTrue(result["verification"]["passed"])

    def test_connectivity_orientation_fallback_returns_three_clean_routes_for_all_effects(self) -> None:
        source = self.root / "connectivity_orientation_network.csv"
        links = [
            ("start", "A", "B", 1.0),
            # Every feasible route must leave the selected start road through
            # A, the reverse of its uploaded row orientation.  This reproduces
            # detector/link tables whose From/To fields encode connectivity
            # but do not declare one-way travel.
            ("short-1", "A", "C", 1.0),
            ("short-2", "C", "D", 1.0),
            ("medium-1", "A", "E", 2.0),
            ("medium-2", "E", "D", 2.0),
            ("long-1", "A", "G", 3.0),
            ("long-2", "G", "D", 3.0),
            ("finish", "D", "F", 1.0),
        ]
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for index, (entity, origin, destination, travel_time) in enumerate(links):
                for timestamp in ("2022-01-01 00:00:00", "2022-01-01 00:10:00"):
                    writer.writerow({
                        "Road Link Code": entity,
                        "Origin Node": origin,
                        "Destination Node": destination,
                        "Length Metres": 1000,
                        "Centre Lat": 27.0 + index * 0.001,
                        "Centre Lon": -81.0 + index * 0.001,
                        "Observed At": timestamp,
                        "Journey Duration Minutes": travel_time,
                        "District": "fallback",
                    })
        database = self.root / "connectivity_orientation.sqlite"
        summary = import_csv_project(
            files=[source],
            mapping=suggest_mapping(FIELDS)["roles"],
            units={"travel_time": "min"},
            timezone="UTC",
            database_path=database,
            project_id="connectivity_orientation",
            project_name="Connectivity orientation",
            hazard_type="flood",
            source="deterministic_test",
        )
        build_network_mapping(database)

        for effect, magnitude in (("closed", 1.0), ("speed_reduction", 0.5), ("travel_time_increase", 0.5)):
            with self.subTest(effect=effect):
                result = historical_route_detour_analysis(
                    database,
                    origin_key="fallback::start",
                    destination_key="fallback::finish",
                    timestamp=summary["time_values"][0],
                    scenario_changes=[{"key": "fallback::short-1", "effect": effect, "magnitude": magnitude}],
                    scenario_name=f"{effect} test",
                    alternative_count=2,
                    maximum_overlap=1.0,
                )
                self.assertEqual(result["evidence"]["routing_direction_mode"], "two_way_connectivity_proxy")
                self.assertTrue(result["visual"]["reachable"])
                self.assertEqual(result["visual"]["returned_count"], 2)
                self.assertEqual(len(result["visual"]["routes"]), 2)
                line_features = [
                    feature for feature in result["feature_collection"]["features"]
                    if feature["geometry"]["type"] == "LineString"
                ]
                self.assertEqual(
                    {feature["properties"]["route_role"] for feature in line_features},
                    {"recommended_route"},
                )
                self.assertEqual(
                    {feature["properties"]["route_colour"] for feature in line_features},
                    {"#1464d2", "#e87500"},
                )
                self.assertTrue(all(feature["properties"]["route_cost_unit"] == "min" for feature in line_features))
                self.assertTrue(result["verification"]["passed"])

    def test_history_and_multiple_saved_scenarios_recompute_distinct_routes(self) -> None:
        result = historical_route_detour_analysis(
            self.database,
            origin_key="test::start",
            destination_key="test::finish",
            timestamp=self.summary["time_values"][-1],
            scenario_variants=[
                {
                    "scenario_id": "scenario_direct_closed",
                    "name": "Direct road closed",
                    "base_timestamp": self.summary["time_values"][0],
                    "changes": [{"key": "test::direct", "effect": "closed", "magnitude": 1.0}],
                },
                {
                    "scenario_id": "scenario_alternative_closed",
                    "name": "Alternative road closed",
                    "base_timestamp": self.summary["time_values"][-1],
                    "changes": [{"key": "test::alt-1", "effect": "closed", "magnitude": 1.0}],
                },
            ],
        )
        self.assertEqual(result["visual"]["mode"], "route_scenario_compare")
        variants = {item["id"]: item for item in result["visual"]["variants"]}
        self.assertEqual(
            variants["historical"]["route_keys"],
            ["test::start", "test::alt-1", "test::alt-2", "test::finish"],
        )
        self.assertEqual(
            variants["scenario_direct_closed"]["route_keys"],
            ["test::start", "test::alt-1", "test::alt-2", "test::finish"],
        )
        self.assertEqual(
            variants["scenario_alternative_closed"]["route_keys"],
            ["test::start", "test::direct", "test::finish"],
        )
        self.assertEqual(len({item["colour"] for item in variants.values()}), 3)
        self.assertTrue(result["verification"]["passed"])
        line_features = [
            feature for feature in result["feature_collection"]["features"]
            if feature["geometry"]["type"] == "LineString" and feature["properties"]["route_role"] == "comparison_route"
        ]
        self.assertTrue(all(feature["properties"].get("route_variant_id") for feature in line_features))
        self.assertTrue(all(feature["properties"].get("route_colour") for feature in line_features))

    def test_parser_requires_exactly_two_endpoints_and_then_becomes_ready(self) -> None:
        project = {"project_id": "route_oracle", "name": "Historical detour oracle", "summary": self.summary}
        blocked = interpret_command(
            text="Compare a historical route and detour.",
            project=project,
            current_metric="travel_time",
            selected_keys=[],
            bbox=None,
            current_time=self.summary["time_values"][-1],
        )
        self.assertFalse(blocked["ready"])
        self.assertTrue(any("exactly two" in item for item in blocked["missing"]))
        ready = interpret_command(
            text="Compare a historical route and detour.",
            project=project,
            current_metric="travel_time",
            selected_keys=["test::start", "test::finish"],
            bbox=None,
            current_time=self.summary["time_values"][-1],
        )
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["analysis_type"], "route_detour_analysis")
        self.assertEqual(ready["analysis_assumptions"]["origin_key"], "test::start")

    def test_parser_understands_point_and_multi_scenario_route_requests(self) -> None:
        project = {"project_id": "route_oracle", "name": "Historical detour oracle", "summary": self.summary}
        for text in ("Recommend routes under scenarios between two points", "给我两点之间不同情景路径推荐"):
            contract = interpret_command(
                text=text,
                project=project,
                current_metric="travel_time",
                selected_keys=["test::start", "test::finish"],
                bbox=None,
                current_time=self.summary["time_values"][-1],
            )
            self.assertTrue(contract["ready"], text)
            self.assertEqual(contract["analysis_type"], "route_detour_analysis")


class ToolTaxonomyAndAssistantTests(unittest.TestCase):
    def test_interface_separates_historical_readings_from_core_tools(self) -> None:
        page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("Basic statistical information", page)
        self.assertIn("Core decision tools", page)
        for label in (
            "Repeated problem locations", "Changes over time", "Find critical roads",
            "Find critical areas", "Plan recovery priorities", "Plan routes under disruption",
            "Test network resilience", "Build a disruption scenario",
        ):
            self.assertIn(label, page)
        self.assertNotIn("+ New scenario", page)
        self.assertIn("Pick from points", page)
        self.assertIn("Create a temporary disruption here", page)
        self.assertIn('id="recovery-builder"', page)
        self.assertIn('id="route-alternative-count"', page)
        self.assertIn('id="route-view-points"', page)
        route_builder_body = app.split("async function openRouteBuilder()", 1)[1].split("function closeRouteBuilder", 1)[0]
        self.assertNotIn("ensureLinkAnalysisView", route_builder_body)
        self.assertIn('target: "history"', app)
        self.assertIn('target: "core"', app)
        self.assertIn('target: "recovery"', app)
        self.assertIn('target: "route"', app)

    def test_offline_assistant_opens_recovery_builder(self) -> None:
        project = {
            "project_id": "recovery_fixture",
            "name": "Recovery fixture",
            "summary": {"capabilities": {"topology_criticality": "available", "restoration_screening": "available"}},
        }
        result = local_assistant_response(
            message="Plan the top 25 recovery priorities for roads or areas",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertIn("open_recovery_builder", {action["type"] for action in result["actions"]})

    def test_offline_assistant_treats_open_recovery_as_ui_navigation(self) -> None:
        project = {
            "project_id": "recovery_fixture",
            "name": "Recovery fixture",
            "summary": {"capabilities": {"topology_criticality": "available", "restoration_screening": "available"}},
        }
        result = local_assistant_response(
            message="Open recovery priorities for the latest critical-road result.",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertFalse(result["safety_precheck"])
        self.assertIn("open_recovery_builder", {action["type"] for action in result["actions"]})

    def test_offline_assistant_opens_route_builder_when_endpoints_are_missing(self) -> None:
        project = {
            "project_id": "route_fixture",
            "name": "Route fixture",
            "summary": {"capabilities": {"topology_criticality": "available", "route_detour_analysis": "available"}},
        }
        result = local_assistant_response(
            message="Compare a historical route and detour",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertIn("open_route_builder", {action["type"] for action in result["actions"]})
        self.assertNotIn("preview_analysis", {action["type"] for action in result["actions"]})

    def test_offline_assistant_opens_resilience_builder_and_research_validation(self) -> None:
        project = {
            "project_id": "resilience_fixture",
            "name": "Resilience fixture",
            "summary": {
                "capabilities": {
                    "topology_criticality": "available",
                    "progressive_robustness": "available",
                }
            },
        }
        stress = local_assistant_response(
            message="Open the network resilience test and let me choose the stages",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertFalse(stress["safety_precheck"])
        self.assertIn("open_stress_builder", {action["type"] for action in stress["actions"]})
        validation = local_assistant_response(
            message="Open scientific validation and compare with NetworkX",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertIn("models", {action["target"] for action in validation["actions"]})
        self.assertIn("scientific_validation", {action["target"] for action in validation["actions"]})

    def test_offline_assistant_navigates_matsim_map_and_memory(self) -> None:
        project = {
            "project_id": "matsim_fixture",
            "name": "MATSim fixture",
            "summary": {
                "capabilities": {"topology_criticality": "available", "route_detour_analysis": "available"},
                "metrics": {"capacity": {"available": True}},
                "transport_structure": {"od_count": 12},
            },
        }
        matsim = local_assistant_response(
            message="Open MATSim and simulate vehicle movement",
            project=project, selected_keys=[], bbox=None, current_metric="travel_time",
        )
        self.assertIn("open_matsim_builder", {action["type"] for action in matsim["actions"]})
        self.assertIn("full demand tier", matsim["message"])
        self.assertIn("field calibration is claimed only", matsim["message"])
        mapped = local_assistant_response(
            message="Open the geographic map",
            project=project, selected_keys=[], bbox=None, current_metric="travel_time",
        )
        self.assertIn("map", {action["target"] for action in mapped["actions"]})
        memory = local_assistant_response(
            message="Show Project Memory and the audit trail",
            project=project, selected_keys=[], bbox=None, current_metric="travel_time",
        )
        self.assertIn("memory", {action["target"] for action in memory["actions"]})

    def test_resilience_preview_round_trip_is_one_unambiguous_goal(self) -> None:
        project = {
            "project_id": "resilience_fixture",
            "name": "Resilience fixture",
            "summary": {
                "capabilities": {
                    "topology_criticality": "available",
                    "progressive_robustness": "available",
                }
            },
        }
        command = "Test network resilience by cumulatively removing the top 17 structurally critical roads."
        contract = interpret_command(
            text=command,
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertTrue(contract["ready"])
        self.assertEqual(contract["analysis_type"], "progressive_robustness")
        self.assertEqual(contract["top_k"], 17)
        self.assertNotIn("critical_components", contract.get("possible_intents", []))

    def test_offline_assistant_understands_two_point_scenario_routing(self) -> None:
        project = {
            "project_id": "route_fixture",
            "name": "Route fixture",
            "summary": {"capabilities": {"topology_criticality": "available", "route_detour_analysis": "available"}},
        }
        result = local_assistant_response(
            message="在点图上选两点，给出不同情景路径推荐",
            project=project,
            selected_keys=[],
            bbox=None,
            current_metric="travel_time",
        )
        self.assertIn("open_route_builder", {action["type"] for action in result["actions"]})
        self.assertIn("点", result["message"])


if __name__ == "__main__":
    unittest.main()
