from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from engine.assistant import (
    AssistantRequestContext,
    _ground_core_analysis_action,
    _safe_project_context,
)
from engine.importers import import_csv_project, preview_csv


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _import(project_id: str, database: Path, files: list[Path]) -> dict:
    previews = [preview_csv(path) for path in files]
    file_mappings = {
        path.name: preview["suggested_mapping"]["roles"]
        for path, preview in zip(files, previews)
    }
    summary = import_csv_project(
        files=files,
        mapping=file_mappings[files[0].name],
        file_mappings=file_mappings,
        units={
            "speed": "uploaded unit",
            "travel_time": "uploaded unit",
            "primary_metric": "uploaded unit",
        },
        timezone="fixture convention",
        topology_node_scope="global",
        entity_id_scope="global",
        database_path=database,
        project_id=project_id,
        project_name=f"LLM cross-data fixture {project_id}",
        hazard_type="other",
        source="llm_cross_data_validation",
    )
    return {
        "project_id": project_id,
        "name": f"LLM cross-data fixture {project_id}",
        "hazard_type": "other",
        "source": "llm_cross_data_validation",
        "mapping": file_mappings[files[0].name],
        "summary": summary,
    }


def _support(project: dict) -> dict:
    return _safe_project_context(project)["data_profile"]["llm_execution_support"]


def _provider_result(tool: str, *, count: int = 10) -> dict:
    return {
        "message": "I will run the requested analysis.",
        "actions": [],
        "analysis_request": {
            "tool": tool,
            "requested_count": count,
            "metric": "current",
            "scope": "whole_project",
            "time_mode": "current",
        },
        "understanding": {
            "intent": "calculation_request",
            "requires_computation": True,
            "needs_clarification": False,
        },
    }


def _context(project: dict, message: str) -> AssistantRequestContext:
    return AssistantRequestContext(
        message=message,
        project=project,
        registry={"models": []},
        selected_keys=[],
        bbox=None,
        current_metric="speed",
        current_time=None,
        current_view="explore",
        history=[],
    )


class CrossDataLLMGeneralityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="twm_llm_cross_data_")
        cls.root = Path(cls.temporary.name)

        detector = cls.root / "brisbane_detector_history.csv"
        _write(
            detector,
            ["sensor_ref", "longitude", "latitude", "recorded_at", "velocity_observed"],
            [
                {"sensor_ref": "D1", "longitude": 153.02, "latitude": -27.47, "recorded_at": "2025-02-01T00:00:00Z", "velocity_observed": 42},
                {"sensor_ref": "D1", "longitude": 153.02, "latitude": -27.47, "recorded_at": "2025-02-01T00:10:00Z", "velocity_observed": 31},
                {"sensor_ref": "D2", "longitude": 153.04, "latitude": -27.49, "recorded_at": "2025-02-01T00:00:00Z", "velocity_observed": 38},
                {"sensor_ref": "D2", "longitude": 153.04, "latitude": -27.49, "recorded_at": "2025-02-01T00:10:00Z", "velocity_observed": 34},
            ],
        )

        direct = cls.root / "direct_roads.csv"
        _write(
            direct,
            ["edge_key", "u", "v", "u_lon", "u_lat", "v_lon", "v_lat", "length_m", "travel_time_min"],
            [
                {"edge_key": "E1", "u": "A", "v": "B", "u_lon": 151.00, "u_lat": -33.80, "v_lon": 151.02, "v_lat": -33.81, "length_m": 2100, "travel_time_min": 4},
                {"edge_key": "E2", "u": "B", "v": "C", "u_lon": 151.02, "u_lat": -33.81, "v_lon": 151.04, "v_lat": -33.82, "length_m": 2400, "travel_time_min": 5},
            ],
        )

        nodes = cls.root / "nodes_inventory.csv"
        links = cls.root / "links_inventory.csv"
        _write(nodes, ["node_id", "lon", "lat"], [
            {"node_id": "N1", "lon": 174.76, "lat": -36.85},
            {"node_id": "N2", "lon": 174.78, "lat": -36.86},
            {"node_id": "N3", "lon": 174.80, "lat": -36.87},
        ])
        _write(links, ["link_id", "from", "to", "length_m", "travel_time_min"], [
            {"link_id": "L1", "from": "N1", "to": "N2", "length_m": 1800, "travel_time_min": 3},
            {"link_id": "L2", "from": "N2", "to": "N3", "length_m": 1900, "travel_time_min": 3.5},
        ])

        topology = cls.root / "topology_only.csv"
        _write(topology, ["arc_id", "tail", "head", "length_m", "travel_time_min"], [
            {"arc_id": "T1", "tail": "x", "head": "y", "length_m": 1000, "travel_time_min": 2},
            {"arc_id": "T2", "tail": "y", "head": "z", "length_m": 1200, "travel_time_min": 2.5},
            {"arc_id": "T3", "tail": "z", "head": "x", "length_m": 1400, "travel_time_min": 3},
        ])

        package = cls.root / "full_transport_package.csv"
        fields = [
            "record_type", "node_id", "lon", "lat", "link_id", "source", "target",
            "length", "travel_time", "free_flow_time", "capacity", "origin", "destination", "demand",
        ]
        _write(package, fields, [
            {"record_type": "node", "node_id": "P1", "lon": 13.40, "lat": 52.50},
            {"record_type": "node", "node_id": "P2", "lon": 13.42, "lat": 52.51},
            {"record_type": "node", "node_id": "P3", "lon": 13.44, "lat": 52.52},
            {"record_type": "link", "link_id": "R1", "source": "P1", "target": "P2", "length": 1500, "travel_time": 3, "free_flow_time": 2.5, "capacity": 1200},
            {"record_type": "link", "link_id": "R2", "source": "P2", "target": "P3", "length": 1600, "travel_time": 3.2, "free_flow_time": 2.7, "capacity": 1100},
            {"record_type": "od", "origin": "P1", "destination": "P3", "demand": 250},
        ])

        static_points = cls.root / "static_fire_locations.csv"
        _write(static_points, ["asset_code", "longitude", "latitude", "burn_severity"], [
            {"asset_code": "F1", "longitude": 144.95, "latitude": -37.81, "burn_severity": 0.8},
            {"asset_code": "F2", "longitude": 144.98, "latitude": -37.83, "burn_severity": 0.4},
        ])

        projected = cls.root / "projected_sensors.csv"
        _write(projected, ["station", "easting", "northing", "timestamp", "speed_kph"], [
            {"station": "Q1", "easting": 334368, "northing": 6250945, "timestamp": "2025-01-01T00:00:00Z", "speed_kph": 40},
            {"station": "Q2", "easting": 334500, "northing": 6251100, "timestamp": "2025-01-01T00:10:00Z", "speed_kph": 35},
        ])

        cls.projects = {
            "detector": _import("detector", cls.root / "detector.sqlite", [detector]),
            "direct": _import("direct", cls.root / "direct.sqlite", [direct]),
            "split": _import("split", cls.root / "split.sqlite", [nodes, links]),
            "topology": _import("topology", cls.root / "topology.sqlite", [topology]),
            "full": _import("full", cls.root / "full.sqlite", [package]),
            "static": _import("static", cls.root / "static.sqlite", [static_points]),
            "projected": _import("projected", cls.root / "projected.sqlite", [projected]),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_seven_transport_archetypes_are_not_collapsed_into_one_template(self) -> None:
        expected = {
            "detector": "geographic_point_time_series",
            "direct": "geographic_link_network",
            "split": "geographic_link_network",
            "topology": "topology_only_network",
            "full": "demand_annotated_transport_network",
            "static": "static_geographic_points",
            "projected": "non_geographic_point_data",
        }
        for key, archetype in expected.items():
            with self.subTest(key=key):
                self.assertEqual(_support(self.projects[key])["data_archetype"], archetype)

    def test_context_preserves_current_upload_fields_and_never_injects_unrelated_project_facts(self) -> None:
        context = _safe_project_context(self.projects["detector"])
        rendered = json.dumps(context, ensure_ascii=False)
        self.assertIn("velocity_observed", rendered)
        self.assertIn("153.02", rendered)
        self.assertNotIn("unrelated-project", rendered)
        self.assertNotIn("unrelated-region", rendered)
        self.assertNotIn("unrelated-hazard", rendered)

    def test_point_time_series_can_offer_spatial_analysis_but_not_network_tools(self) -> None:
        support = _support(self.projects["detector"])
        tools = {item["tool"]: item for item in support["core_tool_support"]}
        self.assertTrue(tools["critical_areas"]["can_start"])
        self.assertFalse(tools["critical_roads"]["can_start"])
        self.assertFalse(tools["routes_under_disruption"]["can_start"])
        self.assertFalse(support["matsim_support"]["can_prepare"])

    def test_all_six_core_tools_are_gated_against_every_data_archetype(self) -> None:
        expected = {
            "detector": {"critical_areas"},
            "direct": {
                "critical_roads", "critical_areas", "recovery_priorities",
                "routes_under_disruption", "network_resilience", "disruption_scenario",
            },
            "split": {
                "critical_roads", "critical_areas", "recovery_priorities",
                "routes_under_disruption", "network_resilience", "disruption_scenario",
            },
            "topology": {
                "critical_roads", "recovery_priorities", "routes_under_disruption",
                "network_resilience", "disruption_scenario",
            },
            "full": {
                "critical_roads", "critical_areas", "recovery_priorities",
                "routes_under_disruption", "network_resilience", "disruption_scenario",
            },
            "static": set(),
            "projected": set(),
        }
        for project_name, runnable in expected.items():
            support = _support(self.projects[project_name])
            actual = {
                item["tool"] for item in support["core_tool_support"] if item["can_start"]
            }
            with self.subTest(project=project_name):
                self.assertEqual(actual, runnable)

    def test_all_six_llm_actions_follow_the_cross_data_execution_matrix(self) -> None:
        prompts = {
            "critical_roads": "Find the top 5 critical roads.",
            "critical_areas": "Find the top 5 critical areas.",
            "recovery_priorities": "Plan the top 5 recovery priorities.",
            "routes_under_disruption": "Plan routes under disruption.",
            "network_resilience": "Test network resilience.",
            "disruption_scenario": "Build a disruption scenario.",
        }
        for project_name, project in self.projects.items():
            support = _support(project)
            expected = {
                item["tool"]: bool(item["can_start"])
                for item in support["core_tool_support"]
            }
            for tool, prompt in prompts.items():
                with self.subTest(project=project_name, tool=tool):
                    result = _ground_core_analysis_action(
                        _context(project, prompt),
                        _provider_result(tool, count=5),
                    )
                    self.assertEqual(bool(result["actions"]), expected[tool])

    def test_topology_only_network_keeps_network_tools_but_withholds_geographic_areas(self) -> None:
        support = _support(self.projects["topology"])
        tools = {item["tool"]: item for item in support["core_tool_support"]}
        self.assertTrue(tools["critical_roads"]["can_start"])
        self.assertTrue(tools["routes_under_disruption"]["can_start"])
        self.assertFalse(tools["critical_areas"]["can_start"])
        self.assertFalse(support["geographic"])
        self.assertEqual(support["geometry_representation"], "topology_layout_not_real_geography")

    def test_complete_demand_package_is_distinguished_from_exploratory_matsim(self) -> None:
        full = _support(self.projects["full"])
        direct = _support(self.projects["direct"])
        self.assertTrue(full["matsim_support"]["can_prepare"])
        self.assertEqual(full["matsim_support"]["tier"], "uploaded_demand")
        self.assertTrue(direct["matsim_support"]["can_prepare"])
        self.assertEqual(direct["matsim_support"]["tier"], "exploratory_probe")
        self.assertIn("not observed demand", direct["matsim_support"]["claim_boundary"])

    def test_static_and_projected_points_do_not_gain_time_or_real_geography_claims(self) -> None:
        static = _support(self.projects["static"])
        projected = _support(self.projects["projected"])
        self.assertFalse(static["has_time_series_metric"])
        self.assertFalse(projected["geographic"])
        self.assertEqual(projected["geometry_representation"], "non_geographic_uploaded_points")
        self.assertEqual(projected["claim_boundaries"]["real_geography"], "not available; any layout is non-geographic")

    def test_llm_blocks_an_impossible_network_request_instead_of_opening_a_dead_workflow(self) -> None:
        project = self.projects["detector"]
        result = _ground_core_analysis_action(
            _context(project, "Find the top 20 critical roads."),
            _provider_result("critical_roads", count=20),
        )
        self.assertEqual(result["actions"], [])
        self.assertIn("cannot run on this upload", result["message"])
        self.assertIn("routable From/To road network", result["message"])

    def test_llm_runs_supported_tools_and_blocks_only_the_incompatible_one(self) -> None:
        detector_area = _ground_core_analysis_action(
            _context(self.projects["detector"], "Find the top 5 critical areas."),
            _provider_result("critical_areas", count=5),
        )
        topology_roads = _ground_core_analysis_action(
            _context(self.projects["topology"], "Find the top 5 critical roads."),
            _provider_result("critical_roads", count=5),
        )
        topology_areas = _ground_core_analysis_action(
            _context(self.projects["topology"], "Find the top 5 critical areas."),
            _provider_result("critical_areas", count=5),
        )
        self.assertEqual(detector_area["actions"][0]["type"], "preview_analysis")
        self.assertEqual(topology_roads["actions"][0]["type"], "preview_analysis")
        self.assertEqual(topology_areas["actions"], [])

    def test_matsim_gate_uses_current_project_structure_not_a_demonstration_default(self) -> None:
        detector = _ground_core_analysis_action(
            _context(self.projects["detector"], "Run a MATSim simulation."),
            _provider_result("matsim"),
        )
        direct = _ground_core_analysis_action(
            _context(self.projects["direct"], "Run a MATSim simulation."),
            _provider_result("matsim"),
        )
        self.assertEqual(detector["actions"], [])
        self.assertIn("MATSim cannot be prepared", detector["message"])
        self.assertEqual(direct["actions"][0]["type"], "open_matsim_builder")


if __name__ == "__main__":
    unittest.main()
