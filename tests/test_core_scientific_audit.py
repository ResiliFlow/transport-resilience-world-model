from __future__ import annotations

import gzip
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

import networkx as nx

from engine.matsim_adapter import (
    LinkRecord,
    _feasible_activity_link_pair,
    _parse_events,
    _validated_matsim_changes,
    assess_matsim_readiness,
)
from engine.importers import import_csv_project
from engine.models import progressive_robustness, restoration_screening
from engine.routing import _route_between_links
from engine.traffic_assignment import compare_assignments


def link(key: str, origin: str, destination: str, weight: float = 1.0) -> tuple[str, str, str, dict]:
    return origin, destination, key, {"weight": weight}


def records_for(edges: list[tuple[str, str, str, dict]]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for index, (origin, destination, key, data) in enumerate(edges):
        records[key] = {
            "key": key,
            "graph_from": origin,
            "graph_to": destination,
            "from_node": origin,
            "to_node": destination,
            "region_id": "audit",
            "entity_id": key,
            "weight": float(data["weight"]),
            "latitude": float(index),
            "longitude": float(index),
            "length_m": 1000.0,
        }
    return records


def matsim_link(key: str, origin: str, destination: str) -> LinkRecord:
    return LinkRecord(
        key=key,
        region_id="audit",
        entity_id=key,
        from_node=origin,
        to_node=destination,
        length_m=1000.0,
        free_speed_mps=10.0,
        capacity_vph=900.0,
        coordinates=((0.0, 0.0), (1.0, 1.0)),
        capacity_uploaded=True,
        impedance_uploaded=True,
    )


class RouteAndScenarioScientificOracleTests(unittest.TestCase):
    def test_route_disconnection_distinguishes_direction_gap_from_network_split(self) -> None:
        direction_edges = [
            link("start", "A", "B"),
            link("reverse-only", "C", "B"),
            link("finish", "C", "D"),
        ]
        graph = nx.MultiDiGraph()
        graph.add_edges_from(direction_edges)
        result = _route_between_links(
            graph,
            records_for(direction_edges),
            "start",
            "finish",
        )
        self.assertFalse(result["reachable"])
        self.assertEqual(result["reason_code"], "direction_continuity_missing")
        self.assertTrue(result["weakly_connected"])
        self.assertIn("No reverse road is invented", result["reason"])

        split_edges = [
            link("start", "A", "B"),
            link("finish", "C", "D"),
        ]
        split = nx.MultiDiGraph()
        split.add_edges_from(split_edges)
        result = _route_between_links(split, records_for(split_edges), "start", "finish")
        self.assertFalse(result["reachable"])
        self.assertEqual(result["reason_code"], "different_weak_components")
        self.assertFalse(result["weakly_connected"])

    def test_proxy_assignment_reroutes_only_when_a_feasible_detour_exists(self) -> None:
        edges = [
            link("selected", "A", "B", 1.0),
            link("detour-1", "A", "C", 2.0),
            link("detour-2", "C", "B", 2.0),
            link("return", "B", "A", 1.0),
            link("bc", "B", "C", 1.0),
            link("ca", "C", "A", 1.0),
        ]
        baseline = nx.MultiDiGraph()
        baseline.add_edges_from(edges)
        scenario = baseline.copy()
        scenario.remove_edge("A", "B", "selected")
        result = compare_assignments(
            baseline,
            scenario,
            records_for(edges),
            seed_keys={"selected"},
            maximum_pairs=20,
        )
        self.assertGreaterEqual(result["rerouted_pair_count"], 1)
        self.assertEqual(result["disconnected_pair_count"], 0)
        self.assertTrue({"detour-1", "detour-2"}.issubset(
            {item["key"] for item in result["indirect_increases"]}
        ))
        self.assertTrue(result["verification"]["passed"])

        cut_edges = [
            link("selected", "A", "B", 1.0),
            link("return", "B", "A", 1.0),
        ]
        cut_baseline = nx.MultiDiGraph()
        cut_baseline.add_edges_from(cut_edges)
        cut_scenario = cut_baseline.copy()
        cut_scenario.remove_edge("A", "B", "selected")
        cut = compare_assignments(
            cut_baseline,
            cut_scenario,
            records_for(cut_edges),
            seed_keys={"selected"},
            maximum_pairs=10,
        )
        self.assertGreaterEqual(cut["disconnected_pair_count"], 1)
        self.assertEqual(cut["rerouted_pair_count"], 0)
        self.assertEqual(cut["indirect_increase_count"], 0)
        self.assertTrue(cut["verification"]["passed"])


class RecoveryAndStressScientificOracleTests(unittest.TestCase):
    @staticmethod
    def _efficiency(graph: nx.MultiDiGraph) -> float:
        nodes = sorted(graph.nodes)
        if len(nodes) < 2:
            return 0.0
        reciprocal = 0.0
        for source in nodes:
            lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
            reciprocal += sum(
                1.0 / distance
                for target, distance in lengths.items()
                if target != source and distance > 0
            )
        return reciprocal / (len(nodes) * (len(nodes) - 1))

    def _network_project(self, root: Path) -> tuple[Path, nx.MultiDiGraph]:
        source = root / "recovery_oracle.csv"
        fields = ["id", "from", "to", "travel_time", "length", "lat", "lon", "region"]
        rows = [
            ("e1", "A", "B", 1.0),
            ("e2", "B", "D", 1.0),
            ("e3", "A", "C", 2.0),
            ("e4", "C", "D", 2.0),
            ("e5", "D", "A", 1.0),
            ("e6", "B", "C", 1.0),
            ("e7", "C", "B", 1.0),
        ]
        graph = nx.MultiDiGraph()
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, (key, origin, destination, travel_time) in enumerate(rows):
                writer.writerow({
                    "id": key,
                    "from": origin,
                    "to": destination,
                    "travel_time": travel_time,
                    "length": 1000,
                    "lat": -33.8 + index * 0.001,
                    "lon": 151.0 + index * 0.001,
                    "region": "r",
                })
                graph.add_edge(origin, destination, key=f"r::{key}", weight=travel_time)
        database = root / "recovery_oracle.sqlite"
        mapping = {
            "entity_id": "id",
            "from_node": "from",
            "to_node": "to",
            "travel_time": "travel_time",
            "length": "length",
            "latitude": "lat",
            "longitude": "lon",
            "region": "region",
        }
        import_csv_project(
            files=[source],
            mapping=mapping,
            file_mappings={source.name: mapping},
            units={"travel_time": "min", "length": "m"},
            timezone="UTC",
            topology_node_scope="global",
            database_path=database,
            project_id="recovery_oracle",
            project_name="Recovery oracle",
            hazard_type="flood",
            source="scientific_audit",
        )
        return database, graph

    def test_adaptive_recovery_matches_independent_greedy_efficiency_oracle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_recovery_oracle_") as folder:
            database, full = self._network_project(Path(folder))
            selected = ["r::e1", "r::e2"]
            disrupted = full.copy()
            disrupted.remove_edge("A", "B", "r::e1")
            disrupted.remove_edge("B", "D", "r::e2")
            full_efficiency = self._efficiency(full)
            disrupted_efficiency = self._efficiency(disrupted)

            independent_gains: dict[str, float] = {}
            for key, origin, destination in (("r::e1", "A", "B"), ("r::e2", "B", "D")):
                trial = disrupted.copy()
                trial.add_edge(origin, destination, key=key, weight=float(full[origin][destination][key]["weight"]))
                independent_gains[key] = self._efficiency(trial) - disrupted_efficiency
            expected_first = max(selected, key=lambda key: (independent_gains[key], key))

            result = restoration_screening(
                database,
                selected_keys=selected,
                bbox=None,
                top_k=2,
                assumptions={"recovery_unit": "road", "priority_objective": "service_gain"},
            )
            self.assertEqual(result["items"][0]["key"], expected_first)
            self.assertAlmostEqual(
                result["items"][0]["marginal_efficiency_gain"],
                independent_gains[expected_first],
                places=12,
            )
            self.assertAlmostEqual(result["evidence"]["full_efficiency"], full_efficiency, places=12)
            self.assertAlmostEqual(result["evidence"]["disrupted_efficiency"], disrupted_efficiency, places=12)
            self.assertEqual(result["evidence"]["method"], "adaptive_marginal_efficiency")
            self.assertTrue(result["verification"]["passed"])

    def test_stress_curve_matches_independent_cumulative_removal_oracle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_stress_oracle_") as folder:
            database, full = self._network_project(Path(folder))
            order = ["r::e1", "r::e2"]
            result = progressive_robustness(
                database,
                top_k=2,
                bbox=None,
                failure_keys=order,
            )
            baseline = self._efficiency(full)
            oracle = []
            working = full.copy()
            for key, origin, destination in (("r::e1", "A", "B"), ("r::e2", "B", "D")):
                working.remove_edge(origin, destination, key)
                oracle.append(self._efficiency(working) / baseline)
            self.assertEqual([item["key"] for item in result["items"]], order)
            for item, expected in zip(result["items"], oracle):
                self.assertAlmostEqual(item["performance_ratio"], expected, places=12)
            self.assertTrue(result["verification"]["passed"])


class MatsimBoundaryScientificOracleTests(unittest.TestCase):
    def test_od_endpoint_selection_finds_a_feasible_incident_link_pair(self) -> None:
        links = [
            matsim_link("a-dead", "O", "X"),
            matsim_link("z-good", "O", "A"),
            matsim_link("middle", "A", "B"),
            matsim_link("d-destination", "B", "D"),
        ]
        graph = nx.DiGraph()
        graph.add_edges_from((item.from_node, item.to_node) for item in links)
        pair = _feasible_activity_link_pair(
            graph,
            [links[0], links[1]],
            [links[3]],
        )
        self.assertIsNotNone(pair)
        self.assertEqual((pair[0].key, pair[1].key), ("z-good", "d-destination"))

    def test_partial_supply_stays_in_exploratory_probe_tier_even_with_od(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_matsim_readiness_oracle_") as folder:
            database = Path(folder) / "project.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE entities (
                    region_id TEXT, entity_id TEXT, capacity REAL, length_m REAL,
                    free_flow_time REAL, baseline_speed REAL, baseline_travel_time REAL
                );
                CREATE TABLE inferred_network_links (
                    link_key TEXT, region_id TEXT, entity_id TEXT
                );
                CREATE TABLE transport_od (demand REAL);
                """
            )
            for index in range(3):
                entity = f"e{index}"
                connection.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                    ("r", entity, 900.0 if index == 0 else None, 1000.0, 1.0),
                )
                connection.execute(
                    "INSERT INTO inferred_network_links VALUES (?, ?, ?)",
                    (f"r::{entity}", "r", entity),
                )
            connection.execute("INSERT INTO transport_od VALUES (100)")
            connection.commit()
            connection.close()

            readiness = assess_matsim_readiness({"database_path": str(database), "summary": {}})
            self.assertEqual(readiness["tier"], "exploratory_probe")
            self.assertFalse(readiness["full_supply_ready"])
            self.assertAlmostEqual(readiness["capacity_coverage"], 1 / 3, places=12)
            self.assertIn("road capacity on every routable link (1/3)", readiness["missing_for_full"])

    def test_matsim_boundary_rejects_ungrounded_or_invalid_scenario_changes(self) -> None:
        links = [matsim_link("road", "A", "B")]
        with self.assertRaisesRegex(ValueError, "cannot ground"):
            _validated_matsim_changes(
                links,
                [{"key": "missing", "effect": "closed", "magnitude": 1.0}],
            )
        with self.assertRaisesRegex(ValueError, "between 1% and 95%"):
            _validated_matsim_changes(
                links,
                [{"key": "road", "effect": "speed_reduction", "magnitude": 0.99}],
            )

    def test_even_trip_count_uses_the_statistical_median(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_matsim_events_oracle_") as folder:
            events = Path(folder) / "events.xml.gz"
            content = """<events>
                <event time="0" type="departure" person="p1" />
                <event time="10" type="arrival" person="p1" />
                <event time="0" type="departure" person="p2" />
                <event time="20" type="arrival" person="p2" />
            </events>"""
            with gzip.open(events, "wt", encoding="utf-8") as stream:
                stream.write(content)
            parsed = _parse_events(events)
            self.assertEqual(parsed["completed_trip_count"], 2)
            self.assertEqual(parsed["median_trip_time_seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
