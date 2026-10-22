from __future__ import annotations

import random
import unittest

import networkx as nx

from engine.routing import _route_alternatives_between_links, _route_between_links


def _case(seed: int) -> tuple[nx.MultiDiGraph, dict[str, dict[str, object]]]:
    randomiser = random.Random(seed)
    node_count = randomiser.randint(5, 11)
    nodes = [f"n{index}" for index in range(node_count)]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    records: dict[str, dict[str, object]] = {}
    edge_index = 0
    for origin in nodes:
        for destination in nodes:
            if origin == destination or randomiser.random() > 0.27:
                continue
            key = f"e{edge_index}"
            edge_index += 1
            weight = round(randomiser.uniform(0.2, 12.0), 6)
            graph.add_edge(origin, destination, key=key, weight=weight)
            records[key] = {
                "key": key,
                "graph_from": origin,
                "graph_to": destination,
                "from_node": origin,
                "to_node": destination,
                "region_id": "property",
                "entity_id": key,
                "weight": weight,
                "length_m": weight * 100.0,
            }
    if len(records) < 2:
        for origin, destination in ((nodes[0], nodes[1]), (nodes[1], nodes[2])):
            key = f"e{edge_index}"
            edge_index += 1
            graph.add_edge(origin, destination, key=key, weight=1.0)
            records[key] = {
                "key": key,
                "graph_from": origin,
                "graph_to": destination,
                "from_node": origin,
                "to_node": destination,
                "region_id": "property",
                "entity_id": key,
                "weight": 1.0,
                "length_m": 100.0,
            }
    return graph, records


class RandomisedRoutePropertyTests(unittest.TestCase):
    def test_route_cost_and_disconnection_match_independent_networkx_oracle(self) -> None:
        for seed in range(75):
            with self.subTest(seed=seed):
                graph, records = _case(seed)
                keys = sorted(records)
                chooser = random.Random(10_000 + seed)
                origin_key, destination_key = chooser.sample(keys, 2)
                origin = records[origin_key]
                destination = records[destination_key]
                source = str(origin["graph_to"])
                target = str(destination["graph_from"])
                expected_reachable = nx.has_path(graph, source, target)

                result = _route_between_links(graph, records, origin_key, destination_key)
                self.assertEqual(result["reachable"], expected_reachable)
                if not expected_reachable:
                    weak = nx.has_path(graph.to_undirected(as_view=True), source, target)
                    expected_reason = (
                        "direction_continuity_missing" if weak else "different_weak_components"
                    )
                    self.assertEqual(result["reason_code"], expected_reason)
                    continue

                middle = nx.shortest_path_length(graph, source, target, weight="weight")
                expected_cost = (
                    float(records[origin_key]["weight"])
                    + float(middle)
                    + float(records[destination_key]["weight"])
                )
                self.assertAlmostEqual(float(result["impedance"]), expected_cost, places=9)
                self.assertEqual(result["keys"][0], origin_key)
                self.assertEqual(result["keys"][-1], destination_key)
                self.assertEqual(result["link_count"], len(result["keys"]))
                self.assertTrue(all(float(value) > 0 for value in result["edge_weights"]))

    def test_alternative_routes_are_repeatable_loopless_and_respect_overlap(self) -> None:
        tested = 0
        for seed in range(120, 220):
            graph, records = _case(seed)
            keys = sorted(records)
            if len(keys) < 2:
                continue
            origin_key, destination_key = random.Random(seed).sample(keys, 2)
            first, first_search = _route_alternatives_between_links(
                graph,
                records,
                origin_key,
                destination_key,
                count=4,
                maximum_overlap=0.85,
            )
            second, second_search = _route_alternatives_between_links(
                graph,
                records,
                origin_key,
                destination_key,
                count=4,
                maximum_overlap=0.85,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_search, second_search)
            reachable = [route for route in first if route.get("reachable")]
            if not reachable:
                continue
            tested += 1
            sequences = [tuple(route["keys"]) for route in reachable]
            self.assertEqual(len(sequences), len(set(sequences)))
            self.assertEqual(
                [float(route["impedance"]) for route in reachable],
                sorted(float(route["impedance"]) for route in reachable),
            )
            for route in reachable:
                self.assertEqual(len(route["nodes"]), len(set(route["nodes"])))
                overlap = float(route["maximum_overlap_with_retained"])
                self.assertGreaterEqual(overlap, 0.0)
                self.assertLessEqual(overlap, 1.0)
            self.assertEqual(
                bool(first_search["overlap_relaxed"]),
                any(float(route["maximum_overlap_with_retained"]) > 0.85 for route in reachable),
            )
        self.assertGreaterEqual(tested, 20)


if __name__ == "__main__":
    unittest.main()
