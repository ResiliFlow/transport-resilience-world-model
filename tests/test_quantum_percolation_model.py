from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.importers import import_csv_project, suggest_mapping
from engine.model_router import model_recommendation
from engine.models import spatial_hotspots
from engine.quantum_percolation import (
    DPCNumericalContract,
    PropagationEdge,
    propagation_connectivity,
    regional_dpc_impacts,
)


class QuantumPercolationKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = [
            PropagationEdge("0", "1", 1.0, "west"),
            PropagationEdge("1", "2", 0.8, "west"),
            PropagationEdge("2", "3", 0.9, "east"),
            PropagationEdge("3", "4", 1.0, "east"),
            PropagationEdge("1", "4", 0.2, "connector"),
        ]

    def test_dense_and_full_dimension_lanczos_agree(self) -> None:
        dense, _ = propagation_connectivity(
            self.edges,
            contract=DPCNumericalContract(dense_node_limit=100, krylov_dimension=10),
        )
        lanczos, diagnostics = propagation_connectivity(
            self.edges,
            contract=DPCNumericalContract(dense_node_limit=2, krylov_dimension=10),
        )
        self.assertAlmostEqual(dense, lanczos, places=9)
        self.assertEqual(diagnostics["basis_dimension"], 5)

    def test_scores_are_deterministic_non_negative_and_scale_invariant(self) -> None:
        first = regional_dpc_impacts(self.edges, ["west", "east", "connector"])
        second = regional_dpc_impacts(self.edges, ["connector", "east", "west"])
        self.assertEqual(first, second)
        self.assertTrue(
            all(float(item["regional_dpc_score"]) >= 0 for item in first["impacts"])
        )
        scaled = [
            PropagationEdge(edge.from_node, edge.to_node, edge.strength * 7.0, edge.region_id)
            for edge in self.edges
        ]
        scaled_value, _ = propagation_connectivity(scaled)
        self.assertAlmostEqual(float(first["baseline_dpc"]), scaled_value, places=9)

    def test_zero_degradation_has_zero_regional_score(self) -> None:
        result = regional_dpc_impacts(
            self.edges,
            ["west", "east"],
            contract=DPCNumericalContract(degradation_alpha=0.0),
        )
        self.assertTrue(
            all(abs(float(item["regional_dpc_score"])) <= 1e-12 for item in result["impacts"])
        )

    def test_small_graph_quadrature_and_ranking_sensitivity_are_bounded(self) -> None:
        default = regional_dpc_impacts(self.edges, ["west", "east", "connector"])
        refined = regional_dpc_impacts(
            self.edges,
            ["west", "east", "connector"],
            contract=DPCNumericalContract(sample_count=257),
        )
        longer = regional_dpc_impacts(
            self.edges,
            ["west", "east", "connector"],
            contract=DPCNumericalContract(scaled_horizon=60.0, sample_count=129),
        )
        relative_quadrature_change = abs(
            float(default["baseline_dpc"]) - float(refined["baseline_dpc"])
        ) / float(refined["baseline_dpc"])
        relative_horizon_change = abs(
            float(default["baseline_dpc"]) - float(longer["baseline_dpc"])
        ) / float(longer["baseline_dpc"])
        self.assertLess(relative_quadrature_change, 0.001)
        self.assertLess(relative_horizon_change, 0.02)
        self.assertEqual(
            [item["region_id"] for item in default["impacts"]],
            [item["region_id"] for item in refined["impacts"]],
        )

    def test_truncated_lanczos_boundary_is_visible(self) -> None:
        edges = [
            PropagationEdge(str(index), str(index + 1), 1.0, "left" if index < 20 else "right")
            for index in range(40)
        ]
        result = regional_dpc_impacts(
            edges,
            ["left", "right"],
            contract=DPCNumericalContract(dense_node_limit=2, krylov_dimension=8),
        )
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["basis_dimension"], 8)
        self.assertGreaterEqual(float(diagnostics["maximum_krylov_residual"]), 0.0)
        self.assertFalse(diagnostics["paper_long_time_limit_exactly_evaluated"])
        self.assertIn("not a certified DPC error bound", diagnostics["residual_interpretation"])


class QuantumPercolationIntegrationTests(unittest.TestCase):
    def test_critical_area_uses_executable_dpc_tier_when_data_support_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_dpc_") as folder:
            root = Path(folder)
            source = root / "dpc_network.csv"
            fields = [
                "road_id", "from_node", "to_node", "length_m", "latitude",
                "longitude", "observed_at", "travel_time_min", "speed_kmh", "region_id",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for link_index in range(24):
                    cell_offset = 0.0 if link_index < 12 else 0.10
                    for time_index, timestamp in enumerate(
                        ("2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00")
                    ):
                        writer.writerow(
                            {
                                "road_id": f"e{link_index}",
                                "from_node": f"n{link_index}",
                                "to_node": f"n{link_index + 1}",
                                "length_m": 1000.0,
                                "latitude": -33.86,
                                "longitude": 151.20 + cell_offset,
                                "observed_at": timestamp,
                                "travel_time_min": (
                                    "" if link_index < 6
                                    else 1.0 + time_index * (0.03 if link_index < 12 else 0.12)
                                ),
                                "speed_kmh": 60.0 - time_index * (1.0 if link_index < 12 else 4.0),
                                "region_id": "network",
                            }
                        )
            mapping = suggest_mapping(fields)["roles"]
            database = root / "project.sqlite"
            summary = import_csv_project(
                files=[source],
                mapping=mapping,
                units={"travel_time": "min", "length": "m"},
                timezone="UTC",
                database_path=database,
                project_id="dpc_integration",
                project_name="DPC integration",
                hazard_type="flood",
                source="automated_test",
            )
            result = spatial_hotspots(
                database,
                metric="speed",
                top_k=2,
                bbox=None,
                timestamp=summary["time_values"][-1],
            )
            self.assertEqual(result["evidence"]["executed_tier"], "regional_dynamic_propagation_connectivity")
            self.assertEqual(len(result["items"]), 2)
            self.assertTrue(all("regional_dpc_score" in item for item in result["items"]))
            self.assertGreaterEqual(result["evidence"]["dpc_inputs"]["dpc_link_coverage"], 0.80)
            source_counts = result["evidence"]["dpc_inputs"]["selected_time_source_counts"]
            self.assertGreater(source_counts["uploaded_travel_time"], 0)
            self.assertGreater(source_counts["derived_from_speed_and_length"], 0)
            self.assertEqual(result["evidence"]["dpc_inputs"]["display_metric_requested"], "speed")
            self.assertEqual(
                result["evidence"]["dpc_numerical_diagnostics"]["hardware"],
                "classical CPU numerical linear algebra",
            )
            self.assertTrue(result["verification"]["passed"])

            candidate_ids = [item["cell_id"] for item in result["items"]]
            zero_result = {
                "baseline_dpc": 5.0,
                "impacts": [
                    {
                        "region_id": cell_id,
                        "baseline_dpc": 5.0,
                        "degraded_dpc": 5.0,
                        "signed_dpc_impact": 0.0,
                        "regional_dpc_score": 0.0,
                    }
                    for cell_id in candidate_ids
                ],
                "diagnostics": {
                    "hardware": "classical CPU numerical linear algebra",
                    "maximum_krylov_residual": 0.0,
                },
            }
            with patch("engine.models.regional_dpc_impacts", return_value=zero_result):
                zero_map = spatial_hotspots(
                    database,
                    metric="speed",
                    top_k=2,
                    bbox=None,
                    timestamp=summary["time_values"][-1],
                )
            self.assertTrue(
                all(feature["properties"]["severity"] == 0.0 for feature in zero_map["feature_collection"]["features"])
            )

    def test_insufficient_dpc_structure_falls_back_with_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="twm_dpc_fallback_") as folder:
            root = Path(folder)
            source = root / "one_cell.csv"
            fields = [
                "road_id", "from_node", "to_node", "length_m", "latitude",
                "longitude", "observed_at", "travel_time_min",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for link_index in range(12):
                    for time_index, timestamp in enumerate(("2026-01-01 00:00:00", "2026-01-01 00:05:00")):
                        writer.writerow({
                            "road_id": f"e{link_index}",
                            "from_node": f"n{link_index}",
                            "to_node": f"n{link_index + 1}",
                            "length_m": 1000.0,
                            "latitude": -33.86,
                            "longitude": 151.20 + link_index * 0.0001,
                            "observed_at": timestamp,
                            "travel_time_min": 1.0 + 0.1 * time_index,
                        })
            database = root / "fallback.sqlite"
            summary = import_csv_project(
                files=[source],
                mapping=suggest_mapping(fields)["roles"],
                units={"travel_time": "min", "length": "m"},
                timezone="UTC",
                database_path=database,
                project_id="dpc_fallback",
                project_name="DPC fallback",
                hazard_type="flood",
                source="automated_test",
            )
            result = spatial_hotspots(
                database,
                metric="travel_time",
                top_k=5,
                bbox=None,
                timestamp=summary["time_values"][-1],
            )
            self.assertEqual(result["evidence"]["executed_tier"], "observed_fixed_grid_degradation")
            self.assertIn("at least two 5 km candidate cells", result["evidence"]["dpc_unavailable_reason"])
            self.assertTrue(all("regional_dpc_score" not in item for item in result["items"]))


class ModelTierRoutingTests(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        return {
            "mapped_entity_count": 100,
            "time_values": [1, 2, 3],
            "capabilities": {"topology_criticality": "available"},
            "metrics": {
                "speed": {"available": True},
                "travel_time": {"available": True},
                "length": {"available": True},
                "capacity": {"available": True},
                "free_flow_time": {"available": False},
            },
            "od_demand_available": True,
        }

    def test_dpc_and_uploaded_demand_matsim_tiers_are_data_routed(self) -> None:
        summary = self._summary()
        dpc = model_recommendation("P5-M10", summary)
        self.assertEqual(dpc["selected_tier"], "Regional Dynamic Propagation Connectivity under observed travel-time states")
        matsim = model_recommendation("P5-M21", summary)
        self.assertEqual(matsim["assumption_level"], "uploaded_demand_and_supply")
        self.assertIn("Uploaded-demand MATSim", matsim["selected_tier"])

    def test_topology_without_uploaded_demand_routes_to_probe_not_full_tier(self) -> None:
        summary = self._summary()
        summary["od_demand_available"] = False
        matsim = model_recommendation("P5-M22", summary)
        self.assertEqual(matsim["assumption_level"], "reproducible_probe")
        self.assertIn("exploratory-probe", matsim["selected_tier"])
        self.assertIn("od_demand", matsim["missing_for_high_fidelity"])


if __name__ == "__main__":
    unittest.main()
