from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from engine.assistant import AssistantRequestContext, assistant_response, local_assistant_response
from engine.importers import import_csv_project, preview_csv
from engine.models import map_features, persistent_degradation


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "synthetic_acceptance"


def import_fixture(
    database: Path,
    files: list[Path],
    *,
    project_id: str,
    hazard_type: str,
    units: dict[str, str] | None = None,
) -> dict:
    mapping = preview_csv(files[0])["suggested_mapping"]["roles"]
    return import_csv_project(
        files=files,
        mapping=mapping,
        units=units or {"speed": "uploaded unit", "travel_time": "uploaded unit", "primary_metric": "uploaded unit"},
        timezone="fixture convention; normalised to UTC",
        database_path=database,
        project_id=project_id,
        project_name=f"Generated acceptance {project_id}",
        hazard_type=hazard_type,
        source="deterministic_synthetic_acceptance_fixture",
    )


class GeneratedSemanticAndGeographicTests(unittest.TestCase):
    def test_chinese_semicolon_schema_is_recognised(self) -> None:
        preview = preview_csv(FIXTURES / "flood_chinese_semicolon.csv")
        roles = preview["suggested_mapping"]["roles"]
        self.assertEqual(preview["delimiter"], ";")
        self.assertEqual(roles["entity_id"], "路段编号")
        self.assertEqual(roles["from_node"], "起始节点")
        self.assertEqual(roles["to_node"], "终止节点")
        self.assertEqual(roles["latitude"], "中心纬度")
        self.assertEqual(roles["longitude"], "中心经度")
        self.assertEqual(roles["primary_metric"], "洪水深度")
        self.assertEqual(preview["coordinate_diagnostics"]["status"], "valid")
        self.assertEqual(preview["mapping_health"]["status"], "ready")

    def test_swapped_coordinates_are_corrected_for_confirmation(self) -> None:
        preview = preview_csv(FIXTURES / "swapped_latitude_longitude.csv")
        roles = preview["suggested_mapping"]["roles"]
        self.assertEqual(roles["latitude"], "Longitude")
        self.assertEqual(roles["longitude"], "Latitude")
        self.assertEqual(preview["coordinate_diagnostics"]["status"], "review_swap")
        self.assertTrue(preview["coordinate_diagnostics"]["recommended_swap"])
        self.assertEqual(preview["mapping_health"]["status"], "review")

    def test_projected_coordinates_open_as_disclosed_non_geographic_view(self) -> None:
        path = FIXTURES / "projected_coordinates_requires_reprojection.csv"
        preview = preview_csv(path)
        self.assertEqual(preview["coordinate_diagnostics"]["status"], "projected_coordinates")
        self.assertEqual(preview["mapping_health"]["status"], "ready")
        self.assertEqual(
            preview["transport_structure"]["normalization"]["target_view"],
            "non_geographic_points",
        )
        with tempfile.TemporaryDirectory(prefix="p5_projected_") as folder:
            summary = import_fixture(
                Path(folder) / "project.sqlite",
                [path],
                project_id="projected",
                hazard_type="other",
            )
        self.assertEqual(summary["mapped_entity_count"], 2)
        self.assertFalse(summary["transport_structure"]["geographic"])
        self.assertEqual(summary["transport_structure"]["geography_mode"], "normalised_uploaded_geometry")

    def test_utf16_tabbed_file_is_readable(self) -> None:
        preview = preview_csv(FIXTURES / "utf16_tabbed_storm.tsv")
        self.assertEqual(preview["delimiter"], "\t")
        self.assertEqual(preview["suggested_mapping"]["roles"]["primary_metric"], "Wind Gust")
        self.assertEqual(preview["mapping_health"]["status"], "ready")

    def test_missing_identifier_gets_a_disclosed_deterministic_internal_id(self) -> None:
        path = FIXTURES / "missing_required_identifier.csv"
        preview = preview_csv(path)
        self.assertEqual(preview["mapping_health"]["status"], "ready")
        self.assertEqual(preview["transport_structure"]["joint_counts"]["generated_node_ids"], 1)
        self.assertTrue(preview["transport_structure"]["normalization"]["warnings"])
        with tempfile.TemporaryDirectory(prefix="p5_generated_id_") as folder:
            summary = import_fixture(
                Path(folder) / "project.sqlite",
                [path],
                project_id="generated_id",
                hazard_type="flood",
            )
        self.assertEqual(summary["entity_count"], 1)
        self.assertEqual(summary["mapped_entity_count"], 1)


class GeneratedImportAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="prototype5_generated_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_mixed_hazard_files_with_different_languages_and_headers(self) -> None:
        database = self.root / "mixed.sqlite"
        summary = import_fixture(
            database,
            [FIXTURES / "wildfire_camelcase_timezone.csv", FIXTURES / "flood_chinese_semicolon.csv"],
            project_id="mixed_headers",
            hazard_type="multi_hazard",
            units={"speed": "uploaded per-file unit", "travel_time": "uploaded per-file unit", "primary_metric": "uploaded per-file unit"},
        )
        self.assertEqual(summary["entity_count"], 7)
        self.assertEqual(summary["mapped_entity_count"], 7)
        self.assertEqual(summary["observation_count"], 28)
        self.assertEqual(summary["region_count"], 2)
        self.assertEqual(summary["capabilities"]["topology_criticality"], "available")
        second_mapping = summary["source_files"][1]["resolved_mapping"]
        self.assertEqual(second_mapping["entity_id"], "路段编号")
        self.assertEqual(second_mapping["primary_metric"], "洪水深度")
        wildfire_collection = map_features(database, metric="speed", timestamp=summary["time_values"][0], limit=100)
        flood_collection = map_features(database, metric="speed", timestamp=summary["time_values"][-1], limit=100)
        self.assertEqual({wildfire_collection["total_count"], flood_collection["total_count"]}, {3, 4})
        persistence = persistent_degradation(
            database,
            metric="speed",
            threshold=0.1,
            top_k=7,
            bbox=None,
            time_start=None,
            time_end=None,
        )
        self.assertTrue(persistence["verification"]["passed"])
        self.assertGreater(len(persistence["items"]), 0)

    def test_mixed_quality_rows_are_flagged_without_poisoning_map_extent(self) -> None:
        database = self.root / "quality.sqlite"
        summary = import_fixture(
            database,
            [FIXTURES / "mixed_quality_timezone.csv"],
            project_id="mixed_quality",
            hazard_type="storm",
            units={"speed": "km/h", "travel_time": "min"},
        )
        self.assertEqual(summary["entity_count"], 2)
        self.assertEqual(summary["mapped_entity_count"], 1)
        self.assertEqual(summary["unmapped_entity_count"], 1)
        self.assertEqual(summary["observation_count"], 3)
        self.assertEqual(summary["quality"]["invalid_coordinates"], 1)
        self.assertEqual(summary["quality"]["missing_entity_ids"], 1)
        self.assertEqual(summary["quality"]["duplicate_entity_times"], 1)
        self.assertEqual(summary["quality"]["nonpositive_speeds"], 1)
        self.assertEqual(summary["quality"]["nonpositive_travel_times"], 1)
        self.assertTrue(summary["time_start"].endswith("Z"))
        collection = map_features(database, metric="speed", timestamp=None, limit=100)
        self.assertEqual(collection["total_count"], 1)
        longitude, latitude = collection["features"][0]["geometry"]["coordinates"]
        self.assertTrue(-180 <= longitude <= 180)
        self.assertTrue(-90 <= latitude <= 90)

    def test_static_primary_metric_remains_descriptive(self) -> None:
        database = self.root / "static.sqlite"
        summary = import_fixture(
            database,
            [FIXTURES / "earthquake_static_snapshot.csv"],
            project_id="static_earthquake",
            hazard_type="earthquake",
            units={"primary_metric": "g"},
        )
        self.assertEqual(summary["mapped_entity_count"], 3)
        self.assertTrue(summary["metrics"]["primary"]["available"])
        self.assertEqual(summary["capabilities"]["persistent_degradation"], "unavailable")
        self.assertEqual(summary["capabilities"]["historical_reconstruction"], "conditional")
        collection = map_features(database, metric="primary", timestamp=None, limit=100)
        self.assertEqual(collection["total_count"], 3)


class AssistantGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = {
            "project_id": "p5_fixture",
            "name": "Assistant fixture",
            "hazard_type": "flood",
            "summary": {
                "quality": {"invalid_coordinates": 2},
                "capabilities": {
                    "topology_criticality": "available",
                    "restoration_screening": "conditional",
                },
            },
        }

    def test_local_assistant_prepares_but_does_not_run_critical_analysis(self) -> None:
        result = local_assistant_response(
            message="帮我找关键路段",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        self.assertEqual(result["provider"], "local")
        self.assertIn("preview_analysis", {action["type"] for action in result["actions"]})
        self.assertNotIn("run_analysis", {action["type"] for action in result["actions"]})

    def test_local_assistant_preserves_explicit_top_100(self) -> None:
        result = local_assistant_response(
            message="帮我找前100条关键路段",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        action = next(action for action in result["actions"] if action["type"] == "preview_analysis")
        self.assertIn("100", action["value"])
        self.assertIn("100", action["label"])

    def test_local_assistant_requires_real_selection_for_closure(self) -> None:
        result = local_assistant_response(
            message="关闭这些道路看看",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        self.assertIn("地图", result["message"])
        self.assertIn("set_selection_mode", {action["type"] for action in result["actions"]})
        self.assertNotIn("preview_analysis", {action["type"] for action in result["actions"]})

    def test_restoration_order_is_understood_as_analysis_not_a_live_order(self) -> None:
        result = local_assistant_response(
            message="Screen a restoration order for the selected failed links",
            project=self.project,
            selected_keys=["region::link-1", "region::link-2"],
            bbox=None,
            current_metric="speed",
        )
        self.assertFalse(result["safety_precheck"])
        self.assertIn("preview_analysis", {action["type"] for action in result["actions"]})
        live = local_assistant_response(
            message="Order the crews to close these roads now",
            project=self.project,
            selected_keys=["region::link-1"],
            bbox=None,
            current_metric="speed",
        )
        self.assertTrue(live["safety_precheck"])
        self.assertNotIn("preview_analysis", {action["type"] for action in live["actions"]})

    def test_assistant_has_deterministic_offline_path(self) -> None:
        context = AssistantRequestContext(
            message="How do I use this platform?",
            project=None,
            registry={"models": []},
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            current_time=None,
            current_view="explore",
            history=[],
        )
        result = assistant_response(context=context, prefer_llm=False)
        self.assertEqual(result["provider"], "local")
        self.assertIn("start_tour", {action["type"] for action in result["actions"]})

        with_project = local_assistant_response(
            message="How do I use this platform?",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        self.assertIn("six decision tools", with_project["message"])
        self.assertIn("start_tour", {action["type"] for action in with_project["actions"]})

    def test_assistant_opens_scenario_builder_and_library_without_claiming_completion(self) -> None:
        builder = local_assistant_response(
            message="打开中断情景构建器",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            scenario_count=3,
        )
        self.assertIn("open_scenario_builder", {action["type"] for action in builder["actions"]})
        self.assertNotIn("已保存", builder["message"])
        library = local_assistant_response(
            message="show my saved scenarios",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            active_scenario_name="Flood A",
            scenario_count=3,
        )
        self.assertIn("open_scenario_library", {action["type"] for action in library["actions"]})
        self.assertIn("3", library["message"])

    def test_assistant_explains_critical_area_tiers_without_overclaiming_dpc(self) -> None:
        result = local_assistant_response(
            message="What is Dynamic Propagation Connectivity?",
            project=self.project,
            selected_keys=[],
            bbox=None,
            current_metric="speed",
        )
        self.assertIn("preprint-derived finite Regional Dynamic Propagation Connectivity", result["message"])
        self.assertIn("only sufficient routable topology", result["message"].lower())
        self.assertIn("observed mean-degradation grid", result["message"])
        self.assertIn("models", {action["target"] for action in result["actions"]})


if __name__ == "__main__":
    unittest.main()
