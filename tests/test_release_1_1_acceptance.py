from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from backend.app.core.storage import Repository
from engine.importers import import_csv_project, preview_csv
from engine.llm_providers import provider_status
from engine.models import area_summary, map_features
from engine.natural_language import extract_requested_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Release11GeneratedDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="twm_release_11_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_flow_only_alternative_schema_is_ready_mapped_and_descriptive(self) -> None:
        source = self.root / "renamed_flow_only.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["Asset Code", "Centre Latitude", "Centre Longitude", "Traffic Volume"],
            )
            writer.writeheader()
            writer.writerow({"Asset Code": "A", "Centre Latitude": -33.90, "Centre Longitude": 151.20, "Traffic Volume": 100})
            writer.writerow({"Asset Code": "B", "Centre Latitude": -33.91, "Centre Longitude": 151.22, "Traffic Volume": 250})

        preview = preview_csv(source)
        self.assertEqual(preview["mapping_health"]["status"], "ready")
        self.assertEqual(preview["suggested_mapping"]["roles"]["flow"], "Traffic Volume")

        database = self.root / "flow_only.sqlite"
        summary = import_csv_project(
            files=[source],
            mapping=preview["suggested_mapping"]["roles"],
            units={"speed": "km/h", "travel_time": "min", "primary_metric": "vehicles/hour", "flow": "vehicles/hour"},
            timezone="UTC",
            database_path=database,
            project_id="flow_only",
            project_name="Renamed flow-only data",
            hazard_type="other",
            source="generated_acceptance",
        )
        self.assertTrue(summary["metrics"]["flow"]["available"])
        collection = map_features(database, metric="flow", limit=100)
        self.assertEqual(len(collection["features"]), 2)
        self.assertTrue(all(feature["properties"]["severity"] is None for feature in collection["features"]))
        self.assertEqual(
            sorted(round(float(feature["properties"]["display_score"]), 6) for feature in collection["features"]),
            [0.0, 1.0],
        )
        area = area_summary(database, metric="flow", bbox=summary["extent"], timestamp=0)
        self.assertEqual(sum(item["count"] for item in area["items"]), 2)
        self.assertAlmostEqual(sum(item["share"] for item in area["items"]), 1.0)
        self.assertEqual(area["items"][0]["area_bucket"], "No severity definition")

    def test_project_memory_is_traceable_exportable_and_excludes_credentials(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        summary = {
            "observation_count": 2,
            "mapped_entity_count": 2,
            "time_start": "static",
            "time_end": "static",
            "metrics": {"flow": {"available": True, "unit": "vehicles/hour"}},
            "mapping": {"by_file": {"source.csv": {"flow": "Traffic Volume"}}},
            "source_files": [{"name": "source.csv", "sha256": "a" * 64, "encoding": "utf-8", "delimiter": ","}],
        }
        repository.save_project(
            project_id="memory_project",
            name="Memory project",
            hazard_type="flood",
            source="generated_acceptance",
            status="ready",
            database_path=str(self.root / "project.sqlite"),
            mapping=summary["mapping"],
            summary=summary,
        )
        run_id = repository.save_run(
            "memory_project",
            "critical_components",
            {"top_k": 20, "metric": "flow"},
            {
                "model": {"id": "P5-M2", "name": "Critical links"},
                "status": "conditional",
                "conclusion": "Twenty links were screened.",
                "items": [{"rank": 1}],
                "verification": {"passed": True},
                "data_boundary": "Topology screen only.",
            },
        )
        scenario_id = repository.save_scenario(
            project_id="memory_project",
            name="Flood closure",
            hazard_type="flood",
            base_timestamp=0,
            changes=[{"key": "r::A", "effect": "closed", "magnitude": 1}],
            result={"status": "conditional", "indirect_impact_count": 2, "verification": {"passed": True}},
        )

        memory = repository.project_memory("memory_project")
        self.assertIsNotNone(memory)
        self.assertEqual(memory["overview"], {"source_file_count": 1, "analysis_count": 1, "scenario_count": 1, "event_count": 3})
        rendered = json.dumps(memory)
        self.assertNotIn("sk-test-secret", rendered)
        self.assertEqual(memory["assistant_grounding"]["api_key_persistence"], "Provider API keys are never stored in project memory or project files.")
        self.assertEqual(memory["assistant_grounding"]["chat_persistence"], "Chat text is session-only and is not added to project memory.")
        self.assertIn("TWM-METHOD-2", rendered)
        self.assertTrue(repository.delete_run(run_id))
        self.assertTrue(repository.delete_scenario(scenario_id))
        self.assertEqual(repository.project_memory("memory_project")["overview"]["event_count"], 1)


class Release11InterfaceAndLanguageTests(unittest.TestCase):
    def test_hotspots_and_links_preserve_arbitrary_requested_counts(self) -> None:
        cases = {
            "Find the 20 geographic degradation hotspots.": 20,
            "Identify the 1,000 most critical links.": 1000,
            "return twenty five critical areas": 25,
            "show 38 instead and use travel time instead": 38,
            "找出1000条关键边": 1000,
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_requested_count(prompt), expected)

    def test_provider_catalog_exposes_multiple_models_without_custom_ids(self) -> None:
        status = provider_status(None)
        self.assertEqual({item["id"] for item in status["providers"]}, {"openai", "anthropic", "deepseek", "gemini"})
        self.assertTrue(all(len(item["models"]) >= 2 for item in status["providers"]))
        self.assertTrue(all(item["custom_model_allowed"] is False for item in status["providers"]))
        catalog = {item["id"]: {model["id"] for model in item["models"]} for item in status["providers"]}
        self.assertEqual(catalog["openai"], {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"})
        self.assertIn("claude-sonnet-5", catalog["anthropic"])
        self.assertEqual(catalog["deepseek"], {"deepseek-v4-flash", "deepseek-v4-pro"})
        self.assertIn("gemini-3.5-flash", catalog["gemini"])

    def test_brand_upload_footer_two_assistants_and_memory_are_visible(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('<span class="platform-title-line">Transport <b>World Model</b> Platform</span>', page)
        self.assertIn('<span class="platform-purpose-line">for Infrastructure Resilience, Response and Recovery</span>', page)
        self.assertNotIn("Prototype 5", page)
        self.assertNotIn("Confirm this file independently", page)
        self.assertIn('class="confirm-fields-scroll"', page)
        self.assertIn('id="confirm-upload"', page)
        self.assertIn('id="assistant-launch"', page)
        self.assertIn('id="llm-copilot-launch"', page)
        self.assertEqual(page.count("data-assistant-mode="), 2)
        self.assertIn('data-view="memory"', page)
        self.assertIn("primaryIsStorageAlias", app)
        self.assertIn("preview.files.map((file, fileIndex)", app)
        self.assertIn('open && state.assistantSurface === "platform"', app)


if __name__ == "__main__":
    unittest.main()
