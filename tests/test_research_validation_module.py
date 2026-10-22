from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine import research_validation


class ResearchValidationModuleTests(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "report_id": "rv_unit_test",
            "project_id": "project_unit_test",
            "project_name": "Unit project",
            "generated_at": "2026-07-27T00:00:00+00:00",
            "status": "evidence_available",
            "profile": "standard",
            "protocol": research_validation.PROTOCOL,
            "summary": {"strong_stages": 1, "limited_stages": 0},
            "reproducibility": {"report_sha256": "abc123"},
            "stages": [
                {
                    "label": "Perception",
                    "grade": "strong",
                    "claim": "The imported evidence is traceable.",
                    "method": "Source hashing and validation.",
                    "interpretation": "The source was read without mutation.",
                    "boundary": "This is not causal attribution.",
                    "metrics": [{"label": "Coverage", "value": 1.0}],
                }
            ],
        }

    def test_report_store_is_isolated_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(research_validation, "REPORT_ROOT", Path(folder)):
                path = research_validation.save_research_report(self._report())
                self.assertEqual(path.parent.name, "project_unit_test")
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8"))["protocol"],
                    research_validation.PROTOCOL,
                )
                listed = research_validation.list_research_reports("project_unit_test")
                self.assertEqual([item["report_id"] for item in listed], ["rv_unit_test"])
                loaded = research_validation.load_research_report("rv_unit_test")
                self.assertEqual(loaded["reproducibility"]["report_sha256"], "abc123")

    def test_exports_preserve_claim_boundary_and_reproduction_identity(self) -> None:
        report = self._report()
        csv_text = research_validation.report_csv(report)
        latex_text = research_validation.report_latex(report)
        self.assertIn("This is not causal attribution.", csv_text)
        self.assertIn("Transport World Model research-validation evidence", latex_text)
        self.assertIn("abc123", latex_text)

    def test_only_the_registered_standard_protocol_is_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "one registered standard protocol"):
            research_validation.run_research_validation(
                {"database_path": "missing.sqlite", "summary": {}},
                software_version="test",
                model_registry_version="test",
                profile="complete",
            )

    def test_reviewer_trace_uses_the_canonical_world_model_order(self) -> None:
        project = {
            "summary": {
                "entity_count": 2,
                "mapped_entity_count": 2,
                "observation_count": 4,
                "time_value_count": 2,
                "geometry_kind": "representative_point",
                "geometry_note": "Representative link points.",
                "quality": {},
                "capabilities": {
                    "topology_criticality": "available",
                    "spatial_hotspots": "available",
                    "restoration_screening": "conditional",
                    "route_detour_analysis": "available",
                    "progressive_robustness": "available",
                    "closure_scenario": "available",
                },
                "source_files": [{"sha256": "abc"}],
            }
        }
        checks = [
            {"id": name, "passed": True, "value": 0.0, "threshold": 0.0}
            for name in (
                "stress-independent-oracle", "route-independent-oracle",
                "critical-vs-baselines", "recovery-vs-baselines",
                "repeatability", "top-k-sensitivity",
            )
        ]
        stages = research_validation._world_model_stage_records(
            project,
            {"checks": checks, "summary": {"network_edges": 2, "passed": 6, "total": 6}},
            {"available": True},
            {"can_run": True, "tier": "exploratory_probe", "od_rows": 0, "topology_links": 2},
            {"executable": 2, "highest_fidelity_selected": 1, "models": []},
            {"agent_task_count": 0, "completed_agent_task_count": 0, "saved_run_count": 0},
            {"case_count": 3, "boundary": "Technical portability only."},
        )
        self.assertEqual(
            [stage["label"] for stage in stages],
            ["Perception", "Prediction", "Planning", "Verification", "Simulation", "Decision", "Feedback", "Memory"],
        )
        self.assertEqual([stage["index"] for stage in stages], list(range(1, 9)))
        self.assertIn("OD", stages[0]["boundary"])
        self.assertEqual(stages[4]["grade"], "limited")


if __name__ == "__main__":
    unittest.main()
