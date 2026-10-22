from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.app.agent_tools import (
    AGENT_TOOL_PROTOCOL_VERSION,
    agent_tool_catalog,
    build_analysis_payload,
    build_matsim_request,
    build_scenario_request,
    validate_agent_tool_input,
)
from backend.app.core.storage import Repository


class AgentToolContractTests(unittest.TestCase):
    def test_catalogue_exposes_six_core_tools_plus_data_and_matsim(self) -> None:
        catalogue = agent_tool_catalog()
        self.assertEqual(catalogue["protocol_version"], AGENT_TOOL_PROTOCOL_VERSION)
        self.assertEqual(catalogue["core_tool_count"], 6)
        self.assertEqual(catalogue["tool_count"], 8)
        tool_ids = {item["tool_id"] for item in catalogue["tools"]}
        self.assertEqual(
            tool_ids,
            {
                "inspect_project_data",
                "find_critical_roads",
                "find_critical_areas",
                "plan_recovery_priorities",
                "plan_routes_under_disruption",
                "test_network_resilience",
                "build_disruption_scenario",
                "run_matsim",
            },
        )
        for item in catalogue["tools"]:
            self.assertIn("input_schema", item)
            self.assertIn("task_contract", item)
            self.assertIn("cancellation", item["task_contract"])

    def test_analysis_adapter_forces_the_registered_core_analysis_type(self) -> None:
        normalised = validate_agent_tool_input(
            "find_critical_areas", {"top_k": 23, "metric": "travel_time"}
        )
        payload = build_analysis_payload(
            project_id="project_abc",
            tool_id="find_critical_areas",
            normalised_input=normalised,
        )
        self.assertEqual(payload["analysis_type"], "spatial_hotspots")
        self.assertEqual(payload["top_k"], 23)
        self.assertEqual(payload["metric"], "travel_time")

    def test_tool_specific_validation_rejects_missing_scenario_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Agent tool input is invalid"):
            validate_agent_tool_input(
                "build_disruption_scenario", {"name": "Flood disruption", "changes": []}
            )

    def test_scenario_and_matsim_adapters_preserve_existing_schemas(self) -> None:
        scenario_input = validate_agent_tool_input(
            "build_disruption_scenario",
            {
                "name": "Road closure",
                "hazard_type": "flood",
                "changes": [{"key": "region::link-1", "effect": "closed"}],
            },
        )
        scenario = build_scenario_request(
            project_id="project_abc", normalised_input=scenario_input
        )
        self.assertEqual(scenario.project_id, "project_abc")
        self.assertEqual(scenario.changes[0].magnitude, 1.0)

        matsim_input = validate_agent_tool_input(
            "run_matsim", {"scenario_id": "scenario_abc", "max_agents": 200}
        )
        scenario_id, matsim = build_matsim_request(matsim_input)
        self.assertEqual(scenario_id, "scenario_abc")
        self.assertEqual(matsim.max_agents, 200)


class AgentTaskStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Repository(Path(self.temp.name) / "metadata.sqlite")
        self.repository.initialise()
        for project_id in ("project_one", "project_two"):
            self.repository.save_project(
                project_id=project_id,
                name=project_id,
                hazard_type="other",
                source="test",
                status="ready",
                database_path=str(Path(self.temp.name) / f"{project_id}.sqlite"),
                mapping={},
                summary={},
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _task(self, *, client_request_id: str | None = None) -> dict:
        task, created = self.repository.create_agent_task(
            tool_id="find_critical_roads",
            project_id="project_one",
            tool_input={"top_k": 10},
            dependencies=[],
            client_request_id=client_request_id,
        )
        self.assertTrue(created)
        return task

    def test_client_request_id_is_idempotent_but_cannot_be_reused_for_other_input(self) -> None:
        first = self._task(client_request_id="request-12345")
        second, created = self.repository.create_agent_task(
            tool_id="find_critical_roads",
            project_id="project_one",
            tool_input={"top_k": 10},
            dependencies=[],
            client_request_id="request-12345",
        )
        self.assertFalse(created)
        self.assertEqual(first["task_id"], second["task_id"])
        with self.assertRaisesRegex(ValueError, "different Agent task"):
            self.repository.create_agent_task(
                tool_id="find_critical_roads",
                project_id="project_one",
                tool_input={"top_k": 20},
                dependencies=[],
                client_request_id="request-12345",
            )

    def test_cancelled_analysis_cannot_save_a_run_or_complete_later(self) -> None:
        task = self._task()
        self.assertEqual(self.repository.cancel_agent_task(task["task_id"]), "cancelled")
        completed = self.repository.complete_agent_analysis_task(
            task["task_id"],
            project_id="project_one",
            analysis_type="critical_components",
            request={"top_k": 10},
            result={"verification": {"passed": True}},
        )
        self.assertFalse(completed)
        self.assertEqual(self.repository.list_runs("project_one"), [])
        saved = self.repository.get_agent_task(task["task_id"])
        self.assertEqual(saved["status"], "cancelled")
        self.assertIsNone(saved["output"])

    def test_analysis_completion_is_atomic_and_project_isolated(self) -> None:
        task = self._task()
        wrong_project = self.repository.complete_agent_analysis_task(
            task["task_id"],
            project_id="project_two",
            analysis_type="critical_components",
            request={"top_k": 10},
            result={"verification": {"passed": True}},
        )
        self.assertFalse(wrong_project)
        self.assertEqual(self.repository.list_runs("project_two"), [])

        completed = self.repository.complete_agent_analysis_task(
            task["task_id"],
            project_id="project_one",
            analysis_type="critical_components",
            request={"top_k": 10},
            result={"verification": {"passed": True}},
        )
        self.assertTrue(completed)
        saved = self.repository.get_agent_task(task["task_id"])
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["output"]["kind"], "analysis")
        self.assertTrue(saved["output"]["memory_saved"])
        self.assertEqual(len(self.repository.list_runs("project_one")), 1)

    def test_scenario_and_matsim_outputs_stay_with_the_same_project(self) -> None:
        scenario_task, _ = self.repository.create_agent_task(
            tool_id="build_disruption_scenario",
            project_id="project_one",
            tool_input={},
            dependencies=[],
        )
        self.assertTrue(
            self.repository.complete_agent_scenario_task(
                scenario_task["task_id"],
                project_id="project_one",
                name="Closure",
                hazard_type="flood",
                base_timestamp=1,
                changes=[{"key": "r::1", "effect": "closed", "magnitude": 1.0}],
                result={"verification": {"passed": True}},
            )
        )
        scenario_id = self.repository.get_agent_task(scenario_task["task_id"])["output"][
            "artifact_refs"
        ][0]["id"]

        matsim_task, _ = self.repository.create_agent_task(
            tool_id="run_matsim",
            project_id="project_one",
            tool_input={"scenario_id": scenario_id},
            dependencies=[{"kind": "scenario", "id": scenario_id, "role": "scenario"}],
        )
        self.assertTrue(
            self.repository.complete_agent_matsim_task(
                matsim_task["task_id"],
                project_id="project_one",
                scenario_id=scenario_id,
                result={"verification": {"passed": True}, "agents": 50},
            )
        )
        scenario = self.repository.get_scenario(scenario_id)
        self.assertEqual(scenario["result"]["matsim"]["agents"], 50)
        self.assertEqual(
            self.repository.get_agent_task(matsim_task["task_id"])["output"]["kind"],
            "matsim",
        )


if __name__ == "__main__":
    unittest.main()
