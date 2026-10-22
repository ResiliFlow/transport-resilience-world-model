from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.app import main
from backend.app.api.schemas import AgentToolTaskRequest
from backend.app.core.storage import Repository


class AgentToolApiTests(unittest.TestCase):
    def test_data_inspection_runs_through_the_uniform_task_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project_data = root / "project.sqlite"
            project_data.touch()
            repository = Repository(root / "metadata.sqlite")
            repository.initialise()
            repository.save_project(
                project_id="project_agent_api",
                name="Agent API fixture",
                hazard_type="flood",
                source="test",
                status="ready",
                database_path=str(project_data),
                mapping={},
                summary={"observation_count": 12, "mapped_entity_count": 3},
            )

            def project_or_404(project_id: str):
                project = repository.get_project(project_id)
                if project is None:
                    raise AssertionError("Fixture project missing")
                return project

            with patch.object(main, "repository", repository), patch.object(
                main, "_project_or_404", side_effect=project_or_404
            ):
                response = main.create_agent_tool_task(
                    AgentToolTaskRequest(
                        project_id="project_agent_api",
                        tool_id="inspect_project_data",
                        input={"include_memory": False, "include_model_readiness": False},
                        client_request_id="agent-api-request-001",
                    )
                )
                deadline = time.time() + 5
                task = repository.get_agent_task(response["task_id"])
                while task["status"] not in {"completed", "failed", "cancelled"}:
                    if time.time() >= deadline:
                        self.fail("Agent data-inspection task did not finish")
                    time.sleep(0.02)
                    task = repository.get_agent_task(response["task_id"])

            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["progress"], 100.0)
            self.assertEqual(task["output"]["kind"], "project_profile")
            self.assertFalse(task["output"]["memory_saved"])
            self.assertEqual(task["output"]["result"]["project_id"], "project_agent_api")
            self.assertEqual(task["output"]["result"]["summary"]["observation_count"], 12)


if __name__ == "__main__":
    unittest.main()
