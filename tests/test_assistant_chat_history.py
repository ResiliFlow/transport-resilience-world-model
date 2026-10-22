from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.app import main
from backend.app.api.schemas import (
    AssistantChatCreateRequest,
    AssistantChatMessageCreateRequest,
)
from backend.app.core.storage import Repository


class AssistantChatStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repository = Repository(Path(self.temp.name) / "metadata.sqlite")
        self.repository.initialise()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_chat_can_start_without_a_project_and_persist_messages(self) -> None:
        chat = self.repository.create_assistant_chat(title=None, project_id=None)
        self.repository.append_assistant_chat_message(
            chat["chat_id"], role="user", content="Find the top 20 critical roads."
        )
        self.repository.append_assistant_chat_message(
            chat["chat_id"],
            role="assistant",
            content="I will check the project and select a supported tier.",
            metadata={"kind": "answer"},
        )

        reopened = self.repository.get_assistant_chat(chat["chat_id"])
        self.assertEqual(reopened["project_id"], None)
        self.assertEqual(reopened["title"], "Find the top 20 critical roads.")
        self.assertEqual([item["role"] for item in reopened["messages"]], ["user", "assistant"])
        self.assertEqual(reopened["messages"][1]["metadata"]["kind"], "answer")
        self.assertEqual(self.repository.list_assistant_chats()[0]["message_count"], 2)

    def test_deleting_a_temporary_project_preserves_chat_and_clears_association(self) -> None:
        project_id = "project_chat_fixture"
        self.repository.save_project(
            project_id=project_id,
            name="Temporary project",
            hazard_type="flood",
            source="test",
            status="ready",
            database_path=str(Path(self.temp.name) / "project.sqlite"),
            mapping={},
            summary={},
            retention="temporary",
        )
        chat = self.repository.create_assistant_chat(title="Flood analysis", project_id=project_id)
        deleted, _ = self.repository.delete_project(project_id)

        self.assertTrue(deleted)
        self.assertIsNone(self.repository.get_assistant_chat(chat["chat_id"])["project_id"])


class AssistantChatApiTests(unittest.TestCase):
    def test_chat_endpoints_accept_project_optional_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repository = Repository(Path(folder) / "metadata.sqlite")
            repository.initialise()
            with patch.object(main, "repository", repository):
                chat = main.create_assistant_chat(AssistantChatCreateRequest())
                message = main.append_assistant_chat_message(
                    chat["chat_id"],
                    AssistantChatMessageCreateRequest(
                        role="user", content="Summarise the current upload."
                    ),
                )
                reopened = main.get_assistant_chat(chat["chat_id"])
                recent = main.list_assistant_chats(limit=10)

            self.assertEqual(message["chat_id"], chat["chat_id"])
            self.assertEqual(reopened["messages"][0]["content"], "Summarise the current upload.")
            self.assertEqual(recent[0]["chat_id"], chat["chat_id"])


class AssistantChatInterfaceContractTests(unittest.TestCase):
    def test_llm_chat_controls_and_adaptive_task_events_are_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
        script = (root / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        for element_id in (
            "assistant-new-chat",
            "assistant-chat-toggle",
            "assistant-chat-drawer",
            "assistant-chat-list",
            "assistant-api-gate",
            "assistant-api-gate-connect",
        ):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("assistant-process-stream", script)
        self.assertIn("Waiting for your input", script)
        self.assertIn("handleLlmBuilderAction", script)
        self.assertIn("Connect an LLM API first", script)
        self.assertIn('state.assistantSurface === "llm" && !llmInferenceReady()', script)
        self.assertNotIn('className = "assistant-activity"', script)
        self.assertIn("Array.isArray(chats) ? chats : []", script)

    def test_platform_assistant_and_llm_copilot_keep_separate_transcripts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn("platformAssistantHistory: []", script)
        self.assertIn("platformAssistantActions: []", script)
        self.assertIn("assistantSurfaceViews:", script)
        self.assertIn("function currentAssistantHistory()", script)
        self.assertIn("function storeAssistantSurfaceView(surface)", script)
        self.assertIn("function restoreAssistantSurfaceView(surface)", script)
        self.assertIn("storeAssistantSurfaceView(state.assistantSurface)", script)
        self.assertIn("restoreAssistantSurfaceView(nextSurface)", script)
        self.assertIn("const activeHistory = assistantHistoryForSurface(requestSurface)", script)

    def test_async_requests_results_and_actions_are_bound_to_their_owner_surface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn("platformAssistantMessageRequestId: 0", script)
        self.assertIn("platformAssistantActivity: null", script)
        self.assertIn("platformAssistantBusy: false", script)
        self.assertIn("const requestSurface = normaliseAssistantSurface(state.assistantSurface)", script)
        self.assertIn("renderAssistantResponse(result, requestSurface)", script)
        self.assertIn('const ownerSurface = "llm";', script)
        self.assertIn('appendAssistantNode(card, "llm")', script)
        self.assertIn('setAssistantActionsForSurface("llm", [])', script)
        self.assertIn('replaceAssistantSurfaceHtml("llm", "actions"', script)
        self.assertNotIn('if (state.assistantSurface !== "llm") return;\n  const container = $("#assistant-actions")', script)

    def test_platform_requests_do_not_supersede_or_persist_llm_work(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn("const requestId = isLlmRequest ? ++state.assistantMessageRequestId : ++state.platformAssistantMessageRequestId", script)
        self.assertIn('if (isLlmRequest) await persistAssistantChatMessage("user", text', script)
        self.assertIn('if (isLlmRequest) {\n    state.commandRequestId += 1;', script)
        self.assertIn('appendAssistantMessage("assistant", result.message, result.provider_notice || "", ownerSurface)', script)
        self.assertIn('assistantHistoryForSurface(ownerSurface).push({ role: "assistant", content: result.message })', script)


if __name__ == "__main__":
    unittest.main()
