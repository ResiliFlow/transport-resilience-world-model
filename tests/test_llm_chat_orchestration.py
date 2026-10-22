from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from backend.app.api.schemas import AssistantMessageRequest
from backend.app.core.storage import Repository
from backend.app.main import _verified_assistant_chat_artifacts
from engine.assistant import (
    AssistantRequestContext,
    _ground_core_analysis_action,
    _safe_chat_artifacts,
)


PROJECT = {"project_id": "project_chat_scope", "name": "Chat scope", "summary": {}}


def context(
    *,
    selected_keys: list[str] | None = None,
    artifacts: list[dict] | None = None,
    message: str = "Run the requested core tool.",
):
    return AssistantRequestContext(
        message=message,
        project=PROJECT,
        registry={"models": []},
        selected_keys=selected_keys or [],
        bbox=None,
        current_metric="speed",
        current_time=1,
        current_view="explore",
        history=[],
        chat_artifacts=artifacts or [],
    )


def provider_result(tool: str, count: int = 10) -> dict:
    return {
        "message": "The requested core analysis is ready.",
        "actions": [],
        "analysis_request": {
            "tool": tool,
            "requested_count": count,
            "metric": "current",
            "scope": "current",
            "time_mode": "current",
        },
        "understanding": {
            "intent": "calculation_request",
            "requires_computation": True,
            "needs_clarification": False,
        },
    }


class SixCoreCopilotRoutingTests(unittest.TestCase):
    def test_explanation_and_evidence_prompts_never_open_core_builders(self) -> None:
        prompts = [
            "The filename proves a named hurricane and zero speed proves closure. Confirm these claims and explain observed demand.",
            "Summarise the project limitations, including disruption and closure evidence.",
            "Assess the six core tools and do not start any analysis.",
            "Explain why a closure scenario does not prove confirmed road closures.",
            "Can the disruption scenario model run on this upload?",
            "What does Plan routes under disruption do?",
            "How can I find critical roads in this platform?",
            "Can this platform find the top 20 critical roads?",
            "How does the platform find critical roads?",
            "Can I run MATSim with this upload?",
            "这个平台能否识别关键道路？",
            "如何使用路径中断功能？",
            "请解释中断情景是否等于真实道路关闭。",
        ]
        for prompt_text in prompts:
            with self.subTest(prompt=prompt_text):
                misleading_provider_result = provider_result("disruption_scenario")
                grounded = _ground_core_analysis_action(
                    context(selected_keys=["road::1"], message=prompt_text),
                    misleading_provider_result,
                )
                self.assertEqual(grounded["actions"], [])
                self.assertFalse(grounded["understanding"]["requires_computation"])
                self.assertEqual(grounded["analysis_request"]["tool"], "none")

    def test_explanation_veto_does_not_block_real_core_work(self) -> None:
        prompts = {
            "critical_roads": "Do not infer observed demand; find the top 20 critical roads.",
            "critical_areas": "Find the top 12 critical areas and explain the claim limit in the result.",
            "recovery_priorities": "Rank 8 recovery priorities, but do not claim an optimal crew schedule.",
            "routes_under_disruption": "Plan 3 routes and explain why they are retrospective rather than live navigation.",
            "network_resilience": "Test network resilience over 25 cumulative road failures and report the boundary.",
            "disruption_scenario": "Run a disruption scenario for the selected closed roads and explain the result.",
            "critical_roads_direct_question": "Can you find the top 5 critical roads?",
            "critical_roads_chinese": "请识别前20条关键道路，并说明结果局限。",
        }
        expected = {
            "critical_roads": "preview_analysis",
            "critical_areas": "preview_analysis",
            "recovery_priorities": "open_recovery_builder",
            "routes_under_disruption": "open_route_builder",
            "network_resilience": "open_stress_builder",
            "disruption_scenario": "open_scenario_builder",
            "critical_roads_direct_question": "preview_analysis",
            "critical_roads_chinese": "preview_analysis",
        }
        for tool, prompt_text in prompts.items():
            with self.subTest(tool=tool):
                grounded = _ground_core_analysis_action(
                    context(selected_keys=["road::1", "road::2"], message=prompt_text),
                    provider_result(
                        "critical_roads" if tool in {"critical_roads_direct_question", "critical_roads_chinese"} else tool,
                        20,
                    ),
                )
                self.assertEqual(grounded["actions"][0]["type"], expected[tool])

    def test_provider_cannot_force_builder_for_answer_only_request(self) -> None:
        misleading = provider_result("routes_under_disruption", 3)
        misleading["understanding"]["requires_computation"] = True
        grounded = _ground_core_analysis_action(
            context(message="Review whether route analysis is supported; answer without running or opening anything."),
            misleading,
        )
        self.assertEqual(grounded["actions"], [])

    def test_all_six_core_tools_route_to_the_existing_execution_or_input_bridge(self) -> None:
        expected = {
            "critical_roads": "preview_analysis",
            "critical_areas": "preview_analysis",
            "recovery_priorities": "open_recovery_builder",
            "routes_under_disruption": "open_route_builder",
            "network_resilience": "open_stress_builder",
            "disruption_scenario": "open_scenario_builder",
        }
        for tool, action_type in expected.items():
            with self.subTest(tool=tool):
                result = _ground_core_analysis_action(
                    context(selected_keys=["region::a", "region::b"]),
                    provider_result(tool),
                )
                self.assertEqual(result["actions"][0]["type"], action_type)
                self.assertFalse(result["actions"][0]["requires_confirmation"])

    def test_route_and_recovery_never_bypass_their_application_input_bridge(self) -> None:
        for tool, expected in (
            ("routes_under_disruption", "open_route_builder"),
            ("recovery_priorities", "open_recovery_builder"),
        ):
            result = _ground_core_analysis_action(
                context(selected_keys=["region::a", "region::b"]),
                provider_result(tool),
            )
            self.assertEqual(result["actions"][0]["type"], expected)

    def test_plain_recovery_request_starts_a_new_source_choice_instead_of_replaying_stale_result(self) -> None:
        stale_provider_result = provider_result("none", 10)
        stale_provider_result["message"] = "The most recent recovery screening returned 10 priorities."
        stale_provider_result["understanding"].update({
            "intent": "existing_result",
            "requires_computation": False,
        })
        grounded = _ground_core_analysis_action(
            context(message="I want to know the recovery priorities"),
            stale_provider_result,
        )
        self.assertEqual(grounded["actions"][0]["type"], "open_recovery_builder")
        self.assertNotIn("top 10", grounded["actions"][0]["value"].lower())
        self.assertIn("choose whether to use critical roads, critical areas", grounded["message"].lower())

    def test_missing_route_endpoints_open_the_map_builder_instead_of_returning_text_only(self) -> None:
        route_context = AssistantRequestContext(
            message="Plan 3 routes between my selected points under a 50% speed reduction.",
            project=PROJECT,
            registry={"models": []},
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            current_time=1,
            current_view="explore",
            history=[],
        )
        result = provider_result("routes_under_disruption", 3)
        result["understanding"]["needs_clarification"] = True
        grounded = _ground_core_analysis_action(route_context, result)
        self.assertEqual(grounded["actions"][0]["type"], "open_route_builder")
        self.assertIn("50% speed reduction", grounded["actions"][0]["value"])

    def test_route_input_follow_up_recovers_the_same_chat_request(self) -> None:
        follow_up = AssistantRequestContext(
            message="Select an origin and destination on the map.",
            project=PROJECT,
            registry={"models": []},
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            current_time=1,
            current_view="explore",
            history=[
                {"role": "user", "content": "Plan 3 routes between my selected points under a 50% speed reduction."},
                {"role": "assistant", "content": "Please select the route endpoints."},
            ],
        )
        result = provider_result("none", 10)
        result["understanding"].update({
            "intent": "clarification_needed",
            "requires_computation": False,
            "needs_clarification": True,
        })
        grounded = _ground_core_analysis_action(follow_up, result)
        self.assertEqual(grounded["actions"][0]["type"], "open_route_builder")
        self.assertIn("Plan 3 routes between my selected points under a 50% speed reduction", grounded["actions"][0]["value"])

    def test_matsim_request_uses_its_own_interactive_builder(self) -> None:
        result = _ground_core_analysis_action(
            context(message="I want to use MATSim."),
            provider_result("matsim"),
        )
        self.assertEqual(result["actions"][0]["type"], "open_matsim_builder")
        self.assertEqual(result["actions"][0]["target"], "matsim")

    def test_matsim_explanation_does_not_start_an_interactive_run(self) -> None:
        result = provider_result("matsim")
        result["understanding"].update({
            "intent": "project_overview",
            "requires_computation": False,
        })
        grounded = _ground_core_analysis_action(
            context(message="What is MATSim?"),
            result,
        )
        self.assertEqual(grounded["actions"], [])

    def test_explicit_matsim_run_opens_builder_even_if_provider_marks_it_as_explanation(self) -> None:
        result = provider_result("none")
        result["understanding"].update({
            "intent": "project_overview",
            "requires_computation": False,
        })
        grounded = _ground_core_analysis_action(
            context(message="I want to run a MATSim simulation."),
            result,
        )
        self.assertEqual(grounded["actions"][0]["type"], "open_matsim_builder")

    def test_new_explicit_tool_overrides_the_previous_tool_in_the_same_chat(self) -> None:
        switched_context = AssistantRequestContext(
            message="Now find the top 20 critical areas.",
            project=PROJECT,
            registry={"models": []},
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            current_time=1,
            current_view="explore",
            history=[
                {"role": "user", "content": "Find the top 20 critical roads."},
                {"role": "assistant", "content": "The critical-road analysis is complete."},
            ],
        )
        stale_provider_result = provider_result("critical_roads", 20)
        grounded = _ground_core_analysis_action(switched_context, stale_provider_result)
        self.assertEqual(grounded["actions"][0]["type"], "preview_analysis")
        self.assertIn("Find critical areas", grounded["actions"][0]["value"])
        self.assertNotIn("Find critical roads", grounded["actions"][0]["value"])

    def test_provider_normalised_prompt_matrix_preserves_tool_and_requested_count(self) -> None:
        prompts = {
            "critical_roads": [
                "Find the top 12 critical roads.",
                "Which 12 links matter most to network movement?",
                "Rank 12 network bottlenecks.",
            ],
            "critical_areas": [
                "Find 8 geographic degradation hotspots.",
                "Show the 8 critical areas.",
                "Which 8 mapped clusters need attention?",
            ],
            "recovery_priorities": [
                "Rank 10 recovery priorities from those roads.",
                "Which 10 damaged roads should be restored first?",
                "Prioritise reopening 10 affected areas.",
            ],
            "routes_under_disruption": [
                "Plan 3 routes under a 50% speed reduction.",
                "Give me 3 detours if roads slow by half.",
                "Recommend 3 paths around the affected roads.",
            ],
            "network_resilience": [
                "Stress-test the network over 25 cumulative failures.",
                "Test resilience for 25 critical-road removals.",
                "Show 25 stages as key roads fail.",
            ],
            "disruption_scenario": [
                "Build 6 selected-road closures.",
                "Create a 6-road slowdown scenario.",
                "Add longer travel time to 6 selected links.",
            ],
        }
        expected_actions = {
            "critical_roads": "preview_analysis",
            "critical_areas": "preview_analysis",
            "recovery_priorities": "open_recovery_builder",
            "routes_under_disruption": "open_route_builder",
            "network_resilience": "open_stress_builder",
            "disruption_scenario": "open_scenario_builder",
        }
        for tool, variants in prompts.items():
            expected_count = {"critical_roads": 12, "critical_areas": 8, "recovery_priorities": 10,
                              "routes_under_disruption": 3, "network_resilience": 25,
                              "disruption_scenario": 6}[tool]
            for prompt_text in variants:
                with self.subTest(tool=tool, prompt=prompt_text):
                    ctx = context(selected_keys=["region::a", "region::b"], message=prompt_text)
                    result = _ground_core_analysis_action(ctx, provider_result(tool, expected_count))
                    action = result["actions"][0]
                    self.assertEqual(action["type"], expected_actions[tool])
                    self.assertIn(str(expected_count), action["value"])


class ConversationArtifactIsolationTests(unittest.TestCase):
    def test_request_schema_accepts_verified_chat_artifact_references(self) -> None:
        request = AssistantMessageRequest(
            message="Rank recovery priorities from those roads.",
            project_id="project_chat_scope",
            chat_artifacts=[{
                "kind": "analysis_result",
                "artifact_id": "run_verified_123",
                "analysis_type": "critical_components",
                "project_id": "project_chat_scope",
            }],
        )
        self.assertEqual(request.chat_artifacts[0].artifact_id, "run_verified_123")

    def test_only_active_project_artifacts_reach_the_provider(self) -> None:
        artifacts = [
            {
                "kind": "analysis_result", "artifact_id": "run_current",
                "analysis_type": "critical_components", "project_id": "project_chat_scope",
            },
            {
                "kind": "analysis_result", "artifact_id": "run_other",
                "analysis_type": "spatial_hotspots", "project_id": "other_project",
            },
        ]
        safe = _safe_chat_artifacts(context(artifacts=artifacts))
        self.assertEqual([item["artifact_id"] for item in safe], ["run_current"])

    def test_new_chat_with_empty_artifact_list_has_no_implicit_dependency(self) -> None:
        self.assertEqual(_safe_chat_artifacts(context(artifacts=[])), [])

    def test_server_resolves_only_current_project_records(self) -> None:
        class FakeRepository:
            def get_run(self, run_id):
                records = {
                    "run_current": {"project_id": "project_chat_scope", "analysis_type": "critical_components", "created_at": "now"},
                    "run_other": {"project_id": "other_project", "analysis_type": "critical_components", "created_at": "now"},
                }
                return records.get(run_id)

            def get_scenario(self, scenario_id):
                records = {
                    "scenario_current": {"project_id": "project_chat_scope", "created_at": "now"},
                    "scenario_other": {"project_id": "other_project", "created_at": "now"},
                }
                return records.get(scenario_id)

        artifacts = [
            {"kind": "analysis_result", "artifact_id": "run_current", "analysis_type": "critical_components", "project_id": "project_chat_scope"},
            {"kind": "analysis_result", "artifact_id": "run_other", "analysis_type": "critical_components", "project_id": "project_chat_scope"},
            {"kind": "analysis_result", "artifact_id": "run_current", "analysis_type": "spatial_hotspots", "project_id": "project_chat_scope"},
            {"kind": "scenario_result", "artifact_id": "scenario_current", "analysis_type": "closure_scenario", "project_id": "project_chat_scope"},
            {"kind": "scenario_result", "artifact_id": "scenario_other", "analysis_type": "closure_scenario", "project_id": "project_chat_scope"},
        ]
        with patch("backend.app.main.repository", FakeRepository()):
            verified = _verified_assistant_chat_artifacts("project_chat_scope", artifacts)
        self.assertEqual(
            [(item["kind"], item["artifact_id"]) for item in verified],
            [("analysis_result", "run_current"), ("scenario_result", "scenario_current")],
        )

    def test_chat_artifact_metadata_survives_close_and_reopen(self) -> None:
        with TemporaryDirectory() as directory:
            repository = Repository(Path(directory) / "test.sqlite")
            repository.initialise()
            repository.save_project(
                project_id="project_chat_scope",
                name="Chat scope",
                hazard_type="other",
                source="test",
                status="ready",
                database_path=str(Path(directory) / "project.sqlite"),
                mapping={},
                summary={},
            )
            chat = repository.create_assistant_chat(
                title="Critical roads then recovery",
                project_id="project_chat_scope",
            )
            repository.append_assistant_chat_message(
                chat["chat_id"],
                role="assistant",
                content="Analysis complete",
                metadata={
                    "kind": "analysis_result",
                    "run_id": "run_persisted_1",
                    "analysis_type": "critical_components",
                    "project_id": "project_chat_scope",
                },
            )
            reopened = repository.get_assistant_chat(chat["chat_id"])
        self.assertEqual(reopened["project_id"], "project_chat_scope")
        self.assertEqual(reopened["messages"][0]["metadata"]["run_id"], "run_persisted_1")


class FrontendConversationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            Path(__file__).resolve().parents[1] / "frontend" / "js" / "app.js"
        ).read_text(encoding="utf-8")

    def test_new_chat_clears_artifacts_and_loading_rebuilds_them_from_message_metadata(self) -> None:
        self.assertIn("state.assistantChatArtifacts = [];", self.script)
        self.assertIn("assistantArtifactFromMetadata(item.metadata || {}, item.created_at)", self.script)
        self.assertIn("chat_artifacts: assistantArtifactsForRequest()", self.script)

    def test_dependency_reuse_is_scoped_to_the_active_chat_and_project(self) -> None:
        self.assertIn("latestAssistantArtifact(acceptedTypes)", self.script)
        self.assertIn("pending.chatId === state.assistantChatId", self.script)
        self.assertIn("pending.projectId === currentProjectId()", self.script)
        self.assertIn("assistant_chat_scope_locked", self.script)

    def test_project_switch_cannot_continue_the_previous_project_chat(self) -> None:
        self.assertIn("assistantChatProjectId", self.script)
        self.assertIn("state.assistantChatProjectId !== currentProjectId()", self.script)
        self.assertIn("Conversation belongs to another project", self.script)
        self.assertIn("(chat.project_id || null) === (projectId || null)", self.script)

    def test_map_input_workflows_return_results_to_the_originating_chat(self) -> None:
        self.assertIn('completeAssistantPendingTask("route", result)', self.script)
        self.assertIn('completeAssistantPendingTask("recovery", result)', self.script)
        self.assertIn("appendAssistantScenarioResult(saved)", self.script)
        self.assertIn('setAssistantOpen(true, "llm")', self.script)
        self.assertIn("handleLlmBuilderAction(builderAction, builderAction.value || text, requestId)", self.script)

    def test_map_inputs_are_captured_one_step_at_a_time_and_return_to_copilot(self) -> None:
        for step in (
            "route_origin",
            "route_destination",
            "route_road",
            "route_area",
            "recovery_road",
            "recovery_area",
            "scenario_road",
            "scenario_area",
            "matsim_road",
            "matsim_area",
            "analysis_point",
            "analysis_area",
        ):
            with self.subTest(step=step):
                self.assertIn(step, self.script)
        self.assertIn("function beginAssistantMapInput(step)", self.script)
        self.assertIn("function returnAssistantMapInput(step, confirmation)", self.script)
        self.assertIn("showAssistantPendingTaskStep();", self.script)
        self.assertIn('persistAssistantChatMessage("assistant", confirmation', self.script)

    def test_llm_not_platform_owns_the_final_run_actions(self) -> None:
        expected = {
            "run_route": "runRouteComparison()",
            "run_recovery": "runRecoveryPriorities({ fromAssistant: true })",
            "run_scenario": "saveScenarioDraft()",
            "run_stress": "runStressTest()",
            "run_analysis": "runAssistantAnalysis(null)",
            "run_matsim": "runAssistantMatsimTask(task)",
        }
        for action, invocation in expected.items():
            with self.subTest(action=action):
                self.assertIn(f'actionId === "{action}"', self.script)
                self.assertIn(invocation, self.script)

    def test_matsim_waits_for_the_explicit_copilot_run_button(self) -> None:
        self.assertIn('actions: [{ id: "run_matsim", label: "Run MATSim", primary: true }]', self.script)
        self.assertIn('Model execution is never auto-started after the final application input.', self.script)
        self.assertNotIn('task.type === "matsim" ? "run_matsim"', self.script)

    def test_every_interactive_ready_state_has_a_persistent_copilot_run_action(self) -> None:
        expected = {
            "route": ("run_route", "Run route analysis"),
            "recovery": ("run_recovery", "Run recovery ranking"),
            "scenario": ("run_scenario", "Run disruption scenario"),
            "matsim": ("run_matsim", "Run MATSim"),
            "analysis": ("run_analysis", "Run analysis"),
            "stress": ("run_stress", "Run resilience test"),
        }
        for task_type, (action_id, label) in expected.items():
            with self.subTest(task_type=task_type):
                self.assertIn(f'{task_type}: ["{action_id}", "{label}"]', self.script)
        self.assertIn('data-llm-pending-run', self.script)
        self.assertIn('syncAssistantPendingRunAction(task);', self.script)

    def test_compound_workflow_waits_for_explicit_model_execution(self) -> None:
        function_body = self.script.split("function continueAssistantCompoundTaskIfReady", 1)[1].split("function failAssistantCompoundWorkflow", 1)[0]
        self.assertNotIn("executeAssistantWorkflowAction", function_body)
        self.assertIn("return false;", function_body)

    def test_new_chat_abandons_only_the_old_copilot_input_builder(self) -> None:
        self.assertIn("function abandonAssistantPendingInputs()", self.script)
        self.assertIn("abandonAssistantPendingInputs();", self.script)
        self.assertIn('pending.type === "route" && state.routeBuilding', self.script)
        self.assertIn('["scenario", "matsim"].includes(pending.type) && state.scenarioBuilding', self.script)

    def test_interactive_parameters_are_not_silently_defaulted(self) -> None:
        for phase in ('task.phase === "count"', 'task.phase === "magnitude"', 'task.phase === "agent_count"'):
            with self.subTest(phase=phase):
                self.assertIn(phase, self.script)
        self.assertIn('title: "What disruption should the routes avoid?"', self.script)
        self.assertIn('title: "Which disruption should MATSim simulate?"', self.script)
        self.assertIn('id: "matsim_use_active"', self.script)
        self.assertIn('id: "matsim_new_scenario"', self.script)
        self.assertIn('requestedMagnitudeFromText(sourceText, null)', self.script)
        self.assertIn("analysisRequestNeedsCount(previewAction.value)", self.script)
        self.assertIn('id: "analysis_count_20"', self.script)
        self.assertNotIn("task.magnitude || 0.5", self.script)

    def test_llm_disruption_percentages_use_synchronised_sliders(self) -> None:
        self.assertIn("function magnitudeWorkflowRange(", self.script)
        self.assertIn('data-llm-magnitude-slider=', self.script)
        self.assertIn("data-llm-magnitude-output", self.script)
        self.assertIn("data-llm-magnitude-confirm", self.script)
        self.assertIn("task.magnitude = percentage / 100;", self.script)
        self.assertIn("platformSlider.value = String(percentage);", self.script)
        for workflow in ("route", "scenario", "matsim"):
            with self.subTest(workflow=workflow):
                self.assertIn(
                    f'magnitudeWorkflowRange("{workflow}", task.effect, task.magnitude)',
                    self.script,
                )
        self.assertNotIn("function magnitudeWorkflowActions(", self.script)

    def test_llm_quantity_cards_accept_custom_whole_numbers(self) -> None:
        self.assertIn("countInput = null", self.script)
        self.assertIn("data-llm-count-input=", self.script)
        self.assertIn("data-llm-count-submit=", self.script)
        self.assertIn("Number.isInteger(value)", self.script)
        self.assertIn("executeAssistantWorkflowAction(`${actionPrefix}_${value}`)", self.script)
        self.assertIn('input.dataset.llmCountInput', self.script)
        for action_prefix in ("analysis_count", "recovery_count", "route_count", "stress_count"):
            with self.subTest(action_prefix=action_prefix):
                self.assertIn(f'actionPrefix: "{action_prefix}"', self.script)
        self.assertIn('label: "Custom route count"', self.script)
        self.assertIn("max: 50", self.script)
        self.assertIn('placeholder: "For example, 100"', self.script)

    def test_exact_scenario_road_count_returns_to_chat_after_each_selection(self) -> None:
        self.assertIn("state.scenarioDraft.size < task.requestedCount", self.script)
        self.assertIn("Select roads one at a time", self.script)
        self.assertIn("Select road ${Math.min(state.scenarioDraft.size + 1, task.requestedCount)}", self.script)

    def test_generic_recovery_request_does_not_silently_reuse_a_prior_result(self) -> None:
        self.assertIn("function recoveryRequestReferencesPriorResult(text)", self.script)
        self.assertIn("requestedCount && recoveryRequestReferencesPriorResult(sourceText)", self.script)

    def test_recovery_source_choices_and_matsim_result_return_to_the_same_chat(self) -> None:
        for action in (
            "recovery_use_critical_roads",
            "recovery_use_critical_areas",
            "recovery_use_active_scenario",
        ):
            with self.subTest(action=action):
                self.assertIn(action, self.script)
        self.assertIn("function appendAssistantMatsimResult(result, scenario)", self.script)
        self.assertIn('pendingAssistantTaskMatches("matsim")', self.script)
        self.assertIn('setAssistantOpen(true, "llm")', self.script)

    def test_matsim_scenario_save_continues_to_the_correct_agent_mode_in_copilot(self) -> None:
        self.assertIn('sourceWorkflow?.type === "matsim"', self.script)
        self.assertIn('pendingMatsimTask.phase = nextAssistantMatsimAgentPhase(pendingMatsimTask);', self.script)
        self.assertIn('saveScenarioDraft({ assistantMatsimExecution: true })', self.script)
        self.assertIn('"Return to Copilot and choose 50, 100, 150 or 200 agents."', self.script)

    def test_copilot_uses_uploaded_od_demand_without_a_fixed_probe_choice(self) -> None:
        self.assertIn("function assistantMatsimUsesUploadedDemand(", self.script)
        self.assertIn('readiness?.tier === "demand_calibrated"', self.script)
        self.assertIn("Number(readiness?.od_rows || 0) > 0", self.script)
        self.assertIn("const ASSISTANT_MATSIM_UPLOADED_DEMAND_LIMIT = 4000;", self.script)
        self.assertIn('return assistantMatsimUsesUploadedDemand() || task?.agentCount ? "ready" : "agent_count";', self.script)
        self.assertIn("Automatically generated from uploaded OD demand", self.script)
        self.assertIn("Uploaded OD demand will determine the simulated agents automatically.", self.script)
        self.assertIn('if (!uploadedDemand) $("#matsim-probe-agent-count").value = String(task.agentCount);', self.script)
        self.assertIn("maxAgentsOverride: uploadedDemand ? ASSISTANT_MATSIM_UPLOADED_DEMAND_LIMIT : null", self.script)

    def test_matsim_result_discloses_area_roads_excluded_from_the_routable_supply(self) -> None:
        self.assertIn("result.manifest?.selected_change_count", self.script)
        self.assertIn("result.manifest?.excluded_change_count", self.script)
        self.assertIn("selected roads were simulated", self.script)
        self.assertIn("Roads simulated", self.script)

    def test_matsim_request_refreshes_and_recovers_derivable_link_topology(self) -> None:
        self.assertIn("async function prepareMatsimReadinessForCopilot()", self.script)
        self.assertIn("await refreshMatsimReadiness();", self.script)
        self.assertIn("state.networkMapping?.available && !state.networkMapping?.ready", self.script)
        self.assertIn("state.networkMapping = await api.buildNetworkMapping(state.project.project_id);", self.script)
        self.assertIn('if (taskType === "matsim") {', self.script)
        self.assertIn("await prepareMatsimReadinessForCopilot();", self.script)

    def test_copilot_matsim_run_focuses_the_real_platform_progress(self) -> None:
        self.assertIn("function focusMatsimRunProgress()", self.script)
        self.assertIn('card.scrollIntoView({ behavior: "smooth", block: "center" });', self.script)
        self.assertIn('progress.classList.add("is-run-focus")', self.script)
        self.assertIn('progressPanel.classList.remove("is-run-focus")', self.script)
        self.assertIn("focusProgress: true,", self.script)
        self.assertIn("runActiveScenarioMatsim({", self.script)

    def test_independent_task_switch_does_not_get_consumed_as_old_workflow_input(self) -> None:
        self.assertIn("function explicitAssistantWorkflow(text)", self.script)
        self.assertIn("const switchesPendingWorkflow = Boolean(", self.script)
        self.assertIn("!switchesPendingWorkflow && await consumeAssistantWorkflowText(text)", self.script)
        self.assertIn("function prepareAssistantWorkflowTransition(action, sourceText", self.script)

    def test_independent_task_resets_only_visible_workspace_state(self) -> None:
        self.assertIn("function resetAssistantVisualWorkspaceForIndependentTask()", self.script)
        self.assertIn("resetAssistantVisualWorkspaceForIndependentTask();", self.script)
        self.assertIn("state.map?.setAnalysisFeatures(null)", self.script)
        self.assertIn("state.map?.setScenarioFeatures(null)", self.script)
        self.assertNotIn("deleteAssistantChat", self.script[
            self.script.index("function resetAssistantVisualWorkspaceForIndependentTask()"):
            self.script.index("function prepareAssistantWorkflowTransition")
        ])

    def test_map_input_return_reopens_and_scrolls_the_same_copilot_chat(self) -> None:
        block = self.script[
            self.script.index("function returnAssistantMapInput"):
            self.script.index("async function executeAssistantWorkflowAction")
        ]
        self.assertIn('setAssistantOpen(true, "llm")', block)
        self.assertIn("messages.scrollTop = messages.scrollHeight", block)

    def test_llm_queue_is_chat_scoped_and_does_not_cancel_the_active_turn(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="assistant-task-queue"', page)
        self.assertIn("assistantTaskQueues: new Map()", self.script)
        self.assertIn("function assistantQueueKey(", self.script)
        self.assertIn("function enqueueAssistantTask(", self.script)
        self.assertIn("async function drainAssistantTaskQueue()", self.script)
        self.assertIn("data-delete-assistant-queue-id", self.script)
        queue_block = self.script[
            self.script.index("function enqueueAssistantTask"):
            self.script.index("async function processAssistantMessage")
        ]
        self.assertNotIn("cancelJob", queue_block)
        self.assertNotIn("assistantCompoundWorkflow = null", queue_block)

    def test_compound_workflow_uses_existing_tools_and_waits_for_application_inputs(self) -> None:
        self.assertIn("function assistantCompoundPlanFromText(", self.script)
        self.assertIn("function runNextAssistantCompoundStep()", self.script)
        for step_type in ("recovery", "scenario", "matsim", "verify"):
            with self.subTest(step_type=step_type):
                self.assertIn(f'type: "{step_type}"', self.script)
        self.assertIn("await runAssistantAnalysis(null)", self.script)
        self.assertIn("workflow.sourceResult = criticalResult", self.script)
        self.assertIn("return workflow?.sourceResult || completedResultFor", self.script)
        self.assertIn("await runRecoveryPriorities({ fromAssistant: true })", self.script)
        self.assertIn("await saveScenarioDraft()", self.script)
        self.assertIn("await runAssistantMatsimTask(state.assistantPendingTask)", self.script)
        self.assertIn('phase: !step.effect ? "effect"', self.script)
        self.assertIn('phase: uploadedDemand || step.agentCount ? "ready" : "agent_count"', self.script)

    def test_compound_workflow_never_invents_missing_result_counts(self) -> None:
        planner = self.script[
            self.script.index("function assistantCompoundPlanFromText"):
            self.script.index("function renderAssistantCompoundPlan")
        ]
        self.assertIn("requestedCountFromText(request, null)", planner)
        self.assertIn('"Find critical roads - count needed"', planner)
        self.assertIn('"Rank recovery priorities - count needed"', planner)
        self.assertNotIn('sourceType === "spatial_hotspots" ? 10 : 20', planner)
        runner = self.script[
            self.script.index("async function runNextAssistantCompoundStep"):
            self.script.index("async function beginAssistantCompoundWorkflow")
        ]
        self.assertIn('compoundInput: "critical_count"', runner)
        self.assertIn('compoundInput: "recovery_count"', runner)
        self.assertIn('"Choose the result count"', runner)
        self.assertIn('"Choose the recovery result count"', runner)
        assistant_source = (Path(__file__).resolve().parents[1] / "engine" / "assistant.py").read_text(encoding="utf-8")
        self.assertIn("Do not invent an omitted critical-result count, recovery-priority count", assistant_source)
        self.assertIn("Never describe a default Top-N as selected", assistant_source)

    def test_llm_and_platform_share_one_execution_per_workflow_step(self) -> None:
        runner = self.script[
            self.script.index("async function runNextAssistantCompoundStep"):
            self.script.index("async function beginAssistantCompoundWorkflow")
        ]
        self.assertIn("if (workflow.runningIndex === workflow.index) return", runner)
        self.assertIn("workflow.runningIndex = workflow.index", runner)
        self.assertEqual(runner.count("await runAssistantAnalysis(null)"), 1)
        self.assertEqual(runner.count("await runRecoveryPriorities({ fromAssistant: true })"), 1)
        self.assertEqual(runner.count("await saveScenarioDraft()"), 1)
        self.assertEqual(runner.count("await runAssistantMatsimTask(state.assistantPendingTask)"), 1)
        self.assertIn("async function runAssistantTaskOnce", self.script)
        self.assertIn("if (!task || task.runStarted) return null", self.script)

    def test_math_receipts_use_native_mathml_not_plain_text_formula_substitutes(self) -> None:
        self.assertIn("const ASSISTANT_MODEL_RECEIPTS = {", self.script)
        self.assertGreaterEqual(self.script.count('<math display="block"'), 7)
        self.assertGreaterEqual(self.script.count('<math display="inline"'), 15)
        for token in ("<mfrac>", "<msub>", "<msubsup>", "<munder>", "<munderover>", "<mo>∑</mo>", "<mo>∈</mo>"):
            with self.subTest(token=token):
                self.assertIn(token, self.script)
        self.assertIn("assistant-math-expression", self.script)
        self.assertIn("${receipt.symbols}", self.script)
        self.assertNotIn("escapeHtml(receipt.symbols)", self.script)
        self.assertNotIn("<mi>arg min</mi>", self.script)
        self.assertNotIn("<mi>arg max</mi>", self.script)

    def test_compound_workflow_blocks_queue_drain_until_the_linked_chain_finishes(self) -> None:
        finally_block = self.script[
            self.script.index("async function processAssistantMessage"):
            self.script.index("function highlightElement")
        ]
        self.assertIn("!state.assistantCompoundWorkflow", finally_block)
        self.assertIn('advanceAssistantCompoundWorkflow("scenario")', self.script)
        self.assertIn('advanceAssistantCompoundWorkflow("matsim")', self.script)


if __name__ == "__main__":
    unittest.main()
