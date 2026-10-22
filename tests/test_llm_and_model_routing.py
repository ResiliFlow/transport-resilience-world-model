from __future__ import annotations

import asyncio
from io import BytesIO
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from backend.app.api.schemas import AssistantMessageRequest, AssistantProviderConfigRequest
from backend.app.main import configure_assistant_provider, provider_request_error_handler
from engine.assistant import (
    AssistantRequestContext,
    _conversation_context,
    _ground_core_analysis_action,
    _provider_response,
    _safe_project_context,
    _sanitise_model_result,
    assistant_response,
)
from engine.command_parser import interpret_command
from engine.llm_providers import (
    ProviderCredential,
    ProviderRequestError,
    _get_json,
    _post_json,
    call_provider_json,
    configure_provider,
    provider_status,
    refresh_provider_models,
    validate_provider_key,
    verify_provider_inference,
)
from engine.model_review import deterministic_model_checks, is_model_design_request
from engine.model_router import model_recommendation


class ProviderAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_id = "p5_test_session_1234567890"
        for provider in ("openai", "anthropic", "deepseek", "gemini"):
            configure_provider(
                session_id=self.session_id,
                provider=provider,
                api_key=None,
                clear=True,
            )

    def tearDown(self) -> None:
        for provider in ("openai", "anthropic", "deepseek", "gemini"):
            configure_provider(
                session_id=self.session_id,
                provider=provider,
                api_key=None,
                clear=True,
            )

    def test_session_key_is_never_returned_by_status(self) -> None:
        secret = "unit-test-key-1"
        status = configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key=secret,
            model="gpt-5.6-terra",
        )
        rendered = json.dumps(status)
        self.assertNotIn(secret, rendered)
        self.assertEqual(status["configured_provider_count"], 1)
        self.assertEqual(status["key_persistence"], "process_memory_only")
        openai = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertEqual(openai["connection"], "verification_pending")
        configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key=None,
            clear=True,
        )
        self.assertFalse(provider_status(self.session_id)["api_configured"])

    def test_connected_provider_can_switch_model_without_reentering_key(self) -> None:
        configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key="unit-test-key-2",
            model="first-model",
        )
        status = configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key=None,
            model="second-model",
        )
        provider = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertTrue(provider["configured"])
        self.assertEqual(provider["model"], "second-model")

    def test_all_four_official_api_shapes_are_adapted_to_json(self) -> None:
        responses = {
            "openai": {"output": [{"content": [{"text": '{"ok": true}'}]}]},
            "anthropic": {"content": [{"type": "text", "text": '{"ok": true}'}]},
            "deepseek": {"choices": [{"message": {"content": '{"ok": true}'}}]},
            "gemini": {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]},
        }
        expected_hosts = {
            "openai": "api.openai.com/v1/responses",
            "anthropic": "api.anthropic.com/v1/messages",
            "deepseek": "api.deepseek.com/chat/completions",
            "gemini": "generativelanguage.googleapis.com/v1beta/models/",
        }
        for provider, response in responses.items():
            with self.subTest(provider=provider), patch(
                "engine.llm_providers._post_json", return_value=response
            ) as post:
                result = call_provider_json(
                    provider=provider,
                    credential=ProviderCredential("provider-test-key", f"{provider}-model", 0.0),
                    system="Return grounded JSON.",
                    user_payload={"request": "test"},
                    schema={
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                    },
                )
                self.assertEqual(result, {"ok": True})
                self.assertIn(expected_hosts[provider], post.call_args.args[0])

    def test_connection_verification_selects_a_working_fallback_model(self) -> None:
        configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key="sk-test-auto-selection",
            model="gpt-5.6-terra",
        )
        outcomes = [
            ProviderRequestError(
                "The selected model is not available to this API account.",
                code="model_not_available",
                http_status=404,
            ),
            {"ok": True},
        ]
        with patch("engine.llm_providers.call_provider_json", side_effect=outcomes) as call:
            status = verify_provider_inference(
                session_id=self.session_id, provider="openai", max_model_attempts=3
            )
        self.assertTrue(status["verification"]["success"])
        self.assertEqual(status["verification"]["attempt_count"], 2)
        self.assertEqual(status["verification"]["selected_model"], "gpt-5.6-luna")
        self.assertEqual(call.call_count, 2)
        openai = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertEqual(openai["connection"], "inference_verified")
        self.assertEqual(openai["model"], "gpt-5.6-luna")

    def test_connection_verification_does_not_cycle_models_for_billing_failure(self) -> None:
        configure_provider(
            session_id=self.session_id,
            provider="openai",
            api_key="sk-test-no-credit",
            model="gpt-5.6-terra",
        )
        error = ProviderRequestError(
            "The API account has no usable credit or its spending limit has been reached.",
            code="billing_or_quota",
            http_status=429,
        )
        with patch("engine.llm_providers.call_provider_json", side_effect=error) as call:
            status = verify_provider_inference(session_id=self.session_id, provider="openai")
        self.assertFalse(status["verification"]["success"])
        self.assertEqual(status["verification"]["attempt_count"], 1)
        self.assertEqual(call.call_count, 1)
        openai = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertEqual(openai["connection"], "inference_failed")
        self.assertEqual(openai["last_error_code"], "billing_or_quota")

    def test_openai_key_without_model_catalog_permission_reaches_real_inference_gate(self) -> None:
        catalog_permission_error = ProviderRequestError(
            "This API key cannot read the provider model catalogue.",
            code="model_catalog_permission_denied",
            http_status=403,
        )
        with patch(
            "engine.llm_providers._provider_model_entries",
            side_effect=catalog_permission_error,
        ):
            validate_provider_key(provider="openai", api_key="sk-test-responses-only")
            configure_provider(
                session_id=self.session_id,
                provider="openai",
                api_key="sk-test-responses-only",
                model="gpt-5.6-terra",
            )
            status = refresh_provider_models(session_id=self.session_id, provider="openai")
        openai = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertEqual(openai["catalog_status"], "unavailable")
        self.assertIn("cannot read", openai["catalog_error"])

        with patch("engine.llm_providers.call_provider_json", return_value={"ok": True}):
            verified = verify_provider_inference(session_id=self.session_id, provider="openai")
        self.assertTrue(verified["verification"]["success"])
        self.assertEqual(verified["verification"]["selected_model"], "gpt-5.6-terra")

    def test_atomic_connection_uses_real_inference_when_model_catalogue_is_down(self) -> None:
        catalogue_outage = ProviderRequestError(
            "The provider model list was unavailable after bounded retries.",
            code="network_or_timeout",
        )
        request = AssistantProviderConfigRequest(
            session_id=self.session_id,
            provider="openai",
            api_key="unit-test-key-3",
            model="gpt-5.6-terra",
        )
        with patch(
            "engine.llm_providers._provider_model_entries",
            side_effect=catalogue_outage,
        ), patch(
            "engine.llm_providers.call_provider_json",
            return_value={"ok": True},
        ) as inference:
            status = configure_assistant_provider(request)
        self.assertTrue(status["verification"]["success"])
        self.assertEqual(inference.call_count, 1)
        openai = next(item for item in status["providers"] if item["id"] == "openai")
        self.assertEqual(openai["connection"], "inference_verified")
        self.assertEqual(openai["catalog_status"], "unavailable")

    def test_provider_post_retries_one_503_then_succeeds(self) -> None:
        unavailable = HTTPError(
            "https://api.openai.com/v1/responses",
            503,
            "temporarily unavailable",
            hdrs=None,
            fp=BytesIO(b'{"error":{"type":"server_error"}}'),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        response.__exit__.return_value = False
        with patch(
            "engine.llm_providers.urlopen",
            side_effect=[unavailable, response],
        ) as opened, patch("engine.llm_providers.time.sleep") as pause:
            result = _post_json(
                "https://api.openai.com/v1/responses",
                {"model": "gpt-5.6-terra", "input": "connection test"},
                {"Authorization": "Bearer sk-not-logged"},
                5.0,
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(opened.call_count, 2)
        pause.assert_called_once()

    def test_invalid_openai_key_keeps_provider_401_instead_of_local_422(self) -> None:
        error = ProviderRequestError(
            "GPT connection check failed. The provider rejected this API key.",
            code="authentication_invalid",
            http_status=401,
        )
        response = asyncio.run(provider_request_error_handler(None, error))
        body = json.loads(response.body)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(body["code"], "authentication_invalid")
        self.assertEqual(body["type"], "provider_request_error")
        self.assertNotIn("sk-invalid-test-key", response.body.decode("utf-8"))

    def test_model_catalog_http_errors_keep_safe_auth_and_permission_codes(self) -> None:
        cases = (
            (401, "authentication_invalid"),
            (403, "model_catalog_permission_denied"),
            (429, "rate_limited"),
        )
        for http_status, expected_code in cases:
            with self.subTest(http_status=http_status), patch(
                "engine.llm_providers.urlopen",
                side_effect=HTTPError(
                    "https://api.openai.com/v1/models",
                    http_status,
                    "provider error",
                    hdrs=None,
                    fp=BytesIO(b'{"error":{"message":"must not be exposed"}}'),
                ),
            ):
                with self.assertRaises(ProviderRequestError) as raised:
                    _get_json("https://api.openai.com/v1/models", {"Authorization": "Bearer secret"})
            self.assertEqual(raised.exception.code, expected_code)
            self.assertEqual(raised.exception.http_status, http_status)
            self.assertNotIn("must not be exposed", str(raised.exception))


class ConditionalModelReviewTests(unittest.TestCase):
    def test_review_trigger_is_explicit_not_normal_analysis(self) -> None:
        self.assertFalse(is_model_design_request("Identify the 20 most critical links."))
        self.assertFalse(is_model_design_request("Run a flood disruption scenario."))
        self.assertTrue(is_model_design_request("Build a new mathematical transport model."))
        self.assertTrue(is_model_design_request("创建一个优化模型"))

    def test_deterministic_gate_checks_units_domains_tests_and_sources(self) -> None:
        proposal = {
            "name": "Auditable test model",
            "decision_question": "Which link to restore?",
            "scope": "Static network",
            "sets": [{"name": "A", "description": "links"}],
            "parameters": [{"name": "c", "indices": ["a"], "description": "cost", "unit": "AUD"}],
            "variables": [{"name": "x", "indices": ["a"], "domain": "binary", "unit": "dimensionless", "description": "restore"}],
            "objective": {"sense": "max", "expression": "sum b_a x_a", "units": "service", "description": "benefit"},
            "constraints": [{"name": "budget", "indices": [], "expression": "sum c_a x_a <= B", "purpose": "budget", "left_units": "AUD", "right_units": "AUD"}],
            "assumptions": ["Static network"],
            "required_data": [{"field": "repair_cost", "unit": "AUD", "role": "cost", "required": True}],
            "solution_method": {"method": "MILP", "termination": "proven gap", "complexity_note": "NP-hard"},
            "verification_tests": [
                {"type": kind, "description": kind, "pass_condition": "declared"}
                for kind in ("unit", "invariant", "numerical", "counterexample")
            ],
            "claim_boundary": ["Not live control"],
            "source_links": [{"citation": "Nurre et al. (2012)", "doi": "10.1016/j.ejor.2012.07.010", "url": "", "used_for": "restoration"}],
        }
        checks = deterministic_model_checks(proposal)
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["passed_count"], checks["check_count"])

    def test_normal_llm_request_does_not_invoke_multi_model_review(self) -> None:
        session_id = "p5_review_session_123456789"
        configure_provider(
            session_id=session_id,
            provider="openai",
            api_key="sk-test-review-routing",
            model="gpt-test",
        )
        context = AssistantRequestContext(
            message="Identify critical links.",
            project=None,
            registry={"models": []},
            selected_keys=[],
            bbox=None,
            current_metric="speed",
            current_time=None,
            current_view="explore",
            history=[],
        )
        provider_result = {
            "message": "Prepared.", "actions": [], "suggestions": [], "grounded_in": [],
            "language": "en", "provider": "openai", "model": "gpt-test", "fallback": False,
        }
        try:
            with patch("engine.assistant._provider_response", return_value=provider_result), patch(
                "engine.assistant.build_and_review_model"
            ) as review:
                result = assistant_response(
                    context=context,
                    session_id=session_id,
                    provider="openai",
                    interaction_mode="assist",
                )
                self.assertEqual(result["message"], "Prepared.")
                review.assert_not_called()
        finally:
            configure_provider(session_id=session_id, provider="openai", api_key=None, clear=True)

    def test_project_question_phase_does_not_invoke_model_creation(self) -> None:
        session_id = "p5_project_qa_session_123456"
        configure_provider(
            session_id=session_id,
            provider="openai",
            api_key="unit-test-key-0",
            model="gpt-test",
        )
        context = AssistantRequestContext(
            message="Build a new mathematical model for this upload.",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        provider_result = {
            "message": "That request is outside this project-data question phase.",
            "actions": [], "suggestions": [], "grounded_in": [], "language": "en",
            "provider": "openai", "model": "gpt-test", "fallback": False,
        }
        try:
            with patch("engine.assistant._provider_response", return_value=provider_result), patch(
                "engine.assistant.build_and_review_model"
            ) as review:
                result = assistant_response(context=context, session_id=session_id, provider="openai")
                self.assertEqual(result["message"], provider_result["message"])
                review.assert_not_called()
        finally:
            configure_provider(session_id=session_id, provider="openai", api_key=None, clear=True)


class ProjectQuestionContextTests(unittest.TestCase):
    def test_connected_llm_receives_normal_negation_instead_of_local_safety_block(self) -> None:
        session_id = "p5_negation_routing_123456789"
        configure_provider(
            session_id=session_id, provider="openai", api_key="sk-negation-test", model="gpt-test"
        )
        context = AssistantRequestContext(
            message="Do not infer causality; explain what the data does not prove.",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        provider_result = {
            "message": "The upload does not establish causality.", "actions": [],
            "suggestions": [], "grounded_in": ["project profile"], "language": "en",
            "provider": "openai", "model": "gpt-test", "fallback": False,
        }
        try:
            with patch(
                "engine.assistant.local_assistant_response",
                return_value={"message": "Blocked locally.", "safety_precheck": True},
            ), patch("engine.assistant._provider_response", return_value=provider_result) as provider_call:
                result = assistant_response(
                    context=context, session_id=session_id, provider="openai", prefer_llm=True
                )
            self.assertEqual(result["message"], provider_result["message"])
            provider_call.assert_called_once()
        finally:
            configure_provider(session_id=session_id, provider="openai", api_key=None, clear=True)

    def test_local_assistant_keeps_its_operation_safety_response(self) -> None:
        context = AssistantRequestContext(
            message="Stop the current operation.", project=None, registry={"models": []},
            selected_keys=[], bbox=None, current_metric="speed", current_time=None,
            current_view="explore", history=[],
        )
        local_result = {"message": "Stopped.", "safety_precheck": True}
        with patch("engine.assistant.local_assistant_response", return_value=local_result) as local_call, patch(
            "engine.assistant._provider_response"
        ) as provider_call:
            result = assistant_response(context=context, prefer_llm=False)
        self.assertEqual(result["message"], "Stopped.")
        provider_call.assert_not_called()

    def test_llm_copilot_requires_a_connected_provider_and_never_uses_local_fallback(self) -> None:
        context = AssistantRequestContext(
            message="Explain this upload.", project=None, registry={"models": []},
            selected_keys=[], bbox=None, current_metric="speed", current_time=None,
            current_view="explore", history=[],
        )
        local_result = {"message": "Local guide answer", "actions": [{"type": "navigate"}]}
        with patch("engine.assistant.local_assistant_response", return_value=local_result) as local_call, patch(
            "engine.assistant._provider_response"
        ) as provider_call:
            result = assistant_response(
                context=context,
                prefer_llm=True,
                session_id="p5_no_provider_session_123456",
            )
        self.assertTrue(result["requires_api"])
        self.assertEqual(result["actions"], [])
        self.assertNotEqual(result["message"], local_result["message"])
        local_call.assert_not_called()
        provider_call.assert_not_called()

    def test_long_message_and_long_history_item_pass_request_validation(self) -> None:
        long_text = "uploaded project detail " * 400
        request = AssistantMessageRequest(
            message=long_text,
            history=[{"role": "assistant", "content": long_text}],
        )
        self.assertEqual(request.message, long_text)
        self.assertEqual(request.history[0].content, long_text)

    def test_long_conversation_keeps_recent_turns_and_compacts_older_turns(self) -> None:
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"turn-{index} " + ("project evidence " * 100),
            }
            for index in range(80)
        ]
        memory = _conversation_context(history)
        self.assertEqual(memory["total_messages"], 80)
        self.assertGreater(memory["compacted_message_count"], 0)
        self.assertTrue(memory["earlier_conversation_digest"])
        self.assertTrue(memory["recent_conversation"][-1]["content"].startswith("turn-79"))

    def test_long_provider_answer_is_not_character_truncated(self) -> None:
        long_answer = "A" * 9000
        result = _sanitise_model_result({"message": long_answer})
        self.assertEqual(result["message"], long_answer)

    def test_unicode_evidence_symbols_are_preserved_exactly(self) -> None:
        answer = "• Direct evidence; 60 × length; project hazard is “other”."
        result = _sanitise_model_result({"message": answer})
        self.assertEqual(result["message"], answer)

    def test_safe_context_distinguishes_full_proxy_and_blocked_model_tiers(self) -> None:
        project = {
            "project_id": "p-tier", "name": "Tier test", "mapping": {},
            "summary": {
                "mapped_entity_count": 12,
                "time_values": [1, 2, 3],
                "metrics": {
                    "speed": {"available": True, "unit": "km/h"},
                    "travel_time": {"available": True, "unit": "min"},
                    "length": {"available": True, "unit": "m"},
                },
                "capabilities": {"topology_criticality": "available"},
            },
        }
        readiness = _safe_project_context(project)["data_profile"]["model_readiness"]
        models = {item["model_id"]: item for item in readiness["models"]}
        self.assertEqual(models["TWM-METHOD-2"]["readiness"], "ready_with_available_data")
        self.assertEqual(models["TWM-METHOD-2"]["assumption_level"], "disclosed_proxy")
        self.assertEqual(models["TWM-METHOD-10"]["readiness"], "full_model_ready")
        self.assertEqual(models["TWM-METHOD-21"]["readiness"], "ready_with_available_data")
        self.assertIn("od_demand", models["TWM-METHOD-21"]["missing_for_highest_tier"])
        self.assertEqual(models["TWM-METHOD-18"]["readiness"], "ready_with_available_data")
        self.assertIn("availability alone never means full-fidelity", readiness["interpretation_rules"]["capability_available"])
        self.assertEqual(
            {model_id for model_id, item in models.items() if item["readiness"] == "full_model_ready"},
            {"TWM-METHOD-0", "TWM-METHOD-10", "TWM-METHOD-23"},
        )

    def test_event_dependent_models_do_not_promote_internal_reference_to_full_tier(self) -> None:
        summary = {
            "mapped_entity_count": 12,
            "time_values": [1, 2, 3],
            "baseline": {"method": "earliest_15_percent", "confirmed": False},
            "metrics": {"speed": {"available": True, "unit": "km/h"}},
            "capabilities": {"topology_criticality": "available"},
        }
        persistence = model_recommendation("P5-M1", summary)
        resilience = model_recommendation("P5-M9", summary)
        self.assertEqual(persistence["selected_tier"], "Gap-aware observed persistence")
        self.assertEqual(resilience["selected_tier"], "Observed normalised loss integral")
        self.assertEqual(persistence["status"], "conditional")
        self.assertEqual(resilience["status"], "conditional")
        self.assertEqual(
            persistence["missing_for_high_fidelity"],
            ["event_labels", "validated_baseline"],
        )
        self.assertEqual(
            resilience["missing_for_high_fidelity"],
            ["event_labels", "validated_baseline", "counterfactual_baseline"],
        )

    def test_event_dependent_models_promote_only_with_explicit_evidence(self) -> None:
        summary = {
            "mapped_entity_count": 12,
            "time_values": [1, 2, 3],
            "event_labels_available": True,
            "validated_baseline_available": True,
            "counterfactual_baseline_available": True,
            "metrics": {"speed": {"available": True, "unit": "km/h"}},
            "capabilities": {"topology_criticality": "available"},
        }
        self.assertEqual(model_recommendation("P5-M1", summary)["status"], "ready")
        self.assertEqual(model_recommendation("P5-M9", summary)["status"], "ready")

    def test_safe_context_defines_the_six_core_tool_model_mapping(self) -> None:
        project = {
            "project_id": "p-core-tools", "name": "Core tool mapping", "mapping": {},
            "summary": {
                "mapped_entity_count": 12,
                "time_values": [1, 2, 3],
                "metrics": {
                    "speed": {"available": True, "unit": "km/h"},
                    "travel_time": {"available": True, "unit": "min"},
                    "length": {"available": True, "unit": "m"},
                },
                "capabilities": {"topology_criticality": "available"},
            },
        }
        tools = _safe_project_context(project)["data_profile"]["model_readiness"]["core_tools"]
        self.assertEqual(
            [(tool["order"], tool["tool"], tool["primary_model_id"]) for tool in tools],
            [
                (1, "Find critical roads", "TWM-METHOD-2"),
                (2, "Find critical areas", "TWM-METHOD-10"),
                (3, "Plan recovery priorities", "TWM-METHOD-4"),
                (4, "Plan routes under disruption", "TWM-METHOD-23"),
                (5, "Test network resilience", "TWM-METHOD-11"),
                (6, "Build a disruption scenario", "TWM-METHOD-21"),
            ],
        )
        self.assertEqual(tools[-1]["supporting_model_ids"], ["TWM-METHOD-22"])

    def test_safe_context_contains_uploaded_profile_without_private_paths_or_hashes(self) -> None:
        project = {
            "project_id": "p1", "name": "Flood upload", "hazard_type": "flood",
            "source": "user_upload", "database_path": "C:/private/project.sqlite",
            "mapping": {"entity_id": "road_code", "speed": "observed_velocity"},
            "summary": {
                "entity_count": 12, "observation_count": 240, "time_value_count": 20,
                "time_start": "2026-01-01T00:00:00Z", "time_end": "2026-01-01T01:35:00Z",
                "extent": [150.0, -34.0, 151.0, -33.0], "geometry_kind": "line",
                "quality": {"invalid_times": 2},
                "metrics": {
                    "speed": {"available": True, "unit": "km/h"},
                    "flow": {"available": False, "unit": "vehicles/hour"},
                },
                "source_files": [{
                    "name": "roads.csv", "sha256": "do-not-send", "encoding": "utf-8",
                    "resolved_mapping": {"speed": "observed_velocity"},
                }],
            },
        }
        context = _safe_project_context(project)
        rendered = json.dumps(context)
        self.assertNotIn("project.sqlite", rendered)
        self.assertNotIn("do-not-send", rendered)
        self.assertEqual(context["data_profile"]["field_mapping"]["speed"], "observed_velocity")
        self.assertEqual(context["data_profile"]["available_metrics"]["speed"]["unit"], "km/h")
        self.assertIn("flow", context["data_profile"]["unavailable_metrics"])

    def test_provider_question_answering_never_returns_platform_actions(self) -> None:
        session_id = "p5_project_qa_provider_123456"
        configure_provider(
            session_id=session_id, provider="openai", api_key="sk-project-qa", model="gpt-test"
        )
        context = AssistantRequestContext(
            message="What does observed_velocity mean?",
            project={
                "project_id": "p1", "name": "Upload",
                "mapping": {"speed": "observed_velocity"},
                "summary": {"metrics": {"speed": {"available": True, "unit": "km/h"}}},
            },
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        raw = {
            "message": "It is the uploaded speed field.",
            "actions": [{
                "type": "navigate", "target": "models", "label": "Open models", "value": "",
                "requires_confirmation": False,
            }],
            "suggestions": ["What unit does it use?"], "grounded_in": ["field mapping"],
            "language": "en",
            "understanding": {
                "intent": "field_meaning", "subject": "observed_velocity",
                "requires_computation": False, "needs_clarification": False,
                "clarification_options": [],
            },
        }
        try:
            with patch("engine.assistant.call_provider_json", return_value=raw) as provider_call:
                result = _provider_response(
                    context, session_id=session_id, provider="openai", interaction_mode="autopilot"
                )
            self.assertEqual(result["actions"], [])
            self.assertEqual(result["interaction_mode"], "project_qa_and_core_analysis")
            payload = provider_call.call_args.kwargs["user_payload"]
            self.assertEqual(payload["task_mode"], "uploaded_project_question_answering_and_validated_core_analysis")
            self.assertNotIn("model_registry", payload)
            self.assertEqual(result["understanding"]["intent"], "field_meaning")
            status = provider_status(session_id)
            openai = next(item for item in status["providers"] if item["id"] == "openai")
            self.assertEqual(openai["connection"], "inference_verified")
        finally:
            configure_provider(session_id=session_id, provider="openai", api_key=None, clear=True)

    def test_typo_tolerant_core_request_prepares_one_confirmed_analysis(self) -> None:
        context = AssistantRequestContext(
            message="I want to find the top 20 criticle links in my data",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=1500000000,
            current_view="explore", history=[],
        )
        result = _ground_core_analysis_action(context, {
            "message": "I will prepare the top 20 critical roads.",
            "actions": [],
            "analysis_request": {
                "tool": "critical_roads", "requested_count": 20,
                "metric": "current", "scope": "whole_project", "time_mode": "current",
            },
            "understanding": {
                "intent": "calculation_request", "requires_computation": True,
                "needs_clarification": False,
            },
        })
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["type"], "preview_analysis")
        self.assertFalse(result["actions"][0]["requires_confirmation"])
        self.assertIn("top 20 results", result["actions"][0]["value"])
        self.assertIn("whole project", result["actions"][0]["value"])

    def test_core_tool_explanation_does_not_prepare_or_run_analysis(self) -> None:
        context = AssistantRequestContext(
            message="What does Find critical roads do?",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        result = _ground_core_analysis_action(context, {
            "message": "It ranks structurally important roads.",
            "actions": [{"type": "preview_analysis"}],
            "analysis_request": {
                "tool": "none", "requested_count": None,
                "metric": "current", "scope": "current", "time_mode": "current",
            },
            "understanding": {
                "intent": "data_capability", "requires_computation": False,
                "needs_clarification": False,
            },
        })
        self.assertEqual(result["actions"], [])

    def test_interactive_core_tools_open_input_builder_when_scope_is_missing(self) -> None:
        project = {"project_id": "p1", "name": "Upload", "summary": {}}
        route_context = AssistantRequestContext(
            message="Plan routes under disruption",
            project=project, registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="travel_time", current_time=None, current_view="explore", history=[],
        )
        result = _ground_core_analysis_action(route_context, {
            "message": "Choose the trip inputs.", "actions": [],
            "analysis_request": {
                "tool": "routes_under_disruption", "requested_count": 2,
                "metric": "travel_time", "scope": "current", "time_mode": "current",
            },
            "understanding": {
                "intent": "calculation_request", "requires_computation": True,
                "needs_clarification": False,
            },
        })
        self.assertEqual(result["actions"][0]["type"], "open_route_builder")

    def test_short_route_request_opens_builder_in_local_fallback(self) -> None:
        context = AssistantRequestContext(
            message="Plan 3 routes between my selected points under a 50% speed reduction.",
            project={
                "project_id": "p1", "name": "Upload",
                "summary": {"capabilities": {"topology_criticality": "available"}},
            },
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        result = assistant_response(context=context, prefer_llm=False)
        self.assertEqual(result["actions"][0]["type"], "open_route_builder")
        self.assertIn("two routable endpoints", result["message"])

    def test_follow_up_parameters_retain_previous_tool_through_structured_request(self) -> None:
        context = AssistantRequestContext(
            message="Make that 50 and use travel time instead",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=1500000000,
            current_view="explore",
            history=[
                {"role": "user", "content": "Find the top 20 critical roads"},
                {"role": "assistant", "content": "I prepared the critical-road analysis."},
            ],
        )
        result = _ground_core_analysis_action(context, {
            "message": "I will update the same critical-road request.", "actions": [],
            "analysis_request": {
                "tool": "critical_roads", "requested_count": 50,
                "metric": "travel_time", "scope": "current", "time_mode": "current",
            },
            "understanding": {
                "intent": "calculation_request", "requires_computation": True,
                "needs_clarification": False,
            },
        })
        value = result["actions"][0]["value"]
        self.assertIn("Find critical roads", value)
        self.assertIn("top 50 results", value)
        self.assertIn("travel time", value)

    def test_critical_road_contract_records_exact_model_tier_and_top_n(self) -> None:
        project = {
            "project_id": "p1", "name": "Upload",
            "summary": {
                "mapped_entity_count": 12,
                "time_values": [1, 2, 3],
                "metrics": {
                    "speed": {"available": True, "unit": "km/h"},
                    "travel_time": {"available": True, "unit": "min"},
                    "length": {"available": True, "unit": "m"},
                },
                "capabilities": {"topology_criticality": "available"},
            },
        }
        contract = interpret_command(
            text="I want to find the top 20 criticle links in my data",
            project=project, selected_keys=[], bbox=[150.0, -34.0, 151.0, -33.0],
            current_metric="speed", current_time=1500000000,
        )
        self.assertTrue(contract["ready"])
        self.assertEqual(contract["analysis_type"], "critical_components")
        self.assertEqual(contract["model_id"], "P5-M2")
        self.assertEqual(contract["top_k"], 20)
        self.assertIsNone(contract["bbox"])
        self.assertEqual(contract["grounding"]["scope_label"], "Whole project")
        self.assertEqual(contract["measure_label"], "Best available network impedance")
        self.assertEqual(contract["selected_model_tier"], "Topology-efficiency and bridge screening")
        self.assertEqual(contract["model_readiness"], "ready_with_available_data")

    def test_failed_llm_request_is_not_replaced_by_local_guide_answer(self) -> None:
        session_id = "p5_no_false_fallback_123456"
        configure_provider(
            session_id=session_id, provider="openai", api_key="sk-project-qa", model="gpt-test"
        )
        context = AssistantRequestContext(
            message="Explain the uploaded fields.",
            project={"project_id": "p1", "name": "Upload", "summary": {}},
            registry={"models": []}, selected_keys=[], bbox=None,
            current_metric="speed", current_time=None, current_view="explore", history=[],
        )
        try:
            with patch(
                "engine.assistant._provider_response",
                side_effect=ProviderRequestError(
                    "The API account has no usable credit or its spending limit has been reached.",
                    code="billing_or_quota",
                    http_status=429,
                ),
            ):
                result = assistant_response(
                    context=context, session_id=session_id, provider="openai", prefer_llm=True
                )
            self.assertTrue(result["fallback"])
            self.assertEqual(result["actions"], [])
            self.assertIn("No local-guide answer was substituted", result["message"])
            self.assertNotIn("open Find critical roads", result["message"])
            status = provider_status(session_id)
            openai = next(item for item in status["providers"] if item["id"] == "openai")
            self.assertEqual(openai["connection"], "inference_failed")
            self.assertEqual(openai["last_error_code"], "billing_or_quota")
        finally:
            configure_provider(session_id=session_id, provider="openai", api_key=None, clear=True)


if __name__ == "__main__":
    unittest.main()
