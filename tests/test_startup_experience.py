from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StartupExperienceTests(unittest.TestCase):
    def test_local_development_startup_and_intro(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1 --port 8005", readme)
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn("function initialiseCinematicIntro()", app)
        self.assertIn('const launchPath = /^\\/launch(?:-|=)[^/]+\\/?$/i.test(window.location.pathname);', app)
        self.assertIn('enter?.addEventListener("click", finish', app)
        self.assertIn('id="intro-skip"', page)
        for visual in ("intro-cyclone", "intro-rain", "intro-flood", "intro-embers"):
            self.assertIn(visual, page)

    def test_existing_projects_are_not_opened_automatically(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        initialise = app[app.index("async function initialise()") : app.index("function renderNoProject()")]
        self.assertIn("renderNoProject();", initialise)
        self.assertNotIn("loadProject(state.projects[0])", initialise)

    def test_start_screen_explicitly_supports_one_csv(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="start-panel"', page)
        self.assertIn("Disaster Transport Resilience Analysis", page)
        self.assertIn("No project or demonstration is opened automatically", page)
        self.assertIn("Upload one CSV or files", page)
        self.assertRegex(
            page,
            re.compile(r'<input id="upload-files" type="file" accept="\.csv,\.tsv,\.txt" multiple'),
        )

    def test_llm_session_is_shared_across_tabs_without_storing_the_api_key(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        session_function = app[app.index("function assistantSessionId()") : app.index("const state =")]
        self.assertIn('const storageKey = "twm-disaster-assistant-session"', session_function)
        self.assertIn("window.localStorage.getItem(storageKey)", session_function)
        self.assertIn("window.localStorage.setItem(storageKey, sessionId)", session_function)
        self.assertIn("window.sessionStorage.removeItem(storageKey)", session_function)
        self.assertNotIn("assistant-api-key", session_function)

    def test_loaded_workspace_exposes_a_non_destructive_back_to_twm_link(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('class="header-workspace-home"', page)
        self.assertIn('href="/?ui=', page)
        self.assertIn("Back to TWM", page)
        self.assertIn("body:not(.is-no-project) .header-workspace-home", styles)

    def test_data_format_guide_and_file_specific_diagnosis_are_reachable(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        for identifier in ("start-data-guide", "data-format-dialog", "mapping-data-help", "data-issue-list"):
            self.assertIn(f'id="{identifier}"', page)
        for structure in ("Locations or detectors", "Direct road links", "Nodes + links", "Topology only"):
            self.assertIn(structure, page)
        self.assertIn("function collectUploadIssues()", app)
        self.assertIn("openDataFormatGuide(\"diagnosis\")", app)
        self.assertIn("Exact example names are never required", page)
        self.assertIn("only genuine ambiguity or missing key structure needs user action", page)

    def test_link_reference_points_are_not_presented_as_topology_nodes(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="entity-kicker"', page)
        self.assertIn('id="entity-topology"', page)
        self.assertIn("Road-link reference point", app)
        self.assertIn("It is not either end node", app)
        self.assertIn("properties.from_node", app)
        self.assertIn("properties.to_node", app)
        self.assertIn("not a topology node or the complete road line", app)
        self.assertNotIn("`Road link ${properties.entity_id} · ${properties.region_id}`", app)

    def test_open_inspector_tracks_measure_and_time_changes(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("async function refreshOpenInspector()", app)
        self.assertGreaterEqual(app.count("await refreshOpenInspector();"), 4)

    def test_saved_point_to_link_view_and_fast_draggable_playback_are_wired(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        map_source = (PROJECT_ROOT / "frontend" / "js" / "map.js").read_text(encoding="utf-8")
        api = (PROJECT_ROOT / "frontend" / "js" / "api.js").read_text(encoding="utf-8")
        self.assertIn('id="view-points"', page)
        self.assertIn('id="view-links"', page)
        self.assertIn('id="network-dialog"', page)
        self.assertIn('id="playback-speed"', page)
        self.assertIn("buildAndOpenNetworkView", app)
        self.assertIn('state.mapView === "links" ? api.networkMap : api.map', app)
        self.assertIn("function playbackTiming()", app)
        self.assertIn("stopPlayback();\n    state.timeIndex = Number(event.target.value);", app)
        self.assertIn('this.map.addSource("network-links"', map_source)
        self.assertIn('id: "network-links"', map_source)
        self.assertIn("networkMappingStatus", api)
        self.assertIn("buildNetworkMapping", api)

    def test_first_detector_batch_is_revealed_and_viewport_refresh_stays_live(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        load_project = app[app.index("async function loadProject") : app.index("function chooseAvailableMetric")]
        self.assertLess(load_project.index('$("#map-empty").classList.add("is-hidden")'), load_project.index("await refreshScenarios()"))
        self.assertIn("mapViewportQueries", app)
        viewport = app[app.index("function viewportChanged") : app.index("function setMapBanner")]
        self.assertIn("if (!state.mapViewportQueries) return", viewport)
        self.assertNotIn("state.mapCompleting) return", viewport)

    def test_drag_area_focus_and_complete_result_table_are_wired(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        map_source = (PROJECT_ROOT / "frontend" / "js" / "map.js").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn('this.map.on("mousedown"', map_source)
        self.assertIn('this.map.on("mousemove"', map_source)
        self.assertIn('this.map.on("mouseup"', map_source)
        self.assertIn("map-area-drag-box", map_source)
        self.assertIn('$("#map-panel")?.scrollIntoView', app)
        self.assertIn("body.innerHTML = items.map", app)
        self.assertIn('id="result-list-count"', page)

    def test_latest_context_invalidates_review_and_assistant_can_complete_analysis(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn("function invalidateAnalysisReview()", app)
        self.assertIn('typeof arguments[0] === "string"', app)
        self.assertIn("state.commandContract = null", app)
        self.assertIn("state.commandRequestId += 1", app)
        self.assertIn("clearAnalysisOutput();", app)
        self.assertIn("interpretCurrentCommand();", app)
        self.assertIn("await prepareAssistantAnalysis(previewAction.value, { autoRun: true, surface: requestSurface })", app)
        self.assertIn("async function runAssistantAnalysis", app)
        self.assertIn('prepareAssistantAnalysis(previewAction.value, { autoRun: true, surface: requestSurface })', app)
        self.assertIn("await runAssistantAnalysis(null)", app)
        self.assertIn("appendAssistantResult(result)", app)
        self.assertIn("run a validated core analysis automatically", page)

    def test_llm_conversation_content_cannot_be_vertically_compressed(self) -> None:
        styles = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".assistant-panel.is-llm .assistant-messages", styles)
        self.assertIn("flex-direction: column", styles)
        self.assertIn("flex: 0 0 auto", styles)
        self.assertIn("max-height: none", styles)
        self.assertIn("overflow-y: scroll", styles)
        self.assertIn(".assistant-panel.is-llm .assistant-message table", styles)
        self.assertIn(".assistant-panel.is-llm .assistant-message img", styles)

    def test_capability_status_contains_only_delivered_or_data_gated_functions(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("Real-time emergency routing", page)
        self.assertNotIn("Evacuation and response optimisation", page)
        self.assertNotIn("Not claimed from static data", page)
        self.assertNotIn("future scope", page)
        self.assertIn("Six core tools, data tiers, Assistant and memory", page)

    def test_critical_area_ui_distinguishes_dpc_from_observed_grid_tier(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        map_source = (PROJECT_ROOT / "frontend" / "js" / "map.js").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn('result?.evidence?.executed_tier === "regional_dynamic_propagation_connectivity"', app)
        self.assertIn("Regional DPC loss", app)
        self.assertIn("Observed mean degradation", app)
        self.assertIn("Preprint-derived finite numerical screen", app)
        self.assertIn("properties.regional_dpc_score", map_source)
        self.assertIn("strongest evidence tier the upload supports", page)

    def test_unavailable_routes_explain_the_model_reason_without_naked_na(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("No directed route — uploaded road directions do not allow this trip. No reverse road was added.", app)
        self.assertIn("No route — endpoints are in different connected parts.", app)
        self.assertIn("No route — the selected start or destination road is closed.", app)
        self.assertIn("A legal route is required in both states.", app)
        self.assertIn("No connected comparison", app)
        self.assertIn("No route was available in both states.", app)
        self.assertIn("routeUnavailableMessage(variant.reason_code, variant.reason)", app)

    def test_independent_analysis_scope_is_explicit_and_old_map_marks_are_isolated(self) -> None:
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        for identifier in ("analysis-scope-control", "scope-whole", "scope-selection", "analysis-scope-note"):
            self.assertIn(f'id="{identifier}"', page)
        self.assertIn("Map marks stay visible but do not enter independent analyses", page)
        self.assertIn("function analysisContextForCommand(text)", app)
        self.assertIn('selected_keys: analysisContext.selected_keys', app)
        self.assertIn('bbox: analysisContext.bbox', app)
        self.assertIn('state.analysisScopeMode === "selection" ? state.routePreviousSelection : []', app)
        self.assertIn('state.analysisScopeMode === "selection" ? state.selectedKeys : []', app)
        self.assertIn('if (state.analysisScopeMode === "selection") invalidateAnalysisReview();', app)
        self.assertIn("Start fresh", page)

    def test_desktop_title_uses_the_current_two_line_platform_lockup(self) -> None:
        styles = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<span class="platform-title-line">Transport <b>World Model</b> Platform</span>', page)
        self.assertIn('<span class="platform-purpose-line">for Infrastructure Resilience, Response and Recovery</span>', page)
        self.assertRegex(styles, re.compile(r"\.site-header h1 \{[^}]*display: grid;[^}]*white-space: normal;"))

    def test_scenario_builder_is_map_first_saved_and_visually_explained(self) -> None:
        page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        app = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        api = (PROJECT_ROOT / "frontend" / "js" / "api.js").read_text(encoding="utf-8")
        map_source = (PROJECT_ROOT / "frontend" / "js" / "map.js").read_text(encoding="utf-8")
        for element_id in (
            "scenario-hub", "scenario-builder", "scenario-impact-panel", "scenario-library-dialog",
            "scenario-effect-closed", "scenario-effect-speed", "scenario-effect-time",
            "quick-build-scenario", "map-new-scenario",
        ):
            self.assertIn(f'id="{element_id}"', page)
        self.assertIn("scenarioAreaSelected", app)
        self.assertIn("scenarioTargets", api)
        self.assertIn("createScenario", api)
        self.assertIn('this.map.addSource("scenario-result"', map_source)
        self.assertIn('this.map.addSource("scenario-draft"', map_source)
        self.assertIn('id: "analysis-result-links"', map_source)
        self.assertIn("body.hazard-hurricane", styles)
        self.assertIn("body.hazard-flood", styles)
        self.assertIn("body.hazard-wildfire", styles)
        self.assertIn("Atmosphere is visual only", page)


if __name__ == "__main__":
    unittest.main()
