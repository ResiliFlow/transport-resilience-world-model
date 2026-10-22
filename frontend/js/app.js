import { api } from "./api.js?v=20260725r1";
import { drawHistory } from "./charts.js?v=20260718r1";
import { DisasterMap } from "./map.js?v=20260721r7";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const MATSIM_FRAME_DURATION_MS = 900;
const MATSIM_RENDER_INTERVAL_MS = 1000 / 30;

function assistantSessionId() {
  const storageKey = "twm-disaster-assistant-session";
  const validSessionId = (value) => /^[A-Za-z0-9_-]{16,128}$/.test(value || "");
  let sessionId = null;
  try {
    // Store only the opaque session identifier across same-origin tabs. The
    // provider API key remains exclusively in backend process memory.
    sessionId = window.localStorage.getItem(storageKey)
      || window.sessionStorage.getItem(storageKey);
  } catch (_) {
    sessionId = window.sessionStorage.getItem(storageKey);
  }
  if (!validSessionId(sessionId)) {
    const random = window.crypto?.randomUUID?.().replaceAll("-", "")
      || `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
    sessionId = `twm_${random}`.slice(0, 96);
  }
  try {
    window.localStorage.setItem(storageKey, sessionId);
    window.sessionStorage.removeItem(storageKey);
  } catch (_) {
    window.sessionStorage.setItem(storageKey, sessionId);
  }
  return sessionId;
}

const state = {
  bootstrap: null,
  project: null,
  projects: [],
  modelRegistry: [],
  modelRecommendations: null,
  preferredModelId: null,
  map: null,
  mapCollection: null,
  mapView: "points",
  networkMapping: null,
  analysisCollection: null,
  selectedKeys: new Set(),
  keptKeys: new Set(),
  focusedKey: null,
  bbox: null,
  analysisScopeMode: "whole",
  metric: "speed",
  timeValues: [],
  timeIndex: 0,
  commandContract: null,
  commandRequestId: 0,
  analysisContextVersion: 0,
  currentResult: null,
  completedResults: new Map(),
  resultStale: false,
  resultVisibleCount: 200,
  lastFeature: null,
  uploadPreview: null,
  uploadId: null,
  uploadPreviewError: null,
  dataGuideTab: "formats",
  playing: false,
  playTimer: null,
  playbackSpeed: 4,
  viewportTimer: null,
  mapRequestId: 0,
  mapCompleting: false,
  mapViewportQueries: false,
  inspectorRequestId: 0,
  suppressViewport: true,
  currentView: "explore",
  assistantStatus: null,
  assistantSessionId: assistantSessionId(),
  assistantProvider: "openai",
  assistantMode: "guide",
  assistantSurface: "platform",
  assistantMessageRequestId: 0,
  platformAssistantMessageRequestId: 0,
  assistantAnalysisRequestId: 0,
  assistantHistory: [],
  platformAssistantHistory: [],
  platformAssistantActions: [],
  assistantSurfaceViews: {
    platform: null,
    llm: null,
  },
  assistantChatId: null,
  assistantChatProjectId: null,
  assistantChats: [],
  assistantChatArtifacts: [],
  assistantChatDrawerOpen: false,
  assistantActivity: null,
  platformAssistantActivity: null,
  assistantPendingTask: null,
  assistantLastWorkflowKey: null,
  assistantTaskQueues: new Map(),
  assistantTurnRunning: false,
  assistantQueueDraining: false,
  assistantActiveQueueItem: null,
  assistantCompoundWorkflow: null,
  assistantActions: [],
  assistantContract: null,
  assistantBusy: false,
  platformAssistantBusy: false,
  assistantLlmDrag: null,
  assistantLlmStyleBackup: null,
  guideIndex: 0,
  activeAnalysisJobId: null,
  analysisWaitCancelled: false,
  analysisWaitCancel: null,
  analysisWaitStarted: 0,
  scenarios: [],
  activeScenarioId: null,
  scenarioRequestId: 0,
  scenarioBuilding: false,
  scenarioEditingId: null,
  scenarioEffect: "closed",
  scenarioDraft: new Map(),
  scenarioDraftFeatures: new Map(),
  scenarioUndo: [],
  scenarioAreaLoading: false,
  routeBuilding: false,
  routePickRole: null,
  routeOrigin: null,
  routeDestination: null,
  routeEffect: "closed",
  routeDisruptionMode: null,
  routeDraft: new Map(),
  routeDraftFeatures: new Map(),
  routeAreaLoading: false,
  routePreviousSelection: [],
  routePreviousScenarioId: null,
  recoveryBuilding: false,
  recoveryMode: null,
  recoveryUnit: "road",
  recoveryDraft: new Set(),
  recoveryDraftFeatures: new Map(),
  recoveryAreas: [],
  recoveryPreviousSelection: [],
  stressBuilding: false,
  atmosphereEnabled: true,
  capabilityIndex: 0,
  capabilityTimer: null,
  capabilityPaused: false,
  projectMemory: null,
  releaseScientificValidation: null,
  scientificValidation: null,
  researchReports: [],
  researchReport: null,
  researchJobId: null,
  researchFigureSvgs: new Map(),
  scenarioEngine: "network",
  matsimReadiness: null,
  matsimJobId: null,
  matsimFrames: [],
  matsimFrameIndex: 0,
  matsimResult: null,
  matsimPlaying: false,
  matsimPlayTimer: null,
  matsimTransitionStartedAt: null,
  matsimTransitionProgress: 0,
  matsimLastRenderAt: 0,
};

const PUBLIC_MODEL_IDS = {
  "P5-M0": "TWM-GEO-SUMMARY", "P5-M1": "TWM-OBS-PERSIST", "P5-M2": "TWM-NET-CRITICALITY",
  "P5-M3": "TWM-NET-CLOSURE", "P5-M4": "TWM-RECOVERY-MARGINAL", "P5-M5": "TWM-EXPOSURE-SENSITIVITY",
  "P5-M6": "TWM-ACCESSIBILITY", "P5-M7": "TWM-DISRUPTION-SCENARIO", "P5-M8": "TWM-RECOVERY-RESOURCE",
  "P5-M9": "TWM-RESILIENCE-CURVE", "P5-M10": "TWM-GEO-HOTSPOT", "P5-M11": "TWM-ROBUSTNESS-PROGRESSIVE",
  "P5-M12": "TWM-RELIABILITY-ENVELOPE", "P5-M13": "TWM-CASCADE-STRESS", "P5-M14": "TWM-HAZARD-EXPOSURE",
  "P5-M15": "TWM-EQUITY-ACCESS", "P5-M16": "TWM-PROTECTION-PRIORITY", "P5-M17": "TWM-RESPONSE-SCENARIO",
  "P5-M18": "TWM-EVACUATION-READINESS", "P5-M19": "TWM-CHRONO-VALIDATION", "P5-M20": "TWM-INVESTMENT-SCREEN",
  "P5-M21": "TWM-DISRUPTION-COUNTERFACTUAL", "P5-M22": "TWM-REROUTING-PROPAGATION",
  "P5-M23": "TWM-HISTORICAL-DETOUR",
};

function publicModelId(modelId) {
  return PUBLIC_MODEL_IDS[String(modelId || "")] || String(modelId || "Model family").replace(/^P5-/, "TWM-");
}

const DISPLAY_METRICS = [
  ["speed", "Speed", "speed"], ["travel_time", "Travel time", "travel_time"],
  ["primary", "Uploaded measure", "primary_metric"], ["flow", "Observed traffic flow", "flow"],
  ["capacity", "Link capacity", "capacity"], ["free_flow_time", "Free-flow travel time", "free_flow_time"],
  ["repair_cost", "Repair cost", "repair_cost"], ["repair_duration", "Repair duration", "repair_duration"],
  ["failure_probability", "Failure probability", "failure_probability"], ["length", "Link length", "length"],
];

const MODEL_ANALYSIS_TYPES = {
  "P5-M0": "area_summary", "P5-M1": "persistent_degradation", "P5-M2": "critical_components",
  "P5-M3": "closure_scenario", "P5-M4": "restoration_screening", "P5-M9": "temporal_resilience",
  "P5-M10": "spatial_hotspots", "P5-M11": "progressive_robustness",
  "P5-M23": "route_detour_analysis",
};

const CAPABILITY_LABELS = {
  historical_reconstruction: "Historical reconstruction",
  persistent_degradation: "Find recurring problem locations",
  topology_criticality: "Find critical roads",
  closure_scenario: "Build a disruption scenario",
  restoration_screening: "Plan recovery priorities",
  temporal_resilience: "Track disruption and recovery",
  spatial_hotspots: "Find critical areas",
  progressive_robustness: "Test network resilience",
  accessibility: "Essential-service accessibility",
  traffic_assignment: "Traffic redistribution",
  resource_optimisation: "Resource recovery optimisation",
  route_detour_analysis: "Plan routes under disruption",
};

const LINK_RESULT_ANALYSES = new Set([
  "critical_components",
  "closure_scenario",
  "restoration_screening",
  "progressive_robustness",
  "route_detour_analysis",
]);
const STRICT_LINE_RESULT_ANALYSES = new Set(["critical_components", "progressive_robustness"]);

const GUIDE_STEPS = [
  {
    title: "Open your data",
    copy: "Drop one or more CSV or TSV files. The platform recognises alternative column names file by file.",
    tip: "Confirm: ID + latitude + longitude + one measured value.",
    target: "upload",
    visual: '<div class="guide-file">CSV</div><i>→</i><div class="guide-check">✓</div>',
  },
  {
    title: "Read the uploaded history",
    copy: "Use the map-side statistical views to find recurring problems, follow the disruption timeline or summarise a drawn area.",
    tip: "These views describe uploaded observations; they do not simulate a response.",
    target: "history",
    visual: '<div class="guide-map"><b></b><b></b><b></b><span>⌖</span></div><i>→</i><div class="guide-result">History</div>',
  },
  {
    title: "Choose a decision tool",
    copy: "Find critical roads or areas, set recovery priorities, plan routes under disruption, test resilience or build a scenario.",
    tip: "Each tool opens only the map inputs it actually needs.",
    target: "core",
    visual: '<div class="guide-review">6<span>tools</span></div><i>→</i><div class="guide-result">Decision</div>',
  },
  {
    title: "Plan recovery priorities",
    copy: "Start from disrupted roads, critical roads, critical areas or a saved scenario. Choose roads or areas and any positive Top-N.",
    tip: "Small sets are re-evaluated after each recovery stage; large sets are clearly labelled as structural screening.",
    target: "recovery",
    visual: '<div class="guide-file">Damage</div><i>→</i><div class="guide-model">Model</div><i>→</i><div class="guide-result">Priority</div>',
  },
  {
    title: "Plan routes under disruption",
    copy: "Pick two routable points or roads, then close, slow or delay selected roads or an area and compare route options.",
    tip: "Routes are loopless, overlap-screened and retrospective—not live navigation.",
    target: "route",
    visual: '<div class="guide-pin">A</div><i>→</i><div class="guide-model">⌁</div><i>→</i><div class="guide-pin">B</div>',
  },
  {
    title: "Ask for help or let the assistant operate",
    copy: "Platform Assistant works locally. LLM Copilot can use GPT, Claude, DeepSeek or Gemini after you connect your own API.",
    tip: "Operate in platform is allowlisted and reversible; it cannot perform live-world actions.",
    target: "assistant",
    visual: '<div class="guide-bubble">Plain request</div><i>→</i><div class="guide-model">AI</div>',
  },
  {
    title: "Review, then confirm",
    copy: "Check the selected data, assumptions and limits. Confirm only when the request matches your intent.",
    tip: "Results are saved locally with evidence and exportable JSON.",
    target: "review",
    visual: '<div class="guide-review">✓<span>model</span></div><i>→</i><div class="guide-result">▥</div>',
  },
  {
    title: "Reopen the evidence",
    copy: "Project Memory records confirmed mappings, model runs, scenarios, verification and evidence boundaries.",
    tip: "Saved results and memory are exportable; API keys and transient chat are not stored there.",
    target: "memory",
    visual: '<div class="guide-file">Data</div><i>→</i><div class="guide-review">✓<span>run</span></div><i>→</i><div class="guide-result">Memory</div>',
  },
];

const CAPABILITY_SLIDES = [
  {
    kicker: "Flexible data entry",
    title: "Different columns, one transport workspace",
    copy: "Upload one or several delimited files. The platform identifies points, direct roads, Node + Link packages or topology, then asks you to confirm only ambiguous roles.",
    proof: "Known coordinates open on a basemap; local grids and topology-only networks open in a clearly labelled network layout.",
    visual: '<div class="cap-file"><b>CSV</b><span>lon · lat · speed</span></div><i>→</i><div class="cap-file alt"><b>TSV</b><span>xCoord · yCoord · delay</span></div><i>→</i><div class="cap-map-icon">⌖</div>',
  },
  {
    kicker: "Uploaded-history views",
    title: "Read what happened beside the map",
    copy: "Recurring problem roads, the disruption timeline and a drawn-area summary sit beside the network view because they describe the history you uploaded.",
    proof: "These views report observations and internal references. They do not invent a response scenario or causal disaster effect.",
    visual: '<div class="cap-dashboard"><div class="cap-mini-map"><i></i><i></i><i></i></div><div class="cap-bars"><span></span><span></span><span></span><span></span></div></div>',
  },
  {
    kicker: "Core decision tools",
    title: "Six practical tasks in one place",
    copy: "Find critical roads, find critical areas, plan recovery priorities, plan routes under disruption, test network resilience or build a disruption scenario.",
    proof: "A tool requests only the map selection and data fields its registered model needs; historical statistics are not mixed into this decision area.",
    visual: '<div class="cap-review"><span>Decision tools</span><span>Critical roads</span><span>Critical areas</span><span>Recovery</span><span>Routes</span><span>Resilience</span><span>Scenario</span></div>',
  },
  {
    kicker: "Disruption scenario builder",
    title: "Change roads, then see direct and indirect effects",
    copy: "Close roads, reduce speed or increase travel time at a chosen historical baseline. The result separates roads you changed from other links that carry rerouted proxy paths.",
    proof: "Topology runs a disclosed deterministic proxy; flow, capacity and free-flow time unlock the richer Beckmann–BPR tier. The platform never labels proxy load as observed traffic.",
    visual: '<div class="cap-flow"><span>Changed roads</span><i>→</i><b>Network model</b><i>→</i><span>Other affected roads</span></div>',
  },
  {
    kicker: "Routes under disruption",
    title: "Choose two points, add disruption, compare options",
    copy: "Pick two routable points or roads. Add a temporary closure, slowdown or travel-time increase, or use a saved scenario.",
    proof: "Dijkstra finds the primary route; Yen-style loopless enumeration supplies sufficiently different alternatives. Ordinary points are not randomly snapped.",
    visual: '<div class="cap-flow"><span>Start point</span><i>→</i><b>History + scenarios</b><i>→</i><span>Destination</span></div>',
  },
  {
    kicker: "Adaptive model routing",
    title: "The strongest honest model follows the data",
    copy: "The registry selects an executable evidence tier instead of demanding one rigid template. Optional demand, capacity, cost or duration fields unlock richer claims.",
    proof: "Every run states its required fields, selected tier, literature source, verification and claim boundary.",
    visual: '<div class="cap-router"><span>Question</span><span>Data profile</span><b>Model router</b><div><i>Base</i><i>Observed</i><i>Capacity</i></div></div>',
  },
  {
    kicker: "Two clear assistant surfaces",
    title: "Local platform help or an optional LLM Copilot",
    copy: "Platform Assistant explains and opens the right tool without an API. LLM Copilot connects GPT, Claude, DeepSeek or Gemini and can operate registered workflows from a plain-language request.",
    proof: "The Copilot can prepare and run registered models, compare saved-run evidence and cross-review an explicit new formulation. New generated mathematics cannot execute until it has an adapter and deterministic tests.",
    visual: '<div class="cap-providers"><strong>Platform Assistant</strong><span>+</span><b>GPT</b><b>Claude</b><b>DeepSeek</b><b>Gemini</b><strong>LLM Copilot</strong></div>',
  },
  {
    kicker: "Memory and evidence boundary",
    title: "Reopen what ran and see what it cannot claim",
    copy: "Project Memory connects confirmed source mappings and hashes with each model run, scenario, result tier and verification boundary. Results separate observed evidence, proxy assumptions and unavailable conclusions.",
    proof: "Memory exports to JSON; API keys and transient chat stay out. Static history supports planning evidence, not live emergency control.",
    visual: '<div class="cap-boundary"><div><b>✓</b><span>Saved evidence</span></div><div><b>!</b><span>Honest boundary</span></div></div>',
  },
];

const MAPPING_ROLES = [
  ["entity_id", "Entity or link ID", true],
  ["latitude", "Latitude", true],
  ["longitude", "Longitude", true],
  ["time", "Date and time", false],
  ["speed", "Speed", false],
  ["travel_time", "Travel time", false],
  ["primary_metric", "Other measured value", false],
  ["from_node", "From node", false],
  ["to_node", "To node", false],
  ["length", "Link length", false],
  ["flow", "Observed traffic flow", false],
  ["capacity", "Link capacity", false],
  ["free_flow_time", "Free-flow travel time", false],
  ["repair_cost", "Repair cost", false],
  ["repair_duration", "Repair duration", false],
  ["failure_probability", "Failure probability", false],
  ["time_step", "Time-step index", false],
  ["region", "Region identifier", false],
];

const PLAYBACK_PROFILES = {
  1: { targetDurationMs: 90000, minimumDelayMs: 200, maximumDelayMs: 700, maximumFrames: 56 },
  2: { targetDurationMs: 45000, minimumDelayMs: 140, maximumDelayMs: 550, maximumFrames: 28 },
  4: { targetDurationMs: 22000, minimumDelayMs: 90, maximumDelayMs: 380, maximumFrames: 14 },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
}

function formatBytes(bytes) {
  const numeric = Number(bytes || 0);
  if (numeric < 1024) return `${numeric} B`;
  if (numeric < 1024 ** 2) return `${(numeric / 1024).toFixed(1)} KB`;
  if (numeric < 1024 ** 3) return `${(numeric / 1024 ** 2).toFixed(1)} MB`;
  return `${(numeric / 1024 ** 3).toFixed(2)} GB`;
}

function hasText(value) {
  return value !== null && value !== undefined && String(value).trim() !== "";
}

function featureUsesLinkTopology(properties = {}) {
  return hasText(properties.from_node) && hasText(properties.to_node);
}

function projectUsesLinkTopology(summary = state.project?.summary) {
  const status = summary?.capabilities?.topology_criticality;
  return status === "available" || status === "conditional";
}

function geometryNoteForProject(summary = state.project?.summary) {
  if (state.mapView === "links" && state.networkMapping?.ready) {
    return state.networkMapping.warning || "The available road-link geometry is displayed directly.";
  }
  if (projectUsesLinkTopology(summary)) {
    return "Each dot is a road-link reference point, not a topology node or the complete road line.";
  }
  return "Each dot is an uploaded entity location. The platform does not infer line geometry.";
}

function mappedObservationLabel(count, summary = state.project?.summary) {
  const total = Number(count || 0).toLocaleString();
  if (state.mapView === "links") return state.networkMapping?.exact_geometry
    ? `${total} road-link observations`
    : `${total} inferred link observations`;
  return projectUsesLinkTopology(summary) ? `${total} road-link observations` : `${total} entity observations`;
}

function metricCopy(metric) {
  const fixed = {
    speed: { observed: "Observed speed", reference: "Reference speed", history: "Speed history" },
    travel_time: { observed: "Observed travel time", reference: "Reference travel time", history: "Travel-time history" },
    primary: { observed: "Observed value", reference: "Reference value", history: "Uploaded-measure history" },
  }[metric];
  if (fixed) return fixed;
  const label = DISPLAY_METRICS.find(([key]) => key === metric)?.[1] || metric.replaceAll("_", " ");
  return { observed: label, reference: "No inferred reference", history: `${label} history` };
}

function metricSourceNames(summary, role) {
  return [...new Set(Object.values(summary?.mapping?.by_file || {})
    .map((mapping) => mapping?.[role])
    .filter(Boolean)
    .map(String))];
}

function availableMetricOptions(summary) {
  const specialisedRoles = DISPLAY_METRICS
    .filter(([key]) => !["speed", "travel_time", "primary"].includes(key))
    .map(([, , role]) => role);
  const specialisedNames = new Set(specialisedRoles.flatMap((role) => metricSourceNames(summary, role)));
  const primaryNames = metricSourceNames(summary, "primary_metric");
  const primaryIsStorageAlias = primaryNames.length > 0 && primaryNames.every((name) => specialisedNames.has(name));
  return DISPLAY_METRICS.filter(([key]) => summary?.metrics?.[key]?.available && !(key === "primary" && primaryIsStorageAlias)).map(([key, label, role]) => {
    const names = metricSourceNames(summary, role);
    const source = names.length === 1 ? ` · ${names[0]}` : names.length > 1 ? ` · ${names.length} source fields` : "";
    const unit = summary.metrics?.[key]?.unit;
    return { key, label: `${label}${source}${unit ? ` (${unit})` : ""}`, unit: unit || "uploaded unit" };
  });
}

function metricSupportsSeverity(metric = state.metric, summary = state.project?.summary) {
  if (["speed", "travel_time"].includes(metric)) return true;
  return metric === "primary" && Boolean(summary?.metrics?.primary?.semantics?.severity_enabled);
}

function setLoading(visible, title = "Working", copy = "Please wait.", { analysis = false } = {}) {
  const overlay = $("#loading-overlay");
  $("#loading-title").textContent = title;
  $("#loading-copy").textContent = copy;
  overlay.classList.toggle("is-hidden", !visible);
  $("#analysis-wait")?.classList.toggle("is-hidden", !visible || !analysis);
}

const WORKLOAD_WAIT_MIN_MS = 15_000;
const WORKLOAD_WAIT_MAX_MS = 50_000;
const RANKING_WAIT_MIN_MS = 12_000;
const RANKING_WAIT_MAX_MS = 45_000;
const WORKLOAD_WAIT_SESSION_JITTER_MS = Math.round(Math.random() * 1_800);
const REAL_RUNTIME_ANALYSES = new Set();
const RANKING_WAIT_ANALYSES = new Set(["critical_components", "spatial_hotspots"]);

function workloadWaitProfile(payload = {}, options = {}) {
  const analysisType = String(payload.analysis_type || options.analysisType || "");
  if (REAL_RUNTIME_ANALYSES.has(analysisType)) {
    return { minimumMs: 0, analysisType, mode: "actual" };
  }
  const projectSize = Math.max(1, Number(state.project?.summary?.entity_count || state.project?.summary?.mapped_entity_count || 1));
  const minimumBoundMs = RANKING_WAIT_ANALYSES.has(analysisType) ? RANKING_WAIT_MIN_MS : WORKLOAD_WAIT_MIN_MS;
  const maximumBoundMs = RANKING_WAIT_ANALYSES.has(analysisType) ? RANKING_WAIT_MAX_MS : WORKLOAD_WAIT_MAX_MS;
  const selectedCount = Math.max(0, Number(options.selectedCount ?? payload.selected_keys?.length ?? 0));
  const requestedCount = Math.max(1, Number(options.requestedCount ?? payload.top_k ?? 1));
  const changeCount = Math.max(0, Number(options.changeCount ?? payload.assumptions?.scenario_changes?.length ?? 0));
  const networkWide = Boolean(options.networkWide) || ["progressive_robustness", "route_detour_analysis", "closure_scenario"].includes(analysisType);
  const extent = state.project?.summary?.extent;
  let bboxScopeCount = 0;
  if (Array.isArray(payload.bbox) && payload.bbox.length === 4 && Array.isArray(extent) && extent.length === 4) {
    const projectArea = Math.abs((Number(extent[2]) - Number(extent[0])) * (Number(extent[3]) - Number(extent[1])));
    const bboxArea = Math.abs((Number(payload.bbox[2]) - Number(payload.bbox[0])) * (Number(payload.bbox[3]) - Number(payload.bbox[1])));
    if (projectArea > 0 && Number.isFinite(bboxArea)) bboxScopeCount = Math.ceil(projectSize * Math.max(0.01, Math.min(1, bboxArea / projectArea)));
  }
  const scopeCount = Math.max(1, Number(options.scopeCount || (networkWide ? projectSize : selectedCount || bboxScopeCount || projectSize)));
  const quantityFactor = Math.min(1, Math.log10(requestedCount + 1) / 3);
  const scopeFactor = Math.min(1, Math.log10(scopeCount + 1) / Math.max(3, Math.log10(projectSize + 1)));
  const changeFactor = changeCount ? Math.min(1, Math.log10(changeCount + 1) / 3) : 0;
  let workload = 0.62 * quantityFactor + 0.38 * scopeFactor;
  if (["route_detour_analysis", "closure_scenario"].includes(analysisType)) {
    workload = 0.32 * quantityFactor + 0.34 * scopeFactor + 0.34 * changeFactor;
  } else if (analysisType === "restoration_screening") {
    workload = 0.48 * quantityFactor + 0.52 * scopeFactor;
  } else if (analysisType === "progressive_robustness") {
    workload = 0.72 * quantityFactor + 0.28 * scopeFactor;
  }
  const jitterMs = Math.round(WORKLOAD_WAIT_SESSION_JITTER_MS * (0.25 + 0.75 * workload));
  const minimumMs = Math.max(
    minimumBoundMs,
    Math.min(maximumBoundMs, Math.round(minimumBoundMs + workload * (maximumBoundMs - minimumBoundMs) + jitterMs)),
  );
  return { minimumMs, analysisType, mode: "workload", requestedCount, scopeCount, changeCount };
}

function updateAnalysisWaitDisplay(startedAt, targetMs, { message = null, complete = false } = {}) {
  const elapsedMs = Math.max(0, Date.now() - startedAt);
  const elapsed = Math.round(elapsedMs / 1000);
  const progress = complete ? 100 : Math.max(0, Math.min(99, targetMs > 0 ? elapsedMs / targetMs * 100 : 0));
  if (message) $("#loading-copy").textContent = message;
  $("#analysis-wait-elapsed").textContent = `Elapsed ${elapsed}s · ${progress.toFixed(0)}% complete`;
  $("#analysis-wait-progress").style.width = `${progress}%`;
  $("#analysis-wait-progress").classList.toggle("is-running", !complete);
}

function beginWorkloadWait(profile, title, copy) {
  const startedAt = Date.now();
  const targetMs = Math.max(WORKLOAD_WAIT_MIN_MS, Number(profile?.minimumMs || WORKLOAD_WAIT_MIN_MS));
  const cancelButton = $("#cancel-analysis-wait");
  let closed = false;
  let cancelled = false;
  state.analysisWaitStarted = startedAt;
  state.analysisWaitCancelled = false;
  state.activeAnalysisJobId = null;
  // Keep the control visible for the entire local preparation wait.
  // It must not disappear while the confirmed model is still being prepared.
  cancelButton.hidden = false;
  $("#analysis-wait small").textContent = "Run preparation scales with the confirmed workload; larger requests and wider scopes take longer.";
  setLoading(true, title, copy, { analysis: true });
  updateAnalysisWaitDisplay(startedAt, targetMs, { message: copy });
  const timer = window.setInterval(() => updateAnalysisWaitDisplay(startedAt, targetMs), 250);
  const close = () => {
    if (closed) return;
    closed = true;
    window.clearInterval(timer);
    if (state.analysisWaitCancel === cancel) state.analysisWaitCancel = null;
    cancelButton.hidden = false;
    setLoading(false);
  };
  const cancel = () => {
    if (closed) return false;
    cancelled = true;
    state.analysisWaitCancelled = true;
    close();
    return true;
  };
  state.analysisWaitCancel = cancel;
  return {
    async complete(message = "Model result ready · preparing the map and visual evidence…") {
      if (cancelled) return false;
      $("#loading-title").textContent = "Preparing the confirmed result";
      $("#loading-copy").textContent = message;
      const remaining = Math.max(0, targetMs - (Date.now() - startedAt));
      if (remaining) await new Promise((resolve) => window.setTimeout(resolve, remaining));
      if (cancelled) return false;
      updateAnalysisWaitDisplay(startedAt, targetMs, { message, complete: true });
      await new Promise((resolve) => window.setTimeout(resolve, 160));
      close();
      return true;
    },
    fail: close,
    close,
    cancel,
    get cancelled() { return cancelled; },
  };
}

function toast(title, message, error = false) {
  const element = document.createElement("div");
  element.className = `toast${error ? " is-error" : ""}`;
  const strong = document.createElement("strong");
  const span = document.createElement("span");
  strong.textContent = title;
  span.textContent = message;
  element.append(strong, span);
  $("#toast-region").append(element);
  window.setTimeout(() => element.remove(), 5200);
}

function markResultStale(reason = "The question, selection, measure or time changed.") {
  if (!state.currentResult) return;
  state.resultStale = true;
  state.analysisCollection = null;
  state.map?.setAnalysisFeatures(null);
  const stale = $("#result-stale");
  stale?.classList.remove("is-hidden");
  const message = stale?.querySelector("span");
  if (message) message.textContent = `${reason} Preview and run again for the current context.`;
  const status = $("#result-status");
  if (status) {
    status.textContent = "Previous result";
    status.classList.add("is-conditional");
  }
  if ($("#focus-result")) $("#focus-result").disabled = true;
  renderMapLegend();
}

function clearAnalysisOutput({ hidePanel = true } = {}) {
  state.currentResult = null;
  state.resultStale = false;
  state.analysisCollection = null;
  state.map?.setAnalysisFeatures(null);
  $("#result-stale")?.classList.add("is-hidden");
  if (hidePanel) $("#result-panel")?.classList.add("is-hidden");
  if ($("#focus-result")) $("#focus-result").disabled = true;
  renderMapLegend();
}

function rememberCompletedResult(result) {
  if (!result?.analysis_type || !state.project?.project_id) return;
  state.completedResults.set(result.analysis_type, {
    projectId: state.project.project_id,
    completedAt: Date.now(),
    result,
  });
}

function completedResultFor(analysisType) {
  const record = state.completedResults.get(analysisType);
  if (!record || record.projectId !== state.project?.project_id) return null;
  return record.result || null;
}

function assistantArtifactFromMetadata(metadata = {}, createdAt = null) {
  const kind = String(metadata.kind || "");
  const analysisType = String(metadata.analysis_type || "");
  const artifactId = String(metadata.run_id || metadata.scenario_id || "");
  const projectId = String(metadata.project_id || "");
  if (!["analysis_result", "scenario_result"].includes(kind) || !analysisType || !artifactId || !projectId) return null;
  return {
    kind,
    artifact_id: artifactId,
    analysis_type: analysisType,
    project_id: projectId,
    created_at: createdAt || metadata.created_at || new Date().toISOString(),
  };
}

function registerAssistantChatArtifact(metadata = {}, createdAt = null) {
  const artifact = assistantArtifactFromMetadata(metadata, createdAt);
  if (!artifact || artifact.project_id !== currentProjectId()) return null;
  state.assistantChatArtifacts = state.assistantChatArtifacts.filter((item) => !(
    item.kind === artifact.kind && item.artifact_id === artifact.artifact_id
  ));
  state.assistantChatArtifacts.push(artifact);
  return artifact;
}

function assistantArtifactsForRequest() {
  const projectId = currentProjectId();
  return state.assistantChatArtifacts
    .filter((item) => item.project_id === projectId)
    .slice(-100)
    .map((item) => ({ ...item }));
}

function latestAssistantArtifact(analysisTypes) {
  const accepted = new Set(Array.isArray(analysisTypes) ? analysisTypes : [analysisTypes]);
  const projectId = currentProjectId();
  return [...state.assistantChatArtifacts].reverse().find((item) => (
    item.project_id === projectId && accepted.has(item.analysis_type)
  )) || null;
}

async function hydrateAssistantArtifact(artifact) {
  if (!artifact || artifact.project_id !== currentProjectId()) return null;
  if (artifact.kind === "analysis_result") {
    const record = await api.run(artifact.artifact_id);
    if (record.project_id !== currentProjectId() || record.analysis_type !== artifact.analysis_type) {
      throw new Error("The referenced result belongs to a different project or analysis.");
    }
    rememberCompletedResult(record.result);
    return record.result;
  }
  if (artifact.kind === "scenario_result") {
    const scenario = await api.scenario(artifact.artifact_id);
    if (scenario.project_id !== currentProjectId()) throw new Error("The referenced scenario belongs to a different project.");
    return scenario;
  }
  return null;
}

function invalidateAnalysisReview() {
  // DOM listeners pass an Event object as their first argument.  Never expose
  // that object as user-facing text when a question or control changes.
  const reason = typeof arguments[0] === "string" ? arguments[0] : "The analysis context changed.";
  state.commandRequestId += 1;
  state.analysisContextVersion += 1;
  state.commandContract = null;
  $("#analysis-review")?.classList.add("is-hidden");
  if ($("#interpret-command")) $("#interpret-command").disabled = false;
  if (state.assistantContract) {
    state.assistantContract = null;
    $$('[data-assistant-run]').forEach((button) => {
      button.disabled = true;
      button.textContent = "Context changed · ask again";
    });
  }
  clearAnalysisOutput();
  updateReadiness();
}

function switchView(view) {
  state.currentView = view;
  $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.viewPanel === view));
  $$(".nav-button").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
  window.scrollTo({ top: 0, behavior: "auto" });
  window.setTimeout(() => state.map?.resize(), 80);
  if (view === "saved") refreshRuns();
  if (view === "memory") refreshProjectMemory();
  if (view === "research") refreshResearchReports({ loadLatest: !state.researchReport }).catch(() => {});
  updateAssistantContext();
  scheduleNextActionCues();
}

function currentTime() {
  return state.timeValues[state.timeIndex] ?? null;
}

function currentTimeLabel() {
  const summary = state.project?.summary;
  const timestamp = currentTime();
  return summary?.time_labels?.[String(timestamp)] || (timestamp === 0 ? "Static snapshot" : "No timeline");
}

function currentProjectId() {
  return state.project?.project_id || null;
}

const HAZARD_LABELS = {
  hurricane: "Hurricane / cyclone",
  flood: "Flood",
  wildfire: "Wildfire",
  earthquake: "Earthquake",
  storm: "Severe storm",
  multi_hazard: "Multi-hazard",
  other: "Other disruption",
};

const SCENARIO_EFFECT_LABELS = {
  closed: "Closed",
  speed_reduction: "Slower",
  travel_time_increase: "Longer time",
};

function hazardLabel(value) {
  return HAZARD_LABELS[value] || HAZARD_LABELS.other;
}

function setHazardAtmosphere(hazard) {
  const classes = ["hurricane", "flood", "wildfire", "earthquake", "storm", "multi-hazard", "other"];
  classes.forEach((name) => document.body.classList.remove(`hazard-${name}`));
  const cleaned = String(hazard || "other").replaceAll("_", "-");
  if (hazard) document.body.classList.add(`hazard-${classes.includes(cleaned) ? cleaned : "other"}`);
  document.body.classList.toggle("atmosphere-off", !state.atmosphereEnabled);
  const button = $("#atmosphere-toggle");
  if (button) {
    button.setAttribute("aria-pressed", String(state.atmosphereEnabled));
    const copy = button.querySelector("b");
    if (copy) copy.textContent = state.atmosphereEnabled ? "Atmosphere on" : "Atmosphere off";
  }
}

function activeScenario() {
  return state.scenarios.find((scenario) => scenario.scenario_id === state.activeScenarioId) || null;
}

function scenarioFeature(feature, change) {
  return {
    type: "Feature",
    geometry: {
      type: feature.geometry?.type || "Point",
      coordinates: feature.geometry?.coordinates || [],
    },
    properties: {
      ...(feature.properties || {}),
      ...change,
      severity: change.effect === "closed" ? 1 : Math.min(1, Number(change.magnitude || 0)),
      scenario: true,
    },
  };
}

function renderScenarioControls() {
  const select = $("#scenario-select");
  select.innerHTML = [
    '<option value="">Baseline · no added disruption</option>',
    ...state.scenarios.map((scenario) => `<option value="${escapeHtml(scenario.scenario_id)}">${escapeHtml(scenario.name)} · ${escapeHtml(hazardLabel(scenario.hazard_type))}</option>`),
  ].join("");
  select.value = state.activeScenarioId || "";
  $("#scenario-count").textContent = String(state.scenarios.length);
  $("#edit-scenario").disabled = !state.activeScenarioId || state.scenarioBuilding;
  renderRouteScenarioOptions();
  renderScenarioLibrary($("#scenario-search")?.value || "");
}

async function refreshScenarios() {
  if (!state.project) return;
  const preferred = state.activeScenarioId;
  state.scenarios = await api.scenarios(state.project.project_id);
  if (preferred && !state.scenarios.some((scenario) => scenario.scenario_id === preferred)) {
    state.activeScenarioId = null;
  }
  renderScenarioControls();
  activateScenario(state.activeScenarioId);
}

function renderComponentNetwork(element, componentCount, split = false) {
  if (!element) return;
  const numeric = Number(componentCount);
  if (!Number.isFinite(numeric)) {
    element.innerHTML = "<span>Network unavailable</span>";
    element.classList.remove("is-split");
    return;
  }
  const dots = Math.max(4, Math.min(12, numeric * 3));
  element.innerHTML = Array.from({ length: dots }, () => "<i></i>").join("");
  element.classList.toggle("is-split", split);
}

function legacyRenderScenarioImpact(scenario) {
  const panel = $("#scenario-impact-panel");
  if (!scenario) {
    panel.classList.add("is-hidden");
    return;
  }
  const result = scenario.result || {};
  const visual = result.visual || {};
  const evidence = result.evidence || {};
  const counts = result.change_counts || {};
  const networkAvailable = visual.mode === "network_compare";
  const retained = networkAvailable ? Number(visual.scenario_efficiency_percent || 0) : null;
  $("#scenario-result-title").textContent = scenario.name;
  $("#scenario-plain-finding").textContent = result.plain_finding || "The saved scenario is ready to inspect.";
  const impactLabels = {
    network_split: "Network split",
    large_impedance: "Large movement loss",
    noticeable_impedance: "Noticeable movement loss",
    limited_topology_change: "Limited whole-network change",
    mapped_locations: "Mapped locations only",
  };
  $("#scenario-impact-chip").textContent = impactLabels[result.impact_level] || "Deterministic";
  $("#scenario-finding-symbol").textContent = result.impact_level === "network_split" ? "↯" : networkAvailable ? "≈" : "•";
  $("#scenario-retained-bar").style.width = networkAvailable ? `${Math.max(0, Math.min(100, retained))}%` : "0%";
  $("#scenario-retained-value").textContent = networkAvailable ? `${evidence.efficiency_exact === false ? "≈" : ""}${retained.toFixed(1)}%` : "Not calculated";
  $("#scenario-retained-value").title = evidence.efficiency_exact === false ? "Deterministic estimate from fixed sampled source nodes" : "Exact network calculation";
  const baselineComponents = visual.baseline_components;
  const scenarioComponents = visual.scenario_components;
  const fragmented = networkAvailable && Number(scenarioComponents) > Number(baselineComponents);
  $("#scenario-component-status")?.classList.toggle("is-fragmented", fragmented);
  const connectivitySymbol = $("#scenario-component-status")?.querySelector(":scope > span");
  if (connectivitySymbol) connectivitySymbol.textContent = fragmented ? "!" : networkAvailable ? "✓" : "–";
  $("#scenario-component-copy").textContent = !networkAvailable
    ? "Network connectivity unavailable without From / To nodes"
    : fragmented
      ? `Network fragmented: ${baselineComponents} → ${scenarioComponents} connected parts`
      : `Network remains connected: ${scenarioComponents} connected part${Number(scenarioComponents) === 1 ? "" : "s"}`;
  $("#scenario-affected-count").textContent = formatNumber(result.affected_count || scenario.changes?.length || 0);
  $("#scenario-result-mix").innerHTML = Object.entries(counts)
    .filter(([, count]) => Number(count) > 0)
    .map(([effect, count]) => `<span class="${escapeHtml(effect)}">${formatNumber(count)} ${escapeHtml(SCENARIO_EFFECT_LABELS[effect] || effect)}</span>`)
    .join("");
  const unavailableNetworkValue = networkAvailable ? "Not returned by this model tier" : "Not calculated: topology unavailable";
  const unavailableRouteValue = networkAvailable ? "Not returned: no comparable route set" : "Not calculated: topology unavailable";
  const exactValues = [
    ["Baseline time", result.timestamp_label || scenario.base_timestamp || "Static snapshot"],
    ["Impedance basis", evidence.impedance_label || "Not available"],
    ["Largest part retained", networkAvailable ? `${Number(visual.largest_component_retained_percent || 0).toFixed(1)}%` : "Not calculated: topology unavailable"],
    ["Calculation", evidence.efficiency_exact === true ? "Exact" : evidence.efficiency_exact === false ? "Deterministic sampled sources" : "Location mapping only"],
  ];
  $("#scenario-evidence-body").innerHTML = `<div class="scenario-evidence-grid">${exactValues.map(([label, value]) => `<div><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join("")}</div><p>${escapeHtml(result.data_boundary || "No unobserved consequence is inferred.")}</p><p>${escapeHtml(evidence.communication_rule || "Every visible change was explicitly selected by the user.")}</p>`;
  panel.classList.remove("is-hidden");
  renderMapLegend();
}

function renderScenarioImpact(scenario) {
  const panel = $("#scenario-impact-panel");
  if (!scenario) {
    stopMatsimPlayback();
    state.map?.setMatsimFeatures(null);
    panel.classList.add("is-hidden");
    return;
  }
  const result = scenario.result || {};
  const visual = result.visual || {};
  const evidence = result.evidence || {};
  const assignment = evidence.assignment || {};
  const counts = result.change_counts || {};
  const networkAvailable = ["network_compare", "propagated_network_compare"].includes(visual.mode);
  const retained = networkAvailable ? Number(visual.scenario_efficiency_percent || 0) : null;
  const initialCount = Number(result.initial_disruption_count || result.affected_count || scenario.changes?.length || 0);
  const indirectCount = Number(result.indirect_impact_count || 0);
  const reroutedCount = Number(assignment.rerouted_pair_count || 0);
  const disconnectedCount = Number(assignment.disconnected_pair_count || 0);
  const testedCount = Number(assignment.comparable_pair_count || 0);
  const assignmentTier = String(assignment.tier || "");
  const observedAssignment = assignmentTier === "observed_flow_calibrated_capacity_ue_proxy";
  $("#scenario-result-title").textContent = scenario.name;
  const plainFinding = result.plain_finding || "The saved scenario is ready to inspect.";
  const activeEffects = Object.entries(counts).filter(([, value]) => Number(value) > 0);
  let initialStatement = `Changes were applied to ${formatNumber(initialCount)} selected roads`;
  if (activeEffects.length === 1) {
    const effect = activeEffects[0][0];
    if (effect === "closed") initialStatement = `${formatNumber(initialCount)} selected road${initialCount === 1 ? " was" : "s were"} closed`;
    else if (effect === "slower") initialStatement = `${formatNumber(initialCount)} selected road${initialCount === 1 ? " was" : "s were"} slowed`;
    else if (effect === "longer_time") initialStatement = `Travel time increased on ${formatNumber(initialCount)} selected road${initialCount === 1 ? "" : "s"}`;
  }
  let scenarioSummary = `${initialStatement}.`;
  if (!networkAvailable) {
    scenarioSummary += " Other network effects could not be calculated because From/To link topology is unavailable.";
  } else if (disconnectedCount > 0 && indirectCount > 0) {
    scenarioSummary += ` ${formatNumber(disconnectedCount)} of ${formatNumber(testedCount)} tested routes disconnected; ${formatNumber(indirectCount)} other road${indirectCount === 1 ? "" : "s"} may carry more traffic.`;
  } else if (disconnectedCount > 0) {
    scenarioSummary += ` ${formatNumber(disconnectedCount)} of ${formatNumber(testedCount)} tested routes disconnected, and no successful rerouting was found.`;
  } else if (indirectCount > 0) {
    scenarioSummary += ` Traffic shifted onto ${formatNumber(indirectCount)} other road${indirectCount === 1 ? "" : "s"}.`;
  } else {
    scenarioSummary += " No tested route disconnected and no additional road loading was detected.";
  }
  $("#scenario-plain-finding").textContent = scenarioSummary;
  const impactLabels = {
    network_split: "Network split",
    large_impedance: "Large movement loss",
    noticeable_impedance: "Noticeable movement loss",
    limited_topology_change: "Limited whole-network change",
    mapped_locations: "Mapped locations only",
    proxy_od_disconnection: "Routes disconnected",
    rerouting_pressure: "Rerouting pressure",
    network_impedance: "Movement impedance",
    limited_propagation: "Limited propagation",
  };
  $("#scenario-impact-chip").textContent = !networkAvailable
    ? "Locations only"
    : observedAssignment ? "Flow/capacity assignment" : "Topology proxy · limited";
  $("#scenario-impact-chip").classList.toggle("is-proxy", networkAvailable && !observedAssignment);
  $("#scenario-finding-symbol").textContent = ["network_split", "proxy_od_disconnection"].includes(result.impact_level) ? "↯" : networkAvailable ? "↗" : "•";
  $("#scenario-retained-bar").style.width = networkAvailable ? `${Math.max(0, Math.min(100, retained))}%` : "0%";
  $("#scenario-retained-value").textContent = networkAvailable ? `${retained.toFixed(1)}%` : "Not calculated";
  $("#scenario-retained-value").title = evidence.efficiency_exact === false ? "Deterministic estimate from fixed source nodes" : "Network calculation";
  const baselineComponents = visual.baseline_components;
  const scenarioComponents = visual.scenario_components;
  renderComponentNetwork($("#baseline-component-visual"), baselineComponents, false);
  renderComponentNetwork(
    $("#scenario-component-visual"),
    scenarioComponents,
    networkAvailable && Number(scenarioComponents) > Number(baselineComponents),
  );
  $("#scenario-component-copy").textContent = networkAvailable
    ? `${baselineComponents} → ${scenarioComponents} connected parts`
    : "Network result unavailable without From / To nodes";
  $("#scenario-affected-count").textContent = formatNumber(initialCount);
  $("#scenario-initial-count").textContent = `${formatNumber(initialCount)} selected road${initialCount === 1 ? "" : "s"}`;
  const indirectLabel = $("#scenario-indirect-label");
  if (!networkAvailable) {
    indirectLabel.textContent = "Network result unavailable";
    $("#scenario-indirect-count").textContent = "Needs link topology";
  } else if (disconnectedCount > 0 && indirectCount > 0) {
    indirectLabel.textContent = "Disconnection and traffic shift";
    $("#scenario-indirect-count").textContent = `${formatNumber(disconnectedCount)} routes disconnected · ${formatNumber(indirectCount)} other road${indirectCount === 1 ? "" : "s"} may carry more traffic`;
  } else if (disconnectedCount > 0) {
    indirectLabel.textContent = "Routes disconnected";
    $("#scenario-indirect-count").textContent = `${formatNumber(disconnectedCount)} of ${formatNumber(testedCount)} tested routes disconnected`;
  } else if (indirectCount > 0) {
    indirectLabel.textContent = "Effects on other roads";
    $("#scenario-indirect-count").textContent = `${formatNumber(indirectCount)} other road${indirectCount === 1 ? "" : "s"} may carry more modelled traffic`;
  } else {
    indirectLabel.textContent = "Network response";
    $("#scenario-indirect-count").textContent = "No disconnection or additional road loading detected";
  }
  $("#scenario-assignment-label").textContent = "network response →";
  const truth = $("#scenario-model-truth");
  truth.classList.toggle("is-observed", observedAssignment);
  $("#scenario-model-tier").textContent = !networkAvailable ? "Location mapping" : observedAssignment ? "Flow/capacity tier" : "Topology proxy tier";
  const tierSummary = !networkAvailable
    ? "Only selected locations are shown because link topology is unavailable."
    : observedAssignment
      ? "Uploaded flow and capacity were used to estimate traffic redistribution."
      : "Topology and travel impedance were used; traffic changes are modelled, not observed.";
  const tierDetail = !networkAvailable
    ? "From/To topology is unavailable. The platform therefore shows only the selected locations and does not infer propagation."
    : observedAssignment
      ? `The capacity-sensitive assignment tested ${formatNumber(testedCount)} fixed proxy origin-destination relations: ${formatNumber(reroutedCount)} rerouted and ${formatNumber(disconnectedCount)} disconnected. It is not a calibrated demand forecast unless an independently calibrated OD matrix is uploaded.`
      : `The deterministic shortest-route proxy tested ${formatNumber(testedCount)} baseline-reachable relations: ${formatNumber(disconnectedCount)} disconnected, ${formatNumber(reroutedCount)} rerouted and ${formatNumber(indirectCount)} external roads carried successful detours. These are modelled route checks, not observed traffic volumes.`;
  $("#scenario-model-explanation").textContent = tierSummary;
  $("#scenario-result-mix").innerHTML = Object.entries(counts)
    .filter(([, count]) => Number(count) > 0)
    .map(([effect, count]) => `<span class="${escapeHtml(effect)}">${formatNumber(count)} ${escapeHtml(SCENARIO_EFFECT_LABELS[effect] || effect)}</span>`)
    .join("");
  const exactValues = [
    ["Baseline time", result.timestamp_label || scenario.base_timestamp || "Static snapshot"],
    ["Impedance basis", evidence.impedance_label || "Not available"],
    ["Assignment tier", assignment.tier?.replaceAll("_", " ") || (networkAvailable ? "Topology only" : "Location mapping")],
    ["Initial roads route-tested", assignment.covered_seed_count !== undefined ? `${formatNumber(assignment.covered_seed_count)} / ${formatNumber(assignment.selected_seed_count || initialCount)}` : unavailableNetworkValue],
    ["Routes tested", assignment.comparable_pair_count !== undefined ? formatNumber(assignment.comparable_pair_count) : unavailableRouteValue],
    ["Routes rerouted", assignment.rerouted_pair_count !== undefined ? `${formatNumber(assignment.rerouted_pair_count)} (${Number(assignment.rerouted_pair_percent || 0).toFixed(1)}%)` : unavailableRouteValue],
    ["Routes disconnected", assignment.disconnected_pair_count !== undefined ? `${formatNumber(assignment.disconnected_pair_count)} (${Number(assignment.disconnected_pair_percent || 0).toFixed(1)}%)` : unavailableRouteValue],
    ["Mean connected detour", assignment.mean_detour_percent !== undefined ? `${Number(assignment.mean_detour_percent).toFixed(1)}%` : unavailableRouteValue],
    ["Calculation", result.randomness === "none" ? "Deterministic · no random draw" : "Location mapping only"],
  ];
  $("#scenario-evidence-body").innerHTML = `<div class="scenario-evidence-grid">${exactValues.map(([label, value]) => `<div><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join("")}</div><p><strong>Finding:</strong> ${escapeHtml(plainFinding)}</p><p><strong>Model tier:</strong> ${escapeHtml(tierDetail)}</p><p><strong>Boundary:</strong> ${escapeHtml(result.data_boundary || "No unobserved consequence is inferred.")}</p>`;
  $("#focus-scenario").textContent = indirectCount ? "Show initial + indirect roads" : "Show affected roads";
  renderMatsimResult(result.matsim || null);
  panel.classList.remove("is-hidden");
}

function stopMatsimPlayback({ resetTransition = false } = {}) {
  state.matsimPlaying = false;
  if (state.matsimPlayTimer !== null) window.cancelAnimationFrame(state.matsimPlayTimer);
  state.matsimPlayTimer = null;
  state.matsimTransitionStartedAt = null;
  state.matsimLastRenderAt = 0;
  if (resetTransition) state.matsimTransitionProgress = 0;
  const button = $("#play-matsim");
  if (button) {
    button.classList.remove("is-playing");
    button.innerHTML = "&#9654;";
    button.setAttribute("aria-label", "Play continuous MATSim vehicle movement");
    button.title = "Play continuous smoothed vehicle movement";
  }
}

function matsimPlaybackCollection() {
  return {
    type: "FeatureCollection",
    features: (state.matsimFrames || []).flatMap((frame) => frame.feature_collection?.features || []),
  };
}

function focusMatsimPlaybackOnMap() {
  const mapStage = $("#disaster-map");
  const collection = matsimPlaybackCollection();
  mapStage?.scrollIntoView({ behavior: "smooth", block: "center" });
  const fitPlayback = () => {
    state.map?.resize();
    state.map?.focusCollection(collection);
  };
  window.setTimeout(fitPlayback, 320);
  window.setTimeout(fitPlayback, 760);
}

function setMatsimFrame(index) {
  const frames = state.matsimFrames || [];
  if (!frames.length) {
    state.map?.setMatsimFeatures(null);
    return;
  }
  state.matsimFrameIndex = Math.max(0, Math.min(frames.length - 1, Number(index) || 0));
  state.matsimTransitionProgress = 0;
  state.matsimTransitionStartedAt = null;
  const frame = frames[state.matsimFrameIndex];
  $("#matsim-slider").value = String(state.matsimFrameIndex);
  $("#matsim-time-label").textContent = frame.time_label || matsimClockLabel(frame.time_seconds);
  state.map?.setMatsimFeatures(frame.feature_collection || null);
  renderMapLegend();
}

function matsimClockLabel(timeSeconds) {
  const value = Number(timeSeconds);
  if (!Number.isFinite(value)) return "Simulated time";
  const seconds = ((Math.round(value) % 86400) + 86400) % 86400;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  return [hours, minutes, remainder].map((part) => String(part).padStart(2, "0")).join(":");
}

function matsimAgentKey(feature, index) {
  const properties = feature?.properties || {};
  const raw = properties.person_id ?? properties.agent_id ?? properties.vehicle_id ?? properties.key ?? properties.id;
  return raw === undefined || raw === null ? `vehicle-${index}` : String(raw).split("@")[0];
}

function matsimVehicleFeature(feature, coordinates, opacity, suffix = "") {
  return {
    ...feature,
    geometry: { ...feature.geometry, coordinates },
    properties: {
      ...(feature.properties || {}),
      playback_opacity: Math.max(0, Math.min(1, opacity)),
      playback_instance: `${matsimAgentKey(feature, 0)}${suffix}`,
    },
  };
}

function matsimTransitionCollection(currentFrame, nextFrame, progress, wrap = false) {
  const currentFeatures = currentFrame?.feature_collection?.features || [];
  const nextFeatures = nextFrame?.feature_collection?.features || [];
  const staticFeatures = (progress < .5 ? currentFeatures : nextFeatures)
    .filter((feature) => feature.geometry?.type !== "Point");
  const currentVehicles = new Map();
  const nextVehicles = new Map();
  currentFeatures.filter((feature) => feature.geometry?.type === "Point")
    .forEach((feature, index) => currentVehicles.set(matsimAgentKey(feature, index), feature));
  nextFeatures.filter((feature) => feature.geometry?.type === "Point")
    .forEach((feature, index) => nextVehicles.set(matsimAgentKey(feature, index), feature));
  const vehicles = [];

  if (wrap) {
    currentVehicles.forEach((feature) => vehicles.push(matsimVehicleFeature(
      feature, feature.geometry.coordinates, 1 - progress, "-loop-out",
    )));
    nextVehicles.forEach((feature) => vehicles.push(matsimVehicleFeature(
      feature, feature.geometry.coordinates, progress, "-loop-in",
    )));
  } else {
    const keys = new Set([...currentVehicles.keys(), ...nextVehicles.keys()]);
    keys.forEach((key) => {
      const current = currentVehicles.get(key);
      const next = nextVehicles.get(key);
      if (current && next) {
        const from = current.geometry.coordinates;
        const to = next.geometry.coordinates;
        const coordinates = [
          Number(from[0]) + (Number(to[0]) - Number(from[0])) * progress,
          Number(from[1]) + (Number(to[1]) - Number(from[1])) * progress,
        ];
        vehicles.push(matsimVehicleFeature(next, coordinates, 1));
      } else if (current) {
        vehicles.push(matsimVehicleFeature(current, current.geometry.coordinates, 1 - progress, "-out"));
      } else if (next) {
        vehicles.push(matsimVehicleFeature(next, next.geometry.coordinates, progress, "-in"));
      }
    });
  }

  return { type: "FeatureCollection", features: [...staticFeatures, ...vehicles] };
}

function matsimTransitionTimeLabel(currentFrame, nextFrame, progress, wrap) {
  const currentTime = Number(currentFrame?.time_seconds);
  const nextTime = Number(nextFrame?.time_seconds);
  if (wrap || !Number.isFinite(currentTime) || !Number.isFinite(nextTime)) {
    return progress < .5
      ? (currentFrame?.time_label || matsimClockLabel(currentTime))
      : (nextFrame?.time_label || matsimClockLabel(nextTime));
  }
  return matsimClockLabel(currentTime + (nextTime - currentTime) * progress);
}

function advanceMatsimPlayback(timestamp) {
  if (!state.matsimPlaying) return;
  const frames = state.matsimFrames || [];
  if (frames.length < 2) return stopMatsimPlayback({ resetTransition: true });
  if (state.matsimTransitionStartedAt === null) {
    state.matsimTransitionStartedAt = timestamp - state.matsimTransitionProgress * MATSIM_FRAME_DURATION_MS;
  }
  const progress = Math.max(0, Math.min(1,
    (timestamp - state.matsimTransitionStartedAt) / MATSIM_FRAME_DURATION_MS,
  ));
  const nextIndex = (state.matsimFrameIndex + 1) % frames.length;
  const wrap = nextIndex === 0;
  if (timestamp - state.matsimLastRenderAt >= MATSIM_RENDER_INTERVAL_MS || progress >= 1) {
    state.map?.setMatsimFeatures(matsimTransitionCollection(
      frames[state.matsimFrameIndex], frames[nextIndex], progress, wrap,
    ));
    $("#matsim-time-label").textContent = matsimTransitionTimeLabel(
      frames[state.matsimFrameIndex], frames[nextIndex], progress, wrap,
    );
    state.matsimLastRenderAt = timestamp;
  }
  if (progress >= 1) {
    state.matsimFrameIndex = nextIndex;
    state.matsimTransitionProgress = 0;
    state.matsimTransitionStartedAt = timestamp;
    $("#matsim-slider").value = String(nextIndex);
    renderMapLegend();
  } else {
    state.matsimTransitionProgress = progress;
  }
  state.matsimPlayTimer = window.requestAnimationFrame(advanceMatsimPlayback);
}

function toggleMatsimPlayback() {
  if (!state.matsimFrames.length) return;
  if (state.matsimPlaying) return stopMatsimPlayback();
  focusMatsimPlaybackOnMap();
  state.matsimPlaying = true;
  $("#play-matsim").classList.add("is-playing");
  $("#play-matsim").innerHTML = "&#10074;&#10074;";
  $("#play-matsim").setAttribute("aria-label", "Pause MATSim vehicle movement");
  $("#play-matsim").title = "Pause playback";
  state.matsimTransitionStartedAt = window.performance.now()
    - state.matsimTransitionProgress * MATSIM_FRAME_DURATION_MS;
  state.matsimPlayTimer = window.requestAnimationFrame(advanceMatsimPlayback);
}

function matsimTierCopy(readiness = state.matsimReadiness) {
  if (!readiness) return "Open a project to check MATSim readiness.";
  if (readiness.tier === "demand_calibrated") {
    return `Full demand run: uploaded OD demand and complete positive length, impedance and capacity coverage are available across ${formatNumber(readiness.topology_links)} routable links.`;
  }
  if (readiness.tier === "exploratory_probe") {
    return `Exploratory run: ${formatNumber(readiness.topology_links)} links are ready. Fixed probe trips test rerouting; they do not represent observed demand.`;
  }
  return readiness.explanation || "This project needs routable From/To links plus positive length and speed or travel time.";
}

function renderMatsimReadiness() {
  const readiness = state.matsimReadiness;
  const canRun = Boolean(readiness?.can_run);
  const compact = $("#matsim-builder-readiness");
  if (compact) {
    compact.className = `matsim-readiness compact ${canRun ? "is-ready" : "is-limited"}`;
    compact.innerHTML = `<strong>${escapeHtml(readiness?.label || "MATSim readiness")}</strong><span>${escapeHtml(matsimTierCopy(readiness))}</span>`;
  }
  const button = $("#scenario-engine-matsim");
  if (button) {
    button.disabled = !canRun;
    button.title = canRun ? matsimTierCopy(readiness) : readiness?.explanation || "MATSim is not ready for this project.";
  }
  const run = $("#run-matsim-scenario");
  const selectedAgents = Number($("#matsim-probe-agent-count")?.value || 0);
  const hasAgentCount = [50, 100, 150, 200].includes(selectedAgents);
  if (run) run.disabled = !canRun || !activeScenario() || Boolean(state.matsimJobId) || !hasAgentCount;
  const probeField = $("#matsim-probe-agent-field");
  if (probeField) probeField.classList.remove("is-hidden");
  if ($("#matsim-probe-agent-count")) $("#matsim-probe-agent-count").disabled = Boolean(state.matsimJobId);
  if ($("#matsim-ready-summary")) $("#matsim-ready-summary").textContent = matsimTierCopy(readiness);
}

async function refreshMatsimReadiness() {
  if (!state.project) return;
  state.matsimReadiness = await api.matsimReadiness(state.project.project_id).catch((error) => ({
    status: "unavailable",
    tier: "unavailable",
    can_run: false,
    label: "MATSim readiness unavailable",
    explanation: error.message,
  }));
  renderMatsimReadiness();
}

async function prepareMatsimReadinessForCopilot() {
  if (!state.project) return false;

  // Readiness can be stale when a point-based upload contains valid From/To
  // fields but its derived link view has not been built yet. Copilot owns this
  // preparation step: a runnable upload must not be rejected merely because
  // the user has not manually pressed "Mapped links" first.
  await refreshMatsimReadiness();
  if (state.matsimReadiness?.can_run) return true;

  if (state.networkMapping?.available && !state.networkMapping?.ready) {
    updateAssistantActivity("decision", "Preparing the uploaded routable link network for MATSim", "active");
    state.networkMapping = await api.buildNetworkMapping(state.project.project_id);
    configureNetworkControls();
    await refreshMatsimReadiness();
    updateAssistantActivity(
      "decision",
      state.matsimReadiness?.can_run
        ? "Routable link network confirmed for MATSim"
        : "The uploaded data still cannot form a routable MATSim network",
      state.matsimReadiness?.can_run ? "complete" : "failed",
    );
  }

  return Boolean(state.matsimReadiness?.can_run);
}

function setScenarioEngine(engine) {
  const requested = engine === "matsim" && state.matsimReadiness?.can_run ? "matsim" : "network";
  state.scenarioEngine = requested;
  [["network", "#scenario-engine-network"], ["matsim", "#scenario-engine-matsim"]].forEach(([value, selector]) => {
    const button = $(selector);
    if (!button) return;
    button.classList.toggle("is-active", value === requested);
    button.setAttribute("aria-pressed", String(value === requested));
  });
  const save = $("#save-scenario");
  if (save) save.textContent = requested === "matsim" ? "Save scenario for MATSim" : "Save and see impact";
  renderMatsimReadiness();
}

function matsimComparisonValue(value, unit = "") {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "Not available";
  return `${formatNumber(Number(value), Number(value) % 1 ? 2 : 0)}${unit}`;
}

function renderMatsimReport(result) {
  const outcomes = result.route_outcomes || {};
  const agentCount = Math.max(1, Number(outcomes.agent_count || result.manifest?.agent_count || 0));
  const noRoute = Number(result.manifest?.pre_simulation_disconnected_count || 0);
  const categories = [
    ["Same route", Number(outcomes.same_route || 0), "#267a74"],
    ["Rerouted", Number(outcomes.rerouted || 0), "#ef8d22"],
    ["Stuck", Number(outcomes.stuck_or_aborted || 0), "#b12f4e"],
    ["No feasible route", noRoute, "#7f1734"],
    ["Not completed", Number(outcomes.unfinished || 0) + Math.max(0, Number(outcomes.unstarted || 0) - noRoute), "#765b78"],
    ["Trace limited", Number(outcomes.completed_unclassified || 0), "#aaa184"],
  ].filter(([, count]) => count > 0);
  $("#matsim-outcome-bar").innerHTML = categories.map(([, count, colour]) => (
    `<i style="width:${Math.max(0, count / agentCount * 100)}%;--outcome-colour:${colour}" aria-hidden="true"></i>`
  )).join("") || '<i style="width:100%;--outcome-colour:#d8cfb7" aria-hidden="true"></i>';
  $("#matsim-outcome-legend").innerHTML = categories.map(([label, count, colour]) => (
    `<span style="--outcome-colour:${colour}"><i aria-hidden="true"></i>${escapeHtml(label)} · ${formatNumber(count)}</span>`
  )).join("") || "<span>No agent outcome was returned.</span>";

  const baseline = result.baseline || {};
  const scenario = result.scenario || {};
  const comparisons = [
    ["Completed trips", baseline.completed_trips, scenario.completed_trips, ""],
    ["Mean trip time", baseline.mean_trip_time_minutes, scenario.mean_trip_time_minutes, " min"],
    ["Probe road entries", baseline.link_entries, scenario.link_entries, ""],
    ["Stuck / aborted", baseline.stuck, scenario.stuck, ""],
  ];
  $("#matsim-comparison").innerHTML = comparisons.map(([label, before, after, unit]) => (
    `<div><small>${escapeHtml(label)}</small><b>${escapeHtml(matsimComparisonValue(before, unit))}</b><em aria-hidden="true">→</em><b>${escapeHtml(matsimComparisonValue(after, unit))}</b></div>`
  )).join("");

  const frames = result.animation_frames || [];
  if (frames.length) {
    const width = 320;
    const height = 64;
    const maximum = Math.max(1, ...frames.map((frame) => Number(frame.vehicle_entries || 0)));
    const points = frames.map((frame, index) => {
      const x = frames.length === 1 ? width / 2 : 8 + index / (frames.length - 1) * (width - 16);
      const y = 6 + (1 - Number(frame.vehicle_entries || 0) / maximum) * (height - 12);
      return { x, y, frame };
    });
    const peakIndex = Math.max(0, Math.min(points.length - 1, Number(result.peak_impact_frame_index || 0)));
    const peak = points[peakIndex];
    const line = points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
    const area = `8,${height} ${line} ${width - 8},${height}`;
    $("#matsim-timeline").innerHTML = `<svg viewBox="0 0 ${width} 78" role="img" aria-label="Probe road entries over simulated time"><line class="timeline-axis" x1="8" y1="${height}" x2="${width - 8}" y2="${height}"></line><polygon class="timeline-area" points="${area}"></polygon><polyline class="timeline-line" points="${line}"></polyline><circle class="timeline-peak" cx="${peak.x.toFixed(1)}" cy="${peak.y.toFixed(1)}" r="4"></circle></svg><footer><span>${escapeHtml(frames[0].time_label || "Start")}</span><strong>Peak · ${escapeHtml(peak.frame.time_label || "")}</strong><span>${escapeHtml(frames.at(-1).time_label || "End")}</span></footer>`;
  } else {
    $("#matsim-timeline").innerHTML = '<div class="empty-state">No event timeline was returned.</div>';
  }

  const shifts = result.top_link_shifts || [];
  $("#matsim-link-shifts").innerHTML = shifts.length
    ? shifts.slice(0, 6).map((item, index) => {
      const relation = item.relation === "changed" ? "Selected change" : item.relation === "nearby" ? "Near disruption" : "Other road";
      return `<button type="button" data-matsim-shift-index="${index}"><strong>${escapeHtml(item.label || item.key)}</strong><b>+${formatNumber(item.additional_entries)} entries</b><small>${escapeHtml(relation)} · ${formatNumber(item.baseline_entries)} → ${formatNumber(item.scenario_entries)}</small></button>`;
    }).join("")
    : '<div class="empty-state">No road received additional probe entries in this run.</div>';
  const snapshot = result.peak_impact_snapshot || {};
  $("#matsim-report-tier").textContent = result.tier === "demand_calibrated" ? "Uploaded-demand simulation result" : "Exploratory probe simulation result";
  $("#matsim-report-snapshot").textContent = snapshot.time_label
    ? `Peak impact · ${snapshot.time_label}`
    : "Peak-impact snapshot";
}

function renderMatsimResult(result) {
  stopMatsimPlayback({ resetTransition: true });
  const card = $("#matsim-simulation-card");
  const output = $("#matsim-result");
  const progress = $("#matsim-progress");
  if (!card || !output) return;
  card.classList.remove("is-hidden");
  progress?.classList.toggle("is-hidden", !state.matsimJobId);
  if (!result) {
    state.matsimFrames = [];
    state.matsimResult = null;
    state.map?.setMatsimFeatures(null);
    output.classList.add("is-hidden");
    $("#matsim-status").textContent = state.matsimReadiness?.can_run ? "Optional MATSim run available" : "MATSim needs richer network data";
    renderMatsimReadiness();
    return;
  }
  const baseline = result.baseline || {};
  const scenario = result.scenario || {};
  const noRoute = Number(result.manifest?.pre_simulation_disconnected_count || 0);
  const change = Number(result.mean_trip_time_change_percent);
  const cards = [
    ["Agents tested", formatNumber(result.manifest?.agent_count || scenario.departures || 0), result.tier === "demand_calibrated" ? "Uploaded OD demand" : "Deterministic probe trips"],
    ["Trips completed", formatNumber(scenario.completed_trips || 0), noRoute ? `${formatNumber(noRoute)} without a feasible route` : `${formatNumber(scenario.stuck || 0)} stuck in disruption`],
    ["Mean trip time", Number.isFinite(change) ? `${change >= 0 ? "+" : ""}${change.toFixed(1)}%` : "Not comparable", `${baseline.mean_trip_time_minutes ?? "–"} to ${scenario.mean_trip_time_minutes ?? "–"} min`],
    ["Road entries", formatNumber(scenario.link_entries || 0), `${formatNumber(result.frame_count || 0)} playback frames`],
  ];
  $("#matsim-metrics").innerHTML = cards.map(([label, value, detail]) => `<article><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong><span>${escapeHtml(detail)}</span></article>`).join("");
  $("#matsim-status").textContent = result.plain_finding || "MATSim comparison complete";
  $("#matsim-boundary").textContent = result.tier === "demand_calibrated"
    ? "Full demand tier: results use uploaded OD demand and road capacity. This does not claim field calibration unless the uploaded inputs were independently calibrated."
    : "Exploratory tier: animation shows fixed probe-agent rerouting and disconnection, not observed traffic volume, population impact or evacuation demand.";
  state.matsimResult = result;
  state.matsimFrames = result.animation_frames || [];
  state.matsimFrameIndex = 0;
  $("#matsim-slider").max = String(Math.max(0, state.matsimFrames.length - 1));
  $("#matsim-slider").disabled = state.matsimFrames.length <= 1;
  $("#play-matsim").disabled = state.matsimFrames.length <= 1;
  output.classList.remove("is-hidden");
  if (state.matsimFrames.length) setMatsimFrame(0);
  renderMatsimReport(result);
  renderMatsimReadiness();
}

function updateMatsimProgress(value, message) {
  const progress = Math.max(1, Math.min(100, Math.round(Number(value) || 1)));
  $("#matsim-progress-bar").style.width = `${progress}%`;
  $("#matsim-progress-value").textContent = `${progress}%`;
  $("#matsim-progress-copy").textContent = message || "MATSim is running...";
}

async function animateMatsimProgress(fromValue, toValue, message, jobId) {
  let current = Math.max(1, Math.min(100, Math.round(Number(fromValue) || 1)));
  const target = Math.max(current, Math.min(100, Math.round(Number(toValue) || current)));
  updateMatsimProgress(current, message);
  while (current < target && state.matsimJobId === jobId) {
    current += 1;
    updateMatsimProgress(current, message);
    await new Promise((resolve) => window.setTimeout(resolve, 18));
  }
  return current;
}

function focusMatsimRunProgress() {
  switchView("explore");
  const card = $("#matsim-simulation-card");
  const progress = $("#matsim-progress");
  if (!card || !progress) return;
  progress.classList.add("is-run-focus");
  window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }));
}

async function runActiveScenarioMatsim({ focusProgress = false, maxAgentsOverride = null, uploadedDemand = false } = {}) {
  const scenario = activeScenario();
  if (!scenario) return toast("Save a scenario first", "MATSim compares one saved disruption with its matching baseline.", true);
  if (!state.matsimReadiness?.can_run) return toast("MATSim is not ready", state.matsimReadiness?.explanation || "This project needs a routable network.", true);
  stopMatsimPlayback();
  if (state.networkMapping?.ready && state.mapView !== "links") await switchMapView("links");
  const progressPanel = $("#matsim-progress");
  progressPanel.classList.remove("is-hidden");
  if (focusProgress) focusMatsimRunProgress();
  $("#matsim-result").classList.add("is-hidden");
  $("#run-matsim-scenario").disabled = true;
  $("#matsim-status").textContent = "Running paired baseline and disruption simulations";
  let completedResult = null;
  try {
    const probeAgents = Number($("#matsim-probe-agent-count")?.value || 0);
    const overrideAgents = Number(maxAgentsOverride);
    const hasOverride = maxAgentsOverride !== null && Number.isFinite(overrideAgents);
    if (!hasOverride && ![50, 100, 150, 200].includes(probeAgents)) {
      progressPanel.classList.add("is-hidden");
      renderMatsimReadiness();
      return toast("Choose the agent count", "Select 50, 100, 150 or 200 agents, then press Run MATSim.", true);
    }
    if (hasOverride && (overrideAgents < 20 || overrideAgents > 20000)) {
      progressPanel.classList.add("is-hidden");
      renderMatsimReadiness();
      return toast("MATSim demand limit is invalid", "Use an automatic demand limit between 20 and 20,000 agents.", true);
    }
    const maxAgents = hasOverride ? Math.round(overrideAgents) : probeAgents;
    let displayedProgress = 1;
    updateMatsimProgress(
      displayedProgress,
      uploadedDemand
        ? "Preparing agents from the uploaded OD demand..."
        : `Starting ${formatNumber(maxAgents)} agents...`,
    );
    const started = await api.runScenarioMatsim(scenario.scenario_id, { iterations: 3, max_agents: maxAgents });
    state.matsimJobId = started.job_id;
    while (state.matsimJobId === started.job_id) {
      const job = await api.job(started.job_id);
      const backendProgress = Math.max(0, Math.min(100, Number(job.progress || 0)));
      const targetProgress = job.status === "completed"
        ? 100
        : Math.min(96, Math.max(displayedProgress + 5, backendProgress));
      displayedProgress = await animateMatsimProgress(
        displayedProgress,
        targetProgress,
        job.message || "MATSim is running...",
        started.job_id,
      );
      if (job.status === "completed") {
        state.matsimJobId = null;
        const refreshed = await api.scenario(scenario.scenario_id);
        state.scenarios = state.scenarios.map((item) => item.scenario_id === refreshed.scenario_id ? refreshed : item);
        renderScenarioControls();
        activateScenario(refreshed.scenario_id);
        completedResult = job.result || refreshed.result?.matsim || null;
        renderMatsimResult(completedResult);
        toast("MATSim simulation ready", "Baseline and disruption used the same agents and seed; map playback is ready.");
        break;
      }
      if (job.status === "failed") throw new Error(job.error || job.message || "MATSim simulation failed.");
      if (job.status === "cancelled") throw new Error("MATSim simulation was stopped.");
      await new Promise((resolve) => window.setTimeout(resolve, 850));
    }
  } catch (error) {
    state.matsimJobId = null;
    $("#matsim-status").textContent = "MATSim did not complete";
    toast("MATSim did not complete", error.message, true);
  } finally {
    progressPanel.classList.remove("is-run-focus");
    progressPanel.classList.add("is-hidden");
    renderMatsimReadiness();
  }
  return completedResult;
}

function activateScenario(scenarioId, { focus = false } = {}) {
  // Any immediate activation supersedes an older asynchronous map refresh.
  state.scenarioRequestId += 1;
  state.activeScenarioId = scenarioId || null;
  const scenario = activeScenario();
  $("#scenario-select").value = scenario?.scenario_id || "";
  $("#edit-scenario").disabled = !scenario || state.scenarioBuilding;
  if (!scenario) {
    $("#scenario-active-title").textContent = "Baseline · no added disruption";
    $("#scenario-active-copy").textContent = "Build a clear what-if change from the uploaded network.";
    state.map?.setScenarioFeatures(null);
    renderScenarioImpact(null);
    setHazardAtmosphere(state.project?.hazard_type || "other");
    renderMapLegend();
    if (state.project) {
      setMapBanner(
        state.mapView === "links" ? "Mapped link view" : "Historical data",
        `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`,
      );
    }
    return;
  }
  const result = scenario.result || {};
  const scenarioCollection = {
    type: "FeatureCollection",
    features: (result.feature_collection?.features || []).map((feature) => ({
      ...feature,
      properties: {
        ...(feature.properties || {}),
        impact_role: feature.properties?.impact_role || (feature.properties?.initial_disruption === false ? "rerouting_pressure" : "initial_disruption"),
      },
    })),
  };
  $("#scenario-active-title").textContent = scenario.name;
  $("#scenario-active-copy").textContent = `${formatNumber(result.initial_disruption_count || result.affected_count || scenario.changes?.length || 0)} initial roads · ${formatNumber(result.indirect_impact_count || 0)} indirect roads · ${result.timestamp_label || "selected baseline"}`;
  state.map?.setScenarioFeatures(scenarioCollection);
  renderScenarioImpact(scenario);
  setHazardAtmosphere(scenario.hazard_type);
  renderMapLegend();
  setMapBanner(
    "Saved disruption",
    `${formatNumber(result.initial_disruption_count || result.affected_count || scenario.changes?.length || 0)} initial roads · ${formatNumber(result.indirect_impact_count || 0)} indirect roads · ${result.timestamp_label || "selected baseline"}`,
    true,
  );
  if (focus) {
    state.map?.focusCollection(scenarioCollection);
    $("#map-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function selectScenario(scenarioId, { focus = true } = {}) {
  const requestId = ++state.scenarioRequestId;
  const requestedId = scenarioId || null;
  const scenario = state.scenarios.find((item) => item.scenario_id === requestedId) || null;
  invalidateAnalysisReview("The active disruption scenario changed.");

  let mapRefreshed = false;
  if (scenario) {
    const requestedTime = Number(scenario.base_timestamp);
    const requestedIndex = state.timeValues.indexOf(requestedTime);
    if (requestedIndex >= 0 && requestedIndex !== state.timeIndex) {
      stopPlayback();
      state.timeIndex = requestedIndex;
      $("#time-slider").value = String(requestedIndex);
      $("#time-label").textContent = currentTimeLabel();
    }
    const hasMappedLines = (scenario.result?.feature_collection?.features || [])
      .some((feature) => feature.geometry?.type === "LineString");
    if (hasMappedLines && state.networkMapping?.ready && state.mapView !== "links") {
      await switchMapView("links");
      if (requestId !== state.scenarioRequestId) return;
      mapRefreshed = true;
    } else if (requestedIndex >= 0) {
      await refreshMap(null);
      if (requestId !== state.scenarioRequestId) return;
      mapRefreshed = true;
    }
  }

  if (requestId !== state.scenarioRequestId) return;
  activateScenario(requestedId, { focus: Boolean(scenario && focus) });
  if (scenario && focus) {
    window.setTimeout(() => $("#scenario-impact-panel")?.scrollIntoView({ behavior: "smooth", block: "start" }), 180);
  } else if (!mapRefreshed && state.project) {
    setMapBanner(
      state.mapView === "links" ? "Mapped link view" : "Historical data",
      `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`,
    );
  }
}

function defaultScenarioName(hazard) {
  return `${hazardLabel(hazard)} disruption ${state.scenarios.length + 1}`;
}

function openScenarioBuilder(sourceScenario = null, { duplicate = false } = {}) {
  if (!state.project) return toast("Choose data first", "Open a transport-network project before building a scenario.", true);
  if (state.routeBuilding) closeRouteBuilder({ restoreSelection: true, restoreScenario: true });
  if (state.recoveryBuilding) closeRecoveryBuilder({ restoreSelection: true });
  if (state.stressBuilding) closeStressBuilder();
  switchView("explore");
  state.scenarioBuilding = true;
  setScenarioEngine("network");
  state.scenarioEditingId = sourceScenario && !duplicate ? sourceScenario.scenario_id : null;
  state.scenarioEffect = "closed";
  state.scenarioDraft.clear();
  state.scenarioDraftFeatures.clear();
  state.routeBuilding = false;
  state.routePickRole = null;
  state.routeOrigin = null;
  state.routeDestination = null;
  state.researchReports = [];
  state.researchReport = null;
  state.researchJobId = null;
  state.researchFigureSvgs = new Map();
  state.recoveryBuilding = false;
  state.scenarioUndo = [];
  const sourceChanges = sourceScenario?.changes || [];
  const featureByKey = new Map((sourceScenario?.result?.feature_collection?.features || []).map((feature) => [String(feature.properties?.key), feature]));
  sourceChanges.forEach((change) => {
    state.scenarioDraft.set(change.key, { ...change });
    const feature = featureByKey.get(String(change.key));
    if (feature) state.scenarioDraftFeatures.set(change.key, scenarioFeature(feature, change));
  });
  const hazard = sourceScenario?.hazard_type || state.project.hazard_type || "other";
  $("#scenario-builder-title").textContent = state.scenarioEditingId ? "Edit this disruption" : duplicate ? "Copy this scenario" : "Build one clear disruption";
  $("#scenario-name").value = sourceScenario
    ? `${sourceScenario.name}${duplicate ? " copy" : ""}`
    : defaultScenarioName(hazard);
  $("#scenario-hazard").value = hazard;
  const timeSelect = $("#scenario-time");
  timeSelect.innerHTML = state.timeValues.map((timestamp) => `<option value="${timestamp}">${escapeHtml(state.project.summary?.time_labels?.[String(timestamp)] || String(timestamp))}</option>`).join("");
  const requestedTime = sourceScenario?.base_timestamp ?? currentTime();
  if (state.timeValues.includes(Number(requestedTime))) timeSelect.value = String(requestedTime);
  document.body.classList.add("is-scenario-building");
  $("#scenario-builder").classList.remove("is-hidden");
  $("#scenario-select").disabled = true;
  $("#edit-scenario").disabled = true;
  state.map?.setScenarioFeatures(null);
  renderScenarioImpact(null);
  state.map?.setMode("point");
  setScenarioEffect("closed");
  renderScenarioDraft();
  renderMatsimReadiness();
  setHazardAtmosphere(hazard);
  setMapBanner("Build a disruption scenario", "Choose an effect, then click roads or draw a rectangle.", true);
  $(".command-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeScenarioBuilder({ restore = true } = {}) {
  state.scenarioBuilding = false;
  state.scenarioEditingId = null;
  state.scenarioDraft.clear();
  state.scenarioDraftFeatures.clear();
  state.routeBuilding = false;
  state.routePickRole = null;
  state.routeOrigin = null;
  state.routeDestination = null;
  state.scenarioUndo = [];
  document.body.classList.remove("is-scenario-building");
  document.body.classList.remove("is-route-building");
  document.body.classList.remove("is-recovery-building");
  document.body.classList.remove("is-stress-building");
  $("#scenario-builder").classList.add("is-hidden");
  $("#route-builder").classList.add("is-hidden");
  $("#recovery-builder").classList.add("is-hidden");
  $("#stress-builder").classList.add("is-hidden");
  state.stressBuilding = false;
  $("#scenario-select").disabled = false;
  state.map?.setScenarioDraftFeatures(null);
  state.map?.clearArea();
  state.map?.setMode("point");
  if (restore) activateScenario(state.activeScenarioId);
  scheduleNextActionCues();
}

function routeMapFeature(key) {
  return (state.mapCollection?.features || []).find((feature) => String(feature.properties?.key || "") === String(key || "")) || null;
}

function routeEndpoint(feature) {
  const properties = feature?.properties || {};
  const key = String(properties.key || "");
  if (!key || !featureUsesLinkTopology(properties)) return null;
  const isPoint = feature?.geometry?.type === "Point";
  return {
    key,
    entityId: String(properties.entity_id || key.split("::").at(-1) || key),
    regionId: String(properties.region_id || key.split("::")[0] || "uploaded"),
    fromNode: String(properties.from_node || "?"),
    toNode: String(properties.to_node || "?"),
    selectionKind: isPoint ? "point" : "mapped_link",
  };
}

function renderRouteScenarioOptions(preferred = null) {
  const select = $("#route-scenario");
  if (!select) return;
  const current = preferred ?? "__quick__";
  select.innerHTML = [
    '<option value="__quick__">Create a temporary disruption here</option>',
    '<option value="">Selected historical time</option>',
  ].join("");
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

function setRoutePickRole(role) {
  if (!state.routeBuilding || !["origin", "destination"].includes(role)) return;
  state.routePickRole = role;
  state.routeDisruptionMode = null;
  state.map?.setMode("point");
  renderRouteBuilder();
  const pointView = state.mapView !== "links";
  setMapBanner(
    role === "origin" ? "Pick the start point / road" : "Pick the destination point / road",
    pointView
      ? "Click one road-reference point. Its own uploaded link ID and From → To direction become the endpoint."
      : "Click one mapped road line. Its uploaded From → To direction becomes the endpoint.",
    true,
  );
}

function currentRouteMagnitude() {
  return state.routeEffect === "closed" ? 1 : Number($("#route-magnitude").value || 0) / 100;
}

function setRouteEffect(effect) {
  if (!SCENARIO_EFFECT_LABELS[effect]) return;
  state.routeEffect = effect;
  $$('[data-route-effect]').forEach((button) => button.classList.toggle("is-active", button.dataset.routeEffect === effect));
  const field = $("#route-magnitude-field");
  const slider = $("#route-magnitude");
  field.classList.toggle("is-hidden", effect === "closed");
  if (effect === "speed_reduction") {
    slider.min = "5"; slider.max = "95"; slider.step = "5"; slider.value = "50";
    $("#route-magnitude-label").textContent = "Speed reduction";
  } else if (effect === "travel_time_increase") {
    slider.min = "5"; slider.max = "300"; slider.step = "5"; slider.value = "50";
    $("#route-magnitude-label").textContent = "Travel-time increase";
  }
  $("#route-magnitude-output").textContent = `${slider.value}%`;
}

function renderRouteDraft() {
  const quick = $("#route-scenario")?.value === "__quick__";
  $("#route-quick-disruption")?.classList.toggle("is-hidden", !quick);
  const count = state.routeDraft.size;
  $("#route-disruption-count").textContent = `${formatNumber(count)} affected road${count === 1 ? "" : "s"}`;
  state.map?.setScenarioDraftFeatures(quick ? { type: "FeatureCollection", features: [...state.routeDraftFeatures.values()] } : null);
  $("#route-pick-disrupted")?.classList.toggle("is-active", quick && state.routeDisruptionMode === "roads");
  $("#route-draw-disrupted")?.classList.toggle("is-active", quick && state.routeDisruptionMode === "area");
  scheduleNextActionCues();
}

function applyRouteDisruptionFeatures(features) {
  const usable = (features || []).filter((feature) => String(feature.properties?.key || "").includes("::"));
  const projected = new Set([...state.routeDraft.keys(), ...usable.map((feature) => String(feature.properties.key))]);
  if (projected.size > 5000) return toast("Area is too large", "Narrow the rectangle to at most 5,000 affected roads.", true);
  const change = { effect: state.routeEffect, magnitude: currentRouteMagnitude() };
  usable.forEach((feature) => {
    const key = String(feature.properties.key);
    const saved = { key, ...change };
    state.routeDraft.set(key, saved);
    state.routeDraftFeatures.set(key, scenarioFeature(feature, saved));
  });
  renderRouteDraft();
  renderRouteBuilder();
  updatePendingRouteTask();
}

function routeDisruptionFeatureSelected(feature) {
  if (!state.routeBuilding || state.routeDisruptionMode !== "roads") return;
  applyRouteDisruptionFeatures([feature]);
  setMapBanner("Affected road added", "Continue clicking roads, draw an area, or calculate the route options.", true);
  if (pendingAssistantMapStep("route_road")) {
    returnAssistantMapInput(
      "route_road",
      `Affected road ${feature?.properties?.entity_id || feature?.properties?.key || "selected"} was added to the route disruption.`,
    );
  }
}

async function routeAreaSelected(bbox) {
  if (!state.project || !state.routeBuilding) return;
  state.routeAreaLoading = true;
  $("#route-builder").setAttribute("aria-busy", "true");
  try {
    const targets = await api.scenarioTargets(state.project.project_id, bbox);
    if (targets.truncated) return toast("Draw a smaller area", targets.message, true);
    if (!targets.selected_count) return toast("Nothing selected", "The rectangle contains no mapped road.", true);
    applyRouteDisruptionFeatures(targets.feature_collection.features);
    state.routeDisruptionMode = "roads";
    state.map?.setMode("point");
    renderRouteDraft();
    setMapBanner("Affected area added", `${formatNumber(targets.selected_count)} roads are included in this temporary disruption.`, true);
    if (pendingAssistantMapStep("route_area")) {
      returnAssistantMapInput(
        "route_area",
        `${formatNumber(targets.selected_count)} affected roads were captured from the drawn area.`,
      );
    }
  } finally {
    state.routeAreaLoading = false;
    $("#route-builder").removeAttribute("aria-busy");
  }
}

function clearRouteDisruption() {
  state.routeDraft.clear();
  state.routeDraftFeatures.clear();
  state.map?.clearArea();
  renderRouteDraft();
  renderRouteBuilder();
}

function renderRouteBuilder() {
  const origin = state.routeOrigin;
  const destination = state.routeDestination;
  const endpointDetail = (endpoint, fallback) => endpoint
    ? `${endpoint.selectionKind === "point" ? "Point → uploaded road" : "Mapped road"} · ${endpoint.regionId} · ${endpoint.fromNode} → ${endpoint.toNode}`
    : fallback;
  $("#route-origin-label").textContent = origin?.entityId || "Not selected";
  $("#route-origin-detail").textContent = endpointDetail(origin, "Click “Pick start”, then choose a routable point or line.");
  $("#route-destination-label").textContent = destination?.entityId || "Not selected";
  $("#route-destination-detail").textContent = endpointDetail(destination, "Choose the routable point or line where the route should end.");
  $("#route-origin-card").classList.toggle("is-ready", Boolean(origin));
  $("#route-destination-card").classList.toggle("is-ready", Boolean(destination));
  $("#route-origin-card").classList.toggle("is-picking", state.routePickRole === "origin");
  $("#route-destination-card").classList.toggle("is-picking", state.routePickRole === "destination");
  $("#route-pick-origin").textContent = origin ? "Change start" : "Pick start";
  $("#route-pick-destination").textContent = destination ? "Change destination" : "Pick destination";
  const endpointsReady = Boolean(origin && destination && origin.key !== destination.key);
  const scenarioId = $("#route-scenario").value;
  const quickReady = scenarioId !== "__quick__" || state.routeDraft.size > 0;
  $("#run-route").disabled = !(endpointsReady && quickReady);
  $("#route-swap").disabled = !endpointsReady;
  [
    ["#route-view-points", state.mapView !== "links"],
    ["#route-view-links", state.mapView === "links"],
  ].forEach(([selector, active]) => {
    const button = $(selector);
    button?.classList.toggle("is-active", active);
    button?.setAttribute("aria-pressed", String(active));
  });
  $("#route-view-note").textContent = state.mapView === "links"
    ? "Mapped links are derived from uploaded topology. They may be straight reconstructions rather than surveyed road centre-lines."
    : "A usable point keeps its uploaded road ID and From → To direction. Ordinary points are never randomly snapped to roads.";
  const steps = $$("#route-builder .route-builder-steps span");
  const activeStep = !origin ? 0 : !destination ? 1 : scenarioId === "__quick__" && !state.routeDraft.size ? 2 : 3;
  steps.forEach((step, index) => step.classList.toggle("is-current", index === activeStep));
  const scenario = state.scenarios.find((item) => item.scenario_id === scenarioId);
  if (scenarioId === "__quick__") {
    $("#route-time-note").textContent = state.routeDraft.size
      ? `${formatNumber(state.routeDraft.size)} affected roads will be applied at ${currentTimeLabel()}.`
      : "Choose affected roads or draw an area below; this temporary disruption is not saved.";
    $("#run-route").textContent = "Calculate route options";
  } else if (scenarioId === "__all__") {
    $("#route-time-note").textContent = `Recomputes one route for the map time and one for each of ${state.scenarios.length} saved scenarios; each scenario uses its own saved baseline time.`;
    $("#run-route").textContent = `Recommend ${state.scenarios.length + 1} routes`;
  } else if (scenario) {
    $("#route-time-note").textContent = `Uses saved scenario “${scenario.name}” at its confirmed baseline time.`;
    $("#run-route").textContent = "Calculate route options";
  } else {
    $("#route-time-note").textContent = `Compares the internal reference with the map time: ${currentTimeLabel()}.`;
    $("#run-route").textContent = "Calculate route options";
  }
  renderRouteDraft();
}

async function openRouteBuilder() {
  if (!state.project) return toast("Choose data first", "Open a transport-network project before comparing routes.", true);
  const capability = state.project.summary?.capabilities?.route_detour_analysis;
  if (capability === "unavailable") return toast("Route topology unavailable", "Map From node and To node fields before comparing routes.", true);
  if (state.scenarioBuilding) closeScenarioBuilder({ restore: false });
  if (state.recoveryBuilding) closeRecoveryBuilder({ restoreSelection: false });
  if (state.stressBuilding) closeStressBuilder();
  switchView("explore");
  state.routePreviousSelection = [...state.selectedKeys];
  state.routePreviousScenarioId = state.activeScenarioId;
  if (state.activeScenarioId) activateScenario(null);
  state.routeBuilding = true;
  state.routeOrigin = null;
  state.routeDestination = null;
  state.routeEffect = "closed";
  state.routeDisruptionMode = null;
  state.routeDraft.clear();
  state.routeDraftFeatures.clear();
  const previous = (state.analysisScopeMode === "selection" ? state.routePreviousSelection : []).map(routeMapFeature).filter(Boolean).map(routeEndpoint).filter(Boolean);
  if (previous.length === 2 && previous[0].key !== previous[1].key) {
    [state.routeOrigin, state.routeDestination] = previous;
  }
  // Route planning always starts with a fresh temporary disruption. Saved
  // scenarios remain in the scenario library and do not populate this input.
  renderRouteScenarioOptions("__quick__");
  document.body.classList.add("is-route-building");
  $("#route-builder").classList.remove("is-hidden");
  setRouteEffect("closed");
  state.map?.setScenarioFeatures(null);
  state.map?.setSelectedKeys([state.routeOrigin?.key, state.routeDestination?.key].filter(Boolean));
  if (state.routeOrigin && state.routeDestination) {
    state.routePickRole = null;
    state.routeDisruptionMode = $("#route-scenario").value === "__quick__" ? "roads" : null;
    setMapBanner("Route endpoints ready", state.routeDisruptionMode ? "Now click affected roads or draw an affected area." : "Review the network state, then calculate route options.", true);
  } else {
    setRoutePickRole(state.routeOrigin ? "destination" : "origin");
  }
  renderRouteBuilder();
  $(".command-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function switchRouteEndpointView(view) {
  if (!state.routeBuilding || !state.project) return;
  const target = view === "links" ? "links" : "points";
  if (target === "links" && !state.networkMapping?.available) {
    return toast("Mapped links unavailable", state.networkMapping?.reason || "The uploaded data needs usable From and To node fields.", true);
  }
  setLoading(true, target === "links" ? "Preparing mapped links" : "Opening point map", "Keeping both selected endpoints while the display changes.");
  try {
    if (target === "links" && !state.networkMapping?.ready) {
      state.networkMapping = await api.buildNetworkMapping(state.project.project_id);
      configureNetworkControls();
      await refreshMatsimReadiness();
    }
    await switchMapView(target);
    state.map?.setSelectedKeys([state.routeOrigin?.key, state.routeDestination?.key].filter(Boolean));
    renderRouteBuilder();
    if (state.routePickRole) setRoutePickRole(state.routePickRole);
    else setMapBanner("Route endpoints ready", "Review the network states, then calculate a recommendation for each state.", true);
  } catch (error) {
    toast("Route display could not change", error.message, true);
  } finally {
    setLoading(false);
  }
}

function closeRouteBuilder({ restoreSelection = true, restoreScenario = true } = {}) {
  state.routeBuilding = false;
  state.routePickRole = null;
  state.routeDisruptionMode = null;
  state.routeDraft.clear();
  state.routeDraftFeatures.clear();
  document.body.classList.remove("is-route-building");
  $("#route-builder").classList.add("is-hidden");
  state.map?.setScenarioDraftFeatures(null);
  state.map?.clearArea();
  if (restoreSelection) {
    state.selectedKeys = new Set(state.routePreviousSelection || []);
    state.map?.setSelectedKeys([...state.selectedKeys]);
  }
  if (restoreScenario && state.routePreviousScenarioId) activateScenario(state.routePreviousScenarioId);
  else if (state.project) setMapBanner("Historical data", `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`);
  renderSelectionSummary();
  scheduleNextActionCues();
}

function routeFeatureSelected(feature) {
  if (!state.routeBuilding) return;
  if (!state.routePickRole && state.routeDisruptionMode === "roads") return routeDisruptionFeatureSelected(feature);
  const selectedRole = state.routePickRole;
  const endpoint = routeEndpoint(feature);
  if (!endpoint) return toast("This point is not routable", "Choose a point or line that contains an uploaded road-link ID plus From and To nodes. Nothing is randomly snapped.", true);
  if (state.routePickRole === "destination" && endpoint.key === state.routeOrigin?.key) {
    return toast("Choose a different destination", "The start and destination roads must be different.", true);
  }
  if (state.routePickRole === "origin" && endpoint.key === state.routeDestination?.key) {
    return toast("Choose a different start", "The start and destination roads must be different.", true);
  }
  if (state.routePickRole === "origin") {
    state.routeOrigin = endpoint;
    state.routePickRole = "destination";
  } else {
    state.routeDestination = endpoint;
    state.routePickRole = state.routeOrigin ? null : "origin";
  }
  state.selectedKeys = new Set([state.routeOrigin?.key, state.routeDestination?.key].filter(Boolean));
  state.map?.setSelectedKeys([...state.selectedKeys]);
  renderRouteBuilder();
  renderSelectionSummary();
  if (state.routeOrigin && state.routeDestination) {
    if ($("#route-scenario").value === "__quick__") {
      state.routeDisruptionMode = "roads";
      setMapBanner("Route endpoints ready", "Now click affected roads or draw an affected area.", true);
    } else {
      setMapBanner("Route endpoints ready", "Review the network state, then calculate route options.", true);
    }
  } else {
    setMapBanner("Start point / road selected", "Now click the destination point or road.", true);
  }
  updatePendingRouteTask();
  if (selectedRole === "origin" && pendingAssistantMapStep("route_origin")) {
    returnAssistantMapInput("route_origin", `Start point confirmed: road ${endpoint.entityId}.`);
  } else if (selectedRole === "destination" && pendingAssistantMapStep("route_destination")) {
    returnAssistantMapInput("route_destination", `Destination confirmed: road ${endpoint.entityId}.`);
  }
}

async function runRouteComparison() {
  if (!state.project || !state.routeOrigin || !state.routeDestination) return;
  const scenarioId = $("#route-scenario").value || "";
  const scenario = state.scenarios.find((item) => item.scenario_id === scenarioId) || null;
  const allScenarios = scenarioId === "__all__";
  const quickScenario = scenarioId === "__quick__";
  if (quickScenario && !state.routeDraft.size) return toast("Add a disruption first", "Click affected roads or draw an affected area before calculating route options.", true);
  const scenarioIds = allScenarios ? state.scenarios.map((item) => item.scenario_id) : [];
  const alternativeCount = Number($("#route-alternative-count").value || 2);
  if (!Number.isInteger(alternativeCount) || alternativeCount < 1 || alternativeCount > 50) {
    return toast("Choose a route count", "Enter a whole number from 1 to 50.", true);
  }
  const disruptionCount = quickScenario
    ? state.routeDraft.size
    : allScenarios
      ? state.scenarios.reduce((total, item) => total + Number(item.changes?.length || 0), 0)
      : Number(scenario?.changes?.length || 0);
  const routeWait = beginWorkloadWait(workloadWaitProfile({
    analysis_type: "route_detour_analysis",
    top_k: alternativeCount,
    selected_keys: [state.routeOrigin.key, state.routeDestination.key],
  }, { requestedCount: alternativeCount, changeCount: disruptionCount, networkWide: true }),
  "Planning routes under disruption",
  `Computing up to ${formatNumber(alternativeCount)} distinct feasible route options under the selected disruption.`);
  try {
    const result = promoteMappedLinkResults(await api.analyse({
      project_id: state.project.project_id,
      analysis_type: "route_detour_analysis",
      model_id: "P5-M23",
      metric: state.metric,
      threshold: 0.2,
      top_k: alternativeCount,
      selected_keys: [state.routeOrigin.key, state.routeDestination.key],
      bbox: null,
      time_start: null,
      time_end: scenario?.base_timestamp ?? currentTime(),
      command_text: allScenarios
        ? "Recommend routes between the selected points for the historical state and all saved disruption scenarios."
        : quickScenario ? "Recommend route options under the temporary disruption defined in this route tool."
        : scenario ? `Compare the selected route under ${scenario.name}.` : "Compare the selected historical route with the internal reference.",
      assumptions: {
        origin_key: state.routeOrigin.key,
        destination_key: state.routeDestination.key,
        scenario_id: scenario?.scenario_id || null,
        scenario_ids: scenarioIds,
        scenario_changes: quickScenario ? [...state.routeDraft.values()] : null,
        scenario_name: quickScenario ? "temporary route disruption" : scenario?.name || null,
        alternative_count: alternativeCount,
        maximum_overlap: 0.60,
        capability: state.project.summary?.capabilities?.route_detour_analysis || "conditional",
        llm_used: false,
      },
    }));
    if (routeWait.cancelled) return;
    if (!await routeWait.complete("Route options ready · preparing the map and route comparison…")) return;
    if (routeWait.cancelled) return;
    state.networkMapping = await api.networkMappingStatus(state.project.project_id).catch(() => state.networkMapping);
    configureNetworkControls();
    closeRouteBuilder({ restoreSelection: false, restoreScenario: false });
    renderResult(result, { scroll: true });
    state.map?.focusCollection(state.analysisCollection);
    await refreshRuns();
    const returned = Number(result.visual?.returned_count ?? result.visual?.routes?.length ?? 0);
    toast("Route recommendations ready", allScenarios
      ? `${scenarioIds.length + 1} historical/scenario recommendations are shown with separate colours.`
      : returned < alternativeCount
        ? `Requested ${formatNumber(alternativeCount)}; ${formatNumber(returned)} distinct feasible route${returned === 1 ? " is" : "s are"} available. The result explains why.`
        : `${formatNumber(returned)} distinct feasible routes are shown. Hover or click a coloured route for the same values listed in the table.`);
    if (pendingAssistantTaskMatches("route")) {
      updateAssistantActivity("execute", "Route options calculated with the confirmed inputs", "complete");
      updateAssistantActivity("verify", "Saving and preparing the route map and complete table", "active");
      completeAssistantPendingTask("route", result);
      updateAssistantActivity("verify", "Route map, route details and Project Memory are ready", "complete");
    }
    return result;
  } catch (error) {
    routeWait.fail();
    toast("Route comparison did not run", error.message, true);
    if (pendingAssistantTaskMatches("route")) {
      updateAssistantActivity("execute", "The route calculation did not complete", "failed");
    }
    return null;
  } finally {
    routeWait.close();
  }
}

function recoveryResultSources() {
  const roads = completedResultFor("critical_components");
  const areas = completedResultFor("spatial_hotspots");
  return [
    ...(roads?.items?.length ? [{ value: "__critical_roads__", label: `Latest critical-road result (${formatNumber(roads.items.length)})`, unit: "road" }] : []),
    ...(areas?.items?.length ? [{ value: "__critical_areas__", label: `Latest critical-area result (${formatNumber(areas.items.length)})`, unit: "area" }] : []),
  ];
}

function renderRecoverySourceOptions(preferred = null) {
  const select = $("#recovery-source");
  const resultSources = recoveryResultSources();
  select.innerHTML = [
    '<option value="manual">Select on this map</option>',
    ...resultSources.map((source) => `<option value="${source.value}">${escapeHtml(source.label)}</option>`),
    ...state.scenarios.map((scenario) => `<option value="scenario:${escapeHtml(scenario.scenario_id)}">Saved scenario · ${escapeHtml(scenario.name)}</option>`),
  ].join("");
  const currentValue = state.currentResult?.analysis_type === "spatial_hotspots" ? "__critical_areas__" : "__critical_roads__";
  const currentResultSource = resultSources.find((source) => source.value === currentValue);
  const fallback = currentResultSource?.value || resultSources[0]?.value || (state.activeScenarioId ? `scenario:${state.activeScenarioId}` : "manual");
  const target = preferred || fallback;
  select.value = [...select.options].some((option) => option.value === target) ? target : "manual";
}

function recoveryPayloadFromSource() {
  const source = $("#recovery-source").value || "manual";
  if (source.startsWith("scenario:")) {
    const scenarioId = source.slice("scenario:".length);
    const scenario = state.scenarios.find((item) => item.scenario_id === scenarioId);
    const changes = scenario?.changes || [];
    return {
      selectedKeys: changes.map((item) => String(item.key)),
      scenarioId,
      scenarioChanges: changes,
      areaCandidates: [],
      label: scenario?.name || "saved disruption",
    };
  }
  if (source === "__critical_roads__") {
    const result = completedResultFor("critical_components");
    const selectedKeys = (result?.items || []).map((item) => String(item.key)).filter(Boolean);
    return { selectedKeys, scenarioId: null, scenarioChanges: [], areaCandidates: [], label: "latest critical-road result" };
  }
  if (source === "__critical_areas__") {
    const result = completedResultFor("spatial_hotspots");
    const areaCandidates = (result?.items || []).map((item) => ({
      id: item.cell_id,
      label: item.cell_id,
      keys: item.entity_keys || [],
      bbox: [item.west, item.south, item.east, item.north],
    }));
    return {
      selectedKeys: [...new Set(areaCandidates.flatMap((item) => item.keys))],
      scenarioId: null,
      scenarioChanges: [],
      areaCandidates,
      label: "latest critical-area result",
    };
  }
  return {
    selectedKeys: [...state.recoveryDraft],
    scenarioId: null,
    scenarioChanges: [],
    areaCandidates: state.recoveryAreas.map((item) => ({ ...item })),
    label: "manual map selection",
  };
}

function renderRecoveryBuilder() {
  const payload = recoveryPayloadFromSource();
  const source = $("#recovery-source").value || "manual";
  const count = payload.selectedKeys.length;
  $("#recovery-selection-count").textContent = `${formatNumber(count)} disrupted road${count === 1 ? "" : "s"} available`;
  $("#recovery-source-note").textContent = source === "manual"
    ? "Click roads or draw one or more geographic areas below."
    : `Using ${payload.label}; switch to map selection at any time.`;
  $("#recovery-selection-detail").textContent = state.recoveryUnit === "area"
    ? `${formatNumber(payload.areaCandidates.length || count)} mapped area package${(payload.areaCandidates.length || count) === 1 ? "" : "s"} will be formed; overlapping roads are restored once.`
    : "Each returned priority represents one uploaded road link.";
  $("#recovery-unit-road").classList.toggle("is-active", state.recoveryUnit === "road");
  $("#recovery-unit-area").classList.toggle("is-active", state.recoveryUnit === "area");
  $("#recovery-unit-road").setAttribute("aria-pressed", String(state.recoveryUnit === "road"));
  $("#recovery-unit-area").setAttribute("aria-pressed", String(state.recoveryUnit === "area"));
  const manual = source === "manual";
  $("#recovery-pick-roads").disabled = !manual;
  $("#recovery-draw-area").disabled = !manual;
  $("#recovery-clear").disabled = !manual || !count;
  $("#run-recovery").disabled = !count;
  const objective = $("#recovery-objective").value;
  const metric = state.project?.summary?.metrics || {};
  if (objective === "gain_per_cost" && !metric.repair_cost?.available) {
    $("#recovery-method-note").textContent = "Repair cost is not present for every candidate. The model will disclose that and fall back to network service restored.";
  } else if (objective === "gain_per_duration" && !metric.repair_duration?.available) {
    $("#recovery-method-note").textContent = "Repair duration is not present for every candidate. The model will disclose that and fall back to network service restored.";
  } else {
    $("#recovery-method-note").textContent = "Tractable candidate sets are recalculated after every restoration stage. Large sets use a clearly labelled structural screen without claiming uncomputed service recovery.";
  }
  if (manual) {
    state.map?.setScenarioDraftFeatures({ type: "FeatureCollection", features: [...state.recoveryDraftFeatures.values()] });
    state.map?.setSelectedKeys([...state.recoveryDraft]);
  } else {
    state.map?.setScenarioDraftFeatures(null);
    state.map?.setSelectedKeys(payload.selectedKeys);
  }
  const steps = $$("#recovery-builder .recovery-builder-steps span");
  steps.forEach((step, index) => step.classList.toggle("is-current", index === (!count ? 0 : 1)));
  scheduleNextActionCues();
}

function openRecoveryBuilder() {
  if (!state.project) return toast("Choose data first", "Open a transport-network project before planning recovery.", true);
  if (state.project.summary?.capabilities?.restoration_screening === "unavailable") return toast("Recovery topology unavailable", "Map From node and To node fields before ranking recovery priorities.", true);
  if (state.scenarioBuilding) closeScenarioBuilder({ restore: false });
  if (state.routeBuilding) closeRouteBuilder({ restoreSelection: false, restoreScenario: false });
  if (state.stressBuilding) closeStressBuilder();
  switchView("explore");
  state.recoveryPreviousSelection = [...state.selectedKeys];
  state.recoveryBuilding = true;
  state.recoveryMode = "roads";
  const preferredSource = recoveryResultSources().find((source) => source.value === (state.currentResult?.analysis_type === "spatial_hotspots" ? "__critical_areas__" : "__critical_roads__"));
  state.recoveryUnit = preferredSource?.unit || "road";
  state.recoveryDraft = new Set(state.analysisScopeMode === "selection" ? state.selectedKeys : []);
  state.recoveryDraftFeatures.clear();
  state.recoveryAreas = [];
  state.recoveryDraft.forEach((key) => {
    const feature = routeMapFeature(key);
    if (feature) state.recoveryDraftFeatures.set(key, scenarioFeature(feature, { key, effect: "closed", magnitude: 1 }));
  });
  renderRecoverySourceOptions();
  document.body.classList.add("is-recovery-building");
  $("#recovery-builder").classList.remove("is-hidden");
  state.map?.setMode("point");
  renderRecoveryBuilder();
  setMapBanner("Choose disrupted roads", "Use the current result, a saved scenario, click roads or draw an area.", true);
  $(".command-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderStressBuilder() {
  const critical = completedResultFor("critical_components");
  const available = (critical?.items || []).filter((item) => item?.key).length;
  const requested = Math.max(1, Number.parseInt($("#stress-top-k")?.value || "20", 10) || 20);
  const reusable = available >= requested;
  const card = $("#stress-source-card");
  card?.classList.toggle("is-fresh", !reusable);
  if (reusable) {
    $("#stress-source-title").textContent = `Reuse the latest ${formatNumber(available)}-road criticality result`;
    $("#stress-source-note").textContent = `The first ${formatNumber(requested)} roads will be removed in that confirmed order. Start fresh clears this dependency.`;
    $("#stress-source-card .stress-source-icon").textContent = "✓";
  } else if (available) {
    $("#stress-source-title").textContent = `Re-identify ${formatNumber(requested)} critical roads`;
    $("#stress-source-note").textContent = `The current result contains only ${formatNumber(available)} roads, so the platform will recompute a complete deterministic order for all requested stages.`;
    $("#stress-source-card .stress-source-icon").textContent = "↻";
  } else {
    $("#stress-source-title").textContent = "Identify critical roads first";
    $("#stress-source-note").textContent = `No current critical-road result is being reused. The platform will identify a fresh deterministic Top ${formatNumber(requested)} order from this network, then run the cumulative test.`;
    $("#stress-source-card .stress-source-icon").textContent = "01";
  }
  $("#stress-top-note").textContent = reusable
    ? `Current dependency is valid: ${formatNumber(requested)} of ${formatNumber(available)} ranked roads will be tested.`
    : "Every requested stage is returned unless the network has fewer eligible roads.";
  const steps = $$("#stress-builder .stress-builder-steps span");
  steps.forEach((step, index) => step.classList.toggle("is-current", index === (requested ? 1 : 0)));
  scheduleNextActionCues();
}

function openStressBuilder() {
  if (!state.project) return toast("Choose data first", "Open a transport-network project before testing network resilience.", true);
  if (state.project.summary?.capabilities?.progressive_robustness === "unavailable") {
    return toast("Network topology unavailable", "Map From node and To node fields before testing resilience.", true);
  }
  if (state.scenarioBuilding) closeScenarioBuilder({ restore: false });
  if (state.routeBuilding) closeRouteBuilder({ restoreSelection: false, restoreScenario: false });
  if (state.recoveryBuilding) closeRecoveryBuilder({ restoreSelection: true });
  switchView("explore");
  state.stressBuilding = true;
  document.body.classList.add("is-stress-building");
  $("#stress-builder").classList.remove("is-hidden");
  const latest = completedResultFor("critical_components");
  const latestCount = (latest?.items || []).length;
  $("#stress-top-k").value = String(latestCount || 20);
  renderStressBuilder();
  setMapBanner("Set the resilience-test length", "Reuse the current critical-road order, or identify a fresh order after a reset.", true);
  $(".command-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeStressBuilder() {
  state.stressBuilding = false;
  document.body.classList.remove("is-stress-building");
  $("#stress-builder")?.classList.add("is-hidden");
  if (state.project) setMapBanner(
    state.mapView === "links" ? "Mapped link view" : "Historical data",
    `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`,
  );
  scheduleNextActionCues();
}

async function runStressTest() {
  if (!state.project) return;
  const requested = Number($("#stress-top-k").value);
  if (!Number.isInteger(requested) || requested < 1) {
    return toast("Enter a valid stage count", "The number of cumulative road failures must be a positive whole number.", true);
  }
  const critical = completedResultFor("critical_components");
  const availableKeys = (critical?.items || []).map((item) => String(item.key || "")).filter(Boolean);
  const failureKeys = availableKeys.length >= requested ? availableKeys.slice(0, requested) : [];
  setLoading(true, "Testing network resilience", `Removing up to ${formatNumber(requested)} critical roads one at a time and recalculating network efficiency after every stage.`);
  try {
    await ensureLinkAnalysisView("progressive_robustness");
    const job = await api.startAnalysis({
      project_id: state.project.project_id,
      analysis_type: "progressive_robustness",
      model_id: "P5-M11",
      metric: state.metric,
      threshold: 0.2,
      top_k: requested,
      selected_keys: [],
      bbox: null,
      time_start: null,
      time_end: currentTime(),
      command_text: `Test network resilience over ${requested} cumulative critical-road failures.`,
      assumptions: {
        failure_keys: failureKeys,
        failure_order_source: failureKeys.length ? "latest_confirmed_critical_road_result" : "fresh_deterministic_criticality_order",
        llm_used: false,
      },
    });
    const result = promoteMappedLinkResults(await pollAnalysisJob(job.job_id, workloadWaitProfile({
      analysis_type: "progressive_robustness",
      top_k: requested,
      selected_keys: failureKeys,
    }, { requestedCount: requested, networkWide: true })));
    closeStressBuilder();
    renderResult(result, { scroll: true });
    state.map?.focusCollection(state.analysisCollection);
    await refreshRuns();
    toast("Network resilience test ready", `${formatNumber(result.items?.length || 0)} cumulative removal stages were returned.`);
    if (pendingAssistantTaskMatches("stress")) {
      updateAssistantActivity("execute", "Network resilience test completed", "complete");
      updateAssistantActivity("verify", "Saving the result and preparing its visual evidence", "active");
      completeAssistantPendingTask("stress", result);
      updateAssistantActivity("verify", "Map, resilience curve, complete table and Project Memory are ready", "complete");
    }
    return result;
  } catch (error) {
    toast("Network resilience test did not run", error.message, true);
    if (pendingAssistantTaskMatches("stress")) updateAssistantActivity("execute", "The resilience test did not complete", "failed");
    return null;
  } finally {
    setLoading(false);
  }
}

function closeRecoveryBuilder({ restoreSelection = true } = {}) {
  state.recoveryBuilding = false;
  state.recoveryMode = null;
  document.body.classList.remove("is-recovery-building");
  $("#recovery-builder").classList.add("is-hidden");
  state.map?.setScenarioDraftFeatures(null);
  state.map?.clearArea();
  state.map?.setMode("point");
  if (restoreSelection) {
    state.selectedKeys = new Set(state.recoveryPreviousSelection || []);
    state.map?.setSelectedKeys([...state.selectedKeys]);
  }
  if (state.project) setMapBanner("Historical data", `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`);
  renderSelectionSummary();
  scheduleNextActionCues();
}

function recoveryFeatureSelected(feature) {
  if (!state.recoveryBuilding || state.recoveryMode !== "roads") return;
  const key = String(feature?.properties?.key || "");
  if (!key.includes("::") || !featureUsesLinkTopology(feature.properties || {})) return toast("Choose a routable road", "Recovery ranking needs an uploaded road ID with From and To nodes.", true);
  $("#recovery-source").value = "manual";
  state.recoveryDraft.add(key);
  state.recoveryDraftFeatures.set(key, scenarioFeature(feature, { key, effect: "closed", magnitude: 1 }));
  renderRecoveryBuilder();
  setMapBanner("Disrupted road added", "Continue selecting or run the recovery priority model.", true);
  if (pendingAssistantMapStep("recovery_road")) {
    returnAssistantMapInput(
      "recovery_road",
      `Disrupted road ${feature?.properties?.entity_id || key.split("::").at(-1) || key} was added to the recovery candidates.`,
    );
  }
}

async function recoveryAreaSelected(bbox) {
  if (!state.project || !state.recoveryBuilding) return;
  const targets = await api.scenarioTargets(state.project.project_id, bbox);
  if (targets.truncated) return toast("Draw a smaller area", targets.message, true);
  if (!targets.selected_count) return toast("Nothing selected", "The rectangle contains no mapped road.", true);
  $("#recovery-source").value = "manual";
  const keys = targets.feature_collection.features
    .map((feature) => String(feature.properties?.key || ""))
    .filter((key) => key.includes("::"));
  keys.forEach((key) => state.recoveryDraft.add(key));
  targets.feature_collection.features.forEach((feature) => {
    const key = String(feature.properties?.key || "");
    if (key) state.recoveryDraftFeatures.set(key, scenarioFeature(feature, { key, effect: "closed", magnitude: 1 }));
  });
  state.recoveryAreas.push({ id: `drawn-area-${state.recoveryAreas.length + 1}`, label: `Drawn area ${state.recoveryAreas.length + 1}`, keys, bbox });
  state.recoveryUnit = "area";
  state.recoveryMode = "roads";
  state.map?.setMode("point");
  renderRecoveryBuilder();
  setMapBanner("Disrupted area added", `${formatNumber(keys.length)} mapped roads are grouped for area recovery ranking.`, true);
  if (pendingAssistantMapStep("recovery_area")) {
    returnAssistantMapInput(
      "recovery_area",
      `${formatNumber(keys.length)} disrupted roads were grouped as one recovery area.`,
    );
  }
}

function clearRecoveryDraft() {
  state.recoveryDraft.clear();
  state.recoveryDraftFeatures.clear();
  state.recoveryAreas = [];
  state.map?.clearArea();
  renderRecoveryBuilder();
}

async function runRecoveryPriorities(options = {}) {
  if (!state.project) return;
  const fromAssistant = Boolean(options?.fromAssistant);
  const source = recoveryPayloadFromSource();
  if (!source.selectedKeys.length) return toast("Choose disrupted roads", "Use a saved scenario, current critical result, click roads or draw an area.", true);
  const requested = Number($("#recovery-top-k").value);
  if (!Number.isInteger(requested) || requested < 1) return toast("Enter a valid result count", "Top-N must be a positive whole number.", true);
  setLoading(true, "Ranking recovery priorities", "Applying the confirmed disruption, then evaluating the defensible model tier for these candidates.");
  try {
    await ensureLinkAnalysisView("restoration_screening");
    const job = await api.startAnalysis({
      project_id: state.project.project_id,
      analysis_type: "restoration_screening",
      model_id: $("#recovery-objective").value === "service_gain" ? "P5-M4" : "P5-M8",
      metric: state.metric,
      threshold: 0.2,
      top_k: requested,
      selected_keys: source.selectedKeys,
      bbox: null,
      time_start: null,
      time_end: currentTime(),
      command_text: `Return the top ${requested} ${state.recoveryUnit} recovery priorities from ${source.label}.`,
      assumptions: {
        recovery_unit: state.recoveryUnit,
        priority_objective: $("#recovery-objective").value,
        scenario_id: source.scenarioId,
        scenario_changes: source.scenarioChanges,
        area_candidates: source.areaCandidates,
        source_label: source.label,
        llm_used: fromAssistant,
      },
    });
    const result = promoteMappedLinkResults(await pollAnalysisJob(job.job_id, workloadWaitProfile({
      analysis_type: "restoration_screening",
      top_k: requested,
      selected_keys: source.selectedKeys,
    }, { requestedCount: requested, selectedCount: source.selectedKeys.length })));
    closeRecoveryBuilder({ restoreSelection: false });
    renderResult(result, { scroll: true });
    state.map?.focusCollection(state.analysisCollection);
    await refreshRuns();
    toast("Recovery priorities ready", `${formatNumber(result.items?.length || 0)} ranked ${state.recoveryUnit} priorities are shown highest first.`);
    if (pendingAssistantTaskMatches("recovery")) {
      updateAssistantActivity("execute", "Recovery ranking completed", "complete");
      updateAssistantActivity("verify", "Saving the result and preparing its visual evidence", "active");
      completeAssistantPendingTask("recovery", result);
      updateAssistantActivity("verify", "Map, chart, complete table and Project Memory are ready", "complete");
    }
    return result;
  } catch (error) {
    toast("Recovery priorities did not run", error.message, true);
    if (fromAssistant) throw error;
    return null;
  } finally {
    setLoading(false);
  }
}

function setScenarioEffect(effect) {
  if (!SCENARIO_EFFECT_LABELS[effect]) return;
  state.scenarioEffect = effect;
  $$('[data-scenario-effect]').forEach((button) => button.classList.toggle("is-active", button.dataset.scenarioEffect === effect));
  const field = $("#scenario-magnitude-field");
  const slider = $("#scenario-magnitude");
  field.classList.toggle("is-hidden", effect === "closed");
  if (effect === "speed_reduction") {
    slider.min = "5"; slider.max = "95"; slider.step = "5"; slider.value = "50";
    $("#scenario-magnitude-label").textContent = "Speed reduction";
  } else if (effect === "travel_time_increase") {
    slider.min = "5"; slider.max = "300"; slider.step = "5"; slider.value = "50";
    $("#scenario-magnitude-label").textContent = "Travel-time increase";
  }
  $("#scenario-magnitude-output").textContent = `${slider.value}%`;
  $("#scenario-map-instruction strong").textContent = effect === "closed" ? "Click roads that become unavailable" : effect === "speed_reduction" ? "Click roads that become slower" : "Click roads with longer travel time";
}

function currentScenarioMagnitude() {
  return state.scenarioEffect === "closed" ? 1 : Number($("#scenario-magnitude").value || 0) / 100;
}

function pushScenarioUndo(keys) {
  const unique = [...new Set(keys)];
  state.scenarioUndo.push(unique.map((key) => ({
    key,
    change: state.scenarioDraft.has(key) ? { ...state.scenarioDraft.get(key) } : null,
    feature: state.scenarioDraftFeatures.get(key) || null,
  })));
  if (state.scenarioUndo.length > 20) state.scenarioUndo.shift();
}

function applyScenarioFeatures(features) {
  const usable = (features || []).filter((feature) => String(feature.properties?.key || "").includes("::"));
  const keys = [...new Set(usable.map((feature) => String(feature.properties.key)))];
  const projectedCount = new Set([...state.scenarioDraft.keys(), ...keys]).size;
  if (projectedCount > 5000) {
    toast("Area is too large", "Narrow the map rectangle so this scenario contains at most 5,000 mapped entities.", true);
    return;
  }
  pushScenarioUndo(keys);
  const change = { effect: state.scenarioEffect, magnitude: currentScenarioMagnitude() };
  usable.forEach((feature) => {
    const key = String(feature.properties.key);
    const saved = { key, ...change };
    state.scenarioDraft.set(key, saved);
    state.scenarioDraftFeatures.set(key, scenarioFeature(feature, saved));
  });
  renderScenarioDraft();
}

function scenarioFeatureSelected(feature) {
  if (!state.scenarioBuilding) return;
  applyScenarioFeatures([feature]);
  if (pendingAssistantMapStep("scenario_road") || pendingAssistantMapStep("matsim_road")) {
    const step = pendingAssistantMapStep("matsim_road") ? "matsim_road" : "scenario_road";
    returnAssistantMapInput(
      step,
      `Affected road ${feature?.properties?.entity_id || feature?.properties?.key || "selected"} was added to the disruption.`,
    );
  }
}

function handleMapFeature(feature) {
  if (state.scenarioBuilding) {
    scenarioFeatureSelected(feature);
    return;
  }
  if (state.routeBuilding) {
    routeFeatureSelected(feature);
    return;
  }
  if (state.recoveryBuilding) {
    recoveryFeatureSelected(feature);
    return;
  }
  inspectFeature(feature);
}

async function scenarioAreaSelected(bbox) {
  if (!state.project) return;
  state.scenarioAreaLoading = true;
  $("#scenario-builder").setAttribute("aria-busy", "true");
  $("#scenario-pick-roads").disabled = true;
  $("#scenario-draw-area").disabled = true;
  renderScenarioDraft();
  setMapBanner("Reading the selected area", "Finding every mapped entity inside the rectangle.", true);
  try {
    const targets = await api.scenarioTargets(state.project.project_id, bbox);
    if (targets.truncated) {
      toast("Draw a smaller area", targets.message, true);
      setMapBanner("Area is too large", targets.message, true);
      return;
    }
    if (!targets.selected_count) {
      toast("Nothing selected", "The rectangle contains no mapped entity.", true);
      return;
    }
    applyScenarioFeatures(targets.feature_collection.features);
    state.map?.setMode("point");
    $("#scenario-draw-area").classList.remove("is-active");
    $("#scenario-pick-roads").classList.add("is-active");
    setMapBanner("Area added to the draft", `${formatNumber(targets.selected_count)} mapped changes are ready to review.`, true);
    if (pendingAssistantMapStep("scenario_area") || pendingAssistantMapStep("matsim_area")) {
      const step = pendingAssistantMapStep("matsim_area") ? "matsim_area" : "scenario_area";
      returnAssistantMapInput(
        step,
        `${formatNumber(targets.selected_count)} affected roads were captured from the drawn area.`,
      );
    }
  } finally {
    state.scenarioAreaLoading = false;
    $("#scenario-builder").removeAttribute("aria-busy");
    $("#scenario-pick-roads").disabled = false;
    $("#scenario-draw-area").disabled = false;
    renderScenarioDraft();
  }
}

function renderScenarioDraft() {
  const changes = [...state.scenarioDraft.values()];
  const counts = changes.reduce((output, change) => {
    output[change.effect] = (output[change.effect] || 0) + 1;
    return output;
  }, {});
  $("#scenario-draft-count").textContent = `${formatNumber(changes.length)} mapped ${changes.length === 1 ? "change" : "changes"}`;
  $("#scenario-change-mix").innerHTML = Object.entries(counts).map(([effect, count]) => `<span class="${escapeHtml(effect)}">${formatNumber(count)} ${escapeHtml(SCENARIO_EFFECT_LABELS[effect] || effect)}</span>`).join("");
  const visible = changes.slice(0, 12);
  $("#scenario-draft-list").innerHTML = visible.length
    ? `${visible.map((change) => {
        const label = change.effect === "closed" ? "Closed" : `${SCENARIO_EFFECT_LABELS[change.effect]} ${Math.round(Number(change.magnitude) * 100)}%`;
        return `<div class="scenario-draft-item"><strong>${escapeHtml(change.key.split("::", 2)[1] || change.key)}</strong><span>${escapeHtml(label)}</span><button type="button" data-remove-scenario-change="${escapeHtml(change.key)}" aria-label="Remove ${escapeHtml(change.key)}">×</button></div>`;
      }).join("")}${changes.length > visible.length ? `<div class="scenario-draft-more">${formatNumber(changes.length - visible.length)} more changes are saved in this draft</div>` : ""}`
    : '<div class="empty-state">Choose an effect, then click the map.</div>';
  const collection = { type: "FeatureCollection", features: [...state.scenarioDraftFeatures.values()] };
  state.map?.setScenarioDraftFeatures(collection);
  $("#scenario-undo").disabled = state.scenarioUndo.length === 0;
  $("#scenario-clear").disabled = changes.length === 0;
  $("#save-scenario").disabled = changes.length === 0 || state.scenarioAreaLoading;
  const activeStep = changes.length ? 3 : 2;
  $$(".scenario-builder-steps span").forEach((item, index) => item.classList.toggle("is-current", index + 1 === activeStep));
  scheduleNextActionCues();
}

function removeScenarioDraftChange(key) {
  if (!state.scenarioDraft.has(key)) return;
  pushScenarioUndo([key]);
  state.scenarioDraft.delete(key);
  state.scenarioDraftFeatures.delete(key);
  renderScenarioDraft();
}

function undoScenarioDraft() {
  const previous = state.scenarioUndo.pop();
  if (!previous) return;
  previous.forEach((item) => {
    if (item.change) state.scenarioDraft.set(item.key, item.change); else state.scenarioDraft.delete(item.key);
    if (item.feature) state.scenarioDraftFeatures.set(item.key, item.feature); else state.scenarioDraftFeatures.delete(item.key);
  });
  renderScenarioDraft();
}

function clearScenarioDraft() {
  const keys = [...state.scenarioDraft.keys()];
  if (!keys.length) return;
  pushScenarioUndo(keys);
  state.scenarioDraft.clear();
  state.scenarioDraftFeatures.clear();
  renderScenarioDraft();
}

async function saveScenarioDraft({ assistantMatsimExecution = false, manualExecution = false } = {}) {
  if (!state.project || !state.scenarioDraft.size) return;
  const name = $("#scenario-name").value.trim();
  if (name.length < 2) {
    toast("Name this scenario", "Use a short name that another user can recognise later.", true);
    $("#scenario-name").focus();
    return;
  }
  const payload = {
    project_id: state.project.project_id,
    name,
    hazard_type: $("#scenario-hazard").value,
    base_timestamp: Number($("#scenario-time").value),
    changes: [...state.scenarioDraft.values()],
  };
  $("#save-scenario").disabled = true;
  const scenarioWait = beginWorkloadWait(workloadWaitProfile({
    analysis_type: "closure_scenario",
    top_k: payload.changes.length,
    selected_keys: payload.changes.map((item) => item.key),
  }, { requestedCount: payload.changes.length, changeCount: payload.changes.length, networkWide: true }),
  "Comparing baseline and disruption",
  "Recomputing the same uploaded network with your confirmed changes.");
  try {
    const saved = state.scenarioEditingId
      ? await api.updateScenario(state.scenarioEditingId, payload)
      : await api.createScenario(payload);
    if (scenarioWait.cancelled) {
      $("#save-scenario").disabled = false;
      return;
    }
    state.scenarios = await api.scenarios(state.project.project_id);
    if (!await scenarioWait.complete("Scenario result ready · preparing direct and indirect effects…")) {
      $("#save-scenario").disabled = false;
      return;
    }
    const pendingMatsimTask = !manualExecution && pendingAssistantTaskMatches("matsim")
      ? state.assistantPendingTask
      : null;
    state.activeScenarioId = saved.scenario_id;
    closeScenarioBuilder({ restore: false });
    renderScenarioControls();
    await selectScenario(saved.scenario_id, { focus: !pendingMatsimTask });
    if (!pendingMatsimTask) {
      $("#scenario-impact-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (!manualExecution && pendingAssistantTaskMatches("scenario")) {
      updateAssistantActivity("execute", "The confirmed disruption was evaluated", "complete");
      updateAssistantActivity("verify", "Saving the scenario and preparing its map evidence", "active");
      appendAssistantScenarioResult(saved);
      updateAssistantActivity("verify", "Scenario map, impact summary and Project Memory are ready", "complete");
    }
    if (pendingMatsimTask && !assistantMatsimExecution) {
      pendingMatsimTask.scenarioId = saved.scenario_id;
      pendingMatsimTask.phase = nextAssistantMatsimAgentPhase(pendingMatsimTask);
      const confirmation = `MATSim disruption confirmed: ${saved.name || "saved disruption"}.`;
      appendAssistantMessage("assistant", confirmation);
      state.assistantHistory.push({ role: "assistant", content: confirmation });
      await persistAssistantChatMessage("assistant", confirmation, {
        kind: "workflow_input",
        task_type: "matsim",
        scenario_id: saved.scenario_id,
        project_id: currentProjectId(),
      });
      showAssistantPendingTaskStep();
      setAssistantOpen(true, "llm");
      toast(
        "MATSim scenario ready",
        assistantMatsimUsesUploadedDemand() || pendingMatsimTask.agentCount
          ? "Return to Copilot and run MATSim."
          : "Return to Copilot and choose 50, 100, 150 or 200 agents.",
      );
      return saved;
    }
    toast("Scenario saved", state.scenarioEngine === "matsim" ? "Choose the agent count, then press Run MATSim." : "The baseline and disruption are now shown together.");
    return saved;
  } catch (error) {
    scenarioWait.fail();
    toast("Scenario could not be saved", error.message, true);
    $("#save-scenario").disabled = false;
    return null;
  } finally {
    scenarioWait.close();
  }
}

function duplicateScenario(scenario = activeScenario()) {
  if (!scenario) return;
  openScenarioBuilder(scenario, { duplicate: true });
}

async function deleteScenario(scenario = activeScenario()) {
  if (!scenario) return;
  if (!window.confirm(`Delete “${scenario.name}”? The uploaded project and analyses will not be changed.`)) return;
  try {
    const deletingActive = state.activeScenarioId === scenario.scenario_id;
    await api.deleteScenario(scenario.scenario_id);
    if (deletingActive) state.activeScenarioId = null;
    state.scenarios = await api.scenarios(state.project.project_id);
    renderScenarioControls();
    activateScenario(deletingActive ? null : state.activeScenarioId);
    renderScenarioLibrary($("#scenario-search").value);
    toast("Scenario removed", "The uploaded data remain unchanged.");
  } catch (error) {
    toast("Scenario could not be removed", error.message, true);
  }
}

function renderScenarioLibrary(filter = "") {
  const list = $("#scenario-library-list");
  if (!list) return;
  const cleaned = String(filter || "").trim().toLowerCase();
  const scenarios = state.scenarios.filter((scenario) => !cleaned || scenario.name.toLowerCase().includes(cleaned));
  list.innerHTML = scenarios.length
    ? scenarios.map((scenario) => {
        const affected = scenario.result?.affected_count || scenario.changes?.length || 0;
        return `<article class="scenario-library-card"><div><strong>${escapeHtml(scenario.name)}</strong><small>${formatNumber(affected)} mapped changes · ${escapeHtml(scenario.result?.timestamp_label || "selected baseline")}</small></div><div class="scenario-card-actions"><button type="button" data-open-scenario="${escapeHtml(scenario.scenario_id)}">Open</button><button type="button" data-edit-scenario="${escapeHtml(scenario.scenario_id)}">Edit</button><button type="button" data-copy-scenario="${escapeHtml(scenario.scenario_id)}">Copy</button><button type="button" data-delete-scenario="${escapeHtml(scenario.scenario_id)}">Delete</button></div><span>${escapeHtml(hazardLabel(scenario.hazard_type))}</span></article>`;
      }).join("")
    : '<div class="empty-state">No matching scenarios. Build one from the map.</div>';
  $("#scenario-library-count").textContent = `${scenarios.length} of ${state.scenarios.length} saved scenario${state.scenarios.length === 1 ? "" : "s"}`;
}

function openScenarioLibrary() {
  $("#scenario-search").value = "";
  renderScenarioLibrary();
  if (!$("#scenario-library-dialog").open) $("#scenario-library-dialog").showModal();
}

let nextActionCueFrame = 0;

function clearNextActionCues() {
  $$(".next-action-cue").forEach((element) => {
    element.classList.remove("next-action-cue");
    element.removeAttribute("data-next-action");
    element.removeAttribute("aria-description");
  });
}

function markNextAction(targets, label) {
  const list = Array.isArray(targets) ? targets : [targets];
  list.forEach((element) => {
    if (!element || element.disabled || element.closest(".is-hidden")) return;
    element.classList.add("next-action-cue");
    element.dataset.nextAction = label;
    element.setAttribute("aria-description", `Required next step: ${label}`);
  });
}

function refreshNextActionCues() {
  clearNextActionCues();
  if (!state.project || state.currentView !== "explore") return;

  const map = $("#disaster-map");
  if (state.scenarioBuilding) {
    if (state.scenarioAreaLoading) return;
    if (!state.scenarioDraft.size) {
      const drawing = $("#scenario-draw-area")?.classList.contains("is-active");
      markNextAction($(".scenario-selection-tools"), "Choose roads or an area");
      markNextAction(map, drawing ? "Drag an area here" : "Click affected roads here");
    } else {
      markNextAction($("#save-scenario"), "Save and see impact");
    }
    return;
  }

  if (state.recoveryBuilding) {
    const payload = recoveryPayloadFromSource();
    if (!payload.selectedKeys.length) {
      markNextAction($(".recovery-selection-tools"), "Choose disrupted roads or areas");
      markNextAction(map, state.recoveryMode === "area" ? "Drag a disrupted area here" : "Click disrupted roads here");
    } else {
      markNextAction($("#run-recovery"), "Run recovery ranking");
    }
    return;
  }

  if (state.routeBuilding) {
    if (!state.routeOrigin) {
      markNextAction($("#route-origin-card"), "Choose the start on the map");
      markNextAction(map, "Click the route start here");
    } else if (!state.routeDestination) {
      markNextAction($("#route-destination-card"), "Choose the destination on the map");
      markNextAction(map, "Click the destination here");
    } else if ($("#route-scenario")?.value === "__quick__" && !state.routeDraft.size) {
      markNextAction($(".route-disruption-tools"), "Choose affected roads or an area");
      markNextAction(map, state.routeDisruptionMode === "area" ? "Drag the affected area here" : "Click affected roads here");
    } else {
      markNextAction($("#run-route"), "Calculate route options");
    }
    return;
  }

  if (state.stressBuilding) {
    markNextAction($("#run-stress"), "Run the resilience test");
    return;
  }

  if (state.analysisScopeMode === "selection" && !hasAnalysisMapSelection()) {
    markNextAction($(".selection-tools"), "Choose on the map");
    markNextAction($("#disaster-map"), "Click a location or draw an area");
    return;
  }

  const contract = state.commandContract;
  if (contract?.ready) {
    if (!state.currentResult || state.resultStale) markNextAction($("#confirm-analysis"), "Confirm and analyse");
    return;
  }
  if (contract) {
    const missing = (contract.missing || []).join(" ").toLowerCase();
    if (/select|draw|map|area|road|link|point|location|scope/.test(missing)) {
      markNextAction($(".selection-tools"), "Choose on the map");
      markNextAction(map, "Click a location or draw an area");
    }
    return;
  }
  if ($("#command-input")?.value.trim()) markNextAction($("#interpret-command"), "Preview analysis");
}

function scheduleNextActionCues() {
  window.cancelAnimationFrame(nextActionCueFrame);
  nextActionCueFrame = window.requestAnimationFrame(refreshNextActionCues);
}

function updateReadiness() {
  const value = $("#readiness-value");
  const copy = $("#readiness-copy");
  if (!state.project) {
    value.textContent = "Start";
    copy.textContent = "Choose data";
    scheduleNextActionCues();
    return;
  }
  if (state.commandContract?.ready) {
    value.textContent = "Ready";
    copy.textContent = "Confirm analysis";
  } else if (state.selectedKeys.size || state.bbox) {
    value.textContent = "Selected";
    copy.textContent = "Ask a question";
  } else {
    value.textContent = "Explore";
    copy.textContent = "Map and time ready";
  }
  scheduleNextActionCues();
}

function setFlowStage(stage, finished = false) {
  $$(".simple-flow li").forEach((item, index) => {
    const number = index + 1;
    item.classList.toggle("is-current", !finished && number === stage);
    item.classList.toggle("is-complete", finished || number < stage);
  });
}

function initialiseCinematicIntro() {
  const intro = $("#cinematic-intro");
  const launchPath = /^\/launch(?:-|=)[^/]+\/?$/i.test(window.location.pathname);
  const launchRequested = launchPath
    || new URLSearchParams(window.location.search).has("intro")
    || document.documentElement.classList.contains("twm-intro-pending");
  if (!intro || !launchRequested) return;
  document.documentElement.classList.add("twm-intro-pending");
  const appShell = $(".app-shell");
  const enter = $("#intro-enter");
  const skip = $("#intro-skip");
  let finished = false;

  const finish = () => {
    if (finished) return;
    finished = true;
    intro.classList.add("is-leaving");
    try {
      const cleanUrl = new URL(window.location.href);
      cleanUrl.searchParams.delete("intro");
      if (/^\/launch(?:-|=)[^/]+\/?$/i.test(cleanUrl.pathname)) cleanUrl.pathname = "/";
      window.history.replaceState({}, "", `${cleanUrl.pathname}${cleanUrl.search}${cleanUrl.hash}`);
    } catch (_) { /* The launch token is harmless if history is unavailable. */ }
    window.setTimeout(() => {
      intro.setAttribute("aria-hidden", "true");
      document.documentElement.classList.remove("twm-intro-pending");
      document.body.classList.remove("intro-active");
      if (appShell) appShell.inert = false;
    }, 620);
  };

  intro.setAttribute("aria-hidden", "false");
  document.body.classList.add("intro-active");
  if (appShell) appShell.inert = true;
  window.requestAnimationFrame(() => intro.classList.add("is-visible"));
  enter?.addEventListener("click", finish, { once: true });
  skip?.addEventListener("click", finish, { once: true });
  window.setTimeout(() => enter?.focus({ preventScroll: true }), 420);
}

async function initialise() {
  initialiseCinematicIntro();
  bindInteractions();
  try {
    state.map = new DisasterMap("disaster-map", {
      onFeature: handleMapFeature,
      onArea: areaSelected,
      onAreaProgress: (message) => setMapBanner("Area selection", message, true),
      onViewport: viewportChanged,
    });
    await state.map.initialise();
    state.bootstrap = await api.bootstrap();
    state.assistantStatus = await api.assistantStatus(state.assistantSessionId).catch(() => ({ status: "local_ready", api_configured: false, providers: [] }));
    state.projects = (state.bootstrap.projects || []).filter((project) => project.retention === "persistent");
    $("#footer-version").textContent = state.bootstrap.version;
    const registryChip = $(".registry-chip");
    if (registryChip) registryChip.textContent = `Model registry v${state.bootstrap.model_registry_version || state.bootstrap.version}`;
    state.modelRegistry = state.bootstrap.models || [];
    renderModels(state.modelRegistry);
    await loadReleaseScientificValidation();
    renderAssistantStatus();
    $("#system-status").classList.add("is-ready");
    $("#system-status").innerHTML = "<span></span>Analysis engine ready";
    renderNoProject();
  } catch (error) {
    toast("Workspace could not start", error.message, true);
    $("#system-status").innerHTML = "<span></span>Engine unavailable";
  } finally {
    setLoading(false);
  }
}

function renderNoProject() {
  stopPlayback();
  stopMatsimPlayback();
  state.mapRequestId += 1;
  state.inspectorRequestId += 1;
  state.scenarioRequestId += 1;
  state.project = null;
  state.modelRecommendations = null;
  state.mapView = "points";
  state.networkMapping = null;
  state.matsimReadiness = null;
  state.matsimJobId = null;
  state.matsimFrames = [];
  state.matsimFrameIndex = 0;
  state.matsimTransitionProgress = 0;
  state.matsimResult = null;
  state.mapCollection = null;
  state.mapCompleting = false;
  state.mapViewportQueries = false;
  state.map?.setFeatures(null, "points");
  state.map?.setMatsimFeatures(null);
  state.scenarios = [];
  state.activeScenarioId = null;
  state.scenarioBuilding = false;
  state.scenarioEditingId = null;
  state.scenarioDraft.clear();
  state.scenarioDraftFeatures.clear();
  state.routeBuilding = false;
  state.routePickRole = null;
  state.routeOrigin = null;
  state.routeDestination = null;
  state.map?.setScenarioFeatures(null);
  state.map?.setScenarioDraftFeatures(null);
  document.body.classList.remove("is-scenario-building");
  document.body.classList.remove("is-route-building");
  setHazardAtmosphere(null);
  document.body.classList.add("is-no-project");
  $("#context-project").textContent = "No project selected";
  $("#context-detail").textContent = "Start with one CSV or combine several files. Nothing is loaded automatically.";
  $("#context-hazard").textContent = "Your data first";
  $("#keep-current-project").classList.add("is-hidden");
  renderBasicHistorySummary(null);
  if ($("#workspace-kind-label")) $("#workspace-kind-label").textContent = "Transport network workspace";
  if ($("#map-kind-label")) $("#map-kind-label").textContent = "Transport network view";
  if ($("#disaster-map")) $("#disaster-map").setAttribute("aria-label", "Interactive disaster transport network view");
  $("#project-picker-label").textContent = "Open existing project";
  $("#explore-heading").classList.add("is-hidden");
  $("#simple-flow").classList.add("is-hidden");
  $("#start-panel").classList.remove("is-hidden");
  $("#explore-layout").classList.add("is-hidden");
  $("#scenario-hub").classList.add("is-hidden");
  $("#scenario-builder").classList.add("is-hidden");
  $("#route-builder").classList.add("is-hidden");
  $("#scenario-impact-panel").classList.add("is-hidden");
  $("#start-existing").disabled = state.projects.length === 0;
  $("#map-empty").classList.remove("is-hidden");
  $("#time-slider").disabled = true;
  $("#play-time").disabled = true;
  $("#playback-speed").disabled = true;
  configureNetworkControls();
  renderDataView();
  setFlowStage(1);
  updateReadiness();
  updateAssistantContext();
  renderModels(state.modelRegistry);
  renderResearchWorkspace();
  if (state.releaseScientificValidation) renderScientificValidation(state.releaseScientificValidation, "release");
}

async function loadProject(project) {
  if (!project?.project_id) return;
  stopPlayback();
  stopMatsimPlayback();
  state.mapRequestId += 1;
  state.inspectorRequestId += 1;
  state.scenarioRequestId += 1;
  invalidateAnalysisReview("The active project changed.");
  setLoading(true, "Opening the transport-network project", "Preparing its map or topology layout, timeline and available analyses.");
  try {
    state.project = project.summary ? project : await api.getProject(project.project_id);
    if (!state.project.database_path) state.project = await api.getProject(project.project_id);
    state.completedResults = new Map();
    state.modelRecommendations = await api.modelRecommendations(state.project.project_id).catch(() => null);
    state.preferredModelId = null;
    renderModels(state.modelRegistry);
    if (state.releaseScientificValidation) renderScientificValidation(state.releaseScientificValidation, "release");
    state.mapView = "points";
    state.networkMapping = await api.networkMappingStatus(state.project.project_id).catch(() => ({
      available: false,
      ready: false,
      status: "unavailable",
      reason: "Link-view availability could not be checked.",
    }));
    state.matsimReadiness = await api.matsimReadiness(state.project.project_id).catch((error) => ({
      status: "unavailable",
      tier: "unavailable",
      can_run: false,
      label: "MATSim readiness unavailable",
      explanation: error.message,
    }));
    if (state.project.summary?.preferred_map_view === "links" && state.networkMapping?.ready) {
      state.mapView = "links";
    }
    if (
      state.project.retention === "persistent"
      && !state.projects.some((item) => item.project_id === state.project.project_id)
    ) state.projects.unshift(state.project);
    state.metric = chooseAvailableMetric(state.project.summary);
    state.mapCollection = null;
    state.mapCompleting = false;
    state.mapViewportQueries = false;
    state.timeValues = state.project.summary.time_values || [0];
    state.timeIndex = Math.max(0, state.timeValues.length - 1);
    state.selectedKeys.clear();
    state.keptKeys.clear();
    state.focusedKey = null;
    state.bbox = null;
    state.analysisScopeMode = "whole";
    state.commandContract = null;
    state.currentResult = null;
    state.resultStale = false;
    state.analysisCollection = null;
    state.scenarios = [];
    state.activeScenarioId = null;
    state.scenarioBuilding = false;
    state.scenarioEditingId = null;
    state.scenarioDraft.clear();
    state.scenarioDraftFeatures.clear();
    state.routeBuilding = false;
    state.routePickRole = null;
    state.routeOrigin = null;
    state.routeDestination = null;
    state.stressBuilding = false;
    state.projectMemory = null;
    state.researchReports = [];
    state.researchReport = null;
    state.researchJobId = null;
    state.researchFigureSvgs = new Map();
    $("#analysis-review").classList.add("is-hidden");
    $("#result-panel").classList.add("is-hidden");
    $("#entity-inspector").classList.add("is-hidden");
    state.map.clearArea();
    state.map.setSelectedKeys([]);
    state.map.setFeatures({ type: "FeatureCollection", features: [] }, state.mapView);
    state.map.setAnalysisFeatures(null);
    state.map.setScenarioFeatures(null);
    state.map.setScenarioDraftFeatures(null);
    state.map.setMatsimFeatures(null);
    document.body.classList.remove("is-scenario-building");
    document.body.classList.remove("is-route-building");
    document.body.classList.remove("is-stress-building");
    $("#route-builder").classList.add("is-hidden");
    $("#stress-builder").classList.add("is-hidden");
    document.body.classList.remove("is-no-project");
    $("#start-panel").classList.add("is-hidden");
    $("#explore-heading").classList.remove("is-hidden");
    $("#simple-flow").classList.remove("is-hidden");
    $("#scenario-hub").classList.remove("is-hidden");
    $("#scenario-impact-panel").classList.add("is-hidden");
    $("#explore-layout").classList.remove("is-hidden");
    $("#project-picker-label").textContent = "Change project";
    state.map.resize();
    configureProjectControls();
    renderMatsimReadiness();
    renderDataView();
    state.suppressViewport = true;
    state.map.setExtent(state.project.summary.extent, true);
    await refreshMap(null);
    // The first detector batch is already on the map. Reveal it immediately
    // while secondary project records continue loading.
    $("#map-empty").classList.add("is-hidden");
    setLoading(false);
    await refreshScenarios();
    window.setTimeout(() => { state.suppressViewport = false; }, 900);
    switchView("explore");
    setFlowStage(2);
    await refreshRuns();
    await refreshResearchReports({ loadLatest: true }).catch(() => renderResearchWorkspace());
    toast("Project ready", `${state.project.name} is available in the transport-network workspace.`);
    updateAssistantContext();
  } finally {
    setLoading(false);
  }
}

function chooseAvailableMetric(summary) {
  return availableMetricOptions(summary)[0]?.key || "primary";
}

function renderBasicHistorySummary(summary = state.project?.summary) {
  const copy = $("#history-basic-copy");
  if (!copy) return;
  if (!summary) {
    copy.textContent = "Upload data to see records, locations and measures.";
    return;
  }
  const metricOptions = availableMetricOptions(summary);
  const metricNames = metricOptions.length
    ? metricOptions.map((option) => option.label.split(" · ")[0]).join(", ")
    : "Uploaded measure";
  const observationCount = Number(summary.observation_count ?? 0);
  const entityCount = Number(summary.mapped_entity_count ?? summary.entity_count ?? 0);
  const entityLabel = projectUsesLinkTopology(summary) ? "road links" : "locations";
  const timeCount = Number(summary.time_value_count ?? summary.time_values?.length ?? 0);
  const hasTimeline = timeCount > 1 && summary.time_start && summary.time_end;
  const timeValue = hasTimeline
    ? `${formatNumber(timeCount)} time points`
    : "Single snapshot";
  const timeDetail = hasTimeline
    ? `${summary.time_start} → ${summary.time_end}`
    : "No time trend is created";
  copy.textContent = `${formatNumber(observationCount)} records · ${formatNumber(entityCount)} ${entityLabel} · ${timeValue} · ${metricNames}`;
}

function configureProjectControls() {
  const summary = state.project.summary;
  renderBasicHistorySummary(summary);
  const geographic = summary.transport_structure?.geographic !== false;
  if ($("#workspace-kind-label")) $("#workspace-kind-label").textContent = geographic ? "Geographic map workspace" : "Non-geographic network workspace";
  if ($("#map-kind-label")) $("#map-kind-label").textContent = geographic ? "Geographic transport map" : "Non-geographic topology layout";
  if ($("#disaster-map")) $("#disaster-map").setAttribute("aria-label", geographic ? "Interactive geographic disaster transport map" : "Interactive non-geographic disaster transport topology layout");
  $("#context-project").textContent = state.project.name;
  const temporaryProject = state.project.retention === "temporary";
  $("#context-detail").textContent = temporaryProject
    ? "Temporary session project · not saved in Open existing project"
    : `${formatNumber(summary.mapped_entity_count ?? summary.entity_count)} mapped entities · ${summary.time_start} to ${summary.time_end}`;
  $("#keep-current-project").classList.remove("is-hidden");
  $("#context-hazard").textContent = String(state.project.hazard_type || "other").replaceAll("_", " ");
  setHazardAtmosphere(state.project.hazard_type || "other");
  $("#map-title").textContent = state.project.name;
  $("#geometry-note").textContent = geometryNoteForProject(summary);
  $("#select-point-label").textContent = projectUsesLinkTopology(summary) ? "Select link points" : "Select map points";
  const select = $("#metric-select");
  const metricOptions = availableMetricOptions(summary);
  select.innerHTML = metricOptions.map((option) => `<option value="${escapeHtml(option.key)}">${escapeHtml(option.label)}</option>`).join("");
  if (!metricOptions.some((option) => option.key === state.metric)) state.metric = metricOptions[0]?.key || "primary";
  select.value = state.metric;
  configureNetworkControls();
  $$('[data-capability]').forEach((button) => {
    const capability = summary.capabilities?.[button.dataset.capability] || "unavailable";
    button.disabled = capability === "unavailable";
    button.classList.toggle("is-conditional", capability === "conditional");
    button.title = capability === "unavailable"
      ? "The uploaded fields do not support this analysis. See Data for requirements."
      : capability === "conditional" ? "Available with explicit limitations shown before running." : "Available for this project.";
  });
  const slider = $("#time-slider");
  slider.min = "0";
  slider.max = String(Math.max(0, state.timeValues.length - 1));
  slider.value = String(state.timeIndex);
  slider.disabled = state.timeValues.length <= 1;
  $("#play-time").disabled = state.timeValues.length <= 1;
  $("#playback-speed").disabled = state.timeValues.length <= 1;
  $("#playback-speed").value = String(state.playbackSpeed);
  $("#time-label").textContent = currentTimeLabel();
  const timelineNote = $("#timeline-sampling-note");
  timelineNote.classList.toggle("is-hidden", !summary.timeline_sampled);
  timelineNote.textContent = summary.timeline_sampled
    ? `Control shows ${formatNumber(state.timeValues.length)} evenly spaced points from ${formatNumber(summary.time_value_count)} uploaded times; first and last are included.`
    : "";
  setMapBanner("Historical data", `${mappedObservationLabel(summary.mapped_entity_count ?? summary.entity_count, summary)} · ${currentTimeLabel()}`);
  renderMapLegend();
  renderSelectionSummary();
  updateReadiness();
}

function configureNetworkControls() {
  const mapping = state.networkMapping || {};
  const points = $("#view-points");
  const links = $("#view-links");
  if (!points || !links) return;
  const hasProject = Boolean(state.project);
  const linksAvailable = Boolean(mapping.ready || mapping.available);
  const pointActive = !hasProject || state.mapView !== "links";
  points.disabled = !hasProject;
  links.disabled = !hasProject || !linksAvailable;
  points.classList.toggle("is-active", pointActive);
  links.classList.toggle("is-active", hasProject && state.mapView === "links");
  points.setAttribute("aria-pressed", String(pointActive));
  links.setAttribute("aria-pressed", String(hasProject && state.mapView === "links"));
  $("#view-links-label").textContent = mapping.ready
    ? (mapping.exact_geometry ? "Road links" : "Mapped links")
    : linksAvailable ? "Map to links" : "No link fields";
  $("#network-view-status").textContent = !hasProject
    ? "Open data first"
    : mapping.ready
      ? `${formatNumber(mapping.mapped_link_count)} links ready · switch any time`
      : mapping.available
        ? `${formatNumber(mapping.source_link_count)} topology links · create once`
        : mapping.reason || "From / To node fields are required";
  if ($("#select-point-label")) {
    $("#select-point-label").textContent = state.mapView === "links"
      ? "Select mapped links"
      : projectUsesLinkTopology() ? "Select link points" : "Select map points";
  }
  renderMapLegend();
}

function openNetworkMappingDialog() {
  if (!state.project) return toast("Choose data first", "Open a transport-network project before creating a link view.", true);
  if (state.networkMapping?.ready) {
    switchMapView("links").catch((error) => toast("Link view unavailable", error.message, true));
    return;
  }
  if (!state.networkMapping?.available) {
    toast("Link mapping unavailable", state.networkMapping?.reason || "Confirm From node and To node fields during import.", true);
    return;
  }
  $("#network-dialog-summary").textContent = `${formatNumber(state.networkMapping.source_link_count)} links have usable topology IDs and coordinates. The source CSV and point view stay unchanged.`;
  const dialog = $("#network-dialog");
  if (!dialog.open) dialog.showModal();
}

async function buildAndOpenNetworkView() {
  if (!state.project || !state.networkMapping?.available) return;
  const button = $("#confirm-network-mapping");
  button.disabled = true;
  $("#network-dialog").close();
  setLoading(true, "Mapping points to a link network", "Estimating topology-node locations and saving the derived view.");
  try {
    state.networkMapping = await api.buildNetworkMapping(state.project.project_id);
    state.mapView = "links";
    configureNetworkControls();
    await refreshMatsimReadiness();
    renderSelectionSummary();
    await refreshMap(null);
    await refreshOpenInspector();
    const excluded = Number(state.networkMapping.excluded_link_count || 0);
    toast(
      "Mapped link view saved",
      `${formatNumber(state.networkMapping.mapped_link_count)} links created${excluded ? ` · ${formatNumber(excluded)} degenerate links excluded` : ""}.`,
    );
  } catch (error) {
    state.mapView = "points";
    configureNetworkControls();
    toast("Link mapping could not be created", error.message, true);
  } finally {
    button.disabled = false;
    setLoading(false);
  }
}

async function switchMapView(view) {
  const target = view === "links" ? "links" : "points";
  if (!state.project || target === state.mapView) return;
  if (target === "links" && !state.networkMapping?.ready) {
    openNetworkMappingDialog();
    return;
  }
  stopPlayback();
  state.mapView = target;
  configureNetworkControls();
  renderSelectionSummary();
  await refreshMap(null);
  await refreshOpenInspector();
}

async function refreshMap(bbox = null) {
  if (!state.project) return;
  const requestId = ++state.mapRequestId;
  state.mapCompleting = false;
  const projectId = state.project.project_id;
  const mapRequest = state.mapView === "links" ? api.networkMap : api.map;
  const requestedView = state.mapView;
  const firstPaint = !state.mapCollection?.features?.length && !bbox;
  const loader = $("#map-local-loading");
  loader?.classList.toggle("is-hidden", !firstPaint);
  if ($("#map-loading-copy")) $("#map-loading-copy").textContent = firstPaint
    ? "Drawing a fast first batch, then completing the visible time slice…"
    : "Updating the selected measure and time…";
  const applyCollection = (collection) => {
    if (requestId !== state.mapRequestId || state.project?.project_id !== projectId || state.mapView !== requestedView) return false;
    state.mapCollection = collection;
    if (!bbox) state.mapViewportQueries = Boolean(collection.truncated);
    state.map.setFeatures(collection, state.mapView);
    state.map.setSelectedKeys([...state.selectedKeys]);
    $("#time-label").textContent = collection.timestamp_label;
    $("#geometry-note").textContent = collection.geometry_note || geometryNoteForProject(state.project.summary);
    const count = collection.features.length;
    setMapBanner(
      state.mapView === "links" ? "Mapped link view" : "Historical data",
      `${mappedObservationLabel(count)} shown · ${collection.timestamp_label}${collection.truncated ? ` · deterministic sample of ${formatNumber(collection.total_count)}; zoom for local detail` : ""}`,
    );
    renderMapLegend(collection);
    return true;
  };
  try {
    const collection = await mapRequest(projectId, {
      metric: state.metric,
      timestamp: currentTime(),
      bbox,
      limit: firstPaint ? 800 : (state.playing ? 5000 : 9000),
    });
    if (!applyCollection(collection)) return;
    loader?.classList.add("is-hidden");
    if (firstPaint && collection.truncated && requestId === state.mapRequestId) {
      state.mapCompleting = true;
      // Fill the larger display sample in the background without hiding or
      // freezing the 800 detector points already shown.
      void (async () => {
        await new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolve)));
        const complete = await mapRequest(projectId, { metric: state.metric, timestamp: currentTime(), bbox, limit: 9000 });
        applyCollection(complete);
      })().catch((error) => {
        if (requestId === state.mapRequestId) toast("Map detail could not finish", error.message, true);
      }).finally(() => {
        if (requestId === state.mapRequestId) state.mapCompleting = false;
      });
    }
  } finally {
    if (requestId === state.mapRequestId) loader?.classList.add("is-hidden");
  }
}

function viewportChanged(bbox) {
  if (!state.project || state.suppressViewport || state.map.mode === "area") return;
  if (!state.mapViewportQueries) return;
  window.clearTimeout(state.viewportTimer);
  // A user pan or zoom supersedes the optional background completion of the
  // first global sample. refreshMap increments the request token, so the old
  // response cannot flash over the new viewport and no blocking map overlay
  // is shown while the user moves the map.
  state.viewportTimer = window.setTimeout(() => refreshMap(bbox).catch((error) => toast("Map refresh failed", error.message, true)), 650);
}

function setMapBanner(label, message, area = false) {
  const banner = $("#map-banner");
  banner.classList.toggle("is-area", area);
  banner.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(message)}</strong>`;
}

function renderMapLegend(collection = state.mapCollection) {
  const legend = $("#map-legend");
  if (!legend) return;
  const markClass = state.mapView === "links" ? "line-swatch" : "";
  const type = state.currentResult?.analysis_type;
  if (!state.resultStale && type === "route_detour_analysis") {
    const routeVisual = state.currentResult?.visual || {};
    if (routeVisual.mode === "route_options") {
      const routes = routeVisual.routes || [];
      const entries = routes.map((route) => {
        const colour = /^#[0-9a-f]{6}$/i.test(String(route.colour || "")) ? route.colour : "#1464d2";
        return `<span><i class="line-swatch" style="background:${colour}"></i>${escapeHtml(route.label)}</span>`;
      }).join("");
      const availability = Number(routeVisual.returned_count || 0) < Number(routeVisual.requested_count || 0)
        ? `<span>Requested ${formatNumber(routeVisual.requested_count)} · ${formatNumber(routeVisual.returned_count)} feasible</span>`
        : "";
      legend.innerHTML = `${entries}<span><i class="route-start"></i>Start</span><span><i class="route-end"></i>Destination</span>${availability}<span>Parallel colours share a road; coloured branches are different road sequences</span>`;
      return;
    }
    if (routeVisual.mode === "route_compare") {
      const routeRoles = new Set((state.currentResult?.feature_collection?.features || [])
        .filter((feature) => feature.geometry?.type === "LineString")
        .map((feature) => feature.properties?.route_role));
      const entries = [
        ["baseline_route", "#2563eb", "Baseline route"],
        ["comparison_route", "#f28c28", "Recommended route 1"],
        ["alternative_route", "#ffd43b", "Recommended route 2"],
      ].filter(([role]) => routeRoles.has(role));
      legend.innerHTML = `${entries.map(([, colour, label]) => `<span><i class="line-swatch" style="background:${colour}"></i>${label}</span>`).join("")}<span><i class="route-start"></i>Start</span><span><i class="route-end"></i>Destination</span>${entries.length < 3 ? "<span>Only feasible routes are shown</span>" : ""}`;
      return;
    }
    const variants = routeVisual.variants || [];
    const variantLegend = variants.map((variant) => {
      const colour = /^#[0-9a-f]{6}$/i.test(String(variant.colour || "")) ? variant.colour : "#f4b400";
      return `<span><i class="line-swatch" style="background:${colour}"></i>${escapeHtml(variant.label)}</span>`;
    }).join("");
    legend.innerHTML = `<span><i class="line-swatch" style="background:#2563eb"></i>Baseline route</span>${variantLegend}<span><i class="route-start"></i>Start</span><span><i class="route-end"></i>Destination</span>`;
    return;
  }
  if (state.matsimResult && state.matsimFrames.length) {
    legend.innerHTML = `<span><i style="background:#7f1734"></i>No-route agents at closure approach</span><span><i style="background:#b12f4e"></i>Agents on changed roads</span><span><i style="background:#f4b400"></i>Agents near disruption</span><span><i style="background:#1f8078"></i>Agents on spillover roads</span><span><i class="line-swatch" style="background:linear-gradient(90deg,#1f8078,#f2bf32,#b12f4e)"></i>Probe road entries</span><span>Smoothed continuous replay</span>`;
    return;
  }
  const scenario = activeScenario();
  if (scenario) {
    legend.innerHTML = `<span><i class="${markClass} scenario-direct"></i>Initial disrupted roads</span><span><i class="line-swatch scenario-indirect"></i>Indirect rerouting pressure</span><span>Gold is an affected-road highlight, not a criticality rank</span>`;
    return;
  }
  if (!state.resultStale && type === "critical_components") {
    legend.innerHTML = '<span><i class="line-swatch critical-link-highlight"></i>Identified critical roads</span><span>One yellow highlight · importance order is shown in the ranked table</span>';
    return;
  }
  if (!state.resultStale && type === "spatial_hotspots") {
    legend.innerHTML = isRegionalDpcResult(state.currentResult)
      ? '<span><i class="rank-gradient critical-area-gradient"></i>Highest-ranked area → greatest Regional DPC loss</span><span>Preprint-derived finite numerical screen</span>'
      : '<span><i class="rank-gradient critical-area-gradient"></i>Highest-ranked area → greatest observed mean degradation</span>';
    return;
  }
  if (metricSupportsSeverity()) {
    legend.innerHTML = `<span><i class="${markClass} normal"></i>Reference</span><span><i class="${markClass} moderate"></i>Degraded</span><span><i class="${markClass} severe"></i>Severe</span><span><i class="${markClass} quality"></i>Data issue</span>`;
  } else {
    const unit = collection?.unit || state.project?.summary?.metrics?.[state.metric]?.unit || "uploaded unit";
    legend.innerHTML = `<span><i class="rank-gradient critical-link-gradient"></i>Lower → higher ${escapeHtml(metricCopy(state.metric).observed)} (${escapeHtml(unit)})</span><span>Relative display range only · not disaster severity</span>`;
  }
}

async function inspectFeature(feature) {
  if (!state.project) return;
  const inspectorRequestId = ++state.inspectorRequestId;
  const inspectorProjectId = state.project.project_id;
  const properties = feature.properties || {};
  const key = String(properties.key);
  state.lastFeature = properties;
  state.focusedKey = key;
  state.selectedKeys = new Set([...state.keptKeys, key]);
  state.map.setSelectedKeys([...state.selectedKeys]);
  if (state.analysisScopeMode === "selection") invalidateAnalysisReview();
  renderSelectionSummary();
  if (pendingAssistantMapStep("analysis_point")) {
    returnAssistantMapInput(
      "analysis_point",
      `Map scope confirmed with ${properties.entity_id ? `entity ${properties.entity_id}` : "the selected map entity"}.`,
    );
  }
  const inspector = $("#entity-inspector");
  inspector.classList.remove("is-hidden");
  const isLink = featureUsesLinkTopology(properties);
  const isMappedLine = properties.geometry_kind === "inferred_topology_line";
  const coordinates = feature.geometry?.coordinates || [];
  const longitude = isMappedLine ? Number(properties.reference_longitude) : Number(coordinates[0]);
  const latitude = isMappedLine ? Number(properties.reference_latitude) : Number(coordinates[1]);
  const coordinateLabel = Number.isFinite(latitude) && Number.isFinite(longitude)
    ? `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`
    : "Unavailable";
  const copy = metricCopy(properties.metric);
  $("#entity-kicker").textContent = isMappedLine ? "Selected inferred network link" : isLink ? "Selected link reference point" : "Selected entity point";
  $("#entity-title").textContent = isLink
    ? `Road link ${properties.entity_id}`
    : `Mapped entity ${properties.entity_id}`;
  $("#entity-meaning").innerHTML = isMappedLine
    ? `<span class="entity-meaning-mark is-line" aria-hidden="true"></span><div><strong>Inferred topology line</strong><small>The line joins estimated From / To node locations. It carries this link's uploaded value, but it is not an exact road alignment.</small></div>`
    : isLink
      ? `<span class="entity-meaning-mark" aria-hidden="true"></span><div><strong>Road-link reference point</strong><small>The dot locates this link observation. It is not either end node and it is not the full road line.</small></div>`
    : `<span class="entity-meaning-mark" aria-hidden="true"></span><div><strong>Uploaded entity location</strong><small>This point comes from the uploaded latitude and longitude. No road link or topology is assumed.</small></div>`;
  const topology = $("#entity-topology");
  topology.classList.toggle("is-hidden", !isLink);
  topology.innerHTML = isLink ? `
    <div class="topology-item is-node"><i></i><small>From node</small><strong>${escapeHtml(properties.from_node)}</strong></div>
    <span class="topology-arrow" aria-hidden="true">→</span>
    <div class="topology-item is-link"><i></i><small>Road link</small><strong>ID ${escapeHtml(properties.entity_id)}</strong></div>
    <span class="topology-arrow" aria-hidden="true">→</span>
    <div class="topology-item is-node"><i></i><small>To node</small><strong>${escapeHtml(properties.to_node)}</strong></div>
  ` : "";
  $("#inspector-actions").classList.toggle("is-hidden", !isLink);
  $("#history-title").textContent = copy.history;
  const value = Number(properties.value);
  const baseline = Number(properties.baseline);
  const severity = Number(properties.severity);
  $("#entity-facts").innerHTML = [
    [copy.observed, Number.isFinite(value) ? `${formatNumber(value, 2)} ${properties.unit || ""}` : "Unavailable"],
    [copy.reference, Number.isFinite(baseline) ? `${formatNumber(baseline, 2)} ${properties.unit || ""}` : "Unavailable"],
    ["Relative degradation", Number.isFinite(severity) ? `${(severity * 100).toFixed(1)}%` : "Not defined"],
    ["Map coordinate", coordinateLabel],
    ...(isMappedLine ? [["Inferred line length", `${formatNumber(properties.inferred_line_length_m, 0)} m`]] : []),
    ["Data quality", properties.quality || "unknown"],
    ["Observed time", properties.timestamp || currentTimeLabel()],
  ].map(([term, description]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(description)}</dd></div>`).join("");
  try {
    const history = await api.history(state.project.project_id, properties.region_id, properties.entity_id, state.metric);
    if (inspectorRequestId !== state.inspectorRequestId || state.project?.project_id !== inspectorProjectId) return;
    drawHistory($("#history-chart"), history);
    $("#history-note").textContent = history.sampling_note || "red: observation · teal: internal reference";
  } catch (error) {
    toast("History unavailable", error.message, true);
  }
  updateReadiness();
}

async function refreshOpenInspector() {
  const inspector = $("#entity-inspector");
  if (!state.lastFeature?.key || inspector.classList.contains("is-hidden")) return;
  const refreshedFeature = (state.mapCollection?.features || []).find(
    (feature) => String(feature.properties?.key) === String(state.lastFeature.key),
  );
  if (refreshedFeature) {
    await inspectFeature(refreshedFeature);
    return;
  }
  inspector.classList.add("is-hidden");
  state.lastFeature = null;
  if (state.routeBuilding) {
    state.routeOrigin = null;
    state.routeDestination = null;
    state.routePickRole = "origin";
    renderRouteBuilder();
  }
}

function hasAnalysisMapSelection() {
  return Boolean(state.selectedKeys.size || state.bbox);
}

function analysisMapSelectionLabel() {
  const parts = [];
  if (state.selectedKeys.size) parts.push(`${formatNumber(state.selectedKeys.size)} selected`);
  if (state.bbox) parts.push("1 area");
  return parts.join(" + ") || "Choose on map";
}

function renderAnalysisScope() {
  const usingSelection = state.analysisScopeMode === "selection";
  const available = hasAnalysisMapSelection();
  $("#analysis-scope-control")?.classList.toggle("is-using-selection", usingSelection);
  $("#scope-whole")?.classList.toggle("is-active", !usingSelection);
  $("#scope-selection")?.classList.toggle("is-active", usingSelection);
  $("#scope-whole")?.setAttribute("aria-pressed", String(!usingSelection));
  $("#scope-selection")?.setAttribute("aria-pressed", String(usingSelection));
  if ($("#scope-selection-count")) $("#scope-selection-count").textContent = analysisMapSelectionLabel();
  if ($("#analysis-scope-label")) {
    $("#analysis-scope-label").textContent = usingSelection
      ? available ? `Map selection · ${analysisMapSelectionLabel()}` : "Map selection · waiting"
      : "Whole project";
  }
  if ($("#analysis-scope-note")) {
    $("#analysis-scope-note").textContent = usingSelection
      ? available ? "Only this explicit selection enters independent analyses." : "Choose roads, points or an area before previewing."
      : available ? "Map marks stay visible but do not enter independent analyses." : "All eligible uploaded data enter independent analyses.";
  }
  scheduleNextActionCues();
}

function setAnalysisScopeMode(mode, { invalidate = true, announce = false } = {}) {
  const next = mode === "selection" ? "selection" : "whole";
  if (state.analysisScopeMode === next) {
    renderAnalysisScope();
    return;
  }
  state.analysisScopeMode = next;
  if (invalidate) invalidateAnalysisReview("The calculation scope changed.");
  renderAnalysisScope();
  if (!announce) return;
  if (next === "selection") {
    setMapBanner(
      hasAnalysisMapSelection() ? "Map selection will be used" : "Choose the calculation area",
      hasAnalysisMapSelection() ? "Only the visible selected roads, points or area will enter the next independent analysis." : "Click map items or draw an area, then preview the analysis.",
      true,
    );
  } else {
    setMapBanner("Whole-project calculation", "Existing map marks remain visible but are excluded from the next independent analysis.", true);
  }
}

function commandRequestsMapSelection(text) {
  return /\b(selected|chosen|drawn area|current map|map selection|these roads|these links|these points|those roads|those links|this area|this region|that area|that region)\b|选中|所选|框选|当前地图|这些道路|这些边|这些点|该区域|这个区域/i.test(String(text || ""));
}

function commandRequestsWholeProject(text) {
  return /\b(whole|entire|all eligible|across the network|full network|full project|project-wide|network-wide|my data|uploaded data|current project)\b|全网|整个网络|全部数据|整个项目/i.test(String(text || ""));
}

function analysisContextForCommand(text) {
  if (commandRequestsWholeProject(text)) setAnalysisScopeMode("whole", { invalidate: false });
  else if (commandRequestsMapSelection(text)) setAnalysisScopeMode("selection", { invalidate: false });
  const useSelection = state.analysisScopeMode === "selection";
  return {
    ready: !useSelection || hasAnalysisMapSelection(),
    selected_keys: useSelection ? [...state.selectedKeys] : [],
    bbox: useSelection ? state.bbox : null,
  };
}

function renderSelectionSummary() {
  const summary = $("#selected-summary");
  const links = projectUsesLinkTopology();
  let title = "No map point or area selected";
  let detail = links
    ? state.mapView === "links"
      ? "Click an inferred link to inspect its two end nodes and historical evidence."
      : "Click a link reference point to inspect its two end nodes."
    : "Click an uploaded entity point or draw an area.";
  if (state.selectedKeys.size && state.bbox) {
    const noun = links ? `road link${state.selectedKeys.size === 1 ? "" : "s"}` : `mapped entit${state.selectedKeys.size === 1 ? "y" : "ies"}`;
    title = `${state.selectedKeys.size} ${noun} and one area selected`;
    detail = "Your next question will be grounded in both selections.";
  } else if (state.selectedKeys.size) {
    const noun = links ? `road link${state.selectedKeys.size === 1 ? "" : "s"}` : `mapped entit${state.selectedKeys.size === 1 ? "y" : "ies"}`;
    title = `${state.selectedKeys.size} ${noun} selected`;
    detail = links
      ? state.mapView === "links"
        ? "The selected entity is a road link shown as an inferred topology line."
        : "The circles are reference locations; the selected entities are road links."
      : "Use the selection to ground a geographic question.";
  } else if (state.bbox) {
    title = "One geographic area selected";
    detail = "Ask for an area summary or persistent change inside the area.";
  }
  summary.innerHTML = `<span class="summary-marker"></span><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></div>`;
  const list = $("#selected-list");
  const keys = [...state.selectedKeys];
  list.classList.toggle("is-hidden", !keys.length);
  list.innerHTML = keys.map((key) => {
    const [region, entity] = String(key).split("::", 2);
    return `<span><b>${escapeHtml(entity || key)}</b><small>${escapeHtml(region || "uploaded")}</small><button type="button" data-remove-selection="${escapeHtml(key)}" aria-label="Remove ${escapeHtml(entity || key)} from scenario">×</button></span>`;
  }).join("");
  renderAnalysisScope();
}

function areaSelected(bbox) {
  if (state.scenarioBuilding) {
    scenarioAreaSelected(bbox).catch((error) => toast("Area could not be added", error.message, true));
    return;
  }
  if (state.routeBuilding) {
    routeAreaSelected(bbox).catch((error) => toast("Affected area could not be added", error.message, true));
    return;
  }
  if (state.recoveryBuilding) {
    recoveryAreaSelected(bbox).catch((error) => toast("Recovery area could not be added", error.message, true));
    return;
  }
  state.bbox = bbox;
  setAnalysisScopeMode("selection", { invalidate: false });
  invalidateAnalysisReview();
  state.map.setMode("point");
  $("#select-area").classList.remove("is-active");
  $("#select-point").classList.add("is-active");
  renderSelectionSummary();
  setMapBanner("Area selected", "The rectangle is geographic and editable by clearing and drawing again.", true);
  if (pendingAssistantMapStep("analysis_area")) {
    returnAssistantMapInput("analysis_area", "The geographic calculation area was confirmed on the map.");
    return;
  }
  $("#command-input").value = "Summarise the uploaded transport observations inside this selected area.";
  interpretCurrentCommand();
}

async function interpretCurrentCommand() {
  if (!state.project) {
    toast("Choose data first", "Upload a dataset to open a transport-network project.", true);
    openUpload();
    return;
  }
  const text = $("#command-input").value.trim();
  if (!text) {
    toast("Write one question", "Use a prompt example or describe what you want to understand.", true);
    return;
  }
  const analysisContext = analysisContextForCommand(text);
  if (!analysisContext.ready) {
    toast("Choose the map scope", "Click roads or points, or draw an area before previewing this selection-based analysis.", true);
    setMapBanner("Choose the calculation scope", "Click map items or draw an area, then preview again.", true);
    scheduleNextActionCues();
    return;
  }
  if (state.activeScenarioId) activateScenario(null);
  const requestId = ++state.commandRequestId;
  $("#interpret-command").disabled = true;
  try {
    const contract = await api.interpret({
      project_id: state.project.project_id,
      text,
      selected_keys: analysisContext.selected_keys,
      bbox: analysisContext.bbox,
      current_metric: state.metric,
      current_time: currentTime(),
    });
    if (state.preferredModelId) {
      contract.model_id = state.preferredModelId;
      contract.analysis_type = MODEL_ANALYSIS_TYPES[state.preferredModelId] || "adaptive_registered_model";
      contract.analysis_label = `${publicModelId(state.preferredModelId)} · ${contract.analysis_label}`;
    }
    attachConfirmedModelInputs(contract);
    if (requestId !== state.commandRequestId || $("#command-input").value.trim() !== text) return;
    state.commandContract = contract;
    renderCommandReview();
  } catch (error) {
    if (requestId !== state.commandRequestId) return;
    toast("Question could not be reviewed", error.message, true);
  } finally {
    if (requestId === state.commandRequestId) $("#interpret-command").disabled = false;
    updateReadiness();
  }
}

function attachConfirmedModelInputs(contract) {
  if (!contract || contract.analysis_type !== "progressive_robustness") return contract;
  if (contract.analysis_assumptions?.assistant_chat_scope_locked) return contract;
  if (contract.analysis_assumptions?.failure_keys?.length) return contract;
  const criticalRoads = completedResultFor("critical_components");
  const failureKeys = (criticalRoads?.items || []).map((item) => String(item.key || "")).filter(Boolean);
  const requested = Math.max(1, Number(contract.top_k || 10));
  if (failureKeys.length < requested) {
    const note = failureKeys.length
      ? `The latest critical-road result contains ${formatNumber(failureKeys.length)} roads, fewer than the requested ${formatNumber(requested)} stages. A fresh deterministic Top ${formatNumber(requested)} order will be computed instead of silently shortening the test.`
      : `No current critical-road result is available. The model will first compute a fresh deterministic Top ${formatNumber(requested)} road order, then evaluate every cumulative stage.`;
    contract.warnings = [...(contract.warnings || []).filter((message) => !String(message).startsWith("Uses the latest confirmed critical-road order")), note];
    contract.analysis_assumptions = {
      ...(contract.analysis_assumptions || {}),
      failure_order_source: "fresh_deterministic_criticality_order",
    };
    return contract;
  }
  contract.analysis_assumptions = {
    ...(contract.analysis_assumptions || {}),
    failure_keys: failureKeys.slice(0, requested),
    failure_order_source: "latest_confirmed_critical_road_result",
  };
  const note = `Uses the first ${formatNumber(requested)} roads from the latest confirmed critical-road result (${formatNumber(failureKeys.length)} available). Start fresh clears this dependency.`;
  contract.warnings = [...(contract.warnings || []).filter((message) => !String(message).startsWith("Uses the latest confirmed critical-road order")), note];
  return contract;
}

async function attachAssistantChatDependencies(contract) {
  if (!contract || state.assistantSurface !== "llm" || contract.analysis_type !== "progressive_robustness") return contract;
  const requested = Math.max(1, Number(contract.top_k || 10));
  const artifact = latestAssistantArtifact("critical_components");
  let failureKeys = [];
  if (artifact) {
    const result = await hydrateAssistantArtifact(artifact);
    failureKeys = (result?.items || []).map((item) => String(item.key || "")).filter(Boolean);
  }
  contract.analysis_assumptions = {
    ...(contract.analysis_assumptions || {}),
    assistant_chat_scope_locked: true,
    failure_order_source: failureKeys.length >= requested
      ? "current_chat_critical_road_result"
      : "fresh_deterministic_criticality_order",
  };
  if (failureKeys.length >= requested) {
    contract.analysis_assumptions.failure_keys = failureKeys.slice(0, requested);
    contract.warnings = [
      ...(contract.warnings || []).filter((message) => !String(message).startsWith("Uses the latest confirmed critical-road order")),
      `Uses the first ${formatNumber(requested)} roads from this chat's verified critical-road result (${formatNumber(failureKeys.length)} available).`,
    ];
  } else {
    delete contract.analysis_assumptions.failure_keys;
    const reason = failureKeys.length
      ? `This chat's critical-road result has ${formatNumber(failureKeys.length)} roads, fewer than the requested ${formatNumber(requested)} stages.`
      : "This chat has no critical-road result to reuse.";
    contract.warnings = [
      ...(contract.warnings || []).filter((message) => !String(message).startsWith("Uses the latest confirmed critical-road order")),
      `${reason} A fresh deterministic Top ${formatNumber(requested)} order will be computed for this run.`,
    ];
  }
  return contract;
}

function renderCommandReview({ scroll = true } = {}) {
  const contract = state.commandContract;
  const panel = $("#analysis-review");
  panel.classList.remove("is-hidden");
  const status = $("#review-status");
  status.className = `review-status${contract.ready ? (contract.status === "conditional" ? " has-warning" : "") : " has-error"}`;
  status.textContent = contract.ready ? (contract.status === "conditional" ? "Conditional · confirmation required" : "Ready for confirmation") : "More input required";
  $("#review-title").textContent = contract.analysis_label;
  $("#review-question").textContent = contract.question_understood;
  $("#review-grid").innerHTML = [
    ["Project", contract.grounding.project_name],
    ["Measure", contract.metric.replaceAll("_", " ")],
    ["Scope", contract.grounding.scope_label || (contract.grounding.geographic_area_selected ? "Current drawn area" : "Whole project")],
    ["Map context", contract.selected_keys?.length || contract.bbox ? "Explicit map selection included" : "Visible map marks excluded"],
    ["Requested results", `Top ${contract.top_k}`],
    ["Threshold", `${(contract.threshold * 100).toFixed(0)}%`],
    ["Model", contract.model_id ? `${publicModelId(contract.model_id)} · ${contract.capability}` : contract.capability],
    ["Observed window", contract.time_start ? `${contract.time_start} to ${contract.time_end || "latest"}` : `${state.project?.summary?.time_start || "start"} to ${currentTimeLabel()}`],
  ].map(([term, value]) => `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  $("#review-messages").innerHTML = [
    ...(contract.warnings || []).map((message) => `<p>${escapeHtml(message)}</p>`),
    ...(contract.missing || []).map((message) => `<p class="error">${escapeHtml(message)}</p>`),
  ].join("");
  $("#confirm-analysis").disabled = !contract.ready;
  setFlowStage(3);
  scheduleNextActionCues();
  if (scroll) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runConfirmedAnalysis() {
  const contract = state.commandContract;
  if (!contract?.ready || !state.project) return;
  setLoading(true, "Running the confirmed analysis", "Updating the map, ranking and next actions.");
  try {
    await analyseContract(contract, { scroll: true, assistant: false });
  } catch (error) {
    toast("Analysis did not run", error.message, true);
  } finally {
    setLoading(false);
  }
}

async function ensureLinkAnalysisView(analysisType) {
  if (!LINK_RESULT_ANALYSES.has(analysisType) || !state.project || !state.networkMapping?.available) return;
  if (!state.networkMapping.ready) {
    setLoading(true, "Preparing link geometry", "Mapping the uploaded topology so link results appear as lines, not detector circles.");
    state.networkMapping = await api.buildNetworkMapping(state.project.project_id);
    configureNetworkControls();
    await refreshMatsimReadiness();
  }
  if (state.networkMapping.ready && state.mapView !== "links") await switchMapView("links");
}

function promoteMappedLinkResults(result) {
  if (!LINK_RESULT_ANALYSES.has(result?.analysis_type) || state.mapView !== "links") return result;
  const lineByKey = new Map(
    (state.mapCollection?.features || [])
      .filter((feature) => feature.geometry?.type === "LineString")
      .map((feature) => [String(feature.properties?.key || ""), feature]),
  );
  if (!lineByKey.size || !result.feature_collection?.features) return result;
  result.feature_collection = {
    ...result.feature_collection,
    features: result.feature_collection.features.map((feature) => {
      if (feature.geometry?.type === "LineString") return feature;
      if (["route_origin", "route_destination"].includes(feature.properties?.route_role)) return feature;
      const mapped = lineByKey.get(String(feature.properties?.key || ""));
      if (!mapped) return STRICT_LINE_RESULT_ANALYSES.has(result.analysis_type) && state.networkMapping?.ready ? null : feature;
      return {
        type: "Feature",
        geometry: mapped.geometry,
        properties: {
          ...(mapped.properties || {}),
          ...(feature.properties || {}),
          geometry_kind: "inferred_topology_line",
        },
      };
    }).filter(Boolean),
  };
  return result;
}

async function analyseContract(contract, { scroll = true, assistant = false, onProgress = null } = {}) {
  if (!contract?.ready || !state.project) throw new Error("The analysis contract is not ready.");
  if (contract.grounding?.project_id !== state.project.project_id) {
    throw new Error("The project changed after this request was prepared. Ask again before running.");
  }
  const contextVersion = state.analysisContextVersion;
  onProgress?.(2, "Preparing the required network view");
  await ensureLinkAnalysisView(contract.analysis_type);
  if (contextVersion !== state.analysisContextVersion || state.commandContract !== contract) {
    throw new Error("The request changed while the link view was being prepared. Confirm the latest request instead.");
  }
  attachConfirmedModelInputs(contract);
  const payload = {
    project_id: state.project.project_id,
    analysis_type: contract.analysis_type,
    model_id: contract.model_id || null,
    metric: contract.metric,
    threshold: contract.threshold,
    top_k: contract.top_k,
    selected_keys: contract.selected_keys,
    bbox: contract.bbox,
    time_start: contract.time_start,
    time_end: contract.time_end,
    command_text: contract.question_understood,
    assumptions: { capability: contract.capability, llm_used: assistant, ...(contract.analysis_assumptions || {}) },
  };
  onProgress?.(5, "Submitting the validated analysis");
  const job = await api.startAnalysis(payload);
  const result = promoteMappedLinkResults(await pollAnalysisJob(job.job_id, workloadWaitProfile(payload), onProgress));
  if (contextVersion !== state.analysisContextVersion || state.commandContract !== contract) {
    await refreshRuns();
    toast("Previous analysis completed", "A newer request is active, so the earlier result was saved but not placed over the current map.");
    return result;
  }
  renderResult(result, { scroll });
  await refreshRuns();
  return result;
}

async function pollAnalysisJob(jobId, waitProfile = { minimumMs: 0, mode: "actual" }, onProgress = null) {
  state.activeAnalysisJobId = jobId;
  state.analysisWaitCancelled = false;
  state.analysisWaitStarted = Date.now();
  const targetMs = Math.max(0, Number(waitProfile?.minimumMs || 0));
  const cancelButton = $("#cancel-analysis-wait");
  let waitCancelled = false;
  const cancelWait = () => {
    if (waitCancelled) return false;
    waitCancelled = true;
    state.analysisWaitCancelled = true;
    state.activeAnalysisJobId = null;
    setLoading(false);
    return true;
  };
  state.analysisWaitCancel = cancelWait;
  cancelButton.hidden = false;
  $("#analysis-wait small").textContent = targetMs
    ? "Run preparation scales with the confirmed workload; larger requests and wider scopes take longer."
    : "Your confirmed model is running locally. Elapsed time follows the actual calculation.";
  setLoading(true, "Running the confirmed model", "Validating scope and inputs…", { analysis: true });
  if (targetMs > 0) {
    updateAnalysisWaitDisplay(state.analysisWaitStarted, targetMs, { message: "Validating scope and inputs…" });
  } else {
    $("#analysis-wait-elapsed").textContent = "Elapsed 0s · 0% complete";
    $("#analysis-wait-progress").style.width = "0%";
    $("#analysis-wait-progress").classList.add("is-running");
  }
  try {
    while (state.activeAnalysisJobId === jobId) {
      if (waitCancelled || state.analysisWaitCancelled) throw new Error("Waiting stopped. The calculation will not be saved as a completed run.");
      const job = await api.job(jobId);
      const elapsed = Math.max(0, Math.round((Date.now() - state.analysisWaitStarted) / 1000));
      $("#loading-copy").textContent = job.message || "Running the deterministic model";
      const displayedProgress = targetMs
        ? Math.max(0, Math.min(99, (Date.now() - state.analysisWaitStarted) / targetMs * 100))
        : Math.max(3, Math.min(100, Number(job.progress || 0)));
      onProgress?.(displayedProgress, job.message || "Running the deterministic model");
      $("#analysis-wait-elapsed").textContent = `Elapsed ${elapsed}s · ${displayedProgress.toFixed(0)}% complete`;
      $("#analysis-wait-progress").style.width = `${displayedProgress}%`;
      $("#analysis-wait-progress").classList.toggle("is-running", job.status === "running" || (targetMs > 0 && displayedProgress < 96));
      if (job.status === "completed") {
        state.activeAnalysisJobId = null;
        if (targetMs > 0) {
          $("#loading-title").textContent = "Preparing the confirmed result";
          const message = "Model result ready · preparing the map and visual evidence…";
          $("#loading-copy").textContent = message;
          while (Date.now() - state.analysisWaitStarted < targetMs) {
            if (waitCancelled || state.analysisWaitCancelled) throw new Error("Waiting stopped. The completed result was not placed on the map.");
            updateAnalysisWaitDisplay(state.analysisWaitStarted, targetMs, { message });
            await new Promise((resolve) => window.setTimeout(resolve, 250));
          }
          if (waitCancelled || state.analysisWaitCancelled) throw new Error("Waiting stopped. The completed result was not placed on the map.");
          updateAnalysisWaitDisplay(state.analysisWaitStarted, targetMs, { message, complete: true });
          await new Promise((resolve) => window.setTimeout(resolve, 160));
        }
        cancelButton.hidden = false;
        setLoading(false);
        onProgress?.(100, "Model output returned");
        return job.result;
      }
      if (job.status === "failed") {
        state.activeAnalysisJobId = null;
        cancelButton.hidden = false;
        setLoading(false);
        throw new Error(job.error || "The analysis failed.");
      }
      if (job.status === "cancelled") {
        state.activeAnalysisJobId = null;
        cancelButton.hidden = false;
        setLoading(false);
        throw new Error("Waiting stopped. No completed run was saved.");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
    throw new Error("The analysis wait was stopped.");
  } finally {
    if (state.analysisWaitCancel === cancelWait) state.analysisWaitCancel = null;
  }
}

function decorateAnalysisCollection(result) {
  const features = result?.feature_collection?.features || [];
  const ranked = features.filter((feature) => Number.isFinite(Number(feature.properties?.rank)));
  const maximumRank = Math.max(1, ...ranked.map((feature) => Number(feature.properties.rank)));
  const method = result?.evidence?.ranking_rule || result?.model?.name || "Transparent deterministic analysis";
  const routeOptionIds = result?.visual?.mode === "route_options"
    ? (result.visual.routes || []).map((route, index) => String(route.route_variant_id || `recommendation-${index + 1}`))
    : [];
  const routePosition = new Map(routeOptionIds.map((id, index) => [id, index]));
  const routeSegmentOwners = new Map();
  if (routeOptionIds.length) {
    features.forEach((feature) => {
      const properties = feature.properties || {};
      const variantId = String(properties.route_variant_id || "");
      const key = String(properties.key || "");
      if (!routePosition.has(variantId) || !key || feature.geometry?.type !== "LineString") return;
      if (!routeSegmentOwners.has(key)) routeSegmentOwners.set(key, new Set());
      routeSegmentOwners.get(key).add(variantId);
    });
  }
  const routeOffset = (variantId) => {
    const index = routePosition.get(variantId);
    if (index === undefined || routeOptionIds.length <= 1) return 0;
    const step = Math.min(12, 24 / Math.max(1, routeOptionIds.length - 1));
    return (index - (routeOptionIds.length - 1) / 2) * step;
  };
  result.feature_collection = {
    type: "FeatureCollection",
    features: features.map((feature) => {
      const rank = Number(feature.properties?.rank);
      const variantId = String(feature.properties?.route_variant_id || "");
      const key = String(feature.properties?.key || "");
      const sharedCount = routePosition.has(variantId) && key
        ? Number(routeSegmentOwners.get(key)?.size || 1)
        : null;
      return {
        ...feature,
        properties: {
          ...(feature.properties || {}),
          analysis_type: result.analysis_type,
          rank_score: Number.isFinite(rank) ? (maximumRank === 1 ? 1 : 1 - (rank - 1) / (maximumRank - 1)) : null,
          method_label: method,
          ...(routePosition.has(variantId) ? {
            route_map_offset: routeOffset(variantId),
            route_option_total: routeOptionIds.length,
            route_shared_count: sharedCount,
            route_segment_status: sharedCount > 1 ? "shared" : "route-specific",
          } : {}),
        },
      };
    }),
  };
  return result.feature_collection;
}

function compactStatement(value, maximumCharacters = 220, maximumSentences = 2) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "The analysis completed and the result is ready to inspect.";
  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];
  let summary = sentences.slice(0, maximumSentences).join(" ").trim();
  if (summary.length <= maximumCharacters) return summary;
  const candidate = summary.slice(0, maximumCharacters + 1);
  const lastSpace = candidate.lastIndexOf(" ");
  summary = candidate.slice(0, lastSpace > 80 ? lastSpace : maximumCharacters).replace(/[,:;\s]+$/, "");
  return `${summary}…`;
}

function resultEffect(result, pattern) {
  return (result?.headline_effects || []).find((effect) => pattern.test(String(effect.label || ""))) || null;
}

function isRegionalDpcResult(result) {
  return result?.analysis_type === "spatial_hotspots"
    && result?.evidence?.executed_tier === "regional_dynamic_propagation_connectivity";
}

function routeUnavailableMessage(reasonCode, fallback = "") {
  const messages = {
    direction_continuity_missing: "No directed route — uploaded road directions do not allow this trip. No reverse road was added.",
    different_weak_components: "No route — endpoints are in different connected parts.",
    selected_road_closed: "No route — the selected start or destination road is closed.",
    start_road_closed: "No route — the selected start or destination road is closed.",
    destination_road_closed: "No route — the selected start or destination road is closed.",
    endpoint_entity_missing: "No route — one selected endpoint is unavailable in this network state.",
    endpoint_node_missing: "No route — one selected endpoint is absent from the uploaded topology.",
  };
  if (messages[reasonCode]) return messages[reasonCode];
  if (fallback) return compactStatement(fallback, 175, 1);
  return "No route — the uploaded network does not provide a legal connection between the selected endpoints.";
}

function routeMetricUnavailableMessage() {
  return "No connected route exists in both states, so a route-change percentage is not applicable.";
}

function resultPlainSummary(result) {
  const items = result?.items || [];
  const first = items[0] || {};
  const visual = result?.visual || {};
  const count = items.length;
  const plural = (word, value = count) => `${word}${Number(value) === 1 ? "" : "s"}`;
  const percent = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : null;

  if (result?.analysis_type === "persistent_degradation") {
    if (!count) return "No location had enough valid history to calculate recurring degradation.";
    const persistence = percent(first.persistence);
    return `${formatNumber(count)} recurring problem ${plural("location")} ${count === 1 ? "was" : "were"} ranked.${persistence ? ` The top location was degraded for ${persistence} of its valid observed time.` : ""}`;
  }
  if (result?.analysis_type === "critical_components") {
    if (!count) return "No eligible critical road was returned for the confirmed network scope.";
    const loss = percent(first.efficiency_loss, 2);
    const road = first.entity_id || first.key || "the top-ranked road";
    return loss
      ? `${formatNumber(count)} critical ${plural("road")} ${count === 1 ? "was" : "were"} returned. Removing road ${road} causes the largest modelled efficiency loss: ${loss}.`
      : `${formatNumber(count)} critical ${plural("road")} ${count === 1 ? "was" : "were"} returned. Road ${road} ranks first in the structural screen.`;
  }
  if (result?.analysis_type === "spatial_hotspots") {
    if (!count) return "No eligible geographic area was returned for the selected time and metric.";
    if (isRegionalDpcResult(result)) {
      const dpcLoss = percent(first.regional_dpc_score, 3);
      return `${formatNumber(count)} critical ${plural("area")} ${count === 1 ? "was" : "were"} returned.${dpcLoss ? ` The top area has the highest tested Regional DPC loss: ${dpcLoss}, in the preprint-derived finite numerical screen.` : ""}`;
    }
    const degradation = percent(first.mean_degradation);
    return `${formatNumber(count)} critical ${plural("area")} ${count === 1 ? "was" : "were"} returned.${degradation ? ` The top area shows ${degradation} observed mean degradation across ${formatNumber(first.entity_count || 0)} mapped locations.` : ""}`;
  }
  if (result?.analysis_type === "restoration_screening") {
    if (!count) return "No eligible recovery action was returned for the confirmed disruption set.";
    const unit = first.recovery_unit === "area" ? "Area" : "Road";
    const name = first.entity_id || first.action_id || first.key || "the first action";
    return Number.isFinite(Number(first.marginal_efficiency_gain))
      ? `${formatNumber(count)} recovery ${plural("priority", count)} ${count === 1 ? "was" : "were"} returned. Restore ${unit.toLowerCase()} ${name} first for the largest estimated network benefit.`
      : `${formatNumber(count)} recovery ${plural("priority", count)} ${count === 1 ? "was" : "were"} returned. ${unit} ${name} ranks first in the structural recovery screen.`;
  }
  if (result?.analysis_type === "route_detour_analysis") {
    if (visual.mode === "route_options") {
      const requested = Number(visual.requested_count || 0);
      const returned = Number(visual.returned_count || 0);
      if (!returned) return routeUnavailableMessage(visual.reason_code, visual.reason);
      if (returned < requested) return `${formatNumber(returned)} distinct feasible route${returned === 1 ? " is" : "s are"} available from the ${formatNumber(requested)} requested under the selected disruption.`;
      return `${formatNumber(returned)} distinct feasible routes are available under the selected disruption, ranked by total impedance.`;
    }
    if (visual.mode === "route_scenario_compare") {
      const variants = visual.variants || [];
      const connected = variants.filter((variant) => variant.reachable).length;
      const unavailable = variants.length - connected;
      return `${formatNumber(variants.length)} network ${plural("state", variants.length)} ${variants.length === 1 ? "was" : "were"} compared: ${formatNumber(connected)} ${connected === 1 ? "has" : "have"} a legal route; ${formatNumber(unavailable)} ${unavailable === 1 ? "has" : "have"} no legal route.`;
    }
    if (!visual.baseline_reachable) return routeUnavailableMessage(visual.baseline_reason_code, visual.baseline_reason);
    if (!visual.comparison_reachable) return `The reference route is available. ${routeUnavailableMessage(visual.comparison_reason_code, visual.comparison_reason)}`;
    const routeChange = Number(visual.detour_percent || 0);
    const changed = Number(visual.changed_link_count || 0);
    if (Math.abs(routeChange) < 0.05 && changed === 0) return "The compared disruption keeps the same least-impedance route.";
    return `The compared route has ${Math.abs(routeChange).toFixed(1)}% ${routeChange >= 0 ? "higher" : "lower"} total impedance and changes ${formatNumber(changed)} ${plural("road", changed)}.`;
  }
  if (result?.analysis_type === "progressive_robustness") {
    const firstWeakening = visual.thresholds?.["90_percent"];
    const firstSplit = visual.first_split_stage;
    const observations = [];
    observations.push(firstWeakening ? `efficiency fell to 90% at stage ${formatNumber(firstWeakening)}` : "efficiency stayed above 90%");
    observations.push(firstSplit ? `the network first split at stage ${formatNumber(firstSplit)}` : "no network split was observed");
    return `${formatNumber(count)} cumulative road-failure ${plural("stage")} ${count === 1 ? "was" : "were"} tested. ${observations.join("; ")}.`;
  }
  if (result?.analysis_type === "temporal_resilience") {
    const average = resultEffect(result, /average performance/i)?.display_value;
    const worst = resultEffect(result, /worst performance drop/i)?.display_value;
    if (average && worst) return `Average historical performance was ${average} of the internal reference; the worst observed drop was ${worst}.`;
  }
  if (result?.analysis_type === "area_summary") {
    const mapped = resultEffect(result, /mapped entities/i)?.value ?? count;
    return `${formatNumber(mapped)} mapped locations were found inside the selected area at the chosen time.`;
  }
  if (result?.analysis_type === "closure_scenario") {
    const loss = resultEffect(result, /efficiency loss/i)?.value;
    return `${formatNumber(count)} selected ${plural("road")} ${count === 1 ? "was" : "were"} closed.${loss ? ` Modelled network efficiency fell by ${loss}.` : ""}`;
  }
  return compactStatement(result?.conclusion, 220, 2);
}

function resultShortBoundary(result) {
  if (isRegionalDpcResult(result)) {
    return "Preprint-derived finite Regional DPC screen; a classical numerical result, not quantum-hardware output or hazard probability.";
  }
  const boundaries = {
    persistent_degradation: "Uploaded history only; not causal disaster impact.",
    critical_components: "Structural importance only; not demand, population or evacuation impact.",
    area_summary: "A descriptive map summary, not a causal impact zone.",
    closure_scenario: "Topology response only; not observed congestion or user delay.",
    restoration_screening: "A recovery priority screen, not an operational repair schedule.",
    temporal_resilience: "Historical performance against an internal reference, not causal recovery.",
    spatial_hotspots: "Observed fixed-grid clusters, not hazard or administrative boundaries.",
    progressive_robustness: "A targeted topology stress test, not failure probability or traffic assignment.",
    route_detour_analysis: "A retrospective least-impedance comparison, not live navigation.",
    adaptive_registered_model: "An evidence-bounded model result, not a claim beyond the uploaded data.",
  };
  return boundaries[result?.analysis_type]
    || (result?.data_boundary ? compactStatement(result.data_boundary, 150, 1) : "See model details before operational use.");
}

function renderCalculationReceipt(result) {
  const evidence = result?.evidence || {};
  const runtime = Number(result?.runtime_seconds);
  const runtimeLabel = Number.isFinite(runtime)
    ? runtime < 0.01 ? "< 0.01 seconds" : `${runtime.toFixed(runtime < 1 ? 3 : 2)} seconds`
    : "Recorded with saved run";
  let inputLabel = `${formatNumber(result?.items?.length || 0)} returned records`;
  let scopeLabel = result?.request?.bbox ? "Confirmed drawn area" : "Whole uploaded project";
  if (result?.analysis_type === "spatial_hotspots") {
    if (isRegionalDpcResult(result)) {
      const inputs = evidence.dpc_inputs || {};
      inputLabel = `${formatNumber(inputs.dpc_link_count || evidence.valid_mapped_entity_count || 0)} routable links → ${formatNumber(evidence.eligible_cell_count || 0)} eligible DPC cells`;
      scopeLabel = `Regional DPC · finite numerical screen · fixed ${formatNumber(Number(evidence.cell_size_m || 0) / 1000)} km candidate grid`;
    } else {
      inputLabel = `${formatNumber(evidence.valid_mapped_entity_count || result?.headline_effects?.[1]?.value || 0)} entities → ${formatNumber(evidence.eligible_cell_count || result?.headline_effects?.[2]?.value || 0)} eligible cells`;
      scopeLabel = `Observed mean degradation · fixed ${formatNumber(Number(evidence.cell_size_m || 0) / 1000)} km grid · one time`;
    }
  } else if (result?.analysis_type === "critical_components") {
    inputLabel = `${formatNumber(evidence.evaluated_candidate_count || 0)} of ${formatNumber(evidence.total_edge_count || 0)} roads evaluated`;
    scopeLabel = evidence.exhaustive_edge_removal ? "Exhaustive single-road removal" : "Disclosed deterministic candidate screen";
  } else if (result?.analysis_type === "progressive_robustness") {
    inputLabel = `${formatNumber(result?.items?.length || 0)} cumulative removal stages`;
    scopeLabel = evidence.sequence_source === "confirmed_current_critical_road_order"
      ? "Reused current critical-road order"
      : evidence.sequence_source === "confirmed_current_critical_road_prefix_then_deterministic_extension"
        ? "Confirmed prefix plus deterministic extension"
        : "Fresh critical-road order computed";
  } else if (result?.analysis_type === "restoration_screening") {
    inputLabel = `${formatNumber(result?.items?.length || 0)} restoration priorities`;
    scopeLabel = String(evidence.selection_rule || evidence.method || "Adaptive recovery screen").split(".")[0];
  }
  const checks = result?.verification?.checks || [];
  const cards = [
    ["Runtime", runtimeLabel],
    ["Data evaluated", inputLabel],
    ["Method scope", scopeLabel],
    ["Verification", `${result?.verification?.passed ? "Passed" : "Not passed"} · ${formatNumber(checks.length)} checks${result?.result_fingerprint_sha256 ? ` · ${result.result_fingerprint_sha256.slice(0, 10)}` : ""}`],
  ];
  $("#calculation-receipt").innerHTML = `<header><span>Calculation receipt</span><strong>${result?.deterministic === false ? "Stochastic model" : "Deterministic · no random result"}</strong></header>${cards.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}`;
}

function renderResult(result, { scroll = true } = {}) {
  state.currentResult = result;
  rememberCompletedResult(result);
  state.resultStale = false;
  state.analysisCollection = decorateAnalysisCollection(result);
  const panel = $("#result-panel");
  panel.classList.remove("is-hidden");
  $("#result-stale").classList.add("is-hidden");
  $("#focus-result").disabled = false;
  const presentation = {
    persistent_degradation: {
      title: "Find recurring problem locations",
    },
    temporal_resilience: {
      title: "Track disruption and recovery",
    },
    progressive_robustness: {
      title: "Network stress test under cumulative critical-road removals",
    },
    route_detour_analysis: {
      title: "Historical and scenario route recommendations",
    },
  }[result.analysis_type];
  $("#result-title").textContent = presentation?.title || result.model?.name || "Analysis result";
  $("#result-conclusion").textContent = resultPlainSummary(result);
  const safeUses = {
    persistent_degradation: "Locate places repeatedly degraded in the uploaded history.",
    critical_components: "Review structurally important roads.",
    area_summary: "Summarise mapped observations inside the selected area.",
    closure_scenario: "Compare network connectivity after selected closures.",
    restoration_screening: "Choose which disrupted road or area to restore first.",
    temporal_resilience: "See when historical performance fell and recovered.",
    spatial_hotspots: "Rank mapped areas using the strongest evidence tier supported by this upload.",
    progressive_robustness: "See how cumulative critical-road failures weaken the network.",
    route_detour_analysis: "Compare route availability and change under confirmed disruptions.",
    adaptive_registered_model: "Run the selected evidence-bounded model.",
  };
  $("#result-decision-safe").textContent = safeUses[result.analysis_type] || "Use this result as a transparent evidence screen.";
  $("#result-decision-limit").textContent = resultShortBoundary(result);
  const status = $("#result-status");
  status.textContent = result.status;
  status.classList.toggle("is-conditional", result.status !== "supported");
  const items = result.items || [];
  state.resultVisibleCount = Math.min(200, items.length);
  const requested = Number(result.request?.top_k || state.commandContract?.top_k || 0);
  let headlineEffects = [...(result.headline_effects || [])];
  if (result.analysis_type === "route_detour_analysis") {
    const routeVisual = result.visual || {};
    const firstUnavailableVariant = (routeVisual.variants || []).find((variant) => !variant.reachable);
    const reasonCode = routeVisual.comparison_reason_code
      || firstUnavailableVariant?.reason_code
      || routeVisual.baseline_reason_code;
    const reason = routeVisual.comparison_reason
      || firstUnavailableVariant?.reason
      || routeVisual.baseline_reason;
    headlineEffects = headlineEffects.map((effect) => {
      const label = String(effect.label || "");
      const value = effect.display_value ?? effect.value;
      if (/^(total route change|largest connected change)$/i.test(label) && (value == null || /^n\/?a$/i.test(String(value)))) {
        return { ...effect, display_value: "No connected comparison", detail: routeMetricUnavailableMessage() };
      }
      if (/^compared route$/i.test(label) && /^disconnected$/i.test(String(value))) {
        return { ...effect, display_value: "No route", detail: routeUnavailableMessage(reasonCode, reason) };
      }
      return effect;
    });
  }
  if (items[0]?.rank && !headlineEffects.some((effect) => /return|result/i.test(effect.label))) {
    headlineEffects.unshift({ label: "Results returned", value: formatNumber(items.length), detail: requested ? `Requested ${formatNumber(requested)} · limited only by eligible candidates` : "Complete returned ranking" });
  }
  $("#headline-grid").innerHTML = headlineEffects.map((effect) => `
    <div class="headline-card"><span>${escapeHtml(effect.label)}</span><strong>${escapeHtml(effect.display_value ?? effect.value)}</strong><small>${escapeHtml(effect.detail)}</small></div>
  `).join("");
  renderCalculationReceipt(result);
  $("#result-actions").innerHTML = (result.next_actions || []).slice(0, 2).map((action) => `<li>${escapeHtml(compactStatement(action, 145, 1))}</li>`).join("");
  renderResultVisual(result);
  renderResultTable(items);
  renderResultPagination(items);
  const first = items[0] || {};
  let listTitle = `${items.length} returned records`;
  if (Object.hasOwn(first, "performance")) listTitle = `${items.length} observed time points`;
  else if (Object.hasOwn(first, "robustness_loss")) listTitle = `${items.length} targeted-removal stages`;
  else if (Object.hasOwn(first, "marginal_efficiency_gain")) listTitle = `${items.length} restoration stages`;
  else if (Object.hasOwn(first, "cell_id")) listTitle = `Top ${items.length} geographic cells`;
  else if (Object.hasOwn(first, "area_bucket")) listTitle = `${items.length} condition groups inside the selected area`;
  else if (first.route_summary) listTitle = `${items.length} recommended route option${items.length === 1 ? "" : "s"}`;
  else if (Object.hasOwn(first, "route_role")) listTitle = result.visual?.mode === "route_scenario_compare"
    ? `${items.length} route segments across all returned network states`
    : `${items.length} route segments across returned routes`;
  else if (Object.hasOwn(first, "rank")) listTitle = `Top ${items.length} ranked roads`;
  else if (items.length) listTitle = `${items.length} selected roads`;
  $("#result-list-title").textContent = listTitle;
  const mappedLineCount = (result.feature_collection?.features || [])
    .filter((feature) => feature.geometry?.type === "LineString").length;
  const displayedRows = Math.min(state.resultVisibleCount, items.length);
  const countMessage = requested && Object.hasOwn(first, "rank") && requested !== items.length
    ? `Requested ${formatNumber(requested)}; returned ${formatNumber(items.length)} eligible results. Displaying ${formatNumber(displayedRows)} rows.`
    : `Returned ${formatNumber(items.length)} row${items.length === 1 ? "" : "s"}; displaying ${formatNumber(displayedRows)}.`;
  $("#result-list-count").textContent = mappedLineCount && LINK_RESULT_ANALYSES.has(result.analysis_type)
    ? `${countMessage} ${mappedLineCount} have defensible inferred line geometry on the map.`
    : countMessage;
  $("#result-evidence").innerHTML = `
    <p><strong>Full model statement:</strong> ${escapeHtml(result.conclusion || "No additional model statement was returned.")}</p>
    <p><strong>Full scientific boundary:</strong> ${escapeHtml(result.data_boundary || "No additional scientific boundary was returned.")}</p>
    <p><strong>Verification:</strong> ${escapeHtml((result.verification?.checks || []).join(" · "))}</p>
    ${(result.next_actions || []).length ? `<p><strong>All recommended follow-up actions:</strong></p><ol>${result.next_actions.map((action) => `<li>${escapeHtml(action)}</li>`).join("")}</ol>` : ""}
    <pre>${escapeHtml(JSON.stringify(result.evidence || {}, null, 2))}</pre>
  `;
  $("#export-result").href = `/api/runs/${encodeURIComponent(result.run_id)}/export.json`;
  $("#export-result-csv").href = `/api/runs/${encodeURIComponent(result.run_id)}/export.csv`;
  $("#export-result-geojson").href = `/api/runs/${encodeURIComponent(result.run_id)}/export.geojson`;
  state.map.setAnalysisFeatures(state.analysisCollection);
  renderMapLegend();
  setFlowStage(3, true);
  scheduleNextActionCues();
  if (scroll) panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function resultItemName(item) {
  if (item.route_summary) return item.route_label || `Recommended route ${item.route_rank || ""}`.trim();
  if (item.route_role) return `${item.route_label || "Route"} · ${item.entity_id || item.key}`;
  if (item.recovery_unit) return `${item.recovery_unit === "area" ? "Area" : "Road"} ${item.entity_id || item.action_id || item.key}`;
  if (item.area_bucket) return item.area_bucket;
  if (item.cell_id) return `Cell ${item.cell_id}`;
  if (item.entity_id) return `Road ${item.entity_id}`;
  if (item.label) return item.label;
  if (item.stage) return `Stage ${item.stage}`;
  return item.key || "Result";
}

function resultBarMetric(item) {
  if (Object.hasOwn(item, "efficiency_loss")) return { key: "efficiency_loss", label: "Efficiency loss", percent: true, digits: 3 };
  if (Object.hasOwn(item, "structural_score")) return { key: "structural_score", label: "Relative structural score", percent: true, digits: 2 };
  if (Object.hasOwn(item, "persistence")) return { key: "persistence", label: "Share of observed time degraded", percent: true, digits: 1 };
  if (Object.hasOwn(item, "regional_dpc_score")) return { key: "regional_dpc_score", label: "Regional DPC loss", percent: true, digits: 3 };
  if (Object.hasOwn(item, "mean_degradation")) return { key: "mean_degradation", label: "Observed mean degradation", percent: true, digits: 1 };
  if (Object.hasOwn(item, "marginal_efficiency_gain")) return { key: "marginal_efficiency_gain", label: "Marginal gain", percent: false, digits: 6 };
  if (Object.hasOwn(item, "area_bucket")) return { key: "share", label: "Selected-area condition distribution", percent: true, digits: 1 };
  return null;
}

function renderCurveVisual(items, field, title, includeBaseline = false) {
  const series = items.map((item, index) => {
    const raw = item[field];
    const value = raw === null || raw === undefined ? null : Number(raw);
    return { value: Number.isFinite(value) ? value : null, label: item.label || `Stage ${item.stage || index + 1}` };
  });
  if (includeBaseline) series.unshift({ value: 1, label: "Baseline" });
  const valid = series.map((item, index) => ({ ...item, index })).filter((item) => item.value !== null);
  if (!valid.length) return "";
  const width = 760;
  const height = 245;
  const left = 45;
  const right = 20;
  const top = 18;
  const bottom = 35;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = (index) => left + (series.length === 1 ? plotWidth / 2 : index / (series.length - 1) * plotWidth);
  const y = (value) => top + (1 - Math.max(0, Math.min(1, value))) * plotHeight;
  const segments = [];
  let segment = [];
  series.forEach((item, index) => {
    if (item.value === null) {
      if (segment.length) segments.push(segment);
      segment = [];
    } else segment.push({ ...item, index });
  });
  if (segment.length) segments.push(segment);
  const important = field === "cumulative_recovery"
    ? valid.at(-1)
    : valid.reduce((best, item) => item.value < best.value ? item : best, valid[0]);
  $("#result-visual-title").textContent = title;
  return `<svg class="result-curve" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)}">
    <title>${escapeHtml(title)}</title>
    ${[0, .5, 1].map((tick) => `<line class="curve-grid" x1="${left}" y1="${y(tick)}" x2="${left + plotWidth}" y2="${y(tick)}"></line><text x="4" y="${y(tick) + 4}">${formatNumber(tick * 100, 0)}%</text>`).join("")}
    ${segments.map((part) => {
      const points = part.map((item) => `${x(item.index).toFixed(2)},${y(item.value).toFixed(2)}`).join(" ");
      const area = `M ${x(part[0].index)} ${top + plotHeight} L ${points.replaceAll(" ", " L ")} L ${x(part.at(-1).index)} ${top + plotHeight} Z`;
      return `<path class="curve-area" d="${area}"></path><polyline class="curve-line" points="${points}"></polyline>`;
    }).join("")}
    ${series.map((item, index) => item.value === null ? `<line class="curve-grid" x1="${x(index)}" y1="${top}" x2="${x(index)}" y2="${top + plotHeight}"></line><text x="${x(index)}" y="${top + 13}" text-anchor="middle">missing</text>` : "").join("")}
    <circle class="curve-marker" cx="${x(important.index)}" cy="${y(important.value)}" r="5"></circle>
    <text class="curve-label" x="${Math.min(width - 150, Math.max(left, x(important.index) - 38))}" y="${Math.max(13, y(important.value) - 11)}">${formatNumber(important.value * 100, 1)}%</text>
    <text x="${left}" y="${height - 8}">${escapeHtml(series[0].label)}</text><text x="${left + plotWidth}" y="${height - 8}" text-anchor="end">${escapeHtml(series.at(-1).label)}</text>
  </svg>`;
}

function renderResultVisual(result) {
  const allItems = result.items || [];
  const items = allItems.slice(0, Math.max(1, state.resultVisibleCount));
  const visual = $("#result-visual");
  if (result.visual?.mode === "route_options") {
    const view = result.visual;
    const routes = view.routes || [];
    const cost = (value) => value == null ? "Unavailable" : `${formatNumber(value, 2)} ${escapeHtml(view.cost_unit || "units")}`;
    const distance = (value) => value == null ? "Distance not uploaded" : `${formatNumber(Number(value) / 1000, 2)} km`;
    const routeColour = (value) => /^#[0-9a-f]{6}$/i.test(String(value || "")) ? String(value) : "#1464d2";
    $("#result-visual-title").textContent = routes.length
      ? `${routes.length} feasible route option${routes.length === 1 ? "" : "s"} under the selected disruption`
      : "No feasible route under the selected disruption";
    if (!routes.length) {
      visual.innerHTML = `<div class="route-options-empty"><strong>No route is available</strong><span>${escapeHtml(routeUnavailableMessage(view.reason_code, view.reason))}</span></div>`;
      return;
    }
    const availability = view.returned_count < view.requested_count
      ? `<p class="route-availability-note">Requested ${formatNumber(view.requested_count)}; ${formatNumber(view.returned_count)} sufficiently different feasible route${Number(view.returned_count) === 1 ? " is" : "s are"} available. Near-duplicate paths are not recoloured as new recommendations.</p>`
      : `<p class="route-availability-note">Showing all ${formatNumber(view.returned_count)} requested route options, ordered by total impedance.</p>`;
    visual.innerHTML = `<div class="route-options-visual">
      <header><strong>${escapeHtml(view.scenario_label || "Selected disruption")}</strong><small>Each colour matches the map legend and route table.</small></header>
      <div class="route-option-grid">${routes.map((route, index) => `<button class="route-option-card" type="button" data-result-index="${index}" style="--route-colour:${routeColour(route.colour)}" aria-label="Focus ${escapeHtml(route.label)} on the map">
        <span><i></i>${escapeHtml(route.label)}</span>
        <strong>${cost(route.impedance)}</strong>
        <small>${formatNumber(route.link_count)} roads · ${distance(route.distance_m)}</small>
        <em>${Number(route.rank) === 1 ? "Lowest total impedance" : `${route.impedance_change_from_primary == null ? "Alternative" : `${route.impedance_change_from_primary >= 0 ? "+" : ""}${formatNumber(route.impedance_change_from_primary * 100, 1)}% vs Route 1`}`}</em>
      </button>`).join("")}</div>
      ${availability}
    </div>`;
    return;
  }
  if (result.visual?.mode === "route_scenario_compare") {
    const view = result.visual;
    const variants = view.variants || [];
    const cost = (value) => value == null ? "No route" : `${formatNumber(value, 2)} ${escapeHtml(view.cost_unit || "units")}`;
    const routeColour = (value) => /^#[0-9a-f]{6}$/i.test(String(value || "")) ? String(value) : "#f4b400";
    const routeChange = (variant) => variant.detour_percent == null
      ? "No connected comparison"
      : `${Number(variant.detour_percent) >= 0 ? "+" : ""}${formatNumber(variant.detour_percent, 1)}% vs reference`;
    const referenceReason = routeUnavailableMessage(view.baseline_reason_code, view.baseline_reason);
    $("#result-visual-title").textContent = `Route recommendation by network state · ${variants.length} compared`;
    visual.innerHTML = `<div class="route-scenario-visual">
      <article class="route-reference-card">
        <span><i></i>Common benchmark</span>
        <strong>${escapeHtml(view.baseline_label || "Internal reference")}</strong>
        <b>${cost(view.baseline_impedance)}</b>
        <small>${view.baseline_reachable ? `${formatNumber(view.baseline_link_count)} links · grey dashed route on the map` : escapeHtml(referenceReason)}</small>
      </article>
      <div class="route-variant-grid">${variants.map((variant) => `<article class="route-variant-card${variant.reachable ? "" : " is-disconnected"}" style="--route-colour:${routeColour(variant.colour)}">
        <span><i></i>${escapeHtml(variant.kind === "historical" ? "Uploaded history" : "Saved scenario")}</span>
        <strong>${escapeHtml(variant.label)}</strong>
        <b>${variant.reachable ? cost(variant.impedance) : "No directed route"}</b>
        <small>${variant.reachable ? `${formatNumber(variant.link_count)} links · ${routeChange(variant)}` : escapeHtml(routeUnavailableMessage(variant.reason_code, variant.reason))}</small>
        <em>${variant.reachable ? `${formatNumber(variant.overlap_percent, 1)}% shared with reference · ${formatNumber(variant.changed_link_count)} links differ` : "Accessibility warning — no route invented"}</em>
      </article>`).join("")}</div>
      <p>Each coloured line is the independently recomputed least-impedance recommendation for that state. Colours identify states, not criticality or live congestion.</p>
    </div>`;
    return;
  }
  if (result.visual?.mode === "route_compare") {
    const view = result.visual;
    const cost = (value) => value == null ? "No route" : `${formatNumber(value, 2)} ${escapeHtml(view.cost_unit || "units")}`;
    const distance = (value) => value == null ? "Distance unavailable" : `${formatNumber(Number(value) / 1000, 2)} km · uploaded length`;
    const detour = view.detour_percent == null
      ? "No route"
      : `${Number(view.detour_percent) >= 0 ? "+" : ""}${formatNumber(view.detour_percent, 1)}%`;
    const routesComparable = Boolean(view.baseline_reachable && view.comparison_reachable);
    const baselineReason = routeUnavailableMessage(view.baseline_reason_code, view.baseline_reason);
    const comparisonReason = routeUnavailableMessage(view.comparison_reason_code, view.comparison_reason);
    const changedRoads = Number(view.changed_link_count || 0);
    const overlapPercent = Number(view.overlap_percent || 0);
    const routeInterpretation = !view.baseline_reachable
      ? baselineReason
      : !view.comparison_reachable
        ? comparisonReason
        : changedRoads === 0 && overlapPercent >= 99.999
          ? "Same road sequence. This is a travel-cost change on the existing route, not a detour."
          : `${formatNumber(changedRoads)} road${changedRoads === 1 ? "" : "s"} enter or leave the recommendation, so this is a genuine route change.`;
    $("#result-visual-title").textContent = "Baseline and two route recommendations";
    const alternatives = view.alternatives || [];
    visual.innerHTML = `<div class="route-compare-visual">
      <article class="route-state baseline"><span>Baseline route · blue</span><strong>${escapeHtml(view.baseline_label || "Baseline")}</strong><b>${cost(view.baseline_impedance)}</b><small>${view.baseline_reachable ? `${formatNumber(view.baseline_link_count)} links · ${distance(view.baseline_distance_m)}` : escapeHtml(baselineReason)}</small></article>
      <div class="route-change"><span>${escapeHtml(detour)}</span><small>${routesComparable ? "Total impedance change" : "A legal route is required in both states."}</small><i aria-hidden="true">→</i></div>
      <article class="route-state comparison"><span>Recommended route 1 · orange</span><strong>${escapeHtml(view.comparison_label || "Compared state")}</strong><b>${cost(view.comparison_impedance)}</b><small>${view.comparison_reachable ? `${formatNumber(view.comparison_link_count)} links · ${distance(view.comparison_distance_m)}` : escapeHtml(comparisonReason)}</small></article>
      <div class="route-overlap${routesComparable ? "" : " is-unavailable"}"><span><b style="width:${routesComparable ? Math.max(0, Math.min(100, overlapPercent)) : 0}%"></b></span><strong>${routesComparable ? `${formatNumber(overlapPercent, 1)}% route overlap` : "No connected comparison"}</strong><small>${routesComparable ? `${formatNumber(changedRoads)} roads enter or leave the route` : "No route was available in both states."}</small></div>
      <p class="route-interpretation">${escapeHtml(routeInterpretation)}</p>
      ${alternatives.length ? `<div class="route-alternative-grid">${alternatives.map((route) => `<article style="--route-colour:${escapeHtml(route.colour || "#0f766e")}"><span><i></i>${escapeHtml(route.label)}</span><strong>${cost(route.impedance)}</strong><small>${formatNumber(route.link_count)} roads · ${route.impedance_change_from_primary == null ? "" : `${route.impedance_change_from_primary >= 0 ? "+" : ""}${formatNumber(route.impedance_change_from_primary * 100, 1)}% vs primary`} · ${formatNumber((route.overlap_with_primary || 0) * 100, 1)}% overlap</small></article>`).join("")}</div>` : ""}
    </div>`;
    return;
  }
  if (!items.length) {
    $("#result-visual-title").textContent = "No plottable result";
    visual.innerHTML = '<div class="empty-state">No visual mark was returned.</div>';
    return;
  }
  if (result.visual?.mode === "network_stress_curve") {
    const view = result.visual;
    const threshold = (key) => view.thresholds?.[key] ? `Stage ${view.thresholds[key]}` : "Not reached";
    $("#result-visual-title").textContent = "When cumulative key-road failures weaken the network";
    visual.innerHTML = `<div class="stress-visual">${renderCurveVisual(items, "performance_ratio", "Network efficiency remaining after each cumulative road failure", true)}<div class="stress-threshold-grid"><article><span>90% left</span><strong>${threshold("90_percent")}</strong></article><article><span>75% left</span><strong>${threshold("75_percent")}</strong></article><article><span>50% left</span><strong>${threshold("50_percent")}</strong></article><article><span>First split</span><strong>${view.first_split_stage ? `Stage ${view.first_split_stage}` : "Not observed"}</strong></article></div></div>`;
    $("#result-visual-title").textContent = "When cumulative key-road failures weaken the network";
    return;
  }
  if (result.visual?.mode === "recovery_priority") {
    const view = result.visual;
    if (view.method === "adaptive_marginal_efficiency") {
      visual.innerHTML = renderCurveVisual(items, "cumulative_recovery", `Cumulative network service restored by ${view.recovery_unit} priority`);
    } else {
      const maximum = Math.max(...items.map((item) => Number(item.priority_score) || 0), 1e-12);
      $("#result-visual-title").textContent = "Structural recovery screening · highest first";
      visual.innerHTML = `<div class="result-rank-list">${items.map((item, index) => `<button class="result-rank-button" type="button" data-result-index="${index}"><span class="result-rank-number">#${item.rank || index + 1}</span><span class="result-rank-name">${escapeHtml(resultItemName(item))}</span><span class="result-bar-track"><i style="width:${Math.max(1.5, Number(item.priority_score || 0) / maximum * 100).toFixed(2)}%"></i></span><span class="result-rank-value">${formatNumber(item.priority_score, 5)}</span></button>`).join("")}</div><p class="visual-boundary-note">This large-candidate tier is a structural screen. It does not display an uncomputed service-recovery curve.</p>`;
    }
    return;
  }
  if (Object.hasOwn(items[0], "performance")) {
    visual.innerHTML = renderCurveVisual(items, "performance", "Observed service remaining over time · 100% is the internal baseline");
    return;
  }
  if (Object.hasOwn(items[0], "performance_ratio")) {
    visual.innerHTML = renderCurveVisual(items, "performance_ratio", "Topology efficiency remaining as ranked roads are removed", true);
    return;
  }
  if (Object.hasOwn(items[0], "cumulative_recovery")) {
    visual.innerHTML = renderCurveVisual(items, "cumulative_recovery", "Cumulative topology recovery by stage");
    return;
  }
  const metric = resultBarMetric(items[0]);
  if (metric) {
    const visible = items;
    const maximum = Math.max(...visible.map((item) => Number(item[metric.key]) || 0), 1e-12);
    $("#result-visual-title").textContent = `${metric.label}: ${visible.length === allItems.length ? "all" : "first"} ${visible.length} of ${allItems.length} returned results`;
    visual.innerHTML = `<div class="result-rank-list">${visible.map((item, index) => {
      const value = Math.max(0, Number(item[metric.key]) || 0);
      const label = metric.percent ? `${formatNumber(value * 100, metric.digits)}%` : formatNumber(value, metric.digits);
      const width = Math.max(1.5, value / maximum * 100);
      return `<button class="result-rank-button" type="button" data-result-index="${index}" aria-label="Focus ${escapeHtml(resultItemName(item))} on the map"><span class="result-rank-number">#${item.rank || index + 1}</span><span class="result-rank-name">${escapeHtml(resultItemName(item))}</span><span class="result-bar-track"><i style="width:${width.toFixed(2)}%"></i></span><span class="result-rank-value">${escapeHtml(label)}</span></button>`;
    }).join("")}</div>`;
    return;
  }
  $("#result-visual-title").textContent = `${items.length} roads in the confirmed scenario`;
  visual.innerHTML = `<div class="result-link-strip">${items.slice(0, 18).map((item) => `<div><strong>${escapeHtml(resultItemName(item))}</strong><small>${escapeHtml(item.from_node || "?")} → ${escapeHtml(item.to_node || "?")}</small></div>`).join("")}</div>`;
}

function focusResultOnMap(collection = state.analysisCollection) {
  if (!collection?.features?.length) return toast("No mapped result", "This result does not contain a geographic feature.", true);
  switchView("explore");
  state.map?.setAnalysisFeatures(collection);
  state.map?.focusCollection(collection);
  window.setTimeout(() => {
    $("#map-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    highlightElement($("#map-panel"));
  }, 120);
}

function focusResultItem(index) {
  const item = state.currentResult?.items?.[Number(index)];
  const features = state.analysisCollection?.features || [];
  if (item?.route_summary && item.route_variant_id) {
    const routeFeatures = features
      .filter((candidate) => String(candidate.properties?.route_variant_id || "") === String(item.route_variant_id))
      .map((candidate) => ({
        ...candidate,
        properties: { ...(candidate.properties || {}), route_map_offset: 0 },
      }));
    if (!routeFeatures.length) return toast("Route geometry unavailable", "This route has no defensible mapped line geometry in the uploaded data.", true);
    focusResultOnMap({ type: "FeatureCollection", features: routeFeatures });
    return;
  }
  const feature = features.find((candidate) => {
    const properties = candidate.properties || {};
    if (item?.key) return String(properties.key) === String(item.key)
      && (!item.route_role || properties.route_role === item.route_role)
      && (!item.route_variant_id || properties.route_variant_id === item.route_variant_id);
    if (item?.cell_id) return String(properties.cell_id) === String(item.cell_id);
    if (item?.stage) return Number(properties.stage) === Number(item.stage);
    return item?.entity_id && String(properties.entity_id) === String(item.entity_id);
  });
  if (!feature) return toast("No mapped location", "This returned row has no valid geographic coordinate to focus.", true);
  if (item?.key) state.map?.setSelectedKeys([String(item.key)]);
  focusResultOnMap({ type: "FeatureCollection", features: [feature] });
}

function renderResultTable(allItems) {
  const items = (allItems || []).slice(0, Math.max(1, state.resultVisibleCount));
  const head = $("#result-table-head");
  const body = $("#result-table-body");
  if (!items.length) {
    head.innerHTML = "<tr><th>Result</th></tr>";
    body.innerHTML = "<tr><td>No ranked item was returned.</td></tr>";
    return;
  }
  if (items[0].route_summary) {
    head.innerHTML = "<tr><th>Rank</th><th>Map colour</th><th>Route</th><th>Total impedance</th><th>Roads</th><th>Distance</th><th>Difference from Route 1</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>#${formatNumber(item.route_rank)}</td><td><span class="table-route-swatch" style="background:${escapeHtml(item.route_colour)}"></span></td><td>${escapeHtml(item.route_label)}</td><td>${formatNumber(item.impedance, 2)} ${escapeHtml(item.cost_unit || "units")}</td><td>${formatNumber(item.link_count)}</td><td>${item.distance_m == null ? "Not uploaded" : `${formatNumber(Number(item.distance_m) / 1000, 2)} km`}</td><td>${Number(item.route_rank) === 1 ? "Best route" : item.impedance_change_from_primary == null ? "Alternative" : `${item.impedance_change_from_primary >= 0 ? "+" : ""}${formatNumber(item.impedance_change_from_primary * 100, 1)}%`}</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "route_role")) {
    head.innerHTML = "<tr><th>Network state / route</th><th>Step</th><th>Road link</th><th>Direction</th><th>Segment impedance</th><th>Length</th><th>Shared with reference?</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${escapeHtml(item.route_label)}</td><td>${formatNumber(item.sequence)}</td><td>${escapeHtml(item.entity_id)}</td><td>${escapeHtml(item.from_node)} → ${escapeHtml(item.to_node)}</td><td>${formatNumber(item.segment_impedance, 3)}</td><td>${item.length_m == null ? "Not uploaded" : `${formatNumber(item.length_m, 1)} m`}</td><td>${item.shared_with_other_route ? "Yes" : "Changed"}</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "performance")) {
    head.innerHTML = "<tr><th>Time</th><th>Service</th><th>Loss</th><th>Valid entities</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${escapeHtml(item.label)}</td><td>${item.performance == null ? "Missing" : `${formatNumber(item.performance * 100, 1)}%`}</td><td>${item.loss == null ? "Not calculated" : `${formatNumber(item.loss * 100, 1)}%`}</td><td>${formatNumber(item.valid_entity_count)}</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "robustness_loss")) {
    head.innerHTML = "<tr><th>Stage</th><th>Removed road</th><th>Efficiency left</th><th>Weak components</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${item.stage}</td><td>${escapeHtml(item.entity_id)} · ${escapeHtml(item.region_id)}</td><td>${formatNumber(item.performance_ratio * 100, 1)}%</td><td>${formatNumber(item.weak_components)}</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "cell_id")) {
    if (Object.hasOwn(items[0], "regional_dpc_score")) {
      head.innerHTML = "<tr><th>Rank</th><th>Geographic cell</th><th>Regional DPC loss</th><th>Observed mean degradation</th><th>Mapped entities</th></tr>";
      body.innerHTML = items.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.cell_id)}</td><td>${formatNumber(item.regional_dpc_score * 100, 3)}%</td><td>${formatNumber(item.mean_degradation * 100, 1)}%</td><td>${formatNumber(item.entity_count)}</td></tr>`).join("");
    } else {
      head.innerHTML = "<tr><th>Rank</th><th>Geographic cell</th><th>Observed mean degradation</th><th>Mapped entities</th></tr>";
      body.innerHTML = items.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.cell_id)}</td><td>${formatNumber(item.mean_degradation * 100, 1)}%</td><td>${formatNumber(item.entity_count)}</td></tr>`).join("");
    }
  } else if (Object.hasOwn(items[0], "area_bucket")) {
    head.innerHTML = "<tr><th>Condition group</th><th>Mapped entities</th><th>Share of selected area</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${escapeHtml(item.area_bucket)}</td><td>${formatNumber(item.count)}</td><td>${formatNumber(item.share * 100, 1)}%</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "priority_score") && items[0].recovery_unit) {
    head.innerHTML = "<tr><th>Priority</th><th>Road or area</th><th>Roads restored</th><th>Model value</th><th>Cumulative result</th><th>Repair cost</th><th>Repair time</th></tr>";
    body.innerHTML = items.map((item) => {
      const adaptive = item.marginal_efficiency_gain != null;
      const cumulative = item.cumulative_recovery != null
        ? `${formatNumber(item.cumulative_recovery * 100, 1)}% service restored`
        : `${formatNumber((item.cumulative_priority_coverage || 0) * 100, 1)}% screened priority coverage`;
      return `<tr><td>#${item.rank || item.stage}</td><td>${escapeHtml(resultItemName(item))}</td><td>${formatNumber(item.road_count || 1)}</td><td>${adaptive ? `${formatNumber(item.marginal_efficiency_gain * 100, 4)}% efficiency gain` : `${formatNumber(item.priority_score, 6)} structural score`}</td><td>${escapeHtml(cumulative)}</td><td>${item.repair_cost == null ? "Not uploaded" : formatNumber(item.repair_cost, 2)}</td><td>${item.repair_duration == null ? "Not uploaded" : formatNumber(item.repair_duration, 2)}</td></tr>`;
    }).join("");
  } else if (items[0].stage) {
    head.innerHTML = "<tr><th>Stage</th><th>Road</th><th>Region</th><th>Marginal gain</th><th>Cumulative recovery</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${item.stage}</td><td>${escapeHtml(item.entity_id)}</td><td>${escapeHtml(item.region_id)}</td><td>${formatNumber(item.marginal_efficiency_gain, 6)}</td><td>${formatNumber(item.cumulative_recovery * 100, 1)}%</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "persistence")) {
    head.innerHTML = "<tr><th>Rank</th><th>Road / location</th><th>Region</th><th>Observed time degraded</th><th>Mean degradation</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.entity_id)}</td><td>${escapeHtml(item.region_id)}</td><td>${formatNumber(item.persistence * 100, 1)}%</td><td>${formatNumber(item.mean_degradation * 100, 1)}%</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "efficiency_loss")) {
    head.innerHTML = "<tr><th>Rank</th><th>Road</th><th>Region</th><th>Efficiency loss</th><th>Single connection</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.entity_id)}</td><td>${escapeHtml(item.region_id)}</td><td>${formatNumber(item.efficiency_loss * 100, 3)}%</td><td>${item.is_bridge ? "Yes" : "No"}</td></tr>`).join("");
  } else if (Object.hasOwn(items[0], "structural_score")) {
    head.innerHTML = "<tr><th>Rank</th><th>Road</th><th>Region</th><th>Structural score</th><th>Single connection</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${item.rank}</td><td>${escapeHtml(item.entity_id)}</td><td>${escapeHtml(item.region_id)}</td><td>${formatNumber(item.structural_score * 100, 2)}%</td><td>${item.is_bridge ? "Yes" : "No"}</td></tr>`).join("");
  } else {
    head.innerHTML = "<tr><th>Road</th><th>Region</th><th>From</th><th>To</th></tr>";
    body.innerHTML = items.map((item) => `<tr><td>${escapeHtml(item.entity_id || item.key)}</td><td>${escapeHtml(item.region_id)}</td><td>${escapeHtml(item.from_node)}</td><td>${escapeHtml(item.to_node)}</td></tr>`).join("");
  }
  if (Object.hasOwn(items[0], "area_bucket")) return;
  const headerRow = head.querySelector("tr");
  if (headerRow) headerRow.insertAdjacentHTML("beforeend", "<th>Map</th>");
  [...body.querySelectorAll("tr")].forEach((row, index) => {
    const featureAvailable = (state.analysisCollection?.features || []).some((feature) => {
      const properties = feature.properties || {};
      const item = items[index] || {};
      return (item.route_summary && item.route_variant_id && String(properties.route_variant_id) === String(item.route_variant_id))
        || (item.key && String(properties.key) === String(item.key)
          && (!item.route_role || properties.route_role === item.route_role)
          && (!item.route_variant_id || properties.route_variant_id === item.route_variant_id))
        || (item.cell_id && String(properties.cell_id) === String(item.cell_id))
        || (item.stage && Number(properties.stage) === Number(item.stage))
        || (item.entity_id && String(properties.entity_id) === String(item.entity_id));
    });
    row.insertAdjacentHTML(
      "beforeend",
      `<td>${featureAvailable ? `<button class="table-map-button ripple" type="button" data-result-index="${index}">Focus</button>` : "—"}</td>`,
    );
  });
}

function renderResultPagination(items = state.currentResult?.items || []) {
  const actions = $("#result-page-actions");
  const more = items.length > state.resultVisibleCount;
  actions.classList.toggle("is-hidden", !more);
  if (!more) return;
  const remaining = items.length - state.resultVisibleCount;
  $("#result-show-more").textContent = `Show next ${formatNumber(Math.min(200, remaining))}`;
  $("#result-show-all").textContent = `Show all ${formatNumber(items.length)} returned rows`;
}

function expandResultRows(showAll = false) {
  const items = state.currentResult?.items || [];
  state.resultVisibleCount = showAll ? items.length : Math.min(items.length, state.resultVisibleCount + 200);
  renderResultVisual(state.currentResult);
  renderResultTable(items);
  renderResultPagination(items);
  const requested = Number(state.currentResult?.request?.top_k || 0);
  $("#result-list-count").textContent = `${requested ? `Requested ${formatNumber(requested)}; ` : ""}returned ${formatNumber(items.length)}; displaying ${formatNumber(state.resultVisibleCount)} rows. Map highlighting includes the full returned collection.`;
}

function renderDataView() {
  const summary = state.project?.summary;
  if (!summary) {
    $("#data-project-title").textContent = "No data project";
    $("#data-project-hazard").textContent = "—";
    $("#data-stats").innerHTML = "";
    $("#capability-list").innerHTML = '<div class="empty-state">Upload data to activate the capability check.</div>';
    $("#quality-grid").innerHTML = "";
    $("#source-list").innerHTML = '<div class="empty-state">No source files loaded.</div>';
    return;
  }
  $("#data-project-title").textContent = state.project.name;
  $("#data-project-hazard").textContent = state.project.hazard_type.replaceAll("_", " ");
  const stats = [
    ["Mapped entities", formatNumber(summary.mapped_entity_count ?? summary.entity_count), `${formatNumber(summary.unmapped_entity_count || 0)} excluded · ${summary.region_count} regions`],
    ["Observations", formatNumber(summary.observation_count), "Uploaded time-stamped records"],
    ["Time points", formatNumber(summary.time_value_count || summary.time_values?.length || 1), `${summary.time_start} → ${summary.time_end}${summary.timeline_sampled ? " · control is evenly sampled" : ""}`],
    ["Geometry", projectUsesLinkTopology(summary) ? "Link reference points" : "Entity points", geometryNoteForProject(summary)],
  ];
  $("#data-stats").innerHTML = stats.map(([label, value, detail]) => `<div class="data-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></div>`).join("");
  $("#data-boundary").textContent = summary.baseline?.warning || summary.geometry_note;
  $("#capability-list").innerHTML = Object.entries(summary.capabilities || {}).map(([key, status]) => `<div class="capability-item ${escapeHtml(status)}"><i></i><strong>${escapeHtml(CAPABILITY_LABELS[key] || key.replaceAll("_", " "))}</strong><span>${escapeHtml(status)}</span></div>`).join("");
  const qualityLabels = {
    invalid_coordinates: "Invalid or missing coordinates",
    invalid_times: "Unreadable times",
    negative_speeds_invalid: "Invalid negative speeds",
    zero_speeds_observed: "Observed zero speeds",
    nonpositive_travel_times: "Invalid travel times",
    duplicate_entity_times: "Duplicate entity-times combined",
    missing_entity_ids: "Missing entity IDs",
  };
  const qualityEntries = Object.entries(summary.quality || {})
    .filter(([key, value]) => Object.hasOwn(qualityLabels, key) && Number.isFinite(Number(value)));
  const denominator = Math.max(1, Number(summary.observation_count || 0));
  $("#quality-grid").innerHTML = qualityEntries.map(([key, raw]) => {
    const value = Number(raw || 0);
    const ratio = Math.min(1, value / denominator);
    const informational = key === "zero_speeds_observed" || key === "duplicate_entity_times";
    return `<div class="quality-card ${informational ? "is-information" : value ? "has-issue" : "is-clear"}"><strong>${formatNumber(value)}</strong><span>${escapeHtml(qualityLabels[key])}</span><small>${(ratio * 100).toFixed(ratio > 0 && ratio < 0.001 ? 3 : 1)}% of stored observations</small><i><b style="width:${Math.max(value ? 2 : 0, ratio * 100).toFixed(2)}%"></b></i></div>`;
  }).join("");
  $("#source-list").innerHTML = (summary.source_files || []).map((file) => `<div class="source-item"><div><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(file.region_id)} · ${formatBytes(file.size_bytes)} · SHA-256 ${escapeHtml(file.sha256?.slice(0, 12))}…</small></div><span>LOCAL</span></div>`).join("");
}

function legacyRenderModels(models) {
  $("#model-grid").innerHTML = models.map((model) => {
    const card = model.card || {};
    const requirements = card.required_data || ["Confirmed uploaded fields"];
    const limitations = card.known_limitations || [card.status_reason || "See model card"];
    const source = card.primary_sources?.[0]?.citation || "Transport world model evidence card";
    const method = card.performance || card.ranking_rule || model.family?.replaceAll("_", " ") || "Data analysis";
    const statusLabel = {
      numerically_verified: "Ready",
      literature_defined: "Needs data",
      formulation_reserved: "Future data",
      adaptive_executable_family: "Executable family",
    }[model.status] || model.status.replaceAll("_", " ");
    return `<article class="model-card">
      <div class="model-card-head"><span class="model-id">${escapeHtml(publicModelId(model.id))}</span><span class="model-status ${escapeHtml(model.status)}">${escapeHtml(statusLabel)}</span></div>
      <h3>${escapeHtml(model.name)}</h3>
      <p>${escapeHtml(card.purpose || model.family?.replaceAll("_", " ") || "Versioned disaster transport model")}</p>
      <dl><dt>Needs</dt><dd>${escapeHtml(requirements.slice(0, 3).join(" · "))}</dd><dt>Method</dt><dd>${escapeHtml(method)}</dd><dt>Source</dt><dd>${escapeHtml(source)}</dd></dl>
      <details><summary>Assumptions and limits</summary><pre>${escapeHtml(limitations.join("\n"))}</pre></details>
    </article>`;
  }).join("");
}

function compactValidationNumber(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) > 0 && Math.abs(number) < 0.001) return number.toExponential(2);
  return number.toFixed(digits).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function renderScientificValidation(report, scope = "release") {
  if (!report) return;
  state.scientificValidation = report;
  const summary = report.summary || {};
  const checks = report.checks || [];
  const external = report.external_data_matrix?.summary || {};
  const status = $("#validation-status");
  const passed = report.status === "passed";
  status.className = `validation-status ${passed ? "is-passed" : report.status === "not_run" ? "" : "is-attention"}`;
  status.textContent = report.status === "not_run"
    ? "Release report unavailable"
    : `${scope === "project" ? "Current project" : "Packaged release"} · ${formatNumber(summary.passed || 0)}/${formatNumber(summary.total || 0)} passed`;
  const cards = [
    [scope === "project" ? "Project result" : "Release result", `${formatNumber(summary.passed || 0)} / ${formatNumber(summary.total || 0)}`, summary.plain_language || "Validation evidence is shown below."],
    [scope === "project" ? "Network" : "External data", scope === "project" && summary.network_nodes != null ? `${formatNumber(summary.network_nodes)} nodes` : external.valid_cases_total != null ? `${formatNumber(external.valid_cases_passed)}/${formatNumber(external.valid_cases_total)} passed` : "Independent fixtures", scope === "project" && summary.network_edges != null ? `${formatNumber(summary.network_edges)} directed roads` : external.regions ? `${formatNumber(external.regions.length)} regions · ${formatNumber(external.hazards?.length || 0)} hazards` : "Multiple hazards and schemas"],
    ["Comparison depth", summary.top_k != null ? `Top ${formatNumber(summary.top_k)}` : "Baseline suite", "NetworkX · traditional · ablation"],
    ["Repeatability", report.repeatability?.second_run_matches === false ? "Mismatch" : "Matched", report.repeatability?.critical_roads_sha256 ? `SHA-256 ${report.repeatability.critical_roads_sha256.slice(0, 10)}…` : "Deterministic reruns"],
    ["Runtime", summary.runtime_seconds != null ? `${compactValidationNumber(summary.runtime_seconds, 2)} s` : "Packaged", "Recorded for this validation run"],
  ];
  $("#validation-summary").innerHTML = cards.map(([label, value, detail]) => `<article><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`).join("");
  $("#validation-checks").innerHTML = checks.length
    ? checks.map((check) => `<span class="validation-check ${check.passed ? "" : "is-failed"}" title="value ${escapeHtml(compactValidationNumber(check.value))}; gate ${escapeHtml(compactValidationNumber(check.threshold))}"><i>${check.passed ? "✓" : "!"}</i>${escapeHtml(check.label)}</span>`).join("")
    : '<span class="validation-check is-failed"><i>!</i>No release checks were found</span>';
  const comparisons = report.comparisons || {};
  const critical = comparisons.critical_roads?.curves || {};
  const recovery = comparisons.recovery_priorities?.curves || {};
  const detailCards = [
    ["Critical roads", critical.platform ? `Mean loss ${compactValidationNumber(critical.platform.mean_individual_loss)}` : "See release report", "Compared with NetworkX edge betweenness, degree-only ablation and fixed-seed random orders."],
    ["Recovery priorities", recovery.platform ? `Curve area ${compactValidationNumber(recovery.platform.auc)}` : "See release report", "Compared with static criticality, reverse order and fixed-seed random restoration."],
    ["Route and stress oracles", comparisons.routing ? `Max route error ${compactValidationNumber(comparisons.routing.maximum_absolute_error)}` : "Independent recomputation", comparisons.network_resilience ? `Stress max error ${compactValidationNumber(comparisons.network_resilience.maximum_absolute_error)}` : "Dijkstra and cumulative-efficiency oracles"],
  ];
  $("#validation-detail-body").innerHTML = `<div class="validation-detail-grid">${detailCards.map(([title, value, copy]) => `<article><strong>${escapeHtml(title)} · ${escapeHtml(value)}</strong><small>${escapeHtml(copy)}</small></article>`).join("")}</div><p class="validation-boundary">${escapeHtml(report.scientific_boundary || "Validation supports numerical and reproducibility claims for the tested data and parameters; it is not a universal operational-safety claim.")}</p>`;
  $("#run-project-validation").disabled = !state.project || state.project.summary?.capabilities?.topology_criticality === "unavailable";
}

async function loadReleaseScientificValidation() {
  try {
    state.releaseScientificValidation = await api.releaseScientificValidation();
    renderScientificValidation(state.releaseScientificValidation, "release");
  } catch (error) {
    renderScientificValidation({ status: "not_run", summary: { passed: 0, total: 0, plain_language: error.message }, checks: [] }, "release");
  }
}

async function runProjectScientificValidation() {
  if (!state.project) return toast("Open a project first", "Project-specific validation needs uploaded topology data.", true);
  const topK = Number($("#validation-top-k").value);
  if (!Number.isInteger(topK) || topK < 3 || topK > 25) return toast("Choose 3 to 25 roads", "This independent comparison suite deliberately caps the validation depth at 25 to keep repeated baseline runs tractable.", true);
  const button = $("#run-project-validation");
  button.disabled = true;
  button.textContent = "Running independent checks…";
  setLoading(true, "Validating the current project", "Repeating the model, comparing NetworkX and traditional baselines, then checking sensitivity and independent numerical oracles.");
  try {
    const report = await api.projectScientificValidation(state.project.project_id, topK);
    renderScientificValidation(report, "project");
    $("#validation-details").open = true;
    toast("Scientific validation complete", `${formatNumber(report.summary?.passed || 0)} of ${formatNumber(report.summary?.total || 0)} declared gates passed; no result is hidden.`);
  } catch (error) {
    toast("Scientific validation did not complete", error.message, true);
  } finally {
    setLoading(false);
    button.textContent = "Validate current project";
    button.disabled = !state.project || state.project.summary?.capabilities?.topology_criticality === "unavailable";
  }
}

const RESEARCH_PALETTE = {
  blue: "#0072b2",
  orange: "#e69f00",
  teal: "#009e73",
  purple: "#7b61a8",
  sky: "#56b4e9",
  grey: "#747b80",
  ink: "#263238",
  grid: "#d8dfdf",
};

function researchGradeLabel(grade) {
  return {
    strong: "Strong evidence",
    limited: "Limited by data",
    attention: "Needs attention",
    not_testable: "Not testable",
  }[grade] || "Evidence recorded";
}

function researchMetricValue(metric) {
  const value = metric?.value;
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Not available";
  if (metric.format === "percent") return `${(Number(value) * 100).toFixed(Number(value) === 0 || Number(value) === 1 ? 0 : 1)}%`;
  if (metric.format === "integer") return formatNumber(Number(value));
  if (metric.format === "scientific") return Number(value) === 0 ? "0" : Number(value).toExponential(2);
  if (metric.format === "binary") return Number(value) ? "Yes" : "No";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function renderResearchWorkspace() {
  const hasProject = Boolean(state.project);
  const running = Boolean(state.researchJobId);
  const hasReport = Boolean(state.researchReport);
  const runButton = $("#research-run");
  if (runButton) {
    runButton.disabled = !hasProject || running;
    runButton.textContent = running ? "Validation running" : "Run complete TWM validation";
  }
  $("#research-empty")?.classList.toggle("is-hidden", hasReport || running);
  $("#research-report")?.classList.toggle("is-hidden", !hasReport);
  $("#research-progress-panel")?.classList.toggle("is-hidden", !running);
  if (!hasProject && $("#research-empty")) {
    $("#research-empty h3").textContent = "Open a project to create its research evidence package.";
    $("#research-empty p").textContent = "The validation protocol uses the current upload; nothing is loaded automatically.";
  } else if (hasProject && !hasReport && !running && $("#research-empty")) {
    $("#research-empty h3").textContent = "Run one fixed, whole-project validation protocol.";
    $("#research-empty p").textContent = "Existing map selections and saved scenarios are not changed. Unsupported evidence is labelled—not invented.";
  }
}

async function refreshResearchReports({ loadLatest = false } = {}) {
  if (!state.project) {
    state.researchReports = [];
    state.researchReport = null;
    renderResearchWorkspace();
    return;
  }
  const projectId = state.project.project_id;
  const reports = await api.researchValidationReports(projectId);
  if (state.project?.project_id !== projectId) return;
  state.researchReports = reports;
  const picker = $("#research-report-picker");
  if (picker) {
    picker.innerHTML = reports.length
      ? reports.map((item) => `<option value="${escapeHtml(item.report_id)}">${escapeHtml(new Date(item.generated_at).toLocaleString())} · ${escapeHtml(item.profile || "standard")}</option>`).join("")
      : '<option value="">No saved reports</option>';
  }
  const currentStillExists = reports.some((item) => item.report_id === state.researchReport?.report_id);
  if ((loadLatest || !currentStillExists) && reports[0]?.report_id) {
    state.researchReport = await api.researchValidationReport(reports[0].report_id);
  } else if (!reports.length) {
    state.researchReport = null;
  }
  renderResearchWorkspace();
  if (state.researchReport) renderResearchReport(state.researchReport);
}

async function openResearchReport(reportId) {
  if (!reportId) return;
  const report = await api.researchValidationReport(reportId);
  if (!state.project || report.project_id !== state.project.project_id) return;
  state.researchReport = report;
  renderResearchWorkspace();
  renderResearchReport(report);
}

async function runResearchValidation() {
  if (!state.project || state.researchJobId) return;
  const projectId = state.project.project_id;
  try {
    const queued = await api.startResearchValidation(projectId, "standard");
    state.researchJobId = queued.job_id;
    renderResearchWorkspace();
    $("#research-progress-title").textContent = "Validating the complete TWM evidence loop";
    $("#research-progress-message").textContent = "Freezing the project-specific protocol.";
    $("#research-progress-bar").style.width = "1%";
    $("#research-progress-percent").textContent = "1%";
    while (state.researchJobId === queued.job_id) {
      const job = await api.job(queued.job_id);
      if (state.project?.project_id !== projectId) return;
      const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
      $("#research-progress-message").textContent = job.message || "Running the registered validation protocol.";
      $("#research-progress-bar").style.width = `${progress}%`;
      $("#research-progress-percent").textContent = `${Math.round(progress)}%`;
      if (job.status === "completed") {
        state.researchReport = job.result;
        state.researchJobId = null;
        await refreshResearchReports({ loadLatest: false });
        renderResearchWorkspace();
        renderResearchReport(job.result);
        toast("Research validation complete", "The reviewer evidence package, figures and reproduction receipt are ready.");
        return;
      }
      if (job.status === "failed") throw new Error(job.error || "The research validation did not complete.");
      if (job.status === "cancelled") {
        state.researchJobId = null;
        renderResearchWorkspace();
        toast("Research validation cancelled", "No completed research report was saved.");
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 550));
    }
  } catch (error) {
    state.researchJobId = null;
    renderResearchWorkspace();
    toast("Research validation did not complete", error.message, true);
  }
}

async function cancelResearchValidation() {
  const jobId = state.researchJobId;
  if (!jobId) return;
  state.researchJobId = null;
  renderResearchWorkspace();
  try {
    await api.cancelJob(jobId);
    toast("Validation cancelled", "The operational project and its saved results were not changed.");
  } catch (error) {
    toast("Cancel request failed", error.message, true);
  }
}

function researchSummaryCards(report) {
  const summary = report.summary || {};
  return [
    ["Evidence loop", `${formatNumber(summary.strong_stages || 0)} strong`, `${formatNumber(summary.limited_stages || 0)} data-limited stages`],
    ["Project scale", formatNumber(summary.observations || 0), `${formatNumber(summary.network_nodes || 0)} nodes · ${formatNumber(summary.network_edges || 0)} links`],
    ["Independent gates", `${formatNumber(report.core_scientific_validation?.summary?.passed || 0)} / ${formatNumber(report.core_scientific_validation?.summary?.total || 0)}`, "Numerical and repeatability checks"],
    ["Model coverage", `${formatNumber(report.model_tiers?.executable || 0)} / ${formatNumber(report.model_tiers?.registered || 0)}`, `${formatNumber(report.model_tiers?.highest_fidelity_selected || 0)} highest-fidelity tiers`],
    ["Runtime", `${Number(summary.runtime_seconds || 0).toFixed(1)} s`, `Protocol ${report.protocol || "—"}`],
  ];
}

function renderResearchReport(report) {
  if (!report) return;
  state.researchFigureSvgs = new Map();
  $("#research-report-title").textContent = report.summary?.title || "Transport World Model research validation";
  $("#research-report-subtitle").textContent = `${report.project_name || "Current project"} · ${new Date(report.generated_at).toLocaleString()} · whole uploaded project`;
  $("#research-summary-grid").innerHTML = researchSummaryCards(report).map(([label, value, detail]) => `<article class="research-summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small title="${escapeHtml(detail)}">${escapeHtml(detail)}</small></article>`).join("");
  $("#research-overall-status").textContent = report.status === "attention_required" ? "Attention required" : "Evidence available";
  $("#research-loop").innerHTML = (report.stages || []).map((stage) => `<button class="research-loop-stage" data-grade="${escapeHtml(stage.grade)}" data-research-stage-target="${escapeHtml(stage.id)}" type="button"><span>${String(stage.index).padStart(2, "0")}</span><h4>${escapeHtml(stage.label)}</h4><p>${escapeHtml(stage.interpretation)}</p><b>${escapeHtml(researchGradeLabel(stage.grade))}</b></button>`).join("");
  $("#research-stage-list").innerHTML = (report.stages || []).map((stage, index) => `<details class="research-stage-card" data-research-stage="${escapeHtml(stage.id)}"${index === 0 ? " open" : ""}><summary><span>${String(stage.index).padStart(2, "0")}</span><strong>${escapeHtml(stage.label)}</strong><p>${escapeHtml(stage.claim)}</p><b class="research-grade" data-grade="${escapeHtml(stage.grade)}">${escapeHtml(researchGradeLabel(stage.grade))}</b></summary><div class="research-stage-body"><article><h5>Method</h5><p>${escapeHtml(stage.method)}</p></article><article class="research-equation" aria-label="Core mathematical expression">${escapeHtml(stage.equation)}</article><article><h5>Evidence</h5><ul>${(stage.evidence || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></article><article><h5>What cannot be claimed</h5><p>${escapeHtml(stage.boundary)}</p></article><div class="research-stage-metrics">${(stage.metrics || []).map((metric) => `<span><b>${escapeHtml(researchMetricValue(metric))}</b><small>${escapeHtml(metric.label)}</small></span>`).join("")}</div></div></details>`).join("");
  const receipt = report.reproducibility || {};
  $("#research-receipt-copy").textContent = "Re-run with the same registered inputs to reproduce the same evidence identity. Generated time is excluded from the scientific fingerprint.";
  $("#research-receipt-values").innerHTML = [
    ["Data SHA-256", receipt.data_sha256], ["Report SHA-256", receipt.report_sha256],
    ["Software", receipt.software_version], ["Model registry", receipt.model_registry_version],
    ["Protocol", receipt.protocol], ["Fixed seeds", (receipt.fixed_seeds || []).join(", ")],
  ].map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd title="${escapeHtml(value || "Not available")}">${escapeHtml(value || "Not available")}</dd></div>`).join("");
  $("#research-claim-boundary").textContent = report.claim_boundary || "Interpret only within the uploaded data and registered protocol.";
  const reportId = encodeURIComponent(report.report_id);
  $("#research-export-json").href = `/api/research-validation/${reportId}/export.json`;
  $("#research-export-csv").href = `/api/research-validation/${reportId}/export.csv`;
  $("#research-export-tex").href = `/api/research-validation/${reportId}/export.tex`;
  const picker = $("#research-report-picker");
  if (picker && [...picker.options].some((option) => option.value === report.report_id)) picker.value = report.report_id;
  renderResearchCharts(report.charts || {});
}

function researchSvgText(text, x, y, options = {}) {
  const anchor = options.anchor || "start";
  const size = options.size || 16;
  const weight = options.weight || 400;
  const fill = options.fill || RESEARCH_PALETTE.ink;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="Arial, sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}">${escapeHtml(text)}</text>`;
}

function researchSvgVerticalText(text, x, y, options = {}) {
  const size = options.size || 14;
  const weight = options.weight || 700;
  const fill = options.fill || RESEARCH_PALETTE.ink;
  return `<text x="${x}" y="${y}" text-anchor="middle" transform="rotate(-90 ${x} ${y})" font-family="Arial, sans-serif" font-size="${size}" font-weight="${weight}" fill="${fill}">${escapeHtml(text)}</text>`;
}

function researchEmptySvg(message) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 360" role="img" aria-label="${escapeHtml(message)}"><rect width="800" height="360" fill="#ffffff"/>${researchSvgText(message, 400, 185, { anchor: "middle", size: 18, fill: RESEARCH_PALETTE.grey })}</svg>`;
}

function researchLineSvg(series, { yLabel = "Value", xLabel = "Stage", percent = false } = {}) {
  const width = 800; const height = 360; const left = 76; const right = 24; const top = 28; const bottom = 64;
  const all = series.flatMap((item) => item.values || []).filter(Number.isFinite);
  if (!all.length) return researchEmptySvg("Not testable with the current uploaded data");
  let minimum = Math.min(...all); let maximum = Math.max(...all);
  if (percent) { minimum = Math.min(0, minimum); maximum = Math.max(1, maximum); }
  if (Math.abs(maximum - minimum) < 1e-12) { minimum -= .5; maximum += .5; }
  const longest = Math.max(...series.map((item) => item.values.length));
  const x = (index) => left + (longest <= 1 ? 0 : index / (longest - 1)) * (width - left - right);
  const y = (value) => top + (maximum - value) / (maximum - minimum) * (height - top - bottom);
  const colors = [RESEARCH_PALETTE.blue, RESEARCH_PALETTE.orange, RESEARCH_PALETTE.teal, RESEARCH_PALETTE.purple];
  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = maximum - index * (maximum - minimum) / 4;
    const yy = y(value);
    const label = percent ? `${Math.round(value * 100)}%` : Math.abs(value) < .01 && value !== 0 ? value.toExponential(1) : value.toFixed(2).replace(/\.00$/, "");
    return `<line x1="${left}" x2="${width - right}" y1="${yy}" y2="${yy}" stroke="${RESEARCH_PALETTE.grid}" stroke-width="1"/>${researchSvgText(label, left - 12, yy + 5, { anchor: "end", size: 13, fill: RESEARCH_PALETTE.grey })}`;
  }).join("");
  const lines = series.map((item, seriesIndex) => {
    const points = item.values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
    const dash = item.dashed ? ' stroke-dasharray="12 9"' : "";
    return `<polyline points="${points}" fill="none" stroke="${colors[seriesIndex % colors.length]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"${dash}/>`;
  }).join("");
  const legend = series.map((item, index) => `<g transform="translate(${left + index * 180},326)"><line x1="0" x2="26" y1="0" y2="0" stroke="${colors[index % colors.length]}" stroke-width="4"${item.dashed ? ' stroke-dasharray="8 6"' : ""}/>${researchSvgText(item.label, 34, 5, { size: 13 })}</g>`).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(yLabel)}"><rect width="${width}" height="${height}" fill="#ffffff"/>${grid}<line x1="${left}" x2="${left}" y1="${top}" y2="${height - bottom}" stroke="#526064"/><line x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}" stroke="#526064"/>${lines}${researchSvgVerticalText(yLabel, 20, 162)}${researchSvgText(xLabel, 430, 315, { anchor: "middle", size: 14, weight: 700 })}${legend}</svg>`;
}

function researchBarSvg(items, yLabel) {
  if (!items?.length) return researchEmptySvg("No baseline comparison is available");
  const width = 800; const height = 360; const left = 78; const right = 28; const top = 28; const bottom = 76;
  const maximum = Math.max(...items.map((item) => Number(item.value) || 0), 1e-12) * 1.16;
  const plotWidth = width - left - right; const plotHeight = height - top - bottom;
  const band = plotWidth / items.length; const barWidth = Math.min(105, band * .62);
  const colors = [RESEARCH_PALETTE.teal, RESEARCH_PALETTE.blue, RESEARCH_PALETTE.orange, RESEARCH_PALETTE.purple];
  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = maximum * (4 - index) / 4; const yy = top + index * plotHeight / 4;
    return `<line x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}" stroke="${RESEARCH_PALETTE.grid}"/>${researchSvgText(value.toFixed(3), left-10, yy+5, {anchor:"end",size:12,fill:RESEARCH_PALETTE.grey})}`;
  }).join("");
  const bars = items.map((item, index) => {
    const value = Number(item.value) || 0; const barHeight = value / maximum * plotHeight; const xx = left + index * band + (band - barWidth) / 2; const yy = top + plotHeight - barHeight;
    return `<rect x="${xx}" y="${yy}" width="${barWidth}" height="${barHeight}" rx="5" fill="${colors[index % colors.length]}"/>${researchSvgText(value.toFixed(4), xx+barWidth/2, Math.max(18,yy-8), {anchor:"middle",size:13,weight:700})}${researchSvgText(item.label, xx+barWidth/2, height-bottom+24, {anchor:"middle",size:13,weight:700})}`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(yLabel)}"><rect width="${width}" height="${height}" fill="#ffffff"/>${grid}<line x1="${left}" x2="${left}" y1="${top}" y2="${height-bottom}" stroke="#526064"/><line x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}" stroke="#526064"/>${bars}${researchSvgVerticalText(yLabel, 20, 162)}</svg>`;
}

function renderResearchCharts(charts) {
  const timelinePoints = charts.timeline?.points || [];
  const timelineMetric = timelinePoints.some((point) => point.mean_speed !== null)
    ? { key: "mean_speed", label: "Mean speed (km/h)" }
    : timelinePoints.some((point) => point.mean_travel_time !== null)
      ? { key: "mean_travel_time", label: "Mean travel time (min)" }
      : { key: "mean_primary", label: "Mean uploaded measure" };
  const timelineSvg = researchLineSvg([{ label: timelineMetric.label, values: timelinePoints.map((point) => Number(point[timelineMetric.key])).filter(Number.isFinite) }], { yLabel: timelineMetric.label, xLabel: "Uploaded time" });
  const criticalSvg = researchBarSvg(charts.critical_baselines || [], "Mean efficiency loss");
  const recoverySvg = researchBarSvg(charts.recovery_baselines || [], "Recovery-curve AUC");
  const stressSvg = researchLineSvg([
    { label: "Registered TWM", values: charts.stress?.platform || [] },
    { label: "Independent oracle", values: charts.stress?.independent || [], dashed: true },
  ], { yLabel: "Efficiency retained", xLabel: "Cumulative road removals", percent: true });
  const figures = { timeline: timelineSvg, critical: criticalSvg, recovery: recoverySvg, stress: stressSvg };
  Object.entries(figures).forEach(([key, svg]) => {
    state.researchFigureSvgs.set(key, svg);
    const target = $(`#research-chart-${key}`);
    if (target) target.innerHTML = svg;
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function downloadResearchFigure(key) {
  const svg = state.researchFigureSvgs.get(key);
  if (!svg || !state.researchReport) return;
  downloadBlob(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }), `${state.researchReport.report_id}-${key}.svg`);
}

async function researchSvgImage(svg) {
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
    return image;
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(url), 500);
  }
}

async function exportResearchFiguresPng() {
  if (!state.researchReport || state.researchFigureSvgs.size < 4) return;
  const keys = ["timeline", "critical", "recovery", "stress"];
  try {
    const images = await Promise.all(keys.map((key) => researchSvgImage(state.researchFigureSvgs.get(key))));
    const canvas = document.createElement("canvas");
    canvas.width = 3840; canvas.height = 2300;
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = RESEARCH_PALETTE.ink; context.font = "700 66px Arial";
    context.fillText("Transport World Model research validation", 120, 105);
    context.font = "34px Arial"; context.fillStyle = RESEARCH_PALETTE.grey;
    context.fillText(`${state.researchReport.project_name} · ${state.researchReport.report_id}`, 120, 160);
    const positions = [[80,230],[1940,230],[80,1240],[1940,1240]];
    images.forEach((image, index) => context.drawImage(image, positions[index][0], positions[index][1], 1820, 820));
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png", 1));
    if (blob) downloadBlob(blob, `${state.researchReport.report_id}-paper-figures.png`);
  } catch (error) {
    toast("PNG export failed", error.message, true);
  }
}

function printResearchReport() {
  if (!state.researchReport) return;
  document.body.classList.add("research-printing");
  window.setTimeout(() => window.print(), 80);
  window.setTimeout(() => document.body.classList.remove("research-printing"), 800);
}

function renderModels(models) {
  const recommendations = new Map((state.modelRecommendations?.models || []).map((item) => [item.model_id, item]));
  const detectedFields = Object.entries(state.project?.summary?.metrics || {}).filter(([, value]) => value?.available).map(([key]) => key.replaceAll("_", " "));
  $("#model-grid").innerHTML = models.map((model) => {
    const card = model.card || {};
    const recommendation = recommendations.get(model.id);
    const requirements = card.required_data || card.execution_tiers?.[0]?.required_data || ["Confirmed uploaded fields"];
    const limitations = card.known_limitations || [card.status_reason || "See model card"];
    const primarySource = model.literature_sources?.[0] || card.primary_sources?.[0] || {};
    const source = primarySource.citation || "Transport world model evidence ledger";
    const evidenceStatus = String(primarySource.evidence_status || "");
    const evidenceLabel = /preprint/i.test(evidenceStatus) ? `Preprint · ${source}` : source;
    const tiers = model.execution_tiers || card.execution_tiers?.map((tier) => tier.name) || [card.performance || model.family?.replaceAll("_", " ") || "Data analysis"];
    const status = recommendation?.status || (state.project ? "conditional" : "adaptive");
    const statusLabel = recommendation
      ? { ready: "Full model ready", conditional: "Ready with available data", needs_data: "More data needed" }[status]
      : "Adaptive family";
    const selectedTier = recommendation?.selected_tier || tiers[0];
    const highTier = recommendation?.high_fidelity_tier || tiers.at(-1);
    const missing = recommendation?.missing_for_high_fidelity || [];
    const scenarioFamily = ["P5-M7", "P5-M17", "P5-M21", "P5-M22"].includes(model.id);
    return `<article class="model-card">
      <div class="model-card-head"><span class="model-id" title="Public model method ID">${escapeHtml(publicModelId(model.id))}</span><span class="model-status ${escapeHtml(status)}">${escapeHtml(statusLabel)}</span></div>
      <h3>${escapeHtml(model.name)}</h3>
      <p>${escapeHtml(model.goal || card.purpose || model.family?.replaceAll("_", " ") || "Versioned disaster transport model")}</p>
      <div class="model-tier-track"><span class="is-selected"><small>Current runnable method</small><b>${escapeHtml(selectedTier)}</b></span><i aria-hidden="true">→</i><span><small>Optional upgrade with additional data</small><b>${escapeHtml(highTier)}</b></span></div>
      ${missing.length ? `<p class="model-unlock">Add ${escapeHtml(missing.join(" + "))} to unlock the richer tier.</p>` : ""}
      <dl><dt>Detected</dt><dd>${escapeHtml(detectedFields.join(" · ") || "Open a project")}</dd><dt>Minimum</dt><dd>${escapeHtml(requirements.slice(0, 3).join(" · "))}</dd><dt>Selected tier</dt><dd>${escapeHtml(selectedTier)}</dd><dt>Evidence</dt><dd>${escapeHtml(evidenceLabel)}</dd></dl>
      <details><summary>Assumptions and limits</summary><pre>${escapeHtml(limitations.join("\n"))}</pre></details>
      <button class="secondary-button ripple model-use" type="button" data-model-use="${escapeHtml(model.id)}" data-model-goal="${escapeHtml(model.goal || card.purpose || model.name)}"${!state.project || recommendation?.executable === false ? " disabled" : ""}>${scenarioFamily ? "Build a scenario" : "Use this model"}</button>
    </article>`;
  }).join("");
}

function useModelCard(modelId, goal) {
  if (!state.project) return;
  if (["P5-M4", "P5-M8", "P5-M20"].includes(modelId)) {
    switchView("explore");
    openRecoveryBuilder();
    toast("Recovery model ready", "Choose disrupted roads or areas, the number of priorities and the ranking objective.");
    return;
  }
  if (modelId === "P5-M23") {
    switchView("explore");
    openRouteBuilder();
    toast("Route model ready", "Choose two routable endpoints, then add a temporary disruption or use a saved scenario.");
    return;
  }
  if (modelId === "P5-M11") {
    switchView("explore");
    openStressBuilder();
    toast("Network resilience model ready", "Choose any positive number of cumulative road-failure stages; the source order is shown before the run.");
    return;
  }
  if (["P5-M7", "P5-M17", "P5-M21", "P5-M22"].includes(modelId)) {
    switchView("explore");
    openScenarioBuilder();
    toast("Scenario model ready", "Select the initial roads, choose Closed, Slower or Longer time, then run the propagated network model.");
    return;
  }
  const prompts = {
    "P5-M0": "Summarise the selected area.",
    "P5-M1": "Find the 20 most persistently degraded roads.",
    "P5-M2": "Find the 20 most critical roads.",
    "P5-M3": "Assume the selected roads are closed and analyse the topology effect.",
    "P5-M5": "Run a top 20 importance and exposure sensitivity screen.",
    "P5-M6": "Run a top 20 network accessibility screen.",
    "P5-M9": "Show the observed resilience curve.",
    "P5-M10": "Identify the 20 critical geographic areas.",
    "P5-M12": "Run a top 20 reliability sensitivity envelope.",
    "P5-M13": "Run a top 20 cascading failure stress test.",
    "P5-M14": "Identify the top 20 hazard-transport exposure hotspots.",
    "P5-M15": "Run a top 20 equity accessibility distribution screen.",
    "P5-M16": "Run a top 20 protection priority screen.",
    "P5-M18": "Run a top 20 evacuation readiness reachability screen.",
    "P5-M19": "Run a top 20 chronological holdout validation.",
  };
  switchView("explore");
  state.preferredModelId = modelId;
  $("#command-input").value = prompts[modelId] || goal || "Recommend an executable analysis for this project.";
  invalidateAnalysisReview("A model family was selected.");
  interpretCurrentCommand();
}

async function refreshRuns() {
  try {
    const runs = await api.runs(state.project?.project_id || null);
    const body = $("#saved-runs-body");
    $("#saved-empty").classList.toggle("is-hidden", runs.length > 0);
    body.innerHTML = runs.map((record) => {
      const result = record.result || {};
      return `<tr><td>${escapeHtml(result.model?.name || record.analysis_type)}</td><td>${escapeHtml(result.status || "saved")}</td><td>${escapeHtml(result.project_name || record.project_id)}</td><td>${escapeHtml(new Date(record.created_at).toLocaleString())}</td><td><button class="table-action ripple" data-open-run="${escapeHtml(record.run_id)}" type="button">Open</button> <a class="table-action" href="/api/runs/${encodeURIComponent(record.run_id)}/export.json">Export</a></td></tr>`;
    }).join("");
  } catch (error) {
    toast("Saved analyses unavailable", error.message, true);
  }
}

function renderProjectMemory(memory) {
  const overview = memory?.overview;
  if (!overview) {
    $("#memory-overview").innerHTML = '<div class="empty-state">Open a project to see its memory.</div>';
    $("#memory-timeline").innerHTML = "";
    return;
  }
  const stats = [
    ["Source files", overview.source_file_count, "Confirmed mappings and SHA-256 provenance"],
    ["Analyses", overview.analysis_count, "Model, parameters, evidence and returned results"],
    ["Scenarios", overview.scenario_count, "Initial changes and indirect network effects"],
    ["Traceable events", overview.event_count, "API keys and transient chat are excluded"],
  ];
  $("#memory-overview").innerHTML = stats.map(([label, value, detail]) => `<div class="memory-stat"><span>${escapeHtml(label)}</span><strong>${formatNumber(value)}</strong><small>${escapeHtml(detail)}</small></div>`).join("");
  const events = [...(memory.events || [])].reverse();
  $("#memory-timeline").innerHTML = events.length ? events.map((event) => {
    const facts = Object.entries(event.facts || {}).filter(([, value]) => value !== null && value !== undefined && value !== "");
    const deletable = ["analysis", "scenario"].includes(event.event_type);
    return `<article class="memory-event ${escapeHtml(event.event_type)}"><header><h3>${escapeHtml(event.title)}</h3><time datetime="${escapeHtml(event.occurred_at)}">${escapeHtml(new Date(event.occurred_at).toLocaleString())}</time></header><p>${escapeHtml(event.detail)}</p><dl>${facts.map(([key, value]) => `<div><dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl><details><summary>Evidence and provenance</summary><pre>${escapeHtml(JSON.stringify(event.provenance || {}, null, 2))}</pre></details>${deletable ? `<button class="text-button danger-text memory-delete" type="button" data-memory-delete="${escapeHtml(event.event_id)}" data-memory-type="${escapeHtml(event.event_type)}">Delete saved record</button>` : ""}</article>`;
  }).join("") : '<div class="empty-state">No project events have been recorded yet.</div>';
}

async function refreshProjectMemory() {
  if (!state.project) return renderProjectMemory(null);
  try {
    state.projectMemory = await api.projectMemory(state.project.project_id);
    renderProjectMemory(state.projectMemory);
  } catch (error) {
    toast("Project memory unavailable", error.message, true);
  }
}

async function exportProjectMemory() {
  if (!state.project) return toast("Open a project first", "Project memory belongs to one uploaded data project.", true);
  if (!state.projectMemory) await refreshProjectMemory();
  if (!state.projectMemory) return;
  const blob = new Blob([JSON.stringify(state.projectMemory, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.project.project_id}-memory.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 500);
}

async function deleteMemoryEvent(eventId, eventType) {
  if (!window.confirm("Delete this saved record from project memory? The uploaded source project is not changed.")) return;
  try {
    if (eventType === "analysis") await api.deleteRun(eventId);
    else if (eventType === "scenario") await api.deleteScenario(eventId);
    else return;
    if (state.activeScenarioId === eventId) state.activeScenarioId = null;
    await Promise.all([refreshProjectMemory(), refreshRuns(), refreshScenarios()]);
    toast("Saved record deleted", "The source data project and its confirmed mapping were not changed.");
  } catch (error) {
    toast("Record could not be deleted", error.message, true);
  }
}

async function openSavedRun(runId) {
  setLoading(true, "Opening the saved analysis", "Restoring the verified result and map evidence.");
  try {
    const record = await api.run(runId);
    if (!state.project || state.project.project_id !== record.project_id) await loadProject({ project_id: record.project_id });
    switchView("explore");
    await ensureLinkAnalysisView(record.result?.analysis_type);
    renderResult(promoteMappedLinkResults(record.result));
  } catch (error) {
    toast("Analysis could not be opened", error.message, true);
  } finally {
    setLoading(false);
  }
}

function currentUploadMapping(fileIndex, file) {
  const selects = $$(`[data-file-index="${fileIndex}"][data-mapping-role]`);
  if (!selects.length) return { ...(file?.suggested_mapping?.roles || {}) };
  const mapping = {};
  selects.forEach((select) => { mapping[select.dataset.mappingRole] = select.value || null; });
  return mapping;
}

function collectUploadIssues() {
  if (state.uploadPreviewError) {
    return [{
      file: "Data check",
      severity: "blocker",
      title: "The data could not be loaded",
      detail: state.uploadPreviewError,
      action: "Save the table as CSV or TSV with one header row, then try again. Keep quotes around geometry values that contain commas.",
    }];
  }
  const preview = state.uploadPreview;
  if (!preview?.files?.length) return [];
  const adaptiveReady = Boolean(preview.transport_package?.auto_importable);
  const issues = [];
  const add = (file, severity, title, detail, action) => issues.push({ file, severity, title, detail, action });
  preview.files.forEach((file, fileIndex) => {
    const mapping = currentUploadMapping(fileIndex, file);
    const health = file.mapping_health || {};
    const coordinates = file.coordinate_diagnostics || {};
    const transport = file.transport_structure || {};
    const normalization = transport.normalization || {};
    if (!adaptiveReady && transport.structure_type !== "flat_observations") {
      (normalization.fatal_missing || []).forEach((message) => add(
        file.name, "blocker", "Key network data is missing", message,
        "Add the named location, geometry or connectivity fields; exact column names are not required.",
      ));
    }
    (normalization.warnings || []).forEach((message) => add(
      file.name, "check", "Automatic conversion note", message,
      "The file can still be imported when no blocking issue is listed; review the disclosed map type and units.",
    ));
    if (!adaptiveReady) {
      if (!mapping.entity_id) add(
        file.name, "blocker", "No stable asset ID detected",
        "Every stored point or road needs an identifier.",
        "Map a column such as link_id, road_id, detector_id, node_id, FID or OBJECTID.",
      );
      if (!mapping.latitude || !mapping.longitude) add(
        file.name, "blocker", "No usable location or network geometry",
        "A map needs coordinates, line geometry or connectivity.",
        "Provide longitude + latitude, a WKT/GeoJSON line, four endpoint coordinates, or Link ID + From + To.",
      );
      if (!["speed", "travel_time", "primary_metric", "flow", "capacity", "free_flow_time", "length", "repair_cost", "repair_duration", "failure_probability"].some((role) => mapping[role])) add(
        file.name, "check", "No measured value detected",
        "The location may be drawable, but condition and resilience analyses need a value.",
        "Add or map speed, travel time, flow, capacity, length or one hazard/condition measure.",
      );
    }
    const coordinateAdvice = {
      review_swap: ["Check latitude and longitude", "The sampled ranges suggest the two fields may be reversed.", "Confirm latitude is between −90 and 90 and longitude between −180 and 180."],
      projected_coordinates: ["Projected coordinates need context", "The values are not ordinary longitude/latitude.", "Use WGS84, Web Mercator endpoint fields, exact line geometry, or From/To topology. An unknown local CRS cannot be placed on a real map safely."],
      invalid_coordinates: ["Coordinates are outside valid ranges", "Some sampled locations cannot be placed on the map.", "Correct non-numeric or out-of-range values; do not put latitude and longitude in one field."],
      no_numeric_pairs: ["Coordinate fields contain no numeric pairs", "The selected fields do not contain usable map positions.", "Choose numeric coordinate columns or provide road geometry/topology instead."],
      partial_coordinates: ["Some rows have incomplete coordinates", "Rows missing either latitude or longitude cannot be mapped as points.", "Complete both values, or provide Link geometry/From–To connectivity for those roads."],
      zero_island_warning: ["Check 0, 0 locations", "Zero coordinates usually mean missing data and would appear in the Atlantic Ocean.", "Replace placeholder zeros with real positions or leave them blank."],
    };
    if (!adaptiveReady && coordinateAdvice[coordinates.status]) {
      const [title, detail, action] = coordinateAdvice[coordinates.status];
      add(file.name, coordinates.status === "review_swap" || coordinates.status === "partial_coordinates" ? "check" : "blocker", title, detail, action);
    }
    const mismatches = health.type_mismatch_roles || [];
    if (!adaptiveReady && mismatches.length) add(
      file.name, "blocker", "Field values do not match their labels",
      `Check: ${mismatches.join(", ")}.`,
      "Use numbers for coordinates and measures, and an ISO-style date/time or a numeric time step for time.",
    );
    const uncertain = (health.low_confidence_roles || []).filter((role) => mapping[role]);
    if (!adaptiveReady && uncertain.length) add(
      file.name, "check", "Confirm uncertain field matches",
      `Review: ${uncertain.join(", ")}.`,
      "Open this file's field card and select the intended column. The platform will not guess between ambiguous fields.",
    );
  });
  return issues.filter((issue, index, all) => all.findIndex((candidate) =>
    candidate.file === issue.file && candidate.title === issue.title
  ) === index);
}

function renderDataGuideDiagnosis() {
  const summary = $("#data-diagnosis-summary");
  const list = $("#data-issue-list");
  const badge = $("#data-issue-count");
  if (!summary || !list || !badge) return;
  const issues = collectUploadIssues();
  const blockers = issues.filter((issue) => issue.severity === "blocker").length;
  badge.textContent = String(issues.length);
  summary.classList.toggle("has-issues", issues.length > 0 || !state.uploadPreview);
  if (!state.uploadPreview && !state.uploadPreviewError) {
    summary.innerHTML = "<div><strong>No file checked yet</strong><small>Choose data and read the preview. File-specific fixes will appear here.</small></div>";
    list.innerHTML = '<div class="data-empty-diagnosis"><strong>Your diagnosis will be specific</strong><small>It will name the affected file, missing field, accepted alternative and exact next action.</small></div>';
    return;
  }
  if (!issues.length) {
    const structure = state.uploadPreview?.transport_package;
    const detected = structure?.preferred_view === "links" ? "road-link network" : "geographic point data";
    const transformations = (state.uploadPreview?.files || []).flatMap((file) =>
      (file.transport_structure?.normalization?.steps || []).map((step) => ({ file: file.name, step }))
    );
    summary.innerHTML = `<div><strong>Ready for automatic conversion</strong><small>The preview recognised ${escapeHtml(detected)}. Exact template column names are not required.</small></div>`;
    list.innerHTML = transformations.length
      ? `<div class="data-conversion-list">${transformations.map((item) => `<article class="data-conversion-step"><div><strong>${escapeHtml(item.file)}</strong><small>${escapeHtml(item.step)}</small></div></article>`).join("")}</div>`
      : '<div class="data-empty-diagnosis"><strong>No blocking format issue found</strong><small>The confirmation screen remains the final check before import.</small></div>';
    return;
  }
  summary.innerHTML = `<div><strong>${blockers ? `${blockers} blocking issue${blockers === 1 ? "" : "s"}` : "Review recommended"}</strong><small>${issues.length} clear action${issues.length === 1 ? "" : "s"} for the current upload.</small></div>`;
  list.innerHTML = issues.map((issue) => `<article class="data-issue-card ${issue.severity === "blocker" ? "is-blocker" : ""}"><span>${escapeHtml(issue.file)}</span><div><strong>${escapeHtml(issue.title)}</strong><p>${escapeHtml(issue.detail)}</p></div><footer>${escapeHtml(issue.action)}</footer></article>`).join("");
}

function setDataGuideTab(tab) {
  state.dataGuideTab = ["fields", "diagnosis"].includes(tab) ? tab : "formats";
  $$('[data-data-guide-tab]').forEach((button) => button.classList.toggle("is-active", button.dataset.dataGuideTab === state.dataGuideTab));
  $$('[data-data-guide-view]').forEach((view) => view.classList.toggle("is-hidden", view.dataset.dataGuideView !== state.dataGuideTab));
  if (state.dataGuideTab === "diagnosis") renderDataGuideDiagnosis();
}

function openDataFormatGuide(tab = "formats") {
  renderDataGuideDiagnosis();
  setDataGuideTab(tab);
  const dialog = $("#data-format-dialog");
  if (!dialog.open) dialog.showModal();
}

function openUpload() {
  const dialog = $("#upload-dialog");
  state.uploadPreview = null;
  state.uploadId = null;
  state.uploadPreviewError = null;
  $("#upload-files").value = "";
  $("#chosen-files").innerHTML = "";
  $("#upload-project-name").value = "My disaster transport project";
  $("#upload-hazard").value = "other";
  $("#upload-speed-unit").value = "km/h";
  $("#upload-travel-unit").value = "min";
  $("#upload-primary-unit").value = "uploaded unit";
  $("#upload-timezone").value = "unspecified";
  $("#upload-step-seconds").value = "300";
  $("#upload-entity-scope").value = "per_region";
  $("#upload-node-scope").value = "per_region";
  $("#upload-primary-direction").value = "descriptive";
  $("#upload-primary-comparison").value = "current_value";
  $("#upload-primary-scale").value = "1";
  showUploadStep(1);
  if (!dialog.open) dialog.showModal();
}

function showUploadStep(step) {
  $$("[data-upload-panel]").forEach((panel) => {
    const active = Number(panel.dataset.uploadPanel) === step;
    panel.classList.toggle("is-active", active);
    if (active) panel.scrollTop = 0;
  });
  $$("[data-upload-step]").forEach((label) => label.classList.toggle("is-active", Number(label.dataset.uploadStep) === step));
}

function renderChosenFiles() {
  const files = [...$("#upload-files").files];
  state.uploadPreviewError = null;
  $("#chosen-files").innerHTML = files.map((file) => `<span class="chosen-file">${escapeHtml(file.name)} · ${formatBytes(file.size)}</span>`).join("");
  renderDataGuideDiagnosis();
}

async function previewUpload() {
  const files = [...$("#upload-files").files];
  if (!files.length) {
    toast("Choose one or more files", "One CSV is enough. CSV, TSV and delimited text files are supported.", true);
    return;
  }
  const button = $("#preview-upload");
  button.disabled = true;
  button.textContent = "Reading local preview…";
  try {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    state.uploadPreview = await api.uploadPreview(formData);
    state.uploadId = state.uploadPreview.upload_id;
    state.uploadPreviewError = null;
    renderMappingStep();
    showUploadStep(2);
  } catch (error) {
    state.uploadPreview = null;
    state.uploadId = null;
    state.uploadPreviewError = error.message;
    toast("Preview failed", error.message, true);
    openDataFormatGuide("diagnosis");
  } finally {
    button.disabled = false;
    button.textContent = "Read a safe preview";
  }
}

function renderMappingStep() {
  const preview = state.uploadPreview;
  const first = preview.files[0];
  const packagePlan = preview.transport_package || {};
  const adaptiveReady = Boolean(packagePlan.auto_importable);
  const geometryChip = $(".geometry-chip");
  const nonGeographic = preview.files.some((file) =>
    ["normalised_uploaded_geometry", "topology_layout"].includes(file.transport_structure?.coordinate_mode)
    || file.transport_structure?.geography_mode === "topology_layout"
  );
  if (geometryChip) geometryChip.textContent = packagePlan.preferred_view === "links"
    ? (nonGeographic ? "Topology network detected" : "Road-link network detected")
    : (nonGeographic ? "Local-coordinate points detected" : "Geographic points detected");
  $("#mapping-file-summary").textContent = `${preview.files.length} file${preview.files.length === 1 ? "" : "s"} · ${formatBytes(preview.total_size_bytes)} · ${first.columns.length} fields`;
  $("#file-audit-list").innerHTML = preview.files.map((file, fileIndex) => {
    const health = file.mapping_health || { status: "review", missing_required: [] };
    const coordinates = file.coordinate_diagnostics || { message: "Coordinate check unavailable." };
    const matched = Object.values(file.suggested_mapping?.roles || {}).filter(Boolean).length;
    const label = health.status === "ready" ? (file.transport_structure?.auto_importable ? "Auto-ready" : "Ready") : health.status === "blocked" ? "Needs correction" : "Review";
    const missing = health.missing_required?.length ? ` · missing ${health.missing_required.join(", ")}` : "";
    const review = (health.review_reasons || []).slice(0, 2).join(" · ");
    const conversionSteps = file.transport_structure?.normalization?.steps || [];
    const structureMessage = adaptiveReady
      ? (conversionSteps.slice(0, 2).join(" ") || file.transport_structure?.explanation || packagePlan.explanation)
      : coordinates.message;
    return `<article class="file-audit ${escapeHtml(health.status)}" data-file-audit="${fileIndex}"><span class="file-audit-state">${escapeHtml(label)}</span><div><strong>${escapeHtml(file.name)}</strong><small>${matched} semantic fields matched${escapeHtml(missing)}</small><p>${escapeHtml(structureMessage)}${review && !adaptiveReady ? ` ${escapeHtml(review)}.` : ""}</p></div></article>`;
  }).join("");
  $("#mapping-grid").innerHTML = preview.files.map((file, fileIndex) => {
    const suggestions = file.suggested_mapping || {};
    const roles = { ...(suggestions.roles || {}) };
    if (!roles.speed && !roles.travel_time && !roles.primary_metric) {
      const fallbackRole = ["flow", "capacity", "free_flow_time", "repair_cost", "repair_duration", "failure_probability", "length"]
        .find((role) => roles[role]);
      if (fallbackRole) roles.primary_metric = roles[fallbackRole];
    }
    const fields = MAPPING_ROLES.map(([role, label, required]) => {
      const selected = roles[role];
      const roleRequired = required && !file.transport_structure?.auto_importable;
      const confidence = suggestions.confidence?.[role] || (role === "primary_metric" && selected ? "please_check" : "not_recognised");
      const score = Number(suggestions.scores?.[role] || 0);
      const confidenceLabel = confidence === "confident"
        ? `✓ matched ${Math.round(score * 100)}%`
        : confidence === "please_check" ? "check match" : "not found";
      const options = ["", ...file.columns].map((column) => `<option value="${escapeHtml(column)}"${column === selected ? " selected" : ""}>${column ? escapeHtml(column) : "Not mapped"}</option>`).join("");
      return `<div class="mapping-field"><label><span>${escapeHtml(label)}${roleRequired ? " *" : ""}<em class="${escapeHtml(confidence.replaceAll("_", "-"))}">${escapeHtml(confidenceLabel)}</em></span><select data-file-index="${fileIndex}" data-mapping-role="${escapeHtml(role)}">${options}</select></label></div>`;
    }).join("");
    const ready = adaptiveReady || file.mapping_health?.status === "ready";
    const instruction = ready ? "Ready · click to review or change" : file.mapping_health?.status === "blocked" ? "Correction required · open now" : "Review required · open now";
    return `<details class="file-mapping-card${ready ? "" : " needs-review"}" data-file-card="${fileIndex}"${ready ? "" : " open"}>
      <summary><span>File ${fileIndex + 1}</span><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(instruction)}</small><i class="mapping-card-toggle" aria-hidden="true"></i></summary>
      <div class="mapping-file-options"><label><span>Text encoding</span><select data-file-encoding="${fileIndex}"><option value="">Auto detected · ${escapeHtml(file.encoding)}</option><option value="utf-8-sig">UTF-8 with/without BOM</option><option value="utf-16">UTF-16</option><option value="gb18030">GB18030 · Simplified Chinese</option><option value="big5">Big5 · Traditional Chinese</option><option value="cp1252">Windows CP1252</option><option value="latin-1">Latin-1</option></select></label><small>If English headers look correct but Chinese place names do not, choose the source encoding here.</small></div>
      <div class="mapping-grid">${fields}</div>
    </details>`;
  }).join("");
  const speedNames = preview.files.map((file) => String(file.suggested_mapping?.roles?.speed || "").toLowerCase()).filter(Boolean);
  const travelNames = preview.files.map((file) => String(file.suggested_mapping?.roles?.travel_time || "").toLowerCase()).filter(Boolean);
  $("#upload-speed-unit").value = speedNames.length && speedNames.every((name) => name.includes("mph"))
    ? "mph"
    : speedNames.length && speedNames.every((name) => name.includes("m/s") || name.includes("mps"))
      ? "m/s"
      : speedNames.length && speedNames.every((name) => name.includes("kmh") || name.includes("kph") || name.includes("km/h"))
        ? "km/h" : "km/h";
  $("#upload-travel-unit").value = travelNames.length && travelNames.every((name) => name.includes("sec"))
    ? "s"
    : travelNames.length && travelNames.every((name) => name.includes("hour") || name.includes("_hr"))
      ? "h"
      : travelNames.length && travelNames.every((name) => name.includes("min"))
        ? "min" : "min";
  $("#preview-table").innerHTML = preview.files.map((file, fileIndex) => {
    const columns = file.columns;
    return `<section class="sample-file"><strong>File ${fileIndex + 1}: ${escapeHtml(file.name)}</strong><table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${file.rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("")}</tbody></table></section>`;
  }).join("");
  validateMappingStep();
}

function validateMappingStep() {
  const files = state.uploadPreview?.files || [];
  const adaptiveReady = Boolean(state.uploadPreview?.transport_package?.auto_importable);
  let readyCount = 0;
  files.forEach((file, fileIndex) => {
    const mapped = {};
    $$(`[data-file-index="${fileIndex}"][data-mapping-role]`).forEach((select) => { mapped[select.dataset.mappingRole] = select.value || null; });
    const coordinateBlocked = ["projected_coordinates", "invalid_coordinates", "no_numeric_pairs", "not_mapped"]
      .includes(file.coordinate_diagnostics?.status);
    const ready = adaptiveReady || Boolean(mapped.entity_id && mapped.latitude && mapped.longitude
      && (mapped.speed || mapped.travel_time || mapped.primary_metric) && !coordinateBlocked);
    if (ready) readyCount += 1;
    const card = $(`[data-file-card="${fileIndex}"]`);
    card?.classList.toggle("needs-review", !ready);
    const message = card?.querySelector("summary small");
    if (message) message.textContent = ready ? "Ready · click to review or change" : "Required field missing · open and correct";
    const audit = $(`[data-file-audit="${fileIndex}"]`);
    if (audit) {
      audit.classList.remove("ready", "review", "blocked");
      audit.classList.add(ready ? "ready" : "blocked");
      const auditState = audit.querySelector(".file-audit-state");
      if (auditState) auditState.textContent = ready ? "Ready" : coordinateBlocked ? "Fix coordinates" : "Needs correction";
    }
  });
  const allReady = files.length > 0 && readyCount === files.length;
  const banner = $("#mapping-ready-banner");
  banner?.classList.toggle("has-review", !allReady);
  const title = banner?.querySelector("strong");
  const copy = banner?.querySelector("small");
  if (title) title.textContent = allReady
    ? (adaptiveReady ? "Automatic conversion is ready" : files.length === 1 ? "1 file is ready" : `All ${files.length} files are ready`)
    : `${readyCount} of ${files.length} files ready`;
  if (copy) copy.textContent = allReady
    ? (adaptiveReady
      ? "The platform will normalise recognised Nodes, Links, geometry, OD and scenario fields automatically. Open a file only to override an ambiguous match."
      : "Ready files are collapsed. Continue with the always-visible import bar, or open a file to change a match.")
    : "Open the highlighted file and confirm ID, latitude, longitude and one measured value.";
  $("#confirm-import-status").textContent = allReady ? "Ready to import" : "Complete required field matches";
  $("#confirm-upload").disabled = !allReady;
  const helpButton = $("#mapping-data-help");
  if (helpButton) helpButton.textContent = allReady ? "Format guide" : "Fix my data";
  renderDataGuideDiagnosis();
  return allReady;
}

async function confirmUpload() {
  if (!validateMappingStep()) {
    const card = $(".file-mapping-card.needs-review");
    if (card) { card.open = true; card.scrollIntoView({ behavior: "smooth", block: "center" }); }
    return toast("Complete the highlighted file", "Confirm ID, latitude, longitude and one measured value before import.", true);
  }
  const fileMappings = {};
  const fileEncodings = {};
  (state.uploadPreview?.files || []).forEach((file, fileIndex) => {
    const mapping = {};
    $$(`[data-file-index="${fileIndex}"][data-mapping-role]`).forEach((select) => {
      mapping[select.dataset.mappingRole] = select.value || null;
    });
    fileMappings[file.name] = mapping;
    const encoding = $(`[data-file-encoding="${fileIndex}"]`)?.value;
    if (encoding) fileEncodings[file.name] = encoding;
  });
  const firstFileName = state.uploadPreview?.files?.[0]?.name;
  const mapping = firstFileName ? fileMappings[firstFileName] : {};
  const adaptiveReady = Boolean(state.uploadPreview?.transport_package?.auto_importable);
  const invalidFile = Object.entries(fileMappings).find(([, candidate]) =>
    !candidate.entity_id || !candidate.latitude || !candidate.longitude
      || (!candidate.speed && !candidate.travel_time && !candidate.primary_metric));
  if (invalidFile && !adaptiveReady) {
    toast("Confirm every file", `${invalidFile[0]} still needs an ID, latitude, longitude and one measured value.`, true);
    return;
  }
  const projectName = $("#upload-project-name").value.trim();
  if (!projectName) {
    toast("Project name required", "Give this static dataset a clear project name.", true);
    return;
  }
  $("#import-title").textContent = adaptiveReady ? "Creating the transport-network project" : "Creating the geographic project";
  $("#import-progress").style.width = "0%";
  $("#import-percent").textContent = "0%";
  $("#import-message").textContent = "Starting the verified import…";
  showUploadStep(3);
  try {
    const job = await api.confirmUpload(state.uploadId, {
      project_name: projectName,
      hazard_type: $("#upload-hazard").value,
      mapping,
      file_mappings: fileMappings,
      file_encodings: fileEncodings,
      units: {
        speed: $("#upload-speed-unit").value,
        travel_time: $("#upload-travel-unit").value,
        primary_metric: $("#upload-primary-unit").value.trim() || "uploaded unit",
      },
      timezone: $("#upload-timezone").value.trim() || "unspecified",
      time_step_seconds: Number($("#upload-step-seconds").value || 300),
      topology_node_scope: $("#upload-node-scope").value,
      entity_id_scope: $("#upload-entity-scope").value,
      primary_metric_direction: $("#upload-primary-direction").value,
      primary_metric_comparison: $("#upload-primary-comparison").value,
      primary_metric_scale: Number($("#upload-primary-scale").value || 1),
    });
    await pollImportJob(job.job_id);
  } catch (error) {
    toast("Import could not start", error.message, true);
    state.uploadPreviewError = error.message;
    showUploadStep(2);
    openDataFormatGuide("diagnosis");
  }
}

async function pollImportJob(jobId) {
  while (true) {
    const job = await api.job(jobId);
    const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
    $("#import-progress").style.width = `${progress}%`;
    $("#import-percent").textContent = `${progress.toFixed(0)}%`;
    $("#import-message").textContent = job.message;
    if (job.status === "completed") {
      $("#import-title").textContent = "Project ready";
      const project = await api.getProject(job.result.project_id);
      window.setTimeout(() => $("#upload-dialog").close(), 450);
      await loadProject(project);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "The import failed.");
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
}

function clearSelection() {
  state.selectedKeys.clear();
  state.keptKeys.clear();
  state.focusedKey = null;
  state.bbox = null;
  state.analysisScopeMode = "whole";
  state.lastFeature = null;
  state.completedResults.clear();
  state.preferredModelId = null;
  if (state.routeBuilding) {
    state.routeOrigin = null;
    state.routeDestination = null;
    state.routePickRole = "origin";
    renderRouteBuilder();
  }
  invalidateAnalysisReview("The map selection was cleared.");
  $("#command-input").value = "";
  $$('[data-command]').forEach((button) => button.classList.remove("is-active"));
  if (state.activeScenarioId) activateScenario(null);
  if (state.stressBuilding) renderStressBuilder();
  state.map.setSelectedKeys([]);
  state.map.clearArea();
  $("#entity-inspector").classList.add("is-hidden");
  setFlowStage(state.project ? 2 : 1);
  renderSelectionSummary();
  updateReadiness();
  if (state.project) setMapBanner(
    state.mapView === "links" ? "Mapped link view" : "Historical data",
    `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`,
  );
}

async function changeProject() {
  try {
    state.projects = (await api.listProjects())
      .filter((project) => project.retention === "persistent")
      .sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")));
  } catch (error) {
    toast("Saved projects could not be refreshed", error.message, true);
    return;
  }
  const savedProjects = state.projects;
  if (!savedProjects.length) {
    toast("No saved projects", "Upload data, then choose Save project only if you want to reopen it later.");
    openUpload();
    return;
  }
  $("#project-list").innerHTML = savedProjects.map((project) => {
    const summary = project.summary || {};
    const mapped = formatNumber(summary.mapped_entity_count ?? summary.entity_count ?? 0);
    const time = summary.time_start && summary.time_end ? `${summary.time_start} to ${summary.time_end}` : "Static snapshot or unspecified time";
    const hazard = String(project.hazard_type || "other").replaceAll("_", " ");
    const projectId = escapeHtml(project.project_id);
    const projectName = escapeHtml(project.name);
    return `<article class="project-entry"><button class="project-option ripple" data-project-id="${projectId}" type="button"><strong>${projectName}</strong><span>${escapeHtml(hazard)}</span><small>${mapped} mapped entities · ${escapeHtml(time)}</small></button><button class="project-delete ripple" data-delete-project-id="${projectId}" type="button" aria-label="Delete ${projectName}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8v10m4-10v10m4-10v10M5 5h14M9 5V3h6v2m3 0-1 16H7L6 5"/></svg><span>Delete</span></button></article>`;
  }).join("");
  const dialog = $("#project-dialog");
  if (!dialog.open) dialog.showModal();
}

async function deleteAllSavedProjects() {
  const button = $("#delete-all-projects");
  const savedProjects = state.projects.filter((project) => project.retention === "persistent");
  if (!savedProjects.length) {
    toast("No saved projects", "There is no saved project history to delete.");
    return;
  }
  const confirmed = window.confirm(`Delete all ${savedProjects.length} saved projects permanently?\n\nThis removes their analysis results and scenarios from this computer. This cannot be undone.`);
  if (!confirmed) return;

  button.disabled = true;
  const originalLabel = button.textContent;
  button.textContent = "Deleting all…";
  const results = await Promise.allSettled(savedProjects.map((project) => api.deleteProject(project.project_id)));
  const failedIds = new Set(results.flatMap((result, index) => result.status === "rejected" ? [savedProjects[index].project_id] : []));
  const deletedCount = savedProjects.length - failedIds.size;
  state.projects = state.projects.filter((project) => failedIds.has(project.project_id));
  button.disabled = false;
  button.textContent = originalLabel;

  if (failedIds.size) {
    await changeProject();
    toast("Some projects could not be deleted", `${deletedCount} deleted; ${failedIds.size} remained.`, true);
    return;
  }

  $("#project-list").innerHTML = '<div class="project-list-empty"><strong>No saved projects</strong><span>Upload new data and choose Save project when you want to keep it.</span></div>';
  toast("All saved projects deleted", `${deletedCount} saved projects were permanently removed from this computer.`);
  if (savedProjects.some((project) => project.project_id === state.project?.project_id)) {
    $("#project-dialog").close();
    window.location.reload();
  }
}

function playbackTiming() {
  const profile = PLAYBACK_PROFILES[state.playbackSpeed] || PLAYBACK_PROFILES[4];
  const transitions = Math.max(1, state.timeValues.length - 1);
  const maximumFrames = Math.max(
    1,
    Math.min(profile.maximumFrames, Math.floor(profile.targetDurationMs / profile.minimumDelayMs)),
  );
  const step = Math.max(1, Math.ceil(transitions / maximumFrames));
  const visibleFrames = Math.max(1, Math.ceil(transitions / step));
  const delay = Math.max(
    profile.minimumDelayMs,
    Math.min(profile.maximumDelayMs, Math.round(profile.targetDurationMs / visibleFrames)),
  );
  return { step, delay };
}

async function advancePlayback() {
  if (!state.playing) return;
  const lastIndex = state.timeValues.length - 1;
  if (state.timeIndex >= lastIndex) {
    stopPlayback();
    return;
  }
  const timing = playbackTiming();
  state.timeIndex = Math.min(lastIndex, state.timeIndex + timing.step);
  $("#time-slider").value = String(state.timeIndex);
  $("#time-label").textContent = currentTimeLabel();
  invalidateAnalysisReview("Historical playback changed the observed time.");
  try {
    await refreshMap(null);
    await refreshOpenInspector();
  } catch (error) {
    toast("Playback stopped", error.message, true);
    stopPlayback();
    return;
  }
  if (state.timeIndex >= lastIndex) {
    stopPlayback();
  } else if (state.playing) {
    state.playTimer = window.setTimeout(advancePlayback, timing.delay);
  }
}

function stopPlayback() {
  state.playing = false;
  window.clearTimeout(state.playTimer);
  state.playTimer = null;
  const button = $("#play-time");
  if (button) {
    button.classList.remove("is-playing");
    button.textContent = "▶";
    button.setAttribute("aria-label", "Play historical time");
  }
}

async function togglePlayback() {
  if (!state.project || state.timeValues.length <= 1) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  state.playing = true;
  const button = $("#play-time");
  button.classList.add("is-playing");
  button.textContent = "Ⅱ";
  button.setAttribute("aria-label", "Pause historical time");
  invalidateAnalysisReview("Historical playback changed the observed time.");
  if (state.timeIndex >= state.timeValues.length - 1) {
    state.timeIndex = 0;
    $("#time-slider").value = "0";
    $("#time-label").textContent = currentTimeLabel();
    try {
      await refreshMap(null);
      await refreshOpenInspector();
    } catch (error) {
      toast("Playback stopped", error.message, true);
      stopPlayback();
      return;
    }
  }
  if (state.playing) state.playTimer = window.setTimeout(advancePlayback, playbackTiming().delay);
}

function llmInferenceReady() {
  const providers = state.assistantStatus?.providers || [];
  const selected = providers.find((item) => item.id === state.assistantProvider);
  const active = selected?.configured ? selected : providers.find((item) => item.configured);
  return active?.connection === "inference_verified";
}

function syncLlmAccessGate(ready = llmInferenceReady()) {
  const panel = $("#assistant-panel");
  const blocked = state.assistantSurface === "llm" && !ready;
  panel?.classList.toggle("is-api-required", blocked);
  const gate = $("#assistant-api-gate");
  if (gate) gate.hidden = !blocked;
  ["#assistant-input", "#assistant-send", "#assistant-upload", "#assistant-new-chat", "#assistant-new-chat-drawer"].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = blocked;
  });
  $$("#assistant-suggestions button").forEach((button) => { button.disabled = blocked; });
  if (blocked && state.assistantChatDrawerOpen) setAssistantChatDrawer(false);
}

function renderAssistantStatus() {
  const status = state.assistantStatus || {};
  const providers = status.providers || [];
  const configuredProviders = providers.filter((item) => item.configured);
  if (!providers.some((item) => item.id === state.assistantProvider) && providers.length) state.assistantProvider = providers[0].id;
  const selected = providers.find((item) => item.id === state.assistantProvider) || null;
  const active = selected?.configured ? selected : configuredProviders[0] || selected;
  const configured = configuredProviders.length > 0;
  const inferenceReady = active?.connection === "inference_verified";
  const inferenceFailed = active?.connection === "inference_failed";
  const settingsButton = $("#assistant-settings");
  if (settingsButton) {
    settingsButton.classList.toggle("is-connected", state.assistantSurface === "llm" && inferenceReady);
    const settingsLabel = settingsButton.querySelector("b");
    if (settingsLabel) settingsLabel.textContent = state.assistantSurface !== "llm"
      ? "Connect API"
      : inferenceReady
        ? "LLM connected"
        : inferenceFailed
          ? "API request failed"
          : configured
            ? "Verification pending"
            : "Connect API";
  }
  $("#assistant-provider").textContent = state.assistantSurface === "llm"
    ? active?.configured ? `${active.label} · ${active.model}` : "Connect a provider in settings"
    : "Grounded local guide · no API needed";
  const modelChip = $("#assistant-model-chip");
  if (state.assistantSurface === "llm" && active?.configured) {
    $("#assistant-provider").textContent = inferenceReady
      ? `${active.label} · ${active.model}`
      : inferenceFailed
        ? `${active.label} · request failed`
        : `${active.label} · verification pending`;
  }
  if (modelChip) {
    const chipProvider = selected?.label || active?.label || "GPT";
    const chipModel = selected?.model || active?.model || "model not selected";
    modelChip.textContent = state.assistantSurface === "llm" ? `${chipProvider} · ${chipModel}` : "";
    modelChip.classList.toggle("is-connected", inferenceReady);
  }
  $("#assistant-launch-status").textContent = "Local · no API needed";
  $("#llm-launch-status").textContent = configured
    ? inferenceReady
      ? `${active.label} connected`
      : inferenceFailed
        ? `${active.label} request failed`
        : `${configuredProviders.length} connection${configuredProviders.length === 1 ? "" : "s"} pending verification`
    : "Connect GPT, Claude, DeepSeek or Gemini";
  $("#llm-copilot-launch").classList.toggle("is-online", inferenceReady);
  const grid = $("#provider-status-grid");
  if (grid) {
    grid.innerHTML = providers.map((item) => {
      const availableCount = Number(item.available_model_count || 0);
      const itemReady = item.connection === "inference_verified";
      const itemFailed = item.connection === "inference_failed";
      const statusLabel = itemReady
        ? "Connected"
        : itemFailed
          ? "Connection failed"
          : item.configured
            ? "Verify API"
            : "Needs API";
      const catalogCopy = itemReady
        ? item.catalog_status === "verified"
          ? `${availableCount} model${availableCount === 1 ? "" : "s"} returned by this API`
          : "Inference verified · model list unavailable"
        : itemFailed
          ? "API request not verified · retry connection"
          : item.configured
            ? "Key held for this session · verification pending"
            : "Connect API to check models";
      return `
        <button class="provider-status-card${itemReady ? " is-connected" : ""}${item.id === state.assistantProvider ? " is-selected" : ""}" data-provider-card="${escapeHtml(item.id)}" type="button">
          <span>${statusLabel}</span><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(catalogCopy)}</small>
        </button>
      `;
    }).join("");
  }
  syncLlmAccessGate(inferenceReady);
}

function updateProviderForm(provider = state.assistantProvider) {
  const details = (state.assistantStatus?.providers || []).find((item) => item.id === provider);
  state.assistantProvider = provider;
  $("#assistant-provider-select").value = provider;
  const models = details?.models || [];
  const catalogVerified = details?.catalog_status === "verified";
  const isDisabled = (item) => catalogVerified && (item.available === false || item.platform_compatible === false);
  const selectable = models.find((item) => !isDisabled(item));
  const currentModel = details?.model;
  const selectedModel = models.some((item) => item.id === currentModel && !isDisabled(item))
    ? currentModel
    : selectable?.id || models[0]?.id || "";
  $("#assistant-model-select").innerHTML = models.length
    ? models.map((item) => {
      const unavailable = isDisabled(item);
      const supportLabel = catalogVerified
        ? unavailable
          ? ` · ${item.available === false ? "not returned by this API" : "not suitable for platform analysis"}`
          : " · available"
        : "";
      return `<option value="${escapeHtml(item.id)}"${unavailable ? " disabled" : ""}>${escapeHtml(item.label)} · ${escapeHtml(item.id)}${item.tool_use ? " · platform tools" : " · text only"}${supportLabel}</option>`;
    }).join("")
    : '<option value="" disabled selected>Connect API to list models</option>';
  $("#assistant-model-select").value = selectedModel;
  const modelHelp = $("#assistant-model-help");
  if (modelHelp) {
    modelHelp.textContent = catalogVerified
      ? `${Number(details?.available_model_count || 0)} model${Number(details?.available_model_count || 0) === 1 ? "" : "s"} returned by this API. Grey options cannot run platform analysis.`
      : details?.connection === "inference_verified"
        ? "Inference is verified, but the provider model list is temporarily unavailable."
        : details?.connection === "inference_failed"
          ? "The API request was not verified. Check the message, then retry the connection."
          : details?.configured
            ? "The key is held for this session and is waiting for verification."
        : "Connect this provider to list the models available to your API.";
  }
  $("#assistant-api-key").value = "";
  $("#assistant-provider-disconnect").disabled = !details?.configured;
  const saveButton = $("#assistant-provider-save");
  if (saveButton) saveButton.textContent = details?.connection === "inference_verified"
    ? "Save model"
    : details?.configured
      ? "Retry connection"
      : "Connect provider";
  renderAssistantStatus();
}

async function refreshAssistantStatus() {
  state.assistantStatus = await api.assistantStatus(state.assistantSessionId);
  renderAssistantStatus();
  return state.assistantStatus;
}

async function refreshAssistantModels(provider = state.assistantProvider, notify = false) {
  try {
    state.assistantStatus = await api.refreshAssistantProviderModels(state.assistantSessionId, provider);
    renderAssistantStatus();
    updateProviderForm(provider);
    if (notify) toast("Models refreshed", "Only models confirmed by this provider are selectable.");
    return true;
  } catch (error) {
    if (notify) toast("Models could not be checked", error.message, true);
    return false;
  }
}

async function openAssistantSettings() {
  try {
    await refreshAssistantStatus();
    const selected = (state.assistantStatus?.providers || []).find((item) => item.id === state.assistantProvider);
    if (selected?.configured && selected.catalog_status !== "verified") await refreshAssistantModels(state.assistantProvider);
  } catch (error) {
    toast("Provider status unavailable", error.message, true);
  }
  updateProviderForm(state.assistantProvider);
  const dialog = $("#assistant-settings-dialog");
  if (!dialog.open) dialog.showModal();
}

async function configureAssistantProvider(clear = false) {
  const provider = $("#assistant-provider-select").value;
  const button = clear ? $("#assistant-provider-disconnect") : $("#assistant-provider-save");
  button.disabled = true;
  try {
    const selectedModel = $("#assistant-model-select").value;
    const selectedOption = $("#assistant-model-select").selectedOptions?.[0];
    if (!clear && (!selectedModel || selectedOption?.disabled)) throw new Error("Connect this provider to list a selectable model first.");
    state.assistantStatus = await api.configureAssistantProvider({
      session_id: state.assistantSessionId,
      provider,
      model: clear ? null : selectedModel,
      api_key: clear ? null : $("#assistant-api-key").value.trim(),
      clear,
    });
    state.assistantProvider = provider;
    $("#assistant-api-key").value = "";
    renderAssistantStatus();
    updateProviderForm(provider);
    const connectedProvider = (state.assistantStatus?.providers || []).find((item) => item.id === provider);
    if (!clear) {
      const verification = state.assistantStatus?.verification || {};
      if (verification.success) {
        toast(
          "LLM connected",
          `${connectedProvider?.label || provider} · ${verification.selected_model} completed a real inference test${verification.attempt_count > 1 ? ` after ${verification.attempt_count} compatible-model attempts` : ""}.`,
        );
        $("#assistant-settings-dialog").close();
      } else {
        const transient = ["network_or_timeout", "upstream_unavailable", "rate_limited"].includes(verification.error_code);
        toast(
          transient ? "Provider temporarily unavailable" : "API connection failed",
          verification.message || "The provider could not generate a test answer.",
          true,
        );
      }
      return;
    }
    toast("API disconnected", "LLM Copilot is locked. Platform Assistant remains available separately without an API.");
  } catch (error) {
    toast(clear ? "Provider could not disconnect" : "Provider could not connect", error.message, true);
  } finally {
    button.disabled = false;
  }
}

function setAssistantMode(mode) {
  if (!["guide", "autopilot"].includes(mode)) return;
  state.assistantMode = mode;
  $$('[data-assistant-mode]').forEach((button) => button.classList.toggle("is-active", button.dataset.assistantMode === mode));
  const notes = {
    guide: "Ask & explain answers without clicking or running analyses.",
    autopilot: "Operate in platform may run allowlisted, reversible steps here; it cannot perform live-world operations.",
  };
  $("#assistant-mode-note").textContent = notes[mode];
  $("#assistant-input").placeholder = mode === "autopilot"
    ? "Describe the outcome — the assistant will do safe in-platform steps…"
    : "Ask how the platform or current evidence works…";
}

function updateAssistantContext() {
  const context = $("#assistant-context");
  if (!context) return;
  const llm = state.assistantSurface === "llm";
  context.innerHTML = state.project
    ? `<span></span><strong>${escapeHtml(state.project.name)} · ${llm ? "LLM Copilot" : escapeHtml(state.currentView)}</strong>`
    : `<span></span><strong>${llm ? "No project · LLM Copilot" : "No project · platform guide"}</strong>`;
}

function assistantSurfaceCopy() {
  if (state.assistantSurface === "llm") {
    return {
      welcome: "Ask about the uploaded project or request one of the six core analyses. I will understand the goal, select the strongest runnable model, validate the scope and run it here automatically.",
      suggestions: ["Summarise this dataset", "Find the top 20 critical roads", "Are there data quality issues?"],
      note: "LLM Copilot answers from the current project and automatically runs validated core analyses.",
      placeholder: "Ask about the project or request a core analysis…",
    };
  }
  return {
    welcome: "Choose a destination above or ask a question. I can explain every workflow, open the exact control and prepare a reviewable analysis without changing your source files. I run a model only after your confirmation, then return its visual result.",
    suggestions: ["How do I use this platform?", "Help me upload data with different column names."],
    note: "Ask & explain answers without clicking or running analyses.",
    placeholder: "Ask how the platform works…",
  };
}

function assistantChatDate(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "";
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function renderAssistantChatList() {
  const list = $("#assistant-chat-list");
  if (!list) return;
  if (!state.assistantChats.length) {
    list.innerHTML = '<p class="assistant-chat-empty">No saved chats yet. Start a task and it will appear here.</p>';
    return;
  }
  list.innerHTML = state.assistantChats.map((chat) => `
    <div class="assistant-recent-row${chat.chat_id === state.assistantChatId ? " is-active" : ""}">
      <button class="assistant-recent-chat" data-assistant-chat-id="${escapeHtml(chat.chat_id)}" type="button">
        <strong>${escapeHtml(chat.title || "New chat")}</strong>
        <small>${escapeHtml(assistantChatDate(chat.updated_at))} · ${escapeHtml(chat.message_count || 0)} messages</small>
        <b aria-hidden="true">›</b>
      </button>
      <button class="assistant-chat-delete" data-delete-assistant-chat-id="${escapeHtml(chat.chat_id)}" data-chat-title="${escapeHtml(chat.title || "New chat")}" type="button" aria-label="Delete conversation" title="Delete conversation">Delete</button>
    </div>
  `).join("");
}

async function refreshAssistantChats() {
  if (state.assistantSurface !== "llm") return [];
  try {
    const chats = await api.assistantChats(60);
    const projectId = currentProjectId();
    state.assistantChats = (Array.isArray(chats) ? chats : []).filter((chat) => (
      (chat.project_id || null) === (projectId || null)
    ));
    renderAssistantChatList();
  } catch (error) {
    const list = $("#assistant-chat-list");
    if (list) list.innerHTML = `<p class="assistant-chat-empty">Recent chats are unavailable: ${escapeHtml(error.message)}</p>`;
  }
  return state.assistantChats;
}

function setAssistantChatDrawer(open) {
  const allowed = state.assistantSurface === "llm" && llmInferenceReady();
  state.assistantChatDrawerOpen = Boolean(open && allowed);
  const drawer = $("#assistant-chat-drawer");
  const panel = $("#assistant-panel");
  const toggle = $("#assistant-chat-toggle");
  drawer?.classList.toggle("is-open", state.assistantChatDrawerOpen);
  drawer?.setAttribute("aria-hidden", String(!state.assistantChatDrawerOpen));
  panel?.classList.toggle("has-chat-drawer", state.assistantChatDrawerOpen);
  toggle?.setAttribute("aria-expanded", String(state.assistantChatDrawerOpen));
  if (state.assistantChatDrawerOpen) refreshAssistantChats();
}

function normaliseAssistantSurface(surface = state.assistantSurface) {
  return surface === "llm" ? "llm" : "platform";
}

function assistantHistoryForSurface(surface = state.assistantSurface) {
  return normaliseAssistantSurface(surface) === "llm" ? state.assistantHistory : state.platformAssistantHistory;
}

function currentAssistantHistory() {
  return assistantHistoryForSurface(state.assistantSurface);
}

function assistantActionsForSurface(surface = state.assistantSurface) {
  return normaliseAssistantSurface(surface) === "llm" ? state.assistantActions : state.platformAssistantActions;
}

function currentAssistantActions() {
  return assistantActionsForSurface(state.assistantSurface);
}

function setAssistantActionsForSurface(surface, actions) {
  if (normaliseAssistantSurface(surface) === "llm") state.assistantActions = actions;
  else state.platformAssistantActions = actions;
}

function setCurrentAssistantActions(actions) {
  setAssistantActionsForSurface(state.assistantSurface, actions);
}

function ensureAssistantSurfaceView(surface) {
  const key = normaliseAssistantSurface(surface);
  if (!state.assistantSurfaceViews[key]) {
    state.assistantSurfaceViews[key] = {
      messageNodes: document.createDocumentFragment(),
      actionNodes: document.createDocumentFragment(),
      suggestionsHtml: "",
      scrollTop: 0,
    };
  }
  return state.assistantSurfaceViews[key];
}

function assistantSurfaceContainer(surface, kind = "messages") {
  const key = normaliseAssistantSurface(surface);
  if (key === state.assistantSurface) return kind === "actions" ? $("#assistant-actions") : $("#assistant-messages");
  const view = ensureAssistantSurfaceView(key);
  return kind === "actions" ? view.actionNodes : view.messageNodes;
}

function appendAssistantNode(node, surface = state.assistantSurface, kind = "messages") {
  const key = normaliseAssistantSurface(surface);
  const container = assistantSurfaceContainer(key, kind);
  container?.append(node);
  if (key === state.assistantSurface && kind === "messages") {
    container.scrollTop = container.scrollHeight;
  } else if (kind === "messages") {
    ensureAssistantSurfaceView(key).scrollTop = Number.MAX_SAFE_INTEGER;
  }
  return node;
}

function replaceAssistantSurfaceHtml(surface, kind, html) {
  const key = normaliseAssistantSurface(surface);
  if (key === state.assistantSurface) {
    const element = kind === "actions" ? $("#assistant-actions") : $("#assistant-suggestions");
    if (element) element.innerHTML = html;
    return;
  }
  const view = ensureAssistantSurfaceView(key);
  if (kind === "suggestions") {
    view.suggestionsHtml = html;
    return;
  }
  const fragment = document.createDocumentFragment();
  const template = document.createElement("template");
  template.innerHTML = html;
  fragment.append(template.content);
  view.actionNodes = fragment;
}

function scrollAssistantSurface(surface = state.assistantSurface) {
  if (normaliseAssistantSurface(surface) !== state.assistantSurface) return;
  const messages = $("#assistant-messages");
  if (messages) messages.scrollTop = messages.scrollHeight;
}

function assistantActivityForSurface(surface = state.assistantSurface) {
  return normaliseAssistantSurface(surface) === "llm" ? state.assistantActivity : state.platformAssistantActivity;
}

function setAssistantActivityForSurface(surface, activity) {
  if (normaliseAssistantSurface(surface) === "llm") state.assistantActivity = activity;
  else state.platformAssistantActivity = activity;
}

function storeAssistantSurfaceView(surface) {
  const messages = $("#assistant-messages");
  const actions = $("#assistant-actions");
  const suggestions = $("#assistant-suggestions");
  if (!messages || !actions || !suggestions) return;
  const messageNodes = document.createDocumentFragment();
  const actionNodes = document.createDocumentFragment();
  while (messages.firstChild) messageNodes.append(messages.firstChild);
  while (actions.firstChild) actionNodes.append(actions.firstChild);
  state.assistantSurfaceViews[surface] = {
    messageNodes,
    actionNodes,
    suggestionsHtml: suggestions.innerHTML,
    scrollTop: messages.scrollTop,
  };
}

function restoreAssistantSurfaceView(surface) {
  const messages = $("#assistant-messages");
  const actions = $("#assistant-actions");
  const suggestions = $("#assistant-suggestions");
  if (!messages || !actions || !suggestions) return;
  messages.replaceChildren();
  actions.replaceChildren();
  const view = state.assistantSurfaceViews[surface];
  if (view) {
    messages.append(view.messageNodes);
    actions.append(view.actionNodes);
    suggestions.innerHTML = view.suggestionsHtml || "";
    messages.scrollTop = view.scrollTop || 0;
    return;
  }
  suggestions.innerHTML = "";
  const welcome = appendAssistantMessage("assistant", assistantSurfaceCopy().welcome, "", surface);
  welcome.id = "assistant-welcome";
}

function resetAssistantConversationView() {
  const llm = state.assistantSurface === "llm";
  if (llm) {
    state.assistantHistory = [];
    state.assistantChatArtifacts = [];
    state.assistantActions = [];
    state.assistantContract = null;
    state.commandContract = null;
    state.assistantActivity = null;
    state.assistantPendingTask = null;
    state.assistantLastWorkflowKey = null;
  } else {
    state.platformAssistantHistory = [];
    state.platformAssistantActions = [];
    state.platformAssistantActivity = null;
    state.platformAssistantBusy = false;
    state.platformAssistantMessageRequestId += 1;
  }
  $("#assistant-messages").innerHTML = "";
  $("#assistant-actions").innerHTML = "";
  const welcome = appendAssistantMessage("assistant", assistantSurfaceCopy().welcome, "", state.assistantSurface);
  welcome.id = "assistant-welcome";
  renderAssistantSurface();
}

function abandonAssistantPendingInputs() {
  const pending = state.assistantPendingTask;
  if (!pending) return;
  if (pending.type === "route" && state.routeBuilding) closeRouteBuilder({ restoreSelection: true, restoreScenario: true });
  else if (pending.type === "recovery" && state.recoveryBuilding) closeRecoveryBuilder({ restoreSelection: true });
  else if (["scenario", "matsim"].includes(pending.type) && state.scenarioBuilding) closeScenarioBuilder({ restore: true });
  else if (pending.type === "stress" && state.stressBuilding) closeStressBuilder();
}

async function startNewAssistantChat({ closeDrawer = true } = {}) {
  state.assistantMessageRequestId += 1;
  state.assistantAnalysisRequestId += 1;
  state.commandRequestId += 1;
  state.analysisContextVersion += 1;
  state.assistantBusy = false;
  state.assistantTurnRunning = false;
  state.assistantActiveQueueItem = null;
  state.assistantCompoundWorkflow = null;
  abandonAssistantPendingInputs();
  state.assistantPendingTask = null;
  if (state.activeAnalysisJobId) {
    const jobId = state.activeAnalysisJobId;
    state.activeAnalysisJobId = null;
    api.cancelJob(jobId).catch(() => null);
  }
  state.assistantChatId = null;
  state.assistantChatProjectId = null;
  resetAssistantConversationView();
  renderAssistantTaskQueue();
  renderAssistantChatList();
  if (closeDrawer && !$("#assistant-panel")?.classList.contains("is-fullscreen")) setAssistantChatDrawer(false);
  window.setTimeout(() => $("#assistant-input")?.focus(), 40);
}

async function ensureAssistantChat(title = "New chat", { allowHidden = false } = {}) {
  if (state.assistantSurface !== "llm" && !allowHidden) return null;
  if (state.assistantChatId && state.assistantChatProjectId === currentProjectId()) return state.assistantChatId;
  const draftQueueKey = assistantQueueKey(null, currentProjectId());
  state.assistantChatId = null;
  state.assistantChatProjectId = null;
  const chat = await api.createAssistantChat({
    title,
    project_id: currentProjectId(),
  });
  state.assistantChatId = chat.chat_id;
  state.assistantChatProjectId = chat.project_id || null;
  const draftQueue = state.assistantTaskQueues.get(draftQueueKey) || [];
  if (draftQueue.length) {
    const chatQueueKey = assistantQueueKey(chat.chat_id, chat.project_id || currentProjectId());
    const chatQueue = state.assistantTaskQueues.get(chatQueueKey) || [];
    draftQueue.forEach((item) => { item.chatId = chat.chat_id; item.projectId = chat.project_id || currentProjectId(); });
    state.assistantTaskQueues.set(chatQueueKey, [...chatQueue, ...draftQueue]);
    state.assistantTaskQueues.delete(draftQueueKey);
  }
  if (state.assistantActiveQueueItem) {
    state.assistantActiveQueueItem.chatId = chat.chat_id;
    state.assistantActiveQueueItem.projectId = chat.project_id || currentProjectId();
  }
  await refreshAssistantChats();
  return chat.chat_id;
}

async function persistAssistantChatMessage(role, content, metadata = {}, owner = {}) {
  const ownerSurface = normaliseAssistantSurface(owner.surface || "llm");
  if (ownerSurface !== "llm" || !String(content || "").trim()) return;
  try {
    const chatId = owner.chatId || await ensureAssistantChat(role === "user" ? content : "New chat", { allowHidden: true });
    if (!chatId) return;
    await api.appendAssistantChatMessage(chatId, { role, content, metadata });
    await refreshAssistantChats();
  } catch (error) {
    toast("Chat history not saved", error.message, true);
  }
}

async function loadAssistantChat(chatId) {
  if (!chatId || chatId === state.assistantChatId) {
    if (!$("#assistant-panel")?.classList.contains("is-fullscreen")) setAssistantChatDrawer(false);
    return;
  }
  try {
    abandonAssistantPendingInputs();
    const chat = await api.assistantChat(chatId);
    if ((chat.project_id || null) !== (currentProjectId() || null)) {
      toast("Conversation belongs to another project", "Open that project before reopening this conversation.", true);
      await refreshAssistantChats();
      return;
    }
    state.assistantMessageRequestId += 1;
    state.assistantAnalysisRequestId += 1;
    state.commandRequestId += 1;
    state.assistantChatId = chat.chat_id;
    state.assistantChatProjectId = chat.project_id || null;
    state.assistantHistory = (chat.messages || []).map((item) => ({ role: item.role, content: item.content }));
    state.assistantChatArtifacts = (chat.messages || [])
      .map((item) => assistantArtifactFromMetadata(item.metadata || {}, item.created_at))
      .filter((artifact) => artifact?.project_id === currentProjectId());
    state.assistantActions = [];
    state.assistantContract = null;
    state.assistantActivity = null;
    state.assistantPendingTask = null;
    state.assistantLastWorkflowKey = null;
    state.assistantTurnRunning = false;
    state.assistantActiveQueueItem = null;
    state.assistantCompoundWorkflow = null;
    $("#assistant-messages").innerHTML = "";
    $("#assistant-actions").innerHTML = "";
    for (const message of state.assistantHistory) appendAssistantMessage(message.role, message.content, "", "llm");
    if (!state.assistantHistory.length) {
      const welcome = appendAssistantMessage("assistant", assistantSurfaceCopy().welcome, "", "llm");
      welcome.id = "assistant-welcome";
    }
    renderAssistantChatList();
    renderAssistantTaskQueue();
    window.setTimeout(() => drainAssistantTaskQueue(), 0);
    if (!$("#assistant-panel")?.classList.contains("is-fullscreen")) setAssistantChatDrawer(false);
  } catch (error) {
    toast("Conversation could not open", error.message, true);
  }
}

async function deleteAssistantChat(chatId, title = "this conversation") {
  if (!chatId || !window.confirm(`Delete “${title}”? This removes only this LLM conversation.`)) return;
  try {
    await api.deleteAssistantChat(chatId);
    state.assistantChats = state.assistantChats.filter((chat) => chat.chat_id !== chatId);
    if (state.assistantChatId === chatId) {
      state.assistantChatId = null;
      state.assistantChatProjectId = null;
      resetAssistantConversationView();
    }
    renderAssistantChatList();
    toast("Conversation deleted", "The uploaded project and analysis results were not changed.");
  } catch (error) {
    toast("Conversation could not be deleted", error.message, true);
  }
}

function startAssistantActivity(surface = "llm") {
  const ownerSurface = normaliseAssistantSurface(surface);
  const stream = document.createElement("div");
  stream.className = "assistant-process-stream";
  appendAssistantNode(stream, ownerSurface);
  setAssistantActivityForSurface(ownerSurface, { stream, events: new Map(), surface: ownerSurface });
  updateAssistantActivity("understand", "Reading your goal", "active", ownerSurface);
  scrollAssistantSurface(ownerSurface);
  return stream;
}

function updateAssistantActivity(step, detail, status = "active", surface = "llm") {
  const ownerSurface = normaliseAssistantSurface(surface);
  const activity = assistantActivityForSurface(ownerSurface);
  if (!activity?.stream || (!activity.stream.isConnected && !activity.stream.parentNode)) return;
  let event = activity.events.get(step);
  if (!event) {
    event = document.createElement("article");
    event.className = "assistant-process-event";
    event.dataset.processStep = step;
    event.innerHTML = '<span aria-hidden="true"></span><div><strong></strong><small></small></div>';
    activity.stream.append(event);
    activity.events.set(step, event);
  }
  event.className = `assistant-process-event is-${status}`;
  event.querySelector("strong").textContent = detail || "Working";
  event.querySelector("small").textContent = status === "complete"
    ? "Completed"
    : status === "failed"
      ? "Could not complete"
      : status === "waiting"
        ? "Waiting for your input"
        : "Working";
  scrollAssistantSurface(ownerSurface);
}

const ASSISTANT_MODEL_RECEIPTS = {
  critical_components: {
    title: "Structural critical-road screening",
    formula: `<math display="block" aria-label="Criticality of road e equals the relative loss of network efficiency after removing road e">
      <msub><mi>C</mi><mi>e</mi></msub><mo>=</mo><mfrac><mrow><mi>E</mi><mo>(</mo><mi>G</mi><mo>)</mo><mo>−</mo><mi>E</mi><mo>(</mo><mi>G</mi><mo>∖</mo><mi>e</mi><mo>)</mo></mrow><mrow><mi>E</mi><mo>(</mo><mi>G</mi><mo>)</mo></mrow></mfrac><mo>,</mo>
      <mspace width="1em"/><mi>E</mi><mo>(</mo><mi>G</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mi>n</mi><mo>(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo>)</mo></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>≠</mo><mi>j</mi></mrow><mrow><mi>i</mi><mo>,</mo><mi>j</mi><mo>∈</mo><mi>V</mi></mrow></munderover><mfrac><mn>1</mn><msub><mi>d</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac>
    </math>`,
    symbols: '<math display="inline"><mi>e</mi></math> is the tested road; <math display="inline"><mi>G</mi></math> is the network; <math display="inline"><msub><mi>d</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></math> is shortest-path impedance between nodes <math display="inline"><mi>i</mi></math> and <math display="inline"><mi>j</mi></math>.',
    algorithm: "Deterministic candidate screening, link removal and network-efficiency recomputation.",
    boundary: "Structural importance; not observed demand, congestion or population impact.",
  },
  spatial_hotspots: {
    title: "Regional propagation-critical area screening",
    formula: `<math display="block" aria-label="Regional score at propagation time tau and degradation alpha">
      <msub><mi>S</mi><mi>R</mi></msub><mo>(</mo><mi>τ</mi><mo>,</mo><mi>α</mi><mo>)</mo><mo>=</mo><mi>max</mi><mfenced><mn>0</mn><mfrac><mrow><mi>DPC</mi><mo>(</mo><mi>τ</mi><mo>)</mo><mo>−</mo><msup><mi>DPC</mi><mrow><mo>−</mo><mi>R</mi><mo>,</mo><mi>α</mi></mrow></msup><mo>(</mo><mi>τ</mi><mo>)</mo></mrow><mrow><mi>DPC</mi><mo>(</mo><mi>τ</mi><mo>)</mo></mrow></mfrac></mfenced>
    </math>`,
    symbols: '<math display="inline"><mi>R</mi></math> is a candidate region; <math display="inline"><mi>τ</mi></math> is propagation time; <math display="inline"><mi>α</mi></math> is the confirmed in-region degradation strength.',
    algorithm: "Regional DPC is used only when its data gates pass; otherwise the labelled fixed-grid observed-degradation tier is used.",
    boundary: "Propagation sensitivity or observed hotspot evidence; not hazard probability or causal impact.",
  },
  recovery: {
    title: "Adaptive marginal recovery ordering",
    formula: `<math display="block" aria-label="Choose the next recovery action with the largest marginal service gain">
      <msup><mi>e</mi><mo>*</mo></msup><mo>=</mo><munder><mo>arg&#x2061;max</mo><mrow><mi>e</mi><mo>∉</mo><mi>R</mi></mrow></munder><mo>[</mo><mi>S</mi><mo>(</mo><mi>R</mi><mo>∪</mo><mo>{</mo><mi>e</mi><mo>}</mo><mo>)</mo><mo>−</mo><mi>S</mi><mo>(</mo><mi>R</mi><mo>)</mo><mo>]</mo>
    </math>`,
    symbols: '<math display="inline"><mi>R</mi></math> is the restored set; <math display="inline"><mi>e</mi></math> is one remaining road or area package; <math display="inline"><mi>S</mi></math> is supported network service.',
    algorithm: "Recompute the marginal service gain after each selected restoration action.",
    boundary: "A recovery screen unless repair cost or duration is uploaded; not a budget-optimal schedule.",
  },
  route: {
    title: "Distinct least-impedance route alternatives",
    formula: `<math display="block" aria-label="Route k minimises the sum of disrupted link impedance">
      <msubsup><mi>P</mi><mi>k</mi><mo>*</mo></msubsup><mo>=</mo><munder><mo>arg&#x2061;min</mo><mrow><mi>P</mi><mo>∈</mo><msub><mi>𝒫</mi><mrow><mi>o</mi><mi>d</mi></mrow></msub></mrow></munder><munder><mo>∑</mo><mrow><mi>e</mi><mo>∈</mo><mi>P</mi></mrow></munder><msubsup><mi>w</mi><mi>e</mi><mo>′</mo></msubsup>
    </math>`,
    symbols: '<math display="inline"><msub><mi>𝒫</mi><mrow><mi>o</mi><mi>d</mi></mrow></msub></math> is the feasible origin–destination path set; <math display="inline"><msubsup><mi>w</mi><mi>e</mi><mo>′</mo></msubsup></math> is disrupted impedance on road <math display="inline"><mi>e</mi></math>.',
    algorithm: "Compute distinct loopless alternatives and rank the feasible paths by total disrupted impedance.",
    boundary: "Retrospective network routing; not live navigation or behavioural route choice.",
  },
  stress: {
    title: "Progressive network-resilience stress test",
    formula: `<math display="block" aria-label="Retained efficiency after q cumulative failures">
      <msub><mi>R</mi><mi>q</mi></msub><mo>=</mo><mfrac><mrow><mi>E</mi><mo>(</mo><msub><mi>G</mi><mi>q</mi></msub><mo>)</mo></mrow><mrow><mi>E</mi><mo>(</mo><msub><mi>G</mi><mn>0</mn></msub><mo>)</mo></mrow></mfrac>
    </math>`,
    symbols: '<math display="inline"><msub><mi>G</mi><mn>0</mn></msub></math> is baseline topology; <math display="inline"><msub><mi>G</mi><mi>q</mi></msub></math> is the network after <math display="inline"><mi>q</mi></math> cumulative tested failures.',
    algorithm: "Remove roads in the verified critical order and recompute retained network efficiency at every stage.",
    boundary: "Topology resilience; not observed traffic delay, demand response or failure probability.",
  },
  scenario: {
    title: "Deterministic disruption counterfactual",
    formula: `<math display="block" aria-label="Disrupted speed and travel time for road e">
      <msubsup><mi>v</mi><mi>e</mi><mo>′</mo></msubsup><mo>=</mo><mo>(</mo><mn>1</mn><mo>−</mo><msub><mi>r</mi><mi>e</mi></msub><mo>)</mo><msub><mi>v</mi><mi>e</mi></msub><mo>,</mo><mspace width="1em"/><msubsup><mi>t</mi><mi>e</mi><mo>′</mo></msubsup><mo>=</mo><mfrac><msub><mi>L</mi><mi>e</mi></msub><msubsup><mi>v</mi><mi>e</mi><mo>′</mo></msubsup></mfrac>
    </math>`,
    symbols: '<math display="inline"><msub><mi>r</mi><mi>e</mi></msub></math> is confirmed speed reduction; <math display="inline"><msub><mi>L</mi><mi>e</mi></msub></math>, <math display="inline"><msub><mi>v</mi><mi>e</mi></msub></math> and <math display="inline"><msubsup><mi>t</mi><mi>e</mi><mo>′</mo></msubsup></math> are road length, baseline speed and disrupted travel time.',
    algorithm: "Apply the confirmed closure, speed or travel-time operator, then recompute the supported network response.",
    boundary: "A deterministic what-if test; not a forecast of a disaster occurring.",
  },
  matsim: {
    title: "MATSim baseline–disruption experiment",
    formula: `<math display="block" aria-label="Each agent selects the route with minimum experienced disrupted travel time">
      <msubsup><mi>P</mi><mi>a</mi><mo>*</mo></msubsup><mo>=</mo><munder><mo>arg&#x2061;min</mo><mrow><mi>P</mi><mo>∈</mo><msub><mi>𝒫</mi><mi>a</mi></msub></mrow></munder><munder><mo>∑</mo><mrow><mi>e</mi><mo>∈</mo><mi>P</mi></mrow></munder><msubsup><mi>t</mi><mrow><mi>e</mi><mo>,</mo><mi>a</mi></mrow><mo>′</mo></msubsup>
    </math>`,
    symbols: '<math display="inline"><mi>a</mi></math> is an agent; <math display="inline"><msub><mi>𝒫</mi><mi>a</mi></msub></math> is its feasible path set; <math display="inline"><msubsup><mi>t</mi><mrow><mi>e</mi><mo>,</mo><mi>a</mi></mrow><mo>′</mo></msubsup></math> is experienced disrupted road travel time.',
    algorithm: "Run paired baseline and disrupted experiments through the existing process-isolated MATSim worker.",
    boundary: "Uploaded OD demand is used when present; otherwise results remain an explicitly exploratory probe simulation.",
  },
};

function appendAssistantModelReceipt(type, { modelId = "", tier = "", parameters = [] } = {}) {
  const receipt = ASSISTANT_MODEL_RECEIPTS[type];
  const activity = assistantActivityForSurface("llm");
  if (!receipt || !activity?.stream || (!activity.stream.isConnected && !activity.stream.parentNode)) return null;
  const event = document.createElement("article");
  event.className = "assistant-process-event assistant-model-receipt is-complete";
  event.dataset.processStep = `model_${type}`;
  event.innerHTML = `
    <span aria-hidden="true"></span>
    <div>
      <strong>${escapeHtml(receipt.title)}${modelId ? ` · ${escapeHtml(modelId)}` : ""}</strong>
      ${tier ? `<small>${escapeHtml(tier)}</small>` : ""}
      <div class="assistant-math-expression">${receipt.formula}</div>
      ${parameters.length ? `<dl class="assistant-model-parameters">${parameters.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
      <p><b>Symbols</b> ${receipt.symbols}</p>
      <p><b>Algorithm</b> ${escapeHtml(receipt.algorithm)}</p>
      <p><b>Claim boundary</b> ${escapeHtml(receipt.boundary)}</p>
    </div>`;
  activity.stream.append(event);
  scrollAssistantSurface("llm");
  return event;
}

function renderAssistantSurface() {
  const llm = state.assistantSurface === "llm";
  const panel = $("#assistant-panel");
  if (panel) {
    panel.classList.toggle("is-platform", !llm);
    panel.classList.toggle("is-llm", llm);
  }
  updateAssistantContext();
  const copy = assistantSurfaceCopy();
  const isFresh = currentAssistantHistory().length === 0;
  const welcome = $("#assistant-welcome");
  if (isFresh && welcome) {
    const paragraph = welcome.querySelector("p");
    if (paragraph) paragraph.textContent = copy.welcome;
  }
  const suggestions = $("#assistant-suggestions");
  if (isFresh && suggestions) {
    suggestions.innerHTML = copy.suggestions
      .map((suggestion) => `<button type="button" data-assistant-prompt="${escapeHtml(suggestion)}">${escapeHtml(suggestion)}</button>`)
      .join("");
  }
  if (llm) {
    $("#assistant-mode-note").textContent = copy.note;
    $("#assistant-input").placeholder = copy.placeholder;
  } else {
    setAssistantMode(state.assistantMode);
  }
  syncLlmAccessGate();
  renderAssistantTaskQueue();
}

function updateLlmWindowToggle() {
  const button = $("#assistant-llm-window-toggle");
  const panel = $("#assistant-panel");
  if (!button || !panel) return;
  const expanded = panel.classList.contains("is-fullscreen");
  button.textContent = expanded ? "↙" : "⛶";
  button.setAttribute("aria-label", expanded ? "Restore LLM Copilot size" : "Maximise LLM Copilot");
  button.title = expanded ? "Restore LLM Copilot size" : "Maximise LLM Copilot";
}

function syncLlmFullscreenScrollLock() {
  const panel = $("#assistant-panel");
  const locked = Boolean(
    panel
    && panel.classList.contains("is-open")
    && panel.classList.contains("is-llm")
    && panel.classList.contains("is-fullscreen")
  );
  document.documentElement.classList.toggle("is-llm-fullscreen", locked);
  document.body.classList.toggle("is-llm-fullscreen", locked);
}

function restoreLlmPanelPosition() {
  const panel = $("#assistant-panel");
  if (!panel) return;
  panel.classList.remove("is-fullscreen", "is-dragging");
  setAssistantChatDrawer(false);
  state.assistantLlmDrag = null;
  if (state.assistantLlmStyleBackup) {
    const backup = state.assistantLlmStyleBackup;
    panel.style.left = backup.left;
    panel.style.top = backup.top;
    panel.style.right = backup.right;
    panel.style.bottom = backup.bottom;
    state.assistantLlmStyleBackup = null;
  }
  updateLlmWindowToggle();
  syncLlmFullscreenScrollLock();
}

function toggleLlmWindowSize() {
  if (state.assistantSurface !== "llm") return;
  const panel = $("#assistant-panel");
  if (!panel) return;
  const enteringFullscreen = !panel.classList.contains("is-fullscreen");
  panel.classList.toggle("is-fullscreen", enteringFullscreen);
  state.assistantLlmDrag = null;
  panel.classList.remove("is-dragging");
  setAssistantChatDrawer(enteringFullscreen && llmInferenceReady());
  updateLlmWindowToggle();
  syncLlmFullscreenScrollLock();
}

function bindLlmWindowInteractions() {
  const panel = $("#assistant-panel");
  const header = panel?.querySelector(":scope > header");
  const toggle = $("#assistant-llm-window-toggle");
  if (!panel || !header || !toggle) return;

  toggle.addEventListener("click", toggleLlmWindowSize);

  const stopDragging = (event) => {
    if (!state.assistantLlmDrag) return;
    if (event && header.hasPointerCapture?.(event.pointerId)) header.releasePointerCapture(event.pointerId);
    state.assistantLlmDrag = null;
    panel.classList.remove("is-dragging");
  };

  header.addEventListener("pointerdown", (event) => {
    if (state.assistantSurface !== "llm" || panel.classList.contains("is-fullscreen") || event.target.closest("button")) return;
    const rectangle = panel.getBoundingClientRect();
    if (!state.assistantLlmStyleBackup) {
      state.assistantLlmStyleBackup = {
        left: panel.style.left,
        top: panel.style.top,
        right: panel.style.right,
        bottom: panel.style.bottom,
      };
    }
    panel.style.left = `${rectangle.left}px`;
    panel.style.top = `${rectangle.top}px`;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    state.assistantLlmDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: rectangle.left,
      top: rectangle.top,
    };
    panel.classList.add("is-dragging");
    header.setPointerCapture?.(event.pointerId);
  });

  header.addEventListener("pointermove", (event) => {
    const drag = state.assistantLlmDrag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const rectangle = panel.getBoundingClientRect();
    const maxLeft = Math.max(12, window.innerWidth - rectangle.width - 12);
    const maxTop = Math.max(12, window.innerHeight - Math.min(rectangle.height, 110) - 12);
    const left = Math.max(12, Math.min(maxLeft, drag.left + event.clientX - drag.startX));
    const top = Math.max(12, Math.min(maxTop, drag.top + event.clientY - drag.startY));
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  });
  header.addEventListener("pointerup", stopDragging);
  header.addEventListener("pointercancel", stopDragging);
  header.addEventListener("lostpointercapture", () => stopDragging());
}

function setAssistantSurface(surface) {
  const nextSurface = surface === "llm" ? "llm" : "platform";
  if (nextSurface !== state.assistantSurface) {
    storeAssistantSurfaceView(state.assistantSurface);
    state.assistantSurface = nextSurface;
    restoreAssistantSurfaceView(nextSurface);
  }
  const platform = state.assistantSurface === "platform";
  if (platform) {
    setAssistantChatDrawer(false);
    restoreLlmPanelPosition();
  }
  $("#assistant-panel").classList.toggle("is-platform", platform);
  $("#assistant-panel").classList.toggle("is-llm", !platform);
  $("#assistant-surface-title").textContent = platform ? "Platform Assistant" : "LLM Copilot";
  $("#assistant-header-orb").textContent = platform ? "?" : "✦";
  renderAssistantStatus();
  renderAssistantSurface();
  syncLlmFullscreenScrollLock();
}

function setAssistantOpen(open, surface = state.assistantSurface) {
  setAssistantSurface(surface);
  $("#assistant-panel").classList.toggle("is-open", open);
  $("#assistant-panel").setAttribute("aria-hidden", String(!open));
  $("#assistant-launch").setAttribute("aria-expanded", String(open && state.assistantSurface === "platform"));
  $("#llm-copilot-launch").setAttribute("aria-expanded", String(open && state.assistantSurface === "llm"));
  if (open) {
    updateAssistantContext();
    if (state.assistantSurface === "llm") {
      refreshAssistantChats();
      if ($("#assistant-panel").classList.contains("is-fullscreen") && llmInferenceReady()) setAssistantChatDrawer(true);
    }
    window.setTimeout(() => $("#assistant-input").focus(), 120);
  } else {
    setAssistantChatDrawer(false);
  }
  syncLlmFullscreenScrollLock();
}

function appendAssistantMessage(role, message, notice = "", surface = "llm") {
  const article = document.createElement("article");
  article.className = `assistant-message ${role === "user" ? "is-user" : "is-assistant"}`;
  const marker = document.createElement("span");
  marker.textContent = role === "user" ? "You" : "✦";
  const body = document.createElement("div");
  const paragraph = document.createElement("p");
  paragraph.textContent = message;
  body.append(paragraph);
  if (notice) {
    const small = document.createElement("small");
    small.textContent = notice;
    body.append(small);
  }
  article.append(marker, body);
  return appendAssistantNode(article, surface);
}

function appendAssistantContract(contract, surface = "llm") {
  const card = document.createElement("article");
  card.className = `assistant-workcard${contract.ready ? " is-ready" : " is-blocked"}`;
  const messages = [...(contract.warnings || []), ...(contract.missing || [])];
  card.innerHTML = `
    <header><span>${contract.ready ? "Ready to run" : "More input needed"}</span><strong>${escapeHtml(contract.analysis_label)}</strong></header>
    <dl>
      <div><dt>Scope</dt><dd>${escapeHtml(contract.grounding?.scope_label || "Whole project")}</dd></div>
      <div><dt>Measure</dt><dd>${escapeHtml(contract.measure_label || contract.metric.replaceAll("_", " "))}</dd></div>
      <div><dt>Return</dt><dd>Top ${escapeHtml(contract.top_k)}</dd></div>
      <div><dt>Model</dt><dd>${escapeHtml(contract.model_id ? String(contract.model_id).replace("P5-M", "TWM-METHOD-") : contract.capability)}</dd></div>
      <div><dt>Tier</dt><dd>${escapeHtml(contract.selected_model_tier || "Current supported tier")}</dd></div>
      <div><dt>Readiness</dt><dd>${escapeHtml(String(contract.model_readiness || contract.status || "ready").replaceAll("_", " "))}</dd></div>
      <div><dt>Time</dt><dd>${escapeHtml(contract.time_end ? new Date(Number(contract.time_end) * 1000).toISOString() : "Current project setting")}</dd></div>
    </dl>
    ${messages.length ? `<div class="assistant-work-notes">${messages.slice(0, 3).map((message) => `<p>${escapeHtml(message)}</p>`).join("")}</div>` : ""}
    <footer>${contract.ready
      ? `<button class="primary-button ripple" data-assistant-run type="button">Confirm and run here</button>`
      : '<span>Change the selection or data, then ask again.</span>'}</footer>
  `;
  return appendAssistantNode(card, surface);
}

async function prepareAssistantAnalysis(command, { autoRun = false, surface = "llm" } = {}) {
  const ownerSurface = normaliseAssistantSurface(surface);
  if (!state.project) return appendAssistantMessage("assistant", "Upload or open a project before I prepare a numerical analysis.", "", ownerSurface);
  const analysisContext = analysisContextForCommand(command);
  if (!analysisContext.ready) {
    if (ownerSurface === "llm") {
      state.assistantPendingTask = {
        type: "analysis",
        sourceText: command,
        analysisType: analysisContext.intent,
        requestedCount: requestedCountFromText(command, null),
        phase: "scope",
        mapStep: null,
        chatId: state.assistantChatId,
        projectId: currentProjectId(),
        requestId: state.assistantMessageRequestId,
      };
      updateAssistantActivity("execute", "A map scope is required before this analysis can run", "waiting", ownerSurface);
      showAssistantPendingTaskStep();
      return null;
    }
    scheduleNextActionCues();
    return appendAssistantMessage("assistant", "Choose roads, points or draw an area first. The current scope is set to use an explicit map selection.", "", ownerSurface);
  }
  state.preferredModelId = null;
  const requestId = ++state.commandRequestId;
  $$('[data-assistant-run]').forEach((button) => {
    button.disabled = true;
    button.textContent = "Superseded by the latest request";
  });
  state.assistantContract = null;
  updateAssistantActivity("decision", "Matching the request to registered models", "active", ownerSurface);
  const working = appendAssistantMessage("assistant", "Preparing a grounded analysis contract…", "", ownerSurface);
  working.classList.add("is-typing");
  try {
    const contract = await api.interpret({
      project_id: state.project.project_id,
      text: command,
      selected_keys: analysisContext.selected_keys,
      bbox: analysisContext.bbox,
      current_metric: state.metric,
      current_time: currentTime(),
    });
    await attachAssistantChatDependencies(contract);
    attachConfirmedModelInputs(contract);
    if (requestId !== state.commandRequestId) {
      working.remove();
      return null;
    }
    working.remove();
    state.commandContract = contract;
    state.assistantContract = contract;
    const publicId = contract.model_id ? String(contract.model_id).replace("P5-M", "TWM-METHOD-") : contract.capability;
    if (ownerSurface === "llm") {
      appendAssistantModelReceipt(analysisContext.intent, {
        modelId: publicId,
        tier: contract.selected_model_tier || "Current supported tier",
        parameters: [
          ["Measure", contract.measure_label || String(contract.metric || state.metric).replaceAll("_", " ")],
          ["Scope", contract.grounding?.scope_label || "Whole project"],
          ["Return", `Top ${contract.top_k}`],
        ],
      });
    }
    updateAssistantActivity(
      "decision",
      contract.ready ? `${publicId} · ${contract.selected_model_tier || "supported tier"}` : "More input is needed",
      contract.ready ? "complete" : "failed",
      ownerSurface,
    );
    $("#command-input").value = command;
    if (autoRun && contract.ready) {
      appendAssistantMessage(
        "assistant",
        `Selected ${publicId} automatically at the “${contract.selected_model_tier || "current supported"}” tier. The validated analysis is starting now.`,
        "",
        ownerSurface,
      );
    } else {
      renderCommandReview({ scroll: false });
      appendAssistantContract(contract, ownerSurface);
    }
    return contract;
  } catch (error) {
    working.remove();
    if (requestId !== state.commandRequestId) return null;
    updateAssistantActivity("decision", "Model selection failed", "failed", ownerSurface);
    appendAssistantMessage("assistant", "I could not prepare that analysis.", error.message, ownerSurface);
    return null;
  }
}

function appendAssistantResult(result, extraMetadata = {}, owner = {}) {
  const ownerSurface = "llm";
  const items = result.items || [];
  const summary = resultPlainSummary(result);
  const metric = items.length ? resultBarMetric(items[0]) : null;
  const ranking = metric ? items.slice(0, 8).map((item, index) => {
    const raw = Math.max(0, Number(item[metric.key]) || 0);
    const value = metric.percent ? `${formatNumber(raw * 100, metric.digits)}%` : formatNumber(raw, metric.digits);
    return `<li><span>#${item.rank || index + 1} ${escapeHtml(resultItemName(item))}</span><strong>${escapeHtml(value)}</strong></li>`;
  }).join("") : "";
  const card = document.createElement("article");
  card.className = "assistant-workcard is-result";
  card.innerHTML = `
    <header><span>Analysis complete</span><strong>${escapeHtml(result.model?.name || "Analysis result")}</strong></header>
    <p class="assistant-result-conclusion">${escapeHtml(summary)}</p>
    <div class="assistant-result-metrics">${(result.headline_effects || []).slice(0, 3).map((effect) => `<div><small>${escapeHtml(effect.label)}</small><strong>${escapeHtml(effect.display_value ?? effect.value)}</strong></div>`).join("")}</div>
    ${ranking ? `<ol class="assistant-result-ranking">${ranking}</ol><small class="assistant-result-count">First ${Math.min(8, items.length)} shown here · all ${items.length} are available in the complete visual and table.</small>` : ""}
    <footer><button class="secondary-button ripple" data-assistant-focus-result type="button">Show on map</button><button class="primary-button ripple" data-assistant-view-result type="button">Open full result</button></footer>
  `;
  appendAssistantNode(card, ownerSurface);
  assistantHistoryForSurface(ownerSurface).push({ role: "assistant", content: `Analysis complete: ${summary}` });
  const metadata = {
    kind: "analysis_result",
    run_id: result.run_id || null,
    analysis_type: result.analysis_type || null,
    project_id: currentProjectId(),
    ...extraMetadata,
  };
  registerAssistantChatArtifact(metadata);
  persistAssistantChatMessage("assistant", `Analysis complete: ${summary}`, metadata, {
    surface: ownerSurface,
    chatId: owner.chatId || state.assistantPendingTask?.chatId || state.assistantCompoundWorkflow?.chatId || state.assistantChatId,
  });
}

function appendModelReview(review, surface = state.assistantSurface) {
  if (!review) return;
  const proposal = review.proposal || {};
  const checks = review.deterministic_checks || {};
  const coverage = review.review_coverage || {};
  const issues = coverage.issue_counts || {};
  const card = document.createElement("article");
  card.className = "assistant-workcard model-review-card";
  card.innerHTML = `
    <header><span>Execution gate closed</span><strong>${escapeHtml(proposal.name || "Mathematical model proposal")}</strong></header>
    <div class="model-review-flow" aria-label="Draft, deterministic checks, independent reviews and execution gate">
      <div><b>1</b><small>Draft</small></div><i>→</i><div><b>${escapeHtml(checks.passed_count || 0)}/${escapeHtml(checks.check_count || 0)}</b><small>Checks</small></div><i>→</i><div><b>${escapeHtml(coverage.independent_review_count || 0)}</b><small>Independent reviews</small></div><i>→</i><div class="is-blocked"><b>×</b><small>Run</small></div>
    </div>
    <dl>
      <div><dt>Critical issues</dt><dd>${escapeHtml(issues.critical || 0)}</dd></div>
      <div><dt>Major issues</dt><dd>${escapeHtml(issues.major || 0)}</dd></div>
      <div><dt>Objective</dt><dd>${escapeHtml(proposal.objective?.expression || "Needs review")}</dd></div>
      <div><dt>Status</dt><dd>${escapeHtml(String(review.status || "review required").replaceAll("_", " "))}</dd></div>
    </dl>
    <div class="assistant-work-notes"><p>${escapeHtml(review.claim_boundary || "LLM agreement is not proof.")}</p><p>${escapeHtml(review.next_gate || "Implement deterministic tests before execution.")}</p></div>
  `;
  appendAssistantNode(card, surface);
}

async function runAssistantAnalysis(button) {
  const ownerSurface = "llm";
  const contract = state.assistantContract;
  if (!contract?.ready || state.assistantBusy) return;
  const runId = ++state.assistantAnalysisRequestId;
  state.assistantBusy = true;
  if (button) {
    button.disabled = true;
    button.textContent = "Running selected model…";
  }
  updateAssistantActivity("execute", "Starting the validated model", "active", ownerSurface);
  const working = appendAssistantMessage("assistant", "Running the selected model against the current project…", "", ownerSurface);
  working.classList.add("is-typing");
  try {
    const result = await analyseContract(contract, {
      scroll: false,
      assistant: true,
      onProgress: (progress, message) => updateAssistantActivity(
        "execute",
        `${Math.max(0, Math.min(100, Math.round(Number(progress) || 0)))}% · ${message}`,
        Number(progress) >= 100 ? "complete" : "active",
        ownerSurface,
      ),
    });
    working.remove();
    if (runId !== state.assistantAnalysisRequestId || state.assistantContract !== contract) return;
    if (button) button.textContent = "Completed";
    state.assistantContract = null;
    updateAssistantActivity("execute", "Model run completed", "complete", ownerSurface);
    updateAssistantActivity("verify", "Checking output and saving the run", "active", ownerSurface);
    appendAssistantResult(result);
    if (pendingAssistantTaskMatches("analysis")) state.assistantPendingTask = null;
    updateAssistantActivity("verify", "Result ready · saved to Project Memory", "complete", ownerSurface);
    return result;
  } catch (error) {
    working.remove();
    if (runId !== state.assistantAnalysisRequestId) return;
    updateAssistantActivity("execute", "The model run did not complete", "failed", ownerSurface);
    appendAssistantMessage("assistant", "The selected analysis could not run.", error.message, ownerSurface);
    persistAssistantChatMessage("assistant", `The selected analysis could not run: ${error.message}`, { kind: "error" }, { surface: ownerSurface, chatId: state.assistantPendingTask?.chatId || state.assistantChatId });
    if (button) {
      button.disabled = false;
      button.textContent = "Confirm and run here";
    }
    return null;
  } finally {
    if (runId === state.assistantAnalysisRequestId) {
      state.assistantBusy = false;
      if (!assistantTurnIsWaitingForInput() && !state.assistantCompoundWorkflow) finishAssistantTurn();
    }
  }
}

function renderAssistantResponse(result, surface = state.assistantSurface) {
  const ownerSurface = normaliseAssistantSurface(surface);
  appendAssistantMessage("assistant", result.message, result.provider_notice || "", ownerSurface);
  if (result.model_review) appendModelReview(result.model_review, ownerSurface);
  const llmAllowedActions = new Set(["preview_analysis", "open_scenario_builder", "open_route_builder", "open_recovery_builder", "open_stress_builder", "open_matsim_builder"]);
  const actions = ownerSurface === "llm"
    ? (result.actions || []).filter((action) => llmAllowedActions.has(action.type))
    : (result.actions || []);
  const previewAction = actions.find((action) => action.type === "preview_analysis") || null;
  const visibleActions = ownerSurface === "llm"
    ? actions.filter((action) => action !== previewAction)
    : state.assistantMode === "guide" ? [] : actions.filter((action) => action !== previewAction);
  setAssistantActionsForSurface(ownerSurface, visibleActions);
  replaceAssistantSurfaceHtml(ownerSurface, "actions", visibleActions.map((action, index) => `<button class="assistant-action ripple" data-assistant-action="${index}" type="button"><span>${ownerSurface === "platform" && state.assistantMode === "autopilot" ? "Auto" : action.requires_confirmation ? "Review" : "Go"}</span>${escapeHtml(action.label)}</button>`).join(""));
  const suggestions = [...(result.suggestions || [])];
  if (String(result.message || "").length >= 8000 && !suggestions.some((item) => /continue|go on/i.test(item))) {
    suggestions.push("Continue from your previous answer without repeating it.");
  }
  replaceAssistantSurfaceHtml(ownerSurface, "suggestions", suggestions.slice(0, 4).map((suggestion) => `<button type="button" data-assistant-prompt="${escapeHtml(suggestion)}">${escapeHtml(suggestion)}</button>`).join(""));
  assistantHistoryForSurface(ownerSurface).push({ role: "assistant", content: result.message });
  if (ownerSurface === "llm") {
    persistAssistantChatMessage("assistant", result.message, { kind: "answer", provider: result.provider, model: result.model }, { surface: ownerSurface });
  }
  const providerLabels = { openai: "GPT", anthropic: "Claude", deepseek: "DeepSeek", gemini: "Gemini" };
  if (ownerSurface === state.assistantSurface) {
    $("#assistant-provider").textContent = providerLabels[result.provider]
      ? `${providerLabels[result.provider]} · ${result.model}`
      : result.fallback ? "Local guide · API fallback" : "Grounded local guide";
    if (result.fallback && providerLabels[result.provider]) {
      $("#assistant-provider").textContent = `${providerLabels[result.provider]} · request failed`;
    }
  }
  return previewAction;
}

async function executeAutopilotResponse(result, previewAction, requestId, surface = "platform") {
  const ownerSurface = normaliseAssistantSurface(surface);
  if (ownerSurface !== "platform" || state.assistantMode !== "autopilot" || requestId !== state.platformAssistantMessageRequestId) return;
  const reversible = (result.actions || []).filter((action) => action !== previewAction).slice(0, 3);
  for (const action of reversible) {
    if (requestId !== state.platformAssistantMessageRequestId || state.assistantMode !== "autopilot") return;
    await executeAssistantAction(action);
  }
  if (previewAction?.value && requestId === state.platformAssistantMessageRequestId) {
    $("#command-input").value = previewAction.value;
    invalidateAnalysisReview("Platform Assistant prepared a fresh question for manual review.");
    interpretCurrentCommand();
    appendAssistantMessage(
      "assistant",
      "I prepared this analysis in the main workspace for review. Platform Assistant does not execute LLM Copilot tasks or consume its pending workflow.",
      "",
      ownerSurface,
    );
  }
}

const LLM_INPUT_BUILDERS = new Set([
  "open_scenario_builder",
  "open_route_builder",
  "open_recovery_builder",
  "open_stress_builder",
  "open_matsim_builder",
]);

function requestedCountFromText(text, fallback = 10) {
  const match = String(text || "").match(/(?:top|first|return|plan|rank|find|show)\s*(\d+)/i);
  const value = Number(match?.[1]);
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function scenarioEffectFromRequest(text) {
  const request = String(text || "");
  if (/slow|speed\s*(?:reduction|loss|drop)|reduce\s+(?:the\s+)?speed|降速|速度(?:下降|降低|减少)/i.test(request)) return "speed_reduction";
  if (/longer|travel[-\s]?time\s*(?:increase|delay)|add\s+(?:travel\s+)?time|增加(?:旅行|通行)?时间|旅行时间增加|延误/i.test(request)) return "travel_time_increase";
  if (/clos(?:e|ed|ure)|block(?:ed)?|unavailable|remove\s+(?:the\s+)?road|关闭|封闭|封路|中断/i.test(request)) return "closed";
  return null;
}

function assistantEffectLabel(effect, magnitude = null) {
  const hasMagnitude = hasAssistantMagnitude(magnitude);
  if (effect === "speed_reduction") return hasMagnitude ? `${Math.round(Number(magnitude) * 100)}% speed reduction` : "Speed reduction - amount needed";
  if (effect === "travel_time_increase") return hasMagnitude ? `${Math.round(Number(magnitude) * 100)}% longer travel time` : "Travel-time increase - amount needed";
  return "road closure";
}

function requestedMagnitudeFromText(text, fallback = null) {
  const match = String(text || "").match(/(\d+(?:\.\d+)?)\s*%/);
  const value = Number(match?.[1]);
  return Number.isFinite(value) && value > 0 ? value / 100 : fallback;
}

function requestedAgentCountFromText(text) {
  const request = String(text || "");
  const match = request.match(/(?:agents?|probes?|代理)\D{0,12}(50|100|150|200)\b|\b(50|100|150|200)\s*(?:agents?|probes?|代理)/i);
  const value = Number(match?.[1] || match?.[2]);
  return [50, 100, 150, 200].includes(value) ? value : null;
}

const ASSISTANT_MATSIM_UPLOADED_DEMAND_LIMIT = 4000;

function assistantMatsimUsesUploadedDemand(readiness = state.matsimReadiness) {
  return Boolean(readiness?.can_run && readiness?.tier === "demand_calibrated" && Number(readiness?.od_rows || 0) > 0);
}

function nextAssistantMatsimAgentPhase(task) {
  return assistantMatsimUsesUploadedDemand() || task?.agentCount ? "ready" : "agent_count";
}

function explicitAssistantWorkflow(text) {
  const request = String(text || "").trim();
  if (!request) return null;
  if (/\b(?:matsim|agent[-\s]?based\s+simulation|traffic\s+microsimulation)\b|代理仿真|交通仿真/i.test(request)) {
    return { type: "matsim", key: "matsim" };
  }
  if (/\b(?:route|routes|routing|path|paths|detour|detours|origin|destination)\b|路径|路线|绕行|起点|终点|目的地/i.test(request)) {
    return { type: "route", key: "route" };
  }
  if (/\b(?:recover|recovery|restore|restoration|repair|reopen).{0,30}(?:priorit|rank|order|first)|\bprioriti[sz]e.{0,30}(?:recover|restore|repair|reopen)|恢复优先|修复优先|恢复排序/i.test(request)) {
    return { type: "recovery", key: "recovery" };
  }
  if (/\b(?:stress[-\s]?test|network\s+resilience|progressive\s+(?:failure|removal)|cumulative\s+(?:failure|removal)|resilience\s+test)\b|网络韧性|压力测试|累计失效/i.test(request)) {
    return { type: "stress", key: "stress" };
  }
  if (/\b(?:build|create|define|make|run)\b.{0,35}\b(?:disruption|closure|slowdown)\s+scenario\b|\bscenario\s+builder\b|构建情景|创建情景|中断情景/i.test(request)) {
    return { type: "scenario", key: "scenario" };
  }
  if (/\b(?:critical|key|important)\s+(?:areas?|regions?|zones?)\b|\b(?:area|geographic)\s+hotspots?\b|关键区域|重点区域|热点区域/i.test(request)) {
    return { type: "analysis", analysisType: "spatial_hotspots", key: "analysis:spatial_hotspots" };
  }
  if (/\b(?:critical|key|important)\s+(?:roads?|links?|edges?)\b|\bnetwork\s+bottlenecks?\b|关键边|关键道路|关键路段|重要道路/i.test(request)) {
    return { type: "analysis", analysisType: "critical_components", key: "analysis:critical_components" };
  }
  return null;
}

function assistantRequestReferencesPriorWork(text) {
  return /\b(?:those|these|them|that\s+(?:result|scenario|selection)|previous|prior|above|same|continue|reuse|based\s+on|using\s+(?:the\s+)?(?:result|roads?|links?|areas?|scenario)|from\s+(?:the\s+)?(?:latest|previous|prior|current)\s+(?:result|roads?|links?|areas?|scenario)|current\s+map\s+selection|selected\s+(?:roads?|links?|areas?|points?|endpoints?))\b|基于(?:上一步|上述|刚才|这些|那些|该结果)|使用(?:上一步|上述|刚才|这些|那些|当前选择)|继续|沿用|复用/i.test(String(text || ""));
}

function workflowKeyFromAction(action, sourceText = "") {
  const builderKeys = {
    open_route_builder: "route",
    open_recovery_builder: "recovery",
    open_stress_builder: "stress",
    open_scenario_builder: "scenario",
    open_matsim_builder: "matsim",
  };
  if (builderKeys[action?.type]) return builderKeys[action.type];
  if (action?.type === "preview_analysis") {
    return explicitAssistantWorkflow(action.value || sourceText)?.key || "analysis";
  }
  return explicitAssistantWorkflow(sourceText)?.key || null;
}

function resetAssistantVisualWorkspaceForIndependentTask() {
  if (!state.project) return;
  if (state.routeBuilding) closeRouteBuilder({ restoreSelection: false, restoreScenario: false });
  if (state.recoveryBuilding) closeRecoveryBuilder({ restoreSelection: false });
  if (state.stressBuilding) closeStressBuilder();
  if (state.scenarioBuilding) closeScenarioBuilder({ restore: false });
  stopMatsimPlayback({ resetTransition: true });
  state.matsimFrames = [];
  state.matsimFrameIndex = 0;
  state.matsimResult = null;
  state.map?.setMatsimFeatures(null);
  state.selectedKeys.clear();
  state.keptKeys.clear();
  state.focusedKey = null;
  state.lastFeature = null;
  state.bbox = null;
  state.analysisScopeMode = "whole";
  state.completedResults.clear();
  state.routeOrigin = null;
  state.routeDestination = null;
  state.routeDraft.clear();
  state.routeDraftFeatures.clear();
  state.recoveryDraft.clear();
  state.recoveryDraftFeatures.clear();
  state.recoveryAreas = [];
  state.scenarioDraft.clear();
  state.scenarioDraftFeatures.clear();
  state.activeScenarioId = null;
  state.map?.setSelectedKeys([]);
  state.map?.clearArea();
  state.map?.setMode("point");
  state.map?.setScenarioFeatures(null);
  state.map?.setScenarioDraftFeatures(null);
  clearAnalysisOutput();
  $("#analysis-review")?.classList.add("is-hidden");
  $("#entity-inspector")?.classList.add("is-hidden");
  $("#scenario-impact-panel")?.classList.add("is-hidden");
  $("#command-input").value = "";
  renderScenarioControls();
  renderScenarioImpact(null);
  renderSelectionSummary();
  renderAnalysisScope();
  updateReadiness();
  setMapBanner(
    state.mapView === "links" ? "Mapped link view" : "Historical data",
    `${mappedObservationLabel(state.mapCollection?.features?.length || 0)} shown · ${currentTimeLabel()}`,
  );
}

function prepareAssistantWorkflowTransition(action, sourceText = "") {
  const nextKey = workflowKeyFromAction(action, sourceText);
  const previousKey = state.assistantPendingTask
    ? (state.assistantPendingTask.workflowKey || state.assistantPendingTask.type)
    : state.assistantLastWorkflowKey;
  const linked = assistantRequestReferencesPriorWork(sourceText);
  const visiblePriorWork = Boolean(
    state.currentResult
    || state.activeScenarioId
    || state.matsimResult
    || state.selectedKeys.size
    || state.bbox
    || state.routeBuilding
    || state.recoveryBuilding
    || state.scenarioBuilding
    || state.stressBuilding
  );
  if (nextKey && !linked && (
    (previousKey && nextKey !== previousKey)
    || (!previousKey && visiblePriorWork)
  )) {
    resetAssistantVisualWorkspaceForIndependentTask();
  }
  if (nextKey) state.assistantLastWorkflowKey = nextKey;
  return nextKey;
}

function analysisRequestNeedsCount(text) {
  const request = String(text || "");
  const workflow = explicitAssistantWorkflow(request);
  return ["critical_components", "spatial_hotspots"].includes(workflow?.analysisType)
    && requestedCountFromText(request, null) === null;
}

function recoveryRequestReferencesPriorResult(text) {
  return /\b(?:latest|previous|prior|those|these|identified|found)\b|critical[-\s]?(?:road|link|area)|from\s+(?:the\s+)?(?:latest|previous|current)\s+(?:result|roads?|links?|areas?)/i.test(String(text || ""));
}

function effectNeedsMagnitude(effect) {
  return effect === "speed_reduction" || effect === "travel_time_increase";
}

function hasAssistantMagnitude(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value));
}

function nextRouteInputPhase(task) {
  if (!task?.requestedCount) return "count";
  if (!state.routeOrigin) return "origin";
  if (!state.routeDestination) return "destination";
  if ($("#route-scenario")?.value !== "__quick__") return "ready";
  if (!task.effect) return "effect";
  if (effectNeedsMagnitude(task.effect) && !hasAssistantMagnitude(task.magnitude)) return "magnitude";
  if (!state.routeDraft.size) return "disruption";
  return "ready";
}

function nextScenarioInputPhase(task, { matsim = false } = {}) {
  if (!task?.effect) return "effect";
  if (effectNeedsMagnitude(task.effect) && !hasAssistantMagnitude(task.magnitude)) return "magnitude";
  if (!state.scenarioDraft.size) return "location";
  if (matsim) return nextAssistantMatsimAgentPhase(task);
  return "ready";
}

function magnitudeWorkflowRange(prefix, effect, magnitude = null) {
  const maximum = effect === "speed_reduction" ? 95 : 300;
  const value = Math.max(5, Math.min(maximum, Math.round((hasAssistantMagnitude(magnitude) ? Number(magnitude) : 0.5) * 100 / 5) * 5));
  return {
    type: prefix,
    label: effect === "speed_reduction" ? "Speed reduction" : "Travel-time increase",
    min: 5,
    max: maximum,
    step: 5,
    value,
  };
}

function pendingAssistantMapStep(step) {
  const pending = state.assistantPendingTask;
  return Boolean(pendingAssistantTaskMatches(pending?.type) && pending.mapStep === step);
}

function retireAssistantWorkflowCards() {
  assistantSurfaceContainer("llm")?.querySelectorAll("[data-assistant-workflow-card='active']").forEach((card) => {
    card.dataset.assistantWorkflowCard = "complete";
    card.querySelectorAll("button, input").forEach((control) => { control.disabled = true; });
  });
}

function appendAssistantWorkflowCard({ kicker = "Next step", title, detail = "", facts = [], range = null, countInput = null, actions = [] }) {
  retireAssistantWorkflowCards();
  const card = document.createElement("article");
  card.className = "assistant-workcard assistant-workflow-card is-ready";
  card.dataset.assistantWorkflowCard = "active";
  card.innerHTML = `
    <header><span>${escapeHtml(kicker)}</span><strong>${escapeHtml(title)}</strong></header>
    ${detail ? `<p class="assistant-workflow-detail">${escapeHtml(detail)}</p>` : ""}
    ${facts.length ? `<dl>${facts.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
    ${range ? `
      <div class="assistant-workflow-range">
        <label>
          <span>${escapeHtml(range.label)}</span>
          <output data-llm-magnitude-output>${escapeHtml(`${range.value}%`)}</output>
          <input
            type="range"
            min="${escapeHtml(range.min)}"
            max="${escapeHtml(range.max)}"
            step="${escapeHtml(range.step)}"
            value="${escapeHtml(range.value)}"
            data-llm-magnitude-slider="${escapeHtml(range.type)}"
            aria-label="${escapeHtml(range.label)}"
          />
        </label>
        <div aria-hidden="true"><span>${escapeHtml(`${range.min}%`)}</span><span>50%</span><span>${escapeHtml(`${range.max}%`)}</span></div>
      </div>
    ` : ""}
    ${countInput ? `
      <div class="assistant-workflow-number">
        <label>
          <span>${escapeHtml(countInput.label || "Enter another number")}</span>
          <input
            type="number"
            min="${escapeHtml(countInput.min ?? 1)}"
            ${Number.isFinite(Number(countInput.max)) ? `max="${escapeHtml(countInput.max)}"` : ""}
            step="${escapeHtml(countInput.step ?? 1)}"
            inputmode="numeric"
            placeholder="${escapeHtml(countInput.placeholder || "Enter a whole number")}"
            data-llm-count-input="${escapeHtml(countInput.actionPrefix)}"
            aria-label="${escapeHtml(countInput.label || "Enter another number")}"
          />
        </label>
        <button class="secondary-button ripple" data-llm-count-submit="${escapeHtml(countInput.actionPrefix)}" type="button">Use this number</button>
        ${countInput.note ? `<small>${escapeHtml(countInput.note)}</small>` : ""}
      </div>
    ` : ""}
    <footer>${actions.map((action) => `<button class="${action.primary ? "primary-button" : "secondary-button"} ripple" data-llm-workflow-action="${escapeHtml(action.id)}"${action.magnitudeConfirm ? " data-llm-magnitude-confirm" : ""} type="button">${escapeHtml(action.label)}</button>`).join("")}</footer>
  `;
  return appendAssistantNode(card, "llm");
}

function assistantPendingRunAction(task) {
  if (!task || task.phase !== "ready") return null;
  const actions = {
    route: ["run_route", "Run route analysis"],
    recovery: ["run_recovery", "Run recovery ranking"],
    scenario: ["run_scenario", "Run disruption scenario"],
    matsim: ["run_matsim", "Run MATSim"],
    analysis: ["run_analysis", "Run analysis"],
    stress: ["run_stress", "Run resilience test"],
  };
  return actions[task.type] || null;
}

function syncAssistantPendingRunAction(task = state.assistantPendingTask) {
  // While Copilot owns an interactive workflow, remove stale navigation actions
  // that would send the user back to the manual platform controls.
  setAssistantActionsForSurface("llm", []);
  const runAction = assistantPendingRunAction(task);
  replaceAssistantSurfaceHtml("llm", "actions", runAction ? `
    <button class="assistant-action ripple" data-llm-workflow-action="${escapeHtml(runAction[0])}" data-llm-pending-run type="button">
      <span>Run in Copilot</span>${escapeHtml(runAction[1])}
    </button>
  ` : "");
}

function showAssistantPendingTaskStep() {
  const task = state.assistantPendingTask;
  if (!task || !pendingAssistantTaskMatches(task.type)) return;
  syncAssistantPendingRunAction(task);
  if (task.type === "route") {
    const facts = [
      ["Routes requested", String(task.requestedCount || 2)],
      ...(state.routeOrigin ? [["Start", `Road ${state.routeOrigin.entityId}`]] : []),
      ...(state.routeDestination ? [["Destination", `Road ${state.routeDestination.entityId}`]] : []),
      ...(task.effect ? [["Disruption", assistantEffectLabel(task.effect, task.magnitude)]] : []),
      ...(state.routeDraft.size ? [["Affected roads", formatNumber(state.routeDraft.size)]] : []),
    ];
    if (task.phase === "count") return appendAssistantWorkflowCard({
      kicker: "Route input 1",
      title: "How many route options should be returned?",
      detail: "Use a shortcut or enter any whole number from 1 to 50.",
      facts,
      countInput: {
        actionPrefix: "route_count",
        label: "Custom route count",
        min: 1,
        max: 50,
        placeholder: "1-50",
        note: "The model returns only distinct feasible routes and explains when fewer are available.",
      },
      actions: [
        { id: "route_count_2", label: "2 routes", primary: true },
        { id: "route_count_3", label: "3 routes" },
        { id: "route_count_5", label: "5 routes" },
      ],
    });
    if (task.phase === "origin") return appendAssistantWorkflowCard({
      kicker: "Route input 1",
      title: "Choose the start point",
      detail: "The map will open only for this selection and then return here automatically.",
      facts,
      actions: [{ id: "route_origin", label: "Select start on map", primary: true }],
    });
    if (task.phase === "destination") return appendAssistantWorkflowCard({
      kicker: "Route input 2",
      title: "Start confirmed — choose the destination",
      detail: "After one routable destination is selected, Copilot will return here again.",
      facts,
      actions: [{ id: "route_destination", label: "Select destination on map", primary: true }],
    });
    if (task.phase === "effect") return appendAssistantWorkflowCard({
      kicker: "Route input 3",
      title: "What disruption should the routes avoid?",
      detail: "Choose the practical road condition; the existing platform model remains unchanged.",
      facts,
      actions: [
        { id: "route_effect_closed", label: "Road closed" },
        { id: "route_effect_speed", label: "Reduce speed" },
        { id: "route_effect_time", label: "Add travel time" },
      ],
    });
    if (task.phase === "magnitude") {
      const range = magnitudeWorkflowRange("route", task.effect, task.magnitude);
      return appendAssistantWorkflowCard({
        kicker: "Route disruption amount",
        title: task.effect === "speed_reduction" ? "How much should speed fall?" : "How much should travel time increase?",
        detail: "Drag the slider to the required percentage. The value is synchronised with the platform control.",
        facts,
        range,
        actions: [{ id: `route_magnitude_${range.value}`, label: `Confirm ${range.value}%`, primary: true, magnitudeConfirm: true }],
      });
    }
    if (task.phase === "disruption") return appendAssistantWorkflowCard({
      kicker: "Route input 4",
      title: "Choose where the disruption applies",
      detail: "Select one affected road or draw an affected area. You can add more after returning.",
      facts,
      actions: [
        { id: "route_road", label: "Select affected road", primary: true },
        { id: "route_area", label: "Draw affected area" },
      ],
    });
    return appendAssistantWorkflowCard({
      kicker: "Ready to run",
      title: "Route inputs confirmed",
      detail: "Run from Copilot. The platform map and saved result will update from the same calculation.",
      facts,
      actions: [
        { id: "route_road", label: "Add another road" },
        { id: "route_area", label: "Add another area" },
        { id: "run_route", label: "Run route analysis", primary: true },
      ],
    });
  }
  if (task.type === "recovery") {
    const payload = state.recoveryBuilding ? recoveryPayloadFromSource() : { selectedKeys: [], areaCandidates: [] };
    const facts = [
      ...(task.requestedCount ? [["Priorities requested", String(task.requestedCount)]] : []),
      ...(payload.selectedKeys.length ? [["Disrupted roads", formatNumber(payload.selectedKeys.length)]] : []),
      ...(payload.areaCandidates.length ? [["Areas", formatNumber(payload.areaCandidates.length)]] : []),
    ];
    if (task.phase === "count") return appendAssistantWorkflowCard({
      kicker: "Recovery input 1",
      title: "How many recovery priorities should be returned?",
      detail: "Use a shortcut or enter any positive whole number.",
      facts,
      countInput: {
        actionPrefix: "recovery_count",
        label: "Custom priority count",
        min: 1,
        placeholder: "For example, 100",
        note: "If fewer eligible roads or areas exist, every available result is returned.",
      },
      actions: [
        { id: "recovery_count_10", label: "Top 10", primary: true },
        { id: "recovery_count_20", label: "Top 20" },
        { id: "recovery_count_50", label: "Top 50" },
      ],
    });
    if (task.phase !== "ready") return appendAssistantWorkflowCard({
      kicker: "Recovery input",
      title: "Choose what is disrupted",
      detail: "The selection will return to this Chat before the recovery ranking runs.",
      facts,
      actions: [
        ...(latestAssistantArtifact("critical_components") ? [{ id: "recovery_use_critical_roads", label: "Use latest critical roads" }] : []),
        ...(latestAssistantArtifact("spatial_hotspots") ? [{ id: "recovery_use_critical_areas", label: "Use latest critical areas" }] : []),
        ...(activeScenario() ? [{ id: "recovery_use_active_scenario", label: "Use active scenario" }] : []),
        { id: "recovery_road", label: "Select disrupted road", primary: true },
        { id: "recovery_area", label: "Draw disrupted area" },
      ],
    });
    return appendAssistantWorkflowCard({
      kicker: "Ready to run",
      title: "Recovery candidates confirmed",
      detail: "Copilot will call the existing recovery-priority model and return its ranked map and table here.",
      facts,
      actions: [
        { id: "recovery_road", label: "Add another road" },
        { id: "recovery_area", label: "Add another area" },
        { id: "run_recovery", label: "Run recovery ranking", primary: true },
      ],
    });
  }
  if (task.type === "scenario") {
    const facts = [
      ...(task.effect ? [["Road change", assistantEffectLabel(task.effect, task.magnitude)]] : []),
      ...(state.scenarioDraft.size ? [["Affected roads", formatNumber(state.scenarioDraft.size)]] : []),
    ];
    if (task.phase === "effect") return appendAssistantWorkflowCard({
      kicker: "Scenario input 1",
      title: "What happens to the affected roads?",
      detail: "Choose one clear disruption condition.",
      facts,
      actions: [
        { id: "scenario_effect_closed", label: "Closed" },
        { id: "scenario_effect_speed", label: "Slower" },
        { id: "scenario_effect_time", label: "Longer time" },
      ],
    });
    if (task.phase === "magnitude") {
      const range = magnitudeWorkflowRange("scenario", task.effect, task.magnitude);
      return appendAssistantWorkflowCard({
        kicker: "Scenario input 2",
        title: task.effect === "speed_reduction" ? "How much should speed fall?" : "How much should travel time increase?",
        detail: "Drag the slider to the required percentage. The value is synchronised with the platform control.",
        facts,
        range,
        actions: [{ id: `scenario_magnitude_${range.value}`, label: `Confirm ${range.value}%`, primary: true, magnitudeConfirm: true }],
      });
    }
    if (task.phase !== "ready") return appendAssistantWorkflowCard({
      kicker: "Scenario input 2",
      title: "Choose the affected location",
      detail: task.requestedCount
        ? `Select roads one at a time. Copilot will return after every selection until ${task.requestedCount} distinct roads are confirmed.`
        : "The map will return to Copilot immediately after the road or area is captured.",
      facts,
      actions: task.requestedCount
        ? [{ id: "scenario_road", label: `Select road ${Math.min(state.scenarioDraft.size + 1, task.requestedCount)} of ${task.requestedCount}`, primary: true }]
        : [
            { id: "scenario_road", label: "Select affected road", primary: true },
            { id: "scenario_area", label: "Draw affected area" },
          ],
    });
    return appendAssistantWorkflowCard({
      kicker: "Ready to run",
      title: "Disruption scenario confirmed",
      detail: "Run the existing deterministic scenario model from Copilot and return the impact result here.",
      facts,
      actions: [
        { id: "scenario_road", label: "Add another road" },
        { id: "scenario_area", label: "Add another area" },
        { id: "run_scenario", label: "Run disruption scenario", primary: true },
      ],
    });
  }
  if (task.type === "matsim") {
    if (task.phase === "agent_count" && assistantMatsimUsesUploadedDemand()) task.phase = "ready";
    const scenario = task.scenarioId
      ? state.scenarios.find((item) => item.scenario_id === task.scenarioId)
      : null;
    const facts = [
      ...(scenario ? [["Scenario", scenario.name || "Saved disruption"]] : []),
      ...(task.effect ? [["Road change", assistantEffectLabel(task.effect, task.magnitude)]] : []),
      ...(state.scenarioDraft.size ? [["Affected roads", formatNumber(state.scenarioDraft.size)]] : []),
      ...(assistantMatsimUsesUploadedDemand()
        ? [["Agent source", "Automatically generated from uploaded OD demand"]]
        : task.agentCount ? [["Agents", formatNumber(task.agentCount)]] : []),
      ...(state.matsimReadiness?.label ? [["Simulation tier", state.matsimReadiness.label]] : []),
    ];
    if (state.matsimReadiness && !state.matsimReadiness.can_run) return appendAssistantWorkflowCard({
      kicker: "MATSim unavailable",
      title: "This upload cannot support a MATSim run",
      detail: state.matsimReadiness.explanation || "A routable From/To network with positive impedance is required.",
      facts,
      actions: [],
    });
    if (task.phase === "scenario_source") return appendAssistantWorkflowCard({
      kicker: "MATSim input 1",
      title: "Which disruption should MATSim simulate?",
      detail: "Use the active saved scenario or build a new disruption without leaving this Chat workflow.",
      facts,
      actions: [
        ...(activeScenario() ? [{ id: "matsim_use_active", label: `Use ${activeScenario().name || "active scenario"}`, primary: true }] : []),
        { id: "matsim_new_scenario", label: "Build a new disruption", primary: !activeScenario() },
      ],
    });
    if (task.phase === "effect") return appendAssistantWorkflowCard({
      kicker: "MATSim disruption",
      title: "What happens to the affected roads?",
      detail: "Choose the road condition that the existing MATSim adapter will compare with baseline.",
      facts,
      actions: [
        { id: "matsim_effect_closed", label: "Closed" },
        { id: "matsim_effect_speed", label: "Slower" },
        { id: "matsim_effect_time", label: "Longer time" },
      ],
    });
    if (task.phase === "magnitude") {
      const range = magnitudeWorkflowRange("matsim", task.effect, task.magnitude);
      return appendAssistantWorkflowCard({
        kicker: "MATSim disruption amount",
        title: task.effect === "speed_reduction" ? "How much should speed fall?" : "How much should travel time increase?",
        detail: "Drag the slider to the required percentage. The value is synchronised with the platform control.",
        facts,
        range,
        actions: [{ id: `matsim_magnitude_${range.value}`, label: `Confirm ${range.value}%`, primary: true, magnitudeConfirm: true }],
      });
    }
    if (task.phase === "location") return appendAssistantWorkflowCard({
      kicker: "MATSim affected location",
      title: "Where does the disruption apply?",
      detail: "The platform map opens for one selection and then returns here automatically.",
      facts,
      actions: [
        { id: "matsim_road", label: "Select affected road", primary: true },
        { id: "matsim_area", label: "Draw affected area" },
      ],
    });
    if (task.phase === "agent_count") return appendAssistantWorkflowCard({
      kicker: "MATSim agent group",
      title: "How many agents should be simulated?",
      detail: "Choose the reproducible exploratory probe group. It does not represent observed demand.",
      facts,
      actions: [50, 100, 150, 200].map((value) => ({
        id: `matsim_agents_${value}`,
        label: `${value} agents`,
        primary: value === 100,
      })),
    });
    return appendAssistantWorkflowCard({
      kicker: "Ready to run",
      title: "MATSim inputs confirmed",
      detail: assistantMatsimUsesUploadedDemand()
        ? "Uploaded OD demand will determine the simulated agents automatically. Run from Copilot to update the map, report and Project Memory."
        : "Run from Copilot. The existing scenario and MATSim worker will update the platform map, report and Project Memory.",
      facts,
      actions: [{ id: "run_matsim", label: "Run MATSim", primary: true }],
    });
  }
  if (task.type === "analysis") {
    if (task.phase === "count") {
      const isCriticalAreaRequest = task.analysisType === "spatial_hotspots"
        || task.workflowKey === "analysis:spatial_hotspots"
        || explicitAssistantWorkflow(task.sourceText)?.analysisType === "spatial_hotspots";
      return appendAssistantWorkflowCard({
      kicker: "Analysis input",
      title: isCriticalAreaRequest
        ? "How many critical areas should be returned?"
        : "How many critical roads should be returned?",
      detail: "Use a shortcut or enter any positive whole number.",
      facts: [],
      countInput: {
        actionPrefix: "analysis_count",
        label: isCriticalAreaRequest ? "Custom area count" : "Custom road count",
        min: 1,
        placeholder: "For example, 100",
        note: "If the project has fewer eligible results, the complete available set is returned.",
      },
      actions: [
        { id: "analysis_count_10", label: "Top 10", primary: true },
        { id: "analysis_count_20", label: "Top 20" },
        { id: "analysis_count_50", label: "Top 50" },
      ],
    });
    }
    if (task.phase !== "ready") return appendAssistantWorkflowCard({
      kicker: "Map scope needed",
      title: "Choose the requested calculation scope",
      detail: "One map selection is collected and returned to this Chat before the model runs.",
      actions: [
        { id: "analysis_point", label: "Select map item", primary: true },
        { id: "analysis_area", label: "Draw map area" },
      ],
    });
    return appendAssistantWorkflowCard({
      kicker: "Ready to run",
      title: "Calculation scope confirmed",
      detail: "Run the original analysis request from Copilot using only this explicit map selection.",
      facts: [
        ["Scope", analysisMapSelectionLabel()],
        ...(task.requestedCount ? [["Return", `Top ${task.requestedCount}`]] : []),
      ],
      actions: [{ id: "run_analysis", label: "Run analysis", primary: true }],
    });
  }
  if (task.type === "stress") return appendAssistantWorkflowCard({
    kicker: task.phase === "stages" ? "Resilience input" : "Ready to run",
    title: task.phase === "stages" ? "How many cumulative road failures should be tested?" : "Network resilience test prepared",
    detail: task.phase === "stages"
      ? "Use a shortcut or enter any positive whole number."
      : "Copilot will reuse a valid critical-road order or deterministically recompute the requested stages.",
    facts: task.requestedCount ? [["Stages", String(task.requestedCount)]] : [],
    countInput: task.phase === "stages" ? {
      actionPrefix: "stress_count",
      label: "Custom failure-stage count",
      min: 1,
      placeholder: "For example, 75",
      note: "The run stops at the number of eligible roads when that is smaller.",
    } : null,
    actions: task.phase === "stages" ? [
      { id: "stress_count_10", label: "10 stages" },
      { id: "stress_count_25", label: "25 stages", primary: true },
      { id: "stress_count_50", label: "50 stages" },
      { id: "stress_count_100", label: "100 stages" },
    ] : [{ id: "run_stress", label: "Run resilience test", primary: true }],
  });
}

function configureScenarioBuilderFromRequest(task) {
  if (!task) return;
  const effect = task.effect || "closed";
  setScenarioEffect(effect);
  if (effect !== "closed" && hasAssistantMagnitude(task.magnitude)) {
    const percentage = Math.round(Math.max(0.05, Math.min(effect === "speed_reduction" ? 0.95 : 3, Number(task.magnitude))) * 100);
    $("#scenario-magnitude").value = String(percentage);
    $("#scenario-magnitude-output").textContent = `${percentage}%`;
  }
}

function applyRouteTaskEffect(task) {
  if (!task?.effect) return;
  setRouteEffect(task.effect);
  if (task.effect !== "closed" && hasAssistantMagnitude(task.magnitude)) {
    const percentage = Math.round(Math.max(0.05, Math.min(task.effect === "speed_reduction" ? 0.95 : 3, Number(task.magnitude))) * 100);
    $("#route-magnitude").value = String(percentage);
    $("#route-magnitude-output").textContent = `${percentage}%`;
  }
  renderRouteBuilder();
}

async function beginAssistantMapInput(step) {
  const task = state.assistantPendingTask;
  if (!task || !pendingAssistantTaskMatches(task.type)) return;
  task.mapStep = step;
  retireAssistantWorkflowCards();
  setAssistantOpen(false);
  if (step.startsWith("route_")) {
    if (!state.routeBuilding) {
      await openRouteBuilder();
      configureRouteBuilderFromRequest(task.sourceText);
      applyRouteTaskEffect(task);
    }
    if (step === "route_origin") setRoutePickRole("origin");
    else if (step === "route_destination") setRoutePickRole("destination");
    else if (step === "route_road") {
      state.routePickRole = null;
      state.routeDisruptionMode = "roads";
      state.map?.setMode("point");
      renderRouteDraft();
      setMapBanner("Select one affected road", "Copilot will return after this road is captured.", true);
    } else if (step === "route_area") {
      state.routePickRole = null;
      state.routeDisruptionMode = "area";
      state.map?.setMode("area");
      renderRouteDraft();
      setMapBanner("Draw the affected area", "Press, drag and release; Copilot will return with the captured road count.", true);
    }
    return;
  }
  if (step.startsWith("recovery_")) {
    if (!state.recoveryBuilding) {
      openRecoveryBuilder();
      $("#recovery-source").value = "manual";
      $("#recovery-top-k").value = String(task.requestedCount || 10);
    }
    state.recoveryMode = step === "recovery_area" ? "area" : "roads";
    state.map?.setMode(step === "recovery_area" ? "area" : "point");
    renderRecoveryBuilder();
    setMapBanner(
      step === "recovery_area" ? "Draw one disrupted area" : "Select one disrupted road",
      "Copilot will return immediately after this recovery input is captured.",
      true,
    );
    return;
  }
  if (step.startsWith("scenario_") || step.startsWith("matsim_")) {
    if (!state.scenarioBuilding) openScenarioBuilder();
    if (step.startsWith("matsim_")) setScenarioEngine("matsim");
    configureScenarioBuilderFromRequest(task);
    if (step === "scenario_area" || step === "matsim_area") {
      state.map?.setMode("area");
      $("#scenario-draw-area").classList.add("is-active");
      $("#scenario-pick-roads").classList.remove("is-active");
      setMapBanner("Draw one affected area", "Press, drag and release; Copilot will return with the captured road count.", true);
    } else {
      state.map?.setMode("point");
      $("#scenario-pick-roads").classList.add("is-active");
      $("#scenario-draw-area").classList.remove("is-active");
      setMapBanner("Select one affected road", "Copilot will return immediately after this road is captured.", true);
    }
    return;
  }
  if (step === "analysis_point" || step === "analysis_area") {
    setAnalysisScopeMode("selection", { invalidate: false });
    state.map?.setMode(step === "analysis_area" ? "area" : "point");
    setMapBanner(
      step === "analysis_area" ? "Draw the calculation area" : "Select one map item",
      "Copilot will return immediately after this scope is captured.",
      true,
    );
  }
}

function returnAssistantMapInput(step, confirmation) {
  const task = state.assistantPendingTask;
  if (!task || task.mapStep !== step || !pendingAssistantTaskMatches(task.type)) return false;
  task.mapStep = null;
  if (step === "route_origin" || step === "route_destination") task.phase = nextRouteInputPhase(task);
  else if (step === "matsim_road" || step === "matsim_area") task.phase = nextScenarioInputPhase(task, { matsim: true });
  else if (step === "scenario_road" || step === "scenario_area") task.phase = task.requestedCount && state.scenarioDraft.size < task.requestedCount ? "location" : "ready";
  else if (step === "route_road" || step === "route_area" || step === "recovery_road" || step === "recovery_area" || step === "analysis_point" || step === "analysis_area") task.phase = "ready";
  updateAssistantActivity(`input_${step}`, confirmation, "complete");
  appendAssistantMessage("assistant", confirmation);
  persistAssistantChatMessage("assistant", confirmation, {
    kind: "workflow_input",
    task_type: task.type,
    step,
    project_id: currentProjectId(),
  });
  showAssistantPendingTaskStep();
  setAssistantOpen(true, "llm");
  continueAssistantCompoundTaskIfReady(task);
  window.setTimeout(() => {
    if (!pendingAssistantTaskMatches(task.type) || task.mapStep) return;
    setAssistantOpen(true, "llm");
    const messages = $("#assistant-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  }, 80);
  return true;
}

async function executeAssistantWorkflowAction(actionId) {
  const task = state.assistantPendingTask;
  if (!task || !pendingAssistantTaskMatches(task.type)) return;
  if (["route_origin", "route_destination", "route_road", "route_area", "recovery_road", "recovery_area", "scenario_road", "scenario_area", "matsim_road", "matsim_area", "analysis_point", "analysis_area"].includes(actionId)) {
    await beginAssistantMapInput(actionId);
    return;
  }
  const effectActions = {
    route_effect_closed: ["route", "closed"],
    route_effect_speed: ["route", "speed_reduction"],
    route_effect_time: ["route", "travel_time_increase"],
    scenario_effect_closed: ["scenario", "closed"],
    scenario_effect_speed: ["scenario", "speed_reduction"],
    scenario_effect_time: ["scenario", "travel_time_increase"],
    matsim_effect_closed: ["matsim", "closed"],
    matsim_effect_speed: ["matsim", "speed_reduction"],
    matsim_effect_time: ["matsim", "travel_time_increase"],
  };
  if (effectActions[actionId]) {
    const [type, effect] = effectActions[actionId];
    if (task.type !== type) return;
    task.effect = effect;
    task.phase = effectNeedsMagnitude(effect)
      ? "magnitude"
      : type === "route" ? "disruption" : "location";
    if (type === "route" && state.routeBuilding) applyRouteTaskEffect(task);
    if (["scenario", "matsim"].includes(type) && state.scenarioBuilding) configureScenarioBuilderFromRequest(task);
    if (type === "scenario" && task.compoundSeed && !effectNeedsMagnitude(effect)) {
      task.phase = seedAssistantCompoundScenario(task) ? "ready" : "location";
    }
    const label = `${type === "route" ? "Route disruption" : type === "matsim" ? "MATSim disruption" : "Scenario effect"} confirmed: ${effect === "closed" ? "road closure" : effect === "speed_reduction" ? "speed reduction" : "longer travel time"}.`;
    appendAssistantMessage("assistant", label);
    persistAssistantChatMessage("assistant", label, { kind: "workflow_input", task_type: type });
    showAssistantPendingTaskStep();
    continueAssistantCompoundTaskIfReady(task);
    return;
  }
  const countMatch = String(actionId).match(/^(analysis|route|recovery|stress)_count_(\d+)$/);
  if (countMatch && task.type === countMatch[1]) {
    task.requestedCount = Number(countMatch[2]);
    if (task.compoundInput) {
      task.phase = "ready";
    } else if (task.type === "analysis") {
      task.sourceText = `${task.sourceText}; return the top ${task.requestedCount} results`;
      task.phase = "ready";
    } else {
      task.phase = task.type === "route" ? nextRouteInputPhase(task) : task.type === "recovery" ? "selection" : "ready";
    }
    appendAssistantMessage("assistant", `${task.type === "stress" ? "Failure stages" : "Requested results"} confirmed: ${task.requestedCount}.`);
    if (continueAssistantCompoundTaskIfReady(task)) return;
    showAssistantPendingTaskStep();
    return;
  }
  const magnitudeMatch = String(actionId).match(/^(route|scenario|matsim)_magnitude_(\d+)$/);
  if (magnitudeMatch && task.type === magnitudeMatch[1]) {
    task.magnitude = Number(magnitudeMatch[2]) / 100;
    task.phase = task.type === "route" ? "disruption" : "location";
    if (task.type === "route" && state.routeBuilding) applyRouteTaskEffect(task);
    if (["scenario", "matsim"].includes(task.type) && state.scenarioBuilding) configureScenarioBuilderFromRequest(task);
    if (task.type === "scenario" && task.compoundSeed) {
      task.phase = seedAssistantCompoundScenario(task) ? "ready" : "location";
    }
    const label = `${assistantEffectLabel(task.effect, task.magnitude)} confirmed.`;
    appendAssistantMessage("assistant", label);
    persistAssistantChatMessage("assistant", label, { kind: "workflow_input", task_type: task.type });
    showAssistantPendingTaskStep();
    continueAssistantCompoundTaskIfReady(task);
    return;
  }
  if (actionId === "matsim_use_active" && task.type === "matsim" && activeScenario()) {
    task.scenarioId = activeScenario().scenario_id;
    task.phase = nextAssistantMatsimAgentPhase(task);
    showAssistantPendingTaskStep();
    return;
  }
  if (actionId === "matsim_new_scenario" && task.type === "matsim") {
    task.scenarioId = null;
    task.phase = nextScenarioInputPhase(task, { matsim: true });
    showAssistantPendingTaskStep();
    return;
  }
  const agentMatch = String(actionId).match(/^matsim_agents_(50|100|150|200)$/);
  if (agentMatch && task.type === "matsim") {
    task.agentCount = Number(agentMatch[1]);
    task.phase = "ready";
    appendAssistantMessage("assistant", `MATSim agent group confirmed: ${task.agentCount}.`);
    showAssistantPendingTaskStep();
    continueAssistantCompoundTaskIfReady(task);
    return;
  }
  if (["recovery_use_critical_roads", "recovery_use_critical_areas", "recovery_use_active_scenario"].includes(actionId) && task.type === "recovery") {
    const sourceValue = actionId === "recovery_use_critical_roads"
      ? "__critical_roads__"
      : actionId === "recovery_use_critical_areas"
        ? "__critical_areas__"
        : activeScenario()?.scenario_id;
    if (!sourceValue) return;
    if (sourceValue.startsWith("__critical_")) {
      const artifact = latestAssistantArtifact(sourceValue === "__critical_areas__" ? "spatial_hotspots" : "critical_components");
      if (!artifact) return;
      await hydrateAssistantArtifact(artifact);
    }
    if (!state.recoveryBuilding) openRecoveryBuilder();
    $("#recovery-source").value = sourceValue;
    state.recoveryUnit = sourceValue === "__critical_areas__" ? "area" : "road";
    $("#recovery-top-k").value = String(task.requestedCount || 10);
    renderRecoveryBuilder();
    task.phase = "ready";
    showAssistantPendingTaskStep();
    return;
  }
  retireAssistantWorkflowCards();
  syncAssistantPendingRunAction(null);
  if (actionId === "run_route" && task.type === "route") {
    await runAssistantTaskOnce(task, async () => {
      updateAssistantActivity("execute", "Running the confirmed route analysis", "active");
      setAssistantOpen(false);
      await runRouteComparison();
    });
    return;
  }
  if (actionId === "run_recovery" && task.type === "recovery") {
    await runAssistantTaskOnce(task, async () => {
      updateAssistantActivity("execute", "Running the confirmed recovery ranking", "active");
      setAssistantOpen(false);
      await runRecoveryPriorities({ fromAssistant: true });
    });
    return;
  }
  if (actionId === "run_scenario" && task.type === "scenario") {
    await runAssistantTaskOnce(task, async () => {
      updateAssistantActivity("execute", "Running the confirmed disruption scenario", "active");
      setAssistantOpen(false);
      await saveScenarioDraft();
    });
    return;
  }
  if (actionId === "run_matsim" && task.type === "matsim") {
    await runAssistantTaskOnce(task, () => runAssistantMatsimTask(task));
    return;
  }
  if (actionId === "run_stress" && task.type === "stress") {
    await runAssistantTaskOnce(task, async () => {
      openStressBuilder();
      $("#stress-top-k").value = String(task.requestedCount || 20);
      renderStressBuilder();
      updateAssistantActivity("execute", "Running the confirmed resilience test", "active");
      setAssistantOpen(false);
      await runStressTest();
    });
    return;
  }
  if (actionId === "run_analysis" && task.type === "analysis") {
    await runAssistantTaskOnce(task, async () => {
      updateAssistantActivity("evidence", "Confirmed map scope received", "complete");
      const contract = await prepareAssistantAnalysis(task.sourceText, { autoRun: true });
      if (contract?.ready) await runAssistantAnalysis(null);
    });
  }
}

async function runAssistantTaskOnce(task, callback) {
  if (!task || task.runStarted) return null;
  task.runStarted = true;
  try {
    return await callback();
  } finally {
    // Successful workflows clear their pending task. A failed workflow keeps it
    // and may be retried, but a fast second click can never submit a duplicate.
    if (state.assistantPendingTask === task) task.runStarted = false;
  }
}

function pendingAssistantTaskMatches(type) {
  const pending = state.assistantPendingTask;
  return Boolean(
    pending
    && pending.type === type
    && pending.chatId === state.assistantChatId
    && pending.projectId === currentProjectId()
  );
}

function completeAssistantPendingTask(type, result, extraMetadata = {}) {
  if (!pendingAssistantTaskMatches(type) || !result) return false;
  const owner = { chatId: state.assistantPendingTask?.chatId || state.assistantCompoundWorkflow?.chatId || state.assistantChatId };
  appendAssistantResult(result, extraMetadata, owner);
  state.assistantPendingTask = null;
  syncAssistantPendingRunAction(null);
  setAssistantOpen(true, "llm");
  if (!state.assistantCompoundWorkflow) finishAssistantTurn();
  return true;
}

function appendAssistantScenarioResult(scenario) {
  if (!scenario?.scenario_id || !pendingAssistantTaskMatches("scenario")) return false;
  const result = scenario.result || {};
  const direct = Number(result.initial_disruption_count ?? result.affected_count ?? scenario.changes?.length ?? 0);
  const indirect = Number(result.indirect_impact_count ?? 0);
  const summary = `${formatNumber(direct)} confirmed road change${direct === 1 ? "" : "s"} were evaluated; ${formatNumber(indirect)} road${indirect === 1 ? "" : "s"} outside the selection received additional modelled activity.`;
  const card = document.createElement("article");
  card.className = "assistant-workcard is-result";
  card.innerHTML = `
    <header><span>Scenario complete</span><strong>${escapeHtml(scenario.name || "Disruption scenario")}</strong></header>
    <p class="assistant-result-conclusion">${escapeHtml(summary)}</p>
    <div class="assistant-result-metrics"><div><small>Initial changes</small><strong>${escapeHtml(formatNumber(direct))}</strong></div><div><small>Other affected roads</small><strong>${escapeHtml(formatNumber(indirect))}</strong></div></div>
    <footer><button class="secondary-button ripple" data-assistant-focus-result type="button">Show on map</button><button class="primary-button ripple" data-assistant-view-result type="button">Open full result</button></footer>
  `;
  appendAssistantNode(card, "llm");
  state.assistantHistory.push({ role: "assistant", content: `Scenario complete: ${summary}` });
  const metadata = {
    kind: "scenario_result",
    scenario_id: scenario.scenario_id,
    analysis_type: "closure_scenario",
    project_id: currentProjectId(),
  };
  registerAssistantChatArtifact(metadata);
  persistAssistantChatMessage("assistant", `Scenario complete: ${summary}`, metadata, { surface: "llm", chatId: state.assistantPendingTask?.chatId || state.assistantCompoundWorkflow?.chatId || state.assistantChatId });
  state.assistantPendingTask = null;
  setAssistantOpen(true, "llm");
  scrollAssistantSurface("llm");
  if (state.assistantCompoundWorkflow) window.setTimeout(() => advanceAssistantCompoundWorkflow("scenario"), 0);
  else finishAssistantTurn();
  return true;
}

function appendAssistantMatsimResult(result, scenario) {
  if (!result || !pendingAssistantTaskMatches("matsim")) return false;
  const outcomes = result.route_outcomes || {};
  const agents = Number(result.manifest?.agent_count || outcomes.agent_count || 0);
  const selectedRoads = Number(result.manifest?.selected_change_count || result.manifest?.changed_link_count || 0);
  const simulatedRoads = Number(result.manifest?.changed_link_count || 0);
  const excludedRoads = Number(result.manifest?.excluded_change_count || 0);
  const rerouted = Number(outcomes.rerouted || 0);
  const stuck = Number(outcomes.stuck_or_aborted || result.scenario?.stuck || 0);
  const timeChange = Number(result.mean_trip_time_change_percent);
  const roadCoverage = excludedRoads
    ? `${formatNumber(simulatedRoads)} of ${formatNumber(selectedRoads)} selected roads were simulated; ${formatNumber(excludedRoads)} ${excludedRoads === 1 ? "was" : "were"} excluded because ${excludedRoads === 1 ? "it is" : "they are"} not in the routable MATSim network. `
    : "";
  const summary = `${roadCoverage}${formatNumber(agents)} agents were simulated; ${formatNumber(rerouted)} rerouted and ${formatNumber(stuck)} became stuck or aborted.`;
  const card = document.createElement("article");
  card.className = "assistant-workcard is-result";
  card.innerHTML = `
    <header><span>MATSim complete</span><strong>${escapeHtml(scenario?.name || "Disruption simulation")}</strong></header>
    <p class="assistant-result-conclusion">${escapeHtml(summary)}</p>
    <div class="assistant-result-metrics">
      <div><small>Agents</small><strong>${escapeHtml(formatNumber(agents))}</strong></div>
      <div><small>Rerouted</small><strong>${escapeHtml(formatNumber(rerouted))}</strong></div>
      <div><small>Mean trip time</small><strong>${escapeHtml(Number.isFinite(timeChange) ? `${timeChange >= 0 ? "+" : ""}${timeChange.toFixed(1)}%` : "Not comparable")}</strong></div>
      ${excludedRoads ? `<div><small>Roads simulated</small><strong>${escapeHtml(`${formatNumber(simulatedRoads)} / ${formatNumber(selectedRoads)}`)}</strong></div>` : ""}
    </div>
    <footer><button class="secondary-button ripple" data-assistant-focus-matsim type="button">Show simulation on map</button><button class="primary-button ripple" data-assistant-view-matsim type="button">Open MATSim report</button></footer>
  `;
  appendAssistantNode(card, "llm");
  state.assistantHistory.push({ role: "assistant", content: `MATSim complete: ${summary}` });
  const metadata = {
    kind: "scenario_result",
    scenario_id: scenario?.scenario_id || state.activeScenarioId,
    analysis_type: "closure_scenario",
    project_id: currentProjectId(),
    matsim: true,
  };
  registerAssistantChatArtifact(metadata);
  persistAssistantChatMessage("assistant", `MATSim complete: ${summary}`, metadata, { surface: "llm", chatId: state.assistantPendingTask?.chatId || state.assistantCompoundWorkflow?.chatId || state.assistantChatId });
  state.assistantPendingTask = null;
  setAssistantOpen(true, "llm");
  scrollAssistantSurface("llm");
  if (state.assistantCompoundWorkflow) window.setTimeout(() => advanceAssistantCompoundWorkflow("matsim"), 0);
  else finishAssistantTurn();
  return true;
}

async function runAssistantMatsimTask(task) {
  if (!task || !pendingAssistantTaskMatches("matsim")) return;
  if (!state.matsimReadiness?.can_run) {
    appendAssistantMessage("assistant", "MATSim cannot run on this project yet.", state.matsimReadiness?.explanation || "A routable network is required.");
    return;
  }
  const uploadedDemand = assistantMatsimUsesUploadedDemand();
  if (!uploadedDemand && ![50, 100, 150, 200].includes(Number(task.agentCount))) {
    task.phase = "agent_count";
    showAssistantPendingTaskStep();
    return;
  }
  retireAssistantWorkflowCards();
  updateAssistantActivity("execute", "Preparing the confirmed MATSim scenario", "active");
  setAssistantOpen(false);
  try {
    let scenario = task.scenarioId
      ? state.scenarios.find((item) => item.scenario_id === task.scenarioId)
      : null;
    if (!scenario) {
      if (!state.scenarioBuilding || !state.scenarioDraft.size) throw new Error("Choose the disruption effect and affected roads or area first.");
      setScenarioEngine("matsim");
      scenario = await saveScenarioDraft({ assistantMatsimExecution: true });
      if (!scenario) throw new Error("The disruption scenario was not saved.");
      task.scenarioId = scenario.scenario_id;
    } else if (state.activeScenarioId !== scenario.scenario_id) {
      await selectScenario(scenario.scenario_id, { focus: false });
    }
    if (!uploadedDemand) $("#matsim-probe-agent-count").value = String(task.agentCount);
    renderMatsimReadiness();
    updateAssistantActivity(
      "execute",
      uploadedDemand
        ? "Generating agents from uploaded OD demand and running MATSim"
        : `Running ${formatNumber(task.agentCount)} MATSim agents`,
      "active",
    );
    const result = await runActiveScenarioMatsim({
      focusProgress: true,
      maxAgentsOverride: uploadedDemand ? ASSISTANT_MATSIM_UPLOADED_DEMAND_LIMIT : null,
      uploadedDemand,
    });
    if (!result) throw new Error("MATSim did not return a completed result.");
    updateAssistantActivity("execute", "MATSim run completed", "complete");
    updateAssistantActivity("verify", "Synchronising the map, report and Project Memory", "active");
    appendAssistantMatsimResult(result, scenario);
    updateAssistantActivity("verify", "Simulation map, report and Project Memory are ready", "complete");
  } catch (error) {
    updateAssistantActivity("execute", "MATSim did not complete", "failed");
    appendAssistantMessage("assistant", "MATSim could not complete this request.", error.message);
    persistAssistantChatMessage("assistant", `MATSim could not complete this request: ${error.message}`, { kind: "error" });
    setAssistantOpen(true, "llm");
    showAssistantPendingTaskStep();
  }
}

function configureRouteBuilderFromRequest(text) {
  const request = String(text || "");
  const count = Math.min(50, requestedCountFromText(request, 2));
  $("#route-alternative-count").value = String(count);
  $("#route-scenario").value = "__quick__";
  const scenarioArtifact = latestAssistantArtifact("closure_scenario");
  if (scenarioArtifact && /\b(?:that|previous|latest|saved)\s+scenario\b/i.test(request)) {
    const option = [...$("#route-scenario").options].find((item) => item.value === scenarioArtifact.artifact_id);
    if (option) $("#route-scenario").value = scenarioArtifact.artifact_id;
  }
  const usingQuickScenario = $("#route-scenario").value === "__quick__";
  state.routeDisruptionMode = usingQuickScenario ? "roads" : null;
  const magnitudeMatch = request.match(/(\d+(?:\.\d+)?)\s*%/);
  const magnitude = Math.max(5, Math.min(300, Number(magnitudeMatch?.[1]) || 50));
  if (!usingQuickScenario) {
    renderRouteBuilder();
    return;
  }
  if (/slow|speed\s*(?:reduction|loss|drop)|reduce\s+(?:the\s+)?speed/i.test(request)) {
    setRouteEffect("speed_reduction");
    $("#route-magnitude").value = String(Math.min(95, magnitude));
  } else if (/longer|travel[-\s]?time\s*(?:increase|delay)|add\s+(?:travel\s+)?time/i.test(request)) {
    setRouteEffect("travel_time_increase");
    $("#route-magnitude").value = String(magnitude);
  } else {
    setRouteEffect("closed");
  }
  $("#route-magnitude-output").textContent = `${$("#route-magnitude").value}%`;
  renderRouteBuilder();
}

function updatePendingRouteTask() {
  const pending = state.assistantPendingTask;
  if (pending?.type !== "route") return;
  if (!state.routeOrigin || !state.routeDestination) {
    updateAssistantActivity("execute", state.routeOrigin ? "Start confirmed; return to Copilot to choose the destination" : "Return to Copilot to begin start-point selection", "waiting");
  } else if ($("#route-scenario")?.value === "__quick__" && !state.routeDraft.size) {
    updateAssistantActivity("execute", "Endpoints confirmed; return to Copilot to define the disruption", "waiting");
  } else {
    updateAssistantActivity("execute", "Inputs confirmed; run the route model from Copilot", "waiting");
  }
}

async function runRecoveryRequestFromLlm(sourceText, requestId) {
  const request = String(sourceText || "");
  const wantsAreas = /critical[-\s]?area|area\s+(?:result|priorit)/i.test(request);
  const wantsRoads = /critical[-\s]?(?:road|link)|road\s+(?:result|priorit)/i.test(request);
  const acceptedTypes = wantsAreas
    ? ["spatial_hotspots"]
    : wantsRoads
      ? ["critical_components"]
      : ["spatial_hotspots", "critical_components"];
  const artifact = latestAssistantArtifact(acceptedTypes);
  if (!artifact) return null;
  await hydrateAssistantArtifact(artifact);
  const sourceValue = artifact.analysis_type === "spatial_hotspots" ? "__critical_areas__" : "__critical_roads__";
  if (!recoveryResultSources().some((source) => source.value === sourceValue)) return null;

  openRecoveryBuilder();
  $("#recovery-source").value = sourceValue;
  state.recoveryUnit = sourceValue === "__critical_areas__" ? "area" : "road";
  $("#recovery-top-k").value = String(requestedCountFromText(request, 10));
  renderRecoveryBuilder();
  updateAssistantActivity("decision", `Using the latest ${state.recoveryUnit === "area" ? "critical-area" : "critical-road"} result`, "complete");
  updateAssistantActivity("execute", "Ranking recovery priorities with the supported tier", "active");
  const result = await runRecoveryPriorities({ fromAssistant: true });
  if (!result || requestId !== state.assistantMessageRequestId) return result;
  return result;
}

async function handleLlmBuilderAction(action, sourceText, requestId) {
  if (!action || requestId !== state.assistantMessageRequestId) return;
  const sourceWorkflow = explicitAssistantWorkflow(sourceText);
  const taskType = sourceWorkflow?.type === "matsim"
    ? "matsim"
    : action.type === "open_route_builder"
    ? "route"
    : action.type === "open_recovery_builder"
      ? "recovery"
      : action.type === "open_scenario_builder"
        ? "scenario"
        : action.type === "open_stress_builder"
          ? "stress"
          : action.type === "open_matsim_builder"
            ? "matsim"
          : action.target || "builder";
  const requestedCount = requestedCountFromText(sourceText, null);
  const effect = ["route", "scenario", "matsim"].includes(taskType) ? scenarioEffectFromRequest(sourceText) : null;
  const magnitude = requestedMagnitudeFromText(sourceText, null);
  const agentCount = taskType === "matsim" ? requestedAgentCountFromText(sourceText) : null;
  const workflowKey = prepareAssistantWorkflowTransition(action, sourceText);
  if (taskType === "matsim") {
    await prepareMatsimReadinessForCopilot();
    if (requestId !== state.assistantMessageRequestId) return;
  }
  state.assistantPendingTask = {
    type: taskType,
    workflowKey: workflowKey || taskType,
    sourceText,
    action,
    chatId: state.assistantChatId,
    projectId: currentProjectId(),
    requestId,
    requestedCount,
    agentCount,
    effect,
    magnitude,
    phase: taskType === "route"
      ? requestedCount ? "origin" : "count"
      : taskType === "scenario"
        ? effect ? effectNeedsMagnitude(effect) && !hasAssistantMagnitude(magnitude) ? "magnitude" : "location" : "effect"
        : taskType === "stress"
          ? requestedCount ? "ready" : "stages"
          : taskType === "recovery"
            ? requestedCount ? "selection" : "count"
            : taskType === "matsim"
              ? activeScenario() && !effect ? "scenario_source" : effect ? effectNeedsMagnitude(effect) && !hasAssistantMagnitude(magnitude) ? "magnitude" : "location" : "scenario_source"
              : "selection",
    mapStep: null,
  };
  appendAssistantModelReceipt(taskType, {
    modelId: taskType === "route" ? "TWM-METHOD-23"
      : taskType === "recovery" ? "TWM-METHOD-4"
        : taskType === "stress" ? "TWM-METHOD-11"
          : taskType === "scenario" ? "TWM-METHOD-21/22"
            : taskType === "matsim" ? "MATSim scenario worker" : "",
    tier: taskType === "matsim"
      ? (state.matsimReadiness?.label || "Current supported simulation tier")
      : "Strongest tier supported by this upload",
    parameters: [
      ...(requestedCount ? [[taskType === "stress" ? "Stages" : "Requested", String(requestedCount)]] : []),
      ...(effect ? [["Effect", assistantEffectLabel(effect, magnitude)]] : []),
      ...(agentCount ? [["Agents", String(agentCount)]] : []),
    ],
  });
  if (action.type === "open_recovery_builder" && requestedCount && recoveryRequestReferencesPriorResult(sourceText)) {
    const result = await runRecoveryRequestFromLlm(sourceText, requestId);
    if (result) return;
  }
  const labels = {
    open_route_builder: "Application inputs are needed before route calculation",
    open_recovery_builder: "A disrupted road, area or saved result is needed",
    open_scenario_builder: "Disruption location and effect are needed",
    open_stress_builder: "Choose how many cumulative failures to test",
    open_matsim_builder: "Scenario, disruption and agent inputs are needed for MATSim",
  };
  updateAssistantActivity("decision", labels[action.type] || "Application inputs are needed", "complete");
  updateAssistantActivity(
    "execute",
    "Continue the required inputs step by step from Copilot",
    "waiting",
  );
  showAssistantPendingTaskStep();
}

async function consumeAssistantWorkflowText(text) {
  const task = state.assistantPendingTask;
  if (!task || !pendingAssistantTaskMatches(task.type)) return false;
  const cleaned = String(text || "").trim();
  let confirmation = "";
  if (["count", "stages"].includes(task.phase)) {
    const value = Number(cleaned.match(/\d+/)?.[0]);
    if (!Number.isInteger(value) || value < 1) return false;
    task.requestedCount = value;
    if (task.compoundInput) {
      task.phase = "ready";
    } else if (task.type === "analysis") {
      task.sourceText = `${task.sourceText}; return the top ${value} results`;
      task.phase = "ready";
    } else {
      task.phase = task.type === "route" ? nextRouteInputPhase(task) : task.type === "recovery" ? "selection" : "ready";
    }
    confirmation = `${task.type === "stress" ? "Failure stages" : "Requested results"} confirmed: ${value}.`;
  } else if (task.phase === "magnitude") {
    const value = Number(cleaned.match(/\d+(?:\.\d+)?/)?.[0]);
    const maximum = task.effect === "speed_reduction" ? 95 : 300;
    if (!Number.isFinite(value) || value <= 0 || value > maximum) return false;
    task.magnitude = value / 100;
    task.phase = task.type === "route" ? "disruption" : "location";
    if (task.type === "scenario" && task.compoundSeed) task.phase = seedAssistantCompoundScenario(task) ? "ready" : "location";
    confirmation = `${assistantEffectLabel(task.effect, task.magnitude)} confirmed.`;
  } else if (task.phase === "effect") {
    const effect = scenarioEffectFromRequest(cleaned);
    if (!effect) return false;
    task.effect = effect;
    task.phase = effectNeedsMagnitude(effect) ? "magnitude" : task.type === "route" ? "disruption" : "location";
    if (task.type === "scenario" && task.compoundSeed && !effectNeedsMagnitude(effect)) task.phase = seedAssistantCompoundScenario(task) ? "ready" : "location";
    confirmation = `${task.type === "route" ? "Route disruption" : task.type === "matsim" ? "MATSim disruption" : "Scenario effect"} confirmed: ${effect === "closed" ? "road closure" : effect === "speed_reduction" ? "speed reduction" : "longer travel time"}.`;
  } else if (task.phase === "agent_count") {
    const value = Number(cleaned.match(/\d+/)?.[0]);
    if (![50, 100, 150, 200].includes(value)) return false;
    task.agentCount = value;
    task.phase = "ready";
    confirmation = `MATSim agent group confirmed: ${value}.`;
  } else {
    return false;
  }
  state.assistantHistory.push({ role: "user", content: cleaned });
  appendAssistantMessage("user", cleaned);
  await persistAssistantChatMessage("user", cleaned, { kind: "workflow_input", task_type: task.type });
  appendAssistantMessage("assistant", confirmation);
  state.assistantHistory.push({ role: "assistant", content: confirmation });
  await persistAssistantChatMessage("assistant", confirmation, { kind: "workflow_input", task_type: task.type });
  if (!continueAssistantCompoundTaskIfReady(task)) showAssistantPendingTaskStep();
  return true;
}

function assistantQueueKey(chatId = state.assistantChatId, projectId = currentProjectId()) {
  return `${projectId || "no-project"}::${chatId || "draft"}`;
}

function assistantCompoundPlanFromText(text) {
  const request = String(text || "").trim();
  if (!request) return null;
  const wantsAreas = /(?:critical|key|important)\s+(?:areas?|regions?|zones?)|area\s+hotspots?|关键区域|重点区域|热点区域/i.test(request);
  const wantsRoads = /(?:critical|key|important)\s+(?:roads?|links?|edges?)|network\s+bottlenecks?|关键边|关键道路|关键路段/i.test(request);
  const wantsRecovery = /recover|recovery|restore|restoration|repair|reopen|恢复|修复/i.test(request);
  const wantsScenario = /disruption\s+scenario|scenario\s+builder|closure\s+scenario|slowdown\s+scenario|情景|场景/i.test(request);
  const wantsMatsim = /matsim|agent[-\s]?based\s+simulation|traffic\s+microsimulation|代理仿真|交通仿真/i.test(request);
  if (!(wantsAreas || wantsRoads)) return null;
  const downstreamCount = [wantsRecovery, wantsScenario, wantsMatsim].filter(Boolean).length;
  if (!downstreamCount) return null;
  const sourceType = wantsAreas && !wantsRoads ? "spatial_hotspots" : "critical_components";
  // A compound request must not invent an application-level result count.
  // Missing counts are collected in Copilot before the corresponding model runs.
  const criticalCount = requestedCountFromText(request, null);
  const recoveryMatch = request.match(/(?:top|first|rank|return)\s*(\d+)\s*(?:recovery|restoration|repair)|(?:recovery|restoration|repair).{0,18}(?:top|first|rank|return)?\s*(\d+)/i);
  const parsedRecoveryCount = Number(recoveryMatch?.[1] || recoveryMatch?.[2]);
  const recoveryCount = Number.isInteger(parsedRecoveryCount) && parsedRecoveryCount > 0
    ? parsedRecoveryCount
    : null;
  const effect = scenarioEffectFromRequest(request);
  const magnitude = requestedMagnitudeFromText(request, null);
  const agentCount = requestedAgentCountFromText(request);
  const steps = [{
    type: sourceType,
    label: sourceType === "spatial_hotspots"
      ? criticalCount ? `Find ${criticalCount} critical areas` : "Find critical areas - count needed"
      : criticalCount ? `Find ${criticalCount} critical roads` : "Find critical roads - count needed",
    count: criticalCount,
  }];
  if (wantsRecovery) steps.push({
    type: "recovery",
    label: recoveryCount ? `Rank ${recoveryCount} recovery priorities` : "Rank recovery priorities - count needed",
    count: recoveryCount,
  });
  if (wantsScenario || wantsMatsim) steps.push({
    type: "scenario",
    label: "Build the linked disruption scenario",
    effect,
    magnitude,
  });
  if (wantsMatsim) steps.push({ type: "matsim", label: "Run the linked MATSim experiment", agentCount });
  steps.push({ type: "verify", label: "Verify outputs and save the workflow" });
  return { sourceText: request, sourceType, steps };
}

function renderAssistantCompoundPlan(workflow) {
  const activity = assistantActivityForSurface("llm");
  if (!workflow || !activity?.stream || (!activity.stream.isConnected && !activity.stream.parentNode)) return;
  const event = document.createElement("article");
  event.className = "assistant-process-event assistant-compound-plan is-complete";
  event.dataset.processStep = "compound_plan";
  event.innerHTML = `
    <span aria-hidden="true"></span>
    <div><strong>Dependency-aware workflow prepared</strong><small>${workflow.steps.length} linked steps</small>
      <ol>${workflow.steps.map((step, index) => `<li data-compound-step="${index}" class="is-pending"><b>${index + 1}</b><span>${escapeHtml(step.label)}</span><small>Pending</small></li>`).join("")}</ol>
    </div>`;
  activity.stream.append(event);
}

function updateAssistantCompoundPlan(workflow, index, status, detail = "") {
  const item = assistantActivityForSurface("llm")?.stream?.querySelector(`[data-compound-step="${index}"]`);
  if (!item) return;
  item.className = `is-${status}`;
  item.querySelector("small").textContent = detail || (status === "complete" ? "Completed" : status === "waiting" ? "Waiting for input" : status === "failed" ? "Failed" : "Running");
}

function assistantSourceResult(workflow = state.assistantCompoundWorkflow) {
  return workflow?.sourceResult || completedResultFor(workflow?.sourceType || "critical_components");
}

function assistantScenarioSourceFeatures(workflow = state.assistantCompoundWorkflow) {
  const result = assistantSourceResult(workflow);
  if (!result) return [];
  const keys = workflow?.sourceType === "spatial_hotspots"
    ? [...new Set((result.items || []).flatMap((item) => item.entity_keys || []).map(String))]
    : (result.items || []).map((item) => String(item.key || "")).filter(Boolean);
  const byKey = new Map();
  for (const collection of [result.feature_collection, state.analysisCollection, state.mapCollection, state.networkMapping?.feature_collection]) {
    for (const feature of collection?.features || []) {
      const key = String(feature.properties?.key || "");
      if (key && !byKey.has(key)) byKey.set(key, feature);
    }
  }
  return keys.map((key) => byKey.get(key)).filter(Boolean);
}

function seedAssistantCompoundScenario(task) {
  const workflow = state.assistantCompoundWorkflow;
  if (!workflow || !task || task.type !== "scenario" || !task.effect) return 0;
  if (effectNeedsMagnitude(task.effect) && !hasAssistantMagnitude(task.magnitude)) return 0;
  if (!state.scenarioBuilding) openScenarioBuilder();
  configureScenarioBuilderFromRequest(task);
  const features = assistantScenarioSourceFeatures(workflow);
  if (features.length) applyScenarioFeatures(features);
  return state.scenarioDraft.size;
}

function continueAssistantCompoundTaskIfReady(task = state.assistantPendingTask) {
  if (!state.assistantCompoundWorkflow || !task || task.phase !== "ready") return false;
  const workflow = state.assistantCompoundWorkflow;
  const step = workflow.steps[workflow.index];
  if (task.compoundInput === "critical_count" && task.type === "analysis" && ["critical_components", "spatial_hotspots"].includes(step?.type)) {
    step.count = task.requestedCount;
    step.label = step.type === "spatial_hotspots"
      ? `Find ${step.count} critical areas`
      : `Find ${step.count} critical roads`;
    state.assistantPendingTask = null;
    workflow.runningIndex = null;
    window.setTimeout(() => runNextAssistantCompoundStep().catch((error) => failAssistantCompoundWorkflow(error.message)), 0);
    return true;
  }
  if (task.compoundInput === "recovery_count" && task.type === "recovery" && step?.type === "recovery") {
    step.count = task.requestedCount;
    step.label = `Rank ${step.count} recovery priorities`;
    state.assistantPendingTask = null;
    workflow.runningIndex = null;
    window.setTimeout(() => runNextAssistantCompoundStep().catch((error) => failAssistantCompoundWorkflow(error.message)), 0);
    return true;
  }
  // Model execution is never auto-started after the final application input.
  // The active Ready card and the persistent Copilot run action remain visible
  // until the user explicitly starts the confirmed calculation.
  return false;
}

function failAssistantCompoundWorkflow(message) {
  const workflow = state.assistantCompoundWorkflow;
  if (workflow) updateAssistantCompoundPlan(workflow, workflow.index, "failed", message || "Could not complete");
  state.assistantCompoundWorkflow = null;
  state.assistantPendingTask = null;
  appendAssistantMessage("assistant", "The linked workflow stopped at this step.", message || "The required result was not returned.");
  finishAssistantTurn();
}

async function advanceAssistantCompoundWorkflow(completedType = null) {
  const workflow = state.assistantCompoundWorkflow;
  if (!workflow) return;
  const step = workflow.steps[workflow.index];
  if (completedType && step?.type !== completedType) return;
  if (step) updateAssistantCompoundPlan(workflow, workflow.index, "complete");
  workflow.runningIndex = null;
  workflow.index += 1;
  await runNextAssistantCompoundStep();
}

async function runNextAssistantCompoundStep() {
  const workflow = state.assistantCompoundWorkflow;
  if (!workflow) return;
  const step = workflow.steps[workflow.index];
  if (!step) {
    state.assistantCompoundWorkflow = null;
    finishAssistantTurn();
    return;
  }
  // A single compound step owns exactly one platform calculation. Re-entrant
  // callbacks may update both Copilot and the main workspace, but may not
  // submit the same model a second time.
  if (workflow.runningIndex === workflow.index) return;
  workflow.runningIndex = workflow.index;
  updateAssistantCompoundPlan(workflow, workflow.index, "running");
  try {
    if (["critical_components", "spatial_hotspots"].includes(step.type)) {
      if (!Number.isInteger(step.count) || step.count < 1) {
        state.assistantPendingTask = {
          type: "analysis",
          workflowKey: `analysis:${step.type}`,
          analysisType: step.type,
          sourceText: workflow.sourceText,
          requestedCount: null,
          compoundInput: "critical_count",
          phase: "count",
          mapStep: null,
          chatId: state.assistantChatId,
          projectId: currentProjectId(),
          requestId: state.assistantMessageRequestId,
        };
        workflow.runningIndex = null;
        updateAssistantCompoundPlan(workflow, workflow.index, "waiting", "Choose the result count");
        showAssistantPendingTaskStep();
        return;
      }
      const command = step.type === "spatial_hotspots"
        ? `Find the top ${step.count} critical areas across the whole network.`
        : `Identify the top ${step.count} critical roads across the whole network.`;
      const contract = await prepareAssistantAnalysis(command, { autoRun: true });
      if (!contract?.ready) throw new Error("The uploaded project cannot run the requested criticality step at a supported tier.");
      const criticalResult = await runAssistantAnalysis(null);
      if (!criticalResult) throw new Error("The criticality step returned no reusable result.");
      workflow.sourceResult = criticalResult;
      await advanceAssistantCompoundWorkflow(step.type);
      return;
    }
    if (step.type === "recovery") {
      if (!Number.isInteger(step.count) || step.count < 1) {
        state.assistantPendingTask = {
          type: "recovery",
          workflowKey: "recovery",
          sourceText: workflow.sourceText,
          requestedCount: null,
          compoundInput: "recovery_count",
          phase: "count",
          mapStep: null,
          chatId: state.assistantChatId,
          projectId: currentProjectId(),
          requestId: state.assistantMessageRequestId,
        };
        workflow.runningIndex = null;
        updateAssistantCompoundPlan(workflow, workflow.index, "waiting", "Choose the recovery result count");
        showAssistantPendingTaskStep();
        return;
      }
      const result = assistantSourceResult(workflow);
      if (!result?.items?.length) throw new Error("Recovery ranking needs the verified critical-road or critical-area result from the previous step.");
      openRecoveryBuilder();
      const source = workflow.sourceType === "spatial_hotspots" ? "__critical_areas__" : "__critical_roads__";
      $("#recovery-source").value = source;
      state.recoveryUnit = workflow.sourceType === "spatial_hotspots" ? "area" : "road";
      $("#recovery-top-k").value = String(step.count || 10);
      renderRecoveryBuilder();
      appendAssistantModelReceipt("recovery", {
        modelId: "TWM-METHOD-4",
        tier: "Strongest recovery tier supported by this upload",
        parameters: [["Candidates", formatNumber(recoveryPayloadFromSource().selectedKeys.length)], ["Return", `Top ${step.count || 10}`]],
      });
      const recovery = await runRecoveryPriorities({ fromAssistant: true });
      if (!recovery) throw new Error("The recovery-priority model did not return a result.");
      appendAssistantResult(recovery, { compound_workflow: true });
      await advanceAssistantCompoundWorkflow("recovery");
      return;
    }
    if (step.type === "scenario") {
      state.assistantPendingTask = {
        type: "scenario",
        workflowKey: "scenario",
        sourceText: workflow.sourceText,
        chatId: state.assistantChatId,
        projectId: currentProjectId(),
        requestId: state.assistantMessageRequestId,
        requestedCount: null,
        effect: step.effect,
        magnitude: step.magnitude,
        compoundSeed: true,
        phase: !step.effect ? "effect" : effectNeedsMagnitude(step.effect) && !hasAssistantMagnitude(step.magnitude) ? "magnitude" : "ready",
        mapStep: null,
      };
      appendAssistantModelReceipt("scenario", {
        modelId: "TWM-METHOD-21/22",
        tier: "Strongest counterfactual tier supported by this upload",
        parameters: step.effect ? [["Effect", assistantEffectLabel(step.effect, step.magnitude)]] : [],
      });
      if (state.assistantPendingTask.phase !== "ready") {
        workflow.runningIndex = null;
        updateAssistantCompoundPlan(workflow, workflow.index, "waiting", "Application input needed");
        showAssistantPendingTaskStep();
        return;
      }
      const seeded = seedAssistantCompoundScenario(state.assistantPendingTask);
      if (!seeded) {
        state.assistantPendingTask.phase = "location";
        workflow.runningIndex = null;
        updateAssistantCompoundPlan(workflow, workflow.index, "waiting", "Affected location needed");
        showAssistantPendingTaskStep();
        return;
      }
      updateAssistantActivity("execute", "Running the disruption scenario from the verified upstream result", "active");
      const scenario = await saveScenarioDraft();
      if (!scenario) throw new Error("The linked disruption scenario was not saved.");
      return;
    }
    if (step.type === "matsim") {
      await prepareMatsimReadinessForCopilot();
      if (!state.matsimReadiness?.can_run) throw new Error(state.matsimReadiness?.explanation || "This upload does not provide a routable MATSim network.");
      const uploadedDemand = assistantMatsimUsesUploadedDemand();
      state.assistantPendingTask = {
        type: "matsim",
        workflowKey: "matsim",
        sourceText: workflow.sourceText,
        chatId: state.assistantChatId,
        projectId: currentProjectId(),
        requestId: state.assistantMessageRequestId,
        scenarioId: activeScenario()?.scenario_id || null,
        agentCount: uploadedDemand ? null : step.agentCount,
        phase: uploadedDemand || step.agentCount ? "ready" : "agent_count",
        mapStep: null,
      };
      appendAssistantModelReceipt("matsim", {
        modelId: "MATSim scenario worker",
        tier: state.matsimReadiness?.label || "Current supported simulation tier",
        parameters: uploadedDemand ? [["Agent source", "Uploaded OD demand"]] : step.agentCount ? [["Agents", String(step.agentCount)]] : [],
      });
      if (state.assistantPendingTask.phase !== "ready") {
        workflow.runningIndex = null;
        updateAssistantCompoundPlan(workflow, workflow.index, "waiting", "Agent input needed");
        showAssistantPendingTaskStep();
        return;
      }
      await runAssistantMatsimTask(state.assistantPendingTask);
      return;
    }
    if (step.type === "verify") {
      updateAssistantActivity("verify", "Checking linked outputs and Project Memory receipts", "active");
      await refreshProjectMemory().catch(() => null);
      updateAssistantActivity("verify", "Linked outputs verified and saved to this Chat and Project Memory", "complete");
      appendAssistantMessage("assistant", "The requested multi-step workflow is complete. Each stage used the existing registered platform model, and its result is available in this Chat, on the platform and in Project Memory.");
      await persistAssistantChatMessage("assistant", "Multi-step workflow complete; linked results were saved to this Chat and Project Memory.", { kind: "compound_workflow_result", project_id: currentProjectId() });
      await advanceAssistantCompoundWorkflow("verify");
    }
  } catch (error) {
    failAssistantCompoundWorkflow(error.message);
  }
}

async function beginAssistantCompoundWorkflow(plan) {
  state.assistantCompoundWorkflow = {
    ...plan,
    id: window.crypto?.randomUUID?.() || `workflow-${Date.now()}`,
    chatId: state.assistantChatId,
    projectId: currentProjectId(),
    index: 0,
    runningIndex: null,
  };
  renderAssistantCompoundPlan(state.assistantCompoundWorkflow);
  await runNextAssistantCompoundStep();
}

function currentAssistantQueue() {
  const key = assistantQueueKey();
  if (!state.assistantTaskQueues.has(key)) state.assistantTaskQueues.set(key, []);
  return state.assistantTaskQueues.get(key);
}

function renderAssistantTaskQueue() {
  const container = $("#assistant-task-queue");
  if (!container) return;
  const llm = state.assistantSurface === "llm";
  const queue = llm ? currentAssistantQueue() : [];
  const active = llm && state.assistantTurnRunning ? state.assistantActiveQueueItem : null;
  container.classList.toggle("is-hidden", !active && !queue.length);
  container.innerHTML = !llm ? "" : `
    ${active ? `<div class="assistant-queue-running"><span></span><b>Running</b><p>${escapeHtml(active.text)}</p></div>` : ""}
    ${queue.map((item, index) => `<div class="assistant-queue-item"><b>Queued ${index + 1}</b><p>${escapeHtml(item.text)}</p><button type="button" data-delete-assistant-queue-id="${escapeHtml(item.id)}" aria-label="Remove queued task ${index + 1}">×</button></div>`).join("")}
  `;
}

function enqueueAssistantTask(text) {
  const item = {
    id: window.crypto?.randomUUID?.() || `queued-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    text: String(text || "").trim(),
    chatId: state.assistantChatId,
    projectId: currentProjectId(),
    createdAt: Date.now(),
  };
  currentAssistantQueue().push(item);
  renderAssistantTaskQueue();
  return item;
}

function removeQueuedAssistantTask(itemId) {
  const queue = currentAssistantQueue();
  const index = queue.findIndex((item) => item.id === itemId);
  if (index >= 0) queue.splice(index, 1);
  renderAssistantTaskQueue();
}

function assistantTurnIsWaitingForInput() {
  return Boolean(state.assistantPendingTask && pendingAssistantTaskMatches(state.assistantPendingTask.type));
}

function finishAssistantTurn() {
  state.assistantTurnRunning = false;
  state.assistantActiveQueueItem = null;
  renderAssistantTaskQueue();
  window.setTimeout(() => drainAssistantTaskQueue(), 0);
}

async function drainAssistantTaskQueue() {
  if (state.assistantQueueDraining || state.assistantTurnRunning || assistantTurnIsWaitingForInput()) return;
  const queue = currentAssistantQueue();
  const next = queue.shift();
  if (!next) {
    renderAssistantTaskQueue();
    return;
  }
  if ((next.projectId || null) !== (currentProjectId() || null) || (next.chatId || null) !== (state.assistantChatId || null)) {
    queue.unshift(next);
    renderAssistantTaskQueue();
    return;
  }
  state.assistantQueueDraining = true;
  renderAssistantTaskQueue();
  try {
    await processAssistantMessage(next.text, { queueItem: next });
  } finally {
    state.assistantQueueDraining = false;
  }
}

async function sendAssistantMessage(message = null) {
  if (state.assistantSurface === "llm" && !llmInferenceReady()) {
    toast("Connect an LLM API first", "LLM Copilot cannot answer or operate through the local Platform Assistant.", true);
    await openAssistantSettings();
    return;
  }
  const input = $("#assistant-input");
  const text = String(message ?? input.value).trim();
  if (!text) return;
  const explicitWorkflow = state.assistantSurface === "llm" ? explicitAssistantWorkflow(text) : null;
  const pendingWorkflowKey = state.assistantPendingTask
    ? (state.assistantPendingTask.workflowKey || state.assistantPendingTask.type)
    : null;
  const switchesPendingWorkflow = Boolean(explicitWorkflow && pendingWorkflowKey && explicitWorkflow.key !== pendingWorkflowKey);
  if (state.assistantSurface === "llm" && !switchesPendingWorkflow && await consumeAssistantWorkflowText(text)) {
    input.value = "";
    return;
  }
  if (state.assistantSurface === "llm" && (state.assistantTurnRunning || state.assistantBusy || state.activeAnalysisJobId || assistantTurnIsWaitingForInput())) {
    enqueueAssistantTask(text);
    input.value = "";
    return;
  }
  await processAssistantMessage(text);
}

async function processAssistantMessage(message = null, { queueItem = null } = {}) {
  const requestSurface = normaliseAssistantSurface(state.assistantSurface);
  const isLlmRequest = requestSurface === "llm";
  if (isLlmRequest && !llmInferenceReady()) {
    toast("Connect an LLM API first", "LLM Copilot cannot answer or operate through the local Platform Assistant.", true);
    await openAssistantSettings();
    return;
  }
  const input = $("#assistant-input");
  const text = String(message ?? input.value).trim();
  if (!text) return;
  if (isLlmRequest) input.value = "";
  const compoundPlan = isLlmRequest ? assistantCompoundPlanFromText(text) : null;
  if (isLlmRequest) state.assistantTurnRunning = true;
  state.assistantActiveQueueItem = isLlmRequest
    ? (queueItem || { id: `active-${Date.now()}`, text, chatId: state.assistantChatId, projectId: currentProjectId() })
    : null;
  renderAssistantTaskQueue();
  const explicitWorkflow = isLlmRequest ? explicitAssistantWorkflow(text) : null;
  const pendingWorkflowKey = state.assistantPendingTask
    ? (state.assistantPendingTask.workflowKey || state.assistantPendingTask.type)
    : null;
  const switchesPendingWorkflow = Boolean(
    explicitWorkflow
    && pendingWorkflowKey
    && explicitWorkflow.key !== pendingWorkflowKey
  );
  if (isLlmRequest && !switchesPendingWorkflow && await consumeAssistantWorkflowText(text)) {
    input.value = "";
    return;
  }
  if (isLlmRequest && state.assistantChatId && state.assistantChatProjectId !== currentProjectId()) {
    await startNewAssistantChat({ closeDrawer: false });
  }
  const requestId = isLlmRequest ? ++state.assistantMessageRequestId : ++state.platformAssistantMessageRequestId;
  // Every new utterance supersedes the previous pending assistant contract.
  // This prevents a stale Confirm button from running after “make it 50”,
  // “use travel time” or “cancel”.
  if (isLlmRequest) {
    state.commandRequestId += 1;
    state.analysisContextVersion += 1;
    state.assistantAnalysisRequestId += 1;
    state.assistantBusy = false;
    state.commandContract = null;
    state.assistantContract = null;
    state.assistantPendingTask = null;
    assistantSurfaceContainer("llm")?.querySelectorAll('[data-assistant-run]').forEach((button) => {
      button.disabled = true;
      button.textContent = "Superseded by your latest message";
    });
  } else {
    state.platformAssistantBusy = true;
  }
  // Retain the full browser-session transcript.  The backend keeps recent
  // turns verbatim and compacts older turns against the model context budget.
  const activeHistory = assistantHistoryForSurface(requestSurface);
  const priorHistory = activeHistory.map((item) => ({ ...item }));
  activeHistory.push({ role: "user", content: text });
  appendAssistantMessage("user", text, "", requestSurface);
  if (isLlmRequest) await persistAssistantChatMessage("user", text, { project_id: currentProjectId() }, { surface: requestSurface });
  if (!isLlmRequest && requestSurface === state.assistantSurface) input.value = "";
  const activity = startAssistantActivity(requestSurface);
  try {
    const result = await api.assistantMessage({
      message: text,
      project_id: currentProjectId(),
      selected_keys: [...state.selectedKeys],
      bbox: state.bbox,
      current_metric: state.metric,
      current_time: currentTime(),
      current_view: state.currentView,
      active_scenario_name: activeScenario()?.name || null,
      scenario_count: state.scenarios.length,
      scenario_building: state.scenarioBuilding,
      history: priorHistory,
      chat_artifacts: assistantArtifactsForRequest(),
      prefer_llm: isLlmRequest,
      session_id: state.assistantSessionId,
      provider: state.assistantProvider,
      interaction_mode: isLlmRequest ? "guide" : state.assistantMode,
    });
    const latestRequestId = isLlmRequest ? state.assistantMessageRequestId : state.platformAssistantMessageRequestId;
    if (requestId !== latestRequestId) {
      activity.remove();
      return;
    }
    updateAssistantActivity("understand", "Request understood from the current conversation", "complete", requestSurface);
    const previewAction = renderAssistantResponse(result, requestSurface);
    if (isLlmRequest) {
      await refreshAssistantStatus().catch(() => null);
      if (compoundPlan && requestId === state.assistantMessageRequestId) {
        updateAssistantActivity("plan", "Building a dependency-aware sequence from the requested outcomes", "complete", requestSurface);
        await beginAssistantCompoundWorkflow(compoundPlan);
        return;
      }
      if (previewAction?.value && requestId === state.assistantMessageRequestId) {
        const workflowKey = prepareAssistantWorkflowTransition(previewAction, previewAction.value || text);
        updateAssistantActivity(
          "evidence",
          state.project ? `Checking ${state.project.name} data, scope and available evidence` : "A project is required for this analysis",
          "active",
          requestSurface,
        );
        if (analysisRequestNeedsCount(previewAction.value)) {
          const analysis = analysisContextForCommand(previewAction.value);
          state.assistantPendingTask = {
            type: "analysis",
            workflowKey: workflowKey || "analysis",
            analysisType: analysis.intent,
            sourceText: previewAction.value,
            requestedCount: null,
            phase: "count",
            mapStep: null,
            chatId: state.assistantChatId,
            projectId: currentProjectId(),
            requestId,
          };
          updateAssistantActivity("evidence", "Choose the result count in Copilot before this analysis runs", "waiting", requestSurface);
          showAssistantPendingTaskStep();
          return;
        }
        const contract = await prepareAssistantAnalysis(previewAction.value, { autoRun: true, surface: requestSurface });
        updateAssistantActivity(
          "evidence",
          contract?.ready ? "Project data and requested scope passed validation" : "More project input is required",
          contract?.ready ? "complete" : "waiting",
          requestSurface,
        );
        if (contract?.ready && requestId === state.assistantMessageRequestId) {
          await runAssistantAnalysis(null);
        }
      } else {
        const builderAction = (result.actions || []).find((action) => LLM_INPUT_BUILDERS.has(action.type));
        if (builderAction) await handleLlmBuilderAction(builderAction, builderAction.value || text, requestId);
      }
    }
    if (!isLlmRequest && state.assistantMode === "autopilot") {
      await executeAutopilotResponse(result, previewAction, requestId, requestSurface);
    }
  } catch (error) {
    const latestRequestId = isLlmRequest ? state.assistantMessageRequestId : state.platformAssistantMessageRequestId;
    if (requestId !== latestRequestId) return;
    const failureStep = assistantActivityForSurface(requestSurface)?.events?.has("execute") ? "execute" : "understand";
    updateAssistantActivity(failureStep, "The request could not be completed", "failed", requestSurface);
    const technicalMessage = String(error?.message || "");
    const readableMessage = /at most\s+\d+\s+characters|validation error/i.test(technicalMessage)
      ? "The conversation could not be prepared for the selected model. Your chat has been kept; please try again."
      : technicalMessage || "The connected provider did not return an answer. Please try again.";
    appendAssistantMessage("assistant", "I could not answer that request yet.", readableMessage, requestSurface);
    if (isLlmRequest) persistAssistantChatMessage("assistant", `I could not answer that request yet: ${readableMessage}`, { kind: "error" }, { surface: requestSurface });
  } finally {
    if (isLlmRequest) {
      if (!assistantTurnIsWaitingForInput() && !state.assistantBusy && !state.activeAnalysisJobId && !state.assistantCompoundWorkflow) finishAssistantTurn();
    } else {
      state.platformAssistantBusy = false;
    }
  }
}

function highlightElement(element) {
  if (!element) return;
  element.scrollIntoView({ behavior: "smooth", block: "center" });
  element.classList.remove("assistant-highlight");
  window.requestAnimationFrame(() => element.classList.add("assistant-highlight"));
  window.setTimeout(() => element.classList.remove("assistant-highlight"), 2800);
}

function focusAssistantTarget(target) {
  const targets = {
    map: ["explore", "#map-panel"],
    history: ["explore", ".historical-tools"],
    core_tools: ["explore", ".core-tool-grid"],
    critical_roads: ["explore", "#quick-critical-roads"],
    critical_areas: ["explore", "#quick-critical-areas"],
    command: ["explore", "#command-input"],
    timeline: ["explore", "#time-control"],
    results: ["explore", "#result-panel"],
    data_quality: ["data", "#quality-grid"],
    scenario: ["explore", "#scenario-builder"],
    scenario_library: ["explore", "#scenario-hub"],
    route: ["explore", "#route-builder"],
    recovery: ["explore", "#recovery-builder"],
    stress: ["explore", "#stress-builder"],
    scientific_validation: ["models", "#validation-panel"],
    models: ["models", "#view-models .view-heading"],
    saved: ["saved", ".saved-panel"],
    memory: ["memory", "#memory-overview"],
    matsim: ["explore", "#matsim-simulation-card"],
  };
  const [view, selector] = targets[target] || [state.currentView, null];
  switchView(view);
  setAssistantOpen(false);
  window.setTimeout(() => highlightElement(selector ? $(selector) : null), 130);
}

async function navigateAssistantFeature(feature) {
  if (feature === "upload") {
    setAssistantOpen(false);
    openUpload();
    return;
  }
  if (!state.project && !["models", "validation"].includes(feature)) {
    setAssistantOpen(false);
    openUpload();
    toast("Choose data first", "This function opens as soon as a project is available.");
    return;
  }
  if (feature === "recovery") {
    setAssistantOpen(false); openRecoveryBuilder();
    return window.setTimeout(() => highlightElement($("#recovery-builder")), 140);
  }
  if (feature === "route") {
    setAssistantOpen(false); await openRouteBuilder();
    return window.setTimeout(() => highlightElement($("#route-builder")), 140);
  }
  if (feature === "stress") {
    setAssistantOpen(false); openStressBuilder();
    return window.setTimeout(() => highlightElement($("#stress-builder")), 140);
  }
  if (feature === "scenario") {
    setAssistantOpen(false); openScenarioBuilder();
    return window.setTimeout(() => highlightElement($("#scenario-builder")), 140);
  }
  if (feature === "matsim") {
    setAssistantOpen(false);
    if (!activeScenario()) {
      openScenarioBuilder();
      setScenarioEngine("matsim");
      return window.setTimeout(() => highlightElement($("#scenario-engine-matsim")), 140);
    }
    switchView("explore");
    renderScenarioImpact(activeScenario());
    return window.setTimeout(() => highlightElement($("#matsim-simulation-card")), 140);
  }
  const aliases = {
    data: "data_quality",
    validation: "scientific_validation",
  };
  focusAssistantTarget(aliases[feature] || feature);
}

async function executeAssistantAction(action, options = {}) {
  const allowed = new Set(["navigate", "open_upload", "start_tour", "focus", "set_command", "set_metric", "set_selection_mode", "fit_map", "preview_analysis", "open_scenario_builder", "open_scenario_library", "open_route_builder", "open_recovery_builder", "open_stress_builder", "open_matsim_builder"]);
  if (!action || !allowed.has(action.type)) return;
  if (action.type === "navigate" && ["explore", "data", "models", "saved", "memory"].includes(action.target)) {
    switchView(action.target);
    setAssistantOpen(false);
    return;
  }
  if (action.type === "open_upload") {
    setAssistantOpen(false);
    openUpload();
    return;
  }
  if (action.type === "start_tour") {
    setAssistantOpen(false);
    openGuide(0);
    return;
  }
  if (action.type === "open_scenario_builder") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    openScenarioBuilder();
    window.setTimeout(() => highlightElement($("#scenario-builder")), 130);
    return;
  }
  if (action.type === "open_matsim_builder") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    openScenarioBuilder();
    setScenarioEngine("matsim");
    window.setTimeout(() => highlightElement($("#scenario-engine-matsim")), 130);
    return;
  }
  if (action.type === "open_scenario_library") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    openScenarioLibrary();
    return;
  }
  if (action.type === "open_route_builder") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    await openRouteBuilder();
    if (options.fromLlm) {
      configureRouteBuilderFromRequest(options.sourceText || action.value || "");
      updatePendingRouteTask();
    }
    window.setTimeout(() => highlightElement($("#route-builder")), 130);
    return;
  }
  if (action.type === "open_recovery_builder") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    openRecoveryBuilder();
    if (options.fromLlm && !latestAssistantArtifact(["spatial_hotspots", "critical_components"])) {
      $("#recovery-source").value = "manual";
      state.recoveryUnit = "road";
      renderRecoveryBuilder();
    }
    window.setTimeout(() => highlightElement($("#recovery-builder")), 130);
    return;
  }
  if (action.type === "open_stress_builder") {
    if (!state.project) return openUpload();
    setAssistantOpen(false);
    openStressBuilder();
    window.setTimeout(() => highlightElement($("#stress-builder")), 130);
    return;
  }
  if (action.type === "focus") {
    focusAssistantTarget(action.target);
    return;
  }
  if (action.type === "set_command") {
    switchView("explore");
    $("#command-input").value = action.value || "";
    invalidateAnalysisReview();
    setAssistantOpen(false);
    window.setTimeout(() => highlightElement($("#command-input")), 130);
    return;
  }
  if (action.type === "set_metric" && availableMetricOptions(state.project?.summary).some((item) => item.key === action.target)) {
    const option = [...$("#metric-select").options].find((item) => item.value === action.target);
    if (!option || option.disabled) return toast("Measure unavailable", "The current project does not contain that measure.", true);
    state.metric = action.target;
    $("#metric-select").value = action.target;
    invalidateAnalysisReview();
    try {
      await refreshMap(null);
      await refreshOpenInspector();
    } catch (error) {
      toast("Measure unavailable", error.message, true);
    }
    focusAssistantTarget("map");
    return;
  }
  if (action.type === "set_selection_mode" && ["point", "area"].includes(action.target)) {
    switchView("explore");
    const button = action.target === "area" ? $("#select-area") : $("#select-point");
    setAssistantOpen(false);
    window.setTimeout(() => { button.click(); highlightElement(button); }, 130);
    return;
  }
  if (action.type === "fit_map") {
    state.map?.fitExtent();
    focusAssistantTarget("map");
    return;
  }
  if (action.type === "preview_analysis") {
    await prepareAssistantAnalysis(action.value || "");
  }
}

function openCapabilityTour() {
  if (!$("#platform-workflow-video source")) {
    openGuide();
    return;
  }
  const dialog = $("#capability-dialog");
  const video = $("#platform-workflow-video");
  if (!dialog.open) dialog.showModal();
  video.currentTime = 0;
  video.play().catch(() => {
    // Browser autoplay policies may require the user to press Play.
  });
}

function closeCapabilityTour() {
  const dialog = $("#capability-dialog");
  const video = $("#platform-workflow-video");
  video.pause();
  if (dialog.open) dialog.close();
}

function renderGuideStep() {
  const step = GUIDE_STEPS[state.guideIndex];
  $("#guide-step-label").textContent = `Step ${state.guideIndex + 1} of ${GUIDE_STEPS.length}`;
  $("#guide-step-title").textContent = step.title;
  $("#guide-step-copy").textContent = step.copy;
  $("#guide-step-tip").textContent = step.tip;
  $("#guide-visual").innerHTML = step.visual;
  $("#guide-progress").innerHTML = GUIDE_STEPS.map((_, index) => `<i class="${index < state.guideIndex ? "is-complete" : index === state.guideIndex ? "is-current" : ""}"></i>`).join("");
  $("#guide-previous").disabled = state.guideIndex === 0;
  $("#guide-next").textContent = state.guideIndex === GUIDE_STEPS.length - 1 ? "Done" : "Next";
}

function openGuide(index = 0) {
  state.guideIndex = Math.max(0, Math.min(GUIDE_STEPS.length - 1, index));
  renderGuideStep();
  const dialog = $("#guide-dialog");
  if (!dialog.open) dialog.showModal();
}

function showGuideTarget() {
  const target = GUIDE_STEPS[state.guideIndex].target;
  $("#guide-dialog").close();
  if (target === "upload") return openUpload();
  switchView("explore");
  if (target === "map") return window.setTimeout(() => highlightElement($("#disaster-map")), 120);
  if (target === "history") return window.setTimeout(() => highlightElement($(".historical-tools")), 120);
  if (target === "core") return window.setTimeout(() => highlightElement($(".core-tool-grid")), 120);
  if (target === "scenario") {
    if (!state.project) return openUpload();
    return openScenarioBuilder();
  }
  if (target === "recovery") {
    if (!state.project) return openUpload();
    return openRecoveryBuilder();
  }
  if (target === "route") {
    if (!state.project) return openUpload();
    return openRouteBuilder();
  }
  if (target === "assistant") return setAssistantOpen(true, "platform");
  if (target === "memory") {
    switchView("memory");
    return window.setTimeout(() => highlightElement($("#memory-timeline")), 120);
  }
  if (target === "command") return window.setTimeout(() => highlightElement($("#command-input")), 120);
  const review = $("#analysis-review").classList.contains("is-hidden") ? $("#interpret-command") : $("#analysis-review");
  window.setTimeout(() => highlightElement(review), 120);
}

function bindInteractions() {
  $$(".nav-button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $("#research-run").addEventListener("click", runResearchValidation);
  $("#research-refresh").addEventListener("click", () => refreshResearchReports({ loadLatest: false }).catch((error) => toast("Reports could not be refreshed", error.message, true)));
  $("#research-cancel").addEventListener("click", cancelResearchValidation);
  $("#research-report-picker").addEventListener("change", (event) => openResearchReport(event.target.value).catch((error) => toast("Report could not be opened", error.message, true)));
  $("#research-loop").addEventListener("click", (event) => {
    const button = event.target.closest("[data-research-stage-target]");
    if (!button) return;
    const detail = $(`[data-research-stage="${CSS.escape(button.dataset.researchStageTarget)}"]`);
    if (detail) { detail.open = true; detail.scrollIntoView({ behavior: "smooth", block: "center" }); }
  });
  $$("[data-research-figure]").forEach((button) => button.addEventListener("click", () => downloadResearchFigure(button.dataset.researchFigure)));
  $("#research-export-png").addEventListener("click", exportResearchFiguresPng);
  $("#research-print").addEventListener("click", printResearchReport);
  $("#atmosphere-toggle").addEventListener("click", () => {
    state.atmosphereEnabled = !state.atmosphereEnabled;
    setHazardAtmosphere(state.scenarioBuilding ? $("#scenario-hazard").value : activeScenario()?.hazard_type || state.project?.hazard_type || null);
  });
  $("#scenario-select").addEventListener("change", (event) => {
    selectScenario(event.target.value || null, { focus: true })
      .catch((error) => toast("Scenario could not be opened", error.message, true));
  });
  $("#new-scenario").addEventListener("click", () => openScenarioBuilder());
  $("#quick-build-scenario").addEventListener("click", () => openScenarioBuilder());
  $("#map-new-scenario").addEventListener("click", () => openScenarioBuilder());
  $("#quick-recovery-analysis").addEventListener("click", openRecoveryBuilder);
  $("#close-recovery-builder").addEventListener("click", () => closeRecoveryBuilder());
  $("#cancel-recovery").addEventListener("click", () => closeRecoveryBuilder());
  $("#recovery-source").addEventListener("change", () => {
    if ($("#recovery-source").value === "__critical_areas__") state.recoveryUnit = "area";
    else if ($("#recovery-source").value === "__critical_roads__") state.recoveryUnit = "road";
    state.recoveryMode = $("#recovery-source").value === "manual" ? "roads" : null;
    state.map?.setMode("point");
    renderRecoveryBuilder();
  });
  $("#recovery-unit-road").addEventListener("click", () => { state.recoveryUnit = "road"; renderRecoveryBuilder(); });
  $("#recovery-unit-area").addEventListener("click", () => { state.recoveryUnit = "area"; renderRecoveryBuilder(); });
  $("#recovery-pick-roads").addEventListener("click", () => {
    state.recoveryMode = "roads";
    state.map?.setMode("point");
    renderRecoveryBuilder();
    setMapBanner("Click disrupted roads", "Each click adds one recovery candidate.", true);
  });
  $("#recovery-draw-area").addEventListener("click", () => {
    state.recoveryMode = "area";
    state.map?.setMode("area");
    renderRecoveryBuilder();
    setMapBanner("Draw a disrupted area", "Press, drag a rectangle and release.", true);
  });
  $("#recovery-clear").addEventListener("click", clearRecoveryDraft);
  $("#recovery-objective").addEventListener("change", renderRecoveryBuilder);
  $("#run-recovery").addEventListener("click", runRecoveryPriorities);
  $("#quick-stress-analysis").addEventListener("click", openStressBuilder);
  $("#close-stress-builder").addEventListener("click", closeStressBuilder);
  $("#cancel-stress").addEventListener("click", closeStressBuilder);
  $("#stress-top-k").addEventListener("input", renderStressBuilder);
  $("#run-stress").addEventListener("click", runStressTest);
  $("#quick-route-analysis").addEventListener("click", () => openRouteBuilder());
  $("#close-route-builder").addEventListener("click", () => closeRouteBuilder());
  $("#cancel-route").addEventListener("click", () => closeRouteBuilder());
  $("#route-pick-origin").addEventListener("click", () => setRoutePickRole("origin"));
  $("#route-pick-destination").addEventListener("click", () => setRoutePickRole("destination"));
  $("#route-view-points").addEventListener("click", () => switchRouteEndpointView("points"));
  $("#route-view-links").addEventListener("click", () => switchRouteEndpointView("links"));
  $("#route-swap").addEventListener("click", () => {
    [state.routeOrigin, state.routeDestination] = [state.routeDestination, state.routeOrigin];
    state.selectedKeys = new Set([state.routeOrigin?.key, state.routeDestination?.key].filter(Boolean));
    state.map?.setSelectedKeys([...state.selectedKeys]);
    renderRouteBuilder();
  });
  $("#route-scenario").addEventListener("change", () => {
    state.routePickRole = null;
    state.routeDisruptionMode = $("#route-scenario").value === "__quick__" ? "roads" : null;
    state.map?.setMode("point");
    renderRouteBuilder();
    setMapBanner(
      state.routeDisruptionMode ? "Add the temporary disruption" : "Route endpoints ready",
      state.routeDisruptionMode ? "Click affected roads or draw an affected area." : "Calculate a recommendation for the selected network state.",
      true,
    );
  });
  $$('[data-route-effect]').forEach((button) => button.addEventListener("click", () => setRouteEffect(button.dataset.routeEffect)));
  $("#route-magnitude").addEventListener("input", (event) => { $("#route-magnitude-output").textContent = `${event.target.value}%`; });
  $("#route-pick-disrupted").addEventListener("click", () => {
    state.routePickRole = null;
    state.routeDisruptionMode = "roads";
    state.map?.setMode("point");
    renderRouteDraft();
    setMapBanner("Click affected roads", "Each click applies the selected temporary disruption.", true);
  });
  $("#route-draw-disrupted").addEventListener("click", () => {
    state.routePickRole = null;
    state.routeDisruptionMode = "area";
    state.map?.setMode("area");
    renderRouteDraft();
    setMapBanner("Draw an affected area", "Press, drag a rectangle and release.", true);
  });
  $("#route-clear-disruption").addEventListener("click", clearRouteDisruption);
  $("#run-route").addEventListener("click", runRouteComparison);
  $("#edit-scenario").addEventListener("click", () => {
    const scenario = activeScenario();
    if (scenario) openScenarioBuilder(scenario);
  });
  $("#manage-scenarios").addEventListener("click", openScenarioLibrary);
  $("#close-scenario-builder").addEventListener("click", () => closeScenarioBuilder());
  $("#cancel-scenario").addEventListener("click", () => closeScenarioBuilder());
  $$("[data-scenario-effect]").forEach((button) => button.addEventListener("click", () => setScenarioEffect(button.dataset.scenarioEffect)));
  $("#scenario-magnitude").addEventListener("input", (event) => { $("#scenario-magnitude-output").textContent = `${event.target.value}%`; });
  $("#scenario-hazard").addEventListener("change", (event) => setHazardAtmosphere(event.target.value));
  $("#scenario-engine-network").addEventListener("click", () => setScenarioEngine("network"));
  $("#scenario-engine-matsim").addEventListener("click", () => setScenarioEngine("matsim"));
  $("#scenario-pick-roads").addEventListener("click", () => {
    state.map?.setMode("point");
    $("#scenario-pick-roads").classList.add("is-active");
    $("#scenario-draw-area").classList.remove("is-active");
    setMapBanner("Click affected roads", "Each click adds the currently selected change.", true);
    scheduleNextActionCues();
  });
  $("#scenario-draw-area").addEventListener("click", () => {
    state.map?.setMode("area");
    $("#scenario-draw-area").classList.add("is-active");
    $("#scenario-pick-roads").classList.remove("is-active");
    setMapBanner("Draw an affected area", "Press, drag a rectangle and release. Every mapped entity inside will be reviewed before saving.", true);
    scheduleNextActionCues();
  });
  $("#scenario-draft-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-scenario-change]");
    if (button) removeScenarioDraftChange(button.dataset.removeScenarioChange);
  });
  $("#scenario-undo").addEventListener("click", undoScenarioDraft);
  $("#scenario-clear").addEventListener("click", clearScenarioDraft);
  $("#save-scenario").addEventListener("click", () => saveScenarioDraft({ manualExecution: true }));
  $("#focus-scenario").addEventListener("click", () => activateScenario(state.activeScenarioId, { focus: true }));
  $("#duplicate-scenario").addEventListener("click", () => duplicateScenario());
  $("#delete-active-scenario").addEventListener("click", () => deleteScenario());
  $("#run-matsim-scenario").addEventListener("click", () => runActiveScenarioMatsim());
  $("#matsim-probe-agent-count").addEventListener("change", renderMatsimReadiness);
  $("#stop-matsim-scenario").addEventListener("click", async () => {
    const jobId = state.matsimJobId;
    if (!jobId) return;
    state.matsimJobId = null;
    try { await api.cancelJob(jobId); } catch (error) { toast("MATSim stop request failed", error.message, true); }
    $("#matsim-progress").classList.add("is-hidden");
    renderMatsimReadiness();
  });
  $("#play-matsim").addEventListener("click", toggleMatsimPlayback);
  $("#matsim-slider").addEventListener("input", (event) => {
    stopMatsimPlayback();
    setMatsimFrame(Number(event.target.value));
  });
  $("#matsim-link-shifts").addEventListener("click", (event) => {
    const button = event.target.closest("[data-matsim-shift-index]");
    if (!button) return;
    const item = state.matsimResult?.top_link_shifts?.[Number(button.dataset.matsimShiftIndex)];
    if (!item?.geometry) return;
    const collection = {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: item.geometry, properties: { key: item.key } }],
    };
    $("#disaster-map")?.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => {
      state.map?.resize();
      state.map?.focusCollection(collection);
    }, 360);
  });
  $("#close-scenario-library").addEventListener("click", () => $("#scenario-library-dialog").close());
  $("#scenario-search").addEventListener("input", (event) => renderScenarioLibrary(event.target.value));
  $("#library-new-scenario").addEventListener("click", () => { $("#scenario-library-dialog").close(); openScenarioBuilder(); });
  $("#scenario-library-list").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const scenarioId = button.dataset.openScenario || button.dataset.editScenario || button.dataset.copyScenario || button.dataset.deleteScenario;
    const scenario = state.scenarios.find((item) => item.scenario_id === scenarioId);
    if (!scenario) return;
    if (button.dataset.openScenario) {
      $("#scenario-library-dialog").close();
      selectScenario(scenarioId, { focus: true })
        .catch((error) => toast("Scenario could not be opened", error.message, true));
    } else if (button.dataset.editScenario) {
      $("#scenario-library-dialog").close();
      openScenarioBuilder(scenario);
    } else if (button.dataset.copyScenario) {
      $("#scenario-library-dialog").close();
      duplicateScenario(scenario);
    } else if (button.dataset.deleteScenario) {
      deleteScenario(scenario);
    }
  });
  ["#empty-upload", "#data-upload", "#start-upload", "#project-upload"].forEach((selector) => $(selector).addEventListener("click", () => {
    if ($("#project-dialog").open) $("#project-dialog").close();
    openUpload();
  }));
  $("#start-data-guide").addEventListener("click", () => openDataFormatGuide("formats"));
  $("#start-existing").addEventListener("click", changeProject);
  $("#open-project-picker").addEventListener("click", changeProject);
  $("#keep-current-project").addEventListener("click", async () => {
    if (!state.project?.project_id) return;
    if (state.project.retention !== "temporary") {
      toast("Project already saved", "This project will remain available in Open existing project.");
      return;
    }
    try {
      const kept = await api.keepProject(state.project.project_id);
      state.project = kept;
      state.projects = (await api.listProjects())
        .filter((project) => project.retention === "persistent")
        .sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")));
      const summary = kept.summary || {};
      $("#context-detail").textContent = `${formatNumber(summary.mapped_entity_count ?? summary.entity_count)} mapped entities · ${summary.time_start} to ${summary.time_end}`;
      toast("Project saved", "This project, its results and memory will remain after the platform stops.");
    } catch (error) {
      toast("Project could not be saved", error.message, true);
    }
  });
  $("#close-project-dialog").addEventListener("click", () => $("#project-dialog").close());
  $("#delete-all-projects").addEventListener("click", () => {
    deleteAllSavedProjects().catch((error) => toast("Saved projects could not be deleted", error.message, true));
  });
  $("#project-list").addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-project-id]");
    if (deleteButton) {
      const project = state.projects.find((item) => item.project_id === deleteButton.dataset.deleteProjectId);
      if (!project) return;
      const confirmed = window.confirm(`Delete "${project.name}" permanently?\n\nThis removes the saved project, its analysis results and its scenarios from this computer. This cannot be undone.`);
      if (!confirmed) return;
      deleteButton.disabled = true;
      api.deleteProject(project.project_id).then(() => {
        state.projects = state.projects.filter((item) => item.project_id !== project.project_id);
        if (state.project?.project_id === project.project_id) {
          $("#project-dialog").close();
          window.location.reload();
          return;
        }
        const remaining = state.projects.filter((item) => item.retention === "persistent");
        if (remaining.length) {
          changeProject().catch((error) => toast("Saved projects could not be refreshed", error.message, true));
        } else {
          $("#project-list").innerHTML = '<div class="project-list-empty"><strong>No saved projects</strong><span>Upload new data and choose Save project when you want to keep it.</span></div>';
        }
        toast("Project deleted", `"${project.name}" was permanently removed from this computer.`);
      }).catch((error) => {
        deleteButton.disabled = false;
        toast("Project could not be deleted", error.message, true);
      });
      return;
    }
    const button = event.target.closest("[data-project-id]");
    if (!button) return;
    const project = state.projects.find((item) => item.project_id === button.dataset.projectId);
    if (!project) return;
    $("#project-dialog").close();
    loadProject(project).catch((error) => toast("Project could not be opened", error.message, true));
  });
  $("#fit-map").addEventListener("click", () => state.map?.fitExtent());
  $("#view-points").addEventListener("click", () => {
    switchMapView("points").catch((error) => toast("Point view unavailable", error.message, true));
  });
  $("#view-links").addEventListener("click", () => {
    if (state.networkMapping?.ready) {
      switchMapView("links").catch((error) => toast("Link view unavailable", error.message, true));
    } else {
      openNetworkMappingDialog();
    }
  });
  $("#close-network-dialog").addEventListener("click", () => $("#network-dialog").close());
  $("#cancel-network-mapping").addEventListener("click", () => $("#network-dialog").close());
  $("#confirm-network-mapping").addEventListener("click", buildAndOpenNetworkView);
  $("#metric-select").addEventListener("change", async (event) => {
    stopPlayback();
    state.metric = event.target.value;
    invalidateAnalysisReview();
    try {
      await refreshMap(null);
      await refreshOpenInspector();
    } catch (error) {
      toast("Measure unavailable", error.message, true);
    }
  });
  $("#time-slider").addEventListener("input", (event) => {
    stopPlayback();
    state.timeIndex = Number(event.target.value);
    invalidateAnalysisReview();
    $("#time-label").textContent = currentTimeLabel();
    window.clearTimeout(state.viewportTimer);
    state.viewportTimer = window.setTimeout(async () => {
      try {
        await refreshMap(null);
        await refreshOpenInspector();
      } catch (error) {
        toast("Time could not be shown", error.message, true);
      }
    }, 160);
  });
  $("#play-time").addEventListener("click", togglePlayback);
  $("#playback-speed").addEventListener("change", (event) => {
    const requested = Number(event.target.value);
    state.playbackSpeed = PLAYBACK_PROFILES[requested] ? requested : 4;
    if (state.playing && state.playTimer) {
      window.clearTimeout(state.playTimer);
      state.playTimer = window.setTimeout(advancePlayback, playbackTiming().delay);
    }
  });
  $("#scope-whole").addEventListener("click", () => setAnalysisScopeMode("whole", { announce: true }));
  $("#scope-selection").addEventListener("click", () => setAnalysisScopeMode("selection", { announce: true }));
  $("#select-point").addEventListener("click", () => {
    state.map?.setMode("point");
    $("#select-point").classList.add("is-active");
    $("#select-area").classList.remove("is-active");
    setMapBanner(
      state.mapView === "links" ? "Mapped-link selection" : projectUsesLinkTopology() ? "Link-point selection" : "Point selection",
      state.mapView === "links"
        ? "Click an inferred line to inspect the link, its From / To nodes and historical values."
        : projectUsesLinkTopology()
        ? "Click a road-link reference point to see its From node, Road link and To node."
        : "Click an uploaded entity point to inspect its historical evidence."
    );
  });
  $("#select-area").addEventListener("click", () => {
    if (!state.project) return toast("Choose data first", "A geographic project is required for area selection.", true);
    state.map?.setMode("area");
    $("#select-area").classList.add("is-active");
    $("#select-point").classList.remove("is-active");
    setMapBanner("Draw a geographic area", "Press on the map, drag a rectangle and release.", true);
  });
  $("#clear-selection").addEventListener("click", clearSelection);
  $("#selected-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-remove-selection]");
    if (!button) return;
    const key = button.dataset.removeSelection;
    state.keptKeys.delete(key);
    state.selectedKeys.delete(key);
    if (state.focusedKey === key) {
      state.focusedKey = null;
      state.lastFeature = null;
      $("#entity-inspector").classList.add("is-hidden");
    }
    state.map?.setSelectedKeys([...state.selectedKeys]);
    if (state.analysisScopeMode === "selection") invalidateAnalysisReview("The map calculation scope changed.");
    renderSelectionSummary();
  });
  $("#close-inspector").addEventListener("click", () => $("#entity-inspector").classList.add("is-hidden"));
  $("#add-selected").addEventListener("click", () => {
    if (!state.lastFeature?.key) return;
    state.keptKeys.add(String(state.lastFeature.key));
    state.selectedKeys = new Set([...state.keptKeys, ...(state.focusedKey ? [state.focusedKey] : [])]);
    state.map.setSelectedKeys([...state.selectedKeys]);
    renderSelectionSummary();
    if (state.analysisScopeMode === "selection") invalidateAnalysisReview();
    toast("Added to scenario", "The selected road will be included in disruption or recovery analysis.");
  });
  $("#ask-about-selected").addEventListener("click", () => {
    setAnalysisScopeMode("selection", { invalidate: false });
    $("#command-input").value = "Assume the selected roads are closed. What is the topology effect?";
    interpretCurrentCommand();
  });
  $$("[data-command]").forEach((button) => button.addEventListener("click", () => {
    state.preferredModelId = null;
    $$("[data-command]").forEach((item) => item.classList.toggle("is-active", item === button));
    $("#command-input").value = button.dataset.command;
    invalidateAnalysisReview("A different analysis goal was selected.");
    interpretCurrentCommand();
  }));
  $("#command-input").addEventListener("input", () => {
    state.preferredModelId = null;
    $$("[data-command]").forEach((button) => button.classList.remove("is-active"));
    invalidateAnalysisReview("The analysis question changed.");
  });
  $("#interpret-command").addEventListener("click", interpretCurrentCommand);
  $("#edit-question").addEventListener("click", () => { $("#analysis-review").classList.add("is-hidden"); $("#command-input").focus(); });
  $("#confirm-analysis").addEventListener("click", runConfirmedAnalysis);
  $("#focus-result").addEventListener("click", () => focusResultOnMap());
  $("#result-visual").addEventListener("click", (event) => {
    const button = event.target.closest("[data-result-index]");
    if (button) focusResultItem(button.dataset.resultIndex);
  });
  $("#result-table-body").addEventListener("click", (event) => {
    const button = event.target.closest("[data-result-index]");
    if (button) focusResultItem(button.dataset.resultIndex);
  });
  $("#result-show-more").addEventListener("click", () => expandResultRows(false));
  $("#result-show-all").addEventListener("click", () => expandResultRows(true));
  $("#print-result").addEventListener("click", () => window.print());
  $("#cancel-analysis-wait").addEventListener("click", async () => {
    const jobId = state.activeAnalysisJobId;
    const cancelWait = state.analysisWaitCancel;
    if (typeof cancelWait === "function" && cancelWait()) {
      if (!jobId) {
        toast("Stopped waiting", "The current request was stopped before its result was placed on the map.");
        state.assistantPendingTask = null;
        state.assistantCompoundWorkflow = null;
        finishAssistantTurn();
        return;
      }
      try {
        await api.cancelJob(jobId);
        toast("Stopped waiting", "The in-process calculation will not be saved as a completed analysis.");
      } catch (error) {
        toast("Could not cancel the wait", error.message, true);
      }
      state.assistantPendingTask = null;
      state.assistantCompoundWorkflow = null;
      finishAssistantTurn();
      return;
    }
    if (!jobId) return;
    state.analysisWaitCancelled = true;
    state.activeAnalysisJobId = null;
    setLoading(false);
    try {
      await api.cancelJob(jobId);
      toast("Stopped waiting", "The in-process calculation will not be saved as a completed analysis.");
    } catch (error) {
      toast("Could not cancel the wait", error.message, true);
    }
    state.assistantPendingTask = null;
    state.assistantCompoundWorkflow = null;
    finishAssistantTurn();
  });
  $("#refresh-runs").addEventListener("click", refreshRuns);
  $("#refresh-memory").addEventListener("click", refreshProjectMemory);
  $("#export-memory").addEventListener("click", exportProjectMemory);
  $("#memory-timeline").addEventListener("click", (event) => {
    const button = event.target.closest("[data-memory-delete]");
    if (button) deleteMemoryEvent(button.dataset.memoryDelete, button.dataset.memoryType);
  });
  $("#saved-runs-body").addEventListener("click", (event) => {
    const button = event.target.closest("[data-open-run]");
    if (button) openSavedRun(button.dataset.openRun);
  });
  $("#model-grid").addEventListener("click", (event) => {
    const button = event.target.closest("[data-model-use]");
    if (button && !button.disabled) useModelCard(button.dataset.modelUse, button.dataset.modelGoal);
  });
  $("#run-project-validation").addEventListener("click", runProjectScientificValidation);
  $("#show-validation-detail").addEventListener("click", () => {
    const details = $("#validation-details");
    details.open = !details.open;
    $("#show-validation-detail").textContent = details.open ? "Hide scientific boundary" : "Show scientific boundary";
  });

  $("#close-upload").addEventListener("click", () => $("#upload-dialog").close());
  $("#close-data-format").addEventListener("click", () => $("#data-format-dialog").close());
  $("#upload-data-guide").addEventListener("click", () => openDataFormatGuide("formats"));
  $$('[data-data-guide-tab]').forEach((button) => button.addEventListener("click", () => setDataGuideTab(button.dataset.dataGuideTab)));
  $("#mapping-data-help").addEventListener("click", () => openDataFormatGuide(validateMappingStep() ? "formats" : "diagnosis"));
  $("#data-guide-upload").addEventListener("click", () => {
    $("#data-format-dialog").close();
    if (!$("#upload-dialog").open) openUpload();
    else showUploadStep(1);
    $("#upload-files").click();
  });
  $("#upload-files").addEventListener("change", renderChosenFiles);
  $("#preview-upload").addEventListener("click", previewUpload);
  $("#back-upload").addEventListener("click", () => showUploadStep(1));
  $("#confirm-upload").addEventListener("click", confirmUpload);
  $("#review-first-mapping").addEventListener("click", () => {
    const card = $(".file-mapping-card.needs-review") || $(".file-mapping-card");
    if (card) { card.open = true; card.scrollIntoView({ behavior: "smooth", block: "start" }); }
  });
  $("#mapping-grid").addEventListener("change", (event) => {
    if (event.target.matches("[data-mapping-role]")) {
      state.uploadPreviewError = null;
      validateMappingStep();
    }
  });
  const drop = $("#file-drop");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("is-dragging"); }));
  drop.addEventListener("drop", (event) => {
    const transfer = new DataTransfer();
    [...event.dataTransfer.files].forEach((file) => transfer.items.add(file));
    $("#upload-files").files = transfer.files;
    renderChosenFiles();
  });

  $("#assistant-launch").addEventListener("click", () => {
    const close = $("#assistant-panel").classList.contains("is-open") && state.assistantSurface === "platform";
    setAssistantOpen(!close, "platform");
  });
  bindLlmWindowInteractions();
  $("#llm-copilot-launch").addEventListener("click", () => {
    const close = $("#assistant-panel").classList.contains("is-open") && state.assistantSurface === "llm";
    setAssistantOpen(!close, "llm");
  });
  $("#assistant-close").addEventListener("click", () => setAssistantOpen(false));
  $("#assistant-new-chat").addEventListener("click", () => startNewAssistantChat());
  $("#assistant-new-chat-drawer").addEventListener("click", () => startNewAssistantChat());
  $("#assistant-chat-toggle").addEventListener("click", () => setAssistantChatDrawer(!state.assistantChatDrawerOpen));
  $("#assistant-chat-drawer-close").addEventListener("click", () => setAssistantChatDrawer(false));
  $("#assistant-chat-list").addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-assistant-chat-id]");
    if (deleteButton) {
      deleteAssistantChat(deleteButton.dataset.deleteAssistantChatId, deleteButton.dataset.chatTitle);
      return;
    }
    const chat = event.target.closest("[data-assistant-chat-id]");
    if (chat) loadAssistantChat(chat.dataset.assistantChatId);
  });
  $("#assistant-guide").addEventListener("click", () => { setAssistantOpen(false); openCapabilityTour(0); });
  $("#assistant-upload").addEventListener("click", () => openUpload());
  $("#assistant-settings").addEventListener("click", openAssistantSettings);
  $("#assistant-api-gate-connect").addEventListener("click", openAssistantSettings);
  $("#assistant-settings-close").addEventListener("click", () => $("#assistant-settings-dialog").close());
  $("#assistant-provider-cancel").addEventListener("click", () => $("#assistant-settings-dialog").close());
  $("#assistant-provider-save").addEventListener("click", () => configureAssistantProvider(false));
  $("#assistant-provider-disconnect").addEventListener("click", () => configureAssistantProvider(true));
  $("#assistant-provider-select").addEventListener("change", (event) => updateProviderForm(event.target.value));
  $("#provider-status-grid").addEventListener("click", (event) => {
    const card = event.target.closest("[data-provider-card]");
    if (card) {
      const provider = card.dataset.providerCard;
      updateProviderForm(provider);
      const details = (state.assistantStatus?.providers || []).find((item) => item.id === provider);
      if (details?.configured && details.catalog_status !== "verified") refreshAssistantModels(provider, true);
    }
  });
  $$('[data-assistant-mode]').forEach((button) => button.addEventListener("click", () => setAssistantMode(button.dataset.assistantMode)));
  $("#assistant-feature-directory").addEventListener("click", (event) => {
    const button = event.target.closest("[data-assistant-go]");
    if (button) navigateAssistantFeature(button.dataset.assistantGo).catch((error) => toast("Navigation unavailable", error.message, true));
  });
  $("#assistant-form").addEventListener("submit", (event) => { event.preventDefault(); sendAssistantMessage(); });
  $("#assistant-task-queue").addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-assistant-queue-id]");
    if (button) removeQueuedAssistantTask(button.dataset.deleteAssistantQueueId);
  });
  $("#assistant-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#assistant-form").requestSubmit();
    }
  });
  $("#assistant-actions").addEventListener("click", (event) => {
    const workflowButton = event.target.closest("[data-llm-workflow-action]");
    if (workflowButton) {
      executeAssistantWorkflowAction(workflowButton.dataset.llmWorkflowAction)
        .catch((error) => toast("Copilot step could not continue", error.message, true));
      return;
    }
    const button = event.target.closest("[data-assistant-action]");
    if (button) executeAssistantAction(currentAssistantActions()[Number(button.dataset.assistantAction)]).catch((error) => toast("Action unavailable", error.message, true));
  });
  $("#assistant-messages").addEventListener("click", (event) => {
    const countSubmit = event.target.closest("[data-llm-count-submit]");
    if (countSubmit) {
      const card = countSubmit.closest("[data-assistant-workflow-card='active']");
      const actionPrefix = countSubmit.dataset.llmCountSubmit;
      const input = card?.querySelector(`[data-llm-count-input="${actionPrefix}"]`);
      const value = Number(input?.value);
      const minimum = Number(input?.min || 1);
      const maximum = input?.max ? Number(input.max) : Number.POSITIVE_INFINITY;
      if (!Number.isInteger(value) || value < minimum || value > maximum) {
        const allowed = Number.isFinite(maximum) ? `between ${minimum} and ${maximum}` : `at least ${minimum}`;
        toast("Enter a whole number", `Choose a whole number ${allowed}.`, true);
        input?.focus();
        return;
      }
      executeAssistantWorkflowAction(`${actionPrefix}_${value}`)
        .catch((error) => toast("Copilot step could not continue", error.message, true));
      return;
    }
    const workflowButton = event.target.closest("[data-llm-workflow-action]");
    if (workflowButton) {
      executeAssistantWorkflowAction(workflowButton.dataset.llmWorkflowAction)
        .catch((error) => toast("Copilot step could not continue", error.message, true));
      return;
    }
    const runButton = event.target.closest("[data-assistant-run]");
    if (runButton) {
      runAssistantAnalysis(runButton);
      return;
    }
    if (event.target.closest("[data-assistant-focus-result]")) {
      setAssistantOpen(false);
      focusResultOnMap();
      return;
    }
    if (event.target.closest("[data-assistant-view-result]")) {
      setAssistantOpen(false);
      switchView("explore");
      window.setTimeout(() => highlightElement($("#result-panel")), 130);
      return;
    }
    if (event.target.closest("[data-assistant-focus-matsim], [data-assistant-view-matsim]")) {
      setAssistantOpen(false);
      switchView("explore");
      if (state.matsimFrames.length) {
        state.map?.focusCollection({ type: "FeatureCollection", features: state.matsimFrames.flatMap((frame) => frame.features || []) });
      }
      window.setTimeout(() => highlightElement($("#matsim-simulation-card")), 130);
    }
  });
  $("#assistant-messages").addEventListener("input", (event) => {
    const slider = event.target.closest("[data-llm-magnitude-slider]");
    if (!slider) return;
    const task = state.assistantPendingTask;
    const type = slider.dataset.llmMagnitudeSlider;
    if (!task || task.type !== type || task.phase !== "magnitude") return;
    const percentage = Math.max(Number(slider.min), Math.min(Number(slider.max), Number(slider.value)));
    task.magnitude = percentage / 100;
    const card = slider.closest("[data-assistant-workflow-card='active']");
    const output = card?.querySelector("[data-llm-magnitude-output]");
    const confirm = card?.querySelector("[data-llm-magnitude-confirm]");
    if (output) output.textContent = `${percentage}%`;
    if (confirm) {
      confirm.dataset.llmWorkflowAction = `${type}_magnitude_${percentage}`;
      confirm.textContent = `Confirm ${percentage}%`;
    }
    const platformSlider = type === "route" ? $("#route-magnitude") : $("#scenario-magnitude");
    const platformOutput = type === "route" ? $("#route-magnitude-output") : $("#scenario-magnitude-output");
    if (platformSlider) platformSlider.value = String(percentage);
    if (platformOutput) platformOutput.textContent = `${percentage}%`;
  });
  $("#assistant-messages").addEventListener("keydown", (event) => {
    const input = event.target.closest("[data-llm-count-input]");
    if (!input || event.key !== "Enter") return;
    event.preventDefault();
    input.closest("[data-assistant-workflow-card='active']")
      ?.querySelector(`[data-llm-count-submit="${input.dataset.llmCountInput}"]`)
      ?.click();
  });
  $("#assistant-suggestions").addEventListener("click", (event) => {
    const button = event.target.closest("[data-assistant-prompt]");
    if (button) sendAssistantMessage(button.dataset.assistantPrompt);
  });
  $("#assistant-clear").addEventListener("click", () => {
    if (state.assistantSurface === "llm") startNewAssistantChat();
    else resetAssistantConversationView();
  });
  $("#close-guide").addEventListener("click", () => $("#guide-dialog").close());
  $("#guide-previous").addEventListener("click", () => { state.guideIndex = Math.max(0, state.guideIndex - 1); renderGuideStep(); });
  $("#guide-next").addEventListener("click", () => {
    if (state.guideIndex >= GUIDE_STEPS.length - 1) return $("#guide-dialog").close();
    state.guideIndex += 1;
    renderGuideStep();
  });
  $("#guide-show").addEventListener("click", showGuideTarget);
  $("#open-capability-tour").addEventListener("click", openCapabilityTour);
  $("#open-capability-status").addEventListener("click", () => $("#capability-status-dialog").showModal());
  $("#close-capability-status").addEventListener("click", () => $("#capability-status-dialog").close());
  $("#capability-status-done").addEventListener("click", () => $("#capability-status-dialog").close());
  $("#close-capability-tour").addEventListener("click", closeCapabilityTour);
  $("#close-workflow-video").addEventListener("click", closeCapabilityTour);
  $("#restart-workflow-video").addEventListener("click", () => {
    const video = $("#platform-workflow-video");
    video.currentTime = 0;
    video.play().catch(() => {});
  });
  $("#capability-dialog").addEventListener("close", () => $("#platform-workflow-video").pause());
  $("#open-contributors").addEventListener("click", () => $("#contributors-dialog").showModal());
  $("#close-contributors").addEventListener("click", () => $("#contributors-dialog").close());
  $("#contributors-done").addEventListener("click", () => $("#contributors-dialog").close());

  document.addEventListener("pointerdown", (event) => {
    const target = event.target.closest(".ripple");
    if (!target || target.disabled) return;
    const rectangle = target.getBoundingClientRect();
    const ink = document.createElement("span");
    const size = Math.max(rectangle.width, rectangle.height) * 1.8;
    ink.className = "ripple-ink";
    ink.style.width = `${size}px`;
    ink.style.height = `${size}px`;
    ink.style.left = `${event.clientX - rectangle.left}px`;
    ink.style.top = `${event.clientY - rectangle.top}px`;
    target.append(ink);
    ink.addEventListener("animationend", () => ink.remove());
  });
  window.addEventListener("resize", () => state.map?.resize());
}

initialise();
