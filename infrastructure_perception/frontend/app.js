const requestedStage7Candidate = new URLSearchParams(window.location.search).get("stage7_candidate");
const stage7CandidateModels = ["r07", "surface_precision"];
stage7CandidateModels.push("surface_precision_v021", "classless_v022");
const stage7CandidateModel = stage7CandidateModels.includes(requestedStage7Candidate)
  ? requestedStage7Candidate
  : null;
function verifiedStage7Mode(mode) {
  // The hash-verified registry is the model contract, not a stale UI version literal.
  return Boolean(mode?.ready === true && mode.model_version
    && /^[a-f0-9]{64}$/i.test(mode.checkpoint_sha256 || "")
    && Object.keys(mode.checks || {}).length > 0
    && Object.values(mode.checks).every(value => value === true));
}

const state = {
  map: null,
  mapReady: false,
  googleConnected: false,
  googleApiKey: null,
  stage7InferenceMode: "local",
  stage7VisionApiKey: null,
  stage7LocalReady: false,
  stage7EnhancedReady: false,
  googleLoading: false,
  googleAuthFailed: false,
  googleLoadFailed: false,
  placesAutocomplete: null,
  satelliteOverlay: null,
  streetViewService: null,
  streetViewPanorama: null,
  streetProvider: null,
  streetDisplayMode: false,
  mapTypeBeforeStreet: "roadmap",
  mapNoticeTimer: null,
  images: [],
  markers: new Map(),
  selectedId: null,
  sequence: [],
  sequenceIndex: -1,
  placementId: null,
  clickMarker: null,
  aoi: null,
  scenes: [],
  activeScene: null,
  comparison: { a: null, b: null },
  streetSearchMode: false,
  streetResults: [],
  streetSequence: [],
  streetIndex: -1,
  streetSelectedId: null,
  streetMarker: null,
  streetCoverageLayer: null,
  selectedRoadPoint: null,
  stage7Status: null,
  stage7Analysis: null,
  stage8Status: null,
  stage8Queue: { records: [], summary: {} },
  stage8Review: null,
  stage8LastSavedReview: null,
  stage7RequestToken: 0,
  stage7RequestController: null,
  stage7RunIntent: 0,
  stage7FrameRevision: 0,
  stage7CurrentFrameId: null,
  stage7MaterialCameraSignature: null,
  locationLabel: "Default map view",
  defaultMap: { longitude: 77.4126, latitude: 23.2599, zoom: 12 },
  aiContract: null,
  datasetContract: null,
  modelReport: null,
  validationReport: null,
  validationV2Report: null,
  stage6Status: null,
  stage6Activity: { observations: [], inference_runs: [] },
  stage6CurrentObservations: [],
};

function stage7AnalyseEndpoint() {
  return stage7CandidateModel
    ? `api/stage7/views/analyse?candidate_model=${encodeURIComponent(stage7CandidateModel)}`
    : "api/stage7/views/analyse";
}

const COPERNICUS_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1/";
const COPERNICUS_STAC_SEARCH = `${COPERNICUS_STAC_ROOT}search`;

const $ = (selector) => document.querySelector(selector);
const elements = {
  workspaceGrid: $(".workspace-grid"), resultsPanel: $("#results-panel"), sceneResultsCard: $("#scene-results-card"), streetResultsCard: $("#street-results-card"), groundResultsCard: $("#ground-results-card"), comparisonPanel: $("#comparison-panel"), groundViewer: $("#ground-viewer"),
  aiMethodButton: $("#ai-method-button"), aiMethodStageStatus: $("#ai-method-stage-status"), stage5BannerTitle: $("#stage5-banner-title"), stage5BannerDetail: $("#stage5-banner-detail"), aiMethodDialog: $("#ai-method-dialog"), closeAiMethod: $("#close-ai-method"), aiContractObjective: $("#ai-contract-objective"), aiContractStatus: $("#ai-contract-status"), aiSurfaceList: $("#ai-surface-list"), aiDefectList: $("#ai-defect-list"), aiSourcePolicies: $("#ai-source-policies"), aiOutputFields: $("#ai-output-fields"), aiWorkflow: $("#ai-workflow"), runContractChecks: $("#run-contract-checks"), contractCheckSummary: $("#contract-check-summary"), contractCheckResults: $("#contract-check-results"), aiContractLegalNote: $("#ai-contract-legal-note"), aiStage8TrainingUnits: $("#ai-stage8-training-units"), aiStage8TrainingAnnotations: $("#ai-stage8-training-annotations"), aiStage8PartialUnits: $("#ai-stage8-partial-units"), aiStage8PendingAnnotations: $("#ai-stage8-pending-annotations"),




  connectGoogleImagery: $("#connect-google-imagery"), connectGoogleFromMap: $("#connect-google-from-map"), googleConnectionStatus: $("#google-connection-status"), googleConnectDialog: $("#google-connect-dialog"), googleConnectForm: $("#google-connect-form"), googleApiKey: $("#google-api-key"), googleConnectStatus: $("#google-connect-status"), submitGoogleConnect: $("#submit-google-connect"), cancelGoogleConnect: $("#cancel-google-connect"), mapConnectionPlaceholder: $("#map-connection-placeholder"),
  stage7ModeLabel: $("#stage7-mode-label"), stage7ModeDetail: $("#stage7-mode-detail"), connectStage7Vision: $("#connect-stage7-vision"), disableStage7Vision: $("#disable-stage7-vision"), stage7VisionDialog: $("#stage7-vision-dialog"), stage7VisionForm: $("#stage7-vision-form"), stage7VisionApiKey: $("#stage7-vision-api-key"), stage7VisionStatus: $("#stage7-vision-status"), submitStage7Vision: $("#submit-stage7-vision"), cancelStage7Vision: $("#cancel-stage7-vision"),
  locationForm: $("#location-form"), locationQuery: $("#location-query"), googlePlaceAutocompleteHost: $("#google-place-autocomplete-host"), locationStatus: $("#location-status"), locationResults: $("#location-results"),
  satelliteForm: $("#satellite-form"), collectionInput: $("#collection-input"), startDate: $("#start-date"), endDate: $("#end-date"), cloudField: $("#cloud-field"), cloudCover: $("#cloud-cover"), cloudValue: $("#cloud-value"), searchSatellite: $("#search-satellite"), satelliteStatus: $("#satellite-status"), sceneCount: $("#scene-count"), sceneCaption: $("#scene-caption"), sceneList: $("#scene-list"),
  startStreetSearch: $("#start-street-search"), streetRadius: $("#street-radius"), streetRadiusValue: $("#street-radius-value"), streetStatus: $("#street-status"), streetCount: $("#street-count"), streetCaption: $("#street-caption"), streetList: $("#street-list"),
  kartaStreetStage: $("#karta-street-stage"), streetViewerImage: $("#street-viewer-image"), stage7StreetOverlay: $("#stage7-street-overlay"), street360Badge: $("#street-360-badge"), streetInlineBar: $("#street-inline-bar"), streetViewerProvider: $("#street-viewer-provider"), streetViewerTime: $("#street-viewer-time"), streetViewerCoordinates: $("#street-viewer-coordinates"), streetViewerDistance: $("#street-viewer-distance"), streetViewerSource: $("#street-viewer-source"), streetViewerNavigation: $("#street-viewer-navigation"), streetViewerPosition: $("#street-viewer-position"), previousStreetImage: $("#previous-street-image"), nextStreetImage: $("#next-street-image"),
  stage7AnalysisCard: $("#stage7-analysis-card"), stage7AnalysisTitle: $("#stage7-analysis-title"), stage7AnalysisBadge: $("#stage7-analysis-badge"), stage7AnalysisPoint: $("#stage7-analysis-point"), stage7AnalysisMessage: $("#stage7-analysis-message"), stage7AnalysisImageWrap: $("#stage7-analysis-image-wrap"), stage7AnalysisImage: $("#stage7-analysis-image"), stage7AnalysisOverlay: $("#stage7-analysis-overlay"), stage7AnalysisDetails: $("#stage7-analysis-details"), stage7AnalysisReasons: $("#stage7-analysis-reasons"), stage7AnalysisSource: $("#stage7-analysis-source"), retryStage7Analysis: $("#retry-stage7-analysis"),
  reviewStage8Annotations: $("#review-stage8-annotations"), stage8ReviewStatusCard: $("#stage8-review-status-card"), stage8ReviewStatusTitle: $("#stage8-review-status-title"), stage8ReviewStatusBadge: $("#stage8-review-status-badge"), stage8ReviewStatusMessage: $("#stage8-review-status-message"), stage8IssueSwitcher: $("#stage8-issue-switcher"), stage8IssueTabs: $("#stage8-issue-tabs"), stage8OriginalOverlay: $("#stage8-original-overlay"), stage8EditorOverlay: $("#stage8-editor-overlay"), stage8MapOriginalOverlay: $("#stage8-map-original-overlay"), stage8MapEditorOverlay: $("#stage8-map-editor-overlay"), stage8MapReviewBanner: $("#stage8-map-review-banner"), stage8InlineEditor: $("#stage8-inline-editor"), stage8InlineClose: $("#stage8-inline-close"), stage8EditorPanel: $("#stage8-editor-panel"), stage8EditorState: $("#stage8-editor-state"), stage8AddBox: $("#stage8-add-box"), stage8MarkClean: $("#stage8-mark-clean"), stage8SelectionSummary: $("#stage8-selection-summary"), stage8SelectedControls: $("#stage8-selected-controls"), stage8Surface: $("#stage8-surface"), stage8DefectClass: $("#stage8-defect-class"), stage8ConfirmBox: $("#stage8-confirm-box"), stage8DeleteBox: $("#stage8-delete-box"), stage8ReviewNote: $("#stage8-review-note"), stage8EditorMessage: $("#stage8-editor-message"), stage8Undo: $("#stage8-undo"), stage8Reset: $("#stage8-reset"), stage8Cancel: $("#stage8-cancel"), stage8Save: $("#stage8-save"), stage8QueuePanel: $("#stage8-queue-panel"), stage8QueueTotal: $("#stage8-queue-total"), stage8QueueSummary: $("#stage8-queue-summary"), stage8QueueRecords: $("#stage8-queue-records"), stage8RefreshQueue: $("#stage8-refresh-queue"),
  mapRoad: $("#map-road"), mapSatellite: $("#map-satellite"), mapStreet: $("#map-street-view"), backToMap: $("#back-to-map"), mapNotice: $("#map-notice"), stage7GoogleOverlay: $("#stage7-google-overlay"), groundLayerToggle: $("#ground-layer-toggle"), streetLayerToggle: $("#street-layer-toggle"), satelliteLayerToggle: $("#satellite-layer-toggle"), satelliteOpacity: $("#satellite-opacity"), opacityValue: $("#opacity-value"), mapTitle: $("#map-title"), mapInstruction: $("#map-instruction"),
  comparisonA: $("#comparison-a"), comparisonB: $("#comparison-b"), projectName: $("#project-name"), saveProject: $("#save-project"), reopenProject: $("#reopen-project"), clearProject: $("#clear-project"), projectStatus: $("#project-status"),
  uploadForm: $("#upload-form"), imageInput: $("#image-input"), sourceInput: $("#source-input"), analysisAuthorised: $("#analysis-authorised"), selectionSummary: $("#selection-summary"), uploadStatus: $("#upload-status"), catalogueList: $("#catalogue-list"), totalCount: $("#total-count"), locatedCount: $("#located-count"), unlocatedCount: $("#unlocated-count"),
  viewerEmpty: $("#viewer-empty"), viewerContent: $("#viewer-content"), viewerImage: $("#viewer-image"), viewerMethod: $("#viewer-method"), viewerName: $("#viewer-name"), viewerSource: $("#viewer-source"), viewerTime: $("#viewer-time"), viewerCoordinates: $("#viewer-coordinates"), viewerDirection: $("#viewer-direction"), viewerSize: $("#viewer-size"), viewerDistance: $("#viewer-distance"), placementAction: $("#placement-action"), manualLatitude: $("#manual-latitude"), manualLongitude: $("#manual-longitude"), saveManualLocation: $("#save-manual-location"), placeImage: $("#place-image"), deleteImage: $("#delete-image"), previousImage: $("#previous-image"), nextImage: $("#next-image"), viewerPosition: $("#viewer-position"),
  detectionOverlay: $("#detection-overlay"), stage6RuntimeBadge: $("#stage6-runtime-badge"), stage6ActiveModel: $("#stage6-active-model"), stage6PackageState: $("#stage6-package-state"), stage6AuthorisationAction: $("#stage6-authorisation-action"), stage6AuthoriseCheckbox: $("#stage6-authorise-checkbox"), stage6AuthoriseImage: $("#stage6-authorise-image"), stage6Surface: $("#stage6-surface"), runStage6Inference: $("#run-stage6-inference"), runStage6SelfTest: $("#run-stage6-self-test"), stage6StatusMessage: $("#stage6-status"), stage6Detections: $("#stage6-detections"), stage6Reviewer: $("#stage6-reviewer"), stage6ReviewNote: $("#stage6-review-note"), refreshStage6History: $("#refresh-stage6-history"), stage6HistorySummary: $("#stage6-history-summary"), stage6History: $("#stage6-history"),
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload && typeof payload === "object" ? payload.detail : payload;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : detail;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return payload;
}

function setStatus(element, message, kind = "") {
  element.textContent = message;
  element.className = `status-message ${kind}`.trim();
}

function updateProgressiveLayout() {
  const hasResults = !elements.sceneResultsCard.hidden || !elements.streetResultsCard.hidden || !elements.groundResultsCard.hidden;
  elements.resultsPanel.hidden = !hasResults;
  elements.workspaceGrid.classList.toggle("has-results", hasResults);
}

function renderStage5Header(validationV2 = {}, stage6 = {}, stage7 = {}) {
  if (stage7.status === "on_demand_runtime_ready") {
    const proof = stage7.target_area_detection_proof_confirmed === true;
    elements.aiMethodStageStatus.textContent = "Standard detection ready · optional OpenAI verification";
    elements.stage5BannerTitle.textContent = proof
      ? "Road condition analysis is ready"
      : "Road condition analysis is being validated";
    elements.stage5BannerDetail.textContent = "Standard detection requires only Google imagery. You may optionally use your own OpenAI key to check road, kerb and footpath results; Platform Assistant credentials are never reused.";
    return;
  }
  if (stage6.status === "research_deployment_ready") {
    const model = stage6.active_model?.model_version || "verified model package";
    elements.aiMethodStageStatus.textContent = `Uploaded-image analysis ready · ${model}`;
    elements.stage5BannerTitle.textContent = "Human-reviewed uploaded-image analysis is ready";
    elements.stage5BannerDetail.textContent = "Authorised uploaded imagery can now run through the hash-verified active model package. Verified location and named human review remain mandatory; raw confidence cannot trigger an automatic operational action.";
    return;
  }
  elements.aiMethodStageStatus.textContent = "Compatible model package required";
  elements.stage5BannerTitle.textContent = "Image analysis is not configured";
  elements.stage5BannerDetail.textContent = "This source release does not include model weights or research data. Install an authorised compatible model package before running detection.";
}

function revealSceneResults() {
  elements.sceneResultsCard.hidden = false;
  updateProgressiveLayout();
}

function revealStreetResults() {
  elements.streetResultsCard.hidden = false;
  updateProgressiveLayout();
}

function revealComparison() {
  elements.comparisonPanel.hidden = false;
}

function policyLabel(value) {
  return String(value || "unknown").replaceAll("_", " ");
}

function renderAiContract(contract) {
  state.aiContract = contract;
  elements.aiContractObjective.textContent = contract.scope.objective;
  elements.aiContractStatus.textContent = `Contract ready · v${contract.contract_version}`;
  elements.aiContractLegalNote.textContent = `${contract.legal_note} Policy reviewed ${contract.last_reviewed}.`;

  elements.aiSurfaceList.innerHTML = contract.surfaces
    .map((surface) => `<span class="contract-chip">${escapeHtml(surface.label)}</span>`)
    .join("");

  const surfaceLabels = Object.fromEntries(contract.surfaces.map((surface) => [surface.id, surface.label]));
  elements.aiDefectList.innerHTML = contract.defects
    .map((defect) => `<div class="contract-defect"><strong>${escapeHtml(defect.label)}</strong><small>${escapeHtml(defect.applicable_surfaces.map((id) => surfaceLabels[id] || id).join(" · "))}</small></div>`)
    .join("");

  elements.aiSourcePolicies.innerHTML = contract.source_policies
    .map((source) => {
      const name = source.official_terms_url
        ? `<a href="${escapeHtml(source.official_terms_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.label)} &nearr;</a>`
        : escapeHtml(source.label);
      return `<tr><td>${name}</td><td><span class="policy-pill ${escapeHtml(source.training)}">${escapeHtml(policyLabel(source.training))}</span></td><td><span class="policy-pill ${escapeHtml(source.automated_inference)}">${escapeHtml(policyLabel(source.automated_inference))}</span></td><td>${escapeHtml(source.role)}</td></tr>`;
    })
    .join("");

  elements.aiOutputFields.innerHTML = contract.required_output_fields
    .map((field) => `<span class="contract-output-field">${escapeHtml(field.label)}</span>`)
    .join("");
  elements.aiWorkflow.innerHTML = contract.future_user_workflow
    .map((step) => `<li>${escapeHtml(step)}</li>`)
    .join("");
}

async function loadAiContract() {
  if (state.aiContract) return state.aiContract;
  const contract = await request("api/ai-contract");
  renderAiContract(contract);
  return contract;
}

async function showAiMethodDialog() {
  if (typeof elements.aiMethodDialog.showModal === "function") elements.aiMethodDialog.showModal();
  else elements.aiMethodDialog.setAttribute("open", "");
  try {
    await loadAiContract();
    renderAiMethodStage8Queue(state.stage8Queue.summary || {});
  } catch (error) {
    elements.aiContractObjective.textContent = `Source policy could not load: ${error.message}`;
  }
}

function closeAiMethodDialog() {
  if (typeof elements.aiMethodDialog.close === "function") elements.aiMethodDialog.close();
  else elements.aiMethodDialog.removeAttribute("open");
}

async function runContractChecks() {
  elements.runContractChecks.disabled = true;
  elements.runContractChecks.textContent = "Running checks...";
  elements.contractCheckSummary.textContent = "Checking labels, confidence, provenance, source policy and scientific claims.";
  elements.contractCheckResults.innerHTML = "";
  try {
    const payload = await request("api/ai-contract/checks", { method: "POST" });
    elements.contractCheckSummary.textContent = payload.all_passed
      ? `${payload.passed}/${payload.total} checks passed · source rules verified`
      : `${payload.passed}/${payload.total} checks passed · contract is not ready`;
    elements.contractCheckResults.innerHTML = payload.checks
      .map((check) => `<div class="contract-check-result ${check.passed ? "passed" : "failed"}"><i>${check.passed ? "✓" : "!"}</i><span>${escapeHtml(check.label)}</span></div>`)
      .join("");
  } catch (error) {
    elements.contractCheckSummary.textContent = `Contract checks failed to run: ${error.message}`;
  } finally {
    elements.runContractChecks.disabled = false;
    elements.runContractChecks.textContent = "Run contract checks";
  }
}

function isoDate(date) { return date.toISOString().slice(0, 10); }
function shortDate(value) {
  if (!value) return "Date not reported";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("en-AU", { dateStyle: "medium", timeStyle: "short" });
}
function coordinateText(latitude, longitude) {
  return latitude == null || longitude == null ? "Not located" : `${Number(latitude).toFixed(6)}, ${Number(longitude).toFixed(6)}`;
}

function initialiseDates() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 180);
  elements.endDate.value = isoDate(end);
  elements.startDate.value = isoDate(start);
}

function setMapControlsEnabled(enabled) {
  [elements.locationQuery, elements.startStreetSearch, elements.streetRadius, elements.mapRoad, elements.mapSatellite, elements.mapStreet, elements.groundLayerToggle, elements.streetLayerToggle, elements.satelliteLayerToggle, elements.satelliteOpacity, elements.searchSatellite]
    .forEach((element) => { element.disabled = !enabled; });
  elements.startStreetSearch.textContent = enabled ? "Choose a road point" : "Connect Google imagery first";
}

function showGoogleConnectDialog() {
  if (state.googleConnected || state.googleLoading) return;
  setStatus(elements.googleConnectStatus, "The key will be used only to load Google services in this tab.");
  if (typeof elements.googleConnectDialog.showModal === "function") elements.googleConnectDialog.showModal();
  else elements.googleConnectDialog.setAttribute("open", "");
  window.setTimeout(() => elements.googleApiKey.focus(), 0);
}

function closeGoogleConnectDialog() {
  elements.googleApiKey.value = "";
  if (typeof elements.googleConnectDialog.close === "function") elements.googleConnectDialog.close();
  else elements.googleConnectDialog.removeAttribute("open");
}

function renderStage7Mode() {
  const enhanced = state.stage7InferenceMode === "enhanced" && Boolean(state.stage7VisionApiKey);
  const shell = elements.stage7ModeLabel.closest(".stage7-mode-control");
  shell.classList.toggle("enhanced", enhanced);
  elements.stage7ModeLabel.textContent = enhanced ? "Standard + OpenAI" : "Standard detection";
  elements.stage7ModeDetail.textContent = enhanced
    ? "Local detectors plus visual verification using your key. Platform Assistant remains separate."
    : "Uses the existing local detectors without an OpenAI key.";
  elements.connectStage7Vision.textContent = enhanced ? "Change OpenAI key" : "Improve this view with OpenAI";
  elements.disableStage7Vision.classList.toggle("hidden", !enhanced);
}

function showStage7VisionDialog() {
  if (!state.stage7EnhancedReady) {
    setStatus(elements.streetStatus, "Optional OpenAI verification is not ready. Standard detection remains available.", "warning");
    return;
  }
  setStatus(elements.stage7VisionStatus, "The key will be held only in this road-perception tab and used only for optional visual verification.");
  if (typeof elements.stage7VisionDialog.showModal === "function") elements.stage7VisionDialog.showModal();
  else elements.stage7VisionDialog.setAttribute("open", "");
  window.setTimeout(() => elements.stage7VisionApiKey.focus(), 0);
}

function closeStage7VisionDialog() {
  elements.stage7VisionApiKey.value = "";
  if (typeof elements.stage7VisionDialog.close === "function") elements.stage7VisionDialog.close();
  else elements.stage7VisionDialog.removeAttribute("open");
}

async function connectStage7Vision(event) {
  event?.preventDefault?.();
  const apiKey = elements.stage7VisionApiKey.value.trim();
  elements.stage7VisionApiKey.value = "";
  if (apiKey.length < 20) {
    setStatus(elements.stage7VisionStatus, "Enter a complete OpenAI API key or keep using local models.", "warning");
    return;
  }
  state.stage7VisionApiKey = apiKey;
  state.stage7InferenceMode = "enhanced";
  renderStage7Mode();
  closeStage7VisionDialog();
  invalidateGoogleStage7Frame("inference_mode_changed");
  if (state.googleConnected && state.streetProvider === "google" && state.streetViewPanorama?.getPano?.()) {
    setStatus(elements.streetStatus, "OpenAI verification is enabled. Re-analysing the current image with local detection and OpenAI...", "loading");
    await runStage7CurrentViewAnalysis();
  } else {
    setStatus(elements.streetStatus, "OpenAI verification is enabled for this tab. Choose a road image to use local detection and OpenAI together.", "success");
  }
}

function disableStage7Vision() {
  state.stage7VisionApiKey = null;
  state.stage7InferenceMode = "local";
  renderStage7Mode();
  invalidateGoogleStage7Frame("inference_mode_changed");
  setStatus(elements.streetStatus, "Standard detection is active. No OpenAI credential will be sent.", "success");
}

function loadGoogleMapsForTab(apiKey) {
  if (window.google?.maps?.importLibrary) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const callbackName = "__twmGoogleMapsReady";
    const script = document.createElement("script");
    let settled = false;
    const finish = (error = null) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      script.remove();
      delete window[callbackName];
      if (error) reject(error); else resolve();
    };
    const timeoutId = window.setTimeout(() => finish(new Error("Google Maps did not finish loading. Check the key restrictions and internet connection.")), 20000);
    window[callbackName] = () => finish();
    window.gm_authFailure = () => {
      state.googleAuthFailed = true;
      if (!settled) finish(new Error("Google rejected this browser key. Check the enabled APIs, billing and localhost referrer restrictions."));
      if (state.googleConnected) handleGoogleAuthorisationFailure();
    };
    script.id = "twm-google-maps-loader";
    script.async = true;
    script.defer = true;
    script.referrerPolicy = "strict-origin-when-cross-origin";
    script.onerror = () => finish(new Error("Google Maps could not be loaded. Check the internet connection and browser-key restrictions."));
    const parameters = new URLSearchParams({ key: apiKey, v: "weekly", loading: "async", libraries: "places", callback: callbackName, auth_referrer_policy: "origin" });
    script.src = `https://maps.googleapis.com/maps/api/js?${parameters.toString()}`;
    document.head.appendChild(script);
  });
}

function handleGoogleAuthorisationFailure() {
  state.googleConnected = false;
  state.googleApiKey = null;
  state.mapReady = false;
  setMapControlsEnabled(false);
  elements.mapConnectionPlaceholder.classList.remove("hidden");
  elements.connectGoogleImagery.classList.remove("connected");
  elements.connectGoogleImagery.querySelector("strong").textContent = "Connect Google imagery";
  elements.googleConnectionStatus.textContent = "Google rejected the key · reload to retry";
  elements.googlePlaceAutocompleteHost.classList.add("hidden");
  elements.locationQuery.classList.remove("hidden");
  setStatus(elements.googleConnectStatus, "Google rejected this browser key. Reload the page, then check the enabled APIs, billing and localhost referrer restrictions before retrying.", "error");
}

async function initialiseMap() {
  const { Map: GoogleMap } = await google.maps.importLibrary("maps");
  const { StreetViewCoverageLayer, StreetViewService } = await google.maps.importLibrary("streetView");
  const { PlaceAutocompleteElement } = await google.maps.importLibrary("places");
  state.map = new GoogleMap(document.querySelector("#map"), {
    center: { lat: state.defaultMap.latitude, lng: state.defaultMap.longitude },
    zoom: state.defaultMap.zoom,
    mapTypeId: google.maps.MapTypeId.ROADMAP,
    mapTypeControl: false,
    streetViewControl: false,
    fullscreenControl: true,
    clickableIcons: false,
    gestureHandling: "auto",
  });
  state.streetViewService = new StreetViewService();
  state.streetCoverageLayer = new StreetViewCoverageLayer();
  state.streetCoverageLayer.setMap(elements.streetLayerToggle.checked ? state.map : null);
  state.placesAutocomplete = new PlaceAutocompleteElement();
  state.placesAutocomplete.placeholder = "Search city, road or place";
  state.placesAutocomplete.addEventListener("gmp-select", async (event) => {
    try {
      const place = event.placePrediction.toPlace();
      await place.fetchFields({ fields: ["displayName", "formattedAddress", "location", "viewport"] });
      selectGooglePlace(place);
    } catch (_) {
      setStatus(elements.locationStatus, "Google Places could not open this selection. Try another result.", "error");
    }
  });
  state.placesAutocomplete.addEventListener("gmp-error", () => setStatus(elements.locationStatus, "Google Places search is unavailable for this key. Check that Places API (New) is enabled.", "error"));
  elements.googlePlaceAutocompleteHost.replaceChildren(state.placesAutocomplete);
  elements.googlePlaceAutocompleteHost.classList.remove("hidden");
  elements.locationQuery.classList.add("hidden");
  state.map.addListener("bounds_changed", () => {
    const bounds = state.map.getBounds();
    if (bounds && state.placesAutocomplete) state.placesAutocomplete.locationBias = bounds;
  });
  state.map.addListener("idle", () => { state.aoi = localAreaFromMapCentre(); });
  state.map.addListener("click", handleMapClick);
  state.streetViewPanorama = state.map.getStreetView();
  state.streetViewPanorama.setOptions({
    addressControl: true,
    clickToGo: true,
    scrollwheel: true,
    showRoadLabels: true,
    visible: false,
  });
  state.streetViewPanorama.addListener("pano_changed", async () => {
    if (state.streetProvider !== "google" || !state.streetDisplayMode) return;
    invalidateGoogleStage7FrameIfMaterial("panorama_changed");
    const pano = state.streetViewPanorama.getPano();
    if (!pano) return;
    try {
      const response = await state.streetViewService.getPanorama({ pano });
      if (state.streetProvider === "google" && state.streetViewPanorama.getPano() === pano) applyGoogleStreetMetadata(response.data);
    } catch (_) {
      const position = state.streetViewPanorama.getPosition();
      if (position) applyGoogleStreetMetadata({ location: { latLng: position } });
    }
  });
  state.streetViewPanorama.addListener("position_changed", () => {
    if (state.streetProvider === "google" && state.streetDisplayMode) invalidateGoogleStage7FrameIfMaterial("position_changed");
  });
  state.streetViewPanorama.addListener("pov_changed", () => {
    if (state.streetProvider === "google" && state.streetDisplayMode) invalidateGoogleStage7FrameIfMaterial("camera_changed");
  });
  state.streetViewPanorama.addListener("zoom_changed", () => {
    if (state.streetProvider === "google" && state.streetDisplayMode) invalidateGoogleStage7FrameIfMaterial("zoom_changed");
  });
  state.mapReady = true;
  renderMarkers();
  state.aoi = localAreaFromMapCentre();
}

async function connectGoogleImagery(event) {
  event?.preventDefault?.();
  if (state.googleLoadFailed) {
    window.location.reload();
    return;
  }
  const apiKey = elements.googleApiKey.value.trim();
  elements.googleApiKey.value = "";
  if (apiKey.length < 20) {
    setStatus(elements.googleConnectStatus, "Enter the complete restricted Google Maps browser key.", "warning");
    return;
  }
  state.googleLoading = true;
  state.googleAuthFailed = false;
  elements.submitGoogleConnect.disabled = true;
  setStatus(elements.googleConnectStatus, "Connecting Google Maps, Places and Street View...", "loading");
  try {
    const stage7Status = await request("api/stage7");
    const localMode = stage7Status.inference_modes?.local || {};
    const enhancedMode = stage7Status.inference_modes?.enhanced || {};
    if (stage7Status.status !== "on_demand_runtime_ready" || localMode.ready !== true) {
      throw new Error("The standard road-detection runtime is not ready. Google connection was stopped so a model failure cannot be hidden.");
    }
    state.stage7Status = stage7Status;
    state.stage7LocalReady = true;
    state.stage7EnhancedReady = Boolean(
      enhancedMode.ready === true
      && verifiedStage7Mode(enhancedMode)
    );
    await loadGoogleMapsForTab(apiKey);
    await initialiseMap();
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    if (state.googleAuthFailed) throw new Error("Google rejected this browser key. Check the enabled APIs, billing and localhost referrer restrictions.");
    state.googleConnected = true;
    state.googleApiKey = apiKey;
    setMapControlsEnabled(true);
    elements.mapConnectionPlaceholder.classList.add("hidden");
    elements.connectGoogleImagery.classList.add("connected");
    elements.connectGoogleImagery.querySelector("strong").textContent = "Google imagery connected";
    elements.googleConnectionStatus.textContent = state.stage7EnhancedReady
      ? "Google imagery and standard detection ready · OpenAI verification optional"
      : "Google imagery and standard detection ready · optional verification unavailable";
    setStatus(elements.locationStatus, "Google Places is ready. Search for a city, road or place.", "success");
    elements.mapInstruction.textContent = "Search a place, choose a road point for Street View, or inspect dated satellite context.";
    closeGoogleConnectDialog();
  } catch (error) {
    state.googleApiKey = null;
    state.stage7LocalReady = false;
    state.googleLoadFailed = Boolean(window.google?.maps?.importLibrary);
    handleGoogleAuthorisationFailure();
    setStatus(elements.googleConnectStatus, error.message, "error");
    elements.googleConnectionStatus.textContent = state.googleLoadFailed ? "Connection failed · reload to retry" : "Connection failed · key was not stored";
    if (state.googleLoadFailed) elements.submitGoogleConnect.textContent = "Reload to retry";
  } finally {
    state.googleLoading = false;
    elements.submitGoogleConnect.disabled = false;
  }
}

function setLayerVisibility(ids, visible) {
  if (!state.mapReady) return;
  if (ids.includes("sentinel-preview") && state.satelliteOverlay) state.satelliteOverlay.setMap(visible ? state.map : null);
}

function eventCoordinate(event) {
  return { longitude: event.latLng.lng(), latitude: event.latLng.lat() };
}

function localAreaFromMapCentre() {
  const centre = state.map?.getCenter?.();
  if (!centre) return null;
  const latitude = centre.lat();
  const longitude = centre.lng();
  const halfHeightKm = 5;
  const halfWidthKm = 5;
  const latitudeDelta = halfHeightKm / 111;
  const longitudeDelta = halfWidthKm / Math.max(20, 111 * Math.cos(latitude * Math.PI / 180));
  return {
    west: Math.max(-180, longitude - longitudeDelta),
    south: Math.max(-90, latitude - latitudeDelta),
    east: Math.min(180, longitude + longitudeDelta),
    north: Math.min(90, latitude + latitudeDelta),
  };
}

function setActiveMapMode(mode) {
  elements.mapRoad.classList.toggle("active", mode === "road");
  elements.mapSatellite.classList.toggle("active", mode === "satellite");
  elements.mapStreet.classList.toggle("active", mode === "street");
}

function showMapNotice(message, duration = 7000) {
  window.clearTimeout(state.mapNoticeTimer);
  elements.mapNotice.textContent = message;
  elements.mapNotice.classList.remove("hidden");
  state.mapNoticeTimer = window.setTimeout(() => elements.mapNotice.classList.add("hidden"), duration);
}

function showStreetDisplay(provider) {
  if (!state.streetDisplayMode) state.mapTypeBeforeStreet = state.map.getMapTypeId() || google.maps.MapTypeId.ROADMAP;
  state.streetDisplayMode = true;
  state.streetProvider = provider;
  elements.backToMap.classList.remove("hidden");
  elements.streetInlineBar.classList.remove("hidden");
  elements.mapNotice.classList.add("hidden");
  setActiveMapMode("street");
}

function returnToMap(mapType = state.mapTypeBeforeStreet) {
  state.streetDisplayMode = false;
  clearStage7GoogleOverlay();
  state.stage7CurrentFrameId = null;
  state.stage7RequestController?.abort();
  state.streetViewPanorama?.setVisible(false);
  elements.kartaStreetStage.classList.add("hidden");
  elements.backToMap.classList.add("hidden");
  elements.streetInlineBar.classList.add("hidden");
  if (mapType) state.map.setMapTypeId(mapType);
  setActiveMapMode(mapType === google.maps.MapTypeId.SATELLITE ? "satellite" : "road");
  elements.mapInstruction.textContent = "Map restored. Search a place, inspect another road point, or view dated satellite context.";
}

async function handleMapClick(event) {
  const point = eventCoordinate(event);
  if (state.placementId) {
    await placeSelectedImage(point.latitude, point.longitude);
    return;
  }
  if (state.streetSearchMode) {
    await searchStreetImagery(point.latitude, point.longitude);
  }
}

function selectGooglePlace(place) {
  const location = place?.location || place?.geometry?.location;
  const viewport = place?.viewport || place?.geometry?.viewport;
  if (!location) {
    setStatus(elements.locationStatus, "This Google Place did not provide a usable map location.", "warning");
    return;
  }
  const label = place.formattedAddress || place.formatted_address || place.displayName || place.name || "Selected Google Place";
  if (state.streetDisplayMode) returnToMap();
  state.locationLabel = label;
  elements.mapTitle.textContent = label;
  if (viewport) state.map.fitBounds(viewport, 60);
  else {
    state.map.setCenter(location);
    state.map.setZoom(15);
  }
  elements.locationResults.innerHTML = "";
  setStatus(elements.locationStatus, `Map moved to ${label}.`, "success");
}

function updateCollectionControls() {
  const optical = elements.collectionInput.value === "sentinel-2-l2a";
  elements.cloudField.classList.toggle("disabled", !optical);
  elements.cloudCover.disabled = !optical;
}

async function searchSatellite(event) {
  event.preventDefault();
  if (!state.mapReady) { showGoogleConnectDialog(); return; }
  state.aoi = localAreaFromMapCentre();
  if (!state.aoi) { setStatus(elements.satelliteStatus, "The current map centre is not ready. Try again in a moment.", "warning"); return; }
  elements.searchSatellite.disabled = true;
  revealSceneResults();
  setStatus(elements.satelliteStatus, "Searching a 10 x 10 km local area in the live Copernicus catalogue...", "loading");
  elements.sceneList.innerHTML = '<div class="empty-state loading">Reading satellite image metadata...</div>';
  try {
    const payload = await request("api/satellite/search", { method: "POST", headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(5000), body: JSON.stringify({ aoi: state.aoi, collection: elements.collectionInput.value, start_date: elements.startDate.value, end_date: elements.endDate.value, max_cloud_cover: Number(elements.cloudCover.value), limit: 24 }) });
    state.scenes = payload.scenes;
    renderScenes();
    setStatus(elements.satelliteStatus, `${payload.count} satellite image record${payload.count === 1 ? "" : "s"} found. ${payload.freshness}`, payload.count ? "success" : "warning");
  } catch (backendError) {
    try {
      const payload = await searchCopernicusDirect();
      state.scenes = payload.scenes;
      renderScenes();
      setStatus(elements.satelliteStatus, `${payload.count} satellite image record${payload.count === 1 ? "" : "s"} found. Direct catalogue connection used.`, payload.count ? "success" : "warning");
    } catch (directError) {
      state.scenes = [];
      renderScenes();
      setStatus(elements.satelliteStatus, `${backendError.message} The direct catalogue connection also failed; check internet access and retry.`, "error");
    }
  } finally { elements.searchSatellite.disabled = false; }
}

async function searchCopernicusDirect() {
  const query = {
    collections: [elements.collectionInput.value],
    bbox: [state.aoi.west, state.aoi.south, state.aoi.east, state.aoi.north],
    datetime: `${elements.startDate.value}T00:00:00Z/${elements.endDate.value}T23:59:59Z`,
    limit: 24,
    sortby: [{ field: "properties.datetime", direction: "desc" }],
  };
  if (elements.collectionInput.value === "sentinel-2-l2a") query.query = { "eo:cloud_cover": { lte: Number(elements.cloudCover.value) } };
  const response = await fetch(COPERNICUS_STAC_SEARCH, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/geo+json, application/json" }, body: JSON.stringify(query) });
  if (!response.ok) throw new Error(`Copernicus returned ${response.status}`);
  const payload = await response.json();
  const scenes = (Array.isArray(payload.features) ? payload.features : []).map((feature) => normaliseDirectScene(feature, elements.collectionInput.value)).filter(Boolean);
  return { count: scenes.length, scenes };
}

function normaliseDirectScene(feature, fallbackCollection) {
  if (!feature || !feature.id) return null;
  const properties = feature.properties || {};
  const collection = feature.collection || fallbackCollection;
  const bbox = Array.isArray(feature.bbox) && feature.bbox.length >= 4 ? feature.bbox.slice(0, 4).map(Number) : [];
  const optical = collection === "sentinel-2-l2a";
  const cloud = properties["eo:cloud_cover"] == null ? null : Number(properties["eo:cloud_cover"]);
  return {
    id: String(feature.id),
    title: properties.title || String(feature.id),
    collection,
    collection_name: optical ? "Sentinel-2 Level-2A optical" : "Sentinel-1 GRD radar",
    sensor: optical ? "Sentinel-2 MSI" : "Sentinel-1 SAR",
    captured_at: properties.datetime || properties.start_datetime || properties.created || null,
    cloud_cover: Number.isFinite(cloud) ? Math.round(cloud * 100) / 100 : null,
    gsd_m: Number.isFinite(Number(properties.gsd)) ? Number(properties.gsd) : null,
    resolution: optical ? "10–20 m principal bands" : "Product-dependent radar resolution",
    best_for: optical ? "Surface context and broad-area visual screening" : "All-weather structural change screening",
    bbox: bbox.length === 4 && bbox.every(Number.isFinite) ? bbox : [],
    preview_url: directPreviewUrl(feature.assets || {}),
    browser_url: `https://browser.stac.dataspace.copernicus.eu/collections/${encodeURIComponent(collection)}/items/${encodeURIComponent(feature.id)}`,
    provider: "Copernicus Data Space Ecosystem",
    licence: "Copernicus Sentinel data: free, full and open access with attribution",
    analysis_allowed: true,
    scientific_limit: "Satellite screening does not confirm individual potholes or footpath defects.",
  };
}

function directPreviewUrl(assets) {
  const candidates = [assets.thumbnail, assets.preview, assets.rendered_preview, ...Object.values(assets)];
  for (const asset of candidates) {
    if (!asset || !asset.href) continue;
    const type = String(asset.type || "").toLowerCase();
    const href = new URL(String(asset.href), COPERNICUS_STAC_ROOT).toString();
    const raster = ["image/jpeg", "image/png", "image/webp"].includes(type) || /\.(jpe?g|png|webp)(\?|$)/i.test(href);
    if (raster) return href;
  }
  return null;
}

function renderScenes() {
  revealSceneResults();
  elements.sceneCount.textContent = state.scenes.length;
  elements.sceneCaption.textContent = state.scenes.length ? "Dated catalogue records, newest first." : "No satellite images match the current filters.";
  if (!state.scenes.length) { elements.sceneList.innerHTML = '<div class="empty-state">No satellite imagery is available for this search.</div>'; return; }
  elements.sceneList.innerHTML = state.scenes.map((scene, index) => {
    const cloud = scene.cloud_cover == null ? "Cloud: not applicable" : `Cloud: ${scene.cloud_cover}%`;
    const preview = scene.preview_url ? `<img src="${escapeHtml(scene.preview_url)}" alt="Preview of ${escapeHtml(scene.id)}" loading="lazy" />` : '<div class="no-preview">Metadata only</div>';
    return `<article class="scene-card" data-scene-index="${index}">${preview}<div class="scene-copy"><p>${escapeHtml(scene.collection_name)}</p><h3>${escapeHtml(shortDate(scene.captured_at))}</h3><span>${escapeHtml(cloud)} &middot; ${escapeHtml(scene.resolution)}</span><small>${escapeHtml(scene.id)}</small><div class="scene-actions"><button type="button" data-action="a">Set A</button><button type="button" data-action="b">Set B</button><button type="button" data-action="map" ${scene.preview_url && scene.bbox.length === 4 ? "" : "disabled"}>Show on map</button><a href="${escapeHtml(scene.browser_url)}" target="_blank" rel="noopener noreferrer">Details</a></div></div></article>`;
  }).join("");
  elements.sceneList.querySelectorAll(".scene-card").forEach((card) => card.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (!action) return;
    const scene = state.scenes[Number(card.dataset.sceneIndex)];
    if (action === "a" || action === "b") setComparison(action, scene);
    if (action === "map") activateScene(scene);
  }));
}

function sceneSnapshot(scene) { return { id: scene.id, collection: scene.collection, captured_at: scene.captured_at, cloud_cover: scene.cloud_cover, bbox: scene.bbox || [] }; }

function setComparison(slot, scene) {
  state.comparison[slot] = scene;
  revealComparison();
  renderComparison(slot);
  if (scene.preview_url && scene.bbox.length === 4) activateScene(scene);
}

function renderComparison(slot) {
  const element = slot === "a" ? elements.comparisonA : elements.comparisonB;
  const scene = state.comparison[slot];
  if (scene || state.comparison.a || state.comparison.b) revealComparison();
  const label = slot.toUpperCase();
  if (!scene) { element.className = "comparison-slot empty"; element.innerHTML = `<span class="slot-label">${label}</span><div class="comparison-placeholder">Choose satellite date ${label}</div>`; return; }
  const image = scene.preview_url ? `<img src="${escapeHtml(scene.preview_url)}" alt="Satellite comparison ${label}" />` : '<div class="comparison-no-image">Saved metadata<br />Run the search again to restore the preview</div>';
  element.className = "comparison-slot";
  element.innerHTML = `<span class="slot-label">${label}</span>${image}<div class="comparison-meta"><strong>${escapeHtml(shortDate(scene.captured_at))}</strong><span>${escapeHtml(scene.collection_name || scene.collection)}</span><small>${scene.cloud_cover == null ? "Cloud not applicable" : `${scene.cloud_cover}% cloud`}</small></div>`;
}

function activateScene(scene) {
  if (!state.mapReady || !scene.preview_url || !Array.isArray(scene.bbox) || scene.bbox.length !== 4) return;
  if (state.streetDisplayMode) returnToMap();
  const [west, south, east, north] = scene.bbox;
  if (state.satelliteOverlay) state.satelliteOverlay.setMap(null);
  state.satelliteOverlay = new google.maps.GroundOverlay(
    scene.preview_url,
    { west, south, east, north },
    { clickable: false, opacity: Number(elements.satelliteOpacity.value) / 100 },
  );
  state.activeScene = scene.id;
  state.satelliteOverlay.setMap(elements.satelliteLayerToggle.checked ? state.map : null);
  state.map.fitBounds({ west, south, east, north }, 55);
  elements.mapInstruction.textContent = `${scene.collection_name}: ${shortDate(scene.captured_at)}. This preview supports broad-area screening only.`;
  document.querySelector("#map").scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderMarkers() {
  state.markers.forEach((marker) => marker.setMap(null));
  state.markers.clear();
  if (!state.mapReady) return;
  state.images.filter((image) => image.latitude != null && image.longitude != null).forEach((image) => {
    const marker = new google.maps.Marker({
      map: elements.groundLayerToggle.checked ? state.map : null,
      position: { lat: image.latitude, lng: image.longitude },
      title: image.original_name,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 7,
        fillColor: image.location_method === "manual" ? "#d89500" : "#147d78",
        fillOpacity: 1,
        strokeColor: "#ffffff",
        strokeOpacity: 1,
        strokeWeight: 2,
      },
    });
    marker.addListener("click", () => selectImage(image.id, state.images));
    state.markers.set(image.id, marker);
  });
}

function renderCatalogue() {
  elements.totalCount.textContent = state.images.length;
  elements.locatedCount.textContent = state.images.filter((image) => image.latitude != null && image.longitude != null).length;
  elements.unlocatedCount.textContent = state.images.filter((image) => image.latitude == null || image.longitude == null).length;
  if (!state.images.length) { elements.catalogueList.innerHTML = '<div class="empty-state">No ground images uploaded.</div>'; return; }
  elements.catalogueList.innerHTML = state.images.map((image) => `<button type="button" class="catalogue-item ${image.id === state.selectedId ? "active" : ""}" data-image-id="${escapeHtml(image.id)}"><span class="catalogue-thumb"><img src="${escapeHtml(image.image_url)}" alt="" loading="lazy" /></span><span><strong>${escapeHtml(image.original_name)}</strong><small>${image.latitude == null ? "Needs a verified location" : escapeHtml(coordinateText(image.latitude, image.longitude))}</small></span><i>${image.analysis_authorised === true ? "AI" : image.latitude == null ? "?" : "VIEW"}</i></button>`).join("");
  elements.catalogueList.querySelectorAll("[data-image-id]").forEach((button) => button.addEventListener("click", () => selectImage(button.dataset.imageId, state.images)));
}

function selectImage(id, sequence = state.images) {
  const image = state.images.find((item) => item.id === id);
  if (!image) return;
  state.selectedId = id;
  state.sequence = sequence.map((item) => item.id);
  state.sequenceIndex = Math.max(0, state.sequence.indexOf(id));
  elements.viewerEmpty.classList.add("hidden");
  elements.viewerContent.classList.remove("hidden");
  elements.viewerImage.src = image.image_url;
  elements.viewerMethod.textContent = image.latitude == null ? "LOCATION REQUIRED" : image.location_method === "manual" ? "MANUALLY LOCATED" : "GPS METADATA";
  elements.viewerName.textContent = image.original_name;
  elements.viewerSource.textContent = image.source || "User upload";
  elements.viewerTime.textContent = image.captured_at ? shortDate(image.captured_at) : "Not recorded";
  elements.viewerCoordinates.textContent = coordinateText(image.latitude, image.longitude);
  elements.viewerDirection.textContent = image.direction == null ? "Not recorded" : `${Number(image.direction).toFixed(1)} degrees`;
  elements.viewerSize.textContent = image.width && image.height ? `${image.width} x ${image.height} px` : "Not recorded";
  elements.viewerDistance.textContent = image.distance_m == null ? "Not from a nearby search" : `${image.distance_m.toLocaleString()} m`;
  elements.placementAction.classList.toggle("hidden", image.latitude != null && image.longitude != null);
  elements.manualLatitude.value = image.latitude == null ? "" : image.latitude;
  elements.manualLongitude.value = image.longitude == null ? "" : image.longitude;
  elements.stage6AuthorisationAction.classList.toggle("hidden", image.analysis_authorised === true);
  elements.stage6AuthoriseCheckbox.checked = false;
  elements.runStage6Inference.disabled = state.stage6Status?.status !== "research_deployment_ready";
  elements.detectionOverlay.replaceChildren();
  state.stage6CurrentObservations = [];
  elements.stage6Detections.innerHTML = '<div class="empty-state">Loading any existing observations for this image.</div>';
  elements.viewerPosition.textContent = `${state.sequenceIndex + 1} of ${state.sequence.length}`;
  elements.previousImage.disabled = state.sequenceIndex <= 0;
  elements.nextImage.disabled = state.sequenceIndex >= state.sequence.length - 1;
  if (state.mapReady && image.latitude != null && image.longitude != null) {
    if (state.streetDisplayMode) returnToMap();
    state.map.setCenter({ lat: image.latitude, lng: image.longitude });
    state.map.setZoom(Math.max(state.map.getZoom(), 15));
  }
  renderCatalogue();
  loadStage6Activity().catch((error) => setStatus(elements.stage6StatusMessage, error.message, "error"));
}

async function loadImages() {
  const payload = await request("api/images");
  state.images = payload.images;
  if (state.selectedId && !state.images.some((image) => image.id === state.selectedId)) state.selectedId = null;
  renderCatalogue();
  renderMarkers();
}

async function uploadImages(event) {
  event.preventDefault();
  if (!elements.imageInput.files.length) return;
  const data = new FormData();
  [...elements.imageInput.files].forEach((file) => data.append("files", file));
  data.append("source", elements.sourceInput.value.trim() || "User field survey");
  data.append("analysis_authorised", elements.analysisAuthorised.checked ? "true" : "false");
  setStatus(elements.uploadStatus, "Reading image metadata and storing the original files...", "loading");
  try {
    const payload = await request("api/images/upload", { method: "POST", body: data });
    await loadImages();
    const rejected = payload.rejected?.length || 0;
    setStatus(elements.uploadStatus, `${payload.accepted.length} image${payload.accepted.length === 1 ? "" : "s"} added${rejected ? `; ${rejected} rejected` : ""}.`, rejected ? "warning" : "success");
    elements.uploadForm.reset(); elements.sourceInput.value = "User field survey"; elements.selectionSummary.textContent = "No files selected";
    if (payload.accepted.length) selectImage(payload.accepted[0].id, payload.accepted);
  } catch (error) { setStatus(elements.uploadStatus, error.message, "error"); }
}

async function placeSelectedImage(latitude, longitude) {
  const imageId = state.placementId;
  try {
    await request(`api/images/${encodeURIComponent(imageId)}/location`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ latitude, longitude }) });
    state.placementId = null;
    state.map.setOptions({ draggableCursor: null });
    elements.mapInstruction.textContent = "Ground image placed. Click any marker to inspect evidence.";
    await loadImages();
    selectImage(imageId, state.images);
  } catch (error) { setStatus(elements.uploadStatus, error.message, "error"); }
}

function stage6Label(value) {
  return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderStage6Runtime(status) {
  state.stage6Status = status;
  const active = status.active_model || {};
  const ready = status.status === "research_deployment_ready" && active.package_ready === true;
  elements.stage6RuntimeBadge.textContent = ready ? "Ready" : "Blocked";
  elements.stage6RuntimeBadge.className = `runtime-badge ${ready ? "ready" : "blocked"}`;
  elements.stage6ActiveModel.textContent = active.model_version || "Unavailable";
  const checks = Object.values(active.checks || {});
  const passed = checks.filter(Boolean).length;
  elements.stage6PackageState.textContent = `${passed}/${checks.length} checks passed`;
  elements.runStage6Inference.disabled = !ready || !state.selectedId;
  elements.runStage6SelfTest.disabled = !ready;
}

async function loadStage6Status() {
  const status = await request("api/stage6");
  renderStage6Runtime(status);
  return status;
}

async function loadStage7Status() {
  const status = await request("api/stage7");
  state.stage7Status = status;
  state.stage7LocalReady = status.inference_modes?.local?.ready === true;
  state.stage7EnhancedReady = Boolean(
    status.inference_modes?.enhanced?.ready === true
    && verifiedStage7Mode(status.inference_modes.enhanced)
  );
  renderStage7Mode();
  elements.retryStage7Analysis.disabled = !state.selectedRoadPoint || status.status !== "on_demand_runtime_ready";
  return status;
}

function renderCurrentStage6Observations(observations) {
  state.stage6CurrentObservations = observations;
  elements.detectionOverlay.replaceChildren();
  if (!observations.length) {
    elements.stage6Detections.innerHTML = '<div class="empty-state">No saved detection is available for this image.</div>';
    return;
  }
  elements.detectionOverlay.innerHTML = observations.map((observation) => {
    const region = observation.evidence_region || {};
    if (region.type !== "bounding_box") return "";
    const left = Number(region.x_min) * 100;
    const top = Number(region.y_min) * 100;
    const width = (Number(region.x_max) - Number(region.x_min)) * 100;
    const height = (Number(region.y_max) - Number(region.y_min)) * 100;
    const label = `${stage6Label(observation.damage_type)} · ${(Number(observation.confidence) * 100).toFixed(1)}%`;
    return `<div class="detection-box" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%"><span>${escapeHtml(label)}</span></div>`;
  }).join("");
  elements.stage6Detections.innerHTML = observations.map((observation) => `<article class="stage6-detection-card"><header><strong>${escapeHtml(stage6Label(observation.damage_type))}</strong><span>${(Number(observation.confidence) * 100).toFixed(1)}%</span></header><small>${escapeHtml(stage6Label(observation.surface_type))} · ${escapeHtml(observation.model?.version || "model unavailable")} · review ${escapeHtml(observation.review?.status || "pending")}</small></article>`).join("");
}

function renderStage6History(activity) {
  state.stage6Activity = activity;
  elements.stage6HistorySummary.textContent = `${activity.observation_count} observation${activity.observation_count === 1 ? "" : "s"} from ${activity.run_count} inference run${activity.run_count === 1 ? "" : "s"}. Confirmed records are available through the governed export endpoint.`;
  if (!activity.observations.length) {
    elements.stage6History.innerHTML = '<div class="empty-state">No uploaded-image observations have been created.</div>';
  } else {
    elements.stage6History.innerHTML = activity.observations.map((envelope) => {
      const observation = envelope.observation;
      const status = observation.review?.status || "pending";
      return `<article class="stage6-history-card ${escapeHtml(status)}"><header><h3>${escapeHtml(stage6Label(observation.damage_type))}</h3><span>${(Number(observation.confidence) * 100).toFixed(1)}%</span></header><p>${escapeHtml(envelope.image_name)} · ${escapeHtml(stage6Label(observation.surface_type))}</p><small>${escapeHtml(observation.model?.version || "model unavailable")} · ${escapeHtml(observation.observation_id)}</small><small>Review: ${escapeHtml(stage6Label(status))}${observation.review?.reviewer_id ? ` by ${escapeHtml(observation.review.reviewer_id)}` : ""}</small><div class="stage6-review-actions"><button type="button" data-observation-id="${escapeHtml(observation.observation_id)}" data-review-status="confirmed">Confirm</button><button type="button" data-observation-id="${escapeHtml(observation.observation_id)}" data-review-status="uncertain">Uncertain</button><button type="button" data-observation-id="${escapeHtml(observation.observation_id)}" data-review-status="rejected">Reject</button></div></article>`;
    }).join("");
    elements.stage6History.querySelectorAll("[data-observation-id]").forEach((button) => button.addEventListener("click", () => reviewStage6Observation(button.dataset.observationId, button.dataset.reviewStatus)));
  }
  if (state.selectedId) {
    const current = activity.observations
      .filter((item) => item.image_id === state.selectedId)
      .map((item) => item.observation);
    renderCurrentStage6Observations(current);
  }
}

async function loadStage6Activity() {
  const activity = await request("api/stage6/activity");
  renderStage6History(activity);
  return activity;
}

async function runStage6Inference() {
  if (!state.selectedId) {
    setStatus(elements.stage6StatusMessage, "Select an authorised ground image first.", "warning");
    return;
  }
  elements.runStage6Inference.disabled = true;
  setStatus(elements.stage6StatusMessage, "Running the active hash-verified model in the isolated GPU process...", "loading");
  try {
    const payload = await request(`api/stage6/images/${encodeURIComponent(state.selectedId)}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ surface_type: elements.stage6Surface.value || null }),
    });
    if (payload.result.result_type === "observations") {
      renderCurrentStage6Observations(payload.result.observations);
      const timing = payload.run.worker?.timing_ms?.worker_inference;
      setStatus(elements.stage6StatusMessage, `${payload.result.observations.length} review-required observation${payload.result.observations.length === 1 ? "" : "s"} created${timing ? ` · isolated worker ${Number(timing).toFixed(0)} ms` : ""}.`, "success");
    } else {
      renderCurrentStage6Observations([]);
      setStatus(elements.stage6StatusMessage, `Safe abstention: ${(payload.result.reasons || []).map(stage6Label).join(", ")}.`, "warning");
    }
    await loadStage6Activity();
  } catch (error) {
    setStatus(elements.stage6StatusMessage, error.message, "error");
  } finally {
    elements.runStage6Inference.disabled = state.stage6Status?.status !== "research_deployment_ready";
  }
}

async function runStage6SelfTest() {
  elements.runStage6SelfTest.disabled = true;
  setStatus(elements.stage6StatusMessage, "Loading the active package and running a synthetic no-project-pixel self-test...", "loading");
  try {
    const result = await request("api/stage6/self-test", { method: "POST" });
    setStatus(elements.stage6StatusMessage, `Self-test passed on device ${result.device} · ${Number(result.timing_ms.worker_inference || 0).toFixed(0)} ms worker runtime.`, "success");
  } catch (error) {
    setStatus(elements.stage6StatusMessage, error.message, "error");
  } finally {
    elements.runStage6SelfTest.disabled = state.stage6Status?.status !== "research_deployment_ready";
  }
}

async function reviewStage6Observation(observationId, reviewStatus) {
  const reviewerId = elements.stage6Reviewer.value.trim();
  if (!reviewerId) {
    setStatus(elements.stage6HistorySummary, "Enter a reviewer ID before recording a decision.", "warning");
    elements.stage6Reviewer.focus();
    return;
  }
  setStatus(elements.stage6HistorySummary, `Saving ${stage6Label(reviewStatus).toLowerCase()} review...`, "loading");
  try {
    await request(`api/stage6/observations/${encodeURIComponent(observationId)}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: reviewStatus, reviewer_id: reviewerId, note: elements.stage6ReviewNote.value.trim() || null }),
    });
    elements.stage6ReviewNote.value = "";
    await loadStage6Activity();
    setStatus(elements.stage6HistorySummary, `${stage6Label(reviewStatus)} review saved with reviewer provenance.`, "success");
  } catch (error) {
    setStatus(elements.stage6HistorySummary, error.message, "error");
  }
}

async function saveManualLocation() {
  if (!state.selectedId) return;
  const latitude = Number(elements.manualLatitude.value);
  const longitude = Number(elements.manualLongitude.value);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    setStatus(elements.stage6StatusMessage, "Enter valid latitude and longitude values.", "warning");
    return;
  }
  try {
    await request(`api/images/${encodeURIComponent(state.selectedId)}/location`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ latitude, longitude }) });
    await loadImages();
    selectImage(state.selectedId, state.images);
    setStatus(elements.stage6StatusMessage, "Verified manual coordinates saved.", "success");
  } catch (error) {
    setStatus(elements.stage6StatusMessage, error.message, "error");
  }
}

async function authoriseSelectedImage() {
  if (!state.selectedId || !elements.stage6AuthoriseCheckbox.checked) {
    setStatus(elements.stage6StatusMessage, "Confirm the image-rights declaration first.", "warning");
    return;
  }
  try {
    await request(`api/images/${encodeURIComponent(state.selectedId)}/analysis-authorisation`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed: true, declaration: "I confirm that I have the right to submit this image for automated research analysis." }),
    });
    await loadImages();
    selectImage(state.selectedId, state.images);
    setStatus(elements.stage6StatusMessage, "Analysis authorisation recorded with the image provenance.", "success");
  } catch (error) {
    setStatus(elements.stage6StatusMessage, error.message, "error");
  }
}

const STAGE7_CLASS_COLOURS = Object.freeze({
  pothole: "#ff334f",
  longitudinal_crack: "#00c2ff",
  transverse_crack: "#8b5cf6",
  alligator_cracking: "#ff8a00",
  uneven_or_broken_footpath: "#24c96b",
  broken_kerb: "#ffe100",
  standing_water: "#1479ff",
  debris_or_vegetation_obstruction: "#e83eaf",
  visible_road_anomaly: "#00b894",
});

function stage7Boxes(detections) {
  return (detections || []).map((detection) => {
    const region = detection.evidence_region || (Array.isArray(detection.box) ? {
      type: "bounding_box",
      x_min: detection.box[0], y_min: detection.box[1], x_max: detection.box[2], y_max: detection.box[3],
    } : {});
    if (region.type !== "bounding_box") return "";
    const left = Number(region.x_min) * 100;
    const top = Number(region.y_min) * 100;
    const width = (Number(region.x_max) - Number(region.x_min)) * 100;
    const height = (Number(region.y_max) - Number(region.y_min)) * 100;
    if (![left, top, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return "";
    const damageType = detection.damage_type || "unknown_detection";
    const humanReviewed = detection.human_reviewed === true;
    const classless = detection.classless === true && !humanReviewed;
    const label = humanReviewed
      ? `Corrected · ${stage6Label(detection.surface_type)} · ${stage6Label(damageType)}`
      : `${stage6Label(damageType)} · ${(Number(detection.confidence) * 100).toFixed(1)}%`;
    const colour = STAGE7_CLASS_COLOURS[damageType] || "#ff334f";
    const labelInk = damageType === "broken_kerb" ? "#171100" : "#ffffff";
    return `<div class="detection-box stage7-detection-box ${humanReviewed ? "human-reviewed" : ""}" data-damage-type="${escapeHtml(damageType)}" data-detection-id="${escapeHtml(detection.detection_id || "")}" style="--stage7-color:${colour};--stage7-label-ink:${labelInk};left:${left}%;top:${top}%;width:${width}%;height:${height}%">${classless ? "" : `<span>${escapeHtml(label)}</span>`}</div>`;
  }).join("");
}

function clearStage7GoogleOverlay() {
  elements.stage7GoogleOverlay.replaceChildren();
  elements.stage7GoogleOverlay.classList.add("hidden");
  elements.stage7GoogleOverlay.classList.remove("human-reviewed");
}

function invalidateGoogleStage7Frame(reason = "camera_changed") {
  endStage8Review({ silent: true, message: state.stage8Review ? "The Street View camera changed, so the unsaved review and old boxes were cleared." : "" });
  state.stage8LastSavedReview = null;
  state.stage7FrameRevision += 1;
  state.stage7CurrentFrameId = null;
  state.stage7RequestToken += 1;
  state.stage7RequestController?.abort();
  state.stage7RequestController = null;
  clearStage7GoogleOverlay();
  if (state.streetProvider !== "google" || !state.streetDisplayMode || !state.selectedRoadPoint) return;
  state.stage7Analysis = null;
  elements.stage7AnalysisCard.className = "stage7-analysis-card idle";
  elements.stage7AnalysisTitle.textContent = "Current view changed";
  elements.stage7AnalysisBadge.textContent = "Run this view";
  elements.stage7AnalysisMessage.textContent = "The old boxes were cleared. Analyse the new panorama and camera direction now visible on the map.";
  elements.stage7AnalysisReasons.innerHTML = `<span>${escapeHtml(stage6Label(reason))}</span>`;
  elements.stage7AnalysisImageWrap.classList.add("hidden");
  elements.stage7AnalysisDetails.innerHTML = '<span><b>Processed</b><i>Waiting for rerun</i></span><span><b>Frame</b><i>Changed</i></span>';
  elements.retryStage7Analysis.disabled = state.stage7Status?.status !== "on_demand_runtime_ready";
  elements.retryStage7Analysis.textContent = "Analyse this view";
  updateStage8ReviewAvailability(null);
}

function invalidateGoogleStage7FrameIfMaterial(reason = "camera_changed") {
  const signature = currentGoogleCameraSignature();
  if (signature && signature === state.stage7MaterialCameraSignature) return;
  state.stage7MaterialCameraSignature = signature;
  invalidateGoogleStage7Frame(reason);
}

function currentGoogleFrameDescriptor() {
  const panorama = state.streetViewPanorama;
  const panoId = panorama?.getPano?.();
  const position = panorama?.getPosition?.();
  const pov = panorama?.getPov?.() || {};
  const rawZoom = Number(panorama?.getZoom?.());
  if (!panoId || !position || !Number.isFinite(Number(pov.heading)) || !Number.isFinite(Number(pov.pitch))) {
    throw new Error("The current Google panorama camera is not ready yet.");
  }
  const viewport = document.querySelector(".map-stage").getBoundingClientRect();
  const viewportWidth = Math.max(128, Math.round(viewport.width));
  const viewportHeight = Math.max(128, Math.round(viewport.height));
  const aspect = viewportWidth / viewportHeight;
  let width;
  let height;
  if (aspect >= 1) {
    width = 640;
    height = Math.max(128, Math.round(width / aspect));
  } else {
    height = 640;
    width = Math.max(128, Math.round(height * aspect));
  }
  width = Math.min(640, width);
  height = Math.min(640, height);
  const heading = ((Number(pov.heading) % 360) + 360) % 360;
  const pitch = Math.max(-90, Math.min(90, Number(pov.pitch)));
  const zoom = Number.isFinite(rawZoom) ? Math.max(0, Math.min(10, rawZoom)) : 1;
  const fov = Math.max(10, Math.min(120, 180 / (2 ** zoom)));
  const latitude = Number(position.lat());
  const longitude = Number(position.lng());
  const frameRevision = state.stage7FrameRevision;
  const frameId = `google-frame-${frameRevision}-${String(panoId).slice(0, 80)}-${heading.toFixed(4)}-${pitch.toFixed(4)}-${zoom.toFixed(3)}-${width}x${height}`;
  return {
    frame_id: frameId,
    pano_id: panoId,
    latitude,
    longitude,
    heading,
    pitch,
    zoom,
    fov,
    width,
    height,
    viewport_width: viewportWidth,
    viewport_height: viewportHeight,
    frame_revision: frameRevision,
  };
}

function currentGoogleCameraSignature() {
  const panorama = state.streetViewPanorama;
  const panoId = panorama?.getPano?.();
  const position = panorama?.getPosition?.();
  const pov = panorama?.getPov?.() || {};
  const zoom = Number(panorama?.getZoom?.());
  const viewport = document.querySelector(".map-stage")?.getBoundingClientRect?.();
  const latitude = Number(position?.lat?.());
  const longitude = Number(position?.lng?.());
  const heading = Number(pov.heading);
  const pitch = Number(pov.pitch);
  if (
    !panoId
    || !Number.isFinite(latitude)
    || !Number.isFinite(longitude)
    || !Number.isFinite(heading)
    || !Number.isFinite(pitch)
    || !Number.isFinite(zoom)
    || !viewport
    || viewport.width < 128
    || viewport.height < 128
  ) return null;
  return [
    panoId,
    latitude.toFixed(6),
    longitude.toFixed(6),
    heading.toFixed(2),
    pitch.toFixed(2),
    zoom.toFixed(3),
    Math.round(viewport.width / 4) * 4,
    Math.round(viewport.height / 4) * 4,
  ].join("|");
}

async function waitForGoogleCurrentViewReady(intent, timeoutMs = 12000, quietMs = 900) {
  const deadline = performance.now() + timeoutMs;
  let previous = null;
  let stableSince = 0;
  while (performance.now() < deadline) {
    if (intent !== state.stage7RunIntent || state.streetProvider !== "google" || !state.streetDisplayMode) return false;
    const signature = currentGoogleCameraSignature();
    const now = performance.now();
    if (!signature || signature !== previous) {
      previous = signature;
      stableSince = signature ? now : 0;
    } else if (stableSince && now - stableSince >= quietMs) {
      return true;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 120));
  }
  return false;
}

function renderStage7MapOverlay() {
  const analysis = state.stage7Analysis;
  const currentPhoto = state.streetSequence[state.streetIndex];
  const matches = Boolean(
    analysis?.imagery?.photo_id
    && currentPhoto?.id
    && String(analysis.imagery.photo_id) === String(currentPhoto.id)
  );
  elements.stage7StreetOverlay.innerHTML = matches ? stage7Boxes(analysis.observations) : "";
}

function stage8SavedReviewForFrame(payload) {
  const saved = state.stage8LastSavedReview;
  if (!saved || !payload?.frame?.frame_id || saved.frameId !== payload.frame.frame_id) return null;
  return saved.record || null;
}

function stage8RecordsForFrame(payload, extraRecord = null) {
  const frameId = payload?.frame?.frame_id;
  if (!frameId) return [];
  const records = [...(state.stage8Queue.records || [])];
  if (extraRecord && !records.some((record) => record.annotation_id === extraRecord.annotation_id)) records.push(extraRecord);
  return records
    .filter((record) => record?.frame?.frame_id === frameId && record.status !== "rejected")
    .sort((left, right) => Number(left.review_version || 0) - Number(right.review_version || 0));
}

function stage8LatestRecordForFrame(payload) {
  const current = stage8SavedReviewForFrame(payload);
  const records = stage8RecordsForFrame(payload, current);
  return records.at(-1) || current || null;
}

function stage8ReviewedDetections(record, payload = state.stage7Analysis) {
  const committed = new Map();
  stage8RecordsForFrame(payload, record).forEach((savedRecord) => {
    (savedRecord?.human_review?.annotations || []).filter(stage8BoxResolved).forEach((annotation) => {
      const key = annotation.source_detection_id || annotation.annotation_box_id;
      committed.set(key, annotation);
    });
  });
  return [...committed.values()].map((annotation) => ({
    box: [...annotation.box],
    damage_type: annotation.damage_type,
    surface_type: annotation.surface_type,
    human_reviewed: true,
  }));
}

function renderStage7GoogleOverlay(payload) {
  const matches = Boolean(
    payload?.frame?.frame_id
    && payload.frame.frame_id === state.stage7CurrentFrameId
    && Number(payload.frame.frame_revision) === state.stage7FrameRevision
    && payload.frame.pano_id === state.streetViewPanorama?.getPano?.()
  );
  const savedReview = stage8SavedReviewForFrame(payload);
  const detections = savedReview ? stage8ReviewedDetections(savedReview, payload) : (payload?.detections || []);
  if (!matches || !detections.length) {
    clearStage7GoogleOverlay();
    if (savedReview) elements.stage7AnalysisOverlay.replaceChildren();
    return;
  }
  const overlayHtml = stage7Boxes(detections);
  elements.stage7GoogleOverlay.innerHTML = overlayHtml;
  elements.stage7AnalysisOverlay.innerHTML = overlayHtml;
  elements.stage7GoogleOverlay.classList.toggle("human-reviewed", Boolean(savedReview));
  elements.stage7GoogleOverlay.classList.remove("hidden");
}

function stage7OutcomeMeta(stateName) {
  if (stateName === "detections_found_review_required") return { title: "Possible issues found", badge: "Check result", kind: "detections" };
  if (stateName === "analysis_complete_no_supported_detection") return { title: "Analysis complete", badge: "No issue found", kind: "clear" };
  if (stateName === "analysis_not_possible") return { title: "This image could not be analysed", badge: "Try again", kind: "blocked" };
  return { title: "Analysing the current image", badge: "Running", kind: "loading" };
}

function renderStage7CurrentViewLoading(frame) {
  endStage8Review({ silent: true });
  state.stage8LastSavedReview = null;
  revealStreetResults();
  const meta = stage7OutcomeMeta("loading");
  clearStage7GoogleOverlay();
  elements.stage7AnalysisCard.className = `stage7-analysis-card ${meta.kind}`;
  elements.stage7AnalysisTitle.textContent = "Analysing this visible frame";
  elements.stage7AnalysisBadge.textContent = "Running";
  elements.stage7AnalysisPoint.textContent = `${coordinateText(frame.latitude, frame.longitude)} · heading ${frame.heading.toFixed(1)}° · pitch ${frame.pitch.toFixed(1)}°`;
  elements.stage7AnalysisMessage.textContent = state.stage7InferenceMode === "enhanced"
    ? "Capturing the visible image, running the local detectors and applying OpenAI visual verification with your key."
    : "Capturing the visible image and running the local road-condition detectors without OpenAI.";
  elements.stage7AnalysisImageWrap.classList.add("hidden");
  elements.stage7AnalysisOverlay.replaceChildren();
  elements.stage7AnalysisDetails.innerHTML = `<span><b>Analysis</b><i>Running</i></span><span><b>Method</b><i>${state.stage7InferenceMode === "enhanced" ? "Standard + OpenAI" : "Standard detection"}</i></span><span><b>Image</b><i>Current Google Street View</i></span>`;
  elements.stage7AnalysisReasons.innerHTML = "";
  elements.stage7AnalysisSource.classList.add("hidden");
  elements.retryStage7Analysis.disabled = true;
  updateStage8ReviewAvailability(null);
}

function renderStage7Analysis(payload) {
  endStage8Review({ silent: true });
  state.stage8LastSavedReview = null;
  state.stage7Analysis = payload;
  revealStreetResults();
  const meta = stage7OutcomeMeta(payload.state);
  elements.stage7AnalysisCard.className = `stage7-analysis-card ${meta.kind}`;
  elements.stage7AnalysisTitle.textContent = meta.title;
  elements.stage7AnalysisBadge.textContent = meta.badge;
  const point = payload.selected_point || state.selectedRoadPoint || {};
  elements.stage7AnalysisPoint.textContent = payload.frame
    ? `${coordinateText(point.latitude, point.longitude)} · heading ${Number(payload.frame.heading).toFixed(1)}° · pitch ${Number(payload.frame.pitch).toFixed(1)}°`
    : `${coordinateText(point.latitude, point.longitude)} · ${Number(point.radius_m || 0).toLocaleString()} m search radius`;
  elements.stage7AnalysisMessage.textContent = payload.message;
  const imagery = payload.imagery;
  const previewUrl = imagery?.preview_data_url || imagery?.thumbnail_url || imagery?.image_url;
  const detections = payload.detections || payload.observations || [];
  if (previewUrl) {
    elements.stage7AnalysisImageWrap.classList.remove("hidden");
    elements.stage7AnalysisImage.src = previewUrl;
    elements.stage7AnalysisOverlay.innerHTML = stage7Boxes(detections);
    if (imagery?.details_url) {
      elements.stage7AnalysisSource.href = imagery.details_url;
      elements.stage7AnalysisSource.textContent = payload.frame ? "Open this location in Google Maps" : `${imagery.provider || "Evidence"} source and attribution`;
      elements.stage7AnalysisSource.classList.remove("hidden");
    } else {
      elements.stage7AnalysisSource.classList.add("hidden");
    }
  } else {
    elements.stage7AnalysisImageWrap.classList.add("hidden");
    elements.stage7AnalysisImage.removeAttribute("src");
    elements.stage7AnalysisOverlay.replaceChildren();
    elements.stage7AnalysisSource.classList.add("hidden");
  }
  const modelVersion = payload.model?.version || "No model execution";
  const inferenceMode = payload.inference_mode === "enhanced"
    ? "Standard + OpenAI"
    : (payload.inference_mode === "local_fallback" ? "Standard fallback" : "Standard detection");
  const provider = imagery?.provider || (payload.requested_provider === "google_current_panorama_frame" ? "Google current view" : "Open-source search");
  const frameIdentity = payload.frame?.frame_id ? `Revision ${payload.frame.frame_revision}` : (imagery?.distance_m == null ? "Not available" : `${Number(imagery.distance_m).toLocaleString()} m from selected point`);
  elements.stage7AnalysisDetails.innerHTML = [
    ["Analysis", payload.processed ? "Completed" : "Not completed"],
    ["Image source", provider],
    [payload.frame ? "Image" : "Evidence", frameIdentity],
    ["Issues found", String(detections.length)],
    ["Method", inferenceMode],
    ["Detection engine", modelVersion],
    ["Image handling", payload.processed ? "Temporary · deleted after analysis" : "No image retained"],
  ].map(([label, value]) => `<span><b>${escapeHtml(label)}</b><i>${escapeHtml(value)}</i></span>`).join("");
  elements.stage7AnalysisReasons.innerHTML = (payload.reason_codes || []).map((reason) => `<span>${escapeHtml(stage6Label(reason))}</span>`).join("");
  elements.retryStage7Analysis.disabled = state.stage7Status?.status !== "on_demand_runtime_ready" || state.streetProvider !== "google" || !state.streetViewPanorama?.getPano?.();
  elements.retryStage7Analysis.textContent = "Analyse this view";
  if (payload.frame) renderStage7GoogleOverlay(payload); else renderStage7MapOverlay();
  if (state.streetProvider === "google" && state.streetDisplayMode) {
    if (detections.length) {
      setStatus(elements.streetStatus, `This exact visible Street View frame was processed and ${detections.length} class-coloured detection${detections.length === 1 ? " is" : "s are"} drawn directly on it.`, "success");
      elements.mapInstruction.textContent = "The boxes are locked to this panorama, direction and zoom. Move forward or rotate to clear them, then analyse the new image.";
    } else if (payload.processed) {
      setStatus(elements.streetStatus, "This exact visible Street View frame was processed successfully; the model retained zero supported detections in this view.", "warning");
      elements.mapInstruction.textContent = "No supported issue was found in this image. Move through Street View and analyse the next image when ready.";
    } else {
      setStatus(elements.streetStatus, `${payload.message} The visible panorama remains open and no substitute image was used.`, "error");
    }
  }
  updateStage8ReviewAvailability(payload);
}

async function runStage7CurrentViewAnalysis() {
  if (!state.googleApiKey || state.streetProvider !== "google") {
    const message = !state.googleApiKey
      ? "Reconnect Google imagery so the current Street View image can be analysed."
      : "Open a Google Street View panorama before analysing the image.";
    setStatus(elements.streetStatus, message, "warning");
    return null;
  }
  const intent = ++state.stage7RunIntent;
  setStatus(elements.streetStatus, "Waiting for the current panorama camera to settle before analysis...", "loading");
  const cameraReady = await waitForGoogleCurrentViewReady(intent);
  if (intent !== state.stage7RunIntent) return null;
  if (!cameraReady) {
    setStatus(elements.streetStatus, "The current Street View camera did not become stable. Keep this image open and try Analyse this view again.", "warning");
    return null;
  }
  state.stage7MaterialCameraSignature = currentGoogleCameraSignature();
  let frame;
  try {
    frame = currentGoogleFrameDescriptor();
  } catch (error) {
    setStatus(elements.streetStatus, error.message, "warning");
    return null;
  }
  state.selectedRoadPoint = {
    latitude: frame.latitude,
    longitude: frame.longitude,
    radius_m: Number(elements.streetRadius.value),
  };
  state.stage7RequestController?.abort();
  const controller = new AbortController();
  state.stage7RequestController = controller;
  const token = ++state.stage7RequestToken;
  state.stage7CurrentFrameId = frame.frame_id;
  renderStage7CurrentViewLoading(frame);
  let timedOut = false;
  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, state.stage7Status?.inference_modes?.local?.display_class_names === false || stage7CandidateModel === "classless_v022" ? 300000 : 180000);
  try {
    const requestBody = {
      api_key: state.googleApiKey,
      inference_mode: state.stage7InferenceMode,
      ...frame,
    };
    if (state.stage7InferenceMode === "enhanced") requestBody.stage7_api_key = state.stage7VisionApiKey;
    const payload = await request(stage7AnalyseEndpoint(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(requestBody),
    });
    if (
      token !== state.stage7RequestToken
      || frame.frame_revision !== state.stage7FrameRevision
      || frame.frame_id !== state.stage7CurrentFrameId
      || frame.pano_id !== state.streetViewPanorama.getPano()
    ) return null;
    renderStage7Analysis(payload);
    return payload;
  } catch (error) {
    if (token !== state.stage7RequestToken || frame.frame_revision !== state.stage7FrameRevision) return null;
    const payload = {
      selected_point: { latitude: frame.latitude, longitude: frame.longitude },
      requested_provider: "google_current_panorama_frame",
      inference_mode: state.stage7InferenceMode,
      frame,
      state: "analysis_not_possible",
      processed: false,
      message: timedOut
        ? "The exact current-frame analysis exceeded its bounded runtime."
        : `The exact current-frame request failed: ${error.message}`,
      reason_codes: [timedOut ? "exact_current_frame_timeout" : "exact_current_frame_request_failed"],
      imagery: null,
      model: null,
      detections: [],
    };
    renderStage7Analysis(payload);
    return payload;
  } finally {
    window.clearTimeout(timeoutId);
    if (state.stage7RequestController === controller) state.stage7RequestController = null;
  }
}

function beginStreetSearch() {
  if (!state.mapReady) {
    setStatus(elements.streetStatus, "The map is still loading. Try again in a moment.", "warning");
    return;
  }
  if (state.streetDisplayMode) returnToMap();
  state.mapTypeBeforeStreet = state.map.getMapTypeId() || google.maps.MapTypeId.ROADMAP;
  state.placementId = null;
  state.streetSearchMode = true;
  elements.streetLayerToggle.checked = true;
  state.streetCoverageLayer?.setMap(state.map);
  elements.startStreetSearch.classList.add("active");
  elements.startStreetSearch.textContent = "Click a road on the map";
  elements.mapInstruction.textContent = "Road-point selection is active. Click a blue Street View coverage road to open and analyse its current panorama.";
  state.map.setOptions({ draggableCursor: "crosshair" });
  document.querySelector("#map").scrollIntoView({ behavior: "smooth", block: "center" });
}

function finishStreetSearchMode() {
  state.streetSearchMode = false;
  elements.startStreetSearch.classList.remove("active");
  elements.startStreetSearch.textContent = "Choose another road point";
  if (state.map) state.map.setOptions({ draggableCursor: null });
}

function renderStreetResults() {
  revealStreetResults();
  elements.streetCount.textContent = state.streetResults.length;
  elements.streetCaption.textContent = state.streetResults.length
    ? "Public photographs nearest to the selected road point."
    : "No public KartaView imagery was found near this point.";
  if (!state.streetResults.length) {
    elements.streetList.innerHTML = '<div class="empty-state"><strong>No open street imagery here</strong><span>Try another road point, increase the radius, or upload your own ground photographs.</span></div>';
    return;
  }
  elements.streetList.innerHTML = state.streetResults.map((photo, index) => `<button type="button" class="street-result ${photo.id === state.streetSelectedId ? "active" : ""}" data-street-index="${index}"><img src="${escapeHtml(photo.thumbnail_url)}" alt="" loading="lazy" /><span><strong>${escapeHtml(shortDate(photo.captured_at))}</strong><small>${photo.distance_m == null ? "Distance not reported" : `${Number(photo.distance_m).toLocaleString()} m away`} &middot; ${photo.direction == null ? "Direction unavailable" : `${Number(photo.direction).toFixed(0)}°`}</small></span></button>`).join("");
  elements.streetList.querySelectorAll("[data-street-index]").forEach((button) => button.addEventListener("click", () => selectStreetPhoto(state.streetResults[Number(button.dataset.streetIndex)], state.streetResults, true)));
}

function updateStreetMarker(photo) {
  if (state.streetMarker) state.streetMarker.setMap(null);
  state.streetMarker = new google.maps.Marker({
    map: elements.streetLayerToggle.checked ? state.map : null,
    position: { lat: photo.latitude, lng: photo.longitude },
    title: photo.provider === "google" ? "Google Street View panorama" : "KartaView photograph",
    icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: "#008b83", fillOpacity: 1, strokeColor: "#ffffff", strokeWeight: 3 },
  });
}

function renderStreetPhoto(photo) {
  state.streetSelectedId = photo.id;
  state.streetIndex = Math.max(0, state.streetSequence.findIndex((item) => item.id === photo.id));
  state.streetViewPanorama.setVisible(false);
  showStreetDisplay("kartaview");
  elements.kartaStreetStage.classList.remove("hidden");
  elements.streetViewerNavigation.classList.remove("hidden");
  elements.streetViewerProvider.textContent = "KARTAVIEW OPEN STREET IMAGERY";
  elements.streetViewerImage.src = photo.image_url;
  elements.streetViewerTime.textContent = shortDate(photo.captured_at);
  elements.streetViewerCoordinates.textContent = coordinateText(photo.latitude, photo.longitude);
  elements.streetViewerDistance.textContent = photo.distance_m == null ? "Not reported" : `${Number(photo.distance_m).toLocaleString()} m from search point`;
  elements.streetViewerSource.href = photo.details_url;
  elements.streetViewerSource.textContent = "KartaView source record";
  elements.street360Badge.classList.toggle("hidden", !photo.is_360);
  elements.streetViewerPosition.textContent = `${state.streetIndex + 1} of ${state.streetSequence.length}`;
  elements.previousStreetImage.disabled = state.streetIndex <= 0;
  elements.nextStreetImage.disabled = state.streetIndex >= state.streetSequence.length - 1;
  updateStreetMarker(photo);
  state.map.setCenter({ lat: photo.latitude, lng: photo.longitude });
  state.map.setZoom(Math.max(state.map.getZoom(), 16));
  renderStreetResults();
  renderStage7MapOverlay();
  document.querySelector(".map-panel").scrollIntoView({ behavior: "smooth", block: "center" });
}

function distanceBetweenMetres(latitudeA, longitudeA, latitudeB, longitudeB) {
  const radians = (value) => value * Math.PI / 180;
  const dLat = radians(latitudeB - latitudeA);
  const dLon = radians(longitudeB - longitudeA);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(radians(latitudeA)) * Math.cos(radians(latitudeB)) * Math.sin(dLon / 2) ** 2;
  return Math.round(6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

function googleStreetSourceUrl(latitude, longitude) {
  return `https://www.google.com/maps?q=${encodeURIComponent(`${latitude},${longitude}`)}&layer=c`;
}

function angularDistanceDegrees(first, second) {
  return Math.abs((((Number(first) - Number(second)) + 540) % 360) - 180);
}

function bearingDegrees(latitudeA, longitudeA, latitudeB, longitudeB) {
  const toRadians = (value) => value * Math.PI / 180;
  const latA = toRadians(latitudeA);
  const latB = toRadians(latitudeB);
  const deltaLongitude = toRadians(longitudeB - longitudeA);
  const y = Math.sin(deltaLongitude) * Math.cos(latB);
  const x = Math.cos(latA) * Math.sin(latB) - Math.sin(latA) * Math.cos(latB) * Math.cos(deltaLongitude);
  return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
}

function initialGoogleRoadHeading(data, queryLatitude, queryLongitude) {
  const links = (data?.links || [])
    .map((link) => Number(link?.heading))
    .filter(Number.isFinite)
    .map((heading) => ((heading % 360) + 360) % 360);
  if (!links.length) return 0;
  const location = data?.location?.latLng;
  if (location && Number.isFinite(queryLatitude) && Number.isFinite(queryLongitude)) {
    const latitude = Number(location.lat());
    const longitude = Number(location.lng());
    if (distanceBetweenMetres(latitude, longitude, queryLatitude, queryLongitude) >= 2) {
      const towardSelection = bearingDegrees(latitude, longitude, queryLatitude, queryLongitude);
      return links.reduce((best, heading) => angularDistanceDegrees(heading, towardSelection) < angularDistanceDegrees(best, towardSelection) ? heading : best, links[0]);
    }
  }
  return links[0];
}

function applyGoogleStreetMetadata(data, queryLatitude = null, queryLongitude = null) {
  const location = data?.location?.latLng;
  if (!location) return;
  const latitude = location.lat();
  const longitude = location.lng();
  const distance = queryLatitude == null ? null : distanceBetweenMetres(queryLatitude, queryLongitude, latitude, longitude);
  elements.streetViewerProvider.textContent = "GOOGLE STREET VIEW";
  elements.streetViewerTime.textContent = data.imageDate || "Capture month not reported";
  elements.streetViewerCoordinates.textContent = coordinateText(latitude, longitude);
  elements.streetViewerDistance.textContent = distance == null ? "Interactive navigation" : `${distance.toLocaleString()} m from selected point`;
  elements.streetViewerSource.href = googleStreetSourceUrl(latitude, longitude);
  elements.streetViewerSource.textContent = "Open location in Google Maps";
  updateStreetMarker({ latitude, longitude, provider: "google" });
}

function renderGoogleStreetPanorama(data, queryLatitude, queryLongitude) {
  revealStreetResults();
  state.streetResults = [];
  state.streetSequence = [];
  state.streetSelectedId = null;
  elements.streetCount.textContent = "1";
  elements.streetCaption.textContent = "Nearest outdoor Google Street View panorama.";
  elements.streetList.innerHTML = '<div class="street-result google-result"><span class="google-result-icon">360°</span><span><strong>Google Street View available</strong><small>Use the panorama displayed directly on the map.</small></span></div>';
  showStreetDisplay("google");
  elements.kartaStreetStage.classList.add("hidden");
  elements.streetViewerNavigation.classList.add("hidden");
  state.stage7MaterialCameraSignature = null;
  const roadHeading = initialGoogleRoadHeading(data, queryLatitude, queryLongitude);
  state.streetViewPanorama.setPano(data.location.pano);
  state.streetViewPanorama.setPov({ heading: roadHeading, pitch: 0 });
  state.streetViewPanorama.setZoom(1);
  state.streetViewPanorama.setVisible(true);
  applyGoogleStreetMetadata(data, queryLatitude, queryLongitude);
  const location = data.location.latLng;
  state.map.setCenter(location);
  state.map.setZoom(Math.max(state.map.getZoom(), 16));
  document.querySelector(".map-panel").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function selectStreetPhoto(photo, initialSequence = state.streetResults, loadFullSequence = false) {
  state.streetSequence = initialSequence.length ? initialSequence : [photo];
  renderStreetPhoto(photo);
  if (!loadFullSequence || !photo.sequence_id) return;
  try {
    const payload = await request(`api/street-imagery/sequences/${encodeURIComponent(photo.sequence_id)}?limit=150`);
    if (!payload.photos.length) return;
    state.streetSequence = payload.photos;
    const anchored = state.streetSequence.find((item) => item.id === photo.id) || state.streetSequence.reduce((nearest, item) => {
      if (photo.sequence_index == null || item.sequence_index == null) return nearest;
      return Math.abs(item.sequence_index - photo.sequence_index) < Math.abs((nearest.sequence_index ?? item.sequence_index) - photo.sequence_index) ? item : nearest;
    }, state.streetSequence[0]);
    renderStreetPhoto(anchored);
  } catch (error) {
    setStatus(elements.streetStatus, `Nearby image opened; the complete sequence could not be loaded: ${error.message}`, "warning");
  }
}

function navigateStreetViewer(offset) {
  const nextIndex = state.streetIndex + offset;
  if (nextIndex < 0 || nextIndex >= state.streetSequence.length) return;
  renderStreetPhoto(state.streetSequence[nextIndex]);
}

async function searchStreetImagery(latitude, longitude) {
  finishStreetSearchMode();
  const radius = Number(elements.streetRadius.value);
  state.selectedRoadPoint = { latitude, longitude, radius_m: radius };
  revealStreetResults();
  elements.streetList.innerHTML = '<div class="empty-state loading">Checking street-level coverage...</div>';
  setStatus(elements.streetStatus, `Checking the nearest outdoor Google Street View panorama within ${radius.toLocaleString()} m...`, "loading");
  let googleStatus = "unavailable";
  try {
    const requestGooglePanorama = (request) => new Promise((resolve) => {
      state.streetViewService.getPanorama(
        request,
        (data, status) => resolve({ data, status }),
      );
    });
    const baseRequest = {
      location: { lat: latitude, lng: longitude },
      radius,
      preference: google.maps.StreetViewPreference?.NEAREST || "nearest",
    };
    const outdoorSource = google.maps.StreetViewSource?.OUTDOOR;
    let result = outdoorSource
      ? await requestGooglePanorama({ ...baseRequest, sources: [outdoorSource] })
      : await requestGooglePanorama(baseRequest);
    if (String(result.status).toUpperCase() !== "OK" || !result.data?.location?.latLng) {
      result = await requestGooglePanorama(baseRequest);
    }
    googleStatus = result.status;
    if (String(result.status).toUpperCase() === "OK" && result.data?.location?.latLng) {
      renderGoogleStreetPanorama(result.data, latitude, longitude);
      setStatus(elements.streetStatus, "Google Street View coverage found. Preparing the visible image for road-condition analysis...", "loading");
      elements.mapInstruction.textContent = "The exact visible panorama, direction and zoom are being analysed. Any detections will appear directly on this map view.";
      await runStage7CurrentViewAnalysis();
      return;
    }
  } catch (error) {
    const detail = String(error?.message || error || "unknown").replace(/[^a-zA-Z0-9 .:_-]/g, " ").slice(0, 180);
    googleStatus = `service-error:${detail}`;
    console.error(`[Stage7] Google panorama lookup failed: ${detail}`);
  }
  console.info(`[Stage7] Google panorama lookup ended with status: ${googleStatus}`);
  await searchKartaViewFallback(latitude, longitude, radius, googleStatus);
  renderStage7Analysis({
    selected_point: { latitude, longitude, radius_m: radius },
    requested_provider: "google_current_panorama_frame",
    state: "analysis_not_possible",
    processed: false,
    message: `No Google panorama was loaded at this point (Google status: ${googleStatus}), so there is no exact current Google frame to analyse.`,
    reason_codes: ["no_google_panorama_loaded", `google_status_${String(googleStatus).toLowerCase()}`],
    imagery: null,
    model: null,
    detections: [],
  });
}

async function searchKartaViewFallback(latitude, longitude, radius, googleStatus) {
  setStatus(elements.streetStatus, "No usable Google outdoor panorama was returned. Checking the KartaView fallback...", "loading");
  try {
    const payload = await request(`api/street-imagery/nearby?latitude=${latitude}&longitude=${longitude}&radius_m=${radius}&limit=30`);
    state.streetResults = payload.photos;
    state.streetSequence = payload.photos;
    state.streetSelectedId = null;
    renderStreetResults();
    if (!payload.photos.length) {
      state.streetProvider = null;
      if (state.streetMarker) { state.streetMarker.setMap(null); state.streetMarker = null; }
      returnToMap();
      const googleDetail = String(googleStatus).toUpperCase() === "ZERO_RESULTS" ? "Google reported no outdoor panorama" : "Google Street View was unavailable";
      setStatus(elements.streetStatus, `${googleDetail}, and no KartaView imagery was found. This is a normal coverage gap; try another point or upload ground imagery.`, "warning");
      elements.mapInstruction.textContent = "No street-level coverage was found at this road point. The map and other imagery tools remain available.";
      showMapNotice("No Google Street View or KartaView coverage was found here. Try another road point or upload ground imagery.");
      return;
    }
    setStatus(elements.streetStatus, `Google coverage was unavailable; ${payload.count} nearby KartaView photograph${payload.count === 1 ? "" : "s"} found as the open fallback.`, "success");
    elements.mapInstruction.textContent = "KartaView fallback imagery found. Use Previous and Next to move through its captured road sequence.";
    await selectStreetPhoto(payload.photos[0], payload.photos, true);
  } catch (error) {
    state.streetResults = [];
    renderStreetResults();
    returnToMap();
    setStatus(elements.streetStatus, `Google Street View returned no usable panorama and the KartaView fallback could not be reached: ${error.message}`, "error");
    elements.mapInstruction.textContent = "Street-level imagery services could not return coverage. Other perception tools remain available.";
    showMapNotice("Street-level imagery could not be reached. The map remains available.");
  }
}

async function searchNearby(latitude, longitude) {
  try {
    const payload = await request(`api/images/nearby?latitude=${latitude}&longitude=${longitude}&radius_km=3&limit=30`);
    if (state.clickMarker) state.clickMarker.setMap(null);
    state.clickMarker = new google.maps.Marker({
      map: state.map,
      position: { lat: latitude, lng: longitude },
      title: "Ground-imagery search point",
      icon: { path: google.maps.SymbolPath.CIRCLE, scale: 6, fillColor: "#9c2647", fillOpacity: 1, strokeColor: "#ffffff", strokeWeight: 2 },
    });
    if (payload.images.length) { selectImage(payload.images[0].id, payload.images); elements.mapInstruction.textContent = `${payload.count} ground image${payload.count === 1 ? "" : "s"} found within 3 km.`; }
    else elements.mapInstruction.textContent = "No uploaded ground image is within 3 km of this point.";
  } catch (error) { elements.mapInstruction.textContent = error.message; }
}

async function deleteSelectedImage() {
  if (!state.selectedId || !window.confirm("Remove this image from the isolated perception catalogue?")) return;
  await request(`api/images/${encodeURIComponent(state.selectedId)}`, { method: "DELETE" });
  state.selectedId = null;
  elements.viewerContent.classList.add("hidden");
  elements.viewerEmpty.classList.remove("hidden");
  await loadImages();
}

function navigateViewer(offset) {
  const nextIndex = state.sequenceIndex + offset;
  if (nextIndex >= 0 && nextIndex < state.sequence.length) selectImage(state.sequence[nextIndex], state.sequence.map((id) => state.images.find((item) => item.id === id)).filter(Boolean));
}

async function saveProject() {
  if (!state.mapReady) { showGoogleConnectDialog(); setStatus(elements.projectStatus, "Connect Google imagery before saving the mapped survey.", "warning"); return; }
  state.aoi = localAreaFromMapCentre();
  const center = state.map.getCenter();
  const payload = { name: elements.projectName.value.trim() || "Untitled imagery survey", location_label: state.locationLabel, map_center: [center.lng(), center.lat()], map_zoom: state.map.getZoom(), aoi: state.aoi, collection: elements.collectionInput.value, start_date: elements.startDate.value, end_date: elements.endDate.value, max_cloud_cover: Number(elements.cloudCover.value), selected_scenes: [state.comparison.a, state.comparison.b].filter(Boolean).map(sceneSnapshot), notes: "Saved by the isolated infrastructure perception workspace. Google credentials and Street View session identifiers are excluded." };
  try { await request("api/survey-project", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); setStatus(elements.projectStatus, "Survey saved inside this isolated module.", "success"); }
  catch (error) { setStatus(elements.projectStatus, error.message, "error"); }
}

async function reopenProject() {
  if (!state.mapReady) { showGoogleConnectDialog(); setStatus(elements.projectStatus, "Connect Google imagery before reopening the mapped survey.", "warning"); return; }
  try {
    const payload = await request("api/survey-project");
    if (!payload.saved) { setStatus(elements.projectStatus, "No saved imagery survey is available.", "warning"); return; }
    const project = payload.project;
    elements.projectName.value = project.name;
    state.locationLabel = project.location_label || state.locationLabel;
    elements.mapTitle.textContent = state.locationLabel;
    if (project.map_center?.length === 2) {
      state.map.setCenter({ lat: project.map_center[1], lng: project.map_center[0] });
      state.map.setZoom(project.map_zoom ?? state.map.getZoom());
    }
    state.aoi = project.aoi || localAreaFromMapCentre();
    elements.collectionInput.value = project.collection || "sentinel-2-l2a";
    elements.startDate.value = project.start_date || elements.startDate.value;
    elements.endDate.value = project.end_date || elements.endDate.value;
    elements.cloudCover.value = project.max_cloud_cover ?? 40;
    elements.cloudValue.textContent = `${elements.cloudCover.value}%`;
    updateCollectionControls();
    state.comparison.a = project.selected_scenes?.[0] || null; state.comparison.b = project.selected_scenes?.[1] || null;
    renderComparison("a"); renderComparison("b");
    setStatus(elements.projectStatus, "Saved survey reopened. Run the catalogue search to restore live previews.", "success");
  } catch (error) { setStatus(elements.projectStatus, error.message, "error"); }
}

async function clearProject() {
  try { await request("api/survey-project", { method: "DELETE" }); setStatus(elements.projectStatus, "Saved survey removed. Current map work remains open.", "success"); }
  catch (error) { setStatus(elements.projectStatus, error.message, "error"); }
}

const STAGE8_SURFACES = Object.freeze(["road", "kerb", "footpath"]);
const STAGE8_DEFECT_CLASSES = Object.freeze([
  "alligator_cracking",
  "longitudinal_crack",
  "transverse_crack",
  "pothole",
  "standing_water",
  "broken_kerb",
  "uneven_or_broken_footpath",
  "debris_or_vegetation_obstruction",
]);

function stage8Clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function stage8Id(prefix = "stage8") {
  if (window.crypto?.randomUUID) return `${prefix}-${window.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function stage8Clamp(value, minimum = 0, maximum = 1) {
  return Math.max(minimum, Math.min(maximum, Number(value)));
}

function stage8NormalisedRegion(detection) {
  const source = Array.isArray(detection?.box)
    ? { x_min: detection.box[0], y_min: detection.box[1], x_max: detection.box[2], y_max: detection.box[3] }
    : detection?.evidence_region;
  if (!source || source.type && source.type !== "bounding_box") return null;
  const box = [source.x_min, source.y_min, source.x_max, source.y_max].map(Number);
  if (!box.every(Number.isFinite)) return null;
  const normalised = [
    stage8Clamp(Math.min(box[0], box[2])),
    stage8Clamp(Math.min(box[1], box[3])),
    stage8Clamp(Math.max(box[0], box[2])),
    stage8Clamp(Math.max(box[1], box[3])),
  ];
  if (normalised[2] - normalised[0] < 0.001 || normalised[3] - normalised[1] < 0.001) return null;
  return normalised;
}

function stage8OriginalDetections(payload) {
  return (payload?.detections || payload?.observations || []).flatMap((detection, index) => {
    const box = stage8NormalisedRegion(detection);
    if (!box) return [];
    return [{
      detection_id: String(detection.detection_id || detection.id || `model-detection-${index + 1}`),
      damage_type: String(detection.damage_type || "visible_road_anomaly"),
      ...(detection.surface_type ? { surface_type: String(detection.surface_type) } : {}),
      ...(Number.isFinite(Number(detection.confidence)) ? { confidence: stage8Clamp(detection.confidence) } : {}),
      box,
      ...(detection.consensus && typeof detection.consensus === "object" ? { consensus: stage8Clone(detection.consensus) } : {}),
    }];
  });
}

function stage8EditableBoxes(originalDetections) {
  return originalDetections.map((detection, index) => {
    const resolvedDamage = STAGE8_DEFECT_CLASSES.includes(detection.damage_type);
    const resolvedSurface = STAGE8_SURFACES.includes(detection.surface_type);
    return {
      annotation_box_id: stage8Id(`model-box-${index + 1}`),
      origin: "model",
      damage_type: resolvedDamage ? detection.damage_type : "uncertain",
      surface_type: resolvedSurface ? detection.surface_type : null,
      uncertain: !(resolvedDamage && resolvedSurface),
      box: [...detection.box],
      source_detection_id: detection.detection_id,
      ...(detection.confidence != null ? { model_confidence: detection.confidence } : {}),
    };
  });
}

function stage8BoxStyle(box) {
  return `left:${box[0] * 100}%;top:${box[1] * 100}%;width:${(box[2] - box[0]) * 100}%;height:${(box[3] - box[1]) * 100}%`;
}

function stage8OriginalOverlayHtml(detections) {
  return detections.map((detection) => `<div class="stage8-original-box" style="${stage8BoxStyle(detection.box)}" title="Original model box: ${escapeHtml(stage6Label(detection.damage_type))}"></div>`).join("");
}

function stage8Snapshot(review) {
  return {
    boxes: stage8Clone(review.boxes),
    selectedId: review.selectedId,
    confirmedIds: stage8Clone(review.confirmedIds || []),
    cleanFrame: review.cleanFrame,
    note: elements.stage8ReviewNote.value,
  };
}

function pushStage8History(action) {
  const review = state.stage8Review;
  if (!review) return;
  review.history.push({ ...stage8Snapshot(review), action });
  if (review.history.length > 100) review.history.shift();
}

function stage8Change(action, details = {}) {
  const review = state.stage8Review;
  if (!review) return;
  review.changeLog.push({ action, at: new Date().toISOString(), ...details });
  review.dirty = true;
}

function stage8SelectedBox() {
  const review = state.stage8Review;
  return review?.boxes.find((box) => box.annotation_box_id === review.selectedId) || null;
}

function stage8BoxResolved(box) {
  return Boolean(
    box
    && box.uncertain !== true
    && STAGE8_SURFACES.includes(box.surface_type)
    && STAGE8_DEFECT_CLASSES.includes(box.damage_type)
  );
}

function stage8UnresolvedBoxes(boxes = []) {
  return boxes.filter((box) => !stage8BoxResolved(box));
}

function setStage8ReviewWorkflow(kind, badge, title, message, { enabled = false, action = "Review on the large image" } = {}) {
  elements.stage8ReviewStatusCard.className = `stage8-review-status-card ${kind}`;
  elements.stage8ReviewStatusBadge.textContent = badge;
  elements.stage8ReviewStatusTitle.textContent = title;
  elements.stage8ReviewStatusMessage.textContent = message;
  elements.reviewStage8Annotations.classList.remove("hidden");
  elements.reviewStage8Annotations.disabled = !enabled;
  elements.reviewStage8Annotations.textContent = action;
}

function selectStage8Box(boxId) {
  const review = state.stage8Review;
  if (!review || !review.boxes.some((box) => box.annotation_box_id === boxId)) return;
  review.selectedId = boxId;
  review.addMode = false;
  review.pointer = null;
  renderStage8Editor();
}

function renderStage8IssueSwitcher() {
  const review = state.stage8Review;
  const boxes = review?.boxes || [];
  elements.stage8IssueSwitcher.classList.toggle("hidden", !review || !boxes.length);
  if (!review || !boxes.length) {
    elements.stage8IssueTabs.replaceChildren();
    return;
  }
  elements.stage8IssueTabs.innerHTML = boxes.map((box, index) => {
    const selected = box.annotation_box_id === review.selectedId;
    const applied = (review.confirmedIds || []).includes(box.annotation_box_id);
    const resolved = stage8BoxResolved(box);
    const stateClass = applied ? "applied" : (resolved ? "ready" : "attention");
    const stateLabel = applied ? "Applied" : (resolved ? "Ready" : "Check");
    const label = resolved ? `${stage6Label(box.surface_type)}, ${stage6Label(box.damage_type)}` : "location or issue type needs review";
    return `<button type="button" data-stage8-select-box="${escapeHtml(box.annotation_box_id)}" class="${selected ? "selected " : ""}${stateClass}" aria-pressed="${selected ? "true" : "false"}" aria-label="Issue ${index + 1}: ${escapeHtml(label)}"><b>${index + 1}</b><small>${stateLabel}</small></button>`;
  }).join("");
}

function renderStage8SelectedControls() {
  const review = state.stage8Review;
  const selected = stage8SelectedBox();
  elements.stage8SelectedControls.hidden = !selected;
  elements.stage8InlineEditor.classList.toggle("hidden", !selected);
  if (selected) {
    const selectedIndex = review.boxes.findIndex((box) => box.annotation_box_id === selected.annotation_box_id) + 1;
    const resolved = stage8BoxResolved(selected);
    elements.stage8Surface.value = selected.surface_type || "";
    elements.stage8DefectClass.value = STAGE8_DEFECT_CLASSES.includes(selected.damage_type) ? selected.damage_type : "uncertain";
    elements.stage8SelectionSummary.innerHTML = `<strong>Issue ${selectedIndex} selected</strong><span>${resolved ? `${escapeHtml(stage6Label(selected.surface_type))} · ${escapeHtml(stage6Label(selected.damage_type))}` : "Choose where the issue is and what type it is."}</span>`;
    elements.stage8ConfirmBox.disabled = !resolved;
  } else {
    elements.stage8SelectionSummary.innerHTML = review?.addMode
      ? "<strong>Draw a new box</strong><span>Drag around the missed road, kerb or footpath issue on the large image.</span>"
      : "<strong>Select an issue</strong><span>Choose a number above. This also reaches boxes hidden below overlapping boxes.</span>";
  }
  elements.stage8Undo.disabled = !review?.history.length;
  elements.stage8MarkClean.classList.toggle("active", Boolean(review?.cleanFrame));
  elements.stage8AddBox.classList.toggle("active", Boolean(review?.addMode));
  elements.stage8EditorOverlay.classList.toggle("adding", Boolean(review?.addMode));
  elements.stage8MapEditorOverlay.classList.toggle("adding", Boolean(review?.addMode));
  const unresolved = stage8UnresolvedBoxes(review?.boxes || []).length;
  elements.stage8EditorState.textContent = review?.cleanFrame
    ? "Image marked clear"
    : (unresolved ? `${unresolved} need attention` : "Ready to save");
}

function renderStage8Editor() {
  const review = state.stage8Review;
  if (!review) return;
  elements.stage8OriginalOverlay.replaceChildren();
  elements.stage8MapOriginalOverlay.replaceChildren();
  let editorHtml = review.boxes.map((box, index) => {
    const selected = box.annotation_box_id === review.selectedId;
    const confirmed = (review.confirmedIds || []).includes(box.annotation_box_id);
    const label = box.uncertain
      ? "Choose location and issue type"
      : `${stage6Label(box.surface_type)} · ${stage6Label(box.damage_type)}`;
    const colour = STAGE7_CLASS_COLOURS[box.damage_type] || "#16ddc7";
    return `<div class="stage8-edit-box ${box.origin === "human" ? "is-human" : ""} ${confirmed ? "confirmed" : ""} ${selected ? "selected" : ""}" data-box-id="${escapeHtml(box.annotation_box_id)}" style="--stage8-color:${colour};${stage8BoxStyle(box.box)}"><b class="stage8-box-index">${index + 1}</b><span>${escapeHtml(label)}</span>${selected ? '<i class="stage8-resize-handle" aria-label="Resize selected issue box"></i>' : ""}</div>`;
  }).join("");
  if (review.pointer?.kind === "create" && review.pointer.draftBox) {
    editorHtml += `<div class="stage8-draft-box" style="${stage8BoxStyle(review.pointer.draftBox)}"></div>`;
  }
  elements.stage8EditorOverlay.innerHTML = editorHtml;
  elements.stage8MapEditorOverlay.innerHTML = editorHtml;
  renderStage8IssueSwitcher();
  renderStage8SelectedControls();
  updateStage8ReviewAvailability();
}

function stage8ReviewEligible(payload = state.stage7Analysis) {
  const previewUrl = payload?.imagery?.preview_data_url || payload?.imagery?.thumbnail_url || payload?.imagery?.image_url;
  return Boolean(
    state.stage8Status?.status === "human_annotation_ready"
    && payload?.processed
    && payload?.frame?.frame_id
    && payload?.frame?.pano_id
    && /^[A-Fa-f0-9]{64}$/.test(String(payload?.imagery?.frame_sha256 || ""))
    && payload?.model?.version
    && /^[A-Fa-f0-9]{64}$/.test(String(payload?.model?.checkpoint_sha256 || ""))
    && previewUrl
  );
}

function updateStage8ReviewAvailability(payload = state.stage7Analysis) {
  const eligible = stage8ReviewEligible(payload);
  if (state.stage8Review) {
    const unresolved = stage8UnresolvedBoxes(state.stage8Review.boxes).length;
    const confirmed = new Set(state.stage8Review.confirmedIds || []);
    const appliedCount = state.stage8Review.boxes.filter((box) => confirmed.has(box.annotation_box_id) && stage8BoxResolved(box)).length;
    const pendingCount = state.stage8Review.boxes.length - appliedCount;
    const readyToSave = state.stage8Review.cleanFrame || appliedCount > 0;
    setStage8ReviewWorkflow(
      "editing",
      "Editing",
      "Your corrected image is shown live",
      readyToSave
        ? `${appliedCount} corrected issue${appliedCount === 1 ? " is" : "s are"} ready for this review record${pendingCount ? `; ${pendingCount} unapplied box${pendingCount === 1 ? " stays" : "es stay"} pending and remain excluded from the confirmed review` : ""}. Use the final Confirm button to submit exactly this unit.`
        : (unresolved
          ? `${unresolved} box${unresolved === 1 ? " needs" : "es need"} review. Apply at least one correction before the final Confirm button becomes available.`
          : "Choose one or more issues and use Apply to image. Apply only updates the preview; the final Confirm button creates the review record."),
      { enabled: readyToSave, action: "Confirm corrected image and save to TWM" },
    );
    return;
  }
  const saved = state.stage8LastSavedReview;
  if (saved && payload?.frame?.frame_id === saved.frameId) {
    const annotations = saved.record?.human_review?.annotations || [];
    const pending = saved.record?.human_review?.pending_annotations || [];
    const clean = saved.record?.human_review?.clean_frame === true;
    setStage8ReviewWorkflow(
      pending.length ? "attention" : "saved",
      pending.length ? "Partially saved" : "Saved",
      pending.length ? "Confirmed corrections saved; other boxes remain pending" : "Review saved successfully",
      clean
        ? "This image was saved as having no relevant road, kerb or footpath issue."
        : `${annotations.length} confirmed issue${annotations.length === 1 ? " was" : "s were"} saved as one review record${pending.length ? `; ${pending.length} box${pending.length === 1 ? " remains" : "es remain"} pending and excluded from the confirmed review` : ""}.`,
      { enabled: eligible, action: "Review this image again" },
    );
    return;
  }
  if (eligible) {
    const count = (payload?.detections || payload?.observations || []).length;
    setStage8ReviewWorkflow(
      "ready",
      "Ready",
      "Check and correct this result",
      count
        ? `${count} possible issue${count === 1 ? " is" : "s are"} ready. Select boxes on the large image, correct their labels or add anything missed.`
        : "No issue was detected. Review the large image, add anything missed or confirm that the image is clear.",
      { enabled: true },
    );
    return;
  }
  setStage8ReviewWorkflow(
    "waiting",
    "Waiting",
    "Check the current detection result",
    "Analyse a Street View image first. You can then select, correct, remove or add boxes directly on the large image.",
  );
}

function beginStage8Review() {
  const payload = state.stage7Analysis;
  if (!stage8ReviewEligible(payload)) {
    setStage8ReviewWorkflow("waiting", "Waiting", "Analyse this image first", "A completed current-image result is required before boxes can be reviewed on the large image.");
    return;
  }
  const originals = stage8OriginalDetections(payload);
  const latestRecord = stage8LatestRecordForFrame(payload);
  const latestPending = latestRecord?.human_review?.pending_annotations;
  const boxes = Array.isArray(latestPending) && latestPending.length
    ? stage8Clone(latestPending)
    : stage8EditableBoxes(originals);
  state.stage8Review = {
    frameId: payload.frame.frame_id,
    frameRevision: Number(payload.frame.frame_revision),
    startedAt: new Date().toISOString(),
    submissionId: stage8Id("stage8-submission"),
    originalDetections: originals,
    initialBoxes: stage8Clone(boxes),
    boxes,
    selectedId: boxes[0]?.annotation_box_id || null,
    confirmedIds: [],
    history: [],
    changeLog: [],
    dirty: false,
    cleanFrame: false,
    addMode: false,
    pointer: null,
  };
  elements.stage8ReviewNote.value = "";
  elements.stage7AnalysisImageWrap.classList.add("stage8-reviewing");
  elements.stage8EditorOverlay.classList.remove("hidden");
  elements.stage8MapEditorOverlay.classList.remove("hidden");
  elements.stage8MapReviewBanner.classList.remove("hidden");
  elements.stage7GoogleOverlay.classList.add("review-hidden");
  document.querySelector(".map-stage").classList.add("reviewing");
  elements.stage7AnalysisCard.classList.add("reviewing");
  elements.stage8EditorPanel.classList.remove("hidden");
  elements.stage8EditorPanel.open = !boxes.length;
  elements.mapInstruction.textContent = "Review mode is open. Select a numbered box on the image or use the number buttons on the right; overlapping boxes remain independently reachable.";
  setStatus(
    elements.stage8EditorMessage,
    boxes.length
      ? "Use the numbered list to reach every box, including boxes hidden below an overlap."
      : "No box was retained. Add every relevant issue visible in this image, or mark the image clear.",
  );
  renderStage8Editor();
  updateStage8ReviewAvailability(payload);
}

function endStage8Review({ silent = false, message = "" } = {}) {
  if (!state.stage8Review) return true;
  if (!silent && state.stage8Review.dirty && !window.confirm("Discard the unsaved review of this Street View image?")) return false;
  state.stage8Review = null;
  elements.stage8OriginalOverlay.replaceChildren();
  elements.stage8EditorOverlay.replaceChildren();
  elements.stage8MapOriginalOverlay.replaceChildren();
  elements.stage8MapEditorOverlay.replaceChildren();
  elements.stage8OriginalOverlay.classList.add("hidden");
  elements.stage8EditorOverlay.classList.add("hidden");
  elements.stage8EditorOverlay.classList.remove("adding");
  elements.stage8MapOriginalOverlay.classList.add("hidden");
  elements.stage8MapEditorOverlay.classList.add("hidden");
  elements.stage8MapEditorOverlay.classList.remove("adding");
  elements.stage8MapReviewBanner.classList.add("hidden");
  elements.stage8InlineEditor.classList.add("hidden");
  elements.stage8IssueSwitcher.classList.add("hidden");
  elements.stage8IssueTabs.replaceChildren();
  elements.stage7GoogleOverlay.classList.remove("review-hidden");
  document.querySelector(".map-stage").classList.remove("reviewing");
  elements.stage7AnalysisCard.classList.remove("reviewing");
  elements.stage8EditorPanel.classList.add("hidden");
  elements.stage8EditorPanel.open = false;
  elements.stage7AnalysisImageWrap.classList.remove("stage8-reviewing");
  if (state.stage7Analysis?.frame) renderStage7GoogleOverlay(state.stage7Analysis);
  if (state.streetProvider === "google" && state.streetDisplayMode) {
    elements.mapInstruction.textContent = "Street View navigation is active. Move the camera to inspect another image, or review the current detection result again.";
  }
  updateStage8ReviewAvailability();
  if (message) setStatus(elements.streetStatus, message, "warning");
  return true;
}

function stage8OverlayPoint(event) {
  const bounds = elements.stage8MapEditorOverlay.getBoundingClientRect();
  return {
    x: stage8Clamp((event.clientX - bounds.left) / Math.max(1, bounds.width)),
    y: stage8Clamp((event.clientY - bounds.top) / Math.max(1, bounds.height)),
  };
}

function stage8PointerDown(event) {
  const review = state.stage8Review;
  if (!review || event.button !== 0) return;
  const point = stage8OverlayPoint(event);
  const boxElement = event.target.closest?.(".stage8-edit-box");
  if (review.addMode && !boxElement) {
    review.pointer = { kind: "create", start: point, draftBox: [point.x, point.y, point.x, point.y] };
    renderStage8Editor();
    event.preventDefault();
    return;
  }
  if (!boxElement) {
    review.selectedId = null;
    renderStage8Editor();
    return;
  }
  const boxId = boxElement.dataset.boxId;
  const box = review.boxes.find((item) => item.annotation_box_id === boxId);
  if (!box) return;
  review.selectedId = boxId;
  pushStage8History(event.target.closest?.(".stage8-resize-handle") ? "resize_box" : "move_box");
  review.pointer = {
    kind: event.target.closest?.(".stage8-resize-handle") ? "resize" : "move",
    start: point,
    startBox: [...box.box],
    boxId,
  };
  renderStage8Editor();
  event.preventDefault();
}

function stage8PointerMove(event) {
  const review = state.stage8Review;
  const pointer = review?.pointer;
  if (!review || !pointer) return;
  const point = stage8OverlayPoint(event);
  if (pointer.kind === "create") {
    pointer.draftBox = [
      Math.min(pointer.start.x, point.x), Math.min(pointer.start.y, point.y),
      Math.max(pointer.start.x, point.x), Math.max(pointer.start.y, point.y),
    ];
  } else {
    const box = review.boxes.find((item) => item.annotation_box_id === pointer.boxId);
    if (!box) return;
    if (pointer.kind === "move") {
      const width = pointer.startBox[2] - pointer.startBox[0];
      const height = pointer.startBox[3] - pointer.startBox[1];
      const left = stage8Clamp(pointer.startBox[0] + point.x - pointer.start.x, 0, 1 - width);
      const top = stage8Clamp(pointer.startBox[1] + point.y - pointer.start.y, 0, 1 - height);
      box.box = [left, top, left + width, top + height];
    } else {
      box.box = [pointer.startBox[0], pointer.startBox[1], Math.max(pointer.startBox[0] + 0.015, point.x), Math.max(pointer.startBox[1] + 0.015, point.y)];
      box.box[2] = stage8Clamp(box.box[2]);
      box.box[3] = stage8Clamp(box.box[3]);
    }
  }
  renderStage8Editor();
  event.preventDefault();
}

function stage8PointerUp(event) {
  const review = state.stage8Review;
  const pointer = review?.pointer;
  if (!review || !pointer) return;
  if (pointer.kind === "create") {
    const box = pointer.draftBox;
    if (box && box[2] - box[0] >= 0.015 && box[3] - box[1] >= 0.015) {
      pushStage8History("add_box");
      const annotation = {
        annotation_box_id: stage8Id("human-box"),
        origin: "human",
        damage_type: "uncertain",
        surface_type: null,
        uncertain: true,
        box: [...box],
        source_detection_id: null,
      };
      review.boxes.push(annotation);
      review.selectedId = annotation.annotation_box_id;
      review.cleanFrame = false;
      stage8Change("add_box", { annotation_box_id: annotation.annotation_box_id });
    }
  } else {
    stage8Change(pointer.kind === "move" ? "move_box" : "resize_box", { annotation_box_id: pointer.boxId });
  }
  review.pointer = null;
  review.addMode = false;
  renderStage8Editor();
  event.preventDefault();
}

function stage8SelectChange() {
  const review = state.stage8Review;
  const box = stage8SelectedBox();
  if (!review || !box) return;
  pushStage8History("change_label");
  const defect = elements.stage8DefectClass.value;
  const surface = elements.stage8Surface.value;
  box.uncertain = defect === "uncertain" || !defect || !surface;
  box.damage_type = defect || "uncertain";
  box.surface_type = surface || null;
  review.confirmedIds = (review.confirmedIds || []).filter((id) => id !== box.annotation_box_id);
  stage8Change("change_label", { annotation_box_id: box.annotation_box_id, damage_type: box.damage_type, surface_type: box.surface_type, uncertain: box.uncertain });
  renderStage8Editor();
}

function confirmStage8SelectedBox() {
  const review = state.stage8Review;
  const box = stage8SelectedBox();
  if (!review || !box) return;
  if (!stage8BoxResolved(box)) {
    setStatus(elements.stage8EditorMessage, "Choose a road, kerb or footpath location and a specific issue type before confirming this box.", "warning");
    return;
  }
  pushStage8History("confirm_box");
  if (!review.confirmedIds.includes(box.annotation_box_id)) review.confirmedIds.push(box.annotation_box_id);
  stage8Change("confirm_box", { annotation_box_id: box.annotation_box_id, damage_type: box.damage_type, surface_type: box.surface_type });
  const index = review.boxes.findIndex((item) => item.annotation_box_id === box.annotation_box_id) + 1;
  setStatus(elements.stage8EditorMessage, `Issue ${index} is applied to this preview. It enters the review queue only when you use the final Confirm button.`, "success");
  renderStage8Editor();
  updateStage8ReviewAvailability();
}

function deleteStage8SelectedBox() {
  const review = state.stage8Review;
  const box = stage8SelectedBox();
  if (!review || !box) return;
  pushStage8History("delete_box");
  review.boxes = review.boxes.filter((item) => item.annotation_box_id !== box.annotation_box_id);
  review.confirmedIds = (review.confirmedIds || []).filter((id) => id !== box.annotation_box_id);
  review.selectedId = review.boxes[0]?.annotation_box_id || null;
  stage8Change("delete_box", { annotation_box_id: box.annotation_box_id });
  renderStage8Editor();
}

function undoStage8Change() {
  const review = state.stage8Review;
  const previous = review?.history.pop();
  if (!review || !previous) return;
  review.boxes = previous.boxes;
  review.selectedId = previous.selectedId;
  review.confirmedIds = previous.confirmedIds || [];
  review.cleanFrame = previous.cleanFrame;
  elements.stage8ReviewNote.value = previous.note || "";
  review.pointer = null;
  review.addMode = false;
  review.dirty = true;
  review.changeLog.push({ action: "undo", at: new Date().toISOString(), reversed: previous.action });
  renderStage8Editor();
}

function resetStage8Review() {
  const review = state.stage8Review;
  if (!review) return;
  pushStage8History("reset_to_model_result");
  review.boxes = stage8Clone(review.initialBoxes);
  review.selectedId = review.boxes[0]?.annotation_box_id || null;
  review.confirmedIds = [];
  review.cleanFrame = false;
  review.addMode = false;
  review.pointer = null;
  stage8Change("reset_to_model_result");
  renderStage8Editor();
}

function markStage8FrameClean() {
  const review = state.stage8Review;
  if (!review) return;
  if (review.boxes.length && !window.confirm("Mark this exact frame clean and remove every editable annotation box?")) return;
  pushStage8History("mark_frame_clean");
  review.boxes = [];
  review.selectedId = null;
  review.confirmedIds = [];
  review.cleanFrame = true;
  review.addMode = false;
  stage8Change("mark_frame_clean");
  renderStage8Editor();
}

function stage8SafeSourceModel(model) {
  return {
    id: String(model?.id || model?.name || "stage7-model"),
    name: String(model?.name || model?.id || "Stage 7 exact-current-view model"),
    version: String(model?.version || ""),
    checkpoint_sha256: String(model?.checkpoint_sha256 || ""),
    ...(model?.components && typeof model.components === "object" ? { components: stage8Clone(model.components) } : {}),
    ...(model?.strategy ? { strategy: String(model.strategy) } : {}),
  };
}

function stage8SavePayload() {
  const review = state.stage8Review;
  const analysis = state.stage7Analysis;
  if (!review || !analysis?.frame) throw new Error("The exact current frame is no longer available.");
  if (analysis.frame.frame_id !== review.frameId || Number(analysis.frame.frame_revision) !== review.frameRevision) {
    throw new Error("The Street View frame changed. Start a new review on the current view.");
  }
  const confirmedIds = new Set(review.confirmedIds || []);
  const confirmedAnnotations = review.boxes.filter((box) => confirmedIds.has(box.annotation_box_id) && stage8BoxResolved(box));
  const pendingAnnotations = review.boxes.filter((box) => !confirmedIds.has(box.annotation_box_id));
  if (!review.cleanFrame && !confirmedAnnotations.length) {
    throw new Error("Apply at least one corrected issue, then use the final Confirm button. Unapplied boxes stay pending and are never treated as clear road.");
  }
  const submissionScope = review.cleanFrame
    ? "clean_frame"
    : (pendingAnnotations.length ? "partial_review" : "complete_review");
  const frame = analysis.frame;
  const selected = analysis.selected_point || {};
  return {
    submission_id: review.submissionId,
    draft_started_at: review.startedAt,
    frame: {
      frame_id: String(frame.frame_id),
      pano_id: String(frame.pano_id),
      frame_revision: Number(frame.frame_revision),
      frame_sha256: String(analysis.imagery.frame_sha256).toUpperCase(),
      latitude: Number(frame.latitude), longitude: Number(frame.longitude),
      heading: Number(frame.heading), pitch: Number(frame.pitch), zoom: Number(frame.zoom), fov: Number(frame.fov),
      width: Number(frame.width), height: Number(frame.height),
      viewport_width: Number(frame.viewport_width), viewport_height: Number(frame.viewport_height),
    },
    selected_point: {
      latitude: Number(selected.latitude ?? frame.latitude),
      longitude: Number(selected.longitude ?? frame.longitude),
      ...(Number(selected.radius_m) >= 25 ? { radius_m: Number(selected.radius_m) } : {}),
    },
    inference_mode: ["local", "enhanced", "local_fallback"].includes(analysis.inference_mode) ? analysis.inference_mode : "local",
    source_model: stage8SafeSourceModel(analysis.model),
    original_detections: stage8Clone(review.originalDetections),
    annotations: stage8Clone(confirmedAnnotations),
    pending_annotations: stage8Clone(pendingAnnotations),
    submission_scope: submissionScope,
    ignore_unreviewed_regions: submissionScope === "partial_review",
    clean_frame: review.cleanFrame,
    reviewer_id: "human-reviewer",
    note: elements.stage8ReviewNote.value.trim() || null,
    change_log: stage8Clone(review.changeLog),
  };
}

async function saveStage8Review() {
  if (!state.stage8Review) return;
  let payload;
  try {
    payload = stage8SavePayload();
  } catch (error) {
    setStatus(elements.stage8EditorMessage, error.message, "warning");
    return;
  }
  elements.stage8Save.disabled = true;
  elements.reviewStage8Annotations.disabled = true;
  setStatus(elements.stage8EditorMessage, "Saving this reviewed image for human review...");
  try {
    const result = await request("api/stage8/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const record = result.record || {};
    state.stage8Queue.summary = result.summary || state.stage8Queue.summary;
    state.stage8Review.dirty = false;
    state.stage8LastSavedReview = { frameId: state.stage8Review.frameId, record };
    endStage8Review({ silent: true });
    await loadStage8Queue();
    elements.stage8QueuePanel.open = false;
    const trainingCount = Number(record.training_unit?.annotation_count || 0);
    const pendingCount = Number(record.training_unit?.pending_annotation_count || 0);
    setStatus(
      elements.streetStatus,
      record.human_review?.clean_frame
        ? "The explicitly clear image was saved as one review record."
        : `${trainingCount} confirmed issue${trainingCount === 1 ? "" : "s"} saved as one review record${pendingCount ? `; ${pendingCount} box${pendingCount === 1 ? " remains" : "es remain"} pending and remain excluded from the confirmed review` : ""}.`,
      "success",
    );
    updateStage8ReviewAvailability(state.stage7Analysis);
  } catch (error) {
    setStatus(elements.stage8EditorMessage, error.message, "error");
  } finally {
    elements.stage8Save.disabled = false;
    updateStage8ReviewAvailability(state.stage7Analysis);
  }
}

function renderStage8Queue() {
  const summary = state.stage8Queue.summary || {};
  const records = state.stage8Queue.records || [];
  elements.stage8QueueTotal.textContent = String(summary.total || 0);
  const cells = [
    [summary.training_units || 0, "Review records"],
    [summary.training_annotations || 0, "Confirmed issues"],
    [summary.partial_ready_for_training || 0, "Partial units"],
    [summary.pending_annotations || 0, "Pending boxes"],
    [summary.clean || 0, "Clear images"],
    [summary.ready_for_training || 0, "Confirmed reviews"],
  ];
  elements.stage8QueueSummary.innerHTML = cells.map(([value, label]) => `<span><b>${Number(value)}</b><i>${escapeHtml(label)}</i></span>`).join("");
  elements.stage8QueueRecords.innerHTML = records.length
    ? records.map((record) => {
      const boxCount = record.human_review?.annotations?.length || 0;
      const pendingCount = record.human_review?.pending_annotations?.length || 0;
      const frameId = String(record.frame?.frame_id || "Unknown frame");
      const friendlyStatus = record.status === "needs_resolution"
        ? "Needs attention"
        : (record.status === "partial_ready_for_training"
          ? "Partial review record"
          : (record.status === "ready_for_training" ? "Complete review record" : stage6Label(record.status)));
      return `<article class="stage8-queue-record"><strong title="${escapeHtml(frameId)}">Reviewed Street View image</strong><span>${escapeHtml(friendlyStatus)}</span><small>Saved review ${Number(record.review_version || 1)} · ${boxCount} confirmed${pendingCount ? ` · ${pendingCount} pending` : ""} · ${escapeHtml(stage6Label(record.inference_mode || "local"))}</small></article>`;
    }).join("")
    : '<div class="empty-state">No reviewed frames have been saved.</div>';
  renderAiMethodStage8Queue(summary);
}

function renderAiMethodStage8Queue(summary = {}) {
  const value = (name) => summary.unavailable ? "Unavailable" : String(Number(summary[name] || 0));
  if (elements.aiStage8TrainingUnits) elements.aiStage8TrainingUnits.textContent = value("training_units");
  if (elements.aiStage8TrainingAnnotations) elements.aiStage8TrainingAnnotations.textContent = value("training_annotations");
  if (elements.aiStage8PartialUnits) elements.aiStage8PartialUnits.textContent = value("partial_ready_for_training");
  if (elements.aiStage8PendingAnnotations) elements.aiStage8PendingAnnotations.textContent = value("pending_annotations");
}

async function loadStage8Queue() {
  const payload = await request("api/stage8/queue?limit=50");
  state.stage8Queue = { records: payload.records || [], summary: payload.summary || {} };
  renderStage8Queue();
  return payload;
}

async function loadStage8Status() {
  const payload = await request("api/stage8");
  state.stage8Status = payload;
  updateStage8ReviewAvailability();
  return payload;
}

function bindEvents() {
  elements.aiMethodButton.addEventListener("click", showAiMethodDialog);
  elements.closeAiMethod.addEventListener("click", closeAiMethodDialog);
  elements.runContractChecks.addEventListener("click", runContractChecks);
  elements.aiMethodDialog.addEventListener("click", (event) => { if (event.target === elements.aiMethodDialog) closeAiMethodDialog(); });
  elements.connectGoogleImagery.addEventListener("click", showGoogleConnectDialog);
  elements.connectGoogleFromMap.addEventListener("click", showGoogleConnectDialog);
  elements.googleConnectForm.addEventListener("submit", connectGoogleImagery);
  elements.cancelGoogleConnect.addEventListener("click", closeGoogleConnectDialog);
  elements.connectStage7Vision.addEventListener("click", showStage7VisionDialog);
  elements.stage7VisionForm.addEventListener("submit", connectStage7Vision);
  elements.cancelStage7Vision.addEventListener("click", closeStage7VisionDialog);
  elements.disableStage7Vision.addEventListener("click", disableStage7Vision);
  elements.satelliteForm.addEventListener("submit", searchSatellite);
  elements.collectionInput.addEventListener("change", updateCollectionControls);
  elements.cloudCover.addEventListener("input", () => { elements.cloudValue.textContent = `${elements.cloudCover.value}%`; });
  elements.startStreetSearch.addEventListener("click", beginStreetSearch);
  elements.streetRadius.addEventListener("input", () => { elements.streetRadiusValue.textContent = `${Number(elements.streetRadius.value).toLocaleString()} m`; });
  elements.previousStreetImage.addEventListener("click", () => navigateStreetViewer(-1));
  elements.nextStreetImage.addEventListener("click", () => navigateStreetViewer(1));
  elements.retryStage7Analysis.addEventListener("click", () => {
    if (!endStage8Review()) return;
    runStage7CurrentViewAnalysis();
  });
  elements.reviewStage8Annotations.addEventListener("click", () => {
    if (state.stage8Review) saveStage8Review();
    else beginStage8Review();
  });
  elements.stage8AddBox.addEventListener("click", () => {
    const review = state.stage8Review;
    if (!review) return;
    review.addMode = !review.addMode;
    review.pointer = null;
    renderStage8Editor();
  });
  elements.stage8MarkClean.addEventListener("click", markStage8FrameClean);
  elements.stage8Surface.addEventListener("change", stage8SelectChange);
  elements.stage8DefectClass.addEventListener("change", stage8SelectChange);
  elements.stage8ConfirmBox.addEventListener("click", confirmStage8SelectedBox);
  elements.stage8DeleteBox.addEventListener("click", deleteStage8SelectedBox);
  elements.stage8InlineClose.addEventListener("click", () => {
    if (!state.stage8Review) return;
    state.stage8Review.selectedId = null;
    renderStage8Editor();
  });
  elements.stage8IssueTabs.addEventListener("click", (event) => {
    const button = event.target.closest?.("button[data-stage8-select-box]");
    if (!button) return;
    selectStage8Box(button.dataset.stage8SelectBox);
  });
  elements.stage8Undo.addEventListener("click", undoStage8Change);
  elements.stage8Reset.addEventListener("click", resetStage8Review);
  elements.stage8Cancel.addEventListener("click", () => endStage8Review());
  elements.stage8Save.addEventListener("click", saveStage8Review);
  elements.stage8ReviewNote.addEventListener("change", () => {
    if (!state.stage8Review) return;
    state.stage8Review.dirty = true;
    state.stage8Review.changeLog.push({ action: "edit_note", at: new Date().toISOString() });
  });
  elements.stage8MapEditorOverlay.addEventListener("pointerdown", stage8PointerDown);
  document.addEventListener("pointermove", stage8PointerMove);
  document.addEventListener("pointerup", stage8PointerUp);
  elements.stage8RefreshQueue.addEventListener("click", () => loadStage8Queue().catch((error) => {
    elements.stage8QueueRecords.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }));
  elements.mapRoad.addEventListener("click", () => {
    if (!state.mapReady) return;
    returnToMap(google.maps.MapTypeId.ROADMAP);
  });
  elements.mapSatellite.addEventListener("click", () => {
    if (!state.mapReady) return;
    returnToMap(google.maps.MapTypeId.SATELLITE);
  });
  elements.mapStreet.addEventListener("click", () => {
    if (!state.mapReady) return;
    if (state.streetProvider === "google" && state.streetViewPanorama.getPano()) {
      showStreetDisplay("google");
      elements.kartaStreetStage.classList.add("hidden");
      elements.streetViewerNavigation.classList.add("hidden");
      state.streetViewPanorama.setVisible(true);
      elements.mapInstruction.textContent = "Google Street View is open. Move or rotate to clear old boxes, then analyse the newly visible image.";
      return;
    }
    if (state.streetProvider === "kartaview" && state.streetSequence[state.streetIndex]) {
      renderStreetPhoto(state.streetSequence[state.streetIndex]);
      elements.mapInstruction.textContent = "KartaView fallback imagery is displayed in the map area.";
      return;
    }
    beginStreetSearch();
  });
  elements.backToMap.addEventListener("click", () => returnToMap());
  elements.satelliteOpacity.addEventListener("input", () => { elements.opacityValue.textContent = `${elements.satelliteOpacity.value}%`; if (state.satelliteOverlay) state.satelliteOverlay.setOpacity(Number(elements.satelliteOpacity.value) / 100); });
  elements.satelliteLayerToggle.addEventListener("change", () => setLayerVisibility(["sentinel-preview"], elements.satelliteLayerToggle.checked));
  elements.groundLayerToggle.addEventListener("change", () => renderMarkers());
  elements.streetLayerToggle.addEventListener("change", () => {
    const targetMap = elements.streetLayerToggle.checked ? state.map : null;
    state.streetCoverageLayer?.setMap(targetMap);
    if (state.streetMarker) state.streetMarker.setMap(targetMap);
  });
  elements.uploadForm.addEventListener("submit", uploadImages);
  elements.imageInput.addEventListener("change", () => { const count = elements.imageInput.files.length; elements.selectionSummary.textContent = count ? `${count} image${count === 1 ? "" : "s"} selected` : "No files selected"; });
  elements.saveManualLocation.addEventListener("click", saveManualLocation);
  elements.stage6AuthoriseImage.addEventListener("click", authoriseSelectedImage);
  elements.runStage6Inference.addEventListener("click", runStage6Inference);
  elements.runStage6SelfTest.addEventListener("click", runStage6SelfTest);
  elements.refreshStage6History.addEventListener("click", () => loadStage6Activity().catch((error) => setStatus(elements.stage6HistorySummary, error.message, "error")));
  elements.placeImage.addEventListener("click", () => { if (!state.mapReady) { showGoogleConnectDialog(); return; } if (state.streetDisplayMode) returnToMap(); state.placementId = state.selectedId; elements.mapInstruction.textContent = "Click the real location of the selected ground image."; state.map.setOptions({ draggableCursor: "crosshair" }); document.querySelector("#map").scrollIntoView({ behavior: "smooth", block: "center" }); });
  elements.deleteImage.addEventListener("click", deleteSelectedImage); elements.previousImage.addEventListener("click", () => navigateViewer(-1)); elements.nextImage.addEventListener("click", () => navigateViewer(1));
  elements.saveProject.addEventListener("click", saveProject); elements.reopenProject.addEventListener("click", reopenProject); elements.clearProject.addEventListener("click", clearProject);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (state.stage8Review) {
      endStage8Review();
      return;
    }
    if (state.streetSearchMode) {
      finishStreetSearchMode();
      elements.mapInstruction.textContent = "Open street-imagery selection cancelled.";
      return;
    }
    if (state.streetDisplayMode) returnToMap();
  });
}

function installStage7AuditBridge() {
  const bridge = document.createElement("div");
  bridge.id = "stage7-audit-bridge";
  bridge.setAttribute("aria-hidden", "true");
  bridge.style.cssText = "position:fixed;left:0;top:0;width:12px;height:2px;overflow:hidden;opacity:0.001;z-index:2147483647";
  bridge.innerHTML = [
    '<input id="stage7-audit-sample-id" type="text" />',
    '<input id="stage7-audit-latitude" type="text" />',
    '<input id="stage7-audit-longitude" type="text" />',
    '<textarea id="stage7-audit-frame"></textarea>',
    '<button id="stage7-audit-run" type="button">Run audit point</button>',
    '<button id="stage7-audit-run-frame" type="button">Run exact audit frame</button>',
    '<output id="stage7-audit-result" data-status="idle"></output>',
  ].join("");
  document.body.appendChild(bridge);
  [...bridge.children].forEach((child, index) => {
    child.style.cssText = `position:absolute;left:${index * 2}px;top:0;width:2px;height:2px;min-width:0;min-height:0;padding:0;border:0`;
  });
  const sampleInput = bridge.querySelector("#stage7-audit-sample-id");
  const latitudeInput = bridge.querySelector("#stage7-audit-latitude");
  const longitudeInput = bridge.querySelector("#stage7-audit-longitude");
  const frameInput = bridge.querySelector("#stage7-audit-frame");
  const result = bridge.querySelector("#stage7-audit-result");
  bridge.querySelector("#stage7-audit-run").addEventListener("click", async () => {
    if (result.dataset.status === "running") return;
    const latitude = Number(latitudeInput.value);
    const longitude = Number(longitudeInput.value);
    const sampleId = String(sampleInput.value || "unlabelled");
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      result.textContent = JSON.stringify({ sample_id: sampleId, error: "invalid_coordinate" });
      result.dataset.status = "error";
      return;
    }
    result.dataset.status = "running";
    result.textContent = "";
    state.stage7Analysis = null;
    try {
      await searchStreetImagery(latitude, longitude);
      const payload = state.stage7Analysis || {};
      const imagery = payload.imagery ? { ...payload.imagery } : null;
      if (imagery) delete imagery.preview_data_url;
      result.textContent = JSON.stringify({
        sample_id: sampleId,
        requested_point: { latitude, longitude },
        analysis_id: payload.analysis_id || null,
        state: payload.state || "analysis_not_returned",
        processed: payload.processed === true,
        message: payload.message || null,
        reason_codes: Array.isArray(payload.reason_codes) ? payload.reason_codes : [],
        selected_point: payload.selected_point || null,
        frame: payload.frame || null,
        imagery,
        model: payload.model || null,
        detections: Array.isArray(payload.detections) ? payload.detections : [],
        inference_run: payload.inference_run || null,
        diagnostics: payload.diagnostics || {},
      });
      result.dataset.status = "complete";
    } catch (error) {
      result.textContent = JSON.stringify({
        sample_id: sampleId,
        requested_point: { latitude, longitude },
        error: String(error?.message || error || "unknown_error"),
      });
      result.dataset.status = "error";
    }
  });
  bridge.querySelector("#stage7-audit-run-frame").addEventListener("click", async () => {
    if (result.dataset.status === "running") return;
    const sampleId = String(sampleInput.value || "unlabelled-frame");
    let frame;
    try {
      frame = JSON.parse(frameInput.value || "null");
    } catch (_error) {
      frame = null;
    }
    if (!state.googleApiKey || !frame || typeof frame !== "object" || !frame.pano_id) {
      result.textContent = JSON.stringify({ sample_id: sampleId, error: "invalid_exact_frame_or_google_connection" });
      result.dataset.status = "error";
      return;
    }
    result.dataset.status = "running";
    result.textContent = "";
    try {
      const payload = await request(stage7AnalyseEndpoint(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: state.googleApiKey,
          inference_mode: state.stage7InferenceMode,
          ...(state.stage7InferenceMode === "enhanced" ? { stage7_api_key: state.stage7VisionApiKey } : {}),
          ...frame,
        }),
      });
      const imagery = payload.imagery ? { ...payload.imagery } : null;
      if (imagery) delete imagery.preview_data_url;
      result.textContent = JSON.stringify({
        sample_id: sampleId,
        requested_point: { latitude: frame.latitude, longitude: frame.longitude },
        analysis_id: payload.analysis_id || null,
        state: payload.state || "analysis_not_returned",
        processed: payload.processed === true,
        message: payload.message || null,
        reason_codes: Array.isArray(payload.reason_codes) ? payload.reason_codes : [],
        selected_point: payload.selected_point || null,
        frame: payload.frame || frame,
        imagery,
        model: payload.model || null,
        detections: Array.isArray(payload.detections) ? payload.detections : [],
        inference_run: payload.inference_run || null,
        diagnostics: payload.diagnostics || {},
      });
      result.dataset.status = "complete";
    } catch (error) {
      result.textContent = JSON.stringify({
        sample_id: sampleId,
        frame: { pano_id: frame.pano_id, frame_id: frame.frame_id || null },
        error: String(error?.message || error || "unknown_error"),
      });
      result.dataset.status = "error";
    }
  });
}

async function initialise() {
  initialiseDates(); bindEvents(); installStage7AuditBridge(); updateCollectionControls(); updateProgressiveLayout(); setMapControlsEnabled(false);
  try {
    const config = await request("api/config");
    state.defaultMap = {
      longitude: Number(config.map.longitude),
      latitude: Number(config.map.latitude),
      zoom: Number(config.map.zoom),
    };
    renderStage5Header(config.validation_v2 || {}, config.stage6 || {}, config.stage7 || {});
    if (config.stage6) renderStage6Runtime(config.stage6);
    if (config.stage7) state.stage7Status = config.stage7;
    if (config.stage8) state.stage8Status = config.stage8;
    const startup = await Promise.allSettled([loadImages(), loadStage6Status(), loadStage7Status(), loadStage6Activity(), loadStage8Status(), loadStage8Queue()]);
    const failed = startup.filter((item) => item.status === "rejected");
    if (failed.length) setStatus(elements.stage6StatusMessage, `Uploaded-image analysis started with ${failed.length} unavailable supporting request${failed.length === 1 ? "" : "s"}. Retry from the workspace controls.`, "warning");
  } catch (error) {
    setStatus(elements.satelliteStatus, `Workspace could not start: ${error.message}`, "error");
  }
}

window.addEventListener("pagehide", () => {
  state.stage7VisionApiKey = null;
  state.stage7InferenceMode = "local";
});

initialise();
