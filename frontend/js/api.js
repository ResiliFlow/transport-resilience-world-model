async function request(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = response.status === 204
    ? null
    : contentType.includes("application/json")
      ? await response.json()
      : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null ? payload.detail : null;
    const message = Array.isArray(detail)
      ? detail.map((item) => typeof item === "string" ? item : item?.msg || item?.message || JSON.stringify(item)).join(" ")
      : detail && typeof detail === "object"
        ? detail.message || detail.msg || JSON.stringify(detail)
        : typeof payload === "object" && payload !== null
          ? payload.message
          : payload;
    throw new Error(message || `Request failed with status ${response.status}.`);
  }
  return payload;
}

function jsonOptions(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  health: () => request("/api/health"),
  bootstrap: () => request("/api/bootstrap"),
  listProjects: () => request("/api/projects"),
  getProject: (projectId) => request(`/api/projects/${encodeURIComponent(projectId)}`),
  keepProject: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}/keep`,
    { method: "POST" },
  ),
  deleteProject: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}`,
    { method: "DELETE" },
  ),
  projectMemory: (projectId) => request(`/api/projects/${encodeURIComponent(projectId)}/memory`),
  map: (projectId, { metric, timestamp, bbox, limit = 20000 }) => {
    const query = new URLSearchParams({ metric, limit: String(limit) });
    if (timestamp !== null && timestamp !== undefined) query.set("timestamp", String(timestamp));
    if (bbox) query.set("bbox", bbox.join(","));
    return request(`/api/projects/${encodeURIComponent(projectId)}/map?${query}`);
  },
  networkMappingStatus: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}/network-mapping`
  ),
  buildNetworkMapping: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}/network-mapping`,
    { method: "POST" },
  ),
  networkMap: (projectId, { metric, timestamp, bbox, limit = 20000 }) => {
    const query = new URLSearchParams({ metric, limit: String(limit) });
    if (timestamp !== null && timestamp !== undefined) query.set("timestamp", String(timestamp));
    if (bbox) query.set("bbox", bbox.join(","));
    return request(`/api/projects/${encodeURIComponent(projectId)}/network-map?${query}`);
  },
  scenarioTargets: (projectId, bbox, limit = 5000) => {
    const query = new URLSearchParams({ bbox: bbox.join(","), limit: String(limit) });
    return request(`/api/projects/${encodeURIComponent(projectId)}/scenario-targets?${query}`);
  },
  history: (projectId, regionId, entityId, metric) => request(
    `/api/projects/${encodeURIComponent(projectId)}/entities/${encodeURIComponent(regionId)}/${encodeURIComponent(entityId)}/history?metric=${encodeURIComponent(metric)}`
  ),
  interpret: (payload) => request("/api/commands/interpret", jsonOptions("POST", payload)),
  analyse: (payload) => request("/api/analyses", jsonOptions("POST", payload)),
  startAnalysis: (payload) => request("/api/analysis-jobs", jsonOptions("POST", payload)),
  runs: (projectId = null) => request(`/api/runs${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  run: (runId) => request(`/api/runs/${encodeURIComponent(runId)}`),
  deleteRun: (runId) => request(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" }),
  scenarios: (projectId) => request(`/api/scenarios?project_id=${encodeURIComponent(projectId)}`),
  scenario: (scenarioId) => request(`/api/scenarios/${encodeURIComponent(scenarioId)}`),
  createScenario: (payload) => request("/api/scenarios", jsonOptions("POST", payload)),
  updateScenario: (scenarioId, payload) => request(
    `/api/scenarios/${encodeURIComponent(scenarioId)}`,
    jsonOptions("POST", payload),
  ),
  deleteScenario: (scenarioId) => request(
    `/api/scenarios/${encodeURIComponent(scenarioId)}`,
    { method: "DELETE" },
  ),
  matsimReadiness: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}/matsim-readiness`
  ),
  runScenarioMatsim: (scenarioId, payload = {}) => request(
    `/api/scenarios/${encodeURIComponent(scenarioId)}/matsim`,
    jsonOptions("POST", payload),
  ),
  models: () => request("/api/model-library"),
  releaseScientificValidation: () => request("/api/scientific-validation/release"),
  projectScientificValidation: (projectId, topK = 10) => request(
    `/api/projects/${encodeURIComponent(projectId)}/scientific-validation?top_k=${encodeURIComponent(topK)}`,
    { method: "POST" },
  ),
  startResearchValidation: (projectId, profile = "standard") => request(
    `/api/projects/${encodeURIComponent(projectId)}/research-validation?profile=${encodeURIComponent(profile)}`,
    { method: "POST" },
  ),
  researchValidationReports: (projectId) => request(
    `/api/projects/${encodeURIComponent(projectId)}/research-validation`
  ),
  researchValidationReport: (reportId) => request(
    `/api/research-validation/${encodeURIComponent(reportId)}`
  ),
  modelRecommendations: (projectId, goal = "") => request(
    `/api/projects/${encodeURIComponent(projectId)}/model-recommendations${goal ? `?goal=${encodeURIComponent(goal)}` : ""}`
  ),
  assistantStatus: (sessionId = "") => request(`/api/assistant/status${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`),
  configureAssistantProvider: (payload) => request("/api/assistant/provider", jsonOptions("POST", payload)),
  refreshAssistantProviderModels: (sessionId, provider) => request(
    `/api/assistant/provider/models?session_id=${encodeURIComponent(sessionId)}&provider=${encodeURIComponent(provider)}`,
    { method: "POST" },
  ),
  verifyAssistantProvider: (sessionId, provider) => request(
    `/api/assistant/provider/verify?session_id=${encodeURIComponent(sessionId)}&provider=${encodeURIComponent(provider)}`,
    { method: "POST" },
  ),
  assistantMessage: (payload) => request("/api/assistant/message", jsonOptions("POST", payload)),
  assistantChats: (limit = 50) => request(`/api/assistant/chats?limit=${encodeURIComponent(limit)}`),
  createAssistantChat: (payload = {}) => request("/api/assistant/chats", jsonOptions("POST", payload)),
  assistantChat: (chatId) => request(`/api/assistant/chats/${encodeURIComponent(chatId)}`),
  appendAssistantChatMessage: (chatId, payload) => request(
    `/api/assistant/chats/${encodeURIComponent(chatId)}/messages`,
    jsonOptions("POST", payload),
  ),
  deleteAssistantChat: (chatId) => request(
    `/api/assistant/chats/${encodeURIComponent(chatId)}`,
    { method: "DELETE" },
  ),
  uploadPreview: (formData) => request("/api/uploads/preview", { method: "POST", body: formData }),
  confirmUpload: (uploadId, payload) => request(`/api/uploads/${encodeURIComponent(uploadId)}/confirm`, jsonOptions("POST", payload)),
  job: (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}`),
  cancelJob: (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }),
};
