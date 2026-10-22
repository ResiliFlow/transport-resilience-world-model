from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


HazardType = Literal["hurricane", "flood", "wildfire", "earthquake", "storm", "multi_hazard", "other"]
MetricType = Literal[
    "speed", "travel_time", "primary", "flow", "capacity", "free_flow_time",
    "repair_cost", "repair_duration", "failure_probability", "length",
]
AnalysisType = Literal[
    "persistent_degradation",
    "critical_components",
    "area_summary",
    "closure_scenario",
    "restoration_screening",
    "temporal_resilience",
    "spatial_hotspots",
    "progressive_robustness",
    "route_detour_analysis",
    "adaptive_registered_model",
]
ScenarioEffect = Literal["closed", "speed_reduction", "travel_time_increase"]
LLMProvider = Literal["openai", "anthropic", "deepseek", "gemini"]
AssistantMode = Literal["guide", "assist", "autopilot"]
AgentToolId = Literal[
    "inspect_project_data",
    "find_critical_roads",
    "find_critical_areas",
    "plan_recovery_priorities",
    "plan_routes_under_disruption",
    "test_network_resilience",
    "build_disruption_scenario",
    "run_matsim",
]
AgentDependencyKind = Literal["run", "scenario", "task"]


class AssistantChatCreateRequest(BaseModel):
    """Create one locally persisted Copilot conversation.

    A chat may be started before a project exists.  When a project is open the
    client associates it automatically; there is deliberately no project
    picker in the conversation UI.
    """

    title: str | None = Field(default=None, max_length=160)
    project_id: str | None = Field(default=None, min_length=3, max_length=120)


class AssistantChatMessageCreateRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssistantHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    # Conversation length is managed against the selected model's context
    # budget in the assistant layer.  A fixed character ceiling here caused a
    # valid long answer to make the *next* turn fail request validation.
    content: str = Field(min_length=1)


class AssistantChatArtifact(BaseModel):
    """One verified result created inside the active Copilot conversation.

    Only stable identifiers and public result types cross the LLM boundary.
    The backend never accepts result values from the browser as evidence; the
    identifiers are resolved again against the current project before reuse.
    """

    kind: Literal["analysis_result", "scenario_result"]
    artifact_id: str = Field(min_length=3, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")
    analysis_type: AnalysisType
    project_id: str = Field(min_length=3, max_length=120)
    created_at: str | None = Field(default=None, max_length=64)


class AssistantMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    project_id: str | None = Field(default=None, min_length=3, max_length=120)
    selected_keys: list[str] = Field(default_factory=list, max_length=5000)
    bbox: list[float] | None = None
    current_metric: MetricType = "speed"
    current_time: int | None = None
    current_view: Literal["explore", "data", "models", "saved", "memory"] = "explore"
    active_scenario_name: str | None = Field(default=None, max_length=120)
    scenario_count: int = Field(default=0, ge=0, le=10000)
    scenario_building: bool = False
    history: list[AssistantHistoryItem] = Field(default_factory=list)
    chat_artifacts: list[AssistantChatArtifact] = Field(default_factory=list, max_length=100)
    prefer_llm: bool = True
    session_id: str | None = Field(default=None, min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    provider: LLMProvider = "openai"
    interaction_mode: AssistantMode = "assist"

    @field_validator("bbox")
    @classmethod
    def validate_assistant_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("Bounding box must contain west, south, east and north.")
        return value


class AssistantProviderConfigRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    provider: LLMProvider
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, min_length=2, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    clear: bool = False

    @model_validator(mode="after")
    def validate_key_or_clear(self) -> "AssistantProviderConfigRequest":
        # An empty key is valid when the provider is already connected in the
        # current session: the backend reuses that in-memory credential while
        # the user changes only the selected model. A first connection still
        # fails safely in the provider layer if no key exists.
        return self


class UploadConfirmRequest(BaseModel):
    project_name: str = Field(min_length=2, max_length=120)
    hazard_type: HazardType = "other"
    mapping: dict[str, str | None]
    file_mappings: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    file_encodings: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    timezone: str = "unspecified"
    time_step_seconds: float = Field(default=300.0, gt=0, le=604800)
    topology_node_scope: Literal["global", "per_region"] = "per_region"
    entity_id_scope: Literal["global", "per_region"] = "per_region"
    primary_metric_direction: Literal["higher_worse", "lower_worse", "descriptive"] = "descriptive"
    primary_metric_comparison: Literal["current_value", "change_from_reference"] = "current_value"
    primary_metric_scale: float = Field(default=1.0, gt=0)

    @field_validator("mapping")
    @classmethod
    def validate_mapping(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        # Flat observations are still checked by the importer, but transport
        # packages may obtain link coordinates by joining a node table, from
        # WKT/GeoJSON, endpoint coordinates or a disclosed topology layout.
        # Requiring latitude/longitude in the API schema prevented every one
        # of those legitimate structures from reaching the adaptive parser.
        if len(value) > 80:
            raise ValueError("The field mapping contains too many roles.")
        return value

    @field_validator("file_mappings")
    @classmethod
    def validate_file_mappings(
        cls, value: dict[str, dict[str, str | None]]
    ) -> dict[str, dict[str, str | None]]:
        """Keep per-file manual corrections explicit and backwards compatible.

        ``mapping`` remains the single-file/default contract.  When several
        files are uploaded, the UI also sends one independently confirmed
        mapping per saved filename so a low-confidence second file is never
        forced to inherit the first file's columns.
        """
        for filename, mapping in value.items():
            if not filename or len(filename) > 260:
                raise ValueError("Each file mapping needs a readable filename.")
            if len(mapping) > 80:
                raise ValueError(f"{filename}: the field mapping contains too many roles.")
        return value

    @field_validator("file_encodings")
    @classmethod
    def validate_file_encodings(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {"utf-8-sig", "utf-8", "utf-16", "gb18030", "big5", "cp1252", "latin-1"}
        for filename, encoding in value.items():
            if not filename or encoding not in allowed:
                raise ValueError(f"{filename or 'File'}: choose a supported text encoding.")
        return value


class CommandRequest(BaseModel):
    project_id: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=3, max_length=1000)
    selected_keys: list[str] = Field(default_factory=list, max_length=5000)
    bbox: list[float] | None = None
    current_metric: MetricType = "speed"
    current_time: int | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("Bounding box must contain west, south, east and north.")
        return value


class AnalysisRequest(BaseModel):
    project_id: str = Field(min_length=3, max_length=120)
    analysis_type: AnalysisType
    model_id: str | None = Field(default=None, pattern=r"^P5-M(?:[0-9]|1[0-9]|2[0-3])$")
    metric: MetricType = "speed"
    threshold: float = Field(default=0.2, ge=0.01, le=2.0)
    top_k: int = Field(default=10, ge=1)
    # A ranked critical-area result can legitimately contain more than 5,000
    # member roads even when the user requested only a few dozen areas.  The
    # analysis engine validates every key against the current project and
    # switches large candidate sets to its disclosed structural tier, so an
    # API-level list cap must not reject a valid saved-result dependency before
    # the model starts.
    selected_keys: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None
    time_start: int | None = None
    time_end: int | None = None
    command_text: str = Field(default="", max_length=1000)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registered_model(self) -> "AnalysisRequest":
        if self.analysis_type == "adaptive_registered_model" and not self.model_id:
            raise ValueError("An adaptive registered-model analysis needs a model ID.")
        return self

    @field_validator("bbox")
    @classmethod
    def validate_analysis_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("Bounding box must contain west, south, east and north.")
        return value


class ScenarioChange(BaseModel):
    key: str = Field(min_length=3, max_length=260)
    effect: ScenarioEffect
    magnitude: float | None = None

    @model_validator(mode="after")
    def validate_effect_magnitude(self) -> "ScenarioChange":
        if "::" not in self.key:
            raise ValueError("Each change needs a mapped entity or road-link key.")
        if self.effect == "closed":
            self.magnitude = 1.0
        elif self.effect == "speed_reduction":
            if self.magnitude is None or not 0.01 <= self.magnitude <= 0.95:
                raise ValueError("Speed reduction must be between 1% and 95%.")
        elif self.magnitude is None or not 0.01 <= self.magnitude <= 5.0:
            raise ValueError("Travel-time increase must be between 1% and 500%.")
        return self


class ScenarioUpsertRequest(BaseModel):
    project_id: str = Field(min_length=3, max_length=120)
    name: str = Field(min_length=2, max_length=120)
    hazard_type: HazardType = "other"
    base_timestamp: int | None = None
    changes: list[ScenarioChange] = Field(min_length=1, max_length=5000)

    @field_validator("changes")
    @classmethod
    def validate_unique_changes(cls, value: list[ScenarioChange]) -> list[ScenarioChange]:
        keys = [change.key for change in value]
        if len(keys) != len(set(keys)):
            raise ValueError("Keep one final change for each mapped entity or road link.")
        return value


class MatsimRunRequest(BaseModel):
    iterations: int = Field(default=3, ge=1, le=20)
    max_agents: int = Field(default=4000, ge=20, le=20_000)
    seed: int = Field(default=41729, ge=1, le=2_147_483_647)
    timeout_seconds: int = Field(default=1800, ge=120, le=7200)


class AgentDependencyReference(BaseModel):
    """One explicit, project-owned upstream artefact used by an Agent tool."""

    kind: AgentDependencyKind
    id: str = Field(min_length=3, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")
    role: str = Field(default="input", min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")


class AgentToolTaskRequest(BaseModel):
    """Stable machine contract used by LLM orchestration, not the manual UI."""

    project_id: str = Field(min_length=3, max_length=120)
    tool_id: AgentToolId
    input: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[AgentDependencyReference] = Field(default_factory=list, max_length=100)
    client_request_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


class AgentAnalysisToolInput(BaseModel):
    """Common input envelope for the five analysis-based core tools."""

    model_id: str | None = Field(default=None, pattern=r"^P5-M(?:[0-9]|1[0-9]|2[0-3])$")
    metric: MetricType = "speed"
    threshold: float = Field(default=0.2, ge=0.01, le=2.0)
    top_k: int = Field(default=10, ge=1)
    selected_keys: list[str] = Field(default_factory=list)
    bbox: list[float] | None = None
    time_start: int | None = None
    time_end: int | None = None
    command_text: str = Field(default="", max_length=4000)
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def validate_agent_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("Bounding box must contain west, south, east and north.")
        return value


class AgentDataInspectionInput(BaseModel):
    include_memory: bool = True
    include_model_readiness: bool = True


class AgentScenarioToolInput(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    hazard_type: HazardType = "other"
    base_timestamp: int | None = None
    changes: list[ScenarioChange] = Field(min_length=1, max_length=5000)

    @field_validator("changes")
    @classmethod
    def validate_agent_unique_changes(cls, value: list[ScenarioChange]) -> list[ScenarioChange]:
        keys = [change.key for change in value]
        if len(keys) != len(set(keys)):
            raise ValueError("Keep one final change for each mapped entity or road link.")
        return value


class AgentMatsimToolInput(MatsimRunRequest):
    scenario_id: str = Field(min_length=3, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")
