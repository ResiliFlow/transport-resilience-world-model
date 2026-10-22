from __future__ import annotations

import csv
from hashlib import sha256
from importlib import metadata
import io
import json
import os
import platform
from pathlib import Path
import re
import secrets
import signal
import shutil
import threading
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine.command_parser import interpret_command
from engine.assistant import AssistantRequestContext, assistant_response, assistant_status
from engine.capabilities import enrich_project_summary
from engine.importers import file_sha256, import_csv_project, preview_csv, read_project_summary
from engine.models import entity_history, map_features, run_analysis
from engine.network_mapping import build_network_mapping, network_features, network_mapping_status
from engine.transport_data import combine_transport_profiles
from engine.scenarios import evaluate_disruption_scenario, select_scenario_targets
from engine.model_router import enrich_registry, recommend_models
from engine.llm_providers import (
    ProviderRequestError,
    configure_provider,
    refresh_provider_models,
    verify_provider_inference,
)
from engine.scientific_validation import validate_project
from engine.matsim_adapter import assess_matsim_readiness, run_matsim_scenario
from engine.research_validation import (
    ResearchValidationCancelled,
    list_research_reports,
    load_research_report,
    report_csv,
    report_latex,
    run_research_validation,
)

from .agent_tools import (
    AGENT_TOOL_PROTOCOL_VERSION,
    agent_tool_catalog,
    build_analysis_payload,
    build_matsim_request,
    build_scenario_request,
    structured_agent_error,
    validate_agent_tool_input,
)
from .api.schemas import (
    AgentToolTaskRequest,
    AnalysisRequest,
    AssistantChatCreateRequest,
    AssistantChatMessageCreateRequest,
    AssistantMessageRequest,
    AssistantProviderConfigRequest,
    CommandRequest,
    MatsimRunRequest,
    ScenarioUpsertRequest,
    UploadConfirmRequest,
)
from .core.config import (
    DATABASE_PATH,
    DEMO_CSV_PATH,
    FRONTEND_ROOT,
    MODEL_LIBRARY_ROOT,
    PROJECT_DATA_ROOT,
    STORAGE_ROOT,
    UPLOAD_ROOT,
    version,
)
from .core.storage import Repository
from infrastructure_perception.backend.app import app as infrastructure_perception_app


app = FastAPI(
    title="Transport World Model Platform for Disaster Resilience, Response and Recovery API",
    version=version(),
    description="Static disaster-transport network reconstruction, deterministic criticality, scenario simulation and evidence-bounded recovery analysis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8005", "http://localhost:8005"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    """Prevent browser history/cache snapshots from restoring obsolete local UI files."""
    response = await call_next(request)
    path = request.url.path.lower()
    content_type = str(response.headers.get("content-type") or "").lower()
    if (
        "text/html" in content_type
        or "text/css" in content_type
        or "javascript" in content_type
        or path.endswith((".html", ".css", ".js"))
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Clear-Site-Data"] = '"cache"'
    return response

repository = Repository(DATABASE_PATH)
repository.initialise()
SHUTDOWN_TOKEN_PATH = STORAGE_ROOT / "backend.shutdown.token"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _delete_temporary_upload(upload: dict[str, Any]) -> bool:
    """Remove only an upload explicitly created as temporary by this release."""
    if not upload.get("is_temporary"):
        return False
    upload_id = str(upload.get("upload_id") or "")
    upload_folder = (UPLOAD_ROOT / upload_id).resolve()
    if upload_folder.parent != UPLOAD_ROOT.resolve():
        return False
    for value in upload.get("paths", []):
        path = Path(value).resolve()
        if _is_within(path, upload_folder) and path.is_file():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    if upload_folder.exists():
        shutil.rmtree(upload_folder, ignore_errors=True)
    repository.delete_upload(upload_id)
    return True


def _delete_temporary_project(project: dict[str, Any]) -> bool:
    """Delete one temporary project without permitting deletion outside managed storage."""
    if project.get("retention") != "temporary":
        return False
    project_id = str(project.get("project_id") or "")
    database_path = Path(str(project.get("database_path") or "")).resolve()
    if database_path.parent != PROJECT_DATA_ROOT.resolve():
        return False
    deleted, scenario_ids = repository.delete_project(project_id)
    if not deleted:
        return False
    for path in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    ):
        if path.parent == PROJECT_DATA_ROOT.resolve():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    matsim_root = (STORAGE_ROOT / "matsim").resolve()
    for scenario_id in scenario_ids:
        scenario_folder = (matsim_root / scenario_id).resolve()
        if scenario_folder.parent == matsim_root and scenario_folder.exists():
            shutil.rmtree(scenario_folder, ignore_errors=True)
    return True


def _delete_saved_project(project: dict[str, Any]) -> bool:
    """Permanently remove one user-saved project from managed platform storage."""
    if project.get("retention") != "persistent":
        return False
    project_id = str(project.get("project_id") or "")
    database_path = Path(str(project.get("database_path") or "")).resolve()
    if database_path.parent != PROJECT_DATA_ROOT.resolve():
        return False
    deleted, scenario_ids = repository.delete_persistent_project(project_id)
    if not deleted:
        return False
    for path in (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
    ):
        if path.parent == PROJECT_DATA_ROOT.resolve():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    matsim_root = (STORAGE_ROOT / "matsim").resolve()
    for scenario_id in scenario_ids:
        scenario_folder = (matsim_root / scenario_id).resolve()
        if scenario_folder.parent == matsim_root and scenario_folder.exists():
            shutil.rmtree(scenario_folder, ignore_errors=True)
    return True


def _cleanup_temporary_storage() -> dict[str, int]:
    projects = repository.list_temporary_projects()
    uploads = repository.list_temporary_uploads()
    project_count = 0
    upload_count = 0
    for project in projects:
        try:
            project_count += int(_delete_temporary_project(project))
        except (OSError, ValueError):
            continue
    for upload in uploads:
        try:
            upload_count += int(_delete_temporary_upload(upload))
        except (OSError, ValueError):
            continue
    return {
        "projects": project_count,
        "uploads": upload_count,
    }


def _analysis_source_digest() -> str:
    digest = sha256()
    project_root = Path(__file__).resolve().parents[2]
    for relative in (
        "VERSION",
        "engine/importers.py",
        "engine/models.py",
        "engine/network_mapping.py",
        "engine/scenarios.py",
        "engine/routing.py",
        "engine/traffic_assignment.py",
        "engine/model_router.py",
        "engine/adaptive_models.py",
        "engine/capabilities.py",
        "engine/llm_providers.py",
        "engine/model_review.py",
        "engine/command_parser.py",
        "engine/assistant.py",
        "engine/scientific_validation.py",
        "engine/matsim_adapter.py",
        "matsim_worker/pom.xml",
        "matsim_worker/src/main/java/au/edu/unsw/rciti/twm/matsim/TwmMatsimRunner.java",
    ):
        candidate = project_root / relative
        if candidate.exists():
            digest.update(relative.encode("utf-8"))
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


@app.exception_handler(ProviderRequestError)
async def provider_request_error_handler(_, exc: ProviderRequestError) -> JSONResponse:
    if exc.http_status in {400, 401, 403, 404, 409, 429}:
        status_code = exc.http_status
    elif exc.code in {"network_or_timeout", "upstream_unavailable"} or (
        exc.http_status is not None and exc.http_status >= 500
    ):
        status_code = 503
    else:
        status_code = 502
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": str(exc),
            "type": "provider_request_error",
            "code": exc.code,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc), "type": "model_validation_error"})


@app.on_event("startup")
def record_backend_process() -> None:
    # A process restart must preserve the user's active workspace. Cleanup is
    # opt-in for an explicit maintenance launch, never a restart side effect.
    if os.getenv("TWM_CLEAN_TEMP_ON_START") == "1":
        _cleanup_temporary_storage()
    (STORAGE_ROOT / "backend.pid").write_text(str(os.getpid()), encoding="ascii")
    SHUTDOWN_TOKEN_PATH.write_text(secrets.token_urlsafe(32), encoding="ascii")


@app.on_event("shutdown")
def clear_backend_process() -> None:
    pid_path = STORAGE_ROOT / "backend.pid"
    if pid_path.exists() and pid_path.read_text(encoding="ascii").strip() == str(os.getpid()):
        pid_path.unlink()
    SHUTDOWN_TOKEN_PATH.unlink(missing_ok=True)


def _enrich_project_record(project: dict[str, Any]) -> dict[str, Any]:
    record = {**project}
    record["summary"] = enrich_project_summary(record.get("summary"))
    return record


def _project_or_404(project_id: str) -> dict[str, Any]:
    project = repository.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    if not Path(project["database_path"]).exists():
        raise HTTPException(status_code=409, detail="Project data store is unavailable.")
    return _enrich_project_record(project)


def _registry() -> dict[str, Any]:
    path = MODEL_LIBRARY_ROOT / "registry.json"
    if not path.exists():
        return {"version": version(), "models": []}
    registry = json.loads(path.read_text(encoding="utf-8"))
    for model in registry.get("models", []):
        card_path = MODEL_LIBRARY_ROOT / "cards" / model.get("card_file", "")
        if card_path.exists():
            model["card"] = json.loads(card_path.read_text(encoding="utf-8"))
    return enrich_registry(registry)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "workspace": "prototype5-platform",
        "version": version(),
        "engine": "deterministic_ready",
        "storage": "ready",
        "random_simulation": "disabled",
    }


@app.post("/api/shutdown", status_code=202)
def shutdown_local_backend(
    request: Request,
    stop_token: str | None = Header(default=None, alias="X-TWM-Stop-Token"),
) -> dict[str, Any]:
    """Stop only this local Transport World Model Platform process after token verification."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Shutdown is restricted to the local machine.")
    if not SHUTDOWN_TOKEN_PATH.exists():
        raise HTTPException(status_code=409, detail="The local shutdown token is unavailable.")
    expected = SHUTDOWN_TOKEN_PATH.read_text(encoding="ascii").strip()
    if not stop_token or not secrets.compare_digest(stop_token, expected):
        raise HTTPException(status_code=403, detail="The local shutdown token is invalid.")

    def terminate_after_response() -> None:
        time.sleep(0.25)
        (STORAGE_ROOT / "backend.pid").unlink(missing_ok=True)
        SHUTDOWN_TOKEN_PATH.unlink(missing_ok=True)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2.0)
        finally:
            os._exit(0)

    threading.Thread(target=terminate_after_response, daemon=True, name="twm-local-shutdown").start()
    return {
        "status": "stopping",
        "temporary_cleanup": {"projects": 0, "uploads": 0},
        "workspace_preserved": True,
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    registry = _registry()
    return {
        "version": version(),
        "projects": [
            _enrich_project_record(project) for project in repository.list_persistent_projects()
        ],
        "models": registry.get("models", []),
        "model_registry_version": registry.get("version", version()),
        "hazard_types": [
            {"id": "hurricane", "label": "Hurricane / cyclone"},
            {"id": "flood", "label": "Flood"},
            {"id": "wildfire", "label": "Wildfire"},
            {"id": "earthquake", "label": "Earthquake"},
            {"id": "storm", "label": "Severe storm"},
            {"id": "multi_hazard", "label": "Multi-hazard"},
            {"id": "other", "label": "Other disruption"},
        ],
        "interface_boundary": "Static uploaded data; no live feed, dispatch or random disaster generation.",
    }


def _safe_filename(name: str | None, fallback: str) -> str:
    base = Path(name or fallback).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or fallback


@app.post("/api/uploads/preview", status_code=201)
async def upload_preview(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one CSV file.")
    upload_id = f"upload_{uuid4().hex[:14]}"
    upload_folder = UPLOAD_ROOT / upload_id
    upload_folder.mkdir(parents=True, exist_ok=False)
    saved_paths: list[str] = []
    previews: list[dict[str, Any]] = []
    used_filenames: set[str] = set()
    for index, uploaded in enumerate(files):
        filename = _safe_filename(uploaded.filename, f"upload_{index + 1}.csv")
        if Path(filename).suffix.lower() not in {".csv", ".txt", ".tsv"}:
            raise HTTPException(status_code=415, detail=f"{filename}: v0.1 accepts CSV, TSV or delimited text files.")
        if filename.casefold() in used_filenames:
            original = Path(filename)
            suffix_number = 2
            while True:
                candidate = f"{original.stem}_{suffix_number}{original.suffix}"
                if candidate.casefold() not in used_filenames:
                    filename = candidate
                    break
                suffix_number += 1
        used_filenames.add(filename.casefold())
        destination = upload_folder / filename
        size = 0
        with destination.open("wb") as handle:
            while chunk := await uploaded.read(8 * 1024 * 1024):
                handle.write(chunk)
                size += len(chunk)
        await uploaded.close()
        if size == 0:
            raise HTTPException(status_code=422, detail=f"{filename} is empty.")
        preview = preview_csv(destination)
        preview["sha256"] = file_sha256(destination)
        previews.append(preview)
        saved_paths.append(str(destination.resolve()))
    combined = {
        "files": previews,
        "shared_columns": sorted(set.intersection(*(set(item["columns"]) for item in previews))) if previews else [],
        "suggested_mapping": previews[0]["suggested_mapping"],
        "total_size_bytes": sum(item["size_bytes"] for item in previews),
        "geometry_warning": "Latitude/longitude pairs will be displayed as representative points unless exact line geometry is supplied separately.",
    }
    combined["transport_package"] = combine_transport_profiles(previews)
    if combined["transport_package"].get("preferred_view") == "links":
        combined["geometry_warning"] = (
            "Road-link data detected. Uploaded geometry, endpoint coordinates, joined nodes or a disclosed topology layout "
            "will be used automatically; no detector-to-link conversion is required."
        )
    repository.save_upload(saved_paths, combined, upload_id=upload_id)
    return {"upload_id": upload_id, **combined}


def _run_import_job(
    *,
    job_id: str,
    project_id: str,
    paths: list[Path],
    request: UploadConfirmRequest,
    source: str,
    upload_id: str | None = None,
) -> None:
    database_path = PROJECT_DATA_ROOT / f"{project_id}.sqlite"

    def report(progress: float, message: str) -> None:
        repository.update_job(job_id, status="running", progress=progress, message=message)

    try:
        summary = import_csv_project(
            files=paths,
            mapping=request.mapping,
            file_mappings=request.file_mappings,
            file_encodings=request.file_encodings,
            units=request.units,
            timezone=request.timezone,
            time_step_seconds=request.time_step_seconds,
            topology_node_scope=request.topology_node_scope,
            entity_id_scope=request.entity_id_scope,
            primary_metric_direction=request.primary_metric_direction,
            primary_metric_comparison=request.primary_metric_comparison,
            primary_metric_scale=request.primary_metric_scale,
            database_path=database_path,
            project_id=project_id,
            project_name=request.project_name,
            hazard_type=request.hazard_type,
            source=source,
            progress=report,
        )
        repository.save_project(
            project_id=project_id,
            name=request.project_name,
            hazard_type=request.hazard_type,
            source=source,
            status="ready",
            database_path=str(database_path.resolve()),
            mapping=request.mapping,
            summary=summary,
            retention="temporary" if source == "user_upload" else "persistent",
        )
        repository.update_job(
            job_id,
            status="completed",
            progress=100.0,
            message="Project is ready",
            result={"project_id": project_id, "summary": summary},
        )
        if upload_id:
            try:
                upload = repository.get_upload(upload_id)
                if upload:
                    _delete_temporary_upload(upload)
            except (OSError, ValueError):
                # The imported project is already self-contained. A failed
                # source-copy cleanup must never turn a valid import into a failure.
                pass
    except Exception as exc:  # background jobs must preserve a readable failure record
        database_path.unlink(missing_ok=True)
        Path(f"{database_path}-wal").unlink(missing_ok=True)
        Path(f"{database_path}-shm").unlink(missing_ok=True)
        repository.update_job(job_id, status="failed", message="Import failed", error=str(exc))


@app.post("/api/uploads/{upload_id}/confirm", status_code=202)
def confirm_upload(upload_id: str, request: UploadConfirmRequest) -> dict[str, Any]:
    upload = repository.get_upload(upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload preview not found.")
    preview_files = upload.get("preview", {}).get("files", [])
    adaptive_ready = bool(upload.get("preview", {}).get("transport_package", {}).get("auto_importable"))
    if len(preview_files) > 1 and not adaptive_ready:
        missing_file_mappings = [
            file["name"] for file in preview_files
            if file["name"] not in request.file_mappings
        ]
        if missing_file_mappings:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Confirm fields independently for every uploaded file: "
                    + ", ".join(missing_file_mappings)
                ),
            )
    for file in preview_files:
        filename = file["name"]
        file_mapping = request.file_mappings.get(filename, request.mapping)
        columns = set(file.get("columns", []))
        invalid_columns = sorted(
            {
                str(column)
                for column in file_mapping.values()
                if column and column not in columns
            }
        )
        if invalid_columns:
            raise HTTPException(
                status_code=422,
                detail=f"{filename}: selected fields are not present: {', '.join(invalid_columns)}",
            )
    slug = re.sub(r"[^a-z0-9]+", "_", request.project_name.lower()).strip("_")[:42] or "project"
    project_id = f"p5_{slug}_{uuid4().hex[:8]}"
    job_id = repository.create_job("csv_import", "Import queued")
    thread = threading.Thread(
        target=_run_import_job,
        kwargs={
            "job_id": job_id,
            "project_id": project_id,
            "paths": [Path(path) for path in upload["paths"]],
            "request": request,
            "source": "user_upload",
            "upload_id": upload_id,
        },
        daemon=True,
        name=f"prototype5-import-{project_id}",
    )
    thread.start()
    return {"job_id": job_id, "project_id": project_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    job.pop("result_json", None)
    return job


@app.post("/api/demo", status_code=201, include_in_schema=False)
def load_demo() -> dict[str, Any]:
    project_id = "p5_transport_demo_v1"
    existing = repository.get_project(project_id)
    if existing:
        return existing
    if not DEMO_CSV_PATH.exists():
        raise HTTPException(status_code=503, detail="No demonstration data package is installed.")
    database_path = PROJECT_DATA_ROOT / f"{project_id}.sqlite"
    if database_path.exists():
        summary = read_project_summary(database_path)
    else:
        summary = import_csv_project(
            files=[DEMO_CSV_PATH],
            mapping={
                "entity_id": "id",
                "from_node": "from",
                "to_node": "to",
                "length": "length_m",
                "latitude": "lat",
                "longitude": "lon",
                "time": "datetime",
                "speed": "speed_kmh",
                "travel_time": "travel_time_min",
                "time_step": "time_step",
                "region": "region_id",
                "primary_metric": None,
            },
            units={"speed": "km/h", "travel_time": "min"},
            timezone="America/New_York (source convention; not converted)",
            database_path=database_path,
            project_id=project_id,
            project_name="Transport Disruption Demonstration",
            hazard_type="hurricane",
            source="separately_supplied_demo_data",
        )
    repository.save_project(
        project_id=project_id,
        name=summary["name"],
        hazard_type="hurricane",
        source=summary["source"],
        status="ready",
        database_path=str(database_path.resolve()),
        mapping=summary["mapping"],
        summary=summary,
        retention="persistent",
    )
    return repository.get_project(project_id) or {"project_id": project_id, "summary": summary}


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return [
        _enrich_project_record(project) for project in repository.list_persistent_projects()
    ]


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return _project_or_404(project_id)


@app.post("/api/projects/{project_id}/keep")
def keep_project(project_id: str) -> dict[str, Any]:
    project = _project_or_404(project_id)
    if project.get("retention") != "persistent":
        repository.keep_project(project_id)
    kept = repository.get_project(project_id)
    if kept is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _enrich_project_record(kept)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_saved_project(project_id: str) -> Response:
    project = _project_or_404(project_id)
    if project.get("retention") != "persistent":
        raise HTTPException(status_code=409, detail="Only an explicitly saved project can be deleted here.")
    if not _delete_saved_project(project):
        raise HTTPException(status_code=409, detail="The saved project could not be deleted safely.")
    return Response(status_code=204)


@app.get("/api/projects/{project_id}/memory")
def get_project_memory(project_id: str) -> dict[str, Any]:
    memory = repository.project_memory(project_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return memory


def _parse_bbox(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Bounding box must contain numeric west,south,east,north values.") from exc
    if len(parts) != 4:
        raise HTTPException(status_code=422, detail="Bounding box must contain west,south,east,north.")
    return parts


@app.get("/api/projects/{project_id}/map")
def project_map(
    project_id: str,
    metric: str = Query(default="speed", pattern="^(speed|travel_time|primary|flow|capacity|free_flow_time|repair_cost|repair_duration|failure_probability|length)$"),
    timestamp: int | None = None,
    bbox: str | None = None,
    limit: int = Query(default=20000, ge=100, le=20000),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return map_features(
        Path(project["database_path"]), metric=metric, timestamp=timestamp, bbox=_parse_bbox(bbox), limit=limit
    )


@app.get("/api/projects/{project_id}/network-mapping")
def project_network_mapping_status(project_id: str) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return network_mapping_status(Path(project["database_path"]))


@app.post("/api/projects/{project_id}/network-mapping")
def project_build_network_mapping(project_id: str) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return build_network_mapping(Path(project["database_path"]))


@app.get("/api/projects/{project_id}/network-map")
def project_network_map(
    project_id: str,
    metric: str = Query(default="speed", pattern="^(speed|travel_time|primary|flow|capacity|free_flow_time|repair_cost|repair_duration|failure_probability|length)$"),
    timestamp: int | None = None,
    bbox: str | None = None,
    limit: int = Query(default=20000, ge=100, le=20000),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return network_features(
        Path(project["database_path"]), metric=metric, timestamp=timestamp, bbox=_parse_bbox(bbox), limit=limit
    )


@app.get("/api/projects/{project_id}/scenario-targets")
def project_scenario_targets(
    project_id: str,
    bbox: str,
    limit: int = Query(default=5000, ge=1, le=5000),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    parsed = _parse_bbox(bbox)
    if parsed is None:
        raise HTTPException(status_code=422, detail="Draw one rectangular area on the map.")
    return select_scenario_targets(Path(project["database_path"]), bbox=parsed, limit=limit)


@app.get("/api/projects/{project_id}/entities/{region_id}/{entity_id}/history")
def project_entity_history(
    project_id: str,
    region_id: str,
    entity_id: str,
    metric: str = Query(default="speed", pattern="^(speed|travel_time|primary|flow|capacity|free_flow_time|repair_cost|repair_duration|failure_probability|length)$"),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return entity_history(Path(project["database_path"]), region_id, entity_id, metric)


@app.post("/api/commands/interpret")
def command_interpretation(request: CommandRequest) -> dict[str, Any]:
    project = _project_or_404(request.project_id)
    return interpret_command(
        text=request.text,
        project=project,
        selected_keys=request.selected_keys,
        bbox=request.bbox,
        current_metric=request.current_metric,
        current_time=request.current_time,
    )


@app.post("/api/analyses", status_code=201)
def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
    project = _project_or_404(request.project_id)
    payload = request.model_dump()
    return _execute_analysis(project, payload)


def _execute_analysis(project: dict[str, Any], payload: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    database_path = Path(project["database_path"])
    if payload.get("analysis_type") in {
        "critical_components", "closure_scenario", "restoration_screening", "progressive_robustness", "route_detour_analysis"
    }:
        mapping = network_mapping_status(database_path)
        if mapping.get("available") and not mapping.get("ready"):
            try:
                build_network_mapping(database_path)
            except ValueError:
                # Numerical topology analysis remains valid when line geometry
                # cannot be inferred. The result then discloses point geometry.
                pass
    if payload.get("analysis_type") == "route_detour_analysis":
        assumptions = dict(payload.get("assumptions") or {})
        scenario_ids = assumptions.get("scenario_ids") or []
        if scenario_ids and not isinstance(scenario_ids, list):
            raise ValueError("Scenario IDs for route comparison must be a list.")
        if scenario_ids:
            unique_ids = list(dict.fromkeys(str(item).strip() for item in scenario_ids if str(item).strip()))
            variants = []
            for selected_id in unique_ids:
                scenario = repository.get_scenario(selected_id)
                if scenario is None:
                    raise ValueError(f"Saved route scenario no longer exists: {selected_id}.")
                if scenario.get("project_id") != project.get("project_id"):
                    raise ValueError("Every selected route scenario must belong to the current project.")
                variants.append(
                    {
                        "scenario_id": selected_id,
                        "name": scenario.get("name") or "saved disruption",
                        "base_timestamp": scenario.get("base_timestamp"),
                        "changes": scenario.get("changes") or [],
                    }
                )
            assumptions["scenario_ids"] = unique_ids
            assumptions["scenario_variants"] = variants
        scenario_id = str(assumptions.get("scenario_id") or "").strip()
        if scenario_id and not scenario_ids:
            scenario = repository.get_scenario(scenario_id)
            if scenario is None:
                raise ValueError("The selected saved scenario no longer exists.")
            if scenario.get("project_id") != project.get("project_id"):
                raise ValueError("The selected scenario belongs to another project.")
            assumptions.update(
                {
                    "scenario_changes": scenario.get("changes") or [],
                    "scenario_name": scenario.get("name") or "saved disruption",
                }
            )
            if payload.get("time_end") is None and payload.get("time_start") is None:
                payload["time_end"] = scenario.get("base_timestamp")
        payload["assumptions"] = assumptions
    if payload.get("analysis_type") == "restoration_screening":
        assumptions = dict(payload.get("assumptions") or {})
        scenario_id = str(assumptions.get("scenario_id") or "").strip()
        if scenario_id:
            scenario = repository.get_scenario(scenario_id)
            if scenario is None:
                raise ValueError("The selected recovery scenario no longer exists.")
            if scenario.get("project_id") != project.get("project_id"):
                raise ValueError("The selected recovery scenario belongs to another project.")
            changes = scenario.get("changes") or []
            assumptions.update(
                {
                    "scenario_changes": changes,
                    "scenario_name": scenario.get("name") or "saved disruption",
                    "scenario_id": scenario_id,
                }
            )
            if not payload.get("selected_keys"):
                payload["selected_keys"] = [str(item.get("key")) for item in changes if item.get("key")]
        payload["assumptions"] = assumptions
    result = run_analysis(database_path, payload)
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    result["software_version"] = version()
    result["model_registry_version"] = _registry().get("version")
    fingerprint_payload = {
        "analysis_type": result.get("analysis_type"),
        "request": payload,
        "items": result.get("items", []),
        "evidence": result.get("evidence", {}),
        "verification": result.get("verification", {}),
    }
    result["result_fingerprint_sha256"] = sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    result["provenance"] = {
        "analysis_source_sha256": _analysis_source_digest(),
        "prototype5_version": version(),
        "model_registry_version": _registry().get("version"),
        "frozen_at_run": True,
    }
    if not result.get("verification", {}).get("passed", False):
        raise ValueError(
            "The analysis failed its internal verification and was not saved. Review the selection and model inputs."
        )
    result["project_id"] = project["project_id"]
    result["project_name"] = project["name"]
    if save:
        repository.save_run(project["project_id"], payload["analysis_type"], payload, result)
    return result


def _run_analysis_job(job_id: str, project_id: str, payload: dict[str, Any]) -> None:
    try:
        repository.update_job(
            job_id,
            status="running",
            progress=8.0,
            message="Validating the confirmed scope and model inputs",
        )
        project = _project_or_404(project_id)
        repository.update_job(
            job_id,
            progress=15.0,
            message="Running the deterministic model; large histories can take about a minute",
        )
        result = _execute_analysis(project, payload, save=False)
        repository.complete_analysis_job(job_id, project_id, payload["analysis_type"], payload, result)
    except Exception as exc:
        current_job = repository.get_job(job_id)
        if current_job and current_job.get("status") == "cancelled":
            return
        repository.update_job(job_id, status="failed", message="Analysis failed", error=str(exc))


@app.post("/api/analysis-jobs", status_code=202)
def create_analysis_job(request: AnalysisRequest) -> dict[str, Any]:
    _project_or_404(request.project_id)
    payload = request.model_dump()
    job_id = repository.create_job("analysis", "Analysis queued")
    threading.Thread(
        target=_run_analysis_job,
        args=(job_id, request.project_id, payload),
        daemon=True,
        name=f"prototype5-analysis-{job_id}",
    ).start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/{job_id}/cancel", status_code=202)
def cancel_job_wait(job_id: str) -> dict[str, Any]:
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": job["status"]}
    status = repository.cancel_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, "status": status}


def _agent_task_cancelled(task_id: str) -> bool:
    task = repository.get_agent_task(task_id)
    return task is None or task.get("status") == "cancelled" or bool(task.get("cancel_requested"))


def _agent_task_progress(task_id: str, progress: float, stage: str, message: str) -> None:
    if not _agent_task_cancelled(task_id):
        repository.update_agent_task(
            task_id,
            status="running",
            progress=progress,
            stage=stage,
            message=message,
        )


def _resolve_agent_dependencies(
    project_id: str, dependencies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate ownership before an upstream artefact can enter an Agent task."""

    resolved: list[dict[str, Any]] = []
    for reference in dependencies:
        kind = str(reference.get("kind") or "")
        reference_id = str(reference.get("id") or "")
        if kind == "run":
            item = repository.get_run(reference_id)
        elif kind == "scenario":
            item = repository.get_scenario(reference_id)
        elif kind == "task":
            item = repository.get_agent_task(reference_id)
            if item is not None and item.get("status") != "completed":
                raise ValueError(f"Upstream Agent task is not completed: {reference_id}.")
        else:
            raise ValueError(f"Unsupported Agent dependency kind: {kind}.")
        if item is None:
            raise ValueError(f"Agent dependency no longer exists: {reference_id}.")
        if item.get("project_id") != project_id:
            raise ValueError("An Agent task cannot use an artefact from another project.")
        resolved.append(
            {
                "kind": kind,
                "id": reference_id,
                "role": reference.get("role") or "input",
                "project_id": project_id,
            }
        )
    return resolved


def _agent_project_profile(project: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, Any]:
    summary = project.get("summary") or {}
    result: dict[str, Any] = {
        "project_id": project["project_id"],
        "project_name": project["name"],
        "hazard_type": project.get("hazard_type"),
        "summary": summary,
    }
    if tool_input.get("include_model_readiness", True):
        result["model_readiness"] = recommend_models(summary)
    if tool_input.get("include_memory", True):
        memory = repository.project_memory(project["project_id"])
        result["memory_overview"] = memory.get("overview") if memory else None
        result["recent_memory_events"] = (memory.get("events") or [])[-20:] if memory else []
    return result


def _run_agent_task(task_id: str) -> None:
    stage = "validation"
    try:
        task = repository.get_agent_task(task_id)
        if task is None or _agent_task_cancelled(task_id):
            return
        _agent_task_progress(task_id, 3.0, stage, "Validating project, tool input and dependencies")
        project = _project_or_404(task["project_id"])
        tool_id = str(task["tool_id"])
        tool_input = validate_agent_tool_input(tool_id, task.get("input") or {})
        resolved_dependencies = _resolve_agent_dependencies(
            project["project_id"], task.get("dependencies") or []
        )
        if _agent_task_cancelled(task_id):
            return

        if tool_id == "inspect_project_data":
            stage = "data_inspection"
            _agent_task_progress(task_id, 35.0, stage, "Reading the current project evidence profile")
            profile = _agent_project_profile(project, tool_input)
            output = {
                "kind": "project_profile",
                "artifact_refs": [{"kind": "project", "id": project["project_id"]}],
                "dependency_refs": resolved_dependencies,
                "memory_saved": False,
                "result": profile,
            }
            repository.complete_agent_task(task_id, output, "Project profile ready")
            return

        if tool_id in {
            "find_critical_roads",
            "find_critical_areas",
            "plan_recovery_priorities",
            "plan_routes_under_disruption",
            "test_network_resilience",
        }:
            stage = "analysis"
            payload = build_analysis_payload(
                project_id=project["project_id"],
                tool_id=tool_id,
                normalised_input=tool_input,
            )
            _agent_task_progress(
                task_id,
                15.0,
                stage,
                "Running the existing deterministic analysis engine",
            )
            result = _execute_analysis(project, payload, save=False)
            if _agent_task_cancelled(task_id):
                return
            _agent_task_progress(task_id, 92.0, "verification", "Verifying and freezing the result")
            repository.complete_agent_analysis_task(
                task_id,
                project_id=project["project_id"],
                analysis_type=payload["analysis_type"],
                request=payload,
                result=result,
            )
            return

        if tool_id == "build_disruption_scenario":
            stage = "scenario"
            request = build_scenario_request(
                project_id=project["project_id"], normalised_input=tool_input
            )
            changes = [change.model_dump() for change in request.changes]
            _agent_task_progress(
                task_id,
                15.0,
                stage,
                "Evaluating the existing deterministic disruption scenario",
            )
            started = time.perf_counter()
            result = evaluate_disruption_scenario(
                Path(project["database_path"]),
                changes=changes,
                base_timestamp=request.base_timestamp,
            )
            result["runtime_seconds"] = round(time.perf_counter() - started, 6)
            result["software_version"] = version()
            result["project_id"] = project["project_id"]
            result["project_name"] = project["name"]
            result["scenario_name"] = request.name
            result["hazard_type"] = request.hazard_type
            result["provenance"] = {
                "analysis_source_sha256": _analysis_source_digest(),
                "prototype5_version": version(),
                "model_registry_version": _registry().get("version"),
                "frozen_at_evaluation": True,
            }
            if not result.get("verification", {}).get("passed", False):
                raise ValueError("The scenario failed its internal grounding checks and was not saved.")
            if _agent_task_cancelled(task_id):
                return
            _agent_task_progress(task_id, 92.0, "verification", "Verifying and freezing the scenario")
            repository.complete_agent_scenario_task(
                task_id,
                project_id=project["project_id"],
                name=request.name.strip(),
                hazard_type=request.hazard_type,
                base_timestamp=result.get("timestamp_epoch"),
                changes=changes,
                result=result,
            )
            return

        if tool_id == "run_matsim":
            stage = "matsim"
            scenario_id, request = build_matsim_request(tool_input)
            scenario = repository.get_scenario(scenario_id)
            if scenario is None:
                raise ValueError("The MATSim scenario no longer exists.")
            if scenario.get("project_id") != project.get("project_id"):
                raise ValueError("MATSim cannot use a scenario from another project.")
            readiness = assess_matsim_readiness(project)
            if not readiness.get("can_run"):
                raise ValueError(readiness.get("explanation") or "MATSim is not ready for this project.")

            def report(progress: float, message: str) -> None:
                _agent_task_progress(task_id, progress, "matsim", message)

            def cancelled() -> bool:
                return _agent_task_cancelled(task_id)

            _agent_task_progress(task_id, 5.0, stage, "Starting the existing MATSim worker")
            result = run_matsim_scenario(
                project,
                scenario,
                job_id=task_id,
                iterations=request.iterations,
                max_agents=request.max_agents,
                seed=request.seed,
                timeout_seconds=request.timeout_seconds,
                progress=report,
                cancelled=cancelled,
            )
            if _agent_task_cancelled(task_id):
                return
            repository.complete_agent_matsim_task(
                task_id,
                project_id=project["project_id"],
                scenario_id=scenario_id,
                result=result,
            )
            return

        raise ValueError(f"No Agent adapter is registered for {tool_id}.")
    except InterruptedError:
        repository.cancel_agent_task(task_id)
    except Exception as exc:
        if _agent_task_cancelled(task_id):
            return
        code = "invalid_tool_input" if stage == "validation" else "tool_execution_failed"
        repository.update_agent_task(
            task_id,
            status="failed",
            stage=stage,
            message="Agent tool failed",
            error=structured_agent_error(exc, stage=stage, code=code),
        )


@app.get("/api/agent-tools")
def get_agent_tool_catalogue() -> dict[str, Any]:
    return agent_tool_catalog()


@app.post("/api/agent-tasks", status_code=202)
def create_agent_tool_task(request: AgentToolTaskRequest) -> dict[str, Any]:
    _project_or_404(request.project_id)
    normalised_input = validate_agent_tool_input(request.tool_id, request.input)
    task, created = repository.create_agent_task(
        tool_id=request.tool_id,
        project_id=request.project_id,
        tool_input=normalised_input,
        dependencies=[item.model_dump() for item in request.dependencies],
        client_request_id=request.client_request_id,
    )
    if created:
        threading.Thread(
            target=_run_agent_task,
            args=(task["task_id"],),
            daemon=True,
            name=f"twm-agent-{task['task_id']}",
        ).start()
    task["protocol_version"] = AGENT_TOOL_PROTOCOL_VERSION
    task["created"] = created
    return task


@app.get("/api/agent-tasks")
def list_agent_tool_tasks(
    project_id: str = Query(min_length=3, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    include_output: bool = False,
) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    tasks = repository.list_agent_tasks(project_id, limit=limit)
    if not include_output:
        for task in tasks:
            task.pop("output", None)
    return tasks


@app.get("/api/agent-tasks/{task_id}")
def get_agent_tool_task(task_id: str) -> dict[str, Any]:
    task = repository.get_agent_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Agent task not found.")
    task["protocol_version"] = AGENT_TOOL_PROTOCOL_VERSION
    return task


@app.post("/api/agent-tasks/{task_id}/cancel", status_code=202)
def cancel_agent_tool_task(task_id: str) -> dict[str, Any]:
    status = repository.cancel_agent_task(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Agent task not found.")
    return {
        "task_id": task_id,
        "status": status,
        "protocol_version": AGENT_TOOL_PROTOCOL_VERSION,
    }


@app.get("/api/runs")
def list_runs(project_id: str | None = None, limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    return repository.list_runs(project_id=project_id, limit=limit)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return run


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str) -> None:
    if repository.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    if not repository.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")


@app.get("/api/runs/{run_id}/export.json")
def export_run(run_id: str) -> StreamingResponse:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    project = _project_or_404(run["project_id"])
    registry = _registry()
    model_id = run.get("result", {}).get("model", {}).get("id")
    model_record = next((item for item in registry.get("models", []) if item.get("id") == model_id), None)
    exporter_digest = _analysis_source_digest()
    frozen_digest = run.get("result", {}).get("provenance", {}).get("analysis_source_sha256")
    dependencies = {}
    for package in ("fastapi", "pydantic", "pandas", "numpy", "networkx"):
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependencies[package] = "unavailable"
    export = {
        "schema": "prototype5.research-run.v1",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software": {
            "prototype5_version": version(),
            "model_registry_version": registry.get("version"),
            "analysis_source_sha256": frozen_digest or "unavailable_for_legacy_run",
            "exporter_current_source_sha256": exporter_digest,
            "source_digest_frozen_at_run": bool(frozen_digest),
            "python": platform.python_version(),
            "dependencies": dependencies,
        },
        "project": {
            "project_id": project["project_id"],
            "name": project["name"],
            "hazard_type": project["hazard_type"],
            "source": project["source"],
            "mapping": project.get("mapping"),
            "summary": project.get("summary"),
        },
        "model_record": model_record,
        "run": run,
        "reproducibility_note": "Source-file SHA-256 values, confirmed mappings, units, metric semantics, time convention, model evidence and software versions are included. Re-run requires the matching source files.",
    }
    content = json.dumps(export, ensure_ascii=False, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


def _scenario_or_404(scenario_id: str) -> dict[str, Any]:
    scenario = repository.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    return scenario


def _evaluate_and_save_scenario(
    request: ScenarioUpsertRequest,
    *,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    project = _project_or_404(request.project_id)
    started = time.perf_counter()
    changes = [change.model_dump() for change in request.changes]
    result = evaluate_disruption_scenario(
        Path(project["database_path"]),
        changes=changes,
        base_timestamp=request.base_timestamp,
    )
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    result["software_version"] = version()
    result["project_id"] = project["project_id"]
    result["project_name"] = project["name"]
    result["scenario_name"] = request.name
    result["hazard_type"] = request.hazard_type
    result["provenance"] = {
        "analysis_source_sha256": _analysis_source_digest(),
        "prototype5_version": version(),
        "model_registry_version": _registry().get("version"),
        "frozen_at_evaluation": True,
    }
    if not result.get("verification", {}).get("passed", False):
        raise ValueError("The scenario failed its internal grounding checks and was not saved.")
    saved_id = scenario_id or f"scenario_{uuid4().hex[:14]}"
    result["scenario_id"] = saved_id
    repository.save_scenario(
        scenario_id=saved_id,
        project_id=request.project_id,
        name=request.name.strip(),
        hazard_type=request.hazard_type,
        base_timestamp=result.get("timestamp_epoch"),
        changes=changes,
        result=result,
    )
    return _scenario_or_404(saved_id)


@app.get("/api/scenarios")
def list_scenarios(project_id: str = Query(min_length=3, max_length=120)) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return repository.list_scenarios(project_id)


@app.post("/api/scenarios", status_code=201)
def create_scenario(request: ScenarioUpsertRequest) -> dict[str, Any]:
    return _evaluate_and_save_scenario(request)


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> dict[str, Any]:
    return _scenario_or_404(scenario_id)


@app.get("/api/projects/{project_id}/matsim-readiness")
def project_matsim_readiness(project_id: str) -> dict[str, Any]:
    return assess_matsim_readiness(_project_or_404(project_id))


def _run_matsim_job(job_id: str, scenario_id: str, request: MatsimRunRequest) -> None:
    scenario = _scenario_or_404(scenario_id)
    project = _project_or_404(scenario["project_id"])

    def report(progress: float, message: str) -> None:
        current = repository.get_job(job_id)
        if current and current["status"] != "cancelled":
            repository.update_job(job_id, status="running", progress=progress, message=message)

    def cancelled() -> bool:
        current = repository.get_job(job_id)
        return not current or current["status"] == "cancelled"

    try:
        repository.update_job(job_id, status="running", progress=2, message="Starting MATSim worker")
        result = run_matsim_scenario(
            project,
            scenario,
            job_id=job_id,
            iterations=request.iterations,
            max_agents=request.max_agents,
            seed=request.seed,
            timeout_seconds=request.timeout_seconds,
            progress=report,
            cancelled=cancelled,
        )
        if cancelled():
            return
        if not repository.attach_matsim_result(scenario_id, result):
            raise ValueError("The saved scenario was removed before MATSim completed.")
        repository.update_job(
            job_id, status="completed", progress=100,
            message="MATSim comparison and playback ready", result=result,
        )
    except InterruptedError:
        repository.update_job(job_id, status="cancelled", message="MATSim run cancelled")
    except Exception as exc:
        if not cancelled():
            repository.update_job(
                job_id, status="failed", message="MATSim simulation failed", error=str(exc)
            )


@app.post("/api/scenarios/{scenario_id}/matsim", status_code=202)
def run_scenario_matsim(scenario_id: str, request: MatsimRunRequest) -> dict[str, Any]:
    scenario = _scenario_or_404(scenario_id)
    project = _project_or_404(scenario["project_id"])
    readiness = assess_matsim_readiness(project)
    if not readiness.get("can_run"):
        detail = readiness.get("explanation") or "MATSim is not ready for this project."
        if readiness.get("status") == "runtime_missing":
            detail = "The MATSim Java worker is not built on this installation."
        raise HTTPException(status_code=409, detail=detail)
    job_id = repository.create_job("matsim_simulation", "MATSim simulation queued")
    threading.Thread(
        target=_run_matsim_job, args=(job_id, scenario_id, request), daemon=True,
        name=f"matsim-{job_id}",
    ).start()
    return {"job_id": job_id, "readiness": readiness}


@app.post("/api/scenarios/{scenario_id}")
def update_scenario(scenario_id: str, request: ScenarioUpsertRequest) -> dict[str, Any]:
    existing = _scenario_or_404(scenario_id)
    if existing["project_id"] != request.project_id:
        raise HTTPException(status_code=409, detail="A scenario cannot be moved to another project.")
    return _evaluate_and_save_scenario(request, scenario_id=scenario_id)


@app.delete("/api/scenarios/{scenario_id}", status_code=204)
def delete_scenario(scenario_id: str) -> None:
    _scenario_or_404(scenario_id)
    if not repository.delete_scenario(scenario_id):
        raise HTTPException(status_code=404, detail="Scenario not found.")


@app.get("/api/runs/{run_id}/export.csv")
def export_run_csv(run_id: str) -> StreamingResponse:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    items = run.get("result", {}).get("items", [])
    fields = sorted({key for item in items for key in item})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in items:
        writer.writerow({
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in item.items()
        })
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@app.get("/api/runs/{run_id}/export.geojson")
def export_run_geojson(run_id: str) -> StreamingResponse:
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    collection = run.get("result", {}).get("feature_collection") or {"type": "FeatureCollection", "features": []}
    return StreamingResponse(
        iter([json.dumps(collection, ensure_ascii=False, indent=2)]),
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.geojson"'},
    )


@app.get("/api/model-library")
def model_library() -> dict[str, Any]:
    return _registry()


@app.get("/api/scientific-validation/release")
def release_scientific_validation() -> dict[str, Any]:
    validation_root = Path(__file__).resolve().parents[2] / "validation"
    report_path = validation_root / "scientific_validation_report.json"
    if not report_path.exists():
        return {
            "protocol": "twm.scientific-validation.v1",
            "status": "not_run",
            "summary": {"passed": 0, "total": 0, "plain_language": "Run the packaged scientific validation suite to create the release report."},
            "checks": [],
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    external_path = validation_root / "external_data_matrix_report.json"
    if external_path.exists():
        external = json.loads(external_path.read_text(encoding="utf-8"))
        report["external_data_matrix"] = {
            "status": external.get("status"),
            "summary": external.get("summary", {}),
            "boundary": external.get("boundary"),
        }
    return report


@app.post("/api/projects/{project_id}/scientific-validation")
def project_scientific_validation(
    project_id: str,
    top_k: int = Query(default=10, ge=3, le=25),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    report = validate_project(Path(project["database_path"]), top_k=top_k)
    report["project_id"] = project_id
    report["project_name"] = project["name"]
    report["software_version"] = version()
    return report


def _run_research_validation_job(job_id: str, project_id: str, profile: str) -> None:
    """Run the publication evidence suite without touching operational state."""

    def cancelled() -> bool:
        job = repository.get_job(job_id)
        return job is None or job.get("status") == "cancelled"

    def progress(value: float, message: str) -> None:
        if not cancelled():
            repository.update_job(
                job_id,
                status="running",
                progress=max(0.0, min(99.0, float(value))),
                message=message,
            )

    try:
        project = _project_or_404(project_id)
        runs = repository.list_runs(project_id=project_id, limit=500)
        tasks = repository.list_agent_tasks(project_id=project_id, limit=500)
        registry = _registry()
        report = run_research_validation(
            project,
            software_version=version(),
            model_registry_version=str(registry.get("version") or version()),
            execution_context={
                "saved_run_count": len(runs),
                "agent_task_count": len(tasks),
                "completed_agent_task_count": sum(item.get("status") == "completed" for item in tasks),
            },
            profile=profile,
            progress=progress,
            cancelled=cancelled,
        )
        if cancelled():
            return
        repository.update_job(
            job_id,
            status="completed",
            progress=100,
            message="Research evidence verified and saved",
            result=report,
        )
    except ResearchValidationCancelled:
        if not cancelled():
            repository.update_job(job_id, status="cancelled", message="Research validation cancelled")
    except Exception as exc:
        if not cancelled():
            repository.update_job(
                job_id,
                status="failed",
                message="Research validation failed",
                error=str(exc),
            )


@app.post("/api/projects/{project_id}/research-validation", status_code=202)
def start_project_research_validation(
    project_id: str,
    profile: str = Query(default="standard", pattern=r"^standard$"),
) -> dict[str, Any]:
    _project_or_404(project_id)
    job_id = repository.create_job("research_validation", "Research validation queued")
    threading.Thread(
        target=_run_research_validation_job,
        args=(job_id, project_id, profile),
        daemon=True,
        name=f"research-validation-{job_id}",
    ).start()
    return {"job_id": job_id, "profile": profile, "project_id": project_id}


@app.get("/api/projects/{project_id}/research-validation")
def project_research_validation_reports(project_id: str) -> list[dict[str, Any]]:
    _project_or_404(project_id)
    return list_research_reports(project_id)


@app.get("/api/research-validation/{report_id}")
def research_validation_report(report_id: str) -> dict[str, Any]:
    report = load_research_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Research-validation report not found.")
    return report


@app.get("/api/research-validation/{report_id}/export.json")
def export_research_validation_json(report_id: str) -> StreamingResponse:
    report = load_research_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Research-validation report not found.")
    return StreamingResponse(
        iter([json.dumps(report, ensure_ascii=False, indent=2)]),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{report_id}.json"'},
    )


@app.get("/api/research-validation/{report_id}/export.csv")
def export_research_validation_csv(report_id: str) -> StreamingResponse:
    report = load_research_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Research-validation report not found.")
    return StreamingResponse(
        iter([report_csv(report)]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{report_id}.csv"'},
    )


@app.get("/api/research-validation/{report_id}/export.tex")
def export_research_validation_latex(report_id: str) -> StreamingResponse:
    report = load_research_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Research-validation report not found.")
    return StreamingResponse(
        iter([report_latex(report)]),
        media_type="application/x-tex; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{report_id}.tex"'},
    )


@app.get("/api/projects/{project_id}/model-recommendations")
def project_model_recommendations(
    project_id: str,
    goal: str | None = Query(default=None, max_length=500),
) -> dict[str, Any]:
    project = _project_or_404(project_id)
    return recommend_models(project.get("summary"), goal=goal)


@app.get("/api/assistant/status")
def get_assistant_status(
    session_id: str | None = Query(default=None, min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
) -> dict[str, Any]:
    return assistant_status(session_id)


@app.get("/api/assistant/chats")
def list_assistant_chats(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """List locally persisted Copilot conversations; credentials are never stored here."""
    return repository.list_assistant_chats(limit=limit)


@app.post("/api/assistant/chats", status_code=201)
def create_assistant_chat(request: AssistantChatCreateRequest) -> dict[str, Any]:
    return repository.create_assistant_chat(
        title=request.title,
        project_id=request.project_id,
    )


@app.get("/api/assistant/chats/{chat_id}")
def get_assistant_chat(chat_id: str) -> dict[str, Any]:
    chat = repository.get_assistant_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return chat


@app.post("/api/assistant/chats/{chat_id}/messages", status_code=201)
def append_assistant_chat_message(
    chat_id: str,
    request: AssistantChatMessageCreateRequest,
) -> dict[str, Any]:
    message = repository.append_assistant_chat_message(
        chat_id,
        role=request.role,
        content=request.content,
        metadata=request.metadata,
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return message


@app.delete("/api/assistant/chats/{chat_id}", status_code=204)
def delete_assistant_chat(chat_id: str) -> None:
    if not repository.delete_assistant_chat(chat_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")


@app.post("/api/assistant/provider")
def configure_assistant_provider(request: AssistantProviderConfigRequest) -> dict[str, Any]:
    api_key = request.api_key.get_secret_value() if request.api_key is not None else None
    status = configure_provider(
        session_id=request.session_id,
        provider=request.provider,
        api_key=api_key,
        model=request.model,
        clear=request.clear,
    )
    if request.clear:
        return status
    # Connection is one backend transaction: model discovery is optional and
    # cannot block a valid key, while a real inference request is the only
    # success criterion. This prevents /v1/models outages from becoming 503
    # connection failures and keeps the contract independent of other UI work.
    refresh_provider_models(session_id=request.session_id, provider=request.provider)
    return verify_provider_inference(session_id=request.session_id, provider=request.provider)


@app.post("/api/assistant/provider/models")
def refresh_assistant_provider_models(
    session_id: str = Query(..., min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    provider: str = Query(..., min_length=2, max_length=24),
) -> dict[str, Any]:
    return refresh_provider_models(session_id=session_id, provider=provider)


@app.post("/api/assistant/provider/verify")
def verify_assistant_provider(
    session_id: str = Query(..., min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    provider: str = Query(..., min_length=2, max_length=24),
) -> dict[str, Any]:
    return verify_provider_inference(session_id=session_id, provider=provider)


def _verified_assistant_chat_artifacts(
    project_id: str | None,
    artifacts: list[Any],
) -> list[dict[str, Any]]:
    """Resolve Copilot artifact references against persisted project records.

    Browser metadata is only a pointer.  A previous conversation, a different
    project or edited client state must never become evidence for the active
    Copilot task.
    """
    if not project_id:
        return []
    verified: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in artifacts:
        payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        kind = str(payload.get("kind") or "")
        artifact_id = str(payload.get("artifact_id") or "")
        analysis_type = str(payload.get("analysis_type") or "")
        identity = (kind, artifact_id)
        if not artifact_id or identity in seen:
            continue
        if kind == "analysis_result":
            record = repository.get_run(artifact_id)
            if (
                record is None
                or record.get("project_id") != project_id
                or record.get("analysis_type") != analysis_type
            ):
                continue
            created_at = record.get("created_at")
        elif kind == "scenario_result":
            record = repository.get_scenario(artifact_id)
            if (
                record is None
                or record.get("project_id") != project_id
                or analysis_type != "closure_scenario"
            ):
                continue
            created_at = record.get("created_at")
        else:
            continue
        seen.add(identity)
        verified.append({
            "kind": kind,
            "artifact_id": artifact_id,
            "analysis_type": analysis_type,
            "project_id": project_id,
            "created_at": created_at or payload.get("created_at"),
        })
    return verified


@app.post("/api/assistant/message")
def assistant_message(request: AssistantMessageRequest) -> dict[str, Any]:
    registry = _registry()
    project = _project_or_404(request.project_id) if request.project_id else None
    if project is not None:
        memory = repository.project_memory(project["project_id"])
        project = {
            **project,
            "model_registry": registry.get("models", []),
            "memory_overview": memory.get("overview") if memory else None,
            "recent_memory_events": (memory.get("events") or [])[-20:] if memory else [],
        }
    context = AssistantRequestContext(
        message=request.message,
        project=project,
        registry=registry,
        selected_keys=request.selected_keys,
        bbox=request.bbox,
        current_metric=request.current_metric,
        current_time=request.current_time,
        current_view=request.current_view,
        history=[item.model_dump() for item in request.history],
        active_scenario_name=request.active_scenario_name,
        scenario_count=request.scenario_count,
        scenario_building=request.scenario_building,
        chat_artifacts=_verified_assistant_chat_artifacts(
            request.project_id,
            request.chat_artifacts,
        ),
    )
    result = assistant_response(
        context=context,
        prefer_llm=request.prefer_llm,
        session_id=request.session_id,
        provider=request.provider,
        interaction_mode=request.interaction_mode,
    )
    result["action_policy"] = f"allowlisted_{request.interaction_mode}"
    return result


app.mount(
    "/perception",
    infrastructure_perception_app,
    name="infrastructure-perception",
)
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/{path:path}", include_in_schema=False)
def application_fallback(path: str) -> FileResponse:
    candidate = (FRONTEND_ROOT / path).resolve()
    if FRONTEND_ROOT.resolve() in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(FRONTEND_ROOT / "index.html")
