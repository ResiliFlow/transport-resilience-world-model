from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """Small metadata store. Large observations live in one SQLite file per project."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._write_lock = threading.RLock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialise(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    paths_json TEXT NOT NULL,
                    preview_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    hazard_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    database_path TEXT NOT NULL,
                    mapping_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_project_time
                    ON runs(project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    hazard_type TEXT NOT NULL,
                    base_timestamp INTEGER,
                    changes_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scenarios_project_time
                    ON scenarios(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    client_request_id TEXT,
                    tool_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    output_json TEXT,
                    error_json TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_project_time
                    ON agent_tasks(project_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_tasks_client_request
                    ON agent_tasks(project_id, client_request_id)
                    WHERE client_request_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS assistant_chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    project_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assistant_chats_time
                    ON assistant_chats(updated_at DESC);
                CREATE TABLE IF NOT EXISTS assistant_chat_messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES assistant_chats(chat_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_assistant_chat_messages_time
                    ON assistant_chat_messages(chat_id, created_at);
                """
            )

            # Backward-compatible lifecycle fields. Existing uploads and projects
            # remain persistent; only records created after this migration are
            # eligible for automatic temporary-session cleanup.
            upload_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(uploads)").fetchall()
            }
            if "is_temporary" not in upload_columns:
                connection.execute(
                    "ALTER TABLE uploads ADD COLUMN is_temporary INTEGER NOT NULL DEFAULT 0"
                )
            project_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
            }
            if "retention" not in project_columns:
                connection.execute(
                    "ALTER TABLE projects ADD COLUMN retention TEXT NOT NULL DEFAULT 'persistent'"
                )

    def save_upload(
        self,
        paths: list[str],
        preview: dict[str, Any],
        upload_id: str | None = None,
        *,
        temporary: bool = True,
    ) -> str:
        upload_id = upload_id or f"upload_{uuid4().hex[:14]}"
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO uploads
                    (upload_id, paths_json, preview_json, created_at, is_temporary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (upload_id, json.dumps(paths), json.dumps(preview), _now(), int(temporary)),
            )
        return upload_id

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)).fetchone()
        if row is None:
            return None
        return {
            "upload_id": row["upload_id"],
            "paths": json.loads(row["paths_json"]),
            "preview": json.loads(row["preview_json"]),
            "created_at": row["created_at"],
            "is_temporary": bool(row["is_temporary"]),
        }

    @staticmethod
    def _assistant_chat_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "chat_id": row["chat_id"],
            "title": row["title"],
            "project_id": row["project_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_assistant_chat(
        self,
        *,
        title: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        chat_id = f"chat_{uuid4().hex[:18]}"
        clean_title = " ".join(str(title or "New chat").split()).strip()[:80] or "New chat"
        now = _now()
        with self._write_lock, self._connect() as connection:
            if project_id is not None:
                exists = connection.execute(
                    "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
                ).fetchone()
                if exists is None:
                    project_id = None
            connection.execute(
                """
                INSERT INTO assistant_chats
                    (chat_id, title, project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, clean_title, project_id, now, now),
            )
            row = connection.execute(
                "SELECT * FROM assistant_chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return self._assistant_chat_from_row(row)

    def list_assistant_chats(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.message_id) AS message_count
                FROM assistant_chats c
                LEFT JOIN assistant_chat_messages m ON m.chat_id = c.chat_id
                GROUP BY c.chat_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        chats: list[dict[str, Any]] = []
        for row in rows:
            chat = self._assistant_chat_from_row(row)
            chat["message_count"] = int(row["message_count"] or 0)
            chats.append(chat)
        return chats

    def get_assistant_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM assistant_chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                """
                SELECT * FROM assistant_chat_messages
                WHERE chat_id = ? ORDER BY created_at, rowid
                """,
                (chat_id,),
            ).fetchall()
        chat = self._assistant_chat_from_row(row)
        chat["messages"] = [
            {
                "message_id": item["message_id"],
                "role": item["role"],
                "content": item["content"],
                "metadata": json.loads(item["metadata_json"] or "{}"),
                "created_at": item["created_at"],
            }
            for item in messages
        ]
        return chat

    def append_assistant_chat_message(
        self,
        chat_id: str,
        *,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        message_id = f"msg_{uuid4().hex[:20]}"
        now = _now()
        with self._write_lock, self._connect() as connection:
            chat = connection.execute(
                "SELECT title FROM assistant_chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if chat is None:
                return None
            connection.execute(
                """
                INSERT INTO assistant_chat_messages
                    (message_id, chat_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, chat_id, role, content, json.dumps(metadata or {}), now),
            )
            title = chat["title"]
            if title == "New chat" and role == "user":
                title = " ".join(content.split()).strip()[:80] or title
            connection.execute(
                "UPDATE assistant_chats SET title = ?, updated_at = ? WHERE chat_id = ?",
                (title, now, chat_id),
            )
        return {
            "message_id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def delete_assistant_chat(self, chat_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM assistant_chats WHERE chat_id = ?", (chat_id,)
            )
        return cursor.rowcount > 0

    def list_temporary_uploads(self, *, created_before: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if created_before:
                rows = connection.execute(
                    "SELECT * FROM uploads WHERE is_temporary = 1 AND created_at < ? ORDER BY created_at",
                    (created_before,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM uploads WHERE is_temporary = 1 ORDER BY created_at"
                ).fetchall()
        return [
            {
                "upload_id": row["upload_id"],
                "paths": json.loads(row["paths_json"]),
                "preview": json.loads(row["preview_json"]),
                "created_at": row["created_at"],
                "is_temporary": True,
            }
            for row in rows
        ]

    def delete_upload(self, upload_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM uploads WHERE upload_id = ?", (upload_id,))
        return cursor.rowcount > 0

    def create_job(self, job_type: str, message: str) -> str:
        job_id = f"job_{uuid4().hex[:14]}"
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (job_id, job_type, "queued", 0.0, message, now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._write_lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return
            connection.execute(
                """
                UPDATE jobs SET status = ?, progress = ?, message = ?, result_json = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status if status is not None else row["status"],
                    progress if progress is not None else row["progress"],
                    message if message is not None else row["message"],
                    json.dumps(result) if result is not None else row["result_json"],
                    error if error is not None else row["error"],
                    _now(),
                    job_id,
                ),
            )

    def cancel_job(self, job_id: str) -> str | None:
        """Atomically cancel an active job without overwriting a completion."""
        with self._write_lock, self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["status"] in {"completed", "failed", "cancelled"}:
                return str(row["status"])
            connection.execute(
                """
                UPDATE jobs SET status = 'cancelled', message = ?, updated_at = ?
                WHERE job_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (
                    "The user stopped waiting; any in-process calculation will not be saved as a completed run.",
                    _now(),
                    job_id,
                ),
            )
            return "cancelled"

    def complete_analysis_job(
        self,
        job_id: str,
        project_id: str,
        analysis_type: str,
        request: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        """Atomically save a run and complete its job only while still active."""
        with self._write_lock, self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or row["status"] in {"completed", "failed", "cancelled"}:
                return False
            run_id = str(result.get("run_id") or f"run_{uuid4().hex[:14]}")
            result["run_id"] = run_id
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, project_id, analysis_type, json.dumps(request), json.dumps(result), _now()),
            )
            connection.execute(
                """
                UPDATE jobs SET status = 'completed', progress = 100.0, message = ?,
                    result_json = ?, error = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                ("Analysis verified and saved", json.dumps(result), _now(), job_id),
            )
            return True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "result_json": row["result_json"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _agent_task_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "client_request_id": row["client_request_id"],
            "tool_id": row["tool_id"],
            "project_id": row["project_id"],
            "status": row["status"],
            "progress": row["progress"],
            "stage": row["stage"],
            "message": row["message"],
            "input": json.loads(row["input_json"]),
            "dependencies": json.loads(row["dependencies_json"]),
            "output": json.loads(row["output_json"]) if row["output_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    def create_agent_task(
        self,
        *,
        tool_id: str,
        project_id: str,
        tool_input: dict[str, Any],
        dependencies: list[dict[str, Any]],
        client_request_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Create an isolated Agent task, with optional idempotent client identity."""

        input_json = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
        dependencies_json = json.dumps(dependencies, sort_keys=True, separators=(",", ":"))
        with self._write_lock, self._connect() as connection:
            project = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project is None:
                raise ValueError("The Agent task project does not exist.")
            if client_request_id:
                existing = connection.execute(
                    "SELECT * FROM agent_tasks WHERE project_id = ? AND client_request_id = ?",
                    (project_id, client_request_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["tool_id"] != tool_id
                        or existing["input_json"] != input_json
                        or existing["dependencies_json"] != dependencies_json
                    ):
                        raise ValueError(
                            "This client request ID is already attached to a different Agent task."
                        )
                    return self._agent_task_from_row(existing), False
            task_id = f"task_{uuid4().hex[:14]}"
            now = _now()
            connection.execute(
                """
                INSERT INTO agent_tasks (
                    task_id, client_request_id, tool_id, project_id, status, progress,
                    stage, message, input_json, dependencies_json, output_json,
                    error_json, cancel_requested, created_at, updated_at, started_at,
                    finished_at
                ) VALUES (?, ?, ?, ?, 'queued', 0.0, 'queued', ?, ?, ?, NULL, NULL, 0, ?, ?, NULL, NULL)
                """,
                (
                    task_id,
                    client_request_id,
                    tool_id,
                    project_id,
                    "Agent tool queued",
                    input_json,
                    dependencies_json,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._agent_task_from_row(row), True

    def get_agent_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._agent_task_from_row(row) if row else None

    def list_agent_tasks(self, project_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_tasks
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [self._agent_task_from_row(row) for row in rows]

    def update_agent_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        stage: str | None = None,
        message: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> bool:
        terminal = {"completed", "failed", "cancelled"}
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] in terminal:
                return False
            next_status = status or row["status"]
            now = _now()
            started_at = row["started_at"] or (now if next_status == "running" else None)
            finished_at = now if next_status in terminal else row["finished_at"]
            connection.execute(
                """
                UPDATE agent_tasks SET status = ?, progress = ?, stage = ?, message = ?,
                    error_json = ?, updated_at = ?, started_at = ?, finished_at = ?
                WHERE task_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (
                    next_status,
                    max(0.0, min(100.0, float(progress))) if progress is not None else row["progress"],
                    stage or row["stage"],
                    message if message is not None else row["message"],
                    json.dumps(error) if error is not None else row["error_json"],
                    now,
                    started_at,
                    finished_at,
                    task_id,
                ),
            )
        return True

    def cancel_agent_task(self, task_id: str) -> str | None:
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            if row["status"] in {"completed", "failed", "cancelled"}:
                return str(row["status"])
            now = _now()
            connection.execute(
                """
                UPDATE agent_tasks SET status = 'cancelled', cancel_requested = 1,
                    stage = 'cancelled', message = ?, updated_at = ?, finished_at = ?
                WHERE task_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (
                    "The Agent task was cancelled; no pending output was saved.",
                    now,
                    now,
                    task_id,
                ),
            )
        return "cancelled"

    def complete_agent_task(self, task_id: str, output: dict[str, Any], message: str) -> bool:
        now = _now()
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None or row["status"] in {"completed", "failed", "cancelled"}:
                return False
            connection.execute(
                """
                UPDATE agent_tasks SET status = 'completed', progress = 100.0,
                    stage = 'completed', message = ?, output_json = ?, error_json = NULL,
                    updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                (message, json.dumps(output), now, now, task_id),
            )
        return True

    def complete_agent_analysis_task(
        self,
        task_id: str,
        *,
        project_id: str,
        analysis_type: str,
        request: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        """Atomically persist one verified run and its owning Agent task output."""

        now = _now()
        with self._write_lock, self._connect() as connection:
            task = connection.execute(
                "SELECT status, project_id FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if (
                task is None
                or task["status"] in {"completed", "failed", "cancelled"}
                or task["project_id"] != project_id
            ):
                return False
            run_id = str(result.get("run_id") or f"run_{uuid4().hex[:14]}")
            result["run_id"] = run_id
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, project_id, analysis_type, json.dumps(request), json.dumps(result), now),
            )
            output = {
                "kind": "analysis",
                "artifact_refs": [{"kind": "run", "id": run_id}],
                "memory_saved": True,
                "result": result,
            }
            connection.execute(
                """
                UPDATE agent_tasks SET status = 'completed', progress = 100.0,
                    stage = 'completed', message = ?, output_json = ?, error_json = NULL,
                    updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                ("Analysis verified and saved", json.dumps(output), now, now, task_id),
            )
        return True

    def complete_agent_scenario_task(
        self,
        task_id: str,
        *,
        project_id: str,
        name: str,
        hazard_type: str,
        base_timestamp: int | None,
        changes: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> bool:
        """Atomically persist one verified scenario and its Agent task output."""

        now = _now()
        scenario_id = str(result.get("scenario_id") or f"scenario_{uuid4().hex[:14]}")
        result["scenario_id"] = scenario_id
        with self._write_lock, self._connect() as connection:
            task = connection.execute(
                "SELECT status, project_id FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if (
                task is None
                or task["status"] in {"completed", "failed", "cancelled"}
                or task["project_id"] != project_id
            ):
                return False
            connection.execute(
                """
                INSERT INTO scenarios (
                    scenario_id, project_id, name, hazard_type, base_timestamp,
                    changes_json, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario_id,
                    project_id,
                    name,
                    hazard_type,
                    base_timestamp,
                    json.dumps(changes),
                    json.dumps(result),
                    now,
                    now,
                ),
            )
            output = {
                "kind": "scenario",
                "artifact_refs": [{"kind": "scenario", "id": scenario_id}],
                "memory_saved": True,
                "result": result,
            }
            connection.execute(
                """
                UPDATE agent_tasks SET status = 'completed', progress = 100.0,
                    stage = 'completed', message = ?, output_json = ?, error_json = NULL,
                    updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                ("Scenario verified and saved", json.dumps(output), now, now, task_id),
            )
        return True

    def complete_agent_matsim_task(
        self,
        task_id: str,
        *,
        project_id: str,
        scenario_id: str,
        result: dict[str, Any],
    ) -> bool:
        """Attach MATSim output only if the same isolated task is still active."""

        now = _now()
        with self._write_lock, self._connect() as connection:
            task = connection.execute(
                "SELECT status, project_id FROM agent_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            scenario = connection.execute(
                "SELECT project_id, result_json FROM scenarios WHERE scenario_id = ?",
                (scenario_id,),
            ).fetchone()
            if (
                task is None
                or scenario is None
                or task["status"] in {"completed", "failed", "cancelled"}
                or task["project_id"] != project_id
                or scenario["project_id"] != project_id
            ):
                return False
            scenario_result = json.loads(scenario["result_json"])
            scenario_result["matsim"] = result
            connection.execute(
                "UPDATE scenarios SET result_json = ?, updated_at = ? WHERE scenario_id = ?",
                (json.dumps(scenario_result), now, scenario_id),
            )
            output = {
                "kind": "matsim",
                "artifact_refs": [{"kind": "scenario", "id": scenario_id}],
                "memory_saved": True,
                "result": result,
            }
            connection.execute(
                """
                UPDATE agent_tasks SET status = 'completed', progress = 100.0,
                    stage = 'completed', message = ?, output_json = ?, error_json = NULL,
                    updated_at = ?, finished_at = ?
                WHERE task_id = ?
                """,
                ("MATSim result verified and attached", json.dumps(output), now, now, task_id),
            )
        return True

    def save_project(
        self,
        *,
        project_id: str,
        name: str,
        hazard_type: str,
        source: str,
        status: str,
        database_path: str,
        mapping: dict[str, Any],
        summary: dict[str, Any],
        retention: str = "persistent",
    ) -> None:
        if retention not in {"temporary", "persistent"}:
            raise ValueError("Project retention must be temporary or persistent.")
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects
                    (project_id, name, hazard_type, source, status, database_path,
                     mapping_json, summary_json, created_at, updated_at, retention)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name=excluded.name,
                    hazard_type=excluded.hazard_type,
                    source=excluded.source,
                    status=excluded.status,
                    database_path=excluded.database_path,
                    mapping_json=excluded.mapping_json,
                    summary_json=excluded.summary_json,
                    updated_at=excluded.updated_at,
                    retention=excluded.retention
                """,
                (
                    project_id,
                    name,
                    hazard_type,
                    source,
                    status,
                    database_path,
                    json.dumps(mapping),
                    json.dumps(summary),
                    now,
                    now,
                    retention,
                ),
            )

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "project_id": row["project_id"],
            "name": row["name"],
            "hazard_type": row["hazard_type"],
            "source": row["source"],
            "status": row["status"],
            "database_path": row["database_path"],
            "mapping": json.loads(row["mapping_json"]),
            "summary": json.loads(row["summary_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "retention": row["retention"],
        }

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,)).fetchone()
        return self._project_from_row(row) if row else None

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [self._project_from_row(row) for row in rows]

    def list_persistent_projects(self) -> list[dict[str, Any]]:
        """Return only projects the user chose to keep beyond the current session."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects WHERE retention = 'persistent' ORDER BY updated_at DESC"
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def keep_project(self, project_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE projects SET retention = 'persistent', updated_at = ? WHERE project_id = ?",
                (_now(), project_id),
            )
        return cursor.rowcount > 0

    def list_temporary_projects(self, *, created_before: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if created_before:
                rows = connection.execute(
                    """
                    SELECT * FROM projects
                    WHERE retention = 'temporary' AND created_at < ?
                    ORDER BY created_at
                    """,
                    (created_before,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM projects WHERE retention = 'temporary' ORDER BY created_at"
                ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def _delete_project_with_retention(
        self,
        project_id: str,
        *,
        required_retention: str,
    ) -> tuple[bool, list[str]]:
        """Delete one project only when its lifecycle state matches the caller."""
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT retention FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if row is None or row["retention"] != required_retention:
                return False, []
            scenario_rows = connection.execute(
                "SELECT scenario_id FROM scenarios WHERE project_id = ?", (project_id,)
            ).fetchall()
            scenario_ids = [str(item["scenario_id"]) for item in scenario_rows]
            connection.execute("DELETE FROM agent_tasks WHERE project_id = ?", (project_id,))
            connection.execute("DELETE FROM runs WHERE project_id = ?", (project_id,))
            connection.execute("DELETE FROM scenarios WHERE project_id = ?", (project_id,))
            connection.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        return True, scenario_ids

    def delete_project(self, project_id: str) -> tuple[bool, list[str]]:
        """Delete one temporary project during bounded session cleanup."""
        return self._delete_project_with_retention(
            project_id,
            required_retention="temporary",
        )

    def delete_persistent_project(self, project_id: str) -> tuple[bool, list[str]]:
        """Delete one explicitly saved project after a deliberate user request."""
        return self._delete_project_with_retention(
            project_id,
            required_retention="persistent",
        )

    def save_run(self, project_id: str, analysis_type: str, request: dict[str, Any], result: dict[str, Any]) -> str:
        run_id = str(result.get("run_id") or f"run_{uuid4().hex[:14]}")
        result["run_id"] = run_id
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, project_id, analysis_type, json.dumps(request), json.dumps(result), _now()),
            )
        return run_id

    def list_runs(self, project_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if project_id:
                rows = connection.execute(
                    "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                    (project_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "project_id": row["project_id"],
                "analysis_type": row["analysis_type"],
                "created_at": row["created_at"],
                "result": json.loads(row["result_json"]),
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "project_id": row["project_id"],
            "analysis_type": row["analysis_type"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }

    def delete_run(self, run_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _scenario_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "scenario_id": row["scenario_id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "hazard_type": row["hazard_type"],
            "base_timestamp": row["base_timestamp"],
            "changes": json.loads(row["changes_json"]),
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_scenario(
        self,
        *,
        project_id: str,
        name: str,
        hazard_type: str,
        base_timestamp: int | None,
        changes: list[dict[str, Any]],
        result: dict[str, Any],
        scenario_id: str | None = None,
    ) -> str:
        scenario_id = scenario_id or f"scenario_{uuid4().hex[:14]}"
        now = _now()
        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT project_id, created_at FROM scenarios WHERE scenario_id = ?",
                (scenario_id,),
            ).fetchone()
            if existing is not None and existing["project_id"] != project_id:
                raise ValueError("A scenario can only be updated inside its original project.")
            created_at = existing["created_at"] if existing is not None else now
            connection.execute(
                """
                INSERT INTO scenarios
                    (scenario_id, project_id, name, hazard_type, base_timestamp,
                     changes_json, result_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET
                    name=excluded.name,
                    hazard_type=excluded.hazard_type,
                    base_timestamp=excluded.base_timestamp,
                    changes_json=excluded.changes_json,
                    result_json=excluded.result_json,
                    updated_at=excluded.updated_at
                """,
                (
                    scenario_id,
                    project_id,
                    name,
                    hazard_type,
                    base_timestamp,
                    json.dumps(changes),
                    json.dumps(result),
                    created_at,
                    now,
                ),
            )
        return scenario_id

    def list_scenarios(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scenarios WHERE project_id = ? ORDER BY updated_at DESC, name",
                (project_id,),
            ).fetchall()
        return [self._scenario_from_row(row) for row in rows]

    def get_scenario(self, scenario_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scenarios WHERE scenario_id = ?",
                (scenario_id,),
            ).fetchone()
        return self._scenario_from_row(row) if row else None

    def attach_matsim_result(self, scenario_id: str, matsim_result: dict[str, Any]) -> bool:
        """Attach a completed process-isolated simulation to its saved scenario."""
        with self._write_lock, self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM scenarios WHERE scenario_id = ?", (scenario_id,)
            ).fetchone()
            if row is None:
                return False
            result = json.loads(row["result_json"])
            result["matsim"] = matsim_result
            connection.execute(
                "UPDATE scenarios SET result_json = ?, updated_at = ? WHERE scenario_id = ?",
                (json.dumps(result), _now(), scenario_id),
            )
        return True

    def delete_scenario(self, scenario_id: str) -> bool:
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM scenarios WHERE scenario_id = ?", (scenario_id,))
        return cursor.rowcount > 0

    def project_memory(self, project_id: str) -> dict[str, Any] | None:
        """Build a read-only provenance ledger from persisted project records.

        This deliberately excludes assistant API credentials and Copilot chat
        transcripts. Chats are persisted in their own local history store and
        are not scientific provenance. Every returned event is reconstructed
        from the project, run and scenario tables so the memory is inspectable
        and exportable.
        """
        with self._connect() as connection:
            project_row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project_row is None:
                return None
            run_rows = connection.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
            scenario_rows = connection.execute(
                "SELECT * FROM scenarios WHERE project_id = ? ORDER BY created_at, updated_at",
                (project_id,),
            ).fetchall()

        project = self._project_from_row(project_row)
        summary = project["summary"]
        source_files = summary.get("source_files", [])
        events: list[dict[str, Any]] = [{
            "event_id": f"import::{project_id}",
            "event_type": "data_import",
            "occurred_at": project["created_at"],
            "title": "Project data imported and mapped",
            "detail": (
                f"{int(summary.get('observation_count') or 0):,} observations and "
                f"{int(summary.get('mapped_entity_count') or summary.get('entity_count') or 0):,} "
                "mapped entities were stored with confirmed field mappings."
            ),
            "facts": {
                "source_files": len(source_files),
                "hazard": project["hazard_type"],
                "time_start": summary.get("time_start"),
                "time_end": summary.get("time_end"),
            },
            "provenance": {
                "mapping": summary.get("mapping", project.get("mapping", {})),
                "units": {key: value.get("unit") for key, value in summary.get("metrics", {}).items() if value.get("available")},
                "sources": [
                    {"name": item.get("name"), "sha256": item.get("sha256"), "encoding": item.get("encoding"), "delimiter": item.get("delimiter")}
                    for item in source_files
                ],
            },
        }]

        for row in run_rows:
            request = json.loads(row["request_json"])
            result = json.loads(row["result_json"])
            items = result.get("items") or []
            model = result.get("model") or {}
            events.append({
                "event_id": row["run_id"],
                "event_type": "analysis",
                "occurred_at": row["created_at"],
                "title": model.get("name") or row["analysis_type"].replace("_", " ").title(),
                "detail": result.get("conclusion") or "A deterministic analysis was completed and saved.",
                "facts": {
                    "analysis_type": row["analysis_type"],
                    "model": model.get("public_id") or str(model.get("id") or "").replace("P5-M", "TWM-METHOD-"),
                    "tier": model.get("tier") or result.get("evidence", {}).get("assignment", {}).get("tier"),
                    "requested": request.get("top_k"),
                    "returned": len(items),
                    "status": result.get("status"),
                },
                "provenance": {
                    "request": request,
                    "verification": result.get("verification"),
                    "evidence_boundary": result.get("data_boundary"),
                    "run_id": row["run_id"],
                },
            })

        for row in scenario_rows:
            result = json.loads(row["result_json"])
            changes = json.loads(row["changes_json"])
            events.append({
                "event_id": row["scenario_id"],
                "event_type": "scenario",
                "occurred_at": row["updated_at"],
                "title": row["name"],
                "detail": result.get("plain_finding") or "A deterministic disruption scenario was saved.",
                "facts": {
                    "hazard": row["hazard_type"],
                    "initial_changes": len(changes),
                    "indirect_links": result.get("indirect_impact_count", 0),
                    "base_time": result.get("timestamp_label") or row["base_timestamp"],
                    "status": result.get("status"),
                },
                "provenance": {
                    "changes": changes,
                    "model": result.get("model"),
                    "verification": result.get("verification"),
                    "evidence_boundary": result.get("data_boundary"),
                },
            })

        events.sort(key=lambda item: (str(item.get("occurred_at") or ""), item["event_id"]))
        return {
            "project_id": project_id,
            "project_name": project["name"],
            "generated_at": _now(),
            "overview": {
                "source_file_count": len(source_files),
                "analysis_count": len(run_rows),
                "scenario_count": len(scenario_rows),
                "event_count": len(events),
            },
            "events": events,
            "assistant_grounding": {
                "scope": "The local assistant and connected LLM Copilot receive current project summary, selection, time, metric and saved-scenario count for each request.",
                "chat_persistence": "Chat text is session-only and is not added to project memory.",
                "api_key_persistence": "Provider API keys are never stored in project memory or project files.",
            },
        }
