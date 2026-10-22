from pathlib import Path
import sqlite3
import tempfile
import unittest

from backend.app.core.storage import Repository


class TemporaryProjectLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="twm_lifecycle_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save_project(self, repository: Repository, project_id: str, retention: str) -> None:
        repository.save_project(
            project_id=project_id,
            name=project_id,
            hazard_type="other",
            source="user_upload",
            status="ready",
            database_path=str(self.root / f"{project_id}.sqlite"),
            mapping={},
            summary={},
            retention=retention,
        )

    def test_only_new_temporary_records_are_cleanup_candidates(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        repository.save_upload(["legacy.csv"], {}, upload_id="legacy", temporary=False)
        repository.save_upload(["temporary.csv"], {}, upload_id="temporary", temporary=True)
        self._save_project(repository, "kept", "persistent")
        self._save_project(repository, "test", "temporary")

        self.assertEqual(
            [item["upload_id"] for item in repository.list_temporary_uploads()],
            ["temporary"],
        )
        self.assertEqual(
            [item["project_id"] for item in repository.list_temporary_projects()],
            ["test"],
        )

    def test_keep_promotes_project_and_prevents_cleanup(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        self._save_project(repository, "test", "temporary")

        self.assertTrue(repository.keep_project("test"))
        self.assertEqual(repository.get_project("test")["retention"], "persistent")
        self.assertEqual(repository.list_temporary_projects(), [])
        self.assertEqual(repository.delete_project("test"), (False, []))

    def test_open_existing_candidates_include_only_persistent_projects(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        self._save_project(repository, "kept", "persistent")
        self._save_project(repository, "session_only", "temporary")

        self.assertEqual(
            [item["project_id"] for item in repository.list_persistent_projects()],
            ["kept"],
        )

    def test_temporary_delete_cascades_metadata_but_never_persistent_project(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        self._save_project(repository, "test", "temporary")
        repository.save_run("test", "critical_components", {}, {"items": []})
        scenario_id = repository.save_scenario(
            project_id="test",
            name="Temporary scenario",
            hazard_type="other",
            base_timestamp=None,
            changes=[],
            result={},
        )

        deleted, scenario_ids = repository.delete_project("test")
        self.assertTrue(deleted)
        self.assertEqual(scenario_ids, [scenario_id])
        self.assertIsNone(repository.get_project("test"))
        self.assertEqual(repository.list_runs(project_id="test"), [])

    def test_explicit_saved_delete_removes_only_the_selected_persistent_project(self) -> None:
        repository = Repository(self.root / "metadata.sqlite")
        repository.initialise()
        self._save_project(repository, "delete_me", "persistent")
        self._save_project(repository, "keep_me", "persistent")
        self._save_project(repository, "session_only", "temporary")
        repository.save_run("delete_me", "critical_components", {}, {"items": []})
        scenario_id = repository.save_scenario(
            project_id="delete_me",
            name="Saved scenario",
            hazard_type="other",
            base_timestamp=None,
            changes=[],
            result={},
        )

        self.assertEqual(repository.delete_persistent_project("session_only"), (False, []))
        deleted, scenario_ids = repository.delete_persistent_project("delete_me")

        self.assertTrue(deleted)
        self.assertEqual(scenario_ids, [scenario_id])
        self.assertIsNone(repository.get_project("delete_me"))
        self.assertIsNotNone(repository.get_project("keep_me"))
        self.assertIsNotNone(repository.get_project("session_only"))
        self.assertEqual(repository.list_runs(project_id="delete_me"), [])

    def test_schema_migration_marks_legacy_rows_persistent(self) -> None:
        database = self.root / "legacy.sqlite"
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE uploads (
                    upload_id TEXT PRIMARY KEY, paths_json TEXT NOT NULL,
                    preview_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE projects (
                    project_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    hazard_type TEXT NOT NULL, source TEXT NOT NULL,
                    status TEXT NOT NULL, database_path TEXT NOT NULL,
                    mapping_json TEXT NOT NULL, summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO uploads VALUES ('old_upload', '[]', '{}', '2020-01-01T00:00:00+00:00');
                INSERT INTO projects VALUES (
                    'old_project', 'Old', 'other', 'legacy', 'ready', 'old.sqlite',
                    '{}', '{}', '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00'
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        repository = Repository(database)
        repository.initialise()
        self.assertFalse(repository.get_upload("old_upload")["is_temporary"])
        self.assertEqual(repository.get_project("old_project")["retention"], "persistent")
        self.assertEqual(repository.list_temporary_uploads(), [])
        self.assertEqual(repository.list_temporary_projects(), [])


if __name__ == "__main__":
    unittest.main()
