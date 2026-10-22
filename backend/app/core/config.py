from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
MODEL_LIBRARY_ROOT = PROJECT_ROOT / "model_library"
DATA_ROOT = PROJECT_ROOT / "data"
STORAGE_ROOT = PROJECT_ROOT / "storage"
UPLOAD_ROOT = STORAGE_ROOT / "uploads"
PROJECT_DATA_ROOT = STORAGE_ROOT / "projects"
LOG_ROOT = STORAGE_ROOT / "logs"
DATABASE_PATH = STORAGE_ROOT / "prototype5.db"
VERSION_PATH = PROJECT_ROOT / "VERSION"
DEMO_CSV_PATH = DATA_ROOT / "demo" / "synthetic_transport_sample.csv"


def ensure_directories() -> None:
    for path in (STORAGE_ROOT, UPLOAD_ROOT, PROJECT_DATA_ROOT, LOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def version() -> str:
    return VERSION_PATH.read_text(encoding="utf-8").strip()


ensure_directories()
