"""Select an explicit detector interpreter or the current development environment."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType


def select_existing_python(primary: Path) -> Path:
    configured = os.environ.get("TWM_PERCEPTION_PYTHON")
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            raise FileNotFoundError("TWM_PERCEPTION_PYTHON does not identify an existing interpreter.")
        return candidate
    return primary if primary.is_file() else Path(sys.executable)


def configure_perception_python(runtime: ModuleType) -> None:
    runtime.PYTHON = select_existing_python(runtime.PYTHON)
    os.environ.setdefault("TWM_STAGE6_PYTHON", str(runtime.PYTHON))
