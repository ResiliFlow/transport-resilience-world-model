import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from infrastructure_perception.backend.python_runtime import configure_perception_python, select_existing_python


class PerceptionPythonTests(unittest.TestCase):
    def test_current_interpreter_is_usable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_existing_python(Path(sys.executable)), Path(sys.executable))

    def test_missing_default_uses_current_interpreter(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_existing_python(Path(folder) / 'missing'), Path(sys.executable))

    def test_invalid_explicit_interpreter_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'TWM_PERCEPTION_PYTHON': str(Path(folder) / 'missing')}):
                with self.assertRaises(FileNotFoundError):
                    select_existing_python(Path(sys.executable))

    def test_stage6_override_is_preserved(self):
        runtime = ModuleType('test_runtime')
        runtime.PYTHON = Path(sys.executable)
        with patch.dict(os.environ, {'TWM_STAGE6_PYTHON': 'independent-runtime'}, clear=True):
            configure_perception_python(runtime)
            self.assertEqual(os.environ['TWM_STAGE6_PYTHON'], 'independent-runtime')
