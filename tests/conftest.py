"""Shared pytest configuration and path setup."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that both `tests.qgis_mocks`
# and `qgis_plugin.*` imports work regardless of how pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
