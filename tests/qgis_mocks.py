"""
Shared QGIS mock setup for testing plugin code without a QGIS installation.

Import this module BEFORE any qgis_plugin imports to inject mocks into sys.modules.
All mock classes and stubs are defined here to ensure consistency across test files.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Real exception/base classes (needed for isinstance/pytest.raises)
# ---------------------------------------------------------------------------

QgsProcessingException = type("QgsProcessingException", (Exception,), {})


class QgsProcessingAlgorithm:
    """Stub for QgsProcessingAlgorithm - must be a real class for subclassing."""

    def parameterAsRasterLayer(self, *a, **kw):
        return None

    def parameterAsOutputLayer(self, *a, **kw):
        return None

    def addParameter(self, *a, **kw):
        pass


class QgsRasterLayer:
    """Stub for QgsRasterLayer."""

    def __init__(self, *a, **kw):
        pass

    def isValid(self):
        return True

    def source(self):
        return ""

    def dataProvider(self):
        return MagicMock()


class QgsProject:
    """Stub for QgsProject with singleton pattern."""

    _inst = MagicMock()

    @classmethod
    def instance(cls):
        return cls._inst


class QgsProcessingProvider:
    """Stub for QgsProcessingProvider - must be a real class for subclassing."""

    def addAlgorithm(self, *a, **kw):
        pass

    def icon(self):
        return None


# ---------------------------------------------------------------------------
# Build mock modules
# ---------------------------------------------------------------------------

_mock_qgis_core = MagicMock()
_mock_qgis_core.QgsProcessingException = QgsProcessingException
_mock_qgis_core.QgsProcessingAlgorithm = QgsProcessingAlgorithm
_mock_qgis_core.QgsProcessingProvider = QgsProcessingProvider
_mock_qgis_core.QgsRasterLayer = QgsRasterLayer
_mock_qgis_core.QgsProject = QgsProject
_mock_qgis_core.QgsProcessingContext = MagicMock
_mock_qgis_core.QgsProcessingFeedback = MagicMock
_mock_qgis_core.QgsApplication = MagicMock()

_mock_qgis_pyqt_core = MagicMock()
_mock_qgis_pyqt_gui = MagicMock()
_mock_qgis_pyqt_widgets = MagicMock()
_mock_qgis_pyqt = MagicMock()
_mock_qgis_pyqt.QtCore = _mock_qgis_pyqt_core
_mock_qgis_pyqt.QtGui = _mock_qgis_pyqt_gui
_mock_qgis_pyqt.QtWidgets = _mock_qgis_pyqt_widgets

_mock_qgis = MagicMock()
_mock_qgis.core = _mock_qgis_core
_mock_qgis.PyQt = _mock_qgis_pyqt

_mock_osgeo = MagicMock()


def _has_real_osgeo() -> bool:
    """Check if a real (non-mock) osgeo package is available."""
    if "osgeo" in sys.modules:
        mod = sys.modules["osgeo"]
        return not isinstance(mod, MagicMock) and hasattr(mod, "__file__")
    try:
        import importlib.util

        return importlib.util.find_spec("osgeo") is not None
    except (ImportError, ValueError):
        return False


def install():
    """Install QGIS/GDAL mocks into sys.modules. Idempotent - safe to call multiple times."""
    # Always force-set qgis modules (never real outside QGIS)
    qgis_mocks = {
        "qgis": _mock_qgis,
        "qgis.core": _mock_qgis_core,
        "qgis.PyQt": _mock_qgis_pyqt,
        "qgis.PyQt.QtCore": _mock_qgis_pyqt_core,
        "qgis.PyQt.QtGui": _mock_qgis_pyqt_gui,
        "qgis.PyQt.QtWidgets": _mock_qgis_pyqt_widgets,
    }
    for name, mock in qgis_mocks.items():
        sys.modules[name] = mock

    # Only mock osgeo if not actually installed (avoid polluting real GDAL for other tests)
    if not _has_real_osgeo():
        osgeo_mocks = {
            "osgeo": _mock_osgeo,
            "osgeo.gdal": _mock_osgeo.gdal,
            "osgeo.osr": _mock_osgeo.osr,
        }
        for name, mock in osgeo_mocks.items():
            sys.modules.setdefault(name, mock)


def uninstall_osgeo():
    """Remove osgeo mocks from sys.modules to avoid polluting other tests."""
    for name in ("osgeo", "osgeo.gdal", "osgeo.osr"):
        if name in sys.modules and isinstance(sys.modules[name], MagicMock):
            del sys.modules[name]
