"""
SOLWEIG QGIS Plugin

Provides QGIS Processing algorithms for calculating Mean Radiant Temperature (Tmrt),
UTCI, and PET thermal comfort indices using the SOLWEIG model.

Adapted from UMEP (Urban Multi-scale Environmental Predictor).
Original code by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

Citation:
    Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L,
    Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F,
    Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor
    (UMEP) - An integrated tool for city-based climate services.
    Environmental Modelling and Software 99, 70-87
    https://doi.org/10.1016/j.envsoft.2017.09.020
"""

from __future__ import annotations

import importlib.util
import platform as _platform
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup bundled library path
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).parent
_BUNDLED_DIR = _PLUGIN_DIR / "_bundled"
_NATIVE_DIR = _PLUGIN_DIR / "_native"
_SOLWEIG_AVAILABLE = False
_SOLWEIG_SOURCE = None  # "bundled", "system", or None
_SOLWEIG_IMPORT_ERROR = None


def _get_platform_tag() -> str:
    """Detect current platform and return tag matching _native/ directory names."""
    system = _platform.system().lower()
    machine = _platform.machine().lower()

    if system == "darwin":
        system_tag = "darwin"
    elif system == "linux":
        system_tag = "linux"
    elif system == "windows":
        system_tag = "windows"
    else:
        raise RuntimeError(f"Unsupported operating system: {_platform.system()}")

    if machine in ("x86_64", "amd64", "x64"):
        arch_tag = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch_tag = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    return f"{system_tag}_{arch_tag}"


def _inject_platform_rustalgos() -> bool:
    """
    Pre-load the correct platform-specific rustalgos binary into sys.modules.

    When the solweig package later does ``from .rustalgos import ...``,
    Python finds the already-loaded module. No file copying or symlinks needed.

    Returns True on success, False on failure (error stored in _SOLWEIG_IMPORT_ERROR).
    """
    global _SOLWEIG_IMPORT_ERROR

    # Skip if rustalgos is already loaded (e.g. system install)
    if "solweig.rustalgos" in sys.modules:
        return True

    try:
        tag = _get_platform_tag()
    except RuntimeError as e:
        _SOLWEIG_IMPORT_ERROR = str(e)
        return False

    ext = ".pyd" if tag.startswith("windows") else ".so"
    binary_path = _NATIVE_DIR / tag / f"rustalgos.abi3{ext}"

    if not binary_path.exists():
        # No _native/ directory — fall back to legacy single-platform layout
        # where rustalgos lives directly in _bundled/solweig/
        return True

    try:
        spec = importlib.util.spec_from_file_location(
            "solweig.rustalgos",
            str(binary_path),
            submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            _SOLWEIG_IMPORT_ERROR = f"Failed to create module spec for {binary_path}"
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules["solweig.rustalgos"] = module
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        _SOLWEIG_IMPORT_ERROR = f"Failed to load {binary_path}: {type(e).__name__}: {e}"
        sys.modules.pop("solweig.rustalgos", None)
        return False


def _setup_solweig_path():
    """
    Set up the import path for solweig library.

    Priority:
    1. Bundled library (_bundled/ directory in plugin)
    2. System-installed solweig (via pip)
    3. Development path (for local development)
    """
    global _SOLWEIG_AVAILABLE, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR

    # Option 1: Try bundled library first
    # Structure: _bundled/solweig/__init__.py (so _bundled goes in sys.path)
    bundled_pkg = _BUNDLED_DIR / "solweig"

    if bundled_pkg.exists() and (bundled_pkg / "__init__.py").exists():
        # Pre-inject platform-specific rustalgos before importing solweig
        if not _inject_platform_rustalgos():
            pass  # Injection failed — error recorded, fall through to system/dev
        else:
            if str(_BUNDLED_DIR) not in sys.path:
                sys.path.insert(0, str(_BUNDLED_DIR))
            try:
                import solweig  # noqa: F401

                _SOLWEIG_AVAILABLE = True
                _SOLWEIG_SOURCE = "bundled"
                _SOLWEIG_IMPORT_ERROR = None
                return
            except Exception as e:
                # Bundled exists but import failed - capture the error
                _SOLWEIG_IMPORT_ERROR = f"bundled import failed: {type(e).__name__}: {e}"
                # Remove from path
                if str(_BUNDLED_DIR) in sys.path:
                    sys.path.remove(str(_BUNDLED_DIR))
                # Clean up injected module
                sys.modules.pop("solweig.rustalgos", None)

    # Option 2: Try system-installed solweig
    try:
        import solweig  # noqa: F401

        _SOLWEIG_AVAILABLE = True
        _SOLWEIG_SOURCE = "system"
        _SOLWEIG_IMPORT_ERROR = None
        return
    except ImportError:
        pass

    # Option 3: Development mode - look for pysrc in parent directories
    dev_paths = [
        _PLUGIN_DIR.parent.parent / "pysrc",  # qgis_plugin/solweig -> pysrc
        _PLUGIN_DIR.parent.parent.parent / "pysrc",  # One more level up
    ]
    for dev_path in dev_paths:
        if dev_path.exists() and (dev_path / "solweig").exists():
            if str(dev_path) not in sys.path:
                sys.path.insert(0, str(dev_path))
            try:
                import solweig  # noqa: F401

                _SOLWEIG_AVAILABLE = True
                _SOLWEIG_SOURCE = "development"
                _SOLWEIG_IMPORT_ERROR = None
                return
            except ImportError:
                _SOLWEIG_IMPORT_ERROR = "development import failed"
                if str(dev_path) in sys.path:
                    sys.path.remove(str(dev_path))

    # No solweig found
    _SOLWEIG_AVAILABLE = False
    _SOLWEIG_SOURCE = None
    if _SOLWEIG_IMPORT_ERROR is None:
        _SOLWEIG_IMPORT_ERROR = "solweig import not attempted or failed silently"


# Run setup on module load
_setup_solweig_path()


def check_dependencies() -> tuple[bool, str]:
    """
    Check if all required dependencies are available.

    Returns:
        Tuple of (success, message)
    """
    if _SOLWEIG_AVAILABLE:
        return True, f"SOLWEIG library loaded ({_SOLWEIG_SOURCE})"

    # Build helpful error message
    error_hint = f"\nLast import error: {_SOLWEIG_IMPORT_ERROR}\n" if _SOLWEIG_IMPORT_ERROR else ""

    msg = f"""SOLWEIG library not found.{error_hint}

To install SOLWEIG:

Option 1 - Using pip (recommended):
  In OSGeo4W Shell (Windows) or Terminal (macOS/Linux):
  pip install solweig

Option 2 - In QGIS Python Console:
  import subprocess
  subprocess.check_call(['pip', 'install', 'solweig'])

Option 3 - From source:
  cd /path/to/solweig
  maturin develop

After installation, restart QGIS and re-enable the plugin.
"""
    return False, msg


def show_dependency_warning():
    """Show a warning dialog if dependencies are missing."""
    success, message = check_dependencies()
    if not success:
        try:
            from qgis.PyQt.QtWidgets import QMessageBox

            QMessageBox.warning(
                None,
                "SOLWEIG Plugin - Missing Dependencies",
                message,
            )
        except ImportError:
            # Not in QGIS environment, just print
            print(f"WARNING: {message}")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

from .provider import SolweigProvider  # noqa: E402


def classFactory(iface):
    """
    QGIS plugin entry point.

    Called by QGIS when the plugin is loaded. Returns the provider instance
    that will register all processing algorithms.

    Args:
        iface: QgisInterface instance providing access to QGIS components.

    Returns:
        SolweigPlugin instance that manages the processing provider.
    """
    return SolweigPlugin(iface)


class SolweigPlugin:
    """
    Main plugin class that manages the SOLWEIG processing provider.

    This class handles plugin lifecycle (load/unload) and registers
    the SolweigProvider with QGIS Processing framework.
    """

    def __init__(self, iface):
        """
        Initialize the plugin.

        Args:
            iface: QgisInterface instance.
        """
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Initialize the processing provider."""
        from qgis.core import QgsApplication

        self.provider = SolweigProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize the plugin GUI (called when plugin is activated)."""
        # Check dependencies and show warning if missing
        if not _SOLWEIG_AVAILABLE:
            show_dependency_warning()

        self.initProcessing()

    def unload(self):
        """Unload the plugin (called when plugin is deactivated)."""
        from qgis.core import QgsApplication

        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)


# ---------------------------------------------------------------------------
# Module-level info for debugging
# ---------------------------------------------------------------------------

__solweig_available__ = _SOLWEIG_AVAILABLE
__solweig_source__ = _SOLWEIG_SOURCE
