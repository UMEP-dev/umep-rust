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

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).parent
_SOLWEIG_AVAILABLE = False
_SOLWEIG_SOURCE = None  # "system", "development", or None
_SOLWEIG_IMPORT_ERROR = None


def _setup_solweig_path():
    """
    Set up the import path for solweig library.

    Priority:
    1. System-installed solweig (via pip)
    2. Development path (for local development)
    """
    global _SOLWEIG_AVAILABLE, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR

    # Already found in a previous call
    if _SOLWEIG_AVAILABLE:
        return

    # Option 1: Try system-installed solweig
    try:
        import solweig  # noqa: F401

        _SOLWEIG_AVAILABLE = True
        _SOLWEIG_SOURCE = "system"
        _SOLWEIG_IMPORT_ERROR = None
        return
    except ImportError:
        pass

    # Option 2: Development mode - look for pysrc in parent directories
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
        _SOLWEIG_IMPORT_ERROR = "solweig package not installed"


def _install_solweig() -> tuple[bool, str]:
    """
    Install solweig via pip in-process.

    Uses pip's internal API rather than subprocess because QGIS embeds Python
    and sys.executable points to the QGIS binary, not a usable Python interpreter.
    See: https://github.com/qgis/QGIS/issues/45646

    Returns:
        Tuple of (success, message).
    """
    import contextlib
    import io

    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        return False, "pip is not available in this QGIS Python environment."

    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exit_code = pip_main(["install", "solweig"])
        if exit_code == 0:
            return True, "SOLWEIG installed successfully."
        return False, f"pip install failed (exit code {exit_code}):\n{output.getvalue()}"
    except Exception as e:
        return False, f"Installation failed: {e}"


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

    error_hint = f"\nLast import error: {_SOLWEIG_IMPORT_ERROR}\n" if _SOLWEIG_IMPORT_ERROR else ""

    msg = f"""SOLWEIG library not found.{error_hint}

To install SOLWEIG manually:

  In OSGeo4W Shell (Windows) or Terminal (macOS/Linux):
  pip install solweig

After installation, restart QGIS and re-enable the plugin.
"""
    return False, msg


def _prompt_install():
    """Offer to auto-install solweig if it's missing."""
    global _SOLWEIG_AVAILABLE, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR

    success, message = check_dependencies()
    if success:
        return

    try:
        from qgis.PyQt.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            None,
            "SOLWEIG Plugin - Install Dependencies",
            "The SOLWEIG library is required but not installed.\n\n"
            "Would you like to install it now?\n\n"
            "This will run:  pip install solweig",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply != QMessageBox.Yes:
            QMessageBox.information(
                None,
                "SOLWEIG Plugin",
                "SOLWEIG was not installed. You can install it manually:\n\n"
                "  pip install solweig\n\n"
                "Then restart QGIS.",
            )
            return

        # Show a wait cursor while installing
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, install_msg = _install_solweig()
        finally:
            QApplication.restoreOverrideCursor()

        if ok:
            # Try importing now that it's installed
            _setup_solweig_path()
            if _SOLWEIG_AVAILABLE:
                QMessageBox.information(
                    None,
                    "SOLWEIG Plugin",
                    "SOLWEIG installed successfully! The plugin is ready to use.",
                )
            else:
                QMessageBox.information(
                    None,
                    "SOLWEIG Plugin",
                    "SOLWEIG installed successfully.\n\nPlease restart QGIS to complete setup.",
                )
        else:
            QMessageBox.warning(
                None,
                "SOLWEIG Plugin - Installation Failed",
                f"{install_msg}\n\nYou can try installing manually:\n\n  pip install solweig\n\nThen restart QGIS.",
            )

    except ImportError:
        # Not in QGIS environment
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
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Initialize the processing provider."""
        from qgis.core import QgsApplication

        self.provider = SolweigProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize the plugin GUI (called when plugin is activated)."""
        if not _SOLWEIG_AVAILABLE:
            _prompt_install()

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
