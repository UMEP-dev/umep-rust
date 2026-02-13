"""
QGIS-compatible logging for SOLWEIG.

Provides automatic environment detection and uses appropriate logging backend:
- QGIS: QgsProcessingFeedback.pushInfo() / pushDebugInfo()
- Python: Standard logging module
- Fallback: Print to stdout

Usage:
    from solweig.solweig_logging import get_logger

    logger = get_logger(__name__)
    logger.info("Surface data loaded: 400×400 pixels")
    logger.debug(f"Using {len(weather_list)} timesteps")
    logger.warning("SVF not provided, will compute on-the-fly (slow)")
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import Any


class LogLevel(IntEnum):
    """Log levels matching Python logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class SolweigLogger:
    """
    Unified logger that works in both QGIS and Python environments.

    Auto-detects environment and uses appropriate backend:
    - QGIS: Uses QgsProcessingFeedback if available
    - Python: Uses standard logging module
    - Fallback: Prints to stdout
    """

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        """
        Initialize logger.

        Args:
            name: Logger name (usually module name)
            level: Minimum log level to display
        """
        self.name = name
        self.level = level
        self._feedback = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect which logging backend to use."""
        # Check if running in QGIS
        try:
            from qgis.core import QgsProcessingFeedback  # noqa: F401

            # QGIS is available, but we need a feedback object to be set
            # This will be set via set_feedback() when running as QGIS processing algorithm
            return "qgis"
        except ImportError:
            pass

        # Use standard Python logging
        return "logging"

    def set_feedback(self, feedback: Any) -> None:
        """
        Set QGIS feedback object for logging.

        Args:
            feedback: QgsProcessingFeedback object
        """
        self._feedback = feedback

    def _log(self, level: LogLevel, message: str) -> None:
        """Internal logging method."""
        if level < self.level:
            return  # Below minimum level

        if self._backend == "qgis" and self._feedback is not None:
            # Use QGIS feedback
            if level >= LogLevel.ERROR:
                self._feedback.reportError(message)
            elif level >= LogLevel.WARNING:
                self._feedback.pushInfo(f"WARNING: {message}")
            elif level >= LogLevel.INFO:
                self._feedback.pushInfo(message)
            else:  # DEBUG
                self._feedback.pushDebugInfo(message)
        elif self._backend in ("logging", "qgis"):
            # Use Python logging (also used as fallback when QGIS backend
            # is detected but no feedback object has been set)
            logger = logging.getLogger(self.name)
            logger.log(level, message)
        else:
            # Fallback: print to stdout
            prefix = {
                LogLevel.DEBUG: "DEBUG",
                LogLevel.INFO: "INFO",
                LogLevel.WARNING: "WARNING",
                LogLevel.ERROR: "ERROR",
            }.get(level, "INFO")
            print(f"[{prefix}] {self.name}: {message}", file=sys.stderr if level >= LogLevel.WARNING else sys.stdout)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message)

    def set_level(self, level: LogLevel | int) -> None:
        """Set minimum log level."""
        self.level = LogLevel(level) if isinstance(level, int) else level


# Global logger registry
_loggers: dict[str, SolweigLogger] = {}


def get_logger(name: str, level: LogLevel | int = LogLevel.INFO) -> SolweigLogger:
    """
    Get or create a logger for the given name.

    Args:
        name: Logger name (usually module name or __name__)
        level: Minimum log level (default: INFO)

    Returns:
        SolweigLogger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.debug(f"Grid size: {rows}×{cols}")
    """
    if name not in _loggers:
        _loggers[name] = SolweigLogger(name, LogLevel(level) if isinstance(level, int) else level)
    return _loggers[name]


def set_global_level(level: LogLevel | int) -> None:
    """
    Set log level for all existing loggers.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)

    Example:
        >>> import solweig.solweig_logging as slog
        >>> slog.set_global_level(slog.LogLevel.DEBUG)  # Show debug messages
    """
    level = LogLevel(level) if isinstance(level, int) else level
    for logger in _loggers.values():
        logger.set_level(level)


def set_global_feedback(feedback: Any) -> None:
    """
    Set QGIS feedback object for all loggers.

    Args:
        feedback: QgsProcessingFeedback object
    """
    for logger in _loggers.values():
        logger.set_feedback(feedback)


# Configure Python logging to be less verbose by default
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
    stream=sys.stdout,
)
