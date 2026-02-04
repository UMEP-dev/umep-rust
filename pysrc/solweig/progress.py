"""
Progress reporting abstraction for SOLWEIG.

Automatically uses the appropriate progress mechanism:
- QGIS: QgsProcessingFeedback (native progress bar)
- Python: tqdm (terminal progress bar)
- Fallback: no-op (silent iteration)

Usage:
    from solweig.progress import get_progress_iterator, ProgressReporter

    # Simple iteration with progress
    for item in get_progress_iterator(items, desc="Processing"):
        process(item)

    # Manual progress control
    progress = ProgressReporter(total=100, desc="Computing")
    for i in range(100):
        do_work(i)
        progress.update(1)
    progress.close()
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Detect environment once at module load
_QGIS_AVAILABLE = False
_TQDM_AVAILABLE = False
_qgis_feedback_class = None

# Check for QGIS
try:
    if "qgis" in sys.modules or "qgis.core" in sys.modules:
        from qgis.core import QgsProcessingFeedback

        _QGIS_AVAILABLE = True
        _qgis_feedback_class = QgsProcessingFeedback
        logger.debug("QGIS environment detected, will use QgsProcessingFeedback")
except ImportError:
    pass

# Check for tqdm
_tqdm: type | None = None
try:
    from tqdm import tqdm as _tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    pass


class ProgressReporter:
    """
    Unified progress reporter that works in QGIS, terminal, or silently.

    Args:
        total: Total number of steps (required for percentage calculation).
        desc: Description shown in progress bar.
        feedback: Optional QGIS QgsProcessingFeedback object. If provided,
                  uses QGIS progress. Otherwise auto-detects environment.
        disable: If True, disable all progress output.
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        feedback: Any = None,
        disable: bool = False,
    ):
        self.total = total
        self.desc = desc
        self.current = 0
        self.disable = disable
        self._closed = False

        # Determine which backend to use
        self._qgis_feedback = None
        self._tqdm_bar = None

        if disable:
            return

        # If explicit QGIS feedback provided, use it
        if feedback is not None:
            self._qgis_feedback = feedback
            if self.desc:
                self._qgis_feedback.pushInfo(f"Starting: {self.desc}")
            return

        # Auto-detect: prefer QGIS if available in environment
        if _QGIS_AVAILABLE and "qgis.core" in sys.modules:
            # In QGIS but no feedback provided - log only, no progress bar
            logger.debug(f"QGIS detected but no feedback provided for: {desc}")
            return

        # Use tqdm if available
        if _tqdm is not None:
            self._tqdm_bar = _tqdm(total=total, desc=desc)
            return

        # Fallback: silent operation
        logger.debug(f"No progress backend available for: {desc}")

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        if self._closed:
            return

        self.current += n

        if self.disable:
            return

        if self._qgis_feedback is not None:
            # QGIS expects percentage 0-100
            percent = min(100, int(100 * self.current / self.total)) if self.total > 0 else 0
            self._qgis_feedback.setProgress(percent)
        elif self._tqdm_bar is not None:
            self._tqdm_bar.update(n)

    def set_description(self, desc: str) -> None:
        """Update the progress description."""
        self.desc = desc
        if self._qgis_feedback is not None:
            self._qgis_feedback.pushInfo(desc)
        elif self._tqdm_bar is not None:
            self._tqdm_bar.set_description(desc)

    def is_cancelled(self) -> bool:
        """Check if user requested cancellation (QGIS only)."""
        if self._qgis_feedback is not None:
            return self._qgis_feedback.isCanceled()
        return False

    def close(self) -> None:
        """Close the progress bar."""
        if self._closed:
            return
        self._closed = True

        if self._tqdm_bar is not None:
            self._tqdm_bar.close()


class _ProgressIterator(Iterator[T]):
    """Iterator wrapper that reports progress."""

    def __init__(self, iterable: Iterable[T], reporter: ProgressReporter):
        self._iterator = iter(iterable)
        self._reporter = reporter

    def __iter__(self) -> _ProgressIterator[T]:
        return self

    def __next__(self) -> T:
        try:
            item = next(self._iterator)
            self._reporter.update(1)
            return item
        except StopIteration:
            self._reporter.close()
            raise


def get_progress_iterator(
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    feedback: Any = None,
    disable: bool = False,
) -> Iterator[T]:
    """
    Wrap an iterable with automatic progress reporting.

    Automatically uses the appropriate progress mechanism:
    - QGIS environment with feedback: QgsProcessingFeedback
    - Terminal: tqdm progress bar
    - Fallback: silent iteration

    Args:
        iterable: The iterable to wrap.
        desc: Description for the progress bar.
        total: Total number of items (computed from len() if not provided).
        feedback: Optional QGIS QgsProcessingFeedback for progress reporting.
        disable: If True, disable progress output entirely.

    Returns:
        Iterator that reports progress as items are consumed.

    Example:
        # Simple usage
        for item in get_progress_iterator(items, desc="Processing"):
            process(item)

        # With QGIS feedback (in processing algorithm)
        for item in get_progress_iterator(items, feedback=self.feedback):
            process(item)
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            # Iterable doesn't have len(), estimate or use 0
            total = 0

    reporter = ProgressReporter(total=total, desc=desc, feedback=feedback, disable=disable)
    return _ProgressIterator(iterable, reporter)


# Convenience function that matches tqdm signature for easy migration
def progress(
    iterable: Iterable[T],
    desc: str = "",
    total: int | None = None,
    **kwargs,
) -> Iterator[T]:
    """
    Drop-in replacement for tqdm that auto-detects environment.

    This function has a similar signature to tqdm for easy migration.
    Additional kwargs are ignored for compatibility.

    Example:
        # Replace: for item in tqdm(items, desc="Processing"):
        # With:    for item in progress(items, desc="Processing"):
    """
    return get_progress_iterator(iterable, desc=desc, total=total)
