"""
Tests for progress reporting abstraction.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from solweig.progress import ProgressReporter, get_progress_iterator, progress


class TestProgressReporter:
    """Test the ProgressReporter class."""

    def test_progress_reporter_basic_usage(self):
        """Test basic ProgressReporter usage."""
        reporter = ProgressReporter(total=10, desc="Test", disable=True)

        for i in range(10):
            reporter.update(1)

        reporter.close()

        assert reporter.current == 10

    def test_progress_reporter_update_increments(self):
        """Test that update() increments current count."""
        reporter = ProgressReporter(total=100, disable=True)

        reporter.update(5)
        assert reporter.current == 5

        reporter.update(10)
        assert reporter.current == 15

        reporter.close()

    def test_progress_reporter_disabled_mode(self):
        """Test that disabled mode doesn't crash."""
        reporter = ProgressReporter(total=10, disable=True)

        # Should work without any backend
        reporter.update(5)
        reporter.set_description("New description")
        reporter.close()

        assert reporter.current == 5

    def test_progress_reporter_no_backend(self):
        """Test ProgressReporter with no tqdm or QGIS available."""
        # Hide both tqdm and QGIS
        with patch.dict(sys.modules, {"tqdm": None, "qgis.core": None}):
            reporter = ProgressReporter(total=10, desc="Test")

            # Should work silently
            reporter.update(5)
            reporter.close()

            assert reporter.current == 5

    def test_progress_reporter_with_tqdm(self):
        """Test ProgressReporter with tqdm backend."""
        try:
            from tqdm import tqdm

            reporter = ProgressReporter(total=10, desc="Test with tqdm")

            # Should have tqdm bar
            assert reporter._tqdm_bar is not None

            reporter.update(5)
            reporter.close()

            assert reporter.current == 5

        except ImportError:
            pytest.skip("tqdm not available")

    def test_progress_reporter_qgis_feedback(self):
        """Test ProgressReporter with QGIS feedback."""
        # Mock QGIS feedback
        mock_feedback = MagicMock()
        mock_feedback.isCanceled.return_value = False

        reporter = ProgressReporter(total=100, desc="Test", feedback=mock_feedback)

        # Should use QGIS feedback
        assert reporter._qgis_feedback is mock_feedback

        reporter.update(50)

        # Should have called setProgress with 50%
        mock_feedback.setProgress.assert_called_with(50)

        reporter.close()

    def test_progress_reporter_qgis_cancel(self):
        """Test that QGIS cancellation is detected."""
        mock_feedback = MagicMock()
        mock_feedback.isCanceled.return_value = True

        reporter = ProgressReporter(total=100, feedback=mock_feedback)

        assert reporter.is_cancelled() is True

    def test_progress_reporter_set_description(self):
        """Test updating progress description."""
        reporter = ProgressReporter(total=10, desc="Initial", disable=True)

        reporter.set_description("Updated")

        assert reporter.desc == "Updated"

        reporter.close()

    def test_progress_reporter_close_idempotent(self):
        """Test that close() can be called multiple times."""
        reporter = ProgressReporter(total=10, disable=True)

        reporter.close()
        reporter.close()  # Should not crash

        assert reporter._closed

    def test_progress_reporter_update_after_close(self):
        """Test that update after close is ignored."""
        reporter = ProgressReporter(total=10, disable=True)

        reporter.update(5)
        reporter.close()
        reporter.update(5)  # Should be ignored

        # Should still be 5, not 10
        assert reporter.current == 5


class TestProgressIterator:
    """Test the get_progress_iterator function."""

    def test_progress_iterator_basic(self):
        """Test basic progress iterator usage."""
        items = range(10)
        count = 0

        for item in get_progress_iterator(items, desc="Test", disable=True):
            count += 1

        assert count == 10

    def test_progress_iterator_with_list(self):
        """Test progress iterator with list."""
        items = [1, 2, 3, 4, 5]
        result = []

        for item in get_progress_iterator(items, desc="Test", disable=True):
            result.append(item)

        assert result == items

    def test_progress_iterator_total_from_len(self):
        """Test that total is computed from len() if not provided."""
        items = [1, 2, 3, 4, 5]

        iterator = get_progress_iterator(items, desc="Test", disable=True)
        # Access the reporter through the iterator
        assert iterator._reporter.total == 5

    def test_progress_iterator_explicit_total(self):
        """Test providing explicit total."""
        items = range(10)

        iterator = get_progress_iterator(items, desc="Test", total=100, disable=True)
        assert iterator._reporter.total == 100

    def test_progress_iterator_generator(self):
        """Test progress iterator with generator."""

        def gen():
            for i in range(5):
                yield i

        result = []
        for item in get_progress_iterator(gen(), desc="Test", total=5, disable=True):
            result.append(item)

        assert result == [0, 1, 2, 3, 4]

    def test_progress_iterator_stops_at_end(self):
        """Test that iterator properly stops at end."""
        items = [1, 2, 3]
        result = []

        for item in get_progress_iterator(items, desc="Test", disable=True):
            result.append(item)

        assert len(result) == 3
        assert result == items


class TestProgressFunction:
    """Test the progress() convenience function."""

    def test_progress_function_drop_in_replacement(self):
        """Test that progress() works like tqdm."""
        items = range(10)
        count = 0

        for item in progress(items, desc="Test", disable=True):
            count += 1

        assert count == 10

    def test_progress_function_kwargs_ignored(self):
        """Test that extra kwargs are ignored (for tqdm compatibility)."""
        items = range(5)
        result = []

        # These kwargs are tqdm-specific and should be ignored
        for item in progress(items, desc="Test", disable=True, leave=False, ncols=80):
            result.append(item)

        assert result == list(items)


class TestProgressQGISIntegration:
    """Test QGIS integration scenarios."""

    def test_qgis_environment_detection(self):
        """Test that QGIS environment is detected."""
        # Mock QGIS being available
        mock_qgis = MagicMock()

        with patch.dict("sys.modules", {"qgis.core": mock_qgis}):
            # Reimport to trigger detection
            from importlib import reload

            import solweig.progress

            reload(solweig.progress)

            # Should detect QGIS
            assert solweig.progress._QGIS_AVAILABLE

    def test_tqdm_environment_detection(self):
        """Test that tqdm environment is detected."""
        try:
            from tqdm import tqdm

            # Reimport to trigger detection
            from importlib import reload

            import solweig.progress

            reload(solweig.progress)

            # Should detect tqdm
            assert solweig.progress._TQDM_AVAILABLE

        except ImportError:
            pytest.skip("tqdm not available")

    def test_progress_without_dependencies(self):
        """Test that progress works without tqdm or QGIS."""
        # Hide both dependencies
        with patch.dict(sys.modules, {"tqdm": None, "qgis.core": None}):
            # Should still work, just silently
            items = range(5)
            result = []

            for item in get_progress_iterator(items):
                result.append(item)

            assert result == list(items)


class TestProgressEdgeCases:
    """Test edge cases and error conditions."""

    def test_progress_zero_total(self):
        """Test progress with zero total."""
        reporter = ProgressReporter(total=0, disable=True)

        reporter.update(1)
        reporter.close()

        # Should handle gracefully
        assert reporter.current == 1

    def test_progress_negative_update(self):
        """Test negative update values."""
        reporter = ProgressReporter(total=10, disable=True)

        reporter.update(-5)

        # Current should be negative
        assert reporter.current == -5

        reporter.close()

    def test_progress_large_update(self):
        """Test update larger than total."""
        reporter = ProgressReporter(total=10, disable=True)

        reporter.update(100)

        assert reporter.current == 100

        reporter.close()

    def test_empty_iterable(self):
        """Test progress with empty iterable."""
        items = []
        result = []

        for item in get_progress_iterator(items, disable=True):
            result.append(item)

        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
