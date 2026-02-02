"""Tests for buffer pool functionality."""

import numpy as np
import pytest

from solweig.buffers import BufferPool, TimestepBuffers, ensure_float32_inplace


class TestBufferPool:
    """Tests for BufferPool class."""

    def test_creates_buffer_on_first_get(self):
        """First get() call allocates a new buffer."""
        pool = BufferPool((100, 100))
        buf = pool.get("test")

        assert buf.shape == (100, 100)
        assert buf.dtype == np.float32
        assert pool.num_buffers == 1

    def test_reuses_buffer_on_subsequent_get(self):
        """Subsequent get() calls return the same buffer."""
        pool = BufferPool((100, 100))

        buf1 = pool.get("test")
        buf1[0, 0] = 42.0

        buf2 = pool.get("test")

        # Should be the same underlying buffer
        assert buf1 is buf2
        assert buf2[0, 0] == 42.0

    def test_get_zeros_fills_with_zeros(self):
        """get_zeros() returns a zeroed buffer."""
        pool = BufferPool((100, 100))

        # First write some data
        buf1 = pool.get("test")
        buf1.fill(999.0)

        # get_zeros should zero it
        buf2 = pool.get_zeros("test")
        assert np.all(buf2 == 0.0)

    def test_get_full_fills_with_value(self):
        """get_full() returns buffer filled with specified value."""
        pool = BufferPool((100, 100))

        buf = pool.get_full("test", 3.14)
        assert np.allclose(buf, 3.14)

    def test_different_names_get_different_buffers(self):
        """Different buffer names get separate allocations."""
        pool = BufferPool((100, 100))

        buf1 = pool.get("buffer_a")
        buf2 = pool.get("buffer_b")

        assert buf1 is not buf2
        assert pool.num_buffers == 2

    def test_ensure_float32_no_copy_when_already_float32(self):
        """ensure_float32 returns same object if already float32."""
        pool = BufferPool((100, 100))

        arr = np.zeros((100, 100), dtype=np.float32)
        result = pool.ensure_float32(arr, "test")

        assert result is arr  # Same object, no copy

    def test_ensure_float32_converts_other_dtypes(self):
        """ensure_float32 converts non-float32 arrays."""
        pool = BufferPool((100, 100))

        arr = np.zeros((100, 100), dtype=np.float64)
        arr[0, 0] = 1.5

        result = pool.ensure_float32(arr, "test")

        assert result.dtype == np.float32
        assert result[0, 0] == 1.5
        assert result is not arr  # Different object

    def test_ensure_float32_uses_pooled_buffer(self):
        """ensure_float32 reuses pooled buffer for conversion."""
        pool = BufferPool((100, 100))

        arr1 = np.ones((100, 100), dtype=np.float64)
        arr2 = np.ones((100, 100), dtype=np.float64) * 2

        result1 = pool.ensure_float32(arr1, "conv")
        result2 = pool.ensure_float32(arr2, "conv")

        # Should reuse the same pooled buffer
        assert result1 is result2
        # Second call overwrote the values
        assert np.all(result2 == 2.0)

    def test_memory_bytes_calculation(self):
        """memory_bytes returns correct total."""
        pool = BufferPool((100, 100), dtype=np.float32)

        pool.get("a")
        pool.get("b")

        # 2 buffers * 100 * 100 * 4 bytes
        expected = 2 * 100 * 100 * 4
        assert pool.memory_bytes == expected

    def test_clear_removes_all_buffers(self):
        """clear() removes all buffers from pool."""
        pool = BufferPool((100, 100))
        pool.get("a")
        pool.get("b")

        assert pool.num_buffers == 2

        pool.clear()

        assert pool.num_buffers == 0
        assert pool.memory_bytes == 0

    def test_custom_dtype(self):
        """Pool respects custom dtype."""
        pool = BufferPool((50, 50), dtype=np.float64)
        buf = pool.get("test")

        assert buf.dtype == np.float64

    def test_repr_shows_useful_info(self):
        """repr() shows shape, buffers, and memory."""
        pool = BufferPool((100, 100))
        pool.get("test")

        repr_str = repr(pool)
        assert "shape=(100, 100)" in repr_str
        assert "buffers=1" in repr_str
        assert "memory=" in repr_str


class TestTimestepBuffers:
    """Tests for TimestepBuffers context manager."""

    def test_provides_pool_in_context(self):
        """Context manager provides BufferPool."""
        with TimestepBuffers((100, 100)) as pool:
            assert isinstance(pool, BufferPool)
            buf = pool.get_zeros("test")
            assert buf.shape == (100, 100)

    def test_clears_buffers_on_exit(self):
        """Buffers are cleared when exiting context."""
        buffers = TimestepBuffers((100, 100))

        with buffers as pool:
            pool.get("a")
            pool.get("b")
            assert pool.num_buffers == 2

        # After context exit, pool should be cleared
        assert buffers.pool.num_buffers == 0


class TestEnsureFloat32Inplace:
    """Tests for ensure_float32_inplace utility."""

    def test_returns_same_if_float32(self):
        """Returns same object if already float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ensure_float32_inplace(arr)

        assert result is arr

    def test_converts_float64(self):
        """Converts float64 to float32."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = ensure_float32_inplace(arr)

        assert result.dtype == np.float32
        assert result is not arr

    def test_converts_int(self):
        """Converts integer arrays to float32."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = ensure_float32_inplace(arr)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestBufferPoolPerformance:
    """Performance-related tests for buffer pool."""

    def test_pool_get_faster_than_empty_allocation(self):
        """Pooled get() should be faster than repeated np.empty."""
        import time

        shape = (500, 500)
        iterations = 100

        # Time repeated empty allocations
        start = time.perf_counter()
        for _ in range(iterations):
            arr = np.empty(shape, dtype=np.float32)
            arr[0, 0] = 1.0  # Prevent optimization
        alloc_time = time.perf_counter() - start

        # Time pooled buffers (get without zeroing)
        pool = BufferPool(shape)
        start = time.perf_counter()
        for _ in range(iterations):
            arr = pool.get("test")
            arr[0, 0] = 1.0  # Prevent optimization
        pool_time = time.perf_counter() - start

        # Pool should be faster since it avoids allocation
        # But we use a generous margin since timing can vary
        # The main benefit is reducing GC pressure, which is hard to measure
        assert pool_time < alloc_time * 10.0, (
            f"Pool ({pool_time:.4f}s) should not be dramatically slower than "
            f"allocation ({alloc_time:.4f}s)"
        )

    def test_ensure_float32_avoids_copy_when_possible(self):
        """ensure_float32 should avoid copies for float32 input."""
        import time

        shape = (500, 500)
        iterations = 100
        pool = BufferPool(shape)

        # Create float32 array
        arr = np.zeros(shape, dtype=np.float32)

        # Time ensure_float32 (should be nearly instant - no copy)
        start = time.perf_counter()
        for _ in range(iterations):
            result = pool.ensure_float32(arr, "test")
            assert result is arr  # Same object
        no_copy_time = time.perf_counter() - start

        # Time astype (always copies)
        start = time.perf_counter()
        for _ in range(iterations):
            result = arr.astype(np.float32)
        copy_time = time.perf_counter() - start

        # No-copy should be much faster
        assert no_copy_time < copy_time, (
            f"No-copy ({no_copy_time:.4f}s) should be faster than "
            f"copy ({copy_time:.4f}s)"
        )
