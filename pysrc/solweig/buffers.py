"""
Pre-allocated buffer pools for reducing per-timestep memory allocation.

This module provides a BufferPool class that manages reusable numpy arrays
to avoid repeated allocation/deallocation during time series calculations.

Usage:
    pool = BufferPool(shape=(1000, 1000))

    # Get a zeroed buffer
    temp = pool.get_zeros("ani_lum")

    # Get an uninitialized buffer (faster, use when you'll overwrite all values)
    temp = pool.get("shadow_temp")

    # Buffers are automatically reused on next get() call with same name
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BufferPool:
    """
    Manages pre-allocated numpy arrays for reuse across timesteps.

    This reduces memory allocation overhead during time series calculations
    by reusing the same memory for intermediate computations.

    The pool uses named buffers - each unique name gets its own buffer that
    persists across calls. When you request a buffer by name, you get the
    same underlying memory (optionally zeroed).

    Attributes:
        shape: The 2D shape for all buffers in this pool
        dtype: Data type for buffers (default: float32)
        _buffers: Dictionary mapping names to pre-allocated arrays

    Example:
        pool = BufferPool((1000, 1000))

        # First call allocates
        buf1 = pool.get_zeros("radiation_temp")
        buf1[:] = some_computation()

        # Second call reuses same memory (zeroed)
        buf1 = pool.get_zeros("radiation_temp")  # Same buffer, zeroed
    """

    __slots__ = ("shape", "dtype", "_buffers")

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype | type = np.float32,
    ) -> None:
        """
        Initialize a buffer pool.

        Args:
            shape: 2D shape (rows, cols) for all buffers
            dtype: NumPy dtype for buffers (default: float32)
        """
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._buffers: dict[str, NDArray[np.floating]] = {}

    def get(self, name: str) -> NDArray[np.floating]:
        """
        Get a buffer by name (uninitialized).

        Returns an uninitialized buffer - use this when you will overwrite
        all values anyway. Faster than get_zeros().

        Args:
            name: Unique identifier for this buffer

        Returns:
            Pre-allocated array (contents undefined)
        """
        if name not in self._buffers:
            self._buffers[name] = np.empty(self.shape, dtype=self.dtype)
        return self._buffers[name]

    def get_zeros(self, name: str) -> NDArray[np.floating]:
        """
        Get a zeroed buffer by name.

        Returns a buffer filled with zeros. Use this when you need
        a clean slate for accumulation operations.

        Args:
            name: Unique identifier for this buffer

        Returns:
            Pre-allocated array filled with zeros
        """
        buf = self.get(name)
        buf.fill(0.0)
        return buf

    def get_full(self, name: str, fill_value: float) -> NDArray[np.floating]:
        """
        Get a buffer filled with a specific value.

        Args:
            name: Unique identifier for this buffer
            fill_value: Value to fill the buffer with

        Returns:
            Pre-allocated array filled with fill_value
        """
        buf = self.get(name)
        buf.fill(fill_value)
        return buf

    def ensure_float32(
        self,
        arr: NDArray,
        name: str | None = None,
    ) -> NDArray[np.float32]:
        """
        Ensure array is float32, using pooled buffer if conversion needed.

        If the array is already float32, returns it unchanged (no copy).
        If conversion is needed and a name is provided, uses a pooled buffer.
        Otherwise, falls back to regular astype().

        Args:
            arr: Input array (any dtype)
            name: Optional buffer name for pooled conversion

        Returns:
            Array with float32 dtype (may be same object if already float32)
        """
        if arr.dtype == np.float32:
            return arr

        if name is not None and arr.shape == self.shape:
            buf = self.get(name)
            np.copyto(buf, arr, casting="unsafe")
            return buf

        return arr.astype(np.float32)

    def clear(self) -> None:
        """
        Clear all buffers from the pool.

        Call this to release memory when done with a calculation series.
        """
        self._buffers.clear()

    @property
    def num_buffers(self) -> int:
        """Number of buffers currently allocated."""
        return len(self._buffers)

    @property
    def memory_bytes(self) -> int:
        """Total memory used by all buffers in bytes."""
        if not self._buffers:
            return 0
        return len(self._buffers) * self.shape[0] * self.shape[1] * self.dtype.itemsize

    def __repr__(self) -> str:
        mb = self.memory_bytes / (1024 * 1024)
        return f"BufferPool(shape={self.shape}, dtype={self.dtype}, buffers={self.num_buffers}, memory={mb:.1f}MB)"


class TimestepBuffers:
    """
    Context manager for timestep-scoped buffer reuse.

    This provides a convenient way to reuse buffers within a single timestep
    calculation without polluting the namespace.

    Usage:
        with TimestepBuffers((1000, 1000)) as buffers:
            temp1 = buffers.get_zeros("radiation")
            temp2 = buffers.get_zeros("shadow")
            # ... use buffers ...
        # Buffers are cleared when exiting context
    """

    __slots__ = ("pool",)

    def __init__(self, shape: tuple[int, int], dtype: np.dtype | type = np.float32):
        self.pool = BufferPool(shape, dtype)

    def __enter__(self) -> BufferPool:
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.pool.clear()
        return None


def ensure_float32_inplace(arr: NDArray) -> NDArray[np.float32]:
    """
    Convert array to float32 in-place if possible, otherwise copy.

    This is a utility function for cases where we want to avoid allocation
    when the input is already float32.

    Args:
        arr: Input array

    Returns:
        Float32 array (same object if already float32, new array otherwise)
    """
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)


def as_float32(arr: NDArray) -> NDArray[np.float32]:
    """
    Ensure array is float32, avoiding copy if already correct dtype.

    Shorthand for ensure_float32_inplace() - use this in component code
    to replace `.astype(np.float32)` calls where the array might already
    be float32.

    Args:
        arr: Input array (any dtype)

    Returns:
        Float32 array (same object if already float32, copy otherwise)

    Example:
        # Instead of:
        svf.astype(np.float32)  # Always copies

        # Use:
        as_float32(svf)  # Only copies if needed
    """
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)
