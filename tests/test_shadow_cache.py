"""Tests for ShadowArrays float32 cache release (memory optimization)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from solweig.models.precomputed import ShadowArrays

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def shadow_arrays():
    """Small ShadowArrays for testing cache behavior."""
    shape = (10, 10, 5)  # rows, cols, patches
    rng = np.random.default_rng(42)
    # Explicit cast to satisfy type checker
    shmat: NDArray[np.uint8] = rng.integers(0, 256, shape, dtype=np.uint8).astype(np.uint8)
    vegshmat: NDArray[np.uint8] = rng.integers(0, 256, shape, dtype=np.uint8).astype(np.uint8)
    vbshmat: NDArray[np.uint8] = rng.integers(0, 256, shape, dtype=np.uint8).astype(np.uint8)
    return ShadowArrays(
        _shmat_u8=shmat,
        _vegshmat_u8=vegshmat,
        _vbshmat_u8=vbshmat,
    )


class TestReleaseFloat32Cache:
    """Tests for release_float32_cache() method."""

    def test_release_clears_cache(self, shadow_arrays):
        """After release, cached float32 arrays are None."""
        # Populate cache
        _ = shadow_arrays.shmat
        _ = shadow_arrays.vegshmat
        _ = shadow_arrays.vbshmat
        assert shadow_arrays._shmat_f32 is not None

        shadow_arrays.release_float32_cache()

        assert shadow_arrays._shmat_f32 is None
        assert shadow_arrays._vegshmat_f32 is None
        assert shadow_arrays._vbshmat_f32 is None

    def test_uint8_unchanged_after_release(self, shadow_arrays):
        """uint8 originals remain intact after cache release."""
        original_shmat = shadow_arrays._shmat_u8.copy()

        _ = shadow_arrays.shmat  # Populate cache
        shadow_arrays.release_float32_cache()

        np.testing.assert_array_equal(shadow_arrays._shmat_u8, original_shmat)

    def test_cache_recreated_on_reaccess(self, shadow_arrays):
        """Accessing properties after release recreates the cache correctly."""
        # First access
        shmat_before = shadow_arrays.shmat.copy()

        # Release and re-access
        shadow_arrays.release_float32_cache()
        shmat_after = shadow_arrays.shmat

        np.testing.assert_array_equal(shmat_before, shmat_after)
        assert shadow_arrays._shmat_f32 is not None

    def test_safe_to_call_before_access(self, shadow_arrays):
        """Calling release before any cache access is a no-op."""
        shadow_arrays.release_float32_cache()  # Should not raise

        assert shadow_arrays._shmat_f32 is None

    def test_safe_to_call_multiple_times(self, shadow_arrays):
        """Calling release multiple times is safe."""
        _ = shadow_arrays.shmat
        shadow_arrays.release_float32_cache()
        shadow_arrays.release_float32_cache()  # Second call is a no-op

        assert shadow_arrays._shmat_f32 is None

    def test_diffsh_works_after_release(self, shadow_arrays):
        """diffsh() still works after cache release (re-converts from uint8)."""
        diffsh_before = shadow_arrays.diffsh(transmissivity=0.03).copy()

        shadow_arrays.release_float32_cache()
        diffsh_after = shadow_arrays.diffsh(transmissivity=0.03)

        np.testing.assert_array_equal(diffsh_before, diffsh_after)
