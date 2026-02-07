"""Memory benchmark tests for CI regression detection.

These tests verify that memory usage stays within expected bounds.
They run on small grids to be fast in CI while still detecting regressions.

Memory target: ~370 bytes/pixel (measured Feb 2026 baseline)
Regression threshold: 500 bytes/pixel (35% headroom for variance)
"""

import tracemalloc
from datetime import datetime

import numpy as np
import pytest
from solweig import Location, SurfaceData, Weather, calculate

pytestmark = pytest.mark.slow


class TestMemoryBenchmark:
    """Memory usage benchmarks for CI."""

    # Target: ~370 bytes/pixel (Feb 2026 baseline)
    # Threshold: 500 bytes/pixel (35% headroom for CI variance)
    MAX_BYTES_PER_PIXEL = 500

    @pytest.fixture
    def benchmark_surface(self):
        """Create a 150x150 benchmark surface.

        Small enough to be fast in CI, large enough to amortize fixed overhead.
        """
        size = 150
        np.random.seed(42)

        dsm = np.ones((size, size), dtype=np.float32) * 10.0

        # Add a few low buildings (5m above ground to keep buffer small)
        for _ in range(5):
            x, y = np.random.randint(15, size - 15, 2)
            w, h = np.random.randint(5, 10, 2)
            dsm[y : y + h, x : x + w] = 15.0  # 5m above ground

        land_cover = np.ones((size, size), dtype=np.int32) * 5
        land_cover[dsm > 12] = 2

        from conftest import make_mock_svf

        return SurfaceData(
            dsm=dsm,
            land_cover=land_cover,
            pixel_size=1.0,
            svf=make_mock_svf((size, size)),
        )

    @pytest.fixture
    def benchmark_location(self):
        """Athens, Greece - good sun angle for testing."""
        return Location(latitude=37.98, longitude=23.73, utc_offset=2)

    @pytest.fixture
    def benchmark_weather(self):
        """Summer noon conditions."""
        return Weather(
            datetime=datetime(2024, 7, 21, 12, 0),
            ta=30.0,
            rh=50.0,
            global_rad=800.0,
            ws=2.0,
        )

    def test_memory_per_pixel_within_threshold(self, benchmark_surface, benchmark_location, benchmark_weather):
        """Verify memory usage stays within acceptable bounds.

        This test catches memory regressions (e.g., accidental float64 usage,
        leaked allocations, or inefficient intermediate arrays).
        """
        # Preprocess surface first (one-time cost not counted in per-timestep)
        benchmark_surface.preprocess()

        # Start memory tracing
        tracemalloc.start()
        tracemalloc.reset_peak()

        # Run calculation
        result = calculate(benchmark_surface, benchmark_location, benchmark_weather)

        # Get peak memory
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate bytes per pixel
        n_pixels = benchmark_surface.shape[0] * benchmark_surface.shape[1]
        bytes_per_pixel = peak / n_pixels

        # Verify result is valid (sanity check)
        assert result.tmrt is not None
        assert np.isfinite(result.tmrt).sum() > 0.8 * n_pixels

        # Verify memory within threshold
        assert bytes_per_pixel < self.MAX_BYTES_PER_PIXEL, (
            f"Memory regression detected: {bytes_per_pixel:.1f} bytes/pixel "
            f"exceeds threshold of {self.MAX_BYTES_PER_PIXEL} bytes/pixel. "
            f"Peak memory: {peak / 1024 / 1024:.1f} MB for {n_pixels:,} pixels."
        )

    def test_float32_arrays_used(self, benchmark_surface, benchmark_location, benchmark_weather):
        """Verify output arrays use float32 (not float64)."""
        benchmark_surface.preprocess()

        result = calculate(benchmark_surface, benchmark_location, benchmark_weather)

        # All output arrays should be float32
        assert result.tmrt.dtype == np.float32, f"tmrt dtype is {result.tmrt.dtype}, expected float32"
        if result.shadow is not None:
            assert result.shadow.dtype == np.float32, f"shadow dtype is {result.shadow.dtype}"
        if result.kdown is not None:
            assert result.kdown.dtype == np.float32, f"kdown dtype is {result.kdown.dtype}"
        if result.kup is not None:
            assert result.kup.dtype == np.float32, f"kup dtype is {result.kup.dtype}"
        if result.ldown is not None:
            assert result.ldown.dtype == np.float32, f"ldown dtype is {result.ldown.dtype}"
        if result.lup is not None:
            assert result.lup.dtype == np.float32, f"lup dtype is {result.lup.dtype}"

    def test_surface_arrays_float32(self):
        """Verify surface data arrays use float32."""
        size = 100
        np.random.seed(42)

        dsm = np.ones((size, size), dtype=np.float32) * 10.0
        cdsm = np.ones((size, size), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, pixel_size=1.0)
        surface.preprocess()

        assert surface.dsm.dtype == np.float32
        if surface.cdsm is not None:
            assert surface.cdsm.dtype == np.float32
        if surface.dem is not None:
            assert surface.dem.dtype == np.float32
