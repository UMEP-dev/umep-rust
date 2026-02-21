"""Tiling performance benchmark scaffold for CI regression detection.

This suite focuses on orchestration-level regressions:
- worker-count scaling does not collapse
- bounded in-flight scheduling runs correctly
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather
from solweig.tiling import _calculate_tiled, _calculate_timeseries_tiled

pytestmark = pytest.mark.slow


class TestTilingBenchmark:
    """Benchmark scaffold for tiled orchestration performance."""

    @pytest.fixture
    def benchmark_surface(self):
        size = 520  # 3x3 tiles with tile_size=256 and zero overlap
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

    @pytest.fixture
    def benchmark_location(self):
        return Location(latitude=57.7, longitude=12.0, utc_offset=1)

    @pytest.fixture
    def benchmark_weather_series(self):
        base = datetime(2024, 7, 15, 11, 0)
        return [
            Weather(
                datetime=base + timedelta(hours=i),
                ta=24.0 + i,
                rh=50.0,
                global_rad=700.0,
                ws=2.0,
            )
            for i in range(3)
        ]

    def test_tile_workers_scaling_sanity(
        self, benchmark_surface, benchmark_location, benchmark_weather_series, tmp_path
    ):
        """Two workers should not be significantly slower than one worker."""
        output_dir_1w = tmp_path / "1w"
        output_dir_2w = tmp_path / "2w"

        t0 = time.perf_counter()
        summary_1w = _calculate_timeseries_tiled(
            benchmark_surface,
            benchmark_weather_series,
            benchmark_location,
            tile_workers=1,
            tile_queue_depth=0,
            prefetch_tiles=False,
            output_dir=output_dir_1w,
        )
        t1 = time.perf_counter()
        summary_2w = _calculate_timeseries_tiled(
            benchmark_surface,
            benchmark_weather_series,
            benchmark_location,
            tile_workers=2,
            tile_queue_depth=2,
            prefetch_tiles=True,
            output_dir=output_dir_2w,
        )
        t2 = time.perf_counter()

        elapsed_1w = t1 - t0
        elapsed_2w = t2 - t1

        assert summary_1w.n_timesteps == len(benchmark_weather_series)
        assert summary_2w.n_timesteps == len(benchmark_weather_series)
        assert elapsed_2w <= elapsed_1w * 1.25, (
            f"Tiled scaling regression: 2 workers too slow ({elapsed_2w:.3f}s vs {elapsed_1w:.3f}s)"
        )

    def test_bounded_inflight_runtime_controls(
        self, benchmark_surface, benchmark_location, benchmark_weather_series, tmp_path
    ):
        """Bounded in-flight scheduling executes correctly with small queue depth."""
        from conftest import read_timestep_geotiff

        output_dir = tmp_path / "bounded"
        summary = _calculate_timeseries_tiled(
            benchmark_surface,
            benchmark_weather_series,
            benchmark_location,
            tile_workers=2,
            tile_queue_depth=0,
            prefetch_tiles=False,
            output_dir=output_dir,
            outputs=["tmrt"],
        )
        assert summary.n_timesteps == len(benchmark_weather_series)
        for i in range(len(benchmark_weather_series)):
            tmrt = read_timestep_geotiff(output_dir, "tmrt", i)
            assert tmrt is not None

    def test_anisotropic_tiled_runtime_smoke(self, benchmark_location):
        """Anisotropic tiled path runs with non-zero overlap/runtime controls."""
        from solweig.models.precomputed import ShadowArrays

        size = 300
        n_patches = 153
        n_pack = (n_patches + 7) // 8

        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        dsm[120:180, 120:180] = 10.0  # 5 m relative height -> non-zero overlap
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
        surface.shadow_matrices = ShadowArrays(
            _shmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
            _vegshmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
            _vbshmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
            _n_patches=n_patches,
        )

        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=27.0,
            rh=45.0,
            global_rad=800.0,
            ws=2.0,
        )

        result = _calculate_tiled(
            surface=surface,
            location=benchmark_location,
            weather=weather,
            tile_size=256,
            use_anisotropic_sky=True,
            tile_workers=2,
            tile_queue_depth=1,
            prefetch_tiles=True,
            max_shadow_distance_m=80.0,
        )
        assert result.tmrt is not None
        assert np.isfinite(result.tmrt).sum() > 0
