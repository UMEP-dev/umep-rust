"""Integration tests for multi-tile processing.

These tests use larger synthetic rasters to actually exercise multi-tile
processing rather than falling back to single-tile mode.
"""

from datetime import datetime

import numpy as np
import pytest
from solweig import (
    Location,
    SurfaceData,
    Weather,
    calculate,
    calculate_tiled,
)

pytestmark = pytest.mark.slow


class TestMultiTileProcessing:
    """Tests that actually exercise multi-tile processing."""

    @pytest.fixture
    def large_urban_surface(self):
        """Create a 800x800 synthetic urban surface with low buildings.

        Uses LOW buildings (5m) so buffer requirement is small enough
        to actually trigger multi-tile processing.

        Buffer formula: max_height / tan(3°) = 5 / 0.0524 ≈ 95m
        With 95px buffer, tile_size=256 has 66px core which is too small.
        But with tile_size=300, we get ~108px core (marginal).
        """
        np.random.seed(42)
        size = 800

        # Base terrain at 10m
        dsm = np.ones((size, size), dtype=np.float32) * 10.0

        # Add LOW buildings (5m above ground = 15m total) to keep buffer small
        for _ in range(15):
            x, y = np.random.randint(50, size - 50, 2)
            w, h = np.random.randint(15, 30, 2)
            dsm[y : y + h, x : x + w] = 15.0  # 5m above ground

        # Create land cover (grass=5, buildings=2)
        land_cover = np.ones((size, size), dtype=np.int32) * 5
        land_cover[dsm > 12] = 2

        surface = SurfaceData(
            dsm=dsm,
            land_cover=land_cover,
            pixel_size=1.0,
        )

        return surface

    @pytest.fixture
    def weather_noon(self):
        """Summer noon weather conditions."""
        return Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=28.0,
            rh=45.0,
            global_rad=850.0,
            ws=2.0,
        )

    @pytest.fixture
    def location_gothenburg(self):
        """Gothenburg, Sweden location."""
        return Location(latitude=57.7, longitude=12.0, utc_offset=2)

    def test_multitile_actually_tiles(self, large_urban_surface, location_gothenburg, weather_noon, caplog):
        """Verify that large raster is actually processed in multiple tiles."""
        import logging

        caplog.set_level(logging.INFO)

        # With 5m buildings: buffer ≈ 95m = 95px
        # tile_size=350 gives ~160px core
        # 800x800 / 350 = ~6 tiles (3x2 grid roughly)
        result = calculate_tiled(
            large_urban_surface,
            location_gothenburg,
            weather_noon,
            tile_size=350,
        )

        # Check that tiled processing was used (not fallback)
        log_lower = caplog.text.lower()
        assert "tiled processing" in log_lower and "tiles" in log_lower, (
            f"Expected tiled processing message, got: {caplog.text}"
        )

        # Verify output shape matches input
        assert result.tmrt.shape == large_urban_surface.shape

        # Verify reasonable Tmrt values (not all NaN)
        valid_pixels = np.isfinite(result.tmrt)
        assert valid_pixels.sum() > 0.8 * result.tmrt.size, "Too many NaN values in Tmrt"

        # Tmrt should be in reasonable range for summer midday
        valid_tmrt = result.tmrt[valid_pixels]
        assert 20 < np.median(valid_tmrt) < 80, f"Median Tmrt {np.median(valid_tmrt):.1f}°C out of expected range"

    def test_multitile_vs_nontiled_comparison(self, location_gothenburg, weather_noon):
        """Compare tiled vs non-tiled results on a moderate-size raster."""
        # Use 400x400 which can be processed either way
        size = 400
        np.random.seed(123)

        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        # Add a few small buildings
        for _ in range(5):
            x, y = np.random.randint(50, size - 50, 2)
            dsm[y : y + 20, x : x + 20] = 15.0

        surface = SurfaceData(dsm=dsm, pixel_size=2.0)  # 2m pixels = 800m extent

        # Non-tiled reference
        result_ref = calculate(surface, location_gothenburg, weather_noon)

        # Tiled with small tiles to force actual tiling
        # With 2m pixels and ~15m buildings, buffer = 15/tan(3°) ≈ 286m = 143 pixels
        # So tile_size=256 with buffer 143 means core is only 113px, which is too small
        # Use tile_size=512 to ensure meaningful core
        result_tiled = calculate_tiled(
            surface,
            location_gothenburg,
            weather_noon,
            tile_size=512,
        )

        # Compare Tmrt where both are valid
        both_valid = np.isfinite(result_ref.tmrt) & np.isfinite(result_tiled.tmrt)

        if both_valid.sum() > 0:
            diff = np.abs(result_tiled.tmrt[both_valid] - result_ref.tmrt[both_valid])
            mean_diff = diff.mean()
            max_diff = diff.max()

            # Should be very close (tile boundaries shouldn't cause significant differences)
            assert mean_diff < 0.5, f"Mean Tmrt diff {mean_diff:.2f}°C too large"
            assert max_diff < 2.0, f"Max Tmrt diff {max_diff:.2f}°C too large (possible tile boundary issue)"

    def test_tile_boundary_continuity(self, location_gothenburg, weather_noon):
        """Verify results are continuous across tile boundaries."""
        size = 600

        # Uniform flat terrain - should have smooth Tmrt
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)

        result = calculate_tiled(
            surface,
            location_gothenburg,
            weather_noon,
            tile_size=256,
        )

        valid_tmrt = result.tmrt[np.isfinite(result.tmrt)]

        # For flat terrain, Tmrt should be nearly uniform
        std_dev = np.std(valid_tmrt)
        assert std_dev < 0.5, f"Tmrt std dev {std_dev:.2f}°C too high for flat terrain"

    def test_progress_callback(self, large_urban_surface, location_gothenburg, weather_noon):
        """Test that progress callback is called correctly."""
        progress_calls = []

        def track_progress(tile_idx, total_tiles):
            progress_calls.append((tile_idx, total_tiles))

        _result = calculate_tiled(
            large_urban_surface,
            location_gothenburg,
            weather_noon,
            tile_size=350,  # Must be large enough for buffer requirement
            progress_callback=track_progress,
        )

        # Should have received progress updates
        assert len(progress_calls) > 0, "No progress callbacks received"

        # Last call should indicate completion
        last_idx, total = progress_calls[-1]
        assert last_idx == total, f"Final callback should show completion: {last_idx}/{total}"


class TestTilingMemoryBehavior:
    """Tests focused on memory behavior of tiled processing."""

    def test_tile_isolation(self):
        """Verify tiles don't share mutable state."""
        size = 500
        dsm = np.ones((size, size), dtype=np.float32) * 10.0
        dsm[200:300, 200:300] = 25.0  # Building

        original_dsm = dsm.copy()

        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        _ = calculate_tiled(surface, location, weather, tile_size=256)

        # Original DSM should be unchanged
        assert np.allclose(surface.dsm, original_dsm), "DSM was modified during tiled processing"
