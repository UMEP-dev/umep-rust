"""Integration tests for multi-tile processing.

These tests use larger synthetic rasters to actually exercise multi-tile
processing rather than falling back to single-tile mode.
"""

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import (
    Location,
    PrecomputedData,
    SurfaceData,
    Weather,
    calculate,
    calculate_tiled,
)
from solweig.errors import MissingPrecomputedData
from solweig.models.state import ThermalState, TileSpec
from solweig.tiling import (
    _calculate_auto_tile_size,
    _extract_tile_surface,
    _merge_tile_state,
    _should_use_tiling,
    _slice_tile_state,
    calculate_buffer_distance,
    compute_max_tile_side,
)

pytestmark = pytest.mark.slow


class TestMultiTileProcessing:
    """Tests that actually exercise multi-tile processing."""

    @pytest.fixture(scope="class")
    def large_urban_surface(self):
        """Create a 400x400 synthetic urban surface with low buildings.

        Uses LOW buildings (5m) so buffer requirement is small enough
        to actually trigger multi-tile processing.

        Buffer formula: max_height / tan(3°) = 5 / 0.0524 ≈ 95m
        With 95px buffer, tile_size=256 has 66px core which is too small.
        But with tile_size=300, we get ~108px core (marginal).
        """
        np.random.seed(42)
        size = 400

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

        from conftest import make_mock_svf

        surface = SurfaceData(
            dsm=dsm,
            land_cover=land_cover,
            pixel_size=1.0,
            svf=make_mock_svf((size, size)),
        )

        return surface

    @pytest.fixture(scope="class")
    def weather_noon(self):
        """Summer noon weather conditions."""
        return Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=28.0,
            rh=45.0,
            global_rad=850.0,
            ws=2.0,
        )

    @pytest.fixture(scope="class")
    def location_gothenburg(self):
        """Gothenburg, Sweden location."""
        return Location(latitude=57.7, longitude=12.0, utc_offset=2)

    def test_multitile_actually_tiles(self, large_urban_surface, location_gothenburg, weather_noon):
        """Verify that large raster is actually processed in multiple tiles."""
        from unittest.mock import patch

        # 5m buildings (relative height = 15m DSM - 10m ground = 5m)
        # buffer = 5 / tan(3°) = 95.4m → 96px (below cap of 100m)
        # tile_size=350, buffer=96px → 4 tiles on a 400×400 raster
        captured = {}
        original_generate_tiles = __import__("solweig.tiling", fromlist=["generate_tiles"]).generate_tiles

        def spy_generate_tiles(rows, cols, tile_size, buffer_pixels):
            captured["n_tiles"] = len(original_generate_tiles(rows, cols, tile_size, buffer_pixels))
            captured["buffer_pixels"] = buffer_pixels
            return original_generate_tiles(rows, cols, tile_size, buffer_pixels)

        with patch("solweig.tiling.generate_tiles", side_effect=spy_generate_tiles):
            result = calculate_tiled(
                large_urban_surface,
                location_gothenburg,
                weather_noon,
                tile_size=350,
                max_shadow_distance_m=100.0,
            )

        # Check that multi-tile processing was used
        assert captured.get("n_tiles", 0) > 1, f"Expected multiple tiles, got {captured.get('n_tiles', 0)}"

        # Buffer from 5m relative height: ceil(95.4 / 1.0) = 96px
        assert captured["buffer_pixels"] == 96

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

        surface = SurfaceData(dsm=dsm, pixel_size=2.0, svf=make_mock_svf((size, size)))  # 2m pixels = 800m extent

        # Non-tiled reference
        result_ref = calculate(surface, location_gothenburg, weather_noon)

        # Tiled with limited shadow distance to keep buffer manageable
        # With max_shadow_distance_m=200 and 2m pixels: buffer = 100 pixels
        result_tiled = calculate_tiled(
            surface,
            location_gothenburg,
            weather_noon,
            tile_size=300,
            max_shadow_distance_m=200.0,
        )

        # Compare Tmrt where both are valid
        both_valid = np.isfinite(result_ref.tmrt) & np.isfinite(result_tiled.tmrt)

        if both_valid.sum() > 0:
            diff = np.abs(result_tiled.tmrt[both_valid] - result_ref.tmrt[both_valid])
            mean_diff = diff.mean()
            max_diff = diff.max()

            # Both paths now use the same mock SVF (tiled path slices from global).
            # Only shadow edge effects from tiling should cause small differences.
            assert mean_diff < 0.01, f"Mean Tmrt diff {mean_diff:.2f}°C too large"
            assert max_diff < 0.1, f"Max Tmrt diff {max_diff:.2f}°C too large (possible tile boundary issue)"

    def test_tile_boundary_continuity(self, location_gothenburg, weather_noon):
        """Verify results are continuous across tile boundaries."""
        size = 300

        # Uniform flat terrain - should have smooth Tmrt
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

        result = calculate_tiled(
            surface,
            location_gothenburg,
            weather_noon,
            tile_size=256,
            max_shadow_distance_m=50.0,
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
            tile_size=350,
            max_shadow_distance_m=100.0,
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
        size = 300
        dsm = np.ones((size, size), dtype=np.float32) * 10.0
        dsm[200:300, 200:300] = 25.0  # Building

        original_dsm = dsm.copy()

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        _ = calculate_tiled(surface, location, weather, tile_size=256, max_shadow_distance_m=50.0)

        # Original DSM should be unchanged
        assert np.allclose(surface.dsm, original_dsm), "DSM was modified during tiled processing"


class TestTilingHelpers:
    """Tests for tiling helper functions."""

    def test_should_use_tiling_below_threshold(self):
        """Rasters below resource limit should not trigger tiling."""
        assert not _should_use_tiling(100, 100)
        assert not _should_use_tiling(400, 400)

    def test_should_use_tiling_above_threshold(self):
        """Rasters exceeding resource-derived max should trigger tiling."""
        max_side = compute_max_tile_side(context="solweig")
        assert _should_use_tiling(max_side + 1, max_side + 1)
        assert _should_use_tiling(max_side + 1, 100)
        assert _should_use_tiling(100, max_side + 1)
        # Below resource limit — no tiling needed
        assert not _should_use_tiling(max_side, max_side)

    def test_calculate_tiled_requires_svf(self):
        """Tiled runtime must not implicitly compute missing SVF."""
        dsm = np.ones((320, 320), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        with pytest.raises(MissingPrecomputedData):
            calculate_tiled(
                surface,
                location,
                weather,
                tile_size=128,
                use_anisotropic_sky=False,
            )

    def test_auto_tile_size_returns_resource_max(self):
        """Auto tile size returns resource-derived maximum."""
        max_side = compute_max_tile_side(context="solweig")
        assert _calculate_auto_tile_size(max_side + 1000, max_side + 1000) == max_side
        assert _calculate_auto_tile_size(100, 100) == max_side

    def test_extract_tile_surface_reuses_svf(self):
        """When surface has precomputed SVF, tile surface should get sliced SVF."""
        size = 100
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        mock_svf = make_mock_svf((size, size))
        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=mock_svf)

        # Create a tile covering rows 10-60, cols 10-60 (with 10px overlap)
        tile = TileSpec(
            row_start=20,
            row_end=50,
            col_start=20,
            col_end=50,
            row_start_full=10,
            row_end_full=60,
            col_start_full=10,
            col_end_full=60,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0)

        # SVF should be set (sliced from global, not recomputed)
        assert tile_surface.svf is not None
        assert tile_surface.svf.svf.shape == (50, 50)  # 60-10 = 50

        # Values should match the sliced region of the global SVF
        np.testing.assert_array_equal(
            tile_surface.svf.svf,
            mock_svf.svf[10:60, 10:60],
        )

    def test_extract_tile_surface_leaves_svf_unset_when_missing(self):
        """When SVF is unavailable globally, tile extraction must not compute it."""
        size = 50
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        assert surface.svf is None

        tile = TileSpec(
            row_start=10,
            row_end=40,
            col_start=10,
            col_end=40,
            row_start_full=0,
            row_end_full=50,
            col_start_full=0,
            col_end_full=50,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0)

        # SVF remains unset; callers enforce the SVF precondition.
        assert tile_surface.svf is None

    def test_extract_tile_surface_uses_precomputed_svf_without_recompute(self):
        """When surface.svf is missing, precomputed.svf should be sliced and reused."""
        from unittest.mock import patch

        size = 50
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        surface = SurfaceData(dsm=dsm, pixel_size=1.0)
        precomputed = PrecomputedData(svf=make_mock_svf((size, size)))

        tile = TileSpec(
            row_start=10,
            row_end=40,
            col_start=10,
            col_end=40,
            row_start_full=0,
            row_end_full=50,
            col_start_full=0,
            col_end_full=50,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

        with patch.object(SurfaceData, "compute_svf", side_effect=AssertionError("compute_svf should not be called")):
            tile_surface = _extract_tile_surface(surface, tile, pixel_size=1.0, precomputed=precomputed)

        assert tile_surface.svf is not None
        assert tile_surface.svf.svf.shape == (50, 50)
        assert precomputed.svf is not None
        np.testing.assert_array_equal(
            tile_surface.svf.svf,
            precomputed.svf.svf[0:50, 0:50],
        )


class TestSliceMergeState:
    """Tests for _slice_tile_state and _merge_tile_state."""

    @pytest.fixture
    def global_state(self):
        """Create a global state with distinctive values."""
        shape = (100, 100)
        state = ThermalState(
            tgmap1=np.random.rand(*shape).astype(np.float32),
            tgmap1_e=np.random.rand(*shape).astype(np.float32),
            tgmap1_s=np.random.rand(*shape).astype(np.float32),
            tgmap1_w=np.random.rand(*shape).astype(np.float32),
            tgmap1_n=np.random.rand(*shape).astype(np.float32),
            tgout1=np.random.rand(*shape).astype(np.float32),
            firstdaytime=1.0,
            timeadd=0.5,
            timestep_dec=0.042,
        )
        return state

    @pytest.fixture
    def tile(self):
        """Create a tile spec for the center of a 100x100 grid."""
        # Core: rows 20-60, cols 30-70
        # Full (with 10px overlap): rows 10-70, cols 20-80
        return TileSpec(
            row_start=20,
            row_end=60,
            col_start=30,
            col_end=70,
            row_start_full=10,
            row_end_full=70,
            col_start_full=20,
            col_end_full=80,
            overlap_top=10,
            overlap_bottom=10,
            overlap_left=10,
            overlap_right=10,
        )

    def test_slice_tile_state_shape(self, global_state, tile):
        """Sliced state should have full tile shape."""
        sliced = _slice_tile_state(global_state, tile)
        expected_shape = (60, 60)  # rows 10-70, cols 20-80
        assert sliced.tgmap1.shape == expected_shape
        assert sliced.tgmap1_e.shape == expected_shape
        assert sliced.tgout1.shape == expected_shape

    def test_slice_tile_state_values(self, global_state, tile):
        """Sliced state should contain correct values from global state."""
        sliced = _slice_tile_state(global_state, tile)
        read_slice = tile.read_slice
        np.testing.assert_array_equal(sliced.tgmap1, global_state.tgmap1[read_slice])

    def test_slice_tile_state_scalars(self, global_state, tile):
        """Sliced state should copy scalar values."""
        sliced = _slice_tile_state(global_state, tile)
        assert sliced.firstdaytime == 1.0
        assert sliced.timeadd == 0.5
        assert sliced.timestep_dec == 0.042

    def test_slice_tile_state_independent(self, global_state, tile):
        """Sliced state should be a copy, not a view."""
        sliced = _slice_tile_state(global_state, tile)
        original_val = sliced.tgmap1[0, 0]
        sliced.tgmap1[0, 0] = -999.0
        assert global_state.tgmap1[tile.row_start_full, tile.col_start_full] == original_val

    def test_merge_tile_state_writes_core(self, global_state, tile):
        """Merge should write tile core region to correct global position."""
        sliced = _slice_tile_state(global_state, tile)

        # Modify tile state values
        sliced.tgmap1[:] = 42.0
        sliced.firstdaytime = 0.0
        sliced.timeadd = 1.5

        _merge_tile_state(sliced, tile, global_state)

        # Core region should be updated
        write_slice = tile.write_slice
        np.testing.assert_array_equal(global_state.tgmap1[write_slice], 42.0)

        # Scalar values should be updated
        assert global_state.firstdaytime == 0.0
        assert global_state.timeadd == 1.5

    def test_merge_tile_state_preserves_outside(self, global_state, tile):
        """Merge should not modify areas outside the tile's write region."""
        original_tgmap1 = global_state.tgmap1.copy()
        sliced = _slice_tile_state(global_state, tile)
        sliced.tgmap1[:] = 42.0

        _merge_tile_state(sliced, tile, global_state)

        # Areas outside write_slice should be unchanged
        # Check top-left corner (row 0, col 0) — outside tile
        assert global_state.tgmap1[0, 0] == original_tgmap1[0, 0]
        # Check bottom-right corner — outside tile
        assert global_state.tgmap1[99, 99] == original_tgmap1[99, 99]


class TestTimeseriesTiledIntegration:
    """Integration tests for tiled timeseries processing."""

    @pytest.fixture(scope="class")
    def small_surface(self):
        """Small 50x50 surface for fast tests (below tiling threshold)."""
        np.random.seed(42)
        size = 50
        dsm = np.ones((size, size), dtype=np.float32) * 5.0
        dsm[20:30, 20:30] = 10.0  # Small building
        return SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

    @pytest.fixture(scope="class")
    def location(self):
        return Location(latitude=57.7, longitude=12.0, utc_offset=2)

    @pytest.fixture(scope="class")
    def weather_pair(self):
        """Two consecutive timesteps for minimal timeseries."""
        return [
            Weather(datetime=datetime(2024, 7, 15, 11, 0), ta=26.0, rh=50.0, global_rad=750.0, ws=2.0),
            Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=45.0, global_rad=850.0, ws=2.0),
        ]

    def test_timeseries_tiled_matches_nontiled(self, small_surface, location, weather_pair):
        """Tiled timeseries should match non-tiled within numerical precision.

        Both paths use the same mock SVF from the surface (tiled path slices
        the global SVF per tile instead of recomputing).
        """
        from solweig import calculate_timeseries, calculate_timeseries_tiled

        # Non-tiled (normal path — uses mock SVF from surface)
        summary_ref = calculate_timeseries(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            timestep_outputs=["tmrt"],
        )

        # Tiled (forced via direct call — slices mock SVF from surface)
        summary_tiled = calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            timestep_outputs=["tmrt"],
        )

        assert len(summary_ref) == len(summary_tiled)

        for i, (ref, tiled) in enumerate(zip(summary_ref.results, summary_tiled.results, strict=False)):
            both_valid = np.isfinite(ref.tmrt) & np.isfinite(tiled.tmrt)
            if both_valid.sum() > 0:
                diff = np.abs(ref.tmrt[both_valid] - tiled.tmrt[both_valid])
                # Both paths now use the same mock SVF (tiled path slices from global).
                assert diff.mean() < 0.01, f"Timestep {i}: mean Tmrt diff {diff.mean():.2f}°C too large"
                assert diff.max() < 0.1, f"Timestep {i}: max Tmrt diff {diff.max():.2f}°C too large"

    def test_timeseries_tiled_state_accumulates(self, small_surface, location, weather_pair):
        """Thermal state should evolve across timesteps in tiled mode."""
        from solweig import calculate_timeseries_tiled

        summary = calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            timestep_outputs=["tmrt"],
        )

        # Both timesteps should produce valid results
        assert len(summary) == 2
        for r in summary.results:
            valid = np.isfinite(r.tmrt)
            assert valid.sum() > 0, "Expected some valid Tmrt values"

    def test_timeseries_tiled_progress_callback(self, small_surface, location, weather_pair):
        """Progress callback should be called for tiled timeseries."""
        from solweig import calculate_timeseries_tiled

        calls = []

        def track(current, total):
            calls.append((current, total))

        calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            progress_callback=track,
        )

        assert len(calls) > 0, "No progress callbacks received"

    def test_timeseries_tiled_default_no_timestep_outputs(self, small_surface, location, weather_pair):
        """Default mode should not retain tiled timestep results."""
        from solweig import calculate_timeseries_tiled

        summary = calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
        )

        assert summary.results == []
        assert len(summary) == 2

    def test_timeseries_tiled_summary_only_requests_tmrt_and_shadow(self, small_surface, location, weather_pair):
        """Summary-only mode should request tmrt and shadow from tiled per-tile calculations."""
        from solweig import SolweigResult, calculate_timeseries_tiled

        captured: list[set[str] | None] = []

        def _fake_calculate(**kwargs):
            captured.append(kwargs.get("_requested_outputs"))
            shape = kwargs["surface"].dsm.shape
            return SolweigResult(
                tmrt=np.zeros(shape, dtype=np.float32),
                shadow=np.zeros(shape, dtype=np.float32),
                kdown=None,
                kup=None,
                ldown=None,
                lup=None,
                utci=None,
                pet=None,
                state=None,
            )

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("solweig.api.calculate", _fake_calculate)
        try:
            summary = calculate_timeseries_tiled(
                surface=small_surface,
                weather_series=weather_pair,
                location=location,
            )
        finally:
            monkeypatch.undo()

        assert summary.results == []
        assert captured and all(req == {"tmrt", "shadow"} for req in captured)

    def test_timeseries_tiled_precreates_tile_surfaces_once(self, small_surface, location, weather_pair):
        """Tile surfaces should be extracted once per tile, not once per timestep."""
        from unittest.mock import patch

        from solweig import calculate_timeseries_tiled
        from solweig import tiling as tiling_module

        extract_calls = 0
        original_extract = tiling_module._extract_tile_surface

        def spy_extract(*args, **kwargs):
            nonlocal extract_calls
            extract_calls += 1
            return original_extract(*args, **kwargs)

        rows, cols = small_surface.shape
        pixel_size = small_surface.pixel_size
        max_height = small_surface.max_height
        buffer_m = calculate_buffer_distance(max_height)
        buffer_pixels = int(np.ceil(buffer_m / pixel_size))
        tile_size = tiling_module._calculate_auto_tile_size(rows, cols)
        adjusted_tile_size, _warning = tiling_module.validate_tile_size(tile_size, buffer_pixels, pixel_size)
        expected_tiles = len(tiling_module.generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels))

        with patch("solweig.tiling._extract_tile_surface", side_effect=spy_extract):
            calculate_timeseries_tiled(
                surface=small_surface,
                weather_series=weather_pair,
                location=location,
            )

        assert extract_calls == expected_tiles, (
            f"Expected {expected_tiles} tile surface extractions, got {extract_calls} "
            "(possible per-timestep recomputation regression)"
        )

    def test_timeseries_tiled_worker_parity(self, small_surface, location, weather_pair):
        """Worker count should not materially change tiled timeseries outputs."""
        from solweig import calculate_timeseries_tiled

        summary_one = calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            tile_workers=1,
            tile_queue_depth=0,
            prefetch_tiles=False,
            timestep_outputs=["tmrt"],
        )
        summary_multi = calculate_timeseries_tiled(
            surface=small_surface,
            weather_series=weather_pair,
            location=location,
            tile_workers=2,
            tile_queue_depth=2,
            prefetch_tiles=True,
            timestep_outputs=["tmrt"],
        )

        assert len(summary_one) == len(summary_multi)
        for ref, got in zip(summary_one.results, summary_multi.results, strict=False):
            both_valid = np.isfinite(ref.tmrt) & np.isfinite(got.tmrt)
            if both_valid.sum() > 0:
                diff = np.abs(ref.tmrt[both_valid] - got.tmrt[both_valid])
                assert diff.mean() < 0.01
                assert diff.max() < 0.1

    def test_invalid_tile_workers_raises(self, small_surface, location, weather_pair):
        """Invalid tile_workers should raise clear ValueError."""
        from solweig import calculate_timeseries_tiled

        with pytest.raises(ValueError, match="tile_workers must be >= 1"):
            calculate_timeseries_tiled(
                surface=small_surface,
                weather_series=weather_pair,
                location=location,
                tile_workers=0,
            )

    def test_invalid_tile_queue_depth_raises(self, small_surface, location, weather_pair):
        """Invalid tile_queue_depth should raise clear ValueError."""
        from solweig import calculate_timeseries_tiled

        with pytest.raises(ValueError, match="tile_queue_depth must be >= 0"):
            calculate_timeseries_tiled(
                surface=small_surface,
                weather_series=weather_pair,
                location=location,
                tile_queue_depth=-1,
            )


class TestTiledAnisotropicParity:
    """Verify anisotropic sky produces matching results in tiled vs non-tiled mode.

    Shadow matrices are spatially sliced per tile, so anisotropic diffuse
    radiation must agree between tiled and non-tiled paths.

    Uses a flat surface (no buildings → max_height=0 → zero buffer) so
    tile boundaries introduce no shadow truncation artifacts, giving a
    clean comparison of shadow matrix slicing.
    """

    @pytest.fixture(scope="class")
    def aniso_surface(self):
        """530x530 flat surface with synthetic shadow matrices (all visible).

        530x530 at tile_size=256 → ceil(530/256)=3 → 9 tiles, ensuring
        multi-tile processing is exercised. Flat terrain (max_height=0)
        means zero overlap buffer so results should match exactly.
        """
        from solweig.models.precomputed import ShadowArrays

        size = 530
        n_patches = 153
        n_pack = (n_patches + 7) // 8

        dsm = np.ones((size, size), dtype=np.float32) * 2.0

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))

        # All-visible shadow matrices (0xFF = every patch visible)
        shmat = np.full((size, size, n_pack), 0xFF, dtype=np.uint8)
        vegshmat = np.full((size, size, n_pack), 0xFF, dtype=np.uint8)
        vbshmat = np.full((size, size, n_pack), 0xFF, dtype=np.uint8)

        surface.shadow_matrices = ShadowArrays(
            _shmat_u8=shmat,
            _vegshmat_u8=vegshmat,
            _vbshmat_u8=vbshmat,
            _n_patches=n_patches,
        )
        return surface

    @pytest.fixture(scope="class")
    def aniso_location(self):
        return Location(latitude=57.7, longitude=12.0, utc_offset=2)

    @pytest.fixture(scope="class")
    def aniso_weather(self):
        return Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

    def test_anisotropic_tiled_vs_nontiled(self, aniso_surface, aniso_location, aniso_weather):
        """Tiled anisotropic sky matches non-tiled within numerical precision."""
        # Non-tiled reference
        result_ref = calculate(
            aniso_surface,
            aniso_location,
            aniso_weather,
            use_anisotropic_sky=True,
        )

        # Tiled: tile_size=256 on 530×530 → 9 tiles (3×3)
        result_tiled = calculate_tiled(
            aniso_surface,
            aniso_location,
            aniso_weather,
            tile_size=256,
            use_anisotropic_sky=True,
        )

        both_valid = np.isfinite(result_ref.tmrt) & np.isfinite(result_tiled.tmrt)
        assert both_valid.sum() > 0, "Expected valid Tmrt pixels"

        diff = np.abs(result_tiled.tmrt[both_valid] - result_ref.tmrt[both_valid])
        mean_diff = diff.mean()
        max_diff = diff.max()

        assert mean_diff < 0.01, f"Mean Tmrt diff {mean_diff:.4f}°C too large (tiled vs non-tiled anisotropic)"
        assert max_diff < 0.1, f"Max Tmrt diff {max_diff:.4f}°C too large (possible shadow matrix slicing issue)"


class TestHeightAwareBuffer:
    """Verify tiling functions compute buffer from actual building heights."""

    def test_short_buildings_get_small_buffer(self):
        """With 5m relative height the buffer should be ~95m, not the 500m default."""
        from unittest.mock import patch

        size = 400
        ground = 10.0
        building_height = 5.0  # above ground
        max_dsm = ground + building_height  # 15m absolute, 5m relative
        dsm = np.full((size, size), ground, dtype=np.float32)
        dsm[100:120, 100:120] = max_dsm

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=45.0, global_rad=850.0, ws=2.0)

        # Buffer uses relative height (5m), not absolute elevation (15m)
        # buffer = 5 / tan(3°) ≈ 95.4m — much less than default 500m cap
        expected_buffer = calculate_buffer_distance(building_height)
        assert 90 < expected_buffer < 100, f"Expected ~95m buffer, got {expected_buffer}"

        # Patch generate_tiles to capture the buffer_pixels that calculate_tiled passes
        captured = {}
        original_generate_tiles = __import__("solweig.tiling", fromlist=["generate_tiles"]).generate_tiles

        def spy_generate_tiles(rows, cols, tile_size, buffer_pixels):
            captured["buffer_pixels"] = buffer_pixels
            return original_generate_tiles(rows, cols, tile_size, buffer_pixels)

        with patch("solweig.tiling.generate_tiles", side_effect=spy_generate_tiles):
            _ = calculate_tiled(surface, location, weather, tile_size=350)

        # If generate_tiles was called, buffer should match relative-height-derived value
        if "buffer_pixels" in captured:
            expected_px = int(np.ceil(expected_buffer / surface.pixel_size))
            assert captured["buffer_pixels"] == expected_px, (
                f"Expected {expected_px}px buffer from {building_height}m relative height, "
                f"got {captured['buffer_pixels']}px"
            )

    def test_tall_buildings_capped_at_max(self):
        """With 30m relative height the buffer should cap at max_shadow_distance_m."""
        from unittest.mock import patch

        size = 400
        ground = 10.0
        building_height = 30.0
        max_dsm = ground + building_height  # 40m absolute, 30m relative
        dsm = np.full((size, size), ground, dtype=np.float32)
        dsm[100:120, 100:120] = max_dsm

        surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf((size, size)))
        location = Location(latitude=57.7, longitude=12.0, utc_offset=2)
        weather = Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=28.0, rh=45.0, global_rad=850.0, ws=2.0)

        cap = 200.0
        # 30m relative height: 30/tan(3°) ≈ 573m → capped at 200m
        expected_buffer = calculate_buffer_distance(building_height, max_shadow_distance_m=cap)
        assert expected_buffer == cap, f"Expected buffer capped at {cap}, got {expected_buffer}"

        captured = {}
        original_generate_tiles = __import__("solweig.tiling", fromlist=["generate_tiles"]).generate_tiles

        def spy_generate_tiles(rows, cols, tile_size, buffer_pixels):
            captured["buffer_pixels"] = buffer_pixels
            return original_generate_tiles(rows, cols, tile_size, buffer_pixels)

        with patch("solweig.tiling.generate_tiles", side_effect=spy_generate_tiles):
            _ = calculate_tiled(surface, location, weather, tile_size=350, max_shadow_distance_m=cap)

        if "buffer_pixels" in captured:
            expected_px = int(np.ceil(cap / surface.pixel_size))
            assert captured["buffer_pixels"] == expected_px, (
                f"Expected {expected_px}px buffer (capped at {cap}m), got {captured['buffer_pixels']}px"
            )
