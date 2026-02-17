"""
Unit tests for the orchestration layer.

Tests internal helper functions in computation.py, timeseries.py, and tiling.py
that aren't exercised by the higher-level integration tests.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import solweig
from solweig.bundles import GvfBundle, LupBundle
from solweig.computation import _apply_thermal_delay
from solweig.models.state import ThermalState, TileSpec
from solweig.tiling import (
    MAX_BUFFER_M,
    MIN_TILE_SIZE,
    _resolve_inflight_limit,
    _resolve_tile_workers,
    calculate_buffer_distance,
    compute_max_tile_pixels,
    compute_max_tile_side,
    generate_tiles,
    validate_tile_size,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_shape():
    return (10, 10)


@pytest.fixture()
def state_10x10(small_shape):
    """Fresh ThermalState for a 10×10 grid."""
    return ThermalState.initial(small_shape)


@pytest.fixture()
def gvf_bundle_10x10(small_shape):
    """GvfBundle with synthetic upwelling longwave values."""
    ones = np.ones(small_shape, dtype=np.float32)
    return GvfBundle(
        lup=ones * 400.0,
        lup_e=ones * 390.0,
        lup_s=ones * 395.0,
        lup_w=ones * 385.0,
        lup_n=ones * 380.0,
        gvfalb=ones * 0.15,
        gvfalb_e=ones * 0.15,
        gvfalb_s=ones * 0.15,
        gvfalb_w=ones * 0.15,
        gvfalb_n=ones * 0.15,
        gvfalbnosh=ones * 0.15,
        gvfalbnosh_e=ones * 0.15,
        gvfalbnosh_s=ones * 0.15,
        gvfalbnosh_w=ones * 0.15,
        gvfalbnosh_n=ones * 0.15,
    )


def _make_weather(*, ta: float = 20.0, sun_altitude: float = -5.0, is_daytime: bool = False) -> MagicMock:
    """Create a mock Weather object with controllable attributes."""
    w = MagicMock()
    w.ta = ta
    w.sun_altitude = sun_altitude
    w.is_daytime = is_daytime
    return w


# ---------------------------------------------------------------------------
# _apply_thermal_delay
# ---------------------------------------------------------------------------


class TestApplyThermalDelay:
    """Tests for _apply_thermal_delay() in computation.py."""

    def test_no_state_returns_raw_gvf(self, gvf_bundle_10x10):
        """Without state, raw GVF lup values are returned (no thermal delay)."""
        weather = _make_weather(ta=25.0, is_daytime=True)
        shadow = np.ones((10, 10), dtype=np.float32) * 0.5
        ground_tg = np.ones((10, 10), dtype=np.float32) * 2.0

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=None)

        assert isinstance(result, LupBundle)
        np.testing.assert_array_equal(result.lup, gvf_bundle_10x10.lup)
        np.testing.assert_array_equal(result.lup_e, gvf_bundle_10x10.lup_e)
        np.testing.assert_array_equal(result.lup_n, gvf_bundle_10x10.lup_n)
        assert result.state is None

    def test_no_state_output_is_float32(self, gvf_bundle_10x10):
        """Output arrays are float32 even without state."""
        weather = _make_weather(ta=25.0)
        shadow = np.ones((10, 10), dtype=np.float32)
        ground_tg = np.ones((10, 10), dtype=np.float32)

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=None)

        assert result.lup.dtype == np.float32
        assert result.lup_e.dtype == np.float32

    @patch("solweig.computation.ground_rust.ts_wave_delay_batch")
    def test_with_state_calls_rust(self, mock_ts_wave, gvf_bundle_10x10, state_10x10):
        """With state, calls Rust ts_wave_delay_batch and returns delayed values."""
        # Set up mock Rust result
        shape = (10, 10)
        mock_result = SimpleNamespace(
            lup=np.ones(shape, dtype=np.float32) * 410.0,
            lup_e=np.ones(shape, dtype=np.float32) * 405.0,
            lup_s=np.ones(shape, dtype=np.float32) * 400.0,
            lup_w=np.ones(shape, dtype=np.float32) * 395.0,
            lup_n=np.ones(shape, dtype=np.float32) * 390.0,
            timeadd=0.5,
            tgmap1=np.ones(shape, dtype=np.float32) * 1.0,
            tgmap1_e=np.ones(shape, dtype=np.float32) * 1.1,
            tgmap1_s=np.ones(shape, dtype=np.float32) * 1.2,
            tgmap1_w=np.ones(shape, dtype=np.float32) * 1.3,
            tgmap1_n=np.ones(shape, dtype=np.float32) * 1.4,
            tgout1=np.ones(shape, dtype=np.float32) * 2.0,
        )
        mock_ts_wave.return_value = mock_result

        weather = _make_weather(ta=25.0, is_daytime=True)
        shadow = np.ones(shape, dtype=np.float32) * 0.5
        ground_tg = np.ones(shape, dtype=np.float32) * 2.0

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=state_10x10)

        mock_ts_wave.assert_called_once()
        np.testing.assert_allclose(result.lup, 410.0)
        np.testing.assert_allclose(result.lup_e, 405.0)
        assert result.state is not None

    @patch("solweig.computation.ground_rust.ts_wave_delay_batch")
    def test_state_updated_from_rust_result(self, mock_ts_wave, gvf_bundle_10x10, state_10x10):
        """State arrays are updated from Rust result."""
        shape = (10, 10)
        mock_result = SimpleNamespace(
            lup=np.ones(shape, dtype=np.float32),
            lup_e=np.ones(shape, dtype=np.float32),
            lup_s=np.ones(shape, dtype=np.float32),
            lup_w=np.ones(shape, dtype=np.float32),
            lup_n=np.ones(shape, dtype=np.float32),
            timeadd=0.75,
            tgmap1=np.full(shape, 3.0, dtype=np.float32),
            tgmap1_e=np.full(shape, 3.1, dtype=np.float32),
            tgmap1_s=np.full(shape, 3.2, dtype=np.float32),
            tgmap1_w=np.full(shape, 3.3, dtype=np.float32),
            tgmap1_n=np.full(shape, 3.4, dtype=np.float32),
            tgout1=np.full(shape, 4.0, dtype=np.float32),
        )
        mock_ts_wave.return_value = mock_result

        weather = _make_weather(ta=20.0, is_daytime=True)
        shadow = np.ones(shape, dtype=np.float32)
        ground_tg = np.zeros(shape, dtype=np.float32)

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=state_10x10)

        # Output state has values from Rust result
        assert result.state is not None
        np.testing.assert_allclose(result.state.tgmap1, 3.0)
        np.testing.assert_allclose(result.state.tgmap1_e, 3.1)
        np.testing.assert_allclose(result.state.tgout1, 4.0)

    @patch("solweig.computation.ground_rust.ts_wave_delay_batch")
    def test_daytime_clears_firstdaytime(self, mock_ts_wave, gvf_bundle_10x10, state_10x10):
        """During daytime, firstdaytime is set to 0.0."""
        shape = (10, 10)
        mock_result = SimpleNamespace(
            lup=np.ones(shape, dtype=np.float32),
            lup_e=np.ones(shape, dtype=np.float32),
            lup_s=np.ones(shape, dtype=np.float32),
            lup_w=np.ones(shape, dtype=np.float32),
            lup_n=np.ones(shape, dtype=np.float32),
            timeadd=0.5,
            tgmap1=np.zeros(shape, dtype=np.float32),
            tgmap1_e=np.zeros(shape, dtype=np.float32),
            tgmap1_s=np.zeros(shape, dtype=np.float32),
            tgmap1_w=np.zeros(shape, dtype=np.float32),
            tgmap1_n=np.zeros(shape, dtype=np.float32),
            tgout1=np.zeros(shape, dtype=np.float32),
        )
        mock_ts_wave.return_value = mock_result

        state_10x10.firstdaytime = 1.0  # Morning state
        weather = _make_weather(ta=25.0, is_daytime=True)
        shadow = np.ones(shape, dtype=np.float32)
        ground_tg = np.zeros(shape, dtype=np.float32)

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=state_10x10)

        assert result.state is not None
        assert result.state.firstdaytime == 0.0

    @patch("solweig.computation.ground_rust.ts_wave_delay_batch")
    def test_nighttime_resets_firstdaytime_and_timeadd(self, mock_ts_wave, gvf_bundle_10x10, state_10x10):
        """At night, firstdaytime resets to 1.0 and timeadd resets to 0.0."""
        shape = (10, 10)
        mock_result = SimpleNamespace(
            lup=np.ones(shape, dtype=np.float32),
            lup_e=np.ones(shape, dtype=np.float32),
            lup_s=np.ones(shape, dtype=np.float32),
            lup_w=np.ones(shape, dtype=np.float32),
            lup_n=np.ones(shape, dtype=np.float32),
            timeadd=0.75,
            tgmap1=np.zeros(shape, dtype=np.float32),
            tgmap1_e=np.zeros(shape, dtype=np.float32),
            tgmap1_s=np.zeros(shape, dtype=np.float32),
            tgmap1_w=np.zeros(shape, dtype=np.float32),
            tgmap1_n=np.zeros(shape, dtype=np.float32),
            tgout1=np.zeros(shape, dtype=np.float32),
        )
        mock_ts_wave.return_value = mock_result

        state_10x10.firstdaytime = 0.0
        state_10x10.timeadd = 5.0
        weather = _make_weather(ta=15.0, is_daytime=False)
        shadow = np.ones(shape, dtype=np.float32)
        ground_tg = np.zeros(shape, dtype=np.float32)

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=state_10x10)

        assert result.state is not None
        assert result.state.firstdaytime == 1.0
        assert result.state.timeadd == 0.0

    @patch("solweig.computation.ground_rust.ts_wave_delay_batch")
    def test_output_state_is_copy(self, mock_ts_wave, gvf_bundle_10x10, state_10x10):
        """Output state is a deep copy (mutating doesn't affect internal state)."""
        shape = (10, 10)
        mock_result = SimpleNamespace(
            lup=np.ones(shape, dtype=np.float32),
            lup_e=np.ones(shape, dtype=np.float32),
            lup_s=np.ones(shape, dtype=np.float32),
            lup_w=np.ones(shape, dtype=np.float32),
            lup_n=np.ones(shape, dtype=np.float32),
            timeadd=0.5,
            tgmap1=np.ones(shape, dtype=np.float32),
            tgmap1_e=np.ones(shape, dtype=np.float32),
            tgmap1_s=np.ones(shape, dtype=np.float32),
            tgmap1_w=np.ones(shape, dtype=np.float32),
            tgmap1_n=np.ones(shape, dtype=np.float32),
            tgout1=np.ones(shape, dtype=np.float32),
        )
        mock_ts_wave.return_value = mock_result

        weather = _make_weather(ta=20.0, is_daytime=True)
        shadow = np.ones(shape, dtype=np.float32)
        ground_tg = np.zeros(shape, dtype=np.float32)

        result = _apply_thermal_delay(gvf_bundle_10x10, ground_tg, shadow, weather, state=state_10x10)

        # Mutating output state doesn't affect input
        assert result.state is not None
        result.state.tgmap1[:] = 999.0
        assert not np.any(state_10x10.tgmap1 == 999.0)


# ---------------------------------------------------------------------------
# _precompute_weather
# ---------------------------------------------------------------------------


class TestPrecomputeWeather:
    """Tests for _precompute_weather() in timeseries.py."""

    def test_empty_list_noop(self):
        """Empty weather series is a no-op."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        _precompute_weather([], location)  # Should not raise

    def test_assigns_precomputed_altmax(self):
        """Each weather object gets precomputed_altmax assigned."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        dt_noon = datetime(2024, 7, 15, 12, 0)
        w = solweig.Weather(datetime=dt_noon, ta=25.0, rh=50.0, global_rad=800.0)

        _precompute_weather([w], location)

        assert hasattr(w, "precomputed_altmax")
        assert w.precomputed_altmax is not None
        assert w.precomputed_altmax > 0  # Summer noon in Sweden: ~55°

    def test_same_day_shares_altmax(self):
        """Multiple timesteps on the same day share the same altmax value."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        base = datetime(2024, 7, 15)

        weather_series = [
            solweig.Weather(datetime=base.replace(hour=h), ta=25.0, rh=50.0, global_rad=800.0) for h in range(6, 18)
        ]

        _precompute_weather(weather_series, location)

        altmaxes = [w.precomputed_altmax for w in weather_series]
        assert all(a == altmaxes[0] for a in altmaxes)

    def test_different_days_may_differ(self):
        """Different days may have different altmax (season effect)."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)

        w_summer = solweig.Weather(datetime=datetime(2024, 6, 21, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)
        w_winter = solweig.Weather(datetime=datetime(2024, 12, 21, 12, 0), ta=0.0, rh=70.0, global_rad=200.0)

        _precompute_weather([w_summer, w_winter], location)

        assert w_summer.precomputed_altmax is not None
        assert w_winter.precomputed_altmax is not None
        assert w_summer.precomputed_altmax > w_winter.precomputed_altmax

    def test_marks_derived_computed(self):
        """After precomputation, weather objects have derived values computed."""
        from solweig.timeseries import _precompute_weather

        location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=2)
        w = solweig.Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0)

        assert not w._derived_computed
        _precompute_weather([w], location)
        assert w._derived_computed


# ---------------------------------------------------------------------------
# ThermalState
# ---------------------------------------------------------------------------


class TestThermalState:
    """Tests for ThermalState creation and copying."""

    def test_initial_state_shape(self, small_shape):
        """Initial state arrays have the requested shape."""
        state = ThermalState.initial(small_shape)
        assert state.tgmap1.shape == small_shape
        assert state.tgmap1_e.shape == small_shape
        assert state.tgmap1_s.shape == small_shape
        assert state.tgmap1_w.shape == small_shape
        assert state.tgmap1_n.shape == small_shape
        assert state.tgout1.shape == small_shape

    def test_initial_state_zeros(self, small_shape):
        """Initial state arrays are all zeros."""
        state = ThermalState.initial(small_shape)
        np.testing.assert_array_equal(state.tgmap1, 0.0)
        np.testing.assert_array_equal(state.tgout1, 0.0)

    def test_initial_state_flags(self, small_shape):
        """Initial state has firstdaytime=1.0, timeadd=0.0."""
        state = ThermalState.initial(small_shape)
        assert state.firstdaytime == 1.0
        assert state.timeadd == 0.0
        assert state.timestep_dec == 0.0

    def test_copy_is_independent(self, small_shape):
        """Copy creates independent arrays."""
        state = ThermalState.initial(small_shape)
        state.tgmap1[:] = 5.0
        state.firstdaytime = 0.0

        copy = state.copy()
        copy.tgmap1[:] = 99.0
        copy.firstdaytime = 1.0

        assert state.tgmap1[0, 0] == 5.0  # Original unchanged
        assert state.firstdaytime == 0.0  # Original unchanged

    def test_copy_preserves_values(self, small_shape):
        """Copy preserves all field values."""
        state = ThermalState.initial(small_shape)
        state.tgmap1[:] = 3.0
        state.timeadd = 1.5
        state.timestep_dec = 0.042

        copy = state.copy()
        np.testing.assert_array_equal(copy.tgmap1, 3.0)
        assert copy.timeadd == 1.5
        assert copy.timestep_dec == 0.042


# ---------------------------------------------------------------------------
# TileSpec
# ---------------------------------------------------------------------------


class TestTileSpec:
    """Tests for TileSpec properties."""

    def test_core_shape(self):
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=200,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=250,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        assert tile.core_shape == (100, 200)

    def test_full_shape(self):
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=200,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=250,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        assert tile.full_shape == (150, 250)

    def test_core_slice_no_overlap(self):
        """First tile (no top/left overlap) has core starting at (0, 0)."""
        tile = TileSpec(
            row_start=0,
            row_end=100,
            col_start=0,
            col_end=100,
            row_start_full=0,
            row_end_full=150,
            col_start_full=0,
            col_end_full=150,
            overlap_top=0,
            overlap_bottom=50,
            overlap_left=0,
            overlap_right=50,
        )
        rs, cs = tile.core_slice
        assert rs == slice(0, 100)
        assert cs == slice(0, 100)

    def test_core_slice_with_overlap(self):
        """Middle tile has core offset by overlap."""
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=100,
            col_end=200,
            row_start_full=50,
            row_end_full=250,
            col_start_full=50,
            col_end_full=250,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.core_slice
        assert rs == slice(50, 150)
        assert cs == slice(50, 150)

    def test_write_slice(self):
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=50,
            col_end=150,
            row_start_full=50,
            row_end_full=250,
            col_start_full=0,
            col_end_full=200,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.write_slice
        assert rs == slice(100, 200)
        assert cs == slice(50, 150)

    def test_read_slice(self):
        tile = TileSpec(
            row_start=100,
            row_end=200,
            col_start=50,
            col_end=150,
            row_start_full=50,
            row_end_full=250,
            col_start_full=0,
            col_end_full=200,
            overlap_top=50,
            overlap_bottom=50,
            overlap_left=50,
            overlap_right=50,
        )
        rs, cs = tile.read_slice
        assert rs == slice(50, 250)
        assert cs == slice(0, 200)


# ---------------------------------------------------------------------------
# calculate_buffer_distance
# ---------------------------------------------------------------------------


class TestCalculateBufferDistance:
    """Tests for calculate_buffer_distance() in tiling.py."""

    def test_zero_height(self):
        assert calculate_buffer_distance(0.0) == 0.0

    def test_negative_height(self):
        assert calculate_buffer_distance(-5.0) == 0.0

    def test_short_building(self):
        """10m building: 10 / tan(3°) ≈ 190.8m."""
        result = calculate_buffer_distance(10.0)
        assert 190 < result < 192

    def test_tall_building_capped(self):
        """60m building would need ~1145m, capped at 1000m."""
        result = calculate_buffer_distance(60.0)
        assert result == MAX_BUFFER_M

    def test_custom_min_elevation(self):
        """Higher min elevation reduces buffer distance."""
        buffer_3 = calculate_buffer_distance(10.0, min_sun_elev_deg=3.0)
        buffer_10 = calculate_buffer_distance(10.0, min_sun_elev_deg=10.0)
        assert buffer_10 < buffer_3


# ---------------------------------------------------------------------------
# validate_tile_size
# ---------------------------------------------------------------------------


class TestComputeMaxTilePixels:
    """Tests for compute_max_tile_pixels() in tiling.py."""

    def test_backend_without_hint_uses_single_buffer_estimate(self, monkeypatch):
        """Unknown backend should use largest-single-buffer bytes/pixel estimate."""
        max_buf = 1_000_000_000  # bytes (1 GB — large enough to exceed MIN_TILE_SIZE²)
        headroom = compute_max_tile_pixels.__globals__["_GPU_HEADROOM"]

        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: {"max_buffer_size": max_buf})
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        svf_pixels = compute_max_tile_pixels(context="svf")
        solweig_pixels = compute_max_tile_pixels(context="solweig")

        expected_svf = int(max_buf * headroom) // 60  # _SVF_GPU_SINGLE_BPP
        expected_solweig = int(max_buf * headroom) // 40  # _SHADOW_GPU_SINGLE_BPP

        assert svf_pixels == expected_svf
        assert solweig_pixels == expected_solweig
        assert svf_pixels < solweig_pixels

    def test_metal_backend_uses_total_working_set_estimate(self, monkeypatch):
        """Metal backend should constrain by aggregate GPU working-set bytes/pixel."""
        max_buf = 1_000_000_000  # bytes (1 GB — large enough to exceed MIN_TILE_SIZE²)
        headroom = compute_max_tile_pixels.__globals__["_GPU_HEADROOM"]

        monkeypatch.setattr(solweig, "get_gpu_limits", lambda: {"max_buffer_size": max_buf, "backend": "Metal"})
        monkeypatch.setitem(compute_max_tile_pixels.__globals__, "_get_total_ram_bytes", lambda: None)

        svf_pixels = compute_max_tile_pixels(context="svf")
        solweig_pixels = compute_max_tile_pixels(context="solweig")

        expected_svf = int(max_buf * headroom) // 384  # _SVF_GPU_TOTAL_BPP
        expected_solweig = int(max_buf * headroom) // 120  # _SHADOW_GPU_TOTAL_BPP

        assert svf_pixels == expected_svf
        assert solweig_pixels == expected_solweig
        assert svf_pixels < solweig_pixels


class TestValidateTileSize:
    """Tests for validate_tile_size() — tile_size is core (excluding overlap)."""

    def test_valid_core_passes(self):
        """Core size that fits with buffer within resource limit passes unchanged."""
        adjusted, warning = validate_tile_size(800, buffer_pixels=50, pixel_size=1.0)
        assert adjusted == 800
        assert warning is None

    def test_below_minimum_adjusted(self):
        adjusted, warning = validate_tile_size(100, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == MIN_TILE_SIZE
        assert warning is not None
        assert "below minimum" in warning

    def test_core_plus_buffer_exceeds_limit(self):
        """Core is reduced so core + 2*buffer fits within resource limit."""
        max_full = compute_max_tile_side(context="solweig")
        buffer_pixels = 50
        # Request core that with buffer would exceed limit
        adjusted, warning = validate_tile_size(max_full, buffer_pixels=buffer_pixels, pixel_size=1.0)
        assert adjusted == max_full - 2 * buffer_pixels
        assert warning is not None
        assert "exceeds resource limit" in warning

    def test_large_buffer_allows_subminimum_core_to_respect_limit(self):
        """When overlap is huge, core may drop below MIN_TILE_SIZE to keep full tile valid."""
        max_full = compute_max_tile_side(context="solweig")
        # Buffer so large that max_core < MIN_TILE_SIZE
        huge_buffer = (max_full - MIN_TILE_SIZE) // 2 + 100
        adjusted, warning = validate_tile_size(800, buffer_pixels=huge_buffer, pixel_size=1.0)
        max_core = max_full - 2 * huge_buffer
        assert adjusted == max(1, max_core)
        assert warning is not None

    def test_exact_minimum(self):
        adjusted, warning = validate_tile_size(MIN_TILE_SIZE, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == MIN_TILE_SIZE
        assert warning is None

    def test_small_buffer_allows_large_core(self):
        """With small buffer, core can use nearly all of the resource limit."""
        max_full = compute_max_tile_side(context="solweig")
        max_core = max_full - 2 * 10
        adjusted, warning = validate_tile_size(max_core, buffer_pixels=10, pixel_size=1.0)
        assert adjusted == max_core
        assert warning is None

    def test_context_uses_context_specific_limits(self, monkeypatch):
        """SVF context should not be constrained by SOLWEIG tile limits."""

        def _fake_max_side(*, context: str = "solweig"):
            return 4000 if context == "svf" else 1000

        # Patch directly on the function's globals dict.  QGIS mock imports can
        # cause a double-load of solweig.tiling so the module object in
        # sys.modules may differ from the one validate_tile_size was defined in.
        monkeypatch.setitem(
            validate_tile_size.__globals__,
            "compute_max_tile_side",
            _fake_max_side,
        )

        svf_adjusted, _ = validate_tile_size(950, buffer_pixels=50, pixel_size=1.0, context="svf")
        solweig_adjusted, _ = validate_tile_size(950, buffer_pixels=50, pixel_size=1.0, context="solweig")

        assert svf_adjusted == 950
        assert solweig_adjusted == 900


# ---------------------------------------------------------------------------
# generate_tiles
# ---------------------------------------------------------------------------


class TestGenerateTiles:
    """Tests for generate_tiles() in tiling.py."""

    def test_single_tile(self):
        """Small raster fits in one tile."""
        tiles = generate_tiles(100, 100, tile_size=256, overlap=50)
        assert len(tiles) == 1
        assert tiles[0].row_start == 0
        assert tiles[0].row_end == 100
        assert tiles[0].col_start == 0
        assert tiles[0].col_end == 100

    def test_single_tile_no_overlap(self):
        """Single tile has no overlap (no neighbors)."""
        tiles = generate_tiles(100, 100, tile_size=256, overlap=50)
        assert tiles[0].overlap_top == 0
        assert tiles[0].overlap_bottom == 0
        assert tiles[0].overlap_left == 0
        assert tiles[0].overlap_right == 0

    def test_2x2_tiles(self):
        """500x500 raster with 256 tile size creates 2x2 grid."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        assert len(tiles) == 4

    def test_tiles_cover_entire_raster(self):
        """All pixels are covered by at least one tile's core area."""
        rows, cols = 500, 700
        tiles = generate_tiles(rows, cols, tile_size=256, overlap=50)

        covered = np.zeros((rows, cols), dtype=bool)
        for tile in tiles:
            covered[tile.row_start : tile.row_end, tile.col_start : tile.col_end] = True

        assert np.all(covered)

    def test_overlap_only_on_inner_edges(self):
        """Edge tiles don't extend beyond raster bounds."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)

        for tile in tiles:
            assert tile.row_start_full >= 0
            assert tile.row_end_full <= 500
            assert tile.col_start_full >= 0
            assert tile.col_end_full <= 500

    def test_first_tile_no_top_left_overlap(self):
        """Top-left tile has no top or left overlap."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        first = tiles[0]
        assert first.overlap_top == 0
        assert first.overlap_left == 0

    def test_last_tile_no_bottom_right_overlap(self):
        """Bottom-right tile has no bottom or right overlap."""
        tiles = generate_tiles(500, 500, tile_size=256, overlap=50)
        last = tiles[-1]
        assert last.overlap_bottom == 0
        assert last.overlap_right == 0

    def test_middle_tile_has_all_overlaps(self):
        """Middle tile in a 3x3 grid has overlap on all sides."""
        tiles = generate_tiles(768, 768, tile_size=256, overlap=50)
        assert len(tiles) == 9  # 3x3

        middle = tiles[4]  # center tile
        assert middle.overlap_top == 50
        assert middle.overlap_bottom == 50
        assert middle.overlap_left == 50
        assert middle.overlap_right == 50

    def test_non_square_raster(self):
        """Non-square raster creates asymmetric tile grid."""
        tiles = generate_tiles(rows=200, cols=600, tile_size=256, overlap=30)
        assert len(tiles) == 3  # 1 row × 3 cols

    def test_exact_tile_size_fit(self):
        """Raster exactly matching tile size creates exactly 1 tile."""
        tiles = generate_tiles(256, 256, tile_size=256, overlap=50)
        assert len(tiles) == 1


# ---------------------------------------------------------------------------
# Tiling runtime controls
# ---------------------------------------------------------------------------


class TestTilingRuntimeControls:
    """Tests for worker and queue-depth resolution helpers."""

    def test_resolve_tile_workers_clamps_to_tile_count(self):
        assert _resolve_tile_workers(tile_workers=16, n_tiles=3) == 3

    def test_resolve_tile_workers_zero_raises(self):
        with pytest.raises(ValueError, match="tile_workers must be >= 1"):
            _resolve_tile_workers(tile_workers=0, n_tiles=8)

    def test_resolve_inflight_limit_with_prefetch_default(self):
        # queue_depth=None + prefetch=True => queue_depth = n_workers
        assert _resolve_inflight_limit(4, n_tiles=20, tile_queue_depth=None, prefetch_tiles=True) == 8

    def test_resolve_inflight_limit_no_prefetch(self):
        # queue_depth=None + prefetch=False => no queued tasks
        assert _resolve_inflight_limit(4, n_tiles=20, tile_queue_depth=None, prefetch_tiles=False) == 4

    def test_resolve_inflight_limit_clamped_to_n_tiles(self):
        assert _resolve_inflight_limit(4, n_tiles=5, tile_queue_depth=8, prefetch_tiles=True) == 5

    def test_resolve_inflight_limit_negative_queue_depth_raises(self):
        with pytest.raises(ValueError, match="tile_queue_depth must be >= 0"):
            _resolve_inflight_limit(2, n_tiles=8, tile_queue_depth=-1, prefetch_tiles=True)
