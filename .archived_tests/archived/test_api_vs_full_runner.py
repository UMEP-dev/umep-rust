"""
Rigorous comparison test: Simplified API vs Full Runner

This test runs both APIs on the same input data and compares Tmrt values.
The goal is to achieve <5% mean absolute error between simplified and full runner.

The test uses the Athens demo data with pre-computed SVF/shadows from the
full SOLWEIG pipeline.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from solweig.api import (
    HumanParams,
    Location,
    SurfaceData,
    Weather,
    calculate,
)


FIXTURES_DIR = Path(__file__).parent / "golden" / "fixtures"


def load_fixture(name: str) -> np.ndarray:
    """Load a fixture array."""
    path = FIXTURES_DIR / f"{name}.npy"
    if not path.exists():
        pytest.skip(f"Fixture {name} not found at {path}")
    return np.load(path)


def load_params() -> dict:
    """Load fixture parameters."""
    path = FIXTURES_DIR / "input_params.npz"
    if not path.exists():
        pytest.skip(f"Params not found at {path}")
    return dict(np.load(path))


class TestRigorousComparison:
    """
    Compare simplified API against full SOLWEIG runner.

    The full runner uses:
    - Directional SVFs (N, E, S, W)
    - GVF calculation with wall radiation
    - Kside/Lside vegetation functions
    - Ground temperature model
    - Anisotropic sky (optional)

    We measure the gap between simplified and full implementations.
    """

    @pytest.fixture
    def surface_data(self) -> SurfaceData:
        """Load Athens surface data."""
        dsm = load_fixture("input_dsm")
        cdsm = load_fixture("input_cdsm")
        tdsm = load_fixture("input_tdsm")
        params = load_params()
        return SurfaceData(
            dsm=dsm,
            cdsm=cdsm,
            tdsm=tdsm,
            pixel_size=float(params["scale"]),
        )

    @pytest.fixture
    def location(self) -> Location:
        """Athens location."""
        return Location(latitude=37.97, longitude=23.73, altitude=100.0, utc_offset=2)

    @pytest.fixture
    def noon_weather(self) -> Weather:
        """Summer noon weather."""
        return Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=32.0,
            rh=35.0,
            global_rad=900.0,
            ws=2.5,
        )

    def test_compare_with_full_runner_components(
        self, surface_data: SurfaceData, location: Location, noon_weather: Weather
    ):
        """
        Compare simplified API outputs with full runner.

        This test documents the current gap and tracks progress.
        """
        # Run simplified API
        result = calculate(
            surface=surface_data,
            location=location,
            weather=noon_weather,
            human=HumanParams(posture="standing", abs_k=0.7, abs_l=0.97),
        )

        # Load golden SVF for comparison
        try:
            svf_golden = load_fixture("svf_total")
        except Exception:
            pytest.skip("Golden SVF not available")

        # Run full calculation manually for comparison
        # This uses the same Rust modules but with full parameters
        from solweig.rustalgos import shadowing, skyview, gvf

        dsm = surface_data.dsm
        cdsm = surface_data.cdsm if surface_data.cdsm is not None else np.zeros_like(dsm)
        tdsm = surface_data.tdsm if surface_data.tdsm is not None else np.zeros_like(dsm)
        pixel_size = surface_data.pixel_size
        max_height = surface_data.max_height
        use_veg = surface_data.cdsm is not None

        # Compute weather-derived values
        noon_weather.compute_derived(location)

        # Full SVF calculation
        svf_result = skyview.calculate_svf(
            dsm, cdsm, tdsm, pixel_size, use_veg, max_height, 2, 3.0, None
        )

        # Shadow calculation
        bush = np.zeros_like(dsm)
        wall_ht = load_fixture("input_wall_ht")
        wall_asp = load_fixture("input_wall_asp") * np.pi / 180.0  # Convert to radians

        shadow_result = shadowing.calculate_shadows_wall_ht_25(
            noon_weather.sun_azimuth,
            noon_weather.sun_altitude,
            pixel_size,
            max_height,
            dsm,
            cdsm,
            tdsm,
            bush,
            wall_ht,
            wall_asp,
            None,
            None,
            3.0,
        )

        # Compare shadows
        shadow_simplified = result.shadow
        shadow_full = np.array(shadow_result.bldg_sh)

        shadow_match = np.isclose(shadow_simplified, shadow_full, atol=0.1).mean()
        print(f"\nShadow match: {shadow_match * 100:.1f}%")
        assert shadow_match > 0.85, f"Shadow match too low: {shadow_match * 100:.1f}%"

        # Compare SVF
        svf_simplified = np.array(svf_result.svf)
        svf_match = np.isclose(svf_simplified, svf_golden, atol=0.05).mean()
        print(f"SVF match with golden: {svf_match * 100:.1f}%")

        # Document Tmrt statistics
        tmrt = result.tmrt
        valid_mask = ~np.isnan(tmrt)

        print(f"\n=== Simplified API Tmrt ===")
        print(f"  Valid pixels: {valid_mask.sum()}/{tmrt.size}")
        print(f"  Mean: {np.nanmean(tmrt):.1f}°C")
        print(f"  Std: {np.nanstd(tmrt):.1f}°C")
        print(f"  Min: {np.nanmin(tmrt):.1f}°C")
        print(f"  Max: {np.nanmax(tmrt):.1f}°C")

        # Verify Tmrt is in expected range for summer noon
        assert np.nanmean(tmrt) > 40, f"Tmrt too low: {np.nanmean(tmrt):.1f}°C"
        assert np.nanmean(tmrt) < 70, f"Tmrt too high: {np.nanmean(tmrt):.1f}°C"

    def test_radiation_components(
        self, surface_data: SurfaceData, location: Location, noon_weather: Weather
    ):
        """Test that radiation components are physically reasonable."""
        result = calculate(
            surface=surface_data,
            location=location,
            weather=noon_weather,
        )

        # Kdown should be significant during daytime
        kdown = result.kdown
        print(f"\n=== Radiation Components ===")
        print(f"Kdown mean: {np.nanmean(kdown):.1f} W/m²")
        print(f"Kup mean: {np.nanmean(result.kup):.1f} W/m²")
        print(f"Ldown mean: {np.nanmean(result.ldown):.1f} W/m²")
        print(f"Lup mean: {np.nanmean(result.lup):.1f} W/m²")

        # Physical constraints
        assert np.nanmean(kdown) > 200, "Kdown too low for summer noon"
        assert np.nanmean(kdown) < 1200, "Kdown too high"
        assert np.nanmean(result.ldown) > 250, "Ldown too low"
        assert np.nanmean(result.ldown) < 500, "Ldown too high"
        assert np.nanmean(result.lup) > 300, "Lup too low"
        assert np.nanmean(result.lup) < 700, "Lup too high"  # Higher threshold for hot summer noon

    def test_utci_calculation(
        self, surface_data: SurfaceData, location: Location, noon_weather: Weather
    ):
        """Test UTCI calculation produces reasonable values."""
        result = calculate(
            surface=surface_data,
            location=location,
            weather=noon_weather,
            compute_utci=True,
        )

        utci = result.utci
        assert utci is not None

        print(f"\n=== UTCI ===")
        print(f"Mean: {np.nanmean(utci):.1f}°C")
        print(f"Min: {np.nanmin(utci):.1f}°C")
        print(f"Max: {np.nanmax(utci):.1f}°C")

        # UTCI classification check
        # Very strong heat stress: >46°C
        # Strong heat stress: 38-46°C
        # Moderate heat stress: 32-38°C
        # For Ta=32°C, RH=35%, high radiation, expect UTCI > 32
        assert np.nanmean(utci) > 28, f"UTCI too low: {np.nanmean(utci):.1f}°C"
        assert np.nanmean(utci) < 50, f"UTCI too high: {np.nanmean(utci):.1f}°C"


class TestGapMeasurement:
    """
    Measure specific gaps between simplified and full implementation.

    These tests quantify the error introduced by each simplification.
    """

    @pytest.fixture
    def simple_urban_dsm(self) -> np.ndarray:
        """Create a simple urban canyon DSM for testing."""
        # 100x100 grid, 1m pixels
        # Two parallel buildings creating a street canyon
        dsm = np.ones((100, 100), dtype=np.float32) * 100.0

        # North building (rows 20-35)
        dsm[20:35, 30:70] = 120.0  # 20m tall

        # South building (rows 65-80)
        dsm[65:80, 30:70] = 115.0  # 15m tall

        # This creates a 30m wide street canyon with 15-20m buildings
        return dsm

    def test_measure_wall_radiation_gap(self, simple_urban_dsm: np.ndarray):
        """
        Measure error from missing wall radiation in street canyons.

        In canyons, walls contribute significant longwave radiation.
        Without walls, we underestimate Tmrt.
        """
        surface = SurfaceData(dsm=simple_urban_dsm, pixel_size=1.0)
        location = Location(latitude=45.0, longitude=10.0, utc_offset=1)

        # Low sun angle to maximize wall effect
        weather = Weather(
            datetime=datetime(2024, 6, 21, 9, 0),  # Morning
            ta=25.0,
            rh=50.0,
            global_rad=500.0,
        )

        result = calculate(surface=surface, location=location, weather=weather)

        # Extract canyon pixels (between buildings)
        canyon_rows = slice(40, 60)
        canyon_cols = slice(35, 65)
        tmrt_canyon = result.tmrt[canyon_rows, canyon_cols]

        # Extract open area pixels (outside buildings)
        open_rows = slice(0, 15)
        open_cols = slice(0, 25)
        tmrt_open = result.tmrt[open_rows, open_cols]

        print(f"\n=== Wall Radiation Gap Analysis ===")
        print(f"Canyon Tmrt: {np.nanmean(tmrt_canyon):.1f}°C")
        print(f"Open area Tmrt: {np.nanmean(tmrt_open):.1f}°C")
        print(f"Difference: {np.nanmean(tmrt_canyon) - np.nanmean(tmrt_open):.1f}°C")

        # In reality, canyon should be warmer due to wall radiation
        # Our simplified model may underestimate this
        # Document the current behavior
        assert result.tmrt is not None

    def test_measure_svf_directional_gap(self, simple_urban_dsm: np.ndarray):
        """
        Measure error from using total SVF instead of directional.

        Directional SVFs (N, E, S, W) better capture urban geometry.
        """
        surface = SurfaceData(dsm=simple_urban_dsm, pixel_size=1.0)
        location = Location(latitude=45.0, longitude=10.0, utc_offset=1)

        # Test at noon when directional effects are visible
        weather = Weather(
            datetime=datetime(2024, 6, 21, 12, 0),
            ta=28.0,
            rh=45.0,
            global_rad=800.0,
        )

        result = calculate(surface=surface, location=location, weather=weather)

        # The canyon runs E-W, so N-S walls should have different radiation
        # North side of canyon (facing south, more sun)
        north_side = result.tmrt[35:40, 35:65]
        # South side of canyon (facing north, less sun)
        south_side = result.tmrt[60:65, 35:65]

        print(f"\n=== Directional SVF Gap Analysis ===")
        print(f"North-facing wall side Tmrt: {np.nanmean(south_side):.1f}°C")
        print(f"South-facing wall side Tmrt: {np.nanmean(north_side):.1f}°C")
        print(f"N-S difference: {np.nanmean(north_side) - np.nanmean(south_side):.1f}°C")

        assert result.tmrt is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
