"""
Field-data validation tests using the Zenodo SOLWEIG v2025 Kolumbus dataset.

Dataset: Wallenberg et al. (2025) - Wall surface temperature validation
Source: https://zenodo.org/records/15309445
Location: Gothenburg, Sweden (57.697°N, 11.930°E)
Period: 2023-05-15 to 2023-08-31 (10-minute intervals)
Measurements: IR radiometer wall surface temperatures (plastered brick + wood)

These tests validate the SOLWEIG wall temperature model (tg_wall) against
field measurements of wall surface temperature. With JSON-based wall params,
material-specific parameters can be passed (e.g., brick vs wood).

Tests are marked @pytest.mark.slow and @pytest.mark.validation since they
require external data files and take significant time.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if validation data not present
DATA_DIR = Path(__file__).parent.parent / "validation_data" / "zenodo_kolumbus"
pytestmark = [
    pytest.mark.skipif(not DATA_DIR.exists(), reason="Zenodo validation data not downloaded"),
    pytest.mark.validation,
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_kolumbus_observations(
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """Load kolumbus.csv wall temperature observations.

    Returns list of dicts with keys:
        time, ta, ts_pb_sim, kin_pb_sim, lin_pb_sim, ts_pb_obs,
        ts_wood_sim, kin_wood_sim, lin_wood_sim, ts_wood_obs
    """
    csv_path = DATA_DIR / "kolumbus.csv"
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None
    # Make end inclusive of the full day
    if end_dt and end_dt.hour == 0 and end_dt.minute == 0:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = datetime.fromisoformat(row["Time"])
            if start_dt and timestamp < start_dt:
                continue
            if end_dt and timestamp > end_dt:
                continue
            rows.append(
                {
                    "time": timestamp,
                    "ta": float(row["Ta"]),
                    "ts_pb_sim": float(row["Ts_pb_sim"]),
                    "kin_pb_sim": float(row["Kin_pb_sim"]),
                    "lin_pb_sim": float(row["Lin_pb_sim"]),
                    "ts_pb_obs": float(row["Ts_pb_obs"]),
                    "ts_wood_sim": float(row["Ts_wood_sim"]),
                    "kin_wood_sim": float(row["Kin_wood_sim"]),
                    "lin_wood_sim": float(row["Lin_wood_sim"]),
                    "ts_wood_obs": float(row["Ts_wood_obs"]),
                }
            )
    return rows


def load_hourly_observations(start: str, end: str) -> list[dict]:
    """Load kolumbus observations, keeping only on-the-hour rows."""
    obs = load_kolumbus_observations(start=start, end=end)
    return [o for o in obs if o["time"].minute == 0]


# ---------------------------------------------------------------------------
# Test: Data loading and sanity checks
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that the validation data loads correctly."""

    def test_kolumbus_csv_loads(self):
        obs = load_kolumbus_observations()
        assert len(obs) > 15000, f"Expected >15000 rows, got {len(obs)}"

    def test_kolumbus_date_range(self):
        obs = load_kolumbus_observations()
        assert obs[0]["time"] == datetime(2023, 5, 15, 0, 0)
        assert obs[-1]["time"].month == 8

    def test_kolumbus_no_nans(self):
        obs = load_kolumbus_observations()
        for row in obs[:100]:  # Spot-check first 100
            for key, val in row.items():
                if key == "time":
                    continue
                assert not np.isnan(val), f"NaN found in {key} at {row['time']}"

    def test_kolumbus_daytime_wall_heating(self):
        """During daytime, wall surface should be warmer than air."""
        obs = load_kolumbus_observations(start="2023-07-01", end="2023-07-01")
        noon = [o for o in obs if 11 <= o["time"].hour <= 14]
        assert len(noon) > 0

        for o in noon:
            # At least one wall type should be warmer than air at noon
            pb_excess = o["ts_pb_obs"] - o["ta"]
            wood_excess = o["ts_wood_obs"] - o["ta"]
            assert pb_excess > 0 or wood_excess > 0, (
                f"Neither wall warmer than air at {o['time']}: pb={pb_excess:.1f}K, wood={wood_excess:.1f}K"
            )

    def test_umep_met_loader(self):
        """Verify Weather.from_umep_met() loads the forcing data."""
        from solweig import Weather

        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_may.txt",
            start="2023-05-15",
            end="2023-05-15",
        )
        assert len(met) == 24
        assert met[0].datetime == datetime(2023, 5, 15, 0, 0)
        assert met[0].ta > 0  # Not -999
        assert met[0].rh > 0
        assert met[0].pressure > 900  # Valid pressure in hPa

    def test_umep_met_multi_file(self):
        """Load multiple monthly UMEP met files."""
        from solweig import Weather

        met = Weather.from_umep_met(
            [DATA_DIR / "metdata_10min_may.txt", DATA_DIR / "metdata_10min_june.txt"],
            start="2023-05-31",
            end="2023-06-01",
        )
        # Should span the month boundary
        assert met[0].datetime.month == 5
        assert met[-1].datetime.month == 6

    def test_geodata_loads(self):
        """Verify DSM/DEM/CDSM/groundcover rasters load."""
        import rasterio

        for name in ["dsm", "dem", "cdsm", "groundcover"]:
            path = DATA_DIR / "geodata" / f"{name}.tif"
            assert path.exists(), f"Missing {name}.tif"
            with rasterio.open(path) as src:
                data = src.read(1)
                assert data.shape == (80, 81), f"{name}.tif shape={data.shape}"


# ---------------------------------------------------------------------------
# Test: Wall temperature model validation
# ---------------------------------------------------------------------------


class TestWallTemperatureValidation:
    """Validate SOLWEIG wall temperature (tg_wall) against field measurements.

    This directly tests the ground temperature component without running the
    full SOLWEIG pipeline (no SVF, shadows, or radiation budget needed).
    """

    @pytest.fixture
    def gothenburg_location(self):
        from solweig import Location

        return Location(latitude=57.6966, longitude=11.9305, utc_offset=2, altitude=10.0)

    def _compute_tg_wall_for_day(
        self,
        weather_list,
        location,
        tgk_wall=None,
        tstart_wall=None,
        tmaxlst_wall=None,
    ):
        """Compute tg_wall for each hourly Weather in a day.

        Args:
            weather_list: List of Weather objects for the day.
            location: Location object.
            tgk_wall: Optional wall TgK (temperature gain coefficient).
            tstart_wall: Optional wall Tstart (baseline offset).
            tmaxlst_wall: Optional wall TmaxLST (hour of max temperature).

        Returns list of (datetime, ta, tg_wall) tuples.
        """
        from solweig.components.ground import compute_ground_temperature

        # We need dummy grids for the ground temp model (any shape works
        # since tg_wall is a scalar). Use 1x1 grids.
        alb = np.array([[0.15]], dtype=np.float32)
        emis = np.array([[0.95]], dtype=np.float32)
        tgk = np.array([[0.37]], dtype=np.float32)
        tstart = np.array([[-3.41]], dtype=np.float32)
        tmaxlst = np.array([[15.0]], dtype=np.float32)

        results = []
        for w in weather_list:
            w.compute_derived(location)
            # Skip nighttime (clearness index has division-by-zero at night)
            if w.sun_altitude <= 0:
                results.append((w.datetime, w.ta, 0.0))
                continue
            bundle = compute_ground_temperature(
                weather=w,
                location=location,
                alb_grid=alb,
                emis_grid=emis,
                tgk_grid=tgk,
                tstart_grid=tstart,
                tmaxlst_grid=tmaxlst,
                tgk_wall=tgk_wall,
                tstart_wall=tstart_wall,
                tmaxlst_wall=tmaxlst_wall,
            )
            results.append((w.datetime, w.ta, bundle.tg_wall))
        return results

    @pytest.mark.slow
    def test_wall_temp_diurnal_pattern(self, gothenburg_location):
        """Wall temperature deviation should follow a diurnal cycle."""
        from solweig import Weather

        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )
        results = self._compute_tg_wall_for_day(met, gothenburg_location)

        # tg_wall should be 0 at night, positive during day
        night_vals = [tg for dt, ta, tg in results if dt.hour < 4 or dt.hour > 22]
        day_vals = [tg for dt, ta, tg in results if 10 <= dt.hour <= 16]

        assert all(v == 0.0 for v in night_vals), "tg_wall should be 0 at night"
        assert any(v > 0 for v in day_vals), "tg_wall should be positive during day"
        assert max(day_vals) > 1.0, f"Expected peak tg_wall > 1K, got {max(day_vals):.2f}K"

    @pytest.mark.slow
    def test_wall_temp_vs_observations_summer(self, gothenburg_location):
        """Compare model wall temperature against observations for a clear summer day.

        This is the primary field-data validation test. We compare:
        - Model: Ta + tg_wall (using SOLWEIG cobblestone parameters)
        - Observed: Ts_pb_obs (plastered brick) and Ts_wood_obs (wood)

        The model uses generic parameters so we expect moderate agreement.
        """
        from solweig import Weather

        # Use a summer day (July 15, 2023)
        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )
        model_results = self._compute_tg_wall_for_day(met, gothenburg_location)

        # Load corresponding hourly observations
        obs = load_hourly_observations("2023-07-15", "2023-07-15")

        # Match timestamps
        model_dict = {dt: (ta, tg) for dt, ta, tg in model_results}
        matched_pb = []
        matched_wood = []

        for o in obs:
            if o["time"] in model_dict:
                ta, tg_wall = model_dict[o["time"]]
                model_ts = ta + tg_wall
                matched_pb.append((o["ts_pb_obs"], model_ts))
                matched_wood.append((o["ts_wood_obs"], model_ts))

        assert len(matched_pb) >= 20, f"Only {len(matched_pb)} matched timestamps"

        # Compute statistics
        obs_pb = np.array([x[0] for x in matched_pb])
        obs_wood = np.array([x[0] for x in matched_wood])
        mod = np.array([x[1] for x in matched_pb])

        rmse_pb = np.sqrt(np.mean((obs_pb - mod) ** 2))
        rmse_wood = np.sqrt(np.mean((obs_wood - mod) ** 2))
        mae_pb = np.mean(np.abs(obs_pb - mod))
        mae_wood = np.mean(np.abs(obs_wood - mod))

        # Report statistics (print for visibility in pytest -v output)
        print("\n--- Wall Temperature Validation (2023-07-15) ---")
        print(f"Plastered brick: RMSE={rmse_pb:.2f}°C, MAE={mae_pb:.2f}°C")
        print(f"Wood:            RMSE={rmse_wood:.2f}°C, MAE={mae_wood:.2f}°C")
        print(f"Model peak Ts:   {mod.max():.1f}°C")
        print(f"PB obs peak Ts:  {obs_pb.max():.1f}°C")
        print(f"Wood obs peak:   {obs_wood.max():.1f}°C")

        # Acceptance criteria: generous thresholds since model uses generic
        # cobblestone parameters (tgk=0.37, tstart=-3.41) rather than
        # material-specific properties. The reference paper (Wallenberg et al.
        # 2025) reports RMSE ~2°C with tuned per-material params.
        # Single-day RMSE is more variable than monthly; use 15°C threshold.
        assert rmse_pb < 15.0, f"Plastered brick RMSE={rmse_pb:.2f}°C exceeds 15°C threshold"
        assert rmse_wood < 15.0, f"Wood RMSE={rmse_wood:.2f}°C exceeds 15°C threshold"

    @pytest.mark.slow
    def test_wall_temp_multi_day_statistics(self, gothenburg_location):
        """Compute validation statistics across multiple days in July.

        This provides a more robust assessment than a single day.
        """
        from solweig import Weather

        met_all = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
        )

        all_pb_errors = []
        all_wood_errors = []

        # Process each day
        for day in range(1, 32):
            day_str = f"2023-07-{day:02d}"
            day_met = [w for w in met_all if w.datetime.date().isoformat() == day_str]
            if len(day_met) < 20:
                continue

            model_results = self._compute_tg_wall_for_day(day_met, gothenburg_location)
            obs = load_hourly_observations(day_str, day_str)

            model_dict = {dt: (ta, tg) for dt, ta, tg in model_results}
            for o in obs:
                if o["time"] in model_dict:
                    ta, tg_wall = model_dict[o["time"]]
                    model_ts = ta + tg_wall
                    all_pb_errors.append(o["ts_pb_obs"] - model_ts)
                    all_wood_errors.append(o["ts_wood_obs"] - model_ts)

        assert len(all_pb_errors) > 500, f"Only {len(all_pb_errors)} matched points"

        pb_errors = np.array(all_pb_errors)
        wood_errors = np.array(all_wood_errors)

        rmse_pb = np.sqrt(np.mean(pb_errors**2))
        rmse_wood = np.sqrt(np.mean(wood_errors**2))
        bias_pb = np.mean(pb_errors)
        bias_wood = np.mean(wood_errors)

        print("\n--- Wall Temperature Validation (July 2023, all days) ---")
        print(f"Matched observations: {len(all_pb_errors)}")
        print(f"Plastered brick: RMSE={rmse_pb:.2f}°C, Bias={bias_pb:+.2f}°C")
        print(f"Wood:            RMSE={rmse_wood:.2f}°C, Bias={bias_wood:+.2f}°C")

        # Multi-day statistics should be somewhat stable
        assert rmse_pb < 10.0, f"Monthly RMSE PB={rmse_pb:.2f}°C too high"
        assert rmse_wood < 10.0, f"Monthly RMSE wood={rmse_wood:.2f}°C too high"

    @pytest.mark.slow
    def test_wall_temp_brick_params(self, gothenburg_location):
        """Validate with brick-specific wall params from JSON.

        The Kolumbus plastered brick wall should be better modeled with
        brick-appropriate thermal response parameters.
        """
        from solweig import Weather

        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )
        # Brick wall params from default_materials.json
        model_results = self._compute_tg_wall_for_day(
            met,
            gothenburg_location,
            tgk_wall=0.40,
            tstart_wall=-4.0,
            tmaxlst_wall=15.0,
        )

        obs = load_hourly_observations("2023-07-15", "2023-07-15")
        model_dict = {dt: (ta, tg) for dt, ta, tg in model_results}
        matched = []
        for o in obs:
            if o["time"] in model_dict:
                ta, tg_wall = model_dict[o["time"]]
                matched.append((o["ts_pb_obs"], ta + tg_wall))

        assert len(matched) >= 20
        obs_arr = np.array([x[0] for x in matched])
        mod_arr = np.array([x[1] for x in matched])
        rmse = np.sqrt(np.mean((obs_arr - mod_arr) ** 2))

        print(f"\n--- Brick params on PB wall (2023-07-15): RMSE={rmse:.2f}°C ---")
        assert rmse < 15.0, f"Brick-param RMSE={rmse:.2f}°C exceeds threshold"

    @pytest.mark.slow
    def test_wall_temp_wood_params(self, gothenburg_location):
        """Validate with wood-specific wall params from JSON.

        The Kolumbus wood wall should be better modeled with
        wood-appropriate thermal response parameters.
        """
        from solweig import Weather

        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )
        # Wood wall params from default_materials.json
        model_results = self._compute_tg_wall_for_day(
            met,
            gothenburg_location,
            tgk_wall=0.50,
            tstart_wall=-2.0,
            tmaxlst_wall=14.0,
        )

        obs = load_hourly_observations("2023-07-15", "2023-07-15")
        model_dict = {dt: (ta, tg) for dt, ta, tg in model_results}
        matched = []
        for o in obs:
            if o["time"] in model_dict:
                ta, tg_wall = model_dict[o["time"]]
                matched.append((o["ts_wood_obs"], ta + tg_wall))

        assert len(matched) >= 20
        obs_arr = np.array([x[0] for x in matched])
        mod_arr = np.array([x[1] for x in matched])
        rmse = np.sqrt(np.mean((obs_arr - mod_arr) ** 2))

        print(f"\n--- Wood params on wood wall (2023-07-15): RMSE={rmse:.2f}°C ---")
        assert rmse < 15.0, f"Wood-param RMSE={rmse:.2f}°C exceeds threshold"

    @pytest.mark.slow
    def test_material_params_vs_default_comparison(self, gothenburg_location):
        """Compare material-specific params against default cobblestone.

        Material-specific params should produce different (ideally better)
        results than the generic cobblestone default.
        """
        from solweig import Weather

        met = Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )

        # Default (cobblestone)
        default_results = self._compute_tg_wall_for_day(met, gothenburg_location)
        # Brick params
        brick_results = self._compute_tg_wall_for_day(
            met, gothenburg_location, tgk_wall=0.40, tstart_wall=-4.0, tmaxlst_wall=15.0
        )
        # Wood params
        wood_results = self._compute_tg_wall_for_day(
            met, gothenburg_location, tgk_wall=0.50, tstart_wall=-2.0, tmaxlst_wall=14.0
        )

        # The different params should produce different peak wall temperatures
        default_peak = max(tg for _, _, tg in default_results)
        brick_peak = max(tg for _, _, tg in brick_results)
        wood_peak = max(tg for _, _, tg in wood_results)

        print("\n--- Material comparison (2023-07-15 peak tg_wall) ---")
        print(f"Default (cobblestone): {default_peak:.2f}K")
        print(f"Brick:                 {brick_peak:.2f}K")
        print(f"Wood:                  {wood_peak:.2f}K")

        # Wood should have higher peak (faster response, higher TgK)
        assert wood_peak > default_peak, "Wood should heat faster than cobblestone"
        # Brick should also differ from default (different TgK/Tstart)
        assert abs(brick_peak - default_peak) > 0.1, "Brick should differ from default"


# ---------------------------------------------------------------------------
# Test: Full SOLWEIG pipeline validation
# ---------------------------------------------------------------------------


class TestFullPipelineValidation:
    """Run the full SOLWEIG pipeline on the Kolumbus site.

    This validates that the complete model chain (shadows → SVF → radiation →
    Tmrt) produces physically reasonable results with real-world inputs.
    """

    @pytest.fixture
    def surface(self, tmp_path):
        """Load SurfaceData from the Kolumbus GeoTIFFs."""
        import solweig

        geodata = DATA_DIR / "geodata"
        surface = solweig.SurfaceData.prepare(
            dsm=str(geodata / "dsm.tif"),
            cdsm=str(geodata / "cdsm.tif"),
            dem=str(geodata / "dem.tif"),
            land_cover=str(geodata / "groundcover.tif"),
            working_dir=str(tmp_path / "kolumbus_work"),
        )
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(latitude=57.6966, longitude=11.9305, utc_offset=2, altitude=10.0)

    @pytest.mark.slow
    def test_single_timestep_noon(self, surface, location):
        """Run SOLWEIG for a single noon timestep and check outputs are physical."""
        import solweig

        met = solweig.Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )
        # Pick noon
        noon = [w for w in met if w.datetime.hour == 12][0]

        result = solweig.calculate(surface=surface, location=location, weather=noon)

        # WOI pixel (row=22, col=28) - at ground level near building wall
        woi_tmrt = result.tmrt[22, 28]

        print("\n--- Single Timestep (2023-07-15 12:00) ---")
        print(f"WOI Tmrt:    {woi_tmrt:.1f}°C")
        print(f"Tmrt range:  {np.nanmin(result.tmrt):.1f} to {np.nanmax(result.tmrt):.1f}°C")
        print(f"Air temp:    {noon.ta:.1f}°C")

        # Tmrt should be reasonable (not NaN, not extreme)
        assert not np.isnan(woi_tmrt), "Tmrt at WOI is NaN"
        assert 10 < woi_tmrt < 80, f"Tmrt at WOI={woi_tmrt:.1f}°C outside physical range"

        # At noon in summer, Tmrt should generally exceed Ta
        assert woi_tmrt > noon.ta - 5, "Tmrt much lower than Ta at noon"

    @pytest.mark.slow
    def test_timeseries_one_day(self, surface, location):
        """Run full timeseries for one day and verify diurnal Tmrt pattern."""
        import solweig

        met = solweig.Weather.from_umep_met(
            DATA_DIR / "metdata_10min_july.txt",
            start="2023-07-15",
            end="2023-07-15",
        )

        summary = solweig.calculate_timeseries(
            surface=surface,
            location=location,
            weather_series=met,
            timestep_outputs=["tmrt"],
        )
        results = summary.results

        assert len(results) == 24

        woi_tmrt = [r.tmrt[22, 28] for r in results]
        hours = [met[i].datetime.hour for i in range(len(met))]
        ta_series = [met[i].ta for i in range(len(met))]

        print("\n--- Timeseries (2023-07-15) ---")
        print(f"{'Hour':>4s} {'Ta':>6s} {'Tmrt':>6s} {'Tmrt-Ta':>7s}")
        for h, ta, tmrt in zip(hours, ta_series, woi_tmrt, strict=False):
            print(f"{h:4d} {ta:6.1f} {tmrt:6.1f} {tmrt - ta:+7.1f}")

        # Daytime Tmrt should exceed nighttime
        day_tmrt = [t for h, t in zip(hours, woi_tmrt, strict=False) if 10 <= h <= 16]
        night_tmrt = [t for h, t in zip(hours, woi_tmrt, strict=False) if h < 5 or h > 22]

        if day_tmrt and night_tmrt:
            assert np.mean(day_tmrt) > np.mean(night_tmrt), (
                f"Daytime Tmrt ({np.mean(day_tmrt):.1f}) should exceed nighttime ({np.mean(night_tmrt):.1f})"
            )
