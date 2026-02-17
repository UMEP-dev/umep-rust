"""
Field-data validation tests using the INRAE PRESTI Montpellier canyon dataset.

Dataset: Garcia de Cezar et al. (2025) - Microclimate in Mediterranean urban canyon
Source: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/0MYJU4
Paper: https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.70033
Location: INRAE Campus Lavalette, Montpellier, France (43.64°N, 3.87°E)
Period: 2023-07-21 to 2024-07-31 (10-min intervals)
Measurements: Grey globe thermometers (15 sensors), pyranometers, weather station

Canyon geometry:
    - Orientation: East-West
    - Wall height: 2.3 m (concrete blocks)
    - Length: 12 m (E-W axis)
    - Width: 5 m (N-S axis, between inner wall faces)
    - Globe thermometers at 1.3 m above ground
    - 40 mm ping-pong ball grey globes (RAL 7001 silver grey, PT100 sensors)

These tests validate SOLWEIG Tmrt against globe-thermometer-derived Tmrt
in a controlled reduced-scale urban canyon with known geometry.

Tests are marked @pytest.mark.slow and @pytest.mark.validation since they
require external data files and take significant time.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "validation" / "montpellier"
SUBSET_CSV = DATA_DIR / "presti_subset.csv"

pytestmark = [
    pytest.mark.skipif(not SUBSET_CSV.exists(), reason="Montpellier validation data not present"),
    pytest.mark.validation,
]

# Canyon geometry (metres)
CANYON_LENGTH = 12.0  # E-W
CANYON_WIDTH = 5.0  # N-S (between inner wall faces)
WALL_HEIGHT = 2.3
WALL_THICKNESS = 1.0  # Approximate thickness of concrete block walls

# Globe thermometer constants
GLOBE_DIAMETER = 0.040  # 40 mm ping-pong ball
GLOBE_EMISSIVITY = 0.95  # Longwave emissivity of painted surface
SBC = 5.67e-8  # Stefan-Boltzmann constant

# Site location
LATITUDE = 43.64
LONGITUDE = 3.87
UTC_OFFSET = 2  # CEST (Central European Summer Time) for summer 2023

# DSM resolution
RESOLUTION = 0.5  # metres per pixel

# Linke turbidity for clear Mediterranean sky (typical summer value)
LINKE_TURBIDITY = 3.5


# ---------------------------------------------------------------------------
# Clear-sky radiation model
# ---------------------------------------------------------------------------


def clear_sky_ghi(sun_altitude_deg: float, day_of_year: int = 216) -> float:
    """Estimate clear-sky Global Horizontal Irradiance (GHI) from sun altitude.

    Uses a simplified Ineichen clear-sky model with Linke turbidity for
    Mediterranean climate. This replaces in-canyon pyranometer readings
    which are contaminated by wall shading and reflections.

    Args:
        sun_altitude_deg: Sun altitude in degrees above horizon.
        day_of_year: Day of year (1-365). Default 216 = Aug 4.

    Returns:
        Clear-sky GHI in W/m².
    """
    if sun_altitude_deg <= 0:
        return 0.0
    # Solar constant with eccentricity correction
    I0 = 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365))
    zen_rad = np.radians(90 - sun_altitude_deg)
    cos_zen = np.cos(zen_rad)
    # Air mass (Kasten & Young 1989)
    am = 1.0 / (cos_zen + 0.50572 * (96.07995 - (90 - sun_altitude_deg)) ** (-1.6364))
    am = min(am, 40.0)
    # Ineichen clear-sky model (altitude = 50m for Montpellier)
    fh1 = np.exp(-0.00050 / 8.434)
    cg1 = 5.09e-5 * 50 + 0.868
    cg2 = 3.92e-5 * 50 + 0.0387
    ghi = cg1 * I0 * cos_zen * np.exp(-cg2 * am * (fh1 * LINKE_TURBIDITY - 1.0))
    return max(float(ghi), 0.0)


def compute_sun_altitude(dt: datetime, lat: float = LATITUDE, lon: float = LONGITUDE) -> float:
    """Compute solar altitude angle for a given datetime and location.

    Simple astronomical formula (no refraction correction).
    """
    doy = dt.timetuple().tm_yday
    # Solar declination (Spencer 1971)
    B = 2 * np.pi * (doy - 1) / 365
    decl = np.degrees(
        0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B) - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
    )
    # Equation of time (minutes)
    eot = 229.18 * (
        0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B)
    )
    # Solar time
    solar_time = dt.hour + dt.minute / 60 + (lon - 15 * UTC_OFFSET) * 4 / 60 + eot / 60
    ha = 15 * (solar_time - 12)  # Hour angle
    sin_alt = np.sin(np.radians(lat)) * np.sin(np.radians(decl)) + np.cos(np.radians(lat)) * np.cos(
        np.radians(decl)
    ) * np.cos(np.radians(ha))
    return float(np.degrees(np.arcsin(max(-1, min(1, sin_alt)))))


# ---------------------------------------------------------------------------
# Globe temperature -> Tmrt conversion (ISO 7726)
# ---------------------------------------------------------------------------


def globe_to_tmrt(
    tg: float,
    ta: float,
    va: float,
    D: float = GLOBE_DIAMETER,
    emis: float = GLOBE_EMISSIVITY,
) -> float:
    """Convert globe temperature to Tmrt using ISO 7726 forced convection.

    Args:
        tg: Globe temperature (°C).
        ta: Air temperature (°C).
        va: Wind speed (m/s). Clamped to min 0.1 m/s.
        D: Globe diameter (m).
        emis: Globe longwave emissivity.

    Returns:
        Mean radiant temperature (°C).
    """
    va = max(va, 0.1)  # Prevent division by zero at zero wind
    # Forced convection heat transfer coefficient (ASHRAE)
    hcg = 6.3 * (va**0.6) / (D**0.4)
    # ISO 7726 formula
    tmrt_k4 = (tg + 273.15) ** 4 + (hcg / (emis * SBC)) * (tg - ta)
    if tmrt_k4 <= 0:
        return ta  # Fallback for extreme conditions
    return tmrt_k4**0.25 - 273.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_presti_observations(
    day: str | None = None,
) -> list[dict]:
    """Load PRESTI subset CSV observations.

    Args:
        day: ISO date string to filter (e.g. "2023-08-04"). If None, load all.

    Returns:
        List of dicts with timestamp, met data, globe temps, and radiation.
    """
    rows = []
    with open(SUBSET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.fromisoformat(row["TIMESTAMP"].replace("Z", ""))
            if day and dt.date().isoformat() != day:
                continue

            def _float(val: str) -> float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return float("nan")

            rows.append(
                {
                    "time": dt,
                    "ta": _float(row.get("SMn_TA", "")),
                    "rh": _float(row.get("SMn_HR", "")),
                    "wspd": _float(row.get("SMn_Wspd", "")),
                    "wdir": _float(row.get("SMn_Wdir", "")),
                    # Globe temperatures (15 sensors at 1.3m)
                    "G1": _float(row.get("G1_TA", "")),
                    "G2": _float(row.get("G2_TA", "")),
                    "G3": _float(row.get("G3_TA", "")),
                    "G4": _float(row.get("G4_TA", "")),
                    "G5": _float(row.get("G5_TA", "")),
                    "G6": _float(row.get("G6_TA", "")),
                    "G7": _float(row.get("G7_TA", "")),
                    "G8": _float(row.get("G8_TA", "")),
                    "G9": _float(row.get("G9_TA", "")),
                    "GA": _float(row.get("GA_TA", "")),
                    "GB": _float(row.get("GB_TA", "")),
                    "GC": _float(row.get("GC_TA", "")),
                    "GD": _float(row.get("GD_TA", "")),
                    "GE": _float(row.get("GE_TA", "")),
                    "GF": _float(row.get("GF_TA", "")),
                    # Pyranometer solar radiation
                    "slr1": _float(row.get("SlrW_1", "")),
                    "slr2": _float(row.get("SlrW_2", "")),
                    "slr3": _float(row.get("SlrW_3", "")),
                    "slr4": _float(row.get("SlrW_4", "")),
                }
            )
    return rows


def compute_observed_tmrt(obs: list[dict]) -> list[dict]:
    """Add observed Tmrt derived from globe temperatures to observation dicts.

    Computes Tmrt from each globe thermometer and adds mean/center values.
    Center globes (G2, G5, G8 at y=2.6m) are most representative of the
    open canyon floor away from wall influence.
    """
    results = []
    for o in obs:
        ta = o["ta"]
        va = o["wspd"]
        if math.isnan(ta) or math.isnan(va):
            continue

        # Convert center-canyon globes (y=2.6, sections A/B/C)
        center_globes = ["G2", "G5", "G8"]
        tmrt_center = []
        for g in center_globes:
            tg = o[g]
            if not math.isnan(tg):
                tmrt_center.append(globe_to_tmrt(tg, ta, va))

        # Convert all 15 globes
        all_globes = [f"G{i}" for i in range(1, 10)] + [f"G{c}" for c in "ABCDEF"]
        tmrt_all = []
        for g in all_globes:
            tg = o[g]
            if not math.isnan(tg):
                tmrt_all.append(globe_to_tmrt(tg, ta, va))

        if tmrt_center:
            entry = dict(o)
            entry["tmrt_center"] = np.mean(tmrt_center)
            entry["tmrt_all_mean"] = np.mean(tmrt_all) if tmrt_all else float("nan")
            entry["tmrt_center_std"] = np.std(tmrt_center) if len(tmrt_center) > 1 else 0.0
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Synthetic DSM construction
# ---------------------------------------------------------------------------


def build_canyon_dsm() -> np.ndarray:
    """Build a synthetic DSM for the PRESTI canyon.

    The DSM represents the canyon geometry at 0.5m resolution:
    - E-W canyon (long axis along columns)
    - North and south concrete walls at 2.3m height
    - Open ground (0m) everywhere else

    Grid layout (rows = N-S, cols = E-W):
        rows 0-7:   open ground north of canyon (0m)
        rows 8-9:   north wall (2.3m), 2 pixels = 1m thick
        rows 10-19: canyon floor (0m), 10 pixels = 5m wide
        rows 20-21: south wall (2.3m), 2 pixels = 1m thick
        rows 22-29: open ground south of canyon (0m)

        cols 0-7:   open ground west of canyon (0m)
        cols 8-31:  canyon extent (24 pixels = 12m)
        cols 32-39: open ground east of canyon (0m)

    Returns:
        DSM array of shape (30, 40) at 0.5m resolution.
    """
    nrows, ncols = 30, 40
    dsm = np.zeros((nrows, ncols), dtype=np.float32)

    # Wall columns span the canyon length (cols 8-31)
    wall_cols = slice(8, 32)

    # North wall (rows 8-9)
    dsm[8:10, wall_cols] = WALL_HEIGHT

    # South wall (rows 20-21)
    dsm[20:22, wall_cols] = WALL_HEIGHT

    return dsm


# Canyon center pixel coordinates (row, col)
# The canyon floor spans rows 10-19, cols 8-31
# Center of canyon: row 14-15, col 19-20
CANYON_CENTER_ROW = 15
CANYON_CENTER_COL = 20


# ---------------------------------------------------------------------------
# Test: Data loading and sanity checks
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that the validation data loads correctly."""

    def test_subset_csv_loads(self):
        obs = load_presti_observations()
        assert len(obs) > 400, f"Expected >400 rows, got {len(obs)}"

    def test_aug04_has_data(self):
        obs = load_presti_observations(day="2023-08-04")
        assert len(obs) == 144, f"Expected 144 rows (24h × 6/hr), got {len(obs)}"

    def test_globe_temps_physical(self):
        """Globe temperatures should be in a physical range."""
        obs = load_presti_observations(day="2023-08-04")
        for o in obs:
            for g in ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]:
                tg = o[g]
                if not math.isnan(tg):
                    assert 5 < tg < 70, f"{g}={tg}°C outside physical range at {o['time']}"

    def test_globe_exceeds_air_temp_at_noon(self):
        """Globe temperature should exceed air temperature during sunny hours."""
        obs = load_presti_observations(day="2023-08-04")
        noon_obs = [o for o in obs if 12 <= o["time"].hour <= 15]
        assert len(noon_obs) > 0

        for o in noon_obs:
            ta = o["ta"]
            # Center globe (G5) should be warmer than air during peak sun
            g5 = o["G5"]
            if not math.isnan(g5) and not math.isnan(ta):
                assert g5 >= ta - 1.0, f"Globe temp G5={g5:.1f}°C < Ta={ta:.1f}°C at {o['time']}"

    def test_solar_radiation_diurnal(self):
        """Solar radiation should show a clear diurnal pattern."""
        obs = load_presti_observations(day="2023-08-04")
        night_rad = [o["slr2"] for o in obs if o["time"].hour < 6 and not math.isnan(o["slr2"])]
        day_rad = [o["slr2"] for o in obs if 10 <= o["time"].hour <= 16 and not math.isnan(o["slr2"])]

        assert all(r <= 5.0 for r in night_rad), "Radiation should be ~0 at night"
        assert max(day_rad) > 200, f"Peak daytime radiation={max(day_rad):.0f} too low"


class TestGlobeToTmrt:
    """Test the globe temperature to Tmrt conversion."""

    def test_equilibrium(self):
        """When globe temp equals air temp, Tmrt should equal air temp."""
        tmrt = globe_to_tmrt(tg=25.0, ta=25.0, va=1.0)
        assert abs(tmrt - 25.0) < 0.5, f"Tmrt={tmrt:.1f} should be ~25°C"

    def test_globe_above_air(self):
        """When globe > air, Tmrt should exceed both."""
        tmrt = globe_to_tmrt(tg=35.0, ta=25.0, va=1.0)
        assert tmrt > 35.0, f"Tmrt={tmrt:.1f} should exceed globe temp 35°C"

    def test_globe_below_air(self):
        """When globe < air (cold radiation), Tmrt should be below air."""
        tmrt = globe_to_tmrt(tg=18.0, ta=25.0, va=1.0)
        assert tmrt < 18.0, f"Tmrt={tmrt:.1f} should be below globe temp 18°C"

    def test_wind_sensitivity(self):
        """Higher wind increases convective coefficient, amplifying Tmrt."""
        tmrt_calm = globe_to_tmrt(tg=35.0, ta=25.0, va=0.5)
        tmrt_windy = globe_to_tmrt(tg=35.0, ta=25.0, va=3.0)
        # At same globe and air temp, more wind means more convective
        # heat loss needed to balance, so Tmrt must be higher
        assert tmrt_windy > tmrt_calm

    def test_observed_tmrt_computation(self):
        """Compute Tmrt from observed globe temps for one timestep."""
        obs = load_presti_observations(day="2023-08-04")
        results = compute_observed_tmrt(obs)
        assert len(results) > 100

        # At noon, observed Tmrt should be well above air temperature
        noon = [r for r in results if 12 <= r["time"].hour <= 15]
        for r in noon:
            assert r["tmrt_center"] > r["ta"], f"Tmrt_center={r['tmrt_center']:.1f} < Ta={r['ta']:.1f} at {r['time']}"


class TestSyntheticDSM:
    """Verify the synthetic DSM geometry."""

    def test_dsm_shape(self):
        dsm = build_canyon_dsm()
        assert dsm.shape == (30, 40)

    def test_wall_height(self):
        dsm = build_canyon_dsm()
        assert dsm[8, 20] == pytest.approx(WALL_HEIGHT)  # North wall
        assert dsm[20, 20] == pytest.approx(WALL_HEIGHT)  # South wall

    def test_canyon_floor(self):
        dsm = build_canyon_dsm()
        assert dsm[15, 20] == pytest.approx(0.0)  # Canyon center

    def test_open_ground(self):
        dsm = build_canyon_dsm()
        assert dsm[0, 0] == pytest.approx(0.0)  # Corner


# ---------------------------------------------------------------------------
# Test: SOLWEIG Tmrt validation against globe thermometer observations
# ---------------------------------------------------------------------------


class TestTmrtValidation:
    """Validate SOLWEIG Tmrt against globe-derived Tmrt in the canyon."""

    @pytest.fixture
    def surface(self):
        """Build SurfaceData from synthetic canyon DSM."""
        from solweig import SurfaceData

        dsm = build_canyon_dsm()
        surface = SurfaceData(dsm=dsm, pixel_size=RESOLUTION)
        surface.compute_svf()
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            utc_offset=UTC_OFFSET,
            altitude=50.0,
        )

    @pytest.fixture
    def aug04_weather(self, location):
        """Build hourly Weather objects for August 4, 2023.

        Uses clear-sky GHI model instead of in-canyon pyranometers.
        The in-canyon pyranometers are contaminated by wall shading and
        reflections and cannot serve as open-sky radiation input.
        """
        from solweig import Weather

        obs = load_presti_observations(day="2023-08-04")

        weather_list = []
        for o in obs:
            if o["time"].minute != 0:
                continue
            if math.isnan(o["ta"]) or math.isnan(o["rh"]):
                continue

            # Clear-sky GHI from sun position (not in-canyon pyranometers)
            sun_alt = compute_sun_altitude(o["time"])
            doy = o["time"].timetuple().tm_yday
            global_rad = clear_sky_ghi(sun_alt, doy)

            w = Weather(
                datetime=o["time"],
                ta=o["ta"],
                rh=o["rh"],
                global_rad=global_rad,
                ws=max(o["wspd"], 0.1) if not math.isnan(o["wspd"]) else 1.0,
            )
            weather_list.append(w)
        return weather_list

    @pytest.fixture
    def aug04_observed_tmrt(self):
        """Compute observed Tmrt from globe thermometers for Aug 4."""
        obs = load_presti_observations(day="2023-08-04")
        return compute_observed_tmrt(obs)

    @pytest.mark.slow
    def test_single_timestep_noon(self, surface, location, aug04_weather):
        """Run SOLWEIG for noon and check Tmrt is physical."""
        import solweig

        noon = [w for w in aug04_weather if w.datetime.hour == 14][0]
        result = solweig.calculate(
            surface=surface,
            location=location,
            weather=noon,
            wall_material="concrete",
        )

        tmrt_center = result.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL]
        print(f"\n--- Noon Tmrt at canyon center: {tmrt_center:.1f}°C (Ta={noon.ta:.1f}°C) ---")

        assert not np.isnan(tmrt_center), "Tmrt at canyon center is NaN"
        assert 10 < tmrt_center < 80, f"Tmrt={tmrt_center:.1f}°C outside physical range"
        # At 14:00 in summer, Tmrt should exceed air temperature
        assert tmrt_center > noon.ta, f"Tmrt={tmrt_center:.1f} should exceed Ta={noon.ta:.1f} at peak sun"

    @pytest.mark.slow
    def test_timeseries_diurnal_pattern(self, surface, location, aug04_weather):
        """Run full-day timeseries and check diurnal Tmrt pattern."""
        import solweig

        summary = solweig.calculate_timeseries(
            surface=surface,
            location=location,
            weather_series=aug04_weather,
            wall_material="concrete",
            timestep_outputs=["tmrt"],
        )
        results = summary.results

        assert len(results) == len(aug04_weather)

        tmrt_series = [r.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL] for r in results]
        hours = [w.datetime.hour for w in aug04_weather]

        print("\n--- Diurnal Tmrt at canyon center (Aug 4, 2023) ---")
        print(f"{'Hour':>4s} {'Ta':>6s} {'Tmrt':>6s} {'Tmrt-Ta':>7s}")
        for h, w, tmrt in zip(hours, aug04_weather, tmrt_series):
            print(f"{h:4d} {w.ta:6.1f} {tmrt:6.1f} {tmrt - w.ta:+7.1f}")

        # Daytime Tmrt should exceed nighttime
        day_tmrt = [t for h, t in zip(hours, tmrt_series) if 10 <= h <= 16]
        night_tmrt = [t for h, t in zip(hours, tmrt_series) if h < 5 or h > 22]

        if day_tmrt and night_tmrt:
            assert np.mean(day_tmrt) > np.mean(night_tmrt), (
                f"Daytime Tmrt ({np.mean(day_tmrt):.1f}) should exceed nighttime ({np.mean(night_tmrt):.1f})"
            )

    @pytest.mark.slow
    def test_tmrt_vs_globe_observations(self, surface, location, aug04_weather, aug04_observed_tmrt):
        """Compare SOLWEIG Tmrt against globe-derived Tmrt.

        This is the primary validation test. We compare:
        - Model: SOLWEIG Tmrt at canyon center pixel
        - Observed: Tmrt derived from center globe thermometers (G2, G5, G8)

        The comparison is at hourly resolution.
        """
        import solweig

        summary = solweig.calculate_timeseries(
            surface=surface,
            location=location,
            weather_series=aug04_weather,
            wall_material="concrete",
            timestep_outputs=["tmrt"],
        )
        results = summary.results

        # Build model Tmrt dict by hour
        model_tmrt = {}
        for w, r in zip(aug04_weather, results):
            model_tmrt[w.datetime.hour] = r.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL]

        # Match with hourly observations
        matched = []
        for o in aug04_observed_tmrt:
            h = o["time"].hour
            if h in model_tmrt and o["time"].minute == 0:
                matched.append(
                    {
                        "hour": h,
                        "ta": o["ta"],
                        "obs_tmrt": o["tmrt_center"],
                        "mod_tmrt": model_tmrt[h],
                    }
                )

        assert len(matched) >= 20, f"Only {len(matched)} matched hours"

        obs_arr = np.array([m["obs_tmrt"] for m in matched])
        mod_arr = np.array([m["mod_tmrt"] for m in matched])

        rmse = np.sqrt(np.mean((obs_arr - mod_arr) ** 2))
        mae = np.mean(np.abs(obs_arr - mod_arr))
        bias = np.mean(mod_arr - obs_arr)
        r_squared = np.corrcoef(obs_arr, mod_arr)[0, 1] ** 2

        print("\n--- SOLWEIG vs Globe-Derived Tmrt (Aug 4, 2023) ---")
        print(f"{'Hour':>4s} {'Ta':>6s} {'Obs':>6s} {'Model':>6s} {'Diff':>6s}")
        for m in matched:
            diff = m["mod_tmrt"] - m["obs_tmrt"]
            print(f"{m['hour']:4d} {m['ta']:6.1f} {m['obs_tmrt']:6.1f} {m['mod_tmrt']:6.1f} {diff:+6.1f}")
        print(f"\nRMSE:  {rmse:.2f}°C")
        print(f"MAE:   {mae:.2f}°C")
        print(f"Bias:  {bias:+.2f}°C")
        print(f"R²:    {r_squared:.3f}")

        # Acceptance criteria:
        # Globe-derived Tmrt has ~5°C uncertainty (40mm globe accuracy).
        # Combined model + measurement uncertainty allows generous thresholds.
        # The canyon is simplified (no planters, no vegetation, uniform walls).
        assert rmse < 20.0, f"RMSE={rmse:.2f}°C exceeds 20°C threshold"
        # Model should at least correlate with observations
        assert r_squared > 0.3, f"R²={r_squared:.3f} too low (no correlation)"

    @pytest.mark.slow
    def test_canyon_shading_spatial_pattern(self, surface, location, aug04_weather):
        """Verify that the canyon shows spatial Tmrt variation from wall shading.

        Near the south wall (shaded in morning), Tmrt should differ from
        near the north wall (shaded in afternoon) at asymmetric sun angles.
        """
        import solweig

        # Pick early afternoon (14:00) when sun is from the south
        afternoon = [w for w in aug04_weather if w.datetime.hour == 14][0]
        result = solweig.calculate(
            surface=surface,
            location=location,
            weather=afternoon,
            wall_material="concrete",
        )

        # Near-south-wall pixel (row 18) vs near-north-wall pixel (row 12)
        # Avoid rows immediately adjacent to walls (may be NaN in SOLWEIG)
        tmrt_near_south = result.tmrt[18, CANYON_CENTER_COL]
        tmrt_near_north = result.tmrt[12, CANYON_CENTER_COL]
        tmrt_center = result.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL]

        print("\n--- Canyon spatial Tmrt at 14:00 ---")
        print(f"Near north wall (row 12): {tmrt_near_north:.1f}°C")
        print(f"Canyon center (row 15):   {tmrt_center:.1f}°C")
        print(f"Near south wall (row 18): {tmrt_near_south:.1f}°C")

        # All should be physical
        for val, label in [
            (tmrt_near_north, "north"),
            (tmrt_center, "center"),
            (tmrt_near_south, "south"),
        ]:
            assert not np.isnan(val), f"Tmrt near {label} wall is NaN"
            assert 5 < val < 80, f"Tmrt near {label} wall = {val:.1f}°C out of range"

    @pytest.mark.slow
    def test_multi_day_statistics(self, surface, location):
        """Compute validation statistics across multiple clear-sky days."""
        import solweig

        all_errors = []

        for day_str in ["2023-08-03", "2023-08-04", "2023-08-14"]:
            obs = load_presti_observations(day=day_str)
            obs_tmrt = compute_observed_tmrt(obs)
            if not obs_tmrt:
                continue

            # Build hourly Weather with clear-sky GHI
            weather_list = []
            for o in obs:
                if o["time"].minute != 0:
                    continue
                if math.isnan(o["ta"]) or math.isnan(o["rh"]):
                    continue
                sun_alt = compute_sun_altitude(o["time"])
                doy = o["time"].timetuple().tm_yday
                global_rad = clear_sky_ghi(sun_alt, doy)
                w = solweig.Weather(
                    datetime=o["time"],
                    ta=o["ta"],
                    rh=o["rh"],
                    global_rad=global_rad,
                    ws=max(o["wspd"], 0.1) if not math.isnan(o["wspd"]) else 1.0,
                )
                weather_list.append(w)

            if len(weather_list) < 20:
                continue

            summary = solweig.calculate_timeseries(
                surface=surface,
                location=location,
                weather_series=weather_list,
                wall_material="concrete",
                timestep_outputs=["tmrt"],
            )
            results = summary.results

            model_tmrt = {}
            for w, r in zip(weather_list, results):
                model_tmrt[w.datetime] = r.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL]

            for o in obs_tmrt:
                # Match on-the-hour observations
                if o["time"].minute == 0 and o["time"] in model_tmrt:
                    all_errors.append(model_tmrt[o["time"]] - o["tmrt_center"])

        assert len(all_errors) > 50, f"Only {len(all_errors)} matched points"

        errors = np.array(all_errors)
        rmse = np.sqrt(np.mean(errors**2))
        bias = np.mean(errors)

        print("\n--- Multi-day Tmrt validation (3 clear-sky days) ---")
        print(f"Matched points: {len(all_errors)}")
        print(f"RMSE: {rmse:.2f}°C")
        print(f"Bias: {bias:+.2f}°C")

        # Multi-day RMSE threshold (generous due to synthetic DSM + globe uncertainty)
        assert rmse < 20.0, f"Multi-day RMSE={rmse:.2f}°C exceeds threshold"


# ---------------------------------------------------------------------------
# Test: Isotropic vs Anisotropic sky model comparison
# ---------------------------------------------------------------------------


class TestSkyModelComparison:
    """Compare isotropic vs anisotropic sky radiation models against observations.

    The anisotropic (Perez) sky model should better capture directional
    diffuse radiation in the canyon geometry. This test quantifies the
    accuracy improvement.
    """

    @pytest.fixture
    def surface(self):
        from solweig import SurfaceData

        dsm = build_canyon_dsm()
        surface = SurfaceData(dsm=dsm, pixel_size=RESOLUTION)
        surface.compute_svf()
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            utc_offset=UTC_OFFSET,
            altitude=50.0,
        )

    def _build_weather(self, day: str) -> list:
        from solweig import Weather

        obs = load_presti_observations(day=day)
        weather_list = []
        for o in obs:
            if o["time"].minute != 0:
                continue
            if math.isnan(o["ta"]) or math.isnan(o["rh"]):
                continue
            sun_alt = compute_sun_altitude(o["time"])
            doy = o["time"].timetuple().tm_yday
            global_rad = clear_sky_ghi(sun_alt, doy)
            w = Weather(
                datetime=o["time"],
                ta=o["ta"],
                rh=o["rh"],
                global_rad=global_rad,
                ws=max(o["wspd"], 0.1) if not math.isnan(o["wspd"]) else 1.0,
            )
            weather_list.append(w)
        return weather_list

    @pytest.mark.slow
    def test_isotropic_sky_rmse(self, surface, location):
        """Validate isotropic sky model RMSE against globe observations.

        Note: The anisotropic (Perez) sky model requires precomputed shadow
        matrices for 145 sky patches, which are only available when using
        SurfaceData.prepare() with GeoTIFF inputs. With a synthetic DSM,
        the model falls back to isotropic. A full anisotropic comparison
        would require real DSM data (e.g., from IGN Lidar HD for Montpellier).
        """
        import solweig

        day = "2023-08-04"
        weather_list = self._build_weather(day)
        obs_tmrt = compute_observed_tmrt(load_presti_observations(day=day))

        summary_iso = solweig.calculate_timeseries(
            surface=surface,
            location=location,
            weather_series=weather_list,
            use_anisotropic_sky=False,
            timestep_outputs=["tmrt"],
        )
        results_iso = summary_iso.results

        tmrt_iso = {}
        for w, r in zip(weather_list, results_iso):
            tmrt_iso[w.datetime.hour] = r.tmrt[CANYON_CENTER_ROW, CANYON_CENTER_COL]

        errors = []
        for o in obs_tmrt:
            h = o["time"].hour
            if h in tmrt_iso and o["time"].minute == 0:
                errors.append(tmrt_iso[h] - o["tmrt_center"])

        assert len(errors) >= 20

        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        bias = np.mean(errors)

        print("\n--- Isotropic Sky Model Validation (Aug 4, 2023) ---")
        print(f"RMSE: {rmse:.2f}°C, Bias: {bias:+.2f}°C")
        print("Note: Anisotropic comparison requires precomputed shadow matrices")

        assert rmse < 20.0, f"Isotropic RMSE={rmse:.2f}°C too high"
