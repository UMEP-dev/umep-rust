"""
Golden Regression Tests for Ground Temperature (TsWaveDelay) Calculations

These tests verify that the Rust TsWaveDelay implementation correctly applies
thermal inertia to ground temperature using an exponential decay model.

Formula: Lup = Tgmap0 * (1 - weight) + Tgmap1 * weight
where:  weight = exp(-33.27 * timeadd)

The decay constant 33.27 day⁻¹ corresponds to a time constant of ~43 minutes.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import ground

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for ground temperature calculations
RTOL = 1e-4
ATOL = 0.1  # 0.1 units (W/m² for Lup)


@pytest.fixture(scope="module")
def ground_temp_inputs():
    """Load ground temperature input fixtures."""
    return {
        "gvflup": np.load(FIXTURES_DIR / "ground_temp_input_gvflup.npy"),
        "tgmap1": np.load(FIXTURES_DIR / "ground_temp_input_tgmap1.npy"),
    }


@pytest.fixture(scope="module")
def case1_data():
    """Load case 1: first timestep of the day."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case1.npz"))


@pytest.fixture(scope="module")
def case2_data():
    """Load case 2: short timestep accumulation."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case2.npz"))


@pytest.fixture(scope="module")
def case3_data():
    """Load case 3: long timestep."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case3.npz"))


class TestGoldenTsWaveDelay:
    """Golden tests for TsWaveDelay thermal inertia model."""

    def test_first_morning_timestep(self, ground_temp_inputs, case1_data):
        """First timestep of the day should reset previous temperature."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case1_data["input_firstdaytime"]),
            timeadd=float(case1_data["input_timeadd"]),
            timestepdec=float(case1_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case1_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 1 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case1_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 1 timeadd differs from golden fixture",
        )

        np.testing.assert_allclose(
            np.array(new_tgmap1),
            case1_data["tgmap1"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 1 Tgmap1 differs from golden fixture",
        )

    def test_short_timestep_accumulation(self, ground_temp_inputs, case2_data):
        """Short timestep should accumulate time and blend temperatures."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case2_data["input_firstdaytime"]),
            timeadd=float(case2_data["input_timeadd"]),
            timestepdec=float(case2_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case2_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 2 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case2_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 2 timeadd differs from golden fixture",
        )

    def test_long_timestep_update(self, ground_temp_inputs, case3_data):
        """Long timestep (>=59 min) should update Tgmap1."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case3_data["input_firstdaytime"]),
            timeadd=float(case3_data["input_timeadd"]),
            timestepdec=float(case3_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case3_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 3 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case3_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 3 timeadd differs from golden fixture",
        )

        np.testing.assert_allclose(
            np.array(new_tgmap1),
            case3_data["tgmap1"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 3 Tgmap1 differs from golden fixture",
        )


class TestGoldenTsWaveDelayProperties:
    """Verify physical properties of TsWaveDelay model."""

    def test_thermal_inertia_effect(self, ground_temp_inputs, case2_data):
        """Output should blend between current and previous temperature."""
        lup, _, _ = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=0,
            timeadd=float(case2_data["input_timeadd"]),
            timestepdec=float(case2_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )
        lup_arr = np.array(lup)

        # Output should be between current and previous values
        gvflup = ground_temp_inputs["gvflup"]
        tgmap1 = ground_temp_inputs["tgmap1"]

        min_vals = np.minimum(gvflup, tgmap1)
        max_vals = np.maximum(gvflup, tgmap1)

        # Allow small numerical tolerance
        assert np.all(lup_arr >= min_vals - 0.1), "Output below minimum bound"
        assert np.all(lup_arr <= max_vals + 0.1), "Output above maximum bound"

    def test_first_morning_resets_state(self, ground_temp_inputs):
        """First morning timestep should set Tgmap1 = Tgmap0."""
        _, _, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=1,  # First morning
            timeadd=0.0,
            timestepdec=30 / 1440,
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        # After first morning, Tgmap1 should equal current input
        np.testing.assert_allclose(
            np.array(new_tgmap1),
            ground_temp_inputs["gvflup"],
            rtol=1e-5,
            err_msg="First morning should reset Tgmap1 to current value",
        )

    def test_exponential_decay_weight(self, ground_temp_inputs):
        """Verify exponential decay weight is applied correctly."""
        timeadd = 0.05  # ~72 minutes (above threshold)

        lup, _, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=0,
            timeadd=timeadd,
            timestepdec=timeadd,
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        # Calculate expected weight
        weight = np.exp(-33.27 * timeadd)

        # Expected output
        expected = ground_temp_inputs["gvflup"] * (1 - weight) + ground_temp_inputs["tgmap1"] * weight

        np.testing.assert_allclose(
            np.array(lup),
            expected,
            rtol=1e-5,
            err_msg="Exponential decay weight not applied correctly",
        )


class TestGoldenGroundTemperatureSinusoidal:
    """Tests for compute_ground_temperature sinusoidal model.

    Verifies the diurnal temperature curve shape:
    - Rises from sunrise to TmaxLST
    - Peaks near TmaxLST
    - Declines after TmaxLST (afternoon cooling)
    - Returns to ~0 by late evening
    """

    @pytest.fixture
    def common_inputs(self):
        """Common inputs for sinusoidal tests (cobblestone defaults)."""
        shape = (3, 3)
        return {
            "ta": 20.0,
            "sun_altitude": 45.0,
            "altmax": 55.0,
            "snup": 5.0,  # Sunrise at 05:00
            "global_rad": 600.0,
            "rad_g0": 650.0,
            "zen_deg": 45.0,
            # Cobblestone params
            "alb_grid": np.full(shape, 0.2, dtype=np.float32),
            "emis_grid": np.full(shape, 0.95, dtype=np.float32),
            "tgk_grid": np.full(shape, 0.37, dtype=np.float32),
            "tstart_grid": np.full(shape, -3.41, dtype=np.float32),
            "tmaxlst_grid": np.full(shape, 15.0, dtype=np.float32),
        }

    def _compute_tg_at_time(self, inputs, hour):
        """Compute tg and tg_wall at a given hour."""
        dectime = hour / 24.0
        tg, tg_wall, ci, _, _ = ground.compute_ground_temperature(
            inputs["ta"],
            inputs["sun_altitude"],
            inputs["altmax"],
            dectime,
            inputs["snup"],
            inputs["global_rad"],
            inputs["rad_g0"],
            inputs["zen_deg"],
            inputs["alb_grid"],
            inputs["emis_grid"],
            inputs["tgk_grid"],
            inputs["tstart_grid"],
            inputs["tmaxlst_grid"],
        )
        return np.array(tg), tg_wall, ci

    def test_afternoon_cooling_ground(self, common_inputs):
        """Ground temperature must decline after TmaxLST (15:00).

        This is the critical test — a phase clamping bug would keep tg
        stuck at its peak value instead of declining.
        """
        tg_peak, _, _ = self._compute_tg_at_time(common_inputs, 15.0)
        tg_after, _, _ = self._compute_tg_at_time(common_inputs, 18.0)
        tg_evening, _, _ = self._compute_tg_at_time(common_inputs, 22.0)

        peak_val = tg_peak[0, 0]
        after_val = tg_after[0, 0]
        evening_val = tg_evening[0, 0]

        # Must decline: peak > 18:00 > 22:00
        assert after_val < peak_val, f"tg at 18:00 ({after_val:.2f}) should be less than peak at 15:00 ({peak_val:.2f})"
        assert evening_val < after_val, (
            f"tg at 22:00 ({evening_val:.2f}) should be less than at 18:00 ({after_val:.2f})"
        )

    def test_afternoon_cooling_wall(self, common_inputs):
        """Wall temperature must decline after TmaxLST (15:00)."""
        _, wall_peak, _ = self._compute_tg_at_time(common_inputs, 15.0)
        _, wall_after, _ = self._compute_tg_at_time(common_inputs, 18.0)
        _, wall_evening, _ = self._compute_tg_at_time(common_inputs, 22.0)

        assert wall_after < wall_peak, (
            f"tg_wall at 18:00 ({wall_after:.2f}) should be less than peak at 15:00 ({wall_peak:.2f})"
        )
        assert wall_evening < wall_after, (
            f"tg_wall at 22:00 ({wall_evening:.2f}) should be less than at 18:00 ({wall_after:.2f})"
        )

    def test_diurnal_curve_shape(self, common_inputs):
        """Full diurnal curve should rise, peak, and decline."""
        hours = [6, 9, 12, 15, 18, 21, 23]
        tg_vals = []
        for h in hours:
            tg, _, _ = self._compute_tg_at_time(common_inputs, h)
            tg_vals.append(tg[0, 0])

        # Temperature should increase from 06:00 to 15:00
        assert tg_vals[1] > tg_vals[0], "09:00 > 06:00"
        assert tg_vals[2] > tg_vals[1], "12:00 > 09:00"
        assert tg_vals[3] > tg_vals[2], "15:00 > 12:00"

        # Peak at 15:00 (index 3) should be the maximum
        assert tg_vals[3] == max(tg_vals), (
            f"Peak should be at 15:00 (TmaxLST), got max at index {tg_vals.index(max(tg_vals))}"
        )

        # Temperature should decrease after 15:00
        assert tg_vals[4] < tg_vals[3], "18:00 < 15:00"
        assert tg_vals[5] < tg_vals[4], "21:00 < 18:00"
        assert tg_vals[6] < tg_vals[5], "23:00 < 21:00"

    def test_before_sunrise_is_zero(self, common_inputs):
        """Temperature before sunrise should be 0."""
        tg, tg_wall, _ = self._compute_tg_at_time(common_inputs, 4.0)
        assert np.all(tg == 0.0), "Ground temp before sunrise should be 0"
        assert tg_wall == 0.0, "Wall temp before sunrise should be 0"

    def test_wall_material_params_affect_output(self, common_inputs):
        """Different wall material params should produce different temperatures."""
        dectime = 12.0 / 24.0
        # Default (cobblestone): tgk=0.37, tstart=-3.41, tmaxlst=15.0
        _, wall_default, _, _, _ = ground.compute_ground_temperature(
            common_inputs["ta"],
            common_inputs["sun_altitude"],
            common_inputs["altmax"],
            dectime,
            common_inputs["snup"],
            common_inputs["global_rad"],
            common_inputs["rad_g0"],
            common_inputs["zen_deg"],
            common_inputs["alb_grid"],
            common_inputs["emis_grid"],
            common_inputs["tgk_grid"],
            common_inputs["tstart_grid"],
            common_inputs["tmaxlst_grid"],
        )

        # Wood wall: higher TgK, higher Tstart → different amplitude
        _, wall_wood, _, _, _ = ground.compute_ground_temperature(
            common_inputs["ta"],
            common_inputs["sun_altitude"],
            common_inputs["altmax"],
            dectime,
            common_inputs["snup"],
            common_inputs["global_rad"],
            common_inputs["rad_g0"],
            common_inputs["zen_deg"],
            common_inputs["alb_grid"],
            common_inputs["emis_grid"],
            common_inputs["tgk_grid"],
            common_inputs["tstart_grid"],
            common_inputs["tmaxlst_grid"],
            tgk_wall=0.50,
            tstart_wall=-2.0,
            tmaxlst_wall=14.0,
        )

        assert wall_wood != wall_default, (
            "Wood wall params should produce different temperature than default cobblestone"
        )


class TestRustVsUMEPNumericalAgreement:
    """Side-by-side numerical comparison of Rust vs UMEP Python formulas.

    Reimplements the exact UMEP formulas from
    Solweig_2025a_calc_forprocessing.py (lines 171-199) in pure NumPy,
    then calls the Rust function with identical inputs and checks for
    exact numerical agreement (within f32 precision).

    This catches any formula transcription errors, ordering differences,
    or missing terms between our Rust implementation and the UMEP reference.
    """

    @staticmethod
    def _umep_ground_temp(
        altmax: float,
        dectime_frac: float,
        snup_hours: float,
        global_rad: float,
        rad_g0: float,
        zen_deg: float,
        tgk: np.ndarray,
        tstart: np.ndarray,
        tmaxlst: np.ndarray,
        tgk_wall: float,
        tstart_wall: float,
        tmaxlst_wall: float,
        sun_altitude: float,
    ) -> tuple[np.ndarray, float, float]:
        """Pure-Python UMEP reference implementation (lines 171-199).

        Returns (Tg, Tgwall, CI_TgG) matching UMEP exactly.
        """
        # --- Tgamp (UMEP line 172, 174) ---
        Tgamp = tgk * altmax + tstart
        Tgampwall = tgk_wall * altmax + tstart_wall

        # --- Phase + sinusoidal (UMEP lines 175-176) ---
        snup_frac = snup_hours / 24.0
        if dectime_frac > snup_frac:
            # Ground (per-pixel)
            tmaxlst_frac = tmaxlst / 24.0
            phase = (dectime_frac - snup_frac) / (tmaxlst_frac - snup_frac)
            Tg = Tgamp * np.sin(phase * np.pi / 2.0)

            # Wall (scalar)
            tmaxlst_wall_frac = tmaxlst_wall / 24.0
            denom_wall = tmaxlst_wall_frac - snup_frac
            phase_wall = (dectime_frac - snup_frac) / denom_wall if denom_wall > 0 else dectime_frac - snup_frac
            Tgwall = Tgampwall * np.sin(phase_wall * np.pi / 2.0)
        else:
            Tg = np.zeros_like(tgk)
            Tgwall = 0.0

        # --- Wall clip before CI (UMEP lines 178-180) ---
        if Tgwall < 0:
            Tgwall = 0.0

        # --- CI_TgG (UMEP lines 184, 189-192) ---
        if sun_altitude > 0 and rad_g0 > 0:
            corr = 0.1473 * np.log(90.0 - zen_deg) + 0.3454
            CI_TgG = (global_rad / rad_g0) + (1.0 - corr)
            if CI_TgG > 1.0 or np.isinf(CI_TgG):
                CI_TgG = 1.0
        else:
            CI_TgG = 1.0

        # --- Apply CI (UMEP lines 196-197) ---
        Tg = Tg * CI_TgG
        Tgwall = Tgwall * CI_TgG

        # --- Ground clip (UMEP lines 198-199, with landcover=1) ---
        Tg[Tg < 0] = 0.0

        return Tg, Tgwall, CI_TgG

    # Test scenarios: (name, inputs_dict)
    # Each scenario tests a specific aspect of the formula
    SCENARIOS = [
        (
            "noon_clear_cobblestone",
            dict(
                altmax=55.0,
                hour=12.0,
                snup=5.0,
                global_rad=600.0,
                rad_g0=650.0,
                zen_deg=35.0,
                sun_altitude=55.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "noon_clear_asphalt",
            dict(
                altmax=55.0,
                hour=12.0,
                snup=5.0,
                global_rad=600.0,
                rad_g0=650.0,
                zen_deg=35.0,
                sun_altitude=55.0,
                tgk_val=0.58,
                tstart_val=-9.78,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "afternoon_decline_15h",
            dict(
                altmax=55.0,
                hour=18.0,
                snup=5.0,
                global_rad=300.0,
                rad_g0=400.0,
                zen_deg=60.0,
                sun_altitude=30.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "evening_22h",
            dict(
                altmax=55.0,
                hour=22.0,
                snup=5.0,
                global_rad=0.0,
                rad_g0=0.0,
                zen_deg=90.0,
                sun_altitude=0.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "before_sunrise",
            dict(
                altmax=55.0,
                hour=3.0,
                snup=5.0,
                global_rad=0.0,
                rad_g0=0.0,
                zen_deg=90.0,
                sun_altitude=0.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "peak_at_tmaxlst",
            dict(
                altmax=55.0,
                hour=15.0,
                snup=5.0,
                global_rad=500.0,
                rad_g0=550.0,
                zen_deg=45.0,
                sun_altitude=45.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "cloudy_ci_low",
            dict(
                altmax=55.0,
                hour=12.0,
                snup=5.0,
                global_rad=200.0,
                rad_g0=650.0,
                zen_deg=35.0,
                sun_altitude=55.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "wood_wall_noon",
            dict(
                altmax=55.0,
                hour=12.0,
                snup=5.0,
                global_rad=600.0,
                rad_g0=650.0,
                zen_deg=35.0,
                sun_altitude=55.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.50,
                tstart_wall=-2.0,
                tmaxlst_wall=14.0,
            ),
        ),
        (
            "brick_wall_afternoon",
            dict(
                altmax=55.0,
                hour=18.0,
                snup=5.0,
                global_rad=300.0,
                rad_g0=400.0,
                zen_deg=60.0,
                sun_altitude=30.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=15.0,
                tgk_wall=0.40,
                tstart_wall=-4.0,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "grass_morning",
            dict(
                altmax=55.0,
                hour=8.0,
                snup=5.0,
                global_rad=300.0,
                rad_g0=320.0,
                zen_deg=60.0,
                sun_altitude=30.0,
                tgk_val=0.21,
                tstart_val=-3.38,
                tmaxlst_val=14.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "water",
            dict(
                altmax=55.0,
                hour=12.0,
                snup=5.0,
                global_rad=600.0,
                rad_g0=650.0,
                zen_deg=35.0,
                sun_altitude=55.0,
                tgk_val=0.0,
                tstart_val=0.0,
                tmaxlst_val=12.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=15.0,
            ),
        ),
        (
            "high_latitude_low_sun",
            dict(
                altmax=15.0,
                hour=12.0,
                snup=9.0,
                global_rad=100.0,
                rad_g0=120.0,
                zen_deg=78.0,
                sun_altitude=12.0,
                tgk_val=0.37,
                tstart_val=-3.41,
                tmaxlst_val=13.0,
                tgk_wall=0.37,
                tstart_wall=-3.41,
                tmaxlst_wall=13.0,
            ),
        ),
    ]

    @pytest.mark.parametrize("name,inputs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_rust_matches_umep_formula(self, name, inputs):
        """Rust output must match the UMEP Python formula exactly (within f32)."""
        shape = (3, 3)
        dectime_frac = inputs["hour"] / 24.0

        # Build grid inputs
        tgk_grid = np.full(shape, inputs["tgk_val"], dtype=np.float32)
        tstart_grid = np.full(shape, inputs["tstart_val"], dtype=np.float32)
        tmaxlst_grid = np.full(shape, inputs["tmaxlst_val"], dtype=np.float32)
        alb_grid = np.full(shape, 0.2, dtype=np.float32)
        emis_grid = np.full(shape, 0.95, dtype=np.float32)

        # UMEP reference (pure Python)
        umep_tg, umep_tgwall, umep_ci = self._umep_ground_temp(
            altmax=inputs["altmax"],
            dectime_frac=dectime_frac,
            snup_hours=inputs["snup"],
            global_rad=inputs["global_rad"],
            rad_g0=inputs["rad_g0"],
            zen_deg=inputs["zen_deg"],
            tgk=tgk_grid.astype(np.float64),
            tstart=tstart_grid.astype(np.float64),
            tmaxlst=tmaxlst_grid.astype(np.float64),
            tgk_wall=inputs["tgk_wall"],
            tstart_wall=inputs["tstart_wall"],
            tmaxlst_wall=inputs["tmaxlst_wall"],
            sun_altitude=inputs["sun_altitude"],
        )

        # Rust
        rust_tg, rust_tgwall, rust_ci, _, _ = ground.compute_ground_temperature(
            float(inputs["sun_altitude"]),  # ta (unused but required)
            inputs["sun_altitude"],
            inputs["altmax"],
            dectime_frac,
            inputs["snup"],
            inputs["global_rad"],
            inputs["rad_g0"],
            inputs["zen_deg"],
            alb_grid,
            emis_grid,
            tgk_grid,
            tstart_grid,
            tmaxlst_grid,
            tgk_wall=inputs["tgk_wall"],
            tstart_wall=inputs["tstart_wall"],
            tmaxlst_wall=inputs["tmaxlst_wall"],
        )
        rust_tg = np.array(rust_tg)

        # Compare CI_TgG
        np.testing.assert_allclose(
            rust_ci,
            umep_ci,
            atol=1e-5,
            err_msg=f"[{name}] CI_TgG differs",
        )

        # Compare ground temperature grid
        np.testing.assert_allclose(
            rust_tg,
            umep_tg.astype(np.float32),
            atol=1e-4,
            err_msg=f"[{name}] Tg grid differs",
        )

        # Compare wall temperature
        np.testing.assert_allclose(
            rust_tgwall,
            float(umep_tgwall),
            atol=1e-4,
            err_msg=f"[{name}] Tg_wall differs",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
