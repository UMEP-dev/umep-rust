"""
Ground Temperature Physics Property Tests

Tests that compute_ground_temperature satisfies physical invariants using
synthetic inputs. Complements the golden regression tests in
tests/golden/test_golden_ground_temp.py.

The sinusoidal ground temperature model computes a diurnal surface
temperature deviation from air temperature, modulated by:
- Maximum sun altitude (altmax): controls amplitude
- Clearness index (CI): cloud/haze correction
- Material parameters (tgk, tstart, tmaxlst): land-cover-specific response
"""

import numpy as np
import pytest
from solweig.rustalgos import ground

SHAPE = (3, 3)


def _compute_tg(
    hour=12.0,
    altmax=55.0,
    snup=5.0,
    global_rad=600.0,
    rad_g0=650.0,
    zen_deg=45.0,
    sun_altitude=45.0,
    tgk=0.37,
    tstart=-3.41,
    tmaxlst=15.0,
    alb=0.2,
    emis=0.95,
):
    """Helper: compute ground temperature with default cobblestone params."""
    tg, tg_wall, ci, _, _ = ground.compute_ground_temperature(
        20.0,  # ta (unused in sinusoidal model but required)
        sun_altitude,
        altmax,
        hour / 24.0,  # dectime
        snup,
        global_rad,
        rad_g0,
        zen_deg,
        np.full(SHAPE, alb, dtype=np.float32),
        np.full(SHAPE, emis, dtype=np.float32),
        np.full(SHAPE, tgk, dtype=np.float32),
        np.full(SHAPE, tstart, dtype=np.float32),
        np.full(SHAPE, tmaxlst, dtype=np.float32),
    )
    return np.array(tg), tg_wall, ci


class TestGroundTempSolarResponse:
    """Ground temperature should respond to solar radiation inputs."""

    def test_higher_clearness_higher_ground_temp(self):
        """Clearer skies (higher global_rad / rad_g0) increase ground temp.

        The clearness index CI = global_rad/rad_g0 + (1 - corr) scales
        the sinusoidal amplitude, so more direct sun means warmer ground.
        """
        # Cloudy: low ratio
        tg_cloudy, _, ci_cloudy = _compute_tg(global_rad=200.0, rad_g0=650.0)
        # Clear: high ratio
        tg_clear, _, ci_clear = _compute_tg(global_rad=600.0, rad_g0=650.0)

        assert ci_clear > ci_cloudy, f"Clear CI ({ci_clear:.3f}) should exceed cloudy CI ({ci_cloudy:.3f})"
        assert tg_clear[0, 0] > tg_cloudy[0, 0], (
            f"Clear ground temp ({tg_clear[0, 0]:.2f}) should exceed cloudy ({tg_cloudy[0, 0]:.2f})"
        )

    def test_higher_altmax_higher_amplitude(self):
        """Higher maximum sun altitude produces larger temperature amplitude.

        Tgamp = tgk * altmax + tstart, so higher altmax means bigger
        sinusoidal swing.
        """
        tg_low, _, _ = _compute_tg(altmax=20.0, sun_altitude=20.0, zen_deg=70.0)
        tg_mid, _, _ = _compute_tg(altmax=45.0, sun_altitude=45.0, zen_deg=45.0)
        tg_high, _, _ = _compute_tg(altmax=70.0, sun_altitude=70.0, zen_deg=20.0)

        assert tg_low[0, 0] < tg_mid[0, 0] < tg_high[0, 0], (
            f"Higher altmax should give higher ground temp: "
            f"low={tg_low[0, 0]:.2f}, mid={tg_mid[0, 0]:.2f}, high={tg_high[0, 0]:.2f}"
        )


class TestGroundTempDiurnalCycle:
    """Ground temperature should follow a realistic diurnal pattern."""

    def test_before_sunrise_zero(self):
        """Temperature deviation should be zero before sunrise."""
        tg, tg_wall, _ = _compute_tg(hour=3.0, sun_altitude=0.0, global_rad=0.0, rad_g0=0.0, zen_deg=90.0)
        assert np.all(tg == 0.0), "Ground temp before sunrise should be 0"
        assert tg_wall == 0.0, "Wall temp before sunrise should be 0"

    def test_temperature_rises_morning(self):
        """Temperature should increase from sunrise towards peak."""
        tg_early, _, _ = _compute_tg(hour=7.0)
        tg_mid, _, _ = _compute_tg(hour=10.0)
        tg_noon, _, _ = _compute_tg(hour=12.0)

        assert tg_early[0, 0] < tg_mid[0, 0] < tg_noon[0, 0], (
            f"Morning warming: 7h={tg_early[0, 0]:.2f}, 10h={tg_mid[0, 0]:.2f}, 12h={tg_noon[0, 0]:.2f}"
        )

    def test_temperature_peaks_at_tmaxlst(self):
        """Peak temperature should occur at TmaxLST (default 15:00)."""
        hours = [9, 12, 15, 18, 21]
        vals = [_compute_tg(hour=float(h))[0][0, 0] for h in hours]

        peak_idx = vals.index(max(vals))
        assert hours[peak_idx] == 15, f"Peak should be at TmaxLST=15h, got peak at {hours[peak_idx]}h"

    def test_temperature_declines_afternoon(self):
        """Temperature should decline after TmaxLST."""
        tg_peak, _, _ = _compute_tg(hour=15.0)
        tg_after, _, _ = _compute_tg(hour=18.0)
        tg_evening, _, _ = _compute_tg(hour=21.0)

        assert tg_peak[0, 0] > tg_after[0, 0] > tg_evening[0, 0], (
            f"Afternoon cooling: 15h={tg_peak[0, 0]:.2f}, 18h={tg_after[0, 0]:.2f}, 21h={tg_evening[0, 0]:.2f}"
        )

    def test_ground_temp_always_non_negative(self):
        """Ground temperature deviation should never be negative (clamped to 0)."""
        for h in range(0, 24):
            sun_alt = max(0.0, 45.0 * np.sin(np.pi * (h - 5) / 14)) if 5 <= h <= 19 else 0.0
            tg, _, _ = _compute_tg(
                hour=float(h),
                sun_altitude=sun_alt,
                global_rad=max(0.0, 600.0 * np.sin(np.pi * (h - 5) / 14)) if 5 <= h <= 19 else 0.0,
                rad_g0=max(0.0, 650.0 * np.sin(np.pi * (h - 5) / 14)) if 5 <= h <= 19 else 0.0,
                zen_deg=90.0 - sun_alt,
            )
            assert np.all(tg >= 0.0), f"Ground temp at {h}h should be >= 0, got {tg.min():.3f}"


class TestGroundTempMaterialResponse:
    """Different material parameters should produce different temperatures."""

    def test_higher_tgk_higher_amplitude(self):
        """Higher tgk (thermal gain coefficient) should increase temperature.

        Asphalt (tgk=0.58) heats more than grass (tgk=0.21).
        """
        tg_grass, _, _ = _compute_tg(tgk=0.21, tstart=-3.38)
        tg_asphalt, _, _ = _compute_tg(tgk=0.58, tstart=-9.78)

        # At noon with altmax=55, asphalt amplitude = 0.58*55-9.78 = 22.12
        # grass amplitude = 0.21*55-3.38 = 8.17
        assert tg_asphalt[0, 0] > tg_grass[0, 0], (
            f"Asphalt ({tg_asphalt[0, 0]:.2f}) should be warmer than grass ({tg_grass[0, 0]:.2f})"
        )

    def test_water_has_zero_deviation(self):
        """Water surface (tgk=0, tstart=0) should have zero temperature deviation."""
        tg_water, _, _ = _compute_tg(tgk=0.0, tstart=0.0)
        assert np.all(tg_water == 0.0), "Water surface should have zero temp deviation"

    def test_wall_material_affects_wall_temp(self):
        """Different wall material parameters should change wall temperature."""
        _, wall_default, _ = _compute_tg()
        _, wall_alt, _, _, _ = ground.compute_ground_temperature(
            20.0,
            45.0,
            55.0,
            12.0 / 24.0,
            5.0,
            600.0,
            650.0,
            45.0,
            np.full(SHAPE, 0.2, dtype=np.float32),
            np.full(SHAPE, 0.95, dtype=np.float32),
            np.full(SHAPE, 0.37, dtype=np.float32),
            np.full(SHAPE, -3.41, dtype=np.float32),
            np.full(SHAPE, 15.0, dtype=np.float32),
            tgk_wall=0.50,
            tstart_wall=-2.0,
            tmaxlst_wall=14.0,
        )

        assert wall_default != wall_alt, (
            f"Different wall params should give different temps: default={wall_default:.2f}, alt={wall_alt:.2f}"
        )


class TestGroundTempClearnessIndex:
    """Clearness index CI should be physically bounded."""

    def test_ci_capped_at_one(self):
        """CI should never exceed 1.0 (clear sky limit)."""
        # global_rad > rad_g0 would give CI > 1 without clamping
        _, _, ci = _compute_tg(global_rad=800.0, rad_g0=650.0)
        assert ci <= 1.0, f"CI should be capped at 1.0, got {ci:.3f}"

    def test_ci_non_negative(self):
        """CI should be non-negative."""
        _, _, ci = _compute_tg(global_rad=50.0, rad_g0=650.0)
        assert ci >= 0.0, f"CI should be non-negative, got {ci:.3f}"

    def test_ci_equals_one_at_night(self):
        """CI should default to 1.0 when sun is below horizon."""
        _, _, ci = _compute_tg(hour=3.0, sun_altitude=0.0, global_rad=0.0, rad_g0=0.0, zen_deg=90.0)
        assert ci == 1.0, f"CI should be 1.0 at night, got {ci:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
