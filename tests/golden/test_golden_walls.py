"""
Golden Regression Tests for Wall Temperature Deviation Calculations

These tests verify that the Rust compute_ground_temperature function correctly
calculates both ground (Tg) and wall (Tg_wall) temperature deviations from
air temperature based on sun position and land cover properties.

The model uses a sinusoidal daily pattern:
    Tg = Tgamp * sin(phase * PI/2) * CI_correction
where:
    Tgamp = TgK * altmax + Tstart
    phase = (dectime - snup) / (tmaxlst - snup)
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import ground

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for wall temperature calculations
RTOL = 1e-3
ATOL = 0.05  # 0.05°C


@pytest.fixture
def wall_temp_inputs():
    """Load wall temperature input fixtures."""
    return {
        "alb": np.load(FIXTURES_DIR / "wall_temp_input_alb.npy"),
        "emis": np.load(FIXTURES_DIR / "wall_temp_input_emis.npy"),
        "tgk": np.load(FIXTURES_DIR / "wall_temp_input_tgk.npy"),
        "tstart": np.load(FIXTURES_DIR / "wall_temp_input_tstart.npy"),
        "tmaxlst": np.load(FIXTURES_DIR / "wall_temp_input_tmaxlst.npy"),
    }


@pytest.fixture
def wall_temp_expected():
    """Load expected wall temperature outputs."""
    data = dict(np.load(FIXTURES_DIR / "wall_temp_output.npz"))
    return {
        "tg": data["tg"],
        "tg_wall": float(data["tg_wall"]),
        "ci_tg": float(data["ci_tg"]),
        "ta": float(data["ta"]),
        "sun_altitude": float(data["sun_altitude"]),
        "altmax": float(data["altmax"]),
        "dectime": float(data["dectime"]),
        "snup": float(data["snup"]),
        "global_rad": float(data["global_rad"]),
        "rad_g0": float(data["rad_g0"]),
        "zen_deg": float(data["zen_deg"]),
    }


class TestGoldenWallTemperature:
    """Golden tests for wall temperature deviation calculations."""

    def test_ground_temp_matches_golden(self, wall_temp_inputs, wall_temp_expected):
        """Ground temperature deviation should match golden fixture."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        np.testing.assert_allclose(
            np.array(tg),
            wall_temp_expected["tg"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Ground temperature (Tg) differs from golden fixture",
        )

    def test_wall_temp_matches_golden(self, wall_temp_inputs, wall_temp_expected):
        """Wall temperature deviation should match golden fixture."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        np.testing.assert_allclose(
            tg_wall,
            wall_temp_expected["tg_wall"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Wall temperature (Tg_wall) differs from golden fixture",
        )

    def test_clearness_index_matches_golden(self, wall_temp_inputs, wall_temp_expected):
        """Clearness index correction should match golden fixture."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        np.testing.assert_allclose(
            ci_tg,
            wall_temp_expected["ci_tg"],
            rtol=RTOL,
            atol=0.01,
            err_msg="Clearness index (CI_Tg) differs from golden fixture",
        )

    def test_output_shape(self, wall_temp_inputs, wall_temp_expected):
        """Output arrays should match input shape."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        assert np.array(tg).shape == wall_temp_inputs["tgk"].shape


class TestGoldenWallTempProperties:
    """Verify physical properties of wall temperature model."""

    def test_temp_non_negative(self, wall_temp_inputs, wall_temp_expected):
        """Temperature deviations should be non-negative during daytime."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        assert np.all(np.array(tg) >= 0), "Ground Tg should be non-negative"
        assert tg_wall >= 0, "Wall Tg should be non-negative"

    def test_land_cover_variation(self, wall_temp_inputs, wall_temp_expected):
        """Different land covers (TgK) should produce different temperatures."""
        tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )
        tg_arr = np.array(tg)

        # Upper half has different TgK (asphalt) than lower half (grass)
        mean_upper = np.mean(tg_arr[:10, :])
        mean_lower = np.mean(tg_arr[10:, :])

        # Different land covers should produce different temperatures
        assert abs(mean_upper - mean_lower) > 0.5, "Different land covers should produce different Tg values"

    def test_higher_altmax_higher_temp(self, wall_temp_inputs, wall_temp_expected):
        """Higher max sun altitude should produce higher temperature amplitude."""
        # Calculate with normal altmax
        tg_normal, _, _, _, _ = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            wall_temp_expected["altmax"],  # 65°
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        # Calculate with higher altmax
        tg_higher, _, _, _, _ = ground.compute_ground_temperature(
            wall_temp_expected["ta"],
            wall_temp_expected["sun_altitude"],
            80.0,  # Higher max altitude
            wall_temp_expected["dectime"],
            wall_temp_expected["snup"],
            wall_temp_expected["global_rad"],
            wall_temp_expected["rad_g0"],
            wall_temp_expected["zen_deg"],
            wall_temp_inputs["alb"],
            wall_temp_inputs["emis"],
            wall_temp_inputs["tgk"],
            wall_temp_inputs["tstart"],
            wall_temp_inputs["tmaxlst"],
        )

        assert np.mean(np.array(tg_higher)) > np.mean(np.array(tg_normal)), (
            "Higher max altitude should produce higher Tg"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
