"""
Golden Regression Tests for Tmrt (Mean Radiant Temperature) Calculations

These tests verify that the Rust Tmrt implementation correctly computes
Mean Radiant Temperature from radiation budget components using the
Stefan-Boltzmann formula:

    Tmrt = (Sstr / (abs_l * SBC))^0.25 - 273.15

where Sstr is the total absorbed shortwave and longwave radiation.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import tmrt

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for Tmrt calculations
RTOL = 1e-4
ATOL = 0.01  # 0.01°C


@pytest.fixture
def tmrt_inputs():
    """Load Tmrt radiation input fixtures."""
    return {
        "kdown": np.load(FIXTURES_DIR / "tmrt_input_kdown.npy"),
        "kup": np.load(FIXTURES_DIR / "tmrt_input_kup.npy"),
        "ldown": np.load(FIXTURES_DIR / "tmrt_input_ldown.npy"),
        "lup": np.load(FIXTURES_DIR / "tmrt_input_lup.npy"),
        "kside_n": np.load(FIXTURES_DIR / "tmrt_input_kside_n.npy"),
        "kside_e": np.load(FIXTURES_DIR / "tmrt_input_kside_e.npy"),
        "kside_s": np.load(FIXTURES_DIR / "tmrt_input_kside_s.npy"),
        "kside_w": np.load(FIXTURES_DIR / "tmrt_input_kside_w.npy"),
        "kside_total": np.load(FIXTURES_DIR / "tmrt_input_kside_total.npy"),
        "lside_n": np.load(FIXTURES_DIR / "tmrt_input_lside_n.npy"),
        "lside_e": np.load(FIXTURES_DIR / "tmrt_input_lside_e.npy"),
        "lside_s": np.load(FIXTURES_DIR / "tmrt_input_lside_s.npy"),
        "lside_w": np.load(FIXTURES_DIR / "tmrt_input_lside_w.npy"),
        "lside_total": np.load(FIXTURES_DIR / "tmrt_input_lside_total.npy"),
    }


@pytest.fixture
def tmrt_expected():
    """Load expected Tmrt outputs."""
    return {
        "aniso": np.load(FIXTURES_DIR / "tmrt_output_aniso.npy"),
        "iso": np.load(FIXTURES_DIR / "tmrt_output_iso.npy"),
    }


@pytest.fixture
def tmrt_params():
    """Load Tmrt parameters."""
    data = dict(np.load(FIXTURES_DIR / "tmrt_params.npz"))
    return {
        "abs_k": float(data["abs_k"]),
        "abs_l": float(data["abs_l"]),
    }


class TestGoldenTmrt:
    """Golden tests for Tmrt calculations."""

    def test_tmrt_anisotropic_matches_golden(self, tmrt_inputs, tmrt_expected, tmrt_params):
        """Tmrt with anisotropic sky model should match golden fixture."""
        params = tmrt.TmrtParams(
            abs_k=tmrt_params["abs_k"],
            abs_l=tmrt_params["abs_l"],
            is_standing=True,
            use_anisotropic_sky=True,
        )

        result = tmrt.compute_tmrt(
            tmrt_inputs["kdown"],
            tmrt_inputs["kup"],
            tmrt_inputs["ldown"],
            tmrt_inputs["lup"],
            tmrt_inputs["kside_n"],
            tmrt_inputs["kside_e"],
            tmrt_inputs["kside_s"],
            tmrt_inputs["kside_w"],
            tmrt_inputs["lside_n"],
            tmrt_inputs["lside_e"],
            tmrt_inputs["lside_s"],
            tmrt_inputs["lside_w"],
            tmrt_inputs["kside_total"],
            tmrt_inputs["lside_total"],
            params,
        )
        result_arr = np.array(result)

        np.testing.assert_allclose(
            result_arr,
            tmrt_expected["aniso"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Anisotropic Tmrt differs from golden fixture",
        )

    def test_tmrt_isotropic_matches_golden(self, tmrt_inputs, tmrt_expected, tmrt_params):
        """Tmrt with isotropic sky model should match golden fixture."""
        params = tmrt.TmrtParams(
            abs_k=tmrt_params["abs_k"],
            abs_l=tmrt_params["abs_l"],
            is_standing=True,
            use_anisotropic_sky=False,
        )

        result = tmrt.compute_tmrt(
            tmrt_inputs["kdown"],
            tmrt_inputs["kup"],
            tmrt_inputs["ldown"],
            tmrt_inputs["lup"],
            tmrt_inputs["kside_n"],
            tmrt_inputs["kside_e"],
            tmrt_inputs["kside_s"],
            tmrt_inputs["kside_w"],
            tmrt_inputs["lside_n"],
            tmrt_inputs["lside_e"],
            tmrt_inputs["lside_s"],
            tmrt_inputs["lside_w"],
            tmrt_inputs["kside_total"],
            tmrt_inputs["lside_total"],
            params,
        )
        result_arr = np.array(result)

        np.testing.assert_allclose(
            result_arr,
            tmrt_expected["iso"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Isotropic Tmrt differs from golden fixture",
        )

    def test_tmrt_shape(self, tmrt_inputs, tmrt_params):
        """Tmrt output should match input shape."""
        params = tmrt.TmrtParams(
            abs_k=tmrt_params["abs_k"],
            abs_l=tmrt_params["abs_l"],
            is_standing=True,
            use_anisotropic_sky=True,
        )

        result = tmrt.compute_tmrt(
            tmrt_inputs["kdown"],
            tmrt_inputs["kup"],
            tmrt_inputs["ldown"],
            tmrt_inputs["lup"],
            tmrt_inputs["kside_n"],
            tmrt_inputs["kside_e"],
            tmrt_inputs["kside_s"],
            tmrt_inputs["kside_w"],
            tmrt_inputs["lside_n"],
            tmrt_inputs["lside_e"],
            tmrt_inputs["lside_s"],
            tmrt_inputs["lside_w"],
            tmrt_inputs["kside_total"],
            tmrt_inputs["lside_total"],
            params,
        )

        assert np.array(result).shape == tmrt_inputs["kdown"].shape


class TestGoldenTmrtProperties:
    """Verify golden fixtures maintain expected physical properties."""

    def test_tmrt_range(self, tmrt_expected):
        """Tmrt values should be in physically plausible range."""
        for name, arr in tmrt_expected.items():
            assert np.all(arr >= -50), f"{name} Tmrt below -50°C"
            assert np.all(arr <= 80), f"{name} Tmrt above 80°C"

    def test_aniso_vs_iso_difference(self, tmrt_expected):
        """Anisotropic model should produce slightly different Tmrt than isotropic."""
        # Due to the additional Lside*Fcyl term in anisotropic mode
        diff = np.abs(tmrt_expected["aniso"] - tmrt_expected["iso"])
        mean_diff = np.mean(diff)

        # Should be some difference (anisotropic adds more longwave)
        assert mean_diff > 0.5, "Anisotropic and isotropic should differ"
        # But not too large
        assert mean_diff < 15, "Anisotropic/isotropic difference too large"

    def test_tmrt_increases_with_radiation(self, tmrt_inputs, tmrt_params):
        """Higher radiation should produce higher Tmrt."""
        params = tmrt.TmrtParams(
            abs_k=tmrt_params["abs_k"],
            abs_l=tmrt_params["abs_l"],
            is_standing=True,
            use_anisotropic_sky=True,
        )

        # Compute Tmrt with normal inputs
        result_normal = np.array(
            tmrt.compute_tmrt(
                tmrt_inputs["kdown"],
                tmrt_inputs["kup"],
                tmrt_inputs["ldown"],
                tmrt_inputs["lup"],
                tmrt_inputs["kside_n"],
                tmrt_inputs["kside_e"],
                tmrt_inputs["kside_s"],
                tmrt_inputs["kside_w"],
                tmrt_inputs["lside_n"],
                tmrt_inputs["lside_e"],
                tmrt_inputs["lside_s"],
                tmrt_inputs["lside_w"],
                tmrt_inputs["kside_total"],
                tmrt_inputs["lside_total"],
                params,
            )
        )

        # Compute with doubled shortwave
        result_doubled = np.array(
            tmrt.compute_tmrt(
                tmrt_inputs["kdown"] * 2,
                tmrt_inputs["kup"] * 2,
                tmrt_inputs["ldown"],
                tmrt_inputs["lup"],
                tmrt_inputs["kside_n"] * 2,
                tmrt_inputs["kside_e"] * 2,
                tmrt_inputs["kside_s"] * 2,
                tmrt_inputs["kside_w"] * 2,
                tmrt_inputs["lside_n"],
                tmrt_inputs["lside_e"],
                tmrt_inputs["lside_s"],
                tmrt_inputs["lside_w"],
                tmrt_inputs["kside_total"] * 2,
                tmrt_inputs["lside_total"],
                params,
            )
        )

        # Mean Tmrt should be higher with more radiation
        assert np.mean(result_doubled) > np.mean(result_normal), "Doubling shortwave radiation should increase Tmrt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
