"""
Golden Regression Tests for UTCI (Universal Thermal Climate Index) Calculations

These tests compare the Rust UTCI implementation against pre-computed golden
fixtures generated from the UMEP Python module.

Both implementations use the same 6th order polynomial approximation from
Bröde et al., so results should match to floating-point precision.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import utci

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for UTCI calculations
# Both use identical polynomial, so should match very closely
RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def single_point_data():
    """Load single-point UTCI test cases."""
    data = dict(np.load(FIXTURES_DIR / "utci_single_point.npz", allow_pickle=True))
    return {
        "inputs": data["inputs"],  # [n_tests, 4] -> [ta, rh, tmrt, va]
        "outputs": data["outputs"],  # [n_tests]
        "descriptions": data["descriptions"],
    }


@pytest.fixture
def grid_data():
    """Load grid UTCI test data."""
    params = dict(np.load(FIXTURES_DIR / "utci_grid_params.npz"))
    return {
        "ta": float(params["ta"]),
        "rh": float(params["rh"]),
        "tmrt": np.load(FIXTURES_DIR / "utci_grid_tmrt.npy"),
        "va": np.load(FIXTURES_DIR / "utci_grid_va.npy"),
        "expected": np.load(FIXTURES_DIR / "utci_grid_output.npy"),
    }


class TestGoldenUtciSinglePoint:
    """Golden tests for single-point UTCI calculations."""

    def test_utci_single_comfortable(self, single_point_data):
        """Test UTCI for comfortable conditions."""
        self._test_case(single_point_data, "comfortable")

    def test_utci_single_hot_summer(self, single_point_data):
        """Test UTCI for hot summer day."""
        self._test_case(single_point_data, "hot_summer")

    def test_utci_single_cold_winter(self, single_point_data):
        """Test UTCI for cold winter day."""
        self._test_case(single_point_data, "cold_winter")

    def test_utci_single_tropical(self, single_point_data):
        """Test UTCI for tropical high humidity conditions."""
        self._test_case(single_point_data, "tropical")

    def test_utci_single_windy(self, single_point_data):
        """Test UTCI for windy conditions."""
        self._test_case(single_point_data, "windy")

    def test_utci_single_high_radiation(self, single_point_data):
        """Test UTCI for high radiation (large Tmrt-Ta delta)."""
        self._test_case(single_point_data, "high_radiation")

    def test_utci_single_low_wind(self, single_point_data):
        """Test UTCI for low wind edge case."""
        self._test_case(single_point_data, "low_wind")

    def _test_case(self, data, description):
        """Helper to test a specific case by description."""
        idx = list(data["descriptions"]).index(description)
        ta, rh, tmrt, va = data["inputs"][idx]
        expected = data["outputs"][idx]

        result = utci.utci_single(float(ta), float(rh), float(tmrt), float(va))

        np.testing.assert_allclose(
            result,
            expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"UTCI mismatch for {description}: got {result}, expected {expected}",
        )


class TestGoldenUtciGrid:
    """Golden tests for grid UTCI calculations."""

    def test_utci_grid_matches_golden(self, grid_data):
        """Grid UTCI should match golden fixture."""
        result = utci.utci_grid(
            grid_data["ta"],
            grid_data["rh"],
            grid_data["tmrt"],
            grid_data["va"],
        )
        result_arr = np.array(result)

        # Mask out invalid values (-9999)
        valid_mask = grid_data["expected"] > -999

        np.testing.assert_allclose(
            result_arr[valid_mask],
            grid_data["expected"][valid_mask],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Grid UTCI differs from golden fixture",
        )

    def test_utci_grid_shape(self, grid_data):
        """Grid UTCI should have correct shape."""
        result = utci.utci_grid(
            grid_data["ta"],
            grid_data["rh"],
            grid_data["tmrt"],
            grid_data["va"],
        )
        assert np.array(result).shape == grid_data["expected"].shape


class TestGoldenUtciProperties:
    """Verify golden fixtures maintain expected physical properties."""

    def test_utci_range(self, single_point_data):
        """UTCI values should be in physically plausible range."""
        outputs = single_point_data["outputs"]
        # UTCI typically ranges from -50°C to +50°C for outdoor conditions
        assert np.all(outputs > -60), "UTCI values below plausible range"
        assert np.all(outputs < 60), "UTCI values above plausible range"

    def test_utci_responds_to_tmrt(self, single_point_data):
        """Higher Tmrt should generally increase UTCI."""
        # Compare comfortable (Tmrt=22) vs high_radiation (Tmrt=60)
        idx_comfort = list(single_point_data["descriptions"]).index("comfortable")
        idx_radiation = list(single_point_data["descriptions"]).index("high_radiation")

        utci_comfort = single_point_data["outputs"][idx_comfort]
        utci_radiation = single_point_data["outputs"][idx_radiation]

        # High radiation case has much higher Tmrt, so UTCI should be higher
        assert utci_radiation > utci_comfort, "UTCI should increase with higher Tmrt"

    def test_utci_responds_to_wind(self, single_point_data):
        """Higher wind speed should generally reduce UTCI in warm conditions."""
        # Compare hot_summer (va=1.0) vs windy (va=8.0)
        # Windy has lower temp but high wind should still show cooling effect
        idx_windy = list(single_point_data["descriptions"]).index("windy")
        idx_hot = list(single_point_data["descriptions"]).index("hot_summer")

        # Windy case: Ta=25, Tmrt=30, va=8.0
        # Hot case: Ta=35, Tmrt=55, va=1.0
        # Hot case should have higher UTCI due to higher temp and Tmrt
        utci_windy = single_point_data["outputs"][idx_windy]
        utci_hot = single_point_data["outputs"][idx_hot]

        assert utci_hot > utci_windy, "Hot summer should have higher UTCI than windy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
