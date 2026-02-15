"""
Golden Regression Tests for PET (Physiologically Equivalent Temperature) Calculations

These tests compare the Rust PET implementation against pre-computed golden
fixtures generated from the UMEP Python module.

PET uses an iterative energy balance solver, which is slower than UTCI but
provides a more physiologically-based thermal comfort index.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import pet

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for PET calculations
# PET uses iterative solver; max observed: single 1.0e-2, grid 2.2e-2
RTOL = 0.005  # 0.5% relative tolerance
ATOL = 0.05  # 0.05°C absolute tolerance


@pytest.fixture(scope="module")
def single_point_data():
    """Load single-point PET test cases."""
    data = dict(np.load(FIXTURES_DIR / "pet_single_point.npz", allow_pickle=True))
    return {
        "inputs": data["inputs"],  # [n_tests, 4] -> [ta, rh, tmrt, va]
        "outputs": data["outputs"],  # [n_tests]
        "descriptions": data["descriptions"],
        "mbody": float(data["mbody"]),
        "age": float(data["age"]),
        "height": float(data["height"]),
        "activity": float(data["activity"]),
        "clo": float(data["clo"]),
        "sex": int(data["sex"]),
    }


@pytest.fixture(scope="module")
def grid_data():
    """Load grid PET test data."""
    params = dict(np.load(FIXTURES_DIR / "pet_grid_params.npz"))
    return {
        "ta": float(params["ta"]),
        "rh": float(params["rh"]),
        "tmrt": np.load(FIXTURES_DIR / "pet_grid_tmrt.npy"),
        "va": np.load(FIXTURES_DIR / "pet_grid_va.npy"),
        "expected": np.load(FIXTURES_DIR / "pet_grid_output.npy"),
        "mbody": float(params["mbody"]),
        "age": float(params["age"]),
        "height": float(params["height"]),
        "activity": float(params["activity"]),
        "clo": float(params["clo"]),
        "sex": int(params["sex"]),
    }


class TestGoldenPetSinglePoint:
    """Golden tests for single-point PET calculations."""

    def test_pet_single_comfortable(self, single_point_data):
        """Test PET for comfortable conditions."""
        self._test_case(single_point_data, "comfortable")

    def test_pet_single_hot_summer(self, single_point_data):
        """Test PET for hot summer day."""
        self._test_case(single_point_data, "hot_summer")

    def test_pet_single_cold_winter(self, single_point_data):
        """Test PET for cold winter day."""
        self._test_case(single_point_data, "cold_winter")

    def test_pet_single_tropical(self, single_point_data):
        """Test PET for tropical high humidity conditions."""
        self._test_case(single_point_data, "tropical")

    def test_pet_single_high_radiation(self, single_point_data):
        """Test PET for high radiation (large Tmrt-Ta delta)."""
        self._test_case(single_point_data, "high_radiation")

    def _test_case(self, data, description):
        """Helper to test a specific case by description."""
        idx = list(data["descriptions"]).index(description)
        ta, rh, tmrt, va = data["inputs"][idx]
        expected = data["outputs"][idx]

        result = pet.pet_calculate(
            float(ta),
            float(rh),
            float(tmrt),
            float(va),
            data["mbody"],
            data["age"],
            data["height"],
            data["activity"],
            data["clo"],
            data["sex"],
        )

        np.testing.assert_allclose(
            result,
            expected,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"PET mismatch for {description}: got {result}, expected {expected}",
        )


class TestGoldenPetGrid:
    """Golden tests for grid PET calculations."""

    def test_pet_grid_matches_golden(self, grid_data):
        """Grid PET should match golden fixture."""
        result = pet.pet_grid(
            grid_data["ta"],
            grid_data["rh"],
            grid_data["tmrt"],
            grid_data["va"],
            grid_data["mbody"],
            grid_data["age"],
            grid_data["height"],
            grid_data["activity"],
            grid_data["clo"],
            grid_data["sex"],
        )
        result_arr = np.array(result)

        # Mask out invalid values (-9999)
        valid_mask = grid_data["expected"] > -999

        np.testing.assert_allclose(
            result_arr[valid_mask],
            grid_data["expected"][valid_mask],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Grid PET differs from golden fixture",
        )

    def test_pet_grid_shape(self, grid_data):
        """Grid PET should have correct shape."""
        result = pet.pet_grid(
            grid_data["ta"],
            grid_data["rh"],
            grid_data["tmrt"],
            grid_data["va"],
            grid_data["mbody"],
            grid_data["age"],
            grid_data["height"],
            grid_data["activity"],
            grid_data["clo"],
            grid_data["sex"],
        )
        assert np.array(result).shape == grid_data["expected"].shape


class TestGoldenPetProperties:
    """Verify golden fixtures maintain expected physical properties."""

    def test_pet_range(self, single_point_data):
        """PET values should be in physically plausible range."""
        outputs = single_point_data["outputs"]
        # PET typically ranges from -20°C to +50°C for outdoor conditions
        assert np.all(outputs > -30), "PET values below plausible range"
        assert np.all(outputs < 60), "PET values above plausible range"

    def test_pet_responds_to_tmrt(self, single_point_data):
        """Higher Tmrt should generally increase PET."""
        # Compare comfortable (Tmrt=22) vs high_radiation (Tmrt=55)
        idx_comfort = list(single_point_data["descriptions"]).index("comfortable")
        idx_radiation = list(single_point_data["descriptions"]).index("high_radiation")

        pet_comfort = single_point_data["outputs"][idx_comfort]
        pet_radiation = single_point_data["outputs"][idx_radiation]

        # High radiation case has much higher Tmrt, so PET should be higher
        assert pet_radiation > pet_comfort, "PET should increase with higher Tmrt"

    def test_pet_hot_vs_cold(self, single_point_data):
        """Hot conditions should have higher PET than cold."""
        idx_cold = list(single_point_data["descriptions"]).index("cold_winter")
        idx_hot = list(single_point_data["descriptions"]).index("hot_summer")

        pet_cold = single_point_data["outputs"][idx_cold]
        pet_hot = single_point_data["outputs"][idx_hot]

        assert pet_hot > pet_cold, "Hot summer should have higher PET than cold winter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
