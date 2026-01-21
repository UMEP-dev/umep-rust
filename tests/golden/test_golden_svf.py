"""
Golden Regression Tests for Sky View Factor (SVF) Calculations

These tests compare the Rust SVF algorithm implementation against
pre-computed golden fixtures generated from the UMEP Python module.

KNOWN DIFFERENCE: The UMEP Python svfForProcessing153 uses shadowingfunction_20
internally, while Rust uses shadowingfunction_wallheight_23. This causes small
numerical differences (~1% max) in some SVF components. See CHANGES.md for details.

Test strategy:
- South/West SVF: Strict tolerance (1e-5) - should match exactly
- North/East/Total/Veg SVF: Relaxed tolerance (0.02) - known ~1% difference
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import shadowing, skyview

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for components that match exactly (South, West)
STRICT_RTOL = 1e-5
STRICT_ATOL = 1e-5

# Relaxed tolerance for components with known differences (North, East, Total, Veg)
# Max observed difference is ~1.1%, so 2% tolerance is reasonable
RELAXED_RTOL = 0.02
RELAXED_ATOL = 0.02


@pytest.fixture
def input_data():
    """Load input data from golden fixtures."""
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy"),
        "cdsm": np.load(FIXTURES_DIR / "input_cdsm.npy"),
        "tdsm": np.load(FIXTURES_DIR / "input_tdsm.npy"),
        "params": dict(np.load(FIXTURES_DIR / "input_params.npz")),
    }


@pytest.fixture
def svf_golden():
    """Load golden SVF fixtures."""
    return {
        "svf": np.load(FIXTURES_DIR / "svf_total.npy"),
        "svf_north": np.load(FIXTURES_DIR / "svf_north.npy"),
        "svf_east": np.load(FIXTURES_DIR / "svf_east.npy"),
        "svf_south": np.load(FIXTURES_DIR / "svf_south.npy"),
        "svf_west": np.load(FIXTURES_DIR / "svf_west.npy"),
        "svf_veg": np.load(FIXTURES_DIR / "svf_veg.npy"),
    }


@pytest.fixture
def svf_result(input_data):
    """Compute current SVF result."""
    shadowing.disable_gpu()
    return skyview.calculate_svf(
        input_data["dsm"],
        input_data["cdsm"],
        input_data["tdsm"],
        float(input_data["params"]["scale"]),
        True,  # usevegdem
        float(input_data["params"]["amaxvalue"]),
        2,  # patch_option
        None,  # min_sun_elev
        None,  # progress_callback
    )


class TestGoldenSvf:
    """Golden tests for SVF calculations.

    Note: Some tests use relaxed tolerance due to known differences
    between UMEP Python (shadowingfunction_20) and Rust (shadowingfunction_23).
    See CHANGES.md for details.
    """

    def test_svf_total_matches_golden(self, svf_result, svf_golden):
        """Total SVF should match golden fixture within relaxed tolerance.

        Known difference: ~1% due to different shadow algorithms.
        """
        np.testing.assert_allclose(
            np.array(svf_result.svf),
            svf_golden["svf"],
            rtol=RELAXED_RTOL,
            atol=RELAXED_ATOL,
            err_msg="Total SVF differs from golden fixture beyond 2% tolerance",
        )

    def test_svf_north_matches_golden(self, svf_result, svf_golden):
        """North SVF should match golden fixture within relaxed tolerance.

        Known difference: ~1% due to different shadow algorithms.
        """
        np.testing.assert_allclose(
            np.array(svf_result.svf_north),
            svf_golden["svf_north"],
            rtol=RELAXED_RTOL,
            atol=RELAXED_ATOL,
            err_msg="North SVF differs from golden fixture beyond 2% tolerance",
        )

    def test_svf_east_matches_golden(self, svf_result, svf_golden):
        """East SVF should match golden fixture within relaxed tolerance.

        Known difference: ~1% due to different shadow algorithms.
        """
        np.testing.assert_allclose(
            np.array(svf_result.svf_east),
            svf_golden["svf_east"],
            rtol=RELAXED_RTOL,
            atol=RELAXED_ATOL,
            err_msg="East SVF differs from golden fixture beyond 2% tolerance",
        )

    def test_svf_south_matches_golden(self, svf_result, svf_golden):
        """South SVF should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(svf_result.svf_south),
            svf_golden["svf_south"],
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
            err_msg="South SVF differs from golden fixture",
        )

    def test_svf_west_matches_golden(self, svf_result, svf_golden):
        """West SVF should match golden fixture exactly."""
        np.testing.assert_allclose(
            np.array(svf_result.svf_west),
            svf_golden["svf_west"],
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
            err_msg="West SVF differs from golden fixture",
        )

    def test_svf_veg_matches_golden(self, svf_result, svf_golden):
        """Vegetation SVF should match golden fixture within relaxed tolerance.

        Known difference: ~1% due to different shadow algorithms.
        """
        np.testing.assert_allclose(
            np.array(svf_result.svf_veg),
            svf_golden["svf_veg"],
            rtol=RELAXED_RTOL,
            atol=RELAXED_ATOL,
            err_msg="Vegetation SVF differs from golden fixture beyond 2% tolerance",
        )


class TestGoldenSvfProperties:
    """Verify golden fixtures maintain expected properties."""

    def test_svf_range(self, svf_golden):
        """Golden SVF values should be in valid range [0, 1]."""
        for name, arr in svf_golden.items():
            valid_mask = ~np.isnan(arr)
            assert np.all(arr[valid_mask] >= 0), f"{name} has values < 0"
            assert np.all(arr[valid_mask] <= 1), f"{name} has values > 1"

    def test_svf_shape_consistency(self, svf_golden):
        """All SVF arrays should have the same shape."""
        shapes = [arr.shape for arr in svf_golden.values()]
        assert all(s == shapes[0] for s in shapes), "SVF arrays have inconsistent shapes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
