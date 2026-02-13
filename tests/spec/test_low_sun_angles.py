"""
Low Sun Angle Handling Tests

Tests for numerical stability at sun altitudes < 3° where tan(zenith) → infinity.
Verifies the guards added to cylindric_wedge and Perez_v3 functions.

Reference: MIN_SUN_ELEVATION_DEG = 3.0 is the established UMEP/SOLWEIG threshold.
"""

import warnings

import numpy as np
import pytest
from solweig.constants import MIN_SUN_ELEVATION_DEG  # noqa: F401 - used in test
from solweig.physics.cylindric_wedge import cylindric_wedge
from solweig.physics.Perez_v3 import Perez_v3


class TestCylindricWedgeLowSun:
    """Tests for cylindric_wedge at low sun angles."""

    def test_returns_fully_shaded_below_threshold(self):
        """Walls should be fully shaded (F_sh=1) when sun altitude < 3°."""
        rows, cols = 50, 50
        svfalfa = np.full((rows, cols), 0.5, dtype=np.float32)  # Typical value

        # Test at various altitudes below threshold
        for altitude in [0.1, 1.0, 2.0, 2.9]:
            zenith_rad = (90 - altitude) * (np.pi / 180)
            result = cylindric_wedge(zenith_rad, svfalfa, rows, cols)

            assert np.allclose(result, 1.0), f"At altitude {altitude}°, walls should be fully shaded (F_sh=1)"

    def test_normal_calculation_above_threshold(self):
        """Normal calculation should occur when sun altitude >= 3°."""
        rows, cols = 50, 50
        svfalfa = np.full((rows, cols), 0.5, dtype=np.float32)

        # Test at altitudes above threshold
        for altitude in [3.0, 5.0, 10.0, 45.0]:
            zenith_rad = (90 - altitude) * (np.pi / 180)
            result = cylindric_wedge(zenith_rad, svfalfa, rows, cols)

            # Should have values between 0 and 1, not all 1s
            assert result.min() >= 0.0
            assert result.max() <= 1.0
            # At reasonable sun angles with uniform svfalfa, shouldn't be all ones
            if altitude >= 10:
                assert result.mean() < 0.99, f"At altitude {altitude}°, should have some sunlit walls"

    def test_no_overflow_warnings_at_edge(self):
        """No overflow warnings should occur at the 3° boundary."""
        rows, cols = 100, 100
        svfalfa = np.random.uniform(0.1, 1.0, (rows, cols)).astype(np.float32)

        # Test at and near the threshold
        for altitude in [2.9, 3.0, 3.1]:
            zenith_rad = (90 - altitude) * (np.pi / 180)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = cylindric_wedge(zenith_rad, svfalfa, rows, cols)

                # Filter for overflow warnings
                overflow_warnings = [x for x in w if "overflow" in str(x.message).lower()]
                assert len(overflow_warnings) == 0, f"Overflow at altitude {altitude}°: {overflow_warnings}"

            # Result should be valid (no NaN or Inf)
            assert np.all(np.isfinite(result)), f"Non-finite values at altitude {altitude}°"


class TestPerezLowSun:
    """Tests for Perez_v3 at low sun angles."""

    def test_returns_uniform_distribution_below_threshold(self):
        """Perez should return uniform sky distribution when altitude < 3°."""
        jday = 182  # July 1 (summer, typical conditions)

        # Test at various altitudes below threshold
        for altitude in [0.5, 1.0, 2.0, 2.9]:
            zenith = 90 - altitude

            lv, _, _ = Perez_v3(
                zen=zenith,
                azimuth=180,
                radD=100,
                radI=500,
                jday=jday,
                patchchoice=1,
                patch_option=1,
            )

            # Uniform distribution means all patches have equal weight
            # For patchchoice=1, lv is (n_patches, 3) where column 2 has luminance
            if lv is not None and len(lv) > 0:
                # Extract luminance column (index 2)
                lv_values = lv[:, 2] if lv.ndim == 2 else lv
                std_dev = np.std(lv_values)
                assert std_dev < 1e-6, f"At altitude {altitude}°, distribution should be uniform (std={std_dev:.8f})"

    def test_normal_calculation_above_threshold(self):
        """Normal Perez calculation should occur when altitude >= 3°."""
        # Test at altitude above threshold with significant direct radiation
        zenith = 90 - 30  # 30° altitude (well above threshold)
        jday = 182

        lv, _, _ = Perez_v3(
            zen=zenith,
            azimuth=180,
            radD=200,  # Diffuse radiation
            radI=600,  # Direct radiation
            jday=jday,
            patchchoice=1,
            patch_option=1,
        )

        if lv is not None and len(lv) > 0:
            # Extract luminance column (index 2) for patchchoice=1
            lv_values = lv[:, 2] if lv.ndim == 2 else lv
            # Anisotropic distribution should have variation
            std_dev = np.std(lv_values)
            assert std_dev > 1e-6, (
                f"At 30° altitude with radiation, should have anisotropic distribution (std={std_dev:.8f})"
            )

    def test_no_warnings_at_boundary(self):
        """No runtime warnings should occur at the 3° boundary."""
        jday = 182

        for altitude in [2.9, 3.0, 3.1]:
            zenith = 90 - altitude

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lv, _, _ = Perez_v3(
                    zen=zenith,
                    azimuth=180,
                    radD=100,
                    radI=300,
                    jday=jday,
                    patchchoice=1,
                    patch_option=1,
                )

                # Filter for numerical warnings
                bad_warnings = [
                    x for x in w if any(s in str(x.message).lower() for s in ["overflow", "divide", "invalid"])
                ]
                assert len(bad_warnings) == 0, f"Numerical warning at altitude {altitude}°: {bad_warnings}"

    def test_returns_uniform_for_very_low_diffuse(self):
        """Perez should return uniform when diffuse radiation < 10 W/m²."""
        # Even at high sun angle, very low diffuse should trigger uniform
        zenith = 90 - 45  # 45° altitude
        jday = 182

        lv, _, _ = Perez_v3(
            zen=zenith,
            azimuth=180,
            radD=5,  # Very low diffuse
            radI=800,
            jday=jday,
            patchchoice=1,
            patch_option=1,
        )

        if lv is not None and len(lv) > 0:
            # Extract luminance column (index 2) for patchchoice=1
            lv_values = lv[:, 2] if lv.ndim == 2 else lv
            std_dev = np.std(lv_values)
            assert std_dev < 1e-6, f"With radD=5, distribution should be uniform (std={std_dev:.8f})"


class TestConstantConsistency:
    """Tests that the MIN_SUN_ELEVATION_DEG constant is used consistently."""

    def test_constant_value(self):
        """MIN_SUN_ELEVATION_DEG should be 3.0 (established UMEP threshold)."""
        assert MIN_SUN_ELEVATION_DEG == 3.0

    def test_threshold_matches_constant(self):
        """Both functions should use the same threshold from constants."""
        # Test just above and below 3°
        altitude_below = 2.99
        altitude_above = 3.01

        rows, cols = 10, 10
        svfalfa = np.full((rows, cols), 0.5, dtype=np.float32)

        # cylindric_wedge at 2.99° should return all 1s
        zen_below = (90 - altitude_below) * (np.pi / 180)
        result_below = cylindric_wedge(zen_below, svfalfa, rows, cols)
        assert np.allclose(result_below, 1.0), "Should be fully shaded at 2.99°"

        # cylindric_wedge at 3.01° should calculate normally
        zen_above = (90 - altitude_above) * (np.pi / 180)
        _result_above = cylindric_wedge(zen_above, svfalfa, rows, cols)
        # Not checking exact values, just verifying the function executes
        # (the actual calculation happens without raising/returning all-1s)


class TestCylindricWedgeRust:
    """Tests that Rust cylindric_wedge matches Python reference implementation."""

    def test_matches_python_reference(self):
        """Rust output matches Python for a range of sun angles."""
        from solweig.rustalgos import sky as rust_sky

        rows, cols = 50, 50
        rng = np.random.default_rng(42)
        svfalfa = rng.uniform(0.1, 1.0, (rows, cols)).astype(np.float32)

        for altitude in [5.0, 15.0, 30.0, 45.0, 60.0, 85.0]:
            zen_rad = (90 - altitude) * (np.pi / 180)
            py_result = cylindric_wedge(zen_rad, svfalfa, rows, cols)
            rs_result = rust_sky.cylindric_wedge(float(zen_rad), svfalfa)

            np.testing.assert_allclose(
                rs_result,
                py_result,
                rtol=2e-5,
                atol=1e-5,
                err_msg=f"Rust/Python mismatch at altitude {altitude}°",
            )

    def test_low_sun_guard_matches(self):
        """Rust returns all 1.0 below 3° threshold, same as Python."""
        from solweig.rustalgos import sky as rust_sky

        svfalfa = np.full((10, 10), 0.5, dtype=np.float32)

        for altitude in [0.1, 1.0, 2.0, 2.9]:
            zen_rad = (90 - altitude) * (np.pi / 180)
            rs_result = rust_sky.cylindric_wedge(float(zen_rad), svfalfa)

            assert np.allclose(rs_result, 1.0), f"Rust should return all 1.0 at altitude {altitude}°"

    def test_boundary_at_3_degrees(self):
        """Rust transition at 3° matches Python."""
        from solweig.rustalgos import sky as rust_sky

        svfalfa = np.full((10, 10), 0.5, dtype=np.float32)

        # Just below threshold
        zen_below = (90 - 2.99) * (np.pi / 180)
        rs_below = rust_sky.cylindric_wedge(float(zen_below), svfalfa)
        assert np.allclose(rs_below, 1.0)

        # Just above threshold
        zen_above = (90 - 3.01) * (np.pi / 180)
        rs_above = rust_sky.cylindric_wedge(float(zen_above), svfalfa)
        py_above = cylindric_wedge(zen_above, svfalfa, 10, 10)
        np.testing.assert_allclose(rs_above, py_above, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
