"""
Universal Thermal Climate Index (UTCI) Tests

Tests derived from specs/utci.md properties.

Note: utci_grid takes (ta_scalar, rh_scalar, tmrt_grid, va_grid) where
ta and rh are scalars applied to the whole grid.
"""

import numpy as np
import pytest
from solweig.rustalgos import utci

# =============================================================================
# Property Tests (from specs/utci.md)
# =============================================================================


class TestUtciProperties:
    """Tests for UTCI calculation properties."""

    def test_property_1_utci_is_temperature(self):
        """Property 1: UTCI is an equivalent temperature in °C."""
        # utci_single(ta, rh, tmrt, va10m)
        result = utci.utci_single(25.0, 50.0, 30.0, 1.0)

        # UTCI should be a reasonable temperature value
        assert isinstance(result, (int, float)), "UTCI should be numeric"
        assert -60 < result < 70, f"UTCI {result}°C outside reasonable range"

    def test_property_2_valid_input_ranges(self):
        """Property 2: UTCI handles valid input ranges."""
        # Test various valid combinations (ta, rh, tmrt, va)
        test_cases = [
            (-10.0, 50.0, -5.0, 1.0),  # Cold
            (25.0, 50.0, 30.0, 2.0),  # Moderate
            (40.0, 30.0, 60.0, 1.0),  # Hot
        ]

        for ta, rh, tmrt, va in test_cases:
            result = utci.utci_single(ta, rh, tmrt, va)
            assert result != -999, f"UTCI returned invalid for ta={ta}, rh={rh}, tmrt={tmrt}, va={va}"

    def test_property_3_higher_tmrt_higher_utci(self):
        """Property 3: Higher Tmrt → higher UTCI."""
        ta = 30.0
        rh = 50.0
        va = 1.0

        # Low Tmrt (shaded)
        utci_shade = utci.utci_single(ta, rh, ta, va)

        # High Tmrt (sunlit) - larger delta to see clear effect
        utci_sun = utci.utci_single(ta, rh, ta + 35, va)

        assert utci_sun > utci_shade, f"Sunlit UTCI ({utci_sun:.1f}) should be > shaded ({utci_shade:.1f})"

    def test_property_8_high_humidity_increases_utci_in_heat(self):
        """Property 8: High humidity increases UTCI in hot conditions."""
        ta = 35.0
        tmrt = 45.0
        va = 1.0

        # Low humidity
        utci_dry = utci.utci_single(ta, 30.0, tmrt, va)

        # High humidity
        utci_humid = utci.utci_single(ta, 80.0, tmrt, va)

        assert utci_humid > utci_dry, (
            f"Humidity should increase UTCI in heat: dry={utci_dry:.1f}, humid={utci_humid:.1f}"
        )


class TestUtciStressCategories:
    """Test that UTCI produces expected stress categories."""

    def test_heat_stress_categories(self):
        """UTCI should produce expected heat stress values."""
        # Hot sunny conditions (high Tmrt)
        utci_val = utci.utci_single(35.0, 50.0, 65.0, 1.0)

        # Should indicate significant heat stress (> moderate threshold of 32)
        assert utci_val > 32, f"Hot sunny UTCI ({utci_val:.1f}) should indicate heat stress"


class TestUtciGrid:
    """Test grid-based UTCI calculation.

    Note: utci_grid signature is (ta_scalar, rh_scalar, tmrt_grid, va_grid)
    Arrays must be float32.
    """

    def test_grid_calculation(self):
        """Test that grid calculation works for 2D arrays."""
        shape = (10, 10)
        ta = 25.0  # scalar
        rh = 50.0  # scalar
        tmrt = np.full(shape, 35.0, dtype=np.float32)
        va = np.full(shape, 1.0, dtype=np.float32)

        result = utci.utci_grid(ta, rh, tmrt, va)

        assert result.shape == shape, f"Output shape {result.shape} should match input {shape}"
        # -9999 is used for invalid pixels
        valid_mask = result != -9999
        assert np.any(valid_mask), "Grid UTCI should have some valid values"

    def test_grid_consistent_values(self):
        """Grid with uniform Tmrt/va should produce uniform output."""
        shape = (5, 5)
        ta = 25.0
        rh = 50.0
        tmrt = np.full(shape, 35.0, dtype=np.float32)
        va = np.full(shape, 1.0, dtype=np.float32)

        result = utci.utci_grid(ta, rh, tmrt, va)

        # All valid values should be the same
        valid_mask = result != -9999
        valid_values = result[valid_mask]
        if len(valid_values) > 1:
            np.testing.assert_allclose(
                valid_values, valid_values[0], rtol=1e-4, err_msg="Uniform inputs should produce uniform output"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
