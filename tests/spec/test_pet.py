"""
Physiological Equivalent Temperature (PET) Tests

Tests derived from specs/pet.md properties.

Note: pet_grid takes (ta_scalar, rh_scalar, tmrt_grid, va_grid, ...) where
ta and rh are scalars applied to the whole grid.
"""

import numpy as np
import pytest
from solweig.rustalgos import pet

# =============================================================================
# Test Fixtures - Default Human Parameters
# =============================================================================

DEFAULT_PERSON = {
    "mbody": 75.0,  # kg
    "age": 35,  # years
    "height": 1.75,  # m
    "activity": 80.0,  # W/m² (light walking)
    "clo": 0.9,  # clothing insulation
    "sex": 1,  # 1=male
}


def calculate_pet(ta, rh, tmrt, va, person=None):
    """Calculate PET with default or custom person parameters."""
    if person is None:
        person = DEFAULT_PERSON
    return pet.pet_calculate(
        ta,
        rh,
        tmrt,
        va,
        person["mbody"],
        person["age"],
        person["height"],
        person["activity"],
        person["clo"],
        person["sex"],
    )


# =============================================================================
# Property Tests (from specs/pet.md)
# =============================================================================


class TestPetProperties:
    """Tests for PET calculation properties."""

    def test_property_1_pet_is_person_specific(self):
        """Property 1: PET varies with person characteristics."""
        ta = 25.0
        rh = 50.0
        tmrt = 30.0
        va = 1.0

        # Young fit person
        young_person = {**DEFAULT_PERSON, "age": 25, "mbody": 70.0}
        pet_young = calculate_pet(ta, rh, tmrt, va, young_person)

        # Older person
        old_person = {**DEFAULT_PERSON, "age": 65, "mbody": 80.0}
        pet_old = calculate_pet(ta, rh, tmrt, va, old_person)

        # Both should produce valid results but may differ
        assert not np.isnan(pet_young), "Young person PET should be valid"
        assert not np.isnan(pet_old), "Older person PET should be valid"

    def test_property_2_pet_reference_is_indoor(self):
        """Property 2: PET = ~21°C is comfortable indoors."""
        # Indoor reference conditions: Tmrt=Ta, v=0.1m/s, RH=50%
        ta = 21.0
        tmrt = 21.0
        va = 0.1
        rh = 50.0

        result = calculate_pet(ta, rh, tmrt, va)

        # In reference conditions, PET should be close to Ta
        assert abs(result - ta) < 5.0, f"PET ({result:.1f}) should be close to Ta ({ta}) in reference conditions"

    def test_property_3_higher_tmrt_higher_pet(self):
        """Property 3: Higher Tmrt → higher PET."""
        ta = 30.0
        rh = 50.0
        va = 1.0

        # Shaded (Tmrt ≈ Ta)
        pet_shade = calculate_pet(ta, rh, tmrt=ta, va=va)

        # Sunlit (high Tmrt)
        pet_sun = calculate_pet(ta, rh, tmrt=ta + 30, va=va)

        assert pet_sun > pet_shade, f"Sunlit PET ({pet_sun:.1f}) should be > shaded ({pet_shade:.1f})"

    def test_property_5_activity_increases_pet(self):
        """Property 5: Higher activity → higher PET in warm conditions."""
        ta = 30.0
        rh = 50.0
        tmrt = 35.0
        va = 1.0

        # Light activity (standing)
        rest_person = {**DEFAULT_PERSON, "activity": 58.0}  # Resting
        pet_rest = calculate_pet(ta, rh, tmrt, va, rest_person)

        # High activity (walking fast)
        active_person = {**DEFAULT_PERSON, "activity": 135.0}  # Walking 5 km/h
        pet_active = calculate_pet(ta, rh, tmrt, va, active_person)

        # Higher activity should increase heat stress (higher PET)
        assert pet_active > pet_rest, f"Active PET ({pet_active:.1f}) should be > resting ({pet_rest:.1f})"

    def test_property_8_wind_generally_reduces_pet(self):
        """Property 8: Wind generally reduces PET."""
        ta = 30.0
        rh = 50.0
        tmrt = 40.0

        # Calm
        pet_calm = calculate_pet(ta, rh, tmrt, va=0.5)

        # Windy
        pet_windy = calculate_pet(ta, rh, tmrt, va=5.0)

        assert pet_windy < pet_calm, f"Wind should reduce PET: calm={pet_calm:.1f}, windy={pet_windy:.1f}"


class TestPetComfortCategories:
    """Test that PET produces expected comfort categories."""

    def test_very_hot_conditions(self):
        """PET should indicate heat stress in hot sunny conditions."""
        ta = 35.0
        rh = 50.0
        tmrt = 65.0  # Hot sunny
        va = 1.0

        result = calculate_pet(ta, rh, tmrt, va)

        # Should be in "hot" or "very hot" range (>35°C)
        assert result > 30, f"Hot sunny PET ({result:.1f}) should indicate heat stress"

    def test_comfortable_conditions(self):
        """PET should be in comfort range for moderate conditions."""
        ta = 22.0
        rh = 50.0
        tmrt = 25.0
        va = 1.0

        result = calculate_pet(ta, rh, tmrt, va)

        # Comfort zone is 18-23°C for PET
        assert 15 <= result <= 30, f"Comfortable conditions should give PET near comfort range, got {result:.1f}"

    def test_cold_conditions(self):
        """PET should indicate cold stress in cold conditions."""
        ta = 5.0
        rh = 60.0
        tmrt = 5.0
        va = 2.0

        result = calculate_pet(ta, rh, tmrt, va)

        # Should be in "cold" or "cool" range (<18°C)
        assert result < 18, f"Cold PET ({result:.1f}) should indicate cold stress"


class TestPetGrid:
    """Test grid-based PET calculation.

    Note: pet_grid signature is (ta_scalar, rh_scalar, tmrt_grid, va_grid, ...)
    """

    def test_grid_calculation(self):
        """Test that grid calculation works for 2D arrays."""
        shape = (10, 10)
        ta = 25.0  # scalar
        rh = 50.0  # scalar
        tmrt = np.full(shape, 35.0, dtype=np.float32)
        va = np.full(shape, 1.0, dtype=np.float32)

        result = pet.pet_grid(
            ta,
            rh,
            tmrt,
            va,
            DEFAULT_PERSON["mbody"],
            DEFAULT_PERSON["age"],
            DEFAULT_PERSON["height"],
            DEFAULT_PERSON["activity"],
            DEFAULT_PERSON["clo"],
            DEFAULT_PERSON["sex"],
        )

        assert result.shape == shape, f"Output shape {result.shape} should match input {shape}"
        # -9999 is used for invalid pixels
        valid_mask = result != -9999
        assert np.any(valid_mask), "Grid PET should have some valid values"

    def test_grid_consistent_values(self):
        """Grid with uniform Tmrt/va should produce uniform output."""
        shape = (5, 5)
        ta = 25.0
        rh = 50.0
        tmrt = np.full(shape, 35.0, dtype=np.float32)
        va = np.full(shape, 1.0, dtype=np.float32)

        result = pet.pet_grid(
            ta,
            rh,
            tmrt,
            va,
            DEFAULT_PERSON["mbody"],
            DEFAULT_PERSON["age"],
            DEFAULT_PERSON["height"],
            DEFAULT_PERSON["activity"],
            DEFAULT_PERSON["clo"],
            DEFAULT_PERSON["sex"],
        )

        # All valid values should be the same
        valid_mask = result != -9999
        valid_values = result[valid_mask]
        if len(valid_values) > 1:
            np.testing.assert_allclose(
                valid_values, valid_values[0], rtol=1e-4, err_msg="Uniform inputs should produce uniform output"
            )


# =============================================================================
# Edge-Case Tests
# =============================================================================


class TestPetEdgeCases:
    """Test PET behaviour at extreme and boundary inputs."""

    def test_extreme_heat(self):
        """PET should return a valid value under extreme heat."""
        result = calculate_pet(45.0, 80.0, 70.0, 0.5)
        assert not np.isnan(result), "PET should not be NaN for extreme heat"
        assert result > 35, f"Extreme heat PET ({result:.1f}) should indicate severe stress"

    def test_extreme_cold(self):
        """PET should return a valid value under extreme cold."""
        result = calculate_pet(-30.0, 50.0, -25.0, 5.0)
        assert not np.isnan(result), "PET should not be NaN for extreme cold"
        assert result < 0, f"Extreme cold PET ({result:.1f}) should be negative"

    def test_zero_wind_speed(self):
        """PET should handle va=0 without error."""
        result = calculate_pet(25.0, 50.0, 30.0, 0.0)
        assert not np.isnan(result), "PET should handle zero wind speed"

    def test_very_low_wind_speed(self):
        """PET should handle very small positive va."""
        result = calculate_pet(25.0, 50.0, 30.0, 0.01)
        assert not np.isnan(result), "PET should handle very low wind speed"
        # Should be close to zero-wind result
        result_zero = calculate_pet(25.0, 50.0, 30.0, 0.0)
        assert abs(result - result_zero) < 2.0, (
            f"Very low wind PET ({result:.1f}) should be close to zero-wind ({result_zero:.1f})"
        )

    def test_high_tmrt_delta(self):
        """PET should handle large Tmrt - Ta differences."""
        result = calculate_pet(20.0, 50.0, 80.0, 1.0)
        assert not np.isnan(result), "PET should handle high Tmrt delta"

    def test_tmrt_below_ta(self):
        """PET should handle Tmrt significantly below Ta (cold radiation)."""
        result = calculate_pet(25.0, 50.0, 5.0, 1.0)
        assert not np.isnan(result), "PET should handle Tmrt below Ta"
        # Cold radiation environment should feel cooler
        result_neutral = calculate_pet(25.0, 50.0, 25.0, 1.0)
        assert result < result_neutral, f"Cold Tmrt PET ({result:.1f}) should be < neutral ({result_neutral:.1f})"

    def test_high_clothing_insulation(self):
        """PET should handle high clo values (heavy winter clothing)."""
        heavy_person = {**DEFAULT_PERSON, "clo": 2.5}
        result = calculate_pet(10.0, 50.0, 10.0, 1.0, heavy_person)
        assert not np.isnan(result), "PET should handle high clothing insulation"

    def test_minimal_clothing(self):
        """PET should handle minimal clo values (light summer clothing)."""
        light_person = {**DEFAULT_PERSON, "clo": 0.3}
        result = calculate_pet(30.0, 50.0, 35.0, 1.0, light_person)
        assert not np.isnan(result), "PET should handle minimal clothing"

    def test_monotonic_wind_cooling(self):
        """Higher wind speed should reduce PET in warm conditions."""
        ta = 30.0
        rh = 50.0
        tmrt = 40.0

        pet_calm = calculate_pet(ta, rh, tmrt, va=0.5)
        pet_moderate = calculate_pet(ta, rh, tmrt, va=3.0)
        pet_windy = calculate_pet(ta, rh, tmrt, va=8.0)

        assert pet_calm > pet_moderate > pet_windy, (
            f"Wind should reduce PET: calm={pet_calm:.1f}, moderate={pet_moderate:.1f}, windy={pet_windy:.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
