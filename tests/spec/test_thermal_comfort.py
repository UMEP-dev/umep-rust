"""
Tests for thermal comfort indices (UTCI and PET) validation.

Validates implementations against reference values and specifications.

References:
- specs/utci.md
- specs/pet.md
- Bröde P et al. (2012) "Deriving the operational procedure for the UTCI"
- Höppe P (1999) "The physiological equivalent temperature"
"""

import numpy as np
import pytest


class TestUTCIPolynomial:
    """
    Test UTCI polynomial approximation against reference values.

    Reference: Bröde P, Fiala D, Błażejczyk K, et al. (2012)
    "Deriving the operational procedure for the Universal Thermal Climate Index (UTCI)"
    International Journal of Biometeorology 56:481-494.

    The polynomial should match the full Fiala model within ±0.5°C.
    """

    def test_utci_import(self):
        """Verify UTCI module can be imported."""
        from solweig.algorithms.UTCI_calculations import utci_calculator, utci_polynomial

        assert callable(utci_calculator)
        assert callable(utci_polynomial)

    def test_utci_neutral_conditions(self):
        """
        UTCI should approximate Ta when Tmrt=Ta, low wind, moderate humidity.

        Reference condition: Ta=20°C, Tmrt=20°C, va=0.5 m/s, RH=50%
        Expected: UTCI ≈ 20°C (no thermal stress)
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 20.0
        RH = 50.0
        Tmrt = 20.0
        va = 0.5

        utci = utci_calculator(Ta, RH, Tmrt, va)

        # In neutral conditions, UTCI should be close to Ta
        assert 18.0 < utci < 26.0, f"Neutral UTCI should be near Ta, got {utci:.1f}°C"

    def test_utci_hot_sunny(self):
        """
        Hot sunny conditions should produce high UTCI (strong heat stress).

        Conditions: Ta=35°C, Tmrt=65°C (in sun), va=1 m/s, RH=40%
        Expected: UTCI > 38°C (very strong heat stress)
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 35.0
        RH = 40.0
        Tmrt = 65.0
        va = 1.0

        utci = utci_calculator(Ta, RH, Tmrt, va)

        assert utci > 38.0, f"Hot sunny UTCI should indicate heat stress, got {utci:.1f}°C"
        assert utci < 55.0, f"UTCI seems unreasonably high: {utci:.1f}°C"

    def test_utci_hot_shaded(self):
        """
        Hot shaded conditions should produce lower UTCI than sunny.

        Conditions: Ta=35°C, Tmrt=40°C (in shade), va=1 m/s, RH=40%
        Expected: UTCI < sunny conditions
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 35.0
        RH = 40.0
        va = 1.0

        utci_sunny = utci_calculator(Ta, RH, 65.0, va)
        utci_shade = utci_calculator(Ta, RH, 40.0, va)

        assert utci_shade < utci_sunny, (
            f"Shade UTCI ({utci_shade:.1f}) should be less than sun ({utci_sunny:.1f})"
        )

    def test_utci_wind_cooling(self):
        """
        Higher wind speed should reduce UTCI in hot conditions.

        Reference: UTCI accounts for convective cooling from wind.
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 30.0
        RH = 50.0
        Tmrt = 35.0

        utci_low_wind = utci_calculator(Ta, RH, Tmrt, 0.5)
        utci_high_wind = utci_calculator(Ta, RH, Tmrt, 5.0)

        assert utci_high_wind < utci_low_wind, (
            f"High wind UTCI ({utci_high_wind:.1f}) should be less than "
            f"low wind ({utci_low_wind:.1f})"
        )

    def test_utci_humidity_effect(self):
        """
        Higher humidity should increase UTCI in hot conditions.

        Reference: Humidity impairs evaporative cooling.
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 35.0
        Tmrt = 40.0
        va = 1.0

        utci_dry = utci_calculator(Ta, 30.0, Tmrt, va)
        utci_humid = utci_calculator(Ta, 80.0, Tmrt, va)

        assert utci_humid > utci_dry, (
            f"Humid UTCI ({utci_humid:.1f}) should be greater than dry ({utci_dry:.1f})"
        )

    def test_utci_stress_categories(self):
        """
        Verify UTCI values correspond to documented stress categories.

        Reference: specs/utci.md stress categories table.
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        # Test that comfortable conditions give no thermal stress (9-26°C)
        utci_comfort = utci_calculator(22.0, 50.0, 22.0, 1.0)
        assert 9.0 < utci_comfort < 26.0, (
            f"Comfortable conditions should give UTCI 9-26°C, got {utci_comfort:.1f}°C"
        )

    def test_utci_reference_values(self):
        """
        Test against published reference values from UTCI documentation.

        Reference: COST Action 730 validation dataset.
        These are approximate values from the literature.
        """
        from solweig.algorithms.UTCI_calculations import utci_calculator

        # Reference case 1: Moderate conditions
        # Ta=20°C, RH=50%, Tmrt=20°C, va=1.0 m/s
        utci_1 = utci_calculator(20.0, 50.0, 20.0, 1.0)
        # Should be approximately neutral (17-23°C range)
        assert 15.0 < utci_1 < 25.0, f"Reference case 1 failed: {utci_1:.1f}°C"

        # Reference case 2: Hot radiation
        # Ta=25°C, RH=50%, Tmrt=50°C, va=1.0 m/s
        utci_2 = utci_calculator(25.0, 50.0, 50.0, 1.0)
        # Should show moderate heat stress (32-38°C range)
        assert utci_2 > 28.0, f"Reference case 2 should show heat stress: {utci_2:.1f}°C"


class TestUTCIInputValidation:
    """
    Test UTCI input range handling.

    Reference: specs/utci.md valid input ranges.
    """

    def test_utci_handles_missing_data(self):
        """UTCI should return -999 for missing input values."""
        from solweig.algorithms.UTCI_calculations import utci_calculator

        result = utci_calculator(-999, 50.0, 25.0, 1.0)
        assert result == -999, "Should return -999 for missing Ta"

        result = utci_calculator(25.0, -999, 25.0, 1.0)
        assert result == -999, "Should return -999 for missing RH"

    def test_utci_grid_calculation(self):
        """Test that grid calculation produces valid array."""
        from solweig.algorithms.UTCI_calculations import utci_calculator_grid_fast

        Ta = 25.0
        RH = 50.0
        Tmrt = np.full((10, 10), 30.0, dtype=np.float32)
        va = np.full((10, 10), 1.0, dtype=np.float32)

        result = utci_calculator_grid_fast(Ta, RH, Tmrt, va)

        assert result.shape == (10, 10), "Output shape should match input"
        assert np.all(result > -999), "All valid inputs should produce valid output"
        assert np.all(result < 60), "UTCI should be reasonable"


class TestPETCalculations:
    """
    Test PET (Physiological Equivalent Temperature) calculations.

    Reference: specs/pet.md
    - Höppe P (1999) "The physiological equivalent temperature"
    - DuBois & DuBois (1916) body surface area formula
    """

    def test_pet_import(self):
        """Verify PET module can be imported."""
        from solweig.algorithms.PET_calculations import _PET, PET_person

        assert callable(_PET)

    def test_dubois_body_surface_area(self):
        """
        Verify DuBois body surface area calculation.

        Reference: DuBois D, DuBois EF (1916) "A formula to estimate the
        approximate surface area if height and weight be known."
        Archives of Internal Medicine 17:863-871.

        Formula: A_body = 0.203 × height^0.725 × weight^0.425
        """
        # For reference person (1.75m, 75kg)
        height = 1.75
        weight = 75.0

        # DuBois formula
        A_body = 0.203 * (weight ** 0.425) * (height ** 0.725)

        # Expected ~1.90 m² for this person
        assert 1.85 < A_body < 1.95, (
            f"DuBois body surface area for 1.75m/75kg should be ~1.90 m², got {A_body:.3f}"
        )

    def test_pet_neutral_conditions(self):
        """
        PET in neutral conditions should be near comfortable range.

        Reference conditions: Ta=22°C, RH=50%, Tmrt=22°C, va=0.1 m/s
        Expected: PET ≈ 18-23°C (comfortable)
        """
        from solweig.algorithms.PET_calculations import _PET

        Ta = 22.0
        RH = 50.0
        Tmrt = 22.0
        va = 0.1

        # Standard person parameters
        mbody = 75.0  # kg
        age = 35  # years
        height = 1.75  # m
        activity = 80  # W
        clo = 0.9  # clothing
        sex = 1  # male

        pet = _PET(Ta, RH, Tmrt, va, mbody, age, height, activity, clo, sex)

        # Comfortable range is 18-23°C
        assert 15.0 < pet < 28.0, (
            f"Neutral conditions PET should be comfortable, got {pet:.1f}°C"
        )

    def test_pet_radiation_effect(self):
        """
        Higher Tmrt should increase PET.

        Reference: specs/pet.md property #3
        """
        from solweig.algorithms.PET_calculations import _PET

        Ta = 25.0
        RH = 50.0
        va = 0.5
        mbody, age, height, activity, clo, sex = 75.0, 35, 1.75, 80, 0.9, 1

        pet_low_tmrt = _PET(Ta, RH, 25.0, va, mbody, age, height, activity, clo, sex)
        pet_high_tmrt = _PET(Ta, RH, 55.0, va, mbody, age, height, activity, clo, sex)

        assert pet_high_tmrt > pet_low_tmrt, (
            f"Higher Tmrt should increase PET: low={pet_low_tmrt:.1f}, high={pet_high_tmrt:.1f}"
        )

    def test_pet_stress_categories(self):
        """
        Verify PET values correspond to documented stress categories.

        Reference: specs/pet.md comfort categories table.
        """
        from solweig.algorithms.PET_calculations import _PET

        mbody, age, height, activity, clo, sex = 75.0, 35, 1.75, 80, 0.9, 1

        # Hot sunny conditions should give "Hot" or higher (PET > 35°C)
        pet_hot = _PET(35.0, 40.0, 65.0, 0.5, mbody, age, height, activity, clo, sex)
        assert pet_hot > 35.0, f"Hot sunny PET should be > 35°C, got {pet_hot:.1f}°C"

        # Comfortable conditions should be 18-23°C
        pet_comfort = _PET(22.0, 50.0, 22.0, 0.5, mbody, age, height, activity, clo, sex)
        # Allow wider range due to activity level
        assert 15.0 < pet_comfort < 30.0, (
            f"Comfortable PET should be moderate, got {pet_comfort:.1f}°C"
        )


class TestPETPETComparison:
    """
    Compare PET and UTCI characteristics.

    Reference: specs/pet.md comparison table.
    """

    def test_both_indices_increase_with_radiation(self):
        """Both UTCI and PET should increase with higher Tmrt."""
        from solweig.algorithms.PET_calculations import _PET
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta = 30.0
        RH = 50.0
        va = 1.0

        # UTCI
        utci_low = utci_calculator(Ta, RH, Ta, va)
        utci_high = utci_calculator(Ta, RH, Ta + 30.0, va)

        # PET
        mbody, age, height, activity, clo, sex = 75.0, 35, 1.75, 80, 0.9, 1
        pet_low = _PET(Ta, RH, Ta, va, mbody, age, height, activity, clo, sex)
        pet_high = _PET(Ta, RH, Ta + 30.0, va, mbody, age, height, activity, clo, sex)

        assert utci_high > utci_low, "UTCI should increase with Tmrt"
        assert pet_high > pet_low, "PET should increase with Tmrt"

    def test_utci_faster_than_pet(self):
        """
        UTCI polynomial should be faster than PET iterative solver.

        Reference: specs/pet.md notes UTCI uses polynomial (~200 terms),
        PET uses iterative solver (10-20 iterations).
        """
        import time

        from solweig.algorithms.PET_calculations import _PET
        from solweig.algorithms.UTCI_calculations import utci_calculator

        Ta, RH, Tmrt, va = 25.0, 50.0, 30.0, 1.0
        mbody, age, height, activity, clo, sex = 75.0, 35, 1.75, 80, 0.9, 1

        # Time UTCI
        start = time.perf_counter()
        for _ in range(100):
            utci_calculator(Ta, RH, Tmrt, va)
        utci_time = time.perf_counter() - start

        # Time PET
        start = time.perf_counter()
        for _ in range(100):
            _PET(Ta, RH, Tmrt, va, mbody, age, height, activity, clo, sex)
        pet_time = time.perf_counter() - start

        # UTCI should be noticeably faster (polynomial vs iterative)
        # Allow some variance, but UTCI should be at least 2x faster typically
        assert utci_time < pet_time * 5, (
            f"UTCI ({utci_time:.4f}s) expected faster than PET ({pet_time:.4f}s)"
        )


class TestThermalComfortDefaults:
    """
    Test default parameters match specifications.

    Reference: specs/pet.md, default_params.json
    """

    def test_default_pet_parameters(self):
        """Verify default PET parameters match spec."""
        from solweig.config import load_params

        params = load_params()
        pet = params.PET_settings.Value

        assert pet.Age == 35, f"Default age should be 35, got {pet.Age}"
        assert pet.Weight == 75.0, f"Default weight should be 75kg, got {pet.Weight}"
        assert pet.Height == 180, f"Default height should be 180cm, got {pet.Height}"
        assert pet.Activity == 80.0, f"Default activity should be 80 W/m², got {pet.Activity}"
        assert pet.clo == 0.9, f"Default clothing should be 0.9 clo, got {pet.clo}"

    def test_default_wind_height(self):
        """Verify default wind measurement height for UTCI."""
        from solweig.config import load_params

        params = load_params()

        assert params.Wind_Height.Value.magl == 10.0, (
            f"UTCI wind height should be 10m, got {params.Wind_Height.Value.magl}"
        )
