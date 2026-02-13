"""
Tests for structured error handling and validate_inputs().

These tests verify that the error system provides actionable messages
and that validate_inputs() catches errors before expensive calculations.
"""

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig.api import (
    GridShapeMismatch,
    InvalidSurfaceData,
    MissingPrecomputedData,
    SolweigError,
    SurfaceData,
    Weather,
    validate_inputs,
)


class TestSolweigErrorHierarchy:
    """Tests for the error class hierarchy."""

    def test_solweig_error_is_base_exception(self):
        """SolweigError can be used to catch all SOLWEIG errors."""
        error = SolweigError("Test error")
        assert isinstance(error, Exception)

    def test_invalid_surface_data_has_fields(self):
        """InvalidSurfaceData has optional field, expected, got attributes."""
        error = InvalidSurfaceData(
            "Grid mismatch",
            field="cdsm",
            expected="(100, 100)",
            got="(50, 50)",
        )
        assert error.field == "cdsm"
        assert error.expected == "(100, 100)"
        assert error.got == "(50, 50)"

    def test_grid_shape_mismatch_is_invalid_surface_data(self):
        """GridShapeMismatch is a subclass of InvalidSurfaceData."""
        error = GridShapeMismatch("cdsm", (100, 100), (50, 50))
        assert isinstance(error, InvalidSurfaceData)
        assert isinstance(error, SolweigError)

    def test_grid_shape_mismatch_has_shapes(self):
        """GridShapeMismatch provides expected_shape and actual_shape."""
        error = GridShapeMismatch("cdsm", (100, 100), (50, 50))
        assert error.field == "cdsm"
        assert error.expected_shape == (100, 100)
        assert error.actual_shape == (50, 50)
        assert "(100, 100)" in str(error)
        assert "(50, 50)" in str(error)

    def test_missing_precomputed_data_has_suggestion(self):
        """MissingPrecomputedData can include a suggestion."""
        error = MissingPrecomputedData(
            "shadow_matrices required for anisotropic sky",
            suggestion="Set use_anisotropic_sky=False",
        )
        assert error.what == "shadow_matrices required for anisotropic sky"
        assert error.suggestion == "Set use_anisotropic_sky=False"
        assert "shadow_matrices" in str(error)
        assert "Set use_anisotropic_sky=False" in str(error)


class TestValidateInputs:
    """Tests for the validate_inputs() preflight function."""

    def test_valid_surface_returns_empty_warnings(self):
        """Valid surface data returns no warnings."""
        dsm = np.ones((100, 100), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((100, 100)))

        warnings = validate_inputs(surface)

        assert warnings == []

    def test_missing_svf_raises_error(self):
        """Surface without SVF raises MissingPrecomputedData."""
        dsm = np.ones((50, 50), dtype=np.float32)
        surface = SurfaceData(dsm=dsm)

        with pytest.raises(MissingPrecomputedData) as excinfo:
            validate_inputs(surface)

        assert "SVF" in str(excinfo.value)
        assert "compute_svf()" in str(excinfo.value)

    def test_mismatched_cdsm_raises_grid_shape_mismatch(self):
        """CDSM with wrong shape raises GridShapeMismatch."""
        dsm = np.ones((100, 100), dtype=np.float32)
        cdsm = np.ones((50, 50), dtype=np.float32)  # Wrong shape

        surface = SurfaceData(dsm=dsm, cdsm=cdsm)

        with pytest.raises(GridShapeMismatch) as excinfo:
            validate_inputs(surface)

        assert excinfo.value.field == "cdsm"
        assert excinfo.value.expected_shape == (100, 100)
        assert excinfo.value.actual_shape == (50, 50)

    def test_mismatched_dem_raises_grid_shape_mismatch(self):
        """DEM with wrong shape raises GridShapeMismatch."""
        dsm = np.ones((100, 100), dtype=np.float32)
        dem = np.ones((100, 50), dtype=np.float32)  # Wrong shape

        surface = SurfaceData(dsm=dsm, dem=dem)

        with pytest.raises(GridShapeMismatch) as excinfo:
            validate_inputs(surface)

        assert excinfo.value.field == "dem"

    def test_anisotropic_without_shadow_matrices_raises_error(self):
        """Anisotropic sky without shadow matrices raises MissingPrecomputedData."""
        dsm = np.ones((50, 50), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((50, 50)))

        with pytest.raises(MissingPrecomputedData) as excinfo:
            validate_inputs(surface, use_anisotropic_sky=True)

        assert "shadow_matrices" in str(excinfo.value)

    def test_unpreprocessed_cdsm_warning(self):
        """Warning issued for unpreprocessed CDSM with cdsm_relative=True."""
        dsm = np.ones((50, 50), dtype=np.float32) * 10.0
        cdsm = np.ones((50, 50), dtype=np.float32) * 5.0  # Relative heights

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert any("preprocess()" in w for w in warnings)

    def test_no_warning_after_preprocess(self):
        """No warning when preprocess() has been called."""
        dsm = np.ones((50, 50), dtype=np.float32) * 10.0
        cdsm = np.ones((50, 50), dtype=np.float32) * 5.0

        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True, svf=make_mock_svf((50, 50)))
        surface.preprocess()

        warnings = validate_inputs(surface)

        # No CDSM preprocessing warning
        assert not any("preprocess()" in w for w in warnings)

    def test_extreme_temperature_warning(self):
        """Warning issued for extreme temperature values."""
        dsm = np.ones((20, 20), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((20, 20)))
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=65.0,  # Extreme temperature
            rh=50.0,
            global_rad=800.0,
        )

        warnings = validate_inputs(surface, weather=weather)

        assert any("ta=" in w and "outside typical range" in w for w in warnings)

    def test_excessive_radiation_warning(self):
        """Warning issued for radiation exceeding solar constant."""
        dsm = np.ones((20, 20), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((20, 20)))
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=1500.0,  # Exceeds solar constant (~1361 W/m²)
        )

        warnings = validate_inputs(surface, weather=weather)

        assert any("global_rad=" in w and "solar constant" in w for w in warnings)

    def test_validates_weather_list(self):
        """validate_inputs() accepts a list of Weather objects."""
        dsm = np.ones((20, 20), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((20, 20)))
        weather_list = [
            Weather(datetime=datetime(2024, 7, 15, 12, 0), ta=25.0, rh=50.0, global_rad=800.0),
            Weather(datetime=datetime(2024, 7, 15, 13, 0), ta=70.0, rh=50.0, global_rad=750.0),  # Extreme
        ]

        warnings = validate_inputs(surface, weather=weather_list)

        # Should warn about the second weather entry
        assert any("[1]" in w and "ta=" in w for w in warnings)

    def test_no_warnings_for_normal_weather(self):
        """No warnings for normal weather values."""
        dsm = np.ones((20, 20), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((20, 20)))
        weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )

        warnings = validate_inputs(surface, weather=weather)

        assert warnings == []


class TestHeightValidationWarnings:
    """Tests for DSM/CDSM/TDSM height sanity warnings."""

    def test_warns_dsm_extreme_height_range(self):
        """DSM with >500m height range triggers warning."""
        dsm = np.ones((50, 50), dtype=np.float32) * 10.0
        dsm[0, 0] = 600.0  # Creates 590m range
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert any("height range" in w and "590" in w for w in warnings)

    def test_warns_dsm_high_minimum_no_dem(self):
        """DSM with min >100m and no DEM triggers warning."""
        dsm = np.ones((50, 50), dtype=np.float32) * 200.0
        dsm[25, 25] = 210.0  # Some buildings
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert any("minimum value is 200m" in w and "no DEM" in w for w in warnings)

    def test_no_warning_dsm_high_minimum_with_dem(self):
        """DSM with min >100m but DEM provided does not warn about elevation."""
        dsm = np.ones((50, 50), dtype=np.float32) * 200.0
        dsm[25, 25] = 210.0
        dem = np.ones((50, 50), dtype=np.float32) * 195.0
        surface = SurfaceData(dsm=dsm, dem=dem, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert not any("no DEM" in w for w in warnings)

    def test_warns_cdsm_looks_absolute_with_relative_flag(self):
        """CDSM with min non-zero >50m and cdsm_relative=True triggers warning."""
        dsm = np.ones((50, 50), dtype=np.float32) * 130.0
        cdsm = np.zeros((50, 50), dtype=np.float32)
        cdsm[10:20, 10:20] = 120.0  # Looks like absolute elevation, not tree height
        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=True, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert any("CDSM minimum non-zero value is 120m" in w for w in warnings)

    def test_warns_cdsm_looks_relative_with_absolute_flag(self):
        """CDSM with values much smaller than DSM and cdsm_relative=False triggers warning."""
        dsm = np.ones((50, 50), dtype=np.float32) * 150.0
        cdsm = np.zeros((50, 50), dtype=np.float32)
        cdsm[10:20, 10:20] = 15.0  # Looks like relative tree heights
        surface = SurfaceData(dsm=dsm, cdsm=cdsm, cdsm_relative=False, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert any("much smaller than DSM" in w for w in warnings)

    def test_no_warning_normal_inputs(self):
        """Typical urban inputs produce no height warnings."""
        dsm = np.ones((50, 50), dtype=np.float32) * 5.0
        dsm[20:30, 20:30] = 15.0  # Buildings 10m range
        surface = SurfaceData(dsm=dsm, svf=make_mock_svf((50, 50)))

        warnings = validate_inputs(surface)

        assert warnings == []

    def test_warns_tdsm_looks_absolute_with_relative_flag(self):
        """TDSM with min non-zero >50m and tdsm_relative=True triggers warning."""
        dsm = np.ones((50, 50), dtype=np.float32) * 130.0
        cdsm = np.zeros((50, 50), dtype=np.float32)
        cdsm[10:20, 10:20] = 120.0
        tdsm = np.zeros((50, 50), dtype=np.float32)
        tdsm[10:20, 10:20] = 115.0  # Looks like absolute trunk elevation
        surface = SurfaceData(
            dsm=dsm, cdsm=cdsm, tdsm=tdsm, cdsm_relative=True, tdsm_relative=True, svf=make_mock_svf((50, 50))
        )

        warnings = validate_inputs(surface)

        assert any("TDSM minimum non-zero value is 115m" in w for w in warnings)


class TestErrorCatching:
    """Tests for catching errors with proper exception types."""

    def test_catch_all_solweig_errors(self):
        """SolweigError catches all SOLWEIG-specific errors."""
        caught = []

        try:
            raise GridShapeMismatch("test", (10, 10), (5, 5))
        except SolweigError:
            caught.append("GridShapeMismatch")

        try:
            raise MissingPrecomputedData("test")
        except SolweigError:
            caught.append("MissingPrecomputedData")

        assert caught == ["GridShapeMismatch", "MissingPrecomputedData"]

    def test_catch_specific_error_types(self):
        """Specific error types can be caught individually."""
        dsm = np.ones((100, 100), dtype=np.float32)
        cdsm = np.ones((50, 50), dtype=np.float32)
        surface = SurfaceData(dsm=dsm, cdsm=cdsm)

        with pytest.raises(GridShapeMismatch):
            validate_inputs(surface)

        # Should NOT raise InvalidSurfaceData (which is the parent)
        # when we specifically want GridShapeMismatch
        try:
            validate_inputs(surface)
        except GridShapeMismatch:
            pass  # Expected
        except InvalidSurfaceData:
            pytest.fail("Should have raised GridShapeMismatch, not generic InvalidSurfaceData")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
