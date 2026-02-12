"""
Tests for I/O functionality including EPW parser.

Note: EPW parser is deliberately pandas-free for QGIS compatibility.
Tests must not assume pd.DataFrame - they test the _EpwDataFrame interface.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig import io


class TestEPWParser:
    """Test the standalone EPW parser (no pandas dependency)."""

    @pytest.fixture
    def sample_epw_content(self):
        """Create a minimal valid EPW file content."""
        # EPW header (8 lines) + data
        # Timezone offset must be between -24 and +24 hours (field 8)
        # EPW data lines must preserve exact format - long lines are intentional
        return """LOCATION,Athens,GRC,NA,Shiny Weather Data,NA,37.90,23.73,2.0,107.0
DESIGN CONDITIONS,1,Climate Design Data 2009 ASHRAE Handbook,,Heating,1,-2.1,-0.3,0.6,2.8,10.7,2.3,3.5,3.4,12.2,11.2,3.1,11.4,2.5,340,Cooling,8,35.2,23.7,33.2,23.3,31.4,23.0,29.7,24.1,27.2,32.8,26.1,31.1,25.2,29.6,4.2,330,23.5,18.5,27.8,22.7,17.8,27.1,22.0,17.2,26.4,68.2,32.9,64.8,31.2,62.0,29.7,951,Extremes,11.6,10.2,9.0,25.3,-3.9,37.5,2.7,1.7,-5.5,38.9,-7.0,39.9,-8.4,40.8,-10.1,42.2
TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,7/ 9,7/15,Summer - Week Nearest Average Temperature For Period,Typical,7/30,8/ 5,Winter - Week Nearest Min Temperature For Period,Extreme,1/28,2/ 3,Winter - Week Nearest Average Temperature For Period,Typical,1/21,1/27,Autumn - Week Nearest Average Temperature For Period,Typical,11/11,11/17,Spring - Week Nearest Average Temperature For Period,Typical,4/22,4/28
GROUND TEMPERATURES,3,.5,,,12.98,11.39,10.73,11.54,14.82,18.56,21.85,23.85,24.08,22.71,19.89,16.54,2,,,,,,,,,,,,,,,,4,,,,,,,,,,,,,,
HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0
COMMENTS 1,Custom/IWEC Data
COMMENTS 2, -- Ground temps produced with a standard soil diffusivity of 2.3225760E-03 {m**2/day}
DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31
2024,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,9.0,3.9,65,101300,0,0,0,0,0,0,0,0,0,0,190,4.6,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,2,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,8.3,3.9,69,101300,0,0,0,0,0,0,0,0,0,0,190,4.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,3,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,7.8,3.9,72,101300,0,0,0,0,0,0,0,0,0,0,200,3.6,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,4,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,7.2,3.9,76,101300,0,0,0,0,0,0,0,0,0,0,200,3.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
2024,1,1,5,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9*_*9*9*9*9*9,6.7,3.3,76,101300,0,0,0,0,0,0,0,0,0,0,200,3.1,10,10,16.1,77777,9,999999999,0,0.0480,0,88,0.000,0.0,0.0
"""

    @pytest.fixture
    def epw_file(self, sample_epw_content, tmp_path):
        """Create a temporary EPW file."""
        epw_path = tmp_path / "test.epw"
        epw_path.write_text(sample_epw_content)
        return epw_path

    def test_read_epw_returns_data_and_metadata(self, epw_file):
        """Test that read_epw returns a data object and metadata dict."""
        df, metadata = io.read_epw(epw_file)

        assert len(df) == 5
        assert isinstance(metadata, dict)

    def test_epw_metadata_parsing(self, epw_file):
        """Test that EPW metadata is correctly parsed."""
        df, metadata = io.read_epw(epw_file)

        assert metadata["city"] == "Athens"
        assert abs(metadata["latitude"] - 37.90) < 0.01
        assert abs(metadata["longitude"] - 23.73) < 0.01
        assert abs(metadata["elevation"] - 107.0) < 0.1

    def test_epw_data_columns(self, epw_file):
        """Test that EPW data has expected columns."""
        df, _ = io.read_epw(epw_file)

        # Check for essential weather columns
        expected_cols = [
            "temp_air",
            "relative_humidity",
            "atmospheric_pressure",
            "wind_speed",
            "wind_direction",
            "ghi",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_epw_datetime_index(self, epw_file):
        """Test that EPW data has proper datetime index."""
        df, _ = io.read_epw(epw_file)

        assert df.index.name == "datetime"

        # Check first timestamp
        first_timestamp = df.index[0]
        assert first_timestamp.year == 2024
        assert first_timestamp.month == 1
        assert first_timestamp.day == 1
        assert first_timestamp.hour == 1

    def test_epw_temperature_values(self, epw_file):
        """Test that temperature values are reasonable."""
        df, _ = io.read_epw(epw_file)

        # Temperature should be in Celsius
        assert df["temp_air"].min() >= -50  # Reasonable minimum
        assert df["temp_air"].max() <= 60  # Reasonable maximum

        # Check specific values from sample data
        assert abs(df.iloc[0]["temp_air"] - 9.0) < 0.1

    def test_epw_humidity_values(self, epw_file):
        """Test that humidity values are in valid range."""
        df, _ = io.read_epw(epw_file)

        assert (df["relative_humidity"] >= 0).all()
        assert (df["relative_humidity"] <= 100).all()

        # Check specific value from sample data
        assert df.iloc[0]["relative_humidity"] == 65

    def test_epw_pressure_values(self, epw_file):
        """Test that pressure values are reasonable."""
        df, _ = io.read_epw(epw_file)

        # Pressure should be in Pa
        assert (df["atmospheric_pressure"] > 50000).all()  # > 500 hPa
        assert (df["atmospheric_pressure"] < 110000).all()  # < 1100 hPa

    def test_epw_handles_pathlib_path(self, epw_file):
        """Test that read_epw accepts pathlib.Path."""
        df, metadata = io.read_epw(Path(epw_file))

        assert len(df) == 5
        assert metadata["city"] == "Athens"

    def test_epw_handles_string_path(self, epw_file):
        """Test that read_epw accepts string path."""
        df, metadata = io.read_epw(str(epw_file))

        assert len(df) == 5
        assert metadata["city"] == "Athens"

    def test_epw_missing_file_raises_error(self):
        """Test that reading non-existent EPW file raises error."""
        with pytest.raises(FileNotFoundError):
            io.read_epw("nonexistent.epw")

    def test_to_dataframe_converts_when_pandas_available(self, epw_file):
        """Test that to_dataframe() converts to pandas when available."""
        import pandas as pd

        df, _ = io.read_epw(epw_file)
        pdf = df.to_dataframe()

        assert isinstance(pdf, pd.DataFrame)
        assert isinstance(pdf.index, pd.DatetimeIndex)
        assert len(pdf) == 5


class TestRasterIO:
    """Test raster I/O with GDAL backend fallback."""

    def test_gdal_backend_env_variable(self, monkeypatch):
        """Test that UMEP_USE_GDAL environment variable works."""
        # Skip if GDAL is not available
        try:
            from osgeo import gdal  # noqa: F401

            del gdal  # Silence unused import warning
        except ImportError:
            pytest.skip("GDAL not available")

        # Set environment variable
        monkeypatch.setenv("UMEP_USE_GDAL", "1")

        # Reload _compat (the source of truth for backend selection)
        # to pick up the environment variable change.
        import importlib

        from solweig import _compat

        importlib.reload(_compat)

        # Should use GDAL backend
        assert _compat.GDAL_ENV

    def test_rasterio_backend_default(self, monkeypatch):
        """Test that rasterio is the default backend in a standard environment."""
        import importlib
        import sys

        from solweig import _compat

        # Ensure environment variable is not set
        monkeypatch.delenv("UMEP_USE_GDAL", raising=False)

        # Remove any QGIS mocks that earlier tests may have injected,
        # so _compat.in_osgeo_environment() returns False.
        qgis_keys = [k for k in sys.modules if k == "qgis" or k.startswith("qgis.")]
        saved = {k: sys.modules.pop(k) for k in qgis_keys}
        try:
            importlib.reload(_compat)
        finally:
            sys.modules.update(saved)

        # In a standard environment with rasterio, GDAL_ENV should be False
        assert _compat.RASTERIO_AVAILABLE is True
        assert _compat.GDAL_ENV is False


class TestGeoTIFFLoading:
    """Test GeoTIFF loading functionality."""

    @pytest.fixture
    def sample_geotiff(self, tmp_path):
        """Create a minimal GeoTIFF file for testing."""
        try:
            from osgeo import gdal, osr

            # Create a simple 10x10 raster
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(
                str(tmp_path / "test.tif"),
                10,
                10,
                1,
                gdal.GDT_Float32,
            )

            # Set geotransform
            ds.SetGeoTransform([0, 1, 0, 0, 0, -1])

            # Set projection (WGS84)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())

            # Write data
            band = ds.GetRasterBand(1)
            data = np.arange(100, dtype=np.float32).reshape(10, 10)
            band.WriteArray(data)
            band.SetNoDataValue(-9999)

            # Close dataset
            ds = None

            return tmp_path / "test.tif"

        except ImportError:
            pytest.skip("GDAL not available for creating test file")

    def test_load_raster_returns_tuple(self, sample_geotiff):
        """Test that load_raster returns expected tuple."""
        result = io.load_raster(str(sample_geotiff))

        # Should return (array, transform, crs, nodata)
        assert len(result) == 4

        array, transform, crs, nodata = result

        assert isinstance(array, np.ndarray)
        assert array.shape == (10, 10)
        assert transform is not None
        assert crs is not None

    def test_load_raster_preserves_data(self, sample_geotiff):
        """Test that loaded data matches written data."""
        array, _, _, _ = io.load_raster(str(sample_geotiff))

        # Should match the data we wrote
        expected = np.arange(100, dtype=np.float32).reshape(10, 10)
        np.testing.assert_array_almost_equal(array, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
