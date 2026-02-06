"""
Converters between QGIS parameters and SOLWEIG dataclasses.

Handles translation of QGIS Processing parameters into the dataclasses
expected by the solweig library API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal, osr
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsRasterLayer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_raster_from_layer(
    layer: QgsRasterLayer,
) -> tuple[NDArray[np.floating], list[float], str]:
    """
    Load QGIS raster layer to numpy array using GDAL.

    Args:
        layer: QGIS raster layer to load.

    Returns:
        tuple of (array, geotransform, crs_wkt):
            - array: 2D numpy float32 array
            - geotransform: GDAL 6-tuple
            - crs_wkt: CRS as WKT string

    Raises:
        QgsProcessingException: If raster cannot be opened.
    """
    source = layer.source()
    ds = gdal.Open(source, gdal.GA_ReadOnly)
    if ds is None:
        raise QgsProcessingException(f"Cannot open raster: {source}")

    try:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)

        # Handle nodata
        nodata = band.GetNoDataValue()
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)

        geotransform = list(ds.GetGeoTransform())
        crs_wkt = ds.GetProjection()

        return array, geotransform, crs_wkt
    finally:
        ds = None


def create_surface_from_parameters(
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    param_handler: Any,  # Algorithm instance with parameterAsRasterLayer
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.SurfaceData
    """
    Create SurfaceData from QGIS processing parameters.

    Loads all surface rasters, handles height conversion, and
    attaches geospatial metadata for output georeferencing.

    Args:
        parameters: Algorithm parameters dict.
        context: Processing context.
        param_handler: Object with parameterAsRasterLayer method.
        feedback: Processing feedback.

    Returns:
        solweig.SurfaceData instance with populated arrays and metadata.

    Raises:
        QgsProcessingException: If required DSM is missing or invalid.
    """
    # Import solweig here to avoid import errors if not installed
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Load required DSM
    dsm_layer = param_handler.parameterAsRasterLayer(parameters, "DSM", context)
    if dsm_layer is None:
        raise QgsProcessingException("DSM layer is required")

    dsm, geotransform, crs_wkt = load_raster_from_layer(dsm_layer)
    feedback.pushInfo(f"Loaded DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")

    # Extract pixel size from geotransform
    pixel_size = abs(geotransform[1])
    feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

    # Load optional surface rasters
    cdsm = None
    if "CDSM" in parameters and parameters["CDSM"]:
        cdsm_layer = param_handler.parameterAsRasterLayer(parameters, "CDSM", context)
        if cdsm_layer:
            cdsm, _, _ = load_raster_from_layer(cdsm_layer)
            feedback.pushInfo("Loaded CDSM (vegetation)")

    dem = None
    if "DEM" in parameters and parameters["DEM"]:
        dem_layer = param_handler.parameterAsRasterLayer(parameters, "DEM", context)
        if dem_layer:
            dem, _, _ = load_raster_from_layer(dem_layer)
            feedback.pushInfo("Loaded DEM (ground elevation)")

    tdsm = None
    if "TDSM" in parameters and parameters["TDSM"]:
        tdsm_layer = param_handler.parameterAsRasterLayer(parameters, "TDSM", context)
        if tdsm_layer:
            tdsm, _, _ = load_raster_from_layer(tdsm_layer)
            feedback.pushInfo("Loaded TDSM (trunk zone)")

    land_cover = None
    if "LAND_COVER" in parameters and parameters["LAND_COVER"]:
        lc_layer = param_handler.parameterAsRasterLayer(parameters, "LAND_COVER", context)
        if lc_layer:
            lc_arr, _, _ = load_raster_from_layer(lc_layer)
            land_cover = lc_arr.astype(np.uint8)
            feedback.pushInfo("Loaded land cover classification")

    # Get height convention flag
    relative_heights = parameters.get("RELATIVE_HEIGHTS", True)

    # Create SurfaceData
    surface = solweig.SurfaceData(
        dsm=dsm,
        cdsm=cdsm,
        dem=dem,
        tdsm=tdsm,
        land_cover=land_cover,
        pixel_size=pixel_size,
        relative_heights=relative_heights,
    )

    # Store geospatial metadata for output georeferencing
    surface._geotransform = geotransform
    surface._crs_wkt = crs_wkt

    # Preprocess if using relative heights and we have vegetation data
    if relative_heights and (cdsm is not None or tdsm is not None):
        feedback.pushInfo("Converting relative vegetation heights to absolute...")
        surface.preprocess()

    return surface


def create_location_from_parameters(
    parameters: dict[str, Any],
    surface: Any,  # solweig.SurfaceData
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.Location
    """
    Create Location from QGIS processing parameters.

    Supports auto-extraction from DSM CRS or manual input.

    Args:
        parameters: Algorithm parameters dict.
        surface: SurfaceData instance (for auto-extraction).
        feedback: Processing feedback.

    Returns:
        solweig.Location instance.

    Raises:
        QgsProcessingException: If location cannot be determined.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    utc_offset = parameters.get("UTC_OFFSET", 0)

    if parameters.get("AUTO_EXTRACT_LOCATION", False):
        # Extract from DSM CRS
        feedback.pushInfo("Auto-extracting location from DSM CRS...")

        if surface._crs_wkt is None:
            raise QgsProcessingException("Cannot auto-extract location: DSM has no CRS information")

        # Get center point of raster
        gt = surface._geotransform
        rows, cols = surface.dsm.shape
        center_x = gt[0] + cols * gt[1] / 2
        center_y = gt[3] + rows * gt[5] / 2

        # Transform to WGS84
        source_srs = osr.SpatialReference()
        source_srs.ImportFromWkt(surface._crs_wkt)

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WGS84

        transform = osr.CoordinateTransformation(source_srs, target_srs)
        lon, lat, _ = transform.TransformPoint(center_x, center_y)

        feedback.pushInfo(f"Location: {lat:.4f}N, {lon:.4f}E")

        location = solweig.Location(
            latitude=lat,
            longitude=lon,
            utc_offset=utc_offset,
        )
    else:
        # Use manual input
        latitude = parameters.get("LATITUDE")
        longitude = parameters.get("LONGITUDE")

        if latitude is None or longitude is None:
            raise QgsProcessingException("Latitude and longitude are required when auto-extract is disabled")

        location = solweig.Location(
            latitude=latitude,
            longitude=longitude,
            utc_offset=utc_offset,
        )
        feedback.pushInfo(f"Location: {latitude:.4f}N, {longitude:.4f}E")

    return location


def create_weather_from_parameters(
    parameters: dict[str, Any],
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.Weather
    """
    Create Weather from QGIS processing parameters.

    Args:
        parameters: Algorithm parameters dict.
        feedback: Processing feedback.

    Returns:
        solweig.Weather instance.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Get datetime from QDateTime parameter
    qdt = parameters["DATETIME"]
    dt = qdt.toPyDateTime()

    weather = solweig.Weather(
        datetime=dt,
        ta=parameters.get("TEMPERATURE", 25.0),
        rh=parameters.get("HUMIDITY", 50.0),
        global_rad=parameters.get("GLOBAL_RADIATION", 800.0),
        ws=parameters.get("WIND_SPEED", 1.0),
        pressure=parameters.get("PRESSURE", 1013.25),
    )

    feedback.pushInfo(
        f"Weather: {dt.strftime('%Y-%m-%d %H:%M')}, "
        f"Ta={weather.ta:.1f}C, RH={weather.rh:.0f}%, "
        f"G={weather.global_rad:.0f}W/m2"
    )

    return weather


def create_human_params_from_parameters(
    parameters: dict[str, Any],
) -> Any:  # Returns solweig.HumanParams
    """
    Create HumanParams from QGIS processing parameters.

    Args:
        parameters: Algorithm parameters dict.

    Returns:
        solweig.HumanParams instance.
    """
    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Map posture enum to string
    posture_map = {0: "standing", 1: "sitting"}
    posture = posture_map.get(parameters.get("POSTURE", 0), "standing")

    # Basic human params
    human = solweig.HumanParams(
        posture=posture,
        abs_k=parameters.get("ABS_K", 0.7),
    )

    # Add detailed body params if present (for PET)
    if "WEIGHT" in parameters:
        human.weight = parameters["WEIGHT"]
    if "HEIGHT" in parameters:
        human.height = parameters["HEIGHT"]
    if "AGE" in parameters:
        human.age = parameters["AGE"]
    if "ACTIVITY" in parameters:
        human.activity = parameters["ACTIVITY"]
    if "CLOTHING" in parameters:
        human.clothing = parameters["CLOTHING"]
    if "SEX" in parameters:
        sex_map = {0: 1, 1: 2}
        human.sex = sex_map.get(parameters["SEX"], 1)

    return human


def load_weather_from_epw(
    epw_path: str,
    start_dt: Any,  # QDateTime or datetime
    end_dt: Any,  # QDateTime or datetime
    hours_filter: str | None,
    feedback: QgsProcessingFeedback,
) -> list:  # Returns list[solweig.Weather]
    """
    Load weather data from EPW file with optional filtering.

    Args:
        epw_path: Path to EPW file.
        start_dt: Start datetime (inclusive).
        end_dt: End datetime (inclusive).
        hours_filter: Comma-separated hours to include (e.g., "9,10,11,12").
        feedback: Processing feedback.

    Returns:
        List of solweig.Weather objects.

    Raises:
        QgsProcessingException: If EPW file cannot be read.
    """
    try:
        import solweig
        from solweig.io import read_epw
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Convert QDateTime to Python datetime if needed
    if hasattr(start_dt, "toPyDateTime"):
        start_dt = start_dt.toPyDateTime()
    if hasattr(end_dt, "toPyDateTime"):
        end_dt = end_dt.toPyDateTime()

    # Parse hours filter
    hours_list = None
    if hours_filter:
        try:
            hours_str = hours_filter.replace(" ", "")
            hours_list = [int(h) for h in hours_str.split(",")]
            feedback.pushInfo(f"Hour filter: {hours_list}")
        except ValueError:
            feedback.reportError(
                f"Invalid hours filter: {hours_filter}. Using all hours.",
                fatalError=False,
            )

    # Read EPW file
    try:
        df, metadata = read_epw(epw_path)
    except FileNotFoundError as e:
        raise QgsProcessingException(f"EPW file not found: {epw_path}") from e
    except Exception as e:
        raise QgsProcessingException(f"Error reading EPW file: {e}") from e

    feedback.pushInfo(
        f"EPW location: {metadata.get('city', 'Unknown')}, "
        f"lat={metadata.get('latitude', 'N/A')}, lon={metadata.get('longitude', 'N/A')}"
    )

    # Filter by date range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df_filtered = df[mask]

    # Filter by hours if specified
    if hours_list:
        df_filtered = df_filtered[df_filtered.index.hour.isin(hours_list)]

    if len(df_filtered) == 0:
        raise QgsProcessingException(f"No timesteps found between {start_dt} and {end_dt}")

    # Convert to Weather objects
    weather_series = []
    for timestamp, row in df_filtered.iterrows():
        w = solweig.Weather(
            datetime=timestamp.to_pydatetime(),
            ta=row["temp_air"],
            rh=row["relative_humidity"],
            global_rad=row["ghi"],
            ws=row.get("wind_speed", 1.0),
            pressure=row.get("atmospheric_pressure", 1013.25),
        )
        weather_series.append(w)

    feedback.pushInfo(f"Loaded {len(weather_series)} timesteps from EPW")
    feedback.pushInfo(f"Period: {weather_series[0].datetime} to {weather_series[-1].datetime}")

    return weather_series
