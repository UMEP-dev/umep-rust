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


def _load_optional_raster(
    parameters: dict[str, Any],
    param_name: str,
    context: QgsProcessingContext,
    param_handler: Any,
) -> tuple[NDArray[np.floating] | None, list[float] | None]:
    """Load optional raster, returning (array, geotransform) or (None, None)."""
    if param_name not in parameters or not parameters[param_name]:
        return None, None
    layer = param_handler.parameterAsRasterLayer(parameters, param_name, context)
    if layer is None:
        return None, None
    arr, gt, _ = load_raster_from_layer(layer)
    return arr, gt


def _align_layer(
    arr: NDArray[np.floating],
    gt: list[float],
    target_bbox: list[float],
    pixel_size: float,
    method: str,
    crs_wkt: str,
) -> NDArray[np.floating]:
    """Resample a raster to the target grid if extents differ."""
    from solweig.utils import extract_bounds, resample_to_grid

    bounds = extract_bounds(gt, arr.shape)
    needs_resample = (
        abs(bounds[0] - target_bbox[0]) > 1e-6
        or abs(bounds[1] - target_bbox[1]) > 1e-6
        or abs(bounds[2] - target_bbox[2]) > 1e-6
        or abs(bounds[3] - target_bbox[3]) > 1e-6
        or abs(abs(gt[1]) - pixel_size) > 1e-6
    )
    if needs_resample:
        arr, _ = resample_to_grid(arr, gt, target_bbox, pixel_size, method=method, src_crs=crs_wkt)
    return arr


def create_surface_from_parameters(
    parameters: dict[str, Any],
    context: QgsProcessingContext,
    param_handler: Any,  # Algorithm instance with parameterAsRasterLayer
    feedback: QgsProcessingFeedback,
    bbox: list[float] | None = None,
    output_dir: str | None = None,
) -> Any:  # Returns solweig.SurfaceData
    """
    Create SurfaceData from QGIS processing parameters.

    Loads all surface rasters, aligns them to a common grid (intersection
    of all extents or user-specified bbox), computes a unified valid mask,
    and saves cleaned rasters to disk.

    Args:
        parameters: Algorithm parameters dict.
        context: Processing context.
        param_handler: Object with parameterAsRasterLayer method.
        feedback: Processing feedback.
        bbox: Optional explicit bounding box [minx, miny, maxx, maxy].
        output_dir: Optional directory for saving cleaned rasters.

    Returns:
        solweig.SurfaceData instance with aligned, masked arrays.

    Raises:
        QgsProcessingException: If required DSM is missing or invalid.
    """
    try:
        import solweig
        from solweig.utils import extract_bounds, intersect_bounds
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Load required DSM (with geotransform)
    dsm_layer = param_handler.parameterAsRasterLayer(parameters, "DSM", context)
    if dsm_layer is None:
        raise QgsProcessingException("DSM layer is required")

    dsm, dsm_gt, crs_wkt = load_raster_from_layer(dsm_layer)
    feedback.pushInfo(f"Loaded DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")

    pixel_size = abs(dsm_gt[1])
    feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

    # Load optional rasters (keeping geotransforms)
    cdsm, cdsm_gt = _load_optional_raster(parameters, "CDSM", context, param_handler)
    if cdsm is not None:
        feedback.pushInfo("Loaded CDSM (vegetation)")

    dem, dem_gt = _load_optional_raster(parameters, "DEM", context, param_handler)
    if dem is not None:
        feedback.pushInfo("Loaded DEM (ground elevation)")

    tdsm, tdsm_gt = _load_optional_raster(parameters, "TDSM", context, param_handler)
    if tdsm is not None:
        feedback.pushInfo("Loaded TDSM (trunk zone)")

    lc_arr, lc_gt = _load_optional_raster(parameters, "LAND_COVER", context, param_handler)
    land_cover = lc_arr.astype(np.uint8) if lc_arr is not None else None
    if land_cover is not None:
        feedback.pushInfo("Loaded land cover classification")

    # Compute extent intersection of all loaded layers
    bounds_list = [extract_bounds(dsm_gt, dsm.shape)]
    for arr, gt in [(cdsm, cdsm_gt), (dem, dem_gt), (tdsm, tdsm_gt), (lc_arr, lc_gt)]:
        if arr is not None and gt is not None:
            bounds_list.append(extract_bounds(gt, arr.shape))

    if bbox is not None:
        target_bbox = bbox
    elif len(bounds_list) > 1:
        target_bbox = intersect_bounds(bounds_list)
        feedback.pushInfo(f"Auto-computed intersection extent: {target_bbox}")
    else:
        target_bbox = bounds_list[0]

    # Align all layers to the target grid
    dsm = _align_layer(dsm, dsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if cdsm is not None and cdsm_gt is not None:
        cdsm = _align_layer(cdsm, cdsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if dem is not None and dem_gt is not None:
        dem = _align_layer(dem, dem_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if tdsm is not None and tdsm_gt is not None:
        tdsm = _align_layer(tdsm, tdsm_gt, target_bbox, pixel_size, "bilinear", crs_wkt)
    if land_cover is not None and lc_gt is not None:
        land_cover = _align_layer(
            land_cover.astype(np.float32),
            lc_gt,
            target_bbox,
            pixel_size,
            "nearest",
            crs_wkt,
        ).astype(np.uint8)

    feedback.pushInfo(f"Aligned grid: {dsm.shape[1]}x{dsm.shape[0]} pixels")

    # Build aligned geotransform for the target bbox
    aligned_gt = [target_bbox[0], pixel_size, 0, target_bbox[3], 0, -pixel_size]

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
    surface._geotransform = aligned_gt
    surface._crs_wkt = crs_wkt

    # Preprocess if using relative heights and we have vegetation data
    if relative_heights and (cdsm is not None or tdsm is not None):
        feedback.pushInfo("Converting relative vegetation heights to absolute...")
        surface.preprocess()

    # Compute unified valid mask, apply across layers, crop to valid bbox
    surface.compute_valid_mask()
    surface.apply_valid_mask()
    surface.crop_to_valid_bbox()
    feedback.pushInfo(f"After NaN masking + crop: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")

    # Save cleaned rasters
    if output_dir:
        surface.save_cleaned(output_dir)

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
