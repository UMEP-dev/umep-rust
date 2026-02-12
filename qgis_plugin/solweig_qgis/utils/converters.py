"""
Converters between QGIS parameters and SOLWEIG dataclasses.

Handles translation of QGIS Processing parameters into the dataclasses
expected by the solweig library API.
"""

from __future__ import annotations

from datetime import datetime
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

        # Handle nodata — only honor negative sentinel values (e.g. -9999)
        # to avoid converting valid zero-height pixels to NaN
        nodata = band.GetNoDataValue()
        if nodata is not None and nodata < 0:
            array = np.where(array == nodata, np.nan, array)

        geotransform = list(ds.GetGeoTransform())
        crs_wkt = ds.GetProjection()

        return array, geotransform, crs_wkt
    finally:
        ds = None


def _read_height_mode(
    parameters: dict[str, Any],
    param_name: str,
    default_absolute: bool = True,
) -> bool:
    """Read a per-layer height mode enum and return True if relative.

    Args:
        parameters: Algorithm parameters dict.
        param_name: Enum parameter name (e.g. "DSM_HEIGHT_MODE").
        default_absolute: If True, default is absolute (enum 1); if False, default is relative (enum 0).

    Returns:
        True if the layer uses relative heights, False if absolute.
    """
    default = 1 if default_absolute else 0
    value = parameters.get(param_name, default)
    return (int(value) if isinstance(value, (int, float)) else default) == 0


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
    lo, hi = float(np.nanmin(dsm)), float(np.nanmax(dsm))
    feedback.pushInfo(f"Loaded DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels, range: {lo:.1f} – {hi:.1f} m")

    pixel_size = abs(dsm_gt[1])
    feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

    # Load optional rasters (keeping geotransforms)
    cdsm, cdsm_gt = _load_optional_raster(parameters, "CDSM", context, param_handler)
    if cdsm is not None:
        feedback.pushInfo(
            f"Loaded CDSM (vegetation), range: {float(np.nanmin(cdsm)):.1f} – {float(np.nanmax(cdsm)):.1f} m"
        )

    dem, dem_gt = _load_optional_raster(parameters, "DEM", context, param_handler)
    if dem is not None:
        feedback.pushInfo(
            f"Loaded DEM (ground elevation), range: {float(np.nanmin(dem)):.1f} – {float(np.nanmax(dem)):.1f} m"
        )

    tdsm, tdsm_gt = _load_optional_raster(parameters, "TDSM", context, param_handler)
    if tdsm is not None:
        feedback.pushInfo(
            f"Loaded TDSM (trunk zone), range: {float(np.nanmin(tdsm)):.1f} – {float(np.nanmax(tdsm)):.1f} m"
        )

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

    # Get per-layer height convention flags (enum: 0=relative, 1=absolute)
    dsm_relative = _read_height_mode(parameters, "DSM_HEIGHT_MODE", default_absolute=True)
    cdsm_relative = _read_height_mode(parameters, "CDSM_HEIGHT_MODE", default_absolute=False)
    tdsm_relative = _read_height_mode(parameters, "TDSM_HEIGHT_MODE", default_absolute=False)

    # Create SurfaceData
    surface = solweig.SurfaceData(
        dsm=dsm,
        cdsm=cdsm,
        dem=dem,
        tdsm=tdsm,
        land_cover=land_cover,
        pixel_size=pixel_size,
        dsm_relative=dsm_relative,
        cdsm_relative=cdsm_relative,
        tdsm_relative=tdsm_relative,
    )

    # Store geospatial metadata for output georeferencing
    surface._geotransform = aligned_gt
    surface._crs_wkt = crs_wkt

    # Convert relative heights to absolute where needed
    needs_preprocess = dsm_relative or (cdsm_relative and cdsm is not None) or (tdsm_relative and tdsm is not None)
    if needs_preprocess:
        feedback.pushInfo("Converting relative heights to absolute...")
        surface.preprocess()

    # Fill NaN with ground reference, mask invalid pixels, crop to valid bbox
    # (uses SurfaceData library methods — single source of truth)
    surface.fill_nan()
    surface.compute_valid_mask()
    surface.apply_valid_mask()
    surface.crop_to_valid_bbox()

    feedback.pushInfo(f"After NaN fill + mask + crop: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")

    # Compute wall heights and aspects from DSM
    feedback.setProgressText("Computing wall heights...")
    feedback.pushInfo("Computing walls from DSM...")
    from solweig.physics import wallalgorithms as wa

    walls = wa.findwalls(surface.dsm, 1.0)
    feedback.pushInfo("Computing wall aspects...")
    feedback.setProgressText("Computing wall aspects...")
    dsm_scale = 1.0 / pixel_size
    dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, dsm_scale, surface.dsm, feedback=feedback)
    surface.wall_height = walls
    surface.wall_aspect = dirwalls
    feedback.pushInfo("Wall computation complete")

    # Save cleaned rasters
    if output_dir:
        surface.save_cleaned(output_dir)

    return surface


def load_prepared_surface(
    surface_dir: str,
    feedback: QgsProcessingFeedback,
) -> Any:  # Returns solweig.SurfaceData
    """
    Load a prepared surface directory into SurfaceData.

    Reads GeoTIFFs and metadata saved by the Surface Preprocessing algorithm.

    Args:
        surface_dir: Path to prepared surface directory.
        feedback: Processing feedback.

    Returns:
        solweig.SurfaceData instance with all arrays loaded.

    Raises:
        QgsProcessingException: If required files are missing.
    """
    import json
    import os

    try:
        import solweig
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found.") from e

    # Load metadata
    metadata_path = os.path.join(surface_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise QgsProcessingException(
            f"Not a valid prepared surface directory: {surface_dir}\n"
            "Missing metadata.json. Run 'Prepare Surface Data' first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    feedback.pushInfo(f"Loading prepared surface from {surface_dir}")

    # Load DSM (required)
    dsm_path = os.path.join(surface_dir, "dsm.tif")
    if not os.path.exists(dsm_path):
        raise QgsProcessingException(f"Missing required file: {dsm_path}")

    dsm, gt, crs_wkt = _load_geotiff(dsm_path)
    feedback.pushInfo(f"DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")

    # Load optional rasters
    cdsm = _load_geotiff_if_exists(os.path.join(surface_dir, "cdsm.tif"))
    dem = _load_geotiff_if_exists(os.path.join(surface_dir, "dem.tif"))
    tdsm = _load_geotiff_if_exists(os.path.join(surface_dir, "tdsm.tif"))
    lc = _load_geotiff_if_exists(os.path.join(surface_dir, "land_cover.tif"))
    land_cover = lc.astype(np.uint8) if lc is not None else None
    wall_height = _load_geotiff_if_exists(os.path.join(surface_dir, "wall_height.tif"))
    wall_aspect = _load_geotiff_if_exists(os.path.join(surface_dir, "wall_aspect.tif"))

    pixel_size = metadata.get("pixel_size", abs(gt[1]))

    surface = solweig.SurfaceData(
        dsm=dsm,
        cdsm=cdsm,
        dem=dem,
        tdsm=tdsm,
        land_cover=land_cover,
        pixel_size=pixel_size,
        dsm_relative=False,  # Always absolute after preprocessing
        cdsm_relative=False,
        tdsm_relative=False,
    )
    surface._geotransform = gt
    surface._crs_wkt = crs_wkt
    surface.wall_height = wall_height
    surface.wall_aspect = wall_aspect

    layers = ["dsm"]
    if cdsm is not None:
        layers.append("cdsm")
    if dem is not None:
        layers.append("dem")
    if tdsm is not None:
        layers.append("tdsm")
    if land_cover is not None:
        layers.append("land_cover")
    if wall_height is not None:
        layers.append("walls")
    feedback.pushInfo(f"Loaded layers: {', '.join(layers)}")

    return surface


def _load_geotiff(path: str) -> tuple[NDArray[np.floating], list[float], str]:
    """Load a GeoTIFF file, returning (array, geotransform, crs_wkt)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise QgsProcessingException(f"Cannot open raster: {path}")
    try:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None and nodata < 0:
            array = np.where(array == nodata, np.nan, array)
        geotransform = list(ds.GetGeoTransform())
        crs_wkt = ds.GetProjection()
        return array, geotransform, crs_wkt
    finally:
        ds = None


def _load_geotiff_if_exists(path: str) -> NDArray[np.floating] | None:
    """Load a GeoTIFF if it exists, return None otherwise."""
    import os

    if not os.path.exists(path):
        return None
    arr, _, _ = _load_geotiff(path)
    return arr


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


def create_physics_from_parameters(
    parameters: dict[str, Any],
) -> Any:  # Returns types.SimpleNamespace
    """
    Create a physics namespace from QGIS vegetation parameters.

    Loads default physics and overrides Tree_settings with user-supplied
    transmissivity and seasonal date values.

    Args:
        parameters: Algorithm parameters dict.

    Returns:
        SimpleNamespace with Tree_settings overridden by QGIS parameters.
    """
    try:
        from solweig.loaders import load_physics
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    physics = load_physics()
    ts = physics.Tree_settings.Value
    ts.Transmissivity = parameters.get("TRANSMISSIVITY", 0.03)
    ts.Transmissivity_leafoff = parameters.get("TRANSMISSIVITY_LEAFOFF", 0.5)
    ts.First_day_leaf = int(parameters.get("LEAF_START", 97))
    ts.Last_day_leaf = int(parameters.get("LEAF_END", 300))

    return physics


def load_weather_from_epw(
    epw_path: str,
    start_dt: Any | None,  # QDateTime, datetime, or None
    end_dt: Any | None,  # QDateTime, datetime, or None
    hours_filter: str | None,
    feedback: QgsProcessingFeedback,
) -> list:  # Returns list[solweig.Weather]
    """
    Load weather data from EPW file with optional filtering.

    Args:
        epw_path: Path to EPW file.
        start_dt: Start datetime (inclusive), or None for EPW start.
        end_dt: End datetime (inclusive), or None for EPW end.
        hours_filter: Comma-separated hours to include (e.g., "9,10,11,12").
        feedback: Processing feedback.

    Returns:
        List of solweig.Weather objects.

    Raises:
        QgsProcessingException: If EPW file cannot be read or dates don't overlap.
    """
    try:
        import solweig
        from solweig.io import read_epw
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    # Convert QDateTime to Python naive datetime
    if start_dt is not None and hasattr(start_dt, "toPyDateTime"):
        start_dt = start_dt.toPyDateTime()
    if end_dt is not None and hasattr(end_dt, "toPyDateTime"):
        end_dt = end_dt.toPyDateTime()
    # Strip timezone info to avoid aware/naive comparison errors
    if start_dt is not None and start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)
    if end_dt is not None and end_dt.tzinfo is not None:
        end_dt = end_dt.replace(tzinfo=None)

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

    # Report EPW date range
    epw_start = df.index.min()
    epw_end = df.index.max()
    feedback.pushInfo(f"EPW date range: {epw_start} to {epw_end}")

    # Default to full EPW range when dates not provided
    if start_dt is None:
        start_dt = epw_start if isinstance(epw_start, datetime) else epw_start.to_pydatetime()
        feedback.pushInfo("No start date specified — using EPW start")
    if end_dt is None:
        end_dt = epw_end if isinstance(epw_end, datetime) else epw_end.to_pydatetime()
        feedback.pushInfo("No end date specified — using EPW end")

    # Filter by date range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df_filtered = df[mask]

    # TMY EPW files mix years (e.g., Jan from 2015, Feb from 2009).
    # If exact date filtering yields nothing, match by month-day-hour instead.
    if len(df_filtered) == 0:
        feedback.pushInfo(
            "No exact date matches — trying month/day filter "
            "(EPW may be a Typical Meteorological Year with mixed years)"
        )
        start_md = (start_dt.month, start_dt.day, start_dt.hour)
        end_md = (end_dt.month, end_dt.day, end_dt.hour)

        def _md_tuple(ts):
            return (ts.month, ts.day, ts.hour)

        if start_md <= end_md:
            # Same-year range (e.g., Feb 1 – Feb 7)
            mask = [start_md <= _md_tuple(t) <= end_md for t in df.index]
        else:
            # Cross-year range (e.g., Dec 15 – Jan 15)
            mask = [_md_tuple(t) >= start_md or _md_tuple(t) <= end_md for t in df.index]

        df_filtered = df[mask]

    # Filter by hours if specified
    if hours_list:
        df_filtered = df_filtered[df_filtered.index.hour.isin(hours_list)]

    if len(df_filtered) == 0:
        raise QgsProcessingException(
            f"No timesteps found between {start_dt} and {end_dt}.\n"
            f"The EPW file contains data from {epw_start} to {epw_end}.\n"
            f"Please adjust the date range to overlap with the EPW data."
        )

    # Convert to Weather objects — normalize timestamps to requested year
    target_year = start_dt.year
    weather_series = []
    for timestamp, row in df_filtered.iterrows():
        dt = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
        # Remap to target year so timestamps are contiguous
        try:
            dt = dt.replace(year=target_year)
        except ValueError:
            # Feb 29 in a non-leap target year → skip
            continue
        w = solweig.Weather(
            datetime=dt,
            ta=float(row["temp_air"]) if not np.isnan(row["temp_air"]) else 20.0,
            rh=float(row["relative_humidity"]) if not np.isnan(row["relative_humidity"]) else 50.0,
            global_rad=float(row["ghi"]) if not np.isnan(row["ghi"]) else 0.0,
            ws=float(row["wind_speed"]) if not np.isnan(row["wind_speed"]) else 1.0,
            pressure=(float(row["atmospheric_pressure"]) / 100.0)  # Pa → hPa
            if not np.isnan(row["atmospheric_pressure"])
            else 1013.25,
            measured_direct_rad=float(row["dni"]) if not np.isnan(row["dni"]) else None,
            measured_diffuse_rad=float(row["dhi"]) if not np.isnan(row["dhi"]) else None,
        )
        weather_series.append(w)

    if not weather_series:
        raise QgsProcessingException(
            f"No timesteps found between {start_dt} and {end_dt}.\n"
            f"The EPW file contains data from {epw_start} to {epw_end}.\n"
            f"Please adjust the date range to overlap with the EPW data."
        )

    feedback.pushInfo(f"Loaded {len(weather_series)} timesteps from EPW")
    feedback.pushInfo(f"Period: {weather_series[0].datetime} to {weather_series[-1].datetime}")

    return weather_series


def load_weather_from_umep_met(
    met_path: str,
    start_dt: Any | None,
    end_dt: Any | None,
    hours_filter: str | None,
    feedback: QgsProcessingFeedback,
) -> list:  # Returns list[solweig.Weather]
    """
    Load weather data from a UMEP/SUEWS meteorological forcing file.

    Args:
        met_path: Path to UMEP met file.
        start_dt: Start datetime (inclusive), or None for full range.
        end_dt: End datetime (inclusive), or None for full range.
        hours_filter: Comma-separated hours to include (e.g., "9,10,11,12").
        feedback: Processing feedback.

    Returns:
        List of solweig.Weather objects.

    Raises:
        QgsProcessingException: If file cannot be read or no data found.
    """
    try:
        from solweig.models.weather import Weather
    except ImportError as e:
        raise QgsProcessingException("SOLWEIG library not found. Please install solweig package.") from e

    if not met_path:
        raise QgsProcessingException("No UMEP met file specified.")

    feedback.pushInfo(f"Loading UMEP met file: {met_path}")

    # Convert QDateTime to Python naive datetime
    if start_dt is not None and hasattr(start_dt, "toPyDateTime"):
        start_dt = start_dt.toPyDateTime()
    if end_dt is not None and hasattr(end_dt, "toPyDateTime"):
        end_dt = end_dt.toPyDateTime()
    if start_dt is not None and start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)
    if end_dt is not None and end_dt.tzinfo is not None:
        end_dt = end_dt.replace(tzinfo=None)

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

    # Load via Weather.from_umep_met()
    try:
        weather_series = Weather.from_umep_met(
            paths=[met_path],
            resample_hourly=True,
            start=start_dt,
            end=end_dt,
        )
    except FileNotFoundError as e:
        raise QgsProcessingException(f"UMEP met file not found: {e}") from e
    except ValueError as e:
        raise QgsProcessingException(f"Error reading UMEP met file: {e}") from e

    if not weather_series:
        raise QgsProcessingException("No valid timesteps found in UMEP met file.")

    # Report date range
    met_start = weather_series[0].datetime
    met_end = weather_series[-1].datetime
    feedback.pushInfo(f"UMEP met date range: {met_start} to {met_end}")

    # Apply hours filter if specified
    if hours_list:
        weather_series = [w for w in weather_series if w.datetime.hour in hours_list]
        if not weather_series:
            raise QgsProcessingException(
                f"No timesteps remaining after hour filter {hours_list}.\n"
                f"The data contains hours from {met_start} to {met_end}."
            )

    feedback.pushInfo(f"Loaded {len(weather_series)} timesteps from UMEP met")
    feedback.pushInfo(f"Period: {weather_series[0].datetime} to {weather_series[-1].datetime}")

    return weather_series
