import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _detect_osgeo_environment() -> bool:
    """
    Detect if we're running in an OSGeo4W or QGIS environment.

    These environments have their own GDAL installation and pip-installed
    rasterio will cause DLL conflicts on Windows.
    """
    # Check for QGIS
    if "qgis" in sys.modules or "qgis.core" in sys.modules:
        return True

    # Check for QGIS environment variables
    if any(key in os.environ for key in ("QGIS_PREFIX_PATH", "QGIS_DEBUG")):
        return True

    # Check for OSGeo4W environment
    if "OSGEO4W_ROOT" in os.environ:
        return True

    # Check if Python executable is inside OSGeo4W or QGIS directory (Windows)
    exe_path = sys.executable.lower()
    return any(marker in exe_path for marker in ("osgeo4w", "qgis"))


def _try_import_gdal() -> bool:
    """Try to import GDAL and return True if successful."""
    try:
        from osgeo import gdal, osr  # noqa: F401

        return True
    except (ImportError, OSError) as e:
        logger.debug(f"GDAL import failed: {e}")
        return False


def _try_import_rasterio() -> bool:
    """
    Try to import rasterio and return True if successful.

    This catches both ImportError and OSError (DLL load failures).
    """
    try:
        import pyproj  # noqa: F401
        import rasterio  # noqa: F401
        from rasterio.features import rasterize  # noqa: F401
        from rasterio.mask import mask  # noqa: F401
        from rasterio.transform import Affine, from_origin  # noqa: F401
        from rasterio.windows import Window  # noqa: F401
        from shapely import geometry  # noqa: F401

        return True
    except (ImportError, OSError) as e:
        logger.debug(f"Rasterio import failed: {e}")
        return False


def _setup_geospatial_backend() -> bool:
    """
    Set up the geospatial backend (rasterio or GDAL).

    Returns GDAL_ENV: True if using GDAL, False if using rasterio.

    Priority:
    1. UMEP_USE_GDAL=1 environment variable forces GDAL
    2. In OSGeo4W/QGIS environments: prefer GDAL (avoids DLL conflicts)
    3. Otherwise: try rasterio first, fall back to GDAL
    """
    # Allow forcing GDAL via environment variable
    if os.environ.get("UMEP_USE_GDAL", "").lower() in ("1", "true", "yes"):
        if _try_import_gdal():
            logger.info("Using GDAL for raster operations (forced via UMEP_USE_GDAL).")
            return True
        else:
            raise ImportError(
                "UMEP_USE_GDAL is set but GDAL could not be imported. "
                "Install GDAL or unset UMEP_USE_GDAL."
            )

    # In OSGeo4W/QGIS: prefer GDAL to avoid DLL conflicts
    in_osgeo = _detect_osgeo_environment()
    if in_osgeo:
        logger.debug("Detected OSGeo4W/QGIS environment, preferring GDAL backend.")
        if _try_import_gdal():
            logger.info("Using GDAL for raster operations (OSGeo4W/QGIS environment).")
            return True
        # GDAL should always be available in OSGeo4W/QGIS, but fall back just in case
        logger.warning("GDAL import failed in OSGeo4W/QGIS environment, trying rasterio...")
        if _try_import_rasterio():
            logger.info("Using rasterio for raster operations.")
            return False
        raise ImportError(
            "Failed to import both GDAL and rasterio in OSGeo4W/QGIS environment.\n"
            "This is unexpected - GDAL should be available. Check your installation."
        )

    # Standard environment: prefer rasterio, fall back to GDAL
    if _try_import_rasterio():
        logger.info("Using rasterio for raster operations.")
        return False

    logger.warning("Rasterio import failed, trying GDAL...")
    if _try_import_gdal():
        logger.info("Using GDAL for raster operations.")
        return True

    # Neither worked
    raise ImportError(
        "Neither rasterio nor GDAL could be imported.\n"
        "Install with: pip install rasterio\n"
        "Or for QGIS/OSGeo4W environments, ensure GDAL is properly configured."
    )


# Determine which backend to use
GDAL_ENV = _setup_geospatial_backend()

# Now do the actual imports based on the backend
if GDAL_ENV:
    from osgeo import gdal, osr
else:
    import pyproj
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.transform import Affine, from_origin
    from rasterio.windows import Window
    from shapely import geometry


FLOAT_TOLERANCE = 1e-9


def _assert_north_up(transform) -> None:
    """Ensure the raster transform describes a north-up raster."""
    if hasattr(transform, "b") and hasattr(transform, "d"):
        if not math.isclose(transform.b, 0.0, abs_tol=FLOAT_TOLERANCE) or not math.isclose(
            transform.d, 0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise ValueError("Only north-up rasters (no rotation) are supported.")
    else:
        # GDAL-style tuple (c, a, b, f, d, e)
        if len(transform) < 6:
            raise ValueError("Transform must contain 6 elements.")
        if not math.isclose(transform[2], 0.0, abs_tol=FLOAT_TOLERANCE) or not math.isclose(
            transform[4], 0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise ValueError("Only north-up rasters (no rotation) are supported.")


def _shrink_axis_to_grid(min_val: float, max_val: float, origin: float, pixel_size: float) -> tuple[float, float]:
    if pixel_size == 0:
        raise ValueError("Pixel size must be non-zero to shrink bbox to pixel grid.")
    step = abs(pixel_size)
    start_idx = math.ceil(((min_val - origin) / step) - FLOAT_TOLERANCE)
    end_idx = math.floor(((max_val - origin) / step) + FLOAT_TOLERANCE)
    new_min = origin + start_idx * step
    new_max = origin + end_idx * step
    if not new_max > new_min:
        raise ValueError("Bounding box collapsed after snapping to the pixel grid.")
    return new_min, new_max


def shrink_bbox_to_pixel_grid(
    bbox: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
) -> tuple[float, float, float, float]:
    """Shrink bbox so its edges land on the pixel grid defined by the raster origin."""

    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("Bounding box is invalid (min must be < max for both axes).")
    snapped_minx, snapped_maxx = _shrink_axis_to_grid(minx, maxx, origin_x, pixel_width)
    snapped_miny, snapped_maxy = _shrink_axis_to_grid(miny, maxy, origin_y, pixel_height)
    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy


def _bounds_to_tuple(bounds) -> tuple[float, float, float, float]:
    if hasattr(bounds, "left"):
        return bounds.left, bounds.bottom, bounds.right, bounds.top
    return tuple(bounds)


def _validate_bbox_within_bounds(
    bbox: tuple[float, float, float, float], bounds, *, tol: float = FLOAT_TOLERANCE
) -> None:
    minx, miny, maxx, maxy = bbox
    left, bottom, right, top = _bounds_to_tuple(bounds)
    if minx < left - tol or maxx > right + tol or miny < bottom - tol or maxy > top + tol:
        raise ValueError("Bounding box is not fully contained within the raster dataset bounds")


def _compute_bounds_from_transform(transform, width: int, height: int) -> tuple[float, float, float, float]:
    """Return raster bounds for a GDAL-style transform tuple."""
    left = transform[0]
    top = transform[3]
    right = transform[0] + width * transform[1]
    bottom = transform[3] + height * transform[5]
    minx = min(left, right)
    maxx = max(left, right)
    miny = min(top, bottom)
    maxy = max(top, bottom)
    return minx, miny, maxx, maxy


def _normalise_bbox(bbox_sequence) -> tuple[float, float, float, float]:
    try:
        minx, miny, maxx, maxy = bbox_sequence
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Bounding box must contain exactly four numeric values") from exc
    return float(minx), float(miny), float(maxx), float(maxy)


def rasterise_gdf(gdf, geom_col, ht_col, bbox=None, pixel_size: int = 1):
    # Define raster parameters
    if bbox is not None:
        # Unpack bbox values
        minx, miny, maxx, maxy = _normalise_bbox(bbox)
    else:
        # Use the total bounds of the GeoDataFrame
        minx, miny, maxx, maxy = map(float, gdf.total_bounds)
    if pixel_size <= 0:
        raise ValueError("Pixel size must be a positive number.")
    minx, miny, maxx, maxy = shrink_bbox_to_pixel_grid(
        (minx, miny, maxx, maxy),
        origin_x=minx,
        origin_y=maxy,
        pixel_width=pixel_size,
        pixel_height=pixel_size,
    )
    width = int(round((maxx - minx) / pixel_size))
    height = int(round((maxy - miny) / pixel_size))
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box collapsed after snapping to pixel grid.")
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    # Create a blank array for the raster
    raster = np.zeros((height, width), dtype=np.float32)
    # Burn geometries into the raster
    shapes = ((geom, value) for geom, value in zip(gdf[geom_col], gdf[ht_col], strict=True))
    raster = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.float32)

    return raster, transform


def check_path(path_str: str | Path, make_dir: bool = False) -> Path:
    # Ensure path exists
    path = Path(path_str).absolute()
    if not path.parent.exists():
        if make_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Parent directory {path.parent} does not exist for path {path}. Set make_dir=True to create it.")
    if not path.exists() and not path.suffix:
        if make_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Path {path} does not exist. Set make_dir=True to create it.")
    return path


def save_raster(
    out_path_str: str,
    data_arr: np.ndarray,
    trf_arr: list[float],
    crs_wkt: str,
    no_data_val: float = -9999,
    coerce_f64_to_f32: bool = True,
):
    """
    Save raster to GeoTIFF.

    Args:
        out_path_str: Output file path
        data_arr: 2D numpy array to save
        trf_arr: GDAL-style geotransform [top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height]
        crs_wkt: CRS in WKT format
        no_data_val: No-data value to use
        coerce_f64_to_f32: If True, convert float64 arrays to float32 before saving
                           (default: True for memory efficiency)
    """
    # Only convert float64 to float32, leave ints/bools unchanged
    if coerce_f64_to_f32 and data_arr.dtype == np.float64:
        data_arr = data_arr.astype(np.float32)

    attempts = 2
    while attempts > 0:
        attempts -= 1
        try:
            # Save raster using GDAL or rasterio
            out_path = check_path(out_path_str, make_dir=True)
            height, width = data_arr.shape
            if GDAL_ENV is False:
                trf = Affine.from_gdal(*trf_arr)
                crs = None
                if crs_wkt:
                    crs = pyproj.CRS(crs_wkt)
                with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=data_arr.dtype,
                    crs=crs,
                    transform=trf,
                    nodata=no_data_val,
                ) as dst:
                    dst.write(data_arr, 1)
            else:
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.Create(str(out_path), width, height, 1, gdal.GDT_Float32)
                ds.SetGeoTransform(trf_arr)
                if crs_wkt:
                    ds.SetProjection(crs_wkt)
                band = ds.GetRasterBand(1)
                band.SetNoDataValue(no_data_val)
                band.WriteArray(data_arr)
                ds = None
            return
        except Exception as e:
            if attempts == 0:
                raise e
            logger.warning(f"Failed to save raster to {out_path_str}: {e}. Retrying...")


def get_raster_metadata(path_str: str | Path) -> dict:
    """
    Get raster metadata without loading the whole file.
    Returns dict with keys: rows, cols, transform, crs, nodata, res.
    Transform is always a list [c, a, b, f, d, e] (GDAL-style).
    CRS is always a WKT string (or None).
    """
    path = check_path(path_str)
    if GDAL_ENV is False:
        with rasterio.open(path) as src:
            # Convert Affine to GDAL-style list
            trf = src.transform
            transform_list = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            # Convert CRS to WKT string
            crs_wkt = src.crs.to_wkt() if src.crs is not None else None
            return {
                "rows": src.height,
                "cols": src.width,
                "transform": transform_list,
                "crs": crs_wkt,
                "nodata": src.nodata,
                "res": src.res,  # (xres, yres)
                "bounds": src.bounds,
            }
    else:
        ds = gdal.Open(str(path))
        if ds is None:
            raise OSError(f"Could not open {path}")
        gt = ds.GetGeoTransform()
        return {
            "rows": ds.RasterYSize,
            "cols": ds.RasterXSize,
            "transform": gt,
            "crs": ds.GetProjection() or None,
            "nodata": ds.GetRasterBand(1).GetNoDataValue(),
            "res": (gt[1], abs(gt[5])),  # Approximate resolution
        }


def read_raster_window(path_str: str | Path, window: tuple[slice, slice], band: int = 1) -> np.ndarray:
    """
    Read a window from a raster file.
    window is (row_slice, col_slice).
    """
    path = check_path(path_str)
    row_slice, col_slice = window

    # Handle None slices (read full dimension)
    # This is tricky without knowing full shape, so we assume caller provides valid slices
    # or we'd need to open file to check shape first.
    # For now, assume valid integer slices.

    if GDAL_ENV is False:
        with rasterio.open(path) as src:
            # rasterio Window(col_off, row_off, width, height)
            # Slices are start:stop
            r_start = row_slice.start if row_slice.start is not None else 0
            r_stop = row_slice.stop if row_slice.stop is not None else src.height
            c_start = col_slice.start if col_slice.start is not None else 0
            c_stop = col_slice.stop if col_slice.stop is not None else src.width

            win = Window(
                col_off=c_start,
                row_off=r_start,
                width=c_stop - c_start,
                height=r_stop - r_start,
            )
            return src.read(band, window=win)
    else:
        ds = gdal.Open(str(path))
        if ds is None:
            raise OSError(f"Could not open {path}")

        r_start = row_slice.start if row_slice.start is not None else 0
        r_stop = row_slice.stop if row_slice.stop is not None else ds.RasterYSize
        c_start = col_slice.start if col_slice.start is not None else 0
        c_stop = col_slice.stop if col_slice.stop is not None else ds.RasterXSize

        xoff = c_start
        yoff = r_start
        xsize = c_stop - c_start
        ysize = r_stop - r_start

        return ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize)


def load_raster(
    path_str: str, bbox: list[int] | None = None, band: int = 0, coerce_f64_to_f32: bool = True
) -> tuple[np.ndarray, list[float], str | None, float | None]:
    """
    Load raster, optionally crop to bbox.

    Args:
        path_str: Path to raster file
        bbox: Optional bounding box [minx, miny, maxx, maxy]
        band: Band index to read (0-based)
        coerce_f64_to_f32: If True, coerce array to float32 (default: True for memory efficiency)

    Returns:
        Tuple of (array, transform, crs_wkt, no_data_value)
    """
    # Load raster, optionally crop to bbox
    path = check_path(path_str, make_dir=False)
    if not path.exists():
        raise FileNotFoundError(f"Raster file {path} does not exist.")
    if GDAL_ENV is False:
        with rasterio.open(path) as dataset:
            _assert_north_up(dataset.transform)
            crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else None
            no_data_val = dataset.nodata
            transform = dataset.transform
            if bbox is not None:
                bbox_tuple = _normalise_bbox(bbox)
                snapped_bbox = shrink_bbox_to_pixel_grid(
                    bbox_tuple,
                    origin_x=transform.c,
                    origin_y=transform.f,
                    pixel_width=transform.a,
                    pixel_height=transform.e,
                )
                _validate_bbox_within_bounds(snapped_bbox, dataset.bounds)
                bbox_geom = geometry.box(*snapped_bbox)
                rast, trf = mask(dataset, [bbox_geom], crop=True)
            else:
                rast = dataset.read()
                trf = transform
            # Convert rasterio Affine to GDAL-style list
            trf_arr = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            # rast shape: (bands, rows, cols)
            if rast.ndim == 3:
                if band < 0 or band >= rast.shape[0]:
                    raise IndexError(f"Requested band {band} out of range; raster has {rast.shape[0]} band(s)")
                rast_arr = rast[band]
                # Only convert float64 to float32, leave ints/bools unchanged
                if coerce_f64_to_f32 and rast_arr.dtype == np.float64:
                    rast_arr = rast_arr.astype(np.float32)
            else:
                rast_arr = rast
                # Only convert float64 to float32, leave ints/bools unchanged
                if coerce_f64_to_f32 and rast_arr.dtype == np.float64:
                    rast_arr = rast_arr.astype(np.float32)
    else:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        trf = dataset.GetGeoTransform()
        _assert_north_up(trf)
        # GetProjection returns WKT string (or empty string)
        crs_wkt = dataset.GetProjection() or None
        rb = dataset.GetRasterBand(band + 1)
        if rb is None:
            dataset = None
            raise IndexError(f"Requested band {band} out of range in GDAL dataset")
        rast_arr = rb.ReadAsArray()
        # Only convert float64 to float32, leave ints/bools unchanged
        if coerce_f64_to_f32 and rast_arr.dtype == np.float64:
            rast_arr = rast_arr.astype(np.float32)
        no_data_val = rb.GetNoDataValue()
        if bbox is not None:
            bbox_tuple = _normalise_bbox(bbox)
            snapped_bbox = shrink_bbox_to_pixel_grid(
                bbox_tuple,
                origin_x=trf[0],
                origin_y=trf[3],
                pixel_width=trf[1],
                pixel_height=trf[5],
            )
            bounds = _compute_bounds_from_transform(trf, dataset.RasterXSize, dataset.RasterYSize)
            _validate_bbox_within_bounds(snapped_bbox, bounds)
            min_x, min_y, max_x, max_y = snapped_bbox
            pixel_width = trf[1]
            pixel_height = abs(trf[5])
            xoff = int(round((min_x - trf[0]) / pixel_width))
            yoff = int(round((trf[3] - max_y) / pixel_height))
            xsize = int(round((max_x - min_x) / pixel_width))
            ysize = int(round((max_y - min_y) / pixel_height))
            # guard offsets/sizes
            if xoff < 0 or yoff < 0 or xsize <= 0 or ysize <= 0:
                dataset = None
                raise ValueError("Computed window from bbox is out of raster bounds or invalid")
            rast_arr = rast_arr[yoff : yoff + ysize, xoff : xoff + xsize]
            trf_arr = [min_x, trf[1], 0, max_y, 0, trf[5]]
        else:
            trf_arr = [trf[0], trf[1], 0, trf[3], 0, trf[5]]
        dataset = None  # ensure dataset closed
    # Handle no-data (support NaN)
    if no_data_val is not None and not np.isnan(no_data_val):
        logger.info(f"No-data value is {no_data_val}, replacing with NaN")
        rast_arr[rast_arr == no_data_val] = np.nan
    if rast_arr.size == 0:
        raise ValueError("Raster array is empty after loading/cropping")
    if rast_arr.min() < 0:
        raise ValueError("Raster contains negative values")
    return rast_arr, trf_arr, crs_wkt, no_data_val


def xy_to_lnglat(crs_wkt: str | None, x, y):
    """Convert x, y coordinates to longitude and latitude.

    Accepts scalar or array-like x/y. If crs_wkt is None the inputs are
    assumed already to be lon/lat and are returned unchanged.
    """
    if crs_wkt is None:
        logger.info("No CRS provided, assuming coordinates are already in WGS84 (lon/lat).")
        return x, y

    try:
        if GDAL_ENV is False:
            source_crs = pyproj.CRS(crs_wkt)
            target_crs = pyproj.CRS(4326)  # WGS84
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            lng, lat = transformer.transform(x, y)
        else:
            old_cs = gdal.osr.SpatialReference()
            old_cs.ImportFromWkt(crs_wkt)
            new_cs = gdal.osr.SpatialReference()
            new_cs.ImportFromEPSG(4326)
            transform = gdal.osr.CoordinateTransformation(old_cs, new_cs)
            out = transform.TransformPoint(float(x), float(y))
            lng, lat = out[0], out[1]

        return lng, lat

    except Exception:
        logger.exception("Failed to transform coordinates")
        raise


def create_empty_raster(
    path_str: str | Path,
    rows: int,
    cols: int,
    transform: list[float],
    crs_wkt: str,
    dtype=np.float32,
    nodata: float = -9999,
    bands: int = 1,
):
    """
    Create an empty GeoTIFF file initialized with nodata.
    """
    path = check_path(path_str, make_dir=True)

    if GDAL_ENV is False:
        trf = Affine.from_gdal(*transform)
        crs = None
        if crs_wkt:
            crs = pyproj.CRS(crs_wkt)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=trf,
            nodata=nodata,
        ) as dst:
            pass  # Just create
    else:
        driver = gdal.GetDriverByName("GTiff")
        # Map numpy dtype to GDAL type
        gdal_type = gdal.GDT_Float32  # Default
        if dtype == np.float64:
            gdal_type = gdal.GDT_Float64
        elif dtype == np.int32:
            gdal_type = gdal.GDT_Int32
        elif dtype == np.int16:
            gdal_type = gdal.GDT_Int16
        elif dtype == np.uint8:
            gdal_type = gdal.GDT_Byte

        ds = driver.Create(str(path), cols, rows, bands, gdal_type)
        ds.SetGeoTransform(transform)
        if crs_wkt:
            ds.SetProjection(crs_wkt)
        for b in range(1, bands + 1):
            band = ds.GetRasterBand(b)
            band.SetNoDataValue(nodata)
            band.Fill(nodata)
        ds = None


def write_raster_window(path_str: str | Path, data: np.ndarray, window: tuple[slice, slice], band: int = 1):
    """
    Write a data array to a specific window in an existing raster.
    window is (row_slice, col_slice).
    """
    path = check_path(path_str)
    row_slice, col_slice = window

    if GDAL_ENV is False:
        from rasterio.windows import Window

        with rasterio.open(path, "r+") as dst:
            win = Window(
                col_off=col_slice.start,
                row_off=row_slice.start,
                width=col_slice.stop - col_slice.start,
                height=row_slice.stop - row_slice.start,
            )
            dst.write(data, band, window=win)
    else:
        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            raise OSError(f"Could not open {path} for update")

        xoff = col_slice.start
        yoff = row_slice.start
        xsize = col_slice.stop - col_slice.start
        ysize = row_slice.stop - row_slice.start

        ds.GetRasterBand(band).WriteArray(data, xoff, yoff)
        ds = None
