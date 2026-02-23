"""Utility functions for geometry and namespace conversion."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._compat import GDAL_AVAILABLE, RASTERIO_AVAILABLE

if TYPE_CHECKING:
    from affine import Affine
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

if RASTERIO_AVAILABLE:
    from rasterio.transform import array_bounds, from_bounds  # noqa: F401
    from rasterio.warp import Resampling, reproject  # noqa: F401
elif GDAL_AVAILABLE:
    from osgeo import gdal, gdalconst  # noqa: F401


# =============================================================================
# Namespace Conversion (for JSON parameter loading)
# =============================================================================


def dict_to_namespace(d: dict[str, Any] | list | Any) -> SimpleNamespace | list | Any:
    """
    Recursively convert dicts to SimpleNamespace.

    This matches the runner's dict_to_namespace function for loading JSON parameters.

    Args:
        d: Dictionary, list, or scalar value to convert

    Returns:
        SimpleNamespace for dicts, list of converted items for lists, or original value for scalars
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def namespace_to_dict(ns: SimpleNamespace | Any) -> dict | list | Any:
    """
    Recursively convert SimpleNamespace to dict for JSON serialization.

    Inverse of dict_to_namespace.

    Args:
        ns: SimpleNamespace, list, or scalar value to convert

    Returns:
        Dict for SimpleNamespace, list of converted items for lists, or original value for scalars
    """
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(i) for i in ns]
    else:
        return ns


# =============================================================================
# Geometric Utilities (for raster operations)
# =============================================================================


def extract_bounds(transform: list[float] | Affine, shape: tuple[int, ...]) -> list[float]:
    """
    Extract bounding box [minx, miny, maxx, maxy] from affine transform and array shape.

    Works with either rasterio or GDAL backend.

    Args:
        transform: Affine transformation matrix (Affine object or GDAL list)
        shape: Array shape (rows, cols)

    Returns:
        Bounding box as [minx, miny, maxx, maxy]
    """
    rows, cols = shape

    if RASTERIO_AVAILABLE:
        from affine import Affine as AffineClass
        from rasterio.transform import array_bounds

        # Convert list to Affine if needed
        if isinstance(transform, list):
            transform = AffineClass.from_gdal(*transform)

        bounds = array_bounds(rows, cols, transform)
        # array_bounds returns (left, bottom, right, top)
        return [bounds[0], bounds[1], bounds[2], bounds[3]]

    elif GDAL_AVAILABLE:
        # GDAL geotransform: [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        # Convert Affine to GDAL list if needed (Affine has .to_gdal() method)
        gt = transform if isinstance(transform, list) else list(transform.to_gdal())

        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = gt

        # Calculate bounds
        minx = x_origin
        maxx = x_origin + cols * x_pixel_size
        maxy = y_origin  # y_origin is typically top-left (north)
        miny = y_origin + rows * y_pixel_size  # y_pixel_size is typically negative

        # Ensure correct order (miny < maxy)
        if miny > maxy:
            miny, maxy = maxy, miny

        return [minx, miny, maxx, maxy]

    else:
        raise ImportError(
            "Neither rasterio nor GDAL available. Install rasterio (pip install rasterio) "
            "or run in OSGeo4W/QGIS environment."
        )


def intersect_bounds(bounds_list: list[list[float]]) -> list[float]:
    """
    Compute intersection of multiple bounding boxes.

    Args:
        bounds_list: List of bounding boxes, each as [minx, miny, maxx, maxy]

    Returns:
        Intersection bounding box as [minx, miny, maxx, maxy]

    Raises:
        ValueError: If bounding boxes don't intersect
    """
    if not bounds_list:
        raise ValueError("No bounding boxes provided")

    # Start with first bounds
    minx = bounds_list[0][0]
    miny = bounds_list[0][1]
    maxx = bounds_list[0][2]
    maxy = bounds_list[0][3]

    # Compute intersection with remaining bounds
    for bounds in bounds_list[1:]:
        minx = max(minx, bounds[0])
        miny = max(miny, bounds[1])
        maxx = min(maxx, bounds[2])
        maxy = min(maxy, bounds[3])

    # Check if intersection is valid
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Bounding boxes don't intersect: intersection would be [{minx}, {miny}, {maxx}, {maxy}]")

    return [minx, miny, maxx, maxy]


def resample_to_grid(
    array: NDArray,
    src_transform: list[float] | Affine,
    target_bbox: list[float],
    target_pixel_size: float,
    method: str = "bilinear",
    src_crs: str | None = None,
) -> tuple[NDArray, list[float] | Affine]:
    """
    Resample array to match target grid specification.

    Works with either rasterio or GDAL backend.

    Args:
        array: Source array to resample
        src_transform: Source affine transformation (Affine object or GDAL list)
        target_bbox: Target bounding box [minx, miny, maxx, maxy]
        target_pixel_size: Target pixel size in map units
        method: Resampling method ("bilinear" or "nearest")
        src_crs: Source CRS (WKT string), required for rasterio reproject

    Returns:
        Tuple of (resampled_array, target_transform as Affine)
    """
    minx, miny, maxx, maxy = target_bbox

    # Calculate target dimensions
    width = int(np.round((maxx - minx) / target_pixel_size))
    height = int(np.round((maxy - miny) / target_pixel_size))

    if RASTERIO_AVAILABLE:
        from affine import Affine as AffineClass
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        # Convert list to Affine if needed
        if isinstance(src_transform, list):
            src_transform = AffineClass.from_gdal(*src_transform)

        # Create target transform
        target_transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Create destination array
        destination = np.zeros((height, width), dtype=array.dtype)

        # Select resampling method
        resampling_method = Resampling.nearest if method == "nearest" else Resampling.bilinear

        # Reproject (same CRS, just resampling)
        reproject(
            source=array,
            destination=destination,
            src_transform=src_transform,
            dst_transform=target_transform,
            src_crs=src_crs,  # Pass through CRS for rasterio
            dst_crs=src_crs,  # Same CRS (no reprojection, just resampling)
            resampling=resampling_method,
        )

        return destination, target_transform

    elif GDAL_AVAILABLE:
        from osgeo import gdal, gdalconst

        # Convert Affine to GDAL geotransform if needed (Affine has .to_gdal() method)
        src_gt = src_transform if isinstance(src_transform, list) else list(src_transform.to_gdal())

        # Create target geotransform (top-left origin, positive x, negative y)
        target_gt = [minx, target_pixel_size, 0, maxy, 0, -target_pixel_size]

        # Map numpy dtype to GDAL type
        dtype_map = {
            np.float32: gdalconst.GDT_Float32,
            np.float64: gdalconst.GDT_Float64,
            np.int32: gdalconst.GDT_Int32,
            np.int16: gdalconst.GDT_Int16,
            np.uint8: gdalconst.GDT_Byte,
            np.uint16: gdalconst.GDT_UInt16,
            np.uint32: gdalconst.GDT_UInt32,
        }
        gdal_dtype = dtype_map.get(array.dtype.type, gdalconst.GDT_Float32)

        # Select resampling method
        resample_alg = gdalconst.GRA_NearestNeighbour if method == "nearest" else gdalconst.GRA_Bilinear

        # Create in-memory source dataset
        src_rows, src_cols = array.shape
        mem_driver = gdal.GetDriverByName("MEM")
        src_ds = mem_driver.Create("", src_cols, src_rows, 1, gdal_dtype)
        src_ds.SetGeoTransform(src_gt)
        if src_crs:
            src_ds.SetProjection(src_crs)
        src_ds.GetRasterBand(1).WriteArray(array)

        # Create in-memory destination dataset
        dst_ds = mem_driver.Create("", width, height, 1, gdal_dtype)
        dst_ds.SetGeoTransform(target_gt)
        if src_crs:
            dst_ds.SetProjection(src_crs)

        # Perform resampling
        gdal.ReprojectImage(src_ds, dst_ds, src_crs, src_crs, resample_alg)

        # Read result
        destination = dst_ds.GetRasterBand(1).ReadAsArray()

        # Clean up
        src_ds = None
        dst_ds = None

        return destination, target_gt

    else:
        raise ImportError(
            "Neither rasterio nor GDAL available. Install rasterio (pip install rasterio) "
            "or run in OSGeo4W/QGIS environment."
        )
