"""Utility functions for geometry and namespace conversion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from affine import Affine
    from numpy.typing import NDArray


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

    Args:
        transform: Affine transformation matrix (Affine object or GDAL list)
        shape: Array shape (rows, cols)

    Returns:
        Bounding box as [minx, miny, maxx, maxy]
    """
    from affine import Affine as AffineClass
    from rasterio.transform import array_bounds

    # Convert list to Affine if needed
    if isinstance(transform, list):
        transform = AffineClass.from_gdal(*transform)

    rows, cols = shape
    bounds = array_bounds(rows, cols, transform)
    # array_bounds returns (left, bottom, right, top)
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


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
) -> tuple[NDArray, Affine]:
    """
    Resample array to match target grid specification.

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
    from affine import Affine as AffineClass
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    # Convert list to Affine if needed
    if isinstance(src_transform, list):
        src_transform = AffineClass.from_gdal(*src_transform)

    minx, miny, maxx, maxy = target_bbox

    # Calculate target dimensions
    width = int(np.round((maxx - minx) / target_pixel_size))
    height = int(np.round((maxy - miny) / target_pixel_size))

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
