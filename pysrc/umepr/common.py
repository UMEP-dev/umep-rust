from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely import geometry


def rasterise_gdf(gdf, geom_col, ht_col, bbox=None, pixel_size: int = 1):
    # Define raster parameters
    if bbox is not None:
        # Unpack bbox values
        minx, miny, maxx, maxy = bbox
    else:
        # Use the total bounds of the GeoDataFrame
        minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    # Create a blank array for the raster
    raster = np.zeros((height, width), dtype=np.float32)
    # Burn geometries into the raster
    shapes = ((geom, value) for geom, value in zip(gdf[geom_col], gdf[ht_col], strict=True))
    raster = rasterize(shapes, out_shape=raster.shape, transform=transform, fill=0, dtype=np.float32)

    return raster, transform


def check_path(path, make_dir=False):
    path = Path(path)
    path = path.absolute()
    if make_dir is False and not path.parent.exists():
        raise OSError(f"Path {path.parent} does not exist.")
    return path


# Helper function to save raster using rasterio
def save_raster(out_path, data, transform, crs):
    """Save raster data using rasterio."""
    out_path = check_path(out_path, make_dir=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def load_raster(
    path: str,
    bbox: list[int] | None = None,
    band: int = 0,
):
    # Open the raster file with rasterio
    path = check_path(path, make_dir=False)
    with rasterio.open(path) as dataset:
        crs = dataset.crs
        dataset_bounds = dataset.bounds
        if bbox is not None:
            bbox_geom = geometry.box(*bbox)
            # Confirm the bbox is within dataset bounds
            if not (
                dataset_bounds.left <= bbox[0] <= dataset_bounds.right
                and dataset_bounds.left <= bbox[2] <= dataset_bounds.right
                and dataset_bounds.bottom <= bbox[1] <= dataset_bounds.top
                and dataset_bounds.bottom <= bbox[3] <= dataset_bounds.top
            ):
                raise ValueError("Bounding box is not fully contained within the raster dataset bounds")
            rast, transf = mask(dataset, [bbox_geom], crop=True)
        else:
            rast = dataset.read()
            transf = dataset.transform
        # Read the specified band and convert to float
        rast = rast[band].astype(float)
        # Handle NoData values
        nd = dataset.nodata
        if nd is not None:
            rast[rast == nd] = 0.0
        # Check for negative values in the raster
        if rast.min() < 0:
            raise ValueError("Raster contains negative values")

    return rast, transf, crs
