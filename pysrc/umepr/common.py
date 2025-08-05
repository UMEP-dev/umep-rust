from pathlib import Path

import numpy as np

try:
    import pyproj
    import rasterio
    from rasterio.features import rasterize
    from rasterio.mask import mask
    from rasterio.transform import Affine, from_origin
    from shapely import geometry

    GDAL_ENV = False

except:
    from osgeo import gdal

    GDAL_ENV = True


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


def check_path(path_str: str | Path, make_dir: bool = False) -> Path:
    # Ensure path exists
    path = Path(path_str).absolute()
    if not path.exists() and not path.suffix:
        if make_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise OSError(f"Path {path} does not exist.")
    return path


def save_raster(
    out_path_str: str, data_arr: np.ndarray, trf_arr: list[float], crs_wkt: str, no_data_val: float = -9999
):
    # Save raster using GDAL or rasterio
    out_path = check_path(out_path_str, make_dir=True)
    height, width = data_arr.shape
    if GDAL_ENV is False:
        trf = Affine.from_gdal(*trf_arr)
        crs = pyproj.CRS.from_wkt(crs_wkt)
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
        # trf is a list: [top left x, w-e pixel size, rotation, top left y, rotation, n-s pixel size]
        ds.SetGeoTransform(trf_arr)
        ds.SetProjection(crs_wkt)
        ds.GetRasterBand(1).WriteArray(data_arr, 0, 0)
        ds.FlushCache()
        ds.SetNoDataValue(no_data_val)
        ds = None


def load_raster(
    path_str: str, bbox: list[int] | None = None, band: int = 0
) -> tuple[np.ndarray, list[float], str, int]:
    # Load raster, optionally crop to bbox
    path = check_path(path_str, make_dir=False)
    if not path.exists():
        raise FileNotFoundError(f"Raster file {path} does not exist.")
    if GDAL_ENV is False:
        with rasterio.open(path) as dataset:
            crs_wkt = dataset.crs.to_wkt() if dataset.crs is not None else None
            dataset_bounds = dataset.bounds
            no_data_val = dataset.nodata
            if bbox is not None:
                # Create bbox geometry for masking
                bbox_geom = geometry.box(*bbox)
                if not (
                    dataset_bounds.left <= bbox[0] <= dataset_bounds.right
                    and dataset_bounds.left <= bbox[2] <= dataset_bounds.right
                    and dataset_bounds.bottom <= bbox[1] <= dataset_bounds.top
                    and dataset_bounds.bottom <= bbox[3] <= dataset_bounds.top
                ):
                    raise ValueError("Bounding box is not fully contained within the raster dataset bounds")
                rast, trf = mask(dataset, [bbox_geom], crop=True)
            else:
                rast = dataset.read()
                trf = dataset.transform
            # Convert rasterio Affine to GDAL-style list
            trf_arr = [trf.c, trf.a, trf.b, trf.f, trf.d, trf.e]
            rast_arr = rast[band].astype(float)
    else:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        trf = dataset.GetGeoTransform()
        crs_wkt = dataset.GetProjection().ExportToWkt()
        rast_arr = dataset.GetRasterBand(band + 1).ReadAsArray().astype(float)
        no_data_val = dataset.GetRasterBand(band + 1).GetNoDataValue()
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            xoff = int((min_x - trf[0]) / trf[1])
            yoff = int((trf[3] - max_y) / abs(trf[5]))
            xsize = int((max_x - min_x) / trf[1])
            ysize = int((max_y - min_y) / abs(trf[5]))
            rast_arr = rast_arr[yoff : yoff + ysize, xoff : xoff + xsize]
            trf_arr = [min_x, trf[1], 0, max_y, 0, trf[5]]
        else:
            trf_arr = [trf[0], trf[1], 0, trf[3], 0, trf[5]]
    if no_data_val is not None:
        rast_arr[rast_arr == no_data_val] = 0.0
    if rast_arr.min() < 0:
        raise ValueError("Raster contains negative values")
    return rast_arr, trf_arr, crs_wkt, no_data_val


def xy_to_lnglat(crs_wkt, x, y):
    """Convert x, y coordinates to longitude and latitude."""
    if GDAL_ENV is False:
        # Define the source and target CRS
        source_crs = pyproj.CRS(crs_wkt)
        target_crs = pyproj.CRS(4326)  # WGS 84
        # Create a transformer object
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
        # Perform the transformation
        lng, lat = transformer.transform(x, y)
    else:
        # Define the source CRS from WKT
        old_cs = gdal.osr.SpatialReference()
        old_cs.ImportFromWkt(crs_wkt)
        # Define WGS 84 CRS
        new_cs = gdal.osr.SpatialReference()
        new_cs.ImportFromWkt("""
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]""")
        # Create a coordinate transformation
        transform = gdal.osr.CoordinateTransformation(old_cs, new_cs)
        lonlat = transform.TransformPoint(x, y)
        # Handle GDAL version differences
        gdalver = float(gdal.__version__[0])
        if gdalver >= 3.0:
            lng = lonlat[1]  # changed to gdal 3
            lat = lonlat[0]  # changed to gdal 3
        else:
            lng = lonlat[0]  # changed to gdal 2
            lat = lonlat[1]  # changed to gdal 2

    return lng, lat
