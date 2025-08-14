"""
Shadow wrapper for Python.

daily_shading import internally calls the Rust implementation for shadow calculations.
"""

import datetime
from pathlib import Path

import pyproj
from rasterio.transform import Affine, xy

from umep import common
from .functions import daily_shading as dsh


def generate_shadows(
    dsm_path: str,
    shadow_date_Ymd: str,  # %Y-%m-%d"
    wall_ht_path: str,
    wall_aspect_path: str,
    bbox: list[int, int, int, int],
    out_dir: str,
    shadow_time_HM: int | None = None,  # "%H:%M"
    time_interval_M=30,
    veg_dsm_path: str | None = None,
    trans_veg: float = 3,
    trunk_zone_ht_perc: float = 0.25,
):
    dsm, dsm_transf, dsm_crs, _dsm_nd = common.load_raster(dsm_path, bbox)
    dsm_height, dsm_width = dsm.shape  # y rows by x cols
    dsm_scale = 1 / dsm_transf[1]
    # y is flipped - so return max for lower row
    minx, miny = xy(Affine.from_gdal(*dsm_transf), dsm.shape[0], 0)
    # Define the source and target CRS
    source_crs = pyproj.CRS(dsm_crs)
    target_crs = pyproj.CRS(4326)  # WGS 84
    # Create a transformer object
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    # Perform the transformation
    lon, lat = transformer.transform(minx, miny)

    # veg transmissivity as percentage
    if not trans_veg >= 0 and trans_veg <= 100:
        raise ValueError("Vegetation transmissivity should be a number between 0 and 100")
    trans = trans_veg / 100.0

    if veg_dsm_path is not None:
        usevegdem = 1
        veg_dsm, veg_dsm_transf, veg_dsm_crs, _veg_dsm_nd = common.load_raster(veg_dsm_path, bbox)
        veg_dsm_height, veg_dsm_width = veg_dsm.shape
        if not (veg_dsm_width == dsm_width) & (veg_dsm_height == dsm_height):
            raise ValueError("Error in Vegetation Canopy DSM: All rasters must be of same extent and resolution")
        trunkratio = trunk_zone_ht_perc / 100.0
        veg_dsm_2 = veg_dsm * trunkratio
        veg_dsm_2_height, veg_dsm_2_width = veg_dsm_2.shape
        if not (veg_dsm_2_width == dsm_width) & (veg_dsm_2_height == dsm_height):
            raise ValueError("Error in Trunk Zone DSM: All rasters must be of same extent and resolution")
    else:
        usevegdem = 0
        veg_dsm = 0
        veg_dsm_2 = 0

    if wall_aspect_path and wall_ht_path:
        print("Facade shadow scheme activated")
        wallsh = 1
        wh_rast, wh_transf, wh_crs, _wh_nd = common.load_raster(wall_ht_path, bbox)
        wh_height, wh_width = wh_rast.shape
        if not (wh_width == dsm_width) & (wh_height == dsm_height):
            raise ValueError("Error in Wall height raster: All rasters must be of same extent and resolution")
        wa_rast, wa_transf, wa_crs, _wa_nd = common.load_raster(wall_aspect_path, bbox)
        wa_height, wa_width = wa_rast.shape
        if not (wa_width == dsm_width) & (wa_height == dsm_height):
            raise ValueError("Error in Wall aspect raster: All rasters must be of same extent and resolution")
    else:
        wallsh = 0
        wh_rast = 0
        wa_height = 0

    dst = 0
    UTC = 0
    target_date = datetime.datetime.strptime(shadow_date_Ymd, "%Y-%m-%d").date()
    year = target_date.year
    month = target_date.month
    day = target_date.day
    if shadow_time_HM is not None:
        onetime = 1
        onetimetime = datetime.datetime.strptime(shadow_time_HM, "%H:%M")
        hour = onetimetime.hour
        minu = onetimetime.minute
        sec = onetimetime.second
    else:
        onetime = 0
        hour = 0
        minu = 0
        sec = 0

    tv = [year, month, day, hour, minu, sec]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path)

    Path.mkdir(out_path / "facade_shdw_bldgs", parents=True, exist_ok=True)
    Path.mkdir(out_path / "facade_shdw_veg", parents=True, exist_ok=True)
    Path.mkdir(out_path / "shadow_ground", parents=True, exist_ok=True)

    shadowresult = dsh.daily_shading(
        dsm.astype("float32"),
        veg_dsm.astype("float32"),
        veg_dsm_2.astype("float32"),
        dsm_scale,
        lon,
        lat,
        dsm_width,
        dsm_height,
        tv,
        UTC,
        usevegdem,
        time_interval_M,
        onetime,
        out_path_str,
        dsm_transf,
        dsm_crs,
        trans,
        dst,
        wallsh,
        wh_rast.astype("float32"),
        wa_rast.astype("float32"),
    )

    shfinal = shadowresult["shfinal"]
    common.save_raster(out_path_str + "/shadow_composite.tif", shfinal, dsm_transf, dsm_crs)
