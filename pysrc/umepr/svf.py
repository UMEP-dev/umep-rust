"""
SVF wrapper for Python - calls full Rust SVF via skyview rust module.
"""

# %%
import os
import zipfile
from pathlib import Path

import numpy as np

from . import common
from .rustalgos import skyview


# %%
def generate_svf(
    dsm_path: str,
    bbox: list[int, int, int, int],
    out_dir: str,
    cdsm_path: str | None = None,
    trans_veg: float = 3,
    trunk_ratio: float = 0.25,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path)

    # Open the DSM file
    dsm_rast, dsm_transf, dsm_crs, _dsm_nd = common.load_raster(dsm_path, bbox)
    dsm_scale = 1 / dsm_transf[1]

    # veg transmissivity as percentage
    if not trans_veg >= 0 and trans_veg <= 100:
        raise ValueError("Vegetation transmissivity should be a number between 0 and 100")
    trans = trans_veg / 100.0

    # CDSM
    rows, cols = dsm_rast.shape
    if cdsm_path is None:
        use_cdsm = False
        cdsm_rast = np.zeros([rows, cols])
    else:
        use_cdsm = True
        cdsm_rast, cdsm_transf, cdsm_crs, _cdsm_nd = common.load_raster(cdsm_path, bbox)
        if not cdsm_crs == dsm_crs:
            raise ValueError("Mismatching CRS for DSM and CDSM.")
        if not dsm_transf == cdsm_transf:
            raise ValueError("Mismatching spatial transform for DSM and CDSM.")

    # CDSM 2
    cdsm_2_rast = cdsm_rast * trunk_ratio  # issue8
    # compute
    # Ensure arrays are float32 before passing to Rust
    dsm_rast_f32 = dsm_rast.astype(np.float32)
    cdsm_rast_f32 = cdsm_rast.astype(np.float32)
    cdsm_2_rast_f32 = cdsm_2_rast.astype(np.float32)
    # 2 = 153 patches
    ret = skyview.calculate_svf(dsm_rast_f32, cdsm_rast_f32, cdsm_2_rast_f32, dsm_scale, use_cdsm, 2)

    # Save the rasters using rasterio
    common.save_raster(out_path_str + "/" + "svf.tif", ret.svf, dsm_transf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfE.tif", ret.svf_east, dsm_transf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfS.tif", ret.svf_south, dsm_transf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfW.tif", ret.svf_west, dsm_transf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfN.tif", ret.svf_north, dsm_transf, dsm_crs)

    # Create or update the ZIP file
    zip_filepath = out_path_str + "/" + "svfs.zip"
    if os.path.isfile(zip_filepath):
        os.remove(zip_filepath)

    with zipfile.ZipFile(zip_filepath, "a") as zippo:
        zippo.write(out_path_str + "/" + "svf.tif", "svf.tif")
        zippo.write(out_path_str + "/" + "svfE.tif", "svfE.tif")
        zippo.write(out_path_str + "/" + "svfS.tif", "svfS.tif")
        zippo.write(out_path_str + "/" + "svfW.tif", "svfW.tif")
        zippo.write(out_path_str + "/" + "svfN.tif", "svfN.tif")

    # Remove the individual TIFF files after zipping
    os.remove(out_path_str + "/" + "svf.tif")
    os.remove(out_path_str + "/" + "svfE.tif")
    os.remove(out_path_str + "/" + "svfS.tif")
    os.remove(out_path_str + "/" + "svfW.tif")
    os.remove(out_path_str + "/" + "svfN.tif")

    if use_cdsm:  # Changed from use_cdsm == 0 to boolean check
        # Save vegetation rasters
        common.save_raster(out_path_str + "/" + "svfveg.tif", ret.svf_veg, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfEveg.tif", ret.svf_veg_east, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfSveg.tif", ret.svf_veg_south, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfWveg.tif", ret.svf_veg_west, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfNveg.tif", ret.svf_veg_north, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfaveg.tif", ret.svf_veg_blocks_bldg_sh, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfEaveg.tif", ret.svf_veg_blocks_bldg_sh_east, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfSaveg.tif", ret.svf_veg_blocks_bldg_sh_south, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfWaveg.tif", ret.svf_veg_blocks_bldg_sh_west, dsm_transf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfNaveg.tif", ret.svf_veg_blocks_bldg_sh_north, dsm_transf, dsm_crs)

        # Add vegetation rasters to the ZIP file
        with zipfile.ZipFile(zip_filepath, "a") as zippo:
            zippo.write(out_path_str + "/" + "svfveg.tif", "svfveg.tif")
            zippo.write(out_path_str + "/" + "svfEveg.tif", "svfEveg.tif")
            zippo.write(out_path_str + "/" + "svfSveg.tif", "svfSveg.tif")
            zippo.write(out_path_str + "/" + "svfWveg.tif", "svfWveg.tif")
            zippo.write(out_path_str + "/" + "svfNveg.tif", "svfNveg.tif")
            zippo.write(out_path_str + "/" + "svfaveg.tif", "svfaveg.tif")
            zippo.write(out_path_str + "/" + "svfEaveg.tif", "svfEaveg.tif")
            zippo.write(out_path_str + "/" + "svfSaveg.tif", "svfSaveg.tif")
            zippo.write(out_path_str + "/" + "svfWaveg.tif", "svfWaveg.tif")
            zippo.write(out_path_str + "/" + "svfNaveg.tif", "svfNaveg.tif")

        # Remove the individual TIFF files after zipping
        os.remove(out_path_str + "/" + "svfveg.tif")
        os.remove(out_path_str + "/" + "svfEveg.tif")
        os.remove(out_path_str + "/" + "svfSveg.tif")
        os.remove(out_path_str + "/" + "svfWveg.tif")
        os.remove(out_path_str + "/" + "svfNveg.tif")
        os.remove(out_path_str + "/" + "svfaveg.tif")
        os.remove(out_path_str + "/" + "svfEaveg.tif")
        os.remove(out_path_str + "/" + "svfSaveg.tif")
        os.remove(out_path_str + "/" + "svfWaveg.tif")
        os.remove(out_path_str + "/" + "svfNaveg.tif")

        # Calculate final total SVF
        svftotal = ret.svf - (1 - ret.svf_veg) * (1 - trans)

    # Save the final svftotal raster
    common.save_raster(out_path_str + "/" + "svf_total.tif", svftotal, dsm_transf, dsm_crs)

    # Save shadow matrices as compressed npz
    shmat = ret.bldg_sh_matrix
    vegshmat = ret.veg_sh_matrix
    vbshvegshmat = ret.veg_blocks_bldg_sh_matrix

    np.savez_compressed(
        out_path_str + "/" + "shadowmats.npz",
        shadowmat=shmat,
        vegshadowmat=vegshmat,
        vbshmat=vbshvegshmat,
    )
