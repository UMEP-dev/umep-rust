"""
SVF wrapper for Python - calls full Rust SVF via skyview rust module.
"""

# %%
import logging
import os
import threading
import time
import zipfile
from pathlib import Path
from queue import Queue

import numpy as np
from tqdm import tqdm
from umep import class_configs, common

from .rustalgos import skyview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
def generate_svf(
    dsm_path: str,
    bbox: list[int, int, int, int],
    out_dir: str,
    dem_path: str | None = None,
    cdsm_path: str | None = None,
    trans_veg_perc: float = 3,
    trunk_ratio_perc: float = 25,
    amax_local_window_m: int = 100,
    amax_local_perc: float = 99.9,
    min_sun_elev_deg: float | None = None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path)

    # Open the DSM file
    dsm, dsm_trf, dsm_crs, _dsm_nd = common.load_raster(dsm_path, bbox)
    dsm_pix_size = dsm_trf[1]
    dsm_scale = 1 / dsm_pix_size

    dem = None
    if dem_path is not None:
        dem, dem_trf, dem_crs, _dem_nd = common.load_raster(dem_path, bbox)
        assert dem.shape == dsm.shape, "Mismatching raster shapes for DSM and DEM."
        assert np.allclose(dsm_trf, dem_trf), "Mismatching spatial transform for DSM and DEM."
        assert dem_crs == dsm_crs, "Mismatching CRS for DSM and DEM."

    use_cdsm = False
    cdsm = None
    if cdsm_path is not None:
        use_cdsm = True
        cdsm, cdsm_trf, cdsm_crs, _cdsm_nd = common.load_raster(cdsm_path, bbox)
        assert cdsm.shape == dsm.shape, "Mismatching raster shapes for DSM and CDSM."
        assert np.allclose(dsm_trf, cdsm_trf), "Mismatching spatial transform for DSM and CDSM."
        assert cdsm_crs == dsm_crs, "Mismatching CRS for DSM and CDSM."

    # veg transmissivity as percentage
    if not (0 <= trans_veg_perc <= 100):
        raise ValueError("Vegetation transmissivity should be a number between 0 and 100")

    trans_veg = trans_veg_perc / 100.0
    trunk_ratio = trunk_ratio_perc / 100.0

    dsm, dem, cdsm, tdsm, amax = class_configs.raster_preprocessing(
        dsm,
        dem,
        cdsm,
        None,
        trunk_ratio,
        dsm_pix_size,
        amax_local_window_m=amax_local_window_m,
        amax_local_perc=amax_local_perc,
    )

    # Run SVF in background and poll progress via SkyviewRunner.progress()
    # 2 = 153 patches
    runner = skyview.SkyviewRunner()
    result_queue: Queue = Queue()

    def _runner_thread(q: Queue):
        try:
            res = runner.calculate_svf(
                dsm.astype(np.float32),
                cdsm.astype(np.float32),
                tdsm.astype(np.float32),
                dsm_scale,
                use_cdsm,
                amax,
                2,
                min_sun_elev_deg,
            )
            q.put(res)
        except Exception as e:
            q.put(e)

    thread = threading.Thread(target=_runner_thread, args=(result_queue,))
    thread.start()

    # show progress bar for 153 patches (patch option 2)
    total_patches = 153
    pbar = tqdm(total=total_patches)
    try:
        while thread.is_alive():
            time.sleep(1)
            try:
                pbar.n = runner.progress()
                pbar.refresh()
            except Exception:
                pass
        # finish
        pbar.n = total_patches
        pbar.refresh()
    finally:
        pbar.close()

    ret = result_queue.get()
    thread.join()
    if isinstance(ret, Exception):
        raise ret

    # Save the rasters using rasterio
    common.save_raster(out_path_str + "/" + "svf.tif", ret.svf, dsm_trf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfE.tif", ret.svf_east, dsm_trf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfS.tif", ret.svf_south, dsm_trf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfW.tif", ret.svf_west, dsm_trf, dsm_crs)
    common.save_raster(out_path_str + "/" + "svfN.tif", ret.svf_north, dsm_trf, dsm_crs)

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
        common.save_raster(out_path_str + "/" + "svfveg.tif", ret.svf_veg, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfEveg.tif", ret.svf_veg_east, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfSveg.tif", ret.svf_veg_south, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfWveg.tif", ret.svf_veg_west, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfNveg.tif", ret.svf_veg_north, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfaveg.tif", ret.svf_veg_blocks_bldg_sh, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfEaveg.tif", ret.svf_veg_blocks_bldg_sh_east, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfSaveg.tif", ret.svf_veg_blocks_bldg_sh_south, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfWaveg.tif", ret.svf_veg_blocks_bldg_sh_west, dsm_trf, dsm_crs)
        common.save_raster(out_path_str + "/" + "svfNaveg.tif", ret.svf_veg_blocks_bldg_sh_north, dsm_trf, dsm_crs)

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
        svftotal = ret.svf - (1 - ret.svf_veg) * (1 - trans_veg)

    # Save the final svftotal raster
    common.save_raster(out_path_str + "/" + "svf_total.tif", svftotal, dsm_trf, dsm_crs)

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
