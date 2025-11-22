"""
SVF wrapper for Python - calls full Rust SVF via skyview rust module.
"""

# %%
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from umep import class_configs, common
from umep.tile_manager import TileManager

from .rustalgos import skyview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
def generate_svf(
    dsm_path: str,
    bbox: list[int],
    out_dir: str,
    dem_path: str | None = None,
    cdsm_path: str | None = None,
    trans_veg_perc: float = 3,
    trunk_ratio_perc: float = 25,
    amax_local_window_m: int = 100,
    amax_local_perc: float = 99.9,
    use_tiled_loading: bool = False,
    tile_size: int = 1000,
    save_shadowmats: bool = True,
):
    """
    Generate Sky View Factor outputs.

    Args:
        save_shadowmats: Save shadow matrices (required for SOLWEIG anisotropic sky).
                        Saved as uint8 (75% smaller than float32). Set to False only
                        if you don't need SOLWEIG's anisotropic modeling.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path)

    # Open the DSM file to get metadata
    # If tiled, we only load metadata first
    if use_tiled_loading:
        dsm_meta = common.get_raster_metadata(dsm_path)
        dsm_trf = dsm_meta["transform"]
        dsm_crs = dsm_meta["crs"]
        dsm_nd = dsm_meta["nodata"]
        rows = dsm_meta["rows"]
        cols = dsm_meta["cols"]

        # Handle rasterio vs GDAL transform
        if "res" in dsm_meta:
            # Convert Rasterio Affine to GDAL transform
            # Affine: (a, b, c, d, e, f) -> GDAL: (c, a, b, f, d, e)
            t = dsm_trf
            dsm_trf = [t.c, t.a, t.b, t.f, t.d, t.e]

        dsm_pix_size = dsm_trf[1]
        dsm_scale = 1 / dsm_pix_size

        # Calculate conservative global amax for buffer estimation
        # Load sample data to estimate terrain complexity
        sample_dsm, _, _, _ = common.load_raster(dsm_path, bbox=None, coerce_f64_to_f32=True)

        if dem_path is None:
            # Without DEM, use DSM range as conservative estimate
            global_amax = float(np.nanmax(sample_dsm) - np.nanmin(sample_dsm))
        else:
            # With DEM, estimate from height differences
            sample_dem, _, _, _ = common.load_raster(dem_path, bbox=None, coerce_f64_to_f32=True)
            height_diff = sample_dsm - sample_dem
            global_amax = float(np.nanpercentile(height_diff[~np.isnan(height_diff)], 99.9))
            del sample_dem

        # Add safety margin and cap at reasonable maximum
        global_amax = min(global_amax * 1.2, 200.0)  # 20% safety margin, max 200m
        del sample_dsm

        logger.info(f"Estimated global amax: {global_amax:.1f}m for buffer calculation")

        # Initialize TileManager with calculated buffer
        tile_manager = TileManager(rows, cols, tile_size, dsm_pix_size, buffer_dist=global_amax)

        if len(tile_manager.tiles) == 0:
            raise ValueError(f"TileManager generated 0 tiles for {rows}x{cols} raster with tile_size={tile_size}")

        logger.info(f"Initialized TileManager with {len(tile_manager.tiles)} tiles.")

        # Initialize output rasters
        # We need to create empty rasters for all outputs
        output_files = ["input-dsm.tif", "svf.tif", "svfE.tif", "svfS.tif", "svfW.tif", "svfN.tif"]
        if dem_path:
            output_files.append("input-dem.tif")
        if cdsm_path:
            output_files.extend(
                [
                    "input-cdsm.tif",
                    "input-tdsm.tif",
                    "svfveg.tif",
                    "svfEveg.tif",
                    "svfSveg.tif",
                    "svfWveg.tif",
                    "svfNveg.tif",
                    "svfaveg.tif",
                    "svfEaveg.tif",
                    "svfSaveg.tif",
                    "svfWaveg.tif",
                    "svfNaveg.tif",
                    "svf_total.tif",
                ]
            )

        for fname in output_files:
            common.create_empty_raster(out_path_str + "/" + fname, rows, cols, dsm_trf, dsm_crs, nodata=-9999.0)

        # Initialize memory-mapped arrays for shadow matrices (only if needed)
        if save_shadowmats:
            # 153 patches is standard for this algorithm
            patches = 153
            shmat_shape = (rows, cols, patches)

            # Create temp file paths
            temp_dir = tempfile.mkdtemp(dir=out_path_str)
            shmat_path = os.path.join(temp_dir, "shmat.dat")
            vegshmat_path = os.path.join(temp_dir, "vegshmat.dat")
            vbshvegshmat_path = os.path.join(temp_dir, "vbshvegshmat.dat")

            # Use uint8 instead of float32 for 75% space savings (shadow mats are binary 0/1)
            # Calculate memory requirements
            memmap_size_mb = (shmat_shape[0] * shmat_shape[1] * shmat_shape[2] * 1) / (1024 * 1024)
            logger.info(f"Creating memory-mapped arrays: {memmap_size_mb * 3:.1f} MB total (3 arrays, uint8)")

            # Create memmapped arrays with error handling
            try:
                shmat_mem = np.memmap(shmat_path, dtype="uint8", mode="w+", shape=shmat_shape)
                vegshmat_mem = np.memmap(vegshmat_path, dtype="uint8", mode="w+", shape=shmat_shape)
                vbshvegshmat_mem = np.memmap(vbshvegshmat_path, dtype="uint8", mode="w+", shape=shmat_shape)
            except OSError as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise OSError(
                    f"Failed to create memory-mapped arrays ({memmap_size_mb * 3:.1f} MB). "
                    f"Check disk space and permissions in {out_path_str}. Error: {e}"
                ) from e

        trans_veg = trans_veg_perc / 100.0
        trunk_ratio = trunk_ratio_perc / 100.0

        # Iterate over tiles
        for i, tile in enumerate(tile_manager.get_tiles()):
            logger.info(f"Processing tile {i + 1}/{len(tile_manager.tiles)}")

            # Load inputs for tile (with overlap)
            dsm_tile = common.read_raster_window(dsm_path, tile.full_slice, band=1)

            dem_tile = None
            if dem_path:
                dem_tile = common.read_raster_window(dem_path, tile.full_slice, band=1)

            cdsm_tile = None
            if cdsm_path:
                cdsm_tile = common.read_raster_window(cdsm_path, tile.full_slice, band=1)

            # Preprocess
            dsm_tile, dem_tile, cdsm_tile, tdsm_tile, amax = class_configs.raster_preprocessing(
                dsm_tile,
                dem_tile,
                cdsm_tile,
                None,
                trunk_ratio,
                dsm_pix_size,
                amax_local_window_m=amax_local_window_m,
                amax_local_perc=amax_local_perc,
                quiet=True,
            )

            # Compute SVF using Rust skyview module
            use_cdsm_bool = cdsm_path is not None
            runner = skyview.SkyviewRunner()
            ret = runner.calculate_svf(
                dsm_tile.astype(np.float32),
                cdsm_tile.astype(np.float32) if cdsm_tile is not None else None,
                tdsm_tile.astype(np.float32) if tdsm_tile is not None else None,
                dsm_scale,
                use_cdsm_bool,
                amax,
                2,  # 153 patches
                5.0,  # min_sun_elev_deg
            )

            # Write outputs (core only)
            core_slice = tile.core_slice()
            write_win = tile.write_window.to_slices()

            # Helper to write core - bind loop vars with default args
            def write_core(fname, data, cs=core_slice, ww=write_win):
                core_data = data[cs]
                common.write_raster_window(out_path_str + "/" + fname, core_data, ww)

            write_core("input-dsm.tif", dsm_tile)
            if dem_tile is not None:
                write_core("input-dem.tif", dem_tile)
            if cdsm_tile is not None:
                write_core("input-cdsm.tif", cdsm_tile)
                write_core("input-tdsm.tif", tdsm_tile)

            write_core("svf.tif", ret.svf)
            write_core("svfE.tif", ret.svf_east)
            write_core("svfS.tif", ret.svf_south)
            write_core("svfW.tif", ret.svf_west)
            write_core("svfN.tif", ret.svf_north)

            if use_cdsm_bool:
                write_core("svfveg.tif", ret.svf_veg)
                write_core("svfEveg.tif", ret.svf_veg_east)
                write_core("svfSveg.tif", ret.svf_veg_south)
                write_core("svfWveg.tif", ret.svf_veg_west)
                write_core("svfNveg.tif", ret.svf_veg_north)
                write_core("svfaveg.tif", ret.svf_veg_blocks_bldg_sh)
                write_core("svfEaveg.tif", ret.svf_veg_blocks_bldg_sh_east)
                write_core("svfSaveg.tif", ret.svf_veg_blocks_bldg_sh_south)
                write_core("svfWaveg.tif", ret.svf_veg_blocks_bldg_sh_west)
                write_core("svfNaveg.tif", ret.svf_veg_blocks_bldg_sh_north)

                # Calculate total SVF
                svftotal_tile = ret.svf - (1 - ret.svf_veg) * (1 - trans_veg)
                write_core("svf_total.tif", svftotal_tile)

            # Write shadow matrices to memmap (if saving)
            if save_shadowmats:
                # Extract core for 3D arrays - use core_slice with added dimension
                core_slice_3d = core_slice + (slice(None),)

                # Destination slice in memmap - use write_window directly
                write_slice_3d = tile.write_window.to_slices() + (slice(None),)

                # Convert to uint8 (shadow matrices are binary 0/1)
                shmat_mem[write_slice_3d] = (ret.bldg_sh_matrix[core_slice_3d] * 255).astype(np.uint8)
                vegshmat_mem[write_slice_3d] = (ret.veg_sh_matrix[core_slice_3d] * 255).astype(np.uint8)
                vbshvegshmat_mem[write_slice_3d] = (ret.veg_blocks_bldg_sh_matrix[core_slice_3d] * 255).astype(np.uint8)

                # Flush memmaps periodically?
                if i % 10 == 0:
                    shmat_mem.flush()
                    vegshmat_mem.flush()
                    vbshvegshmat_mem.flush()

        # Save shadow matrices (if requested)
        if save_shadowmats:
            # Flush final
            shmat_mem.flush()
            vegshmat_mem.flush()
            vbshvegshmat_mem.flush()

            # Save shadow matrices as compressed npz (uint8 format)
            # We read from the memmapped files
            logger.info("Saving shadow matrices to npz (uint8 format, 75% smaller)...")
            np.savez_compressed(
                out_path_str + "/" + "shadowmats.npz",
                shadowmat=shmat_mem,
                vegshadowmat=vegshmat_mem,
                vbshmat=vbshvegshmat_mem,
                dtype="uint8",  # Store metadata about dtype
            )

            # Cleanup temp
            del shmat_mem
            del vegshmat_mem
            del vbshvegshmat_mem
            shutil.rmtree(temp_dir)
        else:
            logger.info("Skipping shadow matrix save (not needed for this workflow)")

        # Zip SVF files (same as standard)
        zip_filepath = out_path_str + "/" + "svfs.zip"
        if os.path.isfile(zip_filepath):
            os.remove(zip_filepath)

        with zipfile.ZipFile(zip_filepath, "a") as zippo:
            zippo.write(out_path_str + "/" + "svf.tif", "svf.tif")
            zippo.write(out_path_str + "/" + "svfE.tif", "svfE.tif")
            zippo.write(out_path_str + "/" + "svfS.tif", "svfS.tif")
            zippo.write(out_path_str + "/" + "svfW.tif", "svfW.tif")
            zippo.write(out_path_str + "/" + "svfN.tif", "svfN.tif")

            if cdsm_path:
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

        # Remove individual files
        files_to_remove = ["svf.tif", "svfE.tif", "svfS.tif", "svfW.tif", "svfN.tif"]
        if cdsm_path:
            files_to_remove.extend(
                [
                    "svfveg.tif",
                    "svfEveg.tif",
                    "svfSveg.tif",
                    "svfWveg.tif",
                    "svfNveg.tif",
                    "svfaveg.tif",
                    "svfEaveg.tif",
                    "svfSaveg.tif",
                    "svfWaveg.tif",
                    "svfNaveg.tif",
                ]
            )

        for f in files_to_remove:
            try:
                os.remove(out_path_str + "/" + f)
            except OSError as e:
                logger.warning(f"Could not remove temporary file {f}: {e}")

        return

    # Standard execution (non-tiled)
    # Open the DSM file
    dsm, dsm_trf, dsm_crs, dsm_nd = common.load_raster(dsm_path, bbox, coerce_f64_to_f32=True)
    dsm_pix_size = dsm_trf[1]
    dsm_scale = 1 / dsm_pix_size

    dem = None
    if dem_path is not None:
        dem, dem_trf, dem_crs, _dem_nd = common.load_raster(dem_path, bbox, coerce_f64_to_f32=True)
        assert dem.shape == dsm.shape, "Mismatching raster shapes for DSM and DEM."
        assert np.allclose(dsm_trf, dem_trf), "Mismatching spatial transform for DSM and DEM."
        assert dem_crs == dsm_crs, "Mismatching CRS for DSM and DEM."

    use_cdsm = False
    cdsm = None
    if cdsm_path is not None:
        use_cdsm = True
        cdsm, cdsm_trf, cdsm_crs, _cdsm_nd = common.load_raster(cdsm_path, bbox, coerce_f64_to_f32=True)
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

    common.save_raster(
        out_path_str + "/input-dsm.tif",
        dsm,
        dsm_trf,
        dsm_crs,
        dsm_nd,
        coerce_f64_to_f32=True,
    )
    if dem is not None:
        common.save_raster(
            out_path_str + "/input-dem.tif",
            dem,
            dsm_trf,
            dsm_crs,
            dsm_nd,
            coerce_f64_to_f32=True,
        )
    if use_cdsm:
        common.save_raster(
            out_path_str + "/input-cdsm.tif",
            cdsm,
            dsm_trf,
            dsm_crs,
            dsm_nd,
            coerce_f64_to_f32=True,
        )
        common.save_raster(
            out_path_str + "/input-tdsm.tif",
            tdsm,
            dsm_trf,
            dsm_crs,
            dsm_nd,
            coerce_f64_to_f32=True,
        )

    # compute using Rust skyview module
    runner = skyview.SkyviewRunner()
    ret = runner.calculate_svf(
        dsm.astype(np.float32),
        cdsm.astype(np.float32) if cdsm is not None else None,
        tdsm.astype(np.float32) if tdsm is not None else None,
        dsm_scale,
        use_cdsm,
        amax,
        2,  # 153 patches
        5.0,  # min_sun_elev_deg
    )

    svfbu = ret.svf
    svfbuE = ret.svf_east
    svfbuS = ret.svf_south
    svfbuW = ret.svf_west
    svfbuN = ret.svf_north

    # Save the rasters using rasterio
    common.save_raster(out_path_str + "/" + "svf.tif", svfbu, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
    common.save_raster(out_path_str + "/" + "svfE.tif", svfbuE, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
    common.save_raster(out_path_str + "/" + "svfS.tif", svfbuS, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
    common.save_raster(out_path_str + "/" + "svfW.tif", svfbuW, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
    common.save_raster(out_path_str + "/" + "svfN.tif", svfbuN, dsm_trf, dsm_crs, coerce_f64_to_f32=True)

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

    if use_cdsm == 0:
        svftotal = svfbu
    else:
        # Report the vegetation-related results
        svfveg = ret.svf_veg
        svfEveg = ret.svf_veg_east
        svfSveg = ret.svf_veg_south
        svfWveg = ret.svf_veg_west
        svfNveg = ret.svf_veg_north
        svfaveg = ret.svf_veg_blocks_bldg_sh
        svfEaveg = ret.svf_veg_blocks_bldg_sh_east
        svfSaveg = ret.svf_veg_blocks_bldg_sh_south
        svfWaveg = ret.svf_veg_blocks_bldg_sh_west
        svfNaveg = ret.svf_veg_blocks_bldg_sh_north

        # Save vegetation rasters
        common.save_raster(out_path_str + "/" + "svfveg.tif", svfveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfEveg.tif", svfEveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfSveg.tif", svfSveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfWveg.tif", svfWveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfNveg.tif", svfNveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfaveg.tif", svfaveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfEaveg.tif", svfEaveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfSaveg.tif", svfSaveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfWaveg.tif", svfWaveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)
        common.save_raster(out_path_str + "/" + "svfNaveg.tif", svfNaveg, dsm_trf, dsm_crs, coerce_f64_to_f32=True)

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
        svftotal = svfbu - (1 - svfveg) * (1 - trans_veg)

    # Save the final svftotal raster
    common.save_raster(out_path_str + "/" + "svf_total.tif", svftotal, dsm_trf, dsm_crs, coerce_f64_to_f32=True)

    # Save shadow matrices as compressed npz (only if requested)
    if save_shadowmats:
        shmat = ret.bldg_sh_matrix
        vegshmat = ret.veg_sh_matrix
        vbshvegshmat = ret.veg_blocks_bldg_sh_matrix

        # Convert to uint8 for 75% space savings (shadow matrices are binary 0/1)
        logger.info("Saving shadow matrices to npz (uint8 format, 75% smaller)...")
        np.savez_compressed(
            out_path_str + "/" + "shadowmats.npz",
            shadowmat=(shmat * 255).astype(np.uint8),
            vegshadowmat=(vegshmat * 255).astype(np.uint8),
            vbshmat=(vbshvegshmat * 255).astype(np.uint8),
            dtype="uint8",  # Store metadata about dtype
        )
    else:
        logger.info("Skipping shadow matrix save (not needed for this workflow)")
