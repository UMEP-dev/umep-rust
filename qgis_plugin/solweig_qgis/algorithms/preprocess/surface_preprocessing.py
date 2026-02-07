"""
Surface Data Preprocessing Algorithm

Aligns rasters, computes valid mask, crops, computes walls and SVF,
and saves cleaned GeoTIFFs to a prepared surface directory. This
directory can then be loaded directly by the SOLWEIG Calculation
algorithm, avoiding repeated preprocessing and making intermediate
results transparent.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from ...utils.converters import _align_layer, _load_optional_raster, load_raster_from_layer
from ...utils.parameters import add_surface_parameters
from ..base import SolweigAlgorithmBase


class SurfacePreprocessingAlgorithm(SolweigAlgorithmBase):
    """
    Prepare surface data for SOLWEIG calculation.

    Loads all surface rasters, aligns them to a common grid, computes
    a unified valid mask, crops to the valid bounding box, computes
    wall heights and aspects, and saves cleaned GeoTIFFs.

    The output directory can be loaded directly by the SOLWEIG Calculation
    algorithm, skipping all preprocessing steps.
    """

    def name(self) -> str:
        return "surface_preprocessing"

    def displayName(self) -> str:
        return self.tr("2. Prepare Surface Data (align, walls, SVF)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Prepare surface data for SOLWEIG calculation.

Aligns all rasters, computes walls and Sky View Factor, and saves
everything needed to run SOLWEIG Calculation directly.

<b>What this does:</b>
<ol>
<li>Loads all surface rasters (DSM, CDSM, DEM, TDSM, Land cover)</li>
<li>Aligns all rasters to a common grid (intersection of extents)</li>
<li>Converts vegetation heights from relative to absolute (if needed)</li>
<li>Computes a unified valid mask (removes NaN borders)</li>
<li>Crops all rasters to the valid bounding box</li>
<li><b>Computes wall heights and wall aspects</b> from the DSM</li>
<li><b>Computes Sky View Factor (SVF)</b> and shadow matrices</li>
<li>Saves all cleaned rasters as GeoTIFFs</li>
</ol>

<b>Outputs:</b>
<pre>
  output_dir/
    dsm.tif
    wall_height.tif
    wall_aspect.tif
    svfs.zip          (Sky View Factor arrays)
    shadowmats.npz    (shadow matrices for anisotropic sky)
    cdsm.tif         (if CDSM provided)
    dem.tif           (if DEM provided)
    tdsm.tif          (if TDSM provided)
    land_cover.tif    (if land cover provided)
    metadata.json     (pixel size, CRS, etc.)
</pre>

<b>Next step:</b>
Run "SOLWEIG Calculation" with the prepared surface directory."""
        )

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # Surface inputs (DSM, CDSM, DEM, TDSM, Land cover, RELATIVE_HEIGHTS)
        add_surface_parameters(self)

        # Processing extent (optional)
        self.addParameter(
            QgsProcessingParameterExtent(
                "EXTENT",
                self.tr("Processing extent (leave empty to use intersection of inputs)"),
                optional=True,
            )
        )

        # Output pixel size (optional — coarser than native for faster processing)
        pixel_size_param = QgsProcessingParameterNumber(
            "PIXEL_SIZE",
            self.tr("Output pixel size (m) — leave 0 to use native DSM resolution"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            minValue=0.0,
            maxValue=100.0,
            optional=True,
        )
        self.addParameter(pixel_size_param)

        # Wall limit (advanced)
        wall_limit = QgsProcessingParameterNumber(
            "WALL_LIMIT",
            self.tr("Minimum wall height (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
            minValue=0.0,
            maxValue=10.0,
        )
        from qgis.core import QgsProcessingParameterDefinition

        wall_limit.setFlags(wall_limit.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(wall_limit)

        # Output directory
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                "OUTPUT_DIR",
                self.tr("Output directory for prepared surface"),
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "SURFACE_DIR",
                self.tr("Prepared surface directory"),
            )
        )
        self.addOutput(
            QgsProcessingOutputNumber(
                "COMPUTATION_TIME",
                self.tr("Computation time (seconds)"),
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the algorithm."""
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG Surface Preprocessing")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()
        from solweig.utils import extract_bounds, intersect_bounds

        # Step 1: Load DSM
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        dsm_layer = self.parameterAsRasterLayer(parameters, "DSM", context)
        if dsm_layer is None:
            raise QgsProcessingException("DSM layer is required")

        dsm, dsm_gt, crs_wkt = load_raster_from_layer(dsm_layer)
        native_pixel_size = abs(dsm_gt[1])
        feedback.pushInfo(f"DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")
        feedback.pushInfo(f"Native pixel size: {native_pixel_size:.2f} m")

        # Resolve output pixel size
        requested_pixel_size = self.parameterAsDouble(parameters, "PIXEL_SIZE", context)
        if requested_pixel_size > 0:
            if requested_pixel_size < native_pixel_size - 1e-6:
                raise QgsProcessingException(
                    f"Requested pixel size ({requested_pixel_size:.2f} m) is finer than the DSM "
                    f"native resolution ({native_pixel_size:.2f} m). Upsampling creates false "
                    f"precision. Use a value >= {native_pixel_size:.2f} or leave at 0 for native."
                )
            pixel_size = requested_pixel_size
            if abs(pixel_size - native_pixel_size) > 1e-6:
                feedback.pushInfo(f"Resampling all rasters from {native_pixel_size:.2f} m to {pixel_size:.2f} m")
        else:
            pixel_size = native_pixel_size

        feedback.pushInfo(f"Output pixel size: {pixel_size:.2f} m")

        # Load optional rasters
        cdsm, cdsm_gt = _load_optional_raster(parameters, "CDSM", context, self)
        if cdsm is not None:
            feedback.pushInfo("Loaded CDSM (vegetation)")

        dem, dem_gt = _load_optional_raster(parameters, "DEM", context, self)
        if dem is not None:
            feedback.pushInfo("Loaded DEM (ground elevation)")

        tdsm, tdsm_gt = _load_optional_raster(parameters, "TDSM", context, self)
        if tdsm is not None:
            feedback.pushInfo("Loaded TDSM (trunk zone)")

        lc_arr, lc_gt = _load_optional_raster(parameters, "LAND_COVER", context, self)
        land_cover = lc_arr.astype(np.uint8) if lc_arr is not None else None
        if land_cover is not None:
            feedback.pushInfo("Loaded land cover classification")

        if feedback.isCanceled():
            return {}

        # Step 2: Compute extent intersection
        feedback.setProgressText("Aligning rasters...")
        feedback.setProgress(15)

        bounds_list = [extract_bounds(dsm_gt, dsm.shape)]
        for arr, gt in [(cdsm, cdsm_gt), (dem, dem_gt), (tdsm, tdsm_gt), (lc_arr, lc_gt)]:
            if arr is not None and gt is not None:
                bounds_list.append(extract_bounds(gt, arr.shape))

        extent_rect = self.parameterAsExtent(parameters, "EXTENT", context)
        if not extent_rect.isNull():
            target_bbox = [
                extent_rect.xMinimum(),
                extent_rect.yMinimum(),
                extent_rect.xMaximum(),
                extent_rect.yMaximum(),
            ]
            feedback.pushInfo(f"Using custom extent: {target_bbox}")
        elif len(bounds_list) > 1:
            target_bbox = intersect_bounds(bounds_list)
            feedback.pushInfo(f"Auto-computed intersection extent: {target_bbox}")
        else:
            target_bbox = bounds_list[0]

        # Align all layers
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

        aligned_gt = [target_bbox[0], pixel_size, 0, target_bbox[3], 0, -pixel_size]
        feedback.pushInfo(f"Aligned grid: {dsm.shape[1]}x{dsm.shape[0]} pixels")

        if feedback.isCanceled():
            return {}

        # Step 3: Create SurfaceData, preprocess, mask, crop
        feedback.setProgressText("Computing valid mask and cropping...")
        feedback.setProgress(25)

        relative_heights = self.parameterAsBool(parameters, "RELATIVE_HEIGHTS", context)

        surface = solweig.SurfaceData(
            dsm=dsm,
            cdsm=cdsm,
            dem=dem,
            tdsm=tdsm,
            land_cover=land_cover,
            pixel_size=pixel_size,
            relative_heights=relative_heights,
        )
        surface._geotransform = aligned_gt
        surface._crs_wkt = crs_wkt

        # Convert relative heights to absolute if needed
        if relative_heights and (cdsm is not None or tdsm is not None):
            feedback.pushInfo("Converting relative vegetation heights to absolute...")
            surface.preprocess()

        # Compute valid mask, apply, and crop (inline to avoid version dependency)
        valid = np.isfinite(surface.dsm)
        for arr in [surface.cdsm, surface.dem, surface.tdsm]:
            if arr is not None:
                valid &= np.isfinite(arr)
        if surface.land_cover is not None:
            valid &= surface.land_cover != 255  # 255 = nodata in UMEP land cover

        # Apply mask: set NaN wherever any layer is invalid
        surface.dsm = np.where(valid, surface.dsm, np.nan)
        if surface.cdsm is not None:
            surface.cdsm = np.where(valid, surface.cdsm, np.nan)
        if surface.dem is not None:
            surface.dem = np.where(valid, surface.dem, np.nan)
        if surface.tdsm is not None:
            surface.tdsm = np.where(valid, surface.tdsm, np.nan)

        # Crop to valid bounding box
        rows_any = np.any(valid, axis=1)
        cols_any = np.any(valid, axis=0)
        if np.any(rows_any) and np.any(cols_any):
            r0, r1 = int(np.argmax(rows_any)), int(valid.shape[0] - np.argmax(rows_any[::-1]))
            c0, c1 = int(np.argmax(cols_any)), int(valid.shape[1] - np.argmax(cols_any[::-1]))

            if r0 > 0 or r1 < valid.shape[0] or c0 > 0 or c1 < valid.shape[1]:
                surface.dsm = surface.dsm[r0:r1, c0:c1].copy()
                if surface.cdsm is not None:
                    surface.cdsm = surface.cdsm[r0:r1, c0:c1].copy()
                if surface.dem is not None:
                    surface.dem = surface.dem[r0:r1, c0:c1].copy()
                if surface.tdsm is not None:
                    surface.tdsm = surface.tdsm[r0:r1, c0:c1].copy()
                if surface.land_cover is not None:
                    surface.land_cover = surface.land_cover[r0:r1, c0:c1].copy()

                # Update geotransform for the crop offset
                gt = aligned_gt
                aligned_gt = [
                    gt[0] + c0 * gt[1],
                    gt[1],
                    gt[2],
                    gt[3] + r0 * gt[5],
                    gt[4],
                    gt[5],
                ]
                surface._geotransform = aligned_gt
                feedback.pushInfo(f"Cropped {r0}:{r1}, {c0}:{c1} from original grid")

        feedback.pushInfo(f"After NaN masking + crop: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")

        if feedback.isCanceled():
            return {}

        # Create output directory early so we can write incrementally
        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)
        os.makedirs(output_dir, exist_ok=True)
        gt = surface._geotransform or aligned_gt
        crs = surface._crs_wkt or crs_wkt

        # Save aligned/cropped surface rasters immediately
        feedback.setProgressText("Saving aligned surface rasters...")
        self.save_georeferenced_output(surface.dsm, os.path.join(output_dir, "dsm.tif"), gt, crs)
        feedback.pushInfo("Saved dsm.tif")

        if surface.cdsm is not None:
            self.save_georeferenced_output(surface.cdsm, os.path.join(output_dir, "cdsm.tif"), gt, crs)
            feedback.pushInfo("Saved cdsm.tif")

        if surface.dem is not None:
            self.save_georeferenced_output(surface.dem, os.path.join(output_dir, "dem.tif"), gt, crs)
            feedback.pushInfo("Saved dem.tif")

        if surface.tdsm is not None:
            self.save_georeferenced_output(surface.tdsm, os.path.join(output_dir, "tdsm.tif"), gt, crs)
            feedback.pushInfo("Saved tdsm.tif")

        if surface.land_cover is not None:
            self.save_georeferenced_output(
                surface.land_cover.astype(np.float32),
                os.path.join(output_dir, "land_cover.tif"),
                gt,
                crs,
            )
            feedback.pushInfo("Saved land_cover.tif")

        # Step 4: Compute walls and save immediately
        feedback.setProgressText("Computing wall heights...")
        feedback.setProgress(25)

        from solweig.physics import wallalgorithms as wa

        wall_limit = self.parameterAsDouble(parameters, "WALL_LIMIT", context)
        feedback.pushInfo(f"Computing walls (min height: {wall_limit:.1f} m)...")

        walls = wa.findwalls(surface.dsm, wall_limit)
        feedback.pushInfo("Wall heights computed")

        feedback.setProgressText("Computing wall aspects...")
        feedback.setProgress(30)

        dsm_scale = 1.0 / pixel_size
        dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, dsm_scale, surface.dsm, feedback=feedback)
        feedback.pushInfo("Wall aspects computed")

        surface.wall_height = walls
        surface.wall_aspect = dirwalls

        # Save walls immediately
        self.save_georeferenced_output(walls, os.path.join(output_dir, "wall_height.tif"), gt, crs)
        feedback.pushInfo("Saved wall_height.tif")
        self.save_georeferenced_output(dirwalls, os.path.join(output_dir, "wall_aspect.tif"), gt, crs)
        feedback.pushInfo("Saved wall_aspect.tif")

        if feedback.isCanceled():
            return {}

        # Step 5: Compute Sky View Factor
        # Uses the same Python API as SurfaceData.prepare() — automatically
        # tiles large grids to stay within GPU buffer limits.
        feedback.setProgressText("Computing Sky View Factor (this may take a while)...")
        feedback.setProgress(35)

        from pathlib import Path

        from solweig.models.surface import SurfaceData as SD

        use_veg = surface.cdsm is not None
        dsm_f32 = surface.dsm.astype(np.float32)

        aligned_rasters = {
            "dsm_arr": dsm_f32,
            "cdsm_arr": surface.cdsm.astype(np.float32) if use_veg else None,
            "tdsm_arr": (
                surface.tdsm.astype(np.float32)
                if surface.tdsm is not None
                else (surface.cdsm * 0.25).astype(np.float32)
                if use_veg
                else None
            ),
            "pixel_size": pixel_size,
            "dsm_transform": gt,
            "dsm_crs": crs,
        }

        rows, cols = dsm_f32.shape
        if rows * cols > 6_700_000:
            feedback.pushInfo(f"Large grid ({rows}x{cols} = {rows * cols:,} px) — using tiled GPU computation")

        try:
            SD._compute_and_cache_svf(
                surface,
                aligned_rasters,
                Path(output_dir),
                trunk_ratio=0.25,
                feedback=feedback,
            )
        except Exception as e:
            raise QgsProcessingException(f"SVF computation failed: {e}") from e

        feedback.pushInfo("Sky View Factor computed")
        feedback.setProgress(75)

        if feedback.isCanceled():
            return {}

        # Save SVF outputs (extract from surface object populated by _compute_and_cache_svf)
        feedback.setProgressText("Saving SVF and shadow matrices...")
        feedback.setProgress(80)

        svf_data = surface.svf
        svf_files = {
            "svf.tif": svf_data.svf,
            "svfN.tif": svf_data.svf_north,
            "svfE.tif": svf_data.svf_east,
            "svfS.tif": svf_data.svf_south,
            "svfW.tif": svf_data.svf_west,
            "svfveg.tif": svf_data.svf_veg,
            "svfNveg.tif": svf_data.svf_veg_north,
            "svfEveg.tif": svf_data.svf_veg_east,
            "svfSveg.tif": svf_data.svf_veg_south,
            "svfWveg.tif": svf_data.svf_veg_west,
            "svfaveg.tif": svf_data.svf_aveg,
            "svfNaveg.tif": svf_data.svf_aveg_north,
            "svfEaveg.tif": svf_data.svf_aveg_east,
            "svfSaveg.tif": svf_data.svf_aveg_south,
            "svfWaveg.tif": svf_data.svf_aveg_west,
        }

        svf_zip_path = os.path.join(output_dir, "svfs.zip")
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, arr in svf_files.items():
                self.save_georeferenced_output(arr, os.path.join(tmpdir, filename), gt, crs)
            with zipfile.ZipFile(svf_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename in svf_files:
                    zf.write(os.path.join(tmpdir, filename), filename)
        feedback.pushInfo("Saved svfs.zip")

        # Save shadow matrices
        sm = surface.shadow_matrices
        shmat_u8 = np.array(sm._shmat_u8)
        vegshmat_u8 = np.array(sm._vegshmat_u8)
        vbshmat_u8 = np.array(sm._vbshmat_u8)
        shadow_path = os.path.join(output_dir, "shadowmats.npz")
        np.savez_compressed(
            shadow_path,
            shadowmat=shmat_u8,
            vegshadowmat=vegshmat_u8,
            vbshmat=vbshmat_u8,
            patch_count=np.array(sm.patch_count),
        )
        feedback.pushInfo("Saved shadowmats.npz")

        # Save metadata last (acts as a completion marker)
        metadata = {
            "pixel_size": pixel_size,
            "geotransform": list(gt),
            "crs_wkt": crs,
            "shape": list(surface.dsm.shape),
            "relative_heights": False,  # Always absolute after preprocessing
            "has_cdsm": surface.cdsm is not None,
            "has_dem": surface.dem is not None,
            "has_tdsm": surface.tdsm is not None,
            "has_land_cover": surface.land_cover is not None,
            "has_walls": True,
            "has_svf": True,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        feedback.pushInfo("Saved metadata.json")

        feedback.setProgress(95)

        # Report summary
        computation_time = time.time() - start_time
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Surface preprocessing complete!")
        feedback.pushInfo(f"  Grid size: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")
        feedback.pushInfo(f"  Pixel size: {pixel_size:.2f} m")
        feedback.pushInfo("  Walls computed: yes")
        feedback.pushInfo("  SVF computed: yes")
        feedback.pushInfo(f"  Computation time: {computation_time:.1f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("=" * 60)

        feedback.setProgress(100)

        return {
            "SURFACE_DIR": output_dir,
            "COMPUTATION_TIME": computation_time,
        }
