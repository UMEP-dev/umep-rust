"""
SVF Preprocessing Algorithm

Pre-computes Sky View Factor (SVF) arrays for reuse across timesteps.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from ...utils.converters import load_prepared_surface
from ..base import SolweigAlgorithmBase


class SvfPreprocessingAlgorithm(SolweigAlgorithmBase):
    """
    Pre-compute Sky View Factor (SVF) arrays.

    SVF computation is expensive (~30-120s for 1000x1000 grid).
    Pre-computing allows reuse across multiple timesteps, providing
    significant speedup (from 60s to 0.3s per timestep).
    """

    TRUNK_RATIO = "TRUNK_RATIO"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "svf_preprocessing"

    def displayName(self) -> str:
        return self.tr("3. Recompute Sky View Factor (advanced)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Recompute Sky View Factor (SVF) with custom parameters.

<b>Note:</b> SVF is already computed during "Prepare Surface Data" (step 2).
Use this tool only if you need to recompute SVF with different parameters
(e.g., different trunk ratio) without re-running the full surface preparation.

<b>Input:</b>
Provide the <b>prepared surface directory</b> from "Prepare Surface Data".
DSM, CDSM, and TDSM are loaded automatically.

<b>Output:</b>
SVF arrays are saved into the prepared surface directory (`svfs.zip`).
Shadow matrices are saved as `shadowmats.npz` when manageable, or kept
as `svf/&lt;pixel&gt;/shadow_memmaps/` for very large grids. The SOLWEIG
Calculation algorithm loads these automatically.

<b>Typical runtime:</b>
- 1000x1000 grid: 30-120 seconds"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterFile(
                "PREPARED_SURFACE_DIR",
                self.tr("Prepared surface directory (from 'Prepare Surface Data')"),
                behavior=QgsProcessingParameterFile.Folder,
            )
        )

        trunk_ratio = QgsProcessingParameterNumber(
            self.TRUNK_RATIO,
            self.tr("Trunk ratio (fraction of canopy height, used when no TDSM provided)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.25,
            minValue=0.0,
            maxValue=1.0,
        )
        trunk_ratio.setFlags(trunk_ratio.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(trunk_ratio)

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output directory for SVF arrays (defaults to prepared surface directory)"),
                optional=True,
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "SVF_DIR",
                self.tr("SVF output directory"),
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
        import time

        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG SVF Preprocessing")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        self.import_solweig()

        # Load surface data from prepared directory
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        prepared_dir = self.parameterAsFile(parameters, "PREPARED_SURFACE_DIR", context)
        surface = load_prepared_surface(prepared_dir, feedback)
        dsm = surface.dsm
        cdsm = surface.cdsm
        tdsm = surface.tdsm
        geotransform = surface._geotransform
        crs_wkt = surface._crs_wkt
        pixel_size = surface.pixel_size

        feedback.pushInfo(f"DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")
        feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

        if feedback.isCanceled():
            return {}

        trunk_ratio = self.parameterAsDouble(parameters, self.TRUNK_RATIO, context)

        import os
        import shutil
        import tempfile
        import zipfile

        import numpy as np

        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        # QGIS auto-generates a temp path ending in the parameter name when left blank
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = prepared_dir
            feedback.pushInfo(f"SVF output will be saved to prepared surface directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Compute SVF using the same Python API as SurfaceData.prepare() —
        # automatically tiles large grids to stay within GPU buffer limits.
        feedback.setProgressText("Computing Sky View Factor...")
        feedback.setProgress(20)

        from pathlib import Path

        from solweig.models.surface import SurfaceData as SD

        use_veg = cdsm is not None
        dsm_f32 = dsm.astype(np.float32)

        aligned_rasters = {
            "dsm_arr": dsm_f32,
            "cdsm_arr": cdsm.astype(np.float32) if use_veg else None,
            "tdsm_arr": (
                tdsm.astype(np.float32)
                if tdsm is not None
                else (cdsm * trunk_ratio).astype(np.float32)
                if use_veg
                else None
            ),
            "pixel_size": pixel_size,
            "dsm_transform": geotransform,
            "dsm_crs": crs_wkt,
        }

        rows, cols = dsm_f32.shape
        from solweig.tiling import compute_max_tile_pixels

        _max_px = compute_max_tile_pixels(context="svf")
        n_pixels = rows * cols
        if n_pixels > _max_px:
            feedback.pushInfo(
                f"Large grid ({rows}x{cols} = {n_pixels:,} px, limit {_max_px:,}) — using tiled computation"
            )
        else:
            feedback.pushInfo(f"Grid {rows}x{cols} = {n_pixels:,} px — single-pass computation")

        try:
            SD._compute_and_cache_svf(
                surface,
                aligned_rasters,
                Path(output_dir),
                trunk_ratio=trunk_ratio,
                feedback=feedback,
                progress_range=(20.0, 90.0),
            )
        except Exception as e:
            raise QgsProcessingException(f"SVF computation failed: {e}") from e
        if surface.svf is None:
            raise QgsProcessingException(
                "SVF computation completed without producing SVF arrays "
                "(surface.svf is None). Check that the active solweig build "
                "matches the current plugin code."
            )

        if feedback.isCanceled():
            return {}

        feedback.setProgress(90)

        # Reuse exports created by SurfaceData._compute_and_cache_svf when available
        # to avoid re-running CPU-heavy serialization/compression in this QGIS step.
        from solweig.cache import pixel_size_tag

        cache_dir = Path(output_dir) / "svf" / pixel_size_tag(pixel_size)
        cache_zip = cache_dir / "svfs.zip"
        cache_shadow = cache_dir / "shadowmats.npz"
        cache_shadow_memmaps = cache_dir / "shadow_memmaps"
        svf_zip_path = Path(output_dir) / "svfs.zip"
        shadow_path = Path(output_dir) / "shadowmats.npz"

        copied_zip = False
        if cache_zip.exists():
            feedback.setProgressText("Finalizing SVF arrays...")
            shutil.copy2(cache_zip, svf_zip_path)
            feedback.pushInfo(f"Copied SVF arrays from cache: {svf_zip_path}")
            copied_zip = True
        else:
            feedback.pushInfo("SVF cache zip not found; generating svfs.zip from in-memory arrays")

        if not copied_zip:
            # Save SVF as svfs.zip (format expected by PrecomputedData.prepare())
            feedback.setProgressText("Saving SVF arrays...")

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

            with tempfile.TemporaryDirectory() as tmpdir:
                for filename, arr in svf_files.items():
                    tif_path = os.path.join(tmpdir, filename)
                    self.save_georeferenced_output(
                        array=arr,
                        output_path=tif_path,
                        geotransform=geotransform,
                        crs_wkt=crs_wkt,
                        feedback=feedback,
                    )
                with zipfile.ZipFile(str(svf_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                    for filename in svf_files:
                        zf.write(os.path.join(tmpdir, filename), filename)

            feedback.pushInfo(f"Saved SVF arrays: {svf_zip_path}")

        copied_shadow = False
        if cache_shadow.exists():
            feedback.setProgressText("Finalizing shadow matrices...")
            shutil.copy2(cache_shadow, shadow_path)
            feedback.pushInfo(f"Copied shadow matrices from cache: {shadow_path}")
            copied_shadow = True
        elif cache_shadow_memmaps.exists() and (cache_shadow_memmaps / "metadata.json").exists():
            feedback.pushInfo(
                f"Using shadow memmap cache (skipping shadowmats.npz export for large grid): {cache_shadow_memmaps}"
            )
            copied_shadow = True
        else:
            feedback.pushInfo("Shadow cache not found; generating shadowmats.npz from in-memory arrays")

        if not copied_shadow:
            # Save shadow matrices as shadowmats.npz (for anisotropic sky)
            feedback.setProgressText("Saving shadow matrices...")

            sm = surface.shadow_matrices
            shmat_u8 = np.array(sm._shmat_u8)
            vegshmat_u8 = np.array(sm._vegshmat_u8)
            vbshmat_u8 = np.array(sm._vbshmat_u8)

            np.savez_compressed(
                str(shadow_path),
                shadowmat=shmat_u8,
                vegshadowmat=vegshmat_u8,
                vbshmat=vbshmat_u8,
                patch_count=np.array(sm.patch_count),
            )

            feedback.pushInfo(f"Saved shadow matrices: {shadow_path}")

        computation_time = time.time() - start_time
        feedback.setProgress(100)

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SVF preprocessing complete!")
        feedback.pushInfo(f"  Computation time: {computation_time:.1f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("")
        feedback.pushInfo(
            "Use this directory as 'Pre-computed SVF directory' in calculation algorithms for ~200x speedup."
        )
        feedback.pushInfo("=" * 60)

        return {
            "SVF_DIR": output_dir,
            "COMPUTATION_TIME": computation_time,
        }
