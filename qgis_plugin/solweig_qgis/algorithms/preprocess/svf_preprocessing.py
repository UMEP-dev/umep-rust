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
SVF arrays are saved into the prepared surface directory (svfs.zip,
shadowmats.npz), replacing any existing SVF files. The SOLWEIG
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
        import threading

        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        # QGIS auto-generates a temp path ending in the parameter name when left blank
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = prepared_dir
            feedback.pushInfo(f"SVF output will be saved to prepared surface directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Compute SVF
        feedback.setProgressText("Computing Sky View Factor (this may take a while)...")
        feedback.setProgress(20)

        import numpy as np
        from solweig.rustalgos import skyview

        use_veg = cdsm is not None
        max_height = float(np.nanmax(dsm))
        if use_veg and cdsm is not None:
            max_height = max(max_height, float(np.nanmax(cdsm)))

        cdsm_svf = cdsm.astype(np.float32) if cdsm is not None else np.zeros_like(dsm, dtype=np.float32)
        if tdsm is not None:
            tdsm_svf = tdsm.astype(np.float32)
        elif cdsm is not None:
            tdsm_svf = (cdsm * trunk_ratio).astype(np.float32)
        else:
            tdsm_svf = np.zeros_like(dsm, dtype=np.float32)

        # Use SkyviewRunner for progress polling (Rust releases GIL during computation)
        total_patches = 153  # patch_option=2
        runner = skyview.SkyviewRunner()
        svf_result = None
        svf_error = None

        def _run_svf():
            nonlocal svf_result, svf_error
            try:
                svf_result = runner.calculate_svf(
                    dsm.astype(np.float32),
                    cdsm_svf,
                    tdsm_svf,
                    pixel_size,
                    use_veg,
                    max_height,
                    2,  # patch_option (153 patches)
                    3.0,  # min_sun_elev_deg
                )
            except Exception as e:
                svf_error = e

        svf_thread = threading.Thread(target=_run_svf, daemon=True)
        svf_thread.start()

        # Poll progress while Rust computes (20-90% range)
        while svf_thread.is_alive():
            svf_thread.join(timeout=0.5)
            patches_done = runner.progress()
            pct = 20 + int(70 * patches_done / total_patches)
            feedback.setProgress(min(pct, 89))
            if feedback.isCanceled():
                break

        svf_thread.join()

        if svf_error is not None:
            raise QgsProcessingException(f"SVF computation failed: {svf_error}") from svf_error
        if svf_result is None:
            raise QgsProcessingException("SVF computation was cancelled")

        if feedback.isCanceled():
            return {}

        feedback.setProgress(90)

        # Save SVF as svfs.zip (format expected by PrecomputedData.prepare())
        feedback.setProgressText("Saving SVF arrays...")

        import tempfile
        import zipfile

        svf_files = {
            "svf.tif": np.array(svf_result.svf),
            "svfN.tif": np.array(svf_result.svf_north),
            "svfE.tif": np.array(svf_result.svf_east),
            "svfS.tif": np.array(svf_result.svf_south),
            "svfW.tif": np.array(svf_result.svf_west),
            "svfveg.tif": np.array(svf_result.svf_veg) if use_veg else np.ones_like(np.array(svf_result.svf)),
            "svfNveg.tif": np.array(svf_result.svf_veg_north) if use_veg else np.ones_like(np.array(svf_result.svf)),
            "svfEveg.tif": np.array(svf_result.svf_veg_east) if use_veg else np.ones_like(np.array(svf_result.svf)),
            "svfSveg.tif": np.array(svf_result.svf_veg_south) if use_veg else np.ones_like(np.array(svf_result.svf)),
            "svfWveg.tif": np.array(svf_result.svf_veg_west) if use_veg else np.ones_like(np.array(svf_result.svf)),
            "svfaveg.tif": np.array(svf_result.svf_veg_blocks_bldg_sh)
            if use_veg
            else np.ones_like(np.array(svf_result.svf)),
            "svfNaveg.tif": np.array(svf_result.svf_veg_blocks_bldg_sh_north)
            if use_veg
            else np.ones_like(np.array(svf_result.svf)),
            "svfEaveg.tif": np.array(svf_result.svf_veg_blocks_bldg_sh_east)
            if use_veg
            else np.ones_like(np.array(svf_result.svf)),
            "svfSaveg.tif": np.array(svf_result.svf_veg_blocks_bldg_sh_south)
            if use_veg
            else np.ones_like(np.array(svf_result.svf)),
            "svfWaveg.tif": np.array(svf_result.svf_veg_blocks_bldg_sh_west)
            if use_veg
            else np.ones_like(np.array(svf_result.svf)),
        }

        svf_zip_path = os.path.join(output_dir, "svfs.zip")
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
            with zipfile.ZipFile(svf_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename in svf_files:
                    zf.write(os.path.join(tmpdir, filename), filename)

        feedback.pushInfo(f"Saved SVF arrays: {svf_zip_path}")

        # Save shadow matrices as shadowmats.npz (for anisotropic sky)
        feedback.setProgressText("Saving shadow matrices...")

        shmat = np.array(svf_result.bldg_sh_matrix)
        vegshmat = np.array(svf_result.veg_sh_matrix)
        vbshmat = np.array(svf_result.veg_blocks_bldg_sh_matrix)

        # Convert to uint8 for compact storage (0.0-1.0 → 0-255)
        shmat_u8 = (np.clip(shmat, 0, 1) * 255).astype(np.uint8)
        vegshmat_u8 = (np.clip(vegshmat, 0, 1) * 255).astype(np.uint8)
        vbshmat_u8 = (np.clip(vbshmat, 0, 1) * 255).astype(np.uint8)

        shadow_path = os.path.join(output_dir, "shadowmats.npz")
        np.savez_compressed(
            shadow_path,
            shadowmat=shmat_u8,
            vegshadowmat=vegshmat_u8,
            vbshmat=vbshmat_u8,
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
