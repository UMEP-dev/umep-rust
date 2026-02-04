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
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from ..base import SolweigAlgorithmBase


class SvfPreprocessingAlgorithm(SolweigAlgorithmBase):
    """
    Pre-compute Sky View Factor (SVF) arrays.

    SVF computation is expensive (~30-120s for 1000x1000 grid).
    Pre-computing allows reuse across multiple timesteps, providing
    significant speedup (from 60s to 0.3s per timestep).
    """

    # Parameter names
    DSM = "DSM"
    CDSM = "CDSM"
    DEM = "DEM"
    TDSM = "TDSM"
    TRANS_VEG = "TRANS_VEG"
    RELATIVE_HEIGHTS = "RELATIVE_HEIGHTS"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "svf_preprocessing"

    def displayName(self) -> str:
        return self.tr("Compute Sky View Factor")

    def shortHelpString(self) -> str:
        return self.tr(
            """Pre-compute Sky View Factor (SVF) arrays for SOLWEIG calculations.

<b>Purpose:</b>
SVF computation is computationally expensive. Pre-computing SVF allows
reuse across multiple timesteps, providing significant speedup.

<b>Inputs:</b>
- DSM (required): Digital Surface Model
- CDSM (optional): Canopy/vegetation heights
- DEM (optional): Ground elevation
- TDSM (optional): Trunk zone heights
- Vegetation transmissivity: Default 0.03

<b>Output:</b>
A directory containing SVF arrays (svf.tif, svf_north.tif, etc.)
that can be used by calculation algorithms.

<b>Typical runtime:</b>
- 1000x1000 grid: 30-120 seconds
- Subsequent calculations with cached SVF: ~0.3 seconds"""
        )

    def group(self) -> str:
        return self.tr("SOLWEIG > Preprocessing")

    def groupId(self) -> str:
        return "solweig_preprocessing"

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DSM,
                self.tr("Digital Surface Model (DSM)"),
                optional=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.CDSM,
                self.tr("Canopy DSM (vegetation heights)"),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM,
                self.tr("Digital Elevation Model (ground)"),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.TDSM,
                self.tr("Trunk zone DSM"),
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.TRANS_VEG,
                self.tr("Vegetation transmissivity"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.03,
                minValue=0.0,
                maxValue=1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RELATIVE_HEIGHTS,
                self.tr("CDSM/TDSM are relative heights (above ground)"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output directory for SVF arrays"),
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
        solweig = self.import_solweig()

        # Load DSM
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        dsm_layer = self.parameterAsRasterLayer(parameters, self.DSM, context)
        dsm, geotransform, crs_wkt = self.load_raster_from_layer(dsm_layer)
        pixel_size = abs(geotransform[1])

        feedback.pushInfo(f"DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")
        feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

        # Load optional rasters
        cdsm = self.load_optional_raster(parameters, self.CDSM, context)
        dem = self.load_optional_raster(parameters, self.DEM, context)
        tdsm = self.load_optional_raster(parameters, self.TDSM, context)

        if cdsm is not None:
            feedback.pushInfo("Loaded CDSM (vegetation)")
        if dem is not None:
            feedback.pushInfo("Loaded DEM (ground)")
        if tdsm is not None:
            feedback.pushInfo("Loaded TDSM (trunk zone)")

        if feedback.isCanceled():
            return {}

        # Get parameters
        trans_veg = self.parameterAsDouble(parameters, self.TRANS_VEG, context)
        relative_heights = self.parameterAsBool(parameters, self.RELATIVE_HEIGHTS, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Create SurfaceData
        feedback.setProgressText("Preparing surface data...")
        feedback.setProgress(15)

        surface = solweig.SurfaceData(
            dsm=dsm,
            cdsm=cdsm,
            dem=dem,
            tdsm=tdsm,
            pixel_size=pixel_size,
            relative_heights=relative_heights,
        )

        # Store geospatial metadata
        surface._geotransform = geotransform
        surface._crs_wkt = crs_wkt

        # Preprocess heights if needed
        if relative_heights and (cdsm is not None or tdsm is not None):
            feedback.pushInfo("Converting relative heights to absolute...")
            surface.preprocess()

        if feedback.isCanceled():
            return {}

        # Compute SVF
        feedback.setProgressText("Computing Sky View Factor (this may take a while)...")
        feedback.setProgress(20)

        try:
            # Use prepare() which computes walls and SVF
            surface.prepare(
                working_dir=output_dir,
                trans_veg=trans_veg,
            )
        except Exception as e:
            raise QgsProcessingException(f"SVF computation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        feedback.setProgress(90)

        # Save SVF arrays as GeoTIFFs for inspection
        feedback.setProgressText("Saving SVF arrays...")

        import os

        if surface.svf is not None:
            svf_arrays = {
                "svf": surface.svf.svf,
                "svf_north": surface.svf.svf_north,
                "svf_east": surface.svf.svf_east,
                "svf_south": surface.svf.svf_south,
                "svf_west": surface.svf.svf_west,
            }

            for name, arr in svf_arrays.items():
                if arr is not None:
                    out_path = os.path.join(output_dir, f"{name}.tif")
                    self.save_georeferenced_output(
                        array=arr,
                        output_path=out_path,
                        geotransform=geotransform,
                        crs_wkt=crs_wkt,
                        feedback=feedback,
                    )

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
