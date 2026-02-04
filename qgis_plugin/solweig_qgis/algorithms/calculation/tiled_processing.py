"""
Tiled Processing SOLWEIG Algorithm

Memory-efficient calculation for large rasters using tiled processing.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
)

from ...utils.converters import (
    create_human_params_from_parameters,
    create_location_from_parameters,
    create_surface_from_parameters,
    create_weather_from_parameters,
)
from ...utils.parameters import (
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_surface_parameters,
    add_weather_parameters,
)
from ..base import SolweigAlgorithmBase


class TiledProcessingAlgorithm(SolweigAlgorithmBase):
    """
    Calculate Tmrt using tiled processing for large rasters.

    Divides the input raster into overlapping tiles, processes each
    separately, and stitches results back together. This reduces
    peak memory usage for large datasets.
    """

    # Parameter names
    TILE_SIZE = "TILE_SIZE"
    AUTO_TILE_SIZE = "AUTO_TILE_SIZE"
    OUTPUT_TMRT = "OUTPUT_TMRT"

    def name(self) -> str:
        return "tiled_processing"

    def displayName(self) -> str:
        return self.tr("Calculate Tmrt (Large Rasters)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Calculate Tmrt using tiled processing for large rasters.

<b>Purpose:</b>
For very large rasters (>4000x4000 pixels), memory usage can become
prohibitive. This algorithm divides the input into overlapping tiles,
processes each separately, and stitches results together.

<b>Tile processing:</b>
- Tiles overlap by a buffer distance calculated from max building height
- Shadow calculations at tile edges require neighboring tile data
- Results are seamlessly stitched with no visible boundaries

<b>Memory usage:</b>
- Standard processing: ~370 bytes/pixel (10k x 10k = 34 GB)
- Tiled processing: Controlled by tile size (1024 x 1024 = ~400 MB peak)

<b>Tip:</b>
Enable "Auto tile size" to let the algorithm choose optimal tile
dimensions based on available memory and raster size."""
        )

    def group(self) -> str:
        return self.tr("SOLWEIG > Calculation")

    def groupId(self) -> str:
        return "solweig_calculation"

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # Surface inputs
        add_surface_parameters(self)

        # Location
        add_location_parameters(self)

        # Weather
        add_weather_parameters(self)

        # Human parameters
        add_human_parameters(self)

        # Options
        add_options_parameters(self)

        # Tiling parameters
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AUTO_TILE_SIZE,
                self.tr("Auto-calculate optimal tile size"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.TILE_SIZE,
                self.tr("Tile size (pixels, if not auto)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1024,
                minValue=256,
                maxValue=4096,
            )
        )

        # Output
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_TMRT,
                self.tr("Mean Radiant Temperature (Tmrt)"),
            )
        )

        # Output statistics
        self.addOutput(
            QgsProcessingOutputNumber(
                "TILE_COUNT",
                self.tr("Number of tiles processed"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "MEAN_TMRT",
                self.tr("Mean Tmrt (°C)"),
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
        feedback.pushInfo("SOLWEIG Tiled Processing")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Step 1: Create SurfaceData
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        surface = create_surface_from_parameters(parameters, context, self, feedback)

        rows, cols = surface.dsm.shape
        feedback.pushInfo(f"Raster size: {cols}x{rows} pixels")

        if feedback.isCanceled():
            return {}

        # Step 2: Create Location
        feedback.setProgressText("Setting up location...")
        feedback.setProgress(10)

        location = create_location_from_parameters(parameters, surface, feedback)

        if feedback.isCanceled():
            return {}

        # Step 3: Create Weather
        feedback.setProgressText("Setting up weather...")
        feedback.setProgress(15)

        weather = create_weather_from_parameters(parameters, feedback)

        if feedback.isCanceled():
            return {}

        # Step 4: Create HumanParams
        human = create_human_params_from_parameters(parameters)

        # Step 5: Determine tile size
        auto_tile = self.parameterAsBool(parameters, self.AUTO_TILE_SIZE, context)
        if auto_tile:
            # Auto-calculate based on raster size
            # Aim for tiles that fit in ~500MB
            # ~370 bytes/pixel, target 500MB = 1.35M pixels = ~1160x1160
            tile_size = min(1024, max(rows, cols))
            if rows * cols > 4000 * 4000:
                tile_size = 1024
            elif rows * cols > 2000 * 2000:
                tile_size = min(rows, cols, 2048)
            else:
                tile_size = max(rows, cols)  # No tiling needed
            feedback.pushInfo(f"Auto tile size: {tile_size}x{tile_size}")
        else:
            tile_size = self.parameterAsInt(parameters, self.TILE_SIZE, context)
            feedback.pushInfo(f"Manual tile size: {tile_size}x{tile_size}")

        # Check if tiling is needed
        if rows <= tile_size and cols <= tile_size:
            feedback.pushInfo("Raster fits in single tile, using standard processing")
            # Use standard calculate() for small rasters
            try:
                result = solweig.calculate(
                    surface=surface,
                    location=location,
                    weather=weather,
                    human=human,
                    use_anisotropic_sky=parameters.get("USE_ANISOTROPIC_SKY", False),
                    conifer=parameters.get("CONIFER", False),
                )
            except Exception as e:
                raise QgsProcessingException(f"Calculation failed: {e}") from e

            tile_count = 1
            tmrt = result.tmrt
        else:
            # Use tiled processing
            feedback.setProgressText(f"Processing with {tile_size}x{tile_size} tiles...")
            feedback.setProgress(20)

            try:
                result = solweig.calculate_tiled(
                    surface=surface,
                    location=location,
                    weather=weather,
                    human=human,
                    tile_size=tile_size,
                    use_anisotropic_sky=parameters.get("USE_ANISOTROPIC_SKY", False),
                    conifer=parameters.get("CONIFER", False),
                )
                tile_count = result.tile_count if hasattr(result, "tile_count") else 1
                tmrt = result.tmrt
            except Exception as e:
                raise QgsProcessingException(f"Tiled calculation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        feedback.setProgress(90)

        # Save output
        feedback.setProgressText("Saving output...")

        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_TMRT, context)

        self.save_georeferenced_output(
            array=tmrt,
            output_path=output_path,
            geotransform=surface._geotransform,
            crs_wkt=surface._crs_wkt,
            feedback=feedback,
        )

        # Add to canvas
        timestamp_str = weather.datetime.strftime("%Y-%m-%d %H:%M")
        self.add_raster_to_canvas(
            path=output_path,
            layer_name=f"Tmrt {timestamp_str} (tiled)",
            style="tmrt",
            context=context,
        )

        elapsed = time.time() - start_time
        feedback.setProgress(100)

        # Statistics
        import numpy as np

        tmrt_valid = tmrt[~np.isnan(tmrt)]
        mean_tmrt = float(np.mean(tmrt_valid)) if len(tmrt_valid) > 0 else 0.0

        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Tiled processing complete!")
        feedback.pushInfo(f"  Tiles processed: {tile_count}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")
        feedback.pushInfo(f"  Mean Tmrt: {mean_tmrt:.1f}C")
        feedback.pushInfo(f"  Output: {output_path}")
        feedback.pushInfo("=" * 60)

        return {
            self.OUTPUT_TMRT: output_path,
            "TILE_COUNT": tile_count,
            "MEAN_TMRT": mean_tmrt,
        }
