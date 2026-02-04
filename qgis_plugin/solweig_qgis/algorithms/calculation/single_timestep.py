"""
Single Timestep SOLWEIG Calculation Algorithm

Calculates Mean Radiant Temperature (Tmrt) for a single datetime.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputNumber,
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


class SingleTimestepAlgorithm(SolweigAlgorithmBase):
    """
    Calculate Mean Radiant Temperature (Tmrt) for a single timestep.

    This algorithm computes Tmrt at all grid cells for a given date/time
    and weather conditions. Use this for exploring specific conditions
    or quick calculations.

    For multi-timestep calculations with thermal state accumulation,
    use the Timeseries algorithm instead.
    """

    # Output parameter names
    OUTPUT_TMRT = "OUTPUT_TMRT"
    OUTPUT_SHADOW = "OUTPUT_SHADOW"
    OUTPUT_KDOWN = "OUTPUT_KDOWN"
    OUTPUT_MEAN_TMRT = "OUTPUT_MEAN_TMRT"
    OUTPUT_MAX_TMRT = "OUTPUT_MAX_TMRT"

    def name(self) -> str:
        """Algorithm identifier used in scripts."""
        return "single_timestep"

    def displayName(self) -> str:
        """Human-readable algorithm name."""
        return self.tr("Calculate Tmrt (Single Timestep)")

    def shortHelpString(self) -> str:
        """Algorithm description shown in help panel."""
        return self.tr(
            """Calculate Mean Radiant Temperature (Tmrt) for a single date/time.

<b>Required inputs:</b>
- DSM (Digital Surface Model): Elevation raster in meters

<b>Optional inputs:</b>
- CDSM: Canopy/vegetation heights
- DEM: Ground elevation (for relative height conversion)
- TDSM: Trunk zone heights
- Land cover: UMEP classification (affects albedo/emissivity)
- Pre-computed SVF directory

<b>Location:</b>
Either auto-extract from DSM CRS or enter latitude/longitude manually.
UTC offset is required for accurate sun position calculation.

<b>Weather:</b>
- Date/time of calculation
- Air temperature (°C)
- Relative humidity (%)
- Global solar radiation (W/m²)

<b>Outputs:</b>
- Tmrt raster (°C) - Mean Radiant Temperature
- Optional: Shadow mask, Downwelling shortwave radiation

<b>Note:</b> For multi-timestep calculations with thermal inertia modeling,
use the "Calculate Tmrt (Timeseries)" algorithm instead."""
        )

    def group(self) -> str:
        """Algorithm group name."""
        return self.tr("SOLWEIG > Calculation")

    def groupId(self) -> str:
        """Algorithm group ID."""
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

        # Outputs
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_TMRT,
                self.tr("Mean Radiant Temperature (Tmrt)"),
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_SHADOW,
                self.tr("Shadow mask (optional)"),
                optional=True,
                createByDefault=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_KDOWN,
                self.tr("Downwelling shortwave radiation (optional)"),
                optional=True,
                createByDefault=False,
            )
        )

        # Output statistics
        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_MEAN_TMRT,
                self.tr("Mean Tmrt (°C)"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.OUTPUT_MAX_TMRT,
                self.tr("Maximum Tmrt (°C)"),
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
        feedback.pushInfo("SOLWEIG Single Timestep Calculation")
        feedback.pushInfo("=" * 60)

        # Import solweig
        solweig = self.import_solweig()

        # Step 1: Create SurfaceData
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        surface = create_surface_from_parameters(parameters, context, self, feedback)

        if feedback.isCanceled():
            return {}

        # Step 2: Create Location
        feedback.setProgressText("Setting up location...")
        feedback.setProgress(15)

        location = create_location_from_parameters(parameters, surface, feedback)

        if feedback.isCanceled():
            return {}

        # Step 3: Create Weather
        feedback.setProgressText("Setting up weather...")
        feedback.setProgress(20)

        weather = create_weather_from_parameters(parameters, feedback)

        if feedback.isCanceled():
            return {}

        # Step 4: Create HumanParams
        human = create_human_params_from_parameters(parameters)

        # Step 5: Load precomputed SVF if provided
        precomputed = None
        svf_dir = parameters.get("SVF_DIR")
        if svf_dir:
            feedback.pushInfo(f"Loading pre-computed SVF from {svf_dir}")
            try:
                precomputed = solweig.PrecomputedData.load(svf_dir=svf_dir)
            except Exception as e:
                feedback.reportError(
                    f"Could not load SVF from {svf_dir}: {e}",
                    fatalError=False,
                )

        # Step 6: Get options
        use_anisotropic_sky = parameters.get("USE_ANISOTROPIC_SKY", False)
        conifer = parameters.get("CONIFER", False)

        # Step 7: Validate inputs
        feedback.setProgressText("Validating inputs...")
        feedback.setProgress(25)

        try:
            warnings = solweig.validate_inputs(
                surface=surface,
                location=location,
                weather=weather,
                use_anisotropic_sky=use_anisotropic_sky,
                precomputed=precomputed,
            )
            for warning in warnings:
                feedback.reportError(f"Warning: {warning}", fatalError=False)
        except solweig.SolweigError as e:
            raise QgsProcessingException(f"Validation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        # Step 8: Calculate Tmrt
        feedback.setProgressText("Calculating Mean Radiant Temperature...")
        feedback.setProgress(30)

        try:
            result = solweig.calculate(
                surface=surface,
                location=location,
                weather=weather,
                human=human,
                precomputed=precomputed,
                use_anisotropic_sky=use_anisotropic_sky,
                conifer=conifer,
            )
        except Exception as e:
            raise QgsProcessingException(f"Calculation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        feedback.setProgress(80)

        # Step 9: Save outputs
        feedback.setProgressText("Saving outputs...")

        outputs = {}

        # Save Tmrt (required output)
        tmrt_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_TMRT, context)
        self.save_georeferenced_output(
            array=result.tmrt,
            output_path=tmrt_path,
            geotransform=surface._geotransform,
            crs_wkt=surface._crs_wkt,
            feedback=feedback,
        )
        outputs[self.OUTPUT_TMRT] = tmrt_path

        # Add to canvas with thermal styling
        timestamp_str = weather.datetime.strftime("%Y-%m-%d %H:%M")
        self.add_raster_to_canvas(
            path=tmrt_path,
            layer_name=f"Tmrt {timestamp_str}",
            style="tmrt",
            context=context,
        )

        # Save optional shadow output
        shadow_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_SHADOW, context)
        if shadow_path:
            self.save_georeferenced_output(
                array=result.shadow,
                output_path=shadow_path,
                geotransform=surface._geotransform,
                crs_wkt=surface._crs_wkt,
                feedback=feedback,
            )
            outputs[self.OUTPUT_SHADOW] = shadow_path
            self.add_raster_to_canvas(
                path=shadow_path,
                layer_name=f"Shadow {timestamp_str}",
                style="shadow",
                context=context,
            )

        # Save optional Kdown output
        kdown_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_KDOWN, context)
        if kdown_path and hasattr(result, "kdown") and result.kdown is not None:
            self.save_georeferenced_output(
                array=result.kdown,
                output_path=kdown_path,
                geotransform=surface._geotransform,
                crs_wkt=surface._crs_wkt,
                feedback=feedback,
            )
            outputs[self.OUTPUT_KDOWN] = kdown_path

        feedback.setProgress(100)

        # Calculate statistics
        import numpy as np

        tmrt_valid = result.tmrt[~np.isnan(result.tmrt)]
        mean_tmrt = float(np.mean(tmrt_valid)) if len(tmrt_valid) > 0 else 0.0
        max_tmrt = float(np.max(tmrt_valid)) if len(tmrt_valid) > 0 else 0.0
        min_tmrt = float(np.min(tmrt_valid)) if len(tmrt_valid) > 0 else 0.0

        outputs[self.OUTPUT_MEAN_TMRT] = mean_tmrt
        outputs[self.OUTPUT_MAX_TMRT] = max_tmrt

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Calculation complete!")
        feedback.pushInfo(f"  Tmrt range: {min_tmrt:.1f}C - {max_tmrt:.1f}C")
        feedback.pushInfo(f"  Mean Tmrt: {mean_tmrt:.1f}C")
        feedback.pushInfo(f"  Output: {tmrt_path}")
        feedback.pushInfo("=" * 60)

        return outputs
