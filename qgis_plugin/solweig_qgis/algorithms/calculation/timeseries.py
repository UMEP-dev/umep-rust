"""
Timeseries SOLWEIG Calculation Algorithm

Calculates Mean Radiant Temperature (Tmrt) for multiple timesteps
with thermal state accumulation.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
)

from ...utils.converters import (
    create_location_from_parameters,
    create_surface_from_parameters,
    load_weather_from_epw,
)
from ...utils.parameters import (
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_surface_parameters,
)
from ..base import SolweigAlgorithmBase


class TimeseriesAlgorithm(SolweigAlgorithmBase):
    """
    Calculate Tmrt for multiple timesteps with thermal state.

    Uses the TsWaveDelay model to accurately simulate thermal
    inertia effects across timesteps. Recommended for multi-hour
    or multi-day simulations.
    """

    # Parameter names
    EPW_FILE = "EPW_FILE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    HOURS_FILTER = "HOURS_FILTER"
    OUTPUT_DIR = "OUTPUT_DIR"
    OUTPUTS = "OUTPUTS"

    def name(self) -> str:
        return "timeseries"

    def displayName(self) -> str:
        return self.tr("Calculate Tmrt (Timeseries)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Calculate Mean Radiant Temperature for multiple timesteps.

<b>Purpose:</b>
For accurate multi-timestep simulations, this algorithm uses the
TsWaveDelay model to account for thermal inertia effects (ground
heating/cooling) across timesteps.

<b>Weather Input:</b>
- EPW (EnergyPlus Weather) file with hourly data
- Date range to process
- Optional hour filter (e.g., "9,10,11,12" for daylight hours only)

<b>Outputs:</b>
- GeoTIFF files for each timestep (tmrt_YYYYMMDD_HHMM.tif)
- run_metadata.json with provenance information
- Optional: shadow, radiation components

<b>Performance:</b>
- First timestep: ~60s (includes SVF computation if not cached)
- Subsequent timesteps: ~0.3s each (with cached SVF)

<b>Tip:</b>
Use "Compute Sky View Factor" preprocessing first, then provide
the SVF directory for much faster timeseries calculations."""
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

        # EPW weather file
        self.addParameter(
            QgsProcessingParameterFile(
                self.EPW_FILE,
                self.tr("EPW weather file"),
                extension="epw",
            )
        )

        self.addParameter(
            QgsProcessingParameterDateTime(
                self.START_DATE,
                self.tr("Start date/time"),
                type=QgsProcessingParameterDateTime.DateTime,
            )
        )

        self.addParameter(
            QgsProcessingParameterDateTime(
                self.END_DATE,
                self.tr("End date/time"),
                type=QgsProcessingParameterDateTime.DateTime,
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.HOURS_FILTER,
                self.tr("Hours to include (comma-separated, e.g., 9,10,11,12)"),
                optional=True,
            )
        )

        # Human parameters
        add_human_parameters(self)

        # Options
        add_options_parameters(self)

        # Output selection
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OUTPUTS,
                self.tr("Outputs to save"),
                options=["tmrt", "shadow", "kdown", "kup", "ldown", "lup"],
                allowMultiple=True,
                defaultValue=[0],  # tmrt only
            )
        )

        # Output directory
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output directory"),
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                self.tr("Output folder"),
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                "METADATA_FILE",
                self.tr("Run metadata file"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "TIMESTEP_COUNT",
                self.tr("Number of timesteps processed"),
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the algorithm."""
        import os
        import time

        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG Timeseries Calculation")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Step 1: Load weather from EPW
        feedback.setProgressText("Loading EPW weather data...")
        feedback.setProgress(5)

        epw_path = self.parameterAsFile(parameters, self.EPW_FILE, context)
        start_dt = self.parameterAsDateTime(parameters, self.START_DATE, context)
        end_dt = self.parameterAsDateTime(parameters, self.END_DATE, context)
        hours_filter = self.parameterAsString(parameters, self.HOURS_FILTER, context)

        weather_series = load_weather_from_epw(
            epw_path=epw_path,
            start_dt=start_dt,
            end_dt=end_dt,
            hours_filter=hours_filter,
            feedback=feedback,
        )

        if not weather_series:
            raise QgsProcessingException("No timesteps found in specified date range")

        if feedback.isCanceled():
            return {}

        # Step 2: Create SurfaceData
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(10)

        surface = create_surface_from_parameters(parameters, context, self, feedback)

        if feedback.isCanceled():
            return {}

        # Step 3: Create Location
        feedback.setProgressText("Setting up location...")
        feedback.setProgress(15)

        location = create_location_from_parameters(parameters, surface, feedback)

        if feedback.isCanceled():
            return {}

        # Step 4: Get options
        use_anisotropic_sky = parameters.get("USE_ANISOTROPIC_SKY", False)
        conifer = parameters.get("CONIFER", False)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Get selected outputs
        output_indices = self.parameterAsEnums(parameters, self.OUTPUTS, context)
        output_names = ["tmrt", "shadow", "kdown", "kup", "ldown", "lup"]
        selected_outputs = [output_names[i] for i in output_indices]

        feedback.pushInfo(f"Outputs: {', '.join(selected_outputs)}")

        # Load precomputed SVF if provided
        svf_dir = parameters.get("SVF_DIR")
        if svf_dir:
            feedback.pushInfo(f"Loading pre-computed SVF from {svf_dir}")
            try:
                solweig.PrecomputedData.load(svf_dir=svf_dir)
            except Exception as e:
                feedback.reportError(
                    f"Could not load SVF: {e}",
                    fatalError=False,
                )

        if feedback.isCanceled():
            return {}

        # Step 5: Run timeseries calculation
        feedback.setProgressText(f"Running timeseries ({len(weather_series)} timesteps)...")
        feedback.setProgress(20)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            results = solweig.calculate_timeseries(
                surface=surface,
                weather_series=weather_series,
                location=location,
                output_dir=output_dir,
                outputs=selected_outputs,
                use_anisotropic_sky=use_anisotropic_sky,
                conifer=conifer,
            )
        except Exception as e:
            raise QgsProcessingException(f"Timeseries calculation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        elapsed = time.time() - start_time
        feedback.setProgress(100)

        # Calculate summary statistics
        import numpy as np

        if results:
            all_tmrt = [r.tmrt[~np.isnan(r.tmrt)] for r in results]
            mean_tmrt = np.mean([arr.mean() for arr in all_tmrt])
            max_tmrt = max(arr.max() for arr in all_tmrt)
            min_tmrt = min(arr.min() for arr in all_tmrt)
        else:
            mean_tmrt = max_tmrt = min_tmrt = 0.0

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Timeseries calculation complete!")
        feedback.pushInfo(f"  Timesteps processed: {len(results)}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")
        feedback.pushInfo(f"  Per timestep: {elapsed / len(results):.2f} seconds")
        feedback.pushInfo(f"  Tmrt range: {min_tmrt:.1f}C - {max_tmrt:.1f}C")
        feedback.pushInfo(f"  Mean Tmrt: {mean_tmrt:.1f}C")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("=" * 60)

        metadata_file = os.path.join(output_dir, "run_metadata.json")

        return {
            "OUTPUT_FOLDER": output_dir,
            "METADATA_FILE": metadata_file,
            "TIMESTEP_COUNT": len(results),
        }
