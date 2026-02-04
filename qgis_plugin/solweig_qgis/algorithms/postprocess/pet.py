"""
PET Post-Processing Algorithm

Computes Physiological Equivalent Temperature from Tmrt outputs.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
)

from ...utils.converters import (
    create_human_params_from_parameters,
    load_weather_from_epw,
)
from ...utils.parameters import add_human_body_parameters
from ..base import SolweigAlgorithmBase


class PetAlgorithm(SolweigAlgorithmBase):
    """
    Compute PET thermal comfort index from Tmrt outputs.

    PET (Physiological Equivalent Temperature) is a detailed thermal
    comfort index based on the MEMI (Munich Energy-balance Model
    for Individuals) heat balance model.

    Note: PET computation is significantly slower than UTCI (~50x)
    due to its iterative solver.
    """

    # Parameter names
    TMRT_DIR = "TMRT_DIR"
    EPW_FILE = "EPW_FILE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "compute_pet"

    def displayName(self) -> str:
        return self.tr("Compute PET (Physiological Equivalent Temperature)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Compute PET thermal comfort index from Tmrt files.

<b>What is PET?</b>
Physiological Equivalent Temperature is a thermal comfort index
based on the MEMI heat balance model that accounts for:
- Meteorological conditions (Ta, RH, wind, Tmrt)
- Personal factors (age, weight, height, sex)
- Activity level (metabolic rate)
- Clothing insulation

<b>Input:</b>
- Directory containing Tmrt GeoTIFF files
- EPW weather file
- Human body parameters (optional, sensible defaults provided)

<b>Output:</b>
- PET GeoTIFF files (pet_YYYYMMDD_HHMM.tif)

<b>PET Thermal Perception Categories (for a "typical" person):</b>
- Below 4°C: Very cold
- 4 to 8°C: Cold
- 8 to 13°C: Cool
- 13 to 18°C: Slightly cool
- 18 to 23°C: Comfortable (neutral)
- 23 to 29°C: Slightly warm
- 29 to 35°C: Warm
- 35 to 41°C: Hot
- Above 41°C: Very hot

<b>Performance Note:</b>
PET uses an iterative heat balance solver, making it ~50x slower
than UTCI. For large datasets, consider using UTCI first and
computing PET only for critical timesteps.

<b>Processing rate:</b> ~4 timesteps/second (1000x1000 grid)"""
        )

    def group(self) -> str:
        return self.tr("SOLWEIG > Post-Processing")

    def groupId(self) -> str:
        return "solweig_postprocess"

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterFile(
                self.TMRT_DIR,
                self.tr("Tmrt output directory"),
                behavior=QgsProcessingParameterFile.Folder,
            )
        )

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

        # Human body parameters
        add_human_body_parameters(self)

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("PET output directory"),
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                self.tr("PET output folder"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "FILE_COUNT",
                self.tr("Number of PET files created"),
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
        feedback.pushInfo("SOLWEIG PET Post-Processing")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("")
        feedback.pushInfo("Note: PET computation is ~50x slower than UTCI due to iterative heat balance solver.")
        feedback.pushInfo("")

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Get parameters
        tmrt_dir = self.parameterAsString(parameters, self.TMRT_DIR, context)
        epw_path = self.parameterAsFile(parameters, self.EPW_FILE, context)
        start_dt = self.parameterAsDateTime(parameters, self.START_DATE, context)
        end_dt = self.parameterAsDateTime(parameters, self.END_DATE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        feedback.pushInfo(f"Tmrt directory: {tmrt_dir}")
        feedback.pushInfo(f"EPW file: {epw_path}")
        feedback.pushInfo(f"Output directory: {output_dir}")

        # Create human params
        human = create_human_params_from_parameters(parameters)
        feedback.pushInfo(
            f"Human params: {human.weight}kg, {human.height}m, {human.age}y, {human.activity}W, {human.clothing}clo"
        )

        # Load weather data
        feedback.setProgressText("Loading weather data...")
        feedback.setProgress(5)

        weather_series = load_weather_from_epw(
            epw_path=epw_path,
            start_dt=start_dt,
            end_dt=end_dt,
            hours_filter=None,
            feedback=feedback,
        )

        estimated_time = len(weather_series) * 0.25  # ~4 timesteps/second estimate
        feedback.pushInfo(f"Estimated processing time: {estimated_time:.0f} seconds ({len(weather_series)} timesteps)")

        if feedback.isCanceled():
            return {}

        # Compute PET
        feedback.setProgressText("Computing PET (this may take a while)...")
        feedback.setProgress(10)

        try:
            n_files = solweig.compute_pet(
                tmrt_dir=tmrt_dir,
                weather_series=weather_series,
                output_dir=output_dir,
                human=human,
            )
        except Exception as e:
            raise QgsProcessingException(f"PET computation failed: {e}") from e

        elapsed = time.time() - start_time
        feedback.setProgress(100)

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("PET computation complete!")
        feedback.pushInfo(f"  Files created: {n_files}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")
        if n_files > 0:
            feedback.pushInfo(f"  Per timestep: {elapsed / n_files:.2f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("=" * 60)

        return {
            "OUTPUT_FOLDER": output_dir,
            "FILE_COUNT": n_files,
        }
