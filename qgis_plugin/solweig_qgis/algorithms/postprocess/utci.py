"""
UTCI Post-Processing Algorithm

Computes Universal Thermal Climate Index from Tmrt outputs.
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

from ...utils.converters import load_weather_from_epw
from ..base import SolweigAlgorithmBase


class UtciAlgorithm(SolweigAlgorithmBase):
    """
    Compute UTCI thermal comfort index from Tmrt outputs.

    UTCI (Universal Thermal Climate Index) is a widely-used thermal
    comfort index that accounts for air temperature, humidity,
    wind speed, and mean radiant temperature.

    Uses a fast polynomial approximation (~200 terms).
    """

    # Parameter names
    TMRT_DIR = "TMRT_DIR"
    EPW_FILE = "EPW_FILE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "compute_utci"

    def displayName(self) -> str:
        return self.tr("Compute UTCI (Universal Thermal Climate Index)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Compute UTCI thermal comfort index from Tmrt files.

<b>What is UTCI?</b>
The Universal Thermal Climate Index is a thermal comfort index
that combines:
- Air temperature
- Relative humidity
- Wind speed
- Mean radiant temperature (from SOLWEIG)

<b>Input:</b>
- Directory containing Tmrt GeoTIFF files (from timeseries calculation)
- EPW weather file (provides Ta, RH, wind for each timestep)

<b>Output:</b>
- UTCI GeoTIFF files (utci_YYYYMMDD_HHMM.tif)

<b>UTCI Thermal Stress Categories:</b>
- Below -40°C: Extreme cold stress
- -40 to -27°C: Very strong cold stress
- -27 to -13°C: Strong cold stress
- -13 to 0°C: Moderate cold stress
- 0 to 9°C: Slight cold stress
- 9 to 26°C: No thermal stress (comfortable)
- 26 to 32°C: Moderate heat stress
- 32 to 38°C: Strong heat stress
- 38 to 46°C: Very strong heat stress
- Above 46°C: Extreme heat stress

<b>Performance:</b>
Very fast polynomial approximation - processes ~200 timesteps/second."""
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

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("UTCI output directory"),
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                self.tr("UTCI output folder"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "FILE_COUNT",
                self.tr("Number of UTCI files created"),
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
        feedback.pushInfo("SOLWEIG UTCI Post-Processing")
        feedback.pushInfo("=" * 60)

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

        # Load weather data
        feedback.setProgressText("Loading weather data...")
        feedback.setProgress(10)

        weather_series = load_weather_from_epw(
            epw_path=epw_path,
            start_dt=start_dt,
            end_dt=end_dt,
            hours_filter=None,
            feedback=feedback,
        )

        if feedback.isCanceled():
            return {}

        # Compute UTCI
        feedback.setProgressText("Computing UTCI...")
        feedback.setProgress(20)

        try:
            n_files = solweig.compute_utci(
                tmrt_dir=tmrt_dir,
                weather_series=weather_series,
                output_dir=output_dir,
            )
        except Exception as e:
            raise QgsProcessingException(f"UTCI computation failed: {e}") from e

        elapsed = time.time() - start_time
        feedback.setProgress(100)

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("UTCI computation complete!")
        feedback.pushInfo(f"  Files created: {n_files}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("=" * 60)

        return {
            "OUTPUT_FOLDER": output_dir,
            "FILE_COUNT": n_files,
        }
