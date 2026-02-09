"""
Common parameter definitions for SOLWEIG algorithms.

Provides reusable parameter builders for consistent UI across algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgis.core import (
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

if TYPE_CHECKING:
    from qgis.core import QgsProcessingAlgorithm


def add_surface_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add standard surface data input parameters.

    Parameters added:
        DSM (required): Digital Surface Model
        CDSM (optional): Canopy DSM (vegetation heights)
        DEM (optional): Digital Elevation Model (ground)
        TDSM (optional): Trunk zone DSM
        LAND_COVER (optional): Land cover classification
        RELATIVE_HEIGHTS: Vegetation height convention (0=relative, 1=absolute)
    """
    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "DSM",
            algorithm.tr("Digital Surface Model (DSM)"),
            optional=False,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "CDSM",
            algorithm.tr("Canopy DSM (vegetation heights)"),
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "DEM",
            algorithm.tr("Digital Elevation Model (ground)"),
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "TDSM",
            algorithm.tr("Trunk zone DSM"),
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterRasterLayer(
            "LAND_COVER",
            algorithm.tr("Land cover classification (UMEP IDs)"),
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "RELATIVE_HEIGHTS",
            algorithm.tr("Vegetation height convention (CDSM/TDSM)"),
            options=[
                "Relative — height above ground (e.g. tree = 8 m)",
                "Absolute — elevation above sea level (e.g. tree = 133 m)",
            ],
            defaultValue=0,  # Relative
        )
    )


def add_location_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add location parameters with auto-extraction option.

    Parameters added:
        AUTO_EXTRACT_LOCATION: Extract lat/lon from DSM CRS
        LATITUDE: Manual latitude input
        LONGITUDE: Manual longitude input
        UTC_OFFSET: UTC timezone offset
    """
    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "AUTO_EXTRACT_LOCATION",
            algorithm.tr("Auto-extract location from DSM CRS"),
            defaultValue=False,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "LATITUDE",
            algorithm.tr("Latitude (degrees)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=57.7,
            minValue=-90.0,
            maxValue=90.0,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "LONGITUDE",
            algorithm.tr("Longitude (degrees)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=12.0,
            minValue=-180.0,
            maxValue=180.0,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "UTC_OFFSET",
            algorithm.tr("UTC offset (hours)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0,
            minValue=-12,
            maxValue=14,
        )
    )


def add_weather_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add weather parameters for single timestep calculation.

    Parameters added:
        DATETIME: Date and time of calculation
        TEMPERATURE: Air temperature (°C)
        HUMIDITY: Relative humidity (%)
        GLOBAL_RADIATION: Global solar radiation (W/m²)
        WIND_SPEED: Wind speed (m/s)
        PRESSURE: Atmospheric pressure (hPa)
    """
    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "DATETIME",
            algorithm.tr("Date and time"),
            type=QgsProcessingParameterDateTime.DateTime,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "TEMPERATURE",
            algorithm.tr("Air temperature (°C)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=25.0,
            minValue=-50.0,
            maxValue=60.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "HUMIDITY",
            algorithm.tr("Relative humidity (%)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=50.0,
            minValue=0.0,
            maxValue=100.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "GLOBAL_RADIATION",
            algorithm.tr("Global solar radiation (W/m²)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=800.0,
            minValue=0.0,
            maxValue=1400.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "WIND_SPEED",
            algorithm.tr("Wind speed (m/s)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
            minValue=0.0,
            maxValue=50.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "PRESSURE",
            algorithm.tr("Atmospheric pressure (hPa)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1013.25,
            minValue=800.0,
            maxValue=1100.0,
        )
    )


def add_human_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add human body parameters.

    Parameters added:
        POSTURE: Standing or sitting
        ABS_K: Shortwave absorption coefficient
    """
    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "POSTURE",
            algorithm.tr("Body posture"),
            options=["Standing", "Sitting"],
            defaultValue=0,  # Standing
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "ABS_K",
            algorithm.tr("Shortwave absorption coefficient"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.7,
            minValue=0.0,
            maxValue=1.0,
        )
    )


def add_human_body_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add detailed human body parameters for PET calculation.

    Parameters added:
        AGE, WEIGHT, HEIGHT, SEX, ACTIVITY, CLOTHING
    """
    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "AGE",
            algorithm.tr("Age (years)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=35,
            minValue=1,
            maxValue=120,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "WEIGHT",
            algorithm.tr("Body weight (kg)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=75.0,
            minValue=20.0,
            maxValue=200.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "HEIGHT",
            algorithm.tr("Body height (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.75,
            minValue=1.0,
            maxValue=2.5,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterEnum(
            "SEX",
            algorithm.tr("Sex"),
            options=["Male", "Female"],
            defaultValue=0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "ACTIVITY",
            algorithm.tr("Metabolic activity (W)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=80.0,
            minValue=40.0,
            maxValue=500.0,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterNumber(
            "CLOTHING",
            algorithm.tr("Clothing insulation (clo)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.9,
            minValue=0.0,
            maxValue=2.0,
        )
    )


def add_options_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add calculation options parameters.

    Parameters added:
        USE_ANISOTROPIC_SKY: Enable anisotropic sky model
        CONIFER: Treat vegetation as evergreen
        SVF_DIR: Override SVF directory (optional)
    """
    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "USE_ANISOTROPIC_SKY",
            algorithm.tr("Use anisotropic sky model"),
            defaultValue=False,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterBoolean(
            "CONIFER",
            algorithm.tr("Treat vegetation as evergreen (conifer)"),
            defaultValue=False,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterFile(
            "SVF_DIR",
            algorithm.tr("Override SVF directory (SVF is included in prepared surface by default)"),
            behavior=QgsProcessingParameterFile.Folder,
            optional=True,
        )
    )


def add_output_tmrt_parameter(algorithm: QgsProcessingAlgorithm) -> None:
    """Add Tmrt output raster parameter."""
    algorithm.addParameter(
        QgsProcessingParameterRasterDestination(
            "OUTPUT_TMRT",
            algorithm.tr("Mean Radiant Temperature (Tmrt)"),
        )
    )


def add_output_dir_parameter(algorithm: QgsProcessingAlgorithm) -> None:
    """Add output directory parameter."""
    algorithm.addParameter(
        QgsProcessingParameterFolderDestination(
            "OUTPUT_DIR",
            algorithm.tr("Output directory"),
        )
    )


def add_epw_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add EPW weather file parameter.

    Parameters added:
        EPW_FILE: Path to EPW file
    """
    algorithm.addParameter(
        QgsProcessingParameterFile(
            "EPW_FILE",
            algorithm.tr("EPW weather file"),
            extension="epw",
            optional=True,
        )
    )


def add_umep_met_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add UMEP/SUEWS meteorological file parameter.

    Parameters added:
        UMEP_MET_FILE: Path to UMEP/SUEWS met file
    """
    algorithm.addParameter(
        QgsProcessingParameterFile(
            "UMEP_MET_FILE",
            algorithm.tr("UMEP/SUEWS meteorological forcing file"),
            extension="txt",
            optional=True,
        )
    )


def add_date_filter_parameters(algorithm: QgsProcessingAlgorithm) -> None:
    """
    Add shared date/time filter parameters (used by EPW and UMEP modes).

    Parameters added:
        START_DATE: Start date for filtering
        END_DATE: End date for filtering
        HOURS_FILTER: Comma-separated hours to include
    """
    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "START_DATE",
            algorithm.tr("Start date (leave empty for full range)"),
            type=QgsProcessingParameterDateTime.DateTime,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterDateTime(
            "END_DATE",
            algorithm.tr("End date (leave empty for full range)"),
            type=QgsProcessingParameterDateTime.DateTime,
            optional=True,
        )
    )

    algorithm.addParameter(
        QgsProcessingParameterString(
            "HOURS_FILTER",
            algorithm.tr("Hours to include (comma-separated, e.g., 9,10,11,12)"),
            optional=True,
        )
    )
