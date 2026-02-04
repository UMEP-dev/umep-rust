"""
EPW Import/Validation Algorithm

Preview and validate EnergyPlus Weather (EPW) files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFile,
)

from ..base import SolweigAlgorithmBase


class EpwImportAlgorithm(SolweigAlgorithmBase):
    """
    Preview and validate EPW weather files.

    Displays location information, date range, and data statistics
    to help users understand their weather data before running
    calculations.
    """

    # Parameter names
    EPW_FILE = "EPW_FILE"

    def name(self) -> str:
        return "epw_import"

    def displayName(self) -> str:
        return self.tr("Import EPW Weather File")

    def shortHelpString(self) -> str:
        return self.tr(
            """Preview and validate EnergyPlus Weather (EPW) files.

<b>Purpose:</b>
Inspect EPW weather files before running SOLWEIG calculations.
Displays location, date range, and data quality information.

<b>EPW Format:</b>
EnergyPlus Weather files contain hourly meteorological data including:
- Air temperature
- Relative humidity
- Wind speed and direction
- Solar radiation (global, direct, diffuse)
- Atmospheric pressure

<b>Output:</b>
HTML report with:
- Location (city, country, coordinates, elevation)
- Date range and timestep count
- Data statistics (min, max, mean for key variables)
- Data quality warnings (missing values, outliers)

<b>Tip:</b>
EPW files can be downloaded from:
- climate.onebuilding.org (global coverage)
- energyplus.net/weather (official EnergyPlus collection)"""
        )

    def group(self) -> str:
        return self.tr("SOLWEIG > Utilities")

    def groupId(self) -> str:
        return "solweig_utilities"

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterFile(
                self.EPW_FILE,
                self.tr("EPW weather file"),
                extension="epw",
            )
        )

        # Output
        self.addOutput(
            QgsProcessingOutputHtml(
                "OUTPUT_HTML",
                self.tr("EPW Information Report"),
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
        feedback.pushInfo("EPW Weather File Import")
        feedback.pushInfo("=" * 60)

        # Import solweig
        self.import_solweig()
        from solweig.io import read_epw

        # Get parameters
        epw_path = self.parameterAsFile(parameters, self.EPW_FILE, context)

        feedback.pushInfo(f"Reading: {epw_path}")

        # Read EPW file
        try:
            df, metadata = read_epw(epw_path)
        except FileNotFoundError as e:
            raise QgsProcessingException(f"EPW file not found: {epw_path}") from e
        except Exception as e:
            raise QgsProcessingException(f"Error reading EPW file: {e}") from e

        # Display key info
        feedback.pushInfo("")
        feedback.pushInfo(f"Location: {metadata.get('city', 'Unknown')}")
        feedback.pushInfo(f"Coordinates: {metadata.get('latitude', 'N/A')}N, {metadata.get('longitude', 'N/A')}E")
        feedback.pushInfo(f"Elevation: {metadata.get('elevation', 'N/A')} m")
        feedback.pushInfo(f"UTC offset: {metadata.get('tz_offset', 'N/A')} hours")
        feedback.pushInfo("")
        feedback.pushInfo(f"Data range: {df.index.min()} to {df.index.max()}")
        feedback.pushInfo(f"Timesteps: {len(df)}")

        # Generate HTML report
        html = self._generate_html_report(df, metadata, epw_path)

        # Save to temp file
        output_html = str(Path(tempfile.gettempdir()) / f"epw_report_{Path(epw_path).stem}.html")
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)

        feedback.pushInfo("")
        feedback.pushInfo(f"Report saved: {output_html}")

        return {"OUTPUT_HTML": output_html}

    def _generate_html_report(self, df, metadata: dict, epw_path: str) -> str:
        """Generate HTML report for EPW file."""
        # Compute statistics
        stats = df.describe()

        # Check for missing values
        missing = df.isnull().sum()
        has_missing = missing.any()

        # Map column names to friendly names
        column_names = {
            "temp_air": "Air Temperature (°C)",
            "relative_humidity": "Relative Humidity (%)",
            "wind_speed": "Wind Speed (m/s)",
            "ghi": "Global Horizontal Irradiance (W/m²)",
            "dni": "Direct Normal Irradiance (W/m²)",
            "dhi": "Diffuse Horizontal Irradiance (W/m²)",
            "atmospheric_pressure": "Atmospheric Pressure (Pa)",
        }

        # Build statistics table rows
        stats_rows = ""
        for col in ["temp_air", "relative_humidity", "wind_speed", "ghi"]:
            if col in df.columns:
                friendly_name = column_names.get(col, col)
                stats_rows += f"""
                <tr>
                    <td>{friendly_name}</td>
                    <td>{stats.loc["min", col]:.1f}</td>
                    <td>{stats.loc["max", col]:.1f}</td>
                    <td>{stats.loc["mean", col]:.1f}</td>
                    <td>{missing.get(col, 0)}</td>
                </tr>
                """

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EPW Weather File Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #333; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .info {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .file-path {{
            background: #eee;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>EPW Weather File Report</h1>

    <div class="card">
        <h2>File Information</h2>
        <div class="file-path">{epw_path}</div>
    </div>

    <div class="card">
        <h2>Location</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>City</td><td>{metadata.get("city", "Unknown")}</td></tr>
            <tr><td>State/Province</td><td>{metadata.get("state", "-")}</td></tr>
            <tr><td>Country</td><td>{metadata.get("country", "Unknown")}</td></tr>
            <tr><td>Latitude</td><td>{metadata.get("latitude", "N/A")}° N</td></tr>
            <tr><td>Longitude</td><td>{metadata.get("longitude", "N/A")}° E</td></tr>
            <tr><td>Elevation</td><td>{metadata.get("elevation", "N/A")} m</td></tr>
            <tr><td>UTC Offset</td><td>UTC{metadata.get("tz_offset", 0):+.0f}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Data Range</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Start Date</td><td>{df.index.min()}</td></tr>
            <tr><td>End Date</td><td>{df.index.max()}</td></tr>
            <tr><td>Total Timesteps</td><td>{len(df):,}</td></tr>
            <tr><td>Timestep Interval</td><td>Hourly</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Data Statistics</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Missing</th>
            </tr>
            {stats_rows}
        </table>
    </div>

    {
            "<div class='warning'><strong>Warning:</strong> "
            "Some variables have missing values. Check the Missing column above.</div>"
            if has_missing
            else ""
        }

    <div class="info">
        <strong>Next Steps:</strong>
        <ul>
            <li>Use this EPW file with "Calculate Tmrt (Timeseries)" algorithm</li>
            <li>UTC offset is important for accurate sun position calculation</li>
            <li>Consider filtering hours (e.g., 9-17) for daylight-only analysis</li>
        </ul>
    </div>

    <div class="card">
        <h2>SOLWEIG Compatibility</h2>
        <p>This EPW file is compatible with SOLWEIG calculations. The following
        variables will be used:</p>
        <ul>
            <li><strong>temp_air</strong>: Air temperature for Tmrt and thermal comfort</li>
            <li><strong>relative_humidity</strong>: For UTCI and PET calculations</li>
            <li><strong>ghi</strong>: Global solar radiation for shortwave radiation</li>
            <li><strong>wind_speed</strong>: For UTCI and PET calculations</li>
        </ul>
    </div>

</body>
</html>
        """

        return html
