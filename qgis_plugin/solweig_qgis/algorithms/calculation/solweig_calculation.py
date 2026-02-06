"""
Unified SOLWEIG Calculation Algorithm

Supports single timestep or EPW timeseries, optional tiled processing,
and optional UTCI/PET post-processing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from ...utils.converters import (
    create_human_params_from_parameters,
    create_location_from_parameters,
    create_surface_from_parameters,
    create_weather_from_parameters,
    load_weather_from_epw,
)
from ...utils.parameters import (
    add_epw_parameters,
    add_human_body_parameters,
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_surface_parameters,
    add_weather_parameters,
)
from ..base import SolweigAlgorithmBase


class SolweigCalculationAlgorithm(SolweigAlgorithmBase):
    """
    Unified SOLWEIG calculation algorithm.

    Combines single timestep, timeseries, tiled processing, and optional
    UTCI/PET post-processing into a single Processing algorithm.
    """

    # Weather source enum values
    WEATHER_SINGLE = 0
    WEATHER_EPW = 1

    def name(self) -> str:
        return "solweig_calculation"

    def displayName(self) -> str:
        return self.tr("SOLWEIG Calculation")

    def shortHelpString(self) -> str:
        return self.tr(
            """Calculate Mean Radiant Temperature (Tmrt) with SOLWEIG.

<b>Weather modes:</b>
<ul>
<li><b>Single timestep:</b> Manual weather input for one date/time</li>
<li><b>EPW weather file:</b> Load hourly data from an EnergyPlus Weather file.
  Thermal state (ground heating/cooling) accumulates across timesteps.</li>
</ul>

<b>Required inputs:</b>
<ul>
<li>DSM (Digital Surface Model)</li>
<li>Location (auto-extract from CRS or manual lat/lon)</li>
<li>Weather data (single timestep or EPW file + date range)</li>
</ul>

<b>Optional inputs:</b>
<ul>
<li>CDSM, DEM, TDSM, Land cover</li>
<li>Pre-computed SVF directory (highly recommended for timeseries)</li>
</ul>

<b>Post-processing (optional):</b>
<ul>
<li><b>UTCI</b> - fast polynomial (~200 timesteps/sec)</li>
<li><b>PET</b> - iterative heat balance (~4 timesteps/sec, ~50x slower than UTCI)</li>
</ul>

<b>Tiling (advanced):</b>
Enable tiled processing for large rasters (>4000x4000 pixels) to reduce
memory usage. Only available for single timestep mode.

<b>Outputs:</b>
GeoTIFF files in the output directory. Select which components to save:
tmrt (required), shadow, kdown, kup, ldown, lup.
UTCI/PET results are saved in subdirectories.

<b>Tips:</b>
<ol>
<li>Run "Compute Sky View Factor" preprocessing first for much faster timeseries</li>
<li>Use EPW mode for multi-timestep simulations (thermal state accumulation)</li>
<li>UTCI is fast; PET is slow - consider computing PET only for critical timesteps</li>
</ol>"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # --- Surface inputs ---
        add_surface_parameters(self)

        # --- Location ---
        add_location_parameters(self)

        # --- Weather source selector ---
        self.addParameter(
            QgsProcessingParameterEnum(
                "WEATHER_SOURCE",
                self.tr("Weather data source"),
                options=["Single timestep (manual entry)", "EPW weather file (timeseries)"],
                defaultValue=self.WEATHER_EPW,
            )
        )

        # --- Single timestep weather (advanced - collapsed by default) ---
        add_weather_parameters(self)
        for name in ("DATETIME", "TEMPERATURE", "HUMIDITY", "GLOBAL_RADIATION", "WIND_SPEED", "PRESSURE"):
            param = self.parameterDefinition(name)
            if param:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)

        # --- EPW weather ---
        add_epw_parameters(self)

        # --- Human parameters ---
        add_human_parameters(self)

        # --- Options ---
        add_options_parameters(self)

        # --- Post-processing ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                "COMPUTE_UTCI",
                self.tr("Compute UTCI (Universal Thermal Climate Index)"),
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                "COMPUTE_PET",
                self.tr("Compute PET (Physiological Equivalent Temperature) - ~50x slower than UTCI"),
                defaultValue=False,
            )
        )

        # PET body parameters (advanced)
        add_human_body_parameters(self)
        for name in ("AGE", "WEIGHT", "HEIGHT", "SEX", "ACTIVITY", "CLOTHING"):
            param = self.parameterDefinition(name)
            if param:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)

        # --- Tiling (advanced) ---
        enable_tiling = QgsProcessingParameterBoolean(
            "ENABLE_TILING",
            self.tr("Enable tiled processing (for large rasters, single timestep only)"),
            defaultValue=False,
        )
        enable_tiling.setFlags(enable_tiling.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(enable_tiling)

        auto_tile = QgsProcessingParameterBoolean(
            "AUTO_TILE_SIZE",
            self.tr("Auto-calculate optimal tile size"),
            defaultValue=True,
        )
        auto_tile.setFlags(auto_tile.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(auto_tile)

        tile_size = QgsProcessingParameterNumber(
            "TILE_SIZE",
            self.tr("Tile size (pixels, if not auto)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1024,
            minValue=256,
            maxValue=4096,
        )
        tile_size.setFlags(tile_size.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(tile_size)

        # --- Output selection ---
        self.addParameter(
            QgsProcessingParameterEnum(
                "OUTPUT_COMPONENTS",
                self.tr("Output components to save"),
                options=["tmrt", "shadow", "kdown", "kup", "ldown", "lup"],
                allowMultiple=True,
                defaultValue=[0],  # tmrt only
            )
        )

        # --- Output directory ---
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                "OUTPUT_DIR",
                self.tr("Output directory"),
            )
        )

        # --- Output metadata ---
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                self.tr("Output folder"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "TIMESTEP_COUNT",
                self.tr("Number of timesteps processed"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "UTCI_COUNT",
                self.tr("Number of UTCI files created"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "PET_COUNT",
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
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG Calculation")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Determine weather mode
        weather_mode = self.parameterAsEnum(parameters, "WEATHER_SOURCE", context)
        is_single = weather_mode == self.WEATHER_SINGLE

        feedback.pushInfo(f"Mode: {'Single timestep' if is_single else 'EPW timeseries'}")

        # Step 1: Load surface data
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        surface = create_surface_from_parameters(parameters, context, self, feedback)

        if feedback.isCanceled():
            return {}

        # Step 2: Create Location
        feedback.setProgressText("Setting up location...")
        feedback.setProgress(10)

        location = create_location_from_parameters(parameters, surface, feedback)

        if feedback.isCanceled():
            return {}

        # Step 3: Load weather
        feedback.setProgressText("Loading weather data...")
        feedback.setProgress(15)

        if is_single:
            weather = create_weather_from_parameters(parameters, feedback)
            weather_series = [weather]
        else:
            epw_path = self.parameterAsFile(parameters, "EPW_FILE", context)
            start_dt = self.parameterAsDateTime(parameters, "START_DATE", context)
            end_dt = self.parameterAsDateTime(parameters, "END_DATE", context)
            hours_filter = self.parameterAsString(parameters, "HOURS_FILTER", context)

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

        # Step 4: Get options
        human = create_human_params_from_parameters(parameters)
        use_anisotropic_sky = self.parameterAsBool(parameters, "USE_ANISOTROPIC_SKY", context)
        conifer = self.parameterAsBool(parameters, "CONIFER", context)
        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)

        # Parse output components
        output_indices = self.parameterAsEnums(parameters, "OUTPUT_COMPONENTS", context)
        output_names = ["tmrt", "shadow", "kdown", "kup", "ldown", "lup"]
        selected_outputs = [output_names[i] for i in output_indices]
        feedback.pushInfo(f"Outputs: {', '.join(selected_outputs)}")

        # Load precomputed SVF if provided
        precomputed = None
        svf_dir = parameters.get("SVF_DIR")
        if svf_dir:
            feedback.pushInfo(f"Loading pre-computed SVF from {svf_dir}")
            try:
                precomputed = solweig.PrecomputedData.prepare(svf_dir=svf_dir)
            except Exception as e:
                feedback.reportError(
                    f"Could not load SVF from {svf_dir}: {e}",
                    fatalError=False,
                )

        # Check tiling
        enable_tiling = self.parameterAsBool(parameters, "ENABLE_TILING", context)
        if enable_tiling and len(weather_series) > 1:
            feedback.reportError(
                "Tiled processing is only supported for single timesteps. "
                "Disabling tiling and using standard processing.",
                fatalError=False,
            )
            enable_tiling = False

        if feedback.isCanceled():
            return {}

        # Auto-fallback: anisotropic sky requires precomputed shadow matrices
        if use_anisotropic_sky:
            has_shadow = (precomputed is not None and precomputed.shadow_matrices is not None) or (
                surface.shadow_matrices is not None
            )
            if not has_shadow:
                feedback.reportError(
                    "Anisotropic sky requires pre-computed SVF with shadow matrices. "
                    "Falling back to isotropic sky model. To use anisotropic sky, "
                    "first run 'Compute Sky View Factor' and provide the SVF directory.",
                    fatalError=False,
                )
                use_anisotropic_sky = False

        # Step 5: Validate inputs
        feedback.setProgressText("Validating inputs...")
        feedback.setProgress(20)

        try:
            warnings = solweig.validate_inputs(
                surface=surface,
                location=location,
                weather=weather_series[0],
                use_anisotropic_sky=use_anisotropic_sky,
                precomputed=precomputed,
            )
            for warning in warnings:
                feedback.reportError(f"Warning: {warning}", fatalError=False)
        except solweig.SolweigError as e:
            raise QgsProcessingException(f"Validation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        # Step 6: Run calculation
        os.makedirs(output_dir, exist_ok=True)

        if is_single and enable_tiling:
            results = self._run_tiled(
                solweig,
                surface,
                location,
                weather_series[0],
                human,
                use_anisotropic_sky,
                conifer,
                parameters,
                context,
                output_dir,
                feedback,
            )
        elif is_single:
            results = self._run_single(
                solweig,
                surface,
                location,
                weather_series[0],
                human,
                use_anisotropic_sky,
                conifer,
                precomputed,
                output_dir,
                selected_outputs,
                feedback,
            )
        else:
            results = self._run_timeseries(
                solweig,
                surface,
                location,
                weather_series,
                human,
                use_anisotropic_sky,
                conifer,
                precomputed,
                output_dir,
                selected_outputs,
                feedback,
            )

        if feedback.isCanceled():
            return {}

        calc_elapsed = time.time() - start_time
        feedback.pushInfo(f"Calculation complete: {len(results)} timestep(s) in {calc_elapsed:.1f}s")

        # Step 7: Post-processing
        utci_count = 0
        pet_count = 0

        compute_utci = self.parameterAsBool(parameters, "COMPUTE_UTCI", context)
        compute_pet = self.parameterAsBool(parameters, "COMPUTE_PET", context)

        if compute_utci:
            utci_count = self._run_utci(solweig, output_dir, weather_series, feedback)

        if compute_pet:
            pet_count = self._run_pet(solweig, output_dir, weather_series, human, feedback)

        # Step 8: Add first Tmrt to canvas (single timestep only)
        if is_single:
            tmrt_files = sorted(Path(output_dir, "tmrt").glob("tmrt_*.tif"))
            if tmrt_files:
                timestamp_str = weather_series[0].datetime.strftime("%Y-%m-%d %H:%M")
                self.add_raster_to_canvas(
                    path=str(tmrt_files[0]),
                    layer_name=f"Tmrt {timestamp_str}",
                    style="tmrt",
                    context=context,
                )

        # Report summary
        total_elapsed = time.time() - start_time
        self._report_summary(results, total_elapsed, utci_count, pet_count, output_dir, feedback)

        return {
            "OUTPUT_FOLDER": output_dir,
            "TIMESTEP_COUNT": len(results),
            "UTCI_COUNT": utci_count,
            "PET_COUNT": pet_count,
        }

    # -------------------------------------------------------------------------
    # Calculation helpers
    # -------------------------------------------------------------------------

    def _run_single(
        self,
        solweig,
        surface,
        location,
        weather,
        human,
        use_anisotropic_sky,
        conifer,
        precomputed,
        output_dir,
        selected_outputs,
        feedback,
    ) -> list:
        """Run single timestep with standard processing."""
        feedback.setProgressText("Calculating Mean Radiant Temperature...")
        feedback.setProgress(25)

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

        feedback.setProgress(80)

        # Save selected outputs to component subdirectories
        timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
        for component in selected_outputs:
            if hasattr(result, component):
                array = getattr(result, component)
                if array is not None:
                    comp_dir = os.path.join(output_dir, component)
                    os.makedirs(comp_dir, exist_ok=True)
                    filepath = os.path.join(comp_dir, f"{component}_{timestamp}.tif")
                    self.save_georeferenced_output(
                        array=array,
                        output_path=filepath,
                        geotransform=surface._geotransform,
                        crs_wkt=surface._crs_wkt,
                        feedback=feedback,
                    )

        feedback.setProgress(90)
        return [result]

    def _run_tiled(
        self,
        solweig,
        surface,
        location,
        weather,
        human,
        use_anisotropic_sky,
        conifer,
        parameters,
        context,
        output_dir,
        feedback,
    ) -> list:
        """Run single timestep with tiled processing."""
        rows, cols = surface.dsm.shape
        feedback.pushInfo(f"Raster size: {cols}x{rows} pixels")

        auto_tile = self.parameterAsBool(parameters, "AUTO_TILE_SIZE", context)
        if auto_tile:
            tile_size = self._calculate_auto_tile_size(rows, cols)
            feedback.pushInfo(f"Auto tile size: {tile_size}x{tile_size}")
        else:
            tile_size = self.parameterAsInt(parameters, "TILE_SIZE", context)
            feedback.pushInfo(f"Manual tile size: {tile_size}x{tile_size}")

        # If raster fits in single tile, use standard processing
        if rows <= tile_size and cols <= tile_size:
            feedback.pushInfo("Raster fits in single tile, using standard processing")
            try:
                result = solweig.calculate(
                    surface=surface,
                    location=location,
                    weather=weather,
                    human=human,
                    use_anisotropic_sky=use_anisotropic_sky,
                    conifer=conifer,
                )
            except Exception as e:
                raise QgsProcessingException(f"Calculation failed: {e}") from e
        else:
            feedback.setProgressText(f"Processing with {tile_size}x{tile_size} tiles...")
            feedback.setProgress(25)

            try:
                result = solweig.calculate_tiled(
                    surface=surface,
                    location=location,
                    weather=weather,
                    human=human,
                    tile_size=tile_size,
                    use_anisotropic_sky=use_anisotropic_sky,
                    conifer=conifer,
                )
            except Exception as e:
                raise QgsProcessingException(f"Tiled calculation failed: {e}") from e

        feedback.setProgress(80)

        # Save tmrt output in subdirectory
        timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
        tmrt_dir = os.path.join(output_dir, "tmrt")
        os.makedirs(tmrt_dir, exist_ok=True)
        filepath = os.path.join(tmrt_dir, f"tmrt_{timestamp}.tif")
        self.save_georeferenced_output(
            array=result.tmrt,
            output_path=filepath,
            geotransform=surface._geotransform,
            crs_wkt=surface._crs_wkt,
            feedback=feedback,
        )

        feedback.setProgress(90)
        return [result]

    def _run_timeseries(
        self,
        solweig,
        surface,
        location,
        weather_series,
        human,
        use_anisotropic_sky,
        conifer,
        precomputed,
        output_dir,
        selected_outputs,
        feedback,
    ) -> list:
        """Run multi-timestep timeseries calculation."""
        feedback.setProgressText(f"Running timeseries ({len(weather_series)} timesteps)...")
        feedback.setProgress(25)

        # Map step progress into the 25-80% range of the overall QGIS progress bar
        def _on_progress(current: int, total: int) -> None:
            pct = 25 + int(55 * current / total) if total > 0 else 25
            feedback.setProgress(pct)
            feedback.setProgressText(f"Timestep {current}/{total}...")

        try:
            results = solweig.calculate_timeseries(
                surface=surface,
                weather_series=weather_series,
                location=location,
                output_dir=output_dir,
                outputs=selected_outputs,
                use_anisotropic_sky=use_anisotropic_sky,
                conifer=conifer,
                precomputed=precomputed,
                progress_callback=_on_progress,
            )
        except Exception as e:
            raise QgsProcessingException(f"Timeseries calculation failed: {e}") from e

        feedback.setProgress(80)
        return results

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------

    def _run_utci(self, solweig, output_dir, weather_series, feedback) -> int:
        """Compute UTCI from Tmrt results."""
        feedback.setProgressText("Computing UTCI...")
        feedback.pushInfo("")
        feedback.pushInfo("Computing UTCI thermal comfort index...")

        utci_dir = os.path.join(output_dir, "utci")
        tmrt_dir = os.path.join(output_dir, "tmrt")

        # Map step progress into the 80-90% range
        def _on_progress(current: int, total: int) -> None:
            pct = 80 + int(10 * current / total) if total > 0 else 80
            feedback.setProgress(pct)
            feedback.setProgressText(f"UTCI {current}/{total}...")

        try:
            n_files = solweig.compute_utci(
                tmrt_dir=tmrt_dir,
                weather_series=weather_series,
                output_dir=utci_dir,
                progress_callback=_on_progress,
            )
        except Exception as e:
            raise QgsProcessingException(f"UTCI computation failed: {e}") from e

        feedback.pushInfo(f"UTCI: {n_files} files created in {utci_dir}")
        return n_files

    def _run_pet(self, solweig, output_dir, weather_series, human, feedback) -> int:
        """Compute PET from Tmrt results."""
        feedback.setProgressText("Computing PET (this may take a while)...")
        feedback.pushInfo("")
        feedback.pushInfo("Computing PET thermal comfort index...")
        feedback.pushInfo(
            f"Human params: {human.weight}kg, {human.height}m, {human.age}y, {human.activity}W, {human.clothing}clo"
        )

        estimated_time = len(weather_series) * 0.25
        feedback.pushInfo(f"Estimated time: {estimated_time:.0f}s ({len(weather_series)} timesteps)")

        tmrt_dir = os.path.join(output_dir, "tmrt")
        pet_dir = os.path.join(output_dir, "pet")

        # Map step progress into the 80-95% range (PET is slow)
        def _on_progress(current: int, total: int) -> None:
            pct = 80 + int(15 * current / total) if total > 0 else 80
            feedback.setProgress(pct)
            feedback.setProgressText(f"PET {current}/{total}...")

        try:
            n_files = solweig.compute_pet(
                tmrt_dir=tmrt_dir,
                weather_series=weather_series,
                output_dir=pet_dir,
                human=human,
                progress_callback=_on_progress,
            )
        except Exception as e:
            raise QgsProcessingException(f"PET computation failed: {e}") from e

        feedback.pushInfo(f"PET: {n_files} files created in {pet_dir}")
        return n_files

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_auto_tile_size(rows: int, cols: int) -> int:
        """Calculate optimal tile size based on raster dimensions."""
        if rows * cols > 4000 * 4000:
            return 1024
        elif rows * cols > 2000 * 2000:
            return min(rows, cols, 2048)
        else:
            return max(rows, cols)

    @staticmethod
    def _report_summary(
        results,
        elapsed,
        utci_count,
        pet_count,
        output_dir,
        feedback,
    ) -> None:
        """Report calculation summary statistics."""
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Calculation complete!")
        feedback.pushInfo(f"  Timesteps: {len(results)}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")

        if len(results) > 1:
            feedback.pushInfo(f"  Per timestep: {elapsed / len(results):.2f} seconds")

        # Tmrt statistics
        if results:
            all_valid = [r.tmrt[~np.isnan(r.tmrt)] for r in results if r.tmrt is not None]
            if all_valid:
                mean_tmrt = np.mean([arr.mean() for arr in all_valid])
                max_tmrt = max(arr.max() for arr in all_valid)
                min_tmrt = min(arr.min() for arr in all_valid)
                feedback.pushInfo(f"  Tmrt range: {min_tmrt:.1f}C - {max_tmrt:.1f}C")
                feedback.pushInfo(f"  Mean Tmrt: {mean_tmrt:.1f}C")

        if utci_count > 0:
            feedback.pushInfo(f"  UTCI files: {utci_count}")
        if pet_count > 0:
            feedback.pushInfo(f"  PET files: {pet_count}")

        feedback.pushInfo(f"  Output: {output_dir}")
        feedback.pushInfo("=" * 60)

        feedback.setProgress(100)
