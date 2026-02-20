"""
Unified SOLWEIG Calculation Algorithm

Supports single timestep, EPW timeseries, or UMEP met timeseries,
with optional tiled processing and UTCI/PET post-processing.
"""

from __future__ import annotations

import contextlib
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
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
)

from ...utils.converters import (
    build_materials_from_lc_mapping,
    create_human_params_from_parameters,
    create_location_from_parameters,
    create_physics_from_parameters,
    create_weather_from_parameters,
    load_prepared_surface,
    load_weather_from_epw,
    load_weather_from_umep_met,
)
from ...utils.parameters import (
    add_date_filter_parameters,
    add_epw_parameters,
    add_heat_threshold_parameters,
    add_human_body_parameters,
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_umep_met_parameters,
    add_vegetation_parameters,
    add_weather_parameters,
)
from ..base import SolweigAlgorithmBase


def _apply_saved_surface_settings(
    prepared_dir: str,
    parameters: dict,
    feedback: QgsProcessingFeedback,
) -> None:
    """Use ``parametersforsolweig.json`` saved during surface preparation.

    If the prepared surface directory contains a ``parametersforsolweig.json``
    and the user has not supplied a custom materials file, point the
    ``CUSTOM_MATERIALS_FILE`` parameter at it so that ``build_materials_from_lc_mapping``
    picks it up automatically.
    """
    params_path = os.path.join(prepared_dir, "parametersforsolweig.json")
    if os.path.exists(params_path):
        # Only apply if the user hasn't explicitly provided a custom file
        custom = parameters.get("CUSTOM_MATERIALS_FILE")
        if not custom:
            parameters["CUSTOM_MATERIALS_FILE"] = params_path
            feedback.pushInfo("Using saved parametersforsolweig.json from prepared surface")


class SolweigCalculationAlgorithm(SolweigAlgorithmBase):
    """
    Unified SOLWEIG calculation algorithm.

    Combines single timestep, timeseries, and optional UTCI/PET post-processing
    into a single Processing algorithm. Large rasters are automatically tiled.
    """

    # Weather source enum values
    WEATHER_SINGLE = 0
    WEATHER_EPW = 1
    WEATHER_UMEP = 2

    def name(self) -> str:
        return "solweig_calculation"

    def displayName(self) -> str:
        return self.tr("3. SOLWEIG Calculation")

    def shortHelpString(self) -> str:
        return self.tr(
            """Calculate Mean Radiant Temperature (Tmrt) with SOLWEIG.

<b>Surface data:</b>
Provide the <b>prepared surface directory</b> from "Prepare Surface Data".
All rasters (DSM, CDSM, DEM, walls) are loaded automatically.

<b>Weather modes:</b>
<ul>
<li><b>Single timestep:</b> Manual weather input for one date/time</li>
<li><b>EPW weather file:</b> Load hourly data from an EnergyPlus Weather file</li>
<li><b>UMEP met file:</b> Load from UMEP/SUEWS meteorological forcing files</li>
</ul>
For timeseries modes, thermal state (ground heating/cooling) accumulates
across timesteps for physically accurate results. Large rasters are
automatically processed using overlapping tiles to manage memory.

<b>Post-processing (optional):</b>
<ul>
<li><b>UTCI</b> - fast polynomial (~200 timesteps/sec)</li>
<li><b>PET</b> - iterative heat balance (~4 timesteps/sec, ~50x slower than UTCI)</li>
</ul>

<b>Outputs:</b>
GeoTIFF files organised into subfolders of the output directory:
<pre>
  output_dir/
    tmrt/        tmrt_YYYYMMDD_HHMM.tif  (always)
    shadow/      shadow_...              (if selected)
    kdown/       kdown_...               (if selected)
    utci/        utci_...                (if enabled)
    pet/         pet_...                 (if enabled)
</pre>

<b>Recommended workflow:</b>
<ol>
<li>Run "Prepare Surface Data" to align rasters and compute walls</li>
<li>Run "Compute Sky View Factor" on the prepared surface (optional, for anisotropic sky)</li>
<li>Run this algorithm with the prepared surface directory</li>
</ol>"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # --- Prepared surface directory (required) ---
        self.addParameter(
            QgsProcessingParameterFile(
                "PREPARED_SURFACE_DIR",
                self.tr("Prepared surface directory (from 'Prepare Surface Data')"),
                behavior=QgsProcessingParameterFile.Behavior.Folder,
            )
        )

        # --- Location ---
        add_location_parameters(self)

        # --- Weather source selector ---
        self.addParameter(
            QgsProcessingParameterEnum(
                "WEATHER_SOURCE",
                self.tr("Weather data source"),
                options=[
                    "Single timestep (manual entry)",
                    "EPW weather file (timeseries)",
                    "UMEP met file (timeseries)",
                ],
                defaultValue=self.WEATHER_EPW,
            )
        )

        # --- Single timestep weather (advanced - collapsed by default) ---
        add_weather_parameters(self)
        for name in ("DATETIME", "TEMPERATURE", "HUMIDITY", "GLOBAL_RADIATION", "WIND_SPEED", "PRESSURE"):
            param = self.parameterDefinition(name)
            if param:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)

        # --- EPW weather ---
        add_epw_parameters(self)

        # --- UMEP met weather ---
        add_umep_met_parameters(self)

        # --- Date/time filtering (shared by EPW and UMEP) ---
        add_date_filter_parameters(self)

        # --- Human parameters ---
        add_human_parameters(self)

        # --- Options ---
        add_options_parameters(self)

        # --- Vegetation (advanced) ---
        add_vegetation_parameters(self)

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
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)

        # --- Heat-stress thresholds (for UTCI exceedance summary grids) ---
        add_heat_threshold_parameters(self)

        # --- Output selection ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_TMRT",
                self.tr("Save Tmrt (Mean Radiant Temperature) per timestep"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_SHADOW",
                self.tr("Save shadow fraction"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_KDOWN",
                self.tr("Save Kdown (incoming shortwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_KUP",
                self.tr("Save Kup (reflected shortwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_LDOWN",
                self.tr("Save Ldown (incoming longwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_LUP",
                self.tr("Save Lup (emitted longwave)"),
                defaultValue=False,
            )
        )

        # --- Output directory ---
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                "OUTPUT_DIR",
                self.tr("Output directory (defaults to 'results' inside prepared surface dir)"),
                optional=True,
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
        mode_names = {0: "Single timestep", 1: "EPW timeseries", 2: "UMEP met timeseries"}
        feedback.pushInfo(f"Mode: {mode_names.get(weather_mode, 'Unknown')}")

        # Step 1: Load surface data from prepared directory
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        prepared_dir = self.parameterAsFile(parameters, "PREPARED_SURFACE_DIR", context)
        surface = load_prepared_surface(prepared_dir, feedback)

        # Apply saved settings from surface preparation as parameter defaults
        _apply_saved_surface_settings(prepared_dir, parameters, feedback)

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

        # Parse shared date/hour filters (used by both EPW and UMEP)
        start_qdt = self.parameterAsDateTime(parameters, "START_DATE", context)
        end_qdt = self.parameterAsDateTime(parameters, "END_DATE", context)
        start_dt = start_qdt if start_qdt.isValid() else None
        end_dt = end_qdt if end_qdt.isValid() else None
        hours_filter = self.parameterAsString(parameters, "HOURS_FILTER", context)

        if is_single:
            weather = create_weather_from_parameters(parameters, feedback)
            weather_series = [weather]
        elif weather_mode == self.WEATHER_EPW:
            epw_path = self.parameterAsFile(parameters, "EPW_FILE", context)
            weather_series = load_weather_from_epw(
                epw_path=epw_path,
                start_dt=start_dt,
                end_dt=end_dt,
                hours_filter=hours_filter,
                feedback=feedback,
            )
            if not weather_series:
                raise QgsProcessingException("No timesteps found in specified date range")
        elif weather_mode == self.WEATHER_UMEP:
            umep_path = self.parameterAsFile(parameters, "UMEP_MET_FILE", context)
            weather_series = load_weather_from_umep_met(
                met_path=umep_path,
                start_dt=start_dt,
                end_dt=end_dt,
                hours_filter=hours_filter,
                feedback=feedback,
            )
            if not weather_series:
                raise QgsProcessingException("No timesteps found in UMEP met file")

        if feedback.isCanceled():
            return {}

        # Step 4: Get options
        human = create_human_params_from_parameters(parameters)
        physics = create_physics_from_parameters(parameters)
        materials = build_materials_from_lc_mapping(parameters, context, self, feedback)
        use_anisotropic_sky = self.parameterAsBool(parameters, "USE_ANISOTROPIC_SKY", context)
        conifer = self.parameterAsBool(parameters, "CONIFER", context)
        max_shadow_distance_m = self.parameterAsDouble(parameters, "MAX_SHADOW_DISTANCE", context)
        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)

        # Default output to 'results/' inside prepared surface directory
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = os.path.join(prepared_dir, "results")
            feedback.pushInfo(f"Output directory: {output_dir} (inside prepared surface dir)")

        # Parse heat-stress thresholds
        heat_thresholds_day = self._parse_thresholds(self.parameterAsString(parameters, "HEAT_THRESHOLDS_DAY", context))
        heat_thresholds_night = self._parse_thresholds(
            self.parameterAsString(parameters, "HEAT_THRESHOLDS_NIGHT", context)
        )
        if heat_thresholds_day:
            feedback.pushInfo(f"Daytime UTCI thresholds: {heat_thresholds_day}")
        if heat_thresholds_night:
            feedback.pushInfo(f"Nighttime UTCI thresholds: {heat_thresholds_night}")

        # Parse output components
        selected_outputs = []
        if self.parameterAsBool(parameters, "OUTPUT_TMRT", context):
            selected_outputs.append("tmrt")
        for comp in ["shadow", "kdown", "kup", "ldown", "lup"]:
            if self.parameterAsBool(parameters, f"OUTPUT_{comp.upper()}", context):
                selected_outputs.append(comp)
        # UTCI/PET are now computed inline during the main calculation
        if self.parameterAsBool(parameters, "COMPUTE_UTCI", context):
            selected_outputs.append("utci")
        if self.parameterAsBool(parameters, "COMPUTE_PET", context):
            selected_outputs.append("pet")
        feedback.pushInfo(f"Outputs: {', '.join(selected_outputs)}")

        # Load precomputed SVF — check explicit SVF_DIR, then prepared surface dir
        precomputed = None
        svf_dir = parameters.get("SVF_DIR") or None
        if not svf_dir:
            # Auto-detect SVF in prepared surface directory
            svfs_path = os.path.join(prepared_dir, "svfs.zip")
            if os.path.exists(svfs_path):
                svf_dir = prepared_dir
                feedback.pushInfo("Auto-detected SVF in prepared surface directory")

        if svf_dir:
            feedback.pushInfo(f"Loading pre-computed SVF from {svf_dir}")
            try:
                precomputed = solweig.PrecomputedData.prepare(svf_dir=svf_dir)
            except Exception as e:
                feedback.reportError(
                    f"Could not load SVF from {svf_dir}: {e}",
                    fatalError=False,
                )

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

        # results is used for single/tiled paths; timeseries uses n_results + tmrt_stats
        results = None
        n_results = 0
        tmrt_stats = {}

        if is_single:
            results = self._run_single(
                solweig,
                surface,
                location,
                weather_series[0],
                human,
                use_anisotropic_sky,
                conifer,
                physics,
                precomputed,
                output_dir,
                selected_outputs,
                max_shadow_distance_m,
                materials,
                feedback,
            )
        else:
            n_results, tmrt_stats = self._run_timeseries(
                solweig,
                surface,
                location,
                weather_series,
                human,
                use_anisotropic_sky,
                conifer,
                physics,
                precomputed,
                output_dir,
                selected_outputs,
                max_shadow_distance_m,
                materials,
                heat_thresholds_day,
                heat_thresholds_night,
                feedback,
            )

        if feedback.isCanceled():
            return {}

        n_timesteps = n_results if results is None else len(results)
        calc_elapsed = time.time() - start_time
        feedback.pushInfo(f"Calculation complete: {n_timesteps} timestep(s) in {calc_elapsed:.1f}s")

        # Step 7: Add first Tmrt to canvas (single timestep only)
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
        utci_count = n_results if "utci" in selected_outputs else 0
        pet_count = n_results if "pet" in selected_outputs else 0
        if results is None:
            # Timeseries path: use incremental stats
            self._report_summary(n_results, total_elapsed, utci_count, pet_count, output_dir, feedback, tmrt_stats)
        else:
            # Single/tiled path: compute stats from results list
            stats = {}
            all_valid = [r.tmrt[~np.isnan(r.tmrt)] for r in results if r.tmrt is not None]
            if all_valid:
                stats = {
                    "mean": np.mean([arr.mean() for arr in all_valid]),
                    "min": float(min(arr.min() for arr in all_valid)),
                    "max": float(max(arr.max() for arr in all_valid)),
                }
            self._report_summary(len(results), total_elapsed, utci_count, pet_count, output_dir, feedback, stats)

        return {
            "OUTPUT_FOLDER": output_dir,
            "TIMESTEP_COUNT": n_timesteps,
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
        physics,
        precomputed,
        output_dir,
        selected_outputs,
        max_shadow_distance_m,
        materials,
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
                physics=physics,
                materials=materials,
                max_shadow_distance_m=max_shadow_distance_m,
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

    def _run_timeseries(
        self,
        solweig,
        surface,
        location,
        weather_series,
        human,
        use_anisotropic_sky,
        conifer,
        physics,
        precomputed,
        output_dir,
        selected_outputs,
        max_shadow_distance_m,
        materials,
        heat_thresholds_day,
        heat_thresholds_night,
        feedback,
    ) -> tuple[int, dict]:
        """Run multi-timestep timeseries with per-timestep progress.

        Delegates to solweig.calculate_timeseries() so QGIS and direct Python
        usage follow the exact same execution path and options.

        Returns (n_results, tmrt_stats).
        """
        n_steps = len(weather_series)
        feedback.setProgressText(f"Running timeseries ({n_steps} timesteps)...")
        feedback.setProgress(25)
        progress_state = {"completed": 0}

        def _qgis_progress(current: int, total: int) -> None:
            progress_state["completed"] = current
            if feedback.isCanceled():
                raise KeyboardInterrupt
            if current > 0 and weather_series:
                idx = min(current - 1, len(weather_series) - 1)
                timestamp_str = weather_series[idx].datetime.strftime("%Y-%m-%d %H:%M")
                feedback.setProgressText(f"Timestep {current}/{total} \u2014 {timestamp_str}")
            total_safe = max(total, 1)
            pct = 25 + int(55 * current / total_safe)
            feedback.setProgress(pct)

        summary = None
        try:
            summary = solweig.calculate_timeseries(
                surface=surface,
                weather_series=weather_series,
                location=location,
                human=human,
                precomputed=precomputed,
                use_anisotropic_sky=use_anisotropic_sky,
                conifer=conifer,
                physics=physics,
                materials=materials,
                max_shadow_distance_m=max_shadow_distance_m,
                output_dir=output_dir,
                outputs=selected_outputs,
                heat_thresholds_day=heat_thresholds_day or None,
                heat_thresholds_night=heat_thresholds_night or None,
                progress_callback=_qgis_progress,
            )
        except KeyboardInterrupt:
            feedback.pushInfo("Timeseries cancelled by user.")
        except Exception as e:
            raise QgsProcessingException(f"Timeseries calculation failed: {e}") from e

        feedback.setProgress(80)
        # If calculate_timeseries returned normally (summary is not None),
        # all timesteps completed.  The progress callback may report tile-level
        # counts when tiling is active, so use n_steps directly on success.
        if summary is not None:
            n_results = n_steps
        elif feedback.isCanceled():
            n_results = 0
        else:
            n_results = n_steps

        # Log the summary report
        if summary is not None and n_results > 0:
            feedback.pushInfo("")
            for line in summary.report().splitlines():
                feedback.pushInfo(line)

        tmrt_stats = self._compute_tmrt_stats_from_outputs(output_dir) if output_dir and n_results > 0 else {}
        return n_results, tmrt_stats

    @staticmethod
    def _compute_tmrt_stats_from_outputs(output_dir: str) -> dict:
        """Compute summary stats from saved Tmrt rasters (mean of per-file means)."""
        from solweig.io import load_raster

        tmrt_dir = Path(output_dir) / "tmrt"
        tmrt_files = sorted(tmrt_dir.glob("tmrt_*.tif"))
        if not tmrt_files:
            return {}

        mean_sum = 0.0
        mean_count = 0
        tmrt_min = np.inf
        tmrt_max = -np.inf

        for tif_path in tmrt_files:
            try:
                arr, *_ = load_raster(str(tif_path))
            except Exception:
                continue
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                continue
            mean_sum += float(np.mean(valid))
            mean_count += 1
            tmrt_min = min(tmrt_min, float(np.min(valid)))
            tmrt_max = max(tmrt_max, float(np.max(valid)))

        if mean_count == 0:
            return {}

        return {"mean": mean_sum / mean_count, "min": float(tmrt_min), "max": float(tmrt_max)}

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_thresholds(raw: str | None) -> list[float]:
        """Parse a comma-separated string of temperature thresholds into a list of floats."""
        if not raw or not raw.strip():
            return []
        values = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                with contextlib.suppress(ValueError):
                    values.append(float(part))
        return values

    @staticmethod
    def _report_summary(
        n_timesteps,
        elapsed,
        utci_count,
        pet_count,
        output_dir,
        feedback,
        tmrt_stats=None,
    ) -> None:
        """Report calculation summary statistics.

        Args:
            n_timesteps: Number of timesteps processed.
            elapsed: Total elapsed time in seconds.
            utci_count: Number of UTCI files created.
            pet_count: Number of PET files created.
            output_dir: Output directory path.
            feedback: QGIS feedback object.
            tmrt_stats: Dict with 'mean', 'min', 'max' Tmrt values (optional).
        """
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Calculation complete!")
        feedback.pushInfo(f"  Timesteps: {n_timesteps}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")

        if n_timesteps > 1:
            feedback.pushInfo(f"  Per timestep: {elapsed / n_timesteps:.2f} seconds")

        if tmrt_stats:
            feedback.pushInfo(f"  Tmrt range: {tmrt_stats['min']:.1f}C - {tmrt_stats['max']:.1f}C")
            feedback.pushInfo(f"  Mean Tmrt: {tmrt_stats['mean']:.1f}C")

        if utci_count > 0:
            feedback.pushInfo(f"  UTCI files: {utci_count}")
        if pet_count > 0:
            feedback.pushInfo(f"  PET files: {pet_count}")

        feedback.pushInfo(f"  Output: {output_dir}")
        feedback.pushInfo("=" * 60)

        feedback.setProgress(100)
