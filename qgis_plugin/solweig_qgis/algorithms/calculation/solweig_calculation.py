"""
Unified SOLWEIG Calculation Algorithm

Supports single timestep, EPW timeseries, or UMEP met timeseries,
with optional tiled processing and UTCI/PET post-processing.
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
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
)

from ...utils.converters import (
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
    add_human_body_parameters,
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_umep_met_parameters,
    add_vegetation_parameters,
    add_weather_parameters,
)
from ..base import SolweigAlgorithmBase


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
                behavior=QgsProcessingParameterFile.Folder,
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
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)

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
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)

        # --- Output selection (Tmrt always saved) ---
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
        use_anisotropic_sky = self.parameterAsBool(parameters, "USE_ANISOTROPIC_SKY", context)
        conifer = self.parameterAsBool(parameters, "CONIFER", context)
        max_shadow_distance_m = self.parameterAsDouble(parameters, "MAX_SHADOW_DISTANCE", context)
        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)

        # Default output to 'results/' inside prepared surface directory
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = os.path.join(prepared_dir, "results")
            feedback.pushInfo(f"Output directory: {output_dir} (inside prepared surface dir)")

        # Parse output components (tmrt always saved)
        selected_outputs = ["tmrt"]
        for comp in ["shadow", "kdown", "kup", "ldown", "lup"]:
            if self.parameterAsBool(parameters, f"OUTPUT_{comp.upper()}", context):
                selected_outputs.append(comp)
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
                feedback,
            )

        if feedback.isCanceled():
            return {}

        n_timesteps = n_results if results is None else len(results)
        calc_elapsed = time.time() - start_time
        feedback.pushInfo(f"Calculation complete: {n_timesteps} timestep(s) in {calc_elapsed:.1f}s")

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
        feedback,
    ) -> tuple[int, dict]:
        """Run multi-timestep timeseries with per-timestep progress.

        Loops over timesteps using solweig.calculate() directly instead of
        delegating to calculate_timeseries(), so we have full control over
        the QGIS progress bar and cancellation between timesteps.

        Returns (n_results, tmrt_stats) instead of a results list to avoid
        accumulating all result arrays in memory (~46 MB per timestep).
        """
        n_steps = len(weather_series)

        # Auto-tile large rasters for memory efficiency and accurate shadows
        from solweig.tiling import _should_use_tiling

        if _should_use_tiling(*surface.shape):
            return self._run_timeseries_tiled(
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
                feedback,
            )

        feedback.setProgressText(f"Running timeseries ({n_steps} timesteps)...")
        feedback.setProgress(25)

        # Pre-compute sun positions and radiation splits for all timesteps.
        # Without this, each calculate() call independently computes altmax
        # (96 sun-position iterations per day per timestep), creating a CPU
        # bottleneck between GPU shadow dispatches.
        feedback.pushInfo("Pre-computing sun positions and radiation splits...")
        from solweig.timeseries import _precompute_weather

        _precompute_weather(weather_series, location)
        feedback.pushInfo(f"  Pre-computed {n_steps} timesteps")

        # Initialize thermal state for accurate ground temperature modelling
        from solweig.models.state import ThermalState

        state = ThermalState.initial(surface.dsm.shape)
        if n_steps >= 2:
            dt0 = weather_series[0].datetime
            dt1 = weather_series[1].datetime
            state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

        # Incremental stats (avoid accumulating all results in memory)
        n_results = 0
        tmrt_sum = 0.0
        tmrt_max = -np.inf
        tmrt_min = np.inf
        tmrt_count = 0

        for i, weather in enumerate(weather_series):
            if feedback.isCanceled():
                break

            feedback.setProgressText(f"Timestep {i + 1}/{n_steps} \u2014 {weather.datetime.strftime('%Y-%m-%d %H:%M')}")

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
                    state=state,
                    max_shadow_distance_m=max_shadow_distance_m,
                )
            except Exception as e:
                raise QgsProcessingException(
                    f"Calculation failed at timestep {i + 1}/{n_steps} ({weather.datetime}): {e}"
                ) from e

            # Carry forward thermal state; free state arrays from result
            if result.state is not None:
                state = result.state
                result.state = None

            # Save outputs incrementally (no per-file logging)
            timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
            for component in selected_outputs:
                array = getattr(result, component, None)
                if array is not None:
                    comp_dir = os.path.join(output_dir, component)
                    os.makedirs(comp_dir, exist_ok=True)
                    filepath = os.path.join(comp_dir, f"{component}_{timestamp}.tif")
                    self.save_georeferenced_output(
                        array=array,
                        output_path=filepath,
                        geotransform=surface._geotransform,
                        crs_wkt=surface._crs_wkt,
                    )

            # Update incremental stats
            valid = result.tmrt[np.isfinite(result.tmrt)]
            if valid.size > 0:
                tmrt_sum += valid.sum()
                tmrt_count += valid.size
                tmrt_max = max(tmrt_max, float(valid.max()))
                tmrt_min = min(tmrt_min, float(valid.min()))

            n_results += 1

            # Update progress bar (25-80% range)
            pct = 25 + int(55 * (i + 1) / n_steps)
            feedback.setProgress(pct)

        feedback.setProgress(80)

        tmrt_stats = {}
        if tmrt_count > 0:
            tmrt_stats = {
                "mean": tmrt_sum / tmrt_count,
                "min": tmrt_min,
                "max": tmrt_max,
            }

        return n_results, tmrt_stats

    def _run_timeseries_tiled(
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
        feedback,
    ) -> tuple[int, dict]:
        """Run multi-timestep timeseries with tiled processing for large rasters.

        Tiles the raster with overlapping buffers and iterates
        timestep -> tile, updating the QGIS progress bar with both
        the current timestamp and tile number.

        Returns (n_results, tmrt_stats), same as _run_timeseries().
        """
        from solweig.models.state import ThermalState
        from solweig.tiling import (
            _calculate_auto_tile_size,
            _extract_tile_surface,
            _merge_tile_state,
            _slice_tile_precomputed,
            _slice_tile_state,
            _write_tile_result,
            calculate_buffer_distance,
            generate_tiles,
            validate_tile_size,
        )
        from solweig.timeseries import _precompute_weather

        n_steps = len(weather_series)
        rows, cols = surface.shape
        pixel_size = surface.pixel_size

        # Height-aware buffer: use actual max building height instead of worst-case
        max_height = float(np.nanmax(surface.dsm)) if surface.dsm is not None else 0.0
        buffer_m = calculate_buffer_distance(max_height, max_shadow_distance_m=max_shadow_distance_m)
        buffer_pixels = int(np.ceil(buffer_m / pixel_size))
        tile_size = _calculate_auto_tile_size(rows, cols)
        adjusted_tile_size, warning = validate_tile_size(tile_size, buffer_pixels, pixel_size)
        if warning:
            feedback.reportError(f"Tile warning: {warning}", fatalError=False)

        tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
        n_tiles = len(tiles)
        total_work = n_steps * n_tiles

        feedback.setProgressText(f"Running tiled timeseries ({n_steps} timesteps, {n_tiles} tiles)...")
        feedback.setProgress(25)
        feedback.pushInfo(
            f"Large raster ({cols}\u00d7{rows}) \u2014 using {n_tiles} tiles "
            f"(size={adjusted_tile_size}, buffer={buffer_m:.0f}m from max height {max_height:.1f}m)"
        )

        # Pre-compute sun positions and radiation splits
        feedback.pushInfo("Pre-computing sun positions and radiation splits...")
        _precompute_weather(weather_series, location)
        feedback.pushInfo(f"  Pre-computed {n_steps} timesteps")

        # Initialize thermal state
        state = ThermalState.initial(surface.shape)
        if n_steps >= 2:
            dt0 = weather_series[0].datetime
            dt1 = weather_series[1].datetime
            state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

        # Incremental stats
        n_results = 0
        tmrt_sum = 0.0
        tmrt_max = -np.inf
        tmrt_min = np.inf
        tmrt_count = 0

        from solweig.computation import _nighttime_result

        for t_idx, weather in enumerate(weather_series):
            if feedback.isCanceled():
                break

            timestamp_str = weather.datetime.strftime("%Y-%m-%d %H:%M")

            # Nighttime shortcut: skip tiling entirely when sun is below horizon
            if weather.sun_altitude <= 0:
                feedback.setProgressText(f"Timestep {t_idx + 1}/{n_steps} ({timestamp_str}) \u2014 nighttime")
                night_result = _nighttime_result(surface, weather, state, None)
                if night_result.state is not None:
                    state = night_result.state

                # Save outputs
                timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
                output_map = {
                    "tmrt": night_result.tmrt,
                    "shadow": night_result.shadow,
                    "kdown": night_result.kdown,
                    "kup": night_result.kup,
                    "ldown": night_result.ldown,
                    "lup": night_result.lup,
                }
                for component in selected_outputs:
                    array = output_map.get(component)
                    if array is not None:
                        comp_dir = os.path.join(output_dir, component)
                        os.makedirs(comp_dir, exist_ok=True)
                        filepath = os.path.join(comp_dir, f"{component}_{timestamp}.tif")
                        self.save_georeferenced_output(
                            array=array,
                            output_path=filepath,
                            geotransform=surface._geotransform,
                            crs_wkt=surface._crs_wkt,
                        )

                valid = night_result.tmrt[np.isfinite(night_result.tmrt)]
                if valid.size > 0:
                    tmrt_sum += valid.sum()
                    tmrt_count += valid.size
                    tmrt_max = max(tmrt_max, float(valid.max()))
                    tmrt_min = min(tmrt_min, float(valid.min()))

                n_results += 1

                # Advance progress by all tiles for this timestep
                step = (t_idx + 1) * n_tiles
                pct = 25 + int(55 * step / total_work)
                feedback.setProgress(pct)
                continue

            # Initialize output arrays for this timestep
            tmrt_out = np.full((rows, cols), np.nan, dtype=np.float32)
            shadow_out = np.full((rows, cols), np.nan, dtype=np.float32)
            kdown_out = np.full((rows, cols), np.nan, dtype=np.float32)
            kup_out = np.full((rows, cols), np.nan, dtype=np.float32)
            ldown_out = np.full((rows, cols), np.nan, dtype=np.float32)
            lup_out = np.full((rows, cols), np.nan, dtype=np.float32)

            for tile_idx, tile in enumerate(tiles):
                if feedback.isCanceled():
                    break

                feedback.setProgressText(
                    f"Timestep {t_idx + 1}/{n_steps} ({timestamp_str}) \u2014 Tile {tile_idx + 1}/{n_tiles}"
                )

                tile_surface = _extract_tile_surface(surface, tile, pixel_size)
                tile_precomputed = _slice_tile_precomputed(precomputed, tile)
                tile_state = _slice_tile_state(state, tile)

                try:
                    tile_result = solweig.calculate(
                        surface=tile_surface,
                        location=location,
                        weather=weather,
                        human=human,
                        precomputed=tile_precomputed,
                        use_anisotropic_sky=use_anisotropic_sky,
                        conifer=conifer,
                        physics=physics,
                        state=tile_state,
                        max_shadow_distance_m=max_shadow_distance_m,
                    )
                except Exception as e:
                    raise QgsProcessingException(
                        f"Calculation failed at timestep {t_idx + 1}/{n_steps} "
                        f"({timestamp_str}), tile {tile_idx + 1}/{n_tiles}: {e}"
                    ) from e

                _write_tile_result(
                    tile_result,
                    tile,
                    tmrt_out,
                    shadow_out,
                    kdown_out,
                    kup_out,
                    ldown_out,
                    lup_out,
                )

                if tile_result.state is not None:
                    _merge_tile_state(tile_result.state, tile, state)

                # Update progress (25-80% range)
                step = t_idx * n_tiles + tile_idx + 1
                pct = 25 + int(55 * step / total_work)
                feedback.setProgress(pct)

            # Save outputs for this timestep
            timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
            output_map = {
                "tmrt": tmrt_out,
                "shadow": shadow_out,
                "kdown": kdown_out,
                "kup": kup_out,
                "ldown": ldown_out,
                "lup": lup_out,
            }
            for component in selected_outputs:
                array = output_map.get(component)
                if array is not None:
                    comp_dir = os.path.join(output_dir, component)
                    os.makedirs(comp_dir, exist_ok=True)
                    filepath = os.path.join(comp_dir, f"{component}_{timestamp}.tif")
                    self.save_georeferenced_output(
                        array=array,
                        output_path=filepath,
                        geotransform=surface._geotransform,
                        crs_wkt=surface._crs_wkt,
                    )

            # Update incremental stats
            valid = tmrt_out[np.isfinite(tmrt_out)]
            if valid.size > 0:
                tmrt_sum += valid.sum()
                tmrt_count += valid.size
                tmrt_max = max(tmrt_max, float(valid.max()))
                tmrt_min = min(tmrt_min, float(valid.min()))

            n_results += 1

        feedback.setProgress(80)

        tmrt_stats = {}
        if tmrt_count > 0:
            tmrt_stats = {
                "mean": tmrt_sum / tmrt_count,
                "min": tmrt_min,
                "max": tmrt_max,
            }

        return n_results, tmrt_stats

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------

    def _run_utci(self, solweig, output_dir, weather_series, feedback) -> int:
        """Compute UTCI from saved Tmrt GeoTIFFs with per-file progress."""
        from osgeo import gdal

        feedback.setProgressText("Computing UTCI...")
        feedback.pushInfo("")
        feedback.pushInfo("Computing UTCI thermal comfort index...")

        tmrt_dir = os.path.join(output_dir, "tmrt")
        utci_dir = os.path.join(output_dir, "utci")
        os.makedirs(utci_dir, exist_ok=True)

        n_steps = len(weather_series)
        processed = 0

        for i, weather in enumerate(weather_series):
            if feedback.isCanceled():
                break

            timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
            tmrt_path = os.path.join(tmrt_dir, f"tmrt_{timestamp}.tif")

            if not os.path.exists(tmrt_path):
                continue

            # Load Tmrt GeoTIFF via GDAL
            ds = gdal.Open(tmrt_path)
            tmrt = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            geotransform = list(ds.GetGeoTransform())
            crs_wkt = ds.GetProjection()
            ds = None

            # Compute UTCI
            utci = solweig.compute_utci_grid(tmrt, weather.ta, weather.rh, weather.ws)

            # Save
            utci_path = os.path.join(utci_dir, f"utci_{timestamp}.tif")
            self.save_georeferenced_output(utci, utci_path, geotransform, crs_wkt)
            processed += 1

            # Progress (80-90% range)
            pct = 80 + int(10 * (i + 1) / n_steps)
            feedback.setProgress(pct)

        feedback.pushInfo(f"UTCI: {processed} files created in {utci_dir}")
        return processed

    def _run_pet(self, solweig, output_dir, weather_series, human, feedback) -> int:
        """Compute PET from saved Tmrt GeoTIFFs with per-file progress."""
        from osgeo import gdal

        feedback.setProgressText("Computing PET (this may take a while)...")
        feedback.pushInfo("")
        feedback.pushInfo("Computing PET thermal comfort index...")
        feedback.pushInfo(
            f"Human params: {human.weight}kg, {human.height}m, {human.age}y, {human.activity}W, {human.clothing}clo"
        )

        tmrt_dir = os.path.join(output_dir, "tmrt")
        pet_dir = os.path.join(output_dir, "pet")
        os.makedirs(pet_dir, exist_ok=True)

        n_steps = len(weather_series)
        processed = 0

        for i, weather in enumerate(weather_series):
            if feedback.isCanceled():
                break

            timestamp = weather.datetime.strftime("%Y%m%d_%H%M")
            tmrt_path = os.path.join(tmrt_dir, f"tmrt_{timestamp}.tif")

            if not os.path.exists(tmrt_path):
                continue

            # Load Tmrt GeoTIFF via GDAL
            ds = gdal.Open(tmrt_path)
            tmrt = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            geotransform = list(ds.GetGeoTransform())
            crs_wkt = ds.GetProjection()
            ds = None

            # Compute PET
            pet = solweig.compute_pet_grid(tmrt, weather.ta, weather.rh, weather.ws, human)

            # Save
            pet_path = os.path.join(pet_dir, f"pet_{timestamp}.tif")
            self.save_georeferenced_output(pet, pet_path, geotransform, crs_wkt)
            processed += 1

            # Progress (90-98% range)
            pct = 90 + int(8 * (i + 1) / n_steps)
            feedback.setProgress(pct)

        feedback.pushInfo(f"PET: {processed} files created in {pet_dir}")
        return processed

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

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
