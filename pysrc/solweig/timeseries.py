"""Time series calculation with thermal state management."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .metadata import create_run_metadata, save_run_metadata
from .models import HumanParams, Location, ThermalState
from .progress import ProgressReporter
from .solweig_logging import get_logger

logger = get_logger(__name__)


def _precompute_weather(weather_series: list, location: Location) -> None:
    """
    Pre-compute derived weather values for all timesteps efficiently.

    Optimizations:
    1. Compute max sun altitude (altmax) only once per unique day
    2. Pre-assign altmax to Weather objects to skip the 96-iteration loop

    This reduces compute_derived() from O(96) iterations to O(1) per timestep
    when multiple timesteps share the same day.

    Args:
        weather_series: List of Weather objects to process
        location: Geographic location for sun position calculations
    """
    if not weather_series:
        return

    from datetime import timedelta

    import numpy as np

    from .physics import sun_position as sp

    location_dict = location.to_sun_position_dict()

    # Step 1: Compute altmax once per unique day
    altmax_cache = {}  # date -> altmax

    for weather in weather_series:
        day = weather.datetime.date()
        if day not in altmax_cache:
            # Compute max sun altitude for this day (iterate in 15-min intervals)
            ymd = weather.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            sunmaximum = -90.0
            fifteen_min = 15.0 / 1440.0  # 15 minutes as fraction of day

            for step in range(96):  # 24 hours * 4 (15-min intervals)
                step_time = ymd + timedelta(days=step * fifteen_min)
                time_dict_step = {
                    "year": step_time.year,
                    "month": step_time.month,
                    "day": step_time.day,
                    "hour": step_time.hour,
                    "min": step_time.minute,
                    "sec": 0,
                    "UTC": location.utc_offset,
                }
                sun_step = sp.sun_position(time_dict_step, location_dict)
                zenith_step = sun_step["zenith"]
                zenith_val = (
                    float(np.asarray(zenith_step).flat[0]) if hasattr(zenith_step, "__iter__") else float(zenith_step)
                )
                altitude_step = 90.0 - zenith_val
                if altitude_step > sunmaximum:
                    sunmaximum = altitude_step

            altmax_cache[day] = max(sunmaximum, 0.0)

    # Step 2: Pre-assign altmax to each weather object
    for weather in weather_series:
        day = weather.datetime.date()
        weather.precomputed_altmax = altmax_cache[day]

    # Step 3: Compute derived values (now fast since altmax is cached)
    for weather in weather_series:
        if not weather._derived_computed:
            weather.compute_derived(location)


if TYPE_CHECKING:
    from .models import (
        ModelConfig,
        PrecomputedData,
        SolweigResult,
        SurfaceData,
        Weather,
    )


def calculate_timeseries(
    surface: SurfaceData,
    weather_series: list[Weather],
    location: Location | None = None,
    config: ModelConfig | None = None,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool | None = None,
    conifer: bool = False,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    wall_material: str | None = None,
    max_shadow_distance_m: float | None = None,
    output_dir: str | Path | None = None,
    outputs: list[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[SolweigResult]:
    """
    Calculate Tmrt for a time series of weather data.

    Maintains thermal state across timesteps for accurate surface temperature
    modeling with thermal inertia (TsWaveDelay_2015a).

    Large rasters are automatically processed using overlapping tiles to
    manage memory. The tile size and buffer distance are computed dynamically
    from available GPU/RAM resources and the maximum building height in the DSM,
    ensuring accurate shadows at tile boundaries without wasting overlap on
    short buildings.

    This is a convenience function that manages state automatically. For custom
    control over state, use calculate() directly with the state parameter.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        weather_series: List of Weather objects in chronological order.
            The datetime of each Weather object determines the timestep size.
        location: Geographic location (lat, lon, UTC offset). If None, automatically
            extracted from surface's CRS metadata.
        config: Model configuration object providing base settings.
            Explicit parameters override config values when provided.
        human: Human body parameters (absorption, posture, weight, height, etc.).
            If None, uses config.human or HumanParams defaults.
        precomputed: Pre-computed SVF and/or shadow matrices. Optional.
        use_anisotropic_sky: Use anisotropic sky model.
            If None, uses config.use_anisotropic_sky (default True).
        conifer: Treat vegetation as evergreen conifers (always leaf-on). Default False.
        physics: Physics parameters (Tree_settings, Posture geometry) from load_physics().
            Site-independent scientific constants. If None, uses config.physics or bundled defaults.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
            If None, uses config.materials.
        wall_material: Wall material type for temperature model.
            One of "brick", "concrete", "wood", "cobblestone" (case-insensitive).
            If None (default), uses generic wall params from materials JSON.
        max_shadow_distance_m: Maximum shadow reach in metres (default 500.0).
            Caps shadow ray computation distance and serves as the tile overlap
            buffer for automatic tiled processing of large rasters. If None,
            uses config.max_shadow_distance_m or 500.0.
        output_dir: Directory to save results. If provided, results are saved
            incrementally as GeoTIFF files during calculation (recommended for
            long timeseries to avoid memory issues).
        outputs: Which outputs to save (e.g., ["tmrt", "shadow", "kdown"]).
            Only used if output_dir is provided. If None, uses config.outputs or ["tmrt"].
        progress_callback: Optional callback(current_step, total_steps) called after
            each timestep. If None, a tqdm progress bar is shown automatically.

    Returns:
        List of SolweigResult objects, one per timestep.
        Each result includes the thermal state at that timestep.
        Note: UTCI and PET fields will be None. Use compute_utci() or compute_pet()
        for post-processing thermal comfort indices.

    Example:
        # Run time series with all defaults
        results = calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            output_dir="output/",
        )

        # With config as base, explicit param override
        config = ModelConfig(use_anisotropic_sky=True)
        results = calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            config=config,
            use_anisotropic_sky=False,  # Explicit param wins
            output_dir="output/",
        )
    """
    if not weather_series:
        return []

    # Auto-extract location from surface if not provided
    if location is None:
        logger.warning(
            "Location not provided - auto-extracting from surface CRS.\n"
            "⚠️  UTC offset will default to 0 if not specified, which may cause incorrect sun positions.\n"
            "   Recommend: provide location explicitly with correct UTC offset."
        )
        location = Location.from_surface(surface)

    # Build effective configuration: explicit params override config
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_outputs = outputs
    effective_max_shadow = max_shadow_distance_m

    if config is not None:
        # Use config values as fallback for None parameters
        if effective_aniso is None:
            effective_aniso = config.use_anisotropic_sky
        if effective_human is None:
            effective_human = config.human
        if effective_physics is None:
            effective_physics = config.physics
        if effective_materials is None:
            effective_materials = config.materials
        if effective_outputs is None and config.outputs:
            effective_outputs = config.outputs
        if effective_max_shadow is None:
            effective_max_shadow = config.max_shadow_distance_m

        # Debug log when explicit params override config
        overrides = []
        if use_anisotropic_sky is not None and use_anisotropic_sky != config.use_anisotropic_sky:
            overrides.append(f"use_anisotropic_sky={use_anisotropic_sky}")
        if human is not None and config.human is not None:
            overrides.append("human")
        if physics is not None and config.physics is not None:
            overrides.append("physics")
        if materials is not None and config.materials is not None:
            overrides.append("materials")
        if outputs is not None and config.outputs:
            overrides.append("outputs")
        if max_shadow_distance_m is not None and max_shadow_distance_m != config.max_shadow_distance_m:
            overrides.append(f"max_shadow_distance_m={max_shadow_distance_m}")
        if overrides:
            logger.debug(f"Explicit params override config: {', '.join(overrides)}")

    # Apply defaults for anything still None
    if effective_aniso is None:
        effective_aniso = False
    # Auto-load bundled UMEP JSON as default materials (single source of truth)
    if effective_materials is None:
        from .loaders import load_params

        effective_materials = load_params()

    # Assign back for use in the rest of the function
    use_anisotropic_sky = effective_aniso
    human = effective_human
    physics = effective_physics
    materials = effective_materials
    outputs = effective_outputs

    # Fill NaN in surface layers (idempotent — skipped if already done)
    surface.fill_nan()

    # Auto-tile large rasters transparently
    from .tiling import _should_use_tiling

    if _should_use_tiling(surface.shape[0], surface.shape[1]):
        from .tiling import calculate_timeseries_tiled

        logger.info(
            f"Raster size {surface.dsm.shape[1]}×{surface.dsm.shape[0]} exceeds tiling threshold — "
            "switching to tiled processing."
        )
        return calculate_timeseries_tiled(
            surface=surface,
            weather_series=weather_series,
            location=location,
            human=human,
            precomputed=precomputed,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
            physics=physics,
            materials=materials,
            wall_material=wall_material,
            max_shadow_distance_m=effective_max_shadow,
            output_dir=output_dir,
            outputs=outputs,
            progress_callback=progress_callback,
        )

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("Starting SOLWEIG timeseries calculation")
    logger.info(f"  Grid size: {surface.dsm.shape[1]}×{surface.dsm.shape[0]} pixels")
    logger.info(f"  Timesteps: {len(weather_series)}")
    start_str = weather_series[0].datetime.strftime("%Y-%m-%d %H:%M")
    end_str = weather_series[-1].datetime.strftime("%Y-%m-%d %H:%M")
    logger.info(f"  Period: {start_str} → {end_str}")
    logger.info(f"  Location: {location.latitude:.2f}°N, {location.longitude:.2f}°E")

    options = []
    if use_anisotropic_sky:
        options.append("anisotropic sky")
    if precomputed is not None:
        options.append("precomputed SVF")
    if options:
        logger.info(f"  Options: {', '.join(options)}")

    if output_dir is not None:
        logger.info(f"  Auto-save: {output_dir} ({', '.join(outputs or ['tmrt'])})")
    logger.info("=" * 60)

    # Create output directory if needed
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Default outputs
    if output_dir is not None and outputs is None:
        outputs = ["tmrt"]

    # Import calculate here to avoid circular import
    from .api import calculate

    # Pre-compute derived weather values in parallel (sun position, radiation split)
    # This is ~4x faster than computing sequentially in the main loop
    logger.info("Pre-computing sun positions and radiation splits...")
    precompute_start = time.time()
    _precompute_weather(weather_series, location)
    precompute_time = time.time() - precompute_start
    logger.info(f"  Pre-computed {len(weather_series)} timesteps in {precompute_time:.1f}s")

    results = []
    state = ThermalState.initial(surface.shape)

    # Incremental stats accumulators (avoids iterating all results for summary)
    _tmrt_sum = 0.0
    _tmrt_max = -np.inf
    _tmrt_min = np.inf
    _tmrt_count = 0

    # Pre-calculate timestep size from first two entries (matching runner behavior)
    # The runner uses a fixed timestep_dec for all iterations, calculated upfront
    if len(weather_series) >= 2:
        dt0 = weather_series[0].datetime
        dt1 = weather_series[1].datetime
        state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

    # Pre-create buffer pool for array reuse across timesteps
    _ = surface.get_buffer_pool()

    # Set up progress reporting (caller callback suppresses tqdm)
    n_steps = len(weather_series)
    _progress = None if progress_callback is not None else ProgressReporter(total=n_steps, desc="SOLWEIG timeseries")

    # Start timing
    start_time = time.time()

    for i, weather in enumerate(weather_series):
        # Process timestep
        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=human,
            precomputed=precomputed,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
            state=state,
            physics=physics,
            materials=materials,
            wall_material=wall_material,
            max_shadow_distance_m=effective_max_shadow,
        )

        # Carry forward state to next timestep
        if result.state is not None:
            state = result.state
            result.state = None  # Free state arrays (~23 MB); state managed externally

        # Save incrementally if output_dir provided
        if output_dir is not None:
            result.to_geotiff(
                output_dir=output_dir,
                timestamp=weather.datetime,
                outputs=outputs,
                surface=surface,
            )

        results.append(result)

        # Update incremental stats
        _valid = result.tmrt[np.isfinite(result.tmrt)]
        if _valid.size > 0:
            _tmrt_sum += _valid.sum()
            _tmrt_count += _valid.size
            _tmrt_max = max(_tmrt_max, float(_valid.max()))
            _tmrt_min = min(_tmrt_min, float(_valid.min()))

        # Report progress
        if progress_callback is not None:
            progress_callback(i + 1, n_steps)
        elif _progress is not None:
            _progress.update(1)

    # Close progress bar
    if _progress is not None:
        _progress.close()

    # Calculate total elapsed time
    total_time = time.time() - start_time
    overall_rate = len(results) / total_time if total_time > 0 else 0

    # Log summary statistics
    logger.info("=" * 60)
    logger.info(f"✓ Calculation complete: {len(results)} timesteps processed")
    logger.info(f"  Total time: {total_time:.1f}s ({overall_rate:.2f} steps/s)")
    if _tmrt_count > 0:
        mean_tmrt = _tmrt_sum / _tmrt_count
        logger.info(f"  Tmrt range: {_tmrt_min:.1f}°C - {_tmrt_max:.1f}°C (mean: {mean_tmrt:.1f}°C)")

    if output_dir is not None and outputs is not None:
        file_count = len(results) * len(outputs)
        logger.info(f"  Files saved: {file_count} GeoTIFFs in {output_dir}")
    logger.info("=" * 60)

    # Save run metadata if output_dir is provided
    if output_dir is not None:
        metadata = create_run_metadata(
            surface=surface,
            location=location,
            weather_series=weather_series,
            human=human,
            physics=physics,
            materials=materials,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
            output_dir=output_dir,
            outputs=outputs,
        )
        save_run_metadata(metadata, output_dir)

    return results
