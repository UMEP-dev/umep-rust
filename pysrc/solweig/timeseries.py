"""Time series calculation with thermal state management."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .logging import get_logger
from .metadata import create_run_metadata, save_run_metadata
from .models import HumanParams, Location, ThermalState

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

    from .algorithms import sun_position as sp

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

            altmax_cache[day] = sunmaximum

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
    use_anisotropic_sky: bool = False,
    conifer: bool = False,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    output_dir: str | Path | None = None,
    outputs: list[str] | None = None,
) -> list[SolweigResult]:
    """
    Calculate Tmrt for a time series of weather data.

    Maintains thermal state across timesteps for accurate surface temperature
    modeling with thermal inertia (TsWaveDelay_2015a).

    This is a convenience function that manages state automatically. For custom
    control over state, use calculate() directly with the state parameter.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        weather_series: List of Weather objects in chronological order.
            The datetime of each Weather object determines the timestep size.
        location: Geographic location (lat, lon, UTC offset). If None, automatically
            extracted from surface's CRS metadata.
        config: Model configuration object. If provided, overrides individual parameters
            (use_anisotropic_sky, human, physics, materials, outputs).
        human: Human body parameters (absorption, posture, weight, height, etc.).
            If None, uses HumanParams defaults. Overridden by config.human if config provided.
        precomputed: Pre-computed SVF and/or shadow matrices. Optional.
        use_anisotropic_sky: Use anisotropic sky model. Overridden by config if provided.
        conifer: Treat vegetation as evergreen conifers (always leaf-on). Default False.
        physics: Physics parameters (Tree_settings, Posture geometry) from load_physics().
            Site-independent scientific constants. If None, uses bundled defaults.
            Overridden by config.physics if config provided.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
            Overridden by config.materials if config provided.
        output_dir: Directory to save results. If provided, results are saved
            incrementally as GeoTIFF files during calculation (recommended for
            long timeseries to avoid memory issues).
        outputs: Which outputs to save (e.g., ["tmrt", "shadow", "kdown"]).
            Only used if output_dir is provided. Overridden by config.outputs if config provided.

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

        # With custom human parameters
        results = calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            human=HumanParams(abs_k=0.65, weight=70),
            output_dir="output/",
        )

        # With custom physics (e.g., different tree transmissivity)
        physics = load_physics("custom_trees.json")
        results = calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            physics=physics,
            output_dir="output/",
        )

        # With landcover materials (requires land_cover grid in surface)
        materials = load_materials("site_materials.json")
        results = calculate_timeseries(
            surface=surface,
            weather_series=weather_list,
            materials=materials,
            output_dir="output/",
        )
    """
    if not weather_series:
        return []

    # Auto-extract location from surface if not provided
    if location is None:
        logger.info("Location not provided, auto-extracting from surface CRS...")
        location = Location.from_surface(surface)

    # Apply config if provided (overrides individual parameters)
    if config is not None:
        use_anisotropic_sky = config.use_anisotropic_sky
        if config.human is not None:
            human = config.human
        if config.physics is not None:
            physics = config.physics
        if config.materials is not None:
            materials = config.materials
        if config.outputs:
            outputs = config.outputs

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

    # Pre-calculate timestep size from first two entries (matching runner behavior)
    # The runner uses a fixed timestep_dec for all iterations, calculated upfront
    if len(weather_series) >= 2:
        dt0 = weather_series[0].datetime
        dt1 = weather_series[1].datetime
        state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

    # Progress reporting interval (log every N timesteps)
    report_interval = max(1, len(weather_series) // 10) if len(weather_series) > 20 else 1

    # Start timing
    start_time = time.time()
    last_report_time = start_time

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
        )

        # Carry forward state to next timestep
        if result.state is not None:
            state = result.state

        # Save incrementally if output_dir provided
        if output_dir is not None:
            result.to_geotiff(
                output_dir=output_dir,
                timestamp=weather.datetime,
                outputs=outputs,
                surface=surface,
            )

        results.append(result)

        # Log progress after processing
        if (i + 1) % report_interval == 0 or i == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            interval_time = current_time - last_report_time
            # For first report, we've processed 1 timestep; for subsequent, report_interval timesteps
            timesteps_processed = 1 if i == 0 else report_interval
            rate = timesteps_processed / interval_time if interval_time > 0 else 0

            logger.info(
                f"  Processed timestep {i + 1}/{len(weather_series)}: {weather.datetime.strftime('%Y-%m-%d %H:%M')} "
                f"[{rate:.2f} steps/s, {elapsed:.1f}s elapsed]"
            )
            last_report_time = current_time

    # Calculate total elapsed time
    total_time = time.time() - start_time
    overall_rate = len(results) / total_time if total_time > 0 else 0

    # Log summary statistics
    logger.info("=" * 60)
    logger.info(f"✓ Calculation complete: {len(results)} timesteps processed")
    logger.info(f"  Total time: {total_time:.1f}s ({overall_rate:.2f} steps/s)")
    if results:
        # Compute summary statistics
        mean_tmrt = sum(r.tmrt.mean() for r in results) / len(results)
        max_tmrt = max(r.tmrt.max() for r in results)
        min_tmrt = min(r.tmrt.min() for r in results)
        logger.info(f"  Tmrt range: {min_tmrt:.1f}°C - {max_tmrt:.1f}°C (mean: {mean_tmrt:.1f}°C)")

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
