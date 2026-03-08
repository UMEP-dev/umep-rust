"""Shared helpers for timeseries and tiled orchestration paths.

These thin wrappers deduplicate the per-timestep result processing
(UTCI/PET computation, file output, array release) and the
post-loop summary finalization (logging, GeoTIFF export, metadata)
that are identical in both ``timeseries._calculate_timeseries`` and
``tiling._calculate_timeseries_tiled``.

Internal module -- not part of the public API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .metadata import create_run_metadata, save_run_metadata
from .postprocess import compute_pet_grid, compute_utci_grid
from .solweig_logging import get_logger

if TYPE_CHECKING:
    from .models import HumanParams, SolweigResult, SurfaceData, Weather
    from .output_async import AsyncGeoTiffWriter
    from .summary import GridAccumulator, TimeseriesSummary

logger = get_logger(__name__)


def resolve_config_params(
    *,
    config,
    use_anisotropic_sky: bool | None,
    human,
    physics,
    materials,
    outputs: list[str] | None,
    max_shadow_distance_m: float | None,
    tile_workers: int | None,
    tile_queue_depth: int | None,
    prefetch_tiles: bool | None,
) -> dict:
    """Resolve effective parameters: explicit params > config fallback > defaults.

    Returns a dict with keys matching the parameter names, all resolved
    to their effective values. Callers unpack the result.
    """
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_outputs = outputs
    effective_max_shadow = max_shadow_distance_m
    effective_tile_workers = tile_workers
    effective_tile_queue_depth = tile_queue_depth
    effective_prefetch_tiles = prefetch_tiles

    if config is not None:
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
        if effective_tile_workers is None:
            effective_tile_workers = config.tile_workers
        if effective_tile_queue_depth is None:
            effective_tile_queue_depth = config.tile_queue_depth
        if effective_prefetch_tiles is None:
            effective_prefetch_tiles = config.prefetch_tiles

    if effective_aniso is None:
        effective_aniso = True
    if effective_physics is None:
        from .loaders import load_physics

        effective_physics = load_physics()
    if effective_materials is None:
        from .loaders import load_params

        effective_materials = load_params()

    return {
        "use_anisotropic_sky": effective_aniso,
        "human": effective_human,
        "physics": effective_physics,
        "materials": effective_materials,
        "outputs": effective_outputs,
        "max_shadow_distance_m": effective_max_shadow,
        "tile_workers": effective_tile_workers,
        "tile_queue_depth": effective_tile_queue_depth,
        "prefetch_tiles": effective_prefetch_tiles,
    }


def process_timestep_result(
    result: SolweigResult,
    weather: Weather,
    outputs: list[str] | None,
    human: HumanParams | None,
    writer: AsyncGeoTiffWriter | None,
    output_dir: str | Path,
    surface: SurfaceData | None = None,
) -> None:
    """Compute optional UTCI/PET, write per-timestep files, then free arrays.

    This is called after the grid accumulator has already been updated.

    Args:
        result: Single-timestep computation result.
        weather: Weather for this timestep.
        outputs: Requested output layer names (may be None).
        human: Human body params (needed for PET).
        writer: Async GeoTIFF writer (None to use sync fallback).
        output_dir: Output directory path.
        surface: Surface data (needed for sync GeoTIFF CRS/transform).
    """
    from .output_async import collect_output_arrays

    # Compute per-timestep UTCI/PET if requested for file output
    if outputs is not None and "utci" in outputs and result.utci is None:
        result.utci = compute_utci_grid(result.tmrt, weather.ta, weather.rh, weather.ws)
    if outputs is not None and "pet" in outputs and result.pet is None:
        result.pet = compute_pet_grid(result.tmrt, weather.ta, weather.rh, weather.ws, human)

    # Save per-timestep outputs to disk
    if writer is not None and outputs:
        writer.submit(
            timestamp=weather.datetime,
            arrays=collect_output_arrays(result, outputs),
        )
    elif outputs:
        result.to_geotiff(
            output_dir=output_dir,
            timestamp=weather.datetime,
            outputs=outputs,
            surface=surface,
        )

    # Free all large arrays after accumulation + disk write
    result.tmrt = None  # type: ignore[assignment]
    result.shadow = None
    result.kdown = None
    result.kup = None
    result.ldown = None
    result.lup = None
    result.utci = None
    result.pet = None


def finalize_summary(
    accumulator: GridAccumulator,
    surface: SurfaceData,
    *,
    processed_steps: int,
    start_time: float,
    location,
    weather_series: list,
    human,
    physics,
    materials,
    use_anisotropic_sky: bool,
    conifer: bool,
    output_dir: str | Path,
    outputs: list[str] | None,
    label: str = "",
) -> TimeseriesSummary:
    """Finalize grid accumulator, log statistics, save GeoTIFFs and metadata.

    Args:
        accumulator: Grid accumulator with per-timestep updates applied.
        surface: Surface data (for GeoTIFF CRS/transform).
        processed_steps: Number of completed timesteps.
        start_time: ``time.time()`` when the loop started.
        location: Location object.
        weather_series: Full weather list (for metadata).
        human, physics, materials: Model parameters (for metadata).
        use_anisotropic_sky: Whether anisotropic sky was used.
        conifer: Whether conifer mode was used.
        output_dir: Output directory path.
        outputs: Requested output layer names.
        label: Extra text appended to completion log line (e.g. "(tiled)").

    Returns:
        Finalized :class:`TimeseriesSummary`.
    """
    summary = accumulator.finalize()
    summary._surface = surface

    total_time = time.time() - start_time
    overall_rate = processed_steps / total_time if total_time > 0 else 0

    logger.info("=" * 60)
    suffix = f" {label}" if label else ""
    logger.info(f"Calculation complete: {processed_steps} timesteps processed{suffix}")
    logger.info(f"  Total time: {total_time:.1f}s ({overall_rate:.2f} steps/s)")
    if summary.n_timesteps > 0:
        _valid_mean = np.nanmean(summary.tmrt_mean)
        _valid_min = np.nanmin(summary.tmrt_min)
        _valid_max = np.nanmax(summary.tmrt_max)
        logger.info(f"  Tmrt range: {_valid_min:.1f}C - {_valid_max:.1f}C (mean: {_valid_mean:.1f}C)")

    if outputs is not None:
        file_count = processed_steps * len(outputs)
        logger.info(f"  Files saved: {file_count} GeoTIFFs in {output_dir}")
    logger.info("=" * 60)

    summary.to_geotiff(output_dir, surface=surface)
    summary._output_dir = Path(output_dir)
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

    return summary
