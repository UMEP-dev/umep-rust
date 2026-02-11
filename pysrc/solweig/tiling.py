"""
Tiled processing for large rasters.

This module provides automatic and manual tiling for SOLWEIG calculations,
supporting both single-timestep and timeseries modes. Large rasters are
automatically divided into overlapping tiles with buffers sized to capture
shadows from the tallest buildings at low sun angles.

Timeseries mode preserves thermal state accumulation across tiles and timesteps,
ensuring physically accurate ground temperature modeling with thermal inertia.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from .models import HumanParams, PrecomputedData, SolweigResult, SurfaceData, ThermalState, TileSpec
from .solweig_logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .models import (
        Location,
        ModelConfig,
        Weather,
    )


# =============================================================================
# Constants
# =============================================================================

MIN_TILE_SIZE = 256  # Minimum tile size in pixels
MAX_TILE_SIZE = 2500  # Maximum tile size in pixels
MIN_SUN_ELEVATION_DEG = 3.0  # Minimum sun elevation for shadow calculations
MAX_BUFFER_M = 500.0  # Default maximum buffer / shadow distance in meters


# =============================================================================
# Helper Functions
# =============================================================================


def _should_use_tiling(rows: int, cols: int) -> bool:
    """Check if raster size requires automatic tiling."""
    return rows > MAX_TILE_SIZE or cols > MAX_TILE_SIZE


def _calculate_auto_tile_size(rows: int, cols: int) -> int:
    """
    Calculate optimal tile size based on raster dimensions.

    Heuristic:
    - >16M pixels (4000x4000): use 1024x1024 tiles
    - >4M pixels (2000x2000): use 2048x2048 tiles
    - Otherwise: no tiling needed (full raster)

    Returns:
        Tile size in pixels.
    """
    total_pixels = rows * cols
    if total_pixels > 4000 * 4000:
        return 1024
    elif total_pixels > 2000 * 2000:
        return min(rows, cols, 2048)
    else:
        return max(rows, cols)


def _extract_tile_surface(
    surface: SurfaceData,
    tile: TileSpec,
    pixel_size: float,
) -> SurfaceData:
    """
    Extract tile slice from full surface, reusing precomputed SVF when available.

    Creates a new SurfaceData with sliced arrays (DSM, CDSM, etc.).
    If the global surface has precomputed SVF (via prepare() or compute_svf()),
    the SVF is sliced to the tile bounds — avoiding expensive per-tile
    recomputation.  When no global SVF exists, compute_svf() computes it fresh.

    Args:
        surface: Full raster surface data.
        tile: Tile specification with slice bounds.
        pixel_size: Pixel size in meters.

    Returns:
        SurfaceData for this tile with SVF available.
    """
    read_slice = tile.read_slice

    tile_dsm = surface.dsm[read_slice].copy()
    tile_cdsm = surface.cdsm[read_slice].copy() if surface.cdsm is not None else None
    tile_tdsm = surface.tdsm[read_slice].copy() if surface.tdsm is not None else None
    tile_dem = surface.dem[read_slice].copy() if surface.dem is not None else None
    tile_lc = surface.land_cover[read_slice].copy() if surface.land_cover is not None else None
    tile_albedo = surface.albedo[read_slice].copy() if surface.albedo is not None else None
    tile_emis = surface.emissivity[read_slice].copy() if surface.emissivity is not None else None

    # Slice precomputed SVF if available (avoids per-tile recomputation)
    tile_svf = None
    if surface.svf is not None:
        tile_svf = surface.svf.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    tile_surface = SurfaceData(
        dsm=tile_dsm,
        cdsm=tile_cdsm,
        tdsm=tile_tdsm,
        dem=tile_dem,
        land_cover=tile_lc,
        albedo=tile_albedo,
        emissivity=tile_emis,
        pixel_size=pixel_size,
        svf=tile_svf,
    )
    tile_surface.compute_svf()  # No-op when tile_svf is set

    return tile_surface


def _slice_tile_precomputed(
    precomputed: PrecomputedData | None,
    tile: TileSpec,
) -> PrecomputedData | None:
    """
    Slice walls from precomputed data for a tile.

    SVF is handled via surface.svf (sliced in _extract_tile_surface).
    Shadow matrices are not supported in tiled mode.

    Args:
        precomputed: Full raster precomputed data (or None).
        tile: Tile specification with slice bounds.

    Returns:
        PrecomputedData with sliced walls, or None.
    """
    if precomputed is None:
        return None

    read_slice = tile.read_slice

    tile_wall_ht = None
    tile_wall_asp = None

    if precomputed.wall_height is not None:
        tile_wall_ht = precomputed.wall_height[read_slice].copy()
    if precomputed.wall_aspect is not None:
        tile_wall_asp = precomputed.wall_aspect[read_slice].copy()

    if tile_wall_ht is None and tile_wall_asp is None:
        return None

    return PrecomputedData(
        wall_height=tile_wall_ht,
        wall_aspect=tile_wall_asp,
        svf=None,
        shadow_matrices=None,
    )


def _write_tile_result(
    tile_result: SolweigResult,
    tile: TileSpec,
    tmrt_out: np.ndarray,
    shadow_out: np.ndarray,
    kdown_out: np.ndarray,
    kup_out: np.ndarray,
    ldown_out: np.ndarray,
    lup_out: np.ndarray,
) -> None:
    """Write core region of tile result to global output arrays."""
    core_slice = tile.core_slice
    write_slice = tile.write_slice

    tmrt_out[write_slice] = tile_result.tmrt[core_slice]
    if tile_result.shadow is not None:
        shadow_out[write_slice] = tile_result.shadow[core_slice]
    if tile_result.kdown is not None:
        kdown_out[write_slice] = tile_result.kdown[core_slice]
    if tile_result.kup is not None:
        kup_out[write_slice] = tile_result.kup[core_slice]
    if tile_result.ldown is not None:
        ldown_out[write_slice] = tile_result.ldown[core_slice]
    if tile_result.lup is not None:
        lup_out[write_slice] = tile_result.lup[core_slice]


def _slice_tile_state(state: ThermalState, tile: TileSpec) -> ThermalState:
    """
    Slice thermal state arrays for a tile.

    Spatial arrays are sliced using tile.read_slice (full tile with overlap).
    Scalar values are copied as-is (they're global, not spatial).

    Args:
        state: Global thermal state for full raster.
        tile: Tile specification with slice bounds.

    Returns:
        ThermalState for this tile.
    """
    read_slice = tile.read_slice

    return ThermalState(
        tgmap1=state.tgmap1[read_slice].copy(),
        tgmap1_e=state.tgmap1_e[read_slice].copy(),
        tgmap1_s=state.tgmap1_s[read_slice].copy(),
        tgmap1_w=state.tgmap1_w[read_slice].copy(),
        tgmap1_n=state.tgmap1_n[read_slice].copy(),
        tgout1=state.tgout1[read_slice].copy(),
        firstdaytime=state.firstdaytime,
        timeadd=state.timeadd,
        timestep_dec=state.timestep_dec,
    )


def _merge_tile_state(
    tile_state: ThermalState,
    tile: TileSpec,
    global_state: ThermalState,
) -> None:
    """
    Merge tile state arrays back into global state (in-place).

    Writes core region (tile.core_slice) of tile state arrays to the
    corresponding region (tile.write_slice) in global state. Updates
    global scalar values from tile state (identical across all tiles
    for a given timestep).

    Args:
        tile_state: Computed state for this tile.
        tile: Tile specification with slice bounds.
        global_state: Global state to update (modified in-place).
    """
    core_slice = tile.core_slice
    write_slice = tile.write_slice

    global_state.tgmap1[write_slice] = tile_state.tgmap1[core_slice]
    global_state.tgmap1_e[write_slice] = tile_state.tgmap1_e[core_slice]
    global_state.tgmap1_s[write_slice] = tile_state.tgmap1_s[core_slice]
    global_state.tgmap1_w[write_slice] = tile_state.tgmap1_w[core_slice]
    global_state.tgmap1_n[write_slice] = tile_state.tgmap1_n[core_slice]
    global_state.tgout1[write_slice] = tile_state.tgout1[core_slice]

    # Scalars are the same across all tiles for a given timestep
    global_state.firstdaytime = tile_state.firstdaytime
    global_state.timeadd = tile_state.timeadd


# =============================================================================
# Public Functions
# =============================================================================


def calculate_buffer_distance(
    max_height: float,
    min_sun_elev_deg: float = MIN_SUN_ELEVATION_DEG,
    max_shadow_distance_m: float = MAX_BUFFER_M,
) -> float:
    """
    Calculate required buffer distance for tiled processing based on max building height.

    The buffer must be large enough to capture shadows cast by the tallest buildings
    at the lowest sun elevation angle.

    Formula: buffer = min(max_height / tan(min_sun_elevation), max_shadow_distance_m)

    Args:
        max_height: Maximum building/DSM height in meters.
        min_sun_elev_deg: Minimum sun elevation angle in degrees. Default 3.0.
        max_shadow_distance_m: Maximum buffer distance in meters. Default 500.0.

    Returns:
        Buffer distance in meters, capped at max_shadow_distance_m.

    Example:
        >>> calculate_buffer_distance(30.0)  # 30m building
        500.0  # Capped (actual would be 573m)
        >>> calculate_buffer_distance(10.0)  # 10m building
        190.8  # 10m / tan(3)
    """
    if max_height <= 0:
        return 0.0

    tan_elev = np.tan(np.radians(min_sun_elev_deg))
    if tan_elev <= 0:
        return max_shadow_distance_m

    buffer = max_height / tan_elev
    return min(buffer, max_shadow_distance_m)


def validate_tile_size(
    tile_size: int,
    buffer_pixels: int,
    pixel_size: float,
) -> tuple[int, str | None]:
    """
    Validate and adjust tile size for tiled processing.

    Ensures the tile size is within bounds and leaves meaningful core area
    after accounting for buffer overlap.

    Args:
        tile_size: Requested tile size in pixels.
        buffer_pixels: Buffer size in pixels.
        pixel_size: Pixel size in meters.

    Returns:
        Tuple of (adjusted_tile_size, warning_message or None).

    Constraints:
        - tile_size >= MIN_TILE_SIZE (256)
        - tile_size <= MAX_TILE_SIZE (2500)
        - Core area (tile_size - 2*buffer) >= 128 pixels
    """
    warning = None
    adjusted = tile_size

    # Enforce minimum
    if adjusted < MIN_TILE_SIZE:
        warning = f"Tile size {tile_size} below minimum, using {MIN_TILE_SIZE}"
        adjusted = MIN_TILE_SIZE

    # Enforce maximum
    if adjusted > MAX_TILE_SIZE:
        warning = f"Tile size {tile_size} above maximum, using {MAX_TILE_SIZE}"
        adjusted = MAX_TILE_SIZE

    # Ensure meaningful core area (at least 128 pixels after buffer)
    min_for_buffer = 2 * buffer_pixels + 128
    if adjusted < min_for_buffer:
        adjusted = min(min_for_buffer, MAX_TILE_SIZE)
        buffer_m = buffer_pixels * pixel_size
        warning = f"Tile size increased to {adjusted} to ensure meaningful core area with {buffer_m:.0f}m buffer"

    return adjusted, warning


def generate_tiles(
    rows: int,
    cols: int,
    tile_size: int,
    overlap: int,
) -> list[TileSpec]:
    """
    Generate tile specifications with overlaps for tiled processing.

    Args:
        rows: Total number of rows in raster.
        cols: Total number of columns in raster.
        tile_size: Core tile size in pixels (without overlap).
        overlap: Overlap size in pixels.

    Returns:
        List of TileSpec objects covering the entire raster.
    """
    tiles = []
    n_tiles_row = int(np.ceil(rows / tile_size))
    n_tiles_col = int(np.ceil(cols / tile_size))

    for i in range(n_tiles_row):
        for j in range(n_tiles_col):
            # Core tile bounds
            row_start = i * tile_size
            row_end = min((i + 1) * tile_size, rows)
            col_start = j * tile_size
            col_end = min((j + 1) * tile_size, cols)

            # Calculate overlaps (bounded by raster edges)
            overlap_top = overlap if i > 0 else 0
            overlap_bottom = overlap if row_end < rows else 0
            overlap_left = overlap if j > 0 else 0
            overlap_right = overlap if col_end < cols else 0

            # Full tile bounds with overlap
            row_start_full = max(0, row_start - overlap_top)
            row_end_full = min(rows, row_end + overlap_bottom)
            col_start_full = max(0, col_start - overlap_left)
            col_end_full = min(cols, col_end + overlap_right)

            tiles.append(
                TileSpec(
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    row_start_full=row_start_full,
                    row_end_full=row_end_full,
                    col_start_full=col_start_full,
                    col_end_full=col_end_full,
                    overlap_top=overlap_top,
                    overlap_bottom=overlap_bottom,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                )
            )

    return tiles


def calculate_tiled(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    tile_size: int = 1024,
    use_anisotropic_sky: bool = False,
    conifer: bool = False,
    physics: SimpleNamespace | None = None,
    materials: SimpleNamespace | None = None,
    max_shadow_distance_m: float = MAX_BUFFER_M,
    progress_callback: Callable[..., Any] | None = None,
) -> SolweigResult:
    """
    Calculate mean radiant temperature using tiled processing for large rasters.

    Processes the raster in tiles with overlapping buffers to ensure accurate
    shadow calculations at tile boundaries.

    Args:
        surface: Surface/terrain data (DSM required).
        location: Geographic location (lat, lon, UTC offset).
        weather: Weather data for a single timestep.
        human: Human body parameters. Uses defaults if not provided.
        precomputed: Pre-computed walls (SVF computed per-tile).
        tile_size: Core tile size in pixels (default 1024).
        use_anisotropic_sky: Use anisotropic sky model. Default False.
        conifer: Treat vegetation as evergreen conifers. Default False.
        physics: Physics parameters. If None, uses bundled defaults.
        materials: Material properties. If None, uses bundled defaults.
        max_shadow_distance_m: Maximum shadow reach / tile buffer in meters.
            Default 500.0.
        progress_callback: Optional callback(tile_idx, total_tiles).

    Returns:
        SolweigResult with Tmrt grid. State is not returned for single-timestep
        tiled mode.
    """

    logger = logging.getLogger(__name__)

    if human is None:
        human = HumanParams()

    # Compute derived weather values
    if not weather._derived_computed:
        weather.compute_derived(location)

    rows, cols = surface.shape
    pixel_size = surface.pixel_size

    # Tile overlap = max_shadow_distance_m (conservative worst case)
    buffer_pixels = int(np.ceil(max_shadow_distance_m / pixel_size))

    # Validate and adjust tile size
    adjusted_tile_size, warning = validate_tile_size(tile_size, buffer_pixels, pixel_size)
    if warning:
        logger.warning(warning)

    # Check if tiling is actually needed
    if rows <= adjusted_tile_size and cols <= adjusted_tile_size:
        logger.info(f"Raster {rows}x{cols} fits in single tile, using non-tiled calculation")
        from .api import calculate

        return calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=human,
            use_anisotropic_sky=use_anisotropic_sky,
            physics=physics,
            materials=materials,
            max_shadow_distance_m=max_shadow_distance_m,
        )

    # Generate tiles
    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)

    from .api import calculate

    logger.info(
        f"Tiled processing: {rows}x{cols} raster, {n_tiles} tiles, "
        f"tile_size={adjusted_tile_size}, buffer={max_shadow_distance_m:.0f}m ({buffer_pixels}px)"
    )

    # Initialize output arrays
    tmrt_out = np.full((rows, cols), np.nan, dtype=np.float32)
    shadow_out = np.full((rows, cols), np.nan, dtype=np.float32)
    kdown_out = np.full((rows, cols), np.nan, dtype=np.float32)
    kup_out = np.full((rows, cols), np.nan, dtype=np.float32)
    ldown_out = np.full((rows, cols), np.nan, dtype=np.float32)
    lup_out = np.full((rows, cols), np.nan, dtype=np.float32)

    # Set up progress reporting
    from .progress import ProgressReporter

    _progress = None if progress_callback is not None else ProgressReporter(total=n_tiles, desc="SOLWEIG tiled")

    # Process each tile
    for tile_idx, tile in enumerate(tiles):
        # Update progress description
        desc = f"Tile {tile_idx + 1}/{n_tiles}"
        if _progress is not None:
            _progress.set_description(desc)
            _progress.set_text(f"Tile {tile_idx + 1}/{n_tiles}")

        if progress_callback:
            progress_callback(tile_idx, n_tiles)

        tile_surface = _extract_tile_surface(surface, tile, pixel_size)
        tile_precomputed = _slice_tile_precomputed(precomputed, tile)

        tile_result = calculate(
            surface=tile_surface,
            location=location,
            weather=weather,
            human=human,
            precomputed=tile_precomputed,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
            state=None,
            physics=physics,
            materials=materials,
            max_shadow_distance_m=max_shadow_distance_m,
        )

        _write_tile_result(tile_result, tile, tmrt_out, shadow_out, kdown_out, kup_out, ldown_out, lup_out)

        if _progress is not None:
            _progress.update(1)

    if progress_callback:
        progress_callback(n_tiles, n_tiles)

    if _progress is not None:
        _progress.close()

    return SolweigResult(
        tmrt=tmrt_out,
        shadow=shadow_out,
        kdown=kdown_out,
        kup=kup_out,
        ldown=ldown_out,
        lup=lup_out,
        utci=None,
        pet=None,
        state=None,
    )


def calculate_timeseries_tiled(
    surface: SurfaceData,
    weather_series: list[Weather],
    location: Location,
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
    Calculate Tmrt timeseries using tiled processing for large rasters.

    Automatically divides large rasters into overlapping tiles and processes
    each timestep tile-by-tile, preserving thermal state accumulation across
    both tiles and timesteps.

    This function is called automatically by calculate_timeseries() when the
    raster exceeds MAX_TILE_SIZE in either dimension.

    Args:
        surface: Surface/terrain data (DSM required).
        weather_series: List of Weather objects in chronological order.
        location: Geographic location (lat, lon, UTC offset).
        config: Model configuration (provides defaults for None params).
        human: Human body parameters. If None, uses config or defaults.
        precomputed: Pre-computed walls (SVF computed per-tile).
        use_anisotropic_sky: Use anisotropic sky model.
            Not supported in tiled mode — raises NotImplementedError.
        conifer: Treat vegetation as evergreen conifers. Default False.
        physics: Physics parameters. If None, uses config or bundled defaults.
        materials: Material properties. If None, uses config or bundled defaults.
        wall_material: Wall material type for temperature model.
        max_shadow_distance_m: Maximum shadow reach / tile buffer in meters.
            If None, uses config or default (500.0).
        output_dir: Directory to save results incrementally as GeoTIFF.
        outputs: Which outputs to save (e.g., ["tmrt", "shadow"]).
        progress_callback: Optional callback(current_step, total_steps).

    Returns:
        List of SolweigResult objects, one per timestep.
    """
    if not weather_series:
        return []

    # Resolve effective parameters from config
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_outputs = outputs
    effective_max_shadow = max_shadow_distance_m

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

    if effective_aniso is None:
        effective_aniso = False
    if effective_human is None:
        effective_human = HumanParams()
    if effective_max_shadow is None:
        effective_max_shadow = MAX_BUFFER_M
    if effective_materials is None:
        from .loaders import load_params

        effective_materials = load_params()
    if effective_physics is None:
        from .loaders import load_physics

        effective_physics = load_physics()

    if output_dir is not None and effective_outputs is None:
        effective_outputs = ["tmrt"]

    # Fill NaN in surface layers
    surface.fill_nan()

    rows, cols = surface.shape
    pixel_size = surface.pixel_size

    # Tile overlap = max_shadow_distance_m (conservative worst case)
    buffer_pixels = int(np.ceil(effective_max_shadow / pixel_size))

    # Determine tile size
    tile_size = _calculate_auto_tile_size(rows, cols)
    adjusted_tile_size, warning = validate_tile_size(tile_size, buffer_pixels, pixel_size)
    if warning:
        logger.warning(warning)

    # Generate tiles
    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)
    n_steps = len(weather_series)

    # Pre-compute weather (sun positions, radiation)
    from .timeseries import _precompute_weather

    logger.info("=" * 60)
    logger.info("Starting SOLWEIG tiled timeseries calculation")
    logger.info(f"  Grid size: {cols}x{rows} pixels")
    logger.info(f"  Timesteps: {n_steps}")
    start_str = weather_series[0].datetime.strftime("%Y-%m-%d %H:%M")
    end_str = weather_series[-1].datetime.strftime("%Y-%m-%d %H:%M")
    logger.info(f"  Period: {start_str} -> {end_str}")
    logger.info(f"  Location: {location.latitude:.2f}N, {location.longitude:.2f}E")
    logger.info(f"  Tiles: {n_tiles} (size={adjusted_tile_size}, buffer={effective_max_shadow:.0f}m)")
    logger.info("=" * 60)

    logger.info("Pre-computing sun positions and radiation splits...")
    precompute_start = time.time()
    _precompute_weather(weather_series, location)
    precompute_time = time.time() - precompute_start
    logger.info(f"  Pre-computed {n_steps} timesteps in {precompute_time:.1f}s")

    # Create output directory if needed
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Import calculate
    from .api import calculate

    # Initialize global state
    state = ThermalState.initial(surface.shape)
    if len(weather_series) >= 2:
        dt0 = weather_series[0].datetime
        dt1 = weather_series[1].datetime
        state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

    results = []
    total_work = n_steps * n_tiles
    start_time = time.time()

    # Incremental stats
    _tmrt_sum = 0.0
    _tmrt_max = -np.inf
    _tmrt_min = np.inf
    _tmrt_count = 0

    # Set up progress reporting
    from .progress import ProgressReporter

    _progress = (
        None if progress_callback is not None else ProgressReporter(total=total_work, desc="SOLWEIG tiled timeseries")
    )

    for t_idx, weather in enumerate(weather_series):
        # Initialize output arrays for this timestep
        tmrt_out = np.full((rows, cols), np.nan, dtype=np.float32)
        shadow_out = np.full((rows, cols), np.nan, dtype=np.float32)
        kdown_out = np.full((rows, cols), np.nan, dtype=np.float32)
        kup_out = np.full((rows, cols), np.nan, dtype=np.float32)
        ldown_out = np.full((rows, cols), np.nan, dtype=np.float32)
        lup_out = np.full((rows, cols), np.nan, dtype=np.float32)

        for tile_idx, tile in enumerate(tiles):
            # Update progress description before computing this tile
            desc = f"Step {t_idx + 1}/{n_steps} | Tile {tile_idx + 1}/{n_tiles}"
            if _progress is not None:
                _progress.set_description(desc)
                _progress.set_text(f"Timestep {t_idx + 1}/{n_steps} \u2014 Tile {tile_idx + 1}/{n_tiles}")

            # Extract tile surface
            tile_surface = _extract_tile_surface(surface, tile, pixel_size)
            tile_precomputed = _slice_tile_precomputed(precomputed, tile)

            # Slice state for this tile
            tile_state = _slice_tile_state(state, tile)

            # Compute tile
            tile_result = calculate(
                surface=tile_surface,
                location=location,
                weather=weather,
                human=effective_human,
                precomputed=tile_precomputed,
                use_anisotropic_sky=effective_aniso,
                conifer=conifer,
                state=tile_state,
                physics=effective_physics,
                materials=effective_materials,
                wall_material=wall_material,
                max_shadow_distance_m=effective_max_shadow,
            )

            # Write core results to global arrays
            _write_tile_result(tile_result, tile, tmrt_out, shadow_out, kdown_out, kup_out, ldown_out, lup_out)

            # Merge tile state back to global state
            if tile_result.state is not None:
                _merge_tile_state(tile_result.state, tile, state)

            # Report progress
            step = t_idx * n_tiles + tile_idx + 1
            if progress_callback is not None:
                progress_callback(step, total_work)
            elif _progress is not None:
                _progress.update(1)

        # Log timestep completion
        elapsed = time.time() - start_time
        rate = (t_idx + 1) / elapsed if elapsed > 0 else 0
        logger.info(f"  Timestep {t_idx + 1}/{n_steps} complete ({rate:.2f} steps/s)")

        # Create result for this timestep
        result = SolweigResult(
            tmrt=tmrt_out,
            shadow=shadow_out,
            kdown=kdown_out,
            kup=kup_out,
            ldown=ldown_out,
            lup=lup_out,
            utci=None,
            pet=None,
            state=None,  # State managed externally
        )

        # Save incrementally if output_dir provided
        if output_dir is not None:
            result.to_geotiff(
                output_dir=output_dir,
                timestamp=weather.datetime,
                outputs=effective_outputs,
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

    # Close progress bar
    if _progress is not None:
        _progress.close()

    # Log summary
    total_time = time.time() - start_time
    overall_rate = len(results) / total_time if total_time > 0 else 0

    logger.info("=" * 60)
    logger.info(f"Calculation complete: {len(results)} timesteps processed (tiled)")
    logger.info(f"  Total time: {total_time:.1f}s ({overall_rate:.2f} steps/s)")
    if _tmrt_count > 0:
        mean_tmrt = _tmrt_sum / _tmrt_count
        logger.info(f"  Tmrt range: {_tmrt_min:.1f}C - {_tmrt_max:.1f}C (mean: {mean_tmrt:.1f}C)")

    if output_dir is not None and effective_outputs is not None:
        file_count = len(results) * len(effective_outputs)
        logger.info(f"  Files saved: {file_count} GeoTIFFs in {output_dir}")
    logger.info("=" * 60)

    # Save run metadata if output_dir provided
    if output_dir is not None:
        from .metadata import create_run_metadata, save_run_metadata

        metadata = create_run_metadata(
            surface=surface,
            location=location,
            weather_series=weather_series,
            human=effective_human,
            physics=effective_physics,
            materials=effective_materials,
            use_anisotropic_sky=effective_aniso,
            conifer=conifer,
            output_dir=output_dir,
            outputs=effective_outputs,
        )
        save_run_metadata(metadata, output_dir)

    return results
