"""
Tiled processing for large rasters.

This module provides the active tiling implementation for SOLWEIG, supporting
memory-efficient processing of large rasters by dividing them into overlapping
tiles. Shadows and radiation are computed per-tile with buffers to ensure
accurate results at tile boundaries.

Note: An older tiles.py module (TileManager, LazyRasterLoader) was removed
during Q3 consolidation (Feb 2026) as it was unused legacy code from pre-Phase 5.
This module (tiling.py) is the canonical implementation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from .logging import get_logger
from .models import HumanParams, PrecomputedData, SolweigResult, SurfaceData, TileSpec

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .models import (
        Location,
        Weather,
    )


# =============================================================================
# Tiled Processing Support
# =============================================================================

# Constants for tiled processing
MIN_TILE_SIZE = 256  # Minimum tile size in pixels
MAX_TILE_SIZE = 4096  # Maximum tile size in pixels (memory limit)
MIN_SUN_ELEVATION_DEG = 3.0  # Minimum sun elevation for shadow calculations
MAX_BUFFER_M = 500.0  # Maximum buffer distance in meters


def calculate_buffer_distance(max_height: float, min_sun_elev_deg: float = MIN_SUN_ELEVATION_DEG) -> float:
    """
    Calculate required buffer distance for tiled processing based on max building height.

    The buffer must be large enough to capture shadows cast by the tallest buildings
    at the lowest sun elevation angle.

    Formula: buffer = max_height / tan(min_sun_elevation)

    Args:
        max_height: Maximum building/DSM height in meters.
        min_sun_elev_deg: Minimum sun elevation angle in degrees. Default 3.0°.

    Returns:
        Buffer distance in meters, capped at MAX_BUFFER_M (500m).

    Example:
        >>> calculate_buffer_distance(30.0)  # 30m building
        500.0  # Capped (actual would be 573m)
        >>> calculate_buffer_distance(10.0)  # 10m building
        190.8  # 10m / tan(3°)
    """
    if max_height <= 0:
        return 0.0

    tan_elev = np.tan(np.radians(min_sun_elev_deg))
    if tan_elev <= 0:
        return MAX_BUFFER_M

    buffer = max_height / tan_elev
    return min(buffer, MAX_BUFFER_M)


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
        - tile_size <= MAX_TILE_SIZE (4096)
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
    progress_callback: Callable[..., Any] | None = None,
) -> SolweigResult:
    """
    Calculate mean radiant temperature using tiled processing for large rasters.

    This function processes the raster in tiles with overlapping buffers to ensure
    accurate shadow calculations at tile boundaries. Use this for rasters larger
    than ~2000x2000 pixels to manage memory usage.

    The buffer distance is calculated dynamically based on the maximum DSM height:
        buffer = min(max_height / tan(3°), 500m)

    Args:
        surface: Surface/terrain data (DSM required).
        location: Geographic location (lat, lon, UTC offset).
        weather: Weather data for a single timestep.
        human: Human body parameters. Uses defaults if not provided.
        precomputed: Pre-computed preprocessing data (walls only for tiled mode).
            SVF is computed per-tile. Shadow matrices not supported in tiled mode.
        tile_size: Core tile size in pixels (default 1024). Actual size may be
            adjusted to ensure meaningful core area after buffer overlap.
        use_anisotropic_sky: Use anisotropic sky model. Default False.
            Note: Anisotropic sky is not yet supported in tiled mode.
        conifer: Treat vegetation as evergreen conifers (always leaf-on). Default False.
        physics: Physics parameters (Tree_settings, Posture geometry) from load_physics().
            Site-independent scientific constants. If None, uses bundled defaults.
        materials: Material properties (albedo, emissivity per landcover class) from load_materials().
            Site-specific landcover parameters. Only needed if surface has land_cover grid.
        progress_callback: Optional callback(tile_idx, total_tiles) for progress.

    Returns:
        SolweigResult with Tmrt grid. UTCI and PET fields will be None - use
        compute_utci() or compute_pet() for post-processing.
        Note: state is not returned for tiled mode (use calculate_timeseries_tiled
        for multi-timestep with state).

    Raises:
        NotImplementedError: If use_anisotropic_sky=True (not yet supported).
        ValueError: If tile_size is invalid.

    Example:
        # Large raster processing with defaults
        result = calculate_tiled(
            surface=SurfaceData(dsm=large_dsm_array),
            location=Location(latitude=57.7, longitude=12.0),
            weather=Weather(datetime=dt, ta=25, rh=50, global_rad=800),
            tile_size=1024,  # 1024x1024 core tiles
        )
    """

    logger = logging.getLogger(__name__)

    if use_anisotropic_sky:
        raise NotImplementedError(
            "Anisotropic sky model is not yet supported in tiled mode. "
            "Use use_anisotropic_sky=False or process full grid with calculate()."
        )

    if human is None:
        human = HumanParams()

    # Compute derived weather values
    if not weather._derived_computed:
        weather.compute_derived(location)

    rows, cols = surface.shape
    pixel_size = surface.pixel_size
    max_height = surface.max_height

    # Calculate buffer distance based on max height
    buffer_m = calculate_buffer_distance(max_height)
    buffer_pixels = int(np.ceil(buffer_m / pixel_size))

    # Validate and adjust tile size
    adjusted_tile_size, warning = validate_tile_size(tile_size, buffer_pixels, pixel_size)
    if warning:
        logger.warning(warning)

    # Check if tiling is actually needed
    if rows <= adjusted_tile_size and cols <= adjusted_tile_size:
        logger.info(f"Raster {rows}x{cols} fits in single tile, using non-tiled calculation")
        # Import here to avoid circular import
        from .api import calculate

        return calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=human,
            use_anisotropic_sky=use_anisotropic_sky,
            physics=physics,
            materials=materials,
        )

    # Generate tiles
    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)

    # Import calculate here (also needed inside loop)
    from .api import calculate

    logger.info(
        f"Tiled processing: {rows}x{cols} raster, {n_tiles} tiles, "
        f"tile_size={adjusted_tile_size}, buffer={buffer_m:.0f}m ({buffer_pixels}px)"
    )

    # Initialize output arrays
    tmrt_out = np.full((rows, cols), np.nan, dtype=np.float32)
    shadow_out = np.full((rows, cols), np.nan, dtype=np.float32)
    kdown_out = np.full((rows, cols), np.nan, dtype=np.float32)
    kup_out = np.full((rows, cols), np.nan, dtype=np.float32)
    ldown_out = np.full((rows, cols), np.nan, dtype=np.float32)
    lup_out = np.full((rows, cols), np.nan, dtype=np.float32)

    # Process each tile
    for tile_idx, tile in enumerate(tiles):
        if progress_callback:
            progress_callback(tile_idx, n_tiles)

        # Extract tile data from surface
        read_slice = tile.read_slice
        tile_dsm = surface.dsm[read_slice].copy()

        tile_cdsm = None
        if surface.cdsm is not None:
            tile_cdsm = surface.cdsm[read_slice].copy()

        tile_tdsm = None
        if surface.tdsm is not None:
            tile_tdsm = surface.tdsm[read_slice].copy()

        tile_dem = None
        if surface.dem is not None:
            tile_dem = surface.dem[read_slice].copy()

        tile_lc = None
        if surface.land_cover is not None:
            tile_lc = surface.land_cover[read_slice].copy()

        tile_albedo = None
        if surface.albedo is not None:
            tile_albedo = surface.albedo[read_slice].copy()

        tile_emis = None
        if surface.emissivity is not None:
            tile_emis = surface.emissivity[read_slice].copy()

        # Slice walls from precomputed if available
        tile_wall_ht = None
        tile_wall_asp = None
        tile_precomputed = None
        if precomputed is not None:
            if precomputed.wall_height is not None:
                tile_wall_ht = precomputed.wall_height[read_slice].copy()
            if precomputed.wall_aspect is not None:
                tile_wall_asp = precomputed.wall_aspect[read_slice].copy()
            # Create tile precomputed with sliced walls
            if tile_wall_ht is not None or tile_wall_asp is not None:
                tile_precomputed = PrecomputedData(
                    wall_height=tile_wall_ht,
                    wall_aspect=tile_wall_asp,
                    svf=None,  # SVF computed per-tile
                    shadow_matrices=None,
                )

        # Create tile surface (without walls)
        tile_surface = SurfaceData(
            dsm=tile_dsm,
            cdsm=tile_cdsm,
            tdsm=tile_tdsm,
            dem=tile_dem,
            land_cover=tile_lc,
            albedo=tile_albedo,
            emissivity=tile_emis,
            pixel_size=pixel_size,
        )

        # Calculate for tile (SVF computed per-tile, walls from precomputed if available)
        tile_result = calculate(
            surface=tile_surface,
            location=location,
            weather=weather,
            human=human,
            precomputed=tile_precomputed,  # Walls from precomputed, SVF computed per-tile
            use_anisotropic_sky=False,
            conifer=conifer,
            state=None,  # No state for single-timestep tiled
            physics=physics,
            materials=materials,
        )

        # Extract core and write to output
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

    if progress_callback:
        progress_callback(n_tiles, n_tiles)

    return SolweigResult(
        tmrt=tmrt_out,
        shadow=shadow_out,
        kdown=kdown_out,
        kup=kup_out,
        ldown=ldown_out,
        lup=lup_out,
        utci=None,
        pet=None,
        state=None,  # No state for tiled mode
    )
