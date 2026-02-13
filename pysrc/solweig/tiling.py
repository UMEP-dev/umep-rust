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

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from .models import HumanParams, PrecomputedData, SolweigResult, SurfaceData, ThermalState, TileSpec
from .solweig_logging import get_logger

logger = get_logger(__name__)

_TIMING_ENABLED = os.environ.get("SOLWEIG_TIMING", "").lower() in ("1", "true")

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
_FALLBACK_MAX_TILE_SIZE = 2500  # Used when GPU + RAM detection both fail
MIN_SUN_ELEVATION_DEG = 3.0  # Minimum sun elevation for shadow calculations
MAX_BUFFER_M = 500.0  # Default maximum buffer / shadow distance in meters

# Backward-compat alias (imported by tests and calculate_timeseries_tiled docstring)
MAX_TILE_SIZE = _FALLBACK_MAX_TILE_SIZE

# Resource estimation constants
_RAM_FRACTION = 0.50  # Use at most 50% of total physical RAM for tile arrays
_SVF_BYTES_PER_PIXEL = 32  # GPU staging bytes per pixel for SVF computation
_SOLWEIG_BYTES_PER_PIXEL = 400  # Peak Python-side bytes per pixel (benchmarked ~370)
_GPU_HEADROOM = 0.80  # Use 80% of GPU max buffer to leave headroom
_MAX_AUTO_TILE_WORKERS = 6  # Hard cap to avoid bandwidth/cache thrash on many-core CPUs

# Cache for computed tile limits (populated once per context on first call)
_cached_max_tile_side: dict[str, int] = {}


# =============================================================================
# Resource detection
# =============================================================================


def _get_total_ram_bytes() -> int | None:
    """
    Detect total physical RAM in bytes.

    Uses ``os.sysconf`` on POSIX (macOS/Linux) and ``ctypes`` on Windows.
    Returns ``None`` if detection fails.  No external dependencies.
    """
    import os
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys
        else:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return pages * page_size
    except (OSError, ValueError, AttributeError):
        pass
    return None


def compute_max_tile_pixels(*, context: str = "solweig") -> int:
    """
    Compute the maximum number of pixels that fit in a single tile,
    based on real GPU buffer limits and system RAM.

    Args:
        context: ``"solweig"`` for timestep tiling, or ``"svf"`` for SVF-only
            tiling. Affects the bytes-per-pixel estimate used for the RAM
            constraint.

    Returns:
        Maximum pixel count for a tile (rows * cols).
    """
    from . import get_gpu_limits

    bytes_per_pixel = _SVF_BYTES_PER_PIXEL if context == "svf" else _SOLWEIG_BYTES_PER_PIXEL

    # GPU constraint: largest single buffer must fit the tile's staging data.
    # Always uses _SVF_BYTES_PER_PIXEL (~32) because the GPU staging buffer
    # holds shadow + SVF arrays regardless of context. The heavier Python-side
    # working memory (~400 bytes/pixel for "solweig") is captured by the RAM
    # constraint below, which uses the context-appropriate bytes_per_pixel.
    gpu_max_pixels = None
    limits = get_gpu_limits()
    if limits is not None:
        max_buf = limits["max_buffer_size"]
        gpu_max_pixels = int(max_buf * _GPU_HEADROOM) // _SVF_BYTES_PER_PIXEL

    # RAM constraint: total physical RAM × fraction / bytes per pixel.
    # Uses context-dependent bytes_per_pixel: ~32 for "svf" (GPU staging only),
    # ~400 for "solweig" (full timestep with radiation grids, state arrays, etc).
    ram_max_pixels = None
    total_ram = _get_total_ram_bytes()
    if total_ram is not None:
        usable_ram = int(total_ram * _RAM_FRACTION)
        ram_max_pixels = usable_ram // bytes_per_pixel

    # Use the tighter of the two constraints
    candidates = [c for c in [gpu_max_pixels, ram_max_pixels] if c is not None]
    if candidates:
        return max(MIN_TILE_SIZE**2, min(candidates))

    # Fallback: no detection succeeded
    return _FALLBACK_MAX_TILE_SIZE**2


def compute_max_tile_side(*, context: str = "solweig") -> int:
    """
    Compute the maximum tile side length (square tiles) from resource limits.

    The result is cached per *context* for the lifetime of the process.

    Returns:
        Maximum tile side in pixels (at least ``MIN_TILE_SIZE``).
    """
    import math

    if context in _cached_max_tile_side:
        return _cached_max_tile_side[context]

    max_pixels = compute_max_tile_pixels(context=context)
    side = max(MIN_TILE_SIZE, int(math.isqrt(max_pixels)))

    # Log once so the user can see what limits are driving tile sizing
    from . import get_gpu_limits

    limits = get_gpu_limits()
    total_ram = _get_total_ram_bytes()
    gpu_str = f"{limits['max_buffer_size']:,} bytes" if limits else "N/A"
    ram_str = f"{total_ram:,} bytes" if total_ram else "N/A"
    logger.info(
        f"Resource-aware tile sizing (context={context}): "
        f"GPU max_buffer={gpu_str}, system RAM={ram_str}, max_tile_side={side} px"
    )

    _cached_max_tile_side[context] = side
    return side


# =============================================================================
# Helper Functions
# =============================================================================


def _should_use_tiling(rows: int, cols: int) -> bool:
    """Check if raster size exceeds resource limits and requires tiling."""
    max_side = compute_max_tile_side(context="solweig")
    return rows > max_side or cols > max_side


def _calculate_auto_tile_size(rows: int, cols: int) -> int:
    """
    Calculate optimal core tile size based on raster dimensions and resources.

    Returns the resource-derived maximum tile side as the core size.
    ``validate_tile_size()`` will further adjust to ensure the full tile
    (core + 2 × overlap) fits within resource limits.

    Returns:
        Core tile size in pixels.
    """
    return compute_max_tile_side(context="solweig")


def _resolve_tile_workers(tile_workers: int | None, n_tiles: int) -> int:
    """Resolve worker count for tiled orchestration."""
    if n_tiles <= 0:
        return 1
    if tile_workers is not None and tile_workers < 1:
        raise ValueError(f"tile_workers must be >= 1, got {tile_workers}")
    if tile_workers is None:
        cpu_count = os.cpu_count() or 2
        tile_workers = max(2, min(_MAX_AUTO_TILE_WORKERS, cpu_count // 2))
    return max(1, min(tile_workers, n_tiles))


def _resolve_inflight_limit(
    n_workers: int,
    n_tiles: int,
    tile_queue_depth: int | None,
    prefetch_tiles: bool,
) -> int:
    """
    Resolve max number of in-flight tile tasks.

    ``tile_queue_depth`` controls queued tasks beyond active workers.
    Effective in-flight task limit is ``n_workers + queue_depth``.
    """
    if tile_queue_depth is not None and tile_queue_depth < 0:
        raise ValueError(f"tile_queue_depth must be >= 0, got {tile_queue_depth}")

    queue_depth = (n_workers if prefetch_tiles else 0) if tile_queue_depth is None else tile_queue_depth

    return max(1, min(n_tiles, n_workers + queue_depth))


def _resolve_prefetch_default(
    n_workers: int,
    n_tiles: int,
    core_tile_size: int,
    buffer_pixels: int,
) -> bool:
    """
    Decide default prefetch behavior based on estimated in-flight memory pressure.

    Uses a conservative estimate of per-tile Python-side working memory.
    Prefetch is enabled only when estimated in-flight bytes are comfortably
    below the usable RAM budget.
    """
    if n_tiles <= 0:
        return False

    total_ram = _get_total_ram_bytes()
    if total_ram is None:
        return True

    full_side = core_tile_size + 2 * buffer_pixels
    tile_pixels = max(MIN_TILE_SIZE**2, full_side * full_side)
    estimated_tile_bytes = tile_pixels * _SOLWEIG_BYTES_PER_PIXEL

    # Default prefetch queues up to n_workers extra tasks.
    estimated_inflight_tiles = min(n_tiles, n_workers * 2)
    estimated_inflight_bytes = estimated_inflight_tiles * estimated_tile_bytes

    usable_ram = int(total_ram * _RAM_FRACTION)
    return estimated_inflight_bytes <= int(usable_ram * 0.5)


def _extract_tile_surface(
    surface: SurfaceData,
    tile: TileSpec,
    pixel_size: float,
    precomputed: PrecomputedData | None = None,
) -> SurfaceData:
    """
    Extract tile slice from full surface, reusing precomputed SVF when available.

    Creates a new SurfaceData with sliced arrays (DSM, CDSM, etc.).
    If the global surface has precomputed SVF (via prepare() or compute_svf()),
    the SVF is sliced to the tile bounds — avoiding expensive per-tile
    recomputation. If surface.svf is absent but precomputed.svf is provided,
    that precomputed SVF is sliced instead. When neither source exists,
    compute_svf() computes it fresh.

    Args:
        surface: Full raster surface data.
        tile: Tile specification with slice bounds.
        pixel_size: Pixel size in meters.
        precomputed: Optional precomputed data containing SVF.

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
    elif precomputed is not None and precomputed.svf is not None:
        tile_svf = precomputed.svf.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    # Slice shadow matrices if available (required for anisotropic sky in tiled mode)
    tile_shadow_matrices = None
    if surface.shadow_matrices is not None:
        tile_shadow_matrices = surface.shadow_matrices.crop(
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
        shadow_matrices=tile_shadow_matrices,
    )
    # Compute only when no precomputed/cached SVF source was available.
    if tile_svf is None:
        tile_surface.compute_svf()

    return tile_surface


def _slice_tile_precomputed(
    precomputed: PrecomputedData | None,
    tile: TileSpec,
) -> PrecomputedData | None:
    """
    Slice walls and shadow matrices from precomputed data for a tile.

    SVF is handled via surface.svf (sliced in _extract_tile_surface).
    Shadow matrices are spatially cropped to the tile bounds for
    anisotropic sky support in tiled mode.

    Args:
        precomputed: Full raster precomputed data (or None).
        tile: Tile specification with slice bounds.

    Returns:
        PrecomputedData with sliced walls and shadow matrices, or None.
    """
    if precomputed is None:
        return None

    read_slice = tile.read_slice

    tile_wall_ht = None
    tile_wall_asp = None
    tile_shadow_matrices = None

    if precomputed.wall_height is not None:
        tile_wall_ht = precomputed.wall_height[read_slice].copy()
    if precomputed.wall_aspect is not None:
        tile_wall_asp = precomputed.wall_aspect[read_slice].copy()
    if precomputed.shadow_matrices is not None:
        tile_shadow_matrices = precomputed.shadow_matrices.crop(
            tile.row_start_full,
            tile.row_end_full,
            tile.col_start_full,
            tile.col_end_full,
        )

    if tile_wall_ht is None and tile_wall_asp is None and tile_shadow_matrices is None:
        return None

    return PrecomputedData(
        wall_height=tile_wall_ht,
        wall_aspect=tile_wall_asp,
        svf=None,
        shadow_matrices=tile_shadow_matrices,
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
    Validate and adjust core tile size for tiled processing.

    ``tile_size`` is the **core** tile side (the region whose results are
    kept). The actual tile in memory is ``core + 2 × buffer_pixels``.
    This function ensures the full tile fits within resource-derived limits.

    Args:
        tile_size: Requested core tile size in pixels.
        buffer_pixels: Overlap buffer size in pixels.
        pixel_size: Pixel size in meters.

    Returns:
        Tuple of (adjusted_core_size, warning_message or None).

    Constraints:
        - core >= MIN_TILE_SIZE (256)
        - core + 2 * buffer_pixels <= resource-derived maximum
    """
    max_full = compute_max_tile_side(context="solweig")
    warning = None
    core = tile_size

    # Enforce minimum core size
    if core < MIN_TILE_SIZE:
        warning = f"Tile core size {tile_size} below minimum, using {MIN_TILE_SIZE}"
        core = MIN_TILE_SIZE

    # Enforce maximum: full tile (core + 2*buffer) must fit resource limit
    max_core = max_full - 2 * buffer_pixels
    if core > max_core:
        core = max(MIN_TILE_SIZE, max_core)
        warning = (
            f"Tile core {tile_size} + 2x{buffer_pixels}px buffer exceeds resource limit "
            f"({max_full}px). Using core={core}"
        )

    return core, warning


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
    tile_workers: int | None = None,
    tile_queue_depth: int | None = None,
    prefetch_tiles: bool | None = None,
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
        max_shadow_distance_m: Upper bound on shadow reach in meters (default 500.0).
            The actual buffer is computed from the tallest DSM pixel via
            calculate_buffer_distance(), capped at this value.
        tile_workers: Number of worker threads for tile execution. If None,
            uses adaptive default based on CPU count.
        tile_queue_depth: Extra queued tile tasks beyond active workers.
            If None, defaults to one queue slot per worker when prefetching.
        prefetch_tiles: Whether to prefetch queued tile tasks. If None,
            chooses automatically based on estimated memory pressure.
        progress_callback: Optional callback(tile_idx, total_tiles).

    Returns:
        SolweigResult with Tmrt grid. State is not returned for single-timestep
        tiled mode.
    """

    if human is None:
        human = HumanParams()

    # Compute derived weather values
    if not weather._derived_computed:
        weather.compute_derived(location)

    rows, cols = surface.shape
    pixel_size = surface.pixel_size

    # Height-aware buffer: use relative max building height (not absolute elevation)
    max_height = surface.max_height
    buffer_m = calculate_buffer_distance(max_height, max_shadow_distance_m=max_shadow_distance_m)
    buffer_pixels = int(np.ceil(buffer_m / pixel_size))
    logger.info(f"Buffer: {buffer_m:.0f}m ({buffer_pixels}px) from max height {max_height:.1f}m")

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
            precomputed=precomputed,
            use_anisotropic_sky=use_anisotropic_sky,
            conifer=conifer,
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
        f"tile_size={adjusted_tile_size}, buffer={buffer_m:.0f}m ({buffer_pixels}px) from max height {max_height:.1f}m"
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

    # Submit tiles in parallel — Rust releases the GIL during compute_timestep.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_workers = _resolve_tile_workers(tile_workers, n_tiles)
    effective_prefetch = (
        prefetch_tiles
        if prefetch_tiles is not None
        else _resolve_prefetch_default(n_workers, n_tiles, adjusted_tile_size, buffer_pixels)
    )
    inflight_limit = _resolve_inflight_limit(n_workers, n_tiles, tile_queue_depth, effective_prefetch)
    logger.info(f"Tiled runtime: workers={n_workers}, inflight_limit={inflight_limit}, prefetch={effective_prefetch}")
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: dict[Any, tuple[int, TileSpec]] = {}
        submit_times: dict[Any, float] = {}
        max_queue = 0
        turnaround_sum = 0.0

        def _submit_tile(tile_idx: int, tile: TileSpec) -> None:
            tile_surface = _extract_tile_surface(surface, tile, pixel_size, precomputed=precomputed)
            tile_precomputed = _slice_tile_precomputed(precomputed, tile)

            future = executor.submit(
                calculate,
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
            futures[future] = (tile_idx, tile)
            submit_times[future] = time.perf_counter()

        next_tile = 0
        while next_tile < n_tiles and len(futures) < inflight_limit:
            tile = tiles[next_tile]
            _submit_tile(next_tile, tile)
            next_tile += 1

        while futures:
            future = next(as_completed(futures))
            tile_idx, tile = futures.pop(future)
            submit_t = submit_times.pop(future)
            tile_result = future.result()
            _write_tile_result(tile_result, tile, tmrt_out, shadow_out, kdown_out, kup_out, ldown_out, lup_out)

            turnaround_sum += time.perf_counter() - submit_t
            completed += 1
            if _progress is not None:
                _progress.set_text(f"Tile {completed}/{n_tiles}")
                _progress.update(1)
            if progress_callback:
                progress_callback(completed, n_tiles)

            while next_tile < n_tiles and len(futures) < inflight_limit:
                tile = tiles[next_tile]
                _submit_tile(next_tile, tile)
                next_tile += 1
            max_queue = max(max_queue, max(0, len(futures) - n_workers))

    if _progress is not None:
        _progress.close()
    if completed > 0:
        mean_turnaround_ms = (turnaround_sum / completed) * 1000.0
        logger.info(f"Tiled telemetry: mean_turnaround={mean_turnaround_ms:.1f}ms, max_queue={max_queue}")

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
    tile_workers: int | None = None,
    tile_queue_depth: int | None = None,
    prefetch_tiles: bool | None = None,
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
    raster exceeds the resource-derived maximum tile side in either dimension.

    Args:
        surface: Surface/terrain data (DSM required).
        weather_series: List of Weather objects in chronological order.
        location: Geographic location (lat, lon, UTC offset).
        config: Model configuration (provides defaults for None params).
        human: Human body parameters. If None, uses config or defaults.
        precomputed: Pre-computed walls (SVF computed per-tile).
        use_anisotropic_sky: Use anisotropic sky model. Default False.
            Shadow matrices are spatially sliced per tile.
        conifer: Treat vegetation as evergreen conifers. Default False.
        physics: Physics parameters. If None, uses config or bundled defaults.
        materials: Material properties. If None, uses config or bundled defaults.
        wall_material: Wall material type for temperature model.
        max_shadow_distance_m: Upper bound on shadow reach in meters.
            If None, uses config or default (500.0). The actual buffer is
            computed from the tallest DSM pixel via calculate_buffer_distance().
        tile_workers: Number of worker threads for tile execution. If None,
            uses config.tile_workers or adaptive default.
        tile_queue_depth: Extra queued tile tasks beyond active workers.
            If None, uses config.tile_queue_depth or runtime default.
        prefetch_tiles: Whether to prefetch queued tile tasks. If None,
            uses config.prefetch_tiles or defaults to True.
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

    # Height-aware buffer: use relative max building height (not absolute elevation)
    max_height = surface.max_height
    buffer_m = calculate_buffer_distance(max_height, max_shadow_distance_m=effective_max_shadow)
    buffer_pixels = int(np.ceil(buffer_m / pixel_size))
    logger.info(f"Buffer: {buffer_m:.0f}m ({buffer_pixels}px) from max height {max_height:.1f}m")

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
    logger.info(
        f"  Tiles: {n_tiles} (size={adjusted_tile_size}, buffer={buffer_m:.0f}m from max height {max_height:.1f}m)"
    )
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

    n_workers = _resolve_tile_workers(effective_tile_workers, n_tiles)
    if effective_prefetch_tiles is None:
        effective_prefetch_tiles = _resolve_prefetch_default(n_workers, n_tiles, adjusted_tile_size, buffer_pixels)
    inflight_limit = _resolve_inflight_limit(
        n_workers,
        n_tiles,
        effective_tile_queue_depth,
        effective_prefetch_tiles,
    )
    logger.info(
        f"Tiled runtime: workers={n_workers}, inflight_limit={inflight_limit}, prefetch={effective_prefetch_tiles}"
    )

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

    from .computation import _nighttime_result

    # Pre-create tile data once — surfaces and precomputed data don't change
    # between timesteps. This eliminates N_timesteps × N_tiles redundant copies
    # and allows GVF geometry cache + buffer pool to persist across timesteps.
    tile_surfaces = [_extract_tile_surface(surface, tile, pixel_size, precomputed=precomputed) for tile in tiles]
    tile_precomputeds = [_slice_tile_precomputed(precomputed, tile) for tile in tiles]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for t_idx, weather in enumerate(weather_series):
            # Nighttime shortcut: skip tiling entirely when sun is below horizon
            if weather.sun_altitude <= 0:
                night_result = _nighttime_result(surface, weather, state, effective_materials)
                if night_result.state is not None:
                    state = night_result.state

                result = SolweigResult(
                    tmrt=night_result.tmrt,
                    shadow=night_result.shadow,
                    kdown=night_result.kdown,
                    kup=night_result.kup,
                    ldown=night_result.ldown,
                    lup=night_result.lup,
                    utci=None,
                    pet=None,
                    state=None,
                )

                _valid = result.tmrt[np.isfinite(result.tmrt)]
                if _valid.size > 0:
                    _tmrt_sum += _valid.sum()
                    _tmrt_count += _valid.size
                    _tmrt_max = max(_tmrt_max, float(_valid.max()))
                    _tmrt_min = min(_tmrt_min, float(_valid.min()))

                if output_dir is not None:
                    result.to_geotiff(
                        output_dir=output_dir,
                        timestamp=weather.datetime,
                        outputs=effective_outputs,
                        surface=surface,
                    )
                    # Free large arrays — data is on disk
                    result.tmrt = None  # type: ignore[assignment]
                    result.shadow = None
                    result.kdown = None
                    result.kup = None
                    result.ldown = None
                    result.lup = None

                results.append(result)

                # Advance progress by all tiles for this timestep
                step = (t_idx + 1) * n_tiles
                if progress_callback is not None:
                    progress_callback(step, total_work)
                elif _progress is not None:
                    _progress.update(n_tiles)

                elapsed = time.time() - start_time
                rate = (t_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  Timestep {t_idx + 1}/{n_steps} nighttime ({rate:.2f} steps/s)")
                continue

            # Initialize output arrays for this timestep
            tmrt_out = np.full((rows, cols), np.nan, dtype=np.float32)
            shadow_out = np.full((rows, cols), np.nan, dtype=np.float32)
            kdown_out = np.full((rows, cols), np.nan, dtype=np.float32)
            kup_out = np.full((rows, cols), np.nan, dtype=np.float32)
            ldown_out = np.full((rows, cols), np.nan, dtype=np.float32)
            lup_out = np.full((rows, cols), np.nan, dtype=np.float32)

            # Pre-slice all tile states before parallel dispatch
            tile_states = [_slice_tile_state(state, tile) for tile in tiles]

            # Submit all tiles in parallel — Rust releases the GIL during
            # compute_timestep, so threads overlap Python prep with Rust compute.
            futures: dict[int, Any] = {}
            submit_times: dict[int, float] = {}
            next_submit = 0
            max_queue = 0
            turnaround_sum = 0.0

            # Collect in tile order (state merge requires ordered writes), while
            # keeping only a bounded number of tile tasks in flight.
            for tile_idx in range(n_tiles):
                while next_submit < n_tiles and len(futures) < inflight_limit:
                    futures[next_submit] = executor.submit(
                        calculate,
                        surface=tile_surfaces[next_submit],
                        location=location,
                        weather=weather,
                        human=effective_human,
                        precomputed=tile_precomputeds[next_submit],
                        use_anisotropic_sky=effective_aniso,
                        conifer=conifer,
                        state=tile_states[next_submit],
                        physics=effective_physics,
                        materials=effective_materials,
                        wall_material=wall_material,
                        max_shadow_distance_m=effective_max_shadow,
                    )
                    submit_times[next_submit] = time.perf_counter()
                    next_submit += 1
                max_queue = max(max_queue, max(0, len(futures) - n_workers))

                future = futures.pop(tile_idx)
                tile = tiles[tile_idx]
                submit_t = submit_times.pop(tile_idx)

                if _TIMING_ENABLED:
                    _t0 = time.perf_counter()

                tile_result = future.result()
                turnaround_sum += time.perf_counter() - submit_t

                if _TIMING_ENABLED:
                    _t_ffi = time.perf_counter() - _t0
                    _t1 = time.perf_counter()

                # Write core results to global arrays (non-overlapping write_slice)
                _write_tile_result(tile_result, tile, tmrt_out, shadow_out, kdown_out, kup_out, ldown_out, lup_out)

                # Merge tile state back to global state (non-overlapping write_slice)
                if tile_result.state is not None:
                    _merge_tile_state(tile_result.state, tile, state)

                if _TIMING_ENABLED:
                    _t_merge = time.perf_counter() - _t1
                    print(
                        f"[TIMING] tile {tile_idx + 1}/{n_tiles} "
                        f"ffi={_t_ffi * 1000:.1f}ms "
                        f"merge={_t_merge * 1000:.1f}ms",
                        file=sys.stderr,
                    )

                # Report progress
                step = t_idx * n_tiles + tile_idx + 1
                if progress_callback is not None:
                    progress_callback(step, total_work)
                elif _progress is not None:
                    _progress.update(1)

            mean_turnaround_ms = (turnaround_sum / n_tiles) * 1000.0 if n_tiles > 0 else 0.0
            logger.debug(
                f"Tiled timestep telemetry: step={t_idx + 1}/{n_steps}, "
                f"mean_turnaround={mean_turnaround_ms:.1f}ms, max_queue={max_queue}"
            )

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

            # Update incremental stats (before potential array release)
            _valid = result.tmrt[np.isfinite(result.tmrt)]
            if _valid.size > 0:
                _tmrt_sum += _valid.sum()
                _tmrt_count += _valid.size
                _tmrt_max = max(_tmrt_max, float(_valid.max()))
                _tmrt_min = min(_tmrt_min, float(_valid.min()))

            # Save incrementally if output_dir provided
            if output_dir is not None:
                result.to_geotiff(
                    output_dir=output_dir,
                    timestamp=weather.datetime,
                    outputs=effective_outputs,
                    surface=surface,
                )
                # Free large arrays — data is on disk
                result.tmrt = None  # type: ignore[assignment]
                result.shadow = None
                result.kdown = None
                result.kup = None
                result.ldown = None
                result.lup = None

            results.append(result)

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
