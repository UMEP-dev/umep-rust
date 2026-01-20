"""
Tile manager for lazy loading of large raster datasets with overlaps.
Enables memory-efficient processing by loading data on-demand.
"""

import logging
from dataclasses import dataclass

import numpy as np

from . import io as common

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Window:
    """Simple window specification with row/col offset and dimensions."""

    row_off: int
    col_off: int
    height: int
    width: int

    def to_slices(self) -> tuple[slice, slice]:
        """Convert to tuple of slices (row_slice, col_slice)."""
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )


@dataclass
class TileSpec:
    """Specification for a tile with overlap."""

    # Core tile bounds (without overlap)
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    # Full tile bounds (with overlap)
    row_start_full: int
    row_end_full: int
    col_start_full: int
    col_end_full: int
    # Overlap sizes
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int

    @property
    def core_shape(self) -> tuple[int, int]:
        """Shape of core tile (without overlap)."""
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def full_shape(self) -> tuple[int, int]:
        """Shape of full tile (with overlap)."""
        return (self.row_end_full - self.row_start_full, self.col_end_full - self.col_start_full)

    def core_slice(self) -> tuple[slice, slice]:
        """Get slices for extracting core from full tile."""
        return (
            slice(self.overlap_top, self.overlap_top + self.core_shape[0]),
            slice(self.overlap_left, self.overlap_left + self.core_shape[1]),
        )

    @property
    def full_slice(self) -> tuple[slice, slice]:
        """Get slices for full tile (with overlap)."""
        return (
            slice(self.row_start_full, self.row_end_full),
            slice(self.col_start_full, self.col_end_full),
        )

    @property
    def read_window(self) -> Window:
        """Get window for reading full tile (with overlap) from global raster."""
        return Window(
            row_off=self.row_start_full,
            col_off=self.col_start_full,
            height=self.row_end_full - self.row_start_full,
            width=self.col_end_full - self.col_start_full,
        )

    @property
    def write_window(self) -> Window:
        """Get window for writing core tile to global raster."""
        return Window(
            row_off=self.row_start,
            col_off=self.col_start,
            height=self.row_end - self.row_start,
            width=self.col_end - self.col_start,
        )


class TileManager:
    """
    Manages tiled loading of raster data with overlaps for shadow calculations.

    The overlap is calculated based on amaxvalue (max building height) and pixel size
    to ensure shadow calculations at tile edges are accurate.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_size: int,
        pixel_size: float,
        buffer_dist: float = 150.0,
    ):
        """
        Initialize tile manager.

        Args:
            rows: Total number of rows in raster
            cols: Total number of columns in raster
            tile_size: Size of tiles (without overlap) in pixels
            pixel_size: Pixel size in meters
            buffer_dist: Buffer distance in meters for overlap (default: 150.0)
        """
        self.rows = rows
        self.cols = cols
        self.tile_size = tile_size
        self.pixel_size = pixel_size
        self.buffer_dist = buffer_dist

        # Calculate overlap in pixels based on buffer distance
        overlap_pixels = int(np.ceil(buffer_dist / pixel_size))

        # Ensure overlap doesn't exceed tile size
        # self.overlap = min(overlap_pixels, tile_size // 2)
        # For 150m buffer, we strictly want that coverage. If tile is too small, so be it.
        self.overlap = overlap_pixels

        logger.info(
            f"TileManager initialized: {rows}x{cols} raster, "
            f"tile_size={tile_size}, overlap={self.overlap} pixels "
            f"({self.overlap * pixel_size:.1f}m)"
        )

        # Pre-calculate tile specifications
        self.tiles = self._generate_tiles()
        logger.info(f"Generated {len(self.tiles)} tiles")

    def _generate_tiles(self) -> list[TileSpec]:
        """Generate tile specifications with overlaps."""
        tiles = []

        # Calculate number of tiles in each dimension
        n_tiles_row = int(np.ceil(self.rows / self.tile_size))
        n_tiles_col = int(np.ceil(self.cols / self.tile_size))

        for i in range(n_tiles_row):
            for j in range(n_tiles_col):
                # Core tile bounds
                row_start = i * self.tile_size
                row_end = min((i + 1) * self.tile_size, self.rows)
                col_start = j * self.tile_size
                col_end = min((j + 1) * self.tile_size, self.cols)

                # Calculate overlaps (bounded by raster edges)
                overlap_top = self.overlap if i > 0 else 0
                overlap_bottom = self.overlap if row_end < self.rows else 0
                overlap_left = self.overlap if j > 0 else 0
                overlap_right = self.overlap if col_end < self.cols else 0

                # Full tile bounds with overlap
                row_start_full = max(0, row_start - overlap_top)
                row_end_full = min(self.rows, row_end + overlap_bottom)
                col_start_full = max(0, col_start - overlap_left)
                col_end_full = min(self.cols, col_end + overlap_right)

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

    def get_tile(self, tile_idx: int) -> TileSpec:
        """Get tile specification by index."""
        return self.tiles[tile_idx]

    @property
    def total_tiles(self) -> int:
        """Get total number of tiles."""
        return len(self.tiles)

    def get_tiles(self) -> list[TileSpec]:
        """Get list of all tiles."""
        return self.tiles


class LazyRasterLoader:
    """
    Lazy loader for raster data that loads tiles on demand.
    """

    def __init__(
        self,
        raster_path: str,
        tile_manager: TileManager,
        band: int = 0,
        coerce_f64_to_f32: bool = True,
    ):
        """
        Initialize lazy raster loader.

        Args:
            raster_path: Path to raster file
            tile_manager: TileManager instance
            band: Band index to read
            coerce_f64_to_f32: Convert float64 to float32
        """
        self.raster_path = raster_path
        self.tile_manager = tile_manager
        self.band = band
        self.coerce_f64_to_f32 = coerce_f64_to_f32

        # Load metadata only
        self._load_metadata()

        # Cache for loaded tiles
        self._tile_cache = {}

    def _load_metadata(self):
        """Load raster metadata without reading data."""
        meta = common.get_raster_metadata(self.raster_path)
        self.shape = (meta["rows"], meta["cols"])
        # We assume float32 if coercing, otherwise we'd need to check meta['dtype'] if we exposed it
        self.dtype = np.float32 if self.coerce_f64_to_f32 else np.float64  # Simplified assumption
        self.trf_arr = meta["transform"]
        self.crs_wkt = meta["crs"]
        self.nd_val = meta["nodata"]

        # Verify shape matches tile manager
        if self.shape != (self.tile_manager.rows, self.tile_manager.cols):
            raise ValueError(
                f"Raster shape {self.shape} doesn't match tile manager "
                f"({self.tile_manager.rows}, {self.tile_manager.cols})"
            )

    def load_tile(self, tile_idx: int, use_cache: bool = True) -> np.ndarray:
        """
        Load a specific tile with overlap.

        Args:
            tile_idx: Index of tile to load
            use_cache: Whether to use cached tile if available

        Returns:
            Tile data with overlap as numpy array
        """
        if use_cache and tile_idx in self._tile_cache:
            return self._tile_cache[tile_idx]

        tile_spec = self.tile_manager.get_tile(tile_idx)

        window = (
            slice(tile_spec.row_start_full, tile_spec.row_end_full),
            slice(tile_spec.col_start_full, tile_spec.col_end_full),
        )

        # Load tile with window
        tile_data = common.read_raster_window(
            self.raster_path,
            window=window,
            band=self.band,
        )

        if self.coerce_f64_to_f32 and tile_data.dtype == np.float64:
            tile_data = tile_data.astype(np.float32)

        # Ensure contiguous array
        tile_data = np.ascontiguousarray(tile_data, dtype=np.float32)

        if use_cache:
            self._tile_cache[tile_idx] = tile_data

        return tile_data

    def load_full_raster(self) -> np.ndarray:
        """
        Load entire raster (for compatibility with non-tiled code).
        Warning: This defeats the purpose of lazy loading!
        """
        logger.warning("Loading full raster - this may use significant memory!")
        full_data, _, _, _ = common.load_raster(
            self.raster_path,
            bbox=None,
            band=self.band,
            coerce_f64_to_f32=self.coerce_f64_to_f32,
        )
        return np.ascontiguousarray(full_data, dtype=np.float32)

    def clear_cache(self):
        """Clear tile cache to free memory."""
        self._tile_cache.clear()
        logger.debug(f"Cleared tile cache for {self.raster_path}")


class TiledRasterData:
    """
    Container for tiled raster data that mimics the interface of regular numpy arrays
    while loading tiles on demand.
    """

    def __init__(self, lazy_loader: LazyRasterLoader):
        """
        Initialize tiled raster data.

        Args:
            lazy_loader: LazyRasterLoader instance
        """
        self.loader = lazy_loader
        self.tile_manager = lazy_loader.tile_manager
        self.shape = lazy_loader.shape
        self.dtype = lazy_loader.dtype

    def get_tile(self, tile_idx: int) -> tuple[np.ndarray, TileSpec]:
        """
        Get tile data and specification.

        Returns:
            Tuple of (tile_data, tile_spec)
        """
        tile_data = self.loader.load_tile(tile_idx)
        tile_spec = self.tile_manager.get_tile(tile_idx)
        return tile_data, tile_spec

    def to_array(self) -> np.ndarray:
        """Convert to full numpy array (loads all data)."""
        return self.loader.load_full_raster()

    def __getitem__(self, key):
        """Support basic indexing (loads full raster)."""
        logger.warning("Direct indexing loads full raster - consider using get_tile()")
        return self.to_array()[key]
