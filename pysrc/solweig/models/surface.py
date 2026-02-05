"""Surface and terrain data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from affine import Affine as AffineClass

from .. import io
from .. import walls as walls_module
from ..buffers import BufferPool
from ..cache import CacheMetadata, clear_stale_cache, validate_cache
from ..loaders import get_lc_properties_from_params
from ..rustalgos import skyview
from ..solweig_logging import get_logger
from ..utils import extract_bounds, intersect_bounds, resample_to_grid
from .precomputed import SvfArrays

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .precomputed import ShadowArrays, SvfArrays

logger = get_logger(__name__)


@dataclass
class SurfaceData:
    """
    Surface/terrain data for SOLWEIG calculations.

    Only `dsm` is required. Other rasters are optional and will be
    treated as absent if not provided.

    Attributes:
        dsm: Digital Surface Model (elevation in meters). Required.
        cdsm: Canopy Digital Surface Model (vegetation heights). Optional.
            Can be relative heights (above ground) or absolute elevations.
            Set relative_heights=True if CDSM contains relative heights.
        dem: Digital Elevation Model (ground elevation). Optional.
        tdsm: Trunk Digital Surface Model (trunk zone heights). Optional.
        land_cover: Land cover classification grid (UMEP standard IDs). Optional.
            IDs: 0=paved, 1=asphalt, 2=buildings, 5=grass, 6=bare_soil, 7=water.
            When provided, albedo and emissivity are derived from land cover.
        wall_height: Preprocessed wall heights (meters). Optional.
            If not provided, computed on-the-fly from DSM.
        wall_aspect: Preprocessed wall aspects (degrees, 0=N). Optional.
            If not provided, computed on-the-fly from DSM.
        svf: Preprocessed Sky View Factor arrays. Optional.
            If not provided, computed on-the-fly.
        shadow_matrices: Preprocessed shadow matrices for anisotropic sky. Optional.
        pixel_size: Pixel size in meters. Default 1.0.
        trunk_ratio: Ratio for auto-generating TDSM from CDSM. Default 0.25.
        relative_heights: Whether CDSM/TDSM contain relative heights (above ground)
            rather than absolute elevations. Default True.
            If True and preprocess() is not called, a warning is issued.

    Note:
        Albedo and emissivity are derived internally from land_cover using
        standard UMEP parameters. They cannot be directly specified.

    Note:
        max_height is auto-computed from dsm as: np.nanmax(dsm) - np.nanmin(dsm)

    Height Conventions:
        SOLWEIG supports two conventions for CDSM/TDSM vegetation data:

        **Relative Heights** (default, `relative_heights=True`):
            - CDSM values represent vegetation height above ground (e.g., 6m tree)
            - TDSM values represent trunk zone height above ground (e.g., 1.5m)
            - Typical range: 0-40m for CDSM, 0-10m for TDSM
            - Must call `preprocess()` before calculations

        **Absolute Heights** (`relative_heights=False`):
            - CDSM values are absolute elevations (same reference as DSM)
            - TDSM values are absolute elevations
            - Example: DSM=127m, CDSM=133m means 6m vegetation
            - No preprocessing needed

        The internal algorithms (Rust) always use **absolute heights**. The
        `preprocess()` method converts relative → absolute using:
            cdsm_absolute = base + cdsm_relative
            tdsm_absolute = base + tdsm_relative
        where `base = DEM` if available, else `base = DSM`.

        **Important**: External tools may have different conventions:
            - UMEP Python `svfForProcessing153` (installed pkg): expects RELATIVE
            - UMEP-core `svfForProcessing153`: expects ABSOLUTE
            - QGIS plugins: typically use RELATIVE

    Example:
        # With relative vegetation heights (common case):
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_relative, relative_heights=True)
        surface.preprocess()  # Convert to absolute heights

        # With absolute vegetation heights (already preprocessed):
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_absolute, relative_heights=False)
        # No need to call preprocess()
    """

    # Surface rasters
    dsm: NDArray[np.floating]
    cdsm: NDArray[np.floating] | None = None
    dem: NDArray[np.floating] | None = None
    tdsm: NDArray[np.floating] | None = None
    albedo: NDArray[np.floating] | None = None
    emissivity: NDArray[np.floating] | None = None
    land_cover: NDArray[np.integer] | None = None

    # Preprocessing data (walls, SVF, shadows)
    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None

    # Grid properties
    pixel_size: float = 1.0
    trunk_ratio: float = 0.25  # Trunk zone ratio for auto-generating TDSM from CDSM
    relative_heights: bool = True  # Whether CDSM/TDSM are relative (not absolute)

    # Internal state
    _preprocessed: bool = field(default=False, init=False, repr=False)
    _geotransform: list[float] | None = field(default=None, init=False, repr=False)  # GDAL geotransform
    _crs_wkt: str | None = field(default=None, init=False, repr=False)  # CRS as WKT string
    _buffer_pool: BufferPool | None = field(default=None, init=False, repr=False)  # Reusable array pool

    def __post_init__(self):
        # Ensure dsm is float32 for memory efficiency
        self.dsm = np.asarray(self.dsm, dtype=np.float32)

        # Convert optional surface arrays if provided
        if self.cdsm is not None:
            self.cdsm = np.asarray(self.cdsm, dtype=np.float32)
        if self.dem is not None:
            self.dem = np.asarray(self.dem, dtype=np.float32)
        if self.tdsm is not None:
            self.tdsm = np.asarray(self.tdsm, dtype=np.float32)
        if self.albedo is not None:
            self.albedo = np.asarray(self.albedo, dtype=np.float32)
        if self.emissivity is not None:
            self.emissivity = np.asarray(self.emissivity, dtype=np.float32)
        if self.land_cover is not None:
            self.land_cover = np.asarray(self.land_cover, dtype=np.uint8)

        # Convert optional preprocessing arrays if provided
        if self.wall_height is not None:
            self.wall_height = np.asarray(self.wall_height, dtype=np.float32)
        if self.wall_aspect is not None:
            self.wall_aspect = np.asarray(self.wall_aspect, dtype=np.float32)

    @classmethod
    def prepare(
        cls,
        dsm: str | Path,
        working_dir: str | Path,
        cdsm: str | Path | None = None,
        dem: str | Path | None = None,
        tdsm: str | Path | None = None,
        land_cover: str | Path | None = None,
        wall_height: str | Path | None = None,
        wall_aspect: str | Path | None = None,
        svf_dir: str | Path | None = None,
        bbox: list[float] | None = None,
        pixel_size: float | None = None,
        trunk_ratio: float = 0.25,
        relative_heights: bool = True,
        force_recompute: bool = False,
    ) -> SurfaceData:
        """
        Prepare surface data and optional preprocessing from GeoTIFF files.

        Loads raster data from disk and prepares it for SOLWEIG calculations.
        Optionally loads preprocessing data (walls, SVF) and automatically
        aligns it to match the surface grid.

        Args:
            dsm: Path to DSM GeoTIFF file (required).
            working_dir: Working directory for caching computed/resampled data (required).
                Computed walls/SVF and resampled rasters are auto-discovered here and
                reused on subsequent runs. Structure: working_dir/walls/, working_dir/svf/,
                working_dir/resampled/. All intermediate results saved for inspection.
                To regenerate cached data, delete the working_dir.
            cdsm: Path to CDSM GeoTIFF file (optional).
            dem: Path to DEM GeoTIFF file (optional).
            tdsm: Path to TDSM GeoTIFF file (optional).
            land_cover: Path to land cover GeoTIFF file (optional).
                Albedo and emissivity are derived from land cover internally.
            wall_height: Path to wall height GeoTIFF file (optional).
                If not provided, walls are auto-discovered in working_dir/walls/ or
                computed from DSM and cached.
            wall_aspect: Path to wall aspect GeoTIFF file (optional, degrees 0=N).
                If not provided, walls are auto-discovered in working_dir/walls/ or
                computed from DSM and cached.
            svf_dir: Directory containing SVF preprocessing files (optional):
                - svfs.zip: SVF arrays (required if svf_dir provided)
                - shadowmats.npz: Shadow matrices for anisotropic sky (optional)
                If not provided, SVF is auto-discovered in working_dir/svf/ or
                computed and cached.
            bbox: Explicit bounding box [minx, miny, maxx, maxy] (optional).
                If provided, all data is cropped/resampled to this extent.
                If None, uses auto-intersection of all provided data.
            pixel_size: Pixel size in meters. If None, computed from DSM geotransform.
            trunk_ratio: Ratio for auto-generating TDSM from CDSM. Default 0.25.
            relative_heights: Whether CDSM/TDSM contain relative heights. Default True.
            force_recompute: If True, skip cache and recompute walls/SVF even if they
                exist in working_dir. Default False (use cached data when available).

        Returns:
            SurfaceData instance with loaded terrain and preprocessing data.

        Note:
            When preprocessing data (walls/SVF) has different extents or resolution
            than the surface data, it is automatically resampled/cropped to match.
            Use bbox parameter to explicitly control the output extent.

        Example:
            # Load surface with preprocessing
            surface = SurfaceData.prepare(
                dsm="data/dsm.tif",
                cdsm="data/cdsm.tif",
                wall_height="preprocessed/walls/wall_hts.tif",
                wall_aspect="preprocessed/walls/wall_aspects.tif",
                svf_dir="preprocessed/svf",
            )

            # Minimal case - walls and SVF computed automatically
            surface = SurfaceData.prepare(dsm="data/dsm.tif")

            # Explicit extent override
            surface = SurfaceData.prepare(
                dsm="data/dsm.tif",
                wall_height="preprocessed/walls/wall_hts.tif",
                wall_aspect="preprocessed/walls/wall_aspects.tif",
                bbox=[476800, 4205850, 477200, 4206250],
                pixel_size=1.0,
            )
        """
        logger.info("Preparing surface data from GeoTIFF files...")

        # Load and validate DSM
        dsm_arr, dsm_transform, dsm_crs, pixel_size = cls._load_and_validate_dsm(dsm, pixel_size)

        # Load optional terrain rasters
        terrain_rasters = cls._load_terrain_rasters(cdsm, dem, tdsm, land_cover, trunk_ratio)

        # Load preprocessing data (walls, SVF)
        working_path = Path(working_dir)
        preprocess_data = cls._load_preprocessing_data(wall_height, wall_aspect, svf_dir, working_path, force_recompute)

        # Compute extent, validate bbox, and resample all rasters
        aligned_rasters = cls._align_rasters(
            dsm_arr,
            dsm_transform,
            dsm_crs,
            pixel_size,
            terrain_rasters,
            preprocess_data,
            bbox,
        )

        # Create SurfaceData instance
        surface_data = cls._create_surface_instance(
            aligned_rasters,
            pixel_size,
            trunk_ratio,
            relative_heights,
        )

        # Validate cached SVF against current inputs (if SVF was loaded)
        if preprocess_data["svf_data"] is not None and not force_recompute:
            svf_cache_dir = working_path / "svf" / "memmap"
            dsm_arr = aligned_rasters["dsm_arr"]
            cdsm_arr = aligned_rasters.get("cdsm_arr")

            if not validate_cache(svf_cache_dir, dsm_arr, pixel_size, cdsm_arr):
                logger.info("  → Cache stale, clearing and recomputing SVF...")
                clear_stale_cache(svf_cache_dir)
                preprocess_data["svf_data"] = None
                preprocess_data["compute_svf"] = True
                surface_data.svf = None

        # Compute and cache walls if needed
        if preprocess_data["compute_walls"]:
            cls._compute_and_cache_walls(surface_data, aligned_rasters, working_path)

        # Compute and cache SVF if needed
        if preprocess_data["compute_svf"]:
            cls._compute_and_cache_svf(surface_data, aligned_rasters, working_path, trunk_ratio)

        # Preprocess CDSM/TDSM if relative heights
        if relative_heights and surface_data.cdsm is not None:
            logger.debug("  Preprocessing CDSM/TDSM (relative → absolute heights)")
            surface_data.preprocess()

        logger.info("✓ Surface data prepared successfully")
        return surface_data

    @staticmethod
    def _load_and_validate_dsm(dsm: str | Path, pixel_size: float | None) -> tuple:
        """
        Load DSM raster and validate CRS.

        Args:
            dsm: Path to DSM GeoTIFF file.
            pixel_size: Optional pixel size in meters. If None, extracted from geotransform.

        Returns:
            Tuple of (dsm_array, dsm_transform, dsm_crs, pixel_size).

        Raises:
            ValueError: If DSM has no CRS or CRS is not projected.
        """
        from .. import io

        # Load required DSM
        dsm_arr, dsm_transform, dsm_crs, _ = io.load_raster(str(dsm))
        logger.info(f"  DSM: {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")

        # Compute pixel size from geotransform if not provided
        if pixel_size is None:
            pixel_size = abs(dsm_transform[1])  # X pixel size
            logger.info(f"  Extracted pixel size from DSM: {pixel_size:.2f} m")
        else:
            logger.info(f"  Using specified pixel size: {pixel_size:.2f} m")

        # Warn if pixel size is less than 1 meter
        if pixel_size < 1.0:
            logger.warning(
                f"  ⚠ Pixel size ({pixel_size:.2f} m) is less than 1 meter - calculations may be slow for large areas"
            )

        # Validate CRS is projected (required for distance calculations)
        if dsm_crs is None:
            raise ValueError("DSM file has no CRS information. SOLWEIG requires a projected coordinate system.")

        try:
            from pyproj import CRS as pyproj_CRS

            crs_obj = pyproj_CRS.from_wkt(dsm_crs)
            if not crs_obj.is_projected:
                raise ValueError(
                    f"DSM CRS is geographic (lat/lon): {crs_obj.name}. "
                    f"SOLWEIG requires a projected coordinate system (e.g., UTM, State Plane) "
                    f"for accurate distance and area calculations. Please reproject your data."
                )
            logger.info(f"  CRS validated: {crs_obj.name} (EPSG:{crs_obj.to_epsg() or 'custom'})")
        except Exception as e:
            logger.warning(f"  ⚠ Could not validate CRS: {e}")

        return dsm_arr, dsm_transform, dsm_crs, pixel_size

    @staticmethod
    def _load_terrain_rasters(
        cdsm: str | Path | None,
        dem: str | Path | None,
        tdsm: str | Path | None,
        land_cover: str | Path | None,
        trunk_ratio: float,
    ) -> dict:
        """
        Load optional terrain rasters (CDSM, DEM, TDSM, land_cover).

        Args:
            cdsm: Path to CDSM GeoTIFF file (optional).
            dem: Path to DEM GeoTIFF file (optional).
            tdsm: Path to TDSM GeoTIFF file (optional).
            land_cover: Path to land cover GeoTIFF file (optional).
            trunk_ratio: Trunk ratio for auto-generating TDSM from CDSM.

        Returns:
            Dictionary with keys: cdsm_arr, cdsm_transform, dem_arr, dem_transform,
            tdsm_arr, tdsm_transform, land_cover_arr, land_cover_transform.
        """
        from .. import io

        result = {}

        # Load CDSM
        if cdsm is not None:
            result["cdsm_arr"], result["cdsm_transform"], _, _ = io.load_raster(str(cdsm))
            logger.info("  ✓ Canopy DSM (CDSM) provided")
        else:
            result["cdsm_arr"], result["cdsm_transform"] = None, None
            logger.info("  → No vegetation data - simulation without trees/vegetation")

        # Load DEM
        if dem is not None:
            result["dem_arr"], result["dem_transform"], _, _ = io.load_raster(str(dem))
            logger.info("  ✓ Ground elevation (DEM) provided")
        else:
            result["dem_arr"], result["dem_transform"] = None, None

        # Load TDSM
        if tdsm is not None:
            result["tdsm_arr"], result["tdsm_transform"], _, _ = io.load_raster(str(tdsm))
            logger.info("  ✓ Trunk DSM (TDSM) provided")
        elif result["cdsm_arr"] is not None:
            result["tdsm_arr"], result["tdsm_transform"] = None, None
            logger.info(f"  → No TDSM provided - will auto-generate from CDSM (ratio={trunk_ratio})")
        else:
            result["tdsm_arr"], result["tdsm_transform"] = None, None

        # Load land cover
        if land_cover is not None:
            result["land_cover_arr"], result["land_cover_transform"], _, _ = io.load_raster(str(land_cover))
            logger.info("  ✓ Land cover provided (albedo/emissivity derived from classification)")
        else:
            result["land_cover_arr"], result["land_cover_transform"] = None, None

        return result

    @staticmethod
    def _load_preprocessing_data(
        wall_height: str | Path | None,
        wall_aspect: str | Path | None,
        svf_dir: str | Path | None,
        working_path: Path,
        force_recompute: bool,
    ) -> dict:
        """
        Load preprocessing data (walls, SVF) with auto-discovery.

        Args:
            wall_height: Path to wall height GeoTIFF file (optional).
            wall_aspect: Path to wall aspect GeoTIFF file (optional).
            svf_dir: Directory containing SVF preprocessing files (optional).
            working_path: Working directory for caching.
            force_recompute: If True, skip cache and recompute.

        Returns:
            Dictionary with keys: wall_height_arr, wall_height_transform, wall_aspect_arr,
            wall_aspect_transform, svf_data, shadow_data, compute_walls, compute_svf.
        """
        from .. import io
        from .precomputed import ShadowArrays, SvfArrays

        logger.info("Checking for preprocessing data...")

        result = {
            "wall_height_arr": None,
            "wall_height_transform": None,
            "wall_aspect_arr": None,
            "wall_aspect_transform": None,
            "svf_data": None,
            "shadow_data": None,
            "compute_walls": False,
            "compute_svf": False,
        }

        # Load walls with auto-discovery
        if wall_height is not None and wall_aspect is not None:
            # Explicit paths provided - use them
            result["wall_height_arr"], result["wall_height_transform"], _, _ = io.load_raster(str(wall_height))
            result["wall_aspect_arr"], result["wall_aspect_transform"], _, _ = io.load_raster(str(wall_aspect))
            logger.info("  ✓ Existing walls found (will use precomputed)")

        elif wall_height is not None or wall_aspect is not None:
            logger.warning("  ⚠ Only one wall file provided - both wall_height and wall_aspect required")
            logger.info("  → Walls will be computed from DSM and cached")
            result["compute_walls"] = True

        else:
            # Try to auto-discover walls in working_dir (unless force_recompute)
            if force_recompute:
                logger.info("  → force_recompute=True - will recompute walls from DSM and cache")
                result["compute_walls"] = True
            else:
                walls_cache_dir = working_path / "walls"
                wall_hts_path = walls_cache_dir / "wall_hts.tif"
                wall_aspects_path = walls_cache_dir / "wall_aspects.tif"

                if wall_hts_path.exists() and wall_aspects_path.exists():
                    # Files exist - load them
                    result["wall_height_arr"], result["wall_height_transform"], _, _ = io.load_raster(
                        str(wall_hts_path)
                    )
                    result["wall_aspect_arr"], result["wall_aspect_transform"], _, _ = io.load_raster(
                        str(wall_aspects_path)
                    )
                    logger.info(f"  ✓ Walls found in working_dir: {walls_cache_dir}")
                else:
                    # No cached walls - will compute and cache
                    logger.info("  → No walls found in working_dir - will compute from DSM and cache")
                    result["compute_walls"] = True

        # Helper to load SVF, preferring memmap for efficiency
        def load_svf_from_dir(svf_path: Path) -> SvfArrays | None:
            memmap_dir = svf_path / "memmap"
            svf_zip_path = svf_path / "svfs.zip"

            # Prefer memmap (more efficient for large rasters)
            if memmap_dir.exists() and (memmap_dir / "svf.npy").exists():
                svf_data = SvfArrays.from_memmap(memmap_dir)
                logger.info("  ✓ SVF loaded from memmap (memory-efficient)")
                return svf_data
            elif svf_zip_path.exists():
                svf_data = SvfArrays.from_zip(str(svf_zip_path))
                logger.info("  ✓ SVF loaded from zip")
                return svf_data
            return None

        # Load SVF with auto-discovery
        if svf_dir is not None:
            # Explicit SVF directory provided - use it
            svf_path = Path(svf_dir)
            shadow_npz_path = svf_path / "shadowmats.npz"

            svf_data = load_svf_from_dir(svf_path)
            if svf_data is not None:
                result["svf_data"] = svf_data
                logger.info("  ✓ Existing SVF found (will use precomputed)")

                if shadow_npz_path.exists():
                    result["shadow_data"] = ShadowArrays.from_npz(str(shadow_npz_path))
                    logger.info("  ✓ Existing shadow matrices found (anisotropic sky enabled)")
            else:
                logger.info(f"  → SVF directory provided but no SVF files found: {svf_path}")
                logger.info("  → SVF will be computed and cached")
                result["compute_svf"] = True

        else:
            # Try to auto-discover SVF in working_dir (unless force_recompute)
            if force_recompute:
                logger.info("  → force_recompute=True - will recompute SVF and cache")
                result["compute_svf"] = True
            else:
                svf_cache_dir = working_path / "svf"
                shadow_npz_path = svf_cache_dir / "shadowmats.npz"

                svf_data = load_svf_from_dir(svf_cache_dir)
                if svf_data is not None:
                    result["svf_data"] = svf_data
                    logger.info(f"  ✓ SVF found in working_dir: {svf_cache_dir}")

                    if shadow_npz_path.exists():
                        result["shadow_data"] = ShadowArrays.from_npz(str(shadow_npz_path))
                        logger.info("  ✓ Shadow matrices found (anisotropic sky enabled)")
                else:
                    # No cached SVF - will compute and cache
                    logger.info("  → No SVF found in working_dir - will compute and cache")
                    result["compute_svf"] = True

        return result

    @staticmethod
    def _align_rasters(
        dsm_arr,
        dsm_transform,
        dsm_crs,
        pixel_size: float,
        terrain_rasters: dict,
        preprocess_data: dict,
        bbox: list[float] | None,
    ) -> dict:
        """
        Compute extent, validate bbox, and resample all rasters to common grid.

        Args:
            dsm_arr: DSM array.
            dsm_transform: DSM geotransform.
            dsm_crs: DSM CRS.
            pixel_size: Target pixel size in meters.
            terrain_rasters: Dictionary with terrain raster data.
            preprocess_data: Dictionary with preprocessing data.
            bbox: Optional explicit bounding box.

        Returns:
            Dictionary with all aligned rasters and metadata.
        """
        logger.info("Computing spatial extent and resolution...")

        # Extract bounds from all loaded rasters
        bounds_list = [extract_bounds(dsm_transform, dsm_arr.shape)]

        if terrain_rasters["cdsm_arr"] is not None and terrain_rasters["cdsm_transform"] is not None:
            bounds_list.append(extract_bounds(terrain_rasters["cdsm_transform"], terrain_rasters["cdsm_arr"].shape))
        if terrain_rasters["dem_arr"] is not None and terrain_rasters["dem_transform"] is not None:
            bounds_list.append(extract_bounds(terrain_rasters["dem_transform"], terrain_rasters["dem_arr"].shape))
        if terrain_rasters["tdsm_arr"] is not None and terrain_rasters["tdsm_transform"] is not None:
            bounds_list.append(extract_bounds(terrain_rasters["tdsm_transform"], terrain_rasters["tdsm_arr"].shape))
        if terrain_rasters["land_cover_arr"] is not None and terrain_rasters["land_cover_transform"] is not None:
            bounds_list.append(
                extract_bounds(terrain_rasters["land_cover_transform"], terrain_rasters["land_cover_arr"].shape)
            )
        if preprocess_data["wall_height_arr"] is not None and preprocess_data["wall_height_transform"] is not None:
            bounds_list.append(
                extract_bounds(preprocess_data["wall_height_transform"], preprocess_data["wall_height_arr"].shape)
            )
        if preprocess_data["wall_aspect_arr"] is not None and preprocess_data["wall_aspect_transform"] is not None:
            bounds_list.append(
                extract_bounds(preprocess_data["wall_aspect_transform"], preprocess_data["wall_aspect_arr"].shape)
            )

        # Determine target bounding box
        if bbox is not None:
            # User provided explicit bbox - validate it's within intersection
            computed_intersection = intersect_bounds(bounds_list)
            user_minx, user_miny, user_maxx, user_maxy = bbox
            int_minx, int_miny, int_maxx, int_maxy = computed_intersection

            # Check if user bbox is within or equal to intersection
            if (
                user_minx < int_minx - 1e-6
                or user_maxx > int_maxx + 1e-6
                or user_miny < int_miny - 1e-6
                or user_maxy > int_maxy + 1e-6
            ):
                raise ValueError(
                    f"Specified bbox {bbox} extends beyond the intersection of input rasters "
                    f"{computed_intersection}. Bbox must be within or equal to the intersection."
                )

            target_bbox = bbox
            logger.info(f"  Using user-specified extent: {target_bbox}")
        else:
            # Auto-compute intersection
            target_bbox = intersect_bounds(bounds_list)
            logger.info(f"  Auto-computed extent from raster intersection: {target_bbox}")

        # Check if resampling is needed (compare DSM to target)
        dsm_bounds = extract_bounds(dsm_transform, dsm_arr.shape)
        dsm_pixel_size = abs(dsm_transform[1])
        needs_resampling = (
            abs(dsm_bounds[0] - target_bbox[0]) > 1e-6
            or abs(dsm_bounds[1] - target_bbox[1]) > 1e-6
            or abs(dsm_bounds[2] - target_bbox[2]) > 1e-6
            or abs(dsm_bounds[3] - target_bbox[3]) > 1e-6
            or abs(dsm_pixel_size - pixel_size) > 1e-6
        )

        if needs_resampling:
            logger.info("  Resampling all rasters to target grid...")

            # Resample DSM
            dsm_arr, dsm_transform = resample_to_grid(
                dsm_arr, dsm_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
            )

            # Resample optional terrain rasters
            if terrain_rasters["cdsm_arr"] is not None and terrain_rasters["cdsm_transform"] is not None:
                terrain_rasters["cdsm_arr"], _ = resample_to_grid(
                    terrain_rasters["cdsm_arr"],
                    terrain_rasters["cdsm_transform"],
                    target_bbox,
                    pixel_size,
                    method="bilinear",
                    src_crs=dsm_crs,
                )
            if terrain_rasters["dem_arr"] is not None and terrain_rasters["dem_transform"] is not None:
                terrain_rasters["dem_arr"], _ = resample_to_grid(
                    terrain_rasters["dem_arr"],
                    terrain_rasters["dem_transform"],
                    target_bbox,
                    pixel_size,
                    method="bilinear",
                    src_crs=dsm_crs,
                )
            if terrain_rasters["tdsm_arr"] is not None and terrain_rasters["tdsm_transform"] is not None:
                terrain_rasters["tdsm_arr"], _ = resample_to_grid(
                    terrain_rasters["tdsm_arr"],
                    terrain_rasters["tdsm_transform"],
                    target_bbox,
                    pixel_size,
                    method="bilinear",
                    src_crs=dsm_crs,
                )
            if terrain_rasters["land_cover_arr"] is not None and terrain_rasters["land_cover_transform"] is not None:
                # Use nearest neighbor for categorical data
                terrain_rasters["land_cover_arr"], _ = resample_to_grid(
                    terrain_rasters["land_cover_arr"],
                    terrain_rasters["land_cover_transform"],
                    target_bbox,
                    pixel_size,
                    method="nearest",
                    src_crs=dsm_crs,
                )

            # Resample preprocessing data
            if preprocess_data["wall_height_arr"] is not None and preprocess_data["wall_height_transform"] is not None:
                preprocess_data["wall_height_arr"], _ = resample_to_grid(
                    preprocess_data["wall_height_arr"],
                    preprocess_data["wall_height_transform"],
                    target_bbox,
                    pixel_size,
                    method="bilinear",
                    src_crs=dsm_crs,
                )
            if preprocess_data["wall_aspect_arr"] is not None and preprocess_data["wall_aspect_transform"] is not None:
                preprocess_data["wall_aspect_arr"], _ = resample_to_grid(
                    preprocess_data["wall_aspect_arr"],
                    preprocess_data["wall_aspect_transform"],
                    target_bbox,
                    pixel_size,
                    method="bilinear",
                    src_crs=dsm_crs,
                )

            # Note: SVF resampling is more complex (multiple arrays) - handled separately if needed
            if preprocess_data["svf_data"] is not None and preprocess_data["svf_data"].svf.shape != dsm_arr.shape:
                logger.warning(
                    f"  ⚠ SVF shape {preprocess_data['svf_data'].svf.shape} doesn't match target shape "
                    f"{dsm_arr.shape} - SVF resampling not yet implemented. "
                    f"SVF will be recomputed on-the-fly if needed."
                )
                preprocess_data["svf_data"] = None
                preprocess_data["shadow_data"] = None

            logger.info(f"  ✓ Resampled to {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")
        else:
            logger.info("  ✓ No resampling needed - all rasters match target grid")

        # Return all aligned data
        return {
            "dsm_arr": dsm_arr,
            "dsm_transform": dsm_transform,
            "dsm_crs": dsm_crs,
            "cdsm_arr": terrain_rasters["cdsm_arr"],
            "dem_arr": terrain_rasters["dem_arr"],
            "tdsm_arr": terrain_rasters["tdsm_arr"],
            "land_cover_arr": terrain_rasters["land_cover_arr"],
            "wall_height_arr": preprocess_data["wall_height_arr"],
            "wall_aspect_arr": preprocess_data["wall_aspect_arr"],
            "svf_data": preprocess_data["svf_data"],
            "shadow_data": preprocess_data["shadow_data"],
        }

    @classmethod
    def _create_surface_instance(
        cls,
        aligned_rasters: dict,
        pixel_size: float,
        trunk_ratio: float,
        relative_heights: bool,
    ) -> SurfaceData:
        """
        Create SurfaceData instance from aligned rasters.

        Args:
            aligned_rasters: Dictionary with all aligned rasters and metadata.
            pixel_size: Pixel size in meters.
            trunk_ratio: Trunk ratio for auto-generating TDSM from CDSM.
            relative_heights: Whether CDSM/TDSM contain relative heights.

        Returns:
            SurfaceData instance with loaded terrain and preprocessing data.
        """
        from affine import Affine as AffineClass

        # Create SurfaceData instance
        surface_data = cls(
            dsm=aligned_rasters["dsm_arr"],
            cdsm=aligned_rasters["cdsm_arr"],
            dem=aligned_rasters["dem_arr"],
            tdsm=aligned_rasters["tdsm_arr"],
            land_cover=aligned_rasters["land_cover_arr"],
            wall_height=aligned_rasters["wall_height_arr"],
            wall_aspect=aligned_rasters["wall_aspect_arr"],
            svf=aligned_rasters["svf_data"],
            shadow_matrices=aligned_rasters["shadow_data"],
            pixel_size=pixel_size,
            trunk_ratio=trunk_ratio,
            relative_heights=relative_heights,
        )

        # Store geotransform and CRS for later export
        dsm_transform = aligned_rasters["dsm_transform"]
        if isinstance(dsm_transform, AffineClass):
            surface_data._geotransform = list(dsm_transform.to_gdal())
        else:
            surface_data._geotransform = dsm_transform
        surface_data._crs_wkt = aligned_rasters["dsm_crs"]

        # Log what was loaded
        layers_loaded = ["DSM"]
        if aligned_rasters["cdsm_arr"] is not None:
            layers_loaded.append("CDSM")
        if aligned_rasters["dem_arr"] is not None:
            layers_loaded.append("DEM")
        if aligned_rasters["tdsm_arr"] is not None:
            layers_loaded.append("TDSM")
        if aligned_rasters["land_cover_arr"] is not None:
            layers_loaded.append("land_cover")
        logger.info(f"  Layers loaded: {', '.join(layers_loaded)}")

        return surface_data

    @staticmethod
    def _compute_and_cache_walls(
        surface_data: SurfaceData,
        aligned_rasters: dict,
        working_path: Path,
    ) -> None:
        """
        Compute wall heights/aspects from DSM and cache to working_dir.

        Args:
            surface_data: SurfaceData instance to update with computed walls.
            aligned_rasters: Dictionary with aligned raster data.
            working_path: Working directory for caching.
        """

        logger.info("Computing walls from DSM and caching to working_dir...")
        walls_cache_dir = working_path / "walls"

        # Save resampled DSM to working_dir so wall computation can use it
        resampled_dir = working_path / "resampled"
        resampled_dir.mkdir(parents=True, exist_ok=True)
        resampled_dsm_path = resampled_dir / "dsm_resampled.tif"

        dsm_transform = aligned_rasters["dsm_transform"]
        io.save_raster(
            str(resampled_dsm_path),
            aligned_rasters["dsm_arr"],
            list(dsm_transform.to_gdal()) if isinstance(dsm_transform, AffineClass) else dsm_transform,
            aligned_rasters["dsm_crs"],
        )

        # Generate walls using the walls module
        walls_module.generate_wall_hts(
            dsm_path=str(resampled_dsm_path),
            bbox=None,  # Already resampled to target extent
            out_dir=str(walls_cache_dir),
        )

        # Load the generated walls back into surface_data
        wall_hts_path = walls_cache_dir / "wall_hts.tif"
        wall_aspects_path = walls_cache_dir / "wall_aspects.tif"

        if wall_hts_path.exists() and wall_aspects_path.exists():
            wall_height_arr, _, _, _ = io.load_raster(str(wall_hts_path))
            wall_aspect_arr, _, _, _ = io.load_raster(str(wall_aspects_path))
            surface_data.wall_height = wall_height_arr
            surface_data.wall_aspect = wall_aspect_arr
            logger.info(f"  ✓ Walls computed and cached to {walls_cache_dir}")
        else:
            logger.warning("  ⚠ Wall generation completed but files not found")

    @staticmethod
    def _compute_and_cache_svf(
        surface_data: SurfaceData,
        aligned_rasters: dict,
        working_path: Path,
        trunk_ratio: float,
    ) -> None:
        """
        Compute SVF from DSM/CDSM/TDSM and cache to working_dir.

        Args:
            surface_data: SurfaceData instance to update with computed SVF.
            aligned_rasters: Dictionary with aligned raster data.
            working_path: Working directory for caching.
            trunk_ratio: Trunk ratio for SVF computation.
        """

        dsm_arr = aligned_rasters["dsm_arr"]
        cdsm_arr = aligned_rasters["cdsm_arr"]
        tdsm_arr = aligned_rasters["tdsm_arr"]
        pixel_size = aligned_rasters.get("pixel_size", 1.0)

        use_veg = cdsm_arr is not None
        if use_veg:
            logger.info("Computing SVF from DSM/CDSM/TDSM...")
        else:
            logger.info("Computing SVF from DSM...")

        # Prepare vegetation arrays (Rust requires all three or none)
        if use_veg:
            cdsm_for_svf = cdsm_arr.astype(np.float32)
            # Auto-generate TDSM if not provided
            if tdsm_arr is not None:
                tdsm_for_svf = tdsm_arr.astype(np.float32)
            else:
                tdsm_for_svf = (cdsm_arr * trunk_ratio).astype(np.float32)
        else:
            cdsm_for_svf = np.zeros_like(dsm_arr, dtype=np.float32)
            tdsm_for_svf = np.zeros_like(dsm_arr, dtype=np.float32)

        # Compute max height for SVF calculation
        max_height = float(np.nanmax(dsm_arr))
        if use_veg and cdsm_arr is not None:
            veg_max = float(np.nanmax(cdsm_arr))
            max_height = max(max_height, veg_max)

        # Compute SVF using Rust module
        svf_result = skyview.calculate_svf(
            dsm_arr.astype(np.float32),
            cdsm_for_svf,
            tdsm_for_svf,
            pixel_size,
            use_veg,
            max_height,
            2,  # patch_option (153 patches)
            3.0,  # min_sun_elev_deg
            None,  # progress callback
        )

        # Create SvfArrays from result
        svf_data = SvfArrays(
            svf=np.array(svf_result.svf),
            svf_north=np.array(svf_result.svf_north),
            svf_east=np.array(svf_result.svf_east),
            svf_south=np.array(svf_result.svf_south),
            svf_west=np.array(svf_result.svf_west),
            svf_veg=np.array(svf_result.svf_veg) if use_veg else np.ones_like(dsm_arr, dtype=np.float32),
            svf_veg_north=np.array(svf_result.svf_veg_north) if use_veg else np.ones_like(dsm_arr, dtype=np.float32),
            svf_veg_east=np.array(svf_result.svf_veg_east) if use_veg else np.ones_like(dsm_arr, dtype=np.float32),
            svf_veg_south=np.array(svf_result.svf_veg_south) if use_veg else np.ones_like(dsm_arr, dtype=np.float32),
            svf_veg_west=np.array(svf_result.svf_veg_west) if use_veg else np.ones_like(dsm_arr, dtype=np.float32),
            svf_aveg=np.array(svf_result.svf_veg_blocks_bldg_sh)
            if use_veg
            else np.ones_like(dsm_arr, dtype=np.float32),
            svf_aveg_north=np.array(svf_result.svf_veg_blocks_bldg_sh_north)
            if use_veg
            else np.ones_like(dsm_arr, dtype=np.float32),
            svf_aveg_east=np.array(svf_result.svf_veg_blocks_bldg_sh_east)
            if use_veg
            else np.ones_like(dsm_arr, dtype=np.float32),
            svf_aveg_south=np.array(svf_result.svf_veg_blocks_bldg_sh_south)
            if use_veg
            else np.ones_like(dsm_arr, dtype=np.float32),
            svf_aveg_west=np.array(svf_result.svf_veg_blocks_bldg_sh_west)
            if use_veg
            else np.ones_like(dsm_arr, dtype=np.float32),
        )

        # Cache to working_dir for future reuse
        svf_cache_dir = working_path / "svf"
        svf_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache metadata for validation on reload
        metadata = CacheMetadata.from_arrays(dsm_arr, pixel_size, cdsm_arr)

        # Save as memmap for efficient large-raster access
        memmap_dir = svf_cache_dir / "memmap"
        svf_data.to_memmap(memmap_dir, metadata=metadata)

        # Store on surface_data for immediate use
        surface_data.svf = svf_data
        logger.info(f"  ✓ SVF computed and cached to {svf_cache_dir}")

    def preprocess(self) -> None:
        """
        Preprocess CDSM/TDSM from relative to absolute heights.

        Call this method if your CDSM and TDSM contain relative vegetation heights
        (height above ground) rather than absolute surface heights (elevation).

        This method:
        1. Auto-generates TDSM from CDSM * trunk_ratio if TDSM is not provided
        2. Converts CDSM/TDSM from relative to absolute heights by adding DEM
           (or DSM if DEM is not provided)
        3. Zeros out pixels with height < 0.1m (below meaningful vegetation threshold)

        The preprocessing matches the logic in configs.py raster_preprocessing().

        Note:
            This method modifies CDSM and TDSM in-place.
            If CDSM/TDSM are already absolute heights, do NOT call this method.
        """
        if self._preprocessed:
            return

        # Auto-generate TDSM from trunk ratio if CDSM provided but not TDSM
        if self.cdsm is not None and self.tdsm is None:
            logger.info(f"Auto-generating TDSM from CDSM using trunk_ratio={self.trunk_ratio}")
            self.tdsm = (self.cdsm * self.trunk_ratio).astype(np.float32)

        # Boost CDSM/TDSM to absolute heights
        if self.cdsm is not None:
            threshold = np.float32(0.1)
            zero32 = np.float32(0.0)
            nan32 = np.float32(np.nan)

            # Use DEM as base if available, otherwise DSM
            base = self.dem if self.dem is not None else self.dsm

            # Store original relative heights for comparison
            cdsm_rel = self.cdsm.copy()

            # CDSM = base + relative_cdsm
            cdsm_abs = np.where(~np.isnan(base), base + cdsm_rel, nan32)
            # Zero out pixels where boosted height is below threshold
            cdsm_abs = np.where(cdsm_abs - base < threshold, zero32, cdsm_abs)
            self.cdsm = cdsm_abs.astype(np.float32)

            # TDSM = base + relative_tdsm
            if self.tdsm is not None:
                tdsm_rel = self.tdsm.copy()
                tdsm_abs = np.where(~np.isnan(base), base + tdsm_rel, nan32)
                tdsm_abs = np.where(tdsm_abs - base < threshold, zero32, tdsm_abs)
                self.tdsm = tdsm_abs.astype(np.float32)

            logger.info(
                f"Preprocessed CDSM/TDSM to absolute heights (base: {'DEM' if self.dem is not None else 'DSM'})"
            )

        self._preprocessed = True

    @property
    def max_height(self) -> float:
        """Auto-compute maximum height difference for shadow buffer calculation.

        Considers both DSM (buildings) and CDSM (vegetation) since both cast shadows.
        Returns max elevation minus ground level.
        """
        dsm_max = float(np.nanmax(self.dsm))
        ground_min = float(np.nanmin(self.dsm))

        # Also consider vegetation if present (CDSM may be taller than buildings)
        if self.cdsm is not None:
            cdsm_max = float(np.nanmax(self.cdsm))
            # After preprocessing, CDSM contains absolute elevations
            # Use the higher of DSM or CDSM
            max_elevation = max(dsm_max, cdsm_max)
        else:
            max_elevation = dsm_max

        return max_elevation - ground_min

    @property
    def shape(self) -> tuple[int, int]:
        """Return DSM shape (rows, cols)."""
        rows, cols = self.dsm.shape
        return (rows, cols)

    @property
    def crs(self) -> str | None:
        """Return CRS as WKT string, or None if not set."""
        return self._crs_wkt

    def get_buffer_pool(self) -> BufferPool:
        """Get or create a buffer pool for this surface.

        The buffer pool provides pre-allocated numpy arrays that can be
        reused across timesteps during timeseries calculations. This
        reduces memory allocation overhead and GC pressure.

        Returns:
            BufferPool sized to this surface's grid dimensions.

        Example:
            pool = surface.get_buffer_pool()
            temp = pool.get_zeros("ani_lum")  # First call allocates
            temp = pool.get_zeros("ani_lum")  # Second call reuses same memory
        """
        if self._buffer_pool is None:
            self._buffer_pool = BufferPool(self.shape)
        return self._buffer_pool

    def clear_buffers(self) -> None:
        """Clear the buffer pool to free memory.

        Call this after completing a timeseries calculation to release
        the pre-allocated arrays.
        """
        if self._buffer_pool is not None:
            self._buffer_pool.clear()
            self._buffer_pool = None

    def _looks_like_relative_heights(self) -> bool:
        """
        Heuristic check if CDSM appears to contain relative heights.

        Returns True if max(CDSM) is much smaller than min(DSM), suggesting
        CDSM contains height-above-ground values rather than absolute elevations.

        This is used to warn users who may have forgotten to call preprocess().
        """
        if self.cdsm is None:
            return False

        cdsm_max = np.nanmax(self.cdsm)
        dsm_min = np.nanmin(self.dsm)

        # If CDSM max is much smaller than DSM min, it's likely relative heights
        # Typical case: DSM min ~100m elevation, CDSM max ~30m tree height
        # Exception: coastal areas where DSM min could be near 0
        if dsm_min > 10 and cdsm_max < dsm_min * 0.5:
            return True

        # Also check if CDSM values are typical vegetation heights (0-50m range)
        # while DSM has larger values
        return bool(cdsm_max < 60 and dsm_min > cdsm_max + 20)

    def _check_preprocessing_needed(self) -> None:
        """
        Warn if CDSM appears to need preprocessing but wasn't preprocessed.

        Called internally before calculations to alert users.
        """
        if self.cdsm is None:
            return

        if self.relative_heights and not self._preprocessed and self._looks_like_relative_heights():
            logger.warning(
                f"CDSM appears to contain relative vegetation heights "
                f"(max CDSM={np.nanmax(self.cdsm):.1f}m < min DSM={np.nanmin(self.dsm):.1f}m), "
                f"but preprocess() was not called. "
                f"Call surface.preprocess() to convert to absolute heights, "
                f"or set relative_heights=False if CDSM already contains absolute elevations."
            )

    def get_land_cover_properties(
        self,
        params: SimpleNamespace | None = None,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Derive surface properties from land cover grid.

        Args:
            params: Optional loaded parameters from JSON file (via load_params()).
                When provided, land cover properties are read from the params.
                When None, uses built-in defaults matching parametersforsolweig.json.

        Returns:
            Tuple of (albedo_grid, emissivity_grid, tgk_grid, tstart_grid, tmaxlst_grid).
            If land_cover is None, returns defaults.

        Land cover parameters from Lindberg et al. 2008, 2016 (parametersforsolweig.json):
            - TgK (Ts_deg): Temperature coefficient for surface heating
            - Tstart: Temperature offset at sunrise
            - TmaxLST: Hour of maximum local surface temperature
        """
        if self.land_cover is None:
            # Use provided grids or defaults
            alb = self.albedo if self.albedo is not None else np.full_like(self.dsm, 0.15)
            emis = self.emissivity if self.emissivity is not None else np.full_like(self.dsm, 0.95)
            tgk = np.full_like(self.dsm, 0.37)  # Default TgK (cobblestone)
            tstart = np.full_like(self.dsm, -3.41)  # Default Tstart (cobblestone)
            tmaxlst = np.full_like(self.dsm, 15.0)  # Default TmaxLST (cobblestone)
            return alb, emis, tgk, tstart, tmaxlst

        # If params provided, use the helper function to extract from JSON
        if params is not None:
            return get_lc_properties_from_params(self.land_cover, params, self.shape)

        # UMEP standard land cover properties from parametersforsolweig.json
        # ID: (albedo, emissivity, TgK, Tstart, TmaxLST)
        # Values must match the JSON parameters file for parity with runner
        lc_properties = {
            0: (0.20, 0.95, 0.37, -3.41, 15.0),  # Paved/cobblestone (Cobble_stone_2014a)
            1: (0.18, 0.95, 0.58, -9.78, 15.0),  # Dark asphalt (albedo from JSON)
            2: (0.18, 0.95, 0.58, -9.78, 15.0),  # Buildings/roofs (emissivity=0.95, albedo=0.18)
            3: (0.20, 0.95, 0.37, -3.41, 15.0),  # Undefined (use paved defaults)
            4: (0.20, 0.95, 0.37, -3.41, 15.0),  # Undefined (use paved defaults)
            5: (0.16, 0.94, 0.21, -3.38, 14.0),  # Grass (Grass_unmanaged) - albedo=0.16, emis=0.94
            6: (0.25, 0.94, 0.33, -3.01, 14.0),  # Bare soil - emis=0.94
            7: (0.05, 0.98, 0.00, 0.00, 12.0),  # Water - albedo=0.05
        }

        rows, cols = self.shape
        alb_grid = np.full((rows, cols), 0.15, dtype=np.float32)
        emis_grid = np.full((rows, cols), 0.95, dtype=np.float32)
        tgk_grid = np.full((rows, cols), 0.37, dtype=np.float32)
        tstart_grid = np.full((rows, cols), -3.41, dtype=np.float32)
        tmaxlst_grid = np.full((rows, cols), 15.0, dtype=np.float32)

        lc = self.land_cover
        for lc_id, (alb, emis, tgk, tstart, tmaxlst) in lc_properties.items():
            mask = lc == lc_id
            if np.any(mask):
                alb_grid[mask] = alb
                emis_grid[mask] = emis
                tgk_grid[mask] = tgk
                tstart_grid[mask] = tstart
                tmaxlst_grid[mask] = tmaxlst

        return alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid
