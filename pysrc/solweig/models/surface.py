"""Surface and terrain data models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

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
from .precomputed import ShadowArrays, SvfArrays

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


def _save_svfs_zip(svf_data: SvfArrays, svf_cache_dir: Path, aligned_rasters: dict) -> None:
    """Save SVF arrays as svfs.zip for PrecomputedData.prepare() compatibility."""
    import tempfile
    import zipfile

    geotransform = aligned_rasters.get("dsm_transform")
    crs_wkt = aligned_rasters.get("dsm_crs")

    # If geotransform/CRS not available, skip zip (memmap still works)
    if geotransform is None:
        logger.debug("  Skipping svfs.zip (no geotransform available)")
        return

    svf_files = {
        "svf.tif": svf_data.svf,
        "svfN.tif": svf_data.svf_north,
        "svfE.tif": svf_data.svf_east,
        "svfS.tif": svf_data.svf_south,
        "svfW.tif": svf_data.svf_west,
        "svfveg.tif": svf_data.svf_veg,
        "svfNveg.tif": svf_data.svf_veg_north,
        "svfEveg.tif": svf_data.svf_veg_east,
        "svfSveg.tif": svf_data.svf_veg_south,
        "svfWveg.tif": svf_data.svf_veg_west,
        "svfaveg.tif": svf_data.svf_aveg,
        "svfNaveg.tif": svf_data.svf_aveg_north,
        "svfEaveg.tif": svf_data.svf_aveg_east,
        "svfSaveg.tif": svf_data.svf_aveg_south,
        "svfWaveg.tif": svf_data.svf_aveg_west,
    }

    # Convert Affine to GDAL geotransform list if needed
    if isinstance(geotransform, AffineClass):
        geotransform = [geotransform.c, geotransform.a, geotransform.b, geotransform.f, geotransform.d, geotransform.e]

    svf_zip_path = svf_cache_dir / "svfs.zip"
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, arr in svf_files.items():
            if arr is not None:
                tif_path = str(Path(tmpdir) / filename)
                io.save_raster(tif_path, arr, geotransform, crs_wkt)
        with zipfile.ZipFile(str(svf_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in svf_files:
                tif_file = Path(tmpdir) / filename
                if tif_file.exists():
                    zf.write(str(tif_file), filename)

    logger.info(f"  ✓ SVF saved as {svf_zip_path}")


def _save_shadow_matrices(svf_result, svf_cache_dir: Path, patch_count: int = 153) -> None:
    """Save shadow matrices as shadowmats.npz for anisotropic sky model."""
    # Shadow matrices are bitpacked uint8 from Rust: shape (rows, cols, ceil(patches/8))
    shadow_path = svf_cache_dir / "shadowmats.npz"
    np.savez_compressed(
        str(shadow_path),
        shadowmat=np.array(svf_result.bldg_sh_matrix),
        vegshadowmat=np.array(svf_result.veg_sh_matrix),
        vbshmat=np.array(svf_result.veg_blocks_bldg_sh_matrix),
        patch_count=np.array(patch_count),
    )

    logger.info(f"  ✓ Shadow matrices saved as {shadow_path}")


@dataclass
class SurfaceData:
    """
    Surface/terrain data for SOLWEIG calculations.

    Only `dsm` is required. Other rasters are optional and will be
    treated as absent if not provided.

    Attributes:
        dsm: Digital Surface Model (elevation in meters). Required.
        cdsm: Canopy Digital Surface Model (vegetation heights). Optional.
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
        dsm_relative: Whether DSM contains relative heights (above ground)
            rather than absolute elevations. Default False. If True, DEM is
            required and preprocess() converts DSM to absolute via DSM + DEM.
        cdsm_relative: Whether CDSM contains relative heights. Default True.
            If True and preprocess() is not called, a warning is issued.
        tdsm_relative: Whether TDSM contains relative heights. Default True.
            If True and preprocess() is not called, a warning is issued.

    Note:
        Albedo and emissivity are derived internally from land_cover using
        standard UMEP parameters. They cannot be directly specified.

    Note:
        max_height is auto-computed from dsm as: np.nanmax(dsm) - np.nanmin(dsm)

    Height Conventions:
        Each raster layer can independently use relative or absolute heights.
        The per-layer flags (``dsm_relative``, ``cdsm_relative``,
        ``tdsm_relative``) control the convention for each layer.

        **Relative Heights** (height above ground):
            - CDSM/TDSM: vegetation height above ground (e.g., 6m tree)
            - DSM: building/surface height above ground (requires DEM)
            - Typical range: 0-40m for CDSM, 0-10m for TDSM
            - Must call ``preprocess()`` before calculations

        **Absolute Heights** (elevation above sea level):
            - Values in the same vertical reference system
            - Example: DSM=127m, CDSM=133m means 6m vegetation
            - No preprocessing needed

        The internal algorithms (Rust) always use **absolute heights**. The
        ``preprocess()`` method converts relative → absolute using:
            dsm_absolute = dem + dsm_relative  (requires DEM)
            cdsm_absolute = base + cdsm_relative
            tdsm_absolute = base + tdsm_relative
        where ``base = DEM`` if available, else ``base = DSM``.

    Example:
        # Relative CDSM (common case):
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_rel)
        surface.preprocess()  # Converts CDSM to absolute

        # Absolute CDSM:
        surface = SurfaceData(dsm=dsm, cdsm=cdsm_abs, cdsm_relative=False)

        # Mixed: absolute DSM, relative CDSM, absolute TDSM:
        surface = SurfaceData(
            dsm=dsm, cdsm=cdsm, tdsm=tdsm,
            cdsm_relative=True, tdsm_relative=False,
        )
        surface.preprocess()  # Only converts CDSM

        # Relative DSM (requires DEM):
        surface = SurfaceData(dsm=ndsm, dem=dem, dsm_relative=True)
        surface.preprocess()  # Converts DSM to absolute via DEM + nDSM
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
    dsm_relative: bool = False  # Whether DSM contains relative heights (requires DEM)
    cdsm_relative: bool = True  # Whether CDSM contains relative heights
    tdsm_relative: bool = True  # Whether TDSM contains relative heights

    # Internal state
    _nan_filled: bool = field(default=False, init=False, repr=False)
    _preprocessed: bool = field(default=False, init=False, repr=False)
    _geotransform: list[float] | None = field(default=None, init=False, repr=False)  # GDAL geotransform
    _crs_wkt: str | None = field(default=None, init=False, repr=False)  # CRS as WKT string
    _buffer_pool: BufferPool | None = field(default=None, init=False, repr=False)  # Reusable array pool
    _gvf_geometry_cache: object = field(default=None, init=False, repr=False)  # Rust GVF geometry cache
    _valid_mask: NDArray[np.bool_] | None = field(default=None, init=False, repr=False)  # Combined valid mask

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
        dsm_relative: bool = False,
        cdsm_relative: bool = True,
        tdsm_relative: bool = True,
        force_recompute: bool = False,
        feedback: Any = None,
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
            dsm_relative: Whether DSM contains relative heights. Default False.
            cdsm_relative: Whether CDSM contains relative heights. Default True.
            tdsm_relative: Whether TDSM contains relative heights. Default True.
            force_recompute: If True, skip cache and recompute walls/SVF even if they
                exist in working_dir. Default False (use cached data when available).
            feedback: Optional QGIS QgsProcessingFeedback for progress/cancellation.

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

            # Mixed height conventions
            surface = SurfaceData.prepare(
                dsm="data/dsm.tif",
                cdsm="data/cdsm.tif",
                tdsm="data/tdsm.tif",
                cdsm_relative=True,
                tdsm_relative=False,
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
            dsm_relative=dsm_relative,
            cdsm_relative=cdsm_relative,
            tdsm_relative=tdsm_relative,
        )

        # Validate cached SVF against current inputs (if SVF was loaded)
        if preprocess_data["svf_data"] is not None and not force_recompute:
            dsm_arr = aligned_rasters["dsm_arr"]
            cdsm_arr = aligned_rasters.get("cdsm_arr")
            svf_source = preprocess_data.get("svf_source", "none")

            cache_valid = False
            if svf_source == "memmap":
                # Memmap has cache_meta.json — use hash-based validation
                svf_cache_dir = working_path / "svf" / "memmap"
                cache_valid = validate_cache(svf_cache_dir, dsm_arr, pixel_size, cdsm_arr)
            elif svf_source == "zip":
                # Zip has no metadata — validate by shape match only
                svf_shape = preprocess_data["svf_data"].svf.shape
                cache_valid = svf_shape == dsm_arr.shape
                if not cache_valid:
                    logger.info(f"  SVF shape {svf_shape} doesn't match DSM {dsm_arr.shape}")

            if not cache_valid:
                logger.info("  → Cache stale, clearing and recomputing SVF...")
                clear_stale_cache(working_path / "svf" / "memmap")
                # Also remove zip/npz so stale data doesn't persist
                for stale_file in ("svfs.zip", "shadowmats.npz"):
                    stale_path = working_path / "svf" / stale_file
                    if stale_path.exists():
                        stale_path.unlink()
                preprocess_data["svf_data"] = None
                preprocess_data["compute_svf"] = True
                surface_data.svf = None

        # Compute and cache walls if needed
        if preprocess_data["compute_walls"]:
            cls._compute_and_cache_walls(surface_data, aligned_rasters, working_path)

        # Compute and cache SVF if needed
        if preprocess_data["compute_svf"]:
            cls._compute_and_cache_svf(surface_data, aligned_rasters, working_path, trunk_ratio, feedback=feedback)

        # Preprocess layers with relative heights to absolute
        needs_preprocess = (
            dsm_relative
            or (cdsm_relative and surface_data.cdsm is not None)
            or (tdsm_relative and surface_data.tdsm is not None)
        )
        if needs_preprocess:
            logger.debug("  Preprocessing relative heights → absolute")
            surface_data.preprocess()

        # Compute unified valid mask, apply across all layers, crop to valid bbox
        surface_data.compute_valid_mask()
        surface_data.apply_valid_mask()
        surface_data.crop_to_valid_bbox()
        surface_data.save_cleaned(working_path)

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
        native_pixel_size = abs(dsm_transform[1])  # X pixel size from DSM
        if pixel_size is None:
            pixel_size = native_pixel_size
            logger.info(f"  Extracted pixel size from DSM: {pixel_size:.2f} m")
        else:
            # Validate against native resolution
            if pixel_size < native_pixel_size - 1e-6:
                raise ValueError(
                    f"Specified pixel_size ({pixel_size:.2f} m) is finer than the DSM native "
                    f"resolution ({native_pixel_size:.2f} m). Upsampling creates false precision. "
                    f"Use pixel_size >= {native_pixel_size:.2f} or omit to use native resolution."
                )
            if abs(pixel_size - native_pixel_size) > 1e-6:
                logger.warning(
                    f"  ⚠ Specified pixel_size ({pixel_size:.2f} m) differs from DSM native "
                    f"resolution ({native_pixel_size:.2f} m) — all rasters will be resampled"
                )
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
            "svf_source": "none",  # "memmap", "zip", or "none"
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

        # Helper to load SVF, preferring memmap for efficiency.
        # Returns (SvfArrays | None, source: str) where source is "memmap", "zip", or "none".
        def load_svf_from_dir(svf_path: Path) -> tuple[SvfArrays | None, str]:
            memmap_dir = svf_path / "memmap"
            svf_zip_path = svf_path / "svfs.zip"

            # Prefer memmap (more efficient for large rasters)
            if memmap_dir.exists() and (memmap_dir / "svf.npy").exists():
                svf_data = SvfArrays.from_memmap(memmap_dir)
                logger.info("  ✓ SVF loaded from memmap (memory-efficient)")
                return svf_data, "memmap"
            elif svf_zip_path.exists():
                svf_data = SvfArrays.from_zip(str(svf_zip_path))
                logger.info("  ✓ SVF loaded from zip")
                return svf_data, "zip"
            return None, "none"

        # Load SVF with auto-discovery
        if svf_dir is not None:
            # Explicit SVF directory provided - use it
            svf_path = Path(svf_dir)
            shadow_npz_path = svf_path / "shadowmats.npz"

            svf_data, svf_source = load_svf_from_dir(svf_path)
            if svf_data is not None:
                result["svf_data"] = svf_data
                result["svf_source"] = svf_source
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

                svf_data, svf_source = load_svf_from_dir(svf_cache_dir)
                if svf_data is not None:
                    result["svf_data"] = svf_data
                    result["svf_source"] = svf_source
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
            "pixel_size": pixel_size,
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
        *,
        dsm_relative: bool = False,
        cdsm_relative: bool = True,
        tdsm_relative: bool = True,
    ) -> SurfaceData:
        """
        Create SurfaceData instance from aligned rasters.

        Args:
            aligned_rasters: Dictionary with all aligned rasters and metadata.
            pixel_size: Pixel size in meters.
            trunk_ratio: Trunk ratio for auto-generating TDSM from CDSM.
            dsm_relative: Whether DSM contains relative heights.
            cdsm_relative: Whether CDSM contains relative heights.
            tdsm_relative: Whether TDSM contains relative heights.

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
            dsm_relative=dsm_relative,
            cdsm_relative=cdsm_relative,
            tdsm_relative=tdsm_relative,
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
        on_tile_complete: Callable | None = None,
        feedback: Any = None,
        progress_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Compute SVF from DSM/CDSM/TDSM and cache to working_dir.

        Automatically tiles the computation for large grids to avoid GPU
        buffer size limits. Tiled mode skips shadow matrix assembly (too
        large for anisotropic sky — consistent with calculate_tiled()).

        Saves three cache formats:
        - memmap/ for fast reload in Python API
        - svfs.zip for PrecomputedData.prepare() compatibility
        - shadowmats.npz for anisotropic sky model (non-tiled only)

        Args:
            surface_data: SurfaceData instance to update with computed SVF.
            aligned_rasters: Dictionary with aligned raster data.
            working_path: Working directory for caching.
            trunk_ratio: Trunk ratio for SVF computation.
            on_tile_complete: Optional callback(tile_idx, n_tiles) called after each tile
                (only invoked when tiling is used for large grids).
            feedback: Optional QGIS QgsProcessingFeedback for progress/cancellation.
            progress_range: Optional (start_pct, end_pct) for QGIS progress sub-range.
        """

        dsm_arr = aligned_rasters["dsm_arr"]
        cdsm_arr = aligned_rasters["cdsm_arr"]
        tdsm_arr = aligned_rasters["tdsm_arr"]
        pixel_size = aligned_rasters.get("pixel_size", 1.0)

        rows, cols = dsm_arr.shape
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

        # Auto-detect whether tiling is needed based on GPU buffer limits.
        # wgpu max buffer = 256 MiB. SVF staging uses ~32 bytes/pixel.
        # Use 80% headroom to trigger tiling before hitting the limit.
        _GPU_MAX_BUFFER = 268_435_456  # 256 MiB
        _BYTES_PER_PIXEL = 32  # empirical: staging buffers for SVF
        _max_pixels = int(_GPU_MAX_BUFFER * 0.8) // _BYTES_PER_PIXEL  # ~6.7M pixels
        needs_tiling = rows * cols > _max_pixels

        svf_cache_dir = working_path / "svf"
        svf_cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = CacheMetadata.from_arrays(dsm_arr, pixel_size, cdsm_arr)

        if needs_tiling:
            svf_data, (shmat_mm, vegshmat_mm, vbshmat_mm) = SurfaceData._compute_svf_tiled(
                dsm_arr.astype(np.float32),
                cdsm_for_svf,
                tdsm_for_svf,
                pixel_size,
                use_veg,
                max_height,
                svf_cache_dir,
                on_tile_complete=on_tile_complete,
                feedback=feedback,
                progress_range=progress_range,
            )
            n_patches = 153  # patch_option=2

            # Cache SVF arrays
            memmap_dir = svf_cache_dir / "memmap"
            svf_data.to_memmap(memmap_dir, metadata=metadata)
            _save_svfs_zip(svf_data, svf_cache_dir, aligned_rasters)

            # Save shadow matrices as npz for cache reload on future runs
            shadow_path = svf_cache_dir / "shadowmats.npz"
            np.savez_compressed(
                str(shadow_path),
                shadowmat=np.asarray(shmat_mm),
                vegshadowmat=np.asarray(vegshmat_mm),
                vbshmat=np.asarray(vbshmat_mm),
                patch_count=np.array(n_patches),
            )
            logger.info(f"  ✓ Shadow matrices saved as {shadow_path}")

            surface_data.svf = svf_data
            # Shadow matrices assembled from tiled memmaps (bitpacked uint8, on disk)
            surface_data.shadow_matrices = ShadowArrays(
                _shmat_u8=shmat_mm,
                _vegshmat_u8=vegshmat_mm,
                _vbshmat_u8=vbshmat_mm,
                _n_patches=n_patches,
            )
            logger.info(f"  ✓ SVF computed (tiled) and cached to {svf_cache_dir}")
        else:
            # Single-shot computation for grids that fit in GPU memory.
            # Use SkyviewRunner with threading + polling for progress and cancel.
            import threading

            from ..progress import ProgressReporter

            n_patches = 153  # patch_option=2

            runner = skyview.SkyviewRunner()
            result_box: list = [None]
            error_box: list = [None]

            def _run_svf():
                try:
                    result_box[0] = runner.calculate_svf(
                        dsm_arr.astype(np.float32),
                        cdsm_for_svf,
                        tdsm_for_svf,
                        pixel_size,
                        use_veg,
                        max_height,
                        2,  # patch_option
                        3.0,  # min_sun_elev_deg
                    )
                except Exception as e:
                    error_box[0] = e

            thread = threading.Thread(target=_run_svf, daemon=True)
            thread.start()

            # Poll progress (153 patches)
            pbar = ProgressReporter(
                total=n_patches,
                desc="Computing Sky View Factor",
                feedback=feedback,
                progress_range=progress_range,
            )
            last = 0
            while thread.is_alive():
                thread.join(timeout=0.05)
                done = runner.progress()
                if done > last:
                    pbar.update(done - last)
                    last = done
                # Check QGIS cancellation
                if feedback is not None and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                    runner.cancel()
                    thread.join(timeout=5.0)
                    pbar.close()
                    return
            if last < n_patches:
                pbar.update(n_patches - last)
            pbar.close()

            thread.join()
            if error_box[0] is not None:
                raise error_box[0]
            svf_result = result_box[0]

            ones = np.ones_like(dsm_arr, dtype=np.float32)

            svf_data = SvfArrays(
                svf=np.array(svf_result.svf),
                svf_north=np.array(svf_result.svf_north),
                svf_east=np.array(svf_result.svf_east),
                svf_south=np.array(svf_result.svf_south),
                svf_west=np.array(svf_result.svf_west),
                svf_veg=np.array(svf_result.svf_veg) if use_veg else ones.copy(),
                svf_veg_north=np.array(svf_result.svf_veg_north) if use_veg else ones.copy(),
                svf_veg_east=np.array(svf_result.svf_veg_east) if use_veg else ones.copy(),
                svf_veg_south=np.array(svf_result.svf_veg_south) if use_veg else ones.copy(),
                svf_veg_west=np.array(svf_result.svf_veg_west) if use_veg else ones.copy(),
                svf_aveg=np.array(svf_result.svf_veg_blocks_bldg_sh) if use_veg else ones.copy(),
                svf_aveg_north=np.array(svf_result.svf_veg_blocks_bldg_sh_north) if use_veg else ones.copy(),
                svf_aveg_east=np.array(svf_result.svf_veg_blocks_bldg_sh_east) if use_veg else ones.copy(),
                svf_aveg_south=np.array(svf_result.svf_veg_blocks_bldg_sh_south) if use_veg else ones.copy(),
                svf_aveg_west=np.array(svf_result.svf_veg_blocks_bldg_sh_west) if use_veg else ones.copy(),
            )

            # Cache SVF arrays
            memmap_dir = svf_cache_dir / "memmap"
            svf_data.to_memmap(memmap_dir, metadata=metadata)
            _save_svfs_zip(svf_data, svf_cache_dir, aligned_rasters)

            # Save shadow matrices (only available in non-tiled mode)
            _save_shadow_matrices(svf_result, svf_cache_dir)

            surface_data.svf = svf_data

            # Shadow matrices are bitpacked uint8 from Rust
            surface_data.shadow_matrices = ShadowArrays(
                _shmat_u8=np.array(svf_result.bldg_sh_matrix),
                _vegshmat_u8=np.array(svf_result.veg_sh_matrix),
                _vbshmat_u8=np.array(svf_result.veg_blocks_bldg_sh_matrix),
                _n_patches=n_patches,
            )

            logger.info(f"  ✓ SVF computed and cached to {svf_cache_dir}")

    @staticmethod
    def _compute_svf_tiled(
        dsm_f32: np.ndarray,
        cdsm_f32: np.ndarray,
        tdsm_f32: np.ndarray,
        pixel_size: float,
        use_veg: bool,
        max_height: float,
        working_path: Path,
        on_tile_complete: Callable | None = None,
        feedback: Any = None,
        progress_range: tuple[float, float] | None = None,
    ) -> tuple[SvfArrays, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute SVF using tiled processing for large grids.

        Automatically determines the largest safe tile size from the GPU
        buffer limit, divides the grid into overlapping tiles, computes
        SVF per tile, and stitches the core regions into full-size arrays.

        Shadow matrices are assembled into memory-mapped bitpacked uint8 files to
        avoid holding the full 3D arrays in RAM.

        Args:
            dsm_f32: DSM array (float32).
            cdsm_f32: Canopy DSM array (float32, zeros if no veg).
            tdsm_f32: Trunk DSM array (float32, zeros if no veg).
            pixel_size: Pixel size in meters.
            use_veg: Whether vegetation is present.
            max_height: Maximum height in the DSM (for buffer calculation).
            working_path: Directory for memmap files.
            on_tile_complete: Optional callback(tile_idx, n_tiles) called after each tile.

        Returns:
            Tuple of (SvfArrays, (shmat_mm, vegshmat_mm, vbshmat_mm))
            where the shadow matrix memmaps are bitpacked uint8 (rows, cols, n_pack).
        """
        from ..progress import ProgressReporter
        from ..tiling import calculate_buffer_distance, generate_tiles, validate_tile_size

        rows, cols = dsm_f32.shape

        buffer_m = calculate_buffer_distance(max_height)
        buffer_pixels = int(np.ceil(buffer_m / pixel_size))

        # Compute the largest safe tile size from GPU buffer limit.
        # The tile includes overlap buffers on each side, so the full tile
        # (core + 2*buffer) must fit in GPU memory.
        # wgpu max buffer = 256MB. SVF uses ~32 bytes/pixel for staging
        # buffers (shadow matrices, intermediate arrays). Use 80% headroom.
        _GPU_MAX_BUFFER = 268_435_456  # 256 MiB
        _BYTES_PER_PIXEL = 32  # empirical: staging buffers for SVF
        max_tile_pixels = int(_GPU_MAX_BUFFER * 0.8) // _BYTES_PER_PIXEL
        # Full tile side = core + 2*buffer, so core = sqrt(max_pixels) - 2*buffer
        max_full_side = int(max_tile_pixels**0.5)
        tile_size = max(256, max_full_side - 2 * buffer_pixels)

        adjusted_tile_size, warning = validate_tile_size(tile_size, buffer_pixels, pixel_size)
        if warning:
            logger.warning(warning)

        tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
        n_tiles = len(tiles)

        # Determine patch count from a small probe (patch_option=2 → 153 patches)
        n_patches = 153

        logger.info(
            f"  Tiled SVF: {rows}x{cols} raster, {n_tiles} tiles, "
            f"tile_size={adjusted_tile_size}, buffer={buffer_m:.0f}m ({buffer_pixels}px)"
        )

        # SVF field names on the Rust result object
        svf_fields = ["svf", "svf_north", "svf_east", "svf_south", "svf_west"]
        veg_fields = [
            "svf_veg",
            "svf_veg_north",
            "svf_veg_east",
            "svf_veg_south",
            "svf_veg_west",
            "svf_veg_blocks_bldg_sh",
            "svf_veg_blocks_bldg_sh_north",
            "svf_veg_blocks_bldg_sh_east",
            "svf_veg_blocks_bldg_sh_south",
            "svf_veg_blocks_bldg_sh_west",
        ]
        all_fields = svf_fields + veg_fields

        # Pre-allocate output arrays (ones = default SVF for unprocessed edges)
        outputs: dict[str, np.ndarray] = {}
        for name in all_fields:
            outputs[name] = np.ones((rows, cols), dtype=np.float32)

        # Pre-allocate memmap files for shadow matrices (bitpacked uint8, on disk)
        memmap_dir = working_path / "shadow_memmaps"
        memmap_dir.mkdir(parents=True, exist_ok=True)
        n_pack = (n_patches + 7) // 8  # ceil(153/8) = 20
        sh_shape = (rows, cols, n_pack)
        shmat_mm = np.memmap(
            memmap_dir / "shmat.dat",
            dtype=np.uint8,
            mode="w+",
            shape=sh_shape,
        )
        vegshmat_mm = np.memmap(
            memmap_dir / "vegshmat.dat",
            dtype=np.uint8,
            mode="w+",
            shape=sh_shape,
        )
        vbshmat_mm = np.memmap(
            memmap_dir / "vbshmat.dat",
            dtype=np.uint8,
            mode="w+",
            shape=sh_shape,
        )

        pbar = ProgressReporter(
            total=n_tiles,
            desc="Computing SVF (tiled)",
            feedback=feedback,
            progress_range=progress_range,
        )

        # Pipeline: overlap GPU computation of tile N+1 with CPU
        # result-copying of tile N.  calculate_svf releases the GIL
        # inside py.allow_threads(), so a background thread can drive
        # the GPU while the main thread does numpy bookkeeping.
        import threading

        def _submit_tile(tile):
            """Prepare inputs and run SVF on background thread."""
            rs = tile.read_slice
            td = dsm_f32[rs].copy()
            tc = cdsm_f32[rs].copy()
            tt = tdsm_f32[rs].copy()
            mh = float(np.nanmax(td))
            if use_veg:
                mh = max(mh, float(np.nanmax(tc)))
            box = [None, None]  # [result, error]

            def _run():
                try:
                    box[0] = skyview.calculate_svf(
                        td,
                        tc,
                        tt,
                        pixel_size,
                        use_veg,
                        mh,
                        2,
                        3.0,
                        None,
                    )
                except Exception as e:
                    box[1] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            return t, box

        def _process_result(tile_result, tile):
            """Copy SVF + shadow matrices from a completed tile."""
            cs = tile.core_slice
            ws = tile.write_slice
            for name in svf_fields:
                outputs[name][ws] = np.array(getattr(tile_result, name))[cs]
            if use_veg:
                for name in veg_fields:
                    outputs[name][ws] = np.array(getattr(tile_result, name))[cs]
            # Shadow matrices are already bitpacked uint8 from Rust
            for src_name, mm in [
                ("bldg_sh_matrix", shmat_mm),
                ("veg_sh_matrix", vegshmat_mm),
                ("veg_blocks_bldg_sh_matrix", vbshmat_mm),
            ]:
                arr = np.array(getattr(tile_result, src_name))
                mm[ws] = arr[cs]

        # Kick off first tile
        thread, box = _submit_tile(tiles[0])

        for tile_idx in range(n_tiles):
            pbar.set_description(f"SVF tile {tile_idx + 1}/{n_tiles}")
            pbar.set_text(f"Computing SVF — Tile {tile_idx + 1}/{n_tiles}")
            # Wait for current tile to finish
            thread.join()
            if box[1] is not None:
                raise box[1]
            cur_result = box[0]
            cur_tile = tiles[tile_idx]

            # Submit next tile (GPU starts while we copy results below)
            if tile_idx + 1 < n_tiles:
                thread, box = _submit_tile(tiles[tile_idx + 1])

            # Copy results on main thread (overlaps with next GPU computation)
            _process_result(cur_result, cur_tile)
            pbar.update(1)
            if on_tile_complete is not None:
                on_tile_complete(tile_idx, n_tiles)
            # Check QGIS cancellation between tiles
            if pbar.is_cancelled():
                pbar.close()
                logger.info("  SVF computation cancelled by user")
                break

        pbar.close()
        # Flush memmaps to disk
        shmat_mm.flush()
        vegshmat_mm.flush()
        vbshmat_mm.flush()

        ones = np.ones((rows, cols), dtype=np.float32)

        svf_data = SvfArrays(
            svf=outputs["svf"],
            svf_north=outputs["svf_north"],
            svf_east=outputs["svf_east"],
            svf_south=outputs["svf_south"],
            svf_west=outputs["svf_west"],
            svf_veg=outputs["svf_veg"] if use_veg else ones.copy(),
            svf_veg_north=outputs["svf_veg_north"] if use_veg else ones.copy(),
            svf_veg_east=outputs["svf_veg_east"] if use_veg else ones.copy(),
            svf_veg_south=outputs["svf_veg_south"] if use_veg else ones.copy(),
            svf_veg_west=outputs["svf_veg_west"] if use_veg else ones.copy(),
            svf_aveg=outputs["svf_veg_blocks_bldg_sh"] if use_veg else ones.copy(),
            svf_aveg_north=outputs["svf_veg_blocks_bldg_sh_north"] if use_veg else ones.copy(),
            svf_aveg_east=outputs["svf_veg_blocks_bldg_sh_east"] if use_veg else ones.copy(),
            svf_aveg_south=outputs["svf_veg_blocks_bldg_sh_south"] if use_veg else ones.copy(),
            svf_aveg_west=outputs["svf_veg_blocks_bldg_sh_west"] if use_veg else ones.copy(),
        )

        return svf_data, (shmat_mm, vegshmat_mm, vbshmat_mm)

    def preprocess(self) -> None:
        """
        Convert layers from relative to absolute heights based on per-layer flags.

        Converts each layer that is flagged as relative (``dsm_relative``,
        ``cdsm_relative``, ``tdsm_relative``) to absolute heights. Layers
        already flagged as absolute are left unchanged.

        This method:
        1. Converts DSM from relative to absolute if ``dsm_relative=True``
           (requires DEM: ``dsm_absolute = dem + dsm_relative``)
        2. Auto-generates TDSM from CDSM * trunk_ratio if TDSM is not provided
        3. Converts CDSM from relative to absolute if ``cdsm_relative=True``
        4. Converts TDSM from relative to absolute if ``tdsm_relative=True``
        5. Zeros out vegetation pixels with height < 0.1m

        Note:
            This method modifies arrays in-place and clears the per-layer
            relative flags once conversion is done.
        """
        if self._preprocessed:
            return

        # Fill NaN in surface layers before any height conversion
        self.fill_nan()

        threshold = np.float32(0.1)
        zero32 = np.float32(0.0)
        nan32 = np.float32(np.nan)

        # Step 1: Convert DSM from relative to absolute (requires DEM)
        if self.dsm_relative:
            if self.dem is None:
                raise ValueError(
                    "DSM is flagged as relative (dsm_relative=True) but no DEM "
                    "is provided. A DEM is required to convert relative DSM "
                    "(height above ground) to absolute elevations."
                )
            logger.info("Converting relative DSM to absolute: DSM = DEM + nDSM")
            self.dsm = (self.dem + self.dsm).astype(np.float32)
            self.dsm_relative = False

        # Step 2: Auto-generate TDSM from trunk ratio if CDSM provided but not TDSM
        if self.cdsm is not None and self.tdsm is None:
            logger.info(f"Auto-generating TDSM from CDSM using trunk_ratio={self.trunk_ratio}")
            self.tdsm = (self.cdsm * self.trunk_ratio).astype(np.float32)
            self.tdsm_relative = self.cdsm_relative

        # Use DEM as base if available, otherwise DSM (now absolute after step 1)
        base = self.dem if self.dem is not None else self.dsm

        # Step 3: Convert CDSM from relative to absolute
        if self.cdsm_relative and self.cdsm is not None:
            cdsm_rel = np.where(np.isnan(self.cdsm), zero32, self.cdsm)
            cdsm_abs = np.where(~np.isnan(base), base + cdsm_rel, nan32)
            cdsm_abs = np.where(cdsm_abs - base < threshold, zero32, cdsm_abs)
            self.cdsm = cdsm_abs.astype(np.float32)
            self.cdsm_relative = False
            logger.info(f"Converted relative CDSM to absolute (base: {'DEM' if self.dem is not None else 'DSM'})")

        # Step 4: Convert TDSM from relative to absolute
        if self.tdsm_relative and self.tdsm is not None:
            tdsm_rel = np.where(np.isnan(self.tdsm), zero32, self.tdsm)
            tdsm_abs = np.where(~np.isnan(base), base + tdsm_rel, nan32)
            tdsm_abs = np.where(tdsm_abs - base < threshold, zero32, tdsm_abs)
            self.tdsm = tdsm_abs.astype(np.float32)
            self.tdsm_relative = False
            logger.info(f"Converted relative TDSM to absolute (base: {'DEM' if self.dem is not None else 'DSM'})")

        self._preprocessed = True

    def compute_svf(self) -> None:
        """
        Compute Sky View Factor (SVF) and store in self.svf.

        This must be called before calculate() or calculate_timeseries()
        when constructing SurfaceData manually (not via prepare()).

        SVF is stored without psi (vegetation transmissivity) adjustment,
        since psi depends on day-of-year and conifer flag which are not
        known at SVF computation time. The adjustment is applied automatically
        during calculation.

        Also computes and stores shadow matrices in self.shadow_matrices
        (required for anisotropic sky model).

        Example:
            surface = SurfaceData(dsm=dsm, cdsm=cdsm)
            surface.preprocess()
            surface.compute_svf()
            result = calculate(surface, location, weather)
        """
        if self.svf is not None:
            return  # Already computed

        use_veg = self.cdsm is not None
        dsm_f32 = self.dsm.astype(np.float32)

        if use_veg:
            assert self.cdsm is not None  # Type narrowing for type checker
            cdsm_f32 = self.cdsm.astype(np.float32)
            if self.tdsm is not None:
                tdsm_f32 = self.tdsm.astype(np.float32)
            else:
                tdsm_f32 = (self.cdsm * self.trunk_ratio).astype(np.float32)
        else:
            cdsm_f32 = np.zeros_like(dsm_f32)
            tdsm_f32 = np.zeros_like(dsm_f32)

        max_height = float(np.nanmax(dsm_f32))
        if use_veg:
            veg_max = float(np.nanmax(cdsm_f32))
            max_height = max(max_height, veg_max)

        logger.info("Computing Sky View Factor...")
        svf_result = skyview.calculate_svf(
            dsm_f32,
            cdsm_f32,
            tdsm_f32,
            self.pixel_size,
            use_veg,
            max_height,
            2,  # patch_option (153 patches)
            3.0,  # min_sun_elev_deg
            None,  # progress callback
        )

        ones = np.ones_like(dsm_f32)
        self.svf = SvfArrays(
            svf=np.array(svf_result.svf),
            svf_north=np.array(svf_result.svf_north),
            svf_east=np.array(svf_result.svf_east),
            svf_south=np.array(svf_result.svf_south),
            svf_west=np.array(svf_result.svf_west),
            svf_veg=np.array(svf_result.svf_veg) if use_veg else ones.copy(),
            svf_veg_north=np.array(svf_result.svf_veg_north) if use_veg else ones.copy(),
            svf_veg_east=np.array(svf_result.svf_veg_east) if use_veg else ones.copy(),
            svf_veg_south=np.array(svf_result.svf_veg_south) if use_veg else ones.copy(),
            svf_veg_west=np.array(svf_result.svf_veg_west) if use_veg else ones.copy(),
            svf_aveg=np.array(svf_result.svf_veg_blocks_bldg_sh) if use_veg else ones.copy(),
            svf_aveg_north=np.array(svf_result.svf_veg_blocks_bldg_sh_north) if use_veg else ones.copy(),
            svf_aveg_east=np.array(svf_result.svf_veg_blocks_bldg_sh_east) if use_veg else ones.copy(),
            svf_aveg_south=np.array(svf_result.svf_veg_blocks_bldg_sh_south) if use_veg else ones.copy(),
            svf_aveg_west=np.array(svf_result.svf_veg_blocks_bldg_sh_west) if use_veg else ones.copy(),
        )

        # Store shadow matrices for anisotropic sky model
        # Shadow matrices are bitpacked uint8 from Rust
        self.shadow_matrices = ShadowArrays(
            _shmat_u8=np.array(svf_result.bldg_sh_matrix),
            _vegshmat_u8=np.array(svf_result.veg_sh_matrix),
            _vbshmat_u8=np.array(svf_result.veg_blocks_bldg_sh_matrix),
            _n_patches=153,  # patch_option=2
        )

        logger.info("  SVF computed successfully")

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

    @property
    def valid_mask(self) -> NDArray[np.bool_] | None:
        """Return computed valid mask, or None if not yet computed."""
        return self._valid_mask

    def fill_nan(self, tolerance: float = 0.1) -> None:
        """Fill NaN in surface layers using DEM as ground reference.

        NaN in DSM/CDSM/TDSM means "no data, assume ground level."
        After filling, values within *tolerance* of ground are clamped
        to exactly the ground value to avoid shadow/SVF noise from
        resampling jitter.

        Fill rules:
            - DSM NaN  → DEM value  (if DEM provided, else left as NaN)
            - CDSM NaN → base value (DEM if available, else DSM)
            - TDSM NaN → base value (DEM if available, else DSM)
            - DEM NaN  → not filled (DEM is the ground-truth baseline)

        Works identically for relative and absolute height conventions.

        Args:
            tolerance: Height difference (m) below which a surface pixel
                is considered "at ground" and clamped. Default 0.1 m.
        """
        if self._nan_filled:
            return

        tol = np.float32(tolerance)

        # DSM: fill with DEM where available
        if self.dem is not None:
            dsm_nan = np.isnan(self.dsm)
            if np.any(dsm_nan):
                n = int(dsm_nan.sum())
                self.dsm = np.where(dsm_nan, self.dem, self.dsm).astype(np.float32)
                logger.info(f"  Filled {n} NaN DSM pixels with DEM")

        base = self.dem if self.dem is not None else self.dsm
        base_label = "DEM" if self.dem is not None else "DSM"

        # CDSM: fill NaN with base, clamp near-ground noise
        if self.cdsm is not None:
            cdsm_nan = np.isnan(self.cdsm)
            if np.any(cdsm_nan):
                n = int(cdsm_nan.sum())
                self.cdsm = np.where(cdsm_nan, base, self.cdsm).astype(np.float32)
                logger.info(f"  Filled {n} NaN CDSM pixels with {base_label}")
            near_ground = np.abs(self.cdsm - base) < tol
            if np.any(near_ground):
                self.cdsm = np.where(near_ground, base, self.cdsm).astype(np.float32)

        # TDSM: same treatment as CDSM
        if self.tdsm is not None:
            tdsm_nan = np.isnan(self.tdsm)
            if np.any(tdsm_nan):
                n = int(tdsm_nan.sum())
                self.tdsm = np.where(tdsm_nan, base, self.tdsm).astype(np.float32)
                logger.info(f"  Filled {n} NaN TDSM pixels with {base_label}")
            near_ground = np.abs(self.tdsm - base) < tol
            if np.any(near_ground):
                self.tdsm = np.where(near_ground, base, self.tdsm).astype(np.float32)

        self._nan_filled = True

    def compute_valid_mask(self) -> NDArray[np.bool_]:
        """Compute combined valid mask: True where ALL ground-reference layers have finite data.

        A pixel is valid only if DSM (and DEM/walls if provided) have finite values.
        CDSM/TDSM are excluded — NaN vegetation means "at ground", not "invalid pixel".
        Call fill_nan() before this to fill vegetation NaN with ground values.

        Returns:
            Boolean array with same shape as DSM. True = valid pixel.
        """
        valid = np.isfinite(self.dsm)
        for arr in [self.dem, self.wall_height, self.wall_aspect]:
            if arr is not None:
                valid &= np.isfinite(arr)
        if self.land_cover is not None:
            valid &= self.land_cover != 255
        self._valid_mask = valid
        n_invalid = int(np.sum(~valid))
        if n_invalid > 0:
            pct = 100.0 * n_invalid / valid.size
            logger.info(f"  Valid mask: {n_invalid} invalid pixels ({pct:.1f}%)")
        else:
            logger.info("  Valid mask: all pixels valid")
        return valid

    def apply_valid_mask(self) -> None:
        """Set NaN in ALL layers where ANY layer has nodata.

        Ensures consistent nodata across all surface arrays.
        Must call compute_valid_mask() first (or it will be called automatically).
        """
        if self._valid_mask is None:
            self.compute_valid_mask()
        assert self._valid_mask is not None  # set by compute_valid_mask
        invalid = ~self._valid_mask
        if not np.any(invalid):
            return
        self.dsm[invalid] = np.nan
        for attr in ("cdsm", "dem", "tdsm", "wall_height", "wall_aspect", "albedo", "emissivity"):
            arr = getattr(self, attr)
            if arr is not None:
                arr[invalid] = np.nan
        if self.land_cover is not None:
            self.land_cover[invalid] = 255

    def crop_to_valid_bbox(self) -> tuple[int, int, int, int]:
        """Crop all arrays to minimum bounding box of valid pixels.

        Eliminates edge NaN bands to reduce wasted computation.
        Updates geotransform to reflect the new origin.

        Returns:
            (row_start, row_end, col_start, col_end) of the crop window.
        """
        if self._valid_mask is None:
            self.compute_valid_mask()
        assert self._valid_mask is not None  # set by compute_valid_mask
        rows_any = np.any(self._valid_mask, axis=1)
        cols_any = np.any(self._valid_mask, axis=0)
        if not np.any(rows_any):
            logger.warning("  No valid pixels found — cannot crop")
            return (0, self.dsm.shape[0], 0, self.dsm.shape[1])
        r0 = int(np.argmax(rows_any))
        r1 = len(rows_any) - int(np.argmax(rows_any[::-1]))
        c0 = int(np.argmax(cols_any))
        c1 = len(cols_any) - int(np.argmax(cols_any[::-1]))

        if r0 == 0 and r1 == self.dsm.shape[0] and c0 == 0 and c1 == self.dsm.shape[1]:
            logger.info("  Crop: no trimming needed (valid bbox = full extent)")
            return (r0, r1, c0, c1)

        old_shape = self.dsm.shape
        self.dsm = self.dsm[r0:r1, c0:c1].copy()
        self._valid_mask = self._valid_mask[r0:r1, c0:c1].copy()
        for attr in ("cdsm", "dem", "tdsm", "wall_height", "wall_aspect", "albedo", "emissivity", "land_cover"):
            arr = getattr(self, attr)
            if arr is not None:
                setattr(self, attr, arr[r0:r1, c0:c1].copy())

        # Update geotransform to reflect new origin
        if self._geotransform is not None:
            gt = self._geotransform
            self._geotransform = [
                gt[0] + c0 * gt[1] + r0 * gt[2],  # new origin X
                gt[1],
                gt[2],
                gt[3] + c0 * gt[4] + r0 * gt[5],  # new origin Y
                gt[4],
                gt[5],
            ]

        # Crop SVF arrays if present
        if self.svf is not None:
            self.svf = self.svf.crop(r0, r1, c0, c1)
        if self.shadow_matrices is not None:
            self.shadow_matrices = self.shadow_matrices.crop(r0, r1, c0, c1)

        # Clear buffer pool (shape changed)
        self.clear_buffers()

        logger.info(f"  Cropped: {old_shape[1]}x{old_shape[0]} → {c1 - c0}x{r1 - r0} pixels")
        return (r0, r1, c0, c1)

    def save_cleaned(self, output_dir: str | Path) -> None:
        """Save cleaned, aligned rasters to disk for inspection.

        Writes all present layers to output_dir/cleaned/ as GeoTIFFs.

        Args:
            output_dir: Parent directory. Files are saved under output_dir/cleaned/.
        """
        out = Path(output_dir) / "cleaned"
        out.mkdir(parents=True, exist_ok=True)
        gt = self._geotransform or [0, self.pixel_size, 0, 0, 0, -self.pixel_size]
        crs = self._crs_wkt or ""
        io.save_raster(str(out / "dsm.tif"), self.dsm, gt, crs)
        for name, arr in [
            ("cdsm", self.cdsm),
            ("dem", self.dem),
            ("tdsm", self.tdsm),
            ("wall_height", self.wall_height),
            ("wall_aspect", self.wall_aspect),
        ]:
            if arr is not None:
                io.save_raster(str(out / f"{name}.tif"), arr, gt, crs)
        if self.land_cover is not None:
            io.save_raster(str(out / "land_cover.tif"), self.land_cover.astype(np.float32), gt, crs)
        if self._valid_mask is not None:
            io.save_raster(str(out / "valid_mask.tif"), self._valid_mask.astype(np.float32), gt, crs)
        logger.info(f"  Cleaned rasters saved to {out}")

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

        if self.cdsm_relative and not self._preprocessed and self._looks_like_relative_heights():
            logger.warning(
                f"CDSM appears to contain relative vegetation heights "
                f"(max CDSM={np.nanmax(self.cdsm):.1f}m < min DSM={np.nanmin(self.dsm):.1f}m), "
                f"but preprocess() was not called. "
                f"Call surface.preprocess() to convert to absolute heights, "
                f"or set cdsm_relative=False if CDSM already contains absolute elevations."
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
