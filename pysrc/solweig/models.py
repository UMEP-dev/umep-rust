"""Data models for SOLWEIG calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .algorithms import sun_position as sp
from .algorithms.clearnessindex_2013b import clearnessindex_2013b
from .algorithms.diffusefraction import diffusefraction
from .logging import get_logger
from .utils import extract_bounds, intersect_bounds, resample_to_grid

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Stefan-Boltzmann constant
SBC = 5.67e-8

# Minimum sun elevation for shadow calculations (degrees)
MIN_SUN_ELEVATION_DEG = 3.0


@dataclass
class ThermalState:
    """
    Thermal state for multi-timestep calculations.

    Carries forward surface temperature history between timesteps
    to model thermal inertia of ground and walls (TsWaveDelay_2015a).

    This enables accurate time-series simulations where surface temperatures
    depend on accumulated heating throughout the day.

    Attributes:
        tgmap1: Upwelling longwave history (center view).
        tgmap1_e: Upwelling longwave history (east view).
        tgmap1_s: Upwelling longwave history (south view).
        tgmap1_w: Upwelling longwave history (west view).
        tgmap1_n: Upwelling longwave history (north view).
        tgout1: Ground temperature output history.
        firstdaytime: Flag for first morning timestep (1.0=first, 0.0=subsequent).
        timeadd: Accumulated time for thermal delay function.
        timestep_dec: Decimal time between steps (fraction of day).

    Example:
        # Manual state management for custom time loops
        state = ThermalState.initial(dsm.shape)
        for weather in weather_list:
            result = calculate(..., state=state)
            state = result.state
    """

    tgmap1: NDArray[np.floating]
    tgmap1_e: NDArray[np.floating]
    tgmap1_s: NDArray[np.floating]
    tgmap1_w: NDArray[np.floating]
    tgmap1_n: NDArray[np.floating]
    tgout1: NDArray[np.floating]
    firstdaytime: float = 1.0
    timeadd: float = 0.0
    timestep_dec: float = 0.0

    @classmethod
    def initial(cls, shape: tuple[int, int]) -> ThermalState:
        """
        Create initial state for first timestep.

        Args:
            shape: Grid shape (rows, cols) matching the DSM.

        Returns:
            ThermalState with zero-initialized arrays.
        """
        zeros = np.zeros(shape, dtype=np.float32)
        return cls(
            tgmap1=zeros.copy(),
            tgmap1_e=zeros.copy(),
            tgmap1_s=zeros.copy(),
            tgmap1_w=zeros.copy(),
            tgmap1_n=zeros.copy(),
            tgout1=zeros.copy(),
            firstdaytime=1.0,
            timeadd=0.0,
            timestep_dec=0.0,
        )

    def copy(self) -> ThermalState:
        """Create a deep copy of this state."""
        return ThermalState(
            tgmap1=self.tgmap1.copy(),
            tgmap1_e=self.tgmap1_e.copy(),
            tgmap1_s=self.tgmap1_s.copy(),
            tgmap1_w=self.tgmap1_w.copy(),
            tgmap1_n=self.tgmap1_n.copy(),
            tgout1=self.tgout1.copy(),
            firstdaytime=self.firstdaytime,
            timeadd=self.timeadd,
            timestep_dec=self.timestep_dec,
        )


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
        from . import io

        logger.info("Preparing surface data from GeoTIFF files...")

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

        # Load optional terrain rasters (store transforms for extent calculation)
        cdsm_arr, cdsm_transform = None, None
        if cdsm is not None:
            cdsm_arr, cdsm_transform, _, _ = io.load_raster(str(cdsm))
            logger.info("  ✓ Canopy DSM (CDSM) provided")
        else:
            logger.info("  → No vegetation data - simulation without trees/vegetation")

        dem_arr, dem_transform = None, None
        if dem is not None:
            dem_arr, dem_transform, _, _ = io.load_raster(str(dem))
            logger.info("  ✓ Ground elevation (DEM) provided")

        tdsm_arr, tdsm_transform = None, None
        if tdsm is not None:
            tdsm_arr, tdsm_transform, _, _ = io.load_raster(str(tdsm))
            logger.info("  ✓ Trunk DSM (TDSM) provided")
        elif cdsm_arr is not None:
            logger.info(f"  → No TDSM provided - will auto-generate from CDSM (ratio={trunk_ratio})")

        land_cover_arr, land_cover_transform = None, None
        if land_cover is not None:
            land_cover_arr, land_cover_transform, _, _ = io.load_raster(str(land_cover))
            logger.info("  ✓ Land cover provided (albedo/emissivity derived from classification)")

        # Load preprocessing data (walls, SVF)
        logger.info("Checking for preprocessing data...")

        # Convert working_dir to Path (always provided now)
        working_path = Path(working_dir)

        # Walls loading with auto-discovery in working_dir
        wall_height_arr, wall_height_transform = None, None
        wall_aspect_arr, wall_aspect_transform = None, None
        compute_walls_to_cache = False  # Flag: compute and save to working_dir after resampling

        if wall_height is not None and wall_aspect is not None:
            # Explicit paths provided - use them
            wall_height_arr, wall_height_transform, _, _ = io.load_raster(str(wall_height))
            wall_aspect_arr, wall_aspect_transform, _, _ = io.load_raster(str(wall_aspect))
            logger.info("  ✓ Existing walls found (will use precomputed)")

        elif wall_height is not None or wall_aspect is not None:
            logger.warning("  ⚠ Only one wall file provided - both wall_height and wall_aspect required")
            logger.info("  → Walls will be computed from DSM and cached")
            compute_walls_to_cache = True

        else:
            # Try to auto-discover walls in working_dir (unless force_recompute)
            if force_recompute:
                logger.info("  → force_recompute=True - will recompute walls from DSM and cache")
                compute_walls_to_cache = True
            else:
                walls_cache_dir = working_path / "walls"
                wall_hts_path = walls_cache_dir / "wall_hts.tif"
                wall_aspects_path = walls_cache_dir / "wall_aspects.tif"

                if wall_hts_path.exists() and wall_aspects_path.exists():
                    # Files exist - load and validate compatibility
                    wall_height_arr, wall_height_transform, _, _ = io.load_raster(str(wall_hts_path))
                    wall_aspect_arr, wall_aspect_transform, _, _ = io.load_raster(str(wall_aspects_path))
                    logger.info(f"  ✓ Walls found in working_dir: {walls_cache_dir}")
                    # Note: Compatibility with DSM checked during resampling below
                else:
                    # No cached walls - will compute and cache
                    logger.info("  → No walls found in working_dir - will compute from DSM and cache")
                    compute_walls_to_cache = True

        # SVF loading with auto-discovery in working_dir
        svf_data = None
        shadow_data = None
        compute_svf_to_cache = False  # Flag: compute and save to working_dir after resampling

        if svf_dir is not None:
            # Explicit SVF directory provided - use it
            svf_path = Path(svf_dir)
            svf_zip_path = svf_path / "svfs.zip"
            shadow_npz_path = svf_path / "shadowmats.npz"

            if svf_zip_path.exists():
                svf_data = SvfArrays.from_zip(str(svf_zip_path))
                logger.info("  ✓ Existing SVF found (will use precomputed)")

                if shadow_npz_path.exists():
                    shadow_data = ShadowArrays.from_npz(str(shadow_npz_path))
                    logger.info("  ✓ Existing shadow matrices found (anisotropic sky enabled)")
            else:
                logger.info(f"  → SVF directory provided but svfs.zip not found: {svf_zip_path}")
                logger.info("  → SVF will be computed and cached")
                compute_svf_to_cache = True

        else:
            # Try to auto-discover SVF in working_dir (unless force_recompute)
            if force_recompute:
                logger.info("  → force_recompute=True - will recompute SVF and cache")
                compute_svf_to_cache = True
            else:
                svf_cache_dir = working_path / "svf"
                svf_zip_path = svf_cache_dir / "svfs.zip"
                shadow_npz_path = svf_cache_dir / "shadowmats.npz"

                if svf_zip_path.exists():
                    # Files exist - load them
                    svf_data = SvfArrays.from_zip(str(svf_zip_path))
                    logger.info(f"  ✓ SVF found in working_dir: {svf_cache_dir}")

                    if shadow_npz_path.exists():
                        shadow_data = ShadowArrays.from_npz(str(shadow_npz_path))
                        logger.info("  ✓ Shadow matrices found (anisotropic sky enabled)")
                    # Note: Compatibility checked during calculation
                else:
                    # No cached SVF - will compute and cache
                    logger.info("  → No SVF found in working_dir - will compute and cache")
                    compute_svf_to_cache = True

        # Compute extent and resample if needed
        logger.info("Computing spatial extent and resolution...")

        # Extract bounds from all loaded rasters
        bounds_list = [extract_bounds(dsm_transform, dsm_arr.shape)]
        if cdsm_arr is not None and cdsm_transform is not None:
            bounds_list.append(extract_bounds(cdsm_transform, cdsm_arr.shape))
        if dem_arr is not None and dem_transform is not None:
            bounds_list.append(extract_bounds(dem_transform, dem_arr.shape))
        if tdsm_arr is not None and tdsm_transform is not None:
            bounds_list.append(extract_bounds(tdsm_transform, tdsm_arr.shape))
        if land_cover_arr is not None and land_cover_transform is not None:
            bounds_list.append(extract_bounds(land_cover_transform, land_cover_arr.shape))
        if wall_height_arr is not None and wall_height_transform is not None:
            bounds_list.append(extract_bounds(wall_height_transform, wall_height_arr.shape))
        if wall_aspect_arr is not None and wall_aspect_transform is not None:
            bounds_list.append(extract_bounds(wall_aspect_transform, wall_aspect_arr.shape))

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
            if cdsm_arr is not None and cdsm_transform is not None:
                cdsm_arr, _ = resample_to_grid(
                    cdsm_arr, cdsm_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
                )
            if dem_arr is not None and dem_transform is not None:
                dem_arr, _ = resample_to_grid(
                    dem_arr, dem_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
                )
            if tdsm_arr is not None and tdsm_transform is not None:
                tdsm_arr, _ = resample_to_grid(
                    tdsm_arr, tdsm_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
                )
            if land_cover_arr is not None and land_cover_transform is not None:
                # Use nearest neighbor for categorical data
                land_cover_arr, _ = resample_to_grid(
                    land_cover_arr, land_cover_transform, target_bbox, pixel_size, method="nearest", src_crs=dsm_crs
                )

            # Resample preprocessing data
            if wall_height_arr is not None and wall_height_transform is not None:
                wall_height_arr, _ = resample_to_grid(
                    wall_height_arr, wall_height_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
                )
            if wall_aspect_arr is not None and wall_aspect_transform is not None:
                wall_aspect_arr, _ = resample_to_grid(
                    wall_aspect_arr, wall_aspect_transform, target_bbox, pixel_size, method="bilinear", src_crs=dsm_crs
                )

            # Note: SVF resampling is more complex (multiple arrays) - handled separately if needed
            if svf_data is not None and svf_data.svf.shape != dsm_arr.shape:
                logger.warning(
                    f"  ⚠ SVF shape {svf_data.svf.shape} doesn't match target shape "
                    f"{dsm_arr.shape} - SVF resampling not yet implemented. "
                    f"SVF will be recomputed on-the-fly if needed."
                )
                svf_data = None
                shadow_data = None

            logger.info(f"  ✓ Resampled to {dsm_arr.shape[1]}×{dsm_arr.shape[0]} pixels")
        else:
            logger.info("  ✓ No resampling needed - all rasters match target grid")

        # Create SurfaceData instance
        surface_data = cls(
            dsm=dsm_arr,
            cdsm=cdsm_arr,
            dem=dem_arr,
            tdsm=tdsm_arr,
            land_cover=land_cover_arr,
            wall_height=wall_height_arr,
            wall_aspect=wall_aspect_arr,
            svf=svf_data,
            shadow_matrices=shadow_data,
            pixel_size=pixel_size,
            trunk_ratio=trunk_ratio,
            relative_heights=relative_heights,
        )

        # Store geotransform and CRS for later export
        # Convert Affine to GDAL list if needed
        from affine import Affine as AffineClass

        if isinstance(dsm_transform, AffineClass):
            surface_data._geotransform = list(dsm_transform.to_gdal())
        else:
            surface_data._geotransform = dsm_transform
        surface_data._crs_wkt = dsm_crs

        # Log what was loaded
        layers_loaded = ["DSM"]
        if cdsm_arr is not None:
            layers_loaded.append("CDSM")
        if dem_arr is not None:
            layers_loaded.append("DEM")
        if tdsm_arr is not None:
            layers_loaded.append("TDSM")
        if land_cover_arr is not None:
            layers_loaded.append("land_cover")
        logger.info(f"  Layers loaded: {', '.join(layers_loaded)}")

        # Compute and cache walls if needed
        if compute_walls_to_cache:
            logger.info("Computing walls from DSM and caching to working_dir...")
            walls_cache_dir = working_path / "walls"

            # First, save resampled DSM to working_dir so wall computation can use it
            resampled_dir = working_path / "resampled"
            resampled_dir.mkdir(parents=True, exist_ok=True)
            resampled_dsm_path = resampled_dir / "dsm_resampled.tif"

            io.save_raster(
                str(resampled_dsm_path),
                dsm_arr,
                list(dsm_transform.to_gdal()) if isinstance(dsm_transform, AffineClass) else dsm_transform,
                dsm_crs,
            )

            # Generate walls using the walls module
            from . import walls as walls_module

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

        # Compute and cache SVF if needed
        if compute_svf_to_cache:
            if tdsm_arr is not None:
                logger.info("Computing SVF from DSM/CDSM/TDSM and caching to working_dir...")
            else:
                logger.info("Computing SVF from DSM/CDSM and caching to working_dir...")
            svf_cache_dir = working_path / "svf"

            # Save resampled DSM if not already saved
            resampled_dir = working_path / "resampled"
            resampled_dir.mkdir(parents=True, exist_ok=True)
            resampled_dsm_path = resampled_dir / "dsm_resampled.tif"

            if not resampled_dsm_path.exists():
                io.save_raster(
                    str(resampled_dsm_path),
                    dsm_arr,
                    list(dsm_transform.to_gdal()) if isinstance(dsm_transform, AffineClass) else dsm_transform,
                    dsm_crs,
                )

            # Save resampled CDSM if present
            resampled_cdsm_path = None
            if cdsm_arr is not None:
                resampled_cdsm_path = resampled_dir / "cdsm_resampled.tif"
                io.save_raster(
                    str(resampled_cdsm_path),
                    cdsm_arr,
                    list(dsm_transform.to_gdal()) if isinstance(dsm_transform, AffineClass) else dsm_transform,
                    dsm_crs,
                )

            # Save resampled TDSM if present (user-provided)
            resampled_tdsm_path = None
            if tdsm_arr is not None:
                resampled_tdsm_path = resampled_dir / "tdsm_resampled.tif"
                io.save_raster(
                    str(resampled_tdsm_path),
                    tdsm_arr,
                    list(dsm_transform.to_gdal()) if isinstance(dsm_transform, AffineClass) else dsm_transform,
                    dsm_crs,
                )

            # Generate SVF using the svf module
            # Compute bbox from resampled extent
            minx, miny, maxx, maxy = extract_bounds(dsm_transform, dsm_arr.shape)
            resampled_bbox = [int(minx), int(miny), int(maxx), int(maxy)]

            from . import svf as svf_module

            svf_module.generate_svf(
                dsm_path=str(resampled_dsm_path),
                bbox=resampled_bbox,
                out_dir=str(svf_cache_dir),
                cdsm_path=str(resampled_cdsm_path) if resampled_cdsm_path else None,
                tdsm_path=str(resampled_tdsm_path) if resampled_tdsm_path else None,
                trans_veg_perc=3.0,  # Default parameter
                trunk_ratio_perc=trunk_ratio * 100,  # Match prepare() trunk_ratio
            )

            # Load the generated SVF back into surface_data
            svf_zip_path = svf_cache_dir / "svfs.zip"
            shadow_npz_path = svf_cache_dir / "shadowmats.npz"

            if svf_zip_path.exists():
                svf_data = SvfArrays.from_zip(str(svf_zip_path))
                surface_data.svf = svf_data
                logger.info(f"  ✓ SVF computed and cached to {svf_cache_dir}")

                if shadow_npz_path.exists():
                    shadow_data = ShadowArrays.from_npz(str(shadow_npz_path))
                    surface_data.shadow_matrices = shadow_data
                    logger.info("  ✓ Shadow matrices also generated (anisotropic sky enabled)")
            else:
                logger.warning("  ⚠ SVF generation completed but files not found")

        # Preprocess CDSM/TDSM if relative heights
        if relative_heights and surface_data.cdsm is not None:
            logger.debug("  Preprocessing CDSM/TDSM (relative → absolute heights)")
            surface_data.preprocess()

        logger.info("✓ Surface data prepared successfully")
        return surface_data

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
        """Auto-compute maximum height difference in DSM."""
        return float(np.nanmax(self.dsm) - np.nanmin(self.dsm))

    @property
    def shape(self) -> tuple[int, int]:
        """Return DSM shape (rows, cols)."""
        return self.dsm.shape

    @property
    def crs(self) -> str | None:
        """Return CRS as WKT string, or None if not set."""
        return self._crs_wkt

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
        if cdsm_max < 60 and dsm_min > cdsm_max + 20:
            return True

        return False

    def _check_preprocessing_needed(self) -> None:
        """
        Warn if CDSM appears to need preprocessing but wasn't preprocessed.

        Called internally before calculations to alert users.
        """
        if self.cdsm is None:
            return

        if self.relative_heights and not self._preprocessed:
            if self._looks_like_relative_heights():
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
            return _get_lc_properties_from_params(self.land_cover, params, self.shape)

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


@dataclass
class SvfArrays:
    """
    Pre-computed Sky View Factor arrays.

    Use this when you have already computed SVF and want to skip
    re-computation. Can be loaded from SOLWEIG svfs.zip format.

    Attributes:
        svf: Total sky view factor (0-1).
        svf_north, svf_east, svf_south, svf_west: Directional SVF components.
        svf_veg: Vegetation SVF (set to ones if no vegetation).
        svf_veg_north, svf_veg_east, svf_veg_south, svf_veg_west: Directional veg SVF.
        svf_aveg: Vegetation blocking building shadow.
        svf_aveg_north, svf_aveg_east, svf_aveg_south, svf_aveg_west: Directional.

    Memory note:
        All arrays are stored as float32. For a 768x768 grid with all 15 arrays,
        total memory is approximately 35 MB.
    """

    svf: NDArray[np.floating]
    svf_north: NDArray[np.floating]
    svf_east: NDArray[np.floating]
    svf_south: NDArray[np.floating]
    svf_west: NDArray[np.floating]
    svf_veg: NDArray[np.floating]
    svf_veg_north: NDArray[np.floating]
    svf_veg_east: NDArray[np.floating]
    svf_veg_south: NDArray[np.floating]
    svf_veg_west: NDArray[np.floating]
    svf_aveg: NDArray[np.floating]
    svf_aveg_north: NDArray[np.floating]
    svf_aveg_east: NDArray[np.floating]
    svf_aveg_south: NDArray[np.floating]
    svf_aveg_west: NDArray[np.floating]

    def __post_init__(self):
        # Ensure all arrays are float32 for memory efficiency
        self.svf = np.asarray(self.svf, dtype=np.float32)
        self.svf_north = np.asarray(self.svf_north, dtype=np.float32)
        self.svf_east = np.asarray(self.svf_east, dtype=np.float32)
        self.svf_south = np.asarray(self.svf_south, dtype=np.float32)
        self.svf_west = np.asarray(self.svf_west, dtype=np.float32)
        self.svf_veg = np.asarray(self.svf_veg, dtype=np.float32)
        self.svf_veg_north = np.asarray(self.svf_veg_north, dtype=np.float32)
        self.svf_veg_east = np.asarray(self.svf_veg_east, dtype=np.float32)
        self.svf_veg_south = np.asarray(self.svf_veg_south, dtype=np.float32)
        self.svf_veg_west = np.asarray(self.svf_veg_west, dtype=np.float32)
        self.svf_aveg = np.asarray(self.svf_aveg, dtype=np.float32)
        self.svf_aveg_north = np.asarray(self.svf_aveg_north, dtype=np.float32)
        self.svf_aveg_east = np.asarray(self.svf_aveg_east, dtype=np.float32)
        self.svf_aveg_south = np.asarray(self.svf_aveg_south, dtype=np.float32)
        self.svf_aveg_west = np.asarray(self.svf_aveg_west, dtype=np.float32)

    @property
    def svfalfa(self) -> NDArray[np.floating]:
        """Compute SVF alpha (angle) from SVF values. Computed on-demand."""
        tmp = self.svf + self.svf_veg - 1.0
        tmp = np.clip(tmp, 0.0, 1.0)
        eps = np.finfo(np.float32).tiny
        safe_term = np.clip(1.0 - tmp, eps, 1.0)
        return np.arcsin(np.exp(np.log(safe_term) / 2.0))

    @property
    def svfbuveg(self) -> NDArray[np.floating]:
        """Combined building + vegetation SVF. Computed on-demand."""
        return np.clip(self.svf + self.svf_veg - 1.0, 0.0, 1.0)

    @classmethod
    def from_zip(cls, zip_path: str | Path, use_vegetation: bool = True) -> SvfArrays:
        """
        Load SVF arrays from SOLWEIG svfs.zip format.

        Args:
            zip_path: Path to svfs.zip file.
            use_vegetation: Whether to load vegetation SVF arrays. Default True.

        Returns:
            SvfArrays instance with loaded data.

        Memory note:
            Files are extracted temporarily and loaded as float32 arrays.
            The zip file contains GeoTIFF rasters.
        """
        import tempfile
        import zipfile

        from . import io as common

        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"SVF zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(tmpdir)

            tmppath = Path(tmpdir)

            def load(filename: str) -> NDArray[np.floating]:
                filepath = tmppath / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Expected SVF file not found in zip: {filename}")
                data, _, _, _ = common.load_raster(str(filepath), coerce_f64_to_f32=True)
                return data

            # Load basic SVF arrays
            svf = load("svf.tif")
            svf_n = load("svfN.tif")
            svf_e = load("svfE.tif")
            svf_s = load("svfS.tif")
            svf_w = load("svfW.tif")

            # Load vegetation arrays or create defaults
            if use_vegetation:
                svf_veg = load("svfveg.tif")
                svf_veg_n = load("svfNveg.tif")
                svf_veg_e = load("svfEveg.tif")
                svf_veg_s = load("svfSveg.tif")
                svf_veg_w = load("svfWveg.tif")
                svf_aveg = load("svfaveg.tif")
                svf_aveg_n = load("svfNaveg.tif")
                svf_aveg_e = load("svfEaveg.tif")
                svf_aveg_s = load("svfSaveg.tif")
                svf_aveg_w = load("svfWaveg.tif")
            else:
                ones = np.ones_like(svf)
                svf_veg = ones
                svf_veg_n = ones
                svf_veg_e = ones
                svf_veg_s = ones
                svf_veg_w = ones
                svf_aveg = ones
                svf_aveg_n = ones
                svf_aveg_e = ones
                svf_aveg_s = ones
                svf_aveg_w = ones

        return cls(
            svf=svf,
            svf_north=svf_n,
            svf_east=svf_e,
            svf_south=svf_s,
            svf_west=svf_w,
            svf_veg=svf_veg,
            svf_veg_north=svf_veg_n,
            svf_veg_east=svf_veg_e,
            svf_veg_south=svf_veg_s,
            svf_veg_west=svf_veg_w,
            svf_aveg=svf_aveg,
            svf_aveg_north=svf_aveg_n,
            svf_aveg_east=svf_aveg_e,
            svf_aveg_south=svf_aveg_s,
            svf_aveg_west=svf_aveg_w,
        )


@dataclass
class ShadowArrays:
    """
    Pre-computed anisotropic shadow matrices for sky patch calculations.

    These are 3D arrays of shape (rows, cols, patches) where patches is
    typically 145, 153, 306, or 612 depending on the resolution.

    Memory optimization:
        Internally stored as uint8 (0-255 representing 0.0-1.0) to reduce
        memory by 75%. Converted to float32 only when accessed via properties.
        Cached after first access to avoid repeated allocations.
        For a 400x400 grid with 153 patches: 24.5 MB as uint8 vs 98 MB as float32.

    Attributes:
        _shmat_u8: Building shadow matrix (uint8 storage).
        _vegshmat_u8: Vegetation shadow matrix (uint8 storage).
        _vbshmat_u8: Combined veg+building shadow matrix (uint8 storage).
        patch_count: Number of sky patches (145, 153, 306, or 612).
    """

    _shmat_u8: NDArray[np.uint8]
    _vegshmat_u8: NDArray[np.uint8]
    _vbshmat_u8: NDArray[np.uint8]
    patch_count: int = field(init=False)
    # Cache for converted float32 arrays (allocated on first access)
    _shmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vegshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vbshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Ensure uint8 storage
        if self._shmat_u8.dtype != np.uint8:
            self._shmat_u8 = (np.clip(self._shmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vegshmat_u8.dtype != np.uint8:
            self._vegshmat_u8 = (np.clip(self._vegshmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vbshmat_u8.dtype != np.uint8:
            self._vbshmat_u8 = (np.clip(self._vbshmat_u8, 0, 1) * 255).astype(np.uint8)

        self.patch_count = self._shmat_u8.shape[2]
        # Initialize cache as None (lazy allocation)
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None

    @property
    def shmat(self) -> NDArray[np.floating]:
        """
        Building shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._shmat_f32 is None:
            self._shmat_f32 = self._shmat_u8.astype(np.float32) / 255.0
        return self._shmat_f32

    @property
    def vegshmat(self) -> NDArray[np.floating]:
        """
        Vegetation shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._vegshmat_f32 is None:
            self._vegshmat_f32 = self._vegshmat_u8.astype(np.float32) / 255.0
        return self._vegshmat_f32

    @property
    def vbshmat(self) -> NDArray[np.floating]:
        """
        Combined shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._vbshmat_f32 is None:
            self._vbshmat_f32 = self._vbshmat_u8.astype(np.float32) / 255.0
        return self._vbshmat_f32

    @property
    def patch_option(self) -> int:
        """Patch option code (1=145, 2=153, 3=306, 4=612 patches)."""
        patch_map = {145: 1, 153: 2, 306: 3, 612: 4}
        return patch_map.get(self.patch_count, 2)

    def diffsh(self, transmissivity: float = 0.03, use_vegetation: bool = True) -> NDArray[np.floating]:
        """
        Compute diffuse shadow matrix.

        Args:
            transmissivity: Vegetation transmissivity (default 0.03).
            use_vegetation: Whether to account for vegetation.

        Returns:
            Diffuse shadow matrix as float32.
        """
        shmat = self.shmat
        if use_vegetation:
            vegshmat = self.vegshmat
            return (shmat - (1 - vegshmat) * (1 - transmissivity)).astype(np.float32)
        return shmat

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> ShadowArrays:
        """
        Load shadow matrices from SOLWEIG shadowmats.npz format.

        Args:
            npz_path: Path to shadowmats.npz file.

        Returns:
            ShadowArrays instance with loaded data.

        Memory note:
            The npz file typically contains uint8 arrays (compressed).
            Data is kept as uint8 internally; conversion to float32
            happens only when arrays are accessed via properties.
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Shadow matrices file not found: {npz_path}")

        data = np.load(str(npz_path))

        # Load arrays - they should be uint8 in the optimized format
        shmat = data["shadowmat"]
        vegshmat = data["vegshadowmat"]
        vbshmat = data["vbshmat"]

        # Convert to uint8 if not already (legacy float32 format)
        if shmat.dtype != np.uint8:
            shmat = (np.clip(shmat, 0, 1) * 255).astype(np.uint8)
        if vegshmat.dtype != np.uint8:
            vegshmat = (np.clip(vegshmat, 0, 1) * 255).astype(np.uint8)
        if vbshmat.dtype != np.uint8:
            vbshmat = (np.clip(vbshmat, 0, 1) * 255).astype(np.uint8)

        return cls(
            _shmat_u8=shmat,
            _vegshmat_u8=vegshmat,
            _vbshmat_u8=vbshmat,
        )


@dataclass
class PrecomputedData:
    """
    Container for pre-computed preprocessing data to skip expensive calculations.

    Use this to provide already-computed walls, SVF, and/or shadow matrices
    to the calculate() function. This is useful when:
    - Running multiple timesteps with the same geometry
    - Using data generated by external tools
    - Optimizing performance by pre-computing once

    Attributes:
        wall_height: Pre-computed wall height grid (meters). If None, computed on-the-fly.
        wall_aspect: Pre-computed wall aspect grid (degrees, 0=N). If None, computed on-the-fly.
        svf: Pre-computed SVF arrays. If None, SVF is computed on-the-fly.
        shadow_matrices: Pre-computed anisotropic shadow matrices.
            If None, isotropic sky model is used.

    Example:
        # Load all preprocessing
        precomputed = PrecomputedData.load(
            walls_dir="preprocessed/walls",
            svf_dir="preprocessed/svf",
        )

        # Or create manually
        svf = SvfArrays.from_zip("path/to/svfs.zip")
        shadows = ShadowArrays.from_npz("path/to/shadowmats.npz")
        precomputed = PrecomputedData(svf=svf, shadow_matrices=shadows)

        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            precomputed=precomputed,
        )
    """

    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None

    @classmethod
    def prepare(
        cls,
        walls_dir: str | Path | None = None,
        svf_dir: str | Path | None = None,
    ) -> PrecomputedData:
        """
        Prepare preprocessing data from directories.

        Loads preprocessing files if they exist. If files don't exist,
        the corresponding data will be None and computed on-the-fly during calculation.

        All parameters are optional.

        Args:
            walls_dir: Directory containing wall preprocessing files:
                - wall_hts.tif: Wall heights (meters)
                - wall_aspects.tif: Wall aspects (degrees, 0=N)
            svf_dir: Directory containing SVF preprocessing files:
                - svfs.zip: SVF arrays (required if svf_dir provided)
                - shadowmats.npz: Shadow matrices for anisotropic sky (optional)

        Returns:
            PrecomputedData with loaded arrays. Missing data is set to None.

        Example:
            # Prepare all preprocessing
            precomputed = PrecomputedData.prepare(
                walls_dir="preprocessed/walls",
                svf_dir="preprocessed/svf",
            )

            # Prepare only SVF
            precomputed = PrecomputedData.prepare(svf_dir="preprocessed/svf")

            # Nothing prepared (all computed on-the-fly)
            precomputed = PrecomputedData.prepare()
        """
        from . import io

        wall_height_arr = None
        wall_aspect_arr = None
        svf_arrays = None
        shadow_arrays = None

        # Load walls if directory provided
        if walls_dir is not None:
            walls_path = Path(walls_dir)
            wall_height_path = walls_path / "wall_hts.tif"
            wall_aspect_path = walls_path / "wall_aspects.tif"

            if wall_height_path.exists():
                wall_height_arr, _, _, _ = io.load_raster(str(wall_height_path))
                logger.info(f"  Loaded wall heights from {walls_dir}")
            else:
                logger.debug(f"  Wall heights not found: {wall_height_path}")

            if wall_aspect_path.exists():
                wall_aspect_arr, _, _, _ = io.load_raster(str(wall_aspect_path))
                logger.info(f"  Loaded wall aspects from {walls_dir}")
            else:
                logger.debug(f"  Wall aspects not found: {wall_aspect_path}")

        # Load SVF if directory provided
        if svf_dir is not None:
            svf_path = Path(svf_dir)
            svf_zip = svf_path / "svfs.zip"

            if svf_zip.exists():
                svf_arrays = SvfArrays.from_zip(str(svf_zip))
                logger.info(f"  Loaded SVF data: {svf_arrays.svf.shape}")
            else:
                logger.debug(f"  SVF not found: {svf_zip}")

            # Load shadow matrices (optional - for anisotropic sky)
            shadow_npz = svf_path / "shadowmats.npz"
            if shadow_npz.exists():
                shadow_arrays = ShadowArrays.from_npz(str(shadow_npz))
                logger.info("  Loaded shadow matrices for anisotropic sky")
            else:
                logger.debug("  No shadow matrices found (anisotropic sky will be slower)")

        return cls(
            wall_height=wall_height_arr,
            wall_aspect=wall_aspect_arr,
            svf=svf_arrays,
            shadow_matrices=shadow_arrays,
        )


@dataclass
class Location:
    """
    Geographic location for sun position calculations.

    Attributes:
        latitude: Latitude in degrees (north positive).
        longitude: Longitude in degrees (east positive).
        altitude: Altitude above sea level in meters. Default 0.
        utc_offset: UTC offset in hours. Default 0.
    """

    latitude: float
    longitude: float
    altitude: float = 0.0
    utc_offset: int = 0

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")

    @classmethod
    def from_dsm_crs(cls, dsm_path: str | Path, utc_offset: int = 0, altitude: float = 0.0) -> Location:
        """
        Extract location from DSM raster's CRS by converting center point to WGS84.

        Args:
            dsm_path: Path to DSM GeoTIFF file with valid CRS.
            utc_offset: UTC offset in hours. Must be provided by user.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If DSM has no CRS or CRS conversion fails.

        Example:
            location = Location.from_dsm_crs("dsm.tif", utc_offset=2)
        """
        from . import io

        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError("pyproj is required for CRS extraction. Install with: pip install pyproj")

        # Load DSM to get CRS and bounds
        _, transform, crs_wkt, _ = io.load_raster(str(dsm_path))

        if not crs_wkt:
            raise ValueError(
                f"DSM has no CRS metadata: {dsm_path}\n"
                f"Either:\n"
                f"  1. Add CRS to GeoTIFF: gdal_edit.py -a_srs EPSG:XXXXX {dsm_path}\n"
                f"  2. Provide location manually: Location(latitude=X, longitude=Y, utc_offset={utc_offset})"
            )

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        # We need the raster dimensions to find center - load again to get shape
        dsm_array, _, _, _ = io.load_raster(str(dsm_path))
        rows, cols = dsm_array.shape

        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)

        logger.info(f"Extracted location from DSM CRS: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+d})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    @classmethod
    def from_surface(cls, surface: SurfaceData, utc_offset: int = 0, altitude: float = 0.0) -> Location:
        """
        Extract location from SurfaceData's CRS by converting center point to WGS84.

        This avoids reloading the DSM raster when you already have loaded SurfaceData.

        Args:
            surface: SurfaceData instance loaded from GeoTIFF.
            utc_offset: UTC offset in hours. Default 0.
            altitude: Altitude above sea level in meters. Default 0.

        Returns:
            Location object with lat/lon from DSM center point.

        Raises:
            ValueError: If surface has no CRS metadata.
            ImportError: If pyproj is not installed.

        Example:
            surface = SurfaceData.from_geotiff("dsm.tif")
            location = Location.from_surface(surface, utc_offset=2)
        """
        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError("pyproj is required for CRS extraction. Install with: pip install pyproj")

        # Check if geotransform and CRS are available
        if not hasattr(surface, "_geotransform") or surface._geotransform is None:
            raise ValueError(
                "Surface data has no geotransform metadata.\n"
                "Load surface with SurfaceData.from_geotiff() or provide location manually."
            )
        if not hasattr(surface, "_crs_wkt") or surface._crs_wkt is None:
            raise ValueError(
                "Surface data has no CRS metadata.\n"
                "Provide location manually: Location(latitude=X, longitude=Y, utc_offset=0)"
            )

        transform = surface._geotransform
        crs_wkt = surface._crs_wkt
        rows, cols = surface.dsm.shape

        # Get center point from geotransform
        # Transform is [x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size]
        center_x = transform[0] + (cols / 2) * transform[1]
        center_y = transform[3] + (rows / 2) * transform[5]

        # Convert to WGS84
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(center_x, center_y)

        logger.debug(f"Auto-extracted location: {lat:.4f}°N, {lon:.4f}°E (UTC{utc_offset:+d})")
        return cls(latitude=lat, longitude=lon, altitude=altitude, utc_offset=utc_offset)

    def to_sun_position_dict(self) -> dict:
        """Convert to dict format expected by sun_position module."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }


@dataclass
class Weather:
    """
    Weather/meteorological data for a single timestep.

    Only basic measurements are required. Derived values (sun position,
    direct/diffuse radiation split) are computed automatically.

    Attributes:
        datetime: Date and time of measurement (end of interval).
        ta: Air temperature in °C.
        rh: Relative humidity in % (0-100).
        global_rad: Global solar radiation in W/m².
        ws: Wind speed in m/s. Default 1.0.
        pressure: Atmospheric pressure in hPa. Default 1013.25.
        timestep_minutes: Data timestep in minutes. Default 60.0.
            Sun position is computed at datetime - timestep/2 to represent
            the center of the measurement interval.
        measured_direct_rad: Optional measured direct beam radiation in W/m².
            If provided with measured_diffuse_rad, these override the computed values.
        measured_diffuse_rad: Optional measured diffuse radiation in W/m².
            If provided with measured_direct_rad, these override the computed values.

    Auto-computed (after calling compute_derived()):
        sun_altitude: Sun altitude angle in degrees.
        sun_azimuth: Sun azimuth angle in degrees.
        direct_rad: Direct beam radiation in W/m² (from measured or computed).
        diffuse_rad: Diffuse radiation in W/m² (from measured or computed).
    """

    datetime: dt
    ta: float
    rh: float
    global_rad: float
    ws: float = 1.0
    pressure: float = 1013.25
    timestep_minutes: float = 60.0  # Timestep in minutes (for half-timestep sun position offset)
    measured_direct_rad: float | None = None  # Optional measured direct beam radiation
    measured_diffuse_rad: float | None = None  # Optional measured diffuse radiation
    precomputed_sun_altitude: float | None = None  # Optional pre-computed sun altitude
    precomputed_sun_azimuth: float | None = None  # Optional pre-computed sun azimuth
    precomputed_altmax: float | None = None  # Optional pre-computed max sun altitude for day

    # Auto-computed values (set by compute_derived)
    sun_altitude: float = field(default=0.0, init=False)
    sun_azimuth: float = field(default=0.0, init=False)
    sun_zenith: float = field(default=90.0, init=False)
    direct_rad: float = field(default=0.0, init=False)
    diffuse_rad: float = field(default=0.0, init=False)
    clearness_index: float = field(default=1.0, init=False)
    altmax: float = field(default=45.0, init=False)  # Maximum sun altitude for the day

    _derived_computed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not 0 <= self.rh <= 100:
            raise ValueError(f"Relative humidity must be in [0, 100], got {self.rh}")
        if self.global_rad < 0:
            raise ValueError(f"Global radiation must be >= 0, got {self.global_rad}")

    def compute_derived(self, location: Location) -> None:
        """
        Compute derived values: sun position and radiation split.

        Must be called before using sun_altitude, sun_azimuth, direct_rad,
        or diffuse_rad.

        Sun position is calculated at the center of the measurement interval
        (datetime - timestep/2), which is standard for meteorological data
        where measurements are averaged over the interval.

        Args:
            location: Geographic location for sun position calculation.
        """
        # Always create location_dict (needed for clearness index calculation)
        location_dict = location.to_sun_position_dict()

        # Use pre-computed sun position if provided, otherwise compute
        if self.precomputed_sun_altitude is not None and self.precomputed_sun_azimuth is not None:
            self.sun_altitude = self.precomputed_sun_altitude
            self.sun_azimuth = self.precomputed_sun_azimuth
            self.sun_zenith = 90.0 - self.sun_altitude
            self.altmax = self.precomputed_altmax if self.precomputed_altmax is not None else self.sun_altitude
        else:
            # Apply half-timestep offset for sun position
            # Meteorological data timestamps typically represent the end of an interval,
            # so we compute sun position at the center of the interval to match SOLWEIG runner
            from datetime import timedelta

            half_timestep = timedelta(minutes=self.timestep_minutes / 2.0)
            sun_time = self.datetime - half_timestep

            # Compute sun position using NREL algorithm
            time_dict = {
                "year": sun_time.year,
                "month": sun_time.month,
                "day": sun_time.day,
                "hour": sun_time.hour,
                "min": sun_time.minute,
                "sec": sun_time.second,
                "UTC": location.utc_offset,
            }
            location_dict = location.to_sun_position_dict()

            sun = sp.sun_position(time_dict, location_dict)

            # Extract scalar values (sun_position may return 0-d arrays)
            zenith = sun["zenith"]
            azimuth = sun["azimuth"]
            self.sun_zenith = float(np.asarray(zenith).flat[0]) if hasattr(zenith, "__iter__") else float(zenith)
            self.sun_azimuth = float(np.asarray(azimuth).flat[0]) if hasattr(azimuth, "__iter__") else float(azimuth)
            self.sun_altitude = 90.0 - self.sun_zenith

            # Calculate maximum sun altitude for the day (iterate in 15-min intervals)
            # This matches the method in configs.py:EnvironData
            from datetime import timedelta

            ymd = self.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
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
                alt_step = 90.0 - zenith_val
                if alt_step > sunmaximum:
                    sunmaximum = alt_step

            self.altmax = max(sunmaximum, 0.0)  # Ensure non-negative

        # Use measured radiation values if provided, otherwise compute
        if self.measured_direct_rad is not None and self.measured_diffuse_rad is not None:
            # Use pre-measured direct and diffuse radiation
            self.direct_rad = self.measured_direct_rad
            self.diffuse_rad = self.measured_diffuse_rad
            self.clearness_index = 1.0  # Not computed when using measured values
        elif self.sun_altitude > 0 and self.global_rad > 0:
            # Compute clearness index
            zen_rad = self.sun_zenith * (np.pi / 180.0)
            result = clearnessindex_2013b(
                zen_rad,
                self.datetime.timetuple().tm_yday,
                self.ta,
                self.rh / 100.0,
                self.global_rad,
                location_dict,
                self.pressure,
            )
            # clearnessindex_2013b returns: (I0, CI, Kt, I0_et, diff_et)
            _, self.clearness_index, kt, _, _ = result

            # Use Reindl model for diffuse fraction
            self.direct_rad, self.diffuse_rad = diffusefraction(
                self.global_rad, self.sun_altitude, kt, self.ta, self.rh
            )
        else:
            # Night or no radiation
            self.direct_rad = 0.0
            self.diffuse_rad = self.global_rad
            self.clearness_index = 1.0

        self._derived_computed = True

    @property
    def is_daytime(self) -> bool:
        """Check if sun is above horizon."""
        return self.sun_altitude > 0

    @classmethod
    def from_epw(
        cls,
        path: str | Path,
        start: str | dt | None = None,
        end: str | dt | None = None,
        hours: list[int] | None = None,
        year: int | None = None,
    ) -> list[Weather]:
        """
        Load weather data from an EnergyPlus Weather (EPW) file.

        Args:
            path: Path to the EPW file.
            start: Start date/datetime. Can be:
                   - ISO date string "YYYY-MM-DD" or "MM-DD" (for TMY with year=None)
                   - datetime object
                   If None, uses first date in file.
            end: End date/datetime (inclusive). Same format as start.
                 If None, uses same as start (single day).
            hours: List of hours to include (0-23). If None, includes all hours.
            year: Year override for TMY files. If None and start/end use MM-DD format,
                  matches any year in the file.

        Returns:
            List of Weather objects for each timestep in the requested range.

        Raises:
            FileNotFoundError: If the EPW file doesn't exist.
            ValueError: If requested dates are outside the EPW file's date range.

        Example:
            # Load a single day
            weather_list = Weather.from_epw("weather.epw", start="2023-07-15", end="2023-07-15")

            # Load with specific hours only (daylight hours)
            weather_list = Weather.from_epw(
                "weather.epw",
                start="2023-07-15",
                end="2023-07-16",
                hours=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            )

            # TMY file (year-agnostic)
            weather_list = Weather.from_epw("tmy.epw", start="07-15", end="07-15")
        """
        from . import io as common

        # Parse EPW file
        df, epw_info = common.read_epw(path)

        # Parse start/end dates
        def parse_date(date_val, is_tmy: bool, default_year: int):
            if date_val is None:
                return None
            if isinstance(date_val, dt):
                return date_val
            # String parsing
            date_str = str(date_val)
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 2:
                    # MM-DD format (TMY)
                    month, day = int(parts[0]), int(parts[1])
                    return dt(default_year, month, day)
                elif len(parts) == 3:
                    # YYYY-MM-DD format
                    return dt.fromisoformat(date_str)
            raise ValueError(f"Cannot parse date: {date_val}. Use 'YYYY-MM-DD' or 'MM-DD' format.")

        # Determine if using TMY mode (year-agnostic)
        is_tmy = year is None and start is not None and isinstance(start, str) and len(start.split("-")) == 2

        # Get default year from EPW data
        if df.index.empty:
            raise ValueError("EPW file contains no data")
        default_year = df.index[0].year if year is None else year

        # Parse dates
        start_dt = parse_date(start, is_tmy, default_year)
        end_dt = parse_date(end, is_tmy, default_year)

        if start_dt is None:
            start_dt = df.index[0].replace(tzinfo=None)
        if end_dt is None:
            end_dt = start_dt

        # Make end_dt inclusive of the full day
        if end_dt.hour == 0 and end_dt.minute == 0:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Filter by date range
        # Remove timezone from index for comparison if needed
        df_idx = df.index.tz_localize(None) if df.index.tz is not None else df.index

        if is_tmy:
            # TMY mode: match month and day, ignore year
            mask = (
                (df_idx.month > start_dt.month) | ((df_idx.month == start_dt.month) & (df_idx.day >= start_dt.day))
            ) & ((df_idx.month < end_dt.month) | ((df_idx.month == end_dt.month) & (df_idx.day <= end_dt.day)))
        else:
            # Normal mode: match full datetime
            mask = (df_idx >= start_dt) & (df_idx <= end_dt)

        df_filtered = df[mask]

        if df_filtered.empty:
            # Build helpful error message
            avail_start = df_idx.min()
            avail_end = df_idx.max()
            raise ValueError(
                f"Requested dates {start_dt.date()} to {end_dt.date()} not found in EPW file.\n"
                f"EPW file '{path}' contains data for: {avail_start.date()} to {avail_end.date()}\n"
                "Suggestions:\n"
                "  - Use dates within the available range\n"
                "  - For TMY files, use 'MM-DD' format (e.g., '07-15') to match any year"
            )

        # Filter by hours if specified
        if hours is not None:
            hours_set = set(hours)
            hour_mask = df_filtered.index.hour.isin(hours_set)
            df_filtered = df_filtered[hour_mask]

        # Create Weather objects
        weather_list = []
        for timestamp, row in df_filtered.iterrows():
            # Get timezone offset from EPW info
            tz_offset = int(epw_info.get("tz_offset", 0))

            # Create Weather object with available data
            # EPW has dni/dhi which we can use as measured values
            w = cls(
                datetime=timestamp.to_pydatetime().replace(tzinfo=None),
                ta=float(row["temp_air"]) if not np.isnan(row["temp_air"]) else 20.0,
                rh=float(row["relative_humidity"]) if not np.isnan(row["relative_humidity"]) else 50.0,
                global_rad=float(row["ghi"]) if not np.isnan(row["ghi"]) else 0.0,
                ws=float(row["wind_speed"]) if not np.isnan(row["wind_speed"]) else 1.0,
                pressure=(float(row["atmospheric_pressure"]) / 100.0)
                if not np.isnan(row["atmospheric_pressure"])
                else 1013.25,  # Convert Pa to hPa
                measured_direct_rad=float(row["dni"]) if not np.isnan(row["dni"]) else None,
                measured_diffuse_rad=float(row["dhi"]) if not np.isnan(row["dhi"]) else None,
            )
            weather_list.append(w)

        if weather_list:
            logger.info(
                f"Loaded {len(weather_list)} timesteps from EPW: "
                f"{weather_list[0].datetime.strftime('%Y-%m-%d %H:%M')} → "
                f"{weather_list[-1].datetime.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            logger.warning(f"No timesteps found in EPW file for date range {start_dt} to {end_dt}")

        return weather_list


@dataclass
class ModelConfig:
    """
    Model configuration for SOLWEIG calculations.

    Groups all computational settings in one typed object.
    Pure configuration - no paths or data.

    Attributes:
        use_anisotropic_sky: Use anisotropic sky model. Default False.
        human: Human body parameters for Tmrt calculations.
        material_params: Optional material properties from JSON file.
        outputs: Which outputs to save in timeseries calculations.
        use_legacy_kelvin_offset: Backward compatibility flag. Default False.

    Note:
        UTCI and PET are now computed via post-processing functions (compute_utci, compute_pet)
        rather than during the main calculation loop for better performance.

    Examples:
        Basic usage with defaults:

        >>> config = ModelConfig.defaults()
        >>> config.save("my_config.json")

        Custom configuration:

        >>> config = ModelConfig(
        ...     use_anisotropic_sky=True,
        ...     human=HumanParams(abs_k=0.7, posture="standing"),
        ... )

        Load from legacy JSON:

        >>> config = ModelConfig.from_json("parametersforsolweig.json")
    """

    use_anisotropic_sky: bool = False
    human: HumanParams | None = None
    material_params: SimpleNamespace | None = None
    outputs: list[str] = field(default_factory=lambda: ["tmrt"])
    use_legacy_kelvin_offset: bool = False

    def __post_init__(self):
        """Initialize default HumanParams if not provided."""
        # Defer import to avoid forward reference issues
        if self.human is None:
            # HumanParams is defined later in this module
            pass  # Will be instantiated when HumanParams is available

    @classmethod
    def defaults(cls) -> ModelConfig:
        """
        Standard configuration for most users.

        Returns:
            ModelConfig with recommended defaults:
            - Anisotropic sky enabled
        """
        return cls(
            use_anisotropic_sky=True,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """
        Load configuration from legacy JSON parameters file.

        Args:
            path: Path to parametersforsolweig.json

        Returns:
            ModelConfig with settings extracted from JSON

        Example:
            >>> config = ModelConfig.from_json("parametersforsolweig.json")
            >>> config.human.abs_k  # From Tmrt_params
            0.7
        """
        params = load_params(path)

        # Extract human parameters from JSON
        human = HumanParams()
        if hasattr(params, "Tmrt_params"):
            human.abs_k = getattr(params.Tmrt_params, "absK", 0.7)
            human.abs_l = getattr(params.Tmrt_params, "absL", 0.97)
            posture_str = getattr(params.Tmrt_params, "posture", "Standing")
            human.posture = posture_str.lower()

        if hasattr(params, "PET_settings"):
            human.age = getattr(params.PET_settings, "Age", 35)
            human.weight = getattr(params.PET_settings, "Weight", 75.0)
            human.height = getattr(params.PET_settings, "Height", 1.75)
            human.sex = getattr(params.PET_settings, "Sex", 1)
            human.activity = getattr(params.PET_settings, "Activity", 80.0)
            human.clothing = getattr(params.PET_settings, "clo", 0.9)

        return cls(
            human=human,
            material_params=params,
        )

    def save(self, path: str | Path):
        """
        Save configuration to JSON file.

        Args:
            path: Output path for JSON file

        Example:
            >>> config = ModelConfig.defaults()
            >>> config.save("my_settings.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to dict
        data = {
            "use_anisotropic_sky": self.use_anisotropic_sky,
            "outputs": self.outputs,
            "use_legacy_kelvin_offset": self.use_legacy_kelvin_offset,
            "human": {
                "posture": self.human.posture,
                "abs_k": self.human.abs_k,
                "abs_l": self.human.abs_l,
                "age": self.human.age,
                "weight": self.human.weight,
                "height": self.human.height,
                "sex": self.human.sex,
                "activity": self.human.activity,
                "clothing": self.human.clothing,
            }
            if self.human
            else None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            ModelConfig loaded from file

        Example:
            >>> config = ModelConfig.load("my_settings.json")
            >>> results = calculate_timeseries(surface, weather, config=config)
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        # Deserialize human params
        human = None
        if data.get("human"):
            human = HumanParams(**data["human"])

        return cls(
            use_anisotropic_sky=data.get("use_anisotropic_sky", False),
            human=human,
            outputs=data.get("outputs", ["tmrt"]),
            use_legacy_kelvin_offset=data.get("use_legacy_kelvin_offset", False),
        )


@dataclass
class HumanParams:
    """
    Human body parameters for thermal comfort calculations.

    These parameters affect how radiation is absorbed by a person.
    Default values represent a standard reference person.

    Attributes:
        posture: Body posture ("standing" or "sitting"). Default "standing".
        abs_k: Shortwave absorption coefficient. Default 0.7.
        abs_l: Longwave absorption coefficient. Default 0.97.

    PET-specific parameters (used by compute_pet() post-processing):
        age: Age in years. Default 35.
        weight: Body weight in kg. Default 75.
        height: Body height in meters. Default 1.75.
        sex: Biological sex (1=male, 2=female). Default 1.
        activity: Metabolic activity in W. Default 80.
        clothing: Clothing insulation in clo. Default 0.9.
    """

    posture: str = "standing"
    abs_k: float = 0.7
    abs_l: float = 0.97

    # PET-specific (optional)
    age: int = 35
    weight: float = 75.0
    height: float = 1.75
    sex: int = 1
    activity: float = 80.0
    clothing: float = 0.9

    def __post_init__(self):
        valid_postures = ("standing", "sitting")
        if self.posture not in valid_postures:
            raise ValueError(f"Posture must be one of {valid_postures}, got {self.posture}")
        if not 0 < self.abs_k <= 1:
            raise ValueError(f"abs_k must be in (0, 1], got {self.abs_k}")
        if not 0 < self.abs_l <= 1:
            raise ValueError(f"abs_l must be in (0, 1], got {self.abs_l}")


@dataclass
class SolweigResult:
    """
    Results from a SOLWEIG calculation.

    All output grids have the same shape as the input DSM.

    Attributes:
        tmrt: Mean Radiant Temperature grid (°C).
        utci: Universal Thermal Climate Index grid (°C). Optional.
        pet: Physiological Equivalent Temperature grid (°C). Optional.
        shadow: Shadow mask (1=shadow, 0=sunlit).
        kdown: Downwelling shortwave radiation (W/m²).
        kup: Upwelling shortwave radiation (W/m²).
        ldown: Downwelling longwave radiation (W/m²).
        lup: Upwelling longwave radiation (W/m²).
        state: Thermal state for multi-timestep chaining. Optional.
            When state parameter was passed to calculate(), this contains
            the updated state for the next timestep.
    """

    tmrt: NDArray[np.floating]
    shadow: NDArray[np.floating] | None = None
    kdown: NDArray[np.floating] | None = None
    kup: NDArray[np.floating] | None = None
    ldown: NDArray[np.floating] | None = None
    lup: NDArray[np.floating] | None = None
    utci: NDArray[np.floating] | None = None
    pet: NDArray[np.floating] | None = None
    state: ThermalState | None = None

    def to_geotiff(
        self,
        output_dir: str | Path,
        timestamp: dt | None = None,
        outputs: list[str] | None = None,
        surface: SurfaceData | None = None,
        transform: list[float] | None = None,
        crs_wkt: str | None = None,
    ) -> None:
        """
        Save results to GeoTIFF files.

        Creates one GeoTIFF file per output variable per timestep.
        Filename pattern: {output}_{YYYYMMDD}_{HHMM}.tif

        Args:
            output_dir: Directory to write GeoTIFF files.
            timestamp: Timestamp for filename. If None, uses current time.
            outputs: List of outputs to save. Options: "tmrt", "utci", "pet",
                "shadow", "kdown", "kup", "ldown", "lup".
                Default: ["tmrt"] (only save Mean Radiant Temperature).
            surface: SurfaceData object (if loaded via from_geotiff, contains CRS/transform).
                If provided and transform/crs_wkt not specified, uses surface metadata.
            transform: GDAL-style geotransform [x_origin, pixel_width, 0,
                y_origin, 0, -pixel_height]. If None, attempts to use surface metadata,
                otherwise uses identity transform.
            crs_wkt: Coordinate reference system in WKT format. If None, attempts to use
                surface metadata, otherwise no CRS set.

        Example:
            # With surface metadata (recommended when using from_geotiff)
            >>> surface, precomputed = SurfaceData.from_geotiff("dsm.tif", svf_dir="svf/")
            >>> result = solweig.calculate(surface, location, weather, precomputed=precomputed)
            >>> result.to_geotiff("output/", timestamp=weather.dt, surface=surface)

            # Without surface metadata (explicit transform/CRS)
            >>> result.to_geotiff(
            ...     "output/",
            ...     timestamp=datetime(2023, 7, 15, 12, 0),
            ...     outputs=["tmrt", "utci", "pet"],
            ...     transform=[0, 1, 0, 0, 0, -1],
            ...     crs_wkt="EPSG:32633",
            ... )
        """
        from . import io

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default outputs: just tmrt
        if outputs is None:
            outputs = ["tmrt"]

        # Default timestamp: current time
        if timestamp is None:
            timestamp = dt.now()

        # Format timestamp for filename
        ts_str = timestamp.strftime("%Y%m%d_%H%M")

        # Use surface metadata if available and not overridden
        if surface is not None:
            if transform is None and surface._geotransform is not None:
                transform = surface._geotransform
            if crs_wkt is None and surface._crs_wkt is not None:
                crs_wkt = surface._crs_wkt

        # Default transform: identity (top-left origin, 1m pixels)
        if transform is None:
            height, width = self.tmrt.shape
            transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

        # Default CRS: empty string (no CRS)
        if crs_wkt is None:
            crs_wkt = ""

        # Map output names to arrays
        available_outputs = {
            "tmrt": self.tmrt,
            "utci": self.utci,
            "pet": self.pet,
            "shadow": self.shadow,
            "kdown": self.kdown,
            "kup": self.kup,
            "ldown": self.ldown,
            "lup": self.lup,
        }

        # Save each requested output
        for name in outputs:
            if name not in available_outputs:
                logger.warning(f"Unknown output '{name}', skipping. Valid: {list(available_outputs.keys())}")
                continue

            array = available_outputs[name]
            if array is None:
                logger.warning(f"Output '{name}' is None (not computed), skipping.")
                continue

            # Write to GeoTIFF
            filepath = output_dir / f"{name}_{ts_str}.tif"
            io.save_raster(
                out_path_str=str(filepath),
                data_arr=array,
                trf_arr=transform,
                crs_wkt=crs_wkt,
                no_data_val=np.nan,
            )
            logger.debug(f"Saved {name} to {filepath}")


@dataclass
class TileSpec:
    """
    Specification for a single tile with overlap regions.

    Attributes:
        row_start, row_end: Core tile row bounds (without overlap).
        col_start, col_end: Core tile column bounds (without overlap).
        row_start_full, row_end_full: Full tile row bounds (with overlap).
        col_start_full, col_end_full: Full tile column bounds (with overlap).
        overlap_top, overlap_bottom: Vertical overlap in pixels.
        overlap_left, overlap_right: Horizontal overlap in pixels.
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_start_full: int
    row_end_full: int
    col_start_full: int
    col_end_full: int
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

    @property
    def core_slice(self) -> tuple[slice, slice]:
        """Slices for extracting core from full tile result."""
        return (
            slice(self.overlap_top, self.overlap_top + self.core_shape[0]),
            slice(self.overlap_left, self.overlap_left + self.core_shape[1]),
        )

    @property
    def write_slice(self) -> tuple[slice, slice]:
        """Slices for writing core to global output."""
        return (
            slice(self.row_start, self.row_end),
            slice(self.col_start, self.col_end),
        )

    @property
    def read_slice(self) -> tuple[slice, slice]:
        """Slices for reading full tile from global input."""
        return (
            slice(self.row_start_full, self.row_end_full),
            slice(self.col_start_full, self.col_end_full),
        )
