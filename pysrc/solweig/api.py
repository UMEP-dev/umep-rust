"""
Simplified SOLWEIG API

This module provides a clean, minimal API for SOLWEIG calculations.
It wraps the complex internal machinery with simple dataclasses that:
- Take minimal user input
- Auto-compute derived values (sun position, diffuse fraction, etc.)
- Provide sensible defaults

Example:
    import solweig
    from datetime import datetime

    result = solweig.calculate(
        surface=solweig.SurfaceData(dsm=my_dsm_array),
        location=solweig.Location(latitude=57.7, longitude=12.0),
        weather=solweig.Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=25.0, rh=50.0, global_rad=800.0
        ),
    )
    print(f"Tmrt: {result.tmrt.mean():.1f}°C")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from datetime import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from .algorithms import sun_position as sp
from .algorithms.clearnessindex_2013b import clearnessindex_2013b
from .algorithms.create_patches import create_patches
from .algorithms.cylindric_wedge import cylindric_wedge
from .algorithms.daylen import daylen
from .algorithms.diffusefraction import diffusefraction
from .algorithms.Kup_veg_2015a import Kup_veg_2015a
from .algorithms.patch_radiation import patch_steradians
from .algorithms.Perez_v3 import Perez_v3
from .algorithms.TsWaveDelay_2015a import TsWaveDelay_2015a
from .rustalgos import gvf as gvf_module
from .rustalgos import pet as pet_rust
from .rustalgos import shadowing, sky, skyview, vegetation
from .rustalgos import utci as utci_rust

# Stefan-Boltzmann constant
SBC = 5.67e-8

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _dict_to_namespace(d: dict[str, Any] | list | Any) -> SimpleNamespace | list | Any:
    """
    Recursively convert dicts to SimpleNamespace.

    This matches the runner's dict_to_namespace function for loading JSON parameters.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d


def load_params(params_json_path: str | Path) -> SimpleNamespace:
    """
    Load SOLWEIG parameters from a JSON file.

    Args:
        params_json_path: Path to the parameters JSON file
            (e.g., parametersforsolweig.json or test_params_solweig.json).

    Returns:
        SimpleNamespace object with nested parameter values accessible via attributes.

    Example:
        params = load_params("parametersforsolweig.json")
        # Access: params.Albedo.Effective.Value.Dark_asphalt -> 0.18
        # Access: params.Names.Value."0" -> "Cobble_stone_2014a"
    """
    params_path = Path(params_json_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path) as f:
        params_dict = json.load(f)

    return _dict_to_namespace(params_dict)


def _get_lc_properties_from_params(
    land_cover: "NDArray[np.integer]",
    params: SimpleNamespace,
    shape: tuple[int, int],
) -> tuple[
    "NDArray[np.floating]",
    "NDArray[np.floating]",
    "NDArray[np.floating]",
    "NDArray[np.floating]",
    "NDArray[np.floating]",
]:
    """
    Derive surface properties from land cover grid using loaded params.

    This mirrors the logic in configs.py TgMaps class.

    Args:
        land_cover: Land cover classification grid (UMEP standard IDs).
        params: Loaded parameters from JSON file.
        shape: Output grid shape (rows, cols).

    Returns:
        Tuple of (albedo_grid, emissivity_grid, tgk_grid, tstart_grid, tmaxlst_grid).
    """
    rows, cols = shape
    alb_grid = np.full((rows, cols), 0.15, dtype=np.float32)
    emis_grid = np.full((rows, cols), 0.95, dtype=np.float32)
    tgk_grid = np.full((rows, cols), 0.37, dtype=np.float32)
    tstart_grid = np.full((rows, cols), -3.41, dtype=np.float32)
    tmaxlst_grid = np.full((rows, cols), 15.0, dtype=np.float32)

    # Get unique land cover IDs and filter to valid ones (0-7)
    lc = np.copy(land_cover)
    lc[lc >= 100] = 2  # Treat wall codes as buildings
    unique_ids = np.unique(lc)
    valid_ids = unique_ids[unique_ids <= 7].astype(int)

    # Build mappings from land cover ID to name to parameter values
    for lc_id in valid_ids:
        # Get land cover name from ID (e.g., 0 -> "Cobble_stone_2014a")
        name = getattr(params.Names.Value, str(lc_id), None)
        if name is None:
            continue

        # Get parameter values for this land cover type
        albedo = getattr(params.Albedo.Effective.Value, name, 0.15)
        emissivity = getattr(params.Emissivity.Value, name, 0.95)
        tgk = getattr(params.Ts_deg.Value, name, 0.37)
        tstart = getattr(params.Tstart.Value, name, -3.41)
        tmaxlst = getattr(params.TmaxLST.Value, name, 15.0)

        # Apply to grid where land cover matches
        mask = lc == lc_id
        if np.any(mask):
            alb_grid[mask] = albedo
            emis_grid[mask] = emissivity
            tgk_grid[mask] = tgk
            tstart_grid[mask] = tstart
            tmaxlst_grid[mask] = tmaxlst

    return alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid


def _detect_building_mask(
    dsm: "NDArray[np.floating]",
    land_cover: "NDArray[np.integer] | None",
    wall_height: "NDArray[np.floating] | None",
    pixel_size: float,
) -> "NDArray[np.floating]":
    """
    Create a building mask for GVF calculation.

    GVF (Ground View Factor) expects: 0=building, 1=ground.
    This is used to normalize GVF values over buildings where GVF doesn't apply.

    Args:
        dsm: Digital Surface Model array.
        land_cover: Optional land cover grid (UMEP standard: ID 2 = buildings).
        wall_height: Optional wall height grid.
        pixel_size: Pixel size in meters.

    Returns:
        Building mask where 0=building pixels, 1=ground pixels.

    Detection strategy:
        1. If land_cover provided: Use ID 2 (buildings) directly
        2. Elif wall_height provided: Dilate wall pixels + detect elevated areas
        3. Else: Assume all ground (no buildings)
    """
    if land_cover is not None:
        # Use land cover directly: ID 2 = buildings
        buildings = np.ones_like(dsm, dtype=np.float32)
        buildings[land_cover == 2] = 0.0
        return buildings

    if wall_height is not None:
        # Approximate building footprints from wall heights
        # Wall pixels mark building edges; dilate to capture interiors
        from scipy import ndimage

        wall_mask = wall_height > 0

        # Dilate to capture building interiors (typical building width up to 50m)
        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        iterations = int(25 / pixel_size) + 1
        dilated = ndimage.binary_dilation(wall_mask, struct, iterations=iterations)

        # Also detect elevated areas (building roofs)
        ground_level = np.nanpercentile(dsm[~wall_mask], 10) if np.any(~wall_mask) else np.nanmin(dsm)
        elevated = dsm > (ground_level + 2.0)  # At least 2m above ground

        # Combine: building pixels where either dilated walls OR elevated flat areas
        is_building = dilated | (elevated & ~np.isnan(dsm))

        # Invert: 0=building, 1=ground
        return (~is_building).astype(np.float32)

    # No building info available - assume all ground
    return np.ones_like(dsm, dtype=np.float32)


def _compute_transmissivity(
    doy: int,
    params: SimpleNamespace | None = None,
    conifer: bool = False,
) -> float:
    """
    Compute vegetation transmissivity based on day of year and leaf status.

    This implements seasonal leaf on/off logic from configs.py EnvironData.
    During leaf-on season, vegetation transmits less light (low psi ~0.03).
    During leaf-off season (winter), bare branches transmit more light (psi ~0.5).

    Args:
        doy: Day of year (1-366)
        params: SOLWEIG params from load_params() containing Tree_settings.
            If provided, reads Transmissivity, First_day_leaf, Last_day_leaf.
        conifer: Override to treat vegetation as conifer (always leaf-on).

    Returns:
        Transmissivity value:
        - 0.03 (default) during leaf-on period
        - 0.5 during leaf-off period (deciduous trees in winter)
    """
    # Default values matching configs.py
    transmissivity = 0.03
    first_day = 100  # ~April 10
    last_day = 300   # ~October 27
    is_conifer = conifer

    # Override from params if provided
    if params is not None and hasattr(params, 'Tree_settings'):
        ts = params.Tree_settings.Value
        transmissivity = getattr(ts, 'Transmissivity', 0.03)
        first_day = int(getattr(ts, 'First_day_leaf', 100))
        last_day = int(getattr(ts, 'Last_day_leaf', 300))
        # Note: Conifer flag may not be in all params files
        is_conifer = conifer or getattr(ts, 'Conifer', False)

    # Determine leaf on/off
    if is_conifer:
        leaf_on = True
    elif first_day > last_day:
        # Wraps around year end (southern hemisphere or unusual dates)
        leaf_on = doy > first_day or doy < last_day
    else:
        # Normal case: leaves on between first_day and last_day
        leaf_on = first_day < doy < last_day

    # Return appropriate transmissivity
    # Leaf-off uses 0.5 to match configs.py: self.psi[self.leafon == 0] = 0.5
    return transmissivity if leaf_on else 0.5


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
    def initial(cls, shape: tuple[int, int]) -> "ThermalState":
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

    def copy(self) -> "ThermalState":
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
        wall_height: Wall height grid (meters). Optional but improves accuracy.
        wall_aspect: Wall aspect grid (degrees, 0=N). Optional.
        albedo: Per-pixel ground albedo (0-1). Optional, defaults to 0.15.
        emissivity: Per-pixel ground emissivity (0-1). Optional, defaults to 0.95.
        land_cover: Land cover classification grid (UMEP standard IDs). Optional.
            IDs: 0=paved, 1=asphalt, 2=buildings, 5=grass, 6=bare_soil, 7=water.
            When provided, albedo and emissivity are derived from land cover.
        pixel_size: Pixel size in meters. Default 1.0.
        trunk_ratio: Ratio for auto-generating TDSM from CDSM. Default 0.25.
        relative_heights: Whether CDSM/TDSM contain relative heights (above ground)
            rather than absolute elevations. Default True.
            If True and preprocess() is not called, a warning is issued.

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

    dsm: NDArray[np.floating]
    cdsm: NDArray[np.floating] | None = None
    dem: NDArray[np.floating] | None = None
    tdsm: NDArray[np.floating] | None = None
    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    albedo: NDArray[np.floating] | None = None
    emissivity: NDArray[np.floating] | None = None
    land_cover: NDArray[np.integer] | None = None
    pixel_size: float = 1.0
    trunk_ratio: float = 0.25  # Trunk zone ratio for auto-generating TDSM from CDSM
    relative_heights: bool = True  # Whether CDSM/TDSM are relative (not absolute)
    _preprocessed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Ensure dsm is float32 for memory efficiency
        self.dsm = np.asarray(self.dsm, dtype=np.float32)

        # Convert optional arrays if provided
        if self.cdsm is not None:
            self.cdsm = np.asarray(self.cdsm, dtype=np.float32)
        if self.dem is not None:
            self.dem = np.asarray(self.dem, dtype=np.float32)
        if self.tdsm is not None:
            self.tdsm = np.asarray(self.tdsm, dtype=np.float32)
        if self.wall_height is not None:
            self.wall_height = np.asarray(self.wall_height, dtype=np.float32)
        if self.wall_aspect is not None:
            self.wall_aspect = np.asarray(self.wall_aspect, dtype=np.float32)
        if self.albedo is not None:
            self.albedo = np.asarray(self.albedo, dtype=np.float32)
        if self.emissivity is not None:
            self.emissivity = np.asarray(self.emissivity, dtype=np.float32)
        if self.land_cover is not None:
            self.land_cover = np.asarray(self.land_cover, dtype=np.uint8)

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
            logger.info(
                f"Auto-generating TDSM from CDSM using trunk_ratio={self.trunk_ratio}"
            )
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
                f"Preprocessed CDSM/TDSM to absolute heights "
                f"(base: {'DEM' if self.dem is not None else 'DSM'})"
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
                    "CDSM appears to contain relative vegetation heights "
                    "(max CDSM=%.1fm < min DSM=%.1fm), but preprocess() was not called. "
                    "Call surface.preprocess() to convert to absolute heights, "
                    "or set relative_heights=False if CDSM already contains absolute elevations.",
                    np.nanmax(self.cdsm),
                    np.nanmin(self.dsm),
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

    @classmethod
    def from_files(
        cls,
        dsm_path: str | Path,
        cdsm_path: str | Path | None = None,
        dem_path: str | Path | None = None,
        tdsm_path: str | Path | None = None,
        pixel_size: float = 1.0,
    ) -> SurfaceData:
        """
        Load surface data from raster files.

        Args:
            dsm_path: Path to DSM raster file.
            cdsm_path: Path to CDSM raster file. Optional.
            dem_path: Path to DEM raster file. Optional.
            tdsm_path: Path to TDSM raster file. Optional.
            pixel_size: Pixel size in meters.

        Returns:
            SurfaceData instance with loaded arrays.
        """
        from . import io as common

        dsm = common.load_raster(str(dsm_path))

        cdsm = None
        if cdsm_path is not None:
            cdsm = common.load_raster(str(cdsm_path))

        dem = None
        if dem_path is not None:
            dem = common.load_raster(str(dem_path))

        tdsm = None
        if tdsm_path is not None:
            tdsm = common.load_raster(str(tdsm_path))

        return cls(dsm=dsm, cdsm=cdsm, dem=dem, tdsm=tdsm, pixel_size=pixel_size)


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

    def __post_init__(self):
        # Ensure uint8 storage
        if self._shmat_u8.dtype != np.uint8:
            self._shmat_u8 = (np.clip(self._shmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vegshmat_u8.dtype != np.uint8:
            self._vegshmat_u8 = (np.clip(self._vegshmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vbshmat_u8.dtype != np.uint8:
            self._vbshmat_u8 = (np.clip(self._vbshmat_u8, 0, 1) * 255).astype(np.uint8)

        self.patch_count = self._shmat_u8.shape[2]

    @property
    def shmat(self) -> NDArray[np.floating]:
        """Building shadow matrix as float32 (0.0-1.0). Converted on-demand."""
        return self._shmat_u8.astype(np.float32) / 255.0

    @property
    def vegshmat(self) -> NDArray[np.floating]:
        """Vegetation shadow matrix as float32 (0.0-1.0). Converted on-demand."""
        return self._vegshmat_u8.astype(np.float32) / 255.0

    @property
    def vbshmat(self) -> NDArray[np.floating]:
        """Combined shadow matrix as float32 (0.0-1.0). Converted on-demand."""
        return self._vbshmat_u8.astype(np.float32) / 255.0

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
    Container for pre-computed data to skip expensive calculations.

    Use this to provide already-computed SVF and/or shadow matrices
    to the calculate() function. This is useful when:
    - Running multiple timesteps with the same geometry
    - Using data generated by external tools
    - Optimizing performance by pre-computing once

    Attributes:
        svf: Pre-computed SVF arrays. If None, SVF is computed on-the-fly.
        shadow_matrices: Pre-computed anisotropic shadow matrices.
            If None, isotropic sky model is used.

    Example:
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

    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None


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
                zenith_val = float(np.asarray(zenith_step).flat[0]) if hasattr(zenith_step, "__iter__") else float(zenith_step)
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

    PET-specific parameters (only used if compute_pet=True):
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


def calculate(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool = False,
    compute_utci: bool = True,
    compute_pet: bool = False,
    poi_coords: list[tuple[int, int]] | None = None,
    state: ThermalState | None = None,
    params: SimpleNamespace | None = None,
) -> SolweigResult:
    """
    Calculate mean radiant temperature and thermal comfort indices.

    This is the main entry point for SOLWEIG calculations.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        location: Geographic location (lat, lon, UTC offset).
        weather: Weather data (datetime, temperature, humidity, radiation).
        human: Human body parameters. Uses defaults if not provided.
        precomputed: Pre-computed SVF and/or shadow matrices. Optional.
            When provided, skips expensive SVF computation.
        use_anisotropic_sky: Use anisotropic sky model for radiation. Default False.
            Requires precomputed.shadow_matrices to be provided.
            Uses Perez diffuse model and patch-based longwave calculation.
        compute_utci: Whether to compute UTCI. Default True.
        compute_pet: Whether to compute PET. Default False (slower).
        poi_coords: Optional list of (row, col) coordinates for POI mode.
            If provided, only computes at these points (much faster).
        state: Thermal state from previous timestep. Optional.
            When provided, enables accurate multi-timestep simulation with
            thermal inertia modeling (TsWaveDelay). The returned result
            will include updated state for the next timestep.
        params: Loaded SOLWEIG parameters from JSON file (via load_params()).
            When provided, uses these parameters for land cover properties.
            When None, uses built-in defaults matching parametersforsolweig.json.

    Returns:
        SolweigResult with Tmrt and optionally UTCI/PET grids.
        When state parameter is provided, result.state contains the
        updated thermal state for the next timestep.

    Example:
        # Single timestep (default)
        result = calculate(
            surface=SurfaceData(dsm=my_dsm),
            location=Location(latitude=57.7, longitude=12.0),
            weather=Weather(datetime=dt, ta=25, rh=50, global_rad=800),
        )

        # Multi-timestep with state management
        state = ThermalState.initial(dsm.shape)
        for weather in weather_list:
            result = calculate(surface, location, weather, state=state)
            state = result.state  # Carry forward to next timestep

        # With pre-computed SVF and anisotropic sky:
        svf = SvfArrays.from_zip("path/to/svfs.zip")
        shadows = ShadowArrays.from_npz("path/to/shadowmats.npz")
        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            precomputed=PrecomputedData(svf=svf, shadow_matrices=shadows),
            use_anisotropic_sky=True,
        )

        # With custom parameters from JSON file:
        params = load_params("parametersforsolweig.json")
        result = calculate(surface=surface, location=location, weather=weather, params=params)
    """
    # Use default human params if not provided
    if human is None:
        human = HumanParams()

    # Compute derived weather values (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Call the core calculation
    return _calculate_core(
        surface=surface,
        location=location,
        weather=weather,
        human=human,
        precomputed=precomputed,
        use_anisotropic_sky=use_anisotropic_sky,
        compute_utci=compute_utci,
        compute_pet=compute_pet,
        poi_coords=poi_coords,
        state=state,
        params=params,
    )


def _calculate_core(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    use_anisotropic_sky: bool,
    compute_utci: bool,
    compute_pet: bool,
    poi_coords: list[tuple[int, int]] | None,
    state: ThermalState | None = None,
    params: SimpleNamespace | None = None,
) -> SolweigResult:
    """
    Core calculation implementation using Rust modules.

    When wall_height and wall_aspect are provided in surface data,
    uses the full GVF module for accurate wall radiation.
    Otherwise falls back to simplified model.

    When precomputed.svf is provided, skips SVF computation.
    When use_anisotropic_sky=True and precomputed.shadow_matrices is provided,
    uses the anisotropic sky model with Perez diffuse distribution.

    When state is provided, applies TsWaveDelay_2015a for thermal inertia
    and returns updated state in the result.
    """
    rows, cols = surface.shape
    dsm = surface.dsm
    pixel_size = surface.pixel_size
    max_height = surface.max_height

    # Check if vegetation data needs preprocessing
    surface._check_preprocessing_needed()

    # Prepare vegetation arrays
    cdsm = surface.cdsm
    tdsm = surface.tdsm
    use_veg = cdsm is not None

    # Bush array (low vegetation where canopy exists but no trunk)
    # Formula from configs.py: bush = logical_not(tdsm * cdsm) * cdsm
    # This identifies shrubs/bushes that have canopy but no defined trunk zone
    if use_veg and tdsm is not None:
        # Bush exists where there's canopy (cdsm > 0) but trunk is zero/absent
        bush = np.where((tdsm == 0) & (cdsm > 0), cdsm, 0.0).astype(np.float32)
    else:
        bush = np.zeros_like(dsm)

    # Check for wall data - enables full radiation model
    has_walls = surface.wall_height is not None and surface.wall_aspect is not None
    wall_ht = surface.wall_height if has_walls else np.zeros_like(dsm)
    wall_asp = surface.wall_aspect if has_walls else np.zeros_like(dsm)
    wall_asp_rad = wall_asp * (np.pi / 180.0) if has_walls else np.zeros_like(dsm)

    # Identify building pixels for GVF calculation
    buildings = _detect_building_mask(dsm, surface.land_cover, wall_ht if has_walls else None, pixel_size)

    # Surface parameters - use provided grids or defaults
    albedo_wall = 0.20
    emis_wall = 0.90

    # Get surface properties: use land cover if available, else explicit grids or defaults
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = surface.get_land_cover_properties(params)

    # Night check - if sun below horizon, return simplified result
    if weather.sun_altitude <= 0:
        # Nighttime: Tmrt ≈ Ta (simplified)
        tmrt = np.full((rows, cols), weather.ta, dtype=np.float32)
        shadow = np.zeros((rows, cols), dtype=np.float32)  # 0 = shaded (night)

        # Nighttime Lup = SBC * emis * Ta^4 (matching reference, Knight=0, Tg=0)
        ta_k = weather.ta + 273.15
        lup_night = SBC * emis_grid * np.power(ta_k, 4)
        ldown_night = np.full((rows, cols), SBC * 0.95 * np.power(ta_k, 4), dtype=np.float32)

        utci_grid = None
        if compute_utci:
            utci_grid = utci_rust.utci_grid(
                weather.ta,
                weather.rh,
                tmrt,
                np.full((rows, cols), weather.ws, dtype=np.float32),
            )

        # Update thermal state for nighttime (matching reference lines 603-604)
        output_state = None
        if state is not None:
            state.firstdaytime = 1.0  # Reset for morning
            state.timeadd = 0.0  # Reset time accumulator
            output_state = state.copy()

        return SolweigResult(
            tmrt=tmrt,
            shadow=shadow,
            kdown=np.zeros((rows, cols), dtype=np.float32),
            kup=np.zeros((rows, cols), dtype=np.float32),
            ldown=ldown_night.astype(np.float32),
            lup=lup_night.astype(np.float32),
            utci=utci_grid,
            state=output_state,
        )

    # === Daytime calculation ===

    # 1. Get SVF arrays - use precomputed if available, otherwise compute
    if precomputed is not None and precomputed.svf is not None:
        # Use precomputed SVF
        svf_data = precomputed.svf
        svf = svf_data.svf
        svf_n = svf_data.svf_north
        svf_e = svf_data.svf_east
        svf_s = svf_data.svf_south
        svf_w = svf_data.svf_west
        svf_veg = svf_data.svf_veg
        svf_veg_n = svf_data.svf_veg_north
        svf_veg_e = svf_data.svf_veg_east
        svf_veg_s = svf_data.svf_veg_south
        svf_veg_w = svf_data.svf_veg_west
        svf_aveg = svf_data.svf_aveg
        svf_aveg_n = svf_data.svf_aveg_north
        svf_aveg_e = svf_data.svf_aveg_east
        svf_aveg_s = svf_data.svf_aveg_south
        svf_aveg_w = svf_data.svf_aveg_west
        svfbuveg = svf_data.svfbuveg
        # Precomputed svfbuveg already includes transmissivity adjustment
        _svfbuveg_needs_psi_adjustment = False
    else:
        # Compute SVF with directional components
        svf_result = skyview.calculate_svf(
            dsm,
            cdsm if use_veg else np.zeros_like(dsm),
            tdsm if use_veg else np.zeros_like(dsm),
            pixel_size,
            use_veg,
            max_height,
            2,  # patch_option (153 patches)
            3.0,  # min_sun_elev_deg
            None,
        )

        svf = np.array(svf_result.svf)
        svf_n = np.array(svf_result.svf_north)
        svf_e = np.array(svf_result.svf_east)
        svf_s = np.array(svf_result.svf_south)
        svf_w = np.array(svf_result.svf_west)
        svf_veg = np.array(svf_result.svf_veg) if use_veg else np.zeros_like(svf)
        svf_veg_n = np.array(svf_result.svf_veg_north) if use_veg else np.zeros_like(svf)
        svf_veg_e = np.array(svf_result.svf_veg_east) if use_veg else np.zeros_like(svf)
        svf_veg_s = np.array(svf_result.svf_veg_south) if use_veg else np.zeros_like(svf)
        svf_veg_w = np.array(svf_result.svf_veg_west) if use_veg else np.zeros_like(svf)
        svf_aveg = np.array(svf_result.svf_veg_blocks_bldg_sh) if use_veg else np.zeros_like(svf)
        svf_aveg_n = np.array(svf_result.svf_veg_blocks_bldg_sh_north) if use_veg else np.zeros_like(svf)
        svf_aveg_e = np.array(svf_result.svf_veg_blocks_bldg_sh_east) if use_veg else np.zeros_like(svf)
        svf_aveg_s = np.array(svf_result.svf_veg_blocks_bldg_sh_south) if use_veg else np.zeros_like(svf)
        svf_aveg_w = np.array(svf_result.svf_veg_blocks_bldg_sh_west) if use_veg else np.zeros_like(svf)

        # Combined SVF (preliminary - will be recomputed after psi is determined)
        # This flag tracks whether svfbuveg needs recalculation with transmissivity
        _svfbuveg_needs_psi_adjustment = True
        svfbuveg = svf + svf_veg - 1.0
        svfbuveg = np.clip(svfbuveg, 0.0, 1.0)

    # 2. Compute shadows with wall data if available
    shadow_result = shadowing.calculate_shadows_wall_ht_25(
        weather.sun_azimuth,
        weather.sun_altitude,
        pixel_size,
        max_height,
        dsm,
        cdsm if use_veg else None,
        tdsm if use_veg else None,
        bush if use_veg else None,
        wall_ht if has_walls else None,
        wall_asp_rad if has_walls else None,
        None,  # walls_scheme
        None,  # aspect_scheme
        3.0,
    )

    # Vegetation transmissivity - compute dynamically based on season
    doy = weather.datetime.timetuple().tm_yday
    psi = _compute_transmissivity(doy, params)

    # Recompute svfbuveg with transmissivity if computed on-the-fly
    # Formula from configs.py: svfbuveg = svf - (1 - svf_veg) * (1 - transmissivity)
    if _svfbuveg_needs_psi_adjustment and use_veg:
        svfbuveg = svf - (1.0 - svf_veg) * (1.0 - psi)
        svfbuveg = np.clip(svfbuveg, 0.0, 1.0)

    # Compute combined shadow accounting for vegetation transmissivity
    # This matches the reference: shadow = bldg_sh - (1 - veg_sh) * (1 - psi)
    # where psi is vegetation transmissivity (fraction of light that passes through)
    bldg_sh = np.array(shadow_result.bldg_sh)
    if use_veg:
        veg_sh = np.array(shadow_result.veg_sh)
        shadow = bldg_sh - (1 - veg_sh) * (1 - psi)
        # Note: No clipping here to match reference exactly. In practice, shadow
        # should stay in [0,1] because veg_sh is constrained by bldg_sh.
    else:
        shadow = bldg_sh
    wallsun = np.array(shadow_result.wall_sun) if has_walls else np.zeros_like(dsm)

    # 3. Ground temperature model
    # Based on SOLWEIG TgMaps with parameterization from land cover
    # Matches the reference implementation in solweig.py (Lindberg et al. 2008, 2016)
    ta_k = weather.ta + 273.15

    # Get day of year and calculate sunrise time
    jday = weather.datetime.timetuple().tm_yday
    _, _, _, snup = daylen(jday, location.latitude)

    # Wall parameters (scalar, use default cobblestone values)
    tstart_wall = -3.41  # Wall baseline offset (Walls in parametersforsolweig.json)
    tmaxlst_wall = 15.0  # Wall max surface temp hour

    # Maximum sun altitude for the day (computed in Weather.compute_derived())
    altmax = weather.altmax

    # Temperature amplitude based on max sun altitude (per-pixel from land cover)
    # tgk_grid contains TgK values: asphalt ~0.58, grass ~0.21, water ~0.0
    # tstart_grid contains Tstart values: asphalt ~-9.78, grass ~-3.38, water ~0.0
    # Formula: Tgamp = TgK * altmax + Tstart (Lindberg et al. 2008, 2016)
    tgamp = tgk_grid * altmax + tstart_grid

    # Wall temperature amplitude
    tgk_wall = 0.37  # Wall TgK (Walls in parametersforsolweig.json)
    tgamp_wall = tgk_wall * altmax + tstart_wall

    # Decimal time (fraction of day)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0

    # Phase calculation matching reference (per-pixel for ground):
    # phase = ((dectime - SNUP/24) / (TmaxLST/24 - SNUP/24))
    # Tg = Tgamp * sin(phase * pi/2)
    snup_frac = snup / 24.0
    tmaxlst_frac = tmaxlst_grid / 24.0  # Per-pixel from land cover
    tmaxlst_wall_frac = tmaxlst_wall / 24.0

    # Per-pixel phase calculation for ground
    # tmaxlst_grid varies by land cover: 15h for paved/asphalt, 14h for grass, 12h for water
    denom = tmaxlst_frac - snup_frac
    denom = np.where(denom > 0, denom, 1.0)  # Avoid division by zero
    phase = (dectime - snup_frac) / denom
    phase = np.clip(phase, 0.0, 1.0)

    # Ground temperature: only positive when after sunrise
    tg = np.where(dectime > snup_frac, tgamp * np.sin(phase * np.pi / 2.0), 0.0)

    # Wall phase (scalar)
    if dectime > snup_frac and tmaxlst_wall_frac > snup_frac:
        phase_wall = (dectime - snup_frac) / (tmaxlst_wall_frac - snup_frac)
        phase_wall = min(max(phase_wall, 0.0), 1.0)
        tg_wall = tgamp_wall * np.sin(phase_wall * np.pi / 2.0)
    else:
        tg_wall = 0.0

    # Clamp negative Tg values (morning transition, can happen with negative Tstart)
    tg = np.maximum(tg, 0.0)
    tg_wall = max(tg_wall, 0.0) if isinstance(tg_wall, (int, float)) else np.maximum(tg_wall, 0.0)

    # CI_TgG correction for non-clear conditions (Lindberg et al. 2008, Reindl et al. 1990)
    # This accounts for reduced ground heating under cloudy skies
    # Full formula from solweig.py: CI_TgG = (radG / radG0) + (1 - corr)
    zen = (90.0 - weather.sun_altitude) * (np.pi / 180.0)  # zenith in radians
    deg2rad = np.pi / 180.0

    # Get clear sky radiation (I0) from clearnessindex function
    location_dict = {"latitude": location.latitude, "longitude": location.longitude, "altitude": 0.0}
    i0, _, _, _, _ = clearnessindex_2013b(
        zen, jday, weather.ta, weather.rh / 100.0, weather.global_rad, location_dict, -999.0
    )

    # Calculate clear sky direct and diffuse components
    if i0 > 0 and weather.sun_altitude > 0:
        rad_i0, rad_d0 = diffusefraction(i0, weather.sun_altitude, 1.0, weather.ta, weather.rh)
        # Clear sky global horizontal radiation
        rad_g0 = rad_i0 * np.sin(weather.sun_altitude * deg2rad) + rad_d0
        # Zenith correction (Lindberg et al. 2008)
        zen_deg = 90.0 - weather.sun_altitude
        if zen_deg > 0 and zen_deg < 90:
            corr = 0.1473 * np.log(90.0 - zen_deg) + 0.3454
        else:
            corr = 0.3454
        # CI_TgG calculation
        if rad_g0 > 0:
            ci_tg = (weather.global_rad / rad_g0) + (1.0 - corr)
            ci_tg = min(ci_tg, 1.0)  # Clamp to max 1
            if np.isinf(ci_tg) or np.isnan(ci_tg):
                ci_tg = 1.0
        else:
            ci_tg = weather.clearness_index
    else:
        ci_tg = weather.clearness_index

    tg = tg * ci_tg
    tg_wall = tg_wall * ci_tg

    # Clamp negative values after CI correction
    tg = np.maximum(tg, 0.0)

    # 4. Compute GVF (Ground View Factors with wall radiation)
    # This is the key to proper wall radiation
    # Note: alb_grid and emis_grid are defined earlier from surface data or defaults

    # Human height parameters for GVF (matching runner: first=round(height), second=round(height*20))
    first = np.round(human.height)
    if first == 0.0:
        first = 1.0
    second = np.round(human.height * 20.0)

    # Land cover settings for gvf_calc
    use_landcover = surface.land_cover is not None
    lc_grid = surface.land_cover.astype(np.float32) if use_landcover else None

    if has_walls:
        # Use full GVF calculation with wall radiation
        gvf_result = gvf_module.gvf_calc(
            wallsun.astype(np.float32),
            wall_ht.astype(np.float32),
            buildings.astype(np.float32),
            pixel_size,
            shadow.astype(np.float32),
            first,
            second,
            wall_asp.astype(np.float32),
            tg.astype(np.float32),
            tg_wall,
            weather.ta,
            emis_grid.astype(np.float32),
            emis_wall,
            alb_grid.astype(np.float32),
            SBC,
            albedo_wall,
            weather.ta,  # Twater = Ta (approximation for water temperature)
            lc_grid,
            use_landcover,
        )

        # Extract GVF results for Lup and Kup calculations
        gvf_lup = np.array(gvf_result.gvf_lup)
        gvf_lup_e = np.array(gvf_result.gvf_lup_e)
        gvf_lup_s = np.array(gvf_result.gvf_lup_s)
        gvf_lup_w = np.array(gvf_result.gvf_lup_w)
        gvf_lup_n = np.array(gvf_result.gvf_lup_n)
        gvfalb = np.array(gvf_result.gvfalb)
        gvfalb_e = np.array(gvf_result.gvfalb_e)
        gvfalb_s = np.array(gvf_result.gvfalb_s)
        gvfalb_w = np.array(gvf_result.gvfalb_w)
        gvfalb_n = np.array(gvf_result.gvfalb_n)
        gvfalbnosh = np.array(gvf_result.gvfalbnosh)
        gvfalbnosh_e = np.array(gvf_result.gvfalbnosh_e)
        gvfalbnosh_s = np.array(gvf_result.gvfalbnosh_s)
        gvfalbnosh_w = np.array(gvf_result.gvfalbnosh_w)
        gvfalbnosh_n = np.array(gvf_result.gvfalbnosh_n)
    else:
        # Simplified GVF (no walls)
        # Note: Multiply tg by shadow - shaded areas have cooler ground
        # (Rust gvf_calc does this internally for the full path)
        gvf_simple = 1.0 - svf
        tg_with_shadow = tg * shadow  # shadow=1 sunlit, shadow=0 shaded
        gvf_lup = emis_grid * SBC * np.power(weather.ta + tg_with_shadow + 273.15, 4)
        gvf_lup_e = gvf_lup
        gvf_lup_s = gvf_lup
        gvf_lup_w = gvf_lup
        gvf_lup_n = gvf_lup
        gvfalb = alb_grid * gvf_simple
        gvfalb_e = gvfalb
        gvfalb_s = gvfalb
        gvfalb_w = gvfalb
        gvfalb_n = gvfalb
        gvfalbnosh = alb_grid
        gvfalbnosh_e = alb_grid
        gvfalbnosh_s = alb_grid
        gvfalbnosh_w = alb_grid
        gvfalbnosh_n = alb_grid

    # 5. Apply TsWaveDelay for thermal inertia (BEFORE lside_veg, matching reference)
    # The reference applies TsWaveDelay immediately after GVF calculation
    output_state = None
    if state is not None:
        # Apply TsWaveDelay_2015a for thermal mass effect
        # This smooths rapid temperature changes to model thermal inertia
        lup, _, state.tgmap1 = TsWaveDelay_2015a(
            gvf_lup, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgmap1
        )
        lup_e, _, state.tgmap1_e = TsWaveDelay_2015a(
            gvf_lup_e, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgmap1_e
        )
        lup_s, _, state.tgmap1_s = TsWaveDelay_2015a(
            gvf_lup_s, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgmap1_s
        )
        lup_w, _, state.tgmap1_w = TsWaveDelay_2015a(
            gvf_lup_w, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgmap1_w
        )
        lup_n, _, state.tgmap1_n = TsWaveDelay_2015a(
            gvf_lup_n, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgmap1_n
        )

        # Ground temperature output with delay
        tg_temp = tg * shadow + weather.ta
        _, state.timeadd, state.tgout1 = TsWaveDelay_2015a(
            tg_temp, state.firstdaytime, state.timeadd, state.timestep_dec, state.tgout1
        )

        # Update firstdaytime flag for next timestep
        if weather.is_daytime:
            state.firstdaytime = 0.0
        else:
            state.firstdaytime = 1.0
            state.timeadd = 0.0

        # Return a copy of state to avoid mutation issues
        output_state = state.copy()
    else:
        # Single timestep: use raw GVF values (no thermal delay)
        lup = gvf_lup
        lup_e = gvf_lup_e
        lup_s = gvf_lup_s
        lup_w = gvf_lup_w
        lup_n = gvf_lup_n

    # 6. Compute radiation components

    # Sky emissivity
    ea = 6.107 * 10 ** ((7.5 * weather.ta) / (237.3 + weather.ta)) * (weather.rh / 100.0)
    msteg = 46.5 * (ea / ta_k)
    esky = 1 - (1 + msteg) * np.exp(-np.sqrt(1.2 + 3.0 * msteg))

    # View factors (from SOLWEIG parameters)
    cyl = human.posture == "standing"
    if cyl:
        f_up = 0.06
        f_side = 0.22
        f_cyl = 0.28  # Cylindrical projection factor for direct beam
    else:
        f_up = 0.166666
        f_side = 0.166666
        f_cyl = 0.2

    # Shortwave radiation
    sin_alt = np.sin(np.radians(weather.sun_altitude))
    rad_i = weather.direct_rad
    rad_d = weather.diffuse_rad
    rad_g = weather.global_rad

    # Check if anisotropic sky model should be used
    has_shadow_matrices = (
        precomputed is not None
        and precomputed.shadow_matrices is not None
    )
    use_aniso = use_anisotropic_sky and has_shadow_matrices

    # Compute svfalfa (SVF angle) from SVF values
    # Formula: svfalfa = arcsin(exp(log(1 - (svf + svf_veg - 1)) / 2))
    tmp = np.clip(svf + svf_veg - 1.0, 0.0, 1.0)
    eps = np.finfo(np.float32).tiny
    safe_term = np.clip(1.0 - tmp, eps, 1.0)
    svfalfa = np.arcsin(np.exp(np.log(safe_term) / 2.0))

    # Compute F_sh (fraction shadow on building walls based on sun altitude and SVF)
    zen = weather.sun_zenith * (np.pi / 180.0)  # Convert to radians for cylindric_wedge
    f_sh = cylindric_wedge(zen, svfalfa, rows, cols)
    f_sh = np.nan_to_num(f_sh, nan=0.5)

    # Compute Kup (ground-reflected shortwave) using full directional model
    kup, kup_e, kup_s, kup_w, kup_n = Kup_veg_2015a(
        rad_i, rad_d, rad_g,
        weather.sun_altitude,
        svfbuveg,
        albedo_wall,
        f_sh,
        gvfalb, gvfalb_e, gvfalb_s, gvfalb_w, gvfalb_n,
        gvfalbnosh, gvfalbnosh_e, gvfalbnosh_s, gvfalbnosh_w, gvfalbnosh_n,
    )

    # Compute diffuse radiation and directional shortwave
    if use_aniso:
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        shadow_mats = precomputed.shadow_matrices
        patch_option = shadow_mats.patch_option
        jday = weather.datetime.timetuple().tm_yday

        # Get Perez luminance distribution
        lv, _, _ = Perez_v3(
            weather.sun_zenith,
            weather.sun_azimuth,
            rad_d,
            rad_i,
            jday,
            patchchoice=1,
            patch_option=patch_option,
        )

        # Get diffuse shadow matrix (accounts for vegetation transmissivity)
        diffsh = shadow_mats.diffsh(psi, use_veg)

        # Total relative luminance from sky patches into each cell
        ani_lum = np.zeros((rows, cols), dtype=np.float32)
        for idx in range(lv.shape[0]):
            ani_lum += diffsh[:, :, idx] * lv[idx, 2]

        drad = ani_lum * rad_d

        # Compute asvf (angle from SVF) for anisotropic calculations
        asvf = np.arccos(np.sqrt(np.clip(svf, 0.0, 1.0)))

        # Get raw shadow matrices for Rust functions
        shmat = shadow_mats.shmat.astype(np.float32)
        vegshmat = shadow_mats.vegshmat.astype(np.float32)
        vbshmat = shadow_mats.vbshmat.astype(np.float32)

        # Compute base Ldown first (needed for lside_veg)
        ldown_base = (
            (svf + svf_veg - 1) * esky * SBC * (ta_k ** 4)
            + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k ** 4)
            + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + 273.15) ** 4)
            + (2 - svf - svf_veg) * (1 - emis_wall) * esky * SBC * (ta_k ** 4)
        )

        # CI correction for non-clear conditions
        ci = weather.clearness_index
        if ci < 0.95:
            c = 1.0 - ci
            ldown_cloudy = (
                (svf + svf_veg - 1) * SBC * (ta_k ** 4)
                + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k ** 4)
                + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + 273.15) ** 4)
                + (2 - svf - svf_veg) * (1 - emis_wall) * SBC * (ta_k ** 4)
            )
            ldown_base = ldown_base * (1 - c) + ldown_cloudy * c

        # Call lside_veg for base directional longwave (Least, Lsouth, Lwest, Lnorth)
        lside_veg_result = vegetation.lside_veg(
            svf_s.astype(np.float32),
            svf_w.astype(np.float32),
            svf_n.astype(np.float32),
            svf_e.astype(np.float32),
            svf_veg_e.astype(np.float32),
            svf_veg_s.astype(np.float32),
            svf_veg_w.astype(np.float32),
            svf_veg_n.astype(np.float32),
            svf_aveg_e.astype(np.float32),
            svf_aveg_s.astype(np.float32),
            svf_aveg_w.astype(np.float32),
            svf_aveg_n.astype(np.float32),
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            tg_wall,
            SBC,
            emis_wall,
            ldown_base.astype(np.float32),
            esky,
            0.0,  # t (instrument offset, matching reference)
            f_sh.astype(np.float32),
            weather.clearness_index,
            lup_e.astype(np.float32),  # TsWaveDelay-processed values (matching reference)
            lup_s.astype(np.float32),
            lup_w.astype(np.float32),
            lup_n.astype(np.float32),
            True,  # anisotropic_sky flag
        )
        # Extract base directional longwave
        lside_e_base = np.array(lside_veg_result.least)
        lside_s_base = np.array(lside_veg_result.lsouth)
        lside_w_base = np.array(lside_veg_result.lwest)
        lside_n_base = np.array(lside_veg_result.lnorth)

        # Compute steradians for patches
        steradians, _, _ = patch_steradians(lv)

        # Create L_patches array for anisotropic sky (altitude, azimuth, luminance)
        l_patches = lv.astype(np.float32)

        # Adjust sky emissivity for cloudy conditions (CI < 0.95)
        # This matches the reference implementation: esky = CI * esky + (1 - CI) * 1.0
        esky_aniso = esky
        ci = weather.clearness_index
        if ci < 0.95:
            esky_aniso = ci * esky + (1 - ci) * 1.0

        # Call full Rust anisotropic sky function
        ani_sky_result = sky.anisotropic_sky(
            shmat,
            vegshmat,
            vbshmat,
            weather.sun_altitude,
            weather.sun_azimuth,
            asvf.astype(np.float32),
            bool(cyl),
            esky_aniso,
            l_patches,
            False,  # wallScheme
            None,   # voxelTable
            None,   # voxelMaps
            steradians.astype(np.float32),
            weather.ta,
            tg_wall,
            emis_wall,
            lup.astype(np.float32),  # TsWaveDelay-processed value (matching reference)
            rad_i,
            rad_d,
            rad_g,
            lv.astype(np.float32),
            albedo_wall,
            False,  # debug
            diffsh.astype(np.float32),
            shadow.astype(np.float32),
            kup_e.astype(np.float32),
            kup_s.astype(np.float32),
            kup_w.astype(np.float32),
            kup_n.astype(np.float32),
            0,  # iteration index
        )

        # Extract results from anisotropic sky
        ldown = np.array(ani_sky_result.ldown)
        lside = np.array(ani_sky_result.lside)
        # For directional longwave, use lside_veg_result (base) values
        # ani_sky_result provides anisotropic additions, but for cyl=1, aniso=1
        # the Sstr formula uses base directional longwave from lside_veg
        lside_e = lside_e_base
        lside_s = lside_s_base
        lside_w = lside_w_base
        lside_n = lside_n_base
        # Shortwave from anisotropic sky result
        kside_e = np.array(ani_sky_result.keast)
        kside_s = np.array(ani_sky_result.ksouth)
        kside_w = np.array(ani_sky_result.kwest)
        kside_n = np.array(ani_sky_result.knorth)
        kside_i = np.array(ani_sky_result.kside_i)
        kside_d = np.array(ani_sky_result.kside_d)
        kside = np.array(ani_sky_result.kside)

    else:
        # Isotropic model - use Rust functions for kside and lside

        # Isotropic diffuse radiation
        drad = rad_d * svfbuveg  # Diffuse weighted by combined SVF

        # Compute asvf for Rust functions (needed even for isotropic)
        asvf = np.arccos(np.sqrt(np.clip(svf, 0.0, 1.0)))

        # Use Rust kside_veg for directional shortwave (isotropic mode: no lv, no shadow matrices)
        kside_result = vegetation.kside_veg(
            rad_i,
            rad_d,
            rad_g,
            shadow.astype(np.float32),
            svf_s.astype(np.float32),
            svf_w.astype(np.float32),
            svf_n.astype(np.float32),
            svf_e.astype(np.float32),
            svf_veg_e.astype(np.float32),
            svf_veg_s.astype(np.float32),
            svf_veg_w.astype(np.float32),
            svf_veg_n.astype(np.float32),
            weather.sun_azimuth,
            weather.sun_altitude,
            psi,
            0.0,  # t (instrument offset)
            albedo_wall,
            f_sh.astype(np.float32),
            kup_e.astype(np.float32),
            kup_s.astype(np.float32),
            kup_w.astype(np.float32),
            kup_n.astype(np.float32),
            bool(cyl),
            None,  # lv (None for isotropic)
            False,  # anisotropic_sky
            None,  # diffsh (None for isotropic)
            asvf.astype(np.float32),
            None,  # shmat (None for isotropic)
            None,  # vegshmat (None for isotropic)
            None,  # vbshvegshmat (None for isotropic)
        )
        kside_e = np.array(kside_result.keast)
        kside_s = np.array(kside_result.ksouth)
        kside_w = np.array(kside_result.kwest)
        kside_n = np.array(kside_result.knorth)
        kside_i = np.array(kside_result.kside_i)

        # Longwave: Ldown (from Jonsson et al. 2006)
        ldown = (
            (svf + svf_veg - 1) * esky * SBC * (ta_k ** 4)
            + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k ** 4)
            + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + 273.15) ** 4)
            + (2 - svf - svf_veg) * (1 - emis_wall) * esky * SBC * (ta_k ** 4)
        )

        # CI correction for non-clear conditions (reference: if CI < 0.95)
        # Under cloudy skies, effective sky emissivity approaches 1.0
        ci = weather.clearness_index
        if ci < 0.95:
            c = 1.0 - ci
            ldown_cloudy = (
                (svf + svf_veg - 1) * SBC * (ta_k ** 4)  # No esky for cloudy
                + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k ** 4)
                + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + 273.15) ** 4)
                + (2 - svf - svf_veg) * (1 - emis_wall) * SBC * (ta_k ** 4)  # No esky
            )
            ldown = ldown * (1 - c) + ldown_cloudy * c

        # Use Rust lside_veg for directional longwave
        lside_veg_result = vegetation.lside_veg(
            svf_s.astype(np.float32),
            svf_w.astype(np.float32),
            svf_n.astype(np.float32),
            svf_e.astype(np.float32),
            svf_veg_e.astype(np.float32),
            svf_veg_s.astype(np.float32),
            svf_veg_w.astype(np.float32),
            svf_veg_n.astype(np.float32),
            svf_aveg_e.astype(np.float32),
            svf_aveg_s.astype(np.float32),
            svf_aveg_w.astype(np.float32),
            svf_aveg_n.astype(np.float32),
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            tg_wall,
            SBC,
            emis_wall,
            ldown.astype(np.float32),
            esky,
            0.0,  # t (instrument offset, matching reference)
            f_sh.astype(np.float32),
            weather.clearness_index,
            lup_e.astype(np.float32),  # TsWaveDelay-processed values (matching reference)
            lup_s.astype(np.float32),
            lup_w.astype(np.float32),
            lup_n.astype(np.float32),
            False,  # anisotropic_sky
        )
        lside_e = np.array(lside_veg_result.least)
        lside_s = np.array(lside_veg_result.lsouth)
        lside_w = np.array(lside_veg_result.lwest)
        lside_n = np.array(lside_veg_result.lnorth)

    # Kdown (downwelling shortwave = direct on horizontal + diffuse sky + wall reflected)
    # Note: Runner uses dRad (anisotropic or isotropic depending on use_aniso setting)
    # drad is already computed earlier: anisotropic at line 1622, isotropic at line 1759
    kdown = rad_i * shadow * sin_alt + drad + albedo_wall * (1 - svfbuveg) * (rad_g * (1 - f_sh) + rad_d * f_sh)

    # Note: TsWaveDelay for Lup is already applied earlier (section 5) to match reference implementation

    # 7. Compute Tmrt
    # Human body as standing cylinder (SOLWEIG standard model)
    # Direct beam absorbed via cylindrical projection (Fcyl)
    # Diffuse absorbed via view factors (Fup, Fside)

    if use_aniso:
        # Anisotropic model formula (cyl=1, aniso=1)
        k_absorbed = human.abs_k * (
            kside * f_cyl  # Anisotropic shortwave on vertical body surface
            + (kdown + kup) * f_up  # Downwelling + upwelling on top/bottom
            + (kside_n + kside_e + kside_s + kside_w) * f_side  # Directional from 4 sides
        )

        l_absorbed = human.abs_l * (
            (ldown + lup) * f_up
            + lside * f_cyl  # Anisotropic longwave on vertical surface
            + (lside_n + lside_e + lside_s + lside_w) * f_side
        )
    else:
        # Isotropic model: use only direct beam on vertical
        k_absorbed = human.abs_k * (
            kside_i * f_cyl  # Direct beam on vertical body surface
            + (kdown + kup) * f_up  # Downwelling + upwelling on top/bottom
            + (kside_n + kside_e + kside_s + kside_w) * f_side  # Diffuse from 4 sides
        )

        l_absorbed = human.abs_l * (
            (ldown + lup) * f_up
            + (lside_n + lside_e + lside_s + lside_w) * f_side
        )

    sstr = k_absorbed + l_absorbed

    tmrt = np.sqrt(np.sqrt(sstr / (human.abs_l * SBC))) - 273.2  # Match reference (historical convention)
    tmrt = np.clip(tmrt.astype(np.float32), -50, 80)

    # 7. Compute UTCI if requested
    utci_grid = None
    if compute_utci:
        wind_grid = np.full((rows, cols), weather.ws, dtype=np.float32)
        utci_grid = utci_rust.utci_grid(weather.ta, weather.rh, tmrt, wind_grid)

    # 8. Compute PET if requested
    pet_grid = None
    if compute_pet:
        wind_grid = np.full((rows, cols), weather.ws, dtype=np.float32)
        pet_grid = pet_rust.pet_grid(
            weather.ta,
            weather.rh,
            tmrt,
            wind_grid,
            human.weight,
            float(human.age),
            human.height,
            human.activity,
            human.clothing,
            human.sex,
        )

    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow,
        kdown=kdown.astype(np.float32),
        kup=kup.astype(np.float32),
        ldown=ldown.astype(np.float32),
        lup=lup.astype(np.float32),
        utci=utci_grid,
        pet=pet_grid,
        state=output_state,
    )


def calculate_timeseries(
    surface: SurfaceData,
    location: Location,
    weather_series: list[Weather],
    human: HumanParams | None = None,
    precomputed: PrecomputedData | None = None,
    use_anisotropic_sky: bool = False,
    compute_utci: bool = True,
    compute_pet: bool = False,
    params: SimpleNamespace | None = None,
) -> list[SolweigResult]:
    """
    Calculate Tmrt for a time series of weather data.

    Maintains thermal state across timesteps for accurate surface temperature
    modeling with thermal inertia (TsWaveDelay_2015a).

    This is a convenience function that manages state automatically. For custom
    control over state, use calculate() directly with the state parameter.

    Args:
        surface: Surface/terrain data (DSM required, CDSM/DEM optional).
        location: Geographic location (lat, lon, UTC offset).
        weather_series: List of Weather objects in chronological order.
            The datetime of each Weather object determines the timestep size.
        human: Human body parameters. Uses defaults if not provided.
        precomputed: Pre-computed SVF and/or shadow matrices. Optional.
        use_anisotropic_sky: Use anisotropic sky model. Default False.
        compute_utci: Whether to compute UTCI. Default True.
        compute_pet: Whether to compute PET. Default False.
        params: Loaded SOLWEIG parameters from JSON file (via load_params()).
            When provided, uses these parameters for land cover properties.
            When None, uses built-in defaults matching parametersforsolweig.json.

    Returns:
        List of SolweigResult objects, one per timestep.
        Each result includes the thermal state at that timestep.

    Example:
        # Create weather data for a day
        weather_list = [
            Weather(datetime=datetime(2024, 7, 15, h, 0), ta=20+5*h/12, rh=60, global_rad=max(0, 800*sin((h-6)*pi/12)))
            for h in range(6, 20)  # 6am to 8pm
        ]

        # Run time series
        results = calculate_timeseries(
            surface=SurfaceData(dsm=my_dsm),
            location=Location(latitude=57.7, longitude=12.0),
            weather_series=weather_list,
        )

        # Access results
        for i, result in enumerate(results):
            print(f"Hour {i}: mean Tmrt = {result.tmrt.mean():.1f}°C")
    """
    if not weather_series:
        return []

    results = []
    state = ThermalState.initial(surface.shape)

    # Pre-calculate timestep size from first two entries (matching runner behavior)
    # The runner uses a fixed timestep_dec for all iterations, calculated upfront
    if len(weather_series) >= 2:
        dt0 = weather_series[0].datetime
        dt1 = weather_series[1].datetime
        state.timestep_dec = (dt1 - dt0).total_seconds() / 86400.0

    for i, weather in enumerate(weather_series):
        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=human,
            precomputed=precomputed,
            use_anisotropic_sky=use_anisotropic_sky,
            compute_utci=compute_utci,
            compute_pet=compute_pet,
            state=state,
            params=params,
        )

        # Carry forward state to next timestep
        if result.state is not None:
            state = result.state

        results.append(result)

    return results


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
        warning = (
            f"Tile size increased to {adjusted} to ensure meaningful core area "
            f"with {buffer_m:.0f}m buffer"
        )

    return adjusted, warning


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
    tile_size: int = 1024,
    use_anisotropic_sky: bool = False,
    compute_utci: bool = True,
    compute_pet: bool = False,
    params: SimpleNamespace | None = None,
    progress_callback: callable | None = None,
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
        tile_size: Core tile size in pixels (default 1024). Actual size may be
            adjusted to ensure meaningful core area after buffer overlap.
        use_anisotropic_sky: Use anisotropic sky model. Default False.
            Note: Anisotropic sky is not yet supported in tiled mode.
        compute_utci: Whether to compute UTCI. Default True.
        compute_pet: Whether to compute PET. Default False.
        params: Loaded SOLWEIG parameters from JSON file (via load_params()).
        progress_callback: Optional callback(tile_idx, total_tiles) for progress.

    Returns:
        SolweigResult with Tmrt and optionally UTCI/PET grids.
        Note: state is not returned for tiled mode (use calculate_timeseries_tiled
        for multi-timestep with state).

    Raises:
        NotImplementedError: If use_anisotropic_sky=True (not yet supported).
        ValueError: If tile_size is invalid.

    Example:
        # Large raster processing
        result = calculate_tiled(
            surface=SurfaceData(dsm=large_dsm_array),
            location=Location(latitude=57.7, longitude=12.0),
            weather=Weather(datetime=dt, ta=25, rh=50, global_rad=800),
            tile_size=1024,  # 1024x1024 core tiles
        )
    """
    import logging

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
        logger.info(
            f"Raster {rows}x{cols} fits in single tile, using non-tiled calculation"
        )
        return calculate(
            surface=surface,
            location=location,
            weather=weather,
            human=human,
            use_anisotropic_sky=use_anisotropic_sky,
            compute_utci=compute_utci,
            compute_pet=compute_pet,
            params=params,
        )

    # Generate tiles
    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)

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
    utci_out = np.full((rows, cols), np.nan, dtype=np.float32) if compute_utci else None
    pet_out = np.full((rows, cols), np.nan, dtype=np.float32) if compute_pet else None

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

        tile_wall_ht = None
        if surface.wall_height is not None:
            tile_wall_ht = surface.wall_height[read_slice].copy()

        tile_wall_asp = None
        if surface.wall_aspect is not None:
            tile_wall_asp = surface.wall_aspect[read_slice].copy()

        tile_lc = None
        if surface.land_cover is not None:
            tile_lc = surface.land_cover[read_slice].copy()

        tile_albedo = None
        if surface.albedo is not None:
            tile_albedo = surface.albedo[read_slice].copy()

        tile_emis = None
        if surface.emissivity is not None:
            tile_emis = surface.emissivity[read_slice].copy()

        # Create tile surface
        tile_surface = SurfaceData(
            dsm=tile_dsm,
            cdsm=tile_cdsm,
            tdsm=tile_tdsm,
            dem=tile_dem,
            wall_height=tile_wall_ht,
            wall_aspect=tile_wall_asp,
            land_cover=tile_lc,
            albedo=tile_albedo,
            emissivity=tile_emis,
            pixel_size=pixel_size,
        )

        # Calculate for tile (without precomputed SVF/shadows - computed per-tile)
        tile_result = calculate(
            surface=tile_surface,
            location=location,
            weather=weather,
            human=human,
            precomputed=None,  # Compute SVF per-tile
            use_anisotropic_sky=False,
            compute_utci=compute_utci,
            compute_pet=compute_pet,
            state=None,  # No state for single-timestep tiled
            params=params,
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
        if compute_utci and tile_result.utci is not None:
            utci_out[write_slice] = tile_result.utci[core_slice]
        if compute_pet and tile_result.pet is not None:
            pet_out[write_slice] = tile_result.pet[core_slice]

    if progress_callback:
        progress_callback(n_tiles, n_tiles)

    return SolweigResult(
        tmrt=tmrt_out,
        shadow=shadow_out,
        kdown=kdown_out,
        kup=kup_out,
        ldown=ldown_out,
        lup=lup_out,
        utci=utci_out,
        pet=pet_out,
        state=None,  # No state for tiled mode
    )
