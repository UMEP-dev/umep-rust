"""Configuration and parameter loading from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .utils import dict_to_namespace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_params(params_json_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load SOLWEIG parameters from a JSON file.

    Args:
        params_json_path: Path to the parameters JSON file.
            If None (default), loads bundled default_params.json with standard values
            for Tmrt, PET, vegetation, and posture parameters.
            For landcover-specific parameters (albedo, emissivity per surface type),
            provide a custom parameters file.

    Returns:
        SimpleNamespace object with nested parameter values accessible via attributes.

    Examples:
        Load bundled defaults:

        >>> params = load_params()  # Uses bundled default_params.json
        >>> params.Tmrt_params.Value.absK  # 0.7
        >>> params.Tree_settings.Value.Transmissivity  # 0.03

        Load custom parameters:

        >>> params = load_params("parametersforsolweig.json")
        >>> params.Albedo.Effective.Value.Dark_asphalt  # 0.18
    """
    if params_json_path is None:
        # Use bundled default parameters
        params_path = Path(__file__).parent / "data" / "default_params.json"
    else:
        params_path = Path(params_json_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path) as f:
        params_dict = json.load(f)

    return dict_to_namespace(params_dict)


def load_physics(physics_json_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load physics parameters (site-independent scientific constants).

    Physics parameters include:
    - Tree_settings: Vegetation transmissivity, seasonal dates, trunk ratio
    - Posture: Human posture geometry (Standing/Sitting projected area fractions)

    These are universal constants that rarely need customization.

    Args:
        physics_json_path: Path to a custom physics JSON file.
            If None (default), loads bundled physics_defaults.json with standard values.

    Returns:
        SimpleNamespace object with physics parameters accessible via attributes.

    Examples:
        Load bundled defaults:

        >>> physics = load_physics()  # Uses bundled physics_defaults.json
        >>> physics.Tree_settings.Value.Transmissivity  # 0.03
        >>> physics.Posture.Standing.Value.Fside  # 0.22

        Load custom physics (e.g., different tree transmissivity):

        >>> physics = load_physics("custom_trees.json")
    """
    if physics_json_path is None:
        # Use bundled physics defaults
        physics_path = Path(__file__).parent / "data" / "physics_defaults.json"
    else:
        physics_path = Path(physics_json_path)

    if not physics_path.exists():
        raise FileNotFoundError(f"Physics parameters file not found: {physics_path}")

    with open(physics_path) as f:
        physics_dict = json.load(f)

    return dict_to_namespace(physics_dict)


def load_materials(materials_json_path: str | Path) -> SimpleNamespace:
    """
    Load material properties (site-specific landcover parameters).

    Material properties include per-landcover-class values for:
    - Names: Landcover class names (e.g., "Dark_asphalt", "Grass_unmanaged")
    - Code: Landcover class IDs
    - Albedo: Surface albedo per class
    - Emissivity: Surface emissivity per class
    - TmaxLST, Ts_deg, Tstart: Ground temperature model parameters per class
    - Specific_heat, Thermal_conductivity, Density, Wall_thickness: Wall thermal properties

    These are site-specific and require a landcover grid (land_cover input).

    Args:
        materials_json_path: Path to a materials JSON file.
            This file must contain landcover-specific property definitions.

    Returns:
        SimpleNamespace object with material parameters accessible via attributes.

    Examples:
        Load site-specific materials:

        >>> materials = load_materials("site_materials.json")
        >>> materials.Albedo.Effective.Value.Dark_asphalt  # 0.18
        >>> materials.Emissivity.Value.Grass_unmanaged  # 0.94

    Notes:
        Materials are ONLY used when a landcover grid is provided to SurfaceData.
        If no landcover grid, uniform default properties are used.
    """
    materials_path = Path(materials_json_path)

    if not materials_path.exists():
        raise FileNotFoundError(f"Materials file not found: {materials_path}")

    with open(materials_path) as f:
        materials_dict = json.load(f)

    return dict_to_namespace(materials_dict)


def get_lc_properties_from_params(
    land_cover: NDArray[np.integer],
    params: SimpleNamespace,
    shape: tuple[int, int],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
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
