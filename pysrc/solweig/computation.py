"""
Orchestration layer for SOLWEIG core calculation.

This module coordinates all computation components in a clean, linear flow:
SVF resolution → Shadows → Ground temp → GVF → Thermal delay → Radiation → Tmrt

Replaces the monolithic 841-line `_calculate_core()` with focused orchestration.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from .bundles import LupBundle
from .components.ground import compute_ground_temperature
from .components.gvf import compute_gvf
from .components.radiation import compute_radiation
from .components.shadows import compute_shadows
from .components.svf_resolution import resolve_svf
from .components.tmrt import compute_tmrt
from .constants import KELVIN_OFFSET, SBC
from .rustalgos import ground as ground_rust

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .api import HumanParams, Location, PrecomputedData, SolweigResult, SurfaceData, ThermalState, Weather
    from .bundles import GvfBundle


def _nighttime_result(
    surface: SurfaceData,
    weather: Weather,
    state: ThermalState | None,
    materials: SimpleNamespace | None,
) -> SolweigResult:
    """
    Compute simplified nighttime result when sun is below horizon.

    At night:
    - Tmrt ≈ Ta (no solar radiation)
    - Shadow = 0 (everything shaded)
    - Kdown = Kup = 0 (no shortwave)
    - Ldown/Lup from atmospheric and surface emission only

    Args:
        surface: Surface data (for emissivity grid)
        weather: Weather data (for air temperature)
        state: Optional thermal state (resets for morning)
        materials: Material properties (for emissivity)

    Returns:
        SolweigResult with nighttime values
    """
    # Import here to avoid circular dependency
    from .api import SolweigResult

    rows, cols = surface.dsm.shape

    # Get emissivity grid for nighttime longwave
    _, emis_grid, _, _, _ = surface.get_land_cover_properties(materials)

    # Nighttime: Tmrt ≈ Ta (simplified, no solar heating)
    tmrt = np.full((rows, cols), weather.ta, dtype=np.float32)
    shadow = np.zeros((rows, cols), dtype=np.float32)  # 0 = shaded (night)

    # Nighttime longwave: Lup = SBC × emis × Ta⁴
    ta_k = weather.ta + KELVIN_OFFSET
    lup_night = SBC * emis_grid * np.power(ta_k, 4)
    # Ldown from sky with typical nighttime emissivity ~0.95
    ldown_night = np.full((rows, cols), SBC * 0.95 * np.power(ta_k, 4), dtype=np.float32)

    # Update thermal state for nighttime (reset for next morning)
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
        utci=None,
        pet=None,
        state=output_state,
    )


def _apply_thermal_delay(
    gvf_bundle: GvfBundle,
    ground_tg: NDArray[np.floating],
    shadow: NDArray[np.floating],
    weather: Weather,
    state: ThermalState | None,
) -> LupBundle:
    """
    Apply thermal inertia (TsWaveDelay) to upwelling longwave radiation.

    This models the thermal mass of ground and walls, smoothing rapid temperature
    changes throughout the day. Essential for accurate time-series simulations.

    Uses batched Rust function to reduce FFI overhead from 6 calls to 1.

    Args:
        gvf_bundle: Ground view factor results (raw Lup before delay)
        ground_tg: Ground temperature deviation from air temperature (K)
        shadow: Shadow fraction (for ground temperature with shadow effect)
        weather: Weather data (for air temperature and daytime flag)
        state: Thermal state carrying forward surface temperature history

    Returns:
        LupBundle with thermally-delayed upwelling longwave and updated state
    """
    from .buffers import as_float32

    output_state = None

    if state is not None:
        # Compute ground temperature with shadow effect
        tg_temp = (ground_tg * shadow + weather.ta).astype(np.float32)

        # Apply TsWaveDelay for thermal mass effect (batched - 6 calls → 1)
        firstdaytime_int = int(state.firstdaytime)
        result = ground_rust.ts_wave_delay_batch(
            as_float32(gvf_bundle.lup),
            as_float32(gvf_bundle.lup_e),
            as_float32(gvf_bundle.lup_s),
            as_float32(gvf_bundle.lup_w),
            as_float32(gvf_bundle.lup_n),
            tg_temp,
            firstdaytime_int,
            state.timeadd,
            state.timestep_dec,
            as_float32(state.tgmap1),
            as_float32(state.tgmap1_e),
            as_float32(state.tgmap1_s),
            as_float32(state.tgmap1_w),
            as_float32(state.tgmap1_n),
            as_float32(state.tgout1),
        )

        # Extract delayed outputs
        lup = np.asarray(result.lup)
        lup_e = np.asarray(result.lup_e)
        lup_s = np.asarray(result.lup_s)
        lup_w = np.asarray(result.lup_w)
        lup_n = np.asarray(result.lup_n)

        # Update state with new values
        state.timeadd = result.timeadd
        state.tgmap1 = np.asarray(result.tgmap1)
        state.tgmap1_e = np.asarray(result.tgmap1_e)
        state.tgmap1_s = np.asarray(result.tgmap1_s)
        state.tgmap1_w = np.asarray(result.tgmap1_w)
        state.tgmap1_n = np.asarray(result.tgmap1_n)
        state.tgout1 = np.asarray(result.tgout1)

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
        lup = gvf_bundle.lup
        lup_e = gvf_bundle.lup_e
        lup_s = gvf_bundle.lup_s
        lup_w = gvf_bundle.lup_w
        lup_n = gvf_bundle.lup_n

    return LupBundle(
        lup=lup.astype(np.float32),
        lup_e=lup_e.astype(np.float32),
        lup_s=lup_s.astype(np.float32),
        lup_w=lup_w.astype(np.float32),
        lup_n=lup_n.astype(np.float32),
        state=output_state,
    )


def calculate_core(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    use_anisotropic_sky: bool,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    conifer: bool = False,
    wall_material: str | None = None,
) -> SolweigResult:
    """
    Core SOLWEIG calculation orchestrating all components.

    This is the clean orchestration layer that wires together all extracted
    components in a linear flow:

    1. Early exit for nighttime (sun below horizon)
    2. SVF resolution (from cache/precomputed or fresh computation)
    3. Shadow computation (buildings + vegetation with transmissivity)
    4. Ground temperature model (TgMaps with land cover parameterization)
    5. Ground View Factor (upwelling radiation from surfaces)
    6. Thermal delay (TsWaveDelay for thermal inertia)
    7. Radiation calculation (shortwave + longwave from all directions)
    8. Tmrt calculation (absorbed radiation → mean radiant temperature)

    Args:
        surface: Surface/terrain data (DSM, vegetation, walls, land cover)
        location: Geographic location (latitude, longitude)
        weather: Weather conditions (temperature, humidity, radiation, sun position)
        human: Human parameters (height, posture, absorptivities)
        precomputed: Optional pre-computed data (SVF, shadow matrices)
        use_anisotropic_sky: Use anisotropic (Perez) diffuse sky model
        state: Optional thermal state for time-series (carries forward temperatures)
        physics: Optional physics parameters (vegetation transmissivity, etc.)
        materials: Optional material properties (albedo, emissivity by land cover)
        conifer: Treat vegetation as evergreen conifers (always leaf-on)

    Returns:
        SolweigResult with Tmrt, shadow, radiation components, and updated state

    Reference:
        Lindberg et al. (2008, 2016) - SOLWEIG model description
    """
    # Import here to avoid circular dependency
    from .api import SolweigResult

    # Early exit for nighttime
    if weather.sun_altitude <= 0:
        return _nighttime_result(surface, weather, state, materials)

    # === Daytime calculation ===

    # Extract grid dimensions
    rows, cols = surface.dsm.shape
    pixel_size = surface.pixel_size

    # Get surface properties: albedo, emissivity, and ground temp parameters
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = surface.get_land_cover_properties(materials)

    # Prepare vegetation inputs
    use_veg = surface.cdsm is not None
    cdsm = surface.cdsm if use_veg else None
    tdsm = surface.tdsm if use_veg else None
    # Bush layer: Create empty mask when vegetation present (no bushes assumed)
    # Rust validation requires all three vegetation inputs (cdsm, tdsm, bush) or none
    if use_veg:
        pool = surface.get_buffer_pool()
        bush = pool.get_zeros("bush")
    else:
        bush = None

    # Prepare wall inputs
    has_walls = surface.wall_height is not None and surface.wall_aspect is not None
    wall_ht = surface.wall_height if has_walls else None
    wall_asp = surface.wall_aspect if has_walls else None

    # Maximum building height for shadow/SVF computation
    max_height = float(np.nanmax(surface.dsm)) if surface.dsm.size > 0 else 50.0

    # Step 1: SVF Resolution (sky view factors from all sources)
    svf_bundle, needs_psi_adjustment = resolve_svf(
        surface=surface,
        precomputed=precomputed,
        dsm=surface.dsm,
        cdsm=cdsm,
        tdsm=tdsm,
        pixel_size=pixel_size,
        use_veg=use_veg,
        max_height=max_height,
        psi=None,  # Will be computed in shadows step
    )

    # Step 2: Shadow Computation (with vegetation transmissivity)
    shadow_bundle = compute_shadows(
        weather=weather,
        dsm=surface.dsm,
        pixel_size=pixel_size,
        max_height=max_height,
        use_veg=use_veg,
        physics=physics,
        conifer=conifer,
        cdsm=cdsm,
        tdsm=tdsm,
        bush=bush,
        wall_ht=wall_ht,
        wall_asp_rad=wall_asp * (np.pi / 180.0) if wall_asp is not None else None,
    )

    # Adjust svfbuveg with psi if needed (when SVF was computed fresh without psi)
    if needs_psi_adjustment and use_veg:
        from .components.svf_resolution import adjust_svfbuveg_with_psi

        svf_bundle.svfbuveg = adjust_svfbuveg_with_psi(
            svf=svf_bundle.svf,
            svf_veg=svf_bundle.svf_veg,
            psi=shadow_bundle.psi,
            use_veg=use_veg,
        )

    # Cache fresh-computed SVF back to surface for subsequent calculate() calls
    # This avoids re-computing SVF on every timestep
    if needs_psi_adjustment and surface.svf is None:
        from .models.precomputed import SvfArrays

        surface.svf = SvfArrays.from_bundle(svf_bundle)

    # Step 3: Ground Temperature Model (TgMaps with land cover parameterization)
    # Resolve wall params: explicit wall_material wins, then materials JSON, then Rust defaults
    tgk_wall = None
    tstart_wall = None
    tmaxlst_wall = None
    if wall_material is not None:
        from .loaders import resolve_wall_params

        tgk_wall, tstart_wall, tmaxlst_wall = resolve_wall_params(wall_material, materials)
    elif materials is not None:
        tgk_wall = getattr(getattr(getattr(materials, "Ts_deg", None), "Value", None), "Walls", None)
        tstart_wall = getattr(getattr(getattr(materials, "Tstart", None), "Value", None), "Walls", None)
        tmaxlst_wall = getattr(getattr(getattr(materials, "TmaxLST", None), "Value", None), "Walls", None)

    ground_bundle = compute_ground_temperature(
        weather=weather,
        location=location,
        alb_grid=alb_grid,
        emis_grid=emis_grid,
        tgk_grid=tgk_grid,
        tstart_grid=tstart_grid,
        tmaxlst_grid=tmaxlst_grid,
        tgk_wall=tgk_wall,
        tstart_wall=tstart_wall,
        tmaxlst_wall=tmaxlst_wall,
    )

    # Step 4: Ground View Factor (upwelling radiation from ground + walls)
    gvf_bundle = compute_gvf(
        surface=surface,
        weather=weather,
        human=human,
        tg=ground_bundle.tg,
        tg_wall=ground_bundle.tg_wall,
        shadow=shadow_bundle.shadow,
        wallsun=shadow_bundle.wallsun,
        alb_grid=ground_bundle.alb_grid,
        emis_grid=ground_bundle.emis_grid,
        svf=svf_bundle.svf,
        pixel_size=pixel_size,
        wall_ht=wall_ht,
        wall_asp=wall_asp,
    )

    # Step 5: Apply Thermal Delay (TsWaveDelay for thermal inertia)
    lup_bundle = _apply_thermal_delay(
        gvf_bundle=gvf_bundle,
        ground_tg=ground_bundle.tg,
        shadow=shadow_bundle.shadow,
        weather=weather,
        state=state,
    )

    # Step 6: Radiation Calculation (complete shortwave + longwave budget)
    radiation_bundle = compute_radiation(
        weather=weather,
        svf_bundle=svf_bundle,
        shadow_bundle=shadow_bundle,
        gvf_bundle=gvf_bundle,
        lup_bundle=lup_bundle,
        human=human,
        use_anisotropic_sky=use_anisotropic_sky,
        precomputed=precomputed,
        albedo_wall=0.20,  # Default cobblestone
        emis_wall=0.90,  # Default brick/concrete
        tg_wall=ground_bundle.tg_wall,
    )

    # Step 7: Tmrt Calculation (absorbed radiation → mean radiant temperature)
    tmrt = compute_tmrt(
        radiation=radiation_bundle,
        human=human,
        use_anisotropic_sky=use_anisotropic_sky,
    )

    return SolweigResult(
        tmrt=tmrt,
        shadow=shadow_bundle.shadow,
        kdown=radiation_bundle.kdown,
        kup=radiation_bundle.kup,
        ldown=radiation_bundle.ldown,
        lup=lup_bundle.lup,
        utci=None,  # Computed separately via post-processing
        pet=None,  # Computed separately via post-processing
        state=lup_bundle.state,
    )
