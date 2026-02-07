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

    # Step 1: SVF Resolution (sky view factors from cached/precomputed sources)
    svf_bundle, _needs_psi_adjustment = resolve_svf(
        surface=surface,
        precomputed=precomputed,
        dsm=surface.dsm,
        cdsm=cdsm,
        tdsm=tdsm,
        pixel_size=pixel_size,
        use_veg=use_veg,
        max_height=max_height,
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


def calculate_core_fused(
    surface: SurfaceData,
    location: Location,
    weather: Weather,
    human: HumanParams,
    precomputed: PrecomputedData | None,
    state: ThermalState | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    conifer: bool = False,
    wall_material: str | None = None,
    use_anisotropic_sky: bool = False,
) -> SolweigResult:
    """
    Fused SOLWEIG calculation — single Rust FFI call per daytime timestep.

    Functionally identical to calculate_core() but orchestrates shadows → ground →
    GVF → thermal delay → radiation → Tmrt entirely within Rust, eliminating
    intermediate numpy allocations and FFI round-trips.

    Supports both isotropic and anisotropic (Perez) sky models.

    Args:
        Same as calculate_core().
    """
    from .api import SolweigResult
    from .buffers import as_float32
    from .components.gvf import detect_building_mask
    from .components.shadows import compute_transmissivity
    from .components.svf_resolution import resolve_svf
    from .models.state import ThermalState
    from .physics.clearnessindex_2013b import clearnessindex_2013b
    from .physics.daylen import daylen
    from .physics.diffusefraction import diffusefraction
    from .rustalgos import pipeline

    # Ensure derived weather fields are computed (sun position, radiation split)
    if not weather._derived_computed:
        weather.compute_derived(location)

    # Early exit for nighttime
    if weather.sun_altitude <= 0:
        return _nighttime_result(surface, weather, state, materials)

    # === Precompute (stays in Python) ===

    rows, cols = surface.dsm.shape
    pixel_size = surface.pixel_size

    # Land cover properties
    alb_grid, emis_grid, tgk_grid, tstart_grid, tmaxlst_grid = surface.get_land_cover_properties(materials)

    # Vegetation inputs
    use_veg = surface.cdsm is not None
    cdsm = surface.cdsm if use_veg else None
    tdsm = surface.tdsm if use_veg else None
    if use_veg:
        pool = surface.get_buffer_pool()
        bush = pool.get_zeros("bush")
    else:
        bush = None

    # Wall inputs
    has_walls = surface.wall_height is not None and surface.wall_aspect is not None
    wall_ht = surface.wall_height if has_walls else None
    wall_asp = surface.wall_aspect if has_walls else None

    max_height = float(np.nanmax(surface.dsm)) if surface.dsm.size > 0 else 50.0

    # SVF resolution (cached between timesteps)
    svf_bundle, _needs_psi_adjustment = resolve_svf(
        surface=surface,
        precomputed=precomputed,
        dsm=surface.dsm,
        cdsm=cdsm,
        tdsm=tdsm,
        pixel_size=pixel_size,
        use_veg=use_veg,
        max_height=max_height,
    )

    # Vegetation transmissivity
    doy = weather.datetime.timetuple().tm_yday
    psi = compute_transmissivity(doy, physics, conifer)

    # Wall material resolution
    tgk_wall = 0.37
    tstart_wall = -3.41
    tmaxlst_wall = 15.0
    albedo_wall = 0.20
    emis_wall = 0.90
    if wall_material is not None:
        from .loaders import resolve_wall_params

        tgk_wall, tstart_wall, tmaxlst_wall = resolve_wall_params(wall_material, materials)
    elif materials is not None:
        tgk_w = getattr(getattr(getattr(materials, "Ts_deg", None), "Value", None), "Walls", None)
        tstart_w = getattr(getattr(getattr(materials, "Tstart", None), "Value", None), "Walls", None)
        tmaxlst_w = getattr(getattr(getattr(materials, "TmaxLST", None), "Value", None), "Walls", None)
        if tgk_w is not None:
            tgk_wall = tgk_w
        if tstart_w is not None:
            tstart_wall = tstart_w
        if tmaxlst_w is not None:
            tmaxlst_wall = tmaxlst_w

    # Weather-derived scalars for ground temperature model
    _, _, _, snup = daylen(doy, location.latitude)
    dectime = (weather.datetime.hour + weather.datetime.minute / 60.0) / 24.0
    zen_deg = 90.0 - weather.sun_altitude

    # Clear-sky radiation for ground temperature CI correction
    zen_rad = zen_deg * (np.pi / 180.0)
    location_dict = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "altitude": 0.0,
    }
    i0, _, _, _, _ = clearnessindex_2013b(
        zen_rad,
        doy,
        weather.ta,
        weather.rh / 100.0,
        weather.global_rad,
        location_dict,
        -999.0,
    )
    if i0 > 0 and weather.sun_altitude > 0:
        rad_i0, rad_d0 = diffusefraction(i0, weather.sun_altitude, 1.0, weather.ta, weather.rh)
        rad_g0 = rad_i0 * np.sin(weather.sun_altitude * np.pi / 180.0) + rad_d0
    else:
        rad_g0 = 0.0

    # === Build Rust input structs ===

    ws = pipeline.WeatherScalars(
        sun_azimuth=float(weather.sun_azimuth),
        sun_altitude=float(weather.sun_altitude),
        sun_zenith=float(weather.sun_zenith),
        ta=float(weather.ta),
        rh=float(weather.rh),
        global_rad=float(weather.global_rad),
        direct_rad=float(weather.direct_rad),
        diffuse_rad=float(weather.diffuse_rad),
        altmax=float(weather.altmax),
        clearness_index=float(weather.clearness_index),
        dectime=float(dectime),
        snup=float(snup),
        rad_g0=float(rad_g0),
        zen_deg=float(zen_deg),
        psi=float(psi),
        is_daytime=weather.sun_altitude > 0,
    )

    hs = pipeline.HumanScalars(
        height=float(human.height),
        abs_k=float(human.abs_k),
        abs_l=float(human.abs_l),
        is_standing=human.posture == "standing",
    )

    cs = pipeline.ConfigScalars(
        pixel_size=float(pixel_size),
        max_height=float(max_height),
        albedo_wall=float(albedo_wall),
        emis_wall=float(emis_wall),
        tgk_wall=float(tgk_wall),
        tstart_wall=float(tstart_wall),
        tmaxlst_wall=float(tmaxlst_wall),
        use_veg=use_veg,
        has_walls=has_walls,
        conifer=conifer,
        use_anisotropic=use_anisotropic_sky,
    )

    # Buildings mask for GVF (computed from DSM/land_cover/walls)
    buildings = detect_building_mask(
        surface.dsm,
        surface.land_cover,
        wall_ht,
        pixel_size,
    )
    lc_grid = surface.land_cover.astype(np.float32) if surface.land_cover is not None else None

    # GVF geometry cache: precompute on first daytime call, reuse on subsequent
    gvf_cache = getattr(surface, "_gvf_geometry_cache", None)
    if gvf_cache is None and has_walls:
        assert wall_asp is not None  # guaranteed by has_walls
        assert wall_ht is not None
        gvf_cache = pipeline.precompute_gvf_cache(
            as_float32(buildings),
            as_float32(wall_asp),
            as_float32(wall_ht),
            as_float32(alb_grid),
            float(pixel_size),
            float(human.height),
            float(albedo_wall),
        )
        surface._gvf_geometry_cache = gvf_cache

    # Anisotropic sky pre-computation (Perez stays in Python, <1ms)
    aniso_shmat = None
    aniso_vegshmat = None
    aniso_vbshmat = None
    aniso_l_patches = None
    aniso_steradians = None
    aniso_lv = None
    aniso_asvf = None
    aniso_esky = None

    if use_anisotropic_sky:
        from .physics.patch_radiation import patch_steradians
        from .physics.Perez_v3 import Perez_v3

        # Get shadow matrices
        shadow_mats = None
        if precomputed is not None and precomputed.shadow_matrices is not None:
            shadow_mats = precomputed.shadow_matrices
        elif surface.shadow_matrices is not None:
            shadow_mats = surface.shadow_matrices

        if shadow_mats is not None:
            patch_option = shadow_mats.patch_option
            jday = weather.datetime.timetuple().tm_yday
            rad_d = float(weather.diffuse_rad)
            rad_i = float(weather.direct_rad)

            # Perez luminance distribution
            lv_arr, _, _ = Perez_v3(
                weather.sun_zenith,
                weather.sun_azimuth,
                rad_d,
                rad_i,
                jday,
                patchchoice=1,
                patch_option=patch_option,
            )

            # Steradians
            ster, _, _ = patch_steradians(lv_arr)

            # ASVF from SVF
            asvf_arr = np.arccos(np.sqrt(np.clip(svf_bundle.svf, 0.0, 1.0)))

            # Esky (Jonsson et al. 2006) with CI correction for anisotropic
            ta_k = weather.ta + 273.15
            ea = 6.107 * 10 ** ((7.5 * weather.ta) / (237.3 + weather.ta)) * (weather.rh / 100.0)
            msteg = 46.5 * (ea / ta_k)
            esky_val = 1 - (1 + msteg) * np.exp(-np.sqrt(1.2 + 3.0 * msteg))
            ci = weather.clearness_index
            if ci < 0.95:
                esky_val = ci * esky_val + (1 - ci) * 1.0

            aniso_shmat = np.ascontiguousarray(shadow_mats._shmat_u8)
            aniso_vegshmat = np.ascontiguousarray(shadow_mats._vegshmat_u8)
            aniso_vbshmat = np.ascontiguousarray(shadow_mats._vbshmat_u8)
            aniso_l_patches = as_float32(lv_arr)
            aniso_steradians = as_float32(ster)
            aniso_lv = as_float32(lv_arr)
            aniso_asvf = as_float32(asvf_arr)
            aniso_esky = float(esky_val)

    # Thermal state (create initial if None)
    if state is None:
        state = ThermalState.initial((rows, cols))

    firstdaytime_int = int(state.firstdaytime)

    # === Call fused Rust pipeline ===

    result = pipeline.compute_timestep(
        # Scalar structs
        ws,
        hs,
        cs,
        # GVF geometry cache (None on first call triggers full GVF, then cached)
        gvf_cache,
        # Surface arrays
        as_float32(surface.dsm),
        as_float32(cdsm) if cdsm is not None else None,
        as_float32(tdsm) if tdsm is not None else None,
        as_float32(bush) if bush is not None else None,
        as_float32(wall_ht) if wall_ht is not None else None,
        as_float32(wall_asp) if wall_asp is not None else None,
        # SVF arrays
        as_float32(svf_bundle.svf),
        as_float32(svf_bundle.svf_directional.north),
        as_float32(svf_bundle.svf_directional.east),
        as_float32(svf_bundle.svf_directional.south),
        as_float32(svf_bundle.svf_directional.west),
        as_float32(svf_bundle.svf_veg),
        as_float32(svf_bundle.svf_veg_directional.north),
        as_float32(svf_bundle.svf_veg_directional.east),
        as_float32(svf_bundle.svf_veg_directional.south),
        as_float32(svf_bundle.svf_veg_directional.west),
        as_float32(svf_bundle.svf_aveg),
        as_float32(svf_bundle.svf_aveg_directional.north),
        as_float32(svf_bundle.svf_aveg_directional.east),
        as_float32(svf_bundle.svf_aveg_directional.south),
        as_float32(svf_bundle.svf_aveg_directional.west),
        as_float32(svf_bundle.svfbuveg),
        as_float32(svf_bundle.svfalfa),
        # Land cover property grids
        as_float32(alb_grid),
        as_float32(emis_grid),
        as_float32(tgk_grid),
        as_float32(tstart_grid),
        as_float32(tmaxlst_grid),
        # Buildings mask + land cover
        as_float32(buildings),
        as_float32(lc_grid) if lc_grid is not None else None,
        # Anisotropic sky inputs (None for isotropic)
        aniso_shmat,
        aniso_vegshmat,
        aniso_vbshmat,
        aniso_l_patches,
        aniso_steradians,
        aniso_lv,
        aniso_asvf,
        aniso_esky,
        # Thermal state
        firstdaytime_int,
        float(state.timeadd),
        float(state.timestep_dec),
        as_float32(state.tgmap1),
        as_float32(state.tgmap1_e),
        as_float32(state.tgmap1_s),
        as_float32(state.tgmap1_w),
        as_float32(state.tgmap1_n),
        as_float32(state.tgout1),
    )

    # === Unpack result and update thermal state ===

    state.timeadd = result.timeadd
    state.tgmap1 = np.asarray(result.tgmap1)
    state.tgmap1_e = np.asarray(result.tgmap1_e)
    state.tgmap1_s = np.asarray(result.tgmap1_s)
    state.tgmap1_w = np.asarray(result.tgmap1_w)
    state.tgmap1_n = np.asarray(result.tgmap1_n)
    state.tgout1 = np.asarray(result.tgout1)

    if weather.is_daytime:
        state.firstdaytime = 0.0
    else:
        state.firstdaytime = 1.0
        state.timeadd = 0.0

    output_state = state.copy()

    return SolweigResult(
        tmrt=np.asarray(result.tmrt),
        shadow=np.asarray(result.shadow),
        kdown=np.asarray(result.kdown),
        kup=np.asarray(result.kup),
        ldown=np.asarray(result.ldown),
        lup=np.asarray(result.lup),
        utci=None,
        pet=None,
        state=output_state,
    )
