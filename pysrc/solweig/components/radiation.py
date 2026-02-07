"""
Radiation calculation component.

Computes complete radiation budget:
- Shortwave: direct beam, diffuse sky, ground reflection, wall reflection
- Longwave: sky emission, ground emission, wall emission
- Directional components (N, E, S, W) for human body sides

Supports both isotropic and anisotropic (Perez et al. 1993) diffuse sky models.

Reference:
- Lindberg et al. (2008, 2016) - SOLWEIG radiation model
- Perez et al. (1993) - Anisotropic sky luminance distribution
- Jonsson et al. (2006) - Longwave radiation formulas
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..buffers import as_float32
from ..bundles import DirectionalArrays, RadiationBundle
from ..constants import F_SIDE_SITTING, F_SIDE_STANDING, F_UP_SITTING, F_UP_STANDING, KELVIN_OFFSET, SBC
from ..physics.Kup_veg_2015a import Kup_veg_2015a
from ..physics.patch_radiation import patch_steradians
from ..physics.Perez_v3 import Perez_v3
from ..rustalgos import sky as _sky

if TYPE_CHECKING:
    from ..api import HumanParams, PrecomputedData, Weather
    from ..bundles import GvfBundle, LupBundle, ShadowBundle, SvfBundle


def compute_radiation(
    weather: Weather,
    svf_bundle: SvfBundle,
    shadow_bundle: ShadowBundle,
    gvf_bundle: GvfBundle,
    lup_bundle: LupBundle,
    human: HumanParams,
    use_anisotropic_sky: bool,
    precomputed: PrecomputedData | None,
    albedo_wall: float = 0.20,
    emis_wall: float = 0.90,
    tg_wall: float = 0.0,
) -> RadiationBundle:
    """
    Compute radiation budget for Tmrt calculation.

    Computes complete shortwave and longwave radiation fluxes from all directions,
    accounting for sky, ground, walls, and vegetation effects.

    Args:
        weather: Weather data (temperature, humidity, radiation, sun position)
        svf_bundle: Sky view factors with directional components
        shadow_bundle: Shadow fractions and vegetation transmissivity
        gvf_bundle: Ground view factors and albedo components
        lup_bundle: Upwelling longwave after thermal delay
        human: Human parameters (height, posture, absorptivities)
        use_anisotropic_sky: Use anisotropic (Perez) diffuse model if shadow matrices available
        precomputed: Optional pre-computed shadow matrices for anisotropic model
        albedo_wall: Wall albedo (default 0.20 for cobblestone)
        emis_wall: Wall emissivity (default 0.90 for brick/concrete)
        tg_wall: Wall temperature deviation from air temperature (K)

    Returns:
        RadiationBundle containing:
            - kdown: Downwelling shortwave (W/m²)
            - kup: Upwelling shortwave (reflected from ground)
            - ldown: Downwelling longwave (sky + wall emission)
            - lup: Upwelling longwave (from lup_bundle after thermal delay)
            - kside: Directional shortwave components (N, E, S, W)
            - lside: Directional longwave components (N, E, S, W)
            - kside_direct: Direct beam on vertical surface (for anisotropic)
            - drad: Diffuse radiation term (anisotropic or isotropic)

    Reference:
        - Lindberg et al. (2008) - SOLWEIG radiation model equations
        - Perez et al. (1993) - Anisotropic sky model
        - Jonsson et al. (2006) - Longwave radiation formulation
    """
    # Import here to avoid circular dependency
    from ..rustalgos import sky, vegetation

    # Sky emissivity (Jonsson et al. 2006)
    ta_k = weather.ta + KELVIN_OFFSET
    ea = 6.107 * 10 ** ((7.5 * weather.ta) / (237.3 + weather.ta)) * (weather.rh / 100.0)
    msteg = 46.5 * (ea / ta_k)
    esky = 1 - (1 + msteg) * np.exp(-np.sqrt(1.2 + 3.0 * msteg))

    # View factors (from SOLWEIG parameters - depends on posture)
    cyl = human.posture == "standing"
    if cyl:
        _f_up = F_UP_STANDING  # Reserved for future cylindric body model
        _f_side = F_SIDE_STANDING  # Reserved for future cylindric body model
        # f_cyl = F_CYL_STANDING  # Cylindrical projection factor for direct beam (not used here)
    else:
        _f_up = F_UP_SITTING  # Reserved for future cylindric body model  # noqa: F841
        _f_side = F_SIDE_SITTING  # Reserved for future cylindric body model  # noqa: F841
        # f_cyl = 0.2

    # Shortwave radiation components
    sin_alt = np.sin(np.radians(weather.sun_altitude))
    rad_i = weather.direct_rad
    rad_d = weather.diffuse_rad
    rad_g = weather.global_rad

    # Extract SVF components
    svf = svf_bundle.svf
    svf_directional = svf_bundle.svf_directional
    svf_veg = svf_bundle.svf_veg
    svf_veg_directional = svf_bundle.svf_veg_directional
    svf_aveg = svf_bundle.svf_aveg
    svf_aveg_directional = svf_bundle.svf_aveg_directional
    svfbuveg = svf_bundle.svfbuveg
    svfalfa = svf_bundle.svfalfa

    # Extract shadow components
    shadow = shadow_bundle.shadow
    psi = shadow_bundle.psi

    # Check if anisotropic sky model should be used
    has_shadow_matrices = precomputed is not None and precomputed.shadow_matrices is not None
    use_aniso = use_anisotropic_sky and has_shadow_matrices

    # Compute F_sh (fraction shadow on building walls based on sun altitude and SVF)
    zen = weather.sun_zenith * (np.pi / 180.0)  # Convert to radians for cylindric_wedge
    f_sh = _sky.cylindric_wedge(float(zen), as_float32(svfalfa))
    f_sh = np.nan_to_num(f_sh, nan=0.5)

    # Compute Kup (ground-reflected shortwave) using full directional model
    kup, kup_e, kup_s, kup_w, kup_n = Kup_veg_2015a(
        rad_i,
        rad_d,
        rad_g,
        weather.sun_altitude,
        svfbuveg,
        albedo_wall,
        f_sh,
        gvf_bundle.gvfalb,
        gvf_bundle.gvfalb_e,
        gvf_bundle.gvfalb_s,
        gvf_bundle.gvfalb_w,
        gvf_bundle.gvfalb_n,
        gvf_bundle.gvfalbnosh,
        gvf_bundle.gvfalbnosh_e,
        gvf_bundle.gvfalbnosh_s,
        gvf_bundle.gvfalbnosh_w,
        gvf_bundle.gvfalbnosh_n,
    )

    # Compute diffuse radiation and directional shortwave
    if use_aniso:
        # Type narrowing - precomputed and shadow_matrices are not None when use_aniso is True
        assert precomputed is not None
        assert precomputed.shadow_matrices is not None
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
        diffsh = shadow_mats.diffsh(psi, use_vegetation=psi < 0.5)
        shadow_mats.release_float32_cache()  # Free unpacked float32; bitpacked still available

        # Total relative luminance from sky patches into each cell
        ani_lum = _sky.weighted_patch_sum(
            as_float32(diffsh),
            as_float32(lv[:, 2]),
        )

        drad = ani_lum * rad_d

        # Compute asvf (angle from SVF) for anisotropic calculations
        asvf = np.arccos(np.sqrt(np.clip(svf, 0.0, 1.0)))

        # Pass bitpacked shadow matrices directly to Rust
        shmat = np.ascontiguousarray(shadow_mats._shmat_u8)
        vegshmat = np.ascontiguousarray(shadow_mats._vegshmat_u8)
        vbshmat = np.ascontiguousarray(shadow_mats._vbshmat_u8)

        # Compute base Ldown first (needed for lside_veg)
        ldown_base = (
            (svf + svf_veg - 1) * esky * SBC * (ta_k**4)
            + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k**4)
            + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + KELVIN_OFFSET) ** 4)
            + (2 - svf - svf_veg) * (1 - emis_wall) * esky * SBC * (ta_k**4)
        )

        # CI correction for non-clear conditions
        ci = weather.clearness_index
        if ci < 0.95:
            c = 1.0 - ci
            ldown_cloudy = (
                (svf + svf_veg - 1) * SBC * (ta_k**4)
                + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k**4)
                + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + KELVIN_OFFSET) ** 4)
                + (2 - svf - svf_veg) * (1 - emis_wall) * SBC * (ta_k**4)
            )
            ldown_base = ldown_base * (1 - c) + ldown_cloudy * c

        # Call lside_veg for base directional longwave (Least, Lsouth, Lwest, Lnorth)
        lside_veg_result = vegetation.lside_veg(
            as_float32(svf_directional.south),
            as_float32(svf_directional.west),
            as_float32(svf_directional.north),
            as_float32(svf_directional.east),
            as_float32(svf_veg_directional.east),
            as_float32(svf_veg_directional.south),
            as_float32(svf_veg_directional.west),
            as_float32(svf_veg_directional.north),
            as_float32(svf_aveg_directional.east),
            as_float32(svf_aveg_directional.south),
            as_float32(svf_aveg_directional.west),
            as_float32(svf_aveg_directional.north),
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            tg_wall,
            SBC,
            emis_wall,
            as_float32(ldown_base),
            esky,
            0.0,  # t (instrument offset, matching reference)
            as_float32(f_sh),
            weather.clearness_index,
            as_float32(lup_bundle.lup_e),  # TsWaveDelay-processed values
            as_float32(lup_bundle.lup_s),
            as_float32(lup_bundle.lup_w),
            as_float32(lup_bundle.lup_n),
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
        l_patches = as_float32(lv)

        # Adjust sky emissivity for cloudy conditions (CI < 0.95)
        # This matches the reference implementation: esky = CI * esky + (1 - CI) * 1.0
        esky_aniso = esky
        ci = weather.clearness_index
        if ci < 0.95:
            esky_aniso = ci * esky + (1 - ci) * 1.0

        # Create parameter structs for cleaner function signature
        sun_params = sky.SunParams(
            altitude=weather.sun_altitude,
            azimuth=weather.sun_azimuth,
        )
        sky_params = sky.SkyParams(
            esky=esky_aniso,
            ta=weather.ta,
            cyl=bool(cyl),
            wall_scheme=False,
            albedo=albedo_wall,
        )
        surface_params = sky.SurfaceParams(
            tgwall=tg_wall,
            ewall=emis_wall,
            rad_i=rad_i,
            rad_d=rad_d,
        )

        # Call full Rust anisotropic sky function with structs
        ani_sky_result = sky.anisotropic_sky(
            shmat,
            vegshmat,
            vbshmat,
            sun_params,
            as_float32(asvf),
            sky_params,
            l_patches,
            None,  # voxelTable
            None,  # voxelMaps
            as_float32(steradians),
            surface_params,
            as_float32(lup_bundle.lup),  # TsWaveDelay-processed value
            as_float32(lv),
            as_float32(shadow),
            as_float32(kup_e),
            as_float32(kup_s),
            as_float32(kup_w),
            as_float32(kup_n),
        )

        # Extract results from anisotropic sky
        ldown = np.array(ani_sky_result.ldown)
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
        # Total radiation on vertical surfaces (for Tmrt f_cyl term)
        kside_total = np.array(ani_sky_result.kside)
        lside_total = np.array(ani_sky_result.lside)

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
            as_float32(shadow),
            as_float32(svf_directional.south),
            as_float32(svf_directional.west),
            as_float32(svf_directional.north),
            as_float32(svf_directional.east),
            as_float32(svf_veg_directional.east),
            as_float32(svf_veg_directional.south),
            as_float32(svf_veg_directional.west),
            as_float32(svf_veg_directional.north),
            weather.sun_azimuth,
            weather.sun_altitude,
            psi,
            0.0,  # t (instrument offset)
            albedo_wall,
            as_float32(f_sh),
            as_float32(kup_e),
            as_float32(kup_s),
            as_float32(kup_w),
            as_float32(kup_n),
            bool(cyl),
            None,  # lv (None for isotropic)
            False,  # anisotropic_sky
            None,  # diffsh (None for isotropic)
            as_float32(asvf),
            None,  # shmat (None for isotropic)
            None,  # vegshmat (None for isotropic)
            None,  # vbshvegshmat (None for isotropic)
        )
        kside_e = np.array(kside_result.keast)
        kside_s = np.array(kside_result.ksouth)
        kside_w = np.array(kside_result.kwest)
        kside_n = np.array(kside_result.knorth)
        kside_i = np.array(kside_result.kside_i)
        # Total radiation on vertical surfaces (for Tmrt f_cyl term)
        kside_total = kside_i  # Isotropic uses direct beam only
        lside_total = np.zeros_like(kside_i)  # Not used in isotropic Tmrt formula

        # Longwave: Ldown (from Jonsson et al. 2006)
        ldown = (
            (svf + svf_veg - 1) * esky * SBC * (ta_k**4)
            + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k**4)
            + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + KELVIN_OFFSET) ** 4)
            + (2 - svf - svf_veg) * (1 - emis_wall) * esky * SBC * (ta_k**4)
        )

        # CI correction for non-clear conditions (reference: if CI < 0.95)
        # Under cloudy skies, effective sky emissivity approaches 1.0
        ci = weather.clearness_index
        if ci < 0.95:
            c = 1.0 - ci
            ldown_cloudy = (
                (svf + svf_veg - 1) * SBC * (ta_k**4)  # No esky for cloudy
                + (2 - svf_veg - svf_aveg) * emis_wall * SBC * (ta_k**4)
                + (svf_aveg - svf) * emis_wall * SBC * ((weather.ta + tg_wall + KELVIN_OFFSET) ** 4)
                + (2 - svf - svf_veg) * (1 - emis_wall) * SBC * (ta_k**4)  # No esky
            )
            ldown = ldown * (1 - c) + ldown_cloudy * c

        # Use Rust lside_veg for directional longwave
        lside_veg_result = vegetation.lside_veg(
            as_float32(svf_directional.south),
            as_float32(svf_directional.west),
            as_float32(svf_directional.north),
            as_float32(svf_directional.east),
            as_float32(svf_veg_directional.east),
            as_float32(svf_veg_directional.south),
            as_float32(svf_veg_directional.west),
            as_float32(svf_veg_directional.north),
            as_float32(svf_aveg_directional.east),
            as_float32(svf_aveg_directional.south),
            as_float32(svf_aveg_directional.west),
            as_float32(svf_aveg_directional.north),
            weather.sun_azimuth,
            weather.sun_altitude,
            weather.ta,
            tg_wall,
            SBC,
            emis_wall,
            as_float32(ldown),
            esky,
            0.0,  # t (instrument offset, matching reference)
            as_float32(f_sh),
            weather.clearness_index,
            as_float32(lup_bundle.lup_e),  # TsWaveDelay-processed values
            as_float32(lup_bundle.lup_s),
            as_float32(lup_bundle.lup_w),
            as_float32(lup_bundle.lup_n),
            False,  # anisotropic_sky
        )
        lside_e = np.array(lside_veg_result.least)
        lside_s = np.array(lside_veg_result.lsouth)
        lside_w = np.array(lside_veg_result.lwest)
        lside_n = np.array(lside_veg_result.lnorth)

    # Kdown (downwelling shortwave = direct on horizontal + diffuse sky + wall reflected)
    kdown = rad_i * shadow * sin_alt + drad + albedo_wall * (1 - svfbuveg) * (rad_g * (1 - f_sh) + rad_d * f_sh)

    return RadiationBundle(
        kdown=as_float32(kdown),
        kup=as_float32(kup),
        ldown=as_float32(ldown),
        lup=lup_bundle.lup,  # Already float32 from LupBundle
        kside=DirectionalArrays(
            north=as_float32(kside_n),
            east=as_float32(kside_e),
            south=as_float32(kside_s),
            west=as_float32(kside_w),
        ),
        lside=DirectionalArrays(
            north=as_float32(lside_n),
            east=as_float32(lside_e),
            south=as_float32(lside_s),
            west=as_float32(lside_w),
        ),
        kside_total=as_float32(kside_total),
        lside_total=as_float32(lside_total),
        drad=as_float32(drad),
    )
