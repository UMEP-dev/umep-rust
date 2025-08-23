"""
Solweig model in Python which calls shadowing and GVF calculations implemented in Rust.

Implemented from SolweigRunRust class, which inherits from SolweigRunCore.

This version is a copy except for the changes made to call the Rust functions directly.
"""

from copy import deepcopy

import numpy as np
from umep.functions.SOLWEIGpython.cylindric_wedge import cylindric_wedge
from umep.functions.SOLWEIGpython.daylen import daylen
from umep.functions.SOLWEIGpython.Kup_veg_2015a import Kup_veg_2015a

# Anisotropic longwave
from umep.functions.SOLWEIGpython.patch_radiation import patch_steradians
from umep.functions.SOLWEIGpython.TsWaveDelay_2015a import TsWaveDelay_2015a

# Wall surface temperature scheme
from umep.functions.SOLWEIGpython.wall_surface_temperature import wall_surface_temperature
from umep.util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b
from umep.util.SEBESOLWEIGCommonFiles.create_patches import create_patches
from umep.util.SEBESOLWEIGCommonFiles.diffusefraction import diffusefraction
from umep.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3

from ..rustalgos import gvf, shadowing, sky, vegetation


def Solweig_2025a_calc(
    i,
    dsm,
    scale,
    rows,
    cols,
    svf,
    svfN,
    svfW,
    svfE,
    svfS,
    svfveg,
    svfNveg,
    svfEveg,
    svfSveg,
    svfWveg,
    svfaveg,
    svfEaveg,
    svfSaveg,
    svfWaveg,
    svfNaveg,
    vegdem,
    vegdem2,
    albedo_b,
    absK,
    absL,
    ewall,
    Fside,
    Fup,
    Fcyl,
    altitude,
    azimuth,
    zen,
    jday,
    usevegdem,
    onlyglobal,
    buildings,
    location,
    psi,
    landcover,
    lc_grid,
    dectime,
    altmax,
    dirwalls,
    walls,
    cyl,
    elvis,
    Ta,
    RH,
    radG,
    radD,
    radI,
    P,
    amaxvalue,
    bush,
    Twater,
    TgK,
    Tstart,
    alb_grid,
    emis_grid,
    TgK_wall,
    Tstart_wall,
    TmaxLST,
    TmaxLST_wall,
    first,
    second,
    svfalfa,
    svfbuveg,
    firstdaytime,
    timeadd,
    timestepdec,
    Tgmap1,
    Tgmap1E,
    Tgmap1S,
    Tgmap1W,
    Tgmap1N,
    CI,
    TgOut1,
    diffsh,
    shmat,
    vegshmat,
    vbshvegshmat,
    anisotropic_sky,
    asvf,
    patch_option,
    voxelMaps,
    voxelTable,
    ws,
    wallScheme,
    timeStep,
    steradians,
    walls_scheme,
    dirwalls_scheme,
):
    # def Solweig_2021a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg,
    #                       svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL,
    #                       ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi,
    #                       landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P,
    #                       amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST,
    #                       TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1,
    #                       Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, ani):

    # This is the core function of the SOLWEIG model
    # 2016-Aug-28
    # Fredrik Lindberg, fredrikl@gvc.gu.se
    # Goteborg Urban Climate Group
    # Gothenburg University
    #
    # Input variables:
    # dsm = digital surface model
    # scale = height to pixel size (2m pixel gives scale = 0.5)
    # svf,svfN,svfW,svfE,svfS = SVFs for building and ground
    # svfveg,svfNveg,svfEveg,svfSveg,svfWveg = Veg SVFs blocking sky
    # svfaveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg = Veg SVFs blocking buildings
    # vegdem = Vegetation canopy DSM
    # vegdem2 = Vegetation trunk zone DSM
    # albedo_b = building wall albedo
    # absK = human absorption coefficient for shortwave radiation
    # absL = human absorption coefficient for longwave radiation
    # ewall = Emissivity of building walls
    # Fside = The angular factors between a person and the surrounding surfaces
    # Fup = The angular factors between a person and the surrounding surfaces
    # Fcyl = The angular factors between a culidric person and the surrounding surfaces
    # altitude = Sun altitude (degree)
    # azimuth = Sun azimuth (degree)
    # zen = Sun zenith angle (radians)
    # jday = day of year
    # usevegdem = use vegetation scheme
    # onlyglobal = calculate dir and diff from global shortwave (Reindl et al. 1990)
    # buildings = Boolena grid to identify building pixels
    # location = geographic location
    # height = height of measurements point (center of gravity of human)
    # psi = 1 - Transmissivity of shortwave through vegetation
    # landcover = use landcover scheme !!!NEW IN 2015a!!!
    # lc_grid = grid with landcoverclasses
    # lc_class = table with landcover properties
    # dectime = decimal time
    # altmax = maximum sun altitude
    # dirwalls = aspect of walls
    # walls = one pixel row outside building footprint. height of building walls
    # cyl = consider man as cylinder instead of cude
    # elvis = dummy
    # Ta = air temp
    # RH
    # radG = global radiation
    # radD = diffuse
    # radI = direct
    # P = pressure
    # amaxvalue = max height of buildings
    # bush = grid representing bushes
    # Twater = temperature of water (daily)
    # TgK, Tstart, TgK_wall, Tstart_wall, TmaxLST,TmaxLST_wall,
    # alb_grid, emis_grid = albedo and emmissivity on ground
    # first, second = conneted to old Ts model (source area based on Smidt et al.)
    # svfalfa = SVF recalculated to angle
    # svfbuveg = complete SVF
    # firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
    # CI = Clearness index
    # TgOut1 = old Ts model
    # diffsh, ani = Used in anisotrpic models (Wallenberg et al. 2019, 2022)

    # # # Core program start # # #
    # Instrument offset in degrees
    t = 0.0

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Degrees to radians
    deg2rad = np.pi / 180

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location["latitude"])

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.0)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    if altitude > 0:  # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100.0, radG, location, P)
        if (CI > 1) or (np.inf == CI):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100.0, radG, location, P)
            if (CI > 1) or (np.inf == CI):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = zen * (180 / np.pi)
            # Relative luminance
            lv, pc_, pb_ = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice, patch_option)
            # Total relative luminance from sky, i.e. from each patch, into each cell
            aniLum = np.zeros((rows, cols))
            for idx in range(lv.shape[0]):
                aniLum += diffsh[:, :, idx] * lv[idx, 2]

            dRad = aniLum * radD  # Total diffuse radiation from sky into each cell
        else:
            dRad = radD * svfbuveg
            patchchoice = 1
            lv = None

        # Shadow  images
        if usevegdem == 1:
            result = shadowing.calculate_shadows_wall_ht_25(
                azimuth,
                altitude,
                scale,
                amaxvalue,
                dsm.astype(np.float32),
                vegdem.astype(np.float32),
                vegdem2.astype(np.float32),
                bush.astype(np.float32),
                walls.astype(np.float32),
                (dirwalls * np.pi / 180.0).astype(np.float32),
                walls_scheme.astype(np.float32),
                (dirwalls_scheme * np.pi / 180.0).astype(np.float32),
                None,
            )
            vegsh = result.veg_sh
            sh = result.bldg_sh
            wallsh = result.wall_sh
            wallsun = result.wall_sun
            wallshve = result.wall_sh_veg
            facesun = result.face_sun
            wallsh_ = result.face_sh
            shadow = result.bldg_sh - (1 - result.veg_sh) * (1 - psi)
        else:
            result = shadowing.calculate_shadows_wall_ht_25(
                azimuth,
                altitude,
                scale,
                dsm.astype(np.float32),
                None,
                None,
                None,
                walls.astype(np.float32),
                (dirwalls * np.pi / 180.0).astype(np.float32),
                None,
                None,
                None,
            )
            sh = result.bldg_sh
            wallsh = result.wall_sh
            wallsun = result.wall_sun
            facesh = result.face_sh
            facesun = result.face_sun
            shadow = result.bldg_sh

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        # Tgamp = (TgK * altmax - Tstart) + Tstart # Old
        Tgamp = TgK * altmax + Tstart  # Fixed 2021
        # Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall) # Old
        Tgampwall = TgK_wall * altmax + Tstart_wall
        Tg = Tgamp * np.sin(
            (((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2
        )  # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * np.sin(
            (((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2
        )  # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1.0, Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radG / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1

        radG0 = radI0 * (np.sin(altitude * deg2rad)) + _
        CI_TgG = (radG / radG0) + (1 - corr)
        if (CI_TgG > 1) or (CI_TgG == np.inf):
            CI_TgG = 1

        # Tg = Tg * CI_Tg  # new estimation
        # Tgwall = Tgwall * CI_Tg
        Tg = Tg * CI_TgG  # new estimation
        Tgwall = Tgwall * CI_TgG
        if landcover == 1:
            Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        # # # # Ground View Factors # # # #
        gvf_result = gvf.gvf_calc(
            wallsun.astype(np.float32),
            walls.astype(np.float32),
            buildings.astype(np.float32),
            scale,
            shadow.astype(np.float32),
            first,
            second,
            dirwalls.astype(np.float32),
            Tg.astype(np.float32),
            Tgwall,
            Ta,
            emis_grid.astype(np.float32),
            ewall,
            alb_grid.astype(np.float32),
            SBC,
            albedo_b,
            Twater,
            lc_grid.astype(np.float32) if lc_grid is not None else None,
            landcover,
        )

        # # # # Lup, daytime # # # #
        # Surface temperature wave delay - new as from 2014a
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvf_result.gvf_lup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(
            gvf_result.gvf_lup_e, firstdaytime, timeadd, timestepdec, Tgmap1E
        )
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(
            gvf_result.gvf_lup_s, firstdaytime, timeadd, timestepdec, Tgmap1S
        )
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(
            gvf_result.gvf_lup_w, firstdaytime, timeadd, timestepdec, Tgmap1W
        )
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(
            gvf_result.gvf_lup_n, firstdaytime, timeadd, timestepdec, Tgmap1N
        )

        # # For Tg output in POIs
        TgTemp = Tg * shadow + Ta
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(
            TgTemp, firstdaytime, timeadd, timestepdec, TgOut1
        )  # timeadd only here v2021a

        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, rows, cols)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[np.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = (
            radI * shadow * np.sin(altitude * (np.pi / 180))
            + dRad
            + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)
        )  # *sin(altitude(i) * (pi / 180))

        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(
            radI,
            radD,
            radG,
            altitude,
            svfbuveg,
            albedo_b,
            F_sh,
            gvf_result.gvfalb,
            gvf_result.gvfalb_e,
            gvf_result.gvfalb_s,
            gvf_result.gvfalb_w,
            gvf_result.gvfalb_n,
            gvf_result.gvfalbnosh,
            gvf_result.gvfalbnosh_e,
            gvf_result.gvfalbnosh_s,
            gvf_result.gvfalbnosh_w,
            gvf_result.gvfalbnosh_n,
        )

        kside_result = vegetation.kside_veg(
            radI,
            radD,
            radG,
            shadow.astype(np.float32),
            svfS.astype(np.float32),
            svfW.astype(np.float32),
            svfN.astype(np.float32),
            svfE.astype(np.float32),
            svfEveg.astype(np.float32),
            svfSveg.astype(np.float32),
            svfWveg.astype(np.float32),
            svfNveg.astype(np.float32),
            azimuth,
            altitude,
            psi,
            t,
            albedo_b,
            F_sh.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            bool(cyl),
            lv.astype(np.float32) if lv is not None else None,
            bool(anisotropic_sky),
            diffsh.astype(np.float32) if diffsh is not None else None,
            asvf.astype(np.float32) if asvf is not None else None,
            shmat.astype(np.float32) if shmat is not None else None,
            vegshmat.astype(np.float32) if vegshmat is not None else None,
            vbshvegshmat.astype(np.float32) if vbshvegshmat is not None else None,
        )
        Keast = kside_result.keast
        Ksouth = kside_result.ksouth
        Kwest = kside_result.kwest
        Knorth = kside_result.knorth
        KsideI = kside_result.kside_i
        KsideD = kside_result.kside_d
        Kside = kside_result.kside

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #
        Tgwall = 0
        # CI_Tg = -999  # F_sh = []

        # Nocturnal K fluxes set to 0
        Knight = np.zeros((rows, cols))
        Kdown = np.zeros((rows, cols))
        Kwest = np.zeros((rows, cols))
        Kup = np.zeros((rows, cols))
        Keast = np.zeros((rows, cols))
        Ksouth = np.zeros((rows, cols))
        Knorth = np.zeros((rows, cols))
        KsideI = np.zeros((rows, cols))
        KsideD = np.zeros((rows, cols))
        F_sh = np.zeros((rows, cols))
        Tg = np.zeros((rows, cols))
        shadow = np.zeros((rows, cols))
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)
        dRad = np.zeros((rows, cols))
        Kside = np.zeros((rows, cols))

        # # # # Lup # # # #
        Lup = SBC * emis_grid * ((Knight + Ta + Tg + 273.15) ** 4)
        if landcover == 1:
            Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp

        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        TgOut = Ta + Tg

        I0 = 0
        timeadd = 0
        firstdaytime = 1

    # # # # Ldown # # # #
    Ldown = (
        (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4)
        + (2 - svfveg - svfaveg) * ewall * SBC * ((Ta + 273.15) ** 4)
        + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4)
        + (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)
    )  # Jonsson et al.(2006)
    # Ldown = Ldown - 25 # Shown by Jonsson et al.(2006) and Duarte et al.(2006)

    if CI < 0.95:  # non - clear conditions
        c = 1 - CI
        Ldown = Ldown * (1 - c) + c * (
            (svf + svfveg - 1) * SBC * ((Ta + 273.15) ** 4)
            + (2 - svfveg - svfaveg) * ewall * SBC * ((Ta + 273.15) ** 4)
            + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4)
            + (2 - svf - svfveg) * (1 - ewall) * SBC * ((Ta + 273.15) ** 4)
        )  # NOT REALLY TESTED!!! BUT MORE CORRECT?

    # # # # Lside # # # #
    lside_veg_result = vegetation.lside_veg(
        svfS.astype(np.float32),
        svfW.astype(np.float32),
        svfN.astype(np.float32),
        svfE.astype(np.float32),
        svfEveg.astype(np.float32),
        svfSveg.astype(np.float32),
        svfWveg.astype(np.float32),
        svfNveg.astype(np.float32),
        svfEaveg.astype(np.float32),
        svfSaveg.astype(np.float32),
        svfWaveg.astype(np.float32),
        svfNaveg.astype(np.float32),
        azimuth,
        altitude,
        Ta,
        Tgwall,
        SBC,
        ewall,
        Ldown.astype(np.float32),
        esky,
        t,
        F_sh.astype(np.float32),
        CI,
        LupE.astype(np.float32),
        LupS.astype(np.float32),
        LupW.astype(np.float32),
        LupN.astype(np.float32),
        bool(anisotropic_sky),
    )
    Least = lside_veg_result.least
    Lsouth = lside_veg_result.lsouth
    Lwest = lside_veg_result.lwest
    Lnorth = lside_veg_result.lnorth

    # New parameterization scheme for wall temperatures
    if wallScheme == 1:
        # albedo_g = 0.15 #TODO Change to correct
        if altitude < 0:
            wallsh_ = 0
        voxelTable = wall_surface_temperature(
            voxelTable, wallsh_, altitude, azimuth, timeStep, radI, radD, radG, Ldown, Lup, Ta, esky
        )
    # Anisotropic sky
    if anisotropic_sky == 1:
        if "lv" not in locals():
            # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)

            patch_emissivities = np.zeros(skyvaultalt.shape[0])

            x = np.transpose(np.atleast_2d(skyvaultalt))
            y = np.transpose(np.atleast_2d(skyvaultazi))
            z = np.transpose(np.atleast_2d(patch_emissivities))

            L_patches = np.append(np.append(x, y, axis=1), z, axis=1)

        else:
            L_patches = deepcopy(lv)

        # Calculate steradians for patches if it is the first model iteration
        if i == 0:
            steradians, skyalt, patch_altitude = patch_steradians(L_patches)

        # Create lv from L_patches if nighttime, i.e. lv does not exist
        if altitude < 0:
            # CI = deepcopy(CI)
            lv = deepcopy(L_patches)
            KupE = np.zeros_like(lv)
            KupS = np.zeros_like(lv)
            KupW = np.zeros_like(lv)
            KupN = np.zeros_like(lv)

        # Adjust sky emissivity under semi-cloudy/hazy/cloudy/overcast conditions, i.e. CI lower than 0.95
        if CI < 0.95:
            esky_c = CI * esky + (1 - CI) * 1.0
            esky = esky_c

        ani_sky_result = sky.anisotropic_sky(
            shmat.astype(np.float32),
            vegshmat.astype(np.float32),
            vbshvegshmat.astype(np.float32),
            altitude,
            azimuth,
            asvf.astype(np.float32),
            bool(cyl),
            esky,
            L_patches.astype(np.float32),
            bool(wallScheme),
            voxelTable.astype(np.float32) if voxelTable is not None else None,
            voxelMaps.astype(np.float32) if voxelMaps is not None else None,
            steradians.astype(np.float32),
            Ta,
            Tgwall,
            ewall,
            Lup.astype(np.float32),
            radI,
            radD,
            radG,
            lv.astype(np.float32),
            albedo_b,
            False,
            diffsh.astype(np.float32),
            shadow.astype(np.float32),
            KupE.astype(np.float32),
            KupS.astype(np.float32),
            KupW.astype(np.float32),
            KupN.astype(np.float32),
            i,
        )
        Ldown = ani_sky_result.ldown
        Lside = ani_sky_result.lside
        Lside_sky = ani_sky_result.lside_sky
        Lside_veg = ani_sky_result.lside_veg
        Lside_sh = ani_sky_result.lside_sh
        Lside_sun = ani_sky_result.lside_sun
        Lside_ref = ani_sky_result.lside_ref
        Least_ = ani_sky_result.least
        Lwest_ = ani_sky_result.lwest
        Lnorth_ = ani_sky_result.lnorth
        Lsouth_ = ani_sky_result.lsouth
        Keast = ani_sky_result.keast
        Ksouth = ani_sky_result.ksouth
        Kwest = ani_sky_result.kwest
        Knorth = ani_sky_result.knorth
        KsideI = ani_sky_result.kside_i
        KsideD = ani_sky_result.kside_d
        Kside = ani_sky_result.kside
        steradians = ani_sky_result.steradians
        skyalt = ani_sky_result.skyalt
    else:
        Lside = np.zeros((rows, cols))
        L_patches = None

    # Box and anisotropic longwave
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # Human body considered as a cylinder with isotropic all-sky diffuse
    if cyl == 1 and anisotropic_sky == 0:
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * (
            (Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside
        )
    # Human body considered as a cylinder with Perez et al. (1993) (anisotropic sky diffuse)
    # and Martin and Berdahl (1984) (anisotropic sky longwave)
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * (
            (Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside
        )
    # Knorth = nan Ksouth = nan Kwest = nan Keast = nan
    else:  # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * (
            (Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside
        )

    Tmrt = np.sqrt(np.sqrt(Sstr / (absL * SBC))) - 273.2

    # Add longwave to cardinal directions for output in POI
    if (cyl == 1) and (anisotropic_sky == 1):
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_

    return (
        Tmrt,
        Kdown,
        Kup,
        Ldown,
        Lup,
        Tg,
        ea,
        esky,
        I0,
        CI,
        shadow,
        firstdaytime,
        timestepdec,
        timeadd,
        Tgmap1,
        Tgmap1E,
        Tgmap1S,
        Tgmap1W,
        Tgmap1N,
        Keast,
        Ksouth,
        Kwest,
        Knorth,
        Least,
        Lsouth,
        Lwest,
        Lnorth,
        KsideI,
        TgOut1,
        TgOut,
        radI,
        radD,
        Lside,
        L_patches,
        CI_Tg,
        CI_TgG,
        KsideD,
        dRad,
        Kside,
        steradians,
        voxelTable,
    )
