"""
Daily shading calculations for a given DSM and vegetation DSM.
Uses Rust algorithms for shadow calculations.
"""

import datetime as dt
from builtins import range

import numpy as np
from tqdm import tqdm
from umep import common
from umep.util.SEBESOLWEIGCommonFiles import sun_position as sp

from ..rustalgos import shadowing


def daily_shading(
    dsm,
    vegdsm,
    vegdsm2,
    scale,
    lon,
    lat,
    dsm_width,
    dsm_height,
    tv,
    UTC,
    usevegdem,
    timeInterval,
    onetime,
    folder,
    dsm_transf,
    dsm_crs,
    trans,
    dst,
    wallshadow,
    wheight,
    waspect,
):
    # lon = lonlat[0]
    # lat = lonlat[1]
    year = tv[0]
    month = tv[1]
    day = tv[2]

    alt = np.median(dsm)
    location = {"longitude": lon, "latitude": lat, "altitude": alt}
    if usevegdem == 1:
        psi = trans
        # amaxvalue
        vegmax = vegdsm.max()
        amaxvalue = dsm.max() - dsm.min()
        amaxvalue = np.maximum(amaxvalue, vegmax)

        # Elevation vegdsms if buildingDSM includes ground heights
        vegdem = vegdsm + dsm
        vegdem[vegdem == dsm] = 0
        vegdem2 = vegdsm2 + dsm
        vegdem2[vegdem2 == dsm] = 0

        # Bush separation
        bush = np.logical_not(vegdem2 * vegdem) * vegdem
    else:
        psi = 1.0
        vegdem = np.zeros_like(dsm)
        vegdem2 = np.zeros_like(dsm)
        amaxvalue = dsm.max() - dsm.min()
        bush = np.zeros_like(dsm)

    shtot = np.zeros((dsm_height, dsm_width))

    if onetime == 1:
        itera = 1
    else:
        itera = int(1440 / timeInterval)

    alt = np.zeros(itera)
    azi = np.zeros(itera)
    hour = 0
    index = 0
    time = dict()
    time["UTC"] = UTC

    if wallshadow == 1:
        walls = wheight
        dirwalls = waspect
    else:
        walls = np.zeros((dsm_height, dsm_width))
        dirwalls = np.zeros((dsm_height, dsm_width))

    for i in tqdm(range(0, itera)):
        if onetime == 0:
            minu = int(timeInterval * i)
            if minu >= 60:
                hour = int(np.floor(minu / 60))
                minu = int(minu - hour * 60)
        else:
            minu = tv[4]
            hour = tv[3]

        doy = day_of_year(year, month, day)

        ut_time = doy - 1.0 + ((hour - dst) / 24.0) + (minu / (60.0 * 24.0)) + (0.0 / (60.0 * 60.0 * 24.0))

        if ut_time < 0:
            year = year - 1
            month = 12
            day = 31
            doy = day_of_year(year, month, day)
            ut_time = ut_time + doy - 1

        HHMMSS = dectime_to_timevec(ut_time)
        time["year"] = year
        time["month"] = month
        time["day"] = day
        time["hour"] = HHMMSS[0]
        time["min"] = HHMMSS[1]
        time["sec"] = HHMMSS[2]

        sun = sp.sun_position(time, location)
        alt[i] = 90.0 - sun["zenith"]
        azi[i] = sun["azimuth"]

        if time["sec"] == 59:  # issue 228 and 256
            time["sec"] = 0
            time["min"] = time["min"] + 1
            if time["min"] == 60:
                time["min"] = 0
                time["hour"] = time["hour"] + 1
                if time["hour"] == 24:
                    time["hour"] = 0

        time_vector = dt.datetime(year, month, day, time["hour"], time["min"], time["sec"])
        timestr = time_vector.strftime("%Y%m%d_%H%M")
        if alt[i] > 0:
            if wallshadow == 1:  # Include wall shadows (Issue #121)
                result = shadowing.calculate_shadows_wall_ht_25(
                    azi[i],
                    alt[i],
                    scale,
                    amaxvalue,
                    dsm,
                    vegdem,
                    vegdem2,
                    bush,
                    wheight if wallshadow == 1 else np.zeros((dsm_height, dsm_width)),
                    waspect * np.pi / 180.0 if wallshadow == 1 else np.zeros((dsm_height, dsm_width)),
                    None,
                    None,
                    None,
                )
                sh = result.bldg_sh - (1 - result.veg_sh) * (1 - psi)
                if onetime == 0:
                    filenamewallshve = folder + "/facade_shdw_veg/facade_shdw_veg_" + timestr + "_LST.tif"
                    common.save_raster(filenamewallshve, result.wall_sh_veg, dsm_transf, dsm_crs)
                if onetime == 0:
                    filename = folder + "/shadow_ground/shadow_ground_" + timestr + "_LST.tif"
                    common.save_raster(filename, sh, dsm_transf, dsm_crs)
                    filenamewallsh = folder + "/facade_shdw_bldgs/facade_shdw_bldgs_" + timestr + "_LST.tif"
                    common.save_raster(filenamewallsh, result.wall_sh, dsm_transf, dsm_crs)
            else:
                result = shadowing.calculate_shadows_wall_ht_25(
                    azi[i],
                    alt[i],
                    scale,
                    amaxvalue,
                    dsm,
                    vegdem,
                    vegdem2,
                    bush,
                    np.zeros((dsm_height, dsm_width)),
                    np.zeros((dsm_height, dsm_width)),
                    None,
                    None,
                    None,
                )
                sh = result.bldg_sh - (1 - result.veg_sh) * (1 - psi)
                if onetime == 0:
                    filename = folder + "/Shadow_" + timestr + "_LST.tif"
                    common.save_raster(filename, sh, dsm_transf, dsm_crs)

            shtot = shtot + sh
            index += 1

    shfinal = shtot / index

    if wallshadow == 1:
        if onetime == 1:
            filenamewallsh = folder + "/facade_shdw_bldgs/facade_shdw_bldgs_" + timestr + "_LST.tif"
            common.save_raster(filenamewallsh, result.wall_sh, dsm_transf, dsm_crs)
            filenamewallshve = folder + "/facade_shdw_veg/facade_shdw_veg_" + timestr + "_LST.tif"
            common.save_raster(filenamewallshve, result.wall_sh_veg, dsm_transf, dsm_crs)

    shadowresult = {"shfinal": shfinal, "time_vector": time_vector}

    return shadowresult


def day_of_year(yy, month, day):
    if (yy % 4) == 0:
        if (yy % 100) == 0:
            if (yy % 400) == 0:
                leapyear = 1
            else:
                leapyear = 0
        else:
            leapyear = 1
    else:
        leapyear = 0

    if leapyear == 1:
        dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    doy = np.sum(dayspermonth[0 : month - 1]) + day

    return doy


def dectime_to_timevec(dectime):
    # This subroutine converts dectime to individual hours, minutes and seconds

    doy = np.floor(dectime)

    DH = dectime - doy
    HOURS = int(24 * DH)

    DM = 24 * DH - HOURS
    MINS = int(60 * DM)

    DS = 60 * DM - MINS
    SECS = int(60 * DS)

    return (HOURS, MINS, SECS)
