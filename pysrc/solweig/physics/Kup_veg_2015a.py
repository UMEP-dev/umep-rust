import numpy as np


def Kup_veg_2015a(
    radI,
    radD,
    radG,
    altitude,
    svfbuveg,
    albedo_b,
    F_sh,
    gvfalb,
    gvfalbE,
    gvfalbS,
    gvfalbW,
    gvfalbN,
    gvfalbnosh,
    gvfalbnoshE,
    gvfalbnoshS,
    gvfalbnoshW,
    gvfalbnoshN,
):
    # Pre-compute common terms once (2x speedup)
    radI_sin_alt = radI * np.sin(altitude * (np.pi / 180.0))
    common_term = radD * svfbuveg + albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)

    Kup = gvfalb * radI_sin_alt + common_term * gvfalbnosh
    KupE = gvfalbE * radI_sin_alt + common_term * gvfalbnoshE
    KupS = gvfalbS * radI_sin_alt + common_term * gvfalbnoshS
    KupW = gvfalbW * radI_sin_alt + common_term * gvfalbnoshW
    KupN = gvfalbN * radI_sin_alt + common_term * gvfalbnoshN

    return Kup, KupE, KupS, KupW, KupN
