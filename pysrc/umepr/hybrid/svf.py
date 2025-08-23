"""
SVF hybrid - Python implementation of the SVF algorithm which calls Rust for the shadowing calculations.

NOTE: Used for testing.
There is a full Rust implementation which should be used instead.
"""

import numpy as np
from tqdm import tqdm
from umep.util import shadowingfunctions as shadow
from umep.util.SEBESOLWEIGCommonFiles.create_patches import create_patches

from ..rustalgos import shadowing


def annulus_weight(altitude, aziinterval):
    n = 90.0
    steprad = (360.0 / aziinterval) * (np.pi / 180.0)
    annulus = 91.0 - altitude
    w = (1.0 / (2.0 * np.pi)) * np.sin(np.pi / (2.0 * n)) * np.sin((np.pi * (2.0 * annulus - 1.0)) / (2.0 * n))
    weight = steprad * w

    return weight


def svfForProcessing153_rust_shdw(dsm, vegdem, vegdem2, scale, usevegdem, amax):
    # setup
    rows = dsm.shape[0]
    cols = dsm.shape[1]
    svf = np.zeros([rows, cols], dtype=np.float32)
    svfE = np.zeros([rows, cols], dtype=np.float32)
    svfS = np.zeros([rows, cols], dtype=np.float32)
    svfW = np.zeros([rows, cols], dtype=np.float32)
    svfN = np.zeros([rows, cols], dtype=np.float32)
    svfveg = np.zeros((rows, cols), dtype=np.float32)
    svfEveg = np.zeros((rows, cols), dtype=np.float32)
    svfSveg = np.zeros((rows, cols), dtype=np.float32)
    svfWveg = np.zeros((rows, cols), dtype=np.float32)
    svfNveg = np.zeros((rows, cols), dtype=np.float32)
    svfaveg = np.zeros((rows, cols), dtype=np.float32)
    svfEaveg = np.zeros((rows, cols), dtype=np.float32)
    svfSaveg = np.zeros((rows, cols), dtype=np.float32)
    svfWaveg = np.zeros((rows, cols), dtype=np.float32)
    svfNaveg = np.zeros((rows, cols), dtype=np.float32)

    # % Elevation vegdems if buildingDSM inclused ground heights
    # vegdem = vegdem + dsm
    # vegdem[vegdem == dsm] = 0
    # vegdem2 = vegdem2 + dsm
    # vegdem2[vegdem2 == dsm] = 0

    # % Bush separation
    # bush = np.logical_not(vegdem2 * vegdem) * vegdem
    bush = np.copy(vegdem)
    bush[vegdem2 > 0] = 0.0

    index = 0

    # patch_option = 1 # 145 patches
    patch_option = 2  # 153 patches
    # patch_option = 3 # 306 patches
    # patch_option = 4 # 612 patches

    # Create patches based on patch_option
    (
        skyvaultalt,
        skyvaultazi,
        annulino,
        skyvaultaltint,
        aziinterval,
        skyvaultaziint,
        azistart,
    ) = create_patches(patch_option)

    skyvaultaziint = np.array([360 / patches for patches in aziinterval])
    iazimuth = np.hstack(np.zeros((1, np.sum(aziinterval))))  # Nils

    # float 32 for memory
    shmat = np.zeros((rows, cols, np.sum(aziinterval)), dtype=np.float32)
    vegshmat = np.zeros((rows, cols, np.sum(aziinterval)), dtype=np.float32)
    vbshvegshmat = np.zeros((rows, cols, np.sum(aziinterval)), dtype=np.float32)

    for j in range(0, skyvaultaltint.shape[0]):
        for k in range(0, int(360 / skyvaultaziint[j])):
            iazimuth[index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[index] > 360.0:
                iazimuth[index] = iazimuth[index] - 360.0
            index = index + 1

    # NOTE: total for progress
    total = 0
    for i in range(0, skyvaultaltint.shape[0]):
        for j in np.arange(0, aziinterval[int(i)]):
            total += 1
    progress = tqdm(total=total)
    #
    aziintervalaniso = np.ceil(aziinterval / 2.0)
    index = 0
    for i in range(0, skyvaultaltint.shape[0]):
        for j in np.arange(0, aziinterval[int(i)]):
            altitude = skyvaultaltint[int(i)]
            azimuth = iazimuth[int(index)]

            # Casting shadow
            if usevegdem == 1:
                # numba doesn't seem to offer notable gains in this instance
                result = shadowing.calculate_shadows_wall_ht_25(
                    azimuth,
                    altitude,
                    scale,
                    amax,
                    dsm,
                    vegdem,
                    vegdem2,
                    bush,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                vegsh = result.veg_sh
                vbshvegsh = result.veg_blocks_bldg_sh
                sh = result.bldg_sh
                vegshmat[:, :, index] = vegsh
                vbshvegshmat[:, :, index] = vbshvegsh
            else:
                sh = shadow.shadowingfunctionglobalradiation(dsm, azimuth, altitude, scale, 1)
            shmat[:, :, index] = sh

            # Calculate svfs
            for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.0)]) + 1):
                weight = annulus_weight(k, aziinterval[i]) * sh
                svf = svf + weight
                weight = annulus_weight(k, aziintervalaniso[i]) * sh
                if (azimuth >= 0) and (azimuth < 180):
                    svfE = svfE + weight
                if (azimuth >= 90) and (azimuth < 270):
                    svfS = svfS + weight
                if (azimuth >= 180) and (azimuth < 360):
                    svfW = svfW + weight
                if (azimuth >= 270) or (azimuth < 90):
                    svfN = svfN + weight

            if usevegdem == 1:
                for k in np.arange(annulino[int(i)] + 1, (annulino[int(i + 1.0)]) + 1):
                    # % changed to include 90
                    weight = annulus_weight(k, aziinterval[i])
                    svfveg = svfveg + weight * vegsh
                    svfaveg = svfaveg + weight * vbshvegsh
                    weight = annulus_weight(k, aziintervalaniso[i])
                    if (azimuth >= 0) and (azimuth < 180):
                        svfEveg = svfEveg + weight * vegsh
                        svfEaveg = svfEaveg + weight * vbshvegsh
                    if (azimuth >= 90) and (azimuth < 270):
                        svfSveg = svfSveg + weight * vegsh
                        svfSaveg = svfSaveg + weight * vbshvegsh
                    if (azimuth >= 180) and (azimuth < 360):
                        svfWveg = svfWveg + weight * vegsh
                        svfWaveg = svfWaveg + weight * vbshvegsh
                    if (azimuth >= 270) or (azimuth < 90):
                        svfNveg = svfNveg + weight * vegsh
                        svfNaveg = svfNaveg + weight * vbshvegsh

            index += 1

            # track progress
            progress.update(1)

    svfS = svfS + 3.0459e-004
    svfW = svfW + 3.0459e-004
    # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
    # %Forcing svf not be greater than 1 (some MATLAB crazyness)
    svf[(svf > 1.0)] = 1.0
    svfE[(svfE > 1.0)] = 1.0
    svfS[(svfS > 1.0)] = 1.0
    svfW[(svfW > 1.0)] = 1.0
    svfN[(svfN > 1.0)] = 1.0

    if usevegdem == 1:
        last = np.zeros((rows, cols))
        last[(vegdem2 == 0.0)] = 3.0459e-004
        svfSveg = svfSveg + last
        svfWveg = svfWveg + last
        svfSaveg = svfSaveg + last
        svfWaveg = svfWaveg + last
        # %Forcing svf not be greater than 1 (some MATLAB crazyness)
        svfveg[(svfveg > 1.0)] = 1.0
        svfEveg[(svfEveg > 1.0)] = 1.0
        svfSveg[(svfSveg > 1.0)] = 1.0
        svfWveg[(svfWveg > 1.0)] = 1.0
        svfNveg[(svfNveg > 1.0)] = 1.0
        svfaveg[(svfaveg > 1.0)] = 1.0
        svfEaveg[(svfEaveg > 1.0)] = 1.0
        svfSaveg[(svfSaveg > 1.0)] = 1.0
        svfWaveg[(svfWaveg > 1.0)] = 1.0
        svfNaveg[(svfNaveg > 1.0)] = 1.0

    svfresult = {
        "svf": svf,
        "svfE": svfE,
        "svfS": svfS,
        "svfW": svfW,
        "svfN": svfN,
        "svfveg": svfveg,
        "svfEveg": svfEveg,
        "svfSveg": svfSveg,
        "svfWveg": svfWveg,
        "svfNveg": svfNveg,
        "svfaveg": svfaveg,
        "svfEaveg": svfEaveg,
        "svfSaveg": svfSaveg,
        "svfWaveg": svfWaveg,
        "svfNaveg": svfNaveg,
        "shmat": shmat,
        "vegshmat": vegshmat,
        "vbshvegshmat": vbshvegshmat,
    }

    return svfresult
