import numpy as np


def cylindric_wedge(zen, svfalfa, rows, cols):
    np.seterr(divide="ignore", invalid="ignore")

    # Fraction of sunlit walls based on sun altitude and svf wieghted building angles
    # input:
    # sun zenith angle "beta"
    # svf related angle "alfa"

    beta = zen
    alfa = np.zeros((rows, cols), dtype=np.float32) + svfalfa

    xa = 1 - 2.0 / (np.tan(alfa) * np.tan(beta))
    ha = 2.0 / (np.tan(alfa) * np.tan(beta))
    ba = 1.0 / np.tan(alfa)
    hkil = 2.0 * ba * ha

    qa = np.zeros((rows, cols), dtype=np.float32)
    qa[xa < 0] = np.tan(beta) / 2

    Za = np.zeros((rows, cols), dtype=np.float32)
    Za[xa < 0] = ((ba[xa < 0] ** 2) - ((qa[xa < 0] ** 2) / 4)) ** 0.5

    phi = np.zeros((rows, cols), dtype=np.float32)
    phi[xa < 0] = np.arctan(Za[xa < 0] / qa[xa < 0])

    A = np.zeros((rows, cols), dtype=np.float32)
    A[xa < 0] = (np.sin(phi[xa < 0]) - phi[xa < 0] * np.cos(phi[xa < 0])) / (1 - np.cos(phi[xa < 0]))

    ukil = np.zeros((rows, cols), dtype=np.float32)
    ukil[xa < 0] = 2 * ba[xa < 0] * xa[xa < 0] * A[xa < 0]

    Ssurf = hkil + ukil

    F_sh = (2 * np.pi * ba - Ssurf) / (2 * np.pi * ba)

    return F_sh


def cylindric_wedge_voxel(zen, svfalfa):
    np.seterr(divide="ignore", invalid="ignore")

    # Fraction of sunlit walls based on sun altitude and svf wieghted building angles
    # input:
    # sun zenith angle "beta"
    # svf related angle "alfa"

    beta = zen

    xa = 1 - 2.0 / (np.tan(svfalfa) * np.tan(beta))
    ha = 2.0 / (np.tan(svfalfa) * np.tan(beta))
    ba = 1.0 / np.tan(svfalfa)
    hkil = 2.0 * ba * ha

    qa = np.zeros((svfalfa.shape[0]), dtype=np.float32)
    qa[xa < 0] = np.tan(beta) / 2

    Za = np.zeros((svfalfa.shape[0]), dtype=np.float32)
    Za[xa < 0] = ((ba[xa < 0] ** 2) - ((qa[xa < 0] ** 2) / 4)) ** 0.5

    phi = np.zeros((svfalfa.shape[0]), dtype=np.float32)
    phi[xa < 0] = np.arctan(Za[xa < 0] / qa[xa < 0])

    A = np.zeros((svfalfa.shape[0]), dtype=np.float32)
    A[xa < 0] = (np.sin(phi[xa < 0]) - phi[xa < 0] * np.cos(phi[xa < 0])) / (1 - np.cos(phi[xa < 0]))

    ukil = np.zeros((svfalfa.shape[0]), dtype=np.float32)
    ukil[xa < 0] = 2 * ba[xa < 0] * xa[xa < 0] * A[xa < 0]

    Ssurf = hkil + ukil

    F_sh = (2 * np.pi * ba - Ssurf) / (2 * np.pi * ba)

    return F_sh
