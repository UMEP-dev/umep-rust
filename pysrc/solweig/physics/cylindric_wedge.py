import numpy as np

from ..constants import MIN_SUN_ELEVATION_DEG

# Convert to radians for internal use
_MIN_SUN_ALTITUDE_RAD = MIN_SUN_ELEVATION_DEG * (np.pi / 180.0)


def cylindric_wedge(zen, svfalfa, rows, cols):
    """
    Fraction of sunlit walls based on sun altitude and SVF-weighted building angles.

    Args:
        zen: Sun zenith angle (radians)
        svfalfa: SVF-related angle grid (2D array, radians)
        rows, cols: Grid dimensions (unused, kept for API compatibility)

    Returns:
        F_sh: Shadow fraction grid (0 = fully sunlit, 1 = fully shaded)

    Note:
        At very low sun altitudes (< 3°), returns F_sh = 1.0 to avoid
        numerical instability from tan(zen) approaching infinity.
    """
    # Guard against low sun angles where tan(zen) → infinity
    # zenith = 90° - altitude, so zen > 87° means altitude < 3°
    altitude_rad = (np.pi / 2.0) - zen
    if altitude_rad < _MIN_SUN_ALTITUDE_RAD:
        # Sun too low - walls fully shaded
        return np.ones_like(svfalfa, dtype=np.float32)

    # Pre-compute trigonometric values once (1.7x speedup)
    tan_zen = np.tan(zen)
    tan_alfa = np.tan(svfalfa)

    # Guard against very small tan_alfa (near-horizontal surfaces)
    tan_alfa = np.maximum(tan_alfa, 1e-6)

    ba = 1.0 / tan_alfa
    tan_product = tan_alfa * tan_zen

    # Guard against division by very small values
    tan_product = np.maximum(tan_product, 1e-6)

    xa = 1 - 2.0 / tan_product
    ha = 2.0 / tan_product
    hkil = 2.0 * ba * ha

    # Use np.where for vectorized conditionals (avoids index assignment overhead)
    mask = xa < 0
    qa = np.where(mask, tan_zen / 2, 0.0).astype(np.float32)

    # Compute Za with safe sqrt
    ba_sq = ba**2
    Za_sq = np.maximum(ba_sq - (qa**2) / 4, 0)
    Za = np.where(mask, np.sqrt(Za_sq), 0.0).astype(np.float32)

    # Safe arctan (avoid division by zero)
    phi = np.where(mask & (qa > 1e-10), np.arctan(Za / np.maximum(qa, 1e-10)), 0.0).astype(np.float32)

    # Compute A with safe denominator
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    denom = np.maximum(1 - cos_phi, 1e-10)
    A = np.where(mask, (sin_phi - phi * cos_phi) / denom, 0.0).astype(np.float32)

    ukil = np.where(mask, 2 * ba * xa * A, 0.0).astype(np.float32)

    Ssurf = hkil + ukil
    F_sh = (2 * np.pi * ba - Ssurf) / (2 * np.pi * ba)

    return F_sh.astype(np.float32)


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
