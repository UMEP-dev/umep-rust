"""Regression tests for SkyviewRunner.calculate_svf_core."""

import numpy as np
from solweig import rustalgos


def _max_height(dsm: np.ndarray, cdsm: np.ndarray) -> float:
    h = float(np.nanmax(np.maximum(dsm, cdsm)) - np.nanmin(dsm))
    return h if h > 0 else 1.0


def _assert_core_match(
    full_result,
    core_result,
    rs: slice,
    cs: slice,
    *,
    use_veg: bool,
) -> None:
    svf_fields = ["svf", "svf_north", "svf_east", "svf_south", "svf_west"]
    veg_fields = [
        "svf_veg",
        "svf_veg_north",
        "svf_veg_east",
        "svf_veg_south",
        "svf_veg_west",
        "svf_veg_blocks_bldg_sh",
        "svf_veg_blocks_bldg_sh_north",
        "svf_veg_blocks_bldg_sh_east",
        "svf_veg_blocks_bldg_sh_south",
        "svf_veg_blocks_bldg_sh_west",
    ]

    for name in svf_fields:
        full = np.asarray(getattr(full_result, name))[rs, cs]
        core = np.asarray(getattr(core_result, name))
        np.testing.assert_allclose(core, full, atol=0.0, rtol=0.0, err_msg=f"Mismatch in {name}")

    if use_veg:
        for name in veg_fields:
            full = np.asarray(getattr(full_result, name))[rs, cs]
            core = np.asarray(getattr(core_result, name))
            np.testing.assert_allclose(core, full, atol=0.0, rtol=0.0, err_msg=f"Mismatch in {name}")

    sh_fields = ["bldg_sh_matrix", "veg_sh_matrix", "veg_blocks_bldg_sh_matrix"]
    for name in sh_fields:
        full = np.asarray(getattr(full_result, name))[rs, cs, :]
        core = np.asarray(getattr(core_result, name))
        np.testing.assert_array_equal(core, full, err_msg=f"Mismatch in {name}")


def test_svf_core_matches_full_without_vegetation():
    rng = np.random.default_rng(42)
    rows, cols = 80, 96
    dsm = (rng.random((rows, cols), dtype=np.float32) * 25.0).astype(np.float32)
    veg = np.zeros_like(dsm, dtype=np.float32)
    trunk = np.zeros_like(dsm, dtype=np.float32)

    runner = rustalgos.skyview.SkyviewRunner()
    max_h = _max_height(dsm, veg)

    full = runner.calculate_svf(dsm, veg, trunk, 1.0, False, max_h, 2, 3.0)
    core = runner.calculate_svf_core(
        dsm,
        veg,
        trunk,
        1.0,
        False,
        max_h,
        2,
        3.0,
        7,
        71,
        11,
        83,
    )
    _assert_core_match(full, core, slice(7, 71), slice(11, 83), use_veg=False)


def test_svf_core_matches_full_with_vegetation():
    rng = np.random.default_rng(7)
    rows, cols = 72, 88
    dsm = (rng.random((rows, cols), dtype=np.float32) * 20.0).astype(np.float32)
    canopy_rel = (rng.random((rows, cols), dtype=np.float32) * 8.0).astype(np.float32)
    trunk_rel = (canopy_rel * 0.25).astype(np.float32)
    canopy_abs = (dsm + canopy_rel).astype(np.float32)
    trunk_abs = (dsm + trunk_rel).astype(np.float32)

    runner = rustalgos.skyview.SkyviewRunner()
    max_h = _max_height(dsm, canopy_abs)

    full = runner.calculate_svf(dsm, canopy_abs, trunk_abs, 1.0, True, max_h, 2, 3.0)
    core = runner.calculate_svf_core(
        dsm,
        canopy_abs,
        trunk_abs,
        1.0,
        True,
        max_h,
        2,
        3.0,
        5,
        68,
        9,
        79,
    )
    _assert_core_match(full, core, slice(5, 68), slice(9, 79), use_veg=True)
