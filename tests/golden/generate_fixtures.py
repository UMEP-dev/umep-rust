"""
Golden Fixture Generator

Run this script once to generate golden fixtures from the Athens demo data.
These fixtures serve as regression reference points during modernization.

IMPORTANT: These fixtures are generated using the **original UMEP Python module**
as ground truth. This ensures we have a neutral reference that doesn't change
during Rust modernization. The tests then verify that Rust matches UMEP Python.

Usage:
    uv run python tests/golden/generate_fixtures.py
"""

from pathlib import Path

import numpy as np
from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore
from umep.functions.svf_functions import svfForProcessing153
from umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import (
    shadowingfunction_wallheight_23,
)

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_PATH = "tests/rustalgos/test_config_shadows.ini"
PARAMS_PATH = "tests/rustalgos/test_params_solweig.json"


def ensure_fixtures_dir():
    """Create fixtures directory if it doesn't exist."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_shadow_fixtures():
    """Generate golden fixtures for shadow calculations using UMEP Python."""
    print("Generating shadow fixtures (using UMEP Python as ground truth)...")

    # Load demo data using existing test infrastructure
    SWC = SolweigRunCore(
        config_path_str=CONFIG_PATH,
        params_json_path=PARAMS_PATH,
    )

    dsm = SWC.raster_data.dsm.astype(np.float32)
    cdsm = SWC.raster_data.cdsm.astype(np.float32)
    tdsm = SWC.raster_data.tdsm.astype(np.float32)
    bush = SWC.raster_data.bush.astype(np.float32)
    wall_ht = SWC.raster_data.wallheight.astype(np.float32)
    wall_asp = (SWC.raster_data.wallaspect * np.pi / 180.0).astype(np.float32)

    # Test with multiple sun positions
    sun_positions = [
        {"name": "morning", "azimuth": 90.0, "altitude": 30.0},
        {"name": "noon", "azimuth": 180.0, "altitude": 60.0},
        {"name": "afternoon", "azimuth": 270.0, "altitude": 45.0},
    ]

    for pos in sun_positions:
        print(f"  Computing shadows for {pos['name']}...")
        # Use UMEP Python shadowingfunction_wallheight_23
        # Returns: (veg_sh, bldg_sh, veg_blocks_bldg_sh, wall_sh, wall_sun,
        #           wall_sh_veg, face_sh, face_sun)
        (
            veg_sh,
            bldg_sh,
            _veg_blocks_bldg_sh,
            wall_sh,
            wall_sun,
            _wall_sh_veg,
            _face_sh,
            _face_sun,
        ) = shadowingfunction_wallheight_23(
            dsm,
            cdsm,
            tdsm,
            pos["azimuth"],
            pos["altitude"],
            SWC.raster_data.scale,
            SWC.raster_data.amaxvalue,
            bush,
            wall_ht,
            wall_asp,
        )

        # Save each shadow component
        prefix = f"shadow_{pos['name']}"
        np.save(FIXTURES_DIR / f"{prefix}_bldg_sh.npy", np.array(bldg_sh))
        np.save(FIXTURES_DIR / f"{prefix}_veg_sh.npy", np.array(veg_sh))
        np.save(FIXTURES_DIR / f"{prefix}_wall_sh.npy", np.array(wall_sh))
        np.save(FIXTURES_DIR / f"{prefix}_wall_sun.npy", np.array(wall_sun))

    # Save input metadata for reproducibility
    np.savez(
        FIXTURES_DIR / "shadow_metadata.npz",
        dsm_shape=dsm.shape,
        scale=SWC.raster_data.scale,
        amaxvalue=SWC.raster_data.amaxvalue,
    )

    print("  Shadow fixtures saved.")


def generate_svf_fixtures():
    """Generate golden fixtures for SVF calculations using UMEP Python."""
    print("Generating SVF fixtures (using UMEP Python as ground truth)...")

    SWC = SolweigRunCore(
        config_path_str=CONFIG_PATH,
        params_json_path=PARAMS_PATH,
    )

    dsm = SWC.raster_data.dsm.astype(np.float32)
    cdsm = SWC.raster_data.cdsm.astype(np.float32)
    tdsm = SWC.raster_data.tdsm.astype(np.float32)

    print("  Computing SVF (this may take a moment)...")
    # Use UMEP Python svfForProcessing153
    # Returns a dictionary with keys: svf, svfE, svfS, svfW, svfN, svfveg, etc.
    result = svfForProcessing153(
        dsm,
        cdsm,
        tdsm,
        SWC.raster_data.scale,
        1,  # usevegdem (1 = True)
        SWC.raster_data.amaxvalue,
    )

    # Save SVF components (mapping UMEP Python keys to fixture names)
    np.save(FIXTURES_DIR / "svf_total.npy", np.array(result["svf"]))
    np.save(FIXTURES_DIR / "svf_north.npy", np.array(result["svfN"]))
    np.save(FIXTURES_DIR / "svf_east.npy", np.array(result["svfE"]))
    np.save(FIXTURES_DIR / "svf_south.npy", np.array(result["svfS"]))
    np.save(FIXTURES_DIR / "svf_west.npy", np.array(result["svfW"]))
    np.save(FIXTURES_DIR / "svf_veg.npy", np.array(result["svfveg"]))

    # Save metadata
    np.savez(
        FIXTURES_DIR / "svf_metadata.npz",
        dsm_shape=dsm.shape,
        scale=SWC.raster_data.scale,
        amaxvalue=SWC.raster_data.amaxvalue,
    )

    print("  SVF fixtures saved.")


def generate_input_fixtures():
    """Save input data as fixtures for test isolation."""
    print("Generating input fixtures...")

    SWC = SolweigRunCore(
        config_path_str=CONFIG_PATH,
        params_json_path=PARAMS_PATH,
    )

    # Save input rasters
    np.save(FIXTURES_DIR / "input_dsm.npy", SWC.raster_data.dsm.astype(np.float32))
    np.save(FIXTURES_DIR / "input_cdsm.npy", SWC.raster_data.cdsm.astype(np.float32))
    np.save(FIXTURES_DIR / "input_tdsm.npy", SWC.raster_data.tdsm.astype(np.float32))
    np.save(FIXTURES_DIR / "input_bush.npy", SWC.raster_data.bush.astype(np.float32))
    np.save(FIXTURES_DIR / "input_wall_ht.npy", SWC.raster_data.wallheight.astype(np.float32))
    np.save(FIXTURES_DIR / "input_wall_asp.npy", SWC.raster_data.wallaspect.astype(np.float32))

    # Save scalar parameters
    np.savez(
        FIXTURES_DIR / "input_params.npz",
        scale=SWC.raster_data.scale,
        amaxvalue=SWC.raster_data.amaxvalue,
    )

    print("  Input fixtures saved.")


def main():
    """Generate all golden fixtures using UMEP Python as ground truth."""
    print("=" * 60)
    print("Golden Fixture Generator")
    print("=" * 60)
    print()
    print("IMPORTANT: These fixtures are generated using the original")
    print("UMEP Python module as ground truth. This ensures a neutral")
    print("reference that doesn't change during Rust modernization.")
    print()
    print("The golden tests verify that Rust implementations match")
    print("the UMEP Python reference outputs.")
    print("=" * 60)

    ensure_fixtures_dir()

    # Generate input fixtures first (for test isolation)
    generate_input_fixtures()

    # Generate algorithm output fixtures (using UMEP Python)
    generate_shadow_fixtures()
    generate_svf_fixtures()

    print("\n" + "=" * 60)
    print("All fixtures generated successfully!")
    print(f"Location: {FIXTURES_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIXTURES_DIR.glob("*.npy")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")
    for f in sorted(FIXTURES_DIR.glob("*.npz")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
