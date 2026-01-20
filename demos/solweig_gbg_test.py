# %%
"""
Demo: Gothenburg SOLWEIG preprocessing

This demo shows how to use the solweig package for:
1. Wall height and aspect generation
2. Sky View Factor (SVF) calculation

Note: Full SOLWEIG model run requires additional porting work.
"""
from pathlib import Path

import solweig

# %%
# Working folder and input files
working_folder = "temp/goteborg"
working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)
working_path_str = str(working_path)

# Input files
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"

# Setup parameters
trans_veg_perc = 3
trunk_ratio_perc = 25

# %%
# Wall info for SOLWEIG (height and aspect)
print("Generating wall heights and aspects...")
solweig.walls.generate_wall_hts(
    dsm_path=dsm_path,
    bbox=None,  # Full extent
    out_dir=working_path_str + "/walls",
)
print(f"  Output: {working_path_str}/walls/wall_hts.tif")
print(f"  Output: {working_path_str}/walls/wall_aspects.tif")

# %%
# Sky View Factor for SOLWEIG
print("\nGenerating SVF (this may take a while for large rasters)...")
solweig.svf.generate_svf(
    dsm_path=dsm_path,
    bbox=None,  # Full extent
    out_dir=working_path_str + "/svf",
    cdsm_path=cdsm_path,
    trans_veg_perc=trans_veg_perc,
    trunk_ratio_perc=trunk_ratio_perc,
)
print(f"  Output: {working_path_str}/svf/")

# %%
print("\nPreprocessing complete!")
print(f"GPU acceleration: {'enabled' if solweig.GPU_ENABLED else 'disabled'}")

# %%
# TODO: Full SOLWEIG run
# The full SOLWEIG model runner is not yet fully ported to solweig package.
# For now, you can use umep.functions.SOLWEIGpython.Solweig_run if umep is installed:
#
# from umep.functions.SOLWEIGpython import Solweig_run as sr
# sr.solweig_run("demos/data/Goteborg_SWEREF99_1200/configsolweig.ini", feedback=None)
