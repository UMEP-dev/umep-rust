# %%
from pathlib import Path

from umep import (
    skyviewfactor_algorithm,
    wall_heightaspect_algorithm,
)
from umep.functions.SOLWEIGpython import Solweig_run as sr

# %%
bbox = [476070, 4203550, 477110, 4204330]
working_folder = "temp/gothenburg"
pixel_resolution = 1  # metres
working_crs = 3007

working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)
working_path_str = str(working_path)

# input files for computing
dsm_path = "demos/data/Goteborg_SWEREF99_1200/DSM_KRbig.tif"
cdsm_path = "demos/data/Goteborg_SWEREF99_1200/CDSM_KRbig.tif"
lc_path = ""

# setup parameters
trans_veg = 3
trunk_ratio = 0.25

# %%
# wall info for SOLWEIG (height and aspect)
if not Path.exists(working_path / "walls"):
    wall_heightaspect_algorithm.generate_wall_hts(
        dsm_path=dsm_path,
        bbox=None,
        out_dir=working_path_str + "/walls",
    )

# %%
# skyview factor for SOLWEIG
if not Path.exists(working_path / "svf"):
    skyviewfactor_algorithm.generate_svf(
        dsm_path=dsm_path,
        bbox=None,
        out_dir=working_path_str + "/svf",
        cdsm_path=cdsm_path,
        trans_veg=trans_veg,
        trunk_ratio=trunk_ratio,
    )

# %%
sr.solweig_run("demos/data/Goteborg_SWEREF99_1200/configsolweig.ini", feedback=None)

# %%
