# %%
from importlib import reload
from pathlib import Path

from umep import (
    solweig_algorithm,
    wall_heightaspect_algorithm,
)

reload(solweig_algorithm)

#
bbox = [789700, 784130, 790100, 784470]
working_folder = "temp/tests/small_nbhd"
pixel_resolution = 1  # metres
working_crs = 32651

working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)
working_path_str = str(working_path)

# %%
dsm_path = Path("demos/data/small_nbhd/dsm_clipped.tif").absolute()
# if not Path.exists(working_path / "walls"):
wall_heightaspect_algorithm.generate_wall_hts(
    dsm_path=str(dsm_path),
    bbox=bbox,
    out_dir=working_path_str + "/walls",
)

# %%
