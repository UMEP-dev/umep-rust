# %%
"""
Demo: Small neighbourhood wall height/aspect generation

This demo shows how to use the solweig package for wall processing.
"""

from pathlib import Path

import solweig

#
bbox = [789700, 784130, 790100, 784470]
working_folder = "temp/demos/small_nbhd"
pixel_resolution = 1  # metres
working_crs = 32651

working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)
working_path_str = str(working_path)

# %%
dsm_path = Path("demos/data/small_nbhd/dsm_clipped.tif").absolute()
solweig.walls.generate_wall_hts(
    dsm_path=str(dsm_path),
    bbox=bbox,
    out_dir=working_path_str + "/walls",
)

# %%
