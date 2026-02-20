# %%
"""
Demo: Wall height/aspect generation for a small neighbourhood.

Generates wall height and wall aspect rasters from a DSM GeoTIFF.
The outputs (wall_hts.tif, wall_aspects.tif) are saved into a ``walls/``
subdirectory and can be loaded by ``SurfaceData.prepare()`` for subsequent
SOLWEIG calculations.

Inputs:
    - DSM GeoTIFF (``demos/data/small_nbhd/dsm_clipped.tif``)

Outputs (written to ``temp/demos/small_nbhd/walls/``):
    - ``wall_hts.tif``  — wall pixel heights in metres
    - ``wall_aspects.tif`` — wall pixel aspect angles in degrees (0 = N)
"""

from pathlib import Path

import solweig

# Bounding box in the DSM's projected CRS [minx, miny, maxx, maxy].
# This clips the DSM to the area of interest before wall detection.
bbox = [789700, 784130, 790100, 784470]

working_folder = "temp/demos/small_nbhd"

working_path = Path(working_folder).absolute()
working_path.mkdir(parents=True, exist_ok=True)

# %%
# Generate wall heights and aspects from the DSM.
# The function writes wall_hts.tif and wall_aspects.tif into out_dir.
dsm_path = Path("demos/data/small_nbhd/dsm_clipped.tif").absolute()
solweig.walls.generate_wall_hts(
    dsm_path=str(dsm_path),
    bbox=bbox,
    out_dir=str(working_path / "walls"),
)

# %%
