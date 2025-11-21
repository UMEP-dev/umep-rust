# %%
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from umep import (
    common,
    wall_heightaspect_algorithm,
)
from umep.functions.SOLWEIGpython import solweig_runner_core
from umepr import solweig_runner_rust, svf

# working folder
input_folder = "demos/data/athens"
input_path = Path(input_folder).absolute()
input_path.mkdir(parents=True, exist_ok=True)
input_path_str = str(input_path)
# output folder
output_folder = "temp/athens"
output_folder_path = Path(output_folder).absolute()
output_folder_path.mkdir(parents=True, exist_ok=True)
output_folder_path_str = str(output_folder_path)
# extents
total_extents = [476800, 4205850, 477200, 4206250]

# %%
# buffer
working_crs = 2100
trees_gdf = gpd.read_file(input_folder + "/trees.gpkg")
trees_gdf = trees_gdf.to_crs(working_crs)
cdsm_rast, cdsm_transf = common.rasterise_gdf(
    trees_gdf,
    "geometry",
    "height",
    bbox=total_extents,
    pixel_size=1.0,
)
# add to DEM then set
common.save_raster(
    str(output_folder_path / "CDSM.tif"),
    cdsm_rast,
    cdsm_transf.to_gdal(),
    CRS.from_epsg(working_crs).to_wkt(),
)
# %%
# wall info for SOLWEIG
wall_heightaspect_algorithm.generate_wall_hts(
    dsm_path=input_path_str + "/DSM.tif",
    bbox=total_extents,
    out_dir=output_folder_path_str + "/walls",
)

# %%
# skyview factor for SOLWEIG
svf.generate_svf(
    dsm_path=input_path_str + "/DSM.tif",
    bbox=total_extents,
    out_dir=output_folder_path_str + "/svf",
    cdsm_path=output_folder_path_str + "/CDSM.tif",
    trans_veg_perc=3,
)

# %%
# skyview factor for SOLWEIG - Tiled
svf.generate_svf(
    dsm_path=input_path_str + "/DSM.tif",
    bbox=total_extents,
    out_dir=output_folder_path_str + "/svf_tiled",
    cdsm_path=output_folder_path_str + "/CDSM.tif",
    trans_veg_perc=3,
    use_tiled_loading=True,
    tile_size=200,
)

# %%
SRR = solweig_runner_rust.SolweigRunRust(
    "demos/data/athens/configsolweig.ini",
    "demos/data/athens/parametersforsolweig.json",
    use_tiled_loading=True,
    tile_size=200,
)
SRR.run()
"""
Running SOLWEIG: 100%|| 72/72 [00:57<00:00,  1.63step/s]
"""

# %%
SRC = solweig_runner_core.SolweigRunCore(
    "demos/data/athens/configsolweig.ini",
    "demos/data/athens/parametersforsolweig.json",
    use_tiled_loading=False,
)
# SRC.run()
"""
Running SOLWEIG: 100%|| 72/72 [04:49<00:00,  4.02s/step]
"""
