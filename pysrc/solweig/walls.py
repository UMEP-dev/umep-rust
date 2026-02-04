"""
Wall height and aspect generation from DSM.

This algorithm identifies wall pixels and their height from ground and building
digital surface models (DSM) using filters as presented by Lindberg et al. (2015a).
Wall aspect is estimated using a specific linear filter as presented by
Goodwin et al. (1999) and further developed by Lindberg et al. (2015b).

References:
- Goodwin NR, Coops NC, Tooke TR, Christen A, Voogt JA (2009)
  Characterizing urban surface cover and structure with airborne lidar technology.
  Can J Remote Sens 35:297–309
- Lindberg F., Grimmond, C.S.B. and Martilli, A. (2015a)
  Sunlit fractions on urban facets - Impact of spatial resolution and approach
  Urban Climate DOI: 10.1016/j.uclim.2014.11.006
- Lindberg F., Jonsson, P. & Honjo, T. and Wästberg, D. (2015b)
  Solar energy on building envelopes - 3D modelling in a 2D environment
  Solar Energy 115 369–378
"""

from __future__ import annotations

from pathlib import Path

from . import io as common
from .algorithms import wallalgorithms as wa


def generate_wall_hts(
    dsm_path: str,
    bbox: list[int] | None,
    out_dir: str,
    wall_limit: float = 1,
):
    """
    Generate wall height and aspect rasters from a DSM.

    Args:
        dsm_path: Path to the Digital Surface Model raster
        bbox: Bounding box [minx, miny, maxx, maxy] or None for full extent
        out_dir: Output directory for wall_hts.tif and wall_aspects.tif
        wall_limit: Minimum height to be considered a wall (default: 1m)

    Outputs:
        wall_hts.tif: Wall heights in meters
        wall_aspects.tif: Wall aspect in degrees (0 = North)
    """
    dsm_rast, dsm_transf, dsm_crs, _dsm_nd = common.load_raster(dsm_path, bbox, coerce_f64_to_f32=True)
    dsm_scale = 1 / dsm_transf[1]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_str = str(out_path)

    walls = wa.findwalls(dsm_rast, wall_limit)
    common.save_raster(out_path_str + "/" + "wall_hts.tif", walls, dsm_transf, dsm_crs, coerce_f64_to_f32=True)

    dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, dsm_scale, dsm_rast)
    common.save_raster(out_path_str + "/" + "wall_aspects.tif", dirwalls, dsm_transf, dsm_crs, coerce_f64_to_f32=True)
