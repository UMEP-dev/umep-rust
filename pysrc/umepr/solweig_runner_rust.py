"""
Subclasses SolweigRunCore - swaps in solweig function which calls Rust implementations of shadowing and GVF calculations
"""

from typing import Any

from umep.functions.SOLWEIGpython.solweig_runner_core import SolweigRunCore

from .functions.solweig import Solweig_2025a_calc as Solweig_2025a_calc_hybrid


class SolweigRunRust(SolweigRunCore):
    """Class to run the SOLWEIG algorithm with Rust optimisations."""

    def calc_solweig(
        self,
        iter: int,
        elvis: float,
        first: float,
        second: float,
        firstdaytime: float,
        timeadd: float,
        timestepdec: float,
        posture: Any,
    ):
        """
        Calculate SOLWEIG results for the given iteration.
        Uses variant with GVF and Shadows rust optimisations.
        """
        return Solweig_2025a_calc_hybrid(  # type: ignore
            iter,
            self.raster_data.dsm,
            self.raster_data.scale,
            self.raster_data.rows,
            self.raster_data.cols,
            self.svf_data.svf,
            self.svf_data.svf_north,
            self.svf_data.svf_west,
            self.svf_data.svf_east,
            self.svf_data.svf_south,
            self.svf_data.svf_veg,
            self.svf_data.svf_veg_north,
            self.svf_data.svf_veg_east,
            self.svf_data.svf_veg_south,
            self.svf_data.svf_veg_west,
            self.svf_data.svf_veg_blocks_bldg_sh,
            self.svf_data.svf_veg_blocks_bldg_sh_east,
            self.svf_data.svf_veg_blocks_bldg_sh_south,
            self.svf_data.svf_veg_blocks_bldg_sh_west,
            self.svf_data.svf_veg_blocks_bldg_sh_north,
            self.raster_data.cdsm,
            self.raster_data.tdsm,
            self.params.Albedo.Effective.Value.Walls,
            self.params.Tmrt_params.Value.absK,
            self.params.Tmrt_params.Value.absL,
            self.params.Emissivity.Value.Walls,
            posture.Fside,
            posture.Fup,
            posture.Fcyl,
            self.environ_data.altitude[iter],
            self.environ_data.azimuth[iter],
            self.environ_data.zen[iter],
            self.environ_data.jday[iter],
            self.config.use_veg_dem,
            self.config.only_global,
            self.raster_data.buildings,
            self.location,
            self.environ_data.psi[iter],
            self.config.use_landcover,
            self.raster_data.lcgrid,
            self.environ_data.dectime[iter],
            self.environ_data.altmax[iter],
            self.raster_data.wallaspect,
            self.raster_data.wallheight,
            int(self.config.person_cylinder),  # expects int though should work either way
            elvis,
            self.environ_data.Ta[iter],
            self.environ_data.RH[iter],
            self.environ_data.radG[iter],
            self.environ_data.radD[iter],
            self.environ_data.radI[iter],
            self.environ_data.P[iter],
            self.raster_data.amaxvalue,
            self.raster_data.bush,
            self.environ_data.Twater[iter],
            self.tg_maps.TgK,
            self.tg_maps.Tstart,
            self.tg_maps.alb_grid,
            self.tg_maps.emis_grid,
            self.tg_maps.TgK_wall,
            self.tg_maps.Tstart_wall,
            self.tg_maps.TmaxLST,
            self.tg_maps.TmaxLST_wall,
            first,
            second,
            self.svf_data.svfalfa,
            self.raster_data.svfbuveg,
            firstdaytime,
            timeadd,
            timestepdec,
            self.tg_maps.Tgmap1,
            self.tg_maps.Tgmap1E,
            self.tg_maps.Tgmap1S,
            self.tg_maps.Tgmap1W,
            self.tg_maps.Tgmap1N,
            self.environ_data.CI[iter],
            self.tg_maps.TgOut1,
            self.shadow_mats.diffsh,
            self.shadow_mats.shmat,
            self.shadow_mats.vegshmat,
            self.shadow_mats.vbshvegshmat,
            int(self.config.use_aniso),  # expects int though should work either way
            self.shadow_mats.asvf,
            self.shadow_mats.patch_option,
            self.walls_data.voxelMaps,
            self.walls_data.voxelTable,
            self.environ_data.Ws[iter],
            self.config.use_wall_scheme,
            self.walls_data.timeStep,
            self.shadow_mats.steradians,
            self.walls_data.walls_scheme,
            self.walls_data.dirwalls_scheme,
        )
