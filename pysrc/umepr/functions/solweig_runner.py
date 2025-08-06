import json
import os
import zipfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from umep.functions import wallalgorithms as wa
from umep.functions.SOLWEIGpython import PET_calculations
from umep.functions.SOLWEIGpython import UTCI_calculations as utci
from umep.functions.SOLWEIGpython.CirclePlotBar import PolarBarPlot
from umep.functions.SOLWEIGpython.patch_characteristics import hemispheric_image
from umep.functions.SOLWEIGpython.wall_surface_temperature import load_walls
from umep.functions.SOLWEIGpython.wallsAsNetCDF import walls_as_netcdf
from umep.util.SEBESOLWEIGCommonFiles.clearnessindex_2013b import clearnessindex_2013b
from umep.util.SEBESOLWEIGCommonFiles.Solweig_v2015_metdata_noload import Solweig_2015a_metdata_noload

from .. import common
from ..class_configs import SolweigConfig, SvfData, WeatherData
from ..hybrid.solweig import Solweig_2025a_calc as Solweig_2025a_calc_hybrid
from ..original.solweig import Solweig_2025a_calc


def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


def Tgmaps_v1(lc_grid, params):
    # Tgmaps_v1 Populates grids with cooeficients for Tg wave
    #   Detailed explanation goes here
    lc_grid[lc_grid >= 100] = 2
    id = np.unique(lc_grid)
    id = lc_grid[lc_grid <= 7].astype(int)
    TgK = np.copy(lc_grid)
    Tstart = np.copy(lc_grid)
    alb_grid = np.copy(lc_grid)
    emis_grid = np.copy(lc_grid)
    TmaxLST = np.copy(lc_grid)

    for i in id:
        name = getattr(params.Names.Value, str(i), None)
        # row = (lc_class[:, 0] == id[i])
        Tstart[Tstart == i] = getattr(params.Tstart.Value, name, None)
        alb_grid[alb_grid == i] = getattr(params.Albedo.Effective.Value, name, None)
        emis_grid[emis_grid == i] = getattr(params.Emissivity.Value, name, None)
        TmaxLST[TmaxLST == i] = getattr(params.TmaxLST.Value, name, None)
        TgK[TgK == i] = getattr(params.Ts_deg.Value, name, None)

    TgK_wall = getattr(params.Ts_deg.Value, "Walls", None)
    Tstart_wall = getattr(params.Tstart.Value, "Walls", None)
    TmaxLST_wall = getattr(params.TmaxLST.Value, "Walls", None)

    return TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall


class SolweigRun:
    """Class to run the SOLWEIG algorithm with given configuration."""

    def __init__(self, config: SolweigConfig, params_json_path: str, qgis_env: bool = True):
        """Initialize the SOLWEIG runner with configuration and parameters."""
        self.config = config
        self.config.validate()
        # Progress tracking settings
        self.progress = None
        self.iters_total: int = 0
        self.iters_count: int = 0
        self.qgis_env = qgis_env
        # Initialize POI data
        self.poi_names: list[Any] = []
        self.poi_pixel_xys: np.ndarray | None = None
        self.poi_results = []
        # Initialize WOI data
        self.woi_names: list[Any] = []
        self.woi_pixel_xys: np.ndarray | None = None
        self.woi_results = []
        # Load parameters from JSON file
        params_path = common.check_path(params_json_path)
        try:
            with open(params_path) as f:
                params_dict = json.load(f)
                self.params = dict_to_namespace(params_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters from {params_json_path}: {e}")

    def prep_progress(self, num: int) -> None:
        """Prepare progress for environment."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def iter_progress(self) -> bool:
        """Iterate progress ."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_epw_weather(self) -> WeatherData:
        """Load weather data from an EPW file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_met_weather(self, header_rows: int = 1, delim: str = " ") -> WeatherData:
        """Load weather data from a MET file."""
        met_data = np.loadtxt(self.config.met_path, skiprows=header_rows, delimiter=delim)
        return WeatherData(
            DOY=met_data[:, 1],
            hours=met_data[:, 2],
            minu=met_data[:, 3],
            Ta=met_data[:, 11],
            RH=met_data[:, 10],
            radG=met_data[:, 14],
            radD=met_data[:, 21],
            radI=met_data[:, 22],
            P=met_data[:, 12],
            Ws=met_data[:, 9],
        )

    def load_poi_data(self, trf_arr: list[float]) -> None:
        """Load point of interest (POI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_poi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save results for points of interest (POIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def load_woi_data(self, trf_arr: list[float]) -> None:
        """Load wall of interest (WOI) data from a file."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def save_woi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save results for walls of interest (WOIs) to files."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def run(self) -> None:
        """Run the SOLWEIG algorithm."""
        # Load DSM
        dsm_arr, dsm_trf_arr, dsm_crs_wkt, dsm_nd_val = common.load_raster(self.config.dsm_path, bbox=None)
        scale = 1 / dsm_trf_arr[1]
        left_x = dsm_trf_arr[0]
        top_y = dsm_trf_arr[3]
        lng, lat = common.xy_to_lnglat(dsm_crs_wkt, left_x, top_y)
        rows = dsm_arr.shape[0]
        cols = dsm_arr.shape[1]

        dsm_arr[dsm_arr == dsm_nd_val] = 0.0
        if dsm_arr.min() < 0:
            dsmraise = np.abs(dsm_arr.min())
            dsm_arr = dsm_arr + dsmraise
        else:
            dsmraise = 0

        alt = np.median(dsm_arr)
        if alt < 0:
            alt = 3

        # Vegetation
        transVeg = self.params.Tree_settings.Value.Transmissivity
        trunkratio = self.params.Tree_settings.Value.Trunk_ratio
        if self.config.use_veg_dem:
            vegdsm, _, _, _ = common.load_raster(self.config.cdsm_path, bbox=None)
            if self.config.tdsm_path:
                vegdsm2, _, _, _ = common.load_raster(self.config.tdsm_path, bbox=None)
            else:
                vegdsm2 = vegdsm * trunkratio
        else:
            vegdsm = None
            vegdsm2 = None

        # Land cover
        if self.config.use_landcover:
            lcgrid, _, _, _ = common.load_raster(self.config.lc_path, bbox=None)
        else:
            lcgrid = None

        # DEM for buildings
        if self.config.use_dem_for_buildings:
            dem, _, _, dem_nd_val = common.load_raster(self.config.dem_path, bbox=None)
            dem[dem == dem_nd_val] = 0.0
            if dem.min() < 0:
                demraise = np.abs(dem.min())
                dem = dem + demraise
            else:
                demraise = 0

        # SVF
        with zipfile.ZipFile(self.config.svf_path, "r") as zip_ref:
            zip_ref.extractall(self.config.working_dir)
        # Load SVF data
        svf_data = SvfData(self.config.working_dir, self.config.use_veg_dem)
        tmp = svf_data.svf + svf_data.svf_veg - 1.0
        tmp[tmp < 0.0] = 0.0
        svfalfa = np.arcsin(np.exp(np.log(1.0 - tmp) / 2.0))

        wallheight, _, _, _ = common.load_raster(self.config.wh_path, bbox=None)
        wallaspect, _, _, _ = common.load_raster(self.config.wa_path, bbox=None)

        # weather data
        if self.config.use_epw_file:
            weather_data = self.load_epw_weather()
        else:
            weather_data = self.load_met_weather(header_rows=1, delim=" ")

        location = {"longitude": lng, "latitude": lat, "altitude": alt}
        YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(
            weather_data.to_array(), location, int(self.config.utc)
        )

        # POIs check
        if self.config.poi_path:
            self.load_poi_data(dsm_trf_arr)

        # Posture settings
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value

        Fside, Fup, height, Fcyl = posture.Fside, posture.Fup, posture.height, posture.Fcyl

        # Radiative surface influence
        first = np.round(height)
        if first == 0.0:
            first = 1.0
        second = np.round(height * 20.0)

        if self.config.use_veg_dem:
            # Conifer or deciduous
            if self.config.conifer:
                leafon = np.ones((1, weather_data.DOY.shape[0]))
            else:
                leafon = np.zeros((1, weather_data.DOY.shape[0]))
                if self.params.Tree_settings.Value.First_day_leaf > self.params.Tree_settings.Value.Last_day_leaf:
                    leaf_bool = (self.params.Tree_settings.Value.First_day_leaf < weather_data.DOY) | (
                        self.params.Tree_settings.Value.Last_day_leaf > weather_data.DOY
                    )
                else:
                    leaf_bool = (self.params.Tree_settings.Value.First_day_leaf < weather_data.DOY) & (
                        self.params.Tree_settings.Value.Last_day_leaf > weather_data.DOY
                    )
                leafon[0, leaf_bool] = 1

            # % Vegetation transmittivity of shortwave radiation
            psi = leafon * transVeg
            psi[leafon == 0] = 0.5
            # amaxvalue
            vegmax = vegdsm.max()
            amaxvalue = dsm_arr.max() - dsm_arr.min()
            amaxvalue = np.maximum(amaxvalue, vegmax)

            # Elevation vegdsms if buildingDEM includes ground heights
            vegdsm = vegdsm + dsm_arr
            vegdsm[vegdsm == dsm_arr] = 0
            vegdsm2 = vegdsm2 + dsm_arr
            vegdsm2[vegdsm2 == dsm_arr] = 0

            # % Bush separation
            bush = np.logical_not(vegdsm2 * vegdsm) * vegdsm

            svfbuveg = svf_data.svf - (1.0 - svf_data.svf_veg) * (1.0 - transVeg)  # % major bug fixed 20141203
        else:
            psi = leafon * 0.0 + 1.0
            svfbuveg = svf_data.svf
            bush = np.zeros([rows, cols])
            amaxvalue = 0

        # Initialization of maps
        Knight = np.zeros((rows, cols))
        Tgmap1 = np.zeros((rows, cols))
        Tgmap1E = np.zeros((rows, cols))
        Tgmap1S = np.zeros((rows, cols))
        Tgmap1W = np.zeros((rows, cols))
        Tgmap1N = np.zeros((rows, cols))

        # Create building boolean raster from either land cover or height rasters
        if not self.config.use_dem_for_buildings:
            buildings = np.copy(lcgrid)
            buildings[buildings == 7] = 1
            buildings[buildings == 6] = 1
            buildings[buildings == 5] = 1
            buildings[buildings == 4] = 1
            buildings[buildings == 3] = 1
            buildings[buildings == 2] = 0
        else:
            buildings = dsm_arr - dem
            buildings[buildings < 2.0] = 1.0
            buildings[buildings >= 2.0] = 0.0

        if self.config.save_buildings:
            common.save_raster(
                self.config.output_dir + "/buildings.tif",
                buildings,
                dsm_trf_arr,
                dsm_crs_wkt,
                dsm_nd_val,
            )

        # Import shadow matrices (Anisotropic sky)
        if self.config.use_aniso:
            data = np.load(self.config.aniso_path)
            shmat = data["shadowmat"]
            vegshmat = data["vegshadowmat"]
            vbshvegshmat = data["vbshmat"]
            if self.config.use_veg_dem:
                diffsh = np.zeros((rows, cols, shmat.shape[2]))
                for i in range(0, shmat.shape[2]):
                    diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (
                        1 - transVeg
                    )  # changes in psi not implemented yet
            else:
                diffsh = shmat

            # Estimate number of patches based on shadow matrices
            if shmat.shape[2] == 145:
                patch_option = 1  # patch_option = 1 # 145 patches
            elif shmat.shape[2] == 153:
                patch_option = 2  # patch_option = 2 # 153 patches
            elif shmat.shape[2] == 306:
                patch_option = 3  # patch_option = 3 # 306 patches
            elif shmat.shape[2] == 612:
                patch_option = 4  # patch_option = 4 # 612 patches

            # asvf to calculate sunlit and shaded patches
            asvf = np.arccos(np.sqrt(svf_data.svf))

            # Empty array for steradians
            steradians = np.zeros(shmat.shape[2])
        else:
            # anisotropic_sky = 0
            diffsh = None
            shmat = None
            vegshmat = None
            vbshvegshmat = None
            asvf = None
            patch_option = 0
            steradians = 0

        # % Ts parameterisation maps
        if self.config.use_landcover:
            # Get land cover properties for Tg wave (land cover scheme based on Bogren et al. 2000, explained in Lindberg et al., 2008 and Lindberg, Onomura & Grimmond, 2016)
            [TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall] = Tgmaps_v1(
                lcgrid.copy(), self.params
            )
        else:
            TgK = Knight + self.params.Ts_deg.Value.Cobble_stone_2014a
            Tstart = Knight - self.params.Tstart.Value.Cobble_stone_2014a
            TmaxLST = self.params.TmaxLST.Value.Cobble_stone_2014a
            alb_grid = Knight + self.params.Albedo.Effective.Value.Cobble_stone_2014a
            emis_grid = Knight + self.params.Emissivity.Value.Cobble_stone_2014a
            TgK_wall = self.params.Ts_deg.Value.Walls
            Tstart_wall = self.params.Tstart.Value.Walls
            TmaxLST_wall = self.params.TmaxLST.Value.Walls

        # Import data for wall temperature parameterization
        if self.config.use_wall_scheme:
            wallData = np.load(self.config.wall_path)
            voxelMaps = wallData["voxelId"]
            voxelTable = wallData["voxelTable"]

            # Get wall type
            if self.qgis_env is False:
                wall_type_standalone = {"Brick_wall": "100", "Concrete_wall": "101", "Wood_wall": "102"}
                wall_type = wall_type_standalone[self.config.wall_type]
            else:
                wall_type = self.config.wall_type

            # Get heights of walls including corners
            walls_scheme = wa.findwalls_sp(dsm_arr, 2, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
            # Get aspects of walls including corners
            dirwalls_scheme = wa.filter1Goodwin_as_aspect_v3(walls_scheme.copy(), scale, dsm_arr, None, 100.0 / 180.0)
            # Calculate timeStep
            first_timestep = (
                pd.to_datetime(YYYY[0][0], format="%Y")
                + pd.to_timedelta(weather_data.DOY[0] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[0], unit="h")
                + pd.to_timedelta(weather_data.minu[0], unit="m")
            )
            second_timestep = (
                pd.to_datetime(YYYY[0][1], format="%Y")
                + pd.to_timedelta(weather_data.DOY[1] - 1, unit="d")
                + pd.to_timedelta(weather_data.hours[1], unit="h")
                + pd.to_timedelta(weather_data.minu[1], unit="m")
            )
            timeStep = (second_timestep - first_timestep).seconds

            # Load voxelTable as Pandas DataFrame
            voxelTable, dirwalls_scheme = load_walls(
                voxelTable,
                self.params,
                wall_type,
                dirwalls_scheme,
                weather_data.Ta[0],
                timeStep,
                alb_grid,
                self.config.use_landcover,
                lcgrid,
                dsm_arr,
            )

            # Use wall of interest
            if self.config.woi_path:
                self.load_woi_data(dsm_trf_arr)

            # Create pandas datetime object for NetCDF output
            if self.config.wall_netcdf:
                met_for_xarray = (
                    pd.to_datetime(YYYY[0][:], format="%Y")
                    + pd.to_timedelta(weather_data.DOY - 1, unit="d")
                    + pd.to_timedelta(weather_data.hours, unit="h")
                    + pd.to_timedelta(weather_data.minu, unit="m")
                )
        else:
            voxelMaps = None
            voxelTable = None
            timeStep = 0
            walls_scheme = np.ones((rows, cols)) * 10.0
            dirwalls_scheme = np.ones((rows, cols)) * 10.0

        # Initialisation of time related variables
        if weather_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = dectime[1] - dectime[0]
        timeadd = 0.0
        firstdaytime = 1.0

        # Save hemispheric image
        if self.config.use_aniso and self.poi_pixel_xys is not None:
            patch_characteristics = hemispheric_image(
                self.poi_pixel_xys,
                shmat,
                vegshmat,
                vbshvegshmat,
                voxelMaps,
                walls_scheme,
            )

        # If metfile starts at night
        CI = 1.0

        # reading variables from config and parameters that is not yet presented
        albedo_b = self.params.Albedo.Effective.Value.Walls
        ewall = self.params.Emissivity.Value.Walls
        elvis = 0.0
        absK = self.params.Tmrt_params.Value.absK
        absL = self.params.Tmrt_params.Value.absL

        # Main loop
        tmrtplot = np.zeros((rows, cols))
        TgOut1 = np.zeros((rows, cols))

        # Initiate array for I0 values
        if np.unique(weather_data.DOY).shape[0] > 1:
            unique_days = np.unique(weather_data.DOY)
            first_unique_day = weather_data.DOY[unique_days[0] == weather_data.DOY]
            I0_array = np.zeros(first_unique_day.shape[0])
        else:
            first_unique_day = weather_data.DOY.copy()
            I0_array = np.zeros(weather_data.DOY.shape[0])

        self.prep_progress(weather_data.Ta.__len__())

        HYBRID = os.environ.get("HYBRID").lower() in ["true", "1"]
        for i in np.arange(0, self.iters_total):
            proceed = self.iter_progress()
            if not proceed:
                break

            # Daily water body temperature
            Twater = np.nan
            if self.config.use_landcover:
                if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
                    Twater = np.mean(weather_data.Ta[jday[0] == np.floor(dectime[i])])

            # Nocturnal cloudfraction from Offerle et al. 2003
            if (dectime[i] - np.floor(dectime[i])) == 0:
                daylines = np.where(np.floor(dectime) == dectime[i])
                if daylines.__len__() > 1:
                    alt_day = altitude[0][daylines]
                    alt2 = np.where(alt_day > 1)
                    rise = alt2[0][0]
                    [_, CI, _, _, _] = clearnessindex_2013b(
                        zen[0, i + rise + 1],
                        jday[0, i + rise + 1],
                        weather_data.Ta[i + rise + 1],
                        weather_data.RH[i + rise + 1] / 100.0,
                        weather_data.radG[i + rise + 1],
                        location,
                        weather_data.P[i + rise + 1],
                    )
                    if (CI > 1.0) or (np.inf == CI):
                        CI = 1.0
                else:
                    CI = 1.0

            if HYBRID is not True:
                (
                    Tmrt,
                    Kdown,
                    Kup,
                    Ldown,
                    Lup,
                    Tg,
                    ea,
                    esky,
                    I0,
                    CI,
                    shadow,
                    firstdaytime,
                    timestepdec,
                    timeadd,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    Keast,
                    Ksouth,
                    Kwest,
                    Knorth,
                    Least,
                    Lsouth,
                    Lwest,
                    Lnorth,
                    KsideI,
                    TgOut1,
                    TgOut,
                    radIout,
                    radDout,
                    Lside,
                    Lsky_patch_characteristics,
                    CI_Tg,
                    CI_TgG,
                    KsideD,
                    dRad,
                    Kside,
                    steradians,
                    voxelTable,
                ) = Solweig_2025a_calc(
                    i,
                    dsm_arr,
                    scale,
                    rows,
                    cols,
                    svf_data.svf,
                    svf_data.svf_north,
                    svf_data.svf_west,
                    svf_data.svf_east,
                    svf_data.svf_south,
                    svf_data.svf_veg,
                    svf_data.svf_veg_north,
                    svf_data.svf_veg_east,
                    svf_data.svf_veg_south,
                    svf_data.svf_veg_west,
                    svf_data.svf_veg_blocks_bldg_sh,
                    svf_data.svf_veg_blocks_bldg_sh_east,
                    svf_data.svf_veg_blocks_bldg_sh_south,
                    svf_data.svf_veg_blocks_bldg_sh_west,
                    svf_data.svf_veg_blocks_bldg_sh_north,
                    vegdsm,
                    vegdsm2,
                    albedo_b,
                    absK,
                    absL,
                    ewall,
                    Fside,
                    Fup,
                    Fcyl,
                    altitude[0][i],
                    azimuth[0][i],
                    zen[0][i],
                    jday[0][i],
                    self.config.use_veg_dem,
                    self.config.only_global,
                    buildings,
                    location,
                    psi[0][i],
                    self.config.use_landcover,
                    lcgrid,
                    dectime[i],
                    altmax[0][i],
                    wallaspect,
                    wallheight,
                    self.config.person_cylinder,
                    elvis,
                    weather_data.Ta[i],
                    weather_data.RH[i],
                    weather_data.radG[i],
                    weather_data.radD[i],
                    weather_data.radI[i],
                    weather_data.P[i],
                    amaxvalue,
                    bush,
                    Twater,
                    TgK,
                    Tstart,
                    alb_grid,
                    emis_grid,
                    TgK_wall,
                    Tstart_wall,
                    TmaxLST,
                    TmaxLST_wall,
                    first,
                    second,
                    svfalfa,
                    svfbuveg,
                    firstdaytime,
                    timeadd,
                    timestepdec,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    CI,
                    TgOut1,
                    diffsh,
                    shmat,
                    vegshmat,
                    vbshvegshmat,
                    self.config.use_aniso,
                    asvf,
                    patch_option,
                    voxelMaps,
                    voxelTable,
                    weather_data.Ws[i],
                    self.config.use_wall_scheme,
                    timeStep,
                    steradians,
                    walls_scheme,
                    dirwalls_scheme,
                )
            else:
                (
                    Tmrt,
                    Kdown,
                    Kup,
                    Ldown,
                    Lup,
                    Tg,
                    ea,
                    esky,
                    I0,
                    CI,
                    shadow,
                    firstdaytime,
                    timestepdec,
                    timeadd,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    Keast,
                    Ksouth,
                    Kwest,
                    Knorth,
                    Least,
                    Lsouth,
                    Lwest,
                    Lnorth,
                    KsideI,
                    TgOut1,
                    TgOut,
                    radIout,
                    radDout,
                    Lside,
                    Lsky_patch_characteristics,
                    CI_Tg,
                    CI_TgG,
                    KsideD,
                    dRad,
                    Kside,
                    steradians,
                    voxelTable,
                ) = Solweig_2025a_calc_hybrid(
                    i,
                    dsm_arr,
                    scale,
                    rows,
                    cols,
                    svf_data.svf,
                    svf_data.svf_north,
                    svf_data.svf_west,
                    svf_data.svf_east,
                    svf_data.svf_south,
                    svf_data.svf_veg,
                    svf_data.svf_veg_north,
                    svf_data.svf_veg_east,
                    svf_data.svf_veg_south,
                    svf_data.svf_veg_west,
                    svf_data.svf_veg_blocks_bldg_sh,
                    svf_data.svf_veg_blocks_bldg_sh_east,
                    svf_data.svf_veg_blocks_bldg_sh_south,
                    svf_data.svf_veg_blocks_bldg_sh_west,
                    svf_data.svf_veg_blocks_bldg_sh_north,
                    vegdsm,
                    vegdsm2,
                    albedo_b,
                    absK,
                    absL,
                    ewall,
                    Fside,
                    Fup,
                    Fcyl,
                    altitude[0][i],
                    azimuth[0][i],
                    zen[0][i],
                    jday[0][i],
                    self.config.use_veg_dem,
                    self.config.only_global,
                    buildings,
                    location,
                    psi[0][i],
                    self.config.use_landcover,
                    lcgrid,
                    dectime[i],
                    altmax[0][i],
                    wallaspect,
                    wallheight,
                    self.config.person_cylinder,
                    elvis,
                    weather_data.Ta[i],
                    weather_data.RH[i],
                    weather_data.radG[i],
                    weather_data.radD[i],
                    weather_data.radI[i],
                    weather_data.P[i],
                    amaxvalue,
                    bush,
                    Twater,
                    TgK,
                    Tstart,
                    alb_grid,
                    emis_grid,
                    TgK_wall,
                    Tstart_wall,
                    TmaxLST,
                    TmaxLST_wall,
                    first,
                    second,
                    svfalfa,
                    svfbuveg,
                    firstdaytime,
                    timeadd,
                    timestepdec,
                    Tgmap1,
                    Tgmap1E,
                    Tgmap1S,
                    Tgmap1W,
                    Tgmap1N,
                    CI,
                    TgOut1,
                    diffsh,
                    shmat,
                    vegshmat,
                    vbshvegshmat,
                    self.config.use_aniso,
                    asvf,
                    patch_option,
                    voxelMaps,
                    voxelTable,
                    weather_data.Ws[i],
                    self.config.use_wall_scheme,
                    timeStep,
                    steradians,
                    walls_scheme,
                    dirwalls_scheme,
                )

            # Save I0 for I0 vs. Kdown output plot to check if UTC is off
            if i < first_unique_day.shape[0]:
                I0_array[i] = I0
            elif i == first_unique_day.shape[0]:
                # Output I0 vs. Kglobal plot
                radG_for_plot = weather_data.radG[first_unique_day[0] == weather_data.DOY]
                dectime_for_plot = dectime[first_unique_day[0] == weather_data.DOY]
                fig, ax = plt.subplots()
                ax.plot(dectime_for_plot, I0_array, label="I0")
                ax.plot(dectime_for_plot, radG_for_plot, label="Kglobal")
                ax.set_ylabel("Shortwave radiation [$Wm^{-2}$]")
                ax.set_xlabel("Decimal time")
                ax.set_title("UTC" + str(self.config.utc))
                ax.legend()
                fig.savefig(self.config.output_dir + "/metCheck.png", dpi=150)

            tmrtplot = tmrtplot + Tmrt

            if altitude[0][i] > 0:
                w = "D"
            else:
                w = "N"

            if weather_data.hours[i] < 10:
                XH = "0"
            else:
                XH = ""

            if weather_data.minu[i] < 10:
                XM = "0"
            else:
                XM = ""

            if self.poi_pixel_xys is not None:
                for n in range(0, self.poi_pixel_xys.shape[0]):
                    idx, row_idx, col_idx = self.poi_pixel_xys[n]
                    row_idx = int(row_idx)
                    col_idx = int(col_idx)
                    result_row = {
                        "poi_idx": idx,
                        "yyyy": YYYY[0, i],
                        "id": jday[0, i],
                        "it": weather_data.hours[i],
                        "imin": weather_data.minu[i],
                        "dectime": dectime[i],
                        "altitude": altitude[0, i],
                        "azimuth": azimuth[0, i],
                        "kdir": radIout,
                        "kdiff": radDout,
                        "kglobal": weather_data.radG[i],
                        "kdown": Kdown[row_idx, col_idx],
                        "kup": Kup[row_idx, col_idx],
                        "keast": Keast[row_idx, col_idx],
                        "ksouth": Ksouth[row_idx, col_idx],
                        "kwest": Kwest[row_idx, col_idx],
                        "knorth": Knorth[row_idx, col_idx],
                        "ldown": Ldown[row_idx, col_idx],
                        "lup": Lup[row_idx, col_idx],
                        "least": Least[row_idx, col_idx],
                        "lsouth": Lsouth[row_idx, col_idx],
                        "lwest": Lwest[row_idx, col_idx],
                        "lnorth": Lnorth[row_idx, col_idx],
                        "Ta": weather_data.Ta[i],
                        "Tg": TgOut[row_idx, col_idx],
                        "RH": weather_data.RH[i],
                        "Esky": esky,
                        "Tmrt": Tmrt[row_idx, col_idx],
                        "I0": I0,
                        "CI": CI,
                        "Shadow": shadow[row_idx, col_idx],
                        "SVF_b": svf_data.svf[row_idx, col_idx],
                        "SVF_bv": svfbuveg[row_idx, col_idx],
                        "KsideI": KsideI[row_idx, col_idx],
                    }
                    # Recalculating wind speed based on powerlaw
                    WsPET = (1.1 / self.params.Wind_Height.Value.magl) ** 0.2 * weather_data.Ws[i]
                    WsUTCI = (10.0 / self.params.Wind_Height.Value.magl) ** 0.2 * weather_data.Ws[i]
                    resultPET = PET_calculations._PET(
                        weather_data.Ta[i],
                        weather_data.RH[i],
                        Tmrt[row_idx, col_idx],
                        WsPET,
                        self.params.PET_settings.Value.Weight,
                        self.params.PET_settings.Value.Age,
                        self.params.PET_settings.Value.Height,
                        self.params.PET_settings.Value.Activity,
                        self.params.PET_settings.Value.clo,
                        self.params.PET_settings.Value.Sex,
                    )
                    result_row["PET"] = resultPET
                    resultUTCI = utci.utci_calculator(
                        weather_data.Ta[i], weather_data.RH[i], Tmrt[row_idx, col_idx], WsUTCI
                    )
                    result_row["UTCI"] = resultUTCI
                    result_row["CI_Tg"] = CI_Tg
                    result_row["CI_TgG"] = CI_TgG
                    result_row["KsideD"] = KsideD[row_idx, col_idx]
                    result_row["Lside"] = Lside[row_idx, col_idx]
                    result_row["diffDown"] = dRad[row_idx, col_idx]
                    result_row["Kside"] = Kside[row_idx, col_idx]
                    self.poi_results.append(result_row)

            if self.woi_pixel_xys is not None:
                for n in range(0, self.woi_pixel_xys.shape[0]):
                    idx, row_idx, col_idx = self.woi_pixel_xys[n]
                    row_idx = int(row_idx)
                    col_idx = int(col_idx)

                    temp_wall = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "wallTemperature"
                    ].to_numpy()
                    K_in = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "K_in"
                    ].to_numpy()
                    L_in = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "L_in"
                    ].to_numpy()
                    wallShade = voxelTable.loc[
                        ((voxelTable["ypos"] == row_idx) & (voxelTable["xpos"] == col_idx)), "wallShade"
                    ].to_numpy()

                    result_row = {
                        "woi_idx": idx,
                        "woi_name": self.woi_names[idx],
                        "yyyy": YYYY[0, i],
                        "id": jday[0, i],
                        "it": weather_data.hours[i],
                        "imin": weather_data.minu[i],
                        "dectime": dectime[i],
                        "Ta": weather_data.Ta[i],
                        "SVF": svf_data.svf[row_idx, col_idx],
                        "Ts": temp_wall,
                        "Kin": K_in,
                        "Lin": L_in,
                        "shade": wallShade,
                        "pixel_x": col_idx,
                        "pixel_y": row_idx,
                    }
                    self.woi_results.append(result_row)

                if self.config.wall_netcdf:
                    netcdf_output = self.config.output_dir + "/walls.nc"
                    walls_as_netcdf(
                        voxelTable, rows, cols, met_for_xarray, i, dsm_arr, self.config.dsm_path, netcdf_output
                    )

            time_code = (
                str(int(YYYY[0, i]))
                + "_"
                + str(int(weather_data.DOY[i]))
                + "_"
                + XH
                + str(int(weather_data.hours[i]))
                + XM
                + str(int(weather_data.minu[i]))
                + w
            )

            if self.config.output_tmrt:
                common.save_raster(
                    self.config.output_dir + "/Tmrt_" + time_code + ".tif",
                    Tmrt,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_lup:
                common.save_raster(
                    self.config.output_dir + "/Lup_" + time_code + ".tif",
                    Lup,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_ldown:
                common.save_raster(
                    self.config.output_dir + "/Ldown_" + time_code + ".tif",
                    Ldown,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )
            if self.config.output_kdiff:
                common.save_raster(
                    self.config.output_dir + "/Kdiff_" + time_code + ".tif",
                    dRad,
                    dsm_trf_arr,
                    dsm_crs_wkt,
                    dsm_nd_val,
                )

            # Sky view image of patches
            if (self.config.aniso_path) and (i == 0) and (self.poi_pixel_xys is not None):
                for k in range(self.poi_pixel_xys.shape[0]):
                    Lsky_patch_characteristics[:, 2] = patch_characteristics[:, k]
                    skyviewimage_out = self.config.output_dir + "/POI_" + str(self.poi_names[k]) + ".png"
                    PolarBarPlot(
                        Lsky_patch_characteristics,
                        altitude[0][i],
                        azimuth[0][i],
                        "Hemisphere partitioning",
                        skyviewimage_out,
                        0,
                        5,
                        0,
                    )

        # Save POI results
        if self.poi_results:
            self.save_poi_results(dsm_trf_arr, dsm_crs_wkt)

        # Save WOI results
        if self.woi_results:
            self.save_woi_results(dsm_trf_arr, dsm_crs_wkt)

        # Save Tree Planter results
        if self.config.output_tree_planter:
            albedo_g = self.params.Albedo.Effective.Value.Cobble_stone_2014a
            eground = self.params.Emissivity.Value.Cobble_stone_2014a
            pos = 1 if self.params.Tmrt_params.Value.posture == "Standing" else 0

            settingsHeader = "UTC, posture, onlyglobal, landcover, anisotropic, cylinder, albedo_walls, albedo_ground, emissivity_walls, emissivity_ground, absK, absL, elevation, patch_option"
            settingsFmt = (
                "%i",
                "%i",
                "%i",
                "%i",
                "%i",
                "%i",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%1.2f",
                "%i",
            )
            settingsData = np.array(
                [
                    [
                        int(self.config.utc),
                        pos,
                        self.config.only_global,
                        self.config.use_landcover,
                        self.config.use_aniso,
                        self.config.person_cylinder,
                        albedo_b,
                        albedo_g,
                        ewall,
                        eground,
                        absK,
                        absL,
                        alt,
                        patch_option,
                    ]
                ]
            )
            np.savetxt(
                self.config.output_dir + "/treeplantersettings.txt",
                settingsData,
                fmt=settingsFmt,
                header=settingsHeader,
                delimiter=" ",
            )

        # Save average Tmrt raster
        tmrtplot = tmrtplot / self.iters_total
        common.save_raster(
            self.config.output_dir + "/Tmrt_average.tif",
            tmrtplot,
            dsm_trf_arr,
            dsm_crs_wkt,
            dsm_nd_val,
        )
