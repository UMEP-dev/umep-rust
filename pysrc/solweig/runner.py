"""
Standalone SOLWEIG runner - no umep dependency required.

This module provides the core SOLWEIG model runner that works independently
of the umep package. For Rust-optimized execution, use SolweigRunRust from
solweig_runner_rust.py which subclasses SolweigRunCore.
"""

import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pvlib.iotools import read_epw
from rasterio.transform import Affine, rowcol
from tqdm import tqdm

from . import io as common
from .configs import (
    EnvironData,
    RasterData,
    ShadowMatrices,
    SolweigConfig,
    SvfData,
    TgMaps,
    WallsData,
)
from .tiles import TileManager
from .algorithms import PET_calculations
from .algorithms import UTCI_calculations as utci

# Import the Solweig calculation function
from .functions.solweig import Solweig_2025a_calc as Solweig_2025a_calc_default

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d


class SolweigRunCore:
    """
    Standalone SOLWEIG runner - works without QGIS or umep dependency.

    This class handles:
    - Loading configuration and parameters
    - Loading raster data (DSM, CDSM, SVF, etc.)
    - Loading weather data (MET or EPW files)
    - Running the SOLWEIG model
    - Saving outputs

    For Rust-optimized calculations, subclass this and override calc_solweig().
    """

    config: SolweigConfig
    progress: Optional[Any]
    iters_total: Optional[int]
    iters_count: int = 0
    poi_names: List[Any] = []
    poi_pixel_xys: Optional[np.ndarray]
    poi_results = []
    woi_names: List[Any] = []
    woi_pixel_xys: Optional[np.ndarray]
    woi_results = []
    raster_data: RasterData
    location: Dict[str, float]
    svf_data: SvfData
    environ_data: EnvironData
    tg_maps: TgMaps
    shadow_mats: ShadowMatrices
    walls_data: WallsData

    def __init__(
        self,
        config_path_str: str,
        params_json_path: str,
        amax_local_window_m: int = 100,
        amax_local_perc: float = 99.9,
        use_tiled_loading: bool = False,
        tile_size: int = 1024,
    ):
        """
        Initialize the SOLWEIG runner.

        Args:
            config_path_str: Path to the SOLWEIG configuration file (.ini)
            params_json_path: Path to the parameters JSON file
            amax_local_window_m: Window size for local amax calculation (meters)
            amax_local_perc: Percentile for amax calculation
            use_tiled_loading: Enable tiled processing for large rasters
            tile_size: Tile size when using tiled loading
        """
        logger.info("Starting SOLWEIG setup")

        # Load configuration
        self.config = SolweigConfig()
        self.config.from_file(config_path_str)
        self.use_tiled_loading = use_tiled_loading
        self.tile_size = tile_size
        self.config.validate()

        # Progress tracking settings
        self.progress = None
        self.iters_total = None
        self.iters_count = 0
        self.proceed = True

        # Initialize POI data
        self.poi_names = []
        self.poi_pixel_xys = None
        self.poi_results = []

        # Initialize WOI data
        self.woi_names = []
        self.woi_pixel_xys = None
        self.woi_results = []

        # Load parameters from JSON file
        params_path = common.check_path(params_json_path)
        try:
            with open(params_path) as f:
                params_dict = json.load(f)
                self.params = dict_to_namespace(params_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters from {params_json_path}: {e}")

        # Initialize SVF and Raster data
        if self.use_tiled_loading:
            self._init_tiled(amax_local_window_m, amax_local_perc)
        else:
            self._init_standard(amax_local_window_m, amax_local_perc)

        # Load weather data
        if self.config.use_epw_file:
            self.environ_data = self.load_epw_weather()
            logger.info("Weather data loaded from EPW file")
        else:
            self.environ_data = self.load_met_weather(header_rows=1, delim=" ")
            logger.info("Weather data loaded from MET file")

        # Load POI data if configured
        if self.config.poi_path:
            self.load_poi_data()
            logger.info("POI data loaded from %s", self.config.poi_path)

        # Load WOI data if configured
        if self.config.woi_path:
            self.load_woi_data()
            logger.info("WOI data loaded from %s", self.config.woi_path)

    def _init_standard(self, amax_local_window_m: int, amax_local_perc: float):
        """Initialize with standard (non-tiled) loading."""
        logger.info("Using eager loading for raster data")
        self.svf_data = SvfData(self.config)
        self.raster_data = RasterData(
            self.config,
            self.params,
            self.svf_data,
            amax_local_window_m,
            amax_local_perc,
        )
        # Location data
        left_x = self.raster_data.trf_arr[0]
        top_y = self.raster_data.trf_arr[3]
        lng, lat = common.xy_to_lnglat(self.raster_data.crs_wkt, left_x, top_y)
        alt = float(np.nanmedian(self.raster_data.dsm))
        if alt < 0:
            alt = 3
        self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

        self.rows = self.raster_data.rows
        self.cols = self.raster_data.cols
        self.transform = self.raster_data.trf_arr
        self.crs = self.raster_data.crs_wkt

    def _init_tiled(self, amax_local_window_m: int, amax_local_perc: float):
        """Initialize with tiled loading for large rasters."""
        logger.info("Using tiled loading for raster data")

        # Get metadata from DSM to initialize TileManager and Location
        dsm_meta = common.get_raster_metadata(self.config.dsm_path)
        rows = dsm_meta["rows"]
        cols = dsm_meta["cols"]
        pixel_size = dsm_meta["transform"][1]

        # Check if svf.tif exists, if not unzip
        svf_file = Path(self.config.working_dir) / "svf.tif"
        if not svf_file.exists():
            svf_zip_path = Path(self.config.svf_path).absolute()
            if not svf_zip_path.exists():
                raise FileNotFoundError(
                    f"SVF zip file not found at {svf_zip_path} and extracted files not found in {self.config.working_dir}. "
                    "Please run SVF generation first."
                )
            logger.info("Unzipping SVF files for tiled access...")
            with zipfile.ZipFile(str(svf_zip_path), "r") as zip_ref:
                zip_ref.extractall(self.config.working_dir)

        # Initialize TileManager
        self.tile_manager = TileManager(
            rows=rows,
            cols=cols,
            tile_size=self.tile_size,
            pixel_size=pixel_size,
            buffer_dist=150.0,
        )

        # Location data from metadata
        left_x = dsm_meta["transform"][0]
        top_y = dsm_meta["transform"][3]
        lng, lat = common.xy_to_lnglat(dsm_meta["crs"], left_x, top_y)

        # Read center altitude
        center_r, center_c = rows // 2, cols // 2
        center_val = common.read_raster_window(
            self.config.dsm_path, (slice(center_r, center_r + 1), slice(center_c, center_c + 1))
        )
        alt = float(center_val[0, 0])
        if alt < 0:
            alt = 3

        self.location = {"longitude": lng, "latitude": lat, "altitude": alt}

        # Store metadata
        self.rows = rows
        self.cols = cols
        self.transform = dsm_meta["transform"]
        self.crs = dsm_meta["crs"]

        # Defer data loading
        self.raster_data = None
        self.svf_data = None
        self.shadow_mats = None
        self.tg_maps = None
        self.walls_data = None

        # Store params for tiled instantiation
        self.amax_local_window_m = amax_local_window_m
        self.amax_local_perc = amax_local_perc

    def prep_progress(self, num: int) -> None:
        """Prepare progress bar for CLI."""
        self.iters_total = num
        self.iters_count = 0
        self.progress = tqdm(total=num, desc="Running SOLWEIG", unit="step")

    def iter_progress(self) -> bool:
        """Update progress bar."""
        self.progress.update(1)
        return True

    def load_poi_data(self) -> Tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        poi_path_str = str(common.check_path(self.config.poi_path))
        pois_gdf = gpd.read_file(poi_path_str)
        trf = Affine.from_gdal(*self.transform)
        self.poi_pixel_xys = np.zeros((len(pois_gdf), 3), dtype=np.float32) - 999
        self.poi_names = []
        for n, (idx, row) in enumerate(pois_gdf.iterrows()):
            self.poi_names.append(idx)
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            self.poi_pixel_xys[n] = (n, x, y)

    def save_poi_results(self) -> None:
        """Save points of interest (POIs) results to a file."""
        xs = [r["col_idx"] * self.transform[1] + self.transform[0] for r in self.poi_results]
        ys = [r["row_idx"] * self.transform[1] + self.transform[3] for r in self.poi_results]
        pois_gdf = gpd.GeoDataFrame(
            self.poi_results,
            geometry=gpd.points_from_xy(xs, ys),
            crs=self.crs,
        )
        pois_gdf["snapshot"] = pd.to_datetime(
            pois_gdf["yyyy"].astype(int).astype(str)
            + "-"
            + pois_gdf["id"].astype(int).astype(str).str.zfill(3)
            + " "
            + pois_gdf["it"].astype(int).astype(str).str.zfill(2)
            + ":"
            + pois_gdf["imin"].astype(int).astype(str).str.zfill(2),
            format="%Y-%j %H:%M",
        )
        pois_gdf.to_file(self.config.output_dir + "/POI.gpkg", driver="GPKG")

    def load_woi_data(self) -> Tuple[Any, Any]:
        """Load walls of interest (WOIs) from a file."""
        woi_gdf = gpd.read_file(self.config.woi_file)
        trf = Affine.from_gdal(*self.transform)
        self.woi_pixel_xys = np.zeros((len(woi_gdf), 3), dtype=np.float32) - 999
        self.woi_names = []
        for n, (idx, row) in enumerate(woi_gdf.iterrows()):
            self.woi_names.append(idx)
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            self.woi_pixel_xys[n] = (n, x, y)

    def save_woi_results(self) -> None:
        """Save walls of interest (WOIs) results to a file."""
        xs = [r["col_idx"] * self.transform[1] + self.transform[0] for r in self.woi_results]
        ys = [r["row_idx"] * self.transform[1] + self.transform[3] for r in self.woi_results]
        woi_gdf = gpd.GeoDataFrame(
            self.woi_results,
            geometry=gpd.points_from_xy(xs, ys),
            crs=self.crs,
        )
        woi_gdf["snapshot"] = pd.to_datetime(
            woi_gdf["yyyy"].astype(int).astype(str)
            + "-"
            + woi_gdf["id"].astype(int).astype(str).str.zfill(3)
            + " "
            + woi_gdf["it"].astype(int).astype(str).str.zfill(2)
            + ":"
            + woi_gdf["imin"].astype(int).astype(str).str.zfill(2),
            format="%Y-%j %H:%M",
        )
        woi_gdf.to_file(self.config.output_dir + "/WOI.gpkg", driver="GPKG")

    def load_epw_weather(self) -> EnvironData:
        """Load weather data from an EPW file."""
        epw_path_str = str(common.check_path(self.config.epw_path))
        epw_df, epw_info = read_epw(epw_path_str)
        tz = epw_df.index.tz
        start_date = pd.Timestamp(
            year=self.config.epw_start_date[0],
            month=self.config.epw_start_date[1],
            day=self.config.epw_start_date[2],
            hour=self.config.epw_start_date[3],
            tzinfo=tz,
        )
        end_date = pd.Timestamp(
            year=self.config.epw_end_date[0],
            month=self.config.epw_end_date[1],
            day=self.config.epw_end_date[2],
            hour=self.config.epw_end_date[3],
            tzinfo=tz,
        )
        filtered_df = epw_df.loc[start_date:end_date]
        filtered_df = filtered_df[filtered_df.index.hour.isin(self.config.epw_hours)]

        if len(filtered_df) == 0:
            raise ValueError("No EPW dates intersect start and end dates and / or hours.")

        umep_df = pd.DataFrame(
            {
                "iy": filtered_df.index.year,
                "id": filtered_df.index.dayofyear,
                "it": filtered_df.index.hour,
                "imin": filtered_df.index.minute,
                "Q": -999,
                "QH": -999,
                "QE": -999,
                "Qs": -999,
                "Qf": -999,
                "Wind": filtered_df["wind_speed"],
                "RH": filtered_df["relative_humidity"],
                "Tair": filtered_df["temp_air"],
                "pres": filtered_df["atmospheric_pressure"].astype(np.float32),
                "rain": -999,
                "Kdown": filtered_df["ghi"],
                "snow": filtered_df["snow_depth"],
                "ldown": filtered_df["ghi_infrared"],
                "fcld": filtered_df["total_sky_cover"],
                "wuh": filtered_df["precipitable_water"],
                "xsmd": -999,
                "lai_hr": -999,
                "Kdiff": filtered_df["dhi"],
                "Kdir": filtered_df["dni"],
                "Wdir": filtered_df["wind_direction"],
            }
        )

        umep_df_filt = umep_df[(umep_df["Kdown"] < 0) & (umep_df["Kdown"] > 1300)]
        if len(umep_df_filt):
            raise ValueError("Error: Kdown - beyond what is expected")

        umep_df = umep_df.fillna(-999)

        return EnvironData(
            self.config,
            self.params,
            YYYY=umep_df["iy"].to_numpy(dtype=np.float32),
            DOY=umep_df["id"].to_numpy(dtype=np.float32),
            hours=umep_df["it"].to_numpy(dtype=np.float32),
            minu=umep_df["imin"].to_numpy(dtype=np.float32),
            Ta=umep_df["Tair"].to_numpy(dtype=np.float32),
            RH=umep_df["RH"].to_numpy(dtype=np.float32),
            radG=umep_df["Kdown"].to_numpy(dtype=np.float32),
            radD=umep_df["ldown"].to_numpy(dtype=np.float32),
            radI=umep_df["Kdiff"].to_numpy(dtype=np.float32),
            P=umep_df["pres"].to_numpy(dtype=np.float32) / 100.0,
            Ws=umep_df["Wind"].to_numpy(dtype=np.float32),
            location=self.location,
            UTC=self.config.utc,
        )

    def load_met_weather(self, header_rows: int = 1, delim: str = " ") -> EnvironData:
        """Load weather data from a MET file."""
        met_path_str = str(common.check_path(self.config.met_path))
        met_data = np.loadtxt(met_path_str, skiprows=header_rows, delimiter=delim, dtype=np.float32)
        return EnvironData(
            self.config,
            self.params,
            YYYY=met_data[:, 0],
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
            location=self.location,
            UTC=self.config.utc,
        )

    def calc_solweig(
        self,
        iter: int,
        elvis: float,
        first: float,
        second: float,
        firstdaytime: float,
        timeadd: float,
        timestepdec: float,
        posture,
    ):
        """
        Calculate SOLWEIG results for a given iteration.

        Override this method in subclasses to use Rust-optimized calculations.
        """
        return Solweig_2025a_calc_default(
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
            int(self.config.person_cylinder),
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
            int(self.config.use_aniso),
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

    def run(self) -> None:
        """Run the SOLWEIG model."""
        if self.use_tiled_loading:
            self.run_tiled()
        else:
            self.run_standard()

    def run_standard(self) -> None:
        """Run SOLWEIG with standard (non-tiled) processing."""
        logger.info("Initializing data for standard execution...")

        # Initialize shadow matrices
        self.shadow_mats = ShadowMatrices(self.config, self.params, self.svf_data)
        logger.info("Shadow matrices initialized")

        # Initialize Tg maps
        self.tg_maps = TgMaps(
            self.config.use_landcover,
            self.params,
            self.raster_data,
        )
        logger.info("TgMaps initialized")

        # Initialize walls data
        self.walls_data = WallsData(
            self.config,
            self.params,
            self.raster_data,
            self.environ_data,
            self.tg_maps,
        )
        logger.info("WallsData initialized")

        # Posture settings
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value

        first = np.round(posture.height)
        if first == 0.0:
            first = 1.0
        second = np.round(posture.height * 20.0)

        # Time initialization
        if self.environ_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = self.environ_data.dectime[1] - self.environ_data.dectime[0]
        timeadd = 0.0
        firstdaytime = 1.0

        # Tmrt aggregation
        tmrt_agg = np.zeros((self.raster_data.rows, self.raster_data.cols), dtype=np.float32)

        # Number of iterations
        num = len(self.environ_data.Ta)
        self.prep_progress(num)
        logger.info("Progress tracking prepared for %d iterations", num)

        elvis = 0.0

        for i in range(num):
            self.proceed = self.iter_progress()
            if not self.proceed:
                break
            self.iters_count += 1

            # Run SOLWEIG calculation
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
                self.tg_maps.Tgmap1,
                self.tg_maps.Tgmap1E,
                self.tg_maps.Tgmap1S,
                self.tg_maps.Tgmap1W,
                self.tg_maps.Tgmap1N,
                Keast,
                Ksouth,
                Kwest,
                Knorth,
                Least,
                Lsouth,
                Lwest,
                Lnorth,
                KsideI,
                self.tg_maps.TgOut1,
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
                self.shadow_mats.steradians,
                voxelTable,
            ) = self.calc_solweig(
                i,
                elvis,
                first,
                second,
                firstdaytime,
                timeadd,
                timestepdec,
                posture,
            )

            # Aggregate Tmrt
            if (~np.isfinite(Tmrt)).any() and self.iters_count > 1:
                logger.warning("Tmrt contains non-finite values, replacing with preceding average.")
                tmrt_avg = tmrt_agg / self.iters_count
                tmrt_agg = np.where(np.isfinite(Tmrt), tmrt_agg + Tmrt, tmrt_avg)
            elif (~np.isfinite(tmrt_agg)).any():
                raise ValueError("Tmrt aggregation contains non-finite values.")
            else:
                tmrt_agg = tmrt_agg + Tmrt

            # Time code for output files
            if self.environ_data.altitude[i] > 0:
                w = "D"
            else:
                w = "N"
            XH = "0" if self.environ_data.hours[i] < 10 else ""
            XM = "0" if self.environ_data.minu[i] < 10 else ""

            # Process POIs
            if self.poi_pixel_xys is not None:
                for n in range(self.poi_pixel_xys.shape[0]):
                    idx, row_idx, col_idx = self.poi_pixel_xys[n]
                    row_idx = int(row_idx)
                    col_idx = int(col_idx)
                    result_row = {
                        "poi_idx": idx,
                        "col_idx": col_idx,
                        "row_idx": row_idx,
                        "yyyy": self.environ_data.YYYY[i],
                        "id": self.environ_data.jday[i],
                        "it": self.environ_data.hours[i],
                        "imin": self.environ_data.minu[i],
                        "dectime": self.environ_data.dectime[i],
                        "altitude": self.environ_data.altitude[i],
                        "azimuth": self.environ_data.azimuth[i],
                        "kdir": radIout,
                        "kdiff": radDout,
                        "kglobal": self.environ_data.radG[i],
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
                        "Ta": self.environ_data.Ta[i],
                        "Tg": TgOut[row_idx, col_idx],
                        "RH": self.environ_data.RH[i],
                        "Esky": esky,
                        "Tmrt": Tmrt[row_idx, col_idx],
                        "I0": I0,
                        "CI": CI,
                        "Shadow": shadow[row_idx, col_idx],
                        "SVF_b": self.svf_data.svf[row_idx, col_idx],
                        "SVF_bv": self.raster_data.svfbuveg[row_idx, col_idx],
                        "KsideI": KsideI[row_idx, col_idx],
                    }
                    # PET and UTCI calculations
                    WsPET = (1.1 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]
                    WsUTCI = (10.0 / self.params.Wind_Height.Value.magl) ** 0.2 * self.environ_data.Ws[i]
                    resultPET = PET_calculations._PET(
                        self.environ_data.Ta[i],
                        self.environ_data.RH[i],
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
                        self.environ_data.Ta[i], self.environ_data.RH[i], Tmrt[row_idx, col_idx], WsUTCI
                    )
                    result_row["UTCI"] = resultUTCI
                    result_row["CI_Tg"] = CI_Tg
                    result_row["CI_TgG"] = CI_TgG
                    result_row["KsideD"] = KsideD[row_idx, col_idx]
                    result_row["Lside"] = Lside[row_idx, col_idx]
                    result_row["diffDown"] = dRad[row_idx, col_idx]
                    result_row["Kside"] = Kside[row_idx, col_idx]
                    self.poi_results.append(result_row)

            time_code = (
                str(int(self.environ_data.YYYY[i]))
                + "_"
                + str(int(self.environ_data.DOY[i]))
                + "_"
                + XH
                + str(int(self.environ_data.hours[i]))
                + XM
                + str(int(self.environ_data.minu[i]))
                + w
            )

            # Save outputs
            if self.config.output_tmrt:
                common.save_raster(
                    self.config.output_dir + "/Tmrt_" + time_code + ".tif",
                    Tmrt,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_kup:
                common.save_raster(
                    self.config.output_dir + "/Kup_" + time_code + ".tif",
                    Kup,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_kdown:
                common.save_raster(
                    self.config.output_dir + "/Kdown_" + time_code + ".tif",
                    Kdown,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )
            if self.config.output_sh:
                common.save_raster(
                    self.config.output_dir + "/Shadow_" + time_code + ".tif",
                    shadow,
                    self.raster_data.trf_arr,
                    self.raster_data.crs_wkt,
                    self.raster_data.nd_val,
                    coerce_f64_to_f32=True,
                )

        # Abort if loop was broken
        if not self.proceed:
            return

        # Save POI results
        if self.poi_results:
            self.save_poi_results()

        # Save WOI results
        if self.woi_results:
            self.save_woi_results()

        # Save average Tmrt
        if self.iters_count > 0:
            tmrt_avg = tmrt_agg / self.iters_count
            common.save_raster(
                self.config.output_dir + "/Tmrt_average.tif",
                tmrt_avg,
                self.raster_data.trf_arr,
                self.raster_data.crs_wkt,
                self.raster_data.nd_val,
                coerce_f64_to_f32=True,
            )

    def run_tiled(self) -> None:
        """Run SOLWEIG with tiled processing for large rasters (Tile -> Timestep loop)."""
        logger.info("Starting tiled execution")

        # Posture settings (same as standard)
        if self.params.Tmrt_params.Value.posture == "Standing":
            posture = self.params.Posture.Standing.Value
        else:
            posture = self.params.Posture.Sitting.Value

        first = np.round(posture.height)
        if first == 0.0:
            first = 1.0
        second = np.round(posture.height * 20.0)

        # Time variables
        num = len(self.environ_data.Ta)
        if num == 0:
            logger.error("No timesteps to process")
            return

        if self.environ_data.Ta.__len__() == 1:
            timestepdec = 0
        else:
            timestepdec = self.environ_data.dectime[1] - self.environ_data.dectime[0]

        # Prepare output files - generate time codes and create empty rasters
        logger.info("Initializing output rasters...")
        time_codes = []
        for i in range(num):
            if self.environ_data.altitude[i] > 0:
                w = "D"
            else:
                w = "N"
            XH = "0" if self.environ_data.hours[i] < 10 else ""
            XM = "0" if self.environ_data.minu[i] < 10 else ""

            time_code = (
                str(int(self.environ_data.YYYY[i]))
                + "_"
                + str(int(self.environ_data.DOY[i]))
                + "_"
                + XH
                + str(int(self.environ_data.hours[i]))
                + XM
                + str(int(self.environ_data.minu[i]))
                + w
            )
            time_codes.append(time_code)

            # Create empty rasters for enabled outputs
            outputs = [
                ("output_tmrt", "Tmrt"),
                ("output_kup", "Kup"),
                ("output_kdown", "Kdown"),
                ("output_sh", "Shadow"),
            ]

            for cfg_attr, prefix in outputs:
                if getattr(self.config, cfg_attr):
                    out_path = str(self.config.output_dir) + "/" + prefix + "_" + time_code + ".tif"
                    common.create_empty_raster(
                        out_path,
                        self.rows,
                        self.cols,
                        self.transform,
                        str(self.crs) if self.crs else "",
                        nodata=-9999.0,
                    )

        # Create average Tmrt raster
        common.create_empty_raster(
            str(self.config.output_dir) + "/Tmrt_average.tif",
            self.rows,
            self.cols,
            self.transform,
            str(self.crs) if self.crs else "",
            nodata=-9999.0,
        )

        # Prepare progress
        self.prep_progress(self.tile_manager.total_tiles * num)

        # Initialize state for all tiles
        logger.info("Initializing state for all tiles...")
        tile_states = []
        tmrt_agg_tiles = []

        tiles_list = list(self.tile_manager.get_tiles())
        if len(tiles_list) == 0:
            logger.error("No tiles generated by TileManager")
            return

        logger.info(f"Initializing {len(tiles_list)} tiles...")

        for tile in tiles_list:
            # Load minimal data for TgMaps (lcgrid)
            if self.config.use_landcover:
                lcgrid = common.read_raster_window(self.config.lc_path, tile.full_slice)
            else:
                lcgrid = None

            # Mock RasterData for TgMaps initialization
            mock_rd = SimpleNamespace(
                rows=tile.full_shape[0],
                cols=tile.full_shape[1],
                lcgrid=lcgrid,
            )

            tg_maps = TgMaps(self.config.use_landcover, self.params, mock_rd)
            tile_states.append(tg_maps)
            tmrt_agg_tiles.append(np.zeros(tile.core_shape, dtype=np.float32))

        # Reset time variables
        elvis = 0.0
        firstdaytime = 1.0
        timeadd = 0.0

        # Cache for steradians (constant across tiles and timesteps)
        cached_steradians = None

        # Iterate timesteps
        for i in range(num):
            logger.debug(f"Processing timestep {i + 1}/{num}")

            # Capture current time state
            current_firstdaytime = firstdaytime
            current_timeadd = timeadd
            current_timestepdec = timestepdec

            next_firstdaytime = None
            next_timeadd = None
            next_timestepdec = None

            # Iterate tiles
            for tile_idx, tile in enumerate(self.tile_manager.get_tiles()):
                self.proceed = self.iter_progress()
                if not self.proceed:
                    break

                # Load tile data
                self.svf_data = SvfData(self.config, tile_spec=tile)
                self.raster_data = RasterData(
                    self.config,
                    self.params,
                    self.svf_data,
                    self.amax_local_window_m,
                    self.amax_local_perc,
                    tile_spec=tile,
                )

                # Restore state
                self.tg_maps = tile_states[tile_idx]

                # Initialize other components for this tile
                self.shadow_mats = ShadowMatrices(self.config, self.params, self.svf_data, tile_spec=tile)

                # Restore cached steradians if available
                if cached_steradians is not None:
                    self.shadow_mats.steradians = cached_steradians

                self.walls_data = WallsData(
                    self.config,
                    self.params,
                    self.raster_data,
                    self.environ_data,
                    self.tg_maps,
                    tile_spec=tile,
                )

                # Run calculation
                (
                    Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow,
                    res_firstdaytime, res_timestepdec, res_timeadd,
                    Tgmap1_new, Tgmap1E_new, Tgmap1S_new, Tgmap1W_new, Tgmap1N_new,
                    Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth,
                    KsideI, TgOut1_new, TgOut, radIout, radDout, Lside,
                    Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside,
                    steradians_new, voxelTable,
                ) = self.calc_solweig(
                    i, elvis, first, second,
                    current_firstdaytime, current_timeadd, current_timestepdec, posture,
                )

                # Update tile state with new thermal arrays
                tile_states[tile_idx].Tgmap1 = Tgmap1_new
                tile_states[tile_idx].Tgmap1E = Tgmap1E_new
                tile_states[tile_idx].Tgmap1S = Tgmap1S_new
                tile_states[tile_idx].Tgmap1W = Tgmap1W_new
                tile_states[tile_idx].Tgmap1N = Tgmap1N_new
                tile_states[tile_idx].TgOut1 = TgOut1_new

                # Update steradians cache
                if steradians_new is not None:
                    self.shadow_mats.steradians = steradians_new

                # Capture next state
                next_firstdaytime = res_firstdaytime
                next_timestepdec = res_timestepdec
                next_timeadd = res_timeadd

                # Cache steradians if first calculation
                if cached_steradians is None and steradians_new is not None and np.any(steradians_new):
                    cached_steradians = steradians_new

                # Crop results to core (remove buffer)
                core_slice = tile.core_slice()
                Tmrt_core = Tmrt[core_slice]

                # Write outputs
                time_code = time_codes[i]

                if self.config.output_tmrt:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Tmrt_" + time_code + ".tif",
                        Tmrt_core,
                        tile.write_window.to_slices(),
                    )
                if self.config.output_kup:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Kup_" + time_code + ".tif",
                        Kup[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_kdown:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Kdown_" + time_code + ".tif",
                        Kdown[core_slice],
                        tile.write_window.to_slices(),
                    )
                if self.config.output_sh:
                    common.write_raster_window(
                        str(self.config.output_dir) + "/Shadow_" + time_code + ".tif",
                        shadow[core_slice],
                        tile.write_window.to_slices(),
                    )

                # Aggregate Tmrt (handle NaN and inf values safely)
                if (~np.isfinite(Tmrt_core)).any():
                    n_invalid = (~np.isfinite(Tmrt_core)).sum()
                    logger.warning(f"Timestep {i + 1}, Tile {tile_idx + 1}: {n_invalid} non-finite Tmrt values")

                tmrt_core_safe = np.nan_to_num(Tmrt_core, nan=0.0, posinf=0.0, neginf=0.0)
                tmrt_agg_tiles[tile_idx] += tmrt_core_safe

                # Clean up tile data to free memory
                self.svf_data = None
                self.raster_data = None
                self.shadow_mats = None
                self.walls_data = None

            if not self.proceed:
                break

            # Update state for next timestep
            if next_firstdaytime is not None:
                firstdaytime = next_firstdaytime
                timestepdec = next_timestepdec
                timeadd = next_timeadd

        # Abort if loop was broken
        if not self.proceed:
            return

        # Save POI results if any
        if self.poi_results:
            self.save_poi_results()

        # Write average Tmrt
        logger.info("Writing average Tmrt raster...")
        for tile_idx, tile in enumerate(self.tile_manager.get_tiles()):
            tmrt_avg_tile = tmrt_agg_tiles[tile_idx] / num
            common.write_raster_window(
                str(self.config.output_dir) + "/Tmrt_average.tif",
                tmrt_avg_tile,
                tile.write_window.to_slices(),
            )

        logger.info("Tiled execution complete")
