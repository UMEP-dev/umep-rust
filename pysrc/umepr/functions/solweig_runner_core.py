from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pvlib.iotools import read_epw
from rasterio.transform import Affine, rowcol
from tqdm import tqdm

from ..class_configs import SolweigConfig, WeatherData
from .solweig_runner import SolweigRun


class SolweigRunCore(SolweigRun):
    """Run SOLWEIG in standalone mode without QGIS."""

    def __init__(self, config_path_str: str, params_json_path: str):
        config = SolweigConfig()
        config.from_file(config_path_str)
        super().__init__(config, params_json_path, qgis_env=False)

    def prep_progress(self, num: int) -> None:
        """Prepare progress for environment."""
        self.iters_total = num
        self.iters_count = 0
        self.progress = tqdm(total=num, desc="Running SOLWEIG", unit="step")

    def iter_progress(self) -> bool:
        """Iterate progress."""
        self.iters_count += 1
        self.progress.update(1)
        return True

    def load_poi_data(self, trf_arr: list[float]) -> tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        pois_gdf = gpd.read_file(self.config.poi_file)
        trf = Affine.from_gdal(*trf_arr)
        self.poi_pixel_xys = np.zeros((len(pois_gdf), 3)) - 999
        self.poi_names = []
        for n, (idx, row) in enumerate(pois_gdf.iterrows()):
            self.poi_names.append(idx)
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            self.poi_pixel_xys[n] = (n, x, y)

    def save_poi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save points of interest (POIs) results to a file."""
        # Convert pixel coordinates to geographic coordinates
        pois_gdf = gpd.GeoDataFrame(
            self.poi_results,
            geometry=gpd.points_from_xy(
                self.poi_pixel_xys[:, 1] * trf_arr[1] + trf_arr[0],
                self.poi_pixel_xys[:, 2] * trf_arr[1] + trf_arr[3],
            ),
            crs=crs_wkt,
        )
        # Create a datetime column for multi-index
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
        # GPD doesn't handle multi-index
        pois_gdf.to_file(self.config.output_dir + "/POI.gpkg", driver="GPKG")

    def load_woi_data(self, trf_arr: list[float]) -> tuple[Any, Any]:
        """Load walls of interest (WOIs) from a file."""
        woi_gdf = gpd.read_file(self.config.woi_file)
        trf = Affine.from_gdal(*trf_arr)
        self.woi_pixel_xys = np.zeros((len(woi_gdf), 3)) - 999
        self.woi_names = []
        for n, (idx, row) in enumerate(woi_gdf.iterrows()):
            self.woi_names.append(idx)
            y, x = rowcol(trf, row["geometry"].centroid.x, row["geometry"].centroid.y)
            self.woi_pixel_xys[n] = (n, x, y)

    def save_woi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save walls of interest (WOIs) results to a file."""
        # Convert pixel coordinates to geographic coordinates
        woi_gdf = gpd.GeoDataFrame(
            self.woi_results,
            geometry=gpd.points_from_xy(
                self.woi_pixel_xys[:, 1] * trf_arr[1] + trf_arr[0],
                self.woi_pixel_xys[:, 2] * trf_arr[1] + trf_arr[3],
            ),
            crs=crs_wkt,
        )
        # Create a datetime column for multi-index
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
        # GPD doesn't handle multi-index
        woi_gdf.to_file(self.config.output_dir + "/WOI.gpkg", driver="GPKG")

    def load_epw_weather(self) -> WeatherData:
        """Load weather data from an EPW file."""
        epw_df, epw_info = read_epw(self.config.epw_path)
        # Filter by date range
        filtered_df = epw_df.loc[self.config.epw_start_date : self.config.epw_end_date]
        # Filter by hours
        filtered_df = filtered_df[filtered_df.index.hour.isin(self.config.epw_hours)]
        # raise if empty
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
                "pres": filtered_df["atmospheric_pressure"].astype(float),  # Pascal, ensure float
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
        # Check for negative Kdown values
        umep_df_filt = umep_df[(umep_df["Kdown"] < 0) & (umep_df["Kdown"] > 1300)]
        if len(umep_df_filt):
            raise ValueError(
                "Error: Kdown - beyond what is expected",
            )

        # use -999 for NaN to mesh with UMEP
        umep_df = umep_df.fillna(-999)

        return WeatherData(
            DOY=umep_df["id"].to_numpy(),
            hours=umep_df["it"].to_numpy(),
            minu=umep_df["imin"].to_numpy(),
            Ta=umep_df["Tair"].to_numpy(),
            RH=umep_df["RH"].to_numpy(),
            radG=umep_df["Kdown"].to_numpy(),
            radD=umep_df["ldown"].to_numpy(),
            radI=umep_df["Kdiff"].to_numpy(),
            P=umep_df["pres"].to_numpy() / 100.0,  # convert from Pa to hPa,
            Ws=umep_df["Wind"].to_numpy(),
        )
