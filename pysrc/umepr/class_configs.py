from dataclasses import dataclass

import numpy as np

from . import common


class SvfData:
    """Class to handle SVF data loading and processing."""

    def __init__(self, in_path_str: str, use_cdsm: bool = False):
        """
        Loads SVF and shadow matrix results from disk and returns a SVFResults dataclass instance.
        """
        in_path_str = str(common.check_path(in_path_str, make_dir=False))

        # Load SVF rasters
        self.svf, _, _, _ = common.load_raster(in_path_str + "/" + "svf.tif")
        self.svf_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfE.tif")
        self.svf_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfS.tif")
        self.svf_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfW.tif")
        self.svf_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfN.tif")
        if use_cdsm:
            self.svf_veg, _, _, _ = common.load_raster(in_path_str + "/" + "svfveg.tif")
            self.svf_veg_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfEveg.tif")
            self.svf_veg_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfSveg.tif")
            self.svf_veg_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfWveg.tif")
            self.svf_veg_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfNveg.tif")
            self.svf_veg_blocks_bldg_sh, _, _, _ = common.load_raster(in_path_str + "/" + "svfaveg.tif")
            self.svf_veg_blocks_bldg_sh_east, _, _, _ = common.load_raster(in_path_str + "/" + "svfEaveg.tif")
            self.svf_veg_blocks_bldg_sh_south, _, _, _ = common.load_raster(in_path_str + "/" + "svfSaveg.tif")
            self.svf_veg_blocks_bldg_sh_west, _, _, _ = common.load_raster(in_path_str + "/" + "svfWaveg.tif")
            self.svf_veg_blocks_bldg_sh_north, _, _, _ = common.load_raster(in_path_str + "/" + "svfNaveg.tif")
        else:
            self.svf_veg = np.ones_like(self.svf)
            self.svf_veg_east = np.ones_like(self.svf)
            self.svf_veg_south = np.ones_like(self.svf)
            self.svf_veg_west = np.ones_like(self.svf)
            self.svf_veg_north = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_east = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_south = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_west = np.ones_like(self.svf)
            self.svf_veg_blocks_bldg_sh_north = np.ones_like(self.svf)


@dataclass
class WeatherData:
    """Class to handle weather data loading and processing."""

    DOY: np.ndarray
    hours: np.ndarray
    minu: np.ndarray
    Ta: np.ndarray
    RH: np.ndarray
    radG: np.ndarray
    radD: np.ndarray
    radI: np.ndarray
    P: np.ndarray
    Ws: np.ndarray

    def to_array(self) -> np.ndarray:
        """Convert weather data to a structured numpy array."""
        return np.array(
            [
                self.DOY,
                self.hours,
                self.minu,
                self.Ta,
                self.RH,
                self.radG,
                self.radD,
                self.radI,
                self.P,
                self.Ws,
            ]
        ).T


@dataclass
class SolweigConfig:
    """Configuration class for SOLWEIG parameters."""

    output_dir: str | None = None
    working_dir: str | None = None
    dsm_path: str | None = None
    svf_path: str | None = None
    wh_path: str | None = None
    wa_path: str | None = None
    use_epw_file: bool = False
    epw_path: str | None = None
    epw_start_date: str | None = None
    epw_end_date: str | None = None
    epw_hours: list[int] | None = None
    met_path: str | None = None
    cdsm_path: str | None = None
    tdsm_path: str | None = None
    dem_path: str | None = None
    lc_path: str | None = None
    aniso_path: str | None = None
    poi_path: str | None = None
    poi_field: str | None = None
    wall_path: str | None = None
    woi_path: str | None = None
    woi_field: str | None = None
    only_global: bool = True
    use_veg_dem: bool = True
    conifer: bool = False
    person_cylinder: bool = True
    utc: bool = True
    use_landcover: bool = True
    use_dem_for_buildings: bool = False
    use_aniso: bool = False
    use_wall_scheme: bool = False
    wall_type: str | None = "Brick"
    output_tmrt: bool = True
    output_kup: bool = True
    output_kdown: bool = True
    output_lup: bool = True
    output_ldown: bool = True
    output_sh: bool = True
    save_buildings: bool = True
    output_kdiff: bool = True
    output_tree_planter: bool = True
    wall_netcdf: bool = False

    def to_file(self, file_path: str):
        """Save configuration to a file."""
        with open(file_path, "w") as f:
            for key in type(self).__annotations__:
                value = getattr(self, key)
                if value is None:
                    value = ""  # Default to empty string if None
                if isinstance(self.__annotations__[key], bool):
                    f.write(f"{key}={int(value)}\n")
                else:
                    f.write(f"{key}={value}\n")

    def from_file(self, config_path_str: str):
        """Load configuration from a file."""
        config_path = common.check_path(config_path_str)
        with open(config_path) as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    if key in type(self).__annotations__:
                        if value.strip() == "":
                            value = None
                        if type(self).__annotations__[key] == bool:
                            setattr(self, key, value == "1" or value.lower() == "true")
                        else:
                            setattr(self, key, value)
                    else:
                        print(f"Unknown key in config: {key}")

    def validate(self):
        """Validate configuration parameters."""
        if not self.output_dir:
            raise ValueError("Output directory must be set.")
        self.output_dir = str(common.check_path(self.output_dir, make_dir=True))
        if not self.working_dir:
            raise ValueError("Working directory must be set.")
        self.working_dir = str(common.check_path(self.working_dir, make_dir=True))
        if not self.dsm_path:
            raise ValueError("DSM path must be set.")
        if (self.met_path is None and self.epw_path is None) or (self.met_path and self.epw_path):
            raise ValueError("Provide either MET or EPW weather file.")
        if self.epw_path is not None:
            if self.epw_start_date is None or self.epw_end_date is None:
                raise ValueError("EPW start and end dates must be provided if EPW path is set.")
            if self.epw_hours is None:
                self.epw_hours = list(range(24))  # Default to all hours if not specified
        # Add more validation as needed
