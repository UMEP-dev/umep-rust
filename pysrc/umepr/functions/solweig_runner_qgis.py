from typing import Any

from umep.functions.SOLWEIGpython.wallOfInterest import pointOfInterest

from ..class_configs import SolweigConfig
from .solweig_runner import SolweigRun


class SolweigRunQgis(SolweigRun):
    """Run SOLWEIG in QGIS environment."""

    def __init__(self, config_path_str: str, params_json_path: str, feedback: Any):
        """ """
        config = SolweigConfig()
        config.from_file(config_path_str)
        super().__init__(config, params_json_path, qgis_env=True)
        self.progress = feedback

    def prep_progress(self, num: int) -> None:
        """Prepare progress for environment."""
        self.iters_total = num
        self.iters_count = 0

    def iter_progress(self) -> bool:
        """Iterate progress."""
        self.iters_count += 1
        self.progress.setProgress(int(self.iters_count * (100.0 / self.iters_total)))  # move progressbar forward
        if self.progress.isCanceled():
            self.progress.setProgressText("Calculation cancelled")
            return False
        return True

    def load_poi_data(self, trf_arr: list[float]) -> tuple[Any, Any]:
        """Load points of interest (POIs) from a file."""
        scale = 1 / trf_arr[1]
        poi_names, poi_pixel_xys = pointOfInterest(self.config.poi_file, self.config.poi_field, scale, trf_arr)
        self.poi_names = poi_names
        self.poi_pixel_xys = poi_pixel_xys

    def save_poi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save points of interest (POIs) results to a text file with geographic coordinates."""
        if not self.poi_results:
            return
        # Extract header from first result and add lng/lat
        header = ["name"] + list(self.poi_results[0].keys()) + ["lng", "lat"]
        # Write results to file
        output_path = self.config.output_dir + "/POI_results.txt"
        with open(output_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for result in self.poi_pixel_xys:
                lng = result["pixel_x"] * trf_arr[1] + trf_arr[0]
                lat = result["pixel_y"] * trf_arr[1] + trf_arr[3]
                row_values = list(result.values()) + [lng, lat]
                f.write("\t".join(map(str, row_values)) + "\n")

    def load_woi_data(self, trf_arr: list[float]) -> tuple[Any, Any]:
        """Load walls of interest (WOIs) from a file."""
        scale = 1 / trf_arr[1]
        woi_names, woi_pixel_xys = pointOfInterest(self.config.woi_file, self.config.woi_field, scale, trf_arr)
        self.woi_names = woi_names
        self.woi_pixel_xys = woi_pixel_xys

    def save_woi_results(self, trf_arr: list[float], crs_wkt: str) -> None:
        """Save walls of interest (WOIs) results to a text file with geographic coordinates."""
        if not self.woi_results:
            return
        # Extract header from first result and add lng/lat
        header = ["name"] + list(self.woi_results[0].keys()) + ["lng", "lat"]
        # Write results to file
        output_path = self.config.output_dir + "/WOI_results.txt"
        with open(output_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for result in self.woi_pixel_xys:
                lng = result["pixel_x"] * trf_arr[1] + trf_arr[0]
                lat = result["pixel_y"] * trf_arr[1] + trf_arr[3]
                row_values = list(result.values()) + [lng, lat]
                f.write("\t".join(map(str, row_values)) + "\n")
