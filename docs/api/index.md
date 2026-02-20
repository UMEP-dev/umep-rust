# API Reference

SOLWEIG provides a clean, minimal API for urban microclimate calculations.

## Quick Overview

### Core Functions

| Function | Description |
|----------|-------------|
| [`calculate()`](functions.md#calculate) | Single or multi-timestep Tmrt calculation (tiling is automatic for large rasters) |
| [`validate_inputs()`](functions.md#validate_inputs) | Pre-flight input validation |
| [`compute_utci_grid()`](functions.md#compute_utci_grid) | UTCI from Tmrt grid |
| [`compute_pet_grid()`](functions.md#compute_pet_grid) | PET from Tmrt grid |

### Data Classes

| Class | Description |
|-------|-------------|
| [`SurfaceData`](dataclasses.md#surfacedata) | Terrain data (DSM, CDSM, walls, SVF) |
| [`Location`](dataclasses.md#location) | Geographic coordinates |
| [`Weather`](dataclasses.md#weather) | Meteorological conditions |
| [`HumanParams`](dataclasses.md#humanparams) | Human body parameters |
| [`SolweigResult`](dataclasses.md#solweigresult) | Calculation output |
| [`TimeseriesSummary`](dataclasses.md#timeseriessummary) | Aggregated timeseries output |
| [`Timeseries`](dataclasses.md#timeseries) | Per-timestep scalar timeseries |
| [`ModelConfig`](dataclasses.md#modelconfig) | Model configuration |

### I/O Functions

| Function | Description |
|----------|-------------|
| [`load_raster()`](io.md#load_raster) | Load a GeoTIFF with optional bbox cropping |
| [`save_raster()`](io.md#save_raster) | Save array as Cloud-Optimized GeoTIFF |
| [`rasterise_gdf()`](io.md#rasterise_gdf) | Rasterise vector data to a height grid |
| [`download_epw()`](io.md#download_epw) | Download EPW weather from PVGIS |
| [`read_epw()`](io.md#read_epw) | Parse an EPW file to weather records |

### GPU Utilities

| Function | Description |
|----------|-------------|
| `is_gpu_available()` | Check if GPU acceleration is available |
| `get_compute_backend()` | Returns `"gpu"` or `"cpu"` |
| `disable_gpu()` | Disable GPU, fall back to CPU |

## Import Pattern

```python
import solweig

# All public API is available at the top level
surface = solweig.SurfaceData(dsm=my_dsm, pixel_size=1.0)
result = solweig.calculate(surface, location, weather)
```

## Type Annotations

SOLWEIG is fully typed. Enable type checking in your IDE for the best experience:

```python
from solweig import SurfaceData, Location, Weather, SolweigResult

def process_area(dsm: np.ndarray) -> SolweigResult:
    surface: SurfaceData = SurfaceData(dsm=dsm, pixel_size=1.0)
    location: Location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    weather: Weather = Weather(...)
    return solweig.calculate(surface, location, weather)
```
