# SOLWEIG

**Map how hot it *feels* across a city — pixel by pixel.**

SOLWEIG computes **Mean Radiant Temperature (Tmrt)** and thermal comfort indices (**UTCI**, **PET**) for urban environments. Give it a building height model and weather data, and it produces high-resolution maps showing where people experience heat stress — and where trees, shade, and cool surfaces make a difference.

Built on Rust for speed, with optional GPU acceleration. Handles everything from a single city block to an entire district.

> **Status:** Beta (v0.1.0). The API is stabilising. Feedback and bug reports welcome — [open an issue](https://github.com/UMEP-dev/solweig/issues).

## What can you do with it?

- **Urban planning** — Compare street canyon designs, tree planting scenarios, or cool-roof strategies by mapping thermal comfort before and after.
- **Heat risk assessment** — Identify the hottest spots in a neighbourhood during a heatwave, hour by hour.
- **Research** — Run controlled microclimate experiments at 1 m resolution with full radiation budgets.
- **Climate services** — Generate thermal comfort maps for public health warnings or outdoor event planning.

## How it works (in brief)

SOLWEIG models the complete radiation budget experienced by a person standing in an urban environment:

1. **Shadows** — Which pixels are shaded by buildings and trees at a given sun angle?
2. **Sky View Factor** — How much sky can a person see from each point? (More sky = more incoming radiation.)
3. **Surface temperatures** — How hot are the ground and surrounding walls?
4. **Radiation balance** — Sum shortwave (sun) and longwave (heat) radiation from all directions.
5. **Tmrt** — Convert total absorbed radiation into a single "felt temperature" metric.
6. **Thermal comfort** — Optionally derive UTCI or PET, which combine Tmrt with air temperature, humidity, and wind.

## Quick start

### Install

```bash
pip install solweig
```

### Minimal example (numpy arrays)

```python
import numpy as np
import solweig
from datetime import datetime

# A flat surface with one 15 m building
dsm = np.full((200, 200), 2.0, dtype=np.float32)
dsm[80:120, 80:120] = 15.0

surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)

location = solweig.Location(latitude=48.8, longitude=2.3, utc_offset=1)  # Paris
weather = solweig.Weather(
    datetime=datetime(2025, 7, 15, 14, 0),
    ta=32.0,          # Air temperature (°C)
    rh=40.0,          # Relative humidity (%)
    global_rad=850.0, # Solar radiation (W/m²)
)

result = solweig.calculate(surface, location, weather)

print(f"Sunlit Tmrt: {result.tmrt[result.shadow > 0.5].mean():.0f}°C")
print(f"Shaded Tmrt: {result.tmrt[result.shadow < 0.5].mean():.0f}°C")
```

### Real-world workflow (GeoTIFFs + EPW weather)

```python
import solweig

# 1. Load surface — walls and sky view factors computed and cached automatically
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    cdsm="data/trees.tif",       # Optional: vegetation canopy heights
    working_dir="cache/",        # Expensive preprocessing cached here
)

# 2. Load weather from an EPW file (standard format from climate databases)
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2025-07-01",
    end="2025-07-03",
)
location = solweig.Location.from_epw("data/weather.epw")

# 3. Run — outputs saved as GeoTIFFs, thermal state carried between timesteps
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    location=location,
    output_dir="output/",
    outputs=["tmrt", "shadow"],
)

# 4. Post-process thermal comfort (optional, runs on saved Tmrt files)
solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)
```

## What you need

| Input | Required? | What it is |
|-------|-----------|------------|
| **DSM** | Yes | Digital Surface Model — a height grid (metres) including buildings. GeoTIFF or numpy array. |
| **Location** | Yes | Latitude, longitude, and UTC offset. Can be extracted from the DSM's CRS or an EPW file. |
| **Weather** | Yes | Air temperature, relative humidity, and global solar radiation. Load from an EPW file or create manually. |
| **CDSM** | No | Canopy heights (trees). Adds vegetation shading. |
| **DEM** | No | Ground elevation. Separates terrain from buildings. |
| **Land cover** | No | Surface type grid (paved, grass, water, etc.). Affects surface temperatures. |

## What you get

| Output | Unit | Description |
|--------|------|-------------|
| **Tmrt** | °C | Mean Radiant Temperature — the main output. How much radiation a person absorbs. |
| **Shadow** | 0–1 | Shadow fraction (1 = sunlit, 0 = fully shaded). |
| **UTCI** | °C | Universal Thermal Climate Index — "feels like" temperature combining all factors. |
| **PET** | °C | Physiological Equivalent Temperature — similar to UTCI but with customisable body parameters. |
| Kdown / Kup | W/m² | Shortwave radiation (down and reflected up). |
| Ldown / Lup | W/m² | Longwave radiation (thermal, down and emitted up). |

### Don't have an EPW file? Download one

```python
# Download weather data for any location (no API key needed)
epw_path = solweig.download_epw(latitude=37.98, longitude=23.73, output_path="athens.epw")
weather_list = solweig.Weather.from_epw(epw_path)
```

## Demos

Complete working scripts you can run directly:

- **[demos/athens-demo.py](demos/athens-demo.py)** — Full workflow: rasterise tree vectors, load GeoTIFFs, run a multi-day timeseries, post-process UTCI.
- **[demos/solweig_gbg_test.py](demos/solweig_gbg_test.py)** — Gothenburg: surface preparation with SVF caching, timeseries calculation.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quick Start Guide](docs/getting-started/quick-start.md) — Step-by-step first calculation
- [User Guide](docs/guide/basic-usage.md) — Common workflows, height conventions, and options
- [API Reference](docs/api/index.md) — All classes and functions
- [Physics](docs/physics/index.md) — How the radiation model works

## QGIS Plugin

SOLWEIG is also available as a QGIS plugin for point-and-click spatial analysis:

1. **Plugins** → **Manage and Install Plugins**
2. **Settings** tab → Check **"Show also experimental plugins"**
3. Search for **"SOLWEIG"** → **Install Plugin**

## Citation

Adapted from [UMEP](https://github.com/UMEP-dev/UMEP-processing) by Fredrik Lindberg, Sue Grimmond, and contributors.

If you use SOLWEIG in your research, please cite the original model paper and the UMEP platform:

1. Lindberg F, Holmer B, Thorsson S (2008) SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology* 52, 697–713 [doi:10.1007/s00484-008-0162-7](https://doi.org/10.1007/s00484-008-0162-7)

2. Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) – An integrated tool for city-based climate services. *Environmental Modelling and Software* 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## Demo data

The Athens demo dataset (`demos/data/athens/`) uses the following sources:

- **DSM/DEM** — Derived from LiDAR data available via the [Hellenic Cadastre geoportal](https://www.ktimatologio.gr/)
- **Tree vectors** (`trees.gpkg`) — Derived from the [Athens Urban Atlas](https://land.copernicus.eu/local/urban-atlas) and municipal open data at [geodata.gov.gr](https://geodata.gov.gr/)
- **EPW weather** (`athens_2023.epw`) — Generated using Copernicus Climate Change Service information [2025] via [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/). Contains modified Copernicus Climate Change Service information; neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
