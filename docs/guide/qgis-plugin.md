# QGIS Plugin

SOLWEIG is available as a **QGIS Processing plugin** for point-and-click spatial analysis — no Python scripting required. All algorithms run inside the QGIS Processing Toolbox and produce standard GeoTIFF outputs that load directly into the map canvas.

!!! warning "Experimental"
    This package and plugin are experimental — released for testing and discussion purposes. The API is stabilising but may change. Feedback and bug reports are welcome via [GitHub Issues](https://github.com/UMEP-dev/solweig/issues).

## Requirements

- **QGIS 4.0+** (Qt6, Python 3.11+)
- The `solweig` Python library (installed automatically on first use)

## Installation

1. Open **Plugins** > **Manage and Install Plugins**
2. Go to the **Settings** tab and check **"Show also experimental plugins"**
3. Switch to the **All** tab and search for **"SOLWEIG"**
4. Click **Install Plugin**

On first use the plugin checks whether the `solweig` library is installed. If it is missing or outdated, a dialog offers to install or upgrade it automatically via pip. No manual setup is needed.

!!! tip "Manual library install"
    If the automatic install fails (e.g. behind a proxy), open the QGIS Python Console and run:

    ```python
    import pip; pip.main(["install", "--upgrade", "solweig"])
    ```

---

## Processing algorithms

Once installed, three algorithms appear in the **Processing Toolbox** under the SOLWEIG group. They are numbered to indicate the recommended workflow order.

### 1. Download / Preview Weather File

Download a Typical Meteorological Year (TMY) weather file from the EU PVGIS service, or preview the contents of an existing EPW file.

| Parameter | Description |
|-----------|-------------|
| **Mode** | *Download from PVGIS* or *Preview existing file* |
| **Latitude / Longitude** | Defaults to the map canvas centre |
| **Output EPW** | Save location for the downloaded file |
| **EPW File** | Path to an existing file (preview mode) |

**Download mode** fetches ERA5 reanalysis data from the [PVGIS API](https://re.jrc.ec.europa.eu/pvg_tools/en/) — global coverage, no API key required. **Preview mode** generates an HTML report with location, date range, and variable statistics.

### 2. Prepare Surface Data (align, walls, SVF)

Load raster layers, align them to a common grid, compute wall heights, wall aspects, and Sky View Factor. Results are cached to a working directory so subsequent calculations skip the expensive preprocessing.

| Parameter | Description |
|-----------|-------------|
| **DSM** | Digital Surface Model (required) |
| **CDSM** | Canopy/vegetation height model (optional) |
| **DEM** | Digital Elevation Model (optional) |
| **TDSM** | Trunk zone heights (optional) |
| **Land cover** | Surface classification grid (optional) |
| **Height modes** | Relative (to ground) or absolute for each input |
| **Extent** | Custom processing area (optional — defaults to DSM extent) |
| **Pixel size** | Output resolution in metres (0 = native DSM resolution) |
| **Vegetation settings** | Transmissivity, leaf-on/off dates, conifer flag |
| **Land cover materials** | Albedo and emissivity per surface class (advanced) |

**Output directory structure:**

```
prepared_surface/
  dsm.tif                    # Aligned DSM
  wall_height.tif            # Computed wall heights
  wall_aspect.tif            # Computed wall aspects
  cdsm.tif                   # Aligned CDSM (if provided)
  dem.tif                    # Aligned DEM (if provided)
  tdsm.tif                   # Aligned TDSM (if provided)
  land_cover.tif             # Aligned land cover (if provided)
  svfs.zip                   # Sky View Factor arrays
  shadowmats.npz             # Shadow matrices (for anisotropic sky)
  parametersforsolweig.json  # Saved material/vegetation settings
  metadata.json              # Grid info and completion marker
```

### 3. SOLWEIG Calculation

Run Tmrt and optional thermal comfort calculations using a prepared surface directory and weather data.

**Weather sources** (select one):

| Source | Description |
|--------|-------------|
| Single timestep | Enter date/time and weather values manually |
| EPW file | Load a timeseries from an EPW weather file |
| UMEP met file | Load a timeseries from a UMEP meteorological file |

**Date filtering** (EPW and UMEP modes): optional start/end dates and hour-of-day filter (e.g. `9,10,11,12,13,14,15,16,17` for daytime only).

**Key options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Anisotropic sky | On | Use Perez sky model (requires shadow matrices from surface prep) |
| Conifer | Off | Treat trees as evergreen (skip seasonal leaf-off) |
| Max shadow distance | 1 000 m | Increase for mountainous terrain |
| Compute UTCI | Off | Calculate Universal Thermal Climate Index |
| Compute PET | Off | Calculate Physiological Equivalent Temperature (~50x slower than UTCI) |
| Heat stress thresholds | — | UTCI thresholds for day/night exceedance grids |

**Output selection:** choose which per-timestep grids to save — Tmrt, shadow, Kdown, Kup, Ldown, Lup.

**Output directory structure:**

```
results/
  tmrt/    tmrt_YYYYMMDD_HHMM.tif
  shadow/  shadow_YYYYMMDD_HHMM.tif
  utci/    utci_YYYYMMDD_HHMM.tif
  ...
```

For timeseries runs, a summary report is logged with mean/max Tmrt, UTCI, sun hours, and heat stress exceedance statistics.

---

## Typical workflow

```
┌─────────────────────────────────────┐
│  1. Download / Preview Weather File │
│     → Get an EPW file               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  2. Prepare Surface Data            │
│     → Load DSM, compute walls & SVF │
│     → Cached for reuse              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  3. SOLWEIG Calculation             │
│     → Point to prepared directory   │
│     → Select EPW + date range       │
│     → Choose outputs (Tmrt, UTCI…)  │
│     → Results load into QGIS canvas │
└─────────────────────────────────────┘
```

1. **Get weather data** — run *Download / Preview Weather File* to fetch a TMY EPW, or use an EPW you already have.
2. **Prepare the surface** — run *Prepare Surface Data* with your DSM (and optionally CDSM, DEM, land cover). This computes walls and SVF once; subsequent calculations reuse the cache.
3. **Calculate** — run *SOLWEIG Calculation*, pointing to the prepared surface directory and your EPW file. Select your date range, enable UTCI/PET if needed, and run. Results are saved as GeoTIFFs and loaded into the canvas.

!!! tip "Re-running with different parameters"
    Surface preparation is the expensive step. Once complete, you can re-run the calculation step with different weather files, date ranges, or human parameters without re-preparing the surface.

---

## QGIS-specific features

**Automatic tiling** — Large rasters are automatically divided into overlapping tiles sized to fit GPU memory. Tile buffers account for building heights and shadow distance to ensure continuity at boundaries.

**GPU acceleration** — Shadow casting and anisotropic sky computations run on the GPU when available (Metal on macOS, Vulkan on Linux, DirectX on Windows). Falls back to multi-threaded CPU automatically.

**Progress and cancellation** — All algorithms report progress through the QGIS Task Manager with per-timestep status. Long-running calculations can be cancelled cleanly.

**Canvas integration** — Single-timestep results are automatically loaded into the QGIS canvas with thermal comfort colour ramps. Timeseries results are saved as GeoTIFFs for manual loading and styling.

**Dependency management** — The plugin detects whether the `solweig` library is installed and at the correct version. On first use or after a plugin update, it offers to install or upgrade automatically.

**GDAL backend** — The plugin uses QGIS's bundled GDAL for all raster I/O, avoiding the need for rasterio or other pip dependencies.
