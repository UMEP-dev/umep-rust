# SOLWEIG QGIS Plugin

QGIS Processing plugin for SOLWEIG (Solar and Longwave Environmental Irradiance Geometry model).

## Overview

This plugin wraps SOLWEIG's Python API to provide native QGIS Processing framework integration. It enables calculation of Mean Radiant Temperature (Tmrt), UTCI, and PET thermal comfort indices directly within QGIS.

**Key Features:**
- Native QGIS Processing Toolbox integration
- Model Builder and batch mode support
- Auto-detects GDAL backend (no rasterio required in QGIS/OSGeo4W)
- Progress reporting via QgsProcessingFeedback
- Outputs auto-load to canvas with thermal comfort color ramps

## Installation

1. Copy the `solweig_qgis/` directory to your QGIS plugins folder:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`

2. Enable the plugin in QGIS: Plugins → Manage and Install Plugins → Installed → SOLWEIG

3. Access algorithms via Processing Toolbox → SOLWEIG

## Algorithms

### Preprocessing

| Algorithm | Description |
|-----------|-------------|
| **Compute Sky View Factor** | Pre-compute SVF arrays for reuse across timesteps |

### Calculation

| Algorithm | Description |
|-----------|-------------|
| **Calculate Tmrt (Single Timestep)** | Calculate Mean Radiant Temperature for one datetime |
| **Calculate Tmrt (Timeseries)** | Multi-timestep calculation with thermal state accumulation |
| **Calculate Tmrt (Large Rasters)** | Memory-efficient tiled processing for large areas |

### Post-Processing

| Algorithm | Description |
|-----------|-------------|
| **Compute UTCI** | Universal Thermal Climate Index (fast polynomial) |
| **Compute PET** | Physiological Equivalent Temperature (detailed model) |

### Utilities

| Algorithm | Description |
|-----------|-------------|
| **Import EPW Weather File** | Preview and validate EnergyPlus weather files |

## Directory Structure

```
qgis_plugin/
├── README.md                       # This file
├── build_plugin.py                 # Build script for bundled distribution
│
└── solweig_qgis/                   # Plugin package (install this to QGIS)
    ├── __init__.py                 # Plugin entry point with classFactory()
    ├── metadata.txt                # QGIS plugin metadata
    ├── provider.py                 # SolweigProvider (registers algorithms)
    ├── _bundled/                   # Bundled SOLWEIG library (for distribution)
    │
    ├── algorithms/
    │   ├── __init__.py
    │   ├── base.py                 # SolweigAlgorithmBase (shared utilities)
    │   │
    │   ├── preprocess/
    │   │   ├── __init__.py
    │   │   └── svf_preprocessing.py    # "Compute Sky View Factor"
    │   │
    │   ├── calculation/
    │   │   ├── __init__.py
    │   │   ├── single_timestep.py      # "Calculate Tmrt (Single Timestep)"
    │   │   ├── timeseries.py           # "Calculate Tmrt (Timeseries)"
    │   │   └── tiled_processing.py     # "Calculate Tmrt (Large Rasters)"
    │   │
    │   ├── postprocess/
    │   │   ├── __init__.py
    │   │   ├── utci.py                 # "Compute UTCI"
    │   │   └── pet.py                  # "Compute PET"
    │   │
    │   └── utilities/
    │       ├── __init__.py
    │       └── epw_import.py           # "Import EPW Weather File"
    │
    └── utils/
        ├── __init__.py
        ├── parameters.py               # Common parameter builders
        └── converters.py               # QGIS ↔ solweig dataclass conversion
```

---

## Implementation Checklist

### Phase 1: Plugin Skeleton ✅

- [x] **1.1** Create `__init__.py` with `classFactory()` entry point
- [x] **1.2** Create `metadata.txt` with plugin metadata
- [x] **1.3** Create `provider.py` with `SolweigProvider` class
- [ ] **1.4** Create placeholder icon.png
- [ ] **1.5** Test plugin loads in QGIS (empty provider)

### Phase 2: Shared Utilities ✅

- [x] **2.1** Create `algorithms/__init__.py`
- [x] **2.2** Create `algorithms/base.py` with `SolweigAlgorithmBase`:
  - [x] `load_raster_from_layer()` - QGIS layer → numpy array via GDAL
  - [x] `load_optional_raster()` - Handle optional raster parameters
  - [x] `save_georeferenced_output()` - Save with CRS/transform via solweig.io
  - [x] `add_raster_to_canvas()` - Add layer to QGIS project
  - [x] `apply_thermal_comfort_style()` - Apply UTCI/Tmrt color ramps
- [x] **2.3** Create `utils/__init__.py`
- [x] **2.4** Create `utils/parameters.py` with common parameter builders:
  - [x] `add_surface_parameters()` - DSM, CDSM, DEM, TDSM, LAND_COVER
  - [x] `add_location_parameters()` - LAT, LON, UTC_OFFSET, AUTO_EXTRACT
  - [x] `add_weather_parameters()` - DATETIME, TA, RH, RAD, WIND
  - [x] `add_human_parameters()` - POSTURE, ABS_K
- [x] **2.5** Create `utils/converters.py`:
  - [x] `create_surface_from_parameters()` - Build SurfaceData from QGIS params
  - [x] `create_location_from_parameters()` - Build Location from params
  - [x] `create_weather_from_parameters()` - Build Weather from params

### Phase 3: Single Timestep Algorithm ✅

- [x] **3.1** Create `algorithms/calculation/__init__.py`
- [x] **3.2** Create `algorithms/calculation/single_timestep.py`:
  - [x] Define all input parameters (surface, location, weather, human, options)
  - [x] Define output parameters (TMRT, optional SHADOW, KDOWN)
  - [x] Implement `processAlgorithm()`:
    - [x] Load rasters from QGIS layers
    - [x] Create SurfaceData, Location, Weather, HumanParams
    - [x] Handle height conversion (relative → absolute)
    - [x] Call `solweig.calculate()`
    - [x] Save output GeoTIFF
    - [x] Add to canvas with styling
- [x] **3.3** Register in provider
- [ ] **3.4** Test in QGIS with Gothenburg test data

### Phase 4: SVF Preprocessing Algorithm ✅

- [x] **4.1** Create `algorithms/preprocess/__init__.py`
- [x] **4.2** Create `algorithms/preprocess/svf_preprocessing.py`:
  - [x] Define input parameters (DSM, CDSM, DEM, TDSM, TRANS_VEG, OUTPUT_DIR)
  - [x] Define output parameters (SVF_DIR, SVF_FILE)
  - [x] Implement `processAlgorithm()`:
    - [x] Load rasters
    - [x] Create SurfaceData
    - [x] Call `surface.prepare()` with working_dir
    - [x] Report progress via feedback
- [x] **4.3** Register in provider
- [ ] **4.4** Test SVF computation and caching

### Phase 5: Timeseries Algorithm ✅

- [x] **5.1** Create `algorithms/calculation/timeseries.py`:
  - [x] Add EPW_FILE, START_DATE, END_DATE, HOURS_FILTER parameters
  - [x] Add OUTPUT_DIR, OUTPUTS selection parameters
  - [x] Implement `processAlgorithm()`:
    - [x] Load and filter weather from EPW
    - [x] Create surface and location
    - [x] Call `solweig.calculate_timeseries()`
    - [x] Report progress per timestep
    - [x] Handle cancellation
- [x] **5.2** Register in provider
- [ ] **5.3** Test with multi-day EPW data

### Phase 6: UTCI Algorithm ✅

- [x] **6.1** Create `algorithms/postprocess/__init__.py`
- [x] **6.2** Create `algorithms/postprocess/utci.py`:
  - [x] Define TMRT_DIR, EPW_FILE, OUTPUT_DIR parameters
  - [x] Implement `processAlgorithm()`:
    - [x] Load weather series from EPW
    - [x] Call `solweig.compute_utci()`
    - [x] Report file count
- [x] **6.3** Register in provider
- [ ] **6.4** Test UTCI computation

### Phase 7: PET Algorithm ✅

- [x] **7.1** Create `algorithms/postprocess/pet.py`:
  - [x] Add human body parameters (AGE, WEIGHT, HEIGHT, SEX, ACTIVITY, CLOTHING)
  - [x] Implement `processAlgorithm()` calling `solweig.compute_pet()`
- [x] **7.2** Register in provider
- [ ] **7.3** Test PET computation

### Phase 8: EPW Import Utility ✅

- [x] **8.1** Create `algorithms/utilities/__init__.py`
- [x] **8.2** Create `algorithms/utilities/epw_import.py`:
  - [x] Define EPW_FILE input parameter
  - [x] Implement `processAlgorithm()`:
    - [x] Parse EPW with `solweig.io.read_epw()`
    - [x] Generate HTML report with location, date range, statistics
- [x] **8.3** Register in provider
- [ ] **8.4** Test with sample EPW files

### Phase 9: Tiled Processing Algorithm ✅

- [x] **9.1** Create `algorithms/calculation/tiled_processing.py`:
  - [x] Add TILE_SIZE, AUTO_TILE_SIZE parameters
  - [x] Implement `processAlgorithm()` calling `solweig.calculate_tiled()`
- [x] **9.2** Register in provider
- [ ] **9.3** Test with large raster

### Phase 10: Build & Distribution ✅

- [x] **10.1** Create `build_plugin.py` build script
- [x] **10.2** Set up `_bundled/` directory support in `__init__.py`
- [x] **10.3** Create GitHub Actions workflow for cross-platform builds
- [x] **10.4** Update README with build instructions

### Phase 11: Testing & Polish (Pending)

- [ ] **11.1** Add docstrings to all algorithms
- [ ] **11.2** Create help strings for QGIS Help panel
- [ ] **11.3** Test full workflow in QGIS
- [ ] **11.4** Verify outputs match standalone Python execution
- [ ] **11.5** Create icon.png
- [ ] **11.6** Update this README with usage examples

---

## Building & Distribution

The plugin can be distributed in two ways:

### Option A: Bundled Distribution (Recommended for Users)

This bundles the compiled Rust extension and Python modules into the plugin, so users don't need to install anything separately.

```bash
# Build for your current platform
cd qgis_plugin
python build_plugin.py

# Create distributable ZIP
python build_plugin.py --package --version 0.1.0

# Clean build artifacts
python build_plugin.py --clean
```

This creates a platform-specific ZIP file (e.g., `solweig-qgis-0.1.0-linux_x86_64.zip`) that can be installed directly in QGIS.

**Supported platforms:**

- Linux x86_64
- Windows x86_64
- macOS x86_64
- macOS aarch64 (Apple Silicon)

### Option B: Development Setup

For development or if you have SOLWEIG installed via pip:

1. Install SOLWEIG in your Python environment:

   ```bash
   pip install solweig
   # or for development
   cd /path/to/solweig && pip install -e .
   ```

2. Symlink the plugin to QGIS:

   ```bash
   # Linux
   ln -s /path/to/solweig/qgis_plugin/solweig_qgis ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/solweig_qgis

   # macOS
   ln -s /path/to/solweig/qgis_plugin/solweig_qgis ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/solweig_qgis

   # Windows (run as admin)
   mklink /D "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\solweig_qgis" "C:\path\to\solweig\qgis_plugin\solweig_qgis"
   ```

The plugin auto-detects SOLWEIG in this order:

1. Bundled (`_bundled/` directory)
2. System-installed (via pip)
3. Development path (`../pysrc/solweig`)

### CI/CD Automated Builds

The GitHub Actions workflow (`.github/workflows/build-qgis-plugin.yml`) automatically builds platform-specific ZIPs:

- Triggered on push to `main` or `dev` branches
- Builds wheels for all 4 platforms using maturin
- Creates GitHub releases on tag push (e.g., `v0.1.0`)

## Dependencies

**For bundled distribution:** No external dependencies required.

**For development:** The plugin requires the SOLWEIG Python package:

```bash
pip install solweig
```

Or point to development source:

```python
import sys
sys.path.insert(0, '/path/to/solweig/pysrc')
```

## Core Library Files Referenced

| File | Purpose |
|------|---------|
| `pysrc/solweig/api.py` | Entry points: `calculate()`, `calculate_timeseries()` |
| `pysrc/solweig/progress.py` | QgsProcessingFeedback integration |
| `pysrc/solweig/io.py` | GDAL backend, EPW parser |
| `pysrc/solweig/models/surface.py` | SurfaceData with height conversion |
| `pysrc/solweig/models/weather.py` | Weather.from_epw() |

## License

Same license as SOLWEIG core library.
