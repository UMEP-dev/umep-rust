# SOLWEIG

High-performance urban microclimate model for computing Mean Radiant Temperature (Tmrt) and thermal comfort indices (UTCI, PET).

**Rust + Python** performance-critical algorithms with GPU and tiled processing support.

> This package is currently in testing as a proof of concept. Please open an issue if you have any feedback or suggestions.

## Documentation

- [Quick Start Guide](docs/getting-started/quick-start.md) - Detailed tutorial
- [API Reference](docs/) - Full documentation site (MkDocs, 25 pages)
- [Physics Specifications](specs/) - Scientific documentation (10 specs)
- [ROADMAP.md](ROADMAP.md) - Development priorities

## Original Code

This package is adapted from the GPLv3-licensed [UMEP-processing](https://github.com/UMEP-dev/UMEP-processing) by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

Licensed under GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

**Citation:**

Adapted from UMEP (Urban Multi-scale Environmental Predictor) by Fredrik Lindberg, Sue Grimmond, and contributors. If you use this plugin in research, please cite:

> Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## Installation

```bash
# Clone and install
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
uv sync                  # Install Python dependencies
maturin develop          # Build Rust extension
```

## Quick Start

```python
import solweig
from datetime import datetime

# Create surface from DSM array
surface = solweig.SurfaceData(dsm=my_dsm_array, pixel_size=1.0)

# Define location and weather
location = solweig.Location(latitude=57.7, longitude=12.0)
weather = solweig.Weather(
    datetime=datetime(2024, 7, 15, 12, 0),
    ta=25.0,        # Air temperature (°C)
    rh=50.0,        # Relative humidity (%)
    global_rad=800.0  # Global radiation (W/m²)
)

# Calculate Tmrt
result = solweig.calculate(surface, location, weather)
print(f"Tmrt: {result.tmrt.mean():.1f}°C")
```

## Loading from GeoTIFFs

```python
import solweig

# Load and prepare surface data (auto-computes walls/SVF)
surface = solweig.SurfaceData.prepare(
    dsm="data/dsm.tif",
    working_dir="cache/",       # Walls/SVF cached here
    cdsm="data/cdsm.tif",       # Optional: vegetation
)

# Load weather from EPW file
weather_list = solweig.Weather.from_epw(
    "data/weather.epw",
    start="2023-07-01",
    end="2023-07-03",
)

# Calculate timeseries
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
    output_dir="output/",
)
```

## Post-Processing (UTCI/PET)

Thermal comfort indices can be computed directly from results:

```python
# Single timestep: compute directly from result
result = solweig.calculate(surface, location, weather)
utci = result.compute_utci(weather)  # Fast polynomial
pet = result.compute_pet(weather)    # Slower iterative solver

# Batch processing: from saved Tmrt files
solweig.compute_utci(tmrt_dir="output/", weather_series=weather_list, output_dir="utci/")
solweig.compute_pet(tmrt_dir="output/", weather_series=weather_list, output_dir="pet/")
```

## Input Validation

Validate inputs before expensive calculations:

```python
try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")
    result = solweig.calculate(surface, location, weather)
except solweig.GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field}")
except solweig.MissingPrecomputedData as e:
    print(f"Missing data: {e}")
```

## Demos

Complete working examples are in the [demos/](demos/) folder:

- [demos/athens-demo.py](demos/athens-demo.py) - Full workflow with GeoTIFFs
- [demos/solweig_gbg_test.py](demos/solweig_gbg_test.py) - Gothenburg test data

## QGIS Plugin

SOLWEIG is available as a QGIS Processing plugin for interactive spatial analysis:

1. Open QGIS → **Plugins** → **Manage and Install Plugins**
2. Go to **Settings** tab → Check **"Show also experimental plugins"**
3. Search for **"SOLWEIG"** in the **All** tab
4. Click **Install Plugin**

See [qgis_plugin/](qgis_plugin/) for source code and development details.

## Build & Test

```bash
maturin develop          # Build Rust extension
pytest tests/            # Run all 353 tests
poe verify_project       # Full verification (format, lint, test)
```

---

## Development

### Project Structure

```text
pysrc/solweig/              # Python source (modular architecture)
  api.py                    # Public API re-exports
  models/                   # Dataclass package (~3,080 lines)
  components/               # Modular component functions
  computation.py            # Core orchestration logic
  timeseries.py             # Batch time series processing
  tiling.py                 # Large raster tiling support
rust/                       # Rust extensions via maturin
qgis_plugin/                # QGIS Processing plugin
tests/                      # 353 tests (100% pass rate)
  golden/                   # Reference data validation
  spec/                     # Physical property tests
docs/                       # MkDocs documentation site
specs/                      # Markdown specifications
```

### Development Setup

```bash
# Install dependencies
uv sync

# Build Rust extension for development
maturin develop

# Run tests with coverage
pytest tests/ --cov=pysrc/solweig

# Format and lint
ruff format pysrc/ tests/
ruff check pysrc/ tests/ --fix

# Type checking
ty pysrc/

# Full verification pipeline
poe verify_project
```

### Building Documentation

```bash
# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

### QGIS Plugin Bundle Preparation

> The easiest way is to let GitHub Actions build the package — push a version tag (e.g. `v0.1.0`) or trigger the workflow manually. Download the universal ZIP from the release artifacts and upload it to the QGIS Plugin Repository.

The plugin source lives in `qgis_plugin/solweig_qgis/` and is built using the `qgis_plugin/build_plugin.py` script, which compiles the Rust extension, bundles it with the Python library, and creates a distributable ZIP.

#### 1. Update Plugin Metadata

Edit `qgis_plugin/solweig_qgis/metadata.txt` and increment the version.

#### 2. Build the Plugin

```bash
# Build for current platform (builds Rust extension + bundles Python library)
python qgis_plugin/build_plugin.py

# Build and create a distributable ZIP for the current platform
python qgis_plugin/build_plugin.py --package

# Create a universal multi-platform ZIP from pre-built wheels in dist/
# (this is what GitHub Actions uses)
python qgis_plugin/build_plugin.py --universal
```

#### 3. Test Plugin Locally

```bash
# Symlink for development (macOS)
ln -s "$(pwd)/qgis_plugin/solweig_qgis" ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/solweig_qgis
```

Then:

1. Open QGIS
2. **Plugins** → **Manage and Install Plugins** → **Installed**
3. Enable **SOLWEIG**
4. Test in **Processing Toolbox** → **SOLWEIG**

#### 4. Upload to QGIS Plugin Repository

1. Register at <https://plugins.qgis.org/>
2. Log in → **My Plugins** → **Upload a plugin**
3. Select the ZIP from `qgis_plugin/solweig-qgis-*-universal.zip`
4. Check **"Experimental"** for pre-release versions
5. Add changelog/release notes
6. Click **Upload**

### Release Checklist

- [ ] Version incremented in `qgis_plugin/solweig_qgis/metadata.txt`
- [ ] All tests passing (`pytest tests/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Plugin tested in QGIS locally
- [ ] ZIP bundle created and validated
- [ ] Uploaded to plugins.qgis.org

### Tooling Preferences

| Tool     | Use For                | Instead Of                   |
| -------- | ---------------------- | ---------------------------- |
| **uv**   | Package management     | pip, poetry, pipenv          |
| **ruff** | Linting and formatting | black, isort, flake8, pylint |
| **ty**   | Type checking          | mypy, pyright                |

### Code Metrics

- **api.py**: 403 lines (simplified from 3,976)
- **models/ package**: ~3,080 lines (6 modules)
- **Component functions**: All ≤ 455 lines
- **Test count**: 353 tests (100% pass rate)
- **Legacy code removed**: 6,100 lines
