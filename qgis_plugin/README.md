# SOLWEIG QGIS Plugin

QGIS Processing plugin for SOLWEIG (Solar and Longwave Environmental Irradiance Geometry model).

## Overview

This plugin wraps SOLWEIG's Python API to provide native QGIS Processing framework integration. It enables calculation of Mean Radiant Temperature (Tmrt), UTCI, and PET thermal comfort indices directly within QGIS.

**Key features:**

- Native QGIS Processing Toolbox integration (Model Builder and batch mode)
- Automatic tiling for large rasters, sized to fit GPU memory
- GPU acceleration (Metal / Vulkan / DirectX) with CPU fallback
- Auto-detects GDAL backend (no rasterio required)
- Progress reporting and cancellation via QGIS Task Manager
- Single-timestep results auto-load to canvas with thermal comfort colour ramps
- Automatic `solweig` library installation on first use

**Documentation:** <https://umep-dev.github.io/solweig/>
**QGIS Plugin Guide:** <https://umep-dev.github.io/solweig/guide/qgis-plugin/>

## Installation

### From QGIS Plugin Repository

1. Open **Plugins** > **Manage and Install Plugins**
2. Go to the **Settings** tab and check **"Show also experimental plugins"**
3. Switch to the **All** tab and search for **"SOLWEIG"**
4. Click **Install Plugin**

On first use the plugin checks whether the `solweig` library is installed. If it is missing or outdated, a dialog offers to install or upgrade it automatically via pip.

### Development setup

For development, symlink the plugin directory into your QGIS plugins folder:

```bash
# Linux
ln -s /path/to/solweig/qgis_plugin/solweig_qgis \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/solweig_qgis

# macOS
ln -s /path/to/solweig/qgis_plugin/solweig_qgis \
  ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/solweig_qgis

# Windows (run as admin)
mklink /D "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\solweig_qgis" ^
  "C:\path\to\solweig\qgis_plugin\solweig_qgis"
```

Then install the library in editable mode:

```bash
cd /path/to/solweig && pip install -e .
```

## Algorithms

Three algorithms are registered in the Processing Toolbox under the **SOLWEIG** group, numbered to indicate the recommended workflow order. A fourth (2b) is available but not registered by default.

| # | Algorithm | File | Registered |
| --- | --------- | ---- | ---------- |
| 1 | Download / Preview Weather File | `algorithms/utilities/epw_import.py` | Yes |
| 2 | Prepare Surface Data (align, walls, SVF) | `algorithms/preprocess/surface_preprocessing.py` | Yes |
| 2b | Recompute Sky View Factor (advanced) | `algorithms/preprocess/svf_preprocessing.py` | No |
| 3 | SOLWEIG Calculation | `algorithms/calculation/solweig_calculation.py` | Yes |

**Workflow:** Download weather data (1) → Prepare surface once (2) → Run calculations (3) with different weather files or date ranges without re-preparing. Step 2b is only needed to recompute SVF with different parameters without re-running the full surface preparation.

## Directory Structure

```
qgis_plugin/
├── README.md                          # This file
├── build_plugin.py                    # Build script for distributable ZIP
└── solweig_qgis/                      # Plugin package
    ├── __init__.py                    # Plugin entry point (classFactory)
    ├── metadata.txt                   # QGIS plugin metadata
    ├── provider.py                    # SolweigProvider (registers algorithms)
    ├── icon.png                       # Plugin icon (32x32)
    ├── icon.svg                       # Plugin icon (vector source)
    ├── icon_128.png                   # Plugin icon (128x128)
    ├── algorithms/
    │   ├── base.py                    # SolweigAlgorithmBase (shared utilities)
    │   ├── utilities/
    │   │   └── epw_import.py          # 1. Download / Preview Weather File
    │   ├── preprocess/
    │   │   ├── surface_preprocessing.py   # 2. Prepare Surface Data
    │   │   └── svf_preprocessing.py       # 2b. Recompute SVF (advanced)
    │   └── calculation/
    │       └── solweig_calculation.py     # 3. SOLWEIG Calculation
    └── utils/
        ├── parameters.py              # Common parameter builders
        └── converters.py              # QGIS ↔ solweig dataclass conversion
```

## Building & Distribution

`build_plugin.py` creates a distributable ZIP for the QGIS Plugin Repository. The version is read from `pyproject.toml` (single source of truth) and stamped into `metadata.txt` before packaging.

```bash
cd qgis_plugin

# Build ZIP (version from pyproject.toml)
python build_plugin.py

# Override version
python build_plugin.py --version 0.2.0

# Clean old ZIPs
python build_plugin.py --clean
```

The GitHub Actions workflow (`.github/workflows/build-qgis-plugin.yml`) runs on version tag pushes and creates the plugin ZIP automatically.

To install the built ZIP in QGIS: **Plugins** > **Manage and Install Plugins** > **Install from ZIP**.

## Citation

If you use SOLWEIG in your research, please cite the original UMEP paper:

> Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software 99, 70-87 [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

## Original Code

This plugin is adapted from the GPLv3-licensed [UMEP-processing](https://github.com/UMEP-dev/UMEP-processing) by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

SOLWEIG plugin maintained by Gareth Simons and the SOLWEIG Development Team.

## License

GNU General Public License v3.0. Same license as SOLWEIG core library and original UMEP code.
