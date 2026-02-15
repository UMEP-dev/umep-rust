# Parameter & Configuration Migration Guide

**Updated: January 2026**

This document maps all original parameters and configuration options to their new API equivalents.

## Overview

The old API used two configuration systems:
1. **Config INI file** (`configsolweig.ini`) - Model behavior flags, file paths
2. **Params JSON file** (`parametersforsolweig.json`) - Physical constants, material properties

The new API simplifies this:
1. **Direct parameters** - Key model options as function arguments
2. **Bundled defaults** - Common physical constants in `default_params.json`
3. **Custom params** - Optional JSON file for landcover-specific properties
4. **Automatic preparation** - Many expensive values are prepared/cached via `SurfaceData.prepare()`

---

## Model Configuration Options

### Boolean Flags (Model Behavior)

| Old API (config.ini)     | New API                           | Status      | Notes |
|--------------------------|-----------------------------------|-------------|-------|
| `use_veg_dem`            | Automatic (from `surface.cdsm`)   | ✅ AUTO     | If CDSM provided, vegetation is used |
| `conifer`                | `conifer=True/False`              | ✅ DIRECT   | Direct parameter in `calculate()` |
| `use_aniso`              | `use_anisotropic_sky=True/False`  | ✅ DIRECT   | Direct parameter in `calculate()` |
| `use_landcover`          | Automatic (from `surface.land_cover`) | ⏳ TODO | Will be automatic when land_cover provided |
| `person_cylinder`        | `human.posture="standing"/"sitting"` | ✅ PARAMS | Via HumanParams object |
| `only_global`            | Removed (always use full radiation) | 🗑️ REMOVED | Simplified assumption |
| `use_dem_for_buildings`  | Automatic (from `surface.dem`)    | ✅ AUTO     | If DEM provided, used for building detection |

### File Paths (Input Data)

| Old API (config.ini)     | New API                           | Status      | Notes |
|--------------------------|-----------------------------------|-------------|-------|
| `dsm_path`               | `SurfaceData.prepare(dsm=...)`    | ✅ COMPLETE | Required input |
| `cdsm_path`              | `SurfaceData.prepare(cdsm=...)`   | ✅ COMPLETE | Optional vegetation |
| `tdsm_path`              | `SurfaceData.prepare(tdsm=...)`   | ✅ COMPLETE | Optional trunk zone |
| `dem_path`               | `SurfaceData.prepare(dem=...)`    | ✅ COMPLETE | Optional ground elevation |
| `lc_path`                | `SurfaceData.prepare(land_cover=...)` | ⏳ TODO | Landcover grid (planned) |
| `wall_path`              | `working_dir/walls/`              | ✅ AUTO     | Auto-generated and cached |
| `svf_path`               | `working_dir/svf/`                | ✅ AUTO     | Auto-generated and cached |
| `aniso_path`             | `working_dir/svf/shadowmats.npz`  | ✅ AUTO     | Auto-generated if use_aniso=True |
| `epw_path`               | `Weather.from_epw(path)`          | ✅ COMPLETE | EPW file loading |
| `output_path`            | `calculate_timeseries(output_dir=...)` | ✅ COMPLETE | Output directory |

### Preprocessing Control

| Old API (config.ini)     | New API                           | Status      | Notes |
|--------------------------|-----------------------------------|-------------|-------|
| Pre-generate walls       | Automatic + cached                | ✅ AUTO     | Generated during `SurfaceData.prepare()`, cached to working_dir |
| Pre-generate SVF         | Automatic + cached                | ✅ AUTO     | Generated during `SurfaceData.prepare()`, cached to working_dir |
| Pre-generate shadowmats  | Automatic + cached                | ✅ AUTO     | Generated during preparation when anisotropic data is requested |
| Wall limit (1.0m)        | Hardcoded default                 | ✅ AUTO     | No user control needed |

---

## Physical Parameters (Material Properties)

### Human Body Parameters (Tmrt)

| Old API (params.json)           | New API                           | Status      | Notes |
|---------------------------------|-----------------------------------|-------------|-------|
| `Tmrt_params.absK`              | `HumanParams(abs_k=0.7)`          | ✅ BUNDLED  | Default 0.7 in bundled params |
| `Tmrt_params.absL`              | `HumanParams(abs_l=0.95)`         | ✅ BUNDLED  | Default 0.95 in bundled params |
| `Tmrt_params.posture`           | `HumanParams(posture="standing")` | ✅ BUNDLED  | "standing" or "sitting" |
| `Posture.Standing.Fside`        | Internal constant                 | ✅ BUNDLED  | 0.22 (from bundled params) |
| `Posture.Standing.Fup`          | Internal constant                 | ✅ BUNDLED  | 0.06 (from bundled params) |
| `Posture.Standing.height`       | Internal constant                 | ✅ BUNDLED  | 1.1m (from bundled params) |
| `Posture.Standing.Fcyl`         | Internal constant                 | ✅ BUNDLED  | 0.28 (from bundled params) |
| `Posture.Sitting.*`             | Internal constant                 | ✅ BUNDLED  | Similar for sitting posture |

### Human Body Parameters (PET/UTCI)

| Old API (params.json)           | New API                           | Status      | Notes |
|---------------------------------|-----------------------------------|-------------|-------|
| `PET_settings.Age`              | `HumanParams(age=35)`             | ✅ BUNDLED  | Default 35 in bundled params |
| `PET_settings.Weight`           | `HumanParams(weight=75)`          | ✅ BUNDLED  | Default 75 kg in bundled params |
| `PET_settings.Height`           | `HumanParams(height=180)`         | ✅ BUNDLED  | Default 180 cm in bundled params |
| `PET_settings.Sex`              | `HumanParams(sex="Male")`         | ✅ BUNDLED  | "Male" or "Female" |
| `PET_settings.Activity`         | `HumanParams(activity=80)`        | ✅ BUNDLED  | Default 80 W in bundled params |
| `PET_settings.clo`              | `HumanParams(clothing=0.9)`       | ✅ BUNDLED  | Default 0.9 clo in bundled params |
| `Wind_Height.magl`              | `weather.wind_speed_height`       | ⏳ TODO     | Currently assumes 10m (planned) |

### Vegetation Parameters

| Old API (params.json)              | New API                           | Status      | Notes |
|------------------------------------|-----------------------------------|-------------|-------|
| `Tree_settings.Transmissivity`     | Bundled default (0.03)            | ✅ BUNDLED  | Leaf-on transmissivity |
| `Tree_settings.Trunk_ratio`        | Bundled default (0.25)            | ✅ BUNDLED  | Trunk height as fraction of total |
| `Tree_settings.First_day_leaf`     | Bundled default (97 = ~Apr 7)     | ✅ BUNDLED  | Day of year for leaf-on |
| `Tree_settings.Last_day_leaf`      | Bundled default (300 = ~Oct 27)   | ✅ BUNDLED  | Day of year for leaf-off |
| Conifer override                   | `conifer=True` parameter          | ✅ DIRECT   | Forces always-leaf-on if True |

### Landcover-Specific Properties (Material Library)

These require **custom params file** with landcover definitions:

| Old API (params.json)              | New API                           | Status      | Notes |
|------------------------------------|-----------------------------------|-------------|-------|
| `Names.Value.*`                    | `load_params("custom.json")`      | ⏳ TODO     | Landcover class names |
| `Code.Value.*`                     | `load_params("custom.json")`      | ⏳ TODO     | Landcover class IDs |
| `Albedo.Effective.Value.*`         | `load_params("custom.json")`      | ⏳ TODO     | Surface albedo per class |
| `Albedo.Material.Value.*`          | `load_params("custom.json")`      | ⏳ TODO     | Wall albedo per material |
| `Emissivity.Value.*`               | `load_params("custom.json")`      | ⏳ TODO     | Surface emissivity per class |
| `Specific_heat.Value.*`            | `load_params("custom.json")`      | ⏳ TODO     | Wall thermal properties |
| `Thermal_conductivity.Value.*`     | `load_params("custom.json")`      | ⏳ TODO     | Wall thermal properties |
| `Density.Value.*`                  | `load_params("custom.json")`      | ⏳ TODO     | Wall thermal properties |
| `Wall_thickness.Value.*`           | `load_params("custom.json")`      | ⏳ TODO     | Wall thermal properties |
| `TmaxLST.Value.*`                  | `load_params("custom.json")`      | ⏳ TODO     | Ground temperature model |
| `Ts_deg.Value.*`                   | `load_params("custom.json")`      | ⏳ TODO     | Ground temperature model |
| `Tstart.Value.*`                   | `load_params("custom.json")`      | ⏳ TODO     | Ground temperature model |

**Note:** Landcover-specific properties are not in bundled defaults because they're highly site-specific. Users who need material variation must provide a custom params file.

---

## Automatic Computations and Preparation

The following values were previously required inputs but are now **computed automatically** or **prepared/cached automatically during `SurfaceData.prepare()`**:

| Parameter                  | Old API                  | New API          | Notes |
|----------------------------|--------------------------|------------------|-------|
| Sun position (azimuth, altitude) | Pre-computed or manual | Auto from datetime + location | Uses `weather.compute_derived()` |
| Max DSM height             | Manual specification     | Auto from DSM    | Computed: `surface.max_height = dsm.max()` |
| Direct/diffuse radiation split | Pre-computed        | Auto from clearness | Reindl model |
| Location (lat/lon)         | Manual or from EPW       | Auto from CRS    | `Location.from_surface(surface)` |
| Shadow matrices            | Pre-computed NPZ files   | Auto-generated   | Prepared/cached during surface preparation |
| Wall heights/aspects       | Pre-computed TIF files   | Auto-generated   | Prepared/cached during surface preparation |
| Sky View Factor            | Pre-computed ZIP files   | Auto-generated   | Prepared via `SurfaceData.prepare()` (or `surface.compute_svf()`) before `calculate*()` |

---

## Usage Examples

### Minimal (uses all bundled defaults)

```python
import solweig

surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    working_dir="cache/",
)

weather = solweig.Weather.from_epw("weather.epw", start="2023-07-01", end="2023-07-01")

results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather,
    output_dir="output/",
)
# Uses bundled defaults:
# - abs_k=0.7, abs_l=0.95, posture="standing"
# - Vegetation transmissivity=0.03, deciduous trees
# - Isotropic sky (use_anisotropic_sky=False)
```

### With direct parameters

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather,
    use_anisotropic_sky=True,  # Enable Perez diffuse model
    conifer=True,              # Evergreen trees (always leaf-on)
    output_dir="output/",
)
```

### With custom human parameters

```python
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather,
    human=solweig.HumanParams(
        abs_k=0.65,      # Lower shortwave absorption
        abs_l=0.97,      # Higher longwave absorption
        posture="sitting",
        weight=70,       # For PET post-processing
        height=1.65,
    ),
    output_dir="output/",
)
```

### With custom landcover parameters

```python
# Load custom material library
params = solweig.load_params("parametersforsolweig.json")

# Requires land_cover grid in SurfaceData (TODO - Phase 3)
surface = solweig.SurfaceData.prepare(
    dsm="dsm.tif",
    land_cover="landcover.tif",  # Grid with class IDs
    working_dir="cache/",
)

results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather,
    params=params,  # Uses custom albedo/emissivity per class
    output_dir="output/",
)
```

### Explicit bundled defaults (for inspection)

```python
# Load bundled defaults to see what's included
params = solweig.load_params()  # Uses pysrc/solweig/data/default_params.json
print(f"Default Tmrt absK: {params.Tmrt_params.Value.absK}")
print(f"Default tree transmissivity: {params.Tree_settings.Value.Transmissivity}")

# Can pass explicitly, but not necessary (loaded automatically if params=None)
results = solweig.calculate_timeseries(..., params=params)
```

---

## Migration Checklist

If you're migrating from the old config-based API, use this checklist:

### ✅ Already Handled (No Action Needed)

- [x] Sun position calculation
- [x] Direct/diffuse radiation split
- [x] Wall generation
- [x] SVF generation
- [x] Shadow matrix generation
- [x] Location extraction from CRS
- [x] Max DSM height
- [x] Default human parameters
- [x] Vegetation seasonal behavior

### 🔧 Requires New API Usage

- [ ] **File paths**: Replace config.ini paths with `SurfaceData.prepare()` arguments
- [ ] **EPW loading**: Use `Weather.from_epw()` instead of manual parsing
- [ ] **Output directory**: Use `calculate_timeseries(output_dir=...)` instead of config
- [ ] **Model flags**: Use direct parameters (`use_anisotropic_sky`, `conifer`) instead of config flags

### ⏳ TODO (Future Work)

- [ ] **Landcover variation**: Custom params file support (planned Phase 3.6)
- [ ] **Wind height**: Currently assumes 10m (planned parameter)
- [ ] **Custom wall/SVF**: Advanced preprocessing control (manual mode available)

---

## API Design Principles

The new API follows these design principles:

1. **Direct parameters for key decisions**
   - `use_anisotropic_sky=True/False` - Major model choice
   - `conifer=True/False` - Vegetation type
   - **NOT** hidden in config object

2. **Bundled defaults for common constants**
   - Human body parameters (absK=0.7, absL=0.95)
   - Vegetation parameters (transmissivity=0.03)
   - Loaded automatically, overridable

3. **Custom params for site-specific values**
   - Landcover material properties (albedo, emissivity per class)
   - Requires explicit JSON file

4. **Automatic for derived values**
   - Sun position from datetime
   - Max height from DSM
   - Location from CRS metadata

5. **Progressive disclosure**
   - Simple case: 3-4 lines of code
   - Advanced: Full control via optional parameters
   - Expert: Direct access to low-level functions

---

## Status Summary

| Category | Total | ✅ Complete | ⏳ TODO | 🗑️ Removed |
|----------|-------|------------|---------|-----------|
| **Model Flags** | 7 | 5 | 1 | 1 |
| **File Paths** | 11 | 9 | 1 | 1 |
| **Preprocessing** | 4 | 4 | 0 | 0 |
| **Human Params** | 12 | 12 | 0 | 0 |
| **Vegetation** | 5 | 5 | 0 | 0 |
| **Landcover** | 11 | 0 | 11 | 0 |
| **Automatic** | 8 | 8 | 0 | 0 |
| **TOTAL** | 58 | 43 | 13 | 2 |

**Overall Progress: 74% Complete (43/58)**

**Remaining work:** Landcover-specific material properties (Phase 3.6 - High Priority)
