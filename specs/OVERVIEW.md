# SOLWEIG Algorithm Overview

SOLWEIG (Solar and Longwave Environmental Irradiance Geometry) calculates mean radiant temperature (Tmrt) and thermal comfort indices in complex urban environments.

**Primary References:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)
- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation." Theoretical and Applied Climatology 105, 311-323.
- Lindberg F, Grimmond CSB, Martilli A (2015) "Sunlit fractions on urban facets - Impact of spatial resolution and approach." Urban Climate 12, 65-84.

## Specification Files

| File                                           | Description                                     |
| ---------------------------------------------- | ----------------------------------------------- |
| [OVERVIEW.md](OVERVIEW.md)                     | This file - pipeline and relationships          |
| [shadows.md](shadows.md)                       | Shadow casting algorithm                        |
| [svf.md](svf.md)                               | Sky View Factor calculation                     |
| [gvf.md](gvf.md)                               | Ground View Factor calculation                  |
| [radiation.md](radiation.md)                   | Shortwave and longwave radiation                |
| [ground_temperature.md](ground_temperature.md) | Surface temperature and thermal delay model     |
| [tmrt.md](tmrt.md)                             | Mean Radiant Temperature                        |
| [utci.md](utci.md)                             | Universal Thermal Climate Index                 |
| [pet.md](pet.md)                               | Physiological Equivalent Temperature            |
| [technical.md](technical.md)                   | Implementation details (tiling, precision, GPU) |
| [runtime-contract.md](runtime-contract.md)     | Runtime API preconditions and output semantics  |

`runtime-contract.md` is the normative source for API/runtime behavior
(SVF/anisotropic preconditions, shadow convention, and return semantics).

## Pipeline

SVF is a **precomputed static input** (depends only on DSM geometry). It is computed
once before the timestep loop and passed into the pipeline as a read-only input.

Each timestep executes the following stages in order:

```
 PRECOMPUTED (static)                  PER-TIMESTEP PIPELINE
┌─────────────────┐    ┌──────────────────────────────────────────────────────┐
│      SVF        │    │                                                      │
│   (svf.md)      │───▶│  1. SHADOWS          shadow masks (bldg, veg, wall) │
│                 │    │     (shadows.md)                                      │
│  sky view       │    │          │                                            │
│  factors        │    │          ▼                                            │
└─────────────────┘    │  2. GROUND TEMP       Tg per land-cover type         │
                       │     (ground_temperature.md)                           │
 INPUT DATA            │          │                                            │
┌─────────────────┐    │          ▼                                            │
│ Geometry        │    │  3. GVF               ground-emitted Lup, albedo,    │
│  - DSM          │    │     (gvf.md)          wall+ground view integration   │
│  - DEM          │───▶│          │                                            │
│  - CDSM (veg)   │    │          ▼                                            │
│  - TDSM (trunk) │    │  4. THERMAL DELAY     smoothed Lup via exponential   │
│  - Land cover   │    │     (ground_temperature.md)  decay (TsWaveDelay)     │
│  - Albedo grid  │    │          │                                            │
│  - Emissivity   │    │          ▼                                            │
│                 │    │  5. RADIATION          Kdown, Kup, Kside,            │
│ Weather         │    │     (radiation.md)     Ldown, Lup, Lside             │
│  - Air temp (Ta)│    │          │                                            │
│  - Humidity (RH)│    │          ▼                                            │
│  - Wind speed   │    │  6. Tmrt              mean radiant temperature       │
│  - Direct rad   │    │     (tmrt.md)                                        │
│  - Diffuse rad  │    │                                                      │
│                 │    └──────────────────────────────────────────────────────┘
│ Time            │
│  - Date/time    │    POST-PROCESSING (Python, not in Rust pipeline)
│  - Location     │    ┌─────────────────┐  ┌─────────────────┐
│    (lat/lon)    │    │      UTCI       │  │      PET        │
└─────────────────┘    │   (utci.md)     │  │   (pet.md)      │
                       └─────────────────┘  └─────────────────┘
```

Note: The Rust pipeline (`pipeline.rs`) fuses stages 1-6 into a single FFI call per
timestep. UTCI and PET are computed separately via `postprocess.py` wrappers around
their respective Rust implementations.

## Module Dependencies

| Module | Depends On | Produces |
| --- | --- | --- |
| **SVF** | DSM, CDSM, TDSM | Sky view factors + directional SVFs (static) |
| **Shadows** | DSM, CDSM, TDSM, walls, wall aspect, sun position | Shadow masks: bldg, veg, wall sun/shade (dynamic) |
| **Ground Temp** | Land cover params, sun altitude, clearness index, weather | Surface temperature deviation Tg per land cover |
| **GVF** | Shadows, walls, buildings, ground temp, weather (Ta), albedo, emissivity | Lup (W/m²), albedo-weighted view, directional |
| **Thermal Delay** | GVF Lup, previous Tgmap1, firstdaytime flag | Smoothed Lup via exponential decay |
| **Radiation** | Shadows, SVF, GVF, thermal delay (Lup), weather, Perez coefficients | K and L fluxes (down, up, side × 4 directions) |
| **Tmrt** | All radiation fluxes, posture view factors | Mean radiant temperature grid |
| **UTCI** | Tmrt, Ta, RH, wind | Thermal comfort index (°C equivalent) |
| **PET** | Tmrt, Ta, RH, wind, weight, age, height, activity (W), clothing, sex | Thermal comfort index (°C equivalent) |

## Static vs Dynamic Calculations

**Calculated Once (static geometry):**

- SVF - depends only on DSM/CDSM/TDSM geometry

**Calculated Per Timestep:**

- Shadows - sun position changes
- Ground temperature - depends on sun altitude, clearness index
- GVF - depends on shadows, ground temperature, weather (Ta). Geometry (ray distances) may be cached, but the thermal integration (Lup, albedo weighting) runs every timestep.
- Thermal delay - exponential smoothing of Lup across timesteps
- Radiation - depends on shadows, SVF, GVF, weather
- Tmrt - depends on all radiation fluxes
- UTCI/PET - all inputs change

## Key Physical Principles

### 1. Shadow Casting

Shadows are cast opposite to the sun direction. Shadow length depends on obstacle height and sun altitude: `L = h / tan(α)`.

### 2. Sky View Factor

SVF represents the fraction of sky visible from a point. In open terrain SVF=1, in deep canyons SVF<0.5. Affects how much sky radiation reaches the surface.

### 3. Radiation Balance

Total radiation at a point combines:

- **Direct shortwave (I)** - blocked by shadows
- **Diffuse shortwave (D)** - reduced by low SVF
- **Reflected shortwave** - from ground and walls
- **Longwave from sky** - depends on SVF and cloud cover
- **Longwave from ground** - depends on ground temperature (via GVF Lup and thermal delay)
- **Longwave from walls** - depends on wall temperature and view factor

### 4. Mean Radiant Temperature

Tmrt integrates radiation from all 6 directions, weighted by human body geometry:

```text
Tmrt = (Sstr / (abs_l × σ))^0.25 - 273.15
```

Where Sstr = absorbed shortwave and longwave radiation from all directions, weighted by posture-dependent view factors (Fup, Fside, Fcyl). `abs_l` is the longwave absorption coefficient (default 0.97).

### 5. Thermal Comfort

UTCI and PET translate the physical environment (Tmrt, Ta, wind, humidity) into equivalent temperatures that represent physiological response. These are computed as a **post-processing** step outside the main Rust pipeline, via `postprocess.py` wrappers.

## Coordinate Conventions

- **DSM arrays**: Row 0 = North, increasing rows = South
- **Azimuth**: 0° = North, 90° = East, 180° = South, 270° = West
- **Altitude**: 0° = horizon, 90° = zenith (directly overhead)

## Units

| Quantity         | Unit                 |
| ---------------- | -------------------- |
| Elevation/height | meters (m)           |
| Temperature      | degrees Celsius (°C) |
| Radiation        | W/m²                 |
| Wind speed       | m/s                  |
| Humidity         | % (relative)         |
| SVF              | dimensionless (0-1)  |
| GVF Lup          | W/m²                 |
| Pixel size       | meters               |
