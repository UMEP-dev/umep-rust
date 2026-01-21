# SOLWEIG Algorithm Overview

SOLWEIG (Solar and Longwave Environmental Irradiance Geometry) calculates mean radiant temperature (Tmrt) and thermal comfort indices in complex urban environments.

**Primary References:**

- Lindberg et al. (2008) - SOLWEIG 1.0 core model
- Lindberg & Grimmond (2011) - Vegetation and building morphology
- Lindberg et al. (2015) - Sunlit fractions methodology

## Specification Files

| File | Description |
| ---- | ----------- |
| [OVERVIEW.md](OVERVIEW.md) | This file - pipeline and relationships |
| [shadows.md](shadows.md) | Shadow casting algorithm |
| [svf.md](svf.md) | Sky View Factor calculation |
| [gvf.md](gvf.md) | Ground View Factor calculation |
| [radiation.md](radiation.md) | Shortwave and longwave radiation |
| [tmrt.md](tmrt.md) | Mean Radiant Temperature |
| [utci.md](utci.md) | Universal Thermal Climate Index |
| [pet.md](pet.md) | Physiological Equivalent Temperature |
| [technical.md](technical.md) | Implementation details (tiling, precision, GPU) |

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Geometry        │  Weather           │  Time                           │
│  - DSM           │  - Air temp (Ta)   │  - Date/time                    │
│  - CDSM (veg)    │  - Humidity (RH)   │  - Location (lat/lon)           │
│  - Buildings     │  - Wind speed      │                                 │
│  - Walls         │  - Global rad (G)  │                                 │
└────────┬────────────────────┬─────────────────────┬────────────────────┘
         │                    │                     │
         ▼                    │                     │
┌─────────────────┐           │                     │
│    SHADOWS      │◄──────────┼─────────────────────┘
│  (shadows.md)   │           │         Sun position
└────────┬────────┘           │
         │ shadow mask        │
         ▼                    │
┌─────────────────┐           │
│      SVF        │           │
│   (svf.md)      │           │
└────────┬────────┘           │
         │ sky view factors   │
         ▼                    │
┌─────────────────┐           │
│      GVF        │           │
│   (gvf.md)      │           │
└────────┬────────┘           │
         │ ground view factors│
         ▼                    ▼
┌─────────────────────────────────────────┐
│             RADIATION                    │
│          (radiation.md)                  │
│  ┌─────────────┐    ┌─────────────┐     │
│  │ Shortwave K │    │ Longwave L  │     │
│  │ Kdown, Kup  │    │ Ldown, Lup  │     │
│  │ Kside       │    │ Lside       │     │
│  └─────────────┘    └─────────────┘     │
└────────────────────┬────────────────────┘
                     │ all radiation fluxes
                     ▼
          ┌─────────────────┐
          │      Tmrt       │
          │   (tmrt.md)     │
          └────────┬────────┘
                   │ mean radiant temperature
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│      UTCI       │  │      PET        │
│   (utci.md)     │  │   (pet.md)      │
└─────────────────┘  └─────────────────┘
     thermal comfort indices
```

## Module Dependencies

| Module | Depends On | Produces |
|--------|------------|----------|
| **Shadows** | DSM, sun position | Shadow mask (per timestep) |
| **SVF** | DSM, CDSM | Sky view factors (static) |
| **GVF** | SVF, walls, albedo | Ground view factors |
| **Radiation** | Shadows, SVF, GVF, weather | K and L fluxes |
| **Tmrt** | All radiation fluxes | Mean radiant temperature |
| **UTCI** | Tmrt, Ta, RH, wind | Thermal comfort index |
| **PET** | Tmrt, Ta, RH, wind, human params | Thermal comfort index |

## Static vs Dynamic Calculations

**Calculated Once (static geometry):**
- SVF - depends only on DSM geometry
- GVF - depends on SVF and surface properties

**Calculated Per Timestep:**
- Shadows - sun position changes
- Radiation - sun position + weather changes
- Tmrt - radiation changes
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
- **Longwave from ground** - depends on ground temperature
- **Longwave from walls** - depends on wall temperature and view factor

### 4. Mean Radiant Temperature
Tmrt integrates radiation from all directions, weighted by human body geometry:
```
Tmrt = (Sstr / (ε × σ))^0.25 - 273.15
```
Where Sstr = absorbed radiation from all 6 directions.

### 5. Thermal Comfort
UTCI and PET translate the physical environment (Tmrt, Ta, wind, humidity) into equivalent temperatures that represent physiological response.

## Coordinate Conventions

- **DSM arrays**: Row 0 = North, increasing rows = South
- **Azimuth**: 0° = North, 90° = East, 180° = South, 270° = West
- **Altitude**: 0° = horizon, 90° = zenith (directly overhead)

## Units

| Quantity | Unit |
|----------|------|
| Elevation/height | meters (m) |
| Temperature | degrees Celsius (°C) |
| Radiation | W/m² |
| Wind speed | m/s |
| Humidity | % (relative) |
| SVF/GVF | dimensionless (0-1) |
| Pixel size | meters |
