# Radiation

Shortwave (solar) and longwave (thermal) radiation calculations.

**Primary References:**

- Lindberg et al. (2008) Sections 2.4-2.6
- Jonsson et al. (2006) - Sky emissivity formulation
- Reindl et al. (1990) - Diffuse fraction correlations
- Perez et al. (1993) - Anisotropic sky luminance distribution

## Overview

Radiation at any point comes from six directions (up, down, and four cardinal sides), split into shortwave (K) and longwave (L) components:

```text
Total radiation = Shortwave (K) + Longwave (L)
                = (Kdown + Kup + Kside) + (Ldown + Lup + Lside)
```

## Shortwave Radiation (K)

Solar radiation, wavelengths ~0.3-3 μm.

### Diffuse Fraction (Reindl Model)

Global radiation (G) is split into direct (I) and diffuse (D) using the Reindl et al. (1990) correlation.

**Reference:** Reindl DT, Beckman WA, Duffie JA (1990) "Diffuse fraction correlations." Solar Energy 45(1):1-7.

The model uses piecewise correlations based on clearness index (Kt):

```text
G = I + D
Kt = G / I0_et    (clearness index = ratio to extraterrestrial)

If Kt ≤ 0.3 (overcast):
    D/G = 1 - 0.232×Kt + 0.0239×sin(α) - 0.000682×Ta + 0.0195×RH

If 0.3 < Kt < 0.78 (partly cloudy):
    D/G = 1.329 - 1.716×Kt + 0.267×sin(α) - 0.00357×Ta + 0.106×RH

If Kt ≥ 0.78 (clear):
    D/G = 0.426×Kt - 0.256×sin(α) + 0.00349×Ta + 0.0734×RH
```

Where:
- α = solar altitude angle (radians)
- Ta = air temperature (°C)
- RH = relative humidity (fraction, 0-1)

When Ta and RH are unavailable, simplified correlations using only Kt are used.

**Properties:**

1. Clear sky: D/G ≈ 0.1-0.2 (mostly direct)
2. Overcast: D/G ≈ 0.9-1.0 (mostly diffuse)
3. D/G increases at low sun altitudes

### Anisotropic Diffuse Sky (Perez Model)

For improved accuracy, diffuse radiation can use anisotropic sky luminance distribution.

**Reference:** Perez R, Seals R, Michalsky J (1993) "All-weather model for sky luminance distribution - Preliminary configuration and validation." Solar Energy 50(3):235-245.

The Perez model divides the sky into three components:

1. **Isotropic background** - uniform diffuse
2. **Circumsolar brightening** - enhanced near sun disk
3. **Horizon brightening** - enhanced near horizon

#### Sky Luminance Distribution

The relative luminance L at any sky element is given by:

```text
L(θ,γ) = (1 + a×exp(b/cos(θ))) × (1 + c×exp(d×γ) + e×cos²(γ))
```

Where:

- θ = zenith angle of sky element (radians)
- γ = angular distance from sun (radians)
- a, b, c, d, e = coefficients from 8 sky clearness bins

#### Sky Clearness Categories

Sky clearness parameter ε determines coefficient bins:

| Bin | ε Range    | Description      | Typical Condition    |
| --- | ---------- | ---------------- | -------------------- |
| 1   | ε < 1.065  | Very overcast    | Heavy cloud cover    |
| 2   | 1.065-1.23 | Overcast         | Thick clouds         |
| 3   | 1.23-1.50  | Cloudy           | Medium clouds        |
| 4   | 1.50-1.95  | Partly cloudy    | Scattered clouds     |
| 5   | 1.95-2.80  | Partly clear     | Few clouds           |
| 6   | 2.80-4.50  | Clear            | Mostly clear         |
| 7   | 4.50-6.20  | Very clear       | Exceptionally clear  |
| 8   | ε > 6.20   | Extremely clear  | Desert/high altitude |

The clearness parameter ε is computed from:

```text
ε = (D + I)/D + 5.535×10⁻⁶×θz³ / (1 + 5.535×10⁻⁶×θz³)
```

Where θz is the solar zenith angle in degrees.

#### Implementation in SOLWEIG

The Rust implementation (`Perez_v3` in `rust/src/sky.rs` and Python wrapper in `algorithms/Perez_v3.py`):

1. Computes sky clearness bin from solar geometry and radiation
2. Retrieves coefficients (a, b, c, d, e) for the bin
3. Evaluates luminance L for each sky patch (altitude, azimuth)
4. Normalizes to ensure integration equals diffuse radiation
5. Returns patch luminance array for anisotropic radiation calculation

The patch luminance is then used to weight diffuse radiation:

```text
drad = Σ (D × L_patch × visibility_patch × steradian_patch)
```

This provides spatially-varying diffuse radiation accounting for sky luminance distribution.

### Kdown (Diffuse from Sky)

Diffuse shortwave from the sky hemisphere:

```text
Isotropic:   Kdown = D × SVF
Anisotropic: Kdown = Σ(D × L_patch × SVF_patch)
```

**Properties:**

1. Kdown proportional to SVF
2. Higher SVF → more diffuse radiation
3. Range: 0 to ~500 W/m² (typical clear sky diffuse)

### Kup (Reflected from Ground)

Shortwave reflected upward from ground:

```text
Kup = (I × shadow + D) × albedo × GVF
```

**Properties:**

1. Higher albedo → more reflection
2. Shaded areas reflect less (no direct component)
3. Range: 0 to ~200 W/m² (depends on albedo)

### Kside (Direct + Reflected to Walls)

Shortwave reaching vertical surfaces:

```text
Kside = I × cos(incidence_angle) × shadow_factor + reflected
```

**Properties:**

1. Depends on wall orientation relative to sun
2. South-facing walls receive more in Northern Hemisphere
3. Directional: Keast, Ksouth, Kwest, Knorth

## Longwave Radiation (L)

Thermal radiation, wavelengths ~3-100 μm.

### Sky Emissivity

**Reference:** Jonsson P, Eliasson I, Holmer B, Grimmond CSB (2006) "Longwave incoming radiation in the Tropics: Results from field work in three African cities." Theoretical and Applied Climatology 85:185-201.

Sky emissivity is computed from air temperature and humidity:

```text
ea = 6.107 × 10^((7.5 × Ta) / (237.3 + Ta)) × (RH / 100)
msteg = 46.5 × (ea / Ta_K)
ε_sky = 1 - (1 + msteg) × exp(-√(1.2 + 3.0 × msteg))
```

Where:
- ea = water vapor pressure (hPa)
- Ta = air temperature (°C)
- Ta_K = air temperature (K)
- RH = relative humidity (%)

**Typical values:**
- Clear dry sky: ε_sky ≈ 0.60-0.75
- Clear humid sky: ε_sky ≈ 0.75-0.85
- Cloudy sky: ε_sky → 1.0

**Cloud correction:**
When clearness index CI < 0.95 (non-clear conditions):

```text
ε_sky_effective = CI × ε_sky + (1 - CI) × 1.0
```

### Ldown (Sky Longwave)

Thermal emission from atmosphere:

```text
Ldown = ε_sky × σ × T_air^4 × SVF
      + wall_contribution
      + vegetation_contribution
```

Where:
- ε_sky = sky emissivity (computed above)
- σ = Stefan-Boltzmann constant (5.67 × 10⁻⁸ W/m²K⁴)
- T_air = air temperature (K)

**Properties:**

1. Increases with humidity and cloud cover
2. Clear sky: Ldown ≈ 250-350 W/m²
3. Overcast: Ldown ≈ 350-450 W/m²

### Lup (Ground Longwave)

Thermal emission from ground surface:

```text
Lup = ε_ground × σ × T_ground^4 × GVF
```

**Properties:**

1. Increases with ground temperature
2. Hot asphalt can emit >500 W/m²
3. ε_ground typically 0.90-0.98

### Lside (Wall Longwave)

Thermal emission from building walls:

```text
Lside = ε_wall × σ × T_wall^4 × wall_view_factor
```

**Properties:**

1. Sun-heated walls emit more
2. Directional: Least, Lsouth, Lwest, Lnorth
3. Important in urban canyons

## Properties Summary

### Conservation

1. **Energy conservation**: Total radiation balanced
2. **Reciprocity**: View factors are symmetric

### Shadow Effects

3. **Shadows block direct shortwave only**
   - Diffuse and longwave unaffected by shadows
   - Shaded areas still receive Kdown, Ldown, Lup

4. **Shadow reduces total K significantly**
   - Sun to shade: ΔK ≈ 200-800 W/m² (depending on direct beam)

### SVF Effects

5. **Low SVF reduces sky radiation**
   - Both Kdown and Ldown reduced
   - But Lside from walls increases

6. **Urban canyon radiation balance**
   - Lower Kdown, Ldown (less sky)
   - Higher Kup, Lside (more surfaces)

### Temperature Effects

7. **Hot surfaces increase longwave**
   - Lup increases with ground temperature
   - Can dominate radiation budget on hot days

### Typical Values

| Component | Clear Day Noon | Shaded | Night |
| --------- | -------------- | ------ | ----- |
| Kdown | 100-200 | 100-200 | 0 |
| Kup | 50-150 | 30-100 | 0 |
| Kside (sunlit) | 200-600 | 0 | 0 |
| I (direct) | 600-900 | 0 | 0 |
| Ldown | 300-400 | 300-400 | 250-350 |
| Lup | 400-600 | 350-500 | 300-450 |
| Lside | 350-550 | 350-500 | 300-450 |

All values in W/m².

## Implementation Notes

### Clearness Index Calculation

**Reference:** Crawford TM, Duchon CE (1999) "An improved parameterization for estimating effective atmospheric emissivity for use in calculating daytime downwelling longwave radiation." Journal of Applied Meteorology 38:474-480.

The clearness index CI is computed from global radiation compared to theoretical clear-sky radiation:

```text
I0 = Itoa × cos(zen) × Trpg × Tw × D × Tar
CI = G / I0
```

Where transmission coefficients account for:
- Trpg = Rayleigh scattering and permanent gases
- Tw = water vapor absorption
- Tar = aerosol attenuation
- D = sun-earth distance correction

### Isotropic vs Anisotropic Mode

The model supports two diffuse radiation modes:

1. **Isotropic** (default): Uniform diffuse sky, faster computation
2. **Anisotropic** (Perez): Non-uniform sky luminance, requires shadow matrices

Use anisotropic mode when:
- High accuracy required near buildings
- Studying directional radiation effects
- SVF < 0.7 (urban canyons)
