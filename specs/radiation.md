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

### Source Code Locations

Radiation logic is spread across multiple Rust files:

- `rust/src/perez.rs` — Perez anisotropic sky model, clearness bins, luminance distribution
- `rust/src/sky.rs` — Isotropic sky radiation, cylindric wedge luminance
- `rust/src/patch_radiation.rs` — Per-patch sky emissivity (Martin & Berdahl band model)
- `rust/src/emissivity_models.rs` — Sky emissivity computation
- `rust/src/pipeline.rs` — Kdown, Kup, Ldown assembly from component outputs

## Shortwave Radiation (K)

Solar radiation, wavelengths ~0.3-3 μm.

### Diffuse Fraction (Preprocessing Step)

Global radiation (G) is split into direct (I) and diffuse (D) using the Reindl et al. (1990) correlation.

**Reference:** Reindl DT, Beckman WA, Duffie JA (1990) "Diffuse fraction correlations." Solar Energy 45(1):1-7.

**Important:** In this codebase, diffuse fraction splitting is performed as a **preprocessing step** (via `clearnessindex_2013b` and `diffusefraction` in the upstream UMEP Python). The Rust pipeline receives pre-split `direct_rad` (rad_i) and `diffuse_rad` (rad_d) as inputs. The Reindl model is documented here for reference but is not computed within the Rust pipeline.

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

**Properties:**

1. Clear sky: D/G ≈ 0.1-0.2 (mostly direct)
2. Overcast: D/G ≈ 0.9-1.0 (mostly diffuse)
3. D/G increases at low sun altitudes

### Anisotropic Diffuse Sky (Perez Model)

For improved accuracy, diffuse radiation can use anisotropic sky luminance distribution.

**Reference:** Perez R, Seals R, Michalsky J (1993) "All-weather model for sky luminance distribution - Preliminary configuration and validation." Solar Energy 50(3):235-245.

The Perez model divides the sky into three components:

1. **Isotropic background** — uniform diffuse
2. **Circumsolar brightening** — enhanced near sun disk
3. **Horizon brightening** — enhanced near horizon

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

| Bin | ε Range | Description | Typical Condition |
| --- | --- | --- | --- |
| 1 | ε < 1.065 | Very overcast | Heavy cloud cover |
| 2 | 1.065-1.23 | Overcast | Thick clouds |
| 3 | 1.23-1.50 | Cloudy | Medium clouds |
| 4 | 1.50-1.95 | Partly cloudy | Scattered clouds |
| 5 | 1.95-2.80 | Partly clear | Few clouds |
| 6 | 2.80-4.50 | Clear | Mostly clear |
| 7 | 4.50-6.20 | Very clear | Exceptionally clear |
| 8 | ε > 6.20 | Extremely clear | Desert/high altitude |

The clearness parameter ε is computed from (in `rust/src/perez.rs`):

```text
ε = ((D + I) / D + 1.041 × zen³) / (1 + 1.041 × zen³)
```

Where zen is the solar zenith angle in **radians**. Note: the UTCI specification and some literature express this with θ_z in degrees using the coefficient 5.535×10⁻⁶. Both forms are equivalent: `1.041 × zen_rad³ = 5.535×10⁻⁶ × zen_deg³`.

#### Bin-0 Special Equations (Robinson Extension)

For clearness bin 0 (most overcast), the c and d coefficients use different equations than bins 1-7:

```text
c = exp((brightness × (c1 + c2 × zen))^c3) - 1
d = -exp(brightness × (d1 + d2 × zen)) + d3 + brightness × d4
```

This Robinson extension handles the overcast limiting case where the standard linear interpolation breaks down.

#### Anisotropic Sky Emissivity (Martin & Berdahl Band Model)

In anisotropic mode, per-patch sky emissivity is computed using the Martin & Berdahl (1984) band emissivity model (`rust/src/patch_radiation.rs`). This accounts for the angular dependence of atmospheric emission — the sky emits more longwave radiation near the horizon than at the zenith due to the longer atmospheric path length.

#### Implementation in SOLWEIG

The Rust implementation (`rust/src/perez.rs`):

1. Computes sky clearness bin from solar geometry and radiation
2. Retrieves coefficients (a, b, c, d, e) for the bin
3. Evaluates luminance L for each sky patch (altitude, azimuth)
4. Normalizes to ensure integration equals diffuse radiation
5. Returns patch luminance array for anisotropic radiation calculation

### Kdown (Downwelling Shortwave)

Shortwave radiation reaching the ground from above:

```text
Kdown = rad_i × shadow × sin(altitude)
      + drad
      + albedo_wall × (1 - svfbuveg) × (rad_g × (1 - f_sh) + rad_d × f_sh)
```

Where:

- `rad_i × shadow × sin(altitude)` = direct beam, blocked by shadows
- `drad` = diffuse sky component:
  - Isotropic: `drad = rad_d × svfbuveg`
  - Anisotropic: `drad = ani_lum × rad_d` (patch-weighted luminance)
- `albedo_wall × (1 - svfbuveg) × (...)` = wall-reflected shortwave
- `f_sh` = fraction of shadow (from SVF shadow matrices)

**Properties:**

1. Kdown proportional to SVF (for diffuse component)
2. Direct component blocked by shadows
3. Range: 0 to ~1000 W/m²

### Kup (Reflected from Ground)

Shortwave reflected upward from ground surfaces:

```text
ct = rad_d × svfbuveg + albedo_b × (1 - svfbuveg) × (rad_g × (1 - f_sh) + rad_d × f_sh)
kup = gvfalb × rad_i × sin(altitude) + ct × gvfalbnosh
```

Where:

- `gvfalb` = GVF albedo-weighted view factor (shadow-dependent, from gvf.md)
- `gvfalbnosh` = GVF albedo view factor (no shadow, geometric only)
- `ct` = combined diffuse + wall-reflected shortwave term

Note: `gvfalb` already contains albedo weighting — no additional multiplication by albedo is needed.

**Properties:**

1. Higher albedo → more reflection
2. Shaded areas reflect less (no direct component)
3. Range: 0 to ~200 W/m²

### Kside (Direct + Reflected to Walls)

Shortwave reaching vertical surfaces, computed per cardinal direction (N, E, S, W):

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

### Ldown (Sky Longwave)

Thermal emission from atmosphere, computed in `pipeline.rs::compute_ldown`:

```text
Ldown = (svf + svf_veg - 1) × esky × σ × Ta_K⁴
      + (2 - svf_veg - svf_aveg) × ewall × σ × Ta_K⁴
      + (svf_aveg - svf) × ewall × σ × (Ta + Tg_wall + 273.15)⁴
      + (2 - svf - svf_veg) × (1 - ewall) × esky × σ × Ta_K⁴
```

The four terms represent:

1. Sky emission through vegetation-free sky opening
2. Ambient-temperature wall emission (walls between vegetation and building canopy)
3. Heated wall emission (walls warmed by sun, between veg-blocking and building canopy)
4. Sky emission reflected off walls (non-absorbed sky radiation bounced from walls)

**Cloud correction:** When clearness index CI < 0.95:

```text
Ldown_effective = Ldown_clear × (1 - c_cloud) + Ldown_cloudy × c_cloud
c_cloud = 1 - CI
```

Where `Ldown_cloudy` uses the same formula but with esky=1.0 for the sky term.

**Properties:**

1. Increases with humidity and cloud cover
2. Clear sky: Ldown ≈ 250-350 W/m²
3. Overcast: Ldown ≈ 350-450 W/m²

### Lup (Ground Longwave)

Thermal emission from ground surface, provided by the GVF module after thermal delay smoothing:

```text
Lup = TsWaveDelay(gvf_lup, ...)
```

Where `gvf_lup` is the ground-emitted longwave from the GVF integration (see gvf.md). The TsWaveDelay function applies exponential smoothing across timesteps to simulate thermal inertia (see ground_temperature.md).

**Properties:**

1. Increases with ground temperature
2. Hot asphalt can emit >500 W/m²
3. ε_ground typically 0.90-0.98

### Lside (Wall Longwave)

Thermal emission from building walls, per cardinal direction:

**Properties:**

1. Sun-heated walls emit more
2. Directional: Least, Lsouth, Lwest, Lnorth
3. Important in urban canyons

## Properties Summary

### Shadow Effects

1. **Shadows block direct shortwave only**
   - Diffuse and longwave unaffected by shadows
   - Shaded areas still receive Kdown (diffuse), Ldown, Lup

2. **Shadow reduces total K significantly**
   - Sun to shade: ΔK ≈ 200-800 W/m² (depending on direct beam)

### SVF Effects

1. **Low SVF reduces sky radiation**
   - Both Kdown and Ldown reduced
   - But Lside from walls increases

2. **Urban canyon radiation balance**
   - Lower Kdown, Ldown (less sky)
   - Higher Kup, Lside (more surfaces)

### Temperature Effects

1. **Hot surfaces increase longwave**
   - Lup increases with ground temperature
   - Can dominate radiation budget on hot days

### Typical Values

| Component | Clear Day Noon | Shaded | Night |
| --- | --- | --- | --- |
| Kdown | 100-200 | 100-200 | 0 |
| Kup | 50-150 | 30-100 | 0 |
| Kside (sunlit) | 200-600 | 0 | 0 |
| I (direct) | 600-900 | 0 | 0 |
| Ldown | 300-400 | 300-400 | 250-350 |
| Lup | 400-600 | 350-500 | 300-450 |
| Lside | 350-550 | 350-500 | 300-450 |

All values in W/m².

## Implementation Notes

### Clearness Index

**Reference:** Crawford TM, Duchon CE (1999) "An improved parameterization for estimating effective atmospheric emissivity for use in calculating daytime downwelling longwave radiation." Journal of Applied Meteorology 38:474-480.

The clearness index CI is pre-computed in the upstream UMEP Python preprocessing (`clearnessindex_2013b`) and supplied to the Rust pipeline as an input. It is not computed within the Rust pipeline.

### Isotropic vs Anisotropic Mode

The model supports two diffuse radiation modes:

1. **Isotropic** (faster): Uniform diffuse sky. `drad = rad_d × svfbuveg`
2. **Anisotropic** (Perez, default): Non-uniform sky luminance, requires SVF shadow matrices from the SVF precomputation. Uses per-patch luminance weighting.

Use anisotropic mode when:

- High accuracy required near buildings
- Studying directional radiation effects
- SVF < 0.7 (urban canyons)

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**Radiation Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.

**Diffuse Fraction:**

- Reindl DT, Beckman WA, Duffie JA (1990) "Diffuse fraction correlations." Solar Energy 45(1), 1-7.

**Anisotropic Sky:**

- Perez R, Seals R, Michalsky J (1993) "All-weather model for sky luminance distribution - Preliminary configuration and validation." Solar Energy 50(3), 235-245.

**Sky Emissivity:**

- Jonsson P, Eliasson I, Holmer B, Grimmond CSB (2006) "Longwave incoming radiation in the Tropics: Results from field work in three African cities." Theoretical and Applied Climatology 85, 185-201.

**Anisotropic Emissivity:**

- Martin M, Berdahl P (1984) "Characteristics of infrared sky radiation in the United States." Solar Energy 33(3-4), 321-336.
