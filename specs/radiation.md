# Radiation

Shortwave (solar) and longwave (thermal) radiation calculations.

**Reference:** Lindberg et al. (2008) Sections 2.4-2.6

## Overview

Radiation at any point comes from six directions (up, down, and four cardinal sides), split into shortwave (K) and longwave (L) components:

```text
Total radiation = Shortwave (K) + Longwave (L)
                = (Kdown + Kup + Kside) + (Ldown + Lup + Lside)
```

## Shortwave Radiation (K)

Solar radiation, wavelengths ~0.3-3 μm.

### Diffuse Fraction

Global radiation (G) is split into direct (I) and diffuse (D):

```text
G = I + D
D/G = f(clearness_index, altitude, temperature, humidity)
```

**Properties:**

1. Clear sky: D/G ≈ 0.1-0.2 (mostly direct)
2. Overcast: D/G ≈ 0.9-1.0 (mostly diffuse)
3. D/G increases at low sun altitudes

### Kdown (Diffuse from Sky)

Diffuse shortwave from the sky hemisphere:

```text
Kdown = D × SVF
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

### Ldown (Sky Longwave)

Thermal emission from atmosphere:

```text
Ldown = ε_sky × σ × T_sky^4 × SVF
```

Where:
- ε_sky = sky emissivity (function of humidity, clouds)
- σ = Stefan-Boltzmann constant (5.67 × 10⁻⁸ W/m²K⁴)
- T_sky = effective sky temperature

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
