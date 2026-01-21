# Mean Radiant Temperature (Tmrt)

The uniform temperature of an imaginary black enclosure that would result in the same radiant heat exchange as the actual non-uniform environment.

**Reference:** Lindberg et al. (2008) Section 2.7, ISO 7726

## Equation

### Absorbed Radiation (Sstr)

Total radiation absorbed by a human body from all directions:

```text
Sstr = absK × (Kside×Fside + (Kdown+Kup)×Fup)
     + absL × (Lside×Fside + (Ldown+Lup)×Fup)
```

Where:
- absK = shortwave absorption coefficient (~0.70 for clothed human)
- absL = longwave absorption coefficient (~0.97)
- Fside = view factor for sides (depends on posture)
- Fup = view factor for top/bottom (depends on posture)

### Mean Radiant Temperature

```text
Tmrt = (Sstr / (absL × σ))^0.25 - 273.15
```

Where σ = Stefan-Boltzmann constant (5.67 × 10⁻⁸ W/m²K⁴).

## Inputs

| Input | Type | Description |
| ----- | ---- | ----------- |
| Kdown | 2D array (W/m²) | Diffuse shortwave from sky |
| Kup | 2D array (W/m²) | Reflected shortwave from ground |
| Kside | 2D arrays (W/m²) | Direct + reflected shortwave (E,S,W,N) |
| Ldown | 2D array (W/m²) | Longwave from sky |
| Lup | 2D array (W/m²) | Longwave from ground |
| Lside | 2D arrays (W/m²) | Longwave from walls (E,S,W,N) |
| absK | float | Shortwave absorption (~0.70) |
| absL | float | Longwave absorption (~0.97) |
| posture | string | "standing" or "sitting" |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| Tmrt | 2D array (°C) | Mean radiant temperature grid |

## Posture Factors

Human body geometry affects how radiation is received:

| Posture | Fup | Fside | Notes |
| ------- | --- | ----- | ----- |
| Standing | 0.06 | 0.22 | Vertical cylinder approximation |
| Sitting | 0.17 | 0.17 | More horizontal surface area |

## Properties

### Fundamental Properties

1. **Tmrt defined for any radiation environment**
   - Always computable if radiation inputs are valid
   - Range typically -20°C to +80°C in urban environments

2. **Tmrt = Ta when no radiation difference**
   - In uniform temperature enclosure with no sun
   - Night with overcast sky approaches this

### Sun/Shade Properties

3. **Sunlit Tmrt > Shaded Tmrt (daytime)**
   - Direct sun adds 10-30°C to Tmrt
   - Largest effect at midday, clear sky

4. **Shadow reduces Tmrt significantly**
   - Moving from sun to shade: ΔTmrt ≈ 10-30°C
   - Most important thermal comfort intervention

### SVF Properties

5. **Higher SVF → higher Tmrt (daytime)**
   - More sky radiation received
   - Open areas warmer than canyons (radiation-wise)

6. **Lower SVF → higher Tmrt (nighttime)**
   - Less longwave loss to cold sky
   - Urban heat island effect

### Surface Temperature Properties

7. **Hot ground increases Tmrt**
   - Lup increases with ground temperature
   - Asphalt vs grass: ΔTmrt ≈ 5-15°C

8. **Hot walls increase Tmrt**
   - Sun-heated walls emit more longwave
   - South-facing walls hottest in afternoon

### Temporal Properties

9. **Tmrt peaks in early afternoon**
   - Maximum direct radiation
   - Ground and walls heated

10. **Tmrt > Ta during day, Tmrt < Ta at night**
    - Daytime: sun adds radiation
    - Nighttime: surfaces cooler than air

## Typical Values

| Condition | Tmrt | Ta | ΔT |
| --------- | ---- | -- | -- |
| Clear day, sun | 55-70°C | 30°C | +25-40°C |
| Clear day, shade | 35-45°C | 30°C | +5-15°C |
| Overcast day | 25-35°C | 25°C | 0-10°C |
| Clear night | 10-20°C | 20°C | -10-0°C |
| Winter sun | 20-35°C | 5°C | +15-30°C |

## Significance

Tmrt is the key variable for outdoor thermal comfort:

- Dominates heat stress in hot climates
- More important than air temperature for comfort
- Directly modifiable through shade provision
- Input to UTCI and PET calculations
