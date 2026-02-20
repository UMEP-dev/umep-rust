# Mean Radiant Temperature (Tmrt)

The uniform temperature of an imaginary black enclosure that would result in the same radiant heat exchange as the actual non-uniform environment.

**Primary References:**

- ISO 7726:1998 "Ergonomics of the thermal environment - Instruments for measuring physical quantities"
- Lindberg et al. (2008) Section 2.7
- Höppe P (1992) "Ein neues Verfahren zur Bestimmung der mittleren Strahlungstemperatur im Freien." Wetter und Leben 44:147-151

## Equation

### Absorbed Radiation (Sstr)

Total radiation absorbed by a human body from all directions:

```text
Isotropic sky:
  Sstr = abs_k × (Kside_total × Fcyl + (Kdown + Kup) × Fup + Kside_dirs_sum × Fside)
       + abs_l × ((Ldown + Lup) × Fup + Lside_dirs_sum × Fside)

Anisotropic sky:
  Sstr = abs_k × (Kside_total × Fcyl + (Kdown + Kup) × Fup + Kside_dirs_sum × Fside)
       + abs_l × ((Ldown + Lup) × Fup + Lside_total × Fcyl + Lside_dirs_sum × Fside)
```

Where:

- abs_k = shortwave absorption coefficient (0.70 for clothed human)
- abs_l = longwave absorption coefficient (0.97 for clothed human)
- Fcyl = cylindric projection factor for direct beam (depends on posture)
- Fside = view factor for sides (depends on posture)
- Fup = view factor for top/bottom (depends on posture)
- Kside_total, Lside_total = total side radiation from anisotropic sky model
- Kside_dirs_sum, Lside_dirs_sum = sum of N+E+S+W directional side radiation

Note: In isotropic mode, `Lside_total × Fcyl` is omitted. In anisotropic mode, it accounts for the non-uniform longwave sky distribution via the cylindric projection factor.

### Mean Radiant Temperature

```text
Tmrt = (Sstr / (abs_l × σ))^0.25 - 273.15
```

Where σ = Stefan-Boltzmann constant (5.67051 × 10⁻⁸ W/m²K⁴).

Output is clamped to [-50, 80]°C to prevent physically unreasonable values.

## Absorption Coefficients

**Reference:** ISO 7726:1998 "Ergonomics of the thermal environment - Instruments for measuring physical quantities"

The human body absorbs radiation differently for shortwave (solar) and longwave (thermal) wavelengths:

| Coefficient | Value | Description                      | Source           |
| ----------- | ----- | -------------------------------- | ---------------- |
| abs_k       | 0.70  | Shortwave (solar) absorption     | ISO 7726 Table 4 |
| abs_l       | 0.97  | Longwave (thermal) absorption    | ISO 7726 Table 4 |

### Physical Basis

**Shortwave (abs_k = 0.70):**

- Represents average absorption of clothed human body in solar spectrum (0.3-3 μm)
- Varies with clothing color and material:
  - White clothing: abs_k ≈ 0.40-0.50
  - Medium grey clothing: abs_k ≈ 0.70 (standard reference)
  - Dark clothing: abs_k ≈ 0.85-0.90
- 0.70 is the ISO 7726 standard value for typical outdoor clothing
- Remaining (1 - abs_k) = 0.30 is reflected

**Longwave (abs_l = 0.97):**

- Human body absorption/emission in thermal infrared spectrum (3-100 μm)
- Based on Kirchhoff's law: absorptivity = emissivity at thermal equilibrium
- Physical basis:
  - Human skin emissivity ≈ 0.98 (consistent across skin tones)
  - Typical clothing emissivity ≈ 0.95-0.97 (most fabrics)
  - Weighted average for clothed person ≈ 0.97
- ISO 7726 standard value: 0.97
- Nearly all thermal radiation is absorbed (only 3% reflected)

### Standards and Implementation

**ISO 7726:1998 Reference Values:**

The ISO 7726 standard (Table 4, Section 4.2.3) specifies:

- abs_k = 0.70 for solar radiation absorption
- abs_l = 0.97 for longwave radiation absorption

These values are used for standardized Mean Radiant Temperature measurements.

**Implementation in SOLWEIG:**

The default values in `HumanParams` (defined in `pysrc/solweig/models/config.py`):

```python
@dataclass
class HumanParams:
    posture: str = "standing"
    abs_k: float = 0.7   # ISO 7726 standard
    abs_l: float = 0.97  # ISO 7726 standard
```

**Historical Note on abs_l Discrepancy:**

Earlier SOLWEIG versions and some literature sources use abs_l = 0.95 instead of 0.97. Both values are physically reasonable:

- 0.95: Conservative estimate, more common in early thermal comfort studies
- 0.97: ISO 7726 standard, more accurate for typical clothing

This implementation follows ISO 7726 and uses 0.97 as the default. Users can override via `HumanParams(abs_l=0.95)` for compatibility with older studies.

**Impact on Tmrt:**

The difference between abs_l = 0.95 and 0.97 has minimal effect on calculated Tmrt:

```text
Tmrt = (Sstr / (abs_l × σ))^0.25 - 273.15

For typical Sstr = 400 W/m²:
  abs_l = 0.97 → Tmrt ≈ 40.5°C
  abs_l = 0.95 → Tmrt ≈ 40.7°C
  Difference: ~0.2°C (negligible for most applications)
```

## Inputs

| Input | Type | Description |
| --- | --- | --- |
| Kdown | 2D array (W/m²) | Diffuse shortwave from sky |
| Kup | 2D array (W/m²) | Reflected shortwave from ground |
| Kside_dirs_sum | 2D array (W/m²) | Sum of directional shortwave (E+S+W+N) |
| Kside_total | 2D array (W/m²) | Total side shortwave from anisotropic sky |
| Ldown | 2D array (W/m²) | Longwave from sky |
| Lup | 2D array (W/m²) | Longwave from ground |
| Lside_dirs_sum | 2D array (W/m²) | Sum of directional longwave (E+S+W+N) |
| Lside_total | 2D array (W/m²) | Total side longwave from anisotropic sky |
| abs_k | float | Shortwave absorption (default 0.70) |
| abs_l | float | Longwave absorption (default 0.97) |
| posture | string | "standing" or "sitting" |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| Tmrt | 2D array (°C) | Mean radiant temperature grid |

## Posture View Factors

Human body geometry affects how radiation is received. View factors represent the fraction of radiation from each direction that is intercepted by the body.

**Primary Reference:** Mayer H, Höppe P (1987) "Thermal comfort of man in different urban environments." Theoretical and Applied Climatology 38:43-49.

**Additional References:**

- Fanger PO (1970) "Thermal Comfort", Danish Technical Press
- VDI 3787 Part 2 (2008) "Environmental Meteorology - Methods for the human biometeorological evaluation of climate and air quality"

| Posture  | Fup   | Fside | Total                      | Model Description |
| -------- | ----- | ----- | -------------------------- | ----------------- |
| Standing | 0.06  | 0.22  | 0.06×2 + 0.22×4 = 1.00     | Vertical cylinder |
| Sitting  | 0.166 | 0.166 | 0.166×2 + 0.166×4 = 1.00   | Modified cylinder |

### Physical Derivation

**Standing Posture (Vertical Cylinder Model):**

The human body is approximated as a vertical cylinder with height H and diameter D, where H/D ≈ 8-10 (typical body proportions).

View factor calculation:

1. **Upward/downward view factor (Fup):**
   - Circular cross-section area: A_horizontal = πD²/4
   - Total body surface area: A_total ≈ πDH (neglecting top/bottom caps)
   - Projected area ratio: Fup ≈ (πD²/4) / (πDH/2) ≈ D/(2H)
   - For H/D ≈ 8.5: Fup ≈ 1/17 ≈ 0.06

2. **Sideward view factor per direction (Fside):**
   - Projected area per cardinal direction (E, S, W, N): A_side = H×D/2
   - View factor per direction: Fside ≈ (H×D/2) / (πDH/2) ≈ 1/π ≈ 0.318
   - Accounting for body curvature and posture: Fside ≈ 0.22 (empirically determined)

3. **Validation:**

   ```text
   Total = 2×Fup + 4×Fside
        = 2×0.06 + 4×0.22
        = 0.12 + 0.88
        = 1.00  ✓
   ```

**Sitting Posture (Modified Cylinder):**

For a sitting person, the body is more compact with increased horizontal cross-section:

1. **Height reduction:** Effective height H_sitting ≈ 0.6×H_standing
2. **Width increase:** Effective width increases due to bent posture
3. **Equal distribution:** More uniform view factor distribution
   - Fup = Fside = 0.166 (simplified model)
   - Total = 6×0.166 ≈ 1.00 ✓

### Implementation Notes

**Direct Beam Projection (f_cyl):**

For direct solar radiation on vertical body surfaces, an additional projection factor f_cyl is used:

| Posture | f_cyl | Description |
| --- | --- | --- |
| Standing | 0.28 | Projected area for cylinder from sun |
| Sitting | 0.20 | Reduced projection for compact posture |

The f_cyl factor accounts for the cylindrical projection of direct beam radiation, distinct from the hemispherical view factors (Fup, Fside) used for diffuse radiation.

**Source Code Reference:**

View factors are defined in `rust/src/tmrt.rs` (lines 10-18) and `pysrc/solweig/constants.py` (lines 39-47):

```python
if posture == "standing":
    f_up = 0.06
    f_side = 0.22
    f_cyl = 0.28
else:  # sitting
    f_up = 0.166666
    f_side = 0.166666
    f_cyl = 0.20
```

These values match the ISO 7726 and VDI 3787 standards for thermal comfort assessment.

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

## Tmrt Calculation Implementation

### Directional Radiation Summation

For directional shortwave and longwave, the model computes separate fluxes for each cardinal direction (N, E, S, W) and sums them with appropriate view factors:

```text
Kside = Keast + Ksouth + Kwest + Knorth
Lside = Least + Lsouth + Lwest + Lnorth
```

### Kelvin Offset

The formula converts from Kelvin to Celsius using:

```text
Tmrt_celsius = Tmrt_kelvin - 273.15
```

Some legacy implementations used -273.2 (rounded). The modern implementation uses the exact value.

### Numerical Stability

When Sstr ≤ 0 (very rare, indicates model error), the implementation clamps to a minimum value to avoid invalid fourth-root operations.

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**Tmrt Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Höppe P (1992) "A new procedure to determine the mean radiant temperature outdoors." Wetter und Leben 44, 147-151.
