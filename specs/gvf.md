# Ground View Factor (GVF)

Fraction of the hemisphere occupied by ground and wall surfaces (as opposed to sky). Determines how much reflected shortwave and emitted longwave radiation from surfaces reaches a point.

**Reference:** Lindberg et al. (2008) Section 2.3

## Equation

GVF is complementary to SVF for an unobstructed point:

```text
GVF = 1 - SVF  (simplified, for flat ground)
```

In practice, GVF accounts for the actual ground and wall surfaces visible:

```text
GVF = Σ (surface_area × view_factor × surface_property)
```

Where surface_property can be albedo (for reflected shortwave) or emissivity (for longwave).

## Wall Integration Method

**Reference:** Lindberg et al. (2008) Section 2.3, Holmer et al. (2015) "SOLWEIG-POI: a new model for estimating Tmrt at points of interest"

When walls are present, GVF is computed using geometric integration of visible surfaces from a person's height above ground. The method considers:

### Full GVF Calculation (with walls)

The implementation in `gvf.py` calls the Rust `gvf_calc` function which:

1. **Person height parameters**: Uses human height to determine view geometry

   - `first = round(height)` - primary height parameter
   - `second = round(height × 20)` - finer height discretization

2. **Wall visibility**: For each pixel, integrates visible wall surfaces in all directions

   - Wall heights (`wall_ht`) define vertical obstruction
   - Wall aspects (`wall_asp`) define cardinal orientation
   - Shadow fraction adjusts wall temperature contribution

3. **Directional components**: Splits GVF into cardinal directions (N, E, S, W)

   - Ground contribution: Based on distance and elevation angle
   - Wall contribution: Based on wall height, orientation, and temperature

4. **Temperature-weighted emission**: Longwave GVF includes thermal emission

   ```text
   Lup = ε_surface × σ × T_surface^4 × GVF
   ```

   Where:

   - Sunlit walls: T_wall = T_air + Tg_wall
   - Shaded walls: T_wall = T_air
   - Ground: T_ground = T_air + Tg (shadow-dependent)

5. **Albedo weighting**: Shortwave GVF weighted by surface albedo

   ```text
   GVF_alb = albedo × GVF
   ```

### Simplified GVF (no walls)

When wall data is unavailable, uses simplified calculation:

```text
GVF_simple = 1 - SVF
Lup = ε_ground × σ × (T_air + Tg × shadow)^4
GVF_alb = albedo_ground × GVF_simple
```

This assumes only ground surfaces contribute (no walls).

## Inputs

| Input | Type | Description |
| ----- | ---- | ----------- |
| SVF arrays | 2D arrays (0-1) | Sky view factors (overall + directional) |
| walls | 2D array (m) | Wall height grid |
| albedo | float or 2D array | Ground surface albedo (0-1) |
| emissivity | float or 2D array | Ground surface emissivity (~0.95) |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| gvf_lup | 2D array | Ground view factor for longwave up |
| gvf_alb | 2D array | Ground view factor weighted by albedo |
| gvf_east | 2D array | GVF from eastern direction |
| gvf_south | 2D array | GVF from southern direction |
| gvf_west | 2D array | GVF from western direction |
| gvf_north | 2D array | GVF from northern direction |

## Properties

### Range Properties

1. **GVF in range [0, 1]**
   - GVF = 0: no ground/walls visible (open sky above)
   - GVF = 1: completely enclosed (no sky visible)

2. **GVF + SVF ≈ 1**
   - For horizontal surfaces: GVF ≈ 1 - SVF
   - Small deviations due to wall contributions

### Geometric Properties

3. **Flat open terrain has GVF ≈ 0**
   - No walls or elevated surfaces to reflect/emit
   - Only ground below contributes

4. **Urban canyon has high GVF**
   - Walls on both sides increase GVF
   - More reflected radiation in canyons

5. **Higher walls increase GVF**
   - Taller buildings → more wall surface visible
   - More longwave emission from walls

### Directional Properties

6. **Directional GVF depends on wall orientation**
   - East-facing wall contributes to gvf_west (seen from west)
   - Asymmetric building layout → asymmetric directional GVF

## Role in Radiation

GVF determines how much radiation comes from surfaces vs sky:

**Reflected Shortwave (Kup)**:
```text
Kup = (I + D) × GVF_alb × ground_albedo
```

**Longwave from Ground (Lup)**:
```text
Lup = ε × σ × Tground^4 × GVF_lup
```

**Longwave from Walls**:
```text
Lwall = ε × σ × Twall^4 × wall_view_factor
```

## Relationship to SVF

| Location | SVF | GVF | Characteristic |
| -------- | --- | --- | -------------- |
| Open field | ~1.0 | ~0.0 | Sky-dominated |
| Street canyon | ~0.4 | ~0.6 | Mixed |
| Courtyard | ~0.2 | ~0.8 | Surface-dominated |
| Under canopy | ~0.1 | ~0.9 | Enclosed |

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**GVF Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
