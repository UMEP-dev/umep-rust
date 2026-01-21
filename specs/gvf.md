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
