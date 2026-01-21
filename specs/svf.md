# Sky View Factor (SVF)

Fraction of the sky hemisphere visible from each point. Determines how much diffuse sky radiation and longwave sky emission reaches the surface.

**Reference:** Lindberg et al. (2008) Section 2.1, Lindberg & Grimmond (2011)

## Equation

SVF is the ratio of radiation received from the sky to that from an unobstructed hemisphere:

```text
SVF = Ω_sky / 2π
```

Where Ω_sky is the solid angle of visible sky (steradians).

For computational purposes, the sky is divided into patches and SVF is accumulated:

```text
SVF = Σ (patch_weight × visibility)
```

Where visibility is 1 if the patch is unobstructed, 0 if blocked.

## Inputs

| Input | Type | Description |
| ----- | ---- | ----------- |
| DSM | 2D array (m) | Digital Surface Model |
| CDSM | 2D array (m) | Canopy DSM for vegetation (optional) |
| pixel_size | float (m) | Resolution |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| svf | 2D array (0-1) | Overall sky view factor |
| svf_north | 2D array (0-1) | SVF from northern sky quadrant |
| svf_east | 2D array (0-1) | SVF from eastern sky quadrant |
| svf_south | 2D array (0-1) | SVF from southern sky quadrant |
| svf_west | 2D array (0-1) | SVF from western sky quadrant |
| svf_veg | 2D array (0-1) | SVF accounting for vegetation |

## Properties

### Range Properties

1. **SVF in range [0, 1]**
   - SVF = 0: no sky visible (e.g., inside building)
   - SVF = 1: full hemisphere visible (open field)
   - All intermediate values valid

2. **Directional SVF in range [0, 1]**
   - Each directional component (N, E, S, W) also bounded by [0, 1]

### Geometric Properties

3. **Flat open terrain = SVF of 1**
   - No obstructions → full sky visibility
   - Tolerance: SVF > 0.95 for truly flat DSM

4. **Deep canyon has low SVF**
   - Urban canyon with H/W ratio > 2 → SVF < 0.5
   - H = building height, W = street width

5. **Taller obstacles reduce SVF**
   - Higher buildings nearby → lower ground-level SVF
   - SVF decreases monotonically with obstacle height

6. **Rooftops have high SVF**
   - Building tops (local maxima) have SVF close to 1
   - Only reduced if taller buildings nearby

7. **Building density reduces SVF**
   - More buildings → lower ground-level SVF
   - SVF is a measure of urban density/openness

### Symmetry Properties

8. **Symmetric obstacles give symmetric directional SVF**
   - Square courtyard center has equal N/E/S/W SVF
   - Asymmetric buildings create asymmetric directional SVF

## Directional SVF

Directional components split the sky into quadrants:

```text
         N (svf_north)
         |
    W ---+--- E
         |
         S
```

Used for calculating radiation from different sky directions, important for:

- Anisotropic sky radiance (brighter near sun)
- Wall orientation effects
- Asymmetric shading

## Vegetation Effects

Trees reduce SVF but not completely (light passes through canopy):

- **SVF_veg**: Sky view through vegetation canopy
- Accounts for leaf area index and transmissivity
- SVF_veg ≥ SVF (vegetation blocks less than buildings)
