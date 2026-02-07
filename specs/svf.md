# Sky View Factor (SVF)

Fraction of the sky hemisphere visible from each point. Determines how much diffuse sky radiation and longwave sky emission reaches the surface.

**Reference:** Lindberg et al. (2008) Section 2.1, Lindberg & Grimmond (2011)

## Equation

SVF is the ratio of radiation received from the sky to that from an unobstructed hemisphere:

```text
SVF = Ω_sky / 2π
```

Where Ω_sky is the solid angle of visible sky (steradians).

## Patch-Based Calculation Method

**Reference:** Robinson & Stone (1990) "Solar Radiation Modelling in the Urban Context", Building and Environment 25(3):201-209.

SOLWEIG uses the patch-based method where the sky hemisphere is divided into discrete angular patches (annuli). SVF is computed by testing visibility to each patch and weighting by the patch's solid angle.

### Patch Configuration

The sky is divided into concentric annuli (altitude bands) from 0° to 90° elevation. Each annulus is further subdivided into azimuthal patches. Standard configurations:

- **Option 1** (145 patches): Coarse for fast computation
- **Option 2** (153 patches - default): Balance of accuracy and speed
- **Option 3** (306 patches): Fine for high accuracy
- **Option 4** (930 patches): Research-grade resolution

For option 2 (153 patches):

- 8 altitude bands: 6°, 18°, 30°, 42°, 54°, 66°, 78°, 90°
- Azimuthal divisions per band: 31, 30, 28, 24, 19, 13, 7, 1

### Solid Angle Weight Calculation

Each patch's contribution to SVF is weighted by its solid angle (steradian):

```text
w_patch = Δφ × (sin(θ_max) - sin(θ_min))
```

Where:
- Δφ = azimuthal width of patch (radians)
- θ_min, θ_max = altitude bounds of annulus (radians)

For patch in annulus i with n_i azimuthal divisions:

```text
Δφ_i = 2π / n_i
w_i = Δφ_i × (sin(θ_i + Δθ/2) - sin(θ_i - Δθ/2))
```

### SVF Accumulation Formula

```text
SVF = Σ_patches (w_patch × visibility_patch)
```

Where:
- w_patch = solid angle weight for the patch
- visibility_patch = 1 if patch center is unobstructed, 0 if blocked by DSM

### Directional SVF

Directional components split patches by azimuth quadrant:

```text
SVF_east  = Σ (w_patch × visibility) for 0° ≤ azimuth < 180°
SVF_south = Σ (w_patch × visibility) for 90° ≤ azimuth < 270°
SVF_west  = Σ (w_patch × visibility) for 180° ≤ azimuth < 360°
SVF_north = Σ (w_patch × visibility) for 270° ≤ azimuth < 90°
```

### Algorithm Implementation

The Rust implementation in `skyview.rs` computes SVF using:

1. **Shadow casting**: For each patch (altitude, azimuth), cast shadows from the DSM
2. **Weight computation**: Calculate solid angle weight using annulus bounds
3. **Accumulation**: Sum weighted visibility across all patches per pixel
4. **Correction factor**: Apply final correction (3.0459e-4) for numerical stability

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

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**SVF Algorithm:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation." Theoretical and Applied Climatology 105, 311-323.

**Patch-Based Method:**

- Robinson D, Stone A (1990) "Solar Radiation Modelling in the Urban Context." Building and Environment 25(3), 201-209.
