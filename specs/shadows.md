# Shadow Calculation

Calculates where shadows fall based on sun position, buildings, and vegetation.

**Reference:** Lindberg et al. (2008) Section 2.2 - Shadow casting algorithm

## Equations

### Shadow Length
```
L = h / tan(α)
```
- L = shadow length (meters)
- h = obstacle height above ground (meters)
- α = sun altitude angle (degrees)

### Ray Marching
The algorithm traces rays from each ground pixel toward the sun:
```
dx = -sign(cos(θ)) × step / tan(θ)    # When E-W dominant
dy = sign(sin(θ)) × step               # When E-W dominant
dz = (ds × step × tan(α)) / scale      # Height gain per step
```
- θ = sun azimuth (radians)
- ds = path length correction for diagonal movement

### Shadow Condition
A pixel is in shadow if any obstacle along the ray to the sun is tall enough:
```
shadow[y,x] = 1  if  propagated_height > DSM[y,x]
            = 0  otherwise
```

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| DSM | 2D array (m) | Digital Surface Model - elevation including buildings |
| sun_altitude | float (0-90°) | Sun elevation above horizon |
| sun_azimuth | float (0-360°) | Sun direction (0=N, 90=E, 180=S, 270=W) |
| pixel_size | float (m) | Resolution of DSM |
| CDSM | 2D array (m) | Optional: Canopy DSM for vegetation shadows |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| bldg_sh | 2D array | Building shadow mask (1=sunlit, 0=shadow) |
| veg_sh | 2D array | Vegetation shadow (accounts for transmissivity) |
| wall_sh | 2D array | Shadow height on walls |

## Properties

### Critical Properties

1. **No shadows below horizon**
   - When: sun_altitude ≤ 0°
   - Then: all pixels are "sunlit" (no shadows cast)
   - Reason: No direct beam radiation when sun below horizon

2. **Flat terrain = no shadows**
   - When: DSM is uniform (no elevation differences)
   - Then: shadow mask is all zeros
   - Reason: No obstacles to cast shadows

3. **Binary shadow values**
   - Building shadows are discrete: 0 or 1
   - No partial shadows (penumbra) in building shadow model
   - Vegetation can have fractional values due to transmissivity

### Geometric Properties

4. **Shadows opposite sun direction**
   - Sun from south (180°) → shadows extend north (toward row 0)
   - Sun from east (90°) → shadows extend west (toward col 0)

5. **Lower sun = longer shadows**
   - As altitude decreases, shadow area increases
   - At 45°: shadow length = obstacle height
   - At 30°: shadow length ≈ 1.73 × height
   - At 15°: shadow length ≈ 3.73 × height

6. **Taller obstacles = longer shadows**
   - Shadow length proportional to height: L ∝ h

7. **Shadow length follows equation**
   - Measured shadow length ≈ h / tan(α) within ±15%
   - Tolerance accounts for pixel discretization

### Rooftop Properties

8. **Building tops are sunlit**
   - Rooftops (local maxima) receive direct sun when altitude > 0
   - Unless shaded by taller neighboring buildings

## Vegetation Shadows

Vegetation shadows differ from building shadows:

- **Transmissivity**: Light partially passes through foliage (typically 3-50%)
- **Trunk zone**: Lower portion of tree trunk doesn't cast foliage shadow
- **Pergola effect**: Dappled shadow where canopy partially blocks sun

```
veg_shadow = 1  if ray passes through canopy but not trunk
           = 0  if ray passes through trunk (solid shadow)
```
