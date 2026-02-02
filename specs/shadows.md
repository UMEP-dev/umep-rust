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

Vegetation shadows differ from building shadows due to partial light transmission through foliage.

**Primary References:**

- Konarska J, Lindberg F, Larsson A, Thorsson S, Holmer B (2014) "Transmissivity of solar radiation through crowns of single urban trees—application for outdoor thermal comfort modelling." Theoretical and Applied Climatology 117:363-376.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas." Theoretical and Applied Climatology 105:311-323.

### Canopy Transmissivity

**Reference:** Konarska et al. (2014)

Light transmission through tree canopies varies with species, leaf area index (LAI), and season:

| Tree Type | Transmissivity | LAI | Description |
|-----------|----------------|-----|-------------|
| Dense deciduous (summer) | 0.02-0.05 | 5-7 | Oak, maple in full leaf |
| Medium deciduous | 0.05-0.15 | 3-5 | Typical urban trees |
| Open canopy | 0.15-0.30 | 2-3 | Young trees, sparse crown |
| Conifers | 0.10-0.20 | 4-6 | Year-round |
| Deciduous (winter) | 0.60-0.80 | 0-1 | Bare branches only |

**SOLWEIG default:** 0.03 (3%) - represents dense summer canopy, conservative for shade provision studies.

The transmitted radiation through vegetation:

```text
I_transmitted = I_direct × transmissivity
```

### Trunk Zone Ratio

**Reference:** Lindberg & Grimmond (2011)

The trunk zone is the lower portion of the tree where only the solid trunk exists (no foliage). This zone casts solid shadows like buildings.

```text
trunk_height = total_tree_height × trunk_ratio
canopy_height = total_tree_height × (1 - trunk_ratio)
```

**SOLWEIG default:** trunk_ratio = 0.25 (25%)

This means for a 10m tree:
- Trunk zone: 0-2.5m (solid shadow)
- Canopy zone: 2.5-10m (transmissive shadow)

Typical values by tree type:

| Tree Form | Trunk Ratio | Example Species |
|-----------|-------------|-----------------|
| Street tree (pollarded) | 0.30-0.40 | Plane tree, linden |
| Natural form | 0.20-0.30 | Oak, beech |
| Conifer | 0.10-0.20 | Pine, spruce |
| Low-branching | 0.05-0.15 | Magnolia, ornamental |

### Vegetation Shadow Formula

```text
veg_shadow = 1.0                    if ray passes only through trunk (solid)
           = transmissivity         if ray passes through canopy
           = 0.0                    if ray is unobstructed
```

The combined shadow (building + vegetation):

```text
total_shadow = building_shadow × (1 - veg_shadow × (1 - transmissivity))
```
