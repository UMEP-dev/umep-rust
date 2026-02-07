# Shadow Calculation

Shadow computation determines which grid cells are shaded from direct solar radiation at a given sun position.

## Sun Position

Sun position is computed from:

- **Latitude/Longitude**: Geographic location
- **DateTime**: Local time with UTC offset
- **Algorithm**: NREL Solar Position Algorithm (SPA)

Outputs:

- **Altitude** ($\alpha$): Angle above horizon (0-90°)
- **Azimuth** ($\psi$): Compass direction (0-360°, north=0°)

## Shadow Algorithm

SOLWEIG uses a shadow volume approach:

1. For each building pixel, compute shadow projection based on sun angle
2. Project shadow along solar azimuth
3. Shadow length: $L = h / \tan(\alpha)$ where $h$ is building height

```python
# Shadow projection distance
shadow_length = height / tan(sun_altitude)

# Shadow direction (opposite to sun)
shadow_azimuth = (sun_azimuth + 180) % 360
```

## Building Shadows

Building shadows are binary (0 or 1):

- **0** = Sunlit
- **1** = Shaded by building

## Vegetation Shadows

Vegetation provides partial shade with transmissivity:

$$F_{sh,veg} = 1 - T_{veg}$$

Where $T_{veg}$ depends on leaf area index (LAI) and path length through canopy.

## Shadow Matrices

For anisotropic sky calculations, SOLWEIG pre-computes shadow patterns for multiple sun positions covering the sky hemisphere.

## Output

The shadow calculation produces:

| Output | Description |
|--------|-------------|
| `shadow` | Combined shadow fraction (0-1) |
| `shadow_building` | Building shadow mask |
| `shadow_vegetation` | Vegetation shadow fraction |

## References

- Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0–Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *International Journal of Biometeorology*, 52(7), 697-713.
