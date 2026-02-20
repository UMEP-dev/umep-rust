# Sky View Factor (SVF)

The Sky View Factor quantifies the fraction of sky visible from a point on the ground.

## Definition

$$SVF = \frac{1}{\pi} \int_0^{2\pi} \int_0^{\pi/2} \cos(\theta) \sin(\theta) \, d\theta \, d\phi$$

Where:

- $\theta$ = zenith angle
- $\phi$ = azimuth angle

SVF ranges from 0 (no sky visible) to 1 (open sky).

## Directional Components

SOLWEIG computes directional SVF for anisotropic radiation:

| Component | Description |
|-----------|-------------|
| `svf` | Total sky view factor |
| `svf_north` | Northern hemisphere contribution |
| `svf_south` | Southern hemisphere contribution |
| `svf_east` | Eastern hemisphere contribution |
| `svf_west` | Western hemisphere contribution |

## Algorithm

SVF is computed using hemisphere sampling with configurable resolution:

1. Cast rays from each ground point across the hemisphere
2. Check occlusion against DSM (buildings) and CDSM (vegetation)
3. Weight visible rays by solid angle
4. Sum contributions for total and directional components

## Vegetation Handling

Vegetation (CDSM) partially blocks sky view with transmissivity:

- **Trans** = trunk zone transmissivity (~0.43 default)
- **TransVeg** = vegetation transmissivity function

```python
# Effective SVF through vegetation
svf_veg = svf * trans_veg + svf_building * (1 - trans_veg)
```

## Performance

SVF computation is expensive (O(n² × rays)):

| Grid Size | Computation Time |
|-----------|-----------------|
| 100×100   | ~5 seconds |
| 200×200   | ~67 seconds |
| 500×500   | ~10 minutes |

SVF only depends on geometry, so it's computed once and cached.

## References

- Lindberg, F., & Grimmond, C. S. B. (2011). The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas: model development and evaluation. *Theoretical and Applied Climatology*, 105(3), 311-323.
