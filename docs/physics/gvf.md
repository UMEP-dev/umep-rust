# Ground View Factor (GVF)

The Ground View Factor quantifies the fraction of surrounding ground visible from a point, important for reflected radiation calculations.

## Definition

GVF represents the hemispherical view of ground surfaces:

$$GVF = 1 - SVF - WVF$$

Where:

- $SVF$ = Sky View Factor
- $WVF$ = Wall View Factor

## Components

| Component | Description |
|-----------|-------------|
| `gvf` | Total ground view factor |
| `gvf_norm` | Normalized GVF for reflected radiation |

## Role in Radiation

GVF affects upwelling radiation calculations:

1. **Reflected shortwave**: Ground reflects incoming solar radiation
2. **Emitted longwave**: Ground emits thermal radiation based on temperature

```python
# Upwelling shortwave from ground
Kup = albedo * Kdown * gvf

# Upwelling longwave from ground
Lup = emissivity * stefan_boltzmann * T_ground^4 * gvf
```

## Computation

GVF is computed during SVF calculation by tracking rays that hit ground instead of sky:

1. Cast rays from each point
2. Rays not blocked by buildings/vegetation that hit ground contribute to GVF
3. Weight by solid angle

## Performance

GVF is computed alongside SVF with minimal additional cost.

## References

- Lindberg, F., & Grimmond, C. S. B. (2011). The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas. *Theoretical and Applied Climatology*, 105(3), 311-323.
