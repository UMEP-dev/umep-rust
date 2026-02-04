# Ground Temperature

Ground surface temperature significantly affects upwelling longwave radiation and thermal comfort.

## TsWaveDelay Model

SOLWEIG uses a simplified thermal mass model that accounts for:

1. **Solar heating**: Ground absorbs shortwave radiation
2. **Thermal inertia**: Temperature responds slowly to forcing
3. **Phase lag**: Peak temperature lags peak radiation

## Governing Equation

Ground temperature evolution:

$$T_g(t) = T_{air} + \Delta T_{max} \cdot f(t - \phi)$$

Where:

- $T_{air}$ = Air temperature
- $\Delta T_{max}$ = Maximum ground-air temperature difference
- $\phi$ = Phase lag (thermal delay)
- $f(t)$ = Diurnal temperature wave function

## Land Cover Dependency

Different surfaces have different thermal properties:

| Surface | Thermal Admittance | Typical $\Delta T_{max}$ |
|---------|-------------------|-------------------------|
| Asphalt | High | 15-25°C |
| Concrete | High | 12-20°C |
| Grass | Low | 5-10°C |
| Water | Very High | 2-5°C |

## Shading Effects

Shaded ground has reduced temperature:

$$T_{g,shade} = T_{g,sun} - \Delta T_{shade}$$

Where $\Delta T_{shade}$ depends on shadow duration and surface properties.

## Timeseries Considerations

Ground temperature requires previous timesteps for accurate modeling:

```python
# CORRECT: Full timeseries preserves thermal state
results = solweig.calculate_timeseries(
    surface=surface,
    weather_series=weather_list,
)

# WRONG: Single timestep loses thermal history
result = solweig.calculate(surface, location, weather_noon)
```

## Implementation

Ground temperature is computed in `components/ground.py` using the TsWaveDelay algorithm from UMEP.

## References

- Lindberg, F., Grimmond, C. S. B., & Martilli, A. (2015). Sunlit fractions on urban facets–Impact of spatial resolution and approach. *Urban Climate*, 12, 65-84.
