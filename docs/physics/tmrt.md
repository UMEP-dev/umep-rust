# Mean Radiant Temperature (Tmrt)

Mean Radiant Temperature is the uniform temperature of an imaginary black enclosure that would result in the same radiant heat exchange as the actual non-uniform environment.

## Definition

$$T_{mrt} = \sqrt[4]{\frac{\sum_i F_i T_i^4}{\sigma}}$$

Where:

- $F_i$ = View factor to surface $i$
- $T_i$ = Temperature of surface $i$
- $\sigma$ = Stefan-Boltzmann constant

## SOLWEIG Calculation

SOLWEIG computes Tmrt from absorbed radiation:

$$T_{mrt} = \sqrt[4]{\frac{S_{str}}{\varepsilon_p \sigma}} - 273.15$$

Where $S_{str}$ is the mean radiant flux density (W/m²).

## Mean Radiant Flux

The mean radiant flux combines all radiation components:

$$S_{str} = a_k (K_{down} + K_{up} + K_{side}) + a_l (L_{down} + L_{up} + L_{side})$$

Where:

- $a_k$ = Shortwave absorptivity (~0.7 for clothed human)
- $a_l$ = Longwave absorptivity (~0.97 for clothed human)
- $K$ = Shortwave radiation components
- $L$ = Longwave radiation components

## Directional Components

For a standing human (cylinder approximation):

| Direction | Weight Factor |
|-----------|--------------|
| Up | 0.06 |
| Down | 0.06 |
| North | 0.22 |
| South | 0.22 |
| East | 0.22 |
| West | 0.22 |

## Typical Values

| Environment | Tmrt Range |
|-------------|-----------|
| Deep shade | ~Air temperature |
| Open sky, summer noon | 50-70°C |
| Near hot pavement | +10-20°C above air |
| Near cool grass | -5-10°C below open |

## Output

```python
result = solweig.calculate(surface, location, weather)

# Tmrt grid (°C)
tmrt = result.tmrt
print(f"Mean Tmrt: {tmrt.mean():.1f}°C")
print(f"Max Tmrt: {tmrt.max():.1f}°C")
```

## References

- Thorsson, S., Lindberg, F., Eliasson, I., & Holmer, B. (2007). Different methods for estimating the mean radiant temperature in an outdoor urban setting. *International Journal of Climatology*, 27(14), 1983-1993.
