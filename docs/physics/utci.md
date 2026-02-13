# UTCI (Universal Thermal Climate Index)

UTCI represents the air temperature of a reference environment that produces the same thermal strain as the actual environment.

## Definition

UTCI is based on a multi-node thermophysiological model that simulates:

- Heat exchange between body and environment
- Thermoregulation (sweating, shivering, vasodilation)
- Clothing adaptation

## Reference Conditions

The reference environment has:

- 50% relative humidity (vapor pressure capped at 20 hPa)
- Wind speed 0.5 m/s at 10m height
- Tmrt = Air temperature
- Metabolic rate 135 W/m² (walking 4 km/h)

## Input Variables

| Variable | Symbol | Units |
|----------|--------|-------|
| Air temperature | $T_a$ | °C |
| Mean radiant temperature | $T_{mrt}$ | °C |
| Wind speed (10m) | $v_{10}$ | m/s |
| Relative humidity | $RH$ | % |

## Polynomial Approximation

SOLWEIG uses a fast polynomial approximation (~200 terms):

$$UTCI = T_a + f(T_a, T_{mrt} - T_a, v_{10}, e)$$

Where $e$ is water vapor pressure (hPa).

## Validity Range

| Variable | Min | Max |
|----------|-----|-----|
| $T_a$ | -50°C | +50°C |
| $T_{mrt} - T_a$ | -30°C | +70°C |
| $v_{10}$ | 0.5 m/s | 17 m/s |

## Stress Categories

| UTCI (°C) | Stress Category |
|-----------|-----------------|
| > 46 | Extreme heat stress |
| 38 to 46 | Very strong heat stress |
| 32 to 38 | Strong heat stress |
| 26 to 32 | Moderate heat stress |
| 9 to 26 | No thermal stress |
| 0 to 9 | Slight cold stress |
| -13 to 0 | Moderate cold stress |
| -27 to -13 | Strong cold stress |
| < -40 | Extreme cold stress |

## Usage

```python
result = solweig.calculate(surface, location, weather)

# Compute UTCI from Tmrt
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")
```

## Performance

UTCI computation is fast (~1ms per grid) due to the polynomial approximation.

## References

- Jendritzky, G., de Dear, R., & Havenith, G. (2012). UTCI—Why another thermal index? *International Journal of Biometeorology*, 56(3), 421-428.
- Bröde, P., et al. (2012). Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). *International Journal of Biometeorology*, 56(3), 481-494.
