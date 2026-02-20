# Universal Thermal Climate Index (UTCI)

Equivalent temperature representing the physiological response to the thermal environment. Based on a multi-node human thermoregulation model.

**Reference:** Bröde et al. (2012), Fiala et al. (2012)

## Equation

UTCI is computed from a 6th-order polynomial approximation of the Fiala model:

```text
UTCI = Ta + offset(Ta, Tmrt-Ta, va, Pa)
```

The offset is a complex polynomial function (~200 terms) of:
- Ta = air temperature (°C)
- ΔTmrt = Tmrt - Ta (°C)
- va = wind speed at 10m (m/s)
- Pa = water vapor pressure (hPa)

## Inputs

| Input | Type | Description |
| ----- | ---- | ----------- |
| Ta | float or 2D array (°C) | Air temperature |
| Tmrt | float or 2D array (°C) | Mean radiant temperature |
| va | float or 2D array (m/s) | Wind speed at 10m height |
| RH | float or 2D array (%) | Relative humidity |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| UTCI | float or 2D array (°C) | Universal Thermal Climate Index |

## Stress Categories

| UTCI (°C) | Stress Category | Physiological Response |
| --------- | --------------- | ---------------------- |
| > 46 | Extreme heat stress | Heat stroke risk |
| 38 to 46 | Very strong heat stress | Failure of thermoregulation |
| 32 to 38 | Strong heat stress | Strong sweating, dehydration |
| 26 to 32 | Moderate heat stress | Sweating, discomfort |
| 9 to 26 | No thermal stress | Comfort zone |
| 0 to 9 | Slight cold stress | Vasoconstriction |
| -13 to 0 | Moderate cold stress | Shivering begins |
| -27 to -13 | Strong cold stress | Risk of frostbite |
| -40 to -27 | Very strong cold stress | Numbness, hypothermia risk |
| < -40 | Extreme cold stress | Frostbite in minutes |

## Properties

### Fundamental Properties

1. **UTCI is an equivalent temperature**
   - Units are °C
   - Represents how the environment "feels"
   - Reference: walking outdoors at 4 km/h

2. **Valid input ranges**
   - Ta: -50°C to +50°C
   - Tmrt-Ta: -30°C to +70°C
   - va: 0.5 to 17 m/s
   - RH: 5% to 100%

### Radiation Properties

3. **Higher Tmrt → higher UTCI**
   - Radiation increases thermal stress
   - Sun to shade: ΔUTCI ≈ 5-20°C

4. **UTCI ≈ Ta when Tmrt = Ta and low wind**
   - No radiation difference, no wind chill
   - Neutral reference condition

### Wind Properties

5. **Wind reduces UTCI in heat**
   - Convective cooling
   - Effect saturates at high wind speeds

6. **Wind reduces UTCI in cold**
   - Wind chill effect
   - Stronger effect in cold conditions

### Humidity Properties

7. **Humidity effect small in cold**
   - Water vapor pressure low at cold temperatures
   - Main effect is in warm/hot conditions

8. **High humidity increases UTCI in heat**
   - Impairs evaporative cooling
   - Tropical conditions feel hotter

## Comparison with Other Indices

| Index | Accounts for | Limitations |
| ----- | ------------ | ----------- |
| UTCI | Ta, Tmrt, wind, humidity | Fixed clothing/activity |
| PET | Ta, Tmrt, wind, humidity, person | More parameters needed |
| Heat Index | Ta, humidity | No radiation or wind |
| Wind Chill | Ta, wind | Cold only, no radiation |

## Typical Values

| Condition | Ta | Tmrt | Wind | UTCI | Category |
| --------- | -- | ---- | ---- | ---- | -------- |
| Hot sunny | 35 | 65 | 1 | 45 | Very strong heat |
| Hot shaded | 35 | 40 | 1 | 36 | Strong heat |
| Comfortable | 22 | 25 | 2 | 22 | No stress |
| Cold windy | -5 | -5 | 10 | -15 | Strong cold |
| Cold calm | -5 | -5 | 1 | -6 | Moderate cold |

## Implementation Notes

1. **Wind height adjustment**
   - Input wind typically at 10m height
   - Model assumes standard reference height

2. **Polynomial approximation**
   - ~200 coefficient polynomial
   - Accurate within ±0.5°C of full model

3. **Extrapolation warning**
   - Results outside valid ranges may be unreliable
   - Clamp or flag out-of-range inputs

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**UTCI Model:**

- Błażejczyk K, Jendritzky G, Bröde P, Fiala D, Havenith G, Epstein Y, Psikuta A, Kampmann B (2013) "An introduction to the Universal Thermal Climate Index (UTCI)." Geographia Polonica 86(1), 5-10.
- Bröde P, Fiala D, Błażejczyk K, Holmér I, Jendritzky G, Kampmann B, Tinz B, Havenith G (2012) "Deriving the operational procedure for the Universal Thermal Climate Index (UTCI)." International Journal of Biometeorology 56(3), 481-494.
