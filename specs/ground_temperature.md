# Ground Temperature Model

Surface temperature parameterization for ground longwave emission calculations.

**Primary References:**

- Lindberg F, Onomura S, Grimmond CSB (2016) "Influence of ground surface characteristics on the mean radiant temperature in urban areas." International Journal of Biometeorology 60(9):1439-1452.
- Lindberg F, Grimmond CSB (2011) "The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas." Theoretical and Applied Climatology 105:311-323.

## Overview

Ground surface temperature directly affects upwelling longwave radiation (Lup), which contributes significantly to mean radiant temperature in urban environments. The model accounts for:

1. **Solar heating** - Direct and diffuse radiation absorption
2. **Thermal inertia** - Delayed response due to material heat capacity
3. **Surface properties** - Albedo, emissivity, thermal conductivity

## TsWaveDelay Model

The thermal delay model simulates ground temperature response to changing radiation conditions using an exponential decay function.

### Equation

```text
T_ground(t) = T_current × (1 - w) + T_previous × w

where:
    w = exp(-33.27 × Δt)
    Δt = time since last update (fraction of day)
```

### Parameters

| Parameter | Value | Description |
| --------- | ----- | ----------- |
| Decay constant | 33.27 | Thermal response rate (day⁻¹) |
| Time threshold | 59/1440 | Minimum time step (~59 minutes) |

### Physical Interpretation

The decay constant (33.27 day⁻¹) corresponds to a thermal time constant of approximately:

```text
τ = 1 / 33.27 ≈ 0.030 days ≈ 43 minutes
```

This represents the characteristic time for surface temperature to respond to changes in radiative forcing. After one time constant:

- 63% of adjustment to new equilibrium
- After 3τ (~2 hours): 95% adjustment

### Algorithm

```python
def TsWaveDelay(T_current, first_morning, time_accumulated, timestep, T_previous):
    """
    Apply thermal delay to ground temperature.

    Args:
        T_current: Current radiative equilibrium temperature
        first_morning: True if first timestep after sunrise
        time_accumulated: Time since last full update (fraction of day)
        timestep: Current timestep duration (fraction of day)
        T_previous: Previous delayed temperature

    Returns:
        T_delayed: Temperature with thermal inertia applied
        time_accumulated: Updated time accumulator
        T_previous: Updated previous temperature for next iteration
    """
    if first_morning:
        T_previous = T_current

    if time_accumulated >= 59/1440:  # ~59 minutes threshold
        weight = exp(-33.27 * time_accumulated)
        T_previous = T_current * (1 - weight) + T_previous * weight
        T_delayed = T_previous
        time_accumulated = timestep if timestep > 59/1440 else 0
    else:
        time_accumulated += timestep
        weight = exp(-33.27 * time_accumulated)
        T_delayed = T_current * (1 - weight) + T_previous * weight

    return T_delayed, time_accumulated, T_previous
```

## Surface Temperature Parameterization

For computing the instantaneous radiative equilibrium temperature, SOLWEIG uses a linear parameterization based on solar altitude.

### Linear Model

```text
T_surface = Tstart + k × α_max

where:
    Tstart = surface temperature at sunrise (°C offset from Ta)
    k = temperature increase per degree of solar altitude (°C/°)
    α_max = maximum solar altitude during the day (°)
```

### Land Cover Parameters

| Surface Type | Tstart (°C) | k (°C/°) | TmaxLST | Source |
| ------------ | ----------- | -------- | ------- | ------ |
| Cobblestone | -3.41 | 0.37 | 15:00 | Lindberg et al. (2016) |
| Dark asphalt | -9.78 | 0.58 | 15:00 | Lindberg et al. (2016) |
| Grass | -3.38 | 0.21 | 14:00 | Lindberg et al. (2016) |
| Bare soil | -3.01 | 0.33 | 14:00 | Estimated |
| Water | 0.0 | 0.05 | 16:00 | Estimated |

Note: Tstart is the temperature offset from air temperature at sunrise. Negative values indicate surfaces cooler than air at dawn.

## Properties

### Thermal Inertia Effects

1. **Morning lag** - Surfaces warm slower than instantaneous equilibrium
2. **Afternoon persistence** - Surfaces remain warm after solar maximum
3. **Evening cooling** - Gradual temperature decrease after sunset

### Material Dependence

4. **High thermal mass** (concrete, stone): Slower response, τ > 1 hour
5. **Low thermal mass** (thin asphalt): Faster response, τ < 30 minutes
6. **Vegetation**: Complex due to evapotranspiration

### Diurnal Pattern

```text
Morning:  T_ground < T_equilibrium (heating lag)
Midday:   T_ground ≈ T_equilibrium (near steady state)
Afternoon: T_ground > T_equilibrium (cooling lag)
Night:    T_ground slowly approaches T_air
```

## Implementation Notes

### State Management

The thermal delay model requires state to be carried between timesteps:

- `T_previous`: Last computed delayed temperature
- `time_accumulated`: Time since last weight reset

For accurate results, use `calculate_timeseries()` which automatically manages thermal state. Single-timestep calculations with `calculate()` will not capture thermal inertia effects.

### Directional Components

Ground temperature affects directional Lup components (Lup_E, Lup_S, Lup_W, Lup_N) which are computed using Ground View Factors in each direction.

### Nighttime Behavior

At night (sun_altitude ≤ 0):

- No solar heating contribution
- Temperature decays toward air temperature
- Emissivity assumed constant (typically 0.95)

## Validation Status

The TsWaveDelay model parameters (decay constant 33.27) require validation against:

- [ ] In-situ surface temperature measurements
- [ ] Comparison with force-restore energy balance models
- [ ] Sensitivity analysis for different surface types

The current parameterization is empirical and may need adjustment for specific climates or surface materials.

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**Ground Temperature Model:**

- Lindberg F, Holmer B, Thorsson S (2008) "SOLWEIG 1.0 - Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings." International Journal of Biometeorology 52(7), 697-713.
- Lindberg F, Onomura S, Grimmond CSB (2016) "Influence of ground surface characteristics on the mean radiant temperature in urban areas." International Journal of Biometeorology 60(9), 1439-1452.
- Offerle B, Grimmond CSB, Oke TR (2003) "Parameterization of net all-wave radiation for urban areas." Journal of Applied Meteorology 42(8), 1157-1173.
