# PET (Physiological Equivalent Temperature)

PET is the air temperature in a reference environment at which the heat balance of the human body is maintained with core and skin temperatures equal to those under the actual conditions.

## Definition

PET is based on the Munich Energy Balance Model for Individuals (MEMI), which solves:

$$M + W = R + C + E_{sk} + E_{res} + S$$

Where:

- $M$ = Metabolic rate
- $W$ = Mechanical work
- $R$ = Net radiation
- $C$ = Convective heat loss
- $E_{sk}$ = Evaporative heat loss (skin)
- $E_{res}$ = Respiratory heat loss
- $S$ = Heat storage

## Reference Conditions

The reference environment for PET has:

- Wind speed 0.1 m/s
- Water vapor pressure 12 hPa
- Tmrt = Air temperature

## Input Variables

| Variable | Symbol | Units |
|----------|--------|-------|
| Air temperature | $T_a$ | °C |
| Mean radiant temperature | $T_{mrt}$ | °C |
| Wind speed | $v$ | m/s |
| Relative humidity | $RH$ | % |

## Human Parameters

Unlike UTCI, PET allows customizable human parameters:

```python
human = solweig.HumanParams(
    weight=70,      # kg
    height=1.75,    # m
    age=35,         # years
    sex=1,          # 1=male, 2=female
    posture="standing",  # or "sitting"
)

pet = result.compute_pet(weather, human=human)
```

## Thermal Perception Scale

| PET (°C) | Thermal Perception | Physiological Stress |
|----------|-------------------|---------------------|
| > 41 | Very hot | Extreme heat stress |
| 35 to 41 | Hot | Strong heat stress |
| 29 to 35 | Warm | Moderate heat stress |
| 23 to 29 | Slightly warm | Slight heat stress |
| 18 to 23 | Comfortable | No thermal stress |
| 13 to 18 | Slightly cool | Slight cold stress |
| 8 to 13 | Cool | Moderate cold stress |
| 4 to 8 | Cold | Strong cold stress |
| < 4 | Very cold | Extreme cold stress |

## Algorithm

PET uses an iterative solver:

1. Calculate body heat balance under actual conditions
2. Determine core and skin temperatures
3. Iteratively find reference air temperature that produces same thermal state
4. Convergence typically requires 20-50 iterations

## Performance

PET is significantly slower than UTCI due to the iterative solver:

| Metric | UTCI | PET |
|--------|------|-----|
| Single point | ~0.01 ms | ~0.5 ms |
| 100×100 grid | ~1 ms | ~50 ms |
| 72 timesteps | ~1 s | ~1 min |

!!! warning "PET is ~50× slower than UTCI"
    For large-scale studies, consider using UTCI unless PET's physiological basis is specifically needed.

## Usage

```python
result = solweig.calculate(surface, location, weather)

# Compute PET with default human
pet = result.compute_pet(weather)

# Compute PET with custom parameters
pet = result.compute_pet(
    weather,
    human=solweig.HumanParams(weight=60, height=1.65)
)
```

## References

- Höppe, P. (1999). The physiological equivalent temperature–a universal index for the biometeorological assessment of the thermal environment. *International Journal of Biometeorology*, 43(2), 71-75.
- Matzarakis, A., Rutz, F., & Mayer, H. (2007). Modelling radiation fluxes in simple and complex environments. *International Journal of Biometeorology*, 51(4), 323-334.
