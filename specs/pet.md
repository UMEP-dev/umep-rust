# Physiological Equivalent Temperature (PET)

The air temperature at which, in a typical indoor setting, the human energy budget is balanced with the same core and skin temperature as under the actual outdoor conditions.

**Primary References:**

- Höppe P (1999) "The physiological equivalent temperature - a universal index for the biometeorological assessment of the thermal environment." International Journal of Biometeorology 43:71-75.
- Mayer H, Höppe P (1987) "Thermal comfort of man in different urban environments." Theoretical and Applied Climatology 38:43-49.
- VDI 3787 Part 2 (2008) "Environmental Meteorology - Methods for the human biometeorological evaluation of climate and air quality for urban and regional planning."

## MEMI Energy Balance Model

**Reference:** Höppe P (1984) "Die Energiebilanz des Menschen." Wiss Mitt Meteorol Inst Univ München 49.

PET is calculated using the Munich Energy Balance Model for Individuals (MEMI), a two-node model of human thermoregulation:

```text
M + W = R + C + E_sk + E_re + S
```

Where:

- M = metabolic rate (W)
- W = mechanical work (W), typically ~0 for sedentary activities
- R = net radiation heat flow (W)
- C = convective heat flow (W)
- E_sk = latent heat flow from skin evaporation (W)
- E_re = respiratory heat loss (latent + sensible) (W)
- S = body heat storage (W), positive = body warming

**PET Definition:** The air temperature at which, in a reference indoor environment (Tmrt = Ta, v = 0.1 m/s, RH = 50%), the human body would have the same core and skin temperature as in the actual outdoor environment. The 50% RH reference condition is approximated by a fixed vapor pressure of 12 hPa.

### Metabolic Rate

**Reference:** ISO 8996:2021 "Ergonomics of the thermal environment - Determination of metabolic rate."

| Activity | Work Parameter (W) | Description |
| --- | --- | --- |
| Resting | 0 | Lying quietly |
| Sitting | 0 | Office work |
| Standing relaxed | 0 | Standing still |
| Light walking | 80 | 2 km/h (SOLWEIG default) |
| Normal walking | 110 | 4 km/h |
| Brisk walking | 150 | 6 km/h |

The `work` parameter (in Watts total, not W/m²) is added directly to the whole-body basal metabolic rate, matching the upstream UMEP MEMI implementation: `met = metbm + work`. The default SOLWEIG value of 80 W represents light outdoor walking.

## Inputs

| Input | Type | Description |
| ----- | ---- | ----------- |
| Ta | float or 2D array (°C) | Air temperature |
| Tmrt | float or 2D array (°C) | Mean radiant temperature |
| v | float or 2D array (m/s) | Wind speed |
| RH | float or 2D array (%) | Relative humidity |
| age | float (years) | Person's age |
| height | float (m) | Person's height |
| weight | float (kg) | Person's weight |
| sex | int | 1=male, 2=female |
| activity | float (W) | Activity level added to basal metabolic rate (Watts total) |
| clothing | float (clo) | Clothing insulation |

## Outputs

| Output | Type | Description |
| ------ | ---- | ----------- |
| PET | float or 2D array (°C) | Physiological Equivalent Temperature |

## Default Human Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| age | 35 years | Middle-aged adult |
| height | 1.75 m | Average height |
| weight | 75 kg | Average weight |
| sex | 1 (male) | Reference person |
| activity | 80 W | Light walking |
| clothing | 0.9 clo | Summer business attire |

## Comfort Categories

| PET (°C) | Thermal Perception | Grade of Stress |
| -------- | ------------------ | --------------- |
| > 41 | Very hot | Extreme heat stress |
| 35 to 41 | Hot | Strong heat stress |
| 29 to 35 | Warm | Moderate heat stress |
| 23 to 29 | Slightly warm | Slight heat stress |
| 18 to 23 | Comfortable | No thermal stress |
| 13 to 18 | Slightly cool | Slight cold stress |
| 8 to 13 | Cool | Moderate cold stress |
| 4 to 8 | Cold | Strong cold stress |
| < 4 | Very cold | Extreme cold stress |

## Properties

### Fundamental Properties

1. **PET is person-specific**
   - Varies with age, sex, fitness level
   - Same environment can have different PET for different people

2. **PET reference is indoor**
   - Reference: Tmrt=Ta, v=0.1m/s, RH=50%
   - PET=21°C is comfortable indoors

### Radiation Properties

3. **Higher Tmrt → higher PET**
   - Radiation increases heat load
   - Sun to shade: ΔPET ≈ 5-15°C

4. **PET more sensitive to radiation than UTCI**
   - Direct sun has larger effect on PET
   - Better captures radiant heat stress

### Personal Factor Properties

5. **Activity increases PET**
   - Higher metabolic rate → more heat generated
   - Running vs standing: ΔPET ≈ 5-10°C

6. **Clothing affects PET bidirectionally**
   - In heat: more clothing → higher PET
   - In cold: more clothing → lower PET (better insulated)

7. **Age affects thermoregulation**
   - Elderly have reduced sweating capacity
   - Children have higher surface-to-mass ratio

### Wind Properties

8. **Wind generally reduces PET**
   - Convective heat loss increases
   - Less effective at high humidity

## Comparison: PET vs UTCI

| Aspect | PET | UTCI |
| ------ | --- | ---- |
| Reference | Indoor environment | Outdoor walking |
| Personal factors | Yes (age, sex, etc.) | No (fixed person) |
| Clothing | Variable input | Fixed (adaptive) |
| Activity | Variable input | Fixed (walking 4 km/h) |
| Computation | Iterative solver | Polynomial |
| Speed | Slower | Faster |

## Typical Values

| Condition | Ta | Tmrt | PET | Perception |
| --------- | -- | ---- | --- | ---------- |
| Hot sunny | 35 | 65 | 48 | Very hot |
| Hot shaded | 35 | 40 | 38 | Hot |
| Pleasant | 22 | 25 | 22 | Comfortable |
| Cool shade | 18 | 18 | 17 | Slightly cool |
| Cold | 5 | 5 | 5 | Cold |

## Implementation Notes

### Iterative Solution

PET requires solving the energy balance iteratively to find the equivalent temperature. The algorithm:

1. Compute skin and core temperatures for actual outdoor conditions using a 7-mode thermoregulation loop (j = 1..7), each mode representing different physiological states (sweating, vasoconstriction, etc.)
2. Within each mode, a 4-pass bracketing scheme finds the clothing temperature (tcl) with decreasing step sizes: 1.0 → 0.1 → 0.01 → 0.001°C. Each pass runs up to 200 iterations, searching for a sign change in the energy balance.
3. Once outdoor skin/core temperatures converge, repeat the same 4-pass bracketing for the PET reference indoor conditions (Tmrt = Ta, v = 0.1 m/s, vapor pressure = 12 hPa) to find the equivalent temperature.
4. Effective precision: 0.001°C.

### Body Surface Area (DuBois Formula)

**Reference:** DuBois D, DuBois EF (1916) "A formula to estimate the approximate surface area if height and weight be known." Archives of Internal Medicine 17:863-871.

The body surface area A_body (m²) is calculated from height (m) and weight (kg):

```text
A_body = 0.203 × height^0.725 × weight^0.425
```

This empirical formula, derived from direct body surface measurements, remains the standard for thermoregulation calculations. For the default person (1.75m, 75kg):

```text
A_body = 0.203 × 1.75^0.725 × 75^0.425 ≈ 1.90 m²
```

### Clothing Insulation

**Reference:** ISO 9920:2007 "Ergonomics of the thermal environment - Estimation of thermal insulation and water vapour resistance of a clothing ensemble."

Clothing insulation is measured in clo units (1 clo = 0.155 m²K/W):

| Ensemble | Insulation (clo) | Description |
|----------|------------------|-------------|
| Shorts only | 0.1 | Minimal |
| Light summer | 0.5 | T-shirt, shorts |
| Summer business | 0.9 | Shirt, trousers (SOLWEIG default) |
| Winter indoor | 1.0 | Sweater, trousers |
| Winter outdoor | 1.5-2.0 | Coat, layers |

Two clothing-related factors are computed:

**Clothing area factor** (linear, from ISO 9920):

```text
fcl = 1 + 0.15 × Icl
```

**Fraction of body covered by clothing** (cubic polynomial, from Hoeppe MEMI):

```text
facl = (-2.36 + 173.51×Icl - 100.76×Icl² + 19.28×Icl³) / 100
```

These serve different roles: `fcl` scales the total clothing surface area for heat transfer calculations, while `facl` determines the fraction of skin covered for radiation absorption.

### Convective Heat Transfer

**Reference:** Höppe P (1999) "The physiological equivalent temperature." International Journal of Biometeorology 43:71-75.

Convective heat transfer coefficient (W/m²K), from the Hoeppe MEMI formulation:

```text
h_c = 2.67 + 6.5 × v^0.67
h_c = h_c × (P / P₀)^0.55
```

Where:

- v = wind speed (m/s)
- P = local barometric pressure (hPa)
- P₀ = standard sea-level pressure (1013.25 hPa)

The pressure correction accounts for altitude effects on air density. During the PET reference phase (indoor conditions), v = 0.1 m/s is used.

## References

**Primary UMEP Citation:**

- Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) "Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services." Environmental Modelling and Software 99, 70-87. [doi:10.1016/j.envsoft.2017.09.020](https://doi.org/10.1016/j.envsoft.2017.09.020)

**PET Model:**

- Höppe P (1999) "The physiological equivalent temperature - a universal index for the biometeorological assessment of the thermal environment." International Journal of Biometeorology 43(2), 71-75.
- Matzarakis A, Mayer H, Iziomon MG (1999) "Applications of a universal thermal index: physiological equivalent temperature." International Journal of Biometeorology 43(2), 76-84.

**Human Thermal Balance:**

- Fanger PO (1970) "Thermal Comfort: Analysis and Applications in Environmental Engineering." Danish Technical Press, Copenhagen.
- Gagge AP, Fobelets AP, Berglund LG (1986) "A standard predictive index of human response to the thermal environment." ASHRAE Transactions 92, 709-731.
