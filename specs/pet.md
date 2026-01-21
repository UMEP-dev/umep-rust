# Physiological Equivalent Temperature (PET)

The air temperature at which, in a typical indoor setting, the human energy budget is balanced with the same core and skin temperature as under the actual outdoor conditions.

**Reference:** Höppe (1999), Mayer & Höppe (1987)

## Equation

PET is calculated using the Munich Energy Balance Model for Individuals (MEMI):

```text
M + W = R + C + E_sk + E_re + S
```

Where:
- M = metabolic rate (W)
- W = mechanical work (W)
- R = net radiation heat flow (W)
- C = convective heat flow (W)
- E_sk = latent heat flow from skin (W)
- E_re = respiratory heat loss (W)
- S = storage (body heating/cooling) (W)

PET is the Ta at which, in a reference indoor environment (Tmrt=Ta, v=0.1m/s, RH=50%), the same S would result.

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
| activity | float (W/m²) | Metabolic activity level |
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
| activity | 80 W/m² | Light walking |
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

1. **Iterative solution**
   - PET requires solving energy balance iteratively
   - Convergence typically within 10-20 iterations

2. **Body surface area**
   - Calculated from height and weight (DuBois formula)
   - A_body = 0.203 × height^0.725 × weight^0.425

3. **Clothing area factor**
   - Clothing increases effective surface area
   - f_cl = 1 + 0.15 × I_cl (where I_cl in clo)
