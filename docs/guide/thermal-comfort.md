# Thermal Comfort Indices

SOLWEIG calculates two thermal comfort indices: UTCI and PET.

## UTCI (Universal Thermal Climate Index)

UTCI represents the air temperature of a reference environment that produces the same thermal strain as the actual environment.

### Single Point

```python
result = solweig.calculate(surface, location, weather)

# Compute UTCI from result
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")
```

### Batch Processing

```python
n_files = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)
```

### UTCI Stress Categories

| UTCI (°C) | Stress Category |
|-----------|-----------------|
| > 46      | Extreme heat stress |
| 38 to 46  | Very strong heat stress |
| 32 to 38  | Strong heat stress |
| 26 to 32  | Moderate heat stress |
| 9 to 26   | No thermal stress |
| 0 to 9    | Slight cold stress |
| -13 to 0  | Moderate cold stress |
| -27 to -13| Strong cold stress |
| < -40     | Extreme cold stress |

### Performance

UTCI uses a fast polynomial approximation (~200 terms):

- Single grid: ~1ms
- 72 timesteps: ~1s

---

## PET (Physiological Equivalent Temperature)

PET is the air temperature in a reference environment at which the heat balance of the human body is maintained with core and skin temperatures equal to those under the actual conditions.

### Single Point

```python
result = solweig.calculate(surface, location, weather)

# Compute PET from result
pet = result.compute_pet(weather)
print(f"Mean PET: {pet.mean():.1f}°C")
```

### With Custom Human Parameters

```python
pet = result.compute_pet(
    weather,
    human=solweig.HumanParams(
        weight=60,
        height=1.65,
        posture="standing",
    )
)
```

### Batch Processing

```python
n_files = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
    human=solweig.HumanParams(weight=70, height=1.75),
)
```

### PET Thermal Sensation

| PET (°C) | Thermal Perception | Physiological Stress |
|----------|-------------------|---------------------|
| > 41     | Very hot          | Extreme heat stress |
| 35 to 41 | Hot               | Strong heat stress |
| 29 to 35 | Warm              | Moderate heat stress |
| 23 to 29 | Slightly warm     | Slight heat stress |
| 18 to 23 | Comfortable       | No thermal stress |
| 13 to 18 | Slightly cool     | Slight cold stress |
| 8 to 13  | Cool              | Moderate cold stress |
| 4 to 8   | Cold              | Strong cold stress |
| < 4      | Very cold         | Extreme cold stress |

### Performance

PET uses an iterative solver and is significantly slower:

- Single grid: ~50ms
- 72 timesteps: ~1 minute

!!! warning "PET is ~50× slower than UTCI"
    For large-scale studies, consider using UTCI unless PET's physiological basis is specifically needed.

---

## Choosing Between UTCI and PET

| Factor | UTCI | PET |
|--------|------|-----|
| Speed | Fast (~200 terms polynomial) | Slow (iterative solver) |
| Applicability | -50°C to +50°C | Wider range |
| Human parameters | Fixed reference person | Customizable |
| Common use | Heat warnings, urban planning | Detailed thermal comfort studies |

For most urban microclimate applications, **UTCI is recommended** due to its speed and standardized interpretation.
