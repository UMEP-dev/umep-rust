# Thermal Comfort Indices

Tmrt tells you how much radiation a person absorbs, but thermal comfort also depends on air temperature, humidity, and wind. SOLWEIG computes two standard indices that combine all these factors into a single "feels like" temperature.

## UTCI (Universal Thermal Climate Index)

UTCI is the most widely used outdoor thermal comfort index. It represents the air temperature of a reference environment that would produce the same thermal strain as the actual conditions.

**Use UTCI when:** You want a standardised, fast metric for heat stress mapping, urban planning, or public health applications.

### From a single result

```python
result = solweig.calculate(surface, location, weather)
utci = result.compute_utci(weather)
print(f"Mean UTCI: {utci.mean():.1f}°C")
```

### Batch from saved Tmrt files

```python
n_files = solweig.compute_utci(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_utci/",
)
```

### UTCI stress categories

| UTCI (°C) | Thermal stress |
| ---------- | -------------- |
| > 46 | Extreme heat stress |
| 38 to 46 | Very strong heat stress |
| 32 to 38 | Strong heat stress |
| 26 to 32 | Moderate heat stress |
| 9 to 26 | No thermal stress |
| 0 to 9 | Slight cold stress |
| -13 to 0 | Moderate cold stress |
| -27 to -13 | Strong cold stress |
| < -40 | Extreme cold stress |

### Performance

UTCI uses a fast polynomial approximation (~200 terms). Processing time is negligible compared to the main Tmrt calculation:

- Single grid: ~1 ms
- 72 timesteps: ~1 s

---

## PET (Physiological Equivalent Temperature)

PET is the air temperature of a reference indoor environment at which the human heat balance equals the actual outdoor conditions. Unlike UTCI, PET allows you to customise body parameters.

**Use PET when:** You need thermal comfort for specific populations (elderly, children, athletes) or when research requires the physiological model.

### Single result

```python
result = solweig.calculate(surface, location, weather)
pet = result.compute_pet(weather)
print(f"Mean PET: {pet.mean():.1f}°C")
```

### With custom human parameters

```python
pet = result.compute_pet(
    weather,
    human=solweig.HumanParams(
        weight=60,           # kg
        height=1.65,         # m
        age=70,              # years
        sex=0,               # 0 = female, 1 = male
        activity=80.0,       # metabolic rate (W)
        clothing=0.5,        # clothing insulation (clo)
        posture="standing",
    ),
)
```

### PET batch processing

```python
n_files = solweig.compute_pet(
    tmrt_dir="output/",
    weather_series=weather_list,
    output_dir="output_pet/",
    human=solweig.HumanParams(weight=70, height=1.75),
)
```

### PET thermal sensation

| PET (°C) | Perception | Physiological stress |
| --------- | ---------- | -------------------- |
| > 41 | Very hot | Extreme heat stress |
| 35 to 41 | Hot | Strong heat stress |
| 29 to 35 | Warm | Moderate heat stress |
| 23 to 29 | Slightly warm | Slight heat stress |
| 18 to 23 | Comfortable | No thermal stress |
| 13 to 18 | Slightly cool | Slight cold stress |
| 8 to 13 | Cool | Moderate cold stress |
| 4 to 8 | Cold | Strong cold stress |
| < 4 | Very cold | Extreme cold stress |

### PET performance

PET uses an iterative solver and is significantly slower than UTCI:

- Single grid: ~50 ms
- 72 timesteps: ~1 minute

!!! warning "PET is ~50x slower than UTCI"
    For large-scale studies, use UTCI unless PET's customisable body parameters are specifically needed.

---

## Choosing between UTCI and PET

| | UTCI | PET |
| - | ---- | --- |
| **Speed** | Fast (polynomial) | Slow (iterative) |
| **Human parameters** | Fixed reference person | Customisable (age, weight, clothing, etc.) |
| **Best for** | Heat warnings, urban planning, large-scale mapping | Detailed comfort studies, vulnerable populations |
| **Common in** | European heat action plans, WMO guidelines | German VDI guidelines, bioclimatology research |

For most day-to-day urban microclimate work, **UTCI is the recommended default**. It's fast, widely understood, and has standardised stress categories used in public health guidance.
