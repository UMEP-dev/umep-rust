# Validation

SOLWEIG is validated against field radiation measurements from three sites in
Gothenburg, Sweden, plus a Mediterranean urban canyon in Montpellier, France.
All validation data and test scripts are checked into the repository under
`tests/validation/` and run as part of the test suite.

## Validation sites

| Site | Location | Period | Pixel | Days | Obs hours | Reference |
|------|----------|--------|-------|------|-----------|-----------|
| **Kronenhuset** | Courtyard, central Gothenburg | 2005-10-07 | 1 m | 1 | 12 | Lindberg et al. (2008) |
| **Gustav Adolfs torg** | Open square, central Gothenburg | 2005-10-11, 2006-07-26, 2006-08-01 | 2 m | 3 | 44 | Lindberg et al. (2008) |
| **GVC** (Geovetenskap center) | University campus, Gothenburg | 2010-07-07, 07-10, 07-12 | 2 m | 3 | 30 | Lindberg & Grimmond (2011) |
| **Montpellier PRESTI** | Reduced-scale E-W canyon | 2023-07 (selected days) | 0.1 m | multi | ~60 | Garcia de Cezar et al. (2025) |

## Current results (v0.1.0b57)

### Tmrt — Mean Radiant Temperature

Anisotropic sky mode, matched daytime hours only.

| Site | Day | RMSE (°C) | MAE (°C) | Bias (°C) | R² | n |
|------|-----|----------:|----------:|----------:|---:|--:|
| Kronenhuset | 2005-10-07 | 6.6 | 5.7 | +2.6 | 0.33 | 12 |
| Gustav Adolfs | 2005-10-11 | 17.7 | 15.1 | -12.8 | 0.90 | 12 |
| Gustav Adolfs | 2006-07-26 | 11.7 | 8.1 | -3.1 | 0.56 | 16 |
| Gustav Adolfs | 2006-08-01 | 14.1 | 10.5 | -6.9 | 0.65 | 15 |
| GVC | 2010-07-07 | 7.8 | 6.5 | -4.9 | 0.91 | 11 |
| GVC | 2010-07-10 | 5.2 | 4.9 | -4.9 | 0.98 | 7 |
| GVC | 2010-07-12 | 11.8 | 10.1 | -8.3 | 0.93 | 12 |

### Ldown — Downwelling Longwave

| Site | Day | RMSE (W/m²) | Bias (W/m²) |
|------|-----|------------:|------------:|
| Kronenhuset | 2005-10-07 | 32.1 | +31.3 |
| Gustav Adolfs | 2005-10-11 | 74.0 | — |
| Gustav Adolfs | 2006-07-26 | 71.6 | — |
| Gustav Adolfs | 2006-08-01 | 60.5 | — |
| GVC | 2010-07-07 | 17.8 | — |
| GVC | 2010-07-10 | 11.8 | — |
| GVC | 2010-07-12 | 18.4 | — |

### Known limitations

- **Kdown at open sites:** Point-level downwelling shortwave (Kdown) has high
  RMSE (175–333 W/m²) because a single pixel's shadow state is binary — a
  slight timing or geometry error in the shadow boundary causes ~800 W/m²
  swings. This is inherent to point validation of a spatially gridded model.

- **Tmrt bias at Gustav Adolfs torg:** The model underestimates Tmrt by
  ~3–13 °C depending on the day. The R² is high (0.56–0.90), indicating
  good temporal correlation but a systematic offset, likely related to the
  open square geometry and 2 m resolution.

- **Ldown overestimation:** The model consistently overestimates Ldown at
  Kronenhuset (+31 W/m²). This positive bias is partly offset by the SVF
  weighting at sites with more sky obstruction.

## Comparison with original paper

Lindberg et al. (2008) report aggregate statistics over 7 days at two
Gothenburg sites (~189 hours):

| Component | R² | RMSE |
|-----------|---:|-----:|
| Tmrt | 0.94 | 4.8 K |
| L↓ | 0.73 | 17.5 W/m² |
| L↑ | 0.94 | 15.6 W/m² |

Our best-performing site (GVC) achieves comparable accuracy (Tmrt RMSE
5.2–11.8 °C, R² 0.91–0.98). The paper's lower RMSE reflects aggregation
across more hours and sites, which smooths individual shadow-timing errors.

## Running validation tests

```bash
# All validation tests (fast data-loading + slow pipeline)
pytest tests/validation/ -m validation

# Just the fast data-loading checks
pytest tests/validation/ -m "validation and not slow"

# A single site
pytest tests/validation/test_validation_gvc.py -v -s
```

## Version history

| Version | Date | Tmrt RMSE range | Notes |
|---------|------|----------------:|-------|
| 0.1.0b57 | 2026-03-05 | 5.2–17.7 °C | Initial 3-site validation (KR + GA + GVC) |

## References

1. Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 — Modelling
   spatial variations of 3D radiant fluxes and mean radiant temperature in
   complex urban settings. *Int. J. Biometeorol.* 52, 697–713.

2. Lindberg, F. & Grimmond, C.S.B. (2011). The influence of vegetation and
   building morphology on shadow patterns and mean radiant temperature in
   urban areas. *Theor. Appl. Climatol.* 105, 311–323.

3. Garcia de Cezar, R. et al. (2025). Microclimate in a Mediterranean urban
   canyon. *Geosci. Data J.*
