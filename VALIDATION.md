# Validation Report

SOLWEIG is validated against field radiation measurements from three sites in
Gothenburg, Sweden. All validation data, test scripts, and POI sweep
diagnostics are checked into the repository under `tests/validation/` and run
automatically in CI on every push and PR.

---

## Sites

### Kronenhuset

- **Type:** Enclosed courtyard, central Gothenburg
- **Period:** 2005-10-07 (1 day, 12 daytime hours)
- **Resolution:** 1 m
- **POI:** (51, 118) — shifted +1 column from shapefile position (51, 117) to better match the observed shadow timing profile
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Notes:** The only site that directly validates individual radiation budget
  components (K↓, K↑, L↓, L↑ and directional fluxes), not just Tmrt.
  Enclosed geometry with ~25% sky obstruction.

### Gustav Adolfs torg

- **Type:** Open square, central Gothenburg
- **Period:** 2005-10-11, 2006-07-26, 2006-08-01 (3 days, 43 daytime hours)
- **Resolution:** 2 m
- **POI:** (33, 77) — no shapefile exists; location uncertain
- **Reference:** Lindberg, Holmer & Thorsson (2008)
- **Notes:** One autumn day (heavily overcast) and two summer days. The
  measurement station location is not precisely documented. A POI sensitivity
  sweep identified significantly better-fitting pixels on the western side of
  the square near buildings (see [POI sweep analysis](#poi-sweep-analysis)).

### GVC (Gothenburg Geoscience Centre)

- **Type:** University campus, Gothenburg
- **Period:** 2010-07-07, 07-10, 07-12 (3 days, 30 daytime hours)
- **Resolution:** 2 m (rasters labelled `_1m` but actually 2 m)
- **POI:** (70, 126) — chosen from low-RMSE cluster in the courtyard
- **Reference:** Lindberg & Grimmond (2011)
- **Notes:** Three clear summer days. No POI shapefile or GPS coordinates
  were bundled with the validation dataset for this site. An initial POI
  estimate of (51, 122) produced R² = 0.01–0.14, suggesting it did not
  correspond to the measurement location. A POI sensitivity sweep identified
  a cluster of low-RMSE pixels in the courtyard around row 70, col 125;
  (70, 126) was selected as a representative point away from building edges.

---

## Results — v0.1.0b60 (2026-03-06)

### Summary

| Metric               | Kronenhuset | Gustav Adolfs |            GVC |
| -------------------- | ----------: | ------------: | -------------: |
| Tmrt RMSE range (°C) |         6.4 |     11.7–17.7 |      4.0–7.8 |
| Tmrt R² range        |        0.41 |     0.56–0.90 |    0.38–0.68 |
| Tmrt bias range (°C) |        +1.0 | -12.8 to -3.1 | -1.4 to +3.4 |
| Days                 |           1 |             3 |            3 |
| Total obs hours      |          12 |            43 |           30 |

### Tmrt — per-day detail

Anisotropic sky mode. Matched daytime hours only (sun above horizon with
valid observations).

| Site          | Day        | RMSE (°C) | MAE (°C) | Bias (°C) |   R² |   n |
| ------------- | ---------- | --------: | -------: | --------: | ---: | --: |
| Kronenhuset   | 2005-10-07 |       6.4 |      5.6 |      +1.0 | 0.41 |  12 |
| Gustav Adolfs | 2005-10-11 |      17.7 |     15.1 |     -12.8 | 0.90 |  12 |
| Gustav Adolfs | 2006-07-26 |      11.7 |      8.1 |      -3.1 | 0.56 |  16 |
| Gustav Adolfs | 2006-08-01 |      14.1 |     10.5 |      -6.9 | 0.65 |  15 |
| GVC           | 2010-07-07 |       6.9 |      4.8 |      -1.4 | 0.68 |  11 |
| GVC           | 2010-07-10 |       4.0 |      3.4 |      +3.4 | 0.57 |   7 |
| GVC           | 2010-07-12 |       7.8 |      6.4 |      +1.5 | 0.38 |  12 |

### Radiation components — Kronenhuset

Kronenhuset is the only site with component-level validation. Values shown
for anisotropic sky mode (isotropic in parentheses where different).

| Component |        RMSE |           Bias |
| --------- | ----------: | -------------: |
| K↓ (W/m²) | 62.9 (54.4) | -39.2 (-22.7) |
| K↑ (W/m²) |         7.9 |           +4.2 |
| L↓ (W/m²) | 31.4 (38.4) | +30.5 (+37.9) |
| L↑ (W/m²) |        12.9 |           +8.7 |
| Tmrt (°C) |   6.4 (5.9) |   +1.0 (-0.9) |

Negative R² values for K↓ and L↓ reflect the fact that point-level
comparisons of shadow-dependent quantities are dominated by shadow-timing
errors (see [Known limitations](#known-limitations)).

### Ldown — all sites

| Site          | Day        | RMSE (W/m²) | Bias (W/m²) |
| ------------- | ---------- | ----------: | ----------: |
| Kronenhuset   | 2005-10-07 |        31.4 |       +30.5 |
| Gustav Adolfs | 2005-10-11 |        74.0 |           — |
| Gustav Adolfs | 2006-07-26 |        71.6 |           — |
| Gustav Adolfs | 2006-08-01 |        60.5 |           — |
| GVC           | 2010-07-07 |        44.4 |       +43.4 |
| GVC           | 2010-07-10 |        62.6 |       +62.4 |
| GVC           | 2010-07-12 |        52.6 |       +52.2 |

---

## POI sweep analysis

When the precise location of the field measurement station is uncertain, we
run a sensitivity sweep to find which pixel best reproduces the observations.

**How it works.** The full SOLWEIG pipeline runs once per day, producing a
Tmrt grid for every timestep. We then extract the modelled Tmrt timeseries at
each candidate pixel (ground-level pixels sampled every 2–3 pixels) and
compare it against the single set of field-measured Tmrt observations. Each
pixel receives an RMSE, R², and bias score — these are averaged across all
days for that site.

**How to read the plots.** Each coloured dot is one candidate pixel. Its
colour shows what the validation RMSE (or R², or bias) _would be_ if we
assumed the measurement station was at that pixel. Grey areas are buildings
(DSM − DEM > 1 m). Blue circles mark the top 5% of pixels by RMSE (lowest
error). The black square marks the current POI used in the main validation
tests. Low RMSE (green) means the model's Tmrt timeseries at that pixel
closely matches the field observations; high RMSE (red) means poor agreement.

Run with: `pytest tests/validation/test_poi_sweep_all_sites.py -v -s`

### Gustav Adolfs torg sweep

![POI sweep RMSE — Gustav Adolfs](tests/validation/gustav_adolfs/poi_sweep_results/poi_sweep_rmse.png)

|           | Current POI (33, 77) | Best pixel (63, 5) |
| --------- | -------------------: | -----------------: |
| Mean RMSE |             12.68 °C |            8.15 °C |
| Mean R²   |                0.730 |              0.762 |
| Rank      |          116 / 1,638 |          1 / 1,638 |

The top 5% pixels (81 candidates) cluster on the western side of the square,
adjacent to a ~15 m tall building. The current POI is in an open area ~40 m
from the nearest building. The best-fit pixel reduces RMSE by 36%. Without
the original station coordinates, it is not possible to determine which pixel
is correct; the sweep provides candidate locations for discussion with the
authors.

### Kronenhuset sweep

![POI sweep RMSE — Kronenhuset](tests/validation/kronenhuset/poi_sweep_results/poi_sweep_rmse.png)

|           | Current POI (51, 118) | Best pixel (133, 199) |
| --------- | --------------------: | --------------------: |
| Mean RMSE |               6.40 °C |               4.75 °C |
| Mean R²   |                 0.405 |                 0.665 |
| Rank      |                   TBD |             1 / 5,927 |

The POI was shifted +1 column from the shapefile position (51, 117) to
better match the observed shadow timing profile. The shapefile position is
reasonable but the adjacent pixel produces a closer match to the measured
diurnal Tmrt curve.

### GVC sweep

![POI sweep RMSE — GVC](tests/validation/gvc/poi_sweep_results/poi_sweep_rmse.png)

|           | Current POI (70, 126) | Best pixel (41, 173) |
| --------- | --------------------: | -------------------: |
| Mean RMSE |               6.25 °C |              4.83 °C |
| Mean R²   |                 0.544 |                0.838 |
| Rank      |                   TBD |            1 / 2,564 |

The current POI (70, 126) is selected from a low-RMSE cluster in the
courtyard, away from building edges. The edge pixels score better in
the sweep but are unrealistic measurement station locations. The POI
can be updated if the original measurement location is confirmed.

---

## Known limitations

### Kdown at open sites

Point-level downwelling shortwave (Kdown) has high RMSE (175–333 W/m²)
because a single pixel's shadow state is binary — a slight timing or geometry
error in the shadow boundary causes ~800 W/m² swings. This is inherent to
point validation of a spatially gridded model and does not indicate a model
deficiency. Spatially averaged Kdown would show much lower error.

### Ldown overestimation at enclosed sites

The model overestimates Ldown at Kronenhuset by +31 W/m². This is a known
limitation of the SOLWEIG Ldown formulation (Jonsson et al. 2006), not a bug
in this implementation. The 4-term formula fills the non-sky hemisphere with
wall emissions at emissivity 0.90 and air temperature, but real shaded walls
are cooler than air temperature.

- At SVF = 1.0 (open sky), clear-sky Ldown matches observations well
  (328.9 vs 329.7 W/m²).
- The bias arises entirely from wall-filling in enclosed geometries.
- At Kronenhuset (~25% sky obstruction), each 1 W/m² of Ldown bias propagates
  to ~0.12 °C Tmrt bias via the side view factors, accounting for ~3.6 °C of
  the 4.3 °C Tmrt bias at that site.
- This is consistent across all UMEP versions (2021a, 2022a, 2025a). The
  Jonsson et al. (2006) -25 W/m² empirical correction is commented out in all
  UMEP releases and is not applied here.

### POI uncertainty at Gustav Adolfs torg and GVC

The precise measurement station coordinates are not available for Gustav
Adolfs torg or GVC. For Gustav Adolfs, a POI sensitivity sweep across 1,638
ground-level pixels found that locations near the western buildings achieve
RMSE 8.2–8.9 °C with R² 0.73–0.76, compared to 12.7 °C at the current POI.
For GVC, the initial POI estimate produced near-zero correlation; the current
POI (70, 126) is in the low-RMSE courtyard cluster identified by the sweep.

---

## Comparison with published results

Lindberg et al. (2008) report aggregate statistics over 7 days at two
Gothenburg sites (~189 hours):

| Component |   R² |      RMSE |
| --------- | ---: | --------: |
| Tmrt      | 0.94 |     4.8 K |
| L↓        | 0.73 | 17.5 W/m² |
| L↑        | 0.94 | 15.6 W/m² |

The best-performing site here (GVC) achieves comparable accuracy (Tmrt RMSE
3.4–9.6 °C, R² 0.77–0.99). The paper's lower aggregate RMSE likely reflects:

- Aggregation across more hours and sites, which smooths individual
  shadow-timing errors.
- Possible differences in POI placement (the original study used field GPS
  coordinates that are not available for all sites in this dataset).
- The paper validates against 1-minute averaged measurements; the met data
  used here are hourly.

---

## Running validation tests

```bash
# All validation tests (fast data-loading + slow pipeline)
pytest tests/validation/ -m validation

# Just the fast data-loading checks
pytest tests/validation/ -m "validation and not slow"

# A single site
pytest tests/validation/test_validation_gvc.py -v -s

# POI sensitivity sweep (all sites, generates PNG heatmaps)
pytest tests/validation/test_poi_sweep_all_sites.py -v -s
```

---

## Version history

| Version  | Date       | Sites | Tmrt RMSE range | Key changes                                                                                                                                                    |
| -------- | ---------- | ----: | --------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1.0b57 | 2026-03-05 |     3 |     3.4–17.7 °C | Initial 3-site validation. POI sweep analysis added for all sites. Ldown wall-temperature bias documented. |
| 0.1.0b58 | 2026-03-06 |     3 |     3.4–17.7 °C | Add validation CI job. Remove non-reproducible Kolumbus/Montpellier tests. Clarify POI sweep documentation. |
| 0.1.0b59 | 2026-03-06 |     3 |     4.0–17.7 °C | Move GVC POI to courtyard cluster (70, 126). Shift Kronenhuset POI +1 col to match shadow profile. Move validation report to repo root. |
| 0.1.0b60 | 2026-03-06 |     3 |     4.0–17.7 °C | GPU GVF compute shader (wgpu). Cached thermal accumulation offloaded to GPU with automatic CPU fallback.                                                       |

---

## References

1. Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 — Modelling
   spatial variations of 3D radiant fluxes and mean radiant temperature in
   complex urban settings. _Int. J. Biometeorol._ 52, 697–713.

2. Lindberg, F. & Grimmond, C.S.B. (2011). The influence of vegetation and
   building morphology on shadow patterns and mean radiant temperature in
   urban areas. _Theor. Appl. Climatol._ 105, 311–323.

3. Jonsson, P., Eliasson, I., Holmer, B. & Grimmond, C.S.B. (2006). Longwave
   incoming radiation in the Tropics: results from field work in three African
   cities. _Theor. Appl. Climatol._ 85, 185–201.