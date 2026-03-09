# Kronenhuset Validation Report

**Generated:** 2026-03-09
**Site:** Kronenhuset courtyard, Gothenburg, Sweden (57.7 N, 12.0 E)
**Date:** 2005-10-07 (DOY 280)
**Resolution:** 1 m, EPSG:3007
**POI:** row 51, col 117 — from `POI_KR.shp` via `poi.geojson`
**Reference:** Lindberg et al. (2008), SOLWEIG 1.0 — Modelling spatial variations
of mean radiant temperature in an urban area.

## Overview

This report compares the Rust SOLWEIG implementation against field observations
(directional radiation measurements from Lindberg et al. 2008). The pipeline is
run in both **isotropic** and **anisotropic** (Perez et al. 1993) sky modes.

## Notes

- **Ldown bias (+39 W/m²):** Both modes overestimate Ldown (mean bias
  +45.2 iso, +38.5 aniso). The anisotropic sky model reduces the bias
  by ~7 W/m².

- **Hour 16 shadow mismatch:** The model predicts shade at hour 16 while
  observations show direct sun (Kdown = 158 W/m²). At 1 m DSM resolution
  the exact sun-to-shade transition time cannot be resolved precisely. This
  hour dominates the Kdown and Tmrt error statistics.

## Summary Statistics

Computed over 12 matched daytime observation hours.

| Metric | Component | Rust iso | Rust aniso |
| ------ | --------- | -------: | ---------: |
| RMSE   | Kdown     |     47.1 |       51.0 |
| Bias   | Kdown     |    -10.9 |      -22.3 |
| RMSE   | Kup       |      6.6 |        6.6 |
| Bias   | Kup       |     +3.5 |       +3.5 |
| RMSE   | Ldown     |     45.4 |       39.0 |
| Bias   | Ldown     |    +45.2 |      +38.5 |
| RMSE   | Lup       |     13.1 |       13.1 |
| Bias   | Lup       |     +9.0 |       +9.0 |
| RMSE   | Tmrt      |      6.4 |        6.0 |
| Bias   | Tmrt      |     +2.7 |       +1.3 |

## Comparison with Original Paper

Lindberg et al. (2008) report aggregate statistics over 7 days at two
Gothenburg sites (~189 hours):

| Component | R²   | RMSE      |
| --------- | ---- | --------- |
| Tmrt      | 0.94 | 4.8 K     |
| L↓        | 0.73 | 17.5 W/m² |
| L↑        | 0.94 | 15.6 W/m² |

This report covers a single day (12 hours), so statistics are noisier. Our
Tmrt RMSE of 6.4 K (iso) / 6.0 K (aniso) is in the same range as the paper's
4.8 K, and shortwave errors are similarly dominated by shadow timing.

**Reference:** Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 —
Modelling spatial variations of 3D radiant fluxes and mean radiant temperature
in complex urban settings. _International Journal of Biometeorology_, 52,
697–713. doi:10.1007/s00484-008-0162-7
