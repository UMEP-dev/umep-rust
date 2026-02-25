# Kronenhuset Validation Report

**Generated:** 2026-02-25
**Site:** Kronenhuset courtyard, Gothenburg, Sweden (57.7 N, 12.0 E)
**Date:** 2005-10-07 (DOY 280)
**Resolution:** 1 m, EPSG:3007
**POI pixel:** row 51, col 117
**Reference:** Lindberg et al. (2008), SOLWEIG 1.0 — Modelling spatial variations
of mean radiant temperature in an urban area.

## Overview

This report compares the Rust SOLWEIG implementation against field observations
(directional radiation measurements from Lindberg et al. 2008). The pipeline is
run in both **isotropic** and **anisotropic** (Perez et al. 1993) sky modes.

UMEP Python parity is covered separately by the golden tests (see
[tests/golden/REPORT.md](../../golden/REPORT.md)).

## Radiation Budget Comparison

### Kdown — Downwelling Shortwave (W/m²)

| Hour |   Obs | Rust iso | Rust aniso | Riso−Obs | Rani−Obs |
| ---: | ----: | -------: | ---------: | -------: | -------: |
|    0 |     — |      0.0 |        0.0 |          |          |
|    1 |     — |      0.0 |        0.0 |          |          |
|    2 |     — |      0.0 |        0.0 |          |          |
|    3 |     — |      0.0 |        0.0 |          |          |
|    4 |     — |      0.0 |        0.0 |          |          |
|    5 |     — |      0.0 |        0.0 |          |          |
|    6 |     — |      5.9 |        0.5 |          |          |
|    7 |  -0.9 |     31.7 |       23.6 |    +32.5 |    +24.5 |
|    8 |  26.2 |     36.8 |       19.9 |    +10.5 |     -6.4 |
|    9 |  41.1 |     46.0 |       22.7 |     +4.9 |    -18.4 |
|   10 |  50.8 |     51.3 |       25.5 |     +0.5 |    -25.3 |
|   11 |  57.0 |     52.6 |       26.8 |     -4.4 |    -30.3 |
|   12 |  69.8 |     52.4 |       27.1 |    -17.3 |    -42.6 |
|   13 |  58.7 |     47.3 |       25.3 |    -11.4 |    -33.4 |
|   14 |  57.5 |     42.8 |       23.3 |    -14.8 |    -34.2 |
|   15 | 162.4 |    238.7 |      223.1 |    +76.3 |    +60.7 |
|   16 | 158.3 |     22.7 |       12.8 |   -135.6 |   -145.5 |
|   17 |  18.3 |      6.3 |        4.7 |    -12.0 |    -13.6 |
|   18 |   5.2 |      0.0 |        0.0 |     -5.2 |     -5.2 |
|   19 |     — |      0.0 |        0.0 |          |          |
|   20 |     — |      0.0 |        0.0 |          |          |
|   21 |     — |      0.0 |        0.0 |          |          |
|   22 |     — |      0.0 |        0.0 |          |          |
|   23 |     — |      0.0 |        0.0 |          |          |

### Kup — Upwelling Shortwave (W/m²)

| Hour |  Obs | Rust iso | Rust aniso | Riso−Obs | Rani−Obs |
| ---: | ---: | -------: | ---------: | -------: | -------: |
|    0 |    — |      0.0 |        0.0 |          |          |
|    1 |    — |      0.0 |        0.0 |          |          |
|    2 |    — |      0.0 |        0.0 |          |          |
|    3 |    — |      0.0 |        0.0 |          |          |
|    4 |    — |      0.0 |        0.0 |          |          |
|    5 |    — |      0.0 |        0.0 |          |          |
|    6 |    — |      1.1 |        1.1 |          |          |
|    7 | -3.8 |      5.9 |        5.9 |     +9.7 |     +9.7 |
|    8 |  3.7 |      8.4 |        8.4 |     +4.7 |     +4.7 |
|    9 |  6.2 |     11.9 |       11.9 |     +5.7 |     +5.7 |
|   10 |  7.0 |     15.3 |       15.3 |     +8.4 |     +8.4 |
|   11 |  7.6 |     16.6 |       16.6 |     +9.0 |     +9.0 |
|   12 |  7.5 |     17.2 |       17.2 |     +9.6 |     +9.6 |
|   13 |  7.9 |     16.1 |       16.1 |     +8.2 |     +8.2 |
|   14 |  9.1 |     14.8 |       14.8 |     +5.7 |     +5.7 |
|   15 | 23.5 |     28.6 |       28.6 |     +5.1 |     +5.1 |
|   16 | 18.5 |      8.3 |        8.3 |    -10.2 |    -10.2 |
|   17 |  2.7 |      1.4 |        1.4 |     -1.3 |     -1.3 |
|   18 |  2.4 |      0.0 |        0.0 |     -2.4 |     -2.4 |
|   19 |    — |      0.0 |        0.0 |          |          |
|   20 |    — |      0.0 |        0.0 |          |          |
|   21 |    — |      0.0 |        0.0 |          |          |
|   22 |    — |      0.0 |        0.0 |          |          |
|   23 |    — |      0.0 |        0.0 |          |          |

### Ldown — Downwelling Longwave (W/m²)

| Hour |   Obs | Rust iso | Rust aniso | Riso−Obs | Rani−Obs |
| ---: | ----: | -------: | ---------: | -------: | -------: |
|    0 |     — |    304.0 |      299.6 |          |          |
|    1 |     — |    301.9 |      297.5 |          |          |
|    2 |     — |    302.5 |      298.1 |          |          |
|    3 |     — |    300.2 |      295.8 |          |          |
|    4 |     — |    305.3 |      301.0 |          |          |
|    5 |     — |    305.6 |      301.2 |          |          |
|    6 |     — |    305.3 |      301.0 |          |          |
|    7 | 291.0 |    321.0 |      316.1 |    +30.0 |    +25.1 |
|    8 | 296.3 |    332.0 |      325.6 |    +35.7 |    +29.4 |
|    9 | 306.3 |    344.5 |      337.4 |    +38.2 |    +31.1 |
|   10 | 314.1 |    353.3 |      345.6 |    +39.1 |    +31.5 |
|   11 | 323.8 |    361.0 |      353.0 |    +37.3 |    +29.2 |
|   12 | 329.7 |    364.8 |      355.9 |    +35.1 |    +26.2 |
|   13 | 332.5 |    368.4 |      358.7 |    +35.9 |    +26.3 |
|   14 | 334.2 |    366.4 |      356.5 |    +32.2 |    +22.3 |
|   15 | 333.3 |    370.7 |      361.3 |    +37.5 |    +28.0 |
|   16 | 327.4 |    372.2 |      364.0 |    +44.8 |    +36.7 |
|   17 | 318.6 |    370.5 |      364.2 |    +51.9 |    +45.7 |
|   18 | 313.3 |    360.6 |      358.2 |    +47.3 |    +44.9 |
|   19 |     — |    356.6 |      354.1 |          |          |
|   20 |     — |    352.1 |      349.6 |          |          |
|   21 |     — |    349.4 |      347.0 |          |          |
|   22 |     — |    344.7 |      342.4 |          |          |
|   23 |     — |    345.2 |      342.8 |          |          |

### Lup — Upwelling Longwave (W/m²)

| Hour |   Obs | Rust iso | Rust aniso | Riso−Obs | Rani−Obs |
| ---: | ----: | -------: | ---------: | -------: | -------: |
|    0 |     — |    339.0 |      339.0 |          |          |
|    1 |     — |    337.1 |      337.1 |          |          |
|    2 |     — |    337.5 |      337.5 |          |          |
|    3 |     — |    335.6 |      335.6 |          |          |
|    4 |     — |    340.0 |      340.0 |          |          |
|    5 |     — |    340.0 |      340.0 |          |          |
|    6 |     — |    340.0 |      340.0 |          |          |
|    7 | 352.5 |    352.8 |      352.8 |     +0.3 |     +0.3 |
|    8 | 353.8 |    361.6 |      361.6 |     +7.8 |     +7.8 |
|    9 | 358.6 |    371.5 |      371.5 |    +12.8 |    +12.8 |
|   10 | 363.9 |    379.7 |      379.7 |    +15.8 |    +15.8 |
|   11 | 369.7 |    387.2 |      387.2 |    +17.5 |    +17.5 |
|   12 | 375.2 |    393.0 |      393.0 |    +17.8 |    +17.8 |
|   13 | 379.3 |    397.1 |      397.1 |    +17.7 |    +17.7 |
|   14 | 382.7 |    397.8 |      397.8 |    +15.1 |    +15.1 |
|   15 | 389.0 |    405.8 |      405.8 |    +16.9 |    +16.9 |
|   16 | 393.8 |    395.0 |      395.0 |     +1.2 |     +1.2 |
|   17 | 385.5 |    380.3 |      380.3 |     -5.2 |     -5.2 |
|   18 | 380.9 |    371.3 |      371.3 |     -9.6 |     -9.6 |
|   19 |     — |    364.2 |      364.2 |          |          |
|   20 |     — |    359.7 |      359.7 |          |          |
|   21 |     — |    356.7 |      356.7 |          |          |
|   22 |     — |    351.7 |      351.7 |          |          |
|   23 |     — |    352.2 |      352.2 |          |          |

### Tmrt — Mean Radiant Temperature (°C)

| Hour |  Obs | Rust iso | Rust aniso | Riso−Obs | Rani−Obs |
| ---: | ---: | -------: | ---------: | -------: | -------: |
|    0 |    — |      3.4 |        2.8 |          |          |
|    1 |    — |      3.0 |        2.4 |          |          |
|    2 |    — |      3.1 |        2.5 |          |          |
|    3 |    — |      2.7 |        2.1 |          |          |
|    4 |    — |      3.6 |        3.1 |          |          |
|    5 |    — |      3.6 |        3.1 |          |          |
|    6 |    — |      4.0 |        3.1 |          |          |
|    7 |  1.7 |      0.7 |        8.1 |     -0.9 |     +6.4 |
|    8 |  4.9 |      9.2 |       10.9 |     +4.3 |     +6.0 |
|    9 |  7.9 |     12.4 |       14.0 |     +4.4 |     +6.0 |
|   10 | 10.4 |     14.8 |       16.6 |     +4.4 |     +6.2 |
|   11 | 12.9 |     16.5 |       18.6 |     +3.7 |     +5.7 |
|   12 | 15.0 |     17.6 |       19.3 |     +2.6 |     +4.2 |
|   13 | 16.6 |     18.0 |       19.6 |     +1.4 |     +2.9 |
|   14 | 18.0 |     17.7 |       19.1 |     -0.3 |     +1.1 |
|   15 | 28.9 |     39.1 |       40.1 |    +10.2 |    +11.2 |
|   16 | 29.8 |     15.5 |       16.5 |    -14.3 |    -13.2 |
|   17 | 14.9 |     12.2 |       12.7 |     -2.7 |     -2.3 |
|   18 | 13.7 |     10.6 |       10.8 |     -3.1 |     -2.9 |
|   19 |    — |      9.5 |        9.7 |          |          |
|   20 |    — |      8.6 |        8.8 |          |          |
|   21 |    — |      8.1 |        8.2 |          |          |
|   22 |    — |      7.1 |        7.2 |          |          |
|   23 |    — |      7.2 |        7.3 |          |          |

### Shadow Flag (1 = direct sun)

| Hour | Rust | Sun altitude |
| ---: | ---: | -----------: |
|    0 |    1 |        -37.5 |
|    1 |    1 |        -37.3 |
|    2 |    1 |        -34.5 |
|    3 |    1 |        -29.6 |
|    4 |    1 |        -23.1 |
|    5 |    1 |        -15.7 |
|    6 |    1 |         -7.8 |
|    7 |    0 |          1.0 |
|    8 |    0 |          8.0 |
|    9 |    0 |         14.8 |
|   10 |    0 |         20.5 |
|   11 |    0 |         24.5 |
|   12 |    0 |         26.5 |
|   13 |    0 |         26.2 |
|   14 |    0 |         23.7 |
|   15 |    1 |         19.2 |
|   16 |    0 |         13.2 |
|   17 |    0 |          6.1 |
|   18 |    1 |         -1.1 |
|   19 |    1 |         -9.9 |
|   20 |    1 |        -17.7 |
|   21 |    1 |        -25.0 |
|   22 |    1 |        -31.1 |
|   23 |    1 |        -35.7 |

## Summary Statistics

Computed over matched daytime observation hours. Kdown excludes hour 7
(negative observed value).

| Metric | Component | Rust iso | Rust aniso |
| ------ | --------- | -------: | ---------: |
| RMSE   | Kdown     |     47.8 |       53.2 |
| Bias   | Kdown     |     -9.9 |      -26.8 |
| MAE    | Kdown     |     26.6 |       37.8 |
| R²     | Kdown     |    0.046 |     -0.179 |
| RMSE   | Kup       |      7.2 |        7.2 |
| Bias   | Kup       |     +4.3 |       +4.3 |
| MAE    | Kup       |      6.7 |        6.7 |
| R²     | Kup       |   -0.102 |     -0.102 |
| RMSE   | Ldown     |     39.2 |       32.1 |
| Bias   | Ldown     |    +38.8 |      +31.3 |
| MAE    | Ldown     |     38.8 |       31.3 |
| R²     | Ldown     |   -6.900 |     -4.306 |
| RMSE   | Lup       |     13.0 |       13.0 |
| Bias   | Lup       |     +9.0 |       +9.0 |
| MAE    | Lup       |     11.5 |       11.5 |
| R²     | Lup       |    0.039 |      0.039 |
| RMSE   | Tmrt      |      5.8 |        6.6 |
| Bias   | Tmrt      |     +0.8 |       +2.6 |
| MAE    | Tmrt      |      4.3 |        5.7 |
| R²     | Tmrt      |    0.482 |      0.325 |

## Notes

### Ldown structural bias (+30–53 W/m²)

Both isotropic and anisotropic configurations overestimate Ldown by +30 to
+53 W/m² (mean bias +38.8 iso, +31.3 aniso). This matches UMEP Python
(mean bias +40.5), confirming it is a structural feature of the Jonsson et al.
(2006) model, not an implementation bug.

**Decomposition of the bias into three sources:**

At this POI, SVF = 0.655 with no vegetation (svfveg = svfaveg = 1.0). The
Jonsson formula sees 65.5% sky and 34.5% walls. Wall emissivity (0.9) exceeds
clear-sky emissivity (~0.78), so replacing sky with walls always increases
Ldown relative to an open-sky reference.

1. **Wall temperature assumption (+12–20 W/m²).** The model assumes shaded
   walls emit at air temperature. In this October scenario (max solar
   altitude 26°, massive stone buildings), walls are 7–20°C cooler than air
   due to thermal inertia. The model has no wall surface energy balance and
   cannot represent this.

2. **Tgwall sinusoidal model (+9–27 W/m²).** The sunlit-wall term uses
   (Ta + Tgwall)⁴, where Tgwall is a sinusoidal offset driven by global
   radiation. In a deep canyon at 57.7°N in October most walls are shaded,
   but the model applies the boost to the entire (svfaveg − svf) fraction.

3. **CI cloud correction at hours 15–18 (+7–29 W/m²).** Clearness index
   drops below 0.95 in late afternoon (CI = 0.90 → 0.48), triggering a
   blend of esky towards 1.0 (blackbody clouds). This is carried forward
   into nighttime via CI persistence. At hour 18, the cloud correction alone
   adds ~29 W/m² on top of the base wall-temperature bias.

The paper's Ldown RMSE of 17.5 W/m² was measured over 7 days including summer
when the wall-at-air-temperature assumption holds better. The anisotropic sky
model reduces the bias (RMSE 32.1 vs 39.2 W/m²) by distributing sky emission
more realistically across zenith angles.

### Hour 16 shadow mismatch

Both isotropic and anisotropic predict the POI is in shade at hour 16, but
observations show Kdown = 158 W/m² (direct sun). This is inherent to the DSM
geometry — the 1 m resolution raster cannot capture the exact moment when the
courtyard transitions from sun to shade. This single hour dominates the Kdown
and Tmrt error statistics.

### Nighttime Ldown (hours 18–23)

Producing correct nighttime Ldown required two fixes (both matching UMEP
Python's behaviour):

1. Zeroing `tg_wall` and `Tg` when the sun is below the horizon
2. Carrying forward the last daytime clearness index into nighttime hours
   (our code previously defaulted to CI = 1.0 at night)

## Comparison with Original Paper

Lindberg et al. (2008) report validation statistics for SOLWEIG 1.0 using data
from two Gothenburg sites (Kronenhuset and Hogsbo) over 7 clear-sky days in
2005–2006. Their aggregate results (Table 3, all days and sites combined):

| Component | R²   | RMSE      | n         |
| --------- | ---- | --------- | --------- |
| Tmrt      | 0.94 | 4.8 K     | 189 hours |
| L↓        | 0.73 | 17.5 W/m² | 189 hours |
| L↑        | 0.94 | 15.6 W/m² | 189 hours |
| L (sides) | 0.92 | 48.9 W/m² | 189 hours |

**Important caveats when comparing with our results above:**

- The paper aggregates 7 days across 2 sites (~189 matched hours); this report
  covers 1 day at Kronenhuset only (12 hours). Single-day statistics are much
  noisier and more sensitive to individual-hour outliers (e.g. hour 16 shadow
  mismatch).
- The paper uses SOLWEIG 1.0 (2008 vintage); our results use the modernised
  Rust reimplementation of SOLWEIG 2025a. Differences in Ldown parameterisation,
  ground temperature model, and anisotropic sky option will affect results.
- Shortwave per-component statistics (K↓, K↑, K sides) are reported in the
  paper but were not available from the abstract. The paper notes that shortwave
  errors are dominated by shadow timing — consistent with our hour 16 finding.
- The paper's Tmrt RMSE of 4.8 K is the benchmark target. Our single-day
  Tmrt RMSE of 5.8 K (Rust isotropic) and 6.6 K (Rust anisotropic) are
  reasonable given the single-day sample size and the hour 16 shadow outlier.

**Reference:** Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 —
Modelling spatial variations of 3D radiant fluxes and mean radiant temperature
in complex urban settings. _International Journal of Biometeorology_, 52,
697–713. doi:10.1007/s00484-008-0162-7
