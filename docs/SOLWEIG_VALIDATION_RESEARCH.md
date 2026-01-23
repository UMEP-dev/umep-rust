# SOLWEIG Validation Research Summary

*Research conducted: January 2026*

## Overview

This document summarizes research into the academic background of the SOLWEIG model, its validation methodology, and available datasets for replicating first-principles validation.

---

## 1. Academic Background

### Original Paper

**Lindberg, F., Holmer, B. & Thorsson, S. (2008)**
"SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings"
*International Journal of Biometeorology* 52, 697–713
DOI: [10.1007/s00484-008-0162-7](https://doi.org/10.1007/s00484-008-0162-7)

### Key Subsequent Papers

| Year | Authors | Focus | Journal |
|------|---------|-------|---------|
| 2011 | Lindberg & Grimmond | Vegetation scheme | Theoretical and Applied Climatology |
| 2016 | Lindberg, Onomura & Grimmond | Ground surface characteristics | Int. J. Biometeorology |
| 2025 | Lindberg et al. | Wall surface temperature scheme | EGUsphere (preprint) |

### Research Group

**Göteborg Urban Climate Group (GUCG)**
Department of Earth Sciences, University of Gothenburg, Sweden
Website: https://www.gu.se/en/research/gucg

---

## 2. Original Validation Methodology (2008)

### Study Design

| Aspect | Details |
|--------|---------|
| **Location** | Göteborg, Sweden (57°N) |
| **Sites** | Large open square + small courtyard |
| **Duration** | 7 days across multiple seasons |
| **Periods** | October 2005, July/August 2006 |
| **Conditions** | Clear to overcast weather |

### Measurement Method

The **six-directional integral radiation method** (ISO 7726 standard):

- Measures shortwave and longwave radiation from 6 directions (up, down, N, S, E, W)
- Angular weighting factors: **0.22** for cardinal directions, **0.06** for up/down
- Instruments positioned at **1.1m height** (center of gravity for standing person)
- Requires pyranometers (shortwave) and pyrgeometers (longwave)

### Validation Results

| Metric | Value |
|--------|-------|
| R² | 0.94 |
| RMSE | 4.8 K |
| p-value | < 0.01 |

### Comparative Performance

Studies comparing SOLWEIG against other models (RayMan, ENVI-met) consistently show SOLWEIG performs best for Tmrt estimation:

- Hong Kong study (670 sites): SOLWEIG showed best correlation with six-directional measurements
- Cold region study: SOLWEIG exhibited better determination performance than RayMan and ENVI-met

---

## 3. Available Public Datasets

### 3.1 UMEP Göteborg Tutorial Dataset

**Source:** [GitHub - Urban Meteorology Reading](https://github.com/Urban-Meteorology-Reading/Urban-Meteorology-Reading.github.io/tree/master/other%20files/Goteborg_SWEREF99_1200.zip)

| Contents | Format |
|----------|--------|
| DSM (Digital Surface Model) | GeoTIFF |
| CDSM (Canopy DSM) | GeoTIFF |
| DEM (Digital Elevation Model) | GeoTIFF |
| Land cover / ground cover | GeoTIFF |
| Study area boundary | Shapefile |

**Coordinate System:** SWEREF99 1200 (EPSG:3007)

**Use Case:** Running SOLWEIG simulations (model inputs only, no validation measurements)

---

### 3.2 Swedish National Data Service - Gothenburg Climate Data

**Source:** [researchdata.se](https://researchdata.se/en/catalogue/dataset/2021-253-1)
**DOI:** 10.5878/a2h2-4s63

| Variable | Unit |
|----------|------|
| Air temperature | °C |
| Wind speed (average) | m/s |
| Wind direction | degrees |
| Relative humidity | % |
| Global radiation | W/m² |
| Diffuse radiation | W/m² |
| Direct-beam radiation | W/m² |
| Mean sea-level pressure | hPa |

**Period:** September 1986 – December 2020
**Resolution:** Hourly
**Format:** CSV, NetCDF

**Use Case:** Meteorological forcing data for SOLWEIG runs

**Note:** Does NOT include Tmrt or six-directional radiation measurements

---

### 3.3 Zenodo SOLWEIG v2025 Validation Dataset

**Source:** [Zenodo Record 15309445](https://zenodo.org/records/15309445)

#### Files Available

| File | Size | Contents |
|------|------|----------|
| `geodata.zip` | 86.7 kB | Urban geometry for validation site |
| `kolumbus.csv` | 1.2 MB | **Wall surface temperature validation data** |
| `metdata_10min_may.txt` | ~700 kB | Meteorological forcing |
| `metdata_10min_june.txt` | ~668 kB | Meteorological forcing |
| `metdata_10min_july.txt` | ~743 kB | Meteorological forcing |
| `metdata_10min_august.txt` | ~743 kB | Meteorological forcing |

#### kolumbus.csv Details

| Aspect | Details |
|--------|---------|
| **Variable** | Wall surface temperature (Ts) |
| **Period** | 2023-05-15 to 2023-08-31 |
| **Resolution** | 10-minute intervals |
| **Observations** | ~15,400 measurements |
| **Surfaces** | Wooden wall + plastered brick wall (albedo ≈ 0.5) |
| **Instrument** | Apogee SI-111 infrared radiometer at 10cm from wall |
| **Reported accuracy** | R² = 0.93-0.94, RMSE = 1.94-2.09°C |

**Use Case:** Validation of wall surface temperature calculation (intermediate variable in SOLWEIG)

---

### 3.4 Datasets NOT Publicly Available

| Dataset | Status | How to Obtain |
|---------|--------|---------------|
| Original 2008 Göteborg Tmrt measurements | Not archived | Contact authors |
| Six-directional radiation data (2005-2006) | Not archived | Contact authors |
| Hong Kong 670-site validation data | On request | Contact paper authors |
| Singapore thermal comfort data | On request | Singapore-ETH Centre |

---

## 4. Gap Analysis

### What First-Principles Validation Requires

1. **Urban geometry** (DSMs, land cover) ✅ Available
2. **Meteorological forcing** (radiation, Ta, RH) ✅ Available
3. **Ground-truth Tmrt measurements** ❌ Not publicly available

### Current Test Strategy (This Repository)

| Layer | Purpose | Data Source |
|-------|---------|-------------|
| Spec property tests | Physical invariants | Synthetic data |
| Golden regression tests | Numerical drift detection | UMEP reference outputs |
| Parity tests | API vs runner match | UMEP implementation |

**Target:** Tmrt bias < 0.5°C against reference implementation

This validates **implementation correctness** but not **physical accuracy** against real-world observations.

---

## 5. Recommendations

### Short-term (No external data needed)

1. **Physics-based unit tests**
   - Verify Tmrt formula: `Tmrt⁴ = (1/σ) × Σ[αk·Ki·Fi + αL·Li·Fi]`
   - Verify angular weighting factors (0.22/0.06)
   - Verify clear-sky radiation (I₀) against astronomical calculations

2. **UTCI reference validation**
   - Compare against [Bröde et al. reference implementation](https://utci.org/)

3. **Wall Ts validation**
   - Download kolumbus.csv from Zenodo
   - Run SOLWEIG with 2023 Gothenburg data
   - Compare wall temperatures (target: R² > 0.9, RMSE < 2.5°C)

### Medium-term (Requires author contact)

4. **Request original validation data**
   - Contact Fredrik Lindberg (University of Gothenburg)
   - Request 2005-2006 six-directional radiation measurements
   - Academic researchers often share data for reproducibility

### Long-term (If resources available)

5. **Conduct independent validation campaign**
   - Six-directional radiation measurements at known location
   - Paired with high-resolution DSM from LiDAR
   - Would provide fully independent validation

---

## 6. References

### Primary SOLWEIG Papers

1. Lindberg, F., Holmer, B. & Thorsson, S. (2008). SOLWEIG 1.0 – Modelling spatial variations of 3D radiant fluxes and mean radiant temperature in complex urban settings. *Int. J. Biometeorol.* 52, 697–713.

2. Lindberg, F. & Grimmond, C.S.B. (2011). The influence of vegetation and building morphology on shadow patterns and mean radiant temperatures in urban areas. *Theor. Appl. Climatol.* 105, 311–323.

3. Lindberg, F., Onomura, S. & Grimmond, C.S.B. (2016). Influence of ground surface characteristics on the mean radiant temperature in urban areas. *Int. J. Biometeorol.* 60, 1439–1452.

### Validation Methodology

4. Thorsson, S., Lindberg, F., Eliasson, I. & Holmer, B. (2007). Different methods for estimating the mean radiant temperature in an outdoor urban setting. *Int. J. Climatol.* 27, 1983–1993.

### Comparative Studies

5. Chen, L. et al. (2024). Estimation of mean radiant temperature across diverse outdoor spaces: A comparative study of different modeling approaches. *Energy and Buildings* 308, 113999.

### Standards

6. ISO 7726:1998. Ergonomics of the thermal environment — Instruments for measuring physical quantities.

---

## 7. Contact Information

**For original validation data:**

- Fredrik Lindberg - Department of Earth Sciences, University of Gothenburg
- Sofia Thorsson - Department of Earth Sciences, University of Gothenburg

**UMEP/SOLWEIG resources:**

- Documentation: https://umep-docs.readthedocs.io/
- GitHub: https://github.com/UMEP-dev/UMEP
- Zenodo (v2025): https://zenodo.org/records/15309384
