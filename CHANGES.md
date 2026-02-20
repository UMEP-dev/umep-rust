# Algorithm Changes and Observations

This document tracks algorithm-related changes, differences, and observations discovered during the SOLWEIG modernization project. It is intended for discussion with the original authors.

## Purpose

During the Rust modernization of SOLWEIG, we are:

1. Creating golden fixtures using the **UMEP Python module** as ground truth
2. Verifying that Rust implementations match UMEP Python outputs
3. Documenting any discrepancies or algorithmic questions

## Testing Strategy

### Three-Layer Testing Approach

| Layer             | Purpose                         | Data Source          |
| ----------------- | ------------------------------- | -------------------- |
| Spec Tests        | Verify physical properties      | Synthetic/mock data  |
| Golden Tests      | Verify Rust matches UMEP Python | Athens demo data     |
| Performance Tests | Benchmark Rust vs Python        | Large tiled datasets |

### Golden Fixtures

Golden fixtures are generated using **UMEP Python** functions, not Rust:

- `shadowingfunction_wallheight_23` for shadow calculations
- `svfForProcessing153` for SVF calculations
- `gvf_2018a` for Ground View Factor
- `Kside_veg_v2022a` / `Lside_veg_v2022a` for radiation

This ensures a neutral reference that doesn't change during modernization.

---

## Observed Differences

### 1. Shadow Calculation (shadowingfunction_wallheight_23)

**Status:** VERIFIED - Rust matches UMEP Python exactly

**UMEP Python function:** `umep.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23`

**Rust function:** `solweig.rustalgos.shadowing.calculate_shadows_wall_ht_25`

**Test Results:**
All shadow components match within tolerance 1e-5:

- `bldg_sh` (building shadows) - PASS
- `veg_sh` (vegetation shadows) - PASS
- `wall_sh` (wall shadows) - PASS
- `wall_sun` (wall sun exposure) - PASS

**Findings:**

- The Rust `_25` is a direct port of Python `_23` - the version increment was for internal tracking
- No algorithmic changes were made during the Rust modernization
- Both implementations produce identical results

**Conclusion:** No action needed - implementations are equivalent.

---

### 2. Sky View Factor (svfForProcessing153)

**Status:** INTENTIONAL DIFFERENCE - documented in earlier modernization

**UMEP Python function:** `umep.functions.svf_functions.svfForProcessing153`

**Rust function:** `solweig.rustalgos.skyview.calculate_svf`

**Test Results (Golden Test Comparison):**

| Component | Match Status | Notes                               |
| --------- | ------------ | ----------------------------------- |
| svf_total | ~1% diff     | Different underlying shadow         |
| svf_north | ~1% diff     | Different underlying shadow         |
| svf_east  | ~1% diff     | Different underlying shadow         |
| svf_south | EXACT        | matches within 1e-5                 |
| svf_west  | EXACT        | matches within 1e-5                 |
| svf_veg   | ~1% diff     | Different underlying shadow         |

**Root Cause (from `test_rustalgos.py`):**

This difference was **intentional and documented** during the earlier modernization:

```python
# Line 201: "# uses older shadowingfunction_20"
# Line 205-206: "# uses rust shadowing based on shadowingfunction_wallheight_23"
# Line 282-283: print("Small differences expected for N and E and totals
#                      due to different shadowing implementations")
```

The UMEP Python `svfForProcessing153` internally calls the older `shadowingfunction_20`, while Rust uses the newer `shadowingfunction_wallheight_23` throughout for architectural consistency.

**Verification:**

A hybrid implementation (`svfForProcessing153_rust_shdw`) exists that uses Python SVF logic with Rust shadows. This hybrid matches the full Rust implementation exactly, proving:

1. The SVF algorithm itself is correctly ported
2. The ~1% difference comes solely from using different shadow algorithms

**Decision:** ACCEPTED

The ~1% difference is accepted. Rust uses the newer `shadowingfunction_wallheight_23` throughout, which is architecturally cleaner and more consistent. The older `shadowingfunction_20` used by Python SVF is legacy code.

Golden tests use 2% tolerance for affected components (total, north, east, veg) and strict 1e-5 tolerance for unaffected components (south, west).

---

### 3. Ground View Factor (gvf_2018a)

**Status:** Not yet tested

**UMEP Python function:** `umep.functions.SOLWEIGpython.gvf_2018a.gvf_2018a`

**Rust function:** `solweig.rustalgos.gvf.gvf_calc`

**Output fields:**

- `gvfSum`, `gvfNorm`
- `gvfLup`, `gvfLupE/S/W/N`
- `gvfalb`, `gvfalbE/S/W/N`
- `gvfalbnosh`, `gvfalbnoshE/S/W/N`

---

### 4. Radiation Calculations

**Status:** Not yet tested

#### Kside (Shortwave Side Radiation)

- **Python:** `Kside_veg_v2022a`
- **Rust:** `vegetation.kside_veg`

#### Lside (Longwave Side Radiation)

- **Python:** `Lside_veg_v2022a`
- **Rust:** `vegetation.lside_veg`

---

### 5. Thermal Comfort Indices

**Status:** Spec tests created, golden fixtures pending

#### UTCI (Universal Thermal Climate Index)

- **Rust:** `solweig.rustalgos.utci`
- Spec tests verify property-based behavior

#### PET (Physiological Equivalent Temperature)

- **Rust:** `solweig.rustalgos.pet`
- Spec tests verify property-based behavior

---

## Bug Fixes Applied

_Document any bug fixes discovered and applied during testing._

### Example Template

```
### [Date] Bug Title

**Location:** file:line
**Symptom:** Description of incorrect behavior
**Root Cause:** Why it was happening
**Fix:** What was changed
**Impact:** How significant was this bug
```

---

## Numerical Precision Notes

### Tolerance Levels Used

| Test Type        | rtol | atol | Rationale                            |
| ---------------- | ---- | ---- | ------------------------------------ |
| Shadow masks     | 1e-5 | 1e-5 | Binary-like values (0/1)             |
| SVF values       | 1e-5 | 1e-5 | Range [0, 1]                         |
| Radiation (W/m²) | 1e-4 | 0.01 | Physical units, ~1% error acceptable |
| Temperature (°C) | 1e-4 | 0.01 | Physical units                       |

### Known Precision Issues

_Document any known floating-point precision issues._

---

## Version Information

- **UMEP Python version:** (check with `pip show umep`)
- **SOLWEIG Rust version:** See Cargo.toml
- **Test data:** Athens demo dataset

---

## Discussion Log

_Record discussions with original authors here._

### [Date] Discussion Topic

- Participants:
- Decision:
- Action items:
