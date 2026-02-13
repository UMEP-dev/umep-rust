# Error Handling

SOLWEIG provides structured exceptions for clear error messages and easy handling.

## Exception Hierarchy

```
SolweigError (base)
├── InvalidSurfaceData
├── GridShapeMismatch
├── MissingPrecomputedData
├── WeatherDataError
└── ConfigurationError
```

## Catching Errors

```python
import solweig
from solweig.errors import GridShapeMismatch, MissingPrecomputedData, SolweigError

try:
    result = solweig.calculate(surface, location, weather)
except GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field}")
    print(f"  Expected: {e.expected}")
    print(f"  Got: {e.got}")
except MissingPrecomputedData as e:
    print(f"Missing data: {e}")
    print(f"  Hint: {e.hint}")
except SolweigError as e:
    # Catch any SOLWEIG error
    print(f"Error: {e}")
```

---

## SolweigError

::: solweig.errors.SolweigError
    options:
      show_source: false
      heading_level: 3

---

## GridShapeMismatch

::: solweig.errors.GridShapeMismatch
    options:
      show_source: false
      heading_level: 3

---

## MissingPrecomputedData

::: solweig.errors.MissingPrecomputedData
    options:
      show_source: false
      heading_level: 3

---

## InvalidSurfaceData

::: solweig.errors.InvalidSurfaceData
    options:
      show_source: false
      heading_level: 3

---

## WeatherDataError

::: solweig.errors.WeatherDataError
    options:
      show_source: false
      heading_level: 3

---

## ConfigurationError

::: solweig.errors.ConfigurationError
    options:
      show_source: false
      heading_level: 3

---

## Pre-flight Validation

Use `validate_inputs()` to catch errors before expensive computations:

```python
from solweig.errors import GridShapeMismatch, MissingPrecomputedData

try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")

    # Now safe to run expensive calculation
    result = solweig.calculate(surface, location, weather)

except GridShapeMismatch as e:
    print(f"Fix grid shapes before proceeding: {e.field}")
except MissingPrecomputedData as e:
    print(f"Missing required data: {e}")
```

This catches shape mismatches, missing data, and other issues *before* SVF computation.
