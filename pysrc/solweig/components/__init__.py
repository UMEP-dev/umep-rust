"""
SOLWEIG computation components.

This package contains modular computation functions extracted from the
monolithic `_calculate_core()` function. Each component is responsible
for a specific part of the Tmrt calculation.

Components:
- ground: Ground temperature model (TgMaps)
- svf_resolution: SVF (Sky View Factor) resolution from multiple sources
- shadows: Shadow computation with vegetation transmissivity
- gvf: Ground View Factor calculation (upwelling radiation from surfaces)
- radiation: Complete radiation budget (shortwave and longwave from all directions)
- tmrt: Mean Radiant Temperature calculation from radiation budget
"""

__all__ = ["ground", "svf_resolution", "shadows", "gvf", "radiation", "tmrt"]
