"""
SOLWEIG computation components.

The production pipeline uses the fused Rust path (``pipeline.compute_timestep``),
which performs shadows, ground temp, GVF, radiation, and Tmrt in a single FFI call.

Active Python modules still called by the fused path:

- **svf_resolution** — ``resolve_svf()``, ``adjust_svfbuveg_with_psi()``
- **shadows** — ``compute_transmissivity()``
- **gvf** — ``detect_building_mask()``
- **ground** — ``compute_ground_temperature()`` (also used by tests)
"""

__all__ = ["ground", "svf_resolution", "shadows", "gvf"]
