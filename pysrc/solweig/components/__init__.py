"""
SOLWEIG computation components.

The production pipeline uses the fused Rust path (``pipeline.compute_timestep``),
which performs shadows, ground temp, GVF, radiation, and Tmrt in a single FFI call.

These Python modules are retained as readable reference implementations and for
use in tests and validation scripts. The modules still called by the fused path are:

- **svf_resolution** — ``resolve_svf()``, ``adjust_svfbuveg_with_psi()``
- **shadows** — ``compute_transmissivity()``
- **gvf** — ``detect_building_mask()``

The remaining component functions (``compute_radiation``, ``compute_tmrt``,
``compute_shadows``, ``compute_gvf``, ``compute_ground_temperature``) are
reference-only and not called by the production ``calculate()`` API.
"""

__all__ = ["ground", "svf_resolution", "shadows", "gvf", "radiation", "tmrt"]
