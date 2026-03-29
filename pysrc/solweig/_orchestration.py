"""Shared helpers for configuration resolution.

Internal module -- not part of the public API.
"""

from __future__ import annotations

from .solweig_logging import get_logger

logger = get_logger(__name__)


def resolve_config_params(
    *,
    config,
    use_anisotropic_sky: bool | None,
    human,
    physics,
    materials,
    outputs: list[str] | None,
    max_shadow_distance_m: float | None,
    tile_size: int | None = None,
    tile_workers: int | None = None,
    tile_queue_depth: int | None = None,
    prefetch_tiles: bool | None = None,
) -> dict:
    """Resolve effective parameters: explicit params > config fallback > defaults.

    Returns a dict with keys matching the parameter names, all resolved
    to their effective values. Callers unpack the result.
    """
    effective_aniso = use_anisotropic_sky
    effective_human = human
    effective_physics = physics
    effective_materials = materials
    effective_outputs = outputs
    effective_max_shadow = max_shadow_distance_m
    effective_tile_size = tile_size
    effective_tile_workers = tile_workers
    effective_tile_queue_depth = tile_queue_depth
    effective_prefetch_tiles = prefetch_tiles

    if config is not None:
        if effective_aniso is None:
            effective_aniso = config.use_anisotropic_sky
        if effective_human is None:
            effective_human = config.human
        if effective_physics is None:
            effective_physics = config.physics
        if effective_materials is None:
            effective_materials = config.materials
        if effective_outputs is None and config.outputs:
            effective_outputs = config.outputs
        if effective_max_shadow is None:
            effective_max_shadow = config.max_shadow_distance_m
        if effective_tile_size is None:
            effective_tile_size = config.tile_size
        if effective_tile_workers is None:
            effective_tile_workers = config.tile_workers
        if effective_tile_queue_depth is None:
            effective_tile_queue_depth = config.tile_queue_depth
        if effective_prefetch_tiles is None:
            effective_prefetch_tiles = config.prefetch_tiles

    if effective_aniso is None:
        effective_aniso = True
    if effective_physics is None:
        from .loaders import load_physics

        effective_physics = load_physics()
    if effective_materials is None:
        from .loaders import load_params

        effective_materials = load_params()

    return {
        "use_anisotropic_sky": effective_aniso,
        "human": effective_human,
        "physics": effective_physics,
        "materials": effective_materials,
        "outputs": effective_outputs,
        "max_shadow_distance_m": effective_max_shadow,
        "tile_size": effective_tile_size,
        "tile_workers": effective_tile_workers,
        "tile_queue_depth": effective_tile_queue_depth,
        "prefetch_tiles": effective_prefetch_tiles,
    }
