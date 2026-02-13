"""Run metadata and provenance tracking."""

from __future__ import annotations

import json
from datetime import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import HumanParams, Location, SurfaceData, Weather


def create_run_metadata(
    surface: SurfaceData,
    location: Location,
    weather_series: list[Weather],
    human: HumanParams | None,
    physics: SimpleNamespace | None,
    materials: SimpleNamespace | None,
    use_anisotropic_sky: bool,
    conifer: bool,
    output_dir: str | Path,
    outputs: list[str] | None,
) -> dict:
    """
    Create run metadata dictionary for provenance tracking.

    Args:
        surface: Surface data used in calculation.
        location: Location parameters.
        weather_series: List of Weather objects.
        human: Human parameters (or None for defaults).
        physics: Physics parameters (or None for defaults).
        materials: Materials parameters (or None).
        use_anisotropic_sky: Whether anisotropic sky model was used.
        conifer: Whether conifer mode was used.
        output_dir: Output directory path.
        outputs: List of output variables saved.

    Returns:
        Dictionary containing run metadata.
    """
    from .utils import namespace_to_dict

    metadata = {
        "solweig_version": "0.0.1a1",
        "run_timestamp": dt.now().isoformat(),
        "grid": {
            "rows": surface.shape[0],
            "cols": surface.shape[1],
            "pixel_size": surface.pixel_size,
            "crs": surface.crs,
        },
        "location": {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "utc_offset": location.utc_offset,
        },
        "timeseries": {
            "start": weather_series[0].datetime.isoformat(),
            "end": weather_series[-1].datetime.isoformat(),
            "timesteps": len(weather_series),
        },
        "parameters": {
            "use_anisotropic_sky": use_anisotropic_sky,
            "conifer": conifer,
        },
        "outputs": {
            "directory": str(output_dir),
            "variables": outputs or ["tmrt"],
        },
    }

    # Add optional parameter info
    if human is not None:
        metadata["human"] = {
            "abs_k": human.abs_k,
            "abs_l": human.abs_l,
            "posture": human.posture,
        }

    if physics is not None:
        physics_info = {}
        try:
            physics_info["full_params"] = namespace_to_dict(physics)
        except Exception:
            physics_info["note"] = "Physics parameters provided but not serializable"
        metadata["physics"] = physics_info

    if materials is not None:
        materials_info = {}
        try:
            materials_info["full_params"] = namespace_to_dict(materials)
        except Exception:
            materials_info["note"] = "Materials parameters provided but not serializable"
        metadata["materials"] = materials_info

    return metadata


def save_run_metadata(metadata: dict, output_dir: str | Path, filename: str = "run_metadata.json") -> Path:
    """
    Save run metadata to JSON file.

    Args:
        metadata: Metadata dictionary from create_run_metadata().
        output_dir: Output directory.
        filename: Filename for metadata JSON (default: run_metadata.json).

    Returns:
        Path to saved metadata file.
    """
    output_path = Path(output_dir)
    metadata_path = output_path / filename

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def load_run_metadata(metadata_path: str | Path) -> dict:
    """
    Load run metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file.

    Returns:
        Metadata dictionary.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata
