"""Model configuration classes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class ModelConfig:
    """
    Model configuration for SOLWEIG calculations.

    Groups all computational settings in one typed object.
    Pure configuration - no paths or data.

    Attributes:
        use_anisotropic_sky: Use anisotropic sky model. Default False.
        human: Human body parameters for Tmrt calculations.
        material_params: Optional material properties from JSON file.
        outputs: Which outputs to save in timeseries calculations.
        max_shadow_distance_m: Maximum shadow reach in meters. Default 500.0.
            Caps shadow ray computation distance and serves as tile overlap buffer
            for automatic tiled processing of large rasters. At low sun angles (3°),
            a 26m building casts a 500m shadow — taller buildings are capped.
        tile_workers: Number of workers for tiled orchestration. If None,
            picks an adaptive default based on CPU count.
        tile_queue_depth: Extra queued tile tasks beyond active workers. If None,
            defaults to one queue slot per worker when prefetching is enabled.
        prefetch_tiles: Whether to prefetch tile tasks beyond active workers.
            If None, runtime chooses automatically based on memory pressure.

    Note:
        UTCI and PET are now computed via post-processing functions (compute_utci, compute_pet)
        rather than during the main calculation loop for better performance.

    Examples:
        Basic usage with defaults:

        >>> config = ModelConfig.defaults()
        >>> config.save("my_config.json")

        Custom configuration:

        >>> config = ModelConfig(
        ...     use_anisotropic_sky=True,
        ...     human=HumanParams(abs_k=0.7, posture="standing"),
        ... )

        Load from legacy JSON:

        >>> config = ModelConfig.from_json("parametersforsolweig.json")
    """

    use_anisotropic_sky: bool = True
    human: HumanParams | None = None
    material_params: SimpleNamespace | None = None
    outputs: list[str] = field(default_factory=lambda: ["tmrt"])
    physics: SimpleNamespace | None = None
    materials: SimpleNamespace | None = None
    max_shadow_distance_m: float = 500.0
    tile_workers: int | None = None
    tile_queue_depth: int | None = None
    prefetch_tiles: bool | None = None

    def __post_init__(self):
        """Initialize default HumanParams if not provided."""
        # Defer import to avoid forward reference issues
        if self.human is None:
            # HumanParams is defined later in this module
            pass  # Will be instantiated when HumanParams is available

    @classmethod
    def defaults(cls) -> ModelConfig:
        """
        Standard configuration for most users.

        Returns:
            ModelConfig with recommended defaults:
            - Anisotropic sky enabled
        """
        return cls(
            use_anisotropic_sky=True,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """
        Load configuration from legacy JSON parameters file.

        Args:
            path: Path to parametersforsolweig.json

        Returns:
            ModelConfig with settings extracted from JSON

        Example:
            >>> config = ModelConfig.from_json("parametersforsolweig.json")
            >>> config.human.abs_k  # From Tmrt_params
            0.7
        """
        from ..loaders import load_params

        params = load_params(path)

        # Extract human parameters from JSON
        human = HumanParams()
        if hasattr(params, "Tmrt_params"):
            human.abs_k = getattr(params.Tmrt_params, "absK", 0.7)
            human.abs_l = getattr(params.Tmrt_params, "absL", 0.97)
            posture_str = getattr(params.Tmrt_params, "posture", "Standing")
            human.posture = posture_str.lower()

        if hasattr(params, "PET_settings"):
            human.age = getattr(params.PET_settings, "Age", 35)
            human.weight = getattr(params.PET_settings, "Weight", 75.0)
            human.height = getattr(params.PET_settings, "Height", 1.75)
            human.sex = getattr(params.PET_settings, "Sex", 1)
            human.activity = getattr(params.PET_settings, "Activity", 80.0)
            human.clothing = getattr(params.PET_settings, "clo", 0.9)

        return cls(
            human=human,
            material_params=params,
        )

    def save(self, path: str | Path):
        """
        Save configuration to JSON file.

        Args:
            path: Output path for JSON file

        Example:
            >>> config = ModelConfig.defaults()
            >>> config.save("my_settings.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to dict
        data = {
            "use_anisotropic_sky": self.use_anisotropic_sky,
            "max_shadow_distance_m": self.max_shadow_distance_m,
            "tile_workers": self.tile_workers,
            "tile_queue_depth": self.tile_queue_depth,
            "prefetch_tiles": self.prefetch_tiles,
            "outputs": self.outputs,
            "human": {
                "posture": self.human.posture,
                "abs_k": self.human.abs_k,
                "abs_l": self.human.abs_l,
                "age": self.human.age,
                "weight": self.human.weight,
                "height": self.human.height,
                "sex": self.human.sex,
                "activity": self.human.activity,
                "clothing": self.human.clothing,
            }
            if self.human
            else None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            ModelConfig loaded from file

        Example:
            >>> config = ModelConfig.load("my_settings.json")
            >>> results = calculate_timeseries(surface, weather, config=config)
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        # Deserialize human params
        human = None
        if data.get("human"):
            human = HumanParams(**data["human"])

        return cls(
            use_anisotropic_sky=data.get("use_anisotropic_sky", False),
            max_shadow_distance_m=data.get("max_shadow_distance_m", 500.0),
            tile_workers=data.get("tile_workers"),
            tile_queue_depth=data.get("tile_queue_depth"),
            prefetch_tiles=data.get("prefetch_tiles"),
            human=human,
            outputs=data.get("outputs", ["tmrt"]),
        )


@dataclass
class HumanParams:
    """
    Human body parameters for thermal comfort calculations.

    These parameters affect how radiation is absorbed by a person.
    Default values represent a standard reference person.

    Attributes:
        posture: Body posture ("standing" or "sitting"). Default "standing".
        abs_k: Shortwave absorption coefficient. Default 0.7.
        abs_l: Longwave absorption coefficient. Default 0.97.

    PET-specific parameters (used by compute_pet() post-processing):
        age: Age in years. Default 35.
        weight: Body weight in kg. Default 75.
        height: Body height in meters. Default 1.75.
        sex: Biological sex (1=male, 2=female). Default 1.
        activity: Metabolic activity in W. Default 80.
        clothing: Clothing insulation in clo. Default 0.9.
    """

    posture: str = "standing"
    abs_k: float = 0.7
    abs_l: float = 0.97

    # PET-specific (optional)
    age: int = 35
    weight: float = 75.0
    height: float = 1.75
    sex: int = 1
    activity: float = 80.0
    clothing: float = 0.9

    def __post_init__(self):
        valid_postures = ("standing", "sitting")
        if self.posture not in valid_postures:
            raise ValueError(f"Posture must be one of {valid_postures}, got {self.posture}")
        if not 0 < self.abs_k <= 1:
            raise ValueError(f"abs_k must be in (0, 1], got {self.abs_k}")
        if not 0 < self.abs_l <= 1:
            raise ValueError(f"abs_l must be in (0, 1], got {self.abs_l}")
