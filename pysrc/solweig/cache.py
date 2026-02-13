"""
Cache validation utilities for SVF and wall data.

Provides hash-based validation to detect stale caches when input data changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cache metadata filename
CACHE_METADATA_FILE = "cache_meta.json"


def pixel_size_tag(pixel_size: float) -> str:
    """Return a directory-safe tag encoding the pixel size, e.g. ``'px1.000'``."""
    return f"px{pixel_size:.3f}"


def compute_array_hash(arr: np.ndarray, *, sample_size: int = 10000) -> str:
    """
    Compute a fast hash of a numpy array.

    Uses a combination of shape, dtype, and sampled values for speed.
    For large arrays, samples evenly spaced values rather than hashing everything.

    Args:
        arr: Numpy array to hash.
        sample_size: Maximum number of values to sample for hashing.

    Returns:
        Hex string hash.
    """
    hasher = hashlib.sha256()

    # Include shape and dtype
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())

    # For small arrays, hash everything
    flat = arr.ravel()
    if len(flat) <= sample_size:
        hasher.update(flat.tobytes())
    else:
        # Sample evenly spaced values for large arrays
        indices = np.linspace(0, len(flat) - 1, sample_size, dtype=np.int64)
        hasher.update(flat[indices].tobytes())

    return hasher.hexdigest()[:16]  # First 16 chars is enough


@dataclass
class CacheMetadata:
    """Metadata for cache validation."""

    dsm_hash: str
    dsm_shape: tuple[int, int]
    pixel_size: float
    cdsm_hash: str | None = None
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "dsm_hash": self.dsm_hash,
            "dsm_shape": list(self.dsm_shape),
            "pixel_size": self.pixel_size,
            "cdsm_hash": self.cdsm_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheMetadata:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            dsm_hash=data["dsm_hash"],
            dsm_shape=tuple(data["dsm_shape"]),
            pixel_size=data["pixel_size"],
            cdsm_hash=data.get("cdsm_hash"),
        )

    @classmethod
    def from_arrays(
        cls,
        dsm: np.ndarray,
        pixel_size: float,
        cdsm: np.ndarray | None = None,
    ) -> CacheMetadata:
        """Create metadata from input arrays."""
        return cls(
            dsm_hash=compute_array_hash(dsm),
            dsm_shape=(dsm.shape[0], dsm.shape[1]),
            pixel_size=pixel_size,
            cdsm_hash=compute_array_hash(cdsm) if cdsm is not None else None,
        )

    def matches(self, other: CacheMetadata) -> bool:
        """Check if this metadata matches another."""
        return (
            self.dsm_hash == other.dsm_hash
            and self.dsm_shape == other.dsm_shape
            and abs(self.pixel_size - other.pixel_size) < 0.001
            and self.cdsm_hash == other.cdsm_hash
        )

    def save(self, directory: Path) -> None:
        """Save metadata to cache directory."""
        meta_path = directory / CACHE_METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> CacheMetadata | None:
        """Load metadata from cache directory. Returns None if not found."""
        meta_path = directory / CACHE_METADATA_FILE
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None


def validate_cache(
    cache_dir: Path,
    dsm: np.ndarray,
    pixel_size: float,
    cdsm: np.ndarray | None = None,
) -> bool:
    """
    Validate that cached data matches current inputs.

    Args:
        cache_dir: Directory containing cached data.
        dsm: Current DSM array.
        pixel_size: Current pixel size.
        cdsm: Current CDSM array (optional).

    Returns:
        True if cache is valid, False if stale or missing.
    """
    stored = CacheMetadata.load(cache_dir)
    if stored is None:
        logger.debug(f"No cache metadata found in {cache_dir}")
        return False

    current = CacheMetadata.from_arrays(dsm, pixel_size, cdsm)

    if stored.matches(current):
        logger.debug(f"Cache validated: {cache_dir}")
        return True
    else:
        logger.info(f"Cache stale (input changed): {cache_dir}")
        logger.debug(f"  Stored: dsm_hash={stored.dsm_hash}, shape={stored.dsm_shape}")
        logger.debug(f"  Current: dsm_hash={current.dsm_hash}, shape={current.dsm_shape}")
        return False


def clear_stale_cache(cache_dir: Path) -> None:
    """
    Remove stale cache files from a directory.

    Deletes all .npy files and the metadata file.
    """
    if not cache_dir.exists():
        return

    import shutil

    for item in cache_dir.iterdir():
        if item.is_file() and (item.suffix == ".npy" or item.name == CACHE_METADATA_FILE):
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    logger.info(f"Cleared stale cache: {cache_dir}")
