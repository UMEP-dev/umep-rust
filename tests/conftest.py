"""Shared pytest configuration and path setup."""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so that both `tests.qgis_mocks`
# and `qgis_plugin.*` imports work regardless of how pytest is invoked.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def make_mock_svf(shape: tuple[int, ...]):
    """Create a mock SvfArrays for tests (fully open sky)."""
    from solweig.models.precomputed import SvfArrays

    ones = np.ones(shape, dtype=np.float32)
    return SvfArrays(
        svf=ones.copy(),
        svf_north=ones.copy(),
        svf_east=ones.copy(),
        svf_south=ones.copy(),
        svf_west=ones.copy(),
        svf_veg=ones.copy(),
        svf_veg_north=ones.copy(),
        svf_veg_east=ones.copy(),
        svf_veg_south=ones.copy(),
        svf_veg_west=ones.copy(),
        svf_aveg=ones.copy(),
        svf_aveg_north=ones.copy(),
        svf_aveg_east=ones.copy(),
        svf_aveg_south=ones.copy(),
        svf_aveg_west=ones.copy(),
    )
