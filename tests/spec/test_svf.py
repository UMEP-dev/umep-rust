"""
Sky View Factor (SVF) Tests

Tests derived from specs/svf.md properties.
"""

import numpy as np
import pytest
from solweig import rustalgos

# =============================================================================
# Test Fixtures
# =============================================================================


def create_flat_dsm(size=(50, 50), elevation=0.0):
    """Create completely flat DSM."""
    return np.full(size, elevation, dtype=np.float32)


def create_canyon_dsm(size=(100, 100), wall_height=30.0, canyon_width=20):
    """Create urban canyon DSM (walls on east and west sides)."""
    dsm = np.zeros(size, dtype=np.float32)
    # West wall
    dsm[:, :20] = wall_height
    # East wall
    dsm[:, -20:] = wall_height
    return dsm


def create_building_dsm(size=(100, 100), building_height=20.0):
    """Create DSM with single building in center."""
    dsm = np.zeros(size, dtype=np.float32)
    cy, cx = size[0] // 2, size[1] // 2
    dsm[cy - 5 : cy + 5, cx - 5 : cx + 5] = building_height
    return dsm


def create_courtyard_dsm(size=(100, 100), wall_height=20.0, courtyard_size=20):
    """Create square courtyard (walls on all sides, open center)."""
    dsm = np.full(size, wall_height, dtype=np.float32)
    cy, cx = size[0] // 2, size[1] // 2
    half = courtyard_size // 2
    dsm[cy - half : cy + half, cx - half : cx + half] = 0.0
    return dsm


def calculate_svf(dsm, pixel_size=1.0):
    """
    Calculate SVF using Rust implementation.

    Returns SvfResult with svf, svf_north, svf_east, svf_south, svf_west.
    """
    max_height = float(np.max(dsm) - np.min(dsm))
    if max_height == 0:
        max_height = 1.0  # Avoid division by zero for flat terrain

    # No vegetation
    vegdem = np.zeros_like(dsm, dtype=np.float32)
    vegdem2 = np.zeros_like(dsm, dtype=np.float32)

    result = rustalgos.skyview.calculate_svf(
        dsm.astype(np.float32),
        vegdem,
        vegdem2,
        float(pixel_size),
        False,  # usevegdem
        max_height,
        None,  # patch_option (default)
        None,  # min_sun_elev
        None,  # progress_callback
    )
    return result


# =============================================================================
# Property Tests (from specs/svf.md)
# =============================================================================


class TestSvfProperties:
    """Tests for SVF calculation properties."""

    def test_property_1_range_0_to_1(self):
        """Property 1: All SVF values must be between 0 and 1."""
        dsm = create_building_dsm()
        result = calculate_svf(dsm)

        svf = np.array(result.svf)
        assert np.all(svf >= 0), "SVF has values < 0"
        assert np.all(svf <= 1), "SVF has values > 1"

    def test_property_2_flat_terrain_equals_1(self):
        """Property 2: Flat open terrain has SVF = 1 everywhere."""
        dsm = create_flat_dsm(size=(50, 50))
        result = calculate_svf(dsm)

        svf = np.array(result.svf)
        # Allow small tolerance for numerical precision
        assert np.allclose(svf, 1.0, atol=0.05), f"Flat terrain SVF should be ~1, got {svf.mean():.3f}"

    def test_property_3_canyon_less_than_half(self):
        """Property 3: Deep urban canyons have SVF < 0.5."""
        # Create very narrow, deep canyon (H/W ratio > 2)
        # 60m walls, 20m wide canyon = H/W = 3
        dsm = np.zeros((100, 100), dtype=np.float32)
        dsm[:, :40] = 60.0  # West wall
        dsm[:, 60:] = 60.0  # East wall (only 20 pixels wide canyon)
        result = calculate_svf(dsm)

        svf = np.array(result.svf)
        # Check canyon floor (center of the narrow gap)
        canyon_floor_svf = svf[40:60, 48:52].mean()
        assert canyon_floor_svf < 0.5, f"Deep canyon SVF should be < 0.5, got {canyon_floor_svf:.3f}"

    def test_property_4_taller_obstacles_lower_svf(self):
        """Property 4: Points near taller obstacles have lower SVF."""
        # Low building
        dsm_low = create_building_dsm(size=(100, 100), building_height=10.0)
        result_low = calculate_svf(dsm_low)
        svf_low = np.array(result_low.svf)

        # Tall building
        dsm_tall = create_building_dsm(size=(100, 100), building_height=40.0)
        result_tall = calculate_svf(dsm_tall)
        svf_tall = np.array(result_tall.svf)

        # Check ground level near building
        ground_svf_low = svf_low[60:70, 45:55].mean()  # South of building
        ground_svf_tall = svf_tall[60:70, 45:55].mean()

        assert ground_svf_tall < ground_svf_low, (
            f"Taller building should reduce SVF: low={ground_svf_low:.3f}, tall={ground_svf_tall:.3f}"
        )

    def test_property_6_rooftops_high_svf(self):
        """Property 6: Building rooftops have SVF close to 1."""
        dsm = create_building_dsm(size=(100, 100), building_height=30.0)
        result = calculate_svf(dsm)

        svf = np.array(result.svf)
        # Check rooftop (center of grid where building is)
        cy, cx = 50, 50
        rooftop_svf = svf[cy - 3 : cy + 3, cx - 3 : cx + 3].mean()

        assert rooftop_svf > 0.8, f"Rooftop SVF should be high (>0.8), got {rooftop_svf:.3f}"

    def test_property_7_more_buildings_lower_svf(self):
        """Property 7: More buildings nearby = lower ground-level SVF."""
        # Single building
        dsm_single = np.zeros((100, 100), dtype=np.float32)
        dsm_single[45:55, 45:55] = 20.0
        result_single = calculate_svf(dsm_single)
        svf_single = np.array(result_single.svf)

        # Multiple buildings
        dsm_multi = np.zeros((100, 100), dtype=np.float32)
        dsm_multi[20:30, 20:30] = 20.0
        dsm_multi[20:30, 70:80] = 20.0
        dsm_multi[70:80, 20:30] = 20.0
        dsm_multi[70:80, 70:80] = 20.0
        dsm_multi[45:55, 45:55] = 20.0  # Center building
        result_multi = calculate_svf(dsm_multi)
        svf_multi = np.array(result_multi.svf)

        # Compare ground-level SVF at center (between buildings)
        center_svf_single = svf_single[30:40, 30:40].mean()
        center_svf_multi = svf_multi[30:40, 30:40].mean()

        assert center_svf_multi < center_svf_single, (
            f"More buildings should reduce SVF: single={center_svf_single:.3f}, multi={center_svf_multi:.3f}"
        )


class TestSvfDirectional:
    """Tests for directional SVF components."""

    def test_directional_svf_range(self):
        """Directional SVF values should be in [0, 1]."""
        dsm = create_building_dsm()
        result = calculate_svf(dsm)

        for direction in ["svf_north", "svf_east", "svf_south", "svf_west"]:
            arr = np.array(getattr(result, direction))
            assert np.all(arr >= 0), f"{direction} has values < 0"
            assert np.all(arr <= 1), f"{direction} has values > 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
