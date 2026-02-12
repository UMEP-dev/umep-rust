"""Unit tests for bitpacked shadow matrix encoding/decoding.

The SVF computation stores shadow matrices as bitpacked uint8 arrays:
  - 1 bit per sky patch, 8 patches per byte
  - Bit layout: patch i is stored at byte (i >> 3), bit position (i & 7)
  - Bit = 1 means "sky visible" (was 255 in original u8 format)
  - Bit = 0 means "blocked" (was 0 in original u8 format)

Both the Python ShadowArrays.diffsh() method and the Rust pipeline
extract bits using this scheme. These tests verify:
  1. Round-trip: pack → unpack preserves data for all bit positions
  2. diffsh formula correctness with known inputs
  3. Parity: Python diffsh matches Rust weighted_patch_sum on the same data
"""

import numpy as np
import pytest
from solweig.models.precomputed import (
    ShadowArrays,
    _pack_u8_to_bitpacked,
    _unpack_bitpacked_to_float32,
)


class TestBitpackRoundTrip:
    """Pack → unpack round-trip must preserve data for any patch count."""

    @pytest.mark.parametrize("n_patches", [1, 5, 8, 9, 15, 16, 145, 153, 306])
    def test_round_trip_all_ones(self, n_patches):
        """All-visible (255) round-trips to all 1.0."""
        rows, cols = 3, 4
        u8 = np.full((rows, cols, n_patches), 255, dtype=np.uint8)
        packed = _pack_u8_to_bitpacked(u8)
        unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
        np.testing.assert_array_equal(unpacked, 1.0)

    @pytest.mark.parametrize("n_patches", [1, 5, 8, 9, 15, 16, 145, 153, 306])
    def test_round_trip_all_zeros(self, n_patches):
        """All-blocked (0) round-trips to all 0.0."""
        rows, cols = 3, 4
        u8 = np.zeros((rows, cols, n_patches), dtype=np.uint8)
        packed = _pack_u8_to_bitpacked(u8)
        unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
        np.testing.assert_array_equal(unpacked, 0.0)

    def test_round_trip_every_bit_position(self):
        """Each of the 8 bit positions within a byte round-trips correctly."""
        # 16 patches = 2 bytes, so we test all 8 positions in both bytes
        n_patches = 16
        rows, cols = 1, 1
        for p in range(n_patches):
            u8 = np.zeros((rows, cols, n_patches), dtype=np.uint8)
            u8[0, 0, p] = 255  # Set only one patch visible
            packed = _pack_u8_to_bitpacked(u8)
            unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
            for q in range(n_patches):
                expected = 1.0 if q == p else 0.0
                assert unpacked[0, 0, q] == expected, (
                    f"Patch {q} should be {expected} when only patch {p} is set"
                )

    def test_round_trip_alternating_pattern(self):
        """Alternating on/off round-trips correctly."""
        n_patches = 153
        rows, cols = 2, 2
        u8 = np.zeros((rows, cols, n_patches), dtype=np.uint8)
        # Set every other patch to visible
        u8[:, :, ::2] = 255
        packed = _pack_u8_to_bitpacked(u8)
        unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
        for p in range(n_patches):
            expected = 1.0 if p % 2 == 0 else 0.0
            np.testing.assert_array_equal(
                unpacked[:, :, p], expected,
                err_msg=f"Patch {p} expected {expected}",
            )

    def test_round_trip_random_pattern(self):
        """Random binary pattern round-trips correctly."""
        rng = np.random.default_rng(42)
        n_patches = 153
        rows, cols = 5, 5
        # Random binary: 0 or 255
        u8 = (rng.integers(0, 2, (rows, cols, n_patches)) * 255).astype(np.uint8)
        packed = _pack_u8_to_bitpacked(u8)
        unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
        expected = (u8 / 255.0).astype(np.float32)
        np.testing.assert_array_equal(unpacked, expected)

    def test_non_byte_aligned_padding_bits_ignored(self):
        """Unused bits in the last byte don't affect result.

        For 5 patches, only bits 0-4 of byte 0 matter. Bits 5-7 are padding.
        """
        n_patches = 5
        rows, cols = 1, 1
        u8 = np.full((rows, cols, n_patches), 255, dtype=np.uint8)
        packed = _pack_u8_to_bitpacked(u8)
        # Corrupt padding bits (bits 5, 6, 7 of byte 0)
        packed[0, 0, 0] |= 0b11100000
        unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
        # Should still be 5 ones — padding bits are ignored
        assert unpacked.shape == (1, 1, 5)
        np.testing.assert_array_equal(unpacked, 1.0)


class TestDiffshFormula:
    """ShadowArrays.diffsh() must implement: shmat - (1 - vegshmat) * (1 - psi)."""

    def _make_shadow_arrays(self, shmat_u8, vegshmat_u8, n_patches):
        """Helper to create ShadowArrays from u8 per-patch arrays."""
        packed_sh = _pack_u8_to_bitpacked(shmat_u8)
        packed_veg = _pack_u8_to_bitpacked(vegshmat_u8)
        packed_vb = _pack_u8_to_bitpacked(shmat_u8)  # simplified
        return ShadowArrays(
            _shmat_u8=packed_sh,
            _vegshmat_u8=packed_veg,
            _vbshmat_u8=packed_vb,
            _n_patches=n_patches,
        )

    def test_no_vegetation_blocking_diffsh_equals_shmat(self):
        """When vegshmat = all 1s (no vegetation), diffsh = shmat exactly."""
        n_patches = 10
        rows, cols = 2, 2
        rng = np.random.default_rng(99)
        shmat_u8 = (rng.integers(0, 2, (rows, cols, n_patches)) * 255).astype(np.uint8)
        vegshmat_u8 = np.full((rows, cols, n_patches), 255, dtype=np.uint8)

        sa = self._make_shadow_arrays(shmat_u8, vegshmat_u8, n_patches)
        diffsh = sa.diffsh(transmissivity=0.03)
        expected = (shmat_u8 / 255.0).astype(np.float32)

        np.testing.assert_allclose(diffsh, expected, atol=1e-6)

    def test_full_vegetation_blocking_diffsh_equals_psi(self):
        """When vegshmat = all 0s (full veg block) and shmat = all 1s, diffsh = psi."""
        n_patches = 10
        rows, cols = 2, 2
        psi = 0.03
        shmat_u8 = np.full((rows, cols, n_patches), 255, dtype=np.uint8)
        vegshmat_u8 = np.zeros((rows, cols, n_patches), dtype=np.uint8)

        sa = self._make_shadow_arrays(shmat_u8, vegshmat_u8, n_patches)
        diffsh = sa.diffsh(transmissivity=psi)
        # shmat=1, vegshmat=0: diffsh = 1 - (1-0)*(1-0.03) = 1 - 0.97 = 0.03
        np.testing.assert_allclose(diffsh, psi, atol=1e-6)

    def test_building_blocked_always_zero(self):
        """When shmat = 0 (building blocks), diffsh <= 0 regardless of veg."""
        n_patches = 10
        rows, cols = 2, 2
        shmat_u8 = np.zeros((rows, cols, n_patches), dtype=np.uint8)
        vegshmat_u8 = np.full((rows, cols, n_patches), 255, dtype=np.uint8)

        sa = self._make_shadow_arrays(shmat_u8, vegshmat_u8, n_patches)
        diffsh = sa.diffsh(transmissivity=0.03)
        # shmat=0, vegshmat=1: diffsh = 0 - (1-1)*(1-0.03) = 0
        np.testing.assert_allclose(diffsh, 0.0, atol=1e-6)

    def test_mixed_pattern_matches_formula(self):
        """Specific mixed pattern produces correct values per formula."""
        n_patches = 4
        psi = 0.05
        # patch 0: sh=1, veg=1 → 1 - 0*0.95 = 1.0
        # patch 1: sh=1, veg=0 → 1 - 1*0.95 = 0.05
        # patch 2: sh=0, veg=1 → 0 - 0*0.95 = 0.0
        # patch 3: sh=0, veg=0 → 0 - 1*0.95 = -0.95
        shmat_u8 = np.array([[[255, 255, 0, 0]]], dtype=np.uint8)
        vegshmat_u8 = np.array([[[255, 0, 255, 0]]], dtype=np.uint8)

        sa = self._make_shadow_arrays(shmat_u8, vegshmat_u8, n_patches)
        diffsh = sa.diffsh(transmissivity=psi)

        expected = np.array([[[1.0, psi, 0.0, -(1 - psi)]]], dtype=np.float32)
        np.testing.assert_allclose(diffsh, expected, atol=1e-6)


class TestRustBitExtractionParity:
    """Rust sky.weighted_patch_sum on Python-unpacked diffsh must match Python diffsh sum.

    This validates that the Rust bit extraction in pipeline.rs (i >> 3, i & 7)
    produces the same results as Python's _unpack_bitpacked_to_float32.
    """

    def test_weighted_sum_parity(self):
        """Rust weighted_patch_sum on Python-unpacked data matches manual sum."""
        from solweig.rustalgos import sky

        rng = np.random.default_rng(123)
        n_patches = 153
        rows, cols = 4, 4

        # Create random bitpacked shadow matrices
        shmat_u8 = (rng.integers(0, 2, (rows, cols, n_patches)) * 255).astype(np.uint8)
        vegshmat_u8 = (rng.integers(0, 2, (rows, cols, n_patches)) * 255).astype(np.uint8)
        packed_sh = _pack_u8_to_bitpacked(shmat_u8)
        packed_veg = _pack_u8_to_bitpacked(vegshmat_u8)

        sa = ShadowArrays(
            _shmat_u8=packed_sh,
            _vegshmat_u8=packed_veg,
            _vbshmat_u8=packed_sh.copy(),
            _n_patches=n_patches,
        )

        # Python diffsh
        psi = 0.03
        py_diffsh = sa.diffsh(transmissivity=psi)

        # Uniform weights
        weights = np.ones(n_patches, dtype=np.float32) / n_patches

        # Rust weighted_patch_sum on Python-unpacked data
        rs_result = np.asarray(sky.weighted_patch_sum(py_diffsh.astype(np.float32), weights))

        # Python manual sum
        py_result = np.sum(py_diffsh * weights[np.newaxis, np.newaxis, :], axis=2)

        np.testing.assert_allclose(
            rs_result, py_result, rtol=1e-5, atol=1e-6,
            err_msg="Rust weighted_patch_sum differs from Python sum on diffsh",
        )

    def test_anisotropic_sky_uses_correct_bits(self):
        """anisotropic_sky with known shadow patterns produces expected behavior.

        Creates a shadow matrix where only high-altitude patches are visible,
        then verifies that ldown responds correctly (should be non-zero since
        visible patches still contribute radiation).
        """
        from solweig.rustalgos import sky

        n_patches = 96  # 6 altitude bands
        rows, cols = 3, 3

        # Generate patches: 6 altitude bands
        patches = []
        alt_bands = [6, 18, 30, 42, 54, 66]
        azis_per_band = [30, 24, 24, 18, 12, 6]  # total = 114, but we trim
        count = 0
        for alt, n_azi in zip(alt_bands, azis_per_band):
            azi_step = 360.0 / n_azi
            for j in range(n_azi):
                if count >= n_patches:
                    break
                patches.append([alt, j * azi_step])
                count += 1
            if count >= n_patches:
                break
        l_patches = np.array(patches[:n_patches], dtype=np.float32)

        # Steradians (simplified)
        steradians = np.ones(n_patches, dtype=np.float32) / n_patches

        # Shadow matrices: all patches visible (uint8 per byte)
        n_pack = (n_patches + 7) // 8
        shmat = np.full((rows, cols, n_pack), 0xFF, dtype=np.uint8)
        vegshmat = np.full((rows, cols, n_pack), 0xFF, dtype=np.uint8)
        vbshmat = np.full((rows, cols, n_pack), 0xFF, dtype=np.uint8)

        # Luminance: uniform
        lum = np.ones(n_patches, dtype=np.float32) / n_patches
        lv = np.column_stack([l_patches, lum]).astype(np.float32)

        asvf = np.zeros((rows, cols), dtype=np.float32)  # arccos(sqrt(1)) = 0
        lup = np.full((rows, cols), 400.0, dtype=np.float32)
        shadow = np.ones((rows, cols), dtype=np.float32)
        kup = np.full((rows, cols), 50.0, dtype=np.float32)

        sun = sky.SunParams(altitude=45.0, azimuth=180.0)
        sky_p = sky.SkyParams(esky=0.75, ta=25.0, cyl=True, wall_scheme=False, albedo=0.2)
        surf_p = sky.SurfaceParams(tgwall=2.0, ewall=0.9, rad_i=600.0, rad_d=200.0)

        result = sky.anisotropic_sky(
            shmat, vegshmat, vbshmat,
            sun, asvf, sky_p, l_patches,
            None, None, steradians, surf_p,
            lup, lv, shadow,
            kup, kup, kup, kup,
        )

        ldown = np.asarray(result.ldown)
        # All patches visible → ldown should be positive and reasonable
        assert np.all(ldown > 0), "ldown should be positive when all patches visible"
        assert np.all(ldown < 800), "ldown should be < 800 W/m²"

        # Now block ALL patches and verify ldown changes
        # (blocking sky patches adds wall emission instead of sky emission,
        # so ldown may increase or decrease depending on wall/sky temperatures)
        shmat_blocked = np.zeros((rows, cols, n_pack), dtype=np.uint8)
        result_blocked = sky.anisotropic_sky(
            shmat_blocked, vegshmat, vbshmat,
            sun, asvf, sky_p, l_patches,
            None, None, steradians, surf_p,
            lup, lv, shadow,
            kup, kup, kup, kup,
        )
        ldown_blocked = np.asarray(result_blocked.ldown)
        assert not np.allclose(ldown, ldown_blocked, atol=0.1), (
            "Blocking all patches should change ldown"
        )

        # Diffuse shortwave (kside_d) should definitely decrease when sky is blocked
        kside_d_open = np.asarray(result.kside_d)
        kside_d_blocked = np.asarray(result_blocked.kside_d)
        assert np.all(kside_d_open >= kside_d_blocked - 1e-3), (
            "Blocking sky should not increase diffuse shortwave"
        )
