"""GPU vs CPU parity tests for the anisotropic sky computation.

Verifies that the GPU (WGSL) anisotropic sky shader produces results
matching the CPU (Rayon) implementation within f32 accumulation tolerance.

The GPU path outputs (ldown, lside, kside_partial) which the pipeline
combines with trivial CPU-side terms (kside_i, keast=kup*0.5, etc.).
"""

from datetime import datetime

import numpy as np
import pytest
from solweig.api import (
    Location,
    SurfaceData,
    Weather,
    calculate,
)
from solweig.models.precomputed import ShadowArrays
from solweig.rustalgos import pipeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def location():
    return Location(latitude=57.7, longitude=12.0, utc_offset=1)


@pytest.fixture(scope="module")
def noon_weather():
    return Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=25.0,
        rh=50.0,
        global_rad=800.0,
    )


def _make_flat_surface_with_shadows(shape=(10, 10), n_patches=153):
    """Create a flat surface with synthetic shadow matrices."""
    from conftest import make_mock_svf

    dsm = np.ones(shape, dtype=np.float32) * 2.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(shape))

    n_pack = (n_patches + 7) // 8
    shmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
    vegshmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
    vbshmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)

    surface.shadow_matrices = ShadowArrays(
        _shmat_u8=shmat_u8,
        _vegshmat_u8=vegshmat_u8,
        _vbshmat_u8=vbshmat_u8,
        _n_patches=n_patches,
    )
    return surface


def _make_partial_shadow_surface(shape=(15, 15), n_patches=153):
    """Surface with spatially varying shadow patterns for thorough parity testing."""
    from conftest import make_mock_svf

    dsm = np.ones(shape, dtype=np.float32) * 2.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(shape))

    n_pack = (n_patches + 7) // 8
    rng = np.random.default_rng(42)

    shmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
    vegshmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
    vbshmat_u8 = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)

    # Block some patches in the right half
    for p in range(40):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        shmat_u8[:, shape[1] // 2 :, byte_idx] &= ~bit_mask

    # Random veg blocking in top half
    for p in range(20, 60):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        mask = rng.integers(0, 2, (shape[0] // 2, shape[1]), dtype=np.uint8)
        vegshmat_u8[: shape[0] // 2, :, byte_idx] &= ~(bit_mask * (1 - mask))

    surface.shadow_matrices = ShadowArrays(
        _shmat_u8=shmat_u8,
        _vegshmat_u8=vegshmat_u8,
        _vbshmat_u8=vbshmat_u8.copy(),
        _n_patches=n_patches,
    )
    return surface


@pytest.fixture(scope="module")
def gpu_available():
    """Check if GPU aniso is available (GPU feature compiled + hardware present)."""
    try:
        return pipeline.is_aniso_gpu_enabled()
    except AttributeError:
        return False


class TestAnisoGpuCpuParity:
    """GPU and CPU anisotropic sky must produce matching results."""

    def _run_with_gpu(self, surface, location, weather, *, gpu_on):
        """Run calculate() with GPU enabled or disabled."""
        try:
            if gpu_on:
                pipeline.enable_aniso_gpu()
            else:
                pipeline.disable_aniso_gpu()
        except AttributeError:
            if gpu_on:
                pytest.skip("GPU feature not compiled")

        summary = calculate(
            surface,
            [weather],
            location,
            use_anisotropic_sky=True,
            timestep_outputs=["tmrt", "kdown"],
        )
        return summary.results[0]

    def test_open_sky_tmrt_parity(self, location, noon_weather, gpu_available):
        """Open sky: GPU and CPU Tmrt match within f32 tolerance."""
        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        result_gpu = self._run_with_gpu(surface_gpu, location, noon_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, noon_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch on open sky",
        )

    def test_open_sky_kdown_parity(self, location, noon_weather, gpu_available):
        """Open sky: GPU and CPU kdown match within f32 tolerance."""
        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        result_gpu = self._run_with_gpu(surface_gpu, location, noon_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, noon_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.kdown) & ~np.isnan(result_cpu.kdown)
        if np.any(valid):
            np.testing.assert_allclose(
                result_gpu.kdown[valid],
                result_cpu.kdown[valid],
                rtol=1e-3,
                atol=1.0,
                err_msg="GPU vs CPU kdown mismatch on open sky",
            )

    def test_partial_shadows_parity(self, location, noon_weather, gpu_available):
        """Partial shadows: GPU and CPU Tmrt match within f32 tolerance."""
        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_partial_shadow_surface()
        surface_cpu = _make_partial_shadow_surface()

        result_gpu = self._run_with_gpu(surface_gpu, location, noon_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, noon_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with partial shadows",
        )

    def test_full_obstruction_parity(self, location, noon_weather, gpu_available):
        """All patches blocked: GPU and CPU should produce matching low-radiation results."""
        if not gpu_available:
            pytest.skip("GPU not available")

        shape = (10, 10)
        n_patches = 153
        surface_gpu = _make_flat_surface_with_shadows(shape=shape, n_patches=n_patches)
        surface_cpu = _make_flat_surface_with_shadows(shape=shape, n_patches=n_patches)

        # Zero out all shadow matrices — every patch blocked
        n_pack = (n_patches + 7) // 8
        zeros = np.zeros((shape[0], shape[1], n_pack), dtype=np.uint8)
        for s in (surface_gpu, surface_cpu):
            s.shadow_matrices = ShadowArrays(
                _shmat_u8=zeros.copy(),
                _vegshmat_u8=zeros.copy(),
                _vbshmat_u8=zeros.copy(),
                _n_patches=n_patches,
            )

        result_gpu = self._run_with_gpu(surface_gpu, location, noon_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, noon_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with full obstruction",
        )

    def test_night_time_parity(self, location, gpu_available):
        """Night time (sun below horizon): GPU and CPU should match."""
        if not gpu_available:
            pytest.skip("GPU not available")

        # January 2 AM — sun well below horizon at lat 57.7
        night_weather = Weather(
            datetime=datetime(2024, 1, 15, 2, 0),
            ta=2.0,
            rh=80.0,
            global_rad=0.0,
        )

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        result_gpu = self._run_with_gpu(surface_gpu, location, night_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, night_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values even at night"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch at night time",
        )

        # Verify kdown is zero or near-zero at night
        if result_gpu.kdown is not None:
            kdown_valid = ~np.isnan(result_gpu.kdown)
            if np.any(kdown_valid):
                assert np.nanmax(result_gpu.kdown) < 1.0, "kdown should be ~0 at night"

    def test_sitting_posture_parity(self, location, noon_weather, gpu_available):
        """Sitting posture (cyl=False): GPU short-circuits to zero, CPU should match."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from solweig.models.config import HumanParams

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        sitting = HumanParams(posture="sitting")

        for gpu_on, _surface in [(True, surface_gpu), (False, surface_cpu)]:
            try:
                if gpu_on:
                    pipeline.enable_aniso_gpu()
                else:
                    pipeline.disable_aniso_gpu()
            except AttributeError:
                if gpu_on:
                    pytest.skip("GPU feature not compiled")

        summary_gpu = calculate(
            surface_gpu,
            [noon_weather],
            location,
            use_anisotropic_sky=True,
            human=sitting,
            timestep_outputs=["tmrt", "kdown"],
        )
        result_gpu = summary_gpu.results[0]
        pipeline.disable_aniso_gpu()
        summary_cpu = calculate(
            surface_cpu,
            [noon_weather],
            location,
            use_anisotropic_sky=True,
            human=sitting,
            timestep_outputs=["tmrt", "kdown"],
        )
        result_cpu = summary_cpu.results[0]

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with sitting posture (cyl=False)",
        )

    def test_zero_radiation_parity(self, location, gpu_available):
        """Zero radiation input: GPU and CPU should match."""
        if not gpu_available:
            pytest.skip("GPU not available")

        # Daytime sun position but zero radiation (overcast edge case)
        zero_rad_weather = Weather(
            datetime=datetime(2024, 7, 15, 12, 0),
            ta=20.0,
            rh=90.0,
            global_rad=0.0,
        )

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        result_gpu = self._run_with_gpu(surface_gpu, location, zero_rad_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, zero_rad_weather, gpu_on=False)

        valid = ~np.isnan(result_gpu.tmrt) & ~np.isnan(result_cpu.tmrt)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            result_gpu.tmrt[valid],
            result_cpu.tmrt[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with zero radiation",
        )

    def test_invalid_pixels_nan_parity(self, location, noon_weather, gpu_available):
        """Invalid pixels (NaN DSM) produce NaN in both GPU and CPU paths."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from conftest import make_mock_svf

        shape = (10, 10)
        n_patches = 153
        dsm = np.ones(shape, dtype=np.float32) * 2.0
        # Mark some pixels as invalid (NaN in DSM)
        dsm[0:3, 0:3] = np.nan

        surface_gpu = SurfaceData(dsm=dsm.copy(), pixel_size=1.0, svf=make_mock_svf(shape))
        surface_cpu = SurfaceData(dsm=dsm.copy(), pixel_size=1.0, svf=make_mock_svf(shape))

        n_pack = (n_patches + 7) // 8
        shmat = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
        vegshmat = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)
        vbshmat = np.full((shape[0], shape[1], n_pack), 0xFF, dtype=np.uint8)

        for s in (surface_gpu, surface_cpu):
            s.shadow_matrices = ShadowArrays(
                _shmat_u8=shmat.copy(),
                _vegshmat_u8=vegshmat.copy(),
                _vbshmat_u8=vbshmat.copy(),
                _n_patches=n_patches,
            )

        result_gpu = self._run_with_gpu(surface_gpu, location, noon_weather, gpu_on=True)
        result_cpu = self._run_with_gpu(surface_cpu, location, noon_weather, gpu_on=False)

        # NaN pixels should be NaN in both
        gpu_nan = np.isnan(result_gpu.tmrt)
        cpu_nan = np.isnan(result_cpu.tmrt)
        np.testing.assert_array_equal(
            gpu_nan,
            cpu_nan,
            err_msg="GPU and CPU should produce NaN at the same pixels",
        )

        # Valid pixels should match
        valid = ~gpu_nan & ~cpu_nan
        if np.any(valid):
            np.testing.assert_allclose(
                result_gpu.tmrt[valid],
                result_cpu.tmrt[valid],
                rtol=1e-3,
                atol=0.5,
                err_msg="GPU vs CPU Tmrt mismatch on valid pixels with NaN neighbors",
            )

    def test_gpu_fallback_when_disabled(self, location, noon_weather):
        """With GPU disabled, results are identical to CPU-only path."""
        surface_a = _make_flat_surface_with_shadows()
        surface_b = _make_flat_surface_with_shadows()

        result_a = self._run_with_gpu(surface_a, location, noon_weather, gpu_on=False)
        result_b = self._run_with_gpu(surface_b, location, noon_weather, gpu_on=False)

        valid = ~np.isnan(result_a.tmrt) & ~np.isnan(result_b.tmrt)
        if np.any(valid):
            np.testing.assert_array_equal(
                result_a.tmrt[valid],
                result_b.tmrt[valid],
                err_msg="Two CPU-only runs should produce identical results",
            )
