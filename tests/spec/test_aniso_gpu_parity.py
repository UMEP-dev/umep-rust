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

    def _run_with_gpu(self, surface, location, weather, output_dir, *, gpu_on):
        """Run calculate() with GPU enabled or disabled.

        Returns the output_dir so callers can read GeoTIFFs from it.
        """
        try:
            if gpu_on:
                pipeline.enable_aniso_gpu()
            else:
                pipeline.disable_aniso_gpu()
        except AttributeError:
            if gpu_on:
                pytest.skip("GPU feature not compiled")

        calculate(
            surface,
            [weather],
            location,
            use_anisotropic_sky=True,
            output_dir=output_dir,
            outputs=["tmrt", "kdown"],
        )
        return output_dir

    def test_open_sky_tmrt_parity(self, location, noon_weather, gpu_available, tmp_path):
        """Open sky: GPU and CPU Tmrt match within f32 tolerance."""
        from conftest import read_timestep_geotiff

        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, noon_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, noon_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch on open sky",
        )

    def test_open_sky_kdown_parity(self, location, noon_weather, gpu_available, tmp_path):
        """Open sky: GPU and CPU kdown match within f32 tolerance."""
        from conftest import read_timestep_geotiff

        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_flat_surface_with_shadows()
        surface_cpu = _make_flat_surface_with_shadows()

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, noon_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, noon_weather, out_cpu, gpu_on=False)

        kdown_gpu = read_timestep_geotiff(out_gpu, "kdown", 0)
        kdown_cpu = read_timestep_geotiff(out_cpu, "kdown", 0)

        valid = ~np.isnan(kdown_gpu) & ~np.isnan(kdown_cpu)
        if np.any(valid):
            np.testing.assert_allclose(
                kdown_gpu[valid],
                kdown_cpu[valid],
                rtol=1e-3,
                atol=1.0,
                err_msg="GPU vs CPU kdown mismatch on open sky",
            )

    def test_partial_shadows_parity(self, location, noon_weather, gpu_available, tmp_path):
        """Partial shadows: GPU and CPU Tmrt match within f32 tolerance."""
        from conftest import read_timestep_geotiff

        if not gpu_available:
            pytest.skip("GPU not available")

        surface_gpu = _make_partial_shadow_surface()
        surface_cpu = _make_partial_shadow_surface()

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, noon_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, noon_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with partial shadows",
        )

    def test_full_obstruction_parity(self, location, noon_weather, gpu_available, tmp_path):
        """All patches blocked: GPU and CPU should produce matching low-radiation results."""
        from conftest import read_timestep_geotiff

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

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, noon_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, noon_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with full obstruction",
        )

    def test_night_time_parity(self, location, gpu_available, tmp_path):
        """Night time (sun below horizon): GPU and CPU should match."""
        from conftest import read_timestep_geotiff

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

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, night_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, night_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values even at night"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch at night time",
        )

        # Verify kdown is zero or near-zero at night
        kdown_gpu = read_timestep_geotiff(out_gpu, "kdown", 0)
        kdown_valid = ~np.isnan(kdown_gpu)
        if np.any(kdown_valid):
            assert np.nanmax(kdown_gpu) < 1.0, "kdown should be ~0 at night"

    def test_sitting_posture_parity(self, location, noon_weather, gpu_available, tmp_path):
        """Sitting posture (cyl=False): GPU short-circuits to zero, CPU should match."""
        from conftest import read_timestep_geotiff

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

        out_gpu = tmp_path / "gpu"
        calculate(
            surface_gpu,
            [noon_weather],
            location,
            use_anisotropic_sky=True,
            human=sitting,
            output_dir=out_gpu,
            outputs=["tmrt", "kdown"],
        )
        pipeline.disable_aniso_gpu()
        out_cpu = tmp_path / "cpu"
        calculate(
            surface_cpu,
            [noon_weather],
            location,
            use_anisotropic_sky=True,
            human=sitting,
            output_dir=out_cpu,
            outputs=["tmrt", "kdown"],
        )

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with sitting posture (cyl=False)",
        )

    def test_zero_radiation_parity(self, location, gpu_available, tmp_path):
        """Zero radiation input: GPU and CPU should match."""
        from conftest import read_timestep_geotiff

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

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, zero_rad_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, zero_rad_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        valid = ~np.isnan(tmrt_gpu) & ~np.isnan(tmrt_cpu)
        assert np.any(valid), "Should have valid Tmrt values"

        np.testing.assert_allclose(
            tmrt_gpu[valid],
            tmrt_cpu[valid],
            rtol=1e-3,
            atol=0.5,
            err_msg="GPU vs CPU Tmrt mismatch with zero radiation",
        )

    def test_invalid_pixels_nan_parity(self, location, noon_weather, gpu_available, tmp_path):
        """Invalid pixels (NaN DSM) produce NaN in both GPU and CPU paths."""
        from conftest import make_mock_svf, read_timestep_geotiff

        if not gpu_available:
            pytest.skip("GPU not available")

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

        out_gpu = tmp_path / "gpu"
        out_cpu = tmp_path / "cpu"
        self._run_with_gpu(surface_gpu, location, noon_weather, out_gpu, gpu_on=True)
        self._run_with_gpu(surface_cpu, location, noon_weather, out_cpu, gpu_on=False)

        tmrt_gpu = read_timestep_geotiff(out_gpu, "tmrt", 0)
        tmrt_cpu = read_timestep_geotiff(out_cpu, "tmrt", 0)

        # NaN pixels should be NaN in both
        gpu_nan = np.isnan(tmrt_gpu)
        cpu_nan = np.isnan(tmrt_cpu)
        np.testing.assert_array_equal(
            gpu_nan,
            cpu_nan,
            err_msg="GPU and CPU should produce NaN at the same pixels",
        )

        # Valid pixels should match
        valid = ~gpu_nan & ~cpu_nan
        if np.any(valid):
            np.testing.assert_allclose(
                tmrt_gpu[valid],
                tmrt_cpu[valid],
                rtol=1e-3,
                atol=0.5,
                err_msg="GPU vs CPU Tmrt mismatch on valid pixels with NaN neighbors",
            )

    def test_gpu_fallback_when_disabled(self, location, noon_weather, tmp_path):
        """With GPU disabled, results are identical to CPU-only path."""
        from conftest import read_timestep_geotiff

        surface_a = _make_flat_surface_with_shadows()
        surface_b = _make_flat_surface_with_shadows()

        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        self._run_with_gpu(surface_a, location, noon_weather, out_a, gpu_on=False)
        self._run_with_gpu(surface_b, location, noon_weather, out_b, gpu_on=False)

        tmrt_a = read_timestep_geotiff(out_a, "tmrt", 0)
        tmrt_b = read_timestep_geotiff(out_b, "tmrt", 0)

        valid = ~np.isnan(tmrt_a) & ~np.isnan(tmrt_b)
        if np.any(valid):
            np.testing.assert_array_equal(
                tmrt_a[valid],
                tmrt_b[valid],
                err_msg="Two CPU-only runs should produce identical results",
            )
