"""Precomputed data models (SVF, shadow matrices)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np

from ..logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class SvfArrays:
    """
    Pre-computed Sky View Factor arrays.

    Use this when you have already computed SVF and want to skip
    re-computation. Can be loaded from SOLWEIG svfs.zip format.

    Attributes:
        svf: Total sky view factor (0-1).
        svf_north, svf_east, svf_south, svf_west: Directional SVF components.
        svf_veg: Vegetation SVF (set to ones if no vegetation).
        svf_veg_north, svf_veg_east, svf_veg_south, svf_veg_west: Directional veg SVF.
        svf_aveg: Vegetation blocking building shadow.
        svf_aveg_north, svf_aveg_east, svf_aveg_south, svf_aveg_west: Directional.

    Memory note:
        All arrays are stored as float32. For a 768x768 grid with all 15 arrays,
        total memory is approximately 35 MB.
    """

    svf: NDArray[np.floating]
    svf_north: NDArray[np.floating]
    svf_east: NDArray[np.floating]
    svf_south: NDArray[np.floating]
    svf_west: NDArray[np.floating]
    svf_veg: NDArray[np.floating]
    svf_veg_north: NDArray[np.floating]
    svf_veg_east: NDArray[np.floating]
    svf_veg_south: NDArray[np.floating]
    svf_veg_west: NDArray[np.floating]
    svf_aveg: NDArray[np.floating]
    svf_aveg_north: NDArray[np.floating]
    svf_aveg_east: NDArray[np.floating]
    svf_aveg_south: NDArray[np.floating]
    svf_aveg_west: NDArray[np.floating]

    def __post_init__(self):
        # Ensure all arrays are float32 for memory efficiency
        # Note: np.asarray preserves memmap arrays (doesn't copy unless dtype changes)
        def ensure_f32(arr):
            if isinstance(arr, np.memmap):
                # Preserve memmap - only convert dtype if needed
                if arr.dtype != np.float32:
                    # This would load into memory - warn user
                    logger.warning("Memmap array has wrong dtype, loading into memory")
                    return np.asarray(arr, dtype=np.float32)
                return arr
            return np.asarray(arr, dtype=np.float32)

        self.svf = ensure_f32(self.svf)
        self.svf_north = ensure_f32(self.svf_north)
        self.svf_east = ensure_f32(self.svf_east)
        self.svf_south = ensure_f32(self.svf_south)
        self.svf_west = ensure_f32(self.svf_west)
        self.svf_veg = ensure_f32(self.svf_veg)
        self.svf_veg_north = ensure_f32(self.svf_veg_north)
        self.svf_veg_east = ensure_f32(self.svf_veg_east)
        self.svf_veg_south = ensure_f32(self.svf_veg_south)
        self.svf_veg_west = ensure_f32(self.svf_veg_west)
        self.svf_aveg = ensure_f32(self.svf_aveg)
        self.svf_aveg_north = ensure_f32(self.svf_aveg_north)
        self.svf_aveg_east = ensure_f32(self.svf_aveg_east)
        self.svf_aveg_south = ensure_f32(self.svf_aveg_south)
        self.svf_aveg_west = ensure_f32(self.svf_aveg_west)

    @property
    def svfalfa(self) -> NDArray[np.floating]:
        """Compute SVF alpha (angle) from SVF values. Computed on-demand."""
        tmp = self.svf + self.svf_veg - 1.0
        tmp = np.clip(tmp, 0.0, 1.0)
        eps = np.finfo(np.float32).tiny
        safe_term = np.clip(1.0 - tmp, eps, 1.0)
        return np.arcsin(np.exp(np.log(safe_term) / 2.0))

    @property
    def svfbuveg(self) -> NDArray[np.floating]:
        """Combined building + vegetation SVF. Computed on-demand."""
        return np.clip(self.svf + self.svf_veg - 1.0, 0.0, 1.0)

    @classmethod
    def from_zip(cls, zip_path: str | Path, use_vegetation: bool = True) -> SvfArrays:
        """
        Load SVF arrays from SOLWEIG svfs.zip format.

        Args:
            zip_path: Path to svfs.zip file.
            use_vegetation: Whether to load vegetation SVF arrays. Default True.

        Returns:
            SvfArrays instance with loaded data.

        Memory note:
            Files are extracted temporarily and loaded as float32 arrays.
            The zip file contains GeoTIFF rasters.
        """
        import tempfile
        import zipfile

        from . import io as common

        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"SVF zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(tmpdir)

            tmppath = Path(tmpdir)

            def load(filename: str) -> NDArray[np.floating]:
                filepath = tmppath / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Expected SVF file not found in zip: {filename}")
                data, _, _, _ = common.load_raster(str(filepath), coerce_f64_to_f32=True)
                return data

            # Load basic SVF arrays
            svf = load("svf.tif")
            svf_n = load("svfN.tif")
            svf_e = load("svfE.tif")
            svf_s = load("svfS.tif")
            svf_w = load("svfW.tif")

            # Load vegetation arrays or create defaults
            if use_vegetation:
                svf_veg = load("svfveg.tif")
                svf_veg_n = load("svfNveg.tif")
                svf_veg_e = load("svfEveg.tif")
                svf_veg_s = load("svfSveg.tif")
                svf_veg_w = load("svfWveg.tif")
                svf_aveg = load("svfaveg.tif")
                svf_aveg_n = load("svfNaveg.tif")
                svf_aveg_e = load("svfEaveg.tif")
                svf_aveg_s = load("svfSaveg.tif")
                svf_aveg_w = load("svfWaveg.tif")
            else:
                ones = np.ones_like(svf)
                svf_veg = ones
                svf_veg_n = ones
                svf_veg_e = ones
                svf_veg_s = ones
                svf_veg_w = ones
                svf_aveg = ones
                svf_aveg_n = ones
                svf_aveg_e = ones
                svf_aveg_s = ones
                svf_aveg_w = ones

        return cls(
            svf=svf,
            svf_north=svf_n,
            svf_east=svf_e,
            svf_south=svf_s,
            svf_west=svf_w,
            svf_veg=svf_veg,
            svf_veg_north=svf_veg_n,
            svf_veg_east=svf_veg_e,
            svf_veg_south=svf_veg_s,
            svf_veg_west=svf_veg_w,
            svf_aveg=svf_aveg,
            svf_aveg_north=svf_aveg_n,
            svf_aveg_east=svf_aveg_e,
            svf_aveg_south=svf_aveg_s,
            svf_aveg_west=svf_aveg_w,
        )

    def to_memmap(self, directory: str | Path) -> Path:
        """
        Save SVF arrays as memory-mapped .npy files for efficient large-raster processing.

        This enables processing of 10k×10k+ rasters without loading all SVF data into RAM.
        The OS handles paging, loading only the needed regions into physical memory.

        Args:
            directory: Directory to save memmap files. Created if doesn't exist.

        Returns:
            Path to the directory containing memmap files.

        Memory note:
            For a 10k×10k grid with 15 arrays: ~6 GB on disk, but only accessed
            regions are loaded into RAM. Typical usage loads <100 MB.

        Example:
            svf = SvfArrays.from_zip("svfs.zip")
            svf.to_memmap("cache/svf_memmap")
            # Later:
            svf = SvfArrays.from_memmap("cache/svf_memmap")
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save each array as .npy file
        array_names = [
            "svf", "svf_north", "svf_east", "svf_south", "svf_west",
            "svf_veg", "svf_veg_north", "svf_veg_east", "svf_veg_south", "svf_veg_west",
            "svf_aveg", "svf_aveg_north", "svf_aveg_east", "svf_aveg_south", "svf_aveg_west",
        ]

        for name in array_names:
            arr = getattr(self, name)
            np.save(directory / f"{name}.npy", arr)

        logger.info(f"Saved SVF memmap cache to {directory} ({len(array_names)} arrays)")
        return directory

    @classmethod
    def from_memmap(cls, directory: str | Path, mode: str = "r") -> "SvfArrays":
        """
        Load SVF arrays as memory-mapped files for efficient large-raster processing.

        Memory-mapped arrays are not loaded into RAM until accessed. The OS handles
        paging, making this suitable for rasters larger than available RAM.

        Args:
            directory: Directory containing memmap .npy files (from to_memmap()).
            mode: Memory-map mode. Default "r" (read-only).
                - "r": Read-only (safest, allows OS caching)
                - "r+": Read-write (modifications saved to disk)
                - "c": Copy-on-write (modifications not saved)

        Returns:
            SvfArrays with memory-mapped backing.

        Memory note:
            Only accessed regions are loaded into physical RAM. For tiled processing,
            this dramatically reduces memory usage compared to loading full arrays.

        Example:
            svf = SvfArrays.from_memmap("cache/svf_memmap")
            # Arrays are loaded on-demand as tiles access them
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"SVF memmap directory not found: {directory}")

        def load_memmap(name: str) -> np.ndarray:
            path = directory / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"SVF memmap file not found: {path}")
            return np.load(path, mmap_mode=mode)

        return cls(
            svf=load_memmap("svf"),
            svf_north=load_memmap("svf_north"),
            svf_east=load_memmap("svf_east"),
            svf_south=load_memmap("svf_south"),
            svf_west=load_memmap("svf_west"),
            svf_veg=load_memmap("svf_veg"),
            svf_veg_north=load_memmap("svf_veg_north"),
            svf_veg_east=load_memmap("svf_veg_east"),
            svf_veg_south=load_memmap("svf_veg_south"),
            svf_veg_west=load_memmap("svf_veg_west"),
            svf_aveg=load_memmap("svf_aveg"),
            svf_aveg_north=load_memmap("svf_aveg_north"),
            svf_aveg_east=load_memmap("svf_aveg_east"),
            svf_aveg_south=load_memmap("svf_aveg_south"),
            svf_aveg_west=load_memmap("svf_aveg_west"),
        )


@dataclass
class ShadowArrays:
    """
    Pre-computed anisotropic shadow matrices for sky patch calculations.

    These are 3D arrays of shape (rows, cols, patches) where patches is
    typically 145, 153, 306, or 612 depending on the resolution.

    Memory optimization:
        Internally stored as uint8 (0-255 representing 0.0-1.0) to reduce
        memory by 75%. Converted to float32 only when accessed via properties.
        Cached after first access to avoid repeated allocations.
        For a 400x400 grid with 153 patches: 24.5 MB as uint8 vs 98 MB as float32.

    Attributes:
        _shmat_u8: Building shadow matrix (uint8 storage).
        _vegshmat_u8: Vegetation shadow matrix (uint8 storage).
        _vbshmat_u8: Combined veg+building shadow matrix (uint8 storage).
        patch_count: Number of sky patches (145, 153, 306, or 612).
    """

    _shmat_u8: NDArray[np.uint8]
    _vegshmat_u8: NDArray[np.uint8]
    _vbshmat_u8: NDArray[np.uint8]
    patch_count: int = field(init=False)
    # Cache for converted float32 arrays (allocated on first access)
    _shmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vegshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vbshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Ensure uint8 storage
        if self._shmat_u8.dtype != np.uint8:
            self._shmat_u8 = (np.clip(self._shmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vegshmat_u8.dtype != np.uint8:
            self._vegshmat_u8 = (np.clip(self._vegshmat_u8, 0, 1) * 255).astype(np.uint8)
        if self._vbshmat_u8.dtype != np.uint8:
            self._vbshmat_u8 = (np.clip(self._vbshmat_u8, 0, 1) * 255).astype(np.uint8)

        self.patch_count = self._shmat_u8.shape[2]
        # Initialize cache as None (lazy allocation)
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None

    @property
    def shmat(self) -> NDArray[np.floating]:
        """
        Building shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._shmat_f32 is None:
            self._shmat_f32 = self._shmat_u8.astype(np.float32) / 255.0
        return self._shmat_f32

    @property
    def vegshmat(self) -> NDArray[np.floating]:
        """
        Vegetation shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._vegshmat_f32 is None:
            self._vegshmat_f32 = self._vegshmat_u8.astype(np.float32) / 255.0
        return self._vegshmat_f32

    @property
    def vbshmat(self) -> NDArray[np.floating]:
        """
        Combined shadow matrix as float32 (0.0-1.0).

        Cached after first access to avoid repeated allocations (~98MB for 400x400 grid).
        """
        if self._vbshmat_f32 is None:
            self._vbshmat_f32 = self._vbshmat_u8.astype(np.float32) / 255.0
        return self._vbshmat_f32

    @property
    def patch_option(self) -> int:
        """Patch option code (1=145, 2=153, 3=306, 4=612 patches)."""
        patch_map = {145: 1, 153: 2, 306: 3, 612: 4}
        return patch_map.get(self.patch_count, 2)

    def diffsh(self, transmissivity: float = 0.03, use_vegetation: bool = True) -> NDArray[np.floating]:
        """
        Compute diffuse shadow matrix.

        Args:
            transmissivity: Vegetation transmissivity (default 0.03).
            use_vegetation: Whether to account for vegetation.

        Returns:
            Diffuse shadow matrix as float32.
        """
        shmat = self.shmat
        if use_vegetation:
            vegshmat = self.vegshmat
            return (shmat - (1 - vegshmat) * (1 - transmissivity)).astype(np.float32)
        return shmat

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> ShadowArrays:
        """
        Load shadow matrices from SOLWEIG shadowmats.npz format.

        Args:
            npz_path: Path to shadowmats.npz file.

        Returns:
            ShadowArrays instance with loaded data.

        Memory note:
            The npz file typically contains uint8 arrays (compressed).
            Data is kept as uint8 internally; conversion to float32
            happens only when arrays are accessed via properties.
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Shadow matrices file not found: {npz_path}")

        data = np.load(str(npz_path))

        # Load arrays - they should be uint8 in the optimized format
        shmat = data["shadowmat"]
        vegshmat = data["vegshadowmat"]
        vbshmat = data["vbshmat"]

        # Convert to uint8 if not already (legacy float32 format)
        if shmat.dtype != np.uint8:
            shmat = (np.clip(shmat, 0, 1) * 255).astype(np.uint8)
        if vegshmat.dtype != np.uint8:
            vegshmat = (np.clip(vegshmat, 0, 1) * 255).astype(np.uint8)
        if vbshmat.dtype != np.uint8:
            vbshmat = (np.clip(vbshmat, 0, 1) * 255).astype(np.uint8)

        return cls(
            _shmat_u8=shmat,
            _vegshmat_u8=vegshmat,
            _vbshmat_u8=vbshmat,
        )



@dataclass
class PrecomputedData:
    """
    Container for pre-computed preprocessing data to skip expensive calculations.

    Use this to provide already-computed walls, SVF, and/or shadow matrices
    to the calculate() function. This is useful when:
    - Running multiple timesteps with the same geometry
    - Using data generated by external tools
    - Optimizing performance by pre-computing once

    Attributes:
        wall_height: Pre-computed wall height grid (meters). If None, computed on-the-fly.
        wall_aspect: Pre-computed wall aspect grid (degrees, 0=N). If None, computed on-the-fly.
        svf: Pre-computed SVF arrays. If None, SVF is computed on-the-fly.
        shadow_matrices: Pre-computed anisotropic shadow matrices.
            If None, isotropic sky model is used.

    Example:
        # Load all preprocessing
        precomputed = PrecomputedData.load(
            walls_dir="preprocessed/walls",
            svf_dir="preprocessed/svf",
        )

        # Or create manually
        svf = SvfArrays.from_zip("path/to/svfs.zip")
        shadows = ShadowArrays.from_npz("path/to/shadowmats.npz")
        precomputed = PrecomputedData(svf=svf, shadow_matrices=shadows)

        result = calculate(
            surface=surface,
            location=location,
            weather=weather,
            precomputed=precomputed,
        )
    """

    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None

    @classmethod
    def prepare(
        cls,
        walls_dir: str | Path | None = None,
        svf_dir: str | Path | None = None,
    ) -> PrecomputedData:
        """
        Prepare preprocessing data from directories.

        Loads preprocessing files if they exist. If files don't exist,
        the corresponding data will be None and computed on-the-fly during calculation.

        All parameters are optional.

        Args:
            walls_dir: Directory containing wall preprocessing files:
                - wall_hts.tif: Wall heights (meters)
                - wall_aspects.tif: Wall aspects (degrees, 0=N)
            svf_dir: Directory containing SVF preprocessing files:
                - svfs.zip: SVF arrays (required if svf_dir provided)
                - shadowmats.npz: Shadow matrices for anisotropic sky (optional)

        Returns:
            PrecomputedData with loaded arrays. Missing data is set to None.

        Example:
            # Prepare all preprocessing
            precomputed = PrecomputedData.prepare(
                walls_dir="preprocessed/walls",
                svf_dir="preprocessed/svf",
            )

            # Prepare only SVF
            precomputed = PrecomputedData.prepare(svf_dir="preprocessed/svf")

            # Nothing prepared (all computed on-the-fly)
            precomputed = PrecomputedData.prepare()
        """
        from . import io

        wall_height_arr = None
        wall_aspect_arr = None
        svf_arrays = None
        shadow_arrays = None

        # Load walls if directory provided
        if walls_dir is not None:
            walls_path = Path(walls_dir)
            wall_height_path = walls_path / "wall_hts.tif"
            wall_aspect_path = walls_path / "wall_aspects.tif"

            if wall_height_path.exists():
                wall_height_arr, _, _, _ = io.load_raster(str(wall_height_path))
                logger.info(f"  Loaded wall heights from {walls_dir}")
            else:
                logger.debug(f"  Wall heights not found: {wall_height_path}")

            if wall_aspect_path.exists():
                wall_aspect_arr, _, _, _ = io.load_raster(str(wall_aspect_path))
                logger.info(f"  Loaded wall aspects from {walls_dir}")
            else:
                logger.debug(f"  Wall aspects not found: {wall_aspect_path}")

        # Load SVF if directory provided
        if svf_dir is not None:
            svf_path = Path(svf_dir)
            svf_zip = svf_path / "svfs.zip"

            if svf_zip.exists():
                svf_arrays = SvfArrays.from_zip(str(svf_zip))
                logger.info(f"  Loaded SVF data: {svf_arrays.svf.shape}")
            else:
                logger.debug(f"  SVF not found: {svf_zip}")

            # Load shadow matrices (optional - for anisotropic sky)
            shadow_npz = svf_path / "shadowmats.npz"
            if shadow_npz.exists():
                shadow_arrays = ShadowArrays.from_npz(str(shadow_npz))
                logger.info("  Loaded shadow matrices for anisotropic sky")
            else:
                logger.debug("  No shadow matrices found (anisotropic sky will be slower)")

        return cls(
            wall_height=wall_height_arr,
            wall_aspect=wall_aspect_arr,
            svf=svf_arrays,
            shadow_matrices=shadow_arrays,
        )


