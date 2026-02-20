"""Precomputed preprocessing data (SVF arrays and shadow matrices).

Defines :class:`SvfArrays` (15 directional sky view factor grids),
:class:`ShadowArrays` (bitpacked shadow matrices for the anisotropic
sky model), and :class:`PrecomputedData` (a convenience wrapper that
bundles both).  These can be loaded from the ``svfs.zip`` /
``shadowmats.npz`` files produced by :meth:`SurfaceData.prepare`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..cache import CacheMetadata, pixel_size_tag
from ..solweig_logging import get_logger

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

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> SvfArrays:
        """Crop all SVF arrays to [r0:r1, c0:c1]."""
        return SvfArrays(
            svf=self.svf[r0:r1, c0:c1].copy(),
            svf_north=self.svf_north[r0:r1, c0:c1].copy(),
            svf_east=self.svf_east[r0:r1, c0:c1].copy(),
            svf_south=self.svf_south[r0:r1, c0:c1].copy(),
            svf_west=self.svf_west[r0:r1, c0:c1].copy(),
            svf_veg=self.svf_veg[r0:r1, c0:c1].copy(),
            svf_veg_north=self.svf_veg_north[r0:r1, c0:c1].copy(),
            svf_veg_east=self.svf_veg_east[r0:r1, c0:c1].copy(),
            svf_veg_south=self.svf_veg_south[r0:r1, c0:c1].copy(),
            svf_veg_west=self.svf_veg_west[r0:r1, c0:c1].copy(),
            svf_aveg=self.svf_aveg[r0:r1, c0:c1].copy(),
            svf_aveg_north=self.svf_aveg_north[r0:r1, c0:c1].copy(),
            svf_aveg_east=self.svf_aveg_east[r0:r1, c0:c1].copy(),
            svf_aveg_south=self.svf_aveg_south[r0:r1, c0:c1].copy(),
            svf_aveg_west=self.svf_aveg_west[r0:r1, c0:c1].copy(),
        )

    @classmethod
    def from_bundle(cls, bundle) -> SvfArrays:
        """
        Create SvfArrays from a SvfBundle (computation result).

        This enables caching fresh-computed SVF back to surface.svf for reuse.

        Args:
            bundle: SvfBundle from resolve_svf() or skyview.calculate_svf()

        Returns:
            SvfArrays instance suitable for caching on SurfaceData.svf
        """
        return cls(
            svf=bundle.svf,
            svf_north=bundle.svf_directional.north,
            svf_east=bundle.svf_directional.east,
            svf_south=bundle.svf_directional.south,
            svf_west=bundle.svf_directional.west,
            svf_veg=bundle.svf_veg,
            svf_veg_north=bundle.svf_veg_directional.north,
            svf_veg_east=bundle.svf_veg_directional.east,
            svf_veg_south=bundle.svf_veg_directional.south,
            svf_veg_west=bundle.svf_veg_directional.west,
            svf_aveg=bundle.svf_aveg,
            svf_aveg_north=bundle.svf_aveg_directional.north,
            svf_aveg_east=bundle.svf_aveg_directional.east,
            svf_aveg_south=bundle.svf_aveg_directional.south,
            svf_aveg_west=bundle.svf_aveg_directional.west,
        )

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

        from .. import io as common

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
                data, _, _, _ = common.load_raster(str(filepath), ensure_float32=True)
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

    def to_memmap(self, directory: str | Path, metadata: CacheMetadata | None = None) -> Path:
        """
        Save SVF arrays as memory-mapped .npy files for efficient large-raster processing.

        This enables processing of 10k×10k+ rasters without loading all SVF data into RAM.
        The OS handles paging, loading only the needed regions into physical memory.

        Args:
            directory: Directory to save memmap files. Created if doesn't exist.
            metadata: Optional cache metadata for validation on reload.
                When provided, enables automatic cache invalidation if inputs change.

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
            "svf",
            "svf_north",
            "svf_east",
            "svf_south",
            "svf_west",
            "svf_veg",
            "svf_veg_north",
            "svf_veg_east",
            "svf_veg_south",
            "svf_veg_west",
            "svf_aveg",
            "svf_aveg_north",
            "svf_aveg_east",
            "svf_aveg_south",
            "svf_aveg_west",
        ]

        for name in array_names:
            arr = getattr(self, name)
            np.save(directory / f"{name}.npy", arr)

        # Save metadata for cache validation
        if metadata is not None:
            metadata.save(directory)

        logger.info(f"Saved SVF memmap cache to {directory} ({len(array_names)} arrays)")
        return directory

    @classmethod
    def from_memmap(cls, directory: str | Path, mode: Literal["r", "r+", "c"] = "r") -> SvfArrays:
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


def _unpack_bitpacked_to_float32(packed: NDArray[np.uint8], patch_count: int) -> NDArray[np.floating]:
    """Unpack bitpacked shadow matrix to float32 (0.0 or 1.0).

    Args:
        packed: Bitpacked array, shape (rows, cols, n_pack) where n_pack = ceil(patch_count/8).
        patch_count: Number of actual patches.

    Returns:
        Float32 array, shape (rows, cols, patch_count) with values 0.0 or 1.0.
    """
    rows, cols, _ = packed.shape
    result = np.zeros((rows, cols, patch_count), dtype=np.float32)
    for p in range(patch_count):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        result[:, :, p] = ((packed[:, :, byte_idx] & bit_mask) != 0).astype(np.float32)
    return result


def _pack_u8_to_bitpacked(
    u8_data: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Pack u8 shadow matrix (0 or 255 per patch) to bitpacked format.

    Args:
        u8_data: Array shape (rows, cols, patch_count) with values 0 or 255.

    Returns:
        Bitpacked array, shape (rows, cols, n_pack) where n_pack = ceil(patch_count/8).
    """
    rows, cols, patch_count = u8_data.shape
    n_pack = (patch_count + 7) // 8
    packed = np.zeros((rows, cols, n_pack), dtype=np.uint8)
    for p in range(patch_count):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        packed[:, :, byte_idx] |= np.where(u8_data[:, :, p] >= 128, bit_mask, np.uint8(0))
    return packed


@dataclass
class ShadowArrays:
    """
    Pre-computed anisotropic shadow matrices for sky patch calculations.

    Internally stored as bitpacked uint8 arrays of shape (rows, cols, n_pack)
    where n_pack = ceil(patch_count / 8). Each bit represents one sky patch
    (1 = sky visible / shadowed value was 255, 0 = blocked / was 0).

    Memory optimization:
        Bitpacking stores 8 patches per byte instead of 1, reducing memory 7.6x.
        For a 2500x2500 grid with 153 patches: 375 MB bitpacked vs 2.87 GB as uint8.
        Converted to float32 only when accessed via properties (e.g. for diffsh).

    Attributes:
        _shmat_u8: Building shadow matrix (bitpacked uint8).
        _vegshmat_u8: Vegetation shadow matrix (bitpacked uint8).
        _vbshmat_u8: Combined veg+building shadow matrix (bitpacked uint8).
        patch_count: Number of sky patches (145, 153, 306, or 612).
    """

    _shmat_u8: NDArray[np.uint8]
    _vegshmat_u8: NDArray[np.uint8]
    _vbshmat_u8: NDArray[np.uint8]
    _n_patches: int = 153
    patch_count: int = field(init=False)
    # Cache for converted float32 arrays (allocated on first access)
    _shmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vegshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vbshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _steradians: NDArray[np.float32] | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Ensure uint8 dtype
        if self._shmat_u8.dtype != np.uint8:
            self._shmat_u8 = self._shmat_u8.astype(np.uint8)
        if self._vegshmat_u8.dtype != np.uint8:
            self._vegshmat_u8 = self._vegshmat_u8.astype(np.uint8)
        if self._vbshmat_u8.dtype != np.uint8:
            self._vbshmat_u8 = self._vbshmat_u8.astype(np.uint8)

        self.patch_count = self._n_patches
        # Initialize cache as None (lazy allocation)
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None
        self._steradians = None

    @property
    def shmat(self) -> NDArray[np.floating]:
        """Building shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._shmat_f32 is None:
            self._shmat_f32 = _unpack_bitpacked_to_float32(self._shmat_u8, self.patch_count)
        return self._shmat_f32

    @property
    def vegshmat(self) -> NDArray[np.floating]:
        """Vegetation shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._vegshmat_f32 is None:
            self._vegshmat_f32 = _unpack_bitpacked_to_float32(self._vegshmat_u8, self.patch_count)
        return self._vegshmat_f32

    @property
    def vbshmat(self) -> NDArray[np.floating]:
        """Combined shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._vbshmat_f32 is None:
            self._vbshmat_f32 = _unpack_bitpacked_to_float32(self._vbshmat_u8, self.patch_count)
        return self._vbshmat_f32

    @property
    def patch_option(self) -> int:
        """Patch option code (1=145, 2=153, 3=306, 4=612 patches)."""
        patch_map = {145: 1, 153: 2, 306: 3, 612: 4}
        return patch_map.get(self.patch_count, 2)

    @property
    def steradians(self) -> NDArray[np.float32]:
        """Patch steradians (cached, depends only on patch layout)."""
        if self._steradians is None:
            from ..physics.create_patches import create_patches
            from ..physics.patch_radiation import patch_steradians

            skyvaultalt, skyvaultazi, *_ = create_patches(self.patch_option)
            # patch_steradians only uses column 0 (altitudes)
            lv_stub = np.column_stack([skyvaultalt.ravel(), skyvaultazi.ravel(), np.zeros(skyvaultalt.size)])
            self._steradians, _, _ = patch_steradians(lv_stub)
        return self._steradians

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

    def release_float32_cache(self) -> None:
        """Release cached float32 shadow matrices to free memory.

        The bitpacked originals remain available. Future property access will
        re-unpack as needed.
        """
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> ShadowArrays:
        """Crop all shadow matrices to [r0:r1, c0:c1] (3D: rows, cols, n_pack)."""
        return ShadowArrays(
            _shmat_u8=self._shmat_u8[r0:r1, c0:c1, :].copy(),
            _vegshmat_u8=self._vegshmat_u8[r0:r1, c0:c1, :].copy(),
            _vbshmat_u8=self._vbshmat_u8[r0:r1, c0:c1, :].copy(),
            _n_patches=self.patch_count,
        )

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> ShadowArrays:
        """
        Load shadow matrices from SOLWEIG shadowmats.npz format.

        Handles both legacy u8-per-patch format and new bitpacked format.
        Legacy files have shape[2] matching patch count (145/153/306/612).
        New files include a 'patch_count' metadata key.
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Shadow matrices file not found: {npz_path}")

        data = np.load(str(npz_path))

        shmat = data["shadowmat"]
        vegshmat = data["vegshadowmat"]
        vbshmat = data["vbshmat"]

        # Detect format: new bitpacked files include 'patch_count' key
        if "patch_count" in data:
            patch_count = int(data["patch_count"])
            # Data is already bitpacked uint8
            return cls(
                _shmat_u8=shmat.astype(np.uint8),
                _vegshmat_u8=vegshmat.astype(np.uint8),
                _vbshmat_u8=vbshmat.astype(np.uint8),
                _n_patches=patch_count,
            )

        # Legacy format: shape[2] == patch_count, values are 0/255 uint8 or 0.0/1.0 float32
        # Convert float32 → uint8 first if needed
        if shmat.dtype != np.uint8:
            shmat = (np.clip(shmat, 0, 1) * 255).astype(np.uint8)
        if vegshmat.dtype != np.uint8:
            vegshmat = (np.clip(vegshmat, 0, 1) * 255).astype(np.uint8)
        if vbshmat.dtype != np.uint8:
            vbshmat = (np.clip(vbshmat, 0, 1) * 255).astype(np.uint8)

        patch_count = shmat.shape[2]

        # Pack u8 → bitpacked
        return cls(
            _shmat_u8=_pack_u8_to_bitpacked(shmat),
            _vegshmat_u8=_pack_u8_to_bitpacked(vegshmat),
            _vbshmat_u8=_pack_u8_to_bitpacked(vbshmat),
            _n_patches=patch_count,
        )

    @classmethod
    def from_memmap(cls, directory: str | Path, mode: Literal["r", "r+", "c"] = "r") -> ShadowArrays:
        """
        Load bitpacked shadow matrices from a memmap directory.

        Expected files:
            - metadata.json (shape, patch_count, file names)
            - shmat.dat
            - vegshmat.dat
            - vbshmat.dat
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Shadow memmap directory not found: {directory}")

        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Shadow memmap metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        shape_raw = meta.get("shape")
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
            raise ValueError(f"Invalid shadow memmap shape metadata in {metadata_path}: {shape_raw}")
        shape = tuple(int(v) for v in shape_raw)
        patch_count = int(meta.get("patch_count", 153))

        sh_file = meta.get("shadowmat_file", "shmat.dat")
        veg_file = meta.get("vegshadowmat_file", "vegshmat.dat")
        vb_file = meta.get("vbshmat_file", "vbshmat.dat")

        sh_path = directory / sh_file
        veg_path = directory / veg_file
        vb_path = directory / vb_file
        for path in (sh_path, veg_path, vb_path):
            if not path.exists():
                raise FileNotFoundError(f"Expected shadow memmap file not found: {path}")

        return cls(
            _shmat_u8=np.memmap(sh_path, dtype=np.uint8, mode=mode, shape=shape),
            _vegshmat_u8=np.memmap(veg_path, dtype=np.uint8, mode=mode, shape=shape),
            _vbshmat_u8=np.memmap(vb_path, dtype=np.uint8, mode=mode, shape=shape),
            _n_patches=patch_count,
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
        wall_height: Pre-computed wall height grid (meters). If None, wall height
            can be prepared from DSM via SurfaceData.prepare().
        wall_aspect: Pre-computed wall aspect grid (degrees, 0=N). If None, wall aspect
            can be prepared from DSM via SurfaceData.prepare().
        svf: Pre-computed SVF arrays. Required for calculate(); if None,
            calculate() raises MissingPrecomputedData.
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
        the corresponding data will be None.

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

            # Nothing prepared (SVF must be provided before calculate())
            precomputed = PrecomputedData.prepare()
        """
        from .. import io

        wall_height_arr = None
        wall_aspect_arr = None
        svf_arrays = None
        shadow_arrays = None

        def _load_svf_from_dir(base: Path) -> SvfArrays | None:
            memmap_dir = base / "memmap"
            svf_zip = base / "svfs.zip"
            if memmap_dir.exists() and (memmap_dir / "svf.npy").exists():
                logger.info(f"  Loaded SVF memmap cache from {memmap_dir}")
                return SvfArrays.from_memmap(memmap_dir)
            if svf_zip.exists():
                logger.info(f"  Loaded SVF zip from {svf_zip}")
                return SvfArrays.from_zip(str(svf_zip))
            return None

        def _load_shadow_from_dir(base: Path) -> ShadowArrays | None:
            shadow_npz = base / "shadowmats.npz"
            if shadow_npz.exists():
                logger.info(f"  Loaded shadow matrices from {shadow_npz}")
                return ShadowArrays.from_npz(str(shadow_npz))

            shadow_memmap_dir = base / "shadow_memmaps"
            metadata = shadow_memmap_dir / "metadata.json"
            if shadow_memmap_dir.exists() and metadata.exists():
                logger.info(f"  Loaded shadow memmaps from {shadow_memmap_dir}")
                return ShadowArrays.from_memmap(shadow_memmap_dir)
            return None

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
            svf_arrays = _load_svf_from_dir(svf_path)
            shadow_arrays = _load_shadow_from_dir(svf_path)

            # Fallback: look for pixel-size-keyed cache under svf/<tag>/ when
            # caller points at a prepared surface directory root.
            if svf_arrays is None or shadow_arrays is None:
                candidate_dirs: list[Path] = []
                meta_path = svf_path / "metadata.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as f:
                            meta = json.load(f)
                        px = meta.get("pixel_size")
                        if px is not None:
                            candidate_dirs.append(svf_path / "svf" / pixel_size_tag(float(px)))
                    except Exception:
                        pass

                svf_root = svf_path / "svf"
                if svf_root.exists():
                    for child in svf_root.iterdir():
                        if child.is_dir():
                            candidate_dirs.append(child)

                seen: set[Path] = set()
                for candidate in candidate_dirs:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    if svf_arrays is None:
                        svf_arrays = _load_svf_from_dir(candidate)
                    if shadow_arrays is None:
                        shadow_arrays = _load_shadow_from_dir(candidate)
                    if svf_arrays is not None and shadow_arrays is not None:
                        break

            if svf_arrays is None:
                logger.debug(f"  SVF not found in {svf_path}")
            else:
                logger.info(f"  Loaded SVF data: {svf_arrays.svf.shape}")

            if shadow_arrays is None:
                logger.debug("  No shadow matrices found (anisotropic sky will be slower)")
            else:
                logger.info("  Loaded shadow matrices for anisotropic sky")

        return cls(
            wall_height=wall_height_arr,
            wall_aspect=wall_aspect_arr,
            svf=svf_arrays,
            shadow_matrices=shadow_arrays,
        )
