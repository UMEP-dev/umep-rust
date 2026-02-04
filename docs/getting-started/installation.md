# Installation

## Requirements

- Python 3.10+
- Rust toolchain (for building from source)

## From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/UMEP-dev/solweig.git
cd solweig

# Install Python dependencies with uv
uv sync

# Build the Rust extension
maturin develop

# Verify installation
python -c "import solweig; print(solweig.__version__)"
```

## Dependencies

SOLWEIG has minimal dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `rasterio` | GeoTIFF I/O (optional, for file-based workflows) |
| `affine` | Geospatial transforms |
| `pyproj` | Coordinate reference systems |

## GPU Acceleration

SOLWEIG automatically uses GPU acceleration for shadow calculations when available.

```python
import solweig

# Check GPU status
print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")
```

To disable GPU acceleration:

```python
solweig.disable_gpu()
```

## Troubleshooting

### "maturin: command not found"

Install maturin first:

```bash
pip install maturin
```

### Build errors on macOS

Ensure you have the Xcode command line tools:

```bash
xcode-select --install
```

### Missing rasterio

If you only use numpy arrays (not GeoTIFFs), rasterio is optional. To install it:

```bash
pip install rasterio
```
