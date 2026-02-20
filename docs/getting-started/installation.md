# Installation

## Install from PyPI

```bash
pip install solweig
```

Verify it worked:

```bash
python -c "import solweig; print(solweig.__version__)"
```

## Install from source (for development)

If you want to modify the code or contribute:

- **Python 3.10+**
- **Rust toolchain** — needed to compile the high-performance core ([install Rust](https://rustup.rs/))
- **uv** — fast Python package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig
uv sync              # Install Python dependencies
maturin develop      # Compile Rust extension and link it
```

## Optional dependencies

SOLWEIG works with just numpy arrays, but file-based workflows benefit from these extras:

| Package | What it enables |
| ------- | --------------- |
| `rasterio` | Loading/saving GeoTIFF rasters (installed by default) |
| `geopandas` | Rasterising vector data (e.g. tree polygons to a canopy grid) |
| `affine` | Geospatial coordinate transforms (installed by default) |
| `pyproj` | CRS handling and coordinate conversion (installed by default) |

If you only work with numpy arrays, `rasterio` and `geopandas` are not needed.

## GPU acceleration

SOLWEIG automatically uses GPU acceleration (via wgpu/Metal/Vulkan) when available. No extra setup is needed.

```python
import solweig

print(f"GPU available: {solweig.is_gpu_available()}")
print(f"Backend: {solweig.get_compute_backend()}")  # "gpu" or "cpu"
```

If no GPU is found, it falls back to CPU transparently. To force CPU mode:

```python
solweig.disable_gpu()
```

## Troubleshooting

### `maturin: command not found`

Install it via uv or pip:

```bash
uv tool install maturin
# or
pip install maturin
```

### Build errors on macOS

Ensure Xcode command line tools are installed:

```bash
xcode-select --install
```

### `import solweig` fails after `maturin develop`

Make sure you're using the same Python environment that `uv sync` created. If using uv:

```bash
uv run python -c "import solweig; print('OK')"
```
