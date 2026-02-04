#!/usr/bin/env python3
"""
Build script for SOLWEIG QGIS plugin.

Builds the Rust extension and bundles it into the plugin directory
for distribution without requiring users to install via pip.

Usage:
    python build_plugin.py              # Build for current platform
    python build_plugin.py --release    # Release build (optimized)
    python build_plugin.py --clean      # Clean build artifacts
    python build_plugin.py --package    # Create distributable ZIP
    python build_plugin.py --target x86_64-apple-darwin  # Cross-compile for Intel Mac
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PLUGIN_DIR = SCRIPT_DIR / "solweig_qgis"  # Plugin directory (renamed to avoid conflict)
BUNDLE_ROOT = PLUGIN_DIR / "_bundled"  # Root of bundled libraries
BUNDLE_DIR = BUNDLE_ROOT / "solweig"  # solweig package inside _bundled
PYSRC_DIR = PROJECT_ROOT / "pysrc" / "solweig"


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        sys.exit(1)
    return result


def get_platform_info(target: str | None = None) -> dict:
    """Get platform information.

    Args:
        target: Optional Rust target triple (e.g., 'x86_64-apple-darwin').
                If None, uses current platform.
    """
    if target:
        # Parse target triple: arch-vendor-os or arch-vendor-os-env
        parts = target.split("-")
        arch = parts[0]
        if "darwin" in target:
            system = "darwin"
        elif "linux" in target:
            system = "linux"
        elif "windows" in target:
            system = "windows"
        else:
            system = parts[2] if len(parts) > 2 else "unknown"

        # Normalize arch names
        if arch == "aarch64":
            arch = "aarch64"
        elif arch in ("x86_64", "amd64"):
            arch = "x86_64"
    else:
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Normalize machine architecture
        if machine in ("x86_64", "amd64"):
            arch = "x86_64"
        elif machine in ("arm64", "aarch64"):
            arch = "aarch64"
        else:
            arch = machine

    # Extension suffix
    ext = ".pyd" if system == "windows" else ".so"

    return {
        "system": system,
        "arch": arch,
        "ext": ext,
        "platform_tag": f"{system}_{arch}",
    }


def clean_bundle_dir():
    """Clean the bundle directory."""
    if BUNDLE_ROOT.exists():
        print(f"  Cleaning {BUNDLE_ROOT}")
        shutil.rmtree(BUNDLE_ROOT)
    # Create _bundled/solweig/ structure
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


def build_rust_extension(release: bool = True, target: str | None = None) -> Path:
    """Build the Rust extension using maturin.

    Builds an abi3 wheel that works across Python 3.9+, ensuring
    compatibility with QGIS (which uses Python 3.9).

    Args:
        release: Build optimized release version.
        target: Optional Rust target triple for cross-compilation
                (e.g., 'x86_64-apple-darwin' for Intel Mac).
    """
    target_msg = f" for {target}" if target else ""
    print(f"\n[1/3] Building Rust extension (abi3 for Python 3.9+){target_msg}...")

    # Use uv run to ensure maturin is available from project environment
    # If uv isn't available, fall back to direct maturin
    build_args = ["maturin", "build"]
    if release:
        build_args.append("--release")
    if target:
        build_args.extend(["--target", target])
    build_args.extend(["--out", "dist"])

    # Try uv run first, fall back to direct maturin
    try:
        run_command(["uv", "run"] + build_args, cwd=PROJECT_ROOT)
    except Exception:
        print("  uv not available, trying maturin directly...")
        run_command(build_args, cwd=PROJECT_ROOT)

    # Find the built wheel
    dist_dir = PROJECT_ROOT / "dist"
    wheels = list(dist_dir.glob("solweig-*.whl"))
    if not wheels:
        print("  ERROR: No wheel found in dist/")
        sys.exit(1)

    # Use the most recent wheel
    wheel_path = max(wheels, key=lambda p: p.stat().st_mtime)
    print(f"  Built: {wheel_path.name}")
    return wheel_path


def extract_solweig_from_wheel(wheel_path: Path) -> None:
    """Extract entire solweig package from wheel to bundle directory.

    This extracts all Python modules, compiled extensions, and data files
    from the wheel in one step, ensuring the bundle matches exactly what
    would be installed via pip.
    """
    print("\n[2/3] Extracting solweig package from wheel...")

    platform_info = get_platform_info()
    has_abi3 = False
    has_extension = False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract wheel
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find the solweig package directory in the extracted wheel
        solweig_pkg = tmpdir / "solweig"
        if not solweig_pkg.exists():
            print("  ERROR: No solweig/ directory found in wheel")
            sys.exit(1)

        # Copy entire solweig package to bundle directory
        # This includes: Python modules, compiled extensions, data files, subdirectories
        for item in solweig_pkg.iterdir():
            dest = BUNDLE_DIR / item.name

            if item.is_dir():
                # Skip __pycache__ directories
                if item.name == "__pycache__":
                    continue
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                print(f"  Copied: {item.name}/")
            else:
                # Skip .pyc files
                if item.suffix == ".pyc":
                    continue
                shutil.copy2(item, dest)

                # Track extension files
                if item.suffix in (".so", ".pyd"):
                    has_extension = True
                    if "abi3" in item.name:
                        has_abi3 = True
                    print(f"  Copied: {item.name} (compiled extension)")
                else:
                    print(f"  Copied: {item.name}")

    # Verify extension was found
    if not has_extension:
        print(f"  WARNING: No compiled extension found ({platform_info['ext']})")
    elif has_abi3:
        print("  ✓ abi3 extension detected (compatible with Python 3.9+)")
    else:
        print("  ⚠ Version-specific extension (may not work in QGIS)")


def create_bundle_init():
    """Create __init__.py for the bundled module if not present in wheel."""
    print("\n[3/3] Verifying bundle __init__.py...")

    init_path = BUNDLE_DIR / "__init__.py"

    # If wheel already included __init__.py, we're done
    if init_path.exists():
        print(f"  ✓ {init_path.name} already exists from wheel")
        return

    # Otherwise create a minimal one
    init_content = '''"""
Bundled SOLWEIG library for QGIS plugin.

This module provides the SOLWEIG library bundled with the plugin,
eliminating the need for separate pip installation.
"""

# Re-export everything from api
from .api import *
'''

    init_path.write_text(init_content)
    print(f"  Created: {init_path.name}")


def create_package_zip(version: str = "0.1.0", target: str | None = None) -> Path:
    """Create distributable ZIP file."""
    print("\nCreating distributable package...")

    platform_info = get_platform_info(target)
    zip_name = f"solweig-qgis-{version}-{platform_info['platform_tag']}.zip"
    zip_path = SCRIPT_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PLUGIN_DIR.rglob("*"):
            if file_path.is_file():
                # Skip __pycache__ and .pyc files
                if "__pycache__" in str(file_path) or file_path.suffix == ".pyc":
                    continue
                arcname = file_path.relative_to(SCRIPT_DIR)
                zf.write(file_path, arcname)

    print(f"  Created: {zip_path.name}")
    print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build SOLWEIG QGIS plugin")
    parser.add_argument("--release", action="store_true", default=True, help="Build release (optimized) version")
    parser.add_argument("--debug", action="store_true", help="Build debug version")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--package", action="store_true", help="Create distributable ZIP")
    parser.add_argument("--version", default="0.1.0", help="Version for package name")
    parser.add_argument(
        "--target",
        help="Rust target triple for cross-compilation (e.g., x86_64-apple-darwin)",
    )
    args = parser.parse_args()

    if args.debug:
        args.release = False

    print("=" * 60)
    print("SOLWEIG QGIS Plugin Builder")
    print("=" * 60)

    platform_info = get_platform_info(args.target)
    if args.target:
        print(f"\nCross-compiling for: {args.target}")
    print(f"Platform: {platform_info['system']} {platform_info['arch']}")
    print(f"Extension type: {platform_info['ext']}")

    if args.clean:
        print("\nCleaning build artifacts...")
        clean_bundle_dir()
        dist_dir = PROJECT_ROOT / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        for zip_file in SCRIPT_DIR.glob("solweig-qgis-*.zip"):
            zip_file.unlink()
        print("Done!")
        return

    # Clean and prepare
    clean_bundle_dir()

    # Build steps
    wheel_path = build_rust_extension(release=args.release, target=args.target)
    extract_solweig_from_wheel(wheel_path)
    create_bundle_init()

    if args.package:
        create_package_zip(args.version, target=args.target)

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nBundled library: {BUNDLE_DIR}")
    print("\nTo test in QGIS:")
    print(f"  1. Symlink or copy {PLUGIN_DIR} to your QGIS plugins folder")
    print("  2. Restart QGIS and enable the plugin")
    print("\nTo distribute:")
    print("  python build_plugin.py --package")


if __name__ == "__main__":
    main()
