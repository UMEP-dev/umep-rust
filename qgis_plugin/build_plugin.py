#!/usr/bin/env python3
"""
Build script for SOLWEIG QGIS plugin.

Packages the plugin into a distributable ZIP for the QGIS Plugin Repository.
The solweig library itself is installed separately via pip (auto-prompted on
first use, or manually with ``pip install solweig``).

Usage:
    python build_plugin.py              # Create distributable ZIP
    python build_plugin.py --version 0.2.0  # With explicit version
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PLUGIN_DIR = SCRIPT_DIR / "solweig_qgis"


def copy_license():
    """Copy LICENSE from project root into the plugin directory (required by QGIS repo)."""
    src = PROJECT_ROOT / "LICENSE"
    dest = PLUGIN_DIR / "LICENSE"
    if src.exists():
        shutil.copy2(src, dest)
        print(f"  Copied LICENSE into {PLUGIN_DIR.name}/")
    else:
        print("  WARNING: No LICENSE file found at project root")


def create_package_zip(version: str = "0.1.0") -> Path:
    """Create distributable ZIP file for QGIS Plugin Repository."""
    zip_name = f"solweig-qgis-{version}.zip"
    zip_path = SCRIPT_DIR / zip_name

    print(f"\nCreating {zip_name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PLUGIN_DIR.rglob("*"):
            if file_path.is_file():
                # Skip __pycache__, .pyc, and macOS metadata files
                if "__pycache__" in str(file_path) or file_path.suffix == ".pyc":
                    continue
                if file_path.name in (".DS_Store", "._DS_Store"):
                    continue
                arcname = file_path.relative_to(SCRIPT_DIR)
                zf.write(file_path, arcname)

    size_kb = zip_path.stat().st_size / 1024
    print(f"  Created: {zip_path.name} ({size_kb:.0f} KB)")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build SOLWEIG QGIS plugin")
    parser.add_argument("--version", default="0.1.0", help="Version for package name")
    parser.add_argument("--clean", action="store_true", help="Clean old ZIP artifacts")
    args = parser.parse_args()

    print("=" * 60)
    print("SOLWEIG QGIS Plugin Builder")
    print("=" * 60)

    if args.clean:
        print("\nCleaning build artifacts...")
        for zip_file in SCRIPT_DIR.glob("solweig-qgis-*.zip"):
            zip_file.unlink()
            print(f"  Removed: {zip_file.name}")
        print("Done!")
        return

    copy_license()
    zip_path = create_package_zip(args.version)

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nPackage: {zip_path}")
    print("\nTo install in QGIS:")
    print("  1. Plugins > Manage and Install Plugins > Install from ZIP")
    print(f"  2. Select {zip_path.name}")
    print("  3. The plugin will prompt to install the solweig library on first use")


if __name__ == "__main__":
    main()
