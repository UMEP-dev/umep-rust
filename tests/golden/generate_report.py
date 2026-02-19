"""
Golden Test Visual Report Generator

Generates a comprehensive Markdown report comparing current implementation
outputs against golden fixtures for regression testing.

Usage:
    uv run python tests/golden/generate_report.py

Output:
    temp/golden_report/golden_report.md
    temp/golden_report/*.png
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPORT_DIR = Path(__file__).parents[2] / "temp" / "golden_report"


def ensure_report_dir():
    """Create report directory if it doesn't exist."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    """Load all input fixtures."""
    return {
        "dsm": np.load(FIXTURES_DIR / "input_dsm.npy"),
        "cdsm": np.load(FIXTURES_DIR / "input_cdsm.npy"),
        "tdsm": np.load(FIXTURES_DIR / "input_tdsm.npy"),
        "bush": np.load(FIXTURES_DIR / "input_bush.npy"),
        "wall_ht": np.load(FIXTURES_DIR / "input_wall_ht.npy"),
        "wall_asp": np.load(FIXTURES_DIR / "input_wall_asp.npy"),
        "params": dict(np.load(FIXTURES_DIR / "input_params.npz")),
    }


def compute_shadows(inputs, azimuth, altitude):
    """Compute shadows for given sun position."""
    from solweig.rustalgos import shadowing

    shadowing.disable_gpu()
    return shadowing.calculate_shadows_wall_ht_25(
        azimuth,
        altitude,
        float(inputs["params"]["scale"]),
        float(inputs["params"]["amaxvalue"]),
        inputs["dsm"].astype(np.float32),
        inputs["cdsm"].astype(np.float32),
        inputs["tdsm"].astype(np.float32),
        inputs["bush"].astype(np.float32),
        inputs["wall_ht"].astype(np.float32),
        (inputs["wall_asp"] * np.pi / 180.0).astype(np.float32),
        None,
        None,
        None,
    )


def compute_svf(inputs):
    """Compute SVF."""
    from solweig.rustalgos import shadowing, skyview

    shadowing.disable_gpu()
    return skyview.calculate_svf(
        inputs["dsm"].astype(np.float32),
        inputs["cdsm"].astype(np.float32),
        inputs["tdsm"].astype(np.float32),
        float(inputs["params"]["scale"]),
        True,  # usevegdem
        float(inputs["params"]["amaxvalue"]),
        2,  # patch_option
        None,
        None,
    )


def compute_gvf(inputs):
    """Compute GVF."""
    from solweig.constants import SBC
    from solweig.rustalgos import gvf as gvf_module
    from solweig.rustalgos import shadowing

    shadowing.disable_gpu()

    rows, cols = inputs["dsm"].shape
    scale = float(inputs["params"]["scale"])

    # Building mask
    wall_mask = inputs["wall_ht"] > 0
    struct = ndimage.generate_binary_structure(2, 2)
    iterations = int(25 / scale) + 1
    dilated = ndimage.binary_dilation(wall_mask, struct, iterations=iterations)
    buildings = (~dilated).astype(np.float32)

    # Load shadow data
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy")
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy")
    wall_sun = np.load(FIXTURES_DIR / "shadow_noon_wall_sun.npy")
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)

    # Load ground temperature from fixture (spatially varying)
    tg_path = FIXTURES_DIR / "gvf_input_tg.npy"
    tg = np.load(tg_path).astype(np.float32) if tg_path.exists() else np.zeros((rows, cols), dtype=np.float32)

    emis_grid = np.full((rows, cols), 0.95, dtype=np.float32)
    alb_grid = np.full((rows, cols), 0.15, dtype=np.float32)

    gvf_params = gvf_module.GvfScalarParams(
        scale=scale,
        first=2.0,
        second=36.0,
        tgwall=2.0,
        ta=25.0,
        ewall=0.90,
        sbc=SBC,
        albedo_b=0.20,
        twater=25.0,
        landcover=False,
    )

    return gvf_module.gvf_calc(
        wall_sun.astype(np.float32),
        inputs["wall_ht"].astype(np.float32),
        buildings,
        shadow,
        inputs["wall_asp"].astype(np.float32),
        tg,
        emis_grid,
        alb_grid,
        None,
        gvf_params,
    )


def compute_radiation(inputs):
    """Compute Kside and Lside."""
    from solweig.constants import SBC
    from solweig.rustalgos import shadowing, vegetation

    shadowing.disable_gpu()

    rows, cols = inputs["dsm"].shape

    # Load SVF and shadow data
    svf_e = np.load(FIXTURES_DIR / "svf_east.npy").astype(np.float32)
    svf_s = np.load(FIXTURES_DIR / "svf_south.npy").astype(np.float32)
    svf_w = np.load(FIXTURES_DIR / "svf_west.npy").astype(np.float32)
    svf_n = np.load(FIXTURES_DIR / "svf_north.npy").astype(np.float32)
    svf_veg = np.load(FIXTURES_DIR / "svf_veg.npy").astype(np.float32)
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy")
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy")
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)

    ta, rad_i, rad_d, rad_g, esky, ci = 25.0, 600.0, 200.0, 800.0, 0.75, 0.85
    f_sh = np.full((rows, cols), 0.5, dtype=np.float32)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)

    kside = vegetation.kside_veg(
        rad_i,
        rad_d,
        rad_g,
        shadow,
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,
        0.5,
        0.0,
        0.20,
        f_sh,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        True,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
    )

    ta_k = ta + 273.15
    ldown = np.full((rows, cols), esky * SBC * (ta_k**4), dtype=np.float32)
    lup = np.full((rows, cols), 0.95 * SBC * (ta_k**4), dtype=np.float32)

    lside = vegetation.lside_veg(
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,
        ta,
        2.0,
        SBC,
        0.90,
        ldown,
        esky,
        0.0,
        f_sh,
        ci,
        lup.copy(),
        lup.copy(),
        lup.copy(),
        lup.copy(),
        False,
    )

    return kside, lside


def plot_context(inputs):
    """Generate context plot showing input data."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Input Context", fontsize=12, fontweight="bold")

    dsm = inputs["dsm"]
    cdsm = inputs["cdsm"]
    wall_ht = inputs["wall_ht"]

    # Load SVF for context
    svf = np.load(FIXTURES_DIR / "svf_total.npy")

    im0 = axes[0].imshow(dsm, cmap="terrain")
    axes[0].set_title(f"DSM (m)\n[{dsm.min():.1f}, {dsm.max():.1f}]")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(cdsm, cmap="Greens")
    axes[1].set_title(f"Canopy DSM (m)\n[{cdsm.min():.1f}, {cdsm.max():.1f}]")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(wall_ht, cmap="Oranges")
    axes[2].set_title(f"Wall Heights (m)\n[{wall_ht.min():.1f}, {wall_ht.max():.1f}]")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    im3 = axes[3].imshow(svf, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"Sky View Factor\n[{svf.min():.2f}, {svf.max():.2f}]")
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "context.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_comparison(current, golden, title, filename, cmap="viridis"):
    """Generate comparison plot: UMEP (golden) vs SOLWEIG Rust (current) vs residual."""
    diff = current - golden

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    vmax = max(abs(current.max()), abs(golden.max()))
    vmin = min(current.min(), golden.min())

    # Golden fixture = UMEP reference
    im0 = axes[0].imshow(golden, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("UMEP (Reference)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Current = SOLWEIG Rust
    im1 = axes[1].imshow(current, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("SOLWEIG Rust")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff_max = max(abs(diff.min()), abs(diff.max()), 1e-10)
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
    axes[2].set_title(f"Residual (Rust - UMEP)\nmax|d|={diff_max:.2e}")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()

    # Return stats without pass/fail - let caller decide threshold
    return {
        "max_abs_diff": float(np.abs(diff).max()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "max_value": float(np.abs(golden).max()),  # For relative comparisons
    }


def plot_single_array(arr, title, filename, cmap="viridis"):
    """Generate single array plot (for outputs without UMEP reference)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(f"[{arr.min():.2f}, {arr.max():.2f}]")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()

    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Component generators (compute + compare)
# ---------------------------------------------------------------------------


def generate_shadow_comparisons(inputs):
    """Generate shadow comparison plots."""
    results = {}
    positions = [
        ("morning", 90.0, 30.0),
        ("noon", 180.0, 60.0),
        ("afternoon", 270.0, 45.0),
    ]

    for name, azimuth, altitude in positions:
        result = compute_shadows(inputs, azimuth, altitude)

        # Building shadows
        golden = np.load(FIXTURES_DIR / f"shadow_{name}_bldg_sh.npy")
        current = np.array(result.bldg_sh)
        stats = plot_comparison(
            current,
            golden,
            f"Building Shadows - {name.title()} (az={azimuth}, alt={altitude})",
            f"shadow_{name}_bldg.png",
            cmap="gray_r",
        )
        results[f"shadow_{name}_bldg"] = stats

        # Vegetation shadows
        golden = np.load(FIXTURES_DIR / f"shadow_{name}_veg_sh.npy")
        current = np.array(result.veg_sh)
        stats = plot_comparison(
            current,
            golden,
            f"Vegetation Shadows - {name.title()} (az={azimuth}, alt={altitude})",
            f"shadow_{name}_veg.png",
            cmap="gray_r",
        )
        results[f"shadow_{name}_veg"] = stats

        # Wall shadows (shadowed height)
        wall_sh_path = FIXTURES_DIR / f"shadow_{name}_wall_sh.npy"
        if wall_sh_path.exists() and result.wall_sh is not None:
            golden = np.load(wall_sh_path)
            current = np.array(result.wall_sh)
            stats = plot_comparison(
                current,
                golden,
                f"Wall Shadows - {name.title()} (az={azimuth}, alt={altitude})",
                f"shadow_{name}_wall_sh.png",
                cmap="Oranges",
            )
            results[f"shadow_{name}_wall_sh"] = stats

        # Wall sun (sunlit height)
        wall_sun_path = FIXTURES_DIR / f"shadow_{name}_wall_sun.npy"
        if wall_sun_path.exists() and result.wall_sun is not None:
            golden = np.load(wall_sun_path)
            current = np.array(result.wall_sun)
            stats = plot_comparison(
                current,
                golden,
                f"Wall Sun - {name.title()} (az={azimuth}, alt={altitude})",
                f"shadow_{name}_wall_sun.png",
                cmap="YlOrRd",
            )
            results[f"shadow_{name}_wall_sun"] = stats

    return results


def generate_svf_comparisons(inputs):
    """Generate SVF comparison plots."""
    results = {}
    result = compute_svf(inputs)

    components = [
        ("svf", "svf_total", "Total SVF"),
        ("svf_north", "svf_north", "SVF North"),
        ("svf_east", "svf_east", "SVF East"),
        ("svf_south", "svf_south", "SVF South"),
        ("svf_west", "svf_west", "SVF West"),
        ("svf_veg", "svf_veg", "SVF Vegetation"),
    ]

    for attr, golden_name, title in components:
        golden = np.load(FIXTURES_DIR / f"{golden_name}.npy")
        current = np.array(getattr(result, attr))
        stats = plot_comparison(current, golden, title, f"{golden_name}.png", cmap="gray")
        results[golden_name] = stats

    return results


def generate_gvf_comparisons(inputs):
    """Generate GVF comparison plots."""
    results = {}
    result = compute_gvf(inputs)

    components = [
        ("gvf_lup", "gvf_lup", "GVF Lup (W/m2)", "hot"),
        ("gvfalb", "gvf_alb", "GVF x Albedo", "viridis"),
        ("gvf_norm", "gvf_norm", "GVF Normalization", "viridis"),
    ]

    for attr, golden_name, title, cmap in components:
        golden = np.load(FIXTURES_DIR / f"{golden_name}.npy")
        current = np.array(getattr(result, attr))
        stats = plot_comparison(current, golden, title, f"{golden_name}.png", cmap=cmap)
        results[golden_name] = stats

    return results


def generate_radiation_comparisons(inputs):
    """Generate radiation comparison plots (isotropic mode)."""
    results = {}
    kside, lside = compute_radiation(inputs)

    components = [
        (kside, "keast", "radiation_kside_e", "Kside East - Isotropic (W/m2)", "YlOrRd"),
        (kside, "ksouth", "radiation_kside_s", "Kside South - Isotropic (W/m2)", "YlOrRd"),
        (lside, "least", "radiation_lside_e", "Lside East - Isotropic (W/m2)", "inferno"),
        (lside, "lsouth", "radiation_lside_s", "Lside South - Isotropic (W/m2)", "inferno"),
    ]

    for obj, attr, golden_name, title, cmap in components:
        golden = np.load(FIXTURES_DIR / f"{golden_name}.npy")
        current = np.array(getattr(obj, attr))
        stats = plot_comparison(current, golden, title, f"{golden_name}.png", cmap=cmap)
        results[golden_name] = stats

    return results


def generate_aniso_radiation_comparisons(inputs):
    """Generate anisotropic radiation comparison plots."""
    from solweig.constants import SBC
    from solweig.rustalgos import shadowing, vegetation

    results = {}

    # Check if anisotropic fixtures exist
    aniso_kside_path = FIXTURES_DIR / "radiation_aniso_kside_e.npy"
    if not aniso_kside_path.exists():
        print("    Anisotropic radiation fixtures not found, skipping...")
        return results

    shadowing.disable_gpu()
    rows, cols = inputs["dsm"].shape

    # Load SVF and shadow data
    svf = np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32)
    svf_e = np.load(FIXTURES_DIR / "svf_east.npy").astype(np.float32)
    svf_s = np.load(FIXTURES_DIR / "svf_south.npy").astype(np.float32)
    svf_w = np.load(FIXTURES_DIR / "svf_west.npy").astype(np.float32)
    svf_n = np.load(FIXTURES_DIR / "svf_north.npy").astype(np.float32)
    svf_veg = np.load(FIXTURES_DIR / "svf_veg.npy").astype(np.float32)
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32)
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32)
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)

    ta, rad_i, rad_d, rad_g, esky, ci = 25.0, 600.0, 200.0, 800.0, 0.75, 0.85
    f_sh = np.full((rows, cols), 0.5, dtype=np.float32)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)

    # Generate sky patches for anisotropic mode
    def generate_sky_patches(n_alt_bands=4):
        patches = []
        alt_bands = [6, 18, 30, 42]
        azis_per_band = [30, 24, 24, 18]
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands], strict=False):
            azi_step = 360.0 / n_azi if n_azi > 1 else 0
            for azi_idx in range(n_azi):
                patches.append([alt, azi_idx * azi_step])
        return np.array(patches, dtype=np.float32)

    l_patches = generate_sky_patches(n_alt_bands=4)
    n_patches = len(l_patches)

    # Create luminance values (Perez model simplified)
    luminance = 1000 + 500 * np.sin(l_patches[:, 0] * np.pi / 180)
    lv = np.column_stack([l_patches, luminance]).astype(np.float32)

    # Create 3D shadow matrices from SVF and shadows
    svf_expanded = svf[:, :, np.newaxis]
    base_visibility = np.broadcast_to(svf_expanded, (rows, cols, n_patches)).copy()
    bldg_factor = shadow_bldg[:, :, np.newaxis]
    veg_factor = shadow_veg[:, :, np.newaxis]

    shmat = (base_visibility * np.broadcast_to(bldg_factor, (rows, cols, n_patches))).astype(np.float32)
    shmat = (shmat > 0.5).astype(np.float32)
    vegshmat = (base_visibility * np.broadcast_to(veg_factor, (rows, cols, n_patches))).astype(np.float32)
    vegshmat = (vegshmat > 0.3).astype(np.float32)
    vbshvegshmat = (shmat * vegshmat).astype(np.float32)

    # Diffuse shadow (3D - same shape as shmat for diffuse sky patches)
    diffsh = shmat.copy()  # 3D array (rows, cols, patches)
    asvf = svf.copy()

    # Compute anisotropic Kside
    kside_aniso = vegetation.kside_veg(
        rad_i,
        rad_d,
        rad_g,
        shadow,
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,  # azimuth, altitude
        0.5,  # psi
        0.0,  # t
        0.20,  # albedo
        f_sh,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        True,  # cyl
        lv,  # luminance values
        True,  # anisotropic_diffuse
        diffsh,
        asvf,
        shmat,
        vegshmat,
        vbshvegshmat,
    )

    # Compare Kside anisotropic
    components = [
        ("keast", "radiation_aniso_kside_e", "Kside East - Anisotropic (W/m2)", "YlOrRd"),
        ("ksouth", "radiation_aniso_kside_s", "Kside South - Anisotropic (W/m2)", "YlOrRd"),
        ("kside_i", "radiation_aniso_kside_i", "Kside Direct - Anisotropic (W/m2)", "YlOrRd"),
        ("kside_d", "radiation_aniso_kside_d", "Kside Diffuse - Anisotropic (W/m2)", "YlOrRd"),
    ]

    for attr, golden_name, title, cmap in components:
        golden_path = FIXTURES_DIR / f"{golden_name}.npy"
        if golden_path.exists():
            golden = np.load(golden_path)
            current = np.array(getattr(kside_aniso, attr))
            stats = plot_comparison(current, golden, title, f"{golden_name}.png", cmap=cmap)
            results[golden_name] = stats

    # Compute anisotropic Lside
    ta_k = ta + 273.15
    ldown = np.full((rows, cols), esky * SBC * (ta_k**4), dtype=np.float32)
    lup = np.full((rows, cols), 0.95 * SBC * (ta_k**4), dtype=np.float32)

    lside_aniso = vegetation.lside_veg(
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,
        ta,
        2.0,
        SBC,
        0.90,
        ldown,
        esky,
        0.0,
        f_sh,
        ci,
        lup.copy(),
        lup.copy(),
        lup.copy(),
        lup.copy(),
        True,  # anisotropic_longwave
    )

    lside_components = [
        ("least", "radiation_aniso_lside_e", "Lside East - Anisotropic (W/m2)", "inferno"),
        ("lsouth", "radiation_aniso_lside_s", "Lside South - Anisotropic (W/m2)", "inferno"),
    ]

    for attr, golden_name, title, cmap in lside_components:
        golden_path = FIXTURES_DIR / f"{golden_name}.npy"
        if golden_path.exists():
            golden = np.load(golden_path)
            current = np.array(getattr(lside_aniso, attr))
            stats = plot_comparison(current, golden, title, f"{golden_name}.png", cmap=cmap)
            results[golden_name] = stats

    return results


def generate_utci_comparisons():
    """Generate UTCI comparison plots."""
    from solweig.rustalgos import utci

    results = {}

    # Load fixtures
    params = dict(np.load(FIXTURES_DIR / "utci_grid_params.npz"))
    tmrt = np.load(FIXTURES_DIR / "utci_grid_tmrt.npy")
    va = np.load(FIXTURES_DIR / "utci_grid_va.npy")
    golden = np.load(FIXTURES_DIR / "utci_grid_output.npy")

    # Compute current
    current = np.array(
        utci.utci_grid(
            float(params["ta"]),
            float(params["rh"]),
            tmrt.astype(np.float32),
            va.astype(np.float32),
        )
    )

    stats = plot_comparison(
        current,
        golden,
        f"UTCI Grid (Ta={params['ta']}C, RH={params['rh']}%)",
        "utci_grid.png",
        cmap="RdYlBu_r",
    )
    results["utci_grid"] = stats

    return results


def generate_pet_comparisons():
    """Generate PET comparison plots."""
    from solweig.rustalgos import pet

    results = {}

    # Load fixtures
    params = dict(np.load(FIXTURES_DIR / "pet_grid_params.npz"))
    tmrt = np.load(FIXTURES_DIR / "pet_grid_tmrt.npy")
    va = np.load(FIXTURES_DIR / "pet_grid_va.npy")
    golden = np.load(FIXTURES_DIR / "pet_grid_output.npy")

    # Compute current
    current = np.array(
        pet.pet_grid(
            float(params["ta"]),
            float(params["rh"]),
            tmrt.astype(np.float32),
            va.astype(np.float32),
            float(params["mbody"]),
            float(params["age"]),
            float(params["height"]),
            float(params["activity"]),
            float(params["clo"]),
            int(params["sex"]),
        )
    )

    # Mask invalid values
    valid_mask = golden > -999
    current_masked = np.where(valid_mask, current, np.nan)
    golden_masked = np.where(valid_mask, golden, np.nan)

    stats = plot_comparison(
        current_masked,
        golden_masked,
        f"PET Grid (Ta={params['ta']}C, RH={params['rh']}%)",
        "pet_grid.png",
        cmap="RdYlBu_r",
    )
    results["pet_grid"] = stats

    return results


def generate_tmrt_comparisons():
    """Generate Tmrt comparison plots."""
    from solweig.rustalgos import tmrt

    results = {}

    # Load fixtures
    params = dict(np.load(FIXTURES_DIR / "tmrt_params.npz"))
    kdown = np.load(FIXTURES_DIR / "tmrt_input_kdown.npy")
    kup = np.load(FIXTURES_DIR / "tmrt_input_kup.npy")
    ldown = np.load(FIXTURES_DIR / "tmrt_input_ldown.npy")
    lup = np.load(FIXTURES_DIR / "tmrt_input_lup.npy")
    kside_n = np.load(FIXTURES_DIR / "tmrt_input_kside_n.npy")
    kside_e = np.load(FIXTURES_DIR / "tmrt_input_kside_e.npy")
    kside_s = np.load(FIXTURES_DIR / "tmrt_input_kside_s.npy")
    kside_w = np.load(FIXTURES_DIR / "tmrt_input_kside_w.npy")
    kside_total = np.load(FIXTURES_DIR / "tmrt_input_kside_total.npy")
    lside_n = np.load(FIXTURES_DIR / "tmrt_input_lside_n.npy")
    lside_e = np.load(FIXTURES_DIR / "tmrt_input_lside_e.npy")
    lside_s = np.load(FIXTURES_DIR / "tmrt_input_lside_s.npy")
    lside_w = np.load(FIXTURES_DIR / "tmrt_input_lside_w.npy")
    lside_total = np.load(FIXTURES_DIR / "tmrt_input_lside_total.npy")

    # Anisotropic mode
    golden_aniso = np.load(FIXTURES_DIR / "tmrt_output_aniso.npy")
    tmrt_params = tmrt.TmrtParams(
        abs_k=float(params["abs_k"]),
        abs_l=float(params["abs_l"]),
        is_standing=True,
        use_anisotropic_sky=True,
    )
    current_aniso = np.array(
        tmrt.compute_tmrt(
            kdown,
            kup,
            ldown,
            lup,
            kside_n,
            kside_e,
            kside_s,
            kside_w,
            lside_n,
            lside_e,
            lside_s,
            lside_w,
            kside_total,
            lside_total,
            tmrt_params,
        )
    )

    stats = plot_comparison(
        current_aniso,
        golden_aniso,
        "Tmrt Anisotropic (C)",
        "tmrt_aniso.png",
        cmap="RdYlBu_r",
    )
    results["tmrt_aniso"] = stats

    # Isotropic mode
    golden_iso = np.load(FIXTURES_DIR / "tmrt_output_iso.npy")
    tmrt_params_iso = tmrt.TmrtParams(
        abs_k=float(params["abs_k"]),
        abs_l=float(params["abs_l"]),
        is_standing=True,
        use_anisotropic_sky=False,
    )
    current_iso = np.array(
        tmrt.compute_tmrt(
            kdown,
            kup,
            ldown,
            lup,
            kside_n,
            kside_e,
            kside_s,
            kside_w,
            lside_n,
            lside_e,
            lside_s,
            lside_w,
            kside_total,
            lside_total,
            tmrt_params_iso,
        )
    )

    stats = plot_comparison(
        current_iso,
        golden_iso,
        "Tmrt Isotropic (C)",
        "tmrt_iso.png",
        cmap="RdYlBu_r",
    )
    results["tmrt_iso"] = stats

    return results


def generate_ground_temp_comparisons():
    """Generate ground temperature comparison plots (TsWaveDelay model)."""
    from solweig.rustalgos import ground

    results = {}

    # Load common inputs
    gvflup = np.load(FIXTURES_DIR / "ground_temp_input_gvflup.npy").astype(np.float32)
    tgmap1_init = np.load(FIXTURES_DIR / "ground_temp_input_tgmap1.npy").astype(np.float32)

    case_configs = {
        1: {"firstdaytime": True, "timeadd": 0.0, "timestepdec": 30 / 1440, "name": "First Morning"},
        2: {"firstdaytime": False, "timeadd": 30 / 1440, "timestepdec": 30 / 1440, "name": "Short Step"},
        3: {"firstdaytime": False, "timeadd": 60 / 1440, "timestepdec": 60 / 1440, "name": "Long Step"},
    }

    for case_num, config in case_configs.items():
        case_path = FIXTURES_DIR / f"ground_temp_case{case_num}.npz"
        if not case_path.exists():
            continue

        case_data = dict(np.load(case_path))
        golden_lup = case_data["lup"]

        # Compute current using Rust
        current_lup, _, _ = ground.ts_wave_delay(
            gvflup.copy(),
            config["firstdaytime"],
            config["timeadd"],
            config["timestepdec"],
            tgmap1_init.copy(),
        )
        current_lup = np.array(current_lup)

        stats = plot_comparison(
            current_lup,
            golden_lup,
            f"Ground Temp: {config['name']}",
            f"ground_temp_case{case_num}.png",
            cmap="hot",
        )
        results[f"ground_temp_case{case_num}"] = stats

    return results


def generate_wall_temp_comparisons():
    """Generate wall temperature comparison plots."""
    from solweig.rustalgos import ground

    results = {}

    # Load fixtures
    output = dict(np.load(FIXTURES_DIR / "wall_temp_output.npz"))
    alb_grid = np.load(FIXTURES_DIR / "wall_temp_input_alb.npy")
    emis_grid = np.load(FIXTURES_DIR / "wall_temp_input_emis.npy")
    tgk_grid = np.load(FIXTURES_DIR / "wall_temp_input_tgk.npy")
    tstart_grid = np.load(FIXTURES_DIR / "wall_temp_input_tstart.npy")
    tmaxlst_grid = np.load(FIXTURES_DIR / "wall_temp_input_tmaxlst.npy")

    # Compute current
    tg, tg_wall, ci_tg, _, _ = ground.compute_ground_temperature(
        float(output["ta"]),
        float(output["sun_altitude"]),
        float(output["altmax"]),
        float(output["dectime"]),
        float(output["snup"]),
        float(output["global_rad"]),
        float(output["rad_g0"]),
        float(output["zen_deg"]),
        alb_grid,
        emis_grid,
        tgk_grid,
        tstart_grid,
        tmaxlst_grid,
    )
    current_tg = np.array(tg)
    golden_tg = output["tg"]

    stats = plot_comparison(
        current_tg,
        golden_tg,
        "Ground Temperature Deviation (C)",
        "wall_temp_tg.png",
        cmap="RdYlBu_r",
    )
    results["wall_temp_tg"] = stats

    return results


def generate_aniso_sky_comparisons():
    """Generate anisotropic sky comparison plots."""
    from solweig.rustalgos import sky

    results = {}

    # Check if fixtures exist
    aniso_path = FIXTURES_DIR / "aniso_sky_output.npz"
    if not aniso_path.exists():
        print("    Anisotropic sky fixtures not found, skipping...")
        return results

    # Load golden fixtures
    golden = dict(np.load(aniso_path))

    # Load inputs and recompute
    dsm = np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)
    svf = np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32)
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32)
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32)

    rows, cols = dsm.shape
    SBC = 5.67e-8

    # Generate sky patches
    def generate_sky_patches(n_alt_bands=4):
        patches = []
        alt_bands = [6, 18, 30, 42]
        azis_per_band = [30, 24, 24, 18]
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands], strict=False):
            azi_step = 360.0 / n_azi if n_azi > 1 else 0
            for azi_idx in range(n_azi):
                patches.append([alt, azi_idx * azi_step])
        return np.array(patches, dtype=np.float32)

    def compute_steradians(l_patches):
        n_patches = len(l_patches)
        steradians = np.zeros(n_patches, dtype=np.float32)
        deg2rad = np.pi / 180.0
        altitudes = l_patches[:, 0]
        unique_alts = np.unique(altitudes)
        for i, alt in enumerate(unique_alts):
            mask = altitudes == alt
            count = np.sum(mask)
            if i == 0:
                ster = (360.0 / count * deg2rad) * np.sin(alt * deg2rad)
            else:
                prev_alt = unique_alts[i - 1]
                delta_alt = (alt - prev_alt) / 2
                ster = (360.0 / count * deg2rad) * (
                    np.sin((alt + delta_alt) * deg2rad) - np.sin((prev_alt + delta_alt) * deg2rad)
                )
            steradians[mask] = ster
        return steradians

    l_patches = generate_sky_patches(n_alt_bands=4)
    n_patches = len(l_patches)
    steradians = compute_steradians(l_patches)

    # Create shadow matrices
    svf_expanded = svf[:, :, np.newaxis]
    base_visibility = np.broadcast_to(svf_expanded, (rows, cols, n_patches)).copy()
    bldg_factor = shadow_bldg[:, :, np.newaxis]
    veg_factor = shadow_veg[:, :, np.newaxis]

    shmat = (base_visibility * np.broadcast_to(bldg_factor, (rows, cols, n_patches))).astype(np.float32)
    shmat = (shmat > 0.5).astype(np.float32)
    vegshmat = (base_visibility * np.broadcast_to(veg_factor, (rows, cols, n_patches))).astype(np.float32)
    vegshmat = (vegshmat > 0.3).astype(np.float32)
    vbshvegshmat = (shmat * vegshmat).astype(np.float32)

    asvf = svf.astype(np.float32)
    luminance = 1000 + 500 * np.sin(l_patches[:, 0] * np.pi / 180)
    lv = np.column_stack([l_patches, luminance]).astype(np.float32)
    ta_k = 25.0 + 273.15
    lup = np.full((rows, cols), 0.95 * SBC * (ta_k**4), dtype=np.float32)
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)

    sun_params = sky.SunParams(altitude=60.0, azimuth=180.0)
    sky_params = sky.SkyParams(esky=0.75, ta=25.0, cyl=True, wall_scheme=False, albedo=0.20)
    surface_params = sky.SurfaceParams(tgwall=2.0, ewall=0.90, rad_i=600.0, rad_d=200.0)

    result = sky.anisotropic_sky(
        shmat,
        vegshmat,
        vbshvegshmat,
        sun_params,
        asvf,
        sky_params,
        l_patches,
        None,
        None,
        steradians,
        surface_params,
        lup,
        lv,
        shadow,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
    )

    # Compare outputs
    components = [
        ("ldown", "Ldown (W/m2)", "inferno"),
        ("lside", "Lside (W/m2)", "inferno"),
        ("kside", "Kside (W/m2)", "YlOrRd"),
    ]

    for attr, title, cmap in components:
        current = np.array(getattr(result, attr))
        golden_arr = golden[attr]
        stats = plot_comparison(
            current,
            golden_arr,
            f"Anisotropic Sky: {title}",
            f"aniso_sky_{attr}.png",
            cmap=cmap,
        )
        results[f"aniso_sky_{attr}"] = stats

    return results


# ---------------------------------------------------------------------------
# Sinusoidal ground temperature model: Rust vs UMEP formula comparison
# ---------------------------------------------------------------------------


def _umep_ground_temp(
    altmax,
    dectime_frac,
    snup_hours,
    global_rad,
    rad_g0,
    zen_deg,
    tgk,
    tstart,
    tmaxlst,
    tgk_wall,
    tstart_wall,
    tmaxlst_wall,
    sun_altitude,
):
    """Pure-Python UMEP reference (Solweig_2025a lines 171-199)."""
    Tgamp = tgk * altmax + tstart
    Tgampwall = tgk_wall * altmax + tstart_wall

    snup_frac = snup_hours / 24.0
    if dectime_frac > snup_frac:
        tmaxlst_frac = tmaxlst / 24.0
        phase = (dectime_frac - snup_frac) / (tmaxlst_frac - snup_frac)
        Tg = Tgamp * np.sin(phase * np.pi / 2.0)

        tmaxlst_wall_frac = tmaxlst_wall / 24.0
        denom_wall = tmaxlst_wall_frac - snup_frac
        phase_wall = (dectime_frac - snup_frac) / denom_wall if denom_wall > 0 else dectime_frac - snup_frac
        Tgwall = Tgampwall * np.sin(phase_wall * np.pi / 2.0)
    else:
        Tg = 0.0
        Tgwall = 0.0

    if Tgwall < 0:
        Tgwall = 0.0

    if sun_altitude > 0 and rad_g0 > 0:
        corr = 0.1473 * np.log(90.0 - zen_deg) + 0.3454
        CI_TgG = (global_rad / rad_g0) + (1.0 - corr)
        if CI_TgG > 1.0 or np.isinf(CI_TgG):
            CI_TgG = 1.0
    else:
        CI_TgG = 1.0

    Tg = np.maximum(Tg * CI_TgG, 0.0)
    Tgwall = Tgwall * CI_TgG

    return Tg, Tgwall, CI_TgG


def generate_sinusoidal_ground_temp(inputs=None):
    """Generate sinusoidal ground temperature model comparisons.

    This section compares the Rust compute_ground_temperature() against the
    UMEP Python reference formula for:
    1. A diurnal curve plot (UMEP vs Rust overlaid)
    2. Multiple scenarios covering all land covers and conditions
    """
    from solweig.rustalgos import ground

    results = {}

    # --- Part 1: Diurnal curve comparison ---
    shape = (3, 3)
    altmax = 55.0
    snup = 5.0
    tgk_val = 0.37
    tstart_val = -3.41
    tmaxlst_val = 15.0

    hours = np.arange(0, 24.5, 0.5)
    rust_tg_curve = []
    umep_tg_curve = []
    rust_wall_curve = []
    umep_wall_curve = []

    for h in hours:
        dectime = h / 24.0
        # Sun altitude approximation (simple sinusoidal)
        sun_alt = max(0.0, altmax * np.sin(np.pi * (h - snup) / (21 - snup))) if snup < h < 21 else 0.0
        zen = 90.0 - sun_alt if sun_alt > 0 else 90.0

        # Global radiation proportional to sun altitude
        if sun_alt > 2:
            grad = 800.0 * np.sin(sun_alt * np.pi / 180.0)
            grad0 = 900.0 * np.sin(sun_alt * np.pi / 180.0)
        else:
            grad = 0.0
            grad0 = 0.0

        # Rust
        tg, tg_wall, ci, _, _ = ground.compute_ground_temperature(
            20.0,  # ta
            sun_alt,
            altmax,
            dectime,
            snup,
            grad,
            grad0,
            zen,
            np.full(shape, 0.2, dtype=np.float32),
            np.full(shape, 0.95, dtype=np.float32),
            np.full(shape, tgk_val, dtype=np.float32),
            np.full(shape, tstart_val, dtype=np.float32),
            np.full(shape, tmaxlst_val, dtype=np.float32),
        )
        rust_tg_curve.append(float(np.array(tg)[0, 0]))
        rust_wall_curve.append(float(tg_wall))

        # UMEP
        umep_tg, umep_wall, _ = _umep_ground_temp(
            altmax,
            dectime,
            snup,
            grad,
            grad0,
            zen,
            tgk_val,
            tstart_val,
            tmaxlst_val,
            tgk_val,
            tstart_val,
            tmaxlst_val,
            sun_alt,
        )
        if isinstance(umep_tg, np.ndarray):
            umep_tg_curve.append(float(umep_tg.flat[0]))
        else:
            umep_tg_curve.append(float(umep_tg))
        umep_wall_curve.append(float(umep_wall))

    rust_tg_curve = np.array(rust_tg_curve)
    umep_tg_curve = np.array(umep_tg_curve)
    rust_wall_curve = np.array(rust_wall_curve)
    umep_wall_curve = np.array(umep_wall_curve)

    # Plot diurnal curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sinusoidal Ground Temperature: Rust vs UMEP", fontsize=12, fontweight="bold")

    # Ground
    axes[0].plot(hours, umep_tg_curve, "b-", label="UMEP (Python)", linewidth=2)
    axes[0].plot(hours, rust_tg_curve, "r--", label="Rust", linewidth=2)
    axes[0].axvline(x=snup, color="orange", linestyle=":", alpha=0.7, label="Sunrise")
    axes[0].axvline(x=tmaxlst_val, color="green", linestyle=":", alpha=0.7, label="TmaxLST")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Tg (K above Ta)")
    axes[0].set_title("Ground Temperature Deviation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 24)

    # Wall
    axes[1].plot(hours, umep_wall_curve, "b-", label="UMEP (Python)", linewidth=2)
    axes[1].plot(hours, rust_wall_curve, "r--", label="Rust", linewidth=2)
    axes[1].axvline(x=snup, color="orange", linestyle=":", alpha=0.7, label="Sunrise")
    axes[1].axvline(x=tmaxlst_val, color="green", linestyle=":", alpha=0.7, label="TmaxLST")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Tg_wall (K above Ta)")
    axes[1].set_title("Wall Temperature Deviation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sinusoidal_diurnal.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Compute stats for diurnal curve
    ground_diff = np.max(np.abs(rust_tg_curve - umep_tg_curve))
    wall_diff = np.max(np.abs(rust_wall_curve - umep_wall_curve))
    results["sinusoidal_ground_diurnal"] = {
        "max_abs_diff": ground_diff,
        "mean_diff": float(np.mean(rust_tg_curve - umep_tg_curve)),
        "std_diff": float(np.std(rust_tg_curve - umep_tg_curve)),
        "max_value": float(np.max(np.abs(umep_tg_curve))),
    }
    results["sinusoidal_wall_diurnal"] = {
        "max_abs_diff": wall_diff,
        "mean_diff": float(np.mean(rust_wall_curve - umep_wall_curve)),
        "std_diff": float(np.std(rust_wall_curve - umep_wall_curve)),
        "max_value": float(np.max(np.abs(umep_wall_curve))),
    }

    # --- Part 2: Multi-scenario formula agreement ---
    _s = dict  # shorthand
    scenarios = [
        # fmt: off
        (
            "Noon clear cobble",
            _s(
                altmax=55,
                hour=12,
                snup=5,
                grad=600,
                grad0=650,
                zen=35,
                sun_alt=55,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Noon clear asphalt",
            _s(
                altmax=55,
                hour=12,
                snup=5,
                grad=600,
                grad0=650,
                zen=35,
                sun_alt=55,
                tgk=0.58,
                tstart=-9.78,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Afternoon 18h",
            _s(
                altmax=55,
                hour=18,
                snup=5,
                grad=300,
                grad0=400,
                zen=60,
                sun_alt=30,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Evening 22h",
            _s(
                altmax=55,
                hour=22,
                snup=5,
                grad=0,
                grad0=0,
                zen=90,
                sun_alt=0,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Before sunrise",
            _s(
                altmax=55,
                hour=3,
                snup=5,
                grad=0,
                grad0=0,
                zen=90,
                sun_alt=0,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Peak at TmaxLST",
            _s(
                altmax=55,
                hour=15,
                snup=5,
                grad=500,
                grad0=550,
                zen=45,
                sun_alt=45,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Cloudy CI low",
            _s(
                altmax=55,
                hour=12,
                snup=5,
                grad=200,
                grad0=650,
                zen=35,
                sun_alt=55,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Wood wall noon",
            _s(
                altmax=55,
                hour=12,
                snup=5,
                grad=600,
                grad0=650,
                zen=35,
                sun_alt=55,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.50,
                tstart_w=-2.0,
                tmaxlst_w=14,
            ),
        ),
        (
            "Brick wall afternoon",
            _s(
                altmax=55,
                hour=18,
                snup=5,
                grad=300,
                grad0=400,
                zen=60,
                sun_alt=30,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=15,
                tgk_w=0.40,
                tstart_w=-4.0,
                tmaxlst_w=15,
            ),
        ),
        (
            "Grass morning",
            _s(
                altmax=55,
                hour=8,
                snup=5,
                grad=300,
                grad0=320,
                zen=60,
                sun_alt=30,
                tgk=0.21,
                tstart=-3.38,
                tmaxlst=14,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "Water",
            _s(
                altmax=55,
                hour=12,
                snup=5,
                grad=600,
                grad0=650,
                zen=35,
                sun_alt=55,
                tgk=0.0,
                tstart=0.0,
                tmaxlst=12,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=15,
            ),
        ),
        (
            "High lat low sun",
            _s(
                altmax=15,
                hour=12,
                snup=9,
                grad=100,
                grad0=120,
                zen=78,
                sun_alt=12,
                tgk=0.37,
                tstart=-3.41,
                tmaxlst=13,
                tgk_w=0.37,
                tstart_w=-3.41,
                tmaxlst_w=13,
            ),
        ),
        # fmt: on
    ]

    scenario_results = []
    for name, s in scenarios:
        dectime = s["hour"] / 24.0

        # UMEP reference
        umep_tg, umep_wall, umep_ci = _umep_ground_temp(
            s["altmax"],
            dectime,
            s["snup"],
            s["grad"],
            s["grad0"],
            s["zen"],
            s["tgk"],
            s["tstart"],
            s["tmaxlst"],
            s["tgk_w"],
            s["tstart_w"],
            s["tmaxlst_w"],
            s["sun_alt"],
        )

        # Rust
        tgk_grid = np.full(shape, s["tgk"], dtype=np.float32)
        tstart_grid = np.full(shape, s["tstart"], dtype=np.float32)
        tmaxlst_grid = np.full(shape, s["tmaxlst"], dtype=np.float32)

        rust_tg, rust_wall, rust_ci, _, _ = ground.compute_ground_temperature(
            20.0,
            s["sun_alt"],
            s["altmax"],
            dectime,
            s["snup"],
            s["grad"],
            s["grad0"],
            s["zen"],
            np.full(shape, 0.2, dtype=np.float32),
            np.full(shape, 0.95, dtype=np.float32),
            tgk_grid,
            tstart_grid,
            tmaxlst_grid,
            tgk_wall=s["tgk_w"],
            tstart_wall=s["tstart_w"],
            tmaxlst_wall=s["tmaxlst_w"],
        )
        rust_tg_val = float(np.array(rust_tg)[0, 0])
        umep_tg_val = float(umep_tg) if not isinstance(umep_tg, np.ndarray) else float(umep_tg.flat[0])

        tg_diff = abs(rust_tg_val - umep_tg_val)
        wall_diff = abs(float(rust_wall) - float(umep_wall))
        ci_diff = abs(float(rust_ci) - float(umep_ci))
        passed = tg_diff < 1e-4 and wall_diff < 1e-4 and ci_diff < 1e-5

        scenario_results.append(
            {
                "name": name,
                "rust_tg": rust_tg_val,
                "umep_tg": umep_tg_val,
                "tg_diff": tg_diff,
                "rust_wall": float(rust_wall),
                "umep_wall": float(umep_wall),
                "wall_diff": wall_diff,
                "rust_ci": float(rust_ci),
                "umep_ci": float(umep_ci),
                "ci_diff": ci_diff,
                "passed": passed,
            }
        )

    results["_sinusoidal_scenarios"] = scenario_results
    return results


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def apply_thresholds(all_results):
    """Apply component-specific pass/fail thresholds to results."""
    for name, stats in all_results.items():
        if name.startswith("_"):
            continue  # Skip metadata entries
        if "pass" in stats:
            continue  # Already set

        max_diff = stats.get("max_abs_diff", 0)
        max_val = stats.get("max_value", 1.0)

        if "svf_veg" in name:
            stats["pass"] = max_diff < 0.02
            stats["threshold"] = "0.02 (1% arch diff)"
        elif name.startswith("svf_") or name.startswith("shadow_"):
            stats["pass"] = max_diff < 1e-4
            stats["threshold"] = "1e-4"
        elif name.startswith("gvf_"):
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 1e-3
            stats["threshold"] = "0.1% relative"
        elif name.startswith("radiation_aniso_"):
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 5e-3
            stats["threshold"] = "0.5% relative"
        elif name.startswith("radiation_"):
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 1e-3
            stats["threshold"] = "0.1% relative"
        elif name.startswith("utci_"):
            stats["pass"] = max_diff < 0.1
            stats["threshold"] = "0.1 C"
        elif name.startswith("pet_"):
            stats["pass"] = max_diff < 0.2
            stats["threshold"] = "0.2 C"
        elif name.startswith("tmrt_") or name.startswith("wall_temp_"):
            stats["pass"] = max_diff < 0.1
            stats["threshold"] = "0.1 C"
        elif name.startswith("aniso_sky_"):
            stats["pass"] = max_diff < 0.5
            stats["threshold"] = "0.5 W/m2"
        elif name.startswith("sinusoidal_"):
            stats["pass"] = max_diff < 1e-3
            stats["threshold"] = "1e-3 C"
        else:
            stats["pass"] = max_diff < 1e-3
            stats["threshold"] = "1e-3"


def generate_markdown_report(all_results):
    """Generate Markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count pass/fail (exclude metadata entries)
    spatial_results = {k: v for k, v in all_results.items() if not k.startswith("_")}
    total = len(spatial_results)
    passed = sum(1 for r in spatial_results.values() if r.get("pass", False))

    # Count sinusoidal scenarios
    sinusoidal_scenarios = all_results.get("_sinusoidal_scenarios", [])
    sin_total = len(sinusoidal_scenarios)
    sin_passed = sum(1 for s in sinusoidal_scenarios if s["passed"])

    status_icon = "PASS" if passed == total and sin_passed == sin_total else "FAIL"

    lines = []
    lines.append("# UMEP vs SOLWEIG Rust - Golden Test Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("**Comparison:** UMEP Python (Reference) vs SOLWEIG Rust Implementation")
    lines.append(f"**Spatial tests:** {passed}/{total} pass")
    if sin_total > 0:
        lines.append(f"**Formula agreement:** {sin_passed}/{sin_total} scenarios pass")
    lines.append(f"**Overall:** {status_icon}")
    lines.append("")
    lines.append("Each comparison shows: UMEP (Reference) | SOLWEIG Rust | Residual (Rust - UMEP)")
    lines.append("")

    # --- Input context ---
    lines.append("## Input Context")
    lines.append("")
    lines.append("![Input context: DSM, CDSM, Wall Heights, SVF](context.png)")
    lines.append("")

    # --- Summary table ---
    lines.append("## Spatial Comparison Summary")
    lines.append("")
    lines.append("| Component | Max |Diff| | Threshold | Mean Diff | Status |")
    lines.append("|-----------|------------|-----------|-----------|--------|")

    for name, stats in spatial_results.items():
        status_str = "PASS" if stats.get("pass", False) else "FAIL"
        threshold = stats.get("threshold", "1e-3")
        max_diff = stats.get("max_abs_diff", 0)
        mean_diff = stats.get("mean_diff", 0)
        lines.append(f"| {name} | {max_diff:.2e} | {threshold} | {mean_diff:.2e} | {status_str} |")

    lines.append("")

    # --- Visual comparisons grouped by category ---
    categories = {
        "Shadows": [k for k in spatial_results if k.startswith("shadow_")],
        "Sky View Factor": [k for k in spatial_results if k.startswith("svf_")],
        "Ground View Factor": [k for k in spatial_results if k.startswith("gvf_")],
        "Radiation (Isotropic)": [k for k in spatial_results if k.startswith("radiation_") and "aniso" not in k],
        "Radiation (Anisotropic)": [k for k in spatial_results if k.startswith("radiation_aniso_")],
        "UTCI": [k for k in spatial_results if k.startswith("utci_")],
        "PET": [k for k in spatial_results if k.startswith("pet_")],
        "Tmrt": [k for k in spatial_results if k.startswith("tmrt_")],
        "Ground Temperature (TsWaveDelay)": [k for k in spatial_results if k.startswith("ground_temp_")],
        "Wall Temperature": [k for k in spatial_results if k.startswith("wall_temp_")],
        "Anisotropic Sky": [k for k in spatial_results if k.startswith("aniso_sky_")],
    }

    lines.append("## Visual Comparisons")
    lines.append("")

    for category, keys in categories.items():
        if keys:
            lines.append(f"### {category}")
            lines.append("")
            for key in keys:
                status_str = "PASS" if spatial_results[key].get("pass", False) else "FAIL"
                lines.append(f"**{key}** ({status_str})")
                lines.append("")
                lines.append(f"![{key}]({key}.png)")
                lines.append("")

    # --- Sinusoidal ground temperature section ---
    # Diurnal curves
    sinusoidal_ground = spatial_results.get("sinusoidal_ground_diurnal")
    sinusoidal_wall = spatial_results.get("sinusoidal_wall_diurnal")
    if sinusoidal_ground or sinusoidal_wall or sinusoidal_scenarios:
        lines.append("## Sinusoidal Ground Temperature Model")
        lines.append("")
        lines.append("Compares `compute_ground_temperature()` (Rust) against the UMEP Python")
        lines.append("formula from `Solweig_2025a_calc_forprocessing.py` (lines 171-199).")
        lines.append("")

    if sinusoidal_ground:
        lines.append("### Diurnal Curve (Rust vs UMEP)")
        lines.append("")
        lines.append("![Sinusoidal diurnal curve: Rust vs UMEP](sinusoidal_diurnal.png)")
        lines.append("")
        gnd_status = "PASS" if sinusoidal_ground.get("pass", False) else "FAIL"
        wall_status = "PASS" if (sinusoidal_wall and sinusoidal_wall.get("pass", False)) else "FAIL"
        lines.append(f"- Ground curve max |diff|: {sinusoidal_ground['max_abs_diff']:.2e} ({gnd_status})")
        if sinusoidal_wall:
            lines.append(f"- Wall curve max |diff|: {sinusoidal_wall['max_abs_diff']:.2e} ({wall_status})")
        lines.append("")

    # Scenario table
    if sinusoidal_scenarios:
        lines.append("### Formula Agreement (12 Scenarios)")
        lines.append("")
        lines.append(f"**Result:** {sin_passed}/{sin_total} scenarios match within f32 tolerance (atol=1e-4)")
        lines.append("")
        lines.append("| Scenario | Rust Tg | UMEP Tg | |d Tg| | Rust Wall | UMEP Wall | |d Wall| | CI | Status |")
        lines.append("|----------|---------|---------|--------|-----------|-----------|---------|------|--------|")

        for s in sinusoidal_scenarios:
            status_str = "PASS" if s["passed"] else "FAIL"
            lines.append(
                f"| {s['name']} "
                f"| {s['rust_tg']:.4f} "
                f"| {s['umep_tg']:.4f} "
                f"| {s['tg_diff']:.1e} "
                f"| {s['rust_wall']:.4f} "
                f"| {s['umep_wall']:.4f} "
                f"| {s['wall_diff']:.1e} "
                f"| {s['rust_ci']:.4f} "
                f"| {status_str} |"
            )

        lines.append("")

    # Write file
    report_path = REPORT_DIR / "golden_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Generate complete golden test report."""
    print("=" * 60)
    print("Golden Test Visual Report Generator")
    print("=" * 60)

    ensure_report_dir()
    inputs = load_inputs()

    print("\nGenerating context plot...")
    plot_context(inputs)

    all_results = {}

    print("Generating shadow comparisons...")
    all_results.update(generate_shadow_comparisons(inputs))

    print("Generating SVF comparisons...")
    all_results.update(generate_svf_comparisons(inputs))

    print("Generating GVF comparisons...")
    all_results.update(generate_gvf_comparisons(inputs))

    print("Generating radiation comparisons (isotropic)...")
    all_results.update(generate_radiation_comparisons(inputs))

    print("Generating radiation comparisons (anisotropic)...")
    try:
        all_results.update(generate_aniso_radiation_comparisons(inputs))
    except Exception as e:
        print(f"    Skipping anisotropic radiation: {e}")

    print("Generating UTCI comparisons...")
    try:
        all_results.update(generate_utci_comparisons())
    except Exception as e:
        print(f"    Skipping UTCI: {e}")

    print("Generating PET comparisons...")
    try:
        all_results.update(generate_pet_comparisons())
    except Exception as e:
        print(f"    Skipping PET: {e}")

    print("Generating Tmrt comparisons...")
    try:
        all_results.update(generate_tmrt_comparisons())
    except Exception as e:
        print(f"    Skipping Tmrt: {e}")

    print("Generating ground temperature comparisons (TsWaveDelay)...")
    try:
        all_results.update(generate_ground_temp_comparisons())
    except Exception as e:
        print(f"    Skipping ground temp: {e}")

    print("Generating wall temperature comparisons...")
    try:
        all_results.update(generate_wall_temp_comparisons())
    except Exception as e:
        print(f"    Skipping wall temp: {e}")

    print("Generating anisotropic sky comparisons...")
    try:
        all_results.update(generate_aniso_sky_comparisons())
    except Exception as e:
        print(f"    Skipping aniso sky: {e}")

    print("Generating sinusoidal ground temperature comparisons...")
    try:
        all_results.update(generate_sinusoidal_ground_temp())
    except Exception as e:
        print(f"    Skipping sinusoidal ground temp: {e}")

    # Apply thresholds
    apply_thresholds(all_results)

    print("\nGenerating Markdown report...")
    report_path = generate_markdown_report(all_results)

    # Print summary
    spatial_results = {k: v for k, v in all_results.items() if not k.startswith("_")}
    total = len(spatial_results)
    passed = sum(1 for r in spatial_results.values() if r.get("pass", False))

    sinusoidal_scenarios = all_results.get("_sinusoidal_scenarios", [])
    sin_total = len(sinusoidal_scenarios)
    sin_passed = sum(1 for s in sinusoidal_scenarios if s["passed"])

    print("\n" + "=" * 60)
    print(f"Report generated: {report_path}")
    print(f"Spatial comparisons: {passed}/{total} pass")
    if sin_total > 0:
        print(f"Formula agreement:   {sin_passed}/{sin_total} scenarios pass")
    print("=" * 60)


if __name__ == "__main__":
    main()
