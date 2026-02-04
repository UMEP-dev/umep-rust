"""
Golden Test Visual Report Generator

Generates a comprehensive visual report comparing current implementation
outputs against golden fixtures for regression testing.

Usage:
    uv run python tests/golden/generate_report.py

Output:
    temp/golden_report/golden_report.html
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
    axes[2].set_title(f"Residual (Rust − UMEP)\nmax|Δ|={diff_max:.2e}")
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
            f"Building Shadows - {name.title()} (az={azimuth}°, alt={altitude}°)",
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
            f"Vegetation Shadows - {name.title()} (az={azimuth}°, alt={altitude}°)",
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
                f"Wall Shadows - {name.title()} (az={azimuth}°, alt={altitude}°)",
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
                f"Wall Sun - {name.title()} (az={azimuth}°, alt={altitude}°)",
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
        ("gvf_lup", "gvf_lup", "GVF Lup (W/m²)", "hot"),
        ("gvfalb", "gvf_alb", "GVF × Albedo", "viridis"),
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
        (kside, "keast", "radiation_kside_e", "Kside East - Isotropic (W/m²)", "YlOrRd"),
        (kside, "ksouth", "radiation_kside_s", "Kside South - Isotropic (W/m²)", "YlOrRd"),
        (lside, "least", "radiation_lside_e", "Lside East - Isotropic (W/m²)", "inferno"),
        (lside, "lsouth", "radiation_lside_s", "Lside South - Isotropic (W/m²)", "inferno"),
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
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands]):
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
        ("keast", "radiation_aniso_kside_e", "Kside East - Anisotropic (W/m²)", "YlOrRd"),
        ("ksouth", "radiation_aniso_kside_s", "Kside South - Anisotropic (W/m²)", "YlOrRd"),
        ("kside_i", "radiation_aniso_kside_i", "Kside Direct - Anisotropic (W/m²)", "YlOrRd"),
        ("kside_d", "radiation_aniso_kside_d", "Kside Diffuse - Anisotropic (W/m²)", "YlOrRd"),
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
        ("least", "radiation_aniso_lside_e", "Lside East - Anisotropic (W/m²)", "inferno"),
        ("lsouth", "radiation_aniso_lside_s", "Lside South - Anisotropic (W/m²)", "inferno"),
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
        f"UTCI Grid (Ta={params['ta']}°C, RH={params['rh']}%)",
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
        f"PET Grid (Ta={params['ta']}°C, RH={params['rh']}%)",
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
        "Tmrt Anisotropic (°C)",
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
        "Tmrt Isotropic (°C)",
        "tmrt_iso.png",
        cmap="RdYlBu_r",
    )
    results["tmrt_iso"] = stats

    return results


def generate_ground_temp_comparisons():
    """Generate ground temperature comparison plots."""
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
        "Ground Temperature Deviation (°C)",
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
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands]):
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
        ("ldown", "Ldown (W/m²)", "inferno"),
        ("lside", "Lside (W/m²)", "inferno"),
        ("kside", "Kside (W/m²)", "YlOrRd"),
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


def generate_html_report(all_results):
    """Generate HTML report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count pass/fail
    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("pass", False))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>UMEP vs SOLWEIG Rust - Golden Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #555; }}
        h4 {{ color: #666; margin-bottom: 5px; }}
        .summary {{ background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
                   padding: 20px; border-radius: 8px; margin-bottom: 25px; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border-radius: 4px; }}
        .component {{ margin-bottom: 30px; padding: 15px; background: #fafafa; border-radius: 8px; }}
        .context {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .context h2 {{ margin-top: 0; color: #2c5282; }}
        .legend {{ font-size: 0.9em; color: #666; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>UMEP vs SOLWEIG Rust - Golden Test Report</h1>
    <div class="summary">
        <strong>Generated:</strong> {timestamp}<br>
        <strong>Comparison:</strong> UMEP Python (Reference) vs SOLWEIG Rust Implementation<br>
        <strong>Result:</strong> <span class="{"pass" if passed == total else "fail"}">
            {passed}/{total} components match</span>
        <p class="legend">Each comparison shows: UMEP (Reference) | SOLWEIG Rust | Residual (Rust − UMEP)</p>
    </div>

    <div class="context">
        <h2>Input Context</h2>
        <p>Input data used for all golden test comparisons:</p>
        <img src="context.png" alt="Input context: DSM, CDSM, Wall Heights, SVF">
    </div>

    <h2>Summary Table</h2>
    <table>
        <tr><th>Component</th><th>Max |Diff|</th><th>Threshold</th><th>Mean Diff</th><th>Status</th></tr>
"""

    for name, stats in all_results.items():
        if "pass" in stats:
            status = '<span class="pass">✓ PASS</span>' if stats["pass"] else '<span class="fail">✗ FAIL</span>'
        else:
            status = '<span style="color:#888">N/A</span>'
        threshold = stats.get("threshold", "1e-3")
        max_diff = stats.get("max_abs_diff", stats.get("max", 0))
        mean_diff = stats.get("mean_diff", stats.get("mean", 0))
        html += f"""        <tr>
            <td>{name}</td>
            <td>{max_diff:.2e}</td>
            <td>{threshold}</td>
            <td>{mean_diff:.2e}</td>
            <td>{status}</td>
        </tr>
"""

    html += """    </table>

    <h2>Visual Comparisons: UMEP (Reference) vs SOLWEIG Rust</h2>
"""

    # Group by category
    categories = {
        "Shadows": [k for k in all_results if k.startswith("shadow_")],
        "Sky View Factor": [k for k in all_results if k.startswith("svf_")],
        "Ground View Factor": [k for k in all_results if k.startswith("gvf_")],
        "Radiation Isotropic": [k for k in all_results if k.startswith("radiation_") and "aniso" not in k],
        "Radiation Anisotropic": [k for k in all_results if k.startswith("radiation_aniso_")],
        "UTCI": [k for k in all_results if k.startswith("utci_")],
        "PET": [k for k in all_results if k.startswith("pet_")],
        "Tmrt": [k for k in all_results if k.startswith("tmrt_")],
        "Ground Temperature": [k for k in all_results if k.startswith("ground_temp_")],
        "Wall Temperature": [k for k in all_results if k.startswith("wall_temp_")],
        "Anisotropic Sky": [k for k in all_results if k.startswith("aniso_sky_")],
    }

    for category, keys in categories.items():
        if keys:
            html += f"""    <h3>{category}</h3>
"""
            for key in keys:
                html += f"""    <div class="component">
        <h4>{key}</h4>
        <img src="{key}.png" alt="{key}">
    </div>
"""

    html += """</body>
</html>
"""

    with open(REPORT_DIR / "golden_report.html", "w") as f:
        f.write(html)


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

    print("Generating ground temperature comparisons...")
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

    # Apply component-specific pass/fail thresholds
    # Thresholds match the pytest golden tests for consistency
    for name, stats in all_results.items():
        if "pass" in stats:
            continue  # Already set

        max_diff = stats.get("max_abs_diff", 0)
        max_val = stats.get("max_value", 1.0)

        if "svf_veg" in name:
            stats["pass"] = max_diff < 0.02
            stats["threshold"] = "0.02 (1% architecture diff)"
        elif name.startswith("svf_") or name.startswith("shadow_"):
            stats["pass"] = max_diff < 1e-4
            stats["threshold"] = "1e-4"
        elif name.startswith("gvf_"):
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 1e-3
            stats["threshold"] = "0.1% relative"
        elif name.startswith("radiation_aniso_"):
            # Anisotropic radiation has more numerical variation
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 5e-3  # 0.5% relative
            stats["threshold"] = "0.5% relative"
        elif name.startswith("radiation_"):
            relative_diff = max_diff / max_val if max_val > 0 else max_diff
            stats["pass"] = relative_diff < 1e-3
            stats["threshold"] = "0.1% relative"
        elif name.startswith("utci_"):
            stats["pass"] = max_diff < 0.1  # 0.1°C
            stats["threshold"] = "0.1°C"
        elif name.startswith("pet_"):
            stats["pass"] = max_diff < 0.2  # 0.2°C (iterative solver)
            stats["threshold"] = "0.2°C"
        elif name.startswith("tmrt_") or name.startswith("wall_temp_"):
            stats["pass"] = max_diff < 0.1  # 0.1°C
            stats["threshold"] = "0.1°C"
        elif name.startswith("aniso_sky_"):
            stats["pass"] = max_diff < 0.5  # 0.5 W/m²
            stats["threshold"] = "0.5 W/m²"
        else:
            stats["pass"] = max_diff < 1e-3
            stats["threshold"] = "1e-3"

    print("\nGenerating HTML report...")
    generate_html_report(all_results)

    # Print summary
    total = len(all_results)
    passed = sum(1 for r in all_results.values() if r.get("pass", False))

    print("\n" + "=" * 60)
    print(f"Report generated: {REPORT_DIR / 'golden_report.html'}")
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
