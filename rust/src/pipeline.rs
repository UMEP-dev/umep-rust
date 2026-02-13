//! Fused timestep pipeline — single FFI entrance/exit per timestep.
//!
//! Orchestrates: shadows → ground_temp → GVF → thermal_delay → radiation → Tmrt
//! All intermediate arrays stay as ndarray::Array2<f32> — never cross FFI boundary.
//!
//! Supports both isotropic and anisotropic (Perez) sky models.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use crate::ground::{compute_ground_temperature_pure, ts_wave_delay_batch_pure, GroundTempResult};
use crate::gvf::{gvf_calc_pure, gvf_calc_with_cache, GvfResultPure};
use crate::gvf_geometry::{precompute_gvf_geometry, GvfGeometryCache};
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};
use crate::sky::{anisotropic_sky_pure, cylindric_wedge_pure_masked};
use crate::tmrt::compute_tmrt_pure;
use crate::vegetation::{kside_veg_isotropic_pure, lside_veg_pure};

#[cfg(feature = "gpu")]
use crate::gpu::AnisoGpuContext;

use std::time::Instant;

const PI: f32 = std::f32::consts::PI;
const SBC: f32 = 5.67e-8;
const KELVIN_OFFSET: f32 = 273.15;

/// Check once per process whether timing output is enabled (``SOLWEIG_TIMING=1``).
fn timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SOLWEIG_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(feature = "gpu")]
fn aniso_gpu_overlap_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SOLWEIG_ANISO_GPU_OVERLAP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// ── GPU anisotropic sky context (lazy-initialized, shares device with shadows) ──

#[cfg(feature = "gpu")]
static ANISO_GPU_CONTEXT: OnceLock<Option<AnisoGpuContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
static ANISO_GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "gpu")]
fn get_aniso_gpu_context() -> Option<&'static AnisoGpuContext> {
    if !ANISO_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    ANISO_GPU_CONTEXT
        .get_or_init(|| {
            // Share device/queue from the shadow GPU context
            let shadow_ctx = crate::shadowing::get_gpu_context()?;
            let device = shadow_ctx.device.clone();
            let queue = shadow_ctx.queue.clone();
            let ctx = AnisoGpuContext::new(device, queue);
            eprintln!("[GPU] Anisotropic sky GPU context initialized");
            Some(ctx)
        })
        .as_ref()
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Enable GPU acceleration for anisotropic sky computation
pub fn enable_aniso_gpu() {
    ANISO_GPU_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] Anisotropic sky GPU acceleration enabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Disable GPU acceleration for anisotropic sky computation (CPU fallback)
pub fn disable_aniso_gpu() {
    ANISO_GPU_ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
    eprintln!("[GPU] Anisotropic sky GPU acceleration disabled");
}

#[cfg(feature = "gpu")]
#[pyfunction]
/// Check if GPU acceleration is enabled for anisotropic sky
pub fn is_aniso_gpu_enabled() -> bool {
    ANISO_GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

// ── Input structs (created once in Python, passed by reference) ───────────

/// Weather scalars for a single timestep.
#[pyclass]
#[derive(Clone)]
pub struct WeatherScalars {
    #[pyo3(get, set)]
    pub sun_azimuth: f32,
    #[pyo3(get, set)]
    pub sun_altitude: f32,
    #[pyo3(get, set)]
    pub sun_zenith: f32,
    #[pyo3(get, set)]
    pub ta: f32,
    #[pyo3(get, set)]
    pub rh: f32,
    #[pyo3(get, set)]
    pub global_rad: f32,
    #[pyo3(get, set)]
    pub direct_rad: f32,
    #[pyo3(get, set)]
    pub diffuse_rad: f32,
    #[pyo3(get, set)]
    pub altmax: f32,
    #[pyo3(get, set)]
    pub clearness_index: f32,
    #[pyo3(get, set)]
    pub dectime: f32,
    #[pyo3(get, set)]
    pub snup: f32,
    #[pyo3(get, set)]
    pub rad_g0: f32,
    #[pyo3(get, set)]
    pub zen_deg: f32,
    #[pyo3(get, set)]
    pub psi: f32,
    #[pyo3(get, set)]
    pub is_daytime: bool,
    #[pyo3(get, set)]
    pub jday: i32,
    #[pyo3(get, set)]
    pub patch_option: i32,
}

#[pymethods]
impl WeatherScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sun_azimuth: f32,
        sun_altitude: f32,
        sun_zenith: f32,
        ta: f32,
        rh: f32,
        global_rad: f32,
        direct_rad: f32,
        diffuse_rad: f32,
        altmax: f32,
        clearness_index: f32,
        dectime: f32,
        snup: f32,
        rad_g0: f32,
        zen_deg: f32,
        psi: f32,
        is_daytime: bool,
        jday: i32,
        patch_option: i32,
    ) -> Self {
        Self {
            sun_azimuth,
            sun_altitude,
            sun_zenith,
            ta,
            rh,
            global_rad,
            direct_rad,
            diffuse_rad,
            altmax,
            clearness_index,
            dectime,
            snup,
            rad_g0,
            zen_deg,
            psi,
            is_daytime,
            jday,
            patch_option,
        }
    }
}

/// Human body parameters.
#[pyclass]
#[derive(Clone)]
pub struct HumanScalars {
    #[pyo3(get, set)]
    pub height: f32,
    #[pyo3(get, set)]
    pub abs_k: f32,
    #[pyo3(get, set)]
    pub abs_l: f32,
    #[pyo3(get, set)]
    pub is_standing: bool,
}

#[pymethods]
impl HumanScalars {
    #[new]
    pub fn new(height: f32, abs_k: f32, abs_l: f32, is_standing: bool) -> Self {
        Self {
            height,
            abs_k,
            abs_l,
            is_standing,
        }
    }
}

/// Configuration scalars (constant across timesteps).
#[pyclass]
#[derive(Clone)]
pub struct ConfigScalars {
    #[pyo3(get, set)]
    pub pixel_size: f32,
    #[pyo3(get, set)]
    pub max_height: f32,
    #[pyo3(get, set)]
    pub albedo_wall: f32,
    #[pyo3(get, set)]
    pub emis_wall: f32,
    #[pyo3(get, set)]
    pub tgk_wall: f32,
    #[pyo3(get, set)]
    pub tstart_wall: f32,
    #[pyo3(get, set)]
    pub tmaxlst_wall: f32,
    #[pyo3(get, set)]
    pub use_veg: bool,
    #[pyo3(get, set)]
    pub has_walls: bool,
    #[pyo3(get, set)]
    pub conifer: bool,
    #[pyo3(get, set)]
    pub use_anisotropic: bool,
}

#[pymethods]
impl ConfigScalars {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pixel_size: f32,
        max_height: f32,
        albedo_wall: f32,
        emis_wall: f32,
        tgk_wall: f32,
        tstart_wall: f32,
        tmaxlst_wall: f32,
        use_veg: bool,
        has_walls: bool,
        conifer: bool,
        use_anisotropic: bool,
    ) -> Self {
        Self {
            pixel_size,
            max_height,
            albedo_wall,
            emis_wall,
            tgk_wall,
            tstart_wall,
            tmaxlst_wall,
            use_veg,
            has_walls,
            conifer,
            use_anisotropic,
        }
    }
}

// ── Output struct ──────────────────────────────────────────────────────────

/// Result from a single fused timestep.
#[pyclass]
pub struct TimestepResult {
    #[pyo3(get)]
    pub tmrt: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub shadow: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub kdown: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub kup: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub ldown: Option<Py<PyArray2<f32>>>,
    #[pyo3(get)]
    pub lup: Option<Py<PyArray2<f32>>>,
    // Updated thermal state arrays (Python extracts and passes back next timestep)
    #[pyo3(get)]
    pub timeadd: f32,
    #[pyo3(get)]
    pub tgmap1: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_e: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_s: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_w: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgmap1_n: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub tgout1: Py<PyArray2<f32>>,
}

/// Raw result struct with owned arrays (no Python types — Send-safe).
struct TimestepResultRaw {
    tmrt: Array2<f32>,
    shadow: Array2<f32>,
    kdown: Array2<f32>,
    kup: Array2<f32>,
    ldown: Array2<f32>,
    lup: Array2<f32>,
    timeadd: f32,
    tgmap1: Array2<f32>,
    tgmap1_e: Array2<f32>,
    tgmap1_s: Array2<f32>,
    tgmap1_w: Array2<f32>,
    tgmap1_n: Array2<f32>,
    tgout1: Array2<f32>,
}

const OUT_SHADOW: u32 = 1 << 0;
const OUT_KDOWN: u32 = 1 << 1;
const OUT_KUP: u32 = 1 << 2;
const OUT_LDOWN: u32 = 1 << 3;
const OUT_LUP: u32 = 1 << 4;
const OUT_ALL: u32 = OUT_SHADOW | OUT_KDOWN | OUT_KUP | OUT_LDOWN | OUT_LUP;

/// Release the GIL for a closure whose captured state may not be `Send`.
///
/// # Safety
/// Caller must guarantee that all borrowed data remains alive for the duration
/// of the closure (i.e. the Python objects backing any `ArrayView` are not
/// deallocated while the GIL is released).
unsafe fn allow_threads_unchecked<T: Send, F: FnOnce() -> T>(py: Python, f: F) -> T {
    // Move f to the heap and erase through usize so the auto-Send derivation
    // for the closure sees only Send types (usize), not the non-Send F.
    let raw = Box::into_raw(Box::new(f));
    let addr = raw as usize;
    py.allow_threads(move || unsafe {
        let f = *Box::from_raw(addr as *mut F);
        f()
    })
}

// ── Radiation helpers (ported from Python physics) ─────────────────────────

/// Compute sky emissivity (Jonsson et al. 2006).
#[inline]
fn compute_esky(ta: f32, rh: f32) -> f32 {
    let ta_k = ta + KELVIN_OFFSET;
    let ea = 6.107 * 10.0_f32.powf((7.5 * ta) / (237.3 + ta)) * (rh / 100.0);
    let msteg = 46.5 * (ea / ta_k);
    1.0 - (1.0 + msteg) * (-((1.2 + 3.0 * msteg) as f32).sqrt()).exp()
}

/// Compute Kup (ground-reflected shortwave) — Kup_veg_2015a.
///
/// Returns (kup, kup_e, kup_s, kup_w, kup_n) as owned arrays.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn compute_kup(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    altitude: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_b: f32,
    f_sh: ArrayView2<f32>,
    gvfalb: ArrayView2<f32>,
    gvfalb_e: ArrayView2<f32>,
    gvfalb_s: ArrayView2<f32>,
    gvfalb_w: ArrayView2<f32>,
    gvfalb_n: ArrayView2<f32>,
    gvfalbnosh: ArrayView2<f32>,
    gvfalbnosh_e: ArrayView2<f32>,
    gvfalbnosh_s: ArrayView2<f32>,
    gvfalbnosh_w: ArrayView2<f32>,
    gvfalbnosh_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let rad_i_sin_alt = rad_i * (altitude * PI / 180.0).sin();

    let shape = svfbuveg.dim();
    let mut kup = Array2::<f32>::zeros(shape);
    let mut kup_e = Array2::<f32>::zeros(shape);
    let mut kup_s = Array2::<f32>::zeros(shape);
    let mut kup_w = Array2::<f32>::zeros(shape);
    let mut kup_n = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut kup)
        .and(&mut kup_e)
        .and(&mut kup_s)
        .and(&mut kup_w)
        .and(&mut kup_n)
        .par_for_each(|(r, c), k, ke, ks, kw, kn| {
            if valid[[r, c]] == 0 {
                *k = f32::NAN;
                *ke = f32::NAN;
                *ks = f32::NAN;
                *kw = f32::NAN;
                *kn = f32::NAN;
                return;
            }
            let sv = svfbuveg[[r, c]];
            let fsh = f_sh[[r, c]];
            let ct = rad_d * sv + albedo_b * (1.0 - sv) * (rad_g * (1.0 - fsh) + rad_d * fsh);
            *k = gvfalb[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh[[r, c]];
            *ke = gvfalb_e[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_e[[r, c]];
            *ks = gvfalb_s[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_s[[r, c]];
            *kw = gvfalb_w[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_w[[r, c]];
            *kn = gvfalb_n[[r, c]] * rad_i_sin_alt + ct * gvfalbnosh_n[[r, c]];
        });

    (kup, kup_e, kup_s, kup_w, kup_n)
}

/// Compute Ldown (downwelling longwave) — Jonsson et al. 2006.
#[allow(clippy::too_many_arguments)]
fn compute_ldown(
    esky: f32,
    ta: f32,
    tg_wall: f32,
    svf: ArrayView2<f32>,
    svf_veg: ArrayView2<f32>,
    svf_aveg: ArrayView2<f32>,
    emis_wall: f32,
    ci: f32,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let ta_k = ta + KELVIN_OFFSET;
    let ta_k4 = ta_k.powi(4);
    let tg_wall_k4 = (ta + tg_wall + KELVIN_OFFSET).powi(4);
    let shape = svf.dim();
    let mut ldown = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut ldown).par_for_each(|(r, c), ld| {
        if valid[[r, c]] == 0 {
            *ld = f32::NAN;
            return;
        }
        let sv = svf[[r, c]];
        let sv_veg = svf_veg[[r, c]];
        let sv_aveg = svf_aveg[[r, c]];

        let val = (sv + sv_veg - 1.0) * esky * SBC * ta_k4
            + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
            + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
            + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * esky * SBC * ta_k4;

        if ci < 0.95 {
            let c_cloud = 1.0 - ci;
            let val_cloudy = (sv + sv_veg - 1.0) * SBC * ta_k4
                + (2.0 - sv_veg - sv_aveg) * emis_wall * SBC * ta_k4
                + (sv_aveg - sv) * emis_wall * SBC * tg_wall_k4
                + (2.0 - sv - sv_veg) * (1.0 - emis_wall) * SBC * ta_k4;
            *ld = val * (1.0 - c_cloud) + val_cloudy * c_cloud;
        } else {
            *ld = val;
        }
    });

    ldown
}

/// Compute Kdown (downwelling shortwave).
#[allow(clippy::too_many_arguments)]
fn compute_kdown(
    rad_i: f32,
    rad_d: f32,
    rad_g: f32,
    shadow: ArrayView2<f32>,
    sin_alt: f32,
    svfbuveg: ArrayView2<f32>,
    albedo_wall: f32,
    f_sh: ArrayView2<f32>,
    drad: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let shape = shadow.dim();
    let mut kdown = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut kdown).par_for_each(|(r, c), kd| {
        if valid[[r, c]] == 0 {
            *kd = f32::NAN;
            return;
        }
        *kd = rad_i * shadow[[r, c]] * sin_alt
            + drad[[r, c]]
            + albedo_wall
                * (1.0 - svfbuveg[[r, c]])
                * (rad_g * (1.0 - f_sh[[r, c]]) + rad_d * f_sh[[r, c]]);
    });

    kdown
}

/// Cached ASVF (`acos(sqrt(clamp(svf, 0, 1)))`) for a static SVF raster.
///
/// SVF is invariant across timesteps for a given surface/tile, so recomputing
/// ASVF every timestep is wasted work in anisotropic mode.
fn asvf_for_svf_cached(svf: ArrayView2<f32>) -> Arc<Vec<f32>> {
    const MAX_ENTRIES: usize = 16;

    type AsvfKey = (usize, usize, u64);
    #[derive(Default)]
    struct AsvfCache {
        map: HashMap<AsvfKey, Arc<Vec<f32>>>,
        lru: VecDeque<AsvfKey>,
    }

    fn fnv1a_u64(mut hash: u64, word: u64) -> u64 {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
        for b in word.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn svf_key(svf: ArrayView2<f32>) -> AsvfKey {
        const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
        let mut hash = FNV_OFFSET;
        hash = fnv1a_u64(hash, svf.nrows() as u64);
        hash = fnv1a_u64(hash, svf.ncols() as u64);
        for &v in svf.iter() {
            hash = fnv1a_u64(hash, v.to_bits() as u64);
        }
        (svf.nrows(), svf.ncols(), hash)
    }

    let key = svf_key(svf);
    static CACHE: OnceLock<Mutex<AsvfCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(AsvfCache::default()));

    let mut guard = cache.lock().expect("ASVF cache mutex poisoned");
    if let Some(hit) = guard.map.get(&key).cloned() {
        if let Some(pos) = guard.lru.iter().position(|k| *k == key) {
            guard.lru.remove(pos);
        }
        guard.lru.push_back(key);
        return hit;
    }

    let data = svf
        .iter()
        .map(|&v| v.clamp(0.0, 1.0).sqrt().acos())
        .collect::<Vec<f32>>();
    let entry = Arc::new(data);

    while guard.map.len() >= MAX_ENTRIES {
        if let Some(oldest) = guard.lru.pop_front() {
            guard.map.remove(&oldest);
        } else {
            break;
        }
    }
    guard.map.insert(key, entry.clone());
    guard.lru.push_back(key);
    entry
}

/// Directional longwave side components for anisotropic mode.
///
/// In this mode `lside_veg` returns `lup_dir * 0.5` per direction with NaN on
/// invalid pixels. Compute it directly to avoid the heavier isotropic pathway.
fn lside_dirs_aniso_from_lup(
    lup_e: ArrayView2<f32>,
    lup_s: ArrayView2<f32>,
    lup_w: ArrayView2<f32>,
    lup_n: ArrayView2<f32>,
    valid: ArrayView2<u8>,
) -> crate::vegetation::LsideVegPureResult {
    let shape = lup_e.dim();
    let mut least = Array2::<f32>::zeros(shape);
    let mut lsouth = Array2::<f32>::zeros(shape);
    let mut lwest = Array2::<f32>::zeros(shape);
    let mut lnorth = Array2::<f32>::zeros(shape);

    Zip::indexed(&mut least)
        .and(&mut lsouth)
        .and(&mut lwest)
        .and(&mut lnorth)
        .par_for_each(|(r, c), le, ls, lw, ln| {
            if valid[[r, c]] == 0 {
                *le = f32::NAN;
                *ls = f32::NAN;
                *lw = f32::NAN;
                *ln = f32::NAN;
                return;
            }
            *le = lup_e[[r, c]] * 0.5;
            *ls = lup_s[[r, c]] * 0.5;
            *lw = lup_w[[r, c]] * 0.5;
            *ln = lup_n[[r, c]] * 0.5;
        });

    crate::vegetation::LsideVegPureResult {
        least,
        lsouth,
        lwest,
        lnorth,
    }
}

struct PatchOptionLut {
    altitudes: Arc<Vec<f32>>,
    azimuths: Arc<Vec<f32>>,
    steradians: Arc<Vec<f32>>,
    altitude_sin: Arc<Vec<f32>>,
}

/// Cached patch LUTs keyed by `patch_option`.
///
/// Geometry is already cached in the Perez module; this cache adds derived
/// `sin(altitude)` values so anisotropic timesteps can skip per-patch trig.
fn patch_lut_for_option_cached(patch_option: i32) -> Arc<PatchOptionLut> {
    static CACHE: OnceLock<Mutex<HashMap<i32, Arc<PatchOptionLut>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let mut guard = cache.lock().expect("patch LUT cache mutex poisoned");
    guard
        .entry(patch_option)
        .or_insert_with(|| {
            let (altitudes, azimuths, steradians) =
                crate::perez::patch_alt_azi_steradians_for_patch_option(patch_option);
            let deg2rad = PI / 180.0;
            let altitude_sin = Arc::new(
                altitudes
                    .iter()
                    .map(|alt| (alt * deg2rad).sin())
                    .collect::<Vec<f32>>(),
            );
            Arc::new(PatchOptionLut {
                altitudes,
                azimuths,
                steradians,
                altitude_sin,
            })
        })
        .clone()
}

/// Compute anisotropic diffuse luminance sum directly from bitpacked shadow matrices.
///
/// Equivalent to:
///   diffsh = shmat_bit - (1 - vegshmat_bit) * (1 - psi)
///   ani_lum = sum_i(diffsh[:,:,i] * lv_lum[i])
///
/// but avoids allocating a full (rows, cols, patches) float array.
fn compute_ani_lum_from_packed(
    shmat: ArrayView3<u8>,
    vegshmat: ArrayView3<u8>,
    lv_lum: ArrayView1<f32>,
    psi: f32,
    valid: ArrayView2<u8>,
) -> Array2<f32> {
    let (rows, cols, _) = shmat.dim();
    let n_patches = lv_lum.len();
    let mut out = Array2::<f32>::zeros((rows, cols));

    let ncols = cols;
    out.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, v)| {
            let r = idx / ncols;
            let c = idx % ncols;

            if valid[[r, c]] == 0 {
                *v = f32::NAN;
                return;
            }

            let mut sum = 0.0_f32;
            for i in 0..n_patches {
                let byte = i >> 3;
                let bit = i & 7;
                let sh = ((shmat[[r, c, byte]] >> bit) & 1) as f32;
                let vsh = ((vegshmat[[r, c, byte]] >> bit) & 1) as f32;
                let diff = sh - (1.0 - vsh) * (1.0 - psi);
                sum += diff * lv_lum[i];
            }
            *v = sum;
        });

    out
}

// ── GVF Geometry Cache (opaque handle for Python) ─────────────────────────

/// Opaque handle to a precomputed GVF geometry cache.
///
/// Created once per DSM via `precompute_gvf_cache()`, then passed to
/// `compute_timestep()` on subsequent calls to skip building ray-tracing.
#[pyclass]
pub struct PyGvfGeometryCache {
    pub(crate) inner: GvfGeometryCache,
}

/// Precompute GVF geometry cache for a given set of surface arrays.
///
/// This runs the building ray-trace once (18 azimuths, parallelized).
/// The returned cache is passed to `compute_timestep()` to skip geometry
/// on subsequent timesteps with the same DSM.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn precompute_gvf_cache(
    buildings: PyReadonlyArray2<f32>,
    wall_asp: PyReadonlyArray2<f32>,
    wall_ht: PyReadonlyArray2<f32>,
    alb_grid: PyReadonlyArray2<f32>,
    pixel_size: f32,
    human_height: f32,
    wall_albedo: f32,
) -> PyResult<PyGvfGeometryCache> {
    let first_ht = human_height.round().max(1.0);
    let second_ht = human_height * 20.0;

    let cache = precompute_gvf_geometry(
        buildings.as_array(),
        wall_asp.as_array(),
        wall_ht.as_array(),
        alb_grid.as_array(),
        pixel_size,
        first_ht,
        second_ht,
        wall_albedo,
    );

    Ok(PyGvfGeometryCache { inner: cache })
}

// ── Main fused timestep function ───────────────────────────────────────────

/// Compute a single daytime timestep entirely in Rust.
///
/// All intermediate arrays stay as ndarray::Array2<f32> — only the final
/// results cross back to Python.
///
/// Parameters are grouped into structs to keep the signature manageable:
/// - weather: Per-timestep scalars (sun position, temperature, radiation)
/// - human: Body parameters (height, posture, absorptivities)
/// - config: Constants (pixel_size, wall materials)
/// - Surface/SVF arrays: Borrowed from Python (zero-copy on input)
/// - Thermal state: Carried forward between timesteps
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_timestep(
    py: Python,
    // Scalar parameter structs
    weather: &WeatherScalars,
    human: &HumanScalars,
    config: &ConfigScalars,
    // Optional GVF geometry cache (skip building ray-tracing if provided)
    gvf_cache: Option<&PyGvfGeometryCache>,
    // Surface arrays (constant across timesteps, borrowed)
    dsm: PyReadonlyArray2<f32>,
    cdsm: Option<PyReadonlyArray2<f32>>,
    tdsm: Option<PyReadonlyArray2<f32>>,
    bush: Option<PyReadonlyArray2<f32>>,
    wall_ht: Option<PyReadonlyArray2<f32>>,
    wall_asp: Option<PyReadonlyArray2<f32>>,
    // SVF arrays (constant across timesteps, borrowed)
    svf: PyReadonlyArray2<f32>,
    svf_n: PyReadonlyArray2<f32>,
    svf_e: PyReadonlyArray2<f32>,
    svf_s: PyReadonlyArray2<f32>,
    svf_w: PyReadonlyArray2<f32>,
    svf_veg: PyReadonlyArray2<f32>,
    svf_veg_n: PyReadonlyArray2<f32>,
    svf_veg_e: PyReadonlyArray2<f32>,
    svf_veg_s: PyReadonlyArray2<f32>,
    svf_veg_w: PyReadonlyArray2<f32>,
    svf_aveg: PyReadonlyArray2<f32>,
    svf_aveg_n: PyReadonlyArray2<f32>,
    svf_aveg_e: PyReadonlyArray2<f32>,
    svf_aveg_s: PyReadonlyArray2<f32>,
    svf_aveg_w: PyReadonlyArray2<f32>,
    svfbuveg: PyReadonlyArray2<f32>,
    svfalfa: PyReadonlyArray2<f32>,
    // Land cover property grids (constant across timesteps, borrowed)
    alb_grid: PyReadonlyArray2<f32>,
    emis_grid: PyReadonlyArray2<f32>,
    tgk_grid: PyReadonlyArray2<f32>,
    tstart_grid: PyReadonlyArray2<f32>,
    tmaxlst_grid: PyReadonlyArray2<f32>,
    // Buildings mask for GVF
    buildings: PyReadonlyArray2<f32>,
    lc_grid: Option<PyReadonlyArray2<f32>>,
    // Anisotropic sky inputs (None for isotropic)
    shmat: Option<PyReadonlyArray3<u8>>,
    vegshmat: Option<PyReadonlyArray3<u8>>,
    vbshmat: Option<PyReadonlyArray3<u8>>,
    // Thermal state (mutable, updated each timestep)
    firstdaytime: i32,
    timeadd: f32,
    timestep_dec: f32,
    tgmap1: PyReadonlyArray2<f32>,
    tgmap1_e: PyReadonlyArray2<f32>,
    tgmap1_s: PyReadonlyArray2<f32>,
    tgmap1_w: PyReadonlyArray2<f32>,
    tgmap1_n: PyReadonlyArray2<f32>,
    tgout1: PyReadonlyArray2<f32>,
    // Valid pixel mask (1=valid, 0=NaN/nodata — skip computation for invalid pixels)
    valid_mask: PyReadonlyArray2<u8>,
    // Optional output selection bitmask for Python conversion (tmrt always returned)
    output_mask: Option<u32>,
) -> PyResult<TimestepResult> {
    // Borrow all arrays (zero-copy from numpy)
    let valid_v = valid_mask.as_array();
    let dsm_v = dsm.as_array();
    let cdsm_v = cdsm.as_ref().map(|a| a.as_array());
    let tdsm_v = tdsm.as_ref().map(|a| a.as_array());
    let bush_v = bush.as_ref().map(|a| a.as_array());
    let wall_ht_v = wall_ht.as_ref().map(|a| a.as_array());
    let wall_asp_v = wall_asp.as_ref().map(|a| a.as_array());
    let svf_v = svf.as_array();
    let svf_n_v = svf_n.as_array();
    let svf_e_v = svf_e.as_array();
    let svf_s_v = svf_s.as_array();
    let svf_w_v = svf_w.as_array();
    let svf_veg_v = svf_veg.as_array();
    let svf_veg_n_v = svf_veg_n.as_array();
    let svf_veg_e_v = svf_veg_e.as_array();
    let svf_veg_s_v = svf_veg_s.as_array();
    let svf_veg_w_v = svf_veg_w.as_array();
    let svf_aveg_v = svf_aveg.as_array();
    let svf_aveg_n_v = svf_aveg_n.as_array();
    let svf_aveg_e_v = svf_aveg_e.as_array();
    let svf_aveg_s_v = svf_aveg_s.as_array();
    let svf_aveg_w_v = svf_aveg_w.as_array();
    let svfbuveg_v = svfbuveg.as_array();
    let svfalfa_v = svfalfa.as_array();
    let alb_grid_v = alb_grid.as_array();
    let emis_grid_v = emis_grid.as_array();
    let tgk_grid_v = tgk_grid.as_array();
    let tstart_grid_v = tstart_grid.as_array();
    let tmaxlst_grid_v = tmaxlst_grid.as_array();
    let buildings_v = buildings.as_array();
    let lc_grid_v = lc_grid.as_ref().map(|a| a.as_array());
    let tgmap1_v = tgmap1.as_array();
    let tgmap1_e_v = tgmap1_e.as_array();
    let tgmap1_s_v = tgmap1_s.as_array();
    let tgmap1_w_v = tgmap1_w.as_array();
    let tgmap1_n_v = tgmap1_n.as_array();
    let tgout1_v = tgout1.as_array();

    // Borrow anisotropic arrays (if provided)
    let shmat_v = shmat.as_ref().map(|a| a.as_array());
    let vegshmat_v = vegshmat.as_ref().map(|a| a.as_array());
    let vbshmat_v = vbshmat.as_ref().map(|a| a.as_array());
    let output_mask_bits = output_mask.unwrap_or(OUT_ALL);

    // Extract GVF cache reference (pure Rust data) before releasing the GIL
    let gvf_cache_inner = gvf_cache.map(|c| &c.inner);

    // SAFETY: All array views borrow from PyReadonlyArray parameters that are alive
    // for the entire function call. Releasing the GIL only allows other Python
    // threads to run — it does not invalidate our borrows or trigger GC of the
    // backing numpy arrays.
    let raw = unsafe {
        allow_threads_unchecked(py, || {
            let shape = dsm_v.dim();

            // Wall aspect in radians for shadows
            let wall_asp_rad: Option<Array2<f32>> = wall_asp_v.map(|a| a.mapv(|d| d * PI / 180.0));
            let wall_asp_rad_view = wall_asp_rad.as_ref().map(|a| a.view());

            // ── Step 1: Shadows ──────────────────────────────────────────────────
            let t_shadow = Instant::now();
            let shadow_result: ShadowingResultRust = calculate_shadows_rust(
                weather.sun_azimuth,
                weather.sun_altitude,
                config.pixel_size,
                config.max_height,
                dsm_v,
                if config.use_veg { cdsm_v } else { None },
                if config.use_veg { tdsm_v } else { None },
                if config.use_veg { bush_v } else { None },
                if config.has_walls { wall_ht_v } else { None },
                if config.has_walls {
                    wall_asp_rad_view
                } else {
                    None
                },
                None, // walls_scheme
                None, // aspect_scheme
                3.0,  // min_sun_altitude
            );

            // Combine shadows with vegetation transmissivity
            let bldg_sh = &shadow_result.bldg_sh;
            let shadow = if config.use_veg {
                let veg_sh = &shadow_result.veg_sh;
                bldg_sh - &((1.0 - veg_sh) * (1.0 - weather.psi))
            } else {
                bldg_sh.clone()
            };
            let shadow_f32 = shadow;

            let wallsun = shadow_result
                .wall_sun
                .unwrap_or_else(|| Array2::zeros(shape));

            let shadow_dur = t_shadow.elapsed();

            // ── Step 2: Ground Temperature ───────────────────────────────────────
            let t_ground = Instant::now();
            let ground: GroundTempResult = compute_ground_temperature_pure(
                weather.sun_altitude,
                weather.altmax,
                weather.dectime,
                weather.snup,
                weather.global_rad,
                weather.rad_g0,
                weather.zen_deg,
                tgk_grid_v,
                tstart_grid_v,
                tmaxlst_grid_v,
                config.tgk_wall,
                config.tstart_wall,
                config.tmaxlst_wall,
            );

            let ground_dur = t_ground.elapsed();

            // ── Step 3: GVF ─────────────────────────────────────────────────────
            let t_gvf = Instant::now();
            let first = {
                let h = human.height.round();
                if h == 0.0 {
                    1.0
                } else {
                    h
                }
            };
            let second = (human.height * 20.0).round();

            let gvf: GvfResultPure = if config.has_walls {
                if let Some(cache) = gvf_cache_inner {
                    // Use cached geometry — thermal-only pass
                    gvf_calc_with_cache(
                        cache,
                        wallsun.view(),
                        buildings_v,
                        shadow_f32.view(),
                        ground.tg.view(),
                        ground.tg_wall,
                        weather.ta,
                        emis_grid_v,
                        config.emis_wall,
                        alb_grid_v,
                        SBC,
                        config.albedo_wall,
                        weather.ta, // twater = ta
                        lc_grid_v,
                        lc_grid_v.is_some(),
                    )
                } else {
                    // Full GVF (first timestep or no cache)
                    let wh = wall_ht_v.unwrap();
                    gvf_calc_pure(
                        wallsun.view(),
                        wh,
                        buildings_v,
                        config.pixel_size,
                        shadow_f32.view(),
                        first,
                        second,
                        wall_asp_v.unwrap(),
                        ground.tg.view(),
                        ground.tg_wall,
                        weather.ta,
                        emis_grid_v,
                        config.emis_wall,
                        alb_grid_v,
                        SBC,
                        config.albedo_wall,
                        weather.ta, // twater = ta
                        lc_grid_v,
                        lc_grid_v.is_some(),
                    )
                }
            } else {
                // Simplified GVF (no walls) - compute inline
                let gvf_simple = 1.0 - &svf_v;
                let tg_with_shadow = &ground.tg * &shadow_f32;
                // Lup = emis × SBC × (Ta + Tg_shadow + 273.15)^4
                let lup_simple = {
                    let mut arr = Array2::<f32>::zeros(shape);
                    Zip::indexed(&mut arr).par_for_each(|(r, c), out| {
                        if valid_v[[r, c]] == 0 {
                            *out = f32::NAN;
                            return;
                        }
                        let t = weather.ta + tg_with_shadow[[r, c]] + KELVIN_OFFSET;
                        *out = emis_grid_v[[r, c]] * SBC * t.powi(4);
                    });
                    arr
                };
                let gvfalb_simple = &alb_grid_v * &gvf_simple;

                GvfResultPure {
                    gvf_lup: lup_simple.clone(),
                    gvfalb: gvfalb_simple.clone(),
                    gvfalbnosh: Some(alb_grid_v.to_owned()),
                    gvf_lup_e: lup_simple.clone(),
                    gvfalb_e: gvfalb_simple.clone(),
                    gvfalbnosh_e: Some(alb_grid_v.to_owned()),
                    gvf_lup_s: lup_simple.clone(),
                    gvfalb_s: gvfalb_simple.clone(),
                    gvfalbnosh_s: Some(alb_grid_v.to_owned()),
                    gvf_lup_w: lup_simple.clone(),
                    gvfalb_w: gvfalb_simple.clone(),
                    gvfalbnosh_w: Some(alb_grid_v.to_owned()),
                    gvf_lup_n: lup_simple.clone(),
                    gvfalb_n: gvfalb_simple,
                    gvfalbnosh_n: Some(alb_grid_v.to_owned()),
                    gvf_sum: Some(Array2::zeros(shape)),
                    gvf_norm: Some(Array2::ones(shape)),
                }
            };

            let gvf_dur = t_gvf.elapsed();

            // ── Step 4: Thermal Delay ────────────────────────────────────────────
            let t_delay = Instant::now();
            let tg_temp = &ground.tg * &shadow_f32 + weather.ta;

            let delay = ts_wave_delay_batch_pure(
                gvf.gvf_lup.view(),
                gvf.gvf_lup_e.view(),
                gvf.gvf_lup_s.view(),
                gvf.gvf_lup_w.view(),
                gvf.gvf_lup_n.view(),
                tg_temp.view(),
                firstdaytime,
                timeadd,
                timestep_dec,
                tgmap1_v,
                tgmap1_e_v,
                tgmap1_s_v,
                tgmap1_w_v,
                tgmap1_n_v,
                tgout1_v,
            );

            let delay_dur = t_delay.elapsed();

            // ── Step 5: Radiation ─────────────────────────────────────────────────
            let t_radiation = Instant::now();
            let esky = compute_esky(weather.ta, weather.rh);
            let sin_alt = (weather.sun_altitude * PI / 180.0).sin();
            let rad_i = weather.direct_rad;
            let rad_d = weather.diffuse_rad;
            let rad_g = weather.global_rad;
            let psi = weather.psi;
            let cyl = human.is_standing;

            // F_sh (cylindric wedge shadow fraction) — shared by both paths
            let zen_rad = weather.sun_zenith * PI / 180.0;
            let f_sh = cylindric_wedge_pure_masked(zen_rad, svfalfa_v, Some(valid_v));

            // Kup — shared by both paths.
            // In cached-GVF mode, read gvfalbnosh* directly from geometry cache to
            // avoid per-timestep cloning of those static arrays.
            let compute_kup_with =
                |gvfalbnosh: ArrayView2<f32>,
                 gvfalbnosh_e: ArrayView2<f32>,
                 gvfalbnosh_s: ArrayView2<f32>,
                 gvfalbnosh_w: ArrayView2<f32>,
                 gvfalbnosh_n: ArrayView2<f32>| {
                    compute_kup(
                        rad_i,
                        rad_d,
                        rad_g,
                        weather.sun_altitude,
                        svfbuveg_v,
                        config.albedo_wall,
                        f_sh.view(),
                        gvf.gvfalb.view(),
                        gvf.gvfalb_e.view(),
                        gvf.gvfalb_s.view(),
                        gvf.gvfalb_w.view(),
                        gvf.gvfalb_n.view(),
                        gvfalbnosh,
                        gvfalbnosh_e,
                        gvfalbnosh_s,
                        gvfalbnosh_w,
                        gvfalbnosh_n,
                        valid_v,
                    )
                };
            let (kup, kup_e, kup_s, kup_w, kup_n) = if let Some(cache) = gvf_cache_inner {
                compute_kup_with(
                    cache.cached_albnosh.view(),
                    cache.cached_albnosh_e.view(),
                    cache.cached_albnosh_s.view(),
                    cache.cached_albnosh_w.view(),
                    cache.cached_albnosh_n.view(),
                )
            } else {
                let gvfalbnosh = gvf
                    .gvfalbnosh
                    .as_ref()
                    .expect("gvfalbnosh missing without cache");
                let gvfalbnosh_e = gvf
                    .gvfalbnosh_e
                    .as_ref()
                    .expect("gvfalbnosh_e missing without cache");
                let gvfalbnosh_s = gvf
                    .gvfalbnosh_s
                    .as_ref()
                    .expect("gvfalbnosh_s missing without cache");
                let gvfalbnosh_w = gvf
                    .gvfalbnosh_w
                    .as_ref()
                    .expect("gvfalbnosh_w missing without cache");
                let gvfalbnosh_n = gvf
                    .gvfalbnosh_n
                    .as_ref()
                    .expect("gvfalbnosh_n missing without cache");
                compute_kup_with(
                    gvfalbnosh.view(),
                    gvfalbnosh_e.view(),
                    gvfalbnosh_s.view(),
                    gvfalbnosh_w.view(),
                    gvfalbnosh_n.view(),
                )
            };

            // Branch: anisotropic vs isotropic
            let use_aniso = config.use_anisotropic && shmat_v.is_some();

            let (
                kdown,
                ldown,
                kside_knorth,
                kside_keast,
                kside_ksouth,
                kside_kwest,
                lside_lnorth,
                lside_least,
                lside_lsouth,
                lside_lwest,
                kside_total,
                lside_total,
            ) = if use_aniso {
                // === Anisotropic sky ===
                let shmat_a = shmat_v.unwrap();
                let vegshmat_a = vegshmat_v.unwrap();
                let vbshmat_a = vbshmat_v.unwrap();

                // Perez sky luminance distribution (computed in Rust — no Python round-trip)
                let lv_arr = crate::perez::perez_v3(
                    weather.zen_deg,
                    weather.sun_azimuth,
                    weather.diffuse_rad,
                    weather.direct_rad,
                    weather.jday,
                    weather.patch_option,
                );
                let patch_lut = patch_lut_for_option_cached(weather.patch_option);
                let patch_altitude_arr = ArrayView1::from(patch_lut.altitudes.as_slice());
                let patch_azimuth_arr = ArrayView1::from(patch_lut.azimuths.as_slice());
                let steradians_arr = ArrayView1::from(patch_lut.steradians.as_slice());
                let patch_altitude_sin_arr = ArrayView1::from(patch_lut.altitude_sin.as_slice());

                // ASVF from SVF (arccos(sqrt(clip(svf, 0, 1)))) cached by SVF buffer.
                let asvf_cache = asvf_for_svf_cached(svf_v);
                let asvf_arr = ArrayView2::from_shape(shape, asvf_cache.as_slice())
                    .expect("ASVF cache shape mismatch");

                // Esky anisotropic (Jonsson + CI correction)
                let esky_a = {
                    let ci = weather.clearness_index;
                    if ci < 0.95 {
                        ci * esky + (1.0 - ci)
                    } else {
                        esky
                    }
                };

                // Full anisotropic sky calculation (ldown, kside, lside totals)
                // Try GPU path first; fall back to CPU if unavailable.
                let deg2rad = PI / 180.0;
                #[allow(unused_variables)]
                let (lum_chi, rad_tot) = if weather.sun_altitude > 0.0 {
                    let patch_luminance = lv_arr.column(2);
                    let mut rad_tot = 0.0f32;
                    let n_patches = patch_luminance.len();
                    for i in 0..n_patches {
                        rad_tot +=
                            patch_luminance[i] * steradians_arr[i] * patch_altitude_sin_arr[i];
                    }
                    if rad_tot > 0.0 {
                        (patch_luminance.mapv(|lum| (lum * rad_d) / rad_tot), rad_tot)
                    } else {
                        (Array1::<f32>::zeros(n_patches), 0.0)
                    }
                } else {
                    (Array1::<f32>::zeros(lv_arr.shape()[0]), 0.0)
                };

                #[allow(unused_variables)]
                let (_, esky_band) = crate::emissivity_models::model2(&lv_arr, esky_a, weather.ta);

                let mut lside_dirs = lside_dirs_aniso_from_lup(
                    delay.lup_e.view(),
                    delay.lup_s.view(),
                    delay.lup_w.view(),
                    delay.lup_n.view(),
                    valid_v,
                );

                #[cfg(feature = "gpu")]
                let gpu_result = if cyl {
                    if let Some(ctx) = get_aniso_gpu_context() {
                        if aniso_gpu_overlap_enabled() {
                            // Overlap mode: start dispatch and complete explicitly.
                            let gpu_result = ctx
                                .dispatch_begin(
                                    shmat_a,
                                    vegshmat_a,
                                    vbshmat_a,
                                    asvf_arr,
                                    delay.lup.view(),
                                    delay.lup_e.view(),
                                    delay.lup_s.view(),
                                    delay.lup_w.view(),
                                    delay.lup_n.view(),
                                    valid_v,
                                    patch_altitude_arr,
                                    patch_azimuth_arr,
                                    steradians_arr,
                                    esky_band.view(),
                                    lum_chi.view(),
                                    weather.sun_altitude,
                                    weather.sun_azimuth,
                                    weather.ta,
                                    cyl,
                                    config.albedo_wall,
                                    ground.tg_wall,
                                    config.emis_wall,
                                    rad_i,
                                    rad_d,
                                    psi,
                                    rad_tot,
                                )
                                .ok();
                            if let Some(pending) = gpu_result {
                                ctx.dispatch_end(pending).ok()
                            } else {
                                None
                            }
                        } else {
                            ctx.dispatch(
                                shmat_a,
                                vegshmat_a,
                                vbshmat_a,
                                asvf_arr,
                                delay.lup.view(),
                                delay.lup_e.view(),
                                delay.lup_s.view(),
                                delay.lup_w.view(),
                                delay.lup_n.view(),
                                valid_v,
                                patch_altitude_arr,
                                patch_azimuth_arr,
                                steradians_arr,
                                esky_band.view(),
                                lum_chi.view(),
                                weather.sun_altitude,
                                weather.sun_azimuth,
                                weather.ta,
                                cyl,
                                config.albedo_wall,
                                ground.tg_wall,
                                config.emis_wall,
                                rad_i,
                                rad_d,
                                psi,
                                rad_tot,
                            )
                            .ok()
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Compute anisotropic sky: GPU path + CPU fallback
                let mut used_gpu = false;
                #[allow(unused_mut)]
                let mut ani_ldown = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_lside = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_kside = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_keast = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_ksouth = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_kwest = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut ani_knorth = Array2::<f32>::zeros(shape);
                #[allow(unused_mut)]
                let mut drad = Array2::<f32>::zeros(shape);

                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu_result {
                    // GPU path: derive kside and k-directional from GPU partial outputs
                    let kside_i = if cyl {
                        &shadow_f32 * rad_i * (weather.sun_altitude * deg2rad).cos()
                    } else {
                        Array2::<f32>::zeros(shape)
                    };
                    if weather.sun_altitude > 0.0 {
                        ani_kside = kside_i + &gpu.kside_partial;
                        ani_keast = &kup_e * 0.5;
                        ani_ksouth = &kup_s * 0.5;
                        ani_kwest = &kup_w * 0.5;
                        ani_knorth = &kup_n * 0.5;
                    }
                    ani_ldown = gpu.ldown;
                    ani_lside = gpu.lside;
                    drad = gpu.drad;
                    lside_dirs = crate::vegetation::LsideVegPureResult {
                        least: gpu.lside_least,
                        lsouth: gpu.lside_lsouth,
                        lwest: gpu.lside_lwest,
                        lnorth: gpu.lside_lnorth,
                    };
                    used_gpu = true;
                }

                if !used_gpu {
                    // drad via direct accumulation from packed shadow matrices.
                    // Shadow matrices are bitpacked: 1 bit per patch, 8 patches per byte.
                    // ani_lum = sum_i((sh_i - (1 - veg_i) * (1 - psi)) * lv_i)
                    let lv_col2 = lv_arr.column(2);
                    let ani_lum =
                        compute_ani_lum_from_packed(shmat_a, vegshmat_a, lv_col2, psi, valid_v);
                    drad = ani_lum.mapv(|v| v * rad_d);

                    let ani = anisotropic_sky_pure(
                        shmat_a,
                        vegshmat_a,
                        vbshmat_a,
                        weather.sun_altitude,
                        weather.sun_azimuth,
                        esky_a,
                        weather.ta,
                        cyl,
                        false, // wall_scheme
                        config.albedo_wall,
                        ground.tg_wall,
                        config.emis_wall,
                        rad_i,
                        rad_d,
                        asvf_arr,
                        lv_arr.view(),
                        steradians_arr,
                        delay.lup.view(),
                        lv_arr.view(),
                        shadow_f32.view(),
                        kup_e.view(),
                        kup_s.view(),
                        kup_w.view(),
                        kup_n.view(),
                        None, // voxel_table
                        None, // voxel_maps
                        Some(valid_v),
                    );
                    ani_ldown = ani.ldown;
                    ani_lside = ani.lside;
                    ani_kside = ani.kside;
                    ani_keast = ani.keast;
                    ani_ksouth = ani.ksouth;
                    ani_kwest = ani.kwest;
                    ani_knorth = ani.knorth;
                }

                // Kdown (shared formula, but with anisotropic drad)
                let kdown = compute_kdown(
                    rad_i,
                    rad_d,
                    rad_g,
                    shadow_f32.view(),
                    sin_alt,
                    svfbuveg_v,
                    config.albedo_wall,
                    f_sh.view(),
                    drad.view(),
                    valid_v,
                );

                // From anisotropic: ldown from ani_sky, lside from lside_veg, kside from ani_sky
                (
                    kdown,
                    ani_ldown,
                    ani_knorth,
                    ani_keast,
                    ani_ksouth,
                    ani_kwest,
                    lside_dirs.lnorth,
                    lside_dirs.least,
                    lside_dirs.lsouth,
                    lside_dirs.lwest,
                    ani_kside,
                    ani_lside,
                )
            } else {
                // === Isotropic sky ===

                // drad (isotropic diffuse)
                let drad = svfbuveg_v.mapv(|sv| rad_d * sv);

                // Ldown
                let ldown = compute_ldown(
                    esky,
                    weather.ta,
                    ground.tg_wall,
                    svf_v,
                    svf_veg_v,
                    svf_aveg_v,
                    config.emis_wall,
                    weather.clearness_index,
                    valid_v,
                );

                // kside_veg (isotropic)
                let kside = kside_veg_isotropic_pure(
                    rad_i,
                    rad_d,
                    rad_g,
                    shadow_f32.view(),
                    svf_s_v,
                    svf_w_v,
                    svf_n_v,
                    svf_e_v,
                    svf_veg_e_v,
                    svf_veg_s_v,
                    svf_veg_w_v,
                    svf_veg_n_v,
                    weather.sun_azimuth,
                    weather.sun_altitude,
                    psi,
                    0.0, // t (instrument offset)
                    config.albedo_wall,
                    f_sh.view(),
                    kup_e.view(),
                    kup_s.view(),
                    kup_w.view(),
                    kup_n.view(),
                    cyl,
                    Some(valid_v),
                );

                // lside_veg (isotropic)
                let lside = lside_veg_pure(
                    svf_s_v,
                    svf_w_v,
                    svf_n_v,
                    svf_e_v,
                    svf_veg_e_v,
                    svf_veg_s_v,
                    svf_veg_w_v,
                    svf_veg_n_v,
                    svf_aveg_e_v,
                    svf_aveg_s_v,
                    svf_aveg_w_v,
                    svf_aveg_n_v,
                    weather.sun_azimuth,
                    weather.sun_altitude,
                    weather.ta,
                    ground.tg_wall,
                    SBC,
                    config.emis_wall,
                    ldown.view(),
                    esky,
                    0.0, // t
                    f_sh.view(),
                    weather.clearness_index,
                    delay.lup_e.view(),
                    delay.lup_s.view(),
                    delay.lup_w.view(),
                    delay.lup_n.view(),
                    false, // isotropic
                    Some(valid_v),
                );

                // Kdown
                let kdown = compute_kdown(
                    rad_i,
                    rad_d,
                    rad_g,
                    shadow_f32.view(),
                    sin_alt,
                    svfbuveg_v,
                    config.albedo_wall,
                    f_sh.view(),
                    drad.view(),
                    valid_v,
                );

                // Isotropic: kside_total = kside_i, lside_total = zeros
                (
                    kdown,
                    ldown,
                    kside.knorth,
                    kside.keast,
                    kside.ksouth,
                    kside.kwest,
                    lside.lnorth,
                    lside.least,
                    lside.lsouth,
                    lside.lwest,
                    kside.kside_i,
                    Array2::<f32>::zeros(shape),
                )
            };

            let radiation_dur = t_radiation.elapsed();

            // ── Step 6: Tmrt ─────────────────────────────────────────────────────
            let t_tmrt = Instant::now();
            let tmrt = compute_tmrt_pure(
                kdown.view(),
                kup.view(),
                ldown.view(),
                delay.lup.view(),
                kside_knorth.view(),
                kside_keast.view(),
                kside_ksouth.view(),
                kside_kwest.view(),
                lside_lnorth.view(),
                lside_least.view(),
                lside_lsouth.view(),
                lside_lwest.view(),
                kside_total.view(),
                lside_total.view(),
                human.abs_k,
                human.abs_l,
                human.is_standing,
                use_aniso,
            );
            let tmrt_dur = t_tmrt.elapsed();

            if timing_enabled() {
                let total =
                    shadow_dur + ground_dur + gvf_dur + delay_dur + radiation_dur + tmrt_dur;
                let total_ms = total.as_secs_f64() * 1000.0;
                let shadow_ms = shadow_dur.as_secs_f64() * 1000.0;
                let ground_ms = ground_dur.as_secs_f64() * 1000.0;
                let gvf_ms = gvf_dur.as_secs_f64() * 1000.0;
                let delay_ms = delay_dur.as_secs_f64() * 1000.0;
                let rad_ms = radiation_dur.as_secs_f64() * 1000.0;
                let tmrt_ms = tmrt_dur.as_secs_f64() * 1000.0;
                // GPU duty cycle: shadow always uses GPU (when available);
                // radiation includes GPU aniso dispatch when anisotropic is active.
                let gpu_ms = shadow_ms + if use_aniso { rad_ms } else { 0.0 };
                let duty = if total_ms > 0.0 {
                    gpu_ms / total_ms * 100.0
                } else {
                    0.0
                };
                eprintln!(
                    "[TIMING] shadow={:.1}ms ground={:.1}ms gvf={:.1}ms delay={:.1}ms \
             radiation={:.1}ms tmrt={:.1}ms | total={:.1}ms gpu_duty={:.0}%",
                    shadow_ms, ground_ms, gvf_ms, delay_ms, rad_ms, tmrt_ms, total_ms, duty,
                );
            }

            TimestepResultRaw {
                tmrt,
                shadow: shadow_f32,
                kdown,
                kup,
                ldown,
                lup: delay.lup,
                timeadd: delay.timeadd,
                tgmap1: delay.tgmap1,
                tgmap1_e: delay.tgmap1_e,
                tgmap1_s: delay.tgmap1_s,
                tgmap1_w: delay.tgmap1_w,
                tgmap1_n: delay.tgmap1_n,
                tgout1: delay.tgout1,
            }
        })
    }; // end allow_threads_unchecked

    // ── Convert final outputs to PyArrays (needs GIL) ────────────────────
    Ok(TimestepResult {
        tmrt: raw.tmrt.into_pyarray(py).unbind(),
        shadow: if output_mask_bits & OUT_SHADOW != 0 {
            Some(raw.shadow.into_pyarray(py).unbind())
        } else {
            None
        },
        kdown: if output_mask_bits & OUT_KDOWN != 0 {
            Some(raw.kdown.into_pyarray(py).unbind())
        } else {
            None
        },
        kup: if output_mask_bits & OUT_KUP != 0 {
            Some(raw.kup.into_pyarray(py).unbind())
        } else {
            None
        },
        ldown: if output_mask_bits & OUT_LDOWN != 0 {
            Some(raw.ldown.into_pyarray(py).unbind())
        } else {
            None
        },
        lup: if output_mask_bits & OUT_LUP != 0 {
            Some(raw.lup.into_pyarray(py).unbind())
        } else {
            None
        },
        timeadd: raw.timeadd,
        tgmap1: raw.tgmap1.into_pyarray(py).unbind(),
        tgmap1_e: raw.tgmap1_e.into_pyarray(py).unbind(),
        tgmap1_s: raw.tgmap1_s.into_pyarray(py).unbind(),
        tgmap1_w: raw.tgmap1_w.into_pyarray(py).unbind(),
        tgmap1_n: raw.tgmap1_n.into_pyarray(py).unbind(),
        tgout1: raw.tgout1.into_pyarray(py).unbind(),
    })
}
