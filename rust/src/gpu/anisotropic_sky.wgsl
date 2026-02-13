// Anisotropic sky radiation — GPU compute shader.
//
// Fuses the per-pixel, per-patch loop from anisotropic_sky_pure() (sky.rs)
// onto the GPU. Each thread handles one pixel and iterates over all sky
// patches, accumulating longwave and shortwave radiation.
//
// Two passes:
//   Pass 1 — sky, vegetation, and building longwave + diffuse shortwave
//   Pass 2 — reflected longwave (depends on ldown_sky from pass 1)
//
// Outputs 4 accumulated arrays consumed by pipeline.rs:
//   out_ldown        — total downwelling longwave  (sky+veg+bldg+ref)
//   out_lside        — total side longwave          (sky+veg+bldg+ref)
//   out_kside_partial — diffuse shortwave + reflected shortwave (kside_d+kref)
//   out_drad         — anisotropic diffuse shortwave for Kdown
//
// The pipeline adds kside_i = shadow * rad_i * cos(alt) on the CPU.

const PI: f32  = 3.14159265358979323846;
const SBC: f32 = 5.67051e-8;  // Stefan-Boltzmann constant  (W m⁻² K⁻⁴)
const DEG2RAD: f32 = 0.017453292519943295;  // PI / 180
const RAD2DEG: f32 = 57.29577951308232;     // 180 / PI
const NAN_BITS: u32 = 0x7FC00000u;          // quiet NaN in IEEE 754

// ── Uniform parameters (one per dispatch) ────────────────────────────────

struct Params {
    total_pixels: u32,
    cols: u32,
    rows: u32,
    n_patches:    u32,
    n_pack:       u32,        // ceil(n_patches / 8)
    cyl:          u32,        // 1 = standing (cylindric), 0 = lying
    solar_altitude: f32,      // degrees
    solar_azimuth:  f32,      // degrees
    ta:           f32,        // air temperature  (°C)
    albedo:       f32,        // ground albedo
    tgwall:       f32,        // wall temperature excess (°C)
    ewall:        f32,        // wall emissivity
    rad_i:        f32,        // direct radiation  (W m⁻²)
    rad_d:        f32,        // diffuse radiation (W m⁻²)
    psi:          f32,        // vegetation transmissivity factor for diffsh
    rad_tot:      f32,        // ∑(lv * steradians * sin(alt)) for drad recovery
};

@group(0) @binding(0)  var<uniform> params: Params;

// ── Per-pixel inputs (flattened row-major) ───────────────────────────────

// Bitpacked shadow matrices — 1 bit per patch, 8 per byte, read as u32.
// Layout: pixel_idx * n_pack_u32 + word, where n_pack_u32 = ceil(n_pack/4).
// Byte within word: little-endian (byte 0 = bits 0..7 of u32).
@group(0) @binding(1)  var<storage, read> shmat:          array<u32>;
@group(0) @binding(2)  var<storage, read> vegshmat:       array<u32>;
@group(0) @binding(3)  var<storage, read> vbshvegshmat:   array<u32>;

@group(0) @binding(4)  var<storage, read> asvf:           array<f32>;
@group(0) @binding(5)  var<storage, read> lup:            array<f32>;
// Valid mask — packed u8 as u32 (4 pixels per word). 0 = invalid.
@group(0) @binding(6)  var<storage, read> valid_mask:     array<u32>;

// ── Per-patch look-up tables (length = n_patches) ────────────────────────

@group(0) @binding(7)  var<storage, read> patch_alt:      array<f32>;
@group(0) @binding(8)  var<storage, read> patch_azi:      array<f32>;
@group(0) @binding(9)  var<storage, read> steradians_buf: array<f32>;
@group(0) @binding(10) var<storage, read> esky_band_buf:  array<f32>;
@group(0) @binding(11) var<storage, read> lum_chi_buf:    array<f32>;

// ── Outputs (per-pixel, flattened row-major) ─────────────────────────────

@group(0) @binding(12) var<storage, read_write> out_ldown:         array<f32>;
@group(0) @binding(13) var<storage, read_write> out_lside:         array<f32>;
@group(0) @binding(14) var<storage, read_write> out_kside_partial: array<f32>;
@group(0) @binding(15) var<storage, read_write> out_drad:          array<f32>;

// ── Bit extraction ───────────────────────────────────────────────────────
//
// Shadow matrices are stored as u8 arrays bitpacked with 1 bit per patch.
// On the GPU they are uploaded as &[u8] reinterpreted as array<u32>.
// Byte order is little-endian (WGSL storage buffer default).

fn sh_bit(pixel: u32, p: u32) -> bool {
    let byte_idx: u32 = pixel * params.n_pack + (p >> 3u);
    let word_idx: u32 = byte_idx >> 2u;
    let byte_in_word: u32 = byte_idx & 3u;
    let byte_val: u32 = (shmat[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    return ((byte_val >> (p & 7u)) & 1u) == 1u;
}

fn veg_bit(pixel: u32, p: u32) -> bool {
    let byte_idx: u32 = pixel * params.n_pack + (p >> 3u);
    let word_idx: u32 = byte_idx >> 2u;
    let byte_in_word: u32 = byte_idx & 3u;
    let byte_val: u32 = (vegshmat[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    return ((byte_val >> (p & 7u)) & 1u) == 1u;
}

fn vb_bit(pixel: u32, p: u32) -> bool {
    let byte_idx: u32 = pixel * params.n_pack + (p >> 3u);
    let word_idx: u32 = byte_idx >> 2u;
    let byte_in_word: u32 = byte_idx & 3u;
    let byte_val: u32 = (vbshvegshmat[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
    return ((byte_val >> (p & 7u)) & 1u) == 1u;
}

// ── Sunlit / shaded classification ───────────────────────────────────────
// Matches sunlit_shaded_patches::shaded_or_sunlit_pixel in Rust.

fn compute_sunlit_degrees(p_alt: f32, p_azi: f32, pixel_asvf: f32) -> f32 {
    let patch_to_sun_azi: f32 = abs(params.solar_azimuth - p_azi);
    let xi: f32 = cos(patch_to_sun_azi * DEG2RAD);
    let yi: f32 = 2.0 * xi * tan(params.solar_altitude * DEG2RAD);
    let hsvf: f32 = tan(pixel_asvf);
    let yi_: f32 = select(yi, 0.0, yi > 0.0);
    let tan_delta: f32 = hsvf + yi_;
    return atan(tan_delta) * RAD2DEG;
}

// ── Main kernel ──────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    if (x >= params.cols || y >= params.rows) {
        return;
    }
    let idx: u32 = y * params.cols + x;
    if (idx >= params.total_pixels) {
        return;
    }

    // ── Valid mask check ─────────────────────────────────────────────────
    let valid_word: u32 = valid_mask[idx >> 2u];
    let valid_byte: u32 = (valid_word >> ((idx & 3u) * 8u)) & 0xFFu;
    if (valid_byte == 0u) {
        out_ldown[idx]         = bitcast<f32>(NAN_BITS);
        out_lside[idx]         = bitcast<f32>(NAN_BITS);
        out_kside_partial[idx] = bitcast<f32>(NAN_BITS);
        out_drad[idx]          = bitcast<f32>(NAN_BITS);
        return;
    }

    // ── Short-circuit when not cylindric (cyl == 0) ─────────────────────
    if (params.cyl == 0u) {
        out_ldown[idx]         = 0.0;
        out_lside[idx]         = 0.0;
        out_kside_partial[idx] = 0.0;
        out_drad[idx]          = 0.0;
        return;
    }

    // ── Pre-compute scalars ─────────────────────────────────────────────
    let ta_k: f32     = params.ta + 273.15;
    let pixel_asvf: f32 = asvf[idx];
    let pixel_lup: f32  = lup[idx];
    let sun_above: bool = params.solar_altitude > 0.0;

    // Surfaces for building longwave
    let sunlit_surface: f32 = (params.ewall * SBC * pow(ta_k + params.tgwall, 4.0)) / PI;
    let shaded_surface: f32 = (params.ewall * SBC * pow(ta_k, 4.0)) / PI;
    // Vegetation surface (same emissivity model as shaded buildings)
    let veg_surface: f32 = shaded_surface;

    // Reflected shortwave surfaces (only when sun is up)
    // CPU formula: (albedo * rad_i * cos(alt) + rad_d * 0.5) / PI
    // Note: albedo only multiplies the direct component, not the diffuse half
    let k_sunlit_surface: f32 = select(
        0.0,
        (params.albedo * params.rad_i * cos(params.solar_altitude * DEG2RAD) + params.rad_d * 0.5) / PI,
        sun_above
    );
    let k_shaded_surface: f32 = select(
        0.0,
        (params.albedo * params.rad_d * 0.5) / PI,
        sun_above
    );

    // ── Accumulators ────────────────────────────────────────────────────
    var ldown_sky: f32 = 0.0;
    var lside_sky: f32 = 0.0;
    var ldown_veg: f32 = 0.0;
    var lside_veg: f32 = 0.0;
    var ldown_sun: f32 = 0.0;
    var lside_sun: f32 = 0.0;
    var ldown_sh: f32 = 0.0;
    var lside_sh: f32 = 0.0;
    var kside_d_acc: f32 = 0.0;
    var kref_acc: f32 = 0.0;   // combined kref_sun + kref_sh + kref_veg
    var drad_norm_acc: f32 = 0.0; // Σ(diffsh * lum_chi); multiply by rad_tot at end

    // ── Pass 1: Main patch loop ─────────────────────────────────────────
    for (var i: u32 = 0u; i < params.n_patches; i++) {
        let sh: bool  = sh_bit(idx, i);
        let vsh: bool = veg_bit(idx, i);
        let vbsh: bool = vb_bit(idx, i);
        let sh_f: f32 = select(0.0, 1.0, sh);
        let vsh_f: f32 = select(0.0, 1.0, vsh);

        // Classification (matches sky.rs)
        let temp_sky: bool   = sh && vsh;
        let temp_vegsh: bool = !vsh || !vbsh;
        let temp_sh: bool    = !sh && vbsh;

        let p_alt: f32    = patch_alt[i];
        let p_azi: f32    = patch_azi[i];
        let steradian: f32 = steradians_buf[i];
        let alt_rad: f32  = p_alt * DEG2RAD;
        let aoi: f32      = cos(alt_rad);   // angle of incidence (vertical surface)
        let aoi_h: f32    = sin(alt_rad);   // horizontal surface projection

        // Diffuse shadow term used by Kdown:
        // diffsh = sh - (1 - vegsh) * (1 - psi)
        let diffsh: f32 = sh_f - (1.0 - vsh_f) * (1.0 - params.psi);
        if (sun_above && params.rad_tot > 0.0) {
            drad_norm_acc += diffsh * lum_chi_buf[i];
        }

        // ── Sky longwave ────────────────────────────────────────────
        if (temp_sky) {
            let esky_i: f32 = esky_band_buf[i];
            let lval: f32 = (esky_i * SBC * pow(ta_k, 4.0)) / PI;
            lside_sky += lval * steradian * aoi;
            ldown_sky += lval * steradian * aoi_h;

            // Diffuse shortwave
            if (sun_above) {
                kside_d_acc += lum_chi_buf[i] * aoi * steradian;
            }
        }

        // ── Vegetation longwave ─────────────────────────────────────
        if (temp_vegsh) {
            lside_veg += veg_surface * steradian * aoi;
            ldown_veg += veg_surface * steradian * aoi_h;

            // Vegetation reflected shortwave
            if (sun_above) {
                kref_acc += k_shaded_surface * steradian * aoi;
            }
        }

        // ── Building longwave ───────────────────────────────────────
        if (temp_sh) {
            let sunlit_deg: f32 = compute_sunlit_degrees(p_alt, p_azi, pixel_asvf);
            let is_sunlit: bool = sunlit_deg < p_alt;
            let is_shaded: bool = sunlit_deg > p_alt;

            let azimuth_difference: f32 = abs(params.solar_azimuth - p_azi);
            let facing_sun: bool = azimuth_difference > 90.0
                                && azimuth_difference < 270.0
                                && sun_above;

            if (facing_sun) {
                let sf: f32 = select(0.0, 1.0, is_sunlit);
                let shf: f32 = select(0.0, 1.0, is_shaded);
                lside_sun += sf * sunlit_surface * steradian * aoi;
                lside_sh  += shf * shaded_surface * steradian * aoi;
                ldown_sun += sf * sunlit_surface * steradian * aoi_h;
                ldown_sh  += shf * shaded_surface * steradian * aoi_h;
            } else {
                // Not facing sun → all shaded
                lside_sh += shaded_surface * steradian * aoi;
                ldown_sh += shaded_surface * steradian * aoi_h;
            }

            // Reflected shortwave from buildings
            if (sun_above) {
                if (is_sunlit) {
                    kref_acc += k_sunlit_surface * steradian * aoi;
                }
                if (is_shaded) {
                    kref_acc += k_shaded_surface * steradian * aoi;
                }
            }
        }
    }

    // ── Pass 2: Reflected longwave ──────────────────────────────────────
    // Reflected radiation depends on ldown_sky from pass 1.
    var lside_ref: f32 = 0.0;
    var ldown_ref: f32 = 0.0;

    let reflected_base: f32 = (ldown_sky + pixel_lup) * (1.0 - params.ewall) * 0.5 / PI;

    for (var i: u32 = 0u; i < params.n_patches; i++) {
        let sh: bool  = sh_bit(idx, i);
        let vsh: bool = veg_bit(idx, i);
        let vbsh: bool = vb_bit(idx, i);

        // Any obstruction → reflected longwave
        let is_obstructed: bool = !sh || !vsh || !vbsh;

        if (is_obstructed) {
            let alt_rad: f32   = patch_alt[i] * DEG2RAD;
            let aoi: f32       = cos(alt_rad);
            let aoi_h: f32     = sin(alt_rad);
            let steradian: f32 = steradians_buf[i];

            lside_ref += reflected_base * steradian * aoi;
            ldown_ref += reflected_base * steradian * aoi_h;
        }
    }

    // ── Write combined outputs ──────────────────────────────────────────
    out_ldown[idx] = ldown_sky + ldown_veg + ldown_sh + ldown_sun + ldown_ref;
    out_lside[idx] = lside_sky + lside_veg + lside_sh + lside_sun + lside_ref;
    out_kside_partial[idx] = kside_d_acc + kref_acc;
    out_drad[idx] = drad_norm_acc * params.rad_tot;
}
