// Directional anisotropic lside components.
//
// Computes:
//   lside_least  = lup_e * 0.5
//   lside_lsouth = lup_s * 0.5
//   lside_lwest  = lup_w * 0.5
//   lside_lnorth = lup_n * 0.5
//
// Invalid pixels receive NaN to match CPU behavior.

const NAN_BITS: u32 = 0x7FC00000u; // quiet NaN in IEEE 754

struct Params {
    total_pixels: u32,
    n_patches: u32,
    n_pack: u32,
    cyl: u32,
    solar_altitude: f32,
    solar_azimuth: f32,
    ta: f32,
    albedo: f32,
    tgwall: f32,
    ewall: f32,
    rad_i: f32,
    rad_d: f32,
    psi: f32,
    rad_tot: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> valid_mask:    array<u32>;
@group(0) @binding(2) var<storage, read> lup_e:         array<f32>;
@group(0) @binding(3) var<storage, read> lup_s:         array<f32>;
@group(0) @binding(4) var<storage, read> lup_w:         array<f32>;
@group(0) @binding(5) var<storage, read> lup_n:         array<f32>;
@group(0) @binding(6) var<storage, read_write> out_least:  array<f32>;
@group(0) @binding(7) var<storage, read_write> out_lsouth: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_lwest:  array<f32>;
@group(0) @binding(9) var<storage, read_write> out_lnorth: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;
    if (idx >= params.total_pixels) {
        return;
    }

    let valid_word: u32 = valid_mask[idx >> 2u];
    let valid_byte: u32 = (valid_word >> ((idx & 3u) * 8u)) & 0xFFu;
    if (valid_byte == 0u) {
        out_least[idx] = bitcast<f32>(NAN_BITS);
        out_lsouth[idx] = bitcast<f32>(NAN_BITS);
        out_lwest[idx] = bitcast<f32>(NAN_BITS);
        out_lnorth[idx] = bitcast<f32>(NAN_BITS);
        return;
    }

    out_least[idx] = lup_e[idx] * 0.5;
    out_lsouth[idx] = lup_s[idx] * 0.5;
    out_lwest[idx] = lup_w[idx] * 0.5;
    out_lnorth[idx] = lup_n[idx] * 0.5;
}
