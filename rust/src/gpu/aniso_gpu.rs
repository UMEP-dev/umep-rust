//! GPU-accelerated anisotropic sky radiation computation.
//!
//! Fuses the per-pixel × per-patch loop from `anisotropic_sky_pure()` onto a
//! single GPU compute dispatch. Each thread handles one pixel and iterates
//! over all sky patches, accumulating longwave and shortwave radiation.
//!
//! Shares `Arc<wgpu::Device>` and `Arc<wgpu::Queue>` with `ShadowGpuContext`.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};
use std::sync::{Arc, Mutex};

// ── Uniform buffer (must match Params in anisotropic_sky.wgsl) ───────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AnisoParams {
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
}

// ── Cached GPU buffers ───────────────────────────────────────────────────

struct CachedBuffers {
    rows: usize,
    cols: usize,
    n_pack: usize,
    n_patches: usize,
    // Uniform
    params_buffer: wgpu::Buffer,
    // Per-pixel inputs
    shmat_buffer: wgpu::Buffer,
    vegshmat_buffer: wgpu::Buffer,
    vbshvegshmat_buffer: wgpu::Buffer,
    asvf_buffer: wgpu::Buffer,
    lup_buffer: wgpu::Buffer,
    valid_buffer: wgpu::Buffer,
    // Per-patch LUTs
    patch_alt_buffer: wgpu::Buffer,
    patch_azi_buffer: wgpu::Buffer,
    steradians_buffer: wgpu::Buffer,
    esky_band_buffer: wgpu::Buffer,
    lum_chi_buffer: wgpu::Buffer,
    // Outputs
    out_ldown_buffer: wgpu::Buffer,
    out_lside_buffer: wgpu::Buffer,
    out_kside_partial_buffer: wgpu::Buffer,
    // Staging for readback
    staging_buffer: wgpu::Buffer,
    // Bind group
    bind_group: wgpu::BindGroup,
}

// ── Public context ───────────────────────────────────────────────────────

pub struct AnisoGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    cached: Mutex<Option<CachedBuffers>>,
}

/// Result of GPU anisotropic sky computation.
pub struct AnisoGpuResult {
    pub ldown: Array2<f32>,
    pub lside: Array2<f32>,
    pub kside_partial: Array2<f32>,
}

impl AnisoGpuContext {
    /// Create a new context, sharing device/queue from the shadow GPU context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Anisotropic Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("anisotropic_sky.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Aniso Bind Group Layout"),
                entries: &Self::bind_group_layout_entries(),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Aniso Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Aniso Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            cached: Mutex::new(None),
        }
    }

    /// Run the anisotropic sky kernel.
    ///
    /// Pre-conditions (computed by the caller, i.e. pipeline.rs):
    ///   - `esky_band`: per-patch emissivity from `emissivity_models::model2`
    ///   - `lum_chi`: per-patch normalised luminance × rad_d / rad_tot
    ///
    /// Returns (ldown, lside, kside_partial) where
    ///   kside_partial = kside_d + kref_sun + kref_sh + kref_veg
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        // Shadow matrices (bitpacked u8, shape rows×cols×n_pack)
        shmat: ArrayView3<u8>,
        vegshmat: ArrayView3<u8>,
        vbshvegshmat: ArrayView3<u8>,
        // Per-pixel arrays (shape rows×cols)
        asvf: ArrayView2<f32>,
        lup: ArrayView2<f32>,
        valid: ArrayView2<u8>,
        // Per-patch LUTs (length n_patches)
        patch_alt: ArrayView1<f32>,
        patch_azi: ArrayView1<f32>,
        steradians: ArrayView1<f32>,
        esky_band: ArrayView1<f32>,
        lum_chi: ArrayView1<f32>,
        // Scalar parameters
        solar_altitude: f32,
        solar_azimuth: f32,
        ta: f32,
        cyl: bool,
        albedo: f32,
        tgwall: f32,
        ewall: f32,
        rad_i: f32,
        rad_d: f32,
    ) -> Result<AnisoGpuResult, String> {
        let rows = shmat.shape()[0];
        let cols = shmat.shape()[1];
        let n_pack = shmat.shape()[2];
        let n_patches = patch_alt.len();
        let total_pixels = rows * cols;

        // Ensure buffers are allocated for this grid size
        self.ensure_buffers(rows, cols, n_pack, n_patches)?;

        let cache = self.cached.lock().unwrap();
        let buffers = cache.as_ref().unwrap();

        // ── Upload uniform params ────────────────────────────────────────
        let params = AnisoParams {
            total_pixels: total_pixels as u32,
            n_patches: n_patches as u32,
            n_pack: n_pack as u32,
            cyl: if cyl { 1 } else { 0 },
            solar_altitude,
            solar_azimuth,
            ta,
            albedo,
            tgwall,
            ewall,
            rad_i,
            rad_d,
        };
        self.queue.write_buffer(
            &buffers.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // ── Upload shadow matrices ───────────────────────────────────────
        // Contiguous u8 data → reinterpreted as u32 on GPU (little-endian)
        let shmat_bytes = Self::contiguous_bytes_3d(&shmat);
        self.queue
            .write_buffer(&buffers.shmat_buffer, 0, &shmat_bytes);

        let vegshmat_bytes = Self::contiguous_bytes_3d(&vegshmat);
        self.queue
            .write_buffer(&buffers.vegshmat_buffer, 0, &vegshmat_bytes);

        let vbshmat_bytes = Self::contiguous_bytes_3d(&vbshvegshmat);
        self.queue
            .write_buffer(&buffers.vbshvegshmat_buffer, 0, &vbshmat_bytes);

        // ── Upload per-pixel arrays ──────────────────────────────────────
        let asvf_bytes = Self::contiguous_f32_2d(&asvf);
        self.queue
            .write_buffer(&buffers.asvf_buffer, 0, bytemuck::cast_slice(&asvf_bytes));

        let lup_bytes = Self::contiguous_f32_2d(&lup);
        self.queue
            .write_buffer(&buffers.lup_buffer, 0, bytemuck::cast_slice(&lup_bytes));

        // Valid mask: u8 → pad to u32 alignment
        let valid_bytes = Self::contiguous_bytes_2d_u8(&valid);
        self.queue
            .write_buffer(&buffers.valid_buffer, 0, &valid_bytes);

        // ── Upload per-patch LUTs ────────────────────────────────────────
        let patch_alt_c = Self::contiguous_f32_1d(&patch_alt);
        self.queue.write_buffer(
            &buffers.patch_alt_buffer,
            0,
            bytemuck::cast_slice(&patch_alt_c),
        );

        let patch_azi_c = Self::contiguous_f32_1d(&patch_azi);
        self.queue.write_buffer(
            &buffers.patch_azi_buffer,
            0,
            bytemuck::cast_slice(&patch_azi_c),
        );

        let ster_c = Self::contiguous_f32_1d(&steradians);
        self.queue.write_buffer(
            &buffers.steradians_buffer,
            0,
            bytemuck::cast_slice(&ster_c),
        );

        let esky_c = Self::contiguous_f32_1d(&esky_band);
        self.queue.write_buffer(
            &buffers.esky_band_buffer,
            0,
            bytemuck::cast_slice(&esky_c),
        );

        let lum_c = Self::contiguous_f32_1d(&lum_chi);
        self.queue.write_buffer(
            &buffers.lum_chi_buffer,
            0,
            bytemuck::cast_slice(&lum_c),
        );

        // ── Dispatch compute shader ──────────────────────────────────────
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Aniso Sky Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Aniso Sky Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            let workgroups = (total_pixels as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // ── Copy outputs to staging buffer ───────────────────────────────
        let pixel_bytes = (total_pixels * 4) as u64; // f32 per pixel
        encoder.copy_buffer_to_buffer(
            &buffers.out_ldown_buffer,
            0,
            &buffers.staging_buffer,
            0,
            pixel_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.out_lside_buffer,
            0,
            &buffers.staging_buffer,
            pixel_bytes,
            pixel_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.out_kside_partial_buffer,
            0,
            &buffers.staging_buffer,
            pixel_bytes * 2,
            pixel_bytes,
        );

        self.queue.submit(Some(encoder.finish()));

        // ── Read back results ────────────────────────────────────────────
        let staging_size = pixel_bytes * 3;
        let buffer_slice = buffers.staging_buffer.slice(..staging_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("GPU poll failed: {:?}", e))?;

        receiver
            .recv()
            .map_err(|e| format!("Channel recv failed: {}", e))?
            .map_err(|e| format!("Failed to map staging buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let all_f32: &[f32] = bytemuck::cast_slice(&data);

        let ldown = Array2::from_shape_vec(
            (rows, cols),
            all_f32[..total_pixels].to_vec(),
        )
        .map_err(|e| format!("ldown array: {}", e))?;

        let lside = Array2::from_shape_vec(
            (rows, cols),
            all_f32[total_pixels..total_pixels * 2].to_vec(),
        )
        .map_err(|e| format!("lside array: {}", e))?;

        let kside_partial = Array2::from_shape_vec(
            (rows, cols),
            all_f32[total_pixels * 2..total_pixels * 3].to_vec(),
        )
        .map_err(|e| format!("kside_partial array: {}", e))?;

        drop(data);
        buffers.staging_buffer.unmap();

        Ok(AnisoGpuResult {
            ldown,
            lside,
            kside_partial,
        })
    }

    // ── Buffer management ────────────────────────────────────────────────

    fn ensure_buffers(
        &self,
        rows: usize,
        cols: usize,
        n_pack: usize,
        n_patches: usize,
    ) -> Result<(), String> {
        let mut cache = self.cached.lock().unwrap();

        // Reuse if dimensions match
        if let Some(ref c) = *cache {
            if c.rows == rows && c.cols == cols && c.n_pack == n_pack && c.n_patches == n_patches {
                return Ok(());
            }
        }

        let total_pixels = rows * cols;
        let pixel_bytes = (total_pixels * 4) as u64;
        let shadow_bytes = (total_pixels * n_pack) as u64;
        // Pad shadow buffer to u32 alignment
        let shadow_bytes_aligned = (shadow_bytes + 3) & !3;
        let patch_bytes = (n_patches * 4) as u64;
        // Valid mask: 1 byte per pixel, padded to u32
        let valid_bytes = ((total_pixels + 3) & !3) as u64;

        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let input = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let output = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let params_buffer = make(
            "Aniso Params",
            std::mem::size_of::<AnisoParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let shmat_buffer = make("Aniso shmat", shadow_bytes_aligned, input);
        let vegshmat_buffer = make("Aniso vegshmat", shadow_bytes_aligned, input);
        let vbshvegshmat_buffer = make("Aniso vbshvegshmat", shadow_bytes_aligned, input);
        let asvf_buffer = make("Aniso asvf", pixel_bytes, input);
        let lup_buffer = make("Aniso lup", pixel_bytes, input);
        let valid_buffer = make("Aniso valid", valid_bytes, input);

        let patch_alt_buffer = make("Aniso patch_alt", patch_bytes, input);
        let patch_azi_buffer = make("Aniso patch_azi", patch_bytes, input);
        let steradians_buffer = make("Aniso steradians", patch_bytes, input);
        let esky_band_buffer = make("Aniso esky_band", patch_bytes, input);
        let lum_chi_buffer = make("Aniso lum_chi", patch_bytes, input);

        let out_ldown_buffer = make("Aniso out_ldown", pixel_bytes, output);
        let out_lside_buffer = make("Aniso out_lside", pixel_bytes, output);
        let out_kside_partial_buffer = make("Aniso out_kside_partial", pixel_bytes, output);

        let staging_buffer = make(
            "Aniso Staging",
            pixel_bytes * 3,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Aniso Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: shmat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vegshmat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vbshvegshmat_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: asvf_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: lup_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: valid_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: patch_alt_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: patch_azi_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: steradians_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: esky_band_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: lum_chi_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: out_ldown_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: out_lside_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: out_kside_partial_buffer.as_entire_binding() },
            ],
        });

        *cache = Some(CachedBuffers {
            rows,
            cols,
            n_pack,
            n_patches,
            params_buffer,
            shmat_buffer,
            vegshmat_buffer,
            vbshvegshmat_buffer,
            asvf_buffer,
            lup_buffer,
            valid_buffer,
            patch_alt_buffer,
            patch_azi_buffer,
            steradians_buffer,
            esky_band_buffer,
            lum_chi_buffer,
            out_ldown_buffer,
            out_lside_buffer,
            out_kside_partial_buffer,
            staging_buffer,
            bind_group,
        });

        Ok(())
    }

    // ── Bind group layout ────────────────────────────────────────────────

    fn bind_group_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        vec![
            uniform_entry(0),   // params
            storage_ro(1),      // shmat
            storage_ro(2),      // vegshmat
            storage_ro(3),      // vbshvegshmat
            storage_ro(4),      // asvf
            storage_ro(5),      // lup
            storage_ro(6),      // valid_mask
            storage_ro(7),      // patch_alt
            storage_ro(8),      // patch_azi
            storage_ro(9),      // steradians
            storage_ro(10),     // esky_band
            storage_ro(11),     // lum_chi
            storage_rw(12),     // out_ldown
            storage_rw(13),     // out_lside
            storage_rw(14),     // out_kside_partial
        ]
    }

    // ── Data conversion helpers ──────────────────────────────────────────

    /// Convert a 3D u8 ndarray to a contiguous byte Vec (row-major).
    fn contiguous_bytes_3d(arr: &ArrayView3<u8>) -> Vec<u8> {
        if let Some(slice) = arr.as_slice() {
            // Already contiguous — pad to u32 alignment
            let mut v = slice.to_vec();
            while v.len() % 4 != 0 {
                v.push(0);
            }
            v
        } else {
            let shape = arr.shape();
            let total = shape[0] * shape[1] * shape[2];
            let mut v = Vec::with_capacity((total + 3) & !3);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    for k in 0..shape[2] {
                        v.push(arr[[r, c, k]]);
                    }
                }
            }
            while v.len() % 4 != 0 {
                v.push(0);
            }
            v
        }
    }

    /// Convert a 2D f32 ndarray to a contiguous Vec<f32>.
    fn contiguous_f32_2d(arr: &ArrayView2<f32>) -> Vec<f32> {
        if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            let shape = arr.shape();
            let mut v = Vec::with_capacity(shape[0] * shape[1]);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    v.push(arr[[r, c]]);
                }
            }
            v
        }
    }

    /// Convert a 2D u8 ndarray to a contiguous byte Vec, padded to u32.
    fn contiguous_bytes_2d_u8(arr: &ArrayView2<u8>) -> Vec<u8> {
        let shape = arr.shape();
        let total = shape[0] * shape[1];
        let mut v = if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            let mut v = Vec::with_capacity(total);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    v.push(arr[[r, c]]);
                }
            }
            v
        };
        while v.len() % 4 != 0 {
            v.push(0);
        }
        v
    }

    /// Convert a 1D f32 ndarray to a contiguous Vec<f32>.
    fn contiguous_f32_1d(arr: &ArrayView1<f32>) -> Vec<f32> {
        if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            arr.iter().cloned().collect()
        }
    }
}
