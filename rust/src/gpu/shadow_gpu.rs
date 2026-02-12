use ndarray::{Array2, ArrayView2};
use std::sync::Arc;

/// Result struct for GPU shadow calculations
pub struct GpuShadowResult {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Option<Array2<f32>>,
    pub veg_blocks_bldg_sh: Option<Array2<f32>>,
    pub propagated_veg_height: Option<Array2<f32>>,
    pub wall_sh: Option<Array2<f32>>,
    pub wall_sun: Option<Array2<f32>>,
    pub wall_sh_veg: Option<Array2<f32>>,
    pub face_sh: Option<Array2<f32>>,
    pub face_sun: Option<Array2<f32>>,
}

/// SVF-specific shadow result: only the 3 arrays needed for SVF computation.
/// Skips wall outputs entirely, reducing staging bandwidth by ~70%.
pub struct SvfShadowResult {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Option<Array2<f32>>,
    pub veg_blocks_bldg_sh: Option<Array2<f32>>,
}

/// Uniform buffer matching the SvfAccumParams struct in svf_accumulation.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SvfAccumParams {
    total_pixels: u32,
    weight_iso: f32,
    weight_n: f32,
    weight_e: f32,
    weight_s: f32,
    weight_w: f32,
    has_veg: u32,
    _padding: u32,
}

/// Uniform buffer matching the U8PackParams struct in shadow_to_u8.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct U8PackParams {
    total_pixels: u32,
    num_quads: u32,
    has_veg: u32,
    _padding: u32,
}

/// Result of GPU SVF accumulation — 15 arrays (5 building + 5 veg + 5 aveg).
pub struct SvfAccumResult {
    pub svf: Array2<f32>,
    pub svf_n: Array2<f32>,
    pub svf_e: Array2<f32>,
    pub svf_s: Array2<f32>,
    pub svf_w: Array2<f32>,
    pub svf_veg: Option<Array2<f32>>,
    pub svf_veg_n: Option<Array2<f32>>,
    pub svf_veg_e: Option<Array2<f32>>,
    pub svf_veg_s: Option<Array2<f32>>,
    pub svf_veg_w: Option<Array2<f32>>,
    pub svf_aveg: Option<Array2<f32>>,
    pub svf_aveg_n: Option<Array2<f32>>,
    pub svf_aveg_e: Option<Array2<f32>>,
    pub svf_aveg_s: Option<Array2<f32>>,
    pub svf_aveg_w: Option<Array2<f32>>,
}

/// Cached GPU buffers for shadow calculations.
/// Reused across calls when grid dimensions remain constant.
struct CachedBuffers {
    rows: usize,
    cols: usize,
    // Binding 0: Params (UNIFORM | COPY_DST)
    params_buffer: wgpu::Buffer,
    // Binding 1: DSM input (STORAGE | COPY_DST)
    dsm_buffer: wgpu::Buffer,
    // Binding 2: Building shadow output (STORAGE | COPY_SRC)
    bldg_shadow_buffer: wgpu::Buffer,
    // Binding 3: Propagated building height (STORAGE | COPY_SRC | COPY_DST)
    propagated_bldg_height_buffer: wgpu::Buffer,
    // Bindings 4-6: Vegetation inputs (STORAGE | COPY_DST)
    veg_canopy_buffer: wgpu::Buffer,
    veg_trunk_buffer: wgpu::Buffer,
    bush_buffer: wgpu::Buffer,
    // Bindings 7-9: Vegetation outputs (STORAGE | COPY_SRC)
    veg_shadow_buffer: wgpu::Buffer,
    propagated_veg_height_buffer: wgpu::Buffer,
    veg_blocks_bldg_shadow_buffer: wgpu::Buffer,
    // Bindings 10-11: Wall inputs (STORAGE | COPY_DST)
    walls_buffer: wgpu::Buffer,
    aspect_buffer: wgpu::Buffer,
    // Bindings 12-16: Wall outputs (STORAGE | COPY_SRC)
    wall_sh_buffer: wgpu::Buffer,
    wall_sun_buffer: wgpu::Buffer,
    wall_sh_veg_buffer: wgpu::Buffer,
    face_sh_buffer: wgpu::Buffer,
    face_sun_buffer: wgpu::Buffer,
    // Staging buffer for GPU -> CPU readback (MAP_READ | COPY_DST)
    staging_buffer: wgpu::Buffer,
    // Bind group (references all buffer handles)
    bind_group: wgpu::BindGroup,
    // --- SVF accumulation (populated by init_svf_accumulation) ---
    svf_params_buffer: Option<wgpu::Buffer>,
    svf_data_buffer: Option<wgpu::Buffer>,
    svf_result_staging: Option<wgpu::Buffer>,
    svf_bind_group: Option<wgpu::BindGroup>,
    svf_has_veg: bool,
    svf_num_arrays: usize, // 5 (no veg) or 15 (with veg)
    // --- Shadow uint8 packing + double-buffered staging ---
    shadow_u8_params_buffer: Option<wgpu::Buffer>,
    shadow_u8_output_buffer: Option<wgpu::Buffer>,
    shadow_u8_staging: [Option<wgpu::Buffer>; 2],
    shadow_u8_bind_group: Option<wgpu::BindGroup>,
    shadow_u8_packed_size: u64, // total bytes in packed output
}

/// GPU context for shadow calculations - maintains GPU resources across multiple calls
pub struct ShadowGpuContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Adapter-reported maximum single buffer size in bytes.
    pub(crate) max_buffer_size: u64,
    pipeline: wgpu::ComputePipeline,
    wall_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    svf_pipeline: wgpu::ComputePipeline,
    svf_bind_group_layout: wgpu::BindGroupLayout,
    shadow_u8_pipeline: wgpu::ComputePipeline,
    shadow_u8_bind_group_layout: wgpu::BindGroupLayout,
    /// Cached buffers reused across calls with same grid dimensions
    cached: std::sync::Mutex<Option<CachedBuffers>>,
}

/// Uniform buffer struct for shadow shader parameters
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowParams {
    rows: u32,
    cols: u32,
    azimuth_rad: f32,
    altitude_rad: f32,
    sin_azimuth: f32,
    cos_azimuth: f32,
    tan_azimuth: f32,
    tan_altitude_by_scale: f32,
    scale: f32,
    max_index: f32,
    max_local_dsm_ht: f32,
    has_veg: u32,
    has_walls: u32,
    _padding: u32,
}

impl ShadowGpuContext {
    /// Initialize GPU context - call once at startup
    pub async fn new() -> Result<Self, String> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find suitable GPU adapter: {:?}", e))?;

        // Request higher limits for storage buffers and buffer sizes
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16; // We need 16 storage buffers
        // Request native max buffer sizes for large SVF accumulation buffers
        // (default 256 MiB is too small for packed 15-array SVF at 6.7M pixels)
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shadow Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow_propagation.wgsl").into()),
        });

        // Create bind group layout for all shadow types
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Bind Group Layout"),
                entries: &[
                    // Binding 0: Params buffer (uniforms)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: DSM input (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: Building shadow output
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: Propagated building height buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 4-9: Vegetation inputs and outputs
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bindings 10-16: Wall inputs and outputs
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Propagation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let wall_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wall Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_wall_shadows"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- SVF accumulation pipeline ---
        let svf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SVF Accumulation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("svf_accumulation.wgsl").into()),
        });

        let svf_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SVF Accumulation Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: veg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: veg_blocks_bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 4: svf_data (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let svf_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SVF Accumulation Pipeline Layout"),
            bind_group_layouts: &[&svf_bind_group_layout],
            push_constant_ranges: &[],
        });

        let svf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SVF Accumulation Pipeline"),
            layout: Some(&svf_pipeline_layout),
            module: &svf_shader,
            entry_point: Some("accumulate_svf"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Shadow uint8 packing pipeline ---
        let shadow_u8_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow U8 Pack Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow_to_u8.wgsl").into()),
        });

        let shadow_u8_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow U8 Pack Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: veg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: veg_blocks_bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 4: packed_output (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shadow_u8_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow U8 Pack Pipeline Layout"),
                bind_group_layouts: &[&shadow_u8_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shadow_u8_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Shadow U8 Pack Pipeline"),
                layout: Some(&shadow_u8_pipeline_layout),
                module: &shadow_u8_shader,
                entry_point: Some("shadow_to_u8"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            max_buffer_size: adapter_limits.max_buffer_size,
            pipeline,
            wall_pipeline,
            bind_group_layout,
            svf_pipeline,
            svf_bind_group_layout,
            shadow_u8_pipeline,
            shadow_u8_bind_group_layout,
            cached: std::sync::Mutex::new(None),
        })
    }

    /// Allocate a fresh set of GPU buffers for the given grid dimensions.
    fn allocate_buffers(&self, rows: usize, cols: usize) -> CachedBuffers {
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let params_size = std::mem::size_of::<ShadowParams>() as u64;

        // Helper to create a storage buffer with given usage flags
        let make_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let input_usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let output_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let params_buffer = make_buffer(
            "Shadow Params Buffer",
            params_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dsm_buffer = make_buffer("DSM Buffer", buffer_size, input_usage);
        let bldg_shadow_buffer = make_buffer("Building Shadow Buffer", buffer_size, output_usage);
        let propagated_bldg_height_buffer =
            make_buffer("Propagated Building Height Buffer", buffer_size, working_usage);
        let veg_canopy_buffer = make_buffer("Veg Canopy Buffer", buffer_size, input_usage);
        let veg_trunk_buffer = make_buffer("Veg Trunk Buffer", buffer_size, input_usage);
        let bush_buffer = make_buffer("Bush Buffer", buffer_size, input_usage);
        let veg_shadow_buffer = make_buffer("Veg Shadow Buffer", buffer_size, output_usage);
        let propagated_veg_height_buffer =
            make_buffer("Propagated Veg Height Buffer", buffer_size, working_usage);
        let veg_blocks_bldg_shadow_buffer =
            make_buffer("Veg Blocks Bldg Shadow Buffer", buffer_size, output_usage);
        let walls_buffer = make_buffer("Walls Buffer", buffer_size, input_usage);
        let aspect_buffer = make_buffer("Aspect Buffer", buffer_size, input_usage);
        let wall_sh_buffer = make_buffer("Wall Shadow Buffer", buffer_size, output_usage);
        let wall_sun_buffer = make_buffer("Wall Sun Buffer", buffer_size, output_usage);
        let wall_sh_veg_buffer = make_buffer("Wall Shadow Veg Buffer", buffer_size, output_usage);
        let face_sh_buffer = make_buffer("Face Shadow Buffer", buffer_size, output_usage);
        let face_sun_buffer = make_buffer("Face Sun Buffer", buffer_size, output_usage);

        let staging_buffer = make_buffer(
            "Staging Buffer",
            buffer_size * 10,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dsm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: propagated_bldg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: veg_canopy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: veg_trunk_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bush_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: propagated_veg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: walls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: aspect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wall_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wall_sun_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wall_sh_veg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: face_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: face_sun_buffer.as_entire_binding(),
                },
            ],
        });

        eprintln!(
            "[GPU] Allocated buffer cache for {}x{} grid ({:.1} MB)",
            rows,
            cols,
            (buffer_size * 17 + buffer_size * 10) as f64 / 1_048_576.0
        );

        CachedBuffers {
            rows,
            cols,
            params_buffer,
            dsm_buffer,
            bldg_shadow_buffer,
            propagated_bldg_height_buffer,
            veg_canopy_buffer,
            veg_trunk_buffer,
            bush_buffer,
            veg_shadow_buffer,
            propagated_veg_height_buffer,
            veg_blocks_bldg_shadow_buffer,
            walls_buffer,
            aspect_buffer,
            wall_sh_buffer,
            wall_sun_buffer,
            wall_sh_veg_buffer,
            face_sh_buffer,
            face_sun_buffer,
            staging_buffer,
            bind_group,
            svf_params_buffer: None,
            svf_data_buffer: None,
            svf_result_staging: None,
            svf_bind_group: None,
            svf_has_veg: false,
            svf_num_arrays: 0,
            shadow_u8_params_buffer: None,
            shadow_u8_output_buffer: None,
            shadow_u8_staging: [None, None],
            shadow_u8_bind_group: None,
            shadow_u8_packed_size: 0,
        }
    }

    /// Optimized version accepting ArrayView to avoid unnecessary copies
    #[allow(clippy::too_many_arguments)]
    pub fn compute_all_shadows_view(
        &self,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
        walls_opt: Option<ArrayView2<f32>>,
        aspect_opt: Option<ArrayView2<f32>>,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
    ) -> Result<GpuShadowResult, String> {
        let (rows, cols) = dsm.dim();
        let total_pixels = rows * cols;

        // Check if vegetation inputs are provided
        let has_veg =
            veg_canopy_dsm_opt.is_some() && veg_trunk_dsm_opt.is_some() && bush_opt.is_some();
        let has_walls = walls_opt.is_some() && aspect_opt.is_some();

        // Helper to get contiguous slice or allocate temp buffer
        let get_slice = |view: ArrayView2<f32>| -> Vec<f32> {
            if view.is_standard_layout() {
                view.as_slice().unwrap().to_vec()
            } else {
                view.iter().copied().collect()
            }
        };

        // Use slice directly when contiguous, otherwise allocate
        let dsm_data = get_slice(dsm);
        let veg_canopy_data = veg_canopy_dsm_opt
            .map(get_slice)
            .unwrap_or_else(|| vec![0.0; total_pixels]);
        let veg_trunk_data = veg_trunk_dsm_opt
            .map(get_slice)
            .unwrap_or_else(|| vec![0.0; total_pixels]);
        let bush_data = bush_opt
            .map(get_slice)
            .unwrap_or_else(|| vec![0.0; total_pixels]);
        let walls_data = walls_opt
            .map(get_slice)
            .unwrap_or_else(|| vec![0.0; total_pixels]);
        let aspect_data = aspect_opt
            .map(get_slice)
            .unwrap_or_else(|| vec![0.0; total_pixels]);

        // Precompute trigonometric values
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let sin_azimuth = azimuth_rad.sin();
        let cos_azimuth = azimuth_rad.cos();
        let tan_azimuth = azimuth_rad.tan();
        let tan_altitude_by_scale = altitude_rad.tan() / scale;
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let max_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_index = (max_reach_m / scale).ceil();

        let params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth,
            cos_azimuth,
            tan_azimuth,
            tan_altitude_by_scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: if has_walls { 1 } else { 0 },
            _padding: 0,
        };

        // Get or create cached buffers for this grid size
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_ref()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Write input data into cached buffers
        self.queue.write_buffer(
            &buffers.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
        self.queue
            .write_buffer(&buffers.dsm_buffer, 0, bytemuck::cast_slice(&dsm_data));
        self.queue.write_buffer(
            &buffers.veg_canopy_buffer,
            0,
            bytemuck::cast_slice(&veg_canopy_data),
        );
        self.queue.write_buffer(
            &buffers.veg_trunk_buffer,
            0,
            bytemuck::cast_slice(&veg_trunk_data),
        );
        self.queue
            .write_buffer(&buffers.bush_buffer, 0, bytemuck::cast_slice(&bush_data));
        self.queue.write_buffer(
            &buffers.walls_buffer,
            0,
            bytemuck::cast_slice(&walls_data),
        );
        self.queue.write_buffer(
            &buffers.aspect_buffer,
            0,
            bytemuck::cast_slice(&aspect_data),
        );

        // Initialize working buffers (shader modifies these, must reset each call)
        self.queue.write_buffer(
            &buffers.propagated_bldg_height_buffer,
            0,
            bytemuck::cast_slice(&dsm_data),
        );
        if has_veg {
            self.queue.write_buffer(
                &buffers.propagated_veg_height_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
        }

        // Encode and submit compute passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shadow Compute Encoder"),
            });

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let num_workgroups_x = (cols as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroups_y = (rows as u32 + workgroup_size_y - 1) / workgroup_size_y;

        // First pass: Main shadow propagation
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Second pass: Wall shadows (if enabled)
        if has_walls {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wall Shadow Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.wall_pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy results to cached staging buffer
        encoder.copy_buffer_to_buffer(
            &buffers.bldg_shadow_buffer,
            0,
            &buffers.staging_buffer,
            0,
            buffer_size,
        );

        // Copy vegetation outputs if enabled
        let veg_offset = buffer_size;
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                veg_offset,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.veg_blocks_bldg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                veg_offset + buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.propagated_veg_height_buffer,
                0,
                &buffers.staging_buffer,
                veg_offset + buffer_size * 2,
                buffer_size,
            );
        }

        // Copy wall outputs if enabled
        let wall_offset = buffer_size * 4;
        if has_walls {
            encoder.copy_buffer_to_buffer(
                &buffers.wall_sh_buffer,
                0,
                &buffers.staging_buffer,
                wall_offset,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.wall_sun_buffer,
                0,
                &buffers.staging_buffer,
                wall_offset + buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.wall_sh_veg_buffer,
                0,
                &buffers.staging_buffer,
                wall_offset + buffer_size * 2,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.face_sh_buffer,
                0,
                &buffers.staging_buffer,
                wall_offset + buffer_size * 3,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.face_sun_buffer,
                0,
                &buffers.staging_buffer,
                wall_offset + buffer_size * 4,
                buffer_size,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        // Read back all results from cached staging buffer
        let buffer_slice = buffers.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let all_data: &[f32] = bytemuck::cast_slice(&data);

        // Extract building shadow
        let bldg_sh = Array2::from_shape_vec((rows, cols), all_data[..total_pixels].to_vec())
            .map_err(|e| format!("Failed to create building shadow array: {}", e))?;

        // Extract vegetation results if enabled
        let (veg_sh, veg_blocks_bldg_sh, propagated_veg_height) = if has_veg {
            let veg_offset_px = total_pixels;
            let veg_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px..veg_offset_px + total_pixels].to_vec(),
            )
            .ok();
            let veg_blocks = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px + total_pixels..veg_offset_px + total_pixels * 2].to_vec(),
            )
            .ok();
            let prop_veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[veg_offset_px + total_pixels * 2..veg_offset_px + total_pixels * 3]
                    .to_vec(),
            )
            .ok();
            (veg_sh, veg_blocks, prop_veg)
        } else {
            (None, None, None)
        };

        // Extract wall results if enabled
        let (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun) = if has_walls {
            let wall_offset_px = total_pixels * 4;
            let wall_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px..wall_offset_px + total_pixels].to_vec(),
            )
            .ok();
            let wall_sun = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels..wall_offset_px + total_pixels * 2].to_vec(),
            )
            .ok();
            let wall_sh_veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 2..wall_offset_px + total_pixels * 3]
                    .to_vec(),
            )
            .ok();
            let face_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 3..wall_offset_px + total_pixels * 4]
                    .to_vec(),
            )
            .ok();
            let face_sun = Array2::from_shape_vec(
                (rows, cols),
                all_data[wall_offset_px + total_pixels * 4..wall_offset_px + total_pixels * 5]
                    .to_vec(),
            )
            .ok();
            (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun)
        } else {
            (None, None, None, None, None)
        };

        drop(data);
        buffers.staging_buffer.unmap();

        Ok(GpuShadowResult {
            bldg_sh,
            veg_sh,
            veg_blocks_bldg_sh,
            propagated_veg_height,
            wall_sh,
            wall_sun,
            wall_sh_veg,
            face_sh,
            face_sun,
        })
    }

    /// SVF-optimized shadow computation.
    ///
    /// Compared to `compute_all_shadows_view()`, this:
    /// - Skips the wall shader dispatch entirely
    /// - Skips writing wall/aspect input buffers (saves ~50MB/call)
    /// - Copies only 3 arrays to staging instead of 10 (~70% less readback)
    #[allow(clippy::too_many_arguments)]
    pub fn compute_shadows_for_svf(
        &self,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
    ) -> Result<SvfShadowResult, String> {
        let (rows, cols) = dsm.dim();
        let total_pixels = rows * cols;

        // Handle zenith case (altitude >= 89.5°): no shadows from directly overhead
        if altitude_deg >= 89.5 {
            let dim = (rows, cols);
            return Ok(SvfShadowResult {
                bldg_sh: Array2::ones(dim),
                veg_sh: if veg_canopy_dsm_opt.is_some() {
                    Some(Array2::ones(dim))
                } else {
                    None
                },
                veg_blocks_bldg_sh: if veg_canopy_dsm_opt.is_some() {
                    Some(Array2::ones(dim))
                } else {
                    None
                },
            });
        }

        let has_veg =
            veg_canopy_dsm_opt.is_some() && veg_trunk_dsm_opt.is_some() && bush_opt.is_some();

        let get_slice = |view: ArrayView2<f32>| -> Vec<f32> {
            if view.is_standard_layout() {
                view.as_slice().unwrap().to_vec()
            } else {
                view.iter().copied().collect()
            }
        };

        let dsm_data = get_slice(dsm);

        // Precompute trigonometric values
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let max_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_index = (max_reach_m / scale).ceil();

        let params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth: azimuth_rad.sin(),
            cos_azimuth: azimuth_rad.cos(),
            tan_azimuth: azimuth_rad.tan(),
            tan_altitude_by_scale: altitude_rad.tan() / scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: 0, // SVF never uses walls
            _padding: 0,
        };

        // Get or create cached buffers
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_ref()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Write only needed inputs (skip walls/aspect entirely)
        self.queue.write_buffer(
            &buffers.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
        self.queue
            .write_buffer(&buffers.dsm_buffer, 0, bytemuck::cast_slice(&dsm_data));
        self.queue.write_buffer(
            &buffers.propagated_bldg_height_buffer,
            0,
            bytemuck::cast_slice(&dsm_data),
        );

        if has_veg {
            let veg_canopy_data = get_slice(veg_canopy_dsm_opt.unwrap());
            let veg_trunk_data = get_slice(veg_trunk_dsm_opt.unwrap());
            let bush_data = get_slice(bush_opt.unwrap());
            self.queue.write_buffer(
                &buffers.veg_canopy_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
            self.queue.write_buffer(
                &buffers.veg_trunk_buffer,
                0,
                bytemuck::cast_slice(&veg_trunk_data),
            );
            self.queue
                .write_buffer(&buffers.bush_buffer, 0, bytemuck::cast_slice(&bush_data));
            self.queue.write_buffer(
                &buffers.propagated_veg_height_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
        }

        // Encode: shadow propagation only (no wall pass)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Shadow Compute Encoder"),
            });

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let num_workgroups_x = (cols as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroups_y = (rows as u32 + workgroup_size_y - 1) / workgroup_size_y;

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SVF Shadow Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }
        // No wall pass — SVF never uses walls

        // Copy only 3 arrays to staging (instead of up to 10)
        encoder.copy_buffer_to_buffer(
            &buffers.bldg_shadow_buffer,
            0,
            &buffers.staging_buffer,
            0,
            buffer_size,
        );
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.veg_blocks_bldg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                buffer_size * 2,
                buffer_size,
            );
        }

        self.queue.submit(Some(encoder.finish()));

        // Map only what we need (1 or 3 arrays)
        let read_size = if has_veg {
            buffer_size * 3
        } else {
            buffer_size
        };
        let buffer_slice = buffers.staging_buffer.slice(..read_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let all_data: &[f32] = bytemuck::cast_slice(&data);

        let bldg_sh = Array2::from_shape_vec((rows, cols), all_data[..total_pixels].to_vec())
            .map_err(|e| format!("Failed to create bldg_sh array: {}", e))?;

        let (veg_sh, veg_blocks_bldg_sh) = if has_veg {
            let veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[total_pixels..total_pixels * 2].to_vec(),
            )
            .ok();
            let veg_blocks = Array2::from_shape_vec(
                (rows, cols),
                all_data[total_pixels * 2..total_pixels * 3].to_vec(),
            )
            .ok();
            (veg, veg_blocks)
        } else {
            (None, None)
        };

        drop(data);
        buffers.staging_buffer.unmap();

        Ok(SvfShadowResult {
            bldg_sh,
            veg_sh,
            veg_blocks_bldg_sh,
        })
    }

    /// Initialize SVF accumulation buffers. Call once before the 153-patch loop.
    ///
    /// Allocates the packed SVF data buffer (15 × pixels for veg, 5 × for no-veg),
    /// zeroes it, and creates the bind group referencing shadow output buffers.
    /// Also writes static inputs (DSM, veg) to shadow buffers once.
    pub fn init_svf_accumulation(
        &self,
        rows: usize,
        cols: usize,
        has_veg: bool,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
    ) -> Result<(), String> {
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let num_arrays: usize = if has_veg { 15 } else { 5 };
        let svf_data_size = buffer_size * num_arrays as u64;

        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        // Ensure shadow buffers are allocated
        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        // Write static inputs to shadow buffers once (avoids re-uploading per patch)
        let get_slice = |view: ArrayView2<f32>| -> Vec<f32> {
            if view.is_standard_layout() {
                view.as_slice().unwrap().to_vec()
            } else {
                view.iter().copied().collect()
            }
        };

        let dsm_data = get_slice(dsm);
        self.queue
            .write_buffer(&buffers.dsm_buffer, 0, bytemuck::cast_slice(&dsm_data));

        if has_veg {
            if let (Some(vc), Some(vt), Some(b)) =
                (veg_canopy_dsm_opt, veg_trunk_dsm_opt, bush_opt)
            {
                let vc_data = get_slice(vc);
                let vt_data = get_slice(vt);
                let b_data = get_slice(b);
                self.queue.write_buffer(
                    &buffers.veg_canopy_buffer,
                    0,
                    bytemuck::cast_slice(&vc_data),
                );
                self.queue.write_buffer(
                    &buffers.veg_trunk_buffer,
                    0,
                    bytemuck::cast_slice(&vt_data),
                );
                self.queue
                    .write_buffer(&buffers.bush_buffer, 0, bytemuck::cast_slice(&b_data));
            }
        }

        // Create SVF-specific buffers
        let make_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let svf_params_buffer = make_buffer(
            "SVF Accum Params",
            std::mem::size_of::<SvfAccumParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let svf_data_buffer = make_buffer(
            "SVF Data Buffer",
            svf_data_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let svf_result_staging = make_buffer(
            "SVF Result Staging",
            svf_data_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Zero-initialize SVF data buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Init Encoder"),
            });
        encoder.clear_buffer(&svf_data_buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));

        // Create SVF bind group referencing shadow output buffers + SVF buffers
        let svf_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SVF Accumulation Bind Group"),
            layout: &self.svf_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: svf_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: svf_data_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Shadow uint8 packing buffers + double-buffered staging ---
        let num_quads = ((total_pixels + 3) / 4) as u64;
        let num_packed_arrays: u64 = if has_veg { 3 } else { 1 };
        let packed_output_size = num_quads * 4 * num_packed_arrays; // bytes (u32 per quad)

        let shadow_u8_params_buffer = make_buffer(
            "Shadow U8 Params",
            std::mem::size_of::<U8PackParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let shadow_u8_output_buffer = make_buffer(
            "Shadow U8 Output",
            packed_output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let shadow_u8_staging_0 = make_buffer(
            "Shadow U8 Staging 0",
            packed_output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let shadow_u8_staging_1 = make_buffer(
            "Shadow U8 Staging 1",
            packed_output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Write static U8 pack params (doesn't change per patch)
        let u8_params = U8PackParams {
            total_pixels: total_pixels as u32,
            num_quads: num_quads as u32,
            has_veg: if has_veg { 1 } else { 0 },
            _padding: 0,
        };
        self.queue.write_buffer(
            &shadow_u8_params_buffer,
            0,
            bytemuck::cast_slice(&[u8_params]),
        );

        let shadow_u8_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow U8 Pack Bind Group"),
            layout: &self.shadow_u8_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_u8_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_u8_output_buffer.as_entire_binding(),
                },
            ],
        });

        eprintln!(
            "[GPU] SVF accumulation initialized: {}x{} grid, {} SVF arrays ({:.1} MB), u8 staging ({:.1} MB × 2)",
            rows,
            cols,
            num_arrays,
            svf_data_size as f64 / 1_048_576.0,
            packed_output_size as f64 / 1_048_576.0
        );

        buffers.svf_params_buffer = Some(svf_params_buffer);
        buffers.svf_data_buffer = Some(svf_data_buffer);
        buffers.svf_result_staging = Some(svf_result_staging);
        buffers.svf_bind_group = Some(svf_bind_group);
        buffers.svf_has_veg = has_veg;
        buffers.svf_num_arrays = num_arrays;
        buffers.shadow_u8_params_buffer = Some(shadow_u8_params_buffer);
        buffers.shadow_u8_output_buffer = Some(shadow_u8_output_buffer);
        buffers.shadow_u8_staging = [Some(shadow_u8_staging_0), Some(shadow_u8_staging_1)];
        buffers.shadow_u8_bind_group = Some(shadow_u8_bind_group);
        buffers.shadow_u8_packed_size = packed_output_size;

        Ok(())
    }

    /// Per-patch: dispatch shadow + SVF accumulate + uint8 pack to GPU (non-blocking).
    ///
    /// Returns a SubmissionIndex for later synchronization. The shadow results are
    /// packed to uint8 on the GPU and copied to staging[slot] for later readback
    /// via `read_shadow_staging()`. SVF accumulation happens on-GPU (no readback).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_shadow_and_accumulate_svf(
        &self,
        staging_slot: usize,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
        weight_iso: f32,
        weight_n: f32,
        weight_e: f32,
        weight_s: f32,
        weight_w: f32,
    ) -> Result<wgpu::SubmissionIndex, String> {
        assert!(staging_slot < 2, "staging_slot must be 0 or 1");

        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache empty — call init_svf_accumulation first".to_string())?;

        let svf_params_buf = buffers
            .svf_params_buffer
            .as_ref()
            .ok_or_else(|| "SVF not initialized".to_string())?;
        let has_veg = buffers.svf_has_veg;
        let rows = buffers.rows;
        let cols = buffers.cols;
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Write shadow params (only thing that changes per patch)
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let max_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_index = (max_reach_m / scale).ceil();

        let shadow_params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth: azimuth_rad.sin(),
            cos_azimuth: azimuth_rad.cos(),
            tan_azimuth: azimuth_rad.tan(),
            tan_altitude_by_scale: altitude_rad.tan() / scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: 0,
            _padding: 0,
        };

        self.queue.write_buffer(
            &buffers.params_buffer,
            0,
            bytemuck::cast_slice(&[shadow_params]),
        );

        // Write SVF accumulation params
        let svf_params = SvfAccumParams {
            total_pixels: total_pixels as u32,
            weight_iso,
            weight_n,
            weight_e,
            weight_s,
            weight_w,
            has_veg: if has_veg { 1 } else { 0 },
            _padding: 0,
        };

        self.queue
            .write_buffer(svf_params_buf, 0, bytemuck::cast_slice(&[svf_params]));

        // Build command encoder with 3 passes + staging copy
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Shadow+Accum+U8 Encoder"),
            });

        // Reset propagated height buffers (GPU→GPU copy from static inputs)
        encoder.copy_buffer_to_buffer(
            &buffers.dsm_buffer,
            0,
            &buffers.propagated_bldg_height_buffer,
            0,
            buffer_size,
        );
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_canopy_buffer,
                0,
                &buffers.propagated_veg_height_buffer,
                0,
                buffer_size,
            );
        }

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let num_workgroups_x = (cols as u32 + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroups_y = (rows as u32 + workgroup_size_y - 1) / workgroup_size_y;

        // Pass 1: Shadow propagation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Propagation (SVF)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Pass 2: SVF accumulation (reads shadow outputs, accumulates into svf_data)
        {
            let svf_bg = buffers
                .svf_bind_group
                .as_ref()
                .ok_or("SVF bind group missing")?;
            let svf_workgroups = (total_pixels as u32 + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SVF Accumulation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.svf_pipeline);
            pass.set_bind_group(0, svf_bg, &[]);
            pass.dispatch_workgroups(svf_workgroups, 1, 1);
        }

        // Pass 3: Pack shadows to uint8 (reads shadow outputs → packed u32 output)
        {
            let u8_bg = buffers
                .shadow_u8_bind_group
                .as_ref()
                .ok_or("Shadow U8 bind group missing")?;
            let num_quads = ((total_pixels + 3) / 4) as u32;
            let u8_workgroups = (num_quads + 255) / 256;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow U8 Pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shadow_u8_pipeline);
            pass.set_bind_group(0, u8_bg, &[]);
            pass.dispatch_workgroups(u8_workgroups, 1, 1);
        }

        // Copy packed uint8 output to staging[slot]
        let packed_size = buffers.shadow_u8_packed_size;
        let staging = buffers.shadow_u8_staging[staging_slot]
            .as_ref()
            .ok_or("Shadow U8 staging not initialized")?;
        let output_buf = buffers
            .shadow_u8_output_buffer
            .as_ref()
            .ok_or("Shadow U8 output buffer missing")?;
        encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, packed_size);

        // Submit — DO NOT poll. Return submission index for caller to sync later.
        let idx = self.queue.submit(Some(encoder.finish()));
        Ok(idx)
    }

    /// Read packed uint8 shadow data from a staging buffer after a previous dispatch.
    ///
    /// Waits for the specific submission to complete, maps the staging buffer,
    /// and returns the raw bytes. The caller unpacks into the shadow matrix.
    pub fn read_shadow_staging(
        &self,
        staging_slot: usize,
        submission_idx: wgpu::SubmissionIndex,
    ) -> Result<Vec<u8>, String> {
        assert!(staging_slot < 2, "staging_slot must be 0 or 1");

        let cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_ref()
            .ok_or_else(|| "Buffer cache empty".to_string())?;

        let packed_size = buffers.shadow_u8_packed_size;
        let staging = buffers.shadow_u8_staging[staging_slot]
            .as_ref()
            .ok_or("Shadow U8 staging not initialized")?;

        // Wait for this specific submission to complete
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_idx),
                timeout: None,
            })
            .map_err(|e| format!("Poll failed: {:?}", e))?;

        // Map staging buffer
        let buffer_slice = staging.slice(..packed_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Submission already complete, this should return immediately
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("Poll for map failed: {:?}", e))?;
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map staging: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let bytes = data.to_vec();

        drop(data);
        staging.unmap();

        Ok(bytes)
    }

    /// After all patches: read back accumulated SVF values from GPU.
    pub fn read_svf_results(&self) -> Result<SvfAccumResult, String> {
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache empty".to_string())?;

        let svf_data_buf = buffers
            .svf_data_buffer
            .as_ref()
            .ok_or_else(|| "SVF not initialized".to_string())?;
        let svf_staging = buffers
            .svf_result_staging
            .as_ref()
            .ok_or_else(|| "SVF staging not initialized".to_string())?;

        let rows = buffers.rows;
        let cols = buffers.cols;
        let total_pixels = rows * cols;
        let has_veg = buffers.svf_has_veg;
        let num_arrays = buffers.svf_num_arrays;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let svf_data_size = buffer_size * num_arrays as u64;

        // Copy svf_data to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Read Results Encoder"),
            });
        encoder.copy_buffer_to_buffer(svf_data_buf, 0, svf_staging, 0, svf_data_size);
        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = svf_staging.slice(..svf_data_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receiver
            .recv()
            .unwrap()
            .map_err(|e| format!("Failed to map SVF staging: {:?}", e))?;

        let data = slice.get_mapped_range();
        let all: &[f32] = bytemuck::cast_slice(&data);
        let n = total_pixels;

        let extract = |offset: usize| -> Array2<f32> {
            Array2::from_shape_vec((rows, cols), all[offset..offset + n].to_vec()).unwrap()
        };

        let svf = extract(0);
        let svf_n = extract(n);
        let svf_e = extract(2 * n);
        let svf_s = extract(3 * n);
        let svf_w = extract(4 * n);

        let (svf_veg, svf_veg_n, svf_veg_e, svf_veg_s, svf_veg_w) = if has_veg {
            (
                Some(extract(5 * n)),
                Some(extract(6 * n)),
                Some(extract(7 * n)),
                Some(extract(8 * n)),
                Some(extract(9 * n)),
            )
        } else {
            (None, None, None, None, None)
        };

        let (svf_aveg, svf_aveg_n, svf_aveg_e, svf_aveg_s, svf_aveg_w) = if has_veg {
            (
                Some(extract(10 * n)),
                Some(extract(11 * n)),
                Some(extract(12 * n)),
                Some(extract(13 * n)),
                Some(extract(14 * n)),
            )
        } else {
            (None, None, None, None, None)
        };

        drop(data);
        svf_staging.unmap();

        Ok(SvfAccumResult {
            svf,
            svf_n,
            svf_e,
            svf_s,
            svf_w,
            svf_veg,
            svf_veg_n,
            svf_veg_e,
            svf_veg_s,
            svf_veg_w,
            svf_aveg,
            svf_aveg_n,
            svf_aveg_e,
            svf_aveg_s,
            svf_aveg_w,
        })
    }
}

/// Synchronous wrapper that blocks on async GPU initialization
pub fn create_shadow_gpu_context() -> Result<ShadowGpuContext, String> {
    pollster::block_on(ShadowGpuContext::new())
}
