// GPU acceleration modules
#[cfg(feature = "gpu")]
pub mod shadow_gpu;
#[cfg(feature = "gpu")]
pub mod aniso_gpu;

#[cfg(feature = "gpu")]
pub use shadow_gpu::{create_shadow_gpu_context, ShadowGpuContext};
#[cfg(feature = "gpu")]
pub use aniso_gpu::AnisoGpuContext;
