use pyo3::prelude::*;

mod emissivity_models;
mod gvf;
mod patch_radiation;
mod shadowing;
mod sky;
mod skyview;
mod sun;
mod sunlit_shaded_patches;
mod vegetation;

#[pymodule]
fn rustalgos(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    // py_module.add_class::<common::Coord>()?;
    // py_module.add_function(wrap_pyfunction!(common::clipped_beta_wt, py_module)?)?;

    // Register submodules
    register_shadowing_module(py_module)?;
    register_skyview_module(py_module)?;
    register_gvf_module(py_module)?;
    register_sky_module(py_module)?;
    register_vegetation_module(py_module)?;
    py_module.add("__doc__", "UMEP algorithms implemented in Rust.")?;

    Ok(())
}

fn register_shadowing_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "shadowing")?;
    submodule.add("__doc__", "Shadow analysis.")?;
    submodule.add_function(wrap_pyfunction!(
        shadowing::calculate_shadows_wall_ht_25,
        &submodule
    )?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_skyview_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "skyview")?;
    submodule.add("__doc__", "Sky View Factor calculation.")?;
    submodule.add_function(wrap_pyfunction!(skyview::calculate_svf, &submodule)?)?;
    // Expose the SkyviewRunner class so Python can create a runner and poll progress()
    submodule.add_class::<skyview::SkyviewRunner>()?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_gvf_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "gvf")?;
    submodule.add("__doc__", "Ground View Factor calculation.")?;
    submodule.add_function(wrap_pyfunction!(gvf::gvf_calc, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_sky_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "sky")?;
    submodule.add("__doc__", "Anisotropic sky radiation calculations.")?;
    submodule.add_function(wrap_pyfunction!(sky::anisotropic_sky, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_vegetation_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "vegetation")?;
    submodule.add("__doc__", "Vegetation-related calculations.")?;
    submodule.add_class::<vegetation::LsideVegResult>()?;
    submodule.add_class::<vegetation::KsideVegResult>()?;
    submodule.add_function(wrap_pyfunction!(vegetation::lside_veg, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(vegetation::kside_veg, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}
