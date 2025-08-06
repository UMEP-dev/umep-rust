use pyo3::prelude::*;

mod shadowing;
mod skyview;
mod sun;

#[pymodule]
fn rustalgos(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    // py_module.add_class::<common::Coord>()?;
    // py_module.add_function(wrap_pyfunction!(common::clipped_beta_wt, py_module)?)?;

    // Register submodules
    register_shadowing_module(py_module)?;
    register_skyview_module(py_module)?;
    register_sun_module(py_module)?;
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
    py_module.add_submodule(&submodule)?;
    Ok(())
}

fn register_sun_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "sun")?;
    submodule.add("__doc__", "Sun position and radiation calculations.")?;
    submodule.add_function(wrap_pyfunction!(sun::sun_on_surface, &submodule)?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}
