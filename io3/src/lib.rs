use pyo3::prelude::*;

use crate::types::image::Image;
use crate::types::pixel::Pixel;

mod types {
    pub mod io3_error;
    pub mod image;
    pub mod pixel;
}

// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// /// A Python module implemented in Rust.
// #[pymodule]
// fn io3(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     Ok(())
// }

#[pymodule]
fn io3(_py: Python, m: &PyModule) -> PyResult<()> {
    
    m.add_class::<Image>()?;
    m.add_class::<Pixel>()?;
    
    Ok(())
}