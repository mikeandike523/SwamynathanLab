use pyo3::prelude::*;

/// Put module declarations for subfolders here
/// See the stackoverflow answer https://stackoverflow.com/a/58936090/5166365

mod types {
    pub mod rgb_image;
    pub mod rgb_color;
}

mod error_type;
mod error_types;

// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }


// All of the relevant RGB image operations go here
mod rgb_image {

    use crate::types::rgb_image::RGBImage;

    use pyo3::prelude::*;

    #[pyfunction]
    pub fn invert(image_json: String) -> PyResult<String> {
        let mut image = RGBImage::from_json_string(image_json).unwrap();
        image.invert_colors().unwrap();
        Ok(image.to_json_string().unwrap())
    }
}






/// A Python module implemented in Rust.
#[pymodule]
fn image_trioxide(_py: Python, m: &PyModule) -> PyResult<()> {

    let m_rgb_image = PyModule::new(_py,"rgb_image")?;

    m_rgb_image.add_function(wrap_pyfunction!(rgb_image::invert, m)?)?;

    m.add_submodule(m_rgb_image)?;

    Ok(())
}