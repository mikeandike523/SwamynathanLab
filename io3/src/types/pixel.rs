use pyo3::prelude::*;

use crate::types::io3_error::Io3Error;

#[derive(Debug, Clone, PartialEq)]
#[pyclass]
pub struct Pixel {
    pub values: Vec<f64>
}

#[pymethods]
impl Pixel {
    #[new]
    pub fn new() -> Pixel {
        Pixel {
            values: Vec::<f64>::new()
        }
    }
    pub fn push(&mut self, value: f64) {
         self.values.push(value)
    }
    pub fn get_value(&self, channel: usize) -> Result<f64, Io3Error> {
        Ok(*(self.values.get(channel).ok_or(Io3Error::new(None))?))
    }
}

impl Pixel {
    pub fn from_vec(v: &Vec<f64>) -> Pixel {
        let mut pixel = Pixel::new();
        pixel.push_vec(&v);
        pixel
    }
    pub fn push_vec(&mut self, values: &Vec<f64>) {
        for value in values.iter() {
            self.values.push(*value)
        }    
    }
}