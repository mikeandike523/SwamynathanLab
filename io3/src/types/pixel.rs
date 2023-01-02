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
    pub fn get_values(&self) -> Result<Vec<f64>, Io3Error> {
        Ok(self.values.to_owned())
    }
    pub fn push_vec(&mut self, v: Vec<f64>) -> Pixel {
        let mut px = Pixel::new();
        px.__push_vec(&v);
        px
    }
}

impl Pixel {
    fn __push_vec(&mut self, values: &Vec<f64>) {
        for value in values.iter() {
            self.values.push(*value)
        }    
    }
}