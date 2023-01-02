use pyo3::prelude::*;

use std::error::Error;

use crate::types::pixel::Pixel;
use crate::types::io3_error::Io3Error;

#[pyclass]
pub struct Image {
    #[pyo3(get,set)]
    w: usize,
    #[pyo3(get,set)]
    h: usize, 
    #[pyo3(get,set)]
    nchannels: usize,
    data: Vec<f64>
}


#[pymethods]
impl Image {

    #[new]
    pub fn new(w: usize, h: usize, nchannels: usize) -> Image {
        let data = vec![0.0; (w*h*nchannels) as usize];
        Image {
            w,
            h,
            nchannels,
            data
        }
    }
    pub fn coord_is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < (self.w as i32) && y >= 0 && y < (self.h as i32)
    }


    pub fn unchecked_get_pixel(&self, x: i32, y: i32) -> Result<Pixel, Io3Error> {
        let start = match usize::try_from((self.nchannels as i32)*(y*(self.w as i32) + x)) {
            Ok(v) => v,
            Err(e) => {
                let boxed = Box::new(e);
                return Err(Io3Error::new(Some(boxed)))
            }
        };
        let mut pixel = Pixel::new();
        for i in 0..self.nchannels {
            pixel.push(*(self.data.get(start+i).ok_or(Io3Error::new(None))?))
        }
        Ok(pixel)
    }

    pub fn set_pixel(&mut self, x: i32, y: i32, pixel: &Pixel) -> Result<(),Io3Error>{

        // Out-of-bounds writes will simply be ignored
        // If needed, another function could be implemented that raises an error on out-of-bounds writes

        if(!self.coord_is_in_bounds(x, y)){
            return Ok(())
        }

        let start = usize::try_from((self.nchannels as i32)*(y*(self.w as i32) + x)).map_err(|e| Io3Error::new(Some(Box::new(e))))?;
        for i in 0..self.nchannels {
            *(self.data.get_mut(start+i).ok_or(Io3Error::new(None))?) = pixel.get_value(i)?;
        }
        Ok(())
    }

    pub fn get_pixel(&self, x: i32, y: i32, default_value: Option<Pixel>) -> Result<Option<Pixel>, Io3Error> {
        match default_value {
            Some(v) => {
                if(!self.coord_is_in_bounds(x, y)){
                    return Ok(Some(v))
                }
                Ok(Some(self.unchecked_get_pixel(x, y)?))
            },
            None => {
                if(!self.coord_is_in_bounds(x, y)){
                    return Ok(None)
                }
                Ok(Some(self.unchecked_get_pixel(x, y)?))
            }
        }
    }

    pub fn set_window(&mut self, x: i32, y: i32, window: &Image) -> Result<(),Io3Error> {
        let ww = window.w;
        let wh = window.h;
        for wx in 0..ww {
            for wy in 0..wh {
                let pixel = window.unchecked_get_pixel(wx as i32, wy as i32)?;
                self.set_pixel(x + (wx as i32), y+(wy as i32), &pixel)?;
            }
        }        
        Ok(())
    }

    pub fn get_window(&self, x: i32, y: i32, ww: usize, wh: usize, default_value: Option<Pixel>) -> Result<Image, Io3Error>{
        let mut window = Image::new(ww, wh, self.nchannels);
        for wx in 0..ww {
            for wy in 0..wh {
                let pixel = self.get_pixel(x+(wx as i32), y+(wy as i32),default_value.to_owned())?;
                window.set_pixel((wx as i32), (wy as i32), &(match pixel{
                    Some(value) => value,
                    None => {
                        return Err(Io3Error::new(None))
                    }
                }))?;
            } 
        } 
        Ok(window)
    }

    pub fn get_data(&self) -> Vec<f64> {
        self.data.clone()
    }

    pub fn set_data(&mut self, data: Vec<f64>){
        self.data.clear();
        self.data.extend(data.clone());
    }
    
}