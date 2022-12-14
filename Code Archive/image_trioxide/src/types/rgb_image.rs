//! Provides a class to store RGB pixel data.
//! Provides useful methods for RGB image manipulation.
//! In the future, this may be generalized to RGBA, but this is not necessary for current projects.
//! In the future, interop with Rust NDArray and Python numpy (via pyO3) may be added.

use crate::types::rgb_color::{RGBColor, BLACK};
use crate::error_type::ErrorType;
use crate::error_types::IndexError;
use crate::error_types::JSONDecodeError;

use serde_json;
use serde::{Serialize, Deserialize};

// Structs that end in 4Serde are "minified" and designed specifically for serial

#[derive(Debug,Serialize,Deserialize)]
pub struct RGBImage {
    pub w: i32,
    pub h: i32,
    pub data: Vec<RGBColor>
}

impl RGBImage {
    //! Constructors and useful methods for RGBImage
    //! For this struct, there will be no constructor named new, as it is too general

    fn blank(w: i32, h: i32, background_color:Option<RGBColor>) -> RGBImage {
        let background_color = background_color.unwrap_or(BLACK);
        let mut data: Vec<RGBColor> = Vec::new();
        for x in 0..w {
            for y in 0..h {
                data.push(background_color);
            }
        }
        RGBImage {
            w,
            h,
            data
        }
    }

    fn is_in_bounds(&self, x:i32, y:i32) -> bool {
        x >= 0 && x < self.w && y >= 0 && y < self.h
    }

    fn write_pixel(&mut self, x:i32, y:i32, color: RGBColor) -> Result<(),ErrorType> {
        if !self.is_in_bounds(x,y) {
            return Err(IndexError::new(Some(format!("Index (x,y) = ({},{}) was out of bounds for image with dimensions (w,,h) = ({},{}).",x,y,self.w,self.h))));
        }
        *self.data.get_mut((y*self.w+x) as usize).unwrap() = color;
        Ok(())
    }

    fn read_pixel(&self, x:i32, y:i32) -> Result<RGBColor,ErrorType> {
        if !self.is_in_bounds(x,y) {
            return Err(IndexError::new(Some(format!("Index (x,y) = ({},{}) was out of bounds for image with dimensions (w,,h) = ({},{}).",x,y,self.w,self.h))));
        }
        Ok(*self.data.get((y*self.w+x) as usize).unwrap())
    }

}

impl RGBImage {
    //! This impl block houses utility functions related to RGB Images

    pub fn invert_colors(&mut self) -> Result<(),ErrorType> {
        for x in 0..self.w {
            for y in 0..self.h {
                self.write_pixel(x,y,self.read_pixel(x,y).unwrap().inverted()).unwrap();
            }
        }
        Ok(())
    }

}

impl RGBImage {
    // This impl block houses all serde utils related to RGB Images

    pub fn from_json_string(json_string: String) -> Result<RGBImage, ErrorType> {

        // let root: serde_json::Value = (match serde_json::from_str(json_string.as_str()) {
        //     Ok(root) => {
        //         Some(root)
        //     },
        //     Err(serde_json_error) => {
        //         return Err(JSONDecodeError::GenericSerdeJSONError::new(serde_json_error));
        //     }
        // }).unwrap();

        let root: serde_json::Value = serde_json::from_str(&json_string).map_err(JSONDecodeError::GenericSerdeJSONError::new)?;

        let w: i64 = root["w"].as_i64().ok_or(JSONDecodeError::IncorrectTypeError::new("w".to_owned(),"Suppports i64".to_owned()))?;

        let h: i64 = root["h"].as_i64().ok_or(JSONDecodeError::IncorrectTypeError::new("h".to_owned(),"Supports i64".to_owned()))?;

        let w: i32 = w as i32;

        let h: i32 = h as i32;

        //println!("{:?}", root["data"]);

        let data = root["data"].as_array().unwrap().iter().map(
            |v:&serde_json::Value| -> RGBColor {
                let values = v.as_array().unwrap();
                let r =  values[0].as_i64().unwrap();
                let g =  values[1].as_i64().unwrap();
                let b =  values[2].as_i64().unwrap();
                RGBColor::new(r as i32, g as i32, b as i32) 
            }
        ).collect::<Vec<RGBColor>>();
        
        let mut image = RGBImage::blank(w, h, None);
        

        for x in 0..image.w {
            for y in 0..image.h {
                image.write_pixel(x, y, *(data.get((y*image.w+x)as usize).unwrap()))?;
            }
        }

        Ok(image)
    }

    pub fn to_json_string(&self) -> Result<String, ErrorType> {
        Ok(serde_json::to_string(self).map_err(JSONDecodeError::GenericSerdeJSONError::new)?)
    }


}