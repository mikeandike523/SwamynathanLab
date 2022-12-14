use serde::{Serialize, Deserialize};

pub const BLACK: RGBColor = RGBColor { r:0, g:0, b:0};
pub const WHITE: RGBColor = RGBColor { r:0, g:0, b:0};

#[derive(Copy, Clone, Debug,Serialize,Deserialize)]
pub struct RGBColor {
    pub r: i32,
    pub g: i32,
    pub b: i32
}

impl RGBColor{

    pub fn new(r:i32, g:i32, b:i32) -> RGBColor {
        RGBColor {
            r,
            g,
            b
        }
    }

}

impl RGBColor {
    //! this impl block houses utility functions for RGBColor

    pub fn inverted(&self) -> RGBColor {
        RGBColor::new(255-self.r, 255-self.g, 255-self.b)
    }

}