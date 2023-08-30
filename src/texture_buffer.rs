use crate::kmath::*;

pub struct TextureBuffer {
    pub buf: Vec<u8>, // RGBARGBARGBA
    pub w: usize,
    pub h: usize,
}

impl TextureBuffer {
    pub fn new(w: usize, h: usize) -> TextureBuffer {
        let mut buf = vec![0; w*h*4];
        for i in 0..w*h {
            buf[i*4 + 3] = 255;
        }

        TextureBuffer {
            buf,
            w,
            h,
        }
    }

    pub fn set(&mut self, x: i32, y: i32, colour: Vec4) {
        // let y = self.h as i32 - y;
        let idx = y * self.w as i32 + x;
        self.buf[(idx * 4) as usize] = (colour.x * 255.0) as u8;
        self.buf[(idx * 4 + 1) as usize] = (colour.y * 255.0) as u8;
        self.buf[(idx * 4 + 2) as usize] = (colour.z * 255.0) as u8;
        self.buf[(idx * 4 + 3) as usize] = (colour.w * 255.0) as u8;
    }
}