use crate::kmath::*;
use crate::renderers::ct_renderer::*;

pub fn glyph_buffer_to_canvas(buf: &GlyphBuffer, a: f32) -> CTCanvas {
    let clip_fn = |mut c: u8| {
        if c >= 'a' as u8 && c <= 'z' as u8 {
            c -= 'a' as u8 - 'A' as u8;
        }
        if c >= '-' as u8 && c <= '_' as u8 {
            let x = c - '-' as u8;
            let w = '_' as u8 - '-' as u8 + 1; // maybe +1
            Some(Rect::new(0.0, 0.0, 1.0, 1.0).grid_child(x as i32, 0, w as i32, 1))
        } else {
            None
        }
    };

    // aspect should be 14/12

    // old font clip fn
    // let clip_fn = |c: u8| {
    //     let idx = c - ' ' as u8;
    //     let w = '@' as u8 - ' ' as u8;
    //     let x = idx % w;
    //     let y = idx / w;
    //     Rect::new(0.0, 0.0, 1.0, 1.0).grid_child(x as i32, y as i32, w as i32, 3)
    // };

    let mut c = CTCanvas::new(a);
    for g in &buf.buf {
        if let Some(r_uv) = clip_fn(g.0 as u8) {
            c.put_rect(g.1, r_uv, g.2, g.3);
        }
    }
    c
}

pub struct GlyphBuffer {
    pub buf: Vec<(char, Rect, f32, Vec4)>,
}

impl GlyphBuffer {
    pub fn new() -> GlyphBuffer {
        GlyphBuffer { buf: Vec::new() }
    }

    pub fn push_glyph(&mut self, c: char, r: Rect, d: f32, colour: Vec4) {
        self.buf.push((c, r, d, colour));
    }

    pub fn push_str(&mut self, s: &str, mut x: f32, y: f32, w: f32, h: f32, d: f32, colour: Vec4) {
        for c in s.chars() {
            self.push_glyph(c, Rect::new(x, y, w, h), d, colour);
            x += w;
        }
    }

    pub fn push_center_str(&mut self, s: &str, mut x: f32, y: f32, w: f32, h: f32, d: f32, colour: Vec4) {
        let max_w = s.len() as f32 * w;
        x = x - max_w / 2.0;
        for c in s.chars() {
            self.push_glyph(c, Rect::new(x, y, w, h), d, colour);
            x += w;
        }
    }

    // takes a rect: height of rect is char height, and char a
    pub fn pushl(&mut self, r: Rect, s: &str, a: f32, c: Vec4, d: f32) {
        let mut x = r.x;

        let w = r.h * a;
        for ch in s.chars() {
            self.push_glyph(ch, Rect::new(x, r.y, w, r.h), d, c);
            x += w;
        }
    }

    pub fn pushc(&mut self, mut r: Rect, s: &str, a: f32, c: Vec4, d: f32) {
        let w = r.h * a;
        r.x -= (w * s.len() as f32)/2.0;
        r.x += r.w / 2.0;
        self.pushl(r, s, a, c, d);
    }
}