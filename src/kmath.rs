pub use std::f32::consts::PI;
pub static DEG_TO_RAD: f32 = 180.0 / PI;

/***************************************************
 * Easing
 ***************************************************/
pub fn lerp(x1: f32, x2: f32, t: f32) -> f32 {
    x1 * (1.0 - t) + x2 * t
}

pub fn unlerp(x: f32, t1: f32, t2: f32) -> f32 {
    (x - t1) / (t2 - t1)
}

pub fn remap(x: f32, from_low: f32, from_high: f32, to_low: f32, to_high: f32) -> f32 {
    lerp(to_low, to_high, unlerp(x, from_low, from_high))
}

pub fn cubic_bezier(start: Vec2, c1: Vec2, c2: Vec2, end: Vec2, t: f32) -> Vec2 {
    start.lerp(c1.lerp(c2.lerp(end, t), t), t)
}

// t 0..1
pub fn smoothstep(t: f32) -> f32 {
    t * t * (3. - 2. * t)
}

/***************************************************
 * RNG
 ***************************************************/

 pub fn floorfrac(x: f32) -> (f32, f32) {
    let floor = x.floor();
    if x < 0.0 {
        (floor, (floor - x).abs())
    } else {
        (floor, x - floor)
    }
}

pub fn khash(mut state: u32) -> u32 {
    state = (state ^ 2747636419).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state
}

pub fn khash2i(x: i32, y: i32, seed: u32) -> u32 {
    khash((x as u32).wrapping_mul(123176957).wrapping_add((y as u32).wrapping_mul(489172373)).wrapping_add(seed))
}

pub fn krand(seed: u32) -> f32 {
    khash(seed) as f32 / 4294967295.0
}

pub fn kuniform(seed: u32, min: f32, max: f32) -> f32 {
    min + (khash(seed) as f32 / 4294967295.0) * (max - min)
}

pub fn chance(seed: u32, percent: f32) -> bool {
    krand(seed) < percent
}

pub fn noise1d(t: f32, seed: u32) -> f32 {
    let hstart = kuniform(seed + 489172373 * t.floor() as u32, 0.0, 1.0);
    let hend = kuniform(seed + 489172373 * (t.floor() + 1.0) as u32, 0.0, 1.0);
    lerp(hstart, hend, smoothstep(t.fract()))
}

pub fn noise2d(x: f32, y: f32, seed: u32) -> f32 {
    let (xfloor, xfrac) = floorfrac(x);
    let (yfloor, yfrac) = floorfrac(y);

    let x0 = xfloor as i32;
    let x1 = x0 + 1;
    let y0 = yfloor as i32;
    let y1 = y0 + 1;

    let s00 = khash2i(x0, y0, seed);
    let s10 = khash2i(x1, y0, seed);
    let s01 = khash2i(x0, y1, seed);
    let s11 = khash2i(x1, y1, seed);

    let h00 = krand(s00);
    let h10 = krand(s10);
    let h01 = krand(s01);
    let h11 = krand(s11);

    let ptop = lerp(h00, h10, smoothstep(xfrac));
    let pbot = lerp(h01, h11, smoothstep(xfrac));

    lerp(ptop, pbot, smoothstep(yfrac))
}

/***************************************************
 * Vec
 ***************************************************/
 pub fn v2(x: f32, y: f32) -> Vec2 {Vec2{x,y}}
 pub fn v3(x: f32, y: f32, z: f32) -> Vec3 {Vec3{x,y,z}}
 pub fn v4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {Vec4{x,y,z,w}}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Vec2 { Vec2{x, y} }
    pub const fn zero() -> Vec2 { Vec2{x: 0.0, y: 0.0} }
    pub fn new_r_theta(r: f32, theta: f32) -> Vec2 { Vec2{x: r * theta.cos(), y: r * theta.sin()} }
    pub fn mul_scalar(&self, scalar: f32) -> Vec2 { Vec2::new(self.x * scalar, self.y * scalar) }
    pub fn div_scalar(&self, scalar: f32) -> Vec2 { Vec2::new(self.x / scalar, self.y / scalar) }
    pub fn magnitude(&self) -> f32 { (self.x*self.x + self.y*self.y).sqrt() }
    pub fn dist(&self, other: Vec2) -> f32 { (*self - other).magnitude() }
    pub fn normalize(&self) -> Vec2 { let m = self.magnitude(); if m == 0.0 { *self } else { self.div_scalar(self.magnitude()) }}
    pub fn lerp(&self, other: Vec2, t: f32) -> Vec2 { Vec2::new(self.x*(1.0-t) + other.x*(t), self.y*(1.0-t) + other.y*(t)) }
    pub fn rotate(&self, radians: f32) -> Vec2 { 
        Vec2::new(
            self.x * radians.cos() - self.y * radians.sin(), 
            self.x * radians.sin() + self.y * radians.cos()
        ) 
    }
    pub fn offset_r_theta(&self, r: f32, theta: f32) -> Vec2 {
        *self + Vec2::new(r, 0.0).rotate(theta)
    }
    pub fn promote(&self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }
    pub fn transform(&self, from: Rect, to: Rect) -> Vec2 {
        // maintains its relative position
        Vec2::new(
            ((self.x - from.x) / from.w) * to.w + to.x,
            ((self.y - from.y) / from.h) * to.h + to.y,
        )
    }
    pub fn complex_mul(&self, other: Vec2) -> Vec2 {
        let a = self.x;
        let b = self.y;
        let c = other.x;
        let d = other.y;
        Vec2::new(a*c - b*d, a*d + c*b)
    }
    pub fn rect_centered(&self, w: f32, h: f32) -> Rect {
        Rect::new(self.x - w/2.0, self.y - h/2.0, w, h)
    }
    
}

impl std::ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, _rhs: Vec2) -> Vec2 {
        Vec2 { x: self.x - _rhs.x, y: self.y - _rhs.y }
    }
}

impl std::ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, _rhs: Vec2) -> Vec2 {
        Vec2 { x: self.x + _rhs.x, y: self.y + _rhs.y }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, _rhs: f32) -> Vec2 {
        self.mul_scalar(_rhs)
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, _rhs: Vec2) -> Vec2 {
        _rhs.mul_scalar(self)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, _rhs: f32) -> Vec2 {
        self.div_scalar(_rhs)
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        self.mul_scalar(-1.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 { Vec3{x, y, z} }
    pub fn mul_scalar(&self, scalar: f32) -> Vec3 { Vec3::new(self.x * scalar, self.y * scalar, self.z * scalar) }
    pub fn div_scalar(&self, scalar: f32) -> Vec3 { Vec3::new(self.x / scalar, self.y / scalar, self.z / scalar) }
    pub fn magnitude(&self) -> f32 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    pub fn square_distance(&self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z }
    pub fn normalize(&self) -> Vec3 { self.div_scalar(self.magnitude()) }
    pub fn lerp(&self, other: Vec3, t: f32) -> Vec3 { Vec3::new(self.x*(1.0-t) + other.x*(t), self.y*(1.0-t) + other.y*(t), self.z*(1.0-t) + other.z*(t)) }
    pub fn dist(&self, other: Vec3) -> f32 {(*self - other).magnitude().sqrt()}
    pub fn dot(&self, other: Vec3) -> f32 {self.x*other.x + self.y*other.y + self.z*other.z} // is squ dist lol
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x,
        )
    }
    pub fn rotate_about_vec3(&self, axis: Vec3, theta: f32) -> Vec3 {
        *self*theta.cos() + (axis.cross(*self)*theta.sin()) + axis * (axis.dot(*self)*(1.0 - theta.cos()))
    }
    pub fn promote(&self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }
}

impl std::ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, _rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x - _rhs.x, y: self.y - _rhs.y, z: self.z - _rhs.z }
    }
}

impl std::ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, _rhs: Vec3) -> Vec3 {
        Vec3 { x: self.x + _rhs.x, y: self.y + _rhs.y, z: self.z + _rhs.z}
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, _rhs: f32) -> Vec3 {
        self.mul_scalar(_rhs)
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, _rhs: Vec3) -> Vec3 {
        _rhs.mul_scalar(self)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, _rhs: f32) -> Vec3 {
        self.div_scalar(_rhs)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        self.mul_scalar(-1.0)
    }
}

impl std::ops::AddAssign for Vec3 {

    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let decimals = f.precision().unwrap_or(2);
        let string = format!("[{:.*}, {:.*}, {:.*}]", decimals, self.x, decimals, self.y, decimals, self.z);
        f.pad_integral(true, "", &string)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4{x, y, z, w} }
    pub fn mul_scalar(&self, scalar: f32) -> Vec4 { Vec4::new(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar) }
    pub fn div_scalar(&self, scalar: f32) -> Vec4 { Vec4::new(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar) }
    pub fn magnitude(&self) -> f32 { (self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w).sqrt() }
    pub fn square_distance(&self) -> f32 { self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w }
    pub fn normalize(&self) -> Vec4 { self.div_scalar(self.magnitude()) }
    pub fn lerp(&self, other: Vec4, t: f32) -> Vec4 { Vec4::new(self.x*(1.0-t) + other.x*(t), self.y*(1.0-t) + other.y*(t), self.z*(1.0-t) + other.z*(t), self.w*(1.0-t) + other.w*t) }
    pub fn dist(&self, other: Vec4) -> f32 {(*self - other).magnitude().sqrt()}
    pub fn dot(&self, other: Vec4) -> f32 {self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w} // is squ dist lol
    pub fn hsv_to_rgb(&self) -> Vec4 {
        let v = self.z;
        let hh = (self.x % 360.0) / 60.0;
        let i = hh.floor() as i32;
        let ff = hh - i as f32;
        let p = self.z * (1.0 - self.y);
        let q = self.z * (1.0 - self.y * ff);
        let t = self.z * (1.0 - self.y * (1.0 - ff));
        match i {
            0 => Vec4::new(v, t, p, self.w),
            1 => Vec4::new(q, v, p, self.w),
            2 => Vec4::new(p, v, t, self.w),
            3 => Vec4::new(p, q, v, self.w),
            4 => Vec4::new(t, p, v, self.w),
            5 => Vec4::new(v, p, q, self.w),
            _ => panic!("unreachable"),
        }
    }

    // h in degrees
    pub fn from_hsv(h: f32, s: f32, v: f32, w: f32) -> Vec4 {
        Vec4::new(h, s, v, w).hsv_to_rgb()
    }

    pub fn grey(v: f32) -> Vec4 {
        Vec4::new(v, v, v, 1.0)
    }
}

impl std::ops::Sub<Vec4> for Vec4 {
    type Output = Vec4;

    fn sub(self, _rhs: Vec4) -> Vec4 {
        Vec4 { x: self.x - _rhs.x, y: self.y - _rhs.y, z: self.z - _rhs.z, w: self.w - _rhs.w}
    }
}

impl std::ops::Add<Vec4> for Vec4 {
    type Output = Vec4;

    fn add(self, _rhs: Vec4) -> Vec4 {
        Vec4 { x: self.x + _rhs.x, y: self.y + _rhs.y, z: self.z + _rhs.z, w: self.w + _rhs.w}
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Vec4;

    fn mul(self, _rhs: f32) -> Vec4 {
        self.mul_scalar(_rhs)
    }
}

impl std::ops::Mul<Vec4> for f32 {
    type Output = Vec4;

    fn mul(self, _rhs: Vec4) -> Vec4 {
        _rhs.mul_scalar(self)
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Vec4;

    fn div(self, _rhs: f32) -> Vec4 {
        self.div_scalar(_rhs)
    }
}

impl std::ops::Neg for Vec4 {
    type Output = Vec4;

    fn neg(self) -> Vec4 {
        self.mul_scalar(-1.0)
    }
}

impl std::ops::AddAssign for Vec4 {

    fn add_assign(&mut self, rhs: Vec4) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::fmt::Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let decimals = f.precision().unwrap_or(2);
        let string = format!("[{:.*}, {:.*}, {:.*}]", decimals, self.x, decimals, self.y, decimals, self.z);
        f.pad_integral(true, "", &string)
    }
}


/***************************************************
 * Shapes
 ***************************************************/

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect{x,y,w,h}
    }
    pub fn centered(p: Vec2, w: f32, h: f32) -> Rect {
        Rect::new(p.x - w/2.0, p.y - h/2.0, w, h)
    }
    pub fn child(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(
            self.x + x*self.w,
            self.y + y*self.h,
            self.w * w,
            self.h * h,
        )
    }
    pub fn grid_child(&self, x: i32, y: i32, w: i32, h: i32) -> Rect {
        let r_w = self.w / w as f32;
        let r_h = self.h / h as f32;

        Rect::new(
            self.x + r_w * x as f32,
            self.y + r_h * y as f32,
            r_w,
            r_h,
        )
    }
    pub fn fit_center_square(&self) -> Rect {
        let s = self.w.min(self.h);
        Rect::new_centered(self.x + self.w / 2.0, self.y + self.h / 2.0, s, s)
    }
    pub fn fit_aspect_ratio(&self, a: f32) -> Rect {
        let our_a = self.w / self.h;
        if our_a < a {
            // big a means wide
            // they want wider
            let other_h = our_a / a * self.h;
            let other_y = self.y + (self.h - other_h) / 2.0;
            Rect::new(self.x, other_y, self.w, other_h)
        } else {
            // they want taller
            let other_w = a / our_a * self.w;
            let other_x = self.x + (self.w - other_w) / 2.0;
            Rect::new(other_x, self.y, other_w, self.h)
        }
    }
    pub fn fill_aspect_ratio(&self, a: f32) -> Rect {
        let our_a = self.w / self.h;
        if our_a < a {
            //wider
            Rect::centered(self.centroid(), self.w * (a / our_a), self.h)
        } else {
            //taller
            Rect::centered(self.centroid(), self.w, self.h * (our_a / a))
        }
    } 
    pub fn lerp(&self, other: Rect, t: f32) -> Rect {
        Rect::new(
            lerp(self.x, other.x, t),
            lerp(self.y, other.y, t),
            lerp(self.w, other.w, t),
            lerp(self.h, other.h, t),
        )
    }
    pub fn aspect(&self) -> f32 {
        self.w / self.h
    }
    pub fn centroid(&self) -> Vec2 {
        Vec2::new(self.x + self.w/2.0, self.y + self.h/2.0)
    }
    pub fn new_centered(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(x-w/2.0, y-h/2.0, w, h)
    }
    pub fn translate(&self, v: Vec2) -> Rect {
        return Rect::new(self.x + v.x, self.y + v.y, self.w, self.h);
    }
    pub fn dilate(&self, d: f32) -> Rect {
        return Rect::new(self.x - d, self.y - d, self.w + 2.0*d, self.h + 2.0*d);
    }
    pub fn dilate_pc(&self, percent: f32) -> Rect {
        let amount = self.w.min(self.h) * percent;
        self.dilate(amount)
    }
    pub fn left(self) -> f32 {
        self.x
    }
    pub fn right(self) -> f32 {
        self.x + self.w
    }
    pub fn top(self) -> f32 {
        self.y
    }
    pub fn bot(self) -> f32 {
        self.y + self.h
    }
    pub fn tl(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
    pub fn tr(self) -> Vec2 {
        Vec2::new(self.x + self.w, self.y)
    }
    pub fn bl(self) -> Vec2 {
        Vec2::new(self.x, self.y + self.h)
    }
    pub fn br(self) -> Vec2 {
        Vec2::new(self.x + self.w, self.y + self.h)
    }
    pub fn contains(self, point: Vec2) -> bool {
        self.x < point.x && self.x + self.w > point.x &&
        self.y < point.y && self.y + self.h > point.y
    }
    pub fn relative_point(self, point: Vec2) -> Vec2 {
        Vec2::new((point.x - self.x) / self.w, (point.y - self.y) / self.h)
    }
    pub fn grid_square(self, point: Vec2, w: i32, h: i32) -> (i32, i32) {
        ((w as f32 * point.x) as i32, (h as f32 * point.y) as i32)
    }
    pub fn snap(&self, mut point: Vec2) -> Vec2 {
        if point.x < self.x {
            point.x = self.x;
        }
        if point.x > self.right() {
            point.x = self.right();
        }
        if point.y < self.y {
            point.y = self.y;
        }
        if point.y > self.bot() {
            point.y = self.bot();
        }
        point
    }
    pub fn tri_child(&self, which: usize) -> Triangle {
        match which {
            0 => Triangle::new(self.tl(), self.tr(), self.centroid()),
            1 => Triangle::new(self.tr(), self.br(), self.centroid()),
            2 => Triangle::new(self.br(), self.bl(), self.centroid()),
            3 => Triangle::new(self.bl(), self.tl(), self.centroid()),
            _ => panic!("bad triangle number"),
        }
    }

    // 5 cases: both a in b, both b in a, a left in b, b left in a, no overlap
    fn overlap_amount(a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
        let a1_in_b = a1 >= b1 && a1 <= b2;
        let a2_in_b = a2 >= b1 && a2 <= b2;
        let b1_in_a = b1 >= a1 && b1 <= a2;
        let b2_in_a = b2 >= a1 && b2 <= a2;

        if !a1_in_b && !a2_in_b && !b1_in_a && !b2_in_a {return 0.0;} // no overlap
        if a1_in_b && a2_in_b {return a2 - a1;} // a fully within b // maybe better to do distance to outside still
        if b1_in_a && b2_in_a {return b2 - b1;} // b fully within a
        if a1_in_b {return b2 - a1;} // a to right of b
        if b1_in_a {return -(a2 - b1);} // b to right of a
        panic!("unreachable overlap");
    }

    // if theres a collision return axis and amount of least penetration
    pub fn overlaps(&self, b: Rect) -> Option<Vec2> {
        let x_overlap = Rect::overlap_amount(self.left(), self.right(), b.left(), b.right());
        let y_overlap = Rect::overlap_amount(self.top(), self.bot(), b.top(), b.bot());

        if x_overlap == 0.0 || y_overlap == 0.0 {return None};

        if x_overlap.abs() < y_overlap.abs() {
            return Some(Vec2::new(x_overlap, 0.0));
        } 
        return Some(Vec2::new(0.0, y_overlap));
    }

    pub fn transform(&self, from: Rect, to: Rect) -> Rect {
        // maintains its relative position
        Rect::new(
            ((self.x - from.x) / from.w) * to.w + to.x,
            ((self.y - from.y) / from.h) * to.h + to.y,
            self.w / from.w * to.w,
            self.h / from.h * to.h,
        )
    }

    pub fn split_ud(&self, t: f32) -> (Rect, Rect) {
        (self.child(0.0, 0.0, 1.0, t), self.child(0.0, t, 1.0, 1.0 - t))
    }
    pub fn split_lr(&self, t: f32) -> (Rect, Rect) {
        (self.child(0.0, 0.0, t, 1.0), self.child(t, 0.0, 1.0 - t, 1.0))
    }

    pub fn split_lrn(&self, n: i32) -> Vec<Rect> {
        (0..n).map(|i| self.grid_child(i, 0, n, 1)).collect()
    }
}

pub struct Triangle {
    pub a: Vec2,
    pub b: Vec2,
    pub c: Vec2,
}

impl Triangle {
    pub fn new(a: Vec2, b: Vec2, c: Vec2) -> Triangle {
        Triangle {a, b, c}
    }

    pub fn dilate(&self, d: f32) -> Triangle {
        let centroid = Vec2::new((self.a.x + self.b.x + self.c.x) / 3.0, (self.a.y + self.b.y + self.c.y) / 3.0);
        Triangle::new(
            self.a + (self.a - centroid) * d,
            self.b + (self.b - centroid) * d,
            self.c + (self.c - centroid) * d,
        )
    }

    pub fn contains(&self, p: Vec2) -> bool {
        let denom = self.a.x * (self.b.y - self.c.y) + self.a.y * (self.c.x - self.b.x) + self.b.x*self.c.y - self.b.y * self.c.x;
        let t1 = (p.x * (self.c.y - self.a.y) + p.y * (self.a.x - self.c.x) - self.a.x*self.c.y + self.a.y*self.c.x) / denom;
        let t2 = (p.x * (self.b.y - self.a.y) + p.y * (self.a.x - self.b.x) - self.a.x*self.b.y + self.a.y*self.b.x) / -denom;
        let s = t1 + t2;
 
         return 0.0 <= t1 && t1 <= 1.0 && 0.0 <= t2 && t2 <= 1.0 && s <= 1.0;
    }

    pub fn aabb(&self) -> Rect {
        let min_x = self.a.x.min(self.b.x.min(self.c.x));
        let min_y = self.a.y.min(self.b.y.min(self.c.y));
        let max_x = self.a.x.max(self.b.x.max(self.c.x));
        let max_y = self.a.y.max(self.b.y.max(self.c.y));
        Rect { x: min_x, y: min_y, w: max_x - min_x, h: max_y - min_y }

    }
}

pub fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[4* j + i] = Vec4::new(a[4*j + 0] as f32, a[4*j + 1] as f32, a[4*j + 2] as f32, a[4*j + 3] as f32).dot(Vec4::new(b[4*0 + i] as f32, b[4*1 + i] as f32, b[4*2 + i] as f32, b[4*3 + i] as f32)) as f32;
        }
    }
    r
}

pub fn view(pos: Vec3, target: Vec3) -> [f32; 16] {
    let up = Vec3::new(0.0, 1.0, 0.0);
    let f = (target - pos).normalize();
    let r = f.cross(up);
    let vup = up - up.dot(f) * f;

    let vr = [
        r.x as f32, r.y as f32, r.z as f32, 0.,
        vup.x as f32, vup.y as f32, vup.z as f32, 0.,
        -f.x as f32, -f.y as f32, -f.z as f32, 0.,
        0., 0., 0., 1.,
    ];

    let vt = [
        1., 0., 0., -pos.x as f32,
        0., 1., 0., -pos.y as f32,
        0., 0., 1., -pos.z as f32,
        0., 0., 0., 1.,
    ];

    mat4_mul(vr, vt)
}

pub fn roty(theta: f32) -> [f32; 16] {
    [
        theta.cos(), 0., theta.sin(), 0.,
        0., 1., 0., 0.,
        -theta.sin(), 0., theta.cos(), 0.,
        0., 0., 0., 1.,
    ]
}

pub fn translation(x: f32, y: f32, z: f32) -> [f32; 16] {
    [
        1., 0., 0., x,
        0., 1., 0., y,
        0., 0., 1., z,
        0., 0., 0., 1.,
    ]
}

pub fn mat4_trans(a: [f32; 16]) -> [f32; 16] {
    [
        a[0], a[4], a[8], a[12],
        a[1], a[5], a[9], a[13],
        a[2], a[6], a[10], a[14],
        a[3], a[7], a[11], a[15],
    ]
}

pub fn proj(fov: f32, a: f32, n: f32, f: f32) -> [f32; 16] {
    let d = 1.0 / (fov/2.0).tan();
    [
        d/a, 0., 0., 0.,
        0., d, 0., 0.,
        0., 0., (n+f)/(n-f), 2.0*(f*n)/(n-f),
        0., 0., -1., 0.,
    ]
}

pub fn print_mat(a: [f32; 16]) {
    for i in 0..16 {
        if i == 0 || i == 4 || i == 8 || i == 12 {
            print!{"\t"};
        }
        if i == 3 || i == 7 || i == 11 || i == 15 {
            print!("{:>8.4}\n", a[i]);
        } else {
            print!("{:>8.4} ", a[i]);
        }
    }
}


// make a lovely test that shows it all super clear

#[test]
fn test_view() {
    let pos = Vec3::new(0.0, 0.0, 0.0);
    let target = Vec3::new(0.0, 0.0, -1.0);
    let view = view(pos, target);

    println!("pos: {}", pos);
    println!("target: {}", target);
    println!("\nview (should be identity):");
    print_mat(view);
    println!("");

}

#[test]
fn test_pv() {
    let pos = Vec3::new(0.0, 0.5, 0.0);
    let target = Vec3::new(0.5, 0.0, 0.5);
    let view = view(pos, target);

    println!("pos: {}", pos);
    println!("target: {}", target);
    println!("\nview:");
    print_mat(view);
    println!("");

    let p = [target.x as f32, target.y as f32, target.z as f32, 1.0];
    let p = mat4_mul_v4(view, p);
    println!("view x target: {:?} (should be in -z)", p);

    let n = 0.01;
    let f = 100.0;
    let proj = proj(1.0, 1.0, n, f);
    println!("\nproj:");
    print_mat(proj);
    println!("");

    println!("proj x view x target {:?} (should be between -1 and 1)", mat4_mul_v4(proj, p));
    println!("proj x thing at -near {:?} (should be -1)", mat4_mul_v4(proj, [0., 0., -n, 1.]));
    println!("proj x thing at -far {:?} (should be 1)", mat4_mul_v4(proj, [0., 0., -f, 1.]));
}

pub fn mat4_mul_v4(a: [f32; 16], b: [f32; 4]) -> [f32; 4] {
    [
        b[0]*a[0] + b[1]*a[1] + b[2]*a[2] + b[3]*a[3],
        b[0]*a[4] + b[1]*a[5] + b[2]*a[6] + b[3]*a[7],
        b[0]*a[8] + b[1]*a[9] + b[2]*a[10] + b[3]*a[11],
        b[0]*a[12] + b[1]*a[13] + b[2]*a[14] + b[3]*a[15],
    ]
}
