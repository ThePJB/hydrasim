use crate::kapp::*;
use crate::kmath::*;
use crate::renderers::mesh_renderer::MeshBuilder;
use crate::texture_buffer::*;

pub struct Erosion {
    w: usize,
    h: usize,
    hm: Vec<f32>,
    seed: u32,
    frame_num: u32,
    gen: Gen,
}

impl Erosion {
    // I think this tangent method produces only
    pub fn tangent_ij(&self, i: usize, j: usize) -> Vec3 {
        let mut tan = v3(0.0, 0.0, 0.0);
        let mut nacc = 0;
        let h = self.hm[j*self.w + i];
        if i > 0 {
            let h2 = self.hm[j*self.w + i - 1];
            nacc += 1;
            tan = tan + v3(-1.0/self.w as f32, h2 - h, 0.0);
        }
        if j > 0 {
            let h2 = self.hm[(j-1)*self.w + i];
            nacc += 1;
            tan = tan + v3(0.0, h2 - h, -1.0/self.h as f32);
        }
        if i < self.w - 1 {
            let h2 = self.hm[j*self.w + i + 1];
            nacc += 1;
            tan = tan - v3(1.0/self.w as f32, h2 - h, 0.0);
        }
        if j < self.h - 1 {
            let h2 = self.hm[(j+1)*self.w + i];
            nacc += 1;
            tan = tan - v3(0.0, h2 - h, 1.0/self.h as f32);
        }
        tan = tan / nacc as f32;
        tan.normalize()
    }
    pub fn normal_ij(&self, i: usize, j: usize) -> Vec3 {
        let tan = self.tangent_ij(i, j);
        let u = v3(-tan.z, tan.y, tan.x);
        let norm = u.cross(tan).normalize();
        norm
    }
    pub fn tangent_xy(&self, x: f32, y: f32) -> Vec3 {
        let fi = x * self.w as f32;
        let fj = y * self.h as f32;

        let i = fi.floor() as usize;
        let j = fj.floor() as usize;

        let ti = fi.fract();
        let tj = fj.fract();

        self.tangent_ij(i, j).lerp(self.tangent_ij(i+1, j), ti).lerp(self.tangent_ij(i, j+1).lerp(self.tangent_ij(i+1, j+1), ti), tj)
    }
    pub fn remesh(&self, outputs: &mut FrameOutputs) {
        let mut mb = MeshBuilder::default();

        for idx in 0..(self.w * self.h) {
            let i = idx % self.w;
            let j = idx / self.w;
            let x = i as f32 / self.w as f32;
            let y = j as f32 / self.h as f32;

            let h = self.hm[idx];
            
            let c = v4(0.5, 1.0-self.hm[idx], 0.5, 1.0);

            let norm = self.normal_ij(i, j);

            let pos = v3(x, h, y);
            let uv = v2(x, y);
            mb.push_element(pos, uv, norm, c);

        }

        for i in 0..self.w-1 {
            for j in 0..self.h-1 {
                let idx = i*self.w + j;
                mb.push_tri(idx, idx+1, idx+self.w);
                mb.push_tri(idx+1, idx+self.w, idx+self.w+1);
            }
        }
        outputs.set_mesh = Some(mb);
    }
    pub fn frame(&mut self, inputs: &FrameInputs, outputs: &mut FrameOutputs) {
        let fseed = khash(self.frame_num * 123137879 + 213125417);
        self.frame_num += 1;
        if inputs.key_rising(VirtualKeyCode::R) || self.frame_num == 1 {
            self.seed = self.seed * 1231237 + 12312497;
            self.gen = Gen::new(self.seed);
            for idx in 0..(self.w * self.h) {
                let i = idx % self.w;
                let j = idx / self.w;
                let x = i as f32 / self.w as f32;
                let y = j as f32 / self.h as f32;

                let h = self.gen.height(x, y);
                self.hm[idx] = h;
            }
            let mut tb = TextureBuffer::new(self.w, self.h);
            for idx in 0..(self.w*self.h) {
                tb.set((idx % self.w) as i32, (idx / self.w) as i32, v4(1., 1., 1., 1.));
            }
            outputs.set_mesh_texture = Some(tb);
        }
        for i in 0..1000 {
            self.drop(krand(fseed + i * 2143145797 + 1313977), krand(fseed + i * 231919177 + 12311767), 0.0, v2(0., 0.), 0);
        }
        self.remesh(outputs);
        let mt = translation(-0.5, -0.0, -0.5);
        let mr = roty(PI/2.0);
        //let mr = roty(inputs.t / 4.0);
        let mm = mat4_mul(mr, mt);

        let cp = v3(1., 0.5, 0.);
        let ct = v3(0., 0., 0.);
        let cd = ct - cp;

        let v = view(cp, ct);
        let p = proj(1.0, inputs.screen_rect.aspect() as f32, 0.001, 100.0);
        let vp = mat4_mul(p, v);

        outputs.draw_mesh = Some((vp, mm, cp, cd));
    }
    fn drop(&mut self, x: f32, y: f32, sediment: f32, v: Vec2, iters: usize) {
       if x < 0.0 || y < 0.0 || x > 1.0 - 1.0/self.w as f32 || y > 1.0 - 1.0/self.h as f32 || iters > 1000 {
           return;
       }

       let fi = x * self.w as f32;
       let fj = y * self.h as f32;

       let i = fi.floor() as usize;
       let j = fj.floor() as usize;

       if i >= self.w-1 || j >= self.h-1 {
           return;
       }

       let ti = fi.fract();
       let tj = fj.fract();

       // i dont think this normal calculating method is right
       // just do bilinear interpolation

       
       // get dx, dy
       let idx = [j*self.w + i, j*self.w + i + 1, (j+1)*self.w + i, (j+1) * self.w + i + 1];
       let w_avg = lerp(lerp(self.hm[idx[0]], self.hm[idx[1]], ti), lerp(self.hm[idx[2]], self.hm[idx[3]], ti), tj);

       let center = v3(x, w_avg, y);
       let p1 = v3(i as f32/self.w as f32, self.hm[idx[0]], j as f32/self.h as f32);
       let p2 = v3((i+1) as f32/self.w as f32, self.hm[idx[1]], j as f32/self.h as f32);
       let p3 = v3(i as f32/self.w as f32, self.hm[idx[2]], (j+1) as f32/self.h as f32);
       let p4 = v3((i+1) as f32/self.w as f32, self.hm[idx[3]], (j+1) as f32/self.h as f32);
       let tan = p1 + p2 + p3 + p4 - 4.0*center;
       let tan = self.tangent_xy(x, y);
       //dbg!(tan);

       let sediment_capacity = (-tan.y * 0.1).max(0.0);
       let delta_sed = sediment_capacity - sediment;

       let dissolve_amount = delta_sed * 0.001;

       let weight = [(ti*ti+tj*tj).sqrt(), ((1.0-ti)*(1.0-ti)+tj*tj).sqrt(), (ti*ti+(1.0-tj)*(1.0-tj)).sqrt(), ((1.0-ti)*(1.0-ti)+(1.0-tj)*(1.0-tj)).sqrt()];

       self.hm[idx[0]] -= weight[0]*dissolve_amount;
       self.hm[idx[1]] -= weight[1]*dissolve_amount;
       self.hm[idx[2]] -= weight[2]*dissolve_amount;
       self.hm[idx[3]] -= weight[3]*dissolve_amount;

       let dir = v2(tan.x, tan.z).normalize().lerp(v, 0.5);
       let dir = v2(tan.x, tan.z).normalize();

       let next_pos = v2(x, y) + dir * 1.0 / self.w as f32;
       let next_v = 0.9*v - tan.y * v2(tan.x, tan.z);

       if tan.y > 0.0 { return; }
       self.drop(next_pos.x, next_pos.y, sediment + dissolve_amount, next_v, iters + 1);
    }
}

impl Default for Erosion {
    fn default() -> Self {
        Erosion {
            w: 64,
            h: 64, 
            hm: vec![0.0; 64*64],
            seed: 21341,
            frame_num: 0,
            gen: Gen::new(21341),
        }
    }
}

struct Gen {
    peaks: Vec<(f32, f32, f32)>,
}

impl Gen {
    fn new(seed: u32) -> Self {
        let mut peaks = vec![];
        let n = 10;
        for i in 0..n {
            for j in 0..10 {
                let si = khash(seed + i * 1231237 + j * 51514793);
                let x = i as f32 / n as f32 + (1.0 / n as f32) * krand(si * 1872313 + 1321247);
                let y = j as f32 / n as f32 + (1.0 / n as f32) * krand(si * 1412417 + 9841717);
                let h = (1.0 - y);
                let pmax = h * 0.5;
                let pp = h;
                if chance(si * 1231399 + 12391289, h) {
                    peaks.push((x, y, krand(si * 1231237 + 12939717) * pmax));
                }
            }
        }
        Gen {
            peaks,
        }
    }
    fn height(&self, x: f32, y: f32) -> f32 {
        let h = (1.0 - y);

        let mut max = 0.0;
        for &(px, py, h) in self.peaks.iter() {
            let slope = 1.0;
            let d = ((x - px) * (x - px) + (y - py) * (y - py)).sqrt();
            let influence = h-(slope * d);
            max = influence.max(max);
        }

        (h/5.0).max(max)
    }
    fn normal(&self, x: f32, y: f32) -> Vec3 {
        let d = 0.001;
        let h = self.height(x, y);
        let h1 = self.height(x + d, y);
        let h2 = self.height(x, y + d);

        let p = v3(x, h, y);
        let p1 = v3(x+d, h1, y);
        let p2 = v3(x, h2, y+d);

        (p1 - p).cross(p2 - p).normalize()
    }
}
