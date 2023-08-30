use glow::*;
use crate::kmath::*;

pub struct SimpleRenderer {
    vbo: NativeBuffer,
    vao: NativeVertexArray,
    program: NativeProgram,
}

impl SimpleRenderer {
    pub fn new(gl: &glow::Context) -> SimpleRenderer {
        unsafe {
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*4 + 4*3, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, 4*4 + 4*3, 4*3);
            gl.enable_vertex_attrib_array(1);

            // Shader
            let program = gl.create_program().expect("Cannot create program");
            let vertex_src = r#"
                #version 330 core
                layout (location = 0) in vec3 in_pos;
                layout (location = 1) in vec4 in_colour;

                const mat4 projection = mat4(
                    2, 0, 0, 0,
                    0, -2, 0, 0,
                    0, 0, -0.001, 0,
                    -1, 1, 1, 1
                );

                out vec4 vert_colour;

                void main() {
                    vert_colour = in_colour;
                    gl_Position = projection * vec4(in_pos, 1.0);
                }
            "#;
        
            let vs = gl.create_shader(glow::VERTEX_SHADER).expect("cannot create vertex shader");
            gl.shader_source(vs, vertex_src);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                panic!("{}", gl.get_shader_info_log(vs));
            }
            gl.attach_shader(program, vs);
            
            let fragment_src = r#"
                #version 330 core
                in vec4 vert_colour;
                
                out vec4 frag_colour;
                
                void main() {
                    frag_colour = vert_colour;
                }
            "#;
            
            let fs = gl.create_shader(glow::FRAGMENT_SHADER).expect("cannot create fragment shader");
            gl.shader_source(fs, fragment_src);
            gl.compile_shader(fs);
            if !gl.get_shader_compile_status(fs) {
                panic!("{}", gl.get_shader_info_log(fs));
            }
            gl.attach_shader(program, fs);

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("{}", gl.get_program_info_log(program));
            }
            gl.detach_shader(program, fs);
            gl.delete_shader(fs);
            gl.detach_shader(program, vs);
            gl.delete_shader(vs);
            
            SimpleRenderer {
                vbo,
                vao,
                program,
            }
        }

    }

    pub fn render(&self, gl: &glow::Context, canvas: &SimpleCanvas) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &canvas.buf, glow::DYNAMIC_DRAW);
            let vert_count = canvas.buf.len() / (7*4);
            gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
        }
    }
}

pub struct SimpleCanvas {
    a: f32,
    buf: Vec<u8>,
}

impl SimpleCanvas {
    pub fn new(a: f32) -> SimpleCanvas {
        SimpleCanvas {
            a,
            buf: Vec::new(),
        }
    }

    fn put_float(&mut self, x: f32) {
        for b in (x as f32).to_le_bytes() {
            self.buf.push(b);
        }
    }

    pub fn put_triangle(&mut self, p1: Vec2, p2: Vec2, p3: Vec2, depth: f32, colour: Vec4) {
        self.put_float(p1.x/self.a);
        self.put_float(p1.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);

        self.put_float(p2.x/self.a);
        self.put_float(p2.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);

        self.put_float(p3.x/self.a);
        self.put_float(p3.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
    }

    pub fn put_rect(&mut self, r: Rect, depth: f32, colour: Vec4) {
        self.put_triangle(r.tl(), r.tr(), r.bl(), depth, colour);
        self.put_triangle(r.bl(), r.tr(), r.br(), depth, colour);
    }

    pub fn put_quad(&mut self, a: Vec2, b: Vec2, c: Vec2, d: Vec2, depth: f32, colour: Vec4) {
        self.put_triangle(a,b,c, depth, colour);
        self.put_triangle(b,d,c, depth, colour);
    }

    pub fn put_line(&mut self, a: Vec2, b: Vec2, w: f32, depth: f32, colour: Vec4) {
        let v = (b - a).normalize();
        let wv = w/2.0 * Vec2::new(-v.y, v.x);
        self.put_quad(a + wv, b + wv, a - wv, b - wv, depth, colour);
    }

    pub fn put_poly(&mut self, center: Vec2, radius: f32, n_sides: i32, depth: f32, colour: Vec4) {
        for i in 0..n_sides {
            let theta_1 = i as f32 * 2.0 * std::f32::consts::PI / n_sides as f32;
            let theta_2 = (i+1) as f32 * 2.0 * std::f32::consts::PI / n_sides as f32;
            self.put_triangle(center, center.offset_r_theta(radius, theta_1), center.offset_r_theta(radius, theta_2), depth, colour);
        }
    }

    pub fn put_poly_part(&mut self, center: Vec2, radius: f32, n_sides: i32, n_to_draw: i32, depth: f32, colour: Vec4) {
        for i in (n_sides - n_to_draw)..n_sides {
            let theta_1 = i as f32 * 2.0 * std::f32::consts::PI / n_sides as f32;
            let theta_2 = (i+1) as f32 * 2.0 * std::f32::consts::PI / n_sides as f32;
            self.put_triangle(center, center.offset_r_theta(radius, theta_1), center.offset_r_theta(radius, theta_2), depth, colour);
        }
    }

    pub fn put_circle(&mut self, center: Vec2, radius: f32, depth: f32, colour: Vec4) {
        let n_sides = (radius as f32).sqrt() * 200.0;
        self.put_poly(center, radius, (n_sides as i32).max(6), depth, colour);
    }

    pub fn put_semicircle(&mut self, center: Vec2, radius: f32, depth: f32, colour: Vec4) {
        let n_sides = (radius as f32).sqrt() * 200.0;
        let n_sides = (n_sides as i32).max(6);
        self.put_poly_part(center, radius, n_sides, n_sides/2, depth, colour);
    }

    pub fn put_vpill(&mut self, c: Vec2, w: f32, h: f32, depth: f32, colour: Vec4) {
        self.put_rect(c.rect_centered(w, h), depth, colour);
        self.put_circle(c + v2(0.0, -h/2.0), w/2.0, depth, colour);
        self.put_circle(c + v2(0.0, h/2.0), w/2.0, depth, colour);
    }
}