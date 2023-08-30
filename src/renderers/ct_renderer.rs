use glow::*;
use crate::kmath::*;
use crate::kimg::*;

// I actually just made a UV renderer
// we can have another system that just collects "char" and position

pub struct CTRenderer {
    vbo: NativeBuffer,
    vao: NativeVertexArray,
    program: NativeProgram,
    texture: NativeTexture,
}

impl CTRenderer {
    pub fn new(gl: &glow::Context, tex_buffer: ImageBufferA) -> CTRenderer {
        unsafe {
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*4 + 4*3 + 4*2, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, 4*4 + 4*3 + 4*2, 4*3);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 4*4 + 4*3 + 4*2, 4*3 + 4*4);
            gl.enable_vertex_attrib_array(2);

            // Shader
            let program = gl.create_program().expect("Cannot create program");
            let vertex_src = r#"
                #version 330 core
                layout (location = 0) in vec3 in_pos;
                layout (location = 1) in vec4 in_colour;
                layout (location = 2) in vec2 in_uv;

                const mat4 projection = mat4(
                    2, 0, 0, 0,
                    0, -2, 0, 0,
                    0, 0, -0.001, 0,
                    -1, 1, 1, 1
                );

                out vec4 vert_colour;
                out vec2 vert_uv;

                void main() {
                    vert_colour = in_colour;
                    vert_uv = in_uv;
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
                in vec2 vert_uv;
                
                out vec4 frag_colour;

                uniform sampler2D tex;
                
                void main() {
                    frag_colour = texture(tex, vert_uv) * vert_colour;
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

            // better make and load the texture
            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32, tex_buffer.w as i32, tex_buffer.h as i32, 0, RGBA, glow::UNSIGNED_BYTE, Some(&tex_buffer.bytes()));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.generate_mipmap(glow::TEXTURE_2D);
            
            CTRenderer {
                vbo,
                vao,
                program,
                texture,
            }
        }
    }

    pub fn render(&self, gl: &glow::Context, canvas: &CTCanvas) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &canvas.buf, glow::DYNAMIC_DRAW);
            let vert_count = canvas.buf.len() / (9*4);
            gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
        }
    }
}

pub struct CTCanvas {
    a: f32,
    buf: Vec<u8>,
}

impl CTCanvas {
    pub fn new(a: f32) -> CTCanvas {
        CTCanvas {
            a,
            buf: Vec::new(),
        }
    }

    fn put_float(&mut self, x: f32) {
        for b in (x as f32).to_le_bytes() {
            self.buf.push(b);
        }
    }

    pub fn put_triangle(&mut self, p1: Vec2, uv1: Vec2, p2: Vec2, uv2: Vec2, p3: Vec2, uv3: Vec2, depth: f32, colour: Vec4) {
        self.put_float(p1.x/self.a);
        self.put_float(p1.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv1.x);
        self.put_float(uv1.y);
        
        self.put_float(p2.x/self.a);
        self.put_float(p2.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv2.x);
        self.put_float(uv2.y);
        
        self.put_float(p3.x/self.a);
        self.put_float(p3.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv3.x);
        self.put_float(uv3.y);
    }

    pub fn put_rect(&mut self, r: Rect, r_uv: Rect, depth: f32, colour: Vec4) {
        self.put_triangle(r.tl(), r_uv.tl(), r.tr(), r_uv.tr(), r.bl(), r_uv.bl(), depth, colour);
        self.put_triangle(r.bl(), r_uv.bl(), r.tr(), r_uv.tr(), r.br(), r_uv.br(), depth, colour);
    }
}