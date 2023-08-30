use crate::kmath::*;
use crate::texture_buffer::*;
use glow::*;

#[derive(Default)]
pub struct MeshBuilder {
    // elements: Vec<(Vec3, Vec2, Vec3, Vec4)>,
    elements: Vec<u8>,
    // tris: Vec<(usize, usize, usize)>,
    indexes: Vec<u8>,
    n_element: usize,
}

impl MeshBuilder {
    fn put_float(&mut self, x: f32) {
        for b in (x as f32).to_le_bytes() {
            self.elements.push(b);
        }
    }
    fn put_index(&mut self, i: usize) {
        self.indexes.push(i as u8);
        self.indexes.push((i >> 8) as u8);
        self.indexes.push((i >> 16) as u8);
        self.indexes.push((i >> 24) as u8);
    }

    pub fn push_element(&mut self, pos: Vec3, uv: Vec2, normal: Vec3, colour: Vec4) -> usize {
        self.put_float(pos.x);
        self.put_float(pos.y);
        self.put_float(pos.z);
        self.put_float(uv.x);
        self.put_float(uv.y);
        self.put_float(normal.x);
        self.put_float(normal.y);
        self.put_float(normal.z);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);

        let n = self.n_element;
        self.n_element += 1;
        n
    }

    pub fn push_tri(&mut self, a: usize, b: usize, c: usize) {
        self.put_index(a);
        self.put_index(b);
        self.put_index(c);
    }
}

pub struct MeshRenderer {
    vbo: NativeBuffer,
    vao: NativeVertexArray,
    ebo: NativeBuffer,
    program: NativeProgram,
    texture: NativeTexture,
    ntris: i32,
}

impl MeshRenderer {
    pub fn new(gl: &glow::Context) -> MeshRenderer {
        unsafe {
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            let ebo = gl.create_buffer().unwrap();
            
            // position uv normal colour
            let stride = 4*3 + 4*2 + 4*3 + 4*4;
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 4*3);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(2, 3, glow::FLOAT, false, stride, 4*3 + 4*2);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(3, 4, glow::FLOAT, false, stride, 4*3 + 4*2 + 4*3);
            gl.enable_vertex_attrib_array(3);

            // Shader
            let program = gl.create_program().expect("Cannot create program");
            let vertex_src = r#"
                #version 330 core
                layout (location = 0) in vec3 in_pos;
                layout (location = 1) in vec2 in_uv;
                layout (location = 2) in vec3 in_normal;
                layout (location = 3) in vec4 in_colour;

                uniform mat4 pv;
                uniform mat4 model;
                uniform vec3 cam_pos;
                uniform vec3 cam_dir;

                const vec3 sun_dir = normalize(vec3(0.2, 0.2, 0.3));


                out vec4 vert_colour;
                out vec2 vert_uv;

                void main() {
                    // float facing_ratio = dot(in_normal, sun_dir) / 2.0 + 0.5;
                    float facing_ratio = dot(in_normal, sun_dir);

                    float brightness = facing_ratio * 1.0;
                    //float brightness = 1.0;

                    vert_colour = vec4(in_colour.xyz * brightness, in_colour.w);
                    //vert_colour = abs(vec4(in_normal.xyz, in_colour.w));
                    vert_uv = in_uv;
                    gl_Position = pv * model * vec4(in_pos, 1.0);
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
                in vec2 vert_uv;
                in vec4 vert_colour;
                
                out vec4 frag_colour;

                uniform sampler2D tex;
                
                void main() {
                    // frag_colour = vert_colour;
                    frag_colour = texture(tex, vert_uv) * vert_colour;
                    //frag_colour = vec4(1.0, 1.0, 1.0, 1.0);
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

            let texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            
            MeshRenderer {
                vbo,
                vao,
                ebo,
                program,
                texture,
                ntris: 0,
            }
        }
    }

    pub fn update_mesh(&mut self, gl: &glow::Context, mb: &MeshBuilder) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &mb.elements, glow::STATIC_DRAW);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &mb.indexes, glow::STATIC_DRAW);
        }
        self.ntris = (mb.indexes.len() / 6) as i32;
    }

    pub fn update_texture(&self, gl: &glow::Context, buf: &TextureBuffer) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32, 
                buf.w as i32, 
                buf.h as i32, 0, RGBA, glow::UNSIGNED_BYTE, 
                Some(&buf.buf));
        }
    }
        
    pub fn render(&self, gl: &glow::Context, pv: [f32; 16], model: [f32; 16], cp: Vec3, cd: Vec3) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program, "pv").as_ref(), true, &pv);
            gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program, "model").as_ref(), true, &model);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "cam_pos").as_ref(), cp.x as f32, cp.y as f32, cp.z as f32);
            gl.uniform_3_f32(gl.get_uniform_location(self.program, "cam_dir").as_ref(), cd.x as f32, cd.y as f32, cd.z as f32);
            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.ebo));
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            gl.draw_elements(glow::TRIANGLES, self.ntris * 3, glow::UNSIGNED_INT, 0);
        }
    }
}
