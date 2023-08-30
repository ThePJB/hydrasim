use crate::kmath::*;
use crate::texture_buffer::*;
use glow::*;

const NUM_TEXTURES: usize = 2;
pub struct TextureRenderer {
    vbo: NativeBuffer,
    vao: NativeVertexArray,
    program: NativeProgram,
    textures: [NativeTexture; NUM_TEXTURES],
}

impl TextureRenderer {
    pub fn new(gl: &glow::Context) -> TextureRenderer {
        unsafe {
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*3 + 4*2, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 4*3 + 4*2, 4*3);
            gl.enable_vertex_attrib_array(1);

            // Shader
            let program = gl.create_program().expect("Cannot create program");
            let vertex_src = r#"
                #version 330 core
                layout (location = 0) in vec3 in_pos;
                layout (location = 1) in vec2 in_uv;

                const mat4 projection = mat4(
                    2, 0, 0, 0,
                    0, -2, 0, 0,
                    0, 0, -0.001, 0,
                    -1, 1, 1, 1
                );

                out vec2 vert_uv;

                void main() {
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
                in vec2 vert_uv;
                
                out vec4 frag_colour;

                uniform sampler2D tex;
                
                void main() {
                    frag_colour = texture(tex, vert_uv);
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

            let textures = [gl.create_texture().unwrap(), gl.create_texture().unwrap()];
            for i in 0..NUM_TEXTURES {
                gl.bind_texture(glow::TEXTURE_2D, Some(textures[i]));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            }
            
            TextureRenderer {
                vbo,
                vao,
                program,
                textures,
            }
        }
        }

    pub fn update(&self, gl: &glow::Context, buf: &TextureBuffer, texture: usize) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[texture]));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32, 
                buf.w as i32, 
                buf.h as i32, 0, RGBA, glow::UNSIGNED_BYTE, 
                Some(&buf.buf));
            }
        }
        
        pub fn render(&self, gl: &glow::Context, rect: Rect, a: f32, texture: usize) {
            unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.textures[texture]));
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));

            let mut buf_floats = Vec::new();
            buf_floats.push(rect.tl().x/a);
            buf_floats.push(rect.tl().y);
            buf_floats.push(2.0);
            buf_floats.push(0.0);
            buf_floats.push(1.0);
            
            buf_floats.push(rect.tr().x/a);
            buf_floats.push(rect.tr().y);
            buf_floats.push(2.0);
            buf_floats.push(1.0);
            buf_floats.push(1.0);
            
            buf_floats.push(rect.bl().x/a);
            buf_floats.push(rect.bl().y);
            buf_floats.push(2.0);
            buf_floats.push(0.0);
            buf_floats.push(0.0);
            
            buf_floats.push(rect.tr().x/a);
            buf_floats.push(rect.tr().y);
            buf_floats.push(2.0);
            buf_floats.push(1.0);
            buf_floats.push(1.0);
            
            buf_floats.push(rect.br().x/a);
            buf_floats.push(rect.br().y);
            buf_floats.push(2.0);
            buf_floats.push(1.0);
            buf_floats.push(0.0);
            
            buf_floats.push(rect.bl().x/a);
            buf_floats.push(rect.bl().y);
            buf_floats.push(2.0);
            buf_floats.push(0.0);
            buf_floats.push(0.0);

            let mut buf = Vec::<u8>::new();
            for f in buf_floats {
                for b in (f as f32).to_le_bytes() {
                    buf.push(b);
                }
            }

            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &buf, glow::DYNAMIC_DRAW);
            let vert_count = buf.len() / (5*4);
            gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
        }
    }
}