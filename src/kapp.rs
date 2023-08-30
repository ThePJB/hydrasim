use crate::erosion::*;

use crate::renderers::font_rendering::*;
use crate::renderers::simple_renderer::*;
use crate::renderers::mesh_renderer::*;
use crate::texture_buffer::*;
use crate::kmath::*;
use crate::video::*;
use crate::audio::*;

use cpal::Stream;
use cpal::traits::*;
use glutin::dpi::LogicalPosition;
use glutin::window::CursorIcon;
use ringbuf::*;

use std::collections::HashSet;
use std::time::{SystemTime, Instant, Duration};

pub use glutin::event::VirtualKeyCode;
use glutin::event::ElementState;
use glutin::event::WindowEvent;
use glutin::event::WindowEvent::*;
use glutin::event::Event;
use glutin::event_loop::*;
use glutin::event::DeviceEvent;


use glow::HasContext;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum KeyStatus {
    Pressed,
    JustPressed,
    JustReleased,
    Released,
}


#[derive(Clone)]
pub struct FrameInputs {
    pub screen_rect: Rect,
    pub mouse_pos: Vec2,
    pub mouse_delta: Vec2,
    
    pub prev_keys: HashSet<VirtualKeyCode>,
    pub curr_keys: HashSet<VirtualKeyCode>,
    pub repeat_keys: HashSet<VirtualKeyCode>,

    pub lmb: KeyStatus,
    pub rmb: KeyStatus,
    pub mmb: KeyStatus,
    pub scroll_delta: f32,
    pub t: f32,
    pub dt: f32,
    pub frame: u32,
    pub seed: u32,
}

impl FrameInputs {
    pub fn key_held(&self, keycode: VirtualKeyCode) -> bool {
        self.curr_keys.contains(&keycode)
    }
    pub fn key_rising(&self, keycode: VirtualKeyCode) -> bool {
        self.curr_keys.contains(&keycode) && !self.prev_keys.contains(&keycode)
    }
    pub fn key_press_or_repeat(&self, keycode: VirtualKeyCode) -> bool {
        (self.curr_keys.contains(&keycode) && !self.prev_keys.contains(&keycode)) || self.repeat_keys.contains(&keycode)
    }
    pub fn key_falling(&self, keycode: VirtualKeyCode) -> bool {
        !self.curr_keys.contains(&keycode) && self.prev_keys.contains(&keycode)
    }
    pub fn new(a: f32) -> FrameInputs {
        FrameInputs { 
            screen_rect: Rect::new(0.0, 0.0, a, 1.0, ), 
            mouse_pos: Vec2::new(0.0, 0.0), 
            mouse_delta: Vec2::new(0.0, 0.0), 
            scroll_delta: 0.0,
            curr_keys: HashSet::new(),
            prev_keys: HashSet::new(),
            repeat_keys: HashSet::new(),
            lmb: KeyStatus::Released, 
            rmb: KeyStatus::Released, 
            mmb: KeyStatus::Released, 
            t: 0.0,
            dt: 0.0,
            frame: 0,
            seed: SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or(Duration::from_nanos(34123123)).subsec_nanos(),
        }
    }
}

pub struct FrameOutputs {
    pub canvas: SimpleCanvas,
    pub set_texture: Vec<(TextureBuffer, usize)>,
    pub draw_texture: Vec<(Rect, usize)>,
    pub set_mesh: Option<MeshBuilder>,
    pub set_mesh_texture: Option<TextureBuffer>,
    pub draw_mesh: Option<([f32; 16], [f32; 16], Vec3, Vec3)>,
    pub glyphs: GlyphBuffer,
    pub sounds: Vec<AudioCommand>,
    pub plant_cursor: bool,
    pub set_cursor: Option<usize>,  // 0 default, 1 hand
}

impl FrameOutputs {
    pub fn new(a: f32) -> FrameOutputs {
        FrameOutputs {
            glyphs: GlyphBuffer::new(),
            canvas: SimpleCanvas::new(a),
            set_texture: Vec::new(),
            draw_texture: Vec::new(),
            set_mesh: None,
            set_mesh_texture: None,
            draw_mesh: None,
            sounds: Vec::new(),
            plant_cursor: false,
            set_cursor: None,
        }
    }
}

pub struct Application {
    video: Video,
    root_scene: Erosion,

    audio_stream: Stream,
    channel: Producer<AudioCommand>,

    t_last: Instant,
    instant_mouse_pos: Vec2,
    current: FrameInputs,

    old_mouse_pos: LogicalPosition<f32>,
    plant_cursor: bool,
}

impl Application {
    pub fn new(event_loop: &EventLoop<()>) -> Application {
        let xres = 1600;
        let yres = 1600;
    
        let video = Video::new("ksynth2", xres as f32, yres as f32, event_loop);

        let rb = RingBuffer::<AudioCommand>::new(64);
        let (mut prod, mut cons) = rb.split();
        
        let app = Application {
            video,
            root_scene: Erosion::default(),
            t_last: Instant::now(),
            old_mouse_pos: LogicalPosition { x: 0.0, y: 0.0 },
            instant_mouse_pos: Vec2::zero(),
            current: FrameInputs::new(xres as f32 / yres as f32),           
            audio_stream: stream_setup_for(sample_next, cons).expect("no can make stream"),
            channel: prod,
            plant_cursor: false,
        };
        app.audio_stream.play().expect("no can play stream");
        app
    }

    pub fn handle_event(&mut self, event: Event<()>) {
        match event {
            Event::LoopDestroyed => self.exit(),
            Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => self.exit(),
            Event::DeviceEvent {event, .. } => match event {
                DeviceEvent::MouseMotion{delta: (dx, dy)} => {
                    if self.plant_cursor {
                        self.current.mouse_pos.x -= dx as f32;
                        self.current.mouse_pos.y -= dy as f32;
                    }
                },
                _ => {},
            },
            Event::WindowEvent {event, ..} => match event {
                KeyboardInput { 
                    input: glutin::event::KeyboardInput { 
                        virtual_keycode: Some(virtual_code), 
                        state, 
                    ..},
                ..} => {
                    if state == ElementState::Pressed {
                        if self.current.curr_keys.contains(&virtual_code) {
                            self.current.repeat_keys.insert(virtual_code);
                        } else {
                            self.current.curr_keys.insert(virtual_code);
                        }
                    } else {
                        self.current.curr_keys.remove(&virtual_code);
                    }
                },
                MouseInput { button: glutin::event::MouseButton::Left, state, ..} => {
                    if state == ElementState::Pressed {
                        self.current.lmb = KeyStatus::JustPressed;
                    } else {
                        self.current.lmb = KeyStatus::JustReleased;
                    }
                },
                MouseInput { button: glutin::event::MouseButton::Middle, state, ..} => {
                    if state == ElementState::Pressed {
                        self.current.mmb = KeyStatus::JustPressed;
                    } else {
                        self.current.mmb = KeyStatus::JustReleased;
                    }
                },
                MouseInput { button: glutin::event::MouseButton::Right, state, ..} => {
                    if state == ElementState::Pressed {
                        self.current.rmb = KeyStatus::JustPressed;
                    } else {
                        self.current.rmb = KeyStatus::JustReleased;
                    }
                },

                // Scroll
                glutin::event::WindowEvent::MouseWheel { delta, ..} => {
                    match delta {
                        glutin::event::MouseScrollDelta::LineDelta(_, y) => {
                            self.current.scroll_delta = y as f32;
                        },
                        glutin::event::MouseScrollDelta::PixelDelta(p) => {
                            self.current.scroll_delta = p.y as f32;
                        },
                    }
                },


                // Mouse motion
                // maybe we actually need mouse device events or something
                CursorMoved {
                    position: pos,
                    ..
                } => {
                    if self.plant_cursor {
                        self.video.window.window().set_cursor_position(self.old_mouse_pos).unwrap_or(());
                    } else {
                        self.old_mouse_pos = pos.to_logical(self.video.window.window().scale_factor());
                        self.instant_mouse_pos = Vec2::new(pos.x as f32 / self.video.yres, pos.y as f32 / self.video.yres);
                    }
                },

                // Resize
                Resized(physical_size) => {
                    self.video.window.resize(physical_size);
                    self.video.xres = physical_size.width as f32;
                    self.video.yres = physical_size.height as f32;
                    unsafe {self.video.gl.viewport(0, 0, physical_size.width as i32, physical_size.height as i32)};
                    self.current.screen_rect = Rect::new(0.0, 0.0, self.video.xres / self.video.yres, 1.0);
                },
                _ => {},
            },
            Event::MainEventsCleared => {
                let t_now = Instant::now();
                let dt = t_now.duration_since(self.t_last).as_secs_f32();
                self.current.dt = dt;
                self.current.t += dt;
                self.t_last = t_now;
                self.current.frame += 1;
                self.current.mouse_delta = self.instant_mouse_pos - self.current.mouse_pos;
                self.current.mouse_pos = self.instant_mouse_pos;
                let state = self.current.clone();
                self.current.prev_keys = self.current.curr_keys.clone();
                self.current.repeat_keys = HashSet::new();
                self.current.seed = khash(self.current.seed * 196513497);
                self.current.scroll_delta = 0.0;
                self.current.lmb = match self.current.lmb {KeyStatus::JustPressed | KeyStatus::Pressed => KeyStatus::Pressed, KeyStatus::JustReleased | KeyStatus::Released => KeyStatus::Released};
                self.current.mmb = match self.current.mmb {KeyStatus::JustPressed | KeyStatus::Pressed => KeyStatus::Pressed, KeyStatus::JustReleased | KeyStatus::Released => KeyStatus::Released};
                self.current.rmb = match self.current.rmb {KeyStatus::JustPressed | KeyStatus::Pressed => KeyStatus::Pressed, KeyStatus::JustReleased | KeyStatus::Released => KeyStatus::Released};

                let mut new_outputs = FrameOutputs::new(state.screen_rect.aspect());

                self.root_scene.frame(&state, &mut new_outputs);

                if self.plant_cursor && !new_outputs.plant_cursor {
                    self.plant_cursor = false;
                    self.video.window.window().set_cursor_visible(true);
                } else if !self.plant_cursor && new_outputs.plant_cursor {
                    self.plant_cursor = true;
                    self.video.window.window().set_cursor_visible(false);
                }

                if let Some(desired_cursor) = new_outputs.set_cursor {
                    match desired_cursor {
                        0 => self.video.window.window().set_cursor_icon(CursorIcon::Default),
                        1 => self.video.window.window().set_cursor_icon(CursorIcon::Hand),
                        _ => {},
                    }
                }

                for sc in new_outputs.sounds.iter() {
                    self.channel.push(*sc).ok().unwrap();
                }
                self.video.render(&new_outputs, state.screen_rect.aspect());
            },
            _ => {},
        }
    }

    pub fn exit(&mut self) {
        println!("exiting");
        std::process::exit(0);
    }
}














fn sample_next(o: &mut SampleRequestOptions) -> f32 {
    o.mixer.tick()
}

pub struct SampleRequestOptions {
    pub sample_rate: f32,
    pub nchannels: usize,

    // pub filter: Filter,

    pub mixer: Mixer,

    pub channel: Consumer<AudioCommand>,
}

pub fn stream_setup_for<F>(on_sample: F, channel: Consumer<AudioCommand>) -> Result<cpal::Stream, anyhow::Error>
where
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => stream_make::<f32, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::I16 => stream_make::<i16, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::U16 => stream_make::<u16, _>(&device, &config.into(), on_sample, channel),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    Ok((host, device, config))
}

pub fn stream_make<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    on_sample: F,
    channel: Consumer<AudioCommand>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let sample_rate = config.sample_rate.0 as f32;
    let nchannels = config.channels as usize;
    let mut request = SampleRequestOptions {
        sample_rate,
        nchannels,

        mixer: Mixer::default(),

        channel,
    };
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            on_window(output, &mut request, on_sample)
        },
        err_fn,
    )?;

    Ok(stream)
}

fn on_window<T, F>(output: &mut [T], request: &mut SampleRequestOptions, mut on_sample: F)
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static,
{
    if let Some(sc) = request.channel.pop() {
        request.mixer.handle_command(sc);
    }
    for frame in output.chunks_mut(request.nchannels) {
        let value: T = cpal::Sample::from::<f32>(&on_sample(request));
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}
