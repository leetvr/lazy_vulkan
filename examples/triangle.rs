use std::{path::Path, time::Instant};

use glam::Vec4;
use lazy_vulkan::{Context, LazyVulkan, SubRenderer};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

pub struct TriangleRenderer {
    pipeline: lazy_vulkan::Pipeline,
    pub colour: glam::Vec4,
}

impl TriangleRenderer {
    pub fn new(renderer: &lazy_vulkan::Renderer) -> Self {
        let pipeline = renderer.create_pipeline::<Registers>(
            Path::new("examples/shaders/triangle.vert.spv"),
            Path::new("examples/shaders/colour.frag.spv"),
        );

        Self {
            pipeline,
            colour: glam::Vec4::ONE,
        }
    }
}

impl SubRenderer for TriangleRenderer {
    type State = RenderState;
    fn draw_opaque(
        &mut self,
        state: &Self::State,
        context: &Context,
        params: lazy_vulkan::DrawParams,
    ) {
        self.begin_rendering(context, &self.pipeline);
        self.colour = psychedelic_vec4(state.t);
        unsafe {
            self.pipeline.update_registers(&Registers {
                colour: self.colour,
            });
            context
                .device
                .cmd_draw(params.draw_command_buffer, 3, 1, 0, 0)
        }
    }

    fn label(&self) -> &'static str {
        "Triangle Renderer"
    }
}

pub struct RenderState {
    last_render_time: Instant,
    t: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {
    colour: glam::Vec4,
}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

#[derive(Default)]
struct App {
    state: Option<State>,
}

struct State {
    lazy_vulkan: LazyVulkan,
    sub_renderers: Vec<Box<dyn SubRenderer<State = RenderState>>>,
    render_state: RenderState,
    window: winit::window::Window,
}

// ------------
// BOILERPLATE
// ------------

impl<'a> ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Triangle Example")
                    .with_inner_size(PhysicalSize::new(1024, 768)),
            )
            .unwrap();

        let lazy_vulkan = LazyVulkan::from_window(&window);
        let sub_renderer = TriangleRenderer::new(&lazy_vulkan.renderer);

        self.state = Some(State {
            lazy_vulkan,
            window,
            sub_renderers: vec![Box::new(sub_renderer)],
            render_state: RenderState {
                t: 0.0,
                last_render_time: Instant::now(),
            },
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::Resized(size) => {
                let state = self.state.as_mut().unwrap();
                state.lazy_vulkan.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let state = self.state.as_mut().unwrap();
                state.render_state.t += state.render_state.last_render_time.elapsed().as_secs_f32();
                let lazy_vulkan = &mut state.lazy_vulkan;
                lazy_vulkan.draw(&state.render_state, &mut state.sub_renderers);
                state.render_state.last_render_time = Instant::now();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();
        state.window.request_redraw();
    }
}

// from chatGPT
pub fn psychedelic_vec4(mut time: f32) -> Vec4 {
    time *= 50.0;

    let r = (time * 1.0 + 0.0).sin() * 0.5 + 0.5;
    let g = (time * 1.3 + std::f32::consts::FRAC_PI_2).sin() * 0.5 + 0.5;
    let b = (time * 1.6 + std::f32::consts::PI).sin() * 0.5 + 0.5;

    let a = (time * 0.4).cos() * 0.5 + 0.5;

    Vec4::new(r, g, b, a)
}

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
