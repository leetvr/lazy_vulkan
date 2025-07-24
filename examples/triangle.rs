use lazy_vulkan::{LazyVulkan, SubRenderer};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};
/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

// Fucking winit
#[derive(Default)]
struct App<'a> {
    lazy_vulkan: Option<LazyVulkan<'a, ()>>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Triangle Example")
                    .with_inner_size(PhysicalSize::new(1024, 768)),
            )
            .unwrap();
        self.lazy_vulkan = Some(LazyVulkan::new(
            window,
            |_: &lazy_vulkan::Renderer<'a, _>| vec![],
        ));
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
                let lazy_vulkan = self.lazy_vulkan.as_mut().unwrap();
                lazy_vulkan.resize(size.height, size.width);
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let lazy_vulkan = self.lazy_vulkan.as_mut().unwrap();
        lazy_vulkan.draw(&());
    }
}

pub struct TriangleRenderer {}

impl<'a, T> SubRenderer<'a, T> for TriangleRenderer {
    fn draw(&mut self, params: lazy_vulkan::DrawParams<'a, T>) {
        todo!()
    }

    fn stage_transfers(&mut self, allocator: &mut lazy_vulkan::Allocator) {
        todo!()
    }
}

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
