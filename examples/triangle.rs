use lazy_vulkan::{DrawCall, LazyRenderer, LazyVulkan, Vertex, Workflow, NO_TEXTURE_ID};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};
/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

// Fucking winit
#[derive(Default)]
struct App {
    lazy_vulkan: Option<LazyVulkan>,
    lazy_renderer: Option<LazyRenderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Oh, you thought you could supply your own Vertex type? What is this, a rendergraph?!
        // Better make sure those shaders use the right layout!
        // **LAUGHS IN VULKAN**
        let vertices = [
            Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
        ];

        // Your own index type?! What are you going to use, `u16`?
        let indices = [0, 1, 2];

        // Alright, let's build some stuff
        let (lazy_vulkan, lazy_renderer) = LazyVulkan::builder()
            .initial_vertices(&vertices)
            .initial_indices(&indices)
            .fragment_shader(FRAGMENT_SHADER)
            .vertex_shader(VERTEX_SHADER)
            .build(event_loop);

        self.lazy_renderer = Some(lazy_renderer);
        self.lazy_vulkan = Some(lazy_vulkan);
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
                let lazy_renderer = self.lazy_renderer.as_mut().unwrap();
                let lazy_vulkan = self.lazy_vulkan.as_mut().unwrap();
                let new_render_surface = lazy_vulkan.resized(size.width, size.height);
                lazy_renderer.update_surface(new_render_surface, &lazy_vulkan.context().device);
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let lazy_renderer = self.lazy_renderer.as_mut().unwrap();
        let lazy_vulkan = self.lazy_vulkan.as_mut().unwrap();
        let framebuffer_index = lazy_vulkan.render_begin();
        lazy_renderer.render(
            &lazy_vulkan.context(),
            framebuffer_index,
            &[DrawCall::new(0, 3, NO_TEXTURE_ID, Workflow::Main)],
        );
        lazy_vulkan.render_end(framebuffer_index, &[lazy_vulkan.present_complete_semaphore]);
    }
}

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
