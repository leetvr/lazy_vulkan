use lazy_vulkan::{DrawCall, LazyVulkan, Vertex, NO_TEXTURE_ID};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
};
/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

pub fn main() {
    env_logger::init();

    // Oh, you thought you could supply your own Vertex type? What is this, a rendergraph?!
    // Better make sure those shaders use the right layout!
    // **LAUGHS IN VULKAN**
    let vertices = [
        Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
        Vertex::new([1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    // Your own index type?! What are you going to use, `u16`?
    let indices = [0, 1, 2, 2, 3, 0];

    // Alright, let's build some stuff
    let (mut lazy_vulkan, mut lazy_renderer, mut event_loop) = LazyVulkan::builder()
        .initial_vertices(&vertices)
        .initial_indices(&indices)
        .fragment_shader(FRAGMENT_SHADER)
        .vertex_shader(VERTEX_SHADER)
        .with_present(true)
        .build();

    // Off we go!
    // TODO: How do we share this between examples?
    let mut winit_initializing = true;
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event:
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,

            Event::NewEvents(cause) => {
                if cause == winit::event::StartCause::Init {
                    winit_initializing = true;
                } else {
                    winit_initializing = false;
                }
            }

            Event::MainEventsCleared => {
                let framebuffer_index = lazy_vulkan.render_begin();
                lazy_renderer.render(
                    &lazy_vulkan.context(),
                    framebuffer_index,
                    &[DrawCall::new(0, indices.len() as _, NO_TEXTURE_ID)],
                );
                lazy_vulkan
                    .render_end(framebuffer_index, &[lazy_vulkan.present_complete_semaphore]);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if winit_initializing {
                    return;
                } else {
                    let new_render_surface = lazy_vulkan.resized(size.width, size.height);
                    lazy_renderer.update_surface(new_render_surface, &lazy_vulkan.context().device);
                }
            }
            _ => (),
        }
    });

    // I guess we better do this or else the Dreaded Validation Layers will complain
    unsafe {
        lazy_renderer.cleanup(&lazy_vulkan.context().device);
    }
}
