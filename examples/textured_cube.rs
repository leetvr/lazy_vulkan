use std::f32::consts::TAU;

use glam::Affine3A;
use lazy_vulkan::{DrawCall, LazyVulkan, Vertex, NO_TEXTURE_ID};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
};
pub fn main() {
    env_logger::init();

    // SQUUUUUUUUUUUUUUUARRRRRRRRRE
    let vertices = [
        Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
        Vertex::new([1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    let indices = [0, 1, 2, 2, 3, 0];

    // Alright, let's build some stuff
    let (mut lazy_vulkan, mut lazy_renderer, mut event_loop) = LazyVulkan::builder()
        .initial_vertices(&vertices)
        .initial_indices(&indices)
        .with_present(true)
        .build();

    lazy_renderer.camera.position.y = 2.;
    lazy_renderer.camera.position.z = 10.;
    lazy_renderer.camera.pitch = -15_f32.to_radians();

    let draw_calls = create_draw_calls();

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
                lazy_renderer.render(&lazy_vulkan.context(), framebuffer_index, &draw_calls);
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

fn create_draw_calls() -> Vec<DrawCall> {
    let number_of_quads = 9;
    let slice_theta = TAU / (number_of_quads as f32);
    let radius = 3.;

    (0..number_of_quads)
        .map(|n| {
            let theta = n as f32 * slice_theta;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            let transform = Affine3A::from_translation([x, 0., z].into());
            DrawCall::new(0, 6, NO_TEXTURE_ID, transform)
        })
        .collect()
}
