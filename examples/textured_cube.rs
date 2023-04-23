use std::{f32::consts::TAU, time::Instant};

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

    let mut last_sim_update = Instant::now();
    let mut carousel = Carousel::new();

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
                let dt = Instant::now() - last_sim_update;
                if dt.as_secs_f32() > (1. / 120.) {
                    carousel.update(dt.as_secs_f32());
                    last_sim_update = Instant::now();
                }
                lazy_renderer.render(
                    &lazy_vulkan.context(),
                    framebuffer_index,
                    &carousel.draw_calls(),
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

            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::A),
                                ..
                            },
                        ..
                    },
                ..
            } => carousel.rotate(),
            _ => (),
        }
    });

    // I guess we better do this or else the Dreaded Validation Layers will complain
    unsafe {
        lazy_renderer.cleanup(&lazy_vulkan.context().device);
    }
}

pub struct Carousel {
    quads: Vec<Quad>,
    state: CarouselState,
    theta: f32,
    animation_state: AnimationState,
}

impl Carousel {
    pub fn new() -> Self {
        Self {
            quads: create_quads(),
            state: CarouselState::Idle,
            theta: 0.,
            animation_state: AnimationState::new(0., 0.),
        }
    }

    pub fn update(&mut self, dt: f32) {
        let rotation_time = self.animation_state.target_time;

        if self.animation_state.elapsed >= rotation_time && self.theta == self.animation_state.end {
            self.theta = self.animation_state.end;
            self.animation_state.start = self.animation_state.end;
            self.animation_state.elapsed = 0.;
            self.animation_state.target_time = 1.;
            let slice_theta = std::f32::consts::TAU / self.quads.len() as f32;

            for (n, quad) in self.quads.iter_mut().enumerate() {
                quad.theta = (n as f32 * slice_theta) + self.theta;
            }

            return;
        }

        self.animation_state.elapsed += dt;
        let progress = self.animation_state.elapsed / rotation_time;
        let target = lerp(
            self.animation_state.start,
            self.animation_state.end,
            simple_easing::expo_in(progress),
        );
        let delta = target - self.theta;
        self.theta += delta;
        println!("{progress} - {delta}");
        for quad in &mut self.quads {
            quad.theta += delta;
        }
    }

    pub fn rotate(&mut self) {
        let slice_theta = std::f32::consts::TAU / self.quads.len() as f32;
        self.animation_state.end = self.animation_state.end + slice_theta;
        self.animation_state.start = self.theta;
        self.animation_state.elapsed -= 0.5;
        if self.animation_state.elapsed <= 0. {
            self.animation_state.elapsed = 0.;
        }
    }

    pub fn draw_calls(&self) -> Vec<DrawCall> {
        let radius = 3.;

        self.quads
            .iter()
            .map(|quad| {
                let theta = quad.theta;
                let x = radius * theta.sin();
                let z = radius * theta.cos();
                let transform = Affine3A::from_translation([x, 0., z].into());
                DrawCall::new(0, 6, NO_TEXTURE_ID, transform)
            })
            .collect()
    }
}

fn get_increment(progress: f32, theta: f32) -> f32 {
    todo!()
}

#[derive(PartialEq, PartialOrd)]
enum CarouselState {
    Idle,
    Rotating(usize),
}

#[derive(PartialEq, PartialOrd)]
pub struct AnimationState {
    start: f32,
    end: f32,
    elapsed: f32,
    target_time: f32,
}

impl AnimationState {
    pub fn new(start: f32, end: f32) -> Self {
        Self {
            start,
            end,
            elapsed: 0.,
            target_time: 1.,
        }
    }
}

pub struct Quad {
    pub theta: f32,
}

fn lerp(start: f32, end: f32, amount: f32) -> f32 {
    start + ((end - start) * amount)
}

fn create_quads() -> Vec<Quad> {
    let number_of_quads = 10;
    let slice_theta = std::f32::consts::TAU / (number_of_quads as f32);
    (0..number_of_quads)
        .map(|n| Quad {
            theta: n as f32 * slice_theta,
        })
        .collect()
}
