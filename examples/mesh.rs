use std::{f32::consts::TAU, path::Path, time::Instant};

use ash::vk;
use glam::{vec2, vec4, Quat};
use lazy_vulkan::{BufferAllocation, Context, Image, LazyVulkan, SubRenderer, TransferToken};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

#[repr(C)]
#[derive(Copy, Debug, Clone)]
struct Vertex {
    position: glam::Vec4,
    uv: glam::Vec2,
}

impl Vertex {
    const fn new(position: glam::Vec4, uv: glam::Vec2) -> Vertex {
        Vertex { position, uv }
    }
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

const CUBE_VERTICES: &[Vertex] = &[
    // Front face (+Z)
    Vertex::new(vec4(-0.5, -0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, -0.5, 0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, -0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, 0.5, 0.5, 1.0), vec2(0.0, 0.0)),
    // Back face (–Z)
    Vertex::new(vec4(0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(-0.5, -0.5, -0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(-0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(-0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(0.5, 0.5, -0.5, 1.0), vec2(0.0, 0.0)),
    // Left face (–X)
    Vertex::new(vec4(-0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(-0.5, -0.5, 0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(-0.5, 0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(-0.5, 0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, 0.5, -0.5, 1.0), vec2(0.0, 0.0)),
    // Right face (+X)
    Vertex::new(vec4(0.5, -0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, -0.5, -0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(0.5, -0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(0.5, 0.5, 0.5, 1.0), vec2(0.0, 0.0)),
    // Top face (+Y)
    Vertex::new(vec4(-0.5, 0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, 0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, 0.5, 0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, 0.5, -0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, 0.5, -0.5, 1.0), vec2(0.0, 0.0)),
    // Bottom face (–Y)
    Vertex::new(vec4(-0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, -0.5, -0.5, 1.0), vec2(1.0, 1.0)),
    Vertex::new(vec4(0.5, -0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, -0.5, -0.5, 1.0), vec2(0.0, 1.0)),
    Vertex::new(vec4(0.5, -0.5, 0.5, 1.0), vec2(1.0, 0.0)),
    Vertex::new(vec4(-0.5, -0.5, 0.5, 1.0), vec2(0.0, 0.0)),
];

pub struct MeshRenderer {
    mvp: glam::Mat4,
    pipeline: lazy_vulkan::Pipeline,
    buffer: BufferAllocation<Vertex>,
    initial_upload: TransferToken,
    rotation: glam::Quat,
    position: glam::Vec3,
    logo_image: Image,
}

impl MeshRenderer {
    pub fn new(renderer: &mut lazy_vulkan::Renderer) -> Self {
        // Create our pipeline
        let pipeline = renderer.create_pipeline::<Registers>(
            Path::new("examples/shaders/mesh.vert.spv"),
            Path::new("examples/shaders/texture.frag.spv"),
        );

        // Allocate some resources
        let allocator = &mut renderer.allocator;
        let mut buffer =
            allocator.allocate_buffer(10 * 1024 * 1024, vk::BufferUsageFlags::STORAGE_BUFFER);
        let initial_upload = allocator.stage_buffer_transfer(&CUBE_VERTICES, &mut buffer);
        let (image_bytes, extent) = decode_png(Path::new("examples/vulkan.png"));
        let logo_image = renderer.create_image(
            vk::Format::R8G8B8A8_SRGB,
            extent,
            image_bytes,
            vk::ImageUsageFlags::SAMPLED,
        );

        // Build up the perspective matrix
        let mvp = build_mvp(renderer.swapchain.extent);

        Self {
            pipeline,
            buffer,
            initial_upload,
            mvp,
            rotation: glam::Quat::IDENTITY,
            position: glam::Vec3::ZERO,
            logo_image,
        }
    }
}

impl SubRenderer for MeshRenderer {
    type State = RenderState;
    fn draw(&mut self, context: &Context, params: lazy_vulkan::DrawParams) {
        // Make sure our resources are available before we use them
        if !self.initial_upload.is_complete() {
            return;
        }

        if !self.logo_image.transfer_complete.is_complete() {
            return;
        }

        self.begin_rendering(context, &self.pipeline);

        let mvp =
            self.mvp * glam::Affine3A::from_rotation_translation(self.rotation, self.position);
        let vertex_count = self.buffer.len() as u32;

        unsafe {
            self.pipeline.update_registers(
                params.draw_command_buffer,
                context,
                &Registers {
                    mvp,
                    vertex_buffer: self.buffer.device_address,
                    texture_id: self.logo_image.id,
                },
            );
            context
                .device
                .cmd_draw(params.draw_command_buffer, vertex_count, 1, 0, 0)
        }
    }

    fn update_state(&mut self, state: &RenderState) {
        self.rotation = glam::Quat::from_rotation_y(state.t * 5.);
        self.position = glam::Vec3::Y * (((state.t * 2.0).sin()) * 0.5 + 0.5) * 2.0;
    }

    fn stage_transfers(&mut self, _allocator: &mut lazy_vulkan::Allocator) {
        // no-op
    }
}

pub struct RenderState {
    last_render_time: Instant,
    t: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Registers {
    mvp: glam::Mat4,
    vertex_buffer: vk::DeviceAddress,
    texture_id: u32,
}

unsafe impl bytemuck::Zeroable for Registers {}
unsafe impl bytemuck::Pod for Registers {}

fn decode_png(path: &Path) -> (Vec<u8>, vk::Extent2D) {
    use std::fs::File;
    let mut decoder = png::Decoder::new(File::open(path).unwrap());
    decoder.set_transformations(png::Transformations::ALPHA);
    let mut reader = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; reader.output_buffer_size()];
    // Read the next frame. An APNG might contain multiple frames.
    let info = reader.next_frame(&mut buf).unwrap();
    // Grab the bytes of the image.
    let bytes = &buf[..info.buffer_size()];

    let extent = vk::Extent2D {
        width: info.width,
        height: info.height,
    };

    (bytes.to_vec(), extent)
}

fn build_mvp(extent: vk::Extent2D) -> glam::Mat4 {
    // Build up the perspective matrix
    let aspect_ratio = extent.width as f32 / extent.height as f32;
    let mut perspective =
        glam::Mat4::perspective_infinite_reverse_rh(60_f32.to_radians(), aspect_ratio, 0.01);

    perspective.y_axis *= -1.0;

    // Get view_from_world
    let world_from_view = glam::Affine3A::from_rotation_translation(
        Quat::from_euler(glam::EulerRot::YXZ, TAU * 0.1, -TAU * 0.1, 0.),
        glam::Vec3::new(4., 4., 4.),
    );
    let view_from_world = world_from_view.inverse();

    perspective * view_from_world
}

// ------------
// BOILERPLATE
// ------------
#[derive(Default)]
struct App {
    state: Option<State>,
}

struct State {
    lazy_vulkan: LazyVulkan,
    sub_renderers: Vec<Box<dyn SubRenderer<State = RenderState>>>,
    render_state: RenderState,
}

impl<'a> ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Mesh Example")
                    .with_inner_size(PhysicalSize::new(1024, 768)),
            )
            .unwrap();

        let mut lazy_vulkan = LazyVulkan::from_window(window);
        let sub_renderer = MeshRenderer::new(&mut lazy_vulkan.renderer);

        self.state = Some(State {
            lazy_vulkan,
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
        let Some(state) = self.state.as_mut() else {
            return;
        };

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
                let lazy_vulkan = &mut state.lazy_vulkan;
                lazy_vulkan.draw(&state.render_state, &mut state.sub_renderers);
                state.render_state.t += state.render_state.last_render_time.elapsed().as_secs_f32();
                state.render_state.last_render_time = Instant::now();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        state.lazy_vulkan.window.request_redraw();
    }
}

pub fn main() {
    env_logger::init();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default()).unwrap();
}
