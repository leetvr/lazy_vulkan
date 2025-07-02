use ash::vk;
use lazy_vulkan::{
    find_memorytype_index, vulkan_texture::VulkanTexture, DrawCall, LazyRenderer, LazyVulkan,
    SwapchainInfo, Vertex,
};
use log::{debug, info};
use std::io::{Read, Write};
#[cfg(not(target_os = "windows"))]
use std::os::unix::net::{UnixListener, UnixStream};
#[cfg(target_os = "windows")]
use uds_windows::{UnixListener, UnixStream};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};

/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'_ [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'_ [u8] = include_bytes!("shaders/triangle.vert.spv");
const SWAPCHAIN_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
static UNIX_SOCKET_PATH: &'_ str = "lazy_vulkan.socket";

// Fucking winit
struct App {
    lazy_vulkan: Option<LazyVulkan>,
    lazy_renderer: Option<LazyRenderer>,
    textures: Vec<VulkanTexture>,
    semaphores: Vec<vk::Semaphore>,
    stream: UnixStream,
    buf: [u8; 1024],
}

impl App {
    fn new() -> Self {
        if std::fs::remove_file(UNIX_SOCKET_PATH).is_ok() {
            debug!("Removed pre-existing unix socket at {UNIX_SOCKET_PATH}");
        }
        // Hello client? Are you there?
        let listener = UnixListener::bind(UNIX_SOCKET_PATH).unwrap();
        info!("Listening on {UNIX_SOCKET_PATH} - waiting for client..");

        // Bonjour, monsieur client!
        let (stream, _) = listener.accept().unwrap();
        info!("Client connected!");
        let buf = [0; 1024];

        Self {
            lazy_vulkan: None,
            lazy_renderer: None,
            stream,
            textures: Default::default(),
            semaphores: Default::default(),
            buf,
        }
    }

    fn get_render_complete(&mut self) {
        self.stream.read(&mut self.buf).unwrap();
    }

    fn send_swapchain_image_index(&mut self, framebuffer_index: u32) {
        self.stream.read(&mut self.buf).unwrap();
        self.stream.write(&mut [framebuffer_index as u8]).unwrap();
    }

    fn send_swapchain_info(&mut self, swapchain_info: &SwapchainInfo) -> std::io::Result<()> {
        self.stream.read(&mut self.buf)?;
        let value = self.buf[0];
        debug!("Read {value}");

        if value == 0 {
            let write = self.stream.write(bytes_of(swapchain_info)).unwrap();
            debug!("Write {write} bytes");
            return Ok(());
        } else {
            panic!("Invalid request!");
        }
    }

    fn send_image_memory_handles(&mut self, image_memory_handles: &[vk::HANDLE]) {
        self.stream.read(&mut self.buf).unwrap();
        let value = self.buf[0];
        debug!("Read {value}");

        if value == 1 {
            let write = self
                .stream
                .write(bytes_of_slice(image_memory_handles))
                .unwrap();
            debug!("Write {write} bytes");
        } else {
            panic!("Invalid request!");
        }
    }

    fn send_semaphore_handles(&mut self, semaphore_handles: &[vk::HANDLE]) {
        self.stream.read(&mut self.buf).unwrap();
        let value = self.buf[0];
        debug!("Read {value}");

        debug!("Sending handles: {semaphore_handles:?}");
        let write = self
            .stream
            .write(bytes_of_slice(semaphore_handles))
            .unwrap();
        debug!("Wrote {write} bytes");
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // it's a square (you can call it a quad if you're fancy)
        let vertices = [
            Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0]), // bottom right
            Vertex::new([-1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0]), // bottom left
            Vertex::new([1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 0.0]), // top right
            Vertex::new([-1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0]), // top left
        ];
        let indices = [0, 1, 2, 2, 1, 3];

        // Alright, let's build some stuff
        let (lazy_vulkan, mut lazy_renderer) = LazyVulkan::builder()
            .initial_vertices(&vertices)
            .initial_indices(&indices)
            .fragment_shader(FRAGMENT_SHADER)
            .vertex_shader(VERTEX_SHADER)
            .with_present(true)
            .build(event_loop);

        let swapchain_info = SwapchainInfo {
            image_count: lazy_vulkan.surface.desired_image_count,
            resolution: lazy_vulkan.surface.surface_resolution,
            format: SWAPCHAIN_FORMAT,
        };
        let (images, image_memory_handles) =
            unsafe { create_render_images(lazy_vulkan.context(), &swapchain_info) };
        let (semaphores, semaphore_handles) =
            unsafe { create_semaphores(lazy_vulkan.context(), swapchain_info.image_count) };
        self.textures = create_render_textures(lazy_vulkan.context(), &mut lazy_renderer, images);

        self.send_swapchain_info(&swapchain_info).unwrap();
        self.send_image_memory_handles(&image_memory_handles);
        self.send_semaphore_handles(&semaphore_handles);

        self.semaphores = semaphores;
        self.lazy_renderer = Some(lazy_renderer);
        self.lazy_vulkan = Some(lazy_vulkan);
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        let framebuffer_index = self.lazy_vulkan.as_ref().unwrap().render_begin();
        self.send_swapchain_image_index(framebuffer_index);
        self.get_render_complete();

        let lazy_renderer = self.lazy_renderer.as_mut().unwrap();
        let lazy_vulkan = self.lazy_vulkan.as_mut().unwrap();

        let texture_id = self.textures[framebuffer_index as usize].id;
        lazy_renderer.render(
            lazy_vulkan.context(),
            framebuffer_index,
            &[DrawCall::new(0, 6, texture_id, lazy_vulkan::Workflow::Main)],
        );

        let semaphore = self.semaphores[framebuffer_index as usize];
        lazy_vulkan.render_end(
            framebuffer_index,
            &[semaphore, lazy_vulkan.rendering_complete_semaphore],
        );
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
}

pub fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()).unwrap();
}

unsafe fn create_semaphores(
    context: &lazy_vulkan::vulkan_context::VulkanContext,
    image_count: u32,
) -> (Vec<vk::Semaphore>, Vec<vk::HANDLE>) {
    let device = &context.device;
    let external_semaphore =
        ash::khr::external_semaphore_win32::Device::new(&context.instance, &context.device);

    let handle_type = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT;
    (0..image_count)
        .map(|_| {
            let mut external_semaphore_info =
                vk::ExportSemaphoreCreateInfo::default().handle_types(handle_type);
            let semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut external_semaphore_info),
                    None,
                )
                .unwrap();

            let handle = external_semaphore
                .get_semaphore_win32_handle(
                    &vk::SemaphoreGetWin32HandleInfoKHR::default()
                        .handle_type(handle_type)
                        .semaphore(semaphore),
                )
                .unwrap();

            (semaphore, handle)
        })
        .unzip()
}

fn create_render_textures(
    vulkan_context: &lazy_vulkan::vulkan_context::VulkanContext,
    renderer: &mut LazyRenderer,
    mut images: Vec<vk::Image>,
) -> Vec<VulkanTexture> {
    let descriptors = &mut renderer.descriptors;
    let address_mode = vk::SamplerAddressMode::REPEAT;
    let filter = vk::Filter::LINEAR;
    images
        .drain(..)
        .map(|image| {
            let view = unsafe { vulkan_context.create_image_view(image, SWAPCHAIN_FORMAT) };
            let sampler = unsafe {
                vulkan_context
                    .device
                    .create_sampler(
                        &vk::SamplerCreateInfo::default()
                            .address_mode_u(address_mode)
                            .address_mode_v(address_mode)
                            .address_mode_w(address_mode)
                            .mag_filter(filter)
                            .min_filter(filter),
                        None,
                    )
                    .unwrap()
            };

            let id =
                unsafe { descriptors.update_texture_descriptor_set(view, sampler, vulkan_context) };

            lazy_vulkan::vulkan_texture::VulkanTexture {
                image,
                memory: vk::DeviceMemory::null(), // todo
                sampler,
                view,
                id,
            }
        })
        .collect()
}

unsafe fn create_render_images(
    context: &lazy_vulkan::vulkan_context::VulkanContext,
    swapchain_info: &SwapchainInfo,
) -> (Vec<vk::Image>, Vec<vk::HANDLE>) {
    let device = &context.device;
    let SwapchainInfo {
        resolution,
        format,
        image_count,
    } = swapchain_info;
    let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;

    (0..(*image_count))
        .map(|_| {
            let mut handle_info =
                vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
            let image = device
                .create_image(
                    &vk::ImageCreateInfo {
                        image_type: vk::ImageType::TYPE_2D,
                        format: *format,
                        extent: (*resolution).into(),
                        mip_levels: 1,
                        array_layers: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        p_next: &mut handle_info as *mut _ as *mut _,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let memory_requirements = device.get_image_memory_requirements(image);
            let memory_index = find_memorytype_index(
                &memory_requirements,
                &context.memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory type for image");
            let mut export_handle_info =
                vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
            let memory = context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(memory_requirements.size)
                        .memory_type_index(memory_index)
                        .push_next(&mut export_handle_info),
                    None,
                )
                .unwrap();

            device.bind_image_memory(image, memory, 0).unwrap();

            let external_memory =
                ash::khr::external_memory_win32::Device::new(&context.instance, &context.device);
            let handle = external_memory
                .get_memory_win32_handle(
                    &vk::MemoryGetWin32HandleInfoKHR::default()
                        .handle_type(handle_type)
                        .memory(memory),
                )
                .unwrap();
            debug!("Created handle {handle:?}");

            (image, handle)
        })
        .unzip()
}

fn bytes_of_slice<T>(t: &[T]) -> &[u8] {
    unsafe {
        let ptr = t.as_ptr();
        std::slice::from_raw_parts(ptr.cast(), std::mem::size_of::<T>() * t.len())
    }
}

fn bytes_of<T>(t: &T) -> &[u8] {
    unsafe {
        let ptr = t as *const T;
        std::slice::from_raw_parts(ptr.cast(), std::mem::size_of::<T>())
    }
}
