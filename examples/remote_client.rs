use lazy_vulkan::vulkan_context::VulkanContext;
use lazy_vulkan::{create_swapchain_image_views, DrawCall, SwapchainInfo, Vertex, NO_TEXTURE_ID};
use std::io::{Read, Write};
#[cfg(not(target_os = "windows"))]
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
#[cfg(target_os = "windows")]
use uds_windows::UnixStream;

use ash::vk;
use log::{debug, error, info};

/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

#[derive(Debug, Clone)]
pub enum Color {
    Blue,
    Red,
    Green,
}

impl Color {
    fn to_rgba(&self) -> [f32; 4] {
        match self {
            Color::Blue => [0., 0., 1., 0.],
            Color::Red => [1., 0., 0., 0.],
            Color::Green => [0., 1., 0., 0.],
        }
    }
}

static mut COLOUR: Mutex<Color> = Mutex::new(Color::Blue);
static UNIX_SOCKET_PATH: &'_ str = "lazy_vulkan.socket";

pub fn main() -> std::io::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let mut vertices = [
        Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    // Your own index type?! What are you going to use, `u16`?
    let indices = [0, 1, 2];

    // Alright, let's build some stuff
    let mut vulkan_context = lazy_vulkan::vulkan_context::VulkanContext::new();
    let builder = lazy_vulkan::LazyVulkan::builder()
        .fragment_shader(FRAGMENT_SHADER)
        .vertex_shader(VERTEX_SHADER)
        .initial_indices(&indices)
        .initial_vertices(&vertices);

    info!("Conecting to server at {UNIX_SOCKET_PATH}..");
    let mut stream = UnixStream::connect(UNIX_SOCKET_PATH)?;
    info!("Connected!");

    let mut buf: [u8; 1024] = [0; 1024];
    let swapchain_info = get_swapchain_info(&mut stream, &mut buf);
    info!("Swapchain info is {swapchain_info:?}!");
    let swapchain_images =
        get_swapchain_images(&mut stream, &vulkan_context, &swapchain_info, &mut buf);
    let semaphores = get_semaphores(
        &mut stream,
        &vulkan_context,
        swapchain_info.image_count,
        &mut buf,
    );
    info!("Images are: {swapchain_images:?}");
    let image_views = create_swapchain_image_views(
        &swapchain_images,
        swapchain_info.format,
        &vulkan_context.device,
    );

    let render_surface = lazy_vulkan::lazy_renderer::RenderSurface {
        resolution: swapchain_info.resolution,
        format: swapchain_info.format,
        image_views,
    };

    let mut renderer =
        lazy_vulkan::lazy_renderer::LazyRenderer::new(&vulkan_context, render_surface, &builder);

    let draw_calls = [DrawCall::new(0, 3, NO_TEXTURE_ID)];
    let fences = create_fences(&vulkan_context, swapchain_info.image_count);
    let command_buffers = create_command_buffers(&vulkan_context, swapchain_info.image_count);

    std::thread::spawn(|| {
        let mut buffer = String::new();
        loop {
            info!("[CLIENT] Waiting for input..");
            if let Ok(_) = std::io::stdin().read_line(&mut buffer) {
                let new_color = match buffer.as_str().trim() {
                    "b" => Some(Color::Blue),
                    "g" => Some(Color::Green),
                    "r" => Some(Color::Red),
                    _ => {
                        error!(
                            "[CLIENT] Invalid input {buffer:?}. Press r, g or b then hit enter."
                        );
                        None
                    }
                };

                if let Some(new_color) = new_color {
                    info!("[CLIENT] Triangle is now {new_color:?}");
                    unsafe { *COLOUR.get_mut().unwrap() = new_color }
                }

                buffer.clear();
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    loop {
        let swapchain_image_index = get_swapchain_image_index(&mut stream, &mut buf);
        let fence = fences[swapchain_image_index as usize];
        let command_buffer = command_buffers[swapchain_image_index as usize];
        let semaphore = semaphores[swapchain_image_index as usize];

        vulkan_context.draw_command_buffer = command_buffer;
        update_colour(&mut vertices, &vulkan_context, &mut renderer);
        begin_frame(&vulkan_context, fence, command_buffer);
        renderer.render(&vulkan_context, swapchain_image_index, &draw_calls);
        end_frame(&vulkan_context, fence, command_buffer);
        fake_submit(&vulkan_context, semaphore);
        send_render_complete(&mut stream);
    }
}

fn fake_submit(vulkan_context: &VulkanContext, semaphore: vk::Semaphore) {
    unsafe {
        vulkan_context
            .device
            .queue_submit(
                vulkan_context.queue,
                std::slice::from_ref(
                    &vk::SubmitInfo::builder().signal_semaphores(std::slice::from_ref(&semaphore)),
                ),
                vk::Fence::null(),
            )
            .unwrap()
    }
}

fn get_semaphores(
    stream: &mut UnixStream,
    vulkan_context: &VulkanContext,
    image_count: u32,
    buf: &mut [u8],
) -> Vec<vk::Semaphore> {
    let device = &vulkan_context.device;
    stream.write(&mut [1]).unwrap();
    let len = stream.read(buf).unwrap();
    debug!("Read {len} bytes");
    let handles: &[vk::HANDLE] =
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), image_count as _) };
    debug!("Got handle {handles:?}");
    let external_semaphore = ash::extensions::khr::ExternalSemaphoreWin32::new(
        &vulkan_context.instance,
        &vulkan_context.device,
    );
    let handle_type = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT;

    handles
        .iter()
        .map(|h| unsafe {
            let mut external_semaphore_info =
                vk::ExportSemaphoreCreateInfo::builder().handle_types(handle_type);
            let semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::builder().push_next(&mut external_semaphore_info),
                    None,
                )
                .unwrap();

            external_semaphore
                .import_semaphore_win32_handle(
                    &vk::ImportSemaphoreWin32HandleInfoKHR::builder()
                        .handle(*h)
                        .semaphore(semaphore)
                        .handle_type(handle_type),
                )
                .unwrap();

            semaphore
        })
        .collect()
}

fn update_colour(
    vertices: &mut [Vertex],
    vulkan_context: &VulkanContext,
    renderer: &mut lazy_vulkan::LazyRenderer,
) {
    if let Ok(colour) = unsafe { COLOUR.get_mut() } {
        for v in vertices.iter_mut() {
            v.colour = colour.to_rgba().into()
        }

        unsafe {
            renderer.vertex_buffer.overwrite(vulkan_context, &vertices);
        }
    }
}

fn create_command_buffers(
    vulkan_context: &VulkanContext,
    image_count: u32,
) -> Vec<vk::CommandBuffer> {
    unsafe {
        vulkan_context
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(vulkan_context.command_pool)
                    .command_buffer_count(image_count),
            )
            .unwrap()
    }
}

fn create_fences(vulkan_context: &VulkanContext, image_count: u32) -> Vec<vk::Fence> {
    (0..image_count)
        .map(|_| unsafe {
            vulkan_context
                .device
                .create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .unwrap()
        })
        .collect()
}

fn end_frame(vulkan_context: &VulkanContext, fence: vk::Fence, command_buffer: vk::CommandBuffer) {
    let device = &vulkan_context.device;
    unsafe {
        device.end_command_buffer(command_buffer).unwrap();

        device
            .queue_submit(
                vulkan_context.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(std::slice::from_ref(&command_buffer))
                    .build()],
                fence,
            )
            .unwrap();
    }
}

fn begin_frame(
    vulkan_context: &VulkanContext,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
) {
    let device = &vulkan_context.device;
    unsafe {
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, std::u64::MAX)
            .unwrap();
        device.reset_fences(std::slice::from_ref(&fence)).unwrap();
        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .unwrap();
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
    }
}

fn send_render_complete(stream: &mut UnixStream) {
    stream.write(&mut [3]).unwrap();
}

fn get_swapchain_image_index(stream: &mut UnixStream, buf: &mut [u8]) -> u32 {
    stream.write(&mut [2]).unwrap();
    stream.read(buf).unwrap();
    buf[0] as _
}

fn get_swapchain_images(
    stream: &mut UnixStream,
    vulkan: &VulkanContext,
    swapchain_info: &SwapchainInfo,
    buf: &mut [u8; 1024],
) -> Vec<vk::Image> {
    let device = &vulkan.device;
    stream.write(&mut [1]).unwrap();
    let len = stream.read(buf).unwrap();
    debug!("Read {len} bytes");
    let handles: &[vk::HANDLE] =
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), swapchain_info.image_count as _) };
    debug!("Got handle {handles:?}");

    handles
        .into_iter()
        .map(|handle| unsafe {
            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;

            let mut external_memory_image_create_info =
                vk::ExternalMemoryImageCreateInfo::builder().handle_types(handle_type);
            let image = device
                .create_image(
                    &vk::ImageCreateInfo {
                        image_type: vk::ImageType::TYPE_2D,
                        format: swapchain_info.format,
                        extent: swapchain_info.resolution.into(),
                        mip_levels: 1,
                        array_layers: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        p_next: &mut external_memory_image_create_info as *mut _ as *mut _,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let requirements = device.get_image_memory_requirements(image);
            let mut external_memory_allocate_info = vk::ImportMemoryWin32HandleInfoKHR::builder()
                .handle(*handle)
                .handle_type(handle_type);
            let memory = vulkan
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(requirements.size)
                        .push_next(&mut external_memory_allocate_info),
                    None,
                )
                .unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            image
        })
        .collect()
}

fn get_swapchain_info(stream: &mut UnixStream, buf: &mut [u8]) -> SwapchainInfo {
    stream.write(&mut [0]).unwrap();
    let len = stream.read(buf).unwrap();
    info!("Read {len} bytes");
    from_bytes(&buf[..len])
}

// Pure, undiluted evil
fn from_bytes<T: Clone>(b: &[u8]) -> T {
    unsafe { std::ptr::read(b.as_ptr().cast::<T>()) }.clone()
}
