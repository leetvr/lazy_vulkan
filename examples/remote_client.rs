use lazy_vulkan::vulkan_context::VulkanContext;
use lazy_vulkan::{
    create_swapchain_image_views, DrawCall, SwapchainInfo, Vertex, Workflow, NO_TEXTURE_ID,
};
use std::io::{Read, Write};
use std::net::TcpStream;

use ash::vk;
use log::info;

/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

pub fn main() -> std::io::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let vertices = [
        Vertex::new([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([-1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0]),
        Vertex::new([0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0]),
    ];

    // Your own index type?! What are you going to use, `u16`?
    let indices = [0, 1, 2];

    // Alright, let's build some stuff
    let vulkan_context = lazy_vulkan::vulkan_context::VulkanContext::new();
    let builder = lazy_vulkan::LazyVulkan::builder()
        .fragment_shader(FRAGMENT_SHADER)
        .vertex_shader(VERTEX_SHADER)
        .initial_indices(&indices)
        .initial_vertices(&vertices);

    info!("Conecting to server..");
    let mut stream = TcpStream::connect("127.0.0.1:8000")?;
    info!("Connected!");

    let mut buf: [u8; 1024] = [0; 1024];
    let swapchain_info = get_swapchain_info(&mut stream, &mut buf);
    info!("Swapchain info is {swapchain_info:?}!");
    let swapchain_images =
        get_swapchain_images(&mut stream, &vulkan_context, &swapchain_info, &mut buf);
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

    let renderer =
        lazy_vulkan::lazy_renderer::LazyRenderer::new(&vulkan_context, render_surface, &builder);

    let draw_calls = [DrawCall::new(0, 3, NO_TEXTURE_ID, Workflow::Main)];

    loop {
        let swapchain_image_index = get_swapchain_image_index(&mut stream, &mut buf);
        renderer.render(&vulkan_context, swapchain_image_index, &draw_calls);
        send_render_complete(&mut stream);
    }
}

fn send_render_complete(stream: &mut std::net::TcpStream) -> {
    todo!()
}

fn get_swapchain_image_index(stream: &mut std::net::TcpStream, buf: &mut [u8]) -> u32 {
    stream.write(&mut [2]).unwrap();
    stream.read(buf).unwrap();
    buf[0] as _
}

fn get_swapchain_images(
    stream: &mut TcpStream,
    vulkan: &VulkanContext,
    swapchain_info: &SwapchainInfo,
    buf: &mut [u8; 1024],
) -> Vec<vk::Image> {
    let device = &vulkan.device;
    stream.write(&mut [1]).unwrap();
    let len = stream.read(buf).unwrap();
    info!("Read {len} bytes");
    let handles: &[vk::HANDLE] =
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), swapchain_info.image_count as _) };
    info!("Got handle {handles:?}");

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

fn get_swapchain_info(stream: &mut TcpStream, buf: &mut [u8]) -> SwapchainInfo {
    stream.write(&mut [0]).unwrap();
    let len = stream.read(buf).unwrap();
    info!("Read {len} bytes");
    from_bytes(&buf[..len])
}

// Pure, undiluted evil
fn from_bytes<T: Clone>(b: &[u8]) -> T {
    unsafe { std::ptr::read(b.as_ptr().cast::<T>()) }.clone()
}
