use ash::vk;
use lazy_vulkan::{
    find_memorytype_index, DrawCall, LazyVulkan, SwapchainInfo, Vertex, Workflow, NO_TEXTURE_ID,
};
use log::info;
use std::io::{Read, Write};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
};
/// Compile your own damn shaders! LazyVulkan is just as lazy as you are!
static FRAGMENT_SHADER: &'static [u8] = include_bytes!("shaders/triangle.frag.spv");
static VERTEX_SHADER: &'static [u8] = include_bytes!("shaders/triangle.vert.spv");

pub fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

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
    let (mut lazy_vulkan, mut lazy_renderer, mut event_loop) = LazyVulkan::builder()
        .initial_vertices(&vertices)
        .initial_indices(&indices)
        .fragment_shader(FRAGMENT_SHADER)
        .vertex_shader(VERTEX_SHADER)
        .with_present(true)
        .build();

    // Let's do something totally normal and wait for a TCP connection
    let listener = std::net::TcpListener::bind("127.0.0.1:8000").unwrap();
    info!("Listening on 0:8000 - waiting for connection");
    let swapchain_info = SwapchainInfo {
        image_count: lazy_vulkan.surface.desired_image_count,
        resolution: lazy_vulkan.surface.surface_resolution,
        format: lazy_vulkan.surface.surface_format.format,
    };
    let memory_handles = unsafe { create_memory_handles(lazy_vulkan.context(), &swapchain_info) };
    let (mut socket, _) = listener.accept().unwrap();
    let mut buf: [u8; 1024] = [0; 1024];
    send_swapchain_info(&mut socket, &swapchain_info, &mut buf).unwrap();
    send_memory_handles(&mut socket, memory_handles, &mut buf).unwrap();

    // Off we go!
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
                send_swapchain_image_index(&mut socket, &mut buf, framebuffer_index);
                lazy_renderer.render(
                    &lazy_vulkan.context(),
                    framebuffer_index,
                    &[DrawCall::new(0, 3, NO_TEXTURE_ID, Workflow::Main)],
                );
                lazy_vulkan.render_end(framebuffer_index);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if winit_initializing {
                    println!("Ignoring resize during init!");
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

fn send_swapchain_image_index(
    socket: &mut std::net::TcpStream,
    buf: &mut [u8; 1024],
    framebuffer_index: u32,
) {
    socket.read(buf).unwrap();
    socket.write(&mut [framebuffer_index as u8]).unwrap();
}

unsafe fn create_memory_handles(
    context: &lazy_vulkan::vulkan_context::VulkanContext,
    swapchain_info: &SwapchainInfo,
) -> Vec<vk::HANDLE> {
    let device = &context.device;
    let SwapchainInfo {
        resolution,
        format,
        image_count,
    } = swapchain_info;

    (0..(*image_count))
        .map(|_| {
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
            let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
            let mut export_handle_info =
                vk::ExportMemoryAllocateInfo::builder().handle_types(handle_type);
            let memory = context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(memory_requirements.size)
                        .memory_type_index(memory_index)
                        .push_next(&mut export_handle_info),
                    None,
                )
                .unwrap();

            let external_memory =
                ash::extensions::khr::ExternalMemoryWin32::new(&context.instance, &context.device);
            let handle = external_memory
                .get_memory_win32_handle(
                    &vk::MemoryGetWin32HandleInfoKHR::builder()
                        .handle_type(handle_type)
                        .memory(memory),
                )
                .unwrap();
            info!("Created handle {handle:?}");

            handle
        })
        .collect()
}

fn send_swapchain_info(
    socket: &mut std::net::TcpStream,
    swapchain_info: &SwapchainInfo,
    buf: &mut [u8],
) -> std::io::Result<()> {
    info!("Processing connection..");
    socket.read(buf)?;
    let value = buf[0];
    info!("Read {value}");

    if value == 0 {
        let write = socket.write(bytes_of(swapchain_info)).unwrap();
        info!("Write {write} bytes");
        return Ok(());
    } else {
        panic!("Invalid request!");
    }
}

fn send_memory_handles(
    socket: &mut std::net::TcpStream,
    handles: Vec<vk::HANDLE>,
    buf: &mut [u8],
) -> std::io::Result<()> {
    info!("Processing connection..");
    socket.read(buf)?;
    let value = buf[0];
    info!("Read {value}");

    if value == 1 {
        info!("Sending handle: {handles:?}");
        let write = socket.write(bytes_of_slice(&handles)).unwrap();
        info!("Write {write} bytes");
        return Ok(());
    } else {
        panic!("Invalid request!");
    }
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
