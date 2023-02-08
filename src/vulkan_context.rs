use ash::vk;
use log::info;
use winit::window::Window;

use crate::{buffer::Buffer, find_memorytype_index, Surface};

#[derive(Clone)]
/// A wrapper around handles into your Vulkan renderer to call various methods on [`crate::YakuiVulkan`]
///
/// ## Safety
/// It is **very** important that you pass the correct handles to this struct, or you will have a terrible time.
/// Once you create a [`crate::YakuiVulkan`] instance, you **must** use the same handles each time you call a
/// method on that instance.
///
/// See the documentation on each member for specific safety requirements.
pub struct VulkanContext {
    pub _entry: ash::Entry,
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    /// A queue that can call render and transfer commands
    pub queue: vk::Queue,
    /// The command buffer that you'll ultimately submit to be presented/rendered
    pub draw_command_buffer: vk::CommandBuffer,
    pub command_pool: vk::CommandPool,
    /// Memory properties used for [`crate::YakuiVulkan`]'s allocation commands
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanContext {
    pub fn new() -> Self {
        Self::new_with_extensions(&[], &[])
    }

    pub fn new_with_extensions(
        instance_extensions: &[*const std::ffi::c_char],
        device_extensions: &[*const std::ffi::c_char],
    ) -> Self {
        let (entry, instance) = init(&mut instance_extensions.to_vec());
        let (physical_device, queue_family_index) = get_physical_device(&instance, None, None);

        let queue_family_index = queue_family_index as u32;
        let device = create_device(
            queue_family_index,
            &mut device_extensions.to_vec(),
            &instance,
            physical_device,
        );
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let (command_pool, draw_command_buffer) = create_command_pool(queue_family_index, &device);
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        VulkanContext {
            _entry: entry,
            instance,
            device,
            physical_device,
            queue,
            draw_command_buffer,
            command_pool,
            memory_properties,
        }
    }

    pub fn new_with_surface(window: &Window, window_resolution: vk::Extent2D) -> (Self, Surface) {
        use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();

        let (entry, instance) = init(&mut extension_names);

        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .unwrap()
        };

        let (physical_device, queue_family_index) =
            get_physical_device(&instance, Some(&surface_loader), Some(&surface));

        let queue_family_index = queue_family_index as u32;
        let mut device_extension_names_raw = vec![ash::extensions::khr::Swapchain::name().as_ptr()];

        let device = create_device(
            queue_family_index,
            &mut device_extension_names_raw,
            &instance,
            physical_device,
        );

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()[0]
        };

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };
        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => window_resolution,
            _ => surface_capabilities.current_extent,
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
        };
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let (pool, draw_command_buffer) = create_command_pool(queue_family_index, &device);
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let context = VulkanContext {
            _entry: entry,
            device,
            physical_device,
            instance,
            queue: present_queue,
            draw_command_buffer,
            command_pool: pool,
            memory_properties: device_memory_properties,
        };

        let surface = Surface {
            surface,
            surface_loader,
            surface_format,
            surface_resolution,
            present_mode,
            desired_image_count,
        };

        (context, surface)
    }

    pub(crate) unsafe fn create_image(
        &self,
        image_data: &[u8],
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> (vk::Image, vk::DeviceMemory) {
        let scratch_buffer = Buffer::new(&self, vk::BufferUsageFlags::TRANSFER_SRC, image_data);
        let device = &self.device;

        let image = device
            .create_image(
                &vk::ImageCreateInfo {
                    image_type: vk::ImageType::TYPE_2D,
                    format,
                    extent: extent.into(),
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
            &self.memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory type for image");
        let image_memory = self.allocate_memory(memory_requirements.size, memory_index);
        device.bind_image_memory(image, image_memory, 0).unwrap();

        self.one_time_command(|command_buffer| {
            let transfer_barrier = vk::ImageMemoryBarrier {
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[transfer_barrier],
            );
            let buffer_copy_regions = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                image_extent: extent.into(),
                ..Default::default()
            };

            device.cmd_copy_buffer_to_image(
                command_buffer,
                scratch_buffer.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&buffer_copy_regions),
            );

            let transition_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&transition_barrier),
            )
        });

        scratch_buffer.cleanup(device);

        (image, image_memory)
    }

    unsafe fn one_time_command<F: FnOnce(vk::CommandBuffer)>(&self, f: F) {
        let device = &self.device;
        let command_buffer = device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                command_buffer_count: 1,
                level: vk::CommandBufferLevel::PRIMARY,
                ..Default::default()
            })
            .unwrap()[0];

        let fence = device.create_fence(&Default::default(), None).unwrap();

        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
            .unwrap();

        f(command_buffer);

        device.end_command_buffer(command_buffer).unwrap();

        let submit_info =
            vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
        device
            .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
            .unwrap();
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .unwrap();

        device.destroy_fence(fence, None);
        device.free_command_buffers(self.command_pool, std::slice::from_ref(&command_buffer));
    }

    unsafe fn allocate_memory(
        &self,
        allocation_size: vk::DeviceSize,
        memory_type_index: u32,
    ) -> vk::DeviceMemory {
        self.device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size,
                    memory_type_index,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
    }

    pub unsafe fn create_image_view(&self, image: vk::Image, format: vk::Format) -> vk::ImageView {
        self.device
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image,
                    format,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    view_type: vk::ImageViewType::TYPE_2D,
                    ..Default::default()
                },
                None,
            )
            .unwrap()
    }
}

pub fn create_command_pool(
    queue_family_index: u32,
    device: &ash::Device,
) -> (vk::CommandPool, vk::CommandBuffer) {
    let pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);
    let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    };
    let draw_command_buffer = command_buffers[0];
    (pool, draw_command_buffer)
}

pub fn create_device(
    queue_family_index: u32,
    extension_names: &mut Vec<*const std::ffi::c_char>,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> ash::Device {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    extension_names.push(KhrPortabilitySubsetFn::name().as_ptr());

    #[cfg(target_os = "windows")]
    extension_names.push(ash::extensions::khr::ExternalMemoryWin32::name().as_ptr());

    #[cfg(target_os = "windows")]
    extension_names.push(ash::extensions::khr::ExternalSemaphoreWin32::name().as_ptr());

    let priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
        .descriptor_binding_partially_bound(true);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&extension_names)
        .push_next(&mut descriptor_indexing_features);

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    };
    device
}

pub fn get_physical_device(
    instance: &ash::Instance,
    surface_loader: Option<&ash::extensions::khr::Surface>,
    surface: Option<&vk::SurfaceKHR>,
) -> (vk::PhysicalDevice, u32) {
    let pdevices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Physical device error")
    };
    let (physical_device, queue_family_index) = unsafe {
        pdevices
            .iter()
            .find_map(|pdevice| {
                instance
                    .get_physical_device_queue_family_properties(*pdevice)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let supports_graphics_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                // blegh
                                && surface_loader
                                    .map(|l| {
                                        l.get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            *surface.unwrap(),
                                        )
                                        .unwrap()
                                    })
                                    .unwrap_or(true);
                        if supports_graphics_and_surface {
                            Some((*pdevice, index))
                        } else {
                            None
                        }
                    })
            })
            .expect("Couldn't find suitable device.")
    };
    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };
    let physical_device_name = unsafe {
        let device_name_raw = std::slice::from_raw_parts(
            &physical_device_properties.device_name as *const _ as *const u8,
            256,
        );
        std::ffi::CStr::from_bytes_with_nul_unchecked(&device_name_raw)
            .to_str()
            .unwrap()
    };
    info!("Using device {physical_device_name}");
    (physical_device, queue_family_index as u32)
}

pub fn init(extension_names: &mut Vec<*const std::ffi::c_char>) -> (ash::Entry, ash::Instance) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(KhrPortabilityEnumerationFn::name().as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extension_names.push(KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    }

    let entry = ash::Entry::linked();
    let app_name = unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"Lazy Vulkan\0") };

    let appinfo = vk::ApplicationInfo::builder()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 2, 0));

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let extensions_list: String = extension_names
        .iter()
        .map(|extension| {
            unsafe { std::ffi::CStr::from_ptr(*extension) }
                .to_str()
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(", ");
    info!("Using extensions {extensions_list}");

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .flags(create_flags)
        .enabled_extension_names(&extension_names);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Instance creation error")
    };

    (entry, instance)
}
