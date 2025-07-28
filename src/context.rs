use ash::vk::{self, MemoryRequirements};

use super::core::Core;

pub struct Context {
    pub device: ash::Device,
    #[allow(unused)]
    pub command_pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub graphics_queue: vk::Queue,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_type: vk::PhysicalDeviceType,
    pub device_properties: vk::PhysicalDeviceProperties,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub dynamic_rendering_pfn: ash::khr::dynamic_rendering::Device,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub sync2_pfn: ash::khr::synchronization2::Device,
}

impl Context {
    pub(crate) fn new(core: &Core) -> Self {
        let instance = &core.instance;
        let physical_device = core.physical_device;

        let device = create_device(instance, physical_device);

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(0)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .unwrap();

        let draw_command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1),
            )
        }
        .unwrap()[0];

        let graphics_queue = unsafe { device.get_device_queue(0, 0) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let dynamic_rendering_pfn =
            ash::khr::dynamic_rendering::Device::new(&core.instance, &device);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let sync2_pfn = ash::khr::synchronization2::Device::new(&core.instance, &device);

        Self {
            device,
            command_pool,
            draw_command_buffer,
            graphics_queue,
            memory_properties,
            device_type: physical_device_properties.device_type,
            device_properties: physical_device_properties,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            dynamic_rendering_pfn,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            sync2_pfn,
        }
    }

    pub fn find_memory_type_index(
        &self,
        requirements: &MemoryRequirements,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_props = self.memory_properties;
        for i in 0..mem_props.memory_type_count {
            if (requirements.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return Some(i);
            }
        }
        None
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfo,
    ) {
        self.device
            .cmd_pipeline_barrier2(command_buffer, dependency_info);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfo,
    ) {
        self.sync2_pfn
            .cmd_pipeline_barrier2(command_buffer, dependency_info);
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub unsafe fn cmd_begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        rendering_info: &vk::RenderingInfo,
    ) {
        self.device
            .cmd_begin_rendering(command_buffer, rendering_info);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn cmd_begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        rendering_info: &vk::RenderingInfo,
    ) {
        self.dynamic_rendering_pfn
            .cmd_begin_rendering(command_buffer, rendering_info);
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub unsafe fn cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) {
        self.context.device.cmd_end_rendering(command_buffer);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn cmd_end_rendering(&self, command_buffer: vk::CommandBuffer) {
        self.dynamic_rendering_pfn.cmd_end_rendering(command_buffer);
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub unsafe fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2KHR],
        fence: vk::Fence,
    ) {
        self.context
            .device
            .queue_submit2(queue, submits, fence)
            .unwrap()
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2KHR],
        fence: vk::Fence,
    ) {
        self.sync2_pfn.queue_submit2(queue, submits, fence).unwrap()
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn create_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> ash::Device {
    let enabled_extension_names = [
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::portability_subset::NAME.as_ptr(),
        ash::khr::dynamic_rendering::NAME.as_ptr(),
        ash::khr::synchronization2::NAME.as_ptr(),
    ];

    let device = unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .enabled_extension_names(&enabled_extension_names)
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(0)
                    .queue_priorities(&[1.0])])
                .enabled_features(
                    &vk::PhysicalDeviceFeatures::default()
                        .fill_mode_non_solid(true)
                        .sampler_anisotropy(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceDynamicRenderingFeatures::default()
                        .dynamic_rendering(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceSynchronization2Features::default()
                        .synchronization2(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan12Features::default()
                        .runtime_descriptor_array(true)
                        .descriptor_indexing(true)
                        .descriptor_binding_partially_bound(true)
                        .descriptor_binding_sampled_image_update_after_bind(true)
                        .shader_sampled_image_array_non_uniform_indexing(true)
                        .buffer_device_address(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan11Features::default()
                        .variable_pointers(true)
                        .variable_pointers_storage_buffer(true),
                ),
            None,
        )
    }
    .unwrap();
    device
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn create_device(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> ash::Device {
    let enabled_extension_names = [ash::khr::swapchain::NAME.as_ptr()];

    let device = unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .enabled_extension_names(&enabled_extension_names)
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(0)
                    .queue_priorities(&[1.0])])
                .enabled_features(
                    &vk::PhysicalDeviceFeatures::default()
                        .fill_mode_non_solid(true)
                        .shader_anisotropy(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan13Features::default()
                        .dynamic_rendering(true)
                        .synchronization2(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan12Features::default()
                        .runtime_descriptor_array(true)
                        .descriptor_indexing(true)
                        .descriptor_binding_partially_bound(true)
                        .descriptor_binding_sampled_image_update_after_bind(true)
                        .shader_sampled_image_array_non_uniform_indexing(true)
                        .buffer_device_address(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan11Features::default()
                        .variable_pointers(true)
                        .variable_pointers_storage_buffer(true),
                ),
            None,
        )
    }
    .unwrap();
    device
}
