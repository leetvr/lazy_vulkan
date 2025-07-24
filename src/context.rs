use ash::vk::{self, MemoryRequirements};

use super::core::Core;

pub struct Context {
    pub device: ash::Device,
    #[allow(unused)]
    pub command_pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,
    pub graphics_queue: vk::Queue,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Context {
    pub(crate) fn new(core: &Core) -> Self {
        let instance = &core.instance;
        let physical_device = core.physical_device;

        #[allow(unused_mut)]
        let mut enabled_extension_names = vec![ash::khr::swapchain::NAME.as_ptr()];

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        enabled_extension_names.push(ash::khr::portability_subset::NAME.as_ptr());

        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .enabled_extension_names(&enabled_extension_names)
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(0)
                        .queue_priorities(&[1.0])])
                    .enabled_features(
                        &vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan13Features::default()
                            .dynamic_rendering(true)
                            .synchronization2(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan12Features::default()
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

        Self {
            device,
            command_pool,
            draw_command_buffer,
            graphics_queue,
            memory_properties,
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
}
