use ash::vk;

use super::{context::Context, swapchain::Swapchain};

#[derive(Debug, Copy, Clone)]
pub struct DepthBuffer {
    pub image: vk::Image,
    pub view: vk::ImageView,
    #[allow(unused)]
    pub memory: vk::DeviceMemory,
}

impl DepthBuffer {
    pub(crate) fn new(context: &Context, swapchain: &Swapchain) -> Self {
        let device = &context.device;
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .array_layers(1)
                    .mip_levels(1)
                    .image_type(vk::ImageType::TYPE_2D)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .extent(swapchain.extent.into())
                    .format(DEPTH_FORMAT),
                None,
            )
        }
        .unwrap();

        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };

        let memory_type_index = context
            .find_memory_type_index(&memory_requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("No memory type index for depth buffer - impossible");

        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )
        }
        .expect("Failed to allocate memory - impossible");

        unsafe {
            device.bind_image_memory2(&[vk::BindImageMemoryInfo::default()
                .image(image)
                .memory(memory)])
        }
        .unwrap();

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(DEPTH_FORMAT)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(DEPTH_RANGE),
                None,
            )
        }
        .unwrap();

        Self {
            image,
            view,
            memory,
        }
    }
}

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub const DEPTH_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::DEPTH,
    layer_count: 1,
    level_count: 1,
    base_mip_level: 0,
    base_array_layer: 0,
};
