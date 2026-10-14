use std::sync::Arc;

use ash::vk;

use crate::Context;

/// An owned, device-local color attachment for resolving into a single-sample image.
pub struct MultisampleColorAttachment {
    context: Arc<Context>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    memory: vk::DeviceMemory,
    pub extent: vk::Extent2D,
    format: vk::Format,
    samples: vk::SampleCountFlags,
}

impl MultisampleColorAttachment {
    pub fn new(
        context: Arc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Self {
        assert!(samples.as_raw().is_power_of_two() && samples != vk::SampleCountFlags::TYPE_1);
        let device = &context.device;
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(extent.into())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(samples)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .expect("failed to create multisample color attachment");
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = context
            .find_memory_type_index(&requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("no device-local memory for multisample color attachment");
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )
        }
        .expect("failed to allocate multisample color attachment");
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    }),
                None,
            )
        }
        .expect("failed to create multisample color view");
        Self {
            context,
            image,
            view,
            memory,
            extent,
            format,
            samples,
        }
    }

    pub fn resize(&mut self, extent: vk::Extent2D) {
        if self.extent == extent {
            return;
        }
        *self = Self::new(self.context.clone(), extent, self.format, self.samples);
    }
}

impl Drop for MultisampleColorAttachment {
    fn drop(&mut self) {
        // Resizing and shutdown may retire an attachment still used by the last frame.
        unsafe {
            let device = &self.context.device;
            device.device_wait_idle().unwrap();
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
