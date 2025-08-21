use crate::{swapchain::Drawable, Context, FULL_IMAGE};
use ash::vk::{self};
use std::sync::Arc;

pub struct HeadlessSwapchain {
    pub context: Arc<Context>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub image: HeadlessSwapchainImage,
    render_complete: vk::Semaphore,
}

impl HeadlessSwapchain {
    pub(crate) fn new(context: Arc<Context>, extent: vk::Extent2D, format: vk::Format) -> Self {
        let image = HeadlessSwapchainImage::new(&context, extent, format);
        let render_complete = unsafe {
            context
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .unwrap();

        Self {
            context,
            extent,
            format,
            image,
            render_complete,
        }
    }

    pub(crate) fn resize(&mut self, new_extent: vk::Extent2D) {
        if self.extent == new_extent {
            return;
        }

        self.extent = new_extent;
        self.image.resize(&self.context, self.extent, self.format);
    }

    pub(crate) fn get_drawable(&self) -> Drawable {
        Drawable {
            image: self.image.image,
            view: self.image.view,
            image_available: None,
            rendering_complete: self.render_complete,
            index: 0,
            extent: self.extent,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeadlessSwapchainImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

impl HeadlessSwapchainImage {
    fn new(context: &Context, extent: vk::Extent2D, format: vk::Format) -> HeadlessSwapchainImage {
        let device = &context.device;

        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .array_layers(1)
                    .mip_levels(1)
                    .image_type(vk::ImageType::TYPE_2D)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .extent(extent.into())
                    .format(format),
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
                    .format(format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(FULL_IMAGE),
                None,
            )
        }
        .unwrap();

        HeadlessSwapchainImage {
            image,
            memory,
            view,
        }
    }

    pub fn resize(&mut self, context: &Context, new_extent: vk::Extent2D, format: vk::Format) {
        let device = &context.device;
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }

        *self = Self::new(context, new_extent, format);
        log::debug!("Resized! Image: {:?}", self.image);
    }
}
