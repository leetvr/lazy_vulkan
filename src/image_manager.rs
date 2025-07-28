use std::sync::Arc;

use ash::vk;

use crate::{descriptors::Descriptors, Allocator, Context, TransferToken, FULL_IMAGE};

pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub sampler: vk::Sampler,
    pub id: u32,
    pub transfer_complete: TransferToken,
}

pub struct ImageManager {
    context: Arc<Context>,
    current_id: u32,
}

impl ImageManager {
    pub fn new(context: Arc<Context>) -> ImageManager {
        ImageManager {
            context,
            current_id: 0,
        }
    }

    pub fn create_image(
        &mut self,
        allocator: &mut Allocator,
        descriptors: &mut Descriptors,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
    ) -> Image {
        let device = &self.context.device;
        let image_bytes = image_bytes.as_ref();

        let handle = unsafe {
            device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(image_usage_flags | vk::ImageUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .unwrap()
        };

        let transfer_complete = allocator.allocate_image(image_bytes, extent, handle);

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(FULL_IMAGE),
                None,
            )
        }
        .unwrap();

        let max_anisotropy = self.context.device_properties.limits.max_sampler_anisotropy;

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(max_anisotropy),
                None,
            )
        }
        .unwrap();

        let id = self.allocate_id();
        unsafe { descriptors.update_texture_descriptor_set(id, view, sampler) };

        Image {
            handle,
            view,
            extent,
            id,
            sampler,
            transfer_complete,
        }
    }

    fn allocate_id(&mut self) -> u32 {
        let id = self.current_id;
        self.current_id += 1;
        id
    }
}
