use std::sync::Arc;

use ash::vk;

use crate::{descriptors::Descriptors, Allocator, Context, TransferToken, FULL_IMAGE};

#[derive(Debug, Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub sampler: vk::Sampler,
    pub id: u32,
    pub transfer_complete: TransferToken,
}

const NO_TEXTURE_ID: u32 = std::u32::MAX;

pub struct ImageManager {
    context: Arc<Context>,
    current_id: u32,
    texture_descriptor_set: vk::DescriptorSet,
}

impl ImageManager {
    pub fn new(context: Arc<Context>, texture_descriptor_set: vk::DescriptorSet) -> ImageManager {
        ImageManager {
            context,
            current_id: 0,
            texture_descriptor_set,
        }
    }

    /// Tries to Do What You Want:
    ///
    /// - If there's some data in `image_bytes`, we'll schedule a transfer to put the bytes in the
    ///   image as you intended
    /// - If `image_usage_flags` contains the SAMPLED flag, we'll create a sampler, allocate a
    ///   texture ID and then write that to the "all the images" descriptor set.
    /// - If `image_usage_flags` contains both SAMPLED and DEPTH_STENCIL_ATTACHMENT, we'll assume
    ///   this is a shadowmap image and set the compare ops on the sampler accordingly.
    /// - If `format` is a depth format, we'll set the correct aspect flags on the iamge view
    ///
    /// Does not yet support mipmaps or multiple image layers.
    pub fn create_image(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
    ) -> Image {
        let id = if image_usage_flags.contains(vk::ImageUsageFlags::SAMPLED) {
            self.allocate_id()
        } else {
            NO_TEXTURE_ID
        };

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

        self.context.set_debug_label(handle, name.as_ref());

        let transfer_complete = allocator.allocate_image(image_bytes, extent, handle);

        let view = unsafe {
            // Another little hack.
            //
            // If the image is in a depth format, then set the depth flags
            let mut subresource_range = FULL_IMAGE;
            if format == vk::Format::D32_SFLOAT {
                subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;
            }

            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(subresource_range),
                None,
            )
        }
        .unwrap();

        let mut sampler = vk::Sampler::null();

        if image_usage_flags.contains(vk::ImageUsageFlags::SAMPLED) {
            sampler = unsafe {
                let mut sampler_create_info = vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(self.context.device_properties.limits.max_sampler_anisotropy);

                // This is a little bit hacky, but reasonable. It doesn't really make a lot of
                // sense to be creating an image with DEPTH_STENCIL and SAMPLED, but you don't
                // want to use it as a shadow map.
                if image_usage_flags.contains(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                ) {
                    sampler_create_info.compare_op = vk::CompareOp::LESS_OR_EQUAL;
                    sampler_create_info.compare_enable = vk::TRUE;
                }

                device.create_sampler(&sampler_create_info, None)
            }
            .unwrap();
            unsafe { self.update_texture_descriptor_set(id, view, sampler) };
        }

        Image {
            handle,
            view,
            extent,
            id,
            sampler,
            transfer_complete,
        }
    }

    pub unsafe fn update_texture_descriptor_set(
        &self,
        texture_id: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        self.context.device.update_descriptor_sets(
            std::slice::from_ref(
                &vk::WriteDescriptorSet::default()
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .sampler(sampler)
                            .image_view(image_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    ))
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(texture_id)
                    .dst_binding(Descriptors::TEXTURE_BINDING)
                    .dst_set(self.texture_descriptor_set),
            ),
            &[],
        );
    }

    fn allocate_id(&mut self) -> u32 {
        let id = self.current_id;
        self.current_id += 1;
        id
    }
}
