use std::{collections::HashMap, sync::Arc};

use ash::vk;

use crate::{descriptors::Descriptors, Allocator, Context, TransferToken, FULL_IMAGE};

#[derive(Debug, Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub mip_levels: u32,
    pub sampler: vk::Sampler,
    pub id: u32,
    pub transfer_complete: TransferToken,
}

const NO_TEXTURE_ID: u32 = std::u32::MAX;

pub struct ImageManager {
    context: Arc<Context>,
    current_id: u32,
    free_ids: Vec<u32>,
    attachments: HashMap<vk::Image, Image>,
    texture_descriptor_set: vk::DescriptorSet,
}

impl ImageManager {
    pub fn new(context: Arc<Context>, texture_descriptor_set: vk::DescriptorSet) -> ImageManager {
        ImageManager {
            context,
            current_id: 0,
            free_ids: Vec::new(),
            attachments: HashMap::new(),
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
    /// Does not yet support multiple image layers.
    /// Depth/stencil attachments are allocated individually and must have no upload data.
    pub fn create_image(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
    ) -> Image {
        self.create_image_inner(
            name,
            allocator,
            format,
            extent,
            image_bytes,
            image_usage_flags,
            1,
            None,
            vk::SampleCountFlags::TYPE_1,
        )
    }

    pub fn create_mipmapped_sampled_image(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
    ) -> Image {
        let mip_levels = mip_level_count(extent);
        self.create_image_inner(
            name,
            allocator,
            format,
            extent,
            image_bytes,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            mip_levels,
            None,
            vk::SampleCountFlags::TYPE_1,
        )
    }

    /// Upload supplied mip levels verbatim (including BCn blocks); never blit compressed data.
    pub fn create_sampled_image_with_mips(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        levels: &[&[u8]],
    ) -> Image {
        assert!(extent.width > 0 && extent.height > 0);
        assert!(!levels.is_empty() && levels.len() <= mip_level_count(extent) as usize);
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(levels.len());
        for level in levels {
            assert!(!level.is_empty());
            bytes.resize((bytes.len() + 15) & !15, 0);
            offsets.push(bytes.len() as vk::DeviceSize);
            bytes.extend_from_slice(level);
        }
        self.create_image_inner(
            name,
            allocator,
            format,
            extent,
            &bytes,
            vk::ImageUsageFlags::SAMPLED,
            levels.len() as u32,
            Some(offsets),
            vk::SampleCountFlags::TYPE_1,
        )
    }

    /// Creates an attachment without upload data or mipmaps.
    pub fn create_attachment(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
    ) -> Image {
        assert!(
            samples.as_raw().is_power_of_two(),
            "expected one sample count"
        );
        assert!(
            samples == vk::SampleCountFlags::TYPE_1
                || !usage.contains(vk::ImageUsageFlags::SAMPLED),
            "multisampled images cannot use the ordinary texture descriptors"
        );
        let image = self.create_image_inner(
            name,
            allocator,
            format,
            extent,
            &[],
            usage,
            1,
            None,
            samples,
        );
        self.attachments.insert(image.handle, image.clone());
        image
    }

    // The submission fence must cover every use, including borrowed views.
    pub(crate) unsafe fn destroy_attachment(
        &mut self,
        handle: vk::Image,
        allocator: &mut Allocator,
    ) {
        let image = self
            .attachments
            .remove(&handle)
            .expect("owned render attachment");
        let device = &self.context.device;
        device.destroy_image_view(image.view, None);
        device.destroy_sampler(image.sampler, None);
        device.destroy_image(image.handle, None);
        allocator.release_image_memory(image.handle);
        if image.id != NO_TEXTURE_ID {
            self.free_ids.push(image.id);
        }
    }

    fn create_image_inner(
        &mut self,
        name: impl AsRef<str>,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
        mip_levels: u32,
        mip_offsets: Option<Vec<vk::DeviceSize>>,
        samples: vk::SampleCountFlags,
    ) -> Image {
        let id = if image_usage_flags.contains(vk::ImageUsageFlags::SAMPLED) {
            self.allocate_id()
        } else {
            NO_TEXTURE_ID
        };

        let device = &self.context.device;
        let image_bytes = image_bytes.as_ref();
        let is_depth_attachment =
            image_usage_flags.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
        assert!(
            !is_depth_attachment || image_bytes.is_empty(),
            "Depth/stencil attachment uploads are not supported"
        );

        let handle = unsafe {
            device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(extent.into())
                        .mip_levels(mip_levels)
                        .array_layers(1)
                        .samples(samples)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(image_usage_flags | vk::ImageUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .unwrap()
        };

        self.context.set_debug_label(handle, name.as_ref());

        let transfer_complete = if is_depth_attachment || samples != vk::SampleCountFlags::TYPE_1 {
            allocator.allocate_dedicated_image(handle)
        } else {
            allocator.allocate_image_with_mip_data(
                image_bytes,
                extent,
                mip_levels,
                handle,
                mip_offsets,
            )
        };

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
            let is_shadow_map = image_usage_flags.contains(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            );
            sampler = unsafe {
                let mut sampler_create_info = vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod((mip_levels - 1) as f32)
                    .address_mode_u(if is_shadow_map {
                        vk::SamplerAddressMode::CLAMP_TO_BORDER
                    } else {
                        vk::SamplerAddressMode::REPEAT
                    })
                    .address_mode_v(if is_shadow_map {
                        vk::SamplerAddressMode::CLAMP_TO_BORDER
                    } else {
                        vk::SamplerAddressMode::REPEAT
                    })
                    .address_mode_w(if is_shadow_map {
                        vk::SamplerAddressMode::CLAMP_TO_BORDER
                    } else {
                        vk::SamplerAddressMode::REPEAT
                    })
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
                    .anisotropy_enable(!is_shadow_map)
                    .max_anisotropy(self.context.device_properties.limits.max_sampler_anisotropy);

                // This is a little bit hacky, but reasonable. It doesn't really make a lot of
                // sense to be creating an image with DEPTH_STENCIL and SAMPLED, but you don't
                // want to use it as a shadow map.
                if is_shadow_map {
                    sampler_create_info.compare_op = vk::CompareOp::GREATER_OR_EQUAL;
                    sampler_create_info.compare_enable = vk::TRUE;
                }

                device.create_sampler(&sampler_create_info, None)
            }
            .unwrap();
            unsafe {
                self.update_texture_descriptor_set(
                    sampled_descriptor_binding(image_usage_flags),
                    id,
                    view,
                    sampler,
                    sampled_descriptor_layout(image_usage_flags),
                )
            };
        }

        Image {
            handle,
            view,
            extent,
            mip_levels,
            id,
            sampler,
            transfer_complete,
        }
    }

    pub unsafe fn update_texture_descriptor_set(
        &self,
        binding: u32,
        texture_id: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) {
        self.context.device.update_descriptor_sets(
            std::slice::from_ref(
                &vk::WriteDescriptorSet::default()
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .sampler(sampler)
                            .image_view(image_view)
                            .image_layout(image_layout),
                    ))
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_array_element(texture_id)
                    .dst_binding(binding)
                    .dst_set(self.texture_descriptor_set),
            ),
            &[],
        );
    }

    fn allocate_id(&mut self) -> u32 {
        if let Some(id) = self.free_ids.pop() {
            return id;
        }
        let id = self.current_id;
        self.current_id += 1;
        id
    }
}

fn mip_level_count(extent: vk::Extent2D) -> u32 {
    assert!(extent.width > 0 && extent.height > 0);
    extent.width.max(extent.height).ilog2() + 1
}

fn sampled_descriptor_layout(usage: vk::ImageUsageFlags) -> vk::ImageLayout {
    if usage.contains(vk::ImageUsageFlags::STORAGE) {
        vk::ImageLayout::GENERAL
    } else if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    }
}

fn sampled_descriptor_binding(usage: vk::ImageUsageFlags) -> u32 {
    if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        Descriptors::SHADOW_TEXTURE_BINDING
    } else {
        Descriptors::TEXTURE_BINDING
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_mip_chain_reaches_one_by_one() {
        assert_eq!(
            mip_level_count(vk::Extent2D {
                width: 2048,
                height: 2048,
            }),
            12
        );
        assert_eq!(
            mip_level_count(vk::Extent2D {
                width: 1024,
                height: 256,
            }),
            11
        );
        assert_eq!(
            mip_level_count(vk::Extent2D {
                width: 1,
                height: 1,
            }),
            1
        );
    }

    #[test]
    fn sampled_images_use_the_read_only_layout() {
        assert_eq!(
            sampled_descriptor_layout(vk::ImageUsageFlags::SAMPLED),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn sampled_storage_images_use_the_general_layout() {
        assert_eq!(
            sampled_descriptor_layout(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            vk::ImageLayout::GENERAL
        );
    }

    #[test]
    fn sampled_depth_images_use_the_depth_read_only_layout() {
        assert_eq!(
            sampled_descriptor_layout(
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            ),
            vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn sampled_depth_images_use_the_shadow_binding() {
        assert_eq!(
            sampled_descriptor_binding(
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            ),
            Descriptors::SHADOW_TEXTURE_BINDING
        );
        assert_eq!(
            sampled_descriptor_binding(vk::ImageUsageFlags::SAMPLED),
            Descriptors::TEXTURE_BINDING
        );
    }
}
