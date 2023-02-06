use ash::vk;

use crate::{descriptors::Descriptors, vulkan_context::VulkanContext};

pub const NO_TEXTURE_ID: u32 = u32::MAX;

pub struct VulkanTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub sampler: vk::Sampler,
    pub view: vk::ImageView,
    pub id: u32,
}

/// A container for information about a texture.
/// Populate this struct and pass it to [`crate::YakuiVulkan::add_user_texture()`] to create user managed
/// textures that you can then use in [`yakui`] code.
pub struct VulkanTextureCreateInfo<T: AsRef<[u8]>> {
    image_data: T,
    format: vk::Format,
    resolution: vk::Extent2D,
    min_filter: vk::Filter,
    mag_filter: vk::Filter,
}

impl<T: AsRef<[u8]>> VulkanTextureCreateInfo<T> {
    /// Construct a new [`VulkanTextureCreateInfo`] wrapper. Ensure `image_data` refers to an image that matches
    /// the rest of the parameters.
    pub fn new(
        image_data: T,
        format: vk::Format,
        resolution: vk::Extent2D,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
    ) -> Self {
        Self {
            image_data,
            format,
            resolution,
            min_filter,
            mag_filter,
        }
    }
}

impl VulkanTexture {
    pub fn new<T: AsRef<[u8]>>(
        vulkan_context: &VulkanContext,
        descriptors: &mut Descriptors,
        create_info: VulkanTextureCreateInfo<T>,
    ) -> Self {
        let VulkanTextureCreateInfo {
            image_data,
            format,
            resolution,
            min_filter,
            mag_filter,
        } = create_info;

        let address_mode = vk::SamplerAddressMode::REPEAT;
        let (image, memory) =
            unsafe { vulkan_context.create_image(image_data.as_ref(), resolution, format) };
        let view = unsafe { vulkan_context.create_image_view(image, format) };

        let sampler = unsafe {
            vulkan_context
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .address_mode_u(address_mode)
                        .address_mode_v(address_mode)
                        .address_mode_w(address_mode)
                        .mag_filter(mag_filter)
                        .min_filter(min_filter),
                    None,
                )
                .unwrap()
        };

        let id =
            unsafe { descriptors.update_texture_descriptor_set(view, sampler, vulkan_context) };

        VulkanTexture {
            image,
            memory,
            view,
            sampler,
            id,
        }
    }

    pub unsafe fn cleanup(&self, device: &ash::Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}
