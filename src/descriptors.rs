use std::sync::Arc;

use ash::vk;

use crate::Context;

pub struct Descriptors {
    context: Arc<Context>,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
    pub layout: vk::DescriptorSetLayout,
}

impl Descriptors {
    pub const TEXTURE_BINDING: u32 = 0;

    pub fn new(context: Arc<Context>) -> Descriptors {
        let device = &context.device;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1000,
        }];

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )
        }
        .unwrap();

        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[
                        // Textures
                        vk::DescriptorSetLayoutBinding {
                            binding: Self::TEXTURE_BINDING,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            stage_flags: vk::ShaderStageFlags::COMPUTE
                                | vk::ShaderStageFlags::FRAGMENT,
                            descriptor_count: 1000,
                            ..Default::default()
                        },
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )
        }
        .unwrap();

        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(std::slice::from_ref(&layout)),
                )
                .unwrap()[0]
        };

        Descriptors {
            context,
            pool,
            set,
            layout,
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
                    .dst_binding(Self::TEXTURE_BINDING)
                    .dst_set(self.set),
            ),
            &[],
        );
    }

    // pub unsafe fn update_buffer_descriptor<T>(
    //     &mut self,
    //     buffer: &Buffer<T>,
    //     binding: u32,
    //     vulkan_context: &VulkanContext,
    // ) {
    //     let descriptor_type = match buffer.usage {
    //         vk::BufferUsageFlags::UNIFORM_BUFFER => vk::DescriptorType::UNIFORM_BUFFER,
    //         vk::BufferUsageFlags::STORAGE_BUFFER => vk::DescriptorType::STORAGE_BUFFER,
    //         d => unimplemented!("Unknown descriptor type: {d:?}"),
    //     };

    //     vulkan_context.device.update_descriptor_sets(
    //         std::slice::from_ref(
    //             &vk::WriteDescriptorSet::default()
    //                 .buffer_info(std::slice::from_ref(
    //                     &vk::DescriptorBufferInfo::default()
    //                         .buffer(buffer.handle)
    //                         .range(vk::WHOLE_SIZE),
    //                 ))
    //                 .descriptor_type(descriptor_type)
    //                 .dst_binding(binding)
    //                 .dst_set(self.set),
    //         ),
    //         &[],
    //     );
    // }
}
