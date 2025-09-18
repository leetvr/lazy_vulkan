use std::{path::Path, sync::Arc};

use ash::vk;

use crate::descriptors::Descriptors;

use super::{context::Context, depth_buffer::DEPTH_FORMAT};

#[derive(Clone)]
pub struct Pipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    context: Arc<Context>,
    // Avoids having to pass &Descriptors around at draw time
    pub descriptor_set: vk::DescriptorSet,
}

impl Pipeline {
    // TODO: Watch shaders!
    pub(crate) fn new<Registers>(
        context: Arc<Context>,
        descriptors: &Descriptors,
        format: vk::Format,
        vertex_shader: impl AsRef<Path>,
        fragment_shader: impl AsRef<Path>,
        cull_mode: vk::CullModeFlags,
    ) -> Self {
        let device = &context.device;

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptors.layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .size(std::mem::size_of::<Registers>() as u32)
                        .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)]),
                None,
            )
        }
        .unwrap();

        let handle = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo::default()
                            .name(c"main")
                            .module(load_module(vertex_shader, &context))
                            .stage(vk::ShaderStageFlags::VERTEX),
                        vk::PipelineShaderStageCreateInfo::default()
                            .name(c"main")
                            .module(load_module(fragment_shader, &context))
                            .stage(vk::ShaderStageFlags::FRAGMENT),
                    ])
                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                    .input_assembly_state(
                        &vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .scissor_count(1)
                            .viewport_count(1),
                    )
                    .dynamic_state(
                        &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                            vk::DynamicState::SCISSOR,
                            vk::DynamicState::VIEWPORT,
                        ]),
                    )
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::default()
                            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                            .cull_mode(cull_mode)
                            .polygon_mode(vk::PolygonMode::FILL)
                            .line_width(1.0),
                    )
                    .depth_stencil_state(
                        &vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_write_enable(true)
                            .depth_test_enable(true)
                            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
                            .stencil_test_enable(false)
                            .depth_bounds_test_enable(false)
                            .max_depth_bounds(1.),
                    )
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                            vk::PipelineColorBlendAttachmentState::default()
                                .blend_enable(false)
                                .color_write_mask(vk::ColorComponentFlags::RGBA),
                        ]),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                    )
                    .layout(layout)
                    .push_next(
                        &mut vk::PipelineRenderingCreateInfo::default()
                            .depth_attachment_format(DEPTH_FORMAT)
                            .color_attachment_formats(&[format]),
                    )],
                None,
            )
        }
        .unwrap()[0];

        Self {
            context,
            layout,
            handle,
            descriptor_set: descriptors.set,
        }
    }

    pub fn update_registers<Registers: bytemuck::Pod>(&self, registers: &Registers) {
        let draw_command_buffer = self.context.draw_command_buffer;
        unsafe {
            self.context.device.cmd_push_constants(
                draw_command_buffer,
                self.layout,
                vk::ShaderStageFlags::ALL_GRAPHICS,
                0,
                bytemuck::bytes_of(registers),
            )
        };
    }

    pub fn bind_descriptor_sets(&self) {
        let command_buffer = self.context.draw_command_buffer;
        unsafe {
            self.context.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }
    }
}

pub fn load_module(path: impl AsRef<Path>, context: &Context) -> vk::ShaderModule {
    let mut file = std::fs::File::open(path).unwrap();
    let words = ash::util::read_spv(&mut file).unwrap();

    unsafe {
        context
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
    }
    .unwrap()
}
