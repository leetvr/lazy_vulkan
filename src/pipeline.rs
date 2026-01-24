use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

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
    vertex_shader_path: PathBuf,
    fragment_shader_path: PathBuf,
    format: vk::Format,
    options: PipelineOptions,
}

impl Pipeline {
    // TODO: Watch shaders!
    pub fn new<Registers>(
        context: Arc<Context>,
        descriptors: &Descriptors,
        colour_format: vk::Format,
        vertex_shader: impl AsRef<Path>,
        fragment_shader: impl AsRef<Path>,
        options: PipelineOptions,
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

        let vertex_shader_path = vertex_shader.as_ref();
        let fragment_shader_path = fragment_shader.as_ref();

        let handle = create_pipeline::<Registers>(
            &context,
            colour_format,
            &options,
            layout,
            vertex_shader_path,
            fragment_shader_path,
        );

        Self {
            context,
            layout,
            handle,
            descriptor_set: descriptors.set,
            vertex_shader_path: vertex_shader_path.into(),
            fragment_shader_path: fragment_shader_path.into(),
            format: colour_format,
            options,
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

    pub fn reload<Registers>(&mut self) {
        self.handle = create_pipeline::<Registers>(
            &self.context,
            self.format,
            &self.options,
            self.layout,
            &self.vertex_shader_path,
            &self.fragment_shader_path,
        );
    }

    pub fn reload_with_new_options<Registers>(&mut self, options: PipelineOptions) {
        self.options = options;
        self.handle = create_pipeline::<Registers>(
            &self.context,
            self.format,
            &self.options,
            self.layout,
            &self.vertex_shader_path,
            &self.fragment_shader_path,
        );
    }
}

fn create_pipeline<Registers>(
    context: &Arc<Context>,
    colour_format: vk::Format,
    options: &PipelineOptions,
    layout: vk::PipelineLayout,
    vertex_shader_path: &Path,
    fragment_shader_path: &Path,
) -> vk::Pipeline {
    let device = &context.device;

    // Extract options
    let topology = if options.polygon_mode == vk::PolygonMode::FILL {
        vk::PrimitiveTopology::TRIANGLE_LIST
    } else {
        vk::PrimitiveTopology::LINE_LIST
    };

    let is_shadow_pass =
        options.depth_bias_constant_factor.is_some() || options.depth_bias_slope_factor.is_some();

    let depth_bias_slope_factor = options.depth_bias_slope_factor.unwrap_or_default();
    let depth_bias_constant_factor = options.depth_bias_constant_factor.unwrap_or_default();
    let color_attachment_formats: &[vk::Format] = if is_shadow_pass {
        &[]
    } else {
        &[colour_format]
    };

    unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[vk::GraphicsPipelineCreateInfo::default()
                .stages(&[
                    vk::PipelineShaderStageCreateInfo::default()
                        .name(c"main")
                        .module(load_module(vertex_shader_path, context))
                        .stage(vk::ShaderStageFlags::VERTEX),
                    vk::PipelineShaderStageCreateInfo::default()
                        .name(c"main")
                        .module(load_module(fragment_shader_path, context))
                        .stage(vk::ShaderStageFlags::FRAGMENT),
                ])
                .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                .input_assembly_state(
                    &vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology),
                )
                .viewport_state(
                    &vk::PipelineViewportStateCreateInfo::default()
                        .scissor_count(1)
                        .viewport_count(1),
                )
                .dynamic_state(
                    &vk::PipelineDynamicStateCreateInfo::default()
                        .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]),
                )
                .rasterization_state(
                    &vk::PipelineRasterizationStateCreateInfo::default()
                        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                        .cull_mode(options.cull_mode)
                        .polygon_mode(options.polygon_mode)
                        .depth_bias_enable(is_shadow_pass)
                        .depth_bias_slope_factor(depth_bias_slope_factor)
                        .depth_bias_constant_factor(depth_bias_constant_factor)
                        .line_width(1.0),
                )
                .depth_stencil_state(
                    &vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_write_enable(options.depth_write)
                        .depth_test_enable(true)
                        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
                        .stencil_test_enable(false)
                        .depth_bounds_test_enable(false)
                        .max_depth_bounds(1.),
                )
                .color_blend_state(
                    &vk::PipelineColorBlendStateCreateInfo::default()
                        .attachments(&[get_blend_attachment(options.blend_mode)]),
                )
                .multisample_state(
                    &vk::PipelineMultisampleStateCreateInfo::default()
                        .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                )
                .layout(layout)
                .push_next(
                    &mut vk::PipelineRenderingCreateInfo::default()
                        .depth_attachment_format(DEPTH_FORMAT)
                        .color_attachment_formats(color_attachment_formats),
                )],
            None,
        )
    }
    .unwrap()[0]
}

fn get_blend_attachment(blend_mode: BlendMode) -> vk::PipelineColorBlendAttachmentState {
    match blend_mode {
        BlendMode::None => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
        // just to be explicit
        BlendMode::Alpha => vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA),
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

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub cull_mode: vk::CullModeFlags,
    pub polygon_mode: vk::PolygonMode,
    pub blend_mode: BlendMode,
    pub depth_write: bool,
    /// Only useful for shadow pipelines
    pub depth_bias_constant_factor: Option<f32>,
    /// Only useful for shadow pipelines
    pub depth_bias_slope_factor: Option<f32>,
    /// Useful for more complex render setups
    pub colour_format: Option<vk::Format>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            cull_mode: vk::CullModeFlags::BACK,
            polygon_mode: vk::PolygonMode::FILL,
            blend_mode: BlendMode::None,
            depth_write: true,
            depth_bias_constant_factor: None,
            depth_bias_slope_factor: None,
            colour_format: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BlendMode {
    None,
    Alpha,
}
