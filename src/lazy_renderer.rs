//! A Vulkan backend for the [`yakui`] crate. Uses [`ash`] to wrap Vulkan related functionality.
//!
//! The main entrypoint is the [`YakuiVulkan`] struct which creates a [`ash::vk::RenderPass`] and [`ash::vk::Pipeline`]
//! to draw yakui GUIs. This is initialised by populating a [`VulkanContext`] helper struct to pass down the relevant hooks
//! into your Vulkan renderer.
//!
//! Like most Vulkan applications, this crate uses unsafe Rust! No checks are made to ensure that Vulkan handles are valid,
//! so take note of the safety warnings on the various methods of [`YakuiVulkan`].
//!
//! Currently this crate only supports drawing to images in the `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` layout, but future
//! releases will support drawing to any arbitrary [`vk::ImageView`].
//!
//! This crate requires at least Vulkan 1.2 and a GPU with support for `VkPhysicalDeviceDescriptorIndexingFeatures.descriptorBindingPartiallyBound`.
//! You should also, you know, enable that feature, or Vulkan Validation Layers will get mad at you. You definitely don't want that.
//!
//! For an example of how to use this crate, check out the cleverly named `it_works` test in `lib.rs` in the GitHub repo.

use crate::buffer::Buffer;
use crate::descriptors::Descriptors;
use crate::vulkan_context::VulkanContext;
use crate::vulkan_texture::VulkanTexture;
use crate::vulkan_texture::VulkanTextureCreateInfo;
use crate::{LazyVulkanBuilder, Vertex};
use std::{ffi::CStr, io::Cursor};

use ash::{util::read_spv, vk};
use bytemuck::{Pod, Zeroable};
use thunderdome::Index;

/// A struct wrapping everything needed to render yakui on Vulkan. This will be your main entry point.
///
/// Uses a simple descriptor system that requires at least Vulkan 1.2 and [VkPhysicalDeviceDescriptorIndexingFeatures](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceDescriptorIndexingFeatures.html)
/// `descriptorBindingPartiallyBound` to be enabled.
///
/// Note that this implementation is currently only able to render to a present surface (ie. the swapchain).
/// Future versions will support drawing to any `vk::ImageView`
///
/// Construct the struct by populating a [`VulkanContext`] with the relevant handles to your Vulkan renderer and
/// call [`YakuiVulkan::new()`].
///
/// Make sure to call [`YakuiVulkan::cleanup()`].
pub struct LazyRenderer {
    /// The render pass used to draw
    render_pass: vk::RenderPass,
    /// One or more framebuffers to draw on. This will match the number of `vk::ImageView` present in [`RenderSurface`]
    framebuffers: Vec<vk::Framebuffer>,
    /// The surface to draw on. Currently only supports present surfaces (ie. the swapchain)
    pub render_surface: RenderSurface,
    /// The pipeline layout used to draw
    pipeline_layout: vk::PipelineLayout,
    /// The graphics pipeline used to draw
    graphics_pipeline: vk::Pipeline,
    /// A single index buffer, shared between all draw calls
    pub index_buffer: Buffer<u32>,
    /// A single vertex buffer, shared between all draw calls
    pub vertex_buffer: Buffer<crate::Vertex>,
    /// Textures owned by the user
    user_textures: thunderdome::Arena<VulkanTexture>,
    /// A wrapper around descriptor set functionality
    pub descriptors: Descriptors,
}

#[derive(Clone)]
/// The surface for yakui to draw on. Currently only supports present surfaces (ie. the swapchain)
pub struct RenderSurface {
    /// The resolution of the surface
    pub resolution: vk::Extent2D,
    /// The image format of the surface
    pub format: vk::Format,
    /// The image views to render to. One framebuffer will be created per view
    pub image_views: Vec<vk::ImageView>,
}

#[derive(Clone, Copy, Debug)]
/// A single draw call to render a yakui mesh
pub struct DrawCall {
    index_offset: u32,
    index_count: u32,
    texture_id: u32,
    workflow: Workflow,
}

impl DrawCall {
    pub fn new(index_offset: u32, index_count: u32, texture_id: u32, workflow: Workflow) -> Self {
        Self {
            index_offset,
            index_count,
            texture_id,
            workflow,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// Push constant used to determine texture and workflow
struct PushConstant {
    texture_id: u32,
    workflow: Workflow,
}

unsafe impl Zeroable for PushConstant {}
unsafe impl Pod for PushConstant {}

impl PushConstant {
    pub fn new(texture_id: u32, workflow: Workflow) -> Self {
        Self {
            texture_id,
            workflow,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
/// The workflow to use in the shader
pub enum Workflow {
    Main,
    Text,
}

unsafe impl Zeroable for Workflow {}
unsafe impl bytemuck::Pod for Workflow {}

impl LazyRenderer {
    /// Create a new [`LazyRenderer`] instance. Currently only supports rendering directly to the swapchain.
    ///
    /// ## Safety
    /// - `vulkan_context` must have valid members
    /// - the members of `render_surface` must have been created with the same [`ash::Device`] as `vulkan_context`.
    pub fn new(
        vulkan_context: &VulkanContext,
        render_surface: RenderSurface,
        builder: &LazyVulkanBuilder,
    ) -> Self {
        let vertex_shader = builder.vertex_shader.as_ref().unwrap();
        let fragment_shader = builder.fragment_shader.as_ref().unwrap();
        let device = &vulkan_context.device;
        let descriptors = Descriptors::new(vulkan_context);
        let final_layout = if builder.with_present {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        // TODO: Don't write directly to the present surface..
        let renderpass_attachments = [vk::AttachmentDescription {
            format: render_surface.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout,
            ..Default::default()
        }];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpass = vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&renderpass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe {
            device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        };

        let framebuffers = create_framebuffers(&render_surface, render_pass, device);

        let index_buffer = Buffer::new(
            vulkan_context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &builder.initial_indices,
        );
        let vertex_buffer = Buffer::new(
            vulkan_context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &builder.initial_vertices,
        );

        let mut vertex_spv_file = Cursor::new(vertex_shader);
        let mut frag_spv_file = Cursor::new(fragment_shader);

        let vertex_code =
            read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&vertex_code);

        let frag_code =
            read_spv(&mut frag_spv_file).expect("Failed to read fragment shader spv file");
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&frag_code);

        let vertex_shader_module = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };

        let fragment_shader_module = unsafe {
            device
                .create_shader_module(&frag_shader_info, None)
                .expect("Fragment shader module error")
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .push_constant_ranges(std::slice::from_ref(
                            &vk::PushConstantRange::builder()
                                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                                .size(std::mem::size_of::<PushConstant>() as _),
                        ))
                        .set_layouts(std::slice::from_ref(&descriptors.layout)),
                    None,
                )
                .unwrap()
        };

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let vertex_input_attribute_descriptions = [
            // position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: bytemuck::offset_of!(Vertex, position) as _,
            },
            // color
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: bytemuck::offset_of!(Vertex, colour) as _,
            },
            // UV / texcoords
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: bytemuck::offset_of!(Vertex, uv) as _,
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_input_binding_descriptions);
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: render_surface.resolution.width as f32,
            height: render_surface.resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [render_surface.resolution.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(&scissors)
            .viewports(&viewports);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass);

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info.build()],
                    None,
                )
                .expect("Unable to create graphics pipeline")
        };

        let graphics_pipeline = graphics_pipelines[0];

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        Self {
            render_pass,
            descriptors,
            framebuffers,
            render_surface,
            pipeline_layout,
            graphics_pipeline,
            index_buffer,
            vertex_buffer,
            user_textures: Default::default(),
        }
    }

    /// Render the draw calls we've built up
    pub fn render(
        &self,
        vulkan_context: &VulkanContext,
        framebuffer_index: u32,
        draw_calls: &[DrawCall],
    ) {
        let device = &vulkan_context.device;
        let command_buffer = vulkan_context.draw_command_buffer;

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let surface = &self.render_surface;

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[framebuffer_index as usize])
            .render_area(surface.resolution.into())
            .clear_values(&clear_values);

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: surface.resolution.width as f32,
            height: surface.resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            let default_scissor = [surface.resolution.into()];

            // We set the scissor first here as it's against the spec not to do so.
            device.cmd_set_scissor(command_buffer, 0, &default_scissor);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer.handle], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptors.set),
                &[],
            );
            for draw_call in draw_calls {
                // Instead of using different pipelines for text and non-text rendering, we just
                // pass the "workflow" down through a push constant and branch in the shader.
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&PushConstant::new(
                        draw_call.texture_id,
                        draw_call.workflow,
                    )),
                );

                // Draw the mesh with the indexes we were provided
                device.cmd_draw_indexed(
                    command_buffer,
                    draw_call.index_count,
                    1,
                    draw_call.index_offset,
                    0,
                    1,
                );
            }
            device.cmd_end_render_pass(command_buffer);
        }
    }

    /// Add a "user managed" texture to this [`YakuiVulkan`] instance. Returns a [`yakui::TextureId`] that can be used
    /// to refer to the texture in your GUI code.
    ///
    /// ## Safety
    /// - `vulkan_context` must be the same as the one used to create this instance
    pub fn add_user_texture(
        &mut self,
        vulkan_context: &VulkanContext,
        texture_create_info: VulkanTextureCreateInfo<Vec<u8>>,
    ) -> Index {
        let texture =
            VulkanTexture::new(vulkan_context, &mut self.descriptors, texture_create_info);
        self.user_textures.insert(texture)
    }

    /// Clean up all Vulkan related handles on this instance. You'll probably want to call this when the program ends, but
    /// before you've cleaned up your [`ash::Device`], or you'll receive warnings from the Vulkan Validation Layers.
    ///
    /// ## Safety
    /// - After calling this function, this instance will be **unusable**. You **must not** make any further calls on this instance
    ///   or you will have a terrible time.
    /// - `device` must be the same [`ash::Device`] used to create this instance.
    pub unsafe fn cleanup(&self, device: &ash::Device) {
        device.device_wait_idle().unwrap();
        self.descriptors.cleanup(device);
        for (_, texture) in &self.user_textures {
            texture.cleanup(device);
        }
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_pipeline(self.graphics_pipeline, None);
        self.index_buffer.cleanup(device);
        self.vertex_buffer.cleanup(device);
        self.destroy_framebuffers(device);
        device.destroy_render_pass(self.render_pass, None);
    }

    /// Update the surface that this [`YakuiVulkan`] instance will render to. You'll probably want to call
    /// this if the user resizes the window to avoid writing to an out-of-date swapchain.
    ///
    /// ## Safety
    /// - Care must be taken to ensure that the new [`RenderSurface`] points to images from a correct swapchain
    /// - You must use the same [`ash::Device`] used to create this instance
    pub fn update_surface(&mut self, render_surface: RenderSurface, device: &ash::Device) {
        unsafe {
            self.destroy_framebuffers(device);
        }
        self.framebuffers = create_framebuffers(&render_surface, self.render_pass, device);
        self.render_surface = render_surface;
    }

    unsafe fn destroy_framebuffers(&self, device: &ash::Device) {
        for framebuffer in &self.framebuffers {
            device.destroy_framebuffer(*framebuffer, None);
        }
    }
}

fn create_framebuffers(
    render_surface: &RenderSurface,
    render_pass: vk::RenderPass,
    device: &ash::Device,
) -> Vec<vk::Framebuffer> {
    let framebuffers: Vec<vk::Framebuffer> = render_surface
        .image_views
        .iter()
        .map(|&present_image_view| {
            let framebuffer_attachments = [present_image_view];
            let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&framebuffer_attachments)
                .width(render_surface.resolution.width)
                .height(render_surface.resolution.height)
                .layers(1);

            unsafe {
                device
                    .create_framebuffer(&frame_buffer_create_info, None)
                    .unwrap()
            }
        })
        .collect();
    framebuffers
}
