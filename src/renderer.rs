use super::{
    allocator::Allocator,
    context::Context,
    depth_buffer::{DepthBuffer, DEPTH_RANGE},
    swapchain::{Drawable, Swapchain},
    FULL_IMAGE,
};
use crate::{
    descriptors::Descriptors,
    headless_swapchain::HeadlessSwapchain,
    image_manager::ImageManager,
    render_plan::{AttachmentState, RenderStage},
    sub_renderer::{AttachmentInfo, LayerInfo, StateFamily, SubRenderer},
    HeadlessSwapchainImage, Image, Pipeline, PipelineOptions, RenderAttachment, RenderPlan,
};
use ash::vk::{self};
use std::{collections::HashMap, path::Path, sync::Arc, u64};

enum SwapchainBackend {
    WSI(Swapchain),
    Headless(HeadlessSwapchain),
}

pub struct Renderer<SF: StateFamily> {
    pub context: Arc<Context>,
    pub fence: vk::Fence,
    pub depth_buffer: DepthBuffer,
    pub allocator: Allocator,
    pub image_manager: ImageManager,
    pub descriptors: Descriptors,
    pub sub_renderers: HashMap<String, Box<dyn for<'s> SubRenderer<'s, State = SF::For<'s>>>>,
    pub render_attachments: HashMap<String, RenderAttachment>,
    swapchain: SwapchainBackend,
    /// Monotonically increasing frame counter
    pub frame: u32,
}

impl<SF: StateFamily> Renderer<SF> {
    fn new(
        context: Arc<Context>,
        swapchain: SwapchainBackend,
        drawable_size: vk::Extent2D,
    ) -> Self {
        let device = &context.device;

        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .unwrap();

        let allocator = Allocator::new(context.clone());
        let descriptors = Descriptors::new(context.clone());
        let image_manager = ImageManager::new(context.clone(), descriptors.set);
        let depth_buffer = DepthBuffer::new(&context, drawable_size);

        Self {
            context,
            fence,
            swapchain,
            depth_buffer,
            allocator,
            image_manager,
            descriptors,
            sub_renderers: Default::default(),
            render_attachments: Default::default(),
            frame: 0,
        }
    }

    pub fn from_wsi(context: Arc<Context>, swapchain: Swapchain) -> Self {
        let extent = swapchain.extent;
        Self::new(context, SwapchainBackend::WSI(swapchain), extent)
    }

    pub fn headless(context: Arc<Context>, extent: vk::Extent2D, format: vk::Format) -> Self {
        Self::new(
            context.clone(),
            SwapchainBackend::Headless(HeadlessSwapchain::new(context, extent, format)),
            extent,
        )
    }

    pub fn draw_render_plan<'s>(
        &mut self,
        state: &SF::For<'s>,
        plan: RenderPlan,
        drawable: &Drawable,
    ) {
        self.context
            .begin_marker("Drawing Render Plan", glam::vec4(0.0, 0.0, 1.0, 1.0));
        let mut attachment_states = HashMap::new();
        for attachment in self.render_attachments.keys() {
            attachment_states.insert(attachment.clone(), AttachmentState::Undefined);
        }

        // Do the main passes
        for pass in &plan.passes {
            self.context
                .begin_marker(&pass.name, glam::vec4(1.0, 0.2, 0.4, 1.0));

            if pass.colour_attachment.is_none() && pass.depth_attachment.is_none() {
                panic!("Pass has no attachments!");
            }

            let mut colour_load_op = vk::AttachmentLoadOp::CLEAR;
            let mut depth_load_op = vk::AttachmentLoadOp::CLEAR;

            if let Some(colour_attachment) = pass.colour_attachment.as_ref() {
                let current_state = attachment_states
                    .get_mut(colour_attachment)
                    .expect(&format!(
                        "Colour attachment {} not found",
                        colour_attachment
                    ));

                if *current_state != AttachmentState::Undefined {
                    colour_load_op = vk::AttachmentLoadOp::LOAD;
                }

                let colour_attachment = self
                    .render_attachments
                    .get(colour_attachment)
                    .copied()
                    .unwrap();

                let desired_state = AttachmentState::ColourOutput;

                // Transition attachment if necessary
                if *current_state != desired_state {
                    self.transition_attachment(colour_attachment, current_state, desired_state);
                }
            }

            if let Some(depth_attachment) = pass.depth_attachment.as_ref() {
                let current_state = attachment_states
                    .get_mut(depth_attachment)
                    .expect(&format!("Depth attachment {} not found", depth_attachment));
                let depth_attachment = self
                    .render_attachments
                    .get(depth_attachment)
                    .copied()
                    .unwrap();

                if *current_state != AttachmentState::Undefined {
                    depth_load_op = vk::AttachmentLoadOp::LOAD;
                }

                let desired_state = AttachmentState::DepthOutput;

                // Transition attachment if necessary
                if *current_state != desired_state {
                    self.transition_attachment(depth_attachment, current_state, desired_state);
                }
            }

            for sample_attachment_name in pass.sample_attachments.iter() {
                let current_state =
                    attachment_states
                        .get_mut(sample_attachment_name)
                        .expect(&format!(
                            "Sample attachment {} not found",
                            sample_attachment_name
                        ));
                let sample_attachment = self
                    .render_attachments
                    .get(sample_attachment_name)
                    .copied()
                    .unwrap();
                let desired_state = AttachmentState::Sampled;

                // Transition attachment if necessary
                if *current_state != desired_state {
                    self.transition_attachment(sample_attachment, current_state, desired_state);
                }
            }

            let subrenderer = self
                .sub_renderers
                .get_mut(&pass.subrenderer)
                .expect(&format!("Subrenderer not found: {}", pass.subrenderer));

            let command_buffer = self.context.draw_command_buffer;
            let device = &self.context.device;
            match pass.stage {
                // The contract for a shadow renderer is that begin rendering will have been
                // recorded on the shadow buffer
                RenderStage::Shadow => {
                    unsafe {
                        let depth_attachment = self
                            .render_attachments
                            .get(pass.depth_attachment.as_ref().unwrap())
                            .unwrap();
                        let render_area = depth_attachment.extent;

                        device.cmd_set_scissor(command_buffer, 0, &[render_area.into()]);
                        device.cmd_set_viewport(
                            command_buffer,
                            0,
                            &[vk::Viewport::default()
                                .width(render_area.width as _)
                                .height(render_area.height as _)
                                .max_depth(1.)],
                        );

                        self.context.cmd_begin_rendering(
                            command_buffer,
                            &vk::RenderingInfo::default()
                                .layer_count(1)
                                .render_area(render_area.into())
                                .depth_attachment(
                                    &vk::RenderingAttachmentInfo::default()
                                        .image_view(depth_attachment.view)
                                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                        .load_op(depth_load_op)
                                        .store_op(vk::AttachmentStoreOp::STORE),
                                ),
                        );
                    }

                    // Perform the shadow pass
                    subrenderer.draw_shadow(state, &self.context);

                    // End rendering
                    unsafe {
                        self.context.cmd_end_rendering(command_buffer);
                    }
                }
                // The contract for an opaque renderer is that begin rendering will have been
                // recorded on the colour buffer with a depth buffer.
                RenderStage::Opaque => {
                    unsafe {
                        let colour_attachment = self
                            .render_attachments
                            .get(pass.colour_attachment.as_ref().unwrap())
                            .unwrap();
                        let depth_attachment = self
                            .render_attachments
                            .get(pass.depth_attachment.as_ref().unwrap())
                            .unwrap();

                        let render_area = colour_attachment.extent;

                        device.cmd_set_scissor(command_buffer, 0, &[render_area.into()]);
                        device.cmd_set_viewport(
                            command_buffer,
                            0,
                            &[vk::Viewport::default()
                                .width(render_area.width as _)
                                .height(render_area.height as _)
                                .max_depth(1.)],
                        );

                        self.context.cmd_begin_rendering(
                            command_buffer,
                            &vk::RenderingInfo::default()
                                .layer_count(1)
                                .render_area(render_area.into())
                                .depth_attachment(
                                    &vk::RenderingAttachmentInfo::default()
                                        .image_view(depth_attachment.view)
                                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                        .load_op(depth_load_op)
                                        .store_op(vk::AttachmentStoreOp::STORE),
                                )
                                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                                    .image_view(colour_attachment.view)
                                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                    .load_op(colour_load_op)
                                    .store_op(vk::AttachmentStoreOp::STORE)]),
                        );
                    }

                    // Perform the opaque pass
                    subrenderer.draw_opaque(state, &self.context);

                    // End rendering
                    unsafe {
                        self.context.cmd_end_rendering(command_buffer);
                    }
                }
                // The contract for a layer renderer is that it is simply given the
                // to do with whatever it damn well pleases.
                RenderStage::Layer => {
                    let colour_attachment = pass
                        .colour_attachment
                        .as_ref()
                        .map(|a| {
                            self.render_attachments.get(a).map(|a| AttachmentInfo {
                                extent: a.extent,
                                view: a.view,
                                handle: a.handle,
                            })
                        })
                        .flatten();

                    let depth_attachment = pass
                        .depth_attachment
                        .as_ref()
                        .map(|a| {
                            self.render_attachments.get(a).map(|a| AttachmentInfo {
                                extent: a.extent,
                                view: a.view,
                                handle: a.handle,
                            })
                        })
                        .flatten();

                    let layer_info = LayerInfo {
                        colour_attachment,
                        depth_attachment,
                    };
                    subrenderer.draw_layer(state, &self.context, layer_info);
                }
            }

            // end plan name marker
            self.context.end_marker();
        }

        // Composite
        {
            self.context
                .begin_marker("Composite", glam::vec4(0.0, 1.0, 0.0, 1.0));

            let current_state = &mut AttachmentState::Swapchain;

            // Transition the swapchain image
            let drawable_render_attachment = RenderAttachment {
                handle: drawable.image,
                view: drawable.view,
                extent: drawable.extent,
                format: self.get_drawable_format(),
                id: 0, // unused
            };
            self.transition_attachment(
                drawable_render_attachment,
                current_state,
                AttachmentState::ColourOutput,
            );

            *current_state = AttachmentState::ColourOutput;

            let sampled_attachment = self
                .render_attachments
                .get(&plan.target_to_composite)
                .copied()
                .expect("Couldn't find target to composite");

            // Transition the sampled attachment
            self.transition_attachment(sampled_attachment, current_state, AttachmentState::Sampled);

            unsafe {
                let context = &self.context;
                let device = &context.device;
                let command_buffer = self.context.draw_command_buffer;
                let render_area = drawable.extent;

                device.cmd_set_scissor(command_buffer, 0, &[render_area.into()]);
                device.cmd_set_viewport(
                    command_buffer,
                    0,
                    &[vk::Viewport::default()
                        .width(render_area.width as _)
                        .height(render_area.height as _)
                        .max_depth(1.)],
                );

                // Begin rendering
                context.cmd_begin_rendering(
                    command_buffer,
                    &vk::RenderingInfo::default()
                        .render_area(drawable.extent.into())
                        .layer_count(1)
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(drawable.view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .clear_value(vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 0.0],
                                },
                            })]),
                );

                let composite_pass = self
                    .sub_renderers
                    .get_mut(&plan.compositor_subrenderer)
                    .expect("Couldn't find compositor subrenderer");

                composite_pass.draw_opaque(state, context);

                // End rendering
                self.context.cmd_end_rendering(command_buffer);
            }

            // end composite marker
            self.context.end_marker();
        }

        // end render plan marker
        self.context.end_marker();
    }

    pub fn draw<'s>(&mut self, state: &SF::For<'s>, drawable: &Drawable) {
        self.context
            .begin_marker("Drawing", glam::vec4(0.0, 0.0, 1.0, 1.0));

        // Draw with our sub-renderers
        // Shadow pass
        self.context
            .begin_marker("Draw Shadow", glam::vec4(1.0, 0.2, 0.4, 1.0));
        for subrenderer in self.sub_renderers.values_mut() {
            let label = format!("{} Shadow Pass", subrenderer.label());
            self.context
                .begin_marker(&label, glam::vec4(1.0, 0.2, 0.4, 1.0));
            subrenderer.draw_shadow(state, &self.context);
            // end pass marker
            self.context.end_marker();
        }

        // end draw shadow marker
        self.context.end_marker();

        // Draw opaque
        self.context
            .begin_marker("Draw Opaque", glam::vec4(1.0, 0.0, 1.0, 1.0));
        // Begin opaque render pass
        self.begin_rendering(&RenderAttachment {
            handle: drawable.image,
            view: drawable.view,
            extent: drawable.extent,
            format: self.get_drawable_format(),
            id: 0, // is is invalid to sample from the colour image during the opaque pass
        });

        let drawable = drawable.clone(); // TODO
        for subrenderer in self.sub_renderers.values_mut() {
            let label = format!("{} Opaque Pass", subrenderer.label());
            self.context
                .begin_marker(&label, glam::vec4(1.0, 0.0, 1.0, 1.0));
            subrenderer.draw_opaque(state, &self.context);
            // end pass marker
            self.context.end_marker();
        }

        // End opaque render pass
        unsafe {
            self.context
                .cmd_end_rendering(self.context.draw_command_buffer)
        };

        // end draw opaque marker
        self.context.end_marker();

        // Draw layers
        self.context
            .begin_marker("Draw Layer", glam::vec4(0.5, 1.0, 0.2, 1.0));
        for subrenderer in self.sub_renderers.values_mut() {
            let label = format!("{} Layer Pass", subrenderer.label());
            self.context
                .begin_marker(&label, glam::vec4(0.5, 1.0, 0.2, 1.0));
            subrenderer.draw_layer(
                state,
                &self.context,
                LayerInfo {
                    colour_attachment: Some(AttachmentInfo {
                        extent: drawable.extent,
                        view: drawable.view,
                        handle: drawable.image,
                    }),
                    depth_attachment: Some(AttachmentInfo {
                        extent: drawable.extent,
                        view: self.depth_buffer.view,
                        handle: self.depth_buffer.image,
                    }),
                },
            );
            // end pass marker
            self.context.end_marker();
        }
        // end draw layer marker
        self.context.end_marker();

        // end "drawing"
        self.context.end_marker();
    }

    pub fn begin_command_buffer(&mut self) {
        let device = &self.context.device;
        // Block the CPU until we're done rendering the previous frame
        unsafe {
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();
            device.reset_fences(&[self.fence]).unwrap();
        }

        self.context.begin_command_buffer();
    }

    pub fn submit_and_present(&mut self, drawable: Drawable) {
        self.context.begin_marker(
            &format!("Submit frame {}", self.frame),
            glam::Vec4::new(0.5, 0.5, 0., 1.),
        );
        // Transition the colour image to the present layout and submit all work
        self.submit_rendering(&drawable);

        // Present
        if let SwapchainBackend::WSI(swapchain) = &mut self.swapchain {
            swapchain.present(drawable, self.context.graphics_queue);
        }

        self.frame += 1;
    }

    fn begin_rendering(&mut self, colour_attachment: &RenderAttachment) {
        let context = &self.context;
        let device = &context.device;
        let command_buffer = context.draw_command_buffer;

        // Get a `Drawable` from the swapchain
        let render_area = colour_attachment.extent;

        // Begin the command buffer
        self.context
            .begin_marker("Begin Opaque Pass", glam::vec4(0.5, 0.5, 0., 1.));

        unsafe {
            // Transition the rendering attachments into their correct state
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    // Swapchain image
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(colour_attachment.handle)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    // Depth buffer
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(DEPTH_RANGE)
                        .image(self.depth_buffer.image)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_access_mask(
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL),
                ]),
            );

            // Begin rendering
            context.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .render_area(render_area.into())
                    .layer_count(1)
                    .depth_attachment(
                        &vk::RenderingAttachmentInfo::default()
                            .image_view(self.depth_buffer.view)
                            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 0.0,
                                    stencil: 0,
                                },
                            }),
                    )
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(colour_attachment.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        })]),
            );

            // Set the dynamic state
            device.cmd_set_scissor(command_buffer, 0, &[render_area.into()]);
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(render_area.width as _)
                    .height(render_area.height as _)
                    .max_depth(1.)],
            );
        }

        // end begin rendering marker
        self.context.end_marker();
    }

    pub fn stage_and_execute_transfers<'s>(&mut self, state: &<SF as StateFamily>::For<'s>) {
        let command_buffer = self.context.draw_command_buffer;
        // Stage transfers for this frame
        self.context
            .begin_marker("Stage Transfers", glam::vec4(1.0, 0.0, 1.0, 1.0));
        for subrenderer in self.sub_renderers.values_mut() {
            self.context
                .begin_marker(subrenderer.label(), glam::vec4(1.0, 0.0, 1.0, 1.0));
            subrenderer.stage_transfers(state, &mut self.allocator, &mut self.image_manager);
            self.context.end_marker();
        }
        self.context.end_marker();

        // Execute them
        self.allocator.execute_transfers(command_buffer);
    }

    pub fn resize(&mut self, extent: vk::Extent2D) {
        match &mut self.swapchain {
            SwapchainBackend::WSI(swapchain) => {
                swapchain.extent = extent;
                swapchain.needs_update = true;
            }
            SwapchainBackend::Headless(headless_swapchain) => headless_swapchain.resize(extent),
        }

        self.depth_buffer.resize(&self.context, extent);
    }

    pub(crate) fn get_drawable(&mut self) -> Drawable {
        let device = &self.context.device;

        match &mut self.swapchain {
            SwapchainBackend::WSI(swapchain) => {
                if swapchain.needs_update {
                    unsafe { device.device_wait_idle().unwrap() };
                    swapchain.resize(&self.context.device);
                }

                let drawable = loop {
                    if let Some(drawable) = swapchain.get_drawable() {
                        break drawable;
                    }

                    unsafe { device.device_wait_idle().unwrap() };
                    swapchain.resize(&self.context.device);
                };

                drawable
            }
            SwapchainBackend::Headless(headless_swapchain) => headless_swapchain.get_drawable(),
        }
    }

    fn submit_rendering(&self, drawable: &Drawable) {
        let context = &self.context;
        let device = &context.device;
        let queue = context.graphics_queue;
        let command_buffer = context.draw_command_buffer;
        unsafe {
            // First, transition the color attachment into the present state
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(drawable.image)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                ]),
            );

            self.context.end_marker();

            // End the command buffer
            device.end_command_buffer(command_buffer).unwrap();

            // Submit the work to the queue
            // blegh
            if let Some(image_available) = drawable.image_available {
                context.queue_submit2(
                    queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(image_available)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(drawable.rendering_complete)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                    self.fence,
                );
            } else {
                context.queue_submit2(
                    queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
                        ])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(drawable.rendering_complete)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                    self.fence,
                );
            }
        }
    }

    pub fn create_pipeline<R>(
        &self,
        vertex_shader: impl AsRef<Path>,
        fragment_shader: impl AsRef<Path>,
    ) -> Pipeline {
        Pipeline::new::<R>(
            self.context.clone(),
            &self.descriptors,
            self.get_drawable_format(),
            vertex_shader,
            fragment_shader,
            Default::default(),
        )
    }

    pub fn create_pipeline_with_options<R>(
        &self,
        vertex_shader: impl AsRef<Path>,
        fragment_shader: impl AsRef<Path>,
        options: PipelineOptions,
    ) -> Pipeline {
        let colour_format = options.colour_format.unwrap_or(self.get_drawable_format());
        Pipeline::new::<R>(
            self.context.clone(),
            &self.descriptors,
            colour_format,
            vertex_shader,
            fragment_shader,
            options,
        )
    }

    pub fn create_image(
        &mut self,
        name: impl AsRef<str>,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
    ) -> Image {
        self.image_manager.create_image(
            name,
            &mut self.allocator,
            format,
            extent,
            image_bytes,
            image_usage_flags,
        )
    }

    pub fn get_drawable_format(&self) -> vk::Format {
        match &self.swapchain {
            SwapchainBackend::WSI(swapchain) => swapchain.format,
            SwapchainBackend::Headless(headless_swapchain) => headless_swapchain.format,
        }
    }

    pub fn get_drawable_extent(&self) -> vk::Extent2D {
        match &self.swapchain {
            SwapchainBackend::WSI(swapchain) => swapchain.extent,
            SwapchainBackend::Headless(headless_swapchain) => headless_swapchain.extent,
        }
    }

    pub fn get_headless_image(&self) -> Option<HeadlessSwapchainImage> {
        match &self.swapchain {
            SwapchainBackend::Headless(headless_swapchain) => {
                Some(headless_swapchain.image.clone())
            }
            _ => None,
        }
    }

    fn transition_attachment(
        &mut self,
        attachment: RenderAttachment,
        current_state: &mut AttachmentState,
        desired_state: AttachmentState,
    ) {
        let context = &self.context;
        let command_buffer = context.draw_command_buffer;

        let (src_access_mask, src_stage_mask, old_layout) = get_flags_for_state(*current_state);
        let (dst_access_mask, dst_stage_mask, new_layout) = get_flags_for_state(desired_state);

        let subresource_range = match attachment.format {
            vk::Format::D32_SFLOAT => DEPTH_RANGE,
            _ => FULL_IMAGE,
        };

        unsafe {
            // Transition the attachment into its correct state
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    // Swapchain image
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(subresource_range)
                        .image(attachment.handle)
                        .src_access_mask(src_access_mask)
                        .src_stage_mask(src_stage_mask)
                        .dst_access_mask(dst_access_mask)
                        .dst_stage_mask(dst_stage_mask)
                        .old_layout(old_layout)
                        .new_layout(new_layout),
                ]),
            );
        }

        // Update the state
        *current_state = desired_state
    }
}

fn get_flags_for_state(
    current_state: AttachmentState,
) -> (vk::AccessFlags2, vk::PipelineStageFlags2, vk::ImageLayout) {
    match current_state {
        AttachmentState::ColourOutput => (
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        AttachmentState::DepthOutput => (
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        ),
        AttachmentState::Sampled => (
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        AttachmentState::Undefined => (
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::NONE,
            vk::ImageLayout::UNDEFINED,
        ),
        AttachmentState::Swapchain => (
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageLayout::UNDEFINED,
        ),
    }
}
