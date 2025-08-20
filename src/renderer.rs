use super::{
    allocator::Allocator,
    context::Context,
    depth_buffer::{DepthBuffer, DEPTH_RANGE},
    swapchain::{Drawable, Swapchain},
    FULL_IMAGE,
};
use crate::{
    descriptors::Descriptors,
    draw_params::DrawParams,
    headless_swapchain::HeadlessSwapchain,
    image_manager::ImageManager,
    sub_renderer::{StateFamily, SubRenderer},
    Image, Pipeline,
};
use ash::vk::{self};
use std::{path::Path, sync::Arc, u64};

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
    pub sub_renderers: Vec<Box<dyn for<'s> SubRenderer<'s, State = SF::For<'s>>>>,
    swapchain: SwapchainBackend,
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
            sub_renderers: Vec::new(),
        }
    }

    pub(crate) fn from_wsi(context: Arc<Context>, swapchain: Swapchain) -> Self {
        let extent = swapchain.extent;
        Self::new(context, SwapchainBackend::WSI(swapchain), extent)
    }

    pub(crate) fn headless(
        context: Arc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
        Self::new(
            context.clone(),
            SwapchainBackend::Headless(HeadlessSwapchain::new(context, extent, format)),
            extent,
        )
    }

    pub fn draw<'s>(&mut self, state: &SF::For<'s>, drawable: &Drawable) {
        // Begin rendering
        self.begin_rendering(state, drawable);

        let drawable = drawable.clone(); // TODO

        // Draw with our sub-renderers
        self.context
            .begin_marker("Drawing", glam::vec4(0.0, 0.0, 1.0, 1.0));
        for subrenderer in &mut self.sub_renderers {
            self.context
                .begin_marker(subrenderer.label(), glam::vec4(1.0, 0.0, 1.0, 1.0));
            let params = DrawParams::new(
                self.context.draw_command_buffer,
                drawable,
                self.depth_buffer,
            );
            subrenderer.draw_opaque(state, &self.context, params);
            self.context.end_marker();
        }
        self.context.end_marker();

        // End dynamic rendering
        unsafe {
            self.context
                .cmd_end_rendering(self.context.draw_command_buffer)
        };

        // Draw layers
        for subrenderer in &mut self.sub_renderers {
            self.context
                .begin_marker(subrenderer.label(), glam::vec4(1.0, 0.0, 1.0, 1.0));
            let params = DrawParams::new(
                self.context.draw_command_buffer,
                drawable,
                self.depth_buffer,
            );
            subrenderer.draw_layer(state, &self.context, params);
            self.context.end_marker();
        }
    }

    pub fn submit_and_present(&mut self, drawable: Drawable) {
        // Transition the colour image to the present layout and submit all work
        self.submit_rendering(&drawable);

        // Present
        if let SwapchainBackend::WSI(swapchain) = &self.swapchain {
            swapchain.present(drawable, self.context.graphics_queue);
        }
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

    pub fn begin_rendering<'s>(&mut self, state: &SF::For<'s>, drawable: &Drawable) {
        let context = &self.context;
        let device = &context.device;
        let command_buffer = context.draw_command_buffer;

        // Get a `Drawable` from the swapchain
        let render_area = drawable.extent;

        // Begin the command buffer
        self.context
            .begin_marker("Begin Rendering", glam::vec4(0.5, 0.5, 0., 1.));

        // Stage transfers for this frame
        self.context
            .begin_marker("Stage Transfers", glam::vec4(1.0, 0.0, 1.0, 1.0));
        for subrenderer in &mut self.sub_renderers {
            subrenderer.stage_transfers(state, &mut self.allocator, &mut self.image_manager);
        }
        self.context.end_marker();

        // Execute them
        self.allocator.execute_transfers();

        unsafe {
            // Transition the rendering attachments into their correct state
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    // Swapchain image
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(drawable.image)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                    // Depth buffer
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(DEPTH_RANGE)
                        .image(self.depth_buffer.image)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .src_stage_mask(vk::PipelineStageFlags2::empty())
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
                        .image_view(drawable.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
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

        self.context.end_marker();
    }

    pub fn resize(&mut self, extent: vk::Extent2D) {
        match &mut self.swapchain {
            SwapchainBackend::WSI(swapchain) => {
                swapchain.extent = extent;
                swapchain.needs_update = true;
            }
            SwapchainBackend::Headless(headless_swapchain) => headless_swapchain.resize(extent),
        }
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

                // Recreate the depth buffer if the swapchain was resized
                self.depth_buffer.validate(&self.context, swapchain);

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
        let swapchain_image = drawable.image;

        unsafe {
            // First, transition the color attachment into the present state
            context.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .subresource_range(FULL_IMAGE)
                        .image(swapchain_image)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                ]),
            );

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
        )
    }

    pub fn create_image(
        &mut self,
        format: vk::Format,
        extent: vk::Extent2D,
        image_bytes: impl AsRef<[u8]>,
        image_usage_flags: vk::ImageUsageFlags,
    ) -> Image {
        self.image_manager.create_image(
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
}
