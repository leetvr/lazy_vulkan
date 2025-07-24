use std::{sync::Arc, u64};

use ash::vk::{self};

use crate::{draw_params::DrawParams, sub_renderer::SubRenderer};

use super::{
    allocator::Allocator,
    context::Context,
    depth_buffer::{DepthBuffer, DEPTH_RANGE},
    swapchain::{Drawable, Swapchain},
    FULL_IMAGE,
};

pub struct Renderer<'a, T> {
    pub context: Arc<Context>,
    pub fence: vk::Fence,
    pub rendering_complete: vk::Semaphore,
    pub swapchain: Swapchain,
    pub depth_buffer: DepthBuffer,
    pub sub_renderers: Vec<Box<dyn SubRenderer<'a, T>>>,
    pub allocator: Allocator,
}

impl<'a, T> Renderer<'a, T> {
    pub(crate) fn new(context: Arc<Context>, swapchain: Swapchain) -> Self {
        let device = &context.device;

        let rendering_complete =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();

        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .unwrap();

        let allocator = Allocator::new(context.clone());
        let depth_buffer = DepthBuffer::new(&context, &swapchain);

        Self {
            context,
            rendering_complete,
            fence,
            swapchain,
            depth_buffer,
            sub_renderers: vec![],
            allocator,
        }
    }

    pub(crate) fn draw(&mut self, state: &'a T) {
        // Begin rendering
        let drawable = self.begin_rendering();

        // Execute any pending transfers from the previous frame
        self.allocator.execute_transfers();

        // Stage transfers for the next frame
        for subrenderer in &mut self.sub_renderers {
            subrenderer.stage_transfers(&mut self.allocator);
        }

        // Draw with our sub-renderers
        for subrenderer in &mut self.sub_renderers {
            let params = DrawParams::new(drawable, self.depth_buffer, state);
            subrenderer.draw(params);
        }

        // End rendering
        self.end_rendering(drawable);

        // Present
        self.swapchain.present(
            drawable,
            self.context.graphics_queue,
            self.rendering_complete,
        );
    }

    fn begin_rendering(&mut self) -> Drawable {
        let device = &self.context.device;

        if self.swapchain.needs_update {
            unsafe { device.device_wait_idle().unwrap() };
            self.swapchain.resize(&self.context.device);
        }

        let drawable = loop {
            if let Some(drawable) = self.swapchain.get_drawable() {
                break drawable;
            }

            unsafe { device.device_wait_idle().unwrap() };
            self.swapchain.resize(&self.context.device);
        };

        // Get a `Drawable` from the swapchain
        let render_area = drawable.extent;

        // Block the CPU until we're done rendering the previous frame
        unsafe {
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .unwrap();
            device.reset_fences(&[self.fence]).unwrap();
        }

        // Begin the command buffer
        let command_buffer = self.context.draw_command_buffer;
        unsafe {
            device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap()
        };

        unsafe {
            // Transition the rendering attachments into their correct state
            device.cmd_pipeline_barrier2(
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
            device.cmd_begin_rendering(
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
                                float32: [0.1, 0.1, 0.1, 1.0],
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

        drawable
    }

    fn end_rendering(&self, drawable: Drawable) {
        let device = &self.context.device;
        let queue = self.context.graphics_queue;
        let command_buffer = self.context.draw_command_buffer;
        let swapchain_image = drawable.image;

        unsafe {
            // End rendering
            device.cmd_end_rendering(command_buffer);

            // Next, transition the color attachment into the present state
            device.cmd_pipeline_barrier2(
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
            device
                .queue_submit2(
                    queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
                        ])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(drawable.ready)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.rendering_complete)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                    self.fence,
                )
                .unwrap();
        }
    }
}
