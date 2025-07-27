use ash::vk;

use crate::{allocator::Allocator, context::Context, draw_params::DrawParams, pipeline::Pipeline};

pub trait SubRenderer {
    type State;

    fn draw(&mut self, context: &Context, params: DrawParams);
    fn stage_transfers(&mut self, allocator: &mut Allocator);
    fn update_state(&mut self, state: &Self::State);

    /// Convenience function to Generally Do the right thing. Ensure that:
    ///
    /// - [`draw command buffer`] is in the RECORDING state
    /// - no other rendering is in progress
    fn begin_rendering(
        &self,
        draw_command_buffer: vk::CommandBuffer,
        context: &Context,
        pipeline: &Pipeline,
    ) {
        let device = &context.device;

        unsafe {
            // Bind the pipeline
            device.cmd_bind_pipeline(
                draw_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
        }
    }
}
