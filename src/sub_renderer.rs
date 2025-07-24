use ash::vk;

use crate::{allocator::Allocator, context::Context, draw_params::DrawParams, pipeline::Pipeline};

pub trait SubRenderer<'a, T> {
    fn draw(&mut self, params: DrawParams<'a, T>);
    fn stage_transfers(&mut self, allocator: &mut Allocator);

    /// Convenience function to Generally Do the right thing. Ensure that:
    ///
    /// - the command buffer is in the RECORDING state
    /// - no other rendering is in progress
    fn begin_rendering(&self, context: &Context, pipeline: &Pipeline) {
        let device = &context.device;
        let command_buffer = context.draw_command_buffer;

        unsafe {
            // Bind the pipeline
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
        }
    }
}
