use ash::vk;

use crate::{allocator::Allocator, context::Context, draw_params::DrawParams, pipeline::Pipeline};

pub trait SubRenderer {
    type State;

    /// Used by debug-utils to provide additional information about operations on this subrenderer
    fn label(&self) -> &'static str;

    /// Most sub-renderers, most of the time, will want to override this function as it provides
    /// the most convenient way to "just render some triangles":
    ///
    /// - The command buffer will be in the recording state
    /// - A dynamic render pass will be in-progress
    fn draw_opaque(&mut self, _: &Self::State, _: &Context, _: DrawParams) {}

    /// Override this method if you'd like to perform any transfer operations BEFORE any drawing
    /// begins.
    fn stage_transfers(&mut self, _: &Self::State, _: &mut Allocator) {}

    /// Override this method if you'd like to perform any drawing on the final colour image before
    /// it's presented. Useful for eg. GUI applications or debug overlays.
    ///
    /// ## NOTE
    /// Unlike [`Self::draw_opaque`], *NO* dynamic render-pass will be in progress.
    fn draw_layer(&mut self, _: &Self::State, _: &Context, _: DrawParams) {}

    /// Convenience function to Generally Do the right thing. Ensure that:
    ///
    /// - draw command buffer is in the RECORDING state
    /// - no other rendering is in progress
    fn begin_rendering(&self, context: &Context, pipeline: &Pipeline) {
        let device = &context.device;
        let draw_command_buffer = context.draw_command_buffer;

        unsafe {
            // Bind the pipeline
            device.cmd_bind_pipeline(
                draw_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
            device.cmd_bind_descriptor_sets(
                draw_command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[pipeline.descriptor_set],
                &[],
            );
        }
    }
}
