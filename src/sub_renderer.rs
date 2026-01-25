use ash::vk;

use crate::{allocator::Allocator, context::Context, pipeline::Pipeline, ImageManager};

/// A family of state types parameterized by a borrow lifetime.
pub trait StateFamily {
    type For<'s>;
}

impl StateFamily for () {
    type For<'s> = ();
}

pub trait SubRenderer<'s> {
    type State;

    /// Used by debug-utils to provide additional information about operations on this subrenderer
    fn label(&self) -> &'static str;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// STAGE CALLBACKS
    ///
    /// These are called IN THE ORDER THEY ARE DECLARED BELOW, in the order you added each
    /// sub-renderer to lazy Vulkan.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /// Override this method if you'd like to perform any transfer operations BEFORE any drawing
    /// begins.
    #[allow(unused)]
    fn stage_transfers(
        &mut self,
        state: &Self::State,
        allocator: &mut Allocator,
        image_manager: &mut ImageManager,
    ) {
    }

    /// Useful for
    ///
    /// - The command buffer will be in the recording state
    #[allow(unused)]
    fn draw_shadow(&mut self, state: &Self::State, context: &Context) {}

    /// Most sub-renderers, most of the time, will want to override this function as it provides
    /// the most convenient way to "just render some triangles":
    ///
    /// - The command buffer will be in the recording state
    /// - A dynamic render pass will be in-progress
    #[allow(unused)]
    fn draw_opaque(&mut self, state: &Self::State, context: &Context) {}

    /// Override this method if you'd like to perform any drawing on the final colour image before
    /// it's presented. Useful for eg. GUI applications or debug overlays.
    ///
    /// ## NOTE
    /// Unlike [`Self::draw_opaque`], *NO* dynamic render-pass will be in progress.
    #[allow(unused)]
    fn draw_layer(&mut self, state: &Self::State, context: &Context, layer_info: LayerInfo) {}

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /// HELPERS
    ///////////////////////////////////////////////////////////////////////////////////////////////

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
            // Bind the descriptor sets
            pipeline.bind_descriptor_sets();
        }
    }
}

pub struct LayerInfo {
    pub colour_attachment: Option<AttachmentInfo>,
    pub depth_attachment: Option<AttachmentInfo>,
}

pub struct AttachmentInfo {
    pub extent: vk::Extent2D,
    pub view: vk::ImageView,
    pub handle: vk::Image,
}
