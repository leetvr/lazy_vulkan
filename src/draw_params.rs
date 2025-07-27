use ash::vk;

use crate::{depth_buffer::DepthBuffer, swapchain::Drawable};

#[derive(Clone, Copy)]
pub struct DrawParams {
    pub draw_command_buffer: vk::CommandBuffer,
    #[allow(unused)]
    pub drawable: Drawable,
    #[allow(unused)]
    pub depth_buffer: DepthBuffer,
}

impl DrawParams {
    pub fn new(
        draw_command_buffer: vk::CommandBuffer,
        drawable: Drawable,
        depth_buffer: DepthBuffer,
    ) -> Self {
        Self {
            draw_command_buffer,
            drawable,
            depth_buffer,
        }
    }
}
