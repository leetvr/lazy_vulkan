use crate::{depth_buffer::DepthBuffer, swapchain::Drawable};

#[derive(Clone, Copy)]
pub struct DrawParams<'a, T> {
    #[allow(unused)]
    pub drawable: Drawable,
    #[allow(unused)]
    pub depth_buffer: DepthBuffer,
    pub state: &'a T,
}

impl<'a, T> DrawParams<'a, T> {
    pub fn new(drawable: Drawable, depth_buffer: DepthBuffer, state: &'a T) -> Self {
        Self {
            drawable,
            depth_buffer,
            state,
        }
    }
}
