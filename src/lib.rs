pub use allocator::Allocator;
pub use ash;
pub use context::Context;
pub use draw_params::DrawParams;
pub use pipeline::Pipeline;
pub use renderer::Renderer;
pub use sub_renderer::SubRenderer;

use core::Core;
use std::sync::Arc;

use ash::vk;
use swapchain::Swapchain;

mod allocator;
mod context;
mod core;
mod depth_buffer;
mod draw_params;
mod pipeline;
mod renderer;
mod sub_renderer;
mod swapchain;

pub struct LazyVulkan<'a, T> {
    #[allow(unused)]
    core: Core,
    #[allow(unused)]
    context: Arc<Context>,
    renderer: Renderer<'a, T>,
    pub window: winit::window::Window,
}

impl<'a, T> LazyVulkan<'a, T> {
    pub fn new<F>(window: winit::window::Window, create_subrenderers: F) -> Self
    where
        F: Fn(&Renderer<'a, T>) -> Vec<Box<dyn SubRenderer<'a, T>>>,
    {
        let core = Core::new(&window);
        let context = Arc::new(Context::new(&core));
        let swapchain = Swapchain::new(&context.device, &core, &window, vk::SwapchainKHR::null());
        let mut renderer = Renderer::new(context.clone(), swapchain);
        renderer.sub_renderers = create_subrenderers(&renderer);

        LazyVulkan {
            core,
            context,
            renderer,
            window,
        }
    }

    pub fn draw(&mut self, state: &'a T) {
        self.renderer.draw(state);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.swapchain.extent = vk::Extent2D { width, height };
        self.renderer.swapchain.needs_update = true;
    }
}

const FULL_IMAGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: vk::REMAINING_MIP_LEVELS,
    base_array_layer: 0,
    layer_count: vk::REMAINING_ARRAY_LAYERS,
};
