pub use allocator::{Allocator, BufferAllocation, TransferToken};
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

pub struct LazyVulkan {
    #[allow(unused)]
    core: Core,
    #[allow(unused)]
    context: Arc<Context>,
    pub renderer: Renderer,
    pub window: winit::window::Window,
}

impl LazyVulkan {
    pub fn from_window(window: winit::window::Window) -> Self {
        let core = Core::from_window(&window);
        let context = Arc::new(Context::new(&core));
        let swapchain = Swapchain::new(&context.device, &core, &window, vk::SwapchainKHR::null());
        let renderer = Renderer::new(&core, context.clone(), swapchain);

        LazyVulkan {
            core,
            context,
            renderer,
            window,
        }
    }

    pub fn draw<S>(&mut self, state: &S, sub_renderers: &mut [Box<dyn SubRenderer<State = S>>]) {
        for sub_renderer in &mut *sub_renderers {
            sub_renderer.update_state(state);
        }
        self.renderer.draw(sub_renderers);
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
