pub use allocator::{Allocator, BufferAllocation, TransferToken};
pub use ash;
pub use context::Context;
pub use draw_params::DrawParams;
pub use image_manager::{Image, ImageManager};
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
mod descriptors;
mod draw_params;
mod image_manager;
mod pipeline;
mod renderer;
mod sub_renderer;
mod swapchain;

pub struct LazyVulkan {
    #[allow(unused)]
    core: Core,
    #[allow(unused)]
    pub context: Arc<Context>,
    pub renderer: Renderer,
}

impl LazyVulkan {
    pub fn from_window(window: &winit::window::Window) -> Self {
        let core = Core::from_window(window);
        let context = Arc::new(Context::new(&core));
        let swapchain = Swapchain::new(&context.device, &core, window, vk::SwapchainKHR::null());
        let renderer = Renderer::from_swapchain(context.clone(), swapchain);

        LazyVulkan {
            core,
            context,
            renderer,
        }
    }

    pub fn headless() -> Self {
        let core = Core::headless();
        let context = Arc::new(Context::new(&core));
        let renderer = Renderer::headless(context.clone());

        LazyVulkan {
            core,
            context,
            renderer,
        }
    }

    pub fn draw<S>(&mut self, state: &S, sub_renderers: &mut [Box<dyn SubRenderer<State = S>>]) {
        self.renderer.draw(state, sub_renderers);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(vk::Extent2D { width, height });
    }
}

pub const FULL_IMAGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: vk::REMAINING_MIP_LEVELS,
    base_array_layer: 0,
    layer_count: vk::REMAINING_ARRAY_LAYERS,
};
