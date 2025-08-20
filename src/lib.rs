pub use crate::swapchain::Drawable;
pub use allocator::{Allocator, BufferAllocation, SlabUpload, TransferToken};
pub use ash;
use ash::vk;
pub use context::Context;
pub use core::Core;
pub use draw_params::DrawParams;
pub use image_manager::{Image, ImageManager};
pub use pipeline::Pipeline;
pub use renderer::Renderer;
use std::sync::Arc;
pub use sub_renderer::{StateFamily, SubRenderer};
use swapchain::Swapchain;

mod allocator;
mod context;
mod core;
mod depth_buffer;
mod descriptors;
mod draw_params;
mod headless_swapchain;
mod image_manager;
mod pipeline;
mod renderer;
mod sub_renderer;
mod swapchain;

pub struct LazyVulkan<SF: StateFamily> {
    pub core: Arc<Core>,
    pub context: Arc<Context>,
    pub renderer: Renderer<SF>,
}

impl<SF: StateFamily> LazyVulkan<SF> {
    pub fn from_window(window: &winit::window::Window) -> Self {
        let core = Arc::new(Core::from_window(window));
        let context = Arc::new(Context::new_from_window(&core));
        let swapchain = Swapchain::new(&context.device, &core, window, vk::SwapchainKHR::null());
        let renderer = Renderer::from_wsi(context.clone(), swapchain);

        LazyVulkan {
            core,
            context,
            renderer,
        }
    }

    pub fn headless(
        core: Arc<Core>,
        context: Arc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Self {
        let renderer = Renderer::headless(context.clone(), extent, format);

        LazyVulkan {
            core,
            context,
            renderer,
        }
    }

    pub fn draw<'s>(&mut self, state: &SF::For<'s>) {
        let drawable = self.renderer.get_drawable();
        self.renderer.begin_command_buffer();
        self.renderer.draw(state, &drawable);
    }

    pub fn get_drawable(&mut self) -> Drawable {
        let drawable = self.renderer.get_drawable();
        drawable
    }

    pub fn draw_to_drawable<'s>(&mut self, state: &SF::For<'s>, drawable: &Drawable) {
        self.renderer.draw(state, &drawable);
    }

    pub fn add_sub_renderer(
        &mut self,
        sub_renderer: Box<dyn for<'s> SubRenderer<'s, State = SF::For<'s>>>,
    ) {
        self.renderer.sub_renderers.push(sub_renderer);
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
