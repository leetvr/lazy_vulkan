pub use crate::swapchain::Drawable;
pub use allocator::{Allocator, BufferAllocation, SlabUpload, TransferToken};
pub use ash::{self, vk};
pub use context::Context;
pub use core::Core;
pub use draw_params::DrawParams;
pub use headless_swapchain::HeadlessSwapchainImage;
pub use image_manager::{Image, ImageManager};
pub use pipeline::{load_module, Pipeline};
pub use render_plan::{RenderAttachment, RenderPlan};
pub use renderer::{BlendMode, PipelineOptions, Renderer};
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
mod render_plan;
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
        self.begin_commands();
        self.renderer.stage_and_execute_transfers(state);
        self.renderer.draw(state, &drawable);
        self.renderer.submit_and_present(drawable);
    }

    pub fn draw_render_plan<'s>(&mut self, state: &SF::For<'s>, plan: RenderPlan) {
        let drawable = self.renderer.get_drawable();
        self.begin_commands();
        self.renderer.stage_and_execute_transfers(state);
        self.renderer.draw_render_plan(state, plan, &drawable);
        self.renderer.submit_and_present(drawable);
    }

    pub fn begin_commands(&mut self) {
        self.renderer.begin_command_buffer();

        // If any transfers were staged on the first frame, we don't want to obliterate them.
        if self.renderer.frame != 0 {
            self.renderer.allocator.transfers_complete();
        }
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
        self.renderer
            .sub_renderers
            .insert(sub_renderer.label().to_string(), sub_renderer);
    }

    pub fn submit_and_present(&mut self, drawable: Drawable) {
        self.renderer.submit_and_present(drawable);
    }

    pub fn resize(&mut self, new_extent: impl IntoExtent) {
        self.renderer.resize(new_extent.into_extent());
    }
}

pub const FULL_IMAGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: vk::REMAINING_MIP_LEVELS,
    base_array_layer: 0,
    layer_count: vk::REMAINING_ARRAY_LAYERS,
};

pub trait IntoExtent {
    fn into_extent(self) -> vk::Extent2D;
}

impl IntoExtent for vk::Extent2D {
    fn into_extent(self) -> vk::Extent2D {
        self
    }
}

impl IntoExtent for winit::dpi::PhysicalSize<u32> {
    fn into_extent(self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}
