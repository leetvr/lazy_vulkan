pub use crate::swapchain::Drawable;
pub use allocator::{
    AllocationError, Allocator, BufferAllocation, ReadbackBuffer, SlabUpload, TransferToken,
};
pub use ash::{self, vk};
pub use context::Context;
pub use core::Core;
pub use descriptors::Descriptors;
pub use draw_params::DrawParams;
pub use headless_swapchain::HeadlessSwapchainImage;
pub use image_manager::{Image, ImageManager};
pub use multisample_color_attachment::MultisampleColorAttachment;
pub use pipeline::{load_module, BlendMode, Pipeline, PipelineOptions};
pub use render_plan::{RenderAttachment, RenderPass, RenderPlan, RenderStage};
pub use renderer::Renderer;
use std::sync::Arc;
pub use sub_renderer::{AttachmentInfo, LayerInfo, StateFamily, SubRenderer};
#[cfg(feature = "window")]
use swapchain::Swapchain;
pub use timestamp_queries::TimestampQueryResults;

mod allocator;
mod context;
mod core;
mod depth_buffer;
mod descriptors;
mod draw_params;
pub mod geometry;
mod headless_swapchain;
mod image_manager;
mod multisample_color_attachment;
mod pipeline;
mod render_plan;
mod renderer;
mod sub_renderer;
mod swapchain;
mod timestamp_queries;

pub struct LazyVulkan<SF: StateFamily> {
    pub core: Arc<Core>,
    pub context: Arc<Context>,
    pub renderer: Renderer<SF>,
}

impl<SF: StateFamily> LazyVulkan<SF> {
    #[cfg(feature = "window")]
    pub fn from_window(
        window: &winit::window::Window,
        pool_bytes: u64,
        staging_bytes: u64,
    ) -> Result<Self, AllocationError> {
        let core = Arc::new(Core::from_window(window));
        let context = Arc::new(Context::new_from_window(&core));
        let allocator = Allocator::new(context.clone(), pool_bytes, staging_bytes)?;
        let swapchain = Swapchain::new(&context.device, &core, window, vk::SwapchainKHR::null());
        let renderer = Renderer::from_wsi(context.clone(), swapchain, allocator);

        Ok(LazyVulkan {
            core,
            context,
            renderer,
        })
    }

    pub fn headless(
        core: Arc<Core>,
        context: Arc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
        pool_bytes: u64,
        staging_bytes: u64,
    ) -> Result<Self, AllocationError> {
        let allocator = Allocator::new(context.clone(), pool_bytes, staging_bytes)?;
        let renderer = Renderer::headless(context.clone(), extent, format, allocator);

        Ok(LazyVulkan {
            core,
            context,
            renderer,
        })
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
    }

    pub fn wait_for_previous_frame(&self) {
        self.renderer.wait_for_previous_frame();
    }

    /// Resolves the previous submitted batch and resets enough timestamp queries for this frame.
    /// Call after `begin_commands` and before writing timestamps.
    pub fn prepare_timestamp_queries(&mut self, query_count: u32) -> Option<TimestampQueryResults> {
        self.renderer.prepare_timestamp_queries(query_count)
    }

    pub fn timestamp_queries_supported(&self) -> bool {
        self.context.timestamp_valid_bits > 0
    }

    pub fn write_timestamp(&self, query: u32, stage: vk::PipelineStageFlags2) {
        self.renderer.write_timestamp(query, stage);
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

    pub fn create_render_attachment(
        &mut self,
        attachment_info: RenderAttachmentInfo,
    ) -> Result<(), AllocationError> {
        assert!(
            attachment_info.samples.as_raw().is_power_of_two(),
            "expected one sample count"
        );
        let supported = unsafe {
            self.core
                .instance
                .get_physical_device_image_format_properties(
                    self.core.physical_device,
                    attachment_info.format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    attachment_info.usage | vk::ImageUsageFlags::TRANSFER_DST,
                    vk::ImageCreateFlags::empty(),
                )
        }
        .expect("unsupported attachment format/usage");
        assert!(
            supported.sample_counts.contains(attachment_info.samples),
            "unsupported attachment sample count"
        );
        let image = self.renderer.image_manager.create_attachment(
            &attachment_info.name,
            &mut self.renderer.allocator,
            attachment_info.format,
            attachment_info.extent,
            attachment_info.usage,
            attachment_info.samples,
        )?;

        self.renderer.render_attachments.insert(
            attachment_info.name,
            RenderAttachment {
                handle: image.handle,
                view: image.view,
                extent: image.extent,
                id: image.id,
                format: attachment_info.format,
                usage: attachment_info.usage,
                samples: attachment_info.samples,
            },
        );
        Ok(())
    }

    pub fn submit_and_present(&mut self, drawable: Drawable) {
        self.renderer.submit_and_present(drawable);
    }

    pub fn submit_commands(&mut self) {
        self.renderer.submit_commands();
    }

    pub fn resize(&mut self, new_extent: impl IntoExtent) {
        self.renderer.resize(new_extent.into_extent());
    }

    pub fn resize_render_attachment(
        &mut self,
        name: &str,
        new_extent: vk::Extent2D,
    ) -> Result<(), AllocationError> {
        let attachment_info = *self
            .renderer
            .render_attachments
            .get(name)
            .expect(&format!("Render attachment {name} not found"));

        // If the image already has the correct extent, skip resizing
        if attachment_info.extent == new_extent {
            return Ok(());
        }

        self.wait_for_previous_frame();
        let image = self.renderer.image_manager.create_attachment(
            name,
            &mut self.renderer.allocator,
            attachment_info.format,
            new_extent,
            attachment_info.usage,
            attachment_info.samples,
        )?;
        unsafe {
            self.renderer
                .image_manager
                .destroy_attachment(attachment_info.handle, &mut self.renderer.allocator);
        }

        self.renderer.render_attachments.insert(
            name.to_string(),
            RenderAttachment {
                handle: image.handle,
                view: image.view,
                extent: image.extent,
                id: image.id,
                format: attachment_info.format,
                usage: attachment_info.usage,
                samples: attachment_info.samples,
            },
        );
        Ok(())
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

#[cfg(feature = "window")]
impl IntoExtent for winit::dpi::PhysicalSize<u32> {
    fn into_extent(self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

pub struct RenderAttachmentInfo {
    pub name: String,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub samples: vk::SampleCountFlags,
}
