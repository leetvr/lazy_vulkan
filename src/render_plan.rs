use ash::vk;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPlan {
    pub target_to_composite: String,
    pub compositor_subrenderer: String,
    pub passes: Vec<RenderPass>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderPass {
    pub name: String,
    pub subrenderer: String,
    pub stage: RenderStage,
    pub colour_attachment: Option<String>,
    pub depth_attachment: Option<String>,
    pub sample_attachments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderStage {
    Shadow,
    Opaque,
    Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentState {
    ColourOutput,
    DepthOutput,
    Sampled,
    Undefined,
    Swapchain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderAttachment {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub id: u32,
}
