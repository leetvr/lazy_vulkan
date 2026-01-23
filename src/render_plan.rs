use std::collections::HashMap;

use ash::vk;

pub struct RenderPlan {
    pub attachments: HashMap<String, RenderAttachment>,
    pub target_to_composite: String,
    pub passes: Vec<RenderPass>,
}

pub struct RenderPass {
    pub subrenderer: String,
    pub stage: RenderStage,
    pub colour_attachment: Option<String>,
    pub depth_attachment: Option<String>,
    pub sample_attachments: Vec<String>,
}

pub enum RenderStage {
    Shadow,
    Opaque,
    Layer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentState {
    Output,
    Sampled,
    Undefined,
}

pub struct RenderAttachment {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
}
